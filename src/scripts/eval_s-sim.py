#!/usr/bin/env python3
"""Speaker similarity: cosine(ref_emb, gen_emb) with one or two embedders.

Embedders:
  - **ecapa** (default): SpeechBrain ``speechbrain/spkrec-ecapa-voxceleb``
  - **wavlm**: UniSpeech-style frozen WavLM + finetuned ECAPA-TDNN head
    ([microsoft/UniSpeech speaker_verification](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification))

Modes:
  1) **Pairwise**: --ref_wav/--gen_wav, --pairs_csv, or --ref_dir/--gen_dir
  2) **LibriTTS test-clean VC** (--libritts_test_clean): cross-speaker ref, VC from ckpt

LibriTTS VC example (both metrics):
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_s-sim.py --libritts_test_clean \\
        --embedder both \\
        --wavlm_ckpt /path/to/WavLM_large_finetune.pth \\
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/mimi_semantic_new/step_latest.ckpt \\
        --out_csv /mnt/data/disk2/yejin/LiveVoice/ssim_mimi_semantic_new_test_clean.csv

Dependencies:
    pip install speechbrain>=1.1.0 "huggingface-hub>=0.23,<1.0"
    # WavLM-TDNN (--embedder wavlm|both):
    pip install s3prl --no-deps   # or UniSpeech README pin
    # Download finetuned head: WavLM large (No fix pre-train) from UniSpeech README table
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.data.libritts_dataset import LibriTTSDataset
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.model import HuBERTContentExtractor, LiveVoiceModel, build_codec
from livevoice.utils.checkpoint import infer_content_source_from_ckpt, load_model_weights_from_ckpt


def _load_full_mono_wav(path: str, target_sr: int) -> torch.Tensor:
    try:
        with sf.SoundFile(path) as f:
            audio_np = f.read(dtype="float32", always_2d=True)
            sr = int(f.samplerate)
        audio = torch.from_numpy(audio_np).float().mean(dim=1)
    except Exception:
        audio_np, sr = librosa.load(path, sr=None, mono=True)
        audio = torch.from_numpy(audio_np.astype("float32"))
        sr = int(sr)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio / (audio.abs().max() + 1e-8)


def _pick_cross_speaker_ref(
    speaker_utts: dict[str, list[tuple[str, str]]],
    content_spk: str,
    rng: random.Random,
) -> tuple[str, str]:
    """Ref = one arbitrary utterance from a speaker id != content_spk."""
    other = [s for s in speaker_utts if s != content_spk]
    if not other:
        raise RuntimeError(f"No other speaker for content speaker {content_spk}")
    ref_spk = rng.choice(other)
    ref_path, _ = rng.choice(speaker_utts[ref_spk])
    return ref_path, ref_spk


def _build_vc_config(args: argparse.Namespace) -> LiveVoiceConfig:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    codec = str(args.codec).lower()
    if codec != "mimi":
        raise ValueError(f"Only codec='mimi' is supported; got {codec!r}.")
    sample_rate = 24000
    return LiveVoiceConfig(
        device=device,
        codec=codec,
        sample_rate=sample_rate,
        hidden_dim=int(args.hidden_dim),
        num_decoder_layers=int(args.num_decoder_layers),
        ffn_dim=4 * int(args.hidden_dim),
        n_codebooks_predict=int(args.n_codebooks),
        content_source=str(args.content_source).lower(),
        features_dir=None,
        output_dir=args.output_dir,
    )


# ──────────────────────────────────────────────────────────────────────
#  Speaker embedders: SpeechBrain ECAPA + UniSpeech WavLM-TDNN
# ──────────────────────────────────────────────────────────────────────


def _try_encoder_classifier():
    try:
        from speechbrain.inference.classifiers import EncoderClassifier

        return EncoderClassifier
    except ImportError:
        pass
    try:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

        return EncoderClassifier
    except ImportError as e:
        raise ImportError(
            "Install SpeechBrain in the active env, e.g.\n"
            "  pip install 'speechbrain>=1.1.0' 'huggingface-hub>=0.23,<1.0'\n"
            f"Original error: {e}"
        ) from e


def _load_mono(path: str, target_sr: int, max_sec: float | None) -> torch.Tensor:
    """Load mono wav without torchaudio.load (2.9+ may require torchcodec)."""
    wav = _load_full_mono_wav(path, target_sr)
    if max_sec is not None and max_sec > 0:
        n = int(max_sec * target_sr)
        if wav.numel() > n:
            wav = wav[:n]
    return wav


def _waveform_for_embedder(wav: torch.Tensor, src_sr: int, target_sr: int) -> torch.Tensor:
    """Mono (T,) float, peak-normalized at embedder sample rate."""
    w = wav.detach().float().cpu().reshape(-1)
    if src_sr != target_sr:
        w = torchaudio.functional.resample(w.unsqueeze(0), src_sr, target_sr).squeeze(0)
    peak = w.abs().max().clamp(min=1e-8)
    return w / peak


class ECAPASpeakerEmbedder:
    """SpeechBrain ECAPA-TDNN (VoxCeleb)."""

    def __init__(self, source: str, device: str, savedir: str | None = None):
        EncoderClassifier = _try_encoder_classifier()
        self.device = device
        self.savedir = savedir or os.path.join(
            tempfile.gettempdir(), "speechbrain_spkrec_cache", source.replace("/", "__")
        )
        os.makedirs(self.savedir, exist_ok=True)
        self.model = EncoderClassifier.from_hparams(
            source=source,
            savedir=self.savedir,
            run_opts={"device": device},
        )
        self.sample_rate = int(getattr(self.model.hparams, "sample_rate", 16000))

    @torch.no_grad()
    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        wav = waveform.unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(wav)
        if emb.dim() > 1:
            emb = emb.squeeze(0)
        return emb.reshape(-1).float()


def _parse_embedder(s: str) -> set[str]:
    s = s.lower().strip()
    if s == "both":
        return {"ecapa", "wavlm"}
    if s in ("ecapa", "wavlm"):
        return {s}
    raise ValueError(f"--embedder must be ecapa, wavlm, or both; got {s!r}")


def _build_embedders(args: argparse.Namespace) -> dict:
    kinds = _parse_embedder(args.embedder)
    out: dict = {}
    if "ecapa" in kinds:
        print(f"[s-sim] Loading ECAPA embedder {args.speechbrain_source!r} ...")
        out["ecapa"] = ECAPASpeakerEmbedder(
            source=args.speechbrain_source,
            device=args.device,
            savedir=args.speechbrain_savedir,
        )
    if "wavlm" in kinds:
        if not args.wavlm_ckpt:
            raise SystemExit("--embedder wavlm|both requires --wavlm_ckpt (UniSpeech finetuned .pth)")
        from livevoice.evaluation.unispeech_sv import UniSpeechWavLMTDNNEmbedder

        print(
            f"[s-sim] Loading UniSpeech WavLM-TDNN ({args.wavlm_variant}) "
            f"from {args.wavlm_ckpt!r} ..."
        )
        out["wavlm"] = UniSpeechWavLMTDNNEmbedder(
            checkpoint=args.wavlm_ckpt,
            device=args.device,
            variant=args.wavlm_variant,
        )
    return out


def _embed_pair(
    embedders: dict,
    ref_w: torch.Tensor,
    gen_w: torch.Tensor,
    ref_sr: int,
    gen_sr: int,
) -> dict[str, float]:
    """Return cosine per embedder key; values may be nan on failure."""
    sims: dict[str, float] = {}
    for name, emb in embedders.items():
        sr = emb.sample_rate
        rw = _waveform_for_embedder(ref_w, ref_sr, sr)
        gw = _waveform_for_embedder(gen_w, gen_sr, sr)
        er = emb.embed(rw)
        eg = emb.embed(gw)
        sims[name] = _cosine(er, eg)
    return sims


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def _load_pairs_csv(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("empty CSV or no header")
        fn = {h.lower().strip(): h for h in r.fieldnames}
        rk = fn.get("ref_wav") or fn.get("reference") or fn.get("ref")
        gk = fn.get("gen_wav") or fn.get("generated") or fn.get("gen")
        if not rk or not gk:
            raise ValueError(
                f"CSV needs columns ref_wav,gen_wav (or reference,generated). Got: {r.fieldnames}"
            )
        for row in r:
            ra = (row.get(rk) or "").strip()
            gb = (row.get(gk) or "").strip()
            if ra and gb:
                rows.append((ra, gb))
    return rows


def _pairs_from_dirs(ref_dir: Path, gen_dir: Path) -> list[tuple[str, str]]:
    refs = {p.name: p for p in ref_dir.glob("*.wav")}
    gens = {p.name: p for p in gen_dir.glob("*.wav")}
    names = sorted(set(refs) & set(gens))
    if not names:
        raise FileNotFoundError(f"No matching *.wav names under {ref_dir} and {gen_dir}")
    return [(str(refs[n]), str(gens[n])) for n in names]


def _done_key(row: dict) -> str:
    c = (row.get("content_wav") or row.get("wav_path") or "").strip()
    r = (row.get("ref_wav") or row.get("ref_path") or "").strip()
    if c and r:
        return f"{c}\t{r}"
    return c or r


def _load_done(csv_path: str) -> set[str]:
    if not os.path.isfile(csv_path):
        return set()
    out: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = _done_key(row)
            if k:
                out.add(k)
    return out


# ──────────────────────────────────────────────────────────────────────
#  LibriTTS test-clean: cross-speaker VC + speaker similarity
# ──────────────────────────────────────────────────────────────────────


def run_libritts_test_clean(args: argparse.Namespace) -> None:
    if not args.ckpt:
        raise SystemExit("--libritts_test_clean requires --ckpt")

    content_source = str(args.content_source).lower()
    if content_source == "auto":
        inferred = infer_content_source_from_ckpt(args.ckpt)
        if inferred is None:
            raise SystemExit(
                "Could not infer content_source from ckpt. "
                "Pass --content_source hubert or mimi_semantic."
            )
        content_source = inferred
        print(f"[s-sim] content_source=auto → {content_source!r}")
    args.content_source = content_source

    cfg_model = _build_vc_config(args)
    if cfg_model.content_source == "mimi_semantic" and cfg_model.codec != "mimi":
        raise SystemExit("content_source=mimi_semantic requires --codec mimi")

    cfg_ds = LiveVoiceConfig(
        libritts_path=args.libritts_path,
        libritts_val_splits=(args.split_dir,),
        sample_rate=cfg_model.sample_rate,
        max_windows=args.max_items,
        seed=int(args.seed),
        pairing="same_speaker",
        audio_duration=4.0,
        features_dir=None,
    )
    ds = LibriTTSDataset(cfg_ds, split="val")
    speaker_utts = ds.speaker_utts
    rng = random.Random(int(args.ref_seed))

    print(
        f"[s-sim] mode=libritts_cross_speaker_vc  split={args.split_dir}  "
        f"utterances={len(ds.items)}  speakers={len(speaker_utts)}"
    )
    print(f"[s-sim] ckpt={args.ckpt}  codec={cfg_model.codec}  content_source={cfg_model.content_source}")

    target_sr = int(cfg_model.sample_rate)
    dev = torch.device(cfg_model.device if torch.cuda.is_available() and not args.cpu else "cpu")

    print("[s-sim] Building VC model...")
    codec_model = build_codec(cfg_model)
    if cfg_model.content_source == "hubert":
        content_extractor = HuBERTContentExtractor(cfg_model)
    else:
        content_extractor = None
    core = LiveVoiceModel(cfg_model, codec_model, content_extractor, prosody_extractor=None)
    missing, unexpected = load_model_weights_from_ckpt(
        core, args.ckpt, log_prefix="[s-sim]"
    )
    if missing:
        print(f"[s-sim] warn: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"[s-sim] warn: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    lit = LiveVoiceLightningModule(cfg_model, core)
    lit.eval()
    lit = lit.to(dev)

    embedders = _build_embedders(args)
    ecapa_sr = embedders["ecapa"].sample_rate if "ecapa" in embedders else 16000

    out_csv = args.out_csv
    out_parent = os.path.dirname(out_csv)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    done = _load_done(out_csv) if args.resume else set()
    csv_mode = "a" if args.resume else "w"
    write_header = csv_mode == "w" or not (
        os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0
    )

    fieldnames = [
        "content_wav",
        "ref_wav",
        "content_speaker",
        "ref_speaker",
        "utt_id",
        "cosine_similarity_ecapa",
        "cosine_similarity_wavlm",
        "cosine_similarity",
        "error",
    ]
    sims_ecapa: list[float] = []
    sims_wavlm: list[float] = []
    temp = float(args.temperature)

    with open(out_csv, csv_mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for content_wav, utt_id, content_spk in tqdm(ds.items, desc="s-sim VC"):
            try:
                ref_wav, ref_spk = _pick_cross_speaker_ref(speaker_utts, content_spk, rng)
            except Exception as e:
                writer.writerow(
                    {
                        "content_wav": content_wav,
                        "ref_wav": "",
                        "content_speaker": content_spk,
                        "ref_speaker": "",
                        "utt_id": utt_id,
                        "cosine_similarity_ecapa": "",
                        "cosine_similarity_wavlm": "",
                        "cosine_similarity": "",
                        "error": f"ref_pick: {e}"[:500],
                    }
                )
                fcsv.flush()
                continue

            row_key = _done_key({"content_wav": content_wav, "ref_wav": ref_wav})
            if row_key in done:
                continue

            err = ""
            sim_ecapa = float("nan")
            sim_wavlm = float("nan")
            try:
                ctn = _load_full_mono_wav(content_wav, target_sr).unsqueeze(0).to(dev)
                ref = _load_full_mono_wav(ref_wav, target_sr).unsqueeze(0).to(dev)
                with torch.no_grad():
                    codes = lit.model.generate(
                        reference_audio=ref,
                        content_audio=ctn,
                        temperature=temp,
                        top_p=float(args.top_p) if temp > 0 and args.top_p > 0 else None,
                        top_k=int(args.top_k) if temp > 0 and args.top_k > 0 else None,
                        cfg_scale=float(args.cfg_scale),
                    )
                    gen = lit.model.decode_to_audio(codes)

                ref_w = _load_mono(ref_wav, ecapa_sr, args.max_audio_sec)
                gen_w = gen.squeeze(0)
                if args.max_audio_sec is not None and args.max_audio_sec > 0:
                    n = int(args.max_audio_sec * target_sr)
                    if gen_w.numel() > n:
                        gen_w = gen_w[:n]

                scores = _embed_pair(embedders, ref_w, gen_w, ecapa_sr, target_sr)
                if "ecapa" in scores:
                    sim_ecapa = scores["ecapa"]
                    if sim_ecapa == sim_ecapa:
                        sims_ecapa.append(sim_ecapa)
                if "wavlm" in scores:
                    sim_wavlm = scores["wavlm"]
                    if sim_wavlm == sim_wavlm:
                        sims_wavlm.append(sim_wavlm)

                del ctn, ref, codes, gen
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                err = str(e)[:800]
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            writer.writerow(
                {
                    "content_wav": content_wav,
                    "ref_wav": ref_wav,
                    "content_speaker": content_spk,
                    "ref_speaker": ref_spk,
                    "utt_id": utt_id,
                    "cosine_similarity_ecapa": f"{sim_ecapa:.6f}" if sim_ecapa == sim_ecapa else "",
                    "cosine_similarity_wavlm": f"{sim_wavlm:.6f}" if sim_wavlm == sim_wavlm else "",
                    "cosine_similarity": f"{sim_ecapa:.6f}" if sim_ecapa == sim_ecapa else "",
                    "error": err,
                }
            )
            fcsv.flush()

    if sims_ecapa:
        print(f"[s-sim] mean cosine (ECAPA): {sum(sims_ecapa) / len(sims_ecapa):.4f} (n={len(sims_ecapa)})")
    if sims_wavlm:
        print(f"[s-sim] mean cosine (WavLM-TDNN): {sum(sims_wavlm) / len(sims_wavlm):.4f} (n={len(sims_wavlm)})")
    print(f"[s-sim] wrote: {out_csv}")


def run_pairwise(args: argparse.Namespace) -> None:
    pairs: list[tuple[str, str]] = []
    if args.ref_wav and args.gen_wav:
        pairs = [(args.ref_wav, args.gen_wav)]
    elif args.pairs_csv:
        pairs = _load_pairs_csv(Path(args.pairs_csv))
    elif args.ref_dir and args.gen_dir:
        pairs = _pairs_from_dirs(Path(args.ref_dir), Path(args.gen_dir))
    else:
        raise SystemExit(
            "Pairwise mode: provide (--ref_wav and --gen_wav) OR --pairs_csv OR "
            "(--ref_dir and --gen_dir), or use --libritts_test_clean."
        )

    for ra, gb in pairs:
        if not os.path.isfile(ra):
            raise FileNotFoundError(f"missing ref: {ra}")
        if not os.path.isfile(gb):
            raise FileNotFoundError(f"missing gen: {gb}")

    print(f"[s-sim] pairs={len(pairs)}  embedder={args.embedder!r}  device={args.device!r}")

    embedders = _build_embedders(args)
    sr = embedders["ecapa"].sample_rate if "ecapa" in embedders else 16000

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    done = _load_done(args.out_csv) if args.resume else set()
    mode = "a" if args.resume else "w"
    write_header = mode == "w" or not (
        os.path.isfile(args.out_csv) and os.path.getsize(args.out_csv) > 0
    )

    sims_ecapa: list[float] = []
    sims_wavlm: list[float] = []
    fieldnames = [
        "ref_wav",
        "gen_wav",
        "cosine_similarity_ecapa",
        "cosine_similarity_wavlm",
        "cosine_similarity",
        "error",
    ]

    with open(args.out_csv, mode, newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for ref_path, gen_path in pairs:
            if ref_path in done:
                continue
            err = ""
            sim_ecapa = float("nan")
            sim_wavlm = float("nan")
            try:
                ref_w = _load_mono(ref_path, sr, args.max_audio_sec)
                gen_w = _load_mono(gen_path, sr, args.max_audio_sec)
                scores = _embed_pair(embedders, ref_w, gen_w, sr, sr)
                if "ecapa" in scores:
                    sim_ecapa = scores["ecapa"]
                    if sim_ecapa == sim_ecapa:
                        sims_ecapa.append(sim_ecapa)
                if "wavlm" in scores:
                    sim_wavlm = scores["wavlm"]
                    if sim_wavlm == sim_wavlm:
                        sims_wavlm.append(sim_wavlm)
            except Exception as e:
                err = str(e)[:800]

            w.writerow(
                {
                    "ref_wav": ref_path,
                    "gen_wav": gen_path,
                    "cosine_similarity_ecapa": f"{sim_ecapa:.6f}" if sim_ecapa == sim_ecapa else "",
                    "cosine_similarity_wavlm": f"{sim_wavlm:.6f}" if sim_wavlm == sim_wavlm else "",
                    "cosine_similarity": f"{sim_ecapa:.6f}" if sim_ecapa == sim_ecapa else "",
                    "error": err,
                }
            )
            fcsv.flush()

    if sims_ecapa:
        print(f"[s-sim] mean cosine (ECAPA): {sum(sims_ecapa) / len(sims_ecapa):.4f} (n={len(sims_ecapa)})")
    if sims_wavlm:
        print(f"[s-sim] mean cosine (WavLM-TDNN): {sum(sims_wavlm) / len(sims_wavlm):.4f} (n={len(sims_wavlm)})")
    print(f"[s-sim] wrote: {args.out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Speaker cosine similarity (ECAPA / UniSpeech WavLM-TDNN) — pairwise or LibriTTS VC."
    )
    p.add_argument(
        "--libritts_test_clean",
        action="store_true",
        help="Run VC on LibriTTS test-clean with cross-speaker ref; write content/ref paths + sim.",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/checkpoints/mimi_semantic_new/step_latest.ckpt",
    )
    p.add_argument("--libritts_path", type=str, default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--split_dir", type=str, default="test-clean")
    p.add_argument("--output_dir", type=str, default="/mnt/data/disk2/yejin/LiveVoice")
    p.add_argument(
        "--out_csv",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/s-sim_mimi_semantic_new.csv",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--seed", type=int, default=42, help="LibriTTS dataset shuffle seed.")
    p.add_argument(
        "--ref_seed",
        type=int,
        default=12345,
        help="Seed for cross-speaker ref selection (fixed across runs if same).",
    )
    # VC model
    p.add_argument("--codec", type=str, default="mimi", choices=["mimi"])
    p.add_argument("--n_codebooks", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_decoder_layers", type=int, default=12)
    p.add_argument(
        "--content_source",
        type=str,
        default="auto",
        choices=["auto", "hubert", "mimi_semantic"],
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true", help="VC model on CPU (slow).")
    # Speaker embedder
    p.add_argument(
        "--embedder",
        type=str,
        default="both",
        choices=["ecapa", "wavlm", "both"],
        help="ecapa=SpeechBrain ECAPA; wavlm=UniSpeech WavLM+TDNN; both=record both columns.",
    )
    p.add_argument(
        "--speechbrain_source",
        type=str,
        default="speechbrain/spkrec-ecapa-voxceleb",
    )
    p.add_argument(
        "--wavlm_ckpt",
        type=str,
        default=None,
        help="UniSpeech finetuned checkpoint (.pth with 'model' key). Required for wavlm|both.",
    )
    p.add_argument(
        "--wavlm_variant",
        type=str,
        default="wavlm_large",
        choices=["wavlm_large", "wavlm_base_plus"],
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--speechbrain_savedir", type=str, default=None)
    p.add_argument("--max_audio_sec", type=float, default=None)
    # Pairwise inputs
    p.add_argument("--ref_wav", type=str, default=None)
    p.add_argument("--gen_wav", type=str, default=None)
    p.add_argument("--pairs_csv", type=str, default=None)
    p.add_argument("--ref_dir", type=str, default=None)
    p.add_argument("--gen_dir", type=str, default=None)

    args = p.parse_args()

    if args.libritts_test_clean:
        run_libritts_test_clean(args)
    else:
        run_pairwise(args)


if __name__ == "__main__":
    main()
