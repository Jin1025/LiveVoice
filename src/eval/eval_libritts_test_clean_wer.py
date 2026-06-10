"""Evaluate WER on LibriTTS test-clean with a LiveVoice checkpoint (full utterance).

**Ground truth text** is always the **content** wav's ``*.normalized.txt`` (what should be spoken).
**Whisper** runs on the **generated** VC output. That part is correct.

**Reference *audio*** (timbre) by default comes from ``LibriTTSDataset`` **same-speaker** val rule:
another utterance from the *same* LibriTTS speaker id — **not** arbitrary cross-speaker pairs like
``generate.py`` demos (e.g. ref 121 + content 5105). If you manually evaluate those pairs, pass
``--pairs_csv`` with ``content_wav,ref_wav`` so this script uses the **same** ref as your inference.

WER can look wrongly "high" if VC is sampled with temperature>0 (noisy output) or if Whisper
uses auto language / segment chaining on long clips. Defaults: temperature=0, language=en,
condition_on_previous_text=False.

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_libritts_test_clean_wer.py \\
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/mimi_semantic_new/step_latest.ckpt \\
        --out_csv /mnt/data/disk2/yejin/LiveVoice/wer_mimi_semantic_new.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.data.libritts_dataset import LibriTTSDataset
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.model import (
    HuBERTContentExtractor,
    StreamVoiceAnonContentEncoder,
    LiveVoiceModel,
    build_codec,
)
from livevoice.utils.checkpoint import (
    infer_content_source_from_ckpt,
    infer_speaker_conditioning_from_ckpt,
    infer_speaker_encoder_from_ckpt,
)


def _normalize_text_for_wer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _word_wer(hypothesis: str, reference: str) -> float:
    hyp_w = _normalize_text_for_wer(hypothesis).split()
    ref_w = _normalize_text_for_wer(reference).split()
    if not ref_w:
        return float("nan")
    dp = list(range(len(hyp_w) + 1))
    for i, rw in enumerate(ref_w, start=1):
        prev = dp[0]
        dp[0] = i
        for j, hw in enumerate(hyp_w, start=1):
            cur = dp[j]
            cost = 0 if rw == hw else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return float(dp[-1]) / float(len(ref_w))


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


def _read_libritts_normalized_text(wav_path: str) -> str | None:
    txt_path = Path(wav_path).with_suffix(".normalized.txt")
    if not txt_path.exists():
        return None
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        return text if text else None
    except Exception:
        return None


def _libritts_pick_ref_path(ds: LibriTTSDataset, content_wav_path: str, utt_id: str, spk: str) -> str:
    spk_utts = ds.speaker_utts[spk]
    pairing = str(getattr(ds, "pairing", "same_speaker"))
    split = str(getattr(ds, "split", "train"))
    if pairing == "reconstruct":
        return content_wav_path
    if split == "train":
        import random

        candidates = [(p, u) for p, u in spk_utts if p != content_wav_path]
        if not candidates:
            return content_wav_path
        return random.choice(candidates)[0]
    ref_path, _ = next(
        ((p, u) for p, u in spk_utts if p != content_wav_path),
        (content_wav_path, utt_id),
    )
    return ref_path


def _build_model_config(args: argparse.Namespace) -> LiveVoiceConfig:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    codec = str(args.codec).lower()
    sample_rate = 24000 if codec == "mimi" else 16000
    dac_model_type = getattr(args, "dac_model_type", "16khz")
    dac_sample_rate = 16000 if dac_model_type == "16khz" else sample_rate

    return LiveVoiceConfig(
        device=device,
        codec=codec,
        sample_rate=sample_rate,
        dac_model_type=dac_model_type,
        dac_sample_rate=dac_sample_rate,
        hidden_dim=int(args.hidden_dim),
        num_decoder_layers=int(args.num_decoder_layers),
        ffn_dim=4 * int(args.hidden_dim),
        n_codebooks_predict=int(args.n_codebooks),
        content_source=str(args.content_source).lower(),
        speaker_conditioning=str(args.speaker_conditioning).lower(),
        speaker_prefix_len=int(args.speaker_prefix_len),
        speaker_encoder_type=str(args.speaker_encoder_type).lower(),
        speechbrain_source=args.speechbrain_source,
        speechbrain_savedir=args.speechbrain_savedir,
        speechbrain_sample_rate=int(args.speechbrain_sample_rate),
        speechbrain_embedding_dim=int(args.speechbrain_embedding_dim),
        streamvoiceanon_repo=args.streamvoiceanon_repo,
        streamvoiceanon_encoder_config=args.streamvoiceanon_encoder_config,
        streamvoiceanon_encoder_ckpt=args.streamvoiceanon_encoder_ckpt,
        features_dir=None,
        output_dir=args.output_dir,
    )


def _libritts_spk_utt_from_content_path(content_wav: str) -> tuple[str, str]:
    p = Path(content_wav)
    return str(p.parts[-3]), p.stem


def _load_pairs_csv(path: Path) -> list[tuple[str, str, str, str]]:
    """Rows: content_wav, ref_wav (+ optional utt_id, speaker). Returns (content, ref, utt_id, spk)."""
    rows: list[tuple[str, str, str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("pairs CSV is empty or has no header")
        low = {h.lower().strip(): h for h in r.fieldnames}
        ck = low.get("content_wav") or low.get("content") or low.get("content_path")
        rk = low.get("ref_wav") or low.get("reference") or low.get("ref_path") or low.get("ref")
        if not ck or not rk:
            raise ValueError(
                f"pairs CSV needs content_wav and ref_wav columns (or content, ref). Got: {r.fieldnames}"
            )
        uk = low.get("utt_id")
        sk = low.get("speaker")
        for row in r:
            c = (row.get(ck) or "").strip()
            refp = (row.get(rk) or "").strip()
            if not c or not refp:
                continue
            utt = (row.get(uk) or "").strip() if uk else ""
            spk = (row.get(sk) or "").strip() if sk else ""
            if not utt or not spk:
                spk2, utt2 = _libritts_spk_utt_from_content_path(c)
                utt = utt or utt2
                spk = spk or spk2
            rows.append((c, refp, utt, spk))
    if not rows:
        raise ValueError(f"no valid rows in {path}")
    return rows


def _done_key(row: dict) -> str:
    """Resume key: content + ref if ref known, else content only (legacy CSV)."""
    w = (row.get("wav_path") or "").strip()
    r = (row.get("ref_path") or "").strip()
    if r:
        return f"{w}\t{r}"
    return w


def _prepare_whisper_audio(
    gen: torch.Tensor, target_sr: int, *, peak_normalize: bool = True
) -> np.ndarray:
    """Mono float32 16 kHz for Whisper (matches eval_whisper_wer_one peak norm)."""
    wavs = gen.detach().float().cpu()
    if wavs.dim() == 1:
        wavs = wavs.unsqueeze(0)
    if peak_normalize:
        peak = wavs.abs().max()
        if peak > 0:
            wavs = wavs / (peak + 1e-8)
    if target_sr != 16000:
        wavs = torchaudio.functional.resample(wavs, target_sr, 16000)
    return wavs.squeeze().numpy().astype(np.float32)


def _load_done_paths(csv_path: str) -> set[str]:
    if not os.path.isfile(csv_path):
        return set()
    done: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            k = _done_key(row)
            if k:
                done.add(k)
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/checkpoints/mimi_semantic_new/step_latest.ckpt",
    )
    p.add_argument("--libritts_path", type=str, default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--split_dir", type=str, default="test-clean", help="Subdir under libritts_path")
    p.add_argument("--codec", type=str, default="mimi", choices=["dac", "mimi"])
    p.add_argument("--n_codebooks", type=int, default=8)
    p.add_argument("--dac_model_type", type=str, default="16khz", choices=["16khz", "24khz", "44khz"])
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_decoder_layers", type=int, default=12)
    p.add_argument(
        "--content_source",
        type=str,
        default="auto",
        choices=["auto", "hubert", "mimi_semantic", "streamvoiceanon"],
        help="Content conditioning path. 'auto' reads ckpt state_dict "
        "(StreamVoiceAnon keys → streamvoiceanon, semantic_proj → mimi_semantic, "
        "content_extractor → hubert).",
    )
    p.add_argument(
        "--speaker_conditioning",
        type=str,
        default="auto",
        choices=["auto", "crossattn", "global_avg", "prefix"],
        help="Speaker path. 'auto' reads ckpt keys; prefix ckpts omit decoder cross-attn.",
    )
    p.add_argument("--speaker_prefix_len", type=int, default=8)
    p.add_argument(
        "--speaker_encoder_type",
        type=str,
        default="auto",
        choices=["auto", "codec", "speechbrain_ecapa"],
    )
    p.add_argument("--speechbrain_source", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument(
        "--speechbrain_savedir",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/pretrained_models/speechbrain__spkrec-ecapa-voxceleb",
    )
    p.add_argument("--speechbrain_sample_rate", type=int, default=16000)
    p.add_argument("--speechbrain_embedding_dim", type=int, default=192)
    p.add_argument("--streamvoiceanon_repo", type=str, default="/home/yejin/StreamVoiceAnon")
    p.add_argument(
        "--streamvoiceanon_encoder_config",
        type=str,
        default="/home/yejin/StreamVoiceAnon/configs/hydra_arcs/speech_tokenizers/causal-encoder-lfq-8192.yaml",
    )
    p.add_argument(
        "--streamvoiceanon_encoder_ckpt",
        type=str,
        default="/home/yejin/StreamVoiceAnon/ckpt/asr_s2s_bsq_8192_causal_down_whisper.pth",
    )
    # For WER evaluation use greedy decoding by default (temperature=0). Sampling (T>0)
    # makes VC output stochastic and often blows up Whisper WER without improving the metric.
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="VC generate temperature. Use 0 for deterministic argmax (recommended for WER).",
    )
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--whisper_model", type=str, default="medium")
    p.add_argument("--whisper_device", type=str, default="cuda")
    p.add_argument(
        "--no_whisper_peak_norm",
        action="store_true",
        help="Skip peak normalization before Whisper (eval_whisper_wer_one normalizes by default).",
    )
    p.add_argument("--fp16", action="store_true", help="Whisper fp16 on GPU (faster).")
    p.add_argument(
        "--whisper_language",
        type=str,
        default="en",
        help="Whisper decode language (LibriTTS is English). Empty string = auto-detect.",
    )
    p.add_argument(
        "--whisper_condition_on_previous_text",
        action="store_true",
        help="If set, use Whisper default chaining (can hurt long utterances). "
        "Default: False (recommended for full-book LibriTTS clips).",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU for the LiveVoice model")
    p.add_argument("--max_items", type=int, default=None, help="Debug: cap number of utterances")
    p.add_argument(
        "--out_csv",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/wer_mimi_semantic_new.csv",
    )
    p.add_argument("--resume", action="store_true", help="Skip wav_path rows already present in out_csv")
    p.add_argument("--output_dir", type=str, default="/mnt/data/disk2/yejin/LiveVoice")
    p.add_argument(
        "--pairs_csv",
        type=str,
        default=None,
        help="Optional CSV with content_wav,ref_wav (same columns as your manual VC). "
        "WER still uses content's .normalized.txt. Skips LibriTTS default same-speaker ref pick.",
    )
    args = p.parse_args()

    content_source = str(args.content_source).lower()
    if content_source == "auto":
        inferred = infer_content_source_from_ckpt(args.ckpt)
        if inferred is None:
            raise SystemExit(
                f"Could not infer content_source from {args.ckpt}. "
                "Pass --content_source hubert, mimi_semantic, or streamvoiceanon explicitly."
            )
        content_source = inferred
        print(f"[eval] content_source=auto → {content_source!r} (from checkpoint keys)")
    args.content_source = content_source

    speaker_conditioning = str(args.speaker_conditioning).lower()
    if speaker_conditioning == "auto":
        inferred_spk = infer_speaker_conditioning_from_ckpt(args.ckpt)
        if inferred_spk is None:
            inferred_spk = "prefix"
            print("[eval] speaker_conditioning=auto → 'prefix' (fallback; ckpt keys ambiguous)")
        else:
            print(f"[eval] speaker_conditioning=auto → {inferred_spk!r} (from checkpoint keys)")
        speaker_conditioning = inferred_spk
    args.speaker_conditioning = speaker_conditioning

    speaker_encoder_type = str(args.speaker_encoder_type).lower()
    if speaker_encoder_type == "auto":
        inferred_enc = infer_speaker_encoder_from_ckpt(args.ckpt)
        if inferred_enc is None:
            inferred_enc = "codec"
            print("[eval] speaker_encoder_type=auto → 'codec' (fallback; ckpt keys ambiguous)")
        else:
            print(f"[eval] speaker_encoder_type=auto → {inferred_enc!r} (from checkpoint keys)")
        speaker_encoder_type = inferred_enc
    args.speaker_encoder_type = speaker_encoder_type

    cfg_model = _build_model_config(args)
    if cfg_model.content_source == "mimi_semantic" and cfg_model.codec != "mimi":
        raise SystemExit("content_source=mimi_semantic requires --codec mimi")

    print(
        f"[eval] ckpt={args.ckpt}\n"
        f"       codec={cfg_model.codec} sr={cfg_model.sample_rate} "
        f"hidden_dim={cfg_model.hidden_dim} num_decoder_layers={cfg_model.num_decoder_layers} "
        f"n_codebooks_predict={cfg_model.n_codebooks_predict} content_source={cfg_model.content_source} "
        f"speaker_conditioning={cfg_model.speaker_conditioning} "
        f"speaker_encoder_type={cfg_model.speaker_encoder_type}"
    )

    ds: LibriTTSDataset | None = None
    if not args.pairs_csv:
        cfg_ds = LiveVoiceConfig(
            libritts_path=args.libritts_path,
            libritts_val_splits=(args.split_dir,),
            sample_rate=cfg_model.sample_rate,
            max_windows=args.max_items,
            seed=42,
            pairing="same_speaker",
            audio_duration=4.0,
            features_dir=None,
        )
        ds = LibriTTSDataset(cfg_ds, split="val")
    elif args.max_items is not None:
        print("[eval] note: --max_items applies to --pairs_csv row count when pairs_csv is set.")
    target_sr = int(cfg_model.sample_rate)

    import whisper

    w_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    print("[eval] Building codec + model...")
    codec_model = build_codec(cfg_model)
    sr_codec = int(getattr(codec_model, "sample_rate", target_sr))
    if sr_codec != target_sr:
        print(f"[eval] warn: codec sample_rate={sr_codec} vs config.sample_rate={target_sr}")

    if cfg_model.content_source == "hubert":
        content_extractor = HuBERTContentExtractor(cfg_model)
    elif cfg_model.content_source == "streamvoiceanon":
        content_extractor = StreamVoiceAnonContentEncoder(cfg_model)
    else:
        content_extractor = None
    core = LiveVoiceModel(cfg_model, codec_model, content_extractor, prosody_extractor=None)
    lit = LiveVoiceLightningModule.load_from_checkpoint(
        args.ckpt, config=cfg_model, model=core, strict=False
    )
    lit.eval()
    dev = torch.device(cfg_model.device if torch.cuda.is_available() and not args.cpu else "cpu")
    lit = lit.to(dev)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    done = _load_done_paths(args.out_csv) if args.resume else set()
    csv_mode = "a" if args.resume else "w"
    write_header = (csv_mode == "w") or not (os.path.isfile(args.out_csv) and os.path.getsize(args.out_csv) > 0)

    if args.pairs_csv:
        work: list[tuple[str, str, str, str]] = _load_pairs_csv(Path(args.pairs_csv))
        if args.max_items is not None:
            work = work[: int(args.max_items)]
        print(f"[eval] mode=pairs_csv  rows={len(work)}  file={args.pairs_csv}")
    else:
        assert ds is not None
        work = [
            (
                wav_path,
                _libritts_pick_ref_path(ds, wav_path, utt_id, spk),
                utt_id,
                spk,
            )
            for wav_path, utt_id, spk in ds.items
        ]
        print(f"[eval] mode=libritts_same_speaker_ref  utterances={len(work)}  (not cross-speaker)")

    wers: list[float] = []
    with open(args.out_csv, csv_mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "wav_path",
                "ref_path",
                "utt_id",
                "speaker",
                "wer",
                "reference",
                "hypothesis",
                "error",
            ],
        )
        if write_header:
            writer.writeheader()

        for wav_path, ref_path, utt_id, spk in tqdm(work, desc="test-clean WER"):
            row_key = _done_key({"wav_path": wav_path, "ref_path": ref_path})
            if row_key in done:
                continue
            err = ""
            wer_v = float("nan")
            ref_txt = ""
            hyp_txt = ""
            try:
                ref_txt = _read_libritts_normalized_text(wav_path) or ""
                if not ref_txt:
                    err = "no_normalized_txt"
                    writer.writerow(
                        {
                            "wav_path": wav_path,
                            "ref_path": ref_path,
                            "utt_id": utt_id,
                            "speaker": spk,
                            "wer": "",
                            "reference": "",
                            "hypothesis": "",
                            "error": err,
                        }
                    )
                    fcsv.flush()
                    continue

                ctn = _load_full_mono_wav(wav_path, target_sr).unsqueeze(0).to(dev)
                ref = _load_full_mono_wav(ref_path, target_sr).unsqueeze(0).to(dev)

                temp = float(args.temperature)
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

                arr = _prepare_whisper_audio(
                    gen,
                    target_sr,
                    peak_normalize=not args.no_whisper_peak_norm,
                )
                whisper_kw: dict = {
                    "fp16": bool(
                        args.fp16
                        and args.whisper_device == "cuda"
                        and torch.cuda.is_available()
                    ),
                    "condition_on_previous_text": bool(args.whisper_condition_on_previous_text),
                }
                if args.whisper_language:
                    whisper_kw["language"] = str(args.whisper_language)
                hyp_txt = w_model.transcribe(arr, **whisper_kw)["text"].strip()
                wer_v = _word_wer(hyp_txt, ref_txt)
                if wer_v == wer_v:
                    wers.append(wer_v)
            except Exception as e:
                err = str(e)[:500]

            writer.writerow(
                {
                    "wav_path": wav_path,
                    "ref_path": ref_path,
                    "utt_id": utt_id,
                    "speaker": spk,
                    "wer": f"{wer_v:.6f}" if wer_v == wer_v else "",
                    "reference": ref_txt,
                    "hypothesis": hyp_txt,
                    "error": err,
                }
            )
            fcsv.flush()

    if wers:
        print(f"[eval] mean WER over successful items: {sum(wers) / len(wers):.4f} (n={len(wers)})")
    print(f"[eval] wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
