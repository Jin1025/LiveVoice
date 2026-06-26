"""VPC eval_pre-style privacy (EER) + utility (ASR/WER) for speaker anonymization.

Matches Voice-Privacy-Challenge-2026 ``eval_pre.yaml``:

  - **ASV**: ``exp/asv_ssl`` (WavLM-Large + ECAPA-TDNN), cosine EER
  - **ASR**: ``exp/asr`` (wav2vec2-large-960h + CTC), SpeechBrain EncoderASR

Attack scenarios (frozen pretrained ASV — eval_pre only, no ASV retraining):
  - **oo**  enrollment = original, trial = original
  - **oa**  enrollment = original, trial = anonymized (VC)
  - **aa**  enrollment = anonymized, trial = anonymized

Higher EER ⇒ better anonymization. Lower WER on anonymized audio ⇒ better utility.
Semi-informed ASV (eval_post) is **not** implemented here.

Data layout (Kaldi / VPC format)
--------------------------------
  enroll_dir/wav.scp, utt2spk, …
  trial_dir/wav.scp, utt2spk, trials
  asr_dir/text for WER (libri_dev / libri_test)

Examples
--------
  # VPC eval_pre batch (libri_dev + libri_test, EER + ASR):
  CUDA_VISIBLE_DEVICES=0 python src/eval/eval_anon.py \\
      --datasets libri_dev,libri_test \\
      --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/prefix+jhcodec+hubert/step_latest.ckpt \\
      --anon_pool_dir /mnt/data/disk2/LibriTTS/train-other-500 \\
      --anon_out_dir /mnt/data/disk2/yejin/LiveVoice/anon \\
      --out_csv vpc_eval_pre.csv

  # Single trial set (EER only):
  CUDA_VISIBLE_DEVICES=0 python src/eval/eval_anon.py \\
      --enroll_dir /path/vpc/data/libri_test_enrolls \\
      --trial_dir  /path/vpc/data/libri_test_trials_mixed \\
      --ckpt ... --anon_pool_dir ... --scenarios oo,oa
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
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
    load_model_weights_from_ckpt,
)
from eval.vpc_pretrained import (
    VPCASREvaluator,
    VPCASVAttacker,
    materialize_anon_kaldi_dir,
    vpc_cosine_eer,
)

DEFAULT_VPC_ROOT = "/mnt/data/disk3/yejin/VPC"


# ──────────────────────────────────────────────────────────────────────
#  IO helpers
# ──────────────────────────────────────────────────────────────────────
def _read_kaldi(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def _resolve_wav(scp_value: str, wav_base: Path | None = None) -> str:
    if scp_value.rstrip().endswith("|"):
        raise ValueError(
            f"wav.scp pipe entries are not supported, materialise wavs first: {scp_value!r}"
        )
    p = Path(scp_value)
    if p.is_file():
        return str(p.resolve())
    if wav_base is not None:
        cand = (wav_base / scp_value).resolve()
        if cand.is_file():
            return str(cand)
    if os.path.isfile(scp_value):
        return os.path.abspath(scp_value)
    raise FileNotFoundError(
        f"wav not found for wav.scp entry {scp_value!r} (wav_base={wav_base})"
    )


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


def _read_trials(path: Path) -> list[tuple[str, str, int]]:
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if len(t) >= 3:
                trials.append((t[0], t[1], int(t[2] == "target")))
    return trials


# ──────────────────────────────────────────────────────────────────────
#  VC anonymizer
# ──────────────────────────────────────────────────────────────────────
def _build_vc_config(args: argparse.Namespace, device: str) -> LiveVoiceConfig:
    codec = str(args.codec).lower()
    sample_rate = 24000 if codec == "mimi" else 16000
    return LiveVoiceConfig(
        device=device,
        codec=codec,
        sample_rate=sample_rate,
        hidden_dim=int(args.hidden_dim),
        num_decoder_layers=int(args.num_decoder_layers),
        ffn_dim=4 * int(args.hidden_dim),
        n_codebooks_predict=int(args.n_codebooks),
        content_source=str(args.content_source).lower(),
        speaker_conditioning=str(args.speaker_conditioning).lower(),
        speaker_prefix_len=int(args.speaker_prefix_len),
        speaker_encoder_type=str(args.speaker_encoder_type).lower(),
        speechbrain_source=args.speechbrain_source,
        speechbrain_sample_rate=int(args.speechbrain_sample_rate),
        speechbrain_embedding_dim=int(args.speechbrain_embedding_dim),
        features_dir=None,
        output_dir=args.output_dir,
        use_content_perturbation=bool(args.use_content_perturbation),
    )


def _auto_infer(args: argparse.Namespace) -> None:
    if str(args.content_source).lower() == "auto":
        args.content_source = infer_content_source_from_ckpt(args.ckpt) or "hubert"
        print(f"[eval] content_source=auto → {args.content_source!r}")
    if str(args.speaker_conditioning).lower() == "auto":
        args.speaker_conditioning = infer_speaker_conditioning_from_ckpt(args.ckpt) or "prefix"
        print(f"[eval] speaker_conditioning=auto → {args.speaker_conditioning!r}")
    if str(args.speaker_encoder_type).lower() == "auto":
        args.speaker_encoder_type = infer_speaker_encoder_from_ckpt(args.ckpt) or "codec"
        print(f"[eval] speaker_encoder_type=auto → {args.speaker_encoder_type!r}")


def _build_vc_model(args, cfg, dev):
    codec_model = build_codec(cfg)
    if cfg.content_source == "hubert":
        content_extractor = HuBERTContentExtractor(cfg)
    elif cfg.content_source == "streamvoiceanon":
        content_extractor = StreamVoiceAnonContentEncoder(cfg)
    else:
        content_extractor = None
    core = LiveVoiceModel(cfg, codec_model, content_extractor, prosody_extractor=None)
    missing, unexpected = load_model_weights_from_ckpt(core, args.ckpt, log_prefix="[eval]")
    if missing:
        print(f"[eval] warn: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"[eval] warn: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    lit = LiveVoiceLightningModule(cfg, core)
    lit.eval()
    return lit.to(dev)


def _build_pool(path: str, wav_base: Path | None = None) -> list[str]:
    p = Path(path)
    scp = p / "wav.scp"
    if scp.is_file():
        return [_resolve_wav(v, wav_base) for v in _read_kaldi(scp).values()]
    wavs = [str(x) for ext in ("*.flac", "*.wav") for x in p.rglob(ext)]
    if not wavs:
        raise SystemExit(f"--anon_pool_dir has no wav.scp and no .flac/.wav under {path}")
    return sorted(wavs)


_STRATEGY_K = {"libri-4rnd": 4, "libri-1rnd": 1, "libri-1fix": 1, "single": 1}


def _assign_pseudo_targets(src_speakers: list[str], pool_wavs: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    return {spk: rng.choice(pool_wavs) for spk in sorted(src_speakers)}


def _select_refs(strategy, utt, utt2spk, pool_wavs, k, seed, fixed_ref, pseudo, level="utt"):
    if strategy == "libri-1fix":
        return fixed_ref
    if strategy == "single":
        return [pseudo[utt2spk[utt]]]
    key = utt2spk[utt] if level == "spk" else utt
    return random.Random(f"{seed}:{key}").sample(pool_wavs, min(k, len(pool_wavs)))


def _load_ref(path, target_sr, crop_sec) -> torch.Tensor:
    w = _load_full_mono_wav(path, target_sr)
    n = int(crop_sec * target_sr)
    if crop_sec and crop_sec > 0 and w.numel() > n:
        start = random.Random(f"crop:{path}").randint(0, w.numel() - n)
        w = w[start : start + n]
    return w


@torch.no_grad()
def _anonymize_svanon(lit, content_wav, ref_wavs, target_sr, dev, alpha, gen_kwargs, crop_sec):
    ctn = _load_full_mono_wav(content_wav, target_sr).unsqueeze(0).to(dev)
    crop = crop_sec if len(ref_wavs) > 1 else 0.0
    refs = [_load_ref(r, target_sr, crop).unsqueeze(0).to(dev) for r in ref_wavs]
    codes = lit.model.generate_anonymized(
        content_audio=ctn, ref_audios=refs, alpha=float(alpha), **gen_kwargs
    )
    return lit.model.decode_to_audio(codes)[0].detach().float().cpu()


@torch.no_grad()
def _anonymize(lit, content_wav, ref_wav, target_sr, dev, gen_kwargs) -> torch.Tensor:
    ctn = _load_full_mono_wav(content_wav, target_sr).unsqueeze(0).to(dev)
    ref = _load_full_mono_wav(ref_wav, target_sr).unsqueeze(0).to(dev)
    codes = lit.model.generate(reference_audio=ref, content_audio=ctn, **gen_kwargs)
    return lit.model.decode_to_audio(codes)[0].detach().float().cpu()


# ──────────────────────────────────────────────────────────────────────
#  Embeddings + metrics
# ──────────────────────────────────────────────────────────────────────
def _enroll_spk_embeddings(
    attacker: VPCASVAttacker,
    enroll_dir: Path,
    target_sr,
    *,
    anonymize: bool = False,
    lit=None,
    cfg=None,
    gen_kwargs=None,
    dev=None,
    args=None,
) -> dict[str, torch.Tensor]:
    wav = _read_kaldi(enroll_dir / "wav.scp")
    utt2spk = _read_kaldi(enroll_dir / "utt2spk")
    acc: dict[str, list[torch.Tensor]] = {}
    pool_wavs = _build_pool(args.anon_pool_dir, args.wav_base) if anonymize else None
    strategy = str(args.anon_strategy).lower() if anonymize else ""
    k = int(args.anon_k) if anonymize and int(args.anon_k) > 0 else _STRATEGY_K.get(strategy, 1)
    fixed_ref = pseudo = None
    if anonymize:
        if strategy == "libri-1fix":
            fixed_ref = [random.Random(args.anon_seed).choice(pool_wavs)]
        elif strategy == "single":
            pseudo = _assign_pseudo_targets(list(set(utt2spk.values())), pool_wavs, args.anon_seed)

    for utt, scp in tqdm(wav.items(), desc="enroll anon+emb" if anonymize else "enroll emb"):
        src = _resolve_wav(scp, args.wav_base)
        if anonymize:
            cached = (
                os.path.join(args.anon_out_dir, "enrolls", f"{utt}.wav")
                if args.anon_out_dir
                else None
            )
            if cached and os.path.isfile(cached):
                path_for_asv = cached
            elif getattr(args, "cache_only", False):
                raise FileNotFoundError(f"--cache_only: missing enroll wav {cached}")
            else:
                refs = _select_refs(
                    strategy, utt, utt2spk, pool_wavs, k, args.anon_seed,
                    fixed_ref, pseudo, args.anon_level,
                )
                aud = _anonymize_svanon(
                    lit, src, refs, int(cfg.sample_rate), dev,
                    args.alpha, gen_kwargs, args.ref_crop_sec,
                )
                if not cached:
                    raise RuntimeError("anon_out_dir required for VPC ASV embedding")
                os.makedirs(os.path.dirname(cached), exist_ok=True)
                sf.write(cached, aud.numpy(), int(cfg.sample_rate))
                path_for_asv = cached
            emb = attacker.embed_path(path_for_asv)
        else:
            emb = attacker.embed_path(src)
        acc.setdefault(utt2spk[utt], []).append(emb)
    return {spk: torch.stack(v).mean(0) for spk, v in acc.items()}


def _trial_utt_embeddings(
    attacker: VPCASVAttacker, trial_dir: Path, anonymize: bool, lit, cfg, gen_kwargs, dev, args,
) -> dict[str, torch.Tensor]:
    wav = _read_kaldi(trial_dir / "wav.scp")
    utt2spk = _read_kaldi(trial_dir / "utt2spk")
    target_sr = int(cfg.sample_rate) if cfg is not None else attacker.sample_rate

    strategy = str(args.anon_strategy).lower()
    k = int(args.anon_k) if int(args.anon_k) > 0 else _STRATEGY_K[strategy]
    fixed_ref = pseudo = None
    pool_wavs = None
    if anonymize:
        pool_wavs = _build_pool(args.anon_pool_dir, args.wav_base)
        if strategy == "libri-1fix":
            fixed_ref = [random.Random(args.anon_seed).choice(pool_wavs)]
        elif strategy == "single":
            pseudo = _assign_pseudo_targets(list(set(utt2spk.values())), pool_wavs, args.anon_seed)
        if args.anon_out_dir:
            os.makedirs(args.anon_out_dir, exist_ok=True)

    out: dict[str, torch.Tensor] = {}
    for utt, scp in tqdm(wav.items(), desc="anon+emb" if anonymize else "trial emb"):
        src = _resolve_wav(scp, args.wav_base)
        if anonymize:
            cached = os.path.join(args.anon_out_dir, "trials", trial_dir.name, f"{utt}.wav") if args.anon_out_dir else None
            if cached and os.path.isfile(cached):
                path_for_asv = cached
            elif getattr(args, "cache_only", False):
                raise FileNotFoundError(f"--cache_only: missing trial wav {cached}")
            else:
                refs = _select_refs(
                    strategy, utt, utt2spk, pool_wavs, k, args.anon_seed,
                    fixed_ref, pseudo, args.anon_level,
                )
                aud = _anonymize_svanon(
                    lit, src, refs, target_sr, dev, args.alpha, gen_kwargs, args.ref_crop_sec
                )
                if not cached:
                    raise RuntimeError("anon_out_dir required for VPC ASV embedding")
                os.makedirs(os.path.dirname(cached), exist_ok=True)
                sf.write(cached, aud.numpy(), target_sr)
                path_for_asv = cached
            emb = attacker.embed_path(path_for_asv)
        else:
            emb = attacker.embed_path(src)
        out[utt] = emb
    return out


def _compute_eer(trials, enroll_embs, trial_embs):
    return vpc_cosine_eer(trials, enroll_embs, trial_embs)


def _scenario_anon_flags(scenario: str) -> tuple[bool, bool]:
    if scenario == "oo":
        return False, False
    if scenario == "oa":
        return False, True
    if scenario == "aa":
        return True, True
    raise ValueError(f"Unknown scenario {scenario!r}")


def _resolve_dataset_jobs(args) -> list[dict]:
    if args.enroll_dir and args.trial_dir:
        return [{
            "dataset": Path(args.trial_dir).name,
            "enroll_dir": Path(args.enroll_dir),
            "trial_dir": Path(args.trial_dir),
            "asr_dir": Path(args.asr_dir) if args.asr_dir else None,
        }]
    if not args.data_root or not args.datasets:
        raise SystemExit("Provide --enroll_dir+--trial_dir OR --data_root+--datasets")
    data_root = Path(args.data_root)
    jobs = []
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        jobs.append({
            "dataset": ds,
            "enroll_dir": data_root / f"{ds}_enrolls",
            "trial_dir": data_root / f"{ds}_trials_mixed",
            "asr_dir": data_root / ds,
        })
    return jobs


@torch.no_grad()
def _anonymize_kaldi_wavs(kaldi_dir: Path, lit, cfg, gen_kwargs, dev, args, out_subdir: str) -> dict[str, str]:
    wav = _read_kaldi(kaldi_dir / "wav.scp")
    utt2spk = _read_kaldi(kaldi_dir / "utt2spk")
    pool_wavs = _build_pool(args.anon_pool_dir, args.wav_base)
    strategy = str(args.anon_strategy).lower()
    k = int(args.anon_k) if int(args.anon_k) > 0 else _STRATEGY_K[strategy]
    fixed_ref = pseudo = None
    if strategy == "libri-1fix":
        fixed_ref = [random.Random(args.anon_seed).choice(pool_wavs)]
    elif strategy == "single":
        pseudo = _assign_pseudo_targets(list(set(utt2spk.values())), pool_wavs, args.anon_seed)
    target_sr = int(cfg.sample_rate)
    anon_map: dict[str, str] = {}
    base_out = Path(args.anon_out_dir) / out_subdir
    base_out.mkdir(parents=True, exist_ok=True)
    for utt, scp in tqdm(wav.items(), desc=f"anon {out_subdir}"):
        src = _resolve_wav(scp, args.wav_base)
        dst = base_out / f"{utt}.wav"
        if dst.is_file():
            anon_map[utt] = str(dst)
            continue
        refs = _select_refs(
            strategy, utt, utt2spk, pool_wavs, k, args.anon_seed,
            fixed_ref, pseudo, args.anon_level,
        )
        aud = _anonymize_svanon(lit, src, refs, target_sr, dev, args.alpha, gen_kwargs, args.ref_crop_sec)
        sf.write(str(dst), aud.numpy(), target_sr)
        anon_map[utt] = str(dst)
    return anon_map


def _run_asr_for_dataset(asr: VPCASREvaluator, orig_dir: Path, anon_dir: Path | None, dataset: str, results_dir: Path | None):
    rows = []
    for tag, data_dir in [("original", orig_dir), ("anon", anon_dir)]:
        if data_dir is None:
            continue
        ref_texts = _read_kaldi(data_dir / "text")
        out_text = results_dir / dataset / tag / "text" if results_dir else None
        hyp_texts = asr.transcribe_dir(data_dir, out_text=out_text)
        wer_path = results_dir / dataset / tag / "wer" if results_dir else None
        wer, _ = asr.compute_wer(ref_texts, hyp_texts, out_file=wer_path)
        print(f"[wer] {dataset} {tag}: WER={wer:.3f}%")
        parts = dataset.split("_", 1)
        rows.append({
            "metric": "WER",
            "dataset": parts[0],
            "split": parts[1] if len(parts) > 1 else dataset,
            "audio": tag,
            "WER": round(wer, 3),
        })
    return rows


def _run_eer_for_job(job, args, attacker: VPCASVAttacker, scenarios, lit, cfg, gen_kwargs, dev):
    trials = _read_trials(job["trial_dir"] / args.trials_name)
    rows = []
    for scenario in scenarios:
        if scenario not in {"oo", "oa", "aa"}:
            print(f"[eval] skip unsupported scenario {scenario!r}")
            continue
        anon_enroll, anon_trial = _scenario_anon_flags(scenario)
        enroll_embs = _enroll_spk_embeddings(
            attacker, job["enroll_dir"], attacker.sample_rate,
            anonymize=anon_enroll, lit=lit, cfg=cfg, gen_kwargs=gen_kwargs, dev=dev, args=args,
        )
        trial_embs = _trial_utt_embeddings(
            attacker, job["trial_dir"], anon_trial, lit, cfg, gen_kwargs, dev, args,
        )
        eer, n_pos, n_neg, skipped = _compute_eer(trials, enroll_embs, trial_embs)
        print(
            f"[eer] {job['dataset']} {scenario.upper()}-EER = {eer:.3f}%  "
            f"(target={n_pos} nontarget={n_neg} skipped={skipped})"
        )
        rows.append({
            "metric": "EER",
            "dataset": job["dataset"],
            "scenario": scenario,
            "enrollment": "original" if not anon_enroll else "anon",
            "trial": "original" if not anon_trial else "anon",
            "EER": round(eer, 3),
            "n_target": n_pos,
            "n_nontarget": n_neg,
            "skipped": skipped,
        })
    return rows


def _build_attacker(args, dev: str) -> VPCASVAttacker:
    vpc_root = args.vpc_root or DEFAULT_VPC_ROOT
    exp_dir = Path(args.vpc_exp_dir or Path(vpc_root) / "exp")
    asv_dir = exp_dir / "asv_ssl"
    print(f"[eval] ASV: VPC ecapa_ssl from {asv_dir}")
    return VPCASVAttacker(asv_dir, dev, vpc_root)


def main() -> None:
    p = argparse.ArgumentParser(description="VPC eval_pre-style EER + ASR (oo/oa/aa)")
    p.add_argument("--enroll_dir", default=None)
    p.add_argument("--trial_dir", default=None)
    p.add_argument("--asr_dir", default=None, help="Kaldi dir with wav.scp+text (single-set ASR)")
    p.add_argument("--data_root", default='/mnt/data/disk3/yejin/VPC/data', help="VPC data/ root for batch mode")
    p.add_argument("--datasets", default="libri_dev,libri_test")
    p.add_argument("--trials_name", default="trials")
    p.add_argument("--metrics", default="eer,wer", help="comma list: eer,wer")
    p.add_argument("--scenarios", default="oo,aa")
    p.add_argument("--out_csv", default="/mnt/data/disk2/yejin/LiveVoice/vpc_eval_pre.csv")
    p.add_argument("--results_dir", default=None, help="optional dir for ASR transcripts/WER files")

    p.add_argument("--vpc_root", default=DEFAULT_VPC_ROOT)
    p.add_argument("--vpc_exp_dir", default=None, help="default: <vpc_root>/exp")

    p.add_argument("--ckpt", default=None)
    p.add_argument("--anon_pool_dir", default="/mnt/data/disk2/LibriTTS/train-other-500")
    p.add_argument("--anon_strategy", default="libri-4rnd",
                   choices=["libri-4rnd", "libri-1rnd", "libri-1fix", "single"])
    p.add_argument("--anon_level", default="utt", choices=["utt", "spk"])
    p.add_argument("--anon_k", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--ref_crop_sec", type=float, default=3.0)
    p.add_argument("--anon_out_dir", default='/mnt/data/disk2/yejin/LiveVoice/anon')
    p.add_argument(
        "--cache_only",
        action="store_true",
        help="use cached anon wavs only (skip VC model load; EER with oa/aa)",
    )
    p.add_argument("--anon_seed", type=int, default=1234)
    p.add_argument("--use_content_perturbation", type=int, default=1)

    p.add_argument("--asr_batch_size", type=int, default=8)

    p.add_argument("--codec", default="jhcodec", choices=["mimi", "jhcodec"])
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_decoder_layers", type=int, default=12)
    p.add_argument("--n_codebooks", type=int, default=8)
    p.add_argument("--content_source", default="auto")
    p.add_argument("--speaker_conditioning", default="auto")
    p.add_argument("--speaker_encoder_type", default="auto")
    p.add_argument("--speaker_prefix_len", type=int, default=8)
    p.add_argument("--speechbrain_source", default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument("--speechbrain_sample_rate", type=int, default=16000)
    p.add_argument("--speechbrain_embedding_dim", type=int, default=192)
    p.add_argument("--output_dir", default="/mnt/data/disk2/yejin/LiveVoice")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    args.wav_base = Path(args.vpc_root or DEFAULT_VPC_ROOT).resolve()

    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    need_anon = ("wer" in metrics) or any(s in {"oa", "aa"} for s in scenarios)
    if args.cache_only and "wer" in metrics:
        raise SystemExit("--cache_only supports EER only (use pre-built anon wavs)")
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    jobs = _resolve_dataset_jobs(args)

    attacker = _build_attacker(args, str(dev))

    lit = cfg = gen_kwargs = None
    if need_anon and not args.cache_only:
        if not args.ckpt:
            raise SystemExit("anonymized metrics/scenarios require --ckpt (or --cache_only)")
        if not args.anon_pool_dir:
            raise SystemExit("anonymized metrics/scenarios require --anon_pool_dir")
        if not args.anon_out_dir:
            raise SystemExit("anonymized metrics/scenarios require --anon_out_dir (cache)")
        _auto_infer(args)
        cfg = _build_vc_config(args, str(dev))
        lit = _build_vc_model(args, cfg, dev)
        temp = float(args.temperature)
        gen_kwargs = dict(
            temperature=temp,
            top_p=float(args.top_p) if temp > 0 and args.top_p > 0 else None,
            top_k=int(args.top_k) if temp > 0 and args.top_k > 0 else None,
            cfg_scale=float(args.cfg_scale),
        )
    elif need_anon and args.cache_only:
        if not args.anon_out_dir:
            raise SystemExit("--cache_only requires --anon_out_dir with pre-built wavs")
        print(f"[eval] cache_only: using anon wavs from {args.anon_out_dir} (no VC load)")

    rows: list[dict] = []
    results_dir = Path(args.results_dir) if args.results_dir else None

    if "eer" in metrics:
        for job in jobs:
            rows.extend(_run_eer_for_job(
                job, args, attacker, scenarios, lit, cfg, gen_kwargs, dev,
            ))

    if "wer" in metrics:
        vpc_root = args.vpc_root or DEFAULT_VPC_ROOT
        asr_dir = Path(args.vpc_exp_dir or Path(vpc_root) / "exp") / "asr"
        print(f"[eval] WER model: VPC wav2vec2-CTC from {asr_dir}")
        asr = VPCASREvaluator(asr_dir, str(dev), vpc_root, batch_size=args.asr_batch_size)
        for job in jobs:
            asr_orig = job["asr_dir"]
            if asr_orig is None or not asr_orig.is_dir():
                print(f"[wer] skip {job['dataset']}: no asr_dir")
                continue
            anon_kaldi = None
            if need_anon:
                anon_map = _anonymize_kaldi_wavs(
                    asr_orig, lit, cfg, gen_kwargs, dev, args, out_subdir=f"asr/{job['dataset']}",
                )
                anon_kaldi = materialize_anon_kaldi_dir(
                    asr_orig, Path(args.anon_out_dir) / f"asr_kaldi/{job['dataset']}", anon_map,
                )
            rows.extend(_run_asr_for_dataset(asr, asr_orig, anon_kaldi, job["dataset"], results_dir))

    if rows:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"[eval] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
