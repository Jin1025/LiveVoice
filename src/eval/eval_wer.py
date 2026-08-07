"""Same-speaker WER for LiveVoice checkpoints (content-preservation sanity check).

  * Ground-truth text = the content wav's ``*.normalized.txt``.
  * Whisper (default large-v3, matching the GT-WER upper bound of ~0.05) runs on the
    generated VC audio.
  * Reference audio = a different utterance from the SAME speaker (LibriTTS val rule).
  * Greedy decoding (temperature=0) — sampling makes VC output stochastic and inflates WER.

Run:

    CUDA_VISIBLE_DEVICES=6 python src/eval/eval_wer.py --speaker_mode same --whisper_model medium --ckpt ecapa_prefix_jh_hubert/step_latest.ckpt

``--ckpt`` is relative to ``CKPT_ROOT`` (override with ``--ckpt_root``) unless absolute.
Pass multiple ``--ckpt`` values to evaluate several checkpoints in sequence.
Writes one CSV each (columns: gt, hyp, wer) and prints the mean WER at the end.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
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
    Sw2vContentEncoder,
    LiveVoiceModel,
    build_codec,
)
from livevoice.utils.checkpoint import (
    infer_content_fsq_from_ckpt,
    infer_content_source_from_ckpt,
    infer_speaker_conditioning_from_ckpt,
    infer_speaker_encoder_from_ckpt,
    infer_codec_prompt_flags_from_ckpt,
)

# ─────────────────────────────────────────────
#  Checkpoints to evaluate
# ─────────────────────────────────────────────
CKPT_ROOT = "/mnt/data/disk2/yejin/LiveVoice/checkpoints"
JHCODEC_CKPT = "/mnt/data/disk3/yejin/jhcodec/jhcodec_mimi_1000000.pt"

# ─────────────────────────────────────────────
#  WER + audio helpers (self-contained)
# ─────────────────────────────────────────────
def _normalize_text_for_wer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _word_wer(hypothesis: str, reference: str) -> float:
    """Levenshtein word error rate = edit_distance(hyp, ref) / len(ref_words)."""
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


def _load_full_mono_wav(path: str, target_sr: int, max_seconds: float | None = None) -> torch.Tensor:
    """Mono utterance at target_sr, peak-normalized. Trimmed to max_seconds if given."""
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
    if max_seconds is not None:
        audio = audio[: int(round(max_seconds * target_sr))]
    return audio / (audio.abs().max() + 1e-8)


def _read_normalized_text(wav_path: str) -> str | None:
    txt_path = Path(wav_path).with_suffix(".normalized.txt")
    if not txt_path.exists():
        return None
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        return text if text else None
    except Exception:
        return None


def _prepare_whisper_audio(gen: torch.Tensor, src_sr: int) -> np.ndarray:
    """Mono float32 16 kHz, peak-normalized (matches eval_whisper_wer_one)."""
    wavs = gen.detach().float().cpu()
    if wavs.dim() == 1:
        wavs = wavs.unsqueeze(0)
    peak = wavs.abs().max()
    if peak > 0:
        wavs = wavs / (peak + 1e-8)
    if src_sr != 16000:
        wavs = torchaudio.functional.resample(wavs, src_sr, 16000)
    return wavs.squeeze().numpy().astype(np.float32)


def _same_speaker_ref(ds: LibriTTSDataset, content_wav: str, utt_id: str, spk: str) -> str:
    """Pick another utterance from the SAME speaker as the timbre reference."""
    spk_utts = ds.speaker_utts[spk]
    ref_path, _ = next(
        ((p, u) for p, u in spk_utts if p != content_wav),
        (content_wav, utt_id),
    )
    return ref_path


def _cross_speaker_ref(ds: LibriTTSDataset, content_spk: str, rng: random.Random) -> str:
    """Pick one utterance from a DIFFERENT speaker (seeded rng → reproducible)."""
    others = [s for s in ds.speaker_utts if s != content_spk]
    if not others:
        raise RuntimeError(f"No speaker other than {content_spk} for cross-speaker ref.")
    ref_spk = rng.choice(others)
    ref_path, _ = rng.choice(ds.speaker_utts[ref_spk])
    return ref_path


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────
def _build_config(args, ckpt: str, device: str) -> LiveVoiceConfig:
    content_source = infer_content_source_from_ckpt(ckpt) or "hubert"
    speaker_conditioning = infer_speaker_conditioning_from_ckpt(ckpt) or "prefix"
    speaker_encoder_type = infer_speaker_encoder_from_ckpt(ckpt) or "codec"
    fsq_levels = infer_content_fsq_from_ckpt(ckpt)
    cp_flags = infer_codec_prompt_flags_from_ckpt(ckpt)
    print(
        f"[eval] {Path(ckpt).parent.name}: content_source={content_source} "
        f"speaker_conditioning={speaker_conditioning} speaker_encoder_type={speaker_encoder_type} "
        f"content_fsq={fsq_levels if fsq_levels else 'off'} codec_prompt={cp_flags}"
    )
    return LiveVoiceConfig(
        **cp_flags,
        device=device,
        codec="jhcodec",
        sample_rate=16000,
        jhcodec_repo=args.jhcodec_repo,
        jhcodec_config=args.jhcodec_config,
        jhcodec_ckpt=args.jhcodec_ckpt,
        hidden_dim=int(args.hidden_dim),
        num_decoder_layers=int(args.num_decoder_layers),
        ffn_dim=4 * int(args.hidden_dim),
        n_codebooks_predict=int(args.n_codebooks),
        content_source=content_source,
        speaker_conditioning=speaker_conditioning,
        speaker_prefix_len=int(args.speaker_prefix_len),
        speaker_encoder_type=speaker_encoder_type,
        use_content_fsq=fsq_levels is not None,
        fsq_levels=fsq_levels if fsq_levels is not None else (8, 5, 5, 5),
        features_dir=None,
    )


def _load_model(cfg: LiveVoiceConfig, ckpt: str, device: torch.device):
    codec_model = build_codec(cfg)
    if cfg.content_source == "hubert":
        content_extractor = HuBERTContentExtractor(cfg)
    elif cfg.content_source == "streamvoiceanon":
        content_extractor = StreamVoiceAnonContentEncoder(cfg)
    elif cfg.content_source == "sw2v":
        content_extractor = Sw2vContentEncoder(cfg)
    else:
        content_extractor = None
    core = LiveVoiceModel(cfg, codec_model, content_extractor, prosody_extractor=None)
    lit = LiveVoiceLightningModule.load_from_checkpoint(
        ckpt, config=cfg, model=core, strict=False
    )
    lit.eval()
    return lit.to(device)


# ─────────────────────────────────────────────
#  Per-checkpoint evaluation
# ─────────────────────────────────────────────
def eval_ckpt(name: str, ckpt: str, ds: LibriTTSDataset, w_model, args) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = _build_config(args, ckpt, "cuda:0" if device.type == "cuda" else "cpu")
    lit = _load_model(cfg, ckpt, device)
    target_sr = int(cfg.sample_rate)

    # For the codec speaker encoder, the *reference audio length* directly sets the
    # prefix length (every jhcodec frame → one prefix token, ~50 fps). Training used
    # audio_duration windows (~4s → ~200 tokens); a full-length reference at inference
    # both mismatches training and can overflow the 1024 KV-cache budget (prefix +
    # generated content share it). So trim the reference for the codec path. ECAPA
    # pools the whole utterance to a single vector → fixed 8 tokens, so it keeps the
    # full reference (more audio = better embedding).
    ref_max_seconds = (
        None if cfg.speaker_encoder_type == "speechbrain_ecapa" else float(args.ref_max_seconds)
    )
    print(
        f"[eval] {name}: reference trim = "
        f"{'full utterance' if ref_max_seconds is None else f'{ref_max_seconds:g}s (~{int(ref_max_seconds*50)} prefix tokens)'}"
    )

    speaker_mode = str(getattr(args, "speaker_mode", "same")).lower()
    if speaker_mode == "cross":
        # Different-speaker reference, seeded per item → same pairs every run.
        ref_seed = int(getattr(args, "ref_seed", 12345))
        work = [
            (wav_path, _cross_speaker_ref(ds, spk, random.Random(ref_seed * 1_000_003 + i)),
             utt_id, spk)
            for i, (wav_path, utt_id, spk) in enumerate(ds.items)
        ]
    else:
        work = [
            (wav_path, _same_speaker_ref(ds, wav_path, utt_id, spk), utt_id, spk)
            for wav_path, utt_id, spk in ds.items
        ]
    print(f"[eval] {name}: {speaker_mode}-speaker pairs = {len(work)}")

    out_csv = args.out_csv_tmpl.format(name=name, mode=speaker_mode)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    whisper_kw = {
        "language": "en",
        "condition_on_previous_text": False,
        "fp16": bool(device.type == "cuda"),
    }

    wers: list[float] = []
    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["gt", "hyp", "wer"])
        writer.writeheader()
        for wav_path, ref_path, _utt, _spk in tqdm(work, desc=f"{name} WER"):
            gt = _read_normalized_text(wav_path) or ""
            hyp = ""
            wer_v = float("nan")
            if not gt:
                writer.writerow({"gt": "", "hyp": "", "wer": ""})
                fcsv.flush()
                continue
            try:
                ctn = _load_full_mono_wav(wav_path, target_sr).unsqueeze(0).to(device)
                ref = _load_full_mono_wav(ref_path, target_sr, max_seconds=ref_max_seconds).unsqueeze(0).to(device)
                with torch.no_grad():
                    codes = lit.model.generate(
                        reference_audio=ref,
                        content_audio=ctn,
                        temperature=0.0,
                        top_p=None,
                        top_k=None,
                        cfg_scale=float(args.cfg_scale),
                    )
                    gen = lit.model.decode_to_audio(codes)
                arr = _prepare_whisper_audio(gen, target_sr)
                hyp = w_model.transcribe(arr, **whisper_kw)["text"].strip()
                wer_v = _word_wer(hyp, gt)
                if wer_v == wer_v:  # not NaN
                    wers.append(wer_v)
            except Exception as e:  # keep going; record the failure in the row
                hyp = f"[ERROR] {str(e)[:300]}"
            writer.writerow(
                {"gt": gt, "hyp": hyp, "wer": f"{wer_v:.6f}" if wer_v == wer_v else ""}
            )
            fcsv.flush()

    mean_wer = float(np.mean(wers)) if wers else float("nan")
    print(f"[eval] {name}: wrote {out_csv}")
    print(f"[eval] {name}: MEAN {speaker_mode}-speaker WER = {mean_wer:.4f}  (n={len(wers)})")

    # Free GPU memory before the next checkpoint.
    del lit
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean_wer


def _resolve_ckpt(c: str, root: str) -> tuple[str, str]:
    """Resolve ``--ckpt`` against ``CKPT_ROOT``; return (run_name, absolute_path)."""
    path = c if os.path.isabs(c) or os.path.isfile(c) else os.path.join(root, c)
    if not os.path.isfile(path):
        raise SystemExit(f"[eval] checkpoint not found: {path}")
    name = Path(path).parent.name
    return name, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, nargs="+", default=["ecapa_prefix_jh_hubert/step_latest.ckpt"])
    p.add_argument("--ckpt_root", type=str, default=CKPT_ROOT)
    p.add_argument("--speaker_mode", type=str, default="same", choices=["same", "cross"],
                   help="Reference speaker: 'same' (intelligibility upper bound — another "
                        "utterance of the same speaker) or 'cross' (a different speaker → "
                        "speaker-transfer VC, exposes content<->speaker leakage).")
    p.add_argument("--ref_seed", type=int, default=12345,
                   help="Seed for cross-speaker reference selection (fixed pairs across runs).")
    p.add_argument("--libritts_path", type=str, default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--split_dir", type=str, default="dev-clean",
                   help="LibriTTS subdir for same-speaker eval (dev-clean matches training val).")
    p.add_argument("--ref_max_seconds", type=float, default=4.0,
                   help="Trim reference to this many seconds for the codec speaker encoder "
                        "(prefix length = ref_seconds*50fps; training used audio_duration=4s). "
                        "Ignored for ECAPA (fixed 8 tokens). Set very large to disable trimming.")
    p.add_argument("--n_codebooks", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_decoder_layers", type=int, default=12)
    p.add_argument("--speaker_prefix_len", type=int, default=4)
    # Default follows config.val_cfg_scale so eval matches what validation reports.
    # cfg=1.0 measures the model well below what it can do (see config.val_cfg_scale).
    p.add_argument("--cfg_scale", type=float, default=LiveVoiceConfig.val_cfg_scale)
    p.add_argument("--whisper_model", type=str, default="medium",
                   help="Whisper model for hyp transcription (GT upper bound used large-v3).")
    p.add_argument("--whisper_device", type=str, default="cuda")
    p.add_argument("--jhcodec_repo", type=str, default="/workspace/jhcodec")
    p.add_argument("--jhcodec_config", type=str,
                   default="/workspace/jhcodec/config/config_mimi_recon.json")
    p.add_argument("--jhcodec_ckpt", type=str, default=JHCODEC_CKPT)
    p.add_argument("--max_items", type=int, default=None, help="Cap #utterances (debug).")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--out_csv_tmpl", type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/exp/wer_{mode}_speaker_{name}.csv",
        help="Output CSV path template; {name}=checkpoint, {mode}=same/cross.",
    )
    args = p.parse_args()

    targets = [_resolve_ckpt(c, args.ckpt_root) for c in args.ckpt]

    # Same-speaker val dataset on the chosen split (paths only; audio loaded full later).
    cfg_ds = LiveVoiceConfig(
        libritts_path=args.libritts_path,
        libritts_val_splits=(args.split_dir,),
        codec="jhcodec",
        sample_rate=16000,
        max_windows=args.max_items,
        seed=42,
        pairing="same_speaker",
        audio_duration=4.0,
        features_dir=None,
    )
    ds = LibriTTSDataset(cfg_ds, split="val")

    import whisper
    print(f"[eval] loading Whisper '{args.whisper_model}' on {args.whisper_device} ...")
    w_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    results: dict[str, float] = {}
    for name, path in targets:
        print(f"\n{'=' * 70}\n[eval] checkpoint: {name}\n       {path}\n{'=' * 70}")
        results[name] = eval_ckpt(name, path, ds, w_model, args)

    print(f"\n{'=' * 70}\n[eval] SUMMARY — {str(args.speaker_mode).lower()}-speaker mean WER\n{'=' * 70}")
    for name, wer in results.items():
        print(f"  {name:35s} : {wer:.4f}")


if __name__ == "__main__":
    main()
