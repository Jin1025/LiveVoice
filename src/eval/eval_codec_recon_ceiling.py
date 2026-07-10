"""Codec reconstruction WER ceiling for jhcodec (intelligibility upper bound).

Motivation
----------
Same-speaker VC WER converged to ~0.10-0.14 and stopped improving. Any model that
emits jhcodec tokens is hard-capped by the codec's own vocoder loss: even a *perfect*
content prediction can only be as intelligible as ``GT audio → jhcodec encode → decode``.
This script measures exactly that floor.

    GT wav → jhcodec.encode(K books) → jhcodec.decode → Whisper → WER vs normalized.txt

Interpretation
--------------
  * recon WER ≈ 0.10-0.13  → you have hit the codec wall. The VC model is essentially
    done on intelligibility; further gains need a better/higher-bitrate codec, not more
    training.
  * recon WER ≈ 0.05-0.06 (≈ GT-WER upper bound) → the codec is innocent; the residual
    VC gap is your model (content leakage / alignment / AR errors) → keep pushing.

The model predicts all 8 codebooks (config.n_codebooks_predict=8), so K=8 is the
fair, matched ceiling. ``--n_codebooks_sweep 1,2,4,8`` shows how the ceiling moves
with bitrate — useful to argue "8 books is / isn't the limiter".

Run (env with whisper/torch — e.g. `conda activate sound`):

    CUDA_VISIBLE_DEVICES=0 python src/eval/eval_codec_recon_ceiling.py \
        --whisper_model large-v3 --n_codebooks_sweep 1,2,4,8 --max_items 50

Writes one CSV per K (columns: gt, hyp, wer) and prints a mean-WER table.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.data.libritts_dataset import LibriTTSDataset
from livevoice.model import build_codec

# Reuse the *identical* WER / audio / text helpers from the same-speaker eval so
# the ceiling and the VC numbers are computed on exactly the same footing.
from eval.eval_same_speaker_wer import (  # noqa: E402
    _word_wer,
    _load_full_mono_wav,
    _read_normalized_text,
    _prepare_whisper_audio,
    JHCODEC_CKPT,
)


def eval_recon(k: int, ds: LibriTTSDataset, codec, w_model, args, device) -> float:
    """Recon WER using the first ``k`` codebooks."""
    target_sr = 16000
    out_csv = args.out_csv_tmpl.format(k=k)
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
        for wav_path, _utt, _spk in tqdm(ds.items, desc=f"recon K={k}"):
            gt = _read_normalized_text(wav_path) or ""
            hyp = ""
            wer_v = float("nan")
            if not gt:
                writer.writerow({"gt": "", "hyp": "", "wer": ""})
                fcsv.flush()
                continue
            try:
                audio = _load_full_mono_wav(wav_path, target_sr).unsqueeze(0).to(device)
                with torch.no_grad():
                    codes = codec.encode(audio)        # (1, K_full, T)
                    codes = codes[:, :k, :].contiguous()  # keep first k books
                    recon = codec.decode(codes)        # (1, T_samples) @ 16 kHz
                arr = _prepare_whisper_audio(recon, target_sr)
                hyp = w_model.transcribe(arr, **whisper_kw)["text"].strip()
                wer_v = _word_wer(hyp, gt)
                if wer_v == wer_v:  # not NaN
                    wers.append(wer_v)
            except Exception as e:
                hyp = f"[ERROR] {str(e)[:300]}"
            writer.writerow(
                {"gt": gt, "hyp": hyp, "wer": f"{wer_v:.6f}" if wer_v == wer_v else ""}
            )
            fcsv.flush()

    mean_wer = float(np.mean(wers)) if wers else float("nan")
    print(f"[recon] K={k}: wrote {out_csv}")
    print(f"[recon] K={k}: MEAN recon WER = {mean_wer:.4f}  (n={len(wers)})")
    return mean_wer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libritts_path", type=str, default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--split_dir", type=str, default="dev-clean",
                   help="LibriTTS subdir (dev-clean matches the same-speaker VC eval).")
    p.add_argument("--n_codebooks_sweep", type=str, default="8",
                   help="Comma-separated codebook counts to evaluate, e.g. '1,2,4,8'. "
                        "The model predicts 8, so K=8 is the matched ceiling.")
    p.add_argument("--n_codebooks", type=int, default=8,
                   help="Full codebook count to encode with before truncation.")
    p.add_argument("--whisper_model", type=str, default="large-v3",
                   help="Whisper model for hyp transcription (GT upper bound used large-v3).")
    p.add_argument("--whisper_device", type=str, default="cuda")
    p.add_argument("--jhcodec_repo", type=str, default="/workspace/jhcodec")
    p.add_argument("--jhcodec_config", type=str,
                   default="/workspace/jhcodec/config/config_mimi_recon.json")
    p.add_argument("--jhcodec_ckpt", type=str, default=JHCODEC_CKPT)
    p.add_argument("--max_items", type=int, default=None,
                   help="Cap #utterances (None means all utterances).")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--out_csv_tmpl", type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/wer_codec_recon_K{k}.csv",
        help="Output CSV path template; {k} is replaced per codebook count.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Path-only dataset on the chosen split (same rule as the VC eval).
    cfg = LiveVoiceConfig(
        device="cuda:0" if device.type == "cuda" else "cpu",
        libritts_path=args.libritts_path,
        libritts_val_splits=(args.split_dir,),
        codec="jhcodec",
        sample_rate=16000,
        jhcodec_repo=args.jhcodec_repo,
        jhcodec_config=args.jhcodec_config,
        jhcodec_ckpt=args.jhcodec_ckpt,
        n_codebooks_predict=int(args.n_codebooks),
        jhcodec_n_codebooks=int(args.n_codebooks),
        max_windows=args.max_items,
        seed=42,
        pairing="same_speaker",
        audio_duration=4.0,
        features_dir=None,
    )
    ds = LibriTTSDataset(cfg, split="val")
    print(f"[recon] {len(ds.items)} utterances from {args.split_dir}")

    print(f"[recon] building jhcodec ({args.jhcodec_ckpt}) ...")
    codec = build_codec(cfg)
    if hasattr(codec, "to"):
        codec = codec.to(device)
    if hasattr(codec, "eval"):
        codec.eval()

    import whisper
    print(f"[recon] loading Whisper '{args.whisper_model}' on {args.whisper_device} ...")
    w_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    ks = [int(x) for x in str(args.n_codebooks_sweep).split(",") if x.strip()]
    results: dict[int, float] = {}
    for k in ks:
        if k < 1 or k > int(args.n_codebooks):
            print(f"[recon] skipping K={k} (out of range 1..{args.n_codebooks})")
            continue
        print(f"\n{'=' * 70}\n[recon] codebooks K={k}\n{'=' * 70}")
        results[k] = eval_recon(k, ds, codec, w_model, args, device)

    print(f"\n{'=' * 70}\n[recon] SUMMARY — jhcodec recon WER ceiling ({args.split_dir}, "
          f"n={args.max_items}, whisper={args.whisper_model})\n{'=' * 70}")
    for k, wer in results.items():
        tag = "  ← matched ceiling (model predicts 8)" if k == 8 else ""
        print(f"  K={k:2d} codebooks : recon WER = {wer:.4f}{tag}")


if __name__ == "__main__":
    main()
