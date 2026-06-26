#!/usr/bin/env python3
"""One-off: run Whisper on a single audio file and WER vs a reference transcript.

Reference: ``--ref_text`` (literal) and/or ``--ref_txt`` (file path). If both are set,
non-empty ``--ref_text`` wins; otherwise the file is read.

Example:
    python scripts/eval_whisper_wer_one.py \\
        --audio /path/to/gen.wav \\
        --ref_text "hello world"

    python scripts/eval_whisper_wer_one.py \\
        --audio /path/to/utt.wav \\
        --ref_txt /path/to/utt.normalized.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio

try:
    import soundfile as sf
except ImportError:
    sf = None


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


def _load_mono_16k(path: str) -> np.ndarray:
    if sf is not None:
        try:
            with sf.SoundFile(path) as f:
                audio_np = f.read(dtype="float32", always_2d=True)
                sr = int(f.samplerate)
            audio = torch.from_numpy(audio_np).float().mean(dim=1)
        except Exception:
            wav, sr = torchaudio.load(path)
            audio = wav.float().mean(dim=0)
            sr = int(sr)
    else:
        wav, sr = torchaudio.load(path)
        audio = wav.float().mean(dim=0)
        sr = int(sr)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio / (audio.abs().max() + 1e-8)
    return audio.numpy().astype(np.float32)


def _read_ref_text(args: argparse.Namespace) -> str:
    if args.ref_text is not None and str(args.ref_text).strip():
        return str(args.ref_text).strip()
    if args.ref_txt is not None:
        p = Path(args.ref_txt)
        if not p.is_file():
            raise FileNotFoundError(f"--ref_txt not found: {p}")
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    raise SystemExit("Provide non-empty --ref_text or a valid --ref_txt file.")


def main() -> None:
    p = argparse.ArgumentParser(description="Whisper transcribe one wav + WER vs reference text.")
    p.add_argument("--audio", type=str, required=True, help="Path to wav/flac/etc.")
    p.add_argument("--ref_text", type=str, default=None, help="Reference transcript (literal string).")
    p.add_argument(
        "--ref_txt",
        type=str,
        default=None,
        help="Path to a text file containing the reference transcript (.normalized.txt ok).",
    )
    p.add_argument("--whisper_model", type=str, default="large-v3")
    p.add_argument("--whisper_device", type=str, default="cuda")
    p.add_argument(
        "--whisper_language",
        type=str,
        default="en",
        help="Whisper language code; empty = auto-detect.",
    )
    p.add_argument("--fp16", action="store_true", help="fp16 decode on GPU (faster).")
    p.add_argument(
        "--whisper_condition_on_previous_text",
        action="store_true",
        help="Pass through to Whisper (default off for long clips).",
    )
    args = p.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise FileNotFoundError(f"--audio not found: {audio_path}")

    if not args.ref_text and not args.ref_txt:
        p.error("Provide --ref_text and/or --ref_txt")

    ref = _read_ref_text(args)
    if not ref:
        raise SystemExit("Reference text is empty after reading.")

    import whisper

    model = whisper.load_model(args.whisper_model, device=args.whisper_device)
    wav = _load_mono_16k(str(audio_path))

    kw: dict = {
        "fp16": bool(args.fp16 and args.whisper_device == "cuda" and torch.cuda.is_available()),
        "condition_on_previous_text": bool(args.whisper_condition_on_previous_text),
    }
    if args.whisper_language:
        kw["language"] = str(args.whisper_language)

    hyp = model.transcribe(wav, **kw)["text"].strip()
    wer = _word_wer(hyp, ref)

    print("---")
    print(f"audio:     {audio_path}")
    print(f"whisper:   model={args.whisper_model!r}  {kw}")
    print(f"reference: {ref[:500]}{'…' if len(ref) > 500 else ''}")
    print(f"hypothesis:{hyp[:500]}{'…' if len(hyp) > 500 else ''}")
    print(f"WER:       {wer:.4f}" if wer == wer else "WER:       nan")
    print("---")


if __name__ == "__main__":
    main()
