from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

try:
    import librosa
except ImportError:
    librosa = None


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


def _read_libritts_normalized_text(wav_path: str) -> str | None:
    txt_path = Path(wav_path).with_suffix(".normalized.txt")
    if not txt_path.exists():
        return None
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        return text if text else None
    except Exception:
        return None


def _load_mono_16k(path: str) -> np.ndarray:
    """Full utterance, mono float32 ~[-1,1], 16 kHz for Whisper."""
    try:
        with sf.SoundFile(path) as f:
            audio_np = f.read(dtype="float32", always_2d=True)
            sr = int(f.samplerate)
        audio = torch.from_numpy(audio_np).float().mean(dim=1)
    except Exception:
        if librosa is None:
            raise
        audio_np, sr = librosa.load(path, sr=None, mono=True)
        audio = torch.from_numpy(audio_np.astype("float32"))
        sr = int(sr)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio / (audio.abs().max() + 1e-8)
    return audio.numpy().astype(np.float32)


def _iter_test_clean_wavs(libritts_path: Path, split_dir: str) -> list[tuple[str, str, str]]:
    """Return sorted list of (wav_path, utt_id, speaker_id)."""
    root = libritts_path / split_dir
    if not root.is_dir():
        raise FileNotFoundError(f"split dir not found: {root}")
    items: list[tuple[str, str, str]] = []
    for wav in sorted(root.glob("**/*.wav")):
        spk = wav.parts[-3]
        utt_id = wav.stem
        items.append((str(wav), utt_id, spk))
    return items


def _load_done_paths(csv_path: str) -> set[str]:
    if not os.path.isfile(csv_path):
        return set()
    done: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = row.get("wav_path", "").strip()
            if p:
                done.add(p)
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="Whisper WER on LibriTTS test-clean (full wav vs normalized.txt).")
    p.add_argument("--libritts_path", type=str, default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--split_dir", type=str, default="test-clean")
    p.add_argument(
        "--whisper_model",
        type=str,
        default="large-v3",
        help="openai-whisper model name (e.g. large-v3, large-v2, large, medium, ...).",
    )
    p.add_argument("--whisper_device", type=str, default="cuda")
    p.add_argument("--language", type=str, default="en", help="Whisper language code (e.g. en).")
    p.add_argument("--fp16", action="store_true", help="Use fp16 in transcribe (faster on GPU).")
    p.add_argument(
        "--out_csv",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/gt_wer.csv",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_items", type=int, default=None, help="Debug: cap utterances.")
    args = p.parse_args()

    import whisper

    libritts_path = Path(args.libritts_path)
    items = _iter_test_clean_wavs(libritts_path, args.split_dir)
    if args.max_items is not None:
        items = items[: int(args.max_items)]

    print(
        f"[whisper_wer] split={args.split_dir}  utterances={len(items)}  "
        f"model={args.whisper_model!r}  device={args.whisper_device!r}  fp16={args.fp16}"
    )

    try:
        model = whisper.load_model(args.whisper_model, device=args.whisper_device)
    except Exception as e:
        print(
            f"[whisper_wer] load_model failed: {e}\n"
            "Install / upgrade: pip install -U openai-whisper\n"
            "Model names: https://github.com/openai/whisper#available-models-and-languages",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    done = _load_done_paths(args.out_csv) if args.resume else set()
    csv_mode = "a" if args.resume else "w"
    write_header = (csv_mode == "w") or not (
        os.path.isfile(args.out_csv) and os.path.getsize(args.out_csv) > 0
    )

    wers: list[float] = []
    with open(args.out_csv, csv_mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=["wav_path", "utt_id", "speaker", "wer", "hypothesis", "error"],
        )
        if write_header:
            writer.writeheader()

        for wav_path, utt_id, spk in tqdm(items, desc="Whisper WER test-clean"):
            if wav_path in done:
                continue
            err = ""
            wer_v = float("nan")
            hyp = ""
            try:
                ref_txt = _read_libritts_normalized_text(wav_path)
                if not ref_txt:
                    err = "no_normalized_txt"
                    writer.writerow(
                        {
                            "wav_path": wav_path,
                            "utt_id": utt_id,
                            "speaker": spk,
                            "wer": "",
                            "hypothesis": "",
                            "error": err,
                        }
                    )
                    fcsv.flush()
                    continue

                audio = _load_mono_16k(wav_path)
                result = model.transcribe(
                    audio,
                    language=args.language if args.language else None,
                    fp16=bool(args.fp16 and args.whisper_device == "cuda" and torch.cuda.is_available()),
                )
                hyp = (result.get("text") or "").strip()
                wer_v = _word_wer(hyp, ref_txt)
                if wer_v == wer_v:
                    wers.append(wer_v)
            except Exception as e:
                err = str(e)[:500]

            writer.writerow(
                {
                    "wav_path": wav_path,
                    "utt_id": utt_id,
                    "speaker": spk,
                    "wer": f"{wer_v:.6f}" if wer_v == wer_v else "",
                    "hypothesis": hyp[:2000] if hyp else "",
                    "error": err,
                }
            )
            fcsv.flush()

    if wers:
        print(f"[whisper_wer] mean WER: {sum(wers) / len(wers):.4f} (n={len(wers)})")
    print(f"[whisper_wer] wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
