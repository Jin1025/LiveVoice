"""Evaluate WER on LibriTTS test-clean with a LiveVoice checkpoint (full utterance).

For each utterance:
  - load full content + reference wav (same-speaker pairing as LibriTTSDataset val),
  - run VC generation,
  - Whisper transcribe generated audio,
  - word-level WER vs the utterance's ``*.normalized.txt``.

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_libritts_test_clean_wer.py \\
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/libritts_mimi_resume2/step_latest.ckpt \\
        --codec mimi --n_codebooks 8 --content_source hubert \\
        --hidden_dim 512 --num_decoder_layers 8 \\
        --out_csv /mnt/data/disk2/yejin/LiveVoice/wer_test_clean_resume2.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.data.libritts_dataset import LibriTTSDataset
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.model import HuBERTContentExtractor, LiveVoiceModel, build_codec


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
        features_dir=None,
        output_dir=args.output_dir,
    )


def _load_done_paths(csv_path: str) -> set[str]:
    if not os.path.isfile(csv_path):
        return set()
    done = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = row.get("wav_path", "").strip()
            if p:
                done.add(p)
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/checkpoints/libritts_mimi_resume2/step_latest.ckpt",
    )
    p.add_argument("--libritts_path", type=str, default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--split_dir", type=str, default="test-clean", help="Subdir under libritts_path")
    p.add_argument("--codec", type=str, default="mimi", choices=["dac", "mimi"])
    p.add_argument("--n_codebooks", type=int, default=8)
    p.add_argument("--dac_model_type", type=str, default="16khz", choices=["16khz", "24khz", "44khz"])
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_decoder_layers", type=int, default=8)
    p.add_argument(
        "--content_source",
        type=str,
        default="hubert",
        choices=["hubert", "mimi_semantic"],
        help="Must match the checkpoint training setting.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--whisper_model", type=str, default="base")
    p.add_argument("--whisper_device", type=str, default="cuda")
    p.add_argument("--cpu", action="store_true", help="Force CPU for the LiveVoice model")
    p.add_argument("--max_items", type=int, default=None, help="Debug: cap number of utterances")
    p.add_argument(
        "--out_csv",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/wer_libritts_test_clean.csv",
    )
    p.add_argument("--resume", action="store_true", help="Skip wav_path rows already present in out_csv")
    p.add_argument("--output_dir", type=str, default="/mnt/data/disk2/yejin/LiveVoice")
    args = p.parse_args()

    cfg_model = _build_model_config(args)
    if cfg_model.content_source == "mimi_semantic" and cfg_model.codec != "mimi":
        raise SystemExit("content_source=mimi_semantic requires --codec mimi")

    print(
        f"[eval] ckpt={args.ckpt}\n"
        f"       codec={cfg_model.codec} sr={cfg_model.sample_rate} "
        f"hidden_dim={cfg_model.hidden_dim} num_decoder_layers={cfg_model.num_decoder_layers} "
        f"n_codebooks_predict={cfg_model.n_codebooks_predict} content_source={cfg_model.content_source}"
    )

    # Dataset index (same pairing rules as training val)
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

    wers: list[float] = []
    with open(args.out_csv, csv_mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=["wav_path", "utt_id", "speaker", "wer", "error"],
        )
        if write_header:
            writer.writeheader()

        for wav_path, utt_id, spk in tqdm(ds.items, desc="test-clean WER"):
            if wav_path in done:
                continue
            err = ""
            wer_v = float("nan")
            try:
                ref_txt = _read_libritts_normalized_text(wav_path)
                if not ref_txt:
                    err = "no_normalized_txt"
                    writer.writerow(
                        {"wav_path": wav_path, "utt_id": utt_id, "speaker": spk, "wer": "", "error": err}
                    )
                    fcsv.flush()
                    continue

                ref_path = _libritts_pick_ref_path(ds, wav_path, utt_id, spk)
                ctn = _load_full_mono_wav(wav_path, target_sr).unsqueeze(0).to(dev)
                ref = _load_full_mono_wav(ref_path, target_sr).unsqueeze(0).to(dev)

                with torch.no_grad():
                    codes = lit.model.generate(
                        reference_audio=ref,
                        content_audio=ctn,
                        temperature=float(args.temperature),
                        top_p=float(args.top_p) if args.top_p > 0 else None,
                        top_k=int(args.top_k) if args.top_k > 0 else None,
                        cfg_scale=float(args.cfg_scale),
                    )
                    gen = lit.model.decode_to_audio(codes)

                wavs = gen.detach().float().cpu()
                if target_sr != 16000:
                    wavs = torchaudio.functional.resample(wavs, target_sr, 16000)
                hyp = w_model.transcribe(wavs[0].numpy(), fp16=False)["text"]
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
                    "error": err,
                }
            )
            fcsv.flush()

    if wers:
        print(f"[eval] mean WER over successful items: {sum(wers) / len(wers):.4f} (n={len(wers)})")
    print(f"[eval] wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
