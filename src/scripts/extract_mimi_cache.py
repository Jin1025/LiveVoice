"""Offline mimi cache extraction for LiveVoice.

Precompute and save full-utterance mimi outputs:
  - target_codes: discrete codes (K, T_frames)
  - reference_z:  continuous latent (T_frames, D)

These caches are later sliced in training using start_sample offsets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import torchaudio.functional as AF
import soundfile as sf
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.model import build_codec


def _cache_key(codec: str, sample_rate: int, path: str) -> str:
    import hashlib
    raw = f"{codec.lower()}|sr={sample_rate}|path={path}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_mono(path: str, target_sr: int) -> torch.Tensor:
    try:
        with sf.SoundFile(path) as f:
            audio_np = f.read(dtype="float32", always_2d=True)
            sr = int(f.samplerate)
        audio = torch.from_numpy(audio_np).float().mean(dim=1)
    except Exception:
        wav, sr = torchaudio.load(path)
        audio = wav.float().mean(dim=0)
        sr = int(sr)
    if sr != target_sr:
        audio = AF.resample(audio, sr, target_sr)
    audio = audio / (audio.abs().max() + 1e-8)
    return audio


def discover_vctk(vctk_path: str) -> list[str]:
    root = Path(vctk_path) / "wav48"
    items = []
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir():
            continue
        for wav in sorted(spk_dir.glob("*.wav")):
            items.append(str(wav))
    return items


def discover_libritts(libritts_path: str, splits: list[str]) -> list[str]:
    root = Path(libritts_path)
    items = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[warn] split not found: {split_dir}")
            continue
        for wav in sorted(split_dir.glob("**/*.wav")):
            items.append(str(wav))
    return items


def _speaker_utt_from_path(path: str) -> tuple[str, str]:
    p = Path(path)
    utt = p.stem
    # VCTK: .../wav48/p225/p225_001.wav  -> spk=p225
    # LibriTTS: .../{split}/19/198/19_198_....wav -> spk=19
    spk = p.parent.name
    if p.parent.parent.name.isdigit() and p.parent.name.isdigit():
        spk = p.parent.parent.name
    return spk, utt


@torch.no_grad()
def run_extract(
    wav_paths: list[str],
    cache_dir: str,
    codec_model,
    codec_name: str,
    sample_rate: int,
    skip_existing: bool,
):
    target_dir = Path(cache_dir) / "target_codes"
    refz_dir = Path(cache_dir) / "reference_z"
    manifest_dir = Path(cache_dir) / "manifests"
    target_dir.mkdir(parents=True, exist_ok=True)
    refz_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for p in wav_paths:
        key = _cache_key(codec_name, sample_rate, p)
        code_path = target_dir / f"{key}.pt"
        z_path = refz_dir / f"{key}.pt"
        if skip_existing and code_path.exists() and z_path.exists():
            continue
        todo.append((p, code_path, z_path))

    print(f"[codec-cache] total={len(wav_paths)} todo={len(todo)} skipped={len(wav_paths)-len(todo)}")

    manifest_path = manifest_dir / "index.jsonl"
    manifest_f = open(manifest_path, "a", encoding="utf-8")
    for wav_path, code_path, z_path in tqdm(todo, desc="extract_mimi_cache"):
        try:
            key = code_path.stem
            spk, utt = _speaker_utt_from_path(wav_path)
            audio = _load_mono(wav_path, sample_rate).unsqueeze(0)  # (1, T)
            audio = audio.to(codec_model.device)
            codes, z = codec_model.encode_continuous(audio)
            # save full utterance features on CPU
            torch.save({"path": wav_path, "codes": codes[0].cpu()}, code_path)
            torch.save({"path": wav_path, "z": z[0].cpu()}, z_path)
            manifest_f.write(
                json.dumps(
                    {
                        "key": key,
                        "speaker_id": spk,
                        "utt_id": utt,
                        "wav_path": wav_path,
                        "target_codes_path": str(code_path),
                        "reference_z_path": str(z_path),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        except Exception as e:
            print(f"[warn] failed: {wav_path} ({e})")
    manifest_f.close()
    print(f"[codec-cache] manifest: {manifest_path}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="dataset", required=True)

    pv = sub.add_parser("vctk")
    pv.add_argument("--vctk_path", default="/mnt/data/disk2/VCTK-Corpus")

    pl = sub.add_parser("libritts")
    pl.add_argument("--libritts_path", default="/mnt/data/disk2/LibriTTS")
    pl.add_argument("--splits", default="train-clean-100,train-clean-360,train-other-500,dev-clean")

    p.add_argument("--cache_dir", default="/mnt/data/disk2/yejin/LiveVoice/mimi_precomputed")
    p.add_argument("--codec", choices=["mimi"], default="mimi")
    p.add_argument("--sample_rate", type=int, default=24000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    cfg = LiveVoiceConfig(codec=args.codec, sample_rate=args.sample_rate, device=args.device)
    codec_model = build_codec(cfg)

    if args.dataset == "vctk":
        wav_paths = discover_vctk(args.vctk_path)
    else:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        wav_paths = discover_libritts(args.libritts_path, splits)

    if not wav_paths:
        raise ValueError("No wav files found.")

    run_extract(
        wav_paths=wav_paths,
        cache_dir=args.cache_dir,
        codec_model=codec_model,
        codec_name=args.codec,
        sample_rate=args.sample_rate,
        skip_existing=args.skip_existing,
    )
    print("[codec-cache] done.")


if __name__ == "__main__":
    main()
