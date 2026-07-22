"""Offline SW2V (jhcodec streaming-wav2vec) feature extraction with source-side perturbation.

Mirrors extract_features.py (HuBERT) but for content_source="sw2v": pre-extracts the
jhcodec AudioEncoder's continuous 1024-d features (50 fps, 320-sample non-overlapping
blocks — the SAME grid as the jhcodec codec tokens, so cache frame i ↔ codec token i).

Source-side perturbation (pitch ±N semitones + 4-band EQ) is baked into the cache exactly
like the HuBERT path, to de-identify content before the encoder. ONE perturbed version per
utterance (the perturbation becomes fixed per utterance across epochs — same tradeoff as
the HuBERT cache).

The features are stored CONTINUOUS (float16). Any information bottleneck (e.g. FSQ) is
applied later *inside the model* on top of a learnable down-projection, NOT baked here —
so the same cache serves every FSQ level/dim without re-extraction.

Saved file format (per utterance), identical to the HuBERT cache so LibriTTSDataset._load_feats
reads it unchanged:
    { "feats": FloatTensor(T, 1024) as float16, "audio_stride": int }

Usage (conda `sound`):
    python scripts/extract_sw2v_features.py libritts \
        --libritts_path /mnt/data/disk2/LibriTTS \
        --splits train-clean-100,train-clean-360,dev-clean \
        --out_dir /mnt/data/disk2/yejin/LiveVoice/features/perturbed/sw2v/libritts \
        --device cuda --batch_size 16
"""
import argparse
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import torchaudio.functional as AF
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.model.content_perturbation import ContentPerturbation
from livevoice.model.sw2v_content import Sw2vContentEncoder

SW2V_SR = 16_000
HOP = 320  # samples per frame at 16 kHz → 50 fps (== jhcodec codec hop)


def load_mono(path: str, target_sr: int = SW2V_SR) -> torch.Tensor:
    """Load wav as mono float32, resampled to target_sr, peak-normalized."""
    try:
        import soundfile as sf
        with sf.SoundFile(path) as f:
            audio_np = f.read(dtype="float32", always_2d=True)
            sr = f.samplerate
        audio = torch.from_numpy(audio_np).mean(dim=1)
    except Exception:
        audio, sr = torchaudio.load(path)
        audio = audio.float().mean(0)
        sr = int(sr)
    if sr != target_sr:
        audio = AF.resample(audio, sr, target_sr)
    audio = audio / (audio.abs().max() + 1e-8)
    return audio


class Sw2vBatchEncoder:
    """Batch-encode audio through the frozen jhcodec AudioEncoder → (ceil(L/320), 1024)."""

    def __init__(self, config, device: str):
        self.device = device
        self.encoder = Sw2vContentEncoder(config).to(device).eval()
        self.out_dim = int(self.encoder.out_dim)

    @torch.no_grad()
    def encode_batch(self, audios: list[torch.Tensor]) -> list[torch.Tensor]:
        """audios: list of (T,) at 16 kHz → list of (ceil(T/320), 1024).

        End-pads the batch to a common multiple of HOP and runs once; since the encoder
        is causal + non-overlapping-block, end-padding does not alter earlier valid frames,
        so batched == per-sample for the kept frames."""
        ns = [max(1, math.ceil(a.shape[0] / HOP)) for a in audios]
        max_len = max(n * HOP for n in ns)
        padded = torch.zeros(len(audios), max_len, device=self.device)
        for i, a in enumerate(audios):
            padded[i, : a.shape[0]] = a.to(self.device)
        feats = self.encoder(padded)  # (B, max_len/HOP, 1024)
        return [feats[i, : ns[i]].cpu() for i in range(len(audios))]


def discover_libritts(libritts_path: str, splits: list[str]) -> list[tuple[str, str, str]]:
    root = Path(libritts_path)
    items = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [warn] split not found: {split_dir}")
            continue
        for wav in sorted(split_dir.glob("**/*.wav")):
            speaker_id = wav.parts[-3]
            items.append((str(wav), speaker_id, wav.stem))
    return items


def extract(items, out_dir, encoder, perturber, training_sr, batch_size, skip_existing):
    out_root = Path(out_dir)
    audio_stride = HOP * training_sr // SW2V_SR

    todo = []
    for wav_path, spk, utt_id in items:
        save_path = out_root / spk / f"{utt_id}.pt"
        if skip_existing and save_path.exists():
            continue
        todo.append((wav_path, spk, utt_id, save_path))
    print(f"  {len(todo)} files to process ({len(items) - len(todo)} already done)")

    for batch_start in tqdm(range(0, len(todo), batch_size), desc="extracting"):
        batch = todo[batch_start : batch_start + batch_size]
        audios = []
        for wav_path, spk, utt_id, save_path in batch:
            try:
                a = load_mono(wav_path, SW2V_SR)
                a = perturber.forward(a.unsqueeze(0)).squeeze(0)  # pitch + EQ (CPU)
                audios.append(a)
            except Exception as e:
                print(f"  [warn] failed {wav_path}: {e}")
                audios.append(None)

        valid_idx = [i for i, a in enumerate(audios) if a is not None]
        valid = [audios[i] for i in valid_idx]
        if not valid:
            continue
        feats_list = encoder.encode_batch(valid)  # list of (T, 1024)
        for rel_i, feat in zip(valid_idx, feats_list):
            _, spk, utt_id, save_path = batch[rel_i]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"feats": feat.half(), "audio_stride": audio_stride}, save_path)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="dataset", required=True)
    pl = sub.add_parser("libritts")
    pl.add_argument("--libritts_path", default="/mnt/data/disk2/LibriTTS")
    pl.add_argument("--splits", default="train-clean-100,train-clean-360,dev-clean")
    pl.add_argument("--out_dir", default="/mnt/data/disk2/yejin/LiveVoice/features/sw2v/libritts")
    pl.add_argument("--training_sr", type=int, default=16000,
                    help="training sample rate (jhcodec=16000) → audio_stride")
    pl.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pl.add_argument("--batch_size", type=int, default=16)
    pl.add_argument("--sw2v_ckpt", default="/mnt/data/disk3/yejin/jhcodec/sw2v_120000.pt")
    pl.add_argument("--sw2v_config", default="/workspace/jhcodec/config/config_w2vcossim.json")
    pl.add_argument("--no_skip_existing", action="store_true")
    pl.add_argument("--seed", type=int, default=42)
    pl.add_argument("--shard", type=int, default=0,
                    help="This process's shard index in [0, num_shards). Run one per GPU; "
                         "all shards write to the same out_dir so the cache merges automatically.")
    pl.add_argument("--num_shards", type=int, default=1,
                    help="Total number of parallel shards (e.g. 3 for GPUs 5,6,7).")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Config: 16 kHz so ContentPerturbation + sw2v agree; perturbation values come from
    # the dataclass defaults (pitch ±4, EQ ±6 dB) — same knobs as the HuBERT cache.
    config = LiveVoiceConfig(sample_rate=SW2V_SR, sw2v_sample_rate=SW2V_SR,
                             sw2v_ckpt=args.sw2v_ckpt, sw2v_config=args.sw2v_config)

    print(f"[extract-sw2v] loading AudioEncoder ({args.sw2v_ckpt}) on {args.device} ...")
    encoder = Sw2vBatchEncoder(config, args.device)
    print(f"[extract-sw2v] out_dim={encoder.out_dim}  "
          f"perturb: pitch±{config.perturb_pitch_semitones} EQ±{config.perturb_eq_gain_db}dB "
          f"vtln={config.use_vtln}")

    perturber = ContentPerturbation(config)
    perturber.train()  # enable perturbation

    splits = [s.strip() for s in args.splits.split(",")]
    print(f"[extract-sw2v] discovering LibriTTS ({splits}) at {args.libritts_path} ...")
    items = discover_libritts(args.libritts_path, splits)
    if args.num_shards > 1:
        if not (0 <= args.shard < args.num_shards):
            raise SystemExit(f"--shard must be in [0,{args.num_shards})")
        total = len(items)
        items = items[args.shard :: args.num_shards]   # strided → balanced
        print(f"[extract-sw2v] shard {args.shard}/{args.num_shards}: "
              f"{len(items)}/{total} utterances")
    print(f"[extract-sw2v] {len(items)} utterances → {args.out_dir}")

    extract(items, out_dir=args.out_dir, encoder=encoder, perturber=perturber,
            training_sr=args.training_sr, batch_size=args.batch_size,
            skip_existing=not args.no_skip_existing)
    print("[extract-sw2v] Done.")


if __name__ == "__main__":
    main()
