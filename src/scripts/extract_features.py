"""Offline HuBERT feature extraction with source-side perturbation.

Pre-extracts HuBERT layer-9 hidden states for every utterance and saves to disk.
During training the model loads these directly, skipping live HuBERT inference
and per-sample Python-loop perturbation — the two main training bottlenecks.

Saves one perturbed version per utterance.  Feature files are stored as float16
to keep disk usage reasonable (~384 KB per 5-second utterance).

Saved file format (per utterance):
    {
        "feats":        FloatTensor(T, 768) as float16  — HuBERT layer-9 hidden states
        "audio_stride": int                             — TRAINING-SR samples per HuBERT frame
    }
    where audio_stride = 320 × (training_sr / 16000).
    e.g. training at 24 kHz → 480, training at 44.1 kHz → 882

Usage:
    # VCTK
    python scripts/extract_features.py vctk \
        --vctk_path /mnt/data/disk2/VCTK-Corpus \
        --out_dir /mnt/data/disk2/yejin/features/vctk \
        --device cuda --batch_size 16

    # LibriTTS (24 kHz)
    python scripts/extract_features.py libritts \
        --libritts_path /mnt/data/disk2/LibriTTS \
        --splits train-clean-100,train-clean-360,train-other-500,dev-clean,dev-other \
        --out_dir /mnt/data/disk2/yejin/LiveVoice/features/libritts \
        --device cuda --batch_size 16
"""
import argparse
import glob
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

HUBERT_SR = 16_000
HUBERT_HOP = 320  # HuBERT-base hop at 16 kHz → 50 fps


# ─────────────────────────────────────────────────────────────────────
#  Audio loading
# ─────────────────────────────────────────────────────────────────────

def load_mono(path: str, target_sr: int = HUBERT_SR) -> torch.Tensor:
    """Load wav as mono float32, resampled to target_sr."""
    try:
        import soundfile as sf
        import numpy as np
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


# ─────────────────────────────────────────────────────────────────────
#  HuBERT batch encoder
# ─────────────────────────────────────────────────────────────────────

class HuBERTBatchEncoder:
    """Thin wrapper to batch-encode audio through HuBERT and extract layer 9."""

    def __init__(self, model_name: str, layer: int, device: str):
        from transformers import HubertModel
        self.model = HubertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.layer = layer
        self.device = device

        # Receptive field + stride of the HuBERT conv extractor (valid convs).
        # hubert-base: RF=400, stride=320. front pad = (RF-hop)/2 = 40 aligns HuBERT
        # frame centers onto the codec token grid (token t center = 320t+159.5); frame
        # count is trimmed to ceil(L/hop) to match the codec's own tokenization.
        # MUST mirror HuBERTContentExtractor._extract_hidden so cached feats == online.
        ks = list(self.model.config.conv_kernel)
        ss = list(self.model.config.conv_stride)
        rf, jump = 1, 1
        for k, s in zip(ks, ss):
            rf += (k - 1) * jump
            jump *= s
        self.rf = int(rf)
        self.stride = int(jump)
        self.front = max(0, (self.rf - HUBERT_HOP) // 2)

    @torch.no_grad()
    def encode_batch(self, audios: list[torch.Tensor]) -> list[torch.Tensor]:
        """audios: list of (T,) tensors at 16 kHz → list of (ceil(T/hop), 768) tensors,
        center-aligned to the codec token grid (front-pad + ceil-count truncate)."""
        front = self.front
        # per-item target frame count = codec grid = ceil(L/hop)
        ns = [max(1, math.ceil(a.shape[0] / HUBERT_HOP)) for a in audios]
        # required padded length so item i yields >= ns[i] valid-conv frames
        needs = [front + self.stride * (n - 1) + self.rf for n in ns]
        max_len = max(needs) + self.stride  # 1-frame margin against flooring
        padded = torch.zeros(len(audios), max_len, device=self.device)
        for i, a in enumerate(audios):
            padded[i, front : front + a.shape[0]] = a.to(self.device)

        out = self.model(padded, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[self.layer]  # (B, T_frames, 768)

        results = []
        for i, n in enumerate(ns):
            h = hidden[i]
            if h.shape[0] < n:  # defensive; margin should prevent this
                h = torch.cat([h, h[-1:].expand(n - h.shape[0], -1)], dim=0)
            results.append(h[:n].cpu())
        return results


# ─────────────────────────────────────────────────────────────────────
#  File discovery
# ─────────────────────────────────────────────────────────────────────

def discover_vctk(vctk_path: str) -> list[tuple[str, str, str]]:
    """Returns [(wav_path, speaker_id, utt_id), ...]"""
    wav_root = Path(vctk_path) / "wav48"
    items = []
    for spk_dir in sorted(wav_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for wav in sorted(spk_dir.glob("*.wav")):
            utt_id = wav.stem
            items.append((str(wav), spk, utt_id))
    return items


def discover_libritts(libritts_path: str, splits: list[str]) -> list[tuple[str, str, str]]:
    """Returns [(wav_path, speaker_id, utt_id), ...]"""
    root = Path(libritts_path)
    items = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [warn] split not found: {split_dir}")
            continue
        for wav in sorted(split_dir.glob("**/*.wav")):
            # e.g. train-clean-100/19/198/19_198_000000_000000.wav
            speaker_id = wav.parts[-3]  # grandparent dir
            utt_id = wav.stem
            items.append((str(wav), speaker_id, utt_id))
    return items


# ─────────────────────────────────────────────────────────────────────
#  Main extraction loop
# ─────────────────────────────────────────────────────────────────────

def extract(items, out_dir, encoder, perturber, training_sr, batch_size, skip_existing):
    """Extract and save features for all items."""
    out_root = Path(out_dir)
    audio_stride = HUBERT_HOP * training_sr // HUBERT_SR  # training-SR samples per HuBERT frame

    # Collect work (skip existing if requested)
    todo = []
    for wav_path, spk, utt_id in items:
        save_path = out_root / spk / f"{utt_id}.pt"
        if skip_existing and save_path.exists():
            continue
        todo.append((wav_path, spk, utt_id, save_path))

    print(f"  {len(todo)} files to process ({len(items) - len(todo)} already done)")

    # Process in batches
    for batch_start in tqdm(range(0, len(todo), batch_size), desc="extracting"):
        batch = todo[batch_start : batch_start + batch_size]
        audios_16k = []
        for wav_path, spk, utt_id, save_path in batch:
            try:
                # Load and resample to 16 kHz for HuBERT
                audio_16k = load_mono(wav_path, HUBERT_SR)
                # Apply perturbation (CPU; perturber is in training mode)
                audio_16k = perturber.forward(audio_16k.unsqueeze(0)).squeeze(0)
                audios_16k.append(audio_16k)
            except Exception as e:
                print(f"  [warn] failed to load {wav_path}: {e}")
                audios_16k.append(None)

        # Filter valid
        valid_idx = [i for i, a in enumerate(audios_16k) if a is not None]
        valid_audios = [audios_16k[i] for i in valid_idx]

        if not valid_audios:
            continue

        feats_list = encoder.encode_batch(valid_audios)  # list of (T, 768)

        for rel_i, feat in zip(valid_idx, feats_list):
            _, spk, utt_id, save_path = batch[rel_i]
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "feats": feat.half(),          # (T, 768) float16
                    "audio_stride": audio_stride,  # source-sr samples per HuBERT frame
                },
                save_path,
            )


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="dataset", required=True)

    pv = sub.add_parser("vctk")
    pv.add_argument("--vctk_path", default="/mnt/data/disk2/VCTK-Corpus")
    pv.add_argument("--out_dir", default="/mnt/data/disk2/yejin/features/vctk")
    pv.add_argument("--training_sr", type=int, default=16000,
                    help="training sample rate — determines audio_stride (e.g. 16000 for DAC 16kHz)")

    pl = sub.add_parser("libritts")
    pl.add_argument("--libritts_path", default="/mnt/data/disk2/LibriTTS")
    pl.add_argument("--splits", default="train-clean-100,train-clean-360,dev-clean")
    pl.add_argument("--out_dir", default="/mnt/data/disk2/yejin/LiveVoice/features/libritts-perturbed")
    pl.add_argument("--training_sr", type=int, default=16000,
                    help="training sample rate — determines audio_stride")

    for sp in [pv, pl]:
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sp.add_argument("--batch_size", type=int, default=16)
        sp.add_argument("--hubert_model", default="facebook/hubert-base-ls960")
        sp.add_argument("--hubert_layer", type=int, default=9)
        sp.add_argument("--no_skip_existing", action="store_true")
        sp.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[extract] Loading HuBERT ({args.hubert_model}, layer {args.hubert_layer}) on {args.device}...")
    encoder = HuBERTBatchEncoder(args.hubert_model, args.hubert_layer, args.device)

    # Build a minimal config for ContentPerturbation defaults
    config = LiveVoiceConfig(sample_rate=HUBERT_SR)
    perturber = ContentPerturbation(config)
    perturber.train()  # enable perturbation

    if args.dataset == "vctk":
        print(f"[extract] Discovering VCTK files at {args.vctk_path}...")
        items = discover_vctk(args.vctk_path)
    else:
        splits = [s.strip() for s in args.splits.split(",")]
        print(f"[extract] Discovering LibriTTS ({splits}) at {args.libritts_path}...")
        items = discover_libritts(args.libritts_path, splits)

    print(f"[extract] {len(items)} utterances → {args.out_dir}  (audio_stride for {args.training_sr} Hz training)")
    extract(
        items,
        out_dir=args.out_dir,
        encoder=encoder,
        perturber=perturber,
        training_sr=args.training_sr,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
    )
    print("[extract] Done.")


if __name__ == "__main__":
    main()
