"""Pretrain the Causal Masked Prosody Model on LibriTTS audio.

Training setup follows Wallbridge et al. (Interspeech 2025):
  - 16 Conformer layers, D=256, codebook c=128
  - "random masking": m ~ U(1,128) per batch, ~50% coverage
  - CE loss normalised by 1/log(c)
  - LibriTTS, batch 256, 10k steps, max 6s utterances

Usage:
    CUDA_VISIBLE_DEVICES=2 python src/scripts/train_mpm.py \
        --data_dir /mnt/data/disk2/LibriTTS \
        --output_dir /mnt/data/disk2/yejin/LiveVoice/checkpoints/mpm \
        --steps 10000 --batch_size 256 --lr 3e-4

The model learns to predict masked pitch/energy/VAD from causal context.
After pretraining, the intermediate hidden state (8th layer) is used as a
prosody latent inside LiveVoice (per-layer additive conditioning).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from livevoice.model.causal_mpm import CausalMPM, CausalMPMConfig


# ---------------------------------------------------------------------------
# Dataset: just loads audio files, random crop to fixed length
# ---------------------------------------------------------------------------
class AudioOnlyDataset(Dataset):
    """Glob .wav / .flac from LibriTTS splits, return fixed-length crops."""

    def __init__(self, root: str, splits: tuple[str, ...], sr: int, duration: float):
        self.sr = sr
        self.target_len = int(sr * duration)
        self.files: list[str] = []
        root = Path(root)
        for split in splits:
            d = root / split
            if not d.exists():
                print(f"[mpm data] WARNING: {d} does not exist, skipping")
                continue
            self.files.extend(sorted(str(f) for f in d.rglob("*.wav")))
            self.files.extend(sorted(str(f) for f in d.rglob("*.flac")))
        print(f"[mpm data] {len(self.files)} files from {splits}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        audio, sr = torchaudio.load(self.files[idx])
        if audio.size(0) > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        audio = audio.squeeze(0)
        # random crop or pad
        if audio.size(0) >= self.target_len:
            start = random.randint(0, audio.size(0) - self.target_len)
            audio = audio[start:start + self.target_len]
        else:
            audio = F.pad(audio, (0, self.target_len - audio.size(0)))
        return audio


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pretrain Causal MPM")
    parser.add_argument("--data_dir", type=str, default="/mnt/data/disk2/LibriTTS")
    parser.add_argument("--splits", type=str, default="train-clean-100,train-clean-360")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=6.0, help="audio crop seconds")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--filter_size", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mask_prob", type=float, default=0.50)
    parser.add_argument("--mask_strategy", type=str, default="random",
                        choices=["random", "span"])
    parser.add_argument("--output_layer", type=int, default=7,
                        help="0-indexed layer for representation extraction (7 = 8th layer)")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument("--use_bap", action="store_true", help="add BAP (5-band aperiodicity) target")
    parser.add_argument("--use_cpps", action="store_true", help="add CPPS target")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = CausalMPMConfig(
        n_layers=args.n_layers,
        filter_size=args.filter_size,
        n_heads=args.n_heads,
        mask_prob=args.mask_prob,
        mask_strategy=args.mask_strategy,
        output_layer=args.output_layer,
        use_bap=args.use_bap,
        use_cpps=args.use_cpps,
    )

    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    model = CausalMPM(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    extras = []
    if args.use_bap:
        extras.append("bap")
    if args.use_cpps:
        extras.append("cpps")
    extra_str = f", extras=[{','.join(extras)}]" if extras else ""
    print(f"[mpm] {n_params / 1e6:.1f}M parameters, {args.n_layers} layers, "
          f"D={args.filter_size}, heads={args.n_heads}, "
          f"mask={args.mask_strategy} p={args.mask_prob}, "
          f"output_layer={args.output_layer}{extra_str}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"[mpm] resumed from step {start_step}")

    splits = tuple(s.strip() for s in args.splits.split(","))
    dataset = AudioOnlyDataset(args.data_dir, splits, cfg.sample_rate, args.duration)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )

    def lr_schedule(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + __import__("math").cos(progress * __import__("math").pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    for _ in range(start_step):
        scheduler.step()

    model.train()
    step = start_step
    running = {"loss": 0.0, "pitch": 0.0, "energy": 0.0, "vad": 0.0, "mask": 0.0}
    if args.use_bap:
        running["bap"] = 0.0
    if args.use_cpps:
        running["cpps"] = 0.0
    t0 = time.time()

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            audio = batch.cuda()

            out = model.forward_pretrain(audio)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            running["loss"] += loss.item()
            running["pitch"] += out["loss_pitch"].item()
            running["energy"] += out["loss_energy"].item()
            running["vad"] += out["loss_vad"].item()
            running["mask"] += out["mask_ratio"].item()
            if "bap" in running:
                running["bap"] += out.get("loss_bap", torch.tensor(0.0)).item()
            if "cpps" in running:
                running["cpps"] += out.get("loss_cpps", torch.tensor(0.0)).item()

            if step % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                extra_log = ""
                if "bap" in running:
                    extra_log += f"  bap {running['bap']/n:.4f}"
                if "cpps" in running:
                    extra_log += f"  cpps {running['cpps']/n:.4f}"
                print(f"step {step:>6d}/{args.steps}  "
                      f"loss {running['loss']/n:.4f}  "
                      f"pitch {running['pitch']/n:.4f}  "
                      f"energy {running['energy']/n:.4f}  "
                      f"vad {running['vad']/n:.4f}{extra_log}  "
                      f"mask {running['mask']/n:.1%}  "
                      f"lr {lr:.2e}  "
                      f"{elapsed/n:.2f}s/step")
                running = {k: 0.0 for k in running}
                t0 = time.time()

            if step % args.save_every == 0 or step == args.steps:
                ckpt_path = os.path.join(args.output_dir, f"step_{step}.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg.__dict__,
                }, ckpt_path)
                # also save as latest
                latest_path = os.path.join(args.output_dir, "latest.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg.__dict__,
                }, latest_path)
                print(f"[mpm] saved {ckpt_path}")

    print(f"[mpm] training complete: {step} steps")


if __name__ == "__main__":
    main()
