"""Evaluate a pretrained Causal MPM following Wallbridge et al. (Interspeech 2025).

Two evaluation modes:

1. **intrinsic** — masked-prediction accuracy on held-out audio (LibriTTS dev).
   Reports per-feature accuracy and loss. No extra data needed beyond LibriTTS.

2. **emotion**  — RAVDESS utterance-level emotion classification with a linear or
   Conformer probe, 5-fold speaker-stratified CV. Compares MPM representations
   against raw (pitch, energy, VAD) features.

   RAVDESS (Livingstone & Russo 2018): 1,440 utterances, 24 speakers, 8 emotions.
   Download from https://zenodo.org/record/1188976 and unzip so that the structure
   looks like:  <ravdess_dir>/Actor_01/03-01-01-01-01-01-01.wav  ...

Usage:
    # intrinsic (reconstruction)
    python src/scripts/eval_mpm.py intrinsic \
        --checkpoint /path/to/mpm/latest.pt \
        --data_dir /mnt/data/disk2/LibriTTS \
        --splits dev-clean

    # emotion probe (RAVDESS)
    python src/scripts/eval_mpm.py emotion \
        --checkpoint /path/to/mpm/latest.pt \
        --ravdess_dir /path/to/RAVDESS/audio_speech \
        --probe conformer
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from livevoice.model.causal_mpm import CausalMPM, CausalMPMConfig


# ── helpers ──────────────────────────────────────────────────────────
def load_mpm(ckpt_path: str, device: str = "cuda") -> CausalMPM:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    cfg = CausalMPMConfig(**cfg_dict)
    model = CausalMPM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    step = ckpt.get("step", "?")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] loaded MPM step={step}  {n_params/1e6:.1f}M params  "
          f"layers={cfg.n_layers}  D={cfg.filter_size}  "
          f"mask={cfg.mask_strategy}  output_layer={cfg.output_layer}")
    return model


# =====================================================================
# 1. Intrinsic evaluation — masked reconstruction
# =====================================================================
class AudioDataset(Dataset):
    def __init__(self, root: str, splits: tuple[str, ...], sr: int, max_dur: float):
        self.sr = sr
        self.max_len = int(sr * max_dur)
        self.files: list[str] = []
        root = Path(root)
        for split in splits:
            d = root / split
            if not d.exists():
                print(f"[eval] WARNING: {d} not found, skipping")
                continue
            self.files.extend(sorted(str(f) for f in d.rglob("*.wav")))
            self.files.extend(sorted(str(f) for f in d.rglob("*.flac")))
        print(f"[eval] {len(self.files)} files from {splits}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.files[idx])
        if audio.size(0) > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        audio = audio.squeeze(0)
        if audio.size(0) > self.max_len:
            audio = audio[:self.max_len]
        return audio


def collate_pad(batch: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(x.size(0) for x in batch)
    return torch.stack([F.pad(x, (0, max_len - x.size(0))) for x in batch])


@torch.no_grad()
def eval_intrinsic(model: CausalMPM, args):
    splits = tuple(s.strip() for s in args.splits.split(","))
    dataset = AudioDataset(args.data_dir, splits, model.cfg.sample_rate, args.max_dur)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_pad)

    total = defaultdict(float)
    n_batches = 0
    n_bins = model.cfg.n_bins

    for batch in loader:
        audio = batch.to(next(model.parameters()).device)
        out = model.forward_pretrain(audio)

        # also compute accuracy
        pitch_bin, energy_bin, vad = model.feature_extractor(audio)
        mask = model._make_mask(audio.size(0), pitch_bin.size(1), audio.device)

        h_in = (model.pitch_emb(torch.where(mask, n_bins, pitch_bin))
                + model.energy_emb(torch.where(mask, n_bins, energy_bin))
                + model.vad_emb(torch.where(mask, 2, vad)))
        h_in = model.pe(h_in)
        h_out, _ = model.encoder(h_in)

        pred_p = model.head_pitch(h_out).argmax(-1)
        pred_e = model.head_energy(h_out).argmax(-1)
        pred_v = (model.head_vad(h_out).squeeze(-1) > 0).long()

        m = mask.float()
        n_m = m.sum().clamp_min(1)
        total["loss"] += out["loss"].item()
        total["loss_pitch"] += out["loss_pitch"].item()
        total["loss_energy"] += out["loss_energy"].item()
        total["loss_vad"] += out["loss_vad"].item()
        total["acc_pitch"] += ((pred_p == pitch_bin).float() * m).sum().item() / n_m.item()
        total["acc_energy"] += ((pred_e == energy_bin).float() * m).sum().item() / n_m.item()
        total["acc_vad"] += ((pred_v == vad).float() * m).sum().item() / n_m.item()
        total["mask_ratio"] += out["mask_ratio"].item()
        n_batches += 1

    print(f"\n{'='*60}")
    print(f"Intrinsic evaluation  ({n_batches} batches, {len(dataset)} files)")
    print(f"{'='*60}")
    for k in ["loss", "loss_pitch", "loss_energy", "loss_vad"]:
        print(f"  {k:15s} {total[k]/n_batches:.4f}")
    print(f"  {'mask_ratio':15s} {total['mask_ratio']/n_batches:.1%}")
    print()
    for k in ["acc_pitch", "acc_energy", "acc_vad"]:
        print(f"  {k:15s} {total[k]/n_batches:.1%}")
    chance = 1.0 / n_bins
    print(f"  {'chance (1/c)':15s} {chance:.1%}")
    print(f"{'='*60}\n")


# =====================================================================
# 2. RAVDESS emotion classification probe
# =====================================================================
RAVDESS_EMOTIONS = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}


class RavdessDataset(Dataset):
    """Loads RAVDESS audio_speech; parses emotion & actor from filename."""

    def __init__(self, root: str, sr: int = 16000):
        self.sr = sr
        self.items: list[dict] = []
        root = Path(root)
        # Only scan Actor_* dirs directly under root to avoid duplicates
        # (some downloads have a nested audio_speech_actors_01-24/ copy).
        seen: set[str] = set()
        for wav in sorted(root.rglob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) != 7:
                continue
            key = wav.stem
            if key in seen:
                continue
            seen.add(key)
            modality, _, emotion, _, _, _, actor = [int(p) for p in parts]
            if modality != 3:  # audio-only speech
                continue
            self.items.append({
                "path": str(wav),
                "emotion": emotion - 1,  # 0-indexed
                "actor": actor,
            })
        actors = sorted(set(d["actor"] for d in self.items))
        print(f"[ravdess] {len(self.items)} utterances, "
              f"{len(actors)} actors, 8 emotions")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        audio, sr = torchaudio.load(item["path"])
        if audio.size(0) > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        return audio.squeeze(0), item["emotion"], item["actor"]


def collate_ravdess(batch):
    audios, emotions, actors = zip(*batch)
    max_len = max(a.size(0) for a in audios)
    padded = torch.stack([F.pad(a, (0, max_len - a.size(0))) for a in audios])
    lengths = torch.tensor([a.size(0) for a in audios])
    return padded, torch.tensor(emotions), torch.tensor(actors), lengths


class ConformerProbe(nn.Module):
    """2-layer Conformer + mean/max pool → classifier (paper's stronger probe)."""

    def __init__(self, d_in: int, n_classes: int, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(d_model * 2, n_classes)  # mean + max

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.encoder(x)
        # mean + max pooling (mask padding)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        x_mean = (x * mask_f).sum(1) / mask_f.sum(1).clamp_min(1)
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        x_max = x_masked.max(1).values
        x_max = x_max.masked_fill(torch.isinf(x_max), 0.0)
        return self.head(torch.cat([x_mean, x_max], dim=-1))


class LinearProbe(nn.Module):
    """Mean+max pool → linear classifier."""

    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.head = nn.Linear(d_in * 2, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        x_mean = (x * mask_f).sum(1) / mask_f.sum(1).clamp_min(1)
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        x_max = x_masked.max(1).values
        x_max = x_max.masked_fill(torch.isinf(x_max), 0.0)
        return self.head(torch.cat([x_mean, x_max], dim=-1))


@torch.no_grad()
def extract_features(model: CausalMPM, audios: torch.Tensor,
                     mode: str = "mpm") -> torch.Tensor:
    """Extract frame-level features: 'mpm' = SSL hidden, 'raw' = pitch/energy/vad."""
    pitch_bin, energy_bin, vad = model.feature_extractor(audios)
    if mode == "raw":
        T = pitch_bin.size(1)
        n_bins = model.cfg.n_bins
        # one-hot-ish normalised features (same dim as paper's raw baseline)
        pitch_f = pitch_bin.float() / n_bins
        energy_f = energy_bin.float() / n_bins
        vad_f = vad.float()
        return torch.stack([pitch_f, energy_f, vad_f], dim=-1)  # (B, T, 3)

    h = model.pitch_emb(pitch_bin) + model.energy_emb(energy_bin) + model.vad_emb(vad)
    h = model.pe(h)
    _, intermediate = model.encoder(h)
    return intermediate  # (B, T, D)


def train_probe(probe, feats_all, labels_all, lengths_all, train_idx, val_idx,
                steps=1000, batch_size=32, lr=4e-5, warmup=100, device="cuda"):
    """Train probe on train split, evaluate on val split. Returns (WA, UA)."""
    probe = probe.to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)

    def lr_fn(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0, 1.0 - (step - warmup) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    train_f = feats_all[train_idx].to(device)
    train_l = labels_all[train_idx].to(device)
    train_len = lengths_all[train_idx].to(device)
    n_train = len(train_idx)

    probe.train()
    for step in range(steps):
        idx = torch.randint(0, n_train, (min(batch_size, n_train),), device="cpu")
        logits = probe(train_f[idx], train_len[idx])
        loss = F.cross_entropy(logits, train_l[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    # evaluate
    probe.eval()
    val_f = feats_all[val_idx].to(device)
    val_l = labels_all[val_idx].to(device)
    val_len = lengths_all[val_idx].to(device)
    with torch.no_grad():
        logits = probe(val_f, val_len)
    preds = logits.argmax(-1)
    targets = val_l

    wa = (preds == targets).float().mean().item()
    n_classes = logits.size(-1)
    per_class_acc = []
    for c in range(n_classes):
        mask_c = targets == c
        if mask_c.sum() > 0:
            per_class_acc.append((preds[mask_c] == c).float().mean().item())
    ua = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0
    return wa, ua


def eval_emotion(model: CausalMPM, args):
    device = next(model.parameters()).device
    dataset = RavdessDataset(args.ravdess_dir, model.cfg.sample_rate)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_ravdess,
                        num_workers=2)

    n_classes = 8
    n_folds = 5

    # extract features for both modes
    results = {}
    for mode in ["mpm", "raw"]:
        print(f"\n--- Extracting {mode} features ---")
        all_feats, all_emotions, all_actors, all_lengths = [], [], [], []
        for audios_b, emo_b, act_b, len_b in loader:
            audios_b = audios_b.to(device)
            feats_b = extract_features(model, audios_b, mode=mode)
            hop = model.cfg.hop_length
            flen_b = (len_b.float() / hop).long().clamp(max=feats_b.size(1))
            all_feats.append(feats_b.cpu())
            all_emotions.append(emo_b)
            all_actors.append(act_b)
            all_lengths.append(flen_b)

        # pad all to same T and stack
        max_T = max(f.size(1) for f in all_feats)
        d_in = all_feats[0].size(-1)
        padded = []
        for f in all_feats:
            if f.size(1) < max_T:
                f = F.pad(f, (0, 0, 0, max_T - f.size(1)))
            padded.append(f)
        feats_cpu = torch.cat(padded, dim=0)
        emotions = torch.cat(all_emotions, dim=0)
        actors = torch.cat(all_actors, dim=0)
        feat_lengths_cpu = torch.cat(all_lengths, dim=0)
        actor_ids = sorted(set(actors.tolist()))
        print(f"  features: {feats_cpu.shape}, d_in={d_in}")

        # 5-fold speaker-stratified CV
        # split actors into 5 roughly equal groups
        fold_actors = [actor_ids[i::n_folds] for i in range(n_folds)]

        fold_wa, fold_ua = [], []
        for fold in range(n_folds):
            val_actors = set(fold_actors[fold])
            val_idx = [i for i in range(len(dataset)) if actors[i].item() in val_actors]
            train_idx = [i for i in range(len(dataset)) if actors[i].item() not in val_actors]

            if args.probe == "conformer":
                probe = ConformerProbe(d_in, n_classes)
            else:
                probe = LinearProbe(d_in, n_classes)

            wa, ua = train_probe(
                probe, feats_cpu, emotions, feat_lengths_cpu,
                train_idx, val_idx,
                steps=args.probe_steps, batch_size=args.probe_batch_size,
                lr=args.probe_lr, warmup=args.probe_warmup, device=str(device),
            )
            fold_wa.append(wa)
            fold_ua.append(ua)
            print(f"  fold {fold+1}/{n_folds}: WA={wa:.3f}  UA={ua:.3f}  "
                  f"(train={len(train_idx)}, val={len(val_idx)})")

        avg_wa = sum(fold_wa) / n_folds
        avg_ua = sum(fold_ua) / n_folds
        results[mode] = (avg_wa, avg_ua)
        print(f"  → {mode} avg: WA={avg_wa:.3f}  UA={avg_ua:.3f}")

    print(f"\n{'='*60}")
    print(f"RAVDESS Emotion Classification ({args.probe} probe, {n_folds}-fold CV)")
    print(f"{'='*60}")
    print(f"  {'Feature':15s}  {'WA':>6s}  {'UA':>6s}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*6}")
    for mode, (wa, ua) in results.items():
        print(f"  {mode:15s}  {wa:6.3f}  {ua:6.3f}")
    print()
    print(f"  Paper ref (linear):    MPM random  WA=0.24  UA=0.23")
    print(f"  Paper ref (conformer): MPM random  WA=0.37  UA=0.36")
    print(f"  Paper ref (raw P,E,V): linear      WA=0.10  UA=0.09")
    print(f"{'='*60}\n")


# =====================================================================
# main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained Causal MPM")
    sub = parser.add_subparsers(dest="task", required=True)

    # intrinsic
    p_intr = sub.add_parser("intrinsic", help="Masked reconstruction accuracy")
    p_intr.add_argument("--checkpoint", type=str, required=True)
    p_intr.add_argument("--data_dir", type=str, default="/mnt/data/disk2/LibriTTS")
    p_intr.add_argument("--splits", type=str, default="dev-clean")
    p_intr.add_argument("--batch_size", type=int, default=32)
    p_intr.add_argument("--max_dur", type=float, default=10.0)
    p_intr.add_argument("--device", type=str, default="cuda")

    # emotion (RAVDESS)
    p_emo = sub.add_parser("emotion", help="RAVDESS emotion classification probe")
    p_emo.add_argument("--checkpoint", type=str, required=True)
    p_emo.add_argument("--ravdess_dir", type=str, required=True,
                       help="Path to RAVDESS audio_speech directory")
    p_emo.add_argument("--probe", type=str, default="linear",
                       choices=["linear", "conformer"])
    p_emo.add_argument("--probe_steps", type=int, default=1000)
    p_emo.add_argument("--probe_batch_size", type=int, default=32)
    p_emo.add_argument("--probe_lr", type=float, default=4e-5)
    p_emo.add_argument("--probe_warmup", type=int, default=100)
    p_emo.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    model = load_mpm(args.checkpoint, device=args.device)

    if args.task == "intrinsic":
        eval_intrinsic(model, args)
    elif args.task == "emotion":
        eval_emotion(model, args)


if __name__ == "__main__":
    main()
