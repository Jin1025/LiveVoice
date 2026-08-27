"""Sequence-level emotion classification with raw prosody features on IEMOCAP.

Compares three feature sets via 1-layer GRU + linear head (5-fold stratified CV):
  1) pitch + energy + VAD          3-d per frame
  2) + BAP (5 bands)               8-d per frame
  3) + BAP + CPPS                  9-d per frame

Frame-level features are z-normalized per-utterance (each feature independently),
then the whole sequence goes into a small GRU. This preserves temporal dynamics
that summary stats (mean/std) would destroy.

Usage:
    CUDA_VISIBLE_DEVICES=1 python src/scripts/probe_prosody_features.py \
        --vpc_root /mnt/data/disk3/yejin/VPC24 \
        --dataset IEMOCAP_dev
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, confusion_matrix

SR = 16000
HOP = 320       # 20ms
N_FFT = 1024
FMIN, FMAX = 55.0, 500.0
YIN_THRESH = 0.15

BAP_BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Feature extraction (frame-level, per utterance)
# ---------------------------------------------------------------------------

def extract_pitch_energy_vad(audio: torch.Tensor) -> dict:
    audio = audio.float().to(DEVICE)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=80, power=2.0, center=False,
    ).to(DEVICE)
    mel = mel_spec(audio).squeeze(0).T
    rms = torch.sqrt(mel.mean(-1).clamp_min(1e-10))
    energy_db = 20.0 * torch.log10(rms + 1e-8)

    tau_min = max(2, int(SR / FMAX))
    tau_max = min(int(SR / FMIN) + 1, N_FFT // 2)
    yin_win = N_FFT - tau_max

    frames = audio.unfold(-1, N_FFT, HOP)
    x = frames.squeeze(0)

    cs = torch.cumsum(F.pad(x * x, (1, 0)), dim=-1)
    p = cs[:, yin_win:] - cs[:, :-yin_win]
    p = p[:, :tau_max + 1]
    n = 1 << int(N_FFT + yin_win - 1).bit_length()
    r = torch.fft.irfft(
        torch.fft.rfft(x, n=n) * torch.fft.rfft(x[:, :yin_win], n=n).conj(), n=n
    )[:, :tau_max + 1]
    d = (p[:, :1] + p - 2.0 * r).clamp_min(0.0)

    cum = torch.cumsum(d, dim=-1)
    lag = torch.arange(1, tau_max + 1, device=DEVICE, dtype=d.dtype)
    dn = torch.ones_like(d)
    dn[:, 1:] = d[:, 1:] * lag / cum[:, 1:].clamp_min(1e-12)

    cand = dn[:, tau_min:tau_max + 1]
    below = cand < YIN_THRESH
    first = torch.where(below.any(dim=-1), below.float().argmax(dim=-1), cand.argmin(dim=-1)) + tau_min

    i = first.clamp(1, tau_max - 1)
    y0 = torch.gather(dn, 1, (i - 1).unsqueeze(1)).squeeze(1)
    y1 = torch.gather(dn, 1, i.unsqueeze(1)).squeeze(1)
    y2 = torch.gather(dn, 1, (i + 1).unsqueeze(1)).squeeze(1)
    denom = y0 - 2 * y1 + y2
    shift = torch.where(denom.abs() > 1e-12, 0.5 * (y0 - y2) / denom, torch.zeros_like(denom))
    tau = i.float() + shift.clamp(-1.0, 1.0)
    f0 = SR / tau.clamp_min(1e-6)

    voiced = (y1 < YIN_THRESH) & (f0 >= FMIN) & (f0 <= FMAX)

    T_out = min(energy_db.size(0), f0.size(0))
    return {
        "f0": f0[:T_out],
        "voiced": voiced[:T_out],
        "energy_db": energy_db[:T_out],
        "tau": tau[:T_out],
    }


def extract_bap(audio: torch.Tensor, f0_tau: torch.Tensor) -> torch.Tensor:
    """Frame-level Band Aperiodicity (5 bands). Returns (T, 5) on DEVICE."""
    audio = audio.float().to(DEVICE)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    win = torch.hann_window(N_FFT, device=DEVICE)
    stft = torch.stft(audio.squeeze(0), N_FFT, HOP, window=win, return_complex=True)
    mag = stft.abs()
    F_bins, T_stft = mag.shape
    T_out = min(T_stft, f0_tau.size(0))
    mag = mag[:, :T_out]
    f0_tau = f0_tau[:T_out]

    bap_out = torch.zeros(T_out, len(BAP_BANDS), device=DEVICE)
    for t in range(T_out):
        period = f0_tau[t].item()
        if period < 2:
            bap_out[t] = 1.0
            continue

        f0_hz = SR / period
        spec = mag[:, t]
        total_pow = (spec ** 2).sum().item()
        if total_pow < 1e-12:
            bap_out[t] = 1.0
            continue

        harmonic_mask = torch.zeros(F_bins, device=DEVICE)
        n_harmonics = int(SR / 2 / f0_hz)
        half_width = max(1, int(f0_hz * 0.25 / (SR / 2) * F_bins))
        for h in range(1, n_harmonics + 1):
            center_bin = int(h * f0_hz / (SR / 2) * (F_bins - 1))
            lo = max(0, center_bin - half_width)
            hi = min(F_bins, center_bin + half_width + 1)
            harmonic_mask[lo:hi] = 1.0

        for bi, (blo, bhi) in enumerate(BAP_BANDS):
            band_lo = int(blo / (SR / 2) * (F_bins - 1))
            band_hi = min(int(bhi / (SR / 2) * (F_bins - 1)), F_bins)
            if band_hi <= band_lo:
                continue
            band_pow = (spec[band_lo:band_hi] ** 2).sum().item()
            if band_pow < 1e-12:
                bap_out[t, bi] = 1.0
                continue
            harm_pow = ((spec[band_lo:band_hi] * harmonic_mask[band_lo:band_hi]) ** 2).sum().item()
            bap_out[t, bi] = 1.0 - harm_pow / band_pow

    return bap_out


def extract_cpps(audio: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Frame-level CPPS. Returns (T,) on CPU."""
    audio = audio.float().cpu().numpy()
    if audio.ndim > 1:
        audio = audio.squeeze()

    frame_len = N_FFT
    T = (len(audio) - frame_len) // HOP + 1
    if T <= 0:
        return torch.zeros(max(1, n_frames))

    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(T, frame_len),
        strides=(audio.strides[0] * HOP, audio.strides[0])
    ).copy()
    frames *= np.hanning(frame_len)

    spec = np.fft.rfft(frames, n=frame_len * 2)
    log_pow = np.log(np.abs(spec) ** 2 + 1e-12)
    cepstrum = np.abs(np.fft.irfft(log_pow)) ** 2

    q_min = int(SR / FMAX)
    q_max = min(int(SR / FMIN), cepstrum.shape[1] - 1)

    cpp = np.zeros(T)
    for t in range(T):
        cep = cepstrum[t, q_min:q_max + 1]
        if cep.size == 0:
            continue
        qs = np.arange(q_min, q_max + 1)
        log_cep = np.log(cep + 1e-12)
        coeffs = np.polyfit(qs, log_cep, 1)
        regression = np.polyval(coeffs, qs)
        peak_idx = np.argmax(log_cep)
        cpp[t] = log_cep[peak_idx] - regression[peak_idx]

    cpps = np.convolve(cpp, np.ones(5) / 5, mode="same")
    return torch.from_numpy(cpps.astype(np.float32))


def normalize_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Per-feature z-normalization within one utterance. (T, D) -> (T, D)."""
    mu = seq.mean(dim=0, keepdim=True)
    std = seq.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (seq - mu) / std


def build_frame_sequences(feats: dict, bap: torch.Tensor, cpps: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build normalized frame-level sequences for each feature set."""
    T = feats["f0"].size(0)

    # base: (T, 3) — pitch_hz, energy_db, vad
    base = torch.stack([
        feats["f0"].cpu(),
        feats["energy_db"].cpu(),
        feats["voiced"].float().cpu(),
    ], dim=-1)  # (T, 3)

    T_bap = min(T, bap.size(0))
    bap_cpu = bap[:T_bap].cpu()

    T_cpps = min(T, cpps.size(0))
    cpps_cpu = cpps[:T_cpps].unsqueeze(-1)  # (T, 1)

    T_all = min(T, T_bap, T_cpps)

    base_norm = normalize_sequence(base[:T_all])
    bap_norm = normalize_sequence(bap_cpu[:T_all])
    cpps_norm = normalize_sequence(cpps_cpu[:T_all])

    return {
        "base": base_norm,                                         # (T, 3)
        "base+bap": torch.cat([base_norm, bap_norm], dim=-1),     # (T, 8)
        "base+bap+cpps": torch.cat([base_norm, bap_norm, cpps_norm], dim=-1),  # (T, 9)
    }


# ---------------------------------------------------------------------------
# GRU classifier
# ---------------------------------------------------------------------------

class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        from torch.nn.utils.rnn import pad_packed_sequence
        out, _ = pad_packed_sequence(out, batch_first=True)
        # mean pooling over valid frames
        mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
        out = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1).float()
        return self.head(out)


def train_and_eval(
    sequences: list[torch.Tensor],
    labels: np.ndarray,
    input_dim: int,
    n_classes: int,
    n_folds: int,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> tuple[float, np.ndarray]:
    """5-fold CV, returns (UAR, per-class recall array)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_preds = np.full_like(labels, -1)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Compute global normalization stats from training set
        train_seqs = [sequences[i] for i in train_idx]
        all_train = torch.cat(train_seqs, dim=0)  # (sum_T, D)
        global_mu = all_train.mean(dim=0)
        global_std = all_train.std(dim=0).clamp_min(1e-6)

        def normalize_global(seq: torch.Tensor) -> torch.Tensor:
            return (seq - global_mu) / global_std

        model = GRUClassifier(input_dim, hidden_dim, n_classes).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        y_train = torch.tensor(labels[train_idx], dtype=torch.long, device=DEVICE)
        # Class weights
        counts = np.bincount(labels[train_idx], minlength=n_classes)
        weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32, device=DEVICE)
        weights = weights / weights.sum() * n_classes
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Training
        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(len(train_idx))
            for start in range(0, len(train_idx), batch_size):
                idx = perm[start:start + batch_size]
                batch_seqs = [normalize_global(sequences[train_idx[i]]) for i in idx]
                lengths = torch.tensor([s.size(0) for s in batch_seqs])
                padded = pad_sequence(batch_seqs, batch_first=True).to(DEVICE)
                targets = y_train[idx]

                logits = model(padded, lengths.to(DEVICE))
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            test_seqs = [normalize_global(sequences[test_idx[i]]) for i in range(len(test_idx))]
            lengths = torch.tensor([s.size(0) for s in test_seqs])
            padded = pad_sequence(test_seqs, batch_first=True).to(DEVICE)
            logits = model(padded, lengths.to(DEVICE))
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds[test_idx] = preds

    uar = recall_score(labels, all_preds, average="macro") * 100
    per_class = recall_score(labels, all_preds, average=None) * 100
    return uar, per_class, all_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vpc_root", default="/mnt/data/disk3/yejin/VPC24")
    ap.add_argument("--dataset", default="IEMOCAP_dev")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.vpc_root) / "data" / args.dataset

    utt2emo = {}
    with open(data_dir / "utt2emo") as f:
        for line in f:
            utt, emo = line.strip().split()
            utt2emo[utt] = emo

    utt2wav = {}
    with open(data_dir / "wav.scp") as f:
        for line in f:
            utt, wav = line.strip().split(None, 1)
            utt2wav[utt] = Path(args.vpc_root) / wav

    utts = sorted(set(utt2emo) & set(utt2wav))
    if args.limit:
        utts = utts[:args.limit]

    emo_list = sorted(set(utt2emo.values()))
    emo2idx = {e: i for i, e in enumerate(emo_list)}
    print(f"[probe] {len(utts)} utterances, {len(emo_list)} classes: {emo_list}")

    # ---- Feature extraction ----
    all_seqs: dict[str, list[torch.Tensor]] = {"base": [], "base+bap": [], "base+bap+cpps": []}
    labels = []
    t0 = time.time()

    for i, utt in enumerate(utts):
        wav_path = utt2wav[utt]
        audio, sr = torchaudio.load(str(wav_path))
        if audio.size(0) > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != SR:
            audio = torchaudio.functional.resample(audio, sr, SR)
        audio = audio.squeeze(0)

        feats = extract_pitch_energy_vad(audio)
        bap = extract_bap(audio, feats["tau"])
        cpps = extract_cpps(audio, feats["f0"].size(0))

        seqs = build_frame_sequences(feats, bap, cpps)
        for k in all_seqs:
            all_seqs[k].append(seqs[k])
        labels.append(emo2idx[utt2emo[utt]])

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(utts)}] {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[probe] feature extraction done in {elapsed:.1f}s\n")

    y = np.array(labels)

    # ---- Train & evaluate ----
    print(f"GRU hidden={args.hidden_dim}, epochs={args.epochs}, lr={args.lr}, "
          f"batch={args.batch_size}, folds={args.folds}")
    print(f"\n{'Feature Set':<20} {'dim':>4}  {'UAR':>6}  "
          + "  ".join(f"{e:>6}" for e in emo_list))
    print("-" * (30 + 8 * len(emo_list)))

    last_preds = None
    for name in ["base", "base+bap", "base+bap+cpps"]:
        seqs = all_seqs[name]
        input_dim = seqs[0].size(-1)

        uar, per_class, preds = train_and_eval(
            seqs, y, input_dim, len(emo_list), args.folds,
            hidden_dim=args.hidden_dim, epochs=args.epochs,
            lr=args.lr, batch_size=args.batch_size,
        )
        last_preds = preds
        per_str = "  ".join(f"{r:6.1f}" for r in per_class)
        print(f"{name:<20} {input_dim:>4}  {uar:6.1f}  {per_str}")

    print(f"\nConfusion matrix (base+bap+cpps):")
    cm = confusion_matrix(y, last_preds)
    print(f"{'':>8}", "  ".join(f"{e:>5}" for e in emo_list))
    for i, row in enumerate(cm):
        print(f"{emo_list[i]:>8}", "  ".join(f"{v:>5d}" for v in row))


if __name__ == "__main__":
    main()
