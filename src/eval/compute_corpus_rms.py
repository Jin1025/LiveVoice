"""Compute median RMS per corpus for universal gain matching.

Usage:
    python src/eval/compute_corpus_rms.py

Outputs the median RMS of LibriTTS (training data) and each VPC eval corpus,
then prints the gain factor needed to match each corpus to LibriTTS level.
"""
import glob
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

VPC_ROOT = "/mnt/data/disk3/yejin/VPC"
LIBRITTS_ROOT = "/mnt/data/disk2/LibriTTS"
LIBRITTS_SPLITS = ("train-clean-100",)  # representative subset, same distribution as train-clean-360
TARGET_SR = 16000
MAX_UTTS = 5000  # sample for speed


def rms_from_wav(path, sr=TARGET_SR):
    try:
        audio, file_sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio).float()
    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)
    if audio.numel() < 1600:  # < 0.1s
        return None
    return float(audio.pow(2).mean().sqrt())


def rms_from_pipe(cmd, vpc_root, sr=TARGET_SR):
    """Decode a Kaldi pipe entry like 'flac -c -d -s path.flac |'."""
    cmd = cmd.rstrip().rstrip("|").strip()
    parts = cmd.split()
    flac_path = None
    for p in parts:
        if p.endswith(".flac") or p.endswith(".wav"):
            flac_path = p
            break
    if flac_path is None:
        return None
    if not os.path.isabs(flac_path):
        flac_path = os.path.join(vpc_root, flac_path)
    if not os.path.isfile(flac_path):
        return None
    return rms_from_wav(flac_path, sr)


def compute_rms_wavscp(wavscp_path, vpc_root, max_utts=MAX_UTTS):
    rms_vals = []
    with open(wavscp_path) as f:
        lines = f.readlines()
    if len(lines) > max_utts:
        np.random.seed(42)
        indices = np.random.choice(len(lines), max_utts, replace=False)
        lines = [lines[i] for i in indices]
    for line in tqdm(lines, desc=os.path.basename(os.path.dirname(wavscp_path)), leave=False):
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        entry = parts[1]
        if entry.rstrip().endswith("|"):
            r = rms_from_pipe(entry, vpc_root)
        else:
            path = entry if os.path.isabs(entry) else os.path.join(vpc_root, entry)
            r = rms_from_wav(path)
        if r is not None and r > 1e-6:
            rms_vals.append(r)
    return rms_vals


def compute_rms_libritts(splits, max_utts=MAX_UTTS):
    rms_vals = []
    all_wavs = []
    for split in splits:
        split_dir = os.path.join(LIBRITTS_ROOT, split)
        all_wavs.extend(glob.glob(os.path.join(split_dir, "**", "*.wav"), recursive=True))
    if len(all_wavs) > max_utts:
        np.random.seed(42)
        indices = np.random.choice(len(all_wavs), max_utts, replace=False)
        all_wavs = [all_wavs[i] for i in indices]
    for path in tqdm(all_wavs, desc="LibriTTS", leave=False):
        r = rms_from_wav(path)
        if r is not None and r > 1e-6:
            rms_vals.append(r)
    return rms_vals


if __name__ == "__main__":
    corpora = {}

    # LibriTTS (training reference)
    print("Computing LibriTTS RMS...")
    corpora["LibriTTS_train"] = compute_rms_libritts(LIBRITTS_SPLITS)

    # VPC eval corpora
    vpc_datasets = [
        "IEMOCAP_dev", "IEMOCAP_test",
        "libri_dev_trials_mixed", "libri_test_trials_mixed",
        "libri_dev_enrolls", "libri_test_enrolls",
        "train-clean-360",
    ]
    for ds in vpc_datasets:
        wavscp = os.path.join(VPC_ROOT, "data", ds, "wav.scp")
        if not os.path.isfile(wavscp):
            print(f"  {ds}: wav.scp not found, skipping")
            continue
        print(f"Computing {ds} RMS...")
        corpora[ds] = compute_rms_wavscp(wavscp, VPC_ROOT)

    # Report
    ref_key = "LibriTTS_train"
    ref_median = float(np.median(corpora[ref_key]))
    print(f"\n{'Corpus':<30} {'median RMS':>12} {'mean RMS':>12} {'N':>8} {'gain':>8}")
    print("-" * 75)
    for name, vals in corpora.items():
        if not vals:
            continue
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        gain = ref_median / med if med > 0 else 0
        marker = " ← reference" if name == ref_key else ""
        print(f"{name:<30} {med:>12.6f} {mean:>12.6f} {len(vals):>8} {gain:>8.3f}{marker}")
