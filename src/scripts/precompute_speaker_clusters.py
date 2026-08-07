"""Precompute k-means speaker clusters for the GRL adversary.

Motivation: a 1151-way per-speaker adversary can't stay competent with batch 8, so
the reversed gradient is too weak/unstable to remove speaker (see
debug/diag_grl_classifier.py). Classifying a smaller set of ACOUSTICALLY-COHERENT
speaker clusters is an easier proxy that still targets speaker-discriminative info —
so the adversary trains fast and the reversal actually bites. Clusters must come from
a speaker embedding (ECAPA here); random groupings would be as hard as full speaker ID.

Pipeline: for each training speaker, average ECAPA embeddings over a few utterances →
L2-normalize → spherical k-means (K clusters) → save {speaker_id: cluster_id}.

Writes JSON: {"num_clusters": K, "speaker_to_cluster": {spk: cid}, "meta": {...}}.
Point config.grl_cluster_file at it and set config.grl_num_clusters = K.

    conda run -n sound python /workspace/LiveVoice/src/scripts/precompute_speaker_clusters.py \
        --num_clusters 64 --utts_per_speaker 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from livevoice.config import LiveVoiceConfig  # noqa: E402
from livevoice.data.speaker_vocab import build_libritts_speaker_vocab  # noqa: E402
from livevoice.model.speechbrain_speaker_encoder import SpeechBrainECAPASpeakerEncoder  # noqa: E402


def _load_audio(path: str, target_sr: int, max_seconds: float) -> torch.Tensor:
    audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio_np).float().mean(dim=1)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    cap = int(max_seconds * target_sr)
    if audio.numel() > cap:
        audio = audio[:cap]
    return audio


def spherical_kmeans(X: torch.Tensor, K: int, iters: int = 100, seed: int = 0):
    """X: (N,D) L2-normalized. Returns (assign (N,), centroids (K,D))."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = X[torch.randperm(X.size(0), generator=g)[:K]].clone()
    assign = torch.zeros(X.size(0), dtype=torch.long)
    for _ in range(iters):
        new_assign = torch.cdist(X, C).argmin(dim=1)
        if torch.equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for k in range(K):
            m = assign == k
            if m.any():
                C[k] = X[m].mean(dim=0)
            else:  # re-seed an empty cluster to the farthest point
                C[k] = X[torch.cdist(X, C).min(dim=1).values.argmax()]
        C = F.normalize(C, dim=-1)
    return assign, C


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_clusters", type=int, default=64)
    ap.add_argument("--utts_per_speaker", type=int, default=5)
    ap.add_argument("--max_seconds", type=float, default=8.0)
    ap.add_argument("--out", type=str, default=None, help="default: config.grl_cluster_file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = LiveVoiceConfig()
    out_path = Path(args.out or config.grl_cluster_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vocab = build_libritts_speaker_vocab(config)   # deterministic train-split speaker list
    speakers = sorted(vocab)
    root = Path(config.libritts_path)
    splits = tuple(getattr(config, "libritts_train_splits",
                           ("train-clean-100", "train-clean-360")))
    print(f"[clusters] {len(speakers)} speakers, K={args.num_clusters}, "
          f"{args.utts_per_speaker} utts/spk, device={args.device}")

    # gather utterance paths per speaker
    rng = random.Random(args.seed)
    spk_wavs: dict[str, list[str]] = {s: [] for s in speakers}
    for s in splits:
        sdir = root / s
        if not sdir.exists():
            continue
        for wav in sdir.glob("**/*.wav"):
            spk = wav.parts[-3]
            if spk in spk_wavs:
                spk_wavs[spk].append(str(wav))

    enc = SpeechBrainECAPASpeakerEncoder(config).to(args.device).eval()

    embs, kept = [], []
    with torch.no_grad():
        for i, spk in enumerate(speakers):
            wavs = spk_wavs[spk]
            if not wavs:
                continue
            rng.shuffle(wavs)
            per = []
            for w in wavs[: args.utts_per_speaker]:
                try:
                    a = _load_audio(w, config.sample_rate, args.max_seconds).to(args.device)
                    per.append(enc(a.unsqueeze(0)).squeeze(0).cpu())  # (192,)
                except Exception:
                    continue
            if per:
                embs.append(torch.stack(per).mean(dim=0))
                kept.append(spk)
            if (i + 1) % 200 == 0:
                print(f"[clusters]   embedded {i + 1}/{len(speakers)} speakers")

    X = F.normalize(torch.stack(embs), dim=-1)     # (S,192) spherical
    assign, _ = spherical_kmeans(X, args.num_clusters, seed=args.seed)

    speaker_to_cluster = {spk: int(assign[i]) for i, spk in enumerate(kept)}
    sizes = torch.bincount(assign, minlength=args.num_clusters).tolist()
    payload = {
        "num_clusters": args.num_clusters,
        "speaker_to_cluster": speaker_to_cluster,
        "meta": {
            "num_speakers": len(kept),
            "utts_per_speaker": args.utts_per_speaker,
            "embedder": "speechbrain_ecapa",
            "cluster_sizes": sizes,
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[clusters] wrote {out_path}  ({len(kept)} speakers → {args.num_clusters} clusters)")
    print(f"[clusters] cluster sizes: min={min(sizes)} max={max(sizes)} "
          f"empty={sizes.count(0)}")


if __name__ == "__main__":
    main()
