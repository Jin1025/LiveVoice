"""Deterministic speaker→index vocabulary over the LibriTTS training splits.

Used by the speaker-GRL adversary (model/speaker_grl.py): the classifier needs
a fixed number of speaker classes known at model-build time, and the dataset
needs the SAME string→id mapping to emit per-utterance labels. Both call this
one function; because it scans the speaker directories in sorted order, the two
call sites get identical ids with no shared cache file.

Only speaker directories are listed (not every wav), so this is a cheap
directory walk even on train-clean-360.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _train_splits(config) -> tuple[str, ...]:
    # Fallback only; config.libritts_train_splits (clean-100 + clean-360) normally wins.
    default_train = ("train-clean-100", "train-clean-360")
    return tuple(getattr(config, "libritts_train_splits", default_train))


@lru_cache(maxsize=8)
def _vocab_cached(libritts_path: str, splits: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    root = Path(libritts_path)
    speakers: set[str] = set()
    for s in splits:
        split_dir = root / s
        if not split_dir.exists():
            continue
        for spk_dir in split_dir.iterdir():
            if spk_dir.is_dir():
                speakers.add(spk_dir.name)
    return tuple((spk, i) for i, spk in enumerate(sorted(speakers)))


def build_libritts_speaker_vocab(config) -> dict[str, int]:
    """Return {speaker_id: contiguous_index} over the configured train splits,
    sorted by speaker id so it is stable across processes/runs."""
    return dict(_vocab_cached(str(config.libritts_path), _train_splits(config)))


def build_libritts_grl_label_map(config) -> tuple[dict[str, int], int]:
    """Label map + class count for the speaker-GRL adversary.

    If config.grl_num_clusters > 0 and the cluster file exists, returns the
    {speaker_id: cluster_id} k-means map (num classes = num_clusters). Otherwise
    falls back to the full per-speaker vocab. The model classifier is sized from the
    returned count, and the dataset emits labels from the returned map — one source of
    truth so both agree.
    """
    K = int(getattr(config, "grl_num_clusters", 0))
    if K > 0:
        path = getattr(config, "grl_cluster_file", None)
        if path and Path(path).exists():
            import json
            with open(path) as f:
                data = json.load(f)
            mapping = {str(s): int(c) for s, c in data["speaker_to_cluster"].items()}
            k = int(data.get("num_clusters", (max(mapping.values()) + 1) if mapping else 0))
            return mapping, k
        print(
            f"[speaker_vocab] WARNING: grl_num_clusters={K} but cluster file missing "
            f"({path}) → run scripts/precompute_speaker_clusters.py. Falling back to "
            f"full per-speaker adversary."
        )
    vocab = build_libritts_speaker_vocab(config)
    return vocab, len(vocab)
