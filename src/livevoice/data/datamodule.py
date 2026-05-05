"""Lightning DataModules for LiveVoice training.

VCTKDataModule   — VCTK-Corpus (16 kHz or any target SR)
LibriTTSDataModule — LibriTTS (24 kHz)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import lightning as L
import soundfile as sf
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .vctk_dataset import VCTKDataset, collate_fn
from .libritts_dataset import LibriTTSDataset, collate_fn


def _build_duration_weights(items_paths: list[str], cache_path: Path) -> list[float]:
    """Return per-item weights = duration in seconds.

    Caches durations to disk so subsequent runs (or `setup` calls) don't re-scan
    the entire dataset. Cache is keyed by (cache_path); regenerated when item
    list changes.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, float] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    missing = [p for p in items_paths if p not in cache]
    if missing:
        print(f"[duration-cache] Scanning {len(missing)} files (cached: {len(cache)})...")
        for p in tqdm(missing, desc="sf.info"):
            try:
                info = sf.info(p)
                cache[p] = float(info.frames) / float(info.samplerate)
            except Exception:
                cache[p] = 1.0  # fallback weight
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        print(f"[duration-cache] Saved → {cache_path}")

    return [cache.get(p, 1.0) for p in items_paths]


def _make_length_weighted_sampler(dataset, item_paths: list[str], cache_path: Path):
    weights = _build_duration_weights(item_paths, cache_path)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


class VCTKDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            self.train_dataset = VCTKDataset(self.config, split="train")
            self.val_dataset = VCTKDataset(self.config, split="val")

    def train_dataloader(self):
        # Length-weighted sampling: long utterances picked proportionally to
        # their duration so all 4-s windows are reached at equal rate over time.
        item_paths = [it[0] for it in self.train_dataset.items]
        cache_path = Path(self.config.output_dir) / "duration_cache" / "vctk_train.json"
        sampler = _make_length_weighted_sampler(self.train_dataset, item_paths, cache_path)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=self.config.num_workers > 0,
        )


class LibriTTSDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            self.train_dataset = LibriTTSDataset(self.config, split="train")
            self.val_dataset = LibriTTSDataset(self.config, split="val")

    def train_dataloader(self):
        # Length-weighted sampling for LibriTTS — utterances vary 1-15s, plain
        # uniform sampling under-represents long ones. This reaches every 4-s
        # window at a roughly equal rate over enough epochs.
        item_paths = [it[0] for it in self.train_dataset.items]
        splits_tag = ",".join(self.config.libritts_train_splits)
        cache_path = Path(self.config.output_dir) / "duration_cache" / f"libritts_train_{splits_tag}.json"
        sampler = _make_length_weighted_sampler(self.train_dataset, item_paths, cache_path)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=self.config.num_workers > 0,
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from livevoice.config import LiveVoiceConfig

    # Quick sanity check
    cfg = LiveVoiceConfig(
        libritts_path="/mnt/data/disk2/LibriTTS",
        sample_rate=24000,
        dac_model_type="24khz",
        dac_sample_rate=24000,
        dac_n_codebooks=9,
        max_windows=32,
    )
    dm = LibriTTSDataModule(cfg)
    dm.setup("fit")
    print(f"Train: {len(dm.train_dataset)}  Val: {len(dm.val_dataset)}")
    batch = next(iter(dm.train_dataloader()))
    print({k: (v.shape if hasattr(v, "shape") else type(v)) for k, v in batch.items()})
