"""Lightning DataModules for LiveVoice training.

VCTKDataModule   — VCTK-Corpus (16 kHz or any target SR)
LibriTTSDataModule — LibriTTS (24 kHz)
"""
from __future__ import annotations

import lightning as L
from torch.utils.data import DataLoader

from .vctk_dataset import VCTKDataset, collate_fn
from .libritts_dataset import LibriTTSDataset, collate_fn


class VCTKDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            self.train_dataset = VCTKDataset(self.config, split="train")
            self.val_dataset = VCTKDataset(self.config, split="val")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
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
