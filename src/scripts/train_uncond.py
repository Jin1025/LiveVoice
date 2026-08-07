"""Train unconditional LiveVoice model (decoder-only AR over DAC codes).

Usage (inside docker `yejin2`, conda `sound`):
    python src/scripts/train_uncond.py --exp_name uncond_vctk --batch_size 16
"""
import argparse
import os
import sys

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
try:
    from lightning.pytorch.loggers import WandbLogger
except Exception:  # pragma: no cover
    WandbLogger = None

from livevoice.config import LiveVoiceConfig
from livevoice.model import DACModel, UnconditionalModel
from livevoice.lightning import UnconditionalLightningModule
from livevoice.data.datamodule import VCTKDataModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="uncond_vctk")
    p.add_argument("--output_dir", type=str, default="/mnt/data/disk2/yejin/LiveVoice")
    p.add_argument("--vctk_path", type=str, default="/mnt/data/disk2/VCTK-Corpus")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--audio_duration", type=float, default=4.0)
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--val_check_interval", type=float, default=0.25)
    p.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    p.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    p.set_defaults(use_wandb=True)
    p.add_argument("--wandb_project", type=str, default="LiveVoice")
    p.add_argument("--wandb_entity", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed)

    config = LiveVoiceConfig(
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        vctk_path=args.vctk_path,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.lr,
        audio_duration=args.audio_duration,
        max_windows=args.max_windows,
        seed=args.seed,
        precision=args.precision,
        compile=args.compile,
        val_check_interval=args.val_check_interval,
    )

    print(f"[train_uncond] Building DAC model ({config.dac_model_type})...")
    dac_model = DACModel(config)

    print("[train_uncond] Building UnconditionalModel...")
    model = UnconditionalModel(config, dac_model)

    if args.compile:
        print("[train_uncond] torch.compile ...")
        model = torch.compile(model)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_uncond] Trainable parameters: {param_count / 1e6:.2f}M")

    lit_model = UnconditionalLightningModule(config, model)
    dm = VCTKDataModule(config)

    log_dir = os.path.join(config.output_dir, "logs")
    ckpt_dir = os.path.join(config.output_dir, "checkpoints", config.exp_name)
    if args.use_wandb and WandbLogger is not None:
        os.environ.setdefault("WANDB_CONSOLE", "off")
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=config.exp_name,
            save_dir=log_dir,
        )
    else:
        logger = TensorBoardLogger(log_dir, name=config.exp_name)

    callbacks = [
        # Overwrite one checkpoint every epoch + keep last.ckpt
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch_latest",
            every_n_epochs=1,
            save_top_k=1,
            enable_version_counter=False,
            save_last=True,
        ),
        # Overwrite one checkpoint every 1000 train steps
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="step_latest",
            every_n_train_steps=1000,
            save_top_k=1,
            enable_version_counter=False,
            save_last=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.max_grad_norm,
        deterministic=False,
    )

    trainer.fit(lit_model, dm, ckpt_path=args.resume_from)
    print("[train_uncond] Done.")


if __name__ == "__main__":
    main()
