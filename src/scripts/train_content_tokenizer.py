"""STAGE 1: train the content tokenizer alone (ASR + GRL, no VC decoder).

Two-stage recipe, following CosyVoice 2 (arXiv:2412.10117), which trains its supervised
semantic tokenizer separately and FREEZES it for TTS training:

  Stage 1 (this script)
      sw2v(frozen) → refiner → sw2v_proj → [FSQ] → sw2v_to_hidden
      loss = asr_loss_weight * ASR + grl_loss_weight * GRL      (NO reconstruction CE)
      → a content representation that keeps phonemes and drops speaker.

  Stage 2 (scripts/train.py with --content_tokenizer_ckpt)
      load these weights, FREEZE the content path, train the VC decoder on top.
      The decoder can no longer pull speaker identity back into content, so it has to
      use the speaker prompt.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_content_tokenizer.py --exp_name stage1_ctc
"""
import argparse
import os
import sys

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
from livevoice.model import Sw2vContentEncoder
from livevoice.model.content_supervision import ContentTokenizerModel
from livevoice.lightning.content_tokenizer_module import ContentTokenizerLightningModule
from livevoice.data.datamodule import LibriTTSDataModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="stage1_content_tokenizer")
    p.add_argument("--output_dir", type=str, default="/mnt/data/disk2/yejin/LiveVoice")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--resume_from", type=str, default=None)
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
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.lr,
        max_windows=args.max_windows,
        seed=args.seed,
        precision=args.precision,
    )
    _cs = str(config.content_source).lower()
    if _cs not in ("sw2v", "zipformer"):
        raise SystemExit(
            f"Stage 1 needs a continuous content encoder (sw2v or zipformer); "
            f"got {config.content_source!r}")
    if not (config.use_asr_supervision or config.use_speaker_grl):
        raise SystemExit("Stage 1 needs use_asr_supervision and/or use_speaker_grl enabled.")

    # GRL classifier size (speakers or k-means clusters) — same map the dataset labels with.
    if config.use_speaker_grl:
        from livevoice.data.speaker_vocab import build_libritts_grl_label_map
        _, config.grl_num_speakers = build_libritts_grl_label_map(config)
        kind = "clusters" if int(getattr(config, "grl_num_clusters", 0)) > 0 else "speakers"
        print(f"[stage1] speaker-GRL: {config.grl_num_speakers} {kind}")

    if _cs == "zipformer":
        print(f"[stage1] Building Zipformer content encoder ({config.zipformer_ckpt}, "
              f"layer={config.zipformer_layer}) ...")
        from livevoice.model.zipformer_content import ZipformerContentEncoder
        _lyr = str(config.zipformer_layer)
        content_extractor = ZipformerContentEncoder(
            config, config.zipformer_ckpt,
            layer=(_lyr if _lyr == "out" else int(_lyr)))
    else:
        print(f"[stage1] Building SW2V content encoder ({config.sw2v_ckpt}) ...")
        content_extractor = Sw2vContentEncoder(config)
    model = ContentTokenizerModel(config, content_extractor)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[stage1] Parameters: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")
    _cache = (config.zipformer_features_dir if _cs == "zipformer" else config.sw2v_features_dir)
    print(f"[stage1] content_source={_cs}  full features: "
          f"{'ONLINE' if config.sw2v_full_online else _cache}")

    lit_model = ContentTokenizerLightningModule(config, model)
    dm = LibriTTSDataModule(config)

    log_dir = os.path.join(config.output_dir, "logs")
    ckpt_dir = os.path.join(config.output_dir, "checkpoints", config.exp_name)
    if args.use_wandb and WandbLogger is not None:
        logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity,
                             name=config.exp_name, save_dir=log_dir)
    else:
        logger = TensorBoardLogger(save_dir=log_dir, name=config.exp_name)

    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, filename="step_latest", every_n_train_steps=2000,
                        save_top_k=1, monitor=None),
        ModelCheckpoint(dirpath=ckpt_dir, filename="last", save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.max_grad_norm,
        log_every_n_steps=50,
    )
    trainer.fit(lit_model, dm, ckpt_path=args.resume_from)
    print(f"[stage1] done → {ckpt_dir}  (use it via train.py --content_tokenizer_ckpt)")


if __name__ == "__main__":
    main()
