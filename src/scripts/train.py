"""Train full LiveVoice VC model.

Usage (inside docker `yejin2`, conda `sound`):
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py --exp_name base_vctk
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import TensorBoardLogger
try:
    from lightning.pytorch.loggers import WandbLogger
except Exception:  # pragma: no cover
    WandbLogger = None

from livevoice.config import LiveVoiceConfig
from livevoice.model import (
    build_codec,
    HuBERTContentExtractor,
    StreamVoiceAnonContentEncoder,
    ProsodyExtractor,
    LiveVoiceModel,
)
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.data.datamodule import VCTKDataModule, LibriTTSDataModule


class EveryNEpochCheckpoint(Callback):
    """Save numbered checkpoints every N epochs (e.g., 5epoch, 10epoch)."""

    def __init__(self, dirpath: str, every_n_epochs: int = 5):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_epochs = int(every_n_epochs)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_num = int(trainer.current_epoch) + 1
        if epoch_num % self.every_n_epochs != 0:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        ckpt_path = os.path.join(self.dirpath, f"{epoch_num}epoch.ckpt")
        trainer.save_checkpoint(ckpt_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", type=str, default="base_libritts")
    p.add_argument("--output_dir", type=str, default="/mnt/data/disk2/yejin/LiveVoice")
    # Dataset selection
    p.add_argument("--dataset", type=str, default="libritts", choices=["vctk", "libritts"],
                   help="Which dataset to use. libritts automatically sets 24 kHz config.")
    p.add_argument("--vctk_path", type=str, default="/mnt/data/disk2/VCTK-Corpus")
    p.add_argument("--libritts_path", type=str, default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--features_dir", type=str, default="/mnt/data/disk2/yejin/LiveVoice/features",
                   help="Path to precomputed HuBERT features (from extract_features.py). "
                        "E.g. /mnt/data/disk2/yejin/LiveVoice/features")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--audio_duration", type=float, default=4.0)
    p.add_argument("--use_prosody", action="store_true")
    # p.add_argument("--content_conditioning", type=str, default="film", choices=["additive", "film"])
    # p.add_argument("--speaker_conditioning", type=str, default="prefix", choices=["crossattn", "global_avg", "prefix"])
    p.add_argument("--speaker_prefix_len", type=int, default=8)
    # p.add_argument("--speaker_encoder_type", type=str, default="speechbrain_ecapa", choices=["codec", "speechbrain_ecapa"])
    # p.add_argument("--speechbrain_source", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    # p.add_argument(
    #     "--speechbrain_savedir",
    #     type=str,
    #     default="/mnt/data/disk2/yejin/LiveVoice/pretrained_models/speechbrain__spkrec-ecapa-voxceleb",
    # )
    p.add_argument("--speechbrain_sample_rate", type=int, default=16000)
    p.add_argument("--speechbrain_embedding_dim", type=int, default=192)
    # p.add_argument("--content_source", type=str, default="streamvoiceanon",
    #                choices=["hubert", "mimi_semantic", "streamvoiceanon"])
    # p.add_argument("--streamvoiceanon_repo", type=str, default="/workspace/StreamVoiceAnon")
    # p.add_argument(
    #     "--streamvoiceanon_encoder_config",
    #     type=str,
    #     default="/workspace/StreamVoiceAnon/configs/hydra_arcs/speech_tokenizers/causal-encoder-lfq-8192.yaml",
    # )
    # p.add_argument(
    #     "--streamvoiceanon_encoder_ckpt",
    #     type=str,
    #     default="/workspace/StreamVoiceAnon/ckpt/asr_s2s_bsq_8192_causal_down_whisper.pth",
    # )
    p.add_argument("--pairing", type=str, default="same_speaker",
                   choices=["same_speaker", "reconstruct"])
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    p.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    p.set_defaults(use_wandb=True)
    p.add_argument("--wandb_project", type=str, default="LiveVoice")
    p.add_argument("--wandb_entity", type=str, default=None)
    # Codec
    # p.add_argument("--codec", type=str, default="mimi", choices=["mimi", "jhcodec"])
    # p.add_argument("--mimi_cache_dir", type=str, default="/mnt/data/disk2/yejin/LiveVoice/mimi_precomputed")
    # Ablations
    p.add_argument("--zero_speaker", action="store_true")
    p.add_argument("--zero_content", action="store_true")
    p.add_argument("--ablate_cross_attn", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed)

    config = LiveVoiceConfig(
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        vctk_path=args.vctk_path,
        libritts_path=args.libritts_path,
        features_dir=args.features_dir,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.lr,
        audio_duration=args.audio_duration,
        use_prosody=args.use_prosody,
        # content_conditioning=args.content_conditioning,
        # speaker_conditioning=args.speaker_conditioning,
        speaker_prefix_len=args.speaker_prefix_len,
        # speaker_encoder_type=args.speaker_encoder_type,
        # speechbrain_source=args.speechbrain_source,
        # speechbrain_savedir=args.speechbrain_savedir,
        speechbrain_sample_rate=args.speechbrain_sample_rate,
        speechbrain_embedding_dim=args.speechbrain_embedding_dim,
        # content_source=args.content_source,
        # streamvoiceanon_repo=args.streamvoiceanon_repo,
        # streamvoiceanon_encoder_config=args.streamvoiceanon_encoder_config,
        # streamvoiceanon_encoder_ckpt=args.streamvoiceanon_encoder_ckpt,
        pairing=args.pairing,
        max_windows=args.max_windows,
        seed=args.seed,
        precision=args.precision,
        compile=args.compile,
        zero_speaker=args.zero_speaker,
        zero_content=args.zero_content,
        ablate_cross_attn=args.ablate_cross_attn,
        # codec=args.codec
    )

    if str(config.content_source).lower() != "hubert":
        config.features_dir = None

    print(f"[train] Dataset: {args.dataset}  SR: {config.sample_rate} Hz  "
          f"codec: {config.codec}  n_codebooks: {config.n_codebooks_predict}  "
          f"speaker_encoder: {config.speaker_encoder_type}")

    print(f"[train] Building codec ({config.codec})...")
    codec_model = build_codec(config)

    # Skip HuBERT when content_source is not "hubert" — it would be loaded
    # only to sit on the GPU unused (94M params + ~500 modules in eval mode).
    content_source = str(getattr(config, "content_source", "hubert")).lower()
    if content_source == "hubert":
        print(f"[train] Building HuBERT content extractor ({config.hubert_model_name}, layer {config.hubert_layer})...")
        content_extractor = HuBERTContentExtractor(config)
    elif content_source == "streamvoiceanon":
        print("[train] Building StreamVoiceAnon causal content encoder...")
        content_extractor = StreamVoiceAnonContentEncoder(config)
    else:
        print(f"[train] content_source={content_source} → skipping HuBERT build")
        content_extractor = None

    prosody_extractor = None
    if config.use_prosody:
        print("[train] Building ProsodyExtractor...")
        prosody_extractor = ProsodyExtractor(config)

    print("[train] Building LiveVoiceModel...")
    model = LiveVoiceModel(config, codec_model, content_extractor, prosody_extractor)

    if args.compile:
        print("[train] torch.compile ...")
        model = torch.compile(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] Parameters: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")

    lit_model = LiveVoiceLightningModule(config, model)
    dm = LibriTTSDataModule(config) if args.dataset == "libritts" else VCTKDataModule(config)

    log_dir = os.path.join(config.output_dir, "logs")
    ckpt_dir = os.path.join(config.output_dir, "checkpoints", config.exp_name)
    if args.use_wandb and WandbLogger is not None:
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=config.exp_name,
            save_dir=log_dir,
        )
        print(f"[train] Logger: WandbLogger(project={args.wandb_project})")
    else:
        logger = TensorBoardLogger(log_dir, name=config.exp_name)
        print("[train] Logger: TensorBoardLogger (W&B disabled or unavailable)")

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
        # EveryNEpochCheckpoint(dirpath=ckpt_dir, every_n_epochs=5),
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
    print("[train] Done.")


if __name__ == "__main__":
    main()
