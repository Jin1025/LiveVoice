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
    Sw2vContentEncoder,
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
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=400000,
                   help="Stop after this many optimizer steps (e.g. 400000). "
                        "Training ends when max_steps or max_epochs is hit first.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use_prosody", action="store_true")
    p.add_argument("--wer_epoch_samples", type=int, default=50,
                   help="Fixed (seeded) #utterances for the epoch-end WER/spk_sim eval.")
    p.add_argument("--wavlm_sv_variant", type=str, default="wavlm_large",
                   choices=["wavlm_large", "wavlm_base_plus"])
    p.add_argument("--max_windows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume_from", type=str, default=None)
    # STAGE 2: start from a Stage-1 content tokenizer (train_content_tokenizer.py) and
    # freeze the content path, so the VC decoder can't pull speaker back into content.
    p.add_argument("--content_tokenizer_ckpt", type=str, default=None,
                   help="Stage-1 checkpoint; loads content_refiner/sw2v_proj/[FSQ]/sw2v_to_hidden")
    p.add_argument("--no_freeze_content", dest="freeze_content", action="store_false",
                   help="fine-tune the loaded content path instead of freezing it")
    p.set_defaults(freeze_content=True)
    p.add_argument("--keep_aux_losses", action="store_true",
                   help="Stage 2: keep computing ASR/GRL even with the content path frozen "
                        "(monitoring only — they can't change frozen weights)")
    p.add_argument("--use_wandb", dest="use_wandb", action="store_true")
    p.add_argument("--no_wandb", dest="use_wandb", action="store_false")
    p.set_defaults(use_wandb=True)
    p.add_argument("--wandb_project", type=str, default="LiveVoice")
    p.add_argument("--wandb_entity", type=str, default=None)
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
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.lr,
        use_prosody=args.use_prosody,
        wer_epoch_samples=args.wer_epoch_samples,
        wavlm_sv_variant=args.wavlm_sv_variant,
        max_windows=args.max_windows,
        seed=args.seed,
        precision=args.precision,
        compile=args.compile,
        zero_speaker=args.zero_speaker,
        zero_content=args.zero_content,
        ablate_cross_attn=args.ablate_cross_attn,
    )

    # HuBERT and SW2V both support precomputed caches, but use separate config paths.
    content_source = str(config.content_source).lower()
    if content_source not in ("hubert", "sw2v", "zipformer"):
        config.features_dir = None

    if content_source == "hubert":
        cache_name = "features_dir"
        cache_base = config.features_dir
    elif content_source == "sw2v":
        cache_name = "sw2v_features_dir"
        cache_base = config.sw2v_features_dir
    elif content_source == "zipformer":
        cache_name = "zipformer_features_dir"
        cache_base = config.zipformer_features_dir
    else:
        cache_name = None
        cache_base = None

    if cache_base:
        cache_dir = os.path.join(cache_base, args.dataset)
        if os.path.isdir(cache_dir):
            print(f"[train] Main-path content features: using {cache_name}={cache_dir}")
        else:
            print(f"[train] Main-path content features: {cache_name}={cache_dir} does not exist; "
                  "falling back to online extraction")
    else:
        if cache_name:
            print(f"[train] Main-path content features: {cache_name}=None; using online extraction")
        else:
            print(f"[train] Main-path content features: content_source={content_source} is online-only")

    # ASR/GRL full-utterance features are a SEPARATE path, driven by sw2v_full_online
    # (not sw2v_features_dir): full_online=True → always online; else cache; else online fallback.
    if content_source == "sw2v" and (
        bool(getattr(config, "use_asr_supervision", False))
        or bool(getattr(config, "use_speaker_grl", False))
    ):
        full_online = bool(getattr(config, "sw2v_full_online", False))
        full_cache_ok = bool(config.sw2v_features_dir) and os.path.isdir(
            os.path.join(config.sw2v_features_dir, args.dataset)
        )
        if full_online:
            print("[train] ASR/GRL full features: ONLINE (sw2v_full_online=True; cache ignored)")
        elif full_cache_ok:
            print(f"[train] ASR/GRL full features: cache sw2v_features_dir={config.sw2v_features_dir}")
        else:
            print("[train] ASR/GRL full features: ONLINE (fallback — sw2v cache missing/None)")

    print(f"[train] Dataset: {args.dataset}  SR: {config.sample_rate} Hz  "
          f"codec: {config.codec}  n_codebooks: {config.n_codebooks_predict}  "
          f"speaker_encoder: {config.speaker_encoder_type}")

    print(f"[train] Building codec ({config.codec})...")
    codec_model = build_codec(config)

    # Skip HuBERT when content_source is not "hubert" — it would be loaded
    # only to sit on the GPU unused (94M params + ~500 modules in eval mode).
    if content_source == "hubert":
        print(f"[train] Building HuBERT content extractor ({config.hubert_model_name}, layer {config.hubert_layer})...")
        content_extractor = HuBERTContentExtractor(config)
    elif content_source == "streamvoiceanon":
        print("[train] Building StreamVoiceAnon causal content encoder...")
        content_extractor = StreamVoiceAnonContentEncoder(config)
    elif content_source == "sw2v":
        print(f"[train] Building SW2V content encoder ({config.sw2v_ckpt})...")
        content_extractor = Sw2vContentEncoder(config)
    elif content_source == "zipformer":
        from livevoice.model.zipformer_content import ZipformerContentEncoder
        print(f"[train] Building Zipformer content encoder ({config.zipformer_ckpt})...")
        _lyr = str(config.zipformer_layer)
        content_extractor = ZipformerContentEncoder(
            config, config.zipformer_ckpt,
            layer=(_lyr if _lyr == "out" else int(_lyr)))
    else:
        print(f"[train] content_source={content_source} → skipping HuBERT build")
        content_extractor = None

    prosody_extractor = None
    if config.use_prosody:
        print("[train] Building ProsodyExtractor...")
        prosody_extractor = ProsodyExtractor(config)

    # Speaker-GRL adversary needs the class count (train-split speaker vocab) at model
    # build time; the dataset later builds the SAME deterministic mapping for labels.
    if bool(getattr(config, "use_speaker_grl", False)) and args.dataset == "libritts":
        from livevoice.data.speaker_vocab import build_libritts_grl_label_map
        _, config.grl_num_speakers = build_libritts_grl_label_map(config)
        kind = "clusters" if int(getattr(config, "grl_num_clusters", 0)) > 0 else "speakers"
        print(f"[train] speaker-GRL: grl_num_speakers={config.grl_num_speakers} ({kind})")

    print("[train] Building LiveVoiceModel...")
    model = LiveVoiceModel(config, codec_model, content_extractor, prosody_extractor)

    if args.compile:
        print("[train] torch.compile ...")
        model = torch.compile(model)

    # STAGE 2: load the Stage-1 content tokenizer and (by default) freeze it. Parameter
    # names are shared via build_content_path, so the Stage-1 state_dict maps 1:1.
    if args.content_tokenizer_ckpt:
        from livevoice.model.content_supervision import CONTENT_PATH_PREFIXES
        obj = torch.load(args.content_tokenizer_ckpt, map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj)
        content_sd = {
            k[len("model."):]: v for k, v in sd.items()
            if k.startswith("model.") and k[len("model."):].split(".")[0] in CONTENT_PATH_PREFIXES
        }
        if not content_sd:
            raise SystemExit(
                f"[train] no content-path weights in {args.content_tokenizer_ckpt} "
                f"(looked for {CONTENT_PATH_PREFIXES})"
            )
        missing, unexpected = model.load_state_dict(content_sd, strict=False)
        loaded = sorted({k.split('.')[0] for k in content_sd})
        print(f"[train] STAGE 2: loaded content path from {args.content_tokenizer_ckpt} "
              f"→ {loaded} ({len(content_sd)} tensors, {len(unexpected)} unexpected)")
        if args.freeze_content:
            n = 0
            for name, p in model.named_parameters():
                if name.split(".")[0] in CONTENT_PATH_PREFIXES:
                    p.requires_grad = False
                    n += p.numel()
            print(f"[train] STAGE 2: content path FROZEN ({n / 1e6:.2f}M params) — the decoder "
                  f"must take speaker identity from the prompt, not from content.")
            # With the content path frozen, ASR/GRL can no longer shape anything shared —
            # their gradients stop at their own heads — so computing them is pure overhead
            # (a full-utterance forward every step). Disable unless explicitly kept.
            if not args.keep_aux_losses:
                config.use_asr_supervision = False
                config.use_speaker_grl = False
                config.asr_loss_weight = 0.0
                config.grl_loss_weight = 0.0
                model.use_asr_supervision = False
                model.use_speaker_grl = False
                print("[train] STAGE 2: ASR/GRL losses disabled (content is frozen; "
                      "--keep_aux_losses to compute them anyway, e.g. for monitoring)")
        else:
            print("[train] STAGE 2: content path is trainable (--no_freeze_content)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] Parameters: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")

    lit_model = LiveVoiceLightningModule(config, model)
    dm = LibriTTSDataModule(config) if args.dataset == "libritts" else VCTKDataModule(config)

    log_dir = os.path.join(config.output_dir, "logs")
    ckpt_dir = os.path.join(config.output_dir, "checkpoints", config.exp_name)
    if args.use_wandb and WandbLogger is not None:
        # Avoid uploading Lightning progress-bar / stdout to W&B (filestream blowups).
        os.environ.setdefault("WANDB_CONSOLE", "off")
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=config.exp_name,
            save_dir=log_dir,
        )
        print(f"[train] Logger: WandbLogger(project={args.wandb_project}, console=off)")
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
        # Best WER (logged in on_train_epoch_end as val/wer_full_epoch_mean)
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="wer_best",
            monitor="val/wer_full_epoch_mean",
            mode="min",
            save_top_k=1,
            enable_version_counter=False,
            save_last=False,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        ),
        # EveryNEpochCheckpoint(dirpath=ckpt_dir, every_n_epochs=5),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer_kwargs = dict(
        max_epochs=config.max_epochs,
        precision=config.precision,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.max_grad_norm,
        deterministic=False,
    )
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = int(args.max_steps)
        print(f"[train] max_steps={args.max_steps}")

    trainer = L.Trainer(**trainer_kwargs)

    trainer.fit(lit_model, dm, ckpt_path=args.resume_from)
    print("[train] Done.")


if __name__ == "__main__":
    main()
