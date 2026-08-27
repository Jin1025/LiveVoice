"""Train full LiveVoice VC model.

Usage (inside docker `yejin2`, conda `sound`):
    CUDA_VISIBLE_DEVICES=2 python scripts/train.py --exp_name base_vctk
"""
import argparse
import os
import sys
from pathlib import Path

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
    CepstralExtractor,
    LiveVoiceModel,
)
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.utils.checkpoint import load_model_weights_from_ckpt
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


def _CLI_OVERRIDES(args) -> dict:
    """CLI flags allowed to override LiveVoiceConfig. Values of None are dropped."""
    return {
        "exp_name": args.exp_name,
        "train_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
    }


def parse_args():
    """Operational flags only.

    Everything that defines the MODEL or the RUN lives in LiveVoiceConfig and nowhere else.
    This used to be a mirror of the config, and the mirror won: `--use_prosody` was declared
    `action="store_true"`, so argparse handed its own default (False) to LiveVoiceConfig on
    every launch and `use_prosody = True` in config.py did nothing. Two runs reached 163k and
    99k steps before anyone noticed. An argument only belongs here if LiveVoiceConfig has no
    field for it — otherwise there are two sources of truth and the CLI silently wins.
    """
    p = argparse.ArgumentParser(description=__doc__)
    # The two exceptions, and the reason they are safe: default=None, so argparse never hands
    # LiveVoiceConfig a value that was not typed on the command line. `action="store_true"` is
    # what broke --use_prosody — its default False was passed in on every launch and overwrote
    # config.py. Any future override belongs in _CLI_OVERRIDES with default=None, not here.
    p.add_argument("--exp_name", type=str, default='base',
                   help="override config.exp_name (run name / checkpoint dir)")
    p.add_argument("--batch_size", dest="train_batch_size", type=int, default=8,
                   help="override config.train_batch_size")
    p.add_argument("--dataset", type=str, default="libritts", choices=["vctk", "libritts"],
                   help="which dataset to build the datamodule from")
    p.add_argument("--max_steps", type=int, default=400000,
                   help="stop after this many optimizer steps (Trainer arg, not a config field)")
    p.add_argument("--lr", dest="learning_rate", type=float, default=None,
                   help="override config.learning_rate (e.g. --lr 5e-5)")
    p.add_argument("--resume_from", type=str, default=None,
                   help="checkpoint to resume the Trainer from (restores optimizer, step, "
                        "epoch — use only to continue an INTERRUPTED run of the same config)")
    p.add_argument("--init_from", type=str, default=None,
                   help="load model weights only and start a fresh run at step 0. This is the "
                        "fine-tune entry point: --resume_from would try to restore an optimizer "
                        "state whose parameter groups no longer match once a new module (e.g. "
                        "the cepstral extractor) adds parameters. Missing keys are reported and "
                        "left at their fresh init.")
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
    return p.parse_args()



# The settings that silently change WHAT gets trained, printed where a run starts. A run named
# "prosody_baseline" reached 163k steps with use_prosody=False because the flag lives in
# config.py and nothing echoed it back; the checkpoint was only distinguishable from the
# previous ablation by reading its stored config afterwards. Anything on this list either
# alters the model's inputs or has already been set wrong once.
_RUN_SETTINGS = [
    ("content",  ["content_source", "content_cmn", "content_cmn_prior_frames",
                  "content_cmn_in_cache", "zipformer_align_pad_frames",
                  "use_content_perturbation"]),
    ("cepstral", ["use_cepstral", "cepstral_spec", "cepstral_hidden_dim"]),
    ("mpm",      ["use_mpm", "mpm_ckpt", "mpm_freeze"]),
    ("prosody",  ["use_prosody", "pitch_method", "pitch_normalize", "pitch_prior_frames",
                  "pitch_prior_hz", "pitch_fmin", "pitch_fmax", "use_random_median_filter"]),
    ("prompt",   ["codec_prompt_content", "codec_prompt_loss_weight",
                  "speaker_encoder_type", "speaker_conditioning", "audio_duration"]),
    ("stream",   ["use_delay_pattern", "delay_cap", "n_codebooks_predict"]),
    ("aux loss", ["use_asr_supervision", "asr_supervision_type", "asr_loss_weight",
                  "use_speaker_grl", "grl_objective", "grl_lambda_max", "grl_num_clusters"]),
    ("optim",    ["precision", "train_batch_size", "learning_rate", "max_epochs"]),
]


def _print_run_settings(config, args) -> None:
    """Echo the run-defining settings, flagging the ones that are OFF.

    Groups whose main switch is off are collapsed to one line: an ablation is defined as much
    by what it disables as by what it enables, and a wall of irrelevant sub-settings is how the
    important line gets skimmed past.
    """
    gates = {"prosody": "use_prosody", "cepstral": "use_cepstral", "mpm": "use_mpm", "aux loss": None}
    print("[train] ---- run settings " + "-" * 46)
    for group, keys in _RUN_SETTINGS:
        gate = gates.get(group)
        if gate is not None and not bool(getattr(config, gate, False)):
            print(f"[train]   {group:9s} {gate}=False  (rest of group inactive)")
            continue
        vals = []
        for k in keys:
            if not hasattr(config, k):
                continue
            v = getattr(config, k)
            vals.append(f"{k}={v!r}")
        for i in range(0, len(vals), 3):
            head = group if i == 0 else ""
            print(f"[train]   {head:9s} " + "  ".join(vals[i:i + 3]))
    print(f"[train]   output    exp_name={config.exp_name}  dir={config.output_dir}")
    print("[train] " + "-" * 65)


def main():
    args = parse_args()
    config = LiveVoiceConfig(**{k: v for k, v in _CLI_OVERRIDES(args).items() if v is not None})
    L.seed_everything(config.seed)

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

    _print_run_settings(config, args)

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

    cepstral_extractor = None
    if getattr(config, "use_cepstral", False):
        print(f"[train] Building CepstralExtractor (spec={config.cepstral_spec})...")
        cepstral_extractor = CepstralExtractor(config)

    prosody_extractor = None
    if config.use_prosody:
        print("[train] Building ProsodyExtractor...")
        prosody_extractor = ProsodyExtractor(config)

    mpm_extractor = None
    if bool(getattr(config, "use_mpm", False)):
        from livevoice.model.causal_mpm import CausalMPM, CausalMPMConfig
        import json as _json
        mpm_ckpt = str(getattr(config, "mpm_ckpt", ""))
        if mpm_ckpt:
            mpm_dir = str(Path(mpm_ckpt).parent)
            cfg_path = Path(mpm_dir) / "config.json"
            mpm_cfg = CausalMPMConfig(**_json.load(open(cfg_path))) if Path(cfg_path).exists() else CausalMPMConfig()
            # the pretrain config.json predates this flag; the LiveVoice config owns it.
            mpm_cfg.causal_window = bool(getattr(config, "mpm_causal_window", True))
            mpm_extractor = CausalMPM(mpm_cfg)
            ckpt_data = torch.load(mpm_ckpt, map_location="cpu", weights_only=False)
            mpm_extractor.load_state_dict(ckpt_data["model"])
            if bool(getattr(config, "mpm_freeze", True)):
                for p in mpm_extractor.parameters():
                    p.requires_grad_(False)
                mpm_extractor.eval()
            print(f"[train] MPM loaded: {mpm_ckpt} "
                  f"({sum(p.numel() for p in mpm_extractor.parameters())/1e6:.1f}M params, "
                  f"freeze={getattr(config, 'mpm_freeze', True)})")
        else:
            print("[train] WARNING: use_mpm=True but mpm_ckpt is empty — MPM disabled")

    # Speaker-GRL adversary needs the class count (train-split speaker vocab) at model
    # build time; the dataset later builds the SAME deterministic mapping for labels.
    if bool(getattr(config, "use_speaker_grl", False)) and args.dataset == "libritts":
        from livevoice.data.speaker_vocab import build_libritts_grl_label_map
        _, config.grl_num_speakers = build_libritts_grl_label_map(config)
        kind = "clusters" if int(getattr(config, "grl_num_clusters", 0)) > 0 else "speakers"
        print(f"[train] speaker-GRL: grl_num_speakers={config.grl_num_speakers} ({kind})")

    print("[train] Building LiveVoiceModel...")
    model = LiveVoiceModel(config, codec_model, content_extractor, prosody_extractor,
                           cepstral_extractor, mpm_extractor=mpm_extractor)

    if args.init_from:
        print(f"[train] init_from: loading model weights from {args.init_from}")
        missing, unexpected = load_model_weights_from_ckpt(model, args.init_from,
                                                          log_prefix="[train]")
        new_params = [k for k in missing if not k.startswith("codec_model.")]
        if new_params:
            print(f"[train]   {len(new_params)} parameter(s) not in the checkpoint — these are "
                  f"the newly added modules and start from fresh init:")
            for k in new_params[:12]:
                print(f"[train]     + {k}")
            if len(new_params) > 12:
                print(f"[train]     ... and {len(new_params) - 12} more")
        if unexpected:
            print(f"[train]   {len(unexpected)} checkpoint key(s) unused (architecture changed): "
                  f"{unexpected[:6]}")

    if config.compile:
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
