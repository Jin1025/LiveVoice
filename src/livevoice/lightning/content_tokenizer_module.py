"""Stage-1 Lightning module: train the content tokenizer with ASR + GRL ONLY.

No codec, no VC decoder, no reconstruction CE — that is the whole point. In joint
training the codec CE back-propagates into the content path and, because training is
same-speaker reconstruction, it actively wants speaker identity there, fighting the GRL.
Here that opposing force simply does not exist, so the GRL can remove speaker while the
ASR loss holds on to phonetic content.

Logs the same metric names as the joint trainer (train/asr_loss, train/grl_loss,
train/grl_acc, train/grl_lambda) so runs are directly comparable in wandb.
"""
from __future__ import annotations

import torch
import lightning as L

from livevoice.model.speaker_grl import grl_lambda_schedule
from livevoice.utils.checkpoint import CONFIG_CKPT_KEY, config_to_ckpt_dict


class ContentTokenizerLightningModule(L.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        print(
            f"[Stage1] content tokenizer: asr={getattr(config,'use_asr_supervision',False)}"
            f"({getattr(config,'asr_supervision_type','?')}, w={getattr(config,'asr_loss_weight','?')})  "
            f"grl={getattr(config,'use_speaker_grl',False)}(w={getattr(config,'grl_loss_weight','?')}, "
            f"clusters={getattr(config,'grl_num_clusters',0) or 'per-speaker'})"
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Same rationale as the VC module: store the config so Stage-2 / probe scripts can
        # rebuild the identical content path without guessing. See utils/checkpoint.
        checkpoint[CONFIG_CKPT_KEY] = config_to_ckpt_dict(self.config)

    def _step(self, batch, stage: str):
        feats_full = batch.get("content_feats_full", None)
        full_audio = batch.get("content_full_audio", None)
        phoneme_ids = batch.get("phoneme_ids", None)
        speaker_labels = batch.get("speaker_label", None)
        have_full = feats_full is not None or full_audio is not None
        if not have_full:
            raise RuntimeError(
                "Stage-1 needs full-utterance content (cache or sw2v_full_online=True) — "
                "got neither. Check config.sw2v_features_dir / sw2v_full_online."
            )
        feats_full = feats_full.to(self.device) if feats_full is not None else None
        full_audio = full_audio.to(self.device) if full_audio is not None else None
        lens = batch["content_feats_full_len"].to(self.device)

        dev = feats_full.device if feats_full is not None else full_audio.device
        asr_w = float(getattr(self.config, "asr_loss_weight", 0.0))
        grl_w = float(getattr(self.config, "grl_loss_weight", 0.0))

        asr_l = torch.zeros((), device=dev)
        if asr_w > 0 and getattr(self.model, "use_asr_supervision", False) and phoneme_ids is not None:
            asr_l = self.model.compute_asr_supervision_loss(
                feats_full, lens, phoneme_ids.to(self.device), content_full_audio=full_audio,
            )

        grl_l = torch.zeros((), device=dev)
        grl_acc = None
        grl_lambda = 0.0
        if grl_w > 0 and getattr(self.model, "use_speaker_grl", False) and speaker_labels is not None:
            grl_lambda = grl_lambda_schedule(
                int(self.global_step),
                int(getattr(self.config, "grl_warmup_steps", 10000)),
                float(getattr(self.config, "grl_lambda_max", 1.0)),
                float(getattr(self.config, "grl_gamma", 10.0)),
                int(getattr(self.config, "grl_start_step", 0)),
            )
            grl_l, grl_acc = self.model.compute_speaker_grl_loss(
                feats_full, lens, speaker_labels.to(self.device), grl_lambda,
                content_full_audio=full_audio,
            )

        # NaN safety (CTC in particular): drop a bad term rather than poison the weights.
        nan_kw = dict(on_step=(stage == "train"), on_epoch=True, batch_size=int(lens.shape[0]))
        if not torch.isfinite(asr_l):
            asr_l = torch.zeros((), device=dev)
            self.log(f"{stage}/asr_nan", 1.0, **nan_kw)
        if not torch.isfinite(grl_l):
            grl_l = torch.zeros((), device=dev)
            self.log(f"{stage}/grl_nan", 1.0, **nan_kw)

        loss = asr_w * asr_l + grl_w * grl_l
        is_train = stage == "train"
        # Batch size is ambiguous to Lightning here (the batch holds mixed collections),
        # so pass it explicitly — otherwise epoch means are weighted by a guessed size.
        bs = int(lens.shape[0])
        log_kw = dict(on_step=is_train, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log(f"{stage}/loss", loss, prog_bar=True, **log_kw)
        if asr_w > 0:
            self.log(f"{stage}/asr_loss", asr_l.detach(), **log_kw)
        if grl_w > 0:
            self.log(f"{stage}/grl_loss", grl_l.detach(), **log_kw)
            if grl_acc is not None:
                self.log(f"{stage}/grl_acc", grl_acc.detach(), **log_kw)
            # confusion mode: grl_loss = CE + λ·conf converges to ln(K)·(1+λ), so log the
            # parts separately — clf→ln(K) = adversary at chance, conf→ln(K) = uniform.
            parts = getattr(getattr(self.model, "speaker_grl_head", None), "last_parts", None)
            if parts:
                for k, v in parts.items():
                    self.log(f"{stage}/{k}", v, **log_kw)
            # λ is a training-schedule value only (0 during val) — log it on train steps.
            if is_train:
                self.log(f"{stage}/grl_lambda", grl_lambda, on_step=True, on_epoch=False,
                         sync_dist=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, "train")
        if not torch.isfinite(loss):
            self.log("train/nonfinite_skip", 1.0, on_step=True, on_epoch=False)
            return None
        return loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
        )
        return opt
