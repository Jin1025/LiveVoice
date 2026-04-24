"""Lightning modules for LiveVoice training.

Two modules:
  UnconditionalLightningModule  — sanity-check, no conditioning
  LiveVoiceLightningModule      — full VC: speaker + HuBERT content (+ optional prosody)
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import lightning as L

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


# ─────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────

def _cross_entropy_loss(all_logits, delayed_targets, weights):
    """Per-codebook cross-entropy averaged over time, then weighted across codebooks.

    all_logits:      (B, T, K, V)
    delayed_targets: (B, T, K) with -100 for padding
    weights:         tuple/list of K floats
    """
    B, T, K, V = all_logits.shape
    total_loss = 0.0
    total_weight = 0.0
    per_book = {}
    for k in range(K):
        lg = all_logits[:, :, k, :].reshape(B * T, V)
        tg = delayed_targets[:, :, k].reshape(B * T)
        lk = F.cross_entropy(lg, tg, ignore_index=-100)
        w = float(weights[k]) if k < len(weights) else 1.0
        total_loss = total_loss + w * lk
        total_weight += w
        per_book[f"loss_cb{k}"] = lk.detach()
    return total_loss / total_weight, per_book


def _cosine_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay to min_lr_ratio."""
    min_lr_ratio = 0.05  # floor at 5% of peak LR (don't decay all the way to zero)
    def _lr(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


def _log_audio_batch(logger, tag: str, wav_batch: torch.Tensor, sample_rate: int, global_step: int):
    """Log a batch of audio tensors to W&B or TensorBoard."""
    try:
        wav_batch = wav_batch.detach().float().cpu().clamp(-1, 1)
        exp = getattr(logger, "experiment", None)
        if exp is None:
            return
        if wandb is not None and hasattr(exp, "log"):
            audios = [
                wandb.Audio(wav_batch[i].numpy(), sample_rate=sample_rate, caption=f"{tag}_{i}")
                for i in range(wav_batch.size(0))
            ]
            exp.log({tag: audios}, step=global_step)
        elif hasattr(exp, "add_audio"):
            for i in range(wav_batch.size(0)):
                exp.add_audio(f"{tag}/{i}", wav_batch[i], global_step=global_step, sample_rate=sample_rate)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  Unconditional module
# ─────────────────────────────────────────────

class UnconditionalLightningModule(L.LightningModule):
    """Train UnconditionalModel: decoder-only AR over DAC codes."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def training_step(self, batch, batch_idx):
        target_audio = batch["target_audio"]
        with torch.no_grad():
            codes = self.model.dac_model.encode(target_audio)  # (B, K_full, T)
        codes = codes[:, : self.config.n_codebooks_predict, :]

        out = self.model(codes)
        loss, per_book = _cross_entropy_loss(
            out["all_logits"], out["delayed_targets"],
            self.config.codebook_loss_weights,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        for k, v in per_book.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False)

        if (
            self.global_step > 0
            and self.global_step % 1000 == 0
            and getattr(self.trainer, "is_global_zero", True)
        ):
            self._log_generated_sample(tag="Media/generated_uncond")
        return loss

    def validation_step(self, batch, batch_idx):
        target_audio = batch["target_audio"]
        with torch.no_grad():
            codes = self.model.dac_model.encode(target_audio)
        codes = codes[:, : self.config.n_codebooks_predict, :]

        out = self.model(codes)
        loss, per_book = _cross_entropy_loss(
            out["all_logits"], out["delayed_targets"],
            self.config.codebook_loss_weights,
        )
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in per_book.items():
            self.log(f"val/{k}", v, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self._log_generated_sample()

    @torch.no_grad()
    def _log_generated_sample(self, tag: str = "val/generated_uncond"):
        num_samples = int(getattr(self.config, "num_audio_log_samples", 4))
        max_steps = int(
            float(getattr(self.config, "audio_duration", 4.0))
            * float(getattr(self.config, "dac_sample_rate", self.config.sample_rate))
            / float(getattr(self.config, "dac_hop_length", 320))
        )
        max_steps = max(1, max_steps)
        codes = self.model.generate(
            batch_size=num_samples,
            max_steps=max_steps,
            temperature=float(getattr(self.config, "temperature", 1.0)),
            top_p=float(getattr(self.config, "top_p", 0.9)),
        )
        audio = self.model.decode_to_audio(codes)  # (N, T)
        _log_audio_batch(self.logger, tag, audio, int(self.config.sample_rate), self.global_step)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # estimated_stepping_batches: Lightning 2.x property that correctly accounts
        # for epochs × batches per epoch × accumulate_grad_batches
        try:
            total_steps = self.trainer.estimated_stepping_batches
        except Exception:
            total_steps = 200_000
        warmup = int(getattr(self.config, "warmup_steps", 1000))
        print(f"[LR schedule] total_steps={total_steps}  warmup={warmup}")
        sched = _cosine_lr_schedule(opt, warmup, total_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


# ─────────────────────────────────────────────
#  Full VC module
# ─────────────────────────────────────────────

class LiveVoiceLightningModule(L.LightningModule):
    """Train LiveVoiceModel: speaker cross-attn + HuBERT content conditioning."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def training_step(self, batch, batch_idx):
        ref = batch["reference_audio"]
        ctn = batch["content_audio"]
        tgt = batch["target_audio"]
        content_feats = batch.get("content_hubert", None)  # (B, T, 768) or None

        with torch.no_grad():
            codes = self.model.dac_model.encode(tgt)
        codes = codes[:, : self.config.n_codebooks_predict, :]

        out = self.model(ref, ctn, codes, prosody_audio=None, content_feats=content_feats)
        loss, per_book = _cross_entropy_loss(
            out["all_logits"], out["delayed_targets"],
            self.config.codebook_loss_weights,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        for k, v in per_book.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        ref = batch["reference_audio"]
        ctn = batch["content_audio"]
        tgt = batch["target_audio"]
        content_feats = batch.get("content_hubert", None)

        with torch.no_grad():
            codes = self.model.dac_model.encode(tgt)
        codes = codes[:, : self.config.n_codebooks_predict, :]

        out = self.model(ref, ctn, codes, content_feats=content_feats)
        loss, per_book = _cross_entropy_loss(
            out["all_logits"], out["delayed_targets"],
            self.config.codebook_loss_weights,
        )
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in per_book.items():
            self.log(f"val/{k}", v, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            n = min(ref.size(0), int(getattr(self.config, "num_audio_log_samples", 4)))
            self._log_vc_sample(ref[:n], ctn[:n])

    @torch.no_grad()
    def _log_vc_sample(self, ref_audio: torch.Tensor, ctn_audio: torch.Tensor):
        codes = self.model.generate(
            reference_audio=ref_audio,
            content_audio=ctn_audio,
            temperature=float(getattr(self.config, "temperature", 1.0)),
            top_p=float(getattr(self.config, "top_p", 0.9)),
        )
        gen_audio = self.model.decode_to_audio(codes)  # (N, T)
        sr = int(self.config.sample_rate)

        # ── Speaker similarity (DAC-z cosine proxy) ──────────────────────
        # Compare mean-pooled DAC continuous z of generated vs reference.
        # This is a cheap proxy: DAC z captures timbre in its first few dimensions.
        try:
            _, z_gen = self.model.dac_model.encode_continuous(gen_audio)  # (N, T, D)
            _, z_ref = self.model.dac_model.encode_continuous(ref_audio)  # (N, T, D)
            z_gen_pool = z_gen.mean(dim=1)  # (N, D)
            z_ref_pool = z_ref.mean(dim=1)  # (N, D)
            spk_sim = F.cosine_similarity(z_gen_pool, z_ref_pool, dim=-1)  # (N,)
            self.log("val/spk_sim_dac", spk_sim.mean(), on_epoch=True, sync_dist=True)
        except Exception:
            pass

        # ── Audio logging ────────────────────────────────────────────────
        _log_audio_batch(self.logger, "val/generated_vc", gen_audio, sr, self.global_step)
        _log_audio_batch(self.logger, "val/reference_audio", ref_audio, sr, self.global_step)
        _log_audio_batch(self.logger, "val/content_audio", ctn_audio, sr, self.global_step)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        try:
            total_steps = self.trainer.estimated_stepping_batches
        except Exception:
            total_steps = 300_000
        warmup = int(getattr(self.config, "warmup_steps", 2000))
        print(f"[LR schedule] total_steps={total_steps}  warmup={warmup}")
        sched = _cosine_lr_schedule(opt, warmup, total_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }
