"""Lightning modules for LiveVoice training.

Two modules:
  UnconditionalLightningModule  — sanity-check, no conditioning
  LiveVoiceLightningModule      — full VC: speaker + HuBERT content (+ optional prosody)
"""
from __future__ import annotations

import math
import os
import hashlib
import random
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
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        for k, v in per_book.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, sync_dist=True)

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
        loss, _ = _cross_entropy_loss(
            out["all_logits"], out["delayed_targets"],
            self.config.codebook_loss_weights,
        )
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

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
        self.use_mimi_cache = bool(getattr(config, "use_mimi_cache", True)) and str(
            getattr(config, "codec", "dac")
        ).lower() == "mimi"
        self.mimi_cache_dir = str(
            getattr(config, "mimi_cache_dir", os.path.join(config.output_dir, "mimi_cache"))
        )
        self._hop = int(getattr(self.model.dac_model, "hop_length", getattr(config, "dac_hop_length", 320)))
        self._target_len_samples = int(round(float(config.audio_duration) * float(config.sample_rate)))
        self._val_ref_bank = None  # dict with keys: "speaker_id" (list[str]), "reference_audio" (list[Tensor])
        self._val_ref_bank_epoch = None
        self._whisper_model = None
        self._whisper_loaded = False

    def _word_wer(self, hyp: str, ref: str) -> float:
        hyp_w = [w for w in hyp.strip().lower().split() if w]
        ref_w = [w for w in ref.strip().lower().split() if w]
        if not ref_w:
            return float("nan")
        dp = list(range(len(hyp_w) + 1))
        for i, rw in enumerate(ref_w, start=1):
            prev = dp[0]
            dp[0] = i
            for j, hw in enumerate(hyp_w, start=1):
                cur = dp[j]
                cost = 0 if rw == hw else 1
                dp[j] = min(
                    dp[j] + 1,      # delete
                    dp[j - 1] + 1,  # insert
                    prev + cost,    # substitute
                )
                prev = cur
        return float(dp[-1]) / float(len(ref_w))

    def _get_whisper(self):
        if self._whisper_loaded:
            return self._whisper_model
        self._whisper_loaded = True
        try:
            import whisper  # type: ignore
        except Exception:
            self._whisper_model = None
            return None
        name = str(getattr(self.config, "wer_whisper_model", "base"))
        device = str(getattr(self.config, "wer_device", "cpu"))
        try:
            self._whisper_model = whisper.load_model(name, device=device)
        except Exception:
            self._whisper_model = None
        return self._whisper_model

    def on_validation_epoch_start(self):
        # Build a small reference bank so we can always pick a different-speaker ref
        # even when val batch itself contains a single speaker.
        if not getattr(self.trainer, "is_global_zero", True):
            return
        try:
            dm = getattr(self.trainer, "datamodule", None)
            ds = getattr(dm, "val_dataset", None) if dm is not None else None
            if ds is None:
                return
            epoch = int(getattr(self.trainer, "current_epoch", 0))
            if self._val_ref_bank_epoch == epoch and self._val_ref_bank is not None:
                return

            # Sample enough items to likely cover multiple speakers
            bank_size = int(getattr(self.config, "val_ref_bank_size", 128))
            bank_size = max(16, bank_size)
            n_items = len(ds)
            idxs = [random.randrange(0, n_items) for _ in range(bank_size)]
            spk_ids: list[str] = []
            ref_wavs: list[torch.Tensor] = []
            for ix in idxs:
                item = ds[ix]
                spk = item.get("speaker_id", None)
                wav = item.get("reference_audio", None)
                if spk is None or wav is None:
                    continue
                if isinstance(wav, torch.Tensor):
                    ref_wavs.append(wav.detach().float().cpu())
                    spk_ids.append(str(spk))

            uniq = len(set(spk_ids))
            if uniq < 2:
                raise RuntimeError(
                    "val_ref_bank has <2 unique speakers. "
                    "Cannot log diff-speaker samples (val set might be single-speaker)."
                )

            self._val_ref_bank = {"speaker_id": spk_ids, "reference_audio": ref_wavs}
            self._val_ref_bank_epoch = epoch
        except Exception:
            # Don't break validation for logging-only issues
            self._val_ref_bank = None
            self._val_ref_bank_epoch = None

    def _pick_diff_speaker_refs(self, target_speakers: list[str], n: int, device: torch.device) -> torch.Tensor:
        """Return (n, T) reference_audio from speakers != target_speakers[i]."""
        if self._val_ref_bank is None:
            raise RuntimeError("val_ref_bank not built; cannot pick diff-speaker refs.")
        bank_spk: list[str] = self._val_ref_bank["speaker_id"]
        bank_wav: list[torch.Tensor] = self._val_ref_bank["reference_audio"]
        out = []
        for i in range(n):
            spk_i = target_speakers[i]
            candidates = [j for j, spk in enumerate(bank_spk) if spk != spk_i]
            if not candidates:
                raise RuntimeError(f"No diff-speaker reference available for speaker_id={spk_i}")
            j = random.choice(candidates)
            out.append(bank_wav[j])
        return torch.stack(out, dim=0).to(device)

    def _cache_key(self, path: str) -> str:
        codec_name = str(getattr(self.config, "codec", "dac")).lower()
        raw = f"{codec_name}|sr={self.config.sample_rate}|path={path}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_file(self, kind: str, path: str) -> str:
        # kind: "target_codes" | "reference_z"
        return os.path.join(self.mimi_cache_dir, kind, f"{self._cache_key(path)}.pt")

    def _slice_cached_target_codes(self, full_codes: torch.Tensor, start_sample: int, n_frames: int):
        # full_codes: (K, T_full)
        if full_codes.dim() != 2:
            return None
        start_f = int(start_sample) // max(1, self._hop)
        end_f = start_f + n_frames
        if start_f < 0 or end_f > full_codes.shape[1]:
            return None
        return full_codes[:, start_f:end_f]

    def _slice_cached_reference_z(self, full_z: torch.Tensor, start_sample: int, n_frames: int):
        # full_z: (T_full, D)
        if full_z.dim() != 2:
            return None
        start_f = int(start_sample) // max(1, self._hop)
        end_f = start_f + n_frames
        if start_f < 0 or end_f > full_z.shape[0]:
            return None
        return full_z[start_f:end_f, :]

    def _load_target_codes_or_fallback(self, tgt: torch.Tensor, content_paths, content_starts):
        K = int(self.config.n_codebooks_predict)
        if not self.use_mimi_cache or content_paths is None or content_starts is None:
            with torch.no_grad():
                return self.model.dac_model.encode(tgt)[:, :K, :]

        n_frames = int(round(self._target_len_samples / max(1, self._hop)))
        out = []
        for i in range(tgt.size(0)):
            p = str(content_paths[i])
            s = int(content_starts[i])
            fpath = self._cache_file("target_codes", p)
            chunk = None
            if os.path.exists(fpath):
                try:
                    obj = torch.load(fpath, map_location="cpu", weights_only=True)
                    full_codes = obj["codes"] if isinstance(obj, dict) and "codes" in obj else obj
                    chunk = self._slice_cached_target_codes(full_codes, s, n_frames)
                except Exception:
                    chunk = None

            if chunk is None:
                with torch.no_grad():
                    c = self.model.dac_model.encode(tgt[i:i + 1])[:, :K, :]
                out.append(c)
            else:
                out.append(chunk.unsqueeze(0).to(tgt.device, dtype=torch.long))
        return torch.cat(out, dim=0)

    def _load_reference_z_or_fallback(self, ref: torch.Tensor, ref_paths, ref_starts):
        if not self.use_mimi_cache or ref_paths is None or ref_starts is None:
            with torch.no_grad():
                _, z = self.model.dac_model.encode_continuous(ref)
            return z

        n_frames = int(round(self._target_len_samples / max(1, self._hop)))
        out = []
        for i in range(ref.size(0)):
            p = str(ref_paths[i])
            s = int(ref_starts[i])
            fpath = self._cache_file("reference_z", p)
            chunk = None
            if os.path.exists(fpath):
                try:
                    obj = torch.load(fpath, map_location="cpu", weights_only=True)
                    full_z = obj["z"] if isinstance(obj, dict) and "z" in obj else obj
                    chunk = self._slice_cached_reference_z(full_z, s, n_frames)
                except Exception:
                    chunk = None

            if chunk is None:
                with torch.no_grad():
                    _, z = self.model.dac_model.encode_continuous(ref[i:i + 1])
                out.append(z)
            else:
                out.append(chunk.unsqueeze(0).to(ref.device, dtype=ref.dtype))
        return torch.cat(out, dim=0)

    def training_step(self, batch, batch_idx):
        ref = batch["reference_audio"]
        ctn = batch["content_audio"]
        tgt = batch["target_audio"]
        content_feats = batch.get("content_hubert", None)  # (B, T, 768) or None
        content_paths = batch.get("content_path", None)
        content_starts = batch.get("content_start_sample", None)
        ref_paths = batch.get("ref_path", None)
        ref_starts = batch.get("ref_start_sample", None)

        codes = self._load_target_codes_or_fallback(tgt, content_paths, content_starts)
        ref_z = self._load_reference_z_or_fallback(ref, ref_paths, ref_starts)

        out = self.model(
            ref, ctn, codes, prosody_audio=None, content_feats=content_feats, reference_z=ref_z
        )
        loss, per_book = _cross_entropy_loss(
            out["all_logits"], out["delayed_targets"],
            self.config.codebook_loss_weights,
        )
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        for k, v in per_book.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ref = batch["reference_audio"]
        ctn = batch["content_audio"]
        tgt = batch["target_audio"]
        content_feats = batch.get("content_hubert", None)

        content_paths = batch.get("content_path", None)
        content_starts = batch.get("content_start_sample", None)
        ref_paths = batch.get("ref_path", None)
        ref_starts = batch.get("ref_start_sample", None)

        codes = self._load_target_codes_or_fallback(tgt, content_paths, content_starts)
        ref_z = self._load_reference_z_or_fallback(ref, ref_paths, ref_starts)
        out = self.model(ref, ctn, codes, content_feats=content_feats, reference_z=ref_z)
        loss, _ = _cross_entropy_loss(
            out["all_logits"], out["delayed_targets"],
            self.config.codebook_loss_weights,
        )
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            if not getattr(self.trainer, "is_global_zero", True):
                return
            n = min(ref.size(0), int(getattr(self.config, "num_audio_log_samples", 4)))
            # Randomize which samples we log each validation
            perm = torch.randperm(ref.size(0), device=ref.device)
            sel = perm[:n].tolist()

            ref_same = ref[sel]
            ctn_same = ctn[sel]
            content_texts = batch.get("content_text", None)
            texts_sel = None
            if isinstance(content_texts, list) and len(content_texts) == ref.size(0):
                texts_sel = [content_texts[i] for i in sel]
            self._log_vc_sample(ref_same, ctn_same, tag_prefix="val", content_texts=texts_sel)

            # Also log a "diff speaker reference" condition for easier listening.
            speaker_ids = batch.get("speaker_id", None)  # list[str] from collate_fn
            if isinstance(speaker_ids, list) and len(speaker_ids) == ref.size(0):
                spk_sel = [str(speaker_ids[i]) for i in sel]
                ref_diff = self._pick_diff_speaker_refs(spk_sel, n=len(spk_sel), device=ref.device)
                self._log_vc_sample(ref_diff, ctn_same, tag_prefix="Media/val_diff_spk", content_texts=texts_sel)

    @torch.no_grad()
    def _log_vc_sample(
        self,
        ref_audio: torch.Tensor,
        ctn_audio: torch.Tensor,
        tag_prefix: str = "val",
        content_texts: list[str | None] | None = None,
    ):
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

        # ── WER (Whisper transcription) ─────────────────────────────────
        if (
            bool(getattr(self.config, "log_val_wer", False))
            and content_texts is not None
            and getattr(self.trainer, "is_global_zero", True)
        ):
            w = self._get_whisper()
            if w is None:
                raise RuntimeError(
                    "log_val_wer=True but Whisper model is unavailable. "
                    "Install openai-whisper and ensure ffmpeg is available."
                )
            try:
                import torchaudio
                wavs = gen_audio.detach().float().cpu()
                if sr != 16000:
                    wavs = torchaudio.functional.resample(wavs, sr, 16000)
                wers = []
                for i in range(wavs.size(0)):
                    ref_txt = content_texts[i] if i < len(content_texts) else None
                    if not ref_txt:
                        continue
                    hyp_txt = w.transcribe(wavs[i].numpy(), fp16=False)["text"]
                    wers.append(self._word_wer(hyp_txt, ref_txt))
                if not wers:
                    raise RuntimeError(
                        "log_val_wer=True but no valid reference texts were found in this validation batch."
                    )
                wer_mean = float(sum(wers) / len(wers))
                exp = getattr(self.logger, "experiment", None)
                if exp is not None and hasattr(exp, "log"):
                    exp.log({f"{tag_prefix}/wer_mean": wer_mean}, step=self.global_step)
                else:
                    self.log(f"{tag_prefix}/wer_mean", wer_mean, on_step=False, on_epoch=True, sync_dist=False)
            except Exception as e:
                raise RuntimeError(f"WER logging failed: {e}") from e
        elif bool(getattr(self.config, "log_val_wer", False)) and getattr(self.trainer, "is_global_zero", True):
            raise RuntimeError(
                "log_val_wer=True but content_texts is missing. "
                "Ensure dataset returns transcript text (e.g., LibriTTS normalized text)."
            )

        # ── Audio logging ────────────────────────────────────────────────
        _log_audio_batch(self.logger, f"{tag_prefix}/generated_vc", gen_audio, sr, self.global_step)
        _log_audio_batch(self.logger, f"{tag_prefix}/reference_audio", ref_audio, sr, self.global_step)
        _log_audio_batch(self.logger, f"{tag_prefix}/content_audio", ctn_audio, sr, self.global_step)

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
