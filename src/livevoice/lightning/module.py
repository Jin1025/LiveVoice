"""Lightning modules for LiveVoice training.

Two modules:
  UnconditionalLightningModule  — sanity-check, no conditioning
  LiveVoiceLightningModule      — full VC: speaker + HuBERT content (+ optional prosody)
"""
from __future__ import annotations

import math
import os
import re
import hashlib
import random
from pathlib import Path

import soundfile as sf
import librosa
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


def _z_loss(all_logits, delayed_targets):
    """PaLM-style logsumexp regularizer averaged over codebooks.

    Penalizes logit magnitudes (logsumexp drift from 0) to prevent
    over-confident codebook distributions. Stabilises training and
    helps sampling diversity. Masks out -100 padding positions.
    """
    B, T, K, _V = all_logits.shape
    z = 0.0
    for k in range(K):
        lg = all_logits[:, :, k, :]                        # (B, T, V)
        valid = (delayed_targets[:, :, k] != -100).float() # (B, T)
        ls = torch.logsumexp(lg, dim=-1)                   # (B, T)
        z = z + ((ls ** 2) * valid).sum() / valid.sum().clamp(min=1)
    return z / K


def _latent_regression_loss(all_logits, delayed_targets, codebook_vectors_list):
    """MSE between expected latent (probs @ codebook_table) and target latent.

    Treats codebook entries as points in a continuous space; near-misses in
    latent space cost less than CE. Reduces metallic/buzzy artefacts because
    the model can hedge across acoustically-similar tokens without being
    harshly penalised by CE.

    all_logits:             (B, T, K, V)
    delayed_targets:        (B, T, K) with -100 padding
    codebook_vectors_list:  list of K (V, D) tensors (already on device)
    """
    B, T, K, _V = all_logits.shape
    total = 0.0
    for k in range(K):
        cb = codebook_vectors_list[k]                  # (V, D)
        probs = F.softmax(all_logits[:, :, k, :], dim=-1)  # (B, T, V)
        pred_z = probs @ cb                            # (B, T, D)
        valid = (delayed_targets[:, :, k] != -100)     # (B, T)
        tgt_codes = delayed_targets[:, :, k].clamp(min=0)
        target_z = cb[tgt_codes]                       # (B, T, D)
        diff = ((pred_z - target_z) ** 2).mean(dim=-1) # (B, T)
        total = total + (diff * valid.float()).sum() / valid.float().sum().clamp(min=1)
    return total / K


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


def _load_full_mono_wav(path: str, target_sr: int) -> torch.Tensor:
    """Load entire wav, resample to target_sr, peak-normalize (no window crop)."""
    import torchaudio

    try:
        with sf.SoundFile(path) as f:
            audio_np = f.read(dtype="float32", always_2d=True)
            sr = int(f.samplerate)
        audio = torch.from_numpy(audio_np).float().mean(dim=1)
    except Exception:
        audio_np, sr = librosa.load(path, sr=None, mono=True)
        audio = torch.from_numpy(audio_np.astype("float32"))
        sr = int(sr)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio / (audio.abs().max() + 1e-8)


def _read_libritts_normalized_text(wav_path: str) -> str | None:
    txt_path = Path(wav_path).with_suffix(".normalized.txt")
    if not txt_path.exists():
        return None
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        return text if text else None
    except Exception:
        return None


def _libritts_pick_ref_path(ds, content_wav_path: str, utt_id: str, spk: str) -> str:
    """Same pairing rule as LibriTTSDataset.__getitem__ (reference utterance path)."""
    spk_utts = ds.speaker_utts[spk]
    pairing = str(getattr(ds, "pairing", "same_speaker"))
    split = str(getattr(ds, "split", "train"))
    if pairing == "reconstruct":
        return content_wav_path
    if split == "train":
        candidates = [(p, u) for p, u in spk_utts if p != content_wav_path]
        if not candidates:
            return content_wav_path
        return random.choice(candidates)[0]
    ref_path, _ = next(
        ((p, u) for p, u in spk_utts if p != content_wav_path),
        (content_wav_path, utt_id),
    )
    return ref_path


def _libritts_pick_diff_speaker_ref_path(ds, spk: str, rng: random.Random) -> str | None:
    """Pick a reference utterance from a DIFFERENT speaker than `spk`.

    Used for cross-speaker VC evaluation (WER + speaker-similarity measured on the
    same converted audio). Returns None if no other speaker is available.
    """
    other_spks = [s for s in ds.speaker_utts.keys() if s != spk]
    if not other_spks:
        return None
    ref_spk = rng.choice(other_spks)
    utts = ds.speaker_utts.get(ref_spk, [])
    if not utts:
        return None
    return rng.choice(utts)[0]


# ─────────────────────────────────────────────
#  Unconditional module
# ─────────────────────────────────────────────

class UnconditionalLightningModule(L.LightningModule):
    """Train UnconditionalModel: decoder-only AR over codec codes."""

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def training_step(self, batch, batch_idx):
        target_audio = batch["target_audio"]
        with torch.no_grad():
            codes = self.model.codec_model.encode(target_audio)  # (B, K_full, T)
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
            codes = self.model.codec_model.encode(target_audio)
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
        codec = self.model.codec_model
        frames_per_sec = float(codec.sample_rate) / float(codec.hop_length)
        max_steps = max(
            1,
            int(float(getattr(self.config, "audio_duration", 4.0)) * frames_per_sec),
        )
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

    # Lazy-loaded eval-only modules (WER Whisper, val spk-sim ECAPA) must not be in ckpt.
    _CKPT_EXCLUDE_PREFIXES = ("_whisper_model.", "_spk_encoder.")

    @classmethod
    def _strip_eval_only_modules_from_state_dict(cls, state_dict: dict) -> dict:
        return {
            k: v
            for k, v in state_dict.items()
            if not any(k.startswith(p) for p in cls._CKPT_EXCLUDE_PREFIXES)
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint.get("state_dict")
        if isinstance(sd, dict):
            checkpoint["state_dict"] = self._strip_eval_only_modules_from_state_dict(sd)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        sd = checkpoint.get("state_dict")
        if isinstance(sd, dict):
            checkpoint["state_dict"] = self._strip_eval_only_modules_from_state_dict(sd)

    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self.use_mimi_cache = bool(getattr(config, "use_mimi_cache", True)) and str(
            getattr(config, "codec", "mimi")
        ).lower() == "mimi"
        self.mimi_cache_dir = str(
            getattr(config, "mimi_cache_dir", os.path.join(config.output_dir, "mimi_cache"))
        )
        self._hop = int(getattr(self.model.codec_model, "hop_length"))
        self._target_len_samples = int(round(float(config.audio_duration) * float(config.sample_rate)))
        self._val_ref_bank = None  # dict with keys: "speaker_id" (list[str]), "reference_audio" (list[Tensor])
        self._val_ref_bank_epoch = None
        self._whisper_model = None
        self._whisper_loaded = False
        self._spk_encoder = None
        self._spk_encoder_loaded = False

        # Announce the configured val spk_sim encoder at construction so it is visible
        # immediately at startup (the actual load stays lazy — first on_train_epoch_end —
        # to avoid holding WavLM-large on the GPU during training). The load-time line
        # ("[spk_sim] val speaker encoder = ...") still confirms success/failure later.
        if bool(getattr(config, "log_val_spk_sim", False)):
            _which = str(getattr(config, "val_spk_encoder", "ecapa")).lower()
            if _which == "wavlm_tdnn":
                print(f"[spk_sim] configured val_spk_encoder = wavlm_tdnn "
                      f"({getattr(config, 'wavlm_sv_variant', 'wavlm_large')}, "
                      f"ckpt={getattr(config, 'wavlm_sv_ckpt', '?')}) — loads at first epoch end")
            else:
                print(f"[spk_sim] configured val_spk_encoder = ecapa "
                      f"({getattr(config, 'speechbrain_source', '?')}) — loads at first epoch end")

        if bool(getattr(config, "use_asr_supervision", False)):
            print(
                f"[ASR-supervision] seq2seq phoneme-level  weight={getattr(config, 'asr_loss_weight', '?')}  "
                f"phoneme_cache_dir={getattr(config, 'phoneme_cache_dir', '?')}"
            )

    @staticmethod
    def _normalize_text_for_wer(s: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace.
        Without this, "Hello, world." vs "hello world" gives WER=1.0 spuriously.
        """
        s = s.lower().strip()
        s = re.sub(r"[^\w\s']", " ", s)   # keep apostrophes for don't, can't
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _word_wer(self, hyp: str, ref: str) -> float:
        hyp_w = self._normalize_text_for_wer(hyp).split()
        ref_w = self._normalize_text_for_wer(ref).split()
        if not ref_w:
            return float("nan")
        # Levenshtein on words; clamp to [0, 2.0] so a runaway hyp doesn't
        # explode the metric (still allows WER>1 which is mathematically valid).
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
            object.__setattr__(self, "_whisper_model", None)
            return None
        name = str(getattr(self.config, "wer_whisper_model", "base"))
        device = str(getattr(self.config, "wer_device", "cpu"))
        try:
            # Avoid nn.Module registration so Whisper weights are not in Lightning ckpt.
            object.__setattr__(
                self, "_whisper_model", whisper.load_model(name, device=device)
            )
        except Exception:
            object.__setattr__(self, "_whisper_model", None)
        return self._whisper_model

    def _get_spk_encoder(self):
        """Independent speaker encoder for val speaker-similarity (cached).

        config.val_spk_encoder selects the ruler:
          "ecapa"      → SpeechBrain ECAPA-TDNN  (GT SIM ~0.6)
          "wavlm_tdnn" → WavLM + TDNN x-vector   (GT SIM ~0.75; matches Vevo/Amphion)
        """
        if self._spk_encoder_loaded:
            return self._spk_encoder
        self._spk_encoder_loaded = True
        which = str(getattr(self.config, "val_spk_encoder", "ecapa")).lower()
        try:
            if which == "wavlm_tdnn":
                from livevoice.model.wavlm_speaker_encoder import (
                    WavLMTDNNSpeakerEncoder,
                )
                enc = WavLMTDNNSpeakerEncoder(self.config)
                print(f"[spk_sim] val speaker encoder = UniSpeech WavLM-TDNN "
                      f"({self.config.wavlm_sv_variant}, ckpt={self.config.wavlm_sv_ckpt})")
            else:
                from livevoice.model.speechbrain_speaker_encoder import (
                    SpeechBrainECAPASpeakerEncoder,
                )
                enc = SpeechBrainECAPASpeakerEncoder(self.config)
                print(f"[spk_sim] val speaker encoder = SpeechBrain ECAPA ({self.config.speechbrain_source})")
            enc.eval()
            object.__setattr__(self, "_spk_encoder", enc)
        except Exception as e:
            print(f"[spk_sim] speaker encoder ({which}) unavailable, skipping val/spk_sim: {e}")
            object.__setattr__(self, "_spk_encoder", None)
        return self._spk_encoder

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
        codec_name = str(getattr(self.config, "codec", "mimi")).lower()
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
                return self.model.codec_model.encode(tgt)[:, :K, :]

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
                    c = self.model.codec_model.encode(tgt[i:i + 1])[:, :K, :]
                out.append(c)
            else:
                out.append(chunk.unsqueeze(0).to(tgt.device, dtype=torch.long))
        return torch.cat(out, dim=0)

    def _load_reference_z_or_fallback(self, ref: torch.Tensor, ref_paths, ref_starts):
        if str(getattr(self.model, "speaker_encoder_type", "codec")).lower() != "codec":
            return None
        if not self.use_mimi_cache or ref_paths is None or ref_starts is None:
            with torch.no_grad():
                _, z = self.model.codec_model.encode_continuous(ref)
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
                    _, z = self.model.codec_model.encode_continuous(ref[i:i + 1])
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
        ce_loss, per_book = _cross_entropy_loss(
            out["all_logits"], out["delayed_targets"],
            self.config.codebook_loss_weights,
        )

        # ── auxiliary losses ────────────────────────────────────────
        z_w = float(getattr(self.config, "z_loss_weight", 0.0))
        lat_w = float(getattr(self.config, "latent_loss_weight", 0.0))

        z_l = torch.tensor(0.0, device=ce_loss.device)
        if z_w > 0:
            z_l = _z_loss(out["all_logits"], out["delayed_targets"])

        lat_l = torch.tensor(0.0, device=ce_loss.device)
        if lat_w > 0:
            K = int(self.config.n_codebooks_predict)
            cb_list = [getattr(self.model, f"codebook_vectors_{k}") for k in range(K)]
            lat_l = _latent_regression_loss(out["all_logits"], out["delayed_targets"], cb_list)

        # Seq2seq ASR supervision on the sw2v content bottleneck (StyleStream-style; see
        # asr_supervision.py). Shapes the content embedding itself, not the decoder —
        # runs on the FULL un-cropped utterance since the phoneme labels have no per-crop
        # timestamps. Silently 0 if the batch has no cached feats/phonemes (collate_fn →
        # None whenever any item in the batch lacks either — see
        # LibriTTSDataset._load_feats_full/_load_phonemes).
        asr_w = float(getattr(self.config, "asr_loss_weight", 0.0))
        asr_l = torch.tensor(0.0, device=ce_loss.device)
        feats_full = batch.get("content_feats_full", None)
        phoneme_ids = batch.get("phoneme_ids", None)
        if (
            asr_w > 0
            and getattr(self.model, "use_asr_supervision", False)
            and feats_full is not None
            and phoneme_ids is not None
        ):
            asr_l = self.model.compute_asr_supervision_loss(
                feats_full.to(self.device),
                batch["content_feats_full_len"].to(self.device),
                phoneme_ids.to(self.device),
            )

        loss = ce_loss + z_w * z_l + lat_w * lat_l + asr_w * asr_l

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/ce_loss", ce_loss.detach(), on_step=True, on_epoch=False, sync_dist=True)
        if z_w > 0:
            self.log("train/z_loss", z_l.detach(), on_step=True, on_epoch=False, sync_dist=True)
        if lat_w > 0:
            self.log("train/latent_loss", lat_l.detach(), on_step=True, on_epoch=False, sync_dist=True)
        if asr_w > 0:
            self.log("train/asr_loss", asr_l.detach(), on_step=True, on_epoch=False, sync_dist=True)
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
            perm = torch.randperm(ref.size(0), device=ref.device)
            sel = perm[:n].tolist()

            ref_same = ref[sel]
            ctn_same = ctn[sel]
            self._log_vc_sample(ref_same, ctn_same, tag_prefix="val")

            speaker_ids = batch.get("speaker_id", None)
            if isinstance(speaker_ids, list) and len(speaker_ids) == ref.size(0):
                spk_sel = [str(speaker_ids[i]) for i in sel]
                ref_diff = self._pick_diff_speaker_refs(spk_sel, n=len(spk_sel), device=ref.device)
                # Cross-speaker audio for listening only. The spk_sim *metric* is
                # computed over the fixed 100-sample set in on_train_epoch_end.
                self._log_vc_sample(ref_diff, ctn_same, tag_prefix="Media/val_diff_spk")

    @torch.no_grad()
    def on_train_epoch_end(self):
        """Once per training epoch: WER on full utterance (full wav vs full normalized text)."""
        if not bool(getattr(self.config, "log_val_wer", False)):
            return
        if not getattr(self.trainer, "is_global_zero", True):
            return
        dm = getattr(self.trainer, "datamodule", None)
        ds = getattr(dm, "val_dataset", None) if dm is not None else None
        if ds is None or not hasattr(ds, "items"):
            return
        if type(ds).__name__ != "LibriTTSDataset":
            print(f"[WER] skip full-utterance WER: val_dataset is {type(ds).__name__}, not LibriTTSDataset")
            return

        w = self._get_whisper()
        if w is None:
            raise RuntimeError(
                "log_val_wer=True but Whisper is unavailable. "
                "Install openai-whisper and ensure ffmpeg is available."
            )

        import torchaudio

        n = int(getattr(self.config, "wer_epoch_samples", 4))
        n = max(1, min(n, len(ds)))
        # FIXED sample set across epochs — seeded RNG so the same N items are
        # measured every epoch. This isolates model improvement from sampling
        # noise; comparing val/wer trends across epochs becomes meaningful.
        wer_seed = int(getattr(self.config, "wer_seed", 12345))
        idxs = sorted(random.Random(wer_seed).sample(range(len(ds)), n))
        target_sr = int(self.config.sample_rate)
        device = next(self.model.parameters()).device

        want_spk_sim = bool(getattr(self.config, "log_val_spk_sim", False))
        spk_enc = self._get_spk_encoder() if want_spk_sim else None

        was_training = self.model.training
        self.model.eval()
        eval_cross = bool(getattr(self.config, "val_eval_cross_spk", True))
        # same-speaker: intelligibility upper bound + spk_sim ceiling
        wers_same: list[float] = []
        spk_sims_same: list[float] = []
        spk_sims_gt: list[float] = []
        # cross-speaker: the real conversion metric (does output adopt a new speaker?)
        wers_cross: list[float] = []
        spk_sims_cross: list[float] = []          # gen vs TARGET ref — want HIGH
        spk_sims_cross_src: list[float] = []      # gen vs SOURCE (content) — want LOW
        n_skipped_long = 0

        def _score(ctn: torch.Tensor, ref_path: str, ref_txt: str):
            """Generate with ref_path as the speaker reference; return (wer, gen, ref)."""
            ref = _load_full_mono_wav(ref_path, target_sr).unsqueeze(0).to(device)
            # Codec speaker encoder emits one prefix token per reference frame (~50 fps),
            # so a full-length reference bloats the sequence (and is off-distribution:
            # training used audio_duration windows). Trim it to the training window.
            # ECAPA pools to a fixed speaker_prefix_len, so it keeps the full reference.
            if str(getattr(self.config, "speaker_encoder_type", "codec")).lower() not in ("speechbrain_ecapa", "spark_global"):
                ref_max_samples = int(float(getattr(self.config, "audio_duration", 4.0)) * target_sr)
                if ref.shape[-1] > ref_max_samples:
                    ref = ref[..., :ref_max_samples]
            # Greedy decoding (temperature=0 → argmax). Removes sampling
            # variance so two runs at the same ckpt give the same WER.
            codes = self.model.generate(
                reference_audio=ref,
                content_audio=ctn,
                temperature=0.0,
                top_p=None,
                top_k=None,
            )
            gen = self.model.decode_to_audio(codes)
            wavs = gen.detach().float().cpu()
            if target_sr != 16000:
                wavs = torchaudio.functional.resample(wavs, target_sr, 16000)
            hyp_txt = w.transcribe(wavs[0].numpy(), fp16=False)["text"]
            return self._word_wer(hyp_txt, ref_txt), gen, ref

        try:
            for ix in idxs:
                wav_path, utt_id, spk = ds.items[ix]
                ref_txt = _read_libritts_normalized_text(wav_path)
                if not ref_txt:
                    continue
                ctn = _load_full_mono_wav(wav_path, target_sr).unsqueeze(0).to(device)
                # The model's context is max_seq_len tokens; total generation length =
                # ref_prefix + ceil(content/hop) + (K-1) must fit. Content longer than
                # val_max_content_sec (default 15s → ~750 tokens) can't be generated, so
                # skip it (rare in LibriTTS) instead of overflowing the attention mask.
                content_max_sec = float(getattr(self.config, "val_max_content_sec", 15.0))
                if ctn.shape[-1] > int(content_max_sec * target_sr):
                    n_skipped_long += 1
                    continue

                # Source-speaker embedding (real content audio) — computed once per
                # item, reused as the same-mode GT ceiling and the cross-mode source
                # baseline (a good conversion moves AWAY from this).
                emb_ctn = spk_enc(ctn.detach().float()) if spk_enc is not None else None

                # ── SAME-speaker: intelligibility + spk_sim ceiling. Fixed across
                # epochs (val split → deterministic same-spk pick) so trends compare. ──
                same_ref_path = _libritts_pick_ref_path(ds, wav_path, utt_id, spk)
                wer_s, gen_s, same_ref = _score(ctn, same_ref_path, ref_txt)
                wers_same.append(wer_s)
                if spk_enc is not None:
                    emb_gen = spk_enc(gen_s.detach().float())
                    emb_ref = spk_enc(same_ref.detach().float())
                    spk_sims_same.append(
                        F.cosine_similarity(emb_gen, emb_ref, dim=-1).mean().item()
                    )
                    # GT ceiling: real content audio vs the same-speaker reference —
                    # the intra-speaker similarity, i.e. the max spk_sim reachable.
                    spk_sims_gt.append(
                        F.cosine_similarity(emb_ctn, emb_ref, dim=-1).mean().item()
                    )

                # ── CROSS-speaker: does the output ADOPT a new speaker? Seeded per
                # item so the same target speaker is used every epoch (comparable). ──
                if eval_cross:
                    cross_rng = random.Random(wer_seed * 1_000_003 + ix)
                    cross_ref_path = _libritts_pick_diff_speaker_ref_path(ds, spk, cross_rng)
                    if cross_ref_path is not None:
                        wer_c, gen_c, cross_ref = _score(ctn, cross_ref_path, ref_txt)
                        wers_cross.append(wer_c)
                        if spk_enc is not None:
                            emb_gen_c = spk_enc(gen_c.detach().float())
                            emb_ref_c = spk_enc(cross_ref.detach().float())
                            spk_sims_cross.append(
                                F.cosine_similarity(emb_gen_c, emb_ref_c, dim=-1).mean().item()
                            )
                            # Source leakage: converted output vs the SOURCE speaker.
                            # Want LOW — the output should move away from the content
                            # speaker toward the target reference.
                            spk_sims_cross_src.append(
                                F.cosine_similarity(emb_gen_c, emb_ctn, dim=-1).mean().item()
                            )
        finally:
            if was_training:
                self.model.train()

        if n_skipped_long:
            print(f"[WER] skipped {n_skipped_long}/{n} val items longer than "
                  f"{float(getattr(self.config, 'val_max_content_sec', 15.0)):g}s "
                  f"(exceed model context {self.config.max_seq_len} tokens).")
        if not wers_same:
            raise RuntimeError(
                "log_val_wer=True but no LibriTTS normalized transcripts found for sampled val items."
            )
        step = int(getattr(self.trainer, "global_step", self.global_step))
        exp = getattr(self.logger, "experiment", None)

        def _log(key: str, values: list[float], prog_bar: bool = False):
            if not values:
                return
            mean = float(sum(values) / len(values))
            self.log(key, mean, on_epoch=True, sync_dist=True, prog_bar=prog_bar)
            if exp is not None and hasattr(exp, "log"):
                exp.log({key: mean}, step=step)

        # same-speaker (keep val/wer_full_epoch_mean + val/spk_sim names for continuity)
        _log("val/wer_full_epoch_mean", wers_same)
        _log("val/spk_sim", spk_sims_same, prog_bar=True)
        _log("val/spk_sim_gt", spk_sims_gt)
        # cross-speaker (the actual conversion quality)
        _log("val/wer_cross", wers_cross)
        _log("val/spk_sim_cross", spk_sims_cross, prog_bar=True)       # → target (HIGH)
        _log("val/spk_sim_cross_src", spk_sims_cross_src)              # → source (LOW)

    @torch.no_grad()
    def _log_vc_sample(
        self,
        ref_audio: torch.Tensor,
        ctn_audio: torch.Tensor,
        tag_prefix: str = "val",
    ):
        codes = self.model.generate(
            reference_audio=ref_audio,
            content_audio=ctn_audio,
            temperature=float(getattr(self.config, "temperature", 1.0)),
            top_p=float(getattr(self.config, "top_p", 0.9)),
        )
        gen_audio = self.model.decode_to_audio(codes)
        sr = int(self.config.sample_rate)
        _log_audio_batch(self.logger, f"{tag_prefix}/generated_vc", gen_audio, sr, self.global_step)
        _log_audio_batch(self.logger, f"{tag_prefix}/reference_audio", ref_audio, sr, self.global_step)
        _log_audio_batch(self.logger, f"{tag_prefix}/content_audio", ctn_audio, sr, self.global_step)
        return gen_audio

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
