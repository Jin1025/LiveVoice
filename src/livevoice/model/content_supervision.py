"""Shared content-path construction + ASR/GRL supervision losses.

Single source of truth for the content bottleneck ("tokenizer") so that the
TWO-STAGE recipe uses byte-identical modules and parameter names:

  Stage 1 (ContentTokenizerModel, train_content_tokenizer.py)
      sw2v(frozen) → refiner → sw2v_proj → [FSQ] → sw2v_to_hidden
      trained by ASR (keep content) + GRL (remove speaker) ONLY.

  Stage 2 (LiveVoiceModel, train.py)
      the same modules, loaded from the Stage-1 checkpoint and FROZEN, with the
      VC decoder trained on top.

Why two stages (CosyVoice 2, arXiv:2412.10117): CosyVoice 2 trains its supervised
semantic tokenizer (FSQ inside the ASR encoder, driven by an ASR loss) SEPARATELY on
200k hours and then FREEZES it for TTS training. In our joint setup the codec
reconstruction CE also back-propagates into the content path, and because training is
same-speaker reconstruction, that CE actively WANTS speaker identity there — it fights
the GRL. Removing the VC decoder in Stage 1 removes that opposing force entirely, and
freezing in Stage 2 means the decoder can no longer pull speaker info back into content
(it must use the speaker prompt).

Parameter names match LiveVoiceModel's attributes, so a Stage-1 state_dict loads into
the full model with strict=False.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from livevoice.model.asr_supervision import AsrSupervisionHead
from livevoice.model.fsq import FSQBottleneck

# Attributes that make up the content path — used to load/freeze Stage-1 weights.
CONTENT_PATH_PREFIXES = ("content_refiner", "sw2v_proj", "content_fsq", "sw2v_to_hidden")


def apply_content_cmn(
    feats: torch.Tensor,
    mode: str = "off",
    use_var: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Cepstral mean (variance) normalisation on content features (B, T, D).

    Why: the speaker residual the probe still finds is specifically UTTERANCE-level —
    frame-level acc fell to 0.101 (3x chance) while utterance-level plateaued at ~0.517
    (16x chance), i.e. what survives is a statistic ACROSS frames, not the individual
    frames. Subtracting the per-utterance mean attacks exactly that statistic. This is the
    oldest speaker/channel normalisation in ASR (CMN/CMVN), and it costs nothing.

    Applied at the FRONTEND (raw encoder output, before the refiner), deliberately — NOT at
    the point the probe mean-pools. Normalising there would drive the pooled vector to
    exactly zero and make the metric meaningless without removing anything the decoder,
    which consumes frames, cannot still see.

    mode:
      "off"        no-op
      "utterance"  subtract the whole-utterance mean — NOT causal, offline only
      "causal"     subtract a running mean over frames <= t — streaming-safe
    """
    if not mode or mode == "off":
        return feats
    if mode == "utterance":
        mu = feats.mean(dim=1, keepdim=True)
        out = feats - mu
        if use_var:
            out = out / feats.std(dim=1, keepdim=True).clamp_min(eps)
        return out
    if mode == "causal":
        n = torch.arange(
            1, feats.size(1) + 1, device=feats.device, dtype=feats.dtype
        ).view(1, -1, 1)
        mu = feats.cumsum(dim=1) / n
        out = feats - mu
        if use_var:
            var = ((feats * feats).cumsum(dim=1) / n - mu * mu).clamp_min(0.0)
            out = out / var.sqrt().clamp_min(eps)
        return out
    raise ValueError(f"content_cmn must be off/utterance/causal, got {mode!r}")


def build_content_path(module: nn.Module, config, sw2v_dim: int, *, log_prefix: str) -> None:
    """Create refiner / sw2v_proj / [FSQ] / sw2v_to_hidden / asr_head / grl_head on `module`."""
    # Deep causal refiner on the raw sw2v feats, BEFORE sw2v_proj — trainable depth for
    # GRL/ASR to shape (content_refiner.py). Cache-compatible, causal/streamable.
    n_ref = int(getattr(config, "content_refiner_layers", 0))
    if n_ref > 0:
        from livevoice.model.content_refiner import ContentRefiner
        module.content_refiner = ContentRefiner(
            sw2v_dim, n_ref, int(getattr(config, "content_refiner_kernel", 5)),
            float(getattr(config, "dropout", 0.1)),
        )
    else:
        module.content_refiner = None

    module.sw2v_proj = nn.Linear(sw2v_dim, config.content_proj_dim)
    module.sw2v_to_hidden = nn.Linear(config.content_proj_dim, config.hidden_dim)
    print(f"{log_prefix} sw2v content path (out_dim={sw2v_dim} → "
          f"content_proj_dim={config.content_proj_dim} → hidden={config.hidden_dim})")

    if bool(getattr(config, "use_content_fsq", False)):
        module.content_fsq = FSQBottleneck(
            config.content_proj_dim, tuple(getattr(config, "fsq_levels", (8, 5, 5, 5)))
        )
        print(f"{log_prefix} content FSQ bottleneck: levels={module.content_fsq.levels} "
              f"→ codebook={module.content_fsq.codebook_size}")
    else:
        module.content_fsq = None

    # ASR supervision (keep phonetic content). seq2seq needs asr_teacher_dropout>0 or it
    # LM-cheats; ctc structurally cannot. Training-only, discarded at inference.
    module.use_asr_supervision = bool(getattr(config, "use_asr_supervision", False))
    module.asr_supervision_type = str(getattr(config, "asr_supervision_type", "seq2seq")).lower()
    if module.use_asr_supervision:
        if module.asr_supervision_type == "ctc":
            from livevoice.model.ctc_supervision import CtcSupervisionHead
            module.asr_head = CtcSupervisionHead(config)
        else:
            module.asr_head = AsrSupervisionHead(config)
    else:
        module.asr_head = None

    # Speaker adversary (remove speaker). Training-only, discarded at inference.
    module.use_speaker_grl = bool(getattr(config, "use_speaker_grl", False))
    if module.use_speaker_grl:
        from livevoice.model.speaker_grl import SpeakerGRLHead
        module.speaker_grl_head = SpeakerGRLHead(
            config, int(getattr(config, "grl_num_speakers", 0))
        )
    else:
        module.speaker_grl_head = None


class ContentSupervisionMixin:
    """ASR + GRL losses over the full-utterance content embedding.

    Requires the attributes created by `build_content_path` plus `content_extractor`
    (sw2v encoder) and `content_perturbation` / `use_content_perturbation`.
    """

    def _full_content_memory(self, content_feats_full, content_full_audio):
        """Full-utterance content embedding for ASR/GRL: cached features OR (online) run
        the sw2v encoder live on full audio. Applies the SAME refiner/proj/[FSQ]/to_hidden
        as the VC path (shared weights). Returns (B, T_full, hidden_dim)."""
        if content_feats_full is not None:
            feats = content_feats_full.to(
                device=self.sw2v_proj.weight.device, dtype=self.sw2v_proj.weight.dtype
            )
        else:
            audio = content_full_audio.to(self.sw2v_proj.weight.device)
            if getattr(self, "use_content_perturbation", False) and self.training:
                audio = self.content_perturbation(audio)
            feats = self.content_extractor(audio).to(self.sw2v_proj.weight.dtype)  # (B,T,1024)
        # `from_cache` mirrors extract_content: cached features may already carry CMN.
        from_cache = content_feats_full is not None
        if not (from_cache and bool(getattr(self.config, "content_cmn_in_cache", False))):
            feats = apply_content_cmn(
                feats,
                str(getattr(self.config, "content_cmn", "off")),
                bool(getattr(self.config, "content_cmn_var", False)),
            )
        if self.content_refiner is not None:
            feats = self.content_refiner(feats)
        sw2v_emb = self.sw2v_proj(feats)
        if self.content_fsq is not None:
            sw2v_emb = self.content_fsq(sw2v_emb)
        return self.sw2v_to_hidden(sw2v_emb)

    def compute_asr_supervision_loss(
        self,
        content_feats_full: torch.Tensor,
        content_feats_full_len: torch.Tensor,
        phoneme_ids: torch.Tensor,
        content_full_audio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ASR loss on the FULL (un-cropped) content embedding — phoneme labels are
        utterance-level with no timestamps, so this only makes sense on the whole
        utterance. Shapes the exact representation the VC decoder consumes."""
        if self.asr_head is None:
            ref = content_feats_full if content_feats_full is not None else content_full_audio
            return torch.tensor(0.0, device=ref.device)
        memory = self._full_content_memory(content_feats_full, content_full_audio)
        content_feats_full_len = content_feats_full_len.to(memory.device)

        T_full = memory.size(1)
        # CTC consumes per-frame memory + true lengths (no teacher forcing → can't LM-cheat).
        # seq2seq needs the padding mask for cross-attention. Dispatch on head type.
        if getattr(self, "asr_supervision_type", "seq2seq") == "ctc":
            return self.asr_head.compute_loss(memory, content_feats_full_len, phoneme_ids)
        arange = torch.arange(T_full, device=memory.device).unsqueeze(0)
        memory_key_padding_mask = arange >= content_feats_full_len.unsqueeze(1)  # True = pad
        return self.asr_head.compute_loss(memory, memory_key_padding_mask, phoneme_ids)

    def compute_speaker_grl_loss(
        self,
        content_feats_full: torch.Tensor,
        content_feats_full_len: torch.Tensor,
        speaker_labels: torch.Tensor,
        grl_lambda: float,
        content_full_audio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adversarial speaker-classification loss on the mean-pooled content embedding.
        Pools over valid (unpadded) frames only. Returns (loss, acc); watch acc (bounded)
        rather than the loss, which can explode as λ pushes the encoder to maximize it."""
        if self.speaker_grl_head is None:
            ref = content_feats_full if content_feats_full is not None else content_full_audio
            zero = torch.tensor(0.0, device=ref.device)
            return zero, zero
        emb = self._full_content_memory(content_feats_full, content_full_audio)
        content_feats_full_len = content_feats_full_len.to(emb.device)

        T_full = emb.size(1)
        arange = torch.arange(T_full, device=emb.device).unsqueeze(0)
        valid = (arange < content_feats_full_len.unsqueeze(1)).to(emb.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (emb * valid.unsqueeze(-1)).sum(dim=1) / denom          # (B, hidden_dim)

        # "confusion" (bounded, recommended) vs "reversal" (classic GRL, can run away —
        # maximizing CE is unbounded and NaN'd our runs). See speaker_grl.py.
        if str(getattr(self.config, "grl_objective", "reversal")).lower() == "confusion":
            return self.speaker_grl_head.compute_confusion_loss(
                pooled, speaker_labels, grl_lambda
            )

        logits = self.speaker_grl_head(pooled, grl_lambda)
        loss = nn.functional.cross_entropy(logits, speaker_labels, ignore_index=-1)
        with torch.no_grad():
            keep = speaker_labels != -1
            acc = ((logits[keep].argmax(dim=-1) == speaker_labels[keep]).float().mean()
                   if keep.any() else torch.zeros((), device=logits.device))
        return loss, acc


class ContentTokenizerModel(nn.Module, ContentSupervisionMixin):
    """Stage-1 model: the content path + supervision heads, NO VC decoder / codec.

    Trained by ASR + GRL only, so nothing pulls speaker identity back into the content
    (unlike joint training, where the reconstruction CE does exactly that).
    """

    def __init__(self, config, content_extractor):
        super().__init__()
        self.config = config
        self.content_extractor = content_extractor        # frozen sw2v encoder
        self.use_content_perturbation = bool(getattr(config, "use_content_perturbation", False))
        if self.use_content_perturbation:
            from livevoice.model.content_perturbation import ContentPerturbation
            self.content_perturbation = ContentPerturbation(config)
        sw2v_dim = int(getattr(content_extractor, "out_dim", 1024))
        build_content_path(self, config, sw2v_dim, log_prefix="[ContentTokenizer]")
