"""HuBERT-based content (linguistic) feature extractor.

HuBERT backbone is kept on CPU at all times to avoid GPU OOM during resampling
and inference. The learnable projection layers follow the model device (GPU).

Two forward paths:
  forward(audio)              — full HuBERT inference + projection
  from_precomputed(feats)     — skip HuBERT, apply projection only (for cached features)
  forward_raw / from_precomputed_raw  — same but returns content_proj_dim (for FiLM)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

HUBERT_NATIVE_SR = 16_000


class HuBERTContentExtractor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.source_sr = int(config.sample_rate)

        from transformers import HubertModel
        self.hubert = HubertModel.from_pretrained(config.hubert_model_name)
        self.layer = int(config.hubert_layer)
        self.hubert_hidden = int(config.hubert_hidden_dim)

        if bool(getattr(config, "freeze_hubert", True)):
            self.hubert.eval()
            for p in self.hubert.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(self.hubert_hidden, config.content_proj_dim),
            nn.LayerNorm(config.content_proj_dim),
            nn.GELU(),
        )
        self.to_hidden = nn.Linear(config.content_proj_dim, config.hidden_dim)
        self.use_film = str(getattr(config, "content_conditioning", "additive")) == "film"

        # Receptive field + total stride of the HuBERT conv feature extractor, from the
        # actual conv config (valid / no-pad convs). For hubert-base: RF=400, stride=320.
        ks = list(self.hubert.config.conv_kernel)
        ss = list(self.hubert.config.conv_stride)
        rf, jump = 1, 1
        for k, s in zip(ks, ss):
            rf += (k - 1) * jump
            jump *= s
        self.hubert_receptive_field = int(rf)   # 400
        self.hubert_stride = int(jump)           # 320 (= 50 fps @ 16 kHz)

        # Center-align content frames to the codec token grid by front-padding the
        # waveform. HuBERT frame t center = stride*t + (RF-1)/2; a hop-based codec
        # (jhcodec: non-overlapping 320-sample blocks, end-padded → ceil(L/hop) tokens)
        # has token t center = hop*t + (hop-1)/2. When stride == hop the two grids
        # differ only by the constant offset (RF-hop)/2 (= 40 samples for jhcodec),
        # removable by a front zero-pad — no embedding resampling. Only valid when the
        # HuBERT stride equals the codec hop (jhcodec); mimi (12.5 fps) can't be aligned
        # this way and falls back to legacy extraction.
        self.center_align = bool(getattr(config, "content_center_align", True))
        codec = str(getattr(config, "codec", "mimi")).lower()
        self.align_hop = (
            int(getattr(config, "jhcodec_hop_length", 320)) if codec == "jhcodec" else None
        )
        if self.align_hop is not None and self.align_hop != self.hubert_stride:
            self.align_hop = None  # stride/hop slope mismatch → padding alone can't align

    @property
    def frame_rate(self) -> int:
        return 50

    def _extract_hidden(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T) → (B, T_frames, hubert_hidden).

        When center-alignment is enabled (jhcodec), the waveform is front-padded so
        HuBERT frame centers coincide with the codec token grid, and back-padded +
        truncated so the frame count equals the codec's ceil(L/hop). This makes the
        content sequence line up 1:1 with the codec tokens with NO downstream resample.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        audio = audio.float()
        if self.source_sr != HUBERT_NATIVE_SR:
            audio = torchaudio.functional.resample(audio, self.source_sr, HUBERT_NATIVE_SR)

        align = self.center_align and self.align_hop is not None
        n_codec = None
        if align:
            hop = self.align_hop
            L = int(audio.shape[-1])
            n_codec = max(1, math.ceil(L / hop))
            # front pad: shift HuBERT frame centers onto the codec token centers.
            front = max(0, (self.hubert_receptive_field - hop) // 2)   # 40 for jhcodec
            # back pad: enough for >= n_codec frames (frame n_codec-1 needs RF ending at
            # stride*(n_codec-1)+RF), plus a 1-frame margin against valid-conv flooring.
            # Total padded length must match extract_features.encode_batch exactly so
            # cached and online features are identical.
            need_len = self.hubert_stride * (n_codec - 1) + self.hubert_receptive_field
            back = max(0, need_len + self.hubert_stride - L)
            audio = F.pad(audio, (front, back))

        ctx = torch.no_grad() if bool(getattr(self.config, "freeze_hubert", True)) else torch.enable_grad()
        with ctx:
            out = self.hubert(audio, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[self.layer]

        if align:
            T = hidden.shape[1]
            if T < n_codec:  # defensive; the +stride margin should prevent this
                hidden = torch.cat([hidden, hidden[:, -1:, :].expand(-1, n_codec - T, -1)], dim=1)
            else:
                hidden = hidden[:, :n_codec, :]
        return hidden

    # ── Full inference paths ──────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, T) waveform → (B, T_frames, hidden_dim)."""
        return self.to_hidden(self.proj(self._extract_hidden(audio)))

    def forward_raw(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, T) waveform → (B, T_frames, content_proj_dim)  [for FiLM]."""
        return self.proj(self._extract_hidden(audio))

    # ── Precomputed feature paths (skip HuBERT) ──────────────────────

    def from_precomputed(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, 768) precomputed HuBERT hidden states → (B, T, hidden_dim)."""
        feats = feats.to(device=next(self.proj.parameters()).device,
                         dtype=next(self.proj.parameters()).dtype)
        return self.to_hidden(self.proj(feats))

    def from_precomputed_raw(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, 768) → (B, T, content_proj_dim)  [for FiLM]."""
        feats = feats.to(device=next(self.proj.parameters()).device,
                         dtype=next(self.proj.parameters()).dtype)
        return self.proj(feats)
