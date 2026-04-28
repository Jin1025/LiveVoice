"""HuBERT-based content (linguistic) feature extractor.

HuBERT backbone is kept on CPU at all times to avoid GPU OOM during resampling
and inference. The learnable projection layers follow the model device (GPU).

Two forward paths:
  forward(audio)              — full HuBERT inference + projection
  from_precomputed(feats)     — skip HuBERT, apply projection only (for cached features)
  forward_raw / from_precomputed_raw  — same but returns content_proj_dim (for FiLM)
"""
from __future__ import annotations

import torch
import torch.nn as nn
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

    @property
    def frame_rate(self) -> int:
        return 50

    def _extract_hidden(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T) → (B, T_frames, hubert_hidden)."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        audio = audio.float()
        if self.source_sr != HUBERT_NATIVE_SR:
            audio = torchaudio.functional.resample(audio, self.source_sr, HUBERT_NATIVE_SR)
        ctx = torch.no_grad() if bool(getattr(self.config, "freeze_hubert", True)) else torch.enable_grad()
        with ctx:
            out = self.hubert(audio, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[self.layer]
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
