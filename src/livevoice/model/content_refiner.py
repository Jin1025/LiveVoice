"""Deep, causal content refiner ("Destylizer") on top of the frozen sw2v features.

Motivation: the de-speaker pressure (GRL) and the keep-content pressure (ASR) only
reach the SHALLOW learnable head (sw2v_proj: 1024→256). Probing showed even a random
linear projection of the frozen sw2v output is ~99.9% speaker-decodable, so a shallow
linear map has a hard time suppressing speaker while keeping content. This module adds
trainable DEPTH exactly where those objectives apply — a small stack of residual causal
convolutions over the cached 1024-d features — WITHOUT unfreezing the encoder (so all
feature caches stay valid) and WITHOUT breaking streaming (strictly causal: left-padded
convs only, no lookahead).

Placed right before sw2v_proj, so it sits in every content path (main VC window +
ASR/GRL full-utterance), and gradient from GRL/ASR flows through its full depth.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConvBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → left-padded (causal) depthwise-ish Conv1d → GELU."""

    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel)   # causal via manual left pad in forward
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pad = kernel - 1
        # Zero-init the conv so each block starts as an IDENTITY (residual + 0): the refiner
        # doesn't perturb the pretrained content path at init and can't destabilize training
        # before it has learned anything (esp. important under GRL reversal). Learns away
        # from zero via gradient — standard zero-residual (ReZero/Fixup) trick.
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        residual = x
        h = self.norm(x).transpose(1, 2)          # (B, D, T)
        h = F.pad(h, (self.pad, 0))               # LEFT pad only → strictly causal
        h = self.conv(h).transpose(1, 2)          # (B, T, D)
        return residual + self.drop(self.act(h))


class ContentRefiner(nn.Module):
    """Stack of causal residual conv blocks; (B, T, dim) → (B, T, dim). Streamable."""

    def __init__(self, dim: int, n_layers: int, kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_CausalConvBlock(dim, kernel, dropout) for _ in range(n_layers)]
        )
        rf = 1 + n_layers * (kernel - 1)
        print(f"[ContentRefiner] causal conv stack: dim={dim} layers={n_layers} "
              f"kernel={kernel} receptive_field={rf} frames (~{rf*20}ms @50fps) — streamable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
