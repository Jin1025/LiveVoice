"""FSQ (Finite Scalar Quantization) information bottleneck for the content path.

StyleStream-style (arXiv:2602.20113) bottleneck, adapted for joint VC training:
StyleStream trains its Destylizer with an ASR loss and takes the *pre*-FSQ
continuous features; our bottleneck is trained jointly with the reconstruction
loss, which would happily route speaker info through anything soft — so we use
the *post*-quantization values (straight-through) to make the constraint
structural. Codebook size = prod(levels); channel count = len(levels).

FSQ itself follows Mentzer et al. 2023 ("Finite Scalar Quantization: VQ-VAE
Made Simple"): per-channel tanh bound to L levels + round with a
straight-through estimator. Parameter-free quantizer; the learnable parts are
the down/up projections that make the low-dim code a real bottleneck.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradient."""
    return z + (z.round() - z).detach()


class FSQBottleneck(nn.Module):
    """dim → len(levels) chans → FSQ round (STE) → dim.

    Per-frame, stateless → fully streamable. Output values are the quantized
    codes renormalized to [-1, 1] before the up-projection.
    """

    def __init__(self, dim: int, levels: tuple[int, ...] | list[int]):
        super().__init__()
        levels = tuple(int(l) for l in levels)
        if any(l < 2 for l in levels):
            raise ValueError(f"FSQ levels must all be >= 2, got {levels}")
        self.dim = int(dim)
        self.down = nn.Linear(self.dim, len(levels))
        self.up = nn.Linear(len(levels), self.dim)
        # Buffer (not param) so levels ride along in the state_dict → eval can
        # recover the exact bottleneck config from a checkpoint.
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.long))
        self.codebook_size = int(math.prod(levels))

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(self._levels.tolist())

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """tanh-bound each channel so round() yields exactly L integer levels."""
        levels = self._levels.to(device=z.device, dtype=z.dtype)
        half_l = (levels - 1) * (1 + 1e-3) / 2
        offset = torch.where(levels % 2 == 0, 0.5, 0.0).to(z.dtype)
        shift = torch.atanh(offset / half_l)
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """(…, len(levels)) unbounded → quantized codes renormalized to [-1, 1]."""
        levels = self._levels.to(device=z.device, dtype=z.dtype)
        q = _round_ste(self._bound(z))
        half_width = levels // 2  # int div, matches reference implementation
        return q / half_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) → (B, T, dim) through the quantized bottleneck."""
        return self.up(self.quantize(self.down(x)))
