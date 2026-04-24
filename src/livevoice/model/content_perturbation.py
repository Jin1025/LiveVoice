"""Source-side audio perturbation to de-identify speaker before HuBERT extraction.

Rationale: HuBERT layer-9 features carry some residual speaker identity (timbre leakage).
By perturbing the *content* audio before encoding — while leaving the *target* untouched —
the model is forced to pull speaker info from the reference path rather than from content.

Three transforms applied in sequence during training only:
  1. Pitch shift  ±pitch_shift_semitones  (torchaudio phase vocoder, GPU-compatible)
  2. VTLN formant shift via resample trick  ±vtln_alpha_range  (approximates vocal tract warp)
  3. 4-band random EQ  ±eq_gain_db  (spectral tilt scramble)

All operations are applied per-sample (different random draw per item in the batch),
using a simple Python loop to allow per-sample random parameters.
"""
from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class ContentPerturbation(nn.Module):
    """Apply speaker-de-identifying perturbations to content audio before HuBERT."""

    def __init__(self, config):
        super().__init__()
        self.sr = int(config.sample_rate)
        self.pitch_max = float(getattr(config, "perturb_pitch_semitones", 4.0))
        self.vtln_range = float(getattr(config, "perturb_vtln_alpha_range", 0.12))
        self.eq_gain_db = float(getattr(config, "perturb_eq_gain_db", 6.0))
        self.p = float(getattr(config, "perturb_prob", 1.0))

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T) — returns perturbed (B, T) during training, identity during eval."""
        if not self.training:
            return audio

        B, T = audio.shape
        device = audio.device
        out_list = []

        for i in range(B):
            x = audio[i].unsqueeze(0)  # (1, T)

            if random.random() > self.p:
                out_list.append(x)
                continue

            # 1. Pitch shift ±N semitones
            if self.pitch_max > 0:
                n_steps = random.uniform(-self.pitch_max, self.pitch_max)
                if abs(n_steps) > 0.5:
                    try:
                        x = torchaudio.functional.pitch_shift(x, self.sr, n_steps)
                    except Exception:
                        pass

            # 2. VTLN-style formant shift via resample trick
            #    Resample to sr*alpha (compresses/expands vocal tract resonances),
            #    then resample back to sr to restore time length.
            #    This approximates vocal tract length perturbation (VTLP).
            if self.vtln_range > 0:
                alpha = 1.0 + random.uniform(-self.vtln_range, self.vtln_range)
                if abs(alpha - 1.0) > 0.02:
                    orig_len = x.shape[-1]
                    shifted_sr = max(8000, int(self.sr * alpha))
                    x = torchaudio.functional.resample(x, self.sr, shifted_sr)
                    x = torchaudio.functional.resample(x, shifted_sr, self.sr)
                    # Restore exact length after resampling rounding
                    if x.shape[-1] > orig_len:
                        x = x[..., :orig_len]
                    elif x.shape[-1] < orig_len:
                        x = F.pad(x, (0, orig_len - x.shape[-1]))

            # 3. 4-band random EQ
            if self.eq_gain_db > 0:
                eq_freqs = [
                    random.uniform(80, 300),
                    random.uniform(300, 1200),
                    random.uniform(1200, 4000),
                    random.uniform(4000, 7500),
                ]
                for cf in eq_freqs:
                    gain = random.uniform(-self.eq_gain_db, self.eq_gain_db)
                    Q = random.uniform(0.5, 2.0)
                    try:
                        x = torchaudio.functional.equalizer_biquad(x, self.sr, cf, gain, Q)
                    except Exception:
                        pass

            # Peak-normalize to avoid clipping after perturbation
            peak = x.abs().max()
            if peak > 1e-6:
                x = x / peak

            out_list.append(x)

        return torch.cat(out_list, dim=0)  # (B, T)
