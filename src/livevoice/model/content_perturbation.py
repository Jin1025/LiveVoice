"""Source-side audio perturbation to de-identify speaker before HuBERT extraction.

Rationale: HuBERT layer-9 features carry some residual speaker identity (timbre leakage).
By perturbing the *content* audio before encoding — while leaving the *target* untouched —
the model is forced to pull speaker info from the reference path rather than from content.

Three transforms applied in sequence during training only:
  1. Pitch shift  ±pitch_shift_semitones  (torchaudio phase vocoder, GPU-compatible)
  2. Formant (VTLN) shift via Praat 'Change gender'  formant ratio 1±formant_ratio_range
     — warps the vocal-tract/gender while PRESERVING pitch and duration (so content
     stays frame-aligned). Replaces the old resample round-trip, which was a no-op.
  3. 4-band random EQ  ±eq_gain_db  (spectral tilt scramble)

All operations are applied per-sample (different random draw per item in the batch),
using a simple Python loop to allow per-sample random parameters.
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Lazy Praat/parselmouth handle (CPU formant shifter; imported on first use).
_PARSELMOUTH = None


def _get_parselmouth():
    global _PARSELMOUTH
    if _PARSELMOUTH is None:
        import parselmouth
        from parselmouth.praat import call
        _PARSELMOUTH = (parselmouth, call)
    return _PARSELMOUTH


class ContentPerturbation(nn.Module):
    """Apply speaker-de-identifying perturbations to content audio before HuBERT."""

    def __init__(self, config):
        super().__init__()
        self.sr = int(config.sample_rate)
        self.pitch_max = float(getattr(config, "perturb_pitch_semitones", 4.0))
        self.use_vtln = bool(getattr(config, "use_vtln", False))
        self.formant_range = float(
            getattr(config, "perturb_formant_ratio_range",
                    getattr(config, "perturb_vtln_alpha_range", 0.25))
        )
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

            # 1. Pitch shift ±N semitones (always exclude near-zero range)
            # Sample from [-pitch_max, -1] U [1, pitch_max].
            if self.pitch_max > 0:
                min_abs = 1.0
                if self.pitch_max >= min_abs:
                    sign = -1.0 if random.random() < 0.5 else 1.0
                    mag = random.uniform(min_abs, self.pitch_max)
                    n_steps = sign * mag
                    try:
                        x = torchaudio.functional.pitch_shift(x, self.sr, n_steps)
                    except Exception:
                        pass

            # 2. Formant (VTLN) shift via Praat 'Change gender' (CPU).
            #    ratio > 1 → formants up (more feminine/child), < 1 → down (more
            #    masculine). new_pitch_median=0 keeps pitch, duration_factor=1 keeps
            #    timing, so the perturbed content stays aligned with the codec grid.
            #    FIXED magnitude (formant_range), random direction → every sample
            #    gets a real warp (no near-zero no-ops), matching the pitch shift.
            if self.use_vtln and self.formant_range > 0:
                sign = -1.0 if random.random() < 0.5 else 1.0
                ratio = 1.0 + sign * self.formant_range   # e.g. 0.25 → 0.75 or 1.25
                if abs(ratio - 1.0) > 0.02:
                    orig_len = x.shape[-1]
                    try:
                        parselmouth, call = _get_parselmouth()
                        x_np = x.squeeze(0).detach().cpu().numpy().astype("float64")
                        snd = parselmouth.Sound(x_np, sampling_frequency=float(self.sr))
                        out = call(snd, "Change gender", 75, 600, float(ratio), 0.0, 1.0, 1.0)
                        y = torch.from_numpy(
                            np.asarray(out.values).squeeze().astype("float32")
                        )
                        if y.shape[-1] > orig_len:
                            y = y[..., :orig_len]
                        elif y.shape[-1] < orig_len:
                            y = F.pad(y, (0, orig_len - y.shape[-1]))
                        x = y.unsqueeze(0).to(device)
                    except Exception:
                        pass  # short/silent segment — leave pitch-shifted audio as-is

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
