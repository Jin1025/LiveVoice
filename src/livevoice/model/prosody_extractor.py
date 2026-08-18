"""Causal F0 + loudness prosody extractor (optional conditioning).

Trimmed-down, speech-focused version of sonic's SketchFeatureExtractor:
- A-weighted loudness from magnitude STFT
- Pitch-probability distribution via YIN (default) or CREPE ("tiny")
- No spectral centroid, no rhythm bands — speech prosody is mostly F0 + energy
- Optional random median filter to make the control sketch-like across speakers

All operations are causal (center=False, left-pad only).

Why YIN and not the old FFT fallback: that fallback did not estimate F0 at all. It rescaled
the 50-2000 Hz magnitude bins to `pitch_bins` and softmaxed them, so what reached the model
was a coarse spectral envelope — harmonics, formants and all — not a pitch track. Anything
downstream conditioned on "pitch" would in fact have been conditioned on timbre, which is the
one thing this system exists to remove. It is deleted rather than kept as a fallback.

Why not CREPE: it is a CNN forward per frame. Correctness is not the issue (it is frame-wise,
so it is causal); cost is. Measured on 5 s of audio at 16 kHz, hop 320:

    torch YIN (this file)   RTF ~0.0006      librosa.pyin   RTF 0.026

YIN also runs batched on GPU inside the training loop, where a numpy pitch tracker would
serialise every step. CREPE is kept selectable for offline comparison.

Framing: a YIN frame needs W analysis samples plus tau_max lag samples. Both are taken from
INSIDE the n_fft window the STFT already reads (W = n_fft - tau_max), so the pitch track adds
no lookahead beyond the loudness track and the two are frame-aligned by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class ProsodyExtractor(nn.Module):
    """F0 (CREPE/FFT) + A-weighted loudness → per-frame prosody features."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sr = int(config.sample_rate)
        self.hop_length = int(config.prosody_hop_length)
        self.n_fft = int(config.n_fft)

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=1.0,
            normalized=False,
            center=False,
        )

        self.register_buffer("a_weighting", self._a_weighting())

        self.pitch_method = str(getattr(config, "pitch_method", "yin"))
        if self.pitch_method not in ("yin", "crepe"):
            raise ValueError(f"pitch_method must be 'yin' or 'crepe', got {self.pitch_method!r}")
        self.pitch_bins = int(config.pitch_bins)
        self.pitch_threshold = float(config.pitch_threshold)
        self.fmin = float(getattr(config, "pitch_fmin", 70.0))
        self.fmax = float(getattr(config, "pitch_fmax", 400.0))
        self.yin_threshold = float(getattr(config, "yin_threshold", 0.15))
        self.pitch_normalize = bool(getattr(config, "pitch_normalize", True))
        self.pitch_prior_frames = float(getattr(config, "pitch_prior_frames", 25.0))
        self.pitch_prior_hz = float(getattr(config, "pitch_prior_hz", 150.0))
        self.pitch_norm_span = float(getattr(config, "pitch_norm_span_cents", 1200.0))
        if self.pitch_method == "crepe":
            try:
                import torchcrepe  # noqa: F401
            except ImportError as e:
                # No silent downgrade: the old code fell back to a spectral-envelope stand-in,
                # so a missing package quietly changed WHAT the model was conditioned on.
                raise ImportError(
                    "pitch_method='crepe' needs torchcrepe. Install it, or set "
                    "config.pitch_method='yin' (cheaper, and the default)."
                ) from e

        # YIN lag range, in samples. tau_max is carved out of the STFT window so the pitch
        # track reads no further ahead than the loudness track does.
        self.tau_min = max(2, int(self.sr / self.fmax))
        self.tau_max = min(int(self.sr / self.fmin) + 1, self.n_fft // 2)
        self.yin_win = self.n_fft - self.tau_max
        if self.yin_win < 2 * self.tau_max:
            raise ValueError(
                f"n_fft={self.n_fft} too small for pitch_fmin={self.fmin} Hz: leaves "
                f"{self.yin_win} analysis samples for {self.tau_max} lags. Raise n_fft or fmin.")

        # CREPE's target grid: 360 bins of 20 cents from C1 (32.70 Hz). Sharing it keeps the
        # two methods interchangeable behind one `pitch_proj`.
        cents = 20.0 * torch.arange(self.pitch_bins, dtype=torch.float32)
        self.register_buffer("bin_hz", 32.70 * 2.0 ** (cents / 1200.0))

        # projections
        self.loudness_proj = nn.Linear(1, config.prosody_hidden_dim)
        self.pitch_proj = nn.Linear(self.pitch_bins, config.prosody_hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(config.prosody_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        self.use_random_median = bool(config.use_random_median_filter)
        self.median_min = int(config.median_filter_min_size)
        self.median_max = int(config.median_filter_max_size)
        self.median_inference = int(config.median_filter_inference_size)

    # ----------------- core ops -----------------
    def _a_weighting(self) -> torch.Tensor:
        freq = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        f = torch.clamp(freq, min=1.0)
        ra = (12194.0 ** 2) * (f ** 4)
        rb = (f ** 2 + 20.6 ** 2) * (f ** 2 + 12194.0 ** 2)
        rc = torch.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
        a_db = 2.0 + 20.0 * torch.log10(ra / (rb * rc + 1e-8))
        a_lin = 10.0 ** (a_db / 20.0)
        return a_lin / torch.max(a_lin)

    def _loudness(self, mag: torch.Tensor) -> torch.Tensor:
        w = mag * self.a_weighting.unsqueeze(0).unsqueeze(0)
        s = torch.sum(w, dim=-1)
        db = 20.0 * torch.log10(s + 1e-8)
        out = (db + 60.0) / 60.0
        return torch.clamp(out, 0.0, 1.0)

    def _pitch_crepe(self, audio: torch.Tensor) -> torch.Tensor:
        import torchcrepe
        B = audio.shape[0]
        outs = []
        for i in range(B):
            x = audio[i : i + 1].float()
            x = x / (torch.max(torch.abs(x)) + 1e-8)
            with torch.no_grad():
                frames_gen = torchcrepe.preprocess(
                    x, self.sr, hop_length=self.hop_length,
                    batch_size=None, device=audio.device, pad=False,
                )
                frames = torch.cat(list(frames_gen), dim=0)
                probs = torchcrepe.infer(frames, model="tiny", device=audio.device, embed=False)
                probs = torch.where(probs > self.pitch_threshold, probs, torch.zeros_like(probs))
                outs.append(probs.unsqueeze(0))
        return torch.cat(outs, dim=0)

    def _pitch_yin(self, audio: torch.Tensor) -> torch.Tensor:
        """YIN F0 → soft one-hot over `pitch_bins`. (B, T, bins), all-zero where unvoiced.

        de Cheveigné & Kawahara (2002), steps 1-5. The difference function is computed from
        power sums and an FFT cross-correlation rather than directly, which is what keeps the
        cost at O(N log N) per frame instead of O(W * tau_max) — the direct form would need a
        (B, T, tau_max, W) intermediate, ~870 MB at batch 8.
        """
        W, tmax = self.yin_win, self.tau_max
        frames = audio.unfold(-1, self.n_fft, self.hop_length)          # (B, T, n_fft)
        B, T, _ = frames.shape
        x = frames.reshape(B * T, self.n_fft)

        # d(tau) = p(0) + p(tau) - 2 r(tau)
        cs = torch.cumsum(F.pad(x * x, (1, 0)), dim=-1)                 # (BT, n_fft+1)
        p = cs[:, W:] - cs[:, :-W]                                      # p(tau), tau=0..tmax
        p = p[:, : tmax + 1]
        n = 1 << int(self.n_fft + W - 1).bit_length()
        r = torch.fft.irfft(
            torch.fft.rfft(x, n=n) * torch.fft.rfft(x[:, :W], n=n).conj(), n=n
        )[:, : tmax + 1]
        d = (p[:, :1] + p - 2.0 * r).clamp_min(0.0)

        # cumulative mean normalised difference; d'(0) := 1
        cum = torch.cumsum(d, dim=-1)
        lag = torch.arange(1, tmax + 1, device=x.device, dtype=d.dtype)
        dn = torch.ones_like(d)
        dn[:, 1:] = d[:, 1:] * lag / cum[:, 1:].clamp_min(1e-12)

        # first lag below threshold, else global minimum (absolute-threshold step)
        cand = dn[:, self.tau_min : tmax + 1]
        below = cand < self.yin_threshold
        first = torch.where(
            below.any(dim=-1), below.float().argmax(dim=-1), cand.argmin(dim=-1)
        ) + self.tau_min

        # parabolic interpolation on the chosen lag
        i = first.clamp(1, tmax - 1)
        g = torch.gather
        y0 = g(dn, 1, (i - 1).unsqueeze(1)).squeeze(1)
        y1 = g(dn, 1, i.unsqueeze(1)).squeeze(1)
        y2 = g(dn, 1, (i + 1).unsqueeze(1)).squeeze(1)
        denom = (y0 - 2 * y1 + y2)
        shift = torch.where(denom.abs() > 1e-12, 0.5 * (y0 - y2) / denom, torch.zeros_like(denom))
        tau = i.to(d.dtype) + shift.clamp(-1.0, 1.0)
        f0 = (self.sr / tau.clamp_min(1e-6)).reshape(B, T)

        # voicing: aperiodicity at the chosen lag. y1 IS d'(tau), so 1-y1 is periodicity.
        conf = (1.0 - y1).clamp(0.0, 1.0).reshape(B, T)
        voiced = ((y1 < self.yin_threshold).reshape(B, T)
                  & (f0 >= self.fmin) & (f0 <= self.fmax))
        return self._bin_pitch(f0, voiced, conf)

    def _bin_pitch(self, f0: torch.Tensor, voiced: torch.Tensor,
                   conf: torch.Tensor) -> torch.Tensor:
        """(B,T) F0 → (B,T,bins) soft one-hot, Gaussian in cents. Zero where unvoiced.

        With ``pitch_normalize`` the grid is cents RELATIVE to a causal running mean of
        log-F0 instead of absolute Hz. That is the whole privacy argument for this feature:
        a speaker's register (the mean) is one of the strongest identity cues in F0, while
        emotion lives in the DEVIATION from it — sadness narrows and lowers the excursion,
        anger widens it. Subtracting the mean drops the identity term and keeps the emotion
        term. Dividing by the standard deviation as well would drop the emotion term too, so
        it is deliberately NOT done.

        The mean is causal (frames <= t, voiced only) so this stays streaming-safe, and it
        carries the same virtual-prior count as content CMN: without one, the first voiced
        frame IS the mean, so its deviation is identically zero for every utterance.
        """
        B, T = f0.shape
        dev = self.pitch_normalize
        if dev:
            lf = torch.log2(f0.clamp_min(1e-6)) * 1200.0            # cents, absolute
            m = voiced.to(lf.dtype)
            n0, p0 = self.pitch_prior_frames, 1200.0 * torch.log2(
                torch.tensor(self.pitch_prior_hz, dtype=lf.dtype, device=lf.device))
            num = torch.cumsum(lf * m, dim=1) + n0 * p0
            den = torch.cumsum(m, dim=1) + n0
            cents = lf - num / den.clamp_min(1e-6)
            grid = torch.linspace(-self.pitch_norm_span, self.pitch_norm_span,
                                  self.pitch_bins, device=f0.device, dtype=lf.dtype)
            sigma = 2.0 * self.pitch_norm_span / (self.pitch_bins - 1)
        else:
            cents = 1200.0 * torch.log2(f0.clamp_min(1e-6) / 32.70)
            grid = 20.0 * torch.arange(self.pitch_bins, device=f0.device, dtype=f0.dtype)
            sigma = 25.0
        probs = torch.exp(-0.5 * ((grid.view(1, 1, -1) - cents.unsqueeze(-1)) / sigma) ** 2)
        probs = probs * conf.unsqueeze(-1) * voiced.to(probs.dtype).unsqueeze(-1)
        return torch.where(probs > self.pitch_threshold, probs, torch.zeros_like(probs))

    # ----------------- public API -----------------
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Causal prosody features aligned to codec frames.

        Args:
            audio: (B, T) at 16 kHz

        Returns:
            features: (B, T_frames, hidden_dim)
        """
        audio = audio.float()
        mag = self.stft(audio).transpose(1, 2)  # (B, T_frames, F)
        loudness = self._loudness(mag)          # (B, T_frames)
        if self.pitch_method == "crepe":
            pitch = self._pitch_crepe(audio)
        else:
            pitch = self._pitch_yin(audio)

        T = loudness.shape[1]
        if pitch.shape[1] != T:
            if pitch.shape[1] > T:
                pitch = pitch[:, :T, :]
            else:
                pitch = F.pad(pitch, (0, 0, 0, T - pitch.shape[1]), mode="replicate")

        if self.use_random_median:
            if self.training:
                loudness, pitch = self._random_median(loudness, pitch)
            else:
                loudness, pitch = self._fixed_median(loudness, pitch, self.median_inference)

        emb = self.loudness_proj(loudness.unsqueeze(-1)) + self.pitch_proj(pitch)
        return self.output_proj(emb)

    # ----------------- median filters -----------------
    def _random_median(self, loudness, pitch):
        B, T = loudness.shape
        out_l, out_p = [], []
        for b in range(B):
            fs = torch.randint(self.median_min, self.median_max + 1, (1,), device=loudness.device).item()
            if fs <= 1:
                out_l.append(loudness[b])
                out_p.append(pitch[b])
                continue
            l_pad = F.pad(loudness[b].unsqueeze(0), (fs - 1, 0), mode="replicate").squeeze(0)
            out_l.append(torch.median(l_pad.unfold(0, fs, 1), dim=-1)[0])
            pp = pitch[b].T
            pp_pad = F.pad(pp, (fs - 1, 0), mode="replicate")
            pp_filt = torch.median(pp_pad.unfold(1, fs, 1), dim=-1)[0]
            out_p.append(pp_filt.T)
        return torch.stack(out_l), torch.stack(out_p)

    def _fixed_median(self, loudness, pitch, fs: int):
        if fs <= 1:
            return loudness, pitch
        out_l, out_p = [], []
        for b in range(loudness.shape[0]):
            l_pad = F.pad(loudness[b].unsqueeze(0), (fs - 1, 0), mode="replicate").squeeze(0)
            out_l.append(torch.median(l_pad.unfold(0, fs, 1), dim=-1)[0])
            pp = pitch[b].T
            pp_pad = F.pad(pp, (fs - 1, 0), mode="replicate")
            pp_filt = torch.median(pp_pad.unfold(1, fs, 1), dim=-1)[0]
            out_p.append(pp_filt.T)
        return torch.stack(out_l), torch.stack(out_p)
