"""Causal cepstral conditioning: c0 (frame energy) and c1 (spectral tilt), per-channel filtered.

Why cepstral and not F0/loudness. Explicit prosody conditioning (pitch contour + A-weighted
loudness) did not move UAR. Feeding the source's low-order cepstrum did: c0 and c1 alone bring
ACC_sad back from ~4 to ~50. They also carry speaker identity, so the raw form costs privacy:

    condition        UAR    ACC_sad  ACC_ang   EER
    original        69.08     63.63    79.78     —
    anon baseline   41.10      ~4       ~74    47.58
    c0,c1 absolute  50.33     49.79      —     38.41   <- sad restored, EER -9.2
    c0,c1 delta     42.38     49.79    10.45   42.50   <- EER half-recovered, ANGER destroyed

The delta row is the finding this module is built around. A first difference is a very
aggressive high-pass, and anger is largely a SUSTAINED level: high energy held across the
utterance. Differencing keeps only the onsets and throws the sustain away, so anger collapses
while sadness — which is about the shape, not the level — survives. Applying one filter to
every channel is what forces that trade.

So the filter is per channel. Each is written `name:mode`, comma separated, in
``config.cepstral_spec``:

    c0:abs                  raw coefficient — keeps the sustained level (anger), leaks speaker
    c1:delta                x[t] - x[t-1] — keeps edges only, strongest speaker removal
    c0:hp0.99               x[t] - EMA_0.99(x[t]) — high-pass with a tunable corner
    c1:ms0.9,0.99           multi-scale: emits TWO channels, (x - EMA_fast) and
                            (EMA_fast - EMA_slow), so the projection can weight the fast and
                            mid bands independently and let the slow speaker baseline go

An EMA with coefficient a has time constant -1/(50*ln a) seconds at 50 fps:

    a=0.90 -> 0.19 s     a=0.95 -> 0.39 s     a=0.98 -> 0.99 s     a=0.995 -> 2.0 s

Anything slower than the corner is removed. c0 wants a slow corner (keep the sustain), c1 wants
a fast one (drop the speaker's fixed tilt) — which is exactly what a single shared filter
cannot give.

Every operation is causal: the STFT is the same `center=False` frame grid the content path
uses, and the EMAs are single-pole IIR filters run forward only. No lookahead is added.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torchaudio

_SPEC_RE = re.compile(r"^c(\d+):(abs|delta|hp([\d.]+)|ms([\d.]+),([\d.]+))$")


def _parse_spec(spec: str) -> list[tuple[int, str, tuple[float, ...]]]:
    """'c0:abs,c1:ms0.9,0.99' -> [(0,'abs',()), (1,'ms',(0.9,0.99))]."""
    out = []
    for part in re.split(r",(?=c\d+:)", str(spec).strip()):
        part = part.strip()
        if not part:
            continue
        m = _SPEC_RE.match(part)
        if not m:
            raise ValueError(
                f"bad cepstral_spec entry {part!r}. Expected c<i>:abs | c<i>:delta | "
                f"c<i>:hp<alpha> | c<i>:ms<fast>,<slow>  (e.g. 'c0:abs,c1:hp0.95')")
        idx = int(m.group(1))
        if m.group(2) == "abs":
            out.append((idx, "abs", ()))
        elif m.group(2) == "delta":
            out.append((idx, "delta", ()))
        elif m.group(3):
            out.append((idx, "hp", (float(m.group(3)),)))
        else:
            f, s = float(m.group(4)), float(m.group(5))
            if not 0.0 < f < s < 1.0:
                raise ValueError(f"ms needs 0 < fast < slow < 1, got {f},{s} in {part!r}")
            out.append((idx, "ms", (f, s)))
    if not out:
        raise ValueError("cepstral_spec is empty")
    return out


class CepstralExtractor(nn.Module):
    """(B, T_samples) -> (B, T_frames, hidden_dim), frame-aligned to the codec grid."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sr = int(config.sample_rate)
        self.hop_length = int(config.prosody_hop_length)
        self.n_fft = int(config.n_fft)
        self.spec = _parse_spec(getattr(config, "cepstral_spec", "c0:abs,c1:delta"))
        self.n_coef = max(i for i, _, _ in self.spec) + 1

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length,
            power=1.0, normalized=False, center=False,
        )
        # DCT-II over log-magnitude bins → cepstral coefficients. Only the first `n_coef`
        # rows are ever needed, so the basis is stored truncated.
        n_bins = self.n_fft // 2 + 1
        k = torch.arange(self.n_coef, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(n_bins, dtype=torch.float32).unsqueeze(0)
        basis = torch.cos(torch.pi * k * (2 * n + 1) / (2 * n_bins))
        basis[0] /= 2.0 ** 0.5
        self.register_buffer("dct", basis * (2.0 / n_bins) ** 0.5)     # (n_coef, n_bins)

        # one input channel per spec entry, two for multi-scale
        self.n_channels = sum(2 if m == "ms" else 1 for _, m, _ in self.spec)
        hid = int(getattr(config, "cepstral_hidden_dim", config.prosody_hidden_dim))
        # A single Linear over the stacked channels IS the per-band learnable weighting:
        # column j of its weight is W for channel j, so fast and mid bands are weighted
        # independently rather than summed before projection.
        self.in_proj = nn.Linear(self.n_channels, hid)
        self.out_proj = nn.Sequential(
            nn.Linear(hid, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _ema(x: torch.Tensor, alpha: float) -> torch.Tensor:
        """Causal single-pole EMA along time, warm-started at x[0]. x: (B, T).

        lfilter initialises y[-1] = 0, which is wrong here: c0 is a log energy sitting near
        -100, so a cold start spends one time constant climbing to the signal and the
        high-pass residual over that stretch is the transient, not the feature. Since the
        filter is linear, running it on (x - x[0]) and adding x[0] back is exactly the same
        filter warm-started at x[0], with no transient at all.
        """
        x0 = x[:, :1]
        b = torch.tensor([1.0 - alpha, 0.0], device=x.device, dtype=x.dtype)
        a = torch.tensor([1.0, -alpha], device=x.device, dtype=x.dtype)
        return x0 + torchaudio.functional.lfilter(x - x0, a, b, clamp=False)

    def _cepstrum(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, T_samples) -> (B, T_frames, n_coef)."""
        mag = self.stft(audio.float()).transpose(1, 2)             # (B, T, bins)
        logmag = torch.log(mag.clamp_min(1e-8))
        return logmag @ self.dct.t()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        c = self._cepstrum(audio)                                   # (B, T, n_coef)
        chans = []
        for idx, mode, args in self.spec:
            x = c[..., idx]                                         # (B, T)
            if mode == "abs":
                chans.append(x)
            elif mode == "delta":
                # first frame has no predecessor; 0 rather than a copy, so the channel
                # starts at "no change" instead of at the speaker's absolute level
                chans.append(torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1))
            elif mode == "hp":
                chans.append(x - self._ema(x, args[0]))
            else:                                                   # ms
                fast = self._ema(x, args[0])
                slow = self._ema(x, args[1])
                chans.append(x - fast)                              # fast band
                chans.append(fast - slow)                           # mid band
        feats = torch.stack(chans, dim=-1)                          # (B, T, n_channels)
        return self.out_proj(self.in_proj(feats))

    def extra_repr(self) -> str:
        return f"spec={getattr(self.config, 'cepstral_spec', '')!r}, channels={self.n_channels}"
