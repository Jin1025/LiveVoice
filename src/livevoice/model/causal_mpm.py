"""Causal Masked Prosody Model — streaming-compatible prosody feature extractor.

Adapted from the Masked Prosody Model (Wallbridge et al., Interspeech 2025,
"Prosodic structure beyond lexical content: A study of self-supervised learning").

Key changes from the original bidirectional MPM:

    1. **Causal.**  Conv1d pads left only; self-attention wears a causal mask.
       No future frames are read, so it slots into a streaming pipeline.
    2. **GPU-native features.**  Pitch via causal YIN (torch, batched);
       energy via ``center=False`` STFT + A-weighted RMS.  No numpy, no pyworld.
    3. **Discrete inputs** like the original: pitch and energy are quantised into
       ``n_bins`` uniform bins and looked up via ``nn.Embedding``.  Quantisation acts
       as an information bottleneck.  Bins 0..n_bins-1 are data; bin n_bins is [MASK].
    4. **Speaker-normalised pitch.**  Causal running-mean subtraction so the model
       never sees the speaker's absolute register.

Training follows the paper: "random masking" strategy samples mask segment length
m ~ Uniform(1, mask_length_max) per batch, places random contiguous segments until
~50% coverage. Loss: CE per feature normalised by 1/log(c).

Architecture:
    pitch  → bucketize(n_bins) → Embedding(n_bins+1, D)  ─┐
    energy → bucketize(n_bins) → Embedding(n_bins+1, D)  ─┼─ add → CausalConformer × N → heads
    VAD    → {0,1}             → Embedding(3, D)         ─┘

    Pretraining loss:  CE on masked frames (pitch_bin, energy_bin) + BCE (vad), / log(c).
    Inference output:  hidden state from layer ``output_layer`` → (B, T, filter_size).
"""
from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CausalMPMConfig:
    n_layers: int = 16
    n_heads: int = 4
    filter_size: int = 256
    conv_kernel_size: int = 7
    dropout: float = 0.1

    # feature extraction
    sample_rate: int = 16000
    hop_length: int = 320
    n_fft: int = 1024

    # Analysis-window placement.  With plain center=False, frame t spans
    # [t*hop, t*hop + n_fft) and therefore reads (n_fft - hop) = 704 samples (44 ms)
    # PAST the end of codec frame t -- a real lookahead, even though the encoder itself
    # is causal.  causal_window=True front-pads by (n_fft - hop) so the window ENDS at
    # the frame boundary, and tail-pads the final block so the frame count is exactly
    # ceil(L/hop): the same grid the codec and the Zipformer content path emit.
    # Set False only to reproduce checkpoints trained before this fix.
    causal_window: bool = True

    # pitch (YIN)
    pitch_fmin: float = 55.0
    pitch_fmax: float = 500.0
    yin_threshold: float = 0.15

    # speaker-normalised pitch
    pitch_normalize: bool = True
    pitch_prior_frames: float = 25.0
    pitch_prior_hz: float = 150.0

    # quantisation
    n_bins: int = 128

    # masking (pretraining) — follows Wallbridge et al. Interspeech 2025
    # "random": sample m ~ U(1, mask_length_max) per batch, place segments until ~50% covered
    # "span":  geometric-length spans (legacy, mean = mask_span_mean)
    mask_strategy: str = "random"
    mask_prob: float = 0.50
    mask_jitter: float = 0.05      # coverage target = mask_prob ± mask_jitter
    mask_length_max: int = 128     # for "random" strategy: m ~ U(1, mask_length_max)
    mask_span_mean: float = 5.0   # for "span" strategy only

    # optional extra targets
    use_bap: bool = False       # Band Aperiodicity (5 bands)
    bap_n_bands: int = 5
    use_cpps: bool = False      # Cepstral Peak Prominence Smoothed
    cpps_lo: float = -5.0       # CPPS normalisation range
    cpps_hi: float = 25.0

    # which layer's hidden state to output at inference (0-indexed)
    output_layer: int = 7  # 8th layer (middle of 16)


# ---------------------------------------------------------------------------
# Causal Conformer building blocks
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    """Conv1d that pads only on the left — no future leakage."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConformerLayer(nn.Module):
    """Pre-norm Conformer block: causal self-attn → causal conv → FFN."""

    def __init__(self, d_model: int, n_heads: int, kernel_size: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.conv1 = CausalConv1d(d_model, d_model, kernel_size)
        self.conv2 = CausalConv1d(d_model, d_model, 1)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, h, h, attn_mask=attn_mask, is_causal=False)[0]
        x = x + self.drop1(h)
        h = self.norm2(x)
        h = self.conv2(self.drop2(F.gelu(self.conv1(h.transpose(1, 2))))).transpose(1, 2)
        x = x + h
        h = self.norm3(x)
        h = self.ff2(self.drop3(F.gelu(self.ff1(h))))
        x = x + self.drop3(h)
        return x


class CausalConformerEncoder(nn.Module):
    def __init__(self, cfg: CausalMPMConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalConformerLayer(cfg.filter_size, cfg.n_heads, cfg.conv_kernel_size, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.filter_size)
        self.output_layer = cfg.output_layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (final_output, output_at_target_layer)."""
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_mask = torch.zeros(T, T, device=x.device, dtype=x.dtype)
        attn_mask.masked_fill_(mask, float("-inf"))

        target = self.output_layer
        if target < 0:
            target = len(self.layers) + target

        intermediate = x
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask)
            if i == target:
                intermediate = x.clone()

        return self.norm(x), intermediate


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)].to(x.dtype))


# ---------------------------------------------------------------------------
# Feature extraction (causal, GPU-native)
# ---------------------------------------------------------------------------
_BAP_BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000)]


class CausalFeatureExtractor(nn.Module):
    """Pitch (YIN) + Mel RMS energy + VAD + optional BAP/CPPS, all causal, all torch."""

    def __init__(self, cfg: CausalMPMConfig):
        super().__init__()
        self.sr = cfg.sample_rate
        self.hop = cfg.hop_length
        self.n_fft = cfg.n_fft
        self.fmin = cfg.pitch_fmin
        self.fmax = cfg.pitch_fmax
        self.yin_threshold = cfg.yin_threshold
        self.pitch_normalize = cfg.pitch_normalize
        self.pitch_prior_frames = cfg.pitch_prior_frames
        self.pitch_prior_hz = cfg.pitch_prior_hz
        self.n_bins = cfg.n_bins
        self.causal_window = bool(getattr(cfg, "causal_window", True))
        self.use_bap = getattr(cfg, "use_bap", False)
        self.use_cpps = getattr(cfg, "use_cpps", False)
        self.cpps_lo = getattr(cfg, "cpps_lo", -5.0)
        self.cpps_hi = getattr(cfg, "cpps_hi", 25.0)

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=80, power=2.0, center=False,
        )
        self.register_buffer("bin_edges", torch.linspace(0, 1, cfg.n_bins + 1)[1:-1])

        self.tau_min = max(2, int(self.sr / self.fmax))
        self.tau_max = min(int(self.sr / self.fmin) + 1, self.n_fft // 2)
        self.yin_win = self.n_fft - self.tau_max

        if self.use_bap:
            self._build_bap_table(getattr(cfg, "bap_n_bands", 5))

    def _causal_pad(self, audio: torch.Tensor) -> torch.Tensor:
        """Shift the analysis grid so frame t ENDS at the end of codec frame t.

        front = n_fft - hop moves the window from [t*hop, t*hop + n_fft) to
        [t*hop - front, t*hop + hop): no future sample is ever read, so the prosody
        branch contributes 0 ms of lookahead instead of 44 ms.

        tail = ceil(L/hop)*hop - L end-pads the final block (jhcodec pads its own the
        same way) so MelSpectrogram/unfold emit exactly ceil(L/hop) frames, 1:1 with the
        codec grid.  Without it the extractor is 3-4 frames SHORT and align_to_tokens
        linspace-stretches the sequence onto the codec timeline, which makes prosody lag
        progressively -- measured at up to 3 frames (60 ms) by the end of an utterance.
        """
        L = int(audio.shape[-1])
        n_target = -(-L // self.hop)                     # ceil == codec frame count
        front = self.n_fft - self.hop
        tail = max(0, n_target * self.hop - L)
        return F.pad(audio, (front, tail))

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """(B, T_samples) → dict with pitch_bin, energy_bin, vad (+ optional bap_bins, cpps_bin)."""
        audio = audio.float()
        if self.causal_window:
            # every framing op below (mel, YIN, BAP, CPPS) reads this padded tensor, so
            # they all land on the same causal, codec-aligned grid.
            audio = self._causal_pad(audio)

        # energy: RMS of Mel Spectrogram frame (Wallbridge et al.)
        mel = self.mel_spec(audio).transpose(1, 2)  # (B, T, n_mels), power
        rms = torch.sqrt(mel.mean(-1).clamp_min(1e-10))
        energy_db = 20.0 * torch.log10(rms + 1e-8)
        energy_norm = ((energy_db + 60.0) / 60.0).clamp(0.0, 1.0)
        energy_bin = torch.bucketize(energy_norm, self.bin_edges)

        # pitch via YIN
        f0, voiced = self._yin(audio)

        # speaker normalisation → [0,1] range → bucketise
        if self.pitch_normalize:
            pitch_norm = self._normalize_pitch(f0, voiced)
            pitch_norm = (pitch_norm + 1.0) * 0.5
        else:
            pitch_norm = (f0 - self.fmin) / (self.fmax - self.fmin)
        pitch_norm = pitch_norm.clamp(0.0, 1.0)
        pitch_bin = torch.bucketize(pitch_norm, self.bin_edges)
        pitch_bin = pitch_bin * voiced.long()

        T = energy_bin.size(1)
        if pitch_bin.size(1) != T:
            pitch_bin = pitch_bin[:, :T] if pitch_bin.size(1) > T else F.pad(pitch_bin, (0, T - pitch_bin.size(1)))
            voiced = voiced[:, :T] if voiced.size(1) > T else F.pad(voiced, (0, T - voiced.size(1)))
            f0 = f0[:, :T] if f0.size(1) >= T else F.pad(f0, (0, T - f0.size(1)))

        vad = voiced.long()
        result: dict[str, torch.Tensor] = {
            "pitch_bin": pitch_bin, "energy_bin": energy_bin, "vad": vad,
        }

        if self.use_bap:
            bap_raw = self._extract_bap(audio, f0, voiced, T)  # (B, T, n_bands) in [0,1]
            bap_bins = torch.stack(
                [torch.bucketize(bap_raw[:, :, i], self.bin_edges) for i in range(bap_raw.size(-1))],
                dim=-1,
            )  # (B, T, n_bands)
            result["bap_bins"] = bap_bins

        if self.use_cpps:
            cpps_raw = self._extract_cpps(audio, T)  # (B, T)
            cpps_norm = ((cpps_raw - self.cpps_lo) / (self.cpps_hi - self.cpps_lo)).clamp(0.0, 1.0)
            result["cpps_bin"] = torch.bucketize(cpps_norm, self.bin_edges)

        return result

    def _yin(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W, tmax = self.yin_win, self.tau_max
        frames = audio.unfold(-1, self.n_fft, self.hop)
        B, T, _ = frames.shape
        x = frames.reshape(B * T, self.n_fft)

        cs = torch.cumsum(F.pad(x * x, (1, 0)), dim=-1)
        p = cs[:, W:] - cs[:, :-W]
        p = p[:, :tmax + 1]
        n = 1 << int(self.n_fft + W - 1).bit_length()
        r = torch.fft.irfft(
            torch.fft.rfft(x, n=n) * torch.fft.rfft(x[:, :W], n=n).conj(), n=n
        )[:, :tmax + 1]
        d = (p[:, :1] + p - 2.0 * r).clamp_min(0.0)

        cum = torch.cumsum(d, dim=-1)
        lag = torch.arange(1, tmax + 1, device=x.device, dtype=d.dtype)
        dn = torch.ones_like(d)
        dn[:, 1:] = d[:, 1:] * lag / cum[:, 1:].clamp_min(1e-12)

        cand = dn[:, self.tau_min:tmax + 1]
        below = cand < self.yin_threshold
        first = torch.where(
            below.any(dim=-1), below.float().argmax(dim=-1), cand.argmin(dim=-1)
        ) + self.tau_min

        i = first.clamp(1, tmax - 1)
        y0 = torch.gather(dn, 1, (i - 1).unsqueeze(1)).squeeze(1)
        y1 = torch.gather(dn, 1, i.unsqueeze(1)).squeeze(1)
        y2 = torch.gather(dn, 1, (i + 1).unsqueeze(1)).squeeze(1)
        denom = y0 - 2 * y1 + y2
        shift = torch.where(denom.abs() > 1e-12, 0.5 * (y0 - y2) / denom, torch.zeros_like(denom))
        tau = i.to(d.dtype) + shift.clamp(-1.0, 1.0)
        f0 = (self.sr / tau.clamp_min(1e-6)).reshape(B, T)

        voiced = ((y1 < self.yin_threshold).reshape(B, T)
                  & (f0 >= self.fmin) & (f0 <= self.fmax))
        return f0, voiced

    def _normalize_pitch(self, f0: torch.Tensor, voiced: torch.Tensor) -> torch.Tensor:
        lf = torch.log2(f0.clamp_min(1e-6)) * 1200.0
        m = voiced.float()
        n0 = self.pitch_prior_frames
        p0 = 1200.0 * torch.log2(torch.tensor(self.pitch_prior_hz, dtype=lf.dtype, device=lf.device))
        num = torch.cumsum(lf * m, dim=1) + n0 * p0
        den = torch.cumsum(m, dim=1) + n0
        deviation = lf - num / den.clamp_min(1e-6)
        return deviation / 1200.0

    # ── BAP (Band Aperiodicity) ────────────────────────────────────────
    def _build_bap_table(self, n_bands: int = 5):
        """Precompute harmonic masks for quantised f0 values → fast lookup at runtime."""
        N_F0 = 512
        F_bins = self.n_fft // 2 + 1
        f0_vals = torch.linspace(self.fmin, self.fmax, N_F0)
        table = torch.zeros(N_F0, F_bins)
        for i in range(N_F0):
            f0 = f0_vals[i].item()
            hw = max(1, int(f0 * 0.25 / (self.sr / 2) * F_bins))
            for h in range(1, int(self.sr / 2 / f0) + 1):
                c = int(h * f0 / (self.sr / 2) * (F_bins - 1))
                lo = max(0, c - hw)
                hi = min(F_bins, c + hw + 1)
                table[i, lo:hi] = 1.0
        self.register_buffer("_bap_harm_table", table)
        self._bap_n_f0 = N_F0

        freq_axis = torch.linspace(0, self.sr / 2, F_bins)
        bands = _BAP_BANDS[:n_bands]
        slices = []
        for blo, bhi in bands:
            lo_bin = int((freq_axis >= blo).nonzero()[0].item()) if blo > 0 else 0
            hi_bin = int((freq_axis <= bhi).nonzero()[-1].item()) + 1
            slices.append((lo_bin, min(hi_bin, F_bins)))
        self._bap_band_slices = slices

    @torch.no_grad()
    def _extract_bap(self, audio: torch.Tensor, f0: torch.Tensor,
                     voiced: torch.Tensor, T: int) -> torch.Tensor:
        """(B, T_samples), (B, T), (B, T), int → (B, T, n_bands) in [0,1]."""
        B_sz = audio.size(0)
        device = audio.device
        frames = audio.unfold(-1, self.n_fft, self.hop)
        T_out = min(T, frames.size(1))
        win = torch.hann_window(self.n_fft, device=device)
        spec = torch.fft.rfft(frames[:, :T_out] * win)
        mag2 = spec.abs().square()
        F_bins = mag2.size(-1)

        f0_clip = f0[:, :T_out].clamp(self.fmin, self.fmax)
        f0_idx = ((f0_clip - self.fmin) / (self.fmax - self.fmin) * (self._bap_n_f0 - 1)).long()
        harm_mask = self._bap_harm_table[f0_idx]  # (B, T_out, F)

        n_bands = len(self._bap_band_slices)
        bap = torch.ones(B_sz, T_out, n_bands, device=device)
        for bi, (lo, hi) in enumerate(self._bap_band_slices):
            if hi <= lo:
                continue
            band_pow = mag2[:, :, lo:hi].sum(dim=-1)
            harm_pow = (mag2[:, :, lo:hi] * harm_mask[:, :, lo:hi]).sum(dim=-1)
            bap[:, :, bi] = (1.0 - harm_pow / band_pow.clamp_min(1e-12)).clamp(0, 1)

        bap[~voiced[:, :T_out].unsqueeze(-1).expand_as(bap)] = 1.0
        if T_out < T:
            bap = F.pad(bap, (0, 0, 0, T - T_out), value=1.0)
        return bap

    # ── CPPS (Cepstral Peak Prominence Smoothed) ──────────────────────
    @torch.no_grad()
    def _extract_cpps(self, audio: torch.Tensor, T: int) -> torch.Tensor:
        """(B, T_samples) → (B, T) CPPS values (vectorised, GPU-native)."""
        device = audio.device
        frames = audio.unfold(-1, self.n_fft, self.hop)
        T_out = min(T, frames.size(1))
        win = torch.hann_window(self.n_fft, device=device)
        windowed = frames[:, :T_out] * win

        n2 = self.n_fft * 2
        spec = torch.fft.rfft(windowed, n=n2)
        log_pow = torch.log(spec.abs().square().clamp_min(1e-12))
        cepstrum = torch.fft.irfft(log_pow).abs().square()

        q_min = int(self.sr / self.fmax)
        q_max = min(int(self.sr / self.fmin), cepstrum.size(-1) - 1)
        cep = cepstrum[:, :, q_min:q_max + 1]
        log_cep = torch.log(cep.clamp_min(1e-12))

        qs = torch.arange(q_min, q_max + 1, dtype=torch.float32, device=device)
        q_mean = qs.mean()
        q_cen = qs - q_mean
        denom = (q_cen ** 2).sum()

        y_mean = log_cep.mean(dim=-1, keepdim=True)
        slope = ((log_cep - y_mean) * q_cen).sum(dim=-1) / denom
        intercept = y_mean.squeeze(-1) - slope * q_mean
        regression = slope.unsqueeze(-1) * qs + intercept.unsqueeze(-1)

        peak_idx = log_cep.argmax(dim=-1)
        peak_val = log_cep.gather(2, peak_idx.unsqueeze(-1)).squeeze(-1)
        reg_at_peak = regression.gather(2, peak_idx.unsqueeze(-1)).squeeze(-1)
        cpp = peak_val - reg_at_peak

        kernel = torch.ones(1, 1, 5, device=device) / 5.0
        cpps = F.conv1d(F.pad(cpp.unsqueeze(1), (4, 0), mode="replicate"), kernel).squeeze(1)
        if T_out < T:
            cpps = F.pad(cpps, (0, T - T_out))
        return cpps


# ---------------------------------------------------------------------------
# CausalMPM
# ---------------------------------------------------------------------------
class CausalMPM(nn.Module):
    """Causal Masked Prosody Model.

    Pretraining: span-mask pitch/energy/VAD, predict the masked bins (CE).
    Inference:   extract intermediate hidden state as prosody latent.
    """

    def __init__(self, cfg: CausalMPMConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = CausalMPMConfig()
        self.cfg = cfg
        D = cfg.filter_size
        B = cfg.n_bins

        self.feature_extractor = CausalFeatureExtractor(cfg)

        # discrete embeddings: bins 0..B-1 = data, bin B = [MASK]
        self.pitch_emb = nn.Embedding(B + 1, D)
        self.energy_emb = nn.Embedding(B + 1, D)
        self.vad_emb = nn.Embedding(3, D)  # 0=unvoiced, 1=voiced, 2=[MASK]

        self.pe = SinusoidalPE(D, dropout=cfg.dropout)
        self.encoder = CausalConformerEncoder(cfg)

        # pretraining heads: predict bin index
        self.head_pitch = nn.Linear(D, B)
        self.head_energy = nn.Linear(D, B)
        self.head_vad = nn.Linear(D, 1)

        if getattr(cfg, "use_bap", False):
            n_bands = getattr(cfg, "bap_n_bands", 5)
            self.bap_embs = nn.ModuleList([nn.Embedding(B + 1, D) for _ in range(n_bands)])
            self.head_bap = nn.ModuleList([nn.Linear(D, B) for _ in range(n_bands)])

        if getattr(cfg, "use_cpps", False):
            self.cpps_emb = nn.Embedding(B + 1, D)
            self.head_cpps = nn.Linear(D, B)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _make_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        if self.cfg.mask_strategy == "random":
            return self._make_random_mask(B, T, device)
        return self._make_span_mask(B, T, device)

    def _make_random_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Paper's 'random masking': sample m ~ U(1, mask_length_max) per batch,
        place random contiguous segments of length m until target coverage."""
        mask = torch.zeros(B, T, device=device, dtype=torch.bool)
        m = _random.randint(1, self.cfg.mask_length_max)
        target = self.cfg.mask_prob + (_random.random() - 0.5) * 2.0 * self.cfg.mask_jitter
        n_target = int(T * target)
        for b in range(B):
            n_masked = 0
            while n_masked < n_target:
                start = _random.randint(0, max(0, T - 1))
                end = min(start + m, T)
                n_new = int((~mask[b, start:end]).sum().item())
                mask[b, start:end] = True
                n_masked += n_new
                if n_new == 0:
                    break
        return mask

    def _make_span_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Legacy: geometric-length spans, target coverage = mask_prob."""
        mask = torch.zeros(B, T, device=device, dtype=torch.bool)
        p_start = self.cfg.mask_prob / self.cfg.mask_span_mean
        p_continue = 1.0 - 1.0 / self.cfg.mask_span_mean
        for b in range(B):
            in_span = False
            for t in range(T):
                if in_span:
                    if _random.random() < p_continue:
                        mask[b, t] = True
                    else:
                        in_span = False
                else:
                    if _random.random() < p_start:
                        mask[b, t] = True
                        in_span = True
        return mask

    def forward_pretrain(
        self, audio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Pretraining forward: extract features, mask, predict masked bins.

        Returns dict with keys: loss, loss_pitch, loss_energy, loss_vad, mask_ratio,
        and optionally loss_bap / loss_cpps.
        """
        feats = self.feature_extractor(audio)
        pitch_bin = feats["pitch_bin"]
        energy_bin = feats["energy_bin"]
        vad = feats["vad"]
        Batch, T = pitch_bin.shape
        mask_token = self.cfg.n_bins

        mask = self._make_mask(Batch, T, audio.device)

        pitch_in = pitch_bin.clone()
        energy_in = energy_bin.clone()
        vad_in = vad.clone()
        pitch_in[mask] = mask_token
        energy_in[mask] = mask_token
        vad_in[mask] = 2

        h = self.pitch_emb(pitch_in) + self.energy_emb(energy_in) + self.vad_emb(vad_in)

        if hasattr(self, "bap_embs"):
            bap_bins = feats["bap_bins"]  # (B, T, n_bands)
            bap_in = bap_bins.clone()
            bap_in[mask.unsqueeze(-1).expand_as(bap_in)] = mask_token
            for i, emb in enumerate(self.bap_embs):
                h = h + emb(bap_in[:, :, i])

        if hasattr(self, "cpps_emb"):
            cpps_bin = feats["cpps_bin"]
            cpps_in = cpps_bin.clone()
            cpps_in[mask] = mask_token
            h = h + self.cpps_emb(cpps_in)

        h = self.pe(h)
        out, _ = self.encoder(h)

        logits_pitch = self.head_pitch(out)
        logits_energy = self.head_energy(out)
        logits_vad = self.head_vad(out).squeeze(-1)

        m = mask.float()
        n_masked = m.sum().clamp_min(1.0)
        inv_log_c = 1.0 / math.log(self.cfg.n_bins)

        loss_pitch = (F.cross_entropy(
            logits_pitch.view(-1, self.cfg.n_bins), pitch_bin.view(-1),
            reduction="none").view(Batch, T) * m).sum() / n_masked * inv_log_c
        loss_energy = (F.cross_entropy(
            logits_energy.view(-1, self.cfg.n_bins), energy_bin.view(-1),
            reduction="none").view(Batch, T) * m).sum() / n_masked * inv_log_c
        loss_vad = (F.binary_cross_entropy_with_logits(
            logits_vad, vad.float(), reduction="none") * m).sum() / n_masked

        loss = loss_pitch + loss_energy + loss_vad

        result = {
            "loss_pitch": loss_pitch.detach(),
            "loss_energy": loss_energy.detach(),
            "loss_vad": loss_vad.detach(),
            "mask_ratio": m.mean().detach(),
        }

        if hasattr(self, "head_bap"):
            bap_bins = feats["bap_bins"]
            loss_bap = torch.tensor(0.0, device=audio.device)
            for i, head in enumerate(self.head_bap):
                logits = head(out)
                l = (F.cross_entropy(
                    logits.view(-1, self.cfg.n_bins), bap_bins[:, :, i].reshape(-1),
                    reduction="none").view(Batch, T) * m).sum() / n_masked * inv_log_c
                loss_bap = loss_bap + l
            loss = loss + loss_bap
            result["loss_bap"] = loss_bap.detach()

        if hasattr(self, "head_cpps"):
            cpps_bin = feats["cpps_bin"]
            logits = self.head_cpps(out)
            loss_cpps = (F.cross_entropy(
                logits.view(-1, self.cfg.n_bins), cpps_bin.view(-1),
                reduction="none").view(Batch, T) * m).sum() / n_masked * inv_log_c
            loss = loss + loss_cpps
            result["loss_cpps"] = loss_cpps.detach()

        result["loss"] = loss
        return result

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """Inference: extract prosody latent from target layer. (B, T_frames, filter_size)."""
        with torch.no_grad():
            feats = self.feature_extractor(audio)
        pitch_bin = feats["pitch_bin"]
        energy_bin = feats["energy_bin"]
        vad = feats["vad"]
        h = self.pitch_emb(pitch_bin) + self.energy_emb(energy_bin) + self.vad_emb(vad)

        if hasattr(self, "bap_embs"):
            bap_bins = feats["bap_bins"]
            for i, emb in enumerate(self.bap_embs):
                h = h + emb(bap_bins[:, :, i])
        if hasattr(self, "cpps_emb"):
            h = h + self.cpps_emb(feats["cpps_bin"])

        h = self.pe(h)
        _, intermediate = self.encoder(h)
        return intermediate

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Default forward = extract (for use as a frozen feature extractor)."""
        return self.extract(audio)

    @property
    def output_dim(self) -> int:
        return self.cfg.filter_size
