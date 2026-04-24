"""Causal F0 + loudness prosody extractor (optional conditioning).

Trimmed-down, speech-focused version of sonic's SketchFeatureExtractor:
- A-weighted loudness from magnitude STFT
- Pitch-probability distribution via CREPE ("tiny") or FFT fallback
- No spectral centroid, no rhythm bands — speech prosody is mostly F0 + energy
- Optional random median filter to make the control sketch-like across speakers

All operations are causal (center=False, left-pad only).
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

        self.pitch_method = str(getattr(config, "pitch_method", "crepe"))
        self.pitch_bins = int(config.pitch_bins)
        self.pitch_threshold = float(config.pitch_threshold)
        self.crepe_available = False
        if self.pitch_method == "crepe":
            try:
                import torchcrepe  # noqa: F401
                self.crepe_available = True
            except ImportError:
                print("[ProsodyExtractor] torchcrepe not installed — falling back to FFT pitch.")
                self.pitch_method = "fft"

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

    def _pitch_fft(self, audio: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=audio.device),
            return_complex=True, center=False,
        )
        mag = torch.abs(stft).transpose(1, 2)  # (B, T, F)
        min_bin = int(50 * self.n_fft / self.sr)
        max_bin = int(2000 * self.n_fft / self.sr)
        pm = mag[..., min_bin:max_bin]
        B, T, P = pm.shape
        pm = F.interpolate(pm.reshape(B * T, 1, P), size=self.pitch_bins, mode="linear", align_corners=False)
        pm = pm.reshape(B, T, self.pitch_bins)
        probs = F.softmax(pm / 1.1, dim=-1)
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
        if self.pitch_method == "crepe" and self.crepe_available:
            pitch = self._pitch_crepe(audio)
        else:
            pitch = self._pitch_fft(audio)

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
