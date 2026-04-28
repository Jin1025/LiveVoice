"""Mimi codec wrapper (kyutai/mimi) for LiveVoice.

Specs:
  sample_rate   : 24 000 Hz
  hop_length    : 1 920 samples  (12.5 fps)
  n_codebooks   : 8 (default, configurable via config.mimi_n_codebooks)
  codebook_size : 2 048
  latent_dim    : 512

Interface is identical to DACModel so the rest of the codebase is codec-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class MimiCodec(nn.Module):
    """Frozen Mimi codec: audio ↔ discrete codes ↔ continuous z."""

    SAMPLE_RATE   = 24_000
    HOP_LENGTH    = 1_920   # 24000 / 12.5
    CODEBOOK_SIZE = 2_048
    LATENT_DIM    = 512

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        # SR of audio coming in (may differ from Mimi's native 24 kHz)
        self.input_sr = int(getattr(config, "sample_rate", self.SAMPLE_RATE))

        from transformers import MimiModel
        self.model = MimiModel.from_pretrained(
            getattr(config, "mimi_model_name", "kyutai/mimi")
        )
        self.model.eval()
        self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.sample_rate   = self.SAMPLE_RATE
        self.n_codebooks   = int(getattr(config, "mimi_n_codebooks", 8))
        self.codebook_size = self.CODEBOOK_SIZE
        self.hop_length    = self.HOP_LENGTH
        self.latent_dim    = self.LATENT_DIM

    # ------------------------------------------------------------------
    def _prep(self, audio: torch.Tensor) -> torch.Tensor:
        """Ensure (B, 1, T) float32 on device at Mimi's native 24 kHz."""
        audio = audio.to(self.device).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)
        if self.input_sr != self.SAMPLE_RATE:
            audio = torchaudio.functional.resample(
                audio.squeeze(1), self.input_sr, self.SAMPLE_RATE
            ).unsqueeze(1)
        return audio

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, T) → codes (B, n_codebooks, T_frames)."""
        out = self.model.encode(self._prep(audio), num_quantizers=self.n_codebooks)
        return out.audio_codes  # (B, K, T)

    @torch.no_grad()
    def encode_continuous(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T) → (codes (B,K,T), z (B, T_frames, latent_dim))."""
        audio = self._prep(audio)
        out = self.model.encode(audio, num_quantizers=self.n_codebooks)
        codes = out.audio_codes  # (B, K, T)

        # Continuous encoder representation (pre-quantisation) for speaker embedding
        T_frames = codes.shape[-1]
        try:
            enc = self.model.encoder(audio)
            # IMPORTANT: do NOT do enc[0] on a plain tensor — that slices batch dim.
            if hasattr(enc, "last_hidden_state"):
                z = enc.last_hidden_state
            elif isinstance(enc, torch.Tensor):
                z = enc
            else:
                z = enc[0]

            if z.dim() == 2:
                z = z.unsqueeze(0).expand(audio.shape[0], -1, -1)

            # Rotate to channels-last (B, T, D) if currently (B, D, T)
            if z.shape[1] == self.latent_dim and z.shape[1] != z.shape[2]:
                z = z.transpose(1, 2)

            # Temporal alignment: encoder may run at a different fps than quantizer
            if z.shape[1] != T_frames:
                z = torch.nn.functional.interpolate(
                    z.float().transpose(1, 2),
                    size=T_frames,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            if z.shape[2] > self.latent_dim:
                z = z[..., :self.latent_dim]
            elif z.shape[2] < self.latent_dim:
                pad = z.new_zeros(*z.shape[:2], self.latent_dim - z.shape[2])
                z = torch.cat([z, pad], dim=-1)

            assert z.shape == (audio.shape[0], T_frames, self.latent_dim), \
                f"MimiCodec.encode_continuous: unexpected z shape {z.shape}, " \
                f"expected ({audio.shape[0]}, {T_frames}, {self.latent_dim})"
        except AssertionError:
            raise
        except Exception:
            z = torch.zeros(audio.shape[0], T_frames, self.latent_dim, device=self.device)
        return codes, z

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, K, T_frames) → audio (B, T_samples)."""
        codes = codes.to(self.device)
        out = self.model.decode(audio_codes=codes)
        audio = out.audio_values  # (B, 1, T) or (B, T)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return audio

    def get_codebook_embeddings(self, k: int) -> torch.Tensor | None:
        """Try to return (codebook_size, latent_dim) embedding table for codebook k."""
        try:
            layer = self.model.quantizer.layers[k]
            for attr in ["codebook", "_codebook"]:
                obj = getattr(layer, attr, None)
                if obj is None:
                    continue
                for sub in ["embed", "weight", "embeddings"]:
                    emb = getattr(obj, sub, None)
                    if emb is not None:
                        emb = emb.detach().float()
                        if emb.dim() == 3:          # (1, codebook_size, dim)
                            emb = emb.squeeze(0)
                        return emb                  # (codebook_size, dim)
        except Exception:
            pass
        return None
