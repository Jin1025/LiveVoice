"""DAC codec wrapper for LiveVoice (speech-optimized 16 kHz model).

Mirrors sonic/src/Foley/model/dac/dac_model.py but defaults to the 16 kHz
speech model: 12 RVQ codebooks, hop=320, sr=16000 → 50 frames/sec.
"""
import torch
import torch.nn as nn
import dac


class DACModel(nn.Module):
    """Frozen DAC codec: audio ↔ discrete codes ↔ continuous z."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

        model_type = config.dac_model_type  # "16khz" for speech
        model_path = dac.utils.download(model_type=model_type)
        self.dac_model = dac.DAC.load(model_path)
        self.dac_model.eval()
        self.dac_model.to(self.device)

        for p in self.dac_model.parameters():
            p.requires_grad = False

        self.sample_rate = config.dac_sample_rate
        self.n_codebooks = config.dac_n_codebooks
        self.depth = config.dac_depth
        self.codebook_size = config.dac_codebook_size
        self.hop_length = getattr(self.dac_model, "hop_length", config.dac_hop_length)
        self.latent_dim = config.dac_latent_dim

        self.to(self.device)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to discrete RVQ codes.

        Args:
            audio: (B, T) or (B, 1, T) at config.dac_sample_rate

        Returns:
            codes: (B, n_codebooks, T_frames)
        """
        audio = audio.to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        _, codes, _, _, _ = self.dac_model.encode(audio)
        return codes

    @torch.no_grad()
    def encode_continuous(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (codes, z) where z is the continuous RVQ sum (B, T_frames, D)."""
        audio = audio.to(self.device)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        z, codes, _, _, _ = self.dac_model.encode(audio)
        z = z.transpose(1, 2)  # (B, T_frames, D_dac)
        return codes, z

    def get_codebook_embeddings(self, k: int) -> torch.Tensor:
        """(codebook_size, latent_dim) — used to init decoder input embeddings."""
        qk = self.dac_model.quantizer.quantizers[k]
        with torch.no_grad():
            cb = qk.codebook.weight                      # (codebook_size, codebook_dim)
            cb_for_conv = cb.T.unsqueeze(0)
            return qk.out_proj(cb_for_conv).squeeze(0).T  # (codebook_size, latent_dim)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode RVQ codes back to audio.

        Args:
            codes: (B, depth, T_frames)

        Returns:
            audio: (B, T_samples)
        """
        z = self.dac_model.quantizer.from_codes(codes)[0]
        audio = self.dac_model.decode(z)
        return audio.squeeze(1)
