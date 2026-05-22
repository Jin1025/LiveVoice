"""WavLM upstream + ECAPA-TDNN head (Microsoft UniSpeech speaker verification)."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from livevoice.evaluation.unispeech_sv.ecapa_tdnn import ECAPA_TDNN_SMALL

WAVLM_VARIANTS = {
    "wavlm_large": dict(feat_dim=1024, feat_type="wavlm_large"),
    "wavlm_base_plus": dict(feat_dim=768, feat_type="wavlm_base_plus"),
}


def load_wavlm_tdnn(
    checkpoint: str | None,
    variant: str = "wavlm_large",
    device: str = "cuda",
) -> torch.nn.Module:
    """Build ECAPA_TDNN on frozen WavLM; optionally load UniSpeech finetuned weights."""
    if variant not in WAVLM_VARIANTS:
        raise ValueError(f"variant must be one of {list(WAVLM_VARIANTS)}")
    cfg = WAVLM_VARIANTS[variant]
    model = ECAPA_TDNN_SMALL(
        feat_dim=cfg["feat_dim"],
        feat_type=cfg["feat_type"],
        config_path=None,
    )
    if checkpoint:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"WavLM-TDNN checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[wavlm-tdnn] warn: {len(missing)} missing keys (first 3): {missing[:3]}")
        if unexpected:
            print(f"[wavlm-tdnn] warn: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()
    return model


class UniSpeechWavLMTDNNEmbedder:
    """16 kHz mono waveform -> 256-d speaker embedding (cosine in [-1, 1])."""

    sample_rate = 16000

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        variant: str = "wavlm_large",
    ):
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.model = load_wavlm_tdnn(checkpoint, variant=variant, device=self.device)

    @torch.no_grad()
    def embed(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: mono float (T,) at any rate; resampled internally to 16 kHz."""
        w = waveform.detach().float().reshape(-1)
        if w.numel() == 0:
            raise ValueError("empty waveform")
        wav = w.unsqueeze(0).to(self.model.device if hasattr(self.model, "device") else next(self.model.parameters()).device)
        emb = self.model([wav.squeeze(0)])
        return emb.squeeze(0).float().cpu()

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())
