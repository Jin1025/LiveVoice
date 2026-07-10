"""WavLM-TDNN speaker embedding wrapper for LiveVoice validation spk_sim.

Wraps the project's existing UniSpeech loader (WavLM-large backbone + finetuned
ECAPA-TDNN head, ``livevoice.evaluation.unispeech_sv``) — the *same* encoder
Vevo / Amphion report SIM with ("WavLM TDNN"). This puts ``val/spk_sim`` and
``val/spk_sim_gt`` on those papers' absolute scale (GT SIM ~0.75) instead of
SpeechBrain ECAPA's compressed scale (GT SIM ~0.6).

Requires:
  * ``s3prl`` in the env (builds the WavLM-large upstream), and
  * the UniSpeech finetuned checkpoint at ``config.wavlm_sv_ckpt``.
If either is missing, the caller (module._get_spk_encoder) catches the error and
skips spk_sim rather than crashing training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class WavLMTDNNSpeakerEncoder(nn.Module):
    """Frozen UniSpeech WavLM-large + ECAPA-TDNN speaker encoder (eval-only)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.source_sr = int(getattr(config, "sample_rate", 24000))
        self.encoder_sr = int(getattr(config, "wavlm_sv_sample_rate", 16000))
        self.ckpt = str(getattr(config, "wavlm_sv_ckpt", ""))
        self.variant = str(getattr(config, "wavlm_sv_variant", "wavlm_large"))
        device = str(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        if device == "cuda":
            device = "cuda:0"

        from livevoice.evaluation.unispeech_sv import UniSpeechWavLMTDNNEmbedder

        # Builds ECAPA_TDNN on the WavLM upstream and loads the finetuned .pth.
        self.embedder = UniSpeechWavLMTDNNEmbedder(
            checkpoint=self.ckpt, device=device, variant=self.variant
        )

    def train(self, mode: bool = True):
        # Underlying model is always eval/frozen; nothing to toggle.
        super().train(mode)
        return self

    def _to_16k(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.float()
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if self.source_sr != self.encoder_sr:
            audio = torchaudio.functional.resample(audio, self.source_sr, self.encoder_sr)
        return audio

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio (B, T) at source_sr → embeddings (B, D). Per-item peak-normalized
        (matches eval_s-sim._waveform_for_embedder)."""
        wavs = self._to_16k(audio)
        embs = []
        for i in range(wavs.size(0)):
            w = wavs[i]
            peak = w.abs().max().clamp(min=1e-8)
            embs.append(self.embedder.embed(w / peak))  # (D,) on cpu
        out = torch.stack(embs, dim=0).float()
        return out.to(audio.device)
