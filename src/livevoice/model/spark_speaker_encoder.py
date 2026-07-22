"""Spark-TTS BiCodec global-token speaker encoder for LiveVoice.

Voice-cloning speaker path only: we load ONLY BiCodec's global tokenizer
(ECAPA-TDNN → perceiver-resampler → residual-FSQ) and its mel transform, and
extract the fixed-length (token_num) sequence of FSQ-quantized "global tokens"
that Spark-TTS uses to represent speaker/timbre identity. These are utterance-
level (length-independent) and trained so BiCodec can *reconstruct* the speaker,
so they carry richer timbre than a classification ECAPA embedding.

We deliberately do NOT touch `sparktts.models.audio_tokenizer` (the semantic
path), which is the only place Spark imports wav2vec2/transformers — so this
encoder pulls in none of that. The BiCodec semantic encoder/decoder submodules
are dropped after load; only mel_transformer + speaker_encoder are kept, so the
frozen weights saved into LiveVoice checkpoints stay small.

Frozen + eval-only: only the downstream projection in the main model learns.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn


class SparkGlobalSpeakerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        repo = str(getattr(config, "spark_repo", "/workspace/Spark-TTS"))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from sparktts.models.bicodec import BiCodec  # noqa: E402 (needs sys.path)

        model_dir = str(getattr(
            config, "spark_bicodec_dir",
            "/workspace/Spark-TTS/pretrained_models/Spark-TTS-0.5B/BiCodec",
        ))
        bicodec = BiCodec.load_from_checkpoint(model_dir)

        # Keep ONLY the global-token path; drop the semantic encoder/decoder/quantizer
        # so their (unused) weights are neither registered nor saved in our ckpts.
        self.mel_transformer = bicodec.mel_transformer          # no learnable params
        self.speaker_encoder = bicodec.speaker_encoder          # SpeakerEncoder (frozen)
        del bicodec

        self.sample_rate = int(getattr(config, "spark_sample_rate", 16000))
        self.token_num = int(getattr(self.speaker_encoder, "perceiver_sampler").num_latents) \
            if hasattr(getattr(self.speaker_encoder, "perceiver_sampler", None), "num_latents") else 32
        # latent_dim of the FSQ codes (== per-token feature dim we hand upstream).
        self.out_dim = int(self.speaker_encoder.project.in_features // self.token_num)

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, ref_wav: torch.Tensor) -> torch.Tensor:
        """(B, T) or (B, 1, T) waveform @ sample_rate → (B, token_num, out_dim).

        Returns the FSQ-quantized global-token latent sequence (continuous),
        one fixed set of `token_num` tokens per utterance regardless of length.
        """
        if ref_wav.dim() == 3:
            ref_wav = ref_wav.squeeze(1)
        mel = self.mel_transformer(ref_wav)          # (B, n_mels, T_mel)
        if mel.dim() == 4:                           # (B, 1, n_mels, T_mel) guard
            mel = mel.squeeze(1)
        mels = mel.transpose(1, 2)                   # (B, T_mel, n_mels)

        se = self.speaker_encoder
        _, features = se.speaker_encoder(mels, True)             # ECAPA features
        x = se.perceiver_sampler(features.transpose(1, 2)).transpose(1, 2)  # (B, latent, token_num)
        zq, _ = se.quantizer(x)                                  # FSQ codes (B, latent, token_num)
        return zq.transpose(1, 2).contiguous()                   # (B, token_num, latent)

    def train(self, mode: bool = True):  # keep frozen encoder in eval (BN/dropout stable)
        super().train(False)
        return self
