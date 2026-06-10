"""SpeechBrain ECAPA speaker embedding wrapper for LiveVoice."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchaudio


class SpeechBrainECAPASpeakerEncoder(nn.Module):
    """Frozen SpeechBrain ECAPA-TDNN speaker encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.source_sr = int(getattr(config, "sample_rate", 24000))
        self.encoder_sr = int(getattr(config, "speechbrain_sample_rate", 16000))
        self.source = str(
            getattr(config, "speechbrain_source", "speechbrain/spkrec-ecapa-voxceleb")
        )
        self.savedir = str(
            getattr(config, "speechbrain_savedir", "")
            or os.path.join(
                getattr(config, "output_dir", "."),
                "pretrained_models",
                self.source.replace("/", "__"),
            )
        )
        device = str(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))

        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
            except ImportError:
                try:
                    from speechbrain.pretrained import EncoderClassifier  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "speaker_encoder_type='speechbrain_ecapa' requires SpeechBrain. "
                        "Install it in the sound env, e.g. pip install speechbrain."
                    ) from e

        self.classifier = EncoderClassifier.from_hparams(
            source=self.source,
            savedir=self.savedir,
            run_opts={"device": device},
        )
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.classifier.eval()
        return self

    def _prep(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.float()
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        if self.source_sr != self.encoder_sr:
            audio = torchaudio.functional.resample(audio, self.source_sr, self.encoder_sr)
        peak = audio.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        return audio / peak

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Return embeddings as (B, D)."""
        wavs = self._prep(audio)
        wav_lens = torch.ones(wavs.size(0), device=wavs.device)
        try:
            emb = self.classifier.encode_batch(wavs, wav_lens=wav_lens)
        except TypeError:
            emb = self.classifier.encode_batch(wavs)
        if emb.dim() == 3 and emb.size(1) == 1:
            emb = emb.squeeze(1)
        elif emb.dim() > 2:
            emb = emb.reshape(emb.size(0), -1)
        return emb.float()
