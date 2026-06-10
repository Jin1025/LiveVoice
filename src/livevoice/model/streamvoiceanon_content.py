"""StreamVoiceAnon speech tokenizer wrapper for LiveVoice content conditioning.

Two feature paths, switched by `config.streamvoiceanon_use_continuous`:

  - continuous (default): pre-quantization z from the encoder's transformer.
    HuBERT-like — structurally coherent, so a random linear projection
    already carries phonetic info from step 0.

  - discrete: BSQ token ids → learnable nn.Embedding. Token ids are arbitrary
    integers; the embedding table has to be learned from scratch, which lets
    the rest of the model fall into a prev_codes+speaker shortcut before the
    content path becomes informative.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio


class StreamVoiceAnonContentEncoder(nn.Module):
    """Frozen StreamVoiceAnon causal content encoder + trainable head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repo = str(getattr(config, "streamvoiceanon_repo", "/workspace/StreamVoiceAnon"))
        self.encoder_config = str(
            getattr(
                config,
                "streamvoiceanon_encoder_config",
                os.path.join(
                    self.repo,
                    "configs/hydra_arcs/speech_tokenizers/causal-encoder-lfq-8192.yaml",
                ),
            )
        )
        self.encoder_ckpt = str(
            getattr(
                config,
                "streamvoiceanon_encoder_ckpt",
                os.path.join(self.repo, "ckpt/asr_s2s_bsq_8192_causal_down_whisper.pth"),
            )
        )
        self.source_sr = int(getattr(config, "sample_rate", 24000))
        self.encoder_sr = int(getattr(config, "streamvoiceanon_sample_rate", 44100))
        self.codebook_size = int(getattr(config, "streamvoiceanon_codebook_size", 8192))
        self.content_dim = int(getattr(config, "content_proj_dim", 256))
        self.use_continuous = bool(getattr(config, "streamvoiceanon_use_continuous", True))

        self.encoder = self._build_encoder()
        self.freeze_encoder = bool(getattr(config, "freeze_streamvoiceanon_encoder", True))
        if self.freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        if self.use_continuous:
            self.continuous_dim = self._infer_continuous_dim()
            self.continuous_proj = nn.Linear(self.continuous_dim, self.content_dim)
            print(
                f"[StreamVoiceAnonContentEncoder] continuous z mode "
                f"(D_z={self.continuous_dim} → content_dim={self.content_dim})"
            )
        else:
            self.code_embedding = nn.Embedding(self.codebook_size, self.content_dim)
            print(
                f"[StreamVoiceAnonContentEncoder] discrete code mode "
                f"(codebook_size={self.codebook_size} → content_dim={self.content_dim})"
            )
        self.out_norm = nn.LayerNorm(self.content_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _build_encoder(self) -> nn.Module:
        repo_path = Path(self.repo).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"StreamVoiceAnon repo not found: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            import hydra
            import yaml
            from omegaconf import DictConfig
        except Exception as e:
            raise ImportError(
                "StreamVoiceAnonContentEncoder requires hydra-core, omegaconf, and pyyaml. "
                "Install StreamVoiceAnon/LiveVoice deps in the active environment."
            ) from e

        cfg_path = Path(self.encoder_config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"StreamVoiceAnon encoder config not found: {cfg_path}")
        ckpt_path = Path(self.encoder_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"StreamVoiceAnon encoder checkpoint not found: {ckpt_path}")

        encoder_cfg = DictConfig(yaml.safe_load(cfg_path.read_text()))
        encoder = hydra.utils.instantiate(encoder_cfg)

        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = obj.get("net", obj) if isinstance(obj, dict) else obj
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unexpected StreamVoiceAnon checkpoint format: {ckpt_path}")
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        print(
            "[StreamVoiceAnonContentEncoder] loaded "
            f"{ckpt_path} missing={len(missing)} unexpected={len(unexpected)}"
        )
        return encoder

    def _infer_continuous_dim(self) -> int:
        """Dim of pre-quantization z = BSQ input dim."""
        q = getattr(self.encoder, "quantizer", None)
        if q is None:
            raise RuntimeError("Encoder has no `quantizer`; cannot infer continuous dim.")
        bsq = getattr(q, "residual_bsq", None)
        if bsq is not None and hasattr(bsq, "dim"):
            return int(bsq.dim)
        # Fall back to the last downsample stage's output channels.
        return 512

    def _prep(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        audio = audio.float()
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        if self.source_sr != self.encoder_sr:
            audio = torchaudio.functional.resample(audio, self.source_sr, self.encoder_sr)
        lengths = torch.full(
            (audio.size(0),),
            audio.size(-1),
            device=audio.device,
            dtype=torch.long,
        )
        return audio, lengths

    def _mel_and_mask(self, audio: torch.Tensor, lengths: torch.Tensor):
        spec = self.encoder.spec_transform
        mels = spec(audio)  # (B, n_mels, T_mel)
        mel_lengths = lengths // spec.hop_length
        T_mel = mels.shape[-1]
        ar = torch.arange(T_mel, device=audio.device, dtype=mel_lengths.dtype)
        mask = (ar.unsqueeze(0) < mel_lengths.unsqueeze(1)).to(mels.dtype).unsqueeze(1)
        return mels * mask, mask

    @torch.no_grad()
    def encode_codes(self, audio: torch.Tensor) -> torch.Tensor:
        """Return semantic token ids as (B, T_codes)."""
        audio, lengths = self._prep(audio)
        codes, _code_lengths = self.encoder.encode(audio, lengths)
        if codes.dim() == 3 and codes.size(0) == 1:
            codes = codes.squeeze(0)  # StreamVoiceAnon: (1, B, T) -> (B, T)
        elif codes.dim() == 3 and codes.size(1) == 1:
            codes = codes.squeeze(1)
        if codes.dim() != 2:
            raise RuntimeError(f"Unexpected StreamVoiceAnon content code shape: {tuple(codes.shape)}")
        return codes.long().clamp(min=0, max=self.codebook_size - 1)

    @torch.no_grad()
    def encode_continuous(self, audio: torch.Tensor) -> torch.Tensor:
        """Return pre-quantization continuous z as (B, T_codes, D_z).

        Mirrors firefly_encoder.encode + quantizer.encode up to (but not
        including) the BSQ step, so the output is the transformer's last
        hidden state — the input that would have been quantized.
        """
        audio, lengths = self._prep(audio)
        mels, mask = self._mel_and_mask(audio, lengths)
        feat = self.encoder.backbone(mels) * mask  # (B, D_backbone, T_mel)
        q = self.encoder.quantizer
        z = q.downsample(feat)                      # (B, D_q, T_codes)
        z = q.pre_module(z)                         # (B, D_q, T_codes)
        # Normalize to (B, T, D) regardless of which layout pre_module returns.
        if z.dim() != 3:
            raise RuntimeError(f"Unexpected pre_module output shape: {tuple(z.shape)}")
        if z.shape[1] == self.continuous_dim and z.shape[-1] != self.continuous_dim:
            z = z.transpose(1, 2)
        elif z.shape[-1] != self.continuous_dim:
            raise RuntimeError(
                f"pre_module output last dim {z.shape[-1]} != continuous_dim {self.continuous_dim}; "
                f"shape={tuple(z.shape)}"
            )
        return z.float()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Return content features as (B, T_codes, content_proj_dim)."""
        if self.use_continuous:
            z = self.encode_continuous(audio)
            return self.out_norm(self.continuous_proj(z))
        codes = self.encode_codes(audio)
        return self.out_norm(self.code_embedding(codes))
