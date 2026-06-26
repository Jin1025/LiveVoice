"""JHCodec wrapper for LiveVoice (mimi-variant: 16 kHz, 50 fps streaming RVQ).

JHCodec ("Reconstruct! Don't Encode") is a pure-transformer streaming neural
audio codec. We use the released `JHCodecMimi` checkpoint:

  sample_rate   : 16 000 Hz
  hop_length    : 320 samples            → 50 fps (low latency)
  n_codebooks   : 8   (1 semantic + 7 RVQ)
  codebook_size : 1 024
  embedding_dim : 1 024  (continuous pre-quantization z dim)

Interface mirrors MimiCodec / DACModel so the rest of LiveVoice stays
codec-agnostic:

  encode(audio)              -> codes (B, K, T)
  encode_continuous(audio)   -> (codes (B, K, T), z (B, T, latent_dim))
  decode(codes (B, K, T))    -> audio (B, T_samples) at 16 kHz
  get_codebook_embeddings(k) -> (codebook_size, latent_dim) | None
  attrs: sample_rate, hop_length, n_codebooks, codebook_size, latent_dim

Note: the codec is at 16 kHz / 50 fps. Set config.sample_rate = 16000 when
using this codec so windowing, frame counts and audio logging stay consistent.

Requires the jhcodec package and its deps (flash-attn, omegaconf, and a torch
new enough to provide torch.nn.RMSNorm). See /workspace/jhcodec/installcu128.sh.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class JHCodecModel(nn.Module):
    """Frozen JHCodec (mimi variant): audio ↔ discrete codes ↔ continuous z."""

    SAMPLE_RATE = 16_000
    HOP_LENGTH = 320  # 16000 / 320 = 50 fps

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = str(getattr(config, "device", "cuda"))
        # Sample rate of audio arriving from the dataset (may differ from 16 kHz).
        self.input_sr = int(getattr(config, "sample_rate", self.SAMPLE_RATE))

        self.repo = str(getattr(config, "jhcodec_repo", "/workspace/jhcodec"))
        self.cfg_path = str(
            getattr(config, "jhcodec_config", f"{self.repo}/config/config_mimi_recon.json")
        )
        self.ckpt_path = str(
            getattr(config, "jhcodec_ckpt", f"{self.repo}/ckpt/jhcodec_mimi_1000000.pt")
        )

        self.model, mc = self._build()

        self.encoder_sr = int(mc["data_sample_rate"])
        self.codebook_size = int(mc["codebook_size"])
        self.embedding_dim = int(mc["embedding_dim"])
        self.n_codebooks = int(getattr(config, "jhcodec_n_codebooks", mc["num_codebooks"]))
        # Continuous z (pre-quantization) dim exposed to the rest of LiveVoice.
        self.latent_dim = self.embedding_dim
        self.sample_rate = self.encoder_sr
        self.hop_length = int(getattr(config, "jhcodec_hop_length", self.HOP_LENGTH))

        self.model.to(self.device)
        self.model.eval()  # triggers RVQMimi.register_up_vq() — must run AFTER ckpt load
        for p in self.model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def _build(self):
        repo = Path(self.repo).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"JHCodec repo not found: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            import omegaconf
            from jhcodec.model.codec import JHCodecMimi
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "JHCodecModel requires the jhcodec package and its deps "
                "(flash-attn, omegaconf, torch>=2.4 with torch.nn.RMSNorm). "
                "Install per /workspace/jhcodec/installcu128.sh in the active env."
            ) from e

        cfg_path = Path(self.cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"JHCodec config not found: {cfg_path}")
        ckpt_path = Path(self.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"JHCodec checkpoint not found: {ckpt_path}")

        cfg = omegaconf.OmegaConf.load(str(cfg_path))
        rvq_type = str(cfg.model.rvq.type)
        if rvq_type != "mimi":
            raise ValueError(
                f"JHCodecModel expects a mimi RVQ config; got rvq.type={rvq_type!r}"
            )

        model = JHCodecMimi(cfg.model)  # training=False by default

        obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
        if not isinstance(sd, dict):
            raise RuntimeError(f"Unexpected JHCodec checkpoint format: {ckpt_path}")
        try:
            model.load_state_dict(sd, strict=True)
            print(f"[JHCodecModel] loaded {ckpt_path} (strict=True)")
        except Exception as e:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(
                f"[JHCodecModel] strict load failed ({e}); fell back to strict=False "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        data_sr = int(cfg.data.sample_rate) if "data" in cfg and "sample_rate" in cfg.data else self.SAMPLE_RATE
        mc = {
            "data_sample_rate": data_sr,
            "num_codebooks": int(cfg.model.rvq.num_codebooks),
            "codebook_size": int(cfg.model.rvq.codebook_size),
            "embedding_dim": int(cfg.model.rvq.embedding_dim),
        }
        return model, mc

    def train(self, mode: bool = True):
        # Keep the frozen codec in eval regardless of the parent's train/eval
        # switches (the JHCodec Decoder applies dropout/drop-path in train mode).
        super().train(mode)
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    def _prep(self, audio: torch.Tensor) -> torch.Tensor:
        """Ensure (B, T) float32 on device at JHCodec's native 16 kHz."""
        audio = audio.to(self.device).float()
        if audio.dim() == 3:        # (B, 1, T)
            audio = audio.squeeze(1)
        elif audio.dim() == 1:      # (T,)
            audio = audio.unsqueeze(0)
        if self.input_sr != self.encoder_sr:
            audio = torchaudio.functional.resample(audio, self.input_sr, self.encoder_sr)
        return audio

    def _n_codebooks_arg(self, k: int, device) -> torch.Tensor:
        return torch.tensor([int(k)], dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """(B, T) → codes (B, n_codebooks, T_frames)."""
        audio = self._prep(audio)
        indices, _ = self.model.encode(
            audio, self._n_codebooks_arg(self.n_codebooks, audio.device), inference_cache=None
        )  # (B, T, K)
        return indices.permute(0, 2, 1).contiguous().long()  # (B, K, T)

    @torch.no_grad()
    def _encode_z(self, audio: torch.Tensor) -> torch.Tensor:
        """Pre-quantization continuous z (B, T_frames, embedding_dim).

        Mirrors JHCodecMimi.encode up to (but excluding) the RVQ step.
        """
        m = self.model
        fs = int(m.feature_size)
        B, T_in = audio.shape
        if T_in % fs != 0:
            audio = F.pad(audio, (0, fs - (T_in % fs)))
        x = audio.view(audio.shape[0], -1, fs)
        x = m.linear_in(x)
        x_encoded, _ = m.encoder.decode(x, inference_cache=None)
        z = m.rvq_in(x_encoded)  # (B, T_frames, embedding_dim)
        return z.float()

    @torch.no_grad()
    def encode_continuous(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T) → (codes (B, K, T), z (B, T_frames, latent_dim))."""
        audio = self._prep(audio)
        indices, _ = self.model.encode(
            audio, self._n_codebooks_arg(self.n_codebooks, audio.device), inference_cache=None
        )
        codes = indices.permute(0, 2, 1).contiguous().long()  # (B, K, T)
        z = self._encode_z(audio)  # (B, T, embedding_dim)
        return codes, z

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, K, T_frames) → audio (B, T_samples) at 16 kHz."""
        codes = codes.to(self.device).long()
        K = int(codes.shape[1])
        indices = codes.permute(0, 2, 1).contiguous()  # (B, T, K)
        audio, _ = self.model.decode(
            indices, self._n_codebooks_arg(K, codes.device), inference_cache=None
        )
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return audio  # (B, T_samples)

    @torch.no_grad()
    def get_codebook_embeddings(self, k: int) -> torch.Tensor | None:
        """Return (codebook_size, embedding_dim) up-projected codebook for codebook k.

        codebook 0 is the semantic codebook; codebooks 1..K-1 are the residual
        RVQ codebooks. Each raw codebook lives in latent space; we up-project it
        to embedding_dim (the space the codec's decoder consumes) so it forms a
        meaningful init for LiveVoice's per-codebook input embeddings.
        """
        rvq = getattr(self.model, "rvq", None)
        if rvq is None:
            return None
        try:
            if k == 0:
                w = rvq.semantic_vq.codebook.weight
                emb = rvq.semantic_up(w)
            else:
                w = rvq.vqs[k - 1].codebook.weight
                emb = rvq.ups[k - 1](w)
        except (AttributeError, IndexError):
            return None
        emb = emb.detach().float()
        if emb.dim() == 2 and emb.shape[0] == self.codebook_size:
            return emb
        return None
