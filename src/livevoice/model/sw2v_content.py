"""SW2V (jhcodec streaming-wav2vec) content encoder for LiveVoice.

jhcodec ships a causal "semantic" audio encoder (``jhcodec.model.sw2v.AudioEncoder``)
that maps 16 kHz audio → continuous 1024-d features at 50 fps (non-overlapping 320-sample
blocks — the SAME grid as the jhcodec codec tokens, so no receptive-field offset like
HuBERT). It is the distillation target the codec's semantic RVQ is trained to match, i.e.
a content representation. We reuse it here as a drop-in ``content_source="sw2v"``.

Interface mirrors the other external content encoders (StreamVoiceAnon / mimi_semantic):
frozen, eval-only, ``forward(audio) -> (B, T_frames, out_dim)`` with T_frames = ceil(L/320).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOP = 320  # samples per frame at 16 kHz → 50 fps (== jhcodec codec hop)


class Sw2vContentEncoder(nn.Module):
    """Frozen jhcodec AudioEncoder as a LiveVoice content source (16 kHz, 50 fps, 1024-d)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.repo = str(getattr(config, "sw2v_repo", "/workspace/jhcodec"))
        self.cfg_path = str(getattr(config, "sw2v_config",
                                    f"{self.repo}/config/config_w2vcossim.json"))
        self.ckpt_path = str(getattr(config, "sw2v_ckpt",
                                     "/mnt/data/disk3/yejin/jhcodec/sw2v_120000.pt"))
        self.encoder_sr = int(getattr(config, "sw2v_sample_rate", 16000))
        self.input_sr = int(getattr(config, "sample_rate", 16000))

        self.encoder, self.out_dim, self.in_features = self._build()
        self.hop_length = HOP

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _build(self):
        repo = Path(self.repo).resolve()
        if not repo.exists():
            raise FileNotFoundError(f"jhcodec repo not found: {repo}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        cfg_path = Path(self.cfg_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"SW2V config not found: {cfg_path}")
        ckpt_path = Path(self.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SW2V checkpoint not found: {ckpt_path}")

        import omegaconf
        from jhcodec.model.sw2v import AudioEncoder
        from jhcodec import utils as jhcodec_utils

        cfg = omegaconf.OmegaConf.load(str(cfg_path))
        w2v_cfg = cfg.w2v
        enc = AudioEncoder(w2v_cfg, compute_dtype=torch.float32, training=False)
        # jhcodec loader reads checkpoint['model_state_dict'] (strict).
        jhcodec_utils.load_checkpoint(enc, None, None, str(ckpt_path), strict_model=True)
        out_dim = int(w2v_cfg.rvq.embedding_dim)          # 1024
        in_features = int(w2v_cfg.mlp_in.in_features)      # 320 (== HOP)
        print(f"[Sw2vContentEncoder] loaded {ckpt_path.name}  out_dim={out_dim}  "
              f"hop={in_features}  ({self.encoder_sr} Hz, 50 fps)")
        return enc, out_dim, in_features

    def train(self, mode: bool = True):
        # Keep the frozen encoder in eval regardless of the parent's train/eval switch —
        # AudioEncoder.forward applies noise_masking augmentation when self.training.
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (B, T) at input_sr → (B, ceil(T/HOP), out_dim).

        End-pads to a multiple of HOP so AudioEncoder's view(B, -1, HOP) is valid and the
        frame count equals ceil(L/HOP) — 1:1 aligned with the jhcodec codec token grid.
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.float()
        if self.input_sr != self.encoder_sr:
            import torchaudio
            audio = torchaudio.functional.resample(audio, self.input_sr, self.encoder_sr)
        B, T = audio.shape
        n = max(1, -(-T // self.in_features))          # ceil(T/HOP)
        pad = n * self.in_features - T
        if pad > 0:
            audio = F.pad(audio, (0, pad))             # end-pad (matches codec tokenization)
        feats = self.encoder(audio.to(next(self.encoder.parameters()).device))  # (B, n, out_dim)
        return feats
