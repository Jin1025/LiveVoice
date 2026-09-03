"""NVIDIA FastConformer encoder as a content (BNF) extractor — standalone, no NeMo dependency.

Architecture: stt_en_fastconformer_hybrid_large_streaming_multi
  - 80-dim mel fbank at 100 Hz
  - DW-striding causal subsampling: 8x → 12.5 fps
  - 17 Conformer layers: d_model=512, ff=2048, 8 heads, causal conv kernel 9
  - RelPositionalEncoding (Transformer-XL style)

Frame rate: 12.5 fps native.
  - codec="mimi" (12.5 fps): direct match, no resampling
  - codec="jhcodec" (50 fps): repeat_interleave(4, dim=1) to upsample

Weights are loaded from a .nemo archive (tar containing model_weights.ckpt).
"""
from __future__ import annotations

import io
import math
import tarfile

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal Conv2D (from NeMo, Apache-2.0)
# ---------------------------------------------------------------------------
class _CausalConv2D(nn.Conv2d):
    """Conv2d with causal padding: left=kernel-1, right=stride-1 on BOTH dims."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        self._left_pad = kernel_size - 1
        self._right_pad = stride - 1
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=0, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._left_pad, self._right_pad,
                       self._left_pad, self._right_pad))
        return super().forward(x)


# ---------------------------------------------------------------------------
# Causal Conv1D
# ---------------------------------------------------------------------------
class _CausalConv1D(nn.Conv1d):
    """Conv1d with left-only padding for causal operation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, bias=True):
        if padding is None:
            self._left_pad = kernel_size - 1
            self._right_pad = stride - 1
        elif isinstance(padding, (list, tuple)):
            self._left_pad, self._right_pad = padding
        else:
            self._left_pad = padding
            self._right_pad = padding
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=0, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._left_pad, self._right_pad))
        return super().forward(x)


# ---------------------------------------------------------------------------
# Causal upsampler: repeat + causal conv to differentiate positions
# ---------------------------------------------------------------------------
class _CausalUpsample(nn.Module):
    """Upsample by `factor` via ConvTranspose1d + causal conv smoothing."""

    def __init__(self, d_model: int, factor: int = 4, kernel_size: int = 9):
        super().__init__()
        self.factor = factor
        self.up = nn.ConvTranspose1d(d_model, d_model, kernel_size=factor, stride=factor)
        self.smooth = nn.Sequential(
            _CausalConv1D(d_model, d_model, kernel_size),
            nn.SiLU(),
            _CausalConv1D(d_model, d_model, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                          # (B, D, T)
        x = self.up(x)                                 # (B, D, factor*T)
        x = self.smooth(x)
        return x.transpose(1, 2)                        # (B, factor*T, D)


# ---------------------------------------------------------------------------
# Subsampling: DW-striding 8x
# ---------------------------------------------------------------------------
class _DWStridingSubsampling(nn.Module):
    """CausalConv2D-based subsampling: 80-dim fbank → (T/8, 512)."""

    def __init__(self, feat_in: int = 80, feat_out: int = 512, conv_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            _CausalConv2D(1, conv_channels, 3, stride=2),                        # 0
            nn.ReLU(inplace=True),                                                # 1
            _CausalConv2D(conv_channels, conv_channels, 3, stride=2,
                          groups=conv_channels),                                  # 2
            nn.Conv2d(conv_channels, conv_channels, 1),                           # 3
            nn.ReLU(inplace=True),                                                # 4
            _CausalConv2D(conv_channels, conv_channels, 3, stride=2,
                          groups=conv_channels),                                  # 5
            nn.Conv2d(conv_channels, conv_channels, 1),                           # 6
            nn.ReLU(inplace=True),                                                # 7
        )
        freq_out = feat_in
        for _ in range(3):
            freq_out = freq_out // 2 + 1  # causal pad adds kernel-1+stride-1, then stride-2 conv
        self.out = nn.Linear(conv_channels * freq_out, feat_out)

    @staticmethod
    def _calc_length(length: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            length = length // 2 + 1
        return length

    @staticmethod
    def _time_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Zero out frames past ``lengths`` — NeMo's MaskedConvSequential behaviour.

        Padded frames must not leak into the causal convs, otherwise every frame
        near the tail differs from NeMo's output.
        """
        t = x.size(2)
        keep = (torch.arange(t, device=x.device).unsqueeze(0)
                < lengths.to(x.device).unsqueeze(1))            # (B, T)
        return x * keep[:, None, :, None].to(x.dtype)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)              # (B, 1, T, F)
        cur = lengths.clone()
        for layer in self.conv:
            x = self._time_mask(x, cur)
            x = layer(x)
            if isinstance(layer, _CausalConv2D) and layer.stride != (1, 1):
                cur = ((cur + layer._left_pad + layer._right_pad
                        - layer.kernel_size[0]) // layer.stride[0] + 1)
        x = self._time_mask(x, cur)     # (B, C, T', F')
        b, c, t, f = x.shape
        x = x.transpose(1, 2).reshape(b, t, c * f)
        x = self.out(x)                 # (B, T', feat_out)
        return x, cur


# ---------------------------------------------------------------------------
# Relative Positional Encoding (Transformer-XL)
# ---------------------------------------------------------------------------
class _RelPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, xscale: float | None = None,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.dropout = nn.Dropout(dropout)
        self.pe: torch.Tensor
        self._extend_pe(max_len, torch.float32, torch.device("cpu"))

    def _extend_pe(self, length: int, dtype: torch.dtype, device: torch.device):
        needed = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed:
            return
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32,
                                 device=device).unsqueeze(1)
        pe = torch.zeros(positions.size(0), self.d_model, device=device)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        pe = pe.unsqueeze(0).to(dtype)
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        length = x.size(1)
        self._extend_pe(length, x.dtype, x.device)
        if self.xscale:
            x = x * self.xscale
        center = self.pe.size(1) // 2 + 1
        pos_emb = self.pe[:, center - length: center + length - 1]
        return self.dropout(x), pos_emb


# ---------------------------------------------------------------------------
# Relative-Position Multi-Head Attention
# ---------------------------------------------------------------------------
class _RelPosMHA(nn.Module):

    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.h = n_head
        self.d_k = d_model // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_head, self.d_k))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.shape
        x = F.pad(x, (1, 0))
        x = x.view(b, h, -1, qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                att_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.h, self.d_k)
        k = self.linear_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)

        q_u = (q + self.pos_bias_u).transpose(1, 2)  # (B, h, T, d_k)
        q_v = (q + self.pos_bias_v).transpose(1, 2)

        ac = torch.matmul(q_u, k.transpose(-2, -1))
        bd = torch.matmul(q_v, p.transpose(-2, -1))
        bd = self._rel_shift(bd)
        bd = bd[:, :, :, :ac.size(-1)]

        scores = (ac + bd) / self.s_d_k

        if att_mask is not None:
            scores = scores.masked_fill(att_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        if att_mask is not None:
            attn = attn.masked_fill(att_mask, 0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                         # (B, h, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, -1)         # (B, T, d_model)
        return self.linear_out(out)


# ---------------------------------------------------------------------------
# Conformer Convolution Module
# ---------------------------------------------------------------------------
class _ConformerConv(nn.Module):

    def __init__(self, d_model: int, kernel_size: int = 9):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, 1, bias=True)
        # causal: left_pad = kernel-1, right_pad = 0
        self.depthwise_conv = _CausalConv1D(
            d_model, d_model, kernel_size, groups=d_model,
            padding=[kernel_size - 1, 0], bias=True,
        )
        self.batch_norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)                               # (B, D, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)                                 # (B, D, T)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)                               # (B, T, D)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = x.transpose(1, 2)                               # (B, D, T)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)                             # (B, T, D)


# ---------------------------------------------------------------------------
# Conformer Feed-Forward
# ---------------------------------------------------------------------------
class _ConformerFFN(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.silu(self.linear1(x))))


# ---------------------------------------------------------------------------
# Conformer Layer
# ---------------------------------------------------------------------------
class _ConformerLayer(nn.Module):

    def __init__(self, d_model: int = 512, d_ff: int = 2048,
                 n_heads: int = 8, conv_kernel_size: int = 9,
                 dropout: float = 0.0, dropout_att: float = 0.0):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = _ConformerFFN(d_model, d_ff, dropout)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = _RelPosMHA(n_heads, d_model, dropout_att)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = _ConformerConv(d_model, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = _ConformerFFN(d_model, d_ff, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor,
                att_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * 0.5

        x = self.norm_self_att(residual)
        x = self.self_attn(x, pos_emb, att_mask)
        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x)
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * 0.5

        return self.norm_out(residual)


# ---------------------------------------------------------------------------
# Full Encoder
# ---------------------------------------------------------------------------
class _FastConformerEncoder(nn.Module):

    def __init__(self, n_layers: int = 17, d_model: int = 512, d_ff: int = 2048,
                 n_heads: int = 8, conv_kernel_size: int = 9,
                 conv_channels: int = 256, feat_in: int = 80,
                 att_context_size: tuple[int, int] = (70, 0)):
        super().__init__()
        self.pre_encode = _DWStridingSubsampling(feat_in, d_model, conv_channels)
        self.pos_enc = _RelPositionalEncoding(d_model, xscale=math.sqrt(d_model))
        self.layers = nn.ModuleList([
            _ConformerLayer(d_model, d_ff, n_heads, conv_kernel_size)
            for _ in range(n_layers)
        ])
        self.att_context_size = att_context_size

    @staticmethod
    def _make_chunked_limited_mask(
        T: int, left_context: int, right_context: int, device: torch.device,
    ) -> torch.Tensor:
        """NeMo-style chunked_limited attention mask.

        Returns bool mask of shape (1, 1, T, T) where True = MASKED (blocked).

        With right_context >= 0, chunk_size = right_context + 1.
        Each position can attend to its own chunk and ``left_context // chunk_size``
        chunks to the left.  right_context=0 → strictly causal within chunks of 1.
        """
        if left_context < 0 and right_context < 0:
            return None
        # att_mask: True where attention IS allowed (flipped at the end)
        att = torch.ones(T, T, dtype=torch.bool, device=device)
        if right_context < 0:
            # unlimited right, just limit left
            if left_context >= 0:
                att = att.triu(diagonal=-left_context)
        else:
            chunk_size = right_context + 1
            left_chunks = left_context // chunk_size if left_context >= 0 else 100000
            idx = torch.arange(T, device=device)
            chunk_idx = torch.div(idx, chunk_size, rounding_mode="trunc")
            # diff[i,j] = query_chunk[i] - key_chunk[j]; >=0 means key is in past
            diff = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)  # (T, T)
            chunk_ok = (diff >= 0) & (diff <= left_chunks)
            att = att & chunk_ok
        return (~att).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T), True=masked

    def forward(self, fbank: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            fbank: (B, T_fbank, 80)
            lengths: (B,)
        Returns:
            (B, T_out, d_model), lengths_out
        """
        x, lengths = self.pre_encode(fbank, lengths)
        x, pos_emb = self.pos_enc(x)
        T = x.size(1)

        # Padding mask: True = padded position
        pad_mask = (torch.arange(T, device=x.device).unsqueeze(0)
                    >= lengths.unsqueeze(1))

        # Chunked-limited causal mask
        left_ctx, right_ctx = self.att_context_size
        causal_mask = self._make_chunked_limited_mask(T, left_ctx, right_ctx, x.device)

        if causal_mask is not None:
            # Combine: pad mask → (B, 1, 1, T), causal → (1, 1, T, T)
            pad_2d = pad_mask.unsqueeze(1).unsqueeze(1)        # block attending TO padded
            pad_2d_src = pad_mask.unsqueeze(1).unsqueeze(2)    # block attending FROM padded
            att_mask = causal_mask | pad_2d | pad_2d_src
        else:
            att_mask = pad_mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            x = layer(x, pos_emb, att_mask)
        return x, lengths


# ---------------------------------------------------------------------------
# Public: FastConformerContentEncoder
# ---------------------------------------------------------------------------
class FastConformerContentEncoder(nn.Module):
    """(B, T_audio) waveform → (B, T_frames, D) content features at 12.5 fps.

    For jhcodec (50 fps), frames are repeated 4× to match the codec grid.
    For mimi (12.5 fps), output is used directly.

    ``layer`` selects the feature tap point:
        -1  (default)  output of the last conformer layer
        0..16          output of that specific layer
    """

    def __init__(self, config, ckpt_path: str | None = None, layer: int = -1):
        super().__init__()
        ckpt = str(ckpt_path or getattr(config, "fastconformer_ckpt", ""))

        # Load state dict from .nemo tar archive
        sd = self._load_nemo(ckpt)

        # Derive architecture from the state dict
        n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers."))
        d_model = sd["encoder.layers.0.self_attn.linear_q.weight"].shape[0]
        d_ff = sd["encoder.layers.0.feed_forward1.linear1.weight"].shape[0]
        n_heads = sd["encoder.layers.0.self_attn.pos_bias_u"].shape[0]
        conv_kernel_size = sd["encoder.layers.0.conv.depthwise_conv.weight"].shape[2]
        conv_channels = sd["encoder.pre_encode.conv.0.weight"].shape[0]

        # att_context_size: (left, right) in subsampled frames.
        # Default [70, 0] = causal (no right context). The pretrained model was trained
        # with multi-mode [[70,13],[70,6],[70,1],[70,0]]; at inference NeMo uses mode 0
        # = [70,13]. We default to [70,0] for streaming causality — set via config to
        # trade latency for quality.
        att_ctx = tuple(getattr(config, "fastconformer_att_context",
                                (70, 0)))
        self.encoder = _FastConformerEncoder(
            n_layers=n_layers, d_model=d_model, d_ff=d_ff,
            n_heads=n_heads, conv_kernel_size=conv_kernel_size,
            conv_channels=conv_channels,
            att_context_size=att_ctx,
        )

        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        m, u = self.encoder.load_state_dict(enc_sd, strict=False)
        if m:
            raise RuntimeError(f"FastConformer missing keys: {m}")
        if u:
            print(f"[fastconformer] ignoring {len(u)} unexpected keys")

        # NeMo preprocessor: stored mel filterbank + window for exact feature reproduction.
        self.register_buffer("_mel_fb",
                             sd["preprocessor.featurizer.fb"].squeeze(0))  # (80, 257)
        self.register_buffer("_stft_window",
                             sd["preprocessor.featurizer.window"])         # (400,)
        self._n_fft = (self._mel_fb.shape[1] - 1) * 2                     # 512
        self._hop_length = 160   # 0.01s × 16 kHz
        self._win_length = self._stft_window.shape[0]                      # 400

        self.layer = layer
        self._tap: torch.Tensor | None = None
        if layer >= 0:
            self.encoder.layers[layer].register_forward_hook(
                lambda _m, _i, out: self._capture(out))

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.out_norm = nn.LayerNorm(d_model)

        codec = str(getattr(config, "codec", "jhcodec")).lower()
        self._repeat = 4 if codec == "jhcodec" else 1
        self._upsample = _CausalUpsample(d_model, self._repeat) if self._repeat > 1 else None
        self._input_sr = int(getattr(config, "sample_rate", 16000))

        self.out_dim = d_model
        print(f"[fastconformer] {n_layers} layers  d_model={d_model}  heads={n_heads}  "
              f"att_context={att_ctx}  tap={layer}  out_dim={d_model}  "
              f"codec={codec}  repeat={self._repeat}  input_sr={self._input_sr}  "
              f"@ {12.5 * self._repeat} fps effective")

    @staticmethod
    def _load_nemo(path: str) -> dict:
        with tarfile.open(path, "r") as t:
            wf = t.extractfile("./model_weights.ckpt")
            assert wf is not None
            return torch.load(io.BytesIO(wf.read()), map_location="cpu", weights_only=False)

    def _capture(self, x: torch.Tensor):
        self._tap = x
        return None

    def _fbank(self, wav: torch.Tensor, sr: int):
        """NeMo-compatible log-mel fbank. Bit-exact against NeMo's preprocessor.

        Returns ``(feats, valid_lengths)`` where ``feats`` is (B, T, 80).  center=True
        STFT emits one frame more than NeMo counts as valid; NeMo zeroes that tail
        frame and reports the shorter length, so we do the same.
        """
        guard = 2 ** -24  # NeMo default log_zero_guard_value
        n_samples = wav.shape[-1]
        wav = torch.cat((wav[:, 0:1], wav[:, 1:] - 0.97 * wav[:, :-1]), dim=1)
        spec = torch.stft(
            wav, self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=self._stft_window.to(wav.device),
            center=True, pad_mode="constant",
            return_complex=True,
        )
        # NeMo takes sqrt of the real/imag sum then squares — keep the same op order.
        power = torch.view_as_real(spec).pow(2).sum(-1).sqrt().pow(2.0)
        mel = torch.matmul(
            self._mel_fb.to(power.device, power.dtype),      # (80, 257)
            power,                                            # (B, 257, T)
        )                                                     # (B, 80, T)
        feats = torch.log(mel + guard)

        n_valid = (n_samples + self._n_fft // 2 * 2 - self._n_fft) // self._hop_length
        lengths = torch.full((feats.size(0),), int(n_valid), dtype=torch.int64,
                             device=feats.device)
        pad = (torch.arange(feats.size(-1), device=feats.device).unsqueeze(0)
               >= lengths.unsqueeze(1))                       # (B, T)
        feats = feats.masked_fill(pad.unsqueeze(1), 0.0)
        return feats.transpose(1, 2), lengths                 # (B, T, 80), (B,)

    @staticmethod
    def _subsample_len(t: int) -> int:
        """Predict # frames after 3× stride-2 causal subsampling."""
        for _ in range(3):
            t = t // 2 + 1
        return t

    def forward(self, audio: torch.Tensor, sample_rate: int | None = None,
                align_to_codec: bool = True) -> torch.Tensor:
        if sample_rate is None:
            sample_rate = self._input_sr
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Codec grid target: the exact frame count the codec would emit.
        # jhcodec: 16 kHz / 320 hop = 50 fps → ceil(samples / 320)
        # mimi:    24 kHz / 1920 hop = 12.5 fps → ceil(samples / 1920)
        if self._repeat == 4:   # jhcodec
            codec_hop = sample_rate // 50
        else:                   # mimi
            codec_hop = sample_rate * 80 // 1000  # 1/12.5 = 80ms
        n_codec = -(-int(audio.shape[-1]) // codec_hop)   # ceil division

        # FastConformer was pretrained on 16 kHz fbank — resample if needed.
        if sample_rate != 16000:
            import torchaudio
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
            sample_rate = 16000

        # How many FC-native (12.5 fps) frames we need before repeat.
        if self._repeat > 1:
            n_fc_target = -(-n_codec // self._repeat)     # ceil(n_codec / 4)
        else:
            n_fc_target = n_codec

        with torch.no_grad():
            dev = self.out_norm.weight.device
            fbank, lengths = self._fbank(audio, sample_rate)
            fbank, lengths = fbank.to(dev), lengths.to(dev)

            # Pad fbank at the tail so subsampling emits >= n_fc_target frames.
            n_sub = self._subsample_len(int(lengths[0]))
            if align_to_codec and n_sub < n_fc_target:
                need = n_fc_target
                for _ in range(3):
                    need = (need - 1) * 2
                pad = need - fbank.size(1)
                if pad > 0:
                    fbank = F.pad(fbank, (0, 0, 0, pad))
                lengths = torch.full_like(lengths, max(need, int(lengths[0])))

            out, out_lengths = self.encoder(fbank, lengths)

            if self.layer >= 0:
                assert self._tap is not None, "tap hook did not fire"
                feats, self._tap = self._tap, None
            else:
                feats = out

        # out_norm + upsample are trainable — must be outside no_grad
        feats = self.out_norm(feats)

        if self._upsample is not None:
            feats = self._upsample(feats)

        # Trim to exact codec grid — subsampling always produces a few extra frames
        if align_to_codec:
            feats = feats[:, :n_codec]

        return feats
