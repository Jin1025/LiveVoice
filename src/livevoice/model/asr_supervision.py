"""Seq2seq (attention-decoder) ASR supervision head — StyleStream-style text supervision
for the sw2v content bottleneck.

Confirmed against the StyleStream paper (arXiv:2602.20113): "the whole pipeline is
trained end-to-end with a sequence-to-sequence ASR loss" — NOT CTC — character-level
labels, a 4-layer transformer decoder (hidden 768, ffn 3072), discarded at inference
(only the content tokenizer, "Destylizer", is kept).

This head hangs off the CONTENT EMBEDDING ALONE (after sw2v_proj [+ content_fsq], i.e.
the "Destylizer" analogue), never touches the main VC decoder, and is deleted at
inference — so it can't affect streaming/runtime. The point isn't to improve WER
directly; it's to give the FSQ bottleneck a supervised reason to keep
phonetically-relevant channels and drop speaker-only ones, which unsupervised FSQ (see
FSQBottleneck) has no way to do. (An earlier CTC-on-the-decoder-hidden-state attempt made
the main codec-token prediction worse — the two objectives competed on the same
representation — and was removed; this path deliberately avoids that by staying off the
decoder entirely.)

Label unit here is phoneme (CMU ARPAbet, see phoneme_vocab.py) rather than StyleStream's
character — deliberate deviation: phonemes are the more standard unit for disentanglement
in the classical PPG-VC literature (stress-free, coarser, less speaker-specific than
raw characters/graphemes).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phoneme_vocab import PAD_ID, PHONEME_VOCAB_SIZE


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        return x + self.pe[: x.size(1)].unsqueeze(0).to(dtype=x.dtype)


class AsrSupervisionHead(nn.Module):
    """Small seq2seq ASR decoder over the content embedding. Training-only."""

    def __init__(self, config):
        super().__init__()
        d_model = int(config.hidden_dim)
        n_layers = int(getattr(config, "asr_decoder_layers", 4))
        n_heads = int(getattr(config, "num_heads", 8))
        ffn_dim = 4 * d_model
        max_phon = int(getattr(config, "asr_max_phoneme_len", 300)) + 1
        vocab_size = PHONEME_VOCAB_SIZE

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model, max_len=max_phon)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=float(getattr(config, "dropout", 0.1)), batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)
        print(
            f"[AsrSupervisionHead] seq2seq (StyleStream-style, phoneme-level), "
            f"layers={n_layers} d_model={d_model} ffn={ffn_dim} vocab={vocab_size} "
            f"— training-only, discarded at inference"
        )

    def forward(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """memory: (B, T_mem, d_model) content embedding (post sw2v_proj [+ FSQ]).
        tgt_ids: (B, T_tgt) teacher-forced input ids (BOS + phones, EOS excluded).
        Returns logits (B, T_tgt, vocab_size).
        """
        tgt = self.pos_enc(self.tok_emb(tgt_ids))
        T_tgt = tgt_ids.size(1)
        # Bool causal mask (not the float -inf variant) so its dtype matches the bool
        # key-padding masks below — avoids PyTorch's "mismatched mask dtype" warning/path.
        causal_mask = torch.triu(
            torch.ones(T_tgt, T_tgt, device=tgt.device, dtype=torch.bool), diagonal=1
        )
        tgt_key_padding_mask = tgt_ids.eq(PAD_ID)
        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(dec_out)

    def compute_loss(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
        phoneme_ids: torch.Tensor,
    ) -> torch.Tensor:
        """phoneme_ids: (B, T) = [BOS, ph..., EOS, PAD...]. Teacher-forces on [:-1],
        predicts [1:]; padded target positions are excluded via ignore_index."""
        tgt_in = phoneme_ids[:, :-1]
        tgt_out = phoneme_ids[:, 1:]
        logits = self.forward(memory, memory_key_padding_mask, tgt_in)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD_ID,
        )
