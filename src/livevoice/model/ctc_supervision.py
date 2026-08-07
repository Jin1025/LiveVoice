"""CTC ASR-supervision head on the content memory — the LM-shortcut-proof alternative.

Why CTC here: the seq2seq head (asr_supervision.py) can minimize its loss with a
phoneme LANGUAGE MODEL over the teacher-forced previous phonemes and IGNORE the content
entirely (verified: debug/diag_asr_uses_content.py showed a content-usage gap ≈ 0). CTC
has NO autoregressive input — the prediction at each content frame depends ONLY on that
frame's embedding, so there is no previous-token channel to cheat through: the loss can
only go down if every frame's content is phonetically informative. That makes it a much
stronger "keep phonetic content" pressure on the per-frame representation the VC decoder
actually consumes.

NOTE — this is NOT the earlier failed CTC. That one hung off the DECODER hidden state and
competed with the codec-token loss on the same representation. This head hangs off the
CONTENT memory only (post sw2v_proj[+refiner][+FSQ] → sw2v_to_hidden), never touches the
main decoder, and is discarded at inference.

Vocab: blank=0, the 39 CMU phones = 1..39 (BOS/EOS/PAD are dropped — CTC targets are the
bare phone sequence). Content frames (T≈50fps) ≫ phones, so the CTC length constraint
(input_len ≥ target_len) holds comfortably.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phoneme_vocab import _CMU_PHONES  # 39 phones

CTC_BLANK = 0
NUM_CTC_CLASSES = len(_CMU_PHONES) + 1          # 40 (blank + 39 phones)
_FIRST_PHONE_ID = 3                              # main vocab: PAD0 BOS1 EOS2, phones 3..41
# main-vocab phone id p → CTC id (p - 2): 3→1 ... 41→39


class CtcSupervisionHead(nn.Module):
    """Per-frame CTC classifier over the content memory. Training-only, discarded at inference."""

    def __init__(self, config):
        super().__init__()
        d = int(config.hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(float(getattr(config, "dropout", 0.1))),
            nn.Linear(d, NUM_CTC_CLASSES),
        )
        print(
            f"[CtcSupervisionHead] per-frame CTC (LM-shortcut-proof), d_model={d} "
            f"classes={NUM_CTC_CLASSES} (blank + {len(_CMU_PHONES)} phones) "
            f"— training-only, discarded at inference"
        )

    def compute_loss(
        self,
        memory: torch.Tensor,          # (B, T, d_model) content embedding
        memory_len: torch.Tensor,      # (B,) true (unpadded) frame counts
        phoneme_ids: torch.Tensor,     # (B, L) main vocab: [BOS, phones, EOS, PAD...]
    ) -> torch.Tensor:
        B, T, _ = memory.shape
        # CTC is numerically fragile: force fp32 (log_softmax overflows/NaNs in fp16/AMP)
        # and the native (non-cuDNN) kernel (cuDNN CTC has known NaN-gradient cases).
        log_probs = self.proj(memory).float().log_softmax(dim=-1)  # (B, T, C) fp32
        log_probs = log_probs.transpose(0, 1)                      # (T, B, C) for F.ctc_loss

        input_lengths = memory_len.clamp(min=1, max=T).to(torch.long)

        # targets: drop BOS/EOS/PAD (ids < 3), remap phone id p → p-2, concat row-major.
        valid = phoneme_ids >= _FIRST_PHONE_ID              # (B, L) bool
        target_lengths = valid.sum(dim=1).to(torch.long)
        targets = (phoneme_ids[valid] - (_FIRST_PHONE_ID - 1)).to(torch.long)  # 1..39, row-major

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                log_probs, targets, input_lengths, target_lengths,
                blank=CTC_BLANK, zero_infinity=True,
            )
        # last-resort guard: never let a bad batch poison the model with NaN/Inf.
        if not torch.isfinite(loss):
            loss = log_probs.new_zeros(())
        return loss
