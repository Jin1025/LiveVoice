"""Speaker adversary via Gradient Reversal (GRL) on the content bottleneck.

Counterpart to the seq2seq ASR supervision (asr_supervision.py):
  ASR loss  → KEEP phonetic content   (protects WER)
  GRL loss  → REMOVE speaker identity  (the disentanglement lever)

A speaker classifier is trained on the mean-pooled content embedding
(post sw2v_proj [+ content_fsq] → sw2v_to_hidden — the exact representation the
decoder consumes). The Gradient Reversal layer flips the sign of the gradient
flowing back into the content path, so while the classifier learns to identify
the speaker, the content encoder learns to make the speaker UN-identifiable.

Why an adversary rather than a tighter bottleneck (FSQ/VQ): a capacity
bottleneck only limits *how much* information passes, not *which* — under
same-speaker reconstruction, speaker info is useful for recon, so a bottleneck
keeps it (and starves content, hurting WER). The adversary is the only lever
that targets speaker *specifically*. Probing (debug/probe_speaker_in_content.py)
showed sw2v content is ~24× chance decodable for speaker even after full
perturbation, which is what this removes.

Training-only: the classifier and GRL head are discarded at inference (they
never touch the decoder), so streaming/runtime is unaffected.

λ (reversal strength) follows the Ganin et al. (2015) schedule — ramped from 0
so the classifier becomes competent before the encoder is pushed to fool it;
starting at full strength tends to collapse training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def grl_lambda_schedule(
    step: int, warmup_steps: int, lambda_max: float, gamma: float, start_step: int = 0
) -> float:
    """Ganin schedule with an optional pure-classifier head-start.

    λ = 0 for step < start_step (the adversary trains WITHOUT reversal, so it can
    actually learn to identify the speaker before the encoder is pushed to fool it —
    a strong adversary is what makes the reversed gradient meaningful). After that,
    Ganin ramp: λ = lambda_max * (2/(1+e^{-gamma*p}) - 1), p = (step-start)/warmup ∈ [0,1].
    """
    if step < start_step:
        return 0.0
    p = min(1.0, max(0.0, (step - start_step) / max(1, warmup_steps)))
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-float(gamma) * p)) - 1.0)


class SpeakerGRLHead(nn.Module):
    """Adversarial speaker classifier over the pooled content embedding. Training-only."""

    def __init__(self, config, num_speakers: int):
        super().__init__()
        if num_speakers < 2:
            raise ValueError(
                f"SpeakerGRLHead needs >= 2 speakers, got {num_speakers}. "
                "Set config.grl_num_speakers from the train-split vocab (see train.py)."
            )
        d = int(config.hidden_dim)
        h = int(getattr(config, "grl_hidden_dim", 0)) or d
        self.num_speakers = int(num_speakers)
        self.net = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(inplace=True),
            nn.Dropout(float(getattr(config, "dropout", 0.1))),
            nn.Linear(h, self.num_speakers),
        )
        print(
            f"[SpeakerGRLHead] adversary MLP d_model={d} hidden={h} "
            f"num_speakers={self.num_speakers} — GRL, training-only, discarded at inference"
        )

    def forward(self, pooled: torch.Tensor, lambd: float) -> torch.Tensor:
        """pooled: (B, d_model) → logits (B, num_speakers), with reversed gradient to `pooled`."""
        return self.net(grad_reverse(pooled, lambd))

    def compute_loss(self, pooled: torch.Tensor, speaker_labels: torch.Tensor, lambd: float) -> torch.Tensor:
        # ignore_index=-1 skips out-of-vocab (e.g. unseen val) speakers defensively;
        # in training all labels come from the train-split vocab so nothing is skipped.
        return F.cross_entropy(self.forward(pooled, lambd), speaker_labels, ignore_index=-1)

    def compute_confusion_loss(
        self, pooled: torch.Tensor, speaker_labels: torch.Tensor, lambd: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """BOUNDED alternative to gradient reversal (Tzeng et al. domain confusion).

        Gradient reversal makes the content path MAXIMIZE the classifier's cross-entropy,
        which is unbounded above (push the wrong-class logits to ±inf and CE → inf). That
        is the runaway we kept hitting: grl_loss climbs past chance and training NaNs out.

        Confusion instead asks the content path to make the classifier's output UNIFORM:
            conf = -(1/K) Σ_k log p_k     (minimized, = log K, exactly at uniform)
        It is bounded below, so there is nothing to run away to — "maximally confused" is
        a finite target rather than an infinite one.

        Two terms with separated gradients:
          * classifier trains normally on CE over DETACHED pooled (stays a strong adversary)
          * content path gets only the confusion gradient (classifier params frozen for it)

        Returns (loss, acc) with the same semantics as compute_loss.
        """
        # 1) adversary update — detached input, so this never touches the content path.
        logits_clf = self.net(pooled.detach())
        loss_clf = F.cross_entropy(logits_clf, speaker_labels, ignore_index=-1)

        # 2) content update — freeze classifier params so the confusion term only shapes
        #    `pooled` (otherwise it would also teach the adversary to be uninformative).
        for p in self.net.parameters():
            p.requires_grad_(False)
        try:
            logits_ctn = self.net(pooled)
        finally:
            for p in self.net.parameters():
                p.requires_grad_(True)
        log_probs = F.log_softmax(logits_ctn, dim=-1)
        keep = speaker_labels != -1
        conf = -log_probs[keep].mean() if keep.any() else log_probs.new_zeros(())

        loss = loss_clf + float(lambd) * conf
        with torch.no_grad():
            acc = ((logits_clf[keep].argmax(dim=-1) == speaker_labels[keep]).float().mean()
                   if keep.any() else torch.zeros((), device=logits_clf.device))
        # Expose the two parts: the combined value is CE + λ·conf, which converges to
        # ln(K)·(1+λ) — NOT ln(K) — so it is easy to misread as "not reaching chance".
        # clf → ln(K) means the adversary is at chance; conf → ln(K) means uniform output.
        self.last_parts = {"grl_clf_loss": loss_clf.detach(), "grl_conf_loss": conf.detach()}
        return loss, acc
