"""Isolate the GRL speaker classifier: can it learn speaker from the pooled content
embedding AT ALL (λ=0, no reversal)? If the online run shows grl_loss pinned at
ln(num_speakers) with grl_acc≈chance even at λ=0, either (a) labels are broken,
(b) the pooled embedding is constant, or (c) it's an optimization issue (tiny batch
vs 1151 classes). This reproduces the exact pooled → MLP path offline, on a small
fixed pool, and tells the three apart.

Fast: reads a few cached full-utterance sw2v features (no dataset glob, no model
load), applies a random sw2v_proj[+to_hidden] like the real path, mean-pools, and
trains only the classifier. Run:

    conda run -n sound python /workspace/LiveVoice/src/debug/diag_grl_classifier.py
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from livevoice.config import LiveVoiceConfig  # noqa: E402
from livevoice.data.speaker_vocab import build_libritts_speaker_vocab  # noqa: E402


def sample_cache(config, feats_dir: Path, vocab: dict, n_spk: int, utts: int, seed: int):
    """Pick n_spk speakers (present in the train vocab) with >= `utts` cached feats."""
    rng = random.Random(seed)
    per_spk: dict[str, list[Path]] = {}
    for spk_dir in sorted(feats_dir.iterdir()):
        if not spk_dir.is_dir() or spk_dir.name not in vocab:
            continue
        pts = sorted(spk_dir.glob("*.pt"))
        if len(pts) >= utts:
            per_spk[spk_dir.name] = pts
    speakers = sorted(per_spk)
    rng.shuffle(speakers)
    speakers = speakers[:n_spk]
    items = []
    for spk in speakers:
        pts = per_spk[spk][:]
        rng.shuffle(pts)
        for p in pts[:utts]:
            items.append((spk, p))
    return items, speakers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_spk", type=int, default=50)
    ap.add_argument("--utts", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_proj", action="store_true",
                    help="also train sw2v_proj/to_hidden (default: frozen random, tests raw decodability)")
    ap.add_argument("--full_vocab", action="store_true",
                    help="classify over the full 1151-way vocab (like the real run) instead of the sampled pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    config = LiveVoiceConfig(use_speaker_grl=True)
    vocab = build_libritts_speaker_vocab(config)
    feats_dir = Path(config.sw2v_features_dir) / "libritts"
    print(f"[diag] feats_dir={feats_dir}  train-vocab speakers={len(vocab)}")

    items, speakers = sample_cache(config, feats_dir, vocab, args.n_spk, args.utts, args.seed)
    print(f"[diag] sampled {len(items)} utts over {len(speakers)} speakers")

    # local label space (0..n_spk-1) unless --full_vocab (0..1150, real setting)
    if args.full_vocab:
        label_of = lambda spk: vocab[spk]                      # noqa: E731
        num_classes = len(vocab)
    else:
        local = {s: i for i, s in enumerate(speakers)}
        label_of = lambda spk: local[spk]                      # noqa: E731
        num_classes = len(speakers)
    print(f"[diag] num_classes={num_classes}  chance_acc={1.0/num_classes:.4f}  "
          f"chance_CE=ln={torch.log(torch.tensor(float(num_classes))).item():.3f}")

    # load + pool through a random projection matching the real content path
    proj = nn.Linear(1024, config.content_proj_dim)
    to_hidden = nn.Linear(config.content_proj_dim, config.hidden_dim)
    for m in (proj, to_hidden):
        for p in m.parameters():
            p.requires_grad = bool(args.train_proj)

    pooled_list, label_list = [], []
    with torch.no_grad():
        for spk, path in items:
            data = torch.load(path, map_location="cpu", weights_only=True)
            feats = data["feats"].float()                       # (T,1024)
            emb = to_hidden(proj(feats))                        # (T,768)
            pooled_list.append(emb.mean(dim=0))                 # (768,)
            label_list.append(label_of(spk))
    X = torch.stack(pooled_list)                                # (N,768)
    y = torch.tensor(label_list, dtype=torch.long)

    # sanity: is the pooled representation actually varied, and labels sane?
    print(f"[diag] pooled X: mean|std across items = {X.mean().item():.3f} / "
          f"{X.std(dim=0).mean().item():.4f}  (std~0 ⇒ constant-pooled BUG)")
    print(f"[diag] labels: unique={y.unique().numel()}  min={y.min().item()} max={y.max().item()}  "
          f"any(-1)={(y==-1).any().item()}")

    from livevoice.model.speaker_grl import SpeakerGRLHead
    clf_cfg = LiveVoiceConfig(use_speaker_grl=True)
    clf = SpeakerGRLHead(clf_cfg, num_classes)
    trainable = list(clf.parameters()) + ([*proj.parameters(), *to_hidden.parameters()] if args.train_proj else [])
    opt = torch.optim.Adam(trainable, lr=args.lr)

    N = X.shape[0]
    for step in range(args.steps):
        idx = torch.randint(0, N, (min(args.batch, N),))
        logits = clf(X[idx], lambd=0.0)                         # λ=0: no reversal, pure classifier
        loss = F.cross_entropy(logits, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            with torch.no_grad():
                full = clf(X, lambd=0.0)
                acc = (full.argmax(-1) == y).float().mean().item()
            print(f"[diag] step {step:4d}  CE={loss.item():.3f}  train_acc(all)={acc:.3f}")

    print("\n[diag] VERDICT:")
    print("  acc → high (≫ chance)  ⇒ classifier CAN learn speaker from pooled emb.")
    print("      ⇒ online stall is NOT a code bug: it's the small-batch/1151-class optimization")
    print("        (fix: bigger batch, higher adversary LR, or fewer classes).")
    print("  acc stays ≈ chance      ⇒ representation/labels problem (see the sanity line above).")


if __name__ == "__main__":
    main()
