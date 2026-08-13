"""Where did the anonymized voice actually land — on the pseudo-speaker, or still on the source?

Answers the question the EER leaves ambiguous. A single-pseudo-speaker system should drive the
lazy-informed EER to ~50%: every anonymized utterance carries the SAME voice, so the attacker
cannot tell target trials from non-target ones. Measuring 30-33% instead means something
speaker-discriminative survived anonymization — but the EER alone does not say what.

Three measurements, on ECAPA embeddings (the same encoder the S-SIM numbers came from):

  1. anon->src vs anon->tgt cosine
        anon->tgt high and anon->src low  => timbre was replaced, leak is elsewhere
        anon->src high                    => the converter is still copying the source voice

  2. same-source-speaker vs different-source-speaker similarity AMONG anonymized utterances
        This is what the attacker exploits, stated directly. If anonymized utterances of one
        source speaker are more similar to each other than to another source speaker's, the
        source identity survived as a cluster structure and the EER cannot reach 50%.
        The gap is the leak; its size predicts how far the EER sits below chance.

  3. the same two contrasts on the ORIGINAL audio, as the scale reference — a gap that is
        90% of the original's means almost nothing was removed.

    conda run -n sound python src/debug/probe_anon_leakage.py \
        --anon_dir /mnt/data/disk3/yejin/VPC/data/libri_dev_trials_mixed_lv_vctk1fix \
        --orig_dir /mnt/data/disk3/yejin/VPC/data/libri_dev_trials_mixed
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _read_kaldi(p: Path) -> dict[str, str]:
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            k = line.strip().split(None, 1)
            if len(k) == 2:
                out[k[0]] = k[1]
    return out


def _load(path: str, sr: int = 16000) -> torch.Tensor:
    a, s = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(a).float().mean(dim=1)
    if s != sr:
        w = torchaudio.functional.resample(w, s, sr)
    return w / (w.abs().max() + 1e-8)


def _pairs(emb: dict[str, np.ndarray], spk: dict[str, str], n: int, seed: int):
    """(same-source-speaker sims, different-source-speaker sims)."""
    by = defaultdict(list)
    for u, e in emb.items():
        by[spk[u]].append(e)
    by = {k: v for k, v in by.items() if len(v) >= 2}
    rng = random.Random(seed)
    same, diff = [], []
    keys = list(by)
    for _ in range(n):
        k = rng.choice(keys)
        a, b = rng.sample(by[k], 2)
        same.append(float(a @ b))
        k2 = rng.choice([x for x in keys if x != k])
        diff.append(float(rng.choice(by[k]) @ rng.choice(by[k2])))
    return np.array(same), np.array(diff)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anon_dir", required=True)
    p.add_argument("--orig_dir", required=True)
    p.add_argument("--prompt", default="/mnt/data/disk2/VCTK-Corpus/wav48/p231/p231_023.wav")
    p.add_argument("--n_utts", type=int, default=300)
    p.add_argument("--n_pairs", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from speechbrain.inference import EncoderClassifier
    src = "/mnt/data/disk2/yejin/LiveVoice/pretrained_models/speechbrain__spkrec-ecapa-voxceleb"
    if not os.path.isdir(src):
        src = "speechbrain/spkrec-ecapa-voxceleb"
    enc = EncoderClassifier.from_hparams(source=src, run_opts={"device": args.device})

    def embed(w: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            e = enc.encode_batch(w.unsqueeze(0).to(args.device)).squeeze().float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-8)

    anon_d, orig_d = Path(args.anon_dir), Path(args.orig_dir)
    spk = _read_kaldi(orig_d / "utt2spk")
    scp = _read_kaldi(orig_d / "wav.scp")
    utts = sorted(u for u in _read_kaldi(anon_d / "wav.scp") if u in spk and u in scp)
    random.Random(args.seed).shuffle(utts)
    utts = utts[: args.n_utts]
    print(f"[probe] {len(utts)} utts, {len(set(spk[u] for u in utts))} source speakers, "
          f"device={args.device}")

    tgt = embed(_load(args.prompt))
    ea, eo = {}, {}
    root = orig_d.parent.parent
    for i, u in enumerate(utts):
        ea[u] = embed(_load(str(anon_d / "wav" / f"{u}.wav")))
        v = scp[u]
        eo[u] = embed(_load(v if os.path.isabs(v) else str(root / v)))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(utts)}")

    a2s = np.array([float(ea[u] @ eo[u]) for u in utts])
    a2t = np.array([float(ea[u] @ tgt) for u in utts])
    o2t = np.array([float(eo[u] @ tgt) for u in utts])
    print("\n=== 1. where the anonymized voice landed (cosine, ECAPA) ===")
    print(f"  anon -> its own source utt : {a2s.mean():.3f} +- {a2s.std():.3f}")
    print(f"  anon -> target prompt      : {a2t.mean():.3f} +- {a2t.std():.3f}")
    print(f"  orig -> target prompt      : {o2t.mean():.3f} +- {o2t.std():.3f}   (floor)")

    for name, emb in (("ORIGINAL", eo), ("ANON", ea)):
        s, d = _pairs(emb, spk, args.n_pairs, args.seed)
        gap = s.mean() - d.mean()
        # d' — how separable the two distributions are; this is what the attacker scores on
        dprime = gap / np.sqrt(0.5 * (s.var() + d.var()) + 1e-12)
        print(f"\n=== 2. source-speaker structure among {name} utterances ===")
        print(f"  same source speaker : {s.mean():.3f} +- {s.std():.3f}")
        print(f"  diff source speaker : {d.mean():.3f} +- {d.std():.3f}")
        print(f"  gap                 : {gap:.3f}      d' = {dprime:.2f}")
        if name == "ORIGINAL":
            ref_gap, ref_dp = gap, dprime
        else:
            print(f"\n=== 3. how much of the source identity survived ===")
            print(f"  gap retained : {100*gap/max(ref_gap,1e-9):5.1f}%  "
                  f"({ref_gap:.3f} -> {gap:.3f})")
            print(f"  d'  retained : {100*dprime/max(ref_dp,1e-9):5.1f}%  "
                  f"({ref_dp:.2f} -> {dprime:.2f})")
            print("\n  A single pseudo-speaker can only reach EER~50% if this is ~0%. "
                  "Whatever remains is what the attacker is scoring on.")


if __name__ == "__main__":
    main()
