"""Which cue is the VPC attacker actually using on our anonymized audio?

The measurements so far bound the problem but do not solve it:
    plain ECAPA on the output      d' = 0.07   -> timbre was fully replaced
    speaking rate alone            d' = 0.16   -> rhythm is not the whole story
    Zipformer content probe        0.156 (chance 0.033, and it overfits)
    VPC asv_ssl on the output      d' = 0.75   -> something survived, EER 33 not 50

So the attacker reads a cue that plain ECAPA does not. This script names it, by DESTROYING
one cue at a time and re-measuring d' with the attacker's own encoder (WavLM-Large + ECAPA,
exp/asv_ssl). Each transform is drawn independently per utterance, so utterances of one
source speaker no longer share that cue -- which is exactly what makes it useless for
identification.

    rhythm   phase-vocoder time-stretch, random rate   (pitch and timbre preserved)
    pitch    pitch-shift, random semitones             (rhythm and duration preserved)
    detail   16k -> 6k -> 16k resample                 (fine spectral structure removed)

Read the result as a RELATIVE drop, never an absolute one: every transform also degrades the
audio, which lowers d' on its own. The control is the same transform applied to the ORIGINAL
recordings. A transform that costs the original 10% of its d' but the anonymized audio 70% has
found the cue; one that costs both about the same has only added damage.

Must run under the VPC venv (it owns WavLM + their ECAPA):

    cd /mnt/data/disk3/yejin/VPC && source env.sh
    PYTHONPATH=/mnt/data/disk3/yejin/VPC python \
        /workspace/LiveVoice/src/debug/ablate_asv_cues.py --n_utts 150
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

VPC = Path(os.environ.get("VPC_ROOT", "/mnt/data/disk3/yejin/VPC"))
sys.path.insert(0, str(VPC))


def _kv(p: Path) -> dict[str, str]:
    return {l.split()[0]: l.split(None, 1)[1].strip() for l in open(p) if l.strip()}


def _load(path: str, sr: int = 16000) -> np.ndarray:
    import librosa
    y, s = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if s != sr:
        y = librosa.resample(y, orig_sr=s, target_sr=sr)
    return y.astype("float32")


def _stretch(y, rate):
    import librosa
    try:
        return librosa.effects.time_stretch(y=y, rate=rate)
    except TypeError:                       # librosa < 0.10 is positional
        return librosa.effects.time_stretch(y, rate)


def _shift(y, sr, steps):
    import librosa
    try:
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)
    except TypeError:
        return librosa.effects.pitch_shift(y, sr, steps)


def transform(y: np.ndarray, sr: int, cond: str, rng: random.Random) -> np.ndarray:
    import librosa
    if cond == "none":
        return y
    if cond == "rhythm":
        # away from 1.0 in both directions, never ~1.0 (which would be a no-op)
        r = rng.choice([-1, 1]) * rng.uniform(0.12, 0.25)
        return _stretch(y, 1.0 + r)
    if cond == "pitch":
        s = rng.choice([-1, 1]) * rng.uniform(2.0, 5.0)
        return _shift(y, sr, s)
    if cond == "detail":
        return librosa.resample(librosa.resample(y, orig_sr=sr, target_sr=6000),
                                orig_sr=6000, target_sr=sr)
    raise ValueError(cond)


def dprime(emb: dict[str, np.ndarray], spk: dict[str, str], n: int, seed: int):
    by = defaultdict(list)
    for u, e in emb.items():
        by[spk[u]].append(e)
    by = {k: v for k, v in by.items() if len(v) >= 2}
    keys, rng = list(by), random.Random(seed)
    same, diff = [], []
    for _ in range(n):
        k = rng.choice(keys)
        a, b = rng.sample(by[k], 2)
        same.append(float(a @ b))
        k2 = rng.choice([x for x in keys if x != k])
        diff.append(float(rng.choice(by[k]) @ rng.choice(by[k2])))
    same, diff = np.array(same), np.array(diff)
    gap = same.mean() - diff.mean()
    return gap / np.sqrt(0.5 * (same.var() + diff.var()) + 1e-12)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="libri_dev_trials_mixed")
    p.add_argument("--anon_suffix", default="_lv_vctk1fix")
    p.add_argument("--conditions", default="none,rhythm,pitch,detail")
    p.add_argument("--n_utts", type=int, default=150)
    p.add_argument("--n_pairs", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from evaluation.privacy.asv.speechbrain_vectors import SpeechBrainVectors
    sv = SpeechBrainVectors("ecapa_ssl", args.device, VPC / "exp" / "asv_ssl")

    ds, sfx = args.dataset, args.anon_suffix
    spk = _kv(VPC / "data" / ds / "utt2spk")
    scp = _kv(VPC / "data" / ds / "wav.scp")
    utts = sorted(u for u in _kv(VPC / "data" / f"{ds}{sfx}" / "wav.scp") if u in spk)
    random.Random(args.seed).shuffle(utts)
    utts = utts[: args.n_utts]
    print(f"[ablate] {len(utts)} utts, {len(set(spk[u] for u in utts))} source speakers")

    srcs = {"ANON": lambda u: str(VPC / "data" / f"{ds}{sfx}" / "wav" / f"{u}.wav"),
            "ORIG": lambda u: (scp[u] if os.path.isabs(scp[u]) else str(VPC / scp[u]))}
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    out: dict[tuple[str, str], float] = {}

    with tempfile.TemporaryDirectory() as td:
        for tag, get in srcs.items():
            for cond in conds:
                emb = {}
                # one rng per (source, condition) so ANON and ORIG get the SAME draws --
                # otherwise the two columns differ by transform luck as well as by content
                rng = random.Random(f"{args.seed}:{cond}")
                for i, u in enumerate(utts):
                    y = transform(_load(get(u)), 16000, cond, rng)
                    y = y / (np.abs(y).max() + 1e-8)
                    wp = os.path.join(td, "x.wav")
                    sf.write(wp, y, 16000, subtype="PCM_16")
                    with torch.no_grad():
                        v = sv.extract_vector(torch.from_numpy(y), 16000, wav_path=wp)
                    v = v.detach().float().cpu().numpy().reshape(-1)
                    emb[u] = v / (np.linalg.norm(v) + 1e-8)
                    if (i + 1) % 50 == 0:
                        print(f"  {tag}/{cond}: {i+1}/{len(utts)}", flush=True)
                out[(tag, cond)] = dprime(emb, spk, args.n_pairs, args.seed)
                print(f"  -> {tag:4s} {cond:7s} d' = {out[(tag,cond)]:.3f}", flush=True)

    print("\n================ CUE ABLATION (attacker = exp/asv_ssl) ================")
    print(f"{'condition':10s} {'ORIG d':>8s} {'ANON d':>8s} {'ORIG drop':>10s} {'ANON drop':>10s}")
    o0, a0 = out[("ORIG", "none")], out[("ANON", "none")]
    for cond in conds:
        o, a = out[("ORIG", cond)], out[("ANON", cond)]
        print(f"{cond:10s} {o:8.3f} {a:8.3f} {100*(1-o/o0):9.1f}% {100*(1-a/a0):9.1f}%")
    print("\nThe cue is whichever condition drops ANON far more than it drops ORIG.")
    print(f"For reference, EER 45% needs ANON d' <= 0.251 (now {a0:.3f}).")


if __name__ == "__main__":
    main()
