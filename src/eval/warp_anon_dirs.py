"""Post-hoc time-warp of already-anonymized VPC dirs, to price the rhythm cue.

Why post-hoc and not a generation flag: the audio already exists. `ablate_asv_cues.py` found
that destroying rhythm is the only ablation that reaches the target --

    condition   ORIG d'   ANON d'   observed/predicted-from-damage
    none         2.365     0.607
    rhythm       0.771     0.139     0.70   <- carries identity beyond generic damage
    pitch        1.223     0.467     1.49   <- matters LESS after conversion
    detail       2.238     0.533     0.93
    (EER 45% needs ANON d' <= 0.251)

-- so the question is what a per-utterance random warp costs in WER and UAR when the OFFICIAL
evaluator scores it, not just what it does to d'. Warping the finished wavs answers that for
the price of a few CPU-minutes; regenerating would cost ~13 GPU-h and change nothing else.

WHAT THIS IS AND IS NOT. It is a measurement: it puts a number on the ceiling that removing
the rhythm cue would buy, and so decides whether a duration model is worth building. It is not
a shippable method. Our converter is frame-aligned 1:1 with the source at 50 fps, and a
constant-rate stretch makes the output drift from the input by rate*T -- 1.5 s over a 10 s
utterance at 15%. That is fine for scoring files offline and fatal for a streaming claim. A
real fix changes the architecture (predict durations, break the 1:1 alignment); this only says
whether that is worth doing.

The warp is drawn per utterance from the utterance id, so it is identical on every rerun, and
it reads no speaker labels -- VPC 2026 §2.1 constrains the pseudo-speaker assignment, which
this does not touch. Note §2.1 also asks not to distort emotional states, and speaking rate is
an emotion cue: the UAR this produces is the honest cost, not a detail to omit.

    conda run -n sound python src/eval/warp_anon_dirs.py \
        --src_suffix _lv_vctk1fix --dst_suffix _lv_vctk1fix_warp --rate_min 0.12 --rate_max 0.25
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

DEFAULT_VPC_ROOT = "/mnt/data/disk3/yejin/VPC"
STAMP = ".livevoice_anon.json"
DATASETS = ("libri_dev_enrolls", "libri_dev_trials_mixed",
            "libri_test_enrolls", "libri_test_trials_mixed",
            "IEMOCAP_dev", "IEMOCAP_test")


def _stretch(y: np.ndarray, rate: float) -> np.ndarray:
    try:
        return librosa.effects.time_stretch(y=y, rate=rate)
    except TypeError:                      # librosa < 0.10 takes it positionally
        return librosa.effects.time_stretch(y, rate)


def _rate_for(utt: str, seed: int, lo: float, hi: float) -> float:
    """Deterministic per-utterance rate, signed so it is never a no-op near 1.0."""
    r = random.Random(f"{seed}:{utt}")
    return 1.0 + r.choice([-1, 1]) * r.uniform(lo, hi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vpc_root", default=DEFAULT_VPC_ROOT)
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--src_suffix", default="_lv_vctk1fix")
    p.add_argument("--dst_suffix", required=True)
    p.add_argument("--rate_min", type=float, default=0.12)
    p.add_argument("--rate_max", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    if args.dst_suffix == args.src_suffix:
        raise SystemExit("--dst_suffix must differ from --src_suffix (this writes a copy)")
    if not 0.0 <= args.rate_min <= args.rate_max < 1.0:
        raise SystemExit(f"need 0 <= rate_min <= rate_max < 1, got "
                         f"{args.rate_min}..{args.rate_max}")
    root = Path(args.vpc_root).resolve()

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        src, dst = root / "data" / f"{ds}{args.src_suffix}", root / "data" / f"{ds}{args.dst_suffix}"
        if not (src / "wav.scp").is_file():
            print(f"[warp] skip {ds}: {src}/wav.scp missing")
            continue
        if not (src / STAMP).is_file():
            raise SystemExit(f"{src} has no {STAMP} — refusing to treat a directory we did "
                             f"not generate as our own")
        (dst / "wav").mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(src / "*")):          # metadata only, never the wav/ dir
            if os.path.isfile(f) and os.path.basename(f) not in ("wav.scp", STAMP):
                shutil.copy(f, dst)

        utts = [l.split()[0] for l in open(src / "wav.scp") if l.strip()]
        rates = []
        for utt in tqdm(utts, desc=f"{ds}{args.dst_suffix}"):
            rate = _rate_for(utt, args.seed, args.rate_min, args.rate_max)
            rates.append(rate)
            y, sr = sf.read(str(src / "wav" / f"{utt}.wav"), dtype="float32", always_2d=True)
            w = _stretch(y.mean(axis=1), rate)
            w = w / (np.abs(w).max() + 1e-8)
            tmp = dst / "wav" / f".{utt}.partial.wav"
            sf.write(str(tmp), w, sr, subtype="PCM_16")
            os.replace(tmp, dst / "wav" / f"{utt}.wav")

        with open(dst / "wav.scp", "w", encoding="utf-8") as f:
            for utt in utts:
                f.write(f"{utt} data/{ds}{args.dst_suffix}/wav/{utt}.wav\n")
        prev = json.loads((src / STAMP).read_text())
        prev.update({"warp_from": f"{ds}{args.src_suffix}", "warp_seed": args.seed,
                     "warp_rate_min": args.rate_min, "warp_rate_max": args.rate_max})
        (dst / STAMP).write_text(json.dumps(prev, indent=2, sort_keys=True))
        r = np.array(rates)
        print(f"[warp] {ds}{args.dst_suffix}: {len(utts)} utts, "
              f"rate {r.min():.3f}..{r.max():.3f} (|mean dev| {np.abs(r-1).mean():.3f})")

    print(f"\n[warp] then, from {root}:\n"
          f"  python run_evaluation.py --config configs/track1/eval_pre.yaml "
          f"--overwrite '{{\"anon_data_suffix\": \"{args.dst_suffix}\"}}'")


if __name__ == "__main__":
    main()
