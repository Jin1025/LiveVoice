"""Restore each utterance's original loudness onto the anonymized audio, then re-score SER.

Why: sad is the only emotion the SER cannot recover from our output -- 3.72/3.90 against an
original 63.63/72.57, i.e. a seventh of the 25% chance rate, which is not "hard" but "never
predicted". Measured on 600 IEMOCAP_dev utterances:

    emo    ORIG rms   ANON rms        sad/ang RMS ratio:  orig 0.145  ->  anon 0.674
    sad     0.0106     0.1129
    neu     0.0180     0.1227
    ang     0.0734     0.1674
    hap     0.0331     0.1351

Sadness is 7x quieter than anger in the recordings and only 1.5x quieter in ours; sad
utterances came out 10x louder than they went in. Loudness is one of the strongest cues for
sadness, and `_load_full_mono_wav` peak-normalises every input to 1.0 before the model sees it,
so the model is never told how loud the utterance was. That information is not lost, though --
it is still in the source file, and can simply be put back.

This rescales each anonymized wav so its peak matches the corresponding SOURCE utterance's
peak. It carries no speaker identity beyond what the original recording already had: it is one
scalar per utterance, applied identically regardless of who spoke, and it does not depend on
speaker labels (VPC 2026 §2.1). It should leave WER untouched and EER nearly so; the point is
to test whether energy is what the SER is missing, before rebuilding 9,786 utterances.

If UAR-sad recovers, fold the same rescale into anonymize_vpc_dirs.py's write path. If it does
not, sadness is being lost somewhere other than loudness and this line of attack is closed.

    conda run -n sound python src/eval/restore_level_dirs.py \
        --src_suffix _lv_vctk1fix --dst_suffix _lv_vctk1fix_lvl \
        --datasets IEMOCAP_dev,IEMOCAP_test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
from tqdm import tqdm

DEFAULT_VPC_ROOT = "/mnt/data/disk3/yejin/VPC"
STAMP = ".livevoice_anon.json"


def _read_src(entry: str, root: Path):
    """wav.scp value -> (audio, sr). Pipe entries decode the file directly (no `flac` binary)."""
    e = entry.strip()
    if e.endswith("|"):
        for tok in e[:-1].split():
            if tok.lower().endswith((".flac", ".wav", ".ogg", ".opus", ".mp3")):
                e = tok
                break
        else:
            raise ValueError(f"cannot decode pipe entry: {entry!r}")
    p = Path(e)
    y, sr = sf.read(str(p if p.is_absolute() else root / p), dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vpc_root", default=DEFAULT_VPC_ROOT)
    p.add_argument("--datasets", default="IEMOCAP_dev,IEMOCAP_test")
    p.add_argument("--src_suffix", default="_lv_vctk1fix")
    p.add_argument("--dst_suffix", required=True)
    p.add_argument("--match", default="peak", choices=["peak", "rms"],
                   help="peak matches the source peak (safe, never clips); rms matches mean "
                        "energy and is closer to perceived loudness but can clip, so it is "
                        "limited to the headroom the source had")
    args = p.parse_args()

    if args.dst_suffix == args.src_suffix:
        raise SystemExit("--dst_suffix must differ from --src_suffix")
    root = Path(args.vpc_root).resolve()

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        base, src = root / "data" / ds, root / "data" / f"{ds}{args.src_suffix}"
        dst = root / "data" / f"{ds}{args.dst_suffix}"
        if not (src / STAMP).is_file():
            raise SystemExit(f"{src} has no {STAMP} — refusing to touch a dir we did not make")
        scp = {l.split()[0]: l.split(None, 1)[1].strip()
               for l in open(base / "wav.scp") if l.strip()}
        utts = [l.split()[0] for l in open(src / "wav.scp") if l.strip()]

        (dst / "wav").mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(src / "*")):
            if os.path.isfile(f) and os.path.basename(f) not in ("wav.scp", STAMP):
                shutil.copy(f, dst)

        gains, clipped = [], 0
        for utt in tqdm(utts, desc=f"{ds}{args.dst_suffix}"):
            y, _ = sf.read(str(src / "wav" / f"{utt}.wav"), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            o, _ = _read_src(scp[utt], root)
            if args.match == "peak":
                tgt, cur = np.abs(o).max(), np.abs(y).max()
            else:
                tgt, cur = np.sqrt((o ** 2).mean()), np.sqrt((y ** 2).mean())
            g = float(tgt / max(cur, 1e-8))
            # never exceed full scale: the source's own peak is the ceiling
            if (peak := np.abs(y).max() * g) > 1.0:
                g *= 1.0 / peak
                clipped += 1
            gains.append(g)
            w = (y * g).astype("float32")
            tmp = dst / "wav" / f".{utt}.partial.wav"
            sf.write(str(tmp), w, 16000, subtype="PCM_16")
            os.replace(tmp, dst / "wav" / f"{utt}.wav")

        with open(dst / "wav.scp", "w", encoding="utf-8") as f:
            for utt in utts:
                f.write(f"{utt} data/{ds}{args.dst_suffix}/wav/{utt}.wav\n")
        st = json.loads((src / STAMP).read_text())
        st.update({"level_from": f"{ds}{args.src_suffix}", "level_match": args.match})
        (dst / STAMP).write_text(json.dumps(st, indent=2, sort_keys=True))
        g = np.array(gains)
        print(f"[level] {ds}{args.dst_suffix}: {len(utts)} utts, gain "
              f"{g.min():.3f}..{g.max():.3f} (median {np.median(g):.3f}), "
              f"{clipped} limited to avoid clipping")

    print(f"\n[level] then, from {root}:\n"
          f"  python run_evaluation.py --config configs/track1/eval_pre.yaml "
          f"--overwrite '{{\"anon_data_suffix\": \"{args.dst_suffix}\"}}'")


if __name__ == "__main__":
    main()
