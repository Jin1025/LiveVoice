"""Re-project anonymized audio from the VPC 2026 tree into the VPC 2024 tree. No regeneration.

The two challenges evaluate the SAME recordings; only the bookkeeping differs. Verified against
both checkouts:

    libri_*_enrolls, IEMOCAP_dev, IEMOCAP_test   identical utterance lists (343/438/2746/2785)
    libri_dev_trials_f  + libri_dev_trials_m  == libri_dev_trials_mixed   (1018+960 = 1978)
    libri_test_trials_f + libri_test_trials_m == libri_test_trials_mixed  ( 734+762 = 1496)
    -> intersection exact, zero utterances on either side only

So VPC24's f/m split is a PARTITION of VPC26's mixed set, and every wav VPC24 needs has already
been generated. Anonymizing again would cost ~13 GPU-h and, because generation is not
bit-deterministic, would also make the 2024 and 2026 numbers describe two different systems --
which is the one thing a cross-protocol comparison must not do. This hard-links instead: same
bytes, same inode, both trees.

What actually differs between the protocols, and therefore what the comparison isolates:

    attacker ASV   VPC24 exp/asv_orig (ECAPA-TDNN)  vs  VPC26 exp/asv_ssl (WavLM-Large+ECAPA)
    trial split    f and m scored separately        vs  mixed

The split turns out not to matter for a single-pseudo-speaker system -- measured on our own
audio, mixed 33.83 vs f/m mean 34.08 on libri_dev (+0.25), because the pseudo-speaker already
erases gender. The attacker does: ECAPA sees d'=0.07 on this audio where WavLM+ECAPA sees 0.75.
StreamVoiceAnon+ reports EER-L 47.19 (vctk-1fix) under the 2024 protocol, i.e. against ECAPA.

    conda run -n sound python src/eval/port_anon_to_vpc24.py --anon_suffix _lv_vctk1fix
    cd /mnt/data/disk3/yejin/VPC24 && source env.sh
    python run_evaluation.py --config configs/eval_pre.yaml \
        --overwrite '{"anon_data_suffix": "_lv_vctk1fix"}'
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

STAMP = ".livevoice_anon.json"

# dst dataset (VPC24) -> src dataset (VPC26) holding its wavs
MAP = {
    "libri_dev_enrolls":       "libri_dev_enrolls",
    "libri_dev_trials_f":      "libri_dev_trials_mixed",
    "libri_dev_trials_m":      "libri_dev_trials_mixed",
    "libri_test_enrolls":      "libri_test_enrolls",
    "libri_test_trials_f":     "libri_test_trials_mixed",
    "libri_test_trials_m":     "libri_test_trials_mixed",
    "IEMOCAP_dev":             "IEMOCAP_dev",
    "IEMOCAP_test":            "IEMOCAP_test",
}


def _link(src: Path, dst: Path) -> None:
    """Hard-link, falling back to copy across filesystems. Never leave a partial file."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        tmp = dst.with_name(f".{dst.name}.partial")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_root", default="/mnt/data/disk3/yejin/VPC")
    p.add_argument("--dst_root", default="/mnt/data/disk3/yejin/VPC24")
    p.add_argument("--anon_suffix", required=True,
                   help="the suffix as it exists in --src_root, reused verbatim in --dst_root")
    p.add_argument("--datasets", default=",".join(MAP))
    args = p.parse_args()

    src_root, dst_root = Path(args.src_root).resolve(), Path(args.dst_root).resolve()
    sfx = args.anon_suffix
    if not (dst_root / "run_evaluation.py").is_file():
        raise SystemExit(f"{dst_root} does not look like a VPC checkout")

    total = 0
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if ds not in MAP:
            raise SystemExit(f"unknown dataset {ds!r}; known: {sorted(MAP)}")
        meta = dst_root / "data" / ds                       # VPC24's own metadata
        src = src_root / "data" / f"{MAP[ds]}{sfx}"         # our generated audio
        dst = dst_root / "data" / f"{ds}{sfx}"
        if not (meta / "wav.scp").is_file():
            raise SystemExit(f"missing VPC24 dataset {meta} — run 01_download_data_model.sh")
        if not (src / "wav.scp").is_file():
            raise SystemExit(f"missing anonymized source {src}")
        if not (src / STAMP).is_file():
            raise SystemExit(f"{src} has no {STAMP} — refusing to copy a dir we did not make")

        utts = [l.split()[0] for l in open(meta / "wav.scp") if l.strip()]
        have = {l.split()[0] for l in open(src / "wav.scp") if l.strip()}
        missing = [u for u in utts if u not in have]
        if missing:
            raise SystemExit(f"{ds}: {len(missing)} utt(s) not anonymized yet, e.g. "
                             f"{missing[:3]} — generate {MAP[ds]}{sfx} first")

        (dst / "wav").mkdir(parents=True, exist_ok=True)
        # VPC24's metadata, not VPC26's: trials/spk2utt/utt2spk differ between the protocols
        # and are what define the f/m split we are here to evaluate.
        for f in glob.glob(str(meta / "*")):
            if os.path.isfile(f) and os.path.basename(f) != "wav.scp":
                shutil.copy(f, dst)
        for u in utts:
            _link(src / "wav" / f"{u}.wav", dst / "wav" / f"{u}.wav")
        with open(dst / "wav.scp", "w", encoding="utf-8") as f:
            for u in utts:
                f.write(f"{u} data/{ds}{sfx}/wav/{u}.wav\n")

        stamp = json.loads((src / STAMP).read_text())
        stamp.update({"ported_from": f"{src_root}/data/{MAP[ds]}{sfx}", "protocol": "vpc2024"})
        (dst / STAMP).write_text(json.dumps(stamp, indent=2, sort_keys=True))
        print(f"[port] {ds}{sfx}: {len(utts)} utts  <- {MAP[ds]}{sfx}")
        total += len(utts)

    print(f"\n[port] {total} utterances linked, 0 regenerated")
    print(f"[port] then, from {dst_root}:\n"
          f"  source env.sh\n"
          f"  python run_evaluation.py --config configs/eval_pre.yaml "
          f"--overwrite '{{\"anon_data_suffix\": \"{sfx}\"}}'")


if __name__ == "__main__":
    main()
