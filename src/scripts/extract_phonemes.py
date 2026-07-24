"""Offline phoneme-id precompute for ASR supervision (config.use_asr_supervision).

Converts each LibriTTS utterance's `*.normalized.txt` to a CMU ARPAbet phoneme-id
sequence (g2p_en; see livevoice/model/phoneme_vocab.py) and caches it as
`{out_dir}/libritts/{spk}/{utt_id}.pt` — a LongTensor `[BOS, ph..., EOS]`, UNPADDED
and UNTRUNCATED (LibriTTSDataset._load_phonemes pads/truncates to
config.asr_max_phoneme_len at load time, so this cache is reusable across any max-len
setting).

CPU-only (g2p_en has no GPU path); use --shard/--num_shards to parallelize across
processes (same convention as extract_sw2v_features.py, minus the GPU part).

Usage (conda `sound`; needs `pip install g2p_en` — pulls in nltk, pure-Python):

    python scripts/extract_phonemes.py libritts \
        --libritts_path /mnt/data/disk2/LibriTTS \
        --splits train-clean-100,train-clean-360,dev-clean \
        --out_dir /mnt/data/disk2/yejin/LiveVoice/features/phonemes \
        --num_workers 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
from multiprocessing import Pool

import torch
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from livevoice.model.phoneme_vocab import text_to_phoneme_ids


def discover_libritts(libritts_path: str, splits: list[str]) -> list[tuple[str, str, str]]:
    """Returns (normalized_txt_path, speaker_id, utt_id) — text-driven, not wav-driven,
    since utterances without a transcript can't be phonemized anyway."""
    root = Path(libritts_path)
    items = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [warn] split not found: {split_dir}")
            continue
        for txt in sorted(split_dir.glob("**/*.normalized.txt")):
            speaker_id = txt.parts[-3]
            utt_id = txt.name[: -len(".normalized.txt")]
            items.append((str(txt), speaker_id, utt_id))
    return items


def _process_one(args: tuple[str, str, str, Path]) -> str | None:
    txt_path, spk, utt_id, save_path = args
    try:
        text = Path(txt_path).read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return None
        ids = text_to_phoneme_ids(text)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(ids, dtype=torch.long), save_path)
        return None
    except Exception as e:
        return f"{txt_path}: {e}"


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="dataset", required=True)
    pl = sub.add_parser("libritts")
    pl.add_argument("--libritts_path", default="/mnt/data/disk2/LibriTTS")
    pl.add_argument("--splits", default="train-clean-100,train-clean-360,dev-clean")
    pl.add_argument("--out_dir", default="/mnt/data/disk2/yejin/LiveVoice/features/phonemes")
    pl.add_argument("--num_workers", type=int, default=8)
    pl.add_argument("--no_skip_existing", action="store_true")
    pl.add_argument("--shard", type=int, default=0,
                     help="This process's shard index in [0, num_shards). All shards "
                          "write to the same out_dir; the cache merges automatically.")
    pl.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]
    print(f"[extract-phonemes] discovering LibriTTS ({splits}) at {args.libritts_path} ...")
    items = discover_libritts(args.libritts_path, splits)

    if args.num_shards > 1:
        if not (0 <= args.shard < args.num_shards):
            raise SystemExit(f"--shard must be in [0,{args.num_shards})")
        total = len(items)
        items = items[args.shard :: args.num_shards]
        print(f"[extract-phonemes] shard {args.shard}/{args.num_shards}: {len(items)}/{total} utterances")

    out_root = Path(args.out_dir) / "libritts"
    skip_existing = not args.no_skip_existing

    jobs = []
    n_skipped = 0
    for txt_path, spk, utt_id in items:
        save_path = out_root / spk / f"{utt_id}.pt"
        if skip_existing and save_path.exists():
            n_skipped += 1
            continue
        jobs.append((txt_path, spk, utt_id, save_path))
    print(f"[extract-phonemes] {len(jobs)} to process ({n_skipped} already done) → {out_root}")

    errors = []
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for err in tqdm(pool.imap_unordered(_process_one, jobs, chunksize=64),
                             total=len(jobs), desc="phonemizing"):
                if err:
                    errors.append(err)
    else:
        for job in tqdm(jobs, desc="phonemizing"):
            err = _process_one(job)
            if err:
                errors.append(err)

    if errors:
        print(f"[extract-phonemes] {len(errors)} failures (showing up to 10):")
        for e in errors[:10]:
            print(f"  [warn] {e}")
    print("[extract-phonemes] Done.")


if __name__ == "__main__":
    main()
