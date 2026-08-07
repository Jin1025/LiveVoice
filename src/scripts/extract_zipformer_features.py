"""Precompute streaming-Zipformer BNF content features for LibriTTS.

Same on-disk contract as the sw2v cache so `LibriTTSDataset._load_feats` works unchanged:

    <out_dir>/<speaker_id>/<utt_id>.pt   ->   {"feats": FloatTensor(T, D) as float16,
                                               "audio_stride": int}

with T == ceil(num_samples / 320), i.e. exactly one frame per jhcodec token (the encoder
front-pads to put its frames on the codec grid — see
`ZipformerContentEncoder.ALIGN_PAD_FRAMES`).

NO PERTURBATION. The sw2v cache needed pitch/formant/EQ perturbation baked in because a
codec's content encoder retains timbre (raw speaker probe: utt 0.983). An ASR encoder does
not: measured on LibriTTS dev-clean, 30 speakers, chance 0.033, Zipformer BNF scores utt
0.306 clean vs 0.289 perturbed — perturbation buys essentially nothing. Dropping it removes
the 28-hour perturbed extraction, the Praat/formant dependency, and the train/inference
mismatch it created (training saw perturbed audio, inference never does).

CMN is NOT baked in either: it stays a runtime setting (`config.content_cmn`) so it can be
toggled without re-extracting. Note the consequence — at runtime the main VC path applies
CMN to the SLICED training window, so a causal running mean restarts at each window, while
the ASR/GRL path and the probe apply it to the full utterance. Pass --cmn to bake the
full-utterance version in instead, but then set config.content_cmn="off" to avoid applying
it twice.

    conda run -n sound python /workspace/LiveVoice/src/scripts/extract_zipformer_features.py \
        --splits train-clean-100,train-clean-360 \
        --out_dir /mnt/data/disk2/yejin/LiveVoice/features/zipformer/libritts \
        --shard 0 --num_shards 4        # one process per GPU, all writing the same dir
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):
        return x

from livevoice.config import LiveVoiceConfig
from livevoice.model.content_supervision import apply_content_cmn
from livevoice.model.zipformer_content import ZipformerContentEncoder

SR = 16000
HOP = 320          # jhcodec: 50 fps at 16 kHz


def load_mono(path: str, target_sr: int) -> torch.Tensor:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    a = torch.from_numpy(y).float().mean(dim=1)
    if sr != target_sr:
        import librosa
        a = torch.from_numpy(
            librosa.resample(a.numpy(), orig_sr=sr, target_sr=target_sr).astype("float32")
        )
    peak = a.abs().max()
    if peak > 1e-8:
        a = a / peak           # peak-normalise, matching extract_sw2v_features.py
    return a


def discover_libritts(libritts_path: str, splits: list[str]) -> list[tuple[str, str, str]]:
    root = Path(libritts_path)
    items = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [warn] split not found: {split_dir}")
            continue
        for wav in sorted(split_dir.glob("**/*.wav")):
            items.append((str(wav), wav.parts[-3], wav.stem))
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libritts_path", default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--splits", default="train-clean-100,train-clean-360,dev-clean")
    p.add_argument("--out_dir",
                   default="/mnt/data/disk2/yejin/LiveVoice/features/zipformer/libritts")
    p.add_argument("--zipformer_ckpt",
                   default="/mnt/data/disk3/yejin/zipformer_pretrained.pt")
    p.add_argument("--zipformer_layer", default="-1",
                   help="-1 = just before the final 50->25Hz downsample (50 fps, full dim); "
                        "0..5 = that stack's output; 'out' = the 25 fps encoder output "
                        "(NOT on the codec grid — needs upsampling downstream)")
    p.add_argument("--align_pad_frames", type=int, default=None,
                   help="front pad in 50fps frames placing encoder frames on the codec grid. "
                        "Given explicitly here (rather than read from config) because a cache "
                        "extracted at one alignment is silently wrong at another. "
                        "0 = no added latency, content ~4 frames stale; -6 = aligned at the "
                        "cost of ~120ms lookahead. Defaults to config.zipformer_align_pad_frames.")
    p.add_argument("--cmn", default="off", choices=["off", "utterance", "causal"],
                   help="bake full-utterance CMN into the cache (then set "
                        "config.content_cmn='off' so it is not applied twice)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_seconds", type=float, default=0.0,
                   help=">0 skips utterances longer than this (memory guard)")
    p.add_argument("--no_skip_existing", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"[extract-zipformer] discovering LibriTTS ({splits}) at {args.libritts_path} ...")
    items = discover_libritts(args.libritts_path, splits)
    if args.num_shards > 1:
        items = items[args.shard :: args.num_shards]
        print(f"[extract-zipformer] shard {args.shard}/{args.num_shards}: {len(items)} files")
    else:
        print(f"[extract-zipformer] {len(items)} files")

    device = torch.device(args.device)
    lyr = args.zipformer_layer
    cfg = LiveVoiceConfig()
    if args.align_pad_frames is not None:
        cfg.zipformer_align_pad_frames = int(args.align_pad_frames)
    enc = ZipformerContentEncoder(
        cfg, args.zipformer_ckpt,
        layer=(lyr if lyr == "out" else int(lyr)),
    ).to(device)
    on_grid = lyr != "out"
    print(f"[extract-zipformer] out_dir={args.out_dir}  cmn={args.cmn}  "
          f"align_pad_frames={cfg.zipformer_align_pad_frames}  perturbation=NONE  "
          f"frames_on_codec_grid={on_grid}")

    out_root = Path(args.out_dir)
    skip_existing = not args.no_skip_existing
    todo = []
    for wav_path, spk, utt_id in items:
        sp = out_root / spk / f"{utt_id}.pt"
        if skip_existing and sp.exists():
            continue
        todo.append((wav_path, spk, utt_id, sp))
    print(f"  {len(todo)} to process ({len(items) - len(todo)} already done)")

    n_ok = n_skip = n_fail = 0
    # one utterance at a time: lengths vary a lot and the encoder is cheap relative to I/O,
    # so batching would mean padding, and padding changes the frame count we just aligned.
    for wav_path, spk, utt_id, save_path in tqdm(todo, desc="extracting"):
        try:
            a = load_mono(wav_path, SR)
            if args.max_seconds > 0 and a.numel() > args.max_seconds * SR:
                n_skip += 1
                continue
            with torch.no_grad():
                f = enc(a.unsqueeze(0).to(device))          # (1, T, D), codec-aligned
            if args.cmn != "off":
                f = apply_content_cmn(f, args.cmn, False)
            f = f.squeeze(0).float().cpu()
            want = -(-a.numel() // HOP)
            if on_grid and f.shape[0] != want:
                raise RuntimeError(f"frame count {f.shape[0]} != ceil(len/{HOP})={want}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"feats": f.half(), "audio_stride": HOP}, save_path)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 10:
                print(f"  [warn] failed {wav_path}: {e}")

    print(f"[extract-zipformer] done: {n_ok} written, {n_skip} skipped (too long), "
          f"{n_fail} failed")
    if n_fail:
        print("  NOTE a nonzero failure count leaves holes in the cache; _load_feats returns "
              "None for a missing file, and collate_fn then nulls the key for the whole "
              "batch — rerun to fill them in before training.")


if __name__ == "__main__":
    main()
