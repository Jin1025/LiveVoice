"""Convert Expresso into the LibriTTS directory layout so LiveVoiceDataset can read it.

Why Expresso. Measured on IEMOCAP_dev, our model renders 19.1% of audible source frames as
silence (25.9% for sad) against 4.3-5.4% on read LibriSpeech, and a codec round-trip drops
0.0% at UAR 68.4 -- so the codec keeps non-verbal sound and the AR decoder does not produce
it. It cannot: LibriTTS is read audiobook speech with almost no laughter, sighs or breaths,
so the decoder has never been asked to make one. Expresso has `laughing` (3.3 h),
`nonverbal` (0.5 h), `sad`, `whisper` and other low-arousal styles.

The conditioning path for it already exists. On Expresso laughter the content BNF barely
separates laughter from speech (Fisher ratio 0.0996) while the MPM latent does (0.2761), so
the decoder can be taught "content is empty but MPM says there is energy -> emit a
non-verbal vocalisation". What is missing is training data where that pattern occurs.

Output layout (see livevoice/data/libritts_dataset.py -- speaker is parts[-3]):

    <out>/<split>/<speaker>/<read|conv>-<style>/<speaker>_<style>_<seq>.wav   16 kHz mono

The "<read|conv>-<style>" level matches textlesslib's own layout and lands where the loader
expects the chapter id, so speaker grouping stays correct and styles remain easy to filter
or weight.

Deliberate difference from textlesslib: VAD segments closer together than --merge_gap are
joined rather than cut apart. Breaths and the in-drawn pause before a laugh live in those
gaps and are the whole reason for using this corpus.

    python src/scripts/prepare_expresso.py \
        --expresso /mnt/data/disk3/yejin/expresso \
        --out /mnt/data/disk2/yejin/LiveVoice/data/expresso16k
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

SR = 16000

# The dialogues are stereo with one actor per channel, and the directory name carries the
# style of each channel: "laughing" (both channels) or "angry-sad" (channel1-channel2).
STYLE_PAIR = re.compile(r"^([a-z_]+)-([a-z_]+)$")


def parse_vad(path: Path) -> dict[str, list[tuple[float, float]]]:
    """"{stem}/{channel}" -> [(start, end), ...] from VAD_segments.txt."""
    out: dict[str, list[tuple[float, float]]] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            key, segs = line.rstrip("\n").split("\t", 1)
            out[key] = [(float(a), float(b)) for a, b in
                        re.findall(r"\(([\d.]+),\s*([\d.]+)\)", segs)]
    return out


def parse_split(path: Path) -> dict[str, tuple[float | None, float | None]]:
    """id -> (start, end), either side None meaning "from the beginning" / "to the end".

    Same parse as textlesslib's create_short_segments_dataset.read_split_file: split on the
    comma and treat an empty side as None. Both "(60.0s,)" and "(,60.0s)" occur, and reading
    the numbers out with a regex instead loses which side is missing.
    """
    out: dict[str, tuple[float | None, float | None]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stem, *seg = line.split("\t")
            if not seg:
                out[stem] = (None, None)
                continue
            rng = seg[0].strip()
            assert rng[0] == "(" and rng[-1] == ")", f"bad range in {path}: {line!r}"
            a, b = rng[1:-1].split(",")
            out[stem] = (None if not a.strip() else float(a.strip().rstrip("s")),
                         None if not b.strip() else float(b.strip().rstrip("s")))
    return out


def index_audio(root: Path) -> dict[str, Path]:
    return {p.stem: p for p in root.rglob("*.wav")}


def channel_meta(stem: str) -> list[tuple[int, str, str]]:
    """[(channel_index, speaker, substyle), ...] for one file, derived from the STEM.

    textlesslib reads speaker and style out of the filename, not the directory tree, and
    keys the output on "<read|conv>-<style>" so that read and conversational material of the
    same style stay apart. Worth keeping: the non-verbal sound we are after is concentrated
    in the dialogues, so the two are not interchangeable when weighting the mix.
    """
    head, style_field = stem.split("_")[0], stem.split("_")[1]
    spks = head.split("-")
    if len(spks) == 2:                                    # conversational, stereo
        styles = style_field.split("-")
        return [(ch, spks[ch], "conv-" + (styles[0] if len(styles) == 1 else styles[ch]))
                for ch in (0, 1)]
    return [(0, head, "read-" + style_field)]             # read speech, mono


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expresso", default="/mnt/data/disk3/yejin/expresso")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--min_sec", type=float, default=6.5,
                    help="drop chunks shorter than this. The VC model pairs two "
                         "audio_duration windows from ONE utterance (reference + source), "
                         "so anything under 2x3s + gap is unusable and only slows the loader")
    ap.add_argument("--max_sec", type=float, default=15.0,
                    help="split longer runs of speech into chunks of at most this")
    ap.add_argument("--merge_gap", type=float, default=0.4,
                    help="join VAD segments separated by less than this. Breaths and the "
                         "in-drawn pause before a laugh sit in these gaps, and they are "
                         "exactly what we are here to keep, so do not cut them out")
    ap.add_argument("--styles", default="",
                    help="comma-separated allowlist of bare styles, e.g. "
                         "laughing,nonverbal,sad,whisper (matches both read- and conv-); "
                         "empty = every style")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.expresso), Path(args.out)
    vad = parse_vad(root / "VAD_segments.txt")
    audio_idx = index_audio(root / "audio_48khz")
    keep = {s.strip() for s in args.styles.split(",") if s.strip()}
    print(f"[expresso] {len(audio_idx)} wav files, {len(vad)} VAD entries"
          + (f", styles={sorted(keep)}" if keep else ", all styles"))

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ids = parse_split(root / "splits" / f"{split}.txt")
        items = list(ids.items())[: args.limit] if args.limit else list(ids.items())
        n_written, n_sec = 0, 0.0
        per_style: dict[str, float] = defaultdict(float)
        seq: dict[tuple[str, str], int] = defaultdict(int)
        missing = 0

        for stem, (t0, t1) in tqdm(items, desc=f"{split}"):
            wav = audio_idx.get(stem)
            if wav is None:
                missing += 1
                continue
            info = sf.info(str(wav))
            for ch, spk, style in channel_meta(stem):
                if ch >= info.channels:
                    continue
                # --styles is matched on the bare style, so "laughing" selects both
                # conv-laughing and read-laughing without the caller spelling out each.
                if keep and style.split("-", 1)[1] not in keep:
                    continue
                # Fall back to the whole file when VAD has no entry (short read speech).
                segs = vad.get(f"{stem}/channel{ch + 1}") or [(0.0, info.duration)]
                # Honour the split's own time range so dev/test material never leaks in.
                lo = t0 if t0 is not None else 0.0
                hi = t1 if t1 is not None else info.duration
                segs = [(max(a, lo), min(b, hi)) for a, b in segs]
                segs = [(a, b) for a, b in segs if b - a > 0.05]
                if not segs:
                    continue

                merged: list[list[float]] = []
                for a, b in sorted(segs):
                    if merged and a - merged[-1][1] <= args.merge_gap:
                        merged[-1][1] = b
                    else:
                        merged.append([a, b])

                y = None
                for a, b in merged:
                    if b - a < args.min_sec:
                        continue
                    if y is None:
                        y, sr = sf.read(str(wav), dtype="float32", always_2d=True)
                        y = y[:, ch]
                        if sr != SR:
                            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                            sr = SR
                    n_chunk = max(1, int(np.ceil((b - a) / args.max_sec)))
                    step = (b - a) / n_chunk
                    for k in range(n_chunk):
                        cs, ce = a + k * step, a + (k + 1) * step
                        if ce - cs < args.min_sec:
                            continue
                        clip = y[int(cs * SR): int(ce * SR)]
                        if clip.size < int(args.min_sec * SR):
                            continue
                        d = out / split / spk / style
                        d.mkdir(parents=True, exist_ok=True)
                        seq[(spk, style)] += 1
                        sf.write(str(d / f"{spk}_{style}_{seq[(spk, style)]:05d}.wav"),
                                 clip, SR, subtype="PCM_16")
                        n_written += 1
                        n_sec += clip.size / SR
                        per_style[style] += clip.size / SR

        print(f"[expresso] {split}: {n_written} clips, {n_sec / 3600:.2f} h"
              + (f"  ({missing} ids had no audio)" if missing else ""))
        for s, sec in sorted(per_style.items(), key=lambda kv: -kv[1])[:12]:
            print(f"           {s:<20} {sec / 60:7.1f} min")


if __name__ == "__main__":
    main()
