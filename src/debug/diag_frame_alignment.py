"""Which audio samples does content frame t actually see? Measured, not derived.

A 40-sample centre offset between HuBERT and jhcodec once pinned WER at ~0.3 for weeks;
it only moved after the frames were aligned 1:1. Any new content encoder has to clear the
same bar, and the safest reference is the encoder that already works (sw2v), not a
hand-derived receptive-field formula.

Method: feed silence, then silence with a single impulse at sample s, and diff the
features. For a CAUSAL encoder the impulse can only affect frames at or after the one
covering s, so the FIRST changed frame gives the mapping sample -> frame. Sweeping s and
fitting a line recovers (samples_per_frame, offset) for each encoder; the difference in
offsets is the padding needed to make frame t of one encoder describe the same audio as
frame t of the other.

    conda run -n sound python /workspace/LiveVoice/src/debug/diag_frame_alignment.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from livevoice.config import LiveVoiceConfig


def onset_map(encode_fn, sr: int, dur_s: float, positions, thresh: float = 1e-4):
    """For each impulse position, the index of the FIRST frame whose feature changes."""
    n = int(dur_s * sr)
    base_wav = torch.zeros(1, n)
    base = encode_fn(base_wav)
    out = []
    for s in positions:
        w = base_wav.clone()
        w[0, s] = 1.0
        f = encode_fn(w)
        T = min(f.shape[1], base.shape[1])
        d = (f[:, :T] - base[:, :T]).abs().amax(dim=-1)[0]      # (T,)
        hit = (d > thresh * max(1.0, d.max().item())).nonzero()
        out.append((s, int(hit[0]) if hit.numel() else None, int(f.shape[1])))
    return out


def fit(pairs):
    """Least-squares slope/intercept of frame index vs impulse sample."""
    pts = [(s, f) for s, f, _ in pairs if f is not None]
    if len(pts) < 2:
        return None, None, 0
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None, None, n
    slope = (n * sxy - sx * sy) / denom          # frames per sample
    intercept = (sy - slope * sx) / n
    return slope, intercept, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zipformer_ckpt", default="/mnt/data/disk3/yejin/zipformer_pretrained.pt")
    ap.add_argument("--zipformer_layer", default="-1")
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sr = 16000
    hop = 320                      # jhcodec: 50 fps at 16 kHz
    device = torch.device(args.device)
    config = LiveVoiceConfig()

    # impulses on the codec grid, away from the edges where padding dominates
    positions = [hop * k for k in range(20, int(args.dur * 50) - 20, 7)]

    encoders = {}

    from livevoice.model.sw2v_content import Sw2vContentEncoder
    sw2v = Sw2vContentEncoder(config).to(device).eval()
    encoders["sw2v"] = lambda w: sw2v(w.to(device)).detach().float().cpu()

    from livevoice.model.zipformer_content import ZipformerContentEncoder
    lyr = args.zipformer_layer
    zf = ZipformerContentEncoder(config, args.zipformer_ckpt,
                                 layer=(lyr if lyr == "out" else int(lyr))).to(device)
    encoders["zipformer"] = lambda w: zf(w.to(device)).detach().float().cpu()

    n_codec = -(-int(args.dur * sr) // hop)      # ceil, what jhcodec produces
    print(f"[align] {args.dur}s audio → jhcodec would emit {n_codec} tokens "
          f"(hop={hop}, {sr/hop:.0f} fps)\n")

    results = {}
    for name, fn in encoders.items():
        with torch.no_grad():
            pairs = onset_map(fn, sr, args.dur, positions)
        slope, intercept, n = fit(pairs)
        if slope is None or slope == 0:
            print(f"[align] {name}: could not fit ({n} usable impulses)")
            continue
        spf = 1.0 / slope                        # samples per frame
        # frame f first sees sample s0(f) = (f - intercept)/slope
        n_frames = pairs[0][2]
        results[name] = (spf, intercept, n_frames)
        print(f"[align] {name:10s} frames={n_frames:4d}  samples/frame={spf:8.2f}  "
              f"frame(s) = {slope:.6f}·s {intercept:+.3f}   [{n} impulses]")

    print()
    if "sw2v" in results and "zipformer" in results:
        spf_a, b_a, n_a = results["sw2v"]
        spf_z, b_z, n_z = results["zipformer"]
        # sample at which each encoder's frame 0 begins to respond
        s0_a = -b_a * spf_a
        s0_z = -b_z * spf_z
        print(f"[align] frame 0 first responds at sample:  sw2v {s0_a:+.1f}   "
              f"zipformer {s0_z:+.1f}")
        print(f"[align] OFFSET (zipformer − sw2v) = {s0_z - s0_a:+.1f} samples "
              f"= {(s0_z - s0_a)/spf_a:+.2f} frames")
        print(f"[align] FRAME COUNT  sw2v {n_a}   zipformer {n_z}   jhcodec {n_codec}")
        print()
        pad_front = int(round(s0_a - s0_z))
        print(f"→ front-pad the audio by {pad_front:+d} samples before the zipformer, then")
        print(f"  pad/truncate the tail so the count reaches {n_codec} "
              f"(currently {n_z - n_codec:+d}).")
        print("  Verify by re-running: the offset should read ~0 and counts should match.")


if __name__ == "__main__":
    main()
