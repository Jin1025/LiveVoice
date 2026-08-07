"""What is the s-sim CEILING, measured with matched durations?

Two things this answers.

1. The val metric is measured with MISMATCHED durations. `lightning/module.py::_score`
   trims the reference to `audio_duration` but embeds the source at full length, so
   `val/spk_sim_cross` (vs a 3-4s clip) and `val/spk_sim_cross_src` (vs the full utterance)
   are not on the same scale, and neither is `val/spk_sim_gt`. Everything here uses two
   windows of the SAME length so the numbers are comparable.

2. `pairing="same_utterance_continuation"` trains with prompt and target taken from ONE
   recording, but inference uses a reference from a DIFFERENT utterance. Those two
   situations have different ceilings, and the gap between them is exactly the size of the
   "copy the channel instead of the speaker" shortcut. Measuring both bounds it.

Conditions (all symmetric, --window_sec each):
  same_utt_diff_window : two NON-OVERLAPPING windows of one utterance  (training ceiling)
  same_spk_diff_utt    : windows from two utterances of the same speaker (inference ceiling)
  diff_spk             : windows from two different speakers            (floor)
  val_style            : FULL utterance vs a window, same speaker — what val reports today,
                         included to quantify the bias it carries

    conda run -n sound python /workspace/LiveVoice/src/debug/diag_spk_sim_protocol.py \
        --n_trials 200 --window_sec 3.0 --device cuda
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from livevoice.config import LiveVoiceConfig


def _mean(xs):
    return sum(xs) / max(1, len(xs))


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libritts", default="/mnt/data/disk2/LibriTTS/dev-clean")
    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--window_sec", type=float, default=3.0,
                    help="window length for ALL symmetric comparisons (= config.audio_duration)")
    ap.add_argument("--encoder", default="wavlm_tdnn", choices=["wavlm_tdnn", "ecapa"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    import soundfile as sf
    import librosa

    sr = 16000
    W = int(args.window_sec * sr)

    cfg = LiveVoiceConfig(val_spk_encoder=args.encoder, sample_rate=sr)
    if args.encoder == "wavlm_tdnn":
        from livevoice.model.wavlm_speaker_encoder import WavLMTDNNSpeakerEncoder
        enc = WavLMTDNNSpeakerEncoder(cfg)
    else:
        from livevoice.model.speechbrain_speaker_encoder import SpeechBrainECAPASpeakerEncoder
        enc = SpeechBrainECAPASpeakerEncoder(cfg)
    device = torch.device(args.device)
    enc = enc.eval().to(device)
    print(f"[proto] encoder={args.encoder} device={device} window={args.window_sec}s")

    # index utterances; note which are long enough to yield TWO disjoint windows
    by_spk_long, by_spk_any = defaultdict(list), defaultdict(list)
    for root, _d, files in os.walk(args.libritts):
        for fn in files:
            if not fn.endswith(".wav"):
                continue
            p = os.path.join(root, fn)
            try:
                info = sf.info(p)
                dur = info.frames / info.samplerate
            except Exception:
                continue
            spk = p[len(args.libritts):].strip("/").split("/")[0]
            if dur >= args.window_sec:
                by_spk_any[spk].append(p)
            if dur >= 2 * args.window_sec:
                by_spk_long[spk].append(p)

    spks = sorted(s for s in by_spk_long if len(by_spk_any[s]) >= 2)
    print(f"[proto] {len(spks)} speakers with a >= {2*args.window_sec:g}s utterance "
          f"and >= 2 utterances >= {args.window_sec:g}s")
    if len(spks) < 2:
        raise SystemExit("[proto] not enough data")

    cache: dict[str, torch.Tensor] = {}

    def load(p):
        if p not in cache:
            y, s = sf.read(p, dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            if s != sr:
                y = librosa.resample(y, orig_sr=s, target_sr=sr)
            cache[p] = torch.from_numpy(y)
        return cache[p]

    @torch.no_grad()
    def embed(wav):
        return enc(wav.unsqueeze(0).to(device).float())

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=-1).mean().item()

    rng = random.Random(args.seed)
    res = defaultdict(list)

    for i in range(args.n_trials):
        spk = rng.choice(spks)
        u1 = rng.choice(by_spk_long[spk])                      # long: two windows fit
        others = [p for p in by_spk_any[spk] if p != u1]
        if not others:
            continue
        u2 = rng.choice(others)
        spk_b = rng.choice([s for s in spks if s != spk])
        u3 = rng.choice(by_spk_any[spk_b])

        w1 = load(u1)
        # two NON-OVERLAPPING windows of u1
        s1 = rng.randint(0, w1.numel() - 2 * W)
        a1 = w1[s1: s1 + W]
        s2 = rng.randint(s1 + W, w1.numel() - W)
        a2 = w1[s2: s2 + W]

        w2, w3 = load(u2), load(u3)
        b1 = w2[: W] if w2.numel() >= W else w2
        c1 = w3[: W] if w3.numel() >= W else w3

        e_a1, e_a2, e_b1, e_c1 = embed(a1), embed(a2), embed(b1), embed(c1)
        e_full = embed(w1)

        res["same_utt_diff_window"].append(cos(e_a1, e_a2))
        res["same_spk_diff_utt"].append(cos(e_a1, e_b1))
        res["diff_spk"].append(cos(e_a1, e_c1))
        res["val_style_full_vs_window"].append(cos(e_full, e_b1))

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{args.n_trials}")

    print(f"\n=========== S-SIM CEILINGS ({args.window_sec:g}s windows, {args.encoder}) ===========")
    order = ["diff_spk", "same_spk_diff_utt", "same_utt_diff_window", "val_style_full_vs_window"]
    label = {
        "diff_spk":                 "different speaker            (FLOOR)",
        "same_spk_diff_utt":        "same spk, different utterance (INFERENCE ceiling)",
        "same_utt_diff_window":     "same utterance, other window  (TRAINING ceiling)",
        "val_style_full_vs_window": "full vs window, same spk      (what val reports)",
    }
    for k in order:
        if res[k]:
            print(f"{label[k]:52} {_mean(res[k]):.4f} ± {_std(res[k]):.3f}")
    print("-" * 74)
    if res["same_utt_diff_window"] and res["same_spk_diff_utt"]:
        gap = _mean(res["same_utt_diff_window"]) - _mean(res["same_spk_diff_utt"])
        print(f"CHANNEL/SESSION COMPONENT = training ceiling - inference ceiling = {gap:+.4f}")
        print("  same_utterance_continuation can reach the TRAINING ceiling by copying the")
        print("  recording session; only the INFERENCE ceiling is reachable at test time, so")
        print("  this gap is how much a channel-copying shortcut would flatter training.")
    if res["same_spk_diff_utt"] and res["val_style_full_vs_window"]:
        d = _mean(res["val_style_full_vs_window"]) - _mean(res["same_spk_diff_utt"])
        print(f"DURATION BIAS in the val protocol (full-vs-window − window-vs-window) = {d:+.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
