"""Control for diag_speaker_control: is high gen-to-gen similarity real, or a codec artifact?

diag_speaker_control found that all generated samples sit at cos ~0.42 with EACH OTHER
while being only 0.22 from their reference and 0.31 from their source — which looks like
collapse onto one generic voice. But every generated sample also passes through the same
jhcodec decoder, and that shared synthesis channel could inflate gen-gen similarity on its
own, with no identity collapse at all.

This isolates the channel: take REAL utterances from DIFFERENT speakers, run them through
codec encode -> (first K codebooks) -> decode, and measure how similar the reconstructions
are to each other. No model involved.

    pairwise cos among ORIGINALS      (different speakers) — the true floor, ~0.08
    pairwise cos among RECONSTRUCTIONS(different speakers) — the CODEC floor
    cos(recon_i, orig_i)                                   — identity preserved by the codec

Reading:
  codec floor ~= original floor  -> the codec adds no common signature; gen-gen 0.42 is
                                    REAL speaker collapse.
  codec floor ~= 0.4             -> the 0.42 is largely the synthesis channel and the
                                    collapse reading must be withdrawn.

    conda run -n sound python /workspace/LiveVoice/src/debug/diag_codec_channel_floor.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from livevoice.config import LiveVoiceConfig
from livevoice.model import build_codec


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
    ap.add_argument("--n_utts", type=int, default=24,
                    help="one utterance each from N different speakers")
    ap.add_argument("--n_codebooks", type=int, default=8,
                    help="keep only the first K codebooks, matching the model's output")
    ap.add_argument("--min_sec", type=float, default=4.0)
    ap.add_argument("--max_sec", type=float, default=10.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    import soundfile as sf
    import librosa

    sr = 16000
    cfg = LiveVoiceConfig(codec="jhcodec", sample_rate=sr, val_spk_encoder="wavlm_tdnn",
                          n_codebooks_predict=args.n_codebooks)
    device = torch.device(args.device)
    codec = build_codec(cfg).to(device).eval()

    from livevoice.model.wavlm_speaker_encoder import WavLMTDNNSpeakerEncoder
    spk_enc = WavLMTDNNSpeakerEncoder(cfg).eval().to(device)

    by_spk = defaultdict(list)
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
            if args.min_sec <= dur <= args.max_sec:
                by_spk[p[len(args.libritts):].strip("/").split("/")[0]].append(p)

    spks = sorted(k for k, v in by_spk.items() if v)
    rng = random.Random(args.seed)
    rng.shuffle(spks)
    spks = spks[: args.n_utts]
    paths = [rng.choice(by_spk[s]) for s in spks]
    print(f"[chan] {len(paths)} utterances from {len(paths)} distinct speakers, "
          f"K={args.n_codebooks} codebooks")

    def load(p):
        y, s = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        if s != sr:
            y = librosa.resample(y, orig_sr=s, target_sr=sr)
        return torch.from_numpy(y).unsqueeze(0).to(device)

    orig_emb, recon_emb, self_sims = [], [], []
    with torch.no_grad():
        for i, p in enumerate(paths):
            wav = load(p)
            codes = codec.encode(wav)[:, : args.n_codebooks, :]
            recon = codec.decode(codes)
            eo = spk_enc(wav.float())
            er = spk_enc(recon.float())
            orig_emb.append(eo)
            recon_emb.append(er)
            self_sims.append(F.cosine_similarity(eo, er, dim=-1).mean().item())
            if (i + 1) % 8 == 0:
                print(f"  ... {i + 1}/{len(paths)}")

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=-1).mean().item()

    pairs = list(combinations(range(len(paths)), 2))
    o_o = [cos(orig_emb[i], orig_emb[j]) for i, j in pairs]
    r_r = [cos(recon_emb[i], recon_emb[j]) for i, j in pairs]
    o_r = [cos(orig_emb[i], recon_emb[j]) for i, j in pairs]

    print("\n============== CODEC CHANNEL FLOOR (different speakers) ==============")
    print(f"pairwise cos among ORIGINALS       : {_mean(o_o):.4f} ± {_std(o_o):.3f}")
    print(f"pairwise cos among RECONSTRUCTIONS : {_mean(r_r):.4f} ± {_std(r_r):.3f}")
    print(f"pairwise cos original vs recon     : {_mean(o_r):.4f} ± {_std(o_r):.3f}")
    print(f"cos(recon_i, orig_i)  same utt     : {_mean(self_sims):.4f} ± {_std(self_sims):.3f}")
    print("-" * 70)
    inflation = _mean(r_r) - _mean(o_o)
    print(f"CHANNEL INFLATION = recon_floor - orig_floor = {inflation:+.4f}")
    print(f"  gen-gen similarity measured by diag_speaker_control was 0.4177 / 0.4284.")
    if _mean(r_r) > 0.30:
        print("  >>> The codec itself pushes different speakers this close together —")
        print("      the high gen-gen number is largely the SYNTHESIS CHANNEL, not collapse.")
    elif inflation < 0.10:
        print("  >>> The codec preserves speaker separation, so gen-gen ~0.42 is REAL")
        print("      collapse onto a generic voice, not a channel artifact.")
    else:
        print("  >>> Partial: the codec inflates similarity somewhat; subtract this floor")
        print("      before reading gen-gen as collapse.")
    print("======================================================================")


if __name__ == "__main__":
    main()
