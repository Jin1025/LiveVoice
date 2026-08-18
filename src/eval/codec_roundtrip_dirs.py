"""Codec round-trip of the VPC IEMOCAP dirs — the emotion ceiling, with no conversion at all.

StreamVoiceAnon+ argues that codec-LM anonymization loses emotion partly to *VQ information
loss*: whatever the quantizer discards can never be recovered downstream, no matter how the LM
is conditioned. If that holds for jhcodec, then prosody conditioning, emotion distillation and
every other decoder-side idea are all working under a hard cap.

This measures the cap directly. Encode each IEMOCAP utterance with jhcodec and decode it
straight back — same speaker, same words, same timing, only the quantizer in between — then let
the official VPC SER score it. Three readings:

    UAR ~ 69/71 (original)   the codec preserves emotion; our 42.7 is the model's to fix
    UAR ~ 45-50              the codec already halves it; conditioning cannot get past that
    ACC_sad collapses here   sadness (breathy, creaky, low-energy) does not survive 8
                             codebooks at 50 fps -- a codec problem, not a conditioning one

Costs one codec pass over 5,531 utterances and one SER run. No training, no generation, no
sampling: the AR model is never loaded.

    conda run -n sound python src/eval/codec_roundtrip_dirs.py --suffix _codecrt
    cd <vpc> && python run_evaluation.py --config configs/track1/eval_pre_seronly.yaml \
        --overwrite '{"anon_data_suffix": "_codecrt"}'
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
import torch
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.model import build_codec

DEFAULT_VPC_ROOT = os.environ.get("VPC_ROOT", "/mnt/data/disk3/yejin/VPC")
STAMP = ".livevoice_anon.json"


def _read_scp(p: Path) -> dict[str, str]:
    return {l.split()[0]: l.split(None, 1)[1].strip() for l in open(p) if l.strip()}


def _load(entry: str, root: Path, sr: int) -> torch.Tensor:
    e = entry.strip()
    if e.endswith("|"):                       # Kaldi pipe: decode the file directly
        for tok in e[:-1].split():
            if tok.lower().endswith((".flac", ".wav", ".ogg", ".opus", ".mp3")):
                e = tok
                break
    p = Path(e)
    y, s = sf.read(str(p if p.is_absolute() else root / p), dtype="float32", always_2d=True)
    w = torch.from_numpy(y).float().mean(dim=1)
    if s != sr:
        import torchaudio
        w = torchaudio.functional.resample(w, s, sr)
    return w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vpc_root", default=DEFAULT_VPC_ROOT)
    p.add_argument("--datasets", default="IEMOCAP_dev,IEMOCAP_test")
    p.add_argument("--suffix", default="_codecrt")
    p.add_argument("--n_codebooks", type=int, default=None,
                   help="quantize with fewer codebooks than the model uses, to see how the "
                        "ceiling moves with rate; default = config.jhcodec_n_codebooks")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    root = Path(args.vpc_root).resolve()
    dev = torch.device("cpu" if args.cpu else "cuda")
    cfg = LiveVoiceConfig()
    codec = build_codec(cfg).to(dev).eval()
    K = int(args.n_codebooks or getattr(cfg, "jhcodec_n_codebooks", 8))
    sr = int(getattr(cfg, "jhcodec_sample_rate", 16000))
    print(f"[rt] {cfg.codec}  {K} codebooks  {sr} Hz  device={dev}")

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        src, dst = root / "data" / ds, root / "data" / f"{ds}{args.suffix}"
        if not (src / "wav.scp").is_file():
            print(f"[rt] skip {ds}: no wav.scp")
            continue
        (dst / "wav").mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(src / "*")):
            if os.path.isfile(f) and os.path.basename(f) != "wav.scp":
                shutil.copy(f, dst)

        scp = _read_scp(src / "wav.scp")
        n_fail = 0
        for utt, entry in tqdm(scp.items(), desc=f"{ds}{args.suffix}"):
            try:
                w = _load(entry, root, sr).to(dev)
                with torch.no_grad():
                    codes = codec.encode(w.unsqueeze(0))[:, :K, :]
                    out = codec.decode(codes)[0].detach().float().cpu()
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  [warn] {utt}: {type(e).__name__}: {e}")
                continue
            tmp = dst / "wav" / f".{utt}.partial.wav"
            sf.write(str(tmp), out.numpy(), sr, subtype="PCM_16")
            os.replace(tmp, dst / "wav" / f"{utt}.wav")

        done = sorted(u for u in scp if (dst / "wav" / f"{u}.wav").is_file())
        with open(dst / "wav.scp", "w", encoding="utf-8") as f:
            for u in done:
                f.write(f"{u} data/{ds}{args.suffix}/wav/{u}.wav\n")
        (dst / STAMP).write_text(json.dumps(
            {"codec_roundtrip": True, "codec": cfg.codec, "n_codebooks": K,
             "source": ds, "sample_rate": sr}, indent=2, sort_keys=True))
        print(f"[rt] {ds}{args.suffix}: {len(done)}/{len(scp)} written"
              + (f", {n_fail} failed" if n_fail else ""))

    print(f"\n[rt] then, from {root}:\n"
          f"  python run_evaluation.py --config configs/track1/eval_pre_seronly.yaml "
          f"--overwrite '{{\"anon_data_suffix\": \"{args.suffix}\"}}'")


if __name__ == "__main__":
    main()
