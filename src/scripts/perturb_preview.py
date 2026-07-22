#!/usr/bin/env python3
"""Listen to source-side perturbation via the real ContentPerturbation module.

For every origin.wav under --in_dir, writes next to it (all through `perturb()`,
i.e. exactly what training applies):
    pitch.wav          pitch shift only            (formant off)
    formant.wav        formant (VTLN) shift only   (pitch off)
    pitch_formant.wav  pitch + formant             (both)
EQ (4-band) is ON by default (config value), matching the training cache; pass
--eq_gain_db 0 to hear pitch/formant in isolation.

Formant = Praat 'Change gender' (pitch/timing preserved). FIXED magnitude
`perturb_formant_ratio_range` (or --formant_range), random direction → ratio is
exactly 1-range or 1+range (e.g. 0.25 → 0.75 or 1.25) per call.

    python scripts/perturb_preview.py --in_dir output/perturbed
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import soundfile as sf

from livevoice.config import LiveVoiceConfig
from livevoice.model.content_perturbation import ContentPerturbation


def load_mono(path: str, sr: int) -> torch.Tensor:
    wav, s = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(wav).mean(dim=1)
    if s != sr:
        x = torchaudio.functional.resample(x, s, sr)
    return x / (x.abs().max() + 1e-8)


def perturb(x: torch.Tensor, sr: int, pitch: float, use_vtln: bool,
            formant_range: float, seed: int, eq_gain_db: float = 0.0) -> torch.Tensor:
    cfg = LiveVoiceConfig(
        sample_rate=sr,
        perturb_pitch_semitones=pitch,
        use_vtln=use_vtln,
        perturb_formant_ratio_range=formant_range,
        perturb_eq_gain_db=eq_gain_db,
        perturb_prob=1.0,
    )
    pert = ContentPerturbation(cfg)
    pert.train()  # perturbation only applies in train mode
    random.seed(seed)  # same seed → same pitch draw across the outputs
    return pert(x.unsqueeze(0)).squeeze(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="output/perturbed")
    p.add_argument("--sr", type=int, default=16000, help="Perturb at this rate (sw2v uses 16 kHz).")
    p.add_argument("--pitch", type=float, default=None, help="Pitch ± semitones (default: config).")
    p.add_argument("--formant_range", type=float, default=None,
                   help="Formant fixed magnitude, random dir (default: config). ratio = 1±this.")
    p.add_argument("--eq_gain_db", type=float, default=None, help="EQ ± gain dB (default: config; 0 = off).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dflt = LiveVoiceConfig(sample_rate=args.sr)
    pitch = args.pitch if args.pitch is not None else float(dflt.perturb_pitch_semitones)
    formant_range = args.formant_range if args.formant_range is not None else float(dflt.perturb_formant_ratio_range)
    eq_gain_db = args.eq_gain_db if args.eq_gain_db is not None else float(dflt.perturb_eq_gain_db)
    print(f"[preview] pitch=±{pitch} semitones  formant=1±{formant_range} (fixed mag, random dir)  "
          f"eq=±{eq_gain_db}dB  sr={args.sr}  seed={args.seed}")

    origins = sorted(Path(args.in_dir).glob("**/origin.wav"))
    if not origins:
        raise SystemExit(f"no origin.wav under {args.in_dir}")

    for og in origins:
        x = load_mono(str(og), args.sr)
        variants = {
            "pitch.wav":         dict(pitch=pitch, use_vtln=False),   # pitch only
            "formant.wav":       dict(pitch=0.0,  use_vtln=True),     # formant only
            "pitch_formant.wav": dict(pitch=pitch, use_vtln=True),    # both
        }
        for name, kw in variants.items():
            y = perturb(x, args.sr, formant_range=formant_range, seed=args.seed,
                        eq_gain_db=eq_gain_db, **kw)
            sf.write(str(og.parent / name), y.numpy(), args.sr)
        print(f"[preview] {og.parent}: wrote {', '.join(variants)}")


if __name__ == "__main__":
    main()
