"""Inference / generation script for LiveVoice.

Supports both unconditional generation and VC inference.

Usage examples:
    # Unconditional generation
    python src/scripts/generate.py uncond \
        --ckpt output/checkpoints/uncond_vctk/last.ckpt \
        --n_samples 4 --max_steps 200 --out_dir output/samples

    # VC inference
    CUDA_VISIBLE_DEVICES=3 python scripts/generate.py vc \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/perturb_film/epoch=020-val/loss=3.9469.ckpt \
        --reference /mnt/data/disk2/VCTK-Corpus/wav48/p335/p335_002.wav \
        --content /mnt/data/disk2/VCTK-Corpus/wav48/p225/p225_001.wav \
        --out_dir output/perturb_film
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import soundfile as sf
import librosa

from livevoice.config import LiveVoiceConfig
from livevoice.model import DACModel, HuBERTContentExtractor, LiveVoiceModel, UnconditionalModel
from livevoice.lightning import UnconditionalLightningModule, LiveVoiceLightningModule


# ─────────────────────────────────────────────
#  Audio I/O helpers
# ─────────────────────────────────────────────

def load_audio(path: str, target_sr: int, duration: float | None = None) -> torch.Tensor:
    """Load mono audio at target_sr, optionally truncated to `duration` seconds."""
    try:
        audio_np, sr = librosa.load(path, sr=target_sr, mono=True, duration=duration)
        audio = torch.from_numpy(audio_np.astype("float32"))
    except Exception:
        audio, sr = torchaudio.load(path)
        audio = audio.mean(0)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        if duration is not None:
            audio = audio[: int(duration * target_sr)]
    audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
    return audio


def save_audio(audio: torch.Tensor, path: str, sr: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio.cpu().numpy(), sr)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  Unconditional generation
# ─────────────────────────────────────────────

def cmd_uncond(args):
    print(f"[generate/uncond] Loading checkpoint: {args.ckpt}")
    config = LiveVoiceConfig(device=("cuda" if torch.cuda.is_available() else "cpu"))
    dac_model = DACModel(config)
    model = UnconditionalModel(config, dac_model)

    lit = UnconditionalLightningModule.load_from_checkpoint(
        args.ckpt, config=config, model=model, strict=False
    )
    lit.eval()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    lit = lit.to(device)

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.n_samples):
        print(f"  Generating sample {i + 1}/{args.n_samples}...")
        codes = lit.model.generate(
            batch_size=1,
            max_steps=args.max_steps,
            temperature=args.temperature,
            top_p=args.top_p if args.top_p > 0 else None,
            top_k=args.top_k if args.top_k > 0 else None,
        )
        audio = lit.model.decode_to_audio(codes)  # (1, T)
        save_audio(audio[0], os.path.join(args.out_dir, f"uncond_{i:03d}.wav"), config.sample_rate)


# ─────────────────────────────────────────────
#  VC inference
# ─────────────────────────────────────────────

def cmd_vc(args):
    print(f"[generate/vc] Loading checkpoint: {args.ckpt}")
    config = LiveVoiceConfig(device=("cuda" if torch.cuda.is_available() else "cpu"))
    dac_model = DACModel(config)
    content_extractor = HuBERTContentExtractor(config)
    model = LiveVoiceModel(config, dac_model, content_extractor)

    lit = LiveVoiceLightningModule.load_from_checkpoint(
        args.ckpt, config=config, model=model, strict=False
    )
    lit.eval()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    lit = lit.to(device)

    sr = config.sample_rate
    ref = load_audio(args.reference, sr, duration=None).unsqueeze(0).to(device)
    ctn = load_audio(args.content, sr, duration=None).unsqueeze(0).to(device)

    print("  Running VC generation...")
    codes = lit.model.generate(
        reference_audio=ref,
        content_audio=ctn,
        temperature=args.temperature,
        top_p=args.top_p if args.top_p > 0 else None,
        top_k=args.top_k if args.top_k > 0 else None,
        cfg_scale=args.cfg_scale,
    )
    audio = lit.model.decode_to_audio(codes)  # (1, T)

    os.makedirs(args.out_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(args.content))[0]
    save_audio(audio[0], os.path.join(args.out_dir, f"vc_{out_name}.wav"), sr)
    save_audio(ref[0], os.path.join(args.out_dir, f"ref_{out_name}.wav"), sr)
    save_audio(ctn[0], os.path.join(args.out_dir, f"ctn_{out_name}.wav"), sr)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── uncond ──────────────────────────────
    pu = sub.add_parser("uncond")
    pu.add_argument("--ckpt", required=True)
    pu.add_argument("--out_dir", default="./output/samples")
    pu.add_argument("--n_samples", type=int, default=4)
    pu.add_argument("--max_steps", type=int, default=200)
    pu.add_argument("--temperature", type=float, default=1.0)
    pu.add_argument("--top_p", type=float, default=0.9)
    pu.add_argument("--top_k", type=int, default=0)

    # ── vc ──────────────────────────────────
    pv = sub.add_parser("vc")
    pv.add_argument("--ckpt", required=True)
    pv.add_argument("--reference", required=True, help="Reference speaker audio (.wav)")
    pv.add_argument("--content", required=True, help="Content/source audio (.wav)")
    pv.add_argument("--out_dir", default="./output/samples")
    pv.add_argument("--temperature", type=float, default=1.0)
    pv.add_argument("--top_p", type=float, default=0.9)
    pv.add_argument("--top_k", type=int, default=0)
    pv.add_argument("--cfg_scale", type=float, default=1.0)

    args = p.parse_args()
    if args.cmd == "uncond":
        cmd_uncond(args)
    else:
        cmd_vc(args)


if __name__ == "__main__":
    main()
