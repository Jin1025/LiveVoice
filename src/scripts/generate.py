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
        --reference /mnt/data/disk2/VCTK-Corpus/wav48/p275/p275_002.wav \
        --content /mnt/data/disk2/VCTK-Corpus/wav48/p225/p225_001.wav \
        --codec dac

    CUDA_VISIBLE_DEVICES=3 python scripts/generate.py vc \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/libritts_mimi_resume/step_latest.ckpt \
        --reference /mnt/data/disk2/VCTK-Corpus/wav48/p275/p275_002.wav \
        --content /mnt/data/disk2/VCTK-Corpus/wav48/p225/p225_001.wav \
        --codec mimi

    CUDA_VISIBLE_DEVICES=3 python scripts/generate.py vc \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/libritts_mimi_resume/step_latest.ckpt \
        --reference /mnt/data/disk2/LibriTTS/test-clean/121/121726/121_121726_000020_000001.wav  \
        --content /mnt/data/disk2/LibriTTS/test-clean/5105/28240/5105_28240_000013_000005.wav  \
        --codec mimi
"""
import argparse
import os
import sys
from pathlib import Path
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import soundfile as sf
import librosa

from livevoice.config import LiveVoiceConfig
from livevoice.model import DACModel, HuBERTContentExtractor, LiveVoiceModel, UnconditionalModel
from livevoice.model import build_codec
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


def _resolve_output_dir(ckpt_path: str, out_index: str | None, reference_path: str | None = None) -> str:
    """Build output dir as ./output/<exp_name>/<idx>.

    - exp_name is inferred from ckpt parent directory name.
      e.g. .../checkpoints/libritts_no_VTLNl/step_latest.ckpt -> libritts_no_VTLNl
    - idx defaults to reference speaker id if reference_path is provided
      (e.g. p240_002.wav -> 240). If parsing fails, auto-increments from 0.
    - if out_index is provided, it must be a non-negative integer string.
    """
    p = Path(ckpt_path).resolve()
    parts = p.parts
    exp_name = None
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        if i + 1 < len(parts):
            exp_name = parts[i + 1]
    if not exp_name:
        # Fallback for non-standard checkpoint paths
        exp_name = p.parent.name
    base = Path("./output") / exp_name
    base.mkdir(parents=True, exist_ok=True)

    if out_index is None:
        idx = None
        if reference_path is not None:
            ref_stem = Path(reference_path).stem
            m = re.match(r"^p?(\d+)_", ref_stem, re.IGNORECASE)
            if m:
                idx = int(m.group(1))
        if idx is None:
            used = set()
            for p in base.iterdir():
                if p.is_dir() and p.name.isdigit():
                    used.add(int(p.name))
            idx = 0
            while idx in used:
                idx += 1
    else:
        s = str(out_index).strip()
        if not s.isdigit():
            raise ValueError("--out_dir must be a numeric index like 0, 1, 2 ...")
        idx = int(s)

    return str(base / str(idx))


# ─────────────────────────────────────────────
#  Inference config
# ─────────────────────────────────────────────

def _build_inference_config(args) -> LiveVoiceConfig:
    """Build a config that matches the ckpt's training-time settings.

    Key idea: at inference, override the fields that may have drifted in the
    repo defaults (sample_rate, n_codebooks_predict) so they match what the
    checkpoint was trained with. Audio is then loaded at the *codec's* native
    SR in cmd_vc/cmd_uncond, independent of config.sample_rate.
    """
    cfg_kwargs = dict(device=("cuda" if torch.cuda.is_available() else "cpu"))
    codec_name = str(getattr(args, "codec", "dac")).lower()
    cfg_kwargs["codec"] = codec_name

    # Force config.sample_rate to the codec's native rate. This makes any
    # downstream consumer (e.g. MimiCodec.input_sr) see a consistent SR.
    if codec_name == "mimi":
        cfg_kwargs["sample_rate"] = 24000
    else:  # dac
        # DAC defaults: 16khz model is the trained one for the existing ckpts
        cfg_kwargs["sample_rate"] = 16000
        cfg_kwargs["dac_model_type"] = getattr(args, "dac_model_type", "16khz")
        cfg_kwargs["dac_sample_rate"] = 16000

    if getattr(args, "n_codebooks", None) is not None:
        cfg_kwargs["n_codebooks_predict"] = int(args.n_codebooks)

    return LiveVoiceConfig(**cfg_kwargs)


# ─────────────────────────────────────────────
#  Unconditional generation
# ─────────────────────────────────────────────

def cmd_uncond(args):
    print(f"[generate/uncond] Loading checkpoint: {args.ckpt}")
    # Build config tied to the codec the ckpt was trained with.
    # SR is forced to the codec's native rate (DAC=16k, Mimi=24k) so audio loaded
    # in this script matches what the model saw during training, regardless of
    # what config.sample_rate happens to default to.
    config = _build_inference_config(args)
    codec_model = build_codec(config)
    print(f"  codec={config.codec}  codec_sr={codec_model.sample_rate}  n_codebooks_predict={config.n_codebooks_predict}")
    model = UnconditionalModel(config, codec_model)

    lit = UnconditionalLightningModule.load_from_checkpoint(
        args.ckpt, config=config, model=model, strict=False
    )
    lit.eval()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    lit = lit.to(device)

    sr = int(codec_model.sample_rate)
    out_dir = _resolve_output_dir(args.ckpt, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output dir: {out_dir}")
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
        save_audio(audio[0], os.path.join(out_dir, f"uncond_{i:03d}.wav"), sr)


# ─────────────────────────────────────────────
#  VC inference
# ─────────────────────────────────────────────

def cmd_vc(args):
    print(f"[generate/vc] Loading checkpoint: {args.ckpt}")
    config = _build_inference_config(args)
    codec_model = build_codec(config)
    print(f"  codec={config.codec}  codec_sr={codec_model.sample_rate}  n_codebooks_predict={config.n_codebooks_predict}")
    content_extractor = HuBERTContentExtractor(config)
    model = LiveVoiceModel(config, codec_model, content_extractor)

    lit = LiveVoiceLightningModule.load_from_checkpoint(
        args.ckpt, config=config, model=model, strict=False
    )
    lit.eval()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    lit = lit.to(device)

    # Load audio at the codec's native SR (NOT config.sample_rate). DAC ckpts
    # were trained at 16 kHz; feeding 24 kHz now would silently corrupt encoding.
    sr = int(codec_model.sample_rate)
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

    out_dir = _resolve_output_dir(args.ckpt, args.out_dir, reference_path=args.reference)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output dir: {out_dir}")
    content_name = os.path.splitext(os.path.basename(args.content))[0]
    ref_name = os.path.splitext(os.path.basename(args.reference))[0]
    save_audio(audio[0], os.path.join(out_dir, f"vc_{content_name}.wav"), sr)
    save_audio(ref[0], os.path.join(out_dir, f"ref_{ref_name}.wav"), sr)
    save_audio(ctn[0], os.path.join(out_dir, f"ctn_{content_name}.wav"), sr)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── uncond ──────────────────────────────
    pu = sub.add_parser("uncond")
    pu.add_argument("--ckpt", required=True)
    pu.add_argument("--codec", type=str, default="dac", choices=["dac", "mimi"])
    pu.add_argument("--n_codebooks", type=int, default=None,
                    help="Override config.n_codebooks_predict to match the ckpt (DAC ckpts often used 9).")
    pu.add_argument("--dac_model_type", type=str, default="16khz", choices=["16khz", "24khz", "44khz"])
    pu.add_argument(
        "--out_dir",
        default=None,
        help="Output index only (e.g. 0,1,2). If omitted, next available index is used.",
    )
    pu.add_argument("--n_samples", type=int, default=4)
    pu.add_argument("--max_steps", type=int, default=200)
    pu.add_argument("--temperature", type=float, default=1.0)
    pu.add_argument("--top_p", type=float, default=0.9)
    pu.add_argument("--top_k", type=int, default=0)

    # ── vc ──────────────────────────────────
    pv = sub.add_parser("vc")
    pv.add_argument("--ckpt", required=True)
    pv.add_argument("--codec", type=str, default="dac", choices=["dac", "mimi"])
    pv.add_argument("--n_codebooks", type=int, default=None,
                    help="Override config.n_codebooks_predict to match the ckpt (DAC ckpts often used 9).")
    pv.add_argument("--dac_model_type", type=str, default="16khz", choices=["16khz", "24khz", "44khz"])
    pv.add_argument("--reference", required=True, help="Reference speaker audio (.wav)")
    pv.add_argument("--content", required=True, help="Content/source audio (.wav)")
    pv.add_argument(
        "--out_dir",
        default=None,
        help="Output index only (e.g. 0,1,2). If omitted, next available index is used.",
    )
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
