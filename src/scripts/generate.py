"""Inference / generation script for LiveVoice.

Supports both unconditional generation and VC inference.

Usage examples:
    # Unconditional generation
    python src/scripts/generate.py uncond \
        --ckpt output/checkpoints/uncond_vctk/last.ckpt \
        --n_samples 4 --max_steps 200 --out_dir output/samples

    # VC inference
    CUDA_VISIBLE_DEVICES=5 python scripts/generate.py vc \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/ecapa_sw2v/step_latest.ckpt \
        --reference /mnt/data/disk2/LibriTTS/test-clean/121/121726/121_121726_000020_000001.wav  \
        --content /mnt/data/disk2/LibriTTS/test-clean/5105/28240/5105_28240_000013_000005.wav
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
from livevoice.model import (
    HuBERTContentExtractor,
    StreamVoiceAnonContentEncoder,
    Sw2vContentEncoder,
    LiveVoiceModel,
    UnconditionalModel,
)
from livevoice.model import build_codec
from livevoice.lightning import UnconditionalLightningModule, LiveVoiceLightningModule
from livevoice.utils.checkpoint import (
    infer_content_fsq_from_ckpt,
    infer_content_source_from_ckpt,
    infer_speaker_conditioning_from_ckpt,
    infer_speaker_encoder_from_ckpt,
)


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
    codec_name = str(getattr(args, "codec", "mimi")).lower()
    cfg_kwargs["codec"] = codec_name

    # Force config.sample_rate to the codec's native rate. This makes any
    # downstream consumer (e.g. MimiCodec.input_sr / JHCodecModel.input_sr) see
    # a consistent SR: mimi=24 kHz, jhcodec=16 kHz.
    cfg_kwargs["sample_rate"] = {"mimi": 24000, "jhcodec": 16000}[codec_name]

    if getattr(args, "n_codebooks", None) is not None:
        cfg_kwargs["n_codebooks_predict"] = int(args.n_codebooks)

    # FSQ content bottleneck: inferred from the ckpt's `model.content_fsq._levels`
    # buffer. Must match training or the sw2v content path won't load / align.
    ckpt_path = getattr(args, "ckpt", None)
    if ckpt_path:
        fsq_levels = infer_content_fsq_from_ckpt(ckpt_path)
        if fsq_levels is not None:
            cfg_kwargs["use_content_fsq"] = True
            cfg_kwargs["fsq_levels"] = fsq_levels
            print(f"[generate] content_fsq=auto → {fsq_levels} (from checkpoint keys)")

    src = str(getattr(args, "content_source", "auto")).lower()
    if src != "auto":
        cfg_kwargs["content_source"] = src
    speaker = str(getattr(args, "speaker_conditioning", "auto")).lower()
    if speaker != "auto":
        cfg_kwargs["speaker_conditioning"] = speaker
    if getattr(args, "speaker_prefix_len", None) is not None:
        cfg_kwargs["speaker_prefix_len"] = int(args.speaker_prefix_len)
    speaker_encoder = str(getattr(args, "speaker_encoder_type", "auto")).lower()
    if speaker_encoder != "auto":
        cfg_kwargs["speaker_encoder_type"] = speaker_encoder
    for name in (
        "speechbrain_source",
        "speechbrain_savedir",
        "speechbrain_sample_rate",
        "speechbrain_embedding_dim",
        "streamvoiceanon_repo",
        "streamvoiceanon_encoder_config",
        "streamvoiceanon_encoder_ckpt",
    ):
        value = getattr(args, name, None)
        if value:
            cfg_kwargs[name] = value

    return LiveVoiceConfig(**cfg_kwargs)


def _resolve_content_source(args, codec_name: str) -> str:
    src = str(getattr(args, "content_source", "auto")).lower()
    if src != "auto":
        return src
    inferred = infer_content_source_from_ckpt(args.ckpt)
    if inferred is not None:
        print(f"[generate] content_source=auto → {inferred!r} (from checkpoint keys)")
        return inferred
    fallback = "hubert"
    print(f"[generate] content_source=auto → {fallback!r} (fallback; ckpt keys ambiguous)")
    return fallback


def _resolve_speaker_conditioning(args) -> str:
    speaker = str(getattr(args, "speaker_conditioning", "auto")).lower()
    if speaker != "auto":
        return speaker
    inferred = infer_speaker_conditioning_from_ckpt(args.ckpt)
    if inferred is not None:
        print(f"[generate] speaker_conditioning=auto → {inferred!r} (from checkpoint keys)")
        return inferred
    fallback = "prefix"
    print(f"[generate] speaker_conditioning=auto → {fallback!r} (fallback; ckpt keys ambiguous)")
    return fallback


def _resolve_speaker_encoder_type(args) -> str:
    speaker_encoder = str(getattr(args, "speaker_encoder_type", "auto")).lower()
    if speaker_encoder != "auto":
        return speaker_encoder
    inferred = infer_speaker_encoder_from_ckpt(args.ckpt)
    if inferred is not None:
        print(f"[generate] speaker_encoder_type=auto → {inferred!r} (from checkpoint keys)")
        return inferred
    fallback = "codec"
    print(f"[generate] speaker_encoder_type=auto → {fallback!r} (fallback; ckpt keys ambiguous)")
    return fallback


# ─────────────────────────────────────────────
#  Unconditional generation
# ─────────────────────────────────────────────

def cmd_uncond(args):
    print(f"[generate/uncond] Loading checkpoint: {args.ckpt}")
    # Build config tied to the codec the ckpt was trained with.
    # SR is forced to the codec's native rate (JHCodec=16k, Mimi=24k) so audio loaded
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
    codec_name = str(getattr(args, "codec", "mimi")).lower()
    content_source = _resolve_content_source(args, codec_name)
    args.content_source = content_source
    args.speaker_conditioning = _resolve_speaker_conditioning(args)
    args.speaker_encoder_type = _resolve_speaker_encoder_type(args)
    config = _build_inference_config(args)
    codec_model = build_codec(config)
    print(
        f"  codec={config.codec}  content_source={config.content_source}  "
        f"speaker_conditioning={config.speaker_conditioning}  "
        f"speaker_encoder_type={config.speaker_encoder_type}  "
        f"codec_sr={codec_model.sample_rate}  n_codebooks_predict={config.n_codebooks_predict}"
    )
    if config.content_source == "hubert":
        content_extractor = HuBERTContentExtractor(config)
    elif config.content_source == "streamvoiceanon":
        content_extractor = StreamVoiceAnonContentEncoder(config)
    elif config.content_source == "sw2v":
        content_extractor = Sw2vContentEncoder(config)
    else:
        content_extractor = None
    model = LiveVoiceModel(config, codec_model, content_extractor, prosody_extractor=None)

    lit = LiveVoiceLightningModule.load_from_checkpoint(
        args.ckpt, config=config, model=model, strict=False
    )
    lit.eval()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    lit = lit.to(device)

    # Load audio at the codec's native SR
    # were trained at 16 kHz; feeding 24 kHz now would silently corrupt encoding.
    sr = int(codec_model.sample_rate)

    # For the codec speaker encoder, the reference length directly sets the prefix
    # length (~50 fps → one prefix token per jhcodec frame). Training used ~4s
    # windows; a full-length reference both mismatches training and can overflow
    # the KV-cache budget. So trim for the codec path. ECAPA / spark_global pool the
    # whole utterance to a fixed token count, so they keep the full reference.
    ref_max_seconds = (
        None if config.speaker_encoder_type in ("speechbrain_ecapa", "spark_global")
        else float(args.ref_max_seconds)
    )
    print(
        f"  reference trim = "
        f"{'full utterance' if ref_max_seconds is None else f'{ref_max_seconds:g}s (~{int(ref_max_seconds*50)} prefix tokens)'}"
    )
    ref = load_audio(args.reference, sr, duration=ref_max_seconds).unsqueeze(0).to(device)
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
    pu.add_argument("--codec", type=str, default="mimi", choices=["mimi", "jhcodec"])
    pu.add_argument("--n_codebooks", type=int, default=None,
                    help="Override config.n_codebooks_predict to match the ckpt.")
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
    pv.add_argument("--codec", type=str, default="jhcodec", choices=["mimi", "jhcodec"])
    pv.add_argument("--n_codebooks", type=int, default=None,
                    help="Override config.n_codebooks_predict to match the ckpt.")
    pv.add_argument("--reference", required=True, help="Reference speaker audio (.wav)")
    pv.add_argument("--content", required=True, help="Content/source audio (.wav)")
    pv.add_argument("--ref_max_seconds", type=float, default=4.0,
                    help="Trim reference to this many seconds for the codec speaker encoder "
                         "(prefix length = ref_seconds*50fps; training used ~4s). Ignored for "
                         "ECAPA / spark_global (fixed token count). Set very large to disable.")
    pv.add_argument("--out_dir",        default=None,        help="Output index only (e.g. 0,1,2). If omitted, next available index is used.",    )
    pv.add_argument("--temperature", type=float, default=1.0)
    pv.add_argument("--top_p", type=float, default=0.9)
    pv.add_argument("--top_k", type=int, default=0)
    pv.add_argument("--cfg_scale", type=float, default=1.0)
    pv.add_argument(
        "--content_source",
        type=str,
        default="auto",
        choices=["auto", "hubert", "mimi_semantic", "streamvoiceanon", "sw2v"],
        help="Content path. 'auto' inspects ckpt (same as eval_libritts_test_clean_wer.py).",
    )
    pv.add_argument(
        "--speaker_conditioning",
        type=str,
        default="auto",
        choices=["auto", "crossattn", "global_avg", "prefix"],
    )
    pv.add_argument("--speaker_prefix_len", type=int, default=4)
    pv.add_argument(
        "--speaker_encoder_type",
        type=str,
        default="auto",
        choices=["auto", "codec", "speechbrain_ecapa", "spark_global"],
    )
    pv.add_argument("--speechbrain_source", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    pv.add_argument(
        "--speechbrain_savedir",
        type=str,
        default="/mnt/data/disk2/yejin/LiveVoice/pretrained_models/speechbrain__spkrec-ecapa-voxceleb",
    )
    pv.add_argument("--speechbrain_sample_rate", type=int, default=16000)
    pv.add_argument("--speechbrain_embedding_dim", type=int, default=192)
    pv.add_argument("--streamvoiceanon_repo", type=str, default="/workspace/StreamVoiceAnon")
    pv.add_argument(
        "--streamvoiceanon_encoder_config",
        type=str,
        default="/workspace/StreamVoiceAnon/configs/hydra_arcs/speech_tokenizers/causal-encoder-lfq-8192.yaml",
    )
    pv.add_argument(
        "--streamvoiceanon_encoder_ckpt",
        type=str,
        default="/workspace/StreamVoiceAnon/ckpt/asr_s2s_bsq_8192_causal_down_whisper.pth",
    )

    args = p.parse_args()
    if args.cmd == "uncond":
        cmd_uncond(args)
    else:
        cmd_vc(args)


if __name__ == "__main__":
    main()
