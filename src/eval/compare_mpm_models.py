"""Compare 3 models on paralinguistic audio (laughing, happy, sad, whisper).

Usage:
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src python src/eval/compare_mpm_models.py

Outputs to /workspace/LiveVoice/mpm_comparison/<model>/<style>_<uttid>.wav
"""
import argparse
import dataclasses
import os
from pathlib import Path

import torch
import torchaudio

MODELS = {
    "baseline":     "/mnt/data/disk2/yejin/LiveVoice/checkpoints/180ms_baseline/step_latest.ckpt",
    "newmpm":       "/mnt/data/disk2/yejin/LiveVoice/checkpoints/180ms_newmpm_expresso/step_latest.ckpt",
    "fullmpm":      "/mnt/data/disk2/yejin/LiveVoice/checkpoints/180ms_fullmpm_expresso/step_latest.ckpt",
}

# Source audios: paralinguistic styles from Expresso
SOURCES = {
    "laughing_ex01_095": "/mnt/data/disk3/yejin/expresso/audio_48khz/read/ex01/laughing/base/ex01_laughing_00095.wav",
    "laughing_ex02_061": "/mnt/data/disk3/yejin/expresso/audio_48khz/read/ex02/laughing/base/ex02_laughing_00061.wav",
    "happy_ex04_298":    "/mnt/data/disk3/yejin/expresso/audio_48khz/read/ex04/happy/base/ex04_happy_00298.wav",
    "sad_ex04_144":      "/mnt/data/disk3/yejin/expresso/audio_48khz/read/ex04/sad/base/ex04_sad_00144.wav",
    "whisper_ex02_059":  "/mnt/data/disk3/yejin/expresso/audio_48khz/read/ex02/whisper/base/ex02_whisper_00059.wav",
}

# Reference speaker (target voice)
REF_AUDIO = "/mnt/data/disk2/LibriTTS/dev-clean/8842/304647/8842_304647_000039_000000.wav"

OUT_DIR = "/workspace/LiveVoice/mpm_comparison"


def load_audio(path, sr=16000):
    wav, orig_sr = torchaudio.load(path)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav[:1]  # mono


def main():
    from livevoice.config import LiveVoiceConfig
    from livevoice.utils.checkpoint import read_config_from_ckpt
    from eval.anonymize_vpc_dirs import _build_vc_model

    dev = torch.device("cuda")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Also copy source originals for reference
    src_dir = os.path.join(OUT_DIR, "source_original")
    os.makedirs(src_dir, exist_ok=True)
    for name, path in SOURCES.items():
        wav = load_audio(path)
        torchaudio.save(os.path.join(src_dir, f"{name}.wav"), wav, 16000)
    print(f"Source originals saved to {src_dir}/")

    ref_audio = load_audio(REF_AUDIO).to(dev)
    # Also save ref
    torchaudio.save(os.path.join(OUT_DIR, "reference_speaker.wav"), ref_audio.cpu(), 16000)

    for model_name, ckpt_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Loading {model_name}: {ckpt_path}")
        print(f"{'='*60}")

        class FakeArgs:
            ckpt = ckpt_path

        cfg_dict = read_config_from_ckpt(ckpt_path)
        known = {f.name for f in dataclasses.fields(LiveVoiceConfig)}
        cfg = LiveVoiceConfig(**{k: v for k, v in cfg_dict.items() if k in known})
        lit = _build_vc_model(FakeArgs(), cfg, dev)
        model = lit.model

        out_dir = os.path.join(OUT_DIR, model_name)
        os.makedirs(out_dir, exist_ok=True)

        for src_name, src_path in SOURCES.items():
            print(f"  Generating: {src_name}")
            content_audio = load_audio(src_path).to(dev)

            with torch.no_grad():
                codes = model.generate(
                    reference_audio=ref_audio,
                    content_audio=content_audio,
                    temperature=1.0,
                    cfg_scale=1.0,
                )
                out_audio = model.decode_to_audio(codes)

            out_path = os.path.join(out_dir, f"{src_name}.wav")
            torchaudio.save(out_path, out_audio.cpu().float(), cfg.sample_rate)
            print(f"    -> {out_path}")

        # Free GPU memory before loading next model
        del lit, model
        torch.cuda.empty_cache()

    print(f"\nDone! All outputs in {OUT_DIR}/")
    print("Compare: source_original/ vs baseline/ vs newmpm/ vs fullmpm/")


if __name__ == "__main__":
    main()
