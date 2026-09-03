"""Measure per-stage RTF of LiveVoice voice conversion.

Tests: fp32 baseline, bf16 autocast, torch.compile + bf16.

Usage:
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src python src/eval/bench_rtf.py \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/180ms_baseline/step_latest.ckpt \
        --n 20 --warmup 3
"""
import argparse
import time
from pathlib import Path

import torch
import torchaudio


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_bench(model, wavs, ref_audio, cfg, dev, n, warmup, label=""):
    rows = []
    for i, w in enumerate(wavs[:n + warmup]):
        audio, sr = torchaudio.load(str(w))
        if sr != cfg.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, cfg.sample_rate)
        audio = audio[:1].to(dev)
        audio_dur = audio.shape[-1] / cfg.sample_rate

        with torch.no_grad():
            sync()
            t0 = time.perf_counter()
            codes = model.generate(
                reference_audio=ref_audio,
                content_audio=audio,
                temperature=1.0,
                cfg_scale=1.0,
            )
            sync()
            t_gen = time.perf_counter() - t0

            sync()
            t0 = time.perf_counter()
            out_audio = model.decode_to_audio(codes)
            sync()
            t_dec = time.perf_counter() - t0

        n_frames = codes.shape[-1]
        if i < warmup:
            pass
        else:
            rows.append({"dur": audio_dur, "frames": n_frames, "gen": t_gen, "dec": t_dec})

    total_dur = sum(r["dur"] for r in rows)
    total_frames = sum(r["frames"] for r in rows)
    total_gen = sum(r["gen"] for r in rows)
    total_dec = sum(r["dec"] for r in rows)
    ms_per_step = 1000 * total_gen / total_frames
    # AR RTF: from internal timing printed by generate()
    # Here we report generate RTF (includes preprocess) and AR-only estimate
    # Since preprocess is ~0.17s and AR is the rest, AR time ≈ gen - 0.17*n
    preprocess_est = 0.17 * len(rows)
    ar_time_est = total_gen - preprocess_est
    ar_rtf = ar_time_est / total_dur

    print(f"\n  [{label}] {len(rows)} utts, {total_dur:.1f}s audio, {total_frames} frames")
    print(f"    generate RTF = {total_gen/total_dur:.4f}  (includes preprocess)")
    print(f"    AR-only RTF  ≈ {ar_rtf:.4f}  (preprocess ~{preprocess_est:.1f}s subtracted)")
    print(f"    codec_dec RTF = {total_dec/total_dur:.4f}")
    print(f"    ms/step = {ms_per_step:.2f}")
    return ms_per_step, ar_rtf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--libritts", default="/mnt/data/disk2/LibriTTS")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import dataclasses
    from livevoice.config import LiveVoiceConfig
    from livevoice.utils.checkpoint import read_config_from_ckpt
    from eval.anonymize_vpc_dirs import _build_vc_model

    class FakeArgs:
        ckpt = args.ckpt

    cfg_dict = read_config_from_ckpt(args.ckpt)
    known = {f.name for f in dataclasses.fields(LiveVoiceConfig)}
    cfg = LiveVoiceConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    dev = torch.device(args.device)
    lit = _build_vc_model(FakeArgs(), cfg, dev)
    model = lit.model

    all_wavs = sorted(Path(args.libritts, args.split).rglob("*.wav"))
    wavs = []
    for w in all_wavs:
        info = torchaudio.info(str(w))
        dur = info.num_frames / info.sample_rate
        if dur >= 2.0:
            wavs.append(w)
            if len(wavs) >= args.n + args.warmup:
                break

    ref_audio, ref_sr = torchaudio.load(str(wavs[0]))
    if ref_sr != cfg.sample_rate:
        ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, cfg.sample_rate)
    ref_audio = ref_audio[:1].to(dev)

    results = {}

    # 1) FP32 baseline (current)
    print("=" * 60)
    print("1) FP32 (baseline)")
    results["fp32"] = run_bench(model, wavs, ref_audio, cfg, dev, args.n, args.warmup, "fp32")

    # 2) BF16 autocast
    print("\n" + "=" * 60)
    print("2) BF16 autocast")
    model_bf16 = model
    orig_generate = model_bf16.generate.__func__
    orig_decode = model_bf16.decode_to_audio.__func__

    import types

    def generate_bf16(self, *a, **kw):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return orig_generate(self, *a, **kw)

    def decode_bf16(self, codes):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return orig_decode(self, codes)

    model_bf16.generate = types.MethodType(generate_bf16, model_bf16)
    model_bf16.decode_to_audio = types.MethodType(decode_bf16, model_bf16)
    results["bf16"] = run_bench(model_bf16, wavs, ref_audio, cfg, dev, args.n, args.warmup, "bf16")

    # 3) torch.compile + BF16
    print("\n" + "=" * 60)
    print("3) torch.compile + BF16")
    print("  Compiling decoder... (first run will be slow)")
    try:
        model.transformer.decode_step = torch.compile(model.transformer.decode_step, mode="reduce-overhead")
        results["compile+bf16"] = run_bench(model_bf16, wavs, ref_audio, cfg, dev, args.n, args.warmup + 2, "compile+bf16")
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        results["compile+bf16"] = (None, None)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Config':<20} {'ms/step':>10} {'AR RTF':>10}")
    print("-" * 42)
    for name, (ms, rtf) in results.items():
        if ms is not None:
            print(f"{name:<20} {ms:>10.2f} {rtf:>10.4f}")


if __name__ == "__main__":
    main()
