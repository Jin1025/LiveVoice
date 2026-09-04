"""Latency/RTF benchmark for LiveVoice inference optimizations.

Unlike the old bench_rtf_63.py, this uses the checkpoint's sampling settings.
For the supplied checkpoint top_k=1 is mathematically greedy, so LiveVoice's
fast argmax path avoids needless reductions, sorting, softmax, and multinomial.

Timing definitions:
  model RTF  = synchronized (generate + codec decode) wall time / input duration
  output RTF = the same wall time / actual decoded output duration
Audio loading, resampling, and host-to-device transfer remain outside timing so
the result is directly comparable with bench_rtf_63.py.
"""
import argparse
import dataclasses
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torchaudio


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def amp_context(precision: str):
    if precision == "bf16-autocast":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wavdir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--zipformer_ckpt", default="")
    ap.add_argument("--jhcodec_ckpt", default="")
    ap.add_argument("--jhcodec_config", default="")
    ap.add_argument("--jhcodec_repo", default="")
    ap.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16", "bf16-autocast"),
        default="fp32",
        help="bf16/fp16 cast only the AR path; bf16-autocast is retained for comparison",
    )
    ap.add_argument("--compile", action="store_true",
                    help="experimentally compile only the eager long-utterance fallback")
    ap.add_argument("--legacy_sampling", action="store_true",
                    help="reproduce old benchmark's unconstrained multinomial instead of ckpt settings")
    ap.add_argument("--show_progress", action="store_true")
    ap.add_argument("--no_fused_heads", action="store_true",
                    help="disable the one-GEMM greedy codebook-head optimization")
    ap.add_argument("--no_fused_qkv", action="store_true",
                    help="disable packed self-attention QKV projections")
    ap.add_argument("--no_cuda_graph", action="store_true",
                    help="disable full AR decoder CUDA Graph replay")
    ap.add_argument("--short_only", action="store_true",
                    help="benchmark only files whose prompt+target+delay timeline is < max_seq_len")
    ap.add_argument("--no_tf32", action="store_true")
    ap.add_argument("--profile", action="store_true",
                    help="print a CUDA/CPU operator profile for the first measured utterance")
    ap.add_argument("--validate", action="store_true",
                    help="compare eager and optimized codes/audio before benchmarking")
    ap.add_argument("--validate_index", type=int, default=0,
                    help="wav index used by --validate (use a long utterance to test fallback)")
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()

    from livevoice.config import LiveVoiceConfig
    from livevoice.utils.checkpoint import read_config_from_ckpt
    from eval.anonymize_vpc_dirs import _build_vc_model

    class FakeArgs:
        ckpt = args.ckpt

    cfg_dict = read_config_from_ckpt(args.ckpt)
    known = {f.name for f in dataclasses.fields(LiveVoiceConfig)}
    filtered = {k: v for k, v in cfg_dict.items() if k in known}
    for name in ("zipformer_ckpt", "jhcodec_ckpt", "jhcodec_config", "jhcodec_repo"):
        value = getattr(args, name)
        if value:
            filtered[name] = value
    if args.jhcodec_repo:
        filtered["sw2v_repo"] = args.jhcodec_repo
    cfg = LiveVoiceConfig(**filtered)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    torch.set_float32_matmul_precision("high" if not args.no_tf32 else "highest")

    dev = torch.device(args.device)
    lit = _build_vc_model(FakeArgs(), cfg, dev)
    model = lit.model.eval()
    model.set_fast_inference(not args.no_fused_qkv)
    if args.precision == "bf16":
        model.set_ar_inference_dtype(torch.bfloat16)
    elif args.precision == "fp16":
        model.set_ar_inference_dtype(torch.float16)

    if args.compile:
        cache_dir = Path(__file__).resolve().parents[2] / ".torchinductor_cache"
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
        print(f"[compile] reduce-overhead CUDA Graph compile; cache={cache_dir}")
        compiled_decode_step = torch.compile(
            model.transformer.decode_step,
            mode="reduce-overhead",
            dynamic=True,
            fullgraph=False,
        )

        def decode_step_compiled(*call_args, **call_kwargs):
            # Each AR token is a new iteration. The clone releases the compiled
            # graph's static output buffer before the next replay overwrites it.
            torch.compiler.cudagraph_mark_step_begin()
            return compiled_decode_step(*call_args, **call_kwargs).clone()

        model.transformer.decode_step = decode_step_compiled

    all_wavs = sorted(Path(args.wavdir).glob("*.wav"))
    if not all_wavs:
        raise SystemExit(f"no wav files under {args.wavdir}")

    ref_audio, ref_sr = torchaudio.load(str(all_wavs[0]))
    if ref_sr != cfg.sample_rate:
        ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, cfg.sample_rate)
    ref_audio = ref_audio[:1].to(dev)

    if args.short_only:
        hop = int(model.codec_model.hop_length)
        ref_frames = math.ceil(ref_audio.shape[-1] / hop)
        delay_tail = int(model._delay_tail())
        eligible = []
        for wav_path in all_wavs:
            info = torchaudio.info(str(wav_path))
            target_samples = math.ceil(info.num_frames * cfg.sample_rate / info.sample_rate)
            target_frames = math.ceil(target_samples / hop)
            if ref_frames + target_frames + delay_tail < int(cfg.max_seq_len):
                eligible.append(wav_path)
        if not eligible:
            raise SystemExit("--short_only found no CUDA-graph-eligible wavs")
        warmup_wavs = [eligible[i % len(eligible)] for i in range(args.warmup)]
        measured_wavs = [eligible[i % len(eligible)] for i in range(args.n)]
        wavs = warmup_wavs + measured_wavs
        print(f"[short_only] eligible={len(eligible)}/{len(all_wavs)}; "
              f"prompt_frames={ref_frames} max_seq_len={cfg.max_seq_len}")
        if args.n > len(eligible):
            print(f"[short_only] n={args.n}: cycling the {len(eligible)} eligible files")
    else:
        wavs = all_wavs[: args.n + args.warmup]
        if len(wavs) < args.n + args.warmup:
            raise SystemExit(f"need {args.n + args.warmup} wavs, found {len(wavs)}")

    if args.legacy_sampling:
        gen_kwargs = dict(temperature=1.0, cfg_scale=1.0)
        sampling_label = "legacy unconstrained multinomial"
    else:
        top_k = getattr(cfg, "top_k", None)
        top_p = getattr(cfg, "top_p", None)
        gen_kwargs = dict(
            temperature=float(getattr(cfg, "temperature", 1.0)),
            top_k=None if top_k is None else int(top_k),
            top_p=None if top_p is None else float(top_p),
            cfg_scale=1.0,
        )
        sampling_label = f"checkpoint temperature={gen_kwargs['temperature']} " \
                         f"top_k={gen_kwargs['top_k']} top_p={gen_kwargs['top_p']}"

    print(f"Found {len(wavs)} wav files")
    print(f"precision={args.precision} tf32={not args.no_tf32} sampling={sampling_label}")

    if args.validate:
        if not 0 <= args.validate_index < len(wavs):
            raise SystemExit(f"validate_index {args.validate_index} outside 0..{len(wavs)-1}")
        val_audio, val_sr = torchaudio.load(str(wavs[args.validate_index]))
        if val_sr != cfg.sample_rate:
            val_audio = torchaudio.functional.resample(val_audio, val_sr, cfg.sample_rate)
        val_audio = val_audio[:1].to(dev)
        with torch.inference_mode(), amp_context(args.precision):
            model.set_fast_inference(False)
            eager_codes = model.generate(
                reference_audio=ref_audio, content_audio=val_audio,
                show_progress=False, fuse_codebook_heads=False,
                use_cuda_graph=False, **gen_kwargs)
            eager_audio = model.decode_to_audio(eager_codes).float()
            model.set_fast_inference(not args.no_fused_qkv)
            fast_codes = model.generate(
                reference_audio=ref_audio, content_audio=val_audio,
                show_progress=False, fuse_codebook_heads=not args.no_fused_heads,
                use_cuda_graph=not args.no_cuda_graph, **gen_kwargs)
            fast_audio = model.decode_to_audio(fast_codes).float()
        sync()
        match = (eager_codes == fast_codes).float()
        per_cb = match.mean(dim=(0, 2)).cpu().tolist()
        diff = eager_audio - fast_audio
        snr = 10.0 * torch.log10(
            eager_audio.square().mean() / diff.square().mean().clamp_min(1e-12))
        print("[validate] code match total="
              f"{match.mean().item():.6f} per_cb={[round(x, 6) for x in per_cb]}")
        print(f"[validate] audio L1={diff.abs().mean().item():.8f} "
              f"max={diff.abs().max().item():.8f} SNR={snr.item():.2f} dB")
        if args.validate_only:
            return

    rows = []
    with torch.inference_mode():
        for i, wav_path in enumerate(wavs):
            audio, sr = torchaudio.load(str(wav_path))
            if sr != cfg.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, cfg.sample_rate)
            audio = audio[:1].to(dev)
            input_dur = audio.shape[-1] / cfg.sample_rate

            do_profile = args.profile and i == args.warmup
            prof_ctx = (
                torch.profiler.profile(activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                )) if do_profile else nullcontext()
            )
            sync()
            t0 = time.perf_counter()
            with prof_ctx as prof, amp_context(args.precision):
                codes = model.generate(
                    reference_audio=ref_audio,
                    content_audio=audio,
                    show_progress=args.show_progress,
                    fuse_codebook_heads=not args.no_fused_heads,
                    use_cuda_graph=not args.no_cuda_graph,
                    **gen_kwargs,
                )
            sync()
            gen_s = time.perf_counter() - t0
            if do_profile:
                print(prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=30))

            sync()
            t0 = time.perf_counter()
            with amp_context(args.precision):
                output = model.decode_to_audio(codes)
            sync()
            dec_s = time.perf_counter() - t0
            output_dur = output.shape[-1] / cfg.sample_rate

            if i < args.warmup:
                print(f"  [warmup {i + 1}] {input_dur:.2f}s {codes.shape[-1]}fr "
                      f"gen={gen_s:.3f}s dec={dec_s:.3f}s")
                continue
            rows.append((input_dur, output_dur, codes.shape[-1], gen_s, dec_s))
            print(f"  [{i + 1 - args.warmup:>2}/{args.n}] {input_dur:.2f}s "
                  f"gen={gen_s:.3f}s dec={dec_s:.3f}s RTF={(gen_s + dec_s) / input_dur:.4f}")

    in_s = sum(r[0] for r in rows)
    out_s = sum(r[1] for r in rows)
    frames = sum(r[2] for r in rows)
    gen_s = sum(r[3] for r in rows)
    dec_s = sum(r[4] for r in rows)
    total_s = gen_s + dec_s
    print("\n" + "=" * 64)
    print(f"{len(rows)} utts, input={in_s:.2f}s output={out_s:.2f}s frames={frames}")
    print(f"generate RTF = {gen_s / in_s:.4f}")
    print(f"codec_dec RTF = {dec_s / in_s:.4f}")
    print(f"TOTAL model RTF (input denominator) = {total_s / in_s:.4f}")
    print(f"TOTAL output RTF (output denominator) = {total_s / out_s:.4f}")
    print(f"ms/AR step = {1000 * gen_s / frames:.2f}")


if __name__ == "__main__":
    main()
