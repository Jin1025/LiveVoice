"""Compare batch inference tokens with stateful chunked-Zipformer tokens.

This first stage keeps waveform alignment, fbank, and Conv2dSubsampling in their
existing batch form and makes only Zipformer itself truly stateful/chunked.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
import torchaudio


def load_wav(path: Path, sample_rate: int, device: torch.device) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav[:1]
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.to(device)


def first_mismatch(a: torch.Tensor, b: torch.Tensor) -> str:
    mismatch = (a != b).nonzero(as_tuple=False)
    if mismatch.numel() == 0:
        return "none"
    batch, codebook, frame = mismatch[0].tolist()
    return (
        f"batch={batch} codebook={codebook} frame={frame} "
        f"batch_token={a[batch, codebook, frame].item()} "
        f"stream_token={b[batch, codebook, frame].item()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--content_wav", required=True)
    parser.add_argument("--reference_wav", default="")
    parser.add_argument("--zipformer_ckpt", default="")
    parser.add_argument("--jhcodec_ckpt", default="")
    parser.add_argument("--jhcodec_config", default="")
    parser.add_argument("--jhcodec_repo", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input_chunk_ms", type=int, default=320)
    args = parser.parse_args()

    from livevoice.config import LiveVoiceConfig
    from livevoice.utils.checkpoint import read_config_from_ckpt
    from eval.anonymize_vpc_dirs import _build_vc_model

    class FakeArgs:
        ckpt = args.ckpt

    cfg_dict = read_config_from_ckpt(args.ckpt)
    known = {field.name for field in dataclasses.fields(LiveVoiceConfig)}
    filtered = {key: value for key, value in cfg_dict.items() if key in known}
    for name in ("zipformer_ckpt", "jhcodec_ckpt", "jhcodec_config", "jhcodec_repo"):
        value = getattr(args, name)
        if value:
            filtered[name] = value
    if args.jhcodec_repo:
        filtered["sw2v_repo"] = args.jhcodec_repo
    cfg = LiveVoiceConfig(**filtered)

    device = torch.device(args.device)
    lit = _build_vc_model(FakeArgs(), cfg, device)
    model = lit.model.eval()
    if model.content_source != "zipformer":
        raise SystemExit(f"checkpoint content_source={model.content_source!r}, expected zipformer")

    content_path = Path(args.content_wav)
    reference_path = Path(args.reference_wav) if args.reference_wav else content_path
    content = load_wav(content_path, cfg.sample_rate, device)
    reference = load_wav(reference_path, cfg.sample_rate, device)
    extractor = model.content_extractor

    top_k = getattr(cfg, "top_k", None)
    if top_k != 1:
        print(f"[warning] overriding checkpoint top_k={top_k!r} with top_k=1 for determinism")
    gen_kwargs = dict(
        temperature=float(getattr(cfg, "temperature", 1.0)),
        top_k=1,
        top_p=None,
        cfg_scale=1.0,
        show_progress=False,
        fuse_codebook_heads=False,
        use_cuda_graph=False,
    )

    with torch.inference_mode():
        batch_feats = extractor(content)
        stream_feats = extractor.forward_encoder_streaming(content)
        feat_diff = (batch_feats - stream_feats).abs()
        print(
            f"[features] shape={tuple(batch_feats.shape)} "
            f"mean_abs={feat_diff.mean().item():.8g} "
            f"max_abs={feat_diff.max().item():.8g}"
        )

        batch_codes = model.generate(
            reference_audio=reference, content_audio=content, **gen_kwargs)
        stream_codes = model.generate(
            reference_audio=reference,
            content_audio=content,
            content_feats=stream_feats,
            **gen_kwargs,
        )

        raw_chunks = extractor.waveform_streaming_chunks(
            content,
            input_chunk_samples=args.input_chunk_ms * cfg.sample_rate // 1000,
        )
        raw_stream_feats = torch.cat(raw_chunks, dim=1)
        raw_feat_diff = (batch_feats - raw_stream_feats).abs()
        print(
            f"[raw-stream features] chunks={[x.shape[1] for x in raw_chunks]} "
            f"mean_abs={raw_feat_diff.mean().item():.8g} "
            f"max_abs={raw_feat_diff.max().item():.8g}"
        )

        from scripts.streaming_inference import generate_streaming
        raw_stream_codes, emission_sizes = generate_streaming(
            model,
            reference,
            content,
            input_chunk_samples=args.input_chunk_ms * cfg.sample_rate // 1000,
        )

    matches = batch_codes == stream_codes
    per_codebook = matches.float().mean(dim=(0, 2)).cpu().tolist()
    print(
        f"[tokens] shape={tuple(batch_codes.shape)} "
        f"total_match={matches.float().mean().item():.8f} "
        f"per_codebook={[round(value, 8) for value in per_codebook]}"
    )
    print(f"[tokens] first_mismatch={first_mismatch(batch_codes, stream_codes)}")
    raw_matches = batch_codes == raw_stream_codes
    raw_per_codebook = raw_matches.float().mean(dim=(0, 2)).cpu().tolist()
    print(
        f"[raw-stream tokens] total_match={raw_matches.float().mean().item():.8f} "
        f"per_codebook={[round(value, 8) for value in raw_per_codebook]} "
        f"emissions={emission_sizes}"
    )
    print(
        f"[raw-stream tokens] first_mismatch="
        f"{first_mismatch(batch_codes, raw_stream_codes)}"
    )
    if not bool(matches.all()) or not bool(raw_matches.all()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
