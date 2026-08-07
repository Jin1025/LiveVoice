#!/usr/bin/env python3
"""Speaker similarity (WavLM-TDNN) — GT ceiling/floor and/or your VC model.

Embedder is ALWAYS UniSpeech WavLM-large + finetuned ECAPA-TDNN head
(github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification),
loaded as ``--model_name wavlm_large`` (feat_dim=1024, WavLM upstream).

Two axes:
  --mode  {same, cross}     which speaker the reference comes from
  --gt                      measure GROUND-TRUTH s-sim only (no model): cos(real
                            content utt, ref utt). same→intra-spk ceiling (~0.66),
                            cross→inter-spk floor (~0.10). Verify this is sane
                            BEFORE trusting model numbers.
  (default, no --gt)        run your VC ckpt: cos(gen, ref) AND the GT ceiling
                            cos(content, ref) side by side.

Datasets (same speaker/chapter/utt layout):
  --dataset libritts    (default)  root/<split>/<spk>/<chap>/*.wav
  --dataset librispeech            root/<split>/<spk>/<chap>/*.flac

Examples:
  # 1) GT sanity first (no model), WavLM-TDNN, LibriSpeech/LibriTTS test-clean:
  python eval/eval_s-sim.py --gt --mode same  --dataset librispeech  --root /mnt/data/disk2/LibriSpeech --split test-clean 
  python eval/eval_s-sim.py --gt --mode same --dataset libritts --root /mnt/data/disk2/LibriTTS --split test-clean

  # 2) Your model (cross-speaker VC), GT ceiling logged alongside:
  CUDA_VISIBLE_DEVICES=0 python eval/eval_s-sim.py --mode cross --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/perturbed_prefix_ecapa_hubert/step_latest.ckpt --out_csv /mnt/data/disk2/yejin/LiveVoice/exp/ssim_<run>_cross.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.evaluation.unispeech_sv import UniSpeechWavLMTDNNEmbedder

EMBED_SR = 16000  # WavLM upstream is 16 kHz


# ──────────────────────────────────────────────────────────────────────
#  Audio
# ──────────────────────────────────────────────────────────────────────
def _load_full_mono_wav(path: str, target_sr: int) -> torch.Tensor:
    try:
        with sf.SoundFile(path) as f:
            audio_np = f.read(dtype="float32", always_2d=True)
            sr = int(f.samplerate)
        audio = torch.from_numpy(audio_np).float().mean(dim=1)
    except Exception:
        audio_np, sr = librosa.load(path, sr=None, mono=True)
        audio = torch.from_numpy(audio_np.astype("float32"))
        sr = int(sr)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio / (audio.abs().max() + 1e-8)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


# ──────────────────────────────────────────────────────────────────────
#  Dataset: LibriTTS (.wav) / LibriSpeech (.flac), speaker/chapter/utt layout
# ──────────────────────────────────────────────────────────────────────
def _duration_sec(path: str) -> float:
    """Utterance length in seconds from the header only (no decode)."""
    try:
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        return -1.0


def _build_speaker_utts(dataset, root, split, seed, max_items, min_dur=None, max_dur=None):
    base = Path(root) / split
    if not base.exists():
        raise SystemExit(f"dataset split not found: {base}")
    ext = "flac" if dataset == "librispeech" else "wav"
    files = sorted(base.glob(f"**/*.{ext}"))
    if not files:
        raise SystemExit(f"no *.{ext} under {base}")

    lo = float(min_dur) if min_dur else None
    hi = float(max_dur) if max_dur else None
    do_filter = lo is not None or hi is not None

    speaker_utts: dict[str, list[tuple[str, str]]] = {}
    n_kept = total_sec = 0
    for wav in tqdm(files, desc="scan durations") if do_filter else [(w) for w in files]:
        if do_filter:
            d = _duration_sec(str(wav))
            if d < 0 or (lo is not None and d < lo) or (hi is not None and d > hi):
                continue
            n_kept += 1
            total_sec += d
        speaker_utts.setdefault(wav.parts[-3], []).append((str(wav), wav.stem))
    if not speaker_utts:
        raise SystemExit(f"no *.{ext} left after duration filter [{lo}, {hi}]s under {base}")
    if do_filter:
        print(f"[s-sim] duration filter [{lo}, {hi}]s → kept {n_kept}/{len(files)} "
              f"utts ({total_sec/3600:.2f} h)")

    items = [(p, u, s) for s, us in speaker_utts.items() for (p, u) in us]
    random.Random(seed).shuffle(items)
    if max_items:
        items = items[: int(max_items)]
    return items, speaker_utts


def _pick_ref(speaker_utts, content_spk, content_path, mode, rng):
    """same → different utt of the SAME speaker; cross → any utt of ANOTHER speaker."""
    if mode == "same":
        cands = [(p, u) for (p, u) in speaker_utts[content_spk] if p != content_path]
        if not cands:
            return None, None
        p, _ = rng.choice(cands)
        return p, content_spk
    others = [s for s in speaker_utts if s != content_spk]
    if not others:
        return None, None
    rs = rng.choice(others)
    p, _ = rng.choice(speaker_utts[rs])
    return p, rs


# ──────────────────────────────────────────────────────────────────────
#  VC model (jhcodec + sw2v/hubert + ecapa/spark_global), auto from ckpt
# ──────────────────────────────────────────────────────────────────────
def _build_vc_model(ckpt: str, device: torch.device, args):
    from livevoice.model import (
        HuBERTContentExtractor,
        StreamVoiceAnonContentEncoder,
        Sw2vContentEncoder,
        LiveVoiceModel,
        build_codec,
    )
    from livevoice.lightning import LiveVoiceLightningModule
    from livevoice.utils.checkpoint import (
        infer_content_source_from_ckpt,
        infer_speaker_conditioning_from_ckpt,
        infer_speaker_encoder_from_ckpt,
        infer_content_fsq_from_ckpt,
        infer_codec_prompt_flags_from_ckpt,
    )

    content_source = infer_content_source_from_ckpt(ckpt) or "hubert"
    speaker_conditioning = infer_speaker_conditioning_from_ckpt(ckpt) or "prefix"
    speaker_encoder_type = infer_speaker_encoder_from_ckpt(ckpt) or "codec"
    fsq_levels = infer_content_fsq_from_ckpt(ckpt)
    cp_flags = infer_codec_prompt_flags_from_ckpt(ckpt)
    print(
        f"[s-sim] ckpt inferred: content_source={content_source} "
        f"speaker_conditioning={speaker_conditioning} speaker_encoder_type={speaker_encoder_type} "
        f"content_fsq={fsq_levels if fsq_levels else 'off'} codec_prompt={cp_flags}"
    )

    cfg = LiveVoiceConfig(
        **cp_flags,
        device=str(device),
        codec="jhcodec",
        sample_rate=16000,
        hidden_dim=int(args.hidden_dim),
        num_decoder_layers=int(args.num_decoder_layers),
        ffn_dim=4 * int(args.hidden_dim),
        n_codebooks_predict=int(args.n_codebooks),
        content_source=content_source,
        speaker_conditioning=speaker_conditioning,
        speaker_prefix_len=int(args.speaker_prefix_len),
        speaker_encoder_type=speaker_encoder_type,
        use_content_fsq=fsq_levels is not None,
        fsq_levels=fsq_levels if fsq_levels is not None else (8, 5, 5, 5),
        features_dir=None,
    )
    codec_model = build_codec(cfg)
    if cfg.content_source == "hubert":
        content_extractor = HuBERTContentExtractor(cfg)
    elif cfg.content_source == "streamvoiceanon":
        content_extractor = StreamVoiceAnonContentEncoder(cfg)
    elif cfg.content_source == "sw2v":
        content_extractor = Sw2vContentEncoder(cfg)
    else:
        content_extractor = None
    core = LiveVoiceModel(cfg, codec_model, content_extractor, prosody_extractor=None)
    lit = LiveVoiceLightningModule.load_from_checkpoint(ckpt, config=cfg, model=core, strict=False)
    lit.eval()
    return lit.to(device), cfg


# ──────────────────────────────────────────────────────────────────────
#  Main dataset run
# ──────────────────────────────────────────────────────────────────────
def run(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[s-sim] Loading WavLM-TDNN ({args.wavlm_variant}) from {args.wavlm_ckpt!r} ...")
    embedder = UniSpeechWavLMTDNNEmbedder(
        checkpoint=args.wavlm_ckpt, device=str(device), variant=args.wavlm_variant
    )

    items, speaker_utts = _build_speaker_utts(
        args.dataset, args.root, args.split, int(args.seed), args.max_items,
        min_dur=args.min_dur, max_dur=args.max_dur,
    )
    rng = random.Random(int(args.ref_seed))
    print(
        f"[s-sim] dataset={args.dataset} split={args.split} mode={args.mode} "
        f"gt_only={bool(args.gt)} utts={len(items)} speakers={len(speaker_utts)}"
    )

    lit = model_sr = None
    if not args.gt:
        if not args.ckpt:
            raise SystemExit("model mode requires --ckpt (or pass --gt for GT-only).")
        print("[s-sim] Building VC model ...")
        lit, cfg = _build_vc_model(args.ckpt, device, args)
        model_sr = int(cfg.sample_rate)
        content_max_sec = float(args.max_content_sec)
        keep_full_ref = str(cfg.speaker_encoder_type).lower() in ("speechbrain_ecapa", "spark_global")

    out_csv = args.out_csv
    if os.path.dirname(out_csv):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = [
        "content_wav", "ref_wav", "content_speaker", "ref_speaker", "mode",
        "cosine_similarity", "cosine_similarity_gt", "cosine_similarity_src", "error",
    ]
    sims_model: list[float] = []   # gen vs ref  (toward TARGET — want high)
    sims_gt: list[float] = []      # content vs ref (GT ceiling/floor)
    sims_src: list[float] = []     # gen vs content (toward SOURCE — want low)

    done: set[str] = set()
    resume = bool(getattr(args, "resume", False)) and os.path.isfile(out_csv)
    if resume:
        with open(out_csv, newline="", encoding="utf-8") as fprev:
            for r in csv.DictReader(fprev):
                cw = (r.get("content_wav") or "").strip()
                if cw:
                    done.add(cw)
                    # keep running means for final summary
                    try:
                        if r.get("cosine_similarity"):
                            sims_model.append(float(r["cosine_similarity"]))
                        if r.get("cosine_similarity_gt"):
                            sims_gt.append(float(r["cosine_similarity_gt"]))
                        if r.get("cosine_similarity_src"):
                            sims_src.append(float(r["cosine_similarity_src"]))
                    except ValueError:
                        pass
        print(f"[s-sim] resume: {len(done)}/{len(items)} already in {out_csv} "
              f"→ {len(items) - len(done)} remaining")

    mode = "a" if resume else "w"
    with open(out_csv, mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if not resume:
            writer.writeheader()
        for content_wav, utt_id, content_spk in tqdm(items, desc=f"s-sim {args.mode}"):
            # Always pick ref first so RNG advances identically on resume.
            ref_wav, ref_spk = _pick_ref(speaker_utts, content_spk, content_wav, args.mode, rng)
            if content_wav in done:
                continue
            row = {
                "content_wav": content_wav, "ref_wav": ref_wav or "",
                "content_speaker": content_spk, "ref_speaker": ref_spk or "",
                "mode": args.mode, "cosine_similarity": "", "cosine_similarity_gt": "",
                "cosine_similarity_src": "",
                "error": "" if ref_wav else "no_ref",
            }
            if not ref_wav:
                writer.writerow(row); fcsv.flush(); continue

            try:
                ref16 = _load_full_mono_wav(ref_wav, EMBED_SR)
                ctn16 = _load_full_mono_wav(content_wav, EMBED_SR)
                e_ref = embedder.embed(ref16)
                e_ctn = embedder.embed(ctn16)      # source-speaker embedding
                sim_gt = _cosine(e_ctn, e_ref)
                sims_gt.append(sim_gt)
                row["cosine_similarity_gt"] = f"{sim_gt:.6f}"

                if not args.gt:
                    ctn = _load_full_mono_wav(content_wav, model_sr)
                    if ctn.numel() > int(content_max_sec * model_sr):
                        row["error"] = "content_too_long_skipped"
                        writer.writerow(row); fcsv.flush(); continue
                    ctn = ctn.unsqueeze(0).to(device)
                    ref = _load_full_mono_wav(ref_wav, model_sr).unsqueeze(0).to(device)
                    if not keep_full_ref:
                        rmax = int(4.0 * model_sr)
                        if ref.shape[-1] > rmax:
                            ref = ref[..., :rmax]
                    with torch.no_grad():
                        codes = lit.model.generate(
                            reference_audio=ref, content_audio=ctn,
                            temperature=0.0, top_p=None, top_k=None,
                            cfg_scale=float(getattr(args, "cfg_scale", 1.0)),
                        )
                        gen = lit.model.decode_to_audio(codes)
                    gen16 = gen.squeeze(0).detach().float().cpu()
                    if model_sr != EMBED_SR:
                        gen16 = torchaudio.functional.resample(gen16, model_sr, EMBED_SR)
                    e_gen = embedder.embed(gen16)
                    sim_model = _cosine(e_gen, e_ref)      # gen → TARGET ref
                    sim_src = _cosine(e_gen, e_ctn)        # gen → SOURCE content
                    sims_model.append(sim_model)
                    sims_src.append(sim_src)
                    row["cosine_similarity"] = f"{sim_model:.6f}"
                    row["cosine_similarity_src"] = f"{sim_src:.6f}"
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            except Exception as e:
                row["error"] = str(e)[:500]
            writer.writerow(row); fcsv.flush()

    print(f"\n[s-sim] dataset={args.dataset} mode={args.mode}  (WavLM-TDNN)")
    if sims_model:
        print(f"[s-sim]   MODEL  cos(gen, TARGET ref):  {sum(sims_model)/len(sims_model):.4f}  (n={len(sims_model)})  ↑ want high")
    if sims_src:
        print(f"[s-sim]   MODEL  cos(gen, SOURCE ctn):  {sum(sims_src)/len(sims_src):.4f}  (n={len(sims_src)})  ↓ want low")
    if sims_gt:
        tag = "intra-spk ceiling" if args.mode == "same" else "inter-spk floor"
        print(f"[s-sim]   GT     cos(content, ref):     {sum(sims_gt)/len(sims_gt):.4f}  (n={len(sims_gt)})  = {tag}")
    if sims_model and sims_src:
        gap = sum(sims_model)/len(sims_model) - sum(sims_src)/len(sims_src)
        verdict = "adopts TARGET" if gap > 0 else "stuck on SOURCE"
        print(f"[s-sim]   → target−source gap = {gap:+.4f}  ({verdict})")
    print(f"[s-sim] wrote: {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Speaker cosine similarity (WavLM-TDNN) — GT and/or VC model.")
    # what to measure
    p.add_argument("--mode", type=str, default="cross", choices=["same", "cross"])
    p.add_argument("--gt", action="store_true", help="GT s-sim only (no model): cos(content, ref).")
    p.add_argument("--ckpt", type=str, default=None, help="VC checkpoint (required unless --gt).")
    # dataset
    p.add_argument("--dataset", type=str, default="libritts", choices=["libritts", "librispeech"])
    p.add_argument("--root", type=str, default="/mnt/data/disk2/LibriTTS",
                   help="Dataset root (LibriSpeech: /mnt/data/disk2/LibriSpeech).")
    p.add_argument("--split", type=str, default="test-clean")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--min_dur", type=float, default=None, help="Keep utts >= this many seconds (e.g. 4).")
    p.add_argument("--max_dur", type=float, default=None, help="Keep utts <= this many seconds (e.g. 10).")
    p.add_argument("--seed", type=int, default=42, help="Item shuffle seed.")
    p.add_argument("--ref_seed", type=int, default=12345, help="Reference-pick seed (stable across runs).")
    # WavLM-TDNN embedder
    p.add_argument("--wavlm_ckpt", type=str, default="/mnt/data/disk3/yejin/wavlm_large_finetune.pth")
    p.add_argument("--wavlm_variant", type=str, default="wavlm_large", choices=["wavlm_large", "wavlm_base_plus"])
    # VC model shape (only used with a ckpt)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_decoder_layers", type=int, default=12)
    p.add_argument("--n_codebooks", type=int, default=8)
    p.add_argument("--speaker_prefix_len", type=int, default=4)
    # Default follows config.val_cfg_scale so eval matches what validation reports.
    p.add_argument("--cfg_scale", type=float, default=LiveVoiceConfig.val_cfg_scale)
    p.add_argument("--max_content_sec", type=float, default=15.0, help="Skip content longer than this (context limit).")
    # output
    p.add_argument("--out_csv", type=str, default="/mnt/data/disk2/yejin/LiveVoice/ssim.csv")
    p.add_argument("--resume", action="store_true",
                   help="Append missing rows to existing --out_csv (skip done content_wav). "
                        "Keeps ref RNG in sync with a full run.")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
