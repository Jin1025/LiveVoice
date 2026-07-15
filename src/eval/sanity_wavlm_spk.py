"""Sanity-check the WavLM-TDNN speaker embedder.

Why: val/spk_sim_gt with WavLM comes out ~0.6 — barely above ECAPA (~0.58). WavLM-large
finetune should sit on a clearly HIGHER same-speaker scale (~0.7+). If it doesn't, the
most likely cause is that the finetuned checkpoint did not actually load into the model
(``load_state_dict(strict=False)`` silently drops keys whose names don't match), leaving a
base/partly-random encoder.

This script checks three things, no training epoch needed:

  1. **Checkpoint key overlap** — how many of the model's params were actually filled by
     the .pth. If matched ≪ total (esp. if the WavLM ``feature_extract.*`` backbone didn't
     load), that is the bug and explains the low GT.
  2. **Speaker separation** — mean cosine on same-speaker pairs (two different utterances of
     one speaker = the GT ceiling) vs different-speaker pairs (floor). A healthy encoder:
     same ≫ diff. A broken one: same ≈ diff, both mushy.
  3. **ECAPA side-by-side** on the *same* pairs — so you can see whether WavLM is genuinely
     on a higher scale or has collapsed to ECAPA-like numbers.

Run (env with s3prl + speechbrain + the finetuned ckpt — e.g. conda activate sound):

    CUDA_VISIBLE_DEVICES=0 python src/eval/sanity_wavlm_spk.py \
        --wavlm_ckpt /mnt/data/disk3/yejin/wavlm_large_finetune.pth \
        --n_speakers 30 --also_ecapa
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import torchaudio

from livevoice.config import LiveVoiceConfig
from livevoice.data.libritts_dataset import LibriTTSDataset


def _load_full_mono_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Mono utterance at target_sr, peak-normalized (16 kHz for the SV encoders)."""
    try:
        import soundfile as sf
        with sf.SoundFile(path) as f:
            audio_np = f.read(dtype="float32", always_2d=True)
            sr = int(f.samplerate)
        audio = torch.from_numpy(audio_np).float().mean(dim=1)
    except Exception:
        import librosa
        audio_np, sr = librosa.load(path, sr=None, mono=True)
        audio = torch.from_numpy(audio_np.astype("float32"))
        sr = int(sr)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio / (audio.abs().max() + 1e-8)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1).item())


def report_ckpt_overlap(model: torch.nn.Module, ckpt_path: str) -> None:
    """Print how many model params the .pth actually fills — the key diagnostic."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    matched = model_keys & ckpt_keys
    missing = sorted(model_keys - ckpt_keys)      # model wants, ckpt lacks
    unexpected = sorted(ckpt_keys - model_keys)   # ckpt has, model lacks
    # How much of the WavLM backbone (feature_extract.*) actually loaded?
    fe_model = {k for k in model_keys if k.startswith("feature_extract.")}
    fe_matched = {k for k in matched if k.startswith("feature_extract.")}
    head_model = {k for k in model_keys if not k.startswith("feature_extract.")}
    head_matched = {k for k in matched if not k.startswith("feature_extract.")}
    print("=" * 72)
    print("[keys] checkpoint vs model overlap")
    print(f"  ckpt keys            : {len(ckpt_keys)}")
    print(f"  model keys           : {len(model_keys)}")
    print(f"  matched (loaded)     : {len(matched)}")
    print(f"  missing (not loaded) : {len(missing)}")
    print(f"  unexpected (ignored) : {len(unexpected)}")
    print(f"  WavLM backbone loaded: {len(fe_matched)}/{len(fe_model)} "
          f"(feature_extract.*)")
    print(f"  ECAPA head loaded    : {len(head_matched)}/{len(head_model)}")
    if missing:
        print(f"  e.g. missing : {missing[:4]}")
    if unexpected:
        print(f"  e.g. unexpected : {unexpected[:4]}")
    # Verdict
    frac = len(matched) / max(1, len(model_keys))
    if frac < 0.9 or len(fe_matched) < 0.9 * len(fe_model):
        print("  >>> LOAD LOOKS BROKEN: a large fraction of params did NOT load. "
              "This alone explains a low GT. Fix key naming before trusting the number.")
    else:
        print("  >>> load looks OK (nearly all params filled).")
    print("=" * 72)


def embed_all(embed_fn, paths: list[str], device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for p in paths:
        wav = _load_full_mono_wav(p, 16000).to(device)
        out[p] = embed_fn(wav).detach().reshape(-1).cpu()
    return out


def pair_stats(name: str, embs: dict[str, torch.Tensor],
               same_pairs: list[tuple[str, str]], diff_pairs: list[tuple[str, str]]) -> None:
    same = [_cos(embs[a], embs[b]) for a, b in same_pairs]
    diff = [_cos(embs[a], embs[b]) for a, b in diff_pairs]
    # identity: an utterance vs itself must be ~1.0 (embedder determinism check)
    any_path = next(iter(embs))
    ident = _cos(embs[any_path], embs[any_path])
    sm = sum(same) / len(same)
    dm = sum(diff) / len(diff)
    print(f"[{name}]")
    print(f"  same-speaker (GT ceiling): {sm:.4f}   (n={len(same)})")
    print(f"  diff-speaker (floor)     : {dm:.4f}   (n={len(diff)})")
    print(f"  separation (same - diff) : {sm - dm:.4f}")
    print(f"  identity cos(x,x)        : {ident:.4f}  (should be ~1.0)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--libritts_path", type=str, default="/mnt/data/disk2/LibriTTS")
    p.add_argument("--split_dir", type=str, default="dev-clean")
    p.add_argument("--wavlm_ckpt", type=str,
                   default="/mnt/data/disk3/yejin/wavlm_large_finetune.pth")
    p.add_argument("--wavlm_variant", type=str, default="wavlm_large",
                   choices=["wavlm_large", "wavlm_base_plus"])
    p.add_argument("--n_speakers", type=int, default=30,
                   help="#speakers to sample (each contributes 1 same-pair + 1 diff-pair).")
    p.add_argument("--also_ecapa", action="store_true",
                   help="Also run SpeechBrain ECAPA on the same pairs for a scale comparison.")
    p.add_argument("--match_training", action="store_true",
                   help="Replicate on_train_epoch_end EXACTLY: the wer_seed-seeded N content "
                        "utterances, each paired with the speaker's FIRST other utterance "
                        "(deterministic), full audio, WavLM. Reproduces the val/spk_sim_gt number.")
    p.add_argument("--wer_seed", type=int, default=12345,
                   help="Must match config.wer_seed for --match_training (default 12345).")
    p.add_argument("--wer_epoch_samples", type=int, default=50,
                   help="Must match config.wer_epoch_samples for --match_training (default 50).")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # ── Path-only dataset → speaker → utterances map ──
    cfg = LiveVoiceConfig(
        device="cuda:0" if device.type == "cuda" else "cpu",
        libritts_path=args.libritts_path,
        libritts_val_splits=(args.split_dir,),
        codec="jhcodec",
        sample_rate=16000,
        wavlm_sv_ckpt=args.wavlm_ckpt,
        wavlm_sv_variant=args.wavlm_variant,
        features_dir=None,
    )
    ds = LibriTTSDataset(cfg, split="val")

    # ── Mode: replicate on_train_epoch_end's exact pairing to reproduce val/spk_sim_gt ──
    if args.match_training:
        def _pick_ref(content_wav, utt_id, spk):
            spk_utts = ds.speaker_utts[spk]
            rp, _ = next(((p, u) for p, u in spk_utts if p != content_wav), (content_wav, utt_id))
            return rp
        n = max(1, min(int(args.wer_epoch_samples), len(ds.items)))
        idxs = sorted(random.Random(int(args.wer_seed)).sample(range(len(ds.items)), n))
        train_pairs = []
        paths_t: set[str] = set()
        for ix in idxs:
            wav_path, utt_id, spk = ds.items[ix]
            rp = _pick_ref(wav_path, utt_id, spk)
            train_pairs.append((wav_path, rp))
            paths_t.add(wav_path); paths_t.add(rp)
        print(f"[sanity] MATCH-TRAINING: {len(train_pairs)} pairs "
              f"(wer_seed={args.wer_seed}, n={n}), first-other-utt pairing, full audio")
        from livevoice.evaluation.unispeech_sv import UniSpeechWavLMTDNNEmbedder
        wavlm = UniSpeechWavLMTDNNEmbedder(
            checkpoint=args.wavlm_ckpt, device=str(device), variant=args.wavlm_variant)
        report_ckpt_overlap(wavlm.model, args.wavlm_ckpt)
        embs = embed_all(lambda w: wavlm.embed(w), sorted(paths_t), device)
        vals = [_cos(embs[a], embs[b]) for a, b in train_pairs]
        import statistics
        mean_v = sum(vals) / len(vals)
        print(f"[WavLM-TDNN | match-training] spk_sim_gt = {mean_v:.4f}  (n={len(vals)})")
        print(f"  min={min(vals):.3f}  max={max(vals):.3f}  median={statistics.median(vals):.3f}")
        print("  → 이 값이 학습 로그의 val/spk_sim_gt와 비슷하면, 0.66과의 차이는 "
              "pairing(first-other-utt) + 50샘플셋 탓 (버그 아님).")
        return

    spk_utts = {s: u for s, u in ds.speaker_utts.items() if len(u) >= 2}
    speakers = sorted(spk_utts.keys())
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    speakers = speakers[: args.n_speakers]
    if len(speakers) < 2:
        raise SystemExit("Need >=2 speakers with >=2 utterances each.")

    # Build same / diff pairs and the set of paths to embed.
    same_pairs: list[tuple[str, str]] = []
    diff_pairs: list[tuple[str, str]] = []
    paths: set[str] = set()
    for i, s in enumerate(speakers):
        (a, _), (b, _) = rng.sample(spk_utts[s], 2)          # two utts, same speaker
        same_pairs.append((a, b))
        paths.add(a); paths.add(b)
        other = speakers[(i + 1) % len(speakers)]            # a different speaker
        (c, _) = rng.choice(spk_utts[other])
        diff_pairs.append((a, c))
        paths.add(c)
    paths = sorted(paths)
    print(f"[sanity] {len(speakers)} speakers, {len(same_pairs)} same-pairs, "
          f"{len(diff_pairs)} diff-pairs, {len(paths)} utterances ({args.split_dir})")

    # ── WavLM-TDNN embedder (+ key overlap report) ──
    from livevoice.evaluation.unispeech_sv import UniSpeechWavLMTDNNEmbedder
    print(f"[sanity] building WavLM-TDNN ({args.wavlm_variant}) from {args.wavlm_ckpt} ...")
    wavlm = UniSpeechWavLMTDNNEmbedder(
        checkpoint=args.wavlm_ckpt, device=str(device), variant=args.wavlm_variant
    )
    report_ckpt_overlap(wavlm.model, args.wavlm_ckpt)

    wavlm_embs = embed_all(lambda w: wavlm.embed(w), paths, device)
    pair_stats("WavLM-TDNN", wavlm_embs, same_pairs, diff_pairs)

    # ── ECAPA on the same pairs (optional scale reference) ──
    if args.also_ecapa:
        from livevoice.model.speechbrain_speaker_encoder import SpeechBrainECAPASpeakerEncoder
        print("[sanity] building SpeechBrain ECAPA ...")
        ecapa = SpeechBrainECAPASpeakerEncoder(cfg).to(device).eval()
        ecapa_embs = embed_all(lambda w: ecapa(w.unsqueeze(0)), paths, device)
        pair_stats("ECAPA", ecapa_embs, same_pairs, diff_pairs)

    print("\n[sanity] interpretation:")
    print("  • WavLM same-speaker should be clearly HIGHER than ECAPA same-speaker.")
    print("  • If WavLM same ≈ ECAPA same (~0.58-0.60) AND/OR key overlap was low →")
    print("    the finetuned ckpt did not load properly (key-name mismatch) = the bug.")
    print("  • If WavLM same ~0.70+ with big same-diff separation → load is fine and the")
    print("    0.6 you saw earlier was something else (e.g. wrong pairing / different run).")


if __name__ == "__main__":
    main()
