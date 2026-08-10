"""VPC eval_pre-style privacy (EER) + utility (ASR/WER) for speaker anonymization.

Matches Voice-Privacy-Challenge-2026 ``eval_pre.yaml``:

  - **ASV**: ``exp/asv_ssl`` (WavLM-Large + ECAPA-TDNN), cosine EER
  - **ASR**: ``exp/asr`` (wav2vec2-large-960h + CTC), SpeechBrain EncoderASR

Attack scenarios (frozen pretrained ASV — eval_pre only, no ASV retraining):
  - **oo**  enrollment = original, trial = original
  - **oa**  enrollment = original, trial = anonymized (VC)
  - **aa**  enrollment = anonymized, trial = anonymized

Higher EER ⇒ better anonymization. Lower WER on anonymized audio ⇒ better utility.
Semi-informed ASV (eval_post) is **not** implemented here.

Pseudo-speaker assignment follows StreamVoiceAnon+ (arXiv:2603.06079): a SINGLE prompt
utterance conditions the codec LM by continuation. ``--anon_strategy`` is ``1fix`` (one fixed
target speaker, SVA+'s vctk-1fix) or ``1rnd`` (one prompt per utterance). The K-reference
embedding blending of StreamVoiceAnon 2024 is not offered — see ``PromptSelector``.

Data layout (Kaldi / VPC format)
--------------------------------
  enroll_dir/wav.scp, utt2spk, …
  trial_dir/wav.scp, utt2spk, trials
  asr_dir/text for WER (libri_dev / libri_test)

Examples
--------
  # VPC eval_pre batch, VCTK pool, SVA+ 1fix:
  CUDA_VISIBLE_DEVICES=0 python src/eval/eval_anon.py \\
      --datasets libri_dev,libri_test \\
      --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/<run>/step_latest.ckpt \\
      --anon_pool_dir /mnt/data/disk2/VCTK-Corpus/wav48 \\
      --anon_strategy 1fix --anon_fixed_spk p225 \\
      --anon_out_dir /mnt/data/disk2/yejin/LiveVoice/anon \\
      --out_csv vpc_eval_pre.csv

  # LibriTTS pool, random prompt per utterance:
  CUDA_VISIBLE_DEVICES=0 python src/eval/eval_anon.py \\
      --ckpt ... --anon_pool_dir /mnt/data/disk2/LibriTTS/train-other-500 \\
      --anon_strategy 1rnd --scenarios oo,aa
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.model import (
    HuBERTContentExtractor,
    StreamVoiceAnonContentEncoder,
    Sw2vContentEncoder,
    LiveVoiceModel,
    build_codec,
)
from livevoice.utils.checkpoint import (
    infer_codec_prompt_flags_from_ckpt,
    infer_content_source_from_ckpt,
    infer_speaker_conditioning_from_ckpt,
    infer_speaker_encoder_from_ckpt,
    load_model_weights_from_ckpt,
    read_config_from_ckpt,
)
from eval.vpc_pretrained import (
    VPCASREvaluator,
    VPCASVAttacker,
    materialize_anon_kaldi_dir,
    vpc_cosine_eer,
)

DEFAULT_VPC_ROOT = "/mnt/data/disk3/yejin/VPC"


# ──────────────────────────────────────────────────────────────────────
#  IO helpers
# ──────────────────────────────────────────────────────────────────────
def _read_kaldi(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def _resolve_wav(scp_value: str, wav_base: Path | None = None) -> str:
    if scp_value.rstrip().endswith("|"):
        raise ValueError(
            f"wav.scp pipe entries are not supported, materialise wavs first: {scp_value!r}"
        )
    p = Path(scp_value)
    if p.is_file():
        return str(p.resolve())
    if wav_base is not None:
        cand = (wav_base / scp_value).resolve()
        if cand.is_file():
            return str(cand)
    if os.path.isfile(scp_value):
        return os.path.abspath(scp_value)
    raise FileNotFoundError(
        f"wav not found for wav.scp entry {scp_value!r} (wav_base={wav_base})"
    )


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


def _read_trials(path: Path) -> list[tuple[str, str, int]]:
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if len(t) >= 3:
                trials.append((t[0], t[1], int(t[2] == "target")))
    return trials


# ──────────────────────────────────────────────────────────────────────
#  VC anonymizer
# ──────────────────────────────────────────────────────────────────────
# Fields the EVAL owns — never taken from the checkpoint. Everything else in a stored
# config describes the model that was trained and must be reproduced exactly, or we score
# the weights inside a different architecture (this is how the 2026-08-04 CFG sweep got
# invalidated: audio_duration and the codec_prompt_* flags silently came from today's
# defaults instead of from the run).
_EVAL_OWNED_FIELDS = frozenset({
    "device", "output_dir",
    # Feature caches only cover LibriTTS utterances. VPC enrol/trial wavs are not in them,
    # so every content path has to run on-the-fly here regardless of how it was trained.
    "features_dir", "sw2v_features_dir", "zipformer_features_dir",
    # Auxiliary training heads take no part in generation, and GRL additionally needs a
    # grl_num_speakers the eval has no speaker vocab for.
    "use_asr_supervision", "use_speaker_grl",
    # Training-time augmentation; inference guidance is controlled by --cfg_scale.
    "use_cfg_dropout",
})

# Decisive for reproducing the model — echoed so a wrong build is visible in the log.
_CONFIG_ECHO_FIELDS = (
    "content_source", "content_conditioning", "content_cmn", "content_cmn_in_cache",
    "content_refiner_layers", "use_content_fsq", "use_content_perturbation",
    "zipformer_layer", "zipformer_align_pad_frames",
    "speaker_encoder_type", "speaker_conditioning", "speaker_prefix_len",
    "codec_prompt_continuation", "codec_prompt_content",
    "audio_duration", "hidden_dim", "num_decoder_layers", "n_codebooks_predict",
)


def _build_vc_config(args: argparse.Namespace, device: str) -> LiveVoiceConfig:
    codec = str(args.codec).lower()
    sample_rate = 24000 if codec == "mimi" else 16000
    kw = dict(
        device=device,
        codec=codec,
        sample_rate=sample_rate,
        hidden_dim=int(args.hidden_dim),
        num_decoder_layers=int(args.num_decoder_layers),
        ffn_dim=4 * int(args.hidden_dim),
        n_codebooks_predict=int(args.n_codebooks),
        content_source=str(args.content_source).lower(),
        speaker_conditioning=str(args.speaker_conditioning).lower(),
        speaker_prefix_len=int(args.speaker_prefix_len),
        speaker_encoder_type=str(args.speaker_encoder_type).lower(),
        speechbrain_source=args.speechbrain_source,
        speechbrain_sample_rate=int(args.speechbrain_sample_rate),
        speechbrain_embedding_dim=int(args.speechbrain_embedding_dim),
        features_dir=None,
        output_dir=args.output_dir,
        use_content_perturbation=bool(args.use_content_perturbation),
    )

    stored = read_config_from_ckpt(args.ckpt) if args.ckpt else None
    if stored:
        known = {f.name for f in dataclasses.fields(LiveVoiceConfig)}
        overridden = []
        for k, v in stored.items():
            if k not in known or k in _EVAL_OWNED_FIELDS:
                continue
            if k in kw and kw[k] != v:
                overridden.append(f"{k}: {kw[k]!r} → {v!r}")
            kw[k] = v
        print(f"[eval] stored config found in ckpt ({len(stored)} fields) — using it")
        for line in overridden:
            print(f"[eval]   CLI overridden by ckpt  {line}")
    else:
        # Pre-CONFIG_CKPT_KEY checkpoint: nothing to read, so fall back to topology
        # inference. A codec_prompt_* field that did not exist yet must read as OFF, not as
        # today's default — that is what infer_codec_prompt_flags_from_ckpt encodes.
        kw.update(infer_codec_prompt_flags_from_ckpt(args.ckpt))
        print("[eval] WARNING: no stored config in this ckpt (saved before configs were "
              "checkpointed). Architecture is inferred; verify hidden_dim/audio_duration/"
              "content_* by hand before trusting the numbers.")

    cfg = LiveVoiceConfig(**kw)
    # Belt and braces: caches must be off even if a field slipped past the filter above.
    cfg.features_dir = None
    cfg.sw2v_features_dir = None
    cfg.zipformer_features_dir = None
    print("[eval] config: " + "  ".join(
        f"{f}={getattr(cfg, f, '<n/a>')}" for f in _CONFIG_ECHO_FIELDS))

    return cfg


def _auto_infer(args: argparse.Namespace) -> None:
    if str(args.content_source).lower() == "auto":
        args.content_source = infer_content_source_from_ckpt(args.ckpt) or "hubert"
        print(f"[eval] content_source=auto → {args.content_source!r}")
    if str(args.speaker_conditioning).lower() == "auto":
        args.speaker_conditioning = infer_speaker_conditioning_from_ckpt(args.ckpt) or "prefix"
        print(f"[eval] speaker_conditioning=auto → {args.speaker_conditioning!r}")
    if str(args.speaker_encoder_type).lower() == "auto":
        args.speaker_encoder_type = infer_speaker_encoder_from_ckpt(args.ckpt) or "codec"
        print(f"[eval] speaker_encoder_type=auto → {args.speaker_encoder_type!r}")


def _build_content_extractor(cfg):
    """Content encoder for the trained content_source. Must mirror train.py — a source that
    silently falls through to None hands the decoder no content at all, and the model then
    rides the AR stream: fluent audio saying the wrong words."""
    cs = str(cfg.content_source).lower()
    if cs == "hubert":
        return HuBERTContentExtractor(cfg)
    if cs == "streamvoiceanon":
        return StreamVoiceAnonContentEncoder(cfg)
    if cs == "sw2v":
        return Sw2vContentEncoder(cfg)
    if cs == "zipformer":
        from livevoice.model.zipformer_content import ZipformerContentEncoder
        layer = str(cfg.zipformer_layer)
        return ZipformerContentEncoder(
            cfg, cfg.zipformer_ckpt, layer=(layer if layer == "out" else int(layer)))
    if cs in ("mimi_semantic", "none", ""):
        return None
    raise SystemExit(
        f"[eval] content_source={cs!r} has no extractor branch here; add one rather than "
        f"running with no content encoder.")


def _build_vc_model(args, cfg, dev):
    codec_model = build_codec(cfg)
    content_extractor = _build_content_extractor(cfg)
    core = LiveVoiceModel(cfg, codec_model, content_extractor, prosody_extractor=None)
    missing, unexpected = load_model_weights_from_ckpt(core, args.ckpt, log_prefix="[eval]")
    if missing:
        print(f"[eval] warn: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"[eval] warn: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    lit = LiveVoiceLightningModule(cfg, core)
    lit.eval()
    return lit.to(dev)


def _scan_pool(root: str, wav_base: Path | None = None) -> dict[str, list[str]]:
    """{speaker: [wav, ...]} for the pseudo-speaker pool.

    Kaldi dirs use wav.scp (+utt2spk). Otherwise the speaker is the first path component
    under `root`, which covers both layouts we use:
        VCTK      <root=.../VCTK-Corpus/wav48>/p225/p225_001.wav
        LibriTTS  <root=.../LibriTTS/train-other-500>/1234/<chapter>/*.wav
    """
    p = Path(root)
    out: dict[str, list[str]] = {}
    if (p / "wav.scp").is_file():
        u2s = _read_kaldi(p / "utt2spk") if (p / "utt2spk").is_file() else {}
        for utt, v in _read_kaldi(p / "wav.scp").items():
            out.setdefault(u2s.get(utt, utt), []).append(_resolve_wav(v, wav_base))
    else:
        for ext in ("*.wav", "*.flac"):
            for x in p.rglob(ext):
                parts = x.relative_to(p).parts
                out.setdefault(parts[0] if len(parts) > 1 else "_flat", []).append(str(x))
    if not out:
        raise SystemExit(f"--anon_pool_dir has no wav.scp and no .wav/.flac under {root}")
    return {k: sorted(v) for k, v in sorted(out.items())}


class PromptSelector:
    """SVA+-style SINGLE-utterance prompt selection.

    StreamVoiceAnon+ (arXiv:2603.06079) drives a codec LM by continuation from one prompt --
    "a neutral utterance from the target anonymous speaker conceals source identity" -- and
    reports the vctk-1fix strategy, i.e. a single fixed target speaker. There is deliberately
    no K-reference blending here: StreamVoiceAnon 2024's
    g_anon = alpha * mean_i(g_i) + (1 - alpha) * g_s
    averages speaker EMBEDDINGS, which has no analogue when the reference is a codec token
    stream occupying the first T_ref positions of the same AR sequence. That is why the 4rnd
    strategies are gone rather than merely unused.

    Both strategies satisfy VPC 2026 evaluation plan v1.2 section 2.1: the pseudo-speaker
    assignment must be identical across utterances and must not rely on speaker labels.
      1fix  one fixed prompt for every trial utterance -- "Voice anonymization systems that
            assign a single pseudo-speaker to all utterances also satisfy this requirement"
      1rnd  one prompt drawn per trial utterance, seeded by the utterance id, so the random
            numbers differ per utterance as the plan requires
    A per-source-speaker mapping is intentionally absent: it would read utt2spk and so break
    the "not rely on speaker labels" rule.
    """

    def __init__(self, pool_dir: str, strategy: str, seed: int,
                 wav_base: Path | None = None,
                 fixed_spk: str | None = None, fixed_utt: str | None = None):
        self.strategy = str(strategy).lower()
        if self.strategy not in ("1fix", "1rnd"):
            raise SystemExit(f"unknown --anon_strategy {strategy!r}; expected 1fix or 1rnd")
        self.seed = int(seed)
        self.by_spk = _scan_pool(pool_dir, wav_base)
        self.all_wavs = [w for v in self.by_spk.values() for w in v]
        self.fixed: str | None = None
        if self.strategy == "1fix":
            if fixed_utt:
                self.fixed = fixed_utt
                spk = "<explicit>"
            else:
                spk = fixed_spk or random.Random(self.seed).choice(list(self.by_spk))
                if spk not in self.by_spk:
                    raise SystemExit(
                        f"--anon_fixed_spk {spk!r} not in pool; have e.g. "
                        f"{list(self.by_spk)[:8]}")
                # VCTK/LibriTTS are read speech, so any utterance is the "neutral utterance"
                # SVA+ asks for; pick deterministically so reruns reuse the same prompt.
                self.fixed = random.Random(f"{self.seed}:{spk}").choice(self.by_spk[spk])
            print(f"[anon] strategy=1fix  target speaker={spk}  prompt={self.fixed}")
        else:
            print(f"[anon] strategy=1rnd  prompt drawn per utterance from the pool")
        print(f"[anon] pool: {len(self.by_spk)} speakers, {len(self.all_wavs)} utterances "
              f"({pool_dir})")
        if len(self.by_spk) < 2:
            print("[anon] WARNING: pool has <2 speakers — for VCTK point --anon_pool_dir at "
                  "the wav48/ directory, not the corpus root.")

    def ref_for(self, utt: str) -> str:
        if self.strategy == "1fix":
            return self.fixed
        return random.Random(f"{self.seed}:{utt}").choice(self.all_wavs)


_SELECTOR_CACHE: dict[tuple, PromptSelector] = {}


def _make_selector(args) -> PromptSelector:
    """One selector per run. Cached because it must be IDENTICAL across enrolment, trials and
    train-clean-360 (1fix would otherwise pick a different prompt per call) and because
    scanning a pool like LibriTTS train-other-500 is not free."""
    key = (args.anon_pool_dir, str(args.anon_strategy).lower(), int(args.anon_seed),
           getattr(args, "anon_fixed_spk", None), getattr(args, "anon_fixed_utt", None))
    if key not in _SELECTOR_CACHE:
        _SELECTOR_CACHE[key] = PromptSelector(
            args.anon_pool_dir, args.anon_strategy, args.anon_seed, args.wav_base,
            getattr(args, "anon_fixed_spk", None), getattr(args, "anon_fixed_utt", None))
    return _SELECTOR_CACHE[key]


def _load_ref(path, target_sr, crop_sec) -> torch.Tensor:
    w = _load_full_mono_wav(path, target_sr)
    n = int(crop_sec * target_sr)
    if crop_sec and crop_sec > 0 and w.numel() > n:
        start = random.Random(f"crop:{path}").randint(0, w.numel() - n)
        w = w[start : start + n]
    return w


@torch.no_grad()
def _anonymize(lit, content_wav, ref_wav, target_sr, dev, gen_kwargs, crop_sec=0.0):
    """One content utterance + one prompt utterance -> anonymized audio."""
    ctn = _load_full_mono_wav(content_wav, target_sr).unsqueeze(0).to(dev)
    ref = _load_ref(ref_wav, target_sr, crop_sec).unsqueeze(0).to(dev)
    codes = lit.model.generate(reference_audio=ref, content_audio=ctn, **gen_kwargs)
    return lit.model.decode_to_audio(codes)[0].detach().float().cpu()


# ──────────────────────────────────────────────────────────────────────
#  Embeddings + metrics
# ──────────────────────────────────────────────────────────────────────
def _enroll_spk_embeddings(
    attacker: VPCASVAttacker,
    enroll_dir: Path,
    target_sr,
    *,
    anonymize: bool = False,
    lit=None,
    cfg=None,
    gen_kwargs=None,
    dev=None,
    args=None,
) -> dict[str, torch.Tensor]:
    wav = _read_kaldi(enroll_dir / "wav.scp")
    utt2spk = _read_kaldi(enroll_dir / "utt2spk")
    acc: dict[str, list[torch.Tensor]] = {}
    sel = _make_selector(args) if anonymize else None

    for utt, scp in tqdm(wav.items(), desc="enroll anon+emb" if anonymize else "enroll emb"):
        src = _resolve_wav(scp, args.wav_base)
        if anonymize:
            cached = (
                os.path.join(args.anon_out_dir, "enrolls", f"{utt}.wav")
                if args.anon_out_dir
                else None
            )
            if cached and os.path.isfile(cached):
                path_for_asv = cached
            elif getattr(args, "cache_only", False):
                raise FileNotFoundError(f"--cache_only: missing enroll wav {cached}")
            else:
                aud = _anonymize(
                    lit, src, sel.ref_for(utt), int(cfg.sample_rate), dev,
                    gen_kwargs, args.ref_crop_sec,
                )
                if not cached:
                    raise RuntimeError("anon_out_dir required for VPC ASV embedding")
                os.makedirs(os.path.dirname(cached), exist_ok=True)
                sf.write(cached, aud.numpy(), int(cfg.sample_rate))
                path_for_asv = cached
            emb = attacker.embed_path(path_for_asv)
        else:
            emb = attacker.embed_path(src)
        acc.setdefault(utt2spk[utt], []).append(emb)
    return {spk: torch.stack(v).mean(0) for spk, v in acc.items()}


def _trial_utt_embeddings(
    attacker: VPCASVAttacker, trial_dir: Path, anonymize: bool, lit, cfg, gen_kwargs, dev, args,
) -> dict[str, torch.Tensor]:
    wav = _read_kaldi(trial_dir / "wav.scp")
    utt2spk = _read_kaldi(trial_dir / "utt2spk")
    target_sr = int(cfg.sample_rate) if cfg is not None else attacker.sample_rate

    sel = None
    if anonymize:
        sel = _make_selector(args)
        if args.anon_out_dir:
            os.makedirs(args.anon_out_dir, exist_ok=True)

    out: dict[str, torch.Tensor] = {}
    for utt, scp in tqdm(wav.items(), desc="anon+emb" if anonymize else "trial emb"):
        src = _resolve_wav(scp, args.wav_base)
        if anonymize:
            cached = os.path.join(args.anon_out_dir, "trials", trial_dir.name, f"{utt}.wav") if args.anon_out_dir else None
            if cached and os.path.isfile(cached):
                path_for_asv = cached
            elif getattr(args, "cache_only", False):
                raise FileNotFoundError(f"--cache_only: missing trial wav {cached}")
            else:
                aud = _anonymize(
                    lit, src, sel.ref_for(utt), target_sr, dev, gen_kwargs, args.ref_crop_sec
                )
                if not cached:
                    raise RuntimeError("anon_out_dir required for VPC ASV embedding")
                os.makedirs(os.path.dirname(cached), exist_ok=True)
                sf.write(cached, aud.numpy(), target_sr)
                path_for_asv = cached
            emb = attacker.embed_path(path_for_asv)
        else:
            emb = attacker.embed_path(src)
        out[utt] = emb
    return out


def _compute_eer(trials, enroll_embs, trial_embs):
    return vpc_cosine_eer(trials, enroll_embs, trial_embs)


def _scenario_anon_flags(scenario: str) -> tuple[bool, bool]:
    if scenario == "oo":
        return False, False
    if scenario == "oa":
        return False, True
    if scenario == "aa":
        return True, True
    raise ValueError(f"Unknown scenario {scenario!r}")


def _resolve_dataset_jobs(args) -> list[dict]:
    if args.enroll_dir and args.trial_dir:
        return [{
            "dataset": Path(args.trial_dir).name,
            "enroll_dir": Path(args.enroll_dir),
            "trial_dir": Path(args.trial_dir),
            "asr_dir": Path(args.asr_dir) if args.asr_dir else None,
        }]
    if not args.data_root or not args.datasets:
        raise SystemExit("Provide --enroll_dir+--trial_dir OR --data_root+--datasets")
    data_root = Path(args.data_root)
    jobs = []
    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        jobs.append({
            "dataset": ds,
            "enroll_dir": data_root / f"{ds}_enrolls",
            "trial_dir": data_root / f"{ds}_trials_mixed",
            "asr_dir": data_root / ds,
        })
    return jobs


@torch.no_grad()
def _anonymize_kaldi_wavs(kaldi_dir: Path, lit, cfg, gen_kwargs, dev, args, out_subdir: str) -> dict[str, str]:
    wav = _read_kaldi(kaldi_dir / "wav.scp")
    sel = _make_selector(args)
    target_sr = int(cfg.sample_rate)
    anon_map: dict[str, str] = {}
    base_out = Path(args.anon_out_dir) / out_subdir
    base_out.mkdir(parents=True, exist_ok=True)
    for utt, scp in tqdm(wav.items(), desc=f"anon {out_subdir}"):
        src = _resolve_wav(scp, args.wav_base)
        dst = base_out / f"{utt}.wav"
        if dst.is_file():
            anon_map[utt] = str(dst)
            continue
        aud = _anonymize(lit, src, sel.ref_for(utt), target_sr, dev, gen_kwargs, args.ref_crop_sec)
        sf.write(str(dst), aud.numpy(), target_sr)
        anon_map[utt] = str(dst)
    return anon_map


def _run_asr_for_dataset(asr: VPCASREvaluator, orig_dir: Path, anon_dir: Path | None, dataset: str, results_dir: Path | None):
    rows = []
    for tag, data_dir in [("original", orig_dir), ("anon", anon_dir)]:
        if data_dir is None:
            continue
        ref_texts = _read_kaldi(data_dir / "text")
        out_text = results_dir / dataset / tag / "text" if results_dir else None
        hyp_texts = asr.transcribe_dir(data_dir, out_text=out_text)
        wer_path = results_dir / dataset / tag / "wer" if results_dir else None
        wer, _ = asr.compute_wer(ref_texts, hyp_texts, out_file=wer_path)
        print(f"[wer] {dataset} {tag}: WER={wer:.3f}%")
        parts = dataset.split("_", 1)
        rows.append({
            "metric": "WER",
            "dataset": parts[0],
            "split": parts[1] if len(parts) > 1 else dataset,
            "audio": tag,
            "WER": round(wer, 3),
        })
    return rows


def _run_eer_for_job(job, args, attacker: VPCASVAttacker, scenarios, lit, cfg, gen_kwargs, dev):
    trials = _read_trials(job["trial_dir"] / args.trials_name)
    rows = []
    for scenario in scenarios:
        if scenario not in {"oo", "oa", "aa"}:
            print(f"[eval] skip unsupported scenario {scenario!r}")
            continue
        anon_enroll, anon_trial = _scenario_anon_flags(scenario)
        enroll_embs = _enroll_spk_embeddings(
            attacker, job["enroll_dir"], attacker.sample_rate,
            anonymize=anon_enroll, lit=lit, cfg=cfg, gen_kwargs=gen_kwargs, dev=dev, args=args,
        )
        trial_embs = _trial_utt_embeddings(
            attacker, job["trial_dir"], anon_trial, lit, cfg, gen_kwargs, dev, args,
        )
        eer, n_pos, n_neg, skipped = _compute_eer(trials, enroll_embs, trial_embs)
        print(
            f"[eer] {job['dataset']} {scenario.upper()}-EER = {eer:.3f}%  "
            f"(target={n_pos} nontarget={n_neg} skipped={skipped})"
        )
        rows.append({
            "metric": "EER",
            "dataset": job["dataset"],
            "scenario": scenario,
            "enrollment": "original" if not anon_enroll else "anon",
            "trial": "original" if not anon_trial else "anon",
            "EER": round(eer, 3),
            "n_target": n_pos,
            "n_nontarget": n_neg,
            "skipped": skipped,
        })
    return rows


def _build_attacker(args, dev: str) -> VPCASVAttacker:
    vpc_root = args.vpc_root or DEFAULT_VPC_ROOT
    exp_dir = Path(args.vpc_exp_dir or Path(vpc_root) / "exp")
    asv_dir = exp_dir / "asv_ssl"
    print(f"[eval] ASV: VPC ecapa_ssl from {asv_dir}")
    return VPCASVAttacker(asv_dir, dev, vpc_root)


def main() -> None:
    p = argparse.ArgumentParser(description="VPC eval_pre-style EER + ASR (oo/oa/aa)")
    p.add_argument("--enroll_dir", default=None)
    p.add_argument("--trial_dir", default=None)
    p.add_argument("--asr_dir", default=None, help="Kaldi dir with wav.scp+text (single-set ASR)")
    p.add_argument("--data_root", default='/mnt/data/disk3/yejin/VPC/data', help="VPC data/ root for batch mode")
    p.add_argument("--datasets", default="libri_dev,libri_test")
    p.add_argument("--trials_name", default="trials")
    p.add_argument("--metrics", default="eer,wer", help="comma list: eer,wer")
    p.add_argument("--scenarios", default="oo,aa")
    p.add_argument("--out_csv", default="/mnt/data/disk2/yejin/LiveVoice/vpc_eval_pre.csv")
    p.add_argument("--results_dir", default=None, help="optional dir for ASR transcripts/WER files")

    p.add_argument("--vpc_root", default=DEFAULT_VPC_ROOT)
    p.add_argument("--vpc_exp_dir", default=None, help="default: <vpc_root>/exp")

    p.add_argument("--ckpt", default=None)
    p.add_argument("--anon_pool_dir", default="/mnt/data/disk2/VCTK-Corpus/wav48",
                   help="pseudo-speaker pool. VCTK: point at wav48/ (SVA+ setting). "
                        "LibriTTS: /mnt/data/disk2/LibriTTS/train-other-500")
    p.add_argument("--anon_strategy", default="1fix", choices=["1fix", "1rnd"],
                   help="1fix = one fixed prompt utterance for everything (SVA+ vctk-1fix); "
                        "1rnd = one prompt drawn per utterance. Both are utterance-level "
                        "under VPC 2026 rules; 4rnd embedding blending is not supported "
                        "because codec-prompt continuation has no embedding to average.")
    p.add_argument("--anon_fixed_spk", default=None,
                   help="1fix only: pool speaker id (e.g. p225). Default: chosen by --anon_seed")
    p.add_argument("--anon_fixed_utt", default=None,
                   help="1fix only: exact prompt wav path, overrides --anon_fixed_spk")
    p.add_argument("--ref_crop_sec", type=float, default=3.0,
                   help="prompt length; SVA crops prompts to 3 s")
    p.add_argument("--anon_out_dir", default='/mnt/data/disk2/yejin/LiveVoice/anon')
    p.add_argument(
        "--cache_only",
        action="store_true",
        help="use cached anon wavs only (skip VC model load; EER with oa/aa)",
    )
    p.add_argument("--anon_seed", type=int, default=1234)
    p.add_argument("--use_content_perturbation", type=int, default=1)

    p.add_argument("--asr_batch_size", type=int, default=8)

    p.add_argument("--codec", default="jhcodec", choices=["mimi", "jhcodec"])
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_decoder_layers", type=int, default=12)
    p.add_argument("--n_codebooks", type=int, default=8)
    p.add_argument("--content_source", default="auto")
    p.add_argument("--speaker_conditioning", default="auto")
    p.add_argument("--speaker_encoder_type", default="auto")
    p.add_argument("--speaker_prefix_len", type=int, default=8)
    p.add_argument("--speechbrain_source", default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument("--speechbrain_sample_rate", type=int, default=16000)
    p.add_argument("--speechbrain_embedding_dim", type=int, default=192)
    p.add_argument("--output_dir", default="/mnt/data/disk2/yejin/LiveVoice")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    args.wav_base = Path(args.vpc_root or DEFAULT_VPC_ROOT).resolve()

    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    need_anon = ("wer" in metrics) or any(s in {"oa", "aa"} for s in scenarios)
    if args.cache_only and "wer" in metrics:
        raise SystemExit("--cache_only supports EER only (use pre-built anon wavs)")
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    jobs = _resolve_dataset_jobs(args)

    attacker = _build_attacker(args, str(dev))

    lit = cfg = gen_kwargs = None
    if need_anon and not args.cache_only:
        if not args.ckpt:
            raise SystemExit("anonymized metrics/scenarios require --ckpt (or --cache_only)")
        if not args.anon_pool_dir:
            raise SystemExit("anonymized metrics/scenarios require --anon_pool_dir")
        if not args.anon_out_dir:
            raise SystemExit("anonymized metrics/scenarios require --anon_out_dir (cache)")
        _auto_infer(args)
        cfg = _build_vc_config(args, str(dev))
        lit = _build_vc_model(args, cfg, dev)
        temp = float(args.temperature)
        gen_kwargs = dict(
            temperature=temp,
            top_p=float(args.top_p) if temp > 0 and args.top_p > 0 else None,
            top_k=int(args.top_k) if temp > 0 and args.top_k > 0 else None,
            cfg_scale=float(args.cfg_scale),
        )
    elif need_anon and args.cache_only:
        if not args.anon_out_dir:
            raise SystemExit("--cache_only requires --anon_out_dir with pre-built wavs")
        print(f"[eval] cache_only: using anon wavs from {args.anon_out_dir} (no VC load)")

    rows: list[dict] = []
    results_dir = Path(args.results_dir) if args.results_dir else None

    if "eer" in metrics:
        for job in jobs:
            rows.extend(_run_eer_for_job(
                job, args, attacker, scenarios, lit, cfg, gen_kwargs, dev,
            ))

    if "wer" in metrics:
        vpc_root = args.vpc_root or DEFAULT_VPC_ROOT
        asr_dir = Path(args.vpc_exp_dir or Path(vpc_root) / "exp") / "asr"
        print(f"[eval] WER model: VPC wav2vec2-CTC from {asr_dir}")
        asr = VPCASREvaluator(asr_dir, str(dev), vpc_root, batch_size=args.asr_batch_size)
        for job in jobs:
            asr_orig = job["asr_dir"]
            if asr_orig is None or not asr_orig.is_dir():
                print(f"[wer] skip {job['dataset']}: no asr_dir")
                continue
            anon_kaldi = None
            if need_anon:
                anon_map = _anonymize_kaldi_wavs(
                    asr_orig, lit, cfg, gen_kwargs, dev, args, out_subdir=f"asr/{job['dataset']}",
                )
                anon_kaldi = materialize_anon_kaldi_dir(
                    asr_orig, Path(args.anon_out_dir) / f"asr_kaldi/{job['dataset']}", anon_map,
                )
            rows.extend(_run_asr_for_dataset(asr, asr_orig, anon_kaldi, job["dataset"], results_dir))

    if rows:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"[eval] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
