"""Anonymize VPC Kaldi data dirs in place-alongside, so the OFFICIAL VPC evaluator can score us.

Why this exists rather than more code in eval_anon.py: the VPC 2026 ranking metric is the
*semi-informed* EER, which requires fine-tuning the attacker ASV (WavLM+ECAPA, pretrained on
SSTC voice-converted data) on anonymized LibriSpeech train-clean-360 and only then scoring the
trials. That whole procedure already exists in the VPC repo
(``run_evaluation.py --config configs/track1/eval_post.yaml``). Reimplementing it would mean
reimplementing the attacker; all it actually needs from us is anonymized audio in its own
directory layout. So this script produces exactly that and nothing else.

Output contract (copied from the B2 baseline, anonymization/modules/mcadams/…):
    <vpc_root>/data/<ds><suffix>/           <- all metadata files copied from <ds>
    <vpc_root>/data/<ds><suffix>/wav/*.wav  <- 16 kHz mono PCM_16
    <vpc_root>/data/<ds><suffix>/wav.scp    <- "<utt> data/<ds><suffix>/wav/<utt>.wav"

Then, from the VPC root:
    python run_evaluation.py --config configs/track1/eval_pre.yaml  \
        --overwrite '{"anon_data_suffix": "<suffix>"}'      # lazy-informed EER + WER + UAR
    python run_evaluation.py --config configs/track1/eval_post.yaml \
        --overwrite '{"anon_data_suffix": "<suffix>"}'      # semi-informed EER  <- ranking

Pseudo-speaker assignment is StreamVoiceAnon+ style and comes from eval_anon.PromptSelector, so
the two entry points can never drift apart.

train-clean-360 is ~104k utterances / ~364 h and its wav.scp holds ``flac -c -d -s … |`` pipe
entries; both are handled (pipes are decoded, and --shard/--num_shards splits the work across
GPUs). Reruns skip finished files, so an interrupted shard just resumes.

    # one shard per GPU, all Track-1 sets
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python src/eval/anonymize_vpc_dirs.py \
          --ckpt .../step_latest.ckpt --anon_suffix _lv_vctk1fix \
          --anon_pool_dir /mnt/data/disk2/VCTK-Corpus/wav48 \
          --anon_strategy 1fix --anon_fixed_spk p225 \
          --shard $i --num_shards 4 &
    done; wait
    # then, once every shard is done, write the scp + metadata:
    python src/eval/anonymize_vpc_dirs.py --anon_suffix _lv_vctk1fix --finalize_only
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import io
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import librosa
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from livevoice.config import LiveVoiceConfig
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.model import (
    CepstralExtractor,
    HuBERTContentExtractor,
    ProsodyExtractor,
    StreamVoiceAnonContentEncoder,
    Sw2vContentEncoder,
    LiveVoiceModel,
    build_codec,
)
from livevoice.model.causal_mpm import CausalMPM, CausalMPMConfig
from livevoice.utils.checkpoint import (
    infer_codec_prompt_flags_from_ckpt,
    infer_content_source_from_ckpt,
    infer_speaker_conditioning_from_ckpt,
    infer_speaker_encoder_from_ckpt,
    load_model_weights_from_ckpt,
    read_config_from_ckpt,
)

DEFAULT_VPC_ROOT = "/mnt/data/disk3/yejin/VPC"

# Pre-measured LibriTTS training corpus median RMS (train-clean-100, 5000 samples, seed 42).
# Used as the reference level for universal gain matching: every input corpus is scaled so
# its median RMS matches this value. Corpus-level (not per-utterance), so causal and
# preserves between-utterance level differences (emotion cues).
_LIBRITTS_TRAIN_MEDIAN_RMS = 0.062233
_GAIN_MATCH_MAX_UTTS = 2000

# The pseudo-speaker, pinned rather than redrawn. The seeded search below is reproducible, but
# it is reproducible only for a fixed pool + seed + screening rule: retuning --ref_min_sec or
# --ref_trim_db, or pointing at a differently-populated VCTK copy, silently moves the target
# speaker and makes two runs incomparable. A published number should name its pseudo-speaker,
# so it is a constant here. Chosen by the screen at seed 1234 and then verified:
#   p231 (VCTK, 23, F, English/Southern England), 9.35s raw -> 3.00s conditioned
#   90% speech, peak 1.000, tail-200ms RMS 0.236 (ends mid-phrase, not in a pause)
# Set --anon_fixed_utt "" to fall back to the seeded search.
DEFAULT_ANON_PROMPT = "/mnt/data/disk2/VCTK-Corpus/wav48/p231/p231_023.wav"

# Track 1 needs all of these: the four ASV dirs, IEMOCAP for UAR, and train-clean-360 to
# fine-tune the semi-informed attacker.
TRACK1_DATASETS = (
    "libri_dev_enrolls", "libri_dev_trials_mixed",
    "libri_test_enrolls", "libri_test_trials_mixed",
    "IEMOCAP_dev", "IEMOCAP_test",
    "train-clean-360",
)


# ──────────────────────────────────────────────────────────────────────
#  Kaldi / audio IO
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
            f"wav.scp pipe entries are not supported here, use _decode_entry: {scp_value!r}"
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


def _load_full_mono_wav(path: str, target_sr: int, peak_normalize: bool = True) -> torch.Tensor:
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
    return audio / (audio.abs().max() + 1e-8) if peak_normalize else audio


# ── prompt conditioning ───────────────────────────────────────────────
# Why this exists. The first run with an untrimmed VCTK prompt produced SILENCE for 52% of
# libri and 69% of IEMOCAP utterances (peak 0.005, flat from sample 0), which drove WER to
# 59/86% -- and 90% of those errors were DELETIONS, i.e. the ASR heard nothing at all.
# The prompt seed 1234 had picked, p345_226.wav, was 2.56s of which 1.1s leading and 0.4s
# trailing were silence:
#     100ms RMS  [.006 .008 .008 .008 .008 .004 .008 .007 .013 .011 .013 |
#                 .215 .316 .364 .355 .184 .226 .231 .252 .211 .166 .098 | .017 .006 .008]
# A VALL-E-style codec LM continues the prompt's token stream, so a prompt ENDING in silence
# is a strong prior to keep emitting silence -- and with greedy decoding there is no sampling
# noise to escape it. VCTK is padded at both ends, so this is the rule, not bad luck.
#
# Three rules follow, all applied here so 1fix and 1rnd both get them:
#   1. trim leading/trailing silence
#   2. require >= min_sec of speech after trimming (a short prompt carries little timbre and,
#      below crop_sec, silently disables cropping altogether -- which is how p345_226 got in)
#   3. end the crop ON SPEECH, never in a pause
_TRIM_FRAME_MS = 20.0


def _frame_rms(w: torch.Tensor, sr: int, frame_ms: float = _TRIM_FRAME_MS):
    """(rms per frame, hop) — hop == frame, no overlap; only used for silence decisions."""
    hop = max(1, int(sr * frame_ms / 1000.0))
    t = w.numel() // hop
    if t == 0:
        return torch.zeros(0), hop
    return w[: t * hop].reshape(t, hop).pow(2).mean(dim=1).sqrt(), hop


def _speech_frames(w: torch.Tensor, sr: int, top_db: float):
    """Boolean mask of frames louder than `top_db` below the loudest frame.

    Relative to the peak rather than absolute, so it is unaffected by the peak-normalisation
    in _load_full_mono_wav and by VCTK's per-speaker level differences.
    """
    rms, hop = _frame_rms(w, sr)
    if rms.numel() == 0:
        return torch.zeros(0, dtype=torch.bool), hop
    thr = rms.max() * (10.0 ** (-abs(top_db) / 20.0))
    return rms > thr, hop


def _trim_silence(w: torch.Tensor, sr: int, top_db: float) -> torch.Tensor:
    mask, hop = _speech_frames(w, sr, top_db)
    idx = mask.nonzero().flatten()
    if idx.numel() == 0:
        return w                      # all-silent file: leave it, the caller screens it out
    return w[int(idx[0]) * hop : min(w.numel(), (int(idx[-1]) + 1) * hop)]


def _speech_seconds(w: torch.Tensor, sr: int, top_db: float) -> float:
    mask, hop = _speech_frames(w, sr, top_db)
    return float(mask.sum()) * hop / sr


def _crop_ending_on_speech(w: torch.Tensor, sr: int, crop_sec: float, top_db: float,
                           tail_ms: float = 200.0) -> torch.Tensor:
    """Best `crop_sec` window, scored on TAIL ENERGY first, then on whole-window energy.

    Scored on RMS rather than on a speech/silence flag. A binary "is the tail speech" score
    ties across most windows of a read-speech file, and the tie-break then picks whichever
    window came first -- which on p231_023 gave a tail RMS of 0.066 where 0.236 was available
    at identical whole-window energy. Since a weak tail is exactly what makes the codec LM
    continue into silence, the thing to maximise is the tail level itself, continuously.

    Whole-window RMS is the secondary term so the winner is not a single loud burst preceded
    by a pause; `top_db` is still honoured as a floor, so an all-silent tail can never win.
    """
    n = int(crop_sec * sr)
    if n <= 0 or w.numel() <= n:
        return w
    rms, hop = _frame_rms(w, sr)
    if rms.numel() == 0:
        return w[-n:]
    floor = rms.max() * (10.0 ** (-abs(top_db) / 20.0))
    n_tail = max(1, int(tail_ms / _TRIM_FRAME_MS))
    n_win = max(1, n // hop)
    best, best_key = None, None
    # step by one frame; a 10s VCTK file is ~460 candidate windows, so this is free
    for end in range(n_win, rms.numel() + 1):
        seg = rms[end - n_win : end]
        tail = seg[-n_tail:]
        key = (float(tail.mean() >= floor), float(tail.mean()), float(seg.mean()))
        if best_key is None or key > best_key:
            best, best_key = end, key
    stop = min(w.numel(), best * hop)
    return w[max(0, stop - n) : stop]


def _load_ref(path, target_sr, crop_sec, trim_db: float = 35.0,
              peak_normalize: bool = True) -> torch.Tensor:
    w = _load_full_mono_wav(path, target_sr, peak_normalize)
    if trim_db > 0:
        w = _trim_silence(w, target_sr, trim_db)
        if crop_sec and crop_sec > 0:
            w = _crop_ending_on_speech(w, target_sr, crop_sec, trim_db)
        return w
    # trim_db <= 0 reproduces the pre-fix behaviour exactly, for A/B against the broken run
    n = int(crop_sec * target_sr)
    if crop_sec and crop_sec > 0 and w.numel() > n:
        start = random.Random(f"crop:{path}").randint(0, w.numel() - n)
        w = w[start : start + n]
    return w


_AUDIO_EXT = (".flac", ".wav", ".ogg", ".opus", ".mp3")


def _decode_entry(entry: str, vpc_root: Path):
    """Read one wav.scp value, including Kaldi ``… |`` pipe entries.

    train-clean-360 stores ``flac -c -d -s corpora/… |``. We do NOT shell out for that: the
    `flac` binary is not installed here, libsndfile decodes FLAC natively, and 104k
    subprocesses would dominate the runtime anyway. So a pipe whose command merely decodes one
    audio file is short-circuited to a direct read; anything else still runs as a shell
    command, matching VPC's utils.data_io.load_wav_from_scp.
    """
    e = entry.strip()
    if not e.endswith("|"):
        p = Path(e)
        return sf.read(str(p if p.is_absolute() else vpc_root / p),
                       dtype="float32", always_2d=True)

    cmd = e[:-1].strip()
    for tok in cmd.split():
        if tok.lower().endswith(_AUDIO_EXT):
            p = Path(tok)
            p = p if p.is_absolute() else vpc_root / p
            if p.is_file():
                return sf.read(str(p), dtype="float32", always_2d=True)
    proc = subprocess.run(cmd, shell=True, cwd=str(vpc_root),
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if proc.returncode != 0 or not proc.stdout:
        raise IOError(f"wav.scp pipe failed: {e}")
    with io.BytesIO(proc.stdout) as buf:
        return sf.read(buf, dtype="float32", always_2d=True)


def _load_scp_audio(entry: str, vpc_root: Path, target_sr: int,
                    peak_normalize: bool = True) -> torch.Tensor:
    audio, sr = _decode_entry(entry, vpc_root)
    w = torch.from_numpy(audio).float().mean(dim=1)
    if int(sr) != target_sr:
        w = torchaudio.functional.resample(w, int(sr), target_sr)
    return w / (w.abs().max() + 1e-8) if peak_normalize else w


def _compute_corpus_gain(wavscp: Path, vpc_root: Path, target_sr: int,
                         peak_normalize: bool) -> float:
    """Compute gain to match this corpus's median RMS to LibriTTS training level.

    Samples up to _GAIN_MATCH_MAX_UTTS utterances (deterministic seed) to keep
    startup fast. Returns 1.0 if the corpus is already close (within 10%).
    """
    import numpy as np
    scp = _read_kaldi(wavscp)
    entries = list(scp.values())
    if len(entries) > _GAIN_MATCH_MAX_UTTS:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(entries), _GAIN_MATCH_MAX_UTTS, replace=False)
        entries = [entries[i] for i in indices]
    rms_vals = []
    for entry in entries:
        try:
            w = _load_scp_audio(entry, vpc_root, target_sr, peak_normalize)
            if w.numel() < 1600:
                continue
            rms = float(w.pow(2).mean().sqrt())
            if rms > 1e-6:
                rms_vals.append(rms)
        except Exception:
            continue
    if len(rms_vals) < 10:
        return 1.0
    median_rms = float(np.median(rms_vals))
    gain = _LIBRITTS_TRAIN_MEDIAN_RMS / median_rms
    if 0.9 <= gain <= 1.1:
        return 1.0
    return gain


# ──────────────────────────────────────────────────────────────────────
#  Rebuilding the trained model
# ──────────────────────────────────────────────────────────────────────
# Fields the EVAL owns — never taken from the checkpoint. Everything else in a stored config
# describes the model that was trained and must be reproduced exactly, or we score the weights
# inside a different architecture (this is how the 2026-08-04 CFG sweep got invalidated:
# audio_duration and the codec_prompt_* flags silently came from today's defaults).
_EVAL_OWNED_FIELDS = frozenset({
    "device", "output_dir",
    # Feature caches only cover LibriTTS utterances. VPC enrol/trial wavs are not in them, so
    # every content path has to run on-the-fly here regardless of how it was trained.
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


def _auto_infer(args: argparse.Namespace) -> None:
    if str(args.content_source).lower() == "auto":
        args.content_source = infer_content_source_from_ckpt(args.ckpt) or "hubert"
    if str(args.speaker_conditioning).lower() == "auto":
        args.speaker_conditioning = infer_speaker_conditioning_from_ckpt(args.ckpt) or "prefix"
    if str(args.speaker_encoder_type).lower() == "auto":
        args.speaker_encoder_type = infer_speaker_encoder_from_ckpt(args.ckpt) or "codec"


def _build_vc_config(args: argparse.Namespace, device: str) -> LiveVoiceConfig:
    codec = str(args.codec).lower()
    kw = dict(
        device=device,
        codec=codec,
        sample_rate=24000 if codec == "mimi" else 16000,
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
        defaults = {f.name: f.default for f in dataclasses.fields(LiveVoiceConfig)}
        overridden, relocated = [], []
        for k, v in stored.items():
            if k not in known or k in _EVAL_OWNED_FIELDS:
                continue
            # A stored config records where the weights lived WHEN IT WAS TRAINED. Those
            # files get moved; an absolute path that no longer exists must not override a
            # working default, or every checkpoint predating a reorganisation becomes
            # unloadable. Only fall back when the current default actually resolves.
            if isinstance(v, str) and v.startswith("/") and not os.path.exists(v):
                d = defaults.get(k)
                if isinstance(d, str) and os.path.exists(d):
                    relocated.append(f"{k}: {v} (missing) → {d}")
                    continue
            if k in kw and kw[k] != v:
                overridden.append(f"{k}: {kw[k]!r} → {v!r}")
            kw[k] = v
        print(f"[eval] stored config found in ckpt ({len(stored)} fields) — using it")
        for line in overridden:
            print(f"[eval]   CLI overridden by ckpt  {line}")
        for line in relocated:
            print(f"[eval]   stale path in ckpt, using config.py default  {line}")
    else:
        # Pre-CONFIG_CKPT_KEY checkpoint: nothing to read, so fall back to topology inference.
        # A codec_prompt_* field that did not exist yet must read as OFF, not as today's
        # default — that is what infer_codec_prompt_flags_from_ckpt encodes.
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
    # The conditioning modules must be rebuilt here too, not just in train.py: LiveVoiceModel
    # gates each one on `extractor is not None`, so passing None silently DISABLES a feature the
    # checkpoint was trained with and the run still completes, producing audio for a different
    # model than the one being evaluated.
    pro = ProsodyExtractor(cfg) if bool(getattr(cfg, "use_prosody", False)) else None
    if pro is not None:
        print(f"[anon] prosody conditioning: pitch_method={cfg.pitch_method!r} "
              f"normalize={getattr(cfg, 'pitch_normalize', None)}")
    cep = CepstralExtractor(cfg) if bool(getattr(cfg, "use_cepstral", False)) else None
    if cep is not None:
        print(f"[anon] cepstral conditioning: spec={cfg.cepstral_spec!r} "
              f"({cep.n_channels} channels)")
    mpm = None
    if bool(getattr(cfg, "use_mpm", False)):
        mpm_ckpt = str(getattr(cfg, "mpm_ckpt", ""))
        if mpm_ckpt:
            import json
            mpm_dir = str(Path(mpm_ckpt).parent)
            cfg_path = Path(mpm_dir) / "config.json"
            if cfg_path.exists():
                mpm_cfg = CausalMPMConfig(**json.load(open(cfg_path)))
            else:
                mpm_cfg = CausalMPMConfig()
            # the pretrain config.json predates this flag; the LiveVoice config owns it.
            mpm_cfg.causal_window = bool(getattr(cfg, "mpm_causal_window", True))
            mpm = CausalMPM(mpm_cfg)
            ckpt_data = torch.load(mpm_ckpt, map_location="cpu", weights_only=False)
            mpm.load_state_dict(ckpt_data["model"])
            if bool(getattr(cfg, "mpm_freeze", True)):
                for p in mpm.parameters():
                    p.requires_grad_(False)
                mpm.eval()
            print(f"[anon] MPM conditioning: {mpm_ckpt} ({sum(p.numel() for p in mpm.parameters())/1e6:.1f}M params)")
        else:
            print("[anon] WARNING: use_mpm=True but mpm_ckpt is empty")
    core = LiveVoiceModel(cfg, codec_model, _build_content_extractor(cfg),
                          prosody_extractor=pro, cepstral_extractor=cep,
                          mpm_extractor=mpm)
    missing, unexpected = load_model_weights_from_ckpt(core, args.ckpt, log_prefix="[eval]")
    if missing:
        print(f"[eval] warn: {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"[eval] warn: {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    lit = LiveVoiceLightningModule(cfg, core)
    lit.eval()
    return lit.to(dev)


# ──────────────────────────────────────────────────────────────────────
#  Pseudo-speaker prompt selection (StreamVoiceAnon+)
# ──────────────────────────────────────────────────────────────────────
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
    no K-reference blending: StreamVoiceAnon 2024's
    g_anon = alpha * mean_i(g_i) + (1 - alpha) * g_s
    averages speaker EMBEDDINGS, which has no analogue when the reference is a codec token
    stream occupying the first T_ref positions of the same AR sequence.

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
                 fixed_spk: str | None = None, fixed_utt: str | None = None,
                 sr: int = 16000, min_sec: float = 4.0, trim_db: float = 35.0,
                 crop_sec: float = 3.0):
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
                spk, self.fixed = self._screen(
                    fixed_spk, sr, min_sec, trim_db, max(crop_sec, min_sec))
            print(f"[anon] strategy=1fix  target speaker={spk}  prompt={self.fixed}")
            self._report(self.fixed, sr, trim_db, crop_sec)
        else:
            print(f"[anon] strategy=1rnd  prompt drawn per utterance from the pool")
        print(f"[anon] pool: {len(self.by_spk)} speakers, {len(self.all_wavs)} utterances "
              f"({pool_dir})")
        if len(self.by_spk) < 2:
            print("[anon] WARNING: pool has <2 speakers — for VCTK point --anon_pool_dir at "
                  "the wav48/ directory, not the corpus root.")

    def _screen(self, fixed_spk: str | None, sr: int, min_sec: float, trim_db: float,
                need_sec: float) -> tuple[str, str]:
        """Pick (speaker, utterance) among candidates that survive the silence rules.

        Speakers are tried in a seeded order rather than by a single draw, so a speaker whose
        recordings are all too short cannot leave the run with no valid prompt. Within a
        speaker the choice is deterministic -- most speech after trimming -- because the SAME
        prompt has to come back on every resume and for every dataset in the run.
        """
        order = list(self.by_spk)
        if fixed_spk:
            if fixed_spk not in self.by_spk:
                raise SystemExit(f"--anon_fixed_spk {fixed_spk!r} not in pool; have e.g. "
                                 f"{list(self.by_spk)[:8]}")
            order = [fixed_spk]
        else:
            random.Random(self.seed).shuffle(order)
        rejected = 0
        for spk in order:
            scored = []
            for w in self.by_spk[spk]:
                try:
                    a = _trim_silence(_load_full_mono_wav(w, sr), sr, trim_db)
                except Exception:
                    continue
                sec = a.numel() / sr
                if sec < max(min_sec, need_sec):
                    rejected += 1
                    continue
                scored.append((_speech_seconds(a, sr, trim_db), sec, w))
            if scored:
                scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
                if rejected:
                    print(f"[anon] prompt screen: rejected {rejected} candidate(s) shorter "
                          f"than {max(min_sec, need_sec):.1f}s of speech after trimming")
                return spk, scored[0][2]
            if fixed_spk:
                raise SystemExit(
                    f"--anon_fixed_spk {fixed_spk!r} has no utterance with >= "
                    f"{max(min_sec, need_sec):.1f}s of speech after trimming at "
                    f"--ref_trim_db {trim_db}. Lower --ref_min_sec or pick another speaker.")
        raise SystemExit(f"no pool utterance has >= {max(min_sec, need_sec):.1f}s of speech "
                         f"after trimming at --ref_trim_db {trim_db}")

    @staticmethod
    def _report(path: str, sr: int, trim_db: float, crop_sec: float) -> None:
        """Print what the model will actually be conditioned on. Cheap, and the last chance to
        catch a silent prompt before committing ~13 GPU-h of generation."""
        try:
            raw = _load_full_mono_wav(path, sr)
            ref = _load_ref(path, sr, crop_sec, trim_db)
        except Exception as e:                                   # pragma: no cover
            print(f"[anon] prompt report unavailable: {e}")
            return
        rms, hop = _frame_rms(ref, sr, 100.0)
        tail = ref[-int(0.2 * sr):]
        print(f"[anon] prompt: raw {raw.numel()/sr:.2f}s -> conditioned {ref.numel()/sr:.2f}s "
              f"({_speech_seconds(ref, sr, trim_db):.2f}s speech), "
              f"peak {float(ref.abs().max()):.3f}, "
              f"tail-200ms rms {float(tail.pow(2).mean().sqrt()):.4f}")
        print(f"[anon] prompt 100ms rms: "
              f"{[round(float(x), 3) for x in rms]}")

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
           getattr(args, "anon_fixed_spk", None), getattr(args, "anon_fixed_utt", None),
           float(args.ref_min_sec), float(args.ref_trim_db), float(args.ref_crop_sec))
    if key not in _SELECTOR_CACHE:
        _SELECTOR_CACHE[key] = PromptSelector(
            args.anon_pool_dir, args.anon_strategy, args.anon_seed, args.wav_base,
            getattr(args, "anon_fixed_spk", None), getattr(args, "anon_fixed_utt", None),
            sr=16000, min_sec=float(args.ref_min_sec), trim_db=float(args.ref_trim_db),
            crop_sec=float(args.ref_crop_sec))
    return _SELECTOR_CACHE[key]


# ──────────────────────────────────────────────────────────────────────
#  Output-directory safety
# ──────────────────────────────────────────────────────────────────────
# Written into every dir we create, so a later run can tell "this is mine, and it was produced
# with these settings" from "this is VPC's, do not touch".
_STAMP = ".livevoice_anon.json"

# Suffixes the VPC baselines themselves write into data/. Reusing one would interleave our
# audio with theirs under a name the evaluator already treats as that baseline.
_RESERVED_SUFFIXES = ("_mcadams", "_sttts", "_sttts_multi", "_nac", "_B4", "_BM1", "_BM2",
                      "_BM3", "_asrbn")


def _run_fingerprint(args) -> dict:
    """Everything that changes the AUDIO. Deliberately excludes --shard/--num_shards: shards
    are parallel workers on ONE output dir and must agree, not conflict."""
    return {
        "ckpt": str(args.ckpt),
        "anon_strategy": str(args.anon_strategy),
        "anon_pool_dir": str(args.anon_pool_dir),
        "anon_seed": int(args.anon_seed),
        "anon_fixed_spk": args.anon_fixed_spk,
        "anon_fixed_utt": args.anon_fixed_utt,
        "ref_crop_sec": float(args.ref_crop_sec),
        "ref_trim_db": float(args.ref_trim_db),
        "ref_min_sec": float(args.ref_min_sec),
        "temperature": float(args.temperature),
        "cfg_scale": float(args.cfg_scale),
        "prosody_cfg_scale": float(args.prosody_cfg_scale),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "input_gain": float(args.input_gain),
        "auto_gain": bool(args.auto_gain),
    }


def _validate_suffix(suffix: str, datasets: list[str], vpc_root: Path) -> None:
    """Refuse suffixes that would land on top of VPC's own directories.

    ``data/<ds><suffix>`` is plain string concatenation, so e.g. ``--datasets libri_dev
    --anon_suffix _enrolls`` resolves to ``data/libri_dev_enrolls`` — the real enrolment set,
    whose metadata and wav.scp we would then overwrite. Nothing about that is recoverable.
    """
    if not suffix or not suffix.strip():
        raise SystemExit("--anon_suffix must be non-empty (it is what keeps our output out of "
                         "VPC's own data dirs)")
    if any(c in suffix for c in "/ \t"):
        raise SystemExit(f"--anon_suffix {suffix!r} must not contain a path separator or space")
    if suffix in _RESERVED_SUFFIXES:
        raise SystemExit(f"--anon_suffix {suffix!r} is a VPC baseline suffix; pick another")
    data = vpc_root / "data"
    for ds in datasets:
        out = f"{ds}{suffix}"
        target = data / out
        if (target / "wav.scp").is_file() and not (target / _STAMP).is_file():
            raise SystemExit(
                f"refusing to write into data/{out}: it already has a wav.scp but no {_STAMP}, "
                f"so it is not ours (an original VPC set, or another tool's output).")


def _claim_output_dir(dst: Path, fingerprint: dict, ds: str) -> None:
    """Create/verify the stamp. Guards the failure mode that actually bites: rerunning with a
    different strategy or checkpoint into the same suffix, where the resume logic would skip
    every existing wav and silently hand back the PREVIOUS run's audio."""
    import json
    dst.mkdir(parents=True, exist_ok=True)
    stamp = dst / _STAMP
    if stamp.is_file():
        try:
            prev = json.loads(stamp.read_text())
        except Exception:
            prev = {}
        diff = {k: (prev.get(k), v) for k, v in fingerprint.items() if prev.get(k) != v}
        if diff:
            raise SystemExit(
                f"data/{dst.name} was generated with different settings:\n" +
                "\n".join(f"    {k}: existing={o!r}  now={n!r}" for k, (o, n) in diff.items()) +
                f"\n  Reusing it would mix two runs — finished utterances are skipped, so the "
                f"old audio would survive. Use a new --anon_suffix, or delete data/{dst.name}.")
        return
    # Shards race here; identical content makes the last writer harmless.
    tmp = dst / f".{_STAMP}.{os.getpid()}"
    tmp.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
    os.replace(tmp, stamp)


def _purge(vpc_root: Path, suffix: str, datasets: list[str], confirmed: bool) -> None:
    """Remove one run: our audio dirs AND everything VPC derived from them.

    Deleting only data/<ds><suffix> is not enough and is the trap worth guarding against.
    VPC caches speaker embeddings under exp/asv_ssl/cosine_out/emb_xvect/<ds><suffix>/, and a
    rerun REUSES them -- so new audio would be scored with the old audio's embeddings and the
    EER would silently describe a run that no longer exists. The ASR/SER working dirs and the
    results CSVs are the same story, one directory up.

    Our own dirs are only removed when they carry the stamp, so a name collision with a VPC
    directory can never turn this into a data-loss event. VPC's derived artifacts are matched
    by suffix instead (they are not ours to stamp), which is why --anon_suffix is required and
    is checked against _RESERVED_SUFFIXES by the caller.
    """
    data_dirs, derived, skipped = [], [], []
    for ds in datasets:
        d = vpc_root / "data" / f"{ds}{suffix}"
        if not d.exists():
            continue
        (data_dirs if (d / _STAMP).is_file() else skipped).append(d)
    for pat in (f"data/*{suffix}", f"exp/*/*{suffix}", f"exp/*/*{suffix}.csv",
                f"exp/*/*{suffix}/", f"exp/asv_ssl/cosine_out/*/*{suffix}",
                f"exp/results_summary/*{suffix}"):
        for p in sorted(glob.glob(str(vpc_root / pat))):
            q = Path(p)
            if q in data_dirs or q in skipped or q in derived:
                continue
            if q.is_dir() and (q / _STAMP).is_file():
                data_dirs.append(q)
            else:
                derived.append(q)

    if not data_dirs and not derived:
        print(f"[purge] nothing matching {suffix!r} under {vpc_root}")
        return
    print(f"[purge] target {vpc_root}   suffix {suffix!r}")
    for d in data_dirs:
        n = len(list((d / "wav").glob("*.wav"))) if (d / "wav").is_dir() else 0
        print(f"    ours     {d.relative_to(vpc_root)}   ({n} wav)")
    for d in derived:
        print(f"    derived  {d.relative_to(vpc_root)}")
    for d in skipped:
        print(f"    KEPT (no {_STAMP}, not ours)  {d.relative_to(vpc_root)}")
    if not confirmed:
        print(f"\n[purge] dry run — nothing deleted. Add --yes to delete the "
              f"{len(data_dirs) + len(derived)} path(s) above.")
        return
    for d in data_dirs + derived:
        shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink(missing_ok=True)
    print(f"\n[purge] deleted {len(data_dirs) + len(derived)} path(s)")


def _copy_metadata(src: Path, dst: Path) -> None:
    """Copy the Kaldi metadata files (utt2spk, spk2utt, text, utt2emo, trials, …) but not the
    directories, which hold audio. Same rule as VPC's utils.copy_data_dir."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in glob.glob(str(src / "*")):
        if os.path.isfile(p):
            shutil.copy(p, dst)


def _finalize(ds: str, vpc_root: Path, suffix: str) -> bool:
    """Write wav.scp + metadata once every utterance exists. Returns True when complete.

    Deliberately refuses to write a partial wav.scp: VPC would silently evaluate on the subset
    that happened to finish, and a shard that died would look like a good result.
    """
    src, dst = vpc_root / "data" / ds, vpc_root / "data" / f"{ds}{suffix}"
    scp = _read_kaldi(src / "wav.scp")
    missing = [u for u in scp if not (dst / "wav" / f"{u}.wav").is_file()]
    if missing:
        print(f"[finalize] {ds}{suffix}: {len(missing)}/{len(scp)} wavs still missing "
              f"(e.g. {missing[:3]}) — wav.scp NOT written")
        return False
    _copy_metadata(src, dst)
    with open(dst / "wav.scp", "w", encoding="utf-8") as f:
        for u in scp:
            f.write(f"{u} data/{ds}{suffix}/wav/{u}.wav\n")
    print(f"[finalize] {ds}{suffix}: {len(scp)} utts — wav.scp + metadata written")
    return True


@torch.no_grad()
def _run_dataset(ds, vpc_root, suffix, lit, cfg, sel, gen_kwargs, dev, args) -> None:
    src, dst = vpc_root / "data" / ds, vpc_root / "data" / f"{ds}{suffix}"
    scp = _read_kaldi(src / "wav.scp")
    _claim_output_dir(dst, _run_fingerprint(args), ds)
    (dst / "wav").mkdir(parents=True, exist_ok=True)
    utts = sorted(scp)[args.shard :: args.num_shards]
    sr = int(cfg.sample_rate)
    pk = bool(getattr(cfg, "audio_peak_normalize", True))

    if args.auto_gain:
        gain = _compute_corpus_gain(src / "wav.scp", vpc_root, sr, pk)
        print(f"[anon] {ds}: auto_gain={gain:.4f}")
    else:
        gain = float(args.input_gain)

    todo = [u for u in utts if not (dst / "wav" / f"{u}.wav").is_file()]
    print(f"[anon] {ds}: shard {args.shard}/{args.num_shards} → {len(todo)}/{len(utts)} to do")
    for utt in tqdm(todo, desc=f"{ds}[{args.shard}]"):
        ctn = _load_scp_audio(scp[utt], vpc_root, sr, pk).unsqueeze(0).to(dev)
        if gain != 1.0:
            ctn = ctn * gain
        ref = _load_ref(sel.ref_for(utt), sr, args.ref_crop_sec,
                        float(args.ref_trim_db), pk).unsqueeze(0).to(dev)
        codes = lit.model.generate(reference_audio=ref, content_audio=ctn, **gen_kwargs)
        aud = lit.model.decode_to_audio(codes)[0].detach().float().cpu()
        # Write via a temp name so an interrupted run never leaves a truncated wav that the
        # resume logic would then treat as finished.
        tmp = dst / "wav" / f".{utt}.partial.wav"
        sf.write(str(tmp), aud.numpy(), sr, subtype="PCM_16")
        os.replace(tmp, dst / "wav" / f"{utt}.wav")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vpc_root", default=DEFAULT_VPC_ROOT)
    p.add_argument("--datasets", default=",".join(TRACK1_DATASETS))
    p.add_argument("--anon_suffix", required=True,
                   help="e.g. _lv_vctk1fix — becomes data/<ds><suffix>/ and VPC's anon_data_suffix")
    p.add_argument("--finalize_only", action="store_true",
                   help="skip generation; just write wav.scp + metadata for completed dirs")
    p.add_argument("--purge", action="store_true",
                   help="delete this suffix's audio dirs AND the VPC eval artifacts derived "
                        "from them (cached ASV embeddings above all — a rerun reuses them and "
                        "would score new audio with the old audio's embeddings). Lists the "
                        "targets and deletes nothing unless --yes is also given.")
    p.add_argument("--yes", action="store_true",
                   help="with --purge, actually delete instead of listing")

    p.add_argument("--ckpt", default=None)
    p.add_argument("--anon_pool_dir", default="/mnt/data/disk2/VCTK-Corpus/wav48")
    p.add_argument("--anon_strategy", default="1fix", choices=["1fix", "1rnd"])
    p.add_argument("--anon_fixed_spk", default=None)
    p.add_argument("--anon_fixed_utt", default=DEFAULT_ANON_PROMPT,
                   help="exact prompt wav; pinned by default so every run and every dataset "
                        "shares one pseudo-speaker. Pass \"\" to re-enable the seeded search "
                        "over --anon_pool_dir.")
    p.add_argument("--anon_seed", type=int, default=1234)
    p.add_argument("--ref_crop_sec", type=float, default=3.0)
    p.add_argument("--ref_trim_db", type=float, default=35.0,
                   help="silence threshold in dB below the prompt's loudest 20ms frame. "
                        "Leading/trailing silence is trimmed and the crop is placed to END "
                        "ON SPEECH; a prompt ending in a pause makes the codec LM continue "
                        "with silence (52%% of outputs were silent before this existed). "
                        "0 turns off every silence-aware step -- no trimming, random crop "
                        "position, and screening falls back to raw duration -- for A/B only.")
    p.add_argument("--ref_min_sec", type=float, default=4.0,
                   help="reject pool utterances with less than this much audio after "
                        "trimming; raised to --ref_crop_sec if that is larger")

    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--input_gain", type=float, default=1.0,
                   help="single scalar applied to the SOURCE audio before the model, to "
                        "close the level gap between the training corpus and the one being "
                        "converted. Overridden per-dataset by --auto_gain.")
    p.add_argument("--auto_gain", action="store_true",
                   help="automatically compute per-corpus gain to match LibriTTS training "
                        "level (median RMS). Universal preprocessing: the same logic applies "
                        "to every corpus, no dataset-specific tuning. Overrides --input_gain.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--prosody_cfg_scale", type=float, default=1.0)

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
    p.add_argument("--use_content_perturbation", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    vpc_root = Path(args.vpc_root).resolve()
    args.wav_base = vpc_root
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    suffix = args.anon_suffix
    _validate_suffix(suffix, datasets, vpc_root)

    # utts[shard::num_shards] is a legal slice for ANY shard, so an out-of-range shard silently
    # produces a partial, overlapping split instead of failing. --shard is an index into the
    # split, not a GPU id; CUDA_VISIBLE_DEVICES picks the GPU.
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(
            f"--shard must be in [0, {args.num_shards}) for --num_shards {args.num_shards}, "
            f"got {args.shard}. Shards 0..{args.num_shards - 1} together cover every utterance; "
            f"any other index leaves gaps. (Pick the GPU with CUDA_VISIBLE_DEVICES, not --shard.)")

    if args.purge:
        _purge(vpc_root, suffix, datasets, bool(args.yes))
        return

    if args.finalize_only:
        ok = all(_finalize(ds, vpc_root, suffix) for ds in datasets)
        print(f"\n[finalize] {'ALL COMPLETE' if ok else 'INCOMPLETE — rerun the missing shards'}")
        print(f"[finalize] then from {vpc_root}:\n"
              f"  python run_evaluation.py --config configs/track1/eval_pre.yaml "
              f"--overwrite '{{\"anon_data_suffix\": \"{suffix}\"}}'\n"
              f"  python run_evaluation.py --config configs/track1/eval_post.yaml "
              f"--overwrite '{{\"anon_data_suffix\": \"{suffix}\"}}'")
        return

    if not args.ckpt:
        raise SystemExit("--ckpt is required unless --finalize_only")
    dev = torch.device("cpu" if args.cpu else "cuda")
    _auto_infer(args)
    cfg = _build_vc_config(args, str(dev))
    lit = _build_vc_model(args, cfg, dev)
    sel = _make_selector(args)
    gen_kwargs = dict(temperature=float(args.temperature), cfg_scale=float(args.cfg_scale),
                      prosody_cfg_scale=float(args.prosody_cfg_scale))
    if args.top_p:
        gen_kwargs["top_p"] = float(args.top_p)
    if args.top_k:
        gen_kwargs["top_k"] = int(args.top_k)

    for ds in datasets:
        if not (vpc_root / "data" / ds / "wav.scp").is_file():
            print(f"[anon] SKIP {ds}: no {vpc_root/'data'/ds/'wav.scp'}")
            continue
        _run_dataset(ds, vpc_root, suffix, lit, cfg, sel, gen_kwargs, dev, args)

    print()
    for ds in datasets:
        if (vpc_root / "data" / ds / "wav.scp").is_file():
            _finalize(ds, vpc_root, suffix)


if __name__ == "__main__":
    main()
