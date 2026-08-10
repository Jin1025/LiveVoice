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

DEFAULT_VPC_ROOT = "/mnt/data/disk3/yejin/VPC"

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


def _load_ref(path, target_sr, crop_sec) -> torch.Tensor:
    w = _load_full_mono_wav(path, target_sr)
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


def _load_scp_audio(entry: str, vpc_root: Path, target_sr: int) -> torch.Tensor:
    audio, sr = _decode_entry(entry, vpc_root)
    w = torch.from_numpy(audio).float().mean(dim=1)
    if int(sr) != target_sr:
        w = torchaudio.functional.resample(w, int(sr), target_sr)
    return w / (w.abs().max() + 1e-8)


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
    core = LiveVoiceModel(cfg, codec_model, _build_content_extractor(cfg), prosody_extractor=None)
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
    (dst / "wav").mkdir(parents=True, exist_ok=True)
    utts = sorted(scp)[args.shard :: args.num_shards]
    sr = int(cfg.sample_rate)
    todo = [u for u in utts if not (dst / "wav" / f"{u}.wav").is_file()]
    print(f"[anon] {ds}: shard {args.shard}/{args.num_shards} → {len(todo)}/{len(utts)} to do")
    for utt in tqdm(todo, desc=f"{ds}[{args.shard}]"):
        ctn = _load_scp_audio(scp[utt], vpc_root, sr).unsqueeze(0).to(dev)
        ref = _load_ref(sel.ref_for(utt), sr, args.ref_crop_sec).unsqueeze(0).to(dev)
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

    p.add_argument("--ckpt", default=None)
    p.add_argument("--anon_pool_dir", default="/mnt/data/disk2/VCTK-Corpus/wav48")
    p.add_argument("--anon_strategy", default="1fix", choices=["1fix", "1rnd"])
    p.add_argument("--anon_fixed_spk", default=None)
    p.add_argument("--anon_fixed_utt", default=None)
    p.add_argument("--anon_seed", type=int, default=1234)
    p.add_argument("--ref_crop_sec", type=float, default=3.0)

    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--cfg_scale", type=float, default=1.0)

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
    gen_kwargs = dict(temperature=float(args.temperature), cfg_scale=float(args.cfg_scale))
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
