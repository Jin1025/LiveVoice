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
import glob
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from eval.eval_anon import (
    DEFAULT_VPC_ROOT,
    _build_vc_config,
    _build_vc_model,
    _make_selector,
    _read_kaldi,
    _load_ref,
)

# Track 1 needs all of these: the four ASV dirs, IEMOCAP for UAR, and train-clean-360 to
# fine-tune the semi-informed attacker.
TRACK1_DATASETS = (
    "libri_dev_enrolls", "libri_dev_trials_mixed",
    "libri_test_enrolls", "libri_test_trials_mixed",
    "IEMOCAP_dev", "IEMOCAP_test",
    "train-clean-360",
)


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
