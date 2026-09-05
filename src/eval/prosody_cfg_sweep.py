"""Prosody CFG scale sweep on IEMOCAP_dev subset.

Generates anonymized audio at multiple prosody_cfg_scale values for two models,
then evaluates UAR using VPC's SpeechBrain wav2vec2 fold models (same as official eval).

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python src/eval/prosody_cfg_sweep.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from sklearn.metrics import recall_score

from livevoice.config import LiveVoiceConfig
from livevoice.model import LiveVoiceModel, build_codec
from livevoice.model.causal_mpm import CausalMPM, CausalMPMConfig
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.utils.checkpoint import (
    load_model_weights_from_ckpt,
    read_config_from_ckpt,
)

VPC_ROOT = Path("/mnt/data/disk3/yejin/VPC")
SER_MODELS_DIR = Path("/mnt/data/disk3/yejin/VPC/exp/ser")
IEMOCAP_LABELS = ["ang", "hap", "neu", "sad"]
LAB2IND = {l: i for i, l in enumerate(IEMOCAP_LABELS)}


def _read_kaldi(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def _sample_balanced(utt2emo, n_total, seed=42):
    by_emo = {}
    for utt, emo in utt2emo.items():
        by_emo.setdefault(emo, []).append(utt)
    rng = random.Random(seed)
    per_emo = n_total // len(IEMOCAP_LABELS)
    selected = []
    for emo in IEMOCAP_LABELS:
        pool = by_emo.get(emo, [])
        selected.extend(rng.sample(pool, min(per_emo, len(pool))))
    return sorted(selected)


def _load_audio(path, sr, peak_normalize=True):
    wav, orig_sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav[:1]
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    if peak_normalize:
        wav = wav / (wav.abs().max() + 1e-8)
    return wav.squeeze(0)


# ── VPC SpeechBrain SER (fold models) ──

def _load_ser_fold_classifiers(device):
    from speechbrain.inference.interfaces import foreign_class
    classifiers = {}
    for fold in range(1, 6):
        ckpt_dir = SER_MODELS_DIR / f"fold_{fold}" / "CKPT+1"
        if not ckpt_dir.exists():
            print(f"  WARNING: SER fold_{fold} not found at {ckpt_dir}")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifiers[str(fold)] = foreign_class(
                source=str(ckpt_dir),
                savedir=str(ckpt_dir),
                run_opts={"device": "cuda:0" if device.type == "cuda" else "cpu"},
                classname="CustomEncoderWav2vec2Classifier",
                pymodule_file="custom_interface.py",
            )
    print(f"  Loaded SER classifiers for folds: {sorted(classifiers.keys())}")
    return classifiers


def _classify_utterance(classifiers, wav_path, fold):
    clf = classifiers.get(fold)
    if clf is None:
        return None
    wav, sr = torchaudio.load(str(wav_path))
    out_prob, score, index, text_lab = clf.classify_batch(wav)
    return text_lab[0]


# ── Model builder ──

def _build_model(ckpt_path, device):
    stored = read_config_from_ckpt(ckpt_path)
    if stored is None:
        raise RuntimeError(f"No livevoice_config in checkpoint: {ckpt_path}")
    stored["device"] = str(device)
    cfg = LiveVoiceConfig(**{k: v for k, v in stored.items() if hasattr(LiveVoiceConfig, k)})
    cs = str(cfg.content_source).lower()
    codec_model = build_codec(cfg)

    content_extractor = None
    if cs == "fastconformer":
        from livevoice.model.fastconformer_content import FastConformerContentEncoder
        content_extractor = FastConformerContentEncoder(
            cfg, cfg.fastconformer_ckpt, layer=cfg.fastconformer_layer)
    elif cs == "zipformer":
        from livevoice.model.zipformer_content import ZipformerContentEncoder
        layer = str(cfg.zipformer_layer)
        content_extractor = ZipformerContentEncoder(
            cfg, cfg.zipformer_ckpt, layer=(layer if layer == "out" else int(layer)))
    elif cs == "sw2v":
        from livevoice.model import Sw2vContentEncoder
        content_extractor = Sw2vContentEncoder(cfg)

    from livevoice.model.prosody_extractor import ProsodyExtractor
    pro = ProsodyExtractor(cfg) if bool(getattr(cfg, "use_prosody", False)) else None

    from livevoice.model.cepstral_extractor import CepstralExtractor
    cep = CepstralExtractor(cfg) if bool(getattr(cfg, "use_cepstral", False)) else None

    mpm = None
    if bool(getattr(cfg, "use_mpm", False)):
        mpm_ckpt = str(getattr(cfg, "mpm_ckpt", ""))
        if mpm_ckpt:
            mpm_dir = str(Path(mpm_ckpt).parent)
            cfg_path = Path(mpm_dir) / "config.json"
            mpm_cfg = CausalMPMConfig(**json.load(open(cfg_path))) if cfg_path.exists() else CausalMPMConfig()
            mpm_cfg.causal_window = bool(getattr(cfg, "mpm_causal_window", True))
            mpm = CausalMPM(mpm_cfg)
            ckpt_data = torch.load(mpm_ckpt, map_location="cpu", weights_only=False)
            mpm.load_state_dict(ckpt_data["model"])
            for p in mpm.parameters():
                p.requires_grad_(False)
            mpm.eval()
            print(f"  MPM: {mpm_ckpt} ({sum(p.numel() for p in mpm.parameters())/1e6:.1f}M)")

    model = LiveVoiceModel(cfg, codec_model, content_extractor,
                           prosody_extractor=pro, cepstral_extractor=cep,
                           mpm_extractor=mpm)
    missing, unexpected = load_model_weights_from_ckpt(model, ckpt_path, log_prefix="[sweep]")
    if missing:
        print(f"  warn: {len(missing)} missing keys")
    if unexpected:
        print(f"  warn: {len(unexpected)} unexpected keys")

    lit = LiveVoiceLightningModule(cfg, model)
    lit.eval()
    return lit.to(device), cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=[
        "/mnt/data/disk2/yejin/LiveVoice/checkpoints/180ms_mpm_finetune/epoch_latest.ckpt",
        "/mnt/data/disk2/yejin/LiveVoice/checkpoints/180ms_fullmpm_finetune/epoch_latest.ckpt",
    ])
    p.add_argument("--cfg_scales", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 1.5, 2.0])
    p.add_argument("--speaker_cfg_scale", type=float, default=1.0)
    p.add_argument("--n_utts", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="/workspace/LiveVoice/prosody_cfg_sweep")
    p.add_argument("--ref_wav", default=None,
                   help="Fixed reference speaker wav. If None, uses VCTK p225.")
    p.add_argument("--ref_crop_sec", type=float, default=4.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load IEMOCAP_dev utterances
    iemo_dir = VPC_ROOT / "data" / "IEMOCAP_dev"
    wav_scp = _read_kaldi(iemo_dir / "wav.scp")
    utt2emo = _read_kaldi(iemo_dir / "utt2emo")
    utt2spk = _read_kaldi(iemo_dir / "utt2spk")
    spk2fold = _read_kaldi(iemo_dir / "spk2fold")
    utts = _sample_balanced(utt2emo, args.n_utts, args.seed)
    print(f"Selected {len(utts)} utterances (balanced across {IEMOCAP_LABELS})")

    # Reference audio (fixed pseudo-speaker)
    ref_path = args.ref_wav
    if ref_path is None:
        vctk = Path("/mnt/data/disk2/VCTK-Corpus/wav48/p225")
        ref_path = str(sorted(vctk.glob("*.wav"))[0])
    print(f"Reference speaker: {ref_path}")

    # VPC SpeechBrain SER classifiers (5-fold)
    print("Loading VPC SER fold classifiers...")
    ser_classifiers = _load_ser_fold_classifiers(device)

    # Ground-truth labels
    gt = {u: utt2emo[u] for u in utts}

    results = []

    for ckpt_path in args.models:
        model_name = Path(ckpt_path).parent.name
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        lit, cfg = _build_model(ckpt_path, device)
        sr = int(cfg.sample_rate)
        pk = bool(getattr(cfg, "audio_peak_normalize", True))

        ref_audio = _load_audio(ref_path, sr, pk).unsqueeze(0).to(device)
        if args.ref_crop_sec > 0:
            max_samples = int(args.ref_crop_sec * sr)
            ref_audio = ref_audio[:, :max_samples]

        for pcfg in args.cfg_scales:
            tag = f"{model_name}_pcfg{pcfg:.1f}"
            wav_dir = out_root / tag
            wav_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  prosody_cfg_scale={pcfg:.1f}")

            # Generate
            todo = [u for u in utts if not (wav_dir / f"{u}.wav").is_file()]
            if todo:
                print(f"  Generating {len(todo)}/{len(utts)} utterances...")
                with torch.no_grad():
                    for utt in tqdm(todo, desc=tag):
                        scp_val = wav_scp[utt]
                        ctn_path = scp_val if Path(scp_val).is_absolute() else str(VPC_ROOT / scp_val)
                        ctn = _load_audio(ctn_path, sr, pk).unsqueeze(0).to(device)
                        codes = lit.model.generate(
                            reference_audio=ref_audio,
                            content_audio=ctn,
                            cfg_scale=args.speaker_cfg_scale,
                            prosody_cfg_scale=pcfg,
                        )
                        aud = lit.model.decode_to_audio(codes)[0].detach().float().cpu()
                        sf.write(str(wav_dir / f"{utt}.wav"), aud.numpy(), sr, subtype="PCM_16")
            else:
                print(f"  All {len(utts)} already generated, skipping.")

            # Evaluate UAR (VPC fold-based)
            hyp, ref_labels = [], []
            for utt in utts:
                wav_path = wav_dir / f"{utt}.wav"
                if not wav_path.exists():
                    continue
                spk = utt2spk[utt]
                fold = spk2fold.get(spk)
                if fold is None:
                    continue
                pred = _classify_utterance(ser_classifiers, wav_path, fold)
                if pred is None:
                    continue
                lab2ind = ser_classifiers[fold].hparams.label_encoder.lab2ind
                hyp.append(lab2ind[pred])
                ref_labels.append(lab2ind[gt[utt]])

            uar = recall_score(ref_labels, hyp, average="macro") * 100
            per_emo = {}
            for emo in IEMOCAP_LABELS:
                if emo not in LAB2IND:
                    continue
                mask = [i for i, r in enumerate(ref_labels) if r == LAB2IND[emo]]
                if mask:
                    acc = sum(1 for i in mask if hyp[i] == LAB2IND[emo]) / len(mask) * 100
                    per_emo[emo] = acc

            row = {
                "model": model_name,
                "prosody_cfg_scale": pcfg,
                "speaker_cfg_scale": args.speaker_cfg_scale,
                "UAR": round(uar, 2),
                **{f"ACC_{k}": round(v, 1) for k, v in per_emo.items()},
                "n_utts": len(hyp),
            }
            results.append(row)
            print(f"  UAR={uar:.2f}%  {per_emo}")

        # Free model memory
        del lit
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Model':<30s} {'p_cfg':>5s} {'UAR':>6s}  {'ang':>5s} {'hap':>5s} {'neu':>5s} {'sad':>5s}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<30s} {r['prosody_cfg_scale']:>5.1f} {r['UAR']:>6.2f}  "
              f"{r.get('ACC_ang', 0):>5.1f} {r.get('ACC_hap', 0):>5.1f} "
              f"{r.get('ACC_neu', 0):>5.1f} {r.get('ACC_sad', 0):>5.1f}")

    # Save results
    results_path = out_root / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
