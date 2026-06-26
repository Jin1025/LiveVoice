"""UAR (utility / emotion-preservation) evaluation — SCAFFOLD.

Mirrors VoicePrivacy-Challenge-2026 SER evaluation
(evaluation/utility/ser/evaluate_ser.py): a pretrained Speech Emotion
Recognition (SER) model classifies each (optionally anonymized) utterance and
UAR = macro-averaged recall over emotion classes. Higher UAR ⇒ emotion better
preserved by anonymization.

⚠️  SCAFFOLD ONLY. Two external pieces are NOT in this repo and must be supplied:
  1. an emotional test set in Kaldi format (IEMOCAP-style), per --data_dir:
         data_dir/wav.scp     <utt_id> <wav_path>
         data_dir/utt2spk     <utt_id> <spk_id>
         data_dir/utt2emo     <utt_id> <emotion_label>
         data_dir/spk2fold    <spk_id> <fold_id>      (IEMOCAP 5-fold CV)
  2. a SER model usable by SpeechBrain `foreign_class`, per --ser_models_path
     with one sub-checkpoint per fold:  <ser_models_path>/fold_<id>/CKPT...
     (e.g. speechbrain wav2vec2 IEMOCAP emotion classifier).

Fill in --data_dir / --ser_models_path once you have them; the evaluation loop
below is complete and matches VPC's `evaluate_ser`.

Example (once data+model are present)
-------------------------------------
  python src/eval/eval_uar.py \
      --data_dir   /path/iemocap_kaldi \
      --ser_models_path /path/ser_w2v2_iemocap \
      --ckpt /mnt/.../step_latest.ckpt --anonymize \
      --anon_pool_dir /path/pseudo_targets --anon_out_dir /path/anon/iemocap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

# Reuse the EER script's IO / VC-anonymizer helpers (same repo, same patterns).
from eval.eval_eer import (  # type: ignore
    _read_kaldi,
    _resolve_wav,
    _load_full_mono_wav,
    _build_vc_config,
    _build_vc_model,
    _auto_infer,
    _assign_pseudo_targets,
    _anonymize,
)


def _require(path: Path, what: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"[uar] missing {what}: {path}\n"
            "      This scaffold needs an IEMOCAP-style Kaldi dir + a per-fold SER model. "
            "See the module docstring."
        )


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="VPC-style UAR (emotion) eval — scaffold")
    p.add_argument("--data_dir", required=True, help="Kaldi dir: wav.scp/utt2spk/utt2emo/spk2fold")
    p.add_argument("--ser_models_path", required=True, help="dir with fold_<id>/CKPT... SER models")
    p.add_argument("--out_csv", default="/tmp/uar.csv")

    # anonymization (optional — measure UAR before vs after)
    p.add_argument("--anonymize", action="store_true")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--anon_pool_dir", default=None)
    p.add_argument("--anon_out_dir", default=None)
    p.add_argument("--anon_seed", type=int, default=1234)
    p.add_argument("--use_content_perturbation", type=int, default=1)

    # VC arch (auto-infer from ckpt) — same surface as eval_eer.py
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

    try:
        from sklearn.metrics import recall_score
        from speechbrain.pretrained.interfaces import foreign_class
    except ImportError as e:
        raise SystemExit(f"[uar] needs scikit-learn + speechbrain: {e}")

    data_dir = Path(args.data_dir)
    ser_root = Path(args.ser_models_path)
    for name in ("wav.scp", "utt2spk", "utt2emo", "spk2fold"):
        _require(data_dir / name, f"{name}")
    _require(ser_root, "ser_models_path")

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    wav = _read_kaldi(data_dir / "wav.scp")
    utt2spk = _read_kaldi(data_dir / "utt2spk")
    utt2emo = _read_kaldi(data_dir / "utt2emo")
    spk2fold = _read_kaldi(data_dir / "spk2fold")

    # ── optional anonymization front-end (our VC model) ──
    lit = cfg = gen_kwargs = pseudo = None
    target_sr = 16000
    if args.anonymize:
        if not (args.ckpt and args.anon_pool_dir):
            raise SystemExit("--anonymize requires --ckpt and --anon_pool_dir")
        _auto_infer(args)
        cfg = _build_vc_config(args, str(dev))
        lit = _build_vc_model(args, cfg, dev)
        target_sr = int(cfg.sample_rate)
        temp = float(args.temperature)
        gen_kwargs = dict(
            temperature=temp,
            top_p=float(args.top_p) if temp > 0 and args.top_p > 0 else None,
            top_k=int(args.top_k) if temp > 0 and args.top_k > 0 else None,
            cfg_scale=float(args.cfg_scale),
        )
        pool = _read_kaldi(Path(args.anon_pool_dir) / "wav.scp")
        pseudo = _assign_pseudo_targets(
            list(set(utt2spk.values())), [_resolve_wav(v) for v in pool.values()], args.anon_seed
        )
        if args.anon_out_dir:
            os.makedirs(args.anon_out_dir, exist_ok=True)

    def _audio_for(utt: str):
        src = _resolve_wav(wav[utt])
        if not args.anonymize:
            a, sr = _load_full_mono_wav(src, 16000), 16000
            return a.unsqueeze(0), sr
        cached = os.path.join(args.anon_out_dir, f"{utt}.wav") if args.anon_out_dir else None
        if cached and os.path.isfile(cached):
            return _load_full_mono_wav(cached, 16000).unsqueeze(0), 16000
        aud = _anonymize(lit, src, pseudo[utt2spk[utt]], target_sr, dev, gen_kwargs)
        if cached:
            import soundfile as sf
            sf.write(cached, aud.numpy(), target_sr)
        if target_sr != 16000:
            import torchaudio
            aud = torchaudio.functional.resample(aud, target_sr, 16000)
        return aud.unsqueeze(0), 16000

    # ── classify per fold, accumulate hyp/ref, UAR = macro recall ──
    classifiers: dict[str, object] = {}
    hyp_all, ref_all = [], []
    from collections import defaultdict
    fold_pred = defaultdict(lambda: {"hyp": [], "ref": []})

    for utt in wav:
        spk = utt2spk[utt]
        fold = spk2fold.get(spk)
        if fold is None or utt not in utt2emo:
            continue
        if fold not in classifiers:
            model_dir = ser_root / f"fold_{fold}"
            classifiers[fold] = foreign_class(
                source=str(model_dir), savedir=str(model_dir),
                run_opts={"device": str(dev)},
                classname="CustomEncoderWav2vec2Classifier",
                pymodule_file="custom_interface.py",
            )
            classifiers[fold].hparams.label_encoder.ignore_len()
        clf = classifiers[fold]
        audio, _sr = _audio_for(utt)
        _, _, _, text_lab = clf.classify_batch(audio.to(dev))
        lab2ind = clf.hparams.label_encoder.lab2ind
        hyp = lab2ind[text_lab[0]]
        ref = lab2ind[utt2emo[utt]]
        hyp_all.append(hyp); ref_all.append(ref)
        fold_pred[fold]["hyp"].append(hyp); fold_pred[fold]["ref"].append(ref)

    if not ref_all:
        raise SystemExit("[uar] no scorable utterances (check utt2emo/spk2fold coverage)")

    uar = recall_score(y_true=ref_all, y_pred=hyp_all, average="macro") * 100
    print(f"[uar] overall UAR = {uar:.3f}%  (n={len(ref_all)}  anon={args.anonymize})")
    for fold, d in sorted(fold_pred.items()):
        f_uar = recall_score(y_true=d["ref"], y_pred=d["hyp"], average="macro") * 100
        print(f"[uar]   fold {fold}: UAR = {f_uar:.3f}%  (n={len(d['ref'])})")

    import csv
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["scope", "UAR", "n", "anonymized"])
        wr.writerow(["overall", round(uar, 3), len(ref_all), args.anonymize])
    print(f"[uar] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
