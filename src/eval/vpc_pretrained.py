"""VPC-2026 eval_pre pretrained backends (ASV ecapa_ssl + ASR wav2vec2-CTC).

Loads models from the Voice-Privacy-Challenge ``exp/`` directory and reuses
VPC evaluation code via ``--vpc_root``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torchaudio
from sklearn.metrics.pairwise import cosine_distances
from speechbrain.utils.metric_stats import EER, ErrorRateStats


def setup_vpc_imports(vpc_root: str | Path) -> Path:
    root = Path(vpc_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VPC root not found: {root}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


class VPCASVAttacker:
    """Pretrained VPC ``exp/asv_ssl`` (WavLM-Large + ECAPA-TDNN)."""

    def __init__(self, model_dir: str | Path, device: str, vpc_root: str | Path):
        setup_vpc_imports(vpc_root)
        from evaluation.privacy.asv.speechbrain_vectors import SpeechBrainVectors
        from evaluation.privacy.asv.utils import normalize_wave

        self.device = device
        self._normalize_wave = normalize_wave
        model_dir = Path(model_dir).resolve()
        if not (model_dir / "embedding_model.ckpt").is_file():
            raise FileNotFoundError(f"Missing embedding_model.ckpt under {model_dir}")
        if not (model_dir / "WavLM-Large.pt").is_file():
            raise FileNotFoundError(f"Missing WavLM-Large.pt under {model_dir}")
        self.sample_rate = 16000
        self._extractor = SpeechBrainVectors(
            vec_type="ecapa_ssl", device=device, model_path=model_dir
        )

    @torch.no_grad()
    def embed_path(self, wav_path: str) -> torch.Tensor:
        with torch.no_grad():
            signal, fs = torchaudio.load(wav_path)
            norm_wave = self._normalize_wave(signal, fs, device=self.device)
            vec = self._extractor.extract_vector(
                audio=norm_wave, sr=fs, wav_path=wav_path
            )
        return vec.float().cpu()


class VPCASREvaluator:
    """Pretrained VPC ``exp/asr`` (wav2vec2-large + CTC, EncoderASR)."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str,
        vpc_root: str | Path,
        hparams_file: str = "hyperparams.yaml",
        batch_size: int = 8,
    ):
        setup_vpc_imports(vpc_root)
        from evaluation.utility.asr.speechbrain_asr.inference import (
            ASRDataset,
            InferenceSpeechBrainASR,
        )

        self.device = device
        self.batch_size = int(batch_size)
        model_dir = Path(model_dir).resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"ASR model dir not found: {model_dir}")
        self._asr = InferenceSpeechBrainASR(
            model_path=str(model_dir),
            asr_hparams=hparams_file,
            model_type="EncoderASR",
            device=device,
        )
        self._dataset_cls = ASRDataset

    @torch.no_grad()
    def transcribe_dir(self, data_dir: Path, out_text: Path | None = None) -> dict[str, str]:
        from copy import deepcopy
        from torch.utils.data import DataLoader

        from utils import save_kaldi_format

        wav_scp = data_dir / "wav.scp"
        if not wav_scp.is_file():
            raise FileNotFoundError(f"Missing {wav_scp}")
        dataset = self._dataset_cls(wav_scp_file=wav_scp, asr_model=self._asr.asr_model)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn,
        )
        texts: dict[str, str] = {}
        for batch in loader:
            filenames, inputs, lengths = batch
            inputs = inputs.to(self.device)
            lengths = lengths.to(self.device)
            predicts, _ = self._asr.asr_model.transcribe_batch(inputs, lengths)
            for i, utt_id in enumerate(filenames):
                texts[deepcopy(utt_id)] = str(predicts[i])
        if out_text is not None:
            out_text.parent.mkdir(parents=True, exist_ok=True)
            save_kaldi_format(texts, out_text)
        return texts

    def compute_wer(
        self, ref_texts: dict[str, str], hyp_texts: dict[str, str], out_file: Path | None = None
    ) -> tuple[float, ErrorRateStats]:
        wer_stats = ErrorRateStats()
        ids, predicted, targets = [], [], []
        for utt_id, ref in ref_texts.items():
            if utt_id not in hyp_texts:
                continue
            ids.append(utt_id)
            targets.append(ref)
            predicted.append(hyp_texts[utt_id])

        def _plain_text_key(paths):
            return [tok.strip().split(" ") for tok in paths]

        wer_stats.append(
            ids=ids,
            predict=_plain_text_key(predicted),
            target=_plain_text_key(targets),
        )
        if out_file is not None:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                wer_stats.write_stats(f)
        wer = float(wer_stats.summarize("error_rate"))
        return wer, wer_stats


def vpc_cosine_eer(
    trials: list[tuple[str, str, int]],
    enroll_embs: dict[str, torch.Tensor],
    trial_embs: dict[str, torch.Tensor],
) -> tuple[float, int, int, int]:
    """VPC-style EER: speaker-mean enroll vs utterance trial, sklearn cosine."""
    enrol_ids, enrol_vecs, test_ids, test_vecs = [], [], [], []
    labels: dict[tuple[str, str], int] = {}
    skipped = 0
    for enrol_spk, test_utt, is_target in trials:
        if enrol_spk not in enroll_embs or test_utt not in trial_embs:
            skipped += 1
            continue
        if enrol_spk not in enrol_ids:
            enrol_ids.append(enrol_spk)
            enrol_vecs.append(enroll_embs[enrol_spk])
        if test_utt not in test_ids:
            test_ids.append(test_utt)
            test_vecs.append(trial_embs[test_utt])
        labels[(enrol_spk, test_utt)] = is_target

    if not labels:
        raise ValueError("No valid trials (all skipped).")

    enrol_mat = torch.stack(enrol_vecs)
    test_mat = torch.stack(test_vecs)
    sim = 1.0 - cosine_distances(X=enrol_mat.cpu().numpy(), Y=test_mat.cpu().numpy())
    enrol_idx = {k: i for i, k in enumerate(enrol_ids)}
    test_idx = {k: i for i, k in enumerate(test_ids)}

    pos, neg = [], []
    for (enrol_spk, test_utt), is_target in labels.items():
        s = float(sim[enrol_idx[enrol_spk], test_idx[test_utt]])
        (pos if is_target else neg).append(s)
    if not pos or not neg:
        raise ValueError(f"Degenerate trials: {len(pos)} target / {len(neg)} nontarget")
    eer, _ = EER(torch.tensor(pos), torch.tensor(neg))
    return float(eer) * 100.0, len(pos), len(neg), skipped


def write_wav_scp(wav_map: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for utt in sorted(wav_map):
            f.write(f"{utt} {wav_map[utt]}\n")


def materialize_anon_kaldi_dir(
    src_dir: Path,
    dst_dir: Path,
    anon_wav_map: dict[str, str],
    copy_meta: bool = True,
) -> Path:
    """Build anonymized Kaldi dir: new wav.scp + copied text/utt2spk/…"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    write_wav_scp(anon_wav_map, dst_dir / "wav.scp")
    if copy_meta:
        for name in ("text", "utt2spk", "spk2utt", "utt2dur", "spk2gender"):
            src = src_dir / name
            if src.is_file():
                dst = dst_dir / name
                if not dst.exists():
                    dst.write_bytes(src.read_bytes())
    return dst_dir
