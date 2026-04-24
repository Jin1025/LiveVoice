"""Evaluation metrics for LiveVoice.

  compute_wer(hyp, ref)        — Word Error Rate (jiwer)
  compute_speaker_similarity   — cosine similarity of speaker embeddings
  batch_evaluate               — run both on a list of (generated, reference, content) triples
"""
from __future__ import annotations

import os
import torch
import torchaudio
from typing import Sequence


# ─────────────────────────────────────────────
#  WER (intelligibility)
# ─────────────────────────────────────────────

def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute WER between hypothesis and reference transcripts."""
    try:
        from jiwer import wer
        return float(wer(reference, hypothesis))
    except ImportError:
        raise ImportError("jiwer is required for WER: pip install jiwer")


# ─────────────────────────────────────────────
#  Speaker similarity (cosine)
# ─────────────────────────────────────────────

class SpeakerSimilarity:
    """ECAPA-TDNN speaker embedding cosine similarity via SpeechBrain."""

    def __init__(self, device: str = "cuda"):
        try:
            from speechbrain.pretrained import EncoderClassifier
        except ImportError:
            raise ImportError("speechbrain is required: pip install speechbrain")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        self.device = device

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        """audio: (T,) or (B, T)"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        return self.model.encode_batch(audio).squeeze(1)

    def similarity(self, audio_a: torch.Tensor, audio_b: torch.Tensor, sr: int = 16000) -> float:
        ea = self.embed(audio_a, sr)
        eb = self.embed(audio_b, sr)
        cos = torch.nn.functional.cosine_similarity(ea, eb, dim=-1)
        return float(cos.mean().item())


# ─────────────────────────────────────────────
#  Batch evaluation helper
# ─────────────────────────────────────────────

def batch_evaluate(
    generated_paths: Sequence[str],
    reference_paths: Sequence[str],
    content_paths: Sequence[str] | None = None,
    sr: int = 16000,
    use_speaker_sim: bool = False,
    device: str = "cuda",
) -> dict:
    """Compute aggregate metrics over a list of generated audio files.

    Args:
        generated_paths:  paths to generated audio files
        reference_paths:  paths to ground-truth reference speaker audio
        content_paths:    (optional) paths to source content audio; if provided,
                          WER is computed via Whisper transcription
        sr:               sample rate
        use_speaker_sim:  whether to compute ECAPA speaker cosine similarity

    Returns:
        dict with keys: speaker_sim (if requested), wer (if content provided)
    """
    results: dict = {}

    if use_speaker_sim:
        spk_sim = SpeakerSimilarity(device=device)
        sims = []
        for gen_p, ref_p in zip(generated_paths, reference_paths):
            gen_audio, _ = torchaudio.load(gen_p)
            ref_audio, _ = torchaudio.load(ref_p)
            gen_audio = gen_audio.mean(0)
            ref_audio = ref_audio.mean(0)
            sims.append(spk_sim.similarity(gen_audio, ref_audio, sr))
        results["speaker_sim_mean"] = float(sum(sims) / len(sims))
        results["speaker_sim_per_file"] = sims

    if content_paths is not None:
        try:
            import whisper
        except ImportError:
            print("[evaluation] whisper not installed — skipping WER. pip install openai-whisper")
            return results

        wmodel = whisper.load_model("base", device=device)
        wers = []
        for gen_p, ctn_p in zip(generated_paths, content_paths):
            gen_text = wmodel.transcribe(gen_p)["text"].strip().lower()
            ref_text = wmodel.transcribe(ctn_p)["text"].strip().lower()
            if ref_text:
                wers.append(compute_wer(gen_text, ref_text))
        if wers:
            results["wer_mean"] = float(sum(wers) / len(wers))
            results["wer_per_file"] = wers

    return results
