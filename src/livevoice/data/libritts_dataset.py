"""LibriTTS dataset for LiveVoice (24 kHz).

Directory structure under `config.libritts_path`:
    {split}/
        {speaker_id}/
            {chapter_id}/
                {spk}_{chap}_{utt}.wav

Each sample returns:
    {
        "reference_audio": (T,)   — same speaker, different utterance
        "content_audio":   (T,)   — the utterance to reproduce
        "target_audio":    (T,)   — == content_audio
        "speaker_id":      str
        "content_hubert":  (T_frames, 768) float32 or None
    }
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import librosa
from torch.utils.data import Dataset

from .vctk_dataset import collate_fn  # shared — re-exported for datamodule


class LibriTTSDataset(Dataset):
    def __init__(self, config, split: str):
        self.config = config
        self.split = split
        self.target_sr = int(config.sample_rate)
        self.duration = float(config.audio_duration)
        self.target_len = int(round(self.duration * self.target_sr))
        self.pairing = str(getattr(config, "pairing", "same_speaker"))

        root = Path(config.libritts_path)

        default_train = ("train-clean-100", "train-clean-360", "train-other-500")
        default_val = ("dev-clean", "dev-other")
        if split == "train":
            use_splits = tuple(getattr(config, "libritts_train_splits", default_train))
        else:
            use_splits = tuple(getattr(config, "libritts_val_splits", default_val))

        feats_base = getattr(config, "features_dir", None)
        self._feats_dir = Path(feats_base) / "libritts" if feats_base else None

        self.speaker_utts: dict[str, list[tuple[str, str]]] = {}
        for s in use_splits:
            split_dir = root / s
            if not split_dir.exists():
                print(f"[LibriTTSDataset] split not found: {split_dir} — skipping")
                continue
            for wav in sorted(split_dir.glob("**/*.wav")):
                spk = wav.parts[-3]
                utt_id = wav.stem
                self.speaker_utts.setdefault(spk, []).append((str(wav), utt_id))

        if self.pairing == "same_speaker":
            self.speaker_utts = {k: v for k, v in self.speaker_utts.items() if len(v) >= 2}

        self.items: list[tuple[str, str, str]] = []
        for spk, utts in self.speaker_utts.items():
            for wav_path, utt_id in utts:
                self.items.append((wav_path, utt_id, spk))

        if not self.items:
            raise ValueError(
                f"[LibriTTSDataset] split={split}: no usable items. "
                f"Check libritts_path={root} and splits {use_splits}."
            )

        rng = random.Random(int(getattr(config, "seed", 42)) + (0 if split == "train" else 1))
        rng.shuffle(self.items)

        max_n = getattr(config, "max_windows", None)
        if max_n is not None:
            self.items = self.items[: int(max_n)]

        print(
            f"[LibriTTSDataset] split={split} utterances={len(self.items)} "
            f"speakers={len(self.speaker_utts)}"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, utt_id, spk = self.items[idx]

        spk_utts = self.speaker_utts[spk]
        if self.pairing == "reconstruct":
            ref_path = wav_path
        elif self.split == "train":
            candidates = [(p, u) for p, u in spk_utts if p != wav_path]
            ref_path, _ = random.choice(candidates) if candidates else (wav_path, utt_id)
        else:
            ref_path, _ = next(((p, u) for p, u in spk_utts if p != wav_path), (wav_path, utt_id))

        try:
            ctn_wave, start_sample = self._load_window(wav_path)
            ref_wave, _ = self._load_window(ref_path)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.items) - 1))

        content_hubert = self._load_feats(spk, utt_id, start_sample)

        return {
            "reference_audio": ref_wave,
            "content_audio": ctn_wave,
            "target_audio": ctn_wave,
            "speaker_id": spk,
            "content_hubert": content_hubert,
        }

    def _load_window(self, path: str) -> tuple[torch.Tensor, int]:
        try:
            with sf.SoundFile(path) as f:
                audio_np = f.read(dtype="float32", always_2d=True)
                sr = f.samplerate
            audio = torch.from_numpy(audio_np).float().mean(dim=1)
        except Exception:
            audio_np, sr = librosa.load(path, sr=None, mono=True)
            audio = torch.from_numpy(audio_np.astype("float32"))
            sr = int(sr)

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)

        n = audio.numel()
        start = 0
        if n >= self.target_len:
            if self.split == "train" and n > self.target_len:
                start = random.randint(0, n - self.target_len)
            audio = audio[start : start + self.target_len]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.target_len - n))

        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        if self.split == "train":
            audio = audio * random.uniform(0.7, 1.0)
        return audio, start

    def _load_feats(self, spk: str, utt_id: str, start_sample: int) -> torch.Tensor | None:
        if self._feats_dir is None:
            return None
        feat_path = self._feats_dir / spk / f"{utt_id}.pt"
        if not feat_path.exists():
            return None
        try:
            data = torch.load(feat_path, map_location="cpu", weights_only=True)
            feats = data["feats"].float()
            # HuBERT is always 50 fps regardless of training SR — compute from target_sr directly
            n_frames = int(round(self.target_len * 50 / self.target_sr))
            start_frame = int(start_sample * 50 / self.target_sr)
            chunk = feats[start_frame : start_frame + n_frames]
            if chunk.shape[0] < n_frames:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, n_frames - chunk.shape[0]))
            return chunk
        except Exception:
            return None
