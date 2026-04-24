"""LibriTTS dataset for LiveVoice (24 kHz).

Directory structure under `config.libritts_path`:
    {split}/
        {speaker_id}/
            {chapter_id}/
                {spk}_{chap}_{utt}.wav

Each item in self.items is one 4-second window from an utterance.
The last window of each utterance is zero-padded if shorter than 4s.

Each sample returns:
    {
        "reference_audio": (T,)   — same speaker, different utterance (random window)
        "content_audio":   (T,)   — deterministic window from content utterance
        "target_audio":    (T,)   — == content_audio
        "speaker_id":      str
        "content_hubert":  (T_frames, 768) float32 or None
    }
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import librosa
from torch.utils.data import Dataset

from .vctk_dataset import collate_fn  # shared collate — re-exported for datamodule



class LibriTTSDataset(Dataset):
    """LibriTTS speaker-paired dataset. Each item = one 4-second sliding window."""

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
            use_splits = self._normalize_splits(getattr(config, "libritts_train_splits", default_train))
        else:
            use_splits = self._normalize_splits(getattr(config, "libritts_val_splits", default_val))

        feats_base = getattr(config, "features_dir", None)
        self._feats_dir = Path(feats_base) / "libritts" if feats_base else None

        # speaker_id → list of (wav_path, utt_id)
        self.speaker_utts: dict[str, list[tuple[str, str]]] = {}
        for s in use_splits:
            split_dir = root / s
            if not split_dir.exists():
                print(f"[LibriTTSDataset] split not found: {split_dir} — skipping")
                continue
            for wav in sorted(split_dir.glob("**/*.wav")):
                spk = wav.parts[-3]  # grandparent dir = speaker_id
                utt_id = wav.stem
                self.speaker_utts.setdefault(spk, []).append((str(wav), utt_id))

        if self.pairing == "same_speaker":
            self.speaker_utts = {k: v for k, v in self.speaker_utts.items() if len(v) >= 2}

        # Build flat window list: (wav_path, utt_id, spk, window_idx)
        self.items: list[tuple[str, str, str, int]] = []
        for spk, utts in self.speaker_utts.items():
            for wav_path, utt_id in utts:
                n_windows = self._count_windows(wav_path)
                for w in range(n_windows):
                    self.items.append((wav_path, utt_id, spk, w))

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
            f"[LibriTTSDataset] split={split} windows={len(self.items)} "
            f"speakers={len(self.speaker_utts)}"
        )

    @staticmethod
    def _normalize_splits(value) -> tuple[str, ...]:
        """Accept tuple/list/set or comma-separated string for split names."""
        if isinstance(value, str):
            parts = [s.strip() for s in value.split(",") if s.strip()]
            return tuple(parts)
        return tuple(value)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        wav_path, utt_id, spk, window_idx = self.items[idx]

        spk_utts = self.speaker_utts[spk]
        if self.pairing == "reconstruct":
            ref_path = wav_path
        else:
            candidates = [(p, u) for p, u in spk_utts if p != wav_path]
            if not candidates:
                ref_path = wav_path
            elif self.split == "train":
                ref_path, _ = random.choice(candidates)
            else:
                ref_path, _ = next(iter(candidates))

        start_sample = window_idx * self.target_len

        try:
            ctn_wave = self._load_window(wav_path, start_sample)
            ref_wave = self._load_random_window(ref_path)
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

    # ------------------------------------------------------------------
    def _count_windows(self, path: str) -> int:
        """Number of non-overlapping target_len windows in this file (at target_sr)."""
        try:
            info = sf.info(path)
            n_target = int(info.frames * self.target_sr / info.samplerate)
            return max(1, math.ceil(n_target / self.target_len))
        except Exception:
            return 1

    def _read_audio(self, path: str) -> torch.Tensor:
        """Load mono audio resampled to target_sr. Returns 1-D float32 tensor."""
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
        return audio

    def _slice_audio(self, audio: torch.Tensor, start: int) -> torch.Tensor:
        """Extract target_len samples starting at start, pad if near end."""
        n = audio.numel()
        end = start + self.target_len
        if end <= n:
            audio = audio[start:end]
        elif start < n:
            chunk = audio[start:]
            audio = torch.nn.functional.pad(chunk, (0, self.target_len - chunk.numel()))
        else:
            audio = torch.zeros(self.target_len)
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        if self.split == "train":
            audio = audio * random.uniform(0.7, 1.0)
        return audio

    def _load_window(self, path: str, start_sample: int) -> torch.Tensor:
        """Load audio and return the window at start_sample (in target_sr units)."""
        return self._slice_audio(self._read_audio(path), start_sample)

    def _load_random_window(self, path: str) -> torch.Tensor:
        """Load a random (train) or first (val) window — used for reference audio."""
        audio = self._read_audio(path)
        n = audio.numel()
        if n > self.target_len and self.split == "train":
            start = random.randint(0, n - self.target_len)
        else:
            start = 0
        return self._slice_audio(audio, start)

    def _load_feats(self, spk: str, utt_id: str, start_sample: int) -> torch.Tensor | None:
        if self._feats_dir is None:
            return None
        feat_path = self._feats_dir / spk / f"{utt_id}.pt"
        if not feat_path.exists():
            return None
        try:
            data = torch.load(feat_path, map_location="cpu", weights_only=True)
            feats = data["feats"].float()          # (T_full, 768)
            audio_stride = int(data["audio_stride"])  # training_sr samples per HuBERT frame
            n_frames = int(round(self.target_len / audio_stride))
            start_frame = start_sample // audio_stride
            chunk = feats[start_frame : start_frame + n_frames]
            if chunk.shape[0] < n_frames:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, n_frames - chunk.shape[0]))
            return chunk
        except Exception:
            return None
