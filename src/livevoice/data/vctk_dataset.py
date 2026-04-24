"""VCTK dataset for LiveVoice.

Layout expected under `config.vctk_path`:
    wav48/
        p225/p225_001.wav  (original 48 kHz; resampled to config.sample_rate at load time)
        ...

Each item in self.items is one 4-second sliding window from an utterance.
The last window is zero-padded if the remaining audio is shorter.

Each sample returns:
    {
        "reference_audio": (T,)              — same speaker, different utterance (random window)
        "content_audio":   (T,)              — deterministic window from content utterance
        "target_audio":    (T,)              — == content_audio
        "speaker_id":      str
        "content_hubert":  (T_frames, 768) or None
    }
"""
from __future__ import annotations

import glob
import math
import random
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import librosa
from torch.utils.data import Dataset



class VCTKDataset(Dataset):
    """VCTK speaker-paired dataset. Each item = one 4-second sliding window."""

    def __init__(self, config, split: str):
        self.config = config
        self.split = split
        self.target_sr = int(config.sample_rate)
        self.duration = float(config.audio_duration)
        self.target_len = int(round(self.duration * self.target_sr))
        self.pairing = str(getattr(config, "pairing", "same_speaker"))

        feats_base = getattr(config, "features_dir", None)
        self._feats_dir = Path(feats_base) / "vctk" if feats_base else None

        root = Path(config.vctk_path)
        wav_root = root / config.vctk_wav_dirname
        if not wav_root.exists():
            raise FileNotFoundError(f"VCTK wav dir not found: {wav_root}")

        speakers = sorted([p.name for p in wav_root.iterdir() if p.is_dir()])
        if not speakers:
            raise ValueError(f"No speaker subdirs under {wav_root}")

        train_speakers, val_speakers = self._split_speakers(speakers, config)
        use_speakers = train_speakers if split == "train" else val_speakers

        exts = tuple(e.lower() for e in config.vctk_extensions)

        self.speaker_utts: dict[str, list[str]] = {}
        for spk in use_speakers:
            utts: list[str] = []
            for ext in exts:
                utts.extend(sorted(glob.glob(str(wav_root / spk / f"*{ext}"))))
            if self.pairing == "same_speaker" and len(utts) < 2:
                continue
            if len(utts) == 0:
                continue
            self.speaker_utts[spk] = utts

        # Build flat window list: (wav_path, spk, window_idx)
        self.items: list[tuple[str, str, int]] = []
        for spk, utts in self.speaker_utts.items():
            for wav_path in utts:
                n_windows = self._count_windows(wav_path)
                for w in range(n_windows):
                    self.items.append((wav_path, spk, w))

        if len(self.items) == 0:
            raise ValueError(
                f"[VCTKDataset] split={split}: no usable windows "
                f"(pairing={self.pairing}, {len(use_speakers)} speakers). "
                f"Check vctk_path={root}."
            )

        rng = random.Random(int(getattr(config, "seed", 42)) + (0 if split == "train" else 1))
        rng.shuffle(self.items)

        max_n = getattr(config, "max_windows", None)
        if max_n is not None:
            self.items = self.items[: int(max_n)]

        print(
            f"[VCTKDataset] split={split} windows={len(self.items)} "
            f"speakers={len(self.speaker_utts)} pairing={self.pairing}"
        )

    # ------------------------------------------------------------------
    def _split_speakers(self, speakers: list[str], config) -> tuple[list[str], list[str]]:
        explicit = tuple(getattr(config, "vctk_val_speakers", ()) or ())
        if explicit:
            val = [s for s in speakers if s in set(explicit)]
            train = [s for s in speakers if s not in set(explicit)]
            return train, val

        n_val = int(getattr(config, "vctk_val_speaker_count", 8))
        n_val = max(0, min(n_val, len(speakers) - 1))
        rng = random.Random(int(getattr(config, "seed", 42)))
        shuffled = list(speakers)
        rng.shuffle(shuffled)
        val = sorted(shuffled[:n_val])
        train = sorted(shuffled[n_val:])
        return train, val

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        wav_path, speaker_id, window_idx = self.items[idx]

        same_spk_utts = self.speaker_utts[speaker_id]
        if self.pairing == "reconstruct":
            ref_path = wav_path
        else:
            candidates = [p for p in same_spk_utts if p != wav_path]
            if not candidates:
                ref_path = wav_path
            elif self.split == "train":
                ref_path = random.choice(candidates)
            else:
                ref_path = next(iter(candidates))

        start_sample = window_idx * self.target_len

        try:
            content_wave = self._load_window(wav_path, start_sample)
            ref_wave = self._load_random_window(ref_path)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.items) - 1))

        utt_id = Path(wav_path).stem
        content_hubert = self._load_feats(speaker_id, utt_id, start_sample)

        return {
            "reference_audio": ref_wave,
            "content_audio": content_wave,
            "target_audio": content_wave,
            "speaker_id": speaker_id,
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
        """Load precomputed HuBERT features and crop to match the audio window."""
        if self._feats_dir is None:
            return None
        feat_path = self._feats_dir / spk / f"{utt_id}.pt"
        if not feat_path.exists():
            return None
        try:
            data = torch.load(feat_path, map_location="cpu", weights_only=True)
            feats = data["feats"].float()             # (T_full, 768)
            audio_stride = int(data["audio_stride"])  # training_sr samples per HuBERT frame
            n_frames = int(round(self.target_len / audio_stride))
            start_frame = start_sample // audio_stride
            chunk = feats[start_frame : start_frame + n_frames]
            if chunk.shape[0] < n_frames:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, n_frames - chunk.shape[0]))
            return chunk
        except Exception:
            return None


def collate_fn(batch):
    """Collate dict samples, handling None values for content_hubert."""
    if not isinstance(batch[0], dict):
        return torch.utils.data.default_collate(batch)

    out: dict = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if all(v is None for v in vals):
            out[key] = None
        elif any(v is None for v in vals):
            out[key] = None  # mixed None/tensor → skip
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out
