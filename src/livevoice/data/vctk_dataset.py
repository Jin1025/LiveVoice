"""VCTK dataset for LiveVoice.

Layout expected under `config.vctk_path`:
    wav48/
        p225/p225_001.wav  (original 48 kHz; resampled to config.sample_rate at load time)
        ...

Each sample returns:
    {
        "reference_audio": (T,)              — same speaker, different utterance
        "content_audio":   (T,)              — source utterance
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
    def __init__(self, config, split: str):
        self.config = config
        self.split = split
        self.target_sr = int(config.sample_rate)
        self.duration = float(config.audio_duration)
        self.target_len = int(round(self.duration * self.target_sr))
        self.pairing = str(getattr(config, "pairing", "same_speaker"))

        # SW2V and HuBERT use separate caches; pick the base by content_source.
        content_source = str(getattr(config, "content_source", "hubert")).lower()
        if content_source == "sw2v":
            feats_base = getattr(config, "sw2v_features_dir", None)
        else:
            feats_base = getattr(config, "features_dir", None)
        self._feats_dir = Path(feats_base) / "vctk" if feats_base else None
        if self._feats_dir is not None and not self._feats_dir.exists():
            print(
                f"[VCTKDataset] WARNING: {content_source} features dir set but {self._feats_dir} "
                f"does not exist → falling back to ONLINE extraction every step (slow). Extract "
                f"features there, or set the dir to None to make the online path explicit."
            )

        root = Path(config.vctk_path)
        wav_root = root / config.vctk_wav_dirname
        if not wav_root.exists():
            raise FileNotFoundError(f"VCTK wav dir not found: {wav_root}")

        speakers = sorted([p.name for p in wav_root.iterdir() if p.is_dir()])
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
            if utts:
                self.speaker_utts[spk] = utts

        self.items: list[tuple[str, str]] = []
        for spk, utts in self.speaker_utts.items():
            for u in utts:
                self.items.append((u, spk))

        if not self.items:
            raise ValueError(
                f"[VCTKDataset] split={split}: no usable utterances "
                f"(pairing={self.pairing}). Check vctk_path={root}."
            )

        random.seed(int(getattr(config, "seed", 42)) + (0 if split == "train" else 1))
        random.shuffle(self.items)

        max_n = getattr(config, "max_windows", None)
        if max_n is not None:
            self.items = self.items[: int(max_n)]

        print(
            f"[VCTKDataset] split={split} utterances={len(self.items)} "
            f"speakers={len(self.speaker_utts)} pairing={self.pairing}"
        )

    def _split_speakers(self, speakers, config):
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
        return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        content_path, speaker_id = self.items[idx]

        same_spk_utts = self.speaker_utts[speaker_id]
        if self.pairing == "reconstruct":
            ref_path = content_path
        elif self.split == "train":
            candidates = [p for p in same_spk_utts if p != content_path]
            ref_path = random.choice(candidates) if candidates else content_path
        else:
            ref_path = next((p for p in same_spk_utts if p != content_path), content_path)

        try:
            content_wave, start_sample = self._load_window(content_path)
            ref_wave, ref_start_sample = self._load_window(ref_path)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.items) - 1))

        utt_id = Path(content_path).stem
        content_hubert = self._load_feats(speaker_id, utt_id, start_sample)

        return {
            "reference_audio": ref_wave,
            "content_audio": content_wave,
            "target_audio": content_wave,
            "speaker_id": speaker_id,
            "content_hubert": content_hubert,
            "content_path": content_path,
            "content_start_sample": start_sample,
            "ref_path": ref_path,
            "ref_start_sample": ref_start_sample,
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
                # Snap window start to the HuBERT/codec frame grid (samples/frame = SR/50)
                # so a sliced cached feature window aligns 1:1 with this window's codec tokens.
                hop = max(1, self.target_sr // 50)
                start -= start % hop
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
            # HuBERT is always 50 fps regardless of training SR. samples-per-frame = SR/50.
            # ceil frame count + floor start_frame match the codec token grid; start_sample
            # is snapped to `hop` in _load_window so start_frame is exact (no drift).
            hop = max(1, self.target_sr // 50)
            n_frames = int(math.ceil(self.target_len / hop))
            start_frame = int(start_sample // hop)
            chunk = feats[start_frame : start_frame + n_frames]
            if chunk.shape[0] < n_frames:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, n_frames - chunk.shape[0]))
            return chunk
        except Exception:
            return None


def collate_fn(batch):
    if not isinstance(batch[0], dict):
        return torch.utils.data.default_collate(batch)
    out: dict = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if all(v is None for v in vals):
            out[key] = None
        elif any(v is None for v in vals):
            out[key] = None
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals, dim=0)
        else:
            out[key] = vals
    return out
