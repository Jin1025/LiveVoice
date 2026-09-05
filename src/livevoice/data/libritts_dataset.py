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

import math
import random
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import librosa
from torch.utils.data import Dataset

from .vctk_dataset import collate_fn  # shared — re-exported for datamodule
from livevoice.model.phoneme_vocab import PAD_ID


class LibriTTSDataset(Dataset):
    def __init__(self, config, split: str):
        self.config = config
        self.split = split
        self.target_sr = int(config.sample_rate)
        self.duration = float(config.audio_duration)
        self.target_len = int(round(self.duration * self.target_sr))
        self.pairing = str(getattr(config, "pairing", "same_speaker"))
        self.peak_normalize = bool(getattr(config, "audio_peak_normalize", True))

        root = Path(config.libritts_path)

        default_train = ("train-clean-100", "train-clean-360")
        default_val = ("dev-clean", "dev-other")
        if split == "train":
            use_splits = tuple(getattr(config, "libritts_train_splits", default_train))
        else:
            use_splits = tuple(getattr(config, "libritts_val_splits", default_val))

        # Pick the feature-cache base by content_source: SW2V and HuBERT have separate
        # caches (different encoders / dims), so they must not share a directory.
        content_source = str(getattr(config, "content_source", "hubert")).lower()
        if content_source == "sw2v":
            feats_base = getattr(config, "sw2v_features_dir", None)
        elif content_source == "zipformer":
            feats_base = getattr(config, "zipformer_features_dir", None)
        elif content_source == "fastconformer":
            feats_base = getattr(config, "fastconformer_features_dir", None)
        else:
            feats_base = getattr(config, "features_dir", None)
        self._feats_dir = Path(feats_base) / "libritts" if feats_base else None
        if self._feats_dir is not None and not self._feats_dir.exists():
            print(
                f"[LibriTTSDataset] WARNING: {content_source} features dir set but {self._feats_dir} "
                f"does not exist → falling back to ONLINE extraction every step (slow). Extract "
                f"features there, or set the dir to None to make the online path explicit."
            )

        # FULL-utterance features for ASR/GRL use the SAME sw2v cache as the main path
        # (sw2v_features_dir). If that's None (online main path), set sw2v_full_online=True
        # to extract the full features live instead.
        self._full_feats_dir = self._feats_dir

        # Seq2seq ASR supervision (config.use_asr_supervision): needs the FULL (un-cropped)
        # sw2v feature cache — reuses self._feats_dir, only valid for content_source=="sw2v"
        # — plus a precomputed phoneme-id cache (scripts/extract_phonemes.py). Both are
        # no-ops (None) when disabled, so the default path has zero extra I/O.
        self.use_asr_supervision = bool(getattr(config, "use_asr_supervision", False))
        self.use_speaker_grl = bool(getattr(config, "use_speaker_grl", False))
        self.asr_max_content_frames = int(getattr(config, "asr_max_content_frames", 750))
        self.asr_max_phoneme_len = int(getattr(config, "asr_max_phoneme_len", 300))
        self._phoneme_dir = None
        # Both ASR supervision and the speaker-GRL adversary run on the FULL
        # (un-cropped) sw2v cache — only meaningful for content_source=="sw2v".
        if (self.use_asr_supervision or self.use_speaker_grl) and content_source not in (
                "sw2v", "zipformer"):
            print(
                "[LibriTTSDataset] WARNING: use_asr_supervision/use_speaker_grl set but "
                f"content_source={content_source!r} is not a continuous-encoder source "
                "(sw2v/zipformer) with a full-utterance cache; disabling for this instance."
            )
            self.use_asr_supervision = False
            self.use_speaker_grl = False
        if self.use_asr_supervision:
            phon_base = getattr(config, "phoneme_cache_dir", None)
            self._phoneme_dir = Path(phon_base) / "libritts" if phon_base else None
            if self._phoneme_dir is not None and not self._phoneme_dir.exists():
                print(
                    f"[LibriTTSDataset] WARNING: phoneme_cache_dir set but "
                    f"{self._phoneme_dir} does not exist → run scripts/extract_phonemes.py "
                    f"first. ASR supervision will silently skip items with no cache."
                )
        # Speaker-GRL: deterministic train-split speaker vocab (same mapping the model's
        # classifier is sized from — see data/speaker_vocab.py and train.py).
        self._need_full_feats = self.use_asr_supervision or self.use_speaker_grl
        # ASR/GRL full-utterance features — precedence:
        #   sw2v_full_online=True   → ALWAYS online (run encoder on full audio; ignore cache)
        #   else cache present      → use the sw2v_features_dir cache
        #   else (cache missing)    → WARN and fall back to online (never silently 0)
        cache_ok = self._full_feats_dir is not None and self._full_feats_dir.exists()
        if bool(getattr(config, "sw2v_full_online", False)):
            self._full_online = True
        elif cache_ok:
            self._full_online = False
        else:
            if self._need_full_feats:
                print(
                    f"[LibriTTSDataset] WARNING: sw2v cache ({self._full_feats_dir}) missing/None "
                    f"and sw2v_full_online=False → falling back to ONLINE full-feature extraction "
                    f"for ASR/GRL (slower; set sw2v_features_dir to skip, or sw2v_full_online=True "
                    f"to silence this)."
                )
            self._full_online = True
        if self._need_full_feats:
            src = "ONLINE (encoder on full audio)" if self._full_online else f"cache {self._full_feats_dir}"
            print(f"[LibriTTSDataset] ASR/GRL full features: {src}")
        self._speaker_to_idx: dict[str, int] = {}
        if self.use_speaker_grl:
            from livevoice.data.speaker_vocab import build_libritts_grl_label_map
            self._speaker_to_idx, n_grl = build_libritts_grl_label_map(config)
            kind = "clusters" if int(getattr(config, "grl_num_clusters", 0)) > 0 else "speakers"
            print(f"[LibriTTSDataset] speaker-GRL label map: {n_grl} {kind}")

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

        # Expresso (same layout as LibriTTS, produced by prepare_expresso.py)
        expresso_path = str(getattr(config, "expresso_path", "") or "")
        self._expresso_spk_set: set[str] = set()
        if expresso_path:
            exp_root = Path(expresso_path)
            if split == "train":
                exp_splits = tuple(getattr(config, "expresso_train_splits", ("train",)))
            else:
                exp_splits = tuple(getattr(config, "expresso_val_splits", ("dev",)))
            n_before = sum(len(v) for v in self.speaker_utts.values())
            for s in exp_splits:
                split_dir = exp_root / s
                if not split_dir.exists():
                    print(f"[LibriTTSDataset] expresso split not found: {split_dir} — skipping")
                    continue
                for wav in sorted(split_dir.glob("**/*.wav")):
                    spk = wav.parts[-3]
                    utt_id = wav.stem
                    self.speaker_utts.setdefault(spk, []).append((str(wav), utt_id))
                    self._expresso_spk_set.add(spk)
            n_after = sum(len(v) for v in self.speaker_utts.values())
            print(f"[LibriTTSDataset] +expresso: {n_after - n_before} utterances, "
                  f"{len(self._expresso_spk_set)} speakers")

        if self.pairing == "same_speaker":
            self.speaker_utts = {k: v for k, v in self.speaker_utts.items() if len(v) >= 2}

        self.items: list[tuple[str, str, str]] = []
        for spk, utts in self.speaker_utts.items():
            for wav_path, utt_id in utts:
                self.items.append((wav_path, utt_id, spk))

        # pairing="same_utterance_window": content and reference are two NON-OVERLAPPING
        # windows of the SAME utterance (identical recording session → no cross-utterance
        # channel mismatch). Needs files >= 2 windows (+ optional gap), so pre-filter by
        # duration (sf.info reads only the header → cheap). Only for this mode.
        if self.pairing in ("same_utterance_window", "same_utterance_continuation"):
            # continuation uses adjacent windows, so no inter-window gap applies
            gap = (0 if self.pairing == "same_utterance_continuation"
                   else int(float(getattr(config, "same_utt_min_gap_seconds", 0.0)) * self.target_sr))
            min_sec = (2 * self.target_len + gap) / self.target_sr
            kept = []
            for wav_path, utt_id, spk in self.items:
                try:
                    info = sf.info(wav_path)
                    if info.frames / info.samplerate >= min_sec:
                        kept.append((wav_path, utt_id, spk))
                except Exception:
                    continue
            print(f"[LibriTTSDataset] {self.pairing}: kept {len(kept)}/{len(self.items)} "
                  f"utterances >= {min_sec:.1f}s (2×{self.duration:.1f}s window + gap)")
            self.items = kept

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

        # same_utterance_*: content + reference are two windows of THIS file — disjoint at
        # random positions ("window"), or adjacent with reference first ("continuation").
        if self.pairing in ("same_utterance_window", "same_utterance_continuation"):
            try:
                two = (
                    self._load_adjacent_windows_same_utt(wav_path)
                    if self.pairing == "same_utterance_continuation"
                    else self._load_two_windows_same_utt(wav_path)
                )
            except Exception:
                two = None
            if two is None:  # too short despite the filter, or read error → resample another item
                return self.__getitem__(random.randint(0, len(self.items) - 1))
            ctn_wave, start_sample, ref_wave, ref_start_sample = two
            ref_path = wav_path
            content_hubert = self._load_feats(spk, utt_id, start_sample)
            item = {
                "reference_audio": ref_wave,
                "content_audio": ctn_wave,
                "target_audio": ctn_wave,
                "speaker_id": spk,
                "content_hubert": content_hubert,
                # Reference content for config.codec_prompt_content, sliced from the SAME
                # cache as the target's — same normalisation, same baked-in perturbation.
                # Extracting it live from ref_wave instead would use a different gain (the
                # cache is normalised per FULL utterance) and cost an extra encoder pass.
                "reference_feats": self._load_feats(spk, utt_id, ref_start_sample),
                "content_path": wav_path,
                "content_start_sample": start_sample,
                "ref_path": ref_path,
                "ref_start_sample": ref_start_sample,
            }
            self._attach_full_feats(item, spk, utt_id, wav_path)
            return item

        if self.pairing == "reconstruct":
            ref_path = wav_path
        elif self.split == "train":
            candidates = [(p, u) for p, u in spk_utts if p != wav_path]
            ref_path, ref_utt_id = random.choice(candidates) if candidates else (wav_path, utt_id)
        else:
            ref_path, ref_utt_id = next(
                ((p, u) for p, u in spk_utts if p != wav_path), (wav_path, utt_id))

        try:
            ctn_wave, start_sample = self._load_window(wav_path)
            ref_wave, ref_start_sample = self._load_window(ref_path)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.items) - 1))

        content_hubert = self._load_feats(spk, utt_id, start_sample)

        item = {
            "reference_audio": ref_wave,
            "content_audio": ctn_wave,
            "target_audio": ctn_wave,
            "speaker_id": spk,
            "content_hubert": content_hubert,
            # See the same_utterance_* branch — cached reference content for
            # config.codec_prompt_content (the reference is a different utterance here,
            # so it comes from that utterance's cache entry).
            "reference_feats": self._load_feats(spk, ref_utt_id, ref_start_sample),
            "content_path": wav_path,
            "content_start_sample": start_sample,
            "ref_path": ref_path,
            "ref_start_sample": ref_start_sample,
        }

        self._attach_full_feats(item, spk, utt_id, wav_path)
        return item

    def _attach_full_feats(self, item, spk, utt_id, wav_path):
        """Attach full-utterance content (cache or online audio) + phonemes + speaker label
        for ASR/GRL, when enabled. Shared by every pairing mode."""
        if not self._need_full_feats:
            return
        full_audio = None
        if self._full_online:
            full_audio, feats_len = self._load_full_audio(wav_path)  # audio + frame count
            feats_full = None
        else:
            feats_full, feats_len = self._load_feats_full(spk, utt_id)  # cached features
        # ASR needs phonemes too; GRL needs only the full content + speaker label.
        phoneme_ids = self._load_phonemes(spk, utt_id) if self.use_asr_supervision else None
        # Keep the group consistent: need content (feats OR audio) AND, if ASR, phonemes.
        # collate_fn nulls the whole key if any item is None, so a partial item would
        # corrupt the batch — drop the whole group here instead.
        have_content = (feats_full is not None) or (full_audio is not None)
        if not have_content or (self.use_asr_supervision and phoneme_ids is None):
            feats_full = feats_len = phoneme_ids = full_audio = None
        item["content_feats_full"] = feats_full
        item["content_feats_full_len"] = feats_len
        item["content_full_audio"] = full_audio
        if self.use_asr_supervision:
            item["phoneme_ids"] = phoneme_ids
        if self.use_speaker_grl:
            # -1 for out-of-vocab (val speakers are unseen); GRL is computed in
            # training_step only, so val labels are never consumed.
            item["speaker_label"] = torch.tensor(
                self._speaker_to_idx.get(spk, -1), dtype=torch.long
            )

    def _load_two_windows_same_utt(self, path: str):
        """Two NON-OVERLAPPING windows from the same utterance → (content_wave,
        content_start, ref_wave, ref_start). Returns None if the file is too short.
        The content start is snapped to the codec frame grid so cached-feature slicing
        (_load_feats) stays aligned; the reference window is audio-only so it doesn't
        need to align. Roles are randomized so content isn't always the earlier half."""
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
        L = self.target_len
        hop = max(1, self.target_sr // 50)
        gap = int(float(getattr(self.config, "same_utt_min_gap_seconds", 0.0)) * self.target_sr)
        if n < 2 * L + gap:
            return None

        # pick two disjoint intervals [s1, s1+L] and [s2, s2+L], s1 < s2, gap between them
        s1 = random.randint(0, n - 2 * L - gap)
        s2 = random.randint(s1 + L + gap, n - L)
        s1 -= s1 % hop
        s2 -= s2 % hop
        if s2 < s1 + L:          # snap-down could have caused a 1-frame overlap
            s2 = s1 + L          # L is a multiple of hop, so this stays grid-aligned

        def _clip(a, s):
            w = a[s : s + L]
            w = w / (torch.max(torch.abs(w)) + 1e-8)
            return w

        w1, w2 = _clip(audio, s1), _clip(audio, s2)
        if random.random() < 0.5:
            return w1, s1, w2, s2   # content=earlier, ref=later
        return w2, s2, w1, s1       # content=later, ref=earlier

    def _load_adjacent_windows_same_utt(self, path: str):
        """Two ADJACENT windows of one utterance: reference immediately precedes content.

        For codec_prompt continuation the joint AR stream is [reference ; target], so it
        should be one genuinely continuous piece of audio. `same_utterance_window` picks two
        windows at random positions in random order, which leaves a seam the model can learn
        to reset at — the very thing continuation is meant to remove.

        Returns (content, content_start, reference, reference_start) with
        reference = audio[s : s+L] and content = audio[s+L : s+2L].
        """
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
        L = self.target_len
        hop = max(1, self.target_sr // 50)
        if n < 2 * L:
            return None

        s = random.randint(0, n - 2 * L)
        s -= s % hop                      # keep both windows on the codec frame grid

        # Normalise the WHOLE 2L span with ONE gain. Normalising each window separately
        # would put a level jump exactly at the prompt->target boundary — an artificial
        # discontinuity teaching the model that the channel may change there.
        span = audio[s : s + 2 * L]
        span = span / (torch.max(torch.abs(span)) + 1e-8)

        ref_w, ctn_w = span[:L], span[L:]
        return ctn_w, s + L, ref_w, s

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
                # Snap the window start to the HuBERT/codec frame grid (samples per
                # frame = target_sr/50) so a sliced cached feature window lines up 1:1
                # with the codec tokens of this window (no sub-frame content↔token drift).
                hop = max(1, self.target_sr // 50)
                start -= start % hop
            audio = audio[start : start + self.target_len]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.target_len - n))

        if self.peak_normalize:
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
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

    def _load_full_audio(self, path: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Full-utterance waveform for ONLINE ASR/GRL feature extraction (the model runs
        the sw2v encoder on this). Resampled to target_sr, capped/zero-padded to a fixed
        asr_max_content_frames*hop length so the batch stacks; returns (audio, n_frames)."""
        try:
            with sf.SoundFile(path) as f:
                audio_np = f.read(dtype="float32", always_2d=True)
                sr = f.samplerate
            audio = torch.from_numpy(audio_np).float().mean(dim=1)
        except Exception:
            try:
                audio_np, sr = librosa.load(path, sr=None, mono=True)
                audio = torch.from_numpy(audio_np.astype("float32"))
                sr = int(sr)
            except Exception:
                return None, None
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio, sr, self.target_sr)
        hop = max(1, self.target_sr // 50)
        cap_frames = self.asr_max_content_frames
        cap_samples = cap_frames * hop
        n = audio.numel()
        n_frames = min(cap_frames, int(math.ceil(n / hop)))
        if n > cap_samples:
            audio = audio[:cap_samples]
        elif n < cap_samples:
            audio = torch.nn.functional.pad(audio, (0, cap_samples - n))
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
        return audio, torch.tensor(n_frames, dtype=torch.long)

    def _load_feats_full(self, spk: str, utt_id: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Full (un-cropped) sw2v feature cache for ASR supervision — unlike `_load_feats`,
        this does NOT slice to the training window: text labels are utterance-level with
        no timestamps, so the ASR loss must see everything the label describes.
        Zero-padded/truncated to asr_max_content_frames; returns (feats, true_len)."""
        if self._full_feats_dir is None:
            return None, None
        feat_path = self._full_feats_dir / spk / f"{utt_id}.pt"
        if not feat_path.exists():
            return None, None
        try:
            data = torch.load(feat_path, map_location="cpu", weights_only=True)
            feats = data["feats"].float()
            n = feats.shape[0]
            cap = self.asr_max_content_frames
            if n > cap:
                feats = feats[:cap]
                n = cap
            elif n < cap:
                feats = torch.nn.functional.pad(feats, (0, 0, 0, cap - n))
            return feats, torch.tensor(n, dtype=torch.long)
        except Exception:
            return None, None

    def _load_phonemes(self, spk: str, utt_id: str) -> torch.Tensor | None:
        """Precomputed phoneme-id sequence (scripts/extract_phonemes.py): [BOS, ph..., EOS].
        Pads with PAD_ID to asr_max_phoneme_len (truncates, keeping the trailing EOS)."""
        if self._phoneme_dir is None:
            return None
        ph_path = self._phoneme_dir / spk / f"{utt_id}.pt"
        if not ph_path.exists():
            return None
        try:
            ids = torch.load(ph_path, map_location="cpu", weights_only=True)
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids, dtype=torch.long)
            ids = ids.to(torch.long)
            cap = self.asr_max_phoneme_len
            n = ids.shape[0]
            if n > cap:
                ids = torch.cat([ids[: cap - 1], ids[-1:]])  # keep trailing EOS
            elif n < cap:
                ids = torch.nn.functional.pad(ids, (0, cap - n), value=PAD_ID)
            return ids
        except Exception:
            return None

