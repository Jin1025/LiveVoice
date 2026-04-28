from pathlib import Path
import soundfile as sf
import torch
import torchaudio


def _peak_norm(x: torch.Tensor) -> torch.Tensor:
    peak = x.abs().max()
    if peak > 1e-6:
        return x / peak
    return x


def _apply_vtln(x: torch.Tensor, sr: int, alpha: float) -> torch.Tensor:
    """VTLN-style formant shift via resample trick."""
    if abs(alpha - 1.0) <= 0.02:
        return x
    orig_len = x.shape[-1]
    shifted_sr = max(8000, int(sr * alpha))
    y = torchaudio.functional.resample(x, sr, shifted_sr)
    y = torchaudio.functional.resample(y, shifted_sr, sr)
    if y.shape[-1] > orig_len:
        y = y[..., :orig_len]
    elif y.shape[-1] < orig_len:
        y = torch.nn.functional.pad(y, (0, orig_len - y.shape[-1]))
    return y


def main():
    in_path = Path("/workspace/livevoice/src/output/libritts_no_VTLN/250/ref_p250_002.wav")
    out_dir = in_path.parent.parent / "perturb_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav, sr = sf.read(str(in_path), dtype="float32", always_2d=True)
    mono = torch.from_numpy(wav).mean(dim=1).unsqueeze(0)  # (1, T)

    # Force integer semitone only.
    semitones = list(range(-4, 5))

    # Deterministic VTLN alphas for on/off comparison.
    alpha_off = 1.0
    alpha_on = 1.12

    saved = []
    for n_steps in semitones:
        x = mono
        if n_steps != 0:
            x = torchaudio.functional.pitch_shift(x, sr, float(n_steps))

        # VTLN OFF
        y_off = _peak_norm(_apply_vtln(x, sr, alpha_off)).squeeze(0).cpu().numpy()
        off_path = out_dir / f"ref_p250_002_pitch_{n_steps:+d}_vtln_off.wav"
        sf.write(str(off_path), y_off, sr)
        saved.append(off_path)

        # VTLN ON
        y_on = _peak_norm(_apply_vtln(x, sr, alpha_on)).squeeze(0).cpu().numpy()
        on_path = out_dir / f"ref_p250_002_pitch_{n_steps:+d}_vtln_on.wav"
        sf.write(str(on_path), y_on, sr)
        saved.append(on_path)

    print(f"INPUT: {in_path}")
    print(f"OUT_DIR: {out_dir}")
    print(f"Saved {len(saved)} files.")
    print(f"Semitones: {semitones} (integer only)")
    print(f"VTLN OFF alpha={alpha_off}, ON alpha={alpha_on}")
    for p in saved:
        print(f"SAVED: {p}")


if __name__ == "__main__":
    main()