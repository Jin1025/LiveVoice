"""Speaker-probing on the SW2V content representation.

Question this answers: *how much speaker identity is still linearly decodable
from the content features the VC model actually consumes?* If a cheap linear
probe classifies the speaker from content features, the content path is leaking
timbre — which is exactly the failure mode behind "output stays glued to the
source speaker". This is the diagnostic to run before adding any speaker
discriminator / cosine loss (see the frame-discriminator discussion): it tells
you whether disentanglement is the problem, and whether a *frame* really is
timbre-only (frame-level probe) or still content-entangled.

Three conditions, evaluated on the SAME set of utterances so the numbers are
comparable:
  clean    : raw audio            -> sw2v encoder
  perturb  : ContentPerturbation  -> sw2v encoder   (on-the-fly, .train() mode)
  cache    : precomputed perturbed feature cache on disk (config.sw2v_features_dir)

Two probe granularities:
  utterance : mean-pool over time -> one vector per utterance  ("is speaker info present at all")
  frame     : each frame classified independently              ("can a single frame reveal the speaker")

Optionally (`--ckpt`) also probes AFTER the trained bottleneck
(sw2v_proj [+ content_fsq]) to check whether FSQ/ASR-supervision actually
removed speaker info the raw encoder still carries.

Closed-set speaker classification: the same speakers appear in train and test,
but split by UTTERANCE (held-out utterances) so the probe can't memorize.
Chance = 1 / num_speakers. Run under the `sound` conda env with absolute paths:

    conda run -n sound python /workspace/LiveVoice/src/debug/probe_speaker_in_content.py \
        --num_speakers 30 --utts_per_speaker 20 --split dev-clean
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- make `livevoice` importable when run as a plain script -------------------
_SRC = Path(__file__).resolve().parents[1]  # .../LiveVoice/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from livevoice.config import LiveVoiceConfig  # noqa: E402
from livevoice.model.sw2v_content import Sw2vContentEncoder  # noqa: E402
from livevoice.model.content_supervision import apply_content_cmn  # noqa: E402
from livevoice.model.content_extractor import HuBERTContentExtractor  # noqa: E402
from livevoice.model.content_perturbation import ContentPerturbation  # noqa: E402


def feats_base_dir(config, content_source: str) -> Path:
    """Cache root for the chosen content source (sw2v vs hubert have separate caches)."""
    base = config.sw2v_features_dir if content_source == "sw2v" else config.features_dir
    return Path(base) / "libritts"


# ----------------------------------------------------------------------------- data
def discover(config, feats_dir: Path, split: str, num_speakers: int, utts_per_speaker: int,
             require_cache: bool, seed: int):
    """Return items = [(spk, wav_path, utt_id), ...], balanced at utts_per_speaker per
    speaker, all sharing a cache file if required."""
    root = Path(config.libritts_path) / split
    if not root.exists():
        raise FileNotFoundError(f"split dir not found: {root}")

    per_spk: dict[str, list[tuple[str, str]]] = {}
    for wav in sorted(root.glob("**/*.wav")):
        spk = wav.parts[-3]
        utt_id = wav.stem
        if require_cache and not (feats_dir / spk / f"{utt_id}.pt").exists():
            continue
        per_spk.setdefault(spk, []).append((str(wav), utt_id))

    rng = random.Random(seed)
    usable = [s for s, u in per_spk.items() if len(u) >= utts_per_speaker]
    usable.sort()
    rng.shuffle(usable)
    chosen = usable[:num_speakers]
    if len(chosen) < num_speakers:
        print(f"[probe] WARNING: only {len(chosen)} speakers have >= "
              f"{utts_per_speaker} usable utts (asked {num_speakers}).")

    items: list[tuple[str, str, str]] = []
    for spk in chosen:
        utts = per_spk[spk][:]
        rng.shuffle(utts)
        for wav_path, utt_id in utts[:utts_per_speaker]:
            items.append((spk, wav_path, utt_id))
    return items


def load_audio(path: str, target_sr: int, max_seconds: float | None) -> torch.Tensor:
    audio_np, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio_np).float().mean(dim=1)  # mono
    if sr != target_sr:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    if max_seconds is not None:
        cap = int(max_seconds * target_sr)
        if audio.numel() > cap:
            audio = audio[:cap]
    peak = audio.abs().max()
    if peak > 1e-8:
        audio = audio / peak
    return audio


# ----------------------------------------------------------------------------- trained bottleneck (optional)
class _Bottleneck(nn.Module):
    """sw2v_proj [+ content_fsq] reconstructed from a checkpoint, for post-bottleneck probing."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        from livevoice.utils.checkpoint import infer_content_fsq_from_ckpt
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = obj.get("state_dict", obj)
        w = sd.get("model.sw2v_proj.weight")
        b = sd.get("model.sw2v_proj.bias")
        if w is None:
            raise RuntimeError("checkpoint has no model.sw2v_proj.* — not an sw2v model?")
        out_dim, in_dim = w.shape

        # Optional deep content refiner (before sw2v_proj) — reconstruct from ckpt so the
        # probe measures the SAME representation training shaped.
        self.refiner = None
        ref_keys = [k for k in sd if k.startswith("model.content_refiner.blocks.")]
        if ref_keys:
            n_layers = 1 + max(int(k.split(".")[3]) for k in ref_keys)
            cw = sd["model.content_refiner.blocks.0.conv.weight"]  # (dim,dim,kernel)
            kernel = cw.shape[2]
            from livevoice.model.content_refiner import ContentRefiner
            self.refiner = ContentRefiner(in_dim, n_layers, kernel, 0.0)
            self.refiner.load_state_dict(
                {k[len("model.content_refiner."):]: v for k, v in sd.items()
                 if k.startswith("model.content_refiner.")}
            )

        self.proj = nn.Linear(in_dim, out_dim)
        self.proj.load_state_dict({"weight": w, "bias": b})
        self.fsq = None
        levels = infer_content_fsq_from_ckpt(ckpt_path)
        if levels is not None:
            from livevoice.model.fsq import FSQBottleneck
            self.fsq = FSQBottleneck(out_dim, levels)
            fsq_sd = {k[len("model.content_fsq."):]: v
                      for k, v in sd.items() if k.startswith("model.content_fsq.")}
            self.fsq.load_state_dict(fsq_sd)
        # sw2v_to_hidden is the representation the decoder actually consumes AND the exact
        # point the ASR/GRL objectives shape — probe there so the number reflects the
        # trained disentanglement, not just the (frozen-topology) proj+FSQ.
        wh = sd.get("model.sw2v_to_hidden.weight")
        bh = sd.get("model.sw2v_to_hidden.bias")
        self.to_hidden = None
        if wh is not None:
            self.to_hidden = nn.Linear(wh.shape[1], wh.shape[0])
            self.to_hidden.load_state_dict({"weight": wh, "bias": bh})
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        print(f"[probe] trained bottleneck: sw2v_proj {in_dim}->{out_dim}"
              + (f" + FSQ levels={self.fsq.levels}" if self.fsq else " (no FSQ)")
              + (" + sw2v_to_hidden" if self.to_hidden is not None else ""))

    @torch.no_grad()
    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B,T,1024)->(B,T,hidden)
        if self.refiner is not None:
            feats = self.refiner(feats)
        x = self.proj(feats)
        if self.fsq is not None:
            x = self.fsq(x)
        if self.to_hidden is not None:
            x = self.to_hidden(x)
        return x


# ----------------------------------------------------------------------------- feature extraction
@torch.no_grad()
def extract_features(items, config, condition, device, encode_fn, perturb, bottleneck,
                     feats_dir, max_seconds, frames_per_utt, seed,
                     cmn="off", cmn_var=False, cmn_prior_frames=0.0):
    """Return (utt_vecs (N,D), utt_labels (N,), frame_vecs (M,D), frame_labels (M,),
    frame_utt_idx (M,)) for one condition. `encode_fn(audio_BT) -> (B,T,D)` is the
    raw content encoder (sw2v AudioEncoder or HuBERT layer-N hidden)."""
    rng = np.random.RandomState(seed)

    utt_vecs, utt_labels = [], []
    frame_vecs, frame_labels, frame_utt_idx = [], [], []

    for u_idx, (spk_lbl, (spk, wav_path, utt_id)) in enumerate(items):
        if condition == "cache":
            data = torch.load(feats_dir / spk / f"{utt_id}.pt",
                              map_location="cpu", weights_only=True)
            feats = data["feats"].float().unsqueeze(0).to(device)  # (1,T,D)
        else:
            audio = load_audio(wav_path, config.sample_rate, max_seconds).to(device)
            if condition == "perturb":
                audio = perturb(audio.unsqueeze(0)).squeeze(0)  # .train() -> real perturb
            feats = encode_fn(audio.unsqueeze(0))               # (1,T,D)

        # CMN at the frontend, exactly where the model applies it (config.content_cmn),
        # i.e. BEFORE the bottleneck — never at the point we mean-pool below, which would
        # force the pooled vector to zero and make the utterance metric vacuous.
        feats = apply_content_cmn(feats.float(), cmn, cmn_var,
                                  prior_frames=float(cmn_prior_frames))

        if bottleneck is not None:
            feats = bottleneck(feats.float())

        f = feats.squeeze(0).float().cpu()  # (T,D)
        if f.shape[0] == 0:
            continue
        utt_vecs.append(f.mean(dim=0))
        utt_labels.append(spk_lbl)

        T = f.shape[0]
        k = min(frames_per_utt, T)
        sel = rng.choice(T, size=k, replace=False)
        for j in sel:
            frame_vecs.append(f[j])
            frame_labels.append(spk_lbl)
            frame_utt_idx.append(u_idx)

    return (torch.stack(utt_vecs), torch.tensor(utt_labels),
            torch.stack(frame_vecs), torch.tensor(frame_labels),
            torch.tensor(frame_utt_idx))


# ----------------------------------------------------------------------------- linear probe
def train_linear_probe(Xtr, ytr, Xte, yte, num_classes, device, epochs=300, lr=1e-2, wd=1e-4):
    """Standardize on train stats, fit a single Linear (multinomial logistic), return test top-1."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).to(device)
    Xte = ((Xte - mu) / sd).to(device)
    ytr, yte = ytr.to(device), yte.to(device)

    clf = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr_acc = (clf(Xtr).argmax(1) == ytr).float().mean().item()
        te_acc = (clf(Xte).argmax(1) == yte).float().mean().item()
    return tr_acc, te_acc


def utt_split_mask(items, test_frac, seed):
    """Per-speaker held-out utterance mask (True = test), so train/test share speakers
    but never utterances. Same mask reused across all conditions (items order is fixed)."""
    rng = random.Random(seed)
    by_spk: dict[int, list[int]] = {}
    for i, (lbl, _) in enumerate(items):
        by_spk.setdefault(lbl, []).append(i)
    is_test = [False] * len(items)
    for lbl, idxs in by_spk.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_frac)))
        for i in idxs[:n_test]:
            is_test[i] = True
    return torch.tensor(is_test)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zipformer_ckpt",
                    default="/mnt/data/disk2/yejin/LiveVoice/pretrained_models/zipformer_pretrained.pt",
                    help="icefall streaming-Zipformer checkpoint (content_source=zipformer)")
    ap.add_argument("--zipformer_layer", default="-1",
                    help="which tap: -1 = just before the final 50->25Hz downsample "
                         "(deepest, 50 fps, matches jhcodec); 0..5 = that stack's output "
                         "(also 50 fps); 'out' = the model's own 25 fps encoder output")
    ap.add_argument("--content_source", default="sw2v",
                    choices=["sw2v", "hubert", "zipformer"],
                    help="which content encoder to probe (raw sw2v AudioEncoder vs raw HuBERT layer-N)")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--num_speakers", type=int, default=30)
    ap.add_argument("--utts_per_speaker", type=int, default=20)
    ap.add_argument("--conditions", default="clean,perturb,cache")
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--frames_per_utt", type=int, default=40,
                    help="frames sampled per utterance for the frame-level probe")
    ap.add_argument("--max_seconds", type=float, default=8.0,
                    help="cap audio length (clean/perturb) for speed; None-like <=0 disables")
    ap.add_argument("--ckpt", default=None,
                    help="optional: also probe AFTER the trained sw2v_proj[+FSQ] bottleneck")
    ap.add_argument("--cmn", default="off", choices=["off", "utterance", "causal"],
                    help="cepstral mean normalisation on the raw features, applied where "
                         "config.content_cmn applies it (frontend, before the bottleneck). "
                         "Tests whether the utterance-level speaker residual is just a "
                         "per-utterance mean offset — no retraining needed.")
    ap.add_argument("--cmn_var", action="store_true",
                    help="also divide by the std (CMVN rather than CMN)")
    ap.add_argument("--cmn_prior_frames", type=float, default=0.0,
                    help="virtual prior count n0 for --cmn causal; match "
                         "config.content_cmn_prior_frames. n0>0 leaves the first ~n0 frames "
                         "near-raw, so expect the utterance-level probe acc to RISE a little")
    ap.add_argument("--probe_epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    max_seconds = None if args.max_seconds is not None and args.max_seconds <= 0 else args.max_seconds

    config = LiveVoiceConfig(content_source=args.content_source)
    device = torch.device(args.device)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    require_cache = "cache" in conditions
    if args.content_source == "zipformer" and require_cache:
        raise SystemExit("[probe] content_source=zipformer has no feature cache — use "
                         "--conditions clean,perturb")
    feats_dir = feats_base_dir(config, args.content_source)

    print(f"[probe] content_source={args.content_source} split={args.split} "
          f"speakers={args.num_speakers} utts/spk={args.utts_per_speaker} "
          f"conditions={conditions} device={device}")
    print(f"[probe] sample_rate={config.sample_rate} feats_dir={feats_dir}")

    raw_items = discover(
        config, feats_dir, args.split, args.num_speakers, args.utts_per_speaker,
        require_cache, args.seed)
    if not raw_items:
        raise SystemExit("[probe] no items discovered — check libritts_path / cache dir.")
    # map speaker -> contiguous label id
    spk_ids = sorted({spk for spk, _, _ in raw_items})
    spk2lbl = {s: i for i, s in enumerate(spk_ids)}
    num_classes = len(spk_ids)
    items = [(spk2lbl[spk], (spk, wav, utt)) for spk, wav, utt in raw_items]
    print(f"[probe] usable: {len(items)} utterances, {num_classes} speakers "
          f"(chance = {1.0/num_classes:.3f})")

    is_test = utt_split_mask(items, args.test_frac, args.seed)
    tr_idx = (~is_test).nonzero(as_tuple=True)[0]
    te_idx = is_test.nonzero(as_tuple=True)[0]
    print(f"[probe] utterance split: train={len(tr_idx)} test={len(te_idx)}")

    # encoder + raw-feature closure (built once, reused for clean/perturb).
    # For hubert we probe the RAW layer-N hidden (_extract_hidden), NOT the random-init
    # proj/to_hidden — that's the SSL feature comparable to the sw2v encoder output.
    need_encoder = any(c in ("clean", "perturb") for c in conditions)
    encode_fn = None
    if need_encoder:
        if args.content_source == "sw2v":
            enc = Sw2vContentEncoder(config).to(device).eval()
            encode_fn = lambda a: enc(a)  # noqa: E731  (1,T,1024)
        elif args.content_source == "zipformer":
            from livevoice.model.zipformer_content import ZipformerContentEncoder
            lyr = args.zipformer_layer
            lyr = lyr if lyr == "out" else int(lyr)
            enc = ZipformerContentEncoder(config, args.zipformer_ckpt, layer=lyr).to(device)
            encode_fn = lambda a: enc(a)  # noqa: E731  (1,T,D)
        else:
            enc = HuBERTContentExtractor(config).to(device).eval()
            print(f"[probe] HuBERT layer={config.hubert_layer} hidden={config.hubert_hidden_dim} "
                  f"— probing raw _extract_hidden (pre-projection)")
            encode_fn = lambda a: enc._extract_hidden(a)  # noqa: E731  (1,T,768)
    perturb = None
    if "perturb" in conditions:
        perturb = ContentPerturbation(config).to(device)
        perturb.train()  # perturbation is identity in eval() — must be train() to apply
    if args.ckpt and args.content_source != "sw2v":
        print("[probe] WARNING: --ckpt bottleneck probing is sw2v-only; ignoring for hubert.")
        args.ckpt = None
    bottleneck = _Bottleneck(args.ckpt).to(device) if args.ckpt else None
    stages = [("encoder-out", None)] if bottleneck is None else \
             [("encoder-out", None), ("post-bottleneck", bottleneck)]

    rows = []
    for cond in conditions:
        for stage_name, stage_mod in stages:
            uX, uy, fX, fy, futt = extract_features(
                items, config, cond, device, encode_fn, perturb, stage_mod,
                feats_dir, max_seconds, args.frames_per_utt, args.seed,
                cmn=args.cmn, cmn_var=args.cmn_var,
                cmn_prior_frames=args.cmn_prior_frames)

            # utterance-level
            u_tr, u_te = train_linear_probe(
                uX[tr_idx], uy[tr_idx], uX[te_idx], uy[te_idx],
                num_classes, device, epochs=args.probe_epochs)

            # frame-level (split by parent utterance)
            f_is_test = is_test[futt]
            ftr, fte = (~f_is_test).nonzero(as_tuple=True)[0], f_is_test.nonzero(as_tuple=True)[0]
            fr_tr, fr_te = train_linear_probe(
                fX[ftr], fy[ftr], fX[fte], fy[fte],
                num_classes, device, epochs=args.probe_epochs)

            rows.append((cond, stage_name, u_tr, u_te, fr_tr, fr_te))
            print(f"[probe]   {cond:8s} | {stage_name:15s} | "
                  f"utt test={u_te:.3f} (train {u_tr:.3f}) | "
                  f"frame test={fr_te:.3f} (train {fr_tr:.3f})")

    chance = 1.0 / num_classes
    print("\n================ SPEAKER-PROBE RESULTS ================")
    print(f"chance = {chance:.3f}   (higher above chance = more speaker leakage)")
    print(f"{'condition':10s} {'stage':16s} {'utt-test':>9s} {'frame-test':>11s}")
    for cond, stage_name, _, u_te, _, fr_te in rows:
        print(f"{cond:10s} {stage_name:16s} {u_te:9.3f} {fr_te:11.3f}")
    print("======================================================")
    print("Reading: clean high + perturb/cache much lower  -> perturbation removes speaker.")
    print("         perturb/cache still high               -> content path leaks timbre (the bug).")
    print("         frame-test ~ utt-test                  -> a single frame already carries speaker")
    print("                                                   (frame discriminator would judge content, not just timbre).")


if __name__ == "__main__":
    main()
