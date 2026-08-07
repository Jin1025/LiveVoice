"""Does the ASR-supervision head actually USE the content memory, or does it just
model the phoneme language (teacher-forced prev tokens) and ignore content?

Symptom that triggered this: fsq_1000_asr got WER≈0.9–1.0 (broken) yet asr_loss fell
to ~2.4. A seq2seq ASR decoder predicts phoneme[i+1] from phonemes[0..i]; English
phoneme sequences are very predictable, so the decoder can drive CE down with a pure
LM WITHOUT reading content — providing ZERO pressure on the content representation.
CE≈2.4 ≈ ln(11) is right in the phoneme-LM ballpark, which is suspicious.

Decisive test: load a trained checkpoint's ASR path and compare asr_loss with
  (a) REAL memory   (content matches the phonemes)
  (b) ZERO memory   (no content at all)
  (c) SHUFFLED memory (content from a DIFFERENT utterance)
If (a) ≈ (b) ≈ (c), the head ignores content → the ASR supervision is doing nothing.
A healthy content-using head has (a) ≪ (b),(c).

    conda run -n sound python /workspace/LiveVoice/src/debug/diag_asr_uses_content.py \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/fsq_1000_asr/last.ckpt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from livevoice.config import LiveVoiceConfig  # noqa: E402
from livevoice.model.phoneme_vocab import PAD_ID, PHONEME_VOCAB  # noqa: E402


def load_batch(config, n: int, seed: int, feats_base: str, phon_base: str):
    """Load n utterances that have BOTH full sw2v feats and a phoneme cache."""
    feats_dir = Path(feats_base) / "libritts"
    phon_dir = Path(phon_base) / "libritts"
    rng = random.Random(seed)
    spk_dirs = [d for d in sorted(feats_dir.iterdir()) if d.is_dir()]
    rng.shuffle(spk_dirs)
    feats, phons = [], []
    for spk in spk_dirs:
        for fp in spk.glob("*.pt"):
            pp = phon_dir / spk.name / fp.name
            if pp.exists():
                fe = torch.load(fp, weights_only=True)["feats"].float()
                ph = torch.load(pp, weights_only=True).long()
                feats.append(fe); phons.append(ph)
                break
        if len(feats) >= n:
            break
    T = max(f.shape[0] for f in feats)
    L = max(p.shape[0] for p in phons)
    B = len(feats)
    feats_pad = torch.zeros(B, T, feats[0].shape[1])
    flen = torch.zeros(B, dtype=torch.long)
    phon_pad = torch.full((B, L), PAD_ID, dtype=torch.long)
    for i, (fe, ph) in enumerate(zip(feats, phons)):
        feats_pad[i, : fe.shape[0]] = fe
        flen[i] = fe.shape[0]
        phon_pad[i, : ph.shape[0]] = ph
    return feats_pad, flen, phon_pad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feats_dir", default="/mnt/data/disk2/yejin/LiveVoice/features/perturbed/sw2v")
    ap.add_argument("--phon_dir", default="/mnt/data/disk2/yejin/LiveVoice/features/phonemes")
    args = ap.parse_args()

    config = LiveVoiceConfig()
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)

    # --- rebuild the ASR path straight from checkpoint weights ---
    w = sd["model.sw2v_proj.weight"]; cpd, in_dim = w.shape
    proj = nn.Linear(in_dim, cpd); proj.load_state_dict(
        {"weight": w, "bias": sd["model.sw2v_proj.bias"]})
    to_hidden_w = sd["model.sw2v_to_hidden.weight"]; hid = to_hidden_w.shape[0]
    to_hidden = nn.Linear(cpd, hid); to_hidden.load_state_dict(
        {"weight": to_hidden_w, "bias": sd["model.sw2v_to_hidden.bias"]})

    fsq = None
    if "model.content_fsq._levels" in sd:
        from livevoice.model.fsq import FSQBottleneck
        levels = tuple(int(x) for x in sd["model.content_fsq._levels"].tolist())
        fsq = FSQBottleneck(cpd, levels)
        fsq.load_state_dict({k[len("model.content_fsq."):]: v
                             for k, v in sd.items() if k.startswith("model.content_fsq.")})
        print(f"[diag] FSQ present: levels={levels} codebook={fsq.codebook_size}")

    refiner = None
    if any(k.startswith("model.content_refiner.") for k in sd):
        from livevoice.model.content_refiner import ContentRefiner
        ref_keys = [k for k in sd if k.startswith("model.content_refiner.blocks.")]
        n_layers = 1 + max(int(k.split(".")[3]) for k in ref_keys)
        kernel = sd["model.content_refiner.blocks.0.conv.weight"].shape[2]
        refiner = ContentRefiner(in_dim, n_layers, kernel, 0.0)
        refiner.load_state_dict({k[len("model.content_refiner."):]: v
                                 for k, v in sd.items() if k.startswith("model.content_refiner.")})

    has_asr = any(k.startswith("model.asr_head.") for k in sd)
    if not has_asr:
        raise SystemExit("[diag] checkpoint has no model.asr_head.* — was ASR supervision on?")
    is_ctc = not any(k.startswith("model.asr_head.decoder.layers.") for k in sd)
    ahcfg = SimpleNamespace(hidden_dim=hid, dropout=0.0,
                            num_heads=int(config.num_heads),
                            asr_max_phoneme_len=int(config.asr_max_phoneme_len))
    if is_ctc:
        from livevoice.model.ctc_supervision import CtcSupervisionHead
        asr = CtcSupervisionHead(ahcfg)
    else:
        n_dec = 1 + max(int(k.split(".")[4]) for k in sd
                        if k.startswith("model.asr_head.decoder.layers."))
        ahcfg.asr_decoder_layers = n_dec
        from livevoice.model.asr_supervision import AsrSupervisionHead
        asr = AsrSupervisionHead(ahcfg)
    asr.load_state_dict({k[len("model.asr_head."):]: v
                         for k, v in sd.items() if k.startswith("model.asr_head.")})
    for m in (proj, to_hidden, asr, fsq, refiner):
        if m is not None:
            m.eval()
    print(f"[diag] rebuilt ASR path: proj {in_dim}->{cpd}, to_hidden->{hid}, "
          f"head={'CTC' if is_ctc else 'seq2seq'}, refiner={'yes' if refiner else 'no'}")

    feats, flen, phon = load_batch(config, args.num, args.seed, args.feats_dir, args.phon_dir)
    print(f"[diag] batch: {feats.shape[0]} utts, T={feats.shape[1]}, phon_len={phon.shape[1]}")

    @torch.no_grad()
    def memory_from(f):
        x = refiner(f) if refiner is not None else f
        x = proj(x)
        if fsq is not None:
            x = fsq(x)
        return to_hidden(x)

    @torch.no_grad()
    def asr_ce(mem):
        if is_ctc:
            return asr.compute_loss(mem, flen, phon).item()
        T = mem.size(1)
        mkpm = torch.arange(T).unsqueeze(0) >= flen.unsqueeze(1)
        return asr.compute_loss(mem, mkpm, phon).item()

    mem = memory_from(feats)
    ce_real = asr_ce(mem)
    ce_zero = asr_ce(torch.zeros_like(mem))
    ce_shuf = asr_ce(mem[torch.roll(torch.arange(mem.size(0)), 1)])  # content from wrong utt

    ln_vocab = torch.log(torch.tensor(float(len(PHONEME_VOCAB)))).item()
    print("\n================ ASR-USES-CONTENT TEST ================")
    print(f"asr_loss  REAL memory     : {ce_real:.3f}   (should be LOWEST)")
    print(f"asr_loss  ZERO memory     : {ce_zero:.3f}")
    print(f"asr_loss  SHUFFLED memory : {ce_shuf:.3f}   (wrong utterance's content)")
    print(f"reference: ln(vocab=42)={ln_vocab:.3f} (uniform)  |  phoneme-LM CE ~2–3")
    gap = min(ce_zero, ce_shuf) - ce_real
    print(f"\ncontent-usage gap = min(zero,shuf) - real = {gap:.3f}")
    if gap < 0.3:
        print("VERDICT: head is IGNORING content (LM shortcut) → ASR supervision is doing")
        print("         almost nothing to the representation. This is the bug to fix.")
    elif gap < 1.0:
        print("VERDICT: head uses content only WEAKLY — LM shortcut is dominating.")
    else:
        print("VERDICT: head genuinely uses content (real ≪ zero/shuf). ASR loss is real.")
    print("=======================================================")


if __name__ == "__main__":
    main()
