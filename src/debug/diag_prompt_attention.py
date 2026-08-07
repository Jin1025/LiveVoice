"""Do the target positions actually ATTEND to the speaker prompt, or does ALiBi bury it?

Context: with speaker_conditioning="prefix", nn/layer.py sets

    self.alibi_prefix_len = config.speaker_prefix_len        # a CONSTANT (4)
    alibi_attn_bias[..., :alibi_prefix_len] = 0.0            # only those columns are exempt

but for speaker_encoder_type="codec_prompt" (and "codec") the REAL prefix is the whole
reference, i.e. audio_duration*50 frames (200 for 4s) — `_build_speaker_prefix` passes the
speaker token sequence through unchanged. So prompt frames [speaker_prefix_len : P) keep the
FULL ALiBi distance penalty and, being 100-400 positions away from the target, get an
exponentially small attention weight. Predicted effect: only the first `speaker_prefix_len`
prompt frames (= 80 ms of reference audio at P=4) are reachable.

This measures it directly on a trained ckpt instead of trusting the bias arithmetic: it
captures the decoder self-attention softmax and reports how much attention mass the TARGET
queries (positions >= P) put on the PROMPT keys (positions < P), split into

    zero-penalty prompt   cols [0, speaker_prefix_len)   ← ALiBi-exempt
    penalised prompt      cols [speaker_prefix_len, P)   ← the other ~98% of the reference

Reading:
  penalised-prompt mass ~ 0 and total prompt mass << uniform
        → CONFIRMED: ALiBi is masking the prompt; the codec_prompt experiment never ran.
  penalised-prompt mass comparable to uniform
        → the learned Q/K compensated for the bias; ALiBi is NOT the binding constraint,
          and the bottleneck is elsewhere (content redundancy).

    conda run -n sound python /workspace/LiveVoice/src/debug/diag_prompt_attention.py \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/stage2_perspk_codec/step_latest.ckpt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch

from livevoice.config import LiveVoiceConfig
from livevoice.model import build_codec, Sw2vContentEncoder, HuBERTContentExtractor, LiveVoiceModel
from livevoice.nn.layer import MultiHeadAttention
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.data.datamodule import LibriTTSDataModule
from livevoice.utils.checkpoint import load_model_weights_from_ckpt

from diag_prefix_used import _infer_config


def _decoder_self_attns(model) -> list[MultiHeadAttention]:
    out = []
    for m in model.modules():
        if isinstance(m, MultiHeadAttention) and (not m.cross_attn) and m.purpose == "decoder":
            out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=4, help="batch size (attn maps are big)")
    ap.add_argument("--speaker_type", default=None,
                    help="force speaker_encoder_type (codec_prompt has no distinctive "
                         "params, so it is NOT inferable from the ckpt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config, kw = _infer_config(args.ckpt)
    if args.speaker_type:
        config.speaker_encoder_type = args.speaker_type
        config.speaker_conditioning = "prefix"
    config.train_batch_size = args.num
    config.max_windows = max(64, args.num * 4)
    config.use_cfg_dropout = False

    print(f"[diag] inferred config: {kw}")
    print(f"[diag] speaker_encoder_type = {config.speaker_encoder_type!r}  "
          f"speaker_conditioning = {config.speaker_conditioning!r}")
    print(f"[diag] config.speaker_prefix_len = {config.speaker_prefix_len}  "
          f"(this is what ALiBi exempts)")
    if config.speaker_encoder_type == "codec_prompt" and not args.speaker_type:
        print("[diag] NOTE: 'codec_prompt' builds no speaker_proj/speaker_prefix_proj, so "
              "infer_speaker_encoder_from_ckpt() cannot detect it — this came from the "
              "config default. Pass --speaker_type to be explicit.")

    device = torch.device(args.device)

    codec_model = build_codec(config)
    cs = str(config.content_source).lower()
    if cs == "sw2v":
        content_extractor = Sw2vContentEncoder(config)
    elif cs == "hubert":
        content_extractor = HuBERTContentExtractor(config)
    else:
        content_extractor = None
    model = LiveVoiceModel(config, codec_model, content_extractor, None)
    missing, unexpected = load_model_weights_from_ckpt(model, args.ckpt, verbose=False)
    print(f"[diag] loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")
    model = model.to(device).eval()
    lit = LiveVoiceLightningModule(config, model).to(device).eval()

    dm = LibriTTSDataModule(config)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    ref = batch["reference_audio"].to(device)
    ctn = batch["content_audio"].to(device)
    tgt = batch["target_audio"].to(device)
    content_feats = batch.get("content_hubert", None)
    content_feats = content_feats.to(device) if content_feats is not None else None
    codes = lit._load_target_codes_or_fallback(
        tgt, batch.get("content_path"), batch.get("content_start_sample"))
    ref_z = lit._load_reference_z_or_fallback(
        ref, batch.get("ref_path"), batch.get("ref_start_sample"))

    # ---- record the ACTUAL prefix length the model builds ----
    seen = {}
    orig_prepend = model._prepend_speaker_prefix

    def _spy(spk, prev_emb, content_add, film_feats):
        r = orig_prepend(spk, prev_emb, content_add, film_feats)
        seen["P"] = int(r[0].shape[1])
        return r

    model._prepend_speaker_prefix = _spy

    attns = _decoder_self_attns(model)
    for m in attns:
        m._capture_attn_weights = True

    with torch.no_grad():
        model(ref, ctn, codes, prosody_audio=None,
              content_feats=content_feats, reference_z=ref_z)

    model._prepend_speaker_prefix = orig_prepend

    P = seen.get("P")
    if P is None:
        raise SystemExit("[diag] no speaker prefix was built — is speaker_conditioning='prefix'?")
    Pz = int(config.speaker_prefix_len)
    print(f"\n[diag] ACTUAL prefix length built by the model : P = {P} frames "
          f"({P / 50.0:.2f}s of reference @50Hz)")
    print(f"[diag] ALiBi zero-penalty columns              : {Pz}  "
          f"({Pz / 50.0 * 1000:.0f} ms)")
    if P != Pz:
        print(f"[diag] >>> MISMATCH: {P - Pz} prompt frames ({100.0 * (P - Pz) / P:.1f}%) "
              f"carry the full ALiBi distance penalty.")

    # ---- aggregate attention mass ----
    rows = []
    tot_zero = tot_pen = tot_unif = 0.0
    n_layers = 0
    for li, m in enumerate(attns):
        w = getattr(m, "_last_attn_weights", None)
        if w is None:
            continue
        n_layers += 1
        w = w.float()                       # (B, H, Lq, Lk)
        B, H, Lq, Lk = w.shape
        if Lq <= P:
            continue
        q = w[:, :, P:, :]                  # target-region queries only
        zero_mass = q[..., :Pz].sum(-1).mean(dim=(0, 2))          # (H,)
        pen_mass = q[..., Pz:P].sum(-1).mean(dim=(0, 2))          # (H,)
        # uniform reference: a query at absolute position p sees p+1 keys
        pos = torch.arange(P, Lq, device=w.device, dtype=torch.float32) + 1.0
        unif = (P / pos).mean().item()
        rows.append((li, zero_mass.cpu(), pen_mass.cpu(), unif))
        tot_zero += zero_mass.sum().item()
        tot_pen += pen_mass.sum().item()
        tot_unif += unif * H
        del m._last_attn_weights, w, q
        m._capture_attn_weights = False

    H = rows[0][1].numel()
    print(f"\n================ PROMPT ATTENTION MASS ({n_layers} decoder layers, {H} heads) ================")
    print("attention mass that TARGET queries place on PROMPT keys, averaged over batch & queries")
    print(f"{'layer':>5} | {'zero-penalty [0,%d)' % Pz:>19} | {'penalised [%d,%d)' % (Pz, P):>19} | "
          f"{'total':>8} | {'uniform':>8}")
    print("-" * 78)
    for li, zm, pm, unif in rows:
        print(f"{li:>5} | {zm.sum().item():>19.5f} | {pm.sum().item():>19.5f} | "
              f"{(zm.sum() + pm.sum()).item():>8.5f} | {unif * H:>8.5f}")

    print("\nper-head (summed over layers):")
    zs = torch.stack([r[1] for r in rows]).sum(0)
    ps = torch.stack([r[2] for r in rows]).sum(0)
    print(f"{'head':>4} | {'zero-penalty':>13} | {'penalised':>13}")
    for h in range(H):
        print(f"{h:>4} | {zs[h].item():>13.5f} | {ps[h].item():>13.5f}")

    n_pen_frames = P - Pz
    print("\n---------------------------------- VERDICT ----------------------------------")
    print(f"  prompt mass (zero-penalty {Pz} frames) : {tot_zero:.5f}")
    print(f"  prompt mass (penalised {n_pen_frames} frames)  : {tot_pen:.5f}")
    print(f"  uniform-attention reference           : {tot_unif:.5f}")
    if tot_zero + tot_pen > 0:
        share = 100.0 * tot_zero / (tot_zero + tot_pen)
        print(f"  → {share:.1f}% of all prompt attention sits in the {Pz} zero-penalty frames, "
              f"which are {100.0 * Pz / P:.1f}% of the prompt.")
    print(f"  → prompt gets {100.0 * (tot_zero + tot_pen) / max(tot_unif, 1e-9):.1f}% of the "
          f"mass a uniform attention would give it.")
    print("  CONFIRMED if the penalised frames are ~0 and the total is far below uniform.")
    print("-----------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
