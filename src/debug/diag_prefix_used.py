"""Does the VC decoder actually USE the speaker prefix, or does it read the target
speaker off the teacher-forced previous codec tokens (prev_emb) and ignore the prefix?

Symptom: giving a null / zero speaker prefix barely changes the metrics, and a richer
(codec) reference didn't help either — i.e. the prefix looks vestigial.

Hypothesis: with same-speaker reconstruction, the AR teacher-forcing input prev_emb
(= previous tokens of the TARGET) already carries the target speaker, so the decoder
predicts the next token from prev_emb and never needs the prefix. This is the AR analogue
of the ASR "LM shortcut". Stage-1 removes speaker from CONTENT but NOT from prev_emb.

Test (analogue of debug/diag_asr_uses_content): load a trained ckpt, take one real batch,
run the training forward under three conditions and compare the codec-token CE loss:
  baseline   : real speaker prefix + real prev_emb
  null_spk   : prefix forced to null_speaker_embedding (the "prefix = 0" case)
  null_prev  : prev_emb forced to null_prev_embedding (kill the AR leak)
Reading:
  null_spk  CE ≈ baseline   → prefix IGNORED (the bug)
  null_prev CE ≫ baseline   → decoder relies on prev_emb (the leak) — the reason prefix is unused

    conda run -n sound python /workspace/LiveVoice/src/debug/diag_prefix_used.py \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/stage2_256_refiner3/step_latest.ckpt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from livevoice.config import LiveVoiceConfig
from livevoice.model import build_codec, Sw2vContentEncoder, HuBERTContentExtractor, LiveVoiceModel
from livevoice.lightning import LiveVoiceLightningModule
from livevoice.lightning.module import _cross_entropy_loss
from livevoice.data.datamodule import LibriTTSDataModule
from livevoice.utils.checkpoint import (
    load_model_weights_from_ckpt,
    infer_content_source_from_ckpt,
    infer_content_fsq_from_ckpt,
    infer_speaker_encoder_from_ckpt,
    infer_speaker_conditioning_from_ckpt,
    read_config_from_ckpt,
    infer_codec_prompt_flags_from_ckpt,
)


def _infer_config(ckpt):
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    kw = {}
    cs = infer_content_source_from_ckpt(ckpt)
    if cs:
        kw["content_source"] = cs
    fsq = infer_content_fsq_from_ckpt(ckpt)
    kw["use_content_fsq"] = fsq is not None
    if fsq is not None:
        kw["fsq_levels"] = fsq
    rk = [k for k in sd if k.startswith("model.content_refiner.blocks.")]
    kw["content_refiner_layers"] = (1 + max(int(k.split(".")[3]) for k in rk)) if rk else 0
    se = infer_speaker_encoder_from_ckpt(ckpt)
    if se:
        kw["speaker_encoder_type"] = se
    sc = infer_speaker_conditioning_from_ckpt(ckpt)
    if sc:
        kw["speaker_conditioning"] = sc
    w = sd.get("model.speaker_prefix_proj.weight")
    if w is not None:
        # out = prefix_len * hidden_dim
        cfg0 = LiveVoiceConfig(**kw)
        kw["speaker_prefix_len"] = int(w.shape[0] // cfg0.hidden_dim)
    # Checkpoints saved after CONFIG_CKPT_KEY was added carry the exact settings — prefer
    # them for the knobs topology inference cannot see. Older ckpts predate continuation,
    # so infer_codec_prompt_flags_from_ckpt turns it off for them.
    stored = read_config_from_ckpt(ckpt)
    if stored:
        if "speaker_prefix_len" in stored:
            kw["speaker_prefix_len"] = int(stored["speaker_prefix_len"])
        # audio_duration sets the reference window, hence the codec_prompt length. Runs
        # differ on it (4.0 vs 3.0), so rebuilding from the current default would evaluate
        # a model at a prompt length it never saw.
        for f in ("audio_duration", "content_source", "content_cmn", "content_cmn_var",
                  "content_cmn_in_cache", "zipformer_layer", "zipformer_ckpt",
                  "zipformer_features_dir", "sw2v_features_dir", "pairing"):
            if f in stored:
                kw[f] = stored[f]
    kw.update(infer_codec_prompt_flags_from_ckpt(ckpt))
    # ASR/GRL heads are irrelevant to the VC forward we test here — disable so the model
    # builds without needing grl_num_speakers (their ckpt weights, if any, load as no-ops).
    kw["use_speaker_grl"] = False
    kw["use_asr_supervision"] = False
    return LiveVoiceConfig(**kw), kw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--speaker_type", default=None,
                    help="override speaker_encoder_type (e.g. codec_prompt) — for a rough "
                         "ZERO-SHOT read of a mode the ckpt wasn't trained with")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config, kw = _infer_config(args.ckpt)
    if args.speaker_type:
        config.speaker_encoder_type = args.speaker_type
        config.speaker_conditioning = "prefix"
        print(f"[diag] OVERRIDE speaker_encoder_type={args.speaker_type} (zero-shot; ckpt not "
              f"trained with it — rough signal only)")
    config.train_batch_size = args.num
    config.max_windows = max(64, args.num * 4)
    config.use_cfg_dropout = False
    print(f"[diag] inferred config: {kw}")
    device = torch.device(args.device)

    # build model exactly like train.py
    codec_model = build_codec(config)
    cs = str(config.content_source).lower()
    if cs == "sw2v":
        content_extractor = Sw2vContentEncoder(config)
    elif cs == "hubert":
        content_extractor = HuBERTContentExtractor(config)
    elif cs == "zipformer":
        from livevoice.model.zipformer_content import ZipformerContentEncoder
        _l = str(config.zipformer_layer)
        content_extractor = ZipformerContentEncoder(
            config, config.zipformer_ckpt, layer=(_l if _l == "out" else int(_l)))
    else:
        content_extractor = None
    model = LiveVoiceModel(config, codec_model, content_extractor, None)
    missing, unexpected = load_model_weights_from_ckpt(model, args.ckpt, verbose=True)
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

    # In continuation mode the prompt occupies the first T_ref positions of the same AR
    # stream, so "kill prev_emb" must spare them — otherwise it would null the prompt too
    # and the PROMPT-ISOLATED comparison would measure nothing.
    use_cont = model._uses_codec_prompt_continuation()
    keep_head = int(model._encode_reference_codes(ref).shape[2]) if use_cont else 0
    if use_cont:
        print(f"[diag] codec_prompt continuation ON — prompt occupies the first "
              f"{keep_head} stream positions; null_prev spares them.")

    @torch.no_grad()
    def ce_under(null_spk=False, null_prev=False, null_ctn=False):
        # null_spk goes through the forward's `null_speaker` argument: in continuation mode
        # the speaker lives in the prompt region of the AR stream, so there is no
        # encode_speaker_reference() call to monkeypatch.
        orig_delay = model._build_delay_input
        orig_nodelay = model._build_nodelay_input
        if null_prev:
            def _np(fn):
                def inner(codes):
                    p = fn(codes)
                    null = model.null_prev_embedding.expand(p.shape[0], p.shape[1] - keep_head, -1).to(p.dtype)
                    if keep_head <= 0:
                        return null
                    return torch.cat([p[:, :keep_head, :], null], dim=1)
                return inner
            model._build_delay_input = _np(orig_delay)
            model._build_nodelay_input = _np(orig_nodelay)
        try:
            out = model(ref, ctn, codes, prosody_audio=None,
                        content_feats=content_feats, reference_z=ref_z,
                        null_speaker=null_spk, null_content=null_ctn)
            ce, _ = _cross_entropy_loss(out["all_logits"], out["delayed_targets"],
                                        config.codebook_loss_weights,
                                        pos_weights=out.get("loss_pos_weights"))
            return ce.item()
        finally:
            model._build_delay_input = orig_delay
            model._build_nodelay_input = orig_nodelay

    base = ce_under()
    nctn = ce_under(null_ctn=True)
    nspk = ce_under(null_spk=True)
    nprev = ce_under(null_prev=True)
    # prompt ISOLATED: with prev_emb killed, does the speaker prompt still lower CE?
    # This is the meaningful test for codec_prompt (teacher-forced CE otherwise hides the
    # prompt because prev_emb leaks the target speaker).
    nprev_prompt = ce_under(null_prev=True, null_spk=False)   # = nprev (real prompt)
    nprev_noprompt = ce_under(null_prev=True, null_spk=True)  # prev AND prompt gone

    print("\n================ PREFIX-USED TEST (codec-token CE) ================")
    print(f"  baseline (real prompt + real prev)     : {base:.4f}")
    print(f"  null speaker prompt (real prev)        : {nspk:.4f}   Δ={nspk-base:+.4f}")
    print(f"  null prev (real prompt)                : {nprev_prompt:.4f}")
    print(f"  null prev + null prompt                : {nprev_noprompt:.4f}")
    print(f"  null CONTENT (real prompt + real prev)  : {nctn:.4f}   Δ={nctn-base:+.4f}")
    print(f"\nprefix usage (teacher-forced) = null_spk - base       = {nspk-base:+.4f}")
    print(f"  (small here even for a GOOD prompt — prev_emb leaks the answer in teacher forcing)")
    print(f"PROMPT-ISOLATED gain          = (nullprev+noprompt) - (nullprev+prompt) = "
          f"{nprev_noprompt - nprev_prompt:+.4f}")
    print(f"  (>0 and large = the prompt DOES carry speaker once the prev leak is removed)")
    print(f"\ncontent usage = null_content - base = {nctn-base:+.4f}")
    print("  ~0 → the decoder IGNORES content and rides the AR stream (prompt + prev);")
    print("       at inference that yields fluent audio saying the WRONG words (high WER).")
    print("  large → content is genuinely driving the prediction.")
    print("===================================================================")


if __name__ == "__main__":
    main()
