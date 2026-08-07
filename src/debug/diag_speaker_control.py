"""Who controls the speaker of the output — the REFERENCE or the CONTENT?

Motivating observation (by ear): with the SAME reference but DIFFERENT content, the
generated voice sounds completely different. If the reference controlled identity that
could not happen.

This tests it in a way that is immune to the absolute calibration of the speaker metric
(duration bias, floor/ceiling, protocol asymmetry all cancel), because it only compares
GENERATED samples with each other:

    generate a full R x C grid   (R references x C content utterances)

    A = mean pairwise cos within a ROW    (same ref, different content)
    B = mean pairwise cos within a COLUMN (same content, different ref)

    A >> B  → the REFERENCE decides the voice   (what we want)
    B >> A  → the CONTENT decides the voice     (the failure mode)
    A ~= B  → neither dominates

Reference speakers and content speakers are disjoint, so every cell is a real
cross-speaker conversion.

    conda run -n sound python /workspace/LiveVoice/src/debug/diag_speaker_control.py \
        --ckpt /mnt/data/disk2/yejin/LiveVoice/checkpoints/stage2_perspk_codec_re/step_latest.ckpt
"""
from __future__ import annotations

import argparse
import os
import copy
import random
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from livevoice.model import build_codec, Sw2vContentEncoder, HuBERTContentExtractor, LiveVoiceModel
from livevoice.utils.checkpoint import load_model_weights_from_ckpt

from diag_prefix_used import _infer_config


def _mean(xs):
    return sum(xs) / max(1, len(xs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--libritts", default="/mnt/data/disk2/LibriTTS/dev-clean")
    ap.add_argument("--n_refs", type=int, default=4)
    ap.add_argument("--n_contents", type=int, default=6)
    ap.add_argument("--min_sec", type=float, default=4.0)
    ap.add_argument("--max_sec", type=float, default=10.0)
    ap.add_argument("--cfg_scale", type=float, default=1.0,
                    help=">1 amplifies the speaker prompt against the null-prompt branch; "
                         "tests whether the prompt is used at all at inference time")
    ap.add_argument("--cfg_scales", default=None,
                    help="comma-separated sweep, e.g. 1.0,1.5,2.0,3.0,5.0. Also measures WER "
                         "at each point, since CFG usually trades intelligibility for "
                         "conditioning strength.")
    ap.add_argument("--whisper_model", default="medium")
    ap.add_argument("--perturb_content", default="off",
                    choices=["off", "full", "pitch_eq", "eq_only"],
                    help="apply speaker-de-identifying perturbation to the SOURCE audio at "
                         "INFERENCE, matching what the frozen Stage-1 tokenizer was trained "
                         "on. The probe shows clean content carries ~2.4x more speaker than "
                         "the perturbed cache (encoder-out 0.711 vs 0.294), and the bottleneck "
                         "only removes ~25% either way — so the clean input may be what pins "
                         "gen->src. Components are split because they cost very different "
                         "latency in a streaming system: eq_only is IIR filtering (~free), "
                         "pitch/formant shift need analysis windows (expensive).")
    ap.add_argument("--speaker_type", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dump_dir", default=None, help="optional: write the grid as wavs to listen")
    args = ap.parse_args()

    import soundfile as sf
    import librosa

    sr = 16000
    config, kw = _infer_config(args.ckpt)
    if args.speaker_type:
        config.speaker_encoder_type = args.speaker_type
        config.speaker_conditioning = "prefix"
    config.features_dir = None
    config.sw2v_features_dir = None
    print(f"[ctrl] inferred config: {kw}")
    print(f"[ctrl] speaker_encoder_type={config.speaker_encoder_type} "
          f"continuation={getattr(config, 'codec_prompt_continuation', None)}")
    device = torch.device(args.device)

    codec_model = build_codec(config)
    cs = str(config.content_source).lower()
    content_extractor = (
        Sw2vContentEncoder(config) if cs == "sw2v"
        else HuBERTContentExtractor(config) if cs == "hubert" else None
    )
    model = LiveVoiceModel(config, codec_model, content_extractor, None)
    missing, unexpected = load_model_weights_from_ckpt(model, args.ckpt, verbose=False)
    print(f"[ctrl] loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    model = model.to(device).eval()

    from livevoice.model.wavlm_speaker_encoder import WavLMTDNNSpeakerEncoder
    spk_enc = WavLMTDNNSpeakerEncoder(config).eval().to(device)

    # ── pick utterances: reference speakers and content speakers are DISJOINT ──
    by_spk = defaultdict(list)
    for root, _d, files in os.walk(args.libritts):
        for fn in files:
            if not fn.endswith(".wav"):
                continue
            p = os.path.join(root, fn)
            try:
                info = sf.info(p)
                dur = info.frames / info.samplerate
            except Exception:
                continue
            if args.min_sec <= dur <= args.max_sec:
                by_spk[p[len(args.libritts):].strip("/").split("/")[0]].append(p)

    spks = sorted(k for k, v in by_spk.items() if v)
    rng = random.Random(args.seed)
    rng.shuffle(spks)
    need = args.n_refs + args.n_contents
    if len(spks) < need:
        raise SystemExit(f"[ctrl] need {need} speakers, found {len(spks)}")
    ref_spks, ctn_spks = spks[: args.n_refs], spks[args.n_refs : need]
    ref_paths = [rng.choice(by_spk[s]) for s in ref_spks]
    ctn_paths = [rng.choice(by_spk[s]) for s in ctn_spks]
    print(f"[ctrl] {len(ref_paths)} refs x {len(ctn_paths)} contents = "
          f"{len(ref_paths) * len(ctn_paths)} generations")

    def load(p):
        y, s = sf.read(p, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        if s != sr:
            y = librosa.resample(y, orig_sr=s, target_sr=sr)
        return torch.from_numpy(y).unsqueeze(0).to(device)

    ref_max = int(float(getattr(config, "audio_duration", 4.0)) * sr)

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    scales = ([float(x) for x in args.cfg_scales.split(",")]
              if args.cfg_scales else [float(args.cfg_scale)])
    want_wer = args.cfg_scales is not None

    whisper_model = None
    ctn_texts = []
    if want_wer:
        import whisper
        whisper_model = whisper.load_model(args.whisper_model, device=str(device))
        for cp in ctn_paths:
            t = cp[: -len(".wav")] + ".normalized.txt"
            ctn_texts.append(open(t).read().strip() if os.path.exists(t) else None)
        n_txt = sum(1 for t in ctn_texts if t)
        print(f"[ctrl] whisper={args.whisper_model}, transcripts for {n_txt}/{len(ctn_paths)} contents")

    def _wer(ref_txt, hyp_txt):
        import re
        norm = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()
        a, b = norm(ref_txt), norm(hyp_txt)
        d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a) + 1):
            d[i][0] = i
        for j in range(len(b) + 1):
            d[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                              d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
        return d[len(a)][len(b)] / max(1, len(a))

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=-1).mean().item()

    # reference / source embeddings are cfg-independent — compute once
    ref_emb, ctn_emb = [], []
    with torch.no_grad():
        for rp in ref_paths:
            ref_emb.append(spk_enc(load(rp)[..., :ref_max].float()))
        for cp in ctn_paths:
            ctn_emb.append(spk_enc(load(cp).float()))

    perturb = None
    if args.perturb_content != "off":
        from livevoice.model.content_perturbation import ContentPerturbation
        pcfg = copy.deepcopy(config)
        if args.perturb_content == "full":
            # Force the formant warp ON: the perturbed feature cache the frozen Stage-1
            # tokenizer was trained on includes it, but config.use_vtln has since been
            # turned off, so "full" must not silently drop it.
            pcfg.use_vtln = True
        else:
            pcfg.use_vtln = False                    # no formant warp
        if args.perturb_content == "eq_only":
            pcfg.perturb_pitch_semitones = 0.0       # EQ only — the cheap-latency case
        perturb = ContentPerturbation(pcfg)
        perturb.train()   # forward() is identity in eval mode
        print(f"[ctrl] SOURCE perturbation at inference: mode={args.perturb_content} "
              f"(pitch=±{pcfg.perturb_pitch_semitones}, vtln={pcfg.use_vtln}, "
              f"eq=±{pcfg.perturb_eq_gain_db}dB)")

    def run_grid(cfg_scale):
        gen_emb = [[None] * len(ctn_paths) for _ in range(len(ref_paths))]
        wers = []
        with torch.no_grad():
            for r, rp in enumerate(ref_paths):
                ref = load(rp)[..., :ref_max]
                for c, cp in enumerate(ctn_paths):
                    ctn = load(cp)
                    if perturb is not None:
                        # Seed per (content, cfg) so every condition sees the SAME random
                        # perturbation — otherwise cfg points would differ by draw, not by cfg.
                        random.seed(args.seed * 1000 + c)
                        ctn = perturb(ctn)
                    codes = model.generate(reference_audio=ref, content_audio=ctn,
                                           temperature=0.0, top_p=None, top_k=None,
                                           cfg_scale=float(cfg_scale))
                    gen = model.decode_to_audio(codes)
                    gen_emb[r][c] = spk_enc(gen.float())
                    if whisper_model is not None and ctn_texts[c]:
                        hyp = whisper_model.transcribe(
                            gen[0].detach().float().cpu().numpy(), fp16=False)["text"]
                        wers.append(_wer(ctn_texts[c], hyp))
                    if args.dump_dir:
                        tag = f"cfg{cfg_scale:g}_" if len(scales) > 1 else ""
                        sf.write(os.path.join(args.dump_dir, f"{tag}ref{r}_ctn{c}.wav"),
                                 gen[0].detach().cpu().numpy(), sr)
                print(f"  [cfg={cfg_scale:g}] ref {r + 1}/{len(ref_paths)} done")
        return gen_emb, wers

    def metrics(gen_emb):
        rows = [_mean([cos(gen_emb[r][i], gen_emb[r][j])
                       for i, j in combinations(range(len(ctn_paths)), 2)])
                for r in range(len(ref_paths))]
        cols = [_mean([cos(gen_emb[i][c], gen_emb[j][c])
                       for i, j in combinations(range(len(ref_paths)), 2)])
                for c in range(len(ctn_paths))]
        to_ref = _mean([cos(gen_emb[r][c], ref_emb[r])
                        for r in range(len(ref_paths)) for c in range(len(ctn_paths))])
        to_src = _mean([cos(gen_emb[r][c], ctn_emb[c])
                        for r in range(len(ref_paths)) for c in range(len(ctn_paths))])
        return rows, cols, to_ref, to_src

    if len(scales) > 1:
        print(f"\n[ctrl] sweeping cfg_scale over {scales}")
        table = []
        for s in scales:
            ge, ws = run_grid(s)
            rows, cols, to_ref, to_src = metrics(ge)
            table.append((s, _mean(rows), _mean(cols), to_ref, to_src,
                          _mean(ws) if ws else float("nan")))
        print("\n=============== CFG SWEEP: conditioning strength vs intelligibility ===============")
        print(f"{'cfg':>5} | {'A same-ref':>10} | {'B same-ctn':>10} | {'A-B':>8} | "
              f"{'gen→ref':>8} | {'gen→src':>8} | {'WER':>7}")
        print("-" * 82)
        for s, a, b, tr, ts, wr in table:
            print(f"{s:>5.2f} | {a:>10.4f} | {b:>10.4f} | {a - b:>+8.4f} | "
                  f"{tr:>8.4f} | {ts:>8.4f} | {wr:>7.4f}")
        print("-" * 82)
        base_wer = table[0][5]
        print("A-B > 0 means the REFERENCE controls the voice. Read it against the WER column:")
        print(f"WER at cfg={table[0][0]:g} is {base_wer:.4f}; pick the largest A-B whose WER")
        print("you are willing to pay for. Speaker embeddings: diff-spk floor ~0.14,")
        print("same-spk ceiling ~0.67 (WavLM-TDNN, LibriTTS).")
        print("==================================================================================")
        return

    gen_emb, _ = run_grid(scales[0])
    rows, cols, to_ref, to_src = metrics(gen_emb)
    A, Bv = _mean(rows), _mean(cols)
    print("\n============== WHO CONTROLS THE OUTPUT SPEAKER? ==============")
    print(f"A  same REF, different content   (want HIGH) : {A:.4f}")
    print(f"     per-ref: {[round(x, 3) for x in rows]}")
    print(f"B  same CONTENT, different ref   (want LOW)  : {Bv:.4f}")
    print(f"     per-content: {[round(x, 3) for x in cols]}")
    print("-" * 62)
    print(f"   mean cos(gen, its reference)              : {to_ref:.4f}")
    print(f"   mean cos(gen, its source)                 : {to_src:.4f}")
    print("-" * 62)
    print(f"CONTROL MARGIN  A - B = {A - Bv:+.4f}")
    if A - Bv > 0.05:
        print("  >>> the REFERENCE decides the voice — speaker conditioning WORKS.")
    elif Bv - A > 0.05:
        print("  >>> the CONTENT decides the voice — the prompt is still being bypassed.")
        print("      (Matches 'same ref + different content sounds like a different speaker'.)")
    else:
        print("  >>> neither dominates; the output speaker is largely independent of both.")
    print("==============================================================")
    print("\nNote: A and B are both similarities BETWEEN GENERATED samples, so the")
    print("duration/protocol bias that makes the absolute val numbers hard to read")
    print("cancels out here. Only the comparison A vs B matters.")


if __name__ == "__main__":
    main()
