"""Measure how much the streaming Zipformer's content stream degrades at smaller chunks.

The content encoder is exported as chunk_16_left_128, i.e. 16 frames @ 50 fps = 320 ms of
input buffering before ANY frame of that chunk can be emitted.  That 320 ms dominates the
end-to-end streaming latency budget, so the question is whether chunk 8 (160 ms) -- the
structural floor, since Zipformer2 asserts chunk_size % downsampling_factor == 0 and the
U-Net schedule contains an 8x stack -- costs anything.

Two metrics, both referenced against chunk 16 (what the AR decoder was trained on):

  feature level : cosine similarity / relative L2 of the 50 fps tap (layer -1, 512-d) that
                  LiveVoice actually consumes as BNF.
  ASR level     : the checkpoint still carries the full pruned-RNN-T decoder + joiner, so
                  greedy transducer decoding runs without a BPE tokenizer (the hypothesis
                  is a BPE id sequence).  Edit distance against the chunk-16 hypothesis is
                  a token error rate that answers "did the linguistic content change".
                  simple_am_proj alone is NOT usable here -- it is the simplified-loss AM
                  term, not a calibrated per-frame posterior, and argmaxes to blank.

chunk -1 (full attention) is included as the non-streaming ceiling.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import torch
import torchaudio


class Transducer(torch.nn.Module):
    """icefall pruned-RNN-T decoder + joiner, rebuilt from the checkpoint tensors."""

    def __init__(self, sd: dict, vocab: int = 500, dim: int = 512, context: int = 2):
        super().__init__()
        self.context = context
        self.blank = 0
        self.embedding = torch.nn.Embedding(vocab, dim)
        # decoder.conv.weight is (512, 4, 2): out=512, in/groups=4 -> groups=128, no bias
        self.conv = torch.nn.Conv1d(dim, dim, kernel_size=context, groups=dim // 4, bias=False)
        self.enc_proj = torch.nn.Linear(dim, dim)
        self.dec_proj = torch.nn.Linear(dim, dim)
        self.output = torch.nn.Linear(dim, vocab)
        self.embedding.weight.data.copy_(sd["decoder.embedding.weight"])
        self.conv.weight.data.copy_(sd["decoder.conv.weight"])
        for dst, src in ((self.enc_proj, "joiner.encoder_proj"), (self.dec_proj, "joiner.decoder_proj"),
                         (self.output, "joiner.output_linear")):
            dst.weight.data.copy_(sd[f"{src}.weight"])
            dst.bias.data.copy_(sd[f"{src}.bias"])
        self.eval()

    def decode_label(self, hyp: list[int]) -> torch.Tensor:
        y = torch.tensor([hyp[-self.context:]], device=self.embedding.weight.device)
        e = self.embedding(y).permute(0, 2, 1)          # (1, D, context)
        e = self.conv(e).permute(0, 2, 1)               # (1, 1, D)
        return self.dec_proj(torch.relu(e))

    @torch.no_grad()
    def greedy(self, encoder_out: torch.Tensor, max_sym_per_frame: int = 3) -> list[int]:
        """encoder_out (1, T, D) -> BPE id hypothesis."""
        enc = self.enc_proj(encoder_out)
        hyp = [self.blank] * self.context
        dec = self.decode_label(hyp)
        for t in range(enc.shape[1]):
            frame = enc[:, t : t + 1, :]
            for _ in range(max_sym_per_frame):
                y = int(self.output(torch.tanh(frame + dec)).argmax(-1).item())
                if y == self.blank:
                    break
                hyp.append(y)
                dec = self.decode_label(hyp)
        return hyp[self.context:]


# LibriTTS .normalized.txt keeps abbreviations that the LibriSpeech BPE model spells out.
# Anything left unmapped inflates the ABSOLUTE WER equally for every chunk setting, so the
# DELTA between settings -- the number this benchmark exists to produce -- stays clean.
_ABBREV = {"MR": "MISTER", "MRS": "MISSUS", "DR": "DOCTOR", "ST": "SAINT"}
_PUNCT = re.compile(r"[^A-Z' ]+")


def normalize(text: str) -> list[int] | list[str]:
    text = _PUNCT.sub(" ", text.upper().replace("-", " "))
    return [_ABBREV.get(w, w) for w in (x.strip("'") for x in text.split()) if w]


def edit_distance(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libritts", default="/mnt/data/disk2/LibriTTS")
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--ckpt", default="/mnt/data/disk2/yejin/LiveVoice/pretrained_models/zipformer_pretrained.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--chunks", default="8,16,32,-1")
    ap.add_argument("--align-pad", type=int, default=-6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bpe", default="", help="sentencepiece model -> also report real WER")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks = [int(c) for c in args.chunks.split(",")]
    ref = 16
    assert ref in chunks, "chunk 16 is the reference the AR decoder was trained on"

    wavs = sorted(Path(args.libritts, args.split).rglob("*.wav"))
    random.Random(args.seed).shuffle(wavs)
    wavs = [w for w in wavs[: args.n * 3]][: args.n]
    print(f"[bench] {len(wavs)} utterances from {args.split}")

    from livevoice.model.zipformer_content import ZipformerContentEncoder

    class Cfg:
        zipformer_ckpt = args.ckpt
        zipformer_align_pad_frames = args.align_pad

    dev = torch.device(args.device)
    tap = ZipformerContentEncoder(Cfg(), layer=-1).to(dev)      # 50 fps BNF LiveVoice uses
    out = ZipformerContentEncoder(Cfg(), layer="out").to(dev)   # 25 fps encoder output

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    asr = Transducer(sd).to(dev)

    sp = None
    if args.bpe:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe)

    feats: dict[int, list[torch.Tensor]] = {c: [] for c in chunks}
    toks: dict[int, list[list[int]]] = {c: [] for c in chunks}
    refs: list[list[str]] = []

    for n, w in enumerate(wavs):
        audio, sr = torchaudio.load(str(w))
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio.mean(0, keepdim=True).to(dev)
        if audio.shape[-1] < 16000:
            continue
        if sp is not None:
            t = Path(str(w).replace(".wav", ".normalized.txt"))
            if not t.exists():
                continue
            refs.append(normalize(t.read_text()))
        for c in chunks:
            tap.encoder.chunk_size = (c,)
            out.encoder.chunk_size = (c,)
            with torch.no_grad():
                feats[c].append(tap(audio).squeeze(0).float().cpu())
                enc = out(audio, align_to_codec=False)
                toks[c].append(asr.greedy(enc))
        if (n + 1) % 20 == 0:
            print(f"  ...{n + 1}")

    hdr = f"\n{'chunk':>7} {'ms':>6} | {'cos vs c16':>11} {'relL2':>8} | {'TER vs c16':>11}"
    if sp is not None:
        hdr += f" | {'WER':>8} {'dWER':>7}"
    print(hdr)
    print("-" * (len(hdr) + 4))

    # WER for every setting first: the reference row is not necessarily the first one
    # printed, so dWER cannot be filled in during a single pass.
    wers: dict[int, float] = {}
    if sp is not None:
        for c in chunks:
            we, wl = 0, 0
            for t, r in zip(toks[c], refs):
                we += edit_distance(normalize(sp.decode(t)), r)
                wl += len(r)
            wers[c] = 100.0 * we / max(wl, 1)

    for c in chunks:
        cos_all, l2_all, err, ln = [], [], 0, 0
        for f, fr, t, tr in zip(feats[c], feats[ref], toks[c], toks[ref]):
            n = min(f.shape[0], fr.shape[0])
            cos_all.append(torch.nn.functional.cosine_similarity(f[:n], fr[:n], dim=-1).mean())
            l2_all.append((f[:n] - fr[:n]).norm() / fr[:n].norm().clamp_min(1e-8))
            err += edit_distance(t, tr)
            ln += len(tr)
        label = "full" if c < 0 else f"{c * 20}"
        row = (f"{c:>7} {label:>6} | {torch.stack(cos_all).mean():>11.4f} "
               f"{torch.stack(l2_all).mean():>8.4f} | {100.0 * err / max(ln, 1):>10.2f}%")
        if sp is not None:
            row += f" | {wers[c]:>7.2f}% {wers[c] - wers[ref]:>+6.2f}"
        print(row)


if __name__ == "__main__":
    main()
