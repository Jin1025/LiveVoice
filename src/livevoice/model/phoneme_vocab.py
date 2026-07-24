"""CMU ARPAbet phoneme vocabulary + G2P text→phoneme-id conversion for ASR supervision.

Stress digits (0/1/2) are stripped: stress is prosodic/quasi-speaker information, not
content, so keeping it would leak exactly what the ASR-supervision loss is meant to
force out of the FSQ content bottleneck (see asr_supervision.py).
"""
from __future__ import annotations

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

_CMU_PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH",
    "UW", "V", "W", "Y", "Z", "ZH",
]
assert len(_CMU_PHONES) == 39

PHONEME_VOCAB = ["<pad>", "<bos>", "<eos>"] + _CMU_PHONES
PHONEME_VOCAB_SIZE = len(PHONEME_VOCAB)  # 42
_PH2ID = {p: i for i, p in enumerate(PHONEME_VOCAB)}

_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


def text_to_phoneme_ids(text: str, max_len: int | None = None) -> list[int]:
    """"the cat sat" -> [BOS, DH,AH, K,AE,T, S,AE,T, EOS] (stress stripped, non-phones dropped).

    g2p_en emits ARPAbet tokens with stress digits for words, and the raw punctuation/
    space characters verbatim for everything else — those are silently dropped here
    since they're not in the phoneme vocab.
    """
    g2p = _get_g2p()
    raw = g2p(text)
    ids = [BOS_ID]
    for tok in raw:
        ph = "".join(c for c in tok if not c.isdigit())  # strip stress digit, e.g. AH0 -> AH
        if ph in _PH2ID:
            ids.append(_PH2ID[ph])
    ids.append(EOS_ID)
    if max_len is not None and len(ids) > max_len:
        ids = ids[: max_len - 1] + [EOS_ID]
    return ids
