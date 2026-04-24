# LiveVoice: Streaming Voice Conversion

Codec-based language modeling for streaming any-to-any voice conversion.
Adapted from the LiveSketch (sonic imitation-to-sound) framework.

## Concept

Same timbre/sketch decomposition as LiveSketch, remapped to speech:

- **Timbre / Reference** → **speaker identity** (ECAPA-style cross-attention on reference audio)
- **Sketch / Control** → **linguistic content** (HuBERT hidden states) + optional F0/loudness prosody

```
Reference utt. (3-5 s)   → DAC encoder   → speaker z      → [Cross-Attention]
                                                                     ↓
Content utt. (variable)  → HuBERT (L9)   → content feats  → [AR Decoder] → DAC codes → audio
(optional) F0 + loudness → prosody feats → FiLM on decoder
```

Output duration is driven by the content utterance length.

## Directory layout

```
livevoice/
├── src/
│   ├── livevoice/
│   │   ├── config.py               # LiveVoiceConfig
│   │   ├── model/
│   │   │   ├── codec/
│   │   │   │   └── dac_model.py    # DAC 16 kHz wrapper
│   │   │   ├── content_extractor.py   # HuBERT content features
│   │   │   ├── prosody_extractor.py   # F0 / loudness (optional, causal)
│   │   │   ├── transformer.py         # LiveVoiceTransformer + LiveVoiceModel
│   │   │   └── unconditional.py       # decoder-only baseline
│   │   ├── nn/layer.py             # TransformerBlock (ALiBi + KV cache)
│   │   ├── data/
│   │   │   ├── dataset.py          # VCTKDataset
│   │   │   └── datamodule.py
│   │   ├── lightning/
│   │   │   ├── lightning_module.py       # conditional VC training
│   │   │   └── lightning_module_uncond.py
│   │   └── evaluation/metrics.py   # WER / SECS stubs
│   ├── scripts/
│   │   ├── train.py
│   │   ├── train_uncond.py
│   │   └── inference.py
│   └── utils/save_audio.py
├── requirements.txt
├── setup.py
└── setup.sh
```

## Install

Inside the container (`docker attach yejin2 && conda activate sound`):

```bash
cd /workspace/livevoice
bash setup.sh
```

## Roadmap

1. **Unconditional** (`train_uncond.py`) — sanity check that the decoder can
   model VCTK speech. Expect babbling but speech-like output.
2. **+ Content** — add HuBERT content conditioning only. Expect intelligible
   speech in an averaged speaker voice.
3. **+ Speaker cross-attention** — add reference audio path. Expect voice
   cloning to start working at training-speaker level.
4. **+ Prosody (optional)** — add F0/loudness FiLM to control prosody.
5. **Streaming** — validate KV-cache AR generation works in real time.

## Dataset

- VCTK-Corpus at `/mnt/data/disk2/VCTK-Corpus` (109 speakers, wav48)
- Held-out speakers form the val split (any-to-any generalization)
- Within-split pairs: two different utterances from the **same speaker**
  → one serves as reference (speaker), the other as content+target

Add LibriTTS-R later for better speaker coverage.

## Codec choice

Starts with **DAC 16 kHz** (12 RVQ codebooks, 50 frames/sec). If streaming
latency becomes a bottleneck, swap to Mimi (12.5 frames/sec, speech-optimized)
by writing a drop-in wrapper in `model/codec/`.
