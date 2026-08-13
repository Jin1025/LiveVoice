# Pinned pseudo-speaker prompt

`DEFAULT_ANON_PROMPT` in `../anonymize_vpc_dirs.py`.
VCTK **p231** (23, F, English / Southern England), utterance 023.

| file | what it is |
|---|---|
| `p231_023_raw48k.wav` | untouched VCTK source, 48 kHz, 9.35 s |
| `p231_023_conditioned_16k_3s.wav` | **what the model is actually conditioned on** — trimmed, resampled to 16 kHz, cropped to 3.00 s ending on speech |
| `p345_226_OLD_broken_16k.wav` | the previous prompt, for comparison. 2.56 s, 43% silence, ends in a 0.4 s pause — this is what made 52% of outputs silent |

Listen to the last half-second of the conditioned file: it must end mid-phrase.
A prompt that fades into a pause makes the codec LM continue with silence.
