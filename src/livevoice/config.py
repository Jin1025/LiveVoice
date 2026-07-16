from dataclasses import dataclass

CODEC_SAMPLE_RATES: dict[str, int] = {"mimi": 24000, "jhcodec": 16000}


def codec_sample_rate(codec: str) -> int:
    """Return the dataset/windowing sample rate for a codec."""
    name = str(codec).lower()
    try:
        return CODEC_SAMPLE_RATES[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown codec {codec!r}; expected one of {sorted(CODEC_SAMPLE_RATES)}"
        ) from e


@dataclass
class LiveVoiceConfig:
    """Configuration for the LiveVoice voice-conversion model.

    Mirrors sonic's SonicConfig but adapted for speech:
    - "mimi" or "jhcodec" codec (speech-optimized)
    - HuBERT-base content features (linguistic)
    - Reference cross-attention for speaker timbre
    - Optional F0/loudness prosody conditioning
    """

    # ------------------------------------------------------------------
    # Transformer architecture
    # ------------------------------------------------------------------
    hidden_dim: int = 768 # 512
    num_encoder_layers: int = 4
    num_decoder_layers: int = 12 # 8
    num_heads: int = 8
    ffn_dim: int = 4 * hidden_dim
    dropout: float = 0.1
    max_seq_len: int = 1024

    # ------------------------------------------------------------------
    # Codec — "mimi" or "jhcodec"
    # ------------------------------------------------------------------
    codec: str = "jhcodec"

    # Mimi (kyutai/mimi) — 24 kHz, 12.5 fps, 8 codebooks, codebook_size 2048
    mimi_model_name: str = "kyutai/mimi"
    mimi_n_codebooks: int = 8

    # ------------------------------------------------------------------
    # JHCodec (mimi variant) — 16 kHz, 50 fps, 8 codebooks, codebook_size 1024.
    # Low-latency streaming codec; no encoder downsampling loss like DAC/Mimi.
    # When codec="jhcodec", set sample_rate=16000 so windowing/frame counts match.
    # ------------------------------------------------------------------
    jhcodec_repo: str = "/workspace/jhcodec"
    jhcodec_config: str = "/workspace/jhcodec/config/config_mimi_recon.json"
    jhcodec_ckpt: str = "/mnt/data/disk3/yejin/jhcodec/jhcodec_mimi_1000000.pt"
    jhcodec_sample_rate: int = 16000
    jhcodec_n_codebooks: int = 8
    jhcodec_codebook_size: int = 1024
    jhcodec_hop_length: int = 320  # 16000/320 = 50 frames/sec at 16 kHz

    # ------------------------------------------------------------------
    # Audio / windowing
    # ------------------------------------------------------------------
    sample_rate: int = 16000  # synced to codec in __post_init__ (mimi=24k, jhcodec=16k)
    audio_duration: float = 4.0  # seconds per training window
    train_batch_size: int = 16
    val_batch_size: int = 4
    num_workers: int = 8

    # ------------------------------------------------------------------
    # Content features (HuBERT)
    # ------------------------------------------------------------------
    # HuBERT-base
    hubert_model_name: str = "facebook/hubert-base-ls960"
    hubert_layer: int = 9          # layer 9 is the standard "content" layer (FreeVC/SoVITS)
    hubert_sample_rate: int = 16000 # 16000
    hubert_hidden_dim: int = 768   # HuBERT-base hidden size
    content_proj_dim: int = 256 # 256, 768
    freeze_hubert: bool = True
    # Center-align HuBERT content frames to the codec token grid via waveform padding
    # (jhcodec only; stride==hop). Removes the sub-frame center offset and the
    # count-mismatch resample. See HuBERTContentExtractor._extract_hidden.
    content_center_align: bool = True

    # ------------------------------------------------------------------
    # Prosody features (optional, causal)
    # ------------------------------------------------------------------
    use_prosody: bool = False  # start without, add later if speaker leaks
    prosody_hop_length: int = 320
    n_fft: int = 1024
    pitch_method: str = "crepe"  # "crepe" | "fft"
    pitch_bins: int = 360
    pitch_threshold: float = 0.1
    prosody_hidden_dim: int = 128

    # median filter to make prosody "sketch-like" (cross-speaker transfer)
    use_random_median_filter: bool = True
    median_filter_min_size: int = 1
    median_filter_max_size: int = 15
    median_filter_inference_size: int = 5

    # ------------------------------------------------------------------
    # Conditioning strategy
    # ------------------------------------------------------------------
    # How content features enter the decoder:
    #   "film"     — FiLM (gamma, beta) per layer from raw content logits
    #   "additive" — project to hidden_dim and add to decoder input
    content_conditioning: str = "film"

    # Speaker/reference handling:
    #   "crossattn"  — per-frame reference z through cross-attention (sonic default)
    #   "global_avg" — pool reference z over time, then use cross-attention to one token
    #   "prefix"     — prepend reference-derived speaker tokens to decoder self-attn
    #                  (decoder-only path; no cross-attention)
    speaker_conditioning: str = "prefix"
    speaker_prefix_len: int = 4

    # Speaker encoder:
    #   "codec"             — codec continuous z (pre-quantization) from reference
    #                         audio. With speaker_conditioning="prefix" this z is
    #                         projected (speaker_proj) and downsampled to
    #                         speaker_prefix_len tokens → decoder prefix.
    #   "speechbrain_ecapa" — SpeechBrain ECAPA-TDNN utterance embedding
    speaker_encoder_type: str = "speechbrain_ecapa"
    speechbrain_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    speechbrain_savedir: str = "/mnt/data/disk2/yejin/LiveVoice/pretrained_models/speechbrain__spkrec-ecapa-voxceleb"
    speechbrain_sample_rate: int = 16000
    speechbrain_embedding_dim: int = 192

    # Which speaker encoder the *validation* spk_sim metric uses (does NOT affect
    # the model's own speaker_encoder_type conditioning path):
    #   "ecapa"      — SpeechBrain ECAPA-TDNN (source above). Same-spk cosine ~0.6.
    #   "wavlm_tdnn" — UniSpeech WavLM-large + ECAPA-TDNN head (finetuned .pth).
    #                  Exactly the encoder Vevo/Amphion report SIM with; higher
    #                  absolute cosines (GT ~0.75), directly comparable to those tables.
    #                  Requires s3prl in the env + the finetuned checkpoint below.
    val_spk_encoder: str = "wavlm_tdnn"
    wavlm_sv_ckpt: str = "/mnt/data/disk3/yejin/wavlm_large_finetune.pth"
    wavlm_sv_variant: str = "wavlm_large"   # or "wavlm_base_plus"
    wavlm_sv_sample_rate: int = 16000

    # Classifier-free guidance dropout (training)
    use_cfg_dropout: bool = True
    cfg_drop_both_p: float = 0.1
    cfg_drop_speaker_p: float = 0.1
    cfg_drop_content_p: float = 0.1
    cfg_drop_prosody_p: float = 0.2
    prev_emb_dropout_p: float = 0.1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    learning_rate: float = 1e-4
    warmup_steps: int = 2000  # ~1% of total steps for 100-epoch VCTK training
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    # MusicGen delay pattern over codec codebooks
    use_delay_pattern: bool = True
    n_codebooks_predict: int = 8  # keep it small at 16 kHz (coarse bookss carry most info)
    # Coarse codebooks weighted higher (carry content/prosody, intelligibility);
    # fine codebooks gently down-weighted but not too low (still need detail).
    codebook_loss_weights: tuple[float, ...] = (2.0, 1.5, 1.2, 1.0, 1.0, 0.9, 0.8, 0.7)

    # Auxiliary losses for artifact reduction
    z_loss_weight: float = 1e-4               # PaLM-style logsumexp regularizer
    latent_loss_weight: float = 0.5           # MSE on expected latent vs target latent

    # CTC loss: forces decoder hidden states to be linguistically transcribable.
    # Directly improves WER. Char-level vocab (ASR standard, no extra deps).
    use_ctc_loss: bool = False
    ctc_loss_weight: float = 0.3
    ctc_vocab_size: int = 34                  # 33 chars + 1 blank

    # Content source: how linguistic features are extracted from content audio.
    #   "hubert"        — HuBERT layer-9 hidden states (heavy, bidirectional)
    #   "sw2v"            — jhcodec streaming-wav2vec AudioEncoder, 16 kHz, 50 fps,
    #                       continuous 1024-d (same grid as jhcodec codec tokens).
    content_source: str = "sw2v"

    # SW2V (jhcodec streaming-wav2vec content encoder).
    sw2v_repo: str = "/workspace/jhcodec"
    sw2v_config: str = "/workspace/jhcodec/config/config_w2vcossim.json"
    sw2v_ckpt: str = "/mnt/data/disk3/yejin/jhcodec/sw2v_120000.pt"
    sw2v_sample_rate: int = 16000

    # StreamVoiceAnon causal content tokenizer.
    streamvoiceanon_repo: str = "/workspace/StreamVoiceAnon"
    streamvoiceanon_encoder_config: str = (
        "/workspace/StreamVoiceAnon/configs/hydra_arcs/"
        "speech_tokenizers/causal-encoder-lfq-8192.yaml"
    )
    streamvoiceanon_encoder_ckpt: str = (
        "/workspace/StreamVoiceAnon/ckpt/asr_s2s_bsq_8192_causal_down_whisper.pth"
    )
    streamvoiceanon_sample_rate: int = 44100
    streamvoiceanon_codebook_size: int = 8192
    freeze_streamvoiceanon_encoder: bool = True
    # If True, use pre-quantization continuous z (HuBERT-like — structurally
    # coherent so a random linear projection already carries phonetic info).
    # If False, use discrete codes + learnable nn.Embedding (slower to learn,
    # prone to letting the model fall into a prev_codes+speaker shortcut).
    streamvoiceanon_use_continuous: bool = True

    # Conditioning ablations
    zero_speaker: bool = False
    zero_content: bool = False
    ablate_cross_attn: bool = False

    # ------------------------------------------------------------------
    # Source-side content perturbation (speaker de-identification)
    # ------------------------------------------------------------------
    use_content_perturbation: bool = True
    perturb_pitch_semitones: float = 4.0    # ±N semitones pitch shift
    use_vtln: bool = False                  # VTLN formant warp
    perturb_vtln_alpha_range: float = 0.12  # ±12% VTLN warp range (only if use_vtln=True)
    perturb_eq_gain_db: float = 6.0         # ±dB per EQ band (4 bands)
    perturb_prob: float = 1.0               # fraction of batch items to perturb

    # ------------------------------------------------------------------
    # Training-time Mimi cache
    # ------------------------------------------------------------------
    use_mimi_cache: bool = True
    mimi_cache_dir: str = "/mnt/data/disk2/yejin/LiveVoice/mimi_precomputed"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    num_audio_log_samples: int = 4
    log_val_wer: bool = True
    log_val_spk_sim: bool = True   # ECAPA cosine(gen, reference) — speaker-transfer metric
    # Reference speaker for the epoch-end WER / spk_sim eval:
    #   "same"  — same speaker as content (different utterance) → intelligibility UPPER BOUND
    #   "cross" — a different speaker → speaker-transfer VC quality
    val_wer_speaker: str = "same"
    wer_whisper_model: str = "base"
    wer_device: str = "cuda"
    wer_epoch_samples: int = 50
    wer_seed: int = 12345                     # fixed seed for sample selection (stable across epochs)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    # Pairing scheme:
    #   "same_speaker" — reference and target from same speaker, different utterances
    #                    (reconstruction training — content+speaker → target)
    #   "reconstruct"  — reference and target from SAME utterance (weakest)
    pairing: str = "same_speaker"

    # VCTK
    vctk_path: str = "/mnt/data/disk2/VCTK-Corpus"
    vctk_wav_dirname: str = "wav48"
    vctk_txt_dirname: str = "txt"
    vctk_extensions: tuple[str, ...] = (".wav",)
    vctk_val_speaker_count: int = 8      # hold out this many speakers for val
    vctk_val_speakers: tuple[str, ...] = ()  # empty → auto-select
    train_split_ratio: float = 0.95       # if vctk_val_speakers is empty, 95/5 random utterance split

    # LibriTTS (24 kHz). With codec="mimi" use sample_rate=24000;
    # with codec="jhcodec" use sample_rate=16000.
    libritts_path: str = "/mnt/data/disk2/LibriTTS"
    libritts_train_splits: tuple[str, ...] = (
        "train-clean-100",
        "train-clean-360",
       # "train-other-500",
    )
    libritts_val_splits: tuple[str, ...] = ("dev-clean",) # "dev-other")

    # Precomputed HuBERT features (from extract_features.py)
    # Layout: features_dir/{vctk,libritts}/{speaker_id}/{utt_id}.pt
    features_dir: str = "/mnt/data/disk2/yejin/LiveVoice/features/perturbed"
    # Precomputed SW2V features (from extract_sw2v_features.py). Used instead of
    # features_dir when content_source=="sw2v". Layout: sw2v_features_dir/libritts/...
    sw2v_features_dir: str = "/mnt/data/disk2/yejin/LiveVoice/features/sw2v"

    # Debug cap
    max_windows: int | None = None

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------
    max_epochs: int = 100
    val_check_interval: float = 0.25
    log_every_n_steps: int = 50
    save_top_k: int = 3

    device: str = "cuda"
    precision: str = "32"
    compile: bool = False

    exp_name: str = "base"
    output_dir: str = "/mnt/data/disk2/yejin/LiveVoice"
    seed: int = 42

    # Streaming / inference
    use_kv_cache: bool = True
    streaming_window_frames: int = 10

    sampling_method: str = "top_p"
    top_p: float = 0.9
    top_k: int = 1
    temperature: float = 1.0

    def __post_init__(self) -> None:
        self.codec = str(self.codec).lower()
        self.sample_rate = codec_sample_rate(self.codec)
