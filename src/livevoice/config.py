from dataclasses import dataclass


@dataclass
class LiveVoiceConfig:
    """Configuration for the LiveVoice voice-conversion model.

    Mirrors sonic's SonicConfig but adapted for speech:
    - "dac" or "mimi" codec (speech-optimized)
    - HuBERT-base content features (linguistic)
    - Reference cross-attention for speaker timbre
    - Optional F0/loudness prosody conditioning
    """

    # ------------------------------------------------------------------
    # Transformer architecture
    # ------------------------------------------------------------------
    hidden_dim: int = 512
    num_encoder_layers: int = 4
    num_decoder_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 4 * hidden_dim
    dropout: float = 0.1
    max_seq_len: int = 1024

    # ------------------------------------------------------------------
    # Codec — "dac" or "mimi"
    # ------------------------------------------------------------------
    codec: str = "mimi"

    # Mimi (kyutai/mimi) — 24 kHz, 12.5 fps, 8 codebooks, codebook_size 2048
    mimi_model_name: str = "kyutai/mimi"
    mimi_n_codebooks: int = 8

    # ------------------------------------------------------------------
    # DAC codec (dac 16 or 24 kHz speech model)
    # ------------------------------------------------------------------
    # dac_model_type: 16kHz speech model has 12 RVQ codebooks, hop=320 (50 frames/sec)
    # dac_model_type: 24kHz speech model has 9 RVQ codebooks, hop=320 (75 frames/sec)
    # dac_model_type: 44kHz speech model has 9 RVQ codebooks, hop=320 (137.5 frames/sec)
    dac_model_type: str = "16khz" 
    dac_sample_rate: int = 16000 
    dac_n_codebooks: int = 12
    dac_codebook_size: int = 1024
    dac_depth: int = 9
    dac_latent_dim: int = 1024
    dac_hop_length: int = 320  # 16000/320 = 50 frames/sec at 16 kHz

    # ------------------------------------------------------------------
    # Audio / windowing
    # ------------------------------------------------------------------
    sample_rate: int = 24000 # 24000 or 44100
    audio_duration: float = 4.0  # seconds per training window
    train_batch_size: int = 16
    val_batch_size: int = 4
    num_workers: int = 8

    # ------------------------------------------------------------------
    # Content features (HuBERT)
    # ------------------------------------------------------------------
    # HuBERT-base operates at 44 kHz with hop=320 → 137.5 frames/sec (matches DAC 44 kHz)
    hubert_model_name: str = "facebook/hubert-base-ls960"
    hubert_layer: int = 9          # layer 9 is the standard "content" layer (FreeVC/SoVITS)
    hubert_sample_rate: int = 16000 # 16000
    hubert_hidden_dim: int = 768   # HuBERT-base hidden size
    content_proj_dim: int = 256
    freeze_hubert: bool = True

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
    #   "global_avg" — pool reference z over time → single speaker vector as prefix
    speaker_conditioning: str = "crossattn"

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
    n_codebooks_predict: int = 4  # keep it small at 16 kHz (coarse bookss carry most info)
    codebook_loss_weights: tuple[float, ...] = (1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

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
    wer_whisper_model: str = "base"
    wer_device: str = "cuda"

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

    # LibriTTS (24 kHz)
    # To use LibriTTS, also set: sample_rate=24000, dac_model_type="24khz",
    # dac_sample_rate=24000, dac_n_codebooks=9, dac_hop_length=320
    libritts_path: str = "/mnt/data/disk2/LibriTTS"
    libritts_train_splits: tuple[str, ...] = (
        "train-clean-100",
        "train-clean-360",
       # "train-other-500",
    )
    libritts_val_splits: tuple[str, ...] = ("dev-clean",) # "dev-other")

    # Precomputed HuBERT features (from extract_features.py)
    # Layout: features_dir/{vctk,libritts}/{speaker_id}/{utt_id}.pt
    features_dir: str = "/mnt/data/disk2/yejin/LiveVoice/features"

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
