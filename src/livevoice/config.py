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
    jhcodec_ckpt: str = "/mnt/data/disk2/yejin/LiveVoice/pretrained_models/jhcodec/jhcodec_mimi_1000000.pt"
    jhcodec_sample_rate: int = 16000
    jhcodec_n_codebooks: int = 8
    jhcodec_codebook_size: int = 1024
    jhcodec_hop_length: int = 320  # 16000/320 = 50 frames/sec at 16 kHz

    # ------------------------------------------------------------------
    # Audio / windowing
    # ------------------------------------------------------------------
    sample_rate: int = 16000  # synced to codec in __post_init__ (mimi=24k, jhcodec=16k)
    audio_duration: float = 3.0  # seconds per training window
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

    # Cepstral mean (variance) normalisation on the content features, applied at the
    # FRONTEND (raw encoder output, before the refiner). Targets the speaker residual that
    # adversarial removal cannot reach: frame-level probe acc fell to 0.101 while
    # utterance-level plateaued at ~0.517, so what survives is a statistic ACROSS frames.
    #   "off" | "utterance" (offline, exact) | "causal" (running mean — streaming-safe)
    # Changing this changes the content path's input distribution, so a Stage-1 checkpoint
    # trained with one setting must not be evaluated with another.
    content_cmn: str = "causal" # off, utterance, causal
    content_cmn_var: bool = False   # also divide by the std (CMVN rather than CMN)
    
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
    speaker_prefix_len: int = 4 # 4

    # Speaker encoder:
    #   "codec"             — codec continuous z (pre-quantization) from reference audio,
    #                         projected by a SEPARATE speaker_proj → prefix. The AR decoder
    #                         treats this as an alien conditioning vector and largely IGNORES
    #                         it (diag_prefix_used: nulling it changes CE by only ~0.04).
    #   "codec_prompt"      — VALL-E-style: reference is encoded to DISCRETE codec tokens and
    #                         embedded with the SAME codebook embeddings as the AR previous
    #                         tokens (no speaker_proj), so the decoder sees it as "audio to
    #                         continue" and carries the voice. Use with speaker_conditioning="prefix".
    #   "speechbrain_ecapa" — SpeechBrain ECAPA-TDNN utterance embedding (single 192-d vector)
    #   "spark_global"      — Spark-TTS BiCodec global tokens: a fixed 32-token
    speaker_encoder_type: str = "codec_prompt"

    # ── codec_prompt: how faithfully to reproduce VALL-E ────────────────────────
    # Measured on stage2_perspk_codec (debug/diag_prefix_used.py): prepending the prompt
    # as a separate prefix left it just as ignorable as the old ECAPA prefix
    # (null-prompt ΔCE = 0.030 vs 0.036), even though the prompt DOES carry speaker once
    # prev_emb is removed (PROMPT-ISOLATED gain +0.278). The decoder takes speaker from
    # prev_emb (ΔCE +1.229) because a prepended prefix competes with prev_emb instead of
    # BEING it. In VALL-E the AR is trained as a pure causal LM over one acoustic stream,
    # so the enrolled prompt simply occupies the earlier positions of that stream and the
    # model's ordinary "continue the voice of prev tokens" behaviour transfers it.
    #
    # True → concatenate reference codes with target codes into ONE delayed AR stream
    #        (single BOS at the very front, no separate prefix), so the reference tail
    #        lands in the prev_emb slots at the target boundary.
    # False → legacy behaviour: prompt prepended as a separate prefix block.
    codec_prompt_continuation: bool = True
    # Give the prompt positions a zero ALiBi distance penalty. Pure VALL-E would say no
    # (the prompt is just earlier audio at its natural distance), but ALiBi decays so fast
    # that 5 of 8 heads cannot reach a 200-frame prompt at all (debug/diag_prompt_attention.py).
    codec_prompt_alibi_exempt: bool = True
    # Feed the REFERENCE's own content features over the prompt region (run the content
    # encoder on reference_audio as well). VALL-E/CosyVoice2 do the equivalent: at inference
    # they concatenate the PROMPT's transcript with the target's, so the prompt region is a
    # conditioned prediction task exactly like the target region. Without this the prompt
    # region has no content signal and scoring it is unconditional audio LM (high entropy).
    # In TRAINING the dataset supplies "reference_feats" sliced from the same feature cache
    # as the target's, so this costs no extra encoder pass and both regions share one
    # normalisation and the same baked-in perturbation. At INFERENCE an arbitrary reference
    # has no cache entry, so the content encoder runs on the reference audio.
    codec_prompt_content: bool = True # False
    # Weight of the codec LM loss on the PROMPT region (0 = don't score it at all, the
    # default). The continuation mechanism does not need it — "continue the voice" is learned
    # from the target region — but scoring the prompt shapes its hidden states, which is what
    # the target attends to; right now those states are only shaped by a very weak attention
    # gradient (null-prompt ΔCE ≈ 0.01). Keep it BELOW 1.0: the target region is what we
    # actually want to be good at, and the prompt is ~half the stream. Pair with
    # codec_prompt_content=True so the scored task is conditioned rather than unconditional.
    codec_prompt_loss_weight: float = 0.3

    speechbrain_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    speechbrain_savedir: str = "/mnt/data/disk2/yejin/LiveVoice/pretrained_models/speechbrain__spkrec-ecapa-voxceleb"
    speechbrain_sample_rate: int = 16000
    speechbrain_embedding_dim: int = 192

    # Spark-TTS BiCodec global-token speaker encoder (speaker_encoder_type="spark_global").
    # Loads ONLY the global tokenizer (ECAPA→perceiver→FSQ) + mel; Reference is encoded once per utterance.
    spark_repo: str = "/workspace/Spark-TTS"
    spark_bicodec_dir: str = "/workspace/Spark-TTS/pretrained_models/Spark-TTS-0.5B/BiCodec"
    spark_sample_rate: int = 16000

    # Which speaker encoder the *validation* spk_sim metric uses 
    #   "ecapa"      — SpeechBrain ECAPA-TDNN (source above).
    #   "wavlm_tdnn" — UniSpeech WavLM-large + ECAPA-TDNN head (finetuned .pth).
    val_spk_encoder: str = "wavlm_tdnn"
    wavlm_sv_ckpt: str = "/mnt/data/disk2/yejin/LiveVoice/pretrained_models/wavlm_large_finetune.pth"
    wavlm_sv_variant: str = "wavlm_large"   # or "wavlm_base_plus"
    wavlm_sv_sample_rate: int = 16000

    # Classifier-free guidance dropout (training)
    val_cfg_scale: float = 1.0

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

    # Seq2seq ASR supervision on the sw2v content embedding (StyleStream-style;
    # arXiv:2602.20113 confirmed: seq2seq, NOT CTC, char-level, joint end-to-end,
    # discarded at inference). This hangs off sw2v_proj[+content_fsq] ALONE — never
    # touches the main decoder — so it gives the FSQ bottleneck a supervised reason to
    # keep phonetic info and drop speaker info, instead of competing with the
    # codec-token cross-entropy. Label unit is phoneme (CMU ARPAbet, phoneme_vocab.py) 
    # rather than StyleStream's character — more standard for disentanglement 
    # (PPG-VC lineage), coarser/less speaker-specific. Runs on the FULL (un-cropped) 
    # cached sw2v features, since text labels are utterance-level with no timestamps 
    # (see libritts_dataset.py / extract_phonemes.py).
    use_asr_supervision: bool = False
    asr_loss_weight: float = 0.3
    asr_decoder_layers: int = 4               
    asr_max_content_frames: int = 750         
    asr_max_phoneme_len: int = 300
    # Teacher-forcing input dropout — CRITICAL. Without it the seq2seq head minimizes
    # asr_loss via the phoneme LM and IGNORES content (verified: content-usage gap ≈ 0),
    # so the "keep content" supervision does nothing. Zeroing this fraction of input
    # phoneme embeddings forces the decoder to read the content memory.
    asr_teacher_dropout: float = 0.2
    # Which ASR-supervision head shapes the content bottleneck:
    #   "seq2seq" — StyleStream-style attention decoder (needs asr_teacher_dropout>0 or it
    #               LM-cheats and ignores content — see diag_asr_uses_content.py).
    #   "ctc"     — per-frame CTC on the content memory; structurally CANNOT LM-cheat, so
    #               it forces every frame to be phonetic. (Different from the earlier failed
    #               CTC that hung off the decoder hidden — this is on content only.)
    asr_supervision_type: str = "seq2seq"

    # ------------------------------------------------------------------
    # Speaker adversary via Gradient Reversal (GRL) on the content bottleneck.
    # Counterpart to the ASR supervision: ASR *keeps* phonetic content, GRL
    # *removes* speaker identity. A speaker classifier is trained on the
    # (mean-pooled) content embedding; the reversed gradient pushes the content
    # path to make speaker un-decodable. Training-only, discarded at inference
    # (see model/speaker_grl.py). Reuses the FULL-utterance sw2v cache like ASR.
    use_speaker_grl: bool = False
    # How the adversary pushes speaker out of content:
    #   "reversal"  — classic GRL: content MAXIMIZES the classifier CE. Unbounded above,
    #                 so it can run away (grl_loss climbs past chance → NaN → crash).
    #   "confusion" — content drives the classifier's output to UNIFORM (bounded below by
    #                 log K). Same goal, cannot diverge. Recommended.
    grl_objective: str = "confusion"
    grl_loss_weight: float = 0.3             # scalar on the (already λ-scaled) GRL loss
    grl_lambda_max: float = 0.5             # peak reversal strength (Ganin schedule); 0.0 = pure-classifier diagnostic
    grl_gamma: float = 5.0                    # schedule steepness (gentler; 10 ramps λ too fast for the slow head)
    grl_start_step: int = 10000                # λ=0 head-start: batch8/1151-class classifier is SLOW
                                              # (≈0.2 acc @1.5k, rising) — let it get competent BEFORE reversal,
                                              # or λ knocks the untrained head back to chance forever
                                              # (verified: debug/diag_grl_classifier.py + λ=0 run).
    grl_warmup_steps: int = 20000             # steps to ramp λ 0→lambda_max after start_step
    grl_hidden_dim: int = 0                   # adversary MLP hidden width; 0 → = hidden_dim (768)
    grl_num_speakers: int = 0                 # 자동으로 채워짐. adversary class count (speakers OR clusters)
    # Adversary target granularity. 0 → full per-speaker ID (1151-way; hard for batch 8).
    # >0 → classify k-means speaker CLUSTERS (on ECAPA embeddings) 
    grl_num_clusters: int = 0 # 256
    grl_cluster_file: str = "/mnt/data/disk2/yejin/LiveVoice/features/speaker_clusters/256_clusters.json"

    # Content source: how linguistic features are extracted from content audio.
    #   "hubert"        — HuBERT layer-9 hidden states (heavy, bidirectional)
    #   "sw2v"            — jhcodec streaming-wav2vec AudioEncoder, 16 kHz, 50 fps,
    #                       continuous 1024-d (same grid as jhcodec codec tokens).
    #   "zipformer"       — icefall streaming-Zipformer ASR bottleneck features, 50 fps, 512-d. 
    content_source: str = "zipformer"

    # Zipformer content encoder (content_source="zipformer").
    # Architecture is recovered from the checkpoint's tensor shapes, so only the path and
    # the tap matter here. zipformer_layer: -1 = just before the final 50->25Hz downsample
    # (deepest, still 50 fps); 0..5 = that stack's output; "out" = the 25 fps encoder output,
    # which is NOT on the codec grid and would need upsampling.
    zipformer_ckpt: str = "/mnt/data/disk2/yejin/LiveVoice/pretrained_models/zipformer_pretrained.pt"
    zipformer_layer: str = "-1"
    # Annotated (unlike a bare ``= None``) so it stays a real dataclass field even when off:
    # an unannotated attribute is skipped by dataclasses.asdict, so it never reached the
    # checkpoint and a ckpt could not say whether it was trained cached or on-the-fly.
    zipformer_features_dir: str | None = None
    # zipformer_features_dir: str | None = "/mnt/data/disk2/yejin/LiveVoice/features/zipformer_cmn_6"
    # Front padding (in 50 fps frames)
    #   0  → best lag −4  (content ~80 ms stale, NO added latency)  ← streaming default
    #  −6  → best lag  0  (aligned, but ~120 ms of lookahead)
    zipformer_align_pad_frames: int = -6
    # Set True when the feature cache was extracted with CMN already applied
    content_cmn_in_cache: bool = False

    # SW2V (jhcodec streaming-wav2vec content encoder).
    sw2v_repo: str = "/workspace/jhcodec"
    sw2v_config: str = "/workspace/jhcodec/config/config_w2vcossim.json"
    sw2v_ckpt: str = "/mnt/data/disk2/yejin/LiveVoice/pretrained_models/jhcodec/sw2v_120000.pt"
    sw2v_sample_rate: int = 16000

    # FSQ information bottleneck on the content path (StyleStream-style, arXiv:2602.20113).
    # Applied AFTER sw2v_proj (content_proj_dim), per-frame:
    #   content_proj_dim → len(fsq_levels) dims → tanh-bound round (STE) → content_proj_dim.
    use_content_fsq: bool = False
    fsq_levels: tuple = (8,5,5,5) # (8,5,5,5)=1000 → (8,6,5)=240 → (5,3,3)=45.

    # Deep causal content refiner ("Destylizer") on the cached sw2v features, BEFORE
    # sw2v_proj. Adds trainable DEPTH where GRL/ASR apply so a nonlinear map (not just the
    # shallow 1024→256 linear) can suppress speaker while keeping content — without
    # unfreezing the encoder (caches stay valid) and staying streamable (causal convs).
    # 0 = off (shallow linear head as before). See model/content_refiner.py.
    content_refiner_layers: int = 0
    content_refiner_kernel: int = 5

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
    use_content_perturbation: bool = False # True
    perturb_pitch_semitones: float = 4.0    # ±N semitones pitch shift
    use_vtln: bool = False              # Praat formant (VTLN) shift; pitch & timing preserved
    perturb_formant_ratio_range: float = 0.4  # formant ratio = 1 ± this, FIXED magnitude / random direction (Praat 'Change gender')
    perturb_eq_gain_db: float = 6.0         # ±dB per EQ band (4 bands)
    perturb_prob: float = 1.0               # fraction of batch items to perturb

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    num_audio_log_samples: int = 4
    log_val_wer: bool = True
    log_val_spk_sim: bool = True   # cosine(generated, reference) 
    # Reference speaker for the epoch-end WER / spk_sim eval:
    #   "same"  — same speaker as content (different utterance) → intelligibility UPPER BOUND
    #   "cross" — a different speaker → speaker-transfer VC quality
    # NOTE: same-speaker is ALWAYS measured (val/wer_full_epoch_mean, val/spk_sim,
    # val/spk_sim_gt). When val_eval_cross_spk=True, a cross-speaker pass is ALSO
    # run per item → val/wer_cross + val/spk_sim_cross (the real conversion metric:
    # does the output adopt a NEW speaker?). This ~doubles generation cost per epoch.
    val_wer_speaker: str = "same"
    val_eval_cross_spk: bool = True
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
    # Content/reference pairing for VC training:
    #   "same_speaker"          — reference = a DIFFERENT utterance of the same speaker (default)
    #   "reconstruct"           — reference = the content utterance itself (autoencoding)
    #   "same_utterance_window" — content + reference are two NON-OVERLAPPING windows of the
    #                             SAME utterance: identical recording session/channel, so the
    #                             reference's speaker identity matches the target exactly.
    #                             Set audio_duration to the window length (e.g. 3.0) and note
    #                             only files >= 2×window survive. WATCH OUT: adjacent windows
    #                             share prosody, so the model may learn to copy THIS utterance's
    #                             intonation rather than speaker identity — ALWAYS eval with a
    #                             cross-utterance reference to check generalization.
    #   "same_utterance_continuation" — like above but the two windows are ADJACENT and
    #                             ORDERED (reference immediately precedes the target), with a
    #                             SINGLE gain applied across both. Pair with
    #                             codec_prompt_continuation=True: the joint AR stream
    #                             [reference ; target] is then one genuinely continuous piece
    #                             of audio, which is how VALL-E trains ("pure casual language
    #                             model training", no explicit prompt at train time). Splicing
    #                             two different recordings — what same_speaker does — puts a
    #                             channel discontinuity exactly at the prompt->target boundary
    #                             and teaches the model it may reset the voice there.
    #                             SAME CAVEAT as above, doubled: prompt and target share the
    #                             recording session, so the model can copy CHANNEL instead of
    #                             speaker identity (VALL-E survives this on 60k h; LibriTTS is
    #                             ~500 h). ALWAYS evaluate with a cross-utterance reference.
    pairing: str = "same_utterance_continuation"
    same_utt_min_gap_seconds: float = 0.0   # min silence gap between the two windows (same_utterance_window)

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

    # ASR/GRL: extract the full-utterance sw2v features ONLINE from audio (run the encoder live) instead of reading the cache above.
    sw2v_full_online: bool = False
    # Precomputed HuBERT features (from extract_features.py)
    features_dir: str = "/mnt/data/disk2/yejin/LiveVoice/features/perturbed/hubert"
    # Precomputed SW2V features (from extract_sw2v_features.py)
    sw2v_features_dir: str = "/mnt/data/disk2/yejin/LiveVoice/features/sw2v"
    # sw2v_features_dir: str = "/mnt/data/disk2/yejin/LiveVoice/features/perturbed/sw2v"
    # Precomputed phoneme-id caches (scripts/extract_phonemes.py); mirrors sw2v_features_dir.
    phoneme_cache_dir: str = "/mnt/data/disk2/yejin/LiveVoice/features/phonemes"

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
