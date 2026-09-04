"""LiveVoice transformer: streaming voice conversion.

Mirrors sonic's SketchTransformer / SketchModel but rewired for speech:
- Reference audio → codec continuous z → cross-attention or decoder prefix (speaker / timbre)
- Content audio   → HuBERT / Mimi z / StreamVoiceAnon tokens → additive or FiLM conditioning
- Optional prosody (F0 + loudness)   → additive (only if config.use_prosody)
- AR decoder over codec codebooks with MusicGen delay pattern
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from livevoice.nn.layer import TransformerBlock, init_kv_cache_ext
from livevoice.model.content_perturbation import ContentPerturbation
from livevoice.model.fsq import FSQBottleneck
from livevoice.model.speechbrain_speaker_encoder import SpeechBrainECAPASpeakerEncoder
from livevoice.model.spark_speaker_encoder import SparkGlobalSpeakerEncoder
from livevoice.model.asr_supervision import AsrSupervisionHead
from livevoice.model.content_supervision import (
    ContentSupervisionMixin,
    apply_content_cmn,
    build_content_path,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


class MPMInputProjection(nn.Module):
    """DiffAnon-style prosody projection: MPM latent -> one additive term at the decoder
    input (arXiv:2604.26281 projects c_pro through a conv module and adds it to the latent).

    Normalisation is NOT here — it is the shared mpm_norm flag, so that both conditioning
    shapes can be run with and without it.

    Convs are left-padded, so frame t never sees t+1 and streaming latency is unchanged.
    """

    def __init__(self, mpm_dim: int, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv1 = nn.Conv1d(mpm_dim, hidden_dim, kernel_size)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size)
        # Zero-init the last conv so the model starts identical to one without prosody and
        # opens the path only if the objective pays for it.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, mpm_feats: torch.Tensor) -> torch.Tensor:
        """(B, T, mpm_dim) -> (B, T, hidden_dim)."""
        x = mpm_feats.transpose(1, 2)                     # (B, C, T) for conv1d
        x = self.act(self.conv1(F.pad(x, (self.pad, 0))))
        x = self.conv2(F.pad(x, (self.pad, 0)))
        return x.transpose(1, 2)


# =====================================================================
# Transformer backbone
# =====================================================================
class LiveVoiceTransformer(nn.Module):
    """Encoder (speaker) + decoder (causal AR with content / prosody control)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = config.max_seq_len
        self.num_heads = config.num_heads

        self.use_speaker_prefix = (
            str(getattr(config, "speaker_conditioning", "crossattn")).lower() == "prefix"
        )
        self.encoder_layers = nn.ModuleList(
            []
            if self.use_speaker_prefix
            else [
                TransformerBlock(config, purpose="encoder")
                for _ in range(config.num_encoder_layers)
            ]
        )
        decoder_purpose = "decoder_only" if self.use_speaker_prefix else "decoder"
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(config, purpose=decoder_purpose)
            for _ in range(config.num_decoder_layers)
        ])

        self.start_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.encoder_norm = nn.LayerNorm(config.hidden_dim)
        self.decoder_norm = nn.LayerNorm(config.hidden_dim)

        # Optional FiLM conditioning from content logits (content_conditioning == "film")
        self.use_film = str(getattr(config, "content_conditioning", "additive")) == "film"
        if self.use_film:
            film_in = int(config.content_proj_dim)
            self.film_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(film_in, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
                )
                for _ in range(config.num_decoder_layers)
            ])
            for mlp in self.film_mlps:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)

        # Prosody conditioning from MPM hidden states. Two shapes, see mpm_conditioning:
        #   perlayer   — a zero-init Linear per decoder layer, added after every layer, on a
        #                gradient path separate from the content FiLM.
        #   input_conv — DiffAnon: one causal-conv projection added at the decoder input.
        _use_mpm = bool(getattr(config, "use_mpm", False))
        _mode = str(getattr(config, "mpm_conditioning", "perlayer")).lower()
        if _use_mpm and _mode not in ("perlayer", "input_conv"):
            raise ValueError(f"unknown mpm_conditioning {_mode!r}; expected "
                             f"'perlayer' or 'input_conv'")
        self.use_mpm_perlayer = _use_mpm and _mode == "perlayer"
        self.use_mpm_input = _use_mpm and _mode == "input_conv"
        mpm_dim = int(getattr(config, "mpm_perlayer_dim", 256))
        # Shared by both shapes: CausalMPM.extract() taps layer 8 and returns it
        # unnormalised (rms~10), which is why the projections only have to grow a little
        # before they swamp the hidden state they are added to.
        self.mpm_norm = (nn.LayerNorm(mpm_dim)
                         if _use_mpm and bool(getattr(config, "mpm_norm", False))
                         else None)
        if self.use_mpm_perlayer:
            self.mpm_layer_projs = nn.ModuleList([
                nn.Linear(mpm_dim, config.hidden_dim)
                for _ in range(config.num_decoder_layers)
            ])
            for proj in self.mpm_layer_projs:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
        elif self.use_mpm_input:
            self.mpm_input_proj = MPMInputProjection(
                mpm_dim, config.hidden_dim,
                int(getattr(config, "mpm_input_conv_kernel", 5)),
            )

    def encode_speaker(self, speaker_embedding: torch.Tensor) -> torch.Tensor:
        x = speaker_embedding
        for layer in self.encoder_layers:
            x = layer(x, mem=None, use_cache=False)
        return self.encoder_norm(x)

    def decode_step(
        self,
        decoder_input: torch.Tensor,
        content_add: torch.Tensor,
        film_feats: torch.Tensor | None,
        encoder_output: torch.Tensor | None,
        use_cache: bool = False,
        mpm_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_film and film_feats is None:
            raise RuntimeError("content_conditioning='film' requires film_feats.")
        x = decoder_input + content_add
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, mem=encoder_output, use_cache=use_cache)
            if self.use_film:
                gb = self.film_mlps[i](film_feats)
                gamma = 1.0 + gb[..., : self.hidden_dim]
                beta = gb[..., self.hidden_dim :]
                x = x * gamma + beta
            if self.use_mpm_perlayer and mpm_feats is not None:
                x = x + self.mpm_layer_projs[i](mpm_feats)
        return self.decoder_norm(x)

    def decode_step_stateless(
        self,
        decoder_input: torch.Tensor,
        content_add: torch.Tensor,
        film_feats: torch.Tensor | None,
        encoder_output: torch.Tensor | None,
        caches: list,
        use_cache: bool = False,
        mpm_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list]:
        if self.use_film and film_feats is None:
            raise RuntimeError("content_conditioning='film' requires film_feats.")
        x = decoder_input + content_add
        new_caches = []
        for i, layer in enumerate(self.decoder_layers):
            kv_in = caches[i] if caches is not None else {"self": None, "cross": None}
            x, kv_out = layer.forward_stateless(x, kv_in, encoder_output, use_cache=use_cache)
            if self.use_film:
                gb = self.film_mlps[i](film_feats)
                gamma = 1.0 + gb[..., : self.hidden_dim]
                beta = gb[..., self.hidden_dim :]
                x = x * gamma + beta
            if self.use_mpm_perlayer and mpm_feats is not None:
                x = x + self.mpm_layer_projs[i](mpm_feats)
            new_caches.append(kv_out)
        return self.decoder_norm(x), new_caches

    def decode_step_cudagraph(
        self,
        decoder_input: torch.Tensor,
        content_add: torch.Tensor,
        film_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fixed-shape one-token decoder body used only during CUDA capture."""
        x = decoder_input + content_add
        for i, layer in enumerate(self.decoder_layers):
            x = x + layer.self_attn.forward_cudagraph(layer.ln1(x))
            x = x + layer.ff(layer.ln3(x))
            if self.use_film:
                gb = self.film_mlps[i](film_feats)
                gamma = 1.0 + gb[..., : self.hidden_dim]
                beta = gb[..., self.hidden_dim :]
                x = x * gamma + beta
        return self.decoder_norm(x)

    def make_decode_cudagraph(self, decoder_input, content_add, film_feats):
        return _LiveVoiceDecodeGraph(self, decoder_input, content_add, film_feats).capture()

    def forward(
        self,
        speaker_embedding: torch.Tensor,
        content_add: torch.Tensor,
        film_feats: torch.Tensor | None,
        prev_embeddings: torch.Tensor,
        use_cache: bool = False,
        mpm_feats: torch.Tensor | None = None,
    ) -> dict:
        encoder_output = self.encode_speaker(speaker_embedding)
        if prev_embeddings is None:
            raise ValueError("prev_embeddings is required (training) — use generate() for inference.")
        decoder_output = self.decode_step(
            prev_embeddings, content_add, film_feats, encoder_output, use_cache,
            mpm_feats=mpm_feats,
        )
        return {"decoder_output": decoder_output, "encoder_output": encoder_output}

    def init_caches(self, batch_size: int, device: torch.device) -> list:
        caches = []
        head_dim = self.hidden_dim // self.num_heads
        for layer in self.decoder_layers:
            self_sink = int(getattr(layer.self_attn, "sink_size", 1))
            caches.append({
                "self": init_kv_cache_ext(
                    batch_size, self.num_heads, self.max_seq_len, head_dim, device,
                    sink_size=self_sink,
                ),
                "cross": init_kv_cache_ext(
                    batch_size, self.num_heads, self.max_seq_len, head_dim, device,
                    sink_size=1,
                ),
            })
        return caches

    def set_speaker_prefix_len(self, prefix_len: int):
        """Broadcast the ACTUAL speaker-prefix length to every decoder self-attention.

        Needed because the ALiBi exemption / KV sink used to follow the CONFIGURED
        `speaker_prefix_len` (a constant, 4), while the codec / codec_prompt speaker
        encoders build one prefix token per reference frame (~200 for a 4s reference).
        Call before priming the cache.
        """
        for layer in self.decoder_layers:
            setter = getattr(layer.self_attn, "set_active_prefix_len", None)
            if setter is not None:
                setter(prefix_len)

    def reset_caches(self):
        for layer in self.decoder_layers:
            if hasattr(layer.self_attn, "_cache_inited"):
                layer.self_attn._cache_inited = False
            if hasattr(layer, "cross_attn") and hasattr(layer.cross_attn, "_cache_inited"):
                layer.cross_attn._cache_inited = False


class _LiveVoiceDecodeGraph:
    """One CUDA-graph replay for the complete 12-layer AR decoder step."""

    def __init__(self, transformer, decoder_input, content_add, film_feats):
        if decoder_input.shape[0] != 1 or decoder_input.shape[1] != 1:
            raise ValueError("LiveVoice CUDA graph currently supports B=1, T=1")
        if transformer.use_mpm_perlayer:
            raise ValueError("LiveVoice CUDA graph does not support per-layer MPM")
        self.transformer = transformer
        self.static_decoder_input = torch.empty_like(decoder_input)
        self.static_content_add = torch.empty_like(content_add)
        self.static_film_feats = None if film_feats is None else torch.empty_like(film_feats)
        for layer in transformer.decoder_layers:
            if layer.do_cross_attn:
                raise ValueError("LiveVoice CUDA graph requires decoder-only/prefix conditioning")
            layer.self_attn.prepare_cudagraph_cache()
        self.graph = None
        self.output = None

    def _fn(self):
        return self.transformer.decode_step_cudagraph(
            self.static_decoder_input, self.static_content_add, self.static_film_feats)

    def reset(self):
        for layer in self.transformer.decoder_layers:
            layer.self_attn.reset_cudagraph_cache()

    def refresh_prefix(self):
        for layer in self.transformer.decoder_layers:
            layer.self_attn.refresh_cudagraph_prefix()

    def capture(self):
        torch.cuda.synchronize()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                self._fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.reset()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.output = self._fn()
        torch.cuda.synchronize()
        self.reset()
        return self

    @torch.no_grad()
    def step(self, decoder_input, content_add, film_feats):
        self.static_decoder_input.copy_(decoder_input)
        self.static_content_add.copy_(content_add)
        if self.static_film_feats is not None:
            self.static_film_feats.copy_(film_feats)
        self.graph.replay()
        return self.output


# =====================================================================
# Full model: codec + content + prosody + transformer
# =====================================================================
class LiveVoiceModel(nn.Module, ContentSupervisionMixin):
    """Full VC pipeline.

    Components:
      - codec_model:        frozen codec (16 kHz or 24 kHz speech)
      - content_extractor: HuBERT or StreamVoiceAnon content features
      - prosody_extractor: (optional) F0 + loudness
      - transformer:      LiveVoiceTransformer
    """

    def __init__(self, config, codec_model, content_extractor, prosody_extractor=None,
                 cepstral_extractor=None, mpm_extractor=None):
        super().__init__()
        self.config = config
        self.codec_model = codec_model
        self.content_extractor = content_extractor
        self.prosody_extractor = prosody_extractor
        self.cepstral_extractor = cepstral_extractor
        self.mpm_extractor = mpm_extractor
        self.use_cepstral = bool(getattr(config, "use_cepstral", False)) and (cepstral_extractor is not None)
        self.use_prosody = bool(getattr(config, "use_prosody", False)) and (prosody_extractor is not None)
        self.use_mpm = bool(getattr(config, "use_mpm", False)) and (mpm_extractor is not None)
        if self.use_mpm:
            config.mpm_perlayer_dim = mpm_extractor.output_dim

        self.transformer = LiveVoiceTransformer(config)

        # Learned null embeddings for CFG-style dropout
        self.null_speaker_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.normal_(self.null_speaker_embedding, std=0.02)
        self.null_content_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.normal_(self.null_content_embedding, std=0.02)
        self.null_prev_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.normal_(self.null_prev_embedding, std=0.02)
        if str(getattr(config, "content_conditioning", "additive")) == "film":
            self.null_film_feature = nn.Parameter(torch.zeros(1, 1, config.content_proj_dim))
            nn.init.normal_(self.null_film_feature, std=0.02)

        # Source-side content perturbation (speaker de-identification before HuBERT)
        self.use_content_perturbation = bool(getattr(config, "use_content_perturbation", True))
        if self.use_content_perturbation:
            self.content_perturbation = ContentPerturbation(config)

        # Per-codebook decoder input projections + output heads (MusicGen delay)
        K = int(config.n_codebooks_predict)
        self.n_codebooks_predict = K
        latent_dim    = codec_model.latent_dim
        codebook_size = codec_model.codebook_size

        n_proper_init = 0
        for k in range(K):
            emb = codec_model.get_codebook_embeddings(k)
            if emb is None:
                emb = torch.randn(codebook_size, latent_dim) * 0.02
            else:
                n_proper_init += 1
                # Pad/truncate to latent_dim if codec's codebook dim differs
                if emb.shape[1] != latent_dim:
                    if emb.shape[1] < latent_dim:
                        pad = torch.zeros(
                            emb.shape[0],
                            latent_dim - emb.shape[1],
                            device=emb.device,
                            dtype=emb.dtype,
                        )
                        emb = torch.cat([emb, pad], dim=1)
                    else:
                        emb = emb[:, :latent_dim]
            self.register_buffer(f"codebook_vectors_{k}", emb.float())
        print(f"[LiveVoiceModel] codebook_vectors: {n_proper_init}/{K} initialized from codec (rest random)")

        self.decoder_input_projs = nn.ModuleList([
            nn.Linear(latent_dim, config.hidden_dim) for _ in range(K)
        ])
        self.codebook_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, codebook_size) for _ in range(K)
        ])

        self.speaker_conditioning = str(
            getattr(config, "speaker_conditioning", "crossattn")
        ).lower()
        if self.speaker_conditioning not in {"crossattn", "global_avg", "prefix"}:
            raise ValueError(
                "speaker_conditioning must be one of: crossattn, global_avg, prefix; "
                f"got {self.speaker_conditioning!r}"
            )
        if self.speaker_conditioning == "prefix" and int(getattr(config, "speaker_prefix_len", 8)) <= 0:
            raise ValueError("speaker_conditioning='prefix' requires speaker_prefix_len > 0.")
        self.speaker_encoder_type = str(getattr(config, "speaker_encoder_type", "codec")).lower()
        if self.speaker_encoder_type not in {"codec", "codec_prompt", "speechbrain_ecapa", "spark_global"}:
            raise ValueError(
                "speaker_encoder_type must be one of: codec, codec_prompt, speechbrain_ecapa, "
                f"spark_global; got {self.speaker_encoder_type!r}"
            )
        if self.speaker_encoder_type == "codec_prompt":
            # VALL-E-style: the reference is embedded with the SAME codebook embeddings as
            # the AR previous tokens (codebook_vectors_* + decoder_input_projs), so the
            # decoder sees it as "audio to continue" and carries the voice — no separate
            # speaker_proj (which the AR just ignored; see debug/diag_prefix_used.py).
            self.speaker_encoder = None
            print("[LiveVoiceModel] codec_prompt speaker: reference codec tokens embedded via "
                  "the shared codebook embeddings (VALL-E-style in-context prompt)")
        elif self.speaker_encoder_type == "speechbrain_ecapa":
            self.speaker_encoder = SpeechBrainECAPASpeakerEncoder(config)
            spk_dim = int(getattr(config, "speechbrain_embedding_dim", 192))
            self.speaker_proj = nn.Linear(spk_dim, config.hidden_dim)
            if self.speaker_conditioning == "prefix":
                prefix_len = int(getattr(config, "speaker_prefix_len", 8))
                self.speaker_prefix_proj = nn.Linear(spk_dim, prefix_len * config.hidden_dim)
        elif self.speaker_encoder_type == "spark_global":
            # Spark-TTS BiCodec global tokens: a fixed token_num-length sequence of
            # FSQ-quantized speaker latents. Each token → one decoder prefix token,
            # so the prefix length is token_num (32), independent of speaker_prefix_len.
            self.speaker_encoder = SparkGlobalSpeakerEncoder(config)
            spk_dim = int(self.speaker_encoder.out_dim)
            self.spark_speaker_proj = nn.Linear(spk_dim, config.hidden_dim)
            print(f"[LiveVoiceModel] spark_global speaker: token_num={self.speaker_encoder.token_num} "
                  f"latent={spk_dim} → prefix of {self.speaker_encoder.token_num} tokens")
        else:
            self.speaker_encoder = None
            self.speaker_proj = nn.Linear(latent_dim, config.hidden_dim)

        # ── Content source path ────────────────────────────────────
        # "hubert"          — uses self.content_extractor (HuBERT)
        # "mimi_semantic"   — uses Mimi continuous z directly.
        # "streamvoiceanon" — uses StreamVoiceAnon causal content tokenizer.
        codec_name = str(getattr(config, "codec", "mimi")).lower()
        default_src = "mimi_semantic" if codec_name == "mimi" else "hubert"
        self.content_source = str(getattr(config, "content_source", default_src)).lower()
        valid_content_sources = {"hubert", "mimi_semantic", "streamvoiceanon", "sw2v",
                                 "zipformer", "fastconformer"}
        if self.content_source not in valid_content_sources:
            raise ValueError(
                f"content_source must be one of {sorted(valid_content_sources)}; "
                f"got {self.content_source!r}"
            )
        if self.content_source == "mimi_semantic" and codec_name != "mimi":
            # Silently fall back; HuBERT works for any codec.
            self.content_source = "hubert"
        # If HuBERT extractor wasn't supplied but is needed, fall back
        if self.content_source == "hubert" and self.content_extractor is None:
            raise ValueError(
                "content_source='hubert' but content_extractor was not provided. "
                "Either pass HuBERTContentExtractor or set content_source='mimi_semantic'."
            )
        if self.content_source == "streamvoiceanon" and self.content_extractor is None:
            raise ValueError(
                "content_source='streamvoiceanon' but content_extractor was not provided. "
                "Pass StreamVoiceAnonContentEncoder."
            )
        if self.content_source in ("sw2v", "zipformer", "fastconformer") and self.content_extractor is None:
            raise ValueError(
                f"content_source={self.content_source!r} but content_extractor was not "
                f"provided. Pass "
                f"{'Sw2vContentEncoder' if self.content_source == 'sw2v' else 'FastConformerContentEncoder' if self.content_source == 'fastconformer' else 'ZipformerContentEncoder'}."
            )
        print(
            f"[LiveVoiceModel] content_source={self.content_source}  codec={codec_name}  "
            f"speaker_conditioning={self.speaker_conditioning}  "
            f"speaker_encoder_type={self.speaker_encoder_type}"
        )

        if self.content_source == "mimi_semantic":
            # Use Mimi *continuous* encoder output z (pre-quantization) as content.
            # Discrete codebook 0 lookup loses too much information — z is
            # 50× richer (512-dim continuous vs 11-bit discrete at 12.5fps).
            # Encoder is causal, so this stays streaming-compatible.
            sem_dim = int(getattr(codec_model, "latent_dim", config.content_proj_dim))
            self.semantic_proj = nn.Linear(sem_dim, config.content_proj_dim)
            self.semantic_to_hidden = nn.Linear(config.content_proj_dim, config.hidden_dim)
            print(f"[LiveVoiceModel] mimi_semantic = continuous z (latent_dim={sem_dim})")
        elif self.content_source == "streamvoiceanon":
            self.streamvoiceanon_to_hidden = nn.Linear(config.content_proj_dim, config.hidden_dim)
            print(
                "[LiveVoiceModel] streamvoiceanon = causal semantic tokenizer "
                f"(content_dim={config.content_proj_dim})"
            )
        elif self.content_source in ("sw2v", "zipformer", "fastconformer"):
            # Continuous encoder output → content_proj_dim → hidden. Both sources share this
            # path (and keep the sw2v_* parameter names so checkpoints stay compatible);
            # only out_dim differs — sw2v 1024, zipformer 512.
            sw2v_dim = int(getattr(self.content_extractor, "out_dim", 1024))
            # Content path + ASR/GRL heads are built by the SHARED builder so the Stage-1
            # tokenizer (ContentTokenizerModel) and this model have identical modules and
            # parameter names — a Stage-1 checkpoint loads straight in. See
            # model/content_supervision.py.
            build_content_path(self, config, sw2v_dim, log_prefix="[LiveVoiceModel]")

    # --------------------- helpers ---------------------
    _warned_short_align = False

    def align_to_tokens(self, feats: torch.Tensor, num_tokens: int, causal: bool = True) -> torch.Tensor:
        """Align (B, T_src, D) features to `num_tokens` decoder positions.

        Uses the same nearest/causal-average logic as sonic.
        """
        B, T_src, D = feats.shape
        if T_src == num_tokens:
            return feats
        if T_src < num_tokens:
            # A SMALL shortfall means an extractor's analysis window ran off the end of the
            # audio, not a genuine frame-rate difference. Stretching it here silently warps
            # the timeline and makes the feature lag more and more towards the end of the
            # utterance (the CausalMPM path used to lose up to 3 frames / 60 ms this way).
            # Extractors on the codec grid must emit exactly num_tokens frames instead.
            if num_tokens - T_src <= 8 and not LiveVoiceModel._warned_short_align:
                LiveVoiceModel._warned_short_align = True
                warnings.warn(
                    f"align_to_tokens: extractor emitted {T_src} frames for {num_tokens} "
                    f"decoder positions ({num_tokens - T_src} short). This is being "
                    f"linspace-stretched, which drifts the alignment. Pad the extractor to "
                    f"the codec grid instead (see CausalFeatureExtractor._causal_pad).",
                    RuntimeWarning, stacklevel=2,
                )
            idx = torch.linspace(0, T_src - 1, num_tokens, device=feats.device).long()
            return feats[:, idx, :]
        if causal:
            out = []
            ratio = T_src / num_tokens
            for t in range(num_tokens):
                s = max(0, int(t * ratio))
                e = int((t + 1) * ratio)
                out.append(feats[:, s:e, :].mean(dim=1))
            return torch.stack(out, dim=1)
        idx = torch.linspace(0, T_src - 1, num_tokens, device=feats.device).long()
        return feats[:, idx, :]

    def _codebook_delays(self) -> list[int]:
        """Delay of each codebook, in frames. delay(k) = min(k, cap).

        The full MusicGen pattern (cap = K-1) buys hierarchical conditioning: predicting
        codebook k at time t happens only after codebooks < k of that same t are already in
        the input. It costs K-1 frames of algorithmic latency — 140 ms at 8 books / 50 fps,
        the largest single term in this system's budget.

        That conditioning is not worth the same everywhere. RVQ is residual: cb0 carries most
        of the signal and cb1 refines it, while cb6 and cb7 encode low-energy leftovers whose
        dependence on each other is weak. Capping the delay keeps the sequential ordering
        where it pays (the coarse books) and lets the fine tail be predicted in parallel:

            cap 7  0 1 2 3 4 5 6 7   140 ms   full pattern (default, unchanged behaviour)
            cap 4  0 1 2 3 4 4 4 4    80 ms
            cap 3  0 1 2 3 3 3 3 3    60 ms
            cap 0  0 0 0 0 0 0 0 0     0 ms   fully parallel (== use_delay_pattern=False)

        Codebooks sharing a delay are predicted independently of each other, but still
        conditioned on every book with a smaller delay.
        """
        K = self.n_codebooks_predict
        if not bool(getattr(self.config, "use_delay_pattern", True)):
            return [0] * K
        cap = int(getattr(self.config, "delay_cap", K - 1))
        if cap < 0:
            raise ValueError(f"delay_cap must be >= 0, got {cap}")
        return [min(k, cap) for k in range(K)]

    def _delay_tail(self) -> int:
        """Extra stream positions the delay pattern appends (== the largest delay)."""
        return max(self._codebook_delays())

    def _align_content_delay(
        self,
        feats: torch.Tensor,
        base_len: int,
        use_delay: bool,
        null_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Align content/prosody to the codec codebook-0 timeline, then null-pad the delay tail.

        In the MusicGen delay pattern, decoder position p has canonical time p
        (codebook-0's time; codebook k targets time p-k). So content must sit 1:1 on
        the codebook-0 timeline of length ``base_len`` (= T at train, max_steps at
        inference) — NOT be linspace-stretched across the full T+K-1 axis. The stretch
        made content[p] = content_raw[p·(T_src-1)/(T+K-2)], lagging up to K-1 frames by
        the end of the utterance (progressive drift → WER worsens over time). The final
        K-1 tail positions carry no codebook-0 token (only fine-codebook cleanup for
        already-decoded times), so they receive a null content signal.
        """
        body = self.align_to_tokens(feats, base_len, causal=True)
        n_tail = self._delay_tail() if use_delay else 0
        if n_tail <= 0:
            return body
        tail = null_vec.to(device=body.device, dtype=body.dtype).expand(feats.size(0), n_tail, -1)
        return torch.cat([body, tail], dim=1)

    # --------------------- delay pattern ---------------------
    def _build_delay_input(self, codes: torch.Tensor) -> torch.Tensor:
        B, K, T = codes.shape
        d = self._codebook_delays()
        T_delay = T + max(d)
        D = self.config.hidden_dim
        device = codes.device
        emb = torch.zeros(B, T_delay, D, device=device)
        for k in range(K):
            cb = getattr(self, f"codebook_vectors_{k}")
            proj = self.decoder_input_projs[k]
            emb_k = proj(cb[codes[:, k, :]])  # (B, T, D)
            start = d[k] + 1
            end = min(d[k] + 1 + T, T_delay)
            emb[:, start:end, :] += emb_k[:, : end - start, :]
        bos = self.transformer.start_token.expand(B, -1, -1)
        emb[:, 0:1, :] = bos
        return emb

    def _build_delay_targets(self, codes: torch.Tensor) -> torch.Tensor:
        B, K, T = codes.shape
        d = self._codebook_delays()
        tgt = torch.full((B, T + max(d), K), -100, device=codes.device, dtype=codes.dtype)
        for k in range(K):
            tgt[:, d[k] : d[k] + T, k] = codes[:, k, :]
        return tgt

    def _build_nodelay_input(self, codes: torch.Tensor) -> torch.Tensor:
        B, K, T = codes.shape
        D = self.config.hidden_dim
        device = codes.device
        emb = torch.zeros(B, T, D, device=device)
        for k in range(K):
            cb = getattr(self, f"codebook_vectors_{k}")
            proj = self.decoder_input_projs[k]
            emb_k = proj(cb[codes[:, k, :]])
            if T > 1:
                emb[:, 1:, :] += emb_k[:, : T - 1, :]
        bos = self.transformer.start_token.expand(B, -1, -1)
        emb[:, 0:1, :] = bos
        return emb

    def _build_nodelay_targets(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.transpose(1, 2)

    # --------------------- feature extractors ---------------------
    def _embed_codec_prompt(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed reference codec tokens with the SAME embeddings as the AR previous tokens
        (codebook_vectors_* + decoder_input_projs), summed over codebooks. Per-frame, NO
        delay pattern and NO BOS — this is an in-context PROMPT, not a shifted AR input.
        codes: (B, K, T_ref) → (B, T_ref, hidden)."""
        B, K, T = codes.shape
        emb = torch.zeros(B, T, self.config.hidden_dim, device=codes.device)
        for k in range(K):
            cb = getattr(self, f"codebook_vectors_{k}")
            proj = self.decoder_input_projs[k]
            emb = emb + proj(cb[codes[:, k, :]])   # (B, T, hidden)
        return emb

    def _encode_reference_codes(self, reference_audio: torch.Tensor | None) -> torch.Tensor:
        """Reference audio → discrete codec tokens (B, K, T_ref) for the K predicted books."""
        if reference_audio is None:
            raise ValueError("codec_prompt speaker requires reference_audio (discrete codec tokens).")
        with torch.no_grad():
            return self.codec_model.encode(reference_audio)[:, : self.n_codebooks_predict, :]

    def _uses_codec_prompt_continuation(self) -> bool:
        """VALL-E-style: the reference is the first part of the SAME AR stream as the target,
        rather than a separate prefix block. See config.codec_prompt_continuation."""
        return (
            self.speaker_encoder_type == "codec_prompt"
            and self._uses_speaker_prefix()
            and bool(getattr(self.config, "codec_prompt_continuation", False))
        )

    def _prompt_region_content(
        self,
        reference_audio: torch.Tensor | None,
        T_ref: int,
        B: int,
        device: torch.device,
        reference_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(content_add, film) for the T_ref prompt positions of a continuation stream.

        With ``config.codec_prompt_content`` the REFERENCE's own content features are used,
        so the prompt region is a conditioned prediction task exactly like the target region
        — the equivalent of VALL-E/CosyVoice2 concatenating the prompt's transcript with the
        target's. The reference goes through the SAME `extract_content` pipeline as the
        source (including perturbation when training), so both regions look alike to the
        decoder. Without the flag the prompt gets null content, as before.

        No delay tail here: the K-1 tail belongs at the END of the joint stream, which the
        target region's `_align_content_delay` already supplies.
        """
        D = self.config.hidden_dim
        use_ref = bool(getattr(self.config, "codec_prompt_content", False))
        if use_ref and (reference_feats is not None or reference_audio is not None):
            # Prefer cached reference features (training): same normalisation and baked-in
            # perturbation as the target's, and no extra encoder pass. Falls back to the
            # audio at inference, where an arbitrary reference has no cache entry.
            add_raw, film_raw = self.extract_content(reference_audio, reference_feats)
            add = self.align_to_tokens(add_raw, T_ref, causal=True)
            if film_raw is not None:
                return add, self.align_to_tokens(film_raw, T_ref, causal=True)
            return add, torch.zeros(B, T_ref, 0, device=device)
        zeros_add = torch.zeros(B, T_ref, D, device=device)
        null_f = self.null_film_feature.expand(B, T_ref, -1)
        return zeros_add, null_f

    def _set_prefix_len_for_attn(self, prefix_len: int):
        """Report the ACTUAL prefix length to the decoder attentions (ALiBi exemption + KV sink)."""
        if self._uses_codec_prompt_continuation() and not bool(
            getattr(self.config, "codec_prompt_alibi_exempt", True)
        ):
            prefix_len = 0
        self.transformer.set_speaker_prefix_len(prefix_len)

    def encode_speaker_reference(
        self,
        reference_audio: torch.Tensor | None,
        reference_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference audio → per-frame speaker embedding at decoder hidden dim."""
        if self.speaker_encoder_type == "codec_prompt":
            if reference_audio is None:
                raise ValueError("codec_prompt speaker requires reference_audio (discrete codec tokens).")
            K = self.n_codebooks_predict
            with torch.no_grad():
                ref_codes = self.codec_model.encode(reference_audio)[:, :K, :]  # (B, K, T_ref)
            return self._embed_codec_prompt(ref_codes)                          # (B, T_ref, hidden)

        if self.speaker_encoder_type == "speechbrain_ecapa":
            if reference_audio is None:
                raise ValueError("reference_audio is required for SpeechBrain speaker encoder.")
            assert self.speaker_encoder is not None
            with torch.no_grad():
                emb = self.speaker_encoder(reference_audio)  # (B, D_ecapa)
            if self.speaker_conditioning == "prefix" and hasattr(self, "speaker_prefix_proj"):
                B = emb.size(0)
                P = int(getattr(self.config, "speaker_prefix_len", 8))
                spk = self.speaker_prefix_proj(emb).view(B, P, self.config.hidden_dim)
            else:
                spk = self.speaker_proj(emb).unsqueeze(1)
            return spk

        if self.speaker_encoder_type == "spark_global":
            if reference_audio is None:
                raise ValueError("reference_audio is required for Spark global speaker encoder.")
            assert self.speaker_encoder is not None
            with torch.no_grad():
                g = self.speaker_encoder(reference_audio)   # (B, token_num, latent)
            spk = self.spark_speaker_proj(g)                # (B, token_num, hidden)
            if self.speaker_conditioning == "global_avg":
                spk = spk.mean(dim=1, keepdim=True)
            return spk

        if reference_z is None:
            if reference_audio is None:
                raise ValueError("Either reference_audio or reference_z must be provided.")
            with torch.no_grad():
                _, z = self.codec_model.encode_continuous(reference_audio)  # (B, T_enc, D_codec)
        else:
            z = reference_z
        spk = self.speaker_proj(z)  # (B, T_enc, D)

        if self.speaker_conditioning == "global_avg":
            # Pool to a single speaker token
            spk = spk.mean(dim=1, keepdim=True)  # (B, 1, D)
        return spk

    def _uses_speaker_prefix(self) -> bool:
        return self.speaker_conditioning == "prefix"

    def _build_speaker_prefix(self, spk: torch.Tensor) -> torch.Tensor:
        """Prefix = the full speaker token sequence as-is (no pooling/striding).

        - codec encoder: every reference frame (B, T_enc, hidden) becomes a prefix
          token — the whole reference is prepended.
        - speechbrain_ecapa: encode_speaker_reference already expanded the single
          utterance embedding into speaker_prefix_len tokens via speaker_prefix_proj,
          so this passes it through unchanged (old behaviour).
        """
        return spk

    def _prepend_speaker_prefix(
        self,
        spk: torch.Tensor,
        prev_emb: torch.Tensor,
        content_add: torch.Tensor,
        film_feats: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        prefix = self._build_speaker_prefix(spk).to(device=prev_emb.device, dtype=prev_emb.dtype)
        B, P, D = prefix.shape
        zero_content = torch.zeros(B, P, D, device=prev_emb.device, dtype=prev_emb.dtype)
        prev_full = torch.cat([prefix, prev_emb], dim=1)
        content_full = torch.cat([zero_content, content_add], dim=1)
        if film_feats is None:
            film_full = None
        else:
            null_f = self.null_film_feature.expand(B, P, -1).to(
                device=film_feats.device,
                dtype=film_feats.dtype,
            )
            film_full = torch.cat([null_f, film_feats], dim=1)
        return prefix, prev_full, content_full, film_full

    def extract_content(
        self,
        content_audio: torch.Tensor | None,
        content_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (content_add, film_feats_or_None).

        Three input modes:
          content_feats + hubert              → project precomputed HuBERT directly
          content_source=="mimi_semantic"     → perturb (train) → Mimi encoder → continuous z
          content_source=="streamvoiceanon"   → perturb (train) → StreamVoiceAnon tokens
          else (HuBERT path)                  → perturb (train) → HuBERT online
        """
        use_film = str(getattr(self.config, "content_conditioning", "additive")) == "film"

        if content_feats is not None and self.content_source == "hubert":
            # Fast path: precomputed HuBERT features (no perturbation needed — already baked in)
            if use_film:
                film_raw = self.content_extractor.from_precomputed_raw(content_feats)
                zeros_add = torch.zeros(
                    film_raw.shape[0], film_raw.shape[1], self.config.hidden_dim,
                    device=film_raw.device, dtype=film_raw.dtype,
                )
                return zeros_add, film_raw
            return self.content_extractor.from_precomputed(content_feats), None

        if content_feats is not None and self.content_source in ("sw2v", "zipformer", "fastconformer"):
            # Fast path: precomputed sw2v features (perturbation already baked into cache).
            feats = content_feats.to(device=self.sw2v_proj.weight.device,
                                     dtype=self.sw2v_proj.weight.dtype)
            if not bool(getattr(self.config, "content_cmn_in_cache", False)):
                feats = apply_content_cmn(
                    feats,
                    str(getattr(self.config, "content_cmn", "off")),
                    bool(getattr(self.config, "content_cmn_var", False)),
                    prior_frames=float(getattr(self.config, "content_cmn_prior_frames", 0.0)),
                )
            if self.content_refiner is not None:
                feats = self.content_refiner(feats)             # deep causal refiner
            sw2v_emb = self.sw2v_proj(feats)                    # (B, T, content_proj_dim)
            if self.content_fsq is not None:
                sw2v_emb = self.content_fsq(sw2v_emb)           # FSQ bottleneck (STE)
            if use_film:
                zeros_add = torch.zeros(
                    sw2v_emb.shape[0], sw2v_emb.shape[1], self.config.hidden_dim,
                    device=sw2v_emb.device, dtype=sw2v_emb.dtype,
                )
                return zeros_add, sw2v_emb
            return self.sw2v_to_hidden(sw2v_emb), None

        # Apply perturbation to source audio first (used by both paths below)
        if self.use_content_perturbation and self.training:
            # ##################################### 디버깅
            # # ── TEMP DEBUG: listen to online content perturbation (delete after check) ──
            # # File: src/livevoice/model/transformer.py  ≈ extract_content() online path
            # # Saves once to {output_dir}/output/ then hits breakpoint().
            # if not getattr(self, "_dbg_saved_perturb", False):
            #     import os
            #     import torchaudio
            #     out_dir = "/workspace/LiveVoice/src/output"
            #     os.makedirs(out_dir, exist_ok=True)
            #     sr = int(getattr(self.config, "sample_rate", 16000))
            #     before = content_audio.detach().float().cpu()
            #     after = self.content_perturbation(content_audio).detach().float().cpu()
            #     n = min(2, before.size(0))
            #     for i in range(n):
            #         torchaudio.save(
            #             os.path.join(out_dir, f"content_orig_{i}.wav"),
            #             before[i : i + 1], sr,
            #         )
            #         torchaudio.save(
            #             os.path.join(out_dir, f"content_perturbed_{i}.wav"),
            #             after[i : i + 1], sr,
            #         )
            #     print(f"[DEBUG perturb] saved {n} pairs → {out_dir}")
            #     self._dbg_saved_perturb = True
            #     breakpoint()  # remove this whole TEMP DEBUG block after listening
            # # ── END TEMP DEBUG ──
            # #####################################  
            content_audio = self.content_perturbation(content_audio)

        # ── Mimi semantic path: codec encoder → continuous z (pre-quantization) ──
        if self.content_source == "mimi_semantic":
            with torch.no_grad():
                _, z = self.codec_model.encode_continuous(content_audio)  # (B, T, latent_dim)
            sem_emb = self.semantic_proj(z)                              # (B, T, content_proj_dim)
            if use_film:
                zeros_add = torch.zeros(
                    sem_emb.shape[0], sem_emb.shape[1], self.config.hidden_dim,
                    device=sem_emb.device, dtype=sem_emb.dtype,
                )
                return zeros_add, sem_emb
            return self.semantic_to_hidden(sem_emb), None

        # ── StreamVoiceAnon path: causal tokenizer → learned content embedding ──
        if self.content_source == "streamvoiceanon":
            sva_emb = self.content_extractor(content_audio)  # (B, T_codes, content_proj_dim)
            if use_film:
                zeros_add = torch.zeros(
                    sva_emb.shape[0], sva_emb.shape[1], self.config.hidden_dim,
                    device=sva_emb.device, dtype=sva_emb.dtype,
                )
                return zeros_add, sva_emb
            return self.streamvoiceanon_to_hidden(sva_emb), None

        # ── SW2V path: jhcodec AudioEncoder → continuous 1024-d → content_proj_dim ──
        if self.content_source in ("sw2v", "zipformer", "fastconformer"):
            feat = self.content_extractor(content_audio)      # (B, T_frames, out_dim)
            feat = apply_content_cmn(
                feat,
                str(getattr(self.config, "content_cmn", "off")),
                bool(getattr(self.config, "content_cmn_var", False)),
                prior_frames=float(getattr(self.config, "content_cmn_prior_frames", 0.0)),
            )
            if self.content_refiner is not None:
                feat = self.content_refiner(feat)             # deep causal refiner
            sw2v_emb = self.sw2v_proj(feat)                   # (B, T_frames, content_proj_dim)
            if self.content_fsq is not None:
                sw2v_emb = self.content_fsq(sw2v_emb)         # FSQ bottleneck (STE)
            if use_film:
                zeros_add = torch.zeros(
                    sw2v_emb.shape[0], sw2v_emb.shape[1], self.config.hidden_dim,
                    device=sw2v_emb.device, dtype=sw2v_emb.dtype,
                )
                return zeros_add, sw2v_emb
            return self.sw2v_to_hidden(sw2v_emb), None

        # ── HuBERT path (legacy) ───────────────────────────────────────────
        if use_film:
            film_raw = self.content_extractor.forward_raw(content_audio)
            zeros_add = torch.zeros(
                film_raw.shape[0], film_raw.shape[1], self.config.hidden_dim,
                device=film_raw.device, dtype=film_raw.dtype,
            )
            return zeros_add, film_raw
        return self.content_extractor(content_audio), None

    # --------------------- forward ---------------------
    def forward(
        self,
        reference_audio: torch.Tensor,
        content_audio: torch.Tensor,
        target_codes: torch.Tensor,
        prosody_audio: torch.Tensor | None = None,
        content_feats: torch.Tensor | None = None,
        reference_z: torch.Tensor | None = None,
        reference_feats: torch.Tensor | None = None,
        null_speaker: bool = False,
        null_content: bool = False,
    ) -> dict:
        """Teacher-forced training forward.

        Args:
            reference_audio: (B, T_ref) — same speaker, different utterance
            content_audio:   (B, T_ctn) — linguistic source (used if content_feats is None)
            target_codes:    (B, K, T)  — codec codes of the target utterance
            prosody_audio:   (B, T)     — optional prosody source
            content_feats:   (B, T_frames, 768) — precomputed HuBERT features (skips HuBERT)
        """
        if target_codes is None:
            raise ValueError("target_codes required for training forward.")
        K = self.n_codebooks_predict
        B, _, T = target_codes.shape
        device = target_codes.device
        use_delay = bool(getattr(self.config, "use_delay_pattern", True))
        d_cb = self._codebook_delays()
        T_seq = (T + self._delay_tail()) if use_delay else T

        # CFG dropout masks
        drop_both = torch.zeros(B, dtype=torch.bool, device=device)
        drop_spk = torch.zeros(B, dtype=torch.bool, device=device)
        drop_ctn = torch.zeros(B, dtype=torch.bool, device=device)
        drop_pro = torch.zeros(B, dtype=torch.bool, device=device)
        if self.training and bool(getattr(self.config, "use_cfg_dropout", False)):
            drop_both = torch.rand(B, device=device) < float(self.config.cfg_drop_both_p)
            drop_spk = torch.rand(B, device=device) < float(self.config.cfg_drop_speaker_p)
            drop_ctn = torch.rand(B, device=device) < float(self.config.cfg_drop_content_p)
            drop_pro = torch.rand(B, device=device) < float(self.config.cfg_drop_prosody_p)

        # 0. VALL-E continuation: the reference codes are the FIRST part of the SAME AR
        # stream as the target (single BOS at the very front, one delay pattern across
        # both), so the reference tail occupies the prev_emb slots that predict the first
        # target frames. See config.codec_prompt_continuation.
        use_cont = self._uses_codec_prompt_continuation()
        if use_cont:
            prompt_codes = self._encode_reference_codes(reference_audio)
            T_ref = int(prompt_codes.shape[2])
            ar_codes = torch.cat([prompt_codes, target_codes], dim=2)
        else:
            T_ref = 0
            ar_codes = target_codes

        # 1. teacher-forced decoder input + targets
        prev_emb = self._build_delay_input(ar_codes) if use_delay else self._build_nodelay_input(ar_codes)
        targets = self._build_delay_targets(ar_codes) if use_delay else self._build_nodelay_targets(ar_codes)

        # prev-token dropout (reduce leakage). In continuation mode this applies to the
        # TARGET region only — dropping the prompt region would be speaker dropout, which
        # the CFG path below owns.
        p_prev = float(getattr(self.config, "prev_emb_dropout_p", 0.0))
        if self.training and p_prev > 0.0:
            mask = torch.rand(B, T_seq, 1, device=device) < p_prev
            null = self.null_prev_embedding.expand(B, T_seq, -1).to(dtype=prev_emb.dtype)
            if use_cont:
                prev_emb = torch.cat(
                    [prev_emb[:, :T_ref, :], torch.where(mask, null, prev_emb[:, T_ref:, :])],
                    dim=1,
                )
            else:
                prev_emb = torch.where(mask, null, prev_emb)

        # 2. speaker
        # `null_speaker=True` forces the speaker signal off for the whole batch — the
        # ablation debug/diag_prefix_used.py measures. It has to be a forward argument
        # because in continuation mode there is no encode_speaker_reference() call to patch:
        # the speaker lives in the prompt region of the AR stream.
        drop_spk_all = drop_spk | drop_both
        if null_speaker:
            drop_spk_all = torch.ones(B, dtype=torch.bool, device=device)
        if use_cont:
            # The prompt IS the speaker conditioning, so CFG speaker-dropout nulls the
            # prompt region of the stream (keeps null_speaker_embedding meaningful for CFG
            # and for debug/diag_prefix_used.py).
            spk = None
            if drop_spk_all.any():
                prev_emb = prev_emb.clone()
                null_s = self.null_speaker_embedding.expand(B, T_ref, -1).to(dtype=prev_emb.dtype)
                prev_emb[drop_spk_all, :T_ref, :] = null_s[drop_spk_all]
        else:
            spk = self.encode_speaker_reference(reference_audio, reference_z=reference_z)
            if drop_spk_all.any():
                spk = spk.clone()
                null = self.null_speaker_embedding.expand_as(spk).to(dtype=spk.dtype)
                spk[drop_spk_all] = null[drop_spk_all]

        # 3. content (precomputed feats take priority over online HuBERT)
        content_add_raw, film_raw = self.extract_content(content_audio, content_feats)
        # Align content 1:1 to the codebook-0 timeline (length T), null-pad the K-1
        # delay tail. Avoids the linspace-stretch drift (see _align_content_delay).
        content_add = self._align_content_delay(content_add_raw, T, use_delay, self.null_content_embedding)
        film_feats = (
            self._align_content_delay(film_raw, T, use_delay, self.null_film_feature)
            if film_raw is not None else None
        )

        # `null_content=True` forces the content signal off for the whole batch. With the
        # FiLM path zero-initialised and an AR stream that can be continued from prev_emb
        # alone, "the decoder ignores content" is a real failure mode; this measures it the
        # same way null_speaker measures prompt usage.
        drop_ctn_all = drop_ctn | drop_both
        if null_content:
            drop_ctn_all = torch.ones(B, dtype=torch.bool, device=device)
        if drop_ctn_all.any():
            content_add = content_add.clone()
            null = self.null_content_embedding.expand(B, T_seq, -1).to(dtype=content_add.dtype)
            content_add[drop_ctn_all] = null[drop_ctn_all]
            if film_feats is not None:
                film_feats = film_feats.clone()
                null_f = self.null_film_feature.expand(B, T_seq, -1).to(dtype=film_feats.dtype)
                film_feats[drop_ctn_all] = null_f[drop_ctn_all]

        # 4. prosody (additive)
        if self.use_prosody:
            pa = prosody_audio if prosody_audio is not None else content_audio
            prosody_raw = self.prosody_extractor(pa)
            prosody_add = self._align_content_delay(prosody_raw, T, use_delay, self.null_content_embedding)
            drop_pro_all = drop_pro | drop_both
            if drop_pro_all.any():
                prosody_add = prosody_add.clone()
                null = self.null_content_embedding.expand(B, T_seq, -1).to(dtype=prosody_add.dtype)
                prosody_add[drop_pro_all] = null[drop_pro_all]
            content_add = content_add + prosody_add

        # 4b. cepstral (additive, same slot as prosody)
        if self.use_cepstral:
            ca = prosody_audio if prosody_audio is not None else content_audio
            cep_raw = self.cepstral_extractor(ca)
            cep_add = self._align_content_delay(cep_raw, T, use_delay, self.null_content_embedding)
            drop_cep_all = drop_pro | drop_both
            if drop_cep_all.any():
                cep_add = cep_add.clone()
                nullc = self.null_content_embedding.expand(B, T_seq, -1).to(dtype=cep_add.dtype)
                cep_add[drop_cep_all] = nullc[drop_cep_all]
            content_add = content_add + cep_add

        # 4c. MPM (causal masked prosody model). "perlayer" keeps mpm_feats and lets
        # decode_step add it after every layer; "input_conv" folds it into content_add here,
        # which is the same injection point DiffAnon uses, and leaves decode_step untouched.
        mpm_feats = None
        if self.use_mpm:
            ma = prosody_audio if prosody_audio is not None else content_audio
            mpm_raw = self.mpm_extractor(ma)                # (B, T_frames, mpm_dim)
            mpm_feats = self._align_content_delay(mpm_raw, T, use_delay,
                torch.zeros(1, 1, mpm_raw.size(-1), device=device, dtype=mpm_raw.dtype))
            if self.transformer.mpm_norm is not None:
                mpm_feats = self.transformer.mpm_norm(mpm_feats)
            drop_mpm_all = drop_pro | drop_both
            if self.transformer.use_mpm_input:
                mpm_add = self.transformer.mpm_input_proj(mpm_feats)
                # Zero AFTER the projection: dropping the input instead would still leave
                # the conv biases, so the unconditional branch would not be prosody-free.
                if drop_mpm_all.any():
                    mpm_add = mpm_add.clone()
                    mpm_add[drop_mpm_all] = 0.0
                content_add = content_add + mpm_add.to(dtype=content_add.dtype)
                mpm_feats = None
            elif drop_mpm_all.any():
                mpm_feats = mpm_feats.clone()
                mpm_feats[drop_mpm_all] = 0.0

        # 5. transformer
        if use_cont:
            prompt_add, prompt_film = self._prompt_region_content(
                reference_audio, T_ref, B, device, reference_feats=reference_feats)
            content_full = torch.cat([prompt_add.to(dtype=content_add.dtype), content_add], dim=1)
            if film_feats is None:
                film_full = None
            else:
                film_full = torch.cat([prompt_film.to(dtype=film_feats.dtype), film_feats], dim=1)
            self._set_prefix_len_for_attn(T_ref)
            if mpm_feats is not None:
                mpm_full = torch.cat([
                    torch.zeros(B, T_ref, mpm_feats.size(-1), device=device, dtype=mpm_feats.dtype),
                    mpm_feats,
                ], dim=1)
            else:
                mpm_full = None
            decoder_full = self.transformer.decode_step(
                prev_emb,
                content_full,
                film_full,
                encoder_output=None,
                use_cache=False,
                mpm_feats=mpm_full,
            )
            w_prompt = float(getattr(self.config, "codec_prompt_loss_weight", 0.0))
            if w_prompt > 0.0:
                # Score the whole stream (pure-LM style) but down-weight the prompt region:
                # the target region is what we actually want to be good at.
                decoder_output = decoder_full
                pos_w = torch.ones(B, decoder_full.size(1), device=device)
                pos_w[:, :T_ref] = w_prompt
                loss_pos_weights = pos_w
            else:
                decoder_output = decoder_full[:, T_ref:, :]
                targets = targets[:, T_ref:, :]
                loss_pos_weights = None
            encoder_output = prev_emb[:, :T_ref, :]
        elif self._uses_speaker_prefix():
            loss_pos_weights = None
            prefix, prev_full, content_full, film_full = self._prepend_speaker_prefix(
                spk, prev_emb, content_add, film_feats,
            )
            self._set_prefix_len_for_attn(prefix.size(1))
            if mpm_feats is not None:
                mpm_full = torch.cat([
                    torch.zeros(B, prefix.size(1), mpm_feats.size(-1), device=device, dtype=mpm_feats.dtype),
                    mpm_feats,
                ], dim=1)
            else:
                mpm_full = None
            decoder_full = self.transformer.decode_step(
                prev_full,
                content_full,
                film_full,
                encoder_output=None,
                use_cache=False,
                mpm_feats=mpm_full,
            )
            decoder_output = decoder_full[:, prefix.size(1) :, :]
            encoder_output = prefix
        else:
            loss_pos_weights = None
            out = self.transformer(
                spk, content_add, film_feats, prev_emb,
                use_cache=False, mpm_feats=mpm_feats,
            )
            decoder_output = out["decoder_output"]
            encoder_output = out["encoder_output"]

        # 6. per-codebook logits
        all_logits = torch.stack([h(decoder_output) for h in self.codebook_heads], dim=2)
        return {
            "all_logits": all_logits,
            "delayed_targets": targets,
            "loss_pos_weights": loss_pos_weights,
            "decoder_output": decoder_output,
            "encoder_output": encoder_output,
        }

    # --------------------- anonymization (StreamVoiceAnon policy) ---------------------
    @staticmethod
    def apply_noise_mixing(spk: torch.Tensor, alpha: float) -> torch.Tensor:
        """Matched-Gaussian noise mixing (StreamVoiceAnon infer_arvc.apply_noise_mixing).

        alpha=1.0 → no noise (pure VC); lower → stronger anonymization.
        """
        if alpha >= 1.0:
            return spk
        mean, std = spk.mean(), spk.std()
        noise = torch.randn_like(spk) * std + mean
        return alpha * spk + (1.0 - alpha) * noise

    @torch.no_grad()
    def anonymized_speaker(
        self, ref_audios: list[torch.Tensor], alpha: float = 1.0
    ) -> torch.Tensor:
        """Blend K pool references then mix noise → anonymized speaker rep.

        Mirrors StreamVoiceAnon: average the speaker representation over K
        reference utterances (avg collation), then apply alpha noise mixing.
        Each ref_audios[i] is (B, T_i). Returns (B, L, hidden).
        """
        if not ref_audios:
            raise ValueError("anonymized_speaker requires at least one reference.")
        spks = [self.encode_speaker_reference(r) for r in ref_audios]
        if len(spks) > 1:
            # speaker token counts can differ across refs (codec path); align all
            # to the shortest before averaging so blending is well-defined.
            L = min(s.size(1) for s in spks)
            spks = [
                s if s.size(1) == L else self.align_to_tokens(s, L, causal=False)
                for s in spks
            ]
        spk = torch.stack(spks, dim=0).mean(dim=0)
        return self.apply_noise_mixing(spk, alpha)

    @torch.no_grad()
    def generate_anonymized(
        self,
        content_audio: torch.Tensor,
        ref_audios: list[torch.Tensor],
        alpha: float = 1.0,
        **gen_kwargs,
    ) -> torch.Tensor:
        """VC with an anonymized speaker rep (K-pool blend + alpha noise)."""
        spk = self.anonymized_speaker(ref_audios, alpha)
        return self.generate(
            reference_audio=None, content_audio=content_audio,
            speaker_override=spk, **gen_kwargs,
        )

    # --------------------- generation ---------------------
    @torch.no_grad()
    def set_ar_inference_dtype(self, dtype: torch.dtype) -> None:
        """Cast only the autoregressive decoder path for mixed-precision inference.

        The waveform codec and Zipformer remain fp32.  This avoids autocast's
        repeated weight conversion on the many B=1 per-token GEMVs.
        """
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported AR inference dtype: {dtype}")
        self.transformer.to(dtype=dtype)
        self.decoder_input_projs.to(dtype=dtype)
        self.codebook_heads.to(dtype=dtype)
        for k in range(self.n_codebooks_predict):
            name = f"codebook_vectors_{k}"
            setattr(self, name, getattr(self, name).to(dtype=dtype))
        for name in (
            "null_speaker_embedding", "null_content_embedding", "null_prev_embedding",
            "null_film_feature",
        ):
            value = getattr(self, name, None)
            if value is not None:
                value.data = value.data.to(dtype=dtype)
        self._inference_head_pack = None

    def set_fast_inference(self, enabled: bool = True) -> None:
        """Enable inference-only operator packing on the AR decoder."""
        for layer in self.transformer.decoder_layers:
            layer.self_attn.fuse_qkv_inference = bool(enabled)
            layer.self_attn._inference_qkv_pack = None

    def _packed_codebook_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Evaluate all codebook heads with one GEMM for greedy inference.

        The regular ModuleList path is retained for training and stochastic/CFG
        decoding.  The packed tensors are rebuilt after any parameter update and
        are deliberately not persistent checkpoint state.
        """
        versions = tuple((h.weight._version, h.bias._version) for h in self.codebook_heads)
        cache = getattr(self, "_inference_head_pack", None)
        if cache is None or cache[0] != versions or cache[1].device != hidden.device:
            weight = torch.cat([h.weight.detach() for h in self.codebook_heads], dim=0)
            bias = torch.cat([h.bias.detach() for h in self.codebook_heads], dim=0)
            self._inference_head_pack = (versions, weight, bias)
        _, weight, bias = self._inference_head_pack
        logits = F.linear(hidden, weight, bias)
        return logits.view(hidden.shape[0], self.n_codebooks_predict, -1)

    @torch.no_grad()
    def _projected_codebook_tables(self) -> torch.Tensor:
        """Cache proj_k(codebook_k) so token feedback becomes gather + sum.

        This removes K small matrix-vector products from every AR step. Biases
        are included in each table, exactly matching the original projections.
        """
        versions = tuple(
            (p.weight._version, p.bias._version,
             getattr(self, f"codebook_vectors_{k}")._version)
            for k, p in enumerate(self.decoder_input_projs)
        )
        cache = getattr(self, "_inference_projected_codebooks", None)
        device = self.decoder_input_projs[0].weight.device
        if cache is None or cache[0] != versions or cache[1].device != device:
            tables = torch.stack([
                proj(getattr(self, f"codebook_vectors_{k}"))
                for k, proj in enumerate(self.decoder_input_projs)
            ])
            self._inference_projected_codebooks = (versions, tables)
        return self._inference_projected_codebooks[1]

    @torch.no_grad()
    def generate(
        self,
        reference_audio: torch.Tensor,
        content_audio: torch.Tensor,
        prosody_audio: torch.Tensor | None = None,
        max_steps: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
        cfg_scale: float = 1.0,
        prosody_cfg_scale: float = 1.0,
        speaker_override: torch.Tensor | None = None,
        show_progress: bool = True,
        fuse_codebook_heads: bool = False,
        use_cuda_graph: bool = False,
        content_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """If `speaker_override` (B, *, hidden) is given it replaces the speaker
        representation from `reference_audio` (used by anonymized generation).

        `prosody_cfg_scale` controls prosody guidance independently of speaker
        guidance (`cfg_scale`).  When both are != 1.0 the final logit is composed
        as::

            lg = lg_null_both
                 + cfg_scale         * (lg_null_pro - lg_null_both)
                 + prosody_cfg_scale * (lg_null_spk - lg_null_both)
        """
        use_delay = bool(getattr(self.config, "use_delay_pattern", True))
        d_cb = self._codebook_delays()
        K = self.n_codebooks_predict
        device = content_audio.device
        B = content_audio.shape[0]
        use_prefix = self._uses_speaker_prefix()
        if use_prefix and not use_cache:
            raise RuntimeError("speaker_conditioning='prefix' generation requires use_cache=True.")

        use_cont = self._uses_codec_prompt_continuation()
        if use_cont:
            if speaker_override is not None:
                raise ValueError(
                    "speaker_override is incompatible with codec_prompt continuation: the "
                    "speaker is carried by reference codec TOKENS inside the AR stream, not "
                    "by a speaker embedding. Set config.codec_prompt_continuation=False."
                )
            prompt_codes = self._encode_reference_codes(reference_audio)
            T_ref = int(prompt_codes.shape[2])
            spk = None
            spk_prefix = None
            spk_enc = None
            self._set_prefix_len_for_attn(T_ref)
        else:
            prompt_codes = None
            T_ref = 0
            spk = (
                speaker_override
                if speaker_override is not None
                else self.encode_speaker_reference(reference_audio)
            )
            if use_prefix:
                spk_prefix = self._build_speaker_prefix(spk)
                spk_enc = None
                self._set_prefix_len_for_attn(spk_prefix.size(1))
            else:
                spk_prefix = None
                spk_enc = self.transformer.encode_speaker(spk)

        content_add_raw, film_raw = self.extract_content(content_audio, content_feats)
        if max_steps is None:
            # Use codec frame count as generation horizon.
            # HuBERT/content frames (e.g. 50 fps) can differ from codec frames
            # (e.g. Mimi 12.5 fps), so using T_dec directly may over-generate.
            if not hasattr(self.codec_model, "hop_length"):
                raise AttributeError(
                    f"{type(self.codec_model).__name__} must set hop_length "
                    "(samples per codec frame); no fallback."
                )
            hop = int(self.codec_model.hop_length)
            if hop <= 0:
                raise ValueError(f"codec hop_length must be positive, got {hop}")
            n_samples = int(content_audio.shape[-1])
            # ceil (not round) to match the codec's own tokenization (end-padded blocks →
            # ceil(L/hop)) and the center-aligned HuBERT content count, so content aligns
            # 1:1 to the generated tokens with no resample.
            max_steps = max(1, -(-n_samples // hop))
        T_total = (max_steps + self._delay_tail()) if use_delay else max_steps

        # Align content 1:1 to the codebook-0 timeline (length max_steps) and null-pad
        # the K-1 delay tail — identical to the training forward. causal=True keeps the
        # source→token mapping causal (no future leakage). See _align_content_delay:
        # the old linspace-to-T_total stretch drifted content up to K-1 frames behind
        # the token it conditions, worsening WER over the course of an utterance.
        content_add = self._align_content_delay(content_add_raw, max_steps, use_delay, self.null_content_embedding)
        film_feats = (
            self._align_content_delay(film_raw, max_steps, use_delay, self.null_film_feature)
            if film_raw is not None else None
        )
        # Content extraction may intentionally remain fp32 while the latency-
        # critical AR stack uses static fp16/bf16 weights.
        ar_dtype = self.transformer.start_token.dtype
        content_add = content_add.to(dtype=ar_dtype)
        if film_feats is not None:
            film_feats = film_feats.to(dtype=ar_dtype)

        use_pro_cfg = prosody_cfg_scale != 1.0
        if use_pro_cfg:
            content_add_null_pro = content_add.clone()

        if self.use_prosody:
            pa = prosody_audio if prosody_audio is not None else content_audio
            prosody_raw = self.prosody_extractor(pa)
            prosody_add = self._align_content_delay(prosody_raw, max_steps, use_delay, self.null_content_embedding)
            content_add = content_add + prosody_add
        if self.use_cepstral:
            ca = prosody_audio if prosody_audio is not None else content_audio
            cep_raw = self.cepstral_extractor(ca)
            content_add = content_add + self._align_content_delay(
                cep_raw, max_steps, use_delay, self.null_content_embedding)
        mpm_feats = None
        mpm_feats_null_pro = None
        if self.use_mpm:
            ma = prosody_audio if prosody_audio is not None else content_audio
            mpm_raw = self.mpm_extractor(ma)
            mpm_feats = self._align_content_delay(mpm_raw, max_steps, use_delay,
                torch.zeros(1, 1, mpm_raw.size(-1), device=device, dtype=mpm_raw.dtype))
            if self.transformer.mpm_norm is not None:
                mpm_feats = self.transformer.mpm_norm(mpm_feats)
            if self.transformer.use_mpm_input:
                content_add = content_add + self.transformer.mpm_input_proj(
                    mpm_feats).to(dtype=content_add.dtype)
                mpm_feats = None
            elif use_pro_cfg:
                mpm_feats_null_pro = torch.zeros_like(mpm_feats)

        generated = [[] for _ in range(K)]
        prev_emb = self.transformer.start_token.expand(B, -1, -1)
        if use_cache:
            self.transformer.reset_caches()

        use_cfg = cfg_scale != 1.0
        if use_cfg:
            null_caches = self.transformer.init_caches(B, device)
            if use_cont:
                null_prefix = self.null_speaker_embedding.expand(B, T_ref, -1).to(
                    device=device, dtype=prev_emb.dtype,
                )
                null_spk_enc = None
            else:
                null_spk = self.null_speaker_embedding.expand_as(spk).to(device=device, dtype=spk.dtype)
                if use_prefix:
                    null_prefix = self._build_speaker_prefix(null_spk).to(
                        device=device,
                        dtype=prev_emb.dtype,
                    )
                    null_spk_enc = None
                else:
                    null_prefix = None
                    null_spk_enc = self.transformer.encode_speaker(null_spk)
        else:
            null_caches = None
            null_prefix = None
            null_spk_enc = None

        if use_pro_cfg:
            pro_null_caches = self.transformer.init_caches(B, device)
            pro_null_spk_enc = spk_enc

        # One [K*V, D] GEMM is substantially more efficient than K tiny [V, D]
        # GEMVs at B=1. It is valid only where each head is decoded greedily and
        # no CFG branch needs separate logits.
        use_packed_greedy = bool(fuse_codebook_heads) and not use_cfg and not use_pro_cfg \
            and (temperature is None or temperature <= 0 or top_k == 1)
        projected_codebooks = self._projected_codebook_tables() if use_packed_greedy else None
        codebook_ids = (
            torch.arange(K, device=device).unsqueeze(0)
            if projected_codebooks is not None else None
        )

        if use_cont:
            # Prime the cache with the reference part of the joint AR stream (BOS + delayed
            # reference codes). Then the first generated step's AR input is position T_ref of
            # that same stream — the delay window sitting entirely on the reference tail,
            # exactly what the training forward feeds at the boundary.
            prime_full = self._build_delay_input(prompt_codes).to(dtype=prev_emb.dtype)
            prefix = prime_full[:, :T_ref, :]
            prev_emb = prime_full[:, T_ref : T_ref + 1, :]
        elif use_prefix:
            assert spk_prefix is not None
            prefix = spk_prefix.to(device=device, dtype=prev_emb.dtype)

        if use_cont or use_prefix:
            if use_cont:
                # Must match the training stream: the prompt region carries the reference's
                # own content when config.codec_prompt_content is on, null content otherwise.
                p_add, p_film = self._prompt_region_content(
                    reference_audio, prefix.size(1), B, device)
                prefix_content = p_add.to(device=device, dtype=prev_emb.dtype)
                prefix_film = (
                    None if film_feats is None
                    else p_film.to(device=device, dtype=film_feats.dtype)
                )
            else:
                prefix_content = torch.zeros(
                    B,
                    prefix.size(1),
                    self.config.hidden_dim,
                    device=device,
                    dtype=prev_emb.dtype,
                )
                prefix_film = (
                    None if film_feats is None
                    else self.null_film_feature.expand(B, prefix.size(1), -1).to(
                        device=device, dtype=film_feats.dtype)
                )
            self.transformer.decode_step(
                prefix,
                prefix_content,
                prefix_film,
                encoder_output=None,
                use_cache=False,
                mpm_feats=None,
            )
            if use_cfg:
                assert null_caches is not None and null_prefix is not None
                null_prefix_content = torch.zeros_like(prefix_content)
                null_prefix_film = None if prefix_film is None else prefix_film.clone()
                _, null_caches = self.transformer.decode_step_stateless(
                    null_prefix,
                    null_prefix_content,
                    null_prefix_film,
                    encoder_output=None,
                    caches=null_caches,
                    use_cache=False,
                    mpm_feats=None,
                )
            if use_pro_cfg:
                _, pro_null_caches = self.transformer.decode_step_stateless(
                    prefix,
                    prefix_content,
                    prefix_film,
                    encoder_output=None,
                    caches=pro_null_caches,
                    use_cache=False,
                    mpm_feats=None,
                )

        decode_graph = None
        graph_supported = (
            bool(use_cuda_graph)
            and B == 1
            and (use_prefix or use_cont)
            and spk_enc is None
            and not use_cfg
            and not use_pro_cfg
            and not self.transformer.use_mpm_perlayer
            and (T_ref + T_total) < int(self.config.max_seq_len)
        )
        if graph_supported:
            graph_key = (
                T_ref, prev_emb.dtype, prev_emb.device,
                None if film_feats is None else film_feats.shape[-1],
            )
            cached = getattr(self.transformer, "_decode_cudagraph_cache", None)
            if cached is None or cached[0] != graph_key:
                decode_graph = self.transformer.make_decode_cudagraph(
                    prev_emb, content_add[:, :1, :],
                    film_feats[:, :1, :] if film_feats is not None else None,
                )
                self.transformer._decode_cudagraph_cache = (graph_key, decode_graph)
            else:
                decode_graph = cached[1]
                decode_graph.refresh_prefix()

        for t in tqdm(
            range(T_total), desc="VC generating", leave=False,
            disable=not show_progress,
        ):
            c_t = content_add[:, t : t + 1, :]
            f_t = film_feats[:, t : t + 1, :] if film_feats is not None else None
            m_t = mpm_feats[:, t : t + 1, :] if mpm_feats is not None else None
            step_use_cache = use_cache and (t > 0 or use_prefix or use_cont)

            if decode_graph is not None:
                dec_out = decode_graph.step(prev_emb, c_t, f_t)
            else:
                dec_out = self.transformer.decode_step(
                    prev_emb, c_t, f_t, spk_enc, use_cache=step_use_cache,
                    mpm_feats=m_t,
                )
            hidden_full = dec_out[:, -1, :]

            packed_codes = None
            if use_packed_greedy:
                packed_codes = torch.argmax(
                    self._packed_codebook_logits(hidden_full), dim=-1)

            if use_cfg:
                assert null_caches is not None
                null_out, null_caches = self.transformer.decode_step_stateless(
                    prev_emb, c_t, f_t, null_spk_enc, null_caches,
                    use_cache=step_use_cache,
                    mpm_feats=m_t,
                )
                hidden_null_spk = null_out[:, -1, :]

            if use_pro_cfg:
                c_t_np = content_add_null_pro[:, t : t + 1, :]
                m_t_np = mpm_feats_null_pro[:, t : t + 1, :] if mpm_feats_null_pro is not None else m_t
                pro_null_out, pro_null_caches = self.transformer.decode_step_stateless(
                    prev_emb, c_t_np, f_t, pro_null_spk_enc, pro_null_caches,
                    use_cache=step_use_cache,
                    mpm_feats=m_t_np,
                )
                hidden_null_pro = pro_null_out[:, -1, :]

            if use_delay:
                for k in range(K):
                    orig = t - d_cb[k]
                    if 0 <= orig < max_steps:
                        if packed_codes is not None:
                            code_k = packed_codes[:, k]
                        else:
                            lg = self.codebook_heads[k](hidden_full)
                            if use_cfg and use_pro_cfg:
                                lg_ns = self.codebook_heads[k](hidden_null_spk)
                                lg_np = self.codebook_heads[k](hidden_null_pro)
                                lg = lg + (cfg_scale - 1.0) * (lg - lg_ns) + (prosody_cfg_scale - 1.0) * (lg - lg_np)
                            elif use_cfg:
                                lg_n = self.codebook_heads[k](hidden_null_spk)
                                lg = lg_n + cfg_scale * (lg - lg_n)
                            elif use_pro_cfg:
                                lg_np = self.codebook_heads[k](hidden_null_pro)
                                lg = lg_np + prosody_cfg_scale * (lg - lg_np)
                            code_k = _sample(lg, temperature, top_k, top_p)
                        generated[k].append(code_k)
                # next prev_emb
                # In the long body all K feedback tokens exist. Use one table
                # gather and reduction instead of K decoder-input GEMVs.
                if projected_codebooks is not None and max(d_cb) <= t < max_steps:
                    feedback_idx = torch.stack(
                        [generated[k][t - d_cb[k]] for k in range(K)], dim=1)
                    nxt = projected_codebooks[codebook_ids, feedback_idx].sum(dim=1, keepdim=True)
                else:
                    nxt = torch.zeros(
                        B, 1, self.config.hidden_dim, device=device, dtype=prev_emb.dtype)
                    for k in range(K):
                        orig = t - d_cb[k]
                        cb = getattr(self, f"codebook_vectors_{k}")
                        if 0 <= orig < len(generated[k]):
                            nxt += self.decoder_input_projs[k](cb[generated[k][orig]]).unsqueeze(1)
                        elif use_cont and orig < 0 and T_ref + orig >= 0:
                            nxt += self.decoder_input_projs[k](
                                cb[prompt_codes[:, k, T_ref + orig]]
                            ).unsqueeze(1)
                prev_emb = nxt
            else:
                for k in range(K):
                    if packed_codes is not None:
                        generated[k].append(packed_codes[:, k])
                        continue
                    lg = self.codebook_heads[k](hidden_full)
                    if use_cfg and use_pro_cfg:
                        lg_ns = self.codebook_heads[k](hidden_null_spk)
                        lg_np = self.codebook_heads[k](hidden_null_pro)
                        lg = lg + (cfg_scale - 1.0) * (lg - lg_ns) + (prosody_cfg_scale - 1.0) * (lg - lg_np)
                    elif use_cfg:
                        lg_n = self.codebook_heads[k](hidden_null_spk)
                        lg = lg_n + cfg_scale * (lg - lg_n)
                    elif use_pro_cfg:
                        lg_np = self.codebook_heads[k](hidden_null_pro)
                        lg = lg_np + prosody_cfg_scale * (lg - lg_np)
                    generated[k].append(_sample(lg, temperature, top_k, top_p))
                if projected_codebooks is not None:
                    feedback_idx = torch.stack([generated[k][-1] for k in range(K)], dim=1)
                    nxt = projected_codebooks[codebook_ids, feedback_idx].sum(dim=1, keepdim=True)
                else:
                    nxt = torch.zeros(
                        B, 1, self.config.hidden_dim, device=device, dtype=prev_emb.dtype)
                    for k in range(K):
                        cb = getattr(self, f"codebook_vectors_{k}")
                        nxt += self.decoder_input_projs[k](cb[generated[k][-1]]).unsqueeze(1)
                prev_emb = nxt

        return torch.stack([torch.stack(generated[k], dim=1) for k in range(K)], dim=1)

    def decode_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.codec_model.decode(codes)


def _sample(logits, temperature, top_k, top_p):
    # top_k=1 has exactly the same distribution as greedy decoding. Handle it
    # before the diagnostic reductions: the old path synchronized CUDA on
    # isfinite(...).all() several times for every codebook and every frame,
    # then ran topk + softmax + multinomial to sample a one-point distribution.
    if temperature is None or temperature <= 0 or top_k == 1:
        return torch.argmax(logits, dim=-1)
    # Guard against a diverged model producing NaN/Inf logits: torch.multinomial on
    # non-finite/negative probs triggers a CUDA device-side assert that corrupts the whole
    # context and crashes the run (surfacing later as a misleading CUBLAS_INTERNAL_ERROR).
    # Sanitize logits and fall back to argmax if the distribution is still invalid, so a bad
    # validation sample degrades to noise instead of killing training.
    if not torch.isfinite(logits).all():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    lg = _top_k_top_p(logits, top_k, top_p)
    probs = F.softmax(lg / temperature, dim=-1)
    if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
        return torch.argmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def _top_k_top_p(logits, top_k=None, top_p=None, min_keep: int = 1):
    if (top_k is None or top_k <= 0) and (top_p is None or top_p <= 0 or top_p >= 1.0):
        return logits
    out = logits.clone()
    V = out.size(-1)
    if top_k is not None and top_k > 0:
        k = min(int(top_k), V)
        kth = torch.topk(out, k, dim=-1).values[..., -1, None]
        out = torch.where(out < kth, torch.full_like(out, -float("inf")), out)
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sl, si = torch.sort(out, descending=True, dim=-1)
        sp = F.softmax(sl, dim=-1)
        cp = torch.cumsum(sp, dim=-1)
        rm = cp > float(top_p)
        rm[..., 1:] = rm[..., :-1].clone()
        rm[..., 0] = False
        if min_keep > 1:
            rm[..., :min_keep] = False
        sl = sl.masked_fill(rm, -float("inf"))
        out = torch.full_like(out, -float("inf"))
        out.scatter_(-1, si, sl)
    return out
