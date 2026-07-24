"""LiveVoice transformer: streaming voice conversion.

Mirrors sonic's SketchTransformer / SketchModel but rewired for speech:
- Reference audio → codec continuous z → cross-attention or decoder prefix (speaker / timbre)
- Content audio   → HuBERT / Mimi z / StreamVoiceAnon tokens → additive or FiLM conditioning
- Optional prosody (F0 + loudness)   → additive (only if config.use_prosody)
- AR decoder over codec codebooks with MusicGen delay pattern
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from livevoice.nn.layer import TransformerBlock, init_kv_cache_ext
from livevoice.model.content_perturbation import ContentPerturbation
from livevoice.model.fsq import FSQBottleneck
from livevoice.model.speechbrain_speaker_encoder import SpeechBrainECAPASpeakerEncoder
from livevoice.model.spark_speaker_encoder import SparkGlobalSpeakerEncoder
from livevoice.model.asr_supervision import AsrSupervisionHead

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


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
    ) -> torch.Tensor:
        if self.use_film:
            if film_feats is None:
                raise RuntimeError("content_conditioning='film' requires film_feats.")
            x = decoder_input
        else:
            x = decoder_input + content_add
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, mem=encoder_output, use_cache=use_cache)
            if self.use_film:
                gb = self.film_mlps[i](film_feats)
                gamma = 1.0 + gb[..., : self.hidden_dim]
                beta = gb[..., self.hidden_dim :]
                x = x * gamma + beta
        return self.decoder_norm(x)

    def decode_step_stateless(
        self,
        decoder_input: torch.Tensor,
        content_add: torch.Tensor,
        film_feats: torch.Tensor | None,
        encoder_output: torch.Tensor | None,
        caches: list,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list]:
        if self.use_film:
            if film_feats is None:
                raise RuntimeError("content_conditioning='film' requires film_feats.")
            x = decoder_input
        else:
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
            new_caches.append(kv_out)
        return self.decoder_norm(x), new_caches

    def forward(
        self,
        speaker_embedding: torch.Tensor,
        content_add: torch.Tensor,
        film_feats: torch.Tensor | None,
        prev_embeddings: torch.Tensor,
        use_cache: bool = False,
    ) -> dict:
        encoder_output = self.encode_speaker(speaker_embedding)
        if prev_embeddings is None:
            raise ValueError("prev_embeddings is required (training) — use generate() for inference.")
        decoder_output = self.decode_step(
            prev_embeddings, content_add, film_feats, encoder_output, use_cache,
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

    def reset_caches(self):
        for layer in self.decoder_layers:
            if hasattr(layer.self_attn, "_cache_inited"):
                layer.self_attn._cache_inited = False
            if hasattr(layer, "cross_attn") and hasattr(layer.cross_attn, "_cache_inited"):
                layer.cross_attn._cache_inited = False


# =====================================================================
# Full model: codec + content + prosody + transformer
# =====================================================================
class LiveVoiceModel(nn.Module):
    """Full VC pipeline.

    Components:
      - codec_model:        frozen codec (16 kHz or 24 kHz speech)
      - content_extractor: HuBERT or StreamVoiceAnon content features
      - prosody_extractor: (optional) F0 + loudness
      - transformer:      LiveVoiceTransformer
    """

    def __init__(self, config, codec_model, content_extractor, prosody_extractor=None):
        super().__init__()
        self.config = config
        self.codec_model = codec_model
        self.content_extractor = content_extractor
        self.prosody_extractor = prosody_extractor
        self.use_prosody = bool(getattr(config, "use_prosody", False)) and (prosody_extractor is not None)

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
        if self.speaker_encoder_type not in {"codec", "speechbrain_ecapa", "spark_global"}:
            raise ValueError(
                "speaker_encoder_type must be one of: codec, speechbrain_ecapa, spark_global; "
                f"got {self.speaker_encoder_type!r}"
            )
        if self.speaker_encoder_type == "speechbrain_ecapa":
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
        valid_content_sources = {"hubert", "mimi_semantic", "streamvoiceanon", "sw2v"}
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
        if self.content_source == "sw2v" and self.content_extractor is None:
            raise ValueError(
                "content_source='sw2v' but content_extractor was not provided. "
                "Pass Sw2vContentEncoder."
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
        elif self.content_source == "sw2v":
            # jhcodec streaming-wav2vec: continuous out_dim → content_proj_dim → hidden.
            sw2v_dim = int(getattr(self.content_extractor, "out_dim", 1024))
            self.sw2v_proj = nn.Linear(sw2v_dim, config.content_proj_dim)
            self.sw2v_to_hidden = nn.Linear(config.content_proj_dim, config.hidden_dim)
            print(f"[LiveVoiceModel] sw2v = jhcodec AudioEncoder (out_dim={sw2v_dim} → "
                  f"content_proj_dim={config.content_proj_dim})")
            # Optional FSQ information bottleneck after sw2v_proj (see config.py).
            if bool(getattr(config, "use_content_fsq", False)):
                self.content_fsq = FSQBottleneck(
                    config.content_proj_dim, tuple(getattr(config, "fsq_levels", (8, 5, 5, 5)))
                )
                print(f"[LiveVoiceModel] content FSQ bottleneck: levels={self.content_fsq.levels} "
                      f"→ codebook={self.content_fsq.codebook_size}")
            else:
                self.content_fsq = None

            # Optional seq2seq ASR (StyleStream-style) supervision on this content
            # embedding: hangs off sw2v_proj[+content_fsq] output only, never touches
            # the main decoder, discarded at inference (see asr_supervision.py).
            self.use_asr_supervision = bool(getattr(config, "use_asr_supervision", False))
            if self.use_asr_supervision:
                self.asr_head = AsrSupervisionHead(config)
            else:
                self.asr_head = None

    # --------------------- helpers ---------------------
    def align_to_tokens(self, feats: torch.Tensor, num_tokens: int, causal: bool = True) -> torch.Tensor:
        """Align (B, T_src, D) features to `num_tokens` decoder positions.

        Uses the same nearest/causal-average logic as sonic.
        """
        B, T_src, D = feats.shape
        if T_src == num_tokens:
            return feats
        if T_src < num_tokens:
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
        K = self.n_codebooks_predict
        if not use_delay or K <= 1:
            return body
        tail = null_vec.to(device=body.device, dtype=body.dtype).expand(feats.size(0), K - 1, -1)
        return torch.cat([body, tail], dim=1)

    # --------------------- delay pattern ---------------------
    def _build_delay_input(self, codes: torch.Tensor) -> torch.Tensor:
        B, K, T = codes.shape
        T_delay = T + K - 1
        D = self.config.hidden_dim
        device = codes.device
        emb = torch.zeros(B, T_delay, D, device=device)
        for k in range(K):
            cb = getattr(self, f"codebook_vectors_{k}")
            proj = self.decoder_input_projs[k]
            emb_k = proj(cb[codes[:, k, :]])  # (B, T, D)
            start = k + 1
            end = min(k + 1 + T, T_delay)
            emb[:, start:end, :] += emb_k[:, : end - start, :]
        bos = self.transformer.start_token.expand(B, -1, -1)
        emb[:, 0:1, :] = bos
        return emb

    def _build_delay_targets(self, codes: torch.Tensor) -> torch.Tensor:
        B, K, T = codes.shape
        T_delay = T + K - 1
        tgt = torch.full((B, T_delay, K), -100, device=codes.device, dtype=codes.dtype)
        for k in range(K):
            tgt[:, k : k + T, k] = codes[:, k, :]
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
    def encode_speaker_reference(
        self,
        reference_audio: torch.Tensor | None,
        reference_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference audio → per-frame speaker embedding at decoder hidden dim."""
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

        if content_feats is not None and self.content_source == "sw2v":
            # Fast path: precomputed sw2v features (perturbation already baked into cache).
            feats = content_feats.to(device=self.sw2v_proj.weight.device,
                                     dtype=self.sw2v_proj.weight.dtype)
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
        if self.content_source == "sw2v":
            feat = self.content_extractor(content_audio)      # (B, T_frames, out_dim)
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

    def compute_asr_supervision_loss(
        self,
        content_feats_full: torch.Tensor,
        content_feats_full_len: torch.Tensor,
        phoneme_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Seq2seq ASR loss on the FULL (un-cropped) sw2v content embedding.

        Runs a SEPARATE forward through sw2v_proj[+content_fsq] on the full-utterance
        cached features (not the training window) — text labels are utterance-level with
        no timestamps, so this only makes sense on the whole utterance. sw2v_to_hidden is
        reused (not duplicated) so the loss shapes the exact representation the main
        decoder consumes elsewhere.

        content_feats_full:     (B, T_full, sw2v_dim) zero-padded
        content_feats_full_len: (B,) true (unpadded) frame counts
        phoneme_ids:             (B, T_ph) = [BOS, ph..., EOS, PAD...]
        """
        if self.asr_head is None:
            return torch.tensor(0.0, device=content_feats_full.device)
        feats = content_feats_full.to(
            device=self.sw2v_proj.weight.device, dtype=self.sw2v_proj.weight.dtype
        )
        sw2v_emb = self.sw2v_proj(feats)                # (B, T_full, content_proj_dim)
        if self.content_fsq is not None:
            sw2v_emb = self.content_fsq(sw2v_emb)        # same FSQ bottleneck as the VC path
        memory = self.sw2v_to_hidden(sw2v_emb)            # (B, T_full, hidden_dim)

        T_full = memory.size(1)
        arange = torch.arange(T_full, device=memory.device).unsqueeze(0)  # (1, T_full)
        memory_key_padding_mask = arange >= content_feats_full_len.unsqueeze(1)  # True = pad

        return self.asr_head.compute_loss(memory, memory_key_padding_mask, phoneme_ids)

    # --------------------- forward ---------------------
    def forward(
        self,
        reference_audio: torch.Tensor,
        content_audio: torch.Tensor,
        target_codes: torch.Tensor,
        prosody_audio: torch.Tensor | None = None,
        content_feats: torch.Tensor | None = None,
        reference_z: torch.Tensor | None = None,
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
        T_seq = (T + K - 1) if use_delay else T

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

        # 1. teacher-forced decoder input + targets
        prev_emb = self._build_delay_input(target_codes) if use_delay else self._build_nodelay_input(target_codes)
        targets = self._build_delay_targets(target_codes) if use_delay else self._build_nodelay_targets(target_codes)

        # prev-token dropout (reduce leakage)
        p_prev = float(getattr(self.config, "prev_emb_dropout_p", 0.0))
        if self.training and p_prev > 0.0:
            mask = torch.rand(B, T_seq, 1, device=device) < p_prev
            null = self.null_prev_embedding.expand(B, T_seq, -1).to(dtype=prev_emb.dtype)
            prev_emb = torch.where(mask, null, prev_emb)

        # 2. speaker
        spk = self.encode_speaker_reference(reference_audio, reference_z=reference_z)
        drop_spk_all = drop_spk | drop_both
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

        drop_ctn_all = drop_ctn | drop_both
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

        # 5. transformer
        if self._uses_speaker_prefix():
            prefix, prev_full, content_full, film_full = self._prepend_speaker_prefix(
                spk, prev_emb, content_add, film_feats,
            )
            decoder_full = self.transformer.decode_step(
                prev_full,
                content_full,
                film_full,
                encoder_output=None,
                use_cache=False,
            )
            decoder_output = decoder_full[:, prefix.size(1) :, :]
            encoder_output = prefix
        else:
            out = self.transformer(
                spk, content_add, film_feats, prev_emb,
                use_cache=False,
            )
            decoder_output = out["decoder_output"]
            encoder_output = out["encoder_output"]

        # 6. per-codebook logits
        all_logits = torch.stack([h(decoder_output) for h in self.codebook_heads], dim=2)
        return {
            "all_logits": all_logits,
            "delayed_targets": targets,
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
        speaker_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """If `speaker_override` (B, *, hidden) is given it replaces the speaker
        representation from `reference_audio` (used by anonymized generation)."""
        use_delay = bool(getattr(self.config, "use_delay_pattern", True))
        K = self.n_codebooks_predict
        device = content_audio.device
        B = content_audio.shape[0]
        use_prefix = self._uses_speaker_prefix()
        if use_prefix and not use_cache:
            raise RuntimeError("speaker_conditioning='prefix' generation requires use_cache=True.")

        spk = (
            speaker_override
            if speaker_override is not None
            else self.encode_speaker_reference(reference_audio)
        )
        if use_prefix:
            spk_prefix = self._build_speaker_prefix(spk)
            spk_enc = None
        else:
            spk_prefix = None
            spk_enc = self.transformer.encode_speaker(spk)

        content_add_raw, film_raw = self.extract_content(content_audio)
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
        T_total = (max_steps + K - 1) if use_delay else max_steps

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
        if self.use_prosody:
            pa = prosody_audio if prosody_audio is not None else content_audio
            prosody_raw = self.prosody_extractor(pa)
            prosody_add = self._align_content_delay(prosody_raw, max_steps, use_delay, self.null_content_embedding)
            content_add = content_add + prosody_add

        generated = [[] for _ in range(K)]
        prev_emb = self.transformer.start_token.expand(B, -1, -1)
        if use_cache:
            self.transformer.reset_caches()

        use_cfg = cfg_scale != 1.0
        if use_cfg:
            null_spk = self.null_speaker_embedding.expand_as(spk).to(device=device, dtype=spk.dtype)
            null_caches = self.transformer.init_caches(B, device)
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

        if use_prefix:
            assert spk_prefix is not None
            prefix = spk_prefix.to(device=device, dtype=prev_emb.dtype)
            prefix_content = torch.zeros(
                B,
                prefix.size(1),
                self.config.hidden_dim,
                device=device,
                dtype=prev_emb.dtype,
            )
            if film_feats is None:
                prefix_film = None
            else:
                prefix_film = self.null_film_feature.expand(B, prefix.size(1), -1).to(
                    device=device,
                    dtype=film_feats.dtype,
                )
            self.transformer.decode_step(
                prefix,
                prefix_content,
                prefix_film,
                encoder_output=None,
                use_cache=False,
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
                )

        for t in tqdm(range(T_total), desc="VC generating", leave=False):
            c_t = content_add[:, t : t + 1, :]
            f_t = film_feats[:, t : t + 1, :] if film_feats is not None else None
            step_use_cache = use_cache and (t > 0 or use_prefix)

            dec_out = self.transformer.decode_step(
                prev_emb, c_t, f_t, spk_enc, use_cache=step_use_cache,
            )
            hidden_full = dec_out[:, -1, :]

            if use_cfg:
                assert null_caches is not None
                null_out, null_caches = self.transformer.decode_step_stateless(
                    prev_emb, c_t, f_t, null_spk_enc, null_caches,
                    use_cache=step_use_cache,
                )
                hidden_null = null_out[:, -1, :]

            if use_delay:
                for k in range(K):
                    orig = t - k
                    if 0 <= orig < max_steps:
                        lg = self.codebook_heads[k](hidden_full)
                        if use_cfg:
                            lg_n = self.codebook_heads[k](hidden_null)
                            lg = lg_n + cfg_scale * (lg - lg_n)
                        code_k = _sample(lg, temperature, top_k, top_p)
                        generated[k].append(code_k)
                # next prev_emb
                nxt = torch.zeros(B, 1, self.config.hidden_dim, device=device)
                for k in range(K):
                    orig = t - k
                    if 0 <= orig < len(generated[k]):
                        cb = getattr(self, f"codebook_vectors_{k}")
                        nxt += self.decoder_input_projs[k](cb[generated[k][orig]]).unsqueeze(1)
                prev_emb = nxt
            else:
                for k in range(K):
                    lg = self.codebook_heads[k](hidden_full)
                    if use_cfg:
                        lg_n = self.codebook_heads[k](hidden_null)
                        lg = lg_n + cfg_scale * (lg - lg_n)
                    generated[k].append(_sample(lg, temperature, top_k, top_p))
                nxt = torch.zeros(B, 1, self.config.hidden_dim, device=device)
                for k in range(K):
                    cb = getattr(self, f"codebook_vectors_{k}")
                    nxt += self.decoder_input_projs[k](cb[generated[k][-1]]).unsqueeze(1)
                prev_emb = nxt

        return torch.stack([torch.stack(generated[k], dim=1) for k in range(K)], dim=1)

    def decode_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.codec_model.decode(codes)


def _sample(logits, temperature, top_k, top_p):
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    lg = _top_k_top_p(logits, top_k, top_p)
    probs = F.softmax(lg / temperature, dim=-1)
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
