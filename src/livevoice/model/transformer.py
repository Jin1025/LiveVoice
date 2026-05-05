"""LiveVoice transformer: streaming voice conversion.

Mirrors sonic's SketchTransformer / SketchModel but rewired for speech:
- Reference audio → DAC continuous z → cross-attention (speaker / timbre)
- Content audio   → HuBERT features  → additive or FiLM conditioning (linguistic)
- Optional prosody (F0 + loudness)   → additive (only if config.use_prosody)
- AR decoder over DAC codebooks with MusicGen delay pattern
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from livevoice.nn.layer import TransformerBlock, init_kv_cache_ext
from livevoice.model.content_perturbation import ContentPerturbation

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

        self.encoder_layers = nn.ModuleList([
            TransformerBlock(config, purpose="encoder")
            for _ in range(config.num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(config, purpose="decoder")
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
        encoder_output: torch.Tensor,
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
        encoder_output: torch.Tensor,
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
        for _ in range(len(self.decoder_layers)):
            caches.append({
                "self": init_kv_cache_ext(batch_size, self.num_heads, self.max_seq_len, head_dim, device),
                "cross": init_kv_cache_ext(batch_size, self.num_heads, self.max_seq_len, head_dim, device),
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
      - dac_model:        frozen DAC (16 kHz speech)
      - content_extractor: HuBERT-based content features
      - prosody_extractor: (optional) F0 + loudness
      - transformer:      LiveVoiceTransformer
    """

    def __init__(self, config, dac_model, content_extractor, prosody_extractor=None):
        super().__init__()
        self.config = config
        self.dac_model = dac_model
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
        latent_dim    = dac_model.latent_dim
        codebook_size = dac_model.codebook_size

        n_proper_init = 0
        for k in range(K):
            emb = dac_model.get_codebook_embeddings(k)
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

        # Speaker / reference → hidden_dim
        self.speaker_proj = nn.Linear(latent_dim, config.hidden_dim)

        # ── Content source path ────────────────────────────────────
        # "hubert"        — uses self.content_extractor (HuBERT)
        # "mimi_semantic" — uses dac_model codebook 0 directly. Lightweight,
        #                   matches codec frame-rate, designed by Kyutai for
        #                   speaker-invariant semantic representation.
        codec_name = str(getattr(config, "codec", "dac")).lower()
        default_src = "mimi_semantic" if codec_name == "mimi" else "hubert"
        self.content_source = str(getattr(config, "content_source", default_src)).lower()
        if self.content_source == "mimi_semantic" and codec_name != "mimi":
            # Silently fall back; HuBERT works for any codec.
            self.content_source = "hubert"
        # If HuBERT extractor wasn't supplied but is needed, fall back
        if self.content_source == "hubert" and self.content_extractor is None:
            raise ValueError(
                "content_source='hubert' but content_extractor was not provided. "
                "Either pass HuBERTContentExtractor or set content_source='mimi_semantic'."
            )
        print(f"[LiveVoiceModel] content_source={self.content_source}  codec={codec_name}")

        if self.content_source == "mimi_semantic":
            # Use Mimi *continuous* encoder output z (pre-quantization) as content.
            # Discrete codebook 0 lookup loses too much information — z is
            # 50× richer (512-dim continuous vs 11-bit discrete at 12.5fps).
            # Encoder is causal, so this stays streaming-compatible.
            sem_dim = int(getattr(dac_model, "latent_dim", config.content_proj_dim))
            self.semantic_proj = nn.Linear(sem_dim, config.content_proj_dim)
            self.semantic_to_hidden = nn.Linear(config.content_proj_dim, config.hidden_dim)
            print(f"[LiveVoiceModel] mimi_semantic = continuous z (latent_dim={sem_dim})")

        # ── Auxiliary CTC head for content alignment ───────────────
        # Predicts character-level text from decoder hidden states. Forces the
        # decoder to encode linguistic content (phoneme-level alignment),
        # directly improving WER. Works with any content_source — operates on
        # the decoder output, not the content path itself.
        self.use_ctc_loss = bool(getattr(config, "use_ctc_loss", False))
        if self.use_ctc_loss:
            self.ctc_vocab_size = int(getattr(config, "ctc_vocab_size", 34))
            self.ctc_head = nn.Linear(config.hidden_dim, self.ctc_vocab_size)
            print(f"[LiveVoiceModel] CTC head: hidden_dim={config.hidden_dim} → vocab={self.ctc_vocab_size}")

    # --------------------- helpers ---------------------
    def align_to_tokens(self, feats: torch.Tensor, num_tokens: int, causal: bool = True) -> torch.Tensor:
        """Align (B, T_src, D) features to `num_tokens` decoder positions.

        Uses the same nearest/causal-average logic as sonic.
        """
        B, T_src, D = feats.shape
        if T_src == num_tokens:
            return feats
        if T_src < num_tokens:
            idx = torch.linspace(0, T_src - 1, num_tokens).long()
            return feats[:, idx, :]
        if causal:
            out = []
            ratio = T_src / num_tokens
            for t in range(num_tokens):
                s = max(0, int(t * ratio))
                e = int((t + 1) * ratio)
                out.append(feats[:, s:e, :].mean(dim=1))
            return torch.stack(out, dim=1)
        idx = torch.linspace(0, T_src - 1, num_tokens).long()
        return feats[:, idx, :]

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
        if reference_z is None:
            if reference_audio is None:
                raise ValueError("Either reference_audio or reference_z must be provided.")
            with torch.no_grad():
                _, z = self.dac_model.encode_continuous(reference_audio)  # (B, T_enc, D_dac)
        else:
            z = reference_z
        spk = self.speaker_proj(z)  # (B, T_enc, D)

        if str(getattr(self.config, "speaker_conditioning", "crossattn")) == "global_avg":
            # Pool to a single speaker token
            spk = spk.mean(dim=1, keepdim=True)  # (B, 1, D)
        return spk

    def extract_content(
        self,
        content_audio: torch.Tensor | None,
        content_feats: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (content_add, film_feats_or_None).

        Three input modes:
          content_feats provided          → skip extractor, project precomputed HuBERT directly
          content_source=="mimi_semantic" → perturb (train) → Mimi encoder → continuous z
          else (HuBERT path)              → perturb (train) → HuBERT online
        """
        use_film = str(getattr(self.config, "content_conditioning", "additive")) == "film"

        if content_feats is not None:
            # Fast path: precomputed HuBERT features (no perturbation needed — already baked in)
            if use_film:
                film_raw = self.content_extractor.from_precomputed_raw(content_feats)
                zeros_add = torch.zeros(
                    film_raw.shape[0], film_raw.shape[1], self.config.hidden_dim,
                    device=film_raw.device, dtype=film_raw.dtype,
                )
                return zeros_add, film_raw
            return self.content_extractor.from_precomputed(content_feats), None

        # Apply perturbation to source audio first (used by both paths below)
        if self.use_content_perturbation and self.training:
            content_audio = self.content_perturbation(content_audio)

        # ── Mimi semantic path: codec encoder → continuous z (pre-quantization) ──
        if self.content_source == "mimi_semantic":
            with torch.no_grad():
                _, z = self.dac_model.encode_continuous(content_audio)  # (B, T, latent_dim)
            sem_emb = self.semantic_proj(z)                              # (B, T, content_proj_dim)
            if use_film:
                zeros_add = torch.zeros(
                    sem_emb.shape[0], sem_emb.shape[1], self.config.hidden_dim,
                    device=sem_emb.device, dtype=sem_emb.dtype,
                )
                return zeros_add, sem_emb
            return self.semantic_to_hidden(sem_emb), None

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
    ) -> dict:
        """Teacher-forced training forward.

        Args:
            reference_audio: (B, T_ref) — same speaker, different utterance
            content_audio:   (B, T_ctn) — linguistic source (used if content_feats is None)
            target_codes:    (B, K, T)  — DAC codes of the target utterance
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
        content_add = self.align_to_tokens(content_add_raw, T_seq, causal=True)
        film_feats = self.align_to_tokens(film_raw, T_seq, causal=True) if film_raw is not None else None

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
            prosody_add = self.align_to_tokens(prosody_raw, T_seq, causal=True)
            drop_pro_all = drop_pro | drop_both
            if drop_pro_all.any():
                prosody_add = prosody_add.clone()
                null = self.null_content_embedding.expand(B, T_seq, -1).to(dtype=prosody_add.dtype)
                prosody_add[drop_pro_all] = null[drop_pro_all]
            content_add = content_add + prosody_add

        # 5. transformer
        out = self.transformer(
            spk, content_add, film_feats, prev_emb,
            use_cache=False,
        )
        decoder_output = out["decoder_output"]

        # 6. per-codebook logits
        all_logits = torch.stack([h(decoder_output) for h in self.codebook_heads], dim=2)
        return {
            "all_logits": all_logits,
            "delayed_targets": targets,
            "decoder_output": decoder_output,
            "encoder_output": out["encoder_output"],
        }

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
    ) -> torch.Tensor:
        use_delay = bool(getattr(self.config, "use_delay_pattern", True))
        K = self.n_codebooks_predict
        device = reference_audio.device
        B = reference_audio.shape[0]

        spk = self.encode_speaker_reference(reference_audio)
        spk_enc = self.transformer.encode_speaker(spk)
        if cfg_scale != 1.0:
            null_spk = self.null_speaker_embedding.expand_as(spk).to(device=device, dtype=spk.dtype)
            spk_enc_null = self.transformer.encode_speaker(null_spk)

        content_add_raw, film_raw = self.extract_content(content_audio)
        T_dec = content_add_raw.shape[1]
        if max_steps is None:
            # Use codec frame count as generation horizon.
            # HuBERT/content frames (e.g. 50 fps) can differ from codec frames
            # (e.g. Mimi 12.5 fps), so using T_dec directly may over-generate.
            hop = int(getattr(self.dac_model, "hop_length", getattr(self.config, "dac_hop_length", 320)))
            n_samples = int(content_audio.shape[-1])
            max_steps = max(1, int(round(n_samples / float(hop))))
        T_total = (max_steps + K - 1) if use_delay else max_steps

        content_add = self.align_to_tokens(content_add_raw, T_total, causal=False)
        film_feats = self.align_to_tokens(film_raw, T_total, causal=False) if film_raw is not None else None
        if self.use_prosody:
            pa = prosody_audio if prosody_audio is not None else content_audio
            prosody_raw = self.prosody_extractor(pa)
            prosody_add = self.align_to_tokens(prosody_raw, T_total, causal=False)
            content_add = content_add + prosody_add

        generated = [[] for _ in range(K)]
        prev_emb = self.transformer.start_token.expand(B, -1, -1)
        if use_cache:
            self.transformer.reset_caches()

        use_cfg = cfg_scale != 1.0
        if use_cfg:
            null_caches = self.transformer.init_caches(B, device)

        for t in tqdm(range(T_total), desc="VC generating", leave=False):
            c_t = content_add[:, t : t + 1, :]
            f_t = film_feats[:, t : t + 1, :] if film_feats is not None else None

            dec_out = self.transformer.decode_step(
                prev_emb, c_t, f_t, spk_enc, use_cache=(use_cache and t > 0),
            )
            hidden_full = dec_out[:, -1, :]

            if use_cfg:
                null_out, null_caches = self.transformer.decode_step_stateless(
                    prev_emb, c_t, f_t, spk_enc_null, null_caches,
                    use_cache=(t > 0),
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
            return self.dac_model.decode(codes)


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
