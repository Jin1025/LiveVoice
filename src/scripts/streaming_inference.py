"""Stateful streaming token inference for the current LiveVoice checkpoint."""
from __future__ import annotations

import torch


class LiveVoiceTokenStreamingSession:
    """Consume finalized Zipformer chunks and emit completed codec-token frames.

    The implementation intentionally supports the supplied 180ms-baseline setup
    first.  Assertions prevent silently changing behavior for configurations that
    require utterance statistics or additional online feature streams.
    """

    def __init__(self, model, reference_audio: torch.Tensor):
        self.model = model
        self.reference_audio = reference_audio
        cfg = model.config
        unsupported = []
        if model.content_source != "zipformer":
            unsupported.append(f"content_source={model.content_source}")
        if str(getattr(cfg, "content_cmn", "off")) != "off":
            unsupported.append(f"content_cmn={getattr(cfg, 'content_cmn', None)}")
        if model.content_refiner is not None:
            unsupported.append("content_refiner")
        self.use_film = str(
            getattr(cfg, "content_conditioning", "additive")) == "film"
        for name in ("use_prosody", "use_cepstral", "use_mpm"):
            if bool(getattr(model, name, False)):
                unsupported.append(name)
        if not model._uses_codec_prompt_continuation():
            unsupported.append("codec_prompt_continuation=False")
        if unsupported:
            raise NotImplementedError(
                "streaming session does not yet support: " + ", ".join(unsupported))

        self.device = reference_audio.device
        self.batch_size = int(reference_audio.shape[0])
        self.codebooks = model.n_codebooks_predict
        self.delays = model._codebook_delays()
        self.delay_tail = model._delay_tail()
        self.generated: list[list[torch.Tensor]] = [
            [] for _ in range(self.codebooks)
        ]
        self.emitted = 0
        self.position = 0
        self.finished = False

        self.prompt_codes = model._encode_reference_codes(reference_audio)
        self.prompt_frames = int(self.prompt_codes.shape[2])
        model._set_prefix_len_for_attn(self.prompt_frames)
        ar_dtype = model.transformer.start_token.dtype
        prime = model._build_delay_input(self.prompt_codes).to(dtype=ar_dtype)
        prefix = prime[:, :self.prompt_frames]
        self.prev_emb = prime[:, self.prompt_frames:self.prompt_frames + 1]

        model.transformer.reset_caches()
        prefix_content, prefix_film = model._prompt_region_content(
            reference_audio, self.prompt_frames, self.batch_size, self.device)
        model.transformer.decode_step(
            prefix,
            prefix_content.to(dtype=ar_dtype),
            None if prefix_film is None else prefix_film.to(dtype=ar_dtype),
            encoder_output=None,
            use_cache=False,
            mpm_feats=None,
        )

    def _completed_since_last_call(self) -> torch.Tensor:
        complete = min(len(items) for items in self.generated)
        if complete <= self.emitted:
            return torch.empty(
                self.batch_size, self.codebooks, 0,
                dtype=torch.long, device=self.device)
        result = torch.stack([
            torch.stack(items[self.emitted:complete], dim=1)
            for items in self.generated
        ], dim=1)
        self.emitted = complete
        return result

    def _step(
        self,
        content: torch.Tensor,
        film: torch.Tensor | None,
        max_steps: int | None = None,
    ) -> torch.Tensor:
        hidden = self.model.transformer.decode_step(
            self.prev_emb,
            content,
            film_feats=film,
            encoder_output=None,
            use_cache=True,
            mpm_feats=None,
        )[:, -1]
        t = self.position
        for k, delay in enumerate(self.delays):
            original = t - delay
            if original >= 0 and (max_steps is None or original < max_steps):
                token = torch.argmax(self.model.codebook_heads[k](hidden), dim=-1)
                self.generated[k].append(token)

        nxt = torch.zeros(
            self.batch_size, 1, self.model.config.hidden_dim,
            device=self.device, dtype=self.prev_emb.dtype)
        for k, delay in enumerate(self.delays):
            original = t - delay
            table = getattr(self.model, f"codebook_vectors_{k}")
            if original >= 0 and (max_steps is None or original < max_steps):
                token = self.generated[k][original]
                nxt += self.model.decoder_input_projs[k](table[token]).unsqueeze(1)
            elif original < 0 and self.prompt_frames + original >= 0:
                token = self.prompt_codes[:, k, self.prompt_frames + original]
                nxt += self.model.decoder_input_projs[k](table[token]).unsqueeze(1)
        self.prev_emb = nxt
        self.position += 1
        return self._completed_since_last_call()

    @torch.no_grad()
    def push_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.finished:
            raise RuntimeError("cannot push features after finish()")
        content, film = self.model.extract_content(None, content_feats=features)
        content = content.to(dtype=self.prev_emb.dtype)
        if film is not None:
            film = film.to(dtype=self.prev_emb.dtype)
        emitted = [
            self._step(
                content[:, i:i + 1],
                None if film is None else film[:, i:i + 1],
            )
            for i in range(content.shape[1])
        ]
        emitted = [item for item in emitted if item.shape[-1] > 0]
        if not emitted:
            return torch.empty(
                self.batch_size, self.codebooks, 0,
                dtype=torch.long, device=self.device)
        return torch.cat(emitted, dim=2)

    @torch.no_grad()
    def finish(self) -> torch.Tensor:
        if self.finished:
            raise RuntimeError("finish() may only be called once")
        self.finished = True
        max_steps = self.position
        tail = self.model.null_content_embedding.to(
            device=self.device, dtype=self.prev_emb.dtype)
        film_tail = (
            self.model.null_film_feature.to(
                device=self.device, dtype=self.prev_emb.dtype)
            if self.use_film else None
        )
        emitted = [
            self._step(tail, film_tail, max_steps=max_steps)
            for _ in range(self.delay_tail)
        ]
        emitted = [item for item in emitted if item.shape[-1] > 0]
        if not emitted:
            return torch.empty(
                self.batch_size, self.codebooks, 0,
                dtype=torch.long, device=self.device)
        return torch.cat(emitted, dim=2)


@torch.no_grad()
def generate_streaming(
    model,
    reference_audio: torch.Tensor,
    content_audio: torch.Tensor,
    input_chunk_samples: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Run waveform frontend and AR decoder incrementally.

    Returns all codec tokens plus the number of fully completed codec frames
    emitted on each content/EOF callback.
    """
    session = LiveVoiceTokenStreamingSession(model, reference_audio)
    token_chunks: list[torch.Tensor] = []
    emission_sizes: list[int] = []

    def consume(features: torch.Tensor) -> None:
        tokens = session.push_features(features)
        if tokens.shape[-1]:
            token_chunks.append(tokens)
            emission_sizes.append(int(tokens.shape[-1]))

    model.content_extractor.waveform_streaming_chunks(
        content_audio,
        sample_rate=int(model.config.sample_rate),
        input_chunk_samples=input_chunk_samples,
        on_chunk=consume,
    )
    tail = session.finish()
    if tail.shape[-1]:
        token_chunks.append(tail)
        emission_sizes.append(int(tail.shape[-1]))
    return torch.cat(token_chunks, dim=2), emission_sizes
