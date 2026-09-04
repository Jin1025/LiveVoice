"""Streaming-Zipformer ASR encoder as a content (BNF) extractor.

Why: sw2v is a CODEC's content encoder — trained for reconstruction, so it must retain
timbre, which is why its features leak speaker so heavily (probe utt 0.789). An ASR
encoder is trained to predict text, so speaker identity is useless to its objective. This
is the same reasoning behind CosyVoice2's supervised semantic tokens, and MeanVC 2
(arXiv:2606.09050) uses exactly this — a pretrained streaming ASR encoder's bottleneck
features — with no explicit speaker-removal machinery at all.

Frame rate: Conv-Embed takes 100 Hz fbank to 50 Hz, every stack returns to 50 Hz (the
U-Net down/up-sampling happens INSIDE each stack), and only a final `downsample_output`
halves it to 25 Hz. So tapping just before that final downsample gives the deepest
representation at **50 Hz — exactly jhcodec's rate**, with no resampling and none of the
frame-centre alignment risk that cost us a long WER debug with HuBERT.

Model code is vendored under `zipformer_enc/` with k2 / icefall.utils replaced by
`_compat` (k2 was used only for the Swoosh activations, whose closed forms are copied from
icefall's own scripting branch).

Checkpoint: icefall streaming Zipformer trained on LibriSpeech — the same LibriVox domain
as LibriTTS, so no domain mismatch (unlike MeanVC 2's WenetSpeech/Mandarin checkpoint).
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import torch
import torch.nn as nn

from livevoice.model.zipformer_enc import Conv2dSubsampling, Zipformer2


def _derive_arch(sd: dict) -> dict:
    """Recover the architecture from tensor shapes.

    The published checkpoints are not always the default recipe (this one has
    feedforward_dim 384,576,768,1152,768,576 rather than the documented defaults), so
    guessing a recipe silently produces a shape mismatch or, worse, a wrong model that
    still loads. Everything here is read off the weights.
    """
    def get(stack: int, name: str):
        for k in (f"encoder.encoders.{stack}.layers.0.{name}",
                  f"encoder.encoders.{stack}.encoder.layers.0.{name}"):
            if k in sd:
                return sd[k]
        return None

    n_stacks = 1 + max(
        int(m.group(1))
        for m in (re.match(r"encoder\.encoders\.(\d+)\.", k) for k in sd)
        if m
    )
    layers, dims, ffs, heads, kernels = [], [], [], [], []
    for s in range(n_stacks):
        n = 1 + max(
            int(m.group(1))
            for m in (
                re.match(rf"encoder\.encoders\.{s}\.(?:encoder\.)?layers\.(\d+)\.", k)
                for k in sd
            )
            if m
        )
        ip = get(s, "self_attn_weights.in_proj.weight")
        lp = get(s, "self_attn_weights.linear_pos.weight")
        # each layer holds three FeedforwardModules sized (3/4·d, d, 5/4·d), so only
        # feed_forward2 carries `feedforward_dim` itself.
        ff = get(s, "feed_forward2.in_proj.weight")
        # causal depthwise conv is split into causal_conv (kernel-1) + chunkwise_conv
        # (kernel); only the latter gives the real cnn_module_kernel, which must be odd.
        cm = get(s, "conv_module1.depthwise_conv.chunkwise_conv.weight")
        pos_head_dim = 4
        nh = lp.shape[0] // pos_head_dim
        layers.append(n)
        dims.append(ip.shape[1])
        ffs.append(ff.shape[0])
        heads.append(nh)
        kernels.append(cm.shape[-1] if cm is not None else 31)
    return {
        "num_encoder_layers": tuple(layers),
        "encoder_dim": tuple(dims),
        "feedforward_dim": tuple(ffs),
        "num_heads": tuple(heads),
        "cnn_module_kernel": tuple(kernels),
        "pos_dim": int(get(0, "self_attn_weights.linear_pos.weight").shape[1]),
        "query_head_dim": (get(0, "self_attn_weights.in_proj.weight").shape[0] // heads[0] - 4) // 2,
        "value_head_dim": get(0, "self_attn1.in_proj.weight").shape[0] // heads[0],
        "feature_dim": (sd["encoder_embed.conv.0.weight"].shape[1] and 80),
    }


class ZipformerContentEncoder(nn.Module):
    """(B, T_audio) waveform → (B, T_frames, D) content features at 50 fps.

    `layer` selects the tap:
        -1  (default) input to the final downsample — deepest, 50 Hz, full dim
        0..5          output of that encoder stack — also 50 Hz
        "out"         the model's own 25 Hz encoder output
    """

    def __init__(self, config, ckpt_path: str | None = None, layer: int | str = -1):
        super().__init__()
        ckpt = str(ckpt_path or getattr(config, "zipformer_ckpt", ""))
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = obj.get("model", obj)
        arch = _derive_arch(sd)
        self.arch = arch

        # downsampling_factor is the one thing not recoverable from shapes (the Downsample
        # modules are parameter-free); it is the standard 1,2,4,8,4,2 U-Net schedule and
        # must match len(encoder_dim).
        ds = (1, 2, 4, 8, 4, 2)[: len(arch["encoder_dim"])]

        self.encoder_embed = Conv2dSubsampling(
            in_channels=80, out_channels=arch["encoder_dim"][0], layer1_channels=8,
            layer2_channels=32, layer3_channels=128, dropout=0.0,
        )
        self.encoder = Zipformer2(
            output_downsampling_factor=2,
            downsampling_factor=ds,
            num_encoder_layers=arch["num_encoder_layers"],
            encoder_dim=arch["encoder_dim"],
            encoder_unmasked_dim=tuple(min(192, d) for d in arch["encoder_dim"]),
            query_head_dim=arch["query_head_dim"],
            pos_head_dim=4,
            value_head_dim=arch["value_head_dim"],
            num_heads=arch["num_heads"],
            feedforward_dim=arch["feedforward_dim"],
            cnn_module_kernel=arch["cnn_module_kernel"],
            pos_dim=arch["pos_dim"],
            dropout=0.0,
            causal=True,                 # streaming checkpoint — must stay causal
            chunk_size=(getattr(config, "zipformer_chunk_size", 16),),
            left_context_frames=(128,),
        )
        emb_sd = {k[len("encoder_embed."):]: v for k, v in sd.items()
                  if k.startswith("encoder_embed.")}
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        m1, u1 = self.encoder_embed.load_state_dict(emb_sd, strict=False)
        m2, u2 = self.encoder.load_state_dict(enc_sd, strict=False)
        if m1 or u1 or m2 or u2:
            raise RuntimeError(
                f"zipformer state_dict mismatch — embed(missing={len(m1)}, unexpected={len(u1)}) "
                f"encoder(missing={len(m2)}, unexpected={len(u2)}). The derived architecture "
                f"does not match the checkpoint: {arch}"
            )

        self.config_pad = config      # for zipformer_align_pad_frames
        self.layer = layer
        self._tap: torch.Tensor | None = None
        if layer == "out":
            pass
        elif layer == -1:
            self.encoder.downsample_output.register_forward_pre_hook(
                lambda _m, inp: self._capture(inp[0]))
        else:
            self.encoder.encoders[int(layer)].register_forward_hook(
                lambda _m, _i, out: self._capture(out))

        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.out_dim = (arch["encoder_dim"][int(layer)] if isinstance(layer, int) and layer >= 0
                        else max(arch["encoder_dim"]))
        print(f"[zipformer] layers={arch['num_encoder_layers']} dims={arch['encoder_dim']} "
              f"heads={arch['num_heads']} causal=True | tap={layer} out_dim={self.out_dim} "
              f"@ {'25' if layer == 'out' else '50'} fps")

    def _capture(self, x):
        self._tap = x                      # (T, N, C) inside the encoder
        return None

    @staticmethod
    def _fbank(wav: torch.Tensor, sr: int) -> torch.Tensor:
        """80-bin Kaldi fbank at 100 Hz, matching icefall's kaldifeat options
        (dither=0, snip_edges=False, high_freq=-400)."""
        import torchaudio
        feats = [
            torchaudio.compliance.kaldi.fbank(
                w.unsqueeze(0), num_mel_bins=80, sample_frequency=float(sr),
                dither=0.0, snip_edges=False, high_freq=-400.0,
            )
            for w in wav
        ]
        return torch.stack(feats, dim=0)

    # Frame alignment to the codec grid, MEASURED (debug/diag_frame_alignment.py + a ridge
    # probe of how well features predict the codec latent at each lag):
    #
    #     pad  0 -> best lag -4     R2@lag0 0.2186     (content 80ms stale, no added latency)
    #     pad -4 -> best lag  0     R2@lag0 0.2258
    #     pad -6 -> best lag  0     R2@lag0 0.2275     (aligned, ~120ms lookahead)
    #     pad +4 -> best lag -9     R2@lag0 0.2085     (worst — do not use)
    #
    # An earlier +4 default was derived from the Conv2dSubsampling receptive field alone and
    # was WRONG: it ignored the causal attention stack's long left context, which pulls the
    # centre of mass of each frame BEHIND its index. Aligning to lag 0 therefore requires a
    # NEGATIVE pad (trimming the front), which is equivalent to letting each feature see
    # further ahead — real lookahead, hence the streaming trade-off above.
    #
    # >0 pads silence at the front, <0 trims it. 0 = no added latency, which is the default
    # for a streaming target; set negative to buy alignment with latency.
    ALIGN_PAD_FRAMES = 0

    def _prepare_aligned_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        align_to_codec: bool,
    ) -> tuple[torch.Tensor, int]:
        """Apply the exact waveform alignment used by the existing batch path."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        hop = sample_rate // 50
        n_target = -(-int(audio.shape[-1]) // hop)
        if align_to_codec:
            pad = int(getattr(self.config_pad, "zipformer_align_pad_frames",
                              self.ALIGN_PAD_FRAMES))
            if pad > 0:
                audio = torch.nn.functional.pad(audio, (pad * hop, 0))
            elif pad < 0:
                audio = audio[..., -pad * hop:]
            need_samples = (2 * n_target + 7) * (sample_rate // 100) - audio.shape[-1]
            if need_samples > 0:
                audio = torch.nn.functional.pad(audio, (0, int(need_samples)))
        return audio, n_target

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, sample_rate: int = 16000,
                align_to_codec: bool = True) -> torch.Tensor:
        audio, n_target = self._prepare_aligned_audio(
            audio, sample_rate, align_to_codec)
        x = self._fbank(audio, sample_rate).to(next(self.parameters()).device)
        x_lens = torch.full((x.size(0),), x.size(1), dtype=torch.int64, device=x.device)
        x, x_lens = self.encoder_embed(x, x_lens)
        x = x.permute(1, 0, 2)                                   # (N,T,C) → (T,N,C)
        mask = torch.arange(x.size(0), device=x.device)[None, :] >= x_lens[:, None]
        out, _ = self.encoder(x, x_lens, mask)
        if self.layer == "out":
            feats = out.permute(1, 0, 2)
        else:
            assert self._tap is not None, "tap hook did not fire"
            tap, self._tap = self._tap, None
            feats = tap.permute(1, 0, 2)                         # (T,N,C) → (N,T,C)
        if align_to_codec and self.layer != "out":
            # trim/pad the tail so the count is exactly ceil(len/hop), like the codec
            T = feats.shape[1]
            if T > n_target:
                feats = feats[:, :n_target]
            elif T < n_target:
                raise RuntimeError(
                    f"zipformer emitted {T} frames but the codec grid needs {n_target}; the "
                    f"tail audio padding above should have prevented this. Zero-filling here "
                    f"would feed the model fabricated content.")
        return feats

    @torch.no_grad()
    def forward_encoder_streaming(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        align_to_codec: bool = True,
    ) -> torch.Tensor:
        """Run Zipformer one chunk at a time with its real state/cache API.

        This first equivalence stage intentionally keeps waveform alignment, fbank,
        and Conv2dSubsampling identical to ``forward``.  It isolates the stateful
        Zipformer boundary before incremental frontend buffering is implemented.
        """
        audio, n_target = self._prepare_aligned_audio(
            audio, sample_rate, align_to_codec)
        x = self._fbank(audio, sample_rate).to(next(self.parameters()).device)
        x_lens = torch.full(
            (x.size(0),), x.size(1), dtype=torch.int64, device=x.device)
        x, x_lens = self.encoder_embed(x, x_lens)

        chunk_size = int(self.encoder.chunk_size[0])
        if chunk_size <= 0:
            raise ValueError(
                "forward_encoder_streaming requires a positive Zipformer chunk size")
        batch_size, total_frames, _ = x.shape
        states = self.encoder.get_init_states(batch_size=batch_size, device=x.device)
        left_context = int(self.encoder.left_context_frames[0])
        cache_padding = torch.ones(
            batch_size, left_context, dtype=torch.bool, device=x.device)
        pieces: list[torch.Tensor] = []

        for start in range(0, total_frames, chunk_size):
            chunk = x[:, start:start + chunk_size]
            valid = int(chunk.shape[1])
            if valid < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, chunk_size - valid))
            chunk_lens = torch.full(
                (batch_size,), valid, dtype=torch.int64, device=x.device)
            current_padding = (
                torch.arange(chunk_size, device=x.device)[None, :] >= chunk_lens[:, None]
            )
            padding_mask = torch.cat([cache_padding, current_padding], dim=1)
            self._tap = None
            out, _, states = self.encoder.streaming_forward(
                chunk.permute(1, 0, 2), chunk_lens, states, padding_mask)
            cache_padding = torch.cat(
                [cache_padding, current_padding], dim=1)[:, -left_context:]
            if self.layer == "out":
                piece = out.permute(1, 0, 2)
            else:
                assert self._tap is not None, "tap hook did not fire in streaming forward"
                piece, self._tap = self._tap.permute(1, 0, 2), None
            pieces.append(piece[:, :valid])

        feats = torch.cat(pieces, dim=1)
        if align_to_codec and self.layer != "out":
            if feats.shape[1] < n_target:
                raise RuntimeError(
                    f"streaming Zipformer emitted {feats.shape[1]} frames but needs {n_target}")
            feats = feats[:, :n_target]
        return feats

    @torch.no_grad()
    def waveform_streaming_chunks(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
        input_chunk_samples: int | None = None,
        align_to_codec: bool = True,
        on_chunk: Callable[[torch.Tensor], None] | None = None,
    ) -> list[torch.Tensor]:
        """Simulate raw waveform arrival and return finalized 50-Hz chunks.

        No future waveform is read while processing a non-final input block.
        Fbank frames whose 25-ms analysis window touches the current right edge
        are held back.  The subsampler consumes overlapping fbank windows and
        both it and Zipformer retain their real streaming states.  EOF uses the
        same waveform tail padding and ConvNeXt right padding as ``forward``.
        """
        if self.layer == "out":
            raise NotImplementedError("25-Hz output taps are not used by LiveVoice streaming")
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.size(0) != 1:
            raise ValueError("waveform_streaming_chunks currently supports batch size 1")
        hop = sample_rate // 50
        chunk_size = int(self.encoder.chunk_size[0])
        if chunk_size <= 0:
            raise ValueError("streaming requires a positive Zipformer chunk size")
        if input_chunk_samples is None:
            input_chunk_samples = chunk_size * hop
        if input_chunk_samples <= 0:
            raise ValueError("input_chunk_samples must be positive")

        pad = int(getattr(self.config_pad, "zipformer_align_pad_frames",
                          self.ALIGN_PAD_FRAMES)) if align_to_codec else 0
        front_zeros = max(0, pad) * hop
        front_skip = max(0, -pad) * hop
        device = next(self.parameters()).device
        embed_state = self.encoder_embed.get_init_states(1, device)
        encoder_states = self.encoder.get_init_states(batch_size=1, device=device)
        left_context = int(self.encoder.left_context_frames[0])
        cache_padding = torch.ones(1, left_context, dtype=torch.bool, device=device)
        fbank_offset = 0
        emitted = 0
        output_chunks: list[torch.Tensor] = []

        def run_encoder(embedded: torch.Tensor, valid: int) -> None:
            nonlocal encoder_states, cache_padding
            if valid < chunk_size:
                embedded = torch.nn.functional.pad(
                    embedded, (0, 0, 0, chunk_size - valid))
            current_padding = (
                torch.arange(chunk_size, device=device)[None, :] >= valid
            )
            padding_mask = torch.cat([cache_padding, current_padding], dim=1)
            self._tap = None
            _, _, encoder_states = self.encoder.streaming_forward(
                embedded.permute(1, 0, 2),
                torch.tensor([valid], dtype=torch.int64, device=device),
                encoder_states,
                padding_mask,
            )
            cache_padding = torch.cat(
                [cache_padding, current_padding], dim=1)[:, -left_context:]
            assert self._tap is not None, "tap hook did not fire in waveform streaming"
            piece, self._tap = self._tap.permute(1, 0, 2)[:, :valid], None
            output_chunks.append(piece)
            if on_chunk is not None:
                on_chunk(piece)

        def aligned_prefix(end: int) -> torch.Tensor:
            prefix = audio[..., :end]
            if front_zeros:
                prefix = torch.nn.functional.pad(prefix, (front_zeros, 0))
            elif front_skip:
                prefix = prefix[..., min(front_skip, prefix.shape[-1]):]
            return prefix

        # A snip_edges=False 25-ms frame at index i needs samples through
        # i*10ms+17.5ms.  Only commit frames whose right edge has arrived.
        frame_shift = sample_rate // 100
        right_extent = 7 * sample_rate // 400  # 17.5 ms
        total_samples = int(audio.shape[-1])
        for end in range(input_chunk_samples, total_samples, input_chunk_samples):
            prefix = aligned_prefix(end)
            n_prefix = int(prefix.shape[-1])
            stable = max(0, (n_prefix - right_extent - 1) // frame_shift + 1)
            # 45 fbank frames -> 19 strided-conv frames -> 16 finalized
            # ConvNeXt frames.  Each following window advances by 32 fbanks.
            if stable < fbank_offset + 2 * chunk_size + 13:
                continue
            fbanks = self._fbank(prefix, sample_rate).to(device)
            while stable >= fbank_offset + 2 * chunk_size + 13:
                segment = fbanks[:, fbank_offset:fbank_offset + 2 * chunk_size + 13]
                embedded, embed_state = self.encoder_embed.streaming_forward_exact(
                    segment, embed_state, chunk_size)
                run_encoder(embedded, chunk_size)
                fbank_offset += 2 * chunk_size
                emitted += chunk_size

        # Flush using exactly the same aligned/tail-padded waveform as batch mode.
        final_audio, n_target = self._prepare_aligned_audio(
            audio, sample_rate, align_to_codec)
        final_fbanks = self._fbank(final_audio, sample_rate).to(device)
        while emitted < n_target:
            valid = min(chunk_size, n_target - emitted)
            segment = final_fbanks[:, fbank_offset:]
            embedded, embed_state = self.encoder_embed.streaming_forward_exact(
                segment, embed_state, valid)
            run_encoder(embedded, valid)
            fbank_offset += 2 * valid
            emitted += valid

        return output_chunks
