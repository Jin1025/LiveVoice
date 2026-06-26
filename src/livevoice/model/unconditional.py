"""Decoder-only unconditional speech generation model (sanity-check baseline).

No conditioning (no speaker ref, no content, no prosody). Pure AR next-token
prediction over DAC-16kHz codebooks with MusicGen delay pattern.

Used as the first roadmap step — confirms the decoder can model VCTK speech
before wiring in HuBERT / reference cross-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from livevoice.nn.layer import TransformerBlock

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


class UnconditionalTransformer(nn.Module):
    """Decoder-only transformer (causal self-attn, no cross-attn)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = config.max_seq_len
        self.num_heads = config.num_heads
        self.gradient_checkpointing = False

        self.layers = nn.ModuleList([
            TransformerBlock(config, purpose="decoder_only")
            for _ in range(config.num_decoder_layers)
        ])

        self.start_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and not use_cache:
                x = grad_checkpoint(layer, x, None, use_cache, use_reentrant=False)
            else:
                x = layer(x, mem=None, use_cache=use_cache)
        return self.norm(x)

    def reset_caches(self):
        for layer in self.layers:
            if hasattr(layer.self_attn, "_cache_inited"):
                layer.self_attn._cache_inited = False


class UnconditionalModel(nn.Module):
    """Unconditional DAC code generation model.

    Keeps only: DAC encode/decode, codebook embeddings, delay pattern,
    per-codebook projection + output heads.
    """

    def __init__(self, config, dac_model):
        super().__init__()
        self.config = config
        self.dac_model = dac_model

        self.transformer = UnconditionalTransformer(config)

        K = int(config.n_codebooks_predict)
        self.n_codebooks_predict = K

        for k in range(K):
            qk = dac_model.dac_model.quantizer.quantizers[k]
            with torch.no_grad():
                cb = qk.codebook.weight
                cb_for_conv = cb.T.unsqueeze(0)
                effective_emb = qk.out_proj(cb_for_conv).squeeze(0).T
            self.register_buffer(f"codebook_vectors_{k}", effective_emb)

        self.decoder_input_projs = nn.ModuleList([
            nn.Linear(int(dac_model.latent_dim), config.hidden_dim) for _ in range(K)
        ])
        self.codebook_heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, int(dac_model.codebook_size)) for _ in range(K)
        ])

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
            emb_k = proj(cb[codes[:, k, :]])
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

    # --------------------- forward ---------------------
    def forward(self, target_codes: torch.Tensor) -> dict:
        prev_emb = self._build_delay_input(target_codes)
        targets = self._build_delay_targets(target_codes)
        decoder_output = self.transformer(prev_emb, use_cache=False)
        all_logits = torch.stack(
            [h(decoder_output) for h in self.codebook_heads], dim=2
        )
        return {
            "all_logits": all_logits,
            "delayed_targets": targets,
            "decoder_output": decoder_output,
        }

    # --------------------- generate ---------------------
    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        max_steps: int = 200,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        K = self.n_codebooks_predict
        device = self.transformer.start_token.device
        use_delay = bool(getattr(self.config, "use_delay_pattern", True))
        T_total = (max_steps + K - 1) if use_delay else max_steps

        generated = [[] for _ in range(K)]
        prev_emb = self.transformer.start_token.expand(batch_size, -1, -1)
        self.transformer.reset_caches()

        for t in tqdm(range(T_total), desc="Generating", leave=False):
            dec = self.transformer(prev_emb, use_cache=(t > 0))
            hidden = dec[:, -1, :]

            if use_delay:
                for k in range(K):
                    orig = t - k
                    if 0 <= orig < max_steps:
                        lg = self.codebook_heads[k](hidden)
                        generated[k].append(_sample(lg, temperature, top_k, top_p))
                nxt = torch.zeros(batch_size, 1, self.config.hidden_dim, device=device)
                for k in range(K):
                    orig = t - k
                    if 0 <= orig < len(generated[k]):
                        cb = getattr(self, f"codebook_vectors_{k}")
                        nxt += self.decoder_input_projs[k](cb[generated[k][orig]]).unsqueeze(1)
                prev_emb = nxt
            else:
                for k in range(K):
                    lg = self.codebook_heads[k](hidden)
                    generated[k].append(_sample(lg, temperature, top_k, top_p))
                nxt = torch.zeros(batch_size, 1, self.config.hidden_dim, device=device)
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
