import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
Foley Transformer Layers with ALiBi positional encoding

본 파일은 Sori 스타일을 따라 ALiBi 기반 causal attention을 구현합니다.
- Stateful 경로: 모듈 내부에서 KV 캐시를 유지
- Stateless 경로: 외부에서 KV 캐시를 관리하여 streamable하게 구현
"""


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network.
    
    Attributes:
        ln1 (nn.LayerNorm): Pre-self-attention layer norm
        ln2 (nn.LayerNorm): Pre-cross-attention layer norm
        ln3 (nn.LayerNorm): Pre-FFN layer norm
        self_attn (MultiHeadAttention): Self-attention module
        cross_attn (MultiHeadAttention): Cross-attention module (optional)
        ff (nn.Sequential): Feed-forward network
    """
    
    def __init__(self, config, purpose="decoder"):
        """Initialize transformer block.
        
        Args:
            config: Configuration object
            purpose: Which module this block belongs to ("encoder" or "decoder")
        """
        super().__init__()
        
        # Select configuration based on purpose
        if purpose == "encoder":
            model_dim = config.hidden_dim
            num_heads = config.num_heads
            dropout = config.dropout
            ffn_dim = config.ffn_dim
            do_cross_attn = False
        elif purpose == "decoder":
            model_dim = config.hidden_dim
            num_heads = config.num_heads
            dropout = config.dropout
            ffn_dim = config.ffn_dim
            do_cross_attn = True
        elif purpose == "decoder_only":
            model_dim = config.hidden_dim
            num_heads = config.num_heads
            dropout = config.dropout
            ffn_dim = config.ffn_dim
            do_cross_attn = False
            purpose = "decoder"
        else:
            raise ValueError(f"Invalid purpose: {purpose}. Must be 'encoder' or 'decoder' or 'decoder_only'")
        
        self.config = config
        self.purpose = purpose
        self.do_cross_attn = do_cross_attn
        
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ln3 = nn.LayerNorm(model_dim)
        self.self_attn = MultiHeadAttention(config, purpose=purpose, cross_attn=False)
        if do_cross_attn:
            self.cross_attn = MultiHeadAttention(config, purpose=purpose, cross_attn=True)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mem=None, use_cache=False):
        """Block forward pass with residual connections.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, model_dim]
            mem (torch.Tensor): Memory tensor for cross-attention [batch_size, mem_len, model_dim]
            use_cache (bool): Enable KV caching
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, model_dim]
        """
        x = x + self.self_attn(self.ln1(x), use_cache=use_cache)
        if self.do_cross_attn and mem is not None:
            cross_out = self.cross_attn(self.ln2(x), mem, use_cache=use_cache)
            if getattr(self.config, "ablate_cross_attn", False):
                cross_out = cross_out * 0.0
            if not hasattr(self, '_log_counter'):
                self._log_counter = 0
            if self.training and self._log_counter % 5000 == 0:
                with torch.no_grad():
                    r = cross_out.norm() / (x.norm() + 1e-8)
                    print(f"  [cross/x ratio] {r.item():.6f}  |cross|={cross_out.norm().item():.4f}  |x|={x.norm().item():.4f}", flush=True)
            self._log_counter += 1
            x = x + cross_out
        return x + self.ff(self.ln3(x))
    
    def forward_stateless(self, x, kv_cache_in, mem=None, use_cache=False):
        """Stateless forward with external KV caches.
        
        Args:
            x (Tensor): [B, L, D]
            kv_cache_in (dict): { 'self': kv_cache_dict, 'cross': kv_cache_dict }
            mem (Tensor | None): [B, Lm, D] or None
            use_cache (bool): True to append into cache, False to reset
            
        Returns:
            Tuple[Tensor, dict]: (x_out, kv_cache_out)
        """
        kv_self_in = kv_cache_in.get("self", None)
        kv_cross_in = kv_cache_in.get("cross", None)
        
        y, kv_self_out = self.self_attn.forward_stateless(self.ln1(x), kv_self_in, None, use_cache=use_cache)
        x = x + y
        if self.do_cross_attn and mem is not None:
            y, kv_cross_out = self.cross_attn.forward_stateless(self.ln2(x), kv_cross_in, mem, use_cache=use_cache)
            x = x + y
        else:
            kv_cross_out = None
        x = x + self.ff(self.ln3(x))
        return x, {"self": kv_self_out, "cross": kv_cross_out}


class MultiHeadAttention(nn.Module):
    """Multi-head attention with KV caching and causal masking using ALiBi.
    
    Attributes:
        num_heads (int): Number of attention heads
        head_dim (int): Dimension per attention head
        key (nn.Linear): Key projection
        value (nn.Linear): Value projection
        query (nn.Linear): Query projection
        proj (nn.Linear): Output projection
        attn_dropout (nn.Dropout): Attention dropout
        proj_dropout (nn.Dropout): Output dropout
        causal_attn_bias (torch.Tensor): Causal attention mask
        alibi_attn_bias (torch.Tensor): ALiBi attention bias
        k_cache (torch.Tensor): Key cache
        v_cache (torch.Tensor): Value cache
    """
    
    def __init__(self, config, purpose="decoder", cross_attn=False, stateless: bool = False):
        """Initialize attention module.
        
        Args:
            config: Configuration object
            purpose: Which module this attention belongs to ("encoder" or "decoder")
            cross_attn: Whether this is cross-attention
            stateless: Whether to use stateless mode (external KV cache management)
        """
        super().__init__()
        self.config = config
        self.purpose = purpose
        self.cross_attn = cross_attn
        self.stateless: bool = bool(stateless)
        
        model_dim = config.hidden_dim
        num_heads = config.num_heads
        dropout = config.dropout
        
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Cache capacity
        if purpose == "encoder":
            self.self_capacity = config.max_seq_len
        elif purpose == "decoder":
            self.self_capacity = config.max_seq_len

        # Prefix-aware behavior: when speaker_conditioning == "prefix" and this
        # is decoder self-attn, the first P tokens are speaker prefix. They get
        # zero ALiBi distance penalty (always accessible regardless of distance)
        # and are pinned in the KV cache as extended attention sink.
        # Cross-attn and encoder self-attn are unaffected.
        spk_cond = str(getattr(config, "speaker_conditioning", "crossattn")).lower()
        if (not cross_attn) and purpose == "decoder" and spk_cond == "prefix":
            self.alibi_prefix_len = int(getattr(config, "speaker_prefix_len", 0))
        else:
            self.alibi_prefix_len = 0
        # The CONFIGURED length above is only correct for speaker encoders that emit a
        # fixed-size prefix (speechbrain_ecapa). The "codec"/"codec_prompt" encoders build
        # one prefix token per reference frame (audio_duration*50, e.g. 200 for 4s), so the
        # exemption has to follow the ACTUAL prefix the model built — otherwise every prompt
        # frame past index `speaker_prefix_len` keeps the full ALiBi distance penalty and,
        # sitting 100-400 positions away, becomes unreachable for the large-slope heads.
        # `set_active_prefix_len` is how the model reports the real length at runtime.
        self._active_prefix_len = self.alibi_prefix_len
        self.sink_size = max(1, self.alibi_prefix_len)

        # Projection layers
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.query = nn.Linear(model_dim, model_dim)
        self.proj = nn.Linear(model_dim, model_dim)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Initialize attention mask with ALiBi
        self._init_attn_mask(self.self_capacity)

        # Rotational cache state (ring buffer with attention sink at indices 0..sink_size-1)
        self._cache_inited = False
        self._ring_write_pos = self.sink_size  # next write position in [sink_size..capacity-1]
        self._non_sink_len = 0    # number of valid non-sink slots currently filled (<= capacity-sink_size)
        self._total_seen = 0      # total number of non-sink tokens appended (monotonic)
        self.cur_cache_len = 0    # sink_size + _non_sink_len (exposed for compatibility)
    
    def set_active_prefix_len(self, prefix_len: int):
        """Tell this attention how many leading positions are speaker-prefix tokens.

        Those columns get a zero ALiBi distance penalty (reachable from any query) and are
        pinned as the KV-cache attention sink. Must be called BEFORE the cache is
        (re-)initialised, i.e. before the use_cache=False priming pass.
        """
        p = max(0, int(prefix_len))
        if p == self._active_prefix_len:
            return
        self._active_prefix_len = p
        if (not self.cross_attn) and self.purpose == "decoder":
            self.sink_size = max(1, p)
            self._cache_inited = False
            self._ring_write_pos = self.sink_size
            self._non_sink_len = 0
            self._total_seen = 0
            self.cur_cache_len = 0

    def _apply_prefix_exemption(self, alibi_slice: torch.Tensor) -> torch.Tensor:
        """Zero the ALiBi distance penalty on the leading prefix columns of a bias slice."""
        p = min(self._active_prefix_len, alibi_slice.size(-1))
        if p <= 0:
            return alibi_slice
        out = alibi_slice.clone()
        out[..., :p] = 0.0
        return out

    def _init_attn_mask(self, capacity: int):
        """Initialize causal mask and ALiBi bias.
        
        Only registers buffers that this specific attention module actually needs:
        - Cross-attention: no buffers (attn_bias = None)
        - Encoder self-attention: symmetric ALiBi only
        - Decoder self-attention: causal mask + asymmetric ALiBi
        """
        if self.cross_attn:
            # Cross-attention uses no positional bias → skip all buffer allocation
            return
        
        def get_alibi_slope(num_heads):
            x = (2**8) ** (1 / num_heads)
            return (
                torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
        
        def get_relative_positions(seq_len: int) -> torch.Tensor:
            x = torch.arange(seq_len)[None, :]
            y = torch.arange(seq_len)[:, None]
            return (x - y).unsqueeze(0)
        
        m = get_alibi_slope(self.num_heads)  # [H, 1, 1]
        relative_positions = get_relative_positions(capacity)  # [1, L, L]
        
        if self.purpose == "encoder":
            # Encoder self-attention: symmetric ALiBi (bidirectional)
            # slope * -|j - i| penalizes distance equally in both directions
            symmetric_alibi = (m * -torch.abs(relative_positions)).unsqueeze(0)  # [1, H, L, L]
            self.register_buffer("symmetric_alibi_attn_bias", symmetric_alibi)
        else:
            # Decoder self-attention: causal mask + asymmetric ALiBi
            causal_attn_bias = torch.zeros(capacity, capacity)
            causal_attn_bias = causal_attn_bias.masked_fill(
                torch.triu(
                    torch.ones(capacity, capacity).bool(),
                    diagonal=1,
                ),
                float("-inf"),
            )
            causal_attn_bias = causal_attn_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
            self.register_buffer("causal_attn_bias", causal_attn_bias)

            alibi_attn_bias = (m * relative_positions).unsqueeze(0)  # [1, H, L, L]
            # NOTE: the prefix exemption is NOT baked into this buffer any more — it is
            # applied to the sliced bias in forward()/forward_stateless() using
            # `_active_prefix_len`, which tracks the prefix the model actually built.
            self.register_buffer("alibi_attn_bias", alibi_attn_bias)
    
    def forward(self, x, mem=None, use_cache=False):
        """Compute attention with optional caching.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            mem (torch.Tensor): Memory tensor [batch_size, mem_len, embed_dim]
            use_cache (bool): Use KV caching
            
        Returns:
            torch.Tensor: Attention output [batch_size, seq_len, embed_dim]
        """
        if self.stateless:
            raise RuntimeError("MultiHeadAttention is in stateless mode. Use forward_stateless with external KV cache.")
        
        assert not (
            mem is None and self.cross_attn and (not use_cache or not self._cache_inited)
        ), "When you don't give mem, there should be cache and you should use it."
        
        batch_size, in_seq_len, _ = x.size()
        # Project inputs
        q = self._reshape(self.query(x))
        if self.cross_attn:
            if mem is None:
                new_k = None
                new_v = None
                in_cache_len = 0
            else:
                new_k = self._reshape(self.key(mem))
                new_v = self._reshape(self.value(mem))
                in_cache_len = mem.size(1)
        else:
            new_k = self._reshape(self.key(x))
            new_v = self._reshape(self.value(x))
            in_cache_len = in_seq_len
        
        # Update cache
        if use_cache:
            # Enforce contract: first call must be with use_cache=False
            if not self._cache_inited:
                raise RuntimeError(
                    "KV cache not initialized. "
                    "First forward pass must use use_cache=False to initialize cache."
                )
            if new_k is not None and in_cache_len > 0:
                self._update_cache(new_k, new_v, in_cache_len)
            k, v = self._gather_cache_view()
        else:
            self._reset_cache(new_k, new_v, in_cache_len)
            k, v = new_k, new_v
        
        # Compute attention
        if use_cache:
            # Inference: query position based on current cache length
            q_start = self.cur_cache_len - in_seq_len
            q_end = self.cur_cache_len
        else:
            # Training: use positions from the beginning (for teacher forcing)
            q_start = 0
            q_end = in_seq_len
        
        # Attention bias selection by module type
        if self.cross_attn:
            # Cross-attention: no positional bias (timbre is global, attend freely)
            attn_bias = None
        elif self.purpose == "encoder":
            # Encoder self-attention: symmetric ALiBi (bidirectional)
            attn_bias = self.symmetric_alibi_attn_bias[:, :, q_start:q_end, : self.cur_cache_len]
        else:
            # Decoder self-attention: causal mask + asymmetric ALiBi
            causal_attn_bias_slice = self.causal_attn_bias[:, :, q_start:q_end, : self.cur_cache_len]
            alibi_attn_bias_slice = self._apply_prefix_exemption(
                self.alibi_attn_bias[:, :, q_start:q_end, : self.cur_cache_len]
            )
            attn_bias = causal_attn_bias_slice + alibi_attn_bias_slice
        
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        # Diagnostic: store attention weights when flag is set (cross- or self-attn)
        if getattr(self, '_capture_attn_weights', False):
            with torch.no_grad():
                d_k = q.size(-1)
                raw_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
                if attn_bias is not None:
                    raw_scores = raw_scores + attn_bias
                self._last_attn_weights = torch.softmax(raw_scores, dim=-1)  # (B, H, Lq, Lk)
        
        # Process output
        y = y.transpose(1, 2).reshape(batch_size, in_seq_len, -1)
        return self.proj_dropout(self.proj(y))
    
    def forward_stateless(self, x, kv_cache_in, mem=None, use_cache=False):
        """Stateless attention with external KV cache.
        
        Args:
            x (torch.Tensor): [B, Lq, D]
            kv_cache_in (dict): 외부 KV 캐시 딕셔너리 (None일 수 있음)
            mem (torch.Tensor | None): [B, Lm, D] or None
            use_cache (bool): True이면 기존 캐시에 append, False이면 캐시 reset 후 사용
            
        Returns:
            Tuple[torch.Tensor, dict]: (attn_out [B, Lq, D], kv_cache_out)
        """
        batch_size, in_seq_len, _ = x.size()
        q = self._reshape(self.query(x))
        
        if self.cross_attn:
            if mem is None:
                new_k = None
                new_v = None
                in_cache_len = 0
            else:
                new_k = self._reshape(self.key(mem))
                new_v = self._reshape(self.value(mem))
                in_cache_len = mem.size(1)
        else:
            new_k = self._reshape(self.key(x))
            new_v = self._reshape(self.value(x))
            in_cache_len = in_seq_len
        
        # kv_cache_in이 None이면 초기화
        if kv_cache_in is None:
            device = x.device
            kv_cache_in = init_kv_cache_ext(
                batch_size, self.num_heads, self.self_capacity, self.head_dim, device,
                sink_size=self.sink_size,
            )
        
        kv_cache_out = kv_cache_in
        if use_cache:
            if new_k is not None and in_cache_len > 0:
                kv_cache_out = update_cache_ext(kv_cache_out, new_k, new_v, in_cache_len)
            k, v, kv_cache_out = gather_cache_view_ext(kv_cache_out)
        else:
            kv_cache_out = reset_cache_ext(kv_cache_out, new_k, new_v, in_cache_len, self.cross_attn, self.num_heads, self.head_dim)
            k, v, kv_cache_out = gather_cache_view_ext(kv_cache_out)
        
        # Compute attention positions
        if use_cache:
            q_start = kv_cache_out["cur_cache_len"] - in_seq_len
            q_end = kv_cache_out["cur_cache_len"]
        else:
            q_start = 0
            q_end = in_seq_len
        
        # Attention bias selection by module type
        if self.cross_attn:
            # Cross-attention: no positional bias (timbre is global, attend freely)
            attn_bias = None
        elif self.purpose == "encoder":
            # Encoder self-attention: symmetric ALiBi (bidirectional)
            attn_bias = self.symmetric_alibi_attn_bias[:, :, q_start:q_end, :kv_cache_out["cur_cache_len"]]
        else:
            # Decoder self-attention: causal mask + asymmetric ALiBi
            causal_attn_bias_slice = self.causal_attn_bias[:, :, q_start:q_end, :kv_cache_out["cur_cache_len"]]
            alibi_attn_bias_slice = self._apply_prefix_exemption(
                self.alibi_attn_bias[:, :, q_start:q_end, :kv_cache_out["cur_cache_len"]]
            )
            attn_bias = causal_attn_bias_slice + alibi_attn_bias_slice
        
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(batch_size, in_seq_len, -1)
        return self.proj_dropout(self.proj(y)), kv_cache_out
    
    def _reshape(self, x):
        """Reshape tensor for multi-head computation."""
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
    
    def _update_cache(self, new_k, new_v, in_cache_len):
        """Append new K/V into rotational cache (indices 0..sink_size-1 are reserved for sink)."""
        if in_cache_len <= 0:
            return
        sink = self.sink_size
        ring_size = self.k_cache.size(2) - sink  # non-sink region length
        # Write possibly in two segments (tail then wrap)
        write_from = 0
        remaining = in_cache_len
        while remaining > 0:
            pos = self._ring_write_pos  # in [sink..sink+ring_size-1]
            space = ring_size - (pos - sink)
            take = min(remaining, space)
            dst_slice = slice(pos, pos + take)
            src_slice = slice(write_from, write_from + take)
            self.k_cache[:, :, dst_slice, :] = new_k[:, :, src_slice, :]
            self.v_cache[:, :, dst_slice, :] = new_v[:, :, src_slice, :]
            write_from += take
            remaining -= take
            self._ring_write_pos = sink + ((pos - sink + take) % ring_size)
        self._non_sink_len = min(ring_size, self._non_sink_len + in_cache_len)
        self._total_seen += in_cache_len
        self.cur_cache_len = sink + self._non_sink_len
        return

    def _reset_cache(self, new_k, new_v, seq_len):
        """Reset cache for new sequence and initialize sink at indices 0..sink_size-1."""
        capacity = self.self_capacity
        device = new_k.device if new_k is not None else self.query.weight.device
        batch_size = new_k.size(0) if new_k is not None else 1
        sink = self.sink_size

        # Allocate caches [B,H,L,Hd]
        self.k_cache = torch.zeros(
            batch_size,
            self.num_heads,
            capacity,
            self.head_dim,
            device=device,
        )
        self.v_cache = torch.zeros(
            batch_size,
            self.num_heads,
            capacity,
            self.head_dim,
            device=device,
        )

        # Initialize sink at indices [0..sink_size-1]
        if new_k is not None and seq_len > 0:
            n_sink = min(sink, seq_len)
            self.k_cache[:, :, :n_sink, :] = new_k[:, :, :n_sink, :]
            self.v_cache[:, :, :n_sink, :] = new_v[:, :, :n_sink, :]
            # Fill the rest (excluding sink) through rotational writer
            self._ring_write_pos = sink
            self._non_sink_len = 0
            self._total_seen = 0
            if seq_len - n_sink > 0:
                self._update_cache(new_k[:, :, n_sink:, :], new_v[:, :, n_sink:, :], seq_len - n_sink)
            else:
                self.cur_cache_len = n_sink
        else:
            # No initial tokens; keep sink zeros
            self._ring_write_pos = sink
            self._non_sink_len = 0
            self._total_seen = 0
            self.cur_cache_len = sink
        self._cache_inited = True

    def _gather_cache_view(self):
        """Return (k, v) view assembled as [sink_tokens] + [oldest..newest] from ring."""
        B, H, L, Hd = self.k_cache.size()
        sink = self.sink_size
        ring_size = L - sink
        tail = self._non_sink_len
        if tail <= 0:
            k = self.k_cache[:, :, :sink, :]
            v = self.v_cache[:, :, :sink, :]
            self.cur_cache_len = sink
            return k, v
        # Compute chronological indices in non-sink ring [sink..L-1]
        start = (self._ring_write_pos - tail - sink) % ring_size  # 0-based in [0..ring_size-1]
        idx = (start + torch.arange(tail, device=self.k_cache.device)) % ring_size
        idx = idx + sink  # shift to [sink..L-1]
        sink_idx = torch.arange(sink, device=self.k_cache.device, dtype=idx.dtype)
        all_idx = torch.cat([sink_idx, idx], dim=0)
        index = all_idx.view(1, 1, -1, 1).expand(B, H, -1, Hd)
        k = torch.gather(self.k_cache, 2, index)
        v = torch.gather(self.v_cache, 2, index)
        self.cur_cache_len = sink + tail
        return k, v


# =============================
# External KV-cache utilities
# =============================
def init_kv_cache_ext(
    batch_size: int,
    num_heads: int,
    capacity: int,
    head_dim: int,
    device: torch.device,
    sink_size: int = 1,
):
    """Initialize external KV cache. sink_size pins the first N tokens (e.g. speaker prefix)."""
    k_cache = torch.zeros(batch_size, num_heads, capacity, head_dim, device=device)
    v_cache = torch.zeros(batch_size, num_heads, capacity, head_dim, device=device)
    sink = max(1, int(sink_size))
    return {
        "k": k_cache,
        "v": v_cache,
        "ring_write_pos": sink,
        "non_sink_len": 0,
        "total_seen": 0,
        "cur_cache_len": 0,
        "cache_inited": False,
        "capacity": capacity,
        "sink_size": sink,
    }


def reset_cache_ext(kv_cache: dict, new_k, new_v, seq_len: int, cross_attn: bool, num_heads: int, head_dim: int):
    """Reset external KV cache."""
    capacity = kv_cache["capacity"]
    sink = int(kv_cache.get("sink_size", 1))
    device = new_k.device if new_k is not None else kv_cache["k"].device
    batch_size = new_k.size(0) if new_k is not None else kv_cache["k"].size(0)

    if kv_cache["k"].size(0) != batch_size or kv_cache["k"].size(2) != capacity:
        kv_cache["k"] = torch.zeros(batch_size, num_heads, capacity, head_dim, device=device)
        kv_cache["v"] = torch.zeros(batch_size, num_heads, capacity, head_dim, device=device)

    # Initialize sink at indices [0..sink-1]
    if new_k is not None and seq_len > 0:
        n_sink = min(sink, seq_len)
        kv_cache["k"][:, :, :n_sink, :] = new_k[:, :, :n_sink, :]
        kv_cache["v"][:, :, :n_sink, :] = new_v[:, :, :n_sink, :]
        kv_cache["ring_write_pos"] = sink
        kv_cache["non_sink_len"] = 0
        kv_cache["total_seen"] = 0
        if seq_len - n_sink > 0:
            kv_cache = update_cache_ext(kv_cache, new_k[:, :, n_sink:, :], new_v[:, :, n_sink:, :], seq_len - n_sink)
        else:
            kv_cache["cur_cache_len"] = n_sink
    else:
        kv_cache["ring_write_pos"] = sink
        kv_cache["non_sink_len"] = 0
        kv_cache["total_seen"] = 0
        kv_cache["cur_cache_len"] = sink
    kv_cache["cache_inited"] = True
    return kv_cache


def update_cache_ext(kv_cache: dict, new_k, new_v, in_cache_len: int):
    """Update external KV cache."""
    if in_cache_len <= 0:
        return kv_cache
    sink = int(kv_cache.get("sink_size", 1))
    ring_size = kv_cache["k"].size(2) - sink
    write_from = 0
    remaining = in_cache_len
    while remaining > 0:
        pos = kv_cache["ring_write_pos"]
        space = ring_size - (pos - sink)
        take = min(remaining, space)
        dst_slice = slice(pos, pos + take)
        src_slice = slice(write_from, write_from + take)
        kv_cache["k"][:, :, dst_slice, :] = new_k[:, :, src_slice, :]
        kv_cache["v"][:, :, dst_slice, :] = new_v[:, :, src_slice, :]
        write_from += take
        remaining -= take
        kv_cache["ring_write_pos"] = sink + ((pos - sink + take) % ring_size)
    kv_cache["non_sink_len"] = min(ring_size, kv_cache["non_sink_len"] + in_cache_len)
    kv_cache["total_seen"] += in_cache_len
    kv_cache["cur_cache_len"] = sink + kv_cache["non_sink_len"]
    return kv_cache


def gather_cache_view_ext(kv_cache: dict):
    """Gather cache view from external KV cache."""
    B, H, L, Hd = kv_cache["k"].size()
    sink = int(kv_cache.get("sink_size", 1))
    ring_size = L - sink
    tail = kv_cache["non_sink_len"]
    if tail <= 0:
        k = kv_cache["k"][:, :, :sink, :]
        v = kv_cache["v"][:, :, :sink, :]
        kv_cache["cur_cache_len"] = sink
        return k, v, kv_cache
    start = (kv_cache["ring_write_pos"] - tail - sink) % ring_size
    idx = (start + torch.arange(tail, device=kv_cache["k"].device)) % ring_size
    idx = idx + sink
    sink_idx = torch.arange(sink, device=kv_cache["k"].device, dtype=idx.dtype)
    all_idx = torch.cat([sink_idx, idx], dim=0)
    index = all_idx.view(1, 1, -1, 1).expand(B, H, -1, Hd)
    k = torch.gather(kv_cache["k"], 2, index)
    v = torch.gather(kv_cache["v"], 2, index)
    kv_cache["cur_cache_len"] = sink + tail
    return k, v, kv_cache


