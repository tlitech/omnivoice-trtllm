"""Qwen3 transformer building blocks for TRT-LLM.

Implements the components needed for OmniVoice's Qwen3 backbone:
- RMSNorm
- GQA Attention with RoPE (16 Q heads, 8 KV heads, head_dim=128)
- SwiGLU MLP
- Qwen3Block (full transformer layer)
"""
from __future__ import annotations

import math

import numpy as np
from tensorrt_llm._common import default_net

from ..._utils import str_dtype_to_trt, trt_dtype_to_np
from ...functional import (
    Tensor,
    cast,
    concat,
    constant,
    expand,
    expand_dims_like,
    expand_mask,
    matmul,
    shape,
    silu,
    slice,
    softmax,
    unsqueeze,
)
from ...layers import ColumnLinear, Linear, RowLinear
from ...module import Module
from ...parameter import Parameter


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (Qwen3-style)."""

    def __init__(self, dim, eps=1e-6, dtype=None):
        super().__init__()
        self.eps = eps
        self.dtype = dtype or str_dtype_to_trt("float16")
        self.weight = Parameter(shape=(dim,), dtype=self.dtype)

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = cast(x, "float32")
        variance = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
        eps_const = constant(np.array([self.eps], dtype=np.float32))
        one = constant(np.array([1.0], dtype=np.float32))
        x_normed = x_f32 * (one / (variance + eps_const).sqrt())
        x_normed = cast(x_normed, orig_dtype)
        return x_normed * self.weight.value


# ---------------------------------------------------------------------------
# RoPE helpers (Qwen3 half-half format)
# ---------------------------------------------------------------------------


def rotate_half(x):
    """Rotate half: [-x2, x1] where x = [x1, x2] split at midpoint.

    Args:
        x: [B, S, H, D] tensor
    Returns:
        [B, S, H, D] tensor with halves rotated
    """
    ndim = x.ndim()
    half_shape = concat(
        [
            shape(x, i) if i != ndim - 1 else shape(x, i) / 2
            for i in range(ndim)
        ]
    )
    half_d = shape(x, ndim - 1) / 2

    # x1 = x[..., :D//2], x2 = x[..., D//2:]
    if ndim == 4:
        x1 = slice(x, [0, 0, 0, 0], half_shape, [1, 1, 1, 1])
        x2 = slice(
            x, concat([0, 0, 0, half_d]), half_shape, [1, 1, 1, 1]
        )
    else:  # ndim == 3: [B, S, D]
        x1 = slice(x, [0, 0, 0], half_shape, [1, 1, 1])
        x2 = slice(x, concat([0, 0, half_d]), half_shape, [1, 1, 1])

    neg_one = constant(
        np.array([-1.0], dtype=trt_dtype_to_np(x.dtype))
    )
    neg_x2 = x2 * neg_one
    return concat([neg_x2, x1], dim=ndim - 1)


def apply_rotary_pos_emb(q, k, rope_cos, rope_sin):
    """Apply RoPE to Q and K tensors.

    Args:
        q: [B, S, num_heads, head_dim]
        k: [B, S, num_kv_heads, head_dim]
        rope_cos: [B, S, head_dim]
        rope_sin: [B, S, head_dim]
    Returns:
        Tuple of rotated (q, k)
    """
    # cos/sin: [B, S, head_dim] -> [B, S, 1, head_dim]
    cos = unsqueeze(rope_cos, 2)
    sin = unsqueeze(rope_sin, 2)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Qwen3 Attention (GQA)
# ---------------------------------------------------------------------------


class Qwen3Attention(Module):
    """Multi-head attention with Grouped Query Attention (GQA) and RoPE."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        dtype=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.dtype = dtype or str_dtype_to_trt("float16")

        q_dim = num_attention_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        # QK normalization (Qwen3)
        self.q_norm = RMSNorm(head_dim, eps=1e-6, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=1e-6, dtype=dtype)

        self.q_proj = ColumnLinear(
            hidden_size, q_dim, bias=False, dtype=self.dtype,
            tp_group=None, tp_size=1,
        )
        self.k_proj = ColumnLinear(
            hidden_size, kv_dim, bias=False, dtype=self.dtype,
            tp_group=None, tp_size=1,
        )
        self.v_proj = ColumnLinear(
            hidden_size, kv_dim, bias=False, dtype=self.dtype,
            tp_group=None, tp_size=1,
        )
        self.o_proj = RowLinear(
            q_dim, hidden_size, bias=False, dtype=self.dtype,
            tp_group=None, tp_size=1,
        )

    def forward(self, x, rope_cos, rope_sin, input_lengths):
        """
        Args:
            x: [B, S, H]
            rope_cos: [B, S, head_dim]
            rope_sin: [B, S, head_dim]
            input_lengths: [B] (int32)
        """
        B = shape(x, 0)
        S = shape(x, 1)

        # Project Q, K, V
        q = self.q_proj(x)  # [B, S, num_heads * head_dim]
        k = self.k_proj(x)  # [B, S, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [B, S, num_kv_heads * head_dim]

        # Reshape to [B, S, num_heads, head_dim]
        q = q.view(concat([B, S, self.num_heads, self.head_dim]))
        k = k.view(concat([B, S, self.num_kv_heads, self.head_dim]))
        v = v.view(concat([B, S, self.num_kv_heads, self.head_dim]))

        # QK normalization (Qwen3) — applied per-head before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            # k: [B, S, num_kv_heads, D] -> [B, S, num_kv_heads, groups, D]
            k = k.view(
                concat([B, S, self.num_kv_heads, 1, self.head_dim])
            )
            k = expand(
                k,
                concat(
                    [B, S, self.num_kv_heads, self.num_kv_groups, self.head_dim]
                ),
            )
            k = k.view(concat([B, S, self.num_heads, self.head_dim]))

            v = v.view(
                concat([B, S, self.num_kv_heads, 1, self.head_dim])
            )
            v = expand(
                v,
                concat(
                    [B, S, self.num_kv_heads, self.num_kv_groups, self.head_dim]
                ),
            )
            v = v.view(concat([B, S, self.num_heads, self.head_dim]))

        # Transpose to [B, num_heads, S, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        # k^T: [B, H, D, S]
        k_t = k.permute([0, 1, 3, 2])
        attn_weights = matmul(q, k_t, use_fp32_acc=False)
        scale_const = constant(
            np.array([self.scale], dtype=trt_dtype_to_np(attn_weights.dtype))
        )
        attn_weights = attn_weights * scale_const

        # Apply attention mask from input_lengths
        # mask[b, i] = i < input_lengths[b]
        max_pos = 4096
        pos_buf = constant(
            np.expand_dims(np.arange(max_pos).astype(np.int32), 0)
        )
        pos_ids = slice(pos_buf, [0, 0], concat([1, S]), [1, 1])
        pos_ids = expand(pos_ids, concat([B, S]))  # [B, S]
        len_expanded = unsqueeze(input_lengths, 1)  # [B, 1]
        len_expanded = expand(len_expanded, concat([B, S]))  # [B, S]
        mask = pos_ids < len_expanded  # [B, S] bool
        mask = cast(mask, "int32")

        attn_mask = expand_mask(mask, shape(q, 2))
        attn_mask = cast(attn_mask, attn_weights.dtype)
        attn_weights = attn_weights + attn_mask

        attn_weights = softmax(attn_weights, dim=-1)

        # attn_output: [B, H, S, D]
        attn_output = matmul(attn_weights, v, use_fp32_acc=False)

        # Transpose back: [B, S, H, D]
        attn_output = attn_output.transpose(1, 2)
        # Reshape: [B, S, num_heads * head_dim]
        attn_output = attn_output.view(
            concat([B, S, self.num_heads * self.head_dim])
        )

        # Output projection
        output = self.o_proj(attn_output)

        # Mask output for padding positions
        mask_3d = mask.view(concat([B, S, 1]))
        mask_3d = expand_dims_like(mask_3d, output)
        mask_3d = cast(mask_3d, output.dtype)
        output = output * mask_3d

        return output


# ---------------------------------------------------------------------------
# Qwen3 MLP (SwiGLU)
# ---------------------------------------------------------------------------


class Qwen3MLP(Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, hidden_size, intermediate_size, dtype=None):
        super().__init__()
        self.dtype = dtype or str_dtype_to_trt("float16")
        self.gate_proj = Linear(
            hidden_size, intermediate_size, bias=False, dtype=self.dtype
        )
        self.up_proj = Linear(
            hidden_size, intermediate_size, bias=False, dtype=self.dtype
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, bias=False, dtype=self.dtype
        )

    def forward(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Qwen3 Transformer Block
# ---------------------------------------------------------------------------


class Qwen3Block(Module):
    """Full Qwen3 transformer block: pre-norm → attention → residual →
    post-norm → MLP → residual."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        rms_norm_eps=1e-6,
        dtype=None,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps, dtype)
        self.self_attn = Qwen3Attention(
            hidden_size, num_attention_heads, num_kv_heads, head_dim, dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, rms_norm_eps, dtype
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, dtype)

    def forward(self, hidden_states, rope_cos, rope_sin, input_lengths):
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, rope_cos, rope_sin, input_lengths
        )
        hidden_states = residual + hidden_states

        # MLP with post-attention norm and residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
