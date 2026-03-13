"""GLM-4.5 Attention Pre-processing with Quantization.

Fused: add residual → RMSNorm → quantize → QKV matmul → split Q/K/V →
Q/K norm → RoPE.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_matmul,
    tensor_mul,
    tensor_rmsnorm,
    tensor_rope,
    tensor_symmetric_quant,
    tensor_view,
)


@pl.function
def attention_pre_quant(
    hidden_states: pl.Tensor,
    residual: pl.Tensor,
    layernorm_weight: pl.Tensor,
    layernorm_bias: pl.Tensor,
    qkv_weight: pl.Tensor,
    qkv_deq_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    query_out: pl.Out[pl.Tensor],
    key_out: pl.Out[pl.Tensor],
    value_out: pl.Out[pl.Tensor],
    residual_out: pl.Out[pl.Tensor],
    eps: float = 1e-5,
    head_dim: int = 128,
) -> None:
    """Fused attention pre-processing with quantized QKV projection.

    hidden_states: [bs, hidden_size]
    residual:      [bs, hidden_size]
    """
    bs = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]

    # Add residual
    fused = tensor_add(hidden_states, residual)
    pl.assemble(residual_out, fused, [0, 0])

    # RMSNorm
    normed = tensor_rmsnorm(fused, layernorm_weight, eps)
    normed = tensor_add(normed, layernorm_bias)

    # Quantized matmul: normed @ qkv_weight^T
    qkv_weight_t = pl.transpose(qkv_weight)
    qkv_proj = tensor_matmul(normed, qkv_weight_t)
    qkv_proj = tensor_mul(qkv_proj, qkv_deq_scale)

    # Split Q, K, V
    total_qkv = qkv_proj.shape[1]
    q_size = total_qkv // 3
    kv_size = q_size

    q_raw = pl.slice(qkv_proj, [bs, q_size], [0, 0])
    k_raw = pl.slice(qkv_proj, [bs, kv_size], [0, q_size])
    v_raw = pl.slice(qkv_proj, [bs, kv_size], [0, q_size + kv_size])
    pl.assemble(value_out, v_raw, [0, 0])

    # RoPE on Q and K
    tensor_rope(q_raw, k_raw, cos, sin, query_out, key_out)
