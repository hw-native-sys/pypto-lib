"""DeepSeek-V3 Lightning Indexer Prolog with Quantization.

Computes Indexer Q, K, and weights from raw hidden states:
Q: linear → dequant → RoPE → Hadamard → quantize
K: linear → LayerNorm → RoPE → Hadamard → quantize
Weights: linear → softmax normalisation
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_layernorm,
    tensor_matmul,
    tensor_mul,
    tensor_neg,
    tensor_softmax,
    tensor_symmetric_quant,
    tensor_view,
)

from .mla_prolog_quant import rope_2d


@pl.function
def rotate_half(x: pl.Tensor,
                output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """RoPE rotate_half: [-x[..., N//2:], x[..., :N//2]]."""
    M, N = x.shape[0], x.shape[1]
    half = N // 2

    first = pl.slice(x, [M, half], [0, 0])
    second = pl.slice(x, [M, half], [0, half])

    neg_second = tensor_neg(second)
    pl.assemble(output, neg_second, [0, 0])
    pl.assemble(output, first, [0, half])
    return output


@pl.function
def lightning_indexer_prolog_quant_compute(
    x: pl.Tensor,
    # Indexer Q weights
    w_qb: pl.Tensor,
    w_qb_scale: pl.Tensor,
    # Indexer K weights
    wk: pl.Tensor,
    ln_gamma_k: pl.Tensor,
    ln_beta_k: pl.Tensor,
    # Weight projection
    w_proj: pl.Tensor,
    # RoPE
    cos: pl.Tensor,
    sin: pl.Tensor,
    # Hadamard
    hadamard_q: pl.Tensor,
    hadamard_k: pl.Tensor,
    # Outputs
    q_int8_out: pl.Out[pl.Tensor],
    q_scale_out: pl.Out[pl.Tensor],
    k_int8_out: pl.Out[pl.Tensor],
    k_scale_out: pl.Out[pl.Tensor],
    weights_out: pl.Out[pl.Tensor],
    # Config
    epsilon: float = 1e-5,
) -> None:
    """Lightning Indexer prolog: Q/K/weights computation with quantization.

    x: [bs, hidden_size]
    """
    bs = x.shape[0]

    # Q path: x → w_qb → dequant(scale) → RoPE → Hadamard → quantize
    w_qb_t = pl.transpose(w_qb)
    q_raw = tensor_matmul(x, w_qb_t)
    q_deq = tensor_mul(q_raw, w_qb_scale)

    q_dim = q_raw.shape[1]
    q_roped = pl.create_tensor([bs, q_dim], dtype=pl.FP32)
    rope_2d(q_deq,
            pl.slice(cos, [bs, q_dim], [0, 0]),
            pl.slice(sin, [bs, q_dim], [0, 0]),
            q_roped)

    q_had = tensor_matmul(q_roped, hadamard_q)
    tensor_symmetric_quant(q_had, q_int8_out, q_scale_out)

    # K path: x → wk → LayerNorm → RoPE → Hadamard → quantize
    wk_t = pl.transpose(wk)
    k_raw = tensor_matmul(x, wk_t)
    k_normed = tensor_layernorm(k_raw, ln_gamma_k, ln_beta_k, eps=epsilon)

    k_dim = k_raw.shape[1]
    k_roped = pl.create_tensor([bs, k_dim], dtype=pl.FP32)
    rope_2d(k_normed,
            pl.slice(cos, [bs, k_dim], [0, 0]),
            pl.slice(sin, [bs, k_dim], [0, 0]),
            k_roped)

    k_had = tensor_matmul(k_roped, hadamard_k)
    tensor_symmetric_quant(k_had, k_int8_out, k_scale_out)

    # Weights path: x → w_proj → softmax
    w_proj_t = pl.transpose(w_proj)
    weights_raw = tensor_matmul(x, w_proj_t)
    weights_norm = tensor_softmax(weights_raw)
    pl.assemble(weights_out, weights_norm, [0, 0])
