"""DeepSeek-V3 MLA Prolog with Quantization.

Implements Multi-head Latent Attention (MLA) pre-computation:
Q/K/V projections → RMSNorm → RoPE → key quantization → cache update.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_matmul,
    tensor_mul,
    tensor_rmsnorm,
    tensor_rope,
    tensor_sub,
    tensor_symmetric_quant,
    tensor_view,
)


@pl.function
def rotate_half(x: pl.Tensor,
                output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """RoPE rotate_half: [-x[..., N//2:], x[..., :N//2]]."""
    M, N = x.shape[0], x.shape[1]
    half = N // 2

    first = pl.slice(x, [M, half], [0, 0])
    second = pl.slice(x, [M, half], [0, half])

    from tensor_functions import tensor_neg
    neg_second = tensor_neg(second)

    pl.assemble(output, neg_second, [0, 0])
    pl.assemble(output, first, [0, half])
    return output


@pl.function
def rope_2d(x: pl.Tensor, cos: pl.Tensor, sin: pl.Tensor,
            output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """2D RoPE: x * cos + rotate_half(x) * sin."""
    M, N = x.shape[0], x.shape[1]
    x_rot = pl.create_tensor([M, N], dtype=x.dtype)
    rotate_half(x, x_rot)

    term1 = tensor_mul(x, cos)
    term2 = tensor_mul(x_rot, sin)
    result = tensor_add(term1, term2)
    pl.assemble(output, result, [0, 0])
    return output


@pl.function
def rms_norm(x: pl.Tensor, gamma: pl.Tensor,
             output: pl.Out[pl.Tensor],
             eps: float = 1e-6) -> pl.Tensor:
    """RMSNorm: gamma * x / sqrt(mean(x^2) + eps)."""
    result = tensor_rmsnorm(x, gamma, eps)
    pl.assemble(output, result, [0, 0])
    return output


@pl.function
def mla_prolog_quant_compute(
    token_x: pl.Tensor,
    w_dq: pl.Tensor,
    w_uq_qr: pl.Tensor,
    dequant_scale: pl.Tensor,
    w_uk: pl.Tensor,
    w_dkv_kr: pl.Tensor,
    gamma_cq: pl.Tensor,
    gamma_ckv: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    query_nope_out: pl.Out[pl.Tensor],
    query_rope_out: pl.Out[pl.Tensor],
    kv_out: pl.Out[pl.Tensor],
    kr_out: pl.Out[pl.Tensor],
    k_scale_out: pl.Out[pl.Tensor],
    epsilon_cq: float = 1e-6,
    epsilon_ckv: float = 1e-6,
) -> None:
    """MLA prolog: compute Q/K/V projections with RoPE and quantization.

    token_x:      [bs, hidden_size]
    w_dq:         [q_lora_rank, hidden_size]   down-projection for Q
    w_uq_qr:      [total_q_dim, q_lora_rank]   up-projection for Q
    w_dkv_kr:      [kv_lora_rank + rope_dim, hidden_size]  down-projection for KV
    w_uk:         [nope_dim, kv_lora_rank]       up-projection for K nope
    gamma_cq:     [1, q_lora_rank]               Q normalisation weight
    gamma_ckv:    [1, kv_lora_rank]              KV normalisation weight
    """
    bs = token_x.shape[0]
    hidden_size = token_x.shape[1]

    # Q path: token_x → w_dq → RMSNorm → w_uq_qr → split nope/rope
    w_dq_t = pl.transpose(w_dq)
    cq = tensor_matmul(token_x, w_dq_t)
    cq_normed = tensor_rmsnorm(cq, gamma_cq, epsilon_cq)

    w_uq_t = pl.transpose(w_uq_qr)
    q_all = tensor_matmul(cq_normed, w_uq_t)

    total_q = q_all.shape[1]
    rope_dim = cos.shape[1]
    nope_dim = total_q - rope_dim

    q_nope = pl.slice(q_all, [bs, nope_dim], [0, 0])
    q_rope_raw = pl.slice(q_all, [bs, rope_dim], [0, nope_dim])

    pl.assemble(query_nope_out, q_nope, [0, 0])

    # RoPE on Q rope part
    rope_2d(q_rope_raw, cos, sin, query_rope_out)

    # KV path: token_x → w_dkv_kr → split kv/kr
    w_dkv_t = pl.transpose(w_dkv_kr)
    ckv_all = tensor_matmul(token_x, w_dkv_t)

    kv_lora_rank = gamma_ckv.shape[1]
    ckv = pl.slice(ckv_all, [bs, kv_lora_rank], [0, 0])
    kr_raw = pl.slice(ckv_all, [bs, rope_dim], [0, kv_lora_rank])

    ckv_normed = tensor_rmsnorm(ckv, gamma_ckv, epsilon_ckv)

    # K nope via up-projection
    w_uk_t = pl.transpose(w_uk)
    k_nope = tensor_matmul(ckv_normed, w_uk_t)

    # KV output (compressed KV for cache)
    pl.assemble(kv_out, ckv_normed, [0, 0])

    # K rope with RoPE
    rope_2d(kr_raw, cos, sin, kr_out)

    # Quantize K nope for cache
    k_quant = pl.create_tensor([bs, k_nope.shape[1]], dtype=pl.INT8)
    tensor_symmetric_quant(k_nope, k_quant, k_scale_out)
