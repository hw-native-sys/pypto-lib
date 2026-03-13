"""DeepSeek-V3 MLA + Lightning Indexer Prolog (fused).

Fuses the MLA prolog (Q/K/V computation, RoPE, cache scatter) with the
Lightning Indexer prolog (Indexer Q/K/weights computation, quantization).
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_layernorm,
    tensor_matmul,
    tensor_mul,
    tensor_rmsnorm,
    tensor_symmetric_quant,
    tensor_view,
)

from .mla_prolog_quant import mla_prolog_quant_compute, rope_2d


@pl.function
def mla_indexer_prolog_quant_compute(
    token_x: pl.Tensor,
    # MLA weights
    mla_w_dq: pl.Tensor,
    mla_w_uq_qr: pl.Tensor,
    mla_dequant_scale: pl.Tensor,
    mla_w_uk: pl.Tensor,
    mla_w_dkv_kr: pl.Tensor,
    mla_gamma_cq: pl.Tensor,
    mla_gamma_ckv: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    # Indexer weights
    ip_w_qb: pl.Tensor,
    ip_w_qb_scale: pl.Tensor,
    ip_wk: pl.Tensor,
    ip_w_proj: pl.Tensor,
    ip_ln_gamma_k: pl.Tensor,
    ip_ln_beta_k: pl.Tensor,
    ip_hadamard_q: pl.Tensor,
    ip_hadamard_k: pl.Tensor,
    # MLA outputs
    mla_query_nope_out: pl.Out[pl.Tensor],
    mla_query_rope_out: pl.Out[pl.Tensor],
    mla_kv_out: pl.Out[pl.Tensor],
    mla_kr_out: pl.Out[pl.Tensor],
    mla_k_scale_out: pl.Out[pl.Tensor],
    # Indexer outputs
    ip_q_int8_out: pl.Out[pl.Tensor],
    ip_q_scale_out: pl.Out[pl.Tensor],
    ip_k_int8_out: pl.Out[pl.Tensor],
    ip_k_scale_out: pl.Out[pl.Tensor],
    ip_weights_out: pl.Out[pl.Tensor],
    # Config
    mla_epsilon_cq: float = 1e-6,
    mla_epsilon_ckv: float = 1e-6,
    ip_epsilon: float = 1e-5,
) -> None:
    """Fused MLA prolog + Lightning Indexer prolog.

    token_x: [bs, hidden_size]
    """
    bs = token_x.shape[0]

    # Part 1: MLA prolog
    mla_prolog_quant_compute(
        token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale,
        mla_w_uk, mla_w_dkv_kr, mla_gamma_cq, mla_gamma_ckv,
        cos, sin,
        mla_query_nope_out, mla_query_rope_out,
        mla_kv_out, mla_kr_out, mla_k_scale_out,
        mla_epsilon_cq, mla_epsilon_ckv,
    )

    # Part 2: Lightning Indexer prolog
    # Q path: token_x → w_qb → dequant → RoPE → Hadamard → quantize
    ip_w_qb_t = pl.transpose(ip_w_qb)
    ip_q_raw = tensor_matmul(token_x, ip_w_qb_t)
    ip_q_raw = tensor_mul(ip_q_raw, ip_w_qb_scale)

    ip_q_dim = ip_q_raw.shape[1]
    ip_cos = pl.slice(cos, [bs, ip_q_dim], [0, 0])
    ip_sin = pl.slice(sin, [bs, ip_q_dim], [0, 0])

    ip_q_roped = pl.create_tensor([bs, ip_q_dim], dtype=pl.FP32)
    rope_2d(ip_q_raw, ip_cos, ip_sin, ip_q_roped)

    # Hadamard transform (Q)
    ip_q_had = tensor_matmul(ip_q_roped, ip_hadamard_q)

    # Quantize Q
    tensor_symmetric_quant(ip_q_had, ip_q_int8_out, ip_q_scale_out)

    # K path: token_x → wk → LayerNorm → RoPE → Hadamard → quantize
    ip_wk_t = pl.transpose(ip_wk)
    ip_k_raw = tensor_matmul(token_x, ip_wk_t)

    ip_k_normed = tensor_layernorm(ip_k_raw, ip_ln_gamma_k, ip_ln_beta_k,
                                   eps=ip_epsilon)

    ip_k_dim = ip_k_raw.shape[1]
    ik_cos = pl.slice(cos, [bs, ip_k_dim], [0, 0])
    ik_sin = pl.slice(sin, [bs, ip_k_dim], [0, 0])

    ip_k_roped = pl.create_tensor([bs, ip_k_dim], dtype=pl.FP32)
    rope_2d(ip_k_normed, ik_cos, ik_sin, ip_k_roped)

    ip_k_had = tensor_matmul(ip_k_roped, ip_hadamard_k)
    tensor_symmetric_quant(ip_k_had, ip_k_int8_out, ip_k_scale_out)

    # Weights path: token_x → w_proj → normalise
    ip_w_proj_t = pl.transpose(ip_w_proj)
    ip_weights = tensor_matmul(token_x, ip_w_proj_t)
    from tensor_functions import tensor_softmax
    ip_weights_norm = tensor_softmax(ip_weights)
    pl.assemble(ip_weights_out, ip_weights_norm, [0, 0])
