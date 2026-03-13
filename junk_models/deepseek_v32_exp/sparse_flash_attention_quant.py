"""DeepSeek-V3 Sparse Flash Attention with Quantized Keys.

Similar to sparse_attention_antiquant but uses quantized key_nope with
per-token scale, and supports the flash-attention incremental softmax
variant.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_amax,
    tensor_cast,
    tensor_dequant,
    tensor_div,
    tensor_exp,
    tensor_full,
    tensor_matmul,
    tensor_maximum,
    tensor_mul,
    tensor_sub,
    tensor_sum,
    tensor_view,
)


@pl.function
def sparse_flash_attention_quant_compute(
    query_nope: pl.Tensor,
    query_rope: pl.Tensor,
    key_nope_cache: pl.Tensor,
    key_rope_cache: pl.Tensor,
    k_nope_scales: pl.Tensor,
    topk_indices: pl.Tensor,
    block_table: pl.Tensor,
    kv_act_seqs: pl.Tensor,
    attention_out: pl.Out[pl.Tensor],
    softmax_scale: float = 1.0,
    block_size: int = 128,
) -> pl.Tensor:
    """Sparse flash attention with quantized keys.

    query_nope:      [bs, nope_dim]
    query_rope:      [bs, rope_dim]
    key_nope_cache:  [num_blocks, block_size, nope_dim] (INT8)
    key_rope_cache:  [num_blocks, block_size, rope_dim]
    k_nope_scales:   [num_blocks, block_size, 1]
    topk_indices:    [bs, topk]
    attention_out:   [bs, nope_dim]
    """
    bs = query_nope.shape[0]
    nope_dim = query_nope.shape[1]
    rope_dim = query_rope.shape[1]
    topk = topk_indices.shape[1]

    for b in pl.range(0, bs, 1):
        q_nope = pl.slice(query_nope, [1, nope_dim], [b, 0])
        q_rope = pl.slice(query_rope, [1, rope_dim], [b, 0])

        out_accum = pl.create_tensor([1, nope_dim], dtype=pl.FP32)
        running_max = pl.create_tensor([1, 1], dtype=pl.FP32)
        running_sum = pl.create_tensor([1, 1], dtype=pl.FP32)
        tensor_full(out_accum, 0.0)
        tensor_full(running_max, -1e30)
        tensor_full(running_sum, 0.0)

        for ti in pl.range(0, topk, 1):
            block_idx = pl.tensor.read(topk_indices, [b, ti])

            # Dequantize key nope
            k_nope_int8 = pl.slice(key_nope_cache,
                                  [block_size, nope_dim],
                                  [block_idx, 0, 0])
            k_scale = pl.slice(k_nope_scales,
                              [block_size, 1],
                              [block_idx, 0, 0])
            k_nope_fp = tensor_dequant(k_nope_int8, k_scale)

            # Key rope
            k_rope = pl.slice(key_rope_cache,
                             [block_size, rope_dim],
                             [block_idx, 0, 0])

            # Nope score + rope score
            k_nope_t = pl.transpose(k_nope_fp)
            score_nope = tensor_matmul(q_nope, k_nope_t)

            k_rope_fp = tensor_cast(k_rope, pl.FP32)
            k_rope_t = pl.transpose(k_rope_fp)
            score_rope = tensor_matmul(q_rope, k_rope_t)

            score = tensor_add(score_nope, score_rope)
            score = tensor_mul(score, softmax_scale)

            # Online softmax
            row_max = tensor_amax(score, axis=-1, keepdim=True)
            new_max = tensor_maximum(running_max, row_max)

            exp_old = tensor_exp(tensor_sub(running_max, new_max))
            exp_new = tensor_exp(tensor_sub(score, new_max))

            pij_sum = tensor_sum(exp_new, axis=-1, keepdim=True)
            new_sum = tensor_add(tensor_mul(running_sum, exp_old), pij_sum)

            pv = tensor_matmul(exp_new, k_nope_fp)
            scaled_prev = tensor_mul(out_accum, exp_old)
            out_accum = tensor_add(scaled_prev, pv)

            running_max = new_max
            running_sum = new_sum

        result = tensor_div(out_accum, running_sum)
        pl.assemble(attention_out, result, [b, 0])

    return attention_out
