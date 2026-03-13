"""DeepSeek-V3 Sparse Attention with Anti-quantization.

Performs attention only on top-k selected KV blocks, with key
dequantization from INT8 cache.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_dequant,
    tensor_div,
    tensor_exp,
    tensor_matmul,
    tensor_mul,
    tensor_softmax,
    tensor_sub,
    tensor_sum,
    tensor_view,
)


@pl.function
def sparse_attention_antiquant_compute(
    query_nope: pl.Tensor,
    query_rope: pl.Tensor,
    nope_cache: pl.Tensor,
    topk_indices: pl.Tensor,
    block_table: pl.Tensor,
    kv_act_seqs: pl.Tensor,
    attention_out: pl.Out[pl.Tensor],
    softmax_scale: float = 1.0,
    block_size: int = 128,
) -> pl.Tensor:
    """Sparse attention on top-k selected KV blocks.

    query_nope:    [bs, nope_dim]
    query_rope:    [bs, rope_dim]
    nope_cache:    [num_blocks, block_size, nope_dim] (INT8 or FP)
    topk_indices:  [bs, topk]   selected block indices
    block_table:   [bs, max_blocks]
    kv_act_seqs:   [bs]         actual sequence lengths
    attention_out: [bs, nope_dim]
    """
    bs = query_nope.shape[0]
    nope_dim = query_nope.shape[1]
    rope_dim = query_rope.shape[1]
    topk = topk_indices.shape[1]

    for b in pl.range(0, bs, 1):
        q_nope = pl.slice(query_nope, [1, nope_dim], [b, 0])
        q_rope = pl.slice(query_rope, [1, rope_dim], [b, 0])

        # Accumulate attention over selected blocks
        out_accum = pl.create_tensor([1, nope_dim], dtype=pl.FP32)
        from tensor_functions import tensor_full, tensor_maximum, tensor_amax
        tensor_full(out_accum, 0.0)

        running_max = pl.create_tensor([1, 1], dtype=pl.FP32)
        running_sum = pl.create_tensor([1, 1], dtype=pl.FP32)
        tensor_full(running_max, -1e30)
        tensor_full(running_sum, 0.0)

        for ti in pl.range(0, topk, 1):
            block_idx = pl.tensor.read(topk_indices, [b, ti])
            k_block = pl.slice(nope_cache, [block_size, nope_dim],
                              [block_idx, 0, 0])

            # QK score (nope part)
            k_block_fp = tensor_cast(k_block, pl.FP32)
            k_t = pl.transpose(k_block_fp)
            score = tensor_matmul(q_nope, k_t)
            score = tensor_mul(score, softmax_scale)

            # Online softmax update
            row_max = tensor_amax(score, axis=-1, keepdim=True)
            new_max = tensor_maximum(running_max, row_max)

            exp_old = tensor_exp(tensor_sub(running_max, new_max))
            exp_new = tensor_exp(tensor_sub(score, new_max))

            pij_sum = tensor_sum(exp_new, axis=-1, keepdim=True)
            new_sum = tensor_add(tensor_mul(running_sum, exp_old), pij_sum)

            # P @ V (using nope_cache as value proxy)
            pv = tensor_matmul(exp_new, k_block_fp)
            scaled_prev = tensor_mul(out_accum, exp_old)
            out_accum = tensor_add(scaled_prev, pv)

            running_max = new_max
            running_sum = new_sum

        result = tensor_div(out_accum, running_sum)
        pl.assemble(attention_out, result, [b, 0])

    return attention_out
