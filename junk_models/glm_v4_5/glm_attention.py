"""GLM-4.5 Flash Attention with paged KV cache.

Implements scaled dot-product attention using online softmax and paged
memory management for variable-length sequences.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_div,
    tensor_exp,
    tensor_matmul,
    tensor_maximum,
    tensor_mul,
    tensor_softmax,
    tensor_sub,
    tensor_sum,
    tensor_view,
)


@pl.function
def flash_attention_block(
    query: pl.Tensor,
    key_block: pl.Tensor,
    value_block: pl.Tensor,
    prev_max: pl.Tensor,
    prev_sum: pl.Tensor,
    prev_out: pl.Tensor,
    softmax_scale: float,
    new_max_out: pl.Out[pl.Tensor],
    new_sum_out: pl.Out[pl.Tensor],
    new_out: pl.Out[pl.Tensor],
) -> None:
    """One block of online-softmax Flash Attention.

    query:       [bs, head_dim]
    key_block:   [block_size, head_dim]
    value_block: [block_size, head_dim]
    prev_max/sum/out: running accumulators

    Computes incremental softmax update for one KV block.
    """
    # S = Q @ K^T * scale
    key_t = pl.transpose(key_block)
    sij = tensor_matmul(query, key_t)
    sij_scaled = tensor_mul(sij, softmax_scale)

    # Online softmax: new_max = max(prev_max, row_max(sij))
    from tensor_functions import tensor_amax
    row_max_sij = tensor_amax(sij_scaled, axis=-1, keepdim=True)
    new_max = tensor_maximum(prev_max, row_max_sij)
    pl.assemble(new_max_out, new_max, [0, 0])

    # Correction factors
    exp_old = tensor_exp(tensor_sub(prev_max, new_max))
    exp_new = tensor_exp(tensor_sub(sij_scaled, new_max))

    # P = exp(S - new_max)
    pij = exp_new

    # Update sum
    pij_sum = tensor_sum(pij, axis=-1, keepdim=True)
    new_sum = tensor_add(tensor_mul(prev_sum, exp_old), pij_sum)
    pl.assemble(new_sum_out, new_sum, [0, 0])

    # Update output: out = exp_old * prev_out + P @ V
    pv = tensor_matmul(pij, value_block)
    scaled_prev = tensor_mul(prev_out, exp_old)
    new_out_val = tensor_add(scaled_prev, pv)
    pl.assemble(new_out, new_out_val, [0, 0])


@pl.function
def attention(
    query: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    block_tables: pl.Tensor,
    actual_seqs: pl.Tensor,
    attn_res: pl.Out[pl.Tensor],
    softmax_scale: float = 1.0,
    block_size: int = 128,
) -> pl.Tensor:
    """Paged Flash Attention main entry.

    query:       [batch*heads, 1, head_dim]
    key_cache:   [num_blocks, block_size, head_dim]
    value_cache: [num_blocks, block_size, head_dim]
    block_tables:[batch, max_blocks]
    actual_seqs: [batch]
    attn_res:    [batch*heads, 1, head_dim]
    """
    # Simplified single-head, single-query-token attention
    B = query.shape[0]
    D = query.shape[2]

    for b in pl.range(0, B, 1):
        q = pl.slice(query, [1, D], [b, 0, 0])

        # Iterate over KV blocks (simplified: sequential over block_table)
        max_blocks = block_tables.shape[1]
        out_accum = pl.create_tensor([1, D], dtype=pl.FP32)
        running_max = pl.create_tensor([1, 1], dtype=pl.FP32)
        running_sum = pl.create_tensor([1, 1], dtype=pl.FP32)
        tensor_full(running_max, -1e30)
        tensor_full(running_sum, 0.0)
        tensor_full(out_accum, 0.0)

        for blk in pl.range(0, max_blocks, 1):
            block_id = pl.tensor.read(block_tables, [b, blk])
            k_block = pl.slice(key_cache, [block_size, D], [block_id, 0, 0])
            v_block = pl.slice(value_cache, [block_size, D], [block_id, 0, 0])

            new_max = pl.create_tensor([1, 1], dtype=pl.FP32)
            new_sum = pl.create_tensor([1, 1], dtype=pl.FP32)
            new_out = pl.create_tensor([1, D], dtype=pl.FP32)

            flash_attention_block(
                q, k_block, v_block,
                running_max, running_sum, out_accum,
                softmax_scale,
                new_max, new_sum, new_out,
            )
            running_max = new_max
            running_sum = new_sum
            out_accum = new_out

        # Normalise by sum
        result = tensor_div(out_accum, running_sum)
        result_cast = tensor_cast(result, query.dtype)
        pl.assemble(attn_res, result_cast, [b, 0, 0])

    return attn_res
