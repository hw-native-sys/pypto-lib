"""DeepSeek-V3 Lightning Indexer (decode).

Computes quantised QK scores, applies weights, and performs multi-level
top-k to select the most relevant KV blocks for sparse attention.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_dequant,
    tensor_matmul,
    tensor_mul,
    tensor_topk,
    tensor_view,
)


@pl.function
def lightning_indexer_decode_compute(
    idx_query_int8: pl.Tensor,
    idx_query_scale: pl.Tensor,
    idx_key_cache_int8: pl.Tensor,
    idx_key_scale_cache: pl.Tensor,
    idx_weight: pl.Tensor,
    act_seq_key: pl.Tensor,
    block_table: pl.Tensor,
    topk_res: pl.Out[pl.Tensor],
    selected_count: int = 8,
) -> pl.Tensor:
    """Lightning Indexer for decode: quantized QK → weighted → multi-level top-k.

    idx_query_int8:       [bs, idx_dim] (INT8)
    idx_query_scale:      [bs, 1]
    idx_key_cache_int8:   [num_blocks, block_size, idx_dim] (INT8)
    idx_key_scale_cache:  [num_blocks, block_size, 1]
    idx_weight:           [bs, n_heads]
    act_seq_key:          [bs]          actual sequence lengths
    block_table:          [bs, max_blocks]
    topk_res:             [bs, selected_count] output top-k indices
    """
    bs = idx_query_int8.shape[0]
    idx_dim = idx_query_int8.shape[1]
    max_blocks = block_table.shape[1]

    for b in pl.range(0, bs, 1):
        q_int8 = pl.slice(idx_query_int8, [1, idx_dim], [b, 0])
        q_scale = pl.slice(idx_query_scale, [1, 1], [b, 0])
        q_fp = tensor_dequant(q_int8, q_scale)

        # Compute scores for all blocks
        total_keys = max_blocks
        scores = pl.create_tensor([1, total_keys], dtype=pl.FP32)

        for blk in pl.range(0, max_blocks, 1):
            block_idx = pl.tensor.read(block_table, [b, blk])

            k_int8 = pl.slice(idx_key_cache_int8,
                             [1, idx_dim], [block_idx, 0, 0])
            k_scale = pl.slice(idx_key_scale_cache,
                              [1, 1], [block_idx, 0, 0])
            k_fp = tensor_dequant(k_int8, k_scale)

            k_t = pl.transpose(k_fp)
            block_score = tensor_matmul(q_fp, k_t)
            pl.assemble(scores, block_score, [0, blk])

        # Apply head weights
        w = pl.slice(idx_weight, [1, idx_weight.shape[1]], [b, 0])
        scores_weighted = tensor_mul(scores, w)

        # Multi-level top-k
        topk_vals = pl.create_tensor([1, selected_count], dtype=pl.FP32)
        topk_idx = pl.create_tensor([1, selected_count], dtype=pl.INT32)
        tensor_topk(scores_weighted, topk_vals, topk_idx,
                    k=selected_count, axis=-1)

        pl.assemble(topk_res, topk_idx, [b, 0])

    return topk_res
