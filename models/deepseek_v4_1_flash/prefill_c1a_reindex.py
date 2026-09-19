# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C1A reindex attention."""

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    if not any(arg == "--tp" or arg.startswith("--tp=") for arg in sys.argv):
        sys.argv.extend(["--tp", "2"])

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.prefill_c1a_common import (
    make_norm,
    make_projection,
    prefill_c1a_partial,
)
from models.deepseek_v4_1_flash.prefill_c1a_indexer import TOPK_LEAF, make_paged_indexer
from models.deepseek_v4_1_flash.prefill_c1a_test_utils import (
    CASE_DEFAULT,
    CASE_MAX_TOKENS,
    CASE_NAMES,
    CASE_TOKENS,
    CACHE_MAX_RELATIVE_L2,
    COMMON_INPUT_NAMES,
    apply_distributed_golden,
    attention_output_compare,
    golden_prefill_c1a_attention,
    make_tensor_specs,
    topk_indices_compare,
)
from models.deepseek_v4_1_flash.attention_common import (
    AttentionGoldenResult,
    golden_compressed_attention,
    quantized_cache_compare,
)
from models.deepseek_v4_1_flash.attention_tp import prefill_tp_output_all_reduce
from models.deepseek_v4_1_flash.config import AttentionMode


D = C.D
Q_LORA = C.Q_LORA
TP_SIZE = C.TP_SIZE
PREFILL_MAX_TOKENS = C.PREFILL_MAX_TOKENS

REINDEX_INPUT_NAMES = COMMON_INPUT_NAMES + (
    "request_ids",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "candidate_mask",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
)


if TP_SIZE not in (1, 2, 4):
    raise ValueError("Prefill C1A currently supports TP1, TP2, and TP4; TP8 requires head-tile padding")

project_index_latent = make_projection(C.D, C.Q_LORA)
normalize_index_latent = make_norm(C.Q_LORA)
paged_indexer = make_paged_indexer(use_candidates=True)
paged_indexer_direct = make_paged_indexer(use_candidates=True, direct_topk=True)


def golden_prefill_c1a_reindex(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    request_ids: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    candidate_mask: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Return one rank's FP32 partial output and reference cache state."""
    return golden_compressed_attention(
        mode=AttentionMode.REINDEX,
        ratio=1,
        x=x,
        wq_a=wq_a,
        wq_a_scale=wq_a_scale,
        q_norm_weight=q_norm_weight,
        wq_b=wq_b,
        wq_b_scale=wq_b_scale,
        wkv=wkv,
        wkv_scale=wkv_scale,
        kv_norm_weight=kv_norm_weight,
        attn_sink=attn_sink,
        wo_a=wo_a,
        wo_b=wo_b,
        wo_b_scale=wo_b_scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        window_slots=window_slots,
        window_indices=window_indices,
        window_cache=window_cache,
        window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        compressed_indices=None,
        compressor_wkv=None,
        compressor_wgate=None,
        compressor_norm_weight=None,
        state_block_table=None,
        state_cache=None,
        compressed_slots=None,
        position_ids=None,
        compressed_lens=compressed_lens,
        compressed_rope_cos=None,
        compressed_rope_sin=None,
        index_wk=None,
        index_norm_weight=None,
        index_wq_b=index_wq_b,
        index_wq_b_scale=index_wq_b_scale,
        index_weights_proj=index_weights_proj,
        index_cache=index_cache,
        index_cache_scale=index_cache_scale,
        index_block_table=index_block_table,
        request_ids=request_ids,
        candidate_mask=candidate_mask,
        attention_fn=golden_prefill_c1a_attention,
        output_dtype=torch.float32,
    )


def make_prefill_c1a_reindex(indexer):
    @pl.jit.inline(auto_scope=False)
    def prefill_c1a_reindex_impl(
        x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN],
        window_cache_scale: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0],
        compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
        compressed_cache_scale: pl.Tensor[
            [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
        ],
        request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
        index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
        index_cache_scale: pl.Tensor[
            [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
        ],
        index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
        candidate_mask: pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8],
        index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
        output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
        output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Refresh ratio-1 Top-K rows inside a supplied candidate set."""
        tokens = pl.tensor.dim(x, 0)
        positions = pl.tensor.dim(candidate_mask, 1)

        index_projection_a = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        project_index_latent(x, wq_a, wq_a_scale, index_projection_a, num_tokens)
        query_latent = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        normalize_index_latent(index_projection_a, q_norm_weight, query_latent, num_tokens)
        score_width = (positions + TOPK_LEAF - 1) // TOPK_LEAF * TOPK_LEAF
        index_scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
        cache_ready = pl.system.task_dummy(deps=[])
        indexer(
            x,
            query_latent,
            request_ids,
            compressed_lens,
            index_cache,
            index_cache_scale,
            index_block_table,
            rope_cos,
            rope_sin,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            candidate_mask,
            index_scores,
            topk_indices,
            num_tokens,
            cache_ready,
        )

        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        prefill_c1a_partial(
            x,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            rope_cos,
            rope_sin,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            compressed_cache,
            compressed_cache_scale,
            topk_indices,
            partial,
            num_tokens,
        )
        prefill_tp_output_all_reduce(
            partial,
            output_window,
            output_arrived,
            output,
            group_base,
            tp_rank,
            num_tokens,
            attention_epoch,
        )
        return output

    return prefill_c1a_reindex_impl


prefill_c1a_reindex = make_prefill_c1a_reindex(paged_indexer)
prefill_c1a_reindex_watch = make_prefill_c1a_reindex(paged_indexer_direct)


@pl.jit
def prefill_c1a_reindex_test(
    x: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP],
        pl.FP8E4M3FN,
    ],
    request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP],
        pl.FP8E8M0,
    ],
    index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8],
    index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM],
        pl.FP8E8M0,
        pl.MX_B_NN,
    ],
    index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
    output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    return prefill_c1a_reindex(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table, candidate_mask,
        index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices,
        output_window, output_arrived, output, 0, tp_rank, num_tokens, 1,
    )


@pl.jit.host
def l3_prefill_c1a_reindex_test(
    x: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.D], pl.BF16],
    wq_a: pl.Tensor[[C.TP_SIZE, C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.TP_SIZE, C.D // 32, C.Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[C.TP_SIZE, C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.TP_SIZE, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[
        [C.TP_SIZE, C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM],
        pl.FP8E8M0,
    ],
    wkv: pl.Tensor[[C.TP_SIZE, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.TP_SIZE, C.D // 32, C.HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[C.TP_SIZE, C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.TP_SIZE, C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.TP_SIZE, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.TP_SIZE, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[
        [C.TP_SIZE, C.LOCAL_O_WIDTH // 32, C.D],
        pl.FP8E8M0,
    ],
    rope_cos: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.TP_SIZE, C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.TP_SIZE, C.T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[
        pl.Tensor[[C.TP_SIZE, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[
            [C.TP_SIZE, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP],
            pl.FP8E8M0,
        ]
    ],
    compressed_cache: pl.Tensor[
        [C.TP_SIZE, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2],
        pl.UINT8,
    ],
    compressed_cache_scale: pl.Tensor[
        [C.TP_SIZE, C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP],
        pl.FP8E4M3FN,
    ],
    request_ids: pl.Tensor[[C.TP_SIZE, C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.TP_SIZE, C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[
        [C.TP_SIZE, C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2],
        pl.UINT8,
    ],
    index_cache_scale: pl.Tensor[
        [C.TP_SIZE, C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP],
        pl.FP8E8M0,
    ],
    index_block_table: pl.Tensor[[C.TP_SIZE, C.B_DYN, C.TABLE_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[C.TP_SIZE, C.T_DYN, C.CMP_POSITIONS_DYN], pl.UINT8],
    index_wq_b: pl.Tensor[[C.TP_SIZE, C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [C.TP_SIZE, C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM],
        pl.FP8E8M0,
    ],
    index_weights_proj: pl.Tensor[[C.TP_SIZE, C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[C.TP_SIZE, C.T_DYN, C.D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output_window_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
    output_arrived_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        output_window = pld.window(output_window_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        output_arrived = pld.window(output_arrived_buf, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_c1a_reindex_test(
            x[rank], wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank], wkv[rank], wkv_scale[rank],
            kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank],
            wo_b_scale[rank], rope_cos[rank], rope_sin[rank], window_slots[rank],
            window_indices[rank], window_cache[rank], window_cache_scale[rank],
            compressed_cache[rank], compressed_cache_scale[rank], request_ids[rank],
            compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
            index_block_table[rank], candidate_mask[rank], index_wq_b[rank],
            index_wq_b_scale[rank], index_weights_proj[rank], topk_indices[rank],
            output_window, output_arrived, output[rank], rank, num_tokens, device=rank,
        )


def build_tensor_specs(token_count=CASE_TOKENS, case_name=CASE_DEFAULT):
    return make_tensor_specs(REINDEX_INPUT_NAMES, ("topk_indices", "output"), token_count, case_name)


def golden_prefill_c1a_reindex_case(tensors):
    apply_distributed_golden("reindex", golden_prefill_c1a_reindex, tensors)


__all__ = ["golden_prefill_c1a_reindex", "prefill_c1a_reindex"]


if __name__ == _SCRIPT_ENTRY_POINT:
    import argparse

    from pypto.ir import DistributedConfig

    from golden import run
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C1A reindex validation")
    parser.add_argument("-p", "--platform", default="a5", choices=("a5",))
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(C.TP_SIZE)))
    parser.add_argument("--tp", type=int, default=2, choices=(1, 2, 4))
    parser.add_argument("--dp", type=int, default=1, choices=(1,))
    parser.add_argument("--tokens", type=int, default=CASE_TOKENS, help="token count for the causal case")
    parser.add_argument("--case", default=CASE_DEFAULT, choices=CASE_NAMES)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    if args.tp != C.TP_SIZE:
        parser.error(f"--tp was parsed as TP{C.TP_SIZE}, got --tp {args.tp}")
    if args.case == "causal" and not 1 <= args.tokens <= CASE_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {CASE_MAX_TOKENS}]")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != C.TP_SIZE:
        parser.error(f"need exactly {C.TP_SIZE} devices, got {device_ids}")

    window_cache_compare = quantized_cache_compare(
        "window_cache",
        "window_cache_scale",
        "window_slots",
        CACHE_MAX_RELATIVE_L2,
    )
    result = run(
        fn=l3_prefill_c1a_reindex_test,
        specs=build_tensor_specs(args.tokens, args.case),
        golden_fn=golden_prefill_c1a_reindex_case,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config={
            "platform": args.platform,
            "distributed_config": DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            "dump_passes": args.dump_passes,
            "enable_chip_swimlane": args.enable_chip_swimlane,
        },
        compare_fn={
            "output": attention_output_compare("reindex"),
            "window_cache": window_cache_compare,
            "window_cache_scale": window_cache_compare,
            "topk_indices": topk_indices_compare("reindex"),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
