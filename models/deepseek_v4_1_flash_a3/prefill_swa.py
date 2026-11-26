# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-sharded packed-prefill SWA with local mHC and head-TP attention."""


import pypto.language as pl
from .activation_quant import quantize_projection_activation
import pypto.language.distributed as pld
from .sparse_attention import attend_shared_cache_direct, decode_shared_cache_after_publish, prefill_publish_window
from .config import WINDOW_CACHE_GROUP, FLASH as M, TP_SIZE
from .dsl_types import ORI_BLOCKS_DYN, T_DYN
from .qkv_proj_rope import q_proj_rope, kv_proj_rope
from .o_proj import prefill_o_proj_nz as o_proj


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank

D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups


@pl.jit.inline(auto_scope=False)
def prefill_swa_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write BF16 head-TP partials for packed causal SWA; write slots must be unique."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_prefill_previous_epoch", allow_early_resolve=False) as cache_ready:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    tokens = pl.tensor.dim(x, 0)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    # Gate every projection tile before scratch storage is reused by a new epoch.
    begin_workers = pl.max(1, pl.min(num_tokens, pl.system.available_aiv_count()))
    with pl.spmd(begin_workers, name_hint="prefill_swa_begin", deps=[cache_ready]):
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, num_tokens, begin_workers):
            kv_projection[row : row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_input_payload, kv_input_scales = quantize_projection_activation(x, num_tokens, D, True, True)
    kv_proj_rope(
        kv_input_payload, kv_input_scales,
        wkv, wkv_scale, kv_norm_weight,
        rope_cos, rope_sin,
        kv_projection, kv,
        num_tokens,
    )
    published = prefill_publish_window(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
    cache_rows = pl.tensor.dim(window_cache, 0) * 128
    key_rows = cache_rows + pl.max(1, num_tokens) * 128
    keys = pl.create_tensor([key_rows, HEAD_DIM], dtype=pl.BF16)
    zero_row = pl.create_tensor([32, HEAD_DIM], dtype=pl.BF16)
    decoded_ready = decode_shared_cache_after_publish(window_cache, window_cache_scale, keys, zero_row, published)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    q_proj_rope(
        kv_input_payload, kv_input_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q, num_tokens,
    )

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attend_shared_cache_direct(
        q, keys, cache_rows, zero_row,
        window_indices, attn_sink,
        attended, num_tokens, decoded_ready,
    )
    o_proj(attended, wo_a, wo_b, rope_cos, rope_sin, partial, num_tokens)
    return partial
