# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode SWA leaf attention for encoder layers 0 and 1."""


import pypto.language as pl
import pypto.language.distributed as pld
import torch


from config import (
    FLASH as M,
    DECODE_MAX_TOKENS,
    WINDOW_CACHE_GROUP,
    ORI_BLOCKS_DYN,
    T_DYN,
    TP_SIZE,
    MAX_BATCH_PER_DP,
)


from o_proj import o_proj
from qkv_proj_rope import qkv_proj_rope
from config import LOCAL_T_DYN
from o_proj import (
    decode_tp_output_all_reduce,
    decode_tp_output_reduce_scatter,
)
from utils import AttentionGoldenResult, run_swa_validation, swa_window_cache_compare as window_cache_compare
from sparse_attention import (
    build_swa_leaf_specs, golden_swa_attention, golden_swa_leaf, make_swa_program, swa_attention_partial,
)


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


SOFTMAX_SCALE = HEAD_DIM ** -0.5


@pl.jit.inline
def publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(num_tokens, name_hint="swa_cache_publish", deps=[cache_ready]) as publish_tid:
        t = pl.tile.get_block_idx()
        slot_i64 = pl.read(slots, [t])
        if slot_i64 >= 0:
            slot = pl.cast(slot_i64, pl.INDEX)
            source = pl.slice(kv, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
            source_padded = pl.fillpad(source, pad_value=pl.PadValue.zero)
            source = pl.set_validshape(source_padded, 1, HEAD_DIM * 2)
            source_fp32 = pl.cast(source, pl.FP32)
            value = pl.reshape(source_fp32, [HEAD_DIM // 16, 32])
            absolute = pl.abs(value)
            row_maximum = pl.row_max(absolute)
            amax = pl.maximum(row_maximum, 1e-4)
            raw_scale = pl.mul(amax, 1.0 / 448.0)
            scale = pl.cast(raw_scale, pl.BF16, mode="rint")
            scale_fp32 = pl.cast(scale, pl.FP32)
            normalized = pl.row_expand_div(value, scale_fp32)
            payload = pl.cast(normalized, pl.FP8E4M3FN, mode="rint")
            payload_row = pl.reshape(payload, [1, HEAD_DIM * 2])
            flat[slot:slot + 1, :] = pl.set_validshape(payload_row, 1, HEAD_DIM)
            scale_row = pl.reshape(scale, [1, HEAD_DIM // 16])
            scale_flat[slot:slot + 1, :] = pl.set_validshape(scale_row, 1, HEAD_DIM // 32)
    return cache, scales


if TP_SIZE not in (1, 2, 4):
    raise ValueError("Decode SWA currently supports TP1, TP2, and TP4; TP8 requires head-tile padding")


@pl.jit.inline(auto_scope=False)
def decode_swa_partial(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Write FP32 local output; return conservative attention completion for caller WAR ordering."""
    tokens = pl.tensor.dim(x, 0)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qkv_proj_rope(
        x,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight,
        rope_cos, rope_sin,
        qr, q, kv,
        num_tokens,
    )
    publish_window(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    # Return attention completion before cache storage is reused.
    cache_consumed = swa_attention_partial(
        q, window_cache, window_cache_scale, window_indices, attn_sink, attended, num_tokens,
    )
    o_proj(attended, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, output, num_tokens)
    return cache_consumed


def make_decode_attn_swa(output_reduce=None):
    """Build the decode_attn_swa leaf around one output collective."""
    reducer = output_reduce or decode_tp_output_all_reduce

    @pl.jit.inline(auto_scope=False)
    def decode_attn_swa(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
        output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Write BF16 TP output using zero-initialized windows and consecutive 1-based epochs."""
        # A later epoch must not overwrite cache or transport storage still being read.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_previous_epoch", allow_early_resolve=False) as cache_ready:
            for peer in pl.range(TP_SIZE):
                pld.system.wait(
                    output_arrived,
                    offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
                )
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        decode_swa_partial(
            x,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            partial, num_tokens, cache_ready,
        )
        reducer(partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch)
        return output

    return decode_attn_swa


decode_attn_swa = make_decode_attn_swa()
decode_attn_swa_sharded = make_decode_attn_swa(decode_tp_output_reduce_scatter)


def golden_decode_attn_swa(
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
) -> AttentionGoldenResult:
    return golden_swa_attention(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale,
        rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
    )


__all__ = ["decode_attn_swa", "decode_attn_swa_sharded", "golden_decode_attn_swa", "window_cache_compare"]


if __name__ == "__main__":
    result = run_swa_validation(
        decode_attn_swa, 'decode', None,
        capacity=DECODE_MAX_TOKENS, default_tokens=MAX_BATCH_PER_DP, default_requests=MAX_BATCH_PER_DP,
        cases=('mixed', 'prefix', 'shuffle', 'masked', 'zero', 'sink'),
        program_builder=make_swa_program, spec_builder=build_swa_leaf_specs, golden_fn=golden_swa_leaf,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
