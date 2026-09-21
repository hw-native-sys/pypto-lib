# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SWA decode Attention half-layer composition entry."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; keep this standalone composition in the device CI entry set.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash import decode_attn_swa as swa
from models.deepseek_v4_1_flash import decode_common as common

# PyPTO resolves inline calls and JIT-body constants only from direct names.
from models.deepseek_v4_1_flash.config import (
    D,
    DECODE_MAX_TOKENS,
    HEAD_DIM,
    LOCAL_H,
    LOCAL_O_WIDTH,
    Q_LORA,
    TP_SIZE,
)
from models.deepseek_v4_1_flash.decode_attn_swa import decode_attn_swa
from models.deepseek_v4_1_flash.decode_common import attention_pre
from models.deepseek_v4_1_flash.decode_layer_plan import (
    DecodeLayerKind,
    REPRESENTATIVE_LAYER_IDS,
    resolve_decode_layer_plan,
)
from models.deepseek_v4_1_flash.hc_post import mhc_post

KIND = DecodeLayerKind.SWA
REPRESENTATIVE_LAYER_ID = REPRESENTATIVE_LAYER_IDS[KIND]
ATTENTION_GOLDEN = swa.golden_decode_attn_swa
KERNEL_READY = True
LEAF_NAMES = tuple(name for name in swa.INPUT_NAMES if name != "x")
ATTENTION_SPEC_NAMES = (
    *common.BOUNDARY_PREFIX_NAMES,
    *LEAF_NAMES,
    *common.BOUNDARY_OUTPUT_NAMES,
    *common.SCALAR_NAMES,
)


def skip_reason(layer_id=REPRESENTATIVE_LAYER_ID):
    kind = resolve_decode_layer_plan(layer_id).kind
    if kind != KIND:
        return f"layer {layer_id} resolves to {kind.name}, expected {KIND.name}"
    return None if KERNEL_READY else "SWA attention half-layer composition is pending"


@pl.jit.inline(auto_scope=False)
def decode_swa(
    x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0],
    attention_output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    attention_hidden: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    attention_pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one SWA attention sublayer and emit the staggered mix for the next sublayer."""
    tokens = pl.tensor.dim(x_hc, 0)
    attention_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    normalized_attention = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post_mix, residual_mix = attention_pre(
        x_hc,
        incoming_pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
        attention_input,
        normalized_attention,
        attention_pre_mix,
        num_tokens,
    )
    group_base = rank // TP_SIZE * TP_SIZE
    tp_rank = rank % TP_SIZE
    decode_attn_swa(
        normalized_attention,
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
        output_window,
        output_arrived,
        attention_output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    mhc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden)
    return attention_hidden, attention_pre_mix


@pl.jit
def decode_swa_rank(
    x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
    ],
    attention_output: pl.InOut[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32]],
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Dispatch one production SWA composition to a device-selected JIT entry."""
    return decode_swa(
        x_hc,
        incoming_pre_mix,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_weight,
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
        attention_output,
        attention_hidden,
        attention_pre_mix,
        output_window,
        output_arrived,
        rank,
        num_tokens,
        attention_epoch,
    )


def make_program(world_size, epochs, specs):
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be in [1, 1000]")
    tensor_specs = common.check_program_specs(world_size, specs, ATTENTION_SPEC_NAMES)

    @pl.jit.host
    def attention_group(
        x_hc: pl.Tensor[tensor_specs.x_hc.shape, tensor_specs.x_hc.dtype],
        incoming_pre_mix: pl.Tensor[tensor_specs.incoming_pre_mix.shape, tensor_specs.incoming_pre_mix.dtype],
        hc_attn_fn: pl.Tensor[tensor_specs.hc_attn_fn.shape, tensor_specs.hc_attn_fn.dtype],
        hc_attn_scale: pl.Tensor[tensor_specs.hc_attn_scale.shape, tensor_specs.hc_attn_scale.dtype],
        hc_attn_base: pl.Tensor[tensor_specs.hc_attn_base.shape, tensor_specs.hc_attn_base.dtype],
        attn_norm_weight: pl.Tensor[tensor_specs.attn_norm_weight.shape, tensor_specs.attn_norm_weight.dtype],
        wq_a: pl.Tensor[tensor_specs.wq_a.shape, tensor_specs.wq_a.dtype],
        wq_a_scale: pl.Tensor[tensor_specs.wq_a_scale.shape, tensor_specs.wq_a_scale.dtype],
        q_norm_weight: pl.Tensor[tensor_specs.q_norm_weight.shape, tensor_specs.q_norm_weight.dtype],
        wq_b: pl.Tensor[tensor_specs.wq_b.shape, tensor_specs.wq_b.dtype],
        wq_b_scale: pl.Tensor[tensor_specs.wq_b_scale.shape, tensor_specs.wq_b_scale.dtype],
        wkv: pl.Tensor[tensor_specs.wkv.shape, tensor_specs.wkv.dtype],
        wkv_scale: pl.Tensor[tensor_specs.wkv_scale.shape, tensor_specs.wkv_scale.dtype],
        kv_norm_weight: pl.Tensor[tensor_specs.kv_norm_weight.shape, tensor_specs.kv_norm_weight.dtype],
        attn_sink: pl.Tensor[tensor_specs.attn_sink.shape, tensor_specs.attn_sink.dtype],
        wo_a: pl.Tensor[tensor_specs.wo_a.shape, tensor_specs.wo_a.dtype],
        wo_b: pl.Tensor[tensor_specs.wo_b.shape, tensor_specs.wo_b.dtype],
        wo_b_scale: pl.Tensor[tensor_specs.wo_b_scale.shape, tensor_specs.wo_b_scale.dtype],
        rope_cos: pl.Tensor[tensor_specs.rope_cos.shape, tensor_specs.rope_cos.dtype],
        rope_sin: pl.Tensor[tensor_specs.rope_sin.shape, tensor_specs.rope_sin.dtype],
        window_slots: pl.Tensor[tensor_specs.window_slots.shape, tensor_specs.window_slots.dtype],
        window_indices: pl.Tensor[tensor_specs.window_indices.shape, tensor_specs.window_indices.dtype],
        window_cache: pl.InOut[pl.Tensor[tensor_specs.window_cache.shape, tensor_specs.window_cache.dtype]],
        window_cache_scale: pl.InOut[
            pl.Tensor[tensor_specs.window_cache_scale.shape, tensor_specs.window_cache_scale.dtype]
        ],
        attention_output: pl.InOut[
            pl.Tensor[tensor_specs.attention_output.shape, tensor_specs.attention_output.dtype]
        ],
        attention_hidden: pl.Out[
            pl.Tensor[tensor_specs.attention_hidden.shape, tensor_specs.attention_hidden.dtype]
        ],
        attention_pre_mix: pl.Out[
            pl.Tensor[tensor_specs.attention_pre_mix.shape, tensor_specs.attention_pre_mix.dtype]
        ],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        data_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for step in pl.range(epochs):
            for rank in pl.range(world_size):
                data = pld.window(data_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
                arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
                # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
                wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
                wq_b_scale_r: pl.Tensor[
                    [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
                ] = wq_b_scale[rank]
                wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
                wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
                decode_swa_rank(
                    x_hc[rank],
                    incoming_pre_mix[rank],
                    hc_attn_fn[rank],
                    hc_attn_scale[rank],
                    hc_attn_base[rank],
                    attn_norm_weight[rank],
                    wq_a[rank],
                    wq_a_scale_r,
                    q_norm_weight[rank],
                    wq_b[rank],
                    wq_b_scale_r,
                    wkv[rank],
                    wkv_scale_r,
                    kv_norm_weight[rank],
                    attn_sink[rank],
                    wo_a[rank],
                    wo_b[rank],
                    wo_b_scale_r,
                    rope_cos[rank],
                    rope_sin[rank],
                    window_slots[rank],
                    window_indices[rank],
                    window_cache[rank],
                    window_cache_scale[rank],
                    attention_output[rank],
                    attention_hidden[rank],
                    attention_pre_mix[rank],
                    data,
                    arrived,
                    rank,
                    num_tokens,
                    attention_epoch + step,
                    device=rank,
                )

    return attention_group


def build_specs(args, initial_state=None):
    del initial_state
    return common.assemble_specs(args, swa.build_specs(args, "decode"), ATTENTION_SPEC_NAMES)


def make_golden(epochs):
    def golden_half(tensors):
        for _ in range(epochs):
            normalized, post, residual = common.golden_attention_pre(tensors)
            leaf = dict(tensors, x=normalized, output=tensors["attention_output"])
            swa.golden_swa(leaf)
            common.golden_attention_post(tensors, post, residual)

    return golden_half


def comparisons(initial_state=None):
    del initial_state
    compare = common.make_boundary_comparisons(swa.compare_output)
    compare.update(
        {
            "attention_output": swa.compare_reduced,
            "window_cache": swa.compare_distributed_cache,
            "window_cache_scale": swa.compare_scales,
        }
    )
    return compare


def main():
    parser = common.make_parser(
        "DeepSeek V4.1 SWA decode Attention composition",
        REPRESENTATIVE_LAYER_ID,
        32,
        ("mixed", "shuffle", "prefix"),
    )
    args = parser.parse_args()
    devices = common.validate_args(parser, args)
    reason = skip_reason(args.layer_id)
    if reason:
        parser.error(reason)
    specs = build_specs(args)
    common.run_attention(
        args,
        make_program(args.tp, args.epochs, specs),
        specs,
        make_golden(args.epochs),
        comparisons(),
        KIND.name,
        devices,
    )


if __name__ == "__main__":
    main()
