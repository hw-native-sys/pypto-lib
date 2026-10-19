# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SWA decode Attention half-layer composition entry."""


import torch

import pypto.language as pl

from config import (
    FLASH as M,
    TP_SIZE,
    T_DYN,
    ORI_BLOCKS_DYN,
    WINDOW_CACHE_GROUP,
)


from sparse_attention import build_swa_leaf_specs, golden_swa_leaf
from golden import ratio_allclose

# PyPTO resolves inline calls and JIT-body constants only from direct names.
from config import LOCAL_T_DYN
from sparse_attention import publish_window_row
from qkv_proj_rope import q_proj_qr, q_proj_rope, kv_proj_rope
from sparse_attention import swa_attention_partial
from o_proj import o_proj
from utils import (
    assemble_specs,
    swa_window_cache_compare as window_cache_compare,
)
from hc import hc_pre, zero_bf16_padding, golden_hc_pre
from hc import golden_mhc_mixes, hc_mixes
from rmsnorm import golden_rms_norm, rms_norm_tiled
from hc import golden_hc_post, hc_post


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
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
MX_GROUP = M.weight_block_size[0]

# Tiling.
Q_ROPE_WORKER_TILE = 64 if TP_SIZE == 1 else 2147483647



@pl.jit.inline(auto_scope=False)
def decode_swa(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    attention_output: pl.Tensor[[LOCAL_T_DYN, D], pl.BF16],
    attention_hidden: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    attention_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one SWA attention sublayer and emit the staggered mix for the next sublayer."""
    tokens = pl.tensor.dim(x_hc, 0)
    x_attn = pl.create_tensor([tokens, D], dtype=pl.BF16)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_mixes(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix,
    )
    hc_pre(x_hc, incoming_pre_mix, x_attn)
    norm_tokens = pl.tensor.dim(x_attn, 0)
    for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(x_attn, attn_norm_weight, x_norm, norm_block, norm_tokens)
    zero_bf16_padding(x_norm, pl.min(num_tokens, pl.tensor.dim(x_norm, 0)))
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    q_proj_qr(x_norm, wq_a, wq_a_scale, q_norm_weight, qr, num_tokens)
    q_proj_rope(qr, wq_b, wq_b_scale, rope_cos, rope_sin, q, num_tokens, max_workers=Q_ROPE_WORKER_TILE)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_norm, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, kv, num_tokens)
    publish_window_row(kv, window_slots, window_cache, window_cache_scale, num_tokens)
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    swa_attention_partial(
        q, window_cache, window_cache_scale, window_indices, attn_sink, attended, num_tokens,
    )
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b,
        wo_b_scale, rope_cos, rope_sin,
        partial, num_tokens, grouped_max_workers=4096,
    )
    output_rows = pl.tensor.dim(attention_output, 0)
    for row in pl.spmd(output_rows, name_hint="swa_output_cast"):
        for col in pl.range(0, D, 512):
            value = pl.tile.full([1, 512], dtype=pl.BF16, value=0.0)
            if row < num_tokens:
                partial_tile = pl.load(partial, [row, col], [1, 512])
                value = pl.cast(partial_tile, pl.BF16, mode="rint")
            attention_output = pl.store(value, [row, col], attention_output)
    hc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden)
    return attention_hidden, attention_pre_mix


@pl.jit
def l2_decode_swa(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    attention_output: pl.InOut[pl.Tensor[[LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
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
        wo_a, wo_a_scale,
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
        num_tokens,
    )


@pl.jit.host
def l3_decode_swa(
    x_hc: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TP_SIZE, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TP_SIZE, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    attention_output: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, LOCAL_T_DYN)
    rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    # MX_B_NN weight-scale views.
    wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[0]
    wq_b_scale_r: pl.Tensor[
        [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = wq_b_scale[0]
    wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[0]
    wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[0]
    l2_decode_swa(
        x_hc[0], incoming_pre_mix[0],
        hc_attn_fn[0], hc_attn_scale[0], hc_attn_base[0],
        attn_norm_weight[0],
        wq_a[0], wq_a_scale_r, q_norm_weight[0], wq_b[0], wq_b_scale_r,
        wkv[0], wkv_scale_r, kv_norm_weight[0],
        attn_sink[0],
        wo_a[0], wo_a_scale[0], wo_b[0], wo_b_scale_r,
        rope_cos[0], rope_sin[0],
        window_slots[0], window_indices[0],
        window_cache[0], window_cache_scale[0],
        attention_output[0], attention_hidden[0], attention_pre_mix[0],
        num_tokens, device=0,
    )


def build_specs(args):
    """Specs for the TP1 SWA attention composition."""
    value_build_swa_specs = build_swa_leaf_specs(args, 'decode')
    return assemble_specs(args, value_build_swa_specs, l3_decode_swa.param_names)


def golden_decode_swa(tensors):
    """Compose TP1 mHC, normalization, QKV, attention and output references."""
    active = int(tensors["num_tokens"])
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"][0],
        tensors["hc_attn_fn"][0], tensors["hc_attn_scale"][0], tensors["hc_attn_base"][0],
    )
    tensors["attention_pre_mix"][0].copy_(pre_mix)
    collapsed = golden_hc_pre(tensors["x_hc"][0], tensors["incoming_pre_mix"][0]).to(torch.bfloat16)
    normalized = golden_rms_norm(collapsed[:active], tensors["attn_norm_weight"][0]).unsqueeze(0)
    leaf = dict(tensors, x=normalized, output=tensors["attention_output"][:, :active])
    golden_swa_leaf(leaf)
    tensors["attention_hidden"][0].zero_()
    tensors["attention_output"][0, active:].zero_()
    if active:
        hidden = golden_hc_post(
            tensors["attention_output"][0, :active], tensors["x_hc"][0, :active],
            post_mix[:active], residual_mix[:active],
        ).to(torch.float32)
        tensors["attention_hidden"][0, :active].copy_(hidden)


def comparisons(args, devices):
    leaf = ratio_allclose(atol=1e-2, rtol=1.0 / 128)
    compare = {
        "attention_hidden": leaf,
        "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
    }
    compare.update(
        {
            "attention_output": leaf,
            # Composed rows can flip one FP8 code at a quantization boundary: allow one
            # subnormal step (2**-9) plus one relative step (2**-3) on the payload, and less
            # than 1% of mapped rows to miss even that.
            **window_cache_compare(
                len(devices), args.tokens, payload_atol=2 ** -9, payload_rtol=2 ** -3,
                payload_error_ratio=0.01,
            ),
        }
    )
    return compare


if __name__ == "__main__":
    from utils import parse_decode_startpos

    import argparse

    from pypto.ir import DistributedConfig
    from golden import run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 TP1 SWA decode composition")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one position per request; omitted: default validation workload"
    parser.add_argument("--startpos", type=parse_decode_startpos, help=startpos_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=("mixed", "shuffle", "prefix"), default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=32, requests=4)
    args = parser.parse_args()
    if args.startpos is not None:
        args.tokens = args.requests = len(args.startpos)
    args.start_pos = args.startpos
    args.tp = args.dp = 1
    args.active_tokens = args.tokens
    args.bench = False
    devices = [args.device]
    result = run(
        fn=l3_decode_swa,
        specs=build_specs(args),
        golden_fn=golden_decode_swa,
        compare_fn=comparisons(args, devices),
        config={
            "backend_type": args.backend_type, "execution_mode": args.execution_mode,
            "enable_chip_swimlane": args.enable_chip_swimlane,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
