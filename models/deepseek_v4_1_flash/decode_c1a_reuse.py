# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A reuse attention wired through mHC."""

import pypto.language as pl
import torch

from config import (
    AttentionMode,
    COMPRESSED_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
    FLASH as M,
    CMP_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    T_DYN,
    TP_SIZE,
    LOCAL_T_DYN,
)


from decode_c1a_full import (
    golden_c1a_hc_case,
    build_hc_validation_values,
    OUTPUT_ATOL,
    OUTPUT_RTOL,
    OUTPUT_MAX_ERROR_RATIO,
    cache_compare,
    golden_c1a_attention,
)
from utils import AttentionGoldenResult
from golden import ScalarSpec, TensorSpec, ratio_allclose, run
from hc import hc_pre
from hc import hc_mixes
from hc import hc_post
from qkv_proj_rope import q_proj_qr, q_proj_rope, kv_proj_rope
from sparse_attention import gather_packed, attend_selected, publish_window_row, GATHER_WORKERS, attend_packed_cohorts
from rmsnorm import rms_norm_tiled
from o_proj import o_proj


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
INDEX_TOPK = M.index_topk


@pl.jit.inline(auto_scope=False)
def decode_c1a_reuse(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix, sinkhorn_blocks=64)
    x_attn = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(x_hc, pre_mix, x_attn)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd((tokens + 7) // 8, name_hint="c1a_input_norm") as x_ready:
        norm_worker = pl.tile.get_block_idx()
        rms_norm_tiled(x_attn, attn_norm_weight, x_norm, norm_worker, tokens, output_fp32=False)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q_proj_qr(x_norm, wq_a, wq_a_scale, q_norm_weight, qr, num_tokens)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    q_proj_rope(qr, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens, max_workers=64)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_norm, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, kv, num_tokens)
    publish_window_row(kv, window_slots, window_cache, window_cache_scale, num_tokens)
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    if TP_SIZE == 1 and num_tokens <= 12:
        attend_packed_cohorts(
            query, window_indices, window_cache, window_cache_scale, compressed_indices,
            compressed_cache, compressed_cache_scale, attn_sink, attended, num_tokens, x_ready,
        )
    else:
        selected = pl.create_tensor([tokens, 640, HEAD_DIM], dtype=pl.BF16)
        indices = pl.create_tensor([tokens, 640], dtype=pl.INT32)
        selected_ret, indices_ret, gather_tid = gather_packed(
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            window_indices, compressed_indices, selected, indices, num_tokens, x_ready, max_workers=GATHER_WORKERS,
        )
        attend_selected(query, selected, indices, attn_sink, attended, num_tokens, gather_tid)
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens, grouped_max_workers=64,
    )
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd(tokens, name_hint="c1a_output_cast"):
        row = pl.tile.get_block_idx()
        for col in pl.range(0, D, 512):
            value = pl.tile.full([1, 512], dtype=pl.BF16, value=0.0)
            if row < num_tokens:
                partial_tile = pl.load(partial, [row, col], [1, 512])
                value = pl.cast(partial_tile, pl.BF16, mode="rint")
            pl.store(value, [row, col], attn_out)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output


@pl.jit
def l2_decode_c1a_reuse(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    return decode_c1a_reuse(
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        compressed_indices,
        output, next_pre_mix,
        num_tokens,
    )


@pl.jit.host
def l3_decode_c1a_reuse(
    x_hc: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TP_SIZE, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TP_SIZE, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    compressed_indices: pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, LOCAL_T_DYN)
    rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    # MX_B_NN scale views.
    wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[0]
    wq_b_scale_r: pl.Tensor[
        [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = wq_b_scale[0]
    wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[0]
    wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[0]
    l2_decode_c1a_reuse(
        x_hc[0], pre_mix[0],
        hc_attn_fn[0], hc_attn_scale[0], hc_attn_base[0],
        attn_norm_weight[0],
        wq_a[0], wq_a_scale_r, q_norm_weight[0],
        wq_b[0], wq_b_scale_r,
        wkv[0], wkv_scale_r, kv_norm_weight[0],
        attn_sink[0],
        wo_a[0], wo_a_scale[0], wo_b[0], wo_b_scale_r,
        rope_cos[0], rope_sin[0],
        window_slots[0], window_indices[0],
        window_cache[0], window_cache_scale[0], compressed_cache[0], compressed_cache_scale[0],
        compressed_indices[0],
        output[0], next_pre_mix[0],
        num_tokens,
        device=0,
    )


def golden_decode_attn_c1a_reuse(
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
    wo_a_scale: torch.Tensor,
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
    compressed_indices: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate C1A reuse attention and cache state."""
    return golden_c1a_attention(
        mode=AttentionMode.REUSE,
        x=x, wq_a=wq_a, wq_a_scale=wq_a_scale, q_norm_weight=q_norm_weight, wq_b=wq_b, wq_b_scale=wq_b_scale,
        wkv=wkv, wkv_scale=wkv_scale, kv_norm_weight=kv_norm_weight, attn_sink=attn_sink, wo_a=wo_a, wo_a_scale=wo_a_scale, wo_b=wo_b,
        wo_b_scale=wo_b_scale, rope_cos=rope_cos, rope_sin=rope_sin, window_slots=window_slots,
        window_indices=window_indices, window_cache=window_cache, window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache, compressed_cache_scale=compressed_cache_scale,
        compressed_indices=compressed_indices,
    )



def golden_decode_c1a_reuse_case(tensors):
    """Fill the mHC-wired reuse entry expectations from the operator reference."""
    golden_c1a_hc_case(tensors, golden_decode_attn_c1a_reuse)


if __name__ == "__main__":
    from utils import parse_decode_startpos

    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1 A5 mHC C1A reuse validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one position per request; omitted: default validation workload"
    parser.add_argument("--startpos", type=parse_decode_startpos, help=startpos_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["random", "ragged", "masked", "sink"], default="random")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=1, requests=1, pages=8)
    args = parser.parse_args()
    if args.startpos is not None:
        args.tokens = args.requests = len(args.startpos)
        args.pages = (max(args.startpos) + 128) // 128
    host = l3_decode_c1a_reuse
    values = build_hc_validation_values(
        "reuse", args.tokens, args.pages, args.requests, args.seed, args.case,
        start_pos=args.startpos,
    )
    # Independent initial storage for each harness allocation.
    specs = [
        TensorSpec(n, list(values[n].shape), values[n].dtype, init_value=lambda n=n: values[n].clone())
        for n in host.param_names if n != "num_tokens"
    ]

    specs.append(ScalarSpec("num_tokens", torch.int32, args.tokens))

    output_budget = ratio_allclose(atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL, max_error_ratio=OUTPUT_MAX_ERROR_RATIO)
    comparisons = {
        "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
        "output": output_budget,
        **cache_compare(1, args.tokens, "reuse"),
    }
    result = run(
        fn=host,
        specs=specs,
        golden_fn=golden_decode_c1a_reuse_case,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            dump_passes=args.dump_passes,
            enable_chip_swimlane=args.enable_chip_swimlane,
            distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=comparisons,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
