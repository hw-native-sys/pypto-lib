# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C2A Reuse decode Attention half-layer composition entry."""


import pypto.language as pl
import torch
from golden import mapped_pool_ratio_allclose

from decode_c2a_full import (
    build_hc_validation_values, golden_c2a_hc_case, golden_reference_c2a_reuse,
)

from config import (
    FLASH as M, LOCAL_T_DYN, TP_SIZE, T_DYN, ORI_BLOCKS_DYN, CMP_BLOCKS_DYN,
    COMPRESSED_CACHE_GROUP, WINDOW_CACHE_GROUP,
)
from hc import hc_mixes
from hc import hc_pre
from hc import hc_post
from rmsnorm import rms_norm_tiled
from qkv_proj_rope import q_proj_qr, q_proj_rope, kv_proj_rope
from sparse_attention import (
    PACKED_QUERY_TILE as QUERY_TILE, SELECTED_KEYS as SPARSE_WIDTH,
    gather_packed_chunk, attend_c2a_chunk, publish_window_row,
)
from o_proj import o_proj
from golden import TensorSpec, ScalarSpec, ratio_allclose, run
from utils import AttentionGoldenResult

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
MX_GROUP = M.weight_block_size[0]


@pl.jit.inline(auto_scope=False)
def decode_c2a_reuse(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    x_attn = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(x_hc, pre_mix, x_attn)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd((tokens + 7) // 8, name_hint="c2a_input_norm") as cache_ready:
        worker = pl.tile.get_block_idx()
        rms_norm_tiled(x_attn, attn_norm_weight, x_norm, worker, tokens)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    q_proj_qr(x_norm, wq_a, wq_a_scale, q_norm_weight, qr, num_tokens)
    q_proj_rope(qr, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens, max_workers=2147483647)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_norm, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, window_kv, num_tokens)
    publish_window_row(window_kv, window_slots, window_cache, window_cache_scale, num_tokens)

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH, HEAD_DIM], dtype=pl.BF16)
    combined = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH], dtype=pl.INT32)
    chunk_done = cache_ready
    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        gather_tid = gather_packed_chunk(
            window_cache, window_cache_scale, window_indices, compressed_cache,
            compressed_cache_scale, compressed_indices, selected, combined, start, active,
            chunk_done,
        )
        chunk_done = attend_c2a_chunk(
            query, selected, combined, attn_sink, attended, start, active, gather_tid
        )

    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b,
        wo_b_scale, rope_cos, rope_sin,
        partial, num_tokens, grouped_max_workers=4096,
    )
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd(tokens, name_hint="c2a_output_cast"):
        row = pl.tile.get_block_idx()
        for col in pl.range(0, D, 512):
            value = pl.tile.full([1, 512], dtype=pl.BF16, value=0.0)
            if row < num_tokens:
                partial_tile = pl.load(partial, [row, col], [1, 512])
                value = pl.cast(partial_tile, pl.BF16, mode="rint")
            pl.store(value, [row, col], attn_out)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output, next_pre_mix


@pl.jit
def l2_decode_c2a_reuse(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    return decode_c2a_reuse(
        x_hc,
        pre_mix,
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
        compressed_cache,
        compressed_cache_scale,
        compressed_indices,
        output,
        next_pre_mix,
        num_tokens,
    )


@pl.jit.host
def l3_decode_c2a_reuse(
    x_hc: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32],
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
    compressed_cache: pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    compressed_indices: pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, LOCAL_T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    wq_a_scale_mx: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[0]
    wq_b_scale_mx: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[0]
    wkv_scale_mx: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[0]
    wo_b_scale_mx: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[0]
    l2_decode_c2a_reuse(
        x_hc[0],
        pre_mix[0],
        hc_attn_fn[0],
        hc_attn_scale[0],
        hc_attn_base[0],
        attn_norm_weight[0],
        wq_a[0],
        wq_a_scale_mx,
        q_norm_weight[0],
        wq_b[0],
        wq_b_scale_mx,
        wkv[0],
        wkv_scale_mx,
        kv_norm_weight[0],
        attn_sink[0],
        wo_a[0], wo_a_scale[0],
        wo_b[0],
        wo_b_scale_mx,
        rope_cos[0],
        rope_sin[0],
        window_slots[0],
        window_indices[0],
        window_cache[0],
        window_cache_scale[0],
        compressed_cache[0],
        compressed_cache_scale[0],
        compressed_indices[0],
        output[0],
        next_pre_mix[0],
        num_tokens,
        device=0,
    )


def golden_decode_attn_c2a_reuse(
    x: "torch.Tensor",
    wq_a: "torch.Tensor",
    wq_a_scale: "torch.Tensor",
    q_norm_weight: "torch.Tensor",
    wq_b: "torch.Tensor",
    wq_b_scale: "torch.Tensor",
    wkv: "torch.Tensor",
    wkv_scale: "torch.Tensor",
    kv_norm_weight: "torch.Tensor",
    attn_sink: "torch.Tensor",
    wo_a: "torch.Tensor",
    wo_a_scale: "torch.Tensor",
    wo_b: "torch.Tensor",
    wo_b_scale: "torch.Tensor",
    rope_cos: "torch.Tensor",
    rope_sin: "torch.Tensor",
    window_slots: "torch.Tensor",
    window_indices: "torch.Tensor",
    window_cache: "torch.Tensor",
    window_cache_scale: "torch.Tensor",
    compressed_cache: "torch.Tensor",
    compressed_cache_scale: "torch.Tensor",
    compressed_indices: "torch.Tensor",
) -> "AttentionGoldenResult":
    """Evaluate C2A decode reuse against published compressed cache rows."""
    result = golden_reference_c2a_reuse(locals())
    return AttentionGoldenResult(
        **result,
        compressed_cache=compressed_cache,
        compressed_cache_scale=compressed_cache_scale,
        index_cache=None,
        index_cache_scale=None,
        state_cache=None,
        topk_indices=compressed_indices,
    )


def golden_decode_c2a_reuse_case(tensors):
    golden_c2a_hc_case(tensors, golden_decode_attn_c2a_reuse)


def c2a_reuse_window_cache_compare(
    world_size, tokens, payload_atol=0.0, payload_rtol=0.0, payload_error_ratio=0.0,
    scale_rtol=0.0, scale_error_ratio=0.0,
):
    """Compare the mapped window rows and require every other row to stay unchanged."""
    options = dict(
        mapping_shape=(world_size, tokens),
        block_size=128,
        leading_rank_axis=True,
    )
    return {
        "window_cache": mapped_pool_ratio_allclose(
            "window_slots", pool_name="window cache",
            atol=payload_atol, rtol=payload_rtol, max_error_ratio=payload_error_ratio, **options
        ),
        "window_cache_scale": mapped_pool_ratio_allclose(
            "window_slots", pool_name="window cache scale",
            atol=0.0, rtol=scale_rtol, max_error_ratio=scale_error_ratio, **options
        ),
    }


if __name__ == "__main__":
    from utils import parse_decode_startpos

    import argparse
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1 A5 mHC C2A reuse validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one position per request; omitted: default validation workload"
    parser.add_argument("--startpos", type=parse_decode_startpos, help=startpos_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["mixed", "long", "masked", "zero"], default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=32, requests=6)
    args = parser.parse_args()
    if args.startpos is not None:
        args.tokens = args.requests = len(args.startpos)
    values = build_hc_validation_values("reuse", args.tokens, args.requests, args.seed, args.case, start_pos=args.startpos)
    host = l3_decode_c2a_reuse
    specs = [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=lambda name=name: values[name].clone())
        for name in host.param_names if name != "num_tokens"
    ]
    specs.append(ScalarSpec("num_tokens", torch.int32, args.tokens))
    comparisons = {
        "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
        "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
    }
    comparisons.update(c2a_reuse_window_cache_compare(
        1, args.tokens, payload_atol=2 ** -9, payload_rtol=2 ** -3,
    ))
    result = run(
        fn=host,
        specs=specs,
        golden_fn=golden_decode_c2a_reuse_case,
        compare_fn=comparisons,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            enable_chip_swimlane=args.enable_chip_swimlane,
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
        ),
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
