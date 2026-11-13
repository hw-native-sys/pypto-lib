# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
# Re-enable CPU validation when flat MX scale packing is corrected:
# https://github.com/hw-native-sys/pto-isa/issues/357
"""V4.1 inverse RoPE, grouped W8A8 MXFP8 output projection, and HC post."""

from pathlib import Path

import torch

from quant import MX_GROUP, _quantize_mxfp8_activation, decode_e8m0, unpack_mx_b_scale

from qkv_proj_rope import golden_rope_interleave as rope_interleave
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum


import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import pypto.language.distributed as pld

from config import (
    FLASH as M, T_DYN,
)


# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


O_GROUP_ROWS = pl.dynamic("O_GROUP_ROWS")
O_PAD_ROWS = pl.dynamic("O_PAD_ROWS")
O_IN_SCALE_COLS = pl.dynamic("O_IN_SCALE_COLS")
O_OUT_SCALE_COLS = pl.dynamic("O_OUT_SCALE_COLS")

# tiling
OB_HC_POST_COLUMNS = 27  # 26 full N192 tiles plus one N128 tile.


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "o_quant_vf.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _quant_o_vf(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    payload: pl.Out[pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN]],
    scales: pl.Out[pl.Tensor[[LOCAL_O_GROUPS, O_IN_SCALE_COLS], pl.FP8E8M0]],
    active_rows: pl.Scalar[pl.INT32],
    groups: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Inverse-rotate BF16 tails and quantize projection inputs to OCP MXFP8."""
    # Preserve both extern Out values: https://github.com/hw-native-sys/pypto/issues/3249
    return payload, scales


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "o_proj_b_hc_post.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "o_proj_b_hc_post.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _o_proj_b_hc_post(
    payload: pl.Tensor[[O_PAD_ROWS, LOCAL_O_WIDTH], pl.FP8E4M3FN],
    weight: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[1, O_OUT_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Fuse runtime-M O-B with SplitM HC-post."""
    return output


@pl.jit.inline(auto_scope=False)
def o_proj_a_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    output_scale: pl.Tensor,
    output_width: pl.Scalar[pl.INDEX],
    active_rows: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    row_tile: pl.constexpr = 80,
):
    projected = pl.create_tensor([padded_rows, output_width], dtype=pl.FP32)
    for cube_group in pl.parallel(output_width // 1024):
        w: pl.Tensor[[4096, 1024], pl.FP8E4M3FN, pl.NZ] = weight[cube_group]
        sw = weight_scale[cube_group]
        sw = pl.tensor.view(sw, [128, 1024], layout=pl.MX_B_NN)
        for cube_mt in pl.parallel(padded_rows // row_tile):
            cube_row = cube_mt * row_tile
            scale_start = cube_row * 128
            scale_end = scale_start + row_tile * 128
            sa = activation_scale[cube_group, scale_start:scale_end]
            sa = pl.reshape(sa, [row_tile, 128])
            sa = pl.tensor.view(sa, [row_tile, 128], layout=pl.MX_A_ZZ)
            for cube_nb in pl.spmd(4):
                cube_col = cube_nb * 256
                a0 = pl.load(payload, [cube_group * padded_rows + cube_row, 0], [row_tile, 256], target_memory=pl.Mem.Mat)
                sa0 = pl.load(sa, [0, 0], [row_tile, 8])
                b0 = pl.load(w, [0, cube_col], [256, 256], target_memory=pl.Mem.Mat)
                sb0 = pl.load(sw, [0, cube_col], [8, 256])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for k in pl.pipeline(256, 4096, 256, stage=2):
                    av = pl.load(payload, [cube_group * padded_rows + cube_row, k], [row_tile, 256], target_memory=pl.Mem.Mat)
                    sav = pl.load(sa, [0, k // 32], [row_tile, 8])
                    bv = pl.load(w, [k, cube_col], [256, 256], target_memory=pl.Mem.Mat)
                    sbv = pl.load(sw, [k // 32, cube_col], [8, 256])
                    acc = pl.matmul_mx_acc(acc, av, sav, bv, sbv)
                projected = pl.store(acc, [cube_row, cube_group * 1024 + cube_col], projected)
    scale_rows = padded_rows // 16
    scale_cols = output_width // 2
    scales = pl.reshape(output_scale, [scale_rows, scale_cols])
    column_tiles = output_width // 256
    for task in pl.spmd(padded_rows // row_tile * column_tiles):
        row = task // column_tiles * row_tile
        group = task % column_tiles // 4
        nb = task % 4
        col = nb * 256
        values = pl.load(projected, [row, group * 1024 + col], [row_tile, 256], target_memory=pl.Mem.Vec)
        flat = pl.reshape(values, [row_tile * 256])
        valid_rows = pl.max(0, pl.min(row_tile, active_rows - row))
        quant_bytes = pl.create_tile([row_tile * 256], dtype=pl.INT8, target_memory=pl.Mem.Vec)
        scale_pairs = pl.create_tile([row_tile * 256 // 64], dtype=pl.UINT16, target_memory=pl.Mem.Vec)
        scale_bytes = pl.reinterpret_view(scale_pairs, pl.UINT8, shape=[row_tile * 256 // 32])
        quant_words = pl.reinterpret_view(quant_bytes, pl.UINT32, shape=[row_tile * 256 // 4])
        with pl.reg_scope():
            q_full16 = pl.reg.full_mask(lanes=128)
            q_lane16 = pl.reg.index_gen(0, lanes=128, dtype=pl.UINT16)
            q_floor = pl.reg.broadcast(1024, lanes=128, dtype=pl.UINT16)
            q_three = pl.reg.broadcast(3, lanes=128, dtype=pl.UINT16)
            q_pair_lane = pl.reg.bitwise_and(q_lane16, q_three)
            q_pair_low_index = pl.reg.shift_lefts(q_pair_lane, 5, q_full16)
            q_pair_high_index = pl.reg.adds(q_pair_low_index, 16, q_full16)
            q_scatter_relative = pl.reg.shift_lefts(q_lane16, 4, q_full16)
            q_four = pl.reg.active_mask(4, lanes=128)
            q_init_zero = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT16)
            for q_init in pl.hardware_loop(row_tile // 16):
                _q_initialized = pl.reg.store(scale_pairs, offset=q_init * 64, value=q_init_zero)
            for q_r in pl.hardware_loop(valid_rows):
                q_raw = pl.reg.load(flat, offset=q_r * 256, lanes=256)
                q_values = pl.reg.convert(q_raw, dtype=pl.BF16, rounding="R", saturate=False)
                q_bits = pl.reg.interpret_cast(q_values, dtype=pl.UINT16)
                q_wide_mask = pl.reg.full_mask(lanes=256)
                q_exp_mask_wide = pl.reg.broadcast(32640, lanes=256, dtype=pl.UINT16)
                q_exponents = pl.reg.bitwise_and(q_bits, q_exp_mask_wide)
                q_groups = pl.reg.reduce_max(q_exponents, q_wide_mask, group_size=32)
                q_maximum = pl.reg.group_broadcast(q_groups, lanes=128)
                q_maximum = pl.reg.maximum(q_maximum, q_floor)
                q_shared = pl.reg.sub(q_maximum, q_floor)
                q_group_mask = pl.reg.full_mask(lanes=8)
                q_group_maximum = pl.reg.maximums(q_groups, 1024, q_group_mask)
                q_group_floor = pl.reg.broadcast(1024, lanes=8, dtype=pl.UINT16)
                q_group_shared = pl.reg.sub(q_group_maximum, q_group_floor)
                q_group_bias = pl.reg.broadcast(32512, lanes=8, dtype=pl.UINT16)
                q_group_inverse = pl.reg.sub(q_group_bias, q_group_shared)
                q_group_nonzero = pl.reg.minimums(q_group_shared, 128, q_group_mask)
                q_group_nonzero = pl.reg.shift_rights(q_group_nonzero, 7, q_group_mask)
                q_group_zero = pl.reg.broadcast(0, lanes=8, dtype=pl.UINT16)
                q_group_keep = pl.reg.sub(q_group_zero, q_group_nonzero)
                q_group_inverse = pl.reg.bitwise_and(q_group_inverse, q_group_keep)
                q_wide_inverse = pl.reg.group_broadcast(q_group_inverse, lanes=256)
                q_wide_reciprocal = pl.reg.interpret_cast(q_wide_inverse, dtype=pl.BF16)
                q_wide_scaled = pl.reg.mul(q_values, q_wide_reciprocal)
                q_wide_float = pl.reg.convert(q_wide_scaled, dtype=pl.FP32)
                q_wide_float_mask = pl.reg.full_mask(lanes=256)
                q_wide_clipped = pl.reg.maximums(q_wide_float, -448.0, q_wide_float_mask)
                q_wide_clipped = pl.reg.minimums(q_wide_clipped, 448.0, q_wide_float_mask)
                q_wide_fp8 = pl.reg.convert(q_wide_clipped, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                q_wide_words = pl.reg.interpret_cast(q_wide_fp8, dtype=pl.UINT32)
                _q_quant_written = pl.reg.store(quant_words, offset=q_r * 64, value=q_wide_words)
                q_codes = pl.reg.shift_rights(q_shared, 7, q_full16)
                q_low = pl.reg.gather_lanes(q_codes, q_pair_low_index)
                q_high = pl.reg.gather_lanes(q_codes, q_pair_high_index)
                q_high = pl.reg.shift_lefts(q_high, 8, q_full16)
                q_packed = pl.reg.bitwise_or(q_low, q_high)
                q_scale_base = q_r // 16 * 64 + q_r % 16
                q_scale_base16 = pl.cast(q_scale_base, pl.UINT16)
                q_offsets = pl.reg.adds(q_scatter_relative, q_scale_base16, q_full16)
                _q_scale_written = pl.reg.scatter(scale_pairs, q_offsets, q_packed, q_four)
            empty = pl.reg.broadcast(0, lanes=256, dtype=pl.INT8)
            for inactive in pl.hardware_loop(row_tile - valid_rows):
                pad_row = valid_rows + inactive
                _padding_written = pl.reg.store(quant_bytes, offset=pad_row * 256, value=empty)
        quantized = pl.reshape(quant_bytes, [row_tile, 256])
        quantized = pl.reinterpret_view(quantized, pl.FP8E4M3FN)
        scale = pl.reshape(scale_bytes, [row_tile // 16, 256 // 2])
        scale = pl.reinterpret_view(scale, pl.FP8E8M0)
        output = pl.store(quantized, [row, group * 1024 + col], output)
        packed = pl.reshape(scale, [row_tile // 16, 128])
        scales = pl.store(packed, [row // 16, group * 512 + nb * 128], scales)
    return output, output_scale


@pl.jit.inline(auto_scope=False)
def o_proj_hc_post(
    attended: pl.Tensor,
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    oa_row_tile: pl.constexpr = 80,
    ob_row_tile: pl.constexpr = 128,
):
    """Return HC output and completion after consuming each N192/N128 accumulator."""
    total_rows = pl.tensor.dim(output, 0)
    oa_tile = oa_row_tile
    if total_rows > oa_row_tile:
        oa_tile = 128
    padded_rows = ((total_rows + oa_tile - 1) // oa_tile) * oa_tile
    payload_all = pl.create_tensor([LOCAL_O_GROUPS * padded_rows, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    scales_all = pl.create_tensor([LOCAL_O_GROUPS, padded_rows * (O_GROUP_IN // 32)], dtype=pl.FP8E8M0)
    vector_workers = 2 * pl.system.available_cluster_count()
    with pl.spmd(vector_workers, name_hint="quant_o_vf"):
        payload_all, scales_all = _quant_o_vf(
            attended, rope_cos, rope_sin, payload_all, scales_all, num_tokens, LOCAL_O_GROUPS, padded_rows,
        )
    payload = pl.create_tensor([padded_rows, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
    activation_scale = pl.create_tensor([1, padded_rows * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
    if total_rows > oa_row_tile:
        payload, activation_scale = o_proj_a_pypto(
            payload_all, wo_a, scales_all, wo_a_scale, payload, activation_scale,
            LOCAL_O_WIDTH, num_tokens, padded_rows, row_tile=128,
        )
    else:
        payload, activation_scale = o_proj_a_pypto(
            payload_all, wo_a, scales_all, wo_a_scale, payload, activation_scale,
            LOCAL_O_WIDTH, num_tokens, padded_rows, row_tile=oa_row_tile,
        )
    ob_tasks = ((total_rows + ob_row_tile - 1) // ob_row_tile) * OB_HC_POST_COLUMNS
    ob_workers = pl.max(1, pl.min(ob_tasks, pl.system.available_cluster_count()))
    with pl.spmd(ob_workers, name_hint="proj_o_b_hc_post") as ready:
        output = _o_proj_b_hc_post(
            payload, wo_b, activation_scale, wo_b_scale,
            residual, post_mix, residual_mix, output, total_rows, ob_row_tile,
        )
    return output, ready


def grouped_mxfp8_linear(
    grouped: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate grouped W8A8 MXFP8 with NZ weights and MX_B_NN scales."""
    from quant import unpack_fp8_nz

    wo_a = unpack_fp8_nz(wo_a)
    activation, activation_scale = _quantize_mxfp8_activation(grouped, scale_alg=0)
    weight_scale = decode_e8m0(unpack_mx_b_scale(wo_a_scale))
    weight_groups = wo_a.float().unflatten(-2, (-1, MX_GROUP))
    partials = torch.einsum("tgmk,gmkr->tgmr", activation.float(), weight_groups)
    partials = partials * activation_scale.unsqueeze(-1) * weight_scale.unsqueeze(0)
    return partials.sum(dim=2).to(output_dtype or grouped.dtype)

def golden_grouped_output(
    grouped: torch.Tensor, wo_a: torch.Tensor, wo_a_scale: torch.Tensor,
) -> torch.Tensor:
    """Grouped W8A8 MXFP8 output GEMM with one BF16 output cast."""
    return grouped_mxfp8_linear(grouped, wo_a, wo_a_scale, output_dtype=torch.bfloat16)


def golden_o_proj(
    attended: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    *,
    wo_a_scale: torch.Tensor,
    output_dtype: torch.dtype | None = None,
    sequential_projection: bool = False,
) -> torch.Tensor:
    from quant import unpack_fp8_nz

    wo_b = unpack_fp8_nz(wo_b)
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    attended = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = attended.flatten(-2).unflatten(-1, (groups, -1))
    latent = golden_grouped_output(grouped, wo_a, wo_a_scale)
    if sequential_projection and wo_b_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(latent.flatten(-2), wo_b, wo_b_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(latent.flatten(-2), wo_b, wo_b_scale, output_dtype=torch.bfloat16)
    return projected.to(output_dtype or torch.bfloat16)


@pl.jit.inline(auto_scope=False)
def retire_prefill_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor,
    group_base: pl.Scalar[pl.INT32],
    group_rank: pl.Scalar[pl.INT32],
    group_size: pl.constexpr,
):
    """Clear ready and consumed signals at a serialized entry boundary."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_signal_retire", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, [0, 0, 0])
        for peer in pl.range(group_size):
            pld.system.notify(
                arrived, peer=group_base + group_rank, offsets=[peer, 0],
                value=0, op=pld.NotifyOp.Set,
            )
        pl.write(completion_anchor, [0, 0, 0], anchor)
    return completion_anchor


@pl.jit
def o_proj_hc_post_validation(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj_hc_post(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        residual, post_mix, residual_mix, output, num_tokens,
        oa_row_tile=80, ob_row_tile=128,
    )
    return output


def build_o_proj_specs(active_tokens: int, *, pad_to_tile: bool = True):
    """Build deterministic output-projection fixtures with padded inactive rows."""
    from golden import ScalarSpec, TensorSpec
    from quant import pack_fp8_nz, quantize_mxfp8_upward_reference, pack_mx_b_scale, quantize_grouped_mxfp8_weight

    tokens = max(16, (active_tokens + 15) // 16 * 16) if pad_to_tile else active_tokens
    values = {}

    def inputs():
        if not values:
            generator = torch.Generator().manual_seed(17)
            attended = torch.randn(tokens, LOCAL_H * HEAD_DIM, generator=generator).bfloat16()
            attended[active_tokens:] = float("nan")
            wo_a = torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, generator=generator) / O_GROUP_IN**0.5
            wo_a, wo_a_scale = quantize_grouped_mxfp8_weight(wo_a.bfloat16())
            wo_b = torch.randn(D, LOCAL_O_WIDTH, generator=generator) / LOCAL_O_WIDTH**0.5
            wo_b, wo_b_scale = quantize_mxfp8_upward_reference(wo_b)
            frequencies = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
            angles = torch.arange(tokens).float()[:, None] * frequencies
            values.update(
                attended=attended, wo_a=wo_a, wo_a_scale=wo_a_scale,
                wo_b=pack_fp8_nz(wo_b.T.contiguous()),
                wo_b_scale=pack_mx_b_scale(wo_b_scale.T.contiguous()).view(torch.float8_e8m0fnu),
                rope_cos=angles.cos(), rope_sin=angles.sin(),
            )
        return values

    shapes = (
        ("attended", [tokens, LOCAL_H * HEAD_DIM], torch.bfloat16),
        ("wo_a", [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], torch.float8_e4m3fn),
        ("wo_a_scale", [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], torch.float8_e8m0fnu),
        ("wo_b", [LOCAL_O_WIDTH, D], torch.float8_e4m3fn),
        ("wo_b_scale", [LOCAL_O_WIDTH // 32, D], torch.float8_e8m0fnu),
        ("rope_cos", [tokens, ROPE_DIM // 2], torch.float32),
        ("rope_sin", [tokens, ROPE_DIM // 2], torch.float32),
    )
    specs = [TensorSpec(name, shape, dtype, init_value=lambda name=name: inputs()[name])
             for name, shape, dtype in shapes]
    return specs + [
        TensorSpec("output", [tokens, D], torch.bfloat16, init_value=13.0),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]


def golden_o_proj_validation(tensors):
    """Project active rows to BF16 and zero inactive rows."""
    active = int(tensors["num_tokens"])
    tensors["output"].zero_()
    if active == 0:
        return
    attended = tensors["attended"][:active].unflatten(-1, (LOCAL_H, HEAD_DIM))
    result = golden_o_proj(
        attended, tensors["rope_cos"][:active], tensors["rope_sin"][:active],
        tensors["wo_a"], tensors["wo_b"], tensors["wo_b_scale"],
        wo_a_scale=tensors["wo_a_scale"], output_dtype=torch.bfloat16, sequential_projection=True,
    )
    tensors["output"][:active].copy_(result)


def build_o_proj_hc_post_specs(active_tokens: int = 72, *, pad_to_tile: bool = False):
    """Add genuine FP32 HC residuals and mixes to active or padded projection inputs."""
    from golden import ScalarSpec, TensorSpec

    specs = build_o_proj_specs(active_tokens, pad_to_tile=pad_to_tile)
    tokens = specs[0].shape[0]
    specs = [spec for spec in specs if spec.name not in ("output", "num_tokens")]
    generator = torch.Generator().manual_seed(29)
    return specs + [
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, D, generator=generator)),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32,
                   init_value=lambda: 2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, HC_MULT, generator=generator)),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]


def golden_o_proj_hc_post_validation(tensors):
    """Compute independent BF16 output projection followed by FP32 HC mixing."""
    from hc import golden_hc_post

    sublayer = torch.zeros_like(tensors["output"][:, 0, :], dtype=torch.bfloat16)
    projected = dict(tensors, output=sublayer)
    golden_o_proj_validation(projected)
    tensors["output"].copy_(golden_hc_post(
        sublayer, tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    ))


def compare_o_proj_hc_post(actual, expected, *, inputs, compare, **kwargs):
    """Check active projection tolerance and strict inactive-row residual mixing."""
    from golden import ratio_allclose

    active = int(inputs["num_tokens"])
    passed, detail = True, "no active rows"
    if active:
        passed, detail = compare(actual[:active], expected[:active], inputs=inputs, **kwargs)
    if active < actual.shape[0]:
        compare_tail = ratio_allclose(atol=1e-6, rtol=1e-5, max_error_ratio=0.0)
        tail_ok, tail_detail = compare_tail(actual[active:], expected[active:], inputs=inputs, **kwargs)
        return passed and tail_ok, f"{detail}; inactive rows: {tail_detail}"
    return passed, detail


if __name__ == "__main__":
    import argparse
    import functools
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="V4.1 fused output projection and HC post validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, help="active rows for a single padded case")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    if args.tokens is not None and args.tokens < 0:
        parser.error("tokens must be nonnegative")
    cases = ((0, True), (17, True), (72, False), (129, True),
             (224, False), (225, False), (240, False), (241, False))
    if args.tokens is not None:
        cases = ((args.tokens, True),)
    for active, padded in cases:
        print(f"[O_PROJ] active={active}, padded={padded}, FP32 HC post", flush=True)
        result = run(
            fn=o_proj_hc_post_validation,
            specs=build_o_proj_hc_post_specs(active, pad_to_tile=padded),
            golden_fn=golden_o_proj_hc_post_validation,
            compare_fn={
                "output": functools.partial(compare_o_proj_hc_post, compare=ratio_allclose(atol=1e-2, rtol=1.0 / 128)),
            },
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device,
            },
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
