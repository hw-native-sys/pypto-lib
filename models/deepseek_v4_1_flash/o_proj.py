# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Re-enable CPU validation when flat MX scale packing is corrected:
# https://github.com/hw-native-sys/pto-isa/issues/357
"""V4.1 inverse RoPE, grouped W8A8 MXFP8 output projection, and TP reductions."""

from pathlib import Path

import torch

from quant import MX_GROUP, _quantize_mxfp8_activation, decode_e8m0, unpack_mx_b_scale

from qkv_proj_rope import golden_rope_interleave as rope_interleave
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum


import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
from hc import hc_post
import pypto.language.distributed as pld

from config import (
    PREFILL_MAX_TOKENS,
    FLASH as M, SP_T_DYN, T_DYN, TP_SIZE,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
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
# TP1 decode retains its measured tiles and worker counts.
O_VECTOR_WORKERS = 64
O_A_N_TILE = 256
O_B_N_TILE = 160 if LOCAL_O_WIDTH == 8192 else 256
SP_REDUCE_ROW_TILE = 2
SP_PUBLISH_ROW_TILE = 16

# o_quant_vf.cpp tiling, transcribed for the PyPTO register implementation.
# kernel_entry selects run_native<80> for groups >= 8, the only shape this model
# builds (LOCAL_O_GROUPS == 8), so the row tile is a compile-time constant here.
O_QUANT_ROW_TILE = 80
O_QUANT_HEADS_PER_GROUP = 8
O_QUANT_LANES = 64                      # FP32 lanes in one 256-byte vector register
O_QUANT_GROUP = 32                      # MX quantization group
O_QUANT_SCALE_BLOCK = 16                # rows sharing one packed scale row
O_QUANT_SCALE_COLS = 256                # scale bytes per block row and head
# Shapes that index a Tile must reach the op as Python integers. A name bound
# inside a traced body is an IR expression even when its value is constant, and
# reg.load / reg.gather then reject it with "Tile dimension must be a static
# integer in P0", so every derived extent below stays at module level.
O_QUANT_HEADS = LOCAL_O_GROUPS * O_QUANT_HEADS_PER_GROUP
O_QUANT_ROPE_PAIRS = ROPE_DIM // 2
O_QUANT_ROPE_START = HEAD_DIM - ROPE_DIM
O_QUANT_TRIG_TILE = O_QUANT_ROW_TILE * O_QUANT_ROPE_PAIRS
O_QUANT_VALUE_TILE = O_QUANT_ROW_TILE * HEAD_DIM
O_QUANT_CODE_TILE = (O_QUANT_ROW_TILE // O_QUANT_SCALE_BLOCK) * (O_QUANT_SCALE_COLS // 2)


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


@pl.jit.inline(auto_scope=False)
def quant_o_vf_pypto(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    cosine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    payload: pl.Out[pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN]],
    scales: pl.Out[pl.Tensor[[LOCAL_O_GROUPS, O_IN_SCALE_COLS], pl.FP8E8M0]],
    active_rows: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    """Register transcription of kernels/o_quant_vf.cpp and o_quant_vf.hpp.

    Both ``__VEC_SCOPE__`` regions of the kernel are rewritten with ``pl.reg``
    inside ``pl.reg_scope``: ``rotate_tail`` and ``o_quant::quant_vf``. The
    kernel's ``groups`` argument is dropped because it only selected the static
    ``run_native<Rows>`` instantiation and the head count, both compile-time
    constants for this model.

    _quant_o_vf keeps driving the model until this implementation passes
    performance acceptance.
    """
    scale_block_rows = padded_rows // O_QUANT_SCALE_BLOCK
    # scale_gm walks 2048-byte block rows inside one group. Expose that stride as
    # a tensor view so the tile store stays an ordinary 2D slice; tensor-level
    # views must be built before pl.spmd, where reshape would lower as a Tile op.
    scale_view = pl.tensor.view(
        scales, [LOCAL_O_GROUPS * scale_block_rows, O_QUANT_HEADS_PER_GROUP * O_QUANT_SCALE_COLS]
    )
    workers = pl.max(1, pl.min(O_VECTOR_WORKERS, 2 * pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint="quant_o_vf_pypto") as published:
        worker = pl.tile.get_block_idx()
        worker_count = pl.tile.get_block_num()
        row_blocks = (padded_rows + O_QUANT_ROW_TILE - 1) // O_QUANT_ROW_TILE
        # PTOAS derives pipe synchronization from Tile and VF memory effects.
        for work in pl.range(worker, row_blocks * O_QUANT_HEADS, worker_count):
            head = work % O_QUANT_HEADS
            t0 = (work // O_QUANT_HEADS) * O_QUANT_ROW_TILE
            capacity = pl.min(padded_rows - t0, O_QUANT_ROW_TILE)
            rows = pl.cast(
                pl.max(0, pl.min(active_rows - pl.cast(t0, pl.INT32), pl.cast(capacity, pl.INT32))),
                pl.INDEX,
            )
            group = head // O_QUANT_HEADS_PER_GROUP
            local_head = head % O_QUANT_HEADS_PER_GROUP
            # Input(rows) / Trig(rows): static 80-row Vec tiles with a dynamic
            # valid extent, so rows past the active count are never transferred.
            window = pl.load(
                attended, [t0, head * HEAD_DIM], [O_QUANT_ROW_TILE, HEAD_DIM],
                valid_shape=[rows, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
            )
            cos_tile = pl.load(
                cosine, [t0, 0], [O_QUANT_ROW_TILE, O_QUANT_ROPE_PAIRS],
                valid_shape=[rows, O_QUANT_ROPE_PAIRS], target_memory=pl.MemorySpace.Vec,
            )
            sin_tile = pl.load(
                sine, [t0, 0], [O_QUANT_ROW_TILE, O_QUANT_ROPE_PAIRS],
                valid_shape=[rows, O_QUANT_ROPE_PAIRS], target_memory=pl.MemorySpace.Vec,
            )
            # reg.load / reg.store address a flat Vec tile, and a tile whose real
            # data ends at a runtime row cannot be flattened. Widen the declared
            # valid extent back to the allocation first: the rows past `rows` hold
            # undefined UB, and no loop below ever reads them.
            value = pl.tile.reshape(
                pl.tile.set_validshape(window, O_QUANT_ROW_TILE, HEAD_DIM),
                [O_QUANT_VALUE_TILE],
            )
            cos_flat = pl.tile.reshape(
                pl.tile.set_validshape(cos_tile, O_QUANT_ROW_TILE, O_QUANT_ROPE_PAIRS),
                [O_QUANT_TRIG_TILE],
            )
            sin_flat = pl.tile.reshape(
                pl.tile.set_validshape(sin_tile, O_QUANT_ROW_TILE, O_QUANT_ROPE_PAIRS),
                [O_QUANT_TRIG_TILE],
            )
            # The kernel's trailing `for row in [active_rows, Rows)` loop writes
            # zeros into both outputs; a zero-filled scratch tile is that store.
            quantized = pl.tile.full([O_QUANT_VALUE_TILE], pl.INT8, 0)
            codes = pl.tile.full([O_QUANT_CODE_TILE], pl.UINT16, 0)

            # --- rotate_tail: inverse RoPE over the trailing 64 BF16 columns ---
            with pl.reg_scope():
                full = pl.reg.full_mask(lanes=O_QUANT_LANES)
                lane = pl.reg.index_gen(0, lanes=O_QUANT_LANES, dtype=pl.UINT32)
                one = pl.reg.broadcast(1, lanes=O_QUANT_LANES, dtype=pl.UINT32)
                # ix1 is the partner lane of each (real, imaginary) pair, and
                # pair_index the trig element both lanes of a pair share.
                tmp = pl.reg.shift_rights(lane, 1, full)
                tmp = pl.reg.shift_lefts(tmp, 1, full)
                tmp = pl.reg.sub(lane, tmp)
                tmp = pl.reg.shift_lefts(tmp, 1, full)
                ix1 = pl.reg.add(lane, one)
                ix1 = pl.reg.sub(ix1, tmp)
                pair_index = pl.reg.shift_rights(lane, 1, full)
                even = pl.reg.compare(ix1, lane, full, predicate="gt")
                for row in pl.hardware_loop(rows):
                    # vgather2 offsets its base pointer; reg.gather takes one
                    # index vector, so the row origin is broadcast and added.
                    trig_base = pl.reg.broadcast(
                        pl.cast(row * O_QUANT_ROPE_PAIRS, pl.UINT32), lanes=O_QUANT_LANES, dtype=pl.UINT32
                    )
                    trig_index = pl.reg.add(pair_index, trig_base)
                    cosine_v = pl.reg.gather(cos_flat, trig_index, full)
                    sine_v = pl.reg.gather(sin_flat, trig_index, full)
                    sine_v = pl.reg.muls(sine_v, -1.0, full)
                    raw = pl.reg.load(value, offset=row * HEAD_DIM + O_QUANT_ROPE_START, lanes=O_QUANT_LANES)
                    rotated = pl.reg.convert(raw, dtype=pl.FP32)
                    swapped = pl.reg.gather_lanes(rotated, ix1)
                    scaled_real = pl.reg.mul(rotated, cosine_v)
                    scaled_imag = pl.reg.mul(swapped, sine_v)
                    real = pl.reg.sub(scaled_real, scaled_imag)
                    imag = pl.reg.add(scaled_imag, scaled_real)
                    rotated = pl.reg.select(even, real, imag)
                    raw = pl.reg.convert(rotated, dtype=pl.BF16, rounding="R", saturate=False)
                    rotated_back = pl.reg.store(value, offset=row * HEAD_DIM + O_QUANT_ROPE_START, value=raw)

            # --- quant_vf: per-32-element MXFP8 quantization of all 512 columns ---
            with pl.reg_scope():
                full = pl.reg.full_mask(lanes=O_QUANT_LANES)
                one_lane = pl.reg.active_mask(1, lanes=2 * O_QUANT_LANES)
                izero = pl.reg.broadcast(0, lanes=O_QUANT_LANES, dtype=pl.INT32)
                ione = pl.reg.broadcast(1, lanes=O_QUANT_LANES, dtype=pl.INT32)
                zero = pl.reg.broadcast(0.0, lanes=O_QUANT_LANES, dtype=pl.FP32)
                onef = pl.reg.broadcast(1.0, lanes=O_QUANT_LANES, dtype=pl.FP32)
                low = pl.reg.broadcast(-448.0, lanes=O_QUANT_LANES, dtype=pl.FP32)
                high = pl.reg.broadcast(448.0, lanes=O_QUANT_LANES, dtype=pl.FP32)
                upper = pl.reg.broadcast(O_QUANT_GROUP, lanes=O_QUANT_LANES, dtype=pl.UINT32)
                # The kernel decodes 256 elements at a time: a DINTLV_B16 load
                # plus the PART_EVEN / PART_ODD halves of four vcvts spread them
                # over four FP32 registers, so that one MX group of 32 lands on
                # the eight lanes vcgmax reduces over. pl.reg.reduce_max takes
                # the group width as a parameter, so that permutation buys
                # nothing here: one register holds 64 contiguous elements,
                # group_size=32 reduces its two MX groups directly, and both the
                # FP8 store and the scale pack stay contiguous.
                for row in pl.hardware_loop(rows):
                    for quarter in pl.hardware_loop(HEAD_DIM // O_QUANT_LANES):
                        offset = row * HEAD_DIM + quarter * O_QUANT_LANES
                        quarter_bits = pl.reg.load(value, offset=offset, lanes=O_QUANT_LANES)
                        magnitudes = pl.reg.convert(quarter_bits, dtype=pl.FP32)
                        amax = pl.reg.reduce_max(
                            pl.reg.abs(magnitudes, full), full, group_size=O_QUANT_GROUP
                        )
                        spread = pl.reg.group_broadcast(amax, lanes=O_QUANT_LANES)
                        exponent = pl.reg.interpret_cast(spread, dtype=pl.INT32)
                        exponent = pl.reg.shift_rights(exponent, 23, full)
                        exponent = pl.reg.adds(exponent, -8, full)
                        exponent = pl.reg.maximum(exponent, izero)
                        nonzero = pl.reg.maximum(exponent, ione)
                        divisor = pl.reg.shift_lefts(nonzero, 23, full)
                        divisor = pl.reg.interpret_cast(divisor, dtype=pl.FP32)
                        nonzero_mask = pl.reg.compares(exponent, 0, full, predicate="gt")
                        floor_mask = pl.reg.select(nonzero_mask, onef, zero)
                        scaled = pl.reg.div(magnitudes, divisor)
                        scaled = pl.reg.mul(scaled, floor_mask)
                        scaled = pl.reg.maximum(scaled, low)
                        scaled = pl.reg.minimum(scaled, high)
                        quant = pl.reg.convert(
                            scaled, dtype=pl.FP8E4M3FN, rounding="R", saturate=False
                        )
                        stored = pl.reg.store(
                            quantized, offset=offset, value=pl.reg.interpret_cast(quant, dtype=pl.INT8)
                        )
                        # Both exponents of this register pack into the single
                        # UINT16 the A-scale layout keeps at (block, head, g/2, r):
                        # the even group is its low byte and the odd group its high
                        # byte, which is exactly the kernel's e0 | e1 << 8.
                        odd_exponent = pl.reg.gather_lanes(exponent, upper)
                        packed = pl.reg.shift_lefts(odd_exponent, 8, full)
                        packed = pl.reg.bitwise_or(exponent, packed)
                        pairs = pl.reg.interpret_cast(packed, dtype=pl.UINT16)
                        code_index = pl.reg.broadcast(
                            pl.cast(
                                (row // O_QUANT_SCALE_BLOCK) * (O_QUANT_SCALE_COLS // 2)
                                + row % O_QUANT_SCALE_BLOCK
                                + quarter * O_QUANT_SCALE_BLOCK,
                                pl.UINT16,
                            ),
                            lanes=2 * O_QUANT_LANES,
                            dtype=pl.UINT16,
                        )
                        scattered = pl.reg.scatter(codes, code_index, pairs, one_lane)

            payload_tile = pl.tile.reinterpret_view(
                pl.tile.reshape(quantized, [O_QUANT_ROW_TILE, HEAD_DIM]), pl.FP8E4M3FN
            )
            payload_tile = pl.tile.set_validshape(payload_tile, capacity, HEAD_DIM)
            pl.store(payload_tile, [group * padded_rows + t0, local_head * HEAD_DIM], payload)
            code_bytes = pl.tile.reinterpret_view(
                pl.tile.reshape(
                    codes, [O_QUANT_ROW_TILE // O_QUANT_SCALE_BLOCK, O_QUANT_SCALE_COLS // 2]
                ),
                pl.UINT8,
            )
            scale_tile = pl.tile.reinterpret_view(code_bytes, pl.FP8E8M0)
            scale_tile = pl.tile.set_validshape(
                scale_tile, capacity // O_QUANT_SCALE_BLOCK, O_QUANT_SCALE_COLS
            )
            pl.store(
                scale_tile,
                [
                    group * scale_block_rows + t0 // O_QUANT_SCALE_BLOCK,
                    local_head * O_QUANT_SCALE_COLS,
                ],
                scale_view,
            )
    return published



@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "o_proj_a.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "o_proj_a.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _o_proj_a(
    payload: pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[LOCAL_O_GROUPS, O_IN_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    output: pl.Out[pl.Tensor[[O_PAD_ROWS, LOCAL_O_WIDTH], pl.FP8E4M3FN]],
    output_scale: pl.Out[pl.Tensor[[1, O_OUT_SCALE_COLS], pl.FP8E8M0]],
    output_width: pl.Scalar[pl.INDEX],
    active_rows: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Round the O-A result to BF16 and quantize on the paired Vector cores."""
    return output, output_scale


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "o_proj_b.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _o_proj_b(
    payload: pl.Tensor[[O_PAD_ROWS, LOCAL_O_WIDTH], pl.FP8E4M3FN],
    weight: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[1, O_OUT_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INDEX], reduction: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project MXFP8 rows with packed MX_A_ZZ and MX_B_NN scales."""
    ...


@pl.jit.inline(auto_scope=False)
def o_proj(
    attended: pl.Tensor,
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    worker_limit: pl.constexpr = 32,
    oa_row_tile: pl.constexpr = 80,
    ob_row_tile: pl.constexpr = 80,
):
    """Project row groups to BF16; both weights hold NZ bytes in logical [K, N] shapes."""
    total_rows = pl.tensor.dim(output, 0)
    # Native row tiles: OA M80/M128 and OB M80/M256.
    ob_padded_rows = ((total_rows + ob_row_tile - 1) // ob_row_tile) * ob_row_tile
    padded_rows = ((ob_padded_rows + oa_row_tile - 1) // oa_row_tile) * oa_row_tile
    payload_all = pl.create_tensor([LOCAL_O_GROUPS * padded_rows, O_GROUP_IN], dtype=pl.FP8E4M3FN)
    scales_all = pl.create_tensor([LOCAL_O_GROUPS, padded_rows * (O_GROUP_IN // 32)], dtype=pl.FP8E8M0)
    vector_workers = pl.max(1, pl.min(O_VECTOR_WORKERS, 2 * pl.system.available_cluster_count()))
    with pl.spmd(vector_workers, name_hint="quant_o_vf"):
        payload_all, scales_all = _quant_o_vf(
            attended, rope_cos, rope_sin, payload_all, scales_all, num_tokens, LOCAL_O_GROUPS, padded_rows,
        )
    payload = pl.create_tensor([padded_rows, LOCAL_O_WIDTH], dtype=pl.FP8E4M3FN)
    activation_scale = pl.create_tensor([1, padded_rows * (LOCAL_O_WIDTH // 32)], dtype=pl.FP8E8M0)
    oa_tasks = (padded_rows // oa_row_tile) * (LOCAL_O_WIDTH // O_A_N_TILE)
    oa_workers = pl.max(1, pl.min(oa_tasks, pl.system.available_cluster_count()))
    with pl.spmd(oa_workers, name_hint="proj_o_a_quant"):
        payload, activation_scale = _o_proj_a(
            payload_all, wo_a, scales_all, wo_a_scale, payload, activation_scale,
            LOCAL_O_WIDTH, num_tokens, padded_rows, oa_row_tile,
        )
    ob_tasks = ((total_rows + ob_row_tile - 1) // ob_row_tile) * (D // O_B_N_TILE)
    ob_workers = pl.max(1, pl.min(ob_tasks, worker_limit))
    with pl.spmd(ob_workers, name_hint="proj_o_b"):
        output = _o_proj_b(payload, wo_b, activation_scale, wo_b_scale, output, total_rows, LOCAL_O_WIDTH, ob_row_tile)
    return output


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
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Clear ready and consumed signals at a serialized entry boundary."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_signal_retire", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, [0, 0, 0])
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                arrived, peer=group_base + tp_rank, offsets=[peer, 0],
                value=0, op=pld.NotifyOp.Set,
            )
        pl.write(completion_anchor, [0, 0, 0], anchor)
    return completion_anchor


@pl.jit.inline(auto_scope=False)
def prefill_sp_output_reduce_scatter(
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Push contributions into source-rank receive slots and sum locally in FP32."""
    local_tokens = pl.tensor.dim(output, 0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived,
                offsets=[peer, 0], expected=(attention_epoch - 1) * 2, cmp=pld.WaitCmp.Ge,
            )
    with pl.spmd(64, name_hint="sp_reduce_publish", deps=[reuse_tid]) as copied_tid:
        worker = pl.tile.get_block_idx()
        for owner in pl.range(TP_SIZE):
            active = pl.max(0, pl.min(local_tokens, num_tokens - owner * local_tokens))
            blocks = (active + SP_PUBLISH_ROW_TILE - 1) // SP_PUBLISH_ROW_TILE
            for block in pl.range(worker, blocks, 64):
                local_row = block * SP_PUBLISH_ROW_TILE
                row = owner * local_tokens + local_row
                target_row = tp_rank * local_tokens + local_row
                if local_row + SP_PUBLISH_ROW_TILE <= active:
                    publish_value = pl.load(partial, [row, 0], [SP_PUBLISH_ROW_TILE, D])
                    pld.tile.remote_store(publish_value, output_window, group_base + owner, [target_row, 0])
                else:
                    for tail in pl.range(local_row, active):
                        tail_publish = pl.load(partial, [owner * local_tokens + tail, 0], [1, D])
                        target_tail = tp_rank * local_tokens + tail
                        pld.tile.remote_store(tail_publish, output_window, group_base + owner, [target_tail, 0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_notify", deps=[copied_tid]) as publish_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_wait", deps=[publish_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2 - 1, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="sp_reduce_scatter", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        local_peer = group_base + tp_rank
        for block in pl.range(worker, (local_tokens + SP_REDUCE_ROW_TILE - 1) // SP_REDUCE_ROW_TILE, 64):
            local_row = block * SP_REDUCE_ROW_TILE
            row = tp_rank * local_tokens + local_row
            if local_row + SP_REDUCE_ROW_TILE <= local_tokens and row + SP_REDUCE_ROW_TILE <= num_tokens:
                acc = pl.tile.full([SP_REDUCE_ROW_TILE, D], dtype=pl.FP32, value=0.0)
                for peer in pl.range(TP_SIZE):
                    source_row = peer * local_tokens + local_row
                    value = pld.tile.remote_load(output_window, local_peer, [source_row, 0], [SP_REDUCE_ROW_TILE, D])
                    acc = pl.add(acc, pl.cast(value, pl.FP32))
                result = pl.cast(acc, pl.BF16, mode="rint")
                output = pl.store(result, [local_row, 0], output)
            else:
                for tail in pl.range(local_row, pl.min(local_row + SP_REDUCE_ROW_TILE, local_tokens)):
                    target_row = tp_rank * local_tokens + tail
                    tail_acc = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
                    if target_row < num_tokens:
                        for peer in pl.range(TP_SIZE):
                            source_row = peer * local_tokens + tail
                            tail_value = pld.tile.remote_load(output_window, local_peer, [source_row, 0], [1, D])
                            tail_acc = pl.add(tail_acc, pl.cast(tail_value, pl.FP32))
                    tail_result = pl.cast(tail_acc, pl.BF16, mode="rint")
                    output = pl.store(tail_result, [tail, 0], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sp_reduce_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_publish", deps=[reuse_tid]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, num_tokens, 64):
            col = 0
            value = pl.load(output_partial, [row, col], [1, D])
            pld.tile.remote_store(value, output_window, peer=group_base + tp_rank, offsets=[row, col])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            ready_epoch = attention_epoch * 2 - 1
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=ready_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(64, name_hint="prefill_tp_reduce", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, num_tokens, 64):
            col = 0
            acc = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
            for peer in pl.range(TP_SIZE):
                value = pld.tile.remote_load(
                    output_window,
                    peer=group_base + peer, offsets=[row, col], shape=[1, D],
                )
                acc = pl.add(acc, pl.cast(value, pl.FP32))
            output = pl.store(pl.cast(acc, pl.BF16, mode="rint"), [row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit
def o_proj_validation(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        output, num_tokens, worker_limit=32, oa_row_tile=80, ob_row_tile=80,
    )
    return output

@pl.jit
def o_proj_with_hc_validation(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sublayer: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        sublayer, num_tokens, worker_limit=32, oa_row_tile=80, ob_row_tile=80,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(sublayer, residual, post_mix, residual_mix, output, deps=hc_post_deps)
    return sublayer, output


@pl.jit
def l2_quant_o_vf_pypto(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    payload: pl.InOut[pl.Tensor[[O_GROUP_ROWS, O_GROUP_IN], pl.FP8E4M3FN]],
    scales: pl.InOut[pl.Tensor[[LOCAL_O_GROUPS, O_IN_SCALE_COLS], pl.FP8E8M0]],
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    """Validation entry for the PyPTO quantizer; the model uses _quant_o_vf."""
    attended.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    payload.bind_dynamic(0, O_GROUP_ROWS)
    scales.bind_dynamic(1, O_IN_SCALE_COLS)
    quant_o_vf_pypto(attended, rope_cos, rope_sin, payload, scales, num_tokens, padded_rows)
    return payload, scales


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
        TensorSpec("sublayer", [tokens, D], torch.bfloat16),
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

    projected = dict(tensors, output=tensors["sublayer"])
    golden_o_proj_validation(projected)
    tensors["output"].copy_(golden_hc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"],
    ))


def compare_o_proj_hc_post(actual, _expected, *, actual_outputs, expected_outputs, inputs, **kwargs):
    """Check FP32 HC mixing independently of the separately checked BF16 projection."""
    from golden import ratio_allclose
    from hc import golden_hc_post

    expected = golden_hc_post(
        actual_outputs["sublayer"], inputs["residual"], inputs["post_mix"], inputs["residual_mix"],
    )
    compare = ratio_allclose(atol=1e-6, rtol=1e-5, max_error_ratio=0.0)
    return compare(
        actual, expected, actual_outputs=actual_outputs, expected_outputs=expected_outputs, inputs=inputs, **kwargs,
    )


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="V4.1 output projection branch and padding validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], default=TP_SIZE)
    parser.add_argument("--case", choices=["all", "empty", "fast", "tail", "hc-post"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    cases = (("empty", 0), ("fast", 12), ("tail", 17))
    for name, active in cases:
        if args.case not in ("all", name):
            continue
        print(f"[O_PROJ] {name}: active={active}", flush=True)
        active_compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128)

        def compare(actual, expected, **kwargs):
            if active == 0:
                zeroed = torch.equal(actual, expected)
                return zeroed, f"empty input zeros output={zeroed}"
            passed, detail = active_compare(actual[:active], expected[:active], **kwargs)
            inactive_ok = torch.equal(actual[active:], expected[active:])
            return passed and inactive_ok, f"{detail}; inactive rows zeroed={inactive_ok}"

        result = run(
            fn=o_proj_validation,
            specs=build_o_proj_specs(active),
            golden_fn=golden_o_proj_validation,
            compare_fn={"output": compare},
            config={
                "backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device,
            },
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
    if args.case in ("all", "hc-post"):
        for active, padded in ((72, False), (17, True), (0, True), (81, True)):
            print(f"[O_PROJ] hc-post: active={active}, padded={padded}, FP32 HC post", flush=True)
            result = run(
                fn=o_proj_with_hc_validation,
                specs=build_o_proj_hc_post_specs(active, pad_to_tile=padded),
                golden_fn=golden_o_proj_hc_post_validation,
                compare_fn={
                    "sublayer": ratio_allclose(atol=1e-2, rtol=1.0 / 128, valid_rows=active, zero_tail=True),
                    "output": compare_o_proj_hc_post,
                },
                config={
                    "backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device,
                },
                compile_only=args.compile_only,
                save_data=args.save_data,
            )
            if not result.passed:
                raise SystemExit(result.error or 1)
