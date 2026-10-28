# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compute-only routed MoE FFN with weighted SwiGLU and compact local-route output."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import pypto.language as pl

from config import FLASH as M
from grouped_mxfp8_mxfp4 import LOCAL_EXPERTS, build_route_fixture, build_smoke_fixture
from grouped_mxfp8_mxfp4 import generate_projection_weights, golden_grouped, tensor_checksum
from grouped_mxfp8_mxfp4 import grouped_mxfp8_mxfp4, grouped_mxfp8_mxfp4_dsl
from quant import pack_mx_a_scale, pack_mx_b_scale, quantize_mxfp8_upward_e8m0, unpack_mx_scale_codes


# model config
H = M.moe_intermediate_size
D = M.hidden_size
SWIGLU_LIMIT = M.swiglu_limit

# tiling
CHUNK_TILE = 64
ROW_TILE = 16
ACT_TILE = 256
QUANT_TILE = 256
QUANT_SCALE_TMP = ((64 + ACT_TILE // 32 + 31) // 32) * 32
RESTORE_TILE = 256


@pl.jit.inline(auto_scope=False)
def _weighted_swiglu(
    gate: pl.Tile[[ROW_TILE, ACT_TILE], pl.BF16],
    up: pl.Tile[[ROW_TILE, ACT_TILE], pl.BF16],
    route_weight: pl.Tile[[ROW_TILE, 1], pl.FP32],
):
    gate_fp32 = pl.cast(gate, pl.FP32)
    up_fp32 = pl.cast(up, pl.FP32)
    gate_clipped = pl.minimum(gate_fp32, SWIGLU_LIMIT)
    up_upper = pl.minimum(up_fp32, SWIGLU_LIMIT)
    up_clipped = pl.maximum(up_upper, -SWIGLU_LIMIT)
    gate_neg = pl.neg(gate_clipped)
    gate_exp = pl.exp(gate_neg)
    sigmoid_denominator = pl.add(gate_exp, 1.0)
    sigmoid = pl.recip(sigmoid_denominator)
    silu = pl.mul(gate_clipped, sigmoid)
    gate_up = pl.mul(silu, up_clipped)
    weighted = pl.row_expand_mul(gate_up, route_weight)
    hidden_bf16 = pl.cast(weighted, pl.BF16, mode="rint")
    hidden_fp32 = pl.cast(hidden_bf16, pl.FP32)
    return hidden_fp32


@pl.jit.inline(auto_scope=False)
def _quantize_upward(
    hidden: pl.Tile[[ROW_TILE, ACT_TILE], pl.FP32],
):
    values = pl.reshape(hidden, [ROW_TILE * (ACT_TILE // 32), 32])
    reduce_tmp = pl.create_tile([ROW_TILE * (ACT_TILE // 32), 32], dtype=pl.FP32)
    magnitude = pl.abs(values)
    maximum = pl.row_max(magnitude, tmp_tile=reduce_tmp)
    clamped_maximum = pl.maximum(maximum, 1e-4)
    scaled_maximum = pl.mul(clamped_maximum, 1.0 / 448.0)
    bits = pl.reinterpret_view(scaled_maximum, pl.INT32)
    ceil_bits = pl.add(bits, 8388607)
    exponent = pl.shrs(ceil_bits, 23)
    scale_bits = pl.shls(exponent, 23)
    scale = pl.reinterpret_view(scale_bits, pl.FP32)
    normalized = pl.row_expand_div(values, scale)
    lower = pl.maximum(normalized, -448.0)
    clipped = pl.minimum(lower, 448.0)
    quantized = pl.cast(clipped, pl.FP8E4M3FN, mode="rint")
    payload = pl.reshape(quantized, [ROW_TILE, ACT_TILE])
    sign = pl.shrs(exponent, 7)
    sign_offset = pl.mul(sign, 256)
    signed_exponent = pl.sub(exponent, sign_offset)
    codes_i8 = pl.cast(signed_exponent, pl.INT8)
    codes = pl.reinterpret_view(codes_i8, pl.UINT8)
    flat = pl.reshape(codes, [1, ROW_TILE * (ACT_TILE // 32)])
    temporary = pl.create_tile([1, QUANT_SCALE_TMP], dtype=pl.UINT8)
    packed = pl.tmov_x2zz(flat, temporary, group_axis=1, dst_rows=16, dst_cols=8)
    packed_scale = pl.reinterpret_view(packed, pl.FP8E8M0)
    return payload, packed_scale


@pl.jit.inline(auto_scope=False)
def act_quant(
    gate: pl.Tensor,
    up: pl.Tensor,
    route_weight: pl.Tensor,
    h: pl.Out[pl.Tensor],
    h_scale: pl.Out[pl.Tensor],
    gate_ready: pl.Scalar[pl.TASK_ID],
    up_ready: pl.Scalar[pl.TASK_ID],
):
    """Return (h, h_scale, task) for BF16[ROWS,H] inputs and FP32[ROWS,1] weights.

    ROWS is divisible by 64; H is a positive multiple of 256. Outputs
    are E4M3FN[ROWS,H] and E8M0[1,ROWS*(H/32)] in global MX_A_ZZ order.
    Padded input rows are finite with zero route weights.
    """
    rows = gate.shape[0]
    width = gate.shape[1]
    with pl.spmd(64, name_hint="ffn_gate_up_act_quant", deps=[gate_ready, up_ready]) as done:
        core = pl.tile.get_block_idx()
        for work in pl.range(core, (rows // ROW_TILE) * (width // QUANT_TILE), 64):
            row = work // (width // QUANT_TILE) * ROW_TILE
            col = work % (width // QUANT_TILE) * QUANT_TILE
            weights = pl.load(route_weight, [row, 0], [ROW_TILE, 1])
            gate_tile = pl.load(gate, [row, col], [ROW_TILE, ACT_TILE])
            up_tile = pl.load(up, [row, col], [ROW_TILE, ACT_TILE])
            hidden = _weighted_swiglu(gate_tile, up_tile, weights)
            payload, scale = _quantize_upward(hidden)
            h = pl.store(payload, [row, col], h)
            scale_offset = row * (width // 32) + (col // QUANT_TILE) * ROW_TILE * (QUANT_TILE // 32)
            scale_flat = pl.reshape(scale, [1, ROW_TILE * (QUANT_TILE // 32)])
            h_scale = pl.store(scale_flat, [0, scale_offset], h_scale)
    return h, h_scale, done


@pl.jit.inline(auto_scope=False)
def act_quant_experts(
    gate: pl.Tensor,
    up: pl.Tensor,
    route_weight: pl.Tensor,
    counts: pl.Tensor,
    h: pl.Out[pl.Tensor],
    h_scale: pl.Out[pl.Tensor],
    gate_ready: pl.Scalar[pl.TASK_ID],
    up_ready: pl.Scalar[pl.TASK_ID],
    experts: pl.constexpr,
    row_stride: pl.constexpr,
    width: pl.constexpr,
):
    """Quantize active fixed-capacity expert rows with zero-padded 16-row tails."""
    with pl.spmd(64, name_hint="ffn_expert_act_quant", deps=[gate_ready, up_ready]) as done:
        core = pl.tile.get_block_idx()
        for expert in pl.range(experts):
            count_read = pl.read(counts, [expert, 0])
            count = pl.cast(count_read, pl.INDEX)
            for work in pl.range(core, ((count + ROW_TILE - 1) // ROW_TILE) * (width // QUANT_TILE), 64):
                local_row = work // (width // QUANT_TILE) * ROW_TILE
                row = expert * row_stride + local_row
                col = work % (width // QUANT_TILE) * QUANT_TILE
                valid_rows = pl.min(ROW_TILE, count - local_row)
                gate_tile = pl.load(gate, [row, col], [ROW_TILE, ACT_TILE])
                up_tile = pl.load(up, [row, col], [ROW_TILE, ACT_TILE])
                weights = pl.load(route_weight, [row, 0], [ROW_TILE, 1])
                hidden_raw = _weighted_swiglu(gate_tile, up_tile, weights)
                hidden_valid = pl.set_validshape(hidden_raw, valid_rows, ACT_TILE)
                hidden_padded = pl.fillpad(hidden_valid, pad_value=pl.PadValue.zero)
                hidden = pl.set_validshape(hidden_padded, ROW_TILE, ACT_TILE)
                payload, scale = _quantize_upward(hidden)
                h = pl.store(payload, [row, col], h)
                scale_offset = row * (width // 32) + (col // QUANT_TILE) * ROW_TILE * (QUANT_TILE // 32)
                scale_flat = pl.reshape(scale, [1, ROW_TILE * (QUANT_TILE // 32)])
                h_scale = pl.store(scale_flat, [0, scale_offset], h_scale)
    return h, h_scale, done


@pl.jit.inline(auto_scope=False)
def local_restore(
    y_chunk: pl.Tensor,
    chunks: pl.Tensor,
    y_compact: pl.Out[pl.Tensor],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Return (y_compact, task) by copying valid BF16 route rows from 64-row chunks.

    chunks[C,4] stores (expert, stage_row, valid_rows, compact_row_start).
    Valid ranges partition y_compact[TOTAL,D]; D is divisible by 256. This
    stage restores local route rows without token aggregation or communication.
    """
    width = y_chunk.shape[1]
    chunk_count = chunks.shape[0]
    with pl.spmd(64, name_hint="ffn_local_restore", deps=[ready]) as done:
        core = pl.tile.get_block_idx()
        for work in pl.range(core, chunk_count * (width // RESTORE_TILE), 64):
            chunk = work // (width // RESTORE_TILE)
            col = work % (width // RESTORE_TILE) * RESTORE_TILE
            stage_read = pl.read(chunks, [chunk, 1])
            stage_row = pl.cast(stage_read, pl.INDEX)
            valid_read = pl.read(chunks, [chunk, 2])
            valid = pl.cast(valid_read, pl.INDEX)
            compact_read = pl.read(chunks, [chunk, 3])
            compact_row = pl.cast(compact_read, pl.INDEX)
            for row in pl.range(0, valid, ROW_TILE):
                valid_rows = pl.min(ROW_TILE, valid - row)
                tile = pl.load(y_chunk, [stage_row + row, col], [ROW_TILE, RESTORE_TILE])
                tile_valid = pl.set_validshape(tile, valid_rows, RESTORE_TILE)
                y_compact = pl.store(tile_valid, [compact_row + row, col], y_compact)
    return y_compact, done


def make_moe_ffn(width, hidden, experts, chunk_count, rows, *, use_cpp):
    """Bind head/down dimensions and the implementation before JIT specialization."""
    projection = grouped_mxfp8_mxfp4 if use_cpp else grouped_mxfp8_mxfp4_dsl

    @pl.jit(auto_scope=False)
    def moe_ffn(
        x: pl.Tensor,
        x_scale: pl.Tensor,
        route_weights: pl.Tensor,
        chunks: pl.Tensor,
        w1: pl.Tensor,
        w1_scale: pl.Tensor,
        w3: pl.Tensor,
        w3_scale: pl.Tensor,
        w2: pl.Tensor,
        w2_scale: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ):
        ready = pl.system.task_dummy(deps=[])
        gate = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        gate, gate_ready = projection(
            x, x_scale, w1, w1_scale, chunks, gate, ready,
            width, hidden, experts, chunk_count, rows,
        )
        up = pl.create_tensor([rows, hidden], dtype=pl.BF16)
        up, up_ready = projection(
            x, x_scale, w3, w3_scale, chunks, up, ready,
            width, hidden, experts, chunk_count, rows,
        )
        h = pl.create_tensor([rows, hidden], dtype=pl.FP8E4M3FN)
        h_scale = pl.create_tensor([1, rows * (hidden // 32)], dtype=pl.FP8E8M0)
        h, h_scale, act_ready = act_quant(gate, up, route_weights, h, h_scale, gate_ready, up_ready)
        down = pl.create_tensor([rows, width], dtype=pl.BF16)
        down, down_ready = projection(
            h, h_scale, w2, w2_scale, chunks, down, act_ready,
            hidden, width, experts, chunk_count, rows,
        )
        out, restore_ready = local_restore(down, chunks, out, down_ready)
        return out

    return moe_ffn


@pl.jit(auto_scope=False)
def act_quant_test(
    gate: pl.Tensor,
    up: pl.Tensor,
    route_weight: pl.Tensor,
    h: pl.Out[pl.Tensor],
    h_scale: pl.Out[pl.Tensor],
    hidden_bf16: pl.Out[pl.Tensor],
):
    gate_ready = pl.system.task_dummy(deps=[])
    up_ready = pl.system.task_dummy(deps=[])
    h, h_scale, done = act_quant(gate, up, route_weight, h, h_scale, gate_ready, up_ready)
    rows = gate.shape[0]
    width = gate.shape[1]
    with pl.spmd(64, name_hint="ffn_activation_validation", deps=[gate_ready, up_ready]):
        core = pl.tile.get_block_idx()
        for work in pl.range(core, (rows // ROW_TILE) * (width // ACT_TILE), 64):
            row = work // (width // ACT_TILE) * ROW_TILE
            col = work % (width // ACT_TILE) * ACT_TILE
            gate_tile = pl.load(gate, [row, col], [ROW_TILE, ACT_TILE])
            up_tile = pl.load(up, [row, col], [ROW_TILE, ACT_TILE])
            weights = pl.load(route_weight, [row, 0], [ROW_TILE, 1])
            hidden = _weighted_swiglu(gate_tile, up_tile, weights)
            rounded_hidden = pl.cast(hidden, pl.BF16, mode="rint")
            hidden_bf16 = pl.store(rounded_hidden, [row, col], hidden_bf16)
    return h, h_scale, hidden_bf16


@pl.jit(auto_scope=False)
def quantize_test(
    hidden: pl.Tensor,
    h: pl.Out[pl.Tensor],
    h_scale: pl.Out[pl.Tensor],
):
    rows = hidden.shape[0]
    with pl.spmd(64, name_hint="ffn_quantization_edges"):
        core = pl.tile.get_block_idx()
        for row in pl.range(core * ROW_TILE, rows, 64 * ROW_TILE):
            source = pl.load(hidden, [row, 0], [ROW_TILE, ACT_TILE])
            payload, scale = _quantize_upward(source)
            h = pl.store(payload, [row, 0], h)
            offset = row * (ACT_TILE // 32)
            flat = pl.reshape(scale, [1, ROW_TILE * (ACT_TILE // 32)])
            h_scale = pl.store(flat, [0, offset], h_scale)
    return h, h_scale


@pl.jit(auto_scope=False)
def local_restore_test(
    y_chunk: pl.Tensor,
    chunks: pl.Tensor,
    y_compact: pl.Out[pl.Tensor],
):
    ready = pl.system.task_dummy(deps=[])
    y_compact, done = local_restore(y_chunk, chunks, y_compact, ready)
    return y_compact


def golden_act_quant(tensors):
    """Round weighted, clipped SwiGLU to BF16, then quantize in blocks of 32."""
    gate = tensors["gate"].float().clamp(max=SWIGLU_LIMIT)
    up = tensors["up"].float().clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    hidden = (F.silu(gate) * up * tensors["route_weight"].float()).to(torch.bfloat16)
    tensors["hidden_bf16"][:] = hidden
    payload, codes = quantize_mxfp8_upward_e8m0(hidden.float())
    tensors["h"].view(torch.uint8)[:] = payload.view(torch.uint8)
    tensors["h_scale"].view(torch.uint8)[:] = pack_mx_a_scale(codes).reshape(1, -1)


def golden_quantize(tensors):
    """Use the current upward E8M0 oracle for unrounded FP32 scale boundaries."""
    payload, codes = quantize_mxfp8_upward_e8m0(tensors["hidden"])
    tensors["h"].view(torch.uint8)[:] = payload.view(torch.uint8)
    tensors["h_scale"].view(torch.uint8)[:] = pack_mx_a_scale(codes).reshape(1, -1)


def golden_local_restore(tensors):
    """Restore only metadata-defined valid local rows into compact route order."""
    for _, stage_row, valid, compact_row in tensors["chunks"].tolist():
        tensors["y_compact"][compact_row:compact_row + valid] = tensors["y_chunk"][stage_row:stage_row + valid]


def golden_ffn(tensors):
    """Compose independent projection goldens with the BF16 weighted SwiGLU boundary."""
    rows = tensors["x"].shape[0]
    hidden = tensors["w1_scale"].shape[1]
    gate = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    up = torch.zeros_like(gate)
    for name, output in (("w1", gate), ("w3", up)):
        golden_grouped(dict(
            x=tensors["x"], x_scale=tensors["x_scale"],
            weight=tensors[name], weight_scale=tensors[f"{name}_scale"], chunks=tensors["chunks"], out=output,
        ))
    activation = dict(
        gate=gate, up=up, route_weight=tensors["route_weights"], hidden_bf16=torch.empty_like(gate),
        h=torch.empty(rows, hidden, dtype=torch.float8_e4m3fn),
        h_scale=torch.empty(1, rows * (hidden // 32), dtype=torch.float8_e8m0fnu),
    )
    golden_act_quant(activation)
    down = torch.zeros(rows, tensors["x"].shape[1], dtype=torch.bfloat16)
    golden_grouped(dict(
        x=activation["h"], x_scale=activation["h_scale"],
        weight=tensors["w2"], weight_scale=tensors["w2_scale"], chunks=tensors["chunks"], out=down,
    ))
    golden_local_restore(dict(y_chunk=down, chunks=tensors["chunks"], y_compact=tensors["out"]))


def build_ffn_specs(scenario="smoke", golden_data=None):
    """Build frozen pregrouped inputs and all three tiled projection-weight tables."""
    from golden import TensorSpec

    dtypes = (
        ("x", torch.float8_e4m3fn), ("x_scale", torch.float8_e8m0fnu),
        ("route_weights", torch.float32), ("chunks", torch.int32),
        ("w1", torch.uint8), ("w1_scale", torch.float8_e8m0fnu),
        ("w3", torch.uint8), ("w3_scale", torch.float8_e8m0fnu),
        ("w2", torch.uint8), ("w2_scale", torch.float8_e8m0fnu),
    )
    if golden_data:
        data = Path(golden_data)
        values = {name: torch.load(data / "in" / f"{name}.pt", mmap=True, weights_only=True) for name, _ in dtypes}
        output = torch.load(data / "out" / "out.pt", mmap=True, weights_only=True)
        specs = [TensorSpec(name, list(values[name].shape), dtype) for name, dtype in dtypes]
        return specs + [TensorSpec("out", list(output.shape), torch.bfloat16)], None

    fixture = build_smoke_fixture() if scenario == "smoke" else build_route_fixture(scenario)
    rows, width = fixture["x"].shape
    hidden = 256 if scenario == "smoke" else H
    experts = 3 if scenario == "smoke" else LOCAL_EXPERTS
    cache = {}

    def get_weights(projection):
        if projection not in cache:
            if scenario == "smoke":
                weight = fixture["weight"]
                codes = unpack_mx_scale_codes(fixture["weight_scale"], side="b")
                if projection == "w1":
                    pair = weight, fixture["weight_scale"]
                elif projection == "w3":
                    packed = torch.roll(weight, shifts=37, dims=1).bitwise_xor(0xA5)
                    changed = (122 + (codes.to(torch.int16) - 123 + 2).remainder(7)).to(torch.uint8)
                    pair = packed, pack_mx_b_scale(changed).view(torch.float8_e8m0fnu)
                else:
                    packed = torch.roll(weight.flip(1), shifts=19, dims=2).bitwise_xor(0x3C)
                    changed = (121 + (codes.to(torch.int16) - 123 + 4).remainder(7)).to(torch.uint8)
                    pair = packed, pack_mx_b_scale(changed).view(torch.float8_e8m0fnu)
            else:
                k, n = (hidden, width) if projection == "w2" else (width, hidden)
                pair = generate_projection_weights(experts, k, n, projection=projection)
            cache[projection] = pair
            fixture[projection], fixture[f"{projection}_scale"] = pair
            checksums = fixture["metadata"]["checksums"]
            checksums[projection] = tensor_checksum(pair[0])
            checksums[f"{projection}_scale"] = tensor_checksum(pair[1])
            print(json.dumps({name: checksums[name] for name in (projection, f"{projection}_scale")}), flush=True)
        return cache[projection]

    metadata = fixture["metadata"]
    metadata.update(
        input_contract="pregrouped MXFP8; compute only; compact local-route BF16 output",
        rows=rows, input_width=width, intermediate_width=hidden, local_experts=experts,
        active_rows=int(fixture["count_e"].sum()), chunk_count=fixture["chunks"].shape[0],
        excluded="gate, dispatch, input quantization, shared experts, communication, across-shard aggregation",
    )
    metadata["checksums"] = {
        name: tensor_checksum(fixture[name]) for name in ("x", "x_scale", "route_weights", "chunks")
    }
    specs = [
        TensorSpec("x", [rows, width], torch.float8_e4m3fn, init_value=lambda: fixture["x"]),
        TensorSpec("x_scale", [1, rows * (width // 32)], torch.float8_e8m0fnu, init_value=lambda: fixture["x_scale"]),
        TensorSpec("route_weights", [rows, 1], torch.float32, init_value=lambda: fixture["route_weights"]),
        TensorSpec("chunks", list(fixture["chunks"].shape), torch.int32, init_value=lambda: fixture["chunks"]),
        TensorSpec("w1", [experts, width * hidden // 256, 128], torch.uint8, init_value=lambda: get_weights("w1")[0]),
        TensorSpec("w1_scale", [experts * (width // 32), hidden], torch.float8_e8m0fnu, init_value=lambda: get_weights("w1")[1]),
        TensorSpec("w3", [experts, width * hidden // 256, 128], torch.uint8, init_value=lambda: get_weights("w3")[0]),
        TensorSpec("w3_scale", [experts * (width // 32), hidden], torch.float8_e8m0fnu, init_value=lambda: get_weights("w3")[1]),
        TensorSpec("w2", [experts, hidden * width // 256, 128], torch.uint8, init_value=lambda: get_weights("w2")[0]),
        TensorSpec("w2_scale", [experts * (hidden // 32), width], torch.float8_e8m0fnu, init_value=lambda: get_weights("w2")[1]),
        TensorSpec("out", [int(fixture["count_e"].sum()), width], torch.bfloat16),
    ]
    return specs, fixture


def _chunk_fixture():
    counts = [0, 1, 15, 16, 17, 63, 64, 65]
    chunks = []
    compact = 0
    for expert, count in enumerate(counts):
        for start in range(0, count, CHUNK_TILE):
            chunks.append([expert, len(chunks) * CHUNK_TILE, min(CHUNK_TILE, count - start), compact + start])
        compact += count
    # Descriptor order differs from both expert and compact order.
    chunks = torch.tensor(chunks, dtype=torch.int32).flip(0).contiguous()
    active = torch.zeros(len(chunks) * CHUNK_TILE, dtype=torch.bool)
    for _, row, valid, _ in chunks.tolist():
        active[row:row + valid] = True
    return chunks, active, compact


def build_act_specs(width=H):
    """Cover clipping, signed weights, zero padding and model/smoke widths."""
    from golden import TensorSpec

    if width != ACT_TILE and (width <= 0 or width % QUANT_TILE):
        raise ValueError("Activation width must be 256 or a positive multiple of 768")
    _, active, _ = _chunk_fixture()
    rows = active.numel()
    generator = torch.Generator().manual_seed(20261028)
    gate = (torch.randn(rows, width, generator=generator) * 7.0).to(torch.bfloat16)
    up = (torch.randn(rows, width, generator=generator) * 9.0).to(torch.bfloat16)
    weights = torch.rand(rows, 1, generator=generator) * 3.0 - 0.5
    weights[~active] = 0.0
    gate[:, :8] = torch.tensor([-100.0, -10.0, -1.0, -0.0, 0.0, 1.0, 10.0, 100.0])
    up[:, :8] = torch.tensor([-100.0, -10.0, -1.0, -0.0, 0.0, 1.0, 10.0, 100.0])
    weights[0, 0] = 0.0
    return [
        TensorSpec("gate", [rows, width], torch.bfloat16, init_value=gate),
        TensorSpec("up", [rows, width], torch.bfloat16, init_value=up),
        TensorSpec("route_weight", [rows, 1], torch.float32, init_value=weights),
        TensorSpec("h", [rows, width], torch.float8_e4m3fn),
        TensorSpec("h_scale", [1, rows * (width // 32)], torch.float8_e8m0fnu),
        TensorSpec("hidden_bf16", [rows, width], torch.bfloat16),
    ]


def build_quant_specs():
    """Cover zero/floor blocks and FP32 neighbors of upward-scale thresholds."""
    from golden import TensorSpec

    boundaries = torch.tensor([448.0 * 2.0**power for power in (-22, -12, -3, 0, 3, 12)])
    lower = torch.nextafter(boundaries, torch.zeros_like(boundaries))
    upper = torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))
    maxima = torch.cat((torch.tensor([0.0, 1e-8, 1e-4]), lower, boundaries, upper))
    value = torch.zeros(CHUNK_TILE, ACT_TILE, dtype=torch.float32)
    ramp = torch.linspace(-1.0, 1.0, 32)
    for index in range(CHUNK_TILE * (ACT_TILE // 32)):
        value.reshape(-1, 32)[index] = ramp * maxima[index % maxima.numel()]
    return [
        TensorSpec("hidden", list(value.shape), torch.float32, init_value=value),
        TensorSpec("h", list(value.shape), torch.float8_e4m3fn),
        TensorSpec("h_scale", [1, value.numel() // 32], torch.float8_e8m0fnu),
    ]


def build_restore_specs(width=D):
    """Cover empty experts, 1/15/16/17/63/64 tails and a 65-row expert."""
    from golden import TensorSpec

    if width <= 0 or width % RESTORE_TILE:
        raise ValueError("Restore width must be a positive multiple of 256")
    chunks, active, total = _chunk_fixture()
    generator = torch.Generator().manual_seed(20261029)
    value = torch.randn(active.numel(), width, generator=generator).to(torch.bfloat16)
    value[~active] = 123.0
    return [
        TensorSpec("y_chunk", list(value.shape), torch.bfloat16, init_value=value),
        TensorSpec("chunks", list(chunks.shape), torch.int32, init_value=chunks),
        TensorSpec("y_compact", [total, width], torch.bfloat16),
    ]


def _exact_bytes(actual, expected, **context):
    from golden import ratio_allclose

    compare = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    return compare(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8), **context)


def compare_activation(actual, expected, **context):
    """Exclude zero-weight padding from the ratio and require exact zero there."""
    from golden import ratio_reldiff

    weights = context["inputs"]["route_weight"].cpu().reshape(-1)
    active = weights != 0
    if torch.count_nonzero(actual[~active].float()).item():
        return False, "Zero-weight rows contain nonzero or nonfinite activations"
    compare = ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)
    return compare(actual[active], expected[active], **context)


def compare_act_payload(actual, expected, **context):
    """Check FP8 bytes against the oracle applied to the observed BF16 boundary."""
    hidden = context["actual_outputs"]["hidden_bf16"].cpu()
    if not torch.isfinite(hidden.float()).all():
        return False, "Nonfinite BF16 activation boundary"
    payload, _ = quantize_mxfp8_upward_e8m0(hidden)
    return _exact_bytes(actual, payload, **context)


def compare_act_scale(actual, expected, **context):
    """Check global MX_A_ZZ bytes against the observed BF16 boundary."""
    hidden = context["actual_outputs"]["hidden_bf16"].cpu()
    if not torch.isfinite(hidden.float()).all():
        return False, "Nonfinite BF16 activation boundary"
    _, codes = quantize_mxfp8_upward_e8m0(hidden)
    packed = pack_mx_a_scale(codes).reshape(1, -1)
    return _exact_bytes(actual, packed, **context)


def _host_validate(stage, specs, golden_fn, comparators):
    """Validate finite fixtures, oracle boundaries, and comparator rejection paths."""
    from golden import validate_golden

    inputs = {spec.name: spec.create_tensor() for spec in specs}
    golden_fn(inputs)
    names = set(comparators)
    outputs = {name: inputs[name].clone() for name in names}
    validate_golden(outputs, inputs, compare_fn=comparators, inputs=inputs)
    if stage == "act":
        context = dict(actual_outputs=outputs, expected_outputs=outputs, inputs=inputs, rtol=0.0, atol=0.0)
        active_rows = torch.nonzero(inputs["route_weight"].reshape(-1) != 0).flatten()
        for bad in (float("nan"), float("inf")):
            changed = {name: value.clone() for name, value in outputs.items()}
            changed["hidden_bf16"][active_rows[0], 0] = bad
            ok, _ = compare_activation(changed["hidden_bf16"], outputs["hidden_bf16"], **context)
            if ok:
                raise ValueError("Activation comparator accepted a nonfinite output")
        changed = outputs["h"].clone()
        changed.view(torch.uint8)[0, 0] ^= 1
        ok, _ = compare_act_payload(changed, outputs["h"], **context)
        if ok:
            raise ValueError("Payload comparator accepted a changed byte")
    print(f"[HOST] {stage} fixture/oracle/comparator checks passed")


def _host_validate_ffn(specs, golden_data=None):
    """Check the independent full-FFN oracle and reject nonfinite compact outputs."""
    from golden import ratio_reldiff

    if golden_data:
        data = Path(golden_data)
        tensors = {
            spec.name: torch.load(data / "in" / f"{spec.name}.pt", weights_only=True)
            for spec in specs if spec.name != "out"
        }
        tensors["out"] = specs[-1].create_tensor()
    else:
        tensors = {spec.name: spec.create_tensor() for spec in specs}
    if not torch.isfinite(tensors["route_weights"]).all():
        raise ValueError("Nonfinite route weights")
    for _, row, valid, _ in tensors["chunks"].tolist():
        if torch.count_nonzero(tensors["route_weights"][row + valid:row + CHUNK_TILE]).item():
            raise ValueError("Padded routes must have zero weight")
    golden_ffn(tensors)
    expected = tensors["out"].clone()
    compare = ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)
    context = dict(actual_outputs={"out": tensors["out"]}, expected_outputs={"out": expected})
    context.update(inputs=tensors, rtol=0.0, atol=0.0)
    ok, message = compare(tensors["out"], expected, **context)
    if not ok:
        raise ValueError(message)
    for bad in (float("nan"), float("inf")):
        changed = expected.clone()
        changed[0, 0] = bad
        ok, _ = compare(changed, expected, **context)
        if ok:
            raise ValueError("Full-FFN comparator accepted a nonfinite output")
    if not torch.count_nonzero(expected).item():
        raise ValueError("Full-FFN smoke oracle produced an all-zero output")
    if golden_data:
        saved = torch.load(Path(golden_data) / "out" / "out.pt", weights_only=True)
        ok, message = compare(expected, saved, **context)
        if not ok:
            raise ValueError(f"Independent full-FFN oracle disagrees with saved data: {message}")
    print(f"[HOST] independent full-FFN oracle passed; output_sha256={tensor_checksum(expected)}")


def validate(argv=None):
    """Run full compute-only FFN smoke by default or explicitly selected stage cases."""
    from golden import ratio_allclose, ratio_reldiff, run
    from pypto import passes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=["cpp", "dsl"], default="cpp")
    parser.add_argument("--scenario", choices=["smoke", "decode", "prefill-full", "prefill-tail"], default="smoke")
    parser.add_argument("--validate-act", action="store_true")
    parser.add_argument("--validate-restore", action="store_true")
    parser.add_argument("--validate-quant", action="store_true", help="Run the isolated quantization edge case")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--width", type=int, default=None, help="Select one supported stage width")
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--memory-planner", choices=["pypto", "dsa-rp"], default="pypto")
    args = parser.parse_args(argv)
    planner = passes.MemoryPlanner.PYPTO if args.memory_planner == "pypto" else passes.MemoryPlanner.DSA_RP
    if not (args.validate_act or args.validate_restore or args.validate_quant):
        if args.width is not None:
            parser.error("--width selects an independent stage; use --scenario for full FFN")
        specs, fixture = build_ffn_specs(args.scenario, golden_data=args.golden_data)
        if fixture is not None:
            print(json.dumps(fixture["metadata"], sort_keys=True), flush=True)
        if args.host_only:
            _host_validate_ffn(specs, args.golden_data)
            if fixture is not None:
                print(json.dumps(fixture["metadata"], sort_keys=True), flush=True)
            return True
        entry = make_moe_ffn(
            specs[0].shape[1], specs[5].shape[1], specs[4].shape[0],
            specs[3].shape[0], specs[0].shape[0], use_cpp=args.impl == "cpp",
        )
        result = run(
            fn=entry,
            specs=specs,
            golden_fn=golden_ffn,
            golden_data=args.golden_data,
            save_data=args.save_data,
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            config=dict(
                backend_type=args.platform,
                device_id=args.device,
                execution_mode=args.sim,
                memory_planner=planner,
                dump_passes=args.dump_passes,
                enable_dep_gen=args.enable_dep_gen,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            compare_fn={"out": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)},
        )
        if fixture is not None:
            print(json.dumps(fixture["metadata"], sort_keys=True), flush=True)
        if not result.passed and result.error:
            print(result.error)
        return result.passed

    print("Independent MoE stage validation (activation/quantization and local restore)")
    stages = []
    if args.validate_act:
        widths = [args.width] if args.width is not None else [ACT_TILE, H]
        for width in widths:
            compare = {"h": compare_act_payload, "h_scale": compare_act_scale, "hidden_bf16": compare_activation}
            stages.append((f"act-{width}", "act", act_quant_test, build_act_specs(width), golden_act_quant, compare))
    if args.validate_quant or (args.validate_act and args.width is None):
        compare = {"h": _exact_bytes, "h_scale": _exact_bytes}
        stages.append(("quant-edges", "quant", quantize_test, build_quant_specs(), golden_quantize, compare))
    if args.validate_restore:
        widths = [args.width] if args.width is not None else [RESTORE_TILE, D]
        for width in widths:
            compare = {"y_compact": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)}
            restore_specs = build_restore_specs(width)
            case = (f"restore-{width}", "restore", local_restore_test, restore_specs, golden_local_restore, compare)
            stages.append(case)
    if len(stages) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("Replay requires one stage and --width, or --validate-quant")
    results = []
    for label, stage, fn, specs, golden_fn, compare in stages:
        print(f"[STAGE] {label}")
        if args.host_only:
            _host_validate(stage, specs, golden_fn, compare)
            continue
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            golden_data=args.golden_data,
            save_data=args.save_data,
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            config=dict(
                backend_type=args.platform,
                device_id=args.device,
                execution_mode=args.sim,
                memory_planner=planner,
                dump_passes=args.dump_passes,
                enable_dep_gen=args.enable_dep_gen,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn=compare,
        )
        results.append(result)
        if not result.passed:
            if result.error:
                print(result.error)
            return False
    return all(result.passed for result in results)


def main():
    if not validate():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
