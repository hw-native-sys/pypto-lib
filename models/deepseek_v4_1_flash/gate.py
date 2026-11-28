# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DeepSeek-V4.1 A5 MoE router: BF16 input, MX quantization, projection, and top-k weights."""

import pypto.language as pl
import torch
import torch.nn.functional as F

from config import FLASH as M

from config import MOE_TOKENS, EP_SIZE, N_LOCAL_EXPERTS


# model config
T = MOE_TOKENS
D = M.hidden_size
BIAS_DYN = pl.dynamic("GATE_BIAS_DYN")
PAIR_WIDTH_DYN = pl.dynamic("GATE_PAIR_WIDTH")
SCORE_COLUMNS_DYN = pl.dynamic("GATE_SCORE_COLUMNS")
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor

# tiling
GATE_T_TILE = 4
GATE_M_TILE = 16
GATE_N_TILE = 32
T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
QUANT_TILE = 256
QUANT_TASK_TILE = QUANT_TILE
assert D % QUANT_TASK_TILE == 0
MX_GROUP = 32
MX_SCALE_GROUPS = D // MX_GROUP
QUANT_WORKER_TILE = 24 if T > 80 else 64
ROUTE_WORKER_TILE = 24 if T > 80 else 64
TOPK_PAD = 8
SORT_PAD = TOPK_PAD * 2


@pl.jit.inline
def gather_routing_bias(
    source: pl.Tensor,
    row_ids: pl.Tensor,
    output: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID],
    n_experts: pl.constexpr,
    refresh_row_ids: pl.constexpr = False,
):
    """Gather token routing biases or copy one broadcast expert vector."""
    source_rows = pl.tensor.dim(source, 0) // n_experts
    output_rows = pl.tensor.dim(output, 0) // n_experts
    source_matrix = pl.reshape(source, [source_rows, n_experts])
    output_matrix = pl.reshape(output, [output_rows, n_experts])
    for row in pl.spmd(output_rows, name_hint="gate_bias_gather", deps=[ready]):
        source_row = pl.cast(0, pl.INT32)
        if source_rows > 1:
            if refresh_row_ids:
                pl.system.cacheinvalid()
            source_row = pl.read(row_ids, [row])
        for column in pl.range(0, n_experts, 32):
            valid_columns = pl.min(32, n_experts - column)
            if source_row >= 0 and source_row < source_rows:
                value = pl.load(source_matrix, [source_row, column], [1, 32], valid_shape=[1, valid_columns])
                pl.store(value, [row, column], output_matrix)
            else:
                zero = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
                pl.store(pl.set_validshape(zero, 1, valid_columns), [row, column], output_matrix)
    return output


@pl.jit.inline(auto_scope=False)
def _gate_prefixes(
    width: pl.constexpr,
    pairs: pl.Tile[[GATE_T_TILE, PAIR_WIDTH_DYN], pl.FP32, pl.Mem.Vec],
    stride: pl.constexpr,
    lists: pl.constexpr,
):
    """Merge sorted expert lists into one top-k prefix per row."""
    prefixes = pl.create_tile([GATE_T_TILE, SORT_PAD], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    for local in pl.unroll(GATE_T_TILE):
        first = pl.tile.slice(pairs, [1, SORT_PAD], [local, 0])
        if lists == 1:
            prefix = first
        else:
            second = pl.tile.slice(pairs, [1, SORT_PAD], [local, stride])
            scratch = pl.create_tile([1, SORT_PAD * 3], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            if lists == 2:
                merged = pl.tile.mrgsort(first, second, tmp=scratch)
            else:
                third = pl.tile.slice(pairs, [1, SORT_PAD], [local, 2 * stride])
                merged = pl.tile.mrgsort(first, second, third, tmp=scratch)
            prefix = pl.tile.slice(merged, [1, SORT_PAD], [0, 0])
        prefixes = pl.tile.assemble(prefixes, prefix, [local, 0])
    return prefixes


@pl.jit.inline(auto_scope=False)
def _gate_rank_rows(
    columns: pl.constexpr,
    raw: pl.Tile[[GATE_T_TILE, SCORE_COLUMNS_DYN], pl.FP32, pl.Mem.Vec],
    gate_bias: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    experts: pl.constexpr,
):
    """Apply router scores and bias, then select each row's strongest experts."""
    valid_columns = pl.min(columns, experts - column)
    bias_rows = pl.tensor.dim(gate_bias, 0)
    if bias_rows == 1:
        bias_row = pl.load(gate_bias, [0, column], [1, columns], valid_shape=[1, valid_columns], target_memory=pl.Mem.Vec)
        bias_row = pl.set_validshape(pl.tile.fillpad(bias_row, pad_value=pl.PadValue.min), 1, columns)
        storage = pl.create_tile([GATE_T_TILE, columns], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        expanded = pl.tile.col_expand(storage, bias_row)
        bias = pl.tile.fillpad(expanded, pad_value=pl.PadValue.min)
    else:
        rows = pl.max(0, pl.min(GATE_T_TILE, bias_rows - row))
        token_bias = pl.load(gate_bias, [row, column], [GATE_T_TILE, columns], valid_shape=[rows, valid_columns], target_memory=pl.Mem.Vec)
        bias = pl.set_validshape(pl.tile.fillpad(token_bias, pad_value=pl.PadValue.min), GATE_T_TILE, columns)
    bias_flat = pl.reshape(bias, [GATE_T_TILE * columns])
    raw_flat = pl.reshape(raw, [GATE_T_TILE * columns])
    scores = pl.create_tile([GATE_T_TILE * columns], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    biased = pl.create_tile([GATE_T_TILE * columns], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=64)
        zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
        for segment in pl.hardware_loop(GATE_T_TILE * columns // 64):
            logits = pl.reg.load(raw_flat, segment * 64, lanes=64)
            bias_value = pl.reg.load(bias_flat, segment * 64, lanes=64)
            relu = pl.reg.maximum(logits, zero, full)
            absolute = pl.reg.abs(logits, full)
            tail = pl.reg.muls(absolute, -1.0, full)
            tail = pl.reg.exp(tail, full)
            tail = pl.reg.adds(tail, 1.0, full)
            tail = pl.reg.log(tail, full)
            tail = pl.reg.add(relu, tail, full)
            mask = pl.reg.muls(logits, -1.0, full)
            mask = pl.reg.adds(mask, -10.0, full)
            mask = pl.reg.maximum(mask, zero, full)
            mask = pl.reg.minimum(mask, one, full)
            floor = pl.reg.minimum(logits, zero, full)
            floor = pl.reg.exp(floor, full)
            floor = pl.reg.mul(mask, floor, full)
            score = pl.reg.maximum(tail, floor, full)
            score = pl.reg.sqrt(score, full)
            _scores_written = pl.reg.store(scores, segment * 64, score)
            selection = pl.reg.add(score, bias_value, full)
            _biased_written = pl.reg.store(biased, segment * 64, selection)
    biased = pl.reshape(biased, [GATE_T_TILE, columns])
    range_row = pl.tile.arange(pl.cast(column, pl.UINT32), [1, columns], dtype=pl.UINT32)
    range_tile = pl.create_tile([GATE_T_TILE, columns], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
    range_tile = pl.tile.col_expand(range_tile, range_row)
    sorted32 = pl.tile.sort32(biased, range_tile)
    if columns > 32:
        sorted_flat = pl.reshape(sorted32, [1, GATE_T_TILE * columns * 2])
        sorted64 = pl.tile.mrgsort(sorted_flat, block_len=64)
        sorted64 = pl.reshape(sorted64, [GATE_T_TILE, columns * 2])
        prefixes = _gate_prefixes(columns * 2, sorted64, 256, (columns + 127) // 128)
    else:
        prefixes = _gate_prefixes(columns * 2, sorted32, 256, 1)
    return prefixes, pl.reshape(scores, [GATE_T_TILE, columns])


@pl.jit.inline(auto_scope=False)
def _gate_store_routes(
    columns: pl.constexpr,
    prefixes: pl.Tile[[GATE_T_TILE, SORT_PAD], pl.FP32, pl.Mem.Vec],
    scores: pl.Tile[[GATE_T_TILE, SCORE_COLUMNS_DYN], pl.FP32, pl.Mem.Vec],
    indices: pl.Tensor,
    weights: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    topk: pl.constexpr,
):
    """Gather unbiased scores for the selected expert IDs and normalize their weights."""
    pairs = pl.reinterpret_view(pl.reshape(prefixes, [GATE_T_TILE * SORT_PAD]), pl.INT32)
    score_flat = pl.reshape(scores, [GATE_T_TILE * columns])
    index_values = pl.create_tile([1, 64], dtype=pl.INT32, target_memory=pl.Mem.Vec)
    weight_values = pl.create_tile([1, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    index_values = pl.reshape(index_values, [64])
    weight_values = pl.reshape(weight_values, [64])
    with pl.reg_scope():
        full = pl.reg.active_mask(GATE_T_TILE * TOPK_PAD, lanes=64)
        lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
        token = pl.reg.shift_rights(lane, 3, full)
        low_bits = pl.reg.broadcast(7, lanes=64, dtype=pl.UINT32)
        selected_column = pl.reg.bitwise_and(lane, low_bits, full)
        selected_mask = pl.reg.compares(selected_column, topk, full, predicate="lt")
        pair_offset = pl.reg.muls(token, SORT_PAD, full)
        pair_offset = pl.reg.add(pair_offset, pl.reg.muls(selected_column, 2, full), full)
        pair_offset = pl.reg.adds(pair_offset, 1, full)
        selected = pl.reg.gather(pairs, pair_offset, full)
        _indices_written = pl.reg.store(index_values, 0, selected, full)
        score_offset = pl.reg.muls(token, columns, full)
        score_offset = pl.reg.add(score_offset, pl.reg.interpret_cast(selected, dtype=pl.UINT32), full)
        value = pl.reg.gather(score_flat, score_offset, selected_mask)
        zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        value = pl.reg.select(selected_mask, value, zero)
        sum_value = value
        for distance in pl.unroll(3):
            bit = pl.reg.broadcast(pl.cast(4 >> distance, pl.UINT32), lanes=64, dtype=pl.UINT32)
            shuffle_index = pl.reg.bitwise_xor(lane, bit, full)
            shuffled = pl.reg.gather_lanes(sum_value, shuffle_index)
            sum_value = pl.reg.add(sum_value, shuffled, full)
        value = pl.reg.div(value, sum_value, full)
        value = pl.reg.muls(value, ROUTE_SCALE, full)
        _weights_written = pl.reg.store(weight_values, 0, value, full)
    index_storage = pl.reshape(index_values, [1, 64])
    weight_storage = pl.reshape(weight_values, [1, 64])
    index_prefix = pl.tile.slice(index_storage, [1, GATE_T_TILE * TOPK_PAD], [0, 0])
    weight_prefix = pl.tile.slice(weight_storage, [1, GATE_T_TILE * TOPK_PAD], [0, 0])
    index_rows = pl.reshape(index_prefix, [GATE_T_TILE, TOPK_PAD])
    weight_rows = pl.reshape(weight_prefix, [GATE_T_TILE, TOPK_PAD])
    pl.store(pl.set_validshape(index_rows, valid_rows, topk), [row, 0], indices)
    pl.store(pl.set_validshape(weight_rows, valid_rows, topk), [row, 0], weights)
    return indices, weights


@pl.jit.inline(auto_scope=False)
def _zero_route_tail(
    indices: pl.Tensor,
    weights: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
    topk: pl.constexpr,
):
    capacity = pl.tensor.dim(indices, 0)
    if pl.tile.get_block_idx() == 0:
        for start in pl.range(active_tokens, capacity, 32):
            rows = pl.min(32, capacity - start)
            zero = pl.tile.full([32, TOPK_PAD], dtype=pl.INT32, value=0)
            pl.store(pl.set_validshape(zero, rows, topk), [start, 0], indices)
            pl.store(pl.set_validshape(pl.reinterpret_view(zero, pl.FP32), rows, topk), [start, 0], weights)
    return indices, weights


@pl.jit.incore
def _gate_select_routes_block(
    raw_scores: pl.Tensor,
    gate_bias: pl.Tensor,
    indices: pl.Tensor,
    weights: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
    topk: pl.constexpr,
    columns: pl.constexpr,
    experts: pl.constexpr,
):
    """Select routes for the rows assigned to this AIV block."""
    worker = pl.tile.get_block_idx()
    blocks = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    for block in pl.range(worker, blocks, pl.tile.get_block_num()):
        row = block * GATE_T_TILE
        valid_rows = pl.min(GATE_T_TILE, active_tokens - row)
        raw_tile = pl.load(raw_scores, [row, 0], [GATE_T_TILE, columns], valid_shape=[GATE_T_TILE, experts], target_memory=pl.Mem.Vec)
        raw = pl.set_validshape(pl.tile.fillpad(raw_tile), GATE_T_TILE, columns)
        prefixes, scores = _gate_rank_rows(columns, raw, gate_bias, row, 0, experts)
        _gate_store_routes(columns, prefixes, scores, indices, weights, row, valid_rows, topk)
    _zero_route_tail(indices, weights, active_tokens, topk)
    return indices, weights


@pl.jit.inline(auto_scope=False)
def _gate_select_routes(
    raw_scores: pl.Tensor,
    gate_bias: pl.Tensor,
    indices: pl.Tensor,
    weights: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
    topk: pl.constexpr,
    columns: pl.constexpr,
    experts: pl.constexpr,
):
    """Apply sqrt-softplus and bias, select routes, and clear the inactive output rows."""
    blocks = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    if T_PAD == 384 and columns == 32:
        workers = pl.max(1, pl.min(blocks, pl.min(40, pl.system.available_aiv_count())))
    elif columns == 128:
        workers = pl.max(1, pl.min(blocks, 64))
    else:
        workers = pl.max(1, pl.min(blocks, ROUTE_WORKER_TILE))
    with pl.spmd(workers, name_hint="gate_select"):
        indices, weights = _gate_select_routes_block(
            raw_scores, gate_bias, indices, weights, active_tokens, topk, columns, experts,
        )
    return indices, weights


@pl.jit.incore
def _gate_quant_vf(
    x: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Quantize normalized input into group-32 FP8 and MX_A_ZZ scale bytes."""
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    chunks = width // QUANT_TILE
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    iterations = ((rows // GATE_M_TILE) * chunks + workers - 1 - worker) // workers
    for iteration in pl.range(iterations):
        block = worker + iteration * workers
        row = block // chunks * GATE_M_TILE
        column = block % chunks * QUANT_TILE
        source = pl.load(x, [row, column], [GATE_M_TILE, QUANT_TILE], target_memory=pl.Mem.Vec)
        source = pl.reshape(source, [GATE_M_TILE * QUANT_TILE])
        quantized = pl.create_tile([GATE_M_TILE * QUANT_TILE], dtype=pl.INT8, target_memory=pl.Mem.Vec)
        scale_pairs = pl.create_tile([128], dtype=pl.UINT16, target_memory=pl.Mem.Vec)
        with pl.reg_scope(auto_sync=False):
            full = pl.reg.full_mask(lanes=64)
            lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
            first_group = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
            second_group = pl.reg.broadcast(32, lanes=64, dtype=pl.UINT32)
            one_pair = pl.reg.active_mask(1, lanes=128)
            floor = pl.reg.broadcast(1e-4, lanes=64, dtype=pl.FP32)
            lower = pl.reg.broadcast(-448.0, lanes=64, dtype=pl.FP32)
            upper = pl.reg.broadcast(448.0, lanes=64, dtype=pl.FP32)
            bit16 = pl.reg.broadcast(16, lanes=64, dtype=pl.UINT32)
            bit8 = pl.reg.broadcast(8, lanes=64, dtype=pl.UINT32)
            bit4 = pl.reg.broadcast(4, lanes=64, dtype=pl.UINT32)
            bit2 = pl.reg.broadcast(2, lanes=64, dtype=pl.UINT32)
            bit1 = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
            index16 = pl.reg.bitwise_xor(lane, bit16, full)
            index8 = pl.reg.bitwise_xor(lane, bit8, full)
            index4 = pl.reg.bitwise_xor(lane, bit4, full)
            index2 = pl.reg.bitwise_xor(lane, bit2, full)
            index1 = pl.reg.bitwise_xor(lane, bit1, full)
            for local in pl.hardware_loop(GATE_M_TILE):
                for segment in pl.hardware_loop(QUANT_TILE // 64):
                    offset = local * QUANT_TILE + segment * 64
                    raw = pl.reg.load(source, offset, lanes=64)
                    value = pl.reg.convert(raw, dtype=pl.FP32)
                    maximum = pl.reg.abs(value, full)
                    shuffled = pl.reg.gather_lanes(maximum, index16)
                    maximum = pl.reg.maximum(maximum, shuffled, full)
                    shuffled = pl.reg.gather_lanes(maximum, index8)
                    maximum = pl.reg.maximum(maximum, shuffled, full)
                    shuffled = pl.reg.gather_lanes(maximum, index4)
                    maximum = pl.reg.maximum(maximum, shuffled, full)
                    shuffled = pl.reg.gather_lanes(maximum, index2)
                    maximum = pl.reg.maximum(maximum, shuffled, full)
                    shuffled = pl.reg.gather_lanes(maximum, index1)
                    maximum = pl.reg.maximum(maximum, shuffled, full)
                    maximum = pl.reg.maximum(maximum, floor, full)
                    maximum = pl.reg.muls(maximum, 1.0 / 448.0, full)
                    bits = pl.reg.interpret_cast(maximum, dtype=pl.INT32)
                    exponent = pl.reg.adds(bits, 8388607, full)
                    exponent = pl.reg.shift_rights(exponent, 23, full)
                    divisor_bits = pl.reg.shift_lefts(exponent, 23, full)
                    divisor = pl.reg.interpret_cast(divisor_bits, dtype=pl.FP32)
                    value = pl.reg.div(value, divisor, full)
                    value = pl.reg.maximum(value, lower, full)
                    value = pl.reg.minimum(value, upper, full)
                    value = pl.reg.convert(value, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                    payload_bytes = pl.reg.interpret_cast(value, dtype=pl.INT8)
                    _payload_written = pl.reg.store(quantized, offset, payload_bytes, full)
                    even_code = pl.reg.gather_lanes(exponent, first_group)
                    odd_code = pl.reg.gather_lanes(exponent, second_group)
                    odd_code = pl.reg.shift_lefts(odd_code, 8, full)
                    pair = pl.reg.bitwise_or(even_code, odd_code, full)
                    pair_position = segment * GATE_M_TILE + local
                    pair_halves = pl.reg.interpret_cast(pair, dtype=pl.UINT16)
                    scalar_pair = pl.reg.reduce_max(pair_halves, one_pair)
                    _scale_written = pl.reg.store(scale_pairs, pair_position, scalar_pair)
            # Complete the vector loads and stores at the VF boundary.
            pl.reg.mem_bar("VV_ALL")
        quantized = pl.reshape(quantized, [GATE_M_TILE, QUANT_TILE])
        quantized = pl.reinterpret_view(quantized, pl.FP8E4M3FN)
        pl.store(quantized, [row, column], payload)
        scale_pairs = pl.reshape(scale_pairs, [1, 128])
        scale_prefix = pl.tile.slice(scale_pairs, [1, GATE_M_TILE * (QUANT_TILE // 64)], [0, 0])
        scale_prefix = pl.reinterpret_view(scale_prefix, pl.UINT8)
        packed = pl.reinterpret_view(scale_prefix, pl.FP8E8M0)
        scale_column = row * (width // MX_GROUP) + (column // MX_GROUP) * GATE_M_TILE
        pl.store(packed, [0, scale_column], scales)
        pl.system.barrier(pipe=pl.PipeType.ALL)
    return payload, scales




def golden_gate_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sqrt-softplus routing, with bias affecting selection only."""
    scores = F.softplus(F.linear(x.float(), weight.to(torch.bfloat16).float()) / M.gate_temperature).sqrt()
    bias = correction_bias.float().reshape(-1, weight.shape[0])[:x.shape[0]]
    indices = (scores + bias).topk(M.num_experts_per_tok, dim=-1).indices
    weights = scores.gather(-1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * M.routed_scaling_factor, indices.to(torch.int32)


def prepare_routing_bias(text_bias, vision_bias, image_mask, capacity):
    """Flatten per-layer token biases in encoder input order, padding with text bias."""
    if text_bias.ndim not in (1, 2) or text_bias.shape != vision_bias.shape:
        raise ValueError("text and vision biases must have matching [experts] or [layers, experts] shapes")
    if text_bias.dtype != torch.float32 or vision_bias.dtype != torch.float32:
        raise ValueError("routing biases must have FP32 dtype")
    if text_bias.device.type != "cpu" or vision_bias.device.type != "cpu":
        raise ValueError("routing bias preparation requires CPU weights")
    mask = torch.as_tensor(image_mask, device="cpu")
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("image_mask must be a one-dimensional boolean sequence")
    if type(capacity) is not int or capacity < max(1, mask.numel()):
        raise ValueError("capacity must be a positive integer covering the image mask")
    padded_mask = torch.zeros(capacity, dtype=torch.bool)
    padded_mask[:mask.numel()] = mask
    text = text_bias.reshape(-1, 1, text_bias.shape[-1])
    vision = vision_bias.reshape_as(text)
    return torch.where(padded_mask[None, :, None], vision, text).flatten().contiguous()


@pl.jit.inline
def quantize_expert_input(
    xg_buf: pl.Tensor[[T_PAD, D], pl.BF16],
    x_norm_mx: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_norm_scale: pl.Tensor[[1, T_PAD * MX_SCALE_GROUPS], pl.FP8E8M0],
):
    with pl.spmd(pl.min((T_PAD // GATE_M_TILE) * (D // QUANT_TASK_TILE), QUANT_WORKER_TILE), name_hint="x_norm_mx_quant"):
        x_norm_mx, x_norm_scale = _gate_quant_vf(xg_buf, x_norm_mx, x_norm_scale)
    return x_norm_mx, x_norm_scale


@pl.jit.inline(auto_scope=False)
def _gate_project_tile(
    x: pl.Tensor,
    gate_w: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    columns: pl.constexpr,
    k_tile: pl.constexpr,
):
    """Project one row group onto a contiguous expert panel."""
    width = pl.tensor.dim(x, 1)
    valid = pl.min(columns, pl.tensor.dim(gate_w, 0) - column)
    acc = pl.create_tile([GATE_M_TILE, columns], dtype=pl.FP32, target_memory=pl.Mem.Acc)
    for k0 in pl.pipeline(0, width, k_tile, stage=2):
        left = pl.load(x, [row, k0], [GATE_M_TILE, k_tile], target_memory=pl.Mem.Mat)
        right = pl.load(gate_w, [column, k0], [columns, k_tile], valid_shape=[valid, k_tile], target_memory=pl.Mem.Mat)
        right = pl.tile.transpose_view(right)
        acc = pl.tile.matmul_acc(acc, left, right, init_cond=k0 == 0)
    return acc


@pl.jit.inline(auto_scope=False)
def _gate_project(xg_buf: pl.Tensor, gate_w: pl.Tensor, active_tokens: pl.Scalar[pl.INDEX],
                  n_experts: pl.constexpr, n_tile: pl.constexpr):
    """Project the active token rows onto the router experts."""
    row_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    column_tiles = (n_experts + n_tile - 1) // n_tile
    logits = pl.create_tensor([T_PAD, column_tiles * n_tile], dtype=pl.FP32)
    tasks = row_tiles * column_tiles
    workers = pl.max(1, pl.min(tasks, pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint="gate_project"):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, tasks, pl.tile.get_block_num()):
            row = block // column_tiles * GATE_M_TILE
            col = block % column_tiles * n_tile
            acc = _gate_project_tile(xg_buf, gate_w, row, col, n_tile, 1024)
            pl.store(acc, [row, col], logits)
    return logits


@pl.jit.inline(auto_scope=False)
def _gate_project_and_select_routes(
    xg_buf: pl.Tensor,
    gate_w: pl.Tensor,
    bias: pl.Tensor,
    indices: pl.Tensor,
    weights: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
    topk: pl.constexpr,
    n_experts: pl.constexpr,
):
    """Project 128 experts per panel and merge three panel prefixes inside each row worker."""
    row_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    clusters = pl.system.available_cluster_count()
    workers = pl.max(1, pl.min(row_tiles, clusters))
    with pl.spmd(workers, name_hint="gate_project_select"):
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, row_tiles, pl.tile.get_block_num()):
            row = block * GATE_M_TILE
            prefixes = pl.create_tile([GATE_M_TILE, SORT_PAD * 3], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            scores = pl.create_tile([GATE_M_TILE, 384], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            for panel in pl.range(3):
                col = panel * 128
                acc = _gate_project_tile(xg_buf, gate_w, row, col, 128, 256)
                raw = pl.tile.move(acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
                raw = pl.tile.move(raw, target_memory=pl.Mem.Vec)
                raw = pl.set_validshape(pl.tile.fillpad(raw), GATE_M_TILE, 128)
                for group in pl.unroll(GATE_M_TILE // GATE_T_TILE):
                    offset = group * GATE_T_TILE
                    raw_rows = pl.tile.slice(raw, [GATE_T_TILE, 128], [offset, 0])
                    local_pairs, local_scores = _gate_rank_rows(128, raw_rows, bias, row + offset, col, n_experts)
                    scores = pl.tile.assemble(scores, local_scores, [offset, col])
                    prefixes = pl.tile.assemble(prefixes, local_pairs, [offset, panel * SORT_PAD])
            for group in pl.unroll(GATE_M_TILE // GATE_T_TILE):
                offset = group * GATE_T_TILE
                if row + offset < active_tokens:
                    rows = pl.min(GATE_T_TILE, active_tokens - row - offset)
                    row_scores = pl.tile.slice(scores, [GATE_T_TILE, 384], [offset, 0])
                    row_scores = pl.tile.move(row_scores, target_memory=pl.Mem.Vec)
                    row_pairs = pl.tile.slice(prefixes, [GATE_T_TILE, SORT_PAD * 3], [offset, 0])
                    final_pairs = _gate_prefixes(SORT_PAD * 3, row_pairs, SORT_PAD, 3)
                    _gate_store_routes(384, final_pairs, row_scores, indices, weights, row + offset, rows, topk)
        _zero_route_tail(indices, weights, active_tokens, topk)
    return weights


@pl.jit.inline
def route_tokens(
    xg_buf: pl.Tensor[[T_PAD, D], pl.BF16],
    gate_w: pl.Tensor,
    gate_bias: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    indices: pl.Tensor,
    weights: pl.Tensor,
    n_experts: pl.constexpr = N_EXPERTS,
    topk: pl.constexpr = TOPK,
):
    """Project BF16 rows and select routes from biased scores."""
    active_tokens = pl.max(0, pl.min(pl.cast(num_tokens, pl.INDEX), T))
    bias = pl.reshape(gate_bias, [pl.tensor.dim(gate_bias, 0) // n_experts, n_experts])
    if n_experts > 256 and T_PAD >= 256:
        _gate_project_and_select_routes(xg_buf, gate_w, bias, indices, weights, active_tokens, topk, n_experts)
    else:
        # Use 16 expert columns when all projection tiles fit in one A5 wave.
        logits = _gate_project(
            xg_buf, gate_w, active_tokens, n_experts,
            16 if (T_PAD // GATE_M_TILE) * ((n_experts + 15) // 16) <= 32 else GATE_N_TILE,
        )
        indices, weights = _gate_select_routes(
            logits, bias, indices, weights, active_tokens, topk,
            32 if n_experts <= 32 else ((n_experts + 127) // 128) * 128, n_experts,
        )
    return weights


@pl.jit.inline
def gate(
    xg_buf: pl.Tensor[[T_PAD, D], pl.BF16],
    gate_w: pl.Tensor,
    gate_bias: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    x_norm_mx: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_norm_scale: pl.Tensor[[1, T_PAD * MX_SCALE_GROUPS], pl.FP8E8M0],
    indices: pl.Tensor,
    weights: pl.Tensor,
    n_experts: pl.constexpr = N_EXPERTS,
    topk: pl.constexpr = TOPK,
):
    quantize_expert_input(xg_buf, x_norm_mx, x_norm_scale)
    return route_tokens(xg_buf, gate_w, gate_bias, num_tokens, indices, weights, n_experts=n_experts, topk=topk)


def _gate_validation_entry(experts):
    """Specialize the standalone validator for a model router width."""
    @pl.jit
    def l2_gate(
        x_normed: pl.Tensor[[T_PAD, D], pl.BF16],
        gate_w: pl.Tensor[[experts, D], pl.BF16],
        gate_bias: pl.Tensor[[BIAS_DYN], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
        x_norm_mx: pl.Out[pl.Tensor[[T_PAD, D], pl.FP8E4M3FN]],
        x_norm_scale: pl.Out[pl.Tensor[[1, T_PAD * MX_SCALE_GROUPS], pl.FP8E8M0]],
        indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
        weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
    ):
        gate_bias.bind_dynamic(0, BIAS_DYN)
        gate(
            x_normed, gate_w, gate_bias,
            num_tokens,
            x_norm_mx, x_norm_scale, indices, weights, n_experts=experts, topk=TOPK,
        )
        return x_norm_mx, x_norm_scale, indices, weights

    return l2_gate


l2_gate = _gate_validation_entry(N_EXPERTS)


def _golden_gate_scores(tensors):
    """Recompute the host router scores from the immutable gate inputs."""
    import torch

    x_norm = tensors["x_normed"].cpu().to(torch.bfloat16).float()[:T].view(T, D)

    gate_w = tensors["gate_w"].cpu().to(torch.bfloat16).float()
    gate_bias = tensors["gate_bias"].cpu().float()
    logits = x_norm @ gate_w.T
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.reshape(-1, gate_w.shape[0])[:T]
    return x_norm, scores, biased


def _golden_gate_core(tensors, x_norm, scores, biased):
    import torch

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    from quant import quantize_mxfp8_upward_e8m0, pack_mx_a_scale

    def host_mxfp8_activation(value):
        # The released activation quantizer rounds E8M0 scales upward.
        quantized, codes = quantize_mxfp8_upward_e8m0(value)
        packed = pack_mx_a_scale(codes)
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is not None:
            packed = packed.contiguous().view(e8m0)
        return quantized, packed

    x_norm[num_tokens:] = 0
    x_norm_padded = torch.zeros(T_PAD, D, dtype=torch.float32)
    x_norm_padded[:T] = x_norm
    x_norm_mx, x_norm_scale = host_mxfp8_activation(x_norm_padded)

    indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]

    topk_vals = torch.gather(scores, dim=-1, index=indices.long())
    denom = topk_vals.sum(dim=-1, keepdim=True)
    weights = (topk_vals / denom) * ROUTE_SCALE
    if num_tokens < T:
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0

    tensors["x_norm_mx"][:] = x_norm_mx
    tensors["x_norm_scale"][:] = x_norm_scale.reshape(1, -1)
    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights.to(torch.float32)


def golden_gate_normalized_core(tensors):
    """Reference routing from normalized BF16 input with a zero inactive tail."""
    x_norm, scores, biased = _golden_gate_scores(tensors)
    _golden_gate_core(tensors, x_norm, scores, biased)


def build_tensor_specs(num_tokens=T, *, token_bias=False, experts=N_EXPERTS):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_normed():
        from rmsnorm import golden_rms_norm

        value = golden_rms_norm(torch.randn(T_PAD, D).to(torch.bfloat16), torch.ones(D))
        value[max(0, min(T, num_tokens)):] = 0
        return value

    def init_gate_w():
        return torch.randn(experts, D) / D ** 0.5

    def init_gate_bias():
        if not token_bias:
            return torch.randn(experts) * 0.1
        bias = torch.full((T_PAD, experts), -10.0)
        for row in range(T_PAD):
            selected = (torch.arange(TOPK) + row * TOPK) % experts
            bias[row, selected] = 10.0
        return bias.flatten()

    # PyTorch CPU has incomplete float8 support for zeros/fill; initialise through
    # a uint8 carrier and view it as FP8 so the harness does not fail before NPU
    # compilation.
    def init_fp8(shape):
        return torch.zeros(shape, dtype=torch.uint8).view(torch.float8_e4m3fn)

    def init_e8m0(shape):
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is None:
            return torch.zeros(shape, dtype=torch.uint8)
        return torch.zeros(shape, dtype=torch.uint8).view(e8m0)

    return [
        TensorSpec("x_normed", [T_PAD, D], torch.bfloat16, init_value=init_x_normed),
        TensorSpec("gate_w", [experts, D], torch.bfloat16, init_value=init_gate_w),
        TensorSpec("gate_bias", [T_PAD * experts if token_bias else experts], torch.float32, init_value=init_gate_bias),
        ScalarSpec("num_tokens", torch.int32, num_tokens, compile_runtime=True),
        TensorSpec("x_norm_mx", [T_PAD, D], torch.float8_e4m3fn, init_value=lambda: init_fp8((T_PAD, D))),
        TensorSpec("x_norm_scale", [1, T_PAD * MX_SCALE_GROUPS], getattr(torch, "float8_e8m0fnu", torch.uint8), init_value=lambda: init_e8m0((1, T_PAD * MX_SCALE_GROUPS))),
        TensorSpec("indices", [T, TOPK], torch.int32),
        TensorSpec("weights", [T, TOPK], torch.float32),
    ]


def gate_indices_compare(
    num_tokens,
    *,
    score_atol=1e-4,
    score_rtol=2e-5,
    max_show=10,
):
    """Validate router ids against bounded score routes.

    ``topk_pair_compare`` cannot stand in for this: it adjudicates an index
    swap by checking whether the paired *weights* field stays locally
    monotonic, but selection is driven by the biased score
    (``scores + gate_bias``), not by the unbiased ``weights`` the device
    reports. Only the host-recomputed biased score tells us whether a swapped
    pair is a genuine tie at the top-k cutoff.
    """
    import torch

    active_tokens = max(0, min(T, int(num_tokens)))

    def cmp(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, rtol, atol
        actual = actual.cpu()
        expected = expected.cpu()
        if actual.shape != expected.shape:
            return False, (
                f"    index shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )

        inactive = actual[active_tokens:]
        inactive_nonzero = int(inactive.count_nonzero().item())
        if inactive_nonzero:
            return False, (
                f"    inactive index tail contains {inactive_nonzero} nonzero values"
            )

        actual = actual[:active_tokens].to(torch.int64)
        expected = expected[:active_tokens].to(torch.int64)
        if actual.numel() == 0:
            return True, ""

        experts = inputs["gate_w"].shape[0]
        invalid = (actual < 0) | (actual >= experts)
        if invalid.any().item():
            bad = invalid.nonzero(as_tuple=False)
            lines = [
                f"    score route contains {bad.shape[0]} out-of-range id(s); "
                f"valid range is [0, {experts})"
            ]
            for row, pos in bad[:max_show].tolist():
                lines.append(f"      [{row},{pos}] actual={int(actual[row, pos])}")
            return False, "\n".join(lines)

        sorted_ids = torch.sort(actual, dim=-1).values
        duplicate = sorted_ids[:, 1:] == sorted_ids[:, :-1]
        if duplicate.any().item():
            rows = duplicate.any(dim=-1).nonzero(as_tuple=False).flatten()
            lines = [f"    score route contains duplicate ids in {rows.numel()} row(s)"]
            for row in rows[:max_show].tolist():
                lines.append(f"      row {row}: ids={actual[row].tolist()}")
            return False, "\n".join(lines)

        _, scores, biased = _golden_gate_scores(inputs)
        scores = scores[:active_tokens]
        biased = biased[:active_tokens]
        if not torch.isfinite(biased).all().item():
            return False, "    CPU router reference contains NaN or Inf"

        score_error = score_atol + score_rtol * scores.abs()
        actual_biased = torch.gather(biased, dim=-1, index=actual)
        actual_error = torch.gather(score_error, dim=-1, index=actual)
        selected_mask = torch.zeros_like(biased, dtype=torch.bool)
        selected_mask.scatter_(dim=-1, index=actual, value=True)
        omitted_lower = (biased - score_error).masked_fill(
            selected_mask,
            float("-inf"),
        )
        best_omitted_lower, best_omitted_id = omitted_lower.max(dim=-1, keepdim=True)
        selected_upper = actual_biased + actual_error
        selected_floor_upper, selected_floor_pos = selected_upper.min(
            dim=-1,
            keepdim=True,
        )
        omitted_better = best_omitted_lower > selected_floor_upper

        # Top-k denotes an expert set, not an order: device sort may differ from
        # CPU stable sort on near-equal scores, but combine pairs (expert_id,
        # weight), so an identical set must not be reported as a mismatch.
        actual_set = torch.sort(actual, dim=-1).values
        expected_set = torch.sort(expected, dim=-1).values
        set_mismatch = actual_set != expected_set
        if not omitted_better.any().item() and not set_mismatch.any().item():
            return True, ""

        lines = [
            "    score route exceeds the calibrated FP32 top-k ambiguity band "
            f"(score_atol={score_atol:g}, score_rtol={score_rtol:g})"
        ]
        bad_rows = omitted_better.nonzero(as_tuple=False)[:max_show, 0].tolist()
        set_rows = set_mismatch.any(dim=-1).nonzero(as_tuple=False)[:max_show, 0].tolist()
        diagnostic_rows = sorted(set(bad_rows + set_rows))
        cpu_topk = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
        for row in diagnostic_rows[:max_show]:
            actual_ids = actual[row].tolist()
            cpu_ids = cpu_topk[row].tolist()
            actual_scores = [float(biased[row, i]) for i in actual_ids]
            cpu_scores = [float(biased[row, i]) for i in cpu_ids]
            lines.append(
                f"      diagnostic row {row}: device_ids={actual_ids} "
                f"device_scores={[round(v, 7) for v in actual_scores]} "
                f"cpu_ids={cpu_ids} cpu_scores={[round(v, 7) for v in cpu_scores]}"
            )
        for row in bad_rows:
            omitted_id = int(best_omitted_id[row, 0])
            selected_pos = int(selected_floor_pos[row, 0])
            selected_id = int(actual[row, selected_pos])
            regret = float(biased[row, omitted_id] - biased[row, selected_id])
            budget = float(score_error[row, omitted_id] + actual_error[row, selected_pos])
            lines.append(
                f"      row {row}: omitted id={omitted_id} beats selected "
                f"id={selected_id}; raw_regret={regret:.8g} budget={budget:.8g}"
            )
        shown = len(bad_rows)
        remaining = max_show - shown
        if remaining:
            for row in set_rows[:remaining]:
                lines.append(
                    f"      row {row} selected expert set differs: "
                    f"device={actual[row].tolist()} cpu={expected[row].tolist()}"
                )
        return False, "\n".join(lines)

    cmp.__name__ = (
        f"gate_indices_compare(score_atol={score_atol},score_rtol={score_rtol})"
    )
    return cmp


def mxfp8_activation_value_compare(actual, expected, *, actual_outputs, expected_outputs, **kwargs):
    """Compare MXFP8 activations by dequantised value, not by non-unique bytes.

    quant_mx may scale the payload and the E8M0 scale by the same power of two
    while keeping the dequantised value identical, so the V4.1 gate cannot reuse
    the V4 dual-bitwise comparison.
    """
    import torch

    from quant import decode_e8m0, unpack_mx_a_scale

    def decode(outputs, payload_name):
        payload = outputs[payload_name].cpu().float()
        scale_bytes = outputs["x_norm_scale"].cpu().contiguous().view(torch.uint8)
        scale = unpack_mx_a_scale(scale_bytes.reshape(T_PAD, MX_SCALE_GROUPS))
        scale = decode_e8m0(scale).repeat_interleave(MX_GROUP, dim=-1)
        return payload * scale

    actual_value = decode(actual_outputs, "x_norm_mx")
    expected_value = decode(expected_outputs, "x_norm_mx")
    from golden import ratio_allclose

    check = ratio_allclose(atol=2e-3, rtol=2e-3, max_error_ratio=0.0, max_show=8)
    return check(
        actual_value, expected_value, actual_outputs=actual_outputs, expected_outputs=expected_outputs, **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import torch
    from golden.runner import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-bias", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()
    if args.execution_mode:
        parser.error("Gate RegScope kernels require A5 device execution; simulation is unsupported")
    if args.num_tokens is None and (args.runtime_dir or args.save_data or args.golden_data):
        parser.error("--num-tokens is required when saving or replaying a case")
    counts = [args.num_tokens] if args.num_tokens is not None else [0, 1, T - 1, T]
    if args.compile_only:
        counts = [T if args.num_tokens is None else args.num_tokens]
    cases = [(n, N_EXPERTS, args.token_bias) for n in counts]
    if args.num_tokens is None and not args.compile_only and (N_EXPERTS != 96 or not args.token_bias):
        cases.append((min(72, T), 96, True))
    for active_count, experts, token_bias in cases:
        torch.manual_seed(args.seed)
        print(f"Gate validation: num_tokens={active_count}, experts={experts}", flush=True)

        result = run(
            a5_default_runtime="host_build_graph",
            fn=l2_gate if experts == N_EXPERTS else _gate_validation_entry(experts),
            specs=build_tensor_specs(num_tokens=active_count, token_bias=token_bias, experts=experts),
            golden_fn=golden_gate_normalized_core,
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            golden_data=args.golden_data,
            config=dict(
                dump_passes=args.dump_passes,
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                # payload/scale may use a different but equivalent E8M0 representation, so
                # compare dequantised values.
                "x_norm_mx": mxfp8_activation_value_compare,
                # The scale is already covered by the combined payload/scale comparator and
                # must not be compared bitwise on its own.
                "x_norm_scale": mxfp8_activation_value_compare,
                "indices": gate_indices_compare(active_count),
            },
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
