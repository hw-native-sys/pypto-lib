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

from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import torch
import torch.nn.functional as F

from config import FLASH as M

from config import MOE_TOKENS, EP_SIZE, N_LOCAL_EXPERTS


# model config
T = MOE_TOKENS
D = M.hidden_size
BIAS_DYN = pl.dynamic("GATE_BIAS_DYN")
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor

# tiling
GATE_T_TILE = 4
GATE_M_TILE = 16
GATE_N_TILE = 16
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


@pl.jit.incore
def _gate_scores_vf(
    x: pl.Tensor,
    gate_w: pl.Tensor,
    gate_bias: pl.Tensor,
    route_scores: pl.Tensor,
    biased_scores: pl.Tensor,
    tasks: pl.Scalar[pl.INDEX],
    per_token_bias: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Project router inputs and compute sqrt-softplus scores with selection bias."""
    experts = pl.tensor.dim(gate_w, 0)
    width = pl.tensor.dim(x, 1)
    columns = (experts + GATE_N_TILE - 1) // GATE_N_TILE
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    for work in pl.range(worker, tasks, workers):
        row = work // columns * GATE_M_TILE
        column = work % columns * GATE_N_TILE
        valid = pl.min(GATE_N_TILE, experts - column)
        acc = pl.create_tile([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        for k0 in pl.pipeline(0, width, 1024, stage=2):
            left = pl.load(x, [row, k0], [GATE_M_TILE, 1024], target_memory=pl.Mem.Mat)
            right = pl.load(gate_w, [column, k0], [GATE_N_TILE, 1024], valid_shape=[valid, 1024], target_memory=pl.Mem.Mat)
            right = pl.tile.transpose_view(right)
            acc = pl.tile.matmul_acc(acc, left, right, init_cond=k0 == 0)
        raw = pl.tile.move(acc, target_memory=pl.Mem.Vec, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box)
        # Materialize local UB before the RegScope capture.
        raw = pl.tile.move(raw, target_memory=pl.Mem.Vec)
        raw = pl.reshape(raw, [GATE_M_TILE * GATE_N_TILE])
        bias_storage = pl.create_tile([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        if not per_token_bias:
            broadcast_bias = pl.load(gate_bias, [0, column], [1, GATE_N_TILE], valid_shape=[1, valid], target_memory=pl.Mem.Vec)
            broadcast_bias = pl.set_validshape(pl.tile.fillpad(broadcast_bias, pad_value=pl.PadValue.min), 1, GATE_N_TILE)
            expanded_bias = pl.tile.col_expand(bias_storage, broadcast_bias)
            bias = pl.tile.fillpad(expanded_bias, pad_value=pl.PadValue.min)
        else:
            bias_rows = pl.tensor.dim(gate_bias, 0)
            valid_bias_rows = pl.max(0, pl.min(GATE_M_TILE, bias_rows - row))
            token_bias = pl.load(gate_bias, [row, column], [GATE_M_TILE, GATE_N_TILE],
                                 valid_shape=[valid_bias_rows, valid], target_memory=pl.Mem.Vec)
            bias = pl.set_validshape(pl.tile.fillpad(token_bias, pad_value=pl.PadValue.min), GATE_M_TILE, GATE_N_TILE)
        bias_flat = pl.reshape(bias, [GATE_M_TILE * GATE_N_TILE])
        scores = pl.create_tile([GATE_M_TILE * GATE_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        biased = pl.create_tile([GATE_M_TILE * GATE_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            full = pl.reg.full_mask(lanes=64)
            lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
            low_bits = pl.reg.broadcast(GATE_N_TILE - 1, lanes=64, dtype=pl.UINT32)
            bias_column = pl.reg.bitwise_and(lane, low_bits, full)
            valid_columns = pl.reg.compares(bias_column, pl.cast(valid, pl.UINT32), full, predicate="lt")
            minimum = pl.reg.broadcast(-3.4028234663852886e38, lanes=64, dtype=pl.FP32)
            zero = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
            for segment in pl.hardware_loop(GATE_M_TILE * GATE_N_TILE // 64):
                bias_value = pl.reg.load(bias_flat, segment * 64, lanes=64)
                bias_value = pl.reg.select(valid_columns, bias_value, minimum)
                logits = pl.reg.load(raw, segment * 64, lanes=64)
                logits = pl.reg.select(valid_columns, logits, zero)
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
        scores = pl.reshape(scores, [GATE_M_TILE, GATE_N_TILE])
        biased = pl.reshape(biased, [GATE_M_TILE, GATE_N_TILE])
        pl.store(scores, [row, column], route_scores)
        pl.store(biased, [row, column], biased_scores)
    return route_scores, biased_scores


@pl.jit.inline
def gather_routing_bias(
    source: pl.Tensor,
    row_ids: pl.Tensor,
    output: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID],
    n_experts: pl.constexpr,
):
    """Gather token routing biases or copy one broadcast expert vector."""
    source_rows = pl.tensor.dim(source, 0) // n_experts
    output_rows = pl.tensor.dim(output, 0) // n_experts
    source_matrix = pl.reshape(source, [source_rows, n_experts])
    output_matrix = pl.reshape(output, [output_rows, n_experts])
    for row in pl.spmd(output_rows, name_hint="gate_bias_gather", deps=[ready]):
        source_row = pl.cast(0, pl.INT32)
        if source_rows > 1:
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


@pl.jit.incore
def _gate_route_sort_vf(
    biased_scores: pl.Tensor,
    route_scores: pl.Tensor,
    indices: pl.Tensor,
    weights: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
    route_scale: pl.Scalar[pl.FP32],
    topk: pl.constexpr,
    columns: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Sort router scores, normalize selected weights, and store active rows."""
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    count = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    for block in pl.range(worker, count, workers):
        row = block * GATE_T_TILE
        valid_rows = pl.min(GATE_T_TILE, active_tokens - row)
        biased = pl.load(biased_scores, [row, 0], [GATE_T_TILE, columns], target_memory=pl.Mem.Vec)
        scores = pl.load(route_scores, [row, 0], [GATE_T_TILE, columns], target_memory=pl.Mem.Vec)
        range_row = pl.tile.arange(0, [1, columns], dtype=pl.UINT32)
        range_tile = pl.create_tile([GATE_T_TILE, columns], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
        range_tile = pl.tile.col_expand(range_tile, range_row)
        sorted32 = pl.tile.sort32(biased, range_tile)
        if columns > 32:
            sorted_flat = pl.reshape(sorted32, [1, GATE_T_TILE * columns * 2])
            sorted64 = pl.tile.mrgsort(sorted_flat, block_len=64)
            sorted64 = pl.reshape(sorted64, [GATE_T_TILE, columns * 2])
        prefixes = pl.create_tile([GATE_T_TILE, SORT_PAD], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for local in pl.unroll(GATE_T_TILE):
            if columns <= 32:
                prefix = pl.tile.slice(sorted32, [1, SORT_PAD], [local, 0])
            elif columns <= 128:
                prefix = pl.tile.slice(sorted64, [1, SORT_PAD], [local, 0])
            else:
                first = pl.tile.slice(sorted64, [1, SORT_PAD], [local, 0])
                second = pl.tile.slice(sorted64, [1, SORT_PAD], [local, 256])
                merge_tmp = pl.create_tile([1, SORT_PAD * 3], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                if columns == 256:
                    merged = pl.tile.mrgsort(first, second, tmp=merge_tmp)
                else:
                    third = pl.tile.slice(sorted64, [1, SORT_PAD], [local, 512])
                    merged = pl.tile.mrgsort(first, second, third, tmp=merge_tmp)
                prefix = pl.tile.slice(merged, [1, SORT_PAD], [0, 0])
            prefixes = pl.tile.assemble(prefixes, prefix, [local, 0])
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
            value = pl.reg.muls(value, route_scale, full)
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


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "gate_quant_vf.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _gate_quant_vf(
    x: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Quantize normalized input into group-32 FP8 and MX_A_ZZ scale bytes."""
    return payload, scales


@pl.jit.incore
def _gate_pre_route_vf(
    indices: pl.Tensor,
    weights: pl.Tensor,
    biased_scores: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
    experts: pl.constexpr,
    topk: pl.constexpr,
    columns: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    """Zero inactive routes and initialize padded expert scores."""
    capacity = pl.tensor.dim(indices, 1) // topk
    active = pl.max(pl.min(active_tokens, capacity), 0)
    for start in pl.range(active * topk, capacity * topk, 512):
        count = pl.min(512, capacity * topk - start)
        zero_index = pl.tile.full([1, 512], dtype=pl.INT32, value=0)
        zero_weight = pl.reinterpret_view(zero_index, pl.FP32)
        pl.store(pl.set_validshape(zero_index, 1, count), [0, start], indices)
        pl.store(pl.set_validshape(zero_weight, 1, count), [0, start], weights)
    padded_rows = pl.tensor.dim(biased_scores, 0)
    if experts < columns:
        for row in pl.range(0, padded_rows, 80):
            rows = pl.min(80, padded_rows - row)
            padding = pl.create_tile([80, 384], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            padding_flat = pl.reshape(padding, [80 * 384])
            for segment in pl.unroll((columns - experts + 63) // 64):
                with pl.reg_scope():
                    fill = pl.reg.broadcast(-3.4028234663852886e38, lanes=64, dtype=pl.FP32)
                    valid = pl.min(64, columns - experts - segment * 64)
                    mask = pl.reg.active_mask(valid, lanes=64)
                    for local in pl.hardware_loop(rows):
                        _padding_written = pl.reg.store(padding_flat, local * 384 + segment * 64, fill, mask)
            padding = pl.reshape(padding_flat, [80, 384])
            padding = pl.set_validshape(padding, rows, columns - experts)
            pl.store(padding, [row, experts], biased_scores)
    return indices, weights, biased_scores


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
    """Route normalized BF16 rows with caller-zeroed inactive and padded rows."""
    if n_experts <= 32:
        score_pad = 32
    elif n_experts <= 256:
        score_pad = 256
    else:
        score_pad = 384
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    task_tokens = pl.max(active_tokens, pl.cast(1, pl.INDEX))
    task_gate_tiles = (task_tokens + GATE_M_TILE - 1) // GATE_M_TILE

    with pl.spmd(pl.min((T_PAD // GATE_M_TILE) * (D // QUANT_TASK_TILE), QUANT_WORKER_TILE), name_hint="x_norm_mx_quant"):
        x_norm_mx, x_norm_scale = _gate_quant_vf(xg_buf, x_norm_mx, x_norm_scale)

    biased_scores_buf = pl.create_tensor([T_PAD, score_pad], dtype=pl.FP32)
    index_flat = pl.reshape(indices, [1, T * topk])
    weight_flat = pl.reshape(weights, [1, T * topk])
    with pl.spmd(1, name_hint="gate_pre_route"):
        index_flat, weight_flat, biased_scores_buf = _gate_pre_route_vf(
            index_flat, weight_flat, biased_scores_buf, active_tokens, n_experts, topk,
            columns=32 if n_experts <= 32 else 256 if n_experts <= 256 else 384,
        )

    route_scores_buf = pl.create_tensor([T_PAD, score_pad], dtype=pl.FP32)
    gate_tasks = task_gate_tiles * ((n_experts + GATE_N_TILE - 1) // GATE_N_TILE)
    gate_bias_matrix = pl.reshape(gate_bias, [pl.tensor.dim(gate_bias, 0) // n_experts, n_experts])
    gate_worker_limit = pl.min(16, pl.system.available_cluster_count())
    gate_workers = pl.max(1, pl.min(gate_tasks, gate_worker_limit))
    if pl.tensor.dim(gate_bias, 0) > n_experts:
        with pl.spmd(gate_workers, name_hint="gate_token_bias", sync_start=True):
            route_scores_buf, biased_scores_buf = _gate_scores_vf(
                xg_buf, gate_w, gate_bias_matrix, route_scores_buf, biased_scores_buf, gate_tasks, True,
            )
    else:
        with pl.spmd(gate_workers, name_hint="gate", sync_start=True):
            route_scores_buf, biased_scores_buf = _gate_scores_vf(
                xg_buf, gate_w, gate_bias_matrix, route_scores_buf, biased_scores_buf, gate_tasks, False,
            )

    task_route_tiles = (task_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    route_workers = pl.max(pl.min(task_route_tiles, ROUTE_WORKER_TILE), 1)
    with pl.spmd(route_workers, name_hint="route_sort"):
        indices, weights = _gate_route_sort_vf(
            biased_scores_buf, route_scores_buf, indices, weights,
            active_tokens, ROUTE_SCALE, topk,
            columns=32 if n_experts <= 32 else 256 if n_experts <= 256 else 384,
        )

    return weights


@pl.jit
def l2_gate(
    x_normed: pl.Tensor[[T_PAD, D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.BF16],
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
        x_norm_mx, x_norm_scale, indices, weights, n_experts=N_EXPERTS, topk=TOPK,
    )
    return x_norm_mx, x_norm_scale, indices, weights


def _golden_gate_scores(tensors):
    """Recompute the host router scores from the immutable gate inputs."""
    import torch

    x_norm = tensors["x_normed"].cpu().to(torch.bfloat16).float()[:T].view(T, D)

    gate_w = tensors["gate_w"].cpu().to(torch.bfloat16).float()
    gate_bias = tensors["gate_bias"].cpu().float()
    logits = x_norm @ gate_w.T
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.reshape(-1, N_EXPERTS)[:T]
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


def build_tensor_specs(num_tokens=T, *, token_bias=False):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_normed():
        from rmsnorm import golden_rms_norm

        value = golden_rms_norm(torch.randn(T_PAD, D).to(torch.bfloat16), torch.ones(D))
        value[max(0, min(T, num_tokens)):] = 0
        return value

    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5

    def init_gate_bias():
        if not token_bias:
            return torch.randn(N_EXPERTS) * 0.1
        bias = torch.full((T_PAD, N_EXPERTS), -10.0)
        for row in range(T_PAD):
            selected = (torch.arange(TOPK) + row * TOPK) % N_EXPERTS
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
        TensorSpec("gate_w", [N_EXPERTS, D], torch.bfloat16, init_value=init_gate_w),
        TensorSpec("gate_bias", [T_PAD * N_EXPERTS if token_bias else N_EXPERTS], torch.float32, init_value=init_gate_bias),
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

        invalid = (actual < 0) | (actual >= N_EXPERTS)
        if invalid.any().item():
            bad = invalid.nonzero(as_tuple=False)
            lines = [
                f"    score route contains {bad.shape[0]} out-of-range id(s); "
                f"valid range is [0, {N_EXPERTS})"
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
    for active_count in counts:
        torch.manual_seed(args.seed)
        print(f"Gate validation: num_tokens={active_count}", flush=True)

        result = run(
            fn=l2_gate,
            specs=build_tensor_specs(num_tokens=active_count, token_bias=args.token_bias),
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
