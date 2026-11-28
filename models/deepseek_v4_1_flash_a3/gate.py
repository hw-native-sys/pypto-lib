# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 MoE router projection, scoring and normalized TopK."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import pypto.language as pl
from .gemm import quantize_activation
import torch
import torch.nn.functional as F

from .config import FLASH as M

from .config import EP_SIZE, N_LOCAL_EXPERTS
from .dsl_types import SLAB_T_DYN


FP32_NEG_INF = -3.4028234663852886e38


# model config
D = M.hidden_size
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor
VOCAB = M.vocab_size
N_HASH_LAYERS = 0

# tiling
GATE_INPUT_TILE = 4
# Dense Prefill slabs use one 64-row panel per gate worker.  The 64-row
# panel was the best A3 gate shape in the single-card sweep while retaining
# the 16-row capacity ABI for decode-sized slabs.
ROUTE_SELECT_T_TILE = 16
ROUTE_SELECT_WORKERS = 48
GATE_INPUT_WORKERS = 40
GATE_QUANT_WORKERS = 48
GATE_WORKERS = 48
# The 256-wide K tile halves the number of Cube launches for the 5120-wide gate.
GATE_D_TILE = 256
PREFILL_GATE_M_TILE = 128
PREFILL_GATE_N_TILE = 128
PREFILL_GATE_K_TILE = 128
PREFILL_GATE_PANEL_TILE = 1024
PREFILL_DENSE_MIN_ROWS = 320
PREFILL_COMPACT_ROWS = 512
PREFILL_SMALL_M_TILE = 64
PREFILL_SMALL_N_TILE = 32
PREFILL_COMPACT_N_TILE = 64
PREFILL_COMPACT_K_TILE = 256
PREFILL_COMPACT_SORT_TILE = 8
QUANT_TILE = 256
QUANT_TASK_TILE = QUANT_TILE * 4
MX_GROUP = 32
SCORE_PAD = 256 if M.name == "flash" else 384
TOPK_PAD = 8
SORT_PAD = TOPK_PAD * 2
QUANT_GROUPS = QUANT_TILE // 32


if M.name == "flash":
    @pl.jit.inline
    def route_topk_row(
        score_row: pl.Tensor[[1, 256], pl.FP32],
        idx_init: pl.Tensor[[1, 256], pl.UINT32],
    ) -> pl.Tensor[[1, TOPK_PAD], pl.INT32]:
        """Sort one Flash router row and return its leading expert ids."""

        sorted_32 = pl.sort32(score_row, idx_init)
        sorted_64 = pl.mrgsort(sorted_32, block_len=64)
        prefix = SORT_PAD
        sorted_pairs = pl.mrgsort(
            sorted_64[:, 0:prefix], sorted_64[:, 256 : 256 + prefix],
        )
        return pl.gather(
            sorted_pairs[:, 0:SORT_PAD],
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )
else:
    @pl.jit.inline
    def route_topk_row(
        score_row: pl.Tensor[[1, 384], pl.FP32],
        idx_init: pl.Tensor[[1, 384], pl.UINT32],
    ) -> pl.Tensor[[1, TOPK_PAD], pl.INT32]:
        """Sort one Pro router row and return its leading expert ids."""

        sorted_32 = pl.sort32(score_row, idx_init)
        sorted_64 = pl.mrgsort(sorted_32, block_len=64)
        prefix = SORT_PAD
        sorted_pairs = pl.mrgsort(
            sorted_64[:, 0:prefix],
            sorted_64[:, 256 : 256 + prefix],
            sorted_64[:, 512 : 512 + prefix],
        )
        return pl.gather(
            sorted_pairs[:, 0:SORT_PAD],
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )


def golden_gate_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    accumulation_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round the exact BF16 dot to FP32 before sqrt-softplus routing and biased selection."""
    source = x.to(accumulation_dtype)
    matrix = weight.to(accumulation_dtype)
    logits = F.linear(source, matrix).float()
    scores = F.softplus(logits / M.gate_temperature).sqrt()
    indices = torch.argsort(scores + correction_bias.float(), dim=-1, descending=True, stable=True)
    indices = indices[:, :M.num_experts_per_tok]
    weights = scores.gather(-1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * M.routed_scaling_factor, indices.to(torch.int32)


@pl.jit.inline
def gate_normalized(
    x_normed: pl.Tensor[[SLAB_T_DYN, D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    x_norm_mx: pl.Tensor[[SLAB_T_DYN, D], pl.INT8],
    x_norm_scale: pl.Tensor[[1, SLAB_T_DYN], pl.FP32],
    indices: pl.Tensor[[SLAB_T_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[SLAB_T_DYN, TOPK], pl.FP32],
    capacity: pl.constexpr,
):
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > capacity:
        active_tokens = pl.cast(capacity, pl.INDEX)

    quantized, quant_scale = quantize_activation(x_normed, num_tokens, D)
    scale_view = pl.reshape(x_norm_scale, [capacity, 1])
    quant_blocks = capacity // 16
    quant_workers = pl.min(pl.min(quant_blocks, GATE_QUANT_WORKERS), pl.system.available_aiv_count())
    for worker in pl.spmd(quant_workers, name_hint="gate_quant_copy"):
        for block in pl.range(worker, quant_blocks, quant_workers):
            row = block * 16
            codes = pl.load(quantized, [row, 0], [16, D])
            x_norm_mx = pl.store(codes, [row, 0], x_norm_mx)
            factor = pl.load(quant_scale, [row, 0], [16, 1])
            scale_view = pl.store(factor, [row, 0], scale_view)
    if capacity >= 64:
        m_tile = 64
    elif capacity >= 32:
        m_tile = 32
    else:
        m_tile = 16
    xg_buf = pl.create_tensor([capacity, D], dtype=pl.FP32)
    input_rows = pl.max(active_tokens, pl.cast(1, pl.INDEX))
    input_rows = (input_rows + m_tile - 1) // m_tile * m_tile
    input_blocks = input_rows // GATE_INPUT_TILE
    input_workers = pl.min(pl.min(input_blocks, GATE_INPUT_WORKERS), pl.system.available_aiv_count())
    for worker in pl.spmd(input_workers, name_hint="gate_normalized_input"):
        for block in pl.range(worker, input_blocks, input_workers):
            t0 = block * GATE_INPUT_TILE
            valid_rows = pl.min(GATE_INPUT_TILE, pl.max(active_tokens - t0, 0))
            normalized = pl.slice(x_normed, [GATE_INPUT_TILE, D], [t0, 0], valid_shape=[valid_rows, D])
            normalized = pl.fillpad(normalized, pad_value=pl.PadValue.zero)
            normalized = pl.set_validshape(normalized, GATE_INPUT_TILE, D)
            xg_buf[t0 : t0 + GATE_INPUT_TILE, :] = pl.cast(normalized, pl.FP32)

    return gate_from_fp32(
        xg_buf, gate_w, gate_bias, num_tokens, indices, weights, capacity, decode=True,
    )


@pl.jit.inline(auto_scope=False)
def gate_from_fp32(
    xg_buf: pl.Tensor,
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    indices: pl.Tensor,
    weights: pl.Tensor,
    capacity: pl.constexpr,
    decode: pl.constexpr,
):
    """Route independent owner panels, including a partial final panel."""
    if not decode and N_EXPERTS == SCORE_PAD and capacity > PREFILL_GATE_PANEL_TILE:
        full_rows = capacity // PREFILL_GATE_PANEL_TILE * PREFILL_GATE_PANEL_TILE
        for start in pl.parallel(0, full_rows, PREFILL_GATE_PANEL_TILE):
            with pl.scope():
                stop = start + PREFILL_GATE_PANEL_TILE
                rows = pl.min(PREFILL_GATE_PANEL_TILE, pl.max(num_tokens - start, 0))
                _route_gate_panel(
                    xg_buf[start:stop, :], gate_w, gate_bias, rows,
                    indices[start:stop, :], weights[start:stop, :], PREFILL_GATE_PANEL_TILE, decode,
                )
        if capacity % PREFILL_GATE_PANEL_TILE != 0:
            tail_rows = pl.max(num_tokens - full_rows, 0)
            _route_gate_panel(
                xg_buf[full_rows:capacity, :], gate_w, gate_bias, tail_rows,
                indices[full_rows:capacity, :], weights[full_rows:capacity, :],
                capacity % PREFILL_GATE_PANEL_TILE, decode,
            )
    else:
        _route_gate_panel(xg_buf, gate_w, gate_bias, num_tokens, indices, weights, capacity, decode)
    return weights


@pl.jit.inline
def _route_gate_panel(
    xg_buf: pl.Tensor,
    gate_w: pl.Tensor,
    gate_bias: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    indices: pl.Tensor,
    weights: pl.Tensor,
    capacity: pl.constexpr,
    decode: pl.constexpr,
):
    dense = not decode and N_EXPERTS == SCORE_PAD
    small = capacity <= PREFILL_GATE_M_TILE and capacity % PREFILL_SMALL_M_TILE == 0
    aligned = capacity % PREFILL_GATE_M_TILE == 0
    medium = aligned and capacity > PREFILL_GATE_M_TILE and capacity <= PREFILL_COMPACT_ROWS
    large = aligned and capacity > PREFILL_COMPACT_ROWS
    if not decode and num_tokens <= 0:
        empty_tiles = pl.max(1, (capacity + ROUTE_SELECT_T_TILE - 1) // ROUTE_SELECT_T_TILE)
        empty_workers = pl.max(1, pl.min(empty_tiles, pl.system.available_aiv_count()))
        for empty_worker in pl.spmd(empty_workers, name_hint="gate_empty_panel"):
            for row in pl.range(empty_worker, empty_tiles, empty_workers):
                start = row * ROUTE_SELECT_T_TILE
                rows = pl.min(ROUTE_SELECT_T_TILE, capacity - start)
                ids = pl.full([ROUTE_SELECT_T_TILE, TOPK_PAD], dtype=pl.INT32, value=0)
                values = pl.full([ROUTE_SELECT_T_TILE, TOPK_PAD], dtype=pl.FP32, value=0.0)
                indices[start : start + ROUTE_SELECT_T_TILE, :] = pl.set_validshape(ids, rows, TOPK)
                weights[start : start + ROUTE_SELECT_T_TILE, :] = pl.set_validshape(values, rows, TOPK)
    elif dense and ((small and num_tokens >= PREFILL_SMALL_M_TILE) or
                    (medium and num_tokens >= PREFILL_GATE_M_TILE) or
                    (large and num_tokens >= PREFILL_DENSE_MIN_ROWS)):
        if capacity <= PREFILL_GATE_M_TILE:
            _gate_panel_from_fp32(
                xg_buf, gate_w, gate_bias, num_tokens, indices, weights, capacity, decode, True,
                PREFILL_SMALL_M_TILE, PREFILL_SMALL_N_TILE, PREFILL_COMPACT_K_TILE, ROUTE_SELECT_T_TILE,
            )
        elif capacity <= PREFILL_COMPACT_ROWS:
            _gate_panel_from_fp32(
                xg_buf, gate_w, gate_bias, num_tokens, indices, weights, capacity, decode, True,
                PREFILL_GATE_M_TILE, PREFILL_COMPACT_N_TILE, PREFILL_COMPACT_K_TILE, PREFILL_COMPACT_SORT_TILE,
            )
        else:
            _gate_panel_from_fp32(
                xg_buf, gate_w, gate_bias, num_tokens, indices, weights, capacity, decode, True,
                PREFILL_GATE_M_TILE, PREFILL_GATE_N_TILE, PREFILL_GATE_K_TILE, ROUTE_SELECT_T_TILE,
            )
    else:
        _gate_panel_from_fp32(
            xg_buf, gate_w, gate_bias, num_tokens, indices, weights, capacity, decode, False,
            64 if capacity >= 64 else 32 if capacity >= 32 else 16,
            64 if capacity >= 64 else 16, GATE_D_TILE, ROUTE_SELECT_T_TILE,
        )
    return weights


@pl.jit.inline
def _gate_panel_from_fp32(
    xg_buf: pl.Tensor,
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    indices: pl.Tensor,
    weights: pl.Tensor,
    capacity: pl.constexpr,
    decode: pl.constexpr,
    dense_prefill: pl.constexpr,
    m_tile: pl.constexpr,
    n_tile: pl.constexpr,
    k_tile: pl.constexpr,
    sort_tile: pl.constexpr,
):
    padded_rows = (capacity + m_tile - 1) // m_tile * m_tile
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > capacity:
        active_tokens = pl.cast(capacity, pl.INDEX)
    task_tokens = pl.max(active_tokens, pl.cast(1, pl.INDEX))
    task_gate_tiles = (task_tokens + m_tile - 1) // m_tile

    route_scores_buf = pl.create_tensor([padded_rows, SCORE_PAD], dtype=pl.FP32)
    biased_scores_buf = route_scores_buf
    if not dense_prefill:
        biased_scores_buf = pl.create_tensor([padded_rows, SCORE_PAD], dtype=pl.FP32)
    if active_tokens < capacity:
        if not decode and capacity >= 256 and capacity % ROUTE_SELECT_T_TILE == 0:
            # The TopK writer fills the active prefix of the first cleared tile.
            first_tile = active_tokens // ROUTE_SELECT_T_TILE
            tail_tiles = capacity // ROUTE_SELECT_T_TILE - first_tile
            workers = pl.min(pl.min(tail_tiles, ROUTE_SELECT_WORKERS), pl.system.available_aiv_count())
            for worker in pl.spmd(workers, name_hint="gate_clear_tail"):
                zero_ids = pl.full([ROUTE_SELECT_T_TILE, TOPK_PAD], dtype=pl.INT32, value=0)
                zero_weights = pl.full([ROUTE_SELECT_T_TILE, TOPK_PAD], dtype=pl.FP32, value=0.0)
                for tile in pl.range(first_tile + worker, capacity // ROUTE_SELECT_T_TILE, workers):
                    row = tile * ROUTE_SELECT_T_TILE
                    indices[row : row + ROUTE_SELECT_T_TILE, :] = pl.set_validshape(zero_ids, ROUTE_SELECT_T_TILE, TOPK)
                    weights[row : row + ROUTE_SELECT_T_TILE, :] = pl.set_validshape(zero_weights, ROUTE_SELECT_T_TILE, TOPK)
        else:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
                for zt in pl.range(active_tokens, capacity):
                    for zk in pl.range(TOPK):
                        zero_index = pl.cast(0, pl.INT32)
                        pl.write(indices, [zt, zk], zero_index)
                        zero_weight = pl.cast(0.0, pl.FP32)
                        pl.write(weights, [zt, zk], zero_weight)
    if N_EXPERTS < SCORE_PAD:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_score_pad"):
            biased_pad = pl.full([padded_rows, SCORE_PAD - N_EXPERTS], dtype=pl.FP32, value=FP32_NEG_INF)
            biased_scores_buf[:, N_EXPERTS:SCORE_PAD] = biased_pad

    # Reserve the available clusters before any local Cube-to-Vector transfer.
    gate_bias_matrix = pl.reshape(gate_bias, [1, N_EXPERTS])
    gate_tasks = task_gate_tiles * ((N_EXPERTS + n_tile - 1) // n_tile)
    if dense_prefill:
        token_panels = (task_tokens + m_tile - 1) // m_tile
        project_tasks = token_panels * (N_EXPERTS // n_tile)
        project_workers = pl.min(project_tasks, pl.system.available_cluster_count())
        for worker in pl.spmd(project_workers, name_hint="gate_project"):
            for task in pl.range(worker, project_tasks, project_workers):
                row = task // (N_EXPERTS // n_tile) * m_tile
                expert = task % (N_EXPERTS // n_tile) * n_tile
                dot = pl.create_tensor([m_tile, n_tile], dtype=pl.FP32)
                for inner in pl.pipeline(0, D, k_tile, stage=2):
                    panel = xg_buf[row : row + m_tile, inner : inner + k_tile]
                    matrix = gate_w[expert : expert + n_tile, inner : inner + k_tile]
                    dot = pl.matmul_acc(dot, panel, matrix, b_trans=True, init_cond=(inner == 0))
                route_scores_buf[row : row + m_tile, expert : expert + n_tile] = dot
    else:
        gate_workers = pl.min(gate_tasks, pl.min(GATE_WORKERS, pl.system.available_cluster_count()))
        for gate_worker in pl.spmd(gate_workers, name_hint="gate"):
            # The gate matrix is streamed once per expert tile. Bypassing L2 avoids
            # evicting the activations and routing buffers used by the following sort.
            pl.set_cache_policy(gate_w, pl.CachePolicy.BYPASS)
            for gb_idx in pl.range(gate_worker, gate_tasks, gate_workers):
                tg = gb_idx // ((N_EXPERTS + n_tile - 1) // n_tile)
                nb = gb_idx % ((N_EXPERTS + n_tile - 1) // n_tile)
                t1 = tg * m_tile
                n0 = nb * n_tile
                gate_logits_tile = pl.tile.create(
                    [m_tile, n_tile],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Acc,
                    compact=True,
                )
                for kb in pl.pipeline(0, D // k_tile, stage=1):
                    gd_kd = kb * k_tile
                    gd_x = pl.tile.load(xg_buf, [t1, gd_kd], [m_tile, k_tile])
                    valid_n = pl.min(n_tile, N_EXPERTS - n0)
                    gd_w = pl.tile.load(
                        gate_w,
                        [n0, gd_kd],
                        [n_tile, k_tile],
                        valid_shape=[valid_n, k_tile],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    gd_w = pl.tile.transpose_view(gd_w)
                    gate_logits_tile = pl.matmul_acc(
                        gate_logits_tile, gd_x, gd_w, init_cond=(kb == 0),
                    )
                if decode and capacity == 16 and N_EXPERTS == SCORE_PAD:
                    route_scores_buf = pl.store(gate_logits_tile, [t1, n0], route_scores_buf)
                else:
                    gp_relu = pl.maximum(gate_logits_tile, 0.0)
                    gp_abs = pl.abs(gate_logits_tile)
                    gp_neg_abs = pl.neg(gp_abs)
                    gp_exp_abs = pl.exp(gp_neg_abs)
                    gp_exp_plus = pl.add(gp_exp_abs, 1.0)
                    # log(1 + t) loses tiny t in FP32. t - t^2/2 is its lower bound.
                    gp_small = pl.mul(gp_exp_abs, pl.add(pl.mul(gp_exp_abs, -0.5), 1.0))
                    gp_softplus_tail = pl.maximum(pl.log(gp_exp_plus), gp_small)
                    gp_softplus = pl.add(gp_relu, gp_softplus_tail)
                    gp_score = pl.sqrt(gp_softplus)
                    route_scores_buf = pl.store(gp_score, [t1, n0], route_scores_buf)
                    bias_valid_n = pl.min(n_tile, N_EXPERTS - n0)
                    gp_bias_row = pl.tile.load(
                        gate_bias_matrix,
                        [0, n0],
                        [1, n_tile],
                        valid_shape=[1, bias_valid_n],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    gp_biased = pl.col_expand_add(gp_score, gp_bias_row)
                    biased_scores_buf = pl.store(gp_biased, [t1, n0], biased_scores_buf)

    if decode and capacity == 16 and N_EXPERTS == SCORE_PAD:
        if active_tokens > 0:
            decode_route_workers = pl.min(pl.min(active_tokens, ROUTE_SELECT_WORKERS), pl.system.available_aiv_count())
            for decode_worker in pl.spmd(decode_route_workers, name_hint="decode_gate_topk"):
                expert_ids = pl.create_tensor([1, SCORE_PAD], dtype=pl.UINT32)
                expert_ids[:, :] = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                for decode_row in pl.range(decode_worker, active_tokens, decode_route_workers):
                    decode_logits = pl.slice(route_scores_buf, [1, SCORE_PAD], [decode_row, 0])
                    decode_exp_abs = pl.exp(pl.neg(pl.abs(decode_logits)))
                    decode_small = pl.mul(decode_exp_abs, pl.add(pl.mul(decode_exp_abs, -0.5), 1.0))
                    decode_tail = pl.maximum(pl.log(pl.add(decode_exp_abs, 1.0)), decode_small)
                    decode_scores = pl.sqrt(pl.add(pl.maximum(decode_logits, 0.0), decode_tail))
                    biased = pl.create_tensor([1, SCORE_PAD], dtype=pl.FP32)
                    biased[:, :] = pl.add(decode_scores, gate_bias_matrix)
                    decode_chosen = route_topk_row(biased, expert_ids)
                    decode_selected = pl.gather(decode_scores, dim=-1, index=decode_chosen)
                    decode_padded = pl.create_tensor([8, TOPK_PAD], dtype=pl.FP32)
                    decode_padded[:, :] = pl.full([8, TOPK_PAD], dtype=pl.FP32, value=0.0)
                    decode_padded[0:1, :] = decode_selected
                    decode_padded = pl.fillpad(pl.set_validshape(decode_padded, 8, TOPK), pad_value=pl.PadValue.zero)
                    decode_denom = pl.add(pl.row_sum(decode_padded), 1e-20)
                    decode_all_weights = pl.mul(pl.row_expand_div(decode_padded, decode_denom), ROUTE_SCALE)
                    decode_normalized = pl.slice(decode_all_weights, [1, TOPK_PAD], [0, 0], valid_shape=[1, TOPK])
                    indices[decode_row : decode_row + 1, :] = pl.set_validshape(decode_chosen, 1, TOPK)
                    weights[decode_row : decode_row + 1, :] = decode_normalized
    else:
        # V4.1 uses score-only top-k: the V4 hash-route loop is dropped so that PTOAS
        # does not emit an invalid function for a route_hash path that does no work.
        task_route_tiles = (task_tokens + sort_tile - 1) // sort_tile
        route_workers = pl.min(pl.min(task_route_tiles, ROUTE_SELECT_WORKERS), pl.system.available_aiv_count())
        for route_worker in pl.spmd(route_workers, name_hint="route_sort"):
            for ts_idx in pl.range(route_worker, task_route_tiles, route_workers):
                t1 = ts_idx * sort_tile
                # ptoas pto.tmrgsort requires a single source row. Keep the
                # row-wise sort inside a larger worker tile to amortize launch and
                # synchronization overhead across the prefill batch.
                topk_idx_tile = pl.create_tensor([sort_tile, TOPK_PAD], dtype=pl.INT32)
                rows_to_sort = pl.cast(sort_tile, pl.INDEX)
                if decode:
                    rows_to_sort = pl.max(0, pl.min(sort_tile, active_tokens - t1))
                    topk_idx_tile[:, :] = pl.full([sort_tile, TOPK_PAD], dtype=pl.INT32, value=0)
                dense_scores = pl.create_tensor([sort_tile, SCORE_PAD], dtype=pl.FP32)
                biased_panel = pl.create_tensor([sort_tile, SCORE_PAD], dtype=pl.FP32)
                if dense_prefill:
                    values = pl.slice(route_scores_buf, [sort_tile, SCORE_PAD], [t1, 0])
                    exp_abs = pl.exp(pl.neg(pl.abs(values)))
                    small = pl.mul(exp_abs, pl.add(pl.mul(exp_abs, -0.5), 1.0))
                    tail = pl.maximum(pl.log(pl.add(exp_abs, 1.0)), small)
                    dense_scores[:, :] = pl.sqrt(pl.add(pl.maximum(values, 0.0), tail))
                    biased_panel[:, :] = pl.col_expand_add(dense_scores, gate_bias_matrix)
                sr_idx_init = pl.create_tensor([1, SCORE_PAD], dtype=pl.UINT32)
                sr_index_range = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                sr_idx_init[:, :] = sr_index_range
                for sr_tt in pl.range(rows_to_sort):
                    if dense_prefill:
                        sr_row = pl.slice(biased_panel, [1, SCORE_PAD], [sr_tt, 0])
                    else:
                        sr_t = t1 + sr_tt
                        sr_row = pl.slice(biased_scores_buf, [1, SCORE_PAD], [sr_t, 0])
                    sr_i = route_topk_row(sr_row, sr_idx_init)
                    topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i

                local_scores = pl.create_tensor([sort_tile, SCORE_PAD], dtype=pl.FP32)
                if dense_prefill:
                    local_scores[:, :] = dense_scores
                else:
                    local_scores[:, :] = route_scores_buf[t1 : t1 + sort_tile, :]
                gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
                gather_valid = pl.set_validshape(gather_all, sort_tile, TOPK)
                topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
                topk_sum = pl.row_sum(topk_vals_pad)
                denom = pl.reshape(topk_sum, [sort_tile, 1])
                denom = pl.add(denom, 1e-20)
                topk_normalized = pl.row_expand_div(topk_vals_pad, denom)
                normalized_weights = pl.mul(topk_normalized, ROUTE_SCALE)
                valid_rows = pl.min(sort_tile, active_tokens - t1)
                if valid_rows > 0:
                    indices[t1 : t1 + sort_tile, :] = pl.set_validshape(
                        topk_idx_tile, valid_rows, TOPK,
                    )
                    weights[t1 : t1 + sort_tile, :] = pl.set_validshape(
                        normalized_weights, valid_rows, TOPK,
                    )

    return weights


VALIDATION_ROWS = 64


@pl.jit
def l2_gate(
    source: pl.Tensor[[VALIDATION_ROWS, D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    codes: pl.Out[pl.Tensor[[VALIDATION_ROWS, D], pl.INT8]],
    scales: pl.Out[pl.Tensor[[1, VALIDATION_ROWS], pl.FP32]],
    indices: pl.Out[pl.Tensor[[VALIDATION_ROWS, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[VALIDATION_ROWS, TOPK], pl.FP32]],
):
    gate_normalized(source, gate_weight, gate_bias, num_tokens, codes, scales, indices, weights, VALIDATION_ROWS)
    return codes, scales, indices, weights


def golden_gate(tensors):
    """Validate exact activation codes, FP32 scales, all six routes and inactive zeros."""
    from .gemm import golden_activation_int8

    active = int(tensors["num_tokens"])
    source = tensors["source"].clone()
    source[active:] = 0
    tensors["codes"].zero_()
    tensors["codes"][:VALIDATION_ROWS].copy_(golden_activation_int8(source))
    scale = (source.float().abs().amax(-1) * (1.0 / 127.0)).clamp_min(1e-30)
    tensors["scales"].zero_()
    tensors["scales"][:, :VALIDATION_ROWS].copy_(scale.reshape(1, VALIDATION_ROWS))
    tensors["indices"].zero_()
    tensors["weights"].zero_()
    if active:
        weights, indices = golden_gate_reference(source[:active], tensors["gate_weight"], tensors["gate_bias"])
        tensors["indices"][:active] = indices
        tensors["weights"][:active] = weights


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from .weights import Checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--input-data", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    checkpoint = Checkpoint(args.checkpoint) if args.checkpoint is not None else None
    if checkpoint is None:
        generator = torch.Generator().manual_seed(4102)
        gate_weight = torch.randn(N_EXPERTS, D, generator=generator) * D**-0.5
        gate_bias = torch.linspace(-0.25, 0.25, N_EXPERTS)
    else:
        prefix = f"layers.{args.layer}.ffn.gate."
        gate_weight = checkpoint.tensor(prefix + "weight")
        gate_bias = checkpoint.tensor(prefix + "bias")
    source = torch.randn(VALIDATION_ROWS, D, generator=torch.Generator().manual_seed(4103)).bfloat16()
    records = []
    cases = [("empty", 0), ("single", 1), ("tail", 17), ("full", VALIDATION_ROWS), ("ties", 17)]
    if checkpoint is None:
        cases.append(("negative-tail", 17))
    selected_rows = []
    if args.input_data is not None:
        if checkpoint is None:
            parser.error("--input-data requires --checkpoint")
        captured = torch.load(args.input_data, map_location="cpu", weights_only=True)
        if not torch.equal(captured["gate_weight"], gate_weight.float()):
            raise ValueError("captured gate weights differ from the selected checkpoint layer")
        if "actual_indices" in captured:
            different = (captured["actual_indices"] != captured["expected_indices"]).any(-1)
            selected_rows = different.nonzero().flatten().tolist()
        if not selected_rows:
            selected_rows = list(range(min(VALIDATION_ROWS, captured["source"].shape[0])))
        selected_rows = selected_rows[:VALIDATION_ROWS]
        source.zero_()
        source[:len(selected_rows)] = captured["source"][selected_rows]
        gate_bias = captured["gate_bias"]
        cases = [("captured", len(selected_rows))]
    for label, count in cases:
        x = torch.zeros_like(source) if label == "ties" else source
        bias = torch.zeros_like(gate_bias) if label == "ties" else gate_bias
        weight = gate_weight
        if label == "negative-tail":
            x = torch.zeros_like(source)
            x[:, 0] = 1
            weight = torch.zeros_like(gate_weight)
            weight[:, 0] = torch.linspace(-32, -18, N_EXPERTS)
            bias = torch.zeros_like(gate_bias)
        specs = [TensorSpec("source", [VALIDATION_ROWS, D], torch.bfloat16, init_value=x),
                 TensorSpec("gate_weight", [N_EXPERTS, D], torch.float32, init_value=weight),
                 TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=bias),
                 ScalarSpec("num_tokens", torch.int32, count),
                 TensorSpec("codes", [VALIDATION_ROWS, D], torch.int8),
                 TensorSpec("scales", [1, VALIDATION_ROWS], torch.float32),
                 TensorSpec("indices", [VALIDATION_ROWS, TOPK], torch.int32),
                 TensorSpec("weights", [VALIDATION_ROWS, TOPK], torch.float32)]
        result = run(
            l2_gate,
            specs,
            golden_fn=golden_gate,
            config=dict(backend_type=args.platform, execution_mode=args.sim, device_id=args.device),
            compare_fn={"codes": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
                        "scales": ratio_allclose(atol=0.0, rtol=1e-6, max_error_ratio=0.0),
                        "indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
                        "weights": ratio_allclose(atol=1e-6, rtol=3e-5, max_error_ratio=0.0)},
            compile_only=args.compile_only,
        )
        records.append(dict(case=label, count=count, passed=result.passed, work_dir=str(result.work_dir)))
        if not result.passed:
            raise RuntimeError(result.error)
    if args.report:
        args.report.write_text(json.dumps(dict(checkpoint=checkpoint.identity if checkpoint else None, cases=records,
                                              captured_rows=selected_rows), indent=2) + "\n")
