# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 MoE single-layer (prefill), FLASH preset. --ep picks the EP world
size: 2/4/8/16 run N-rank distributed; each rank keeps 32 experts. --tokens picks
the per-rank token capacity."""


# Sub-kernels freeze EP_WORLD_SIZE / n_routed_experts into their shapes at import
# time, so read --ep from argv and override config before importing them below.
# --tokens is read the same way: it sizes this module's own tensors.
import dataclasses
import sys

import config

_EP_CHOICES = (2, 4, 8, 16)
_EP_DEFAULT = 2
_TOKENS_DEFAULT = config.PREFILL_TOKENS


def _parse_argv_int(flag: str, default: int) -> int:
    for i, tok in enumerate(sys.argv):
        if tok == flag and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{flag}="):
            return int(tok.split("=", 1)[1])
    return default


EP = _parse_argv_int("--ep", _EP_DEFAULT)

config.FLASH = dataclasses.replace(
    config.FLASH, n_routed_experts=config.FLASH.n_routed_experts // config.EP_WORLD_SIZE * EP
)
config.EP_WORLD_SIZE = EP

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

from config import FLASH as M, EP_WORLD_SIZE, INT8_AMAX_EPS, INT8_SCALE_MAX
from hc_pre import hc_pre
from hc_post import hc_post
from gate import gate
from expert_shared import expert_shared
from expert_routed import (
    ACT_INTER_TILE, D_OUT_TILE, D_OUT_ACT_TILE, INTER_K_TILE, K_TILE,
    MM_INTER_TILE, QUANT_TILE, RECV_TILE, W2_ACT_INNER, W2_INNER,
)


# Per-rank capacity. --tokens picks a serving layout's share: a single 8192-token
# batch over EP16 is 512 rows per rank.
T = _parse_argv_int("--tokens", _TOKENS_DEFAULT)
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size

HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

N_RANKS = EP_WORLD_SIZE
N_EXPERTS_GLOBAL = M.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS

# Recipes-style prefill transport.  One source may legally send all of its
# routes to one destination, so each destination lane has T * TOPK rows.  The
# receiver owns one such lane per source.  Only live rows cross the wire via
# all_to_all_v; the static capacity is the dropless upper bound, not padding
# that should be computed over.
PREFILL_MOE_SCALE_PAD = 8  # one 32-byte FP32 row; cols 0/1 are dequant scale/routing weight
PREFILL_MOE_EXPERT_SCALE_PAD = 16  # one 64-byte line per receiver expert row
PREFILL_MOE_ROUTE_MAP_PAD = 16  # one 64-byte INT32 line per source-side route
PREFILL_MOE_RETURN_ROWS_PER_BLOCK = 128
PREFILL_MOE_FINALIZE_TOKEN_TILE = 16
PREFILL_MOE_GATE_ZERO_TILE = 16
PREFILL_MOE_GROUPED_EXPERT_TILE = 16

assert N_RANKS in _EP_CHOICES, f"--ep must be one of {_EP_CHOICES} (got {N_RANKS})"
assert N_EXPERTS_GLOBAL == N_RANKS * N_LOCAL


@dataclasses.dataclass(frozen=True)
class PrefillMoELayout:
    """Per-rank token capacity and the corresponding dropless EP storage."""

    tokens: int

    @property
    def routes_per_source(self) -> int:
        return self.tokens * TOPK

    @property
    def total_capacity(self) -> int:
        return N_RANKS * self.routes_per_source

    @property
    def grouped_capacity(self) -> int:
        return self.total_capacity + N_LOCAL * (PREFILL_MOE_GROUPED_EXPERT_TILE - 1)


def check_prefill_moe_slab(token_capacity: int) -> None:
    """Validate a prefill layout without changing the decode module's shape."""
    layout = PrefillMoELayout(token_capacity)
    assert token_capacity > 0
    assert TOPK == 6
    assert N_LOCAL == 32
    assert layout.routes_per_source % PREFILL_MOE_RETURN_ROWS_PER_BLOCK == 0
    assert token_capacity % PREFILL_MOE_FINALIZE_TOKEN_TILE == 0
    assert token_capacity % PREFILL_MOE_GATE_ZERO_TILE == 0
    assert layout.grouped_capacity % PREFILL_MOE_GROUPED_EXPERT_TILE == 0


@pl.jit.inline
def clear_prefill_moe_signals(
    completion_anchor: pl.Tensor[[1, 1, 8], pl.FP32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """clear_moe_signals for the prefill MoE transport's three credit banks.

    Same contract and the same three roles -- counts, payload, combine -- over
    the prefill MoE windows and that path's small completion anchor.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear"):
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(count_signal, [src, 0], zero)
            pl.write(x_signal, [src, 0], zero)
            pl.write(reverse_signal, [src, 0], zero)


# === Prefill routed experts =================================================
def _make_expert_gate_up_quant_tile(grouped_capacity: int, row_tile: int):
    """Build a gate/up projection, SwiGLU, and INT8 quantization tile function."""
    ROW_TILE = row_tile
    VEC_ROW_TILE = RECV_TILE
    ACT_TILE = ACT_INTER_TILE
    H_QUANT_TILE = QUANT_TILE

    @pl.jit.inline(auto_scope=False)
    def expert_gate_up_quant_tile(
        expert_x: pl.Tensor[[grouped_capacity, D], pl.INT8],
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        h_i8: pl.Tensor[[grouped_capacity, MOE_INTER], pl.INT8],
        h_scale_dq: pl.Tensor[[grouped_capacity, 1], pl.FP32],
        tile_row_start: pl.Scalar[pl.INDEX],
        local_expert_id: pl.Scalar[pl.INDEX],
        valid_rows: pl.Scalar[pl.INDEX],
        layout_tid: pl.Scalar[pl.TASK_ID],
        routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
        routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
        routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    ):
        """Compute gate/up projections, apply SwiGLU, and quantize a token tile."""
        weight_e = local_expert_id
        with pl.scope():
            w13_tile_i32 = pl.create_tensor([ROW_TILE, 2 * MOE_INTER], dtype=pl.INT32)

            # Each block computes one gate or up output tile.
            with pl.spmd((2 * MOE_INTER) // MM_INTER_TILE, name_hint="prefill_exp_w13_mm", deps=[layout_tid],) as w13_tid:
                block = pl.tile.get_block_idx()
                n0 = block * MM_INTER_TILE
                if n0 < MOE_INTER:
                    w13_acc = pl.create_tensor([1, ROW_TILE, MM_INTER_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                        x_chunk = expert_x[tile_row_start : tile_row_start + ROW_TILE, k0 : k0 + K_TILE]
                        w13_chunk = routed_w1[weight_e : weight_e + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                        w13_acc = pl.matmul_acc(w13_acc, x_chunk, w13_chunk, b_trans=True, init_cond=k0 == 0)
                    w13_tile_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(w13_acc, [ROW_TILE, MM_INTER_TILE])
                else:
                    up_n0 = n0 - MOE_INTER
                    w13_acc = pl.create_tensor([1, ROW_TILE, MM_INTER_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                        x_chunk = expert_x[tile_row_start : tile_row_start + ROW_TILE, k0 : k0 + K_TILE]
                        w13_chunk = routed_w3[weight_e : weight_e + 1, up_n0 : up_n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                        w13_acc = pl.matmul_acc(w13_acc, x_chunk, w13_chunk, b_trans=True, init_cond=k0 == 0)
                    w13_tile_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(w13_acc, [ROW_TILE, MM_INTER_TILE])

            h_tile_fp32 = pl.create_tensor([ROW_TILE, MOE_INTER], dtype=pl.FP32)
            h_tile_i8 = h_i8[tile_row_start : tile_row_start + ROW_TILE]
            h_tile_scale_dq = h_scale_dq[tile_row_start : tile_row_start + ROW_TILE]
            # Compute activation, row maxima, and INT8 quantization.
            with pl.spmd((ROW_TILE // VEC_ROW_TILE), name_hint="prefill_exp_gate_up_quant", deps=[w13_tid]):
                vector_row = pl.tile.get_block_idx() * VEC_ROW_TILE
                vector_flat_row = tile_row_start + vector_row
                vector_valid_rows = pl.min(VEC_ROW_TILE, pl.max(valid_rows - vector_row, 0))
                x_scale_padded = expert_scale[vector_flat_row : vector_flat_row + VEC_ROW_TILE, 0:PREFILL_MOE_EXPERT_SCALE_PAD]
                x_scale_transposed = pl.transpose(x_scale_padded, axis1=0, axis2=1)
                x_scale_tile = pl.reshape(x_scale_transposed[0:1, :], [VEC_ROW_TILE, 1])
                row_amax = pl.full([1, VEC_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for inter0 in pl.pipeline(0, MOE_INTER, ACT_TILE, stage=2):
                    gate_i32 = w13_tile_i32[vector_row : vector_row + VEC_ROW_TILE, inter0 : inter0 + ACT_TILE]
                    up_i32 = w13_tile_i32[vector_row : vector_row + VEC_ROW_TILE, MOE_INTER + inter0 : MOE_INTER + inter0 + ACT_TILE]
                    gate_fp32 = pl.col_expand_mul(
                        pl.row_expand_mul(pl.cast(gate_i32, target_type=pl.FP32, mode="none"), x_scale_tile,),
                        routed_w1_scale[local_expert_id : local_expert_id + 1, inter0 : inter0 + ACT_TILE],
                    )
                    up_fp32 = pl.col_expand_mul(
                        pl.row_expand_mul(pl.cast(up_i32, target_type=pl.FP32, mode="none"), x_scale_tile),
                        routed_w3_scale[local_expert_id : local_expert_id + 1, inter0 : inter0 + ACT_TILE],
                    )
                    if SWIGLU_LIMIT > 0.0:
                        gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                        up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                    activated_chunk = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                    h_valid = pl.set_validshape(activated_chunk, vector_valid_rows, ACT_TILE)
                    h_padded = pl.fillpad(h_valid, pad_value=pl.PadValue.zero)
                    h_tile_fp32[vector_row : vector_row + VEC_ROW_TILE, inter0 : inter0 + ACT_TILE] = h_padded
                    h_abs = pl.maximum(h_padded, pl.neg(h_padded))
                    row_amax = pl.maximum(row_amax, pl.reshape(pl.row_max(h_abs), [1, VEC_ROW_TILE]))
                quant_scale = pl.div(pl.full([1, VEC_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), row_amax)
                h_tile_scale_dq[vector_row : vector_row + VEC_ROW_TILE, :] = pl.reshape(pl.recip(quant_scale), [VEC_ROW_TILE, 1])
                quant_scale_col = pl.reshape(quant_scale, [VEC_ROW_TILE, 1])
                for k0 in pl.pipeline(0, MOE_INTER, H_QUANT_TILE, stage=2):
                    h_quant_chunk = h_tile_fp32[vector_row : vector_row + VEC_ROW_TILE, k0 : k0 + H_QUANT_TILE]
                    h_scaled = pl.row_expand_mul(h_quant_chunk, quant_scale_col)
                    h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                    h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                    h_tile_i8[vector_row : vector_row + VEC_ROW_TILE, k0 : k0 + H_QUANT_TILE] = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")

    return expert_gate_up_quant_tile


def _make_expert_down_proj_tile(grouped_capacity: int, row_tile: int):
    """Build an expert down-projection tile function with routing weights."""
    ROW_TILE = row_tile
    VEC_ROW_TILE = RECV_TILE
    OUTPUT_ACT_TILE = D_OUT_ACT_TILE

    @pl.jit.inline(auto_scope=False)
    def expert_down_proj_tile(
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        h_i8: pl.Tensor[[grouped_capacity, MOE_INTER], pl.INT8],
        h_scale_dq: pl.Tensor[[grouped_capacity, 1], pl.FP32],
        tile_row_start: pl.Scalar[pl.INDEX],
        local_expert_id: pl.Scalar[pl.INDEX],
        routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
        routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
        expert_y: pl.Tensor[[grouped_capacity, D], pl.BF16],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Write routing-weighted BF16 down-projection output and return its TaskId."""
        weight_e = local_expert_id
        h_tile_i8 = h_i8[tile_row_start : tile_row_start + ROW_TILE]
        h_tile_scale_dq = h_scale_dq[tile_row_start : tile_row_start + ROW_TILE]

        y_i32 = pl.create_tensor([ROW_TILE, D], dtype=pl.INT32)
        with pl.spmd(D // (W2_INNER * D_OUT_TILE), name_hint="prefill_exp_w2_mm", allow_early_resolve=True,) as w2_tid:
            block = pl.tile.get_block_idx()
            d_base = block * (W2_INNER * D_OUT_TILE)
            for inner in pl.range(W2_INNER):
                d0 = d_base + inner * D_OUT_TILE
                y_acc = pl.create_tensor([1, ROW_TILE, D_OUT_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, MOE_INTER, INTER_K_TILE, stage=2):
                    h_w2_chunk = h_tile_i8[:, k0 : k0 + INTER_K_TILE]
                    w2_chunk = routed_w2[weight_e : weight_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K_TILE]
                    y_acc = pl.matmul_acc(y_acc, h_w2_chunk, w2_chunk, b_trans=True, init_cond=k0 == 0)
                y_i32[:, d0 : d0 + D_OUT_TILE] = pl.reshape(y_acc, [ROW_TILE, D_OUT_TILE])

        # Expose the disjoint row tile to dependency tracking.
        expert_y_tile = expert_y[tile_row_start : tile_row_start + ROW_TILE, :]
        with pl.spmd(
            (ROW_TILE // VEC_ROW_TILE) * (D // (W2_ACT_INNER * OUTPUT_ACT_TILE)),
            name_hint="prefill_exp_w2_act",
            deps=[w2_tid],
            allow_early_resolve=False,
        ) as w2_act_tid:
            grid_block = pl.tile.get_block_idx()
            vector_row = (grid_block // (D // (W2_ACT_INNER * OUTPUT_ACT_TILE))) * VEC_ROW_TILE
            block = grid_block % (D // (W2_ACT_INNER * OUTPUT_ACT_TILE))
            vector_flat_row = tile_row_start + vector_row
            d_base = block * (W2_ACT_INNER * OUTPUT_ACT_TILE)
            # Load an aligned tile before extracting the weight column;
            # a direct narrow column view would retain the row pitch.
            scale_padded = expert_scale[vector_flat_row : vector_flat_row + VEC_ROW_TILE, 0:PREFILL_MOE_EXPERT_SCALE_PAD]
            scale_transposed = pl.transpose(scale_padded, axis1=0, axis2=1)
            weight_col = pl.reshape(scale_transposed[1:2, :], [VEC_ROW_TILE, 1])
            row_scale = pl.mul(h_tile_scale_dq[vector_row : vector_row + VEC_ROW_TILE, :], weight_col)
            for inner in pl.pipeline(W2_ACT_INNER, stage=2):
                d0 = d_base + inner * OUTPUT_ACT_TILE
                y_fp32 = pl.cast(y_i32[vector_row : vector_row + VEC_ROW_TILE, d0 : d0 + OUTPUT_ACT_TILE], target_type=pl.FP32, mode="none")
                y_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(y_fp32, row_scale),
                    routed_w2_scale[local_expert_id : local_expert_id + 1, d0 : d0 + OUTPUT_ACT_TILE],
                )
                expert_y_tile[vector_row : vector_row + VEC_ROW_TILE, d0 : d0 + OUTPUT_ACT_TILE] = pl.cast(
                    y_fp32,
                    target_type=pl.BF16,
                    mode="rint",
                )
        return w2_act_tid

    return expert_down_proj_tile


def make_prefill_expert_grouped(grouped_capacity: int):
    """Specialize scratch and TaskId-array capacity without module-global overrides.

    The input/output layout is caller-owned; each local expert still occupies
    a 16-row-aligned slab. TaskId arrays require a compile-time extent, so only
    storage capacity is specialized while computation remains count-driven.
    Complete groups use M128; the aligned tail uses M64/M32/M16 tiles.
    """
    if grouped_capacity <= 0 or grouped_capacity % RECV_TILE:
        raise ValueError("grouped expert capacity must be a positive multiple of RECV_TILE")
    grouped_tile_capacity = grouped_capacity // RECV_TILE
    gate_up_quant_m128 = _make_expert_gate_up_quant_tile(grouped_capacity, 128)
    gate_up_quant_m64 = _make_expert_gate_up_quant_tile(grouped_capacity, 64)
    gate_up_quant_m32 = _make_expert_gate_up_quant_tile(grouped_capacity, 32)
    gate_up_quant_m16 = _make_expert_gate_up_quant_tile(grouped_capacity, RECV_TILE)
    down_proj_m128 = _make_expert_down_proj_tile(grouped_capacity, 128)
    down_proj_m64 = _make_expert_down_proj_tile(grouped_capacity, 64)
    down_proj_m32 = _make_expert_down_proj_tile(grouped_capacity, 32)
    down_proj_m16 = _make_expert_down_proj_tile(grouped_capacity, RECV_TILE)

    @pl.jit.inline(auto_scope=False)
    def prefill_expert_grouped(
        expert_x: pl.Tensor[[grouped_capacity, D], pl.INT8],
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
        routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
        routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
        routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
        routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
        expert_y: pl.Tensor[[grouped_capacity, D], pl.BF16],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Count-driven grouped routed-expert compute with MTP quantization.

        ``expert_x`` and ``expert_y`` are expert-major. Each expert starts at the
        sum of the preceding expert counts rounded up to ``RECV_TILE`` rows. Only
        columns zero and one of ``expert_scale`` hold the FP32 input dequant scale
        and routing weight. As in ordinary MTP, W2 dequantization applies the
        routing weight before the expert output is rounded to BF16.

        The dispatch contract guarantees non-negative counts whose sum is at most
        the live-route capacity of the EP transport. Extra grouped capacity is
        padding that keeps every final 16-row tile inside its owning expert.
        """
        # Host orchestration reads these bases immediately after the layout task.
        # Keep automatic dependency tracking so that read waits for this writer.
        expert_bases = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
        # Keep the aligned A8 workspace in the caller's MoE stage scope.
        h_i8 = pl.create_tensor([grouped_capacity, MOE_INTER], dtype=pl.INT8)
        h_scale_dq = pl.create_tensor([grouped_capacity, 1], dtype=pl.FP32, manual_dep=True)
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_exp_group_layout",
            allow_early_resolve=True,
        ) as layout_tid:
            grouped_base = pl.cast(0, pl.INDEX)
            for local_expert_id in pl.range(N_LOCAL):
                pl.write(expert_bases, [local_expert_id, 0], pl.cast(grouped_base, pl.INT32))
                rows = pl.cast(pl.read(expert_counts, [local_expert_id, 0]), pl.INDEX)
                aligned_rows = ((rows + RECV_TILE - 1) // RECV_TILE) * RECV_TILE
                grouped_base = grouped_base + aligned_rows

        # Hoisted so the scope-external terminal fence can consume every expert's
        # completion TaskId after the compute scratch lifetime closes.
        expert_completion_tids = pl.array.create(N_LOCAL, pl.TASK_ID)

        with pl.scope():
            # Gate/up projections share one dispatch and intermediate layout while
            # reading the same separate resident weight roots as decode.
            for local_expert_id in pl.parallel(N_LOCAL):
                flat_base = pl.cast(pl.read(expert_bases, [local_expert_id, 0]), pl.INDEX)
                n_rows = pl.read(expert_counts, [local_expert_id, 0])
                main_tiles = n_rows // 128
                aligned_tail = (((n_rows % 128) + RECV_TILE - 1) // RECV_TILE) * RECV_TILE
                tail64 = aligned_tail // 64
                tail32 = (aligned_tail % 64) // 32
                tail16 = (aligned_tail % 32) // RECV_TILE
                for tile in pl.parallel(main_tiles):
                    tile_row_start = flat_base + tile * 128
                    gate_up_quant_m128(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, pl.cast(128, pl.INDEX), layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail64):
                    tile_row = main_tiles * 128 + 0 + tile * 64
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(64, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m64(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail32):
                    tile_row = main_tiles * 128 + tail64 * 64 + tile * 32
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(32, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m32(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail16):
                    tile_row = main_tiles * 128 + tail64 * 64 + tail32 * 32 + tile * 16
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(16, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m16(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )

            # W2 produces weighted BF16 expert output. Each expert accumulates the
            # TaskId of every live output tile, then one dummy task fences those
            # producers. The scope-external dummy below fans the per-expert fences
            # into the single completion TaskId that prefill_moe_combine consumes.
            for local_expert_id in pl.parallel(N_LOCAL):
                w2_act_tids = pl.array.create(grouped_tile_capacity, pl.TASK_ID)
                flat_base = pl.cast(pl.read(expert_bases, [local_expert_id, 0]), pl.INDEX)
                n_rows = pl.read(expert_counts, [local_expert_id, 0])
                main_tiles = n_rows // 128
                aligned_tail = (((n_rows % 128) + RECV_TILE - 1) // RECV_TILE) * RECV_TILE
                tail64 = aligned_tail // 64
                tail32 = (aligned_tail % 64) // 32
                tail16 = (aligned_tail % 32) // RECV_TILE
                for tile in pl.parallel(main_tiles):
                    tile_row_start = flat_base + tile * 128
                    w2_act_tid = down_proj_m128(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y,
                    )
                    w2_act_tids[tile] = w2_act_tid
                for tile in pl.parallel(tail64):
                    tile_row_start = flat_base + main_tiles * 128 + 0 + tile * 64
                    w2_act_tid = down_proj_m64(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[main_tiles + tile] = w2_act_tid
                for tile in pl.parallel(tail32):
                    tile_row_start = flat_base + main_tiles * 128 + tail64 * 64 + tile * 32
                    w2_act_tid = down_proj_m32(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[main_tiles + tail64 + tile] = w2_act_tid
                for tile in pl.parallel(tail16):
                    tile_row_start = flat_base + main_tiles * 128 + tail64 * 64 + tail32 * 32 + tile * 16
                    w2_act_tid = down_proj_m16(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[main_tiles + tail64 + tail32 + tile] = w2_act_tid

                expert_completion_tid = pl.system.task_dummy(deps=[w2_act_tids])
                expert_completion_tids[local_expert_id] = expert_completion_tid

        completion_tid = pl.system.task_dummy(
            deps=[expert_completion_tids[local_expert_id] for local_expert_id in range(N_LOCAL)]
        )

        return completion_tid


    return prefill_expert_grouped


# === Prefill MoE dispatch ===================================================
def make_prefill_moe(layout: PrefillMoELayout):
    """Build the shared prefill MoE for a caller-owned token/window capacity."""
    check_prefill_moe_slab(layout.tokens)
    T = layout.tokens
    GROUPED_PACK_BLOCKS_PER_EXPERT = 8 if layout.tokens >= 1024 else 1
    SOURCE_PACK_BLOCKS = 48 if layout.tokens >= 1024 else N_RANKS
    RESORT_BLOCKS_PER_EXPERT = 8 if layout.tokens >= 1024 else 1
    DISPATCH_ROW_TILE = 16 if layout.tokens >= 1024 else 1
    REVERSE_ROW_TILE = 16 if layout.tokens >= 1024 else 1
    PREFILL_MOE_ROUTES_PER_SRC = layout.routes_per_source
    PREFILL_MOE_PEER_CAP = layout.routes_per_source
    PREFILL_MOE_TOTAL_CAP = layout.total_capacity
    PREFILL_MOE_GROUPED_TOTAL_CAP = layout.grouped_capacity
    PREFILL_MOE_SEND_SCALE_PAD = 16  # one 64-byte line per independently packed row
    prefill_expert_grouped = make_prefill_expert_grouped(layout.grouped_capacity)

    @pl.jit.inline
    def prefill_moe_dispatch(
        indices: pl.Tensor[[T, TOPK], pl.INT32],
        x_norm_i8: pl.Tensor[[T, D], pl.INT8],
        x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
        weights: pl.Tensor[[T, TOPK], pl.FP32],
        # Receiver-side expert-major tensors.  Only sum(expert_counts_out) leading
        # rows are live; consumers must never compute over the static tail.
        expert_x_out: pl.Tensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
        expert_scale_out: pl.Tensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        expert_counts_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
        recv_expert_counts_out: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
        # Source-side state retained for the reverse exchange/combine. Routing
        # weights travel in the existing padded FP32 scale payload's column one.
        send_counts_out: pl.Tensor[[N_RANKS, 1], pl.INT32],
        route_to_packed_out: pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD], pl.INT32],
        # Counts and the INT8/FP32 payload rails use independent transport windows.
        # The scale rail shares the payload's credit bank, so it needs no signal of
        # its own.
        count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
        count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
        x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
        x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
        scale_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
        prior_dep: pl.Scalar[pl.TASK_ID],
        # Monotonic per-forward epoch for the manual transport's barriers, shared
        # with the reverse exchange's convention.
        epoch: pl.Scalar[pl.INT32],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Count/payload dispatch with MTP FP32 routing weights and expert re-sort.

        Stable destination/expert packing plus the exchanged ``[source, expert]``
        count matrix makes expert-id and route-id payload rails unnecessary.  The
        payload carries INT8 hidden rows, FP32 dequant scales, and routing weights.
        ``route_to_packed_out`` stays on the source for the reverse combine.
        """

        send_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)

        # Source routes are packed contiguously across destinations. Receive
        # windows retain one worst-case lane per source for dropless routing.
        x_send = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, D], dtype=pl.INT8, manual_dep=True)
        scale_send = pl.create_tensor(
            [PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_SEND_SCALE_PAD],
            dtype=pl.FP32,
            manual_dep=True,
        )

        # Count once in a single InCore task.  This avoids cache-line lost updates
        # on the INT32 count rows and gives packing stable expert prefixes.
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_moe_dispatch_count",
            allow_early_resolve=True,
        ) as count_tid:
            active_tokens = pl.cast(num_tokens, pl.INDEX)
            if active_tokens < 0:
                active_tokens = pl.cast(0, pl.INDEX)
            if active_tokens > T:
                active_tokens = pl.cast(T, pl.INDEX)

            counts = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
            for dst in pl.range(N_RANKS):
                for local_e in pl.range(N_LOCAL):
                    counts[dst * N_LOCAL + local_e] = 0

            for token in pl.range(active_tokens):
                for topk in pl.range(TOPK):
                    expert = pl.read(indices, [token, topk])
                    dst = expert // N_LOCAL
                    local_e = expert - dst * N_LOCAL
                    lane = dst * N_LOCAL + local_e
                    counts[lane] = counts[lane] + 1

            for dst in pl.range(N_RANKS):
                rank_total = pl.const(0, pl.INT32)
                for local_e in pl.range(N_LOCAL):
                    count = counts[dst * N_LOCAL + local_e]
                    pl.write(send_expert_counts, [dst, local_e], count)
                    rank_total = rank_total + count
                pl.write(send_counts_out, [dst, 0], rank_total)

            # Assign stable expert-major route positions from the histogram.
            packed_cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
            packed_prefix = pl.const(0, pl.INT32)
            for expert_id in pl.range(N_RANKS * N_LOCAL):
                packed_cursor[expert_id] = packed_prefix
                packed_prefix = packed_prefix + counts[expert_id]
            for token in pl.range(active_tokens):
                for topk in pl.range(TOPK):
                    route_expert_id = pl.read(indices, [token, topk])
                    packed_slot = packed_cursor[route_expert_id]
                    pl.write(route_to_packed_out, [token * TOPK + topk, 0], packed_slot)
                    packed_cursor[route_expert_id] = packed_slot + 1

        # Pack complete tokens into padded expert-major route rows.
        with pl.spmd(
            SOURCE_PACK_BLOCKS,
            name_hint="prefill_moe_dispatch_pack",
            deps=[count_tid],
            allow_early_resolve=True,
        ) as pack_tid:
            pack_block = pl.tile.get_block_idx()
            active_tokens = pl.cast(num_tokens, pl.INDEX)
            if active_tokens < 0:
                active_tokens = pl.cast(0, pl.INDEX)
            if active_tokens > T:
                active_tokens = pl.cast(T, pl.INDEX)
            scale_row = pl.tile.full([1, PREFILL_MOE_SEND_SCALE_PAD], dtype=pl.FP32, value=0.0)
            route_map_row = pl.tile.full([1, PREFILL_MOE_ROUTE_MAP_PAD], dtype=pl.INT32, value=0)
            for token in pl.range(pack_block, active_tokens, SOURCE_PACK_BLOCKS):
                hidden_row = x_norm_i8[token:token + 1, :]
                token_scale = pl.read(x_norm_scale, [token, 0])
                for topk in pl.range(TOPK):
                    route = token * TOPK + topk
                    packed_slot = pl.read(route_to_packed_out, [route, 0])
                    packed_row = pl.cast(packed_slot, pl.INDEX)
                    x_send[packed_row:packed_row + 1, :] = hidden_row
                    pl.tile.write(scale_row, [0, 0], token_scale)
                    pl.tile.write(scale_row, [0, 1], pl.read(weights, [token, topk]))
                    pl.tile.store(scale_row, [packed_row, 0], scale_send)
                    pl.tile.write(route_map_row, [0, 0], packed_slot)
                    pl.tile.store(route_map_row, [route, 0], route_to_packed_out)

        # Counts and both payload rails ride the same hand-written transport the
        # reverse exchange uses: per-destination puts into the peer's own lane,
        # then a monotonic-epoch notify/wait. Serialize count -> payload like
        # Recipes: besides providing the authoritative receive splits, completing
        # the small count exchange before the payload prevents its scalar consumer
        # from racing a capacity-sized payload rail on CP8.
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_moe_dispatch_count_exchange",
            # prior_dep (the caller's attention-done anchor) transitively
            # post-dominates the PREVIOUS layer's combine on this rank. Without it
            # the scheduler may start this layer's exchange on the shared
            # count/payload signal banks while the previous layer's credit barrier
            # is still in flight on a peer — a cross-rank deadlock invisible to
            # local dataflow (cf. #1076).
            deps=[count_tid, prior_dep],
        ) as count_exchange_tid:
            my_rank = pld.system.rank(pld.system.get_comm_ctx(count_target))
            count_row = pl.tile.full([1, N_LOCAL], dtype=pl.INT32, value=0)
            for dest in pl.range(N_RANKS):
                for local_e in pl.range(N_LOCAL):
                    pl.tile.write(count_row, [0, local_e], pl.read(send_expert_counts, [dest, local_e]))
                pld.tile.remote_store(count_row, target=count_target, peer=dest, offsets=[my_rank, 0])
                if dest != my_rank:
                    pld.system.notify(
                        target=count_signal,
                        peer=dest,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(signal=count_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

            for local_e in pl.range(N_LOCAL):
                expert_total = pl.const(0, pl.INT32)
                for src in pl.range(N_RANKS):
                    count = pl.read(count_target, [src, local_e])
                    pl.write(recv_expert_counts_out, [src, local_e], count)
                    expert_total = expert_total + count
                pl.write(expert_counts_out, [local_e, 0], expert_total)

        # Four disjoint row streams per destination push payload and scale.
        # All streams finish before the shared notify. The scale rail rides the
        # same credit as the payload: it is stored before the notify in program
        # order, exactly as dispatch()'s aux/route rails ride data_arrived.
        with pl.spmd(
            N_RANKS * 4,
            name_hint="prefill_moe_dispatch_payload_put",
            deps=[pack_tid, count_exchange_tid],
            allow_early_resolve=False,
        ) as payload_put_tid:
            transfer_block = pl.tile.get_block_idx()
            dest = transfer_block // 4
            row_group = transfer_block % 4
            my_rank = pld.system.rank(pld.system.get_comm_ctx(x_target))
            n_rows = pl.cast(pl.read(send_counts_out, [dest, 0]), pl.INDEX)
            if n_rows < 0:
                n_rows = pl.cast(0, pl.INDEX)
            if n_rows > PREFILL_MOE_PEER_CAP:
                n_rows = pl.cast(PREFILL_MOE_PEER_CAP, pl.INDEX)
            src_base = pl.cast(0, pl.INDEX)
            for prior_dst in pl.range(dest):
                src_base = src_base + pl.cast(pl.read(send_counts_out, [prior_dst, 0]), pl.INDEX)
            dst_base = pl.cast(my_rank, pl.INDEX) * PREFILL_MOE_PEER_CAP
            full_rows = (n_rows // DISPATCH_ROW_TILE) * DISPATCH_ROW_TILE
            for row in pl.range(row_group * DISPATCH_ROW_TILE, full_rows, 4 * DISPATCH_ROW_TILE):
                pld.tensor.put(
                    dst=x_target,
                    peer=dest,
                    src=x_send,
                    dst_offsets=[dst_base + row, 0],
                    src_offsets=[src_base + row, 0],
                    shape=[DISPATCH_ROW_TILE, D],
                )
                pld.tensor.put(
                    dst=scale_target,
                    peer=dest,
                    src=scale_send,
                    dst_offsets=[dst_base + row, 0],
                    src_offsets=[src_base + row, 0],
                    shape=[DISPATCH_ROW_TILE, PREFILL_MOE_SCALE_PAD],
                )
            for row in pl.range(full_rows + row_group, n_rows, 4):
                pld.tensor.put(
                    dst=x_target,
                    peer=dest,
                    src=x_send,
                    dst_offsets=[dst_base + row, 0],
                    src_offsets=[src_base + row, 0],
                    shape=[1, D],
                )
                pld.tensor.put(
                    dst=scale_target,
                    peer=dest,
                    src=scale_send,
                    dst_offsets=[dst_base + row, 0],
                    src_offsets=[src_base + row, 0],
                    shape=[1, PREFILL_MOE_SCALE_PAD],
                )

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_moe_dispatch_payload_wait",
            deps=[payload_put_tid],
        ) as payload_exchange_tid:
            my_rank = pld.system.rank(pld.system.get_comm_ctx(x_target))
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=x_signal,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(signal=x_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

        # Count-driven copies trim every source block before reading the transport
        # windows. Output order is expert-major, then source-major, preserving the
        # source's token/topk order inside each expert.
        with pl.spmd(
            N_LOCAL * RESORT_BLOCKS_PER_EXPERT,
            name_hint="prefill_moe_dispatch_resort",
            deps=[count_exchange_tid, payload_exchange_tid],
            allow_early_resolve=False,
        ) as resort_tid:
            resort_block = pl.tile.get_block_idx()
            local_e = resort_block // RESORT_BLOCKS_PER_EXPERT
            row_group = resort_block % RESORT_BLOCKS_PER_EXPERT
            if pl.read(expert_counts_out, [local_e, 0]) > 0:
                expert_scale_row = pl.tile.full([1, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32, value=0.0)

                expert_base = pl.cast(0, pl.INDEX)
                for prior_e in pl.range(local_e):
                    expert_base = expert_base + pl.cast(pl.read(expert_counts_out, [prior_e, 0]), pl.INDEX)

                source_prefix = pl.cast(0, pl.INDEX)
                for src in pl.range(N_RANKS):
                    source_expert_base = pl.cast(0, pl.INDEX)
                    for prior_e in pl.range(local_e):
                        source_expert_base = source_expert_base + pl.cast(
                            pl.read(recv_expert_counts_out, [src, prior_e]),
                            pl.INDEX,
                        )
                    n_rows = pl.cast(pl.read(recv_expert_counts_out, [src, local_e]), pl.INDEX)
                    input_base = src * PREFILL_MOE_PEER_CAP + source_expert_base
                    output_base = expert_base + source_prefix
                    full_rows = (n_rows // 16) * 16
                    for row in pl.range(row_group * 16, full_rows, RESORT_BLOCKS_PER_EXPERT * 16):
                        input_row = input_base + row
                        output_row = output_base + row
                        expert_x_out[output_row : output_row + 16, :] = x_target[input_row : input_row + 16, :]
                        for inner in pl.range(16):
                            pl.tile.write(expert_scale_row, [0, 0], pl.read(scale_target, [input_row + inner, 0]))
                            pl.tile.write(expert_scale_row, [0, 1], pl.read(scale_target, [input_row + inner, 1]))
                            pl.tile.store(expert_scale_row, [output_row + inner, 0], expert_scale_out)
                    for row in pl.range(full_rows + row_group, n_rows, RESORT_BLOCKS_PER_EXPERT):
                        input_row = input_base + row
                        output_row = output_base + row
                        expert_x_out[output_row : output_row + 1, :] = x_target[input_row : input_row + 1, :]
                        pl.tile.write(expert_scale_row, [0, 0], pl.read(scale_target, [input_row, 0]))
                        pl.tile.write(expert_scale_row, [0, 1], pl.read(scale_target, [input_row, 1]))
                        pl.tile.store(expert_scale_row, [output_row, 0], expert_scale_out)
                    source_prefix = source_prefix + n_rows

        return resort_tid


    @pl.jit.inline
    def _prefill_moe_reverse_exchange(
        reverse_send: pl.Tensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
        reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
        reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
        reverse_send_counts: pl.Tensor[[N_RANKS, 1], pl.INT32],
        epoch: pl.Scalar[pl.INT32],
        resort_tid: pl.Scalar[pl.TASK_ID],
        counts_tid: pl.Scalar[pl.TASK_ID],
    ) -> pl.Scalar[pl.TASK_ID]:
        # Four disjoint row streams per destination share one completion TaskId.
        # The existing notify/wait runs only after all payload blocks finish.
        # pypto all_to_all_v sizes its transfer by the runtime row count, and a
        # partial extent deadlocks 8 ranks under skewed counts (pypto#2536).
        with pl.spmd(
            N_RANKS * 4,
            name_hint="prefill_moe_combine_y_put",
            deps=[resort_tid, counts_tid],
            allow_early_resolve=False,
        ) as reverse_put_tid:
            transfer_block = pl.tile.get_block_idx()
            dest = transfer_block // 4
            row_group = transfer_block % 4
            my_rank = pld.system.rank(pld.system.get_comm_ctx(reverse_target))
            n_rows = pl.cast(pl.read(reverse_send_counts, [dest, 0]), pl.INDEX)
            if n_rows < 0:
                n_rows = pl.cast(0, pl.INDEX)
            if n_rows > PREFILL_MOE_PEER_CAP:
                n_rows = pl.cast(PREFILL_MOE_PEER_CAP, pl.INDEX)
            src_base = dest * PREFILL_MOE_PEER_CAP
            dst_base = pl.cast(my_rank, pl.INDEX) * PREFILL_MOE_PEER_CAP
            bulk_rows = (n_rows // REVERSE_ROW_TILE) * REVERSE_ROW_TILE
            for row in pl.range(row_group * REVERSE_ROW_TILE, bulk_rows, 4 * REVERSE_ROW_TILE):
                pld.tensor.put(
                    dst=reverse_target,
                    peer=dest,
                    src=reverse_send,
                    dst_offsets=[dst_base + row, 0],
                    src_offsets=[src_base + row, 0],
                    shape=[REVERSE_ROW_TILE, D],
                )
            for row in pl.range(bulk_rows + row_group, n_rows, 4):
                pld.tensor.put(
                    dst=reverse_target,
                    peer=dest,
                    src=reverse_send,
                    dst_offsets=[dst_base + row, 0],
                    src_offsets=[src_base + row, 0],
                    shape=[1, D],
                )

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_moe_combine_y_wait",
            deps=[reverse_put_tid],
        ) as reverse_exchange_tid:
            my_rank = pld.system.rank(pld.system.get_comm_ctx(reverse_target))
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=reverse_signal,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(signal=reverse_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        return reverse_exchange_tid


    @pl.jit.inline
    def prefill_moe_combine(
        expert_y: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, D], pl.BF16],
        expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
        recv_expert_counts: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
        route_to_packed: pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD], pl.INT32],
        forward_send_counts: pl.Tensor[[N_RANKS, 1], pl.INT32],
        shared_y: pl.Tensor[[T, D], pl.BF16],
        ffn_out: pl.Tensor[[T, D], pl.BF16],
        returned_y: pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, D], pl.BF16],
        reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
        reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        dispatch_tid: pl.Scalar[pl.TASK_ID],
        expert_tid: pl.Scalar[pl.TASK_ID],
        # Monotonic per-forward epoch for the manual transport's barrier (the
        # collective's own barrier is self-clearing and ignores it).
        epoch: pl.Scalar[pl.INT32],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Reverse exchange and MTP combine directly from aligned expert outputs.

        ``expert_y`` has 16-row-aligned expert slabs, source-major within each slab. It is
        restored to the forward collective's source-major order before the reverse
        exchange returns each row to the rank that routed it.  Its fixed peer lanes
        are compacted to the Recipe's concatenated live output splits, then the
        source-local packed route map selects already weighted BF16 rows for the
        final top-k sum. No route or expert metadata crosses EP.
        """

        reverse_send = pl.create_tensor([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.BF16, manual_dep=True)
        reverse_send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32, manual_dep=True)

        # Row count per reverse destination: on an expert rank each destination is
        # one original source, so it receives the sum across this rank's local
        # experts.  The exchange below walks exactly these counts.
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_moe_combine_counts",
            deps=[dispatch_tid],
            allow_early_resolve=True,
        ) as reverse_counts_tid:
            for src in pl.range(N_RANKS):
                source_total = pl.const(0, pl.INT32)
                for local_e in pl.range(N_LOCAL):
                    source_total = source_total + pl.read(recv_expert_counts, [src, local_e])
                pl.write(reverse_send_counts, [src, 0], source_total)

        # Invert prefill_moe_dispatch's receiver re-sort.  Each task owns the complete
        # fixed-capacity lane for one original source; only its live prefix is
        # written and subsequently transferred.
        with pl.spmd(
            N_RANKS,
            name_hint="prefill_moe_combine_inverse_resort",
            deps=[dispatch_tid, expert_tid],
            allow_early_resolve=False,
        ) as inverse_resort_tid:
            src = pl.tile.get_block_idx()
            send_base = src * PREFILL_MOE_PEER_CAP
            send_prefix = pl.cast(0, pl.INDEX)
            expert_base = pl.cast(0, pl.INDEX)
            for local_e in pl.range(N_LOCAL):
                source_prefix = pl.cast(0, pl.INDEX)
                for prior_src in pl.range(src):
                    source_prefix = source_prefix + pl.cast(
                        pl.read(recv_expert_counts, [prior_src, local_e]),
                        pl.INDEX,
                    )
                n_rows = pl.cast(pl.read(recv_expert_counts, [src, local_e]), pl.INDEX)
                input_base = expert_base + source_prefix
                output_base = send_base + send_prefix
                full_rows = (n_rows // PREFILL_MOE_GROUPED_EXPERT_TILE) * PREFILL_MOE_GROUPED_EXPERT_TILE
                for row in pl.range(0, full_rows, PREFILL_MOE_GROUPED_EXPERT_TILE):
                    input_row = input_base + row
                    output_row = output_base + row
                    reverse_send[output_row : output_row + PREFILL_MOE_GROUPED_EXPERT_TILE, :] = (
                        expert_y[input_row : input_row + PREFILL_MOE_GROUPED_EXPERT_TILE, :]
                    )
                for row in pl.range(full_rows, n_rows):
                    input_row = input_base + row
                    output_row = output_base + row
                    reverse_send[output_row : output_row + 1, :] = expert_y[input_row : input_row + 1, :]
                send_prefix = send_prefix + n_rows
                expert_rows = pl.cast(pl.read(expert_counts, [local_e, 0]), pl.INDEX)
                aligned_rows = ((expert_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1) // PREFILL_MOE_GROUPED_EXPERT_TILE) * PREFILL_MOE_GROUPED_EXPERT_TILE
                expert_base = expert_base + aligned_rows

        reverse_exchange_tid = _prefill_moe_reverse_exchange(
            reverse_send,
            reverse_target,
            reverse_signal,
            reverse_send_counts,
            epoch,
            inverse_resort_tid,
            reverse_counts_tid,
        )

        # PTO all_to_all_v receives into fixed peer-capacity lanes.  Recipe's
        # all_to_all_single instead returns the known output splits concatenated
        # into exactly T*TOPK rows.  Narrow the communication staging window to
        # that source-local layout before finalize-routing.
        return_blocks_per_rank = (PREFILL_MOE_PEER_CAP // PREFILL_MOE_RETURN_ROWS_PER_BLOCK)
        with pl.spmd(
            N_RANKS * return_blocks_per_rank,
            name_hint="prefill_moe_combine_return",
            deps=[reverse_exchange_tid],
            allow_early_resolve=False,
        ) as return_rows_tid:
            block = pl.tile.get_block_idx()
            expert_rank = block // return_blocks_per_rank
            chunk = block - expert_rank * return_blocks_per_rank
            row0 = chunk * PREFILL_MOE_RETURN_ROWS_PER_BLOCK
            live_row_base = pl.cast(0, pl.INDEX)
            for prior_rank in pl.range(expert_rank):
                live_row_base = live_row_base + pl.cast(pl.read(forward_send_counts, [prior_rank, 0]), pl.INDEX)
            n_rows = pl.cast(pl.read(forward_send_counts, [expert_rank, 0]), pl.INDEX)
            staging_base = expert_rank * PREFILL_MOE_PEER_CAP
            if row0 < n_rows:
                chunk_rows = n_rows - row0
                if chunk_rows > PREFILL_MOE_RETURN_ROWS_PER_BLOCK:
                    chunk_rows = PREFILL_MOE_RETURN_ROWS_PER_BLOCK
                for row in pl.range(chunk_rows):
                    returned_y[live_row_base + row0 + row : live_row_base + row0 + row + 1, :] = reverse_target[
                        staging_base + row0 + row : staging_base + row0 + row + 1,
                        :,
                    ]

        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)

        # Match ordinary MTP combine: add shared and already weighted BF16 expert
        # rows in FP32 in top-k order, then round the final sum to BF16.
        with pl.spmd(
            T // PREFILL_MOE_FINALIZE_TOKEN_TILE,
            name_hint="prefill_moe_combine_finalize",
            deps=[return_rows_tid],
        ) as finalize_tid:
            token0 = pl.tile.get_block_idx() * PREFILL_MOE_FINALIZE_TOKEN_TILE
            for lane in pl.range(PREFILL_MOE_FINALIZE_TOKEN_TILE):
                token = token0 + lane
                if token < active_tokens:
                    acc = pl.cast(shared_y[token : token + 1, :], pl.FP32)
                    for topk in pl.range(TOPK):
                        route = token * TOPK + topk
                        packed_row = pl.cast(pl.read(route_to_packed, [route, 0]), pl.INDEX)
                        route_y = pl.cast(returned_y[packed_row : packed_row + 1, 0:D], pl.FP32)
                        acc = pl.add(acc, route_y)
                    ffn_out[token : token + 1, :] = pl.cast(acc, pl.BF16, mode="rint")
                else:
                    ffn_out[token : token + 1, :] = shared_y[token : token + 1, :]

        return finalize_tid


    @pl.jit.inline
    def _prefill_moe_pack_grouped_experts(
        dense_x: pl.Tensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
        dense_scale: pl.Tensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
        grouped_x: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, D], pl.INT8],
        grouped_scale: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        dispatch_tid: pl.Scalar[pl.TASK_ID],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Pad each dense expert slab to a private 16-row compute boundary."""
        with pl.spmd(
            N_LOCAL * GROUPED_PACK_BLOCKS_PER_EXPERT,
            name_hint="prefill_moe_grouped_pack",
            deps=[dispatch_tid],
            allow_early_resolve=False,
        ) as pack_tid:
            pack_block = pl.tile.get_block_idx()
            local_e = pack_block // GROUPED_PACK_BLOCKS_PER_EXPERT
            row_group = pack_block % GROUPED_PACK_BLOCKS_PER_EXPERT
            dense_base = pl.cast(0, pl.INDEX)
            grouped_base = pl.cast(0, pl.INDEX)
            for prior_e in pl.range(local_e):
                prior_rows = pl.cast(pl.read(expert_counts, [prior_e, 0]), pl.INDEX)
                dense_base = dense_base + prior_rows
                grouped_base = grouped_base + (
                    (prior_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1)
                    // PREFILL_MOE_GROUPED_EXPERT_TILE
                ) * PREFILL_MOE_GROUPED_EXPERT_TILE

            n_rows = pl.cast(pl.read(expert_counts, [local_e, 0]), pl.INDEX)
            grouped_rows = (
                (n_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1)
                // PREFILL_MOE_GROUPED_EXPERT_TILE
            ) * PREFILL_MOE_GROUPED_EXPERT_TILE
            zero_x = pl.cast(pl.full([1, D], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc")
            zero_scale = pl.full([1, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32, value=0.0)
            full_rows = (n_rows // PREFILL_MOE_GROUPED_EXPERT_TILE) * PREFILL_MOE_GROUPED_EXPERT_TILE
            for row in pl.range(
                row_group * PREFILL_MOE_GROUPED_EXPERT_TILE,
                full_rows,
                GROUPED_PACK_BLOCKS_PER_EXPERT * PREFILL_MOE_GROUPED_EXPERT_TILE,
            ):
                grouped_row = grouped_base + row
                dense_row = dense_base + row
                grouped_x[grouped_row : grouped_row + PREFILL_MOE_GROUPED_EXPERT_TILE, :] = (
                    dense_x[dense_row : dense_row + PREFILL_MOE_GROUPED_EXPERT_TILE, :]
                )
                grouped_scale[grouped_row : grouped_row + PREFILL_MOE_GROUPED_EXPERT_TILE, :] = (
                    dense_scale[dense_row : dense_row + PREFILL_MOE_GROUPED_EXPERT_TILE, :]
                )
            for row in pl.range(full_rows + row_group, grouped_rows, GROUPED_PACK_BLOCKS_PER_EXPERT):
                grouped_row = grouped_base + row
                if row < n_rows:
                    dense_row = dense_base + row
                    grouped_x[grouped_row : grouped_row + 1, :] = dense_x[dense_row : dense_row + 1, :]
                    grouped_scale[grouped_row : grouped_row + 1, :] = dense_scale[dense_row : dense_row + 1, :]
                else:
                    grouped_x[grouped_row : grouped_row + 1, :] = zero_x
                    grouped_scale[grouped_row : grouped_row + 1, :] = zero_scale
        return pack_tid


    @pl.jit.inline(auto_scope=False)
    def prefill_moe(
        x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
        hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[3], pl.FP32],
        hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
        norm_w: pl.Tensor[[D], pl.BF16],
        gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
        gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
        tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
        input_ids: pl.Tensor[[T], pl.INT64],
        routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
        routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
        routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
        routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
        shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
        shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
        shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
        shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
        shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
        shared_w2_scale: pl.Tensor[[D], pl.FP32],
        x_next: pl.Tensor[[T, HC_MULT, D], pl.FP32],
        # Caller-owned, layer-reused workspaces.  The multi-layer forward keeps
        # these resident, matching the DSpark prefill ownership contract instead
        # of allocating capacity-sized tensors in every per-layer scope.
        x_mixed: pl.InOut[pl.Tensor[[T, D], pl.BF16]],
        post_ffn: pl.InOut[pl.Tensor[[T, HC_MULT], pl.FP32]],
        comb_ffn: pl.InOut[pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32]],
        ffn_out: pl.InOut[pl.Tensor[[T, D], pl.BF16]],
        dense_x: pl.InOut[pl.Tensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8]],
        dense_scale: pl.InOut[pl.Tensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32]],
        returned_y: pl.InOut[pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, D], pl.BF16]],
        count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
        count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
        x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
        x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
        scale_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
        reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
        reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
        prior_dep: pl.Scalar[pl.TASK_ID],
        layer_id: pl.Scalar[pl.INT32],
        # 1-based MoE call id within this forward, like the decode path's moe_epoch:
        # the reverse exchange waits for `>= moe_epoch` credits on a window shared
        # by every call, so it counts calls, not layers.
        moe_epoch: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Run one production prefill MoE slab.

        Ordinary and CP callers select their token/window capacity explicitly,
        without changing decode or another prefill caller through import order.
        The CP caller packs its two real segment prefixes into this capacity.
        Empty ranks retain the same transport; HC and shared-expert stages still
        execute at the configured capacity.

        This path implements count-driven A2Av transport and 16-row-aligned
        grouped routed experts with ordinary MTP quantization semantics. One
        token-wise INT8 input/scale is shared across the six routes; each FP32
        routing weight is applied during W2 dequantization before the expert
        output is rounded to BF16. Source combine sums those weighted BF16 rows.
        """
        hc_pre(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed, post_ffn, comb_ffn)

        x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
        x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
        indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
        weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
        if num_tokens > 0:
            gate(
                x_mixed,
                norm_w,
                gate_w,
                gate_bias,
                layer_id,
                num_tokens,
                tid2eid,
                input_ids,
                x_norm_i8,
                x_norm_scale,
                indices,
                weights,
            )
        else:
            for block in pl.spmd(T // PREFILL_MOE_GATE_ZERO_TILE, name_hint="prefill_moe_empty_gate"):
                zero_x = pl.cast(pl.full([1, D], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc")
                for lane in pl.range(PREFILL_MOE_GATE_ZERO_TILE):
                    token = block * PREFILL_MOE_GATE_ZERO_TILE + lane
                    x_norm_i8[token : token + 1, 0:D] = zero_x
                    pl.write(x_norm_scale, [token, 0], pl.cast(0.0, pl.FP32))
                    for topk in pl.range(TOPK):
                        pl.write(indices, [token, topk], pl.cast(0, pl.INT32))
                        pl.write(weights, [token, topk], pl.cast(0.0, pl.FP32))

        shared_y = pl.create_tensor([T, D], dtype=pl.BF16)
        expert_shared(
            x_norm_i8,
            x_norm_scale,
            shared_w1,
            shared_w1_scale,
            shared_w3,
            shared_w3_scale,
            shared_w2,
            shared_w2_scale,
            shared_y,
        )

        expert_counts = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
        recv_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
        send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32, manual_dep=True)
        route_to_packed = pl.create_tensor(
            [PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD],
            dtype=pl.INT32,
            manual_dep=True,
        )
        dispatch_completion = pl.create_tensor([1, 8], dtype=pl.INT32)
        with pl.scope():
            dispatch_inner_tid = prefill_moe_dispatch(
                indices,
                x_norm_i8,
                x_norm_scale,
                weights,
                dense_x,
                dense_scale,
                expert_counts,
                recv_expert_counts,
                send_counts,
                route_to_packed,
                count_target,
                count_signal,
                x_target,
                x_signal,
                scale_target,
                num_tokens,
                prior_dep,
                moe_epoch,
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_dispatch_completion",
                deps=[dispatch_inner_tid],
                allow_early_resolve=False,
            ):
                dispatch_completion[0:1, :] = pl.full([1, 8], dtype=pl.INT32, value=1)
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_dispatch_ready",
            allow_early_resolve=False,
        ) as dispatch_tid:
            _completed = pl.read(dispatch_completion, [0, 0])


        with pl.scope():
            grouped_x = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, D], dtype=pl.INT8)
            grouped_scale = pl.create_tensor(
                [PREFILL_MOE_GROUPED_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32
            )
            grouped_y = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, D], dtype=pl.BF16)
            _prefill_moe_pack_grouped_experts(
                dense_x,
                dense_scale,
                expert_counts,
                grouped_x,
                grouped_scale,
                dispatch_tid,
            )

            grouped_expert_tid = prefill_expert_grouped(
                grouped_x,
                grouped_scale,
                expert_counts,
                routed_w1,
                routed_w1_scale,
                routed_w3,
                routed_w3_scale,
                routed_w2,
                routed_w2_scale,
                grouped_y,
            )

            # Reverse exchange consumes the live rows of the aligned expert output.
            with pl.scope():
                prefill_moe_combine(
                    grouped_y, expert_counts, recv_expert_counts,
                    route_to_packed, send_counts,
                    shared_y, ffn_out, returned_y,
                    reverse_target, reverse_signal,
                    num_tokens, dispatch_tid, grouped_expert_tid, moe_epoch,
                )

        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="prefill_moe_grouped_complete",
            allow_early_resolve=False,
        ) as completion_tid:
            # The RAW edge on x_next joins the HC-post SPMD before exposing a
            # reusable completion token to the enclosing production forward.
            _completion_anchor = pl.read(x_next, [0, 0, 0])
        return completion_tid


    return prefill_moe


# === Standalone test =========================================================
MOE_EPOCH = 1
PREFILL_MOE_LAYOUT = PrefillMoELayout(T)
PREFILL_MOE_ROUTES_PER_SRC = PREFILL_MOE_LAYOUT.routes_per_source
PREFILL_MOE_TOTAL_CAP = PREFILL_MOE_LAYOUT.total_capacity
prefill_moe = make_prefill_moe(PREFILL_MOE_LAYOUT)


@pl.jit(auto_scope=False)
def prefill_moe_test(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # final output
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    # windows
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scale_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    dense_x = pl.create_tensor([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.INT8)
    dense_scale = pl.create_tensor([PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32)
    returned_y = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, D], dtype=pl.BF16)
    completion = pl.create_tensor([1, 1, 8], dtype=pl.FP32)
    with pl.scope():
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_input_ready", allow_early_resolve=False) as input_ready:
            _input_ready = pl.read(x_hc, [0, 0, 0])
        moe_tid = prefill_moe(
            x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale, x_next,
            x_mixed, post_ffn, comb_ffn, ffn_out,
            dense_x, dense_scale,
            returned_y,
            count_target, count_signal, x_target, x_signal,
            scale_target, reverse_target, reverse_signal,
            input_ready, layer_id,
            pl.cast(MOE_EPOCH, pl.INT32), num_tokens,
        )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_complete",
                   deps=[moe_tid], allow_early_resolve=False):
            completion[0:1, 0:1, 0:8] = pl.slice(x_next, [1, 1, 8], [0, 0, 0])
    clear_prefill_moe_signals(completion, count_signal, x_signal, reverse_signal)
    return x_next


@pl.jit.host
def l3_prefill_moe(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    count_target_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    count_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    x_target_buf = pld.alloc_window_buffer([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.INT8)
    x_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    scale_target_buf = pld.alloc_window_buffer([PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], dtype=pl.FP32)
    reverse_target_buf = pld.alloc_window_buffer([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.BF16)
    reverse_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        count_target = pld.window(count_target_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        count_signal = pld.window(count_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        x_target = pld.window(x_target_buf, [PREFILL_MOE_TOTAL_CAP, D], dtype=pl.INT8)
        x_signal = pld.window(x_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        scale_target = pld.window(scale_target_buf, [PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], dtype=pl.FP32)
        reverse_target = pld.window(reverse_target_buf, [PREFILL_MOE_TOTAL_CAP, D], dtype=pl.BF16)
        reverse_signal = pld.window(reverse_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        prefill_moe_test(
            x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            x_next[r],
            count_target, count_signal, x_target, x_signal, scale_target,
            reverse_target, reverse_signal,
            layer_id, num_tokens,
            device=r,
        )


# === Golden + test ==========================================================
def golden_prefill_moe(tensors):
    """Per-rank torch reference: hc_pre, gate, and the shared expert run on each
    source rank; every live route runs its routed expert on the owning rank; the
    top-k rows are summed back onto the source token before hc_post.

    Each routed row's SwiGLU output depends only on that row, so the result is
    invariant to the device's expert-major packing. Rows are packed per
    destination expert in source-major, token/top-k order."""
    import torch

    from hc_pre import golden_hc_pre
    from hc_post import golden_hc_post
    from gate import golden_gate_core
    from expert_shared import golden_expert_shared
    from expert_routed import golden_expert_routed

    T = tensors["x_hc"].shape[1]
    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # hc_pre + gate + shared expert on every source rank.
    ranks = []
    for r in range(N_RANKS):
        x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
        post = torch.zeros(T, HC_MULT, dtype=torch.float32)
        comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
        golden_hc_pre({
            "x":        tensors["x_hc"][r],
            "hc_fn":    tensors["hc_ffn_fn"][r],
            "hc_scale": tensors["hc_ffn_scale"][r],
            "hc_base":  tensors["hc_ffn_base"][r],
            "x_mixed":  x_mixed,
            "post":     post,
            "comb":     comb,
        })
        x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
        x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
        indices = torch.zeros(T, TOPK, dtype=torch.int32)
        weights = torch.zeros(T, TOPK, dtype=torch.float32)
        golden_gate_core({
            "x_mixed":      x_mixed,
            "norm_w":       tensors["norm_w"][r],
            "gate_w":       tensors["gate_w"][r],
            "gate_bias":    tensors["gate_bias"][r],
            "layer_id":     tensors["layer_id"],
            "num_tokens":   tensors["num_tokens"],
            "tid2eid":      tensors["tid2eid"][r],
            "input_ids":    tensors["input_ids"][r],
            "x_norm_i8":    x_norm_i8,
            "x_norm_scale": x_norm_scale,
            "indices":      indices,
            "weights":      weights,
        })
        sh = torch.zeros(T, D, dtype=torch.bfloat16)
        golden_expert_shared({
            "x_local_i8":       x_norm_i8,
            "x_local_scale_dq": x_norm_scale,
            "num_tokens":       tensors["num_tokens"],
            "shared_w1":        tensors["shared_w1"][r],
            "shared_w1_scale":  tensors["shared_w1_scale"][r],
            "shared_w3":        tensors["shared_w3"][r],
            "shared_w3_scale":  tensors["shared_w3_scale"][r],
            "shared_w2":        tensors["shared_w2"][r],
            "shared_w2_scale":  tensors["shared_w2_scale"][r],
            "sh":               sh,
        })
        ranks.append((post, comb, x_norm_i8, x_norm_scale, indices, weights, sh))

    # lanes[dst][local_e] = live (src, token, topk) routes in source-major order.
    lanes = [[[] for _ in range(N_LOCAL)] for _ in range(N_RANKS)]
    for src in range(N_RANKS):
        indices = ranks[src][4]
        for t in range(num_tokens):
            for k in range(TOPK):
                eid = int(indices[t, k].item())
                lanes[eid // N_LOCAL][eid % N_LOCAL].append((src, t, k))

    routed_y = torch.zeros(N_RANKS, T, TOPK, D, dtype=torch.bfloat16)
    recv_rows = N_RANKS * T
    for dst in range(N_RANKS):
        recv_x = torch.zeros(N_LOCAL, recv_rows, D, dtype=torch.int8)
        recv_scale = torch.zeros(N_LOCAL, recv_rows, dtype=torch.float32)
        recv_w = torch.zeros(N_LOCAL, recv_rows, dtype=torch.float32)
        recv_count = torch.zeros(N_LOCAL, 1, dtype=torch.int32)
        for e, lane in enumerate(lanes[dst]):
            recv_count[e, 0] = len(lane)
            for slot, (src, t, k) in enumerate(lane):
                recv_x[e, slot, :] = ranks[src][2][t, :]
                recv_scale[e, slot] = float(ranks[src][3][t, 0].item())
                recv_w[e, slot] = float(ranks[src][5][t, k].item())
        recv_y = torch.zeros(N_LOCAL, recv_rows, D, dtype=torch.bfloat16)
        golden_expert_routed({
            "recv_x":            recv_x,
            "recv_scale_dq":     recv_scale,
            "recv_weights":      recv_w,
            "recv_expert_count": recv_count,
            "routed_w1":         tensors["routed_w1"][dst],
            "routed_w1_scale":   tensors["routed_w1_scale"][dst],
            "routed_w3":         tensors["routed_w3"][dst],
            "routed_w3_scale":   tensors["routed_w3_scale"][dst],
            "routed_w2":         tensors["routed_w2"][dst],
            "routed_w2_scale":   tensors["routed_w2_scale"][dst],
            "recv_y":            recv_y,
        })
        for e, lane in enumerate(lanes[dst]):
            for slot, (src, t, k) in enumerate(lane):
                routed_y[src, t, k, :] = recv_y[e, slot, :]

    # Combine: shared + top-k routed rows in FP32, rounded to BF16, then hc_post.
    x_next_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    for r in range(N_RANKS):
        post, comb, _, _, _, _, sh = ranks[r]
        acc = sh.float().clone()
        for k in range(TOPK):
            acc[:num_tokens, :] += routed_y[r, :num_tokens, k, :].float()
        x_next_r = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
        golden_hc_post({
            "x":        acc.to(torch.bfloat16),
            "residual": tensors["x_hc"][r],
            "post":     post,
            "comb":     comb,
            "y":        x_next_r,
        })
        x_next_out[r] = x_next_r

    tensors["x_next"][:] = x_next_out


def build_tensor_specs(layer_id=0, num_tokens=None, *, token_capacity=T):
    T = token_capacity
    if num_tokens is None:
        num_tokens = T
    import torch
    from golden import ScalarSpec, TensorSpec
    from expert_routed import gen_routed_weight
    from expert_shared import gen_shared_weight

    # Routed = MXFP4 (gen_routed_weight), shared = MXFP8 (gen_shared_weight). This
    # is an integration test whose x_next-equivalent output is dominated by near-zero
    # residual+FFN cancellations, so it keeps the smaller *behaviorally-calibrated* magnitude
    # (random fixtures blow up the relative metric at the real ~2.5e-2 magnitude); only the
    # grid SHAPE (FP4/FP8 discreteness, scale CV) matches the real distribution.
    ROUTED_DEQUANT_STD = {"w1": 1.08e-2, "w2": 2.54e-2, "w3": 1.10e-2}
    SHARED_DEQUANT_STD = {"w1": 7.65e-3, "w2": 2.39e-2, "w3": 7.39e-3}

    # Shared (replicated) weights are broadcast across ranks; the routed
    # weights are per-rank shards.
    def init_x_hc():
        return torch.randn(N_RANKS, T, HC_MULT, D)

    # Real layer-0 hc_ffn scale/base (fn synthetic at real magnitude). A synthetic
    # scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling the FFN output and
    # hc residual to near-zero in x_next where W8A8 noise blows up the relative tail.
    def init_hc_ffn_fn():
        x = torch.randn(MIX_HC, HC_DIM) * 0.0635
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_hc_ffn_scale():
        x = torch.tensor([0.11334, 0.035901, 0.058183])
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_hc_ffn_base():
        x = torch.tensor([
            2.4153, -2.0252, -2.0019, -2.1947,
            -1.5430, -3.0228, -6.8248, 0.5894,
            2.1916, -7.2132, -3.0938, -2.1119,
            -3.0161, 3.3293, -3.2224, -4.0226,
            -2.0428, -3.3478, 3.0893, -3.4166,
            -1.8144, -3.8147, -3.1307, 1.7862,
        ])
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_norm_w():
        x = torch.ones(D)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_gate_w():
        x = torch.randn(N_EXPERTS_GLOBAL, D) / D ** 0.5
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_gate_bias():
        x = torch.zeros(N_EXPERTS_GLOBAL)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_tid2eid():
        # Distinct experts per token (sample without replacement) like real top-k,
        # so the route-keyed distributed combine stays unambiguous.
        x = torch.argsort(torch.rand(VOCAB, N_EXPERTS_GLOBAL), dim=1)[:, :TOPK].to(torch.int32)
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_input_ids():
        # Distinct per-rank token streams.
        return torch.randint(0, VOCAB, (N_RANKS, T), dtype=torch.int64)

    # Per-rank routed expert weights (different shards).
    routed_w1_i8_list = []
    routed_w1_s_list = []
    routed_w3_i8_list = []
    routed_w3_s_list = []
    routed_w2_i8_list = []
    routed_w2_s_list = []
    for _ in range(N_RANKS):
        w1_i8, w1_s = gen_routed_weight((N_LOCAL, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
        w3_i8, w3_s = gen_routed_weight((N_LOCAL, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
        w2_i8, w2_s = gen_routed_weight((N_LOCAL, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])
        routed_w1_i8_list.append(w1_i8)
        routed_w1_s_list.append(w1_s)
        routed_w3_i8_list.append(w3_i8)
        routed_w3_s_list.append(w3_s)
        routed_w2_i8_list.append(w2_i8)
        routed_w2_s_list.append(w2_s)

    rw1_i8 = torch.stack(routed_w1_i8_list)
    rw1_s = torch.stack(routed_w1_s_list)
    rw3_i8 = torch.stack(routed_w3_i8_list)
    rw3_s = torch.stack(routed_w3_s_list)
    rw2_i8 = torch.stack(routed_w2_i8_list)
    rw2_s = torch.stack(routed_w2_s_list)

    # Shared expert weights — replicated across ranks.
    sw1_i8, sw1_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3_i8, sw3_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2_i8, sw2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)
    sw1_i8 = sw1_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw1_s = sw1_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    sw3_i8 = sw3_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw3_s = sw3_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    sw2_i8 = sw2_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw2_s = sw2_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    specs = [
        TensorSpec("x_hc",          [N_RANKS, T, HC_MULT, D],     torch.float32, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",     [N_RANKS, MIX_HC, HC_DIM],       torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale",  [N_RANKS, 3],                    torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",   [N_RANKS, MIX_HC],               torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",        [N_RANKS, D],                    torch.bfloat16,  init_value=init_norm_w),
        TensorSpec("gate_w",        [N_RANKS, N_EXPERTS_GLOBAL, D],  torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",     [N_RANKS, N_EXPERTS_GLOBAL],     torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",       [N_RANKS, VOCAB, TOPK],          torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids",     [N_RANKS, T],                 torch.int64,    init_value=init_input_ids),
        TensorSpec("routed_w1",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: rw1_i8),
        TensorSpec("routed_w1_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: rw1_s),
        TensorSpec("routed_w3",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: rw3_i8),
        TensorSpec("routed_w3_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: rw3_s),
        TensorSpec("routed_w2",        [N_RANKS, N_LOCAL, D, MOE_INTER], torch.int8,    init_value=lambda: rw2_i8),
        TensorSpec("routed_w2_scale",  [N_RANKS, N_LOCAL, D],            torch.float32, init_value=lambda: rw2_s),
        TensorSpec("shared_w1",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2",        [N_RANKS, D, MOE_INTER],          torch.int8,    init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale",  [N_RANKS, D],                     torch.float32, init_value=lambda: sw2_s),
        TensorSpec("x_next",           [N_RANKS, T, HC_MULT, D],      torch.float32),
        ScalarSpec("layer_id",         torch.int32,                      layer_id),
        ScalarSpec("num_tokens",       torch.int32,                      num_tokens),
    ]

    # Keep the static weight parameters device-resident (child_memory), sharded
    # per rank: each shard is a leading-dim-stacked [N_RANKS, *tail] tensor sliced
    # as weight[r] and dispatched to device=r; resident="stacked" uploads shard r
    # to card r once and reuses it across dispatches, skipping the per-dispatch
    # H2D/D2H. Covers the routed/shared expert weights and their scales, the gate,
    # the HC-FFN constants, the RMSNorm gamma, and the static tid2eid route table —
    # but NOT the per-step activation (x_hc), per-step input_ids, or the output.
    # All resident names are pure inputs, so the flag is always valid.
    RESIDENT_WEIGHT_NAMES = frozenset([
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
    ])
    for spec in specs:
        if spec.name in RESIDENT_WEIGHT_NAMES:
            spec.resident = "stacked"

    return specs


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=_EP_DEFAULT, choices=list(_EP_CHOICES),
                        help="EP world size / rank count")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=T,
                        help=f"per-rank token capacity, read at import time (default {T})")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help=f"active token count for MoE dispatch/combine (0..{T})")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--golden-only", action="store_true", default=False,
                        help="compute and persist the golden, then stop before the device run")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None,
                        help="dir with cached in/{name}.pt + out/{name}.pt; reuses them "
                             "instead of regenerating inputs + recomputing golden.")
    parser.add_argument("--log-level", type=str, default=None,
                        help="runtime log threshold: debug, v0..v9, info, warn, error, null")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    result = run(
        fn=l3_prefill_moe,
        specs=build_tensor_specs(
            layer_id=args.layer_id,
            num_tokens=args.num_tokens,
            token_capacity=T,
        ),
        golden_fn=golden_prefill_moe,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        golden_only=args.golden_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            log_level=args.log_level,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
