# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill HC head, vocabulary projection and sampling."""

import pypto.language as pl
from models.deepseek_v4_flash_lowlat.rmsnorm_common import rms_norm
from models.deepseek_v4_flash_lowlat.config import FLASH as M
from prefill_shapes import DECODE_BATCH, DECODE_SEQ, TP
from math import gcd
import sys
from projection_exchange import PROJECTION_WEIGHT_LAYOUT
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from models.deepseek_v4_flash_lowlat.config import FP32_NEG_INF
from prefill_shapes import MOE_TOKENS


# ruff: noqa: F401,F403,F405,F821




# Longest sequence the model config admits; the host-side bound on token_count.
MAX_SEQ_LEN = M.max_position_embeddings

# Dynamic shape variables.
HC_T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
EPS = M.rms_norm_eps
HC_EPS = M.hc_eps
HC_DIM_INV = 1.0 / HC_DIM

# tiling
HC_PAD = 16
HC_VPAD = 8
T_TILE = 16
LINEAR_T_TILE = 16
RMS_T_TILE = 64
RMS_K_TILE = 256
LINEAR_K_TILE = 256
D_TILE = 256
# The vector array is 48 cores wide. RMS_T_TILE and D_SPMD are sized so the rms
# and reduce fan-outs land at 48 tasks for a decode-forward tail (3 target layers
# x MOE_TOKENS rows) instead of queueing four waves of short ones.
D_SPMD = 2048
# Each sq_part column spans the 32 B reduction floor.
RMS_OK = 8
# Rows every stage pads its row count to, so the taller rms tile still lands
# inside the shared sq_part / mixes_raw slabs.
ROW_ALIGN = max(T_TILE, LINEAR_T_TILE, RMS_T_TILE)


def _hc_head(
    x_hc: pl.Tensor[[HC_T_DYN, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[HC_T_DYN, D], pl.BF16]],
):
    x_hc.bind_dynamic(0, HC_T_DYN)
    y.bind_dynamic(0, HC_T_DYN)
    t_dim = pl.tensor.dim(x_hc, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    rms_tiles = (t_dim + RMS_T_TILE - 1) // RMS_T_TILE
    linear_full_rows = (t_dim // LINEAR_T_TILE) * LINEAR_T_TILE
    t_linear = ((t_dim + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, D])
    # rms: split-K sum-of-squares, fanned over (rms-tile x K-slice). RMS_T_TILE is
    # taller than the gate's T_TILE so the fan-out stays inside the vector array
    # instead of queueing several waves of short tasks.
    sq_part = pl.create_tensor([RMS_OK, t_linear], dtype=pl.FP32)
    for task in pl.spmd(rms_tiles * RMS_OK, name_hint="hc_head_rms"):
        t0 = (task // RMS_OK) * RMS_T_TILE
        ok = task % RMS_OK
        k_base = ok * (HC_DIM // RMS_OK)
        valid_rows = pl.min(RMS_T_TILE, t_dim - t0)
        sq_sum = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_OK // RMS_K_TILE, stage=1):
            k0 = k_base + kb * RMS_K_TILE
            if valid_rows == RMS_T_TILE:
                x_rms_full = x_flat[t0 : t0 + RMS_T_TILE, k0 : k0 + RMS_K_TILE]
                sq_col_full = pl.row_sum(pl.mul(x_rms_full, x_rms_full))
                sq_sum = pl.add(sq_sum, pl.reshape(sq_col_full, [1, RMS_T_TILE]))
            else:
                x_rms_tail = pl.slice(
                    x_flat,
                    [RMS_T_TILE, RMS_K_TILE],
                    [t0, k0],
                    valid_shape=[valid_rows, RMS_K_TILE],
                )
                sq_col_tail = pl.row_sum(pl.mul(x_rms_tail, x_rms_tail))
                sq_sum = pl.add(sq_sum, pl.reshape(sq_col_tail, [1, RMS_T_TILE]))
        sq_part[ok : ok + 1, t0 : t0 + RMS_T_TILE] = sq_sum

    # One writer per row block computes the complete head projection.
    mixes_raw = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    if linear_full_rows > 0:
        with pl.spmd(
            linear_full_rows // LINEAR_T_TILE,
            name_hint="hc_head_linear",
        ) as _linear_tid:
            # Weight reads bypass L2.
            pl.set_cache_policy(hc_head_fn, pl.CachePolicy.BYPASS)
            task = pl.tile.get_block_idx()
            t0 = task * LINEAR_T_TILE
            acc_full = pl.create_tensor([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
                k0 = kb * LINEAR_K_TILE
                x_lin_full = x_flat[t0 : t0 + LINEAR_T_TILE, k0 : k0 + LINEAR_K_TILE]
                w_full = pl.slice(
                    hc_head_fn,
                    [HC_PAD, LINEAR_K_TILE],
                    [0, k0],
                    valid_shape=[HC_MULT, LINEAR_K_TILE],
                )
                acc_full = pl.matmul_acc(acc_full, x_lin_full, w_full, b_trans=True, init_cond=(kb == 0))
            mixes_raw[t0 : t0 + LINEAR_T_TILE, 0:HC_PAD] = acc_full

    # At most one incomplete M=16 block exists. Keep it in a separate conditional
    # task so every aligned block above retains the original static-M Cube path.
    if linear_full_rows < t_dim:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="hc_head_linear_tail",
        ):
            pl.set_cache_policy(hc_head_fn, pl.CachePolicy.BYPASS)
            tail_rows = t_dim - linear_full_rows
            acc_tail = pl.create_tensor([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
                k0 = kb * LINEAR_K_TILE
                x_lin_tail = pl.slice(
                    x_flat,
                    [LINEAR_T_TILE, LINEAR_K_TILE],
                    [linear_full_rows, k0],
                    valid_shape=[tail_rows, LINEAR_K_TILE],
                )
                w_tail = pl.slice(
                    hc_head_fn,
                    [HC_PAD, LINEAR_K_TILE],
                    [0, k0],
                    valid_shape=[HC_MULT, LINEAR_K_TILE],
                )
                acc_tail = pl.matmul_acc(acc_tail, x_lin_tail, w_tail, b_trans=True, init_cond=(kb == 0))
            mixes_raw[linear_full_rows : linear_full_rows + LINEAR_T_TILE, 0:HC_PAD] = acc_tail

    # reduce: gate + hc mix, fanned over (token-tile x D-slice). The rsqrt/sigmoid gate is
    # recomputed per task instead of being published by its own scope.
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="hc_head_reduce"):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        scale = pl.read(hc_head_scale, [0])
        base = pl.reshape(pl.slice(hc_head_base, [HC_VPAD], [0], valid_shape=[HC_MULT]), [1, HC_VPAD])
        # sq_part is [RMS_OK, T], so the per-token total is a column sum; transposing the
        # slab makes it one row_sum landing in the [T_TILE, 1] shape row_expand_mul wants
        ssq_slab = pl.set_validshape(sq_part[0:RMS_OK, t0 : t0 + T_TILE], RMS_OK, T_TILE)
        ssq = pl.row_sum(pl.transpose(ssq_slab, axis1=0, axis2=1))
        inv_row = pl.rsqrt(pl.add(pl.mul(ssq, HC_DIM_INV), EPS), high_precision=True)
        inv_col = pl.reshape(inv_row, [T_TILE, 1])
        mix = mixes_raw[t0 : t0 + T_TILE, 0:HC_VPAD]
        scaled = pl.mul(pl.row_expand_mul(mix, inv_col), scale)
        logits = pl.add(scaled, pl.col_expand(scaled, base))
        pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
        # pin the valid shape: the gate carries valid=?x? from the dynamic-offset slice
        pre_val = pl.set_validshape(pre_val, T_TILE, HC_VPAD)
        pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            if valid_rows == T_TILE:
                x_h0_full = x_flat[t0 : t0 + T_TILE, 0 * D + d0 : 0 * D + d0 + D_TILE]
                x_h1_full = x_flat[t0 : t0 + T_TILE, 1 * D + d0 : 1 * D + d0 + D_TILE]
                x_h2_full = x_flat[t0 : t0 + T_TILE, 2 * D + d0 : 2 * D + d0 + D_TILE]
                x_h3_full = x_flat[t0 : t0 + T_TILE, 3 * D + d0 : 3 * D + d0 + D_TILE]
                y01_full = pl.add(
                    pl.row_expand_mul(x_h0_full, pre0),
                    pl.row_expand_mul(x_h1_full, pre1),
                )
                y23_full = pl.add(
                    pl.row_expand_mul(x_h2_full, pre2),
                    pl.row_expand_mul(x_h3_full, pre3),
                )
                y_tile_full = pl.add(y01_full, y23_full)
                y_flat[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.cast(
                    y_tile_full,
                    target_type=pl.BF16,
                    mode="rint",
                )
            else:
                x_h0_tail = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, 0 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h1_tail = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, 1 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h2_tail = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, 2 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h3_tail = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, 3 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                y01_tail = pl.add(
                    pl.row_expand_mul(x_h0_tail, pre0),
                    pl.row_expand_mul(x_h1_tail, pre1),
                )
                y23_tail = pl.add(
                    pl.row_expand_mul(x_h2_tail, pre2),
                    pl.row_expand_mul(x_h3_tail, pre3),
                )
                y_tile_tail = pl.add(y01_tail, y23_tail)
                y_bf16_tail = pl.cast(y_tile_tail, target_type=pl.BF16, mode="rint")
                y_valid_tail = pl.set_validshape(y_bf16_tail, valid_rows, D_TILE)
                y_flat[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = y_valid_tail

    y = pl.reshape(y_flat, [t_dim, D])
    return y


hc_head = pl.jit.inline(_hc_head)
hc_head_test = pl.jit(_hc_head)


def golden_hc_head(tensors):
    import torch

    x = tensors["x_hc"]
    shape = x.shape
    token_count = x.shape[0]
    x_flat_2d = x.reshape(token_count, HC_DIM).float()
    hc_head_fn = tensors["hc_head_fn"].float()

    sq_sum = torch.zeros(token_count, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0 : k0 + RMS_K_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)

    mix_cols = []
    for h in range(HC_MULT):
        mix_col = torch.zeros(token_count, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_TILE):
            x_chunk = x_flat_2d[:, k0 : k0 + LINEAR_K_TILE]
            w_chunk = hc_head_fn[h : h + 1, k0 : k0 + LINEAR_K_TILE]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1).reshape(token_count, HC_MULT)

    pre = torch.sigmoid(mixes * tensors["hc_head_scale"].float() + tensors["hc_head_base"].float()) + HC_EPS
    x_view = x.float().view(shape)
    if HC_MULT == 4:
        y = (x_view[:, 0, :] * pre[:, 0:1] + x_view[:, 1, :] * pre[:, 1:2]) + (
            x_view[:, 2, :] * pre[:, 2:3] + x_view[:, 3, :] * pre[:, 3:4]
        )
    else:
        y = torch.zeros(token_count, D, dtype=torch.float32)
        for h in range(HC_MULT):
            y += x_view[:, h, :] * pre[:, h : h + 1]

    # Match the kernel's mode="rint" cast (round to nearest, ties to even).
    tensors["y"][:] = y.to(torch.bfloat16)


HC_VALIDATION_MAX_ROWS = 8192


def build_tensor_specs_hc(token_count: int = T):
    import torch
    from golden import TensorSpec

    if not 1 <= token_count <= HC_VALIDATION_MAX_ROWS:
        raise ValueError(f"token_count must be in [1, {HC_VALIDATION_MAX_ROWS}], got {token_count}")

    def init_x_hc():
        return torch.randn(token_count, HC_MULT, D) * 0.05

    def init_hc_head_fn():
        return torch.randn(HC_MULT, HC_DIM) * 0.0519

    return [
        TensorSpec("x_hc", [token_count, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("hc_head_fn", [HC_MULT, HC_DIM], torch.float32, init_value=init_hc_head_fn),
        TensorSpec("hc_head_scale", [1], torch.float32, init_value=lambda: torch.tensor([0.076099])),
        TensorSpec(
            "hc_head_base",
            [HC_MULT],
            torch.float32,
            init_value=lambda: torch.tensor([5.9166, -3.6223, -2.9324, -3.3124]),
        ),
        TensorSpec("y", [token_count, D], torch.bfloat16),
    ]


# ci: devices=2






LM_T_DYN = pl.dynamic("LM_HEAD_T_DYN")

# model config
VOCAB = M.vocab_size
# Rows one owner projects: the TP group splits the step across its ranks, so a
# rank carries exactly the tokens it owns through MoE and the group window holds
# exactly one step. Taken from config rather than derived from TP, which the
# forward entries rewrite at import before this module loads.
MAX_LOGIT_ROWS = MOE_TOKENS

# parallelism
_TP_CHOICES = (1, 2, 4, 8, 16)
_DP_CHOICES = (1, 2, 4, 8, 16)
_TP_DEFAULT = 2


def _parse_int_argv(name, default=None):
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default


# Vocabulary sharding and owner groups use an independent head TP.
TP_SIZE: int = _parse_int_argv("--lm-head-tp", _parse_int_argv("--tp", _TP_DEFAULT))
DP_SIZE: int = _parse_int_argv("--dp", 1)
# --dp describes the standalone head fixture; --ep is the composed model world.
WORLD_SIZE = _parse_int_argv("--ep", TP_SIZE * DP_SIZE)
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"LM-head TP must be one of {_TP_CHOICES}, got {TP_SIZE}")
if WORLD_SIZE <= 0 or WORLD_SIZE % TP_SIZE:
    raise ValueError(f"world size {WORLD_SIZE} must be a positive multiple of LM-head TP={TP_SIZE}")
if VOCAB % TP_SIZE:
    raise ValueError(f"vocab size {VOCAB} must be divisible by LM-head TP={TP_SIZE}")
VOCAB_PER_TP = VOCAB // TP_SIZE
GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS
TEST_TOKENS = 2 * MAX_LOGIT_ROWS  # standalone fixture: hidden rows per card

# tiling
# MM_ROW_TILE x FUSED_VOCAB_TILE fp32 keeps the accumulator inside the 128KiB
# Acc space and divides MAX_LOGIT_ROWS: every row block is one owner's rows.
# One whole owner per block is the widest M the Acc space admits, and it halves
# the weight tile re-reads a narrower row tile forces.
FUSED_K_TILE = 256
FUSED_VOCAB_TILE = 128
MM_ROW_TILE = gcd(MAX_LOGIT_ROWS, 128)
HIDDEN_GATHER_TILE = 1024
HIDDEN_GATHER_ROW_TILE = gcd(GROUP_LOGIT_ROWS, min(GROUP_LOGIT_ROWS, 64))
PUSH_ROW_TILE = gcd(MAX_LOGIT_ROWS, 16)  # rows pushed per dispatch_push block (fewer, larger ring puts)
LOGITS_GATHER_ROW_TILE = gcd(MAX_LOGIT_ROWS, 8)
LOGITS_COMM_TILE = 2048
# Greedy sampling scans each row as a [GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH] grid.
# 8 rows keeps a reduction result at 32 B; alloc_tile rejects the 4 B a [1, 1] gives.
GREEDY_ROW_WIDTH = 808
GREEDY_BLOCK_ROWS = 8
SAMPLED_IDS_PAD = 8
FUSED_LM_HEAD_CORES = 24
# AIV lanes per AICore: the fused kernel splits each row block's accumulator
# across the two lanes, and each lane pushes a contiguous half of one owner's rows.
AIV_LANES = 2
LANE_ROWS = (MM_ROW_TILE + AIV_LANES - 1) // AIV_LANES
VOCAB_TAIL = VOCAB_PER_TP % FUSED_VOCAB_TILE
VOCAB_FULL_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE
LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE
N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE
LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))
LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS
GREEDY_GRID_ROWS = VOCAB // GREEDY_ROW_WIDTH
GREEDY_BLOCK_SPAN = GREEDY_BLOCK_ROWS * GREEDY_ROW_WIDTH
GRAMMAR_LOOKUP_WIDTH = 816  # 808 logits plus the 32-byte int16 tile alignment.
GRAMMAR_SEGMENT_WORDS = 64  # 808 bits plus aligned padding per scan lane.
# 2^30: above every vocab id, clear of int32 overflow once a block base is added.
GREEDY_INDEX_SENTINEL = 1073741824
DONE_VALUE = 1
LM_HEAD_RING_HEAP = 1 << 30

assert MAX_LOGIT_ROWS % MM_ROW_TILE == 0, "each row block must be one owner's rows"
assert MAX_LOGIT_ROWS % PUSH_ROW_TILE == 0, "row block must cover whole rows"
assert MAX_LOGIT_ROWS % LOGITS_GATHER_ROW_TILE == 0, "logits row blocks must cover whole rows"
# TP1 notifies nobody, so its empty peer loop needs no lane pairing.
assert TP_SIZE == 1 or TP_SIZE % AIV_LANES == 0, "peer owners must pair across the AIV lanes"
assert GREEDY_BLOCK_ROWS * 4 % 32 == 0, "reduction result must clear the 32 B column floor"
assert GREEDY_ROW_WIDTH * 4 % 32 == 0, "block row must clear the 32 B row floor"
assert VOCAB < GREEDY_INDEX_SENTINEL, "sentinel must lose every row_min against a real id"


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, PROJECTION_WEIGHT_LAYOUT],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    hidden_ready_tid: pl.Scalar[pl.TASK_ID],
) -> tuple[
    pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    pl.Scalar[pl.TASK_ID],
]:
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)

    # Publish this card's logit rows into every group member's window slot: the
    # window holds one slot per group member and each card writes only its own,
    # `tp_rank * MAX_LOGIT_ROWS`. One block per logit row, one [1, D] put per peer.
    # One block per PUSH_ROW_TILE rows: PUSH_ROW_TILE-row ring puts instead
    # of one [1, D] put per row. This cuts the per-dispatch put and notify
    # counts (512 -> 32 per card) to the MTP fixture's scale; the earlier
    # per-row form stalled lm_head_dispatch_gather on persistent re-dispatch
    # (SCHEDULER_TIMEOUT S1 with the gather task's core never exiting).
    with pl.spmd(
        MAX_LOGIT_ROWS // PUSH_ROW_TILE,
        name_hint="lm_head_dispatch_push",
        deps=[hidden_ready_tid],
    ) as _dispatch_push_tid:
        blk = pl.tile.get_block_idx()
        r0 = blk * PUSH_ROW_TILE
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        for i in pl.range(PUSH_ROW_TILE):
            row = r0 + i
            source_row_raw = pl.read(logit_row_indices, [row])
            # Clamp so the load address is always inside hidden_states even if a
            # caller hands over a stale index; the -1 guard below decides whether
            # the row is actually used.
            safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
            selected_hidden[row : row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
            if source_row_raw >= 0:
                source_row = pl.cast(safe_raw, target_type=pl.INDEX)
                selected_hidden[row : row + 1, :] = hidden_states[source_row : source_row + 1, :]

        # Self-target rides the same put; put drains before the notify issues.
        for peer_tp in pl.range(TP_SIZE):
            pld.tensor.put(
                dst=hidden_window,
                peer=group_base + peer_tp,
                src=selected_hidden,
                dst_offsets=[tp_rank * MAX_LOGIT_ROWS + r0, 0],
                src_offsets=[r0, 0],
                shape=[PUSH_ROW_TILE, D],
            )

        # Notify folded into the push: one notify per block per source per epoch.
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=hidden_done,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # Start the wait after this rank has published. Signal credits persist until
    # the final clear, so peer notifies may safely arrive before this task starts;
    # anchoring avoids an idle wait occupying a core group while the push is pending.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", deps=[_dispatch_push_tid]
    ) as _dwait_tid:
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.wait(
                    signal=hidden_done,
                    offsets=[owner_tp, 0],
                    expected=pl.cast(done_epoch * (MAX_LOGIT_ROWS // PUSH_ROW_TILE), pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Window -> matmul operand: a local copy split over k-tiles. Keeps the matmul's
    # auto-dep on owner_hiddens.
    owner_hiddens = pl.create_tensor([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    with pl.spmd(
        (GROUP_LOGIT_ROWS // HIDDEN_GATHER_ROW_TILE) * (D // HIDDEN_GATHER_TILE),
        name_hint="lm_head_dispatch_gather",
        deps=[_dwait_tid, _dispatch_push_tid],
    ) as _dgather_tid:
        gblock = pl.tile.get_block_idx()
        gk0 = (gblock % (D // HIDDEN_GATHER_TILE)) * HIDDEN_GATHER_TILE
        gr0 = (gblock // (D // HIDDEN_GATHER_TILE)) * HIDDEN_GATHER_ROW_TILE
        owner_hiddens[gr0 : gr0 + HIDDEN_GATHER_ROW_TILE, gk0 : gk0 + HIDDEN_GATHER_TILE] = hidden_window[
            gr0 : gr0 + HIDDEN_GATHER_ROW_TILE, gk0 : gk0 + HIDDEN_GATHER_TILE
        ]

    # Fused cube+comm kernel: matmul one [MM_ROW_TILE, FUSED_VOCAB_TILE] tile,
    # carry the accumulator across the C->V edge with pl.aiv_shard, and
    # remote_store each lane's row half to its owner's window while the cube
    # projects the next tile. No GM staging: every row block is one owner's rows,
    # so each lane's shard is already the exact rows it pushes.
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul_push",
        deps=[_dgather_tid],
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as _push_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(lm_head_weight, pl.CachePolicy.BYPASS)
        lm_core = pl.tile.get_block_idx()
        vocab_base = tp_rank * VOCAB_PER_TP
        for mm_ob in pl.range(lm_core, VOCAB_FULL_TILES, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            for mm_rb in pl.range(GROUP_LOGIT_ROWS // MM_ROW_TILE):
                mm_r0 = mm_rb * MM_ROW_TILE
                mm_acc = pl.create_tensor([MM_ROW_TILE, FUSED_VOCAB_TILE], dtype=pl.FP32)
                for mm_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                    mm_k0 = mm_kb * FUSED_K_TILE
                    mm_hidden_tile = owner_hiddens[mm_r0 : mm_r0 + MM_ROW_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                    mm_weight_tile = lm_head_weight[mm_o0 : mm_o0 + FUSED_VOCAB_TILE, mm_k0 : mm_k0 + FUSED_K_TILE]
                    mm_acc = pl.matmul_acc(mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True, init_cond=(mm_kb == 0))

                # The block is one owner's rows; the two lanes push its two
                # contiguous halves straight to that owner's window.
                for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                    mm_shard = pl.aiv_shard(mm_acc)
                    owner_tp = mm_rb // (MAX_LOGIT_ROWS // MM_ROW_TILE)
                    owner_r0 = (mm_rb % (MAX_LOGIT_ROWS // MM_ROW_TILE)) * MM_ROW_TILE + aiv_id * LANE_ROWS
                    if aiv_id * LANE_ROWS < MM_ROW_TILE:
                        pld.tensor.remote_store(
                            mm_shard, logits_window, group_base + owner_tp,
                            [owner_r0, vocab_base + mm_o0],
                        )

        # Ragged tail: its own matmul on one block; the accumulator is already
        # VOCAB_TAIL wide. Hoisted out of the strided loop: a dynamic branch would
        # give the accumulator two static shapes across the C->V edge.
        if VOCAB_TAIL != 0:
            if lm_core == VOCAB_FULL_TILES % FUSED_LM_HEAD_CORES:
                mm_tail_o0 = VOCAB_FULL_TILES * FUSED_VOCAB_TILE
                for tail_rb in pl.range(GROUP_LOGIT_ROWS // MM_ROW_TILE):
                    tail_r0 = tail_rb * MM_ROW_TILE
                    tail_acc = pl.create_tensor([MM_ROW_TILE, VOCAB_TAIL], dtype=pl.FP32)
                    for tail_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                        tail_k0 = tail_kb * FUSED_K_TILE
                        tail_hidden_tile = owner_hiddens[
                            tail_r0 : tail_r0 + MM_ROW_TILE, tail_k0 : tail_k0 + FUSED_K_TILE
                        ]
                        tail_weight_tile = lm_head_weight[
                            mm_tail_o0 : mm_tail_o0 + VOCAB_TAIL,
                            tail_k0 : tail_k0 + FUSED_K_TILE,
                        ]
                        tail_acc = pl.matmul_acc(tail_acc, tail_hidden_tile, tail_weight_tile, b_trans=True, init_cond=(tail_kb == 0))

                    for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                        tail_shard = pl.aiv_shard(tail_acc)
                        owner_tp = tail_rb // (MAX_LOGIT_ROWS // MM_ROW_TILE)
                        owner_r0 = (tail_rb % (MAX_LOGIT_ROWS // MM_ROW_TILE)) * MM_ROW_TILE + aiv_id * LANE_ROWS
                        if aiv_id * LANE_ROWS < MM_ROW_TILE:
                            pld.tensor.remote_store(
                                tail_shard, logits_window, group_base + owner_tp,
                                [owner_r0, vocab_base + mm_tail_o0],
                            )

        # Notify folded into the push: each block signals every peer after its own
        # stores, so a peer sees FUSED_LM_HEAD_CORES notifies per source per epoch.
        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            for notify_pair in pl.range(TP_SIZE // AIV_LANES):
                owner_tp = notify_pair * AIV_LANES + aiv_id
                if owner_tp != tp_rank:
                    pld.system.notify(
                        target=logits_done,
                        peer=group_base + owner_tp,
                        offsets=[tp_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

    # Wait only (the notify rides inside the push). deps on the push scope so the
    # wait runs alongside our own push; an unanchored wait dispatches immediately
    # and spins holding a core group.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", deps=[_push_tid]
    ) as _cwait_tid:
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=logits_done,
                    offsets=[src_tp, 0],
                    expected=pl.cast(done_epoch * FUSED_LM_HEAD_CORES, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Assemble full-vocabulary logits, same vocab-tile split. deps on _cwait_tid for
    # the peers' stores; our own tiles ride the local RAW edge on logits_window.
    with pl.spmd(
        LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_gather", deps=[_cwait_tid]
    ) as _gather_tid:
        gblk = pl.tile.get_block_idx()
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                for gr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                    logits[gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE] = logits_window[
                        gr : gr + LOGITS_GATHER_ROW_TILE, lo : lo + LOGITS_COMM_TILE
                    ]

            if LOGITS_COMM_TAIL != 0:
                if gblk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    tl = src_vocab_base + tail_o0
                    for tr in pl.range(0, MAX_LOGIT_ROWS, LOGITS_GATHER_ROW_TILE):
                        logits[tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL] = logits_window[
                            tr : tr + LOGITS_GATHER_ROW_TILE, tl : tl + LOGITS_COMM_TAIL
                        ]

    # Every local wait has observed all current-round peer notifies before the
    # logits gather can complete. Clear only this rank's counters so a retained
    # CommDomain can safely reuse the fixed done_epoch on the next forward.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_signal_clear",
        deps=[_gather_tid],
    ) as _clear_tid:
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return logits, _clear_tid


@pl.jit
def l2_lm_head(
    hidden_states: pl.Tensor[[LM_T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, PROJECTION_WEIGHT_LAYOUT],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]:
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_input_ready",
    ) as hidden_ready_tid:
        _input_anchor = pl.read(hidden_states, [0, 0])
    lm_head(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        group_base, tp_rank, done_epoch, hidden_ready_tid,
    )
    return logits


@pl.jit.inline
def greedy_sample(
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    grammar_mask: pl.Tensor[[MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Select the first maximum allowed token id per active row; inactive rows read -1.

    The mask uses 16 bits per word for each greedy scan segment. The final
    padding word of segment zero is -1 for an unconstrained row and 0 for a
    constrained row; it never overlaps vocabulary bits. An all--1 mask takes
    the native argmax fast path without expanding the bitset.
    """
    # One pass per row into a [BLOCK_ROWS, ROW_WIDTH] accumulator carrying the
    # running maximum and the block that set it; a lane's column is its position.
    # The inactive-row mask rides the same block: a dead row skips the scan
    # instead of being overwritten by a second pass over sampled_ids.
    logits_grid = pl.reshape(logits, [MAX_LOGIT_ROWS * GREEDY_GRID_ROWS, GREEDY_ROW_WIDTH])
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_greedy_sample"):
        if pl.read(logit_row_indices, [row]) < 0:
            sampled_ids[row : row + 1, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=-1)
        else:
            # The last padding word of segment zero is a row-mode header.
            # -1 selects the native argmax without reading or expanding mask bits.
            if pl.read(grammar_mask, [row, 0, GRAMMAR_SEGMENT_WORDS - 1]) == -1:
                row_base = row * GREEDY_GRID_ROWS
                running_max = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF)
                running_base = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.INT32, value=0)
                for block in pl.range(GREEDY_GRID_ROWS // GREEDY_BLOCK_ROWS):
                    block_row = row_base + block * GREEDY_BLOCK_ROWS
                    scores = logits_grid[block_row : block_row + GREEDY_BLOCK_ROWS, 0:GREEDY_ROW_WIDTH]
                    # Strict greater-than, so a lane keeps the earliest block it peaked at.
                    is_newer = pl.cmp(scores, running_max, cmp_type=4)
                    newer = pl.cast(is_newer, target_type=pl.INT32)
                    running_max = pl.maximum(running_max, scores)
                    block_base = pl.cast(block * GREEDY_BLOCK_SPAN, pl.INT32)
                    to_new = pl.neg(pl.sub(running_base, block_base))
                    running_base = pl.add(running_base, pl.mul(newer, to_new))

                # Broadcast the lane maxima back and column-reduce: every entry is then the
                # row maximum. A scalar pl.max over the lanes miscompiles on fp32
                # (ptoas_bitcast has no float overload).
                lane_maxima = pl.row_max(running_max)
                lane_zeros = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.FP32, value=0.0)
                lane_broadcast = pl.row_expand_add(lane_zeros, lane_maxima)
                best_value = pl.read(pl.col_max(lane_broadcast), [0, 0])

                # Flat index of every lane still at the row maximum, sentinel for the rest.
                # The lane * width term folds into the scalar combine below, keeping the
                # ramp a broadcast row rather than an (illegal) 2D arange.
                ramp_zeros = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.INT32, value=0)
                column_ramp = pl.col_expand(ramp_zeros, pl.arange(0, [1, GREEDY_ROW_WIDTH], dtype=pl.INT32))
                flat_index = pl.add(running_base, column_ramp)
                is_max = pl.cmp(running_max, best_value, cmp_type=0)
                hit = pl.cast(is_max, target_type=pl.INT32)
                offset_index = pl.sub(flat_index, GREEDY_INDEX_SENTINEL)
                candidates = pl.add(pl.mul(hit, offset_index), GREEDY_INDEX_SENTINEL)
                lane_indices = pl.row_min(candidates)
                best_index = pl.read(lane_indices, [0, 0])
                for lane in pl.range(1, GREEDY_BLOCK_ROWS):
                    lane_term = pl.cast(lane * GREEDY_ROW_WIDTH, pl.INT32)
                    lane_best = pl.read(lane_indices, [lane, 0]) + lane_term
                    best_index = pl.min(best_index, lane_best)

            else:
                row_base = row * GREEDY_GRID_ROWS
                token_ids = pl.col_expand(
                    pl.full([GREEDY_BLOCK_ROWS, GRAMMAR_LOOKUP_WIDTH], dtype=pl.INT32, value=0),
                    pl.arange(0, [1, GRAMMAR_LOOKUP_WIDTH], dtype=pl.INT32),
                )
                word_indices = pl.shr(token_ids, 4)
                bit_indices = pl.sub(token_ids, pl.mul(word_indices, 16))
                # Build the 16 bit selectors once, including signed INT16 bit 15.
                bit_basis = pl.shr(
                    pl.full([1, 16], dtype=pl.INT32, value=32768),
                    pl.sub(
                        pl.full([1, 16], dtype=pl.INT32, value=15),
                        pl.arange(0, [1, 16], dtype=pl.INT32),
                    ),
                )
                pl.write(bit_basis, [0, 15], pl.cast(-32768, pl.INT32))
                bit_basis_rows = pl.col_expand(
                    pl.full([GREEDY_BLOCK_ROWS, 16], dtype=pl.INT16, value=0),
                    pl.cast(bit_basis, pl.INT16),
                )
                bit_selectors = pl.gather(bit_basis_rows, dim=-1, index=bit_indices)
                running_max = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF)
                running_base = pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.INT32, value=0)
                for block in pl.range(GREEDY_GRID_ROWS // GREEDY_BLOCK_ROWS):
                    block_row = row_base + block * GREEDY_BLOCK_ROWS
                    scores = logits_grid[block_row : block_row + GREEDY_BLOCK_ROWS, 0:GREEDY_ROW_WIDTH]
                    mask_words = pl.gather(
                        grammar_mask[row, block * GREEDY_BLOCK_ROWS : (block + 1) * GREEDY_BLOCK_ROWS, :],
                        dim=-1,
                        index=word_indices,
                    )
                    bits = pl.and_(mask_words, bit_selectors)
                    # Compare signed bit values numerically, including INT16 bit 15.
                    forbidden = pl.cmp(pl.cast(bits, pl.FP32), 0.0, cmp_type=0)
                    masked_scores = pl.create_tensor([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.FP32)
                    masked_scores[:, :] = pl.add(
                        scores, pl.mul(forbidden[:, 0:GREEDY_ROW_WIDTH], -1.0e30)
                    )
                    newer = pl.cast(pl.cmp(masked_scores, running_max, cmp_type=4), target_type=pl.INT32)
                    running_max = pl.maximum(running_max, masked_scores)
                    block_base = pl.cast(block * GREEDY_BLOCK_SPAN, pl.INT32)
                    to_new = pl.neg(pl.sub(running_base, block_base))
                    running_base = pl.add(running_base, pl.mul(newer, to_new))

                lane_maxima = pl.row_max(running_max)
                lane_broadcast = pl.row_expand_add(
                    pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.FP32, value=0.0),
                    lane_maxima,
                )
                best_value = pl.read(pl.col_max(lane_broadcast), [0, 0])
                column_ramp = pl.col_expand(
                    pl.full([GREEDY_BLOCK_ROWS, GREEDY_ROW_WIDTH], dtype=pl.INT32, value=0),
                    pl.arange(0, [1, GREEDY_ROW_WIDTH], dtype=pl.INT32),
                )
                flat_index = pl.add(running_base, column_ramp)
                hit = pl.cast(pl.cmp(running_max, best_value, cmp_type=0), target_type=pl.INT32)
                candidates = pl.add(
                    pl.mul(hit, pl.sub(flat_index, GREEDY_INDEX_SENTINEL)),
                    GREEDY_INDEX_SENTINEL,
                )
                lane_indices = pl.row_min(candidates)
                best_index = pl.read(lane_indices, [0, 0])
                for lane in pl.range(1, GREEDY_BLOCK_ROWS):
                    lane_term = pl.cast(lane * GREEDY_ROW_WIDTH, pl.INT32)
                    lane_best = pl.read(lane_indices, [lane, 0]) + lane_term
                    best_index = pl.min(best_index, lane_best)

            sampled_row = pl.create_tensor([1, SAMPLED_IDS_PAD], dtype=pl.INT32)
            sampled_row[:, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
            pl.write(sampled_row, [0, 0], best_index)
            sampled_ids[row : row + 1, :] = sampled_row

    return sampled_ids


@pl.jit
def l2_lm_head_sample(
    hidden_states: pl.Tensor[[LM_T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, PROJECTION_WEIGHT_LAYOUT],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    grammar_mask: pl.Tensor[[MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
):
    """Projection plus greedy sampling: the decode-forward tail in one entry."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="lm_head_input_ready",
    ) as hidden_ready_tid:
        _input_anchor = pl.read(hidden_states, [0, 0])
    lm_head(
        hidden_states, lm_head_weight, logit_row_indices, logits,
        hidden_window, hidden_done, logits_window, logits_done,
        group_base, tp_rank, done_epoch, hidden_ready_tid,
    )
    greedy_sample(logits, logit_row_indices, grammar_mask, sampled_ids)
    return logits, sampled_ids


@pl.jit.host
def l3_lm_head_sample(
    hidden_states: pl.Tensor[[WORLD_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    grammar_mask: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    sampled_ids: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        l2_lm_head_sample(
            hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r], grammar_mask[r], sampled_ids[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // TP_SIZE * TP_SIZE, r % TP_SIZE, DONE_VALUE, device=r,
        )


PREFILL_ROWS = pl.dynamic("LOWLAT_PREFILL_HEAD_ROWS")
LOCAL_ROWS = pl.dynamic("LOWLAT_PREFILL_HEAD_LOCAL_ROWS")


@pl.jit(auto_scope=False)
def l2_prefill_tap(
    x_hc: pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32],
    target_hidden: pl.InOut[pl.Tensor[[LOCAL_ROWS, 3 * D], pl.BF16]],
    tap_index: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, PREFILL_ROWS)
    target_hidden.bind_dynamic(0, LOCAL_ROWS)
    local_rows = pl.tensor.dim(target_hidden, 0)
    first_row = pl.cast(rank, pl.INDEX) * local_rows
    column = pl.cast(tap_index, pl.INDEX) * D
    for token in pl.spmd(local_rows, name_hint="prefill_tp_target_tap"):
        row = first_row + token
        if row < group_tokens:
            hc_row = x_hc[row : row + 1, 0:HC_MULT, 0:D]
            mean = pl.mul(pl.col_sum(pl.reshape(hc_row, [HC_MULT, D])), 1.0 / HC_MULT)
            target_hidden[token : token + 1, column : column + D] = pl.cast(mean, pl.BF16, mode="rint")
        else:
            target_hidden[token : token + 1, column : column + D] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    return target_hidden


@pl.jit.host
def l3_prefill_tap(
    x_hc: pl.Tensor[[TP_SIZE, PREFILL_ROWS, HC_MULT, D], pl.FP32],
    target_hidden: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_ROWS, 3 * D], pl.BF16]],
    tap_index: pl.Scalar[pl.INT32],
    group_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, PREFILL_ROWS)
    target_hidden.bind_dynamic(1, LOCAL_ROWS)
    for rank in pl.range(pld.world_size()):
        l2_prefill_tap(x_hc[rank], target_hidden[rank], tap_index, group_tokens, rank, device=rank)
    return target_hidden


@pl.jit(auto_scope=False)
def l2_prefill_head(
    x_hc: pl.Tensor[[PREFILL_ROWS, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16, PROJECTION_WEIGHT_LAYOUT],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    grammar_mask: pl.Tensor[[MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    x_out: pl.Out[pl.Tensor[[PREFILL_ROWS, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, PREFILL_ROWS)
    x_out.bind_dynamic(0, PREFILL_ROWS)
    rows = pl.tensor.dim(x_hc, 0)
    hidden = pl.create_tensor([rows, D], dtype=pl.BF16)
    hidden = hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, hidden)
    ready = rms_norm(hidden, final_norm_w, x_out)
    lm_head(x_out, lm_head_weight, logit_row_indices, logits,
            hidden_window, hidden_done, logits_window, logits_done,
            pl.const(0, pl.INT32), rank, pl.const(1, pl.INT32), ready)
    greedy_sample(logits, logit_row_indices, grammar_mask, sampled_ids)
    return x_out


@pl.jit.host
def l3_prefill_head(
    x_hc: pl.Tensor[[TP_SIZE, PREFILL_ROWS, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[TP_SIZE, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[TP_SIZE, 1], pl.FP32],
    hc_head_base: pl.Tensor[[TP_SIZE, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[TP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[[TP_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[TP_SIZE, MAX_LOGIT_ROWS], pl.INT32],
    grammar_mask: pl.Tensor[[TP_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    x_out: pl.Out[pl.Tensor[[TP_SIZE, PREFILL_ROWS, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[TP_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[TP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
):
    x_hc.bind_dynamic(1, PREFILL_ROWS)
    x_out.bind_dynamic(1, PREFILL_ROWS)
    hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    hidden_done_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
    logits_done_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        l2_prefill_head(x_hc[rank], hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank],
                        final_norm_w[rank], lm_head_weight[rank], logit_row_indices[rank], grammar_mask[rank],
                        x_out[rank], logits[rank], sampled_ids[rank], hidden_window, hidden_done,
                        logits_window, logits_done, rank, device=rank)
    return x_out


def golden_lm_head_sample(tensors):
    import torch

    golden_lm_head(tensors)
    logits = tensors["logits"]
    indices = tensors["logit_row_indices"]
    sampled = torch.full(
        (WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD), -1, dtype=torch.int32,
    )
    for rank in range(WORLD_SIZE):
        for row in range(MAX_LOGIT_ROWS):
            if int(indices[rank, row]) < 0:
                continue
            sampled[rank, row, :] = 0
            sampled[rank, row, 0] = expected_sampled_id(logits[rank, row], tensors["grammar_mask"][rank, row])
    tensors["sampled_ids"][:] = sampled


def expected_sampled_id(logits, grammar_mask):
    """Compute the masked argmax for a golden or device-output comparison."""
    import torch

    if int(grammar_mask[0, GRAMMAR_SEGMENT_WORDS - 1]) == -1:
        return int(torch.argmax(logits))
    token_ids = torch.arange(VOCAB, device=logits.device)
    segment = token_ids // GREEDY_ROW_WIDTH
    offset = token_ids % GREEDY_ROW_WIDTH
    words = grammar_mask[segment, offset // 16].to(torch.int32)
    allowed = (words & (1 << (offset % 16))) != 0
    return int(torch.argmax(logits + (~allowed).to(logits.dtype) * -1.0e30))


@pl.jit.host
def l3_lm_head(
    hidden_states: pl.Tensor[[WORLD_SIZE, TEST_TOKENS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[WORLD_SIZE, VOCAB_PER_TP, D], pl.BF16],
    logits: pl.Out[pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    logit_row_indices: pl.Tensor[[WORLD_SIZE, MAX_LOGIT_ROWS], pl.INT32],
):
    # Windows are group-local: hidden_window holds one row slot per group member,
    # and every card receives only its own full-vocabulary logits.
    hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * VOCAB * 4)
    hidden_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)
    logits_done_buf = pld.alloc_window_buffer(TP_SIZE * 4)

    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(hidden_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(logits_window_buf, [MAX_LOGIT_ROWS, VOCAB], dtype=pl.FP32)
        logits_done = pld.window(logits_done_buf, [TP_SIZE, 1], dtype=pl.INT32)
        l2_lm_head(
            hidden_states[r], lm_head_weight[r], logit_row_indices[r], logits[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // TP_SIZE * TP_SIZE, r % TP_SIZE, DONE_VALUE, device=r,
        )


def golden_lm_head(tensors):
    import torch

    from utils import unpack_nz

    hidden = tensors["hidden_states"].float()
    # Card r holds shard r % TP_SIZE; concatenating shards in index order
    # reproduces the global vocabulary order.
    weight = unpack_nz(tensors["lm_head_weight"]).float()
    full_weight = torch.cat([weight[tp] for tp in range(TP_SIZE)], dim=0)
    full_logits = []
    for owner_rank in range(WORLD_SIZE):
        selected = torch.zeros((MAX_LOGIT_ROWS, D), dtype=torch.float32)
        for row in range(MAX_LOGIT_ROWS):
            source_row = int(tensors["logit_row_indices"][owner_rank, row])
            if source_row >= 0:
                source_row = min(source_row, hidden.shape[1] - 1)
                selected[row].copy_(hidden[owner_rank, source_row])
        full_logits.append(torch.matmul(selected, full_weight.t()))
    tensors["logits"][:] = torch.stack(full_logits, dim=0)


def build_tensor_specs_lm(num_tokens=TEST_TOKENS):
    import torch
    from golden import TensorSpec
    from utils import pack_nz

    active = max(min(num_tokens, MAX_LOGIT_ROWS), 0)

    def init_hidden_states():
        return (torch.randn(WORLD_SIZE, TEST_TOKENS, D) * 0.1).to(torch.bfloat16)

    def init_lm_head_weight():
        shards = (torch.randn(TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)
        return pack_nz(torch.stack([shards[r % TP_SIZE] for r in range(WORLD_SIZE)], dim=0))

    def init_logit_row_indices():
        indices = torch.full((WORLD_SIZE, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:, :active] = torch.arange(active, dtype=torch.int32)
        return indices

    return [
        TensorSpec("hidden_states", [WORLD_SIZE, TEST_TOKENS, D], torch.bfloat16, init_value=init_hidden_states),
        # One vocab shard per DP rank: card r carries a copy of shard
        # r % TP_SIZE, matching how resident args are handed out per rank. Keep
        # each rank-local shard on its consuming card across dispatches.
        TensorSpec(
            "lm_head_weight", [WORLD_SIZE, VOCAB_PER_TP, D], torch.bfloat16,
            init_value=init_lm_head_weight, resident="stacked",
        ),
        TensorSpec("logits", [WORLD_SIZE, MAX_LOGIT_ROWS, VOCAB], torch.float32),
        TensorSpec("logit_row_indices", [WORLD_SIZE, MAX_LOGIT_ROWS], torch.int32, init_value=init_logit_row_indices),
    ]


def golden_prefill_tap(tensors):
    import torch

    tap_index = int(tensors["tap_index"])
    group_tokens = int(tensors["group_tokens"])
    target = tensors["target_hidden"]
    local_rows = target.shape[1]
    column = tap_index * D
    for rank in range(TP_SIZE):
        first_row = rank * local_rows
        live_rows = max(0, min(local_rows, group_tokens - first_row))
        rows = torch.zeros((local_rows, D), dtype=torch.bfloat16)
        if live_rows:
            source = tensors["x_hc"][rank, first_row:first_row + live_rows]
            rows[:live_rows] = source.mean(dim=1).to(torch.bfloat16)
        target[rank, :, column:column + D] = rows


def golden_prefill_head(tensors):
    import torch
    from models.deepseek_v4_flash_lowlat.rmsnorm_common import golden_rms_norm

    output = tensors["x_out"]
    for rank in range(TP_SIZE):
        hidden = torch.empty_like(output[rank])
        golden_hc_head({
            "x_hc": tensors["x_hc"][rank],
            "hc_head_fn": tensors["hc_head_fn"][rank],
            "hc_head_scale": tensors["hc_head_scale"][rank],
            "hc_head_base": tensors["hc_head_base"][rank],
            "y": hidden,
        })
        output[rank] = golden_rms_norm(hidden, tensors["final_norm_w"][rank])
    golden_lm_head_sample({**tensors, "hidden_states": output})


def validate_composed_head(config, *, compile_only):
    import torch
    from golden import ScalarSpec, TensorSpec, ratio_allclose, run

    local_rows = 4
    physical_rows = TP_SIZE * local_rows
    logical_rows = physical_rows - 1
    x_hc = torch.randn((physical_rows, HC_MULT, D), dtype=torch.float32) * 0.05
    x_hc[logical_rows:] = 0
    x_hc = torch.stack([x_hc] * TP_SIZE)
    x_spec = TensorSpec("x_hc", list(x_hc.shape), torch.float32, init_value=x_hc)
    tap_shape = (TP_SIZE, local_rows, 3 * D)
    tap_result = run(
        fn=l3_prefill_tap,
        specs=[
            x_spec,
            TensorSpec("target_hidden", list(tap_shape), torch.bfloat16,
                       init_value=lambda: torch.full(tap_shape, 3, dtype=torch.bfloat16)),
            ScalarSpec("tap_index", torch.int32, 1),
            ScalarSpec("group_tokens", torch.int32, logical_rows),
        ],
        golden_fn=golden_prefill_tap, config=config, compile_only=compile_only,
        rtol=0, atol=0,
    )
    if not tap_result.passed:
        raise SystemExit(tap_result.error or 1)

    lm_specs = {spec.name: spec for spec in build_tensor_specs_lm(1)}
    indices = torch.full((TP_SIZE, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
    indices[0, 0] = logical_rows - 1
    mask = torch.full((TP_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS), -1, dtype=torch.int16)
    mask[0, 0] = 0
    mask[0, 0, 0] = -32768
    final_weight = torch.linspace(0.5, 1.5, D, dtype=torch.bfloat16).repeat(TP_SIZE, 1)
    head_result = run(
        fn=l3_prefill_head,
        specs=[
            x_spec,
            TensorSpec("hc_head_fn", [TP_SIZE, HC_MULT, HC_DIM], torch.float32,
                       init_value=lambda: torch.randn(TP_SIZE, HC_MULT, HC_DIM) * 0.0519),
            TensorSpec("hc_head_scale", [TP_SIZE, 1], torch.float32,
                       init_value=lambda: torch.full((TP_SIZE, 1), 0.076099)),
            TensorSpec("hc_head_base", [TP_SIZE, HC_MULT], torch.float32,
                       init_value=lambda: torch.tensor([5.9166, -3.6223, -2.9324, -3.3124]).repeat(TP_SIZE, 1)),
            TensorSpec("final_norm_w", [TP_SIZE, D], torch.bfloat16, init_value=final_weight),
            lm_specs["lm_head_weight"],
            TensorSpec("logit_row_indices", list(indices.shape), torch.int32, init_value=indices),
            TensorSpec("grammar_mask", list(mask.shape), torch.int16, init_value=mask),
            TensorSpec("x_out", [TP_SIZE, physical_rows, D], torch.bfloat16),
            lm_specs["logits"],
            TensorSpec("sampled_ids", [TP_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], torch.int32),
        ],
        golden_fn=golden_prefill_head, config=config, compile_only=compile_only,
        compare_fn={"x_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
        rtol=1e-3, atol=1e-3,
    )
    if not head_result.passed:
        raise SystemExit(head_result.error or 1)


def validate_hc():
    import argparse
    import torch
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--token-count", type=int, default=T, help=f"Physical token rows, up to {HC_VALIDATION_MAX_ROWS}.",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    # Int mode (0=off; 1=timing only, most accurate; 2=timing + dep graph, two runs).
    # `nargs="?"` so a bare `--enable-chip-swimlane` -> mode 1 (int, not bool True).
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run(
        fn=hc_head_test,
        specs=build_tensor_specs_hc(args.token_count),
        golden_fn=golden_hc_head,
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
            "y": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)



def validate_lm():
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES), help="LM-head tensor-parallel world size")
    parser.add_argument("--dp", type=int, default=DP_SIZE, choices=list(_DP_CHOICES), help="DP groups (world size = tp * dp)")
    parser.add_argument("--num-tokens", type=int, help="Active hidden rows each owner projects")
    parser.add_argument("--entry", choices=("projection", "sample"))
    device_default = ",".join(str(i) for i in range(WORLD_SIZE))
    parser.add_argument("-d", "--device", type=str, default=device_default, help=f"comma-separated device ids; need at least {WORLD_SIZE}")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    check_composed = (
        args.entry is None and args.num_tokens is None and args.runtime_dir is None and WORLD_SIZE == TP_SIZE
    )
    if args.num_tokens is None:
        args.num_tokens = TEST_TOKENS
    device_ids = [int(d) for d in args.device.split(",")]
    required_devices = WORLD_SIZE
    assert len(device_ids) >= required_devices, f"need at least {required_devices} devices, got {device_ids}"
    assert args.tp == TP_SIZE and args.dp == DP_SIZE
    assert 1 <= args.num_tokens <= TEST_TOKENS

    fn = l3_lm_head
    specs = build_tensor_specs_lm(args.num_tokens)
    golden_fn = golden_lm_head
    if args.entry == "sample":
        import torch

        from golden import TensorSpec

        fn = l3_lm_head_sample
        golden_fn = golden_lm_head_sample

        def init_grammar_mask():
            mask = torch.full(
                (WORLD_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS),
                -1, dtype=torch.int16,
            )
            mask[0, 0] = 0
            mask[0, 0, 0] = -32768  # Only token 15 is allowed in this row.
            return mask

        specs.append(
            TensorSpec(
                "grammar_mask", [WORLD_SIZE, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], torch.int16,
                init_value=init_grammar_mask,
            ),
        )
        specs.append(
            TensorSpec(
                "sampled_ids", [WORLD_SIZE, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], torch.int32,
            ),
        )

    config = dict(
        dump_passes=args.dump_passes,
        distributed_config=DistributedConfig(device_ids=device_ids[:required_devices], num_sub_workers=0),
        backend_type=args.backend_type,
        execution_mode=args.execution_mode,
        enable_chip_swimlane=args.enable_chip_swimlane,
        enable_scope_stats=args.enable_scope_stats,
        ring_heap=LM_HEAD_RING_HEAP,
    )
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=config,
        rtol=1e-3,
        atol=1e-3,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
    if check_composed:
        validate_composed_head(config, compile_only=args.compile_only)


if __name__ == "__main__":
    from models.deepseek_v4_flash_lowlat.utils import run_operator_validations

    run_operator_validations(
        {
            'hc': validate_hc,
            'lm': validate_lm,
        },
        single_device=('hc',),
        automatic=('lm', 'hc'),
        standalone_options={
            "lm": {"--entry", "--num-tokens", "--tp", "--dp", "--runtime-dir", "--enable-scope-stats"},
            "hc": {"--token-count", "--seed"},
        },
    )
