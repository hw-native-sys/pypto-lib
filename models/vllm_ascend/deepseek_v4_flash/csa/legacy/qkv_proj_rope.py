# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/qkv_proj_rope.py."""

import pypto.language as pl

from ..common.config import (
    FLASH as M,
)
from ..common.config import (
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from ..common.nz_mode import BF16_WEIGHT_LAYOUT, BF16_WEIGHT_NZ, QUANT_WEIGHT_LAYOUT
from .q_projection import (
    PREFILL_DENSE_TILE,
    QPROJ_M_TILE,
    QPROJ_MM_N_TILE,
    QPROJ_MM_T_DYN,
    QPROJ_PIPE_M_TILE,
    QPROJ_T_PAD,
    QPROJ_TAIL_M_TILE,
    q_proj_q_matmul,
)
from ..common.reduction import ATOMIC_ADD, STORE_ATOMIC

T_DYN = pl.dynamic("QKV_Q_T_DYN")  # T = B * S

KV_T_DYN = pl.dynamic("QKV_KV_T_DYN")

KV_MM_T_DYN = pl.dynamic("QKV_KV_MM_T_DYN")

ROPE_T_DYN = pl.dynamic("QKV_ROPE_T_DYN")


D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

ROPE_DIM_SCALE = float(ROPE_DIM)

ROPE_HALF = ROPE_DIM // 2

NOPE_DIM = M.nope_head_dim

Q_LORA = M.q_lora_rank

EPS = M.rms_norm_eps

MAX_SEQ_LEN = M.max_position_embeddings

Q_LORA_TILE = 256  # qr rms-norm / quant N granularity

KV_TILE = 64  # kv rms-norm / rope / NOPE N granularity

QUANT_TILE = 256

T_TILE = 8


MATMUL_T_TILE = 16

QR_M_TILE = MATMUL_T_TILE  # qr_proj token (M) tile; cube rows must be a 16-row boxed tile

QR_DENSE_M_TILE = 64

QR_FIXED_SMALL_M_TILE = 32

QR_FIXED_SMALL_ROWS = 128

QR_FIXED_M_GROUPS = 3

QR_N_TILE = 128  # qr_proj Q_LORA (N) per matmul；上游值。取 32 是为了对齐 Native 的
# 96 列分组且不跨 K 序边界，代价是每行只读 32 列 BF16 = 64 字节，
# in-core 实测 MTE2 占 55.6%、Cube 只有 28.4%。

QR_K_TILE = 256  # qr_proj D (K) reduction tile   | divides QR_SPLIT_K_TILE

# Keep the deployment split counts; smaller upstream counts did not lower the full-layer interval.
QR_OK = 8 if ATOMIC_ADD else 1  # 单分片按固定 K 顺序累加，避免跨核 atomic add。

QR_SPLIT_K_TILE = D // QR_OK

# CANN 9.0 MatMulV2 NT at M96/144/192/240: paired forward/reverse K256 traversals.
# Sources: tests/pypto_test/results/cann90_20260921/qa_native_codegen_six/.
# 性能版去掉 QR_NATIVE_* 的 K 块乱序：那是为复刻 Native 的 K 累加次序，
# 乱序读会打掉硬件预取，正是 MTE2 居高不下的另一半原因。

KV_M_TILE = MATMUL_T_TILE  # kv_proj token (M) tile; decode pads from 8 real rows to 16

KV_DENSE_M_TILE = 64

KV_FIXED_SMALL_M_TILE = 32

KV_FIXED_SMALL_ROWS = 128

KV_N_TILE = 128  # kv_proj HEAD_DIM (N) per matmul

KV_K_TILE = 256 if ATOMIC_ADD else 512  # Fixed-K: fewer L1 transfers; atomic path unchanged.

# A single split keeps a fixed K accumulation order.
KV_OK = 8 if ATOMIC_ADD else 1

KV_OM = 3  # maximum kv_proj split-M factor

KV_SPLIT_K_TILE = D // KV_OK

KV_RMS_T_TILE = 32  # kv rms-norm + rope fused token (T) tile

Q_ROPE_T_TILE = 8

Q_ROPE_WORKERS = 48

QR_NORM_WORKERS = 16  # qr_rms_norm_quant 的 worker 上限，上游值

Q_ROPE_H_TILE = 4  # heads per fused qproj dequant/rms/rope task

Q_DEQUANT_WORKERS = 48

assert QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C Acc cap

assert QPROJ_M_TILE % QPROJ_TAIL_M_TILE == 0


@pl.jit.inline
def rope_prepare(
    rope_sin: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.INT32],
):
    """Fold Native interleaved sin signs and build the pair-swap indices."""
    t_dim = pl.tensor.dim(rope_sin, 0)
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    # The tail guard below writes the outputs from inside a conditional region.
    # Binding them to the scalar extent keeps the region result off the dynamic
    # symbol, which has no definition in an inlining caller that passes a
    # statically shaped token axis.
    rope_sin_signed_view = pl.reshape(rope_sin_signed, [t_dim, ROPE_DIM])
    rope_swap_idx_view = pl.reshape(rope_swap_idx, [t_dim, ROPE_DIM])

    token_tiles = (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
    for qrp_worker in pl.spmd(
        pl.min(Q_ROPE_WORKERS, token_tiles), name_hint="q_rope_prepare", allow_early_resolve=True
    ):
        # The pair lane, swap index and sign fold only depend on the column
        # index, so they are built once per worker instead of once per tile.
        qrp_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        for qrp_idx in pl.range(qrp_worker, token_tiles, pl.min(Q_ROPE_WORKERS, token_tiles)):
            qrp_t0 = qrp_idx * Q_ROPE_T_TILE
            qrp_valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - qrp_t0)
            if qrp_valid_rows == Q_ROPE_T_TILE:
                qrp_sin_rows_full = rope_sin_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
                qrp_sin_signed_full = pl.mul(qrp_sin_rows_full, qrp_sign)
                rope_sin_signed_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_sin_signed_full
                rope_swap_idx_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_swap_idx
            else:
                qrp_sin_rows_tail = pl.load(
                    rope_sin_view,
                    [qrp_t0, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[qrp_valid_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                qrp_tail_col = pl.col_expand_mul(
                    pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                    pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                )
                qrp_tail_dup_f = pl.cast(
                    pl.cast(pl.mul(qrp_tail_col, 0.5), target_type=pl.INT32, mode="trunc"),
                    target_type=pl.FP32,
                )
                qrp_tail_lane = pl.sub(qrp_tail_col, pl.mul(qrp_tail_dup_f, 2.0))
                qrp_tail_swap_f = pl.sub(pl.add(qrp_tail_col, 1.0), pl.mul(qrp_tail_lane, 2.0))
                qrp_tail_sign = pl.sub(pl.mul(qrp_tail_lane, 2.0), 1.0)
                qrp_tail_sign = pl.set_validshape(qrp_tail_sign, qrp_valid_rows, ROPE_DIM)
                qrp_sin_signed_tail = pl.mul(qrp_sin_rows_tail, qrp_tail_sign)
                pl.store(
                    pl.set_validshape(qrp_sin_signed_tail, qrp_valid_rows, ROPE_DIM),
                    [qrp_t0, 0],
                    rope_sin_signed_view,
                )
                pl.store(
                    pl.set_validshape(
                        pl.cast(qrp_tail_swap_f, target_type=pl.INT32),
                        qrp_valid_rows,
                        ROPE_DIM,
                    ),
                    [qrp_t0, 0],
                    rope_swap_idx_view,
                )


# NZ 与 ND 拆成两个独立函数；末尾择一。见 _q_proj_qa_nz 的说明。
QR_N_BLOCKS = Q_LORA // QR_N_TILE


@pl.jit.inline(auto_scope=False)
def _q_proj_qa_nd(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16],
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    DENSE_M: pl.constexpr,
    M_GROUP_LIMIT: pl.constexpr,
):
    """ND 版：K 分片用取模、N 分块用整除，与上游同形。

    Reference QA matmul, shared by the production QR and its diagnostic.
    """
    qa_tokens = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [qa_tokens, D])
    qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
    qr_full_rows = (tile_rows // DENSE_M) * DENSE_M
    qr_m_groups = pl.min(M_GROUP_LIMIT, pl.max(1, tile_rows // DENSE_M))
    # A single K partition overwrites each output tile exactly once.
    # Only split-K atomic accumulation needs a zero seed in GM.
    if ATOMIC_ADD == 1:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
            # Reuse one full-width zero tile; keep the same single seed task
            # and padded row range, avoiding repeated narrow strided stores.
            qr_seed = pl.full([QR_M_TILE, Q_LORA], dtype=pl.FP32, value=0.0)
            for ts0 in pl.range(0, qr_t_matmul, QR_M_TILE):
                qr_fp32[ts0 : ts0 + QR_M_TILE, :] = qr_seed

    with pl.spmd(QR_N_BLOCKS * QR_OK * qr_m_groups, name_hint="qr_proj_matmul", allow_early_resolve=True) as qa_tid:
        qbg_idx = pl.tile.get_block_idx()
        pl.set_cache_policy(wq_a, pl.CachePolicy.BYPASS)
        q_a_col0 = ((qbg_idx // QR_OK) % QR_N_BLOCKS) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_SPLIT_K_TILE
        qr_m_group = qbg_idx // (QR_N_BLOCKS * QR_OK)
        for dense_t0 in pl.range(qr_m_group * DENSE_M, qr_full_rows, qr_m_groups * DENSE_M):
            dense_x0 = tile_base + dense_t0
            dense_acc = pl.create_tensor([DENSE_M, QR_N_TILE], dtype=pl.FP32)
            for dense_k in pl.pipeline(0, QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                dense_d0 = qr_k_base + dense_k * QR_K_TILE
                dense_x = x_view[dense_x0 : dense_x0 + DENSE_M, dense_d0 : dense_d0 + QR_K_TILE]
                dense_w = wq_a[q_a_col0 : q_a_col0 + QR_N_TILE, dense_d0 : dense_d0 + QR_K_TILE]
                dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, b_trans=True, init_cond=(dense_k == 0))
            qr_fp32 = pl.assemble(qr_fp32, dense_acc, [dense_t0, q_a_col0], atomic=STORE_ATOMIC)
        for t0 in pl.range(qr_full_rows + qr_m_group * QR_M_TILE, qr_t_matmul, qr_m_groups * QR_M_TILE):
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, tile_rows - t0)
                x_t0 = tile_base + t0
                q_x_chunk_bf16 = pl.slice(
                    x_view,
                    [QR_M_TILE, QR_K_TILE],
                    [x_t0, qr_d0],
                    valid_shape=[qr_rows, QR_K_TILE],
                )
                w_chunk = wq_a[q_a_col0 : q_a_col0 + QR_N_TILE, qr_d0 : qr_d0 + QR_K_TILE]
                q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk, b_trans=True, init_cond=(db == 0))
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=STORE_ATOMIC)

    return qa_tid


@pl.jit.inline(auto_scope=False)
def _q_proj_qa_nz(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    DENSE_M: pl.constexpr,
    M_GROUP_LIMIT: pl.constexpr,
):
    """NZ 路径按 K 分片和 N 块分解 block 索引。

    偏移证明使用上游 NZ 修复对非负整数除法/取模的支持；保留原 K 分片内的累加顺序，
    atomic 开关仍决定是否跨分片累加。根布局匹配时直接借用 Native 存储，不为此路径重新打包权重。
    """
    qa_tokens = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [qa_tokens, D])
    qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
    qr_full_rows = (tile_rows // DENSE_M) * DENSE_M
    qr_m_groups = pl.min(M_GROUP_LIMIT, pl.max(1, tile_rows // DENSE_M))
    # A single K partition overwrites each output tile exactly once.
    # Only split-K atomic accumulation needs a zero seed in GM.
    if ATOMIC_ADD == 1:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
            # Reuse one full-width zero tile; keep the same single seed task
            # and padded row range, avoiding repeated narrow strided stores.
            qr_seed = pl.full([QR_M_TILE, Q_LORA], dtype=pl.FP32, value=0.0)
            for ts0 in pl.range(0, qr_t_matmul, QR_M_TILE):
                qr_fp32[ts0 : ts0 + QR_M_TILE, :] = qr_seed

    # 用 `with pl.spmd(...)` + `get_block_idx()` 而不是 `for ... in pl.spmd(...)`：
    # 后者给出的是 IterArg，`IsProvableNonNegative` 不追它，于是由它导出的行偏移
    # 不可证（实测报 `slice offset on shape[-2] to be non-negative cannot be proven`）。
    # `get_block_idx()` 是明确的 SPMD block 索引，属于可证形式。
    with pl.spmd(QR_N_BLOCKS * QR_OK * qr_m_groups, name_hint="qr_proj_matmul", allow_early_resolve=True) as qa_tid:
        pl.set_cache_policy(wq_a, pl.CachePolicy.BYPASS)
        qbg_idx = pl.tile.get_block_idx()
        qr_k_base = ((qbg_idx // QR_N_BLOCKS) % QR_OK) * QR_SPLIT_K_TILE
        q_a_col0 = (qbg_idx % QR_N_BLOCKS) * QR_N_TILE
        qr_m_group = qbg_idx // (QR_N_BLOCKS * QR_OK)
        for dense_t0 in pl.range(qr_m_group * DENSE_M, qr_full_rows, qr_m_groups * DENSE_M):
            dense_x0 = tile_base + dense_t0
            dense_acc = pl.create_tensor([DENSE_M, QR_N_TILE], dtype=pl.FP32)
            for dense_k in pl.pipeline(0, QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                dense_d0 = qr_k_base + dense_k * QR_K_TILE
                dense_x = x_view[dense_x0 : dense_x0 + DENSE_M, dense_d0 : dense_d0 + QR_K_TILE]
                dense_w = wq_a[q_a_col0 : q_a_col0 + QR_N_TILE, dense_d0 : dense_d0 + QR_K_TILE]
                dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, b_trans=True, init_cond=(dense_k == 0))
            qr_fp32 = pl.assemble(qr_fp32, dense_acc, [dense_t0, q_a_col0], atomic=STORE_ATOMIC)
        for t0 in pl.range(qr_full_rows + qr_m_group * QR_M_TILE, qr_t_matmul, qr_m_groups * QR_M_TILE):
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, tile_rows - t0)
                x_t0 = tile_base + t0
                q_x_chunk_bf16 = pl.slice(
                    x_view,
                    [QR_M_TILE, QR_K_TILE],
                    [x_t0, qr_d0],
                    valid_shape=[qr_rows, QR_K_TILE],
                )
                w_chunk = wq_a[q_a_col0 : q_a_col0 + QR_N_TILE, qr_d0 : qr_d0 + QR_K_TILE]
                q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk, b_trans=True, init_cond=(db == 0))
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=STORE_ATOMIC)

    return qa_tid


_q_proj_qa_tiles = _q_proj_qa_nz if BF16_WEIGHT_NZ else _q_proj_qa_nd



QA_M16_WORKER_LIMIT = QR_N_BLOCKS * QR_FIXED_M_GROUPS
QA_SINGLE_M16_N_TILE = QR_N_TILE // 2


@pl.jit.inline(auto_scope=False)
def q_proj_qa_m16_parallel(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    N_TILE: pl.constexpr = QR_N_TILE,
    SYNC_START: pl.constexpr = False,
):
    """66882634：每个M16/N_TILE独立输出分核，K次序及实际尾行保持。"""
    launch_workers = pl.min(
        QA_M16_WORKER_LIMIT,
        (Q_LORA // N_TILE) * ((tile_rows + QR_M_TILE - 1) // QR_M_TILE),
    )
    with pl.spmd(
        launch_workers, name_hint="qr_proj_matmul_m16", allow_early_resolve=True, sync_start=SYNC_START
    ) as qa_tid:
        pl.set_cache_policy(wq_a, pl.CachePolicy.BYPASS)
        worker_id = pl.tile.get_block_idx()
        tokens = tile_rows
        m16_blocks = (tokens + QR_M_TILE - 1) // QR_M_TILE
        for logical_worker in pl.range(pl.max(worker_id, 0), (Q_LORA // N_TILE) * m16_blocks, QA_M16_WORKER_LIMIT):
            column = (logical_worker % (Q_LORA // N_TILE)) * N_TILE
            row = (logical_worker // (Q_LORA // N_TILE)) * QR_M_TILE
            m16_acc = pl.create_tensor([QR_M_TILE, N_TILE], dtype=pl.FP32)
            for k_block in pl.pipeline(D // QR_K_TILE, stage=2):
                k0 = k_block * QR_K_TILE
                valid_rows = pl.min(QR_M_TILE, tokens - row)
                m16_x = pl.slice(x, [QR_M_TILE, QR_K_TILE], [tile_base + row, k0],
                                 valid_shape=[valid_rows, QR_K_TILE])
                m16_weight = wq_a[column : column + N_TILE, k0 : k0 + QR_K_TILE]
                m16_acc = pl.matmul_acc(m16_acc, m16_x, m16_weight, b_trans=True, init_cond=(k_block == 0))
            qr_fp32 = pl.assemble(qr_fp32, m16_acc, [row, column])
    return qa_tid


@pl.jit.inline(auto_scope=False)
def q_proj_qa_policy(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    qa_m16_parallel: pl.Scalar[pl.INT32],
    SINGLE_M16_QA: pl.constexpr = False,
):
    """Keep complete K traversal; partition independent M/N outputs for fixed-K."""
    if SINGLE_M16_QA:
        if ATOMIC_ADD == 1:
            qa_tid = _q_proj_qa_tiles(
                x,
                wq_a,
                qr_fp32,
                tile_base,
                tile_rows,
                QR_DENSE_M_TILE,
                1,
            )
        elif qa_m16_parallel == 2:
            qa_tid = q_proj_qa_m16_parallel(
                x,
                wq_a,
                qr_fp32,
                tile_base,
                tile_rows,
                N_TILE=QA_SINGLE_M16_N_TILE,
                SYNC_START=True,
            )
        elif qa_m16_parallel != 0:
            qa_tid = q_proj_qa_m16_parallel(
                x,
                wq_a,
                qr_fp32,
                tile_base,
                tile_rows,
                N_TILE=QR_N_TILE,
                SYNC_START=False,
            )
        elif tile_rows < QR_FIXED_SMALL_ROWS:
            qa_tid = _q_proj_qa_tiles(
                x,
                wq_a,
                qr_fp32,
                tile_base,
                tile_rows,
                QR_FIXED_SMALL_M_TILE,
                QR_FIXED_M_GROUPS,
            )
        else:
            qa_tid = _q_proj_qa_tiles(
                x,
                wq_a,
                qr_fp32,
                tile_base,
                tile_rows,
                QR_DENSE_M_TILE,
                QR_FIXED_M_GROUPS,
            )
    else:
        if ATOMIC_ADD == 1:
            qa_tid = _q_proj_qa_tiles(x, wq_a, qr_fp32, tile_base, tile_rows, QR_DENSE_M_TILE, 1)
        elif qa_m16_parallel != 0:
            qa_tid = q_proj_qa_m16_parallel(x, wq_a, qr_fp32, tile_base, tile_rows, N_TILE=QR_N_TILE, SYNC_START=False)
        elif tile_rows < QR_FIXED_SMALL_ROWS:
            qa_tid = _q_proj_qa_tiles(x, wq_a, qr_fp32, tile_base, tile_rows, QR_FIXED_SMALL_M_TILE, QR_FIXED_M_GROUPS)
        else:
            qa_tid = _q_proj_qa_tiles(x, wq_a, qr_fp32, tile_base, tile_rows, QR_DENSE_M_TILE, QR_FIXED_M_GROUPS)
    return qa_tid


@pl.jit.inline(auto_scope=False)
def q_proj_qr_policy(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    qa_m16_parallel: pl.Scalar[pl.INT32],
    SINGLE_M16_QA: pl.constexpr = False,
):
    """Q LoRA, RMSNorm and quantization -- the half the indexer query chain needs.

    The two staging tensors are held by the caller so the qproj half can be issued
    separately; they are rewritten per dense tile, so a caller that runs more than one
    tile iteration must let the qproj half finish before the next one starts. Every
    current caller has t_dim <= PREFILL_DENSE_TILE, i.e. a single iteration.
    """
    qa_tids = pl.array.create(1, pl.TASK_ID)
    qa_tids[0] = pl.system.task_invalid()
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
            qr_fp32 = pl.create_tensor([qr_t_matmul, Q_LORA], dtype=pl.FP32)
            # Bind before the array update: nested inline expansion otherwise
            # creates a reserved auto-name in the current PyPTO compiler.
            qa_task = q_proj_qa_policy(
                x, wq_a, qr_fp32, tile_base, tile_rows, qa_m16_parallel, SINGLE_M16_QA=SINGLE_M16_QA
            )
            qa_tids[0] = qa_task

            q_proj_qr_normalize(qr_fp32, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store, tile_base, tile_rows)

    return qa_tids[0]


@pl.jit.inline(auto_scope=False)
def q_proj_qr_normalize(
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """QR 的 RMSNorm + 动态量化（上游写法）。

    第一遍同时求平方和与 gamma 加权后的 amax：RMSNorm 是逐行常数缩放，
    amax(normed) = inv_rms * amax(qr * gamma)，所以不必为 amax 再扫一遍。
    已封存的精度版为复刻 Native 的舍入，先把 FP32 落成 BF16、按 1024->...->64 折半规约、
    再逐行用整数比较修正 sqrt 的末位并逐行做标量除法，这些本版都不需要。
    """
    t_dim = pl.tensor.dim(qr, 0)
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    gamma_cq_view = pl.reshape(gamma_cq, [1, Q_LORA])

    qr_token_tiles = (tile_rows + T_TILE - 1) // T_TILE
    qr_norm_workers = pl.min(qr_token_tiles, QR_NORM_WORKERS)
    for tg_idx_worker in pl.spmd(qr_norm_workers, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
        # Keep gamma and one full row group in UB, as in AscendC's
        # RmsNormDynamicQuantNormal. Preserve the existing 256-column
        # reduction order and the RMS-then-gamma quantization arithmetic.
        qr_gamma_ub = pl.cast(
            pl.load(gamma_cq_view, [0, 0], [1, Q_LORA], target_memory=pl.MemorySpace.Vec),
            target_type=pl.FP32,
        )
        for tg_idx in pl.range(tg_idx_worker, qr_token_tiles, qr_norm_workers):
            tg = tg_idx * T_TILE
            valid_rows = pl.min(T_TILE, tile_rows - tg)
            out_tg = tile_base + tg
            qr_input_ub = pl.load(
                qr_fp32, [tg, 0], [T_TILE, Q_LORA], target_memory=pl.MemorySpace.Vec
            )
            qr_sq_sum = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            qr_amax_g = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for qr_rms_col0 in pl.pipeline(0, Q_LORA, Q_LORA_TILE, stage=2):
                qr_rms_chunk = pl.tile.extract(
                    qr_input_ub, 0, qr_rms_col0, [T_TILE, Q_LORA_TILE], target_memory=pl.MemorySpace.Vec
                )
                qr_rms_sq = pl.mul(qr_rms_chunk, qr_rms_chunk)
                qr_rms_row_sum = pl.reshape(
                    pl.row_sum(qr_rms_sq, tmp_tile=pl.tile.create([T_TILE, Q_LORA_TILE], dtype=pl.FP32)),
                    [1, T_TILE],
                )
                qr_sq_sum = pl.add(qr_sq_sum, qr_rms_row_sum)
                gamma_rms_chunk = pl.tile.extract(
                    qr_gamma_ub, 0, qr_rms_col0, [1, Q_LORA_TILE], target_memory=pl.MemorySpace.Vec
                )
                qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
                qr_g_row_max = pl.reshape(
                    pl.row_max(pl.abs(qr_g), tmp_tile=pl.tile.create([T_TILE, Q_LORA_TILE], dtype=pl.FP32)),
                    [1, T_TILE],
                )
                qr_amax_g = pl.maximum(qr_amax_g, qr_g_row_max)
            qr_inv_rms = pl.tile.rsqrt(
                pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS),
                tmp=pl.tile.create([1, T_TILE], dtype=pl.FP32),
            )
            qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
            qr_amax_floor = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            qr_tile_amax = pl.maximum(qr_amax_floor, pl.mul(qr_inv_rms, qr_amax_g))

            qr_scale_quant_row = pl.div(pl.tile.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
            qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
            qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
            pl.store(qr_tile_scale_dq, [tg, 0], qr_scale_pad_store)
            if valid_rows == T_TILE:
                pl.store(qr_tile_scale_dq, [out_tg, 0], qr_scale_view)
            else:
                # Publish the current UB value directly. A side-effect store
                # followed by a GM reload must not become a stale tail read.
                qr_scale_tail = pl.set_validshape(qr_tile_scale_dq, valid_rows, 1)
                pl.store(qr_scale_tail, [out_tg, 0], qr_scale_view)

            for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
                qr_chunk = pl.tile.extract(
                    qr_input_ub, 0, qa, [T_TILE, QUANT_TILE], target_memory=pl.MemorySpace.Vec
                )
                gamma_q_chunk = pl.tile.extract(
                    qr_gamma_ub, 0, qa, [1, QUANT_TILE], target_memory=pl.MemorySpace.Vec
                )
                qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
                qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
                qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
                qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
                qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
                pl.store(qr_q_i8, [tg, qa], qr_i8_matmul)
                if valid_rows == T_TILE:
                    pl.store(qr_q_i8, [out_tg, qa], qr_view)
                else:
                    qr_q_tail = pl.set_validshape(qr_q_i8, valid_rows, QUANT_TILE)
                    pl.store(qr_q_tail, [out_tg, qa], qr_view)


# NZ 与 ND 拆成两个独立函数，不写成一个函数里的 `if QUANT_WEIGHT_NZ`——@pl.jit 读源
# 文件做 AST 分析，两个分支会一起被 trace 而冲突。末尾择一。


@pl.jit.inline(auto_scope=False)
def q_proj_q_dequant(
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM], pl.INT32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Dequantize, normalize, and rotate one projected Q tile."""
    t_dim = pl.tensor.dim(q, 0)
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for dq_worker in pl.spmd(
        Q_DEQUANT_WORKERS,
        name_hint="qproj_dequant_rms_nope_rope",
        allow_early_resolve=True,
    ):
        # 9d5327e5：完整八行按48-AIV整波选最大head组；尾行保持4。
        dq_token_tiles = (tile_rows + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
        dq_head_tile = Q_ROPE_H_TILE
        # 只合并完整八行块的独立 head：选择最大的 64/32/16/8，使
        # dq_token_tiles*(H/head_tile) 恰为正的 48-AIV 整波；不满足则保留4。
        # 扩组后每个AIV的head迭代数不增，只减少scale/cos/sin/swap的重复准备。
        # 此条件取实际设备T，不用B/S或测试点；WORKERS仍只决定任务条带。
        # 有尾行必须留4，保持下面固定4-head尾循环及原recip(sqrt)数学。
        if tile_rows % Q_ROPE_T_TILE == 0:
            if dq_token_tiles >= Q_DEQUANT_WORKERS and dq_token_tiles % Q_DEQUANT_WORKERS == 0:
                dq_head_tile = H
            elif (
                dq_token_tiles >= Q_DEQUANT_WORKERS * 32 // H
                and dq_token_tiles % (Q_DEQUANT_WORKERS * 32 // H) == 0
            ):
                dq_head_tile = 32
            elif (
                dq_token_tiles >= Q_DEQUANT_WORKERS * 16 // H
                and dq_token_tiles % (Q_DEQUANT_WORKERS * 16 // H) == 0
            ):
                dq_head_tile = 16
            elif (
                dq_token_tiles >= Q_DEQUANT_WORKERS * 8 // H
                and dq_token_tiles % (Q_DEQUANT_WORKERS * 8 // H) == 0
            ):
                dq_head_tile = 8
        for dq_work in pl.range(
            dq_worker,
            ((tile_rows + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE) * (H // dq_head_tile),
            Q_DEQUANT_WORKERS,
        ):
            hg = (dq_work % (H // dq_head_tile)) * dq_head_tile
            tg = (dq_work // (H // dq_head_tile)) * Q_ROPE_T_TILE
            out_tg = tile_base + tg
            if tg + Q_ROPE_T_TILE <= tile_rows:
                qr_scale_dq_t = pl.load(
                    qr_scale_pad_store, [tg, 0], [Q_ROPE_T_TILE, 1], target_memory=pl.MemorySpace.Vec,
                )
                q_cos_il = pl.load(
                    rope_cos_il, [out_tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec,
                )
                q_sin_signed = pl.load(
                    rope_sin_signed, [out_tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec,
                )
                q_swap_idx = pl.load(
                    rope_swap_idx, [out_tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec,
                )
                # Preserve the caller's within-row indices; add row starts once per work tile.
                q_flat_i = pl.cast(
                    pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32), pl.FP32,
                )
                q_row_i = pl.cast(pl.mul(q_flat_i, 1.0 / ROPE_DIM), pl.INT32, mode="trunc")
                q_row_starts = pl.mul(pl.cast(q_row_i, pl.FP32), ROPE_DIM_SCALE)
                q_abs_swap = pl.cast(pl.add(
                    pl.cast(pl.reshape(q_swap_idx, [1, Q_ROPE_T_TILE * ROPE_DIM]), pl.FP32), q_row_starts,
                ), pl.INT32)
                q_flat_gather_tmp = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
                for h_inner in pl.pipeline(dq_head_tile, stage=2):
                    h = hg + h_inner
                    h0 = h * HEAD_DIM
                    q_head_acc = pl.load(
                        q_proj_i32, [tg, h0], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale = pl.reshape(pl.load(
                        wq_b_scale, [h0], [HEAD_DIM], target_memory=pl.MemorySpace.Vec,
                    ), [1, HEAD_DIM])
                    q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                    q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                    q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                    q_head_sq = pl.mul(q_head_dq, q_head_dq)
                    q_head_sq_row = pl.row_sum(
                        q_head_sq, tmp_tile=pl.tile.create([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32),
                    )
                    q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                    q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                    q_head_var = pl.add(q_head_sq_mean, EPS)
                    q_head_inv_rms = pl.tile.rsqrt(
                        q_head_var, pl.tile.create([1, Q_ROPE_T_TILE], dtype=pl.FP32),
                    )
                    q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])
                    q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                    q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                    pl.store(q_nope_bf16, [out_tg, h0], q_flat)
                    q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                    q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                    q_rope_flat = pl.reshape(q_rope_chunk, [1, Q_ROPE_T_TILE * ROPE_DIM])
                    q_swapped_flat = pl.tile.gather(q_rope_flat, q_abs_swap, q_flat_gather_tmp)
                    q_rope_swapped = pl.reshape(q_swapped_flat, [Q_ROPE_T_TILE, ROPE_DIM])
                    q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                    q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                    q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                    q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                    pl.store(q_rope_bf16, [out_tg, h0 + NOPE_DIM], q_flat)
            else:
                valid_tail_rows = tile_rows - tg
                # q_proj and its dequant scale are padded to the cube boundary.
                # Keep the math on a static eight-row tile and crop public stores.
                qr_scale_dq_tail = pl.load(
                    qr_scale_pad_store,
                    [tg, 0],
                    [Q_ROPE_T_TILE, 1],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_cos_il_tail = pl.load(
                    rope_cos_il,
                    [out_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[valid_tail_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_sin_signed_tail = pl.load(
                    rope_sin_signed,
                    [out_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[valid_tail_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_col = pl.col_expand_mul(
                    pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                    pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                )
                q_dup_f = pl.cast(pl.cast(pl.mul(q_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
                q_lane = pl.sub(q_col, pl.mul(q_dup_f, 2.0))
                q_swap_f = pl.sub(pl.add(q_col, 1.0), pl.mul(q_lane, 2.0))
                # Row-major flattening offsets stay in fp32: col-expand is defined
                # for half / float only.
                q_row_seed = pl.mul(
                    pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                    ROPE_DIM_SCALE,
                )
                q_row_grid = pl.col_expand_mul(
                    pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0),
                    q_row_seed,
                )
                q_row_offset = pl.transpose(q_row_grid, axis1=0, axis2=1)
                q_swap_idx_tail = pl.cast(pl.add(q_swap_f, q_row_offset), target_type=pl.INT32)
                q_head_reduce_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, HEAD_DIM],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                q_gather_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    dtype=pl.INT32,
                    target_memory=pl.MemorySpace.Vec,
                )
                for h_inner_tail in pl.range(Q_ROPE_H_TILE):
                    h_tail = hg + h_inner_tail
                    h0_tail = h_tail * HEAD_DIM
                    q_head_acc_tail = pl.load(
                        q_proj_i32,
                        [tg, h0_tail],
                        [Q_ROPE_T_TILE, HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale_input_tail = pl.load(
                        wq_b_scale,
                        [h0_tail],
                        [HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale_tail = pl.reshape(q_head_scale_input_tail, [1, HEAD_DIM])
                    q_head_acc_fp32_tail = pl.cast(q_head_acc_tail, target_type=pl.FP32, mode="none")
                    q_head_row_scaled_tail = pl.row_expand_mul(q_head_acc_fp32_tail, qr_scale_dq_tail)
                    q_head_dq_tail = pl.col_expand_mul(q_head_row_scaled_tail, q_head_scale_tail)

                    q_head_sq_tail = pl.mul(q_head_dq_tail, q_head_dq_tail)
                    q_head_sq_sum_tail = pl.row_sum(q_head_sq_tail, q_head_reduce_tmp)
                    q_head_inv_rms_tail = pl.recip(
                        pl.sqrt(pl.add(pl.mul(q_head_sq_sum_tail, 1.0 / HEAD_DIM), EPS)),
                    )

                    q_nope_normed_tail = pl.row_expand_mul(q_head_dq_tail[:, 0:NOPE_DIM], q_head_inv_rms_tail)
                    q_nope_bf16_tail = pl.cast(q_nope_normed_tail, target_type=pl.BF16, mode="rint")
                    q_nope_valid = pl.set_validshape(q_nope_bf16_tail, valid_tail_rows, NOPE_DIM)
                    pl.store(q_nope_valid, [out_tg, h0_tail], q_flat)

                    q_rope_chunk_raw_tail = q_head_dq_tail[:, NOPE_DIM:HEAD_DIM]
                    q_rope_chunk_tail = pl.row_expand_mul(q_rope_chunk_raw_tail, q_head_inv_rms_tail)
                    q_rope_swapped_tail = pl.tile.gather(q_rope_chunk_tail, q_swap_idx_tail, q_gather_tmp)
                    # RoPE inputs contain only the live rows of this tail.
                    q_rope_chunk_tail = pl.set_validshape(q_rope_chunk_tail, valid_tail_rows, ROPE_DIM)
                    q_rope_swapped_tail = pl.set_validshape(q_rope_swapped_tail, valid_tail_rows, ROPE_DIM)
                    q_rope_base_tail = pl.mul(q_rope_chunk_tail, q_cos_il_tail)
                    q_rope_delta_tail = pl.mul(q_rope_swapped_tail, q_sin_signed_tail)
                    q_rope_rot_tail = pl.add(q_rope_base_tail, q_rope_delta_tail)
                    q_rope_bf16_tail = pl.cast(q_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    q_rope_valid = pl.set_validshape(q_rope_bf16_tail, valid_tail_rows, ROPE_DIM)
                    pl.store(q_rope_valid, [out_tg, h0_tail + NOPE_DIM], q_flat)
    return q


@pl.jit.inline(auto_scope=False)
def q_proj_q(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    """Q projection and its dequant + RMSNorm + RoPE over bounded tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            # Reserve full cube row tiles. NZ marks the tail's actual rows;
            # ND computes the padded INT8 rows. Dequant reads tile_rows only.
            # Q_B 按 M128 行块整块写回，缓冲按同一粒度取整。
            qproj_t_matmul = ((tile_rows + QPROJ_PIPE_M_TILE - 1) // QPROJ_PIPE_M_TILE) * QPROJ_PIPE_M_TILE
            q_proj_i32 = pl.create_tensor([qproj_t_matmul, H * HEAD_DIM], dtype=pl.INT32)
            q_proj_i32, _qproj_tid = q_proj_q_matmul(
                wq_b,
                qr_i8_matmul,
                q_proj_i32,
                tile_rows,
                qproj_dep,
            )
            q_proj_q_dequant(
                wq_b_scale,
                rope_cos_il,
                rope_sin_signed,
                rope_swap_idx,
                q,
                qr_scale_pad_store,
                q_proj_i32,
                tile_base,
                tile_rows,
            )
    return q


@pl.jit.inline(auto_scope=False)
def q_proj_rope_policy(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qa_m16_parallel: pl.Scalar[pl.INT32],
    SINGLE_M16_QA: pl.constexpr = False,
):
    """Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles."""
    qr_i8_matmul = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    # The quant scale rides the qr_i8 -> qproj_matmul -> dequant chain.
    qr_scale_pad_store = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
    qa_tid = q_proj_qr_policy(
        x, wq_a, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store, qa_m16_parallel, SINGLE_M16_QA=SINGLE_M16_QA
    )
    q_seq_dep = pl.system.task_dummy(deps=[])
    q_proj_q(
        x,
        wq_b,
        wq_b_scale,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        q,
        qr_i8_matmul,
        qr_scale_pad_store,
        q_seq_dep,
    )

    return qa_tid


@pl.jit.inline(auto_scope=False)
def _kv_project(
    x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    kv_fp32: pl.Tensor[[KV_MM_T_DYN, HEAD_DIM], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    t_matmul: pl.Scalar[pl.INDEX],
    late_dep: pl.Scalar[pl.TASK_ID],
    DENSE_M: pl.constexpr,
    GROUP_ROWS: pl.constexpr,
):
    """Partition M/N independently; every block preserves its complete K traversal."""
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    kv_full_rows = (tile_rows // DENSE_M) * DENSE_M
    kv_m_groups = pl.min(KV_OM, pl.max(1, tile_rows // GROUP_ROWS))
    with pl.spmd(
        (HEAD_DIM // KV_N_TILE) * KV_OK * kv_m_groups,
        name_hint="kv_proj_matmul", allow_early_resolve=True,
        deps=[late_dep],
    ) as _kv_tid:
        pl.set_cache_policy(wkv, pl.CachePolicy.BYPASS)
        kbg = pl.tile.get_block_idx()
        kv_col0 = (kbg // (KV_OK * kv_m_groups)) * KV_N_TILE
        kv_k_base = ((kbg // kv_m_groups) % KV_OK) * KV_SPLIT_K_TILE
        kv_m_group = kbg % kv_m_groups
        for dense_t0 in pl.range(kv_m_group * DENSE_M, kv_full_rows, kv_m_groups * DENSE_M):
            dense_x0 = tile_base + dense_t0
            dense_acc = pl.create_tensor([DENSE_M, KV_N_TILE], dtype=pl.FP32)
            for dense_k in pl.pipeline(0, KV_SPLIT_K_TILE // KV_K_TILE, stage=2):
                dense_d0 = kv_k_base + dense_k * KV_K_TILE
                dense_x = x_view[dense_x0 : dense_x0 + DENSE_M, dense_d0 : dense_d0 + KV_K_TILE]
                dense_w = wkv[dense_d0 : dense_d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, init_cond=(dense_k == 0))
            kv_fp32 = pl.assemble(kv_fp32, dense_acc, [dense_t0, kv_col0], atomic=STORE_ATOMIC)
        for t0 in pl.range(kv_full_rows + kv_m_group * KV_M_TILE, t_matmul, kv_m_groups * KV_M_TILE):
            kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(KV_SPLIT_K_TILE // KV_K_TILE, stage=2):
                d0 = kv_k_base + db * KV_K_TILE
                kv_rows = pl.min(KV_M_TILE, tile_rows - t0)
                x_t0 = tile_base + t0
                kv_x_chunk_bf16 = pl.slice(
                    x_view,
                    [KV_M_TILE, KV_K_TILE],
                    [x_t0, d0],
                    valid_shape=[kv_rows, KV_K_TILE],
                )
                wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk, init_cond=(db == 0))
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [t0, kv_col0], atomic=STORE_ATOMIC)
    return kv_fp32


@pl.jit.inline(auto_scope=False)
def kv_proj_rope(
    x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.INT32],
    kv: pl.Tensor[[KV_T_DYN, HEAD_DIM], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """KV LoRA, RMSNorm, and RoPE over bounded dense tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            t_matmul = ((tile_rows + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE

            # Fixed-K blocks own disjoint M/N outputs. Split-K blocks instead
            # atomic-add partial sums, which requires a zero-seeded output.
            kv_fp32 = pl.create_tensor([t_matmul, HEAD_DIM], dtype=pl.FP32)
            # With KV_OK=1, disjoint M/N blocks overwrite every output tile.
            if ATOMIC_ADD == 1:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
                    kv_seed = pl.full([KV_M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                    for kts0 in pl.range(0, t_matmul, KV_M_TILE):
                        kv_fp32[kts0 : kts0 + KV_M_TILE, :] = kv_seed

            if ATOMIC_ADD == 1:
                kv_fp32 = _kv_project(
                    x, wkv, kv_fp32, tile_base, tile_rows, t_matmul, late_dep,
                    KV_DENSE_M_TILE, 2 * KV_DENSE_M_TILE,
                )
            elif tile_rows < KV_FIXED_SMALL_ROWS:
                kv_fp32 = _kv_project(
                    x, wkv, kv_fp32, tile_base, tile_rows, t_matmul, late_dep,
                    KV_FIXED_SMALL_M_TILE, KV_FIXED_SMALL_M_TILE,
                )
            else:
                kv_fp32 = _kv_project(
                    x, wkv, kv_fp32, tile_base, tile_rows, t_matmul, late_dep,
                    KV_DENSE_M_TILE, KV_DENSE_M_TILE,
                )

            kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])

            # Fused KV RMSNorm + interleaved (CANN A3) RoPE, one spmd task per
            # [KV_RMS_T_TILE, HEAD_DIM] row block. NOPE columns [0:NOPE_DIM) and rope columns
            # [NOPE_DIM:HEAD_DIM) are disjoint, so each task writes a conflict-free row block.
            kv_token_tiles = (tile_rows + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
            for tg_idx in pl.spmd(
                kv_token_tiles,
                name_hint="kv_rms_norm_rope", allow_early_resolve=True,
                sync_start=True,
            ):
                tg = tg_idx * KV_RMS_T_TILE
                valid_rows = pl.min(KV_RMS_T_TILE, tile_rows - tg)
                out_tg = tile_base + tg
                if valid_rows == KV_RMS_T_TILE:
                    # Native RMSNorm ReduceSumMultiN accumulates the eight
                    # 64-column vectors lane-wise, then WholeReduceSum.
                    # Reducing each vector first changes FP32 rounding.
                    kv_sq_lanes = pl.full([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
                        kv_sq = pl.mul(kv_chunk, kv_chunk)
                        kv_sq_lanes = pl.add(kv_sq_lanes, kv_sq)
                    kv_sq_sum = pl.reshape(pl.row_sum(kv_sq_lanes), [1, KV_RMS_T_TILE])
                    kv_rms = pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS))
                    kv_inv_rms = pl.div(pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_rms)
                    kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

                    for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
                        gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
                        gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
                        kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                        kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
                        kv_view[out_tg : out_tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16

                    gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
                    gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
                    kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
                    kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
                    kv_cos_il_full = rope_cos_il[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_sin_signed_full = rope_sin_signed[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swap_idx_full = rope_swap_idx[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swapped_full = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx_full)
                    kv_rope_rot_full = pl.add(
                        pl.mul(kv_rope_norm_chunk, kv_cos_il_full),
                        pl.mul(kv_swapped_full, kv_sin_signed_full),
                    )
                    kv_rope_i16_full = pl.cast(kv_rope_rot_full, target_type=pl.BF16, mode="rint")
                    kv_view[out_tg : out_tg + KV_RMS_T_TILE, NOPE_DIM:HEAD_DIM] = kv_rope_i16_full
                else:
                    kv_reduce_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    kv_sq_lanes_tail = pl.tile.full([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0_tail in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk_tail = pl.load(
                            kv_fp32,
                            [tg, kv_sq_col0_tail],
                            [KV_RMS_T_TILE, KV_TILE],
                            valid_shape=[valid_rows, KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        kv_sq_tail = pl.mul(kv_chunk_tail, kv_chunk_tail)
                        kv_sq_lanes_tail = pl.add(kv_sq_lanes_tail, kv_sq_tail)
                    kv_sq_sum_tail = pl.reshape(pl.row_sum(kv_sq_lanes_tail, kv_reduce_tmp), [1, KV_RMS_T_TILE])
                    kv_rms_tail = pl.sqrt(pl.add(pl.mul(kv_sq_sum_tail, 1.0 / HEAD_DIM), EPS))
                    kv_inv_rms_tail = pl.div(
                        pl.tile.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_rms_tail
                    )
                    kv_inv_rms_t_tail = pl.reshape(kv_inv_rms_tail, [KV_RMS_T_TILE, 1])

                    for n0_tail in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                        kv_chunk_tail = pl.load(
                            kv_fp32,
                            [tg, n0_tail],
                            [KV_RMS_T_TILE, KV_TILE],
                            valid_shape=[valid_rows, KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        gamma_kv_input_tail = pl.load(
                            gamma_ckv,
                            [n0_tail],
                            [KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        gamma_kv_cast_tail = pl.cast(gamma_kv_input_tail, target_type=pl.FP32)
                        gamma_kv_chunk_tail = pl.reshape(gamma_kv_cast_tail, [1, KV_TILE])
                        kv_normed_tail = pl.col_expand_mul(
                            pl.row_expand_mul(kv_chunk_tail, kv_inv_rms_t_tail),
                            gamma_kv_chunk_tail,
                        )
                        kv_normed_bf16_tail = pl.cast(kv_normed_tail, target_type=pl.BF16, mode="rint")
                        kv_normed_valid = pl.set_validshape(kv_normed_bf16_tail, valid_rows, KV_TILE)
                        pl.store(kv_normed_valid, [out_tg, n0_tail], kv_view)

                    gamma_rope_input_tail = pl.load(
                        gamma_ckv,
                        [NOPE_DIM],
                        [ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    gamma_rope_cast_tail = pl.cast(gamma_rope_input_tail, target_type=pl.FP32)
                    gamma_rope_tail = pl.reshape(gamma_rope_cast_tail, [1, ROPE_DIM])
                    kv_rope_chunk_tail = pl.load(
                        kv_fp32,
                        [tg, NOPE_DIM],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_rope_norm_tail = pl.col_expand_mul(
                        pl.row_expand_mul(kv_rope_chunk_tail, kv_inv_rms_t_tail),
                        gamma_rope_tail,
                    )
                    kv_cos_il_tail = pl.load(
                        rope_cos_il,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_sin_signed_tail = pl.load(
                        rope_sin_signed,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_col = pl.col_expand_mul(
                        pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                        pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                    )
                    kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
                    kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))
                    kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))
                    # Row-major flattening offsets stay in fp32: col-expand is defined
                    # for half / float only.
                    kv_row_seed = pl.mul(
                        pl.cast(pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                        ROPE_DIM_SCALE,
                    )
                    kv_row_grid = pl.col_expand_mul(
                        pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0),
                        kv_row_seed,
                    )
                    kv_row_offset = pl.transpose(kv_row_grid, axis1=0, axis2=1)
                    kv_swap_idx_tail = pl.cast(pl.add(kv_swap_f, kv_row_offset), target_type=pl.INT32)
                    kv_gather_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, ROPE_DIM],
                        dtype=pl.INT32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_swapped_tail = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
                    kv_rope_rot_tail = pl.add(
                        pl.mul(kv_rope_norm_tail, kv_cos_il_tail),
                        pl.mul(kv_swapped_tail, kv_sin_signed_tail),
                    )
                    kv_rope_i16_tail = pl.cast(kv_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    kv_rope_valid = pl.set_validshape(kv_rope_i16_tail, valid_rows, ROPE_DIM)
                    pl.store(kv_rope_valid, [out_tg, NOPE_DIM], kv_view)


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope_policy(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
    qa_m16_parallel: pl.Scalar[pl.INT32],
    SINGLE_M16_QA: pl.constexpr = False,
):
    """Fused q + kv projection: both branches share one token axis and one rope table."""
    t_dim = pl.tensor.dim(x, 0)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_sin, q_rope_sin_signed, q_rope_swap_idx)
    qa_tid = q_proj_rope_policy(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        rope_cos,
        q_rope_sin_signed,
        q_rope_swap_idx,
        q,
        qr,
        qr_scale,
        qa_m16_parallel,
        SINGLE_M16_QA=SINGLE_M16_QA,
    )
    kv_proj_rope(
        x,
        wkv,
        gamma_ckv,
        rope_cos,
        q_rope_sin_signed,
        q_rope_swap_idx,
        kv,
        late_dep,
    )
    return q, qa_tid


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Fused q + kv projection: both branches share one token axis and one rope table."""
    return qkv_proj_rope_policy(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos,
        rope_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        late_dep,
        pl.cast(0, pl.INT32),
        SINGLE_M16_QA=False,
    )


@pl.jit.inline(auto_scope=False)
def q_proj_qa(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Keep complete K traversal; partition independent M/N outputs for fixed-K."""
    return q_proj_qa_policy(x, wq_a, qr_fp32, tile_base, tile_rows, pl.cast(0, pl.INT32), SINGLE_M16_QA=False)


@pl.jit.inline(auto_scope=False)
def q_proj_qr(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
):
    """Q LoRA, RMSNorm and quantization -- the half the indexer query chain needs.

    The two staging tensors are held by the caller so the qproj half can be issued
    separately; they are rewritten per dense tile, so a caller that runs more than one
    tile iteration must let the qproj half finish before the next one starts. Every
    current caller has t_dim <= PREFILL_DENSE_TILE, i.e. a single iteration.
    """
    return q_proj_qr_policy(
        x, wq_a, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store, pl.cast(0, pl.INT32), SINGLE_M16_QA=False
    )


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
):
    """Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles."""
    return q_proj_rope_policy(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        q,
        qr,
        qr_scale,
        pl.cast(0, pl.INT32),
        SINGLE_M16_QA=False,
    )
