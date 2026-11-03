# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA/HCA 共用的 INT8 Q 展开：ND 分块 K，NZ 整段 K 常驻与有效尾行。

INT8×INT8 在本模型 K=1024 下不溢出 INT32；只改变搬运和整数矩阵分块。
Native 的浮点规约、反量化及舍入继续由各自入口负责。
"""

import pypto.language as pl

from ..common.config import FLASH as M
from ..common.nz_mode import QUANT_WEIGHT_LAYOUT, QUANT_WEIGHT_NZ

H = M.num_attention_heads
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
PREFILL_DENSE_TILE = 512
QPROJ_MM_T_DYN = pl.dynamic("QKV_QPROJ_MM_T_DYN")
Q_PROJ_TILE = 128
QPROJ_MM_N_TILE = 256 if QUANT_WEIGHT_NZ else 512
MATMUL_T_TILE = 16
QPROJ_M_TILE = 64
QPROJ_WORKERS = 20  # 留余量：QPROJ_N_BLOCKS=128 时 24 个 worker 只比核数少一点，派发时若有核
# 还被占着（实测 hca_kv_score_proj 会占掉 4 个 AIC 核）就要付两波的钱，wall 翻倍。20 块
# 换成 7 轮但恒为一波。2026-09-29 同卡六样本实测 −16.5 μs，与 allow_early_resolve 合计 −34.75。
QPROJ_TAIL_M_TILE = QPROJ_M_TILE if QUANT_WEIGHT_NZ else MATMUL_T_TILE
QPROJ_T_PAD = ((PREFILL_DENSE_TILE + QPROJ_TAIL_M_TILE - 1) // QPROJ_TAIL_M_TILE) * QPROJ_TAIL_M_TILE
QPROJ_N_BLOCKS = H * HEAD_DIM // QPROJ_MM_N_TILE
QPROJ_PIPE_M_TILE = 128
QPROJ_PIPE_K_TILE = 256

assert QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024
assert QPROJ_M_TILE % QPROJ_TAIL_M_TILE == 0


@pl.jit.inline(auto_scope=False)
def _q_proj_q_matmul_nd(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM], pl.INT32],
    tile_rows: pl.Scalar[pl.INDEX],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    """ND 版：N 索引直接用 `pl.range` 的循环变量，与上游同形。

    Project one bounded Q tile and expose its cube task ID.
    """
    qproj_t_matmul = pl.tensor.dim(q_proj_i32, 0)
    qproj_full_rows = qproj_t_matmul  # 调用方已按 QPROJ_M_TILE 取整
    with pl.spmd(
        QPROJ_WORKERS,
        name_hint="qproj_matmul",
        deps=[qproj_dep],
    ) as qproj_tid:
        # Match upstream weight streaming; arithmetic is unchanged.
        pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
        qproj_worker = pl.tile.get_block_idx()
        for qproj_n_idx in pl.range(
            qproj_worker,
            QPROJ_N_BLOCKS,
            QPROJ_WORKERS,
        ):
            w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
            # ND retains the split-K pipeline and its existing padded row path.
            for t0 in pl.range(0, qproj_full_rows, QPROJ_M_TILE):
                col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
                for qr_proj_col0 in pl.pipeline(0, Q_LORA, Q_PROJ_TILE, stage=2):
                    qr_i8_chunk = qr_i8_matmul[
                        t0 : t0 + QPROJ_M_TILE,
                        qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE,
                    ]
                    wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                    col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk, init_cond=(qr_proj_col0 == 0))
                q_proj_i32[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc

    return q_proj_i32, qproj_tid


@pl.jit.inline(auto_scope=False)
def _q_proj_q_matmul_nz(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM], pl.INT32],
    tile_rows: pl.Scalar[pl.INDEX],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    """N256 列块内按 K256 双缓冲：下一段权重搬运与本段 Cube 计算重叠。

    原实现每轮先整段读入 K1024×N256 权重再计算，搬运与计算串行；这里对每个 M128 行块
    按 K 顺序 matmul_acc。INT8×INT8→INT32 累加精确，分块不改变结果。
    """
    qproj_t_matmul = pl.tensor.dim(q_proj_i32, 0)
    with pl.spmd(QPROJ_WORKERS, name_hint="qproj_matmul", deps=[qproj_dep]) as qproj_tid:
        # Match upstream weight streaming; arithmetic is unchanged.
        pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
        qproj_worker = pl.tile.get_block_idx()
        # Keep the column offset provably nonnegative after NZ outlining.
        for qproj_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + QPROJ_WORKERS - 1) // QPROJ_WORKERS):
            qproj_n_idx = qproj_worker + qproj_round * QPROJ_WORKERS
            w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
            for t0 in pl.range(0, qproj_t_matmul, QPROJ_PIPE_M_TILE):
                m_rows = pl.min(QPROJ_PIPE_M_TILE, tile_rows - t0)
                qr_first = pl.slice(qr_i8_matmul, [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE], [t0, 0],
                                    valid_shape=[m_rows, QPROJ_PIPE_K_TILE])
                wq_first = wq_b[0:QPROJ_PIPE_K_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                # 首段由 matmul 生成累加器，携带有效行的 compact 形态。
                col_acc = pl.matmul(qr_first, wq_first, out_dtype=pl.INT32)
                for k0 in pl.pipeline(QPROJ_PIPE_K_TILE, Q_LORA, QPROJ_PIPE_K_TILE, stage=2):
                    qr_chunk = pl.slice(qr_i8_matmul, [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE], [t0, k0],
                                        valid_shape=[m_rows, QPROJ_PIPE_K_TILE])
                    wq_chunk = wq_b[k0 : k0 + QPROJ_PIPE_K_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                    col_acc = pl.matmul_acc(col_acc, qr_chunk, wq_chunk)
                q_proj_i32[t0 : t0 + QPROJ_PIPE_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc
    return q_proj_i32, qproj_tid


q_proj_q_matmul = _q_proj_q_matmul_nz if QUANT_WEIGHT_NZ else _q_proj_q_matmul_nd
