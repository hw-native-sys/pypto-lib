# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态CSA的Indexer query前处理：六个显式worker的核内数学阶段。

一、范围与阅读顺序
    query_project→dequant_rope→hadamard→query_quant形成query分支；
    weights_project→weights_reduce形成独立head权重分支。只复用当前高层
    decode_indexer.py数学，不导入历史IR，也不包含历史QK/TopK MIX。

二、布局与动态容量
    QR是INT8[T,1024]，Indexer投影权重是ND INT8[1024,8192]，输出INT32
    投影至少pad64(T)行。RoPE后的BF16为[pad16(T),8192]，外层还须对同一
    数据建立[pad16(T)*64,128]的GM别名供Hadamard使用；helper不在incore内
    reshape整张GM。Hadamard矩阵为BF16[128,128]，FP32中间量至少T*64行，
    INT8结果和FP32 scale各有独立动态行容量，至少T*64，后续消费者可要求
    pad16(T)*64。weights分支输入x_normed BF16[T,4096]和ND BF16[4096,64]，
    partial至少4*pad16(T)行，输出FP32至少pad16(T)行。
    cos沿用Native交错RoPE布局；sin必须是外层已按偶负奇正乘符号的FP32
    signed_sin，不能直接把未带符号的频率sin传入。
    T取真实Tensor轴；Hadamard/量化/weights_reduce的tokens由外层相同T传入，
    不能用较大的容量推导有效token。此范围不使用S/请求边界或Host取值。

三、分工与同步合同
    所有helper只有普通核内range/pipeline，没有spmd/submit或task依赖。
    query投影按N256列块轮转AIC、整条K1024权重供所有M64行块复用；反量化
    按8行×4head轮转AIV，T不足8或最后尾行沿用逐行数学。Hadamard与量化
    按64个head行一块，因每token正好64heads，任意整数T均没有head行尾块。
    weights按M16行块×4个K1024 split轮转AIC，reduce按M16轮转AIV。
    外层负责真实producer通知/等待、业务fence和缓存维护；return仅表达GM
    别名。本核全部wave结束后才可发布phase，缓冲在消费者完成前不能覆盖。

四、数值顺序与当前边界
    以当前canonical源码为准，dequant严格保持(acc*activation_scale)*weight_scale，
    不混入历史“先合并scale”的变体。Hadamard后乘128**-0.5，amax先下限再
    两个64列块，scale=127/amax→recip，量化rint→FP16 round→INT8 trunc。
    weights四份partial按原左折叠相加，再乘WEIGHTS_SCALE，保留FP32输出。
    gamma/权重不做Host转换；数学等价审计和CPU后端不能证明设备精度、同步或性能。
"""

import pypto.language as pl

from ..common.config import FLASH as M
from ..common.config import INT8_AMAX_EPS, INT8_SCALE_MAX

D = M.hidden_size
Q_LORA = M.q_lora_rank
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
WEIGHTS_SCALE = M.index_weights_scale
HADAMARD_SCALE = IDX_HEAD_DIM**-0.5
QR_MM_ROW_TILE = 64
QR_MM_N_TILE = 256
DEQUANT_T_TILE = 8
DQ_ROPE_H_TILE = 4
QH_MM_TILE = 64
QH_QUANT_TILE = 64
QH_HEAD_DIM_TILE = 64
MM_ROW_TILE = 16
WEIGHTS_OK = 4
WEIGHTS_K_TILE = D // WEIGHTS_OK
D_TILE = 512

T_DYN = pl.dynamic("CSA_INDEX_QUERY_T")
PROJ_ROWS_DYN = pl.dynamic("CSA_INDEX_PROJ_PAD64")
ROPE_ROWS_DYN = pl.dynamic("CSA_INDEX_ROPE_PAD16")
ROPE_HEAD_ROWS_DYN = pl.dynamic("CSA_INDEX_ROPE_HEAD_ROWS")
HADAMARD_ROWS_DYN = pl.dynamic("CSA_INDEX_HADAMARD_ROWS")
QUANT_ROWS_DYN = pl.dynamic("CSA_INDEX_QUANT_ROWS")
SCALE_ROWS_DYN = pl.dynamic("CSA_INDEX_SCALE_ROWS")
WEIGHTS_ROWS_DYN = pl.dynamic("CSA_INDEX_WEIGHTS_PAD16")
WEIGHTS_PARTIAL_ROWS_DYN = pl.dynamic("CSA_INDEX_WEIGHTS_PARTIAL_ROWS")


@pl.jit.inline(auto_scope=False)
def indexer_query_project_stage(
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    qr_acc_pad: pl.Tensor[[PROJ_ROWS_DYN, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIC：按N256条带投影，先等QR；返回后发布query投影完成。"""
    bs = pl.tensor.dim(qr, 0)
    row_blocks = (bs + QR_MM_ROW_TILE - 1) // QR_MM_ROW_TILE
    qr_proj_worker = worker_id
    # 同一worker负责N256条带的所有M64行块；后继不能只等某一行的假想owner。
    for qr_col in pl.range(qr_proj_worker, IDX_N_HEADS * IDX_HEAD_DIM // QR_MM_N_TILE, WORKERS):
        o_base = qr_col * QR_MM_N_TILE
        wq_tile = wq_b[0:Q_LORA, o_base : o_base + QR_MM_N_TILE]
        for qr_rb in pl.range(row_blocks):
            qr_r0 = qr_rb * QR_MM_ROW_TILE
            qr_rows = pl.min(QR_MM_ROW_TILE, bs - qr_r0)
            qr_tile = pl.slice(qr, [QR_MM_ROW_TILE, Q_LORA], [qr_r0, 0], valid_shape=[qr_rows, Q_LORA])
            qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
            qr_acc_pad[qr_r0 : qr_r0 + QR_MM_ROW_TILE, o_base : o_base + QR_MM_N_TILE] = qr_acc
    return qr_acc_pad


@pl.jit.inline(auto_scope=False)
def indexer_query_dequant_rope_stage(
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    qr_acc_pad: pl.Tensor[[PROJ_ROWS_DYN, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT32],
    qr_bf16_2d: pl.Tensor[[ROPE_ROWS_DYN, IDX_N_HEADS * IDX_HEAD_DIM], pl.BF16],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV：等query INT32投影与QR scale，原反量化/RoPE后写BF16，外层发布供Hadamard消费。"""
    bs = pl.tensor.dim(qr_scale, 0)
    dq_rope_units = ((bs + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE) * (IDX_N_HEADS // DQ_ROPE_H_TILE)
    dq_rope_workers = pl.min(dq_rope_units, WORKERS)
    if worker_id < dq_rope_workers:
        dq_rope_worker = worker_id
        sw_ones = pl.full([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        sw_index = pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
        sw_col = pl.col_expand_mul(sw_ones, sw_index)
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx = pl.cast(pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)
        # 每单元为8个token、4个head；核内多wave覆盖动态T，不按某个B/S冻结分配。
        for dq_unit in pl.range(dq_rope_worker, dq_rope_units, dq_rope_workers):
            hg = dq_unit % (IDX_N_HEADS // DQ_ROPE_H_TILE) * DQ_ROPE_H_TILE
            dq_t0 = dq_unit // (IDX_N_HEADS // DQ_ROPE_H_TILE) * DEQUANT_T_TILE
            if dq_t0 + DEQUANT_T_TILE <= bs:
                qr_scale_tile = pl.load(qr_scale, [dq_t0, 0], [DEQUANT_T_TILE, 1])
                cos_tile = pl.load(cos, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                sin_tile = pl.load(sin, [dq_t0, 0], [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                flat_i = pl.tile.ci(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
                flat_tmp = pl.create_tile([1, ROPE_HEAD_DIM], dtype=pl.INT32)
                flat_lane = pl.tile.rems(flat_i, 2, flat_tmp)
                swap_row = pl.tile.adds(pl.tile.sub(flat_i, pl.tile.muls(flat_lane, 2)), IDX_NOPE_HEAD_DIM + 1)
                swap_base = pl.create_tile([DEQUANT_T_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
                swap_source = pl.col_expand(swap_base, swap_row)
                row_offsets = pl.tile.muls(pl.tile.ci(0, [1, DEQUANT_T_TILE], dtype=pl.INT32), IDX_HEAD_DIM)
                flat_swap = pl.reshape(
                    pl.row_expand_add(swap_source, pl.reshape(row_offsets, [DEQUANT_T_TILE, 1])),
                    [1, DEQUANT_T_TILE * ROPE_HEAD_DIM],
                )
                gather_tmp = pl.create_tile([1, DEQUANT_T_TILE * ROPE_HEAD_DIM], dtype=pl.INT32)
                for h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                    h0 = (hg + h_inner) * IDX_HEAD_DIM
                    wq_scale = pl.reshape(pl.load(wq_b_scale, [h0], [IDX_HEAD_DIM]), [1, IDX_HEAD_DIM])
                    acc_fp32 = pl.cast(
                        pl.load(qr_acc_pad, [dq_t0, h0], [DEQUANT_T_TILE, IDX_HEAD_DIM]),
                        target_type=pl.FP32,
                        mode="none",
                    )
                    qr_dequant = pl.col_expand_mul(pl.row_expand_mul(acc_fp32, qr_scale_tile), wq_scale)
                    qr_nope_bf16 = pl.cast(qr_dequant[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint")
                    qr_rope_slice = qr_dequant[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                    qr_swapped_flat = pl.tile.gather(
                        pl.reshape(qr_dequant, [1, DEQUANT_T_TILE * IDX_HEAD_DIM]), flat_swap, gather_tmp
                    )
                    qr_swapped = pl.reshape(qr_swapped_flat, [DEQUANT_T_TILE, ROPE_HEAD_DIM])
                    rope_rot = pl.add(pl.mul(qr_rope_slice, cos_tile), pl.mul(qr_swapped, sin_tile))
                    rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
                    pl.store(qr_nope_bf16, [dq_t0, h0], qr_bf16_2d)
                    pl.store(rope_bf16, [dq_t0, h0 + IDX_NOPE_HEAD_DIM], qr_bf16_2d)
            else:
                # 不读越过真实T的完整8行；不足8行沿用原逐行数学，合法行逐个写回。
                tail_swap_idx = rope_swap_idx[0:1, :]
                for tail_t0 in pl.range(dq_t0, bs):
                    tail_qr_scale_value = pl.read(qr_scale, [tail_t0, 0])
                    tail_cos_tile = cos[tail_t0 : tail_t0 + 1, 0:ROPE_HEAD_DIM]
                    tail_sin_tile = sin[tail_t0 : tail_t0 + 1, 0:ROPE_HEAD_DIM]
                    for tail_h_inner in pl.pipeline(DQ_ROPE_H_TILE, stage=2):
                        tail_h0 = (hg + tail_h_inner) * IDX_HEAD_DIM
                        tail_wq_scale = pl.reshape(wq_b_scale[tail_h0 : tail_h0 + IDX_HEAD_DIM], [1, IDX_HEAD_DIM])
                        tail_acc_fp32 = pl.cast(
                            qr_acc_pad[tail_t0 : tail_t0 + 1, tail_h0 : tail_h0 + IDX_HEAD_DIM],
                            target_type=pl.FP32,
                            mode="none",
                        )
                        tail_qr_dequant = pl.col_expand_mul(pl.mul(tail_acc_fp32, tail_qr_scale_value), tail_wq_scale)
                        tail_qr_nope_bf16 = pl.cast(
                            tail_qr_dequant[:, 0:IDX_NOPE_HEAD_DIM], target_type=pl.BF16, mode="rint"
                        )
                        tail_qr_rope_slice = tail_qr_dequant[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
                        tail_qr_swapped = pl.gather(tail_qr_rope_slice, dim=-1, index=tail_swap_idx)
                        tail_rope_rot = pl.add(
                            pl.mul(tail_qr_rope_slice, tail_cos_tile), pl.mul(tail_qr_swapped, tail_sin_tile)
                        )
                        tail_rope_bf16 = pl.cast(tail_rope_rot, target_type=pl.BF16, mode="rint")
                        qr_bf16_2d[tail_t0 : tail_t0 + 1, tail_h0 : tail_h0 + IDX_NOPE_HEAD_DIM] = tail_qr_nope_bf16
                        qr_bf16_2d[tail_t0 : tail_t0 + 1, tail_h0 + IDX_NOPE_HEAD_DIM : tail_h0 + IDX_HEAD_DIM] = (
                            tail_rope_bf16
                        )
    return qr_bf16_2d


@pl.jit.inline(auto_scope=False)
def indexer_query_hadamard_stage(
    qr_bf16: pl.Tensor[[ROPE_HEAD_ROWS_DYN, IDX_HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    qh_acc_gm: pl.Tensor[[HADAMARD_ROWS_DYN, IDX_HEAD_DIM], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIC：先等BF16 query所有head；64行一块即一个token，保持原完整K128矩阵乘。"""
    bs_heads = tokens * IDX_N_HEADS
    qh_worker = worker_id
    qh_hadamard = hadamard[0:IDX_HEAD_DIM, 0:IDX_HEAD_DIM]
    for idx in pl.range(qh_worker, bs_heads // QH_MM_TILE, WORKERS):
        o0 = idx * QH_MM_TILE
        qh_acc = pl.matmul(qr_bf16[o0 : o0 + QH_MM_TILE, :], qh_hadamard, out_dtype=pl.FP32)
        qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc
    return qh_acc_gm


@pl.jit.inline(auto_scope=False)
def indexer_query_quant_stage(
    qh_acc_gm: pl.Tensor[[HADAMARD_ROWS_DYN, IDX_HEAD_DIM], pl.FP32],
    qr_hadamard_i8: pl.Tensor[[QUANT_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    qr_hadamard_scale_dq: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV：先等Hadamard FP32，逐64head行归一比例并量化；输出INT8与独立FP32 scale。"""
    bs_heads = tokens * IDX_N_HEADS
    qh_quant_worker = worker_id
    for idx in pl.range(qh_quant_worker, bs_heads // QH_QUANT_TILE, WORKERS):
        o0 = idx * QH_QUANT_TILE
        qh_full_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM]
        qh_full_f32 = pl.mul(qh_full_f32, HADAMARD_SCALE)
        qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_full_f32[:, h0 : h0 + QH_HEAD_DIM_TILE]
            qh_a_neg = pl.neg(qh_a_f32)
            qh_a_abs = pl.maximum(qh_a_f32, qh_a_neg)
            qh_a_max_col = pl.row_max(qh_a_abs)
            qh_a_max = pl.reshape(qh_a_max_col, [1, QH_QUANT_TILE])
            qh_amax = pl.maximum(qh_amax, qh_a_max)
        qh_scale_numerator = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        qh_scale_quant_row = pl.div(qh_scale_numerator, qh_amax)
        qh_scale_recip = pl.recip(qh_scale_quant_row)
        qh_scale_dq = pl.reshape(qh_scale_recip, [QH_QUANT_TILE, 1])
        qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
        qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
        qh_q_scaled = pl.row_expand_mul(qh_full_f32, qh_scale_quant)
        qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
        qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
        qh_i8 = pl.cast(qh_q_half, target_type=pl.INT8, mode="trunc")
        qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, 0:IDX_HEAD_DIM] = qh_i8
    return qr_hadamard_i8, qr_hadamard_scale_dq


@pl.jit.inline(auto_scope=False)
def indexer_weights_project_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    weights_partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIC：等x_normed；M16×四段K1024，每段两个K512原顺序，写动态四份partial。"""
    bs = pl.tensor.dim(x, 0)
    row_blocks = (bs + MM_ROW_TILE - 1) // MM_ROW_TILE
    # partial分段stride是本轮真实T的pad16，不能误用可能更大的allocation容量。
    t_pad = row_blocks * MM_ROW_TILE
    x_flat = x
    w_worker = worker_id
    for w_unit in pl.range(w_worker, WEIGHTS_OK * row_blocks, WORKERS):
        w_rb = w_unit // WEIGHTS_OK
        kb = w_unit - w_rb * WEIGHTS_OK
        w_r0 = w_rb * MM_ROW_TILE
        w_rows = pl.min(MM_ROW_TILE, bs - w_r0)
        k_base = kb * WEIGHTS_K_TILE
        weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
        for db in pl.range(WEIGHTS_K_TILE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(x_flat, [MM_ROW_TILE, D_TILE], [w_r0, d0], valid_shape=[w_rows, D_TILE])
            weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
            weights_acc = pl.matmul_acc(weights_acc, x_tile, weights_proj_tile, init_cond=db == 0)
        weights_partial[kb * t_pad + w_r0 : kb * t_pad + w_r0 + MM_ROW_TILE, :] = weights_acc
    return weights_partial


@pl.jit.inline(auto_scope=False)
def indexer_weights_reduce_stage(
    weights_partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    weights: pl.Tensor[[WEIGHTS_ROWS_DYN, IDX_N_HEADS], pl.FP32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV：等当前M16的四个投影split，原左折叠后乘模型scale，保持FP32给score消费。"""
    row_blocks = (tokens + MM_ROW_TILE - 1) // MM_ROW_TILE
    t_pad = row_blocks * MM_ROW_TILE
    for w_rb in pl.range(worker_id, row_blocks, WORKERS):
        w_r0 = w_rb * MM_ROW_TILE
        w_sum = weights_partial[w_r0 : w_r0 + MM_ROW_TILE, :]
        # 与原核相同的((p0+p1)+p2)+p3；不把四项改成树形归约。
        for kb in pl.unroll(1, WEIGHTS_OK):
            partial_r0 = kb * t_pad + w_r0
            w_sum = pl.add(w_sum, weights_partial[partial_r0 : partial_r0 + MM_ROW_TILE, :])
        weights[w_r0 : w_r0 + MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)
    return weights
