# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""动态 CSA 的 Q_B 投影、query 反量化/RMS/RoPE 与 RoPE 准备阶段。

阅读顺序
    先看 q_b_stage 所选择的 ND/NZ AIC 投影，再看可与投影并行的
    q_rope_prepare_stage，最后看消费二者的 AIV q_dequant_rope_stage。
    文件把较长的反量化数学放在中间；这不表示要求先执行它再准备 RoPE。

输入输出和动态变量
    Q_B 读取 QR INT8 暂存与 Native INT8 wq_b[1024,32768]，输出 INT32 投影。
    tokens 是外层从真实输入 Tensor 维度取得的设备 Scalar，不是 Host 固定 B。
    反量化阶段接收每行 QR scale、每列权重 scale、cos/sin/swap，写 BF16
    q_flat[T,32768]；它必须与下游 q[T,64,512] 共用同一块 GM。
    T、QR 暂存行数和投影工作区行数均为动态 Tensor 轴。

数学 tile 与分工
    NZ 保留 M128/N256/K256，ND 保留 M64/N512/K128；按实际 T 计算
    ceil(T/128)*128 个工作区行，不按缓存容量扫描额外行。每个显式 AIC
    worker 负责列块 n=worker+k*WORKERS，沿 K 顺序做原 INT32 累加。
    反量化用八行、4/8/16/32/64 个独立 head 的分组策略；完整八行时选
    最大且能形成48-AIV整波的组，尾行固定4。WORKERS只决定条带owner，
    每个head的512列数学不变；RoPE准备仍按八行条带。

数据流、同步与所有权
    三个 helper 内没有 spmd/submit/notify。外层须等 QR 的 INT8 写回再运行
    Q_B；反量化同时依赖相关 Q_B 列块、QR scale 和 RoPE 准备写回。数据
    fence、通知和等待由统一 SPMD 编排负责，返回 Tensor 不代表跨核完成。
    Q_B 列块 owner=(column//N_TILE)%WORKERS。反量化的八行/head 组编号
    work=(row//8)*(64/head_tile)+head//head_tile，其 owner=work%WORKERS；
    每个 head 完整包含两个 NZ N256 列块。RoPE owner=(row//8)%active，
    active=min(ceil(T/8),WORKERS)。各阶段只写自己负责的独立行/列片段。

GM 视图与工作区合同
    q_flat 是外层创建的无损二维 GM 别名；不得在 incore helper 内把三维 GM
    reshape 后当作另一个 GM Tensor 再 load/store。QR 和投影暂存至少预留
    ceil(T/128)*128 行，scale 暂存至少 ceil(T/8)*8 行；有效输出只写真实 T。
    ND 会计算暂存的填充行，这些行不参与有效输出；未写容量尾部不保证清零。
    外层须直接在 spmd 体设置 wq_b 的 BYPASS 属性，并验证工作区不互相覆盖。

数值与范围边界
    数学来自同仓 q_projection.py/qkv_proj_rope.py，不使用冻结 IR。保留
    (INT32→FP32 * activation_scale) * weight_scale、512 列规约和原 RoPE
    运算顺序；完整八行用 rsqrt，尾行沿用 recip(sqrt)，不混成另一套公式。
    NZ 仅按 Native 同一模式声明，不转换权重、不执行三维权重格式探针。
    本文件只补齐 query 分支，不包含 KV、Indexer、Attention 或完整 CSA；
    源码/CPU 编译检查不能代替真实设备精度、跨核同步或性能验证。
"""

import pypto.language as pl

from ..common.options import weight_nz_mode

from ..common.config import FLASH as M

WEIGHT_NZ_MODE = weight_nz_mode()
if WEIGHT_NZ_MODE not in (0, 1, 2):
    raise ValueError("VLLM_ASCEND_ENABLE_NZ 只接受 0/1/2")
QUANT_WEIGHT_NZ = WEIGHT_NZ_MODE >= 1
QUANT_WEIGHT_LAYOUT = pl.NZ if QUANT_WEIGHT_NZ else None

H = M.num_attention_heads
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
ROPE_DIM = M.qk_rope_head_dim
ROPE_DIM_SCALE = float(ROPE_DIM)
NOPE_DIM = HEAD_DIM - ROPE_DIM
EPS = M.rms_norm_eps
Q_PROJ_TILE = 128
QPROJ_MM_N_TILE = 256 if QUANT_WEIGHT_NZ else 512
QPROJ_M_TILE = 64
QPROJ_N_BLOCKS = H * HEAD_DIM // QPROJ_MM_N_TILE
QPROJ_PIPE_M_TILE = 128
QPROJ_PIPE_K_TILE = 256
Q_ROPE_T_TILE = 8
Q_ROPE_H_TILE = 4
Q_DEQUANT_WORKERS = 48

T_DYN = pl.dynamic("CSA_QUERY_T_DYN")
PAD_ROWS_DYN = pl.dynamic("CSA_QUERY_PAD_ROWS_DYN")
QPROJ_ROWS_DYN = pl.dynamic("CSA_QUERY_PROJ_ROWS_DYN")


@pl.jit.inline(auto_scope=False)
def _q_b_nd_stage(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    qr_i8_matmul: pl.Tensor[[PAD_ROWS_DYN, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """ND Q_B：显式 AIC 按列块条带领取，原 K 流水不变；调用前必须等待 QR。"""
    qproj_worker = pl.max(worker_id, 0)
    qproj_t_matmul = (tokens + QPROJ_PIPE_M_TILE - 1) // QPROJ_PIPE_M_TILE * QPROJ_PIPE_M_TILE
    qproj_full_rows = qproj_t_matmul
    # 列块条带保证每个输出列唯一归属，worker 不必与列块数量一一对应。
    for qproj_n_idx in pl.range(qproj_worker, QPROJ_N_BLOCKS, WORKERS):
        w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
        # ND 沿用填充行路径，外层需保证 QR 暂存至少按 M128 容纳完整盒子。
        for t0 in pl.range(0, qproj_full_rows, QPROJ_M_TILE):
            col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
            # 双缓冲只重叠搬运与计算，K128 的逐段累加和首次清零条件不变。
            for qr_proj_col0 in pl.pipeline(0, Q_LORA, Q_PROJ_TILE, stage=2):
                qr_i8_chunk = qr_i8_matmul[t0 : t0 + QPROJ_M_TILE, qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE]
                wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk, init_cond=qr_proj_col0 == 0)
            q_proj_i32[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc
    return q_proj_i32


@pl.jit.inline(auto_scope=False)
def _q_b_nz_stage(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    qr_i8_matmul: pl.Tensor[[PAD_ROWS_DYN, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """NZ Q_B：显式 AIC 按列块条带领取，原 K 流水不变；调用前必须等待 QR。"""
    qproj_worker = pl.max(worker_id, 0)
    qproj_t_matmul = (tokens + QPROJ_PIPE_M_TILE - 1) // QPROJ_PIPE_M_TILE * QPROJ_PIPE_M_TILE
    # 由实际 tokens 决定最后一个 M128，不能用较大的工作区容量延长循环。
    for qproj_round in pl.range(0, (QPROJ_N_BLOCKS - qproj_worker + WORKERS - 1) // WORKERS):
        # 用非负 worker+round*WORKERS 表达 NZ 列偏移，保留编译器的非负证明。
        qproj_n_idx = qproj_worker + qproj_round * WORKERS
        w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
        for t0 in pl.range(0, qproj_t_matmul, QPROJ_PIPE_M_TILE):
            # 最后一盒不足128行时，用实际有效行约束载入及矩阵计算。
            m_rows = pl.min(QPROJ_PIPE_M_TILE, tokens - t0)
            qr_first = pl.slice(
                qr_i8_matmul, [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE], [t0, 0], valid_shape=[m_rows, QPROJ_PIPE_K_TILE]
            )
            wq_first = wq_b[0:QPROJ_PIPE_K_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
            # 第一段 matmul 初始化 INT32 累加器，后三段按 K256 原顺序追加。
            col_acc = pl.matmul(qr_first, wq_first, out_dtype=pl.INT32)
            for k0 in pl.pipeline(QPROJ_PIPE_K_TILE, Q_LORA, QPROJ_PIPE_K_TILE, stage=2):
                qr_chunk = pl.slice(
                    qr_i8_matmul,
                    [QPROJ_PIPE_M_TILE, QPROJ_PIPE_K_TILE],
                    [t0, k0],
                    valid_shape=[m_rows, QPROJ_PIPE_K_TILE],
                )
                wq_chunk = wq_b[k0 : k0 + QPROJ_PIPE_K_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                col_acc = pl.matmul_acc(col_acc, qr_chunk, wq_chunk)
            q_proj_i32[t0 : t0 + QPROJ_PIPE_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc
    return q_proj_i32


@pl.jit.inline(auto_scope=False)
def q_dequant_rope_stage(
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q_flat: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    qr_scale_pad_store: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    q_proj_i32: pl.Tensor[[QPROJ_ROWS_DYN, H * HEAD_DIM], pl.INT32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """显式 AIV 反量化、RMS、NOPE/RoPE；q_flat 必须是外层建立的 GM 二维别名。

    完整八行与尾块沿用不同的既有 RMS 算术；每次工作只写八行中的一个 head 组。"""
    tokens = pl.tensor.dim(q_flat, 0)
    dq_token_tiles = (tokens + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
    dq_head_tile = Q_ROPE_H_TILE
    # 只合并完整八行块的独立 head：选择最大的 64/32/16/8，使
    # dq_token_tiles*(H/head_tile) 恰为正的 48-AIV 整波；不满足则保留4。
    # 扩组后每个AIV的head迭代数不增，只减少scale/cos/sin/swap的重复准备。
    # 此条件取实际设备T，不用B/S或测试点；WORKERS仍只决定任务条带。
    # 有尾行必须留4，保持下面固定4-head尾循环及原recip(sqrt)数学。
    if tokens % Q_ROPE_T_TILE == 0:
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
    # 将“八行块、head组”展平成任务号，worker只处理自己的条带，避免重复写。
    for dq_work in pl.range(worker_id, (tokens + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE * (H // dq_head_tile), WORKERS):
        hg = dq_work % (H // dq_head_tile) * dq_head_tile
        tg = dq_work // (H // dq_head_tile) * Q_ROPE_T_TILE
        if tg + Q_ROPE_T_TILE <= tokens:
            qr_scale_dq_t = pl.load(qr_scale_pad_store, [tg, 0], [Q_ROPE_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
            q_cos_il = pl.load(rope_cos_il, [tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
            q_sin_signed = pl.load(
                rope_sin_signed, [tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec
            )
            q_swap_idx = pl.load(rope_swap_idx, [tg, 0], [Q_ROPE_T_TILE, ROPE_DIM], target_memory=pl.MemorySpace.Vec)
            q_flat_i = pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32), pl.FP32)
            q_row_i = pl.cast(pl.mul(q_flat_i, 1.0 / ROPE_DIM), pl.INT32, mode="trunc")
            q_row_starts = pl.mul(pl.cast(q_row_i, pl.FP32), ROPE_DIM_SCALE)
            q_abs_swap = pl.cast(
                pl.add(pl.cast(pl.reshape(q_swap_idx, [1, Q_ROPE_T_TILE * ROPE_DIM]), pl.FP32), q_row_starts), pl.INT32
            )
            q_flat_gather_tmp = pl.tile.create([1, Q_ROPE_T_TILE * ROPE_DIM], dtype=pl.INT32)
            # head之间数学独立；pipeline复用八行临时空间，逐head保持512列规约。
            for h_inner in pl.pipeline(dq_head_tile, stage=2):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_head_acc = pl.load(q_proj_i32, [tg, h0], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
                q_head_scale = pl.reshape(
                    pl.load(wq_b_scale, [h0], [HEAD_DIM], target_memory=pl.MemorySpace.Vec), [1, HEAD_DIM]
                )
                q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                # 先激活 scale、后权重 scale；不能合并或交换两次 FP32 乘法。
                q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                q_head_sq = pl.mul(q_head_dq, q_head_dq)
                q_head_sq_row = pl.row_sum(q_head_sq, tmp_tile=pl.tile.create([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32))
                q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                q_head_var = pl.add(q_head_sq_mean, EPS)
                q_head_inv_rms = pl.tile.rsqrt(q_head_var, pl.tile.create([1, Q_ROPE_T_TILE], dtype=pl.FP32))
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])
                q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                pl.store(q_nope_bf16, [tg, h0], q_flat)
                q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                q_rope_flat = pl.reshape(q_rope_chunk, [1, Q_ROPE_T_TILE * ROPE_DIM])
                q_swapped_flat = pl.tile.gather(q_rope_flat, q_abs_swap, q_flat_gather_tmp)
                q_rope_swapped = pl.reshape(q_swapped_flat, [Q_ROPE_T_TILE, ROPE_DIM])
                q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                pl.store(q_rope_bf16, [tg, h0 + NOPE_DIM], q_flat)
        else:
            valid_tail_rows = tokens - tg
            # 原尾块使用八行暂存计算、仅写有效行。这个分支保留 recip(sqrt)，
            # 不因完整块采用 rsqrt 就把两条数值路径强行统一。
            qr_scale_dq_tail = pl.load(
                qr_scale_pad_store, [tg, 0], [Q_ROPE_T_TILE, 1], target_memory=pl.MemorySpace.Vec
            )
            q_cos_il_tail = pl.load(
                rope_cos_il,
                [tg, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shape=[valid_tail_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            q_sin_signed_tail = pl.load(
                rope_sin_signed,
                [tg, 0],
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
            q_row_seed = pl.mul(
                pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), target_type=pl.FP32), ROPE_DIM_SCALE
            )
            q_row_grid = pl.col_expand_mul(
                pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0), q_row_seed
            )
            q_row_offset = pl.transpose(q_row_grid, axis1=0, axis2=1)
            q_swap_idx_tail = pl.cast(pl.add(q_swap_f, q_row_offset), target_type=pl.INT32)
            q_head_reduce_tmp = pl.create_tile(
                [Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            q_gather_tmp = pl.create_tile([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            # 扩组分支显式要求T整除8；有尾行必回退4。此处继续固定4，
            # 不改变原尾块的有效写回和recip(sqrt)路径。
            for h_inner_tail in pl.range(Q_ROPE_H_TILE):
                h_tail = hg + h_inner_tail
                h0_tail = h_tail * HEAD_DIM
                q_head_acc_tail = pl.load(
                    q_proj_i32, [tg, h0_tail], [Q_ROPE_T_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec
                )
                q_head_scale_input_tail = pl.load(wq_b_scale, [h0_tail], [HEAD_DIM], target_memory=pl.MemorySpace.Vec)
                q_head_scale_tail = pl.reshape(q_head_scale_input_tail, [1, HEAD_DIM])
                q_head_acc_fp32_tail = pl.cast(q_head_acc_tail, target_type=pl.FP32, mode="none")
                q_head_row_scaled_tail = pl.row_expand_mul(q_head_acc_fp32_tail, qr_scale_dq_tail)
                q_head_dq_tail = pl.col_expand_mul(q_head_row_scaled_tail, q_head_scale_tail)
                q_head_sq_tail = pl.mul(q_head_dq_tail, q_head_dq_tail)
                q_head_sq_sum_tail = pl.row_sum(q_head_sq_tail, q_head_reduce_tmp)
                # 原尾行是sqrt后recip，完整块是rsqrt；保留差异以免改变舍入边界。
                q_head_inv_rms_tail = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum_tail, 1.0 / HEAD_DIM), EPS)))
                q_nope_normed_tail = pl.row_expand_mul(q_head_dq_tail[:, 0:NOPE_DIM], q_head_inv_rms_tail)
                q_nope_bf16_tail = pl.cast(q_nope_normed_tail, target_type=pl.BF16, mode="rint")
                q_nope_valid = pl.set_validshape(q_nope_bf16_tail, valid_tail_rows, NOPE_DIM)
                pl.store(q_nope_valid, [tg, h0_tail], q_flat)
                q_rope_chunk_raw_tail = q_head_dq_tail[:, NOPE_DIM:HEAD_DIM]
                q_rope_chunk_tail = pl.row_expand_mul(q_rope_chunk_raw_tail, q_head_inv_rms_tail)
                q_rope_swapped_tail = pl.tile.gather(q_rope_chunk_tail, q_swap_idx_tail, q_gather_tmp)
                q_rope_chunk_tail = pl.set_validshape(q_rope_chunk_tail, valid_tail_rows, ROPE_DIM)
                q_rope_swapped_tail = pl.set_validshape(q_rope_swapped_tail, valid_tail_rows, ROPE_DIM)
                q_rope_base_tail = pl.mul(q_rope_chunk_tail, q_cos_il_tail)
                q_rope_delta_tail = pl.mul(q_rope_swapped_tail, q_sin_signed_tail)
                q_rope_rot_tail = pl.add(q_rope_base_tail, q_rope_delta_tail)
                q_rope_bf16_tail = pl.cast(q_rope_rot_tail, target_type=pl.BF16, mode="rint")
                q_rope_valid = pl.set_validshape(q_rope_bf16_tail, valid_tail_rows, ROPE_DIM)
                pl.store(q_rope_valid, [tg, h0_tail + NOPE_DIM], q_flat)
    return q_flat


@pl.jit.inline(auto_scope=False)
def q_rope_prepare_stage(
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """显式 AIV 构造 interleaved sin 符号与行内交换索引；只处理有效 token tile。

    这段只依赖本轮 RoPE 输入，可与 QR/Q_B 并行；反量化消费前必须等待发布。"""
    t_dim = pl.tensor.dim(rope_sin, 0)
    token_tiles = (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
    if worker_id < pl.min(WORKERS, token_tiles):
        # 行内交换和符号只依赖列号，每个有效 worker 准备一次，再复用到各行块。
        qrp_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        # trunc(col/2)取得pair号，再求奇偶lane：sign=(-1,+1)，swap=(1,0,3,2,...)。
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        # active由实际T决定，外层if同时避免空T进入步长为零的循环。
        for qrp_idx in pl.range(worker_id, token_tiles, pl.min(WORKERS, token_tiles)):
            qrp_t0 = qrp_idx * Q_ROPE_T_TILE
            qrp_valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - qrp_t0)
            if qrp_valid_rows == Q_ROPE_T_TILE:
                qrp_sin_rows_full = rope_sin[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
                qrp_sin_signed_full = pl.mul(qrp_sin_rows_full, qrp_sign)
                rope_sin_signed[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_sin_signed_full
                rope_swap_idx[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_swap_idx
            else:
                # 最后不足八行时，只加载/写回有效行；列内符号和交换规则保持不变。
                qrp_sin_rows_tail = pl.load(
                    rope_sin,
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
                    pl.cast(pl.mul(qrp_tail_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32
                )
                qrp_tail_lane = pl.sub(qrp_tail_col, pl.mul(qrp_tail_dup_f, 2.0))
                qrp_tail_swap_f = pl.sub(pl.add(qrp_tail_col, 1.0), pl.mul(qrp_tail_lane, 2.0))
                qrp_tail_sign = pl.sub(pl.mul(qrp_tail_lane, 2.0), 1.0)
                qrp_tail_sign = pl.set_validshape(qrp_tail_sign, qrp_valid_rows, ROPE_DIM)
                qrp_sin_signed_tail = pl.mul(qrp_sin_rows_tail, qrp_tail_sign)
                pl.store(pl.set_validshape(qrp_sin_signed_tail, qrp_valid_rows, ROPE_DIM), [qrp_t0, 0], rope_sin_signed)
                pl.store(
                    pl.set_validshape(pl.cast(qrp_tail_swap_f, target_type=pl.INT32), qrp_valid_rows, ROPE_DIM),
                    [qrp_t0, 0],
                    rope_swap_idx,
                )
    return (rope_sin_signed, rope_swap_idx)


# 布局在进程配置时确定，避免同一 JIT 函数解析互斥的 ND/NZ 数学分支。
q_b_stage = _q_b_nz_stage if QUANT_WEIGHT_NZ else _q_b_nd_stage
