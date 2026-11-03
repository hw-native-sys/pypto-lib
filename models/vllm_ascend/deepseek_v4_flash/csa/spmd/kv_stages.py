# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 的 KV 投影、RMSNorm 与 RoPE：显式分核的核内数学阶段。

输入、输出与布局
    x 是归一化后的 BF16 [T,4096]；wkv 沿用 Native 的 ND BF16 [4096,512]，
    不按全局 NZ 开关改写这张权重。gamma_ckv 直接接 Native BF16 [512]，只在
    AIV 内转 FP32。投影工作区 kv_fp32 为 FP32 [PAD_ROWS,512]，最终 kv 为
    BF16 [T,512]。cos/signed_sin 是 FP32 [T,64]，swap 是 INT32 [T,64]；
    它们必须采用原 interleaved RoPE 的符号和相邻列交换约定：奇偶列的
    sin 符号为 (-1,+1)，swap 为 (1,0,3,2,...)。512 列由前 448 列 NOPE
    和后 64 列 RoPE 组成；这里没有 query 的多 head 维，也不拆 B/S 轴。
    T 与 PAD_ROWS 都保留 pl.dynamic；PAD_ROWS 至少为 ceil(T/16)*16。
    外层可以复用更大容量，但所有循环以真实 T 为界，不能按容量猜有效行。

分块策略、所有权与阅读顺序
    先读 kv_project_stage 的动态分段，再读 _kv_project_worker 的 M/N 分工，
    最后读 kv_norm_rope_stage 的全块/尾块归一化。投影保持原最多 512 行一段，
    每段小于 128 行用 M32，否则 M64；这里判断的是该段实际 tile_rows，
    所以最后一段可以与前面完整 512 行段选不同 M。它是核内区间分块，
    不要求按实值 T 另生成展开代码。尾部 M16，N128，K512。每个 M/N 块
    完整顺序累加 K4096，禁止偷偷切换 split-K atomic 或重新排列归约。
    AIC 逻辑编号是 column_block*m_groups+m_group，由 worker+k*WORKERS
    领取；同一行的四个 N128 列块可能来自不同核，RMS 必须等齐四份列数据。
    AIV 将连续 32 行作为一个独立任务块，owner=(row//32)%WORKERS；一个
    worker 完成该行全部 512 列的 RMS、gamma、NOPE/RoPE，没有跨核 RMS。
    worker_id 是对应角色内的零起点编号，必须满足 0<=worker_id<WORKERS，
    WORKERS 必须为正。无有效块的核跳过数学，外层仍负责其同步参与合同。

阶段数据流、工作区与同步
    外层先确保 x 可读，AIC 写 kv_fp32，然后 AIV 消费投影与已准备的 RoPE。
    本文件只有 inline helper，没有 spmd、submit、task ID、cache 写入或隐式
    核间同步；返回 Tensor 只表达别名。外层负责本核流水完成、GM 发布、通知、
    消费者等待和业务缓存维护，也负责 wkv 的 BYPASS 属性直接设置在 spmd 内。
    一个 worker 的函数返回表示其所有行/列波次已写入同一 GM 别名，不能
    当成其它 worker 已完成；外层只能在本核完成流水并发布后向消费者通知。
    两级 pl.pipeline 只重叠本核搬运与计算，不产生第二个 task 或核间 barrier。
    这里保留每段投影结果的全 T 工作区，不复用 512 行环形槽，避免引入尚未
    建立的消费者 ACK。所有阶段共用同一 token 顺序，各 Tensor 不得非法别名。

尾块与数值边界
    投影尾块读取 valid_rows，完整写回 M16 到 pad16 工作区；AIV 尾块仍分配
    32 行 Tile，但只加载/写回有效行，不假定工作区物理尾部有 32 行或已经清零。
    RMS 先将八个 K64 分片按 lane 相加，再做 row_sum；不能改成每段先求和。
    保留 sqrt→1/div、gamma 乘法、RoPE 两次乘法再相加及最终 BF16 rint。
    只输出 kv[T,512]，尚不写 cache；dummy/请求边界由后续 metadata 阶段处理。
    数学来自相邻 qkv_proj_rope.py；CPU 编译不证明完整动态 CSA 精度或同步。
"""

import pypto.language as pl

from ..common.config import FLASH as M
from ..common.reduction import ATOMIC_ADD

# 只读取模型配置和纯规约开关，避免导入 nz_mode 的三维权重设备格式探针。
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_DIM_SCALE = float(ROPE_DIM)
NOPE_DIM = M.nope_head_dim
EPS = M.rms_norm_eps
PREFILL_DENSE_TILE = 512
KV_M_TILE = 16
KV_DENSE_M_TILE = 64
KV_FIXED_SMALL_M_TILE = 32
KV_FIXED_SMALL_ROWS = 128
KV_N_TILE = 128
KV_K_TILE = 512
KV_OM = 3
KV_RMS_T_TILE = 32
KV_TILE = 64

if ATOMIC_ADD:
    raise ValueError("动态 KV 阶段目前只支持固定 K；不支持 split-K atomic 模式")

T_DYN = pl.dynamic("CSA_KV_T_DYN")
PAD_ROWS_DYN = pl.dynamic("CSA_KV_PAD_ROWS_DYN")


@pl.jit.inline(auto_scope=False)
def _kv_project_worker(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    kv_fp32: pl.Tensor[[PAD_ROWS_DYN, HEAD_DIM], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    DENSE_M: pl.constexpr,
):
    """AIC 处理当前最多 512 行分段，按列块×行组的逻辑编号分配给物理 worker。

    tile_base 是全 T 工作区的全局起点，tile_rows 是本段真实行数；DENSE_M
    只控制本段完整行块。每个 worker 对自己领取的所有输出都走完整 K4096，
    其余行组由其他核负责，消费者必须等待同一行全部四个 N128 生产者。
    """
    t_matmul = (tile_rows + KV_M_TILE - 1) // KV_M_TILE * KV_M_TILE
    kv_full_rows = (tile_rows // DENSE_M) * DENSE_M
    kv_m_groups = pl.min(KV_OM, pl.max(1, tile_rows // DENSE_M))
    # logical 的列块/行组拆分与原固定 K 分支一致；只把 task 编号变成核内波次。
    # 512 同时是 M16/M32/M64 的倍数，下一段起点不会切开前一段的物理尾块。
    for logical in pl.range(worker_id, (HEAD_DIM // KV_N_TILE) * kv_m_groups, WORKERS):
        kv_col0 = (logical // kv_m_groups) * KV_N_TILE
        kv_m_group = logical % kv_m_groups
        # 当前行组以 m_groups 个完整块为步长走自己的条带，不重复覆盖别的组。
        for dense_t0 in pl.range(kv_m_group * DENSE_M, kv_full_rows, kv_m_groups * DENSE_M):
            dense_x0 = tile_base + dense_t0
            dense_acc = pl.create_tensor([DENSE_M, KV_N_TILE], dtype=pl.FP32)
            # 两级搬运流水不改变八次 K512 的累加次序，第一次 matmul_acc 清零。
            for dense_k in pl.pipeline(0, D // KV_K_TILE, stage=2):
                dense_d0 = dense_k * KV_K_TILE
                dense_x = x[dense_x0 : dense_x0 + DENSE_M, dense_d0 : dense_d0 + KV_K_TILE]
                dense_w = wkv[dense_d0 : dense_d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, init_cond=(dense_k == 0))
            kv_fp32 = pl.assemble(kv_fp32, dense_acc, [tile_base + dense_t0, kv_col0])
        # 完整 M32/M64 之后的剩余行继续按 M16 分给相同行组；只有全局最后
        # 一个 M16 块可能不足 16 行。读侧 valid_shape 保留真实行数，写侧可覆盖
        # 调用者分配的 pad16；不能为方便读取而把 padding 当成真实 x。
        for t0 in pl.range(kv_full_rows + kv_m_group * KV_M_TILE, t_matmul, kv_m_groups * KV_M_TILE):
            kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(D // KV_K_TILE, stage=2):
                d0 = db * KV_K_TILE
                kv_rows = pl.min(KV_M_TILE, tile_rows - t0)
                x_t0 = tile_base + t0
                kv_x_chunk_bf16 = pl.slice(
                    x,
                    [KV_M_TILE, KV_K_TILE],
                    [x_t0, d0],
                    valid_shape=[kv_rows, KV_K_TILE],
                )
                wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk, init_cond=(db == 0))
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [tile_base + t0, kv_col0])
    return kv_fp32


@pl.jit.inline(auto_scope=False)
def kv_project_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    kv_fp32: pl.Tensor[[PAD_ROWS_DYN, HEAD_DIM], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """按实际 T 分段并选择原 M32/M64 策略，所有波次仍在同一 incore 内执行。"""
    tokens = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, tokens, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, tokens - tile_base)
        if tile_rows < KV_FIXED_SMALL_ROWS:
            kv_fp32 = _kv_project_worker(
                x, wkv, kv_fp32, tile_base, tile_rows, worker_id, WORKERS, KV_FIXED_SMALL_M_TILE
            )
        else:
            kv_fp32 = _kv_project_worker(x, wkv, kv_fp32, tile_base, tile_rows, worker_id, WORKERS, KV_DENSE_M_TILE)
    return kv_fp32


@pl.jit.inline(auto_scope=False)
def kv_norm_rope_stage(
    kv_fp32: pl.Tensor[[PAD_ROWS_DYN, HEAD_DIM], pl.FP32],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV 独占 32 行的全部 512 列；调用前由外层等齐投影和 RoPE 输入。

    本 helper 内没有 wait。完整块先对 512 列做同一 RMS，再分别处理 NOPE
    和 RoPE；尾块走显式 load/store 的有效行路径，不访问 pad16 之外的 GM。
    """
    tokens = pl.tensor.dim(kv, 0)
    token_tiles = (tokens + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
    # 原512行分段边界与32行tile对齐，因此扁平化此循环不改变完整/尾块数学。
    # 这是worker条带循环，不派发任务；空T和未分到tile的核不读gamma/RoPE。
    for tg_idx in pl.range(worker_id, token_tiles, WORKERS):
        tg = tg_idx * KV_RMS_T_TILE
        valid_rows = pl.min(KV_RMS_T_TILE, tokens - tg)
        if valid_rows == KV_RMS_T_TILE:
            # 八段64列先按lane累加，再一次row_sum，保持Native的归约树。
            kv_sq_lanes = pl.full([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
            for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
                kv_sq = pl.mul(kv_chunk, kv_chunk)
                kv_sq_lanes = pl.add(kv_sq_lanes, kv_sq)
            kv_sq_sum = pl.reshape(pl.row_sum(kv_sq_lanes), [1, KV_RMS_T_TILE])
            # 保留sqrt后用1除，不替换为rsqrt/recip，以免移动舍入边界。
            kv_rms = pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS))
            kv_inv_rms = pl.div(pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_rms)
            kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])
            for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
                gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
                gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
                kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
                kv[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16
            gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
            gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
            kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
            kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
            kv_cos_il_full = rope_cos_il[tg : tg + KV_RMS_T_TILE, :]
            kv_sin_signed_full = rope_sin_signed[tg : tg + KV_RMS_T_TILE, :]
            kv_swap_idx_full = rope_swap_idx[tg : tg + KV_RMS_T_TILE, :]
            kv_swapped_full = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx_full)
            kv_rope_rot_full = pl.add(
                pl.mul(kv_rope_norm_chunk, kv_cos_il_full), pl.mul(kv_swapped_full, kv_sin_signed_full)
            )
            kv_rope_i16_full = pl.cast(kv_rope_rot_full, target_type=pl.BF16, mode="rint")
            kv[tg : tg + KV_RMS_T_TILE, NOPE_DIM:HEAD_DIM] = kv_rope_i16_full
        else:
            # 最后不足 32 行时，pad16 可能只剩 16 行，不能直接读完整 32 行
            # GM 窗口。每次 load/store 都带 valid_rows；无效 lane 的中间数值
            # 不属于模型结果，也不能用来推断 padding 已经初始化。
            kv_reduce_tmp = pl.create_tile([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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
            kv_inv_rms_tail = pl.div(pl.tile.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_rms_tail)
            kv_inv_rms_t_tail = pl.reshape(kv_inv_rms_tail, [KV_RMS_T_TILE, 1])
            for n0_tail in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                kv_chunk_tail = pl.load(
                    kv_fp32,
                    [tg, n0_tail],
                    [KV_RMS_T_TILE, KV_TILE],
                    valid_shape=[valid_rows, KV_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                gamma_kv_input_tail = pl.load(gamma_ckv, [n0_tail], [KV_TILE], target_memory=pl.MemorySpace.Vec)
                gamma_kv_cast_tail = pl.cast(gamma_kv_input_tail, target_type=pl.FP32)
                gamma_kv_chunk_tail = pl.reshape(gamma_kv_cast_tail, [1, KV_TILE])
                kv_normed_tail = pl.col_expand_mul(
                    pl.row_expand_mul(kv_chunk_tail, kv_inv_rms_t_tail), gamma_kv_chunk_tail
                )
                kv_normed_bf16_tail = pl.cast(kv_normed_tail, target_type=pl.BF16, mode="rint")
                kv_normed_valid = pl.set_validshape(kv_normed_bf16_tail, valid_rows, KV_TILE)
                pl.store(kv_normed_valid, [tg, n0_tail], kv)
            gamma_rope_input_tail = pl.load(gamma_ckv, [NOPE_DIM], [ROPE_DIM], target_memory=pl.MemorySpace.Vec)
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
                pl.row_expand_mul(kv_rope_chunk_tail, kv_inv_rms_t_tail), gamma_rope_tail
            )
            kv_cos_il_tail = pl.load(
                rope_cos_il,
                [tg, 0],
                [KV_RMS_T_TILE, ROPE_DIM],
                valid_shape=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            kv_sin_signed_tail = pl.load(
                rope_sin_signed,
                [tg, 0],
                [KV_RMS_T_TILE, ROPE_DIM],
                valid_shape=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            kv_col = pl.col_expand_mul(
                pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
            )
            # pair=floor(col/2)，相邻交换为(1,0,3,2,...)；只改变索引，不改RoPE算术。
            kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
            kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))
            kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))
            # tile.gather按整个Tile展开寻址，补上每行64列偏移，防止跨行取错。
            kv_row_seed = pl.mul(
                pl.cast(pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32), target_type=pl.FP32), ROPE_DIM_SCALE
            )
            kv_row_grid = pl.col_expand_mul(
                pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_row_seed
            )
            kv_row_offset = pl.transpose(kv_row_grid, axis1=0, axis2=1)
            kv_swap_idx_tail = pl.cast(pl.add(kv_swap_f, kv_row_offset), target_type=pl.INT32)
            kv_gather_tmp = pl.create_tile([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            kv_swapped_tail = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
            kv_rope_rot_tail = pl.add(
                pl.mul(kv_rope_norm_tail, kv_cos_il_tail), pl.mul(kv_swapped_tail, kv_sin_signed_tail)
            )
            # BF16舍入仍在RoPE两项乘加之后，只有有效行写回最终kv。
            kv_rope_i16_tail = pl.cast(kv_rope_rot_tail, target_type=pl.BF16, mode="rint")
            kv_rope_valid = pl.set_validshape(kv_rope_i16_tail, valid_rows, ROPE_DIM)
            pl.store(kv_rope_valid, [tg, NOPE_DIM], kv)
    return kv
