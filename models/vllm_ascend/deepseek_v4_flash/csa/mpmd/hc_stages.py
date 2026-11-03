# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""完整动态 CSA 的 HC 核内阶段；保留原算术，不创建独立任务。

输入输出与动态布局
    输入为 Native BF16 残差 x[T,4,4096] 的二维连续视图 [T,16384]，以及
    FP32 HC 权重/scale/base、BF16 norm 权重。输出是 post[T,4]、comb[T,16]
    和归一化输入 x_normed[T,4096]；后续 HCpost 仍需要 post 与 comb。
    T 从 Tensor 动态轴读取，不假定 B×固定 S。PAD16=ceil(T/16)*16，
    PAD8=ceil(T/8)*8 只是中间容量，不能用它们越界读取原始 T 行。
    所有整块 GM reshape 均由外层绑定描述符；本核只 reshape 已加载的 Tile。

整体数据流与阅读顺序
    workspace_contract 先列容量，然后按以下六段阅读：
    1. hc_widen_rms_range：AIV 加宽 x，并累加输入 RMS，发布 HC_INPUT。
    2. hc_project_range：AIC 做四份 K4096 投影，发布 HC_PARTIAL。
    3. hc_reduce_range：AIV 按原次序合并四份，发布 HC_RAW。
    4. hc_gates_range：AIV 计算 pre/post 门值，发布 HC_GATES。
    5. hc_comb_range：AIV 计算 4×4 Sinkhorn，发布 HC_COMB。
    6. hc_mix_norm_range：AIV 按 pre 混合四路残差并归一化，发布 HC_NORM。
    第 4、5 段共享 HC_RAW，但互不依赖；第 6 段只等 HC_GATES 和输入。
    “发布”均由外层执行，helper 返回本身不代表数据已经能被别的核读取。

分核、工作区和同步
    worker 通过 logical=worker+wave*workers 领取动态波次。Vector 的八行
    tile、归约的十六行 tile 各只有一个 owner；AIC 投影按 (十六行块,K分片)
    分工，同一行会有四个生产者。partials 的 GM 行是 split*PAD16+token，
    归约必须等四份实际 owner。inv_rms/raw/pre 用 PAD16；mixed 用 PAD8。
    最后不足八行的块是唯一 tail scratch 写者，修改分工时须重审这个条件。
    本文件没有 pl.spmd、submit 或核间通知；完整 CSA 调度层负责本核流水
    完成、GM 发布、消费者等待与缓存维护，不把每个阶段变成全核屏障。

当前策略与验证边界
    首版保留六个数学阶段及 GM 中间量；固定的 8/16 行 tile、四份 K 拆分是
    现有数学策略，不是 batch 白名单。源码不展开固定 B16 的 IR；尚未在这里
    实施基于测量的新分档，也没有声明尾段融合或性能提升。
    数学来自相邻 deepseek_v4_flash_csa/{decode_csa,hc_pre,rmsnorm}.py，
    保持 FP32 投影输入、BF16 norm 权重及原舍入点。CPU 源码/编译证据不能
    替代设备数值、尾块、核间同步或完整 CSA 验证。

    HC_CONSUMER_REDUCE候选借鉴独立hc_pre的消费者就地归约：门控和Sinkhorn
    直接从四份partials按原左折叠顺序读/加所需列，不再经独立RAW阶段交接。
    两消费者按M8行分工，外层必须先等该行块四个AIC生产者；inv仍由同一AIV
    的widen阶段产生。只删RAW写回/重读/通知，不更改投影、RMS或Sinkhorn数学。
    raw容量暂留供同进程开关消融，开启时不读写其数值。两档完整对照精确
    通过，但128K/B16慢1.06%、8K/B32快2.54%；默认关闭，区间收益尚待确认。
"""

import pypto.language as pl

from ..common.rmsnorm import rms_norm_apply, rms_norm_inverse

T_DYN = pl.dynamic("HC_STAGE_T")
PAD_ROWS_DYN = pl.dynamic("HC_STAGE_PAD16_ROWS")
PARTIAL_ROWS_DYN = pl.dynamic("HC_STAGE_PARTIAL_ROWS")
MIX_ROWS_DYN = pl.dynamic("HC_STAGE_PAD8_ROWS")

# DSV4 Flash 模型维度及已验证数学分块；这些不是 batch/sequence 白名单。
D = 4096
HC_MULT = 4
HC_DIM = HC_MULT * D
HC_DIM_INV = 1.0 / HC_DIM
MIX_HC = 24
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
COMB_T_TILE = T_TILE
LINEAR_T_TILE = 16
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_K_TILE = 256
RMS_K_TILE = 512
D_TILE = 256
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6
NORM_EPS = 1e-6


def workspace_contract(tokens):
    """Host 只按已有 shape 描述分配空间，不读取设备元素。

    tokens 是本次实际 T；若使用容量缓冲，应传下列实际形状的连续子视图。
    group_workers/vector_workers 必须为正，worker 为零起点且小于对应数量。
    所有核按相同分块公式分工；Vector 阶段的一个行块只归属一个 worker，
    AIC 投影的唯一 owner 单位则是 (行块,K分片)，同一行对应四份部分和。
    四份 tail scratch 各仅被全局唯一的最后短块写入，不能用于改变行分工后
    多个 worker 同时拥有短块的方案；并发执行整层必须拥有独立工作区。
    """
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("HC 工作区要求正整数实际 token 数")
    padded16 = (tokens + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    padded8 = (tokens + T_TILE - 1) // T_TILE * T_TILE
    return {
        "x_fp32": {"shape": [tokens, HC_DIM], "dtype": "FP32"},
        "inv_rms": {"shape": [padded16, 1], "dtype": "FP32"},
        "widen_tail": {"shape": [T_TILE, HC_DIM], "dtype": "FP32"},
        "mixes_partials": {"shape": [LINEAR_OK * padded16, MIX_PAD], "dtype": "FP32"},
        "mixes_raw": {"shape": [padded16, MIX_PAD], "dtype": "FP32"},
        "pre_val_store": {"shape": [padded16, HC_PAD], "dtype": "FP32"},
        "post_tail_store": {"shape": [T_TILE, HC_PAD], "dtype": "FP32"},
        "comb_tail_store": {"shape": [COMB_T_TILE, HC_PAD * HC_MULT], "dtype": "FP32"},
        "x_mixed": {"shape": [padded8, D], "dtype": "BF16"},
        "x_normed_tail_store": {"shape": [T_TILE, D], "dtype": "BF16"},
    }


@pl.jit.inline(auto_scope=False)
def hc_widen_rms_range(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],
    x_fp32: pl.Out[pl.Tensor[[T_DYN, HC_DIM], pl.FP32]],
    inv_rms: pl.Out[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[T_TILE, HC_DIM], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
):
    """AIV：BF16 无损扩展并按原 32 个 K512 顺序累加输入平方和。

    调用方在本核返回后发布 HC_INPUT；AIC 投影及后续门值消费者等待所需行。
    尾行显式恢复 valid_shape 并清零后归约，避免 cast 丢失有效行标记。"""
    t_dim = pl.tensor.dim(x, 0)
    x_hc_flat = x
    x_hc32_flat = x_fp32
    widen_rows = (t_dim + T_TILE - 1) // T_TILE
    # 固定 worker 数只决定条带间隔；ceil 计算动态波数，末波用 guard 排除空核。
    for wave in pl.range((widen_rows + vector_workers - 1) // vector_workers):
        widen_blk = vector_worker + wave * vector_workers
        if widen_blk < widen_rows:
            w_t0 = widen_blk * T_TILE
            w_rows = pl.min(T_TILE, t_dim - w_t0)
            w_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            # stage=4 让搬运与 Vector 运算流水化，不拆开同一行的平方和累加链。
            # 32 个 K512 依次累加，不能为了并核改变归约树与浮点舍入。
            for w_db in pl.pipeline(HC_MULT * D // RMS_K_TILE, stage=4):
                w_d0 = w_db * RMS_K_TILE
                w_src = pl.slice(x_hc_flat, [T_TILE, RMS_K_TILE], [w_t0, w_d0], valid_shape=[w_rows, RMS_K_TILE])
                w_val = pl.cast(w_src, pl.FP32)
                if w_rows == T_TILE:
                    x_hc32_flat[w_t0 : w_t0 + T_TILE, w_d0 : w_d0 + RMS_K_TILE] = w_val
                    w_sq = pl.mul(w_val, w_val)
                    w_sq_row = pl.reshape(pl.row_sum(w_sq), [1, T_TILE])
                    w_sq_sum = pl.add(w_sq_sum, w_sq_row)
                else:
                    # cast 后重新声明有效行并清零填充，防止无效行污染归约。
                    # 写实际 T 的 x_fp32 时借唯一尾块 scratch 恢复有效 store 边界。
                    w_valid = pl.set_validshape(w_val, w_rows, RMS_K_TILE)
                    w_clean = pl.fillpad(w_valid, pad_value=pl.PadValue.zero)
                    w_sq_tail = pl.mul(w_clean, w_clean)
                    w_sq_row_tail = pl.reshape(pl.row_sum(w_sq_tail), [1, T_TILE])
                    w_sq_sum = pl.add(w_sq_sum, w_sq_row_tail)
                    widen_tail[0:T_TILE, w_d0 : w_d0 + RMS_K_TILE] = w_val
                    w_out = pl.load(
                        widen_tail,
                        [0, w_d0],
                        [T_TILE, RMS_K_TILE],
                        valid_shape=[w_rows, RMS_K_TILE],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    pl.store(w_out, [w_t0, w_d0], x_hc32_flat)
            w_mean = pl.add(pl.mul(w_sq_sum, HC_DIM_INV), NORM_EPS)
            w_inv = pl.reshape(pl.rsqrt(w_mean, high_precision=True), [T_TILE, 1])
            inv_rms[w_t0 : w_t0 + T_TILE, 0:1] = w_inv
    return x_fp32, inv_rms


@pl.jit.inline(auto_scope=False)
def hc_project_range(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    mixes_partials: pl.Out[pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32]],
    group_worker: pl.Scalar[pl.INDEX],
    group_workers: pl.Scalar[pl.INDEX],
):
    """AIC：M16、四段 K4096，每段仍按原 16 次 K256 顺序累加。

    调用前等待 HC_INPUT，确保所有本核输入行的 FP32 写回完成；返回后发布
    HC_PARTIAL。逻辑 task=(行块×4+K段)，多波次属于同一组核，不另派任务。"""
    t_dim = pl.tensor.dim(x, 0)
    t_linear = (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE * LINEAR_T_TILE
    x_flat = x
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    # task 是核内的逻辑投影块编号，不是新的运行时 incore task。
    # 连续四个编号对应同一 M16 的四个 K 段，消费者要等齐四份部分和。
    for wave in pl.range((linear_units + group_workers - 1) // group_workers):
        task = group_worker + wave * group_workers
        if task < linear_units:
            t0 = task // LINEAR_OK * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            # stage=2 流水保持原 K256 顺序；只有每份首轮清空 accumulator。
            for kb in pl.pipeline(0, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                x_linear_chunk = pl.slice(
                    x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[t_rows, LINEAR_K_TILE]
                )
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True, init_cond=kb == 0)
            partial_row0 = linear_split * t_linear + t0
            mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc
    return mixes_partials


@pl.jit.inline(auto_scope=False)
def hc_reduce_range(
    mixes_partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    mixes_raw: pl.Out[pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
):
    """AIV：原四份投影按 ((p0+p1)+p2)+p3 顺序归约。

    调用前等待所需四段 HC_PARTIAL；返回后发布 HC_RAW。按完整 M16
    归约含 padding 的输出行，不把四份左折叠改成树形 col_sum。"""
    t_linear = pl.tensor.dim(mixes_raw, 0)
    linear_blocks = t_linear // LINEAR_T_TILE
    # 按 M16 行块轮转，每个 Vector 独立完成本块四份归约，避免新增跨核 RMS。
    for wave in pl.range((linear_blocks + vector_workers - 1) // vector_workers):
        linear_block = vector_worker + wave * vector_workers
        if linear_block < linear_blocks:
            linear_t0 = linear_block * LINEAR_T_TILE
            mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            for linear_split in pl.range(1, LINEAR_OK):
                partial_t0 = linear_split * t_linear + linear_t0
                partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
                mixes_total = pl.add(mixes_total, partial_tile)
            mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total
    return mixes_raw


@pl.jit.inline(auto_scope=False)
def hc_projection_piece(
    partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],
    row: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    COLUMN: pl.constexpr,
    VALID_COLUMNS: pl.constexpr,
    CONSUMER_REDUCE: pl.constexpr,
):
    """加载消费者需要的M8×N8片段；直接路径保持((p0+p1)+p2)+p3。

    VALID_COLUMNS=4用于Sinkhorn，8用于门值，尾行由调用方显式给出。
    partials四段各占PAD16行，列仍为32；所有访问只读已由AIC发布的区域。
    raw仅在旧路径load；直接路径只取其容量描述符，绝不消费填毒的raw内容。
    """
    if CONSUMER_REDUCE:
        rows_per_split = pl.tensor.dim(raw, 0)
        result = pl.load(partials, [row, COLUMN], [T_TILE, HC_PAD],
                         valid_shape=[valid_rows, VALID_COLUMNS], target_memory=pl.MemorySpace.Vec)
        for split in pl.range(1, LINEAR_OK):
            piece = pl.load(partials, [split * rows_per_split + row, COLUMN], [T_TILE, HC_PAD],
                            valid_shape=[valid_rows, VALID_COLUMNS], target_memory=pl.MemorySpace.Vec)
            result = pl.add(result, piece)
    else:
        result = pl.load(raw, [row, COLUMN], [T_TILE, HC_PAD],
                         valid_shape=[valid_rows, VALID_COLUMNS], target_memory=pl.MemorySpace.Vec)
    return result


@pl.jit.inline(auto_scope=False)
def hc_gates_range(
    inv_rms: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Out[pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_tail_store: pl.InOut[pl.Tensor[[T_TILE, HC_PAD], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
    partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    CONSUMER_REDUCE: pl.constexpr,
):
    """AIV：保留原 pre/post sigmoid、scale 乘法和 eps 顺序。

    调用前等待 HC_INPUT 与 HC_RAW（直接归约时改等四个HC_PARTIAL）；返回后发布 HC_GATES，供 mix/norm
    和最终 HCpost 消费。pre 按八列物理布局保留，post 只有四列有效。"""
    t_dim = pl.tensor.dim(post, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    # pre 的物理八列是对齐布局，post 只写四列；最后短行不能把 PAD 行写到输出。
    for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
        ob = vector_worker + wave * vector_workers
        if ob < token_tiles:
            t0 = ob * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            # partial归约直接返回UB Tile，其余参与运算的输入也显式load，
            # 避免把GM Tensor表达式与UB Tile混用于同一个算术操作。
            inv_col = pl.load(inv_rms, [t0, 0], [T_TILE, 1])
            pre_base = pl.reshape(pl.load(hc_base, [0], [HC_PAD]), [1, HC_PAD])
            pre_raw = hc_projection_piece(partials, mixes_raw, t0, T_TILE, 0, HC_PAD, CONSUMER_REDUCE)
            pre_scaled = pl.mul(pl.row_expand_mul(pre_raw, inv_col), scale0)
            pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
            pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
            pre_val = pl.add(pre_sig, HC_EPS)
            pl.store(pre_val, [t0, 0], pre_val_store)
            post_base = pl.reshape(pl.load(hc_base, [HC_MULT], [HC_PAD]), [1, HC_PAD])
            post_raw = hc_projection_piece(partials, mixes_raw, t0, T_TILE, HC_MULT, HC_PAD, CONSUMER_REDUCE)
            post_scaled = pl.mul(pl.row_expand_mul(post_raw, inv_col), scale1)
            post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
            post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
            post_pad = pl.mul(post_sig, 2.0)
            if valid_rows == T_TILE:
                post_valid = pl.set_validshape(post_pad, T_TILE, HC_MULT)
                pl.store(post_valid, [t0, 0], post)
            else:
                pl.store(post_pad, [0, 0], post_tail_store)
                post_tile = pl.load(
                    post_tail_store,
                    [0, 0],
                    [T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(post_tile, [t0, 0], post)
    return pre_val_store, post


@pl.jit.inline(auto_scope=False)
def hc_comb_range(
    inv_rms: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base_2d: pl.Tensor[[1, MIX_HC], pl.FP32],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
    comb_tail_store: pl.InOut[pl.Tensor[[COMB_T_TILE, HC_PAD * HC_MULT], pl.FP32]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
    partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    CONSUMER_REDUCE: pl.constexpr,
):
    """AIV：CSA 原除法路径的二十轮 Sinkhorn，保持行列归约树。

    调用前等待 HC_INPUT 与 HC_RAW（直接归约时改等四个HC_PARTIAL）；与 hc_gates_range/mix 独立，返回后
    发布 HC_COMB。不引入 HCA 的 recip 替代路径，不改变迭代次数。"""
    t_dim = pl.tensor.dim(comb, 0)
    token_tiles = (t_dim + COMB_T_TILE - 1) // COMB_T_TILE
    # 每个逻辑块包含八个 token 各自的 4×4 矩阵，不能把四行误作四个请求。
    for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
        ob = vector_worker + wave * vector_workers
        if ob < token_tiles:
            scale2 = pl.read(hc_scale, [2])
            t0 = ob * COMB_T_TILE
            valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
            inv_col_t = pl.load(
                inv_rms, [t0, 0], [COMB_T_TILE, 1], valid_shape=[valid_rows, 1], target_memory=pl.MemorySpace.Vec
            )
            comb_off = HC_MULT * 2
            # 24 个投影值前八项为 pre/post，其后 16 项按 4×4 的行存储。
            # UB 物理列宽为八，但各行只有前四列有效；以下 load 显式保留此边界。
            mix_g0 = hc_projection_piece(
                partials, mixes_raw, t0, valid_rows, HC_MULT * 2, HC_MULT, CONSUMER_REDUCE
            )
            mix_g1 = hc_projection_piece(
                partials, mixes_raw, t0, valid_rows, HC_MULT * 3, HC_MULT, CONSUMER_REDUCE
            )
            mix_g2 = hc_projection_piece(
                partials, mixes_raw, t0, valid_rows, HC_MULT * 4, HC_MULT, CONSUMER_REDUCE
            )
            mix_g3 = hc_projection_piece(
                partials, mixes_raw, t0, valid_rows, HC_MULT * 5, HC_MULT, CONSUMER_REDUCE
            )
            cb0 = pl.load(
                hc_base_2d,
                [0, comb_off + 0 * HC_MULT],
                [1, HC_PAD],
                valid_shape=[1, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            cb1 = pl.load(
                hc_base_2d,
                [0, comb_off + 1 * HC_MULT],
                [1, HC_PAD],
                valid_shape=[1, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            cb2 = pl.load(
                hc_base_2d,
                [0, comb_off + 2 * HC_MULT],
                [1, HC_PAD],
                valid_shape=[1, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            cb3 = pl.load(
                hc_base_2d,
                [0, comb_off + 3 * HC_MULT],
                [1, HC_PAD],
                valid_shape=[1, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
            row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
            row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
            row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
            # softmax 前填最小值，使额外四列不贡献 exp；随后重新填零再做列和。
            # 两种填充值分别服务 max/exp 与 sum，不能省略或互换。
            row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
            row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
            row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
            row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)
            row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row0_max = pl.row_max(row0_p, row_max_tmp)
            row1_max = pl.row_max(row1_p, row_max_tmp)
            row2_max = pl.row_max(row2_p, row_max_tmp)
            row3_max = pl.row_max(row3_p, row_max_tmp)
            row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
            row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
            row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
            row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
            row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
            row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
            row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
            row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
            row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)
            row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
            row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
            row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
            row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
            row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
            row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
            row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
            row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)
            row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_eff, col_sum)
            row1_cur = pl.div(row1_eff, col_sum)
            row2_cur = pl.div(row2_eff, col_sum)
            row3_cur = pl.div(row3_eff, col_sum)
            # 上面已经完成首轮列归一化，这里只做余下 19 轮。
            # pipeline=2 不消除相邻迭代的数据依赖；保留行/列的原加法树与 div。
            for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
                row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
                row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
                row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
                row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
                row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
                row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
                row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
                row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
                col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
                col_sum = pl.add(col_sum, HC_EPS)
                row0_cur = pl.div(row0_norm, col_sum)
                row1_cur = pl.div(row1_norm, col_sum)
                row2_cur = pl.div(row2_norm, col_sum)
                row3_cur = pl.div(row3_norm, col_sum)
            if valid_rows == COMB_T_TILE:
                row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
                row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
                row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
                row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
                pl.store(row0_out, [t0, 0 * HC_MULT], comb)
                pl.store(row1_out, [t0, 1 * HC_MULT], comb)
                pl.store(row2_out, [t0, 2 * HC_MULT], comb)
                pl.store(row3_out, [t0, 3 * HC_MULT], comb)
            else:
                # 四行先分开写唯一 scratch，再按有效 token 数和四列加载。
                # 直接把物理八列/八行写入 comb 会覆盖邻接矩阵或真实 T 之外。
                pl.store(row0_cur, [0, 0 * HC_PAD], comb_tail_store)
                pl.store(row1_cur, [0, 1 * HC_PAD], comb_tail_store)
                pl.store(row2_cur, [0, 2 * HC_PAD], comb_tail_store)
                pl.store(row3_cur, [0, 3 * HC_PAD], comb_tail_store)
                row0_tail = pl.load(
                    comb_tail_store,
                    [0, 0 * HC_PAD],
                    [COMB_T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                row1_tail = pl.load(
                    comb_tail_store,
                    [0, 1 * HC_PAD],
                    [COMB_T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                row2_tail = pl.load(
                    comb_tail_store,
                    [0, 2 * HC_PAD],
                    [COMB_T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                row3_tail = pl.load(
                    comb_tail_store,
                    [0, 3 * HC_PAD],
                    [COMB_T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
                pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
                pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
                pl.store(row3_tail, [t0, 3 * HC_MULT], comb)
    return comb


@pl.jit.inline(auto_scope=False)
def hc_mix_norm_range(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    pre_val_store: pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_mixed: pl.InOut[pl.Tensor[[MIX_ROWS_DYN, D], pl.BF16]],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_normed_tail_store: pl.InOut[pl.Tensor[[T_TILE, D], pl.BF16]],
    vector_worker: pl.Scalar[pl.INDEX],
    vector_workers: pl.Scalar[pl.INDEX],
):
    """AIV：四路混合、BF16 舍入和完整 D4096 RMSNorm，原序不变。

    调用前等待 HC_GATES 以及本核输入行；返回后发布 HC_NORM。每行始终
    由一个 worker 完成全部 D，保留16个K256平方和的顺序，不新增跨核RMS。
    norm_w 直接接 Native BF16，不在初始化时转换为 FP32。"""
    t_dim = pl.tensor.dim(x, 0)
    x_flat = x
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    # 同一 worker 持有行的全部 4096 列；这样平方和保持单链，无需跨核归约。
    for wave in pl.range((token_tiles + vector_workers - 1) // vector_workers):
        token_block = vector_worker + wave * vector_workers
        if token_block < token_tiles:
            t0 = token_block * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            pre_tile = pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD]
            pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
            pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
            pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
            pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
            pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            # 第一遍：四路 pairwise 混合→BF16 舍入→FP32 平方和，按 16 个 K256
            # 顺序累加。RMS 输入是舍入后的 mixed，不能改用未舍入的 FP32 y。
            for mix_db in pl.pipeline(D // D_TILE, stage=2):
                d0 = mix_db * D_TILE
                x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, d0], valid_shape=[valid_rows, D_TILE])
                x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, D + d0], valid_shape=[valid_rows, D_TILE])
                x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
                x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
                y0 = pl.row_expand_mul(x0, pre0)
                y1 = pl.row_expand_mul(x1, pre1)
                y2 = pl.row_expand_mul(x2, pre2)
                y3 = pl.row_expand_mul(x3, pre3)
                y01 = pl.add(y0, y1)
                y23 = pl.add(y2, y3)
                y = pl.add(y01, y23)
                y_bf16 = pl.cast(y, pl.BF16, mode="rint")
                x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = y_bf16
                y_rounded = pl.cast(y_bf16, pl.FP32)
                y_sq = pl.mul(y_rounded, y_rounded)
                y_sq_sum = pl.row_sum(y_sq)
                y_sq_row = pl.reshape(y_sq_sum, [1, T_TILE])
                sq_sum = pl.add(sq_sum, y_sq_row)
            norm_sq_sum = pl.create_tensor([1, T_TILE], dtype=pl.FP32)
            norm_sq_sum[:, :] = sq_sum
            y_inv = pl.create_tensor([1, T_TILE], dtype=pl.FP32)
            inverse_result = rms_norm_inverse(norm_sq_sum)
            y_inv[:, :] = inverse_result
            # 第二遍：取得全行逆 RMS 后重读 mixed，乘 Native BF16 norm 权重。
            # 两遍间的数据依赖由本核编译流水保证，不能与跨核阶段信号混淆。
            for norm_db in pl.pipeline(D // D_TILE, stage=2):
                d0 = norm_db * D_TILE
                mixed_input = x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE]
                norm_w_input = norm_w[d0 : d0 + D_TILE]
                norm_w_row = pl.reshape(norm_w_input, [1, D_TILE])
                normed_bf16 = rms_norm_apply(mixed_input, norm_w_row, y_inv)
                if valid_rows == T_TILE:
                    x_normed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = normed_bf16
                else:
                    x_normed_tail_store[0:T_TILE, d0 : d0 + D_TILE] = normed_bf16
                    normed_valid = pl.load(
                        x_normed_tail_store,
                        [0, d0],
                        [T_TILE, D_TILE],
                        valid_shape=[valid_rows, D_TILE],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    pl.store(normed_valid, [t0, d0], x_normed)
    return x_mixed, x_normed
