# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 的 Q_A 与 QR 源级阶段：由外层单 SPMD 指定 worker 和同步。

输入、输出与动态量
    Q_A 接收归一化后的 BF16 x[T,4096] 和 Native BF16 权重 [1024,4096]，
    输出 FP32 qa[QA_ROWS,1024]。QR 接收 qa 与 BF16 gamma[1024]，输出真实
    token 行的 INT8 qr[T,1024]、FP32 scale[T,1]，同时写 Q_B 所需的暂存区。
    T 来自 Tensor 的运行时维度；QA_ROWS/MATMUL_ROWS/SCALE_ROWS 是实际工作区
    维度。这里不固定 B/S，也不读取长度 Tensor 来猜本轮 token 数。

区间策略与核内分工
    沿用原 qkv_proj_rope.py：T<128 用 M32，较大 T 用 M64，尾块 M16；
    N128、K256 固定，完整 K4096 按原顺序累加。至多三组 M 分工形成 8～24
    份独立 M/N 工作，由显式 AIC worker 按 WORKERS 步长领取。QR 仍按八行
    一个 tile、每 256 列一次规约，由显式 AIV worker 按实际有效 tile 数分工。
    这些区间决定数学 tile，循环边界仍来自实际 T，并非为某个 batch 展开图。
    QA_M16_PARALLEL为Host选择的constexpr，默认False保留原完整策略；
    Host仅在G6且16<T<32（原两个M16尾块、总权重请求不增）开启全局M16分核。
    开启时每块八个N128列任务按logical%WORKERS领取；T及尾部仍动态。
    关闭时不编译该分支，避免只增运行时判断也扰动旧QA流水。所有核的
    QA发布保留，QR在开启时等待min(WORKERS,8*ceil(T/16))个真实owner。
    权重 BYPASS 缓存策略须由外层直接在 spmd 体内设置；PyPTO 不允许把该
    scope 属性藏在 inline helper 中。

阶段数据流、同步与工作区
    外层必须先保证 x 可读；Q_A 的所有相关 N128 列生产者完成 GM 写回后，
    QR 才能读该行的完整 1024 列。QR 发布 INT8 和 scale 后，下游 Indexer/Q_B
    才能消费。helper 内没有 spmd、submit、task ID 或 notify/wait，不能自行
    证明跨核完成：外层负责 producer 数据 fence、信号发布、消费者等待和必要
    的缓存处理。阶段开始前不能仅等待 flag 而省略业务数据的可见性协议。
    qa 至少分配 ceil(T/16)*16 行，两个 Q_B 暂存区至少 ceil(T/8)*8 行；
    这里的八行对齐只是 QR 本阶段写入的最小容量。接入 Q_B 后，INT8 暂存
    必须取后者要求的 ceil(T/128)*128 行，scale 暂存仍按八行；不能只按 QR
    的最小值分配整链工作区。qr/scale 仅存有效尾行，容量尾部不承诺清零。

阅读顺序
    q_a_stage 是动态 tile 策略入口，_q_a_worker 解释各 M/N 块如何归核；
    随后看 q_r_stage 的平方和/amax、量化比例、尾行写回三段。外层统一
    SPMD 再根据这些所有权计算前驱，不能把 helper 的返回当作跨核同步。

数值合同与当前边界
    原高层源码是唯一数学来源，不使用冻结 IR。当前保留固定 K 非 atomic
    路径，不能替代 split-K atomic 模式。QR 保留 256 列平方和/amax 顺序、
    rsqrt、127/amax 的 pl.div、pl.recip 与 rint→FP16→INT8 转换链；gamma
    仍直接接 BF16，在 AIV 内转 FP32。没有在初始化时替换 Native 权重格式。
    这里只实现完整 CSA 的 Q_A→QR 前段，未包含 Q_B、RoPE、KV 或整层融合。
    CPU 编译能证明可解析/可 lowering，不能代替设备精度或同步验证。
"""

import pypto.language as pl

from ..common.options import weight_nz_mode

from ..common.config import FLASH as M
from ..common.config import INT8_AMAX_EPS, INT8_SCALE_MAX
from ..common.reduction import ATOMIC_ADD

# 只复用 wq_a 的真实布局规则，不导入原 nz_mode 对另一张三维 wo_a 权重的设备探针。
# 三档的解释与 Native/原数学一致；这里只声明布局，没有重排或转换权重。
WEIGHT_NZ_MODE = weight_nz_mode()
if WEIGHT_NZ_MODE not in (0, 1, 2):
    raise ValueError("VLLM_ASCEND_ENABLE_NZ 只接受 0/1/2")
BF16_WEIGHT_LAYOUT = pl.NZ if WEIGHT_NZ_MODE >= 2 else None

# 从 qkv_proj_rope.py 提取的模型维度和 tile 常量；源码审计逐项比较原表达式值。
# 不导入原整个多 task 模块，避免仅为了常量而触发无关权重的设备格式探针。
D = M.hidden_size
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
Q_LORA_TILE = 256
QR_DENSE_M_TILE = 64
QR_FIXED_SMALL_M_TILE = 32
QR_FIXED_SMALL_ROWS = 128
QR_FIXED_M_GROUPS = 3
QR_M_TILE = 16
QR_N_TILE = 128
QR_N_BLOCKS = Q_LORA // QR_N_TILE
QR_K_TILE = 256
QUANT_TILE = 256
T_TILE = 8

# 本阶段没有 split-K seed/atomic 完成协议；显式拒绝，不能偷偷改归约语义。
if ATOMIC_ADD:
    raise ValueError("动态 Q_A 阶段目前只支持固定 K；不支持 split-K atomic 模式")

T_DYN = pl.dynamic("CSA_QKV_T_DYN")
QA_ROWS_DYN = pl.dynamic("CSA_QA_ROWS_DYN")
# 两份QR工作区的容量可以不同：INT8在接Q_B时需pad128，scale只需pad8。
# 入口必须保留两个动态轴，不能因为本阶段都按八行写就约束它们等高。
MATMUL_ROWS_DYN = pl.dynamic("CSA_QR_MATMUL_ROWS_DYN")
SCALE_ROWS_DYN = pl.dynamic("CSA_QR_SCALE_ROWS_DYN")


@pl.jit.inline(auto_scope=False)
def _q_a_worker(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qa_fp32: pl.Tensor[[QA_ROWS_DYN, Q_LORA], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    DENSE_M: pl.constexpr,
):
    """一个物理 AIC 领取原投影的 M/N 逻辑分块，保持完整 K 累加链。

    WORKERS 必须为正，worker_id 属于 [0,WORKERS)。同一行需八个列块全部
    完成才能归一化；返回 Tensor 只表达数据别名，不表达跨核同步完成。
    """
    tokens = pl.tensor.dim(x, 0)
    padded_rows = ((tokens + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
    full_rows = (tokens // DENSE_M) * DENSE_M
    m_groups = pl.min(QR_FIXED_M_GROUPS, pl.max(1, tokens // DENSE_M))
    # max 仅显式提供 NZ 切片所需的非负证明；合法 worker 的值保持不变。
    # 这里是已有 incore 内的普通循环，不会派发新的运行时任务。
    for logical_worker in pl.range(pl.max(worker_id, 0), QR_N_BLOCKS * m_groups, WORKERS):
        column = (logical_worker % QR_N_BLOCKS) * QR_N_TILE
        m_group = logical_worker // QR_N_BLOCKS
        for row in pl.range(m_group * DENSE_M, full_rows, m_groups * DENSE_M):
            dense_acc = pl.create_tensor([DENSE_M, QR_N_TILE], dtype=pl.FP32)
            # 仅搬运流水化；K256 的顺序以及首次 matmul_acc 的清零语义不变。
            for k_block in pl.pipeline(0, D // QR_K_TILE, stage=2):
                k0 = k_block * QR_K_TILE
                dense_x = x[row : row + DENSE_M, k0 : k0 + QR_K_TILE]
                dense_weight = wq_a[column : column + QR_N_TILE, k0 : k0 + QR_K_TILE]
                dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_weight, b_trans=True, init_cond=(k_block == 0))
            qa_fp32 = pl.assemble(qa_fp32, dense_acc, [row, column])
        # 尾部保持原 M16，输入只读实际有效行；不能改成另一条 K 累加链。
        for row in pl.range(full_rows + m_group * QR_M_TILE, padded_rows, m_groups * QR_M_TILE):
            tail_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for k_block in pl.pipeline(D // QR_K_TILE, stage=2):
                k0 = k_block * QR_K_TILE
                valid_rows = pl.min(QR_M_TILE, tokens - row)
                tail_x = pl.slice(x, [QR_M_TILE, QR_K_TILE], [row, k0],
                                   valid_shape=[valid_rows, QR_K_TILE])
                tail_weight = wq_a[column : column + QR_N_TILE, k0 : k0 + QR_K_TILE]
                tail_acc = pl.matmul_acc(tail_acc, tail_x, tail_weight, b_trans=True, init_cond=(k_block == 0))
            qa_fp32 = pl.assemble(qa_fp32, tail_acc, [row, column])
    return qa_fp32


@pl.jit.inline(auto_scope=False)
def _q_a_m16_parallel_worker(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qa_fp32: pl.Tensor[[QA_ROWS_DYN, Q_LORA], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """全局M16块按八个N128列任务分核；仅改变owner，不改变原尾块数学。

    Host策略只为G6且16<T<32开启，所以正式调用每核至多一块且总权重请求不增。
    helper仍按实际T计算循环上界，较大T也可按WORKERS跨波次领取；不得将
    T24或两块写死。每块仍K256×16、首段初始化、valid_shape及pad16写回。
    新增QA参与核仍承担后继Indexer，不能按核数预记整层性能收益。
    """
    tokens = pl.tensor.dim(x, 0)
    m16_blocks = (tokens + QR_M_TILE - 1) // QR_M_TILE
    for logical_worker in pl.range(pl.max(worker_id, 0), QR_N_BLOCKS * m16_blocks, WORKERS):
        column = (logical_worker % QR_N_BLOCKS) * QR_N_TILE
        row = (logical_worker // QR_N_BLOCKS) * QR_M_TILE
        m16_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
        for k_block in pl.pipeline(D // QR_K_TILE, stage=2):
            k0 = k_block * QR_K_TILE
            valid_rows = pl.min(QR_M_TILE, tokens - row)
            m16_x = pl.slice(x, [QR_M_TILE, QR_K_TILE], [row, k0],
                             valid_shape=[valid_rows, QR_K_TILE])
            m16_weight = wq_a[column : column + QR_N_TILE, k0 : k0 + QR_K_TILE]
            m16_acc = pl.matmul_acc(m16_acc, m16_x, m16_weight, b_trans=True, init_cond=(k_block == 0))
        qa_fp32 = pl.assemble(qa_fp32, m16_acc, [row, column])
    return qa_fp32


@pl.jit.inline(auto_scope=False)
def q_a_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qa_fp32: pl.Tensor[[QA_ROWS_DYN, Q_LORA], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    QA_M16_PARALLEL: pl.constexpr = False,
):
    """由外层 AIC 调用；按实际 T 选 M32/M64，不按固定 batch 生成入口。"""
    tokens = pl.tensor.dim(x, 0)
    if QA_M16_PARALLEL:
        qa_fp32 = _q_a_m16_parallel_worker(x, wq_a, qa_fp32, worker_id, WORKERS)
    elif tokens < QR_FIXED_SMALL_ROWS:
        qa_fp32 = _q_a_worker(x, wq_a, qa_fp32, worker_id, WORKERS, QR_FIXED_SMALL_M_TILE)
    else:
        qa_fp32 = _q_a_worker(x, wq_a, qa_fp32, worker_id, WORKERS, QR_DENSE_M_TILE)
    return qa_fp32


@pl.jit.inline(auto_scope=False)
def q_r_stage(
    qa_fp32: pl.Tensor[[QA_ROWS_DYN, Q_LORA], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Tensor[[MATMUL_ROWS_DYN, Q_LORA], pl.INT8],
    qr_scale_pad: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """由外层 AIV 调用，复用原八行归约与量化算术。

    默认外层可沿用 16 个逻辑 AIV worker。传入的编号必须按 AIV 角色编号，
    不能直接混用 AIC 核组号；WORKERS 必须为正。未获有效 tile 的 worker
    跳过 gamma 搬运，空 T 也不会进入步长为零的循环。
    """
    tokens = pl.tensor.dim(qr, 0)
    token_tiles = (tokens + T_TILE - 1) // T_TILE
    workers = pl.min(token_tiles, WORKERS)
    if worker_id < workers:
        # 原 task 外的 GM reshape 不能机械搬到 incore 内：那里会变成 Tile。
        # 直接读相同的 1024 个 BF16 元素，再在 UB 内改形状，数据与转换不变。
        gamma_cq_row = pl.reshape(
            pl.load(gamma_cq, [0], [Q_LORA], target_memory=pl.MemorySpace.Vec),
            [1, Q_LORA],
        )
        qr_gamma_ub = pl.cast(
            gamma_cq_row,
            target_type=pl.FP32,
        )
        # active worker 数随实际八行块数缩小；保证每块恰由一个 worker 写回，
        # 小 T 不会让空 worker 以零步长运行。多波次复用本核 gamma UB。
        for tile_index in pl.range(worker_id, token_tiles, workers):
            tg = tile_index * T_TILE
            valid_rows = pl.min(T_TILE, tokens - tg)
            qr_input_ub = pl.load(
                qa_fp32, [tg, 0], [T_TILE, Q_LORA], target_memory=pl.MemorySpace.Vec,
            )
            qr_sq_sum = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            qr_amax_g = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            # 依序处理 0/256/512/768 四段，不改为全行规约或其它结合次序。
            for qr_rms_col0 in pl.pipeline(0, Q_LORA, Q_LORA_TILE, stage=2):
                qr_rms_chunk = pl.tile.extract(
                    qr_input_ub, 0, qr_rms_col0, [T_TILE, Q_LORA_TILE], target_memory=pl.MemorySpace.Vec,
                )
                qr_rms_sq = pl.mul(qr_rms_chunk, qr_rms_chunk)
                qr_rms_row_sum = pl.reshape(
                    pl.row_sum(qr_rms_sq, tmp_tile=pl.tile.create([T_TILE, Q_LORA_TILE], dtype=pl.FP32)),
                    [1, T_TILE],
                )
                qr_sq_sum = pl.add(qr_sq_sum, qr_rms_row_sum)
                gamma_rms_chunk = pl.tile.extract(
                    qr_gamma_ub, 0, qr_rms_col0, [1, Q_LORA_TILE], target_memory=pl.MemorySpace.Vec,
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
            # 必须保留“先除、再 recip”；改成 amax/127 会改变当前舍入行为。
            qr_scale_quant_row = pl.div(pl.tile.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
            qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
            qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
            pl.store(qr_tile_scale_dq, [tg, 0], qr_scale_pad)
            if valid_rows == T_TILE:
                pl.store(qr_tile_scale_dq, [tg, 0], qr_scale)
            else:
                # 直接存 UB 的有效尾行，不能先存 GM 再读回来引入旧值风险。
                qr_scale_tail = pl.set_validshape(qr_tile_scale_dq, valid_rows, 1)
                pl.store(qr_scale_tail, [tg, 0], qr_scale)
            # 已取得整行尺度后，第二遍按 K256 量化；stage=2 只改本核搬运重叠，
            # 保留 INT32 rint→FP16 round→INT8 trunc 的原转换链。
            for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
                qr_chunk = pl.tile.extract(
                    qr_input_ub, 0, qa, [T_TILE, QUANT_TILE], target_memory=pl.MemorySpace.Vec,
                )
                gamma_q_chunk = pl.tile.extract(
                    qr_gamma_ub, 0, qa, [1, QUANT_TILE], target_memory=pl.MemorySpace.Vec,
                )
                qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
                qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
                qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
                qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
                qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
                pl.store(qr_q_i8, [tg, qa], qr_i8_matmul)
                if valid_rows == T_TILE:
                    pl.store(qr_q_i8, [tg, qa], qr)
                else:
                    qr_q_tail = pl.set_validshape(qr_q_i8, valid_rows, QUANT_TILE)
                    pl.store(qr_q_tail, [tg, qa], qr)
    return qr, qr_scale, qr_i8_matmul, qr_scale_pad
