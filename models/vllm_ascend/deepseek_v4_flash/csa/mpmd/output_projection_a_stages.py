# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 的 O_A 分组投影与分组量化：外层显式分核的三个核内阶段。

输入输出与阅读顺序
    先读 o_projection_a_stage_nd/nz：AIC 消费 attention 的 packed heads，
    按八个 O group 独立做 K4096→N1024，写 FP32[P_MM,8192]。再读
    o_projection_group_quant_stage：AIV 对每行每组1024列独立取 amax，
    写 INT8[P_MM,8192] 和 FP32[8,P_SCALE]。这不是跨8192列的统一量化。
    wo_a 保持 Native BF16[8,4096,1024] 的实际 ND 或 NZ 存储；两个函数
    分开声明布局，调用方按真实权重选择，不做格式转换或导入期设备探针。

动态轴、容量与别名
    tokens 是调用方从真实业务 Tensor 动态轴得到的设备 Scalar T，不从
    packed/workspace 容量反推。packed 是 BF16[PACK_TOTAL_ROWS,4096]，
    PACK_TOTAL_ROWS=8*P_PACK，组 g 的第 t 行位于 g*P_PACK+t，P_PACK>=T。
    FP32/INT8 工作区共享 P_MM 容量合同但不共享存储；P_SCALE 独立且>=T。
    O_B 使用的 ROW_TILE 为32/96/128之一，P_MM>=ceil(T/ROW_TILE)*ROW_TILE，
    并须足以保存实际T行。packed总行数须整除8，各视图连续且不得非法别名。
    所有容量轴均 pl.dynamic；固定8组、模型维度和128/256列tile不是B/S白名单。
    外层捕获前拒绝不一致容量/layout；不能让生产者跳写而消费者读取旧值。

分工、数学与尾行
    O_A 以(group,ceil(T/128)行块,N列块)展平，logical%WORKERS 为AIC owner；
    每块先K256 matmul，再原顺序追加15个K256，pipeline只重叠本核搬运。
    仅迁移当前实际启用的 PIPELINE_OA=False，不搬未启用的显式L0流水候选。
    量化以(group,32行块)展平轮转AIV，块内每8行流水；有效行数贯穿load、
    row_max、常数、div、cast和store，绝不读取 O_A 未写的 FP32 padding。
    每组最后一个活动32行块是唯一尾owner，额外将INT8的[T,P_USED)写零，
    P_USED=ceil(T/ROW_TILE)*ROW_TILE；O_B 才能安全整块读取。T=0不产生任务。
    量化保留127/amax、amax*(1/127)及INT32 rint→FP16 round→INT8 trunc；
    反量化scale不能替换为recip(量化乘数)，也不增加BF16往返舍入。

同步与生命周期
    groups helper 允许外层按输出组发布，而不改变原有逻辑任务的 owner。
    group_begin/end 是核内 Scalar 范围；先定位该组内满足 logical%WORKERS
    的首项，后续仍按 WORKERS 条带循环。只改变独立组之间的可见时机，不
    切断组内 K 累加、amax 归约或 padding 清零。全阶段入口调用相同数学体。
    helper不创建spmd、task或notify/wait。O_A先等所消费packed行的全部
    attention生产者，量化先等本组本行的所有O_A列块。量化的真实INT8行、
    scale和所需尾零均完成发布后，O_B才可读整ROW_TILE；外层负责业务fence、
    缓存维护和阶段信号。本核return仅返回GM别名，不能当作其它核完成。
    每个输出行列块唯一写；返回前所有本核wave已经执行，但消费者完成前
    不得复用GM。工作区标InOut保留未写区，P_USED以后的INT8和T以后的scale
    不承诺初始化。BYPASS策略如需使用，须在外层spmd直属设置。
    当前是源级阶段迁移；CPU编译不等于整层设备精度、同步或性能已通过。
"""

import pypto.language as pl

from ..common.config import INT8_AMAX_EPS, INT8_SCALE_MAX

O_GROUPS = 8
O_GROUP_IN = 4096
O_LORA = 1024
PROJ_A_ROW_TILE = 128
A_K_TILE = 256
QUANT_TASK_T_TILE = 32
QUANT_TOKEN_TILE = 8
PACK_TOTAL_ROWS_DYN = pl.dynamic("CSA_OA_PACK_TOTAL_ROWS")
MM_ROWS_DYN = pl.dynamic("CSA_OA_MM_ROWS")
SCALE_ROWS_DYN = pl.dynamic("CSA_OA_SCALE_ROWS")


@pl.jit.inline(auto_scope=False)
def o_projection_a_stage_nd(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
):
    """ND O_A；WORKERS>0、worker合法、A_COL_TILE为128/256，调用前等attention。"""
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    row_blocks = (tokens + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    column_blocks = O_LORA // A_COL_TILE
    # 将原group并行与行/列任务组合为本核条带；每个输出块仍完整做相同K累加。
    for logical in pl.range(worker_id, O_GROUPS * row_blocks * column_blocks, WORKERS):
        g = logical // (row_blocks * column_blocks)
        pa_unit = logical % (row_blocks * column_blocks)
        pa_rb = pa_unit // column_blocks
        nf = pa_unit % column_blocks
        pa_r0 = pa_rb * PROJ_A_ROW_TILE
        pa_rows = pl.min(PROJ_A_ROW_TILE, tokens - pa_r0)
        pa_src0 = g * pack_stride + pa_r0
        out_col_g = g * O_LORA
        n0 = nf * A_COL_TILE
        xa_first = pl.slice(
            o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE]
        )
        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
        # 首块matmul建立实际M行的accumulator布局；不能改成任意全128行零种子。
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(
                o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE]
            )
            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
        # wo_a保留group轴，acc仍带该轴；assemble沿原映射写回有效M，不reshape整个GM。
        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def o_projection_a_groups_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """NZ O_A 的动态组区间；0≤begin≤end≤8，保持全局条带 owner 与原 K 链。"""
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    row_blocks = (tokens + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    column_blocks = O_LORA // A_COL_TILE
    group_first = group_begin * row_blocks * column_blocks
    group_limit = group_end * row_blocks * column_blocks
    # 不是每组重新从worker编号起算；保留全局logical%WORKERS才能保持负载
    # 和量化消费者的owner公式。空组或本核无任务时，首项≥上界，循环自然为空。
    first = worker_id + ((group_first + WORKERS - 1 - worker_id) // WORKERS) * WORKERS
    # NZ证明器不利用owner余数<WORKERS的区间事实；显式提供非负起点。
    # 合法输入下first本就非负，max不改变任何任务编号。
    for logical in pl.range(pl.max(0, first), group_limit, WORKERS):
        g = pl.max(0, logical // (row_blocks * column_blocks))
        pa_unit = logical % (row_blocks * column_blocks)
        pa_rb = pa_unit // column_blocks
        # row_blocks*column_blocks整除列块数，因此与pa_unit%column_blocks等价。
        # 除数直接写constexpr表达式：NZ证明器不沿Scalar别名追踪模数常量。
        nf = logical % (O_LORA // A_COL_TILE)
        pa_r0 = pa_rb * PROJ_A_ROW_TILE
        pa_rows = pl.min(PROJ_A_ROW_TILE, tokens - pa_r0)
        pa_src0 = g * pack_stride + pa_r0
        out_col_g = g * O_LORA
        # NZ访问需要编译器可证明的非负列偏移，合法worker的数学值不因此改变。
        # 只显式表达非负合同，不依赖max阻止优化；关键是上述constexpr模数。
        n0 = pl.max(0, nf) * A_COL_TILE
        xa_first = pl.slice(
            o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE]
        )
        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(
                o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE]
            )
            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def o_projection_a_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
):
    """全八组兼容入口：与按组通知分支复用同一数学，不另维护计算版本。"""
    o_r_pad = o_projection_a_groups_stage_nz(
        o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, A_COL_TILE, 0, O_GROUPS
    )
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def o_projection_groups_quant_stage(
    o_r_pad: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32],
    o_r_i8_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.INT8]],
    act_scale_dq: pl.InOut[pl.Tensor[[O_GROUPS, SCALE_ROWS_DYN], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """量化动态组区间；尾零属于该组的最后活动块，必须先于本组通知完成。"""
    task_rows = (tokens + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE
    padded_rows = (tokens + ROW_TILE - 1) // ROW_TILE * ROW_TILE
    # T=0时循环为空，不计算负尾owner；每个group的活动行及padding各只有一个写者。
    group_first = group_begin * task_rows
    group_limit = group_end * task_rows
    first = worker_id + ((group_first + WORKERS - 1 - worker_id) // WORKERS) * WORKERS
    for logical in pl.range(first, group_limit, WORKERS):
        g = logical // task_rows
        block = logical % task_rows
        col_g = g * O_LORA
        quant_start = block * QUANT_TASK_T_TILE
        # 保留原32行任务内部的8行双级流水，不跨组取amax；尾行不读FP32 pad。
        for qt in pl.pipeline(
            quant_start, pl.min(quant_start + QUANT_TASK_T_TILE, tokens), QUANT_TOKEN_TILE, stage=2
        ):
            rows = pl.min(QUANT_TOKEN_TILE, tokens - qt)
            oc_amax = pl.load(o_r_pad, [qt, col_g], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA])
            reduce_tmp = pl.tile.create([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP32)
            g_row_max = pl.reshape(
                pl.set_validshape(pl.row_max(pl.abs(oc_amax), reduce_tmp), rows, 1), [1, QUANT_TOKEN_TILE]
            )
            # row_max转行后，所有常数与算术结果均保留[1,rows]，不能恢复成8个有效值。
            g_amax_floor = pl.set_validshape(
                pl.tile.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS), 1, rows
            )
            g_amax = pl.set_validshape(pl.maximum(g_amax_floor, g_row_max), 1, rows)
            g_scale_num = pl.set_validshape(
                pl.tile.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), 1, rows
            )
            g_sq_row = pl.set_validshape(pl.div(g_scale_num, g_amax), 1, rows)
            # 与canonical一致：dequant尺度为amax乘1/127，不改成recip(127/amax)。
            dequant_scale = pl.set_validshape(pl.mul(g_amax, 1.0 / INT8_SCALE_MAX), 1, rows)
            pl.store(dequant_scale, [g, qt], act_scale_dq)
            g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
            oc_q = pl.load(o_r_pad, [qt, col_g], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA])
            oq_scaled = pl.set_validshape(pl.row_expand_mul(oc_q, g_sq_col), rows, O_LORA)
            oq_i32 = pl.set_validshape(pl.cast(oq_scaled, target_type=pl.INT32, mode="rint"), rows, O_LORA)
            oq_half = pl.set_validshape(pl.cast(oq_i32, target_type=pl.FP16, mode="round"), rows, O_LORA)
            oq_i8 = pl.set_validshape(pl.cast(oq_half, target_type=pl.INT8, mode="trunc"), rows, O_LORA)
            pl.store(oq_i8, [qt, col_g], o_r_i8_pad)
        # 只最后活动32行块有权填本组尾零。零填不读o_r_pad，不写scale尾部，
        # 并在同一阶段发布之前完成；O_B整RB读取不可仅依赖实际行的quant完成。
        if quant_start + QUANT_TASK_T_TILE >= tokens:
            for zt in pl.range(tokens, padded_rows, QUANT_TOKEN_TILE):
                zero_half = pl.tile.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc")
                zero_rows = pl.min(QUANT_TOKEN_TILE, padded_rows - zt)
                pl.store(pl.set_validshape(zero_i8, zero_rows, O_LORA), [zt, col_g], o_r_i8_pad)
    return o_r_i8_pad, act_scale_dq


@pl.jit.inline(auto_scope=False)
def o_projection_group_quant_stage(
    o_r_pad: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32],
    o_r_i8_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.INT8]],
    act_scale_dq: pl.InOut[pl.Tensor[[O_GROUPS, SCALE_ROWS_DYN], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
):
    """全八组入口，与按组通知分支共享 amax、量化舍入和尾行清零数学。"""
    o_r_i8_pad, act_scale_dq = o_projection_groups_quant_stage(
        o_r_pad, o_r_i8_pad, act_scale_dq, tokens, worker_id, WORKERS, ROW_TILE, 0, O_GROUPS
    )
    return o_r_i8_pad, act_scale_dq
