# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""动态 CSA 输出后段：O_B INT8 投影、八组反量化与 HC post。

整体数据流与阅读顺序
    NZ 大 M128 路径只计算真实 T 内的行：每个 activation 切片显式带
    min(128,T-t0) 有效行，matmul 和 partial 写回沿用这一动态范围。
    T 内的 dummy/inactive 行仍完整计算，不按请求 metadata 跳算。
    权重驻留、K1024 整数点积、N256 和八组 FP32 反量化顺序保持不变。
    工作区容量和上游 INT8 补零合同保留；HC post 本来就只读取 T 行。
    上游 O_A/按组量化产生 INT8 activation[P_MM,8192] 和 FP32 scale[8,P_SCALE]。
    先阅读 o_b_stage 所选择的 ND/NZ AIC helper，再阅读末尾的 AIV
    o_dequant_hc_post_stage；其中 hc_post_row_block 是最后一步的单行数学。
    O_B 写 INT32 partials[P_MM,32768]，每个组独占4096列；AIV 依组号顺序
    反量化求和、乘权重 scale、舍入到 BF16，再完成四路 HC 残差组合。

输入输出、动态变量和布局
    实际 T 从输出 flat Tensor 的动态行数取得；O_B 的 tokens 参数由外层
    从同一真实输入的 dim 取得。T、P_MM、P_SCALE 是三个独立 dynamic 轴，
    不是 B*S 的编译期常量。P_MM 同时约束 activation 和 partials 的行容量；
    P_SCALE 只约束 scale 容量，不能误绑到 P_MM。Native 二维 INT8 权重保持
    [8192,4096] 根布局，NZ 的 group 只改变 K 偏移，不构造私有转置权重。
    residual_flat/out_flat 是外层绑定的 BF16[T,16384] 无损二维视图，分别
    与 Native [T,4,4096] 共享 storage；helper 内不做需要 Host 的 GM reshape。

区间 tile 与核分工
    ROW_TILE 由外层策略选32、96或128，实际循环长度仍是 ceil(T/ROW_TILE)。
    ND 展开(group,row_block,D512块)，每个任务计算两个 N256 列片段。
    NZ 展开(group,D512块)，保留原 row/N/K 循环层次：32/96行用整 activation
    驻留 L1、权重 K256 流水；128行用每个 N256 权重驻留后循环行块。
    显式 worker 轮转领取逻辑任务，helper 不启动 spmd，不隐式读取核号。
    groups 入口接收核内 group_begin/end，保留原全局任务编号及取模 owner；
    外层只需等待这些输出组的量化和尾零，不必等其它组。全八组入口复用
    同一数学函数，不增加第二份矩阵实现，也不改组内 row/N/K 驻留顺序。
    最终 AIV 逐真实 token 轮转，owner=t%WORKERS；单行门值不能广播到其它
    token。模型的8个输出组、4096维和合法数学 tile 可以保持 constexpr。

同步、工作区和生命周期
    本文件没有 notify/wait。外层先等待 O_A 量化的真实 owner，包括所有
    [T,ceil(T/ROW_TILE)*ROW_TILE) INT8 填充行清零，保持原有阶段依赖。
    ND 和 NZ 小中 tile 仍读完整 M；NZ 大 M128 用有效行数截断最后一块。
    O_B 不读原始 FP32 O_A padding；P_MM 保留整块容量，P_SCALE>=T。
    每个有效 partial 区域唯一写入且首 K 块初始化 Acc，无需整份清零。
    大 M128 路径不再初始化 [T,P_MM) partial 尾区；唯一消费者按真实 T
    逐行加载 [1,4096]，不得把这段未写区域当作有效输出或跨调用初值。
    终端 AIV 必须等待所读 token 的八组 O_B partial、八个 activation scale、
    HC pre post/comb 完成并可见。ND/NZ 的 owner 不同，不能套单纯行 tile 的
    wait_row_owners。返回 Tensor 引用只表达数据流，不等于其它核已经完成。
    只写实际 T 的输出；大于当前使用范围的工作区保持原值。输入/输出默认
    不重叠；alias、容量、合法 ROW_TILE 和正 WORKERS 由捕获前统一验证。

数值边界与当前范围
    保留同仓 decode_o_proj.py 的 INT32 累加与八组 FP32 左折叠顺序，以及
    hc_post.py 的逐项残差加法。先 BF16 舍入 O 投影输出再进入 HC post，不能
    合并这次舍入；也不能先求 comb*residual 的和再加 post*x 改变求和树。
    本模块只补齐完整 CSA 的输出后段；没有 attention、O_A/量化和阶段同步，
    不是可独立调用的完整 CSA 服务入口，CPU 编译不能代表设备精度或性能。
"""

import pypto.language as pl

from ..common.options import weight_nz_mode

from ..common.config import FLASH as M

WEIGHT_NZ_MODE = weight_nz_mode()
if WEIGHT_NZ_MODE not in (0, 1, 2):
    raise ValueError("VLLM_ASCEND_ENABLE_NZ 只接受 0/1/2")
QUANT_WEIGHT_NZ = WEIGHT_NZ_MODE >= 1
QUANT_WEIGHT_LAYOUT = pl.NZ if QUANT_WEIGHT_NZ else None

D = M.hidden_size
O_GROUPS = M.o_groups
O_LORA = M.o_lora_rank
HC_MULT = M.hc_mult
assert HC_MULT == 4
B_K_TILE = 256
PROJ_B_D_TILE = 512
PROJ_B_MM_N_TILE = 256
PROJ_B_MEDIUM_T_TILE = 96
T_DYN = pl.dynamic("CSA_OUTPUT_B_T")
MM_ROWS_DYN = pl.dynamic("CSA_OUTPUT_B_MM_ROWS")
SCALE_ROWS_DYN = pl.dynamic("CSA_OUTPUT_B_SCALE_ROWS")


@pl.jit.inline(auto_scope=False)
def _o_b_nd_groups_stage(
    o_r_i8_pad: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.INT8],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    partials: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * D], pl.INT32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """ND：owner=(g*row_blocks*8+tb*8+dc)%WORKERS，实际T决定任务数。"""
    proj_b_t_rows = (tokens + ROW_TILE - 1) // ROW_TILE
    # group、行块、D512块展开；每个输出矩形只归一个 worker。
    group_first = group_begin * proj_b_t_rows * (D // PROJ_B_D_TILE)
    group_limit = group_end * proj_b_t_rows * (D // PROJ_B_D_TILE)
    first = worker_id + ((group_first + WORKERS - 1 - worker_id) // WORKERS) * WORKERS
    for work in pl.range(first, group_limit, WORKERS):
        g = work // (proj_b_t_rows * (D // PROJ_B_D_TILE))
        pb_unit = work % (proj_b_t_rows * (D // PROJ_B_D_TILE))
        col_g = g * O_LORA
        tb = pb_unit // (D // PROJ_B_D_TILE)
        dc = pb_unit - tb * (D // PROJ_B_D_TILE)
        t0 = tb * ROW_TILE
        d0 = dc * PROJ_B_D_TILE
        # 每D512块含两个N256；保留原K256顺序及首段初始化条件。
        for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
            n0 = d0 + nf * PROJ_B_MM_N_TILE
            # ND操作数仍是高层Tensor切片，种子也保持Tensor层级；不能混入Tile
            # Acc再调用高层matmul_acc。后端将该原始种子lower为矩阵累加器。
            acc_b = pl.create_tensor([ROW_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                k0 = col_g + kb * B_K_TILE
                b_act = o_r_i8_pad[t0 : t0 + ROW_TILE, k0 : k0 + B_K_TILE]
                b_weight = wo_b[k0 : k0 + B_K_TILE, n0 : n0 + PROJ_B_MM_N_TILE]
                acc_b = pl.matmul_acc(acc_b, b_act, b_weight, init_cond=(kb == 0))
            partials[t0 : t0 + ROW_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
    return partials


@pl.jit.inline(auto_scope=False)
def _o_b_nz_groups_stage(
    o_r_i8_pad: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.INT8],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    partials: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * D], pl.INT32],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """NZ：owner=(g*8+dc)%WORKERS；每任务保持全部实际行块的原驻留次序。"""
    proj_b_t_rows = (tokens + ROW_TILE - 1) // ROW_TILE
    # 非负 worker+round*WORKERS 使 NZ group/K/N 偏移的非负性显式可证。
    worker = pl.max(worker_id, 0)
    group_first = pl.max(group_begin, 0) * (D // PROJ_B_D_TILE)
    group_limit = group_end * (D // PROJ_B_D_TILE)
    # 从包含组起点的全局wave开始，保留work=非负worker+非负wave*核数。
    # 若本核在首wave落于前一组，只跳过这一个work；不产生负余数，也让
    # NZ证明器直接识别非负列偏移。实际每组首项仍与全局条带分配完全一致。
    for round_id in pl.range(group_first // WORKERS, (group_limit - worker + WORKERS - 1) // WORKERS):
        work = worker + round_id * WORKERS
        if work >= group_first:
            g = work // (D // PROJ_B_D_TILE)
            col_g = g * O_LORA
            d0 = (work % (D // PROJ_B_D_TILE)) * PROJ_B_D_TILE
            if ROW_TILE <= PROJ_B_MEDIUM_T_TILE:
                # 小/中tile：每行块activation只入L1一次，复用到两个N256输出。
                for tb in pl.range(proj_b_t_rows):
                    t0 = tb * ROW_TILE
                    activation_l1 = pl.load(
                        o_r_i8_pad, [t0, col_g], [ROW_TILE, O_LORA], target_memory=pl.MemorySpace.Mat,
                    )
                    for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        acc_b = pl.create_tile([ROW_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32,
                                               target_memory=pl.MemorySpace.Acc)
                        for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                            k0 = kb * B_K_TILE
                            wk0 = pl.max(g, 0) * O_LORA + k0
                            b_weight_l1 = pl.load(
                                wo_b, [wk0, n0], [B_K_TILE, PROJ_B_MM_N_TILE], target_memory=pl.MemorySpace.Mat,
                            )
                            b_act_l1 = pl.tile.slice(activation_l1, [ROW_TILE, B_K_TILE], [0, k0])
                            # 两操作数仍是Mat，保留后端原K128 L0交错lowering，不强改L0操作。
                            acc_b = pl.tile.matmul_acc(acc_b, b_act_l1, b_weight_l1, init_cond=(kb == 0))
                        pl.store(acc_b, [t0, g * D + n0], partials)
            else:
                # 大tile：N块在外，权重驻留后再逐行块计算；不能交换成上面的循环树。
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    weight_k0 = pl.max(g, 0) * O_LORA
                    weight_resident = wo_b[weight_k0 : weight_k0 + O_LORA, n0 : n0 + PROJ_B_MM_N_TILE]
                    for tb in pl.range(proj_b_t_rows):
                        t0 = tb * ROW_TILE
                        # 当前tb来自ceil(T/ROW_TILE)，所以有效行数在1..ROW_TILE。
                        # 这是删去T以外的算术padding，T内的dummy或inactive行仍完整计算。
                        # ND activation允许动态有效M；NZ权重的K/N范围没有变化。
                        active_rows = pl.min(ROW_TILE, tokens - t0)
                        b_act_full = pl.slice(
                            o_r_i8_pad, [ROW_TILE, O_LORA], [t0, col_g],
                            valid_shape=[active_rows, O_LORA],
                        )
                        full_acc = pl.matmul(b_act_full, weight_resident, out_dtype=pl.INT32)
                        # assemble沿用源Acc的有效行，不能仍按完整ROW_TILE写回。
                        # [T,P_MM)不再由此路径初始化；所有消费者必须只读取真实T。
                        partials = pl.assemble(partials, full_acc, [t0, g * D + n0])
    return partials


# ND/NZ分开解析，避免把互不兼容的layout数学分支同时放进同一JIT控制流。
o_b_groups_stage = _o_b_nz_groups_stage if QUANT_WEIGHT_NZ else _o_b_nd_groups_stage


@pl.jit.inline(auto_scope=False)
def o_b_stage(
    o_r_i8_pad: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.INT8],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    partials: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * D], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
):
    """全八组兼容入口；与组就绪分支共用数学和动态行循环。"""
    partials = o_b_groups_stage(
        o_r_i8_pad, wo_b, partials, tokens, worker_id, WORKERS, ROW_TILE, 0, O_GROUPS
    )
    return partials


@pl.jit.inline(auto_scope=False)
def hc_post_row_block(
    x_block: pl.Tensor,
    residual_flat: pl.Tensor,
    post: pl.Tensor,
    comb: pl.Tensor,
    y_flat: pl.Out[pl.Tensor],
    t0: pl.Scalar[pl.INDEX],
    n0: pl.Scalar[pl.INDEX],
    COLS: pl.constexpr,
):
    """严格单行HC post；x_block已经BF16舍入，不把一个token门值广播到多行。"""
    x_f32 = pl.cast(x_block, pl.FP32)
    # 先读取本token的四份残差再输出；外层仍须拒绝未经证明的输入输出alias。
    res_0 = pl.cast(pl.slice(residual_flat, [1, COLS], [t0, n0], valid_shape=[1, COLS], clamp=True), pl.FP32)
    res_1 = pl.cast(pl.slice(residual_flat, [1, COLS], [t0, D + n0], valid_shape=[1, COLS], clamp=True), pl.FP32)
    res_2 = pl.cast(pl.slice(residual_flat, [1, COLS], [t0, 2 * D + n0], valid_shape=[1, COLS], clamp=True), pl.FP32)
    res_3 = pl.cast(pl.slice(residual_flat, [1, COLS], [t0, 3 * D + n0], valid_shape=[1, COLS], clamp=True), pl.FP32)
    # 四个输出组各自先post*x，再依in_h=0,1,2,3左折叠；不预求残差和。
    for out_h in pl.unroll(HC_MULT):
        post_scalar = pl.read(post, [t0, out_h])
        single_y = pl.mul(x_f32, post_scalar)
        comb_scalar_0 = pl.read(comb, [t0, out_h])
        single_weighted_0 = pl.mul(res_0, comb_scalar_0)
        single_y = pl.add(single_y, single_weighted_0)
        comb_scalar_1 = pl.read(comb, [t0, HC_MULT + out_h])
        single_weighted_1 = pl.mul(res_1, comb_scalar_1)
        single_y = pl.add(single_y, single_weighted_1)
        comb_scalar_2 = pl.read(comb, [t0, 2 * HC_MULT + out_h])
        single_weighted_2 = pl.mul(res_2, comb_scalar_2)
        single_y = pl.add(single_y, single_weighted_2)
        comb_scalar_3 = pl.read(comb, [t0, 3 * HC_MULT + out_h])
        single_weighted_3 = pl.mul(res_3, comb_scalar_3)
        single_y = pl.add(single_y, single_weighted_3)
        single_result = pl.cast(single_y, pl.BF16, mode="rint")
        y_flat[t0 : t0 + 1, out_h * D + n0 : out_h * D + n0 + COLS] = pl.set_validshape(single_result, 1, COLS)
    return y_flat


@pl.jit.inline(auto_scope=False)
def o_dequant_hc_post_stage(
    partials: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * D], pl.INT32],
    act_scale_dq: pl.Tensor[[O_GROUPS, SCALE_ROWS_DYN], pl.FP32],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual_flat: pl.Tensor[[T_DYN, HC_MULT * D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    out_flat: pl.Out[pl.Tensor[[T_DYN, HC_MULT * D], pl.BF16]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV按实际token条带输出；每行先等八组partial及该行post/comb，不含隐式同步。"""
    tokens = pl.tensor.dim(out_flat, 0)
    # 权重scale与token无关；用Tile reshape，不能对GM Tensor直接reshape再load。
    wb_scale = pl.load(wo_b_scale, [0], [D])
    wb_scale_chunk = pl.reshape(wb_scale, [1, D])
    for b_tb in pl.range(worker_id, tokens, WORKERS):
        # 保留每token1x4096、group递增stage=2流水；仅变更独立token的owner。
        acc = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
        for act_g in pl.pipeline(O_GROUPS, stage=2):
            p_col0 = act_g * D
            p_g = pl.load(partials, [b_tb, p_col0], [1, D])
            g_scale_scalar = pl.read(act_scale_dq, [act_g, b_tb])
            p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
            acc = pl.add(acc, pl.mul(p_g_f32, g_scale_scalar))
        out_t = pl.col_expand_mul(acc, wb_scale_chunk)
        # 声明仅供inline参数的Tensor元数据追踪，随后的cast覆盖它；不要求GM临时。
        # 这次BF16舍入是原HCpost输入的数值边界，不可与最后一次BF16合并。
        out_bf16 = pl.create_tensor([1, D], dtype=pl.BF16)
        out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
        out_flat = hc_post_row_block(out_bf16, residual_flat, post, comb, out_flat, b_tb, 0, D)
    return out_flat
