# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 的两路压缩投影：保留原数学、显式 AIC 分工的核内阶段。

输入输出、布局与动态容量
    x 为归一化后的 ND BF16[T,4096]，不要求 T=B*S，也不读取请求 metadata。
    每个 token 独立投影出重叠压缩窗口使用的 value 和 score：主 Compressor
    的 wkv/wgate 为 ND BF16[1024,4096]，两份输出为 FP32[PAD_MAIN,1024]；
    Indexer Compressor 对应为 ND BF16[256,4096]、FP32[PAD_INDEXER,256]。
    两张权重均按 [输出列,K] 存放，matmul 使用 b_trans=True，不引入 NZ 转换。
    T、PAD_MAIN、PAD_INDEXER 均保留 pl.dynamic；前者决定实际工作量，后两者
    只是物理容量，分别至少 align_up(T,64)、align_up(T,16)。输出只承诺前 T 行
    的数值，padding 不保证初始化。value/score 工作区与输入、权重不得非法别名。

算法、分核与阅读顺序
    先读 compressor_project_stage 的 M64/N64 主投影，再读
    indexer_compressor_project_stage 的 M16/N32 投影。两者均按行块优先、
    列块其次把二维 M/N 网格编号为 idx；worker_id+k*WORKERS 领取完整网格点，
    每个点独占 value/score 的同一输出块。每核完成全部 K4096 的八个 K512
    分片，不做跨核 split-K、atomic 累加或归约重排。stage=2 只保留原搬运流水。
    worker_id 是 AIC 角色内的零起点编号，要求 WORKERS>0 且
    0<=worker_id<WORKERS。T=0 或核数大于工作块数时，空核不读权重/输入。

为何两段不合并成一种 accumulator 初始化
    主投影 M64 跨四个 16 行分形，动态尾行需要首个 pl.matmul 建立 compact
    accumulator 的有效行距；不能改成 create_tensor+matmul_acc(init_cond)。
    Indexer 的 M16 保留原 matmul_acc(init_cond=kb==0)。两段 K 顺序和尾块
    valid_shape 均来自 canonical 多 task 数学，未借迁移改变舍入或数值策略。

阶段依赖、工作区与边界
    本文件只提供 inline helper，不创建 spmd/task，不通知或等待，也不读写
    state/cache。外层先等 x_norm 可见，再运行投影；pool 消费该行所需的所有
    列块前，须等待相应 AIC 生产者。helper 返回两个 Tensor 只表达 GM 别名，
    不代表其他核已完成或数据已经发布。原 late_dep/chain_dep 的顺序责任也在
    外层保留，不能因本文件没有 TASK_ID 就默认删除。每个工作区覆盖整份 T-pad，
    本实现不循环复用槽位，因而不引入尚未建立的消费者 ACK。
    最后 M 块只加载 valid_rows；源码写回视图仍保留原完整 M 块，因此调用方
    必须提供相应 pad 容量。tile 是算法分块，不是 batch/sequence 测试点白名单。
    当前只迁移投影；请求池化、RMS/RoPE、压缩 cache 与整层同步另行组装验证。
"""

import pypto.language as pl

from ..common.config import FLASH as M

# 只取模型固有维度；不导入含设备探针的 nz_mode，也不读取 DECODE_BATCH/SEQ。
D = M.hidden_size
CMP_OUT_DIM = 2 * M.head_dim
INDEXER_OUT_DIM = 2 * M.index_head_dim
K_TILE = 512
CMP_M_TILE = 64
CMP_N_TILE = 64
INDEXER_M_TILE = 16
INDEXER_N_TILE = 32

T_DYN = pl.dynamic("CSA_COMPRESSOR_PROJECTION_T")
MAIN_PAD_ROWS_DYN = pl.dynamic("CSA_COMPRESSOR_PROJECTION_MAIN_PAD_ROWS")
INDEXER_PAD_ROWS_DYN = pl.dynamic("CSA_COMPRESSOR_PROJECTION_INDEXER_PAD_ROWS")


@pl.jit.inline(auto_scope=False)
def compressor_project_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[CMP_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[CMP_OUT_DIM, D], pl.BF16],
    kv_proj_pad: pl.Tensor[[MAIN_PAD_ROWS_DYN, CMP_OUT_DIM], pl.FP32],
    score_proj_pad: pl.Tensor[[MAIN_PAD_ROWS_DYN, CMP_OUT_DIM], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """主压缩投影；一个 AIC worker 独占若干 M64/N64 块及两份完整 K 归约。"""
    bs = pl.tensor.dim(x, 0)
    t_matmul = ((bs + CMP_M_TILE - 1) // CMP_M_TILE) * CMP_M_TILE
    # 16 个列块覆盖1024维；M方向按真实T向上取整，绝不按pad容量派工作。
    for idx in pl.range(worker_id, t_matmul * CMP_OUT_DIM // (CMP_M_TILE * CMP_N_TILE), WORKERS):
        global_row0 = (idx // (CMP_OUT_DIM // CMP_N_TILE)) * CMP_M_TILE
        o0 = (idx % (CMP_OUT_DIM // CMP_N_TILE)) * CMP_N_TILE
        x_rows = pl.min(CMP_M_TILE, bs - global_row0)
        kv_acc = pl.create_tensor([CMP_M_TILE, CMP_N_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([CMP_M_TILE, CMP_N_TILE], dtype=pl.FP32)
        # 仅对K搬运做两级流水；八块按0..7顺序更新同一accumulator，不移动加法。
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_tile = pl.slice(x, [CMP_M_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            wkv_tile = wkv[o0 : o0 + CMP_N_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + CMP_N_TILE, k0 : k0 + K_TILE]
            # 动态M64尾块的物理acc行距是ceil(x_rows/16)*16。首块matmul会记录
            # 该compact属性；若用create_tensor的64行距直接累加，会破坏
            # AccCompactValid。即使某次T正好整除64，也保留同一源级分支。
            if kb == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
        # 保留原目标M64视图；caller预留pad64，消费者只读取真实token行。
        kv_proj_pad[global_row0 : global_row0 + CMP_M_TILE, o0 : o0 + CMP_N_TILE] = kv_acc
        score_proj_pad[global_row0 : global_row0 + CMP_M_TILE, o0 : o0 + CMP_N_TILE] = score_acc
    return kv_proj_pad, score_proj_pad


@pl.jit.inline(auto_scope=False)
def indexer_compressor_project_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    kv_proj_pad: pl.Tensor[[INDEXER_PAD_ROWS_DYN, INDEXER_OUT_DIM], pl.FP32],
    score_proj_pad: pl.Tensor[[INDEXER_PAD_ROWS_DYN, INDEXER_OUT_DIM], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """Indexer压缩投影；M16/N32、BF16转置权重与原init_cond累加顺序不变。"""
    bs = pl.tensor.dim(x, 0)
    t_matmul = ((bs + INDEXER_M_TILE - 1) // INDEXER_M_TILE) * INDEXER_M_TILE
    # 每个M16块有8个列块；worker条带覆盖全网格，尾部不足16行也只领取一次。
    for idx in pl.range(worker_id, t_matmul * INDEXER_OUT_DIM // (INDEXER_M_TILE * INDEXER_N_TILE), WORKERS):
        global_row0 = (idx // (INDEXER_OUT_DIM // INDEXER_N_TILE)) * INDEXER_M_TILE
        o0 = (idx % (INDEXER_OUT_DIM // INDEXER_N_TILE)) * INDEXER_N_TILE
        kv_acc = pl.create_tensor([INDEXER_M_TILE, INDEXER_N_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([INDEXER_M_TILE, INDEXER_N_TILE], dtype=pl.FP32)
        # M16只跨一个分形，保留原init_cond；不能照抄主M64首块策略来改数学。
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(INDEXER_M_TILE, bs - global_row0)
            x_tile = pl.slice(x, [INDEXER_M_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            wkv_tile = wkv[o0 : o0 + INDEXER_N_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + INDEXER_N_TILE, k0 : k0 + K_TILE]
            kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(kb == 0))
            score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(kb == 0))
        # 工作区pad16承接原M16写回视图；pad内容不作为后续池化的有效输入。
        kv_proj_pad[global_row0 : global_row0 + INDEXER_M_TILE, o0 : o0 + INDEXER_N_TILE] = kv_acc
        score_proj_pad[global_row0 : global_row0 + INDEXER_M_TILE, o0 : o0 + INDEXER_N_TILE] = score_acc
    return kv_proj_pad, score_proj_pad
