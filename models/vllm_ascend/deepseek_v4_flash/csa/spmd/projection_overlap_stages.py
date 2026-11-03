# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态CSA投影分支的核组分工：让独立矩阵链在同一SPMD内并行推进。

整体思路与阅读顺序
    外层已完成HC、QA、Indexer压缩和query投影，并发布对应阶段。本文件先
    按原24个AIC分工先完成主压缩投影和head权重，再分为两个不重叠集合：
    前R核负责compact/query Hadamard，剩余24-R核负责主Q_B。主压缩先发出，
    让AIV在后续Q_B/Hadamard期间完成其pool/state/cache；KV投影放在两分支
    之后，仍分配原24个逻辑worker，实际非空列块从Q_B核集合起轮转。
    先读主函数的共同前段和两分支，再看
    query Hadamard的四token数学块。R是区间策略的constexpr，不是batch白名单。

输入输出、动态范围与工作区
    normed是BF16[T,4096]，主Q_B仍读原pad128 INT8 QR和Native NZ权重；
    query Hadamard读取已有BF16[pad16(T)*64,128]视图。T由设备Tensor维度取得，
    compact行数与请求数量仍读本轮device描述符。所有输出使用原有GM缓冲，
    不增加Host回读、静态任务表或新的工作区。每个helper仍遍历本核负责的
    全部动态行/列块；核组分工固定不等于每核实际工作量固定。

分工与同步合同
    辅助分支的逻辑worker等于物理AIC编号；Q_B后部连续24-R个物理编号对
    24-R取模，恰好得到完整逻辑worker集合的一个循环置换。
    每阶段仍由全部24个AIC各发布一次：没有该阶段业务的核先发布空完成，
    真正producer必须写回全部业务后发布。原AIV的等待集合是新producer集合
    的超集，因此本候选保留其较保守的等待和缓存获取，不伪造数据就绪。
    head权重在等待Indexer RoPE/compact norm之前发布；AIV先完成Indexer
    压缩归一化，再在主Q_B期间消费主压缩。原KV的norm/cache只在Score之后
    由实际有工作的AIV处理，不挡住Score消费者。Q_B分支在计算前发布其两个
    Hadamard空阶段，避免AIV等待没有此项工作的核。KV逻辑worker采用循环
    置换，AIV在Score之后保守等待全部24个已发布角色，不沿用旧物理owner公式。

数值与验证边界
    主压缩、权重、KV和Q_B复用原数学helper，只变执行位置和worker条带。
    query Hadamard每四token合为M256，保持每行完整K128和FP32输出；最后
    不足四token时使用实际有效行，不读取容量尾部的未初始化RoPE。四token
    对应64个head/每token及64KiB Left容量，不是某个测试shape。与之前失败
    候选的差别是同时前移主压缩生产/消费，并把原KV后处理移出Score前驱链；
    既不把KV压到4核，也不只前移生产者。七档完整同轮验证已取得一致改善；
    默认选择与精度/性能验证边界见policy.py，不将控制图检查当性能证据。
"""

import pypto.language as pl

from .compressor_projection_stages import compressor_project_stage
from .indexer_compressor_output_stages import indexer_compressor_hadamard_stage
from .indexer_query_stages import indexer_weights_project_stage
from .kv_stages import kv_project_stage
from .phase_sync import publish_phase, wait_producer, wait_row_owners
from .query_projection_stages import QUANT_WEIGHT_LAYOUT, q_b_stage
from .runtime_abi import (
    CORES,
    GROUPS,
    ICPAD,
    IDX_HEAD_ROWS,
    IDX_ROPE_HEAD_ROWS,
    KV_ROWS,
    MPROJ,
    PHASE_IDX_HADAMARD,
    PHASE_IDX_NORM,
    PHASE_IDX_QUERY_HAD,
    PHASE_IDX_QUERY_ROPE,
    PHASE_KV_PROJECT,
    PHASE_MAIN_PROJECT,
    PHASE_META,
    PHASE_QB,
    PHASE_WEIGHTS_PROJECT,
    QPROJ_ROWS,
    QR_ROWS,
    VECTORS,
    WEIGHTS_PARTIAL_ROWS,
    WORDS,
    B,
    T,
)


@pl.jit.inline(auto_scope=False)
def _query_hadamard_four_tokens(
    query_rope: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """M256只合并独立行；每行的K128乘加、输出舍入和head顺序保持不变。"""
    rows = tokens * 64
    weight = hadamard[0:128, 0:128]
    # 每块至多四个token；末块仍以真实T裁剪GM读取和写回，不用pad容量当有效行。
    for block in pl.range(worker, (rows + 255) // 256, WORKERS):
        first = block * 256
        valid = pl.min(256, rows - first)
        source = pl.slice(query_rope, [256, 128], [first, 0], valid_shape=[valid, 128])
        result = pl.matmul(source, weight, out_dtype=pl.FP32)
        # 此helper使用Tensor级数学，assemble由lowering生成保留有效行的Tile store。
        query_had = pl.assemble(query_had, result, [first, 0])
    return query_had


@pl.jit.inline(auto_scope=False)
def projection_overlap_stage(
    normed: pl.Tensor[[T, 4096], pl.BF16],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    weights_proj: pl.Tensor[[4096, 64], pl.BF16],
    weights_partial: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
    query_rope: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    qr_matmul: pl.Tensor[[QR_ROWS, 1024], pl.INT8],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_count: pl.Tensor[[1], pl.INT32],
    idx_normed: pl.Tensor[[ICPAD, 128], pl.BF16],
    idx_had: pl.InOut[pl.Tensor[[ICPAD, 128], pl.FP32]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
    worker: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    BRANCH_GROUPS: pl.constexpr,
):
    """0<R<24；Host只选择R，动态行数及所有输入依赖仍留在核内。"""
    tokens = pl.tensor.dim(normed, 0)
    # 主压缩只依赖normed，提早发布可让其AIV消费者与后续Q_B重叠。
    # 保留原M64/N64和完整K归约，各核只发布自己的结果，不等待其它AIC。
    main_value, main_score = compressor_project_stage(
        normed, main_wkv, main_wgate, main_value, main_score, worker, GROUPS
    )
    signal = publish_phase(signal, PHASE_MAIN_PROJECT, worker, CORES, generation)
    weights_partial = indexer_weights_project_stage(normed, weights_proj, weights_partial, worker, GROUPS)
    signal = publish_phase(signal, PHASE_WEIGHTS_PROJECT, worker, CORES, generation)
    if worker < BRANCH_GROUPS:
        # 本核从不写Q_B，不参与其真实producer集合；空发布不能代替后20核的发布。
        signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)
        # compact norm已由AIV优先准备，先处理短的compact Hadamard，避免
        # key量化/scale发布再等待长query Hadamard之后的一个尾阶段。
        planner = pl.min((tokens + 7) // 8, VECTORS - 1)
        signal = wait_producer(signal, PHASE_META, GROUPS + planner, CORES, generation)
        requests = pl.tensor.dim(idx_lengths, 0)
        signal = wait_row_owners(signal, PHASE_IDX_NORM, CORES, generation, 0, requests, 1, VECTORS, GROUPS, 1)
        compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
        idx_had = indexer_compressor_hadamard_stage(idx_normed, hadamard, idx_had, compact_count, worker, BRANCH_GROUPS)
        signal = publish_phase(signal, PHASE_IDX_HADAMARD, worker, CORES, generation)
        query_rope_tasks = ((tokens + 7) // 8) * 16
        signal = wait_row_owners(
            signal,
            PHASE_IDX_QUERY_ROPE,
            CORES,
            generation,
            0,
            query_rope_tasks,
            1,
            VECTORS,
            GROUPS,
            1,
        )
        query_had = _query_hadamard_four_tokens(query_rope, hadamard, query_had, tokens, worker, BRANCH_GROUPS)
        signal = publish_phase(signal, PHASE_IDX_QUERY_HAD, worker, CORES, generation)
    else:
        # 本核不写两个Hadamard输出，先完成空参与合同；主压缩和head权重
        # 已真正计算并发布。KV留在本分支之后，不向消费者伪作就绪。
        signal = publish_phase(signal, PHASE_IDX_QUERY_HAD, worker, CORES, generation)
        signal = publish_phase(signal, PHASE_IDX_HADAMARD, worker, CORES, generation)
        # 后部连续N个物理核对N取模是双射，既覆盖全部列块，也让NZ地址从
        # 非负表达式构成。消费者等待全部Q_B实际生产者，不依赖置换前的编号。
        qb_worker = worker % (GROUPS - BRANCH_GROUPS)
        q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, qb_worker, GROUPS - BRANCH_GROUPS)
        signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)
    # 同一组24个KV逻辑worker做循环置换，保持全部M/N块唯一归属。常见的
    # 最多12个非空worker落在Q_B一侧，不继续堆到Hadamard辅助核；即便
    # 其它动态形状需要更多波次，也仍按完整原KV算法覆盖，未冻结任务数。
    kv_worker = (worker + GROUPS - BRANCH_GROUPS) % GROUPS
    kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, kv_worker, GROUPS)
    signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)
    return main_value, main_score, weights_partial, query_had, q_proj, idx_had, kv_fp32, signal
