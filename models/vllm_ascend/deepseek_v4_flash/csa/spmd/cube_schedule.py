# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""完整动态CSA的AIC编排：阶段数据流、分工与阅读顺序。

本文件和vector_schedule.py是在同一个MIX任务中并发运行的两种角色，不是
两个独立kernel。24个AIC按真实T及device请求计划循环，数学tile由constexpr
策略选择；没有输入规格白名单、固定S推导请求或冻结的历史DAG。

默认阅读顺序：HC投影→QA→Indexer压缩/query投影→projection_overlap_stage
中的主压缩/head权重、Hadamard与Q_B分核、KV→Indexer Score→Attention→O_A/O_B。矩阵计算调用各
*_stages.py的inline函数；本文件只描述何时允许消费数据，不另写数学。
HC行依赖路径按每个AIC实际消费的M16行等待M8输入，不等无关行；
每个producer仍发布自己的完整阶段，消费者等待后保留业务缓存获取。
先发投影再等其AIV消费者，避免融合后造成环形等待。默认主压缩先发出，
让AIV在4核Hadamard/20核Q_B期间消费；KV后处理退出Score输入链。
COOPERATIVE_PREFIX关闭时保留旧次序：head权重/KV→query Hadamard→Q_B→
compact Hadamard→主压缩。两种压缩独立state，仍保留真实owner及共享读完约束。
输出尾段按策略选择整阶段或逐输出组通知；逐组路径不改变本核O_A任务序，
已做完自己O_A的AIC可读就绪量化组做O_B，而不等待其它组完成量化。

布局：x_fp32[T,16384]、normed[T,4096]；QA与KV输出pad16，QB为pad128。
qr与q_heads是调用层绑定的GM视图，q_heads必须恰好64*T行；compact实际行数
读设备计数，不能用其pad容量替代。主/Indexer state、cache始终是Native
存储；本角色只通过显式等候读取其他核产物。输出packed为8组，组stride由
其真实容量除8得到。完整的外部容量/别名合同见entry.py及各数学文件头部。

同步：每核begin一次，即使本轮无工作也不能跳过。phase是单写者整阶段
完成标记，消费者只等对应owner；发布和等待都包含业务流水/缓存维护。
Score与Attention内部还有组内环槽READY/ACK，必须让双AIV同时推进，不能
等AIV整段结束才启动AIC。Scalar完成标记不能代替MTE/FIX完成或读侧缓存维护。

当前入口覆盖Cube Score策略；完整CPU后端、真机精度及性能以验证记录为准。
前段已有的设备PASS不能证明本文件新接入的全链路或七档性能达标。
"""

import pypto.language as pl

from .attention_dependencies import wait_attention_plan_rows
from .attention_stream_stages import attention_qk_pv_cube_stage
from .compressor_projection_stages import compressor_project_stage, indexer_compressor_project_stage
from .hc_stages import hc_project_range
from .indexer_compressor_output_stages import indexer_compressor_hadamard_stage
from .indexer_query_stages import (
    indexer_query_hadamard_stage,
    indexer_query_project_stage,
    indexer_weights_project_stage,
)
from .indexer_score_stages import indexer_score_cube_stage
from .kv_stages import kv_project_stage
from .output_projection_a_stages import o_projection_a_groups_stage_nz, o_projection_a_stage_nz
from .output_projection_b_stages import o_b_groups_stage, o_b_stage
from .phase_sync import begin_invocation, publish_phase, wait_producer, wait_row_owners
from .projection_overlap_stages import projection_overlap_stage
from .qkv_stages import BF16_WEIGHT_LAYOUT, q_a_stage
from .query_projection_stages import QUANT_WEIGHT_LAYOUT, q_b_stage
from .runtime_abi import (
    ATTN_PLAN_ROWS,
    COEFF_COLS,
    COEFF_ROWS,
    CORES,
    GROUP_OFFSET_ROWS,
    GROUPS,
    HEAD_RING_ROWS,
    ICACHE_BYTES,
    ICACHE_PAGES,
    ICPAD,
    IDX_HEAD_ROWS,
    IDX_PROJ_ROWS,
    IDX_ROPE_HEAD_ROWS,
    INDEX_TRANSFER_COLS,
    INDEX_TRANSFER_ROWS,
    INDEXER_COLS,
    IPROJ,
    IQ,
    KEY0_ROWS,
    KEY64_ROWS,
    KV_RING_ROWS,
    KV_ROWS,
    MPROJ,
    O_MM_ROWS,
    P16,
    PACK_ROWS,
    PARTIAL,
    PHASE_ATTN_CUBE,
    PHASE_ATTN_VECTOR,
    PHASE_COEFF,
    PHASE_IDX_HADAMARD,
    PHASE_IDX_NORM,
    PHASE_IDX_PROJECT,
    PHASE_IDX_QUERY_HAD,
    PHASE_IDX_QUERY_PROJECT,
    PHASE_IDX_QUERY_QUANT,
    PHASE_IDX_QUERY_ROPE,
    PHASE_IDX_SCALE,
    PHASE_INPUT,
    PHASE_KV_PROJECT,
    PHASE_MAIN_PROJECT,
    PHASE_META,
    PHASE_NORM,
    PHASE_O_QUANT,
    PHASE_O_QUANT_GROUP_BASE,
    PHASE_OA,
    PHASE_OA_GROUP_BASE,
    PHASE_OB,
    PHASE_PARTIAL,
    PHASE_QA,
    PHASE_QB,
    PHASE_QR,
    PHASE_QUERY,
    PHASE_SCORE_CUBE,
    PHASE_WEIGHTS_PROJECT,
    QPROJ_ROWS,
    QR_ROWS,
    QUERY_HEAD_ROWS,
    VECTORS,
    WEIGHTS_PARTIAL_ROWS,
    WORDS,
    B,
    T,
)

# The existing N128 output grid has 64 jobs on 24 physical AICs: the first
# sixteen get three O_A jobs, the last eight get two. Software O_B ranks
# compensate with two/four jobs respectively; they do not add physical cores.
OUTPUT_BALANCE_HEAVY_CORES = 16
OUTPUT_BALANCE_VIRTUAL_WORKERS = 32
OUTPUT_BALANCE_LAST_RANK = OUTPUT_BALANCE_VIRTUAL_WORKERS - 1
OUTPUT_BALANCE_COLUMNS_PER_GROUP = 8


@pl.jit.incore
def cube_schedule(
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    x_fp32: pl.Tensor[[T, 16384], pl.FP32],
    partials: pl.InOut[pl.Tensor[[PARTIAL, 32], pl.FP32]],
    normed: pl.Tensor[[T, 4096], pl.BF16],
    qa: pl.InOut[pl.Tensor[[P16, 1024], pl.FP32]],
    qr: pl.Tensor[[T, 1024], pl.INT8],
    qr_matmul: pl.Tensor[[QR_ROWS, 1024], pl.INT8],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    positions: pl.Tensor[[T], pl.INT64],
    idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
    idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
    idx_value: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    idx_score: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_count: pl.Tensor[[1], pl.INT32],
    idx_normed: pl.Tensor[[ICPAD, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    idx_had: pl.InOut[pl.Tensor[[ICPAD, 128], pl.FP32]],
    idx_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    q_heads: pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
    idx_query_acc: pl.InOut[pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32]],
    idx_query_rope_heads: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    idx_query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    idx_query_i8: pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8],
    weights_proj: pl.Tensor[[4096, 64], pl.BF16],
    weights_partial: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    group_plan: pl.Tensor[[8], pl.INT32],
    coefficients: pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16],
    idx_key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    idx_key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    indexer_transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],
    attention_mask: pl.Tensor[[ATTN_PLAN_ROWS, 16], pl.INT32],
    kv_ring: pl.Tensor[[KV_RING_ROWS, 512], pl.BF16],
    score_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.FP32]],
    probability_ring: pl.Tensor[[HEAD_RING_ROWS, 128], pl.BF16],
    pv_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 512], pl.FP32]],
    packed: pl.Tensor[[PACK_ROWS, 4096], pl.BF16],
    ffts: pl.Tensor[[256], pl.INT64],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, BF16_WEIGHT_LAYOUT],
    o_r: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.FP32]],
    o_quant: pl.Tensor[[O_MM_ROWS, 8192], pl.INT8],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    o_partials: pl.InOut[pl.Tensor[[O_MM_ROWS, 32768], pl.INT32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    ROOTS: pl.constexpr,
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
    OUTPUT_GROUP_PIPELINE: pl.constexpr,
    HC_ROW_DEPENDENCIES: pl.constexpr,
    QA_M16_PARALLEL: pl.constexpr = False,
    COOPERATIVE_PREFIX: pl.constexpr = False,
    SPLIT_SCORE_ROWS: pl.constexpr = False,
    OUTPUT_WORK_BALANCE: pl.constexpr = False,
):
    # 恢复原分task五份流式权重的BYPASS；预定义incore使用函数属性表达。
    # 当前PyPTO此属性只接受(参数序号,策略整数)，1表示CachePolicy.BYPASS。
    # 1=wq_a、8=wq_b、10=wkv、50=wo_a、53=wo_b；ABI审计必须检查对应关系，
    # 修改签名时同步更新。该属性只影响GM读的L2提示，不改变发布/等待协议。
    pl.func_attr({"cache_policy": [(1, 1), (8, 1), (10, 1), (50, 1), (53, 1)]})
    partials_gm = partials
    qa_gm = qa
    q_proj_gm = q_proj
    kv_fp32_gm = kv_fp32
    main_value_gm = main_value
    main_score_gm = main_score
    idx_value_gm = idx_value
    idx_score_gm = idx_score
    idx_had_gm = idx_had
    idx_query_acc_gm = idx_query_acc
    idx_query_had_gm = idx_query_had
    weights_partial_gm = weights_partial
    indexer_transfer_gm = indexer_transfer
    score_ring_gm = score_ring
    pv_ring_gm = pv_ring
    o_r_gm = o_r
    o_partials_gm = o_partials
    signal_gm = signal
    worker = pl.tile.get_block_idx()
    signal, generation = begin_invocation(signal, worker)
    pl.system.set_ffts(ffts)
    tokens = pl.tensor.dim(x_fp32, 0)
    if HC_ROW_DEPENDENCIES:
        # HC投影task=(M16行块*4+K4096分片)，task=worker+wave*24。
        # 一个task只消费对应两块M8输入。wave每次把行块推进6，故M8 owner
        # 推进12；48个AIV的owner周期为4波。超过四波也只等同一批producer
        # 的整阶段完成信号，不能对同槽逐tile重复发布。尾部不足8/16行按T裁剪。
        hc_tasks = ((tokens + 15) // 16) * 4
        hc_waves = (hc_tasks + GROUPS - 1 - worker) // GROUPS
        for hc_wave in pl.range(pl.min(hc_waves, VECTORS // (2 * (GROUPS // 4)))):
            hc_first = ((worker + hc_wave * GROUPS) // 4) * 16
            signal = wait_row_owners(
                signal, PHASE_INPUT, CORES, generation,
                hc_first, pl.min(16, tokens - hc_first), 8, VECTORS, GROUPS, 1
            )
    else:
        signal = wait_row_owners(
            signal,
            PHASE_INPUT,
            CORES,
            generation,
            0,
            tokens,
            8,
            VECTORS,
            GROUPS,
            1,
        )
    partials = hc_project_range(x_fp32, hc_fn, partials, worker, GROUPS)
    signal = publish_phase(signal, PHASE_PARTIAL, worker, CORES, generation)
    signal = wait_row_owners(
        signal,
        PHASE_NORM,
        CORES,
        generation,
        0,
        tokens,
        8,
        VECTORS,
        GROUPS,
        1,
    )
    # cluster 的 GM 合同显式保留原参数别名；q_a_stage 的动态 M 分支只写同一缓冲。
    # Host仅在G6的原两M16尾块区间启用；T与全局M16任务数仍由device维度求得。
    qa = q_a_stage(normed, wq_a, qa, worker, GROUPS, QA_M16_PARALLEL)
    signal = publish_phase(signal, PHASE_QA, worker, CORES, generation)
    # 先准备Indexer压缩，再做Indexer query；这两条链决定长历史Score何时可启动。
    # 独立KV投影填在等待query RoPE之前，随后发Hadamard，再做主Q_B。
    # AIV在Q_B期间可完成Indexer量化/系数；所有K链、owner和权重保持不变。
    idx_value, idx_score = indexer_compressor_project_stage(
        normed,
        idx_wkv,
        idx_wgate,
        idx_value,
        idx_score,
        worker,
        GROUPS,
    )
    signal = publish_phase(signal, PHASE_IDX_PROJECT, worker, CORES, generation)
    signal = wait_row_owners(
        signal,
        PHASE_QR,
        CORES,
        generation,
        0,
        tokens,
        8,
        VECTORS,
        GROUPS,
        1,
    )
    # Indexer query投影与head权重都只依赖前段结果；先发出需要AIV处理的投影。
    # 不能在写出这些结果之前等待Indexer归一化或量化，否则会形成等待环。
    idx_query_acc = indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, worker, GROUPS)
    signal = publish_phase(signal, PHASE_IDX_QUERY_PROJECT, worker, CORES, generation)
    if COOPERATIVE_PREFIX:
        # 主压缩先生产；4核Hadamard/20核Q_B并行，KV仍由24个逻辑worker
        # 分摊。AIV同时调整消费者顺序，不能单独启用这一侧。
        (
            main_value, main_score, weights_partial, idx_query_had,
            q_proj, idx_had, kv_fp32, signal,
        ) = projection_overlap_stage(
            normed, main_wkv, main_wgate, main_value, main_score,
            weights_proj, weights_partial, idx_query_rope_heads, hadamard,
            idx_query_had, qr_matmul, wq_b, q_proj, idx_lengths, idx_count,
            idx_normed, idx_had, wkv, kv_fp32, signal, worker, generation, 4,
        )
    else:
        weights_partial = indexer_weights_project_stage(normed, weights_proj, weights_partial, worker, GROUPS)
        signal = publish_phase(signal, PHASE_WEIGHTS_PROJECT, worker, CORES, generation)
        # normed已经等待HC_NORM；KV与QA读取同一BF16输入，wkv固定ND。
        # 原GM别名显式返回cluster，实际KV写回与通知次序必须另查编译产物。
        kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, worker, GROUPS)
        signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)

        # query的8行×4head反量化块共有ceil(T/8)*16个，不把T误当作其owner数量。
        query_rope_tasks = ((tokens + 7) // 8) * 16
        signal = wait_row_owners(
            signal, PHASE_IDX_QUERY_ROPE, CORES, generation, 0, query_rope_tasks, 1, VECTORS, GROUPS, 1
        )
        idx_query_had = indexer_query_hadamard_stage(
            idx_query_rope_heads,
            hadamard,
            idx_query_had,
            tokens,
            worker,
            GROUPS,
        )
        signal = publish_phase(signal, PHASE_IDX_QUERY_HAD, worker, CORES, generation)
        # Q_B列块n归n%24；每个owner处理所有实际M128行块。返回原GM别名给cluster。
        q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, worker, GROUPS)
        signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)

        # 主QB之后优先完成已就绪的Indexer compact Hadamard，随后做独立主压缩。
        # 本轮META早于投影完成；仍显式获取本次计划与所有IDX_NORM生产者。
        # 不假定CPU时序已保证可见性。每次从真实T重新计算同一planner owner。
        planner = pl.min((tokens + 7) // 8, VECTORS - 1)
        signal = wait_producer(signal, PHASE_META, GROUPS + planner, CORES, generation)
        requests = pl.tensor.dim(idx_lengths, 0)
        signal = wait_row_owners(signal, PHASE_IDX_NORM, CORES, generation, 0, requests, 1, VECTORS, GROUPS, 1)
        compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
        idx_had = indexer_compressor_hadamard_stage(
            idx_normed,
            hadamard,
            idx_had,
            compact_count,
            worker,
            GROUPS,
        )
        signal = publish_phase(signal, PHASE_IDX_HADAMARD, worker, CORES, generation)

        # compact Hadamard已发布；AIV可在这段主压缩投影期间量化并提交Indexer
        # cache/scale。主压缩的数学、owner和累加顺序均不变。若别的形状下
        # IDX_NORM较晚，前移的等待可能抵消收益，因此必须实测，不能假定免费。
        main_value, main_score = compressor_project_stage(
            normed,
            main_wkv,
            main_wgate,
            main_value,
            main_score,
            worker,
            GROUPS,
        )
        signal = publish_phase(signal, PHASE_MAIN_PROJECT, worker, CORES, generation)


    # Score消费query INT8、系数和Native key/scale。Key写回与scale RMW不是一件事：
    # 后者按物理页分配唯一AIV写者；全部48个writer发布后，cache才可整体消费。
    signal = wait_row_owners(signal, PHASE_IDX_QUERY_QUANT, CORES, generation, 0, tokens, 1, VECTORS, GROUPS, 1)
    groups = pl.cast(pl.read(group_plan, [0]), pl.INDEX)
    signal = wait_row_owners(signal, PHASE_COEFF, CORES, generation, 0, groups, 1, VECTORS, GROUPS, 1)
    signal = wait_row_owners(signal, PHASE_IDX_SCALE, CORES, generation, 0, VECTORS, 1, VECTORS, GROUPS, 1)
    indexer_transfer = indexer_score_cube_stage(
        idx_query_i8,
        coefficients,
        idx_key0,
        idx_key64,
        idx_cache,
        indexer_table,
        positions,
        idx_lengths,
        idx_bounds,
        group_offsets,
        group_plan,
        indexer_transfer,
        worker,
        QUERY_GROUP,
        QK_PANEL,
        SCORE_TILE,
        PREFETCH,
        TO_L0,
        BALANCED,
        0,
        1,
        SPLIT_SCORE_ROWS,
    )
    signal = publish_phase(signal, PHASE_SCORE_CUBE, worker, CORES, generation)

    # Attention有效块依赖本层TopK，不能合进前置的公共metadata扫描。
    # 每个实际读取的计划行都在对应Top-K根可见后发布；无需等待无关行。
    signal = wait_attention_plan_rows(signal, tokens, worker, generation)
    # query反量化head分组随T变化，首版等待48个明确发布的角色完成；没有猜测tile公式。
    signal = wait_row_owners(signal, PHASE_QUERY, CORES, generation, 0, VECTORS, 1, VECTORS, GROUPS, 1)
    score_ring, pv_ring = attention_qk_pv_cube_stage(
        q_heads,
        attention_mask,
        kv_ring,
        score_ring,
        probability_ring,
        pv_ring,
        ffts,
        worker,
        GROUPS,
        2,
    )
    signal = publish_phase(signal, PHASE_ATTN_CUBE, worker, CORES, generation)

    # 每个query归q%24组，其双AIV共同完成packed。这里只等真实活跃组的两个AIV，
    # 等待的是发布完packed，而不是只等AIC的PV矩阵指令已经发出。
    packed_workers = pl.min(tokens, GROUPS) * 2
    signal = wait_row_owners(signal, PHASE_ATTN_VECTOR, CORES, generation, 0, packed_workers, 1, VECTORS, GROUPS, 1)
    if OUTPUT_GROUP_PIPELINE:
        oa_per_group = ((tokens + 127) // 128) * (1024 // OA_COLS)
        # 固定8是模型的输出组数，不是B/S档位。每组内部任务数由本次T决定；
        # 首项公式保留全局logical%24分核及原每核组序，不把所有核绑到同一组。
        # 本核完成某组即可通知AIV；该AIC继续其它组的O_A，与已就绪组量化重叠。
        for oa_group in pl.range(8):
            oa_first = oa_group * oa_per_group
            oa_owned = worker + ((oa_first + GROUPS - 1 - worker) // GROUPS) * GROUPS
            if oa_owned < oa_first + oa_per_group:
                o_r = o_projection_a_groups_stage_nz(
                    packed, wo_a, o_r, tokens, worker, GROUPS, OA_COLS, oa_group, oa_group + 1
                )
                signal = publish_phase(signal, PHASE_OA_GROUP_BASE + oa_group, worker, CORES, generation)
        quant_per_group = (tokens + 31) // 32
        # 本核自己的O_A全部完成后，即可处理已量化的O_B组，不再等无关组。
        # O_B的NZ owner为(group*8+dc)%24，每个dc内部保留所有动态行块。
        # 量化末块发布仍包含INT8尾零，保留这些owner的原依赖。
        # NZ大M路径在O_B内部按真实T截断；其它路径仍可安全读取完整OB_ROWS。
        if OUTPUT_WORK_BALANCE:
            # In the existing N128, single-M128 domain, O_A gives cores 0..15
            # three jobs and cores 16..23 two. Use 32 logical O_B ranks on the
            # same 24 physical AICs: respectively two/four independent jobs.
            # These are software ranks, not additional SPMD participants.
            for ob_group in pl.range(8):
                ob_worker = worker + OUTPUT_BALANCE_HEAVY_CORES
                if worker >= OUTPUT_BALANCE_HEAVY_CORES:
                    ob_worker = worker - OUTPUT_BALANCE_HEAVY_CORES + (ob_group % 2) * OUTPUT_BALANCE_COLUMNS_PER_GROUP
                ob_first = ob_group * OUTPUT_BALANCE_COLUMNS_PER_GROUP
                ob_owned = ob_worker + (
                    (ob_first + OUTPUT_BALANCE_LAST_RANK - ob_worker) // OUTPUT_BALANCE_VIRTUAL_WORKERS
                ) * OUTPUT_BALANCE_VIRTUAL_WORKERS
                if ob_owned < ob_first + OUTPUT_BALANCE_COLUMNS_PER_GROUP:
                    signal = wait_row_owners(
                        signal, PHASE_O_QUANT_GROUP_BASE + ob_group, CORES, generation,
                        ob_group * quant_per_group, quant_per_group, 1, VECTORS, GROUPS, 1
                    )
                    o_partials = o_b_groups_stage(
                        o_quant, wo_b, o_partials, tokens, ob_worker, OUTPUT_BALANCE_VIRTUAL_WORKERS,
                        OB_ROWS, ob_group, ob_group + 1
                    )
        else:
            for ob_group in pl.range(8):
                ob_first = ob_group * 8
                ob_owned = worker + ((ob_first + GROUPS - 1 - worker) // GROUPS) * GROUPS
                if ob_owned < ob_first + 8:
                    signal = wait_row_owners(
                        signal, PHASE_O_QUANT_GROUP_BASE + ob_group, CORES, generation,
                        ob_group * quant_per_group, quant_per_group, 1, VECTORS, GROUPS, 1
                    )
                    o_partials = o_b_groups_stage(
                        o_quant, wo_b, o_partials, tokens, worker, GROUPS, OB_ROWS, ob_group, ob_group + 1
                    )
    else:
        # 保留整阶段路径供区间策略与同轮对照；两分支复用全部数学helper。
        o_r = o_projection_a_stage_nz(packed, wo_a, o_r, tokens, worker, GROUPS, OA_COLS)
        signal = publish_phase(signal, PHASE_OA, worker, CORES, generation)
        quant_tasks = 8 * ((tokens + 31) // 32)
        signal = wait_row_owners(signal, PHASE_O_QUANT, CORES, generation, 0, quant_tasks, 1, VECTORS, GROUPS, 1)
        o_partials = o_b_stage(o_quant, wo_b, o_partials, tokens, worker, GROUPS, OB_ROWS)
    signal = publish_phase(signal, PHASE_OB, worker, CORES, generation)
    return (
        partials_gm,
        qa_gm,
        q_proj_gm,
        kv_fp32_gm,
        main_value_gm,
        main_score_gm,
        idx_value_gm,
        idx_score_gm,
        idx_had_gm,
        idx_query_acc_gm,
        idx_query_had_gm,
        weights_partial_gm,
        indexer_transfer_gm,
        score_ring_gm,
        pv_ring_gm,
        o_r_gm,
        o_partials_gm,
        signal_gm,
    )
