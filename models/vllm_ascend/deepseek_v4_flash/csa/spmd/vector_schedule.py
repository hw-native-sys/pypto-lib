# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""完整动态CSA的双AIV编排：先读整体数据流，再沿阶段注释阅读。

48个AIV来自24个常驻MIX组，worker=group*2+lane。split_aiv让两条Vector
角色分别执行同一段源码，按worker领取不同的动态行、请求或head块；它不
自动建立跨核数据依赖。T和B均为Tensor动态轴，Q/L/position/slots是device
Tensor；不能用T/固定S反推请求，Native dummy及不等长请求由各阶段Q/L处理。

第一段：HC加宽/RMS并发布；根据ceil(T/8)选前段空闲AIV计算公共metadata。
满48核时选47号，先发布其widen，再计算计划，不永久保留一核。其他核继续
HC归约、门值、混合归一化，然后Sinkhorn；HC行依赖路径只等本核
消费的四个K片及一到两个RAW owner，数学、发布协议与其它阶段依赖不变。归一化先发布，让QA与Sinkhorn
重叠，再进入QR、RoPE与KV/query处理。这里只算
请求分组、compact前缀和scale页所有权；SCALE_PAGE_COALESCING启用时，
同一planner还证明整请求页表连续性，写group_offsets尾部2*B项供Score复用。
关闭不额外扫描证明页表或分配这块尾区；碎片页保持原逐页读取。TopK依赖的
Attention有效块计划仍保留到TopK之后。

第二段：默认先Indexer压缩pool/norm，再Indexer query反量化/head权重，
提前消费主压缩，然后Indexer query量化/系数、compact key量化与scale提交，
完成主Query反量化后进入Score。主压缩消费与AIC上的Q_B/Hadamard重叠，
原KV后处理留到本核Score完成后。没有等整段Score或跳过ACK。主/Indexer
两条压缩分支直接读写各自Native state。pool全部读完共享prefix历史后才
提交state。Indexer key并行写槽，32个FP16 scale所在64B区按物理page%48
分配唯一writer，同页更新顺序执行，避免读改写覆盖；其地址计划由早期planner准备。

第三段：与AIC并发执行Score/双槽ACK→实际owner处理原KV norm/cache→
跨叶TopK→Attention有效块计划→
与AIC并发执行三槽QK/PV的采KV/softmax/归并→分组量化→反量化和HCpost。
Attention查询几何复用开启时，plan在原64B mask行内同时发布request/position；
消费者取得同一PHASE_ATTN_PLAN后读取，避免逐候选块重复Q/L扫描。关闭时
两侧都采用原约定；该开关不能只作用于生产者或消费者一侧。
PLAN_PHYSICAL_ROWS统一转发给前置几何、最终plan及采KV；在容量支持时，
页合法性集中到最终选中地址的Vector检查，内部indices保存INT32物理行。
消费者仍检查最终物理行边界；无效行每轮写-1。SWA、能力外回退和通知顺序
不变。该开关只依赖已有Host描述符选区间，真实地址和有效范围仍在device求。
组流水启用时，量化只等本输出组的O_A owner；末块尾零属于同组通知合同，
每组独立槽位。HCpost仍等待所有O_B partial，保留八组反量化左折叠顺序。
数学、精度顺序、循环尾块均保留在各*_stages.py，编排只指定owner和依赖。

COOPERATIVE_PREFIX策略同步改变两种角色的顺序：Indexer压缩先于query RoPE，
主压缩pool/state/cache在主Q_B期间消费已前移的投影；主Query反量化完成后
进入Score。原KV的norm/cache在本核Score归还环槽后执行，无相关行块的AIV
直接继续Top-K。三条AIV阶段链拆到vector_prefix_stages.py；开关关闭维持旧次序。
这是完整前段协作策略，T/B和有效请求仍动态；默认由select_policy启用，
完整同轮收益及当前验证边界见policy.py与统一优化日志。

EARLY_ATTENTION_GEOMETRY由policy按工作量区间选择；开启时，
attention_geometry_prepare_stage在QR之后、ROPE发布之前计算本轮只读几何
和页范围，将结果暂放最终attention_indices的前五列。该区此时尚无最终
索引，Top-K获取既有ROPE阶段后读取；晚期plan等Top-K完成再覆盖。复用
已有buffer和发布，不新建Host适配或GM根，候选范围与页保护仍逐轮生效。

每个GM写缓冲有独立生存期，源码没有跨阶段复用以偷减依赖。phase发布要
在业务写回完成之后，wait之后做读侧维护；单写者免atomic不等于cache一致。
计划本轮只写一次、后续只读，signal另占512B隔离槽。组内环槽另有READY/ACK，
不能用整阶段信号替代。所有参与核每次begin一次，首次signal清零、单流/图
独占、失败后重置及代际上限由调用层保证。验证状态见统一优化日志。
"""

import pypto.language as pl

from .attention_dependencies import wait_attention_plan_rows
from .attention_plan_stages import (
    attention_geometry_prepare_stage,
    attention_inverse_rope_sign_stage,
    attention_slot_plan_stage,
)
from .attention_stream_stages import attention_qk_pv_vector_lane_stage
from .hc_stages import hc_comb_range, hc_gates_range, hc_mix_norm_range, hc_reduce_range, hc_widen_rms_range
from .indexer_cache_stages import indexer_compact_quant_cache_stage
from .indexer_group_stages import indexer_group_plan_stage, indexer_head_coefficients_stage
from .indexer_publish_stages import indexer_pair_publish_stage
from .indexer_query_stages import (
    indexer_query_dequant_rope_stage,
    indexer_query_quant_stage,
    indexer_weights_reduce_stage,
)
from .indexer_scale_layout_stages import indexer_scale_layout_plan_stage
from .indexer_scale_plan_stages import indexer_scale_page_commit_stage, indexer_scale_page_plan_stage
from .indexer_score_stages import indexer_score_vector_lane_stage
from .metadata_stages import compact_offsets_and_count_stage
from .output_projection_a_stages import o_projection_group_quant_stage, o_projection_groups_quant_stage
from .output_projection_b_stages import o_dequant_hc_post_stage
from .phase_sync import begin_invocation, publish_phase, wait_row_owners
from .qkv_stages import q_r_stage
from .query_projection_stages import q_dequant_rope_stage, q_rope_prepare_stage
from .runtime_abi import (
    ATTN_PLAN_ROWS,
    CACHE_ROWS,
    COEFF_COLS,
    COEFF_ROWS,
    COMPRESSED_COLS,
    CORES,
    GROUP_OFFSET_ROWS,
    GROUPS,
    HEAD_RING_ROWS,
    IC,
    ICACHE_BYTES,
    ICACHE_PAGES,
    ICPAD,
    IDX_HEAD_ROWS,
    IDX_PROJ_ROWS,
    IDX_ROPE_ROWS,
    IDX_SCALE_T_ROWS,
    INDEX_TRANSFER_COLS,
    INDEX_TRANSFER_ROWS,
    INDEXER_COLS,
    IPOOL,
    IPROJ,
    IQ,
    ISCALES,
    ISLOTS,
    ISTATE_ELEMS,
    ISTATE_PAGES,
    ITABLE_COLS,
    KV_RING_ROWS,
    KV_ROWS,
    MC,
    MCACHE_ROWS,
    MNORM,
    MPOOL,
    MPROJ,
    MQ,
    MSLOTS,
    MSTATE_ELEMS,
    MSTATE_PAGES,
    MTABLE_COLS,
    O_MM_ROWS,
    O_SCALE_ROWS,
    ORIGINAL_COLS,
    P8,
    P16,
    PACK_ROWS,
    PAIR_ROWS,
    PARTIAL,
    PHASE_ATTN_PLAN,
    PHASE_ATTN_VECTOR,
    PHASE_COEFF,
    PHASE_COMB,
    PHASE_GATES,
    PHASE_IDX_HADAMARD,
    PHASE_IDX_QUANT,
    PHASE_IDX_QUERY_HAD,
    PHASE_IDX_QUERY_PROJECT,
    PHASE_IDX_QUERY_QUANT,
    PHASE_IDX_QUERY_ROPE,
    PHASE_IDX_SCALE,
    PHASE_INPUT,
    PHASE_MAIN_OUTPUT,
    PHASE_META,
    PHASE_NORM,
    PHASE_O_QUANT,
    PHASE_O_QUANT_GROUP_BASE,
    PHASE_OA,
    PHASE_OA_GROUP_BASE,
    PHASE_OB,
    PHASE_ORIGINAL_CACHE,
    PHASE_OUT,
    PHASE_PARTIAL,
    PHASE_QA,
    PHASE_QB,
    PHASE_QR,
    PHASE_QUERY,
    PHASE_RAW,
    PHASE_ROPE,
    PHASE_SCORE_VECTOR,
    PHASE_TOPK,
    PHASE_WEIGHTS,
    PHASE_WEIGHTS_PROJECT,
    QPROJ_ROWS,
    QR_ROWS,
    SCORE_ROWS,
    VECTORS,
    WEIGHTS_PARTIAL_ROWS,
    WEIGHTS_ROWS,
    WORDS,
    B,
    T,
)
from .vector_prefix_stages import (
    indexer_compressor_vector_stage,
    main_compressor_vector_stage,
    original_kv_vector_stage,
)


@pl.jit.incore
def vector_schedule(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    gamma_cq: pl.Tensor[[1024], pl.BF16],
    x_fp32: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
    partials: pl.Tensor[[PARTIAL, 32], pl.FP32],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[P8, 4096], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    qa: pl.Tensor[[P16, 1024], pl.FP32],
    qr: pl.InOut[pl.Tensor[[T, 1024], pl.INT8]],
    qr_scale: pl.InOut[pl.Tensor[[T, 1], pl.FP32]],
    qr_matmul: pl.InOut[pl.Tensor[[QR_ROWS, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_sin: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    rope_swap: pl.InOut[pl.Tensor[[T, 64], pl.INT32]],
    q_proj: pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32],
    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    kv_fp32: pl.Tensor[[KV_ROWS, 512], pl.FP32],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    main_value: pl.Tensor[[MPROJ, 1024], pl.FP32],
    main_score: pl.Tensor[[MPROJ, 1024], pl.FP32],
    main_state: pl.InOut[pl.Tensor[[MSTATE_PAGES, MSTATE_ELEMS], pl.FP32]],
    main_state_table: pl.Tensor[[B, MTABLE_COLS], pl.INT32],
    main_ape: pl.Tensor[[4, 1024], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    main_bounds: pl.Tensor[[MQ], pl.INT32],
    main_lengths: pl.Tensor[[B], pl.INT32],
    main_state_slots: pl.Tensor[[T, 2], pl.INT32],
    main_pooled: pl.InOut[pl.Tensor[[MPOOL, 512], pl.FP32]],
    main_gamma: pl.Tensor[[512], pl.BF16],
    main_cos: pl.Tensor[[MC, 64], pl.FP32],
    main_sin: pl.Tensor[[MC, 64], pl.FP32],
    main_offsets: pl.InOut[pl.Tensor[[B], pl.INT32]],
    main_count: pl.InOut[pl.Tensor[[1], pl.INT32]],
    main_slots: pl.Tensor[[MSLOTS, 2], pl.INT32],
    main_normed: pl.InOut[pl.Tensor[[MNORM, 512], pl.FP32]],
    main_debug: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    idx_value: pl.Tensor[[IPROJ, 256], pl.FP32],
    idx_score: pl.Tensor[[IPROJ, 256], pl.FP32],
    idx_state: pl.InOut[pl.Tensor[[ISTATE_PAGES, ISTATE_ELEMS], pl.FP32]],
    idx_state_table: pl.Tensor[[B, ITABLE_COLS], pl.INT32],
    idx_ape: pl.Tensor[[4, 256], pl.FP32],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_state_slots: pl.Tensor[[T, 2], pl.INT32],
    idx_pooled: pl.InOut[pl.Tensor[[IPOOL, 128], pl.FP32]],
    idx_gamma: pl.Tensor[[128], pl.BF16],
    idx_cos: pl.Tensor[[IC, 64], pl.FP32],
    idx_sin: pl.Tensor[[IC, 64], pl.FP32],
    idx_offsets: pl.InOut[pl.Tensor[[B], pl.INT32]],
    idx_count: pl.InOut[pl.Tensor[[1], pl.INT32]],
    idx_slots: pl.Tensor[[ISLOTS, 2], pl.INT32],
    idx_normed: pl.InOut[pl.Tensor[[ICPAD, 128], pl.BF16]],
    idx_had: pl.Tensor[[ICPAD, 128], pl.FP32],
    idx_scales: pl.InOut[pl.Tensor[[ISCALES, 1], pl.FP32]],
    idx_debug: pl.InOut[pl.Tensor[[T, 128], pl.FP32]],
    idx_cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    original_table: pl.Tensor[[B, ORIGINAL_COLS], pl.INT32],
    compressed_table: pl.Tensor[[B, COMPRESSED_COLS], pl.INT32],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
    idx_query_acc: pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32],
    idx_query_rope: pl.InOut[pl.Tensor[[IDX_ROPE_ROWS, 8192], pl.BF16]],
    idx_query_had: pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32],
    idx_query_i8: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8]],
    idx_query_scale: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 1], pl.FP32]],
    idx_query_scale_heads: pl.Tensor[[IDX_SCALE_T_ROWS, 64], pl.FP32],
    weights_partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
    head_weights: pl.InOut[pl.Tensor[[WEIGHTS_ROWS, 64], pl.FP32]],
    group_offsets: pl.InOut[pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32]],
    group_plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16]],
    indexer_transfer: pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS, 8192], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS, 1024], pl.FP32]],
    topk_scores: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    topk_indices: pl.InOut[pl.Tensor[[T, 512], pl.INT32]],
    attention_indices: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 512], pl.INT32]],
    attention_bias: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 640], pl.FP32]],
    attention_mask: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 16], pl.INT32]],
    inverse_sin: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    attn_sink: pl.Tensor[[64, 1], pl.FP32],
    kv_ring: pl.InOut[pl.Tensor[[KV_RING_ROWS, 512], pl.BF16]],
    score_ring: pl.Tensor[[HEAD_RING_ROWS, 128], pl.FP32],
    probability_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.BF16]],
    pv_ring: pl.Tensor[[HEAD_RING_ROWS, 512], pl.FP32],
    mi_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    li_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    packed: pl.InOut[pl.Tensor[[PACK_ROWS, 4096], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
    o_r: pl.Tensor[[O_MM_ROWS, 8192], pl.FP32],
    o_quant: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.INT8]],
    o_scales: pl.InOut[pl.Tensor[[8, O_SCALE_ROWS], pl.FP32]],
    o_partials: pl.Tensor[[O_MM_ROWS, 32768], pl.INT32],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 16384], pl.BF16]],
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
    ATTENTION_QUERY_METADATA: pl.constexpr,
    HC_CONSUMER_REDUCE: pl.constexpr,
    QA_M16_PARALLEL: pl.constexpr = False,
    COOPERATIVE_PREFIX: pl.constexpr = False,
    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,
    LOCAL_ATTENTION_STATS: pl.constexpr = False,
    SCALE_PAGE_COALESCING: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
):
    pl.func_attr({"split_aiv": True})
    x_fp32_gm = x_fp32
    inv_gm = inv
    widen_tail_gm = widen_tail
    raw_gm = raw
    pre_gm = pre
    post_gm = post
    post_tail_gm = post_tail
    comb_gm = comb
    comb_tail_gm = comb_tail
    mixed_gm = mixed
    normed_gm = normed
    norm_tail_gm = norm_tail
    qr_gm = qr
    qr_scale_gm = qr_scale
    qr_matmul_gm = qr_matmul
    qr_scale_pad_gm = qr_scale_pad
    rope_signed_gm = rope_signed
    rope_swap_gm = rope_swap
    q_flat_gm = q_flat
    kv_gm = kv
    cache_rows_gm = cache_rows
    main_state_gm = main_state
    main_pooled_gm = main_pooled
    main_offsets_gm = main_offsets
    main_count_gm = main_count
    main_normed_gm = main_normed
    main_debug_gm = main_debug
    main_cache_gm = main_cache
    idx_state_gm = idx_state
    idx_pooled_gm = idx_pooled
    idx_offsets_gm = idx_offsets
    idx_count_gm = idx_count
    idx_normed_gm = idx_normed
    idx_scales_gm = idx_scales
    idx_debug_gm = idx_debug
    idx_cache_gm = idx_cache
    idx_query_rope_gm = idx_query_rope
    idx_query_i8_gm = idx_query_i8
    idx_query_scale_gm = idx_query_scale
    head_weights_gm = head_weights
    group_offsets_gm = group_offsets
    group_plan_gm = group_plan
    coefficients_gm = coefficients
    score_arena_gm = score_arena
    pair_arena_gm = pair_arena
    topk_scores_gm = topk_scores
    topk_indices_gm = topk_indices
    attention_indices_gm = attention_indices
    attention_bias_gm = attention_bias
    attention_mask_gm = attention_mask
    inverse_sin_gm = inverse_sin
    kv_ring_gm = kv_ring
    probability_ring_gm = probability_ring
    mi_ring_gm = mi_ring
    li_ring_gm = li_ring
    packed_gm = packed
    o_quant_gm = o_quant
    o_scales_gm = o_scales
    output_gm = output
    signal_gm = signal
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = GROUPS + worker
    signal, generation = begin_invocation(signal, core)
    pl.system.set_ffts(ffts)
    tokens = pl.tensor.dim(x, 0)
    x_fp32, inv = hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, VECTORS)
    signal = publish_phase(signal, PHASE_INPUT, core, CORES, generation)

    # 公共metadata只有Q/L/positions依赖，可与HC投影重叠。选择第一个没有HC M8条带的
    # AIV；若48核都有工作，则选47号，先完成其本轮全部widen并发布，再计算计划。
    # 所以既没有永久保留一核，也没有让AIC等metadata才能开始HC。
    planner = pl.min((tokens + 7) // 8, VECTORS - 1)
    if worker == planner:
        # 两套compact Q/L分开扫描；只共用真实T，不合并Native语义不同的Tensor。
        # helper的worker=0表示“由调用方选定的唯一写者执行”，不是固定物理0核。
        main_offsets, main_count = compact_offsets_and_count_stage(
            main_bounds,
            main_lengths,
            main_offsets,
            main_count,
            tokens,
            0,
        )
        idx_offsets, idx_count = compact_offsets_and_count_stage(
            idx_bounds,
            idx_lengths,
            idx_offsets,
            idx_count,
            tokens,
            0,
        )
        group_offsets, group_plan = indexer_group_plan_stage(
            idx_bounds,
            idx_lengths,
            positions,
            group_offsets,
            group_plan,
            tokens,
            0,
            QUERY_GROUP,
            BALANCED,
            ROOTS,
        )
        # scale的地址/页所有权也只依赖本轮metadata，可在量化值产生之前准备。
        # 输入slots在整次调用中不可变；计划与组描述符共用一块有明确分区的工作区。
        group_offsets = indexer_scale_page_plan_stage(
            idx_bounds,
            idx_lengths,
            positions,
            idx_offsets,
            idx_slots,
            idx_cache,
            group_offsets,
            pl.tensor.dim(idx_scales, 0),
            VECTORS,
        )
        if BALANCED and SCALE_PAGE_COALESCING:
            # 整个请求的连续页证明与数值无关，移到同一 META producer；
            # 每叶两条 AIV 复用结果。碎片/越界/失活请求记录0方向，保留原读取。
            group_offsets = indexer_scale_layout_plan_stage(
                idx_bounds,
                idx_lengths,
                indexer_table,
                group_offsets,
                tokens,
                pl.tensor.dim(idx_cache, 0),
                VECTORS,
            )
        # publish先排空业务写回并维护缓存，随后发单写者完成信号；不是atomic计数器。
        # 计划在本次调用保持只读，消费者直到首次使用才等这个真实producer。
        signal = publish_phase(signal, PHASE_META, core, CORES, generation)

    if HC_CONSUMER_REDUCE:
        # 门控/comb的M8块ob=worker+wave*48，投影M16块为ob//2。
        # 四个K片归AIC (4*(ob//2)+k)%24；后续wave增加96，owner不变。
        # 只等本核所有真实消费行的四个producer，随后直接归约所需列。
        # inv来自本AIV已经发布的INPUT；RAW缓冲不再读写，也没有RAW通知。
        if worker < (tokens + 7) // 8:
            signal = wait_row_owners(signal, PHASE_PARTIAL, CORES, generation, (worker // 2) * 4, 4, 1, GROUPS, 0, 1)
    else:
        # 投影 task=(M16 行块*4+K4096 split)，一维 task % 24 决定真实 AIC owner。
        if HC_ROW_DEPENDENCIES:
            # 本AIV归约M16行块rb=worker+wave*48，四个K片的AIC owner是
            # (4*rb+k)%24。48*4能被24整除，所有wave均对应相同四个producer。
            # 空归约者不读取partials，也不应为无关行等待；仍照合同发布RAW。
            if worker < (tokens + 15) // 16:
                signal = wait_row_owners(signal, PHASE_PARTIAL, CORES, generation, worker * 4, 4, 1, GROUPS, 0, 1)
        else:
            projection_tasks = ((tokens + 15) // 16) * 4
            signal = wait_row_owners(
                signal,
                PHASE_PARTIAL,
                CORES,
                generation,
                0,
                projection_tasks,
                1,
                GROUPS,
                0,
                1,
            )
        raw = hc_reduce_range(partials, raw, worker, VECTORS)
        # RAW消费者只覆盖真实T的M16 owner；较大P16的原数学/尾写仍保留。
        if worker < (tokens + 15) // 16:
            signal = publish_phase(signal, PHASE_RAW, core, CORES, generation)
        if HC_ROW_DEPENDENCIES:
            # gates/comb按M8块ob=worker+wave*48分工；对应RAW M16 owner
            # 为floor(ob/2)%48，随wave只在worker//2及worker//2+24间交替。
            # 一波只等前者，多波至多等两者。inv由同一个AIV的widen先写完，
            # PHASE_INPUT发布已保证本核写回，因此无需再等其它核的INPUT。
            hc_gate_tiles = (tokens + 7) // 8
            if worker < hc_gate_tiles:
                signal = wait_row_owners(signal, PHASE_RAW, CORES, generation, worker // 2, 1, 1, VECTORS, GROUPS, 1)
                if worker + VECTORS < hc_gate_tiles:
                    signal = wait_row_owners(
                        signal, PHASE_RAW, CORES, generation, worker // 2 + VECTORS // 2, 1, 1, VECTORS, GROUPS, 1
                    )
        else:
            signal = wait_row_owners(
                signal,
                PHASE_RAW,
                CORES,
                generation,
                0,
                tokens,
                16,
                VECTORS,
                GROUPS,
                1,
            )
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
    pre, post = hc_gates_range(
        inv,
        raw,
        hc_scale,
        hc_base,
        pre,
        post,
        post_tail,
        worker,
        VECTORS,
        partials,
        HC_CONSUMER_REDUCE,
    )
    # 只省略没有本阶段有效输出的M8角色通知；活跃角色发布与全部wait/math不变。
    if worker < (tokens + 7) // 8:
        signal = publish_phase(signal, PHASE_GATES, core, CORES, generation)
    # gates与mix_norm按相同M8条带归同一AIV；pre由本核写，且GATES发布
    # 已排空写回，无需等待所有其他gate owner。先混合/归一化并发布NORM，
    # AIC便可开始QA；后面的Sinkhorn只供最终HCpost，不是QA的依赖。
    mixed, normed = hc_mix_norm_range(x_fp32, pre, norm_w, mixed, normed, norm_tail, worker, VECTORS)
    # 只省略没有本阶段有效输出的M8角色通知；活跃角色发布与全部wait/math不变。
    if worker < (tokens + 7) // 8:
        signal = publish_phase(signal, PHASE_NORM, core, CORES, generation)
    comb = hc_comb_range(inv, raw, hc_scale, hc_base_2d, comb, comb_tail, worker, VECTORS, partials, HC_CONSUMER_REDUCE)
    # 只省略没有本阶段有效输出的M8角色通知；活跃角色发布与全部wait/math不变。
    if worker < (tokens + 7) // 8:
        signal = publish_phase(signal, PHASE_COMB, core, CORES, generation)

    # Q_A 每个 M 组有八个 N128 列块；不能只等每行的一个 owner。
    if QA_M16_PARALLEL:
        # 与AIC同一constexpr：global M16块×八列，跨波次仍只等实际物理owner。
        qa_tasks = pl.min(GROUPS, 8 * ((tokens + 15) // 16))
        signal = wait_row_owners(signal, PHASE_QA, CORES, generation, 0, qa_tasks, 1, GROUPS, 0, 1)
    elif tokens < 128:
        qa_tasks = 8 * pl.min(3, pl.max(1, tokens // 32))
        signal = wait_row_owners(signal, PHASE_QA, CORES, generation, 0, qa_tasks, 1, GROUPS, 0, 1)
    else:
        qa_tasks = 8 * pl.min(3, pl.max(1, tokens // 64))
        signal = wait_row_owners(signal, PHASE_QA, CORES, generation, 0, qa_tasks, 1, GROUPS, 0, 1)
    qr, qr_scale, qr_matmul, qr_scale_pad = q_r_stage(
        qa,
        gamma_cq,
        qr,
        qr_scale,
        qr_matmul,
        qr_scale_pad,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_QR, core, CORES, generation)
    # RoPE只依赖输入sin；放在QR发布后，与AIC的QB阶段并行，不等待QB。
    rope_signed, rope_swap = q_rope_prepare_stage(rope_sin, rope_signed, rope_swap, worker, VECTORS)
    inverse_sin = attention_inverse_rope_sign_stage(rope_sin, inverse_sin, worker, VECTORS)
    # 区间策略开启时，只读几何和页合法性与投影并行；关闭则在Top-K/plan
    # 原位置计算，不读未初始化的前缀。每个八行writer与原ROPE owner相同，
    # 后面的既有ROPE获取同时保护该前缀；不能移到发布之后而不补依赖。
    if EARLY_ATTENTION_GEOMETRY:
        attention_indices = attention_geometry_prepare_stage(
            positions,
            idx_bounds,
            idx_lengths,
            compressed_table,
            attention_indices,
            pl.tensor.dim(main_cache, 0) // 32,
            worker,
            VECTORS,
            PLAN_PHYSICAL_ROWS,
        )
    signal = publish_phase(signal, PHASE_ROPE, core, CORES, generation)
    # 主Query与Indexer query数学独立；先完成后者的RoPE/pool/norm，
    # 让AIC继续query Hadamard并在其主Q_B期间做量化/系数。QR/ROPE获取
    # 随第一个消费者前移，不能只移动计算却遗漏跨核数据可见性。
    signal = wait_row_owners(
        signal,
        PHASE_ROPE,
        CORES,
        generation,
        0,
        tokens,
        8,
        VECTORS,
        GROUPS,
        1,
    )
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
    idx_requests = pl.tensor.dim(idx_lengths, 0)
    main_requests = pl.tensor.dim(main_lengths, 0)
    if COOPERATIVE_PREFIX:
        idx_pooled, idx_state, idx_normed, signal = indexer_compressor_vector_stage(
            positions,
            idx_value,
            idx_score,
            idx_state,
            idx_state_table,
            idx_ape,
            idx_bounds,
            idx_lengths,
            idx_state_slots,
            idx_pooled,
            idx_gamma,
            idx_cos,
            idx_sin,
            idx_offsets,
            idx_normed,
            signal,
            tokens,
            worker,
            core,
            generation,
            planner,
            COOPERATIVE_PREFIX,
        )
    # 前段QR及RoPE已等待过；Indexer INT8投影按32个N256列块归24个AIC。
    signal = wait_row_owners(signal, PHASE_IDX_QUERY_PROJECT, CORES, generation, 0, 32, 1, GROUPS, 0, 1)
    idx_query_rope = indexer_query_dequant_rope_stage(
        qr_scale,
        idx_wq_b_scale,
        rope_cos,
        rope_signed,
        idx_query_acc,
        idx_query_rope,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_IDX_QUERY_ROPE, core, CORES, generation)
    weights_tasks = ((tokens + 15) // 16) * 4
    signal = wait_row_owners(signal, PHASE_WEIGHTS_PROJECT, CORES, generation, 0, weights_tasks, 1, GROUPS, 0, 1)
    head_weights = indexer_weights_reduce_stage(weights_partial, head_weights, tokens, worker, VECTORS)
    signal = publish_phase(signal, PHASE_WEIGHTS, core, CORES, generation)
    if COOPERATIVE_PREFIX:
        main_pooled, main_state, main_normed, main_debug, main_cache, signal = main_compressor_vector_stage(
            main_value,
            main_score,
            main_state,
            main_state_table,
            main_ape,
            positions,
            main_bounds,
            main_lengths,
            main_state_slots,
            main_pooled,
            main_gamma,
            main_cos,
            main_sin,
            main_offsets,
            main_slots,
            main_normed,
            main_debug,
            main_cache,
            signal,
            tokens,
            worker,
            core,
            generation,
            COOPERATIVE_PREFIX,
        )
    else:
        idx_pooled, idx_state, idx_normed, signal = indexer_compressor_vector_stage(
            positions,
            idx_value,
            idx_score,
            idx_state,
            idx_state_table,
            idx_ape,
            idx_bounds,
            idx_lengths,
            idx_state_slots,
            idx_pooled,
            idx_gamma,
            idx_cos,
            idx_sin,
            idx_offsets,
            idx_normed,
            signal,
            tokens,
            worker,
            core,
            generation,
            planner,
            COOPERATIVE_PREFIX,
        )
    if not COOPERATIVE_PREFIX:
        kv, cache_rows, signal = original_kv_vector_stage(
            rope_cos,
            rope_signed,
            rope_swap,
            gamma_ckv,
            kv_fp32,
            kv,
            native_slots,
            cache_rows,
            signal,
            tokens,
            worker,
            core,
            generation,
            COOPERATIVE_PREFIX,
        )
    signal = wait_row_owners(signal, PHASE_IDX_QUERY_HAD, CORES, generation, 0, tokens, 1, GROUPS, 0, 1)
    idx_query_i8, idx_query_scale = indexer_query_quant_stage(
        idx_query_had,
        idx_query_i8,
        idx_query_scale,
        tokens,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_IDX_QUERY_QUANT, core, CORES, generation)

    # 系数要读取其他核的head权重/scale及请求分组。公共计划此前已等待，
    # planner在HC早期已启动；等待只建立必要数据依赖，不在SPMD入口做全核集合。
    signal = wait_row_owners(signal, PHASE_WEIGHTS, CORES, generation, 0, tokens, 16, VECTORS, GROUPS, 1)
    signal = wait_row_owners(signal, PHASE_IDX_QUERY_QUANT, CORES, generation, 0, tokens, 1, VECTORS, GROUPS, 1)
    coefficients = indexer_head_coefficients_stage(
        idx_query_scale_heads,
        head_weights,
        positions,
        idx_bounds,
        idx_lengths,
        group_offsets,
        coefficients,
        worker,
        VECTORS,
        QUERY_GROUP,
    )
    signal = publish_phase(signal, PHASE_COEFF, core, CORES, generation)
    # 系数之后先提交Indexer compact key/scale。主Query和主压缩仅供后继
    # Attention消费，仍在本AIV进入Score之前完成；这里不先等待整段Score。
    # AIC可以用既有两槽提前生产，槽复用继续由原SCORE_READY/ACK保证。
    compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
    had_tasks = ((compact_count + 15) // 16) * 2
    signal = wait_row_owners(signal, PHASE_IDX_HADAMARD, CORES, generation, 0, had_tasks, 1, GROUPS, 0, 1)
    idx_scales, idx_debug, idx_cache = indexer_compact_quant_cache_stage(
        idx_had,
        positions,
        idx_bounds,
        idx_lengths,
        idx_offsets,
        idx_slots,
        idx_scales,
        idx_debug,
        idx_cache,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_IDX_QUANT, core, CORES, generation)
    # 量化key/中间scale先全部完成；随后物理page%48确定唯一scale页写者。
    # 同页多offset由同核顺序RMW，不同页并行。空writer也发布完成，消费者
    # 根据本阶段48个参与者等待；这不同于上一版只等AIV0的阶段合同。
    signal = wait_row_owners(signal, PHASE_IDX_QUANT, CORES, generation, 0, idx_requests, 1, VECTORS, GROUPS, 1)
    idx_cache = indexer_scale_page_commit_stage(
        idx_scales,
        group_offsets,
        idx_cache,
        tokens,
        idx_requests,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_IDX_SCALE, core, CORES, generation)

    # 每个head占两个NZ N256列块。首版等全部128列块的24个真实owner。
    signal = wait_row_owners(signal, PHASE_QB, CORES, generation, 0, 128, 1, GROUPS, 0, 1)
    q_flat = q_dequant_rope_stage(
        wq_b_scale,
        rope_cos,
        rope_signed,
        rope_swap,
        q_flat,
        qr_scale_pad,
        q_proj,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_QUERY, core, CORES, generation)

    if not COOPERATIVE_PREFIX:
        main_pooled, main_state, main_normed, main_debug, main_cache, signal = main_compressor_vector_stage(
            main_value,
            main_score,
            main_state,
            main_state_table,
            main_ape,
            positions,
            main_bounds,
            main_lengths,
            main_state_slots,
            main_pooled,
            main_gamma,
            main_cos,
            main_sin,
            main_offsets,
            main_slots,
            main_normed,
            main_debug,
            main_cache,
            signal,
            tokens,
            worker,
            core,
            generation,
            COOPERATIVE_PREFIX,
        )
    # Score两种角色必须同时可推进。不能先等PHASE_SCORE_CUBE“整段完成”，否则
    # Cube等待AIV归还双缓冲槽，而AIV等待Cube退出，形成环。这里只等输入就绪。
    signal = wait_row_owners(signal, PHASE_IDX_SCALE, CORES, generation, 0, VECTORS, 1, VECTORS, GROUPS, 1)
    score_arena, pair_arena = indexer_score_vector_lane_stage(
        idx_cache,
        indexer_table,
        positions,
        idx_lengths,
        idx_bounds,
        group_offsets,
        group_plan,
        indexer_transfer,
        score_arena,
        pair_arena,
        pl.tile.get_subblock_idx(),
        pl.tile.get_block_idx(),
        QUERY_GROUP,
        QK_PANEL,
        SCORE_TILE,
        PREFETCH,
        TO_L0,
        BALANCED,
        SCALE_PAGE_COALESCING,
        0,
        1,
    )
    signal = publish_phase(signal, PHASE_SCORE_VECTOR, core, CORES, generation)
    if COOPERATIVE_PREFIX:
        # Score已归还本组环槽后，处理原KV；无工作AIV继续Top-K，不额外集合。
        kv, cache_rows, signal = original_kv_vector_stage(
            rope_cos,
            rope_signed,
            rope_swap,
            gamma_ckv,
            kv_fp32,
            kv,
            native_slots,
            cache_rows,
            signal,
            tokens,
            worker,
            core,
            generation,
            COOPERATIVE_PREFIX,
        )
    # 每条query按本轮group/leaf映射获取真正的根写者；完整Score阶段信号
    # 保持不变，不能用槽ACK替代它。数学helper在实际读根之前执行获取。
    topk_scores, topk_indices, signal = indexer_pair_publish_stage(
        idx_bounds,
        idx_lengths,
        positions,
        group_plan,
        group_offsets,
        attention_indices,
        signal,
        pair_arena,
        topk_scores,
        topk_indices,
        worker,
        VECTORS,
        ROOTS,
        BALANCED,
        generation,
        QUERY_GROUP,
        EARLY_ATTENTION_GEOMETRY,
    )
    signal = publish_phase(signal, PHASE_TOPK, core, CORES, generation)
    attention_indices, attention_bias, attention_mask, signal = attention_slot_plan_stage(
        topk_indices,
        positions,
        idx_bounds,
        idx_lengths,
        original_table,
        compressed_table,
        attention_indices,
        attention_bias,
        attention_mask,
        signal,
        generation,
        pl.tensor.dim(cache_rows, 0) // 32,
        pl.tensor.dim(main_cache, 0) // 32,
        worker,
        VECTORS,
        ATTENTION_QUERY_METADATA,
        EARLY_ATTENTION_GEOMETRY,
        PLAN_PHYSICAL_ROWS,
    )
    signal = publish_phase(signal, PHASE_ATTN_PLAN, core, CORES, generation)

    # 每组Attention处理q=group+k*24，只等这些query对应的八行plan owner。
    # Native cache可包含共享页，继续保留原完整写者依赖，不擅自按request缩窄。
    signal = wait_attention_plan_rows(signal, tokens, pl.tile.get_block_idx(), generation)
    signal = wait_row_owners(signal, PHASE_ORIGINAL_CACHE, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1)
    signal = wait_row_owners(signal, PHASE_MAIN_OUTPUT, CORES, generation, 0, main_requests, 1, VECTORS, GROUPS, 1)
    signal = wait_row_owners(signal, PHASE_ROPE, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1)
    # 事件2..5用于Attention三槽流水，和Indexer的0/1分开。publish完成phase
    # 不能代替这些逐槽READY/ACK；每个helper返回前均需排空自己的末尾在途项。
    kv_ring, probability_ring, mi_ring, li_ring, packed = attention_qk_pv_vector_lane_stage(
        cache_rows,
        main_cache,
        original_table,
        compressed_table,
        idx_bounds,
        idx_lengths,
        positions,
        attention_indices,
        attention_bias,
        attention_mask,
        attn_sink,
        rope_cos,
        inverse_sin,
        kv_ring,
        score_ring,
        probability_ring,
        pv_ring,
        mi_ring,
        li_ring,
        packed,
        ffts,
        pl.tile.get_block_idx(),
        pl.tile.get_subblock_idx(),
        GROUPS,
        2,
        ATTENTION_QUERY_METADATA,
        LOCAL_ATTENTION_STATS,
        PLAN_PHYSICAL_ROWS,
    )
    signal = publish_phase(signal, PHASE_ATTN_VECTOR, core, CORES, generation)
    if OUTPUT_GROUP_PIPELINE:
        oa_per_group = ((tokens + 127) // 128) * (1024 // OA_COLS)
        quant_per_group = (tokens + 31) // 32
        # 仍按(group,32行块)全局编号%48分配AIV，不让每组都只用0号核。
        # 只对本核有工作量的组等待/量化/发布；空组无消费者，也不发送假就绪。
        for quant_group in pl.range(8):
            quant_first = quant_group * quant_per_group
            quant_owned = worker + ((quant_first + VECTORS - 1 - worker) // VECTORS) * VECTORS
            if quant_owned < quant_first + quant_per_group:
                # amax需要本行的全部1024列。首版按整组所有行的真实O_A owner
                # 获取数据，已比等待全八组更细；不声称做到逐行块最小依赖。
                signal = wait_row_owners(
                    signal,
                    PHASE_OA_GROUP_BASE + quant_group,
                    CORES,
                    generation,
                    quant_group * oa_per_group,
                    oa_per_group,
                    1,
                    GROUPS,
                    0,
                    1,
                )
                o_quant, o_scales = o_projection_groups_quant_stage(
                    o_r, o_quant, o_scales, tokens, worker, VECTORS, OB_ROWS, quant_group, quant_group + 1
                )
                # 量化helper包括该组最后活动32行块负责的INT8尾零，发布前统一
                # 完成业务流水及DCache维护。本组单独槽位不与其它组重复发布。
                signal = publish_phase(signal, PHASE_O_QUANT_GROUP_BASE + quant_group, core, CORES, generation)
    else:
        oa_tasks = 8 * ((tokens + 127) // 128) * (1024 // OA_COLS)
        signal = wait_row_owners(signal, PHASE_OA, CORES, generation, 0, oa_tasks, 1, GROUPS, 0, 1)
        o_quant, o_scales = o_projection_group_quant_stage(o_r, o_quant, o_scales, tokens, worker, VECTORS, OB_ROWS)
        signal = publish_phase(signal, PHASE_O_QUANT, core, CORES, generation)
    signal = wait_row_owners(signal, PHASE_OB, CORES, generation, 0, 64, 1, GROUPS, 0, 1)
    # 所有有有效comb输出的HC角色先发布COMB，再进入未改动的Score全48核完成集合；post/comb此时已
    # 完整，且未复用其工作区。原hidden也必须保持只读直到最后的HCpost结束。
    output = o_dequant_hc_post_stage(
        o_partials,
        o_scales,
        wo_b_scale,
        x,
        post,
        comb,
        output,
        worker,
        VECTORS,
    )
    signal = publish_phase(signal, PHASE_OUT, core, CORES, generation)
    return (
        x_fp32_gm,
        inv_gm,
        widen_tail_gm,
        raw_gm,
        pre_gm,
        post_gm,
        post_tail_gm,
        comb_gm,
        comb_tail_gm,
        mixed_gm,
        normed_gm,
        norm_tail_gm,
        qr_gm,
        qr_scale_gm,
        qr_matmul_gm,
        qr_scale_pad_gm,
        rope_signed_gm,
        rope_swap_gm,
        q_flat_gm,
        kv_gm,
        cache_rows_gm,
        main_state_gm,
        main_pooled_gm,
        main_offsets_gm,
        main_count_gm,
        main_normed_gm,
        main_debug_gm,
        main_cache_gm,
        idx_state_gm,
        idx_pooled_gm,
        idx_offsets_gm,
        idx_count_gm,
        idx_normed_gm,
        idx_scales_gm,
        idx_debug_gm,
        idx_cache_gm,
        idx_query_rope_gm,
        idx_query_i8_gm,
        idx_query_scale_gm,
        head_weights_gm,
        group_offsets_gm,
        group_plan_gm,
        coefficients_gm,
        score_arena_gm,
        pair_arena_gm,
        topk_scores_gm,
        topk_indices_gm,
        attention_indices_gm,
        attention_bias_gm,
        attention_mask_gm,
        inverse_sin_gm,
        kv_ring_gm,
        probability_ring_gm,
        mi_ring_gm,
        li_ring_gm,
        packed_gm,
        o_quant_gm,
        o_scales_gm,
        output_gm,
        signal_gm,
    )
