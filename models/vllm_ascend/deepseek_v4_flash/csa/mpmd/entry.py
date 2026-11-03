# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""原生 TND 多 task CSA：沿用 BSH MPMD 的独立计算链与 TMR 依赖。

Q/L、slots、页表和 compact metadata 保持 Native 原存储。核内数学来自
84751e34 的 TND SPMD；每个 pl.spmd/pl.cluster 是独立调度单元，完整 CSA
不会合成一个 SPMD。跨 task 用 Tensor/TaskId 依赖，局部 MIX 内保留 FFTS。
"""

import pypto.language as pl

from .attention_plan_stages import (
    attention_geometry_prepare_stage,
    attention_inverse_rope_sign_stage,
    attention_slot_plan_stage,
)
from .attention_stream_stages import attention_qk_pv_cube_stage, attention_qk_pv_vector_lane_stage
from .cache_stages import original_kv_writeback_stage
from .compressor_output_stages import compressor_norm_rope_cache_stage
from .compressor_pool_stages import (
    compressor_pool_stage,
    compressor_state_commit_stage,
    indexer_compressor_pool_stage,
    indexer_compressor_state_commit_stage,
)
from .compressor_projection_stages import (
    CMP_M_TILE,
    CMP_N_TILE,
    CMP_OUT_DIM,
    INDEXER_M_TILE,
    INDEXER_N_TILE,
    INDEXER_OUT_DIM,
    compressor_project_stage,
    indexer_compressor_project_stage,
)
from .hc_stages import (
    hc_comb_range,
    hc_gates_range,
    hc_mix_norm_range,
    hc_project_range,
    hc_reduce_range,
    hc_widen_rms_range,
)
from .indexer_cache_stages import indexer_compact_quant_cache_stage
from .indexer_compressor_output_stages import (
    HEAD_DIM as COMPACT_HEAD_DIM,
)
from .indexer_compressor_output_stages import (
    OUT_TILE as COMPACT_HAD_COL_TILE,
)
from .indexer_compressor_output_stages import (
    RMS_PAD_TILE as COMPACT_HAD_ROW_TILE,
)
from .indexer_compressor_output_stages import (
    indexer_compressor_hadamard_stage,
    indexer_compressor_norm_rope_stage,
)
from .indexer_group_stages import indexer_group_plan_stage, indexer_head_coefficients_stage
from .indexer_publish_stages import indexer_pair_publish_stage
from .indexer_query_stages import (
    DEQUANT_T_TILE,
    DQ_ROPE_H_TILE,
    IDX_N_HEADS,
    MM_ROW_TILE,
    WEIGHTS_OK,
    indexer_query_dequant_rope_stage,
    indexer_query_hadamard_stage,
    indexer_query_project_stage,
    indexer_query_quant_stage,
    indexer_weights_project_stage,
    indexer_weights_reduce_stage,
)
from .indexer_scale_layout_stages import indexer_scale_layout_plan_stage
from .indexer_scale_plan_stages import indexer_scale_page_commit_stage, indexer_scale_page_plan_stage
from .indexer_score_stages import indexer_score_cube_stage, indexer_score_vector_lane_stage
from .kv_stages import (
    HEAD_DIM as KV_HEAD_DIM,
)
from .kv_stages import (
    KV_N_TILE,
    KV_OM,
    kv_norm_rope_stage,
    kv_project_stage,
)
from .metadata_stages import compact_offsets_and_count_stage
from .output_projection_a_stages import (
    O_GROUPS,
    QUANT_TASK_T_TILE,
    o_projection_a_stage_nz,
    o_projection_group_quant_stage,
)
from .output_projection_b_stages import o_b_stage, o_dequant_hc_post_stage
from .qkv_stages import (
    BF16_WEIGHT_LAYOUT,
    QR_DENSE_M_TILE,
    QR_FIXED_M_GROUPS,
    QR_FIXED_SMALL_M_TILE,
    QR_FIXED_SMALL_ROWS,
    QR_M_TILE,
    QR_N_BLOCKS,
    q_a_stage,
    q_r_stage,
)
from .query_projection_stages import QUANT_WEIGHT_LAYOUT, q_b_stage, q_dequant_rope_stage, q_rope_prepare_stage
from .runtime_abi import (
    ATTN_PLAN_ROWS,
    CACHE_ROWS,
    COEFF_COLS,
    COEFF_ROWS,
    COMPRESSED_COLS,
    GROUP_OFFSET_ROWS,
    GROUPS,
    HEAD_RING_ROWS,
    IC,
    ICACHE_BYTES,
    ICACHE_PAGES,
    ICPAD,
    IDX_HEAD_ROWS,
    IDX_PROJ_ROWS,
    IDX_ROPE_HEAD_ROWS,
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
    KEY0_ROWS,
    KEY64_ROWS,
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
    QPROJ_ROWS,
    QR_ROWS,
    QUERY_HEAD_ROWS,
    SCORE_ROWS,
    VECTORS,
    WEIGHTS_PARTIAL_ROWS,
    WEIGHTS_ROWS,
    B,
    T,
)


@pl.jit.incore
def index_score_topk_cube(
    positions: pl.Tensor[[T], pl.INT64],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_query_i8: pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    group_plan: pl.Tensor[[8], pl.INT32],
    coefficients: pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16],
    idx_key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    idx_key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    indexer_transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    SPLIT_SCORE_ROWS: pl.constexpr,
):
    worker = pl.tile.get_block_idx()
    pl.system.set_ffts(ffts)
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
    return indexer_transfer


@pl.jit.incore
def index_score_topk_vector(
    positions: pl.Tensor[[T], pl.INT64],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    group_offsets: pl.InOut[pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32]],
    group_plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
    indexer_transfer: pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS, 8192], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS, 1024], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    SCALE_PAGE_COALESCING: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    pl.system.set_ffts(ffts)
    (score_arena, pair_arena) = indexer_score_vector_lane_stage(
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
    return (score_arena, pair_arena)


@pl.jit.inline(auto_scope=False)
def index_score_topk_group(
    positions: pl.Tensor[[T], pl.INT64],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_query_i8: pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    group_plan: pl.Tensor[[8], pl.INT32],
    coefficients: pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16],
    idx_key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    idx_key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    indexer_transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    SPLIT_SCORE_ROWS: pl.constexpr,
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS, 8192], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS, 1024], pl.FP32]],
    SCALE_PAGE_COALESCING: pl.constexpr,
):
    with pl.cluster(name_hint="index_score_topk"):
        indexer_transfer = index_score_topk_cube(
            positions,
            idx_bounds,
            idx_lengths,
            idx_cache,
            indexer_table,
            idx_query_i8,
            group_offsets,
            group_plan,
            coefficients,
            idx_key0,
            idx_key64,
            indexer_transfer,
            ffts,
            QUERY_GROUP,
            QK_PANEL,
            SCORE_TILE,
            PREFETCH,
            TO_L0,
            BALANCED,
            SPLIT_SCORE_ROWS,
        )
        (score_arena, pair_arena) = index_score_topk_vector(
            positions,
            idx_bounds,
            idx_lengths,
            idx_cache,
            indexer_table,
            group_offsets,
            group_plan,
            indexer_transfer,
            score_arena,
            pair_arena,
            ffts,
            QUERY_GROUP,
            QK_PANEL,
            SCORE_TILE,
            PREFETCH,
            TO_L0,
            BALANCED,
            SCALE_PAGE_COALESCING,
        )
    return (indexer_transfer, score_arena, pair_arena)


@pl.jit.incore
def sparse_attn_cube(
    q_heads: pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16],
    positions: pl.Tensor[[T], pl.INT64],
    attention_mask: pl.Tensor[[ATTN_PLAN_ROWS, 16], pl.INT32],
    kv_ring: pl.Tensor[[KV_RING_ROWS, 512], pl.BF16],
    score_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.FP32]],
    probability_ring: pl.Tensor[[HEAD_RING_ROWS, 128], pl.BF16],
    pv_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 512], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    worker = pl.tile.get_block_idx()
    pl.system.set_ffts(ffts)
    (score_ring, pv_ring) = attention_qk_pv_cube_stage(
        q_heads,
        attention_mask,
        kv_ring,
        score_ring,
        probability_ring,
        pv_ring,
        ffts,
        worker,
        pl.tensor.dim(positions, 0),
        GROUPS,
        2,
    )
    return (score_ring, pv_ring)


@pl.jit.incore
def sparse_attn_vector(
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    positions: pl.Tensor[[T], pl.INT64],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    original_table: pl.Tensor[[B, ORIGINAL_COLS], pl.INT32],
    compressed_table: pl.Tensor[[B, COMPRESSED_COLS], pl.INT32],
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
    ATTENTION_QUERY_METADATA: pl.constexpr,
    LOCAL_ATTENTION_STATS: pl.constexpr,
    PLAN_PHYSICAL_ROWS: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    pl.system.set_ffts(ffts)
    (kv_ring, probability_ring, mi_ring, li_ring, packed) = attention_qk_pv_vector_lane_stage(
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
    return (kv_ring, probability_ring, mi_ring, li_ring, packed)


@pl.jit.inline(auto_scope=False)
def sparse_attn_group(
    q_heads: pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16],
    attention_mask: pl.Tensor[[ATTN_PLAN_ROWS, 16], pl.INT32],
    kv_ring: pl.Tensor[[KV_RING_ROWS, 512], pl.BF16],
    score_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.FP32]],
    probability_ring: pl.Tensor[[HEAD_RING_ROWS, 128], pl.BF16],
    pv_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 512], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    positions: pl.Tensor[[T], pl.INT64],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    original_table: pl.Tensor[[B, ORIGINAL_COLS], pl.INT32],
    compressed_table: pl.Tensor[[B, COMPRESSED_COLS], pl.INT32],
    attention_indices: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 512], pl.INT32]],
    attention_bias: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 640], pl.FP32]],
    inverse_sin: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    attn_sink: pl.Tensor[[64, 1], pl.FP32],
    mi_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    li_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    packed: pl.InOut[pl.Tensor[[PACK_ROWS, 4096], pl.BF16]],
    ATTENTION_QUERY_METADATA: pl.constexpr,
    LOCAL_ATTENTION_STATS: pl.constexpr,
    PLAN_PHYSICAL_ROWS: pl.constexpr,
):
    with pl.cluster(name_hint="sparse_attn"):
        (score_ring, pv_ring) = sparse_attn_cube(
            q_heads, positions, attention_mask, kv_ring, score_ring, probability_ring, pv_ring, ffts
        )
        (kv_ring, probability_ring, mi_ring, li_ring, packed) = sparse_attn_vector(
            rope_cos,
            cache_rows,
            positions,
            main_cache,
            idx_bounds,
            idx_lengths,
            original_table,
            compressed_table,
            attention_indices,
            attention_bias,
            attention_mask,
            inverse_sin,
            attn_sink,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            mi_ring,
            li_ring,
            packed,
            ffts,
            ATTENTION_QUERY_METADATA,
            LOCAL_ATTENTION_STATS,
            PLAN_PHYSICAL_ROWS,
        )
    return (score_ring, pv_ring, kv_ring, probability_ring, mi_ring, li_ring, packed)


@pl.jit
def decode_csa_tnd(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_cq: pl.Tensor[[1024], pl.BF16],
    x_fp32: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
    partials: pl.InOut[pl.Tensor[[PARTIAL, 32], pl.FP32]],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[P8, 4096], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    qa: pl.InOut[pl.Tensor[[P16, 1024], pl.FP32]],
    qr: pl.InOut[pl.Tensor[[T, 1024], pl.INT8]],
    qr_scale: pl.InOut[pl.Tensor[[T, 1], pl.FP32]],
    qr_matmul: pl.InOut[pl.Tensor[[QR_ROWS, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_sin: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    rope_swap: pl.InOut[pl.Tensor[[T, 64], pl.INT32]],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
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
    idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
    idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
    idx_value: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    idx_score: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
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
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    idx_had: pl.InOut[pl.Tensor[[ICPAD, 128], pl.FP32]],
    idx_scales: pl.InOut[pl.Tensor[[ISCALES, 1], pl.FP32]],
    idx_debug: pl.InOut[pl.Tensor[[T, 128], pl.FP32]],
    idx_cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    q_heads: pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16],
    original_table: pl.Tensor[[B, ORIGINAL_COLS], pl.INT32],
    compressed_table: pl.Tensor[[B, COMPRESSED_COLS], pl.INT32],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
    idx_query_acc: pl.InOut[pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32]],
    idx_query_rope: pl.InOut[pl.Tensor[[IDX_ROPE_ROWS, 8192], pl.BF16]],
    idx_query_rope_heads: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    idx_query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    idx_query_i8: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8]],
    idx_query_scale: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 1], pl.FP32]],
    idx_query_scale_heads: pl.Tensor[[IDX_SCALE_T_ROWS, 64], pl.FP32],
    weights_proj: pl.Tensor[[4096, 64], pl.BF16],
    weights_partial: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
    head_weights: pl.InOut[pl.Tensor[[WEIGHTS_ROWS, 64], pl.FP32]],
    group_offsets: pl.InOut[pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32]],
    group_plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16]],
    indexer_transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],
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
    score_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.FP32]],
    probability_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.BF16]],
    pv_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 512], pl.FP32]],
    mi_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    li_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    packed: pl.InOut[pl.Tensor[[PACK_ROWS, 4096], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, BF16_WEIGHT_LAYOUT],
    o_r: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.FP32]],
    o_quant: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.INT8]],
    o_scales: pl.InOut[pl.Tensor[[8, O_SCALE_ROWS], pl.FP32]],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    o_partials: pl.InOut[pl.Tensor[[O_MM_ROWS, 32768], pl.INT32]],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 16384], pl.BF16]],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    ROOTS: pl.constexpr,
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
    ATTENTION_QUERY_METADATA: pl.constexpr,
    HC_CONSUMER_REDUCE: pl.constexpr,
    QA_M16_PARALLEL: pl.constexpr = False,
    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,
    LOCAL_ATTENTION_STATS: pl.constexpr = False,
    SCALE_PAGE_COALESCING: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
    SPLIT_SCORE_ROWS: pl.constexpr = False,
    LIMIT_EMPTY_WORKERS: pl.constexpr = False,
):
    tokens = pl.tensor.dim(x, 0)
    requests = pl.tensor.dim(idx_lengths, 0)
    main_requests = pl.tensor.dim(main_lengths, 0)
    q_heads = pl.reshape(q_flat, [tokens * 64, 512])
    idx_rope_tokens = pl.tensor.dim(idx_query_rope, 0)
    idx_query_rope_heads = pl.reshape(idx_query_rope, [idx_rope_tokens * 64, 128])
    idx_query_scale_heads = pl.reshape(idx_query_scale, [tokens, 64])
    pages = pl.tensor.dim(idx_cache, 0)
    page_bytes = pl.tensor.dim(idx_cache, 1)
    cache_flat = pl.reshape(idx_cache, [pages * page_bytes])
    key_prefix = cache_flat[0 : (pages * page_bytes // 128) * 128]
    key_shifted = cache_flat[64 : 64 + ((pages * page_bytes - 64) // 128) * 128]
    idx_key0 = pl.reshape(key_prefix, [pages * page_bytes // 128, 128])
    idx_key64 = pl.reshape(key_shifted, [(pages * page_bytes - 64) // 128, 128])

    # 小容量域删除确定无工作的 worker；大容量保留已测的原整组分工。
    # 只改变 launch，不改变 helper 内的 GROUPS/VECTORS 条带步长和数学。
    qa_workers = GROUPS
    idx_project_workers = GROUPS
    idx_hadamard_workers = GROUPS
    query_rope_workers = VECTORS
    query_hadamard_workers = GROUPS
    weight_workers = GROUPS
    main_project_workers = GROUPS
    kv_workers = GROUPS
    output_quant_workers = VECTORS
    post_workers = VECTORS
    if LIMIT_EMPTY_WORKERS:
        if QA_M16_PARALLEL:
            qa_units = QR_N_BLOCKS * ((tokens + QR_M_TILE - 1) // QR_M_TILE)
        elif tokens < QR_FIXED_SMALL_ROWS:
            qa_units = QR_N_BLOCKS * pl.min(QR_FIXED_M_GROUPS, pl.max(1, tokens // QR_FIXED_SMALL_M_TILE))
        else:
            qa_units = QR_N_BLOCKS * pl.min(QR_FIXED_M_GROUPS, pl.max(1, tokens // QR_DENSE_M_TILE))
        qa_workers = pl.min(GROUPS, qa_units)
        idx_project_workers = pl.min(
            GROUPS, (INDEXER_OUT_DIM // INDEXER_N_TILE) * ((tokens + INDEXER_M_TILE - 1) // INDEXER_M_TILE)
        )
        # compact 有效行数来自设备；这里只用已有容量作保守上界，不回读计数。
        compact_hadamard_units = (COMPACT_HEAD_DIM // COMPACT_HAD_COL_TILE) * (
            (pl.tensor.dim(idx_normed, 0) + COMPACT_HAD_ROW_TILE - 1) // COMPACT_HAD_ROW_TILE
        )
        idx_hadamard_workers = pl.min(GROUPS, compact_hadamard_units)
        query_rope_workers = pl.min(
            VECTORS, (IDX_N_HEADS // DQ_ROPE_H_TILE) * ((tokens + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE)
        )
        query_hadamard_workers = pl.min(GROUPS, tokens)
        weight_workers = pl.min(GROUPS, WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE))
        main_project_workers = pl.min(GROUPS, (CMP_OUT_DIM // CMP_N_TILE) * ((tokens + CMP_M_TILE - 1) // CMP_M_TILE))
        # 每个分段最多 KV_OM 个 M group，每组 HEAD_DIM/KV_N_TILE 个工作项。
        kv_workers = (KV_HEAD_DIM // KV_N_TILE) * KV_OM
        output_quant_workers = pl.min(VECTORS, O_GROUPS * ((tokens + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE))
        post_workers = pl.min(VECTORS, tokens)

    # HC 前段沿用 BSH MPMD 的 widen → split-K → reduce/gates → norm/comb 分工。
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_widen_rms", allow_early_resolve=True, sync_start=True
    ):
        (x_fp32, inv) = hc_widen_rms_range(x, x_fp32, inv, widen_tail, pl.tile.get_block_idx(), VECTORS)

    with pl.spmd(
        pl.min(((tokens + 15) // 16) * 4, GROUPS), name_hint="hc_linear", allow_early_resolve=True, sync_start=True
    ):
        partials = hc_project_range(x_fp32, hc_fn, partials, pl.tile.get_block_idx(), GROUPS)

    with pl.spmd(
        pl.min((tokens + 15) // 16, VECTORS), name_hint="hc_reduce", allow_early_resolve=True, sync_start=True
    ):
        raw = hc_reduce_range(partials, raw, pl.tile.get_block_idx(), VECTORS)

    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_pre_gates", allow_early_resolve=True, sync_start=True
    ):
        (pre, post) = hc_gates_range(
            inv,
            raw,
            hc_scale,
            hc_base,
            pre,
            post,
            post_tail,
            pl.tile.get_block_idx(),
            VECTORS,
            partials,
            HC_CONSUMER_REDUCE,
        )

    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_mix_norm", allow_early_resolve=True, sync_start=True
    ):
        (mixed, normed) = hc_mix_norm_range(
            x_fp32, pre, norm_w, mixed, normed, norm_tail, pl.tile.get_block_idx(), VECTORS
        )

    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_sinkhorn", allow_early_resolve=True, sync_start=True
    ):
        comb = hc_comb_range(
            inv,
            raw,
            hc_scale,
            hc_base_2d,
            comb,
            comb_tail,
            pl.tile.get_block_idx(),
            VECTORS,
            partials,
            HC_CONSUMER_REDUCE,
        )

    # 与现有 csa_row_offsets 相同：设备任务生成 Q/L 计划，可与投影链并发。
    # group/scale/compact 计划每次调用只生成一次；Host 只提供原 Tensor 描述符。
    with pl.spmd(1, name_hint="csa_row_offsets", allow_early_resolve=True, sync_start=True) as metadata_tid:
        metadata_worker = pl.tile.get_block_idx()
        (main_offsets, main_count) = compact_offsets_and_count_stage(
            main_bounds, main_lengths, main_offsets, main_count, tokens, metadata_worker
        )
        (idx_offsets, idx_count) = compact_offsets_and_count_stage(
            idx_bounds, idx_lengths, idx_offsets, idx_count, tokens, metadata_worker
        )
        (group_offsets, group_plan) = indexer_group_plan_stage(
            idx_bounds, idx_lengths, positions, group_offsets, group_plan, tokens, 0, QUERY_GROUP, BALANCED, ROOTS
        )
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
            group_offsets = indexer_scale_layout_plan_stage(
                idx_bounds, idx_lengths, indexer_table, group_offsets, tokens, pl.tensor.dim(idx_cache, 0), VECTORS
            )

    with pl.spmd(
        pl.min((tokens + 3) // 4, VECTORS), name_hint="csa_rope_sign", allow_early_resolve=True, sync_start=True
    ):
        (rope_signed, rope_swap) = q_rope_prepare_stage(
            rope_sin, rope_signed, rope_swap, pl.tile.get_block_idx(), VECTORS
        )

    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="csa_inverse_rope", allow_early_resolve=True, sync_start=True
    ):
        inverse_sin = attention_inverse_rope_sign_stage(rope_sin, inverse_sin, pl.tile.get_block_idx(), VECTORS)

    if EARLY_ATTENTION_GEOMETRY:
        with pl.spmd(
            pl.min((tokens + 7) // 8, VECTORS),
            name_hint="csa_query_geometry",
            allow_early_resolve=True,
            sync_start=True,
        ):
            attention_indices = attention_geometry_prepare_stage(
                positions,
                idx_bounds,
                idx_lengths,
                compressed_table,
                attention_indices,
                pl.tensor.dim(main_cache, 0) // 32,
                pl.tile.get_block_idx(),
                VECTORS,
                PLAN_PHYSICAL_ROWS,
            )

    with pl.spmd(qa_workers, name_hint="qproj_a", allow_early_resolve=True, sync_start=True) as qa_tid:
        qa = q_a_stage(normed, wq_a, qa, pl.tile.get_block_idx(), GROUPS, QA_M16_PARALLEL)

    with pl.spmd(
        idx_project_workers, name_hint="indexer_compressor_kv_score", allow_early_resolve=True, sync_start=True
    ):
        (idx_value, idx_score) = indexer_compressor_project_stage(
            normed, idx_wkv, idx_wgate, idx_value, idx_score, pl.tile.get_block_idx(), GROUPS
        )

    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="qproj_a_rms_quant", allow_early_resolve=True, sync_start=True
    ):
        (qr, qr_scale, qr_matmul, qr_scale_pad) = q_r_stage(
            qa, gamma_cq, qr, qr_scale, qr_matmul, qr_scale_pad, pl.tile.get_block_idx(), VECTORS
        )

    with pl.spmd(
        pl.min(requests, VECTORS), name_hint="indexer_compressor_pool", allow_early_resolve=True, sync_start=True
    ) as idx_pool_tid:
        idx_pooled = indexer_compressor_pool_stage(
            idx_value,
            idx_score,
            idx_state,
            idx_state_table,
            idx_ape,
            positions,
            idx_bounds,
            idx_lengths,
            idx_pooled,
            pl.tile.get_block_idx(),
            VECTORS,
        )

    with pl.spmd(
        pl.min(requests, VECTORS),
        name_hint="indexer_state_commit",
        allow_early_resolve=True,
        sync_start=True,
        deps=[idx_pool_tid],
    ):
        idx_state = indexer_compressor_state_commit_stage(
            idx_value,
            idx_score,
            idx_ape,
            positions,
            idx_bounds,
            idx_lengths,
            idx_state_slots,
            idx_state,
            pl.tile.get_block_idx(),
            VECTORS,
        )

    with pl.spmd(
        pl.min(requests, VECTORS),
        name_hint="indexer_compressor_norm_rope",
        allow_early_resolve=True,
        sync_start=True,
        deps=[metadata_tid],
    ):
        idx_normed = indexer_compressor_norm_rope_stage(
            idx_pooled,
            idx_gamma,
            idx_cos,
            idx_sin,
            positions,
            idx_bounds,
            idx_lengths,
            idx_offsets,
            idx_normed,
            pl.tile.get_block_idx(),
            VECTORS,
        )

    with pl.spmd(
        idx_hadamard_workers, name_hint="indexer_compressor_hadamard", allow_early_resolve=True, sync_start=True
    ):
        compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
        idx_had = indexer_compressor_hadamard_stage(
            idx_normed, hadamard, idx_had, compact_count, pl.tile.get_block_idx(), GROUPS
        )

    with pl.spmd(GROUPS, name_hint="indexer_query_proj", allow_early_resolve=True, sync_start=True):
        idx_query_acc = indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, pl.tile.get_block_idx(), GROUPS)

    with pl.spmd(query_rope_workers, name_hint="indexer_query_rope", allow_early_resolve=True, sync_start=True):
        idx_query_rope = indexer_query_dequant_rope_stage(
            qr_scale,
            idx_wq_b_scale,
            rope_cos,
            rope_signed,
            idx_query_acc,
            idx_query_rope,
            pl.tile.get_block_idx(),
            VECTORS,
        )

    with pl.spmd(query_hadamard_workers, name_hint="indexer_query_hadamard", allow_early_resolve=True, sync_start=True):
        idx_query_had = indexer_query_hadamard_stage(
            idx_query_rope_heads, hadamard, idx_query_had, tokens, pl.tile.get_block_idx(), GROUPS
        )

    with pl.spmd(pl.min(tokens, VECTORS), name_hint="indexer_query_quant", allow_early_resolve=True, sync_start=True):
        (idx_query_i8, idx_query_scale) = indexer_query_quant_stage(
            idx_query_had, idx_query_i8, idx_query_scale, tokens, pl.tile.get_block_idx(), VECTORS
        )

    with pl.spmd(weight_workers, name_hint="indexer_weights_proj", allow_early_resolve=True, sync_start=True):
        weights_partial = indexer_weights_project_stage(
            normed, weights_proj, weights_partial, pl.tile.get_block_idx(), GROUPS
        )

    with pl.spmd(
        pl.min((tokens + 15) // 16, VECTORS),
        name_hint="indexer_weights_reduce",
        allow_early_resolve=True,
        sync_start=True,
    ):
        head_weights = indexer_weights_reduce_stage(
            weights_partial, head_weights, tokens, pl.tile.get_block_idx(), VECTORS
        )

    with pl.spmd(
        VECTORS, name_hint="indexer_head_coefficients", allow_early_resolve=True, sync_start=True, deps=[metadata_tid]
    ):
        coefficients = indexer_head_coefficients_stage(
            idx_query_scale_heads,
            head_weights,
            positions,
            idx_bounds,
            idx_lengths,
            group_offsets,
            coefficients,
            pl.tile.get_block_idx(),
            VECTORS,
            QUERY_GROUP,
        )

    with pl.spmd(
        pl.min(requests, VECTORS), name_hint="indexer_compressor_cache_write", allow_early_resolve=True, sync_start=True
    ):
        (idx_scales, idx_debug, idx_cache) = indexer_compact_quant_cache_stage(
            idx_had,
            positions,
            idx_bounds,
            idx_lengths,
            idx_offsets,
            idx_slots,
            idx_scales,
            idx_debug,
            idx_cache,
            pl.tile.get_block_idx(),
            VECTORS,
        )

    with pl.spmd(
        VECTORS, name_hint="indexer_scale_page_commit", allow_early_resolve=True, sync_start=True
    ) as idx_cache_tid:
        idx_cache = indexer_scale_page_commit_stage(
            idx_scales, group_offsets, idx_cache, tokens, requests, pl.tile.get_block_idx(), VECTORS
        )

    with pl.spmd(
        main_project_workers, name_hint="compressor_kv_score", allow_early_resolve=True, sync_start=True, deps=[qa_tid]
    ):
        (main_value, main_score) = compressor_project_stage(
            normed, main_wkv, main_wgate, main_value, main_score, pl.tile.get_block_idx(), GROUPS
        )

    with pl.spmd(
        pl.min(main_requests, VECTORS), name_hint="compressor_pool", allow_early_resolve=True, sync_start=True
    ) as main_pool_tid:
        main_pooled = compressor_pool_stage(
            main_value,
            main_score,
            main_state,
            main_state_table,
            main_ape,
            positions,
            main_bounds,
            main_lengths,
            main_pooled,
            pl.tile.get_block_idx(),
            VECTORS,
        )

    with pl.spmd(
        pl.min(main_requests, VECTORS),
        name_hint="compressor_state_commit",
        allow_early_resolve=True,
        sync_start=True,
        deps=[main_pool_tid],
    ):
        main_state = compressor_state_commit_stage(
            main_value,
            main_score,
            main_ape,
            positions,
            main_bounds,
            main_lengths,
            main_state_slots,
            main_state,
            pl.tile.get_block_idx(),
            VECTORS,
        )

    with pl.spmd(
        pl.min(main_requests, VECTORS),
        name_hint="compressor_norm_rope_cache",
        allow_early_resolve=True,
        sync_start=True,
        deps=[metadata_tid],
    ):
        (main_normed, main_debug, main_cache) = compressor_norm_rope_cache_stage(
            main_pooled,
            main_gamma,
            main_cos,
            main_sin,
            main_bounds,
            main_lengths,
            positions,
            main_offsets,
            main_slots,
            main_normed,
            main_debug,
            main_cache,
            pl.tile.get_block_idx(),
            VECTORS,
        )

    with pl.spmd(GROUPS, name_hint="qproj_b", allow_early_resolve=True, sync_start=True):
        q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, pl.tile.get_block_idx(), GROUPS)

    with pl.spmd(VECTORS, name_hint="qproj_dequant_rms_nope_rope", allow_early_resolve=True, sync_start=True):
        q_flat = q_dequant_rope_stage(
            wq_b_scale, rope_cos, rope_signed, rope_swap, q_flat, qr_scale_pad, q_proj, pl.tile.get_block_idx(), VECTORS
        )

    with pl.spmd(kv_workers, name_hint="kv_proj", allow_early_resolve=True, sync_start=True):
        kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, pl.tile.get_block_idx(), GROUPS)

    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="kv_norm_rope", allow_early_resolve=True, sync_start=True
    ):
        kv = kv_norm_rope_stage(
            kv_fp32, gamma_ckv, rope_cos, rope_signed, rope_swap, kv, pl.tile.get_block_idx(), VECTORS
        )

    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="csa_cache_writeback", allow_early_resolve=True, sync_start=True
    ):
        cache_rows = original_kv_writeback_stage(kv, native_slots, cache_rows, pl.tile.get_block_idx(), VECTORS)

    # 仅 Score 与 Attention 保留各自的 MIX Cube/双 Vector 流水。
    # 二者、Top-K、plan 与前后投影是独立 task，依赖交给 TMR。
    with pl.spmd(
        GROUPS, name_hint="index_score_topk", allow_early_resolve=True, sync_start=True, deps=[idx_cache_tid]
    ) as score_tid:
        indexer_transfer, score_arena, pair_arena = index_score_topk_group(
            positions,
            idx_bounds,
            idx_lengths,
            idx_cache,
            indexer_table,
            idx_query_i8,
            group_offsets,
            group_plan,
            coefficients,
            idx_key0,
            idx_key64,
            indexer_transfer,
            ffts,
            QUERY_GROUP,
            QK_PANEL,
            SCORE_TILE,
            PREFETCH,
            TO_L0,
            BALANCED,
            SPLIT_SCORE_ROWS,
            score_arena,
            pair_arena,
            SCALE_PAGE_COALESCING,
        )

    with pl.spmd(
        pl.min(tokens, VECTORS),
        name_hint="indexer_topk_query",
        allow_early_resolve=True,
        sync_start=True,
        deps=[score_tid],
    ) as topk_tid:
        (topk_scores, topk_indices) = indexer_pair_publish_stage(
            idx_bounds,
            idx_lengths,
            positions,
            group_plan,
            group_offsets,
            attention_indices,
            pair_arena,
            topk_scores,
            topk_indices,
            pl.tile.get_block_idx(),
            VECTORS,
            ROOTS,
            BALANCED,
            QUERY_GROUP,
            EARLY_ATTENTION_GEOMETRY,
        )

    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS),
        name_hint="csa_attention_plan",
        allow_early_resolve=True,
        sync_start=True,
        deps=[topk_tid],
    ):
        (attention_indices, attention_bias, attention_mask) = attention_slot_plan_stage(
            topk_indices,
            positions,
            idx_bounds,
            idx_lengths,
            original_table,
            compressed_table,
            attention_indices,
            attention_bias,
            attention_mask,
            pl.tensor.dim(cache_rows, 0) // 32,
            pl.tensor.dim(main_cache, 0) // 32,
            pl.tile.get_block_idx(),
            VECTORS,
            ATTENTION_QUERY_METADATA,
            EARLY_ATTENTION_GEOMETRY,
            PLAN_PHYSICAL_ROWS,
        )

    with pl.spmd(GROUPS, name_hint="sparse_attn_csa", allow_early_resolve=True, sync_start=True) as attn_tid:
        score_ring, pv_ring, kv_ring, probability_ring, mi_ring, li_ring, packed = sparse_attn_group(
            q_heads,
            attention_mask,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            ffts,
            rope_cos,
            cache_rows,
            positions,
            main_cache,
            idx_bounds,
            idx_lengths,
            original_table,
            compressed_table,
            attention_indices,
            attention_bias,
            inverse_sin,
            attn_sink,
            mi_ring,
            li_ring,
            packed,
            ATTENTION_QUERY_METADATA,
            LOCAL_ATTENTION_STATS,
            PLAN_PHYSICAL_ROWS,
        )

    # 输出保持 BSH 的 O_A → 分组量化 → O_B → HCpost 多 task 链。
    # 首版保留相同数学与 NZ 原存储，后续按完整 L1 决定组流水/分档。
    with pl.spmd(GROUPS, name_hint="oproj_a", allow_early_resolve=True, sync_start=True, deps=[attn_tid]):
        o_r = o_projection_a_stage_nz(packed, wo_a, o_r, tokens, pl.tile.get_block_idx(), GROUPS, OA_COLS)

    with pl.spmd(output_quant_workers, name_hint="oproj_quant", allow_early_resolve=True, sync_start=True):
        (o_quant, o_scales) = o_projection_group_quant_stage(
            o_r, o_quant, o_scales, tokens, pl.tile.get_block_idx(), VECTORS, OB_ROWS
        )

    with pl.spmd(GROUPS, name_hint="oproj_b", allow_early_resolve=True, sync_start=True):
        o_partials = o_b_stage(o_quant, wo_b, o_partials, tokens, pl.tile.get_block_idx(), GROUPS, OB_ROWS)

    with pl.spmd(post_workers, name_hint="hc_post", allow_early_resolve=True, sync_start=True):
        output = o_dequant_hc_post_stage(
            o_partials, o_scales, wo_b_scale, x, post, comb, output, pl.tile.get_block_idx(), VECTORS
        )

    return (
        x_fp32,
        inv,
        widen_tail,
        partials,
        raw,
        pre,
        post,
        post_tail,
        comb,
        comb_tail,
        mixed,
        normed,
        norm_tail,
        qa,
        qr,
        qr_scale,
        qr_matmul,
        qr_scale_pad,
        rope_signed,
        rope_swap,
        q_proj,
        q_flat,
        kv_fp32,
        kv,
        cache_rows,
        main_value,
        main_score,
        main_state,
        main_pooled,
        main_offsets,
        main_count,
        main_normed,
        main_debug,
        main_cache,
        idx_value,
        idx_score,
        idx_state,
        idx_pooled,
        idx_offsets,
        idx_count,
        idx_normed,
        idx_had,
        idx_scales,
        idx_debug,
        idx_cache,
        idx_query_acc,
        idx_query_rope,
        idx_query_had,
        idx_query_i8,
        idx_query_scale,
        weights_partial,
        head_weights,
        group_offsets,
        group_plan,
        coefficients,
        indexer_transfer,
        score_arena,
        pair_arena,
        topk_scores,
        topk_indices,
        attention_indices,
        attention_bias,
        attention_mask,
        inverse_sin,
        kv_ring,
        score_ring,
        probability_ring,
        pv_ring,
        mi_ring,
        li_ring,
        packed,
        o_r,
        o_quant,
        o_scales,
        o_partials,
        output,
    )
