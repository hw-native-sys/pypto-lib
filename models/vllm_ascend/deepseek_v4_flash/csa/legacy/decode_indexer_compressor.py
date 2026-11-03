# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/decode_indexer_compressor.py."""

import pypto.language as pl

from .compact_metadata import ROPE_TILE_ROWS, load_compact_rope_rows
from ..common.config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FP32_NEG_INF,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    TP,
)
from ..common.config import (
    FLASH as M,
)
from ..common.layout import (
    INDEXER_KEY_BYTES,
    INDEXER_PAGE_BYTES_DYN,
    INDEXER_ROWS_DYN,
    INNER_STATE_PAGE_ELEMENTS_DYN,
    INNER_STATE_TABLE_COLUMNS_DYN,
)

B_DYN = pl.dynamic("DECODE_IDX_C4_B_DYN")

S_DYN = pl.dynamic("DECODE_IDX_C4_S_DYN")

T_DYN = pl.dynamic("DECODE_IDX_C4_T_DYN")  # T = B * S

B = DECODE_BATCH // TP

S = DECODE_SEQ

EPS = M.rms_norm_eps

D = M.hidden_size

HEAD_DIM = M.index_head_dim

HADAMARD_SCALE = HEAD_DIM**-0.5

HEAD_DIM_INV = 1.0 / HEAD_DIM

ROPE_HEAD_DIM = M.qk_rope_head_dim

NOPE_HEAD_DIM = M.index_nope_head_dim

MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 4

OVERLAP = COMPRESS_RATIO == 4

COFF = 1 + int(OVERLAP)

OUT_DIM = COFF * HEAD_DIM

STATE_LEN = COFF * COMPRESS_RATIO

COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

COMPRESS_STATE_DIM = 2 * OUT_DIM

IDX_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE

IDX_NATIVE_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_NATIVE_CACHE_BLOCK_NUM_DYN")

COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")

K_TILE = 512

OUT_TILE = 64

PROJ_OUT_TILE = 32

# Native A3 CompressorKernelPerf uses eight 16-column groups for the
# supported uniform S=6, head-dim128 decode shapes. Each group starts its
# K256 traversal at a different block; the two overlap halves share it.
NATIVE_PROJECTION_N_GROUP = HEAD_DIM // 8

assert PROJ_OUT_TILE % 16 == 0, "cube tile cols must be a multiple of 16"

MM_B_TILE = 16

KV_SCORE_WORKERS = 24  # KV-score projection workers

POOL_WORKERS = 48  # Pool workers

RMS_WORKERS = 2  # RMSNorm + RoPE workers

COMMIT_WORKERS = 48

GROUP_BS = DECODE_BATCH * DECODE_SEQ

BS_PAD = ((GROUP_BS + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE

HEAD_TILE = 64

RMS_PAD_TILE = ROPE_TILE_ROWS  # 16-row block of B (hadamard matmul M multiple of 16)
BOUNDARY_ROWS_PER_REQUEST = (S + COMPRESS_RATIO - 1) // COMPRESS_RATIO


@pl.jit.inline(auto_scope=False)
def indexer_compressor_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    kv_proj_pad: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    score_proj_pad: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
    chain_dep: pl.Scalar[pl.TASK_ID],
):
    """Project token-local compressor values and scores in FP32."""
    bs = pl.tensor.dim(x, 0)
    t_matmul = ((bs + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE
    x_flat = x

    # 各生产者允许消费者预派发，计算开始仍由原tensor和TaskId依赖保护。
    with pl.spmd(
        KV_SCORE_WORKERS,
        name_hint="kv_score_proj",
        allow_early_resolve=True,
        deps=[late_dep, chain_dep],
    ) as _kv_score_tid:
        kv_worker = pl.tile.get_block_idx()
        for idx in pl.range(kv_worker, t_matmul * OUT_DIM // (MM_B_TILE * PROJ_OUT_TILE), KV_SCORE_WORKERS):
            global_row0 = (idx // (OUT_DIM // PROJ_OUT_TILE)) * MM_B_TILE
            o0 = (idx % (OUT_DIM // PROJ_OUT_TILE)) * PROJ_OUT_TILE
            kv_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
            score_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(0, D // K_TILE, stage=2):
                # 性能版按上游取平凡的 K 序。原式用 NATIVE_PROJECTION_* 把 K 块按
                # 输出列组轮转，是为复刻 Native 的累加次序。实测该项收益在噪声内，
                # 保留是因为与上游一致、少一处精度锁。
                k0 = kb * K_TILE
                x_rows = pl.min(MM_B_TILE, bs - global_row0)
                x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
                # Transposed [OUT_DIM, D] projection weights.
                wkv_tile = wkv[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
                wgate_tile = wgate[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(kb == 0))
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(kb == 0))

            kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = kv_acc
            score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = score_acc

    return _kv_score_tid


@pl.jit.inline(auto_scope=False)
def indexer_compressor_pool_projected(
    kv_proj_pad: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    score_proj_pad: pl.Tensor[[BS_PAD, OUT_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32],
    state_table: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    compact_offsets: pl.Tensor[[B_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    pooled_kv: pl.Out[pl.Tensor[[BS_PAD, HEAD_DIM], pl.FP32]],
    normed_kv: pl.Out[pl.Tensor[[BS_PAD, HEAD_DIM], pl.BF16]],
    late_dep: pl.Scalar[pl.TASK_ID],
    chain_dep: pl.Scalar[pl.TASK_ID],
):
    """Pool projected rows, commit state, and normalize boundary KV rows."""
    b_dim = pl.tensor.dim(state_table, 0)
    bs = pl.tensor.dim(position_ids, 0)
    s_dim = bs // b_dim
    rms_blocks = (bs + RMS_PAD_TILE - 1) // RMS_PAD_TILE

    _kv_score_tid = late_dep

    # Ratio-4 pooling reads Native historical state and current projections.
    pool_workers = pl.min(b_dim, POOL_WORKERS)
    with pl.spmd(pool_workers, name_hint="scatter_softmax_pool",
                 allow_early_resolve=True, deps=[_kv_score_tid]) as pool_tid:
        pool_worker = pl.tile.get_block_idx()
        for c_idx in pl.range(pool_worker, b_dim, pool_workers):
            first_pos_b = pl.read(position_ids, [c_idx * s_dim])
            for s_idx in pl.range(s_dim):
                token = c_idx * s_dim + s_idx
                token_pos = pl.read(position_ids, [token])
                pooled_kv[token : token + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.FP32, value=0.0)
                if (token_pos + 1) % COMPRESS_RATIO == 0:
                    window_start = token_pos - STATE_LEN + 1
                    for h0 in pl.range(0, HEAD_DIM, HEAD_TILE):
                        # 在线 softmax（上游写法）：以本 token 自己这一格起步，再把其余
                        # STATE_LEN-1 格逐个并入。已封存的精度版先把 8 格按 Native 的交错次序写进
                        # GM 暂存、读回后先归一化概率再做 8->4->2->1 规约，只为复刻 Native 的舍入次序。
                        last_ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                        mi = pl.add(
                            score_proj_pad[token : token + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE],
                            ape[last_ape_row : last_ape_row + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE],
                        )
                        li = pl.exp(pl.sub(mi, mi))
                        oi = kv_proj_pad[token : token + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
                        for state_idx in pl.range(STATE_LEN - 1):
                            logical_pos = window_start + state_idx
                            value = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                            score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            state_half = 0
                            if state_idx >= COMPRESS_RATIO:
                                state_half = HEAD_DIM
                            if logical_pos >= 0 and logical_pos < first_pos_b:
                                # Match the former gather's zero row for a negative page.
                                score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                                history_page = pl.cast(
                                    pl.read(state_table, [c_idx, logical_pos // COMPRESS_STATE_BLOCK_SIZE]), pl.INDEX
                                )
                                if history_page >= 0:
                                    history_column = (
                                        pl.cast(logical_pos % COMPRESS_STATE_BLOCK_SIZE, pl.INDEX) * COMPRESS_STATE_DIM
                                        + state_half
                                        + h0
                                    )
                                    value = compress_state[
                                        history_page : history_page + 1,
                                        history_column : history_column + HEAD_TILE,
                                    ]
                                    score = compress_state[
                                        history_page : history_page + 1,
                                        history_column + OUT_DIM : history_column + OUT_DIM + HEAD_TILE,
                                    ]
                            if logical_pos >= first_pos_b:
                                if logical_pos <= token_pos:
                                    overlay_token = c_idx * s_dim + logical_pos - first_pos_b
                                    ape_row = pl.cast(logical_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                                    value = kv_proj_pad[
                                        overlay_token : overlay_token + 1,
                                        state_half + h0 : state_half + h0 + HEAD_TILE,
                                    ]
                                    score = pl.add(
                                        score_proj_pad[
                                            overlay_token : overlay_token + 1,
                                            state_half + h0 : state_half + h0 + HEAD_TILE,
                                        ],
                                        ape[
                                            ape_row : ape_row + 1,
                                            state_half + h0 : state_half + h0 + HEAD_TILE,
                                        ],
                                    )
                            mi_next = pl.maximum(mi, score)
                            alpha = pl.exp(pl.sub(mi, mi_next))
                            beta = pl.exp(pl.sub(score, mi_next))
                            li = pl.add(pl.mul(alpha, li), beta)
                            oi = pl.add(pl.mul(oi, alpha), pl.mul(value, beta))
                            mi = mi_next
                        pooled_kv[token : token + 1, h0 : h0 + HEAD_TILE] = pl.div(oi, li)

    # Direct Native state commit, after all history reads in this pool task.
    commit_workers = pl.min(b_dim, COMMIT_WORKERS)
    with pl.spmd(commit_workers, name_hint="compress_state_commit", deps=[pool_tid]):
        commit_worker = pl.tile.get_block_idx()
        for c_idx in pl.range(commit_worker, b_dim, commit_workers):
            for s_idx in pl.range(s_dim):
                token = c_idx * s_dim + s_idx
                state_page = pl.read(inner_state_slot_mapping, [token, 0])
                state_offset = pl.read(inner_state_slot_mapping, [token, 1])
                if state_page >= 0 and state_offset >= 0:
                    token_pos = pl.read(position_ids, [token])
                    native_page = pl.cast(state_page, pl.INDEX)
                    native_column = pl.cast(state_offset, pl.INDEX) * COMPRESS_STATE_DIM
                    ape_row = pl.cast(token_pos % COMPRESS_RATIO, target_type=pl.INDEX)
                    compress_state[
                        native_page : native_page + 1, native_column : native_column + OUT_DIM
                    ] = kv_proj_pad[
                        token : token + 1, 0:OUT_DIM
                    ]
                    compress_state[
                        native_page : native_page + 1,
                        native_column + OUT_DIM : native_column + COMPRESS_STATE_DIM,
                    ] = pl.add(
                        score_proj_pad[token : token + 1, 0:OUT_DIM],
                        ape[ape_row : ape_row + 1, 0:OUT_DIM],
                    )

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    # S=6 intervals can close one or two compression groups. Reserve the
    # maximum per request, and initialize unused rows before Hadamard reads.
    with pl.spmd(b_dim, name_hint="indexer_boundary_init",
                 allow_early_resolve=True, deps=[pool_tid]) as boundary_init_tid:
        init_request = pl.tile.get_block_idx()
        compact_begin = init_request * BOUNDARY_ROWS_PER_REQUEST
        normed_kv[compact_begin : compact_begin + BOUNDARY_ROWS_PER_REQUEST, :] = pl.full(
            [BOUNDARY_ROWS_PER_REQUEST, HEAD_DIM], dtype=pl.BF16, value=0.0
        )
    rms_workers = pl.min(rms_blocks, RMS_WORKERS)
    with pl.spmd(rms_workers, name_hint="rmsnorm_rope",
                 allow_early_resolve=True, deps=[pool_tid, boundary_init_tid]) as rms_tid:
        rms_worker = pl.tile.get_block_idx()
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_index = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        rope_index_f = pl.cast(rope_index, target_type=pl.FP32)
        rope_col = pl.col_expand_mul(rope_ones, rope_index_f)
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))  # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        for rms_blk in pl.range(rms_worker, rms_blocks, rms_workers):
            # Padded token block and interleaved inverse-RoPE rows.
            b0 = rms_blk * RMS_PAD_TILE
            rms_blk_rows = pl.min(RMS_PAD_TILE, bs - b0)
            cos_b, sin_b = load_compact_rope_rows(cos, sin, position_ids, compact_offsets, b0, rms_blk_rows)
            # 平方和先把两段 64 列折到一起再做一次 row_sum。
            kv_rms_low = pooled_kv[b0 : b0 + RMS_PAD_TILE, 0:HEAD_TILE]
            kv_rms_high = pooled_kv[b0 : b0 + RMS_PAD_TILE, HEAD_TILE:HEAD_DIM]
            folded_sq = pl.add(pl.mul(kv_rms_low, kv_rms_low), pl.mul(kv_rms_high, kv_rms_high))
            square_sum = pl.row_sum(folded_sq)
            variance = pl.add(pl.mul(square_sum, HEAD_DIM_INV), EPS)
            inv_rms = pl.recip(pl.sqrt(variance))
            kv_norm_chunk = pooled_kv[b0 : b0 + RMS_PAD_TILE, 0:NOPE_HEAD_DIM]
            gamma = pl.cast(norm_w_2d[:, 0:NOPE_HEAD_DIM], target_type=pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_nope = pl.cast(normed_chunk, target_type=pl.BF16, mode="rint")

            kv_rope_norm = pooled_kv[b0 : b0 + RMS_PAD_TILE, NOPE_HEAD_DIM:HEAD_DIM]
            gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], target_type=pl.FP32)
            # Interleaved RMSNorm and inverse-RoPE rotation.
            rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
            swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
            sin_signed = pl.mul(sin_b, pl.sub(pl.mul(rope_lane, 2.0), 1.0))
            rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_signed))
            normed_rope = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
            for inner in pl.range(rms_blk_rows):
                token = b0 + inner
                token_pos = pl.read(position_ids, [token])
                if (token_pos + 1) % COMPRESS_RATIO == 0:
                    request = token // S
                    first_pos = pl.read(position_ids, [request * S])
                    first_boundary = COMPRESS_RATIO - 1 - first_pos % COMPRESS_RATIO
                    compact_token = request * BOUNDARY_ROWS_PER_REQUEST + (token % S - first_boundary) // COMPRESS_RATIO
                    normed_kv[compact_token : compact_token + 1, 0:NOPE_HEAD_DIM] = normed_nope[inner : inner + 1, :]
                    normed_kv[compact_token : compact_token + 1, NOPE_HEAD_DIM:HEAD_DIM] = normed_rope[
                        inner : inner + 1, :
                    ]

    return _kv_score_tid, rms_tid


@pl.jit.inline(auto_scope=False)
def indexer_compressor_pool(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32],
    state_table: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    compact_offsets: pl.Tensor[[B_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    normed_kv: pl.Out[pl.Tensor[[BS_PAD, HEAD_DIM], pl.BF16]],
    late_dep: pl.Scalar[pl.TASK_ID],
    chain_dep: pl.Scalar[pl.TASK_ID],
):
    """Project, then pool and commit the token stream."""
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    pooled_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    projection_tid = indexer_compressor_project(
        x,
        wkv,
        wgate,
        kv_proj_pad,
        score_proj_pad,
        late_dep,
        chain_dep,
    )
    projection_ready_tid, rms_tid = indexer_compressor_pool_projected(
        kv_proj_pad,
        score_proj_pad,
        compress_state,
        state_table,
        ape,
        norm_w,
        cos,
        sin,
        compact_offsets,
        position_ids,
        inner_state_slot_mapping,
        pooled_kv,
        normed_kv,
        projection_tid,
        chain_dep,
    )
    return projection_ready_tid, rms_tid


@pl.jit.inline(auto_scope=False)
def indexer_compressor_write(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    normed_kv: pl.Tensor[[BS_PAD, HEAD_DIM], pl.BF16],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_native_kv_cache: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    idx_slot_mapping: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32],
    compact_offsets: pl.Tensor[[B_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    rms_tid: pl.Scalar[pl.TASK_ID],
    hadamard_dep: pl.Scalar[pl.TASK_ID],
):
    """Rotate compact boundary rows and write their quantized indexer KV cache."""
    bs = pl.tensor.dim(position_ids, 0)
    compact_rows = (bs // S) * BOUNDARY_ROWS_PER_REQUEST
    rms_blocks = (compact_rows + RMS_PAD_TILE - 1) // RMS_PAD_TILE
    kv_flat = kv
    idx_kv_scale_values = pl.create_tensor([BS_PAD, 1], dtype=pl.FP32)

    kv_final = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    # Caller-ordered KV Hadamard projection.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="kv_hadamard",
        allow_early_resolve=True,
        deps=[rms_tid, hadamard_dep],
    ) as hadamard_tid:
        # Hadamard column tiles.
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0:HEAD_DIM, o0 : o0 + OUT_TILE]
            for had_blk in pl.range(rms_blocks):
                had_b0 = had_blk * RMS_PAD_TILE
                had_rows = pl.min(RMS_PAD_TILE, compact_rows - had_b0)
                kv_proj_tile = pl.slice(
                    normed_kv, [RMS_PAD_TILE, HEAD_DIM], [had_b0, 0], valid_shape=[had_rows, HEAD_DIM]
                )
                kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
                kv_final[had_b0 : had_b0 + RMS_PAD_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc

    with pl.spmd(
        rms_blocks,
        name_hint="kv_and_cache_write",
        deps=[hadamard_tid],
        allow_early_resolve=True,
    ) as _write_tid:
        wr_blk = pl.tile.get_block_idx()
        # C8 per-row INT8 cache quantization.
        wr_b0 = wr_blk * RMS_PAD_TILE
        wr_blk_rows = pl.min(RMS_PAD_TILE, compact_rows - wr_b0)
        kv_blk_f32 = pl.cast(
            pl.cast(kv_final[wr_b0 : wr_b0 + RMS_PAD_TILE, 0:HEAD_DIM], target_type=pl.BF16, mode="rint"),
            target_type=pl.FP32,
        )
        # Native rotate_activation scales after its BF16 Hadamard matmul.
        kv_blk_f32 = pl.mul(kv_blk_f32, HADAMARD_SCALE)
        # Per-row absolute maximum.
        kv_amax = pl.reshape(pl.row_max(pl.abs(kv_blk_f32)), [1, RMS_PAD_TILE])
        kv_amax = pl.maximum(kv_amax, pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS))
        kv_scale_q_row = pl.div(pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax)
        kv_scale_dq_col = pl.reshape(pl.recip(kv_scale_q_row), [RMS_PAD_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [RMS_PAD_TILE, 1])
        idx_kv_scale_values[
            wr_b0 : wr_b0 + RMS_PAD_TILE,
            0:1,
        ] = kv_scale_dq_col
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode="rint")
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode="round")
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode="trunc")
        for inner in pl.range(wr_blk_rows):
            compact_token = wr_b0 + inner
            request = compact_token // BOUNDARY_ROWS_PER_REQUEST
            first_pos = pl.read(position_ids, [request * S])
            local_token = (
                (compact_token % BOUNDARY_ROWS_PER_REQUEST) * COMPRESS_RATIO
                + COMPRESS_RATIO
                - 1
                - first_pos % COMPRESS_RATIO
            )
            # compact metadata 按本步实际 token 数分配；补位请求的 position 是上一步残留，
            # 据此推出的行号会越界（实测 84 vs 8 行）。Native 把补位请求 seq_lens 清零，
            # 真实 decode 请求恒 >= S，故可无歧义排除。
            if pl.read(seq_lens, [request]) > 0 and local_token < S:
                token = request * S + local_token
                token_pos = pl.read(position_ids, [token])
                metadata_row = pl.cast(pl.read(compact_offsets, [request]), pl.INDEX) + pl.cast(
                    (token_pos + 1) // COMPRESS_RATIO, pl.INDEX
                )
                # compact 表只有 Native 算好的行数，超出即无效。图捕获的 dummy run
                # 把所有 position 填成 127，(127+1)%4==0 成立而 seq_lens 非零，
                # 推出的行号会远超该档行数，必须按张量真实行数兜住。
                idx_rows = pl.tensor.dim(idx_slot_mapping, 0)
                safe_row = pl.min(metadata_row, idx_rows - 1)
                native_page = pl.read(idx_slot_mapping, [safe_row, 0])
                native_offset = pl.read(idx_slot_mapping, [safe_row, 1])
                if metadata_row < idx_rows and native_page >= 0 and native_offset >= 0:
                    kv_flat[token : token + 1, :] = kv_blk_f32[inner : inner + 1, :]
                    # Update the canonical Native slot directly.
                    native_cache_page = pl.cast(native_page, pl.INDEX)
                    native_key_begin = pl.cast(native_offset, pl.INDEX) * HEAD_DIM
                    idx_native_kv_cache[
                        native_cache_page : native_cache_page + 1,
                        native_key_begin : native_key_begin + HEAD_DIM,
                    ] = kv_i8_blk[inner : inner + 1, :]

    # Serialized indexer-cache scale commit.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="idx_kv_scale_commit",
        deps=[_write_tid],
        allow_early_resolve=True,
    ) as scale_commit_tid:
        for compact_token in pl.range(compact_rows):
            request = compact_token // BOUNDARY_ROWS_PER_REQUEST
            first_pos = pl.read(position_ids, [request * S])
            local_token = (
                (compact_token % BOUNDARY_ROWS_PER_REQUEST) * COMPRESS_RATIO
                + COMPRESS_RATIO
                - 1
                - first_pos % COMPRESS_RATIO
            )
            # compact metadata 按本步实际 token 数分配；补位请求的 position 是上一步残留，
            # 据此推出的行号会越界（实测 84 vs 8 行）。Native 把补位请求 seq_lens 清零，
            # 真实 decode 请求恒 >= S，故可无歧义排除。
            if pl.read(seq_lens, [request]) > 0 and local_token < S:
                token = request * S + local_token
                token_pos = pl.read(position_ids, [token])
                metadata_row = pl.cast(pl.read(compact_offsets, [request]), pl.INDEX) + pl.cast(
                    (token_pos + 1) // COMPRESS_RATIO, pl.INDEX
                )
                # 同上：按 compact 表的真实行数兜住，dummy run 的 position=127
                # 会推出远超该档行数的行号。
                idx_rows = pl.tensor.dim(idx_slot_mapping, 0)
                safe_row = pl.min(metadata_row, idx_rows - 1)
                native_page = pl.read(idx_slot_mapping, [safe_row, 0])
                native_offset = pl.read(idx_slot_mapping, [safe_row, 1])
                if metadata_row < idx_rows and native_page >= 0 and native_offset >= 0:
                    scale_value = pl.cast(pl.read(idx_kv_scale_values, [compact_token, 0]), pl.FP16)
                    # Native packs 32 FP16 scales into one aligned 64-byte
                    # region. Serialize read-modify-write to preserve the
                    # neighboring history slots, as in the archived precision path.
                    native_scale_page = pl.cast(native_page, pl.INDEX)
                    native_scale_bytes = pl.tile.load(
                        idx_native_kv_cache, [native_scale_page, INDEXER_KEY_BYTES], [1, BLOCK_SIZE * 2]
                    )
                    native_scale_half = pl.tile.reinterpret_view(native_scale_bytes, pl.FP16)
                    pl.tile.write(native_scale_half, [0, native_offset], scale_value)
                    native_updated_bytes = pl.tile.reinterpret_view(native_scale_half, pl.INT8)
                    pl.tile.store(native_updated_bytes, [native_scale_page, INDEXER_KEY_BYTES], idx_native_kv_cache)

    return hadamard_tid, scale_commit_tid


@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32],
    state_table: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    compact_offsets: pl.Tensor[[B_DYN], pl.INT32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_native_kv_cache: pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    idx_slot_mapping: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32],
    inner_state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
    hadamard_dep: pl.Scalar[pl.TASK_ID],
):
    normed_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.BF16)
    _kv_score_tid, rms_tid = indexer_compressor_pool(
        x,
        compress_state,
        state_table,
        wkv,
        wgate,
        ape,
        norm_w,
        cos,
        sin,
        compact_offsets,
        position_ids,
        inner_state_slot_mapping,
        normed_kv,
        late_dep,
        late_dep,
    )
    _hadamard_tid, write_tid = indexer_compressor_write(
        kv,
        normed_kv,
        hadamard,
        idx_native_kv_cache,
        idx_slot_mapping,
        compact_offsets,
        position_ids,
        seq_lens,
        rms_tid,
        hadamard_dep,
    )
    return write_tid
