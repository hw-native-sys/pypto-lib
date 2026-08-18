# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer (decode). Mirrors model.py Indexer (line 380-433);
golden is a port of forward's decode branch (prefill `start_pos == 0` path is omitted).
The inner Compressor is invoked via golden_compressor (placeholder)."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_LOCAL_REQUESTS,
    DECODE_SEQ,
    BLOCK_SIZE,
    CSA_CANDIDATES_PER_LEAF,
    CSA_MAX_CANDIDATES,
    CSA_MAX_NODES_PER_QUERY,
    CSA_MAX_QUERIES,
    CSA_PAIR_WIDTH,
    CSA_TOPK,
    CSA_TOPK_INVALID_TASK_SLOT,
    FP32_NEG_INF,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from decode_indexer_topk import (
    IDX_ROW_DYN,
    LEAF_DYN,
    PAGE_DYN,
    PAIR_GROUP_DYN,
    REQUEST_DYN as B_DYN,
    REQUEST_OFFSET_DYN,
    SINGLETON_DYN,
    UPPER_MERGE_DYN,
    active_score_topk_forest,
)
from decode_metadata import (
    PHASE_D_LEAF_BEGIN,
    PHASE_D_LEAF_FIELDS,
    PHASE_D_LEAF_QUERY,
    PHASE_D_LEAF_VALID,
    PHASE_D_PAIR_FIELDS,
    PHASE_D_ROOT_FIELDS,
    PHASE_D_SINGLETON_FIELDS,
    PHASE_D_UPPER_FIELDS,
)
from rope_interleave import _rope_interleave_active_body

# Dynamic shape variables. S stays static: the score/topk scopes divide by it.
# Keep the indexer's local chunk extent separate from the enclosing CSA
# attention's active-token dynamic.  ``indexer`` is invoked once per 16-row
# chunk from ``decode_csa``; sharing the symbol made inline expansion bind the
# enclosing T to 16 and left later HC-pre rows unwritten.
T_DYN = pl.dynamic("IDX_T_DYN")  # local indexer query count
# Padded query-extent for the InOut query tensors.  The forest's fixed
# 16-row pl.slice views require the query output tensors to be allocated at a
# CSA_INDEXER_CHUNK_T multiple.  Input tensors (x, qr, qr_scale, cos_il,
# sin_signed) stay on T_DYN: the projection SPMD keys on ``query_count``
# from x.dim(0).  Padded rows are excluded by the descriptor counts and are
# never published by the commit guard.  See ``build_tensor_specs`` for the
# spec-layer pad.
T_PAD_DYN = pl.dynamic("IDX_T_PAD_DYN")
# Forest descriptors are chunk-major because the exact forest has a fixed
# 16-query task-array bound. The ragged descriptor count is bound at axis 1.
# The pair arena is a compile-time [4080, 1024] tensor created inside each
# chunk's pl.scope(); it never appears in the indexer ABI.

# model config
B = DECODE_LOCAL_REQUESTS
S = DECODE_SEQ
T = B * S
D = M.hidden_size
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_NOPE_HEAD_DIM = M.index_nope_head_dim
WEIGHTS_SCALE = M.index_weights_scale

# kernel-local
COMPRESS_RATIO = 4   # the indexer only runs on ratio-4 layers

# Indexer forest micro-chunking.  The exact Top-512 forest has a fixed 16-query
# task-array bound (CSA_MAX_QUERIES); the pair arena is a compile-time 4080 rows.
CSA_INDEXER_CHUNK_T = CSA_MAX_QUERIES
CSA_INDEXER_MAX_CHUNKS = T // CSA_INDEXER_CHUNK_T
CSA_INDEXER_ARENA_ROWS = CSA_INDEXER_CHUNK_T * CSA_MAX_NODES_PER_QUERY
assert CSA_INDEXER_ARENA_ROWS == 4080, "packed forest arena is fixed at 4080 rows"

# tiling
Q_TILE = 256
# Q_OUT_TILE is the per-task N granularity (sets idx_qr_proj task count); MM_N_TILE
# is the Mat-safe cube N-tile. Q_OUT_TILE fans Q_OUT_TILE // MM_N_TILE cube ops per
# task so task count halves without growing the [Q_TILE, MM_N_TILE] L1 wq load.
Q_OUT_TILE = 1024
T_PAD = ((T + 16 - 1) // 16) * 16  # static upper bound on the token axis
# Matmul M at the 16-row cube floor: a tile taller than the dynamic source is not expressible.
MM_ROW_TILE = 16
# INT32 Acc is MM_ROW_TILE * MM_N_TILE * 4B and must stay under the 128KiB L0C wall.
MM_N_TILE = min(512, (128 * 1024) // (MM_ROW_TILE * 4))
QR_OT_COUNT = IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE  # qr_proj N-tasks per row block
assert Q_OUT_TILE % MM_N_TILE == 0
# Dequant token tile: a whole-T [T, Q_OUT_TILE] FP32 tile does not fit UB.
DEQUANT_T_TILE = min(T, 8)
assert T % DEQUANT_T_TILE == 0
D_TILE = 512
# weights_proj splits K, not N: a [D_TILE, IDX_N_HEADS] row block reads contiguous GM,
# while an N slice would take 32B out of every 128B row. Each task writes its own
# partial row block, summed by a separate reduce scope. Partials are laid out
# [K slice][T_PAD rows] so the reduce adds whole T_PAD-row blocks.
# WEIGHTS_K_SLICE // D_TILE == 2, so the inner loop is a pl.range: a degenerate
# 2-iteration pl.pipeline(stage=2) miscompiles over matmul.
WEIGHTS_OK = 4
WEIGHTS_K_SLICE = D // WEIGHTS_OK
assert WEIGHTS_K_SLICE % D_TILE == 0
QH_HEAD_DIM_TILE = 64
# qr_rope SPMD tile == row block: one ROPE_ROW_TILE-row block per SPMD tile.
ROPE_ROW_TILE = 32
@pl.jit.inline(auto_scope=False)
def indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos_il: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin_signed: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    query_vectors: pl.InOut[pl.Tensor[
        [T_PAD_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8
    ]],
    query_scales: pl.InOut[pl.Tensor[[T_PAD_DYN, IDX_N_HEADS], pl.FP32]],
    query_weights: pl.InOut[pl.Tensor[[T_PAD_DYN, IDX_N_HEADS], pl.FP32]],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[T_PAD_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32
    ],
    pair_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, PAIR_GROUP_DYN, PHASE_D_PAIR_FIELDS], pl.INT32
    ],
    singleton_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS],
        pl.INT32,
    ],
    upper_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, UPPER_MERGE_DYN, PHASE_D_UPPER_FIELDS],
        pl.INT32,
    ],
    root_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, CSA_INDEXER_CHUNK_T, PHASE_D_ROOT_FIELDS],
        pl.INT32,
    ],
    # ``topk_scores`` was a second ``pl.Out`` paired with ``topk_indices`` across
    # the chunk loop.  PyPTO's implicit loop-carried SSA versioning for two
    # same-shaped ``pl.Out`` tensors of different dtype lowered into a
    # destructive swap that aliased both handles to the scores buffer from
    # iteration 2 on (see §0.3c).  The scores are never consumed downstream
    # (sparse attention reads only ``topk_indices``), so the global scores
    # output is dropped; the per-chunk ``chunk_topk_scores`` scratch inside
    # the forest is unchanged.  Workaround until the PyPTO phi bug is fixed.
    topk_indices: pl.Out[pl.Tensor[[T_DYN, CSA_TOPK], pl.INT32]],
    pair_group_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    singleton_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    upper_merge_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    index_commit_dep: pl.Scalar[pl.TASK_ID],
    late_dep: pl.Scalar[pl.TASK_ID],
    descriptor_chunk_offset: pl.Scalar[pl.INDEX],
    rope_ready_dep: pl.Scalar[pl.TASK_ID],
    completion: pl.Array[1, pl.TASK_ID],
):
    """Project token-local index queries and run the exact active CSA forest.

    Query projection (qr_proj, RoPE, Hadamard, quant, weights) runs once over
    the full active token set.  The exact Top-512 forest is then micro-chunked
    into 16-query batches, each with its own ``pl.scope()`` and a fresh fixed
    ``[4080, 1024]`` pair arena. The global descriptors
    (query IDs, pages, page offsets, windows, epochs, RoPE) are full-active and
    reused per chunk; only the packed forest descriptors and per-chunk actual
    counts are chunk-major.  The chunk's Top-512 indices are published to the
    full-active ``topk_indices`` output before the scope closes.  Per-chunk
    scores remain internal to the forest (``chunk_topk_scores``) and are not
    published as a second ``pl.Out`` (see the param comment above).
    """
    query_count = pl.tensor.dim(x, 0)
    query_heads = query_count * IDX_N_HEADS
    row_blocks = (query_count + MM_ROW_TILE - 1) // MM_ROW_TILE

    qr_acc_pad = pl.create_tensor(
        [T_PAD, IDX_N_HEADS * IDX_HEAD_DIM],
        dtype=pl.INT32,
    )
    for qr_unit in pl.spmd(
        QR_OT_COUNT * row_blocks,
        name_hint="phase_d_idx_qr_proj_matmul",
    ):
        qr_rb = qr_unit // QR_OT_COUNT
        ot = qr_unit - qr_rb * QR_OT_COUNT
        qr_r0 = qr_rb * MM_ROW_TILE
        qr_rows = pl.min(MM_ROW_TILE, query_count - qr_r0)
        o_base = ot * Q_OUT_TILE
        for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
            qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                q0 = kb * Q_TILE
                qr_tile = pl.slice(
                    qr,
                    [MM_ROW_TILE, Q_TILE],
                    [qr_r0, q0],
                    valid_shape=[qr_rows, Q_TILE],
                )
                wq_tile = wq_b[
                    q0 : q0 + Q_TILE,
                    o_base + ns : o_base + ns + MM_N_TILE,
                ]
                if q0 == 0:
                    qr_acc = pl.matmul(qr_tile, wq_tile, out_dtype=pl.INT32)
                else:
                    qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile)
            qr_acc_pad[
                qr_r0 : qr_r0 + MM_ROW_TILE,
                o_base + ns : o_base + ns + MM_N_TILE,
            ] = qr_acc

    qr_proj = pl.create_tensor(
        [query_count, IDX_N_HEADS * IDX_HEAD_DIM],
        dtype=pl.FP32,
    )
    with pl.spmd(
        (query_count // DEQUANT_T_TILE) * QR_OT_COUNT,
        name_hint="phase_d_idx_qr_proj_dequant",
    ) as qr_proj_tid:
        unit = pl.tile.get_block_idx()
        query_block = unit // QR_OT_COUNT
        ot = unit - query_block * QR_OT_COUNT
        query = query_block * DEQUANT_T_TILE
        o_base = ot * Q_OUT_TILE
        acc_fp32 = pl.cast(
            qr_acc_pad[
                query : query + DEQUANT_T_TILE,
                o_base : o_base + Q_OUT_TILE,
            ],
            target_type=pl.FP32,
            mode="none",
        )
        wq_scale = pl.reshape(
            wq_b_scale[o_base : o_base + Q_OUT_TILE],
            [1, Q_OUT_TILE],
        )
        qr_dequant = pl.col_expand_mul(
            pl.row_expand_mul(
                acc_fp32,
                qr_scale[query : query + DEQUANT_T_TILE, :],
            ),
            wq_scale,
        )
        qr_proj[
            query : query + DEQUANT_T_TILE,
            o_base : o_base + Q_OUT_TILE,
        ] = qr_dequant

    qr_proj_flat = pl.reshape(qr_proj, [query_heads, IDX_HEAD_DIM])
    qr_bf16 = pl.create_tensor([query_heads, IDX_HEAD_DIM], dtype=pl.BF16)
    rope_swap_idx_t = pl.create_tensor(
        [ROPE_ROW_TILE, ROPE_HEAD_DIM],
        dtype=pl.INT32,
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="phase_d_idx_rope_swap_idx",
    ):
        sw_col = pl.col_expand_mul(
            pl.full(
                [ROPE_ROW_TILE, ROPE_HEAD_DIM],
                dtype=pl.FP32,
                value=1.0,
            ),
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        sw_dup_f = pl.cast(
            pl.cast(
                pl.mul(sw_col, 0.5),
                target_type=pl.INT32,
                mode="trunc",
            ),
            target_type=pl.FP32,
        )
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx_t[:, :] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)),
            target_type=pl.INT32,
        )

    with pl.spmd(
        query_heads // ROPE_ROW_TILE,
        name_hint="phase_d_idx_query_rope",
        deps=[qr_proj_tid, rope_ready_dep],
    ) as _query_rope_tid:
        rope_unit = pl.tile.get_block_idx()
        row0 = rope_unit * ROPE_ROW_TILE
        query = row0 // IDX_N_HEADS
        qr_nope = qr_proj_flat[
            row0 : row0 + ROPE_ROW_TILE,
            0:IDX_NOPE_HEAD_DIM,
        ]
        qr_rope = qr_proj_flat[
            row0 : row0 + ROPE_ROW_TILE,
            IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM,
        ]
        qr_swapped = pl.gather(
            qr_rope,
            dim=-1,
            index=rope_swap_idx_t,
        )
        rope_rot = pl.add(
            pl.col_expand_mul(
                qr_rope,
                cos_il[query : query + 1, :],
            ),
            pl.col_expand_mul(
                qr_swapped,
                sin_signed[query : query + 1, :],
            ),
        )
        qr_bf16[row0 : row0 + ROPE_ROW_TILE, :] = pl.concat(
            pl.cast(qr_nope, target_type=pl.BF16, mode="rint"),
            pl.cast(rope_rot, target_type=pl.BF16, mode="rint"),
        )

    qh_acc_gm = pl.create_tensor([query_heads, IDX_HEAD_DIM], dtype=pl.FP32)
    with pl.spmd(
        query_count,
        name_hint="phase_d_idx_hadamard_matmul",
        deps=[_query_rope_tid],
    ) as qh_tid:
        query = pl.tile.get_block_idx()
        row0 = query * IDX_N_HEADS
        qh_acc_gm[row0 : row0 + IDX_N_HEADS, :] = pl.matmul(
            qr_bf16[row0 : row0 + IDX_N_HEADS, :],
            hadamard,
            out_dtype=pl.FP32,
        )

    with pl.spmd(
        query_count,
        name_hint="phase_d_idx_query_quant",
        deps=[qh_tid],
    ) as _query_quant_tid:
        query = pl.tile.get_block_idx()
        row0 = query * IDX_N_HEADS
        qh_amax = pl.full(
            [1, IDX_N_HEADS],
            dtype=pl.FP32,
            value=INT8_AMAX_EPS,
        )
        for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_a_f32 = qh_acc_gm[
                row0 : row0 + IDX_N_HEADS,
                h0 : h0 + QH_HEAD_DIM_TILE,
            ]
            qh_a_abs = pl.maximum(qh_a_f32, pl.neg(qh_a_f32))
            qh_amax = pl.maximum(
                qh_amax,
                pl.reshape(pl.row_max(qh_a_abs), [1, IDX_N_HEADS]),
            )
        qh_scale_quant_row = pl.div(
            pl.full(
                [1, IDX_N_HEADS],
                dtype=pl.FP32,
                value=INT8_SCALE_MAX,
            ),
            qh_amax,
        )
        query_scales[query : query + 1, :] = pl.reshape(
            pl.recip(qh_scale_quant_row),
            [1, IDX_N_HEADS],
        )
        qh_scale_quant = pl.reshape(
            qh_scale_quant_row,
            [IDX_N_HEADS, 1],
        )
        for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
            qh_q_f32 = qh_acc_gm[
                row0 : row0 + IDX_N_HEADS,
                h1 : h1 + QH_HEAD_DIM_TILE,
            ]
            qh_q_i32 = pl.cast(
                pl.row_expand_mul(qh_q_f32, qh_scale_quant),
                target_type=pl.INT32,
                mode="rint",
            )
            query_vectors[
                query : query + 1,
                :,
                h1 : h1 + QH_HEAD_DIM_TILE,
            ] = pl.reshape(
                pl.cast(
                    pl.cast(qh_q_i32, target_type=pl.FP16, mode="round"),
                    target_type=pl.INT8,
                    mode="trunc",
                ),
                [1, IDX_N_HEADS, QH_HEAD_DIM_TILE],
            )

    weights_partial = pl.create_tensor(
        [WEIGHTS_OK * T_PAD, IDX_N_HEADS],
        dtype=pl.FP32,
    )
    with pl.spmd(
        WEIGHTS_OK * row_blocks,
        name_hint="phase_d_idx_weights_proj",
        deps=[late_dep],
    ) as _weights_proj_tid:
        unit = pl.tile.get_block_idx()
        row_block = unit // WEIGHTS_OK
        k_block = unit - row_block * WEIGHTS_OK
        row0 = row_block * MM_ROW_TILE
        valid_rows = pl.min(MM_ROW_TILE, query_count - row0)
        k_base = k_block * WEIGHTS_K_SLICE
        weights_acc = pl.create_tensor(
            [MM_ROW_TILE, IDX_N_HEADS],
            dtype=pl.FP32,
        )
        for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
            d0 = k_base + db * D_TILE
            x_tile = pl.slice(
                x,
                [MM_ROW_TILE, D_TILE],
                [row0, d0],
                valid_shape=[valid_rows, D_TILE],
            )
            weight_tile = weights_proj[d0 : d0 + D_TILE, :]
            if db == 0:
                weights_acc = pl.matmul(
                    x_tile,
                    weight_tile,
                    out_dtype=pl.FP32,
                )
            else:
                weights_acc = pl.matmul_acc(
                    weights_acc,
                    x_tile,
                    weight_tile,
                )
        weights_partial[
            k_block * T_PAD + row0 : k_block * T_PAD + row0 + MM_ROW_TILE,
            :,
        ] = weights_acc

    with pl.spmd(
        query_count,
        name_hint="phase_d_idx_weights_reduce",
        deps=[_weights_proj_tid],
    ) as _weights_reduce_tid:
        query = pl.tile.get_block_idx()
        weights_sum = weights_partial[query : query + 1, :]
        for k_block in pl.unroll(1, WEIGHTS_OK):
            weights_sum = pl.add(
                weights_sum,
                weights_partial[
                    k_block * T_PAD + query : k_block * T_PAD + query + 1,
                    :,
                ],
            )
        query_weights[query : query + 1, :] = pl.mul(
            weights_sum,
            WEIGHTS_SCALE,
        )

    # Explicit projection-done barrier: the query projection SPMDs above use
    # ``allow_early_resolve=True``, which can break AUTO tensor tracking and
    # let the forest's chunk-inputs SPMD read stale/zero query_vectors/
    # query_scales/query_weights before the projection completes.  Joining
    # the last writers (query_quant for vectors/scales, weights_reduce for
    # weights) ensures the chunk-inputs copy sees fully projected data.
    projection_done = pl.system.task_dummy(
        deps=[_query_quant_tid, _weights_reduce_tid]
    )

    # Exact Top-512 forest: micro-chunked into 16-query batches, each with its
    # own pl.scope() and a fresh fixed [4080, 1024] pair arena that dies at
    # scope exit.  Query projection above ran once over full active T; only the
    # forest is micro-chunked. The global descriptors
    # (query IDs, pages, page offsets, windows, epochs) are full-active and
    # reused per chunk; only the packed forest descriptors + per-chunk actual
    # counts are chunk-major, sliced on the chunk axis.
    chunk_count = (query_count + CSA_INDEXER_CHUNK_T - 1) // CSA_INDEXER_CHUNK_T
    leaf_rows = pl.tensor.dim(leaf_descriptors, 1)
    pair_rows = pl.tensor.dim(pair_descriptors, 1)
    singleton_rows = pl.tensor.dim(singleton_descriptors, 1)
    upper_rows = pl.tensor.dim(upper_descriptors, 1)
    # Fan-in buffer for each chunk's commit TaskId.  Sized at the compile-time
    # chunk ceiling so it can be indexed by the dynamic loop variable.
    # Without this explicit fan-in, the post-loop topk-ready barrier relied on
    # AUTO tensor tracking + allow_early_resolve=True, which let the sparse
    # consumer read partially-committed topk_indices (root cause of the 16K
    # non-deterministic x_out divergence, see §0.3d).  Each chunk's commit
    # TaskId is recorded here; the post-loop join waits on every active chunk.
    commit_tids = pl.array.create(CSA_INDEXER_MAX_CHUNKS, pl.TASK_ID)
    # Fan-in buffer for each chunk's commit TaskId.  Sized at the compile-time
    # chunk ceiling so it can be indexed by the dynamic loop variable.
    # Without this explicit fan-in, the post-loop topk-ready barrier relied on
    # AUTO tensor tracking + allow_early_resolve=True, which let the sparse
    # consumer read partially-committed topk_indices (root cause of the 16K
    # non-deterministic x_out divergence, see §0.3d).  Each chunk's commit
    # TaskId is recorded here; the post-loop join waits on every active chunk.
    for chunk, (commit_tids_iter,) in pl.range(
        chunk_count, init_values=(commit_tids,)
    ):
        # The forest owns its inner manual scope.  Keep this chunk boundary an
        # ordinary scope: the updated frontend rejects nested manual scopes,
        # while an automatic scope may contain the forest's manual region.
        with pl.scope():
            chunk_pair_arena = pl.create_tensor(
                [CSA_INDEXER_ARENA_ROWS, CSA_PAIR_WIDTH], dtype=pl.FP32
            )
            chunk_t0 = chunk * CSA_INDEXER_CHUNK_T
            descriptor_chunk = descriptor_chunk_offset + chunk
            # Pass views of the full-T projection directly to the forest.
            # PyPTO's inlined task lowering can bind a local 3-D INT8 view to
            # the following FP32/INT32 argument (or omit it) in the generated
            # host ABI.  External projection views retain the original tensor
            # handle and therefore preserve the kernel argument order.  The
            # descriptor counts ensure padded rows in the final chunk are
            # never submitted.
            forest_query_vectors = pl.slice(
                query_vectors,
                [CSA_INDEXER_CHUNK_T, IDX_N_HEADS, IDX_HEAD_DIM],
                [chunk_t0, 0, 0],
            )
            forest_query_scales = pl.slice(
                query_scales,
                [CSA_INDEXER_CHUNK_T, IDX_N_HEADS],
                [chunk_t0, 0],
            )
            forest_query_weights = pl.slice(
                query_weights,
                [CSA_INDEXER_CHUNK_T, IDX_N_HEADS],
                [chunk_t0, 0],
            )
            # Request IDs are immutable metadata just like the projected query
            # rows.  Use the global view directly; copying them through the
            # chunk-input SPMD is unnecessary and can expose a stale row when
            # the forest block reads a later request in the same chunk.
            forest_query_request_ids = pl.slice(
                query_request_ids,
                [CSA_INDEXER_CHUNK_T],
                [chunk_t0],
            )
            forest_ready = pl.system.task_dummy(
                deps=[index_commit_dep, late_dep, projection_done]
            )
            chunk_topk_scores = pl.create_tensor(
                [CSA_INDEXER_CHUNK_T, CSA_TOPK], dtype=pl.FP32
            )
            chunk_topk_indices = pl.create_tensor(
                [CSA_INDEXER_CHUNK_T, CSA_TOPK], dtype=pl.INT32
            )
            # Global page list + request offsets are full-active and reused per
            # chunk: the forest reads only its own candidates via the
            # leaf descriptors, so the whole global list is passed unchanged.
            chunk_leaf_view = pl.slice(
                leaf_descriptors,
                [1, leaf_rows, PHASE_D_LEAF_FIELDS],
                [descriptor_chunk, 0, 0],
            )
            chunk_leaf = pl.reshape(
                chunk_leaf_view, [leaf_rows, PHASE_D_LEAF_FIELDS]
            )
            chunk_pair_view = pl.slice(
                pair_descriptors,
                [1, pair_rows, PHASE_D_PAIR_FIELDS],
                [descriptor_chunk, 0, 0],
            )
            chunk_pair = pl.reshape(
                chunk_pair_view, [pair_rows, PHASE_D_PAIR_FIELDS]
            )
            chunk_singleton_view = pl.slice(
                singleton_descriptors,
                [1, singleton_rows, PHASE_D_SINGLETON_FIELDS],
                [descriptor_chunk, 0, 0],
            )
            chunk_singleton = pl.reshape(
                chunk_singleton_view,
                [singleton_rows, PHASE_D_SINGLETON_FIELDS],
            )
            chunk_upper_view = pl.slice(
                upper_descriptors,
                [1, upper_rows, PHASE_D_UPPER_FIELDS],
                [descriptor_chunk, 0, 0],
            )
            chunk_upper = pl.reshape(
                chunk_upper_view, [upper_rows, PHASE_D_UPPER_FIELDS]
            )
            chunk_root_view = pl.slice(
                root_descriptors,
                [1, CSA_INDEXER_CHUNK_T, PHASE_D_ROOT_FIELDS],
                [descriptor_chunk, 0, 0],
            )
            chunk_root = pl.reshape(
                chunk_root_view, [CSA_INDEXER_CHUNK_T, PHASE_D_ROOT_FIELDS]
            )
            pair_group_actual_count = pl.read(
                pair_group_chunk_counts, [descriptor_chunk]
            )
            singleton_actual_count = pl.read(
                singleton_chunk_counts, [descriptor_chunk]
            )
            upper_merge_actual_count = pl.read(
                upper_merge_chunk_counts, [descriptor_chunk]
            )
            root_completion = pl.array.create(1, pl.TASK_ID)
            active_score_topk_forest(
                forest_query_vectors,
                forest_query_scales,
                forest_query_weights,
                idx_kv_cache_flat,
                idx_kv_scale_flat,
                forest_query_request_ids,
                idx_pages,
                idx_page_offsets,
                idx_windows,
                request_epochs,
                chunk_leaf,
                chunk_pair,
                chunk_singleton,
                chunk_upper,
                chunk_root,
                chunk_pair_arena,
                chunk_topk_scores,
                chunk_topk_indices,
                pair_group_actual_count,
                singleton_actual_count,
                upper_merge_actual_count,
                forest_ready,
                root_completion,
            )
            # Publish the chunk's Top-512 indices to the full-active output.
            # For B=1/S=8 has 8 real and 8 padding queries in the final chunk.
            # Only ``topk_indices`` is a ``pl.Out``; per-chunk scores stay
            # internal (see the param comment above) so the chunk loop carries
            # a single Out and avoids the PyPTO phi-swap bug.
            with pl.spmd(
                CSA_INDEXER_CHUNK_T,
                name_hint="csa_idx_topk_chunk_commit",
                deps=[root_completion[0]],
            ) as topk_commit_tid:
                row = pl.tile.get_block_idx()
                global_row = chunk_t0 + row
                if global_row < query_count:
                    topk_indices[global_row : global_row + 1, :] = (
                        chunk_topk_indices[row : row + 1, :]
                    )
            # Record this chunk's commit TaskId in the fan-in buffer and yield
            # the updated array for the next iteration / post-loop join.  This
            # mirrors the HCA-precedent pattern (decode_indexer_topk.py
            # ``root_tids_iter``/``root_tids_after_roots``): the spmd TaskId
            # and the array record+yield live inside the loop scope.
            commit_tids_iter[chunk] = topk_commit_tid
            commit_tids_after = pl.yield_(commit_tids_iter)

    # Explicit fan-in: wait for every active chunk's commit before publishing
    # the forest completion.  ``task_dummy(deps=[...])`` takes an ``Array`` of
    # TaskIds per its ``Sequence[Scalar | Array | None]`` signature, so the
    # whole fixed-size buffer is joined here.  Chunks beyond ``chunk_count``
    # were never written (their slots retain the array's default); those are
    # filtered as no-op deps.  This replaces the prior ``csa_idx_topk_ready``
    # barrier, which used ``allow_early_resolve=True`` with no explicit
    # dependency on the chunk commits and let the sparse consumer observe
    # partially-committed topk_indices (16K non-deterministic divergence).
    topk_done = pl.system.task_dummy(deps=[commit_tids_after])
    completion[0] = topk_done
    return topk_indices


@pl.jit
def indexer_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[T_DYN, HALF_ROPE], pl.FP32],
    sin: pl.Tensor[[T_DYN, HALF_ROPE], pl.FP32],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    query_vectors: pl.InOut[pl.Tensor[
        [T_PAD_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8
    ]],
    query_scales: pl.InOut[pl.Tensor[[T_PAD_DYN, IDX_N_HEADS], pl.FP32]],
    query_weights: pl.InOut[pl.Tensor[[T_PAD_DYN, IDX_N_HEADS], pl.FP32]],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROW_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROW_DYN, 1], pl.FP32],
    query_request_ids: pl.Tensor[[T_PAD_DYN], pl.INT32],
    idx_pages: pl.Tensor[[PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[REQUEST_OFFSET_DYN], pl.INT32],
    idx_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32
    ],
    pair_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, PAIR_GROUP_DYN, PHASE_D_PAIR_FIELDS], pl.INT32
    ],
    singleton_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS],
        pl.INT32,
    ],
    upper_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, UPPER_MERGE_DYN, PHASE_D_UPPER_FIELDS],
        pl.INT32,
    ],
    root_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, CSA_INDEXER_CHUNK_T, PHASE_D_ROOT_FIELDS],
        pl.INT32,
    ],
    pair_group_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    singleton_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    upper_merge_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    # Only ``topk_indices`` is a ``pl.Out``; ``topk_scores`` is dropped to keep
    # a single Out crossing the chunk loop (PyPTO phi-swap bug, see indexer).
    topk_indices: pl.Out[pl.Tensor[[T_DYN, CSA_TOPK], pl.INT32]],
):
    """Standalone gate for projection plus the exact Top-512 forest.

    The half-width cos/sin rows arrive per token (baseline ABI); the
    interleaved/sign-folded token-local rows are derived once via
    ``_rope_interleave_active_body``.  Forest descriptors are chunk-major
    with their per-chunk ragged count bound at axis 1. The standalone fixture
    uses the same chunk-major descriptors as the full CSA path; active rows
    may span multiple 16-query chunks.
    """
    x.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    query_vectors.bind_dynamic(0, T_PAD_DYN)
    query_scales.bind_dynamic(0, T_PAD_DYN)
    query_weights.bind_dynamic(0, T_PAD_DYN)
    idx_kv_cache_flat.bind_dynamic(0, IDX_ROW_DYN)
    idx_kv_scale_flat.bind_dynamic(0, IDX_ROW_DYN)
    query_request_ids.bind_dynamic(0, T_PAD_DYN)
    idx_pages.bind_dynamic(0, PAGE_DYN)
    idx_page_offsets.bind_dynamic(0, REQUEST_OFFSET_DYN)
    idx_windows.bind_dynamic(0, B_DYN)
    request_epochs.bind_dynamic(0, B_DYN)
    leaf_descriptors.bind_dynamic(1, LEAF_DYN)
    pair_descriptors.bind_dynamic(1, PAIR_GROUP_DYN)
    singleton_descriptors.bind_dynamic(1, SINGLETON_DYN)
    upper_descriptors.bind_dynamic(1, UPPER_MERGE_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    # Derive the interleaved/sign-folded token-local RoPE rows once.  Keep the
    # scratch extent tied to the runtime query count; using the compile-time
    # upper bound here conflicts with indexer's dynamic T_DYN for B < B_MAX.
    query_count = pl.tensor.dim(x, 0)
    cos_il = pl.create_tensor([query_count, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([query_count, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_completion = pl.array.create(1, pl.TASK_ID)
    _rope_interleave_active_body(cos, sin, cos_il, sin_signed, rope_completion)
    index_commit_dep = pl.system.task_dummy(deps=[])
    late_dep = pl.system.task_dummy(deps=[rope_completion[0]])
    completion = pl.array.create(1, pl.TASK_ID)
    indexer(
        x,
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        weights_proj,
        cos_il,
        sin_signed,
        hadamard,
        query_vectors,
        query_scales,
        query_weights,
        idx_kv_cache_flat,
        idx_kv_scale_flat,
        query_request_ids,
        idx_pages,
        idx_page_offsets,
        idx_windows,
        request_epochs,
        leaf_descriptors,
        pair_descriptors,
        singleton_descriptors,
        upper_descriptors,
        root_descriptors,
        topk_indices,
        pair_group_chunk_counts,
        singleton_chunk_counts,
        upper_merge_chunk_counts,
        index_commit_dep,
        late_dep,
        pl.cast(0, pl.INDEX),
        late_dep,
        completion,
    )
    return topk_indices


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize a per-output-channel-symmetric INT8 weight + FP32 scale by simulating the
    real DeepSeek-V4-Flash MXFP8 quant grid (e4m3, 128x128-block E8M0 scale), then re-quantizing
    per-output-channel. Used for the indexer ``idx wq_b`` (and shared by decode_csa),
    which follows the same FP8 grid as the shared experts: ~200 discrete levels, ~1.1% zero
    spike, per-channel scale CV ~0.61. A plain randn INT8 misses that level/scale structure.
    ``chan_cv`` (log-space source-gain std) injects the per-output-channel magnitude spread the
    coarse 128-block scale leaves behind; per-channel INT8 is scale-invariant, so the grid sets
    the level shape and ``dequant_std`` only sets the absolute scale magnitude.

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel scale
    shape ([out, in] -> scale [out]).
    """
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):   # e4m3 + 128x128-block E8M0 (round-up) scale on (out, in)
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))  # per-channel gain
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def golden_indexer(tensors):
    """Torch reference for the packed Top-512 forest.

    The standalone and full CSA fixtures provide the complete indexer inputs,
    so use the same projected-query, INT8 score, page translation, and Top-K
    semantics as the device path.  A descriptor-only fallback is retained for
    small metadata probes that intentionally do not provide the numerical
    inputs.
    """
    import torch

    required = {
        "x", "qr", "qr_scale", "wq_b", "wq_b_scale", "weights_proj",
        "cos", "sin", "hadamard", "idx_kv_cache_flat", "idx_kv_scale_flat",
        "query_request_ids", "idx_pages", "idx_page_offsets", "idx_windows",
        "request_epochs",
    }
    if required.issubset(tensors):
        from utils import int8_quant_per_row

        x = tensors["x"].float()
        qr = tensors["qr"].to(torch.int32)
        qr_scale = tensors["qr_scale"].float()
        wq_b = tensors["wq_b"].to(torch.int32)
        wq_b_scale = tensors["wq_b_scale"].float()
        query_count = x.shape[0]

        qr_proj = qr @ wq_b
        qr_proj = qr_proj.float() * qr_scale * wq_b_scale.view(1, -1)
        query = qr_proj.view(query_count, IDX_N_HEADS, IDX_HEAD_DIM)
        rope_dim = ROPE_HEAD_DIM
        rope = query[..., -rope_dim:]
        # cos/sin carry token-local rows followed by event-local rows
        # (appended by decode_csa's spec); the indexer only consumes the
        # token-local prefix.
        cos = tensors["cos"][:query_count].float()
        sin = tensors["sin"][:query_count].float()
        dup = torch.arange(rope_dim, device=cos.device, dtype=torch.long) // 2
        sign = torch.where(
            torch.arange(rope_dim, device=cos.device) % 2 == 0,
            -1.0,
            1.0,
        ).to(torch.float32)
        cos_il = cos[:, dup]
        sin_signed = sin[:, dup] * sign
        swap = torch.arange(rope_dim, device=cos.device) ^ 1
        rotated = rope * cos_il[:, None, :] + rope[..., swap] * sin_signed[:, None, :]
        query = torch.cat([query[..., :IDX_NOPE_HEAD_DIM], rotated], dim=-1)
        # The device qh matmul consumes BF16 query and BF16 Hadamard weights
        # and accumulates into FP32.  Keep both operands rounded to BF16 in
        # the standalone reference so boundary lanes follow the device.
        query = query.to(torch.bfloat16).float() @ tensors["hadamard"].to(
            torch.bfloat16
        ).float()
        query_i8, query_scale = int8_quant_per_row(query)
        query_i8 = query_i8.view(query_count, IDX_N_HEADS, IDX_HEAD_DIM)
        query_scale = query_scale.view(query_count, IDX_N_HEADS, 1)
        query_weights = (
            x.to(torch.bfloat16).float()
            @ tensors["weights_proj"].to(torch.bfloat16).float()
        ) * WEIGHTS_SCALE
        indices = torch.full(
            (query_count, CSA_TOPK), -1, dtype=torch.int32, device=x.device
        )
        leaf_descriptors = tensors["leaf_descriptors"]
        logical_candidates = [[] for _ in range(query_count)]
        for chunk in range(leaf_descriptors.shape[0]):
            for leaf in leaf_descriptors[chunk]:
                local_query = int(leaf[PHASE_D_LEAF_QUERY].item())
                valid = int(leaf[PHASE_D_LEAF_VALID].item())
                if local_query < 0 or valid <= 0:
                    continue
                global_query = chunk * CSA_INDEXER_CHUNK_T + local_query
                if global_query >= query_count:
                    continue
                begin = int(leaf[PHASE_D_LEAF_BEGIN].item())
                logical_candidates[global_query].extend(range(begin, begin + valid))

        idx_cache = tensors["idx_kv_cache_flat"].to(torch.int32)
        idx_scale = tensors["idx_kv_scale_flat"].float().reshape(-1)
        query_request_ids = tensors["query_request_ids"].to(torch.int64)
        idx_pages = tensors["idx_pages"].to(torch.int64)
        idx_page_offsets = tensors["idx_page_offsets"].to(torch.int64)
        idx_windows = tensors["idx_windows"].to(torch.int64)
        request_epochs = tensors["request_epochs"].to(torch.int64)

        for query_id, logical_ids in enumerate(logical_candidates):
            if not logical_ids:
                continue
            request = int(query_request_ids[query_id].item())
            if request < 0 or request + 1 >= idx_page_offsets.numel():
                continue
            page_begin = int(idx_page_offsets[request].item())
            page_end = int(idx_page_offsets[request + 1].item())
            page_total = page_end - page_begin
            if page_total <= 0:
                continue
            valid_begin = int(idx_windows[request, 0].item())
            valid_end = int(idx_windows[request, 1].item())
            head = int(idx_windows[request, 2].item())
            epoch = int(request_epochs[request].item())
            page_base = valid_begin // BLOCK_SIZE
            source_rows = []
            source_ids = []
            for logical_id in logical_ids:
                if logical_id < valid_begin or logical_id >= valid_end:
                    continue
                relative_page = logical_id // BLOCK_SIZE - page_base
                if relative_page < 0 or relative_page >= page_total:
                    continue
                page_index = (head + relative_page) % page_total
                page_entry = page_begin + page_index
                if page_entry < 0 or page_entry >= idx_pages.shape[0]:
                    continue
                physical_page, page_epoch = idx_pages[page_entry].tolist()
                if page_epoch != epoch or physical_page < 0:
                    continue
                source_row = physical_page * BLOCK_SIZE + logical_id % BLOCK_SIZE
                if source_row < 0 or source_row >= idx_cache.shape[0]:
                    continue
                source_rows.append(source_row)
                source_ids.append(logical_id)
            if not source_rows:
                continue
            source_rows_t = torch.tensor(source_rows, dtype=torch.long, device=x.device)
            kv_i8 = idx_cache[source_rows_t]
            kv_scale = idx_scale[source_rows_t]
            score_i32 = query_i8[query_id].to(torch.int32) @ kv_i8.transpose(0, 1)
            score = score_i32.float() * query_scale[query_id]
            score = torch.relu(score) * query_weights[query_id].view(-1, 1)
            score = score.sum(dim=0) * kv_scale
            order = torch.argsort(score, descending=True, stable=True)
            keep = min(CSA_TOPK, order.numel())
            selected = torch.tensor(
                [source_ids[int(i)] for i in order[:keep].tolist()],
                dtype=torch.int32,
                device=x.device,
            )
            indices[query_id, :keep] = selected
        tensors["topk_indices"][:] = indices
        return

    query_count = tensors["topk_indices"].shape[0]
    scores = torch.full(
        (query_count, CSA_TOPK), FP32_NEG_INF, dtype=torch.float32
    )
    indices = torch.full((query_count, CSA_TOPK), -1, dtype=torch.int32)
    leaf_descriptors = tensors["leaf_descriptors"]
    chunk_count = leaf_descriptors.shape[0]
    for chunk in range(chunk_count):
        for row in leaf_descriptors[chunk]:
            local_query = int(row[PHASE_D_LEAF_QUERY].item())
            if local_query < 0:
                continue
            global_query = chunk * CSA_INDEXER_CHUNK_T + local_query
            if global_query >= query_count:
                continue
            begin = int(row[PHASE_D_LEAF_BEGIN].item())
            valid = int(row[PHASE_D_LEAF_VALID].item())
            if valid <= 0:
                continue
            candidates = list(range(begin, begin + valid))
            existing = indices[global_query]
            existing = [c for c in existing.tolist() if c >= 0]
            merged = sorted(set(existing) | set(candidates))
            kept = min(CSA_TOPK, len(merged))
            if kept:
                scores[global_query, :kept] = 0.0
                indices[global_query, :kept] = torch.tensor(
                    merged[:kept], dtype=torch.int32
                )
    # ``topk_scores`` is no longer a device output (single-Out workaround for
    # the PyPTO phi-swap bug); scores remain a golden-side internal helper and
    # are not written back to ``tensors``.  Only ``topk_indices`` is published.
    tensors["topk_indices"][:] = indices


def build_tensor_specs(start_pos=None, batch=B):
    """Build indexer fixtures from ``start_pos`` and ``batch`` (baseline pattern).

    The standalone test is single-chunk (tokens <= CSA_MAX_QUERIES=16), so the
    chunk-count tensors carry one row.  Candidate counts are derived from each
    request's final position so the forest exercises real leaf/merge shapes.
    """
    import torch

    from golden import TensorSpec
    from utils import (
        build_rope_tables,
        csa_decode_start_set,
        int8_quant_per_row,
        materialize_half_rope_tables,
        position_ids_from_starts,
        resolve_start_positions,
    )

    tokens = batch * S
    # Pad the query output tensors to a CSA_INDEXER_CHUNK_T multiple so the
    # forest's fixed 16-row pl.slice views never read out-of-bounds memory
    # in the final partial chunk.  For B=1/S=8: tokens=8 -> padded_tokens=16.
    # Input tensors (x, qr, qr_scale, cos, sin) stay at ``tokens``: the
    # projection SPMD keys on ``query_count`` from x.dim(0), not the padded
    # extent.  Padding rows [tokens:padded_tokens] carry spec-init zeros (or
    # -1 for request IDs) and are never written by the device projection.
    # The forest scores them deterministically (zero query -> zero score ->
    # harmless Top-K output) and the commit SPMD discards them via
    # ``if global_row < query_count``.
    padded_tokens = (
        (tokens + CSA_INDEXER_CHUNK_T - 1) // CSA_INDEXER_CHUNK_T
    ) * CSA_INDEXER_CHUNK_T
    if batch <= 0 or batch > B:
        raise ValueError(
            f"--batch must be in [1, {B}], got {batch}"
        )

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(
        M, COMPRESS_RATIO, dtype=torch.bfloat16, max_seq_len=M.max_position_embeddings
    )

    def init_default_start_pos():
        return csa_decode_start_set(
            batch=batch, seq=S, compress_ratio=COMPRESS_RATIO,
            state_block_size=2, cache_tile=min(64, BLOCK_SIZE),
        )

    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=batch,
            seq=S,
            max_seq_len=M.max_position_embeddings,
            default_fn=init_default_start_pos,
        )

    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S)

    starts = init_start_pos()
    # Each request's final position determines its candidate count for the
    # ratio-4 indexer: candidates = (final_position + row + 1) // 4 per S-row.
    candidate_counts = []
    query_request_ids = []
    request_candidate_caps = []
    for request in range(batch):
        final_length = starts[request] + S
        if final_length > M.max_position_embeddings:
            raise ValueError("trace request exceeds the 1M context ceiling")
        candidate_counts.extend(
            (final_length - S + row + 1) // COMPRESS_RATIO
            for row in range(S)
        )
        query_request_ids.extend([request] * S)
        request_candidate_caps.append(final_length // COMPRESS_RATIO)
    logical_begins = [0] * tokens

    def _pack_forest(candidate_counts, logical_begins):
        """Pack a per-chunk heterogeneous forest using the chunk-major ABI.

        Each 16-query chunk gets its own independent leaf/pair/singleton/
        upper/root descriptors with LOCAL query indices (0-15) and LOCAL
        leaf IDs/slots.  Chunks are stacked along axis 0 with per-chunk
        padding to the max row count, matching the production
        ``[CSA_INDEXER_MAX_CHUNKS, *_DYN, FIELDS]`` ABI.
        """
        if len(candidate_counts) != len(logical_begins):
            raise ValueError("candidate_counts and logical_begins must have equal length")

        ready_frontier = 8
        invalid_slot = CSA_TOPK_INVALID_TASK_SLOT
        chunk_count = (len(candidate_counts) + CSA_INDEXER_CHUNK_T - 1) // CSA_INDEXER_CHUNK_T

        chunk_leaf = []
        chunk_pair = []
        chunk_singleton = []
        chunk_upper = []
        chunk_root = []
        chunk_pair_counts = []
        chunk_singleton_counts = []
        chunk_upper_counts = []

        for chunk in range(chunk_count):
            q0 = chunk * CSA_INDEXER_CHUNK_T
            q1 = min(q0 + CSA_INDEXER_CHUNK_T, len(candidate_counts))
            node_kinds = []
            node_leaf_ids = []
            node_left_slots = []
            node_right_slots = []
            node_credit_slots = []
            leaf_rows = []
            root_slots = []

            def append_leaf(local_query, local_leaf, valid, credit_slot):
                leaf_id = len(leaf_rows)
                slot = len(node_kinds)
                global_query = q0 + local_query
                begin = logical_begins[global_query] + local_leaf * CSA_CANDIDATES_PER_LEAF
                leaf_rows.append([local_query, begin, valid, slot, credit_slot])
                node_kinds.append(0)
                node_leaf_ids.append(leaf_id)
                node_left_slots.append(-1)
                node_right_slots.append(-1)
                node_credit_slots.append(credit_slot)
                return slot

            def append_merge(left_slot, right_slot):
                slot = len(node_kinds)
                node_kinds.append(1)
                node_leaf_ids.append(-1)
                node_left_slots.append(left_slot)
                node_right_slots.append(right_slot)
                node_credit_slots.append(invalid_slot)
                return slot

            for local_query in range(q1 - q0):
                global_query = q0 + local_query
                candidates = candidate_counts[global_query]
                leaves = (candidates + CSA_CANDIDATES_PER_LEAF - 1) // CSA_CANDIDATES_PER_LEAF
                if leaves == 0:
                    root_slots.append(-1)
                    continue

                level1_slots = []
                for group in range(leaves // 2):
                    credit = invalid_slot
                    if group >= ready_frontier:
                        credit = level1_slots[group - ready_frontier]
                    left_leaf = group * 2
                    left_valid = min(
                        CSA_CANDIDATES_PER_LEAF,
                        candidates - left_leaf * CSA_CANDIDATES_PER_LEAF,
                    )
                    left_slot = append_leaf(local_query, left_leaf, left_valid, credit)
                    right_leaf = left_leaf + 1
                    right_valid = min(
                        CSA_CANDIDATES_PER_LEAF,
                        candidates - right_leaf * CSA_CANDIDATES_PER_LEAF,
                    )
                    right_slot = append_leaf(local_query, right_leaf, right_valid, credit)
                    level1_slots.append(append_merge(left_slot, right_slot))

                if leaves % 2:
                    local_leaf = leaves - 1
                    credit = invalid_slot
                    if len(level1_slots) >= ready_frontier:
                        credit = level1_slots[-ready_frontier]
                    valid = min(
                        CSA_CANDIDATES_PER_LEAF,
                        candidates - local_leaf * CSA_CANDIDATES_PER_LEAF,
                    )
                    level1_slots.append(append_leaf(local_query, local_leaf, valid, credit))

                current = level1_slots
                while len(current) > 1:
                    next_level = []
                    for pair in range(len(current) // 2):
                        next_level.append(append_merge(current[2 * pair], current[2 * pair + 1]))
                    if len(current) % 2:
                        next_level.append(current[-1])
                    current = next_level
                root_slots.append(current[0])

            pair_rows = []
            upper_rows = []
            paired_leaf_slots = set()
            for output_slot, kind in enumerate(node_kinds):
                if kind != 1:
                    continue
                left_slot = node_left_slots[output_slot]
                right_slot = node_right_slots[output_slot]
                if node_kinds[left_slot] == 0 and node_kinds[right_slot] == 0:
                    pair_rows.append([
                        node_leaf_ids[left_slot],
                        node_leaf_ids[right_slot],
                        left_slot,
                        right_slot,
                        output_slot,
                        node_credit_slots[left_slot],
                    ])
                    paired_leaf_slots.update((left_slot, right_slot))
                else:
                    upper_rows.append([left_slot, right_slot, output_slot])

            singleton_slots = [
                slot
                for slot, kind in enumerate(node_kinds)
                if kind == 0 and slot not in paired_leaf_slots
            ]
            singleton_rows = [
                [node_leaf_ids[slot], slot, node_credit_slots[slot]]
                for slot in singleton_slots
            ]
            root_deps = [root if root >= 0 else invalid_slot for root in root_slots]
            # Pad root rows to CSA_INDEXER_CHUNK_T (partial chunk at the tail).
            while len(root_deps) < CSA_INDEXER_CHUNK_T:
                root_deps.append(invalid_slot)
            while len(root_slots) < CSA_INDEXER_CHUNK_T:
                root_slots.append(-1)

            chunk_leaf.append(leaf_rows)
            chunk_pair.append(pair_rows)
            chunk_singleton.append(singleton_rows)
            chunk_upper.append(upper_rows)
            chunk_root.append(
                [[r, d] for r, d in zip(root_slots, root_deps)]
            )
            chunk_pair_counts.append(len(pair_rows))
            chunk_singleton_counts.append(len(singleton_rows))
            chunk_upper_counts.append(len(upper_rows))

        # The public ABI keeps a fixed leading chunk axis even when the
        # runtime query count uses fewer than the compile-time eight chunks.
        # Fill inactive chunks with invalid descriptors and zero counts so
        # B=1/4 fixtures bind the same tensor rank and extent as B=16.
        invalid_root_chunk = [
            [-1, invalid_slot] for _ in range(CSA_INDEXER_CHUNK_T)
        ]
        while len(chunk_leaf) < CSA_INDEXER_MAX_CHUNKS:
            chunk_leaf.append([])
            chunk_pair.append([])
            chunk_singleton.append([])
            chunk_upper.append([])
            chunk_root.append([list(row) for row in invalid_root_chunk])
            chunk_pair_counts.append(0)
            chunk_singleton_counts.append(0)
            chunk_upper_counts.append(0)

        # Runtime tensor handles must have a non-null base address.  A short
        # request can legitimately have no pair or upper-merge descriptors
        # (for example B=1 at start_pos=120), but exposing a zero-row tensor
        # would produce a null BufferDescriptor before the device is launched.
        # Keep one invalid sentinel row for every empty descriptor family;
        # the corresponding actual-count scalar remains zero, so no task
        # consumes the sentinel.
        max_leaf = max(1, max((len(r) for r in chunk_leaf), default=0))
        max_pair = max(1, max((len(r) for r in chunk_pair), default=0))
        max_singleton = max(1, max((len(r) for r in chunk_singleton), default=0))
        max_upper = max(1, max((len(r) for r in chunk_upper), default=0))

        def pad_chunk(rows, max_rows, width):
            padded = [list(r) for r in rows]
            while len(padded) < max_rows:
                padded.append([-1] * width)
            return padded

        def stack(chunk_rows, max_rows, width):
            if max_rows == 0:
                return torch.empty(
                    CSA_INDEXER_MAX_CHUNKS, 0, width, dtype=torch.int32
                )
            return torch.tensor(
                [pad_chunk(rows, max_rows, width) for rows in chunk_rows],
                dtype=torch.int32,
            )

        return {
            "leaf_descriptors": stack(chunk_leaf, max_leaf, PHASE_D_LEAF_FIELDS),
            "pair_descriptors": stack(chunk_pair, max_pair, PHASE_D_PAIR_FIELDS),
            "singleton_descriptors": stack(chunk_singleton, max_singleton, PHASE_D_SINGLETON_FIELDS),
            "upper_descriptors": stack(chunk_upper, max_upper, PHASE_D_UPPER_FIELDS),
            "root_descriptors": torch.tensor(
                chunk_root, dtype=torch.int32
            ).reshape(
                CSA_INDEXER_MAX_CHUNKS,
                CSA_INDEXER_CHUNK_T,
                PHASE_D_ROOT_FIELDS,
            ),
            "pair_group_chunk_counts": torch.tensor(chunk_pair_counts, dtype=torch.int32),
            "singleton_chunk_counts": torch.tensor(chunk_singleton_counts, dtype=torch.int32),
            "upper_merge_chunk_counts": torch.tensor(chunk_upper_counts, dtype=torch.int32),
        }

    forest = _pack_forest(candidate_counts, logical_begins)

    # Ragged index pool: pages, offsets, windows, epochs.
    page_counts = []
    for begin, candidates in zip(logical_begins, request_candidate_caps):
        if candidates <= 0:
            page_counts.append(0)
        else:
            page_counts.append(
                (begin + candidates + BLOCK_SIZE - 1) // BLOCK_SIZE
                - begin // BLOCK_SIZE
            )
    page_offsets = [0]
    for count in page_counts:
        page_offsets.append(page_offsets[-1] + count)
    total_pages = page_offsets[-1]
    pages_list = []
    for request, count in enumerate(page_counts):
        base = page_offsets[request]
        # Compact physical pages are deliberately permuted per request.
        pages_list.extend([base + local for local in reversed(range(count))])
    pages = torch.tensor(
        [[page, 11] for page in pages_list], dtype=torch.int32
    ).reshape(-1, 2)
    windows = torch.tensor(
        [
            [begin, begin + candidates, request % max(page_counts[request], 1)]
            for request, (begin, candidates) in enumerate(
                zip(logical_begins, request_candidate_caps)
            )
        ],
        dtype=torch.int32,
    )
    page_offsets_t = torch.tensor(page_offsets, dtype=torch.int32)

    # Half-width token-local RoPE tables: one row per active token.
    rope_positions = init_position_ids().to(torch.int64).reshape(-1)
    cos_half, sin_half = materialize_half_rope_tables(
        shared_freqs_cos, shared_freqs_sin, rope_positions
    )

    # idx wq_b: simulate the real MXFP8 grid via gen_shared_weight.
    wq_b_i8_T, wq_b_scale = gen_shared_weight(
        (IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56
    )
    wq_b_i8 = wq_b_i8_T.t().contiguous()
    qr_i8, qr_scale = int8_quant_per_row(torch.rand(tokens, Q_LORA))

    # Real Hadamard matrix (Sylvester construction), matching the full
    # CSA fixture.  An identity stand-in left the score landscape flat
    # and made boundary candidates hypersensitive to the BF16 tiling
    # difference between the device's per-query Hadamard matmul and
    # golden's batched matmul, producing spurious Top-512 mismatches
    # that never occur in the full pipeline.
    _h = torch.ones((1, 1))
    while _h.shape[0] < IDX_HEAD_DIM:
        _h = torch.cat([
            torch.cat([_h, _h], dim=1),
            torch.cat([_h, -_h], dim=1),
        ], dim=0)
    hadamard_init = (_h / (IDX_HEAD_DIM ** 0.5)).to(torch.bfloat16)
    x_init = torch.rand(tokens, D).to(torch.bfloat16)
    weights_proj_init = torch.randn(D, IDX_N_HEADS) * 0.2218
    idx_kv_bf16 = torch.randn(total_pages * BLOCK_SIZE, IDX_HEAD_DIM).to(torch.bfloat16)
    idx_kv_i8, idx_kv_scale = int8_quant_per_row(idx_kv_bf16.float())

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=lambda: x_init),
        TensorSpec("qr", [tokens, Q_LORA], torch.int8, init_value=lambda: qr_i8),
        TensorSpec("qr_scale", [tokens, 1], torch.float32, init_value=lambda: qr_scale),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16,
                   init_value=lambda: weights_proj_init),
        TensorSpec("cos", [tokens, HALF_ROPE], torch.float32, init_value=lambda: cos_half),
        TensorSpec("sin", [tokens, HALF_ROPE], torch.float32, init_value=lambda: sin_half),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=lambda: hadamard_init),
        TensorSpec("query_vectors", [padded_tokens, IDX_N_HEADS, IDX_HEAD_DIM], torch.int8),
        TensorSpec("query_scales", [padded_tokens, IDX_N_HEADS], torch.float32),
        TensorSpec("query_weights", [padded_tokens, IDX_N_HEADS], torch.float32),
        TensorSpec("idx_kv_cache_flat", [total_pages * BLOCK_SIZE, IDX_HEAD_DIM],
                   torch.int8, init_value=lambda: idx_kv_i8),
        TensorSpec("idx_kv_scale_flat", [total_pages * BLOCK_SIZE, 1],
                   torch.float32, init_value=lambda: idx_kv_scale),
        TensorSpec(
            "query_request_ids",
            [padded_tokens],
            torch.int32,
            init_value=lambda: torch.tensor(
                query_request_ids + [-1] * (padded_tokens - tokens),
                dtype=torch.int32,
            ),
        ),
        TensorSpec("idx_pages", list(pages.shape), torch.int32, init_value=lambda: pages),
        TensorSpec("idx_page_offsets", [len(page_offsets)], torch.int32, init_value=lambda: page_offsets_t),
        TensorSpec("idx_windows", list(windows.shape), torch.int32, init_value=lambda: windows),
        TensorSpec("request_epochs", [len(page_counts)], torch.int32, init_value=lambda: torch.full((len(page_counts),), 11, dtype=torch.int32)),
        TensorSpec("leaf_descriptors", list(forest["leaf_descriptors"].shape), torch.int32, init_value=lambda: forest["leaf_descriptors"]),
        TensorSpec("pair_descriptors", list(forest["pair_descriptors"].shape), torch.int32, init_value=lambda: forest["pair_descriptors"]),
        TensorSpec("singleton_descriptors", list(forest["singleton_descriptors"].shape), torch.int32, init_value=lambda: forest["singleton_descriptors"]),
        TensorSpec("upper_descriptors", list(forest["upper_descriptors"].shape), torch.int32, init_value=lambda: forest["upper_descriptors"]),
        TensorSpec("root_descriptors", list(forest["root_descriptors"].shape), torch.int32, init_value=lambda: forest["root_descriptors"]),
        TensorSpec("pair_group_chunk_counts", list(forest["pair_group_chunk_counts"].shape), torch.int32, init_value=lambda: forest["pair_group_chunk_counts"]),
        TensorSpec("singleton_chunk_counts", list(forest["singleton_chunk_counts"].shape), torch.int32, init_value=lambda: forest["singleton_chunk_counts"]),
        TensorSpec("upper_merge_chunk_counts", list(forest["upper_merge_chunk_counts"].shape), torch.int32, init_value=lambda: forest["upper_merge_chunk_counts"]),
        # ``topk_scores`` is intentionally not an output spec: single ``pl.Out``
        # workaround for the PyPTO phi-swap bug (see indexer param comment).
        TensorSpec("topk_indices", [tokens, CSA_TOPK], torch.int32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse

    import torch

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=B,
        help=f"runtime request count from 1 to {B} (the compile-time "
             "upper bound). The batch axes are pl.dynamic, so one compiled program "
             "serves every value.",
    )
    parser.add_argument(
        "--enable-l2-swimlane",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="L2 swimlane level: 0=off, 1=AICore timing, 2=+AICPU timing.",
    )
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument(
        "--start-pos",
        type=int,
        default=None,
        help="Uniform fixture-only start_pos override for all batches; "
        "default (unset) uses the canonical per-batch CSA set.",
    )
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")

    def topk_indices_baseline_compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        """Check Top-K indices with the baseline score-order semantics.

        This wrapper no longer publishes a second score output because the
        PyPTO two-``pl.Out`` loop carry has a known phi bug.  Reconstruct the
        paired reference scores only for rows whose index multiset differs.
        A set difference is accepted only when the actual and expected
        difference scores are equivalent within the comparison tolerance;
        this preserves legitimate cutoff ties without accepting an arbitrary
        valid-looking candidate subset.
        """
        del actual_outputs, expected_outputs
        if actual.shape != expected.shape:
            return False, (
                f"    topk shape mismatch: {tuple(actual.shape)} "
                f"vs {tuple(expected.shape)}"
            )
        if torch.any((actual < -1) | (actual >= CSA_MAX_CANDIDATES)):
            return False, "    topk contains an invalid candidate index"
        actual_sorted = torch.sort(actual.to(torch.int64), dim=-1).values
        expected_sorted = torch.sort(expected.to(torch.int64), dim=-1).values
        set_mismatch = torch.nonzero(
            actual_sorted != expected_sorted,
            as_tuple=False,
        )
        position_mismatch = torch.nonzero(actual != expected, as_tuple=False)
        if set_mismatch.numel() or position_mismatch.numel():
            row_source = set_mismatch if set_mismatch.numel() else position_mismatch
            row = int(row_source[0, 0].item())
            from collections import Counter

            actual_counts = Counter(actual[row].tolist())
            expected_counts = Counter(expected[row].tolist())
            actual_only = sorted(
                (candidate, count)
                for candidate, count in (actual_counts - expected_counts).items()
            )
            expected_only = sorted(
                (candidate, count)
                for candidate, count in (expected_counts - actual_counts).items()
            )

            # A Top-K tie may replace one candidate with another candidate at
            # the same cutoff score, but it may not change the number of
            # padding entries or emit duplicate/out-of-domain candidates.
            if actual_counts.get(-1, 0) != expected_counts.get(-1, 0):
                return False, (
                    f"    topk padding count mismatch at row {row}: "
                    f"actual={actual_counts.get(-1, 0)} "
                    f"expected={expected_counts.get(-1, 0)}"
                )
            actual_valid_ids = [
                int(candidate) for candidate in actual[row].tolist() if candidate >= 0
            ]
            if len(actual_valid_ids) != len(set(actual_valid_ids)):
                return False, f"    topk contains duplicate candidates at row {row}"

            chunk = row // CSA_INDEXER_CHUNK_T
            local_query = row % CSA_INDEXER_CHUNK_T
            valid_ids = set()
            leaf_rows = inputs["leaf_descriptors"][chunk]
            for leaf in leaf_rows:
                if int(leaf[PHASE_D_LEAF_QUERY].item()) != local_query:
                    continue
                begin = int(leaf[PHASE_D_LEAF_BEGIN].item())
                valid = int(leaf[PHASE_D_LEAF_VALID].item())
                if begin >= 0 and valid > 0:
                    valid_ids.update(range(begin, begin + valid))
            if not all(candidate in valid_ids for candidate in actual_valid_ids):
                return False, (
                    f"    topk contains a candidate outside the row domain at {row}"
                )

            # Reconstruct the per-candidate reference score using the same
            # batched Hadamard projection as golden_indexer.  An earlier
            # version computed the query projection per-candidate with a
            # single-row matmul (qr[q:q+1] @ wq_b); that BF16 reduction
            # diverges from golden's batched (qr @ wq_b) reduction by enough
            # to flip int8 quantization on boundary candidates, mislabeling
            # legitimate device Top-512 selections as monotonicity
            # violations.  Project all queries once, then slice.
            from utils import int8_quant_per_row

            x = inputs["x"].float()
            qr = inputs["qr"].to(torch.int32)
            qr_scale = inputs["qr_scale"].float()
            wq_b = inputs["wq_b"].to(torch.int32)
            wq_b_scale = inputs["wq_b_scale"].float()
            qc = x.shape[0]
            qp = qr @ wq_b
            qp = qp.float() * qr_scale * wq_b_scale.view(1, -1)
            q = qp.view(qc, IDX_N_HEADS, IDX_HEAD_DIM)
            cos = inputs["cos"].float()[:qc]
            sin = inputs["sin"].float()[:qc]
            dup = torch.arange(ROPE_HEAD_DIM, dtype=torch.long) // 2
            sign = torch.where(
                torch.arange(ROPE_HEAD_DIM) % 2 == 0, -1.0, 1.0
            ).to(torch.float32)
            rope = q[..., -ROPE_HEAD_DIM:]
            rot = (
                rope * cos[:, None, dup]
                + rope[..., torch.arange(ROPE_HEAD_DIM) ^ 1]
                * sin[:, None, dup]
                * sign
            )
            q = torch.cat([q[..., :IDX_NOPE_HEAD_DIM], rot], dim=-1)
            q = q.to(torch.bfloat16).float() @ inputs[
                "hadamard"
            ].to(torch.bfloat16).float()
            qi8_all, qsc_all = int8_quant_per_row(q)
            qi8_all = qi8_all.view(qc, IDX_N_HEADS, IDX_HEAD_DIM)
            qsc_all = qsc_all.view(qc, IDX_N_HEADS, 1)
            qw_all = (
                x.to(torch.bfloat16).float()
                @ inputs["weights_proj"].to(torch.bfloat16).float()
            ) * WEIGHTS_SCALE

            def candidate_score(query_id, logical_id):
                qi8 = qi8_all[query_id]
                qsc = qsc_all[query_id]
                qw = qw_all[query_id]
                request = int(inputs["query_request_ids"][query_id].item())
                page_begin = int(inputs["idx_page_offsets"][request].item())
                page_end = int(inputs["idx_page_offsets"][request + 1].item())
                page_total = page_end - page_begin
                valid_begin = int(inputs["idx_windows"][request, 0].item())
                page_base = valid_begin // BLOCK_SIZE
                head = int(inputs["idx_windows"][request, 2].item())
                relative_page = logical_id // BLOCK_SIZE - page_base
                page_index = (head + relative_page) % page_total
                page_entry = page_begin + page_index
                physical_page = int(inputs["idx_pages"][page_entry, 0].item())
                source_row = physical_page * BLOCK_SIZE + logical_id % BLOCK_SIZE
                kv = inputs["idx_kv_cache_flat"][source_row].to(torch.int32)
                scale = inputs["idx_kv_scale_flat"][source_row, 0].float()
                score = (qi8 * kv).sum(dim=-1).float() * qsc.squeeze(-1)
                score = torch.relu(score) * qw
                return float((score.sum() * scale).item())

            actual_scores = [
                FP32_NEG_INF
                if candidate < 0
                else candidate_score(row, candidate)
                for candidate in actual[row].tolist()
            ]
            expected_scores = [
                FP32_NEG_INF
                if candidate < 0
                else candidate_score(row, candidate)
                for candidate in expected[row].tolist()
            ]
            score_rtol = max(float(rtol), 1e-5)
            score_atol = max(float(atol), 1e-5)

            def scores_are_descending(scores):
                if any(
                    not torch.isfinite(torch.tensor(score)) for score in scores
                ):
                    return False
                for previous, current in zip(scores, scores[1:]):
                    if current > previous and not torch.isclose(
                        torch.tensor(current),
                        torch.tensor(previous),
                        rtol=score_rtol,
                        atol=score_atol,
                    ):
                        return False
                return True

            if not scores_are_descending(actual_scores):
                return False, (
                    f"    topk score order mismatch at row {row}: "
                    f"actual_scores={actual_scores[:8]}"
                )

            actual_only_ids = [
                int(candidate)
                for candidate, count in actual_only
                for _ in range(count)
                if candidate >= 0
            ]
            expected_only_ids = [
                int(candidate)
                for candidate, count in expected_only
                for _ in range(count)
                if candidate >= 0
            ]
            if len(actual_only_ids) != len(expected_only_ids):
                return False, (
                    f"    topk candidate count mismatch at row {row}: "
                    f"actual-only={len(actual_only_ids)} "
                    f"expected-only={len(expected_only_ids)}"
                )
            if actual_only_ids:
                actual_only_scores = sorted(
                    candidate_score(row, candidate) for candidate in actual_only_ids
                )
                expected_only_scores = sorted(
                    candidate_score(row, candidate) for candidate in expected_only_ids
                )
                if not torch.allclose(
                    torch.tensor(actual_only_scores, dtype=torch.float64),
                    torch.tensor(expected_only_scores, dtype=torch.float64),
                    rtol=score_rtol,
                    atol=score_atol,
                ):
                    return False, (
                        f"    topk candidate validation mismatch at row {row}: "
                        f"actual-only={[candidate for candidate, _ in actual_only[:8]]}, "
                        f"expected-only={[candidate for candidate, _ in expected_only[:8]]} "
                        f"actual-only-scores={actual_only_scores[:4]} "
                        f"expected-only-scores={expected_only_scores[:4]}"
                    )
        return True, ""

    result = run_jit(
        fn=indexer_test,
        specs=build_tensor_specs(args.start_pos, batch=args.batch),
        golden_fn=golden_indexer,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        compare_fn={
            "topk_indices": topk_indices_baseline_compare,
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
