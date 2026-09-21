# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B PyPTO Q/K norm, RoPE, cache append, and paged-attention pipeline."""

import math

import pypto.language as pl

from constants import QWEN3_14B as M

BATCH = 16
NUM_HEADS = 40
NUM_KV_HEADS = 8
GROUP = NUM_HEADS // NUM_KV_HEADS  # 5 query heads per KV head
HEAD_DIM = 128
BLOCK_SIZE = 128  # paged KV page size
STACK_PAGES = 4  # physical pages combined into one attention compute tile
STACK_TOKENS = STACK_PAGES * BLOCK_SIZE  # 512-token QK/softmax/PV granularity
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 1024 (BSND row width)
SEQ_LEN = 3584  # 7 * 512 — page & stack clean
MAX_BLOCKS = SEQ_LEN // BLOCK_SIZE  # 28
NUM_BLOCKS = BATCH * MAX_BLOCKS  # 448
NUM_TASKS = BATCH * NUM_KV_HEADS  # 128
ATTN_SPMD_BLOCKS = 24
PRE_LAUNCH = 2
TRANSFER_SLOTS = PRE_LAUNCH + 1
FFTS_WORKSPACE_ELEMENTS = 256
QK_READY_EVENT = 0
SOFTMAX_READY_EVENT = 1
PV_READY_EVENT = 2
ROW_TILE = 16
AIV_ROW_TILE = 8
AIV_LANE0_ROWS = 2
TRANSFER_ROWS = ATTN_SPMD_BLOCKS * TRANSFER_SLOTS * ROW_TILE
SCALE = 1.0 / math.sqrt(HEAD_DIM)
KCACHE_ROWS = NUM_BLOCKS * BLOCK_SIZE
HALF_DIM = HEAD_DIM // 2
HIDDEN = NUM_HEADS * HEAD_DIM
Q_HEAD_PAD = M.q_head_pad
HEAD_DIM_INV = M.head_dim_inv
EPS = M.eps
ROPE_CORES = 32
ROPE_ITEMS_PER_CORE = (NUM_KV_HEADS * BATCH + ROPE_CORES - 1) // ROPE_CORES
K_RED_ROWS = 8

assert NUM_HEADS % NUM_KV_HEADS == 0
assert BLOCK_SIZE == HEAD_DIM
assert STACK_TOKENS == 512
assert SEQ_LEN % BLOCK_SIZE == 0
assert (BATCH, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM) == (
    M.batch_pad,
    M.num_heads,
    M.num_kv_heads,
    M.head_dim,
)
assert GROUP == M.q_per_kv == M.q_head_batch
assert Q_HEAD_PAD == 16 and K_RED_ROWS == 8
assert ROPE_ITEMS_PER_CORE == 4


@pl.jit.inline(auto_scope=False)
def paged_attention_pypto_swpipe(  # noqa: PLR0913 -- fused Phase-0/attention API
    q_tnd_flat: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    block_table: pl.Tensor,
    seq_lens: pl.Tensor,
    inv_rms_states: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    q_proj: pl.Tensor,
    k_proj: pl.Tensor,
    v_proj: pl.Tensor,
    q_norm_w: pl.Tensor,
    k_norm_w: pl.Tensor,
    layer_cache_base_token_rows: pl.Scalar[pl.INDEX],
    out: pl.Tensor,
    score_transfer: pl.Tensor,
    probability_transfer: pl.Tensor,
    pv_transfer: pl.Tensor,
    ffts_workspace: pl.Tensor,
    q_proj_tid: pl.Scalar[pl.TASK_ID],
    k_proj_tid: pl.Scalar[pl.TASK_ID],
    v_proj_tid: pl.Scalar[pl.TASK_ID],
    rms_tid: pl.Scalar[pl.TASK_ID],
    attn_out_seed_tid: pl.Scalar[pl.TASK_ID],
    mlp_out_seed_tid: pl.Scalar[pl.TASK_ID],
    scratch_ready_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Run fused Q/K norm, RoPE, cache append, and dynamic paged attention."""
    active_batch = pl.tensor.dim(seq_lens, 0)
    num_tasks = active_batch * NUM_KV_HEADS
    query_rows = pl.tensor.dim(q_tnd_flat, 0)
    cache_token_rows = (pl.tensor.dim(key_cache, 0) * pl.tensor.dim(key_cache, 1)) // KV_HIDDEN
    key_cache_bsnd = pl.reshape(key_cache, [cache_token_rows, KV_HIDDEN])
    value_cache_bsnd = pl.reshape(value_cache, [cache_token_rows, KV_HIDDEN])
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // active_batch
    block_table_2d = pl.reshape(block_table, [active_batch, max_blocks_per_seq])
    q2d = pl.reshape(q_tnd_flat, [query_rows, HEAD_DIM])
    cache_base = layer_cache_base_token_rows
    out2d = pl.reshape(out, [query_rows, HEAD_DIM])
    with pl.spmd(
        ATTN_SPMD_BLOCKS,
        name_hint="attn_swpipe_spmd",
        sync_start=True,
        allow_early_resolve=True,
        deps=[
            q_proj_tid,
            k_proj_tid,
            v_proj_tid,
            rms_tid,
            attn_out_seed_tid,
            mlp_out_seed_tid,
            scratch_ready_tid,
        ],
    ) as attn_tid:
        core = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace)

        # Run Phase 0 on the first 32 physical AIV lanes, then publish Q/K/V.
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            rope_core = core * 2 + aiv_id
            if rope_core < ROPE_CORES:
                q_red_pad = pl.full(
                    [1, (Q_HEAD_PAD - GROUP) * HEAD_DIM],
                    dtype=pl.FP32,
                    value=0.0,
                )
                k_red_pad = pl.full(
                    [1, (K_RED_ROWS - 1) * HEAD_DIM],
                    dtype=pl.FP32,
                    value=0.0,
                )
                for it in pl.pipeline(ROPE_ITEMS_PER_CORE, stage=2):
                    g_idx = rope_core + it * ROPE_CORES
                    if g_idx < NUM_KV_HEADS * active_batch:
                        kv_head = g_idx // active_batch
                        batch_idx = g_idx - kv_head * active_batch
                        position = pl.read(seq_lens, [batch_idx]) - 1
                        inv_rms = pl.read(inv_rms_states, [batch_idx, 0])
                        # Serving uses -1 during its one-page profile warmup.
                        # Clamp the slot before adding the per-layer cache base;
                        # clamping only the final tensor-view offset would make
                        # layer N write the last row of layer N-1 instead.
                        write_slot = pl.max(
                            pl.cast(pl.tensor.read(slot_mapping, [batch_idx]), pl.INDEX),
                            0,
                        )
                        cos_lo = rope_cos[position : position + 1, 0:HALF_DIM]
                        cos_hi = rope_cos[position : position + 1, HALF_DIM:HEAD_DIM]
                        sin_lo = rope_sin[position : position + 1, 0:HALF_DIM]
                        sin_hi = rope_sin[position : position + 1, HALF_DIM:HEAD_DIM]

                        cache_col = kv_head * HEAD_DIM
                        k_raw = pl.mul(
                            pl.reshape(
                                pl.concat(
                                    k_proj[
                                        batch_idx : batch_idx + 1,
                                        cache_col : cache_col + HEAD_DIM,
                                    ],
                                    k_red_pad,
                                ),
                                [K_RED_ROWS, HEAD_DIM],
                            ),
                            inv_rms,
                        )
                        k_ss = pl.row_sum(pl.mul(k_raw, k_raw))
                        k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS)))
                        k_normed = pl.row_expand_mul(
                            pl.col_expand_mul(k_raw, k_norm_w),
                            k_inv,
                        )
                        k_full = k_normed[0:1, :]
                        k_lo = k_full[:, 0:HALF_DIM]
                        k_hi = k_full[:, HALF_DIM:HEAD_DIM]
                        k_rotated = pl.concat(
                            pl.sub(
                                pl.col_expand_mul(k_lo, cos_lo),
                                pl.col_expand_mul(k_hi, sin_lo),
                            ),
                            pl.add(
                                pl.col_expand_mul(k_hi, cos_hi),
                                pl.col_expand_mul(k_lo, sin_hi),
                            ),
                        )
                        cache_row = layer_cache_base_token_rows + write_slot
                        key_cache_bsnd[
                            cache_row : cache_row + 1,
                            cache_col : cache_col + HEAD_DIM,
                        ] = pl.cast(k_rotated, target_type=pl.BF16)

                        v_row_bf16 = pl.cast(
                            pl.mul(
                                v_proj[
                                    batch_idx : batch_idx + 1,
                                    cache_col : cache_col + HEAD_DIM,
                                ],
                                inv_rms,
                            ),
                            target_type=pl.BF16,
                        )
                        value_cache_bsnd[
                            cache_row : cache_row + 1,
                            cache_col : cache_col + HEAD_DIM,
                        ] = v_row_bf16

                        q_head_base = kv_head * GROUP
                        q_raw = pl.mul(
                            pl.reshape(
                                pl.concat(
                                    q_proj[
                                        batch_idx : batch_idx + 1,
                                        q_head_base * HEAD_DIM : (q_head_base + GROUP) * HEAD_DIM,
                                    ],
                                    q_red_pad,
                                ),
                                [Q_HEAD_PAD, HEAD_DIM],
                            ),
                            inv_rms,
                        )
                        q_ss = pl.row_sum(pl.mul(q_raw, q_raw))
                        q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
                        q_heads = pl.row_expand_mul(
                            pl.col_expand_mul(q_raw, q_norm_w),
                            q_inv,
                        )
                        q_lo = q_heads[:, 0:HALF_DIM]
                        q_hi = q_heads[:, HALF_DIM:HEAD_DIM]
                        q_rotated = pl.concat(
                            pl.sub(
                                pl.col_expand_mul(q_lo, cos_lo),
                                pl.col_expand_mul(q_hi, sin_lo),
                            ),
                            pl.add(
                                pl.col_expand_mul(q_hi, cos_hi),
                                pl.col_expand_mul(q_lo, sin_hi),
                            ),
                        )
                        q_row = batch_idx * NUM_HEADS + q_head_base
                        q2d[q_row : q_row + GROUP, :] = pl.cast(
                            q_rotated[0:GROUP, :],
                            target_type=pl.BF16,
                        )

        # SYNCALL is arrival-only. Publish the Phase-0 GM writes before the
        # barrier, then invalidate each consumer's GM cache before PA reads
        # Q/K/V.
        pl.system.cacheinvalid()
        pl.system.fence()
        pl.system.syncall(core_type=pl.KernelType.MIX)
        pl.system.cacheinvalid()

        for task in pl.range(core, num_tasks, ATTN_SPMD_BLOCKS):
            batch = task // NUM_KV_HEADS
            kv_head = task % NUM_KV_HEADS
            seq_len = pl.read(seq_lens, [batch])
            page_count = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            stack_count = (seq_len + STACK_TOKENS - 1) // STACK_TOKENS
            col = kv_head * HEAD_DIM
            transfer_base = core * TRANSFER_SLOTS * ROW_TILE
            qp_row = batch * NUM_HEADS + kv_head * GROUP
            q_tile = pl.load(
                q2d,
                [qp_row, 0],
                [ROW_TILE, HEAD_DIM],
                valid_shape=[GROUP, HEAD_DIM],
                target_memory=pl.MemorySpace.Mat,
            )

            # Keep 128-token cache pages inside each 512-token softmax/update stack.
            for tick in pl.range(stack_count + PRE_LAUNCH):
                if tick < stack_count:
                    produce_stack = tick
                    produce_page = produce_stack * STACK_PAGES
                    produce_row = transfer_base + (produce_stack % TRANSFER_SLOTS) * ROW_TILE
                    if produce_page + STACK_PAGES <= page_count:
                        for qk_page_offset in pl.pipeline(STACK_PAGES, stage=2):
                            qk_ph = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, produce_page + qk_page_offset]),
                                pl.INDEX,
                            )
                            qk_k_page = pl.load(
                                key_cache_bsnd,
                                [cache_base + qk_ph * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            qk_score_page = pl.matmul(
                                q_tile,
                                pl.tile.transpose_view(qk_k_page),
                                out_dtype=pl.FP32,
                            )
                            pl.store(
                                qk_score_page,
                                [produce_row, qk_page_offset * BLOCK_SIZE],
                                score_transfer,
                            )
                    else:
                        qk_tail_ph0 = pl.cast(
                            pl.tensor.read(block_table_2d, [batch, produce_page]),
                            pl.INDEX,
                        )
                        k_tail0 = pl.load(
                            key_cache_bsnd,
                            [cache_base + qk_tail_ph0 * BLOCK_SIZE, col],
                            [BLOCK_SIZE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        qk_tail0 = pl.matmul(
                            q_tile,
                            pl.tile.transpose_view(k_tail0),
                            out_dtype=pl.FP32,
                        )
                        pl.store(qk_tail0, [produce_row, 0], score_transfer)
                        if produce_page + 1 < page_count:
                            qk_tail_ph1 = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, produce_page + 1]),
                                pl.INDEX,
                            )
                            k_tail1 = pl.load(
                                key_cache_bsnd,
                                [cache_base + qk_tail_ph1 * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            qk_tail1 = pl.matmul(
                                q_tile,
                                pl.tile.transpose_view(k_tail1),
                                out_dtype=pl.FP32,
                            )
                            pl.store(qk_tail1, [produce_row, BLOCK_SIZE], score_transfer)
                        if produce_page + 2 < page_count:
                            qk_tail_ph2 = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, produce_page + 2]),
                                pl.INDEX,
                            )
                            k_tail2 = pl.load(
                                key_cache_bsnd,
                                [cache_base + qk_tail_ph2 * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            qk_tail2 = pl.matmul(
                                q_tile,
                                pl.tile.transpose_view(k_tail2),
                                out_dtype=pl.FP32,
                            )
                            pl.store(qk_tail2, [produce_row, 2 * BLOCK_SIZE], score_transfer)
                    pl.system.sync_set(
                        QK_READY_EVENT,
                        pipe=pl.PipeType.FIX,
                        ffts_mode=2,
                        core_type=pl.KernelType.AIC,
                    )

                if tick >= PRE_LAUNCH:
                    consume_stack = tick - PRE_LAUNCH
                    consume_page = consume_stack * STACK_PAGES
                    consume_row = transfer_base + (consume_stack % TRANSFER_SLOTS) * ROW_TILE
                    # Prefetch the complete valid V stack, then ping-pong P into one Acc tile.
                    if consume_page + STACK_PAGES <= page_count:
                        pv_ph0 = pl.cast(
                            pl.tensor.read(block_table_2d, [batch, consume_page]),
                            pl.INDEX,
                        )
                        pv_ph1 = pl.cast(
                            pl.tensor.read(block_table_2d, [batch, consume_page + 1]),
                            pl.INDEX,
                        )
                        pv_ph2 = pl.cast(
                            pl.tensor.read(block_table_2d, [batch, consume_page + 2]),
                            pl.INDEX,
                        )
                        pv_ph3 = pl.cast(
                            pl.tensor.read(block_table_2d, [batch, consume_page + 3]),
                            pl.INDEX,
                        )
                        v0: pl.Tile[
                            [BLOCK_SIZE, HEAD_DIM],
                            pl.BF16,
                            pl.MemRef("pv_v_l1", slots=STACK_PAGES)[0],
                            pl.Mem.Mat,
                        ] = pl.load(
                            value_cache_bsnd,
                            [cache_base + pv_ph0 * BLOCK_SIZE, col],
                            [BLOCK_SIZE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        v1: pl.Tile[
                            [BLOCK_SIZE, HEAD_DIM],
                            pl.BF16,
                            pl.MemRef("pv_v_l1", slots=STACK_PAGES)[1],
                            pl.Mem.Mat,
                        ] = pl.load(
                            value_cache_bsnd,
                            [cache_base + pv_ph1 * BLOCK_SIZE, col],
                            [BLOCK_SIZE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        v2: pl.Tile[
                            [BLOCK_SIZE, HEAD_DIM],
                            pl.BF16,
                            pl.MemRef("pv_v_l1", slots=STACK_PAGES)[2],
                            pl.Mem.Mat,
                        ] = pl.load(
                            value_cache_bsnd,
                            [cache_base + pv_ph2 * BLOCK_SIZE, col],
                            [BLOCK_SIZE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        v3: pl.Tile[
                            [BLOCK_SIZE, HEAD_DIM],
                            pl.BF16,
                            pl.MemRef("pv_v_l1", slots=STACK_PAGES)[3],
                            pl.Mem.Mat,
                        ] = pl.load(
                            value_cache_bsnd,
                            [cache_base + pv_ph3 * BLOCK_SIZE, col],
                            [BLOCK_SIZE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pl.system.sync_wait(
                            SOFTMAX_READY_EVENT,
                            pipe=pl.PipeType.MTE2,
                            core_type=pl.KernelType.AIC,
                        )
                        probability0: pl.Tile[
                            [ROW_TILE, BLOCK_SIZE],
                            pl.BF16,
                            pl.MemRef("pv_p_l1", slots=2)[0],
                            pl.Mem.Mat,
                        ] = pl.load(
                            probability_transfer,
                            [consume_row, 0],
                            [ROW_TILE, BLOCK_SIZE],
                            valid_shape=[GROUP, BLOCK_SIZE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv0 = pl.matmul(
                            pl.tile.move(probability0, target_memory=pl.MemorySpace.Left),
                            pl.tile.move(v0, target_memory=pl.MemorySpace.Right),
                            out_dtype=pl.FP32,
                        )
                        probability1: pl.Tile[
                            [ROW_TILE, BLOCK_SIZE],
                            pl.BF16,
                            pl.MemRef("pv_p_l1", slots=2)[1],
                            pl.Mem.Mat,
                        ] = pl.load(
                            probability_transfer,
                            [consume_row, BLOCK_SIZE],
                            [ROW_TILE, BLOCK_SIZE],
                            valid_shape=[GROUP, BLOCK_SIZE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv1 = pl.matmul_acc(
                            pv0,
                            pl.tile.move(probability1, target_memory=pl.MemorySpace.Left),
                            pl.tile.move(v1, target_memory=pl.MemorySpace.Right),
                        )
                        probability2: pl.Tile[
                            [ROW_TILE, BLOCK_SIZE],
                            pl.BF16,
                            pl.MemRef("pv_p_l1", slots=2)[0],
                            pl.Mem.Mat,
                        ] = pl.load(
                            probability_transfer,
                            [consume_row, 2 * BLOCK_SIZE],
                            [ROW_TILE, BLOCK_SIZE],
                            valid_shape=[GROUP, BLOCK_SIZE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv2 = pl.matmul_acc(
                            pv1,
                            pl.tile.move(probability2, target_memory=pl.MemorySpace.Left),
                            pl.tile.move(v2, target_memory=pl.MemorySpace.Right),
                        )
                        probability3: pl.Tile[
                            [ROW_TILE, BLOCK_SIZE],
                            pl.BF16,
                            pl.MemRef("pv_p_l1", slots=2)[1],
                            pl.Mem.Mat,
                        ] = pl.load(
                            probability_transfer,
                            [consume_row, 3 * BLOCK_SIZE],
                            [ROW_TILE, BLOCK_SIZE],
                            valid_shape=[GROUP, BLOCK_SIZE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv3 = pl.matmul_acc(
                            pv2,
                            pl.tile.move(probability3, target_memory=pl.MemorySpace.Left),
                            pl.tile.move(v3, target_memory=pl.MemorySpace.Right),
                        )
                        pl.store(pv3, [consume_row, 0], pv_transfer)
                    else:
                        tail_pages = page_count - consume_page
                        if tail_pages == 1:
                            pv1_ph0 = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, consume_page]),
                                pl.INDEX,
                            )
                            v1_0: pl.Tile[
                                [BLOCK_SIZE, HEAD_DIM],
                                pl.BF16,
                                pl.MemRef("pv_v_l1", slots=STACK_PAGES)[0],
                                pl.Mem.Mat,
                            ] = pl.load(
                                value_cache_bsnd,
                                [cache_base + pv1_ph0 * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pl.system.sync_wait(
                                SOFTMAX_READY_EVENT,
                                pipe=pl.PipeType.MTE2,
                                core_type=pl.KernelType.AIC,
                            )
                            probability1_0: pl.Tile[
                                [ROW_TILE, BLOCK_SIZE],
                                pl.BF16,
                                pl.MemRef("pv_p_l1", slots=2)[0],
                                pl.Mem.Mat,
                            ] = pl.load(
                                probability_transfer,
                                [consume_row, 0],
                                [ROW_TILE, BLOCK_SIZE],
                                valid_shape=[GROUP, BLOCK_SIZE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pv_tail1 = pl.matmul(
                                pl.tile.move(probability1_0, target_memory=pl.MemorySpace.Left),
                                pl.tile.move(v1_0, target_memory=pl.MemorySpace.Right),
                                out_dtype=pl.FP32,
                            )
                            pl.store(pv_tail1, [consume_row, 0], pv_transfer)
                        if tail_pages == 2:
                            pv2_ph0 = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, consume_page]),
                                pl.INDEX,
                            )
                            pv2_ph1 = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, consume_page + 1]),
                                pl.INDEX,
                            )
                            v2_0: pl.Tile[
                                [BLOCK_SIZE, HEAD_DIM],
                                pl.BF16,
                                pl.MemRef("pv_v_l1", slots=STACK_PAGES)[0],
                                pl.Mem.Mat,
                            ] = pl.load(
                                value_cache_bsnd,
                                [cache_base + pv2_ph0 * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            v2_1: pl.Tile[
                                [BLOCK_SIZE, HEAD_DIM],
                                pl.BF16,
                                pl.MemRef("pv_v_l1", slots=STACK_PAGES)[1],
                                pl.Mem.Mat,
                            ] = pl.load(
                                value_cache_bsnd,
                                [cache_base + pv2_ph1 * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pl.system.sync_wait(
                                SOFTMAX_READY_EVENT,
                                pipe=pl.PipeType.MTE2,
                                core_type=pl.KernelType.AIC,
                            )
                            probability2_0: pl.Tile[
                                [ROW_TILE, BLOCK_SIZE],
                                pl.BF16,
                                pl.MemRef("pv_p_l1", slots=2)[0],
                                pl.Mem.Mat,
                            ] = pl.load(
                                probability_transfer,
                                [consume_row, 0],
                                [ROW_TILE, BLOCK_SIZE],
                                valid_shape=[GROUP, BLOCK_SIZE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pv_tail2_0 = pl.matmul(
                                pl.tile.move(probability2_0, target_memory=pl.MemorySpace.Left),
                                pl.tile.move(v2_0, target_memory=pl.MemorySpace.Right),
                                out_dtype=pl.FP32,
                            )
                            probability2_1: pl.Tile[
                                [ROW_TILE, BLOCK_SIZE],
                                pl.BF16,
                                pl.MemRef("pv_p_l1", slots=2)[1],
                                pl.Mem.Mat,
                            ] = pl.load(
                                probability_transfer,
                                [consume_row, BLOCK_SIZE],
                                [ROW_TILE, BLOCK_SIZE],
                                valid_shape=[GROUP, BLOCK_SIZE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pv_tail2_1 = pl.matmul_acc(
                                pv_tail2_0,
                                pl.tile.move(probability2_1, target_memory=pl.MemorySpace.Left),
                                pl.tile.move(v2_1, target_memory=pl.MemorySpace.Right),
                            )
                            pl.store(pv_tail2_1, [consume_row, 0], pv_transfer)
                        if tail_pages == 3:
                            pv3_ph0 = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, consume_page]),
                                pl.INDEX,
                            )
                            pv3_ph1 = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, consume_page + 1]),
                                pl.INDEX,
                            )
                            pv3_ph2 = pl.cast(
                                pl.tensor.read(block_table_2d, [batch, consume_page + 2]),
                                pl.INDEX,
                            )
                            v3_0: pl.Tile[
                                [BLOCK_SIZE, HEAD_DIM],
                                pl.BF16,
                                pl.MemRef("pv_v_l1", slots=STACK_PAGES)[0],
                                pl.Mem.Mat,
                            ] = pl.load(
                                value_cache_bsnd,
                                [cache_base + pv3_ph0 * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            v3_1: pl.Tile[
                                [BLOCK_SIZE, HEAD_DIM],
                                pl.BF16,
                                pl.MemRef("pv_v_l1", slots=STACK_PAGES)[1],
                                pl.Mem.Mat,
                            ] = pl.load(
                                value_cache_bsnd,
                                [cache_base + pv3_ph1 * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            v3_2: pl.Tile[
                                [BLOCK_SIZE, HEAD_DIM],
                                pl.BF16,
                                pl.MemRef("pv_v_l1", slots=STACK_PAGES)[2],
                                pl.Mem.Mat,
                            ] = pl.load(
                                value_cache_bsnd,
                                [cache_base + pv3_ph2 * BLOCK_SIZE, col],
                                [BLOCK_SIZE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pl.system.sync_wait(
                                SOFTMAX_READY_EVENT,
                                pipe=pl.PipeType.MTE2,
                                core_type=pl.KernelType.AIC,
                            )
                            probability3_0: pl.Tile[
                                [ROW_TILE, BLOCK_SIZE],
                                pl.BF16,
                                pl.MemRef("pv_p_l1", slots=2)[0],
                                pl.Mem.Mat,
                            ] = pl.load(
                                probability_transfer,
                                [consume_row, 0],
                                [ROW_TILE, BLOCK_SIZE],
                                valid_shape=[GROUP, BLOCK_SIZE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pv_tail3_0 = pl.matmul(
                                pl.tile.move(probability3_0, target_memory=pl.MemorySpace.Left),
                                pl.tile.move(v3_0, target_memory=pl.MemorySpace.Right),
                                out_dtype=pl.FP32,
                            )
                            probability3_1: pl.Tile[
                                [ROW_TILE, BLOCK_SIZE],
                                pl.BF16,
                                pl.MemRef("pv_p_l1", slots=2)[1],
                                pl.Mem.Mat,
                            ] = pl.load(
                                probability_transfer,
                                [consume_row, BLOCK_SIZE],
                                [ROW_TILE, BLOCK_SIZE],
                                valid_shape=[GROUP, BLOCK_SIZE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pv_tail3_1 = pl.matmul_acc(
                                pv_tail3_0,
                                pl.tile.move(probability3_1, target_memory=pl.MemorySpace.Left),
                                pl.tile.move(v3_1, target_memory=pl.MemorySpace.Right),
                            )
                            probability3_2: pl.Tile[
                                [ROW_TILE, BLOCK_SIZE],
                                pl.BF16,
                                pl.MemRef("pv_p_l1", slots=2)[0],
                                pl.Mem.Mat,
                            ] = pl.load(
                                probability_transfer,
                                [consume_row, 2 * BLOCK_SIZE],
                                [ROW_TILE, BLOCK_SIZE],
                                valid_shape=[GROUP, BLOCK_SIZE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pv_tail3_2 = pl.matmul_acc(
                                pv_tail3_1,
                                pl.tile.move(probability3_2, target_memory=pl.MemorySpace.Left),
                                pl.tile.move(v3_2, target_memory=pl.MemorySpace.Right),
                            )
                            pl.store(pv_tail3_2, [consume_row, 0], pv_transfer)
                    pl.system.sync_set(
                        PV_READY_EVENT,
                        pipe=pl.PipeType.FIX,
                        ffts_mode=2,
                        core_type=pl.KernelType.AIC,
                    )

            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts_workspace)
                lane_row = aiv_id * AIV_LANE0_ROWS
                lane_rows = AIV_LANE0_ROWS + aiv_id
                tmp = pl.create_tile(
                    [AIV_ROW_TILE, STACK_TOKENS],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                zero_scores = pl.tile.full(
                    [AIV_ROW_TILE, STACK_TOKENS],
                    dtype=pl.FP32,
                    value=0.0,
                )
                zero_scores_valid = pl.set_validshape(
                    zero_scores,
                    lane_rows,
                    STACK_TOKENS,
                )
                zrow = pl.tile.muls(pl.row_max(zero_scores_valid, tmp), 0.0)
                m = pl.tile.adds(zrow, -3.0e38)
                l_sum = zrow
                o_init = pl.tile.full(
                    [AIV_ROW_TILE, HEAD_DIM],
                    dtype=pl.FP32,
                    value=0.0,
                )
                o = pl.set_validshape(o_init, lane_rows, HEAD_DIM)
                # Rescale FIFO states: (1, 1) -> (1, r0) -> (r0, r1).
                rescale_pending0 = pl.tile.adds(zrow, 1.0)
                rescale_pending1 = pl.tile.adds(zrow, 1.0)

                for tick, (m_iter, l_iter, o_iter, pending0_iter, pending1_iter) in pl.range(
                    stack_count + PRE_LAUNCH,
                    init_values=(m, l_sum, o, rescale_pending0, rescale_pending1),
                ):
                    if tick < stack_count:
                        produce_stack = tick
                        produce_row = transfer_base + (produce_stack % TRANSFER_SLOTS) * ROW_TILE
                        pl.system.sync_wait(
                            QK_READY_EVENT,
                            pipe=pl.PipeType.MTE2,
                            core_type=pl.KernelType.AIV,
                        )
                        score_aiv = pl.load(
                            score_transfer,
                            [produce_row + lane_row, 0],
                            [AIV_ROW_TILE, STACK_TOKENS],
                            valid_shape=[lane_rows, STACK_TOKENS],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        score_scaled = pl.tile.muls(score_aiv, SCALE)
                        valid_cols = pl.min(
                            STACK_TOKENS,
                            seq_len - produce_stack * STACK_TOKENS,
                        )
                        score_valid = pl.set_validshape(score_scaled, lane_rows, valid_cols)
                        score_filled = pl.fillpad(score_valid, pad_value=pl.PadValue.min)
                        score_masked = pl.set_validshape(
                            score_filled,
                            lane_rows,
                            STACK_TOKENS,
                        )
                        local_m = pl.row_max(score_masked, tmp)
                        next_m = pl.maximum(local_m, m_iter)
                        rescale = pl.exp(pl.sub(m_iter, next_m))
                        probability_aiv = pl.exp(pl.row_expand_sub(score_masked, next_m))
                        probability_reduce = pl.tile.move(
                            probability_aiv,
                            target_memory=pl.MemorySpace.Vec,
                            slayout=pl.TileLayout.none_box,
                        )
                        next_l = pl.add(
                            pl.mul(rescale, l_iter),
                            pl.row_sum(probability_reduce, tmp),
                        )
                        probability_bf16 = pl.cast(
                            probability_aiv,
                            target_type=pl.BF16,
                            mode="rint",
                        )
                        probability_valid = pl.set_validshape(
                            probability_bf16,
                            lane_rows,
                            STACK_TOKENS,
                        )
                        pl.store(
                            probability_valid,
                            [produce_row + lane_row, 0],
                            probability_transfer,
                        )
                        pl.system.sync_set(
                            SOFTMAX_READY_EVENT,
                            pipe=pl.PipeType.MTE3,
                            ffts_mode=2,
                            core_type=pl.KernelType.AIV,
                        )
                        m_after, l_after, rescale_after = pl.yield_(next_m, next_l, rescale)
                    else:
                        m_after, l_after, rescale_after = pl.yield_(m_iter, l_iter, pending1_iter)

                    if tick >= PRE_LAUNCH:
                        consume_stack = tick - PRE_LAUNCH
                        consume_row = transfer_base + (consume_stack % TRANSFER_SLOTS) * ROW_TILE
                        pl.system.sync_wait(
                            PV_READY_EVENT,
                            pipe=pl.PipeType.MTE2,
                            core_type=pl.KernelType.AIV,
                        )
                        pv_aiv = pl.load(
                            pv_transfer,
                            [consume_row + lane_row, 0],
                            [AIV_ROW_TILE, HEAD_DIM],
                            valid_shape=[lane_rows, HEAD_DIM],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        o_after = pl.yield_(
                            pl.add(pl.row_expand_mul(o_iter, pending0_iter), pv_aiv),
                        )
                    else:
                        o_after = pl.yield_(o_iter)

                    pending0_after = pl.tile.move(
                        pending1_iter,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    pending1_after = pl.tile.move(
                        rescale_after,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    m, l_sum, o, rescale_pending0, rescale_pending1 = pl.yield_(
                        m_after,
                        l_after,
                        o_after,
                        pending0_after,
                        pending1_after,
                    )

                ctx = pl.cast(
                    pl.row_expand_mul(o, pl.recip(l_sum)),
                    target_type=pl.BF16,
                    mode="rint",
                )
                pl.store(
                    ctx,
                    [batch * NUM_HEADS + kv_head * GROUP + lane_row, 0],
                    out2d,
                )
    return attn_tid
