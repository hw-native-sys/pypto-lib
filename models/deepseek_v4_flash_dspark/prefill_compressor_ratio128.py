# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill compressor for the ratio-128 state cache."""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    FLASH as M,
    HCA_STATE_PHYSICAL_BLOCKS,
    PREFILL_SEQ,
)
from prefill_metadata import QUERY_START_LOC_DYN, REQUESTS_DYN



# Bounded physical-row tile for ratio-128 projection/state updates.
PREFILL_STATE_TILE = 512


# Dynamic shape variables.
T_DYN = pl.dynamic("PREFILL_HCA_C128_T_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")

# model config
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = M.max_position_embeddings
START_POS = 0
COMPRESS_RATIO = 128
OUT_DIM = HEAD_DIM
STATE_LEN = COMPRESS_RATIO
COMPRESS_STATE_DIM = 2 * OUT_DIM
MAX_CMP_WRITES = PREFILL_STATE_TILE // COMPRESS_RATIO
CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE

# paged compressor state
HCA_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = HCA_STATE_PHYSICAL_BLOCKS

# tiling
K_TILE = 512  # projection D (K) reduction tile
OUT_TILE = 32  # projection OUT_DIM (N) tile
PROJ_ROW_TILE = 128  # projection token-row tile; Acc = ROW*OUT_TILE*4 sits under the a2a3 L0C wall
PROJ_TAIL_ROW_TILE = 16  # reliable partial-M cube shape for the final incomplete projection block
PROJ_FULL_ROW_BLOCKS = PREFILL_STATE_TILE // PROJ_ROW_TILE
PROJ_TAIL_ROW_BLOCKS = PROJ_ROW_TILE // PROJ_TAIL_ROW_TILE
HEAD_TILE = 64  # head-dim tile for the pool and rmsnorm
HCA_KV_STORE_TILE = 16
HCA_C128_RMS_TILE = 8  # rmsnorm/rope write-row tile
HCA_C128_RMS_PAD_ROWS = ((MAX_CMP_WRITES + HCA_C128_RMS_TILE - 1) // HCA_C128_RMS_TILE) * HCA_C128_RMS_TILE

assert PREFILL_STATE_TILE % PROJ_ROW_TILE == 0
assert PREFILL_STATE_TILE % COMPRESS_RATIO == 0


@pl.jit.inline(auto_scope=False)
def _prefill_compressor_ratio128_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    rope_dup_idx_template: pl.Tensor[[HCA_C128_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_swap_idx_template: pl.Tensor[[HCA_C128_RMS_TILE, ROPE_HEAD_DIM], pl.INT32],
    rope_sign_template: pl.Tensor[[HCA_C128_RMS_TILE, ROPE_HEAD_DIM], pl.FP32],
    state_order_fence: pl.InOut[pl.Tensor[[1], pl.INT32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    state_block_num = pl.tensor.dim(compress_state, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    kv_proj_scratch = pl.create_tensor([PREFILL_STATE_TILE, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([PREFILL_STATE_TILE, OUT_DIM], dtype=pl.FP32)
    compress_state_flat = pl.reshape(
        compress_state,
        [state_block_num * HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
    )
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])

    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, D])
    proj_full_rows = (tile_rows // PROJ_ROW_TILE) * PROJ_ROW_TILE
    proj_padded_rows = ((tile_rows + PROJ_TAIL_ROW_TILE - 1) // PROJ_TAIL_ROW_TILE) * PROJ_TAIL_ROW_TILE

    if proj_full_rows > 0:
        for proj_idx in pl.spmd(
            (OUT_DIM // OUT_TILE) * PROJ_FULL_ROW_BLOCKS,
            name_hint="prefill_hca_c128_kv_score_proj_full",
        ):
            proj_n = proj_idx // PROJ_FULL_ROW_BLOCKS
            proj_row_block = proj_idx - proj_n * PROJ_FULL_ROW_BLOCKS
            o0 = proj_n * OUT_TILE
            local_t0 = proj_row_block * PROJ_ROW_TILE
            if local_t0 < proj_full_rows:
                global_t0 = tile_base + local_t0
                kv_acc = pl.create_tensor([PROJ_ROW_TILE, OUT_TILE], dtype=pl.FP32)
                score_acc = pl.create_tensor([PROJ_ROW_TILE, OUT_TILE], dtype=pl.FP32)
                for kb in pl.pipeline(0, D // K_TILE, stage=2):
                    k0 = kb * K_TILE
                    x_tile = x_flat[
                        global_t0 : global_t0 + PROJ_ROW_TILE,
                        k0 : k0 + K_TILE,
                    ]
                    # Weights stored transposed [OUT_DIM, D] + b_trans=True -> DN2ZN
                    # (K-contiguous) load.
                    wkv_tile = wkv[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                    wgate_tile = wgate[o0 : o0 + OUT_TILE, k0 : k0 + K_TILE]
                    if k0 == 0:
                        kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32, b_trans=True)
                        score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32, b_trans=True)
                    else:
                        kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True)
                        score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True)
                kv_proj_scratch[
                    local_t0 : local_t0 + PROJ_ROW_TILE,
                    o0 : o0 + OUT_TILE,
                ] = kv_acc
                score_proj_scratch[
                    local_t0 : local_t0 + PROJ_ROW_TILE,
                    o0 : o0 + OUT_TILE,
                ] = score_acc

    # The final incomplete 128-row block is padded through the proven 16-row
    # partial-M shape. Only physical rows are consumed below.
    if proj_full_rows < proj_padded_rows:
        for tail_idx in pl.spmd(
            (OUT_DIM // OUT_TILE) * PROJ_TAIL_ROW_BLOCKS,
            name_hint="prefill_hca_c128_kv_score_proj_tail",
        ):
            tail_n = tail_idx // PROJ_TAIL_ROW_BLOCKS
            tail_row_block = tail_idx - tail_n * PROJ_TAIL_ROW_BLOCKS
            tail_o0 = tail_n * OUT_TILE
            tail_t0 = proj_full_rows + tail_row_block * PROJ_TAIL_ROW_TILE
            if tail_t0 < proj_padded_rows:
                tail_valid_rows = pl.min(PROJ_TAIL_ROW_TILE, tile_rows - tail_t0)
                tail_global_t0 = tile_base + tail_t0
                kv_tail_acc = pl.create_tensor([PROJ_TAIL_ROW_TILE, OUT_TILE], dtype=pl.FP32)
                score_tail_acc = pl.create_tensor([PROJ_TAIL_ROW_TILE, OUT_TILE], dtype=pl.FP32)
                for tail_kb in pl.pipeline(0, D // K_TILE, stage=2):
                    tail_k0 = tail_kb * K_TILE
                    x_tail = pl.slice(
                        x_flat,
                        [PROJ_TAIL_ROW_TILE, K_TILE],
                        [tail_global_t0, tail_k0],
                        valid_shape=[tail_valid_rows, K_TILE],
                    )
                    wkv_tail = wkv[
                        tail_o0 : tail_o0 + OUT_TILE,
                        tail_k0 : tail_k0 + K_TILE,
                    ]
                    wgate_tail = wgate[
                        tail_o0 : tail_o0 + OUT_TILE,
                        tail_k0 : tail_k0 + K_TILE,
                    ]
                    if tail_k0 == 0:
                        kv_tail_acc = pl.matmul(x_tail, wkv_tail, out_dtype=pl.FP32, b_trans=True)
                        score_tail_acc = pl.matmul(
                            x_tail,
                            wgate_tail,
                            out_dtype=pl.FP32,
                            b_trans=True,
                        )
                    else:
                        kv_tail_acc = pl.matmul_acc(kv_tail_acc, x_tail, wkv_tail, b_trans=True)
                        score_tail_acc = pl.matmul_acc(
                            score_tail_acc,
                            x_tail,
                            wgate_tail,
                            b_trans=True,
                        )
                kv_proj_scratch[
                    tail_t0 : tail_t0 + PROJ_TAIL_ROW_TILE,
                    tail_o0 : tail_o0 + OUT_TILE,
                ] = kv_tail_acc
                score_proj_scratch[
                    tail_t0 : tail_t0 + PROJ_TAIL_ROW_TILE,
                    tail_o0 : tail_o0 + OUT_TILE,
                ] = score_tail_acc

    # write_i -> (position, dst cache row) map, consumed by pool / rmsnorm_rope / kv_finalize.
    # Sized to HCA_C128_RMS_PAD_ROWS: rmsnorm_rope indexes padded rows past MAX_CMP_WRITES, which stay -1.
    write_pos_map = pl.create_tensor([1, HCA_C128_RMS_PAD_ROWS], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, HCA_C128_RMS_PAD_ROWS], dtype=pl.INT32)
    write_src_map = pl.create_tensor([1, HCA_C128_RMS_PAD_ROWS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_write_map"):
        write_pos_map[0:1, 0:HCA_C128_RMS_PAD_ROWS] = pl.full(
            [1, HCA_C128_RMS_PAD_ROWS], dtype=pl.INT32, value=0
        )
        write_dst_map[0:1, 0:HCA_C128_RMS_PAD_ROWS] = pl.full(
            [1, HCA_C128_RMS_PAD_ROWS], dtype=pl.INT32, value=-1
        )
        write_src_map[0:1, 0:HCA_C128_RMS_PAD_ROWS] = pl.full([1, HCA_C128_RMS_PAD_ROWS], dtype=pl.INT32, value=-1)
        map_seen = pl.cast(0, pl.INDEX)
        for map_local in pl.range(tile_rows):
            map_global = tile_base + map_local
            map_slot_raw = pl.read(cmp_slot_mapping, [map_global])
            if map_slot_raw >= 0:
                pl.write(write_pos_map, [0, map_seen], pl.read(position_ids, [map_global]))
                pl.write(write_dst_map, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                pl.write(write_src_map, [0, map_seen], pl.cast(map_global, pl.INT32))
                map_seen = map_seen + 1

    # State scatter: every token's raw projection (+APE on score) into paged kv_state/score_state.
    # Must land before softmax_pool, which reads its window straight from state; the RAW edge on
    # compress_state is what carries that order.
    with pl.spmd(
        tile_rows,
        name_hint="prefill_hca_c128_state_scatter_pre",
    ) as _scatter_tid:
        scatter_local = pl.tile.get_block_idx()
        scatter_order = pl.read(state_order_fence, [0])
        if scatter_order >= 0:
            scatter_global = tile_base + scatter_local
            scatter_row_raw = pl.read(state_slot_mapping, [scatter_global])
            if scatter_row_raw >= 0:
                scatter_row = pl.cast(scatter_row_raw, pl.INDEX)
                scatter_pos = pl.read(position_ids, [scatter_global])
                scatter_ape_slot = pl.cast(scatter_pos % COMPRESS_RATIO, pl.INDEX)
                compress_state_flat[scatter_row : scatter_row + 1, 0:OUT_DIM] = kv_proj_scratch[
                    scatter_local : scatter_local + 1,
                    0:OUT_DIM,
                ]
                scatter_score = score_proj_scratch[
                    scatter_local : scatter_local + 1,
                    0:OUT_DIM,
                ]
                scatter_ape = ape[scatter_ape_slot : scatter_ape_slot + 1, 0:OUT_DIM]
                compress_state_flat[
                    scatter_row : scatter_row + 1,
                    OUT_DIM:COMPRESS_STATE_DIM,
                ] = pl.add(scatter_score, scatter_ape)

    pooled_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    for pool_idx in pl.spmd(
        MAX_CMP_WRITES * (HEAD_DIM // HEAD_TILE), name_hint="prefill_hca_c128_softmax_pool"
    ):
        write_i = pool_idx // (HEAD_DIM // HEAD_TILE)
        hb = pool_idx - write_i * (HEAD_DIM // HEAD_TILE)
        h0 = hb * HEAD_TILE
        pool_kv_tile = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
        pool_score_tile = pl.create_tensor([STATE_LEN, HEAD_TILE], dtype=pl.FP32)
        write_slot_raw = pl.read(write_dst_map, [0, write_i])
        if write_slot_raw >= 0:
            write_pos = pl.read(write_pos_map, [0, write_i])
            for pool_state_i in pl.range(STATE_LEN):
                pool_kv_tile[pool_state_i : pool_state_i + 1, 0:HEAD_TILE] = pl.full(
                    [1, HEAD_TILE],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_score_tile[pool_state_i : pool_state_i + 1, 0:HEAD_TILE] = pl.full(
                    [1, HEAD_TILE],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_abs = write_pos + 1 - COMPRESS_RATIO + pool_state_i
                pool_state_block = pl.cast(pool_abs // HCA_STATE_BLOCK_SIZE, pl.INDEX)
                pool_state_intra = pl.cast(pool_abs - pool_state_block * HCA_STATE_BLOCK_SIZE, pl.INDEX)
                pool_phys_block_raw = pl.read(compress_state_block_table, [pool_state_block])
                if pool_phys_block_raw >= 0:
                    pool_phys_block = pl.cast(pool_phys_block_raw, pl.INDEX)
                    pool_state_row = pool_phys_block * HCA_STATE_BLOCK_SIZE + pool_state_intra
                    pool_kv_tile[pool_state_i : pool_state_i + 1, 0:HEAD_TILE] = compress_state_flat[
                        pool_state_row : pool_state_row + 1,
                        h0 : h0 + HEAD_TILE,
                    ]
                    pool_score_tile[pool_state_i : pool_state_i + 1, 0:HEAD_TILE] = compress_state_flat[
                        pool_state_row : pool_state_row + 1,
                        OUT_DIM + h0 : OUT_DIM + h0 + HEAD_TILE,
                    ]
            # Vectorized softmax over all STATE_LEN slots: transpose the assembled
            # [STATE_LEN, HEAD_TILE] tile, then row_max/exp/sum/div and the weighted sum.
            pool_score_t = pl.transpose(pool_score_tile, axis1=0, axis2=1)
            pool_kv_t = pl.transpose(pool_kv_tile, axis1=0, axis2=1)
            score_max = pl.row_max(pool_score_t)
            score_exp = pl.exp(pl.row_expand_sub(pool_score_t, score_max))
            score_sum = pl.row_sum(score_exp)
            score_prob = pl.row_expand_div(score_exp, score_sum)
            pooled_chunk_t = pl.row_sum(pl.mul(pool_kv_t, score_prob))
            pooled_kv_pad[write_i : write_i + 1, h0 : h0 + HEAD_TILE] = pl.reshape(
                pooled_chunk_t, [1, HEAD_TILE]
            )
        else:
            pooled_zero = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
            pooled_kv_pad[write_i : write_i + 1, h0 : h0 + HEAD_TILE] = pooled_zero

    # Publish completion through a one-output task.  Reading the pool result
    # makes this task wait for the whole 32-block pool dispatch; writing the
    # fence registers the producer that the next tile's scatter reads.  A
    # separate task avoids the orchestration ambiguity of a mixed kernel with
    # two InOut returns (pooled_kv_pad plus the fence).
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_state_order_commit"):
        pool_sample = pl.read(pooled_kv_pad, [0, 0])
        fence_bit = pl.cast(pool_sample == pool_sample, pl.INT32)
        pl.write(state_order_fence, [0], fence_bit * fence_bit)

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    normed_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    for rms_blk in pl.spmd(
        HCA_C128_RMS_PAD_ROWS // HCA_C128_RMS_TILE, name_hint="prefill_hca_c128_rmsnorm_rope"
    ):
        r0 = rms_blk * HCA_C128_RMS_TILE
        cos_b = pl.full([HCA_C128_RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([HCA_C128_RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        for norm_i in pl.range(HCA_C128_RMS_TILE):
            norm_slot_raw = pl.read(write_dst_map, [0, r0 + norm_i])
            if norm_slot_raw >= 0:
                write_src = pl.cast(pl.read(write_src_map, [0, r0 + norm_i]), pl.INDEX)
                cos_row = pl.cast(
                    cmp_freqs_cos[write_src : write_src + 1, 0:ROPE_HALF], target_type=pl.FP32
                )
                sin_row = pl.cast(
                    cmp_freqs_sin[write_src : write_src + 1, 0:ROPE_HALF], target_type=pl.FP32
                )
                cos_b[norm_i : norm_i + 1, 0:ROPE_HALF] = cos_row
                sin_b[norm_i : norm_i + 1, 0:ROPE_HALF] = sin_row
        partial_sq = pl.full([1, HCA_C128_RMS_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=2):
            rms_h0 = rms_kb * HEAD_TILE
            kv_rms_chunk = pooled_kv_pad[r0 : r0 + HCA_C128_RMS_TILE, rms_h0 : rms_h0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, HCA_C128_RMS_TILE]))

        variance = pl.reshape(pl.add(pl.mul(partial_sq, 1.0 / HEAD_DIM), EPS), [HCA_C128_RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for norm_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_TILE, stage=2):
            norm_h0 = norm_kb * HEAD_TILE
            kv_norm_chunk = pooled_kv_pad[r0 : r0 + HCA_C128_RMS_TILE, norm_h0 : norm_h0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, norm_h0 : norm_h0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv_pad[r0 : r0 + HCA_C128_RMS_TILE, norm_h0 : norm_h0 + HEAD_TILE] = normed_chunk

        kv_rope = pooled_kv_pad[r0 : r0 + HCA_C128_RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope, inv_rms), gamma_rope)
        # A3 interleaved swap-gather: one data gather + sign trick.
        # out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]; idx built in-kernel from pl.arange.
        # The index templates are prepared at the parent scope. Level-3 child
        # scopes do not run PTOPlanMemory, so generating TCI/arange here would
        # require an unavailable implicit temporary in PTOAS 0.54.
        rope_dup_idx = rope_dup_idx_template[:, :]
        rope_swap_idx = rope_swap_idx_template[:, :]
        rope_sign = rope_sign_template[:, :]
        cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_kv_pad[r0 : r0 + HCA_C128_RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_rot

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_kv_finalize"):
        for final_i in pl.range(MAX_CMP_WRITES):
            final_cmp_row_raw = pl.read(write_dst_map, [0, final_i])
            if final_cmp_row_raw >= 0:
                final_cmp_row = pl.cast(final_cmp_row_raw, pl.INDEX)
                for final_hb in pl.range(HEAD_DIM // HEAD_TILE):
                    final_h0 = final_hb * HEAD_TILE
                    final_chunk = normed_kv_pad[final_i : final_i + 1, final_h0 : final_h0 + HEAD_TILE]
                    cmp_kv_flat[final_cmp_row : final_cmp_row + 1, final_h0 : final_h0 + HEAD_TILE] = pl.cast(
                        final_chunk,
                        target_type=pl.BF16,
                        mode="rint",
                    )

    # Writes through the flattened views already update the caller-owned buffers; a dynamic
    # reshape-back is not valid in this nested inline kernel.

    return cmp_kv, compress_state


@pl.jit.inline(auto_scope=False)
def prefill_compressor_ratio128(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    """Compress packed requests independently through ordered 512-row state tiles."""
    request_count = pl.tensor.dim(query_start_loc, 0) - 1
    rope_dup_idx_template = pl.create_tensor([HCA_C128_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_swap_idx_template = pl.create_tensor([HCA_C128_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_sign_template = pl.create_tensor([HCA_C128_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_rope_index_prepare"):
        # Scalar construction avoids TCI's implicit temporary, which is not
        # available when this inline program is itself nested under HCA.
        for rope_r in pl.range(HCA_C128_RMS_TILE):
            for rope_c in pl.range(ROPE_HEAD_DIM):
                rope_lane = rope_c % 2
                pl.write(
                    rope_dup_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c // 2, pl.INT32),
                )
                pl.write(
                    rope_swap_idx_template,
                    [rope_r, rope_c],
                    pl.cast(rope_c + 1 - rope_lane * 2, pl.INT32),
                )
                pl.write(
                    rope_sign_template,
                    [rope_r, rope_c],
                    pl.cast(pl.cast(rope_lane * 2 - 1, pl.INT32), pl.FP32),
                )

    # One GM-resident token orders the stateful 512-row tiles without exposing
    # a TaskId carrier through the inline boundary.  Scatter reads it (INPUT),
    # while the post-pool commit writes it (InOut), yielding
    # commit[i-1] -> scatter[i] -> pool[i] -> commit[i] in TensorMap.
    state_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_state_order_init"):
        pl.write(state_order_fence, [0], pl.cast(0, pl.INT32))
    for request in pl.range(request_count):
        request_start = pl.cast(pl.read(query_start_loc, [request]), pl.INDEX)
        request_end = pl.cast(pl.read(query_start_loc, [request + 1]), pl.INDEX)
        request_table = compress_state_block_table[request]
        for request_offset in pl.range(0, request_end - request_start, PREFILL_STATE_TILE):
            tile_base = request_start + request_offset
            tile_rows = pl.min(PREFILL_STATE_TILE, request_end - tile_base)
            with pl.scope():
                _prefill_compressor_ratio128_tile(
                    x,
                    compress_state,
                    request_table,
                    wkv,
                    wgate,
                    ape,
                    norm_w,
                    cmp_freqs_cos,
                    cmp_freqs_sin,
                    cmp_kv,
                    position_ids,
                    cmp_slot_mapping,
                    state_slot_mapping,
                    rope_dup_idx_template,
                    rope_swap_idx_template,
                    rope_sign_template,
                    state_order_fence,
                    tile_base,
                    tile_rows,
                )
    return cmp_kv, compress_state


@pl.jit
def prefill_compressor_ratio128_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
):
    x.bind_dynamic(0, T_DYN)
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    compress_state.bind_dynamic(0, STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)

    return prefill_compressor_ratio128(
        x,
        query_start_loc,
        compress_state,
        compress_state_block_table,
        wkv,
        wgate,
        ape,
        norm_w,
        cmp_freqs_cos,
        cmp_freqs_sin,
        cmp_kv,
        position_ids,
        cmp_slot_mapping,
        state_slot_mapping,
    )


def golden_prefill_compressor_ratio128(tensors):
    import torch

    token_count = tensors["x"].shape[0]
    kv_proj = tensors["x"].float() @ tensors["wkv"].float().t()  # wkv stored [OUT_DIM, D] for b_trans
    score_proj = tensors["x"].float() @ tensors["wgate"].float().t()
    compress_state_flat = tensors["compress_state"].view(
        HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE,
        COMPRESS_STATE_DIM,
    )
    kv_state_flat = compress_state_flat[:, :OUT_DIM]
    score_state_flat = compress_state_flat[:, OUT_DIM:]
    state_block_table = tensors["compress_state_block_table"][0]
    cmp_kv_flat = tensors["cmp_kv"].view(CMP_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * HCA_STATE_BLOCK_SIZE + intra

    for token_id in range(token_count):
        dst_row = int(tensors["cmp_slot_mapping"][token_id].item())
        if dst_row < 0:
            continue
        write_pos = int(tensors["position_ids"][token_id].item())
        pool_kv_state = torch.zeros(STATE_LEN, OUT_DIM, dtype=torch.float32)
        pool_score_state = torch.zeros(STATE_LEN, OUT_DIM, dtype=torch.float32)
        for slot in range(STATE_LEN):
            row = state_row(write_pos + 1 - COMPRESS_RATIO + slot)
            if row >= 0:
                pool_kv_state[slot] = kv_state_flat[row]
                pool_score_state[slot] = score_state_flat[row]
        for t in range(token_count):
            pos = int(tensors["position_ids"][t].item())
            if pos > write_pos:
                continue
            slot = pos % COMPRESS_RATIO
            pool_kv_state[slot] = kv_proj[t]
            pool_score_state[slot] = score_proj[t] + tensors["ape"][slot]
        pooled = (pool_kv_state * pool_score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
        inv = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = pooled * inv * tensors["norm_w"].float().view(1, HEAD_DIM)
        rope_pair = normed[..., NOPE_HEAD_DIM:].unflatten(-1, (-1, 2))
        even = rope_pair[..., 0].float()
        odd = rope_pair[..., 1].float()
        cos = tensors["cmp_freqs_cos"][token_id : token_id + 1, 0:ROPE_HALF].float()
        sin = tensors["cmp_freqs_sin"][token_id : token_id + 1, 0:ROPE_HALF].float()
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        normed[:, NOPE_HEAD_DIM:] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cmp_kv_flat[dst_row] = normed[0]

    for t in range(token_count):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][slot]


def build_tensor_specs(start_pos: int = START_POS, token_count: int = PREFILL_SEQ):
    import torch
    from golden import TensorSpec
    from utils import token_local_rope

    if token_count <= 0 or token_count > MAX_SEQ_LEN:
        raise ValueError(f"token_count must be in [1, {MAX_SEQ_LEN}], got {token_count}")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + token_count > MAX_SEQ_LEN:
        raise ValueError("start_pos + token_count exceeds max_position_embeddings")

    def init_compress_state_block_table():
        logical_blocks = torch.arange(HCA_STATE_MAX_BLOCKS, dtype=torch.int64)
        return ((logical_blocks * 17 + 3) % HCA_STATE_PHYSICAL_BLOCKS).to(torch.int32).unsqueeze(0)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        physical_block = (block * 17 + 3) % HCA_STATE_PHYSICAL_BLOCKS
        return physical_block * HCA_STATE_BLOCK_SIZE + intra

    def init_x():
        return ((torch.rand(token_count, D) - 0.5) * 0.1).to(torch.bfloat16)

    def init_compress_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, start_pos - COMPRESS_RATIO), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                state.view(-1, COMPRESS_STATE_DIM)[row] = (torch.rand(COMPRESS_STATE_DIM) - 0.5) * 0.05
        return state

    # BF16 weight std and RMSNorm gamma mean/std, averaged over DeepSeek-V4-Flash-0731
    # layers 7/9 (the ratio-128 HCA main compressor).
    def init_wkv():
        return torch.randn(OUT_DIM, D) * 0.0240

    def init_wgate():
        return torch.randn(OUT_DIM, D) * 0.0309

    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.0332

    def init_norm_w():
        return 0.0982 + 0.0539 * torch.randn(HEAD_DIM)

    def init_cmp_kv():
        return torch.zeros(CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)

    def init_position_ids():
        return torch.arange(start_pos, start_pos + token_count, dtype=torch.int32)

    def init_cmp_rope_positions():
        positions = init_position_ids().to(torch.int64)
        boundary = (positions + 1) % COMPRESS_RATIO == 0
        return torch.where(boundary, positions - (COMPRESS_RATIO - 1), torch.zeros_like(positions))

    def init_cmp_freqs_cos():
        cos, _ = token_local_rope(
            M, COMPRESS_RATIO, init_cmp_rope_positions(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return cos.contiguous()

    def init_cmp_freqs_sin():
        _, sin = token_local_rope(
            M, COMPRESS_RATIO, init_cmp_rope_positions(),
            max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
        )
        return sin.contiguous()

    def init_cmp_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        for token_id in range(token_count):
            pos = start_pos + token_id
            if pos + 1 >= COMPRESS_RATIO and (pos + 1) % COMPRESS_RATIO == 0:
                mapping[token_id] = (pos + 1) // COMPRESS_RATIO - 1
        return mapping

    def init_state_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        for token_id in range(token_count):
            mapping[token_id] = state_row(start_pos + token_id)
        return mapping

    return [
        TensorSpec("x", [token_count, D], torch.bfloat16, init_value=init_x),
        TensorSpec("query_start_loc", [2], torch.int32, init_value=torch.tensor([0, token_count], dtype=torch.int32)),
        TensorSpec(
            "compress_state",
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [1, HCA_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cmp_freqs_cos", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_cos),
        TensorSpec("cmp_freqs_sin", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_sin),
        TensorSpec(
            "cmp_kv",
            [CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
        TensorSpec("position_ids", [token_count], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [token_count], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [token_count], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(
        description="Standalone token-major DeepSeek V4 prefill compressor ratio128 validation."
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only; also the implicit behavior on the *sim platforms CI uses.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help="Fixture-only absolute position for token 0; not a JIT kernel parameter.",
    )
    parser.add_argument(
        "--token-count",
        "--num-tokens",
        dest="token_count",
        type=int,
        default=PREFILL_SEQ,
        help="Physical token rows, up to 8192.",
    )
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=prefill_compressor_ratio128_test,
        specs=build_tensor_specs(args.start_pos, args.token_count),
        golden_fn=golden_prefill_compressor_ratio128,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform, device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
