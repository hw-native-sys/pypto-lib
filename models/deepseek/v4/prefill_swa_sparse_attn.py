# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill SWA sparse attention and output projection."""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_DIM = M.nope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# SWA cache contract. The ratio-0 path has only the sliding-window cache.
ORI_MAX_BLOCKS = 1
MAX_BLOCKS = ORI_MAX_BLOCKS
BLOCK_NUM = B * MAX_BLOCKS

# SWA + o_proj tiling.
NEG_INF = -1e20
ATTN_HEAD_TILE = 16
ATTN_TASK_TILE = 2
ATTN_ONLINE_VALUE_CHUNK = 64
SPARSE_ATTN_TILE = 64
SPARSE_ATTN_BLOCKS = (WIN + SPARSE_ATTN_TILE - 1) // SPARSE_ATTN_TILE
KV_CACHE_WRITE_TILE = 16
KV_WINDOW_ROWS = T * SPARSE_ATTN_BLOCKS * SPARSE_ATTN_TILE
SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK
ROPE_TOKEN_TILE = 4
ROPE_PACK_TOKEN_TILE = 16
ROPE_PACK_SPMD_BLOCKS = (T // ROPE_PACK_TOKEN_TILE) * O_GROUPS
O_PROJ_T_TILE = 16
A_K_CHUNK = 128
A_N_CHUNK = 128
B_K_CHUNK = 128
B_N_CHUNK = 128
QUANT_CHUNK = 32
QUANT_TOKEN_TILE = 8

assert WIN == BLOCK_SIZE, "SWA prefill currently assumes one window page per batch"
assert S <= WIN, "SWA prefill tile must not exceed the sliding-window ring size"
assert H % ATTN_HEAD_TILE == 0, "attention head tile must divide H"
assert T % ATTN_TASK_TILE == 0, "attention token task tile must divide prefill T"
assert HEAD_DIM % ATTN_ONLINE_VALUE_CHUNK == 0, "online attention value chunk must divide head dim"
assert NOPE_DIM % ATTN_ONLINE_VALUE_CHUNK == 0, "online attention chunk must split noPE/rope boundary"
assert S % KV_CACHE_WRITE_TILE == 0, "KV cache write tile must divide prefill S tile"
assert T % O_PROJ_T_TILE == 0, "o_proj token tile must divide prefill T"


@pl.jit.inline
def prefill_swa_write_kv_cache(
    kv:          pl.Tensor[[T, HEAD_DIM],                    pl.BF16],
    kv_cache:    pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, MAX_BLOCKS],                  pl.INT32],
    start_pos:   pl.Scalar[pl.INT32],
):
    kv_cache_flat = pl.reshape(kv_cache, [BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    tile_start_pos = pl.cast(start_pos, pl.INDEX)

    # Write the current prefill KV into the sliding-window ring through block_table.
    for s0 in pl.range(0, S, KV_CACHE_WRITE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_kv_cache_store"):
            slot0 = (tile_start_pos + s0) % WIN
            intra0 = slot0 % BLOCK_SIZE
            for b in pl.range(B):
                if intra0 + KV_CACHE_WRITE_TILE <= BLOCK_SIZE:
                    blk_id = pl.cast(pl.read(block_table, [b, slot0 // BLOCK_SIZE]), pl.INDEX)
                    dst_row = blk_id * BLOCK_SIZE + intra0
                    kv_tile = pl.load(
                        kv,
                        [b * S + s0, 0],
                        [KV_CACHE_WRITE_TILE, HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_cache_flat = pl.store(kv_tile, [dst_row, 0], kv_cache_flat)
                else:
                    for ds in pl.range(KV_CACHE_WRITE_TILE):
                        slot = (tile_start_pos + s0 + ds) % WIN
                        blk_id = pl.cast(pl.read(block_table, [b, slot // BLOCK_SIZE]), pl.INDEX)
                        dst_row = blk_id * BLOCK_SIZE + (slot % BLOCK_SIZE)
                        kv_row = pl.load(
                            kv,
                            [b * S + s0 + ds, 0],
                            [1, HEAD_DIM],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        kv_cache_flat = pl.store(kv_row, [dst_row, 0], kv_cache_flat)

    return pl.reshape(kv_cache_flat, [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])


@pl.jit.inline
def prefill_swa_build_kv_window(
    kv:                pl.Tensor[[T, HEAD_DIM],                    pl.BF16],
    kv_cache:          pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table:       pl.Tensor[[B, MAX_BLOCKS],                  pl.INT32],
    kv_window_rows:    pl.Tensor[[KV_WINDOW_ROWS, HEAD_DIM],       pl.BF16],
    start_pos:         pl.Scalar[pl.INT32],
):
    kv_cache_flat = pl.reshape(kv_cache, [BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    # Materialize each token's causal SWA window: current prompt KV first,
    # historical KV from kv_cache through block_table when start_pos > 0.
    for attn_t0 in pl.parallel(0, T, ATTN_TASK_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_kv_window"):
            for attn_dt in pl.range(ATTN_TASK_TILE):
                attn_t = attn_t0 + attn_dt
                attn_b = attn_t // S
                attn_s = attn_t - attn_b * S
                tile_start_pos = pl.cast(start_pos, pl.INDEX)
                attn_abs_pos = tile_start_pos + attn_s
                window_valid = pl.min(WIN, attn_abs_pos + 1)
                kv_start_abs = attn_abs_pos + 1 - window_valid
                for sb in pl.range(SPARSE_ATTN_BLOCKS):
                    tile_start = sb * SPARSE_ATTN_TILE
                    kv_window = pl.full([SPARSE_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    if tile_start < window_valid:
                        tile_valid = pl.min(SPARSE_ATTN_TILE, window_valid - tile_start)
                        for key_i in pl.range(SPARSE_ATTN_TILE):
                            if key_i < tile_valid:
                                key_abs_pos = kv_start_abs + tile_start + key_i
                                if key_abs_pos >= tile_start_pos:
                                    key_local_s = key_abs_pos - tile_start_pos
                                    kv_window = pl.assemble(
                                        kv_window,
                                        kv[
                                            attn_b * S + key_local_s : attn_b * S + key_local_s + 1,
                                            0:HEAD_DIM,
                                        ],
                                        [key_i, 0],
                                    )
                                else:
                                    ori_slot = key_abs_pos % WIN
                                    blk_id = pl.cast(pl.read(block_table, [attn_b, ori_slot // BLOCK_SIZE]), pl.INDEX)
                                    intra = ori_slot % BLOCK_SIZE
                                    cache_row = blk_id * BLOCK_SIZE + intra
                                    kv_window = pl.assemble(
                                        kv_window,
                                        kv_cache_flat[cache_row : cache_row + 1, 0:HEAD_DIM],
                                        [key_i, 0],
                                    )
                    window_row = (attn_t * SPARSE_ATTN_BLOCKS + sb) * SPARSE_ATTN_TILE
                    kv_window_rows = pl.assemble(kv_window_rows, kv_window, [window_row, 0])

    return kv_window_rows


@pl.jit.inline
def prefill_swa_o_proj_from_packed(
    o_packed:   pl.Tensor[[O_GROUPS * T, O_GROUP_IN],      pl.BF16],
    wo_a:       pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],  pl.BF16],
    wo_b:       pl.Tensor[[D, O_GROUPS * O_LORA],          pl.INT8],
    wo_b_scale: pl.Tensor[[D],                             pl.FP32],
    attn_out:   pl.Tensor[[T, D],                          pl.BF16],
):
    a_k_blocks = O_GROUP_IN // A_K_CHUNK
    a_n_blocks = O_LORA // A_N_CHUNK
    a_amax_blocks = O_GROUPS * a_n_blocks
    b_k_blocks = (O_GROUPS * O_LORA) // B_K_CHUNK
    b_n_blocks = D // B_N_CHUNK

    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    o_r_amax_parts = pl.create_tensor([a_amax_blocks, T], dtype=pl.FP32)
    o_r_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)

    # First low-rank o_proj leg: grouped BF16 matmul into O_GROUPS * O_LORA.
    for g in pl.parallel(0, O_GROUPS, 1):
        row_base_o = g * T
        out_col_g = g * O_LORA
        for nb in pl.parallel(0, a_n_blocks, 1):
            n0 = nb * A_N_CHUNK
            for t0 in pl.parallel(0, T, O_PROJ_T_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_wo_a"):
                    xa0_chunk = o_packed[row_base_o + t0 : row_base_o + t0 + O_PROJ_T_TILE, 0:A_K_CHUNK]
                    wa0_chunk = wo_a[g : g + 1, n0 : n0 + A_N_CHUNK, 0:A_K_CHUNK]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, a_k_blocks, stage=2):
                        k0 = kb * A_K_CHUNK
                        xa_k_chunk = o_packed[
                            row_base_o + t0 : row_base_o + t0 + O_PROJ_T_TILE,
                            k0 : k0 + A_K_CHUNK,
                        ]
                        wa_k_chunk = wo_a[g : g + 1, n0 : n0 + A_N_CHUNK, k0 : k0 + A_K_CHUNK]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

                    acc_a_2d = pl.reshape(acc_a, [O_PROJ_T_TILE, A_N_CHUNK])
                    acc_a_bf16 = pl.cast(acc_a_2d, target_type=pl.BF16)
                    o_r[t0 : t0 + O_PROJ_T_TILE, out_col_g + n0 : out_col_g + n0 + A_N_CHUNK] = acc_a_bf16
                    acc_a_f32 = pl.cast(acc_a_bf16, target_type=pl.FP32)
                    acc_a_abs = pl.maximum(acc_a_f32, pl.neg(acc_a_f32))
                    acc_a_amax = pl.reshape(pl.row_max(acc_a_abs), [1, O_PROJ_T_TILE])
                    amax_part_row = g * a_n_blocks + nb
                    o_r_amax_parts[amax_part_row : amax_part_row + 1, t0 : t0 + O_PROJ_T_TILE] = acc_a_amax

    # Match decode quantization: rint to INT32, round through FP16, then INT8.
    for quant_t0 in pl.parallel(0, T, QUANT_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_wo_b_quant"):
            or_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for ab in pl.range(0, a_amax_blocks, 1):
                or_a_part = o_r_amax_parts[ab : ab + 1, quant_t0 : quant_t0 + QUANT_TOKEN_TILE]
                or_amax = pl.maximum(or_amax, or_a_part)
            or_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
            or_scale_dq = pl.reshape(pl.recip(or_sq_row), [QUANT_TOKEN_TILE, 1])
            o_r_scale_dq[quant_t0 : quant_t0 + QUANT_TOKEN_TILE, 0:1] = or_scale_dq
            or_sq_col = pl.reshape(or_sq_row, [QUANT_TOKEN_TILE, 1])
            for k1 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
                or_q_f32 = pl.cast(
                    o_r[quant_t0 : quant_t0 + QUANT_TOKEN_TILE, k1 : k1 + QUANT_CHUNK],
                    target_type=pl.FP32,
                )
                or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
                or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
                or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
                o_r_i8[quant_t0 : quant_t0 + QUANT_TOKEN_TILE, k1 : k1 + QUANT_CHUNK] = pl.cast(
                    or_q_half,
                    target_type=pl.INT8,
                    mode="trunc",
                )

    # Second low-rank leg consumes int8 activations and per-row dequant scale.
    for nb in pl.parallel(0, b_n_blocks, 1):
        n0 = nb * B_N_CHUNK
        for t0 in pl.parallel(0, T, O_PROJ_T_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_wo_b"):
                xb0_chunk = o_r_i8[t0 : t0 + O_PROJ_T_TILE, 0:B_K_CHUNK]
                wb0_chunk = wo_b[n0 : n0 + B_N_CHUNK, 0:B_K_CHUNK]
                acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
                for kb in pl.pipeline(1, b_k_blocks, stage=2):
                    k0 = kb * B_K_CHUNK
                    xb_k_chunk = o_r_i8[t0 : t0 + O_PROJ_T_TILE, k0 : k0 + B_K_CHUNK]
                    wb_k_chunk = wo_b[n0 : n0 + B_N_CHUNK, k0 : k0 + B_K_CHUNK]
                    acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

                wb_scale_chunk = pl.reshape(wo_b_scale[n0 : n0 + B_N_CHUNK], [1, B_N_CHUNK])
                attn_chunk = pl.cast(acc_b, target_type=pl.FP32, mode="none")
                attn_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(attn_chunk, o_r_scale_dq[t0 : t0 + O_PROJ_T_TILE, 0:1]),
                    wb_scale_chunk,
                )
                attn_out[t0 : t0 + O_PROJ_T_TILE, n0 : n0 + B_N_CHUNK] = pl.cast(
                    attn_chunk,
                    target_type=pl.BF16,
                    mode="rint",
                )

    return attn_out


@pl.jit.inline
def prefill_swa_attention_values(
    q:                 pl.Tensor[[T, H, HEAD_DIM],                 pl.BF16],
    kv:                pl.Tensor[[T, HEAD_DIM],                    pl.BF16],
    kv_cache:          pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table:       pl.Tensor[[B, MAX_BLOCKS],                  pl.INT32],
    attn_sink:         pl.Tensor[[H],                              pl.FP32],
    attn_values:       pl.Tensor[[T * H, HEAD_DIM],                pl.BF16],
    start_pos:         pl.Scalar[pl.INT32],
):
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    attn_exp = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, SPARSE_ATTN_TILE], dtype=pl.BF16)
    attn_blk_mi = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, 1], dtype=pl.FP32)
    attn_blk_li = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, 1], dtype=pl.FP32)
    # Match the proven decode sparse-attn shape: keep the full value vector per
    # sparse block. The earlier 64-wide scratch reused the same GM buffer across
    # unrolled value chunks and produced chunk-boundary errors on NPU.
    attn_blk_oi = pl.create_tensor([T * H * SPARSE_ATTN_BLOCKS, HEAD_DIM], dtype=pl.FP32)
    kv_window_rows = pl.create_tensor([KV_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    kv_window_rows = prefill_swa_build_kv_window(kv, kv_cache, block_table, kv_window_rows, start_pos)

    # QK pass stores per-block exp(scores), max, and denominator fragments.
    for attn_t0 in pl.parallel(0, T, ATTN_TASK_TILE):
        for h0 in pl.parallel(0, H, ATTN_HEAD_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_qk"):
                for attn_dt in pl.range(ATTN_TASK_TILE):
                    attn_t = attn_t0 + attn_dt
                    attn_b = attn_t // S
                    attn_s = attn_t - attn_b * S
                    tile_start_pos = pl.cast(start_pos, pl.INDEX)
                    attn_abs_pos = tile_start_pos + attn_s
                    window_valid = pl.min(WIN, attn_abs_pos + 1)
                    q_heads = q_flat[attn_t * H + h0 : attn_t * H + h0 + ATTN_HEAD_TILE, 0:HEAD_DIM]
                    for sb in pl.range(SPARSE_ATTN_BLOCKS):
                        tile_start = sb * SPARSE_ATTN_TILE
                        if tile_start < window_valid:
                            tile_valid = pl.min(SPARSE_ATTN_TILE, window_valid - tile_start)
                            window_row = (attn_t * SPARSE_ATTN_BLOCKS + sb) * SPARSE_ATTN_TILE
                            kv_window_qk = kv_window_rows[window_row : window_row + SPARSE_ATTN_TILE, 0:HEAD_DIM]
                            raw_scores = pl.matmul(q_heads, kv_window_qk, b_trans=True, out_dtype=pl.FP32)
                            scores_valid = pl.slice(
                                pl.mul(raw_scores, SOFTMAX_SCALE),
                                [ATTN_HEAD_TILE, SPARSE_ATTN_TILE],
                                [0, 0],
                                valid_shape=[ATTN_HEAD_TILE, tile_valid],
                            )
                            scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                            score_max = pl.row_max(scores)
                            exp_scores = pl.exp(pl.row_expand_sub(scores, score_max))
                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                            li = pl.row_sum(pl.cast(exp_scores_bf16, target_type=pl.FP32))
                            block_row = attn_t * H * SPARSE_ATTN_BLOCKS + sb * H + h0
                            attn_exp[block_row : block_row + ATTN_HEAD_TILE, 0:SPARSE_ATTN_TILE] = exp_scores_bf16
                            attn_blk_mi[block_row : block_row + ATTN_HEAD_TILE, 0:1] = score_max
                            attn_blk_li[block_row : block_row + ATTN_HEAD_TILE, 0:1] = li
                        else:
                            block_row = attn_t * H * SPARSE_ATTN_BLOCKS + sb * H + h0
                            zero_exp = pl.full([ATTN_HEAD_TILE, SPARSE_ATTN_TILE], dtype=pl.BF16, value=0.0)
                            neg_mi = pl.reshape(
                                pl.full([1, ATTN_HEAD_TILE], dtype=pl.FP32, value=NEG_INF),
                                [ATTN_HEAD_TILE, 1],
                            )
                            zero_li = pl.reshape(
                                pl.full([1, ATTN_HEAD_TILE], dtype=pl.FP32, value=0.0),
                                [ATTN_HEAD_TILE, 1],
                            )
                            attn_exp[block_row : block_row + ATTN_HEAD_TILE, 0:SPARSE_ATTN_TILE] = zero_exp
                            attn_blk_mi[block_row : block_row + ATTN_HEAD_TILE, 0:1] = neg_mi
                            attn_blk_li[block_row : block_row + ATTN_HEAD_TILE, 0:1] = zero_li

    # PV pass reuses the stored exp(scores) to build one value vector per block.
    for attn_t0 in pl.parallel(0, T, ATTN_TASK_TILE):
        for h0 in pl.parallel(0, H, ATTN_HEAD_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_pv"):
                for attn_dt in pl.range(ATTN_TASK_TILE):
                    attn_t = attn_t0 + attn_dt
                    attn_s = attn_t % S
                    tile_start_pos = pl.cast(start_pos, pl.INDEX)
                    attn_abs_pos = tile_start_pos + attn_s
                    window_valid = pl.min(WIN, attn_abs_pos + 1)
                    for sb in pl.range(SPARSE_ATTN_BLOCKS):
                        tile_start = sb * SPARSE_ATTN_TILE
                        if tile_start < window_valid:
                            block_row = attn_t * H * SPARSE_ATTN_BLOCKS + sb * H + h0
                            exp_scores_bf16 = attn_exp[block_row : block_row + ATTN_HEAD_TILE, 0:SPARSE_ATTN_TILE]
                            window_row = (attn_t * SPARSE_ATTN_BLOCKS + sb) * SPARSE_ATTN_TILE
                            kv_window_pv = kv_window_rows[window_row : window_row + SPARSE_ATTN_TILE, 0:HEAD_DIM]
                            cur_oi = pl.matmul(exp_scores_bf16, kv_window_pv, out_dtype=pl.FP32)
                            attn_blk_oi = pl.assemble(attn_blk_oi, cur_oi, [block_row, 0])

    # Merge real KV blocks first; attn_sink only extends the final denominator.
    for attn_t0 in pl.parallel(0, T, ATTN_TASK_TILE):
        for h0 in pl.parallel(0, H, ATTN_HEAD_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_merge_norm"):
                for attn_dt in pl.range(ATTN_TASK_TILE):
                    attn_t = attn_t0 + attn_dt
                    attn_s = attn_t % S
                    tile_start_pos = pl.cast(start_pos, pl.INDEX)
                    attn_abs_pos = tile_start_pos + attn_s
                    window_valid = pl.min(WIN, attn_abs_pos + 1)
                    head_row = attn_t * H + h0
                    block_row0 = attn_t * H * SPARSE_ATTN_BLOCKS + h0
                    merge_mi = attn_blk_mi[block_row0 : block_row0 + ATTN_HEAD_TILE, 0:1]
                    merge_li = attn_blk_li[block_row0 : block_row0 + ATTN_HEAD_TILE, 0:1]
                    merge_oi = attn_blk_oi[block_row0 : block_row0 + ATTN_HEAD_TILE, 0:HEAD_DIM]

                    for sb in pl.range(1, SPARSE_ATTN_BLOCKS):
                        tile_start = sb * SPARSE_ATTN_TILE
                        if tile_start < window_valid:
                            block_row = attn_t * H * SPARSE_ATTN_BLOCKS + sb * H + h0
                            cur_mi = attn_blk_mi[block_row : block_row + ATTN_HEAD_TILE, 0:1]
                            cur_li = attn_blk_li[block_row : block_row + ATTN_HEAD_TILE, 0:1]
                            cur_oi = attn_blk_oi[block_row : block_row + ATTN_HEAD_TILE, 0:HEAD_DIM]
                            mi_new = pl.maximum(merge_mi, cur_mi)
                            alpha = pl.exp(pl.sub(merge_mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            merge_li = pl.add(pl.mul(alpha, merge_li), pl.mul(beta, cur_li))
                            merge_oi = pl.add(
                                pl.row_expand_mul(merge_oi, alpha),
                                pl.row_expand_mul(cur_oi, beta),
                            )
                            merge_mi = mi_new

                    sink_bias = pl.reshape(attn_sink[h0 : h0 + ATTN_HEAD_TILE], [ATTN_HEAD_TILE, 1])
                    denom = pl.add(merge_li, pl.exp(pl.sub(sink_bias, merge_mi)))
                    attn_value = pl.cast(pl.row_expand_div(merge_oi, denom), target_type=pl.BF16)
                    attn_values = pl.assemble(attn_values, attn_value, [head_row, 0])

    return attn_values


@pl.jit.inline
def prefill_swa_pack_context_from_values(
    attn_values:       pl.Tensor[[T * H, HEAD_DIM],                pl.BF16],
    freqs_cos_t:       pl.Tensor[[T, ROPE_HEAD_DIM],               pl.BF16],
    freqs_sin_t:       pl.Tensor[[T, ROPE_HEAD_DIM],               pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local:  pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    o_packed:          pl.Tensor[[O_GROUPS * T, O_GROUP_IN],       pl.BF16],
):
    o_proj_even = pl.create_tensor([T * H, HALF_ROPE], dtype=pl.FP32)
    o_proj_odd = pl.create_tensor([T * H, HALF_ROPE], dtype=pl.FP32)
    rope_even_interleave_buf = pl.create_tensor([T * H, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_odd_interleave_buf = pl.create_tensor([T * H, ROPE_HEAD_DIM], dtype=pl.FP32)

    # noPE dimensions can be packed directly into grouped o_proj layout.
    for nope_pack_block in pl.spmd(ROPE_PACK_SPMD_BLOCKS, name_hint="prefill_swa_nope_pack"):
        nope_pack_token_block = nope_pack_block // O_GROUPS
        nope_pack_g = nope_pack_block - nope_pack_token_block * O_GROUPS
        nope_pack_t0 = nope_pack_token_block * ROPE_PACK_TOKEN_TILE
        for nope_pack_dt in pl.range(ROPE_PACK_TOKEN_TILE):
            nope_pack_t = nope_pack_t0 + nope_pack_dt
            nope_pack_head_row = nope_pack_t * H + nope_pack_g * HEADS_PER_GROUP
            nope_pack_row = nope_pack_g * T + nope_pack_t
            for nope_v0 in pl.range(0, NOPE_DIM, ATTN_ONLINE_VALUE_CHUNK):
                nope_tile = attn_values[
                    nope_pack_head_row : nope_pack_head_row + HEADS_PER_GROUP,
                    nope_v0 : nope_v0 + ATTN_ONLINE_VALUE_CHUNK,
                ]
                for nope_pack_hh in pl.range(HEADS_PER_GROUP):
                    nope_pack_col = nope_pack_hh * HEAD_DIM + nope_v0
                    o_packed = pl.assemble(
                        o_packed,
                        nope_tile[nope_pack_hh : nope_pack_hh + 1, 0:ATTN_ONLINE_VALUE_CHUNK],
                        [nope_pack_row, nope_pack_col],
                    )

    # RoPE dimensions are split to even/odd pairs before inverse RoPE.
    for rope_t0 in pl.parallel(0, T, ROPE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_rope_slice"):
            for rope_dt in pl.range(ROPE_TOKEN_TILE):
                rope_t = rope_t0 + rope_dt
                rope_head_row = rope_t * H
                for rope_r0 in pl.range(0, HALF_ROPE, SPARSE_ROPE_CHUNK):
                    rope_tile = attn_values[
                        rope_head_row : rope_head_row + H,
                        NOPE_DIM + 2 * rope_r0 : NOPE_DIM + 2 * rope_r0 + SPARSE_ROPE_INTERLEAVE_CHUNK,
                    ]
                    rope_even_chunk = pl.matmul(rope_tile, even_select_local, out_dtype=pl.FP32)
                    rope_odd_chunk = pl.matmul(rope_tile, odd_select_local, out_dtype=pl.FP32)
                    o_proj_even = pl.assemble(o_proj_even, rope_even_chunk, [rope_head_row, rope_r0])
                    o_proj_odd = pl.assemble(o_proj_odd, rope_odd_chunk, [rope_head_row, rope_r0])

    # Attention output is rotated back before feeding the model-space o_proj.
    for rope_apply_t0 in pl.parallel(0, T, ROPE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_rope_apply"):
            for rope_dt in pl.range(ROPE_TOKEN_TILE):
                rope_t = rope_apply_t0 + rope_dt
                rope_head_row = rope_t * H
                for rope_r0 in pl.range(0, HALF_ROPE, SPARSE_ROPE_CHUNK):
                    cos_chunk = pl.cast(
                        freqs_cos_t[rope_t : rope_t + 1, rope_r0 : rope_r0 + SPARSE_ROPE_CHUNK],
                        target_type=pl.FP32,
                    )
                    sin_chunk = pl.cast(
                        freqs_sin_t[rope_t : rope_t + 1, rope_r0 : rope_r0 + SPARSE_ROPE_CHUNK],
                        target_type=pl.FP32,
                    )
                    rope_even_chunk = o_proj_even[
                        rope_head_row : rope_head_row + H,
                        rope_r0 : rope_r0 + SPARSE_ROPE_CHUNK,
                    ]
                    rope_odd_chunk = o_proj_odd[
                        rope_head_row : rope_head_row + H,
                        rope_r0 : rope_r0 + SPARSE_ROPE_CHUNK,
                    ]
                    inv_even = pl.add(
                        pl.col_expand_mul(rope_even_chunk, cos_chunk),
                        pl.col_expand_mul(rope_odd_chunk, sin_chunk),
                    )
                    inv_odd = pl.sub(
                        pl.col_expand_mul(rope_odd_chunk, cos_chunk),
                        pl.col_expand_mul(rope_even_chunk, sin_chunk),
                    )
                    rope_even_interleave = pl.matmul(
                        pl.cast(inv_even, target_type=pl.BF16, mode="rint"),
                        even_select_local,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    rope_odd_interleave = pl.matmul(
                        pl.cast(inv_odd, target_type=pl.BF16, mode="rint"),
                        odd_select_local,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    rope_even_interleave_buf = pl.assemble(
                        rope_even_interleave_buf,
                        rope_even_interleave,
                        [rope_head_row, 2 * rope_r0],
                    )
                    rope_odd_interleave_buf = pl.assemble(
                        rope_odd_interleave_buf,
                        rope_odd_interleave,
                        [rope_head_row, 2 * rope_r0],
                    )

    # Re-interleave rotated RoPE dimensions into the same grouped layout as noPE.
    for rope_pack_block in pl.spmd(ROPE_PACK_SPMD_BLOCKS, name_hint="prefill_swa_rope_pack"):
        rope_pack_token_block = rope_pack_block // O_GROUPS
        rope_pack_g = rope_pack_block - rope_pack_token_block * O_GROUPS
        rope_pack_t0 = rope_pack_token_block * ROPE_PACK_TOKEN_TILE
        for rope_pack_dt in pl.range(ROPE_PACK_TOKEN_TILE):
            rope_pack_t = rope_pack_t0 + rope_pack_dt
            rope_pack_head_row = rope_pack_t * H + rope_pack_g * HEADS_PER_GROUP
            rope_even_tile = rope_even_interleave_buf[
                rope_pack_head_row : rope_pack_head_row + HEADS_PER_GROUP,
                0:ROPE_HEAD_DIM,
            ]
            rope_odd_tile = rope_odd_interleave_buf[
                rope_pack_head_row : rope_pack_head_row + HEADS_PER_GROUP,
                0:ROPE_HEAD_DIM,
            ]
            rope_full = pl.cast(pl.add(rope_even_tile, rope_odd_tile), target_type=pl.BF16)
            rope_pack_row = rope_pack_g * T + rope_pack_t
            for rope_pack_hh in pl.range(HEADS_PER_GROUP):
                rope_pack_col = rope_pack_hh * HEAD_DIM + NOPE_DIM
                o_packed = pl.assemble(
                    o_packed,
                    rope_full[rope_pack_hh : rope_pack_hh + 1, 0:ROPE_HEAD_DIM],
                    [rope_pack_row, rope_pack_col],
                )

    return o_packed


@pl.jit.inline
def prefill_swa_sparse_attn(
    q:                 pl.Tensor[[T, H, HEAD_DIM],                 pl.BF16],
    kv:                pl.Tensor[[T, HEAD_DIM],                    pl.BF16],
    kv_cache:          pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table:       pl.Tensor[[B, MAX_BLOCKS],                  pl.INT32],
    attn_sink:         pl.Tensor[[H],                              pl.FP32],
    freqs_cos_t:       pl.Tensor[[T, ROPE_HEAD_DIM],               pl.BF16],
    freqs_sin_t:       pl.Tensor[[T, ROPE_HEAD_DIM],               pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local:  pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    wo_a:              pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],   pl.BF16],
    wo_b:              pl.Tensor[[D, O_GROUPS * O_LORA],           pl.INT8],
    wo_b_scale:        pl.Tensor[[D],                              pl.FP32],
    attn_out:          pl.Tensor[[T, D],                           pl.BF16],
    start_pos:         pl.Scalar[pl.INT32],
):
    attn_values = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.BF16)
    attn_values = prefill_swa_attention_values(q, kv, kv_cache, block_table, attn_sink, attn_values, start_pos)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    o_packed = prefill_swa_pack_context_from_values(
        attn_values,
        freqs_cos_t,
        freqs_sin_t,
        even_select_local,
        odd_select_local,
        o_packed,
    )
    attn_out = prefill_swa_o_proj_from_packed(o_packed, wo_a, wo_b, wo_b_scale, attn_out)

    # Cache is updated after attention so current tokens do not read future KV.
    kv_cache = prefill_swa_write_kv_cache(kv, kv_cache, block_table, start_pos)
    return kv_cache, attn_out


@pl.jit
def prefill_swa_sparse_attn_test(
    q:                 pl.Tensor[[T, H, HEAD_DIM],                 pl.BF16],
    kv:                pl.Tensor[[T, HEAD_DIM],                    pl.BF16],
    kv_cache:          pl.Out[pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table:       pl.Tensor[[B, MAX_BLOCKS],                  pl.INT32],
    attn_sink:         pl.Tensor[[H],                              pl.FP32],
    freqs_cos_t:       pl.Tensor[[T, ROPE_HEAD_DIM],               pl.BF16],
    freqs_sin_t:       pl.Tensor[[T, ROPE_HEAD_DIM],               pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local:  pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    wo_a:              pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],   pl.BF16],
    wo_b:              pl.Tensor[[D, O_GROUPS * O_LORA],           pl.INT8],
    wo_b_scale:        pl.Tensor[[D],                              pl.FP32],
    attn_out:          pl.Out[pl.Tensor[[T, D],                    pl.BF16]],
    start_pos:         pl.Scalar[pl.INT32],
):
    kv_cache, attn_out = prefill_swa_sparse_attn(
        q,
        kv,
        kv_cache,
        block_table,
        attn_sink,
        freqs_cos_t,
        freqs_sin_t,
        even_select_local,
        odd_select_local,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        start_pos,
    )
    return kv_cache, attn_out


def _int8_quant_per_row(x):
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _quant_w_per_row(w):
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def _golden_prefill_swa_values(tensors):
    import torch

    start_pos = int(tensors["start_pos"])
    q = tensors["q"].float().view(B, S, H, HEAD_DIM)
    kv = tensors["kv"].float().view(B, S, HEAD_DIM)
    kv_cache = tensors["kv_cache"].float()
    block_table = tensors["block_table"]
    attn_sink = tensors["attn_sink"].float()

    o = torch.zeros(B, S, H, HEAD_DIM, dtype=torch.float32)
    for b in range(B):
        for s in range(S):
            abs_pos = start_pos + s
            first = max(0, abs_pos - WIN + 1)
            kv_rows = []
            for key_abs in range(first, abs_pos + 1):
                if key_abs >= start_pos:
                    kv_rows.append(kv[b, key_abs - start_pos])
                else:
                    ori_slot = key_abs % WIN
                    blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
                    kv_rows.append(kv_cache[blk_id, ori_slot % BLOCK_SIZE, 0])
            kv_b = torch.stack(kv_rows, dim=0)
            q_t = q[b, s]

            block_mi = []
            block_li = []
            block_oi = []
            for tile_start in range(0, kv_b.shape[0], SPARSE_ATTN_TILE):
                kv_tile = kv_b[tile_start:tile_start + SPARSE_ATTN_TILE]
                scores = torch.einsum("hd,kd->hk", q_t, kv_tile) * SOFTMAX_SCALE
                cur_mi = scores.max(dim=-1, keepdim=True).values
                exp_scores = torch.exp(scores - cur_mi).to(torch.bfloat16).float()
                cur_li = exp_scores.sum(dim=-1, keepdim=True)
                cur_oi = exp_scores @ kv_tile.to(torch.bfloat16).float()
                block_mi.append(cur_mi)
                block_li.append(cur_li)
                block_oi.append(cur_oi)

            merge_mi = block_mi[0]
            merge_li = block_li[0]
            merge_oi = block_oi[0]
            for cur_mi, cur_li, cur_oi in zip(block_mi[1:], block_li[1:], block_oi[1:], strict=True):
                mi_new = torch.maximum(merge_mi, cur_mi)
                alpha = torch.exp(merge_mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                merge_li = alpha * merge_li + beta * cur_li
                merge_oi = alpha * merge_oi + beta * cur_oi
                merge_mi = mi_new

            denom = merge_li + torch.exp(attn_sink.unsqueeze(-1) - merge_mi)
            o[b, s] = (merge_oi / denom).to(torch.bfloat16).float()

    return o


def _golden_swa_attention_o_proj(tensors):
    import torch

    cos = tensors["freqs_cos_t"].float().view(B, S, ROPE_HEAD_DIM)
    sin = tensors["freqs_sin_t"].float().view(B, S, ROPE_HEAD_DIM)
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = _golden_prefill_swa_values(tensors)

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[..., :HALF_ROPE].unsqueeze(2)
    sin_half = sin[..., :HALF_ROPE].unsqueeze(2)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(B, S, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)
    tensors["attn_out"][:] = out.to(torch.bfloat16)


def golden_prefill_swa_sparse_attn(tensors):
    start_pos = int(tensors["start_pos"])

    _golden_swa_attention_o_proj(tensors)

    kv = tensors["kv"].view(B, S, HEAD_DIM)
    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]
    for b in range(B):
        for s in range(S):
            ori_slot = (start_pos + s) % WIN
            blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
            kv_cache[blk_id, ori_slot % BLOCK_SIZE, 0] = kv[b, s]


def build_tensor_specs(start_pos: int = 0):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_q():
        return torch.randn(T, H, HEAD_DIM) * 0.05
    def init_kv():
        return torch.randn(T, HEAD_DIM) * 0.05
    def init_kv_cache():
        if start_pos == 0:
            return torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        return torch.randn(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) * 0.05
    def init_block_table():
        tbl = torch.full((B, MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            tbl[b, 0] = b
        return tbl
    def init_attn_sink():
        return torch.zeros(H)
    def init_freqs_cos_t():
        return torch.cos(torch.arange(T * ROPE_HEAD_DIM).reshape(T, ROPE_HEAD_DIM) * 1e-3)
    def init_freqs_sin_t():
        return torch.sin(torch.arange(T * ROPE_HEAD_DIM).reshape(T, ROPE_HEAD_DIM) * 1e-3)
    def init_even_select_local():
        m = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            m[2 * i, i] = 1
        return m
    def init_odd_select_local():
        m = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            m[2 * i + 1, i] = 1
        return m
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("kv", [T, HEAD_DIM], torch.bfloat16, init_value=init_kv),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("block_table", [B, MAX_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos_t", [T, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos_t),
        TensorSpec("freqs_sin_t", [T, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin_t),
        TensorSpec("even_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK],
                   torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK],
                   torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
        ScalarSpec("start_pos", torch.int32, start_pos),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_swa_sparse_attn_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_swa_sparse_attn,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
