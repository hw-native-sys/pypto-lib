# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 prefill attention compressor for ratio-4 overlapping KV cache (rotate=False)."""

import pypto.language as pl

from config import FP32_NEG_INF
from decode_compressor_ratio4 import *  # noqa: F401,F403
from prefill_sparse_attn import CMP_MAX_BLOCKS as PREFILL_CMP_BLOCK_NUM

B = 1
S = 128
START_POS = 0
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
PREFILL_ROWS = B * PREFILL_COMPRESSED_LEN
HEAD_CHUNK = 256
assert HEAD_DIM % HEAD_CHUNK == 0
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK
K_CHUNK = 512
OUT_CHUNK = 32
HEAD_TILE = 64
RMS_TILE = 16

T = B * S
CSA_STATE_BLOCK_SIZE = 4
CSA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + CSA_STATE_BLOCK_SIZE - 1) // CSA_STATE_BLOCK_SIZE
CSA_STATE_BLOCK_NUM = CSA_STATE_MAX_BLOCKS
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)
PACKED_PROJ_BLOCKS = OUT_DIM // OUT_CHUNK
PACKED_POOL_BLOCKS = MAX_CMP_WRITES * HEAD_BLOCKS
PACKED_STATE_UPDATE_TILE = 16
PACKED_RMS_TILE = 16


@pl.jit.inline
def prefill_compressor_ratio4(
    x: pl.Tensor[[T, D], pl.BF16],
    kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    kv_proj = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    kv_state_flat = pl.reshape(kv_state, [CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, OUT_DIM])
    score_state_flat = pl.reshape(score_state, [CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, OUT_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    pooled_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)

    for proj_idx in pl.spmd(PACKED_PROJ_BLOCKS, name_hint="prefill_c4_state_proj"):
        o0 = proj_idx * OUT_CHUNK
        kv_acc = pl.create_tensor([T, OUT_CHUNK], dtype=pl.FP32)
        score_acc = pl.create_tensor([T, OUT_CHUNK], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_CHUNK, stage=2):
            k0 = kb * K_CHUNK
            x_tile = x[0:T, k0 : k0 + K_CHUNK]
            wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
            wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)
        kv_proj[0:T, o0 : o0 + OUT_CHUNK] = kv_acc
        score_proj[0:T, o0 : o0 + OUT_CHUNK] = score_acc

    for pool_idx in pl.spmd(PACKED_POOL_BLOCKS, name_hint="prefill_c4_softmax_pool"):
        write_i = pool_idx // HEAD_BLOCKS
        hb = pool_idx - write_i * HEAD_BLOCKS
        h0 = hb * HEAD_CHUNK
        pool_kv_tile = pl.create_tensor([STATE_LEN, HEAD_CHUNK], dtype=pl.FP32)
        pool_score_tile = pl.create_tensor([STATE_LEN, HEAD_CHUNK], dtype=pl.FP32)
        write_token = 0
        write_slot_raw = pl.cast(-1, pl.INT64)
        write_seen = pl.cast(0, pl.INDEX)
        for scan_w in pl.range(T):
            if scan_w < num_tokens:
                scan_slot_raw = pl.read(cmp_slot_mapping, [scan_w])
                if scan_slot_raw >= 0:
                    if write_seen == write_i:
                        write_token = scan_w
                        write_slot_raw = scan_slot_raw
                    write_seen = write_seen + 1
        if write_slot_raw >= 0:
            write_pos = pl.read(position_ids, [write_token])
            cur_start = write_pos + 1 - COMPRESS_RATIO
            prev_start = cur_start - COMPRESS_RATIO
            for pool_s in pl.range(COMPRESS_RATIO):
                prev_abs = prev_start + pool_s
                front_slot = pool_s
                pool_kv_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_score_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
                if write_pos >= 2 * COMPRESS_RATIO - 1:
                    prev_state_block = pl.cast(prev_abs // CSA_STATE_BLOCK_SIZE, pl.INDEX)
                    prev_state_intra = pl.cast(prev_abs - prev_state_block * CSA_STATE_BLOCK_SIZE, pl.INDEX)
                    prev_phys_block_raw = pl.read(compress_state_block_table, [prev_state_block])
                    if prev_phys_block_raw >= 0:
                        prev_phys_block = pl.cast(prev_phys_block_raw, pl.INDEX)
                        prev_state_row = prev_phys_block * CSA_STATE_BLOCK_SIZE + prev_state_intra
                        pool_kv_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = kv_state_flat[
                            prev_state_row : prev_state_row + 1,
                            h0 : h0 + HEAD_CHUNK,
                        ]
                        pool_score_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = score_state_flat[
                            prev_state_row : prev_state_row + 1,
                            h0 : h0 + HEAD_CHUNK,
                        ]

                cur_abs = cur_start + pool_s
                back_slot = COMPRESS_RATIO + pool_s
                pool_kv_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_score_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )
                cur_state_block = pl.cast(cur_abs // CSA_STATE_BLOCK_SIZE, pl.INDEX)
                cur_state_intra = pl.cast(cur_abs - cur_state_block * CSA_STATE_BLOCK_SIZE, pl.INDEX)
                cur_phys_block_raw = pl.read(compress_state_block_table, [cur_state_block])
                if cur_phys_block_raw >= 0:
                    cur_phys_block = pl.cast(cur_phys_block_raw, pl.INDEX)
                    cur_state_row = cur_phys_block * CSA_STATE_BLOCK_SIZE + cur_state_intra
                    pool_kv_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = kv_state_flat[
                        cur_state_row : cur_state_row + 1,
                        HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                    ]
                    pool_score_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = score_state_flat[
                        cur_state_row : cur_state_row + 1,
                        HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                    ]

            for pool_t in pl.range(T):
                if pool_t < num_tokens:
                    pool_pos = pl.read(position_ids, [pool_t])
                    if pool_pos <= write_pos:
                        if pool_pos >= prev_start:
                            if pool_pos < cur_start:
                                pool_slot = pl.cast(pool_pos - prev_start, pl.INDEX)
                                pool_col0 = h0
                            else:
                                pool_slot = pl.cast(COMPRESS_RATIO + pool_pos - cur_start, pl.INDEX)
                                pool_col0 = HEAD_DIM + h0
                            pool_ape_slot = pl.cast(pool_pos % COMPRESS_RATIO, pl.INDEX)
                            pool_ape = ape[pool_ape_slot : pool_ape_slot + 1, pool_col0 : pool_col0 + HEAD_CHUNK]
                            pool_score = pl.add(
                                score_proj[pool_t : pool_t + 1, pool_col0 : pool_col0 + HEAD_CHUNK],
                                pool_ape,
                            )
                            pool_kv_tile[pool_slot : pool_slot + 1, 0:HEAD_CHUNK] = kv_proj[
                                pool_t : pool_t + 1,
                                pool_col0 : pool_col0 + HEAD_CHUNK,
                            ]
                            pool_score_tile[pool_slot : pool_slot + 1, 0:HEAD_CHUNK] = pool_score

            init_slot = STATE_LEN - 1
            mi_buf = pl.create_tensor([1, HEAD_CHUNK], dtype=pl.FP32)
            li_buf = pl.create_tensor([1, HEAD_CHUNK], dtype=pl.FP32)
            oi_buf = pl.create_tensor([1, HEAD_CHUNK], dtype=pl.FP32)
            mi_buf[0:1, 0:HEAD_CHUNK] = pool_score_tile[init_slot : init_slot + 1, 0:HEAD_CHUNK]
            li_buf[0:1, 0:HEAD_CHUNK] = pl.exp(pl.sub(mi_buf[0:1, 0:HEAD_CHUNK], mi_buf[0:1, 0:HEAD_CHUNK]))
            oi_buf[0:1, 0:HEAD_CHUNK] = pool_kv_tile[init_slot : init_slot + 1, 0:HEAD_CHUNK]
            for pool_slot_i in pl.range(STATE_LEN - 1):
                if pool_slot_i >= COMPRESS_RATIO or write_pos >= 2 * COMPRESS_RATIO - 1:
                    mi = mi_buf[0:1, 0:HEAD_CHUNK]
                    li = li_buf[0:1, 0:HEAD_CHUNK]
                    oi = oi_buf[0:1, 0:HEAD_CHUNK]
                    slot_score = pool_score_tile[pool_slot_i : pool_slot_i + 1, 0:HEAD_CHUNK]
                    slot_kv = pool_kv_tile[pool_slot_i : pool_slot_i + 1, 0:HEAD_CHUNK]
                    mi_next = pl.maximum(mi, slot_score)
                    alpha = pl.exp(pl.sub(mi, mi_next))
                    beta = pl.exp(pl.sub(slot_score, mi_next))
                    li_next = pl.add(pl.mul(alpha, li), beta)
                    oi_next = pl.add(pl.mul(oi, alpha), pl.mul(slot_kv, beta))
                    mi_buf[0:1, 0:HEAD_CHUNK] = mi_next
                    li_buf[0:1, 0:HEAD_CHUNK] = li_next
                    oi_buf[0:1, 0:HEAD_CHUNK] = oi_next
            pooled_kv[write_i : write_i + 1, h0 : h0 + HEAD_CHUNK] = pl.div(
                oi_buf[0:1, 0:HEAD_CHUNK],
                li_buf[0:1, 0:HEAD_CHUNK],
            )
        else:
            pooled_kv[write_i : write_i + 1, h0 : h0 + HEAD_CHUNK] = pl.full([1, HEAD_CHUNK], dtype=pl.FP32, value=0.0)

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    for final_block in pl.spmd(MAX_CMP_WRITES // PACKED_RMS_TILE, name_hint="prefill_c4_norm_rope_write"):
        final_base = final_block * PACKED_RMS_TILE
        cos_b = pl.full([PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([PACKED_RMS_TILE, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        for final_dt in pl.range(PACKED_RMS_TILE):
            final_i = final_base + final_dt
            write_token = 0
            write_slot_raw = pl.cast(-1, pl.INT64)
            write_seen = pl.cast(0, pl.INDEX)
            for scan_w in pl.range(T):
                if scan_w < num_tokens:
                    scan_slot_raw = pl.read(cmp_slot_mapping, [scan_w])
                    if scan_slot_raw >= 0:
                        if write_seen == final_i:
                            write_token = scan_w
                            write_slot_raw = scan_slot_raw
                        write_seen = write_seen + 1
            if write_slot_raw >= 0:
                write_pos = pl.read(position_ids, [write_token])
                cmp_pos = pl.cast(write_pos + 1 - COMPRESS_RATIO, pl.INDEX)
                cos_b[final_dt : final_dt + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(
                    freqs_cos[cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2],
                    target_type=pl.FP32,
                )
                sin_b[final_dt : final_dt + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(
                    freqs_sin[cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2],
                    target_type=pl.FP32,
                )

        partial_sq = pl.full([1, PACKED_RMS_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, HEAD_DIM, HEAD_TILE):
            kv_rms_chunk = pooled_kv[final_base : final_base + PACKED_RMS_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, PACKED_RMS_TILE]))
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [PACKED_RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.range(0, NOPE_HEAD_DIM, HEAD_TILE):
            kv_norm_chunk = pooled_kv[final_base : final_base + PACKED_RMS_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[final_base : final_base + PACKED_RMS_TILE, k0 : k0 + HEAD_TILE] = normed_chunk
        kv_rope_norm = pooled_kv[final_base : final_base + PACKED_RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        even_tile = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P0101)
        odd_tile = pl.gather(rope_normed, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_even = pl.sub(pl.mul(even_tile, cos_b), pl.mul(odd_tile, sin_b))
        rope_odd = pl.add(pl.mul(even_tile, sin_b), pl.mul(odd_tile, cos_b))
        rope_buf = pl.full([PACKED_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_buf = pl.tensor.scatter(rope_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=rope_buf)
        rope_buf = pl.tensor.scatter(rope_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=rope_buf)
        normed_kv[final_base : final_base + PACKED_RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM] = rope_buf

    for final_block in pl.spmd(MAX_CMP_WRITES // PACKED_RMS_TILE, name_hint="prefill_c4_cache_write"):
        final_base = final_block * PACKED_RMS_TILE
        for final_dt in pl.range(PACKED_RMS_TILE):
            final_i = final_base + final_dt
            dst_row_raw = pl.cast(-1, pl.INT64)
            write_seen = pl.cast(0, pl.INDEX)
            for scan_w in pl.range(T):
                if scan_w < num_tokens:
                    scan_slot_raw = pl.read(cmp_slot_mapping, [scan_w])
                    if scan_slot_raw >= 0:
                        if write_seen == final_i:
                            dst_row_raw = scan_slot_raw
                        write_seen = write_seen + 1
            if dst_row_raw >= 0:
                dst_row = pl.cast(dst_row_raw, pl.INDEX)
                cmp_kv_flat[dst_row : dst_row + 1, 0:HEAD_DIM] = pl.cast(
                    normed_kv[final_i : final_i + 1, 0:HEAD_DIM],
                    target_type=pl.BF16,
                    mode="rint",
                )
            else:
                keepalive_row = PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE - MAX_CMP_WRITES + final_i
                cmp_kv_flat[keepalive_row : keepalive_row + 1, 0:HEAD_DIM] = cmp_kv_flat[
                    keepalive_row : keepalive_row + 1,
                    0:HEAD_DIM,
                ]

    # State writeback: one SPMD task per token (was per token x out-block =
    # T*PACKED_PROJ_BLOCKS tiny tasks). The per-token guard is checked
    # once so a skipped token costs one empty task instead of PACKED_PROJ_BLOCKS
    # of them; out-blocks are looped inside the task at the OUT_CHUNK width.
    # pool_dep keeps the (zero-weighted) ordering after the pool and is hoisted
    # to once per token.
    for update_t in pl.spmd(T, name_hint="prefill_c4_state_update"):
        if update_t < num_tokens:
            state_row_raw = pl.read(state_slot_mapping, [update_t])
            if state_row_raw >= 0:
                state_row = pl.cast(state_row_raw, pl.INDEX)
                update_pos = pl.read(position_ids, [update_t])
                ape_slot = pl.cast(update_pos % COMPRESS_RATIO, pl.INDEX)
                pool_dep = pl.mul(pooled_kv[0:1, 0:OUT_CHUNK], 0.0)
                for update_ob in pl.range(PACKED_PROJ_BLOCKS):
                    update_o0 = update_ob * OUT_CHUNK
                    ape_row = ape[ape_slot : ape_slot + 1, update_o0 : update_o0 + OUT_CHUNK]
                    kv_state_flat[state_row : state_row + 1, update_o0 : update_o0 + OUT_CHUNK] = pl.add(
                        kv_proj[
                            update_t : update_t + 1,
                            update_o0 : update_o0 + OUT_CHUNK,
                        ],
                        pool_dep,
                    )
                    score_state_flat[state_row : state_row + 1, update_o0 : update_o0 + OUT_CHUNK] = pl.add(
                        pl.add(
                            score_proj[update_t : update_t + 1, update_o0 : update_o0 + OUT_CHUNK],
                            ape_row,
                        ),
                        pool_dep,
                    )

    cmp_kv = pl.reshape(cmp_kv_flat, [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv_state = pl.reshape(kv_state_flat, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM])
    return cmp_kv, kv_state, score_state


def golden_prefill_compressor_ratio4(tensors):
    """Packed token-major torch reference for ratio-4 prefill compressor."""
    import torch

    x = tensors["x"].view(T, D).float()
    kv_state_flat = tensors["kv_state"].view(CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, OUT_DIM)
    score_state_flat = tensors["score_state"].view(CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE, OUT_DIM)
    state_block_table = tensors["compress_state_block_table"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cmp_kv = tensors["cmp_kv"]
    cache_rows = cmp_kv.view(cmp_kv.shape[0] * BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    position_ids = tensors["position_ids"]

    kv_proj = x @ wkv
    score_proj = x @ wgate

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * CSA_STATE_BLOCK_SIZE + intra

    for token_id in range(int(tensors["num_tokens"])):
        dst_row = int(tensors["cmp_slot_mapping"][token_id].item())
        if dst_row < 0:
            continue
        write_pos = int(position_ids[token_id].item())
        cur_start = write_pos + 1 - COMPRESS_RATIO
        prev_start = cur_start - COMPRESS_RATIO
        pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
        pool_score = torch.full((STATE_LEN, HEAD_DIM), float("-inf"), dtype=torch.float32)

        for s in range(COMPRESS_RATIO):
            prev_abs = prev_start + s
            if write_pos >= 2 * COMPRESS_RATIO - 1:
                prev_row = state_row(prev_abs)
                if prev_row >= 0:
                    pool_kv[s] = kv_state_flat[prev_row, :HEAD_DIM]
                    pool_score[s] = score_state_flat[prev_row, :HEAD_DIM]

            cur_abs = cur_start + s
            cur_row = state_row(cur_abs)
            if cur_row >= 0:
                pool_kv[COMPRESS_RATIO + s] = kv_state_flat[cur_row, HEAD_DIM:OUT_DIM]
                pool_score[COMPRESS_RATIO + s] = score_state_flat[cur_row, HEAD_DIM:OUT_DIM]

        for t in range(int(tensors["num_tokens"])):
            pos = int(position_ids[t].item())
            if pos < prev_start or pos > write_pos:
                continue
            if pos < cur_start:
                pool_slot = pos - prev_start
                col0 = 0
            else:
                pool_slot = COMPRESS_RATIO + pos - cur_start
                col0 = HEAD_DIM
            ape_slot = pos % COMPRESS_RATIO
            pool_kv[pool_slot] = kv_proj[t, col0 : col0 + HEAD_DIM]
            pool_score[pool_slot] = score_proj[t, col0 : col0 + HEAD_DIM] + ape[ape_slot, col0 : col0 + HEAD_DIM]

        init_slot = STATE_LEN - 1
        mi = pool_score[init_slot : init_slot + 1].clone()
        li = torch.exp(mi - mi)
        oi = pool_kv[init_slot : init_slot + 1].clone()
        for slot_i in range(STATE_LEN - 1):
            if slot_i < COMPRESS_RATIO and write_pos < 2 * COMPRESS_RATIO - 1:
                continue
            slot_score = pool_score[slot_i : slot_i + 1]
            slot_kv = pool_kv[slot_i : slot_i + 1]
            mi_next = torch.maximum(mi, slot_score)
            alpha = torch.exp(mi - mi_next)
            beta = torch.exp(slot_score - mi_next)
            li = alpha * li + beta
            oi = oi * alpha + slot_kv * beta
            mi = mi_next
        pooled = oi / li
        inv_rms = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = pooled * inv_rms * norm_w.float().view(1, HEAD_DIM)
        rope_pair = normed[..., NOPE_HEAD_DIM:HEAD_DIM].unflatten(-1, (-1, 2))
        rope_even = rope_pair[..., 0]
        rope_odd = rope_pair[..., 1]
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2].float()
        rot_even = rope_even * cos - rope_odd * sin
        rot_odd = rope_even * sin + rope_odd * cos
        normed[:, NOPE_HEAD_DIM:HEAD_DIM] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cache_rows[dst_row] = normed.to(torch.bfloat16)[0]

    for t in range(int(tensors["num_tokens"])):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        ape_slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][ape_slot]
    tensors["cmp_kv"][:] = cmp_kv
    tensors["kv_state"][:] = kv_state_flat.view_as(tensors["kv_state"])
    tensors["score_state"][:] = score_state_flat.view_as(tensors["score_state"])


@pl.jit
def prefill_compressor_ratio4_test(
    x: pl.Tensor[[T, D], pl.BF16],
    kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Out[pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    return prefill_compressor_ratio4(
        x, kv_state, score_state, compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv, position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    if start_pos < 0 or start_pos + T > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - T}, got {start_pos}")

    def init_compress_state_block_table():
        table = torch.full((CSA_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(CSA_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % CSA_STATE_MAX_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        return int(table[block].item()) * CSA_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM)
        flat = state.view(-1, OUT_DIM)
        for abs_pos in range(max(0, start_pos - STATE_LEN), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(OUT_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash CSA (ratio-4) compressor (mean l8/l32 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors decode_compressor_ratio4.
    def init_wkv():
        return torch.randn(D, OUT_DIM) * 0.0245
    def init_wgate():
        return torch.randn(D, OUT_DIM) * 0.0388
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.1243
    def init_norm_w():
        return 0.9666 + 0.1929 * torch.randn(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_kv():
        return torch.zeros(PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + T, dtype=torch.int32)
    def init_cmp_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            pos = start_pos + t
            if (pos + 1) % COMPRESS_RATIO == 0:
                dst_row = (pos + 1) // COMPRESS_RATIO - 1
                if dst_row >= PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE:
                    raise ValueError("fixture compressed slot exceeds standalone cmp_kv capacity")
                mapping[t] = dst_row
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            mapping[t] = state_row(start_pos + t)
        return mapping

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM], torch.float32, init_value=init_state, is_output=True),
        TensorSpec("score_state", [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, OUT_DIM], torch.float32, init_value=init_state, is_output=True),
        TensorSpec("compress_state_block_table", [CSA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv", [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv, is_output=True),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, T),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill compressor ratio4 validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only absolute position for token 0; lowered into position_ids and dense cmp_slot_mapping.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_compressor_ratio4_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_compressor_ratio4,
        runtime_cfg=dict(platform=args.platform, device_id=args.device, enable_l2_swimlane=args.enable_l2_swimlane),
        compile_only=args.compile_only,
        compare_fn={
            "kv_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "score_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
