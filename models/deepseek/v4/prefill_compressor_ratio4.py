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
from prefill_sparse_attn import HCA_CMP_BLOCK_NUM as PREFILL_CMP_BLOCK_NUM

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

MAX_REQS = 2
MAX_TOKENS = B * S
MAX_CMP_WRITES = MAX_REQS * max(1, MAX_TOKENS // COMPRESS_RATIO)
PACKED_PROJ_BLOCKS = OUT_DIM // OUT_CHUNK
PACKED_POOL_BLOCKS = MAX_CMP_WRITES * HEAD_BLOCKS
PACKED_STATE_UPDATE_TILE = 16
PACKED_RMS_TILE = 16


@pl.jit.inline
def prefill_compressor_ratio4(
    x: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    kv_state: pl.Tensor[[MAX_REQS, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[MAX_REQS, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    num_cmp_writes: pl.Scalar[pl.INT32],
    cmp_write_token_ids: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
):
    kv_proj = pl.create_tensor([MAX_TOKENS, OUT_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([MAX_TOKENS, OUT_DIM], dtype=pl.FP32)
    kv_state_flat = pl.reshape(kv_state, [MAX_REQS * STATE_LEN, OUT_DIM])
    score_state_flat = pl.reshape(score_state, [MAX_REQS * STATE_LEN, OUT_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    pooled_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)
    normed_kv = pl.create_tensor([MAX_CMP_WRITES, HEAD_DIM], dtype=pl.FP32)

    for proj_idx in pl.spmd(PACKED_PROJ_BLOCKS, name_hint="prefill_c4_state_proj"):
        o0 = proj_idx * OUT_CHUNK
        kv_acc = pl.create_tensor([MAX_TOKENS, OUT_CHUNK], dtype=pl.FP32)
        score_acc = pl.create_tensor([MAX_TOKENS, OUT_CHUNK], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_CHUNK, stage=2):
            k0 = kb * K_CHUNK
            x_tile = x[0:MAX_TOKENS, k0 : k0 + K_CHUNK]
            wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
            wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)
        kv_proj[0:MAX_TOKENS, o0 : o0 + OUT_CHUNK] = kv_acc
        score_proj[0:MAX_TOKENS, o0 : o0 + OUT_CHUNK] = score_acc

    for pool_idx in pl.spmd(PACKED_POOL_BLOCKS, name_hint="prefill_c4_softmax_pool"):
        write_i = pool_idx // HEAD_BLOCKS
        hb = pool_idx - write_i * HEAD_BLOCKS
        h0 = hb * HEAD_CHUNK
        pool_kv_tile = pl.create_tensor([STATE_LEN, HEAD_CHUNK], dtype=pl.FP32)
        pool_score_tile = pl.create_tensor([STATE_LEN, HEAD_CHUNK], dtype=pl.FP32)
        if write_i < num_cmp_writes:
            write_token = pl.cast(pl.read(cmp_write_token_ids, [write_i]), pl.INDEX)
            write_pos = pl.read(position_ids, [write_token])
            req = pl.cast(pl.read(token_to_request, [write_token]), pl.INDEX)
            cur_start = write_pos + 1 - COMPRESS_RATIO
            prev_start = cur_start - COMPRESS_RATIO
            state_row0 = req * STATE_LEN
            for pool_s in pl.range(COMPRESS_RATIO):
                prev_abs = prev_start + pool_s
                front_slot = pool_s
                if write_pos >= 2 * COMPRESS_RATIO - 1:
                    prev_state_slot = pl.cast(prev_abs % STATE_LEN, pl.INDEX)
                    prev_state_row = state_row0 + prev_state_slot
                    pool_kv_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = kv_state_flat[
                        prev_state_row : prev_state_row + 1,
                        h0 : h0 + HEAD_CHUNK,
                    ]
                    pool_score_tile[front_slot : front_slot + 1, 0:HEAD_CHUNK] = score_state_flat[
                        prev_state_row : prev_state_row + 1,
                        h0 : h0 + HEAD_CHUNK,
                    ]
                else:
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

                cur_abs = cur_start + pool_s
                back_slot = COMPRESS_RATIO + pool_s
                cur_state_slot = pl.cast(cur_abs % STATE_LEN, pl.INDEX)
                cur_state_row = state_row0 + cur_state_slot
                pool_kv_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = kv_state_flat[
                    cur_state_row : cur_state_row + 1,
                    HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                ]
                pool_score_tile[back_slot : back_slot + 1, 0:HEAD_CHUNK] = score_state_flat[
                    cur_state_row : cur_state_row + 1,
                    HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_CHUNK,
                ]

            for pool_t in pl.range(MAX_TOKENS):
                if pool_t < num_tokens:
                    pool_req = pl.cast(pl.read(token_to_request, [pool_t]), pl.INDEX)
                    pool_pos = pl.read(position_ids, [pool_t])
                    if pool_req == req:
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
            if final_i < num_cmp_writes:
                write_token = pl.cast(pl.read(cmp_write_token_ids, [final_i]), pl.INDEX)
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
            gamma = norm_w_2d[:, k0 : k0 + HEAD_TILE]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[final_base : final_base + PACKED_RMS_TILE, k0 : k0 + HEAD_TILE] = normed_chunk
        kv_rope_norm = pooled_kv[final_base : final_base + PACKED_RMS_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM]
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
            if final_i < num_cmp_writes:
                dst_row = pl.cast(pl.read(cmp_slot_mapping, [final_i]), pl.INDEX)
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

    for update_idx in pl.spmd(MAX_REQS * STATE_LEN * PACKED_PROJ_BLOCKS, name_hint="prefill_c4_state_update"):
        update_ob = update_idx % PACKED_PROJ_BLOCKS
        update_slot_tmp = update_idx // PACKED_PROJ_BLOCKS
        update_slot = update_slot_tmp % STATE_LEN
        update_req = update_slot_tmp // STATE_LEN
        update_o0 = update_ob * OUT_CHUNK
        state_row = update_req * STATE_LEN + update_slot
        latest_pos = pl.cast(-1, pl.INT32)
        latest_token = 0
        for scan_t in pl.range(MAX_TOKENS):
            if scan_t < num_tokens:
                scan_req = pl.cast(pl.read(token_to_request, [scan_t]), pl.INDEX)
                if scan_req == update_req:
                    scan_pos = pl.read(position_ids, [scan_t])
                    scan_slot = pl.cast(scan_pos % STATE_LEN, pl.INDEX)
                    if scan_slot == update_slot:
                        if scan_pos > latest_pos:
                            latest_pos = scan_pos
                            latest_token = scan_t
        if latest_pos >= 0:
            ape_slot = pl.cast(latest_pos % COMPRESS_RATIO, pl.INDEX)
            ape_row = ape[ape_slot : ape_slot + 1, update_o0 : update_o0 + OUT_CHUNK]
            pool_dep = pl.mul(pooled_kv[0:1, 0:OUT_CHUNK], 0.0)
            kv_state_flat[state_row : state_row + 1, update_o0 : update_o0 + OUT_CHUNK] = pl.add(
                kv_proj[
                    latest_token : latest_token + 1,
                    update_o0 : update_o0 + OUT_CHUNK,
                ],
                pool_dep,
            )
            score_state_flat[state_row : state_row + 1, update_o0 : update_o0 + OUT_CHUNK] = pl.add(
                pl.add(
                    score_proj[latest_token : latest_token + 1, update_o0 : update_o0 + OUT_CHUNK],
                    ape_row,
                ),
                pool_dep,
            )

    cmp_kv = pl.reshape(cmp_kv_flat, [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv_state = pl.reshape(kv_state_flat, [MAX_REQS, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [MAX_REQS, STATE_LEN, OUT_DIM])
    return cmp_kv, kv_state, score_state


def golden_prefill_compressor_ratio4(tensors):
    """Packed token-major torch reference for ratio-4 prefill compressor."""
    import torch

    x = tensors["x"].view(MAX_TOKENS, D).float()
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"]
    norm_w = tensors["norm_w"]
    cmp_kv = tensors["cmp_kv"]
    cache_rows = cmp_kv.view(cmp_kv.shape[0] * BLOCK_SIZE, 1, HEAD_DIM)[:, 0, :]
    token_to_req = tensors["token_to_request"]
    position_ids = tensors["position_ids"]

    kv_proj = x @ wkv
    score_proj = x @ wgate

    for write_i in range(int(tensors["num_cmp_writes"])):
        token_id = int(tensors["cmp_write_token_ids"][write_i].item())
        req = int(token_to_req[token_id].item())
        write_pos = int(position_ids[token_id].item())
        cur_start = write_pos + 1 - COMPRESS_RATIO
        prev_start = cur_start - COMPRESS_RATIO
        pool_kv = torch.zeros(STATE_LEN, HEAD_DIM, dtype=torch.float32)
        pool_score = torch.full((STATE_LEN, HEAD_DIM), float("-inf"), dtype=torch.float32)

        for s in range(COMPRESS_RATIO):
            prev_abs = prev_start + s
            if write_pos >= 2 * COMPRESS_RATIO - 1:
                prev_slot = prev_abs % STATE_LEN
                pool_kv[s] = kv_state[req, prev_slot, :HEAD_DIM]
                pool_score[s] = score_state[req, prev_slot, :HEAD_DIM]

            cur_abs = cur_start + s
            cur_slot = cur_abs % STATE_LEN
            pool_kv[COMPRESS_RATIO + s] = kv_state[req, cur_slot, HEAD_DIM:OUT_DIM]
            pool_score[COMPRESS_RATIO + s] = score_state[req, cur_slot, HEAD_DIM:OUT_DIM]

        for t in range(int(tensors["num_tokens"])):
            if int(token_to_req[t].item()) != req:
                continue
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
        dst_row = int(tensors["cmp_slot_mapping"][write_i].item())
        cache_rows[dst_row] = normed.to(torch.bfloat16)[0]

    for t in range(int(tensors["num_tokens"])):
        req = int(tensors["token_to_request"][t].item())
        pos = int(tensors["position_ids"][t].item())
        slot = pos % STATE_LEN
        ape_slot = pos % COMPRESS_RATIO
        kv_state[req, slot] = kv_proj[t]
        score_state[req, slot] = score_proj[t] + tensors["ape"][ape_slot]
    tensors["cmp_kv"][:] = cmp_kv
    tensors["kv_state"][:] = kv_state
    tensors["score_state"][:] = score_state


@pl.jit
def prefill_compressor_ratio4_test(
    x: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    kv_state: pl.Tensor[[MAX_REQS, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[MAX_REQS, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Out[pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    num_cmp_writes: pl.Scalar[pl.INT32],
    cmp_write_token_ids: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
):
    return prefill_compressor_ratio4(
        x, kv_state, score_state, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv, token_to_request, position_ids, num_tokens, num_cmp_writes,
        cmp_write_token_ids, cmp_slot_mapping,
    )


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec

    if start_pos < 0 or start_pos + MAX_TOKENS > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - MAX_TOKENS}, got {start_pos}")

    cmp_write_records = [
        (t, (start_pos + t + 1) // COMPRESS_RATIO - 1)
        for t in range(MAX_TOKENS)
        if (start_pos + t + 1) % COMPRESS_RATIO == 0
    ]
    if len(cmp_write_records) > MAX_CMP_WRITES:
        raise ValueError(f"fixture generated {len(cmp_write_records)} compressed writes, cap is {MAX_CMP_WRITES}")
    if cmp_write_records and max(dst_row for _, dst_row in cmp_write_records) >= PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE:
        raise ValueError("fixture compressed slot exceeds standalone cmp_kv capacity")

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale
    def init_x():
        return seeded_uniform((MAX_TOKENS, D), 1, 0.1).to(torch.bfloat16)
    def init_state():
        return torch.zeros(MAX_REQS, STATE_LEN, OUT_DIM)
    def init_wkv():
        return seeded_uniform((D, OUT_DIM), 2, D ** -0.5).to(torch.bfloat16)
    def init_wgate():
        return seeded_uniform((D, OUT_DIM), 3, D ** -0.5).to(torch.bfloat16)
    def init_ape():
        return seeded_uniform((COMPRESS_RATIO, OUT_DIM), 4, 0.01)
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_cmp_kv():
        return torch.zeros(PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    def init_token_to_request():
        return torch.zeros(MAX_TOKENS, dtype=torch.int32)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + MAX_TOKENS, dtype=torch.int32)
    def init_cmp_write_token_ids():
        ids = torch.zeros(MAX_CMP_WRITES, dtype=torch.int32)
        for i, (token_id, _) in enumerate(cmp_write_records):
            ids[i] = token_id
        return ids
    def init_cmp_slot_mapping():
        mapping = torch.zeros(MAX_CMP_WRITES, dtype=torch.int32)
        for i, (_, dst_row) in enumerate(cmp_write_records):
            mapping[i] = dst_row
        return mapping

    return [
        TensorSpec("x", [MAX_TOKENS, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [MAX_REQS, STATE_LEN, OUT_DIM], torch.float32, init_value=init_state, is_output=True),
        TensorSpec("score_state", [MAX_REQS, STATE_LEN, OUT_DIM], torch.float32, init_value=init_state, is_output=True),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv", [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv, is_output=True),
        TensorSpec("token_to_request", [MAX_TOKENS], torch.int32, init_value=init_token_to_request),
        TensorSpec("position_ids", [MAX_TOKENS], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, MAX_TOKENS),
        ScalarSpec("num_cmp_writes", torch.int32, len(cmp_write_records)),
        TensorSpec("cmp_write_token_ids", [MAX_CMP_WRITES], torch.int32, init_value=init_cmp_write_token_ids),
        TensorSpec("cmp_slot_mapping", [MAX_CMP_WRITES], torch.int32, init_value=init_cmp_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill compressor ratio4 validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Fixture-only absolute position for token 0; lowered into position_ids for the kernel.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_compressor_ratio4_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_compressor_ratio4,
        runtime_cfg=dict(platform=args.platform, device_id=args.device, enable_l2_swimlane=args.enable_l2_swimlane),
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
