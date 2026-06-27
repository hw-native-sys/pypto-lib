# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 single-request token-major prefill compressor, ratio=128.

The public contract is single-request token-major prefill: the
layer owns the per-request loop and feeds this op one contiguous run of <=T
tokens.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, PREFILL_BATCH, PREFILL_SEQ


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 128
OUT_DIM = HEAD_DIM
STATE_LEN = COMPRESS_RATIO
START_POS = 0

K_TILE = 512
OUT_TILE = 32
HEAD_TILE = 64
K_BLOCKS = D // K_TILE
OUT_BLOCKS = OUT_DIM // OUT_TILE
HEAD_BLOCKS = HEAD_DIM // HEAD_TILE

assert S == COMPRESS_RATIO, "ratio128 prefill compressor bring-up expects one full compression chunk"


HCA_STATE_BLOCK_SIZE = 8
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = HCA_STATE_MAX_BLOCKS
MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)
# Compressed-KV cache pool consumed by prefill_sparse_attn: one BLOCK_SIZE page
# per PREFILL_MAX_COMPRESSED compressed entries.
PREFILL_MAX_COMPRESSED = max(1, min(M.index_topk, M.sliding_window + M.sliding_window // 2))
HCA_CMP_BLOCK_NUM = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
HCA_KV_STORE_TILE = 16
HCA_C128_RMS_TILE = 8
HCA_C128_RMS_PAD_ROWS = HCA_C128_RMS_TILE

PACKED_C128_PROJ_BLOCKS = OUT_BLOCKS
PACKED_C128_POOL_BLOCKS = MAX_CMP_WRITES * HEAD_BLOCKS


@pl.jit.inline
def prefill_compressor_ratio128(
    x: pl.Tensor[[T, D], pl.BF16],
    kv_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    x_flat = x
    kv_proj_scratch = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    score_proj_scratch = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
    kv_state_flat = pl.reshape(kv_state, [HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, OUT_DIM])
    score_state_flat = pl.reshape(score_state, [HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, OUT_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    pooled_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    normed_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_norm_pad_init"):
        for init_hb in pl.pipeline(HEAD_BLOCKS, stage=2):
            init_h0 = init_hb * HEAD_TILE
            zero_chunk = pl.full([HCA_C128_RMS_TILE, HEAD_TILE], dtype=pl.FP32, value=0.0)
            pooled_kv_pad[0:HCA_C128_RMS_TILE, init_h0 : init_h0 + HEAD_TILE] = zero_chunk
            normed_kv_pad[0:HCA_C128_RMS_TILE, init_h0 : init_h0 + HEAD_TILE] = zero_chunk

    for proj_idx in pl.spmd(PACKED_C128_PROJ_BLOCKS, name_hint="prefill_hca_c128_kv_score_proj"):
        o0 = proj_idx * OUT_TILE
        kv_acc = pl.create_tensor([T, OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([T, OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, K_BLOCKS, stage=2):
            k0 = kb * K_TILE
            x_tile = x_flat[0:T, k0 : k0 + K_TILE]
            wkv_tile = wkv[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            wgate_tile = wgate[k0 : k0 + K_TILE, o0 : o0 + OUT_TILE]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)
        kv_proj_scratch[0:T, o0 : o0 + OUT_TILE] = kv_acc
        score_proj_scratch[0:T, o0 : o0 + OUT_TILE] = score_acc

    # Precompute write_i -> (position, dst cache row) once (input-only deps -> overlaps the matmul),
    # replacing the O(T) write-discovery scan in pool / rmsnorm_rope / kv_finalize. Sized to
    # HCA_C128_RMS_TILE because rmsnorm_rope indexes padded rows beyond MAX_CMP_WRITES (rest stay -1).
    write_pos_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    write_dst_map = pl.create_tensor([1, HCA_C128_RMS_TILE], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_write_map"):
        write_pos_map[0:1, 0:HCA_C128_RMS_TILE] = pl.full([1, HCA_C128_RMS_TILE], dtype=pl.INT32, value=0)
        write_dst_map[0:1, 0:HCA_C128_RMS_TILE] = pl.full([1, HCA_C128_RMS_TILE], dtype=pl.INT32, value=-1)
        map_seen = pl.cast(0, pl.INDEX)
        for map_w in pl.range(T):
            if map_w < num_tokens:
                map_slot_raw = pl.read(cmp_slot_mapping, [map_w])
                if map_slot_raw >= 0:
                    pl.write(write_pos_map, [0, map_seen], pl.read(position_ids, [map_w]))
                    pl.write(write_dst_map, [0, map_seen], pl.cast(map_slot_raw, pl.INT32))
                    map_seen = map_seen + 1

    # State scatter (decode order): write every token's raw projection (+APE on score) into
    # paged kv_state/score_state BEFORE pooling, so softmax_pool reads its window straight from
    # state (no seed+overlay, no pool_dep ordering hack). pool depends on this via kv_state RAW.
    for scatter_t in pl.spmd(T, name_hint="prefill_hca_c128_state_scatter_pre"):
        if scatter_t < num_tokens:
            scatter_row_raw = pl.read(state_slot_mapping, [scatter_t])
            if scatter_row_raw >= 0:
                scatter_row = pl.cast(scatter_row_raw, pl.INDEX)
                scatter_pos = pl.read(position_ids, [scatter_t])
                scatter_ape_slot = pl.cast(scatter_pos % COMPRESS_RATIO, pl.INDEX)
                kv_state_flat[scatter_row : scatter_row + 1, 0:OUT_DIM] = kv_proj_scratch[scatter_t : scatter_t + 1, 0:OUT_DIM]
                score_state_flat[scatter_row : scatter_row + 1, 0:OUT_DIM] = pl.add(
                    score_proj_scratch[scatter_t : scatter_t + 1, 0:OUT_DIM],
                    ape[scatter_ape_slot : scatter_ape_slot + 1, 0:OUT_DIM],
                )

    for pool_idx in pl.spmd(PACKED_C128_POOL_BLOCKS, name_hint="prefill_hca_c128_softmax_pool"):
        write_i = pool_idx // HEAD_BLOCKS
        hb = pool_idx - write_i * HEAD_BLOCKS
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
                    pool_kv_tile[pool_state_i : pool_state_i + 1, 0:HEAD_TILE] = kv_state_flat[
                        pool_state_row : pool_state_row + 1,
                        h0 : h0 + HEAD_TILE,
                    ]
                    pool_score_tile[pool_state_i : pool_state_i + 1, 0:HEAD_TILE] = score_state_flat[
                        pool_state_row : pool_state_row + 1,
                        h0 : h0 + HEAD_TILE,
                    ]
            # Vectorized softmax over all STATE_LEN slots (matches decode128): transpose the
            # assembled [STATE_LEN, HEAD_TILE] tile and do row_max/exp/sum/div + weighted sum,
            # replacing the STATE_LEN-1 serial online-flash fold. Same result, no long chain.
            pool_score_t = pl.transpose(pool_score_tile, axis1=0, axis2=1)
            pool_kv_t = pl.transpose(pool_kv_tile, axis1=0, axis2=1)
            score_max = pl.row_max(pool_score_t)
            score_exp = pl.exp(pl.row_expand_sub(pool_score_t, score_max))
            score_sum = pl.row_sum(score_exp)
            score_prob = pl.row_expand_div(score_exp, score_sum)
            pooled_chunk_t = pl.row_sum(pl.mul(pool_kv_t, score_prob))
            pooled_kv_pad[write_i : write_i + 1, h0 : h0 + HEAD_TILE] = pl.reshape(pooled_chunk_t, [1, HEAD_TILE])
        else:
            pooled_kv_pad[write_i : write_i + 1, h0 : h0 + HEAD_TILE] = pooled_kv_pad[
                write_i : write_i + 1,
                h0 : h0 + HEAD_TILE,
            ]

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_rmsnorm_rope"):
        cos_b = pl.full([HCA_C128_RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([HCA_C128_RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        for norm_i in pl.range(HCA_C128_RMS_TILE):
            norm_slot_raw = pl.read(write_dst_map, [0, norm_i])
            if norm_slot_raw >= 0:
                norm_cmp_pos = pl.cast(pl.read(write_pos_map, [0, norm_i]) + 1 - COMPRESS_RATIO, pl.INDEX)
                cos_row = pl.cast(freqs_cos[norm_cmp_pos : norm_cmp_pos + 1, 0:ROPE_HALF], target_type=pl.FP32)
                sin_row = pl.cast(freqs_sin[norm_cmp_pos : norm_cmp_pos + 1, 0:ROPE_HALF], target_type=pl.FP32)
                cos_b[norm_i : norm_i + 1, 0:ROPE_HALF] = cos_row
                sin_b[norm_i : norm_i + 1, 0:ROPE_HALF] = sin_row
        partial_sq = pl.full([1, HCA_C128_RMS_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_BLOCKS, stage=2):
            rms_h0 = rms_kb * HEAD_TILE
            kv_rms_chunk = pooled_kv_pad[0:HCA_C128_RMS_TILE, rms_h0 : rms_h0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, HCA_C128_RMS_TILE]))

        variance = pl.reshape(pl.add(pl.mul(partial_sq, 1.0 / HEAD_DIM), EPS), [HCA_C128_RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for norm_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_TILE, stage=2):
            norm_h0 = norm_kb * HEAD_TILE
            kv_norm_chunk = pooled_kv_pad[0:HCA_C128_RMS_TILE, norm_h0 : norm_h0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, norm_h0 : norm_h0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv_pad[0:HCA_C128_RMS_TILE, norm_h0 : norm_h0 + HEAD_TILE] = normed_chunk

        kv_rope = pooled_kv_pad[0:HCA_C128_RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM], pl.FP32)
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope, inv_rms), gamma_rope)
        # A3 interleaved swap-gather (matches decode): single data gather + sign trick instead of
        # the P0101/P1010 de-interleave gather + rotate + re-interleave scatter.
        # out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]; idx built in-kernel from pl.arange.
        rope_ones = pl.full([HCA_C128_RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_dup_idx = pl.cast(rope_dup_f, target_type=pl.INT32)                                       # j>>1
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))                                          # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        rope_sign = pl.sub(pl.mul(rope_lane, 2.0), 1.0)                                                # [-1,+1,...]
        cos_il = pl.gather(cos_b, dim=-1, index=rope_dup_idx)
        sin_il = pl.gather(sin_b, dim=-1, index=rope_dup_idx)
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_il), pl.mul(pl.mul(swapped, rope_sign), sin_il))
        normed_kv_pad[0:HCA_C128_RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_rot

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_kv_finalize"):
        for final_i in pl.range(MAX_CMP_WRITES):
            final_cmp_row_raw = pl.read(write_dst_map, [0, final_i])
            if final_cmp_row_raw >= 0:
                final_cmp_row = pl.cast(final_cmp_row_raw, pl.INDEX)
                for final_hb in pl.range(HEAD_BLOCKS):
                    final_h0 = final_hb * HEAD_TILE
                    final_chunk = normed_kv_pad[final_i : final_i + 1, final_h0 : final_h0 + HEAD_TILE]
                    cmp_kv_flat[final_cmp_row : final_cmp_row + 1, final_h0 : final_h0 + HEAD_TILE] = pl.cast(
                        final_chunk,
                        target_type=pl.BF16,
                        mode="rint",
                    )

    cmp_kv = pl.reshape(cmp_kv_flat, [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv_state = pl.reshape(kv_state_flat, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM])
    return cmp_kv, kv_state, score_state


@pl.jit
def prefill_compressor_ratio128_test(
    x: pl.Tensor[[T, D], pl.BF16],
    kv_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    position_ids: pl.Tensor[[T], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
):
    return prefill_compressor_ratio128(
        x, kv_state, score_state, compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv, position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )


def golden_prefill_compressor_ratio128(tensors):
    import torch

    num_tokens = int(tensors["num_tokens"])
    kv_proj = tensors["x"].float() @ tensors["wkv"].float()
    score_proj = tensors["x"].float() @ tensors["wgate"].float()
    kv_state_flat = tensors["kv_state"].view(HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, OUT_DIM)
    score_state_flat = tensors["score_state"].view(HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, OUT_DIM)
    state_block_table = tensors["compress_state_block_table"]
    cmp_kv_flat = tensors["cmp_kv"].view(HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * HCA_STATE_BLOCK_SIZE + intra

    for token_id in range(num_tokens):
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
        for t in range(num_tokens):
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
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
        normed[:, NOPE_HEAD_DIM:] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cmp_kv_flat[dst_row] = normed[0]

    for t in range(num_tokens):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][slot]


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    num_tokens = T
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + num_tokens > MAX_SEQ_LEN:
        raise ValueError("start_pos + num_tokens exceeds max_position_embeddings")

    def init_compress_state_block_table():
        table = torch.full((HCA_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(HCA_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % HCA_STATE_MAX_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(table[block].item()) * HCA_STATE_BLOCK_SIZE + intra
    def init_x():
        return ((torch.rand(T, D) - 0.5) * 0.1).to(torch.bfloat16)
    def init_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM)
        for abs_pos in range(max(0, start_pos - COMPRESS_RATIO), start_pos):
            row = state_row(abs_pos)
            if row >= 0:
                state.view(-1, OUT_DIM)[row] = (torch.rand(OUT_DIM) - 0.5) * 0.05
        return state
    # Calibrated to the real DeepSeek-V4-Flash HCA (ratio-128) main compressor (mean l7/l9 of
    # extract_weights_flash): zero-mean Gaussian BF16 weights at the measured std; the RMSNorm
    # gamma centers near the measured mean (not ones / not uniform). Mirrors decode_compressor_ratio128.
    def init_wkv():
        return torch.randn(D, OUT_DIM) * 0.0246
    def init_wgate():
        return torch.randn(D, OUT_DIM) * 0.0316
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.0340
    def init_norm_w():
        return 0.1001 + 0.0549 * torch.randn(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_kv():
        return torch.zeros(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + T, dtype=torch.int32)
    def init_cmp_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for token_id in range(num_tokens):
            pos = start_pos + token_id
            if pos + 1 >= COMPRESS_RATIO and (pos + 1) % COMPRESS_RATIO == 0:
                mapping[token_id] = (pos + 1) // COMPRESS_RATIO - 1
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for token_id in range(num_tokens):
            mapping[token_id] = state_row(start_pos + token_id)
        return mapping

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM], torch.float32, init_value=init_state, is_output=True),
        TensorSpec("score_state", [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, OUT_DIM], torch.float32, init_value=init_state, is_output=True),
        TensorSpec("compress_state_block_table", [HCA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv", [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv, is_output=True),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill compressor ratio128 validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help=(
            "Fixture-only absolute position for token 0. It is lowered into position_ids and compressed write "
            "slot mapping; it is not a JIT kernel parameter."
        ),
    )
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_compressor_ratio128_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_compressor_ratio128,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device, enable_l2_swimlane=args.enable_l2_swimlane),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only,
        compare_fn={
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "kv_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "score_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
