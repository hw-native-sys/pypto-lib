# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill SWA attention.

This kernel uses the current [PREFILL_BATCH, PREFILL_SEQ] kernel shape from
config through prefill_qkv_proj_rope. Q/KV projection is shared with
prefill_qkv_proj_rope; SWA attention reads the previous sliding-window cache
when `start_pos > 0`, uses current KV for keys inside this invocation, and
writes the current KV after attention.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ
from prefill_hc_post import golden_prefill_hc_post, prefill_hc_post
from prefill_hc_pre import golden_prefill_hc_pre, prefill_hc_pre
from prefill_qkv_proj_rope import (
    golden_prefill_attn_norm,
    golden_prefill_qkv_proj_rope,
    prefill_attn_norm,
    prefill_qkv_proj_rope_core,
)
from prefill_sparse_attn import golden_prefill_sparse_attn, prefill_sparse_attn


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HEAD_DIM = ROPE_DIM
NOPE_DIM = M.nope_head_dim
NOPE_HEAD_DIM = NOPE_DIM
Q_LORA = M.q_lora_rank
ROPE_HALF = ROPE_DIM // 2
HALF_ROPE = ROPE_HALF
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# SWA cache/topk contract. The ratio-0 path has only the sliding-window cache.
ORI_MAX_BLOCKS = 1
MAX_BLOCKS = ORI_MAX_BLOCKS
BLOCK_NUM = B * MAX_BLOCKS
SPARSE_IDX_TOPK = M.index_topk
START_POS = 0

# Unified prefill sparse-attn shape contract, derived from the same public config
# as prefill_sparse_attn.py rather than imported from that module's internals.
PREFILL_MAX_COMPRESSED = max(1, min(SPARSE_IDX_TOPK, S // 4))
PREFILL_CORE_TOPK = WIN + SPARSE_IDX_TOPK
PREFILL_ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
PREFILL_ORI_BLOCK_NUM = B * PREFILL_ORI_MAX_BLOCKS
PREFILL_CMP_MAX_BLOCKS = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
PREFILL_CMP_BLOCK_NUM = B * PREFILL_CMP_MAX_BLOCKS
PREFILL_ATTN_TILE = 64
PREFILL_SPARSE_TOPK = min(PREFILL_CORE_TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
PREFILL_SPARSE_PAD = ((PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE) * PREFILL_ATTN_TILE

# HC tiling, mirrored from hc_pre/hc_post but using prefill B/S/T.
MIX_PAD = 32
NEG_INF = -1e20
T_TILE = 16
RMS_T_TILE = 16
LINEAR_T_TILE = 16
COMB_T_TILE = 16
RMS_K_CHUNK = 128
LINEAR_K_CHUNK = 512
D_CHUNK = 512
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK
RMS_PIPE_STAGE = 1 if T >= 64 else 4

# SWA + o_proj tiling.
ATTN_HEAD_TILE = 16
ATTN_TASK_TILE = 4
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
assert B == 1, "SWA adapter to unified prefill sparse-attn currently assumes PREFILL_BATCH == 1"
assert PREFILL_ORI_BLOCK_NUM == BLOCK_NUM, "unified prefill sparse-attn ori cache must match SWA cache"
assert PREFILL_ORI_MAX_BLOCKS == MAX_BLOCKS, "unified prefill sparse-attn block table must match SWA table"


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
def prefill_swa_prepare_sparse_inputs(
    kv:                 pl.Tensor[[T, HEAD_DIM],                    pl.BF16],
    kv_cache:           pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table:        pl.Tensor[[B, MAX_BLOCKS],                  pl.INT32],
    ori_kv:             pl.Tensor[[PREFILL_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv:             pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table:    pl.Tensor[[B, PREFILL_ORI_MAX_BLOCKS],      pl.INT32],
    cmp_block_table:    pl.Tensor[[B, PREFILL_CMP_MAX_BLOCKS],      pl.INT32],
    sparse_indices:     pl.Tensor[[T, PREFILL_CORE_TOPK],           pl.INT32],
    start_pos:          pl.Scalar[pl.INT32],
):
    ori_kv_flat = pl.reshape(ori_kv, [PREFILL_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    kv_cache_flat = pl.reshape(kv_cache, [BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [PREFILL_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    tile_start_pos = pl.cast(start_pos, pl.INDEX)
    hist_valid = pl.min(tile_start_pos, WIN - 1)
    hist_start_abs = tile_start_pos - hist_valid

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_sparse_tables"):
        pl.write(ori_block_table, [0, 0], pl.cast(0, pl.INT32))
        pl.write(cmp_block_table, [0, 0], pl.cast(0, pl.INT32))

    for s0 in pl.range(0, S, KV_CACHE_WRITE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_prompt_kv"):
            kv_tile = pl.load(
                kv,
                [s0, 0],
                [KV_CACHE_WRITE_TILE, HEAD_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            ori_kv_flat = pl.store(kv_tile, [s0, 0], ori_kv_flat)
    ori_kv = pl.reshape(ori_kv_flat, [PREFILL_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    # Materialize the historical part of the sliding window as the unified
    # sparse-attn compressed tail. Current-prompt KV remains in ori_kv.
    for hist0 in pl.range(0, WIN, KV_CACHE_WRITE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_history_kv"):
            unused_row = pl.load(kv, [0, 0], [1, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
            for ds in pl.range(KV_CACHE_WRITE_TILE):
                hist_i = hist0 + ds
                if hist_i < hist_valid:
                    key_abs_pos = hist_start_abs + hist_i
                    ori_slot = key_abs_pos % WIN
                    blk_id = pl.cast(pl.read(block_table, [0, ori_slot // BLOCK_SIZE]), pl.INDEX)
                    cache_row = blk_id * BLOCK_SIZE + ori_slot % BLOCK_SIZE
                    hist_row = pl.load(
                        kv_cache_flat,
                        [cache_row, 0],
                        [1, HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    cmp_kv_flat = pl.store(hist_row, [hist_i, 0], cmp_kv_flat)
                else:
                    cmp_kv_flat = pl.store(unused_row, [hist_i, 0], cmp_kv_flat)

    cmp_kv = pl.reshape(cmp_kv_flat, [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    # Build per-token causal SWA indices in the unified prefill convention:
    # raw < S reads current prompt KV, raw >= S reads the historical tail above.
    for idx_t0 in pl.parallel(0, T, ATTN_TASK_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_sparse_indices"):
            for idx_dt in pl.range(ATTN_TASK_TILE):
                idx_t = idx_t0 + idx_dt
                idx_s = idx_t % S
                abs_pos = tile_start_pos + idx_s
                window_valid = pl.min(WIN, abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                # The unified core reads the padded prefix, so clear every consumed slot first.
                for pad_i in pl.range(PREFILL_SPARSE_PAD):
                    pl.write(sparse_indices, [idx_t, pad_i], pl.cast(-1, pl.INT32))
                for key_i in pl.range(WIN):
                    if key_i < window_valid:
                        key_abs_pos = key_start_abs + key_i
                        if key_abs_pos >= tile_start_pos:
                            raw_idx = key_abs_pos - tile_start_pos
                        else:
                            raw_idx = S + key_abs_pos - hist_start_abs
                        pl.write(sparse_indices, [idx_t, key_i], pl.cast(raw_idx, pl.INT32))

    return ori_kv, cmp_kv, ori_block_table, cmp_block_table, sparse_indices


@pl.jit
def prefill_attention_swa(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    # Full prefill path mirrors the official block: hc_pre -> qkv/rope -> SWA
    # attention/o_proj -> KV writeback -> hc_post.
    x_mixed, post, comb = prefill_hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )

    # Reuse the shared prefill QKV/RoPE projection to stay aligned with decode.
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    x_normed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    x_normed = prefill_attn_norm(x_mixed, attn_norm_w, x_normed)
    q, kv, qr, qr_scale = prefill_qkv_proj_rope_core(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        start_pos,
    )

    # Gather the RoPE rows used later to undo RoPE on the attention output.
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    for b in pl.range(B):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_rope_rows"):
            pos = pl.cast(start_pos, pl.INDEX)
            cos_rows = pl.slice(freqs_cos, [S, ROPE_HEAD_DIM], [pos, 0])
            sin_rows = pl.slice(freqs_sin, [S, ROPE_HEAD_DIM], [pos, 0])
            rope_cos_t = pl.assemble(rope_cos_t, cos_rows, [b * S, 0])
            rope_sin_t = pl.assemble(rope_sin_t, sin_rows, [b * S, 0])

    # SWA attention now feeds the unified prefill sparse-attn core. The SWA
    # orchestration still owns sliding-window history materialization and cache
    # writeback.
    ori_kv = pl.create_tensor([PREFILL_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_kv = pl.create_tensor([PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    ori_block_table = pl.create_tensor([B, PREFILL_ORI_MAX_BLOCKS], dtype=pl.INT32)
    cmp_block_table = pl.create_tensor([B, PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
    sparse_indices = pl.create_tensor([T, PREFILL_CORE_TOPK], dtype=pl.INT32)
    ori_kv, cmp_kv, ori_block_table, cmp_block_table, sparse_indices = prefill_swa_prepare_sparse_inputs(
        kv,
        kv_cache,
        block_table,
        ori_kv,
        cmp_kv,
        ori_block_table,
        cmp_block_table,
        sparse_indices,
        start_pos,
    )

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_sparse_attn(
        q,
        ori_kv,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        sparse_indices,
        attn_sink,
        seqused_kv,
        rope_cos_t,
        rope_sin_t,
        even_select_local,
        odd_select_local,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    kv_cache = prefill_swa_write_kv_cache(kv, kv_cache, block_table, start_pos)

    # create_tensor seeds static metadata required by the JIT for hc_post input.
    attn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out = prefill_hc_post(
        attn_out_3d,
        x_hc,
        post,
        comb,
        x_out,
    )
    return kv_cache, x_out


def _quant_w_per_output_channel(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def _quant_w_per_row(w):
    import torch

    amax = w.float().abs().amax(dim=1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(-1, 1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_prefill_attention_swa(tensors):
    """Torch reference for the official SWA prefill branch."""
    import torch

    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
    golden_prefill_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })
    if "x_mixed" in tensors:
        tensors["x_mixed"][:] = x_mixed
    if "post_t" in tensors:
        tensors["post_t"][:] = post
    if "comb_t" in tensors:
        tensors["comb_t"][:] = comb

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_prefill_attn_norm(x_mixed, tensors["attn_norm_w"])
    golden_prefill_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
        "start_pos": tensors["start_pos"],
    })
    if "q_out" in tensors:
        tensors["q_out"][:] = q
    if "kv_out" in tensors:
        tensors["kv_out"][:] = kv
    if "qr_out" in tensors:
        tensors["qr_out"][:] = qr
    if "qr_scale_out" in tensors:
        tensors["qr_scale_out"][:] = qr_scale

    start_pos = int(tensors["start_pos"])
    positions = torch.arange(start_pos, start_pos + S, device=tensors["freqs_cos"].device)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).unsqueeze(0).expand(B, S, ROPE_HEAD_DIM)
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).unsqueeze(0).expand(B, S, ROPE_HEAD_DIM)

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    ori_kv = torch.zeros(PREFILL_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    ori_kv[0, :S, 0] = kv.view(B, S, HEAD_DIM)[0]
    ori_block_table = torch.zeros(B, PREFILL_ORI_MAX_BLOCKS, dtype=torch.int32)
    cmp_kv = torch.zeros(PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    cmp_block_table = torch.zeros(B, PREFILL_CMP_MAX_BLOCKS, dtype=torch.int32)
    sparse_indices = torch.full((T, PREFILL_CORE_TOPK), -1, dtype=torch.int32)

    hist_valid = min(start_pos, WIN - 1)
    hist_start_abs = start_pos - hist_valid
    kv_cache_for_attn = tensors["kv_cache"]
    block_table = tensors["block_table"]
    for hist_i in range(hist_valid):
        key_abs_pos = hist_start_abs + hist_i
        ori_slot = key_abs_pos % WIN
        blk_id = int(block_table[0, ori_slot // BLOCK_SIZE].item())
        cmp_kv[0, hist_i, 0] = kv_cache_for_attn[blk_id, ori_slot % BLOCK_SIZE, 0]

    for t in range(T):
        s = t % S
        abs_pos = start_pos + s
        window_valid = min(WIN, abs_pos + 1)
        key_start_abs = abs_pos + 1 - window_valid
        for key_i in range(window_valid):
            key_abs_pos = key_start_abs + key_i
            if key_abs_pos >= start_pos:
                raw_idx = key_abs_pos - start_pos
            else:
                raw_idx = S + key_abs_pos - hist_start_abs
            sparse_indices[t, key_i] = raw_idx

    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": ori_kv,
        "ori_block_table": ori_block_table,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "cmp_sparse_indices": sparse_indices,
        "attn_sink": tensors["attn_sink"],
        "seqused_kv": tensors["seqused_kv"],
        "freqs_cos": rope_cos_t.reshape(T, ROPE_HEAD_DIM).contiguous(),
        "freqs_sin": rope_sin_t.reshape(T, ROPE_HEAD_DIM).contiguous(),
        "even_select_local": tensors["even_select_local"],
        "odd_select_local": tensors["odd_select_local"],
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })
    kv_cache = tensors["kv_cache"]
    for b in range(B):
        for s in range(S):
            ori_slot = (start_pos + s) % WIN
            blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
            kv_cache[blk_id, ori_slot % BLOCK_SIZE, 0] = kv.view(B, S, HEAD_DIM)[b, s]
    if "attn_out" in tensors:
        tensors["attn_out"][:] = attn_out

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_prefill_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post,
        "comb": comb,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_hc():
        return torch.randn(B, S, HC_MULT, D) * 0.05
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    def init_hc_attn_scale():
        return torch.ones(3) * 0.5
    def init_hc_attn_base():
        return torch.zeros(MIX_HC)
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return torch.randn(D, Q_LORA) / D ** 0.5
    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
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
    def init_block_table():
        tbl = torch.full((B, MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            tbl[b, 0] = b
        return tbl
    def init_kv_cache():
        if start_pos == 0:
            return torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        return torch.randn(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) * 0.05
    def init_attn_sink():
        return torch.zeros(H)
    def init_seqused_kv():
        return torch.full((B,), min(WIN, start_pos + S), dtype=torch.int32)
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.float32, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("even_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("block_table", [B, MAX_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("start_pos", torch.int32, start_pos),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--block-dim", type=int, default=None)
    parser.add_argument("--aicpu-thread-num", type=int, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_attention_swa,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_attention_swa,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            block_dim=args.block_dim,
            aicpu_thread_num=args.aicpu_thread_num,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_allclose(atol=6e-3, rtol=2.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
