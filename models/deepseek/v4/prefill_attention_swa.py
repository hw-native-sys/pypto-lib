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

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX
from prefill_qkv_proj_rope import (
    B,
    D,
    H,
    HEAD_DIM,
    MAX_SEQ_LEN,
    NOPE_DIM,
    PREFILL_START_POS,
    Q_LORA,
    ROPE_DIM,
    ROPE_HALF,
    S,
    T,
    golden_prefill_qkv_proj_rope,
    prefill_qkv_proj_rope_core,
)


# model config
EPS = M.rms_norm_eps
ROPE_HEAD_DIM = ROPE_DIM
NOPE_HEAD_DIM = NOPE_DIM
HALF_ROPE = ROPE_HALF
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
TOPK = WIN
SPARSE_IDX_TOPK = M.index_topk
SPARSE_TOPK = WIN
START_POS = PREFILL_START_POS

# HC tiling, mirrored from hc_pre/hc_post but using prefill B/S/T.
MIX_PAD = 32
HC_PAD = 8
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
def prefill_hc_pre(
    x:        pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_fn:    pl.Tensor[[MIX_HC, HC_DIM],   pl.FP32],
    hc_scale: pl.Tensor[[3],                pl.FP32],
    hc_base:  pl.Tensor[[MIX_HC],           pl.FP32],
    x_mixed:  pl.Tensor[[B, S, D],          pl.BF16],
    post_pad_store:  pl.Tensor[[T, HC_PAD], pl.FP32],
    comb0_pad_store: pl.Tensor[[T, HC_PAD], pl.FP32],
    comb1_pad_store: pl.Tensor[[T, HC_PAD], pl.FP32],
    comb2_pad_store: pl.Tensor[[T, HC_PAD], pl.FP32],
    comb3_pad_store: pl.Tensor[[T, HC_PAD], pl.FP32],
):
    x_flat = pl.reshape(x, [T, HC_DIM])
    inv_rms = pl.create_tensor([1, T], dtype=pl.FP32)
    mixes = pl.create_tensor([T, MIX_PAD], dtype=pl.FP32)
    mix_raw = pl.create_tensor([T, MIX_PAD], dtype=pl.FP32)
    pre_val_store = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    pre_val_t = pl.create_tensor([HC_PAD, T], dtype=pl.FP32)

    # Official HC pre starts with RMS over the concatenated hidden-choice lanes.
    for t0 in pl.parallel(0, T, RMS_T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_rms"):
            sq_sum = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(RMS_K_BLOCKS, stage=RMS_PIPE_STAGE):
                k0 = kb * RMS_K_CHUNK
                x_chunk = pl.cast(
                    pl.slice(x_flat, [RMS_T_TILE, RMS_K_CHUNK], [t0, k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, RMS_T_TILE]),
                )
            inv_rms_val = pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), EPS), high_precision=True)
            inv_rms = pl.assemble(inv_rms, inv_rms_val, [0, t0])

    # Project normalized HC features to pre/post/comb logits.
    for t0 in pl.parallel(0, T, LINEAR_T_TILE):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
            name_hint="prefill_hc_linear",
        ):
            x_lin_0 = pl.cast(
                pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, 0]),
                target_type=pl.FP32,
            )
            w_lin_0 = pl.slice(
                hc_fn,
                [MIX_PAD, LINEAR_K_CHUNK],
                [0, 0],
                valid_shape=[MIX_HC, LINEAR_K_CHUNK],
            )
            mix_acc = pl.matmul(x_lin_0, w_lin_0, b_trans=True, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, LINEAR_K_BLOCKS, stage=2):
                kl0 = kb * LINEAR_K_CHUNK
                x_lin = pl.cast(
                    pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, kl0]),
                    target_type=pl.FP32,
                )
                w_lin = pl.slice(
                    hc_fn,
                    [MIX_PAD, LINEAR_K_CHUNK],
                    [0, kl0],
                    valid_shape=[MIX_HC, LINEAR_K_CHUNK],
                )
                mix_acc = pl.matmul_acc(mix_acc, x_lin, w_lin, b_trans=True)
            mix_raw = pl.assemble(mix_raw, mix_acc, [t0, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_linear_scale"):
        mixes = pl.assemble(
            mixes,
            pl.row_expand_mul(pl.slice(mix_raw, [T, MIX_PAD], [0, 0]), pl.reshape(inv_rms, [T, 1])),
            [0, 0],
        )

    # Split logits into pre gates, post gates, and the comb Sinkhorn matrix.
    comb_logits = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    for split_t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_split"):
            scale0 = pl.tensor.read(hc_scale, [0])
            scale1 = pl.tensor.read(hc_scale, [1])
            scale2 = pl.tensor.read(hc_scale, [2])

            ones_hc = pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0)
            pre_base = pl.reshape(pl.slice(hc_base, [HC_PAD], [0]), [1, HC_PAD])
            pre_logits = pl.add(
                pl.mul(pl.slice(mixes, [T_TILE, HC_PAD], [split_t0, 0]), scale0),
                pl.col_expand_mul(ones_hc, pre_base),
            )
            pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
            pre_val_store = pl.assemble(pre_val_store, pre_val, [split_t0, 0])

            post_base = pl.reshape(pl.slice(hc_base, [HC_PAD], [HC_MULT]), [1, HC_PAD])
            post_logits = pl.add(
                pl.mul(pl.slice(mixes, [T_TILE, HC_PAD], [split_t0, HC_MULT]), scale1),
                pl.col_expand_mul(ones_hc, post_base),
            )
            post_pad = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
            post_pad_store = pl.assemble(post_pad_store, post_pad, [split_t0, 0])

            ones_comb = pl.full([T_TILE, HC_MULT * HC_MULT], dtype=pl.FP32, value=1.0)
            comb_base = pl.reshape(
                pl.slice(hc_base, [HC_MULT * HC_MULT], [HC_MULT * 2]),
                [1, HC_MULT * HC_MULT],
            )
            comb_mix = pl.slice(mixes, [T_TILE, HC_MULT * HC_MULT], [split_t0, HC_MULT * 2])
            comb_logits_val = pl.add(
                pl.mul(comb_mix, scale2),
                pl.col_expand_mul(ones_comb, comb_base),
            )
            comb_logits = pl.assemble(comb_logits, comb_logits_val, [split_t0, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_transpose_pre"):
        for t0 in pl.range(0, T, T_TILE):
            pre_tile = pl.load(
                pre_val_store,
                [t0, 0],
                [T_TILE, HC_PAD],
                target_memory=pl.MemorySpace.Vec,
            )
            pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
            pre_val_t = pl.store(pre_tile_t, [0, t0], pre_val_t)

    for t0 in pl.parallel(0, T, COMB_T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_comb_sinkhorn"):
            row0 = pl.fillpad(pl.load(
                comb_logits,
                [t0, 0 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row1 = pl.fillpad(pl.load(
                comb_logits,
                [t0, 1 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row2 = pl.fillpad(pl.load(
                comb_logits,
                [t0, 2 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row3 = pl.fillpad(pl.load(
                comb_logits,
                [t0, 3 * HC_MULT],
                [COMB_T_TILE, HC_PAD],
                valid_shapes=[COMB_T_TILE, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)

            row_max_tmp = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_sum_tmp = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row0_exp = pl.exp(pl.row_expand_sub(row0, pl.row_max(row0, row_max_tmp)))
            row1_exp = pl.exp(pl.row_expand_sub(row1, pl.row_max(row1, row_max_tmp)))
            row2_exp = pl.exp(pl.row_expand_sub(row2, pl.row_max(row2, row_max_tmp)))
            row3_exp = pl.exp(pl.row_expand_sub(row3, pl.row_max(row3, row_max_tmp)))
            row0_soft = pl.add(pl.row_expand_div(row0_exp, pl.row_sum(row0_exp, row_sum_tmp)), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, pl.row_sum(row1_exp, row_sum_tmp)), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, pl.row_sum(row2_exp, row_sum_tmp)), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, pl.row_sum(row3_exp, row_sum_tmp)), HC_EPS)

            row0_eff = pl.tile.fillpad(pl.tile.set_validshape(row0_soft, COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)
            row1_eff = pl.tile.fillpad(pl.tile.set_validshape(row1_soft, COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)
            row2_eff = pl.tile.fillpad(pl.tile.set_validshape(row2_soft, COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)
            row3_eff = pl.tile.fillpad(pl.tile.set_validshape(row3_soft, COMB_T_TILE, HC_MULT), pad_value=pl.PadValue.zero)

            row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_eff, col_sum)
            row1_cur = pl.div(row1_eff, col_sum)
            row2_cur = pl.div(row2_eff, col_sum)
            row3_cur = pl.div(row3_eff, col_sum)

            for _ in pl.unroll(HC_SINKHORN_ITER - 1):
                row0_norm = pl.row_expand_div(row0_cur, pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS))
                row1_norm = pl.row_expand_div(row1_cur, pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS))
                row2_norm = pl.row_expand_div(row2_cur, pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS))
                row3_norm = pl.row_expand_div(row3_cur, pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS))
                col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
                col_sum = pl.add(col_sum, HC_EPS)
                row0_cur = pl.div(row0_norm, col_sum)
                row1_cur = pl.div(row1_norm, col_sum)
                row2_cur = pl.div(row2_norm, col_sum)
                row3_cur = pl.div(row3_norm, col_sum)

            # These padded row tensors are the SSA dependency used by the
            # fused hc_post consumer.
            comb0_pad_store = pl.store(row0_cur, [t0, 0], comb0_pad_store)
            comb1_pad_store = pl.store(row1_cur, [t0, 0], comb1_pad_store)
            comb2_pad_store = pl.store(row2_cur, [t0, 0], comb2_pad_store)
            comb3_pad_store = pl.store(row3_cur, [t0, 0], comb3_pad_store)

    # Mix the HC lanes into the model dimension before qkv projection.
    x_mixed_view = pl.reshape(x_mixed, [T, D])
    for t0 in pl.parallel(0, T, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hc_mix_x"):
            pre0 = pl.reshape(
                pl.load(pre_val_t, [0, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre1 = pl.reshape(
                pl.load(pre_val_t, [1, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre2 = pl.reshape(
                pl.load(pre_val_t, [2, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            pre3 = pl.reshape(
                pl.load(pre_val_t, [3, t0], [1, T_TILE], target_memory=pl.MemorySpace.Vec),
                [T_TILE, 1],
            )
            for db in pl.range(D_BLOCKS):
                d0 = db * D_CHUNK
                x0 = pl.cast(
                    pl.load(x_flat, [t0, 0 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x1 = pl.cast(
                    pl.load(x_flat, [t0, 1 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x2 = pl.cast(
                    pl.load(x_flat, [t0, 2 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                x3 = pl.cast(
                    pl.load(x_flat, [t0, 3 * D + d0], [T_TILE, D_CHUNK], target_memory=pl.MemorySpace.Vec),
                    target_type=pl.FP32,
                )
                y_tile = pl.add(
                    pl.add(pl.row_expand_mul(x0, pre0), pl.row_expand_mul(x1, pre1)),
                    pl.add(pl.row_expand_mul(x2, pre2), pl.row_expand_mul(x3, pre3)),
                )
                x_mixed_view = pl.store(
                    pl.cast(y_tile, target_type=pl.BF16, mode="rint"),
                    [t0, d0],
                    x_mixed_view,
                )
    x_mixed = pl.reshape(x_mixed_view, [B, S, D])
    return x_mixed, post_pad_store, comb0_pad_store, comb1_pad_store, comb2_pad_store, comb3_pad_store


@pl.jit.inline
def prefill_hc_post_from_padded(
    x:               pl.Tensor[[B, S, D],             pl.BF16],
    residual:        pl.Tensor[[B, S, HC_MULT, D],    pl.BF16],
    post_pad:        pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb0_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb1_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb2_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    comb3_pad:       pl.Tensor[[T, HC_PAD],           pl.FP32],
    y:               pl.Tensor[[B, S, HC_MULT, D],    pl.BF16],
):
    x_flat = pl.reshape(x, [T, D])
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    y_flat = pl.reshape(y, [T, HC_DIM])

    # Same math as prefill_hc_post, but consume the padded SSA tensors emitted
    # by prefill_hc_pre.  This gives the scheduler explicit producer->consumer
    # dependencies and avoids using side-effect-only post/comb scratch writes.
    for out_h in pl.parallel(HC_MULT):
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer,
                   name_hint="prefill_hc_post_from_padded"):
            for t in pl.parallel(0, T, 1, chunk=16):
                post_w = pl.read(post_pad, [t, out_h])
                comb0_w = pl.read(comb0_pad, [t, out_h])
                comb1_w = pl.read(comb1_pad, [t, out_h])
                comb2_w = pl.read(comb2_pad, [t, out_h])
                comb3_w = pl.read(comb3_pad, [t, out_h])
                for db in pl.range(D_BLOCKS):
                    d0 = db * D_CHUNK
                    x_row = pl.cast(
                        pl.slice(x_flat, [1, D_CHUNK], [t, d0]),
                        target_type=pl.FP32,
                    )
                    y_row = pl.mul(x_row, post_w)
                    residual0 = pl.cast(
                        pl.slice(residual_flat, [1, D_CHUNK], [t, 0 * D + d0]),
                        target_type=pl.FP32,
                    )
                    residual1 = pl.cast(
                        pl.slice(residual_flat, [1, D_CHUNK], [t, 1 * D + d0]),
                        target_type=pl.FP32,
                    )
                    residual2 = pl.cast(
                        pl.slice(residual_flat, [1, D_CHUNK], [t, 2 * D + d0]),
                        target_type=pl.FP32,
                    )
                    residual3 = pl.cast(
                        pl.slice(residual_flat, [1, D_CHUNK], [t, 3 * D + d0]),
                        target_type=pl.FP32,
                    )
                    y_row = pl.add(y_row, pl.mul(residual0, comb0_w))
                    y_row = pl.add(y_row, pl.mul(residual1, comb1_w))
                    y_row = pl.add(y_row, pl.mul(residual2, comb2_w))
                    y_row = pl.add(y_row, pl.mul(residual3, comb3_w))
                    y_flat = pl.assemble(
                        y_flat,
                        pl.cast(y_row, target_type=pl.BF16, mode="rint"),
                        [t, out_h * D + d0],
                    )
    y = pl.reshape(y_flat, [B, S, HC_MULT, D])
    return y


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
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
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
    post_pad = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    comb0_pad = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    comb1_pad = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    comb2_pad = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    comb3_pad = pl.create_tensor([T, HC_PAD], dtype=pl.FP32)
    # Full prefill path mirrors the official block: hc_pre -> qkv/rope -> SWA
    # attention/o_proj -> KV writeback -> hc_post.
    x_mixed, post_pad, comb0_pad, comb1_pad, comb2_pad, comb3_pad = prefill_hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post_pad,
        comb0_pad,
        comb1_pad,
        comb2_pad,
        comb3_pad,
    )

    # Reuse the shared prefill QKV/RoPE projection to stay aligned with decode.
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q, kv, qr, qr_scale = prefill_qkv_proj_rope_core(
        x_mixed,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        even_select_t,
        odd_select_t,
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

    # SWA attention computes grouped head values, then packs and projects them.
    attn_values = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.BF16)
    attn_values = prefill_swa_attention_values(q, kv, kv_cache, block_table, attn_sink, attn_values, start_pos)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    o_packed = prefill_swa_pack_context_from_values(
        attn_values,
        rope_cos_t,
        rope_sin_t,
        even_select_local,
        odd_select_local,
        o_packed,
    )
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_swa_o_proj_from_packed(o_packed, wo_a, wo_b, wo_b_scale, attn_out)

    # Cache is updated after attention so current tokens do not read future KV.
    kv_cache = prefill_swa_write_kv_cache(kv, kv_cache, block_table, start_pos)

    # create_tensor seeds static metadata required by the JIT for hc_post input.
    attn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out = prefill_hc_post_from_padded(
        attn_out_3d,
        x_hc,
        post_pad,
        comb0_pad,
        comb1_pad,
        comb2_pad,
        comb3_pad,
        x_out,
    )
    return kv_cache, x_out


def _int8_quant_per_row(x):
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


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

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def _golden_hc_pre(tensors):
    import torch

    x = tensors["x"].float()
    hc_fn = tensors["hc_fn"].float()
    hc_scale = tensors["hc_scale"].float()
    hc_base = tensors["hc_base"].float()

    x_flat = x.flatten(2)
    x_flat_2d = x_flat.reshape(T, HC_DIM)
    sq_sum = torch.zeros(T, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + EPS)

    mix_cols = []
    for m in range(MIX_HC):
        mix_col = torch.zeros(T, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_CHUNK):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_CHUNK]
            w_chunk = hc_fn[m:m + 1, k0:k0 + LINEAR_K_CHUNK]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1).reshape(B, S, MIX_HC)

    pre_logits = mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]
    post_logits = mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1] + hc_base[HC_MULT:HC_MULT * 2]
    pre = torch.sigmoid(pre_logits) + HC_EPS
    post_t = 2 * torch.sigmoid(post_logits)
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]).view(
        B, S, HC_MULT, HC_MULT
    )
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    # Match prefill_hc_mix_x's pairwise add tree. Tiny HC pre differences are
    # amplified by the downstream q-path row quantization.
    y01 = x[:, :, 0, :] * pre[:, :, 0:1] + x[:, :, 1, :] * pre[:, :, 1:2]
    y23 = x[:, :, 2, :] * pre[:, :, 2:3] + x[:, :, 3, :] * pre[:, :, 3:4]
    y = y01 + y23

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["x_mixed"][:] = _to_device_bf16(y)
    tensors["post"][:] = post_t
    tensors["comb"][:] = comb_t


def _golden_hc_post(tensors):
    import torch

    x = tensors["x"].float()
    residual = tensors["residual"].float()
    post = tensors["post"].float()
    comb = tensors["comb"].float()

    y_fp32 = torch.zeros(B, S, HC_MULT, D, dtype=torch.float32)
    for out_h in range(HC_MULT):
        y_row = x * post[:, :, out_h:out_h + 1]
        for in_h in range(HC_MULT):
            y_row = y_row + residual[:, :, in_h, :] * comb[:, :, in_h, out_h:out_h + 1]
        y_fp32[:, :, out_h, :] = y_row
    tensors["y"][:] = y_fp32.to(torch.bfloat16)


def _golden_swa_attention_o_proj(tensors):
    import torch

    cos = tensors["freqs_cos"].float().view(B, S, ROPE_HEAD_DIM)
    sin = tensors["freqs_sin"].float().view(B, S, ROPE_HEAD_DIM)
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
    if "o_packed" in tensors:
        tensors["o_packed"][:] = o_model.permute(2, 0, 1, 3).contiguous().view(O_GROUPS * T, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)
    tensors["attn_out"][:] = out.to(torch.bfloat16)


def golden_prefill_attention_swa(tensors):
    """Torch reference for the official SWA prefill branch."""
    import torch

    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT)
    _golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })
    if "x_mixed" in tensors:
        tensors["x_mixed"][:] = x_mixed
    if "post_t" in tensors:
        tensors["post_t"][:] = post_t
    if "comb_t" in tensors:
        tensors["comb_t"][:] = comb_t

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    golden_prefill_qkv_proj_rope({
        "x": x_mixed,
        "norm_w": tensors["attn_norm_w"],
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
    _golden_swa_attention_o_proj({
        "q": q,
        "kv": kv,
        "kv_cache": tensors["kv_cache"],
        "block_table": tensors["block_table"],
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_t.reshape(T, ROPE_HEAD_DIM).contiguous(),
        "freqs_sin": rope_sin_t.reshape(T, ROPE_HEAD_DIM).contiguous(),
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
        "start_pos": tensors["start_pos"],
    })
    if "attn_out" in tensors:
        tensors["attn_out"][:] = attn_out

    kv_cache = tensors["kv_cache"]
    block_table = tensors.get("block_table")
    for t in range(T):
        b = t // S
        s = t % S
        ori_slot = (start_pos + s) % WIN
        blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
        kv_cache[blk_id, ori_slot % BLOCK_SIZE, 0] = kv[t]

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    _golden_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })
    tensors["x_out"][:] = y


def _golden_prefill_swa_values(tensors):
    import torch

    start_pos = int(tensors["start_pos"])
    q = tensors["q"].float().view(B, S, H, HEAD_DIM)
    kv = tensors["kv"].float().view(B, S, HEAD_DIM)
    kv_cache = tensors["kv_cache"].float()
    block_table = tensors.get("block_table")
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
                    blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item()) if block_table is not None else b * MAX_BLOCKS
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
    def init_even_select_t():
        m = torch.zeros((ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM))
        for i in range(ROPE_HEAD_DIM // 2):
            m[i, 2 * i] = 1
        return m
    def init_odd_select_t():
        m = torch.zeros((ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM))
        for i in range(ROPE_HEAD_DIM // 2):
            m[i, 2 * i + 1] = 1
        return m
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
        TensorSpec("even_select_t", [ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_even_select_t),
        TensorSpec("odd_select_t", [ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_odd_select_t),
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
