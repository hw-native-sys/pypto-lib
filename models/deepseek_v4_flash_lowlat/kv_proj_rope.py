# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ordered split-K KV projection with RMS normalization and RoPE."""

import pypto.language as pl

from .model_config import FLASH as M

KV_T_DYN = pl.dynamic("NATIVE_D_QKV_KV_T_DYN")
PREFILL_DENSE_TILE = 512
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_DIM_SCALE = float(ROPE_DIM)
NOPE_DIM = M.nope_head_dim
EPS = M.rms_norm_eps
KV_TILE = 64  # kv rms-norm / rope / NOPE N granularity
MATMUL_T_TILE = 16
KV_M_TILE = MATMUL_T_TILE  # kv_proj token (M) tile; decode pads from 8 real rows to 16
KV_DENSE_M_TILE = 64
KV_N_TILE = 256  # kv_proj HEAD_DIM (N) per matmul
KV_K_TILE = 256  # kv_proj D (K) reduction tile   | divides KV_SPLIT_K_TILE
KV_OK = 4  # kv_proj split-K factor         | D//KV_OK cores share each N-group
KV_OM = 3  # maximum kv_proj split-M factor
KV_SPLIT_K_TILE = D // KV_OK  # kv_proj K per split (=1024)
KV_RMS_T_TILE = 32  # kv rms-norm + rope fused token (T) tile


@pl.jit.inline(auto_scope=False)
def kv_proj_rope(
    x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.INT32],
    kv: pl.Tensor[[KV_T_DYN, HEAD_DIM], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """KV LoRA, RMSNorm, and RoPE over bounded dense tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            x_view = pl.reshape(x, [t_dim, D])
            t_matmul = ((tile_rows + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE
            kv_full_rows = (tile_rows // KV_DENSE_M_TILE) * KV_DENSE_M_TILE
            kv_m_groups = pl.min(KV_OM, pl.max(1, tile_rows // (2 * KV_DENSE_M_TILE)))

            # Each split writes a disjoint FP32 plane. Preserve the existing
            # per-split K accumulation; only the cross-split reduction changes.
            kv_partials = pl.create_tensor([KV_OK * t_matmul, HEAD_DIM], dtype=pl.FP32)
            kv_fp32 = pl.create_tensor([t_matmul, HEAD_DIM], dtype=pl.FP32)

            # KV projection consumes the caller's readiness dependency.
            with pl.spmd(
                (HEAD_DIM // KV_N_TILE) * KV_OK * kv_m_groups,
                name_hint="kv_proj_matmul",
                deps=[late_dep],
            ) as _kv_tid:
                # Weight reads bypass L2.
                pl.set_cache_policy(wkv, pl.CachePolicy.BYPASS)
                kbg = pl.tile.get_block_idx()
                kv_col0 = (kbg // (KV_OK * kv_m_groups)) * KV_N_TILE
                kv_split = (kbg // kv_m_groups) % KV_OK
                kv_k_base = ((kbg // kv_m_groups) % KV_OK) * KV_SPLIT_K_TILE
                kv_m_group = kbg % kv_m_groups
                for dense_t0 in pl.range(
                    kv_m_group * KV_DENSE_M_TILE, kv_full_rows, kv_m_groups * KV_DENSE_M_TILE
                ):
                    dense_x0 = tile_base + dense_t0
                    dense_acc = pl.create_tensor([KV_DENSE_M_TILE, KV_N_TILE], dtype=pl.FP32)
                    for dense_k in pl.pipeline(0, KV_SPLIT_K_TILE // KV_K_TILE, stage=2):
                        dense_d0 = kv_k_base + dense_k * KV_K_TILE
                        dense_x = x_view[
                            dense_x0 : dense_x0 + KV_DENSE_M_TILE, dense_d0 : dense_d0 + KV_K_TILE
                        ]
                        dense_w = wkv[dense_d0 : dense_d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                        dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, init_cond=(dense_k == 0))
                    split_row = kv_split * t_matmul + dense_t0
                    kv_partials[split_row : split_row + KV_DENSE_M_TILE, kv_col0 : kv_col0 + KV_N_TILE] = dense_acc
                for t0 in pl.range(kv_full_rows + kv_m_group * KV_M_TILE, t_matmul, kv_m_groups * KV_M_TILE):
                    kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
                    for db in pl.pipeline(KV_SPLIT_K_TILE // KV_K_TILE, stage=2):
                        d0 = kv_k_base + db * KV_K_TILE
                        kv_rows = pl.min(KV_M_TILE, tile_rows - t0)
                        x_t0 = tile_base + t0
                        kv_x_chunk_bf16 = pl.slice(
                            x_view,
                            [KV_M_TILE, KV_K_TILE],
                            [x_t0, d0],
                            valid_shape=[kv_rows, KV_K_TILE],
                        )
                        wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                        kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk, init_cond=(db == 0))
                    split_row = kv_split * t_matmul + t0
                    kv_partials[split_row : split_row + KV_M_TILE, kv_col0 : kv_col0 + KV_N_TILE] = kv_acc

            # Fixed split order, with an explicit complete-producer edge.
            # Every valid/padded row of every split plane is written above.
            reduce_blocks = (t_matmul // KV_M_TILE) * (HEAD_DIM // KV_N_TILE)
            for reduce_block in pl.spmd(reduce_blocks, name_hint="kv_split_ordered_sum", deps=[_kv_tid]):
                reduce_t = (reduce_block // (HEAD_DIM // KV_N_TILE)) * KV_M_TILE
                reduce_n = (reduce_block % (HEAD_DIM // KV_N_TILE)) * KV_N_TILE
                reduced = pl.full([KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0)
                for split in pl.range(KV_OK):
                    partial_t = split * t_matmul + reduce_t
                    partial = kv_partials[partial_t : partial_t + KV_M_TILE, reduce_n : reduce_n + KV_N_TILE]
                    reduced = pl.add(reduced, partial)
                kv_fp32[reduce_t : reduce_t + KV_M_TILE, reduce_n : reduce_n + KV_N_TILE] = reduced

            kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])

            # Fused KV RMSNorm + interleaved (CANN A3) RoPE, one spmd task per
            # [KV_RMS_T_TILE, HEAD_DIM] row block. NOPE columns [0:NOPE_DIM) and rope columns
            # [NOPE_DIM:HEAD_DIM) are disjoint, so each task writes a conflict-free row block.
            kv_token_tiles = (tile_rows + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
            for tg_idx in pl.spmd(
                kv_token_tiles,
                name_hint="kv_rms_norm_rope",
                sync_start=True,
            ):
                tg = tg_idx * KV_RMS_T_TILE
                valid_rows = pl.min(KV_RMS_T_TILE, tile_rows - tg)
                out_tg = tile_base + tg
                if valid_rows == KV_RMS_T_TILE:
                    kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
                        kv_sq = pl.mul(kv_chunk, kv_chunk)
                        kv_row_sum = pl.reshape(pl.row_sum(kv_sq), [1, KV_RMS_T_TILE])
                        kv_sq_sum = pl.add(kv_sq_sum, kv_row_sum)
                    kv_inv_rms = pl.rsqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS), high_precision=True)
                    kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

                    for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
                        gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
                        gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
                        kv_normed = pl.col_expand_mul(
                            pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk
                        )
                        kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
                        kv_view[out_tg : out_tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16

                    gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
                    gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
                    kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
                    kv_rope_norm_chunk = pl.col_expand_mul(
                        pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope
                    )
                    kv_cos_il_full = rope_cos_il[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_sin_signed_full = rope_sin_signed[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swap_idx_full = rope_swap_idx[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swapped_full = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx_full)
                    kv_rope_rot_full = pl.add(
                        pl.mul(kv_rope_norm_chunk, kv_cos_il_full),
                        pl.mul(kv_swapped_full, kv_sin_signed_full),
                    )
                    kv_rope_i16_full = pl.cast(kv_rope_rot_full, target_type=pl.BF16, mode="rint")
                    kv_view[out_tg : out_tg + KV_RMS_T_TILE, NOPE_DIM:HEAD_DIM] = kv_rope_i16_full
                else:
                    kv_reduce_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    kv_sq_sum_tail = pl.tile.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0_tail in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk_tail = pl.load(
                            kv_fp32,
                            [tg, kv_sq_col0_tail],
                            [KV_RMS_T_TILE, KV_TILE],
                            valid_shape=[valid_rows, KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        kv_sq_tail = pl.mul(kv_chunk_tail, kv_chunk_tail)
                        kv_row_sum_tail = pl.reshape(
                            pl.row_sum(kv_sq_tail, kv_reduce_tmp), [1, KV_RMS_T_TILE]
                        )
                        kv_sq_sum_tail = pl.add(kv_sq_sum_tail, kv_row_sum_tail)
                    kv_inv_rms_tail = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum_tail, 1.0 / HEAD_DIM), EPS)))
                    kv_inv_rms_t_tail = pl.reshape(kv_inv_rms_tail, [KV_RMS_T_TILE, 1])

                    for n0_tail in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                        kv_chunk_tail = pl.load(
                            kv_fp32,
                            [tg, n0_tail],
                            [KV_RMS_T_TILE, KV_TILE],
                            valid_shape=[valid_rows, KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        gamma_kv_input_tail = pl.load(
                            gamma_ckv,
                            [n0_tail],
                            [KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        gamma_kv_cast_tail = pl.cast(gamma_kv_input_tail, target_type=pl.FP32)
                        gamma_kv_chunk_tail = pl.reshape(gamma_kv_cast_tail, [1, KV_TILE])
                        kv_normed_tail = pl.col_expand_mul(
                            pl.row_expand_mul(kv_chunk_tail, kv_inv_rms_t_tail),
                            gamma_kv_chunk_tail,
                        )
                        kv_normed_bf16_tail = pl.cast(kv_normed_tail, target_type=pl.BF16, mode="rint")
                        kv_normed_valid = pl.set_validshape(kv_normed_bf16_tail, valid_rows, KV_TILE)
                        pl.store(kv_normed_valid, [out_tg, n0_tail], kv_view)

                    gamma_rope_input_tail = pl.load(
                        gamma_ckv,
                        [NOPE_DIM],
                        [ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    gamma_rope_cast_tail = pl.cast(gamma_rope_input_tail, target_type=pl.FP32)
                    gamma_rope_tail = pl.reshape(gamma_rope_cast_tail, [1, ROPE_DIM])
                    kv_rope_chunk_tail = pl.load(
                        kv_fp32,
                        [tg, NOPE_DIM],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_rope_norm_tail = pl.col_expand_mul(
                        pl.row_expand_mul(kv_rope_chunk_tail, kv_inv_rms_t_tail),
                        gamma_rope_tail,
                    )
                    kv_cos_il_tail = pl.load(
                        rope_cos_il,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_sin_signed_tail = pl.load(
                        rope_sin_signed,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_col = pl.col_expand_mul(
                        pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                        pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                    )
                    kv_dup_f = pl.cast(
                        pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32
                    )
                    kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))
                    kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))
                    # Row-major flattening offsets stay in fp32: col-expand is defined
                    # for half / float only.
                    kv_row_seed = pl.mul(
                        pl.cast(pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                        ROPE_DIM_SCALE,
                    )
                    kv_row_grid = pl.col_expand_mul(
                        pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0),
                        kv_row_seed,
                    )
                    kv_row_offset = pl.transpose(kv_row_grid, axis1=0, axis2=1)
                    kv_swap_idx_tail = pl.cast(pl.add(kv_swap_f, kv_row_offset), target_type=pl.INT32)
                    kv_gather_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, ROPE_DIM],
                        dtype=pl.INT32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_swapped_tail = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
                    kv_rope_rot_tail = pl.add(
                        pl.mul(kv_rope_norm_tail, kv_cos_il_tail),
                        pl.mul(kv_swapped_tail, kv_sin_signed_tail),
                    )
                    kv_rope_i16_tail = pl.cast(kv_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    kv_rope_valid = pl.set_validshape(kv_rope_i16_tail, valid_rows, ROPE_DIM)
                    pl.store(kv_rope_valid, [out_tg, NOPE_DIM], kv_view)
