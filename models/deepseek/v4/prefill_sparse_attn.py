# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 token-major prefill sparse attention.

The public entry `prefill_sparse_attn` consumes lowered token-major metadata and a
unified overlay raw-index contract:
- `-1`: invalid
- `[0, WIN)`: historical sliding-window ring KV
- `[WIN, WIN + MAX_TOKENS)`: current suffix overlay KV
- `[WIN + MAX_TOKENS, ...)`: compressed KV

`cmp_sparse_lens[t]` is the authoritative usable prefix length for
`cmp_sparse_indices[t]`; any entries after that prefix are ignored even if they
look like valid raw indices. The standalone harness keeps the decode-style
`--compress-ratio {0,4,128}` as a fixture generator only. The kernel itself does
not branch on ratio; the prebuilt raw indices fully describe which KV source
each row comes from.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ


# Prefill target shape for correctness bring-up.
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S

# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
IDX_TOPK = M.index_topk
WIN = M.sliding_window
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# cache shapes
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 0
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
PREFILL_SPARSE_TOPK = min(TOPK, min(M.sliding_window, S) + PREFILL_MAX_COMPRESSED)
ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# RoPE selector shapes kept to match the decode sparse-attn interface.
ROPE_CHUNK = 16
ROPE_INTERLEAVE_CHUNK = 2 * ROPE_CHUNK

# Correctness-first kernel tiling. These mirror the proven decode sparse-attn
# shapes where possible, while padding prompt-K tiles so S=16 can still use
# cube-friendly 64-column attention blocks without out-of-bounds slices.
GATHER_TOKEN_TILE = 4
ATTN_TOKEN_TILE = 32
ROPE_TOKEN_TILE = 8
ROPE_PACK_TOKEN_TILE = 16
MATMUL_ROW_PAD = 16
PV_HEAD_TILE = 16
MERGE_NORM_TOKEN_TILE = 16
# Use a wider sparse-attention tile for packed overlay. Long-prefix CSA can
# expose >64 compressed rows; keeping them in <=3 merge blocks avoids the
# current 4th/5th-block numeric drift while preserving the same raw-index set.
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
assert 1 <= PREFILL_ATTN_BLOCKS <= 3, (
    f"PREFILL_ATTN_BLOCKS={PREFILL_ATTN_BLOCKS} from PREFILL_ATTN_TILE={PREFILL_ATTN_TILE} "
    "must fit the 3-way merge buffers"
)
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
ROPE_PACK_SPMD_BLOCKS = ((T + ROPE_PACK_TOKEN_TILE - 1) // ROPE_PACK_TOKEN_TILE) * O_GROUPS
A_K_CHUNK = 128
A_N_CHUNK = 128
B_K_CHUNK = 128
B_N_CHUNK = 128 if T >= 128 else 256
QUANT_CHUNK = 128 if T >= 128 else (128 if T >= 64 else 256)
QUANT_TOKEN_TILE = 32
# Keep the standalone sparse-attn simulator kernels below the scheduler
# no-progress timeout; the packed HCA path below keeps its own 128-token tile.
PROJ_TOKEN_TILE = 64 if T >= 128 else T
assert T % QUANT_TOKEN_TILE == 0, "T must be divisible by QUANT_TOKEN_TILE for full-row quantization coverage"
assert T % PROJ_TOKEN_TILE == 0, "T must be divisible by PROJ_TOKEN_TILE for projection tiling"
assert (O_GROUPS * O_LORA) % 2 == 0, "2-way quant K split requires an even O_GROUPS * O_LORA width"


# Token-major sparse-attention helpers consume lowered sparse indices and
# request metadata. The public overlay entry is `prefill_sparse_attn` below.
MAX_REQS = 2
MAX_TOKENS = T
HCA_ORI_BLOCK_NUM = MAX_REQS * ORI_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = MAX_REQS * CMP_MAX_BLOCKS
HCA_GATHER_TOKEN_TILE = 4
ROPE_HALF = HALF_ROPE
SPARSE_A_K_CHUNK = A_K_CHUNK
SPARSE_A_N_CHUNK = A_N_CHUNK
SPARSE_ATTN_TOKEN_TILE = ATTN_TOKEN_TILE
SPARSE_B_K_CHUNK = B_K_CHUNK
SPARSE_B_N_CHUNK = B_N_CHUNK
SPARSE_MATMUL_ROW_PAD = MATMUL_ROW_PAD
SPARSE_MERGE_NORM_TOKEN_TILE = MERGE_NORM_TOKEN_TILE
SPARSE_ORI_MAX_BLOCKS = ORI_MAX_BLOCKS
SPARSE_CMP_MAX_BLOCKS = CMP_MAX_BLOCKS
SPARSE_PREFILL_ATTN_BLOCKS = PREFILL_ATTN_BLOCKS
SPARSE_PREFILL_ATTN_TILE = PREFILL_ATTN_TILE
SPARSE_PREFILL_SPARSE_PAD = PREFILL_SPARSE_PAD
HCA_GATHER_SPMD_BLOCKS = ((T + HCA_GATHER_TOKEN_TILE - 1) // HCA_GATHER_TOKEN_TILE) * SPARSE_PREFILL_ATTN_BLOCKS
SPARSE_PROJ_TOKEN_TILE = 128 if T >= 128 else T
SPARSE_PV_HEAD_TILE = PV_HEAD_TILE
SPARSE_QUANT_CHUNK = QUANT_CHUNK
SPARSE_QUANT_K_TILE = O_GROUPS * O_LORA // 2
SPARSE_QUANT_K_BLOCKS = (O_GROUPS * O_LORA) // SPARSE_QUANT_K_TILE
SPARSE_QUANT_SPMD_BLOCKS = (T // QUANT_TOKEN_TILE) * SPARSE_QUANT_K_BLOCKS
SPARSE_QUANT_TOKEN_TILE = QUANT_TOKEN_TILE
SPARSE_ROPE_CHUNK = ROPE_CHUNK
SPARSE_ROPE_INTERLEAVE_CHUNK = ROPE_INTERLEAVE_CHUNK
SPARSE_ROPE_PACK_SPMD_BLOCKS = ROPE_PACK_SPMD_BLOCKS
SPARSE_ROPE_PACK_TOKEN_TILE = ROPE_PACK_TOKEN_TILE
# Keep packed inverse-RoPE token tiles small enough to expose per-output-group
# AIV parallelism; larger tiles reduce task count but leave longer RoPE work on
# the critical path.
SPARSE_ROPE_TOKEN_TILE = 4
SPARSE_ROPE_APPLY_SPMD_BLOCKS = (T + SPARSE_ROPE_TOKEN_TILE - 1) // SPARSE_ROPE_TOKEN_TILE
SPARSE_TOPK = TOPK

@pl.jit.inline
def _prefill_hca_sparse_from_gathered_kv(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_PREFILL_SPARSE_PAD], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    sparse_kv: pl.Tensor[[T * SPARSE_PREFILL_SPARSE_PAD, HEAD_DIM], pl.BF16],
):
    A_K_BLOCKS = O_GROUP_IN // SPARSE_A_K_CHUNK
    A_N_BLOCKS = O_LORA // SPARSE_A_N_CHUNK
    A_AMAX_BLOCKS = O_GROUPS * A_N_BLOCKS
    B_K_BLOCKS = (O_GROUPS * O_LORA) // SPARSE_B_K_CHUNK
    B_N_BLOCKS = D // SPARSE_B_N_CHUNK

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.BF16)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)

    # Per-(token, slot) additive bias: 0 for valid raw indices, -3e38 for
    # padding. QK adds this once-built bias row instead of rescanning slot
    # validity for every head tile.
    sparse_bias = pl.create_tensor([T, SPARSE_PREFILL_SPARSE_PAD], dtype=pl.FP32)
    for bias_t0 in pl.parallel(0, T, SPARSE_QUANT_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_attn_pad_bias"):
            for bias_sb in pl.range(SPARSE_PREFILL_ATTN_BLOCKS):
                bias_start = bias_sb * SPARSE_PREFILL_ATTN_TILE
                bias_idx = pl.cast(
                    cmp_sparse_indices[bias_t0:bias_t0 + SPARSE_QUANT_TOKEN_TILE, bias_start:bias_start + SPARSE_PREFILL_ATTN_TILE],
                    target_type=pl.FP32,
                )
                bias_flag = pl.minimum(pl.maximum(pl.add(bias_idx, 1.0), 0.0), 1.0)
                sparse_bias[bias_t0:bias_t0 + SPARSE_QUANT_TOKEN_TILE, bias_start:bias_start + SPARSE_PREFILL_ATTN_TILE] = pl.mul(
                    pl.sub(bias_flag, 1.0),
                    3.0e38,
                )

    # Stage 2: causal prefill attention, tiled across context rows.
    for attn_t0 in pl.parallel(0, T, SPARSE_ATTN_TOKEN_TILE):
        for h0 in pl.parallel(0, H, SPARSE_MATMUL_ROW_PAD):
            prefill_exp = pl.create_tensor(
                [SPARSE_ATTN_TOKEN_TILE * SPARSE_MATMUL_ROW_PAD * SPARSE_PREFILL_ATTN_BLOCKS, SPARSE_PREFILL_ATTN_TILE],
                dtype=pl.BF16,
            )
            prefill_blk_mi = pl.create_tensor(
                [SPARSE_ATTN_TOKEN_TILE * SPARSE_MATMUL_ROW_PAD * SPARSE_PREFILL_ATTN_BLOCKS, 1],
                dtype=pl.FP32,
            )
            prefill_blk_li = pl.create_tensor(
                [SPARSE_ATTN_TOKEN_TILE * SPARSE_MATMUL_ROW_PAD * SPARSE_PREFILL_ATTN_BLOCKS, 1],
                dtype=pl.FP32,
            )
            prefill_blk_oi0 = pl.create_tensor([SPARSE_ATTN_TOKEN_TILE * SPARSE_MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32)
            prefill_blk_oi1 = pl.create_tensor([SPARSE_ATTN_TOKEN_TILE * SPARSE_MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32)
            prefill_blk_oi2 = pl.create_tensor([SPARSE_ATTN_TOKEN_TILE * SPARSE_MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_attn_qk_softmax_tile"):
                for qk_dt in pl.range(SPARSE_ATTN_TOKEN_TILE):
                    qk_t = attn_t0 + qk_dt
                    if qk_t < num_tokens:
                        qk_head_row = qk_t * H + h0
                        qk_q_batch = q_flat[qk_head_row : qk_head_row + SPARSE_MATMUL_ROW_PAD, 0 : HEAD_DIM]
                        qk_kv_base = qk_t * SPARSE_PREFILL_SPARSE_PAD

                        for qk_sb in pl.range(SPARSE_PREFILL_ATTN_BLOCKS):
                            qk_tile_start = qk_sb * SPARSE_PREFILL_ATTN_TILE
                            qk_tile_valid_raw = pl.read(cmp_sparse_indices, [qk_t, qk_tile_start])
                            if qk_tile_valid_raw >= 0:
                                qk_kv_tile = sparse_kv[
                                    qk_kv_base + qk_tile_start : qk_kv_base + qk_tile_start + SPARSE_PREFILL_ATTN_TILE,
                                    0 : HEAD_DIM,
                                ]
                                qk_raw_scores = pl.matmul(qk_q_batch, qk_kv_tile, b_trans=True, out_dtype=pl.FP32)
                                qk_block_row = (
                                    qk_dt * SPARSE_MATMUL_ROW_PAD * SPARSE_PREFILL_ATTN_BLOCKS + qk_sb * SPARSE_MATMUL_ROW_PAD
                                )
                                qk_scaled_scores = pl.mul(qk_raw_scores, SOFTMAX_SCALE)
                                qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_tile_start:qk_tile_start + SPARSE_PREFILL_ATTN_TILE]
                                softmax_scores = pl.add(
                                    qk_scaled_scores,
                                    pl.col_expand(
                                        pl.full(
                                            [SPARSE_MATMUL_ROW_PAD, SPARSE_PREFILL_ATTN_TILE],
                                            dtype=pl.FP32,
                                            value=0.0,
                                        ),
                                        qk_bias_row,
                                    )
                                )
                                softmax_mi = pl.row_max(softmax_scores)
                                softmax_exp_scores = pl.exp(pl.row_expand_sub(softmax_scores, softmax_mi))
                                softmax_exp_scores_bf16 = pl.cast(softmax_exp_scores, target_type=pl.BF16)
                                softmax_li = pl.row_sum(pl.cast(softmax_exp_scores_bf16, target_type=pl.FP32))
                                prefill_blk_mi = pl.assemble(prefill_blk_mi, softmax_mi, [qk_block_row, 0])
                                prefill_blk_li = pl.assemble(prefill_blk_li, softmax_li, [qk_block_row, 0])
                                prefill_exp = pl.assemble(prefill_exp, softmax_exp_scores_bf16, [qk_block_row, 0])

            for pv_h_delta in pl.parallel(0, SPARSE_MATMUL_ROW_PAD, SPARSE_PV_HEAD_TILE):
                pv_h0 = h0 + pv_h_delta
                pv_h_local = pv_h_delta

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_attn_pv_head_tile"):
                    for pv_dt in pl.range(SPARSE_ATTN_TOKEN_TILE):
                        pv_t = attn_t0 + pv_dt
                        if pv_t < num_tokens:
                            pv_kv_base = pv_t * SPARSE_PREFILL_SPARSE_PAD

                            for pv_sb in pl.range(SPARSE_PREFILL_ATTN_BLOCKS):
                                pv_tile_start = pv_sb * SPARSE_PREFILL_ATTN_TILE
                                pv_tile_valid_raw = pl.read(cmp_sparse_indices, [pv_t, pv_tile_start])
                                if pv_tile_valid_raw >= 0:
                                    pv_kv_tile = sparse_kv[
                                        pv_kv_base + pv_tile_start : pv_kv_base + pv_tile_start + SPARSE_PREFILL_ATTN_TILE,
                                        0 : HEAD_DIM,
                                    ]
                                    pv_block_row = (
                                        pv_dt * SPARSE_MATMUL_ROW_PAD * SPARSE_PREFILL_ATTN_BLOCKS
                                        + pv_sb * SPARSE_MATMUL_ROW_PAD
                                        + pv_h_local
                                    )
                                    pv_exp_scores = prefill_exp[
                                        pv_block_row : pv_block_row + SPARSE_PV_HEAD_TILE,
                                        0 : SPARSE_PREFILL_ATTN_TILE,
                                    ]
                                    pv_oi = pl.matmul(pv_exp_scores, pv_kv_tile, out_dtype=pl.FP32)
                                    pv_head_row = pv_dt * SPARSE_MATMUL_ROW_PAD + pv_h_local
                                    if pv_sb == 0:
                                        prefill_blk_oi0 = pl.assemble(prefill_blk_oi0, pv_oi, [pv_head_row, 0])
                                    if pv_sb == 1:
                                        prefill_blk_oi1 = pl.assemble(prefill_blk_oi1, pv_oi, [pv_head_row, 0])
                                    if pv_sb == 2:
                                        prefill_blk_oi2 = pl.assemble(prefill_blk_oi2, pv_oi, [pv_head_row, 0])

                for merge_norm_t_delta in pl.parallel(0, SPARSE_ATTN_TOKEN_TILE, SPARSE_MERGE_NORM_TOKEN_TILE):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_attn_merge_norm_head_tile"):
                        zero_head_tile = pl.full([SPARSE_PV_HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        for merge_norm_dt in pl.range(SPARSE_MERGE_NORM_TOKEN_TILE):
                            merge_norm_t = attn_t0 + merge_norm_t_delta + merge_norm_dt
                            merge_norm_t_local = merge_norm_t_delta + merge_norm_dt
                            merge_norm_head_row = merge_norm_t * H + pv_h0
                            merge_norm_head_row_local = merge_norm_t_local * SPARSE_MATMUL_ROW_PAD + pv_h_local
                            if merge_norm_t < num_tokens:
                                merge_norm_block_row0 = (
                                    merge_norm_t_local * SPARSE_MATMUL_ROW_PAD * SPARSE_PREFILL_ATTN_BLOCKS + pv_h_local
                                )
                                merge_norm_mi = prefill_blk_mi[
                                    merge_norm_block_row0 : merge_norm_block_row0 + SPARSE_PV_HEAD_TILE,
                                    0 : 1,
                                ]
                                merge_norm_li = prefill_blk_li[
                                    merge_norm_block_row0 : merge_norm_block_row0 + SPARSE_PV_HEAD_TILE,
                                    0 : 1,
                                ]
                                merge_norm_oi = prefill_blk_oi0[
                                    merge_norm_head_row_local : merge_norm_head_row_local + SPARSE_PV_HEAD_TILE,
                                    0 : HEAD_DIM,
                                ]

                                if SPARSE_PREFILL_ATTN_BLOCKS > 1:
                                    merge_norm_tile_start1 = SPARSE_PREFILL_ATTN_TILE
                                    merge_norm_block1_raw = pl.read(cmp_sparse_indices, [merge_norm_t, merge_norm_tile_start1])
                                    if merge_norm_block1_raw >= 0:
                                        merge_norm_block_row1 = (
                                            merge_norm_t_local * SPARSE_MATMUL_ROW_PAD * SPARSE_PREFILL_ATTN_BLOCKS
                                            + SPARSE_MATMUL_ROW_PAD
                                            + pv_h_local
                                        )
                                        merge_norm_cur_mi = prefill_blk_mi[
                                            merge_norm_block_row1 : merge_norm_block_row1 + SPARSE_PV_HEAD_TILE,
                                            0 : 1,
                                        ]
                                        merge_norm_cur_li = prefill_blk_li[
                                            merge_norm_block_row1 : merge_norm_block_row1 + SPARSE_PV_HEAD_TILE,
                                            0 : 1,
                                        ]
                                        merge_norm_cur_oi = prefill_blk_oi1[
                                            merge_norm_head_row_local : merge_norm_head_row_local + SPARSE_PV_HEAD_TILE,
                                            0 : HEAD_DIM,
                                        ]
                                        merge_norm_mi_new = pl.maximum(merge_norm_mi, merge_norm_cur_mi)
                                        merge_norm_alpha = pl.exp(pl.sub(merge_norm_mi, merge_norm_mi_new))
                                        merge_norm_beta = pl.exp(pl.sub(merge_norm_cur_mi, merge_norm_mi_new))
                                        merge_norm_li = pl.add(
                                            pl.mul(merge_norm_alpha, merge_norm_li),
                                            pl.mul(merge_norm_beta, merge_norm_cur_li),
                                        )
                                        merge_norm_oi = pl.add(
                                            pl.row_expand_mul(merge_norm_oi, merge_norm_alpha),
                                            pl.row_expand_mul(merge_norm_cur_oi, merge_norm_beta),
                                        )
                                        merge_norm_mi = merge_norm_mi_new

                                if SPARSE_PREFILL_ATTN_BLOCKS > 2:
                                    merge_norm_tile_start2 = 2 * SPARSE_PREFILL_ATTN_TILE
                                    merge_norm_block2_raw = pl.read(cmp_sparse_indices, [merge_norm_t, merge_norm_tile_start2])
                                    if merge_norm_block2_raw >= 0:
                                        merge_norm_block_row2 = (
                                            merge_norm_t_local * SPARSE_MATMUL_ROW_PAD * SPARSE_PREFILL_ATTN_BLOCKS
                                            + 2 * SPARSE_MATMUL_ROW_PAD
                                            + pv_h_local
                                        )
                                        merge_norm_cur_mi2 = prefill_blk_mi[
                                            merge_norm_block_row2 : merge_norm_block_row2 + SPARSE_PV_HEAD_TILE,
                                            0 : 1,
                                        ]
                                        merge_norm_cur_li2 = prefill_blk_li[
                                            merge_norm_block_row2 : merge_norm_block_row2 + SPARSE_PV_HEAD_TILE,
                                            0 : 1,
                                        ]
                                        merge_norm_cur_oi2 = prefill_blk_oi2[
                                            merge_norm_head_row_local : merge_norm_head_row_local + SPARSE_PV_HEAD_TILE,
                                            0 : HEAD_DIM,
                                        ]
                                        merge_norm_mi_new2 = pl.maximum(merge_norm_mi, merge_norm_cur_mi2)
                                        merge_norm_alpha2 = pl.exp(pl.sub(merge_norm_mi, merge_norm_mi_new2))
                                        merge_norm_beta2 = pl.exp(pl.sub(merge_norm_cur_mi2, merge_norm_mi_new2))
                                        merge_norm_li = pl.add(
                                            pl.mul(merge_norm_alpha2, merge_norm_li),
                                            pl.mul(merge_norm_beta2, merge_norm_cur_li2),
                                        )
                                        merge_norm_oi = pl.add(
                                            pl.row_expand_mul(merge_norm_oi, merge_norm_alpha2),
                                            pl.row_expand_mul(merge_norm_cur_oi2, merge_norm_beta2),
                                        )
                                        merge_norm_mi = merge_norm_mi_new2

                                merge_norm_sink_bias = pl.reshape(attn_sink[pv_h0 : pv_h0 + SPARSE_PV_HEAD_TILE], [SPARSE_PV_HEAD_TILE, 1])
                                merge_norm_sink_tile = pl.add(pl.sub(merge_norm_mi, merge_norm_mi), merge_norm_sink_bias)
                                merge_norm_denom = pl.add(
                                    merge_norm_li,
                                    pl.exp(pl.sub(merge_norm_sink_tile, merge_norm_mi)),
                                )
                                merge_norm_out = pl.row_expand_div(merge_norm_oi, merge_norm_denom)
                                attn_stage_row = pl.cast(
                                    merge_norm_out[0 : SPARSE_PV_HEAD_TILE, 0 : HEAD_DIM],
                                    target_type=pl.BF16,
                                )
                            else:
                                attn_stage_row = zero_head_tile

                            attn_rope_stage = pl.assemble(
                                attn_rope_stage,
                                attn_stage_row[0 : SPARSE_PV_HEAD_TILE, NOPE_DIM:HEAD_DIM],
                                [merge_norm_head_row, 0],
                            )

                            for merge_norm_head_i in pl.range(SPARSE_PV_HEAD_TILE):
                                merge_norm_global_head = pv_h0 + merge_norm_head_i
                                merge_norm_g = merge_norm_global_head // HEADS_PER_GROUP
                                merge_norm_hh = merge_norm_global_head - merge_norm_g * HEADS_PER_GROUP
                                merge_norm_pack_row = merge_norm_g * T + merge_norm_t
                                merge_norm_head_col = merge_norm_hh * HEAD_DIM
                                o_packed = pl.assemble(
                                    o_packed,
                                    attn_stage_row[merge_norm_head_i : merge_norm_head_i + 1, 0:NOPE_DIM],
                                    [merge_norm_pack_row, merge_norm_head_col],
                                )

    # Stage 3: inverse RoPE on the rope slice of the attention output.
    # Split by token tile and output group so the long vector RoPE task fans
    # out across all AIV lanes instead of processing all heads in one task.
    # In-kernel interleaved swap-gather (ported from decode_sparse_attn): one
    # gather (j^1 swap) + dup-gathered cos/sin replaces the de-interleave/rotate/
    # re-interleave mask gather+scatter; out[j] = x[j]*cos[j>>1] + x[j^1]*sign[j]
    # *sin[j>>1] with sign = [+1,-1,...] (conjugate/inverse rotation). All H heads
    # of a token share cos/sin so they rotate together; rope_buf is BF16 (halves
    # GM traffic, drops the rope_pack down-cast). Bit-equivalent to the mask path.
    rope_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.BF16)
    for rope_apply_block in pl.spmd(SPARSE_ROPE_APPLY_SPMD_BLOCKS, name_hint="prefill_hca_rope_apply_assemble_group"):
        rope_apply_t0 = rope_apply_block * SPARSE_ROPE_TOKEN_TILE
        # swap_idx (j^1), sign and dup_idx (j>>1) are chunk-independent column
        # patterns from pl.arange -- build once per task.
        sp_ones = pl.full([H, SPARSE_ROPE_INTERLEAVE_CHUNK], dtype=pl.FP32, value=1.0)
        sp_col = pl.col_expand_mul(sp_ones, pl.cast(pl.arange(0, [1, SPARSE_ROPE_INTERLEAVE_CHUNK], dtype=pl.INT32), target_type=pl.FP32))
        sp_dup_f = pl.cast(pl.cast(pl.mul(sp_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sp_dup_idx = pl.cast(sp_dup_f, target_type=pl.INT32)
        sp_lane = pl.sub(sp_col, pl.mul(sp_dup_f, 2.0))
        sp_swap_idx = pl.cast(pl.sub(pl.add(sp_col, 1.0), pl.mul(sp_lane, 2.0)), target_type=pl.INT32)
        sp_sign = pl.neg(pl.sub(pl.mul(sp_lane, 2.0), 1.0))
        for rope_apply_dt in pl.range(SPARSE_ROPE_TOKEN_TILE):
            rope_apply_t = rope_apply_t0 + rope_apply_dt
            if rope_apply_t < T:
                rope_apply_head_row = rope_apply_t * H

                for rope_asm_r0 in pl.range(0, ROPE_HALF, SPARSE_ROPE_CHUNK):
                    rope_c0 = 2 * rope_asm_r0
                    r_tile_fp32 = pl.cast(
                        attn_rope_stage[
                            rope_apply_head_row : rope_apply_head_row + H,
                            rope_c0 : rope_c0 + SPARSE_ROPE_INTERLEAVE_CHUNK,
                        ],
                        target_type=pl.FP32,
                    )
                    r_cos = pl.cast(freqs_cos[rope_apply_t : rope_apply_t + 1, rope_asm_r0 : rope_asm_r0 + SPARSE_ROPE_CHUNK], target_type=pl.FP32)
                    r_sin = pl.cast(freqs_sin[rope_apply_t : rope_apply_t + 1, rope_asm_r0 : rope_asm_r0 + SPARSE_ROPE_CHUNK], target_type=pl.FP32)
                    r_cos_h = pl.col_expand_mul(pl.full([H, SPARSE_ROPE_CHUNK], dtype=pl.FP32, value=1.0), r_cos)
                    r_sin_h = pl.col_expand_mul(pl.full([H, SPARSE_ROPE_CHUNK], dtype=pl.FP32, value=1.0), r_sin)
                    r_cos_il = pl.gather(r_cos_h, dim=-1, index=sp_dup_idx)
                    r_sin_il = pl.gather(r_sin_h, dim=-1, index=sp_dup_idx)
                    r_swapped = pl.gather(r_tile_fp32, dim=-1, index=sp_swap_idx)
                    r_rot = pl.add(pl.mul(r_tile_fp32, r_cos_il), pl.mul(pl.mul(r_swapped, sp_sign), r_sin_il))
                    r_rot = pl.cast(r_rot, target_type=pl.BF16, mode="rint")
                    rope_buf[rope_apply_head_row : rope_apply_head_row + H, rope_c0 : rope_c0 + SPARSE_ROPE_INTERLEAVE_CHUNK] = r_rot

    # Pack the per-head rope into o_packed's strided rope columns. For a fixed
    # head, the rope segment lands at the SAME columns for every token, so write
    # all T tokens of a head in one [T, ROPE_DIM] strided store (ported from
    # decode_sparse_attn) instead of T separate [1, ROPE_DIM] assembles.
    rope_buf_3d = pl.reshape(rope_buf, [T, H, ROPE_DIM])
    for rope_pack_gh in pl.spmd(H, name_hint="prefill_hca_rope_pack_group_spmd"):
        rope_pack_g = rope_pack_gh // HEADS_PER_GROUP
        rope_pack_hh = rope_pack_gh - rope_pack_g * HEADS_PER_GROUP
        rope_pack_tile = pl.reshape(rope_buf_3d[0:T, rope_pack_gh : rope_pack_gh + 1, 0:ROPE_DIM], [T, ROPE_DIM])
        rope_pack_col = rope_pack_hh * HEAD_DIM + NOPE_DIM
        o_packed[rope_pack_g * T : rope_pack_g * T + T, rope_pack_col : rope_pack_col + ROPE_DIM] = rope_pack_tile

    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    o_r_amax_parts = pl.create_tensor([A_AMAX_BLOCKS, T], dtype=pl.FP32)
    o_r_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)

    # Stage 5: grouped BF16 projection `o_packed @ wo_a^T`.
    for g in pl.parallel(0, O_GROUPS, 1):
        row_base_o = g * T
        out_col_g = g * O_LORA

        for nb in pl.parallel(0, A_N_BLOCKS, 1):
            n0 = nb * SPARSE_A_N_CHUNK

            for proj_t0 in pl.parallel(0, T, SPARSE_PROJ_TOKEN_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_stage_a_accum_tile"):
                    xa0_chunk = o_packed[row_base_o + proj_t0:row_base_o + proj_t0 + SPARSE_PROJ_TOKEN_TILE, 0:SPARSE_A_K_CHUNK]
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + SPARSE_A_N_CHUNK, 0:SPARSE_A_K_CHUNK]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, A_K_BLOCKS, stage=2):
                        k0 = kb * SPARSE_A_K_CHUNK
                        xa_k_chunk = o_packed[
                            row_base_o + proj_t0:row_base_o + proj_t0 + SPARSE_PROJ_TOKEN_TILE,
                            k0:k0 + SPARSE_A_K_CHUNK,
                        ]
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + SPARSE_A_N_CHUNK, k0:k0 + SPARSE_A_K_CHUNK]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_stage_a_store_amax_tile"):
                    acc_a_2d = pl.reshape(acc_a, [SPARSE_PROJ_TOKEN_TILE, SPARSE_A_N_CHUNK])
                    acc_a_bf16 = pl.cast(acc_a_2d, target_type=pl.BF16)
                    o_r[proj_t0:proj_t0 + SPARSE_PROJ_TOKEN_TILE, out_col_g + n0:out_col_g + n0 + SPARSE_A_N_CHUNK] = acc_a_bf16
                    acc_a_f32 = pl.cast(acc_a_bf16, target_type=pl.FP32)
                    acc_a_abs = pl.maximum(acc_a_f32, pl.neg(acc_a_f32))
                    acc_a_amax = pl.reshape(pl.row_max(acc_a_abs), [1, SPARSE_PROJ_TOKEN_TILE])
                    amax_part_row = g * A_N_BLOCKS + nb
                    o_r_amax_parts[
                        amax_part_row:amax_part_row + 1,
                        proj_t0:proj_t0 + SPARSE_PROJ_TOKEN_TILE,
                    ] = acc_a_amax

    # Stage 6: per-row symmetric INT8 activation quantization. Split the
    # output-rank dimension into two SPMD blocks, matching the decode path, so
    # quantization does not serialize the full O_GROUPS * O_LORA width in one
    # task.
    for quant_block in pl.spmd(SPARSE_QUANT_SPMD_BLOCKS, name_hint="prefill_hca_stage_b_quant_k_tile"):
        quant_t_block = quant_block // SPARSE_QUANT_K_BLOCKS
        quant_k_block = quant_block - quant_t_block * SPARSE_QUANT_K_BLOCKS
        quant_t0 = quant_t_block * SPARSE_QUANT_TOKEN_TILE
        quant_k0 = quant_k_block * SPARSE_QUANT_K_TILE

        or_amax = pl.full([1, SPARSE_QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for ab in pl.range(0, A_AMAX_BLOCKS, 1):
            or_a_part = o_r_amax_parts[ab:ab + 1, quant_t0:quant_t0 + SPARSE_QUANT_TOKEN_TILE]
            or_amax = pl.maximum(or_amax, or_a_part)
        or_sq_row = pl.div(pl.full([1, SPARSE_QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
        or_scale_dq = pl.reshape(pl.recip(or_sq_row), [SPARSE_QUANT_TOKEN_TILE, 1])
        if quant_k_block == 0:
            o_r_scale_dq[quant_t0:quant_t0 + SPARSE_QUANT_TOKEN_TILE, 0:1] = or_scale_dq
        or_sq_col = pl.reshape(or_sq_row, [SPARSE_QUANT_TOKEN_TILE, 1])
        for k1 in pl.range(quant_k0, quant_k0 + SPARSE_QUANT_K_TILE, SPARSE_QUANT_CHUNK):
            or_q_f32 = pl.cast(
                o_r[quant_t0:quant_t0 + SPARSE_QUANT_TOKEN_TILE, k1:k1 + SPARSE_QUANT_CHUNK],
                target_type=pl.FP32,
            )
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[quant_t0:quant_t0 + SPARSE_QUANT_TOKEN_TILE, k1:k1 + SPARSE_QUANT_CHUNK] = pl.cast(
                or_q_half,
                target_type=pl.INT8,
                mode="trunc",
            )

    # Stage 7: INT8 output projection and dequantization.
    for nb in pl.parallel(0, B_N_BLOCKS, 1):
        n0 = nb * SPARSE_B_N_CHUNK

        for proj_t0 in pl.parallel(0, T, SPARSE_PROJ_TOKEN_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_stage_b_accum_tile"):
                xb0_chunk = o_r_i8[proj_t0:proj_t0 + SPARSE_PROJ_TOKEN_TILE, 0:SPARSE_B_K_CHUNK]
                wb0_chunk = wo_b[n0:n0 + SPARSE_B_N_CHUNK, 0:SPARSE_B_K_CHUNK]
                acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
                for kb in pl.pipeline(1, B_K_BLOCKS, stage=2):
                    k0 = kb * SPARSE_B_K_CHUNK
                    xb_k_chunk = o_r_i8[proj_t0:proj_t0 + SPARSE_PROJ_TOKEN_TILE, k0:k0 + SPARSE_B_K_CHUNK]
                    wb_k_chunk = wo_b[n0:n0 + SPARSE_B_N_CHUNK, k0:k0 + SPARSE_B_K_CHUNK]
                    acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_stage_b_store_tile"):
                wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + SPARSE_B_N_CHUNK], [1, SPARSE_B_N_CHUNK])
                attn_chunk = pl.cast(acc_b, target_type=pl.FP32, mode="none")
                attn_scale_tile = o_r_scale_dq[proj_t0:proj_t0 + SPARSE_PROJ_TOKEN_TILE, 0:1]
                attn_chunk = pl.col_expand_mul(pl.row_expand_mul(attn_chunk, attn_scale_tile), wb_scale_chunk)
                attn_out[proj_t0:proj_t0 + SPARSE_PROJ_TOKEN_TILE, n0:n0 + SPARSE_B_N_CHUNK] = pl.cast(
                    attn_chunk,
                    target_type=pl.BF16,
                    mode="rint",
                )

    return attn_out




@pl.jit.inline
def prefill_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[MAX_TOKENS, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Unified token-major sparse attention with current-suffix KV overlay.

    Raw index contract for this wrapper:
      -1                              invalid
      [0, WIN)                        historical sliding-window ring KV
      [WIN, WIN + MAX_TOKENS)         current suffix overlay row
      [WIN + MAX_TOKENS, ...)         compressed KV slot

    HCA/CSA use all three sources. SWA is the two-source subset and must provide
    dummy compressed tensors while keeping compressed raw indices unreachable.
    """
    ori_kv_flat = pl.reshape(ori_kv, [HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [MAX_REQS * SPARSE_ORI_MAX_BLOCKS])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [MAX_REQS * SPARSE_CMP_MAX_BLOCKS])

    sparse_kv = pl.create_tensor([T * SPARSE_PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    sparse_indices_eff = pl.create_tensor([T, SPARSE_PREFILL_SPARSE_PAD], dtype=pl.INT32)

    # Stage 1: gather historical ring rows, current suffix overlay rows, and compressed rows.
    for gather_block in pl.spmd(HCA_GATHER_SPMD_BLOCKS, name_hint="prefill_hca_overlay_gather_kv_block"):
        gather_token_block = gather_block // SPARSE_PREFILL_ATTN_BLOCKS
        gather_sb = gather_block - gather_token_block * SPARSE_PREFILL_ATTN_BLOCKS
        gather_t0 = gather_token_block * HCA_GATHER_TOKEN_TILE
        gather_k0 = gather_sb * SPARSE_PREFILL_ATTN_TILE
        zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for gather_dt in pl.range(HCA_GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                if gather_t < num_tokens:
                    gather_len = pl.read(cmp_sparse_lens, [gather_t])
                    gather_len_eff = pl.cast(0, pl.INT32)
                    if gather_len > 0:
                        gather_len_eff = gather_len
                    gather_b = pl.cast(pl.read(token_to_request, [gather_t]), pl.INDEX)
                    for gather_ki in pl.pipeline(0, SPARSE_PREFILL_ATTN_TILE, stage=4):
                        gather_k = gather_k0 + gather_ki
                        gather_raw = pl.cast(-1, pl.INT32)
                        if gather_k < SPARSE_TOPK:
                            if gather_k < gather_len_eff:
                                gather_raw = pl.read(cmp_sparse_indices, [gather_t, gather_k])
                        pl.write(sparse_indices_eff, [gather_t, gather_k], gather_raw)
                        gather_dst_row = gather_t * SPARSE_PREFILL_SPARSE_PAD + gather_k
                        if gather_raw >= 0:
                            if gather_raw < WIN:
                                gather_ori_slot = gather_raw
                                gather_block_slot = gather_ori_slot // BLOCK_SIZE
                                gather_block_pos = gather_b * SPARSE_ORI_MAX_BLOCKS + gather_block_slot
                                gather_blk = pl.cast(pl.read(ori_block_table_flat, [gather_block_pos]), pl.INDEX)
                                gather_intra = gather_ori_slot - gather_block_slot * BLOCK_SIZE
                                gather_src_row = gather_blk * BLOCK_SIZE + gather_intra
                                sparse_kv = pl.assemble(
                                    sparse_kv,
                                    ori_kv_flat[gather_src_row : gather_src_row + 1, 0 : HEAD_DIM],
                                    [gather_dst_row, 0],
                                )
                            elif gather_raw < WIN + MAX_TOKENS:
                                gather_overlay_row = pl.cast(gather_raw - WIN, pl.INDEX)
                                sparse_kv = pl.assemble(
                                    sparse_kv,
                                    kv_overlay[gather_overlay_row : gather_overlay_row + 1, 0 : HEAD_DIM],
                                    [gather_dst_row, 0],
                                )
                            else:
                                gather_cmp_slot = gather_raw - (WIN + MAX_TOKENS)
                                gather_cmp_block_slot = gather_cmp_slot // BLOCK_SIZE
                                gather_cmp_block_pos = gather_b * SPARSE_CMP_MAX_BLOCKS + gather_cmp_block_slot
                                gather_cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [gather_cmp_block_pos]), pl.INDEX)
                                gather_cmp_intra = gather_cmp_slot - gather_cmp_block_slot * BLOCK_SIZE
                                gather_cmp_src_row = gather_cmp_blk * BLOCK_SIZE + gather_cmp_intra
                                sparse_kv = pl.assemble(
                                    sparse_kv,
                                    cmp_kv_flat[gather_cmp_src_row : gather_cmp_src_row + 1, 0 : HEAD_DIM],
                                    [gather_dst_row, 0],
                                )
                        else:
                            sparse_kv = pl.assemble(sparse_kv, zero_kv_row, [gather_dst_row, 0])
                else:
                    for gather_ki in pl.pipeline(0, SPARSE_PREFILL_ATTN_TILE, stage=4):
                        gather_k = gather_k0 + gather_ki
                        pl.write(sparse_indices_eff, [gather_t, gather_k], pl.cast(-1, pl.INT32))
                        gather_dst_row = gather_t * SPARSE_PREFILL_SPARSE_PAD + gather_k
                        sparse_kv = pl.assemble(sparse_kv, zero_kv_row, [gather_dst_row, 0])

    attn_out = _prefill_hca_sparse_from_gathered_kv(
        q,
        sparse_indices_eff,
        attn_sink,
        num_tokens,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        sparse_kv,
    )
    return attn_out


@pl.jit.inline
def prefill_sparse_attn_padded_indices(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[MAX_TOKENS, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    """Sparse attention for kernel-generated rows that are already -1 padded.

    External serving callers should use `prefill_sparse_attn` with
    `cmp_sparse_lens`. This variant is only for internal producers such as CSA
    that generate every sparse row in-kernel and explicitly fill unused entries
    with -1, so reading the full padded row cannot consume stale memory.
    """
    ori_kv_flat = pl.reshape(ori_kv, [HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [MAX_REQS * SPARSE_ORI_MAX_BLOCKS])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [MAX_REQS * SPARSE_CMP_MAX_BLOCKS])

    sparse_kv = pl.create_tensor([T * SPARSE_PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    sparse_indices_eff = pl.create_tensor([T, SPARSE_PREFILL_SPARSE_PAD], dtype=pl.INT32)

    for gather_block in pl.spmd(HCA_GATHER_SPMD_BLOCKS, name_hint="prefill_hca_overlay_gather_kv_block"):
        gather_token_block = gather_block // SPARSE_PREFILL_ATTN_BLOCKS
        gather_sb = gather_block - gather_token_block * SPARSE_PREFILL_ATTN_BLOCKS
        gather_t0 = gather_token_block * HCA_GATHER_TOKEN_TILE
        gather_k0 = gather_sb * SPARSE_PREFILL_ATTN_TILE
        zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for gather_dt in pl.range(HCA_GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                if gather_t < num_tokens:
                    gather_b = pl.cast(pl.read(token_to_request, [gather_t]), pl.INDEX)
                    for gather_ki in pl.pipeline(0, SPARSE_PREFILL_ATTN_TILE, stage=4):
                        gather_k = gather_k0 + gather_ki
                        gather_raw = pl.cast(-1, pl.INT32)
                        if gather_k < SPARSE_TOPK:
                            gather_raw = pl.read(cmp_sparse_indices, [gather_t, gather_k])
                        pl.write(sparse_indices_eff, [gather_t, gather_k], gather_raw)
                        gather_dst_row = gather_t * SPARSE_PREFILL_SPARSE_PAD + gather_k
                        if gather_raw >= 0:
                            if gather_raw < WIN:
                                gather_ori_slot = gather_raw
                                gather_block_slot = gather_ori_slot // BLOCK_SIZE
                                gather_block_pos = gather_b * SPARSE_ORI_MAX_BLOCKS + gather_block_slot
                                gather_blk = pl.cast(pl.read(ori_block_table_flat, [gather_block_pos]), pl.INDEX)
                                gather_intra = gather_ori_slot - gather_block_slot * BLOCK_SIZE
                                gather_src_row = gather_blk * BLOCK_SIZE + gather_intra
                                sparse_kv = pl.assemble(
                                    sparse_kv,
                                    ori_kv_flat[gather_src_row : gather_src_row + 1, 0 : HEAD_DIM],
                                    [gather_dst_row, 0],
                                )
                            elif gather_raw < WIN + MAX_TOKENS:
                                gather_overlay_row = pl.cast(gather_raw - WIN, pl.INDEX)
                                sparse_kv = pl.assemble(
                                    sparse_kv,
                                    kv_overlay[gather_overlay_row : gather_overlay_row + 1, 0 : HEAD_DIM],
                                    [gather_dst_row, 0],
                                )
                            else:
                                gather_cmp_slot = gather_raw - (WIN + MAX_TOKENS)
                                gather_cmp_block_slot = gather_cmp_slot // BLOCK_SIZE
                                gather_cmp_block_pos = gather_b * SPARSE_CMP_MAX_BLOCKS + gather_cmp_block_slot
                                gather_cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [gather_cmp_block_pos]), pl.INDEX)
                                gather_cmp_intra = gather_cmp_slot - gather_cmp_block_slot * BLOCK_SIZE
                                gather_cmp_src_row = gather_cmp_blk * BLOCK_SIZE + gather_cmp_intra
                                sparse_kv = pl.assemble(
                                    sparse_kv,
                                    cmp_kv_flat[gather_cmp_src_row : gather_cmp_src_row + 1, 0 : HEAD_DIM],
                                    [gather_dst_row, 0],
                                )
                        else:
                            sparse_kv = pl.assemble(sparse_kv, zero_kv_row, [gather_dst_row, 0])
                else:
                    for gather_ki in pl.pipeline(0, SPARSE_PREFILL_ATTN_TILE, stage=4):
                        gather_k = gather_k0 + gather_ki
                        pl.write(sparse_indices_eff, [gather_t, gather_k], pl.cast(-1, pl.INT32))
                        gather_dst_row = gather_t * SPARSE_PREFILL_SPARSE_PAD + gather_k
                        sparse_kv = pl.assemble(sparse_kv, zero_kv_row, [gather_dst_row, 0])

    attn_out = _prefill_hca_sparse_from_gathered_kv(
        q,
        sparse_indices_eff,
        attn_sink,
        num_tokens,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        sparse_kv,
    )
    return attn_out


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis."""
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


@pl.jit
def prefill_sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[MAX_TOKENS, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    return prefill_sparse_attn(
        q,
        ori_kv,
        ori_block_table,
        kv_overlay,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        cmp_sparse_lens,
        attn_sink,
        token_to_request,
        num_tokens,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )


def golden_prefill_sparse_attn(tensors):
    """Self-contained torch reference for the unified overlay sparse-attn entry."""
    import torch

    num_tokens = int(tensors["num_tokens"])
    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    kv_overlay = tensors["kv_overlay"].float()
    cmp_kv = tensors["cmp_kv"].float()
    ori_block_table = tensors["ori_block_table"]
    cmp_block_table = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    cmp_sparse_lens = tensors["cmp_sparse_lens"]
    token_to_request = tensors["token_to_request"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(num_tokens):
        req = int(token_to_request[t].item())
        gathered = []
        sparse_len = max(0, min(int(cmp_sparse_lens[t].item()), SPARSE_PREFILL_SPARSE_PAD, SPARSE_TOPK))
        for raw_i in cmp_sparse_indices[t, :sparse_len].tolist():
            raw = int(raw_i)
            if raw < 0:
                continue
            if raw < WIN:
                block_id = int(ori_block_table[req, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[block_id, intra, 0])
            elif raw < WIN + MAX_TOKENS:
                overlay_t = raw - WIN
                if 0 <= overlay_t < num_tokens:
                    gathered.append(kv_overlay[overlay_t])
            else:
                cmp_slot = raw - (WIN + MAX_TOKENS)
                if cmp_slot < 0 or cmp_slot >= HCA_CMP_BLOCK_NUM * BLOCK_SIZE:
                    continue
                block_id = int(cmp_block_table[req, cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[block_id, intra, 0])

        if not gathered:
            continue
        kv_rows = torch.stack(gathered, dim=0)

        mi = None
        li = None
        oi = None
        for tile_start in range(0, kv_rows.shape[0], SPARSE_PREFILL_ATTN_TILE):
            kv_tile = kv_rows[tile_start : tile_start + SPARSE_PREFILL_ATTN_TILE]
            scores = (q[t] @ kv_tile.T) * SOFTMAX_SCALE
            cur_mi = scores.max(dim=-1, keepdim=True).values
            exp_scores_bf16 = torch.exp(scores - cur_mi).to(torch.bfloat16)
            cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
            cur_oi = exp_scores_bf16.float() @ kv_tile.to(torch.bfloat16).float()
            if mi is None:
                mi = cur_mi
                li = cur_li
                oi = cur_oi
            else:
                mi_new = torch.maximum(mi, cur_mi)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                li = alpha * li + beta * cur_li
                oi = oi * alpha + cur_oi * beta
                mi = mi_new

        if mi is not None:
            denom = li + torch.exp(attn_sink.unsqueeze(-1) - mi)
            o[t] = oi / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :ROPE_HALF].unsqueeze(1)
    sin_half = sin[:, :ROPE_HALF].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(T, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("tgd,grd->tgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(1).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)
    tensors["attn_out"][:] = out.to(torch.bfloat16)


def get_prefill_cmp_valid(compress_ratio: int) -> int:
    """Map standalone ratio modes to visible compressed-cache length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio in (4, 128):
        return min(IDX_TOPK, S // compress_ratio, HCA_CMP_BLOCK_NUM * BLOCK_SIZE)
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")


def build_tensor_specs(compress_ratio: int = DEFAULT_COMPRESS_RATIO):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_token_rope_tables

    num_tokens = T
    cmp_valid = get_prefill_cmp_valid(compress_ratio)
    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, compress_ratio, dtype=torch.bfloat16)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(
        shared_freqs_cos,
        shared_freqs_sin,
        torch.arange(T, dtype=torch.int32),
    )

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale
    def init_q():
        return seeded_uniform((T, H, HEAD_DIM), 1, 0.05).to(torch.bfloat16)
    def init_ori_kv():
        return seeded_uniform((HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 2, 0.05).to(torch.bfloat16)
    def init_ori_block_table():
        table = torch.zeros(MAX_REQS, SPARSE_ORI_MAX_BLOCKS, dtype=torch.int32)
        for req in range(MAX_REQS):
            for blk in range(SPARSE_ORI_MAX_BLOCKS):
                table[req, blk] = req * SPARSE_ORI_MAX_BLOCKS + blk
        return table
    def init_kv_overlay():
        return seeded_uniform((MAX_TOKENS, HEAD_DIM), 3, 0.05).to(torch.bfloat16)
    def init_cmp_kv():
        return seeded_uniform((HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 4, 0.05).to(torch.bfloat16)
    def init_cmp_block_table():
        table = torch.zeros(MAX_REQS, SPARSE_CMP_MAX_BLOCKS, dtype=torch.int32)
        for req in range(MAX_REQS):
            for blk in range(SPARSE_CMP_MAX_BLOCKS):
                table[req, blk] = req * SPARSE_CMP_MAX_BLOCKS + blk
        return table
    def init_cmp_sparse_indices():
        idx = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
        for t in range(num_tokens):
            window = torch.arange(t + 1, dtype=torch.int32) + WIN
            cursor = min(window.numel(), SPARSE_PREFILL_SPARSE_PAD)
            idx[t, :cursor] = window[:cursor]
            if compress_ratio:
                comp_count = min(cmp_valid, (t + 1) // compress_ratio)
                comp_count = min(comp_count, SPARSE_PREFILL_SPARSE_PAD - cursor)
                if comp_count > 0:
                    comp = torch.arange(comp_count, dtype=torch.int32) + WIN + MAX_TOKENS
                    idx[t, cursor : cursor + comp_count] = comp
        return idx
    def init_cmp_sparse_lens():
        idx = init_cmp_sparse_indices()
        lens = torch.zeros(T, dtype=torch.int32)
        for t in range(num_tokens):
            valid = (idx[t] >= 0).nonzero()
            if valid.numel():
                lens[t] = int(valid[-1].item()) + 1
        return lens
    def init_attn_sink():
        return torch.zeros(H)
    def init_token_to_request():
        return torch.zeros(MAX_TOKENS, dtype=torch.int32)
    def init_freqs_cos():
        return shared_rope_cos.clone()
    def init_freqs_sin():
        return shared_rope_sin.clone()
    def init_wo_a():
        return seeded_uniform((O_GROUPS, O_LORA, O_GROUP_IN), 5, O_GROUP_IN ** -0.5).to(torch.bfloat16)
    def init_wo_b():
        return seeded_uniform((D, O_GROUPS * O_LORA), 6, (O_GROUPS * O_LORA) ** -0.5).to(torch.bfloat16)

    wo_b_i8, wo_b_scale = _quant_w_per_channel(init_wo_b())

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table", [MAX_REQS, SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("kv_overlay", [MAX_TOKENS, HEAD_DIM], torch.bfloat16, init_value=init_kv_overlay),
        TensorSpec("cmp_kv", [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [MAX_REQS, SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, SPARSE_TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("cmp_sparse_lens", [T], torch.int32, init_value=init_cmp_sparse_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("token_to_request", [MAX_TOKENS], torch.int32, init_value=init_token_to_request),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(
        description=(
            "Standalone DeepSeek V4 prefill sparse-attn validation. "
            "The kernel consumes prebuilt overlay raw indices; --compress-ratio only controls "
            "the deterministic fixture/golden sparse-index pattern."
        )
    )
    parser.add_argument(
        "-p", "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
        help="PyPTO compile/runtime backend for this standalone validation. Default: %(default)s.",
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=0,
        help="NPU device id passed to runtime_cfg.device_id. Under task-submit, '{}' is usually substituted here.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument(
        "--compress-ratio",
        type=int,
        default=DEFAULT_COMPRESS_RATIO,
        choices=list(SUPPORTED_COMPRESS_RATIOS),
        help=(
            "Standalone sparse-index fixture mode: 0 uses current-suffix overlay KV only; "
            "4 and 128 append deterministic visible compressed slots."
        ),
    )
    parser.add_argument(
        "--enable-l2-swimlane",
        nargs="?",
        const=4,
        default=0,
        type=int,
        metavar="PERF_LEVEL",
        help=(
            "Enable L2 swimlane profiling/report generation. May be passed without a value "
            "to use level 4, or with an explicit PERF_LEVEL."
        ),
    )
    parser.add_argument(
        "--enable-pmu",
        nargs="?",
        const=2,
        default=0,
        type=int,
        choices=[0, 1, 2, 4],
        help="Enable PMU profiling level for runtime_cfg. Omit for 0; pass without value for level 2.",
    )
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_sparse_attn_test,
        specs=build_tensor_specs(args.compress_ratio),
        golden_fn=golden_prefill_sparse_attn,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only or args.platform.endswith("sim"),
        compare_fn={"attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
