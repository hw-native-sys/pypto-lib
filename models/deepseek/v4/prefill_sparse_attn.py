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
- `[WIN, WIN + T)`: current suffix overlay KV
- `[WIN + T, ...)`: compressed KV

`cmp_sparse_lens[t]` is the authoritative usable prefix length for
`cmp_sparse_indices[t]`; any entries after that prefix are ignored even if they
look like valid raw indices. The standalone harness keeps the decode-style
`--compress-ratio {0,4,128}` as a fixture generator only. The kernel itself does
not branch on ratio; the prebuilt raw indices fully describe which KV source
each row comes from.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ


# Prefill target shape. T is fixed at 128.
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S

# Model config.
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
ROPE_HALF = HALF_ROPE
NOPE_DIM = M.nope_head_dim
IDX_TOPK = M.index_topk
WIN = M.sliding_window
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# Cache shapes.
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 4
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
PREFILL_SPARSE_TOPK = min(TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS
HCA_ORI_BLOCK_NUM = ORI_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = CMP_MAX_BLOCKS

# Kernel tiling (mirrors decode sparse-attn).
HEAD_TILE = 16                       # heads per QK/PV/merge tile
ATTN_TOKEN_TILE = 32
MERGE_NORM_TOKEN_TILE = 16
HCA_GATHER_TOKEN_TILE = 4
QUANT_TOKEN_TILE = 32
PROJ_TOKEN_TILE = 128
ROPE_OUT_TOK_TILE = T // 2
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 128
A_N_TILE = 128
B_K_TILE = 128
B_N_TILE = 128
QUANT_TILE = 128
QUANT_K_TILE = O_GROUPS * O_LORA // 2
# Sparse K split into <=3 merge blocks of PREFILL_ATTN_TILE rows.
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE

@pl.jit.inline
def _prefill_hca_sparse_from_gathered_kv(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    cmp_sparse_indices: pl.Tensor[[T, PREFILL_SPARSE_PAD], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    sparse_kv: pl.Tensor[[T * PREFILL_SPARSE_PAD, HEAD_DIM], pl.BF16],
):
    A_K_BLOCKS = O_GROUP_IN // A_K_TILE
    A_N_BLOCKS = O_LORA // A_N_TILE
    A_AMAX_BLOCKS = O_GROUPS * A_N_BLOCKS
    B_K_BLOCKS = (O_GROUPS * O_LORA) // B_K_TILE
    B_N_BLOCKS = D // B_N_TILE

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    # FP32 so the inverse rotation runs on the full-precision attention output
    # (the NOPE half still narrows to BF16 when packed into o_packed).
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)

    # Per-(token, slot) additive bias: 0 for valid raw indices, -3e38 for
    # padding. QK adds this once-built bias row instead of rescanning slot
    # validity for every head tile.
    sparse_bias = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    for bias_t0 in pl.parallel(0, T, QUANT_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_attn_pad_bias"):
            for bias_sb in pl.range(PREFILL_ATTN_BLOCKS):
                bias_start = bias_sb * PREFILL_ATTN_TILE
                bias_idx = pl.cast(
                    cmp_sparse_indices[bias_t0:bias_t0 + QUANT_TOKEN_TILE, bias_start:bias_start + PREFILL_ATTN_TILE],
                    target_type=pl.FP32,
                )
                bias_flag = pl.minimum(pl.maximum(pl.add(bias_idx, 1.0), 0.0), 1.0)
                sparse_bias[bias_t0:bias_t0 + QUANT_TOKEN_TILE, bias_start:bias_start + PREFILL_ATTN_TILE] = pl.mul(
                    pl.sub(bias_flag, 1.0),
                    3.0e38,
                )

    # Causal prefill attention over the gathered sparse KV.
    for attn_t0 in pl.parallel(0, T, ATTN_TOKEN_TILE):
        for h0 in pl.parallel(0, H, HEAD_TILE):
            blk_rows = ATTN_TOKEN_TILE * HEAD_TILE * PREFILL_ATTN_BLOCKS
            prefill_blk_mi = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
            prefill_blk_li = pl.create_tensor([blk_rows, 1], dtype=pl.FP32)
            prefill_blk_oi = pl.create_tensor([blk_rows, HEAD_DIM], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_pv"):
                for qk_dt in pl.range(ATTN_TOKEN_TILE):
                    qk_t = attn_t0 + qk_dt
                    if qk_t < num_tokens:
                        qk_head_row = qk_t * H + h0
                        qk_q_batch = q_flat[qk_head_row : qk_head_row + HEAD_TILE, 0 : HEAD_DIM]
                        qk_kv_base = qk_t * PREFILL_SPARSE_PAD

                        # Process all blocks unconditionally: an all-invalid block carries a
                        # -inf bias so it dies via beta == 0 in the merge. qk_kv_k / qk_kv_v are
                        # two views of one KV tile so QK (b_trans) and PV do not collide (#1532).
                        for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                            qk_tile_start = qk_sb * PREFILL_ATTN_TILE
                            qk_kv0 = qk_kv_base + qk_tile_start
                            qk_kv_k = sparse_kv[qk_kv0:qk_kv0 + PREFILL_ATTN_TILE, :]
                            qk_kv_v = sparse_kv[qk_kv0:qk_kv0 + PREFILL_ATTN_TILE, :]
                            qk_raw_scores = pl.matmul(qk_q_batch, qk_kv_k, b_trans=True, out_dtype=pl.FP32)
                            qk_block_row = qk_dt * HEAD_TILE * PREFILL_ATTN_BLOCKS + qk_sb * HEAD_TILE
                            qk_scaled_scores = pl.mul(qk_raw_scores, SOFTMAX_SCALE)
                            qk_bias_row = sparse_bias[qk_t:qk_t + 1, qk_tile_start:qk_tile_start + PREFILL_ATTN_TILE]
                            zero_tile = pl.full([HEAD_TILE, PREFILL_ATTN_TILE], dtype=pl.FP32, value=0.0)
                            softmax_scores = pl.add(qk_scaled_scores, pl.col_expand(zero_tile, qk_bias_row))
                            softmax_mi = pl.row_max(softmax_scores)
                            softmax_exp_scores = pl.exp(pl.row_expand_sub(softmax_scores, softmax_mi))
                            softmax_exp_scores_bf16 = pl.cast(softmax_exp_scores, target_type=pl.BF16, mode="rint")
                            # li sums the FP32 exp; only the PV matmul uses the BF16 cast.
                            softmax_li = pl.row_sum(softmax_exp_scores)
                            pv_oi = pl.matmul(softmax_exp_scores_bf16, qk_kv_v, out_dtype=pl.FP32)
                            prefill_blk_mi[qk_block_row:qk_block_row + HEAD_TILE, :] = softmax_mi
                            prefill_blk_li[qk_block_row:qk_block_row + HEAD_TILE, :] = softmax_li
                            prefill_blk_oi[qk_block_row:qk_block_row + HEAD_TILE, :] = pv_oi

            for merge_norm_t_delta in pl.parallel(0, ATTN_TOKEN_TILE, MERGE_NORM_TOKEN_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_attn_merge_norm_head_tile"):
                    zero_head_tile = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                    zero_head_tile_fp32 = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
                    for merge_norm_dt in pl.range(MERGE_NORM_TOKEN_TILE):
                        merge_norm_t = attn_t0 + merge_norm_t_delta + merge_norm_dt
                        merge_norm_t_local = merge_norm_t_delta + merge_norm_dt
                        merge_norm_head_row = merge_norm_t * H + h0
                        if merge_norm_t < num_tokens:
                            row0 = merge_norm_t_local * HEAD_TILE * PREFILL_ATTN_BLOCKS
                            merge_norm_mi = prefill_blk_mi[row0:row0 + HEAD_TILE, :]
                            merge_norm_li = prefill_blk_li[row0:row0 + HEAD_TILE, :]
                            merge_norm_oi = prefill_blk_oi[row0:row0 + HEAD_TILE, :]

                            # Merge the remaining blocks unconditionally: an all-invalid block has
                            # mi == -inf so beta == 0 leaves li/oi unchanged. Updating the carry
                            # every iteration avoids the pl.range conditional-carry phi corruption.
                            for merge_norm_sb in pl.range(1, PREFILL_ATTN_BLOCKS):
                                sb_row = row0 + merge_norm_sb * HEAD_TILE
                                cur_mi = prefill_blk_mi[sb_row:sb_row + HEAD_TILE, :]
                                cur_li = prefill_blk_li[sb_row:sb_row + HEAD_TILE, :]
                                cur_oi = prefill_blk_oi[sb_row:sb_row + HEAD_TILE, :]
                                mi_new = pl.maximum(merge_norm_mi, cur_mi)
                                alpha = pl.exp(pl.sub(merge_norm_mi, mi_new))
                                beta = pl.exp(pl.sub(cur_mi, mi_new))
                                merge_norm_li = pl.add(pl.mul(alpha, merge_norm_li), pl.mul(beta, cur_li))
                                merge_norm_oi = pl.add(pl.row_expand_mul(merge_norm_oi, alpha),
                                                       pl.row_expand_mul(cur_oi, beta))
                                merge_norm_mi = mi_new

                            sink_bias = pl.reshape(attn_sink[h0:h0 + HEAD_TILE], [HEAD_TILE, 1])
                            sink_tile = pl.add(pl.sub(merge_norm_mi, merge_norm_mi), sink_bias)
                            denom = pl.add(merge_norm_li, pl.exp(pl.sub(sink_tile, merge_norm_mi)))
                            attn_stage_full = pl.row_expand_div(merge_norm_oi, denom)[0:HEAD_TILE, :]
                            attn_stage_row = pl.cast(attn_stage_full, target_type=pl.BF16, mode="rint")
                        else:
                            attn_stage_full = zero_head_tile_fp32
                            attn_stage_row = zero_head_tile

                        attn_rope_stage[merge_norm_head_row:merge_norm_head_row + HEAD_TILE, :] = \
                            attn_stage_full[0:HEAD_TILE, NOPE_DIM:HEAD_DIM]

                        for merge_norm_head_i in pl.range(HEAD_TILE):
                            gh = h0 + merge_norm_head_i
                            g = gh // HEADS_PER_GROUP
                            pack_row = g * T + merge_norm_t
                            col = (gh - g * HEADS_PER_GROUP) * HEAD_DIM
                            o_packed[pack_row:pack_row + 1, col:col + NOPE_DIM] = \
                                attn_stage_row[merge_norm_head_i:merge_norm_head_i + 1, 0:NOPE_DIM]

    # Inverse RoPE fused with the rope-column pack: out[j] = x[j]*cos_il[j] + x[j^1]*sin_signed[j].
    # Precompute the head-invariant cos_il / sign-folded sin once, then rotate each head's rope
    # segment and store it straight into o_packed (no rope_buf round-trip).
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    for cp in pl.spmd(ROPE_HALF // ROPE_TILE, name_hint="rope_cs"):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...]
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(
            pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

    attn_rope_stage_3d = pl.reshape(attn_rope_stage, [T, H, ROPE_DIM])
    for rp_idx in pl.spmd((H // 4) * (T // ROPE_OUT_TOK_TILE), name_hint="rope"):
        rp_hg = rp_idx // (T // ROPE_OUT_TOK_TILE)
        rp_tt = rp_idx - rp_hg * (T // ROPE_OUT_TOK_TILE)
        rp_t0 = rp_tt * ROPE_OUT_TOK_TILE
        # Head-invariant swap index (j^1), built once and reused across the head group.
        sp_col = pl.col_expand_mul(
            pl.full([ROPE_OUT_TOK_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        sp_dup_f = pl.cast(pl.cast(pl.mul(sp_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sp_lane = pl.sub(sp_col, pl.mul(sp_dup_f, 2.0))                                           # j%2
        sp_swap_idx = pl.cast(pl.sub(pl.add(sp_col, 1.0), pl.mul(sp_lane, 2.0)), target_type=pl.INT32)  # j^1
        for rp_hl in pl.range(0, 4):
            rp_gh = rp_hg * 4 + rp_hl
            rp_g = rp_gh // HEADS_PER_GROUP
            rp_hh = rp_gh - rp_g * HEADS_PER_GROUP
            rp_col = rp_hh * HEAD_DIM + NOPE_DIM
            rp_o0 = rp_g * T + rp_t0
            for r_r0 in pl.range(0, ROPE_HALF, ROPE_TILE):
                c0 = 2 * r_r0
                r_tile_fp32 = pl.reshape(
                    attn_rope_stage_3d[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, rp_gh : rp_gh + 1, c0 : c0 + ROPE_INTERLEAVE_TILE],
                    [ROPE_OUT_TOK_TILE, ROPE_INTERLEAVE_TILE])
                r_cos_il = rope_cos_il[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_sin_signed = rope_sin_signed[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_swapped = pl.gather(r_tile_fp32, dim=-1, index=sp_swap_idx)
                r_rot = pl.add(pl.mul(r_tile_fp32, r_cos_il), pl.mul(r_swapped, r_sin_signed))
                r_rot = pl.cast(r_rot, target_type=pl.BF16, mode="rint")
                o_packed[rp_o0 : rp_o0 + ROPE_OUT_TOK_TILE, rp_col + c0 : rp_col + c0 + ROPE_INTERLEAVE_TILE] = r_rot

    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    o_r_amax_parts = pl.create_tensor([A_AMAX_BLOCKS, T], dtype=pl.FP32)
    o_r_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)

    # Grouped BF16 projection o_packed @ wo_a^T, with per-row amax for the INT8 quant.
    for g in pl.parallel(0, O_GROUPS, 1):
        row_base_o = g * T
        out_col_g = g * O_LORA

        for nb in pl.parallel(0, A_N_BLOCKS, 1):
            n0 = nb * A_N_TILE

            for proj_t0 in pl.parallel(0, T, PROJ_TOKEN_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a"):
                    xa0_chunk = o_packed[row_base_o + proj_t0:row_base_o + proj_t0 + PROJ_TOKEN_TILE, 0:A_K_TILE]
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + A_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, A_K_BLOCKS, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = o_packed[
                            row_base_o + proj_t0:row_base_o + proj_t0 + PROJ_TOKEN_TILE,
                            k0:k0 + A_K_TILE,
                        ]
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + A_N_TILE, k0:k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_amax"):
                    acc_a_2d = pl.reshape(acc_a, [PROJ_TOKEN_TILE, A_N_TILE])
                    o_r[proj_t0:proj_t0 + PROJ_TOKEN_TILE, out_col_g + n0:out_col_g + n0 + A_N_TILE] = acc_a_2d
                    acc_a_abs = pl.maximum(acc_a_2d, pl.neg(acc_a_2d))
                    acc_a_amax = pl.reshape(pl.row_max(acc_a_abs), [1, PROJ_TOKEN_TILE])
                    amax_part_row = g * A_N_BLOCKS + nb
                    o_r_amax_parts[
                        amax_part_row:amax_part_row + 1,
                        proj_t0:proj_t0 + PROJ_TOKEN_TILE,
                    ] = acc_a_amax

    # Per-row symmetric INT8 activation quant; K-dim split into 2 SPMD blocks.
    for quant_block in pl.spmd((T // QUANT_TOKEN_TILE) * 2, name_hint="quant"):
        quant_t_block = quant_block // 2
        quant_k_block = quant_block - quant_t_block * 2
        quant_t0 = quant_t_block * QUANT_TOKEN_TILE
        quant_k0 = quant_k_block * QUANT_K_TILE

        or_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for ab in pl.range(0, A_AMAX_BLOCKS, 1):
            or_a_part = o_r_amax_parts[ab:ab + 1, quant_t0:quant_t0 + QUANT_TOKEN_TILE]
            or_amax = pl.maximum(or_amax, or_a_part)
        or_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
        or_scale_dq = pl.reshape(pl.recip(or_sq_row), [QUANT_TOKEN_TILE, 1])
        if quant_k_block == 0:
            o_r_scale_dq[quant_t0:quant_t0 + QUANT_TOKEN_TILE, 0:1] = or_scale_dq
        or_sq_col = pl.reshape(or_sq_row, [QUANT_TOKEN_TILE, 1])
        for k1 in pl.range(quant_k0, quant_k0 + QUANT_K_TILE, QUANT_TILE):
            or_q_f32 = o_r[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE]
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_TILE] = pl.cast(
                or_q_half,
                target_type=pl.INT8,
                mode="trunc",
            )

    # INT8 output projection, then dequant to BF16.
    for nb in pl.parallel(0, B_N_BLOCKS, 1):
        n0 = nb * B_N_TILE

        for proj_t0 in pl.parallel(0, T, PROJ_TOKEN_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b"):
                xb0_chunk = o_r_i8[proj_t0:proj_t0 + PROJ_TOKEN_TILE, 0:B_K_TILE]
                wb0_chunk = wo_b[n0:n0 + B_N_TILE, 0:B_K_TILE]
                acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
                for kb in pl.pipeline(1, B_K_BLOCKS, stage=2):
                    k0 = kb * B_K_TILE
                    xb_k_chunk = o_r_i8[proj_t0:proj_t0 + PROJ_TOKEN_TILE, k0:k0 + B_K_TILE]
                    wb_k_chunk = wo_b[n0:n0 + B_N_TILE, k0:k0 + B_K_TILE]
                    acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_store"):
                wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + B_N_TILE], [1, B_N_TILE])
                attn_chunk = pl.cast(acc_b, target_type=pl.FP32, mode="none")
                attn_scale_tile = o_r_scale_dq[proj_t0:proj_t0 + PROJ_TOKEN_TILE, 0:1]
                attn_chunk = pl.col_expand_mul(pl.row_expand_mul(attn_chunk, attn_scale_tile), wb_scale_chunk)
                attn_out[proj_t0:proj_t0 + PROJ_TOKEN_TILE, n0:n0 + B_N_TILE] = pl.cast(
                    attn_chunk,
                    target_type=pl.BF16,
                    mode="rint",
                )

    return attn_out




@pl.jit.inline
def prefill_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
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
      [WIN, WIN + T)         current suffix overlay row
      [WIN + T, ...)         compressed KV slot

    HCA/CSA use all three sources. SWA is the two-source subset and must provide
    dummy compressed tensors while keeping compressed raw indices unreachable.
    """
    ori_kv_flat = pl.reshape(ori_kv, [HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])

    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    sparse_indices_eff = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.INT32)

    # Gather sliding-window, current-overlay, and compressed KV rows per token.
    for gather_block in pl.spmd(((T + HCA_GATHER_TOKEN_TILE - 1) // HCA_GATHER_TOKEN_TILE) * PREFILL_ATTN_BLOCKS, name_hint="prefill_hca_overlay_gather_kv_block"):
        gather_token_block = gather_block // PREFILL_ATTN_BLOCKS
        gather_sb = gather_block - gather_token_block * PREFILL_ATTN_BLOCKS
        gather_t0 = gather_token_block * HCA_GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for gather_dt in pl.range(HCA_GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                if gather_t < num_tokens:
                    gather_len = pl.read(cmp_sparse_lens, [gather_t])
                    gather_len_eff = pl.cast(0, pl.INT32)
                    if gather_len > 0:
                        gather_len_eff = gather_len
                    for gather_ki in pl.pipeline(0, PREFILL_ATTN_TILE, stage=4):
                        gather_k = gather_k0 + gather_ki
                        gather_raw = pl.cast(-1, pl.INT32)
                        if gather_k < TOPK:
                            if gather_k < gather_len_eff:
                                gather_raw = pl.read(cmp_sparse_indices, [gather_t, gather_k])
                        pl.write(sparse_indices_eff, [gather_t, gather_k], gather_raw)
                        gather_dst_row = gather_t * PREFILL_SPARSE_PAD + gather_k
                        if gather_raw >= 0:
                            if gather_raw < WIN:
                                gather_ori_slot = gather_raw
                                gather_block_slot = gather_ori_slot // BLOCK_SIZE
                                gather_blk = pl.cast(pl.read(ori_block_table, [gather_block_slot]), pl.INDEX)
                                gather_intra = gather_ori_slot - gather_block_slot * BLOCK_SIZE
                                gather_src_row = gather_blk * BLOCK_SIZE + gather_intra
                                sparse_kv[gather_dst_row:gather_dst_row + 1, :] = \
                                    ori_kv_flat[gather_src_row:gather_src_row + 1, :]
                            elif gather_raw < WIN + T:
                                gather_overlay_row = pl.cast(gather_raw - WIN, pl.INDEX)
                                sparse_kv[gather_dst_row:gather_dst_row + 1, :] = \
                                    kv_overlay[gather_overlay_row:gather_overlay_row + 1, :]
                            else:
                                gather_cmp_slot = gather_raw - (WIN + T)
                                gather_cmp_block_slot = gather_cmp_slot // BLOCK_SIZE
                                gather_cmp_blk = pl.cast(pl.read(cmp_block_table, [gather_cmp_block_slot]), pl.INDEX)
                                gather_cmp_intra = gather_cmp_slot - gather_cmp_block_slot * BLOCK_SIZE
                                gather_cmp_src_row = gather_cmp_blk * BLOCK_SIZE + gather_cmp_intra
                                sparse_kv[gather_dst_row:gather_dst_row + 1, :] = \
                                    cmp_kv_flat[gather_cmp_src_row:gather_cmp_src_row + 1, :]
                        else:
                            sparse_kv[gather_dst_row:gather_dst_row + 1, :] = zero_kv_row
                else:
                    for gather_ki in pl.pipeline(0, PREFILL_ATTN_TILE, stage=4):
                        gather_k = gather_k0 + gather_ki
                        pl.write(sparse_indices_eff, [gather_t, gather_k], pl.cast(-1, pl.INT32))
                        gather_dst_row = gather_t * PREFILL_SPARSE_PAD + gather_k
                        sparse_kv[gather_dst_row:gather_dst_row + 1, :] = zero_kv_row

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
    ori_block_table: pl.Tensor[[ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
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

    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    sparse_indices_eff = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.INT32)

    for gather_block in pl.spmd(((T + HCA_GATHER_TOKEN_TILE - 1) // HCA_GATHER_TOKEN_TILE) * PREFILL_ATTN_BLOCKS, name_hint="prefill_hca_overlay_gather_kv_block"):
        gather_token_block = gather_block // PREFILL_ATTN_BLOCKS
        gather_sb = gather_block - gather_token_block * PREFILL_ATTN_BLOCKS
        gather_t0 = gather_token_block * HCA_GATHER_TOKEN_TILE
        gather_k0 = gather_sb * PREFILL_ATTN_TILE
        zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for gather_dt in pl.range(HCA_GATHER_TOKEN_TILE):
            gather_t = gather_t0 + gather_dt
            if gather_t < T:
                if gather_t < num_tokens:
                    for gather_ki in pl.pipeline(0, PREFILL_ATTN_TILE, stage=4):
                        gather_k = gather_k0 + gather_ki
                        gather_raw = pl.cast(-1, pl.INT32)
                        if gather_k < TOPK:
                            gather_raw = pl.read(cmp_sparse_indices, [gather_t, gather_k])
                        pl.write(sparse_indices_eff, [gather_t, gather_k], gather_raw)
                        gather_dst_row = gather_t * PREFILL_SPARSE_PAD + gather_k
                        if gather_raw >= 0:
                            if gather_raw < WIN:
                                gather_ori_slot = gather_raw
                                gather_block_slot = gather_ori_slot // BLOCK_SIZE
                                gather_blk = pl.cast(pl.read(ori_block_table, [gather_block_slot]), pl.INDEX)
                                gather_intra = gather_ori_slot - gather_block_slot * BLOCK_SIZE
                                gather_src_row = gather_blk * BLOCK_SIZE + gather_intra
                                sparse_kv[gather_dst_row:gather_dst_row + 1, :] = \
                                    ori_kv_flat[gather_src_row:gather_src_row + 1, :]
                            elif gather_raw < WIN + T:
                                gather_overlay_row = pl.cast(gather_raw - WIN, pl.INDEX)
                                sparse_kv[gather_dst_row:gather_dst_row + 1, :] = \
                                    kv_overlay[gather_overlay_row:gather_overlay_row + 1, :]
                            else:
                                gather_cmp_slot = gather_raw - (WIN + T)
                                gather_cmp_block_slot = gather_cmp_slot // BLOCK_SIZE
                                gather_cmp_blk = pl.cast(pl.read(cmp_block_table, [gather_cmp_block_slot]), pl.INDEX)
                                gather_cmp_intra = gather_cmp_slot - gather_cmp_block_slot * BLOCK_SIZE
                                gather_cmp_src_row = gather_cmp_blk * BLOCK_SIZE + gather_cmp_intra
                                sparse_kv[gather_dst_row:gather_dst_row + 1, :] = \
                                    cmp_kv_flat[gather_cmp_src_row:gather_cmp_src_row + 1, :]
                        else:
                            sparse_kv[gather_dst_row:gather_dst_row + 1, :] = zero_kv_row
                else:
                    for gather_ki in pl.pipeline(0, PREFILL_ATTN_TILE, stage=4):
                        gather_k = gather_k0 + gather_ki
                        pl.write(sparse_indices_eff, [gather_t, gather_k], pl.cast(-1, pl.INT32))
                        gather_dst_row = gather_t * PREFILL_SPARSE_PAD + gather_k
                        sparse_kv[gather_dst_row:gather_dst_row + 1, :] = zero_kv_row

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


@pl.jit
def prefill_sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[ORI_MAX_BLOCKS], pl.INT32],
    kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
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
        num_tokens,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )


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
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(num_tokens):
        gathered = []
        sparse_len = max(0, min(int(cmp_sparse_lens[t].item()), PREFILL_SPARSE_PAD, TOPK))
        for raw_i in cmp_sparse_indices[t, :sparse_len].tolist():
            raw = int(raw_i)
            if raw < 0:
                continue
            if raw < WIN:
                block_id = int(ori_block_table[raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[block_id, intra, 0])
            elif raw < WIN + T:
                overlay_t = raw - WIN
                if 0 <= overlay_t < num_tokens:
                    gathered.append(kv_overlay[overlay_t])
            else:
                cmp_slot = raw - (WIN + T)
                if cmp_slot < 0 or cmp_slot >= HCA_CMP_BLOCK_NUM * BLOCK_SIZE:
                    continue
                block_id = int(cmp_block_table[cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[block_id, intra, 0])

        if not gathered:
            continue
        kv_rows = torch.stack(gathered, dim=0)

        mi = None
        li = None
        oi = None
        for tile_start in range(0, kv_rows.shape[0], PREFILL_ATTN_TILE):
            kv_tile = kv_rows[tile_start : tile_start + PREFILL_ATTN_TILE]
            scores = (q[t] @ kv_tile.T) * SOFTMAX_SCALE
            cur_mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - cur_mi)
            cur_li = exp_scores.sum(dim=-1, keepdim=True)
            exp_scores_bf16 = exp_scores.to(torch.bfloat16)
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

    def init_q():
        return ((torch.rand(T, H, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_ori_kv():
        return ((torch.rand(HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_ori_block_table():
        table = torch.zeros(ORI_MAX_BLOCKS, dtype=torch.int32)
        for blk in range(ORI_MAX_BLOCKS):
            table[blk] = blk
        return table
    def init_kv_overlay():
        return ((torch.rand(T, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_kv():
        return ((torch.rand(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
    def init_cmp_block_table():
        table = torch.zeros(CMP_MAX_BLOCKS, dtype=torch.int32)
        for blk in range(CMP_MAX_BLOCKS):
            table[blk] = blk
        return table
    def init_cmp_sparse_indices():
        idx = torch.full((T, TOPK), -1, dtype=torch.int32)
        for t in range(num_tokens):
            window = torch.arange(t + 1, dtype=torch.int32) + WIN
            cursor = min(window.numel(), PREFILL_SPARSE_PAD)
            idx[t, :cursor] = window[:cursor]
            if compress_ratio:
                comp_count = min(cmp_valid, (t + 1) // compress_ratio)
                comp_count = min(comp_count, PREFILL_SPARSE_PAD - cursor)
                if comp_count > 0:
                    comp = torch.arange(comp_count, dtype=torch.int32) + WIN + T
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
    def init_freqs_cos():
        return shared_rope_cos.clone()
    def init_freqs_sin():
        return shared_rope_sin.clone()
    def init_wo_a():
        return ((torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5).to(torch.bfloat16)
    def init_wo_b():
        return ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5).to(torch.bfloat16)

    wo_b_i8, wo_b_scale = _quant_w_per_channel(init_wo_b())

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table", [ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("kv_overlay", [T, HEAD_DIM], torch.bfloat16, init_value=init_kv_overlay),
        TensorSpec("cmp_kv", [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("cmp_sparse_lens", [T], torch.int32, init_value=init_cmp_sparse_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--enable-l2-swimlane", nargs="?", const=4, default=0, type=int)
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
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
        compile_only=args.compile_only,
        compare_fn={"attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
