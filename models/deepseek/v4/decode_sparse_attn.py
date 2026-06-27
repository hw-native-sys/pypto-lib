# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped output projection (decode)."""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, INT8_SCALE_MAX, INT8_AMAX_EPS


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
IDX_TOPK = M.index_topk
TOPK_FULL = WIN + IDX_TOPK           # full sparse-K width (window + indexer topk)
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# kernel-local
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 4
ORI_MAX_BLOCKS = 1  # paged-KV pool: ori (sliding-window) blocks per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 8   # paged-KV pool: compressed blocks per batch
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# tiling
VALID_TOKEN_TILE = 16
ROPE_OUT_TOK_TILE = 64
H_TILE = 16
# qk_pv cube-batch tile (M for the QK/PV matmuls). Batching QK_M_TILE head rows
# per matmul extracts the shared KV tile L1->L0 once per QK_M_TILE/H_TILE
# head-tiles (2x reuse at 32) instead of per H_TILE head-tile, then slices the
# [QK_M_TILE, ...] result back into H_TILE-row stores so the sparse_blk_* layout
# and merge_norm stay unchanged. 32 keeps the [32,128] softmax inside the 192KB
# Vec budget without a cross-core split. 64 is infeasible without further work
# (its [64,128] softmax and co-resident QK+PV L0C accumulators overflow Vec/L0C).
QK_M_TILE = 32
ATTN_K_TILE = 128
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
# proj_a cube K-frag. 256 (not 128) keeps the B-cache-line floor: B is K-contiguous
# under b_trans, so K*2B(bf16) = 512B == the a2a3 L2 line (K=128 was 256B, half a
# line -> wasted MTE2 DMA). At 256 the cube's L0A/L0B operand staging hits 100%
# (the wall); 512 would spill it for no gain (swept: K=512 net-negative).
A_K_TILE = 256
# proj_a is a pure-cube matmul scope (proj_a_mm) writing the fp32 GM intermediate
# o_r (cf. expert_routed w2 decouple), consumed directly by the fused amax+quant
# scope below; the decouple frees the cube N-frag from any vector-side UB constraint.
PROJ_A_MM_N_TILE = 128   # cube N frag; Mat/L0C have room but L0A/L0B are the wall at
                         # K=256, and a wider N raised cube exec without a TTT win (swept).
B_T_TILE = 32   # proj_b_act token tile; a device sweep found it neutral over [16, 32]
B_K_TILE = 256  # proj_b_mm cube K frag; the GEMM is not the proj_b bottleneck (see below),
                # and a device sweep found growing it (512/1024) only re-streamed more weight
                # for no TTT gain, so it stays at the cache-line-safe 256 (256 B per INT8 row).
# proj_b is decoupled into a pure-cube GEMM scope (proj_b_mm) and a pure-vector dequant
# scope (proj_b_act) meeting through the attn_acc_i32 GM intermediate, so each sizes its
# own N-fragment to its own wall (cf. the expert_routed w2 decouple). A device tile sweep
# showed the cube is NOT the bottleneck (proj_b_mm ~32us, ~78% Exec, with L0C/L1 headroom)
# while the vector dequant dominated: growing PROJ_B_ACT_N_TILE 128->1024 cut the per-N-block
# vector-setup count 4x and dropped standalone TTT ~835->~730us. The vector frag goes to
# the UB wall; the cube frag sizes to its own L0C wall (below).
PROJ_B_MM_N_TILE = 256    # cube N frag. A later device-bias-controlled sweep (paired
                          # within-device, post vector-retune) found 128->256 worth ~1.5-2%
                          # (fewer matmul setups: per-D-chunk inner N-frags 8->4; proj_b_mm
                          # exec ~33->31us). Acc = M*N*4 = 128*256*4 = 128KB rides the L0C
                          # wall exactly (fits a2a3); Mat ~192KB has room. proj_a N stayed
                          # 128: growing it was TTT-neutral (64->32 task drop offset by 2x
                          # per-task cube exec). B_K_TILE stayed 256: the K=512 cache-line
                          # fix was ambiguous (raised proj_b dispatch-prop for no clear gain).
PROJ_B_ACT_N_TILE = 512   # vector N frag for the decoupled per-group dequant+sum (proj_b_act
                          # now sums O_GROUPS INT32 partials, each x its group act scale, then
                          # x the per-channel weight scale -> BF16). 512 (not 1024) keeps the
                          # O_GROUPS-way accumulate inside UB and gives D/512 = 8 vector tasks.
QUANT_TILE = 512
# Fused amax+quant token tile. The fused scope streams o_r twice (amax pass + quant
# pass) per token-tile rather than holding the whole row, so UB stays small; 8 keeps
# the [1, QUANT_TOKEN_TILE] fp32 amax tile 32-byte aligned (8*4=32B, the alloc-tile
# row floor that a [QUANT_TOKEN_TILE, 1] column accumulator would violate). The
# full-row hold (read o_r once) needs QUANT_TOKEN_TILE<=4 for UB but >=8 for that
# alignment -- mutually exclusive -- so we stream; the 2nd pass mostly hits L2.
QUANT_TOKEN_TILE = 8
# Per-group back-to-back o_proj (manual-scope, qwen3-style fine-grained deps):
# proj_a[g] -> quant[g] (PER-GROUP amax, no global barrier) -> proj_b[g] pipeline.
# Each proj_b group-slab dequantizes its INT32 partial (per-row group act scale x
# per-channel weight scale) and FP32 atomic-adds into a zero-seeded accumulator.
SEED_N_CHUNK = 128   # attn_acc_f32 zero-seed column chunk ([T, 128] FP32 x2 pipeline = 128KB UB)
PA_NFRAGS = O_LORA // PROJ_A_MM_N_TILE   # proj_a cube N-frags per group
PB_NFRAGS = D // PROJ_B_MM_N_TILE         # proj_b total cube N-frags over D (module-level
                                          # so pl.array.create / range() get static ints)
# proj_b is one task per (D-chunk, group): the D-chunk's N-frags loop INSIDE the task,
# so the per-group split does NOT multiply the task count by N-frags. PROJ_B_D_CHUNK=1024
# keeps proj_b at O_GROUPS*(D//PROJ_B_D_CHUNK) = 32 tasks (= the baseline proj_b count),
# each doing the same total work, instead of 256 tiny tasks (more dispatch overhead).
PROJ_B_D_CHUNK = 1024
PB_DCHUNKS = D // PROJ_B_D_CHUNK
# quant is split per (group, token-chunk) so the per-group amax+quant spreads over more
# vector cores: O_GROUPS*NUM_QUANT_T_CHUNKS = 8*4 = 32 tasks.
NUM_QUANT_T_CHUNKS = 4
QUANT_T_CHUNK = T // NUM_QUANT_T_CHUNKS
# proj_b_act is split per (D-region, token-block) so the O_GROUPS-way dequant+sum spreads
# over more vector cores: (D//PROJ_B_ACT_N_TILE)*(T//PROJ_B_ACT_TBLK) = 8*4 = 32 tasks.
PROJ_B_ACT_T_TILE = 16   # inner token tile for the proj_b_act O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TBLK = 32     # proj_b_act token block per task
PB_ACT_NREG = D // PROJ_B_ACT_N_TILE
PB_ACT_TBLKS = T // PROJ_B_ACT_TBLK
NEG_INF = -1.0e20

assert T % VALID_TOKEN_TILE == 0
assert T % 2 == 0
assert T % ROPE_OUT_TOK_TILE == 0  # rope-pack loop tiles tokens by ROPE_OUT_TOK_TILE
assert H % 4 == 0
assert QK_M_TILE % H_TILE == 0
assert H % QK_M_TILE == 0
assert T % QUANT_TOKEN_TILE == 0
assert H % O_GROUPS == 0
assert (O_GROUPS * O_LORA) % B_K_TILE == 0
assert D % PROJ_B_MM_N_TILE == 0, "proj_b_mm cube N-loop must cover D"
assert D % PROJ_B_D_CHUNK == 0, "proj_b D-chunk loop must cover D"
assert PROJ_B_D_CHUNK % PROJ_B_MM_N_TILE == 0, "proj_b inner N-frag loop must cover the D-chunk"
assert T % NUM_QUANT_T_CHUNKS == 0 and QUANT_T_CHUNK % QUANT_TOKEN_TILE == 0
assert T % PROJ_B_ACT_TBLK == 0 and PROJ_B_ACT_TBLK % PROJ_B_ACT_T_TILE == 0
assert D % PROJ_B_ACT_N_TILE == 0, "proj_b_act vector N-loop must cover D"
assert (O_GROUPS * O_LORA) % QUANT_TILE == 0
assert O_LORA % QUANT_TILE == 0, "per-group quant streams O_LORA in QUANT_TILE chunks"
assert O_LORA % B_K_TILE == 0, "proj_b group K-loop covers O_LORA in B_K_TILE iters"
assert D % SEED_N_CHUNK == 0


def get_standalone_cmp_valid(compress_ratio: int) -> int:
    """Map demo compress-ratio modes to the valid compressed-cache tail length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return IDX_TOPK
    if compress_ratio == 128:
        return MAX_SEQ_LEN // compress_ratio
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")


# CSA/full sparse-K width. SWA and HCA use explicit sibling modules so a
# combined decode layer can import all three variants in one Python process
# without relying on import-time config mutation and module-cache order.
TOPK = TOPK_FULL
# Floor to 2: a single sparse-K block miscompiles in pypto (S-stride cross-token
# output mixup); a 2-block build with an all-invalid 2nd block is bit-exact.
SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)
PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE
assert WIN <= TOPK <= TOPK_FULL, f"TOPK ({TOPK}) must be in [WIN={WIN}, TOPK_FULL={TOPK_FULL}]"


@pl.jit.inline
def sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    mtp_kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    # Gather the sliding-window + current-MTP overlay + compressed-cache rows
    # into a per-token packed KV list. Raw index contract:
    #   -1              invalid
    #   [0, WIN)        historical ring/window KV
    #   [WIN, WIN + S)  current MTP overlay KV for this batch
    #   [WIN + S, ...)  compressed KV slots
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([T, PADDED_TOPK], dtype=pl.FP32)

    # Additive softmax bias (0 valid / NEG_INF invalid) that qk_pv adds onto the
    # scaled scores, so invalid lanes exp to ~0 with no per-block mask multiply.
    for v_blk in pl.spmd(T // VALID_TOKEN_TILE, name_hint="build_valid"):
        v_t0 = v_blk * VALID_TOKEN_TILE
        v_idx_f = pl.cast(cmp_sparse_indices[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : TOPK], target_type=pl.FP32)
        # Index contract (line 138): raw == -1 invalid, raw >= 0 valid. min(idx, 0)
        # is -1 for invalid / 0 for valid; * -NEG_INF gives NEG_INF / 0. Bit-exact,
        # 2 vector ops instead of the add/max/min/sub clamp chain.
        sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, 0 : TOPK] = pl.mul(pl.minimum(v_idx_f, 0.0), -NEG_INF)
        if PADDED_TOPK > TOPK:
            sparse_bias[v_t0 : v_t0 + VALID_TOKEN_TILE, TOPK : PADDED_TOPK] = pl.full(
                [VALID_TOKEN_TILE, PADDED_TOPK - TOPK], dtype=pl.FP32, value=NEG_INF)

    # WAR marker (pypto-lib#481): the fused gather reads ori_kv inside qk_pv, but a
    # scalar-driven gather_row does not by itself mark the param add_inout (and an
    # in-qk_pv self-copy collides with the gather's tensor view). A separate per-token
    # no-op self-copy -- same [g_t:g_t+1] granularity the old gather_kv used -- marks
    # ori_kv add_inout before qk_pv, so the enclosing layer's in-place KV-cache
    # writeback gets its WAR edge against the gather read.
    for kvt_t in pl.spmd(T, name_hint="kv_touch"):
        kvt_row = ori_kv_flat[kvt_t : kvt_t + 1, 0 : HEAD_DIM]
        ori_kv_flat[kvt_t : kvt_t + 1, 0 : HEAD_DIM] = kvt_row

    # Flash-fusion: the per-token KV gather is folded INTO qk_pv (below) -- each
    # (token, sparse-block) gathers its ATTN_K_TILE rows straight into an L1 matmul
    # operand via pl.gather_row (GM->L1, no tmov), so the sparse_kv GM staging
    # tensor and the standalone gather_kv / gather_kv_cmp kernels are gone. The raw
    # index -> physical row contract is unchanged (see line 138); invalid (raw < 0)
    # slots gather ori_kv_flat[0] (a finite, in-range row) and are zeroed out by the
    # NEG_INF softmax bias instead of by a zero KV row -- equivalent because the bias
    # drives exp(score) to ~0, and an all-invalid block is annihilated in merge_norm
    # by beta = exp(NEG_INF - m) = 0 regardless of its (finite) numerator.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    sparse_blk_mi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * (H // H_TILE) * SPARSE_BLOCKS * H_TILE, HEAD_DIM], dtype=pl.FP32)

    for qk_t in pl.spmd(T, name_hint="qk_pv"):
        qk_b = qk_t // S
        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE
        qk_ori_base = pl.cast(pl.read(ori_block_table, [qk_b, 0]), pl.INDEX) * BLOCK_SIZE
        qk_overlay_base = qk_b * S
        # Sparse-block OUTER / head-tile INNER: gather the block's KV into L1 once,
        # then both head-batches' QK (b_trans) and PV consume the SAME normal-layout
        # tile -- one gather per (token, block), no GM staging.
        for qk_sb in pl.range(SPARSE_BLOCKS):
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
            # Kernel-driven paged gather of this block's ATTN_K_TILE rows into an L1
            # matmul operand. Raw-index -> physical-row contract is identical to the
            # old gather_kv/gather_kv_cmp; invalid (raw < 0) gathers row 0 (finite,
            # in-range) and the NEG_INF bias zeros its softmax weight.
            qk_kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
            for qk_r in pl.range(ATTN_K_TILE):
                qk_k = qk_s0 + qk_r
                # Guard padded lanes: cmp_sparse_indices is [T, TOPK], so only read
                # slots < TOPK (same bound the old gather_kv_cmp used). A padded lane
                # (qk_k >= TOPK, only when PADDED_TOPK > TOPK) has no topk entry --
                # gather row 0 and let the NEG_INF pad bias zero its softmax weight.
                if qk_k < TOPK:
                    qk_ridx = pl.read(cmp_sparse_indices, [qk_t, qk_k])
                    if qk_ridx >= 0:
                        if qk_ridx < WIN:
                            qk_src = qk_ori_base + qk_ridx
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [qk_src, 0], [1, HEAD_DIM])
                        elif qk_ridx < WIN + S:
                            qk_ov = qk_overlay_base + (qk_ridx - WIN)
                            qk_kv = pl.gather_row(qk_kv, mtp_kv_overlay, [qk_r, 0], [qk_ov, 0], [1, HEAD_DIM])
                        else:
                            qk_slot = qk_ridx - (WIN + S)
                            qk_cblk = pl.cast(pl.read(cmp_block_table, [qk_b, qk_slot // BLOCK_SIZE]), pl.INDEX)
                            qk_csrc = qk_cblk * BLOCK_SIZE + qk_slot % BLOCK_SIZE
                            qk_kv = pl.gather_row(qk_kv, cmp_kv_flat, [qk_r, 0], [qk_csrc, 0], [1, HEAD_DIM])
                    else:
                        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])
                else:
                    qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])

            # Cube-batch QK_M_TILE head rows per QK/PV matmul so the shared KV
            # tile is extracted L1->L0 once per QK_M_TILE/H_TILE head-tiles
            # (2x reuse at QK_M_TILE=32) instead of per head-tile. The
            # [QK_M_TILE, ...] softmax result is sliced back into H_TILE-row
            # stores at the SAME offsets as the per-head-tile path
            # (qk_h_idx == qk_hb * (QK_M_TILE // H_TILE) + qk_sub), so the
            # sparse_blk_* layout and merge_norm are bit-identical.
            for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                qk_h0 = qk_hb * QK_M_TILE
                qk_head_row = qk_t * H + qk_h0
                qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]
                qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                # Broadcast-add the per-block bias directly (col_expand_add) instead
                # of col_expand into a dead pl.full(0) base + a separate add.
                qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                qk_mi = pl.row_max(qk_scores)
                # Invalid lanes (NEG_INF bias, zero kv rows) exp to ~0; all-invalid
                # blocks die in the merge alpha/beta -- no mask multiply needed.
                qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                qk_li = pl.row_sum(qk_exp)
                qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                for qk_sub in pl.unroll(QK_M_TILE // H_TILE):
                    qk_h_idx = qk_hb * (QK_M_TILE // H_TILE) + qk_sub
                    qk_r0 = qk_sub * H_TILE
                    qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE
                    qk_row = qk_blk_base + qk_sb * H_TILE
                    sparse_blk_mi[qk_row : qk_row + H_TILE, 0 : 1] = qk_mi[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                    sparse_blk_li[qk_row : qk_row + H_TILE, 0 : 1] = qk_li[qk_r0 : qk_r0 + H_TILE, 0 : 1]
                    sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi[qk_r0 : qk_r0 + H_TILE, 0 : HEAD_DIM]

    # Online-softmax merge across sparse-K tiles, then sink-norm.
    # with-form spmd so the dispatch TaskId (merge_tid) can be an explicit dep of
    # the manual-scope proj_a tasks below (which read merge_norm's o_packed NOPE cols).
    with pl.spmd(T, name_hint="merge_norm") as merge_tid:
        m_t = pl.tile.get_block_idx()
        m_token_base = m_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE

        for m_h_idx in pl.range((H // H_TILE)):
            m_h0 = m_h_idx * H_TILE
            m_blk_base = m_token_base + m_h_idx * SPARSE_BLOCKS * H_TILE
            m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
            m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

            # Guarded so the SWA (SPARSE_BLOCKS == 1) specialization uses the
            # single block's stats directly instead of an empty merge loop.
            if SPARSE_BLOCKS > 1:
                for m_sb in pl.range(1, SPARSE_BLOCKS):
                    m_row = m_blk_base + m_sb * H_TILE
                    m_cur_mi = sparse_blk_mi[m_row : m_row + H_TILE, 0 : 1]
                    m_cur_li = sparse_blk_li[m_row : m_row + H_TILE, 0 : 1]
                    m_cur_oi = sparse_blk_oi[m_row : m_row + H_TILE, 0 : HEAD_DIM]
                    m_mi_new = pl.maximum(m_mi, m_cur_mi)
                    m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
                    m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
                    m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
                    m_oi = pl.add(pl.row_expand_mul(m_oi, m_alpha), pl.row_expand_mul(m_cur_oi, m_beta))
                    m_mi = m_mi_new

            n_sink_bias = pl.reshape(attn_sink[m_h0 : m_h0 + H_TILE], [H_TILE, 1])
            n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
            n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
            n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]
            n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
            n_rope_row = m_t * H + m_h0
            attn_rope_stage[n_rope_row : n_rope_row + H_TILE, 0 : ROPE_DIM] = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]

            for n_hi in pl.range(H_TILE):
                n_gh = m_h0 + n_hi
                n_g = n_gh // HEADS_PER_GROUP
                n_hh = n_gh - n_g * HEADS_PER_GROUP
                n_pack_row = n_g * T + m_t
                n_col = n_hh * HEAD_DIM
                o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + NOPE_DIM] = n_bf16[n_hi : n_hi + 1, 0 : NOPE_DIM]

    # Precompute the head-invariant interleaved cos and sign*sin once: they depend
    # only on (token, column), not head, so building them per head would repeat the
    # same dup-gather H times on the bottleneck Vec engine. sign is folded into sin
    # (multiply by +/-1). The conjugate (inverse) rotation is:
    #   out[j] = x[j]*cos_il[j] + x[j^1]*sign[j]*sin_il[j]
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    for cp in pl.spmd(HALF_ROPE // ROPE_TILE, name_hint="rope_cs"):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...] (conjugate)
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(
            pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

    # Inverse RoPE fused with the rope-column pack: each task rotates its heads'
    # rope segments and stores them straight into o_packed's strided rope columns,
    # dropping a separate rope_pack stage and GM round-trip. cos_il / sign*sin come
    # from the pre-pass above; only swap_idx (j^1) is rebuilt per task.
    attn_rope_stage_3d = pl.reshape(attn_rope_stage, [T, H, ROPE_DIM])
    # with-form spmd so the dispatch TaskId (rope_tid) can be an explicit dep of the
    # manual-scope proj_a tasks below (which read rope's o_packed rope cols).
    with pl.spmd((H // 4) * (T // ROPE_OUT_TOK_TILE), name_hint="rope") as rope_tid:
        rp_idx = pl.tile.get_block_idx()
        rp_hg = rp_idx // (T // ROPE_OUT_TOK_TILE)
        rp_tt = rp_idx - rp_hg * (T // ROPE_OUT_TOK_TILE)
        rp_t0 = rp_tt * ROPE_OUT_TOK_TILE
        # Head-invariant swap index (j^1), built once and reused across the head
        # group -- the only per-head input is this head's strided rope slice.
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
            for r_r0 in pl.range(0, HALF_ROPE, ROPE_TILE):
                c0 = 2 * r_r0
                # This head's rope rows for this token tile (stride H).
                r_tile_fp32 = pl.reshape(
                    attn_rope_stage_3d[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, rp_gh : rp_gh + 1, c0 : c0 + ROPE_INTERLEAVE_TILE],
                    [ROPE_OUT_TOK_TILE, ROPE_INTERLEAVE_TILE])
                r_cos_il = rope_cos_il[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_sin_signed = rope_sin_signed[rp_t0 : rp_t0 + ROPE_OUT_TOK_TILE, c0 : c0 + ROPE_INTERLEAVE_TILE]
                r_swapped = pl.gather(r_tile_fp32, dim=-1, index=sp_swap_idx)
                r_rot = pl.add(pl.mul(r_tile_fp32, r_cos_il), pl.mul(r_swapped, r_sin_signed))
                # Store BF16-rounded rotated values straight into o_packed's rope
                # columns for this head (golden also rounds inverse-RoPE to bf16).
                r_rot = pl.cast(r_rot, target_type=pl.BF16, mode="rint")
                o_packed[rp_o0 : rp_o0 + ROPE_OUT_TOK_TILE, rp_col + c0 : rp_col + c0 + ROPE_INTERLEAVE_TILE] = r_rot

    # ========================================================================
    # Back-to-back grouped output projection (manual scope, PER-GROUP INT8 quant).
    #
    # Per-GROUP amax localizes the quant reduction to each O_LORA group (vs the
    # per-ROW-amax form, where a full-8192-row reduction is a hard barrier between
    # proj_a and proj_b), so the three stages PIPELINE per group with qwen3-style
    # fine-grained deps: proj_b[*, g] waits only on quant[g], which waits only on
    # proj_a[g, *] -- so proj_b's cube for group g runs while proj_a/quant of later
    # groups are still in flight (a genuine proj_a<->proj_b back-to-back GEMM).
    #
    # manual_scope SUPPRESSES auto-dep, so every edge is explicit: proj_a reads
    # o_packed (auto region) -> deps=[merge_tid, rope_tid]; quant[g] deps on group
    # g's proj_a tasks; proj_b deps on [seed, quant[g]]. Each proj_b group-slab
    # dequantizes its INT32 partial by the group's per-row act scale (the per-group
    # scale cannot factor out of the K-sum) and FP32 atomic-adds into a zero-seeded
    # accumulator; proj_b_act (auto region) applies the per-channel weight scale and
    # is the consolidated writer that registers attn_out's return tensormap edge.
    # ========================================================================
    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T], dtype=pl.FP32)   # [G, T] so each group's
                                                                     # per-row scale is a contiguous
                                                                     # row (column reads would be a
                                                                     # strided GM->VecTile load)
    # Per-group INT32 partials: proj_b_mm (pure cube) writes group g's contribution to
    # output channel n at partials[:, g*D + n]; proj_b_act (pure vector) sums the
    # O_GROUPS partials with their per-group act scales. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS * PA_NFRAGS, pl.TASK_ID)
    quant_tids = pl.array.create(O_GROUPS * NUM_QUANT_T_CHUNKS, pl.TASK_ID)
    proj_b_tids = pl.array.create(PB_DCHUNKS * O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        # proj_a[g, nf]: BF16 grouped GEMM -> o_r[:, group g], peel-first-iter form.
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T
            out_col_g = g * O_LORA
            for nf in pl.range(PA_NFRAGS):
                n0 = nf * PROJ_A_MM_N_TILE
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_a_mm", deps=[merge_tid, rope_tid]) as pa_tid:
                    xa0_chunk = o_packed[row_base_o:row_base_o + T, 0:A_K_TILE]
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                        k0 = kb * A_K_TILE
                        xa_k_chunk = o_packed[row_base_o:row_base_o + T, k0:k0 + A_K_TILE]
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + PROJ_A_MM_N_TILE, k0:k0 + A_K_TILE]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                    o_r = pl.assemble(o_r, pl.reshape(acc_a, [T, PROJ_A_MM_N_TILE]), [0, out_col_g + n0])
                proj_a_tids[g * PA_NFRAGS + nf] = pa_tid

        # quant[g, tc]: PER-GROUP amax + symmetric INT8 quant of o_r[:, group g] over a
        # token-chunk (no cross-group barrier -- what unlocks the per-group pipeline).
        # Deps on ONLY group g's proj_a tasks; stores the per-row group dequant scale
        # into act_scale_dq[g, :]. The token-chunk split spreads it over more vector
        # cores. Cast chain (rint INT32 -> round FP16 -> trunc INT8) = per-row form per group.
        for tc in pl.parallel(NUM_QUANT_T_CHUNKS):
            t_base = tc * QUANT_T_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant",
                           deps=[proj_a_tids[g * PA_NFRAGS + j] for j in range(PA_NFRAGS)]) as q_tid:
                    for qt in pl.range(t_base, t_base + QUANT_T_CHUNK, QUANT_TOKEN_TILE):
                        g_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            g_amax = pl.maximum(g_amax, pl.reshape(pl.row_max(pl.maximum(oc, pl.neg(oc))), [1, QUANT_TOKEN_TILE]))
                        g_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), g_amax)
                        act_scale_dq = pl.assemble(act_scale_dq, pl.recip(g_sq_row), [g, qt])
                        g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                        for k1 in pl.range(0, O_LORA, QUANT_TILE):
                            oc = o_r[qt:qt + QUANT_TOKEN_TILE, col_g + k1:col_g + k1 + QUANT_TILE]
                            oq_i32 = pl.cast(pl.row_expand_mul(oc, g_sq_col), target_type=pl.INT32, mode="rint")
                            oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                            o_r_i8 = pl.assemble(o_r_i8, pl.cast(oq_half, target_type=pl.INT8, mode="trunc"), [qt, col_g + k1])
                quant_tids[g * NUM_QUANT_T_CHUNKS + tc] = q_tid

        # proj_b_mm[dc, g]: PURE-CUBE INT8 GEMM of group g's contribution to a
        # PROJ_B_D_CHUNK-wide slab of D, written as INT32 partials to partials[:, g*D+n]
        # (NO vector work here -- the cube never stalls on a dequant; the dequant/sum
        # moves to proj_b_act). The slab's N-frags loop INSIDE the task, so proj_b stays
        # at PB_DCHUNKS*O_GROUPS = 32 tasks. Deps = quant[g] (all token-chunks) ONLY ->
        # group g's proj_b fires as soon as quant[g] lands, overlapping proj_a[g+1] (the
        # back-to-back). Peel-first matmul (matmul_acc from a zero carry trips the TLOAD
        # DN->NZ assertion, pypto#1540).
        for dc in pl.parallel(PB_DCHUNKS):
            d0 = dc * PROJ_B_D_CHUNK
            for g in pl.range(O_GROUPS):
                col_g = g * O_LORA
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_b_mm",
                           deps=[quant_tids[g * NUM_QUANT_T_CHUNKS + tc] for tc in range(NUM_QUANT_T_CHUNKS)]) as pb_tid:
                    for nf in pl.range(PROJ_B_D_CHUNK // PROJ_B_MM_N_TILE):
                        n0 = d0 + nf * PROJ_B_MM_N_TILE
                        acc_b = pl.matmul(o_r_i8[:, col_g:col_g + B_K_TILE],
                                          wo_b[n0:n0 + PROJ_B_MM_N_TILE, col_g:col_g + B_K_TILE],
                                          b_trans=True, out_dtype=pl.INT32)
                        for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                            k0 = col_g + kb * B_K_TILE
                            acc_b = pl.matmul_acc(acc_b, o_r_i8[:, k0:k0 + B_K_TILE],
                                                  wo_b[n0:n0 + PROJ_B_MM_N_TILE, k0:k0 + B_K_TILE], b_trans=True)
                        partials = pl.assemble(partials, acc_b, [0, g * D + n0])
                proj_b_tids[dc * O_GROUPS + g] = pb_tid

    # proj_b_act (PURE-VECTOR consolidated writer, auto region): sum the O_GROUPS INT32
    # partials -- each dequantized by its group's per-row act scale -- then apply the
    # per-channel weight scale -> BF16. Explicit deps on all proj_b_mm tasks bridge
    # manual_scope -> the return's auto-dep (this auto-region write registers the edge).
    with pl.spmd(PB_ACT_NREG * PB_ACT_TBLKS, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(PB_DCHUNKS * O_GROUPS)]) as act_tid:
        act_idx = pl.tile.get_block_idx()
        nreg = act_idx // PB_ACT_TBLKS
        tblk = act_idx - nreg * PB_ACT_TBLKS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TBLK
        wb_scale_chunk = pl.reshape(wo_b_scale[ob_n0:ob_n0 + PROJ_B_ACT_N_TILE], [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TBLK, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for g in pl.range(O_GROUPS):
                p_g = partials[b_tb:b_tb + PROJ_B_ACT_T_TILE, g * D + ob_n0:g * D + ob_n0 + PROJ_B_ACT_N_TILE]
                g_scale = pl.reshape(act_scale_dq[g:g + 1, b_tb:b_tb + PROJ_B_ACT_T_TILE], [PROJ_B_ACT_T_TILE, 1])
                acc = pl.add(acc, pl.row_expand_mul(pl.cast(p_g, target_type=pl.FP32, mode="none"), g_scale))
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            attn_out[b_tb:b_tb + PROJ_B_ACT_T_TILE, ob_n0:ob_n0 + PROJ_B_ACT_N_TILE] = pl.cast(out_t, target_type=pl.BF16, mode="rint")

    return attn_out

@pl.jit
def sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    mtp_kv_overlay: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    sparse_attn(
        q,
        ori_kv,
        ori_block_table,
        mtp_kv_overlay,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    return attn_out


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis."""
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_sparse_attn(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    ori_block_table = tensors["ori_block_table"]
    mtp_kv_overlay = tensors["mtp_kv_overlay"].float()
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)

    # Per-query-token attention. cmp_sparse_indices is the authoritative
    # topk list: -1 invalid; raw < WIN selects ring cache;
    # WIN <= raw < WIN+S selects the current MTP overlay; raw >= WIN+S
    # selects the compressed-cache tail.
    for t in range(T):
        b = t // S
        kv_rows = []
        valid = []

        for raw in cmp_sparse_indices[t].tolist():
            if raw < 0:
                kv_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                valid.append(False)
                continue
            if raw < WIN:
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                kv_rows.append(ori_kv[blk_id, intra, 0])
                valid.append(True)
            elif raw < WIN + S:
                overlay_s = raw - WIN
                kv_rows.append(mtp_kv_overlay[b * S + overlay_s])
                valid.append(True)
            else:
                cmp_slot = raw - (WIN + S)
                blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                kv_rows.append(cmp_kv[blk_id, intra, 0])
                valid.append(True)

        if not any(valid):
            continue

        pad_k = PADDED_TOPK - TOPK
        if pad_k:
            kv_rows.extend(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype) for _ in range(pad_k))
            valid.extend(False for _ in range(pad_k))

        kv_b = torch.stack(kv_rows, dim=0)
        valid_b = torch.tensor(valid, dtype=torch.bool)
        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for tile_start in range(0, PADDED_TOPK, ATTN_K_TILE):
            kv_tile = kv_b[tile_start:tile_start + ATTN_K_TILE]
            valid_tile = valid_b[tile_start:tile_start + ATTN_K_TILE]
            scores = (q_t @ kv_tile.T) * SOFTMAX_SCALE
            scores = scores.masked_fill(~valid_tile.unsqueeze(0), NEG_INF)
            mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - mi).masked_fill(~valid_tile.unsqueeze(0), 0.0)
            li = exp_scores.sum(dim=-1, keepdim=True)
            oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
            block_mi.append(mi)
            block_li.append(li)
            block_oi.append(oi)

        score_max = block_mi[0]
        li = block_li[0]
        oi_num = block_oi[0]
        for mi_cur, li_cur, oi_cur in zip(block_mi[1:], block_li[1:], block_oi[1:]):
            score_max_new = torch.maximum(score_max, mi_cur)
            alpha = torch.exp(score_max - score_max_new)
            beta = torch.exp(mi_cur - score_max_new)
            li = alpha * li + beta * li_cur
            oi_num = alpha * oi_num + beta * oi_cur
            score_max = score_max_new

        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[t] = oi_num / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    seq_per_batch = T // B
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    # PER-GROUP INT8 activation quant (one amax per O_LORA group, not per full row):
    # this localizes the reduction so proj_a[g]->quant[g]->proj_b[g] can pipeline
    # back-to-back. Each group's INT32 partial is dequantized by its OWN per-row
    # activation scale before the groups are summed (the per-group scale cannot
    # factor out of the K-sum), then the per-channel weight scale is applied.
    o_r_g = o_r.reshape(T, O_GROUPS, O_LORA)
    amax_g = o_r_g.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)   # [T, G, 1]
    scale_q_g = INT8_SCALE_MAX / amax_g
    o_r_i8_g = torch.round(o_r_g * scale_q_g).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq_g = 1.0 / scale_q_g                                              # [T, G, 1]
    wo_b_g = wo_b_i8.reshape(D, O_GROUPS, O_LORA)
    out = torch.zeros(T, D, dtype=torch.float32)
    for g in range(O_GROUPS):
        p_g = o_r_i8_g[:, g].to(torch.int32) @ wo_b_g[:, g].to(torch.int32).T   # [T, D]
        out = out + p_g.float() * scale_dq_g[:, g]                             # per-row group scale
    out = out * wo_b_scale.unsqueeze(0)                                        # per-channel weight scale

    tensors["attn_out"][:] = out.to(torch.bfloat16)

def build_tensor_specs(
    compress_ratio: int = DEFAULT_COMPRESS_RATIO,
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    overlay_replacement_fixture: bool = False,
):
    """Build deterministic demo tensors for the merged standalone harness."""
    import torch
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_token_rope_tables

    cmp_valid = get_standalone_cmp_valid(compress_ratio)
    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, compress_ratio, dtype=torch.bfloat16)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(
        shared_freqs_cos,
        shared_freqs_sin,
        torch.arange(T, dtype=torch.int32),
    )

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = torch.rand(T, H, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = torch.rand(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            kv[0, WIN - 1, 0].fill_(8.0)
        return kv

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_mtp_kv_overlay():
        """Initialize the current decode-chunk overlay KV rows."""
        overlay = torch.rand(T, HEAD_DIM) - 0.5
        if overlay_replacement_fixture:
            overlay[:, :] = 0.0
            overlay[:, 0] = 4.0
        return overlay

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_ori_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_cmp_sparse_indices():
        """Build the sparse index list with a full window prefix and padded compressed tail.

        The compressed tail width follows the active specialization (TOPK - WIN):
        the pruned build narrows it to `cmp_valid` columns, the full-blocks
        baseline keeps the IDX_TOPK-wide padded tail.
        """
        win_part = torch.arange(WIN, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        cmp_width = TOPK - WIN
        cmp_part = torch.full((T, cmp_width), -1, dtype=torch.int32)
        cmp_part[:, :cmp_valid] = (torch.arange(cmp_valid, dtype=torch.int32) + WIN + S).unsqueeze(0).expand(T, -1)
        indices = torch.cat([win_part, cmp_part], dim=-1).contiguous()
        if short_window_fixture:
            indices[:, :] = -1
            indices[:, :17] = torch.arange(17, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        if mixed_topk_fixture:
            indices[:, :] = -1
            indices[:, :17] = torch.arange(17, dtype=torch.int32).unsqueeze(0).expand(T, -1)
            mixed_cmp_valid = min(cmp_valid, IDX_TOPK)
            if mixed_cmp_valid:
                indices[:, WIN:WIN + mixed_cmp_valid] = (
                    torch.arange(mixed_cmp_valid, dtype=torch.int32) + WIN + S
                ).unsqueeze(0).expand(T, -1)
        if overlay_replacement_fixture:
            indices[:, :] = -1
            indices[:, :17] = torch.arange(17, dtype=torch.int32).unsqueeze(0).expand(T, -1)
            indices[:, 16] = WIN
        if causal_regression_fixture:
            indices[0, WIN - 1] = WIN - 1
        return indices

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        return shared_rope_cos.clone()

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        return shared_rope_sin.clone()

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights."""
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN ** 0.5)

    wo_b_bf16 = ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        """Initialize the second-stage output-projection weights in per-channel INT8 form."""
        return wo_b_i8

    def init_wo_b_scale():
        """Initialize the dequant scales paired with the INT8 second-stage weights."""
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("mtp_kv_overlay", [T, HEAD_DIM], torch.bfloat16, init_value=init_mtp_kv_overlay),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    # --compress-ratio only selects which compressed-tail data pattern to validate;
    # the pruned widths are covered by the swa/hca variant tests.
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression; use with --compress-ratio 0.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--mixed-topk-fixture", action="store_true", default=False,
                        help="Use -1-padded window slots with valid compressed raw indices.")
    parser.add_argument("--overlay-replacement-fixture", action="store_true", default=False,
                        help="Place a compressed raw index inside the window prefix order.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    compress_ratio = args.compress_ratio
    print(f"compress_ratio={compress_ratio} "
          f"-> TOPK={TOPK} SPARSE_BLOCKS={SPARSE_BLOCKS} PADDED_TOPK={PADDED_TOPK}", flush=True)

    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            compress_ratio,
            args.causal_regression_fixture,
            args.short_window_fixture,
            args.mixed_topk_fixture,
            args.overlay_replacement_fixture,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
