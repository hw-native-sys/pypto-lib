# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Q/KV LoRA + RoPE (dynamic shape): projects token-major
attention-normalized inputs for both decode and prefill attention paths."""


import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings

# tiling
Q_PROJ_TILE = 128       # qproj K-tile (Q_LORA reduction); 8 slices -> deep stage=2 pipeline, double-buffered Mat fits
QPROJ_MM_N_TILE = 1024  # full L2 lines per wq_b row (no over-fetch)
# N-tiles folded into one qproj_matmul SPMD block; the grid is
# (H*HEAD_DIM // QPROJ_MM_N_TILE) // QPROJ_TILES_PER_BLOCK. PRO's 64 N-tiles over a5's
# 28 AIC cores make the block count the dominant term: 4 gives 16 blocks in a single
# wave, while 2 gives 32 -- one full wave plus 4 stragglers that run with 24 cores idle
# -- and 1 gives 64, whose per-shard launch stagger exceeds the wave it saves. The cube
# tile itself is unchanged, so L0A/L0B/L0C occupancy is identical at every setting.
QPROJ_TILES_PER_BLOCK = 4
Q_LORA_TILE = 256       # qr rms-norm / quant N granularity (decoupled from qr_proj matmul)
KV_TILE = 64            # kv rms-norm / rope / NOPE N granularity (decoupled from kv_proj matmul)
QUANT_TILE = 256
T_TILE = 8
MATMUL_T_TILE = 16
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)

# Per-projection matmul tiles. Decoupled so each projection's M/N/K can be tuned
# independently of one another AND of the downstream rms/rope granularity above
# (e.g. the matmul N-tile is no longer chained to KV_TILE / Q_LORA_TILE, which the
# NOPE_DIM=448 constraint caps at <=64).
QR_M_TILE = MATMUL_T_TILE  # qr_proj token (M) tile; cube rows must be a 16-row boxed tile
QR_N_TILE = 128         # qr_proj Q_LORA (N) per matmul
QR_K_TILE = 256         # qr_proj D (K) reduction tile    | divides QR_K_SLICE
QR_OK = 2               # qr_proj split-K factor          | D//QR_OK cores share each N-group
QR_K_SLICE = D // QR_OK # qr_proj K per split (=2048)     | QR_K_SLICE//QR_K_TILE inner chunks
KV_M_TILE = MATMUL_T_TILE  # kv_proj token (M) tile; decode pads from 8 real rows to 16
KV_N_TILE = 128         # kv_proj HEAD_DIM (N) per matmul
# 128 (not 256) so the inner pipeline trip count stays EVEN -- see the
# _even_pipeline_trip assert below. PRO's D = 7168 gives KV_K_SLICE = 1792,
# which at a 256 tile is 7 chunks; 128 gives 14.
KV_K_TILE = 128         # kv_proj D (K) reduction tile    | divides KV_K_SLICE
KV_OK = 4               # kv_proj split-K factor          | D//KV_OK cores share each N-group
KV_K_SLICE = D // KV_OK # kv_proj K per split (PRO: 1792) | KV_K_SLICE//KV_K_TILE inner chunks
QPROJ_M_TILE = MATMUL_T_TILE  # qproj token (M) tile; decode pads from 8 real rows to 16
KV_RMS_T_TILE = 8       # kv rms-norm + rope fused token (T) tile
Q_ROPE_T_TILE = 8
Q_ROPE_H_TILE = 4       # heads per fused qproj dequant/rms/rope task; cos/sin build amortizes over them
MX_BLOCK_K = 32
MX_K_TILE = 64
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
MX_N_TILE = 256
X_MX_K_TILES = D // MX_K_TILE
QR_MX_K_TILES = Q_LORA // MX_K_TILE
WQA_SCALE_ROWS = (Q_LORA // MX_N_TILE) * X_MX_K_TILES * MX_K_SCALE_TILE
WQB_SCALE_ROWS = ((H * HEAD_DIM) // MX_N_TILE) * QR_MX_K_TILES * MX_K_SCALE_TILE
WKV_SCALE_ROWS = (HEAD_DIM // MX_N_TILE) * X_MX_K_TILES * MX_K_SCALE_TILE
assert H % Q_ROPE_H_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
assert DECODE_BATCH * DECODE_SEQ <= MATMUL_T_TILE
for _m_tile in (QR_M_TILE, KV_M_TILE, QPROJ_M_TILE):
    assert (PREFILL_BATCH * PREFILL_SEQ) % _m_tile == 0
assert Q_LORA % QR_N_TILE == 0 and D % QR_OK == 0 and QR_K_SLICE % QR_K_TILE == 0
assert HEAD_DIM % KV_N_TILE == 0 and D % KV_OK == 0 and KV_K_SLICE % KV_K_TILE == 0


def _even_pipeline_trip(name: str, trip: int) -> None:
    """Split-K matmul loops run as ``pl.pipeline(..., stage=2)`` over a loop-carried
    L0C accumulator. With an ODD trip count the final partial lands in the alternate
    pipeline buffer, so codegen has to reconcile the two with an acc->acc ``pto.tmov``
    -- an address-space pair A5 does not implement, and ptoas rejects the kernel with
    ``'pto.tmov' op expects a supported tmov address-space pair for this target``.
    An even trip count leaves the result in the canonical buffer and emits no move.

    This bit us switching FLASH (D=4096) -> PRO (D=7168): kv_proj went 4 -> 7 chunks.
    """
    assert trip % 2 == 0, (
        f"{name} pipeline trip count must be even, got {trip}; an odd count makes "
        "codegen emit an unsupported acc->acc pto.tmov. Retune the K tile or split-K factor."
    )


_even_pipeline_trip("qr_proj", QR_K_SLICE // QR_K_TILE)
_even_pipeline_trip("kv_proj", KV_K_SLICE // KV_K_TILE)
assert (H * HEAD_DIM) % QPROJ_MM_N_TILE == 0 and ((H * HEAD_DIM) // QPROJ_MM_N_TILE) % 4 == 0
assert ((H * HEAD_DIM) // QPROJ_MM_N_TILE) % QPROJ_TILES_PER_BLOCK == 0
assert Q_LORA % Q_PROJ_TILE == 0 and QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C Acc cap
assert (DECODE_BATCH * DECODE_SEQ) % KV_RMS_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % KV_RMS_T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % Q_ROPE_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % Q_ROPE_T_TILE == 0


@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(position_ids, 0)
    for rope_t0 in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="qkv_rope_rows"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < num_tokens:
                rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]

@pl.jit.inline
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[WQA_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WQB_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[WKV_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    t_matmul = pl.max(t_dim, MATMUL_T_TILE)

    # RoPE indices and interleaved cos/signed-sin rows are head-invariant.
    # Prepare them once per token tile so the 16 Q head-group tasks do not each
    # rebuild the same arange/cast/gather chain on their critical AIV path.
    q_rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    for qrp_idx in pl.spmd(t_dim // Q_ROPE_T_TILE, name_hint="q_rope_prepare", allow_early_resolve=True):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        # A5 gather indices stay row-sized because wider TCI tiles can overrun UB.
        qrp_ones = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_dup_idx = pl.cast(qrp_dup_f, target_type=pl.INT32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        for qrp_dt in pl.range(Q_ROPE_T_TILE):
            qrp_t = qrp_t0 + qrp_dt
            qrp_cos = pl.cast(rope_cos_view[qrp_t : qrp_t + 1, :], target_type=pl.FP32)
            qrp_sin = pl.cast(rope_sin_view[qrp_t : qrp_t + 1, :], target_type=pl.FP32)
            qrp_cos_il = pl.gather(qrp_cos, dim=-1, index=qrp_dup_idx)
            qrp_sin_il = pl.gather(qrp_sin, dim=-1, index=qrp_dup_idx)
            qrp_sin_signed = pl.mul(qrp_sin_il, qrp_sign)
            q_rope_cos_il[qrp_t : qrp_t + 1, :] = qrp_cos_il
            q_rope_sin_signed[qrp_t : qrp_t + 1, :] = qrp_sin_signed
            q_rope_swap_idx[qrp_t : qrp_t + 1, :] = qrp_swap_idx

    x_mx = pl.create_tensor([T_MAX, D], dtype=pl.FP8E4M3FN)
    x_scale_store = pl.create_tensor(
        [(T_MAX // MATMUL_T_TILE) * X_MX_K_TILES, MATMUL_T_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0
    )
    for quant_idx in pl.parallel((t_matmul // MATMUL_T_TILE) * X_MX_K_TILES):
        mt = quant_idx // X_MX_K_TILES
        kb = quant_idx % X_MX_K_TILES
        t0 = mt * MATMUL_T_TILE
        k0 = kb * MX_K_TILE
        valid_rows = pl.min(MATMUL_T_TILE, t_dim - t0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qkv_x_mx_quant"):
            x_src = pl.load(
                x_view,
                [t0, k0],
                [MATMUL_T_TILE, MX_K_TILE],
                valid_shape=[valid_rows, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            x_src = pl.fillpad(x_src, pad_value=pl.PadValue.zero)
            x_q, x_scale = pl.quant_mx(x_src, layout=pl.MX_A_ZZ)
            pl.store(x_q, [t0, k0], x_mx)
            x_scale_flat = pl.reshape(x_scale, [1, MATMUL_T_TILE * MX_K_SCALE_TILE])
            pl.store(x_scale_flat, [quant_idx, 0], x_scale_store)

    qr_fp32 = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    qr_partial = pl.create_tensor([T_MAX * X_MX_K_TILES, Q_LORA], dtype=pl.FP32)
    qr_norm_fp32 = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    for task_idx in pl.parallel((t_matmul // MATMUL_T_TILE) * X_MX_K_TILES * (Q_LORA // MX_N_TILE)):
        mt = task_idx // (X_MX_K_TILES * (Q_LORA // MX_N_TILE))
        local_idx = task_idx % (X_MX_K_TILES * (Q_LORA // MX_N_TILE))
        kb = local_idx // (Q_LORA // MX_N_TILE)
        nb = local_idx % (Q_LORA // MX_N_TILE)
        t0 = mt * MATMUL_T_TILE
        k0 = kb * MX_K_TILE
        n0 = nb * MX_N_TILE
        x_scale_idx = mt * X_MX_K_TILES + kb
        x_scale_slice = x_scale_store[x_scale_idx : x_scale_idx + 1, :]
        x_scale_mx = pl.tensor.view(x_scale_slice, [MATMUL_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ)
        w_scale_offset = (nb * X_MX_K_TILES + kb) * MX_K_SCALE_TILE
        w_scale_slice = wq_a_scale[w_scale_offset : w_scale_offset + MX_K_SCALE_TILE, :]
        w_scale_mx = pl.tensor.view(w_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_mx"):
            x_k = pl.move(
                pl.load(x_mx, [t0, k0], [MATMUL_T_TILE, MX_K_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            x_scale_k = pl.move(
                pl.load(x_scale_mx, [0, 0], [MATMUL_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            w_k = pl.move(
                pl.load(wq_a, [k0, n0], [MX_K_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Right,
            )
            w_scale_k = pl.move(
                pl.load(w_scale_mx, [0, 0], [MX_K_SCALE_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            qr_partial_acc = pl.matmul_mx(x_k, x_scale_k, w_k, w_scale_k)
            partial_row = (mt * X_MX_K_TILES + kb) * MATMUL_T_TILE
            pl.store(qr_partial_acc, [partial_row, n0], qr_partial)

    for task_idx in pl.parallel((t_matmul // MATMUL_T_TILE) * (Q_LORA // MX_N_TILE)):
        mt = task_idx // (Q_LORA // MX_N_TILE)
        nb = task_idx % (Q_LORA // MX_N_TILE)
        t0 = mt * MATMUL_T_TILE
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_reduce"):
            qr_sum = pl.tile.full([MATMUL_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(X_MX_K_TILES, stage=2):
                partial_row = (mt * X_MX_K_TILES + kb) * MATMUL_T_TILE
                qr_partial_vec = pl.load(
                    qr_partial, [partial_row, n0], [MATMUL_T_TILE, MX_N_TILE], target_memory=pl.Mem.Vec
                )
                qr_sum = pl.add(qr_sum, qr_partial_vec)
            pl.store(qr_sum, [t0, n0], qr_fp32)

    # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
        tg = tg_idx * T_TILE
        qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        qr_amax_g = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for qr_rms_qb in pl.pipeline(Q_LORA // Q_LORA_TILE, stage=2):
            qr_rms_col0 = qr_rms_qb * Q_LORA_TILE
            qr_rms_chunk = qr_fp32[tg : tg + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
            qr_sq_sum = pl.add(qr_sq_sum, pl.reshape(pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk)), [1, T_TILE]))
            gamma_rms_cast = pl.cast(gamma_cq[qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE], target_type=pl.FP32)
            gamma_rms_chunk = pl.reshape(gamma_rms_cast, [1, Q_LORA_TILE])
            qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
            qr_g_abs = pl.abs(qr_g)
            qr_amax_g = pl.maximum(qr_amax_g, pl.reshape(pl.row_max(qr_g_abs), [1, T_TILE]))
        qr_inv_rms = pl.rsqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS), high_precision=True)
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
        qr_amax_floor = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qr_amax_normed = pl.mul(qr_inv_rms, qr_amax_g)
        qr_tile_amax = pl.maximum(qr_amax_floor, qr_amax_normed)

        qr_scale_quant_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
        qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
        qr_scale_view[tg : tg + T_TILE, :] = qr_tile_scale_dq

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = qr_fp32[tg : tg + T_TILE, qa : qa + QUANT_TILE]
            gamma_q_cast = pl.cast(gamma_cq[qa : qa + QUANT_TILE], target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
            qr_norm_fp32[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_normed
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
            qr_view[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8

    qr_mx = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP8E4M3FN)
    qr_mx_scale_store = pl.create_tensor(
        [(T_MAX // MATMUL_T_TILE) * QR_MX_K_TILES, MATMUL_T_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0
    )
    for quant_idx in pl.parallel((t_matmul // MATMUL_T_TILE) * QR_MX_K_TILES):
        mt = quant_idx // QR_MX_K_TILES
        kb = quant_idx % QR_MX_K_TILES
        t0 = mt * MATMUL_T_TILE
        k0 = kb * MX_K_TILE
        valid_rows = pl.min(MATMUL_T_TILE, t_dim - t0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qproj_input_mx_quant"):
            qr_src = pl.load(
                qr_norm_fp32,
                [t0, k0],
                [MATMUL_T_TILE, MX_K_TILE],
                valid_shape=[valid_rows, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            qr_src = pl.fillpad(qr_src, pad_value=pl.PadValue.zero)
            qr_q, qr_mx_scale = pl.quant_mx(qr_src, layout=pl.MX_A_ZZ)
            pl.store(qr_q, [t0, k0], qr_mx)
            qr_scale_flat = pl.reshape(qr_mx_scale, [1, MATMUL_T_TILE * MX_K_SCALE_TILE])
            pl.store(qr_scale_flat, [quant_idx, 0], qr_mx_scale_store)

    q_proj_partial = pl.create_tensor([T_MAX * QR_MX_K_TILES, H * HEAD_DIM], dtype=pl.FP32)
    for task_idx in pl.parallel(
        (t_matmul // MATMUL_T_TILE) * QR_MX_K_TILES * ((H * HEAD_DIM) // MX_N_TILE)
    ):
        mt = task_idx // (QR_MX_K_TILES * ((H * HEAD_DIM) // MX_N_TILE))
        local_idx = task_idx % (QR_MX_K_TILES * ((H * HEAD_DIM) // MX_N_TILE))
        kb = local_idx // ((H * HEAD_DIM) // MX_N_TILE)
        nb = local_idx % ((H * HEAD_DIM) // MX_N_TILE)
        t0 = mt * MATMUL_T_TILE
        k0 = kb * MX_K_TILE
        n0 = nb * MX_N_TILE
        scale_idx = mt * QR_MX_K_TILES + kb
        qr_scale_slice = qr_mx_scale_store[scale_idx : scale_idx + 1, :]
        qr_scale_mx = pl.tensor.view(qr_scale_slice, [MATMUL_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ)
        w_scale_offset = (nb * QR_MX_K_TILES + kb) * MX_K_SCALE_TILE
        w_scale_slice = wq_b_scale[w_scale_offset : w_scale_offset + MX_K_SCALE_TILE, :]
        w_scale_mx = pl.tensor.view(w_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qproj_mx"):
            qr_k = pl.move(
                pl.load(qr_mx, [t0, k0], [MATMUL_T_TILE, MX_K_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            qr_scale_k = pl.move(
                pl.load(qr_scale_mx, [0, 0], [MATMUL_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            w_k = pl.move(
                pl.load(wq_b, [k0, n0], [MX_K_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Right,
            )
            w_scale_k = pl.move(
                pl.load(w_scale_mx, [0, 0], [MX_K_SCALE_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            q_partial_acc = pl.matmul_mx(qr_k, qr_scale_k, w_k, w_scale_k)
            partial_row = (mt * QR_MX_K_TILES + kb) * MATMUL_T_TILE
            pl.store(q_partial_acc, [partial_row, n0], q_proj_partial)

    q_proj_fp32 = pl.create_tensor([T_MAX, H * HEAD_DIM], dtype=pl.FP32)
    for task_idx in pl.parallel((t_matmul // MATMUL_T_TILE) * ((H * HEAD_DIM) // MX_N_TILE)):
        mt = task_idx // ((H * HEAD_DIM) // MX_N_TILE)
        nb = task_idx % ((H * HEAD_DIM) // MX_N_TILE)
        t0 = mt * MATMUL_T_TILE
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qproj_reduce"):
            q_sum = pl.tile.full([MATMUL_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(QR_MX_K_TILES, stage=2):
                partial_row = (mt * QR_MX_K_TILES + kb) * MATMUL_T_TILE
                q_partial_vec = pl.load(
                    q_proj_partial, [partial_row, n0], [MATMUL_T_TILE, MX_N_TILE], target_memory=pl.Mem.Vec
                )
                q_sum = pl.add(q_sum, q_partial_vec)
            pl.store(q_sum, [t0, n0], q_proj_fp32)

    # Fuse qproj dequant, per-head RMSNorm, NOPE writeback, and interleaved RoPE.
    # A full [token, head] tile fits in Vec UB, so dequantize each head once and
    # retain it across the RMS reduction instead of rereading/recomputing NOPE.
    # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for hg_idx in pl.spmd(H // Q_ROPE_H_TILE, name_hint="qproj_dequant_rms_nope_rope"):
        hg = hg_idx * Q_ROPE_H_TILE
        for tg_idx in pl.range(t_dim // Q_ROPE_T_TILE):
            tg = tg_idx * Q_ROPE_T_TILE
            q_cos_il = q_rope_cos_il[tg : tg + Q_ROPE_T_TILE, :]
            q_sin_signed = q_rope_sin_signed[tg : tg + Q_ROPE_T_TILE, :]
            q_swap_idx = q_rope_swap_idx[tg : tg + Q_ROPE_T_TILE, :]
            # Pipeline adjacent heads so the next head's GM reads overlap the
            # current head's vector RMS/rotation work, as in Qwen's decode loop.
            for h_inner in pl.pipeline(Q_ROPE_H_TILE, stage=2):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_head_dq = q_proj_fp32[tg : tg + Q_ROPE_T_TILE, h0 : h0 + HEAD_DIM]
                q_head_sq = pl.mul(q_head_dq, q_head_dq)
                q_head_sq_row = pl.row_sum(q_head_sq)
                q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                q_head_var = pl.add(q_head_sq_mean, EPS)
                q_head_inv_rms = pl.rsqrt(q_head_var, high_precision=True)
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])

                q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                q_flat[tg : tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_bf16

                # RoPE writeback on columns [h0+NOPE_DIM:h0+HEAD_DIM). Fold inv_rms in
                # BEFORE the rotation (normalize-then-rotate), matching the kv path. This
                # is mathematically equivalent to rotating then normalizing — inv_rms is a
                # per-row scalar so inv_rms*(a*cos+b*sin) == (a*inv_rms)*cos+(b*inv_rms)*sin
                # — but keeps the rotation intermediates small. Rotating the raw (large)
                # dequantized values first produced large intermediates that lost precision
                # on Ascend950 (A5), corrupting the query RoPE region; normalizing first
                # avoids that without changing the result on A2A3.
                q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                q_rope_col0 = h0 + NOPE_DIM
                q_rope_col1 = q_rope_col0 + ROPE_DIM
                for q_rope_row in pl.range(Q_ROPE_T_TILE):
                    q_rope_row_chunk = q_rope_chunk[q_rope_row : q_rope_row + 1, :]
                    q_rope_swap_row = q_swap_idx[q_rope_row : q_rope_row + 1, :]
                    q_rope_cos_row = q_cos_il[q_rope_row : q_rope_row + 1, :]
                    q_rope_sin_row = q_sin_signed[q_rope_row : q_rope_row + 1, :]
                    q_rope_swapped = pl.gather(q_rope_row_chunk, dim=-1, index=q_rope_swap_row)
                    q_rope_base = pl.mul(q_rope_row_chunk, q_rope_cos_row)
                    q_rope_delta = pl.mul(q_rope_swapped, q_rope_sin_row)
                    q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                    q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                    q_rope_out_row = tg + q_rope_row
                    q_flat[q_rope_out_row : q_rope_out_row + 1, q_rope_col0:q_rope_col1] = q_rope_bf16

    kv_fp32 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    kv_partial = pl.create_tensor([T_MAX * X_MX_K_TILES, HEAD_DIM], dtype=pl.FP32)
    for task_idx in pl.parallel((t_matmul // MATMUL_T_TILE) * X_MX_K_TILES * (HEAD_DIM // MX_N_TILE)):
        mt = task_idx // (X_MX_K_TILES * (HEAD_DIM // MX_N_TILE))
        local_idx = task_idx % (X_MX_K_TILES * (HEAD_DIM // MX_N_TILE))
        kb = local_idx // (HEAD_DIM // MX_N_TILE)
        nb = local_idx % (HEAD_DIM // MX_N_TILE)
        t0 = mt * MATMUL_T_TILE
        k0 = kb * MX_K_TILE
        n0 = nb * MX_N_TILE
        x_scale_idx = mt * X_MX_K_TILES + kb
        x_scale_slice = x_scale_store[x_scale_idx : x_scale_idx + 1, :]
        x_scale_mx = pl.tensor.view(x_scale_slice, [MATMUL_T_TILE, MX_K_SCALE_TILE], layout=pl.MX_A_ZZ)
        w_scale_offset = (nb * X_MX_K_TILES + kb) * MX_K_SCALE_TILE
        w_scale_slice = wkv_scale[w_scale_offset : w_scale_offset + MX_K_SCALE_TILE, :]
        w_scale_mx = pl.tensor.view(w_scale_slice, [MX_K_SCALE_TILE, MX_N_TILE], layout=pl.MX_B_NN)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_mx"):
            x_k = pl.move(
                pl.load(x_mx, [t0, k0], [MATMUL_T_TILE, MX_K_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Left,
            )
            x_scale_k = pl.move(
                pl.load(x_scale_mx, [0, 0], [MATMUL_T_TILE, MX_K_SCALE_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.LeftScale,
            )
            w_k = pl.move(
                pl.load(wkv, [k0, n0], [MX_K_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.Right,
            )
            w_scale_k = pl.move(
                pl.load(w_scale_mx, [0, 0], [MX_K_SCALE_TILE, MX_N_TILE], target_memory=pl.Mem.Mat),
                target_memory=pl.Mem.RightScale,
            )
            kv_partial_acc = pl.matmul_mx(x_k, x_scale_k, w_k, w_scale_k)
            partial_row = (mt * X_MX_K_TILES + kb) * MATMUL_T_TILE
            pl.store(kv_partial_acc, [partial_row, n0], kv_partial)

    for task_idx in pl.parallel((t_matmul // MATMUL_T_TILE) * (HEAD_DIM // MX_N_TILE)):
        mt = task_idx // (HEAD_DIM // MX_N_TILE)
        nb = task_idx % (HEAD_DIM // MX_N_TILE)
        t0 = mt * MATMUL_T_TILE
        n0 = nb * MX_N_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_reduce"):
            kv_sum = pl.tile.full([MATMUL_T_TILE, MX_N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(X_MX_K_TILES, stage=2):
                partial_row = (mt * X_MX_K_TILES + kb) * MATMUL_T_TILE
                kv_partial_vec = pl.load(
                    kv_partial, [partial_row, n0], [MATMUL_T_TILE, MX_N_TILE], target_memory=pl.Mem.Vec
                )
                kv_sum = pl.add(kv_sum, kv_partial_vec)
            pl.store(kv_sum, [t0, n0], kv_fp32)

    # Fused KV RMSNorm + interleaved (CANN A3) RoPE. One spmd task per [KV_RMS_T_TILE, HEAD_DIM]
    # row block computes the per-row inv_rms once (pass 1) and consumes it locally for
    # BOTH the NOPE writeback and the rope rotation -- so inv_rms no longer round-trips
    # through GM (the old kv_inv_rms_tensor) and the two passes collapse into a single
    # dispatch. NOPE columns [0:NOPE_DIM) and rope columns [NOPE_DIM:HEAD_DIM) are
    # disjoint, so each task writes a clean, conflict-free row block of kv. Vec UB stays
    # well under the 192 KB cap (chunks are at most [KV_RMS_T_TILE, KV_TILE] fp32).
    for tg_idx in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="kv_rms_norm_rope"):
        tg = tg_idx * KV_RMS_T_TILE
        # Pass 1: per-row sum of squares over the full HEAD_DIM -> inv_rms.
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HEAD_DIM // KV_TILE, stage=2):
            kv_sq_col0 = kb * KV_TILE
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
            kv_sq_sum = pl.add(kv_sq_sum, pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, KV_RMS_T_TILE]))
        kv_inv_rms = pl.rsqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS), high_precision=True)
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

        # NOPE writeback: rms-normalize columns [0:NOPE_DIM) with per-column gamma.
        for nb in pl.pipeline(NOPE_DIM // KV_TILE, stage=2):
            n0 = nb * KV_TILE
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
            gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv_view[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")

        # RoPE writeback on columns [NOPE_DIM:HEAD_DIM), interleaved (CANN A3) swap-gather
        # (same form as qproj_dequant_rms_nope_rope), built in-kernel. inv_rms (per-row, the same
        # factor used for NOPE above) and gamma (per-column, full ROPE_DIM) are folded into
        # kv_rope_norm_chunk BEFORE the swap so the swapped lane n[j^1] carries gamma[j^1]
        # (gamma does NOT commute with the rotation; inv_rms does).
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]
        gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
        kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
        kv_ones = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=1.0)
        kv_col = pl.col_expand_mul(kv_ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        kv_dup_idx = pl.cast(kv_dup_f, target_type=pl.INT32)                                       # j>>1
        kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))                                            # j%2
        kv_swap_idx = pl.cast(pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0)), target_type=pl.INT32)  # j^1
        kv_sign = pl.sub(pl.mul(kv_lane, 2.0), 1.0)                                                # [-1,+1,...]
        for kv_rope_row in pl.range(KV_RMS_T_TILE):
            kv_rope_t = tg + kv_rope_row
            kv_cos = pl.cast(rope_cos_view[kv_rope_t : kv_rope_t + 1, :], target_type=pl.FP32)
            kv_sin = pl.cast(rope_sin_view[kv_rope_t : kv_rope_t + 1, :], target_type=pl.FP32)
            kv_cos_il = pl.gather(kv_cos, dim=-1, index=kv_dup_idx)
            kv_sin_il = pl.gather(kv_sin, dim=-1, index=kv_dup_idx)
            kv_rope_norm_row = kv_rope_norm_chunk[kv_rope_row : kv_rope_row + 1, :]
            kv_swapped = pl.gather(kv_rope_norm_row, dim=-1, index=kv_swap_idx)
            kv_rope_base = pl.mul(kv_rope_norm_row, kv_cos_il)
            kv_rope_signed = pl.mul(kv_swapped, kv_sign)
            kv_rope_delta = pl.mul(kv_rope_signed, kv_sin_il)
            kv_rope_rot = pl.add(kv_rope_base, kv_rope_delta)
            kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode="rint")
            kv_view[kv_rope_t : kv_rope_t + 1, NOPE_DIM : NOPE_DIM + ROPE_DIM] = kv_rope_i16

    return q


@pl.jit
def qkv_proj_rope_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[WQA_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[WQB_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[WKV_SCALE_ROWS, MX_N_TILE], pl.FP8E8M0],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    q.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)

    # Standalone: no rms_norm producer, so the barrier fences nothing (ready on submit).
    late_dep = pl.system.task_dummy(deps=[])
    qkv_proj_rope(
        x,
        wq_a,
        wq_a_scale,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        rope_cos,
        rope_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        late_dep,
    )
    return q


def golden_qkv_proj_rope(tensors):
    """Torch reference: Q/KV LoRA + RoPE for an already attention-normalized input."""
    import torch
    from expert_shared import _dynamic_mxfp8_matmul, _unpack_b_scale_tiled

    x = tensors["x"].float()
    wq_a = tensors["wq_a"]
    wq_a_scale = _unpack_b_scale_tiled(tensors["wq_a_scale"], D, Q_LORA)
    wq_b = tensors["wq_b"]
    wq_b_scale = _unpack_b_scale_tiled(tensors["wq_b_scale"], Q_LORA, H * HEAD_DIM)
    wkv = tensors["wkv"]
    wkv_scale = _unpack_b_scale_tiled(tensors["wkv_scale"], D, HEAD_DIM)
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def int8_quant_per_row(x):
        rows = x.reshape(-1, x.shape[-1]).float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = rows * scale_quant
        out_i32 = torch.round(scaled).to(torch.int32)
        out_half = out_i32.to(torch.float16)
        out_i8 = out_half.to(torch.int8)
        return out_i8.reshape_as(x), (1.0 / scale_quant).reshape(*x.shape[:-1], 1)

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

    def apply_rope(x_rope, cos, sin):
        # x_rope: [T, ..., ROPE_DIM] with interleaved even/odd rotary pairs.
        x_pair = x_rope.unflatten(-1, (-1, 2))
        x_even, x_odd = x_pair[..., 0], x_pair[..., 1]
        cos_v = cos[..., :ROPE_HALF]
        sin_v = sin[..., :ROPE_HALF]
        while cos_v.ndim < x_even.ndim:
            cos_v = cos_v.unsqueeze(-2)
            sin_v = sin_v.unsqueeze(-2)
        y_even = (x_even * cos_v - x_odd * sin_v).to(torch.bfloat16)
        y_odd = (x_even * sin_v + x_odd * cos_v).to(torch.bfloat16)
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)

    t_dim = x.shape[0]
    token_x = x.view(t_dim, D)

    # Q path
    qr_out = rms_norm(_dynamic_mxfp8_matmul(token_x, wq_a, wq_a_scale), gamma_cq)
    qr_i8, qr_scale = int8_quant_per_row(qr_out.float())
    q_full = _dynamic_mxfp8_matmul(qr_out, wq_b, wq_b_scale).view(t_dim, H, HEAD_DIM)
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv                                            # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    # KV path
    kv_full = rms_norm(_dynamic_mxfp8_matmul(token_x, wkv, wkv_scale), gamma_ckv)
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:]  = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale


def _reference_q_from_quantized_qr(
    qr,
    qr_scale,
    wq_b,
    wq_b_scale,
    rope_cos,
    rope_sin,
):
    """Recompute the Q path downstream of the emitted INT8 QR boundary."""
    import torch

    t_dim = qr.shape[0]
    q_i32 = torch.matmul(qr.to(torch.int32), wq_b.to(torch.int32))
    q_full = (
        q_i32.float()
        * qr_scale.float()
        * wq_b_scale.float().view(1, -1)
    ).view(t_dim, H, HEAD_DIM)
    q_full = q_full * torch.rsqrt(
        q_full.square().mean(dim=-1, keepdim=True) + EPS
    )

    q_pair = q_full[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    q_even, q_odd = q_pair[..., 0], q_pair[..., 1]
    cos = rope_cos.float()[..., :ROPE_HALF].unsqueeze(-2)
    sin = rope_sin.float()[..., :ROPE_HALF].unsqueeze(-2)
    y_even = (q_even * cos - q_odd * sin).to(torch.bfloat16)
    y_odd = (q_even * sin + q_odd * cos).to(torch.bfloat16)
    q_rope = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return torch.cat([q_full[..., :NOPE_DIM], q_rope], dim=-1).to(torch.bfloat16)


def quantized_qr_compare(
    *,
    max_code_step=1,
    max_changed_ratio=0.005,
    max_changed_per_row_ratio=0.005,
    max_show=10,
):
    """Bound both the magnitude and population of QR quantization-boundary changes."""
    import torch

    if max_code_step < 0:
        raise ValueError(f"max_code_step must be non-negative, got {max_code_step}")
    if not 0.0 <= max_changed_ratio <= 1.0:
        raise ValueError(
            f"max_changed_ratio must be in [0, 1], got {max_changed_ratio}"
        )
    if not 0.0 <= max_changed_per_row_ratio <= 1.0:
        raise ValueError(
            "max_changed_per_row_ratio must be in [0, 1], got "
            f"{max_changed_per_row_ratio}"
        )

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, inputs, rtol, atol
        actual = actual.cpu()
        expected = expected.cpu()
        if actual.shape != expected.shape:
            return False, (
                f"    QR shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if actual.dtype != torch.int8 or expected.dtype != torch.int8:
            return False, (
                f"    QR comparator requires int8 tensors, got "
                f"{actual.dtype} and {expected.dtype}"
            )
        diff = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
        changed = diff != 0
        changed_count = int(changed.count_nonzero().item())
        changed_limit = round(max_changed_ratio * diff.numel())
        changed_per_row = changed.reshape(-1, changed.shape[-1]).sum(dim=-1)
        row_limit = int(max_changed_per_row_ratio * changed.shape[-1])
        overfull_rows = changed_per_row > row_limit
        overfull_row_count = int(overfull_rows.count_nonzero().item())
        too_large = diff > max_code_step
        too_large_count = int(too_large.count_nonzero().item())
        if (
            changed_count <= changed_limit
            and overfull_row_count == 0
            and too_large_count == 0
        ):
            return True, ""

        changed_indices = changed.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        flat_diff = diff.flatten()
        lines = []
        for index in changed_indices[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={int(flat_actual[index])} "
                f"expected={int(flat_expected[index])} "
                f"code_step={int(flat_diff[index])}"
            )
        return False, (
            f"    QR quantization-boundary mismatch: changed={changed_count}/"
            f"{diff.numel()} (allowed<={max_changed_ratio:.4%}, "
            f"threshold={changed_limit}), code_step>{max_code_step}: "
            f"{too_large_count}, rows>{row_limit} changed codes: "
            f"{overfull_row_count}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"quantized_qr_compare(max_code_step={max_code_step},"
        f"max_changed_ratio={max_changed_ratio},"
        f"max_changed_per_row_ratio={max_changed_per_row_ratio})"
    )
    return compare


def qr_scale_compare(
    *,
    atol=2.5e-5,
    rtol=5e-3,
    max_error_ratio=0.0,
    max_show=10,
):
    """Validate QR dequant scales with ratio and per-row bounds.

    A scale is a whole-row quantization boundary: one bad value affects every
    downstream channel for that token. Keep the aggregate ratio for parity
    with the numeric harness, but also reject any individual scale outside a
    small absolute/relative cap so conditioned Q validation cannot hide it.
    """
    import torch

    if atol < 0 or rtol < 0:
        raise ValueError("QR scale tolerances must be non-negative")
    if not 0.0 <= max_error_ratio <= 1.0:
        raise ValueError(
            f"max_error_ratio must be in [0, 1], got {max_error_ratio}"
        )
    scale_atol = atol
    scale_rtol = rtol

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, inputs, rtol, atol
        actual = actual.cpu().to(torch.float32)
        expected = expected.cpu().to(torch.float32)
        if actual.shape != expected.shape:
            return False, (
                f"    QR scale shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            return False, "    QR scales contain NaN or Inf"
        if (actual <= 0).any().item() or (expected <= 0).any().item():
            return False, "    QR scales must be positive"

        diff = (actual - expected).abs()
        tolerance = scale_atol + scale_rtol * expected.abs()
        bad = diff > tolerance
        bad_count = int(bad.count_nonzero().item())
        threshold = round(max_error_ratio * actual.numel())
        # Even if a caller opts into an aggregate budget, no individual scale
        # may exceed a modest 2x cap.
        hard_bad = diff > (2.0 * tolerance)
        hard_bad_count = int(hard_bad.count_nonzero().item())
        if bad_count <= threshold and hard_bad_count == 0:
            return True, ""

        bad_indices = bad.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        flat_diff = diff.flatten()
        flat_tolerance = tolerance.flatten()
        lines = []
        for index in bad_indices[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={float(flat_actual[index]):.8g} "
                f"expected={float(flat_expected[index]):.8g} "
                f"diff={float(flat_diff[index]):.4g} "
                f"tol={float(flat_tolerance[index]):.4g}"
            )
        return False, (
            f"    QR scale mismatch: bad={bad_count}/{actual.numel()} "
            f"(allowed<={max_error_ratio:.4%}, threshold={threshold}), "
            f"hard_bad={hard_bad_count}, atol={scale_atol}, rtol={scale_rtol}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"qr_scale_compare(atol={scale_atol},rtol={scale_rtol},"
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def q_from_runtime_qr_compare(
    *,
    atol=1e-4,
    rtol=1.0 / 128,
    max_error_ratio=0.005,
):
    """Validate Q after conditioning the reference on the emitted QR codes.

    CPU and A5 split-K reductions can place a few QR values on opposite sides
    of an INT8 rounding boundary. Comparing Q against the CPU QR path amplifies
    one legal code-step across an entire projected token. QR and QR scale are
    validated independently, so the downstream Q reference must start from the
    device-emitted quantized boundary to isolate Q projection/RMS/RoPE accuracy.
    """
    from golden import ratio_allclose

    base_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del expected
        required_outputs = ("qr", "qr_scale")
        required_inputs = ("wq_b", "wq_b_scale", "rope_cos", "rope_sin")
        missing_outputs = [name for name in required_outputs if name not in actual_outputs]
        missing_inputs = [name for name in required_inputs if name not in inputs]
        if missing_outputs or missing_inputs:
            return False, (
                "    conditioned Q comparator is missing "
                f"outputs={missing_outputs}, inputs={missing_inputs}"
            )

        conditioned = _reference_q_from_quantized_qr(
            actual_outputs["qr"].cpu(),
            actual_outputs["qr_scale"].cpu(),
            inputs["wq_b"].cpu(),
            inputs["wq_b_scale"].cpu(),
            inputs["rope_cos"].cpu(),
            inputs["rope_sin"].cpu(),
        )
        if actual.shape != conditioned.shape:
            return False, (
                f"    conditioned Q shape mismatch: actual={tuple(actual.shape)} "
                f"reference={tuple(conditioned.shape)}"
            )
        ok, detail = base_compare(
            actual,
            conditioned,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if ok:
            return True, ""
        return False, "    Q downstream of emitted QR does not match:\n" + detail

    compare.__name__ = (
        f"q_from_runtime_qr_compare(atol={atol},rtol={rtol},"
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def build_tensor_specs(B, S):
    import torch
    from expert_shared import _gen_mxfp8_weight_kn
    from golden import TensorSpec

    T = B * S

    # Inputs match cann test_mla_prolog_quant_pypto gen_mla_prolog_input_data (uniform).
    def init_x():
        return torch.empty([T, D], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_cos():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_sin():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_gamma_cq():
        return torch.empty([Q_LORA], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_gamma_ckv():
        return torch.empty([HEAD_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    wq_a_fp8, wq_a_scale = _gen_mxfp8_weight_kn((D, Q_LORA), dequant_std=0.058, chan_cv=0.25)
    wq_b_fp8, wq_b_scale = _gen_mxfp8_weight_kn((Q_LORA, H * HEAD_DIM), dequant_std=0.058, chan_cv=0.25)
    wkv_fp8, wkv_scale = _gen_mxfp8_weight_kn((D, HEAD_DIM), dequant_std=0.058, chan_cv=0.25)

    return [
        TensorSpec("x",         [T, D],                 torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a", [D, Q_LORA], torch.float8_e4m3fn, init_value=lambda: wq_a_fp8),
        TensorSpec("wq_a_scale", [WQA_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wq_a_scale),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wq_b_fp8),
        TensorSpec("wq_b_scale", [WQB_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: wkv_fp8),
        TensorSpec("wkv_scale", [WKV_SCALE_ROWS, MX_N_TILE], torch.float8_e8m0fnu, init_value=lambda: wkv_scale),
        TensorSpec("rope_cos",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_cos),
        TensorSpec("rope_sin",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_sin),
        TensorSpec("gamma_cq",  [Q_LORA],               torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM],             torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("q",         [T, H, HEAD_DIM],       torch.bfloat16, is_output=True),
        TensorSpec("kv",        [T, HEAD_DIM],          torch.bfloat16, is_output=True),
        TensorSpec("qr",        [T, Q_LORA],            torch.int8,     is_output=True),
        TensorSpec("qr_scale",  [T, 1],                 torch.float32,  is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-l2-swimlane", type=int, choices=[0, 1, 2, 4], default=0,
                        help="L2 swimlane level: 0=off, 1=per-kernel AICore timing "
                             "(prints the per-function Task Statistics table), 2=+AICPU timing.")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- qkv_proj_rope {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=qkv_proj_rope_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_qkv_proj_rope,
            # W8A8C16 q_proj adds INT8 quant/dequant round-off before per-head RMSNorm.
            rtol=5e-3,
            atol=5e-3,
            # Precision reference: pypto mla_prolog —
            # cann-recipes-infer/ops/pypto_python/example/test_mla_prolog_pypto.py
            compare_fn={
                "q":        q_from_runtime_qr_compare(atol=1e-4, rtol=1.0 / 128),
                "kv":       ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "qr":       quantized_qr_compare(max_code_step=1, max_changed_ratio=0.005),
                "qr_scale": qr_scale_compare(
                    atol=2.5e-5,
                    rtol=5e-3,
                    max_error_ratio=0.0,
                ),
            },
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
