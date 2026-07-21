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

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS
T_DYN = pl.dynamic("QKV_PROJ_ROPE_T_DYN")


# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_DIM_SCALE = float(ROPE_DIM)
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings

# tiling
Q_PROJ_TILE = 128       # qproj K-tile (Q_LORA reduction)
QPROJ_MM_N_TILE = 512   # 64 KiB INT8 Right tile on A2/A3
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
KV_K_TILE = 256         # kv_proj D (K) reduction tile    | divides KV_K_SLICE
KV_OK = 4               # kv_proj split-K factor          | D//KV_OK cores share each N-group
KV_K_SLICE = D // KV_OK # kv_proj K per split (=1024)     | KV_K_SLICE//KV_K_TILE inner chunks
QPROJ_M_TILE = MATMUL_T_TILE  # qproj token (M) tile; decode pads from 8 real rows to 16
KV_RMS_T_TILE = 8       # kv rms-norm + rope fused token (T) tile
Q_ROPE_T_TILE = 8
Q_ROPE_H_TILE = 4       # heads per fused qproj dequant/rms/rope task; cos/sin build amortizes over them
assert H % Q_ROPE_H_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
assert DECODE_BATCH * DECODE_SEQ <= MATMUL_T_TILE
for _m_tile in (QR_M_TILE, KV_M_TILE, QPROJ_M_TILE):
    assert (PREFILL_BATCH * PREFILL_SEQ) % _m_tile == 0
assert Q_LORA % QR_N_TILE == 0 and D % QR_OK == 0 and QR_K_SLICE % QR_K_TILE == 0
assert HEAD_DIM % KV_N_TILE == 0 and D % KV_OK == 0 and KV_K_SLICE % KV_K_TILE == 0
assert (H * HEAD_DIM) % QPROJ_MM_N_TILE == 0 and ((H * HEAD_DIM) // QPROJ_MM_N_TILE) % 4 == 0
assert Q_LORA % Q_PROJ_TILE == 0 and QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C Acc cap
assert (DECODE_BATCH * DECODE_SEQ) % KV_RMS_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % KV_RMS_T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % Q_ROPE_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % Q_ROPE_T_TILE == 0


# These helpers are shared by several dynamic prefill entry points and by
# fixed-shape decode. Infer the token dimension from each caller instead of
# cascading this module's standalone-test DynVar through the whole call graph.
@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor,
    rope_cos_t: pl.Tensor,
    rope_sin_t: pl.Tensor,
):
    t_dim = pl.tensor.dim(position_ids, 0)
    rope_tiles = (t_dim + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
    for rope_t0 in pl.spmd(rope_tiles, name_hint="qkv_rope_rows"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < t_dim:
                rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]

@pl.jit.inline
def qkv_proj_rope(
    x: pl.Tensor,
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor,
    kv: pl.Tensor,
    qr: pl.Tensor,
    qr_scale: pl.Tensor,
    late_dep: pl.Scalar[pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    t_matmul = ((t_dim + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE

    x_matmul = pl.create_tensor([T_MAX, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qkv_dynamic_pad_x"):
        for pad_t in pl.range(T_MAX):
            x_row = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
            if pad_t < t_dim:
                x_row = pl.load(x_view, [pad_t, 0], [1, D], target_memory=pl.MemorySpace.Vec)
            pl.store(x_row, [pad_t, 0], x_matmul)

    # RoPE indices and interleaved cos/signed-sin rows are head-invariant.
    # Prepare them once per token tile so the 16 Q head-group tasks do not each
    # rebuild the same arange/cast/gather chain on their critical AIV path.
    q_rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    # TGATHER executes the full physical tile, so tail rows also need valid indices.
    q_rope_swap_idx = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.INT32)
    q_rope_swap_idx_local = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    for qrp_idx in pl.spmd(
        (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE,
        name_hint="q_rope_prepare",
        allow_early_resolve=True,
    ):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        qrp_valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - qrp_t0)
        qrp_ones = pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx_local = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_row_seed = pl.cast(
            pl.mul(
                pl.cast(
                    pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32),
                    target_type=pl.FP32,
                ),
                ROPE_DIM_SCALE,
            ),
            target_type=pl.INT32,
        )
        qrp_row_grid = pl.col_expand_mul(
            pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.INT32, value=1),
            qrp_row_seed,
        )
        qrp_row_offset = pl.transpose(qrp_row_grid, axis1=0, axis2=1)
        qrp_dup_idx = pl.add(pl.cast(qrp_dup_f, target_type=pl.INT32), qrp_row_offset)
        qrp_swap_idx = pl.add(qrp_swap_idx_local, qrp_row_offset)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        qrp_cos_rows = pl.load(
            rope_cos_view,
            [qrp_t0, 0],
            [Q_ROPE_T_TILE, ROPE_DIM],
            valid_shapes=[qrp_valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        qrp_sin_rows = pl.load(
            rope_sin_view,
            [qrp_t0, 0],
            [Q_ROPE_T_TILE, ROPE_DIM],
            valid_shapes=[qrp_valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        qrp_cos = pl.cast(qrp_cos_rows, target_type=pl.FP32)
        qrp_sin = pl.cast(qrp_sin_rows, target_type=pl.FP32)
        qrp_gather_tmp = pl.create_tile(
            [Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
        )
        qrp_cos_il = pl.tile.gather(qrp_cos, qrp_dup_idx, qrp_gather_tmp)
        qrp_sin_il = pl.tile.gather(qrp_sin, qrp_dup_idx, qrp_gather_tmp)
        qrp_sin_signed = pl.mul(qrp_sin_il, qrp_sign)
        pl.store(
            pl.set_validshape(qrp_cos_il, qrp_valid_rows, ROPE_DIM),
            [qrp_t0, 0],
            q_rope_cos_il,
        )
        pl.store(
            pl.set_validshape(qrp_sin_signed, qrp_valid_rows, ROPE_DIM),
            [qrp_t0, 0],
            q_rope_sin_signed,
        )
        pl.store(qrp_swap_idx, [qrp_t0, 0], q_rope_swap_idx)
        pl.store(
            pl.set_validshape(qrp_swap_idx_local, qrp_valid_rows, ROPE_DIM),
            [qrp_t0, 0],
            q_rope_swap_idx_local,
        )

    # Split-K qr_proj (M=t_dim, K=D=4096, N=Q_LORA=1024). QR_N_TILE=128 gives
    # eight N-groups; QR_OK=2 expands them to 16 cube blocks and atomic-adds the
    # K partials into a zero-seeded output. Auto-dep on qr_fp32 orders the seed
    # before every atomic RMW.
    qr_fp32 = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.FP32)
    qr_i8_matmul = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.INT8)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
        for tc in pl.range(t_matmul // QR_M_TILE):
            ts0 = tc * QR_M_TILE
            for nb in pl.range(Q_LORA // QR_N_TILE):
                nseed0 = nb * QR_N_TILE
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = pl.full(
                    [QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0
                )
                qr_i8_matmul[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = pl.cast(
                    pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP16, value=0.0),
                    target_type=pl.INT8,
                    mode="trunc",
                )
    for qbg_idx in pl.spmd((Q_LORA // QR_N_TILE) * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):
        q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_K_SLICE
        for tc in pl.range(t_matmul // QR_M_TILE):
            t0 = tc * QR_M_TILE
            q_acc = pl.create_tile(
                [QR_M_TILE, QR_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc
            )
            for db in pl.pipeline(QR_K_SLICE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                q_x_chunk_bf16 = pl.load(
                    x_matmul,
                    [t0, qr_d0],
                    [QR_M_TILE, QR_K_TILE],
                )
                w_chunk = pl.load(
                    wq_a,
                    [qr_d0, q_a_col0],
                    [QR_K_TILE, QR_N_TILE],
                )
                if db == 0:
                    q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32 = pl.store(q_acc, [t0, q_a_col0], qr_fp32, atomic=pl.AtomicType.Add)

    # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
    for tg_idx in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
        tg = tg_idx * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - tg)
        qr_sum_tmp = pl.create_tile(
            [T_TILE, Q_LORA_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        qr_max_tmp = pl.create_tile(
            [T_TILE, Q_LORA_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        qr_rms_chunk0 = pl.load(
            qr_fp32,
            [tg, 0],
            [T_TILE, Q_LORA_TILE],
            valid_shapes=[valid_rows, Q_LORA_TILE],
            target_memory=pl.MemorySpace.Vec,
        )
        qr_sq_sum = pl.reshape(
            pl.row_sum(pl.mul(qr_rms_chunk0, qr_rms_chunk0), qr_sum_tmp),
            [1, T_TILE],
        )
        gamma_rms_chunk0 = pl.reshape(
            pl.cast(
                pl.load(gamma_cq, [0], [Q_LORA_TILE], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            ),
            [1, Q_LORA_TILE],
        )
        qr_g0 = pl.col_expand_mul(qr_rms_chunk0, gamma_rms_chunk0)
        qr_amax_g = pl.reshape(pl.row_max(pl.abs(qr_g0), qr_max_tmp), [1, T_TILE])
        for qr_rms_qb in pl.range(1, Q_LORA // Q_LORA_TILE):
            qr_rms_col0 = qr_rms_qb * Q_LORA_TILE
            qr_rms_chunk = pl.load(
                qr_fp32,
                [tg, qr_rms_col0],
                [T_TILE, Q_LORA_TILE],
                valid_shapes=[valid_rows, Q_LORA_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            qr_sq_part = pl.reshape(
                pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk), qr_sum_tmp),
                [1, T_TILE],
            )
            gamma_rms_input = pl.load(
                gamma_cq,
                [qr_rms_col0],
                [Q_LORA_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_rms_cast = pl.cast(gamma_rms_input, target_type=pl.FP32)
            gamma_rms_chunk = pl.reshape(gamma_rms_cast, [1, Q_LORA_TILE])
            qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
            qr_g_abs = pl.abs(qr_g)
            qr_amax_part = pl.reshape(pl.row_max(qr_g_abs, qr_max_tmp), [1, T_TILE])
            qr_sq_sum = pl.add(qr_sq_sum, qr_sq_part)
            qr_amax_g = pl.maximum(qr_amax_g, qr_amax_part)
        qr_rsqrt_tmp = pl.create_tile(
            [1, T_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        qr_inv_rms_row = pl.tile.rsqrt(
            pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS),
            tmp=qr_rsqrt_tmp,
        )
        qr_inv_rms = pl.reshape(qr_inv_rms_row, [T_TILE, 1])
        qr_amax_normed = pl.mul(qr_inv_rms_row, qr_amax_g)
        qr_tile_amax = pl.maximum(qr_amax_normed, INT8_AMAX_EPS)
        qr_scale_quant_row = pl.mul(pl.recip(qr_tile_amax), INT8_SCALE_MAX)
        # Preserve the original reciprocal-of-quant-scale rounding order.
        # Rewriting this as amax / INT8_SCALE_MAX is algebraically equivalent
        # but changes FP32 rounding enough to flip downstream BF16 query values.
        qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
        pl.store(pl.set_validshape(qr_tile_scale_dq, valid_rows, 1), [tg, 0], qr_scale_view)
        qr_scale_quant = pl.reshape(qr_scale_quant_row, [T_TILE, 1])

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = pl.load(
                qr_fp32,
                [tg, qa],
                [T_TILE, QUANT_TILE],
                valid_shapes=[valid_rows, QUANT_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_q_input = pl.load(
                gamma_cq,
                [qa],
                [QUANT_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_q_cast = pl.cast(gamma_q_input, target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms), gamma_q_chunk)
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
            qr_q_out = pl.set_validshape(qr_q_i8, valid_rows, QUANT_TILE)
            pl.store(qr_q_out, [tg, qa], qr_view)
            pl.store(qr_q_out, [tg, qa], qr_i8_matmul)

    # UN-MIXED qproj: keep the pure-matmul scope (cube, INT32 -> GM) separate from
    # downstream vector work. This lets the scheduler defer q dequant until AIV is free
    # instead of pinning it next to qproj and competing with the critical qr_proj AIV work.
    q_proj_i32 = pl.create_tensor([T_MAX, H * HEAD_DIM], dtype=pl.INT32)
    for hg_idx in pl.spmd(
        (H * HEAD_DIM) // QPROJ_MM_N_TILE,
        name_hint="qproj_matmul",
        allow_early_resolve=True,
    ):
        w_col0 = hg_idx * QPROJ_MM_N_TILE
        for tc in pl.range(t_matmul // QPROJ_M_TILE):
            t0 = tc * QPROJ_M_TILE
            col_acc = pl.create_tile(
                [QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32, target_memory=pl.MemorySpace.Acc
            )
            for qb in pl.pipeline(0, Q_LORA // Q_PROJ_TILE, stage=1):
                qr_proj_col0 = qb * Q_PROJ_TILE
                qr_i8_chunk = pl.load(
                    qr_i8_matmul,
                    [t0, qr_proj_col0],
                    [QPROJ_M_TILE, Q_PROJ_TILE],
                )
                wq_chunk = pl.load(
                    wq_b,
                    [qr_proj_col0, w_col0],
                    [Q_PROJ_TILE, QPROJ_MM_N_TILE],
                )
                if qr_proj_col0 == 0:
                    col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                else:
                    col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
            pl.store(col_acc, [t0, w_col0], q_proj_i32)

    # Fuse qproj dequant, per-head RMSNorm, NOPE writeback, and interleaved RoPE.
    # A full [token, head] tile fits in Vec UB, so dequantize each head once and
    # retain it across the RMS reduction instead of rereading/recomputing NOPE.
    # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for hg_idx in pl.spmd(H // Q_ROPE_H_TILE, name_hint="qproj_dequant_rms_nope_rope", allow_early_resolve=True):
        hg = hg_idx * Q_ROPE_H_TILE
        for tg_idx in pl.range((t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE):
            q_tg = tg_idx * Q_ROPE_T_TILE
            q_valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - q_tg)
            if q_valid_rows == Q_ROPE_T_TILE:
                # Keep aligned tiles on the original Tensor-level dataflow so
                # decode and aligned prefill retain constant valid rows through
                # TROWSUM/rsqrt. Only the final non-aligned tile needs TileView.
                qr_scale_dq_full = qr_scale_view[q_tg : q_tg + Q_ROPE_T_TILE, :]
                q_cos_il_full = q_rope_cos_il[q_tg : q_tg + Q_ROPE_T_TILE, :]
                q_sin_signed_full = q_rope_sin_signed[q_tg : q_tg + Q_ROPE_T_TILE, :]
                q_swap_idx_full = q_rope_swap_idx_local[q_tg : q_tg + Q_ROPE_T_TILE, :]
                for h_inner_full in pl.pipeline(Q_ROPE_H_TILE, stage=2):
                    h_full = hg + h_inner_full
                    h0_full = h_full * HEAD_DIM
                    q_head_acc_full = q_proj_i32[
                        q_tg : q_tg + Q_ROPE_T_TILE,
                        h0_full : h0_full + HEAD_DIM,
                    ]
                    q_head_scale_full = pl.reshape(
                        wq_b_scale[h0_full : h0_full + HEAD_DIM],
                        [1, HEAD_DIM],
                    )
                    q_head_acc_fp32_full = pl.cast(
                        q_head_acc_full,
                        target_type=pl.FP32,
                        mode="none",
                    )
                    q_head_row_scaled_full = pl.row_expand_mul(
                        q_head_acc_fp32_full,
                        qr_scale_dq_full,
                    )
                    q_head_dq_full = pl.col_expand_mul(
                        q_head_row_scaled_full,
                        q_head_scale_full,
                    )
                    q_head_sq_full = pl.mul(q_head_dq_full, q_head_dq_full)
                    q_head_sq_row_full = pl.row_sum(q_head_sq_full)
                    q_head_sq_sum_full = pl.reshape(q_head_sq_row_full, [1, Q_ROPE_T_TILE])
                    q_head_inv_rms_full = pl.rsqrt(
                        pl.add(pl.mul(q_head_sq_sum_full, 1.0 / HEAD_DIM), EPS),
                        high_precision=True,
                    )
                    q_head_inv_rms_t_full = pl.reshape(
                        q_head_inv_rms_full,
                        [Q_ROPE_T_TILE, 1],
                    )
                    q_nope_normed_full = pl.row_expand_mul(
                        q_head_dq_full[:, 0:NOPE_DIM],
                        q_head_inv_rms_t_full,
                    )
                    q_flat[
                        q_tg : q_tg + Q_ROPE_T_TILE,
                        h0_full : h0_full + NOPE_DIM,
                    ] = pl.cast(q_nope_normed_full, target_type=pl.BF16, mode="rint")

                    q_rope_chunk_full = pl.row_expand_mul(
                        q_head_dq_full[:, NOPE_DIM:HEAD_DIM],
                        q_head_inv_rms_t_full,
                    )
                    q_rope_swapped_full = pl.gather(
                        q_rope_chunk_full,
                        dim=-1,
                        index=q_swap_idx_full,
                    )
                    q_rope_rot_full = pl.add(
                        pl.mul(q_rope_chunk_full, q_cos_il_full),
                        pl.mul(q_rope_swapped_full, q_sin_signed_full),
                    )
                    q_flat[
                        q_tg : q_tg + Q_ROPE_T_TILE,
                        h0_full + NOPE_DIM : h0_full + NOPE_DIM + ROPE_DIM,
                    ] = pl.cast(q_rope_rot_full, target_type=pl.BF16, mode="rint")
            else:
                qr_scale_dq_t = pl.load(
                    qr_scale_view,
                    [q_tg, 0],
                    [Q_ROPE_T_TILE, 1],
                    valid_shapes=[q_valid_rows, 1],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_cos_il = pl.load(
                    q_rope_cos_il,
                    [q_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shapes=[q_valid_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_sin_signed = pl.load(
                    q_rope_sin_signed,
                    [q_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shapes=[q_valid_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_swap_idx = pl.load(
                    q_rope_swap_idx,
                    [q_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_head_reduce_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                q_gather_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                )
                # Heads share the explicit row-reduction scratch tile, so process
                # them sequentially to prevent adjacent stages from overwriting it.
                for h_inner in pl.range(Q_ROPE_H_TILE):
                    h = hg + h_inner
                    h0 = h * HEAD_DIM
                    q_head_acc = pl.load(
                        q_proj_i32,
                        [q_tg, h0],
                        [Q_ROPE_T_TILE, HEAD_DIM],
                        valid_shapes=[q_valid_rows, HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale_input = pl.load(
                        wq_b_scale,
                        [h0],
                        [HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale = pl.reshape(q_head_scale_input, [1, HEAD_DIM])
                    q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                    q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                    q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                    q_head_sq = pl.mul(q_head_dq, q_head_dq)
                    q_head_sq_sum = pl.row_sum(q_head_sq, q_head_reduce_tmp)
                    q_head_inv_rms = pl.rsqrt(
                        pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS),
                        high_precision=True,
                    )

                    q_nope_normed = pl.row_expand_mul(
                        q_head_dq[:, 0:NOPE_DIM],
                        q_head_inv_rms,
                    )
                    q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                    pl.store(
                        pl.set_validshape(q_nope_bf16, q_valid_rows, NOPE_DIM),
                        [q_tg, h0],
                        q_flat,
                    )

                    q_rope_chunk = pl.row_expand_mul(
                        q_head_dq[:, NOPE_DIM:HEAD_DIM],
                        q_head_inv_rms,
                    )
                    q_rope_swapped = pl.tile.gather(
                        q_rope_chunk,
                        q_swap_idx,
                        q_gather_tmp,
                    )
                    q_rope_rot = pl.add(
                        pl.mul(q_rope_chunk, q_cos_il),
                        pl.mul(q_rope_swapped, q_sin_signed),
                    )
                    q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                    pl.store(
                        pl.set_validshape(q_rope_bf16, q_valid_rows, ROPE_DIM),
                        [q_tg, h0 + NOPE_DIM],
                        q_flat,
                    )

    # Split-K kv_proj uses four 128-column N-groups and KV_OK=4, again producing
    # 16 cube blocks. KV is off the critical path, so more K splits only add atomic
    # contention without shortening decode.
    kv_fp32 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
        for tc in pl.range(t_matmul // KV_M_TILE):
            kts0 = tc * KV_M_TILE
            for nb in pl.range(HEAD_DIM // KV_N_TILE):
                kvseed0 = nb * KV_N_TILE
                kv_fp32[kts0 : kts0 + KV_M_TILE, kvseed0 : kvseed0 + KV_N_TILE] = pl.full(
                    [KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0
                )
    # `late_dep` is a dummy barrier hung off the rms_norm TaskId: kv_proj is off the
    # critical path, so it resolves one hop after rms_norm and lets qr_proj_matmul
    # take the cores first.
    with pl.spmd((HEAD_DIM // KV_N_TILE) * KV_OK, name_hint="kv_proj_matmul", deps=[late_dep]) as _kv_tid:
        kbg = pl.tile.get_block_idx()
        kv_col0 = (kbg // KV_OK) * KV_N_TILE
        kv_k_base = (kbg % KV_OK) * KV_K_SLICE
        for tc in pl.range(t_matmul // KV_M_TILE):
            t0 = tc * KV_M_TILE
            kv_acc = pl.create_tile(
                [KV_M_TILE, KV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc
            )
            for db in pl.pipeline(KV_K_SLICE // KV_K_TILE, stage=2):
                d0 = kv_k_base + db * KV_K_TILE
                kv_x_chunk_bf16 = pl.load(
                    x_matmul,
                    [t0, d0],
                    [KV_M_TILE, KV_K_TILE],
                )
                wkv_chunk = pl.load(
                    wkv,
                    [d0, kv_col0],
                    [KV_K_TILE, KV_N_TILE],
                )
                if db == 0:
                    kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32 = pl.store(kv_acc, [t0, kv_col0], kv_fp32, atomic=pl.AtomicType.Add)

    # Fused KV RMSNorm + interleaved (CANN A3) RoPE. One spmd task per [KV_RMS_T_TILE, HEAD_DIM]
    # row block computes the per-row inv_rms once (pass 1) and consumes it locally for
    # BOTH the NOPE writeback and the rope rotation -- so inv_rms no longer round-trips
    # through GM (the old kv_inv_rms_tensor) and the two passes collapse into a single
    # dispatch. NOPE columns [0:NOPE_DIM) and rope columns [NOPE_DIM:HEAD_DIM) are
    # disjoint, so each task writes a clean, conflict-free row block of kv. Vec UB stays
    # well under the 192 KB cap (chunks are at most [KV_RMS_T_TILE, KV_TILE] fp32).
    for tg_idx in pl.spmd((t_dim + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE, name_hint="kv_rms_norm_rope"):
        tg = tg_idx * KV_RMS_T_TILE
        valid_rows = pl.min(KV_RMS_T_TILE, t_dim - tg)
        kv_reduce_tmp = pl.create_tile(
            [KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        kv_gather_tmp = pl.create_tile(
            [KV_RMS_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
        )
        # Pass 1: per-row sum of squares over the full HEAD_DIM -> inv_rms.
        kv_sq_sum = pl.tile.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HEAD_DIM // KV_TILE, stage=2):
            kv_sq_col0 = kb * KV_TILE
            kv_chunk = pl.load(
                kv_fp32,
                [tg, kv_sq_col0],
                [KV_RMS_T_TILE, KV_TILE],
                valid_shapes=[valid_rows, KV_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            kv_sq_sum = pl.add(
                kv_sq_sum,
                pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk), kv_reduce_tmp), [1, KV_RMS_T_TILE]),
            )
        kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

        # NOPE writeback: rms-normalize columns [0:NOPE_DIM) with per-column gamma.
        for nb in pl.pipeline(NOPE_DIM // KV_TILE, stage=2):
            n0 = nb * KV_TILE
            kv_chunk = pl.load(
                kv_fp32,
                [tg, n0],
                [KV_RMS_T_TILE, KV_TILE],
                valid_shapes=[valid_rows, KV_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_kv_input = pl.load(
                gamma_ckv,
                [n0],
                [KV_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_kv_cast = pl.cast(gamma_kv_input, target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv_nope_out = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
            pl.store(pl.set_validshape(kv_nope_out, valid_rows, KV_TILE), [tg, n0], kv_view)

        # RoPE writeback on columns [NOPE_DIM:HEAD_DIM), interleaved (CANN A3) swap-gather
        # (same form as qproj_dequant_rms_nope_rope), built in-kernel. inv_rms (per-row, the same
        # factor used for NOPE above) and gamma (per-column, full ROPE_DIM) are folded into
        # kv_rope_norm_chunk BEFORE the swap so the swapped lane n[j^1] carries gamma[j^1]
        # (gamma does NOT commute with the rotation; inv_rms does).
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j]
        gamma_rope_input = pl.load(
            gamma_ckv,
            [NOPE_DIM],
            [ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        gamma_rope_cast = pl.cast(gamma_rope_input, target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = pl.load(
            kv_fp32,
            [tg, NOPE_DIM],
            [KV_RMS_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
        kv_ones = pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        kv_col = pl.col_expand_mul(
            kv_ones,
            pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
        )
        kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))                                            # j%2
        kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))                              # j^1
        kv_row_seed = pl.cast(
            pl.mul(
                pl.cast(
                    pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32),
                    target_type=pl.FP32,
                ),
                ROPE_DIM_SCALE,
            ),
            target_type=pl.INT32,
        )
        kv_row_grid = pl.col_expand_mul(
            pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.INT32, value=1),
            kv_row_seed,
        )
        kv_row_offset = pl.transpose(kv_row_grid, axis1=0, axis2=1)
        kv_dup_idx = pl.add(pl.cast(kv_dup_f, target_type=pl.INT32), kv_row_offset)
        kv_swap_idx = pl.add(pl.cast(kv_swap_f, target_type=pl.INT32), kv_row_offset)
        kv_sign = pl.sub(pl.mul(kv_lane, 2.0), 1.0)                                                # [-1,+1,...]
        kv_cos_rows = pl.load(
            rope_cos_view,
            [tg, 0],
            [KV_RMS_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        kv_sin_rows = pl.load(
            rope_sin_view,
            [tg, 0],
            [KV_RMS_T_TILE, ROPE_DIM],
            valid_shapes=[valid_rows, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        kv_cos_il = pl.tile.gather(pl.cast(kv_cos_rows, target_type=pl.FP32), kv_dup_idx, kv_gather_tmp)
        kv_sin_il = pl.tile.gather(pl.cast(kv_sin_rows, target_type=pl.FP32), kv_dup_idx, kv_gather_tmp)
        kv_swapped = pl.tile.gather(kv_rope_norm_chunk, kv_swap_idx, kv_gather_tmp)
        kv_rope_rot = pl.add(pl.mul(kv_rope_norm_chunk, kv_cos_il), pl.mul(pl.mul(kv_swapped, kv_sign), kv_sin_il))
        kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode="rint")
        pl.store(pl.set_validshape(kv_rope_i16, valid_rows, ROPE_DIM), [tg, NOPE_DIM], kv_view)

    return q


@pl.jit
def qkv_proj_rope_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
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
        wq_b,
        wq_b_scale,
        wkv,
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

    x = tensors["x"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
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

    def matmul_bf16_input_fp32(a, b):
        a_fp32 = a.to(torch.bfloat16).float()
        b_fp32 = b.to(torch.bfloat16).float()
        return torch.matmul(a_fp32, b_fp32).float()

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
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)   # [T, Q_LORA]
    # W8A8C16: wq_b W8 per-output-channel int8; qr_out A8 per-token int8.
    # flash: also quantizes wq_a/wkv to fp8 (default Linear dtype).
    qr_i8, qr_scale = int8_quant_per_row(qr_out.float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(t_dim, H, HEAD_DIM)
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv                                            # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    # KV path
    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)  # [T, HEAD_DIM]
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:]  = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    # Inputs match cann test_mla_prolog_quant_pypto gen_mla_prolog_input_data (uniform).
    def init_x():
        return torch.empty([T, D], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_wq_a():
        return torch.empty([D, Q_LORA], dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    def init_wq_b():
        return torch.empty([Q_LORA, H * HEAD_DIM], dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    def init_wkv():
        return torch.empty([D, HEAD_DIM], dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    def init_cos():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_sin():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_gamma_cq():
        return torch.empty([Q_LORA], dtype=torch.bfloat16).uniform_(-1, 1)
    def init_gamma_ckv():
        return torch.empty([HEAD_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wq_b_scale = wq_b_scale.view(H * HEAD_DIM)

    return [
        TensorSpec("x",         [T, D],                 torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a",      [D, Q_LORA],            torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b",      [Q_LORA, H * HEAD_DIM], torch.int8,     init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv",       [D, HEAD_DIM],          torch.bfloat16, init_value=init_wkv),
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
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="Override the physical token dimension for one selected mode.",
    )
    parser.add_argument("--enable-l2-swimlane", type=int, choices=[0, 1, 2, 4], default=0,
                        help="L2 swimlane level: 0=off, 1=per-kernel AICore timing "
                             "(prints the per-function Task Statistics table), 2=+AICPU timing.")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.num_tokens is not None:
        if args.mode == "all":
            parser.error("--num-tokens requires --mode decode or --mode prefill")
        if not 1 <= args.num_tokens <= T_MAX:
            parser.error(f"--num-tokens must be in [1, {T_MAX}]")

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        if args.num_tokens is not None:
            B, S = 1, args.num_tokens
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
                "q":        ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "kv":       ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "qr":       ratio_allclose(atol=1, rtol=0, max_error_ratio=0),
                "qr_scale": ratio_allclose(atol=2.5e-5, rtol=5e-3),
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
