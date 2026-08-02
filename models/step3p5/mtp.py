# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 Multi-Token-Prediction (MTP) layers — TP-aware (Phase 9 Wave 3).

The step3p5 checkpoint carries three "next-N-predict" layers (indices
45/46/47 in the 48-long ``LAYER_TYPES`` table) hanging off the tail of
the 45 main layers. Each MTP layer fuses an input projection from the
previous-hidden + next-token embedding back into the hidden width, then
runs a step3p5 SWA decoder block (LAYER_TYPES[45..47] all evaluate to
``sliding_attention``) followed by a per-MTP shared_head producing its
own next-token logits.

TP slicing (Phase 9 Wave 3):
  * ``enorm`` / ``hnorm`` gammas are REPLICATED on every rank (per-row
    RMSNorm runs identically across the group).
  * ``eh_proj`` is **row-sliced** in checkpoint orientation
    ``[HIDDEN, 2*HIDDEN]`` to ``[HIDDEN_LOCAL=512, 2*HIDDEN]`` per card;
    in kernel orientation (``b_trans=True`` matmul) that lands as
    ``[2*HIDDEN, HIDDEN_LOCAL=512]`` per card. The matmul produces a
    rank-local ``[BATCH, HIDDEN_LOCAL]`` partial. The body writes that
    partial into the rank's column slot of a zero-filled
    ``[BATCH, HIDDEN]`` buffer (other ranks' slots stay zero), then
    runs :func:`tp_all_reduce` so every rank ends up with the same
    fully-assembled ``[BATCH, HIDDEN]`` ``mtp_in``.
  * The standard SWA attention block is provided by Wave-2's
    ``attention_swa`` (TP-sharded by KV/Q heads + TP all-reduce on
    o_proj). Reused as-is via ``pl.inline``.
  * The dense MLP follows the SWA block uses the same TP slicing as
    ``decode_layer._dense_mlp_body_tp`` — ``w_gate / w_up`` shape
    ``[HIDDEN, INTERMEDIATE_LOCAL=1408]`` and ``w_down``
    ``[INTERMEDIATE_LOCAL, HIDDEN]`` per card, plus a tp_all_reduce on
    the partial hidden before the residual add.
  * ``shared_head.output`` is vocab-sliced to
    ``[HIDDEN, VOCAB_LOCAL=16112]`` per card (same convention as
    ``rms_lm_head.py``); the kernel emits the per-rank logits shard
    ``[USER_BATCH, VOCAB_LOCAL]`` without a cross-rank gather.

Per-MTP-layer weight tables stack along the leading axis:
  * ``enorm_weight`` ``[NUM_MTP_LAYERS, HIDDEN]`` FP32 (replicated)
  * ``hnorm_weight`` ``[NUM_MTP_LAYERS, HIDDEN]`` FP32 (replicated)
  * ``eh_proj_weight`` ``[NUM_MTP_LAYERS * HIDDEN_LOCAL, EH_IN]`` BF16
    (row slice of the checkpoint's ``[HIDDEN, 2*HIDDEN]`` weight)
  * ``shared_head_norm_weight`` ``[NUM_MTP_LAYERS, HIDDEN]`` FP32
    (replicated)
  * ``shared_head_output_weight``
    ``[NUM_MTP_LAYERS * VOCAB_LOCAL, HIDDEN]`` BF16 (vocab slice)

Window contract (caller supplies):
  * For SWA attention: ``attn_tmp_window`` BF16
    ``[BATCH, HIDDEN // TP_WORLD_SIZE]`` and ``attn_signal_window``
    INT32 ``[TP_WORLD_SIZE, 1]``.
  * For eh_proj's tp_all_reduce: ``eh_tmp_window`` /
    ``eh_signal_window`` (same shapes).
  * For dense MLP's tp_all_reduce: ``mlp_tmp_window`` /
    ``mlp_signal_window`` (same shapes).
  Each call site allocates a fresh signal_window slot so the
  AtomicAdd ring-step counters do not collide across layers /
  collectives.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import zero_centered_rmsnorm_apply
from .attention_swa import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_SWA,
    attention_swa,
)
from .config import (
    BATCH,
    BATCH_TILE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    FINAL_RMS_K_CHUNK,
    HEAD_DIM,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_SWA_LOCAL,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LM_HEAD_K_CHUNK,
    NUM_HEADS_SWA_LOCAL,
    NUM_HIDDEN_LAYERS,
    NUM_NEXTN_PREDICT_LAYERS,
    OUT_PROJ_N_CHUNK,
    ROPE_SEQ_DYN,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    VOCAB_CHUNK,
    VOCAB_LOCAL,
)
from .decode_layer import _dense_mlp_body_tp


# -----------------------------------------------------------------------------
# Compile-time constants and dynamic dims.
# -----------------------------------------------------------------------------
NUM_MTP_LAYERS = NUM_NEXTN_PREDICT_LAYERS  # 3
MTP_START_LAYER = NUM_HIDDEN_LAYERS        # 45
EH_IN = 2 * HIDDEN                         # 8192 (concat[enorm, hnorm] width)
HIDDEN_LOCAL = HIDDEN // TP_WORLD_SIZE     # 512 — eh_proj's per-card output dim
INTER_LOCAL = INTERMEDIATE_LOCAL           # 1408
TP_CHUNK = HIDDEN // TP_WORLD_SIZE         # 512 — tp_all_reduce ring chunk

MTP_EH_ROWS_DYN = pl.dynamic("MTP_EH_ROWS_DYN")
MTP_VOCAB_ROWS_DYN = pl.dynamic("MTP_VOCAB_ROWS_DYN")

assert EH_IN % K_CHUNK == 0
assert HIDDEN % FINAL_RMS_K_CHUNK == 0
assert HIDDEN % LM_HEAD_K_CHUNK == 0
assert VOCAB_LOCAL % VOCAB_CHUNK == 0


# Pick the eh_proj output chunk; HIDDEN_LOCAL = 512 with OUT_PROJ_N_CHUNK = 256
# gives 2 SPMD steps. If config.OUT_PROJ_N_CHUNK changes to a value > 512 we
# fall back to the whole HIDDEN_LOCAL in one chunk.
EH_OUT_CHUNK = OUT_PROJ_N_CHUNK if OUT_PROJ_N_CHUNK <= HIDDEN_LOCAL else HIDDEN_LOCAL
assert HIDDEN_LOCAL % EH_OUT_CHUNK == 0


# =============================================================================
# Inline helper: TP-aware MTP input projection.
# enorm + hnorm + concat + row-sliced eh_proj + tp_all_reduce -> mtp_in.
# =============================================================================
@pl.jit.inline
def _mtp_input_proj_body(
    prev_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    embed_next: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    enorm_weight: pl.Tensor[[NUM_NEXTN_PREDICT_LAYERS, HIDDEN], pl.FP32],
    hnorm_weight: pl.Tensor[[NUM_NEXTN_PREDICT_LAYERS, HIDDEN], pl.FP32],
    eh_proj_weight: pl.Tensor[[MTP_EH_ROWS_DYN, 2 * HIDDEN], pl.BF16],
    mtp_in_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    mtp_layer_idx: pl.Scalar[pl.INT32],
    eh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN // TP_WORLD_SIZE], pl.BF16],
    eh_signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """Build mtp_in via row-sliced eh_proj + TP all-reduce.

    enorm(embed_next) and hnorm(prev_hidden) run REPLICATED on every
    rank (per-row RMSNorm with replicated gamma); the concat tile
    ``[BATCH, EH_IN]`` is identical on every rank. The eh_proj weight is
    row-sliced: each rank holds ``HIDDEN_LOCAL`` rows of the checkpoint's
    ``[HIDDEN, 2*HIDDEN]`` weight, kernel orientation
    ``[HIDDEN_LOCAL, EH_IN]``. The matmul ``concat @ W.T`` (b_trans=True)
    produces a rank-local ``[BATCH, HIDDEN_LOCAL]`` shard. The body
    writes that shard into the rank's column slot
    ``[my_rank * HIDDEN_LOCAL : (my_rank + 1) * HIDDEN_LOCAL]`` of a
    zero-filled ``[BATCH, HIDDEN]`` partial buffer; the rest of the
    columns stay at zero. ``tp_all_reduce`` then sums the partials so
    every rank ends up with the fully-assembled mtp_in.
    """
    hidden_blocks = HIDDEN // K_CHUNK
    eh_blocks = (2 * HIDDEN) // K_CHUNK
    eh_out_blocks = (HIDDEN // TP_WORLD_SIZE) // OUT_PROJ_N_CHUNK
    layer_out_base = mtp_layer_idx * (HIDDEN // TP_WORLD_SIZE)

    concat_tile = pl.create_tensor([BATCH, 2 * HIDDEN], dtype=pl.BF16)

    # ── 1. enorm(embed_next) -> lower half [0, HIDDEN). ────────────────
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_enorm"):
            e_sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden_blocks):
                e_sq_k0 = kb * K_CHUNK
                e_sq_chunk = pl.cast(
                    pl.slice(embed_next, [BATCH_TILE, K_CHUNK], [b0, e_sq_k0]),
                    target_type=pl.FP32,
                )
                e_sq_sum = pl.add(
                    e_sq_sum,
                    pl.reshape(
                        pl.row_sum(pl.mul(e_sq_chunk, e_sq_chunk)),
                        [1, BATCH_TILE],
                    ),
                )
            e_inv_rms = pl.reshape(
                pl.recip(pl.sqrt(pl.add(pl.mul(e_sq_sum, HIDDEN_INV), EPS))),
                [BATCH_TILE, 1],
            )
            for kb in pl.range(hidden_blocks):
                e_norm_k0 = kb * K_CHUNK
                e_chunk = pl.cast(
                    pl.slice(embed_next, [BATCH_TILE, K_CHUNK], [b0, e_norm_k0]),
                    target_type=pl.FP32,
                )
                e_gamma = pl.slice(
                    enorm_weight, [1, K_CHUNK], [mtp_layer_idx, e_norm_k0],
                )
                e_scaled = pl.row_expand_mul(e_chunk, e_inv_rms)
                e_normed = pl.col_expand_mul(e_scaled, pl.add(e_gamma, 1.0))
                concat_tile = pl.assemble(
                    concat_tile,
                    pl.cast(e_normed, target_type=pl.BF16),
                    [b0, e_norm_k0],
                )

    # ── 2. hnorm(prev_hidden) -> upper half [HIDDEN, 2*HIDDEN). ────────
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_hnorm"):
            h_sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden_blocks):
                h_sq_k0 = kb * K_CHUNK
                h_sq_chunk = pl.cast(
                    pl.slice(prev_hidden, [BATCH_TILE, K_CHUNK], [b0, h_sq_k0]),
                    target_type=pl.FP32,
                )
                h_sq_sum = pl.add(
                    h_sq_sum,
                    pl.reshape(
                        pl.row_sum(pl.mul(h_sq_chunk, h_sq_chunk)),
                        [1, BATCH_TILE],
                    ),
                )
            h_inv_rms = pl.reshape(
                pl.recip(pl.sqrt(pl.add(pl.mul(h_sq_sum, HIDDEN_INV), EPS))),
                [BATCH_TILE, 1],
            )
            for kb in pl.range(hidden_blocks):
                h_norm_k0 = kb * K_CHUNK
                h_chunk = pl.cast(
                    pl.slice(prev_hidden, [BATCH_TILE, K_CHUNK], [b0, h_norm_k0]),
                    target_type=pl.FP32,
                )
                h_gamma = pl.slice(
                    hnorm_weight, [1, K_CHUNK], [mtp_layer_idx, h_norm_k0],
                )
                h_scaled = pl.row_expand_mul(h_chunk, h_inv_rms)
                h_normed = pl.col_expand_mul(h_scaled, pl.add(h_gamma, 1.0))
                concat_tile = pl.assemble(
                    concat_tile,
                    pl.cast(h_normed, target_type=pl.BF16),
                    [b0, HIDDEN + h_norm_k0],
                )

    # ── 3. row-sliced eh_proj + zero-pad into per-rank slot. ───────────
    # The partial buffer's per-rank slot is
    # ``[:, my_rank * HIDDEN_LOCAL : (my_rank + 1) * HIDDEN_LOCAL]`` —
    # the rest of the columns stay zero so the subsequent tp_all_reduce
    # produces the fully-assembled mtp_in.
    partial = pl.full([BATCH, HIDDEN], dtype=pl.BF16, value=0.0)
    rank_col_base = my_rank * (HIDDEN // TP_WORLD_SIZE)

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        for ob in pl.spmd(
            eh_out_blocks, name_hint="mtp_eh_proj_tp",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            o0 = ob * OUT_PROJ_N_CHUNK
            a_chunk_0 = pl.slice(concat_tile, [BATCH_TILE, K_CHUNK], [b0, 0])
            w_chunk_0 = pl.slice(
                eh_proj_weight,
                [OUT_PROJ_N_CHUNK, K_CHUNK],
                [layer_out_base + o0, 0],
            )
            eh_acc = pl.matmul(
                a_chunk_0, w_chunk_0, out_dtype=pl.FP32, b_trans=True,
            )
            for kb in pl.range(1, eh_blocks):
                k0 = kb * K_CHUNK
                a_chunk = pl.slice(concat_tile, [BATCH_TILE, K_CHUNK], [b0, k0])
                w_chunk = pl.slice(
                    eh_proj_weight,
                    [OUT_PROJ_N_CHUNK, K_CHUNK],
                    [layer_out_base + o0, k0],
                )
                eh_acc = pl.matmul_acc(eh_acc, a_chunk, w_chunk, b_trans=True)
            partial = pl.assemble(
                partial,
                pl.cast(eh_acc, target_type=pl.BF16),
                [b0, rank_col_base + o0],
            )

    # ── 4. tp_all_reduce: assemble the full mtp_in across the group. ───
    # Phase X.2: ``self.tp_all_reduce`` resolves to a class method on the
    # @pl.program that ultimately wires in this MTP layer (TBD, Phase 9
    # Wave-4); the body must be supplied by that class.
    # Phase 15.1 single-rank gate: at TP=1 skip the call so orchestration
    # codegen does not emit a stale SSA rename for the SimplifyPass-elided
    # ring loop body (mirror of attention_full.py 15.B).
    if TP_WORLD_SIZE > 1:
        self.tp_all_reduce(
            partial, eh_tmp_window, eh_signal_window, my_rank,
        )

    # ── 5. copy reduced partial -> mtp_in_out. ─────────────────────────
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_eh_copy_out"):
        for kb in pl.range(hidden_blocks):
            k0 = kb * K_CHUNK
            chunk = pl.slice(partial, [BATCH, K_CHUNK], [0, k0])
            mtp_in_out = pl.assemble(mtp_in_out, chunk, [0, k0])

    return mtp_in_out


# =============================================================================
# Inline helper: per-MTP shared_head — TP vocab-sliced.
# zero-centered RMSNorm + LM-head matmul -> [USER_BATCH, VOCAB_LOCAL] FP32.
# =============================================================================
@pl.jit.inline
def _mtp_shared_head_body(
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    shared_head_norm_weight: pl.Tensor[[NUM_NEXTN_PREDICT_LAYERS, HIDDEN], pl.FP32],
    shared_head_output_weight: pl.Tensor[
        [MTP_VOCAB_ROWS_DYN, HIDDEN], pl.BF16
    ],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    logits_out: pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32],
    mtp_layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]:
    """Per-MTP shared_head: zero-centred RMSNorm + vocab-sliced matmul.

    Mirrors ``rms_lm_head.rms_lm_head`` but indexes the per-MTP norm /
    output slabs by ``mtp_layer_idx``. Emits the per-rank
    ``[USER_BATCH, VOCAB_LOCAL]`` shard; the caller plumbs whatever
    cross-rank gather (e.g. argmax + (idx, val) all-gather) is needed.
    """
    user_batch = pl.tensor.dim(seq_lens, 0)
    layer_vocab_base = mtp_layer_idx * VOCAB_LOCAL

    final_normed = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="mtp_shared_head_rmsnorm",
        ):
            f_sq_sum = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                f_sq_k0 = kb * FINAL_RMS_K_CHUNK
                f_sq_chunk = pl.cast(
                    pl.slice(
                        hidden_states,
                        [BATCH_TILE, FINAL_RMS_K_CHUNK],
                        [b0, f_sq_k0],
                    ),
                    target_type=pl.FP32,
                )
                f_sq_sum = pl.add(
                    f_sq_sum,
                    pl.reshape(
                        pl.row_sum(pl.mul(f_sq_chunk, f_sq_chunk)),
                        [1, BATCH_TILE],
                    ),
                )
            f_inv_rms = pl.reshape(
                pl.rsqrt(pl.add(pl.mul(f_sq_sum, HIDDEN_INV), EPS)),
                [BATCH_TILE, 1],
            )
            for kb in pl.range(HIDDEN // FINAL_RMS_K_CHUNK):
                f_norm_k0 = kb * FINAL_RMS_K_CHUNK
                f_chunk = pl.cast(
                    pl.slice(
                        hidden_states,
                        [BATCH_TILE, FINAL_RMS_K_CHUNK],
                        [b0, f_norm_k0],
                    ),
                    target_type=pl.FP32,
                )
                f_gamma = pl.slice(
                    shared_head_norm_weight,
                    [1, FINAL_RMS_K_CHUNK],
                    [mtp_layer_idx, f_norm_k0],
                )
                f_scaled = pl.row_expand_mul(f_chunk, f_inv_rms)
                f_normed = pl.col_expand_mul(f_scaled, pl.add(f_gamma, 1.0))
                final_normed = pl.assemble(
                    final_normed,
                    pl.cast(f_normed, target_type=pl.BF16),
                    [b0, f_norm_k0],
                )

    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        lm_valid_rows = pl.min(BATCH_TILE, user_batch - b0)
        for ob in pl.parallel(VOCAB_LOCAL // VOCAB_CHUNK):
            lm_o0 = ob * VOCAB_CHUNK
            lm_acc_gm = pl.create_tensor(
                [BATCH_TILE, VOCAB_CHUNK], dtype=pl.FP32,
            )
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mtp_lm_head_tp"):
                lm_hidden_chunk = pl.slice(
                    final_normed, [BATCH_TILE, LM_HEAD_K_CHUNK], [b0, 0],
                )
                lm_weight_chunk = pl.slice(
                    shared_head_output_weight,
                    [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                    [layer_vocab_base + lm_o0, 0],
                )
                lm_acc = pl.matmul(
                    lm_hidden_chunk, lm_weight_chunk,
                    out_dtype=pl.FP32, b_trans=True,
                )
                for kb in pl.range(1, HIDDEN // LM_HEAD_K_CHUNK):
                    lm_k0 = kb * LM_HEAD_K_CHUNK
                    lm_hidden_chunk = pl.slice(
                        final_normed,
                        [BATCH_TILE, LM_HEAD_K_CHUNK],
                        [b0, lm_k0],
                    )
                    lm_weight_chunk = pl.slice(
                        shared_head_output_weight,
                        [VOCAB_CHUNK, LM_HEAD_K_CHUNK],
                        [layer_vocab_base + lm_o0, lm_k0],
                    )
                    lm_acc = pl.matmul_acc(
                        lm_acc, lm_hidden_chunk, lm_weight_chunk, b_trans=True,
                    )
                lm_acc_gm = pl.assemble(lm_acc_gm, lm_acc, [0, 0])

            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="mtp_lm_head_store_tp",
            ):
                lm_acc_chunk = pl.slice(
                    lm_acc_gm, [BATCH_TILE, VOCAB_CHUNK], [0, 0],
                )
                lm_acc_trimmed = pl.slice(
                    lm_acc_chunk,
                    [BATCH_TILE, VOCAB_CHUNK],
                    [0, 0],
                    valid_shape=[lm_valid_rows, VOCAB_CHUNK],
                )
                logits_out = pl.assemble(
                    logits_out, lm_acc_trimmed, [b0, lm_o0],
                )

    return logits_out


# =============================================================================
# Inline helper: one full TP-aware MTP layer.
# =============================================================================
@pl.jit.inline
def mtp_layer(
    prev_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    embed_next: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    enorm_weight: pl.Tensor[[NUM_NEXTN_PREDICT_LAYERS, HIDDEN], pl.FP32],
    hnorm_weight: pl.Tensor[[NUM_NEXTN_PREDICT_LAYERS, HIDDEN], pl.FP32],
    eh_proj_weight: pl.Tensor[[MTP_EH_ROWS_DYN, 2 * HIDDEN], pl.BF16],
    # Standard step3p5 SWA attention bundle (per-card TP slice, identical
    # to main layers' shapes for global indices 45/46/47).
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_SWA_LOCAL], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, 128], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, 128], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_QHIDDEN_ROWS_DYN_SWA, HIDDEN], pl.BF16],
    w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL], pl.BF16],
    # TP-sliced dense MLP bundle (per the decode_layer convention).
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE_LOCAL], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE_LOCAL], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    # Per-MTP shared head (vocab-sliced).
    shared_head_norm_weight: pl.Tensor[[NUM_NEXTN_PREDICT_LAYERS, HIDDEN], pl.FP32],
    shared_head_output_weight: pl.Tensor[
        [MTP_VOCAB_ROWS_DYN, HIDDEN], pl.BF16
    ],
    # Outputs.
    mtp_hidden_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    mtp_logits_out: pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32],
    # Indices.
    mtp_layer_idx: pl.Scalar[pl.INT32],
    global_layer_idx: pl.Scalar[pl.INT32],
    # TP windows (eh_proj, attention, dense MLP) — caller allocates fresh.
    eh_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN // TP_WORLD_SIZE], pl.BF16],
    eh_signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    attn_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN // TP_WORLD_SIZE], pl.BF16],
    attn_signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    mlp_tmp_window: pld.DistributedTensor[[BATCH, HIDDEN // TP_WORLD_SIZE], pl.BF16],
    mlp_signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """One full TP-aware MTP layer.

    The caller chains MTP layers by feeding ``mtp_hidden_out`` of layer
    ``k`` as ``prev_hidden`` of layer ``k+1``.

    ``mtp_layer_idx`` is the LOCAL index 0..NUM_MTP_LAYERS-1 (used to
    slice the per-MTP weight stacks). ``global_layer_idx`` is the global
    index 45+mtp_layer_idx (used to slice the standard
    [LAYER_DYN, ...] tables that span all 48 layers).
    """
    # 1. enorm + hnorm + concat + row-sliced eh_proj + tp_all_reduce.
    mtp_in = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    mtp_in = _mtp_input_proj_body(
        prev_hidden, embed_next,
        enorm_weight, hnorm_weight, eh_proj_weight,
        mtp_in, mtp_layer_idx,
        eh_tmp_window, eh_signal_window, my_rank,
    )

    # 2. SWA attention — uses the Wave-2 TP attention body (input
    #    RMSNorm + tp_all_reduce on o_proj live inside).
    resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    resid1 = attention_swa(
        mtp_in, input_rms_weight,
        wq, wk, wv,
        q_norm_weight, k_norm_weight,
        seq_lens, block_table, slot_mapping,
        rope_cos, rope_sin,
        k_cache, v_cache,
        wo, w_g, resid1, global_layer_idx,
        attn_tmp_window, attn_signal_window, my_rank,
    )

    # 3. TP-sliced dense MLP + tp_all_reduce + residual.
    mtp_hidden_out = _dense_mlp_body_tp(
        resid1, post_rms_weight,
        w_gate, w_up, w_down,
        mtp_hidden_out, global_layer_idx,
        mlp_tmp_window, mlp_signal_window, my_rank,
    )

    # 4. Per-MTP shared head: zero-centred RMSNorm + vocab-sliced LM head.
    mtp_logits_out = _mtp_shared_head_body(
        mtp_hidden_out, shared_head_norm_weight, shared_head_output_weight,
        seq_lens, mtp_logits_out, mtp_layer_idx,
    )

    return mtp_hidden_out, mtp_logits_out


# =============================================================================
# Torch references.
# =============================================================================
def _zero_centered_rmsnorm_torch(x, gamma, eps):
    import torch

    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    gamma_eff = gamma.float() + 1.0
    return x_f * torch.rsqrt(var + eps) * gamma_eff


def golden_mtp_input_proj(tensors):
    """Torch reference for the (single-rank) ``_mtp_input_proj_body``.

    Matches a per-rank kernel run by computing the corresponding row
    block of the eh_proj matmul. Multi-rank acceptance is policed by
    the distributed-mock harness in ``decode_fwd.py``.
    """
    import torch

    prev = tensors["prev_hidden"]
    embed = tensors["embed_next"]
    enorm_w = tensors["enorm_weight"][0:1, :]
    hnorm_w = tensors["hnorm_weight"][0:1, :]
    eh_w = tensors["eh_proj_weight"][0:HIDDEN_LOCAL, :]

    e_normed = _zero_centered_rmsnorm_torch(embed, enorm_w, EPS).to(torch.bfloat16)
    h_normed = _zero_centered_rmsnorm_torch(prev, hnorm_w, EPS).to(torch.bfloat16)
    concat = torch.cat([e_normed, h_normed], dim=-1)
    mtp_in_local = (concat.float() @ eh_w.float().T).to(torch.bfloat16)
    tensors["out"][:] = mtp_in_local


def golden_mtp_shared_head(tensors):
    """Torch reference for the per-rank shared head (vocab-sliced shard)."""
    import torch

    hidden = tensors["hidden_states"]
    norm_w = tensors["shared_head_norm_weight"][0:1, :]
    out_w = tensors["shared_head_output_weight"][0:VOCAB_LOCAL, :]

    normed = _zero_centered_rmsnorm_torch(hidden, norm_w, EPS).to(torch.bfloat16)
    logits = normed.float() @ out_w.float().T
    tensors["out"][:] = logits.to(torch.float32)


# =============================================================================
# Tensor-spec helpers and CLI smoke entry.
# =============================================================================
def make_pass_rate_compare(threshold: float):
    def cmp(actual, expected, *, rtol, atol, **_):
        import torch

        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        rate = close.float().mean().item()
        n_fail = int((~close).sum().item())
        ok = rate >= threshold
        msg = (
            f"    pass_rate={rate:.6f} (threshold {threshold:.6f}), "
            f"{n_fail}/{actual.numel()} mismatched rtol={rtol} atol={atol}"
        )
        return ok, msg
    cmp.__name__ = f"pass_rate>={threshold:.4f}"
    return cmp


if __name__ == "__main__":
    # Phase 9 Wave 3 smoke: parse-clean + per-card shape sanity. The
    # eh_proj's tp_all_reduce requires a distributed runtime to validate
    # numerically; the multi-rank harness lives in decode_fwd.py.
    print("[mtp] module loaded; TP/EP shapes:")
    print(f"      HIDDEN_LOCAL={HIDDEN_LOCAL}  (eh_proj per-rank output dim)")
    print(f"      INTER_LOCAL={INTER_LOCAL}    (dense MLP per-rank intermediate)")
    print(f"      VOCAB_LOCAL={VOCAB_LOCAL}    (shared_head per-rank output)")
    print(
        f"      NUM_MTP_LAYERS={NUM_MTP_LAYERS}  global indices "
        f"{tuple(range(MTP_START_LAYER, MTP_START_LAYER + NUM_MTP_LAYERS))}"
    )


__all__ = [
    "NUM_MTP_LAYERS",
    "MTP_START_LAYER",
    "EH_IN",
    "HIDDEN_LOCAL",
    "TP_CHUNK",
    "MTP_EH_ROWS_DYN",
    "MTP_VOCAB_ROWS_DYN",
    "_mtp_input_proj_body",
    "_mtp_shared_head_body",
    "mtp_layer",
    "golden_mtp_input_proj",
    "golden_mtp_shared_head",
    "make_pass_rate_compare",
]
