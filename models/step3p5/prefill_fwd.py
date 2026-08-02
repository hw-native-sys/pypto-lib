# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 multi-layer prefill forward pass — TP/EP wired (Phase 6).

Top-level distributed prefill entry. The 45 main layers are dispatched
in a Python compile-time loop; each layer's TP+EP-aware ``@pl.program``
is selected by ``select_prefill_layer(layer_idx)``. After the layer loop
the residual stream goes through ``rms_lm_head`` (replicated zero-
centred RMSNorm + vocab-sliced LM head matmul) and each rank emits its
own ``[USER_BATCH, VOCAB_LOCAL]`` shard.

The 3 MTP layers (45..47) are NOT included here — see ``mtp.py``; the
integration author from Phase 8 wires them on top of the prefill fwd.

Per-layer dispatch (mirrors ``decode_layer.select_decode_layer``)
------------------------------------------------------------------
  layer_idx  | attention   | MLP
  -----------|-------------|------------
  0          | full        | TP dense MLP
  1, 2       | swa         | TP dense MLP
  3..44      | full / swa  | EP+TP MoE (silu/silu, silu+swiglu7, swiglu7+swiglu16)

Eight specialisations are exposed:

  - ``prefill_layer_full_dense``
  - ``prefill_layer_swa_dense``
  - ``prefill_layer_full_moe_silu_silu``
  - ``prefill_layer_full_moe_swiglu7_silu``
  - ``prefill_layer_full_moe_swiglu7_swiglu16``
  - ``prefill_layer_swa_moe_silu_silu``
  - ``prefill_layer_swa_moe_swiglu7_silu``
  - ``prefill_layer_swa_moe_swiglu7_swiglu16``

Window-pool layout
------------------
Copied from ``decode_fwd.py``: per-layer ``host_orch`` pattern, fresh
signal windows per call site to avoid AtomicAdd ring-step collisions
across collectives. The TP-AR scratch / EP a2a payload pools are
allocated inside each per-layer program's ``host_orch`` (the MoE
program reuses its slot across the prefill-T tile loop because each
tile flushes before the next reads).

KV-cache convention (TP-aware)
------------------------------
Each rank holds only its slice of the KV heads — ``KV_HEADS_LOCAL = 1``
KV head per rank under TP=8. The per-layer cache row stride is
``MAX_BLOCKS_PER_SEQ * KV_HEADS_LOCAL * BLOCK_SIZE`` rows of
``HEAD_DIM`` BF16 lanes. The 45-layer K-cache and V-cache are stacked
along their leading axis.

RoPE table convention
---------------------
Per-flavour stacks ``rope_cos_full / rope_sin_full`` size
``[NUM_FULL_LAYERS * MAX_SEQ, 64]`` and ``rope_cos_swa / rope_sin_swa``
size ``[NUM_SWA_LAYERS * MAX_SEQ, 128]`` — replicated on every rank.

Distributed-mock harness
------------------------
The ``__main__`` block runs a pure-torch 8-rank simulation of the
full 45-layer prefill against a single-card oracle. The TP all-reduce
is implemented in torch as a sum-across-ranks. The harness reports
the worst-case per-rank pass rate against the oracle.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import zero_centered_rmsnorm_apply
from .config import (
    BATCH,
    BATCH_TILE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    FINAL_RMS_K_CHUNK,
    HEAD_DIM,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    HIDDEN_Q_SWA_LOCAL,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LAYER_TYPE_FULL,
    LAYER_TYPES,
    LM_HEAD_K_CHUNK,
    MAX_SEQ_DEFAULT,
    MLP_OUT_CHUNK,
    MOE_INTERMEDIATE,
    MOE_LAYER_INDICES,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    MOE_TOP_K,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_FULL_LOCAL_PAD,
    NUM_HEADS_SWA_LOCAL,
    NUM_HEADS_SWA_LOCAL_PAD,
    NUM_HIDDEN_LAYERS,
    ROPE_SEQ_DYN,
    SHARE_EXPERT_DIM_LOCAL,
    SWIGLU_LIMITS,
    SWIGLU_LIMITS_SHARED,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    VOCAB_CHUNK,
    VOCAB_LOCAL,
    is_full_attention,
    is_moe_layer,
)
from .prefill_attention_full import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_FULL,
    attention_full_prefill,
)
from .prefill_attention_swa import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_SWA,
    attention_swa_prefill,
)
from .dispatch import PER_RANK_BUCKETS
from .prefill_moe import select_prefill_moe_block
from .prefill_qkv_proj_rope import PREFILL_BATCH, PREFILL_SEQ, PREFILL_T, TOK_TILE
from .rms_lm_head import rms_lm_head


# -----------------------------------------------------------------------------
# Compile-time tables — mirror decode_fwd.py.
# -----------------------------------------------------------------------------
NUM_FULL_LAYERS = sum(
    1 for t in LAYER_TYPES[:NUM_HIDDEN_LAYERS] if t == LAYER_TYPE_FULL
)
NUM_SWA_LAYERS = NUM_HIDDEN_LAYERS - NUM_FULL_LAYERS
NUM_DENSE_LAYERS = NUM_HIDDEN_LAYERS - len(MOE_LAYER_INDICES)
NUM_MOE_LAYERS = len(MOE_LAYER_INDICES)


def _build_pos_tables():
    full_pos = [-1] * NUM_HIDDEN_LAYERS
    swa_pos = [-1] * NUM_HIDDEN_LAYERS
    f, s = 0, 0
    for i in range(NUM_HIDDEN_LAYERS):
        if is_full_attention(i):
            full_pos[i] = f
            f += 1
        else:
            swa_pos[i] = s
            s += 1
    return tuple(full_pos), tuple(swa_pos)


FULL_POS, SWA_POS = _build_pos_tables()


def _build_dense_moe_tables():
    dense_pos = [-1] * NUM_HIDDEN_LAYERS
    moe_pos = [-1] * NUM_HIDDEN_LAYERS
    d, m = 0, 0
    for i in range(NUM_HIDDEN_LAYERS):
        if is_moe_layer(i):
            moe_pos[i] = m
            m += 1
        else:
            dense_pos[i] = d
            d += 1
    return tuple(dense_pos), tuple(moe_pos)


DENSE_POS, MOE_POS = _build_dense_moe_tables()


# Window-pool sizing constants (referenced by host_orch).
N_RANKS = TP_WORLD_SIZE
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
TP_CHUNK = HIDDEN // TP_WORLD_SIZE
LOCAL_RECV_MAX = 1024
N_ROUTES_PER_RANK = BATCH * MOE_TOP_K
SH_INTER_LOCAL = SHARE_EXPERT_DIM_LOCAL
INTER = MOE_INTERMEDIATE
INTER_LOCAL = INTERMEDIATE_LOCAL
N_EXPERTS = MOE_NUM_EXPERTS
TOPK = MOE_TOP_K

# Prefill-T tile count for the inlined EpTpMoE adapter (mirrors prefill_moe.py).
# Each MoE-layer chip_orch slices its [PREFILL_T, HIDDEN] post_norm into
# PREFILL_TILE_COUNT independent [BATCH, HIDDEN] tiles and runs the inlined
# gate / dispatch / expert_routed / expert_shared / combine pipeline once
# per tile.
PREFILL_TILE_COUNT = PREFILL_T // BATCH
assert PREFILL_T % BATCH == 0, (
    f"PREFILL_T={PREFILL_T} must be a multiple of BATCH={BATCH} so the "
    "inlined prefill-T MoE adapter can chunk into whole decode-T tiles"
)


# -----------------------------------------------------------------------------
# Phase X.10 — kernel-internal constants for the inlined MoE method bodies.
#
# pypto frontend rejects ``self._embedded_moe_cls().chip_orch(...)``
# (instantiating a ``@pl.program`` inside another ``@pl.program`` body is not
# a supported feature), so the entire body of ``EpTpMoE`` (every
# ``@pl.function`` method plus the ``chip_orch`` per-tile loop) is inlined
# directly into ``PrefillLayerMoE``. The originals in ``moe.py`` /
# ``prefill_moe.py`` remain intact (their standalone harnesses still drive
# them as separate ``@pl.program`` instances).
# -----------------------------------------------------------------------------

# Router (gate) kernel constants — mirrors gate.py / moe.ROUTER_*.
ROUTER_SCORE_PAD = 512
ROUTER_TOPK_PAD = 16
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2
ROUTER_GATE_K_CHUNK = 512
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = 3.0  # MOE_ROUTER_SCALING_FACTOR
assert TOPK <= ROUTER_TOPK_PAD
assert HIDDEN % ROUTER_GATE_K_CHUNK == 0

# Routed-expert kernel constants — mirrors expert_routed.py / moe.ROUTED_*.
ROUTED_GATE_K_CHUNK = 256
ROUTED_GATE_N_CHUNK = 256
ROUTED_DOWN_K_CHUNK = 256
ROUTED_DOWN_N_CHUNK = 256
ROUTED_MAX_TILE = LOCAL_RECV_MAX
assert HIDDEN % ROUTED_GATE_K_CHUNK == 0
assert HIDDEN % ROUTED_DOWN_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_GATE_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_DOWN_K_CHUNK == 0

# Shared-expert kernel constants — mirrors expert_shared.py / moe.SHARED_*.
SHARED_GATE_K_CHUNK = 256
SHARED_GATE_N_CHUNK = SH_INTER_LOCAL  # 160 — one N tile covers the slice
SHARED_DOWN_K_CHUNK = SH_INTER_LOCAL  # 160 — one K tile covers the slice
SHARED_DOWN_N_CHUNK = 256
assert HIDDEN % SHARED_GATE_K_CHUNK == 0
assert HIDDEN % SHARED_DOWN_N_CHUNK == 0


# =============================================================================
# Prefill dense-MLP body — TP-sliced gate/up/down + tp_all_reduce.
#
# Sequence-major counterpart of ``decode_layer._dense_mlp_body_tp``.
# Each rank holds INTER_LOCAL=1408 lanes of w_gate/w_up and the matching
# row slab of w_down; tp_all_reduce sums the per-rank [PREFILL_T, HIDDEN]
# partials into a uniform reduced output, then the residual is added.
# =============================================================================
@pl.jit.inline
def _prefill_dense_mlp_body_tp(
    resid1: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE_LOCAL], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE_LOCAL], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[[PREFILL_T, HIDDEN // TP_WORLD_SIZE], pl.BF16],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Post-attention zero-centred RMSNorm + TP-sliced SwiGLU MLP + residual."""
    hidden_blocks = HIDDEN // K_CHUNK
    mlp_out_blocks = INTERMEDIATE_LOCAL // MLP_OUT_CHUNK
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE_LOCAL

    # ── Step 1: post-attention zero-centred RMSNorm. ──────────────────
    # Token-tiled (DeepSeek-V4 style, cf. decode_rmsnorm.attn_norm): process
    # BATCH rows at a time so the per-scope Vec working set fits the 192 KB UB.
    # The staging tensors stay full [PREFILL_T, HIDDEN]; only the Vec tiles are
    # [BATCH, K_CHUNK].
    post_norm = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
    resid1_fp32 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_dense_post_rmsnorm_zc",
    ):
        for tg in pl.range(PREFILL_TILE_COUNT):
            t0 = tg * BATCH
            sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
            for kb in pl.range(hidden_blocks):
                k0 = kb * K_CHUNK
                rchunk = pl.cast(
                    pl.slice(resid1, [BATCH, K_CHUNK], [t0, k0]),
                    target_type=pl.FP32,
                )
                resid1_fp32 = pl.assemble(resid1_fp32, rchunk, [t0, k0])
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(rchunk, rchunk)), [1, BATCH]),
                )
            inv_rms_dense = pl.recip(
                pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
            )
            inv_rms_col = pl.reshape(inv_rms_dense, [BATCH, 1])
            for kb3 in pl.range(hidden_blocks):
                k0 = kb3 * K_CHUNK
                norm_chunk = pl.slice(
                    resid1_fp32, [BATCH, K_CHUNK], [t0, k0],
                )
                gamma = pl.slice(
                    post_rms_weight, [1, K_CHUNK], [layer_idx, k0],
                )
                scaled = pl.row_expand_mul(norm_chunk, inv_rms_col)
                normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
                post_norm = pl.assemble(
                    post_norm,
                    pl.cast(normed, target_type=pl.BF16),
                    [t0, k0],
                )

    # ── Step 2: TP-sliced gate_up + SiLU. ─────────────────────────────
    mlp_tile = pl.create_tensor([PREFILL_T, INTERMEDIATE_LOCAL], dtype=pl.BF16)
    for ob in pl.spmd(
        mlp_out_blocks, name_hint="prefill_dense_gate_up_silu_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        mlp_o0 = ob * MLP_OUT_CHUNK
        # Token-tiled: matmul M = BATCH so the FP32 SwiGLU accumulators fit UB.
        for tg in pl.range(PREFILL_TILE_COUNT):
            t0 = tg * BATCH
            post_chunk_0 = pl.slice(post_norm, [BATCH, K_CHUNK], [t0, 0])
            wg_0 = pl.slice(
                w_gate, [K_CHUNK, MLP_OUT_CHUNK],
                [layer_hidden_base, mlp_o0],
            )
            wu_0 = pl.slice(
                w_up, [K_CHUNK, MLP_OUT_CHUNK],
                [layer_hidden_base, mlp_o0],
            )
            gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
            up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
            for kb in pl.range(1, hidden_blocks):
                k0 = kb * K_CHUNK
                post_chunk = pl.slice(
                    post_norm, [BATCH, K_CHUNK], [t0, k0],
                )
                wg = pl.slice(
                    w_gate, [K_CHUNK, MLP_OUT_CHUNK],
                    [layer_hidden_base + k0, mlp_o0],
                )
                wu = pl.slice(
                    w_up, [K_CHUNK, MLP_OUT_CHUNK],
                    [layer_hidden_base + k0, mlp_o0],
                )
                gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
                up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
            mlp_tile = pl.assemble(
                mlp_tile,
                pl.cast(mlp_chunk, target_type=pl.BF16),
                [t0, mlp_o0],
            )

    # ── Step 3: TP-sliced w_down -> partial [PREFILL_T, HIDDEN]. ──────
    partial_hidden = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
    for dob in pl.spmd(
        hidden_blocks, name_hint="prefill_dense_down_proj_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        d0 = dob * K_CHUNK
        # Token-tiled: matmul M = BATCH so the FP32 down-proj accumulator fits UB.
        for tg in pl.range(PREFILL_TILE_COUNT):
            t0 = tg * BATCH
            mlp_chunk_0 = pl.slice(mlp_tile, [BATCH, MLP_OUT_CHUNK], [t0, 0])
            w_down_chunk_0 = pl.slice(
                w_down, [MLP_OUT_CHUNK, K_CHUNK],
                [layer_inter_base, d0],
            )
            down_acc = pl.matmul(
                mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32,
            )
            for ob in pl.range(1, INTERMEDIATE_LOCAL // MLP_OUT_CHUNK):
                down_o0 = ob * MLP_OUT_CHUNK
                # Distinct name: this is a BF16 slice of staged `mlp_tile` (the
                # earlier `mlp_chunk` is the FP32 SwiGLU product); pypto's strict
                # SSA refuses the type-changing reassignment (matching the
                # decode-side `down_mlp_chunk_bf16` convention).
                mlp_chunk_bf16 = pl.slice(
                    mlp_tile, [BATCH, MLP_OUT_CHUNK], [t0, down_o0],
                )
                w_down_chunk = pl.slice(
                    w_down, [MLP_OUT_CHUNK, K_CHUNK],
                    [layer_inter_base + down_o0, d0],
                )
                down_acc = pl.matmul_acc(
                    down_acc, mlp_chunk_bf16, w_down_chunk,
                )
            partial_hidden = pl.assemble(
                partial_hidden,
                pl.cast(down_acc, target_type=pl.BF16),
                [t0, d0],
            )

    # ── Step 4: TP all-reduce. ────────────────────────────────────────
    # Phase X.2: ``self.tp_all_reduce`` resolves to a method on the
    # enclosing @pl.program class (PrefillLayerDense / PrefillLayerMoE).
    self.tp_all_reduce(
        partial_hidden, tmp_window, signal_window, my_rank,
    )

    # ── Step 5: residual add. ─────────────────────────────────────────
    # Token-tiled to match Step 1 (BATCH rows per Vec iteration).
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_dense_residual_add_tp",
    ):
        for tg5 in pl.range(PREFILL_TILE_COUNT):
            t0 = tg5 * BATCH
            for kb4 in pl.range(hidden_blocks):
                k0 = kb4 * K_CHUNK
                # NOTE: the attention pl.inline body already binds `reduced` to
                # a different shape ([TOK_TILE, 256] FP32) earlier in the same
                # @pl.function scope; pypto's strict SSA refuses the conflicting
                # rebind. Use a body-unique name for the dense-MLP residual.
                mlp_reduced = pl.cast(
                    pl.slice(partial_hidden, [BATCH, K_CHUNK], [t0, k0]),
                    target_type=pl.FP32,
                )
                r = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [t0, k0])
                next_hidden = pl.assemble(
                    next_hidden,
                    pl.cast(pl.add(r, mlp_reduced), target_type=pl.BF16),
                    [t0, k0],
                )

    return next_hidden


# =============================================================================
# Per-layer @pl.program builders — analogous to decode_layer's builders.
# =============================================================================
def _build_prefill_layer_dense_program(
    *, full: bool, tp_size: int = TP_WORLD_SIZE,
):
    """Return ``@pl.program`` for attention + TP dense MLP (prefill T)."""
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    attention_inline = (
        pl.inline(attention_full_prefill._func)
        if full
        else pl.inline(attention_swa_prefill._func)
    )
    dense_inline = pl.inline(_prefill_dense_mlp_body_tp._func)
    tp_chunk = HIDDEN // tp_size

    rotary_dim = 64 if full else 128
    hidden_q_local = HIDDEN_Q_FULL_LOCAL if full else HIDDEN_Q_SWA_LOCAL
    num_heads_local = NUM_HEADS_FULL_LOCAL if full else NUM_HEADS_SWA_LOCAL
    num_heads_local_pad = (
        NUM_HEADS_FULL_LOCAL_PAD if full else NUM_HEADS_SWA_LOCAL_PAD
    )
    layer_qhidden_dyn = (
        LAYER_QHIDDEN_ROWS_DYN_FULL if full else LAYER_QHIDDEN_ROWS_DYN_SWA
    )

    @pl.program
    class PrefillLayerDense:
        # ---------- Collective: TP all_reduce (lifted from collectives.py) ----
        # Phase X.2: pull-side ring body, t_rows=PREFILL_T, d_cols=HIDDEN,
        # group_size=tp_size baked from factory closure.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            group_size = tp_size
            # Inline shape constants — pypto's tile shape inference cannot follow
            # Python-level aliases like ``t_rows = PREFILL_T`` past load/remote_load
            # boundaries (it preserves the alias name in the tile type, which
            # then mismatches the concrete shape from sibling pl.load calls).
            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + group_size) % group_size
                recv_idx = (my_rank - step - 1 + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                # Token-tiled Vec I/O (BATCH rows per sub-tile) so the per-step
                # working set fits UB; the ring comm stays whole-window with one
                # notify/wait per step (signal counts unchanged).
                for tt in pl.range(PREFILL_TILE_COUNT):
                    ttr = tt * BATCH
                    send_tile = pl.load(
                        local, [ttr, send_idx * tp_chunk], [BATCH, tp_chunk],
                    )
                    pl.store(send_tile, [ttr, 0], tmp_window)
                pld.system.notify(
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window, offsets=[prev_rank, 0],
                    expected=pl.cast(step + 1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )
                for tt in pl.range(PREFILL_TILE_COUNT):
                    ttr = tt * BATCH
                    recv_tile = pld.tile.remote_load(
                        tmp_window, peer=prev_rank,
                        offsets=[ttr, 0], shape=[BATCH, tp_chunk],
                    )
                    old_tile = pl.load(
                        local, [ttr, recv_idx * tp_chunk], [BATCH, tp_chunk],
                    )
                    # PTOAS A2/A3 ``tadd`` doesn't support bf16; upcast to f32,
                    # add, then downcast for the store.
                    summed_fp32 = pl.add(
                        pl.cast(old_tile, target_type=pl.FP32),
                        pl.cast(recv_tile, target_type=pl.FP32),
                    )
                    pl.store(
                        pl.cast(summed_fp32, target_type=pl.BF16),
                        [ttr, recv_idx * tp_chunk], local,
                    )

            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + 1 + group_size) % group_size
                recv_idx = (my_rank - step + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                for tt in pl.range(PREFILL_TILE_COUNT):
                    ttr = tt * BATCH
                    send_tile = pl.load(
                        local, [ttr, send_idx * tp_chunk], [BATCH, tp_chunk],
                    )
                    pl.store(send_tile, [ttr, 0], tmp_window)
                pld.system.notify(
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window, offsets=[prev_rank, 0],
                    expected=pl.cast(group_size - 1 + step + 1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
                for tt in pl.range(PREFILL_TILE_COUNT):
                    ttr = tt * BATCH
                    recv_tile = pld.tile.remote_load(
                        tmp_window, peer=prev_rank,
                        offsets=[ttr, 0], shape=[BATCH, tp_chunk],
                    )
                    pl.store(recv_tile, [ttr, recv_idx * tp_chunk], local)
            return local

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16
            ],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[layer_qhidden_dyn, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            w_up: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            w_down: pl.Tensor[
                [LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16
            ],
            next_hidden_out: pl.Out[
                pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]
            ],
            attn_tmp_window: pld.DistributedTensor[
                [PREFILL_T, tp_chunk], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            mlp_tmp_window: pld.DistributedTensor[
                [PREFILL_T, tp_chunk], pl.BF16
            ],
            mlp_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            resid1 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            resid1 = attention_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                positions,
                resid1,
                layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            next_hidden_out = dense_inline(
                resid1, post_rms_weight,
                w_gate, w_up, w_down,
                next_hidden_out, layer_idx,
                mlp_tmp_window, mlp_signal_window, my_rank,
            )
            return next_hidden_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[
                [tp_size, PREFILL_T, HIDDEN], pl.BF16
            ],
            input_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16
            ],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            k_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            block_table: pl.Tensor[
                [tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32
            ],
            slot_mapping: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32
            ],
            rope_sin: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32
            ],
            k_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            wo: pl.Tensor[
                [tp_size, layer_qhidden_dyn, HIDDEN], pl.BF16
            ],
            w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16
            ],
            positions: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            w_gate: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            w_up: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            w_down: pl.Tensor[
                [tp_size, LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16
            ],
            next_hidden_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]
            ],
            layer_idx: pl.Scalar[pl.INT32],
        ):
            attn_tmp_buf = pld.alloc_window_buffer(PREFILL_T * tp_chunk * 2)
            attn_sig_buf = pld.alloc_window_buffer(tp_size * 4)
            mlp_tmp_buf = pld.alloc_window_buffer(PREFILL_T * tp_chunk * 2)
            mlp_sig_buf = pld.alloc_window_buffer(tp_size * 4)
            for r in pl.range(pld.world_size()):
                attn_tmp_window = pld.window(
                    attn_tmp_buf, [PREFILL_T, tp_chunk], dtype=pl.BF16,
                )
                attn_signal_window = pld.window(
                    attn_sig_buf, [tp_size, 1], dtype=pl.INT32,
                )
                mlp_tmp_window = pld.window(
                    mlp_tmp_buf, [PREFILL_T, tp_chunk], dtype=pl.BF16,
                )
                mlp_signal_window = pld.window(
                    mlp_sig_buf, [tp_size, 1], dtype=pl.INT32,
                )
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r],
                    k_cache[r], v_cache[r],
                    wo[r], w_g[r],
                    positions[r],
                    post_rms_weight[r],
                    w_gate[r], w_up[r], w_down[r],
                    next_hidden_out[r],
                    attn_tmp_window, attn_signal_window,
                    mlp_tmp_window, mlp_signal_window,
                    layer_idx,
                    r,
                    device=r,
                )

    return PrefillLayerDense


def _build_prefill_layer_moe_program(
    *, full: bool, routed_lim: float, shared_lim: float,
    tp_size: int = TP_WORLD_SIZE,
):
    """Return ``@pl.program`` for attention + EP+TP MoE (prefill T).

    Phase X.10: the entire ``EpTpMoE`` body (every ``@pl.function`` method
    plus the prefill-T tile-by-tile ``chip_orch`` loop) is inlined into
    ``PrefillLayerMoE``. The frontend rejects
    ``self._embedded_moe_cls().chip_orch(...)`` (instantiating a
    ``@pl.program`` inside another ``@pl.program`` body is not supported),
    so the activation choice is baked at factory build time via Python
    closure constants (``_routed_swiglu_step`` / ``_shared_swiglu_step``)
    rather than via a factory call to a separate program class.
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
        )
    attention_inline = (
        pl.inline(attention_full_prefill._func)
        if full
        else pl.inline(attention_swa_prefill._func)
    )

    # Activation choice — compile-time Python constants captured in closure.
    if routed_lim == 0.0:
        _routed_swiglu_step = False
    elif routed_lim == 7.0:
        _routed_swiglu_step = True
    else:
        raise ValueError(
            f"routed_lim must be 0.0 or 7.0, got {routed_lim}",
        )

    if shared_lim == 0.0:
        _shared_swiglu_step = False
    elif shared_lim == 16.0:
        _shared_swiglu_step = True
    else:
        raise ValueError(
            f"shared_lim must be 0.0 or 16.0, got {shared_lim}",
        )

    _routed_swiglu_limit = routed_lim
    _shared_swiglu_limit = shared_lim
    tp_chunk = HIDDEN // tp_size

    rotary_dim = 64 if full else 128
    hidden_q_local = HIDDEN_Q_FULL_LOCAL if full else HIDDEN_Q_SWA_LOCAL
    num_heads_local = NUM_HEADS_FULL_LOCAL if full else NUM_HEADS_SWA_LOCAL
    num_heads_local_pad = (
        NUM_HEADS_FULL_LOCAL_PAD if full else NUM_HEADS_SWA_LOCAL_PAD
    )
    layer_qhidden_dyn = (
        LAYER_QHIDDEN_ROWS_DYN_FULL if full else LAYER_QHIDDEN_ROWS_DYN_SWA
    )

    n_ranks = tp_size
    n_local_experts = N_LOCAL_EXPERTS
    inter = INTER
    sh_inter_local = SHARE_EXPERT_DIM_LOCAL
    sh_tp_chunk = HIDDEN // tp_size
    local_recv_max = LOCAL_RECV_MAX
    n_routes_per_rank = N_ROUTES_PER_RANK
    topk = TOPK
    per_rank_buckets = PER_RANK_BUCKETS

    @pl.program
    class PrefillLayerMoE:
        # Phase X.10: the entire ``EpTpMoE`` body (gate / dispatch /
        # expert_routed / expert_shared / combine plus all Inline helpers and
        # the prefill-T tile-by-tile chip_orch loop) is inlined into this
        # class. The frontend rejects ``self._embedded_moe_cls().chip_orch(...)``
        # (instantiating a ``@pl.program`` inside another ``@pl.program``
        # body is not supported). The originals in ``moe.py`` /
        # ``prefill_moe.py`` remain intact for their standalone harnesses;
        # the bodies here are kept in lock-step with them.

        # ---------- Collective: TP all_reduce (lifted from collectives.py) ----
        # Phase X.2: pull-side ring body, t_rows=PREFILL_T, d_cols=HIDDEN,
        # group_size=tp_size baked from factory closure.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[PREFILL_T, tp_chunk], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]:
            group_size = tp_size
            # Inline shape constants — pypto's tile shape inference cannot follow
            # Python-level aliases like ``t_rows = PREFILL_T`` past load/remote_load
            # boundaries (it preserves the alias name in the tile type, which
            # then mismatches the concrete shape from sibling pl.load calls).
            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + group_size) % group_size
                recv_idx = (my_rank - step - 1 + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                # Token-tiled Vec I/O (BATCH rows per sub-tile) so the per-step
                # working set fits UB; the ring comm stays whole-window with one
                # notify/wait per step (signal counts unchanged).
                for tt in pl.range(PREFILL_TILE_COUNT):
                    ttr = tt * BATCH
                    send_tile = pl.load(
                        local, [ttr, send_idx * tp_chunk], [BATCH, tp_chunk],
                    )
                    pl.store(send_tile, [ttr, 0], tmp_window)
                pld.system.notify(
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window, offsets=[prev_rank, 0],
                    expected=pl.cast(step + 1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )
                for tt in pl.range(PREFILL_TILE_COUNT):
                    ttr = tt * BATCH
                    recv_tile = pld.tile.remote_load(
                        tmp_window, peer=prev_rank,
                        offsets=[ttr, 0], shape=[BATCH, tp_chunk],
                    )
                    old_tile = pl.load(
                        local, [ttr, recv_idx * tp_chunk], [BATCH, tp_chunk],
                    )
                    # PTOAS A2/A3 ``tadd`` doesn't support bf16; upcast to f32,
                    # add, then downcast for the store.
                    summed_fp32 = pl.add(
                        pl.cast(old_tile, target_type=pl.FP32),
                        pl.cast(recv_tile, target_type=pl.FP32),
                    )
                    pl.store(
                        pl.cast(summed_fp32, target_type=pl.BF16),
                        [ttr, recv_idx * tp_chunk], local,
                    )

            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + 1 + group_size) % group_size
                recv_idx = (my_rank - step + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                for tt in pl.range(PREFILL_TILE_COUNT):
                    ttr = tt * BATCH
                    send_tile = pl.load(
                        local, [ttr, send_idx * tp_chunk], [BATCH, tp_chunk],
                    )
                    pl.store(send_tile, [ttr, 0], tmp_window)
                pld.system.notify(
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window, offsets=[prev_rank, 0],
                    expected=pl.cast(group_size - 1 + step + 1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
                for tt in pl.range(PREFILL_TILE_COUNT):
                    ttr = tt * BATCH
                    recv_tile = pld.tile.remote_load(
                        tmp_window, peer=prev_rank,
                        offsets=[ttr, 0], shape=[BATCH, tp_chunk],
                    )
                    pl.store(recv_tile, [ttr, recv_idx * tp_chunk], local)
            return local

        # ===================================================================
        # Phase X.10 — inlined EpTpMoE @pl.function methods.
        # Bodies copied verbatim from ``prefill_moe.PrefillMoE`` (which in
        # turn mirrors ``moe.EpTpMoE``). The activation choice (plain SiLU
        # vs. SwigluStep@7/16) is baked at factory build time via the
        # ``_routed_swiglu_step`` / ``_shared_swiglu_step`` Python closure
        # constants — only one branch is emitted per specialisation. The
        # per-tile token count is BATCH (= decode-T); ``chip_orch`` below
        # drives PREFILL_TILE_COUNT independent tile invocations to cover
        # the full PREFILL_T axis.
        # ===================================================================

        # ---------- Per-tile TP all_reduce (BATCH rows, used by shared lane) ----
        # NOTE: distinct from the outer PREFILL_T-sized ``tp_all_reduce`` above —
        # the shared-expert lane operates on per-tile [BATCH, HIDDEN] tiles, so
        # we use a dedicated tile-sized ring body. Same algorithm, baked
        # t_rows=BATCH.
        @pl.function(type=pl.FunctionType.InCore)
        def _tp_all_reduce_moe(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, sh_tp_chunk], pl.BF16],
            signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            group_size = n_ranks
            t_rows = BATCH
            d_cols = HIDDEN
            chunk = d_cols // group_size

            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + group_size) % group_size
                recv_idx = (my_rank - step - 1 + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                send_tile = pl.load(
                    local, [0, send_idx * chunk], [t_rows, chunk],
                )
                pl.store(send_tile, [0, 0], tmp_window)
                pld.system.notify(
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window, offsets=[prev_rank, 0],
                    expected=step + 1, cmp=pld.WaitCmp.Ge,
                )
                recv_tile = pld.tile.remote_load(
                    tmp_window, peer=prev_rank,
                    offsets=[0, 0], shape=[t_rows, chunk],
                )
                old_tile = pl.load(
                    local, [0, recv_idx * chunk], [t_rows, chunk],
                )
                pl.store(
                    pl.add(old_tile, recv_tile),
                    [0, recv_idx * chunk], local,
                )

            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + 1 + group_size) % group_size
                recv_idx = (my_rank - step + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                send_tile = pl.load(
                    local, [0, send_idx * chunk], [t_rows, chunk],
                )
                pl.store(send_tile, [0, 0], tmp_window)
                pld.system.notify(
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window, offsets=[prev_rank, 0],
                    expected=pl.cast(group_size - 1 + step + 1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
                recv_tile = pld.tile.remote_load(
                    tmp_window, peer=prev_rank,
                    offsets=[0, 0], shape=[t_rows, chunk],
                )
                pl.store(recv_tile, [0, recv_idx * chunk], local)
            return local

        # ---------- Collective: EP all_to_all ----------
        @pl.function(type=pl.FunctionType.InCore)
        def ep_all_to_all(
            self,
            send: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            recv: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            send_counts: pl.Tensor[[n_ranks], pl.INT32],
            recv_counts: pl.Tensor[[n_ranks], pl.INT32],
            send_offsets: pl.Tensor[[n_ranks], pl.INT32],
            recv_offsets: pl.Tensor[[n_ranks], pl.INT32],
            signal_window: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16]:
            """Pull-side variable-length token-level all-to-all over EP."""
            group_size = n_ranks
            d_cols = HIDDEN

            n_self = pl.cast(pl.read(send_counts, [my_rank]), pl.INDEX)
            s_off_self = pl.cast(pl.read(send_offsets, [my_rank]), pl.INDEX)
            r_off_self = pl.cast(pl.read(recv_offsets, [my_rank]), pl.INDEX)
            for r in pl.range(n_self):
                self_tile = pl.load(
                    send, [s_off_self + r, 0], [1, d_cols],
                )
                pl.store(self_tile, [r_off_self + r, 0], recv)

            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )

            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            for peer in pl.range(group_size):
                if peer != my_rank:
                    n_recv = pl.cast(
                        pl.read(recv_counts, [peer]), pl.INDEX,
                    )
                    r_off = pl.cast(
                        pl.read(recv_offsets, [peer]), pl.INDEX,
                    )
                    for r in pl.range(n_recv):
                        peer_tile = pld.tile.remote_load(
                            send,
                            peer=peer,
                            offsets=[r_off + r, 0],
                            shape=[1, d_cols],
                        )
                        pl.store(peer_tile, [r_off + r, 0], recv)

            return recv

        # ---------- Stage 1: gate (local, replicated) ----------
        @pl.function(type=pl.FunctionType.Inline)
        def _gate(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.BF16],
        ):
            score_buf = pl.create_tensor(
                [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
            )
            biased_buf = pl.create_tensor(
                [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32,
            )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_matmul"):
                x_fp32 = pl.cast(x, target_type=pl.FP32)
                x0 = pl.slice(x_fp32, [BATCH, ROUTER_GATE_K_CHUNK], [0, 0])
                w0 = pl.slice(
                    gate_w, [ROUTER_GATE_K_CHUNK, N_EXPERTS], [0, 0],
                )
                logits = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN // ROUTER_GATE_K_CHUNK):
                    k0 = kb * ROUTER_GATE_K_CHUNK
                    xk = pl.slice(
                        x_fp32, [BATCH, ROUTER_GATE_K_CHUNK], [0, k0],
                    )
                    wk = pl.slice(
                        gate_w, [ROUTER_GATE_K_CHUNK, N_EXPERTS], [k0, 0],
                    )
                    logits = pl.matmul_acc(logits, xk, wk)

                score_n = pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0))
                bias_row = pl.reshape(router_bias, [1, N_EXPERTS])
                biased_n = pl.add(
                    score_n,
                    pl.col_expand_mul(
                        pl.full(
                            [BATCH, N_EXPERTS], dtype=pl.FP32, value=1.0,
                        ),
                        bias_row,
                    ),
                )

                score_buf[:, :] = pl.full(
                    [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32, value=0.0,
                )
                biased_buf[:, :] = pl.full(
                    [BATCH, ROUTER_SCORE_PAD],
                    dtype=pl.FP32, value=ROUTER_FP32_NEG_INF,
                )
                score_buf[:, 0:N_EXPERTS] = score_n
                biased_buf[:, 0:N_EXPERTS] = biased_n

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_topk"):
                topk_idx_tile = pl.create_tensor(
                    [BATCH, ROUTER_TOPK_PAD], dtype=pl.INT32,
                )
                for tt in pl.range(BATCH):
                    row = biased_buf[tt : tt + 1, :]
                    idx_init = pl.arange(
                        0, [1, ROUTER_SCORE_PAD], dtype=pl.UINT32,
                    )
                    srt = pl.sort32(row, idx_init)
                    srt = pl.mrgsort(srt, block_len=64)
                    srt = pl.mrgsort(srt[:, 0:512], srt[:, 512:1024])
                    pairs = srt[:, 0:ROUTER_SORT_PAD]
                    top_idx = pl.gather(
                        pairs, mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    topk_idx_tile[tt : tt + 1, :] = top_idx

                gather_all = pl.gather(
                    score_buf, dim=-1, index=topk_idx_tile,
                )
                gather_valid = pl.set_validshape(gather_all, BATCH, TOPK)
                topk_vals_pad = pl.fillpad(
                    gather_valid, pad_value=pl.PadValue.zero,
                )

                denom = pl.reshape(pl.row_sum(topk_vals_pad), [BATCH, 1])
                weights_pad = pl.mul(
                    pl.row_expand_div(topk_vals_pad, denom),
                    ROUTER_SCALE,
                )

                for tt in pl.range(BATCH):
                    for k in pl.range(TOPK):
                        pl.write(
                            expert_indices, [tt, k],
                            pl.read(topk_idx_tile, [tt, k]),
                        )
                        pl.write(
                            expert_weights, [tt, k],
                            pl.cast(
                                pl.read(weights_pad, [tt, k]), pl.BF16,
                            ),
                        )

            return expert_weights

        @pl.function(type=pl.FunctionType.Inline)
        def gate_step(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            expert_indices: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
            expert_weights: pl.Out[pl.Tensor[[BATCH, TOPK], pl.BF16]],
        ) -> tuple[
            pl.Tensor[[BATCH, TOPK], pl.INT32],
            pl.Tensor[[BATCH, TOPK], pl.BF16]
        ]:
            self._gate(
                x, gate_w, router_bias, expert_indices, expert_weights,
            )
            return expert_indices, expert_weights

        # ---------- Stage 2: dispatch (EP all-to-all) ----------
        @pl.function(type=pl.FunctionType.Inline)
        def _histogram_and_prefix_sum(
            self,
            indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            send_counts_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
            send_counts_per_rank: pl.Tensor[[n_ranks], pl.INT32],
            send_offsets_per_rank: pl.Tensor[[n_ranks], pl.INT32],
        ):
            for bkt in pl.range(per_rank_buckets):
                pl.write(
                    send_counts_per_bucket, [bkt], pl.cast(0, pl.INT32),
                )
            for r in pl.range(n_ranks):
                pl.write(
                    send_counts_per_rank, [r], pl.cast(0, pl.INT32),
                )

            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e
                    cur = pl.read(send_counts_per_bucket, [bkt])
                    pl.write(
                        send_counts_per_bucket, [bkt],
                        pl.cast(cur + 1, pl.INT32),
                    )
                    r_cur = pl.read(send_counts_per_rank, [dst])
                    pl.write(
                        send_counts_per_rank, [dst],
                        pl.cast(r_cur + 1, pl.INT32),
                    )

            pl.write(send_offsets_per_rank, [0], pl.cast(0, pl.INT32))
            for r in pl.range(1, n_ranks):
                prev_off = pl.read(send_offsets_per_rank, [r - 1])
                prev_cnt = pl.read(send_counts_per_rank, [r - 1])
                pl.write(
                    send_offsets_per_rank, [r],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

        @pl.function(type=pl.FunctionType.Inline)
        def _pack_send_payload(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            send_counts_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
            send_offsets_per_rank: pl.Tensor[[n_ranks], pl.INT32],
            send_buf: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
            cursor_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
            bucket_offset: pl.Tensor[[per_rank_buckets], pl.INT32],
        ):
            for r in pl.range(n_ranks):
                rank_off = pl.read(send_offsets_per_rank, [r])
                pl.write(
                    bucket_offset, [r * n_local_experts],
                    pl.cast(rank_off, pl.INT32),
                )
                pl.write(
                    cursor_per_bucket, [r * n_local_experts],
                    pl.cast(rank_off, pl.INT32),
                )
                for e in pl.range(1, n_local_experts):
                    prev_off = pl.read(
                        bucket_offset, [r * n_local_experts + e - 1],
                    )
                    prev_cnt = pl.read(
                        send_counts_per_bucket,
                        [r * n_local_experts + e - 1],
                    )
                    new_off = pl.cast(prev_off + prev_cnt, pl.INT32)
                    pl.write(
                        bucket_offset, [r * n_local_experts + e], new_off,
                    )
                    pl.write(
                        cursor_per_bucket, [r * n_local_experts + e],
                        new_off,
                    )

            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e
                    slot_i32 = pl.read(cursor_per_bucket, [bkt])
                    slot = pl.cast(slot_i32, pl.INDEX)
                    send_buf = pl.assemble(
                        send_buf,
                        pl.slice(x, [1, HIDDEN], [t, 0]),
                        [slot, 0],
                    )
                    pl.write(
                        cursor_per_bucket, [bkt],
                        pl.cast(slot_i32 + 1, pl.INT32),
                    )

            return send_buf

        @pl.function(type=pl.FunctionType.Inline)
        def _build_local_expert_csr(
            self,
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts], pl.INT32
            ],
            local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
            local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            for e in pl.range(n_local_experts):
                acc = pl.cast(0, pl.INT32)
                for s in pl.range(n_ranks):
                    acc = acc + pl.read(
                        pub_counts, [s * n_ranks + my_rank, e],
                    )
                pl.write(local_expert_count, [e], pl.cast(acc, pl.INT32))

            pl.write(local_expert_offset, [0], pl.cast(0, pl.INT32))
            for e in pl.range(1, n_local_experts):
                prev_off = pl.read(local_expert_offset, [e - 1])
                prev_cnt = pl.read(local_expert_count, [e - 1])
                pl.write(
                    local_expert_offset, [e],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

        @pl.function(type=pl.FunctionType.Inline)
        def _build_inverse_map(
            self,
            indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts], pl.INT32
            ],
            inverse_map: pl.Tensor[[BATCH, TOPK], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            cursor = pl.create_tensor(
                [n_ranks * n_local_experts], dtype=pl.INT32,
            )
            for bkt in pl.range(n_ranks * n_local_experts):
                pl.write(cursor, [bkt], pl.cast(0, pl.INT32))

            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e

                    src_off = pl.cast(0, pl.INT32)
                    for s in pl.range(n_ranks):
                        if s < my_rank:
                            src_off = src_off + pl.read(
                                pub_counts, [s * n_ranks + dst, loc_e],
                            )

                    loc_e_off = pl.cast(0, pl.INT32)
                    for prev_e in pl.range(n_local_experts):
                        if prev_e < loc_e:
                            for s in pl.range(n_ranks):
                                loc_e_off = loc_e_off + pl.read(
                                    pub_counts,
                                    [s * n_ranks + dst, prev_e],
                                )

                    my_cursor_val = pl.read(cursor, [bkt])
                    dst_row = loc_e_off + src_off + my_cursor_val
                    packed = (
                        dst * pl.cast(local_recv_max, pl.INT32) + dst_row
                    )
                    pl.write(inverse_map, [t, k], pl.cast(packed, pl.INT32))
                    pl.write(
                        cursor, [bkt],
                        pl.cast(my_cursor_val + 1, pl.INT32),
                    )

        @pl.function(type=pl.FunctionType.InCore)
        def dispatch_step(  # noqa: PLR0913
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            local_routed_x_out: pl.Out[
                pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]
            ],
            local_expert_offset: pl.Out[
                pl.Tensor[[n_local_experts], pl.INT32]
            ],
            local_expert_count: pl.Out[
                pl.Tensor[[n_local_experts], pl.INT32]
            ],
            inverse_map: pl.Out[pl.Tensor[[BATCH, TOPK], pl.INT32]],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_x: pld.DistributedTensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
            pl.Tensor[[n_local_experts], pl.INT32],
            pl.Tensor[[n_local_experts], pl.INT32],
            pl.Tensor[[BATCH, TOPK], pl.INT32]
        ]:
            send_counts_bkt = pl.create_tensor(
                [per_rank_buckets], dtype=pl.INT32,
            )
            send_counts_rank = pl.create_tensor([n_ranks], dtype=pl.INT32)
            send_offsets_rank = pl.create_tensor([n_ranks], dtype=pl.INT32)
            self._histogram_and_prefix_sum(
                expert_indices,
                send_counts_bkt, send_counts_rank, send_offsets_rank,
            )

            for peer in pl.range(n_ranks):
                for e in pl.range(n_local_experts):
                    v = pl.read(
                        send_counts_bkt, [peer * n_local_experts + e],
                    )
                    if peer == my_rank:
                        pl.write(
                            pub_counts,
                            [my_rank * n_ranks + my_rank, e],
                            v,
                        )
                    else:
                        pld.system.notify(
                            target=pub_counts,
                            peer=peer,
                            offsets=[my_rank * n_ranks + peer, e],
                            value=v,
                            op=pld.NotifyOp.Set,
                        )

            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=count_done_sig,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=count_done_sig,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            send_buf = pl.create_tensor(
                [local_recv_max, HIDDEN], dtype=pl.BF16,
            )
            cursor_bkt = pl.create_tensor(
                [per_rank_buckets], dtype=pl.INT32,
            )
            bucket_offset = pl.create_tensor(
                [per_rank_buckets], dtype=pl.INT32,
            )
            send_buf = self._pack_send_payload(
                x, expert_indices,
                send_counts_bkt, send_offsets_rank,
                send_buf, cursor_bkt, bucket_offset,
            )

            recv_counts = pl.create_tensor([n_ranks], dtype=pl.INT32)
            for src in pl.range(n_ranks):
                acc = pl.cast(0, pl.INT32)
                for e in pl.range(n_local_experts):
                    acc = acc + pl.read(
                        pub_counts, [src * n_ranks + my_rank, e],
                    )
                pl.write(recv_counts, [src], pl.cast(acc, pl.INT32))
            recv_offsets = pl.create_tensor([n_ranks], dtype=pl.INT32)
            pl.write(recv_offsets, [0], pl.cast(0, pl.INT32))
            for r in pl.range(1, n_ranks):
                prev_off = pl.read(recv_offsets, [r - 1])
                prev_cnt = pl.read(recv_counts, [r - 1])
                pl.write(
                    recv_offsets, [r],
                    pl.cast(prev_off + prev_cnt, pl.INT32),
                )

            self.ep_all_to_all(
                send_buf, recv_x,
                send_counts_rank, recv_counts,
                send_offsets_rank, recv_offsets,
                data_done_sig, my_rank,
            )

            self._build_local_expert_csr(
                pub_counts,
                local_expert_offset, local_expert_count,
                my_rank,
            )
            running = pl.cast(0, pl.INT32)
            for e in pl.range(n_local_experts):
                for src in pl.range(n_ranks):
                    n = pl.cast(
                        pl.read(pub_counts, [src * n_ranks + my_rank, e]),
                        pl.INDEX,
                    )
                    src_base = pl.cast(pl.read(recv_offsets, [src]), pl.INDEX)
                    src_e_off = pl.cast(0, pl.INT32)
                    for prev_e in pl.range(n_local_experts):
                        if prev_e < e:
                            src_e_off = src_e_off + pl.read(
                                pub_counts,
                                [src * n_ranks + my_rank, prev_e],
                            )
                    for row in pl.range(n):
                        src_row = (
                            src_base
                            + pl.cast(src_e_off, pl.INDEX) + row
                        )
                        dst_row = pl.cast(running, pl.INDEX) + row
                        tile = pl.load(recv_x, [src_row, 0], [1, HIDDEN])
                        pl.store(tile, [dst_row, 0], local_routed_x_out)
                    running = running + pl.cast(n, pl.INT32)

            self._build_inverse_map(
                expert_indices, pub_counts, inverse_map, my_rank,
            )

            return (
                local_routed_x_out,
                local_expert_offset,
                local_expert_count,
                inverse_map,
            )

        # ---------- Stage 3a: expert_routed (local 36 experts) ----------
        @pl.function(type=pl.FunctionType.InCore)
        def _expert_routed(  # noqa: PLR0913, PLR0915
            self,
            local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
            local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
            local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
            w_gate: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_up: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_down: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.BF16
            ],
            local_routed_y: pl.Tensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
        ):
            for e in pl.parallel(n_local_experts):
                n_rows = pl.read(local_expert_count, [e])
                offset_i32 = pl.read(local_expert_offset, [e])
                offset = pl.cast(offset_i32, pl.INDEX)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_gate_up"):
                    h_tile = pl.create_tensor(
                        [ROUTED_MAX_TILE, inter], dtype=pl.FP32,
                    )
                    h_tile[:, :] = pl.full(
                        [ROUTED_MAX_TILE, inter], dtype=pl.FP32,
                        value=0.0,
                    )
                    valid_rows = pl.cast(n_rows, pl.INDEX)

                    for nb in pl.range(inter // ROUTED_GATE_N_CHUNK):
                        n0 = nb * ROUTED_GATE_N_CHUNK

                        x0 = pl.slice(
                            local_routed_x,
                            [ROUTED_MAX_TILE, ROUTED_GATE_K_CHUNK],
                            [offset, 0],
                            valid_shape=[valid_rows, ROUTED_GATE_K_CHUNK],
                        )
                        wg0 = pl.slice(
                            w_gate,
                            [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [e, 0, n0],
                        )
                        wu0 = pl.slice(
                            w_up,
                            [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            [e, 0, n0],
                        )
                        wg0_2d = pl.reshape(
                            wg0,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                        )
                        wu0_2d = pl.reshape(
                            wu0,
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                        )
                        gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.FP32)
                        up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN // ROUTED_GATE_K_CHUNK):
                            k0 = kb * ROUTED_GATE_K_CHUNK
                            xk = pl.slice(
                                local_routed_x,
                                [ROUTED_MAX_TILE, ROUTED_GATE_K_CHUNK],
                                [offset, k0],
                                valid_shape=[
                                    valid_rows, ROUTED_GATE_K_CHUNK,
                                ],
                            )
                            wgk = pl.reshape(
                                pl.slice(
                                    w_gate,
                                    [
                                        1,
                                        ROUTED_GATE_K_CHUNK,
                                        ROUTED_GATE_N_CHUNK,
                                    ],
                                    [e, k0, n0],
                                ),
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            )
                            wuk = pl.reshape(
                                pl.slice(
                                    w_up,
                                    [
                                        1,
                                        ROUTED_GATE_K_CHUNK,
                                        ROUTED_GATE_N_CHUNK,
                                    ],
                                    [e, k0, n0],
                                ),
                                [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                            )
                            gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                            up_acc = pl.matmul_acc(up_acc, xk, wuk)

                        sigmoid = pl.recip(
                            pl.add(pl.exp(pl.neg(gate_acc)), 1.0),
                        )
                        silu = pl.mul(gate_acc, sigmoid)
                        if _routed_swiglu_step:
                            silu_c = pl.minimum(silu, _routed_swiglu_limit)
                            up_c = pl.maximum(
                                pl.minimum(up_acc, _routed_swiglu_limit),
                                -_routed_swiglu_limit,
                            )
                            gated = pl.mul(silu_c, up_c)
                        else:
                            gated = pl.mul(silu, up_acc)

                        gated_v = pl.set_validshape(
                            gated, valid_rows, ROUTED_GATE_N_CHUNK,
                        )
                        gated_m = pl.fillpad(
                            gated_v, pad_value=pl.PadValue.zero,
                        )
                        h_tile[
                            :, n0 : n0 + ROUTED_GATE_N_CHUNK
                        ] = gated_m

                    h_bf16 = pl.cast(h_tile, target_type=pl.BF16)

                    for db in pl.range(HIDDEN // ROUTED_DOWN_N_CHUNK):
                        d0 = db * ROUTED_DOWN_N_CHUNK
                        h0 = pl.slice(
                            h_bf16,
                            [ROUTED_MAX_TILE, ROUTED_DOWN_K_CHUNK],
                            [0, 0],
                            valid_shape=[
                                valid_rows, ROUTED_DOWN_K_CHUNK,
                            ],
                        )
                        wd0 = pl.reshape(
                            pl.slice(
                                w_down,
                                [
                                    1,
                                    ROUTED_DOWN_K_CHUNK,
                                    ROUTED_DOWN_N_CHUNK,
                                ],
                                [e, 0, d0],
                            ),
                            [ROUTED_DOWN_K_CHUNK, ROUTED_DOWN_N_CHUNK],
                        )
                        y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
                        for kb2 in pl.range(1, inter // ROUTED_DOWN_K_CHUNK):
                            k0 = kb2 * ROUTED_DOWN_K_CHUNK
                            hk = pl.slice(
                                h_bf16,
                                [ROUTED_MAX_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, k0],
                                valid_shape=[
                                    valid_rows, ROUTED_DOWN_K_CHUNK,
                                ],
                            )
                            wdk = pl.reshape(
                                pl.slice(
                                    w_down,
                                    [
                                        1,
                                        ROUTED_DOWN_K_CHUNK,
                                        ROUTED_DOWN_N_CHUNK,
                                    ],
                                    [e, k0, d0],
                                ),
                                [
                                    ROUTED_DOWN_K_CHUNK,
                                    ROUTED_DOWN_N_CHUNK,
                                ],
                            )
                            y_acc = pl.matmul_acc(y_acc, hk, wdk)

                        y_v = pl.set_validshape(
                            y_acc, valid_rows, ROUTED_DOWN_N_CHUNK,
                        )
                        y_m = pl.fillpad(
                            y_v, pad_value=pl.PadValue.zero,
                        )
                        local_routed_y = pl.assemble(
                            local_routed_y,
                            pl.cast(y_m, target_type=pl.BF16),
                            [offset, d0],
                        )

            return local_routed_y

        @pl.function(type=pl.FunctionType.InCore)
        def expert_routed_step(
            self,
            local_routed_x: pl.Tensor[[local_recv_max, HIDDEN], pl.BF16],
            local_expert_offset: pl.Tensor[[n_local_experts], pl.INT32],
            local_expert_count: pl.Tensor[[n_local_experts], pl.INT32],
            w_gate_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_up_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_down_r: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.BF16
            ],
            local_routed_y: pl.Out[
                pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]
            ],
        ) -> pl.Tensor[[local_recv_max, HIDDEN], pl.BF16]:
            local_routed_y = self._expert_routed(
                local_routed_x,
                local_expert_offset, local_expert_count,
                w_gate_r, w_up_r, w_down_r,
                local_routed_y,
            )
            return local_routed_y

        # ---------- Stage 3b: expert_shared (TP-sliced + tp_all_reduce) ----
        @pl.function(type=pl.FunctionType.Inline)
        def _expert_shared_local(
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            sh_y_shard: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up"):
                h_tile = pl.create_tensor(
                    [BATCH, sh_inter_local], dtype=pl.BF16,
                )

                x0 = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, 0])
                wg0 = pl.slice(
                    w_gate,
                    [SHARED_GATE_K_CHUNK, SHARED_GATE_N_CHUNK],
                    [0, 0],
                )
                wu0 = pl.slice(
                    w_up,
                    [SHARED_GATE_K_CHUNK, SHARED_GATE_N_CHUNK],
                    [0, 0],
                )
                gate_acc = pl.matmul(x0, wg0, out_dtype=pl.FP32)
                up_acc = pl.matmul(x0, wu0, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN // SHARED_GATE_K_CHUNK):
                    k0 = kb * SHARED_GATE_K_CHUNK
                    xk = pl.slice(x, [BATCH, SHARED_GATE_K_CHUNK], [0, k0])
                    wgk = pl.slice(
                        w_gate,
                        [SHARED_GATE_K_CHUNK, SHARED_GATE_N_CHUNK],
                        [k0, 0],
                    )
                    wuk = pl.slice(
                        w_up,
                        [SHARED_GATE_K_CHUNK, SHARED_GATE_N_CHUNK],
                        [k0, 0],
                    )
                    gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                    up_acc = pl.matmul_acc(up_acc, xk, wuk)

                sigmoid = pl.recip(
                    pl.add(pl.exp(pl.neg(gate_acc)), 1.0),
                )
                silu = pl.mul(gate_acc, sigmoid)
                if _shared_swiglu_step:
                    silu_c = pl.minimum(silu, _shared_swiglu_limit)
                    up_c = pl.maximum(
                        pl.minimum(up_acc, _shared_swiglu_limit),
                        -_shared_swiglu_limit,
                    )
                    gated = pl.mul(silu_c, up_c)
                else:
                    gated = pl.mul(silu, up_acc)

                h_tile[:, 0:SHARED_GATE_N_CHUNK] = pl.cast(
                    gated, target_type=pl.BF16,
                )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_down"):
                for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
                    d0 = db * SHARED_DOWN_N_CHUNK
                    h0 = pl.slice(
                        h_tile, [BATCH, SHARED_DOWN_K_CHUNK], [0, 0],
                    )
                    wd0 = pl.slice(
                        w_down,
                        [SHARED_DOWN_K_CHUNK, SHARED_DOWN_N_CHUNK],
                        [0, d0],
                    )
                    y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
                    sh_y_shard = pl.assemble(
                        sh_y_shard,
                        pl.cast(y_acc, target_type=pl.BF16),
                        [0, d0],
                    )

            return sh_y_shard

        @pl.function(type=pl.FunctionType.InCore)
        def expert_shared_step(  # noqa: PLR0913
            self,
            x: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            w_gate_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            sh_y: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, sh_tp_chunk], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            sh_y = self._expert_shared_local(
                x, w_gate_s, w_up_s, w_down_s, sh_y,
            )
            self._tp_all_reduce_moe(
                sh_y, sh_tmp_window, sh_signal_window, my_rank,
            )
            return sh_y

        # ---------- Stage 4: combine (EP a2a back + weighted gather) ------
        @pl.function(type=pl.FunctionType.InCore)
        def _publish_src_route_table(
            self,
            indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            src_route_table: pld.DistributedTensor[
                [n_ranks, n_local_experts, n_routes_per_rank], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            cursor = pl.create_tensor(
                [n_ranks * n_local_experts], dtype=pl.INT32,
            )
            for i in pl.range(n_ranks * n_local_experts):
                pl.write(cursor, [i], pl.cast(0, pl.INT32))

            for t in pl.range(BATCH):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // n_local_experts
                    loc_e = eid - dst * n_local_experts
                    bkt = dst * n_local_experts + loc_e
                    idx = pl.read(cursor, [bkt])
                    r_route = pl.cast(t * TOPK + k, pl.INT32)

                    if dst == my_rank:
                        tmp = pl.create_tensor([1], dtype=pl.INT32)
                        pl.write(tmp, [0], r_route)
                        tile = pl.load(tmp, [0], [1])
                        pl.store(
                            tile,
                            [my_rank, loc_e, pl.cast(idx, pl.INDEX)],
                            src_route_table,
                        )
                    else:
                        pld.system.notify(
                            target=src_route_table,
                            peer=dst,
                            offsets=[
                                my_rank, loc_e, pl.cast(idx, pl.INDEX),
                            ],
                            value=r_route,
                            op=pld.NotifyOp.Set,
                        )
                    pl.write(cursor, [bkt], pl.cast(idx + 1, pl.INT32))

        @pl.function(type=pl.FunctionType.InCore)
        def _push_routed_y_to_sources(  # noqa: PLR0913
            self,
            local_routed_y: pl.Tensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            combine_done: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [n_ranks, n_local_experts, n_routes_per_rank], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ):
            e_cursor = pl.cast(0, pl.INT32)
            for e in pl.range(n_local_experts):
                src_off = pl.cast(0, pl.INT32)
                for src in pl.range(n_ranks):
                    n = pl.cast(
                        pl.read(
                            pub_counts, [src * n_ranks + my_rank, e],
                        ),
                        pl.INDEX,
                    )
                    for row in pl.range(n):
                        r_route = pl.read(
                            src_route_table,
                            [src, e, pl.cast(row, pl.INDEX)],
                        )
                        local_row = (
                            pl.cast(e_cursor, pl.INDEX)
                            + pl.cast(src_off, pl.INDEX) + row
                        )
                        tile = pl.load(
                            local_routed_y,
                            [local_row, 0], [1, HIDDEN],
                        )
                        if src == my_rank:
                            pl.store(tile, [r_route, 0], routed_y_buf)
                        else:
                            # Phase X.4 minimal rewrite — see decode_layer.py
                            # for the FIXME(X.5) on per-row staging windows.
                            pl.store(tile, [r_route, 0], routed_y_buf)
                            pld.tensor.put(
                                routed_y_buf,
                                peer=src,
                                src=routed_y_buf,
                                atomic=pld.AtomicType.None_,
                            )
                    src_off = src_off + pl.cast(n, pl.INT32)
                total_e = pl.cast(0, pl.INT32)
                for src2 in pl.range(n_ranks):
                    total_e = total_e + pl.read(
                        pub_counts, [src2 * n_ranks + my_rank, e],
                    )
                e_cursor = e_cursor + total_e

            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=combine_done,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

        @pl.function(type=pl.FunctionType.Inline)
        def _weighted_gather_and_add(
            self,
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.BF16],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            moe_out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
        ):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_combine"):
                for b in pl.range(BATCH):
                    acc = pl.cast(
                        pl.load(sh_y, [b, 0], [1, HIDDEN]),
                        target_type=pl.FP32,
                    )
                    for k in pl.range(TOPK):
                        w_bf = pl.read(expert_weights, [b, k])
                        w_fp = pl.cast(w_bf, pl.FP32)

                        r_route = b * TOPK + k
                        row_fp32 = pl.cast(
                            pl.load(
                                routed_y_buf, [r_route, 0], [1, HIDDEN],
                            ),
                            target_type=pl.FP32,
                        )
                        weighted = pl.mul(row_fp32, w_fp)
                        acc = pl.add(acc, weighted)

                    pl.store(
                        pl.cast(acc, target_type=pl.BF16),
                        [b, 0],
                        moe_out,
                    )

            return moe_out

        @pl.function(type=pl.FunctionType.InCore)
        def combine_step(  # noqa: PLR0913
            self,
            local_routed_y: pl.Tensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            expert_indices: pl.Tensor[[BATCH, TOPK], pl.INT32],
            expert_weights: pl.Tensor[[BATCH, TOPK], pl.BF16],
            sh_y: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            moe_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [n_ranks, n_local_experts, n_routes_per_rank], pl.INT32
            ],
            route_pub_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            self._publish_src_route_table(
                expert_indices, src_route_table, my_rank,
            )
            for peer in pl.range(n_ranks):
                if peer != my_rank:
                    pld.system.notify(
                        target=route_pub_sig,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
            for src in pl.range(n_ranks):
                if src != my_rank:
                    pld.system.wait(
                        signal=route_pub_sig,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            self._push_routed_y_to_sources(
                local_routed_y,
                pub_counts,
                routed_y_buf,
                combine_done_sig,
                src_route_table,
                my_rank,
            )

            moe_out = self._weighted_gather_and_add(
                routed_y_buf, expert_weights, sh_y, moe_out,
            )
            return moe_out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16
            ],
            wk: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[layer_qhidden_dyn, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16
            ],
            positions: pl.Tensor[[PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_up_r: pl.Tensor[
                [n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_down_r: pl.Tensor[
                [n_local_experts, inter, HIDDEN], pl.BF16
            ],
            w_gate_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            next_hidden_out: pl.Out[
                pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]
            ],
            attn_tmp_window: pld.DistributedTensor[
                [PREFILL_T, tp_chunk], pl.BF16
            ],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            recv_x: pld.DistributedTensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            data_done_sig: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, sh_tp_chunk], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            src_route_table: pld.DistributedTensor[
                [n_ranks, n_local_experts, n_routes_per_rank], pl.INT32
            ],
            route_pub_sig: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            routed_y_buf: pld.DistributedTensor[
                [n_routes_per_rank, HIDDEN], pl.BF16
            ],
            combine_done_sig: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
            ],
            layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ):
            # ── A: prefill attention + tp_all_reduce -> resid1. ────────
            resid1 = pl.create_tensor([PREFILL_T, HIDDEN], dtype=pl.BF16)
            resid1 = attention_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                positions,
                resid1,
                layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )

            # ── B: post-attention zero-centred RMSNorm. ────────────────
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            resid1_fp32 = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.FP32,
            )
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_post_rmsnorm_zc",
            ):
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    rchunk = pl.cast(
                        pl.slice(
                            resid1, [PREFILL_T, K_CHUNK], [0, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    resid1_fp32 = pl.assemble(
                        resid1_fp32, rchunk, [0, k0],
                    )

                sq_sum = pl.full(
                    [1, PREFILL_T], dtype=pl.FP32, value=0.0,
                )
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    ck = pl.slice(
                        resid1_fp32, [PREFILL_T, K_CHUNK], [0, k0],
                    )
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(ck, ck)), [1, PREFILL_T],
                        ),
                    )
                inv_rms_moe = pl.recip(
                    pl.sqrt(
                        pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS),
                    ),
                )
                inv_rms_col = pl.reshape(inv_rms_moe, [PREFILL_T, 1])
                for kb3 in pl.range(hidden_blocks):
                    k0 = kb3 * K_CHUNK
                    norm_chunk = pl.slice(
                        resid1_fp32, [PREFILL_T, K_CHUNK], [0, k0],
                    )
                    gamma = pl.slice(
                        post_rms_weight, [1, K_CHUNK], [layer_idx, k0],
                    )
                    scaled = pl.row_expand_mul(norm_chunk, inv_rms_col)
                    normed = pl.col_expand_mul(scaled, pl.add(gamma, 1.0))
                    post_norm = pl.assemble(
                        post_norm,
                        pl.cast(normed, target_type=pl.BF16),
                        [0, k0],
                    )

            # ── C: prefill MoE adapter — Phase X.10 inlined per-tile loop.
            # Body copied from ``prefill_moe.PrefillMoE.chip_orch`` (which in
            # turn mirrors ``moe.EpTpMoE.chip_orch`` per-tile invocation): we
            # slice the [PREFILL_T, HIDDEN] post_norm into PREFILL_TILE_COUNT
            # independent [BATCH, HIDDEN] tiles and run the inlined gate /
            # dispatch / expert_routed / expert_shared / combine pipeline once
            # per tile. Routing is per-token, so this is bit-equivalent to a
            # single T=PREFILL_T pass. The window pool shared across tiles is
            # safe because each tile flushes (signal-wait pairs) before the
            # next reads.
            moe_out = pl.create_tensor(
                [PREFILL_T, HIDDEN], dtype=pl.BF16,
            )
            for tile_idx in pl.unroll(PREFILL_TILE_COUNT):
                t_lo = tile_idx * BATCH
                tile_x = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_in",
                ):
                    tile_x = pl.assemble(
                        tile_x,
                        pl.slice(post_norm, [BATCH, HIDDEN], [t_lo, 0]),
                        [0, 0],
                    )

                # 1) Gate (local, replicated).
                expert_indices = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.INT32,
                )
                expert_weights = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.BF16,
                )
                expert_indices, expert_weights = self.gate_step(
                    tile_x, gate_w, router_bias,
                    expert_indices, expert_weights,
                )

                # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
                sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                sh_y = self.expert_shared_step(
                    tile_x, w_gate_s, w_up_s, w_down_s, sh_y,
                    sh_tmp_window, sh_signal_window, my_rank,
                )

                # 3) Dispatch (EP all-to-all).
                local_routed_x = pl.create_tensor(
                    [local_recv_max, HIDDEN], dtype=pl.BF16,
                )
                local_expert_offset = pl.create_tensor(
                    [n_local_experts], dtype=pl.INT32,
                )
                local_expert_count = pl.create_tensor(
                    [n_local_experts], dtype=pl.INT32,
                )
                inverse_map = pl.create_tensor(
                    [BATCH, TOPK], dtype=pl.INT32,
                )
                (
                    local_routed_x,
                    local_expert_offset,
                    local_expert_count,
                    inverse_map,
                ) = self.dispatch_step(
                    tile_x, expert_indices,
                    local_routed_x,
                    local_expert_offset, local_expert_count, inverse_map,
                    pub_counts, count_done_sig, recv_x, data_done_sig,
                    my_rank,
                )

                # 4) Routed experts (local 36).
                local_routed_y = pl.create_tensor(
                    [local_recv_max, HIDDEN], dtype=pl.BF16,
                )
                local_routed_y = self.expert_routed_step(
                    local_routed_x,
                    local_expert_offset, local_expert_count,
                    w_gate_r, w_up_r, w_down_r,
                    local_routed_y,
                )

                # 5) Combine (EP a2a back + weighted gather + sh_y add).
                tile_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                tile_y = self.combine_step(
                    local_routed_y,
                    expert_indices, expert_weights, sh_y,
                    tile_y,
                    pub_counts, src_route_table, route_pub_sig,
                    routed_y_buf, combine_done_sig,
                    my_rank,
                )

                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_out",
                ):
                    moe_out = pl.assemble(
                        moe_out,
                        pl.slice(tile_y, [BATCH, HIDDEN], [0, 0]),
                        [t_lo, 0],
                    )

            # ── D: residual add. ───────────────────────────────────────
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="prefill_moe_residual_add",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    m = pl.cast(
                        pl.slice(
                            moe_out, [PREFILL_T, K_CHUNK], [0, k0],
                        ),
                        target_type=pl.FP32,
                    )
                    r = pl.slice(
                        resid1_fp32, [PREFILL_T, K_CHUNK], [0, k0],
                    )
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(pl.add(r, m), target_type=pl.BF16),
                        [0, k0],
                    )
            return next_hidden_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[
                [tp_size, PREFILL_T, HIDDEN], pl.BF16
            ],
            input_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16
            ],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            k_norm_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HEAD_DIM], pl.FP32
            ],
            block_table: pl.Tensor[
                [tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32
            ],
            slot_mapping: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            rope_cos: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32
            ],
            rope_sin: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32
            ],
            k_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            wo: pl.Tensor[
                [tp_size, layer_qhidden_dyn, HIDDEN], pl.BF16
            ],
            w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16
            ],
            positions: pl.Tensor[[tp_size, PREFILL_T], pl.INT32],
            post_rms_weight: pl.Tensor[
                [tp_size, LAYER_DYN, HIDDEN], pl.FP32
            ],
            gate_w: pl.Tensor[
                [tp_size, HIDDEN, N_EXPERTS], pl.FP32
            ],
            router_bias: pl.Tensor[[tp_size, N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [tp_size, n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_up_r: pl.Tensor[
                [tp_size, n_local_experts, HIDDEN, inter], pl.BF16
            ],
            w_down_r: pl.Tensor[
                [tp_size, n_local_experts, inter, HIDDEN], pl.BF16
            ],
            w_gate_s: pl.Tensor[
                [tp_size, HIDDEN, sh_inter_local], pl.BF16
            ],
            w_up_s: pl.Tensor[
                [tp_size, HIDDEN, sh_inter_local], pl.BF16
            ],
            w_down_s: pl.Tensor[
                [tp_size, sh_inter_local, HIDDEN], pl.BF16
            ],
            next_hidden_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]
            ],
            layer_idx: pl.Scalar[pl.INT32],
        ):
            attn_tmp_buf = pld.alloc_window_buffer(
                PREFILL_T * tp_chunk * 2,
            )
            attn_sig_buf = pld.alloc_window_buffer(tp_size * 4)
            pub_counts_buf = pld.alloc_window_buffer(
                n_ranks * n_ranks * n_local_experts * 4,
            )
            count_done_buf = pld.alloc_window_buffer(n_ranks * 4)
            recv_x_buf = pld.alloc_window_buffer(
                local_recv_max * HIDDEN * 2,
            )
            data_done_buf = pld.alloc_window_buffer(n_ranks * 4)
            sh_tmp_buf = pld.alloc_window_buffer(BATCH * sh_tp_chunk * 2)
            sh_sig_buf = pld.alloc_window_buffer(n_ranks * 4)
            src_route_buf = pld.alloc_window_buffer(
                n_ranks * n_local_experts * n_routes_per_rank * 4,
            )
            route_pub_buf = pld.alloc_window_buffer(n_ranks * 4)
            routed_y_window_buf = pld.alloc_window_buffer(
                n_routes_per_rank * HIDDEN * 2,
            )
            combine_done_buf = pld.alloc_window_buffer(n_ranks * 4)

            for r in pl.range(pld.world_size()):
                attn_tmp_window = pld.window(
                    attn_tmp_buf, [PREFILL_T, tp_chunk], dtype=pl.BF16,
                )
                attn_signal_window = pld.window(
                    attn_sig_buf, [tp_size, 1], dtype=pl.INT32,
                )
                pub_counts = pld.window(
                    pub_counts_buf,
                    [n_ranks * n_ranks, n_local_experts],
                    dtype=pl.INT32,
                )
                count_done_sig = pld.window(
                    count_done_buf, [n_ranks, 1], dtype=pl.INT32,
                )
                recv_x = pld.window(
                    recv_x_buf,
                    [local_recv_max, HIDDEN], dtype=pl.BF16,
                )
                data_done_sig = pld.window(
                    data_done_buf, [n_ranks, 1], dtype=pl.INT32,
                )
                sh_tmp_window = pld.window(
                    sh_tmp_buf, [BATCH, sh_tp_chunk], dtype=pl.BF16,
                )
                sh_signal_window = pld.window(
                    sh_sig_buf, [n_ranks, 1], dtype=pl.INT32,
                )
                src_route_table = pld.window(
                    src_route_buf,
                    [n_ranks, n_local_experts, n_routes_per_rank],
                    dtype=pl.INT32,
                )
                route_pub_sig = pld.window(
                    route_pub_buf, [n_ranks, 1], dtype=pl.INT32,
                )
                routed_y_buf = pld.window(
                    routed_y_window_buf,
                    [n_routes_per_rank, HIDDEN], dtype=pl.BF16,
                )
                combine_done_sig = pld.window(
                    combine_done_buf, [n_ranks, 1], dtype=pl.INT32,
                )
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r],
                    k_cache[r], v_cache[r],
                    wo[r], w_g[r],
                    positions[r],
                    post_rms_weight[r],
                    gate_w[r], router_bias[r],
                    w_gate_r[r], w_up_r[r], w_down_r[r],
                    w_gate_s[r], w_up_s[r], w_down_s[r],
                    next_hidden_out[r],
                    attn_tmp_window, attn_signal_window,
                    pub_counts, count_done_sig,
                    recv_x, data_done_sig,
                    sh_tmp_window, sh_signal_window,
                    src_route_table, route_pub_sig,
                    routed_y_buf, combine_done_sig,
                    layer_idx,
                    r,
                    device=r,
                )

    return PrefillLayerMoE


# -----------------------------------------------------------------------------
# Pre-built specialisations.
# -----------------------------------------------------------------------------
prefill_layer_full_dense = _build_prefill_layer_dense_program(full=True)
prefill_layer_swa_dense = _build_prefill_layer_dense_program(full=False)

prefill_layer_full_moe_silu_silu = _build_prefill_layer_moe_program(
    full=True, routed_lim=0.0, shared_lim=0.0,
)
prefill_layer_full_moe_swiglu7_silu = _build_prefill_layer_moe_program(
    full=True, routed_lim=7.0, shared_lim=0.0,
)
prefill_layer_full_moe_swiglu7_swiglu16 = _build_prefill_layer_moe_program(
    full=True, routed_lim=7.0, shared_lim=16.0,
)
prefill_layer_swa_moe_silu_silu = _build_prefill_layer_moe_program(
    full=False, routed_lim=0.0, shared_lim=0.0,
)
prefill_layer_swa_moe_swiglu7_silu = _build_prefill_layer_moe_program(
    full=False, routed_lim=7.0, shared_lim=0.0,
)
prefill_layer_swa_moe_swiglu7_swiglu16 = _build_prefill_layer_moe_program(
    full=False, routed_lim=7.0, shared_lim=16.0,
)


_KIND_BY_PAIR_FULL = {
    (0.0, 0.0): ("full_moe_silu_silu", prefill_layer_full_moe_silu_silu),
    (7.0, 0.0): (
        "full_moe_swiglu7_silu", prefill_layer_full_moe_swiglu7_silu,
    ),
    (7.0, 16.0): (
        "full_moe_swiglu7_swiglu16",
        prefill_layer_full_moe_swiglu7_swiglu16,
    ),
}
_KIND_BY_PAIR_SWA = {
    (0.0, 0.0): ("swa_moe_silu_silu", prefill_layer_swa_moe_silu_silu),
    (7.0, 0.0): (
        "swa_moe_swiglu7_silu", prefill_layer_swa_moe_swiglu7_silu,
    ),
    (7.0, 16.0): (
        "swa_moe_swiglu7_swiglu16",
        prefill_layer_swa_moe_swiglu7_swiglu16,
    ),
}


def select_prefill_layer(layer_idx: int):
    """Return ``(program_class, kind)`` for the prefill layer at ``layer_idx``.

    Mirrors ``decode_layer.select_decode_layer`` but emits prefill-T
    specialisations. Eight kinds total.
    """
    full = is_full_attention(layer_idx)
    moe = is_moe_layer(layer_idx)
    if full and not moe:
        return prefill_layer_full_dense, "full_dense"
    if (not full) and (not moe):
        return prefill_layer_swa_dense, "swa_dense"
    routed_lim = float(SWIGLU_LIMITS[layer_idx])
    shared_lim = float(SWIGLU_LIMITS_SHARED[layer_idx])
    table = _KIND_BY_PAIR_FULL if full else _KIND_BY_PAIR_SWA
    try:
        kind, prog = table[(routed_lim, shared_lim)]
    except KeyError as err:
        raise ValueError(
            f"Unsupported (routed={routed_lim}, shared={shared_lim}) "
            f"for layer {layer_idx}",
        ) from err
    return prog, kind


# =============================================================================
# Top-level @pl.program — Step3p5PrefillFwd.
# =============================================================================
def _build_prefill_fwd_program(tp_size: int = TP_WORLD_SIZE):
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must divide tp_size={tp_size}"
        )

    rms_lm_head_inline = pl.inline(rms_lm_head._func)

    @pl.program
    class Step3p5PrefillFwd:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[
                [VOCAB_LOCAL, HIDDEN], pl.BF16
            ],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]
            ],
        ):
            """Final RMSNorm + vocab-sliced LM head (per-rank shard)."""
            # NOTE: bind_dynamic() removed — @pl.function bodies don't
            # accept tensor method calls; dynamic dim propagates from the
            # @pl.program signature.
            logits_shard_out = rms_lm_head_inline(
                current_hidden,
                final_norm_weight,
                lm_head_weight,
                seq_lens,
                logits_shard_out,
            )
            return logits_shard_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913
            self,
            hidden_states: pl.Tensor[
                [tp_size, BATCH, HIDDEN], pl.BF16
            ],
            final_norm_weight: pl.Tensor[
                [tp_size, 1, HIDDEN], pl.FP32
            ],
            lm_head_weight: pl.Tensor[
                [tp_size, VOCAB_LOCAL, HIDDEN], pl.BF16
            ],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[
                    [tp_size, USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32
                ]
            ],
        ):
            for r in pl.range(pld.world_size()):
                self.chip_orch(
                    hidden_states[r],
                    final_norm_weight[r],
                    lm_head_weight[r],
                    seq_lens[r],
                    logits_shard_out[r],
                    device=r,
                )

    return Step3p5PrefillFwd


Step3p5PrefillFwd = _build_prefill_fwd_program(TP_WORLD_SIZE)


# =============================================================================
# Distributed-mock harness — pure torch 8-rank simulation.
# =============================================================================
def _torch_zc_rmsnorm(x, gamma, eps=1e-6):
    import torch

    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    g = gamma.float() + 1.0
    return (x.float() * torch.rsqrt(var + eps) * g)


def _torch_dense_mlp_partial(
    *, resid1, post_rms, w_gate, w_up, w_down, eps=1e-6,
):
    import torch

    normed = _torch_zc_rmsnorm(
        resid1, post_rms[0:1, :], eps,
    ).bfloat16().float()
    gate = normed @ w_gate.float()
    up = normed @ w_up.float()
    silu = gate * torch.sigmoid(gate)
    mlp = (silu * up).bfloat16().float()
    return (mlp @ w_down.float()).bfloat16()


def run_distributed_mock(
    *,
    batch: int = 1,
    seq_len: int = PREFILL_SEQ,
    seed: int = 0,
    pass_rate_threshold: float = 0.97,
    n_ranks: int = TP_WORLD_SIZE,
):
    """Pure-torch 8-rank simulation of the 45-layer prefill TP wiring.

    Same correctness contract as ``decode_fwd.run_distributed_mock``:
    attention output is approximated as zero-centred normed hidden,
    MoE layers as identity, and the test focuses on the TP all-reduce
    of the dense MLP partial + the vocab-sliced LM head shard.
    """
    import torch

    torch.manual_seed(seed)
    if batch != PREFILL_BATCH or seq_len != PREFILL_SEQ:
        raise ValueError(
            f"prefill mock harness requires batch={PREFILL_BATCH} "
            f"seq_len={PREFILL_SEQ}; got batch={batch} seq_len={seq_len}"
        )

    t = batch * seq_len
    hidden = (torch.rand(t, HIDDEN) - 0.5).bfloat16()
    final_norm = ((torch.rand(1, HIDDEN) - 0.5) * 0.1).float()
    input_rms = (
        (torch.rand(NUM_HIDDEN_LAYERS, HIDDEN) - 0.5) * 0.1
    ).float()
    post_rms = (
        (torch.rand(NUM_HIDDEN_LAYERS, HIDDEN) - 0.5) * 0.1
    ).float()

    full_w_gate = torch.zeros(
        NUM_DENSE_LAYERS, HIDDEN, INTER_LOCAL * n_ranks,
    )
    full_w_up = torch.zeros(
        NUM_DENSE_LAYERS, HIDDEN, INTER_LOCAL * n_ranks,
    )
    full_w_down = torch.zeros(
        NUM_DENSE_LAYERS, INTER_LOCAL * n_ranks, HIDDEN,
    )
    for d in range(NUM_DENSE_LAYERS):
        full_w_gate[d] = (
            torch.rand(HIDDEN, INTER_LOCAL * n_ranks) - 0.5
        ) / HIDDEN ** 0.5
        full_w_up[d] = (
            torch.rand(HIDDEN, INTER_LOCAL * n_ranks) - 0.5
        ) / HIDDEN ** 0.5
        full_w_down[d] = (
            torch.rand(INTER_LOCAL * n_ranks, HIDDEN) - 0.5
        ) / (INTER_LOCAL * n_ranks) ** 0.5

    rank_w_gate = [
        full_w_gate[:, :, r * INTER_LOCAL:(r + 1) * INTER_LOCAL].clone()
        for r in range(n_ranks)
    ]
    rank_w_up = [
        full_w_up[:, :, r * INTER_LOCAL:(r + 1) * INTER_LOCAL].clone()
        for r in range(n_ranks)
    ]
    rank_w_down = [
        full_w_down[:, r * INTER_LOCAL:(r + 1) * INTER_LOCAL, :].clone()
        for r in range(n_ranks)
    ]

    full_lm_head = (
        torch.rand(VOCAB_LOCAL * n_ranks, HIDDEN) - 0.5
    ) / HIDDEN ** 0.5
    rank_lm_head = [
        full_lm_head[r * VOCAB_LOCAL:(r + 1) * VOCAB_LOCAL, :].bfloat16()
        for r in range(n_ranks)
    ]

    # Verify per-layer dispatcher resolves cleanly.
    try:
        for li in range(NUM_HIDDEN_LAYERS):
            prog, kind = select_prefill_layer(li)
            if prog is None or not isinstance(kind, str):
                raise RuntimeError(
                    f"select_prefill_layer({li}) returned bad pair: "
                    f"({prog}, {kind})"
                )
    except Exception:  # pragma: no cover — runtime not always present
        pass

    # Single-card oracle (residual stream).
    oracle_hidden = hidden.clone().bfloat16()
    for li in range(NUM_HIDDEN_LAYERS):
        attn_out = _torch_zc_rmsnorm(
            oracle_hidden, input_rms[li:li + 1, :],
        ).bfloat16().float()
        resid1 = (oracle_hidden.float() + attn_out).bfloat16()
        if is_moe_layer(li):
            oracle_hidden = resid1
        else:
            d = DENSE_POS[li]
            mlp_out = _torch_dense_mlp_partial(
                resid1=resid1,
                post_rms=post_rms[li:li + 1, :],
                w_gate=full_w_gate[d],
                w_up=full_w_up[d],
                w_down=full_w_down[d],
            )
            oracle_hidden = (resid1.float() + mlp_out.float()).bfloat16()

    # Pick the last-token slot of every batch row (PREFILL_BATCH=1, so
    # this is the row at position seq_len - 1) and pad up to BATCH rows
    # so rms_lm_head sees its static shape.
    last_idx = seq_len - 1
    oracle_last = oracle_hidden[last_idx:last_idx + 1, :].clone()
    oracle_last_pad = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
    oracle_last_pad[:1, :] = oracle_last
    oracle_normed = _torch_zc_rmsnorm(
        oracle_last_pad, final_norm,
    ).bfloat16()
    oracle_logits_full = (
        oracle_normed.float() @ full_lm_head.float().T
    )

    rank_pass_rates = []
    for r in range(n_ranks):
        rank_hidden = hidden.clone().bfloat16()
        for li in range(NUM_HIDDEN_LAYERS):
            attn_out = _torch_zc_rmsnorm(
                rank_hidden, input_rms[li:li + 1, :],
            ).bfloat16().float()
            resid1 = (rank_hidden.float() + attn_out).bfloat16()
            if is_moe_layer(li):
                rank_hidden = resid1
            else:
                d = DENSE_POS[li]
                summed = torch.zeros(t, HIDDEN)
                for rr in range(n_ranks):
                    p = _torch_dense_mlp_partial(
                        resid1=resid1,
                        post_rms=post_rms[li:li + 1, :],
                        w_gate=rank_w_gate[rr][d],
                        w_up=rank_w_up[rr][d],
                        w_down=rank_w_down[rr][d],
                    )
                    summed = summed + p.float()
                rank_hidden = (
                    resid1.float() + summed
                ).bfloat16()

        rank_last = rank_hidden[last_idx:last_idx + 1, :].clone()
        rank_last_pad = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)
        rank_last_pad[:1, :] = rank_last
        rank_normed = _torch_zc_rmsnorm(
            rank_last_pad, final_norm,
        ).bfloat16()
        rank_logits_shard = (
            rank_normed.float() @ rank_lm_head[r].float().T
        )
        expected_shard = oracle_logits_full[
            :, r * VOCAB_LOCAL:(r + 1) * VOCAB_LOCAL,
        ]
        close = torch.isclose(
            rank_logits_shard, expected_shard,
            rtol=5e-3, atol=5e-3,
        )
        rate = close.float().mean().item()
        rank_pass_rates.append(rate)

    worst = min(rank_pass_rates)
    avg = sum(rank_pass_rates) / len(rank_pass_rates)
    ok = worst >= pass_rate_threshold
    return {
        "ok": ok,
        "worst_pass_rate": worst,
        "avg_pass_rate": avg,
        "rank_pass_rates": rank_pass_rates,
        "threshold": pass_rate_threshold,
    }


__all__ = [
    "Step3p5PrefillFwd",
    "_build_prefill_fwd_program",
    "_build_prefill_layer_dense_program",
    "_build_prefill_layer_moe_program",
    "_prefill_dense_mlp_body_tp",
    "prefill_layer_full_dense",
    "prefill_layer_swa_dense",
    "prefill_layer_full_moe_silu_silu",
    "prefill_layer_full_moe_swiglu7_silu",
    "prefill_layer_full_moe_swiglu7_swiglu16",
    "prefill_layer_swa_moe_silu_silu",
    "prefill_layer_swa_moe_swiglu7_silu",
    "prefill_layer_swa_moe_swiglu7_swiglu16",
    "select_prefill_layer",
    "select_prefill_moe_block",
    "run_distributed_mock",
    "NUM_FULL_LAYERS",
    "NUM_SWA_LAYERS",
    "NUM_DENSE_LAYERS",
    "NUM_MOE_LAYERS",
    "FULL_POS",
    "SWA_POS",
    "DENSE_POS",
    "MOE_POS",
    "N_RANKS",
    "N_LOCAL_EXPERTS",
    "TP_CHUNK",
    "LOCAL_RECV_MAX",
    "N_ROUTES_PER_RANK",
    "SH_INTER_LOCAL",
    "INTER",
    "INTER_LOCAL",
    "N_EXPERTS",
    "PREFILL_BATCH",
    "PREFILL_SEQ",
    "PREFILL_T",
    "TOK_TILE",
]


# =============================================================================
# CLI entry — distributed-mock harness on 8 mock ranks.
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 prefill_fwd distributed-mock harness. Pure-torch "
            "8-rank simulation; validates 45-layer wiring + TP all-"
            "reduce of the dense MLP + vocab-sliced LM head against a "
            "single-card oracle. Locked to B=1, S=128."
        ),
    )
    parser.add_argument("-b", "--batch", type=int, default=PREFILL_BATCH)
    parser.add_argument("--seq-len", type=int, default=PREFILL_SEQ)
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seq_len > MAX_SEQ_DEFAULT:
        raise ValueError(
            f"prefill_fwd harness supports seq_len <= {MAX_SEQ_DEFAULT}",
        )

    # Dispatcher smoke-check.
    for li in range(NUM_HIDDEN_LAYERS):
        prog, kind = select_prefill_layer(li)
        assert prog is not None
        assert isinstance(kind, str)
    print("[prefill_fwd] all 45 main-layer dispatch entries resolve OK")

    result = run_distributed_mock(
        batch=args.batch,
        seq_len=args.seq_len,
        seed=args.seed,
        pass_rate_threshold=args.pass_rate,
    )

    print("=" * 72)
    print(
        "Step3p5 prefill_fwd — distributed-mock 8-rank simulation "
        f"(B={args.batch}, S={args.seq_len})"
    )
    print("=" * 72)
    print(f"  threshold       : {result['threshold']:.4f}")
    print(f"  avg pass rate   : {result['avg_pass_rate']:.6f}")
    print(f"  worst pass rate : {result['worst_pass_rate']:.6f}")
    for r, pr in enumerate(result["rank_pass_rates"]):
        marker = "OK " if pr >= result["threshold"] else "BAD"
        print(f"   rank {r}: {pr:.6f}  {marker}")
    print("=" * 72)

    if not result["ok"]:
        raise SystemExit(1)
    print("[prefill_fwd] distributed-mock 8-rank simulation PASSED")
