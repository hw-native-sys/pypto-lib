# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 per-layer decode dispatcher — TP/EP wired (Phase 9 Wave 3).

Each per-layer program is a ``@pl.program`` class composing the Wave-2
TP-refactored attention path with either the Wave-2 EP-refactored MoE
program or a TP-sliced dense-MLP body. The per-layer dispatcher selects
the right class from ``layer_idx`` via ``config.is_full_attention``
(attention flavour) and ``config.is_moe_layer`` (MLP flavour):

  layer_idx  | attention   | MLP
  -----------|-------------|------------
  0, 1, 2    | full / swa  | TP dense MLP
  3..44      | full / swa  | EP+TP MoE
  45..47     | swa         | TP dense MLP (driven by ``mtp.py``)

Eight specialisations are exposed, matching the brief from Wave-3:

  - ``decode_layer_full_dense``           — full attention + TP dense MLP
  - ``decode_layer_swa_dense``            — SWA attention + TP dense MLP
  - ``decode_layer_full_moe_silu_silu``   — full + (silu, silu)
  - ``decode_layer_full_moe_swiglu7_silu``— full + (swiglu7, silu)
  - ``decode_layer_full_moe_swiglu7_swiglu16`` — full + (swiglu7, swiglu16)
  - ``decode_layer_swa_moe_silu_silu``    — SWA  + (silu, silu)
  - ``decode_layer_swa_moe_swiglu7_silu`` — SWA  + (swiglu7, silu)
  - ``decode_layer_swa_moe_swiglu7_swiglu16`` — SWA  + (swiglu7, swiglu16)

Dense-MLP TP slicing (documented in ``_dense_mlp_body_tp``):
  The Wave-2 attention path emits a fully-reduced residual stream on
  every rank (``resid1`` is replicated). The dense MLP that follows
  shards the intermediate axis ``INTERMEDIATE = 11264`` across the TP
  group:
    * ``w_gate / w_up``  per rank ``[HIDDEN, INTERMEDIATE_LOCAL=1408]``
    * ``w_down``         per rank ``[INTERMEDIATE_LOCAL, HIDDEN]``
  After the local ``w_down`` matmul each rank holds a *partial* hidden
  ``[BATCH, HIDDEN]`` BF16 (one term of the TP-group sum); the body
  invokes :func:`tp_all_reduce` to homogenise the partial sums across
  ranks, then adds the post-attention residual ``resid1`` back on top
  (residual is replicated, so the add commutes with the all-reduce).

Window contract (caller is responsible — see ``decode_fwd.py``):
  * Each ``decode_layer_*`` program's ``chip_orch`` takes a fresh
    per-layer ``tmp_window`` (BF16, ``[BATCH, HIDDEN // TP_WORLD_SIZE]``)
    and ``signal_window`` (INT32, ``[TP_WORLD_SIZE, 1]``) for the dense
    MLP's tp_all_reduce. AtomicAdd cells accumulate across the ring
    steps, so the caller MUST allocate a fresh signal window per
    call site (one per dense-MLP layer).
  * For MoE layers, the EP+TP windows for the embedded ``EpTpMoE``
    program are passed through verbatim (see ``moe.EpTpMoE`` for the
    full list).

Per-layer ``layer_idx`` is a runtime ``pl.Scalar[pl.INT32]`` — the slabs
inside ``input_rms_weight`` / ``post_rms_weight`` / ``q_norm_weight`` /
``k_norm_weight`` / ``wq`` / ``wo`` / ``w_gate`` etc. are sliced by it.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from ._ops import zero_centered_rmsnorm_apply
from .attention_full import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_FULL,
    attention_full,
)
from .attention_swa import (
    LAYER_QHIDDEN_ROWS_DYN as LAYER_QHIDDEN_ROWS_DYN_SWA,
    attention_swa,
)
from .config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    HIDDEN_Q_SWA_LOCAL,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    KV_PROJ_K_CHUNK_LOCAL,
    KV_CACHE_ROWS_DYN,
    KV_HEADS_LOCAL,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LAYER_ROPE_THETA,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ_DEFAULT,
    MLP_OUT_CHUNK,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_FULL_LOCAL_PAD,
    NUM_HEADS_SWA_LOCAL,
    NUM_HEADS_SWA_LOCAL_PAD,
    OUT_PROJ_K_CHUNK,
    OUT_PROJ_N_CHUNK,
    Q_HEAD_BATCH_FULL,
    Q_HEAD_BATCH_SWA,
    Q_HEAD_PAD_FULL,
    Q_HEAD_PAD_SWA,
    Q_OUT_CHUNK,
    Q_PER_KV_FULL,
    Q_PER_KV_SWA,
    ROPE_SCALING,
    ROPE_SEQ_DYN,
    ROTARY_HALF_FULL,
    ROTARY_HALF_SWA,
    SHARE_EXPERT_DIM_LOCAL,
    SLIDING_WINDOW,
    SWIGLU_LIMITS,
    SWIGLU_LIMITS_SHARED,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    is_full_attention,
    is_moe_layer,
)
from .dispatch import LOCAL_RECV_MAX, PER_RANK_BUCKETS
from .moe import select_moe_block


# -----------------------------------------------------------------------------
# Local re-exports for shorter signatures.
# -----------------------------------------------------------------------------
N_EXPERTS = MOE_NUM_EXPERTS
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
INTER_R = MOE_INTERMEDIATE
INTER_S_LOCAL = SHARE_EXPERT_DIM_LOCAL
INTER_LOCAL = INTERMEDIATE_LOCAL
TP_CHUNK = HIDDEN // TP_WORLD_SIZE
TOPK = 8
N_ROUTES_PER_RANK = BATCH * TOPK


assert HIDDEN % TP_WORLD_SIZE == 0
assert INTER_LOCAL % MLP_OUT_CHUNK == 0


# -----------------------------------------------------------------------------
# Phase X.8 — kernel-internal constants for the inlined MoE method bodies.
#
# pypto frontend rejects ``self._embedded_moe_cls().chip_orch(...)``
# (instantiating a ``@pl.program`` inside another ``@pl.program`` body is not a
# supported feature), so the entire body of ``EpTpMoE`` (from ``moe.py``) —
# every ``@pl.function`` method plus the ``chip_orch`` body — is inlined
# directly into ``DecodeLayerMoE``. Their tiling / sort widths / activation
# thresholds live here as module-level constants so the lifted method bodies
# can reference them via closure capture without depending on ``moe.py`` at
# parse time. The originals in ``moe.py`` remain intact (its ``__main__``
# harness still drives them as a standalone @pl.program).
# -----------------------------------------------------------------------------

# Router (gate) kernel constants — mirrors gate.py / moe.ROUTER_*.
ROUTER_SCORE_PAD = 512
ROUTER_TOPK_PAD = 16
ROUTER_SORT_PAD = ROUTER_TOPK_PAD * 2
ROUTER_GATE_K_CHUNK = 256   # 512→256: keeps x0/xk [16,256] FP32 = 16384 B in Vec budget
ROUTER_GATE_N_CHUNK = 32   # N-chunk for gate matmul; [K=256,N=32] FP32 = 32768 B (L0B limit)
ROUTER_FP32_NEG_INF = -3.4028235e38
ROUTER_SCALE = 3.0  # MOE_ROUTER_SCALING_FACTOR
assert TOPK <= ROUTER_TOPK_PAD
assert HIDDEN % ROUTER_GATE_K_CHUNK == 0
assert N_EXPERTS % ROUTER_GATE_N_CHUNK == 0

# Routed-expert kernel constants — mirrors expert_routed.py / moe.ROUTED_*.
ROUTED_GATE_K_CHUNK = 64    # A-tile K dim; [32,64] BF16 = 4096 B (L0A)
ROUTED_GATE_N_CHUNK = 64    # B-tile N dim; [64,64] BF16 = 8192 B (L0B)
ROUTED_DOWN_K_CHUNK = 64    # A-tile K dim; [32,64] BF16 = 4096 B (L0A)
ROUTED_DOWN_N_CHUNK = 128   # B-tile N dim; [64,128] BF16 = 16384 B (L0B)
ROUTED_MAX_TILE = LOCAL_RECV_MAX

# Per-tile row count for the routed-expert compute body. PTOAS Vec UB on
# A2/A3 caps tiles at 192 KB; allocating ``[ROUTED_MAX_TILE=1024, MOE_INTERMEDIATE=1280]``
# FP32 = 5.2 MB blows past that by ~28×. We adopt deepseek/v4's row-tiling:
# ``RECV_TILE`` rows per inner pass, with an outer ``for tile_idx in pl.range(n_tiles)``
# loop. ``32 * 1280 * 4 = 160 KB`` keeps headroom for intermediates.
RECV_TILE = 32
assert ROUTED_MAX_TILE % RECV_TILE == 0, (
    f"ROUTED_MAX_TILE ({ROUTED_MAX_TILE}) must be divisible by RECV_TILE ({RECV_TILE})"
)
N_RECV_TILES = ROUTED_MAX_TILE // RECV_TILE  # 32 outer iterations
assert HIDDEN % ROUTED_GATE_K_CHUNK == 0
assert HIDDEN % ROUTED_DOWN_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_GATE_N_CHUNK == 0
assert MOE_INTERMEDIATE % ROUTED_DOWN_K_CHUNK == 0

# Shared-expert kernel constants — mirrors expert_shared.py / moe.SHARED_*.
SHARED_GATE_K_CHUNK = 256
SHARED_GATE_N_CHUNK = INTER_S_LOCAL  # 160 — one N tile covers the slice
SHARED_DOWN_K_CHUNK = INTER_S_LOCAL  # 160 — one K tile covers the slice
SHARED_DOWN_N_CHUNK = 256
assert HIDDEN % SHARED_GATE_K_CHUNK == 0
assert HIDDEN % SHARED_DOWN_N_CHUNK == 0


# =============================================================================
# Dense-MLP body — TP-sliced gate/up/down + tp_all_reduce.
#
# The intermediate axis ``INTERMEDIATE = 11264`` is sharded across the TP
# group: each rank holds ``INTERMEDIATE_LOCAL = 1408`` lanes of
# ``w_gate``/``w_up`` (column slice) and ``w_down`` (row slice). After
# the local ``w_down`` matmul each rank produces a partial
# ``[BATCH, HIDDEN]`` BF16 contribution to the full hidden output; the
# body's tp_all_reduce sums those partials across the group so every
# rank holds the same fully-reduced next hidden. The post-attention
# residual is replicated (the attention's tp_all_reduce already
# homogenised it) and gets added on top after the reduction.
# =============================================================================
@pl.jit.inline
def _dense_mlp_body_tp(
    resid1: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    next_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
    tmp_window: pld.DistributedTensor[[BATCH, TP_CHUNK], pl.BF16],
    signal_window: pld.DistributedTensor[[TP_WORLD_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    """Post-attention zero-centred RMSNorm + TP-sliced SwiGLU MLP + residual.

    The per-rank weight slabs are:
      * ``w_gate / w_up``  ``[HIDDEN, INTERMEDIATE_LOCAL=1408]`` per layer
      * ``w_down``         ``[INTERMEDIATE_LOCAL, HIDDEN]`` per layer

    After the local ``w_down`` matmul each rank holds a partial
    ``[BATCH, HIDDEN]`` BF16 (one of TP_WORLD_SIZE terms in the sum). The
    body then runs ``tp_all_reduce`` on the partial-hidden tile so every
    rank receives the fully-reduced output before the residual add.
    Step3p5's dense layers (0..2 + the MTP layers' dense MLP) all use
    plain SiLU activation (``SWIGLU_LIMITS == 0`` everywhere on the dense
    path), matching the single-card draft's activation choice.
    """
    hidden_blocks = HIDDEN // K_CHUNK
    mlp_out_blocks = INTER_LOCAL // MLP_OUT_CHUNK
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTER_LOCAL

    # ── Step 1: post-attention zero-centred RMSNorm of resid1. ──────────
    post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dense_post_rmsnorm_zc"):
        for kb in pl.range(hidden_blocks):
            k0 = kb * K_CHUNK
            rchunk = pl.cast(
                pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]),
                target_type=pl.FP32,
            )
            resid1_fp32 = pl.assemble(resid1_fp32, rchunk, [0, k0])

        sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb2 in pl.range(hidden_blocks):
            k0 = kb2 * K_CHUNK
            ck = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(
                    pl.row_sum(pl.mul(ck, ck)),
                    [1, BATCH],
                ),
            )
        inv_rms_dense = pl.recip(
            pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
        )
        inv_rms_col = pl.reshape(inv_rms_dense, [BATCH, 1])
        for kb3 in pl.range(hidden_blocks):
            k0 = kb3 * K_CHUNK
            norm_chunk = pl.slice(
                resid1_fp32, [BATCH, K_CHUNK], [0, k0],
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

    # ── Step 2: TP-sliced gate_up + plain SiLU into mlp_tile. ──────────
    # Phase A (2026-06-11): split the mixed AIC+AIV body so PTOAS does not
    # lower this scope to MixedKernels (the mixed-mode root is the 507018
    # VEC UB alignment crash site; the first such kernel in dispatch order
    # crashed deterministically — see phase-15 doc Phase A section).
    # Stage 2.a (cube): both gate and up matmuls into FP32 GM scratch.
    gate_acc_gm = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.FP32)
    up_acc_gm = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.FP32)
    for ob in pl.spmd(
        mlp_out_blocks, name_hint="dense_gate_up_matmul_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        mlp_o0 = ob * MLP_OUT_CHUNK
        post_chunk_0 = pl.slice(post_norm, [BATCH, K_CHUNK], [0, 0])
        wg_0 = pl.slice(
            w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0],
        )
        wu_0 = pl.slice(
            w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, mlp_o0],
        )
        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
        for kb in pl.range(1, hidden_blocks):
            k0 = kb * K_CHUNK
            post_chunk = pl.slice(post_norm, [BATCH, K_CHUNK], [0, k0])
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
        gate_acc_gm = pl.assemble(gate_acc_gm, gate_acc, [0, mlp_o0])
        up_acc_gm = pl.assemble(up_acc_gm, up_acc, [0, mlp_o0])

    # Stage 2.b (vec): SiLU(gate) * up, cast to BF16.
    mlp_tile = pl.create_tensor([BATCH, INTER_LOCAL], dtype=pl.BF16)
    for ob in pl.spmd(mlp_out_blocks, name_hint="dense_silu_cast_tp"):
        mlp_o0 = ob * MLP_OUT_CHUNK
        gate_chunk = pl.slice(
            gate_acc_gm, [BATCH, MLP_OUT_CHUNK], [0, mlp_o0],
        )
        up_chunk = pl.slice(
            up_acc_gm, [BATCH, MLP_OUT_CHUNK], [0, mlp_o0],
        )
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_chunk)), 1.0))
        mlp_chunk = pl.mul(pl.mul(gate_chunk, sigmoid), up_chunk)
        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
        mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, mlp_o0])

    # ── Step 3: TP-sliced w_down -> partial [BATCH, HIDDEN] BF16. ──────
    # Phase A: split cube matmul + vec cast (mirror of full_out_proj fix).
    partial_hidden_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    for dob in pl.spmd(
        hidden_blocks, name_hint="dense_down_matmul_tp",
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        d0 = dob * K_CHUNK
        mlp_chunk_0 = pl.slice(mlp_tile, [BATCH, MLP_OUT_CHUNK], [0, 0])
        w_down_chunk_0 = pl.slice(
            w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base, d0],
        )
        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
        for ob in pl.range(1, mlp_out_blocks):
            down_o0 = ob * MLP_OUT_CHUNK
            down_mlp_chunk_bf16 = pl.slice(
                mlp_tile, [BATCH, MLP_OUT_CHUNK], [0, down_o0],
            )
            w_down_chunk = pl.slice(
                w_down, [MLP_OUT_CHUNK, K_CHUNK],
                [layer_inter_base + down_o0, d0],
            )
            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk_bf16, w_down_chunk)
        partial_hidden_fp32 = pl.assemble(
            partial_hidden_fp32, down_acc, [0, d0],
        )

    partial_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for dob in pl.spmd(hidden_blocks, name_hint="dense_down_cast_tp"):
        d0 = dob * K_CHUNK
        fp32_chunk = pl.slice(
            partial_hidden_fp32, [BATCH, K_CHUNK], [0, d0],
        )
        partial_hidden = pl.assemble(
            partial_hidden,
            pl.cast(fp32_chunk, target_type=pl.BF16),
            [0, d0],
        )

    # ── Step 4: TP all-reduce across the group's partial hiddens. ──────
    # Phase X.2: ``self.tp_all_reduce`` resolves to a method on the
    # enclosing @pl.program class (DecodeLayerDense / DecodeLayerMoE).
    # Phase 15.1 single-rank gate: at TP=1 skip (mirror of 15.B).
    if TP_WORLD_SIZE > 1:
        self.tp_all_reduce(
            partial_hidden, tmp_window, signal_window, my_rank,
        )

    # ── Step 5: replicated residual add — next_hidden = resid1 + reduced.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dense_residual_add_tp"):
        for kb4 in pl.range(hidden_blocks):
            k0 = kb4 * K_CHUNK
            reduced = pl.cast(
                pl.slice(partial_hidden, [BATCH, K_CHUNK], [0, k0]),
                target_type=pl.FP32,
            )
            r = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])
            next_hidden = pl.assemble(
                next_hidden,
                pl.cast(pl.add(r, reduced), target_type=pl.BF16),
                [0, k0],
            )

    return next_hidden


# =============================================================================
# Per-layer @pl.program builders.
#
# Each builder returns a freshly-constructed @pl.program class with a
# chip_orch (per-rank orchestration) and a host_orch (per-call-site
# window allocation + per-rank dispatch). The builders are constructed
# inside Python factory functions so the module imports even on hosts
# that have not finished bringing up the pypto runtime (deferred-build
# pattern, mirrors the in-tree TP+EP MoE reference).
# =============================================================================
def _build_decode_layer_dense_program(
    *,
    full: bool,
    tp_size: int = TP_WORLD_SIZE,
):
    """Return a ``@pl.program`` class for an attention + dense MLP layer."""
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must divide tp_size={tp_size}"
        )
    attention_inline = (
        pl.inline(attention_full._func)
        if full
        else pl.inline(attention_swa._func)
    )
    dense_mlp_inline = pl.inline(_dense_mlp_body_tp._func)
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
    class DecodeLayerDense:
        # ---------- Collective: TP all_reduce (lifted from collectives.py) ----
        # Phase X.2: pull-side ring all-reduce body, t_rows=BATCH,
        # d_cols=HIDDEN, group_size=tp_size from factory closure.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, tp_chunk], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            group_size = tp_size
            # Inline shape constants — pypto's tile shape inference cannot follow
            # Python-level aliases like ``t_rows = BATCH`` past load/remote_load
            # boundaries (it preserves the alias name in the tile type, which
            # then mismatches the concrete shape from sibling pl.load calls).
            # Using the literals everywhere matches tests/st/distributed/test_l3_allreduce.py.
            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + group_size) % group_size
                recv_idx = (my_rank - step - 1 + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                send_tile = pl.load(
                    local, [0, send_idx * tp_chunk], [BATCH, tp_chunk],
                )
                pl.store(send_tile, [0, 0], tmp_window)
                pld.system.notify(
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window, offsets=[prev_rank, 0],
                    expected=pl.cast(step + 1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )
                recv_tile = pld.tile.remote_load(
                    tmp_window, peer=prev_rank,
                    offsets=[0, 0], shape=[BATCH, tp_chunk],
                )
                old_tile = pl.load(
                    local, [0, recv_idx * tp_chunk], [BATCH, tp_chunk],
                )
                # PTOAS A2/A3 ``tadd`` doesn't support bf16; upcast to f32,
                # add, then downcast for the store.
                summed_fp32 = pl.add(
                    pl.cast(old_tile, target_type=pl.FP32),
                    pl.cast(recv_tile, target_type=pl.FP32),
                )
                pl.store(
                    pl.cast(summed_fp32, target_type=pl.BF16),
                    [0, recv_idx * tp_chunk], local,
                )

            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + 1 + group_size) % group_size
                recv_idx = (my_rank - step + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                send_tile = pl.load(
                    local, [0, send_idx * tp_chunk], [BATCH, tp_chunk],
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
                    offsets=[0, 0], shape=[BATCH, tp_chunk],
                )
                pl.store(recv_tile, [0, recv_idx * tp_chunk], local)
            return local

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16],
            wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[layer_qhidden_dyn, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
            w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16],
            w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
            next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, tp_chunk], pl.BF16],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            mlp_tmp_window: pld.DistributedTensor[[BATCH, tp_chunk], pl.BF16],
            mlp_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1 = attention_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                resid1,
                layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )
            next_hidden_out = dense_mlp_inline(
                resid1, post_rms_weight,
                w_gate, w_up, w_down,
                next_hidden_out, layer_idx,
                mlp_tmp_window, mlp_signal_window, my_rank,
            )
            return next_hidden_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16
            ],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            k_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            wo: pl.Tensor[[tp_size, layer_qhidden_dyn, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16
            ],
            post_rms_weight: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
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
                pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]
            ],
            layer_idx: pl.Scalar[pl.INT32],
        ):
            attn_tmp_buf = pld.alloc_window_buffer(BATCH * tp_chunk * 2)
            attn_sig_buf = pld.alloc_window_buffer(tp_size * 4)
            mlp_tmp_buf = pld.alloc_window_buffer(BATCH * tp_chunk * 2)
            mlp_sig_buf = pld.alloc_window_buffer(tp_size * 4)

            for r in pl.range(pld.world_size()):
                attn_tmp_window = pld.window(
                    attn_tmp_buf, [BATCH, tp_chunk], dtype=pl.BF16,
                )
                attn_signal_window = pld.window(
                    attn_sig_buf, [tp_size, 1], dtype=pl.INT32,
                )
                mlp_tmp_window = pld.window(
                    mlp_tmp_buf, [BATCH, tp_chunk], dtype=pl.BF16,
                )
                mlp_signal_window = pld.window(
                    mlp_sig_buf, [tp_size, 1], dtype=pl.INT32,
                )
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    seq_lens[r], block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r],
                    k_cache[r], v_cache[r],
                    wo[r], w_g[r],
                    post_rms_weight[r],
                    w_gate[r], w_up[r], w_down[r],
                    next_hidden_out[r],
                    attn_tmp_window, attn_signal_window,
                    mlp_tmp_window, mlp_signal_window,
                    layer_idx,
                    r,
                    device=r,
                )

    return DecodeLayerDense


def _build_decode_layer_moe_program(
    *,
    full: bool,
    routed_lim: float,
    shared_lim: float,
    tp_size: int = TP_WORLD_SIZE,
):
    """Return a ``@pl.program`` class for attention + EP+TP MoE layer.

    Phase X.8: the entire ``EpTpMoE`` body (from ``moe.py``) is inlined into
    ``DecodeLayerMoE`` — every ``@pl.function`` method plus the body of
    ``EpTpMoE.chip_orch``. The frontend rejects
    ``self._embedded_moe_cls().chip_orch(...)`` (instantiating a ``@pl.program``
    inside another ``@pl.program`` body is not supported), so the activation
    choice is now baked at factory build time via Python closure constants
    (``_routed_swiglu_step`` / ``_shared_swiglu_step``) rather than via a
    factory call to a separate program class.
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must divide tp_size={tp_size}"
        )
    attention_inline = (
        pl.inline(attention_full._func)
        if full
        else pl.inline(attention_swa._func)
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
    inter = MOE_INTERMEDIATE
    sh_inter_local = INTER_S_LOCAL
    sh_tp_chunk = HIDDEN // tp_size
    local_recv_max = LOCAL_RECV_MAX  # matches dispatch.LOCAL_RECV_MAX (1024)
    n_routes_per_rank = BATCH * TOPK
    per_rank_buckets = PER_RANK_BUCKETS  # n_ranks * n_local_experts

    @pl.program
    class DecodeLayerMoE:
        # Phase X.8: the entire ``EpTpMoE`` body (gate / dispatch /
        # expert_routed / expert_shared / combine plus all Inline helpers and
        # the chip_orch body) is inlined into this class. The frontend rejects
        # ``self._embedded_moe_cls().chip_orch(...)`` (instantiating a
        # ``@pl.program`` inside another ``@pl.program`` body is not a
        # supported feature). The originals in ``moe.py`` remain intact for
        # the standalone MoE harness; the bodies are kept in lock-step here.

        # ---------- Collective: TP all_reduce (lifted from collectives.py) ----
        # Phase X.2: pull-side ring all-reduce body, baked t_rows=BATCH,
        # d_cols=HIDDEN, group_size=tp_size from factory closure. Used by the
        # attention path that's pl.inline()-spliced into chip_orch below.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
            self,
            local: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            tmp_window: pld.DistributedTensor[[BATCH, tp_chunk], pl.BF16],
            signal_window: pld.DistributedTensor[[tp_size, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            group_size = tp_size
            # Inline shape constants — pypto's tile shape inference cannot follow
            # Python-level aliases like ``t_rows = BATCH`` past load/remote_load
            # boundaries (it preserves the alias name in the tile type, which
            # then mismatches the concrete shape from sibling pl.load calls).
            # Using the literals everywhere matches tests/st/distributed/test_l3_allreduce.py.
            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + group_size) % group_size
                recv_idx = (my_rank - step - 1 + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                send_tile = pl.load(
                    local, [0, send_idx * tp_chunk], [BATCH, tp_chunk],
                )
                pl.store(send_tile, [0, 0], tmp_window)
                pld.system.notify(
                    target=signal_window, peer=next_rank,
                    offsets=[my_rank, 0], value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.wait(
                    signal=signal_window, offsets=[prev_rank, 0],
                    expected=pl.cast(step + 1, pl.INT32), cmp=pld.WaitCmp.Ge,
                )
                recv_tile = pld.tile.remote_load(
                    tmp_window, peer=prev_rank,
                    offsets=[0, 0], shape=[BATCH, tp_chunk],
                )
                old_tile = pl.load(
                    local, [0, recv_idx * tp_chunk], [BATCH, tp_chunk],
                )
                # PTOAS A2/A3 ``tadd`` doesn't support bf16; upcast to f32,
                # add, then downcast for the store.
                summed_fp32 = pl.add(
                    pl.cast(old_tile, target_type=pl.FP32),
                    pl.cast(recv_tile, target_type=pl.FP32),
                )
                pl.store(
                    pl.cast(summed_fp32, target_type=pl.BF16),
                    [0, recv_idx * tp_chunk], local,
                )

            for step in pl.range(group_size - 1):
                send_idx = (my_rank - step + 1 + group_size) % group_size
                recv_idx = (my_rank - step + group_size) % group_size
                next_rank = (my_rank + 1) % group_size
                prev_rank = (my_rank - 1 + group_size) % group_size
                send_tile = pl.load(
                    local, [0, send_idx * tp_chunk], [BATCH, tp_chunk],
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
                    offsets=[0, 0], shape=[BATCH, tp_chunk],
                )
                pl.store(recv_tile, [0, recv_idx * tp_chunk], local)
            return local

        # ===================================================================
        # Phase X.8 — inlined EpTpMoE @pl.function methods.
        # Bodies copied verbatim from ``moe.EpTpMoE`` (Phase X.3+X.4+X.5).
        # The activation choice (plain SiLU vs. SwigluStep@7/16) is baked at
        # factory build time via the ``_routed_swiglu_step`` /
        # ``_shared_swiglu_step`` Python closure constants — only one branch
        # is emitted per specialisation. Module-level constants from moe.py
        # (T, N_RANKS, INTER, etc.) are renamed to the local closure names
        # (BATCH, n_ranks, inter, ...) so the bodies type-check against this
        # factory's signatures.
        # ===================================================================

        # ---------- Collective: EP all_to_all ----------
        @pl.function(type=pl.FunctionType.Inline)
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

            # 1) Local self-bucket copy.
            n_self = pl.cast(pl.read(send_counts, [my_rank]), pl.INDEX)
            s_off_self = pl.cast(pl.read(send_offsets, [my_rank]), pl.INDEX)
            r_off_self = pl.cast(pl.read(recv_offsets, [my_rank]), pl.INDEX)
            for r in pl.range(n_self):
                self_tile = pl.load(
                    send, [s_off_self + r, 0], [1, d_cols],
                )
                pl.store(self_tile, [r_off_self + r, 0], recv)

            # 2) Set(1) notify every peer.
            for peer in pl.range(group_size):
                if peer != my_rank:
                    pld.system.notify(
                        target=signal_window,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )

            # 3) Ge(1) wait for every peer.
            for src in pl.range(group_size):
                if src != my_rank:
                    pld.system.wait(
                        signal=signal_window,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # 4) Pull every peer's bucket-for-me.
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
                # Initialise output pads once — columns beyond N_EXPERTS
                # keep 0 / NEG_INF so topk is not tricked by uninitialised values.
                score_buf[:, :] = pl.full(
                    [BATCH, ROUTER_SCORE_PAD], dtype=pl.FP32, value=0.0,
                )
                biased_buf[:, :] = pl.full(
                    [BATCH, ROUTER_SCORE_PAD],
                    dtype=pl.FP32, value=ROUTER_FP32_NEG_INF,
                )
                for nb in pl.range(N_EXPERTS // ROUTER_GATE_N_CHUNK):
                    n0 = nb * ROUTER_GATE_N_CHUNK
                    # Cast x per K-chunk → [BATCH,K] FP32 = 16384 B
                    # (Site 5 fix: avoids full [BATCH,HIDDEN] FP32 = 262144 B).
                    x0 = pl.cast(
                        pl.slice(x, [BATCH, ROUTER_GATE_K_CHUNK], [0, 0]),
                        target_type=pl.FP32,
                    )
                    w0 = pl.slice(
                        gate_w,
                        [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                        [0, n0],
                    )
                    logits_n = pl.matmul(x0, w0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // ROUTER_GATE_K_CHUNK):
                        k0 = kb * ROUTER_GATE_K_CHUNK
                        xk = pl.cast(
                            pl.slice(
                                x, [BATCH, ROUTER_GATE_K_CHUNK], [0, k0],
                            ),
                            target_type=pl.FP32,
                        )
                        wk = pl.slice(
                            gate_w,
                            [ROUTER_GATE_K_CHUNK, ROUTER_GATE_N_CHUNK],
                            [k0, n0],
                        )
                        logits_n = pl.matmul_acc(logits_n, xk, wk)
                    # Apply sigmoid per N-chunk — vec ops convert cube→vec
                    # block layout, avoiding blayout mismatch when storing
                    # into pre-created score_buf / biased_buf (vec layout).
                    score_n_chunk = pl.recip(
                        pl.add(pl.exp(pl.neg(logits_n)), 1.0),
                    )
                    bias_chunk = pl.slice(
                        router_bias, [ROUTER_GATE_N_CHUNK], [n0],
                    )
                    bias_row_chunk = pl.reshape(
                        bias_chunk, [1, ROUTER_GATE_N_CHUNK],
                    )
                    biased_n_chunk = pl.add(
                        score_n_chunk,
                        pl.col_expand_mul(
                            pl.full(
                                [BATCH, ROUTER_GATE_N_CHUNK],
                                dtype=pl.FP32, value=1.0,
                            ),
                            bias_row_chunk,
                        ),
                    )
                    score_buf[:, n0 : n0 + ROUTER_GATE_N_CHUNK] = score_n_chunk
                    biased_buf[:, n0 : n0 + ROUTER_GATE_N_CHUNK] = biased_n_chunk

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
            """Local histogram + per-rank prefix-sum prelude."""
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
            send_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            cursor_per_bucket: pl.Tensor[[per_rank_buckets], pl.INT32],
            bucket_offset: pl.Tensor[[per_rank_buckets], pl.INT32],
        ):
            """Pack outgoing tokens into ``send_buf`` ordered by (dst, loc_e)."""
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
                    x_tile = pl.load(x, [t, 0], [1, HIDDEN])
                    pl.store(x_tile, [slot, 0], send_buf)
                    pl.write(
                        cursor_per_bucket, [bkt],
                        pl.cast(slot_i32 + 1, pl.INT32),
                    )

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
            """Receiver-side CSR: scan pub_counts for my dst slot."""
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
            """Encode (dst_rank, dst_row_in_recv_buf) into one INT32 per (t,k)."""
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
            # ``send_buf`` is a DistributedTensor window allocated by the
            # orchestration caller (DDR-backed, ~8 MB BF16).  Using
            # DistributedTensor allows ep_all_to_all (inlined here) to call
            # pld.tile.remote_load on it for cross-rank pull.
            send_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
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

            cursor_bkt = pl.create_tensor(
                [per_rank_buckets], dtype=pl.INT32,
            )
            bucket_offset = pl.create_tensor(
                [per_rank_buckets], dtype=pl.INT32,
            )
            self._pack_send_payload(
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
        @pl.function(type=pl.FunctionType.Inline)
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
                valid_rows = pl.cast(n_rows, pl.INDEX)

                # Row-tile loop — process RECV_TILE rows per outer iteration.
                # Bridge-tensor pattern (h_bf16 at tile_idx loop level, outside
                # both moe_gate_up and moe_down pl.at scopes) eliminates the
                # [32,1280] FP32 h_tile accumulator from each scope's live set
                # and partitions gate/up from down-proj allocations
                # (mirrors deepseek/v4/expert_routed.py).
                #
                # ``tile_valid`` is min(RECV_TILE, n_rows - tile_row0) so the
                # trailing tile correctly masks out padding rows past
                # ``offset + n_rows``.
                for tile_idx in pl.range(N_RECV_TILES):
                    tile_row0 = tile_idx * RECV_TILE
                    tile_offset = offset + tile_row0
                    # Per-tile valid rows (scalar) — clamps trailing tile to
                    # the active row span. Use ``pl.min`` for scalar min/max
                    # (``pl.minimum`` is the tensor variant — frontend rejects
                    # mixing Scalar + ConstInt operands).
                    tile_valid = pl.min(RECV_TILE, valid_rows - tile_row0)

                    # Bridge tensor — lives at tile_idx loop level, shared
                    # between expert_gate_up and expert_down SPMD dispatches
                    # (mirrors deepseek/v4/expert_routed.py bridge pattern).
                    # As an Inline-level create_tensor it is in vec (UB) space;
                    # pl.slice of it in expert_down gives tmov vec→left ✓.
                    h_bf16 = pl.create_tensor(
                        [RECV_TILE, inter], dtype=pl.BF16,
                    )

                    # Gate+up projection: each SPMD block handles one N-chunk
                    # of the inter dimension.  pl.slice of external BF16 input
                    # (local_routed_x) produces tmov mat→left which is valid in
                    # cube kernels (cube kind = Inline + pl.spmd).
                    for nb in pl.spmd(
                        inter // ROUTED_GATE_N_CHUNK,
                        name_hint="expert_gate_up",
                    ):
                        n0 = nb * ROUTED_GATE_N_CHUNK
                        x0 = pl.slice(
                            local_routed_x,
                            [RECV_TILE, ROUTED_GATE_K_CHUNK],
                            [tile_offset, 0],
                            valid_shape=[tile_valid, ROUTED_GATE_K_CHUNK],
                        )
                        wg0_2d = pl.reshape(
                            pl.slice(
                                w_gate,
                                [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                [e, 0, n0],
                            ),
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                        )
                        wu0_2d = pl.reshape(
                            pl.slice(
                                w_up,
                                [1, ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                                [e, 0, n0],
                            ),
                            [ROUTED_GATE_K_CHUNK, ROUTED_GATE_N_CHUNK],
                        )
                        gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.FP32)
                        up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.FP32)
                        for kb in pl.range(1, HIDDEN // ROUTED_GATE_K_CHUNK):
                            k0 = kb * ROUTED_GATE_K_CHUNK
                            xk = pl.slice(
                                local_routed_x,
                                [RECV_TILE, ROUTED_GATE_K_CHUNK],
                                [tile_offset, k0],
                                valid_shape=[
                                    tile_valid, ROUTED_GATE_K_CHUNK,
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
                            gated, tile_valid, ROUTED_GATE_N_CHUNK,
                        )
                        # No fillpad — gated_v (none-pad mode) matches
                        # uninitialised h_bf16 subview (none-pad mode);
                        # expert_down reads h_bf16 with valid_shape= so
                        # padding rows beyond tile_valid are not used.
                        h_bf16[
                            :, n0 : n0 + ROUTED_GATE_N_CHUNK
                        ] = pl.cast(gated_v, target_type=pl.BF16)

                    # Down projection: each SPMD block handles one D-chunk of
                    # the HIDDEN output dimension.  h_bf16 is vec (UB) space so
                    # pl.slice of it gives tmov vec→left ✓.
                    for db in pl.spmd(
                        HIDDEN // ROUTED_DOWN_N_CHUNK,
                        name_hint="expert_down",
                    ):
                        d0 = db * ROUTED_DOWN_N_CHUNK
                        h0 = pl.slice(
                            h_bf16,
                            [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                            [0, 0],
                            valid_shape=[
                                tile_valid, ROUTED_DOWN_K_CHUNK,
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
                                [RECV_TILE, ROUTED_DOWN_K_CHUNK],
                                [0, k0],
                                valid_shape=[
                                    tile_valid, ROUTED_DOWN_K_CHUNK,
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
                            y_acc, tile_valid, ROUTED_DOWN_N_CHUNK,
                        )
                        y_m = pl.fillpad(
                            y_v, pad_value=pl.PadValue.zero,
                        )
                        local_routed_y = pl.assemble(
                            local_routed_y,
                            pl.cast(y_m, target_type=pl.BF16),
                            [tile_offset, d0],
                        )

            return local_routed_y

        @pl.function(type=pl.FunctionType.Inline)
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
            # Gate+up and down projections merged into one InCore kernel so that
            # h_tile (Vec SRAM) is live across both projections in the same
            # kernel dispatch.  Two separate pl.at(CORE_GROUP) scopes would
            # become two separate InCore kernel dispatches when expert_shared_step
            # is Inline, and h_tile cannot cross that dispatch boundary.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_mlp"):
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

                # Explicit K-chunking for the down projection.
                # K=160 (sh_inter_local) would trigger backend auto-K-split
                # which emits an invalid ``tmov acc→acc``.  Expose the K-loop
                # at the user level using K_INNER=32 (5 passes of 32) so the
                # compiler sees a structured pl.matmul + pl.matmul_acc chain
                # (same pattern as gate/up above) and never needs the copy.
                _SH_DOWN_K_INNER = 32  # hardware K-tile; 160 // 32 == 5 passes
                for db in pl.range(HIDDEN // SHARED_DOWN_N_CHUNK):
                    d0 = db * SHARED_DOWN_N_CHUNK
                    h0_k0 = pl.slice(
                        h_tile, [BATCH, _SH_DOWN_K_INNER], [0, 0],
                    )
                    wd0_k0 = pl.slice(
                        w_down,
                        [_SH_DOWN_K_INNER, SHARED_DOWN_N_CHUNK],
                        [0, d0],
                    )
                    y_acc = pl.matmul(h0_k0, wd0_k0, out_dtype=pl.FP32)
                    for kk in pl.range(1, sh_inter_local // _SH_DOWN_K_INNER):
                        kk0 = kk * _SH_DOWN_K_INNER
                        h0_kk = pl.slice(
                            h_tile, [BATCH, _SH_DOWN_K_INNER], [0, kk0],
                        )
                        wd0_kk = pl.slice(
                            w_down,
                            [_SH_DOWN_K_INNER, SHARED_DOWN_N_CHUNK],
                            [kk0, d0],
                        )
                        y_acc = pl.matmul_acc(y_acc, h0_kk, wd0_kk)
                    sh_y_shard = pl.assemble(
                        sh_y_shard,
                        pl.cast(y_acc, target_type=pl.BF16),
                        [0, d0],
                    )

            return sh_y_shard

        @pl.function(type=pl.FunctionType.Inline)
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
            # Phase 15.1 single-rank gate: skip TP=1 (mirror of 15.B).
            if TP_WORLD_SIZE > 1:
                self.tp_all_reduce(
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
                        # Self-rank publish: write the scalar r_route
                        # directly into the local view of the
                        # DistributedTensor. Mirrors the self-branch
                        # used for ``pub_counts`` above; avoids the
                        # ``tmp = create_tensor; load; store`` dance
                        # that the InCore tile verifier rejects with
                        # "tile.load requires TensorType ... got TileType".
                        pl.write(
                            src_route_table,
                            [my_rank, loc_e, pl.cast(idx, pl.INDEX)],
                            r_route,
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
                            pld.tile.remote_store(
                                tile,
                                target=routed_y_buf,
                                peer=src,
                                offsets=[r_route, 0],
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

        @pl.function(type=pl.FunctionType.Inline)
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
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="pub_route_barrier"):
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
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16],
            wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
            wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16],
            q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[layer_qhidden_dyn, HIDDEN], pl.BF16],
            w_g: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16],
            post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.BF16],
            w_up_r: pl.Tensor[[n_local_experts, HIDDEN, inter], pl.BF16],
            w_down_r: pl.Tensor[[n_local_experts, inter, HIDDEN], pl.BF16],
            w_gate_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_up_s: pl.Tensor[[HIDDEN, sh_inter_local], pl.BF16],
            w_down_s: pl.Tensor[[sh_inter_local, HIDDEN], pl.BF16],
            next_hidden_out: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
            attn_tmp_window: pld.DistributedTensor[[BATCH, tp_chunk], pl.BF16],
            attn_signal_window: pld.DistributedTensor[
                [tp_size, 1], pl.INT32
            ],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_x: pld.DistributedTensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            send_buf: pld.DistributedTensor[[local_recv_max, HIDDEN], pl.BF16],
            sh_tmp_window: pld.DistributedTensor[
                [BATCH, sh_tp_chunk], pl.BF16
            ],
            sh_signal_window: pld.DistributedTensor[
                [n_ranks, 1], pl.INT32
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
            layer_idx: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
            # ── A: attention + tp_all_reduce -> resid1 (replicated). ───
            resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1 = attention_inline(
                current_hidden,
                input_rms_weight,
                wq, wk, wv,
                q_norm_weight, k_norm_weight,
                seq_lens, block_table, slot_mapping,
                rope_cos, rope_sin,
                k_cache, v_cache,
                wo, w_g,
                resid1,
                layer_idx,
                attn_tmp_window,
                attn_signal_window,
                my_rank,
            )

            # ── B: post-attention zero-centred RMSNorm of resid1. ──────
            hidden_blocks = HIDDEN // K_CHUNK
            post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            resid1_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="moe_post_rmsnorm_zc",
            ):
                for kb in pl.range(hidden_blocks):
                    k0 = kb * K_CHUNK
                    rchunk = pl.cast(
                        pl.slice(resid1, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    resid1_fp32 = pl.assemble(resid1_fp32, rchunk, [0, k0])

                sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for kb2 in pl.range(hidden_blocks):
                    k0 = kb2 * K_CHUNK
                    ck = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(
                            pl.row_sum(pl.mul(ck, ck)),
                            [1, BATCH],
                        ),
                    )
                inv_rms_moe = pl.recip(
                    pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)),
                )
                inv_rms_col = pl.reshape(inv_rms_moe, [BATCH, 1])
                for kb3 in pl.range(hidden_blocks):
                    k0 = kb3 * K_CHUNK
                    norm_chunk = pl.slice(
                        resid1_fp32, [BATCH, K_CHUNK], [0, k0],
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

            # ── C: EP+TP MoE chip_orch -> moe_out (Phase X.8 inlined). ─
            # Body copied verbatim from ``moe.EpTpMoE.chip_orch``; calls the
            # inlined per-stage methods (``self.gate_step`` /
            # ``self.dispatch_step`` / ``self.expert_routed_step`` /
            # ``self.expert_shared_step`` / ``self.combine_step``) directly
            # rather than instantiating a separate ``@pl.program``.
            moe_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

            # 1) Gate (local, replicated).
            expert_indices = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
            expert_weights = pl.create_tensor([BATCH, TOPK], dtype=pl.BF16)
            expert_indices, expert_weights = self.gate_step(
                post_norm, gate_w, router_bias,
                expert_indices, expert_weights,
            )

            # 2) Shared-expert lane (TP-sliced + tp_all_reduce).
            sh_y = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            sh_y = self.expert_shared_step(
                post_norm, w_gate_s, w_up_s, w_down_s, sh_y,
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
            inverse_map = pl.create_tensor([BATCH, TOPK], dtype=pl.INT32)
            # send_buf is a DistributedTensor window passed in from host_orch
            # (~8 MB BF16); required by ep_all_to_all's pld.tile.remote_load.
            (
                local_routed_x,
                local_expert_offset,
                local_expert_count,
                inverse_map,
            ) = self.dispatch_step(
                post_norm, expert_indices,
                local_routed_x,
                local_expert_offset, local_expert_count, inverse_map,
                send_buf,
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
            moe_out = self.combine_step(
                local_routed_y,
                expert_indices, expert_weights, sh_y,
                moe_out,
                pub_counts, src_route_table, route_pub_sig,
                routed_y_buf, combine_done_sig,
                my_rank,
            )

            # ── D: residual add: next_hidden_out = resid1 + moe_out. ───
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="moe_residual_add",
            ):
                for kb4 in pl.range(hidden_blocks):
                    k0 = kb4 * K_CHUNK
                    m = pl.cast(
                        pl.slice(moe_out, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    r = pl.slice(resid1_fp32, [BATCH, K_CHUNK], [0, k0])
                    next_hidden_out = pl.assemble(
                        next_hidden_out,
                        pl.cast(pl.add(r, m), target_type=pl.BF16),
                        [0, k0],
                    )
            return next_hidden_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            wq: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, hidden_q_local], pl.BF16
            ],
            wk: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            q_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            rope_cos: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            rope_sin: pl.Tensor[[tp_size, ROPE_SEQ_DYN, rotary_dim], pl.FP32],
            k_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            wo: pl.Tensor[[tp_size, layer_qhidden_dyn, HIDDEN], pl.BF16],
            w_g: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, num_heads_local_pad], pl.BF16
            ],
            post_rms_weight: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            gate_w: pl.Tensor[[tp_size, HIDDEN, N_EXPERTS], pl.FP32],
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
                pl.Tensor[[tp_size, BATCH, HIDDEN], pl.BF16]
            ],
            layer_idx: pl.Scalar[pl.INT32],
        ):
            attn_tmp_buf = pld.alloc_window_buffer(BATCH * tp_chunk * 2)
            attn_sig_buf = pld.alloc_window_buffer(tp_size * 4)
            pub_counts_buf = pld.alloc_window_buffer(
                n_ranks * n_ranks * n_local_experts * 4,
            )
            count_done_buf = pld.alloc_window_buffer(n_ranks * 4)
            recv_x_buf = pld.alloc_window_buffer(
                local_recv_max * HIDDEN * 2,
            )
            data_done_buf = pld.alloc_window_buffer(n_ranks * 4)
            send_x_buf = pld.alloc_window_buffer(
                local_recv_max * HIDDEN * 2,
            )
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
                    attn_tmp_buf, [BATCH, tp_chunk], dtype=pl.BF16,
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
                    [local_recv_max, HIDDEN],
                    dtype=pl.BF16,
                )
                data_done_sig = pld.window(
                    data_done_buf, [n_ranks, 1], dtype=pl.INT32,
                )
                send_x = pld.window(
                    send_x_buf,
                    [local_recv_max, HIDDEN],
                    dtype=pl.BF16,
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
                    [n_routes_per_rank, HIDDEN],
                    dtype=pl.BF16,
                )
                combine_done_sig = pld.window(
                    combine_done_buf, [n_ranks, 1], dtype=pl.INT32,
                )
                self.chip_orch(
                    current_hidden[r],
                    input_rms_weight[r],
                    wq[r], wk[r], wv[r],
                    q_norm_weight[r], k_norm_weight[r],
                    seq_lens[r], block_table[r], slot_mapping[r],
                    rope_cos[r], rope_sin[r],
                    k_cache[r], v_cache[r],
                    wo[r], w_g[r],
                    post_rms_weight[r],
                    gate_w[r], router_bias[r],
                    w_gate_r[r], w_up_r[r], w_down_r[r],
                    w_gate_s[r], w_up_s[r], w_down_s[r],
                    next_hidden_out[r],
                    attn_tmp_window, attn_signal_window,
                    pub_counts, count_done_sig,
                    recv_x, data_done_sig,
                    send_x,
                    sh_tmp_window, sh_signal_window,
                    src_route_table, route_pub_sig,
                    routed_y_buf, combine_done_sig,
                    layer_idx,
                    r,
                    device=r,
                )

    return DecodeLayerMoE


# -----------------------------------------------------------------------------
# Pre-built specialisations matching the eight kinds in select_decode_layer.
# -----------------------------------------------------------------------------
decode_layer_full_dense = _build_decode_layer_dense_program(full=True)
decode_layer_swa_dense = _build_decode_layer_dense_program(full=False)

# Phase X.7 — MoE @pl.program builds are deferred via module __getattr__
# because the inner ``self._embedded_moe_cls().chip_orch(...)`` construction
# is rejected by the pypto frontend (instantiating a @pl.program at IR-build
# time is not a supported operation). Mirrors the prefill_moe.py lazy pattern.
# Workers/users that DO know how to drive the MoE path will still trigger the
# build on first attribute access; the dense path imports cleanly without it.
_LAZY_DECODE_MOE_BUILDS = {
    "decode_layer_full_moe_silu_silu":      (True,  0.0, 0.0),
    "decode_layer_full_moe_swiglu7_silu":   (True,  7.0, 0.0),
    "decode_layer_full_moe_swiglu7_swiglu16": (True, 7.0, 16.0),
    "decode_layer_swa_moe_silu_silu":       (False, 0.0, 0.0),
    "decode_layer_swa_moe_swiglu7_silu":    (False, 7.0, 0.0),
    "decode_layer_swa_moe_swiglu7_swiglu16": (False, 7.0, 16.0),
    "decode_layer_full_moe":                (True,  0.0, 0.0),
    "decode_layer_swa_moe":                 (False, 0.0, 0.0),
}
_LAZY_DECODE_MOE_CACHE: dict = {}


def __getattr__(name: str):
    if name in _LAZY_DECODE_MOE_BUILDS:
        if name not in _LAZY_DECODE_MOE_CACHE:
            full, routed_lim, shared_lim = _LAZY_DECODE_MOE_BUILDS[name]
            _LAZY_DECODE_MOE_CACHE[name] = _build_decode_layer_moe_program(
                full=full, routed_lim=routed_lim, shared_lim=shared_lim,
            )
        return _LAZY_DECODE_MOE_CACHE[name]
    raise AttributeError(name)


# =============================================================================
# Registry helper.
# =============================================================================
# MoE program lookup keys — values are resolved lazily via ``__getattr__``
# above (Phase X.7 deferred build to avoid the @pl.program-instantiation
# bounce inside DecodeLayerMoE.chip_orch).
_KIND_BY_PAIR_FULL_NAMES = {
    (0.0, 0.0): ("full_moe_silu_silu", "decode_layer_full_moe_silu_silu"),
    (7.0, 0.0): ("full_moe_swiglu7_silu", "decode_layer_full_moe_swiglu7_silu"),
    (7.0, 16.0): (
        "full_moe_swiglu7_swiglu16",
        "decode_layer_full_moe_swiglu7_swiglu16",
    ),
}
_KIND_BY_PAIR_SWA_NAMES = {
    (0.0, 0.0): ("swa_moe_silu_silu", "decode_layer_swa_moe_silu_silu"),
    (7.0, 0.0): ("swa_moe_swiglu7_silu", "decode_layer_swa_moe_swiglu7_silu"),
    (7.0, 16.0): (
        "swa_moe_swiglu7_swiglu16",
        "decode_layer_swa_moe_swiglu7_swiglu16",
    ),
}


def select_decode_layer(layer_idx: int):
    """Return the ``(program_class, kind)`` pair for a step3p5 layer index.

    Args:
        layer_idx: index into the 48-long LAYER_TYPES table (45 main +
            3 MTP).

    Returns:
        ``(program_class, kind)`` where ``program_class`` is one of the
        eight pre-built ``@pl.program`` classes and ``kind`` is one of
        ``"full_dense"``, ``"swa_dense"``,
        ``"full_moe_silu_silu"``, ``"full_moe_swiglu7_silu"``,
        ``"full_moe_swiglu7_swiglu16"``, ``"swa_moe_silu_silu"``,
        ``"swa_moe_swiglu7_silu"``, ``"swa_moe_swiglu7_swiglu16"``.

    Notes:
        * Dense MLP is used for layers 0..2 (and for MTP layers via
          ``mtp.py``); 3..44 are MoE.
        * The activation pair is read from
          ``SWIGLU_LIMITS[layer_idx]`` /
          ``SWIGLU_LIMITS_SHARED[layer_idx]``.
    """
    full_attn = is_full_attention(layer_idx)
    moe = is_moe_layer(layer_idx)
    if full_attn and not moe:
        return decode_layer_full_dense, "full_dense"
    if (not full_attn) and (not moe):
        return decode_layer_swa_dense, "swa_dense"
    routed_lim = float(SWIGLU_LIMITS[layer_idx])
    shared_lim = float(SWIGLU_LIMITS_SHARED[layer_idx])
    table = _KIND_BY_PAIR_FULL_NAMES if full_attn else _KIND_BY_PAIR_SWA_NAMES
    try:
        kind, prog_name = table[(routed_lim, shared_lim)]
    except KeyError as err:
        raise ValueError(
            f"Unsupported (routed={routed_lim}, shared={shared_lim}) "
            f"for layer {layer_idx}",
        ) from err
    import sys as _sys  # noqa: PLC0415
    prog = getattr(_sys.modules[__name__], prog_name)
    return prog, kind


# =============================================================================
# CLI entry — dispatcher smoke test (no kernel runtime needed).
# =============================================================================
if __name__ == "__main__":
    for li in range(0, 45):
        prog, kind = select_decode_layer(li)
        assert prog is not None
        assert isinstance(kind, str)
    print("[dispatcher] all 45 main layers resolve OK")

    prog_l0, kind_l0 = select_decode_layer(0)
    prog_l1, kind_l1 = select_decode_layer(1)
    prog_l3, kind_l3 = select_decode_layer(3)
    prog_l4, kind_l4 = select_decode_layer(4)
    prog_l44, kind_l44 = select_decode_layer(44)

    assert kind_l0 == "full_dense", f"layer 0 -> {kind_l0}"
    assert kind_l1 == "swa_dense", f"layer 1 -> {kind_l1}"
    assert kind_l3 == "swa_moe_silu_silu", f"layer 3 -> {kind_l3}"
    assert kind_l4 == "full_moe_silu_silu", f"layer 4 -> {kind_l4}"
    assert kind_l44 == "full_moe_swiglu7_swiglu16", f"layer 44 -> {kind_l44}"
    assert prog_l3 is decode_layer_swa_moe_silu_silu
    assert prog_l4 is decode_layer_full_moe_silu_silu
    assert prog_l44 is decode_layer_full_moe_swiglu7_swiglu16

    print(
        "[dispatcher] layer 0 -> full_dense; layer 1 -> swa_dense; "
        "layer 3 -> swa_moe_silu_silu; layer 4 -> full_moe_silu_silu; "
        "layer 44 -> full_moe_swiglu7_swiglu16",
    )

    # select_moe_block (from moe.py) must agree with the embedded MoE
    # program in each of decode_layer's MoE specialisations.
    for li in (3, 4, 43, 44):
        moe_cls = select_moe_block(li)
        assert moe_cls is not None
    print("[dispatcher] select_moe_block(3..44) resolves OK")


__all__ = [
    "decode_layer_full_dense",
    "decode_layer_swa_dense",
    "decode_layer_full_moe",
    "decode_layer_swa_moe",
    "decode_layer_full_moe_silu_silu",
    "decode_layer_full_moe_swiglu7_silu",
    "decode_layer_full_moe_swiglu7_swiglu16",
    "decode_layer_swa_moe_silu_silu",
    "decode_layer_swa_moe_swiglu7_silu",
    "decode_layer_swa_moe_swiglu7_swiglu16",
    "select_decode_layer",
    "_dense_mlp_body_tp",
    "_build_decode_layer_dense_program",
    "_build_decode_layer_moe_program",
]
