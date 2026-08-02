# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 prefill MoE driver — TP=EP=8 (Phase 6).

Adapter that runs the existing Wave-2 ``EpTpMoE`` pipeline (gate →
dispatch → expert_routed → expert_shared+TP-AR → combine) on the
prefill token shape ``T_prefill = PREFILL_BATCH * PREFILL_SEQ = 128``.

T-axis decision (option b — thin prefill-T adapter)
----------------------------------------------------
Every Wave-2 MoE building block (``gate.py``, ``dispatch.py``,
``expert_routed.py``, ``expert_shared.py``, ``combine.py``, ``moe.py``)
hard-codes ``T = BATCH = 16`` as a module-level Python constant. None of
the ``@pl.jit.inline`` / ``@pl.program`` signatures accept a runtime
``T``. The team-lead brief lists two options:

  (a) re-specialise the MoE program with a different T axis;
  (b) wrap with a thin prefill-T adapter that chunks the prefill
      ``T_prefill`` into ``BATCH``-sized tiles and invokes the existing
      ``EpTpMoE`` program once per tile.

Option (a) requires editing each of the six locked Wave-2 files (the
brief's "DO NOT EDIT" list); option (b) reuses the locked decode MoE
programs unchanged. We pick **(b)**:

  ``PREFILL_TILE_COUNT = PREFILL_T // BATCH``  (8 tiles for B=1, S=128)

For each tile we slice the prefill ``[PREFILL_T, HIDDEN]`` input into a
``[BATCH, HIDDEN]`` window, drive one ``EpTpMoE.chip_orch`` call, then
stitch the per-tile ``moe_out[BATCH, HIDDEN]`` slabs back into a
``[PREFILL_T, HIDDEN]`` result.

This is **routing-equivalent** to a full prefill-T MoE pass: routing is
purely per-token (gate runs row-independent sigmoid + bias + top-K), so
running ``T_prefill`` routes over ``PREFILL_TILE_COUNT`` independent
``BATCH``-sized invocations yields identical per-token expert outputs
to a single ``T = PREFILL_T`` pass. The trade-off is one EP a2a per
tile instead of one EP a2a for the whole prefill batch, but at
prefill ``B=1``, ``S=128`` the routing cost is negligible compared
to the prefill attention matmul cost.

Per-card weight bundle (host weight loader contract)
----------------------------------------------------
Same as the decode-side ``moe.py`` (the underlying ``EpTpMoE`` program
is reused unchanged):

  Replicated on every rank:
    * ``gate_w[HIDDEN, MOE_NUM_EXPERTS=288]`` FP32
    * ``router_bias[MOE_NUM_EXPERTS=288]`` FP32

  EP-sliced (rank ``r`` gets global expert ids
  ``[r * 36 .. (r + 1) * 36)``):
    * ``w_gate_r[MOE_NUM_EXPERTS_LOCAL=36, HIDDEN, MOE_INTERMEDIATE=1280]`` BF16
    * ``w_up_r  [MOE_NUM_EXPERTS_LOCAL=36, HIDDEN, MOE_INTERMEDIATE=1280]`` BF16
    * ``w_down_r[MOE_NUM_EXPERTS_LOCAL=36, MOE_INTERMEDIATE=1280, HIDDEN]`` BF16

  TP-sliced (rank ``r`` gets intermediate lanes
  ``[r * 160 .. (r + 1) * 160)``):
    * ``w_gate_s[HIDDEN, SHARE_EXPERT_DIM_LOCAL=160]`` BF16
    * ``w_up_s  [HIDDEN, SHARE_EXPERT_DIM_LOCAL=160]`` BF16
    * ``w_down_s[SHARE_EXPERT_DIM_LOCAL=160, HIDDEN]`` BF16

Window contract
---------------
The wrapped MoE program allocates its own per-tile windows inside its
host_orch (see ``moe.py``'s ``EpTpMoE.host_orch``). The prefill adapter
runs a Python compile-time loop over ``PREFILL_TILE_COUNT`` tiles, so
each tile gets a fresh signal-window slot — no AtomicAdd ring-step
counters collide across tiles.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from .config import (
    BATCH,
    EP_WORLD_SIZE,
    HIDDEN,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    MOE_TOP_K,
    SHARE_EXPERT_DIM_LOCAL,
    SWIGLU_LIMITS,
    SWIGLU_LIMITS_SHARED,
    TP_WORLD_SIZE,
)
from .dispatch import LOCAL_RECV_MAX as DISPATCH_LOCAL_RECV_MAX
from .dispatch import PER_RANK_BUCKETS as DISPATCH_PER_RANK_BUCKETS
from .prefill_qkv_proj_rope import PREFILL_BATCH, PREFILL_SEQ, PREFILL_T


# Compile-time tile count.
PREFILL_TILE_COUNT = PREFILL_T // BATCH
assert PREFILL_T % BATCH == 0, (
    f"PREFILL_T={PREFILL_T} must be a multiple of BATCH={BATCH} so the "
    "prefill-T adapter can chunk into whole decode-T tiles"
)

# Re-exports for caller signatures.
N_RANKS = TP_WORLD_SIZE
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
N_EXPERTS = MOE_NUM_EXPERTS
TOPK = MOE_TOP_K
INTER = MOE_INTERMEDIATE
SH_INTER_LOCAL = SHARE_EXPERT_DIM_LOCAL
LOCAL_RECV_MAX = DISPATCH_LOCAL_RECV_MAX
PER_RANK_BUCKETS = DISPATCH_PER_RANK_BUCKETS
N_ROUTES_PER_RANK = BATCH * TOPK
SH_TP_CHUNK = HIDDEN // TP_WORLD_SIZE


# -----------------------------------------------------------------------------
# Phase X.8 — kernel-internal constants for the inlined MoE method bodies.
#
# pypto frontend rejects ``self._embedded_moe_cls().chip_orch(...)``
# (instantiating a ``@pl.program`` inside another ``@pl.program`` body is not a
# supported feature), so the entire body of ``EpTpMoE`` (from ``moe.py``) —
# every ``@pl.function`` method plus the ``chip_orch`` body — is inlined
# directly into ``PrefillMoE``. The originals in ``moe.py`` remain intact.
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
assert INTER % ROUTED_GATE_N_CHUNK == 0
assert INTER % ROUTED_DOWN_K_CHUNK == 0

# Shared-expert kernel constants — mirrors expert_shared.py / moe.SHARED_*.
SHARED_GATE_K_CHUNK = 256
SHARED_GATE_N_CHUNK = SH_INTER_LOCAL  # 160 — one N tile covers the slice
SHARED_DOWN_K_CHUNK = SH_INTER_LOCAL  # 160 — one K tile covers the slice
SHARED_DOWN_N_CHUNK = 256
assert HIDDEN % SHARED_GATE_K_CHUNK == 0
assert HIDDEN % SHARED_DOWN_N_CHUNK == 0


# =============================================================================
# Adapter @pl.program — prefill-T wrapping of an EpTpMoE specialisation.
# =============================================================================
def _build_prefill_moe_program(
    routed_lim: float,
    shared_lim: float,
    *,
    tp_size: int = TP_WORLD_SIZE,
):
    """Build a @pl.program that runs the MoE pipeline tile-by-tile.

    Phase X.8: the entire ``EpTpMoE`` body (every ``@pl.function`` method plus
    the ``chip_orch`` body from ``moe.py``) is inlined into ``PrefillMoE``.
    The frontend rejects ``self._embedded_moe_cls().chip_orch(...)``
    (instantiating a ``@pl.program`` inside another ``@pl.program`` body is
    not a supported feature). The activation choice is baked at factory build
    time via Python closure constants (``_routed_swiglu_step`` /
    ``_shared_swiglu_step``).

    Constructed inside a Python factory so the module imports cleanly even on
    hosts without a pypto runtime (deferred-build pattern, same as
    ``moe.py``'s reference).
    """
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must be divisible by tp_size={tp_size}"
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

    # Closure aliases used by the inlined method bodies — match the names used
    # in the lifted ``moe.EpTpMoE`` bodies so the type annotations resolve
    # cleanly. ``T`` here is the per-tile token count (= BATCH = decode-T).
    n_ranks = tp_size
    n_local_experts = N_LOCAL_EXPERTS
    inter = INTER
    sh_inter_local = SH_INTER_LOCAL
    sh_tp_chunk = SH_TP_CHUNK
    local_recv_max = LOCAL_RECV_MAX
    n_routes_per_rank = N_ROUTES_PER_RANK
    per_rank_buckets = PER_RANK_BUCKETS

    @pl.program
    class PrefillMoE:
        # Phase X.8: the entire ``EpTpMoE`` body (gate / dispatch /
        # expert_routed / expert_shared / combine plus all Inline helpers and
        # the chip_orch body) is inlined into this class. The frontend rejects
        # ``self._embedded_moe_cls().chip_orch(...)``; the originals in
        # ``moe.py`` remain intact.

        # ---------- Collective: TP all_reduce (lifted from moe.py) ----
        # Pull-side ring all-reduce body, t_rows=BATCH (per-tile token count),
        # d_cols=HIDDEN, group_size=tp_size. Used by expert_shared_step.
        @pl.function(type=pl.FunctionType.InCore)
        def tp_all_reduce(
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
                    expected=group_size - 1 + step + 1,
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
            x: pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16],
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
            moe_out: pl.Out[pl.Tensor[[PREFILL_T, HIDDEN], pl.BF16]],
            pub_counts: pld.DistributedTensor[
                [n_ranks * n_ranks, n_local_experts], pl.INT32
            ],
            count_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
            recv_x: pld.DistributedTensor[
                [local_recv_max, HIDDEN], pl.BF16
            ],
            data_done_sig: pld.DistributedTensor[[n_ranks, 1], pl.INT32],
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
            my_rank: pl.Scalar[pl.INT32],
        ):
            """Run the inlined MoE pipeline once per BATCH-sized tile.

            Phase X.8: per-tile invocation calls the inlined per-stage
            methods (``self.gate_step`` → ``self.dispatch_step`` → ... →
            ``self.combine_step``) directly instead of an embedded
            ``@pl.program`` instance.
            """
            for tile_idx in pl.unroll(PREFILL_TILE_COUNT):
                t_lo = tile_idx * BATCH
                tile_x = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="prefill_moe_tile_in",
                ):
                    tile_x = pl.assemble(
                        tile_x,
                        pl.slice(x, [BATCH, HIDDEN], [t_lo, 0]),
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
            return moe_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            x: pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16],
            gate_w: pl.Tensor[[tp_size, HIDDEN, N_EXPERTS], pl.FP32],
            router_bias: pl.Tensor[[tp_size, N_EXPERTS], pl.FP32],
            w_gate_r: pl.Tensor[
                [tp_size, N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16
            ],
            w_up_r: pl.Tensor[
                [tp_size, N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16
            ],
            w_down_r: pl.Tensor[
                [tp_size, N_LOCAL_EXPERTS, INTER, HIDDEN], pl.BF16
            ],
            w_gate_s: pl.Tensor[
                [tp_size, HIDDEN, SH_INTER_LOCAL], pl.BF16
            ],
            w_up_s: pl.Tensor[
                [tp_size, HIDDEN, SH_INTER_LOCAL], pl.BF16
            ],
            w_down_s: pl.Tensor[
                [tp_size, SH_INTER_LOCAL, HIDDEN], pl.BF16
            ],
            moe_out: pl.Out[
                pl.Tensor[[tp_size, PREFILL_T, HIDDEN], pl.BF16]
            ],
        ):
            pub_counts_buf = pld.alloc_window_buffer(
                N_RANKS * N_RANKS * N_LOCAL_EXPERTS * 4,
            )
            count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
            recv_x_buf = pld.alloc_window_buffer(
                LOCAL_RECV_MAX * HIDDEN * 2,
            )
            data_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
            sh_tmp_buf = pld.alloc_window_buffer(BATCH * SH_TP_CHUNK * 2)
            sh_sig_buf = pld.alloc_window_buffer(N_RANKS * 4)
            src_route_buf = pld.alloc_window_buffer(
                N_RANKS * N_LOCAL_EXPERTS * N_ROUTES_PER_RANK * 4,
            )
            route_pub_buf = pld.alloc_window_buffer(N_RANKS * 4)
            routed_y_window_buf = pld.alloc_window_buffer(
                N_ROUTES_PER_RANK * HIDDEN * 2,
            )
            combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)

            for r in pl.range(pld.world_size()):
                pub_counts = pld.window(
                    pub_counts_buf,
                    [N_RANKS * N_RANKS, N_LOCAL_EXPERTS], dtype=pl.INT32,
                )
                count_done_sig = pld.window(
                    count_done_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                recv_x = pld.window(
                    recv_x_buf,
                    [LOCAL_RECV_MAX, HIDDEN], dtype=pl.BF16,
                )
                data_done_sig = pld.window(
                    data_done_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                sh_tmp_window = pld.window(
                    sh_tmp_buf, [BATCH, SH_TP_CHUNK], dtype=pl.BF16,
                )
                sh_signal_window = pld.window(
                    sh_sig_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                src_route_table = pld.window(
                    src_route_buf,
                    [N_RANKS, N_LOCAL_EXPERTS, N_ROUTES_PER_RANK],
                    dtype=pl.INT32,
                )
                route_pub_sig = pld.window(
                    route_pub_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                routed_y_buf = pld.window(
                    routed_y_window_buf,
                    [N_ROUTES_PER_RANK, HIDDEN], dtype=pl.BF16,
                )
                combine_done_sig = pld.window(
                    combine_done_buf, [N_RANKS, 1], dtype=pl.INT32,
                )
                self.chip_orch(
                    x[r], gate_w[r], router_bias[r],
                    w_gate_r[r], w_up_r[r], w_down_r[r],
                    w_gate_s[r], w_up_s[r], w_down_s[r],
                    moe_out[r],
                    pub_counts, count_done_sig,
                    recv_x, data_done_sig,
                    sh_tmp_window, sh_signal_window,
                    src_route_table, route_pub_sig,
                    routed_y_buf, combine_done_sig,
                    r,
                    device=r,
                )

    return PrefillMoE


# Lazy specialisation cache — same rationale as moe.py: building the
# @pl.program at module-import time triggers the AST parser before the
# runtime context is wired. PEP 562 __getattr__ keeps the public names
# importable while deferring construction to first access.
_PREFILL_MOE_CACHE: dict[tuple[float, float], object] = {}

# Map public name -> (routed_lim, shared_lim). Phase X.8: the prefill MoE
# program no longer wraps a separate ``EpTpMoE`` @pl.program — the activation
# choice drives the inlined methods directly via factory closure constants.
_LAZY_PREFILL_NAMES = {
    "PrefillMoE_silu_silu": (0.0, 0.0),
    "PrefillMoE_swiglu7_silu": (7.0, 0.0),
    "PrefillMoE_swiglu7_swiglu16": (7.0, 16.0),
}


def _get_prefill_moe(routed_lim: float, shared_lim: float):
    key = (routed_lim, shared_lim)
    prog = _PREFILL_MOE_CACHE.get(key)
    if prog is None:
        prog = _build_prefill_moe_program(routed_lim, shared_lim)
        _PREFILL_MOE_CACHE[key] = prog
    return prog


def __getattr__(name: str):  # PEP 562
    if name in _LAZY_PREFILL_NAMES:
        return _get_prefill_moe(*_LAZY_PREFILL_NAMES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def select_prefill_moe_block(layer_idx: int):
    """Return the prefill MoE program class for ``layer_idx``.

    Mirrors ``moe.select_moe_block`` but emits the prefill-T wrapper class.
    The underlying ``@pl.program`` class is built lazily on first access and
    cached.
    """
    routed_lim = float(SWIGLU_LIMITS[layer_idx])
    shared_lim = float(SWIGLU_LIMITS_SHARED[layer_idx])
    if routed_lim == 0.0 and shared_lim == 0.0:
        return _get_prefill_moe(0.0, 0.0)
    if routed_lim == 7.0 and shared_lim == 0.0:
        return _get_prefill_moe(7.0, 0.0)
    if routed_lim == 7.0 and shared_lim == 16.0:
        return _get_prefill_moe(7.0, 16.0)
    raise ValueError(
        f"Unsupported (routed={routed_lim}, shared={shared_lim}) for "
        f"layer {layer_idx}; expected one of (0, 0), (7, 0), (7, 16).",
    )


# =============================================================================
# Distributed-mock harness — torch-only.
#
# The harness validates the equivalence of the prefill-T adapter to a
# full prefill-T MoE pass: routing decisions are per-token, so chunked
# BATCH-sized invocations of the decode MoE pipeline reproduce the same
# per-token expert outputs as a single PREFILL_T-sized pass.
# =============================================================================
def _torch_moe_ref(
    routed_lim, shared_lim,
    x, gate_w, router_bias,
    w_gate_r, w_up_r, w_down_r,
    w_gate_s, w_up_s, w_down_s,
):
    """Pure-torch global MoE reference (same math as moe._torch_moe_ref)."""
    import torch
    import torch.nn.functional as F

    t = x.shape[0]
    logits = x.float() @ gate_w.float()
    score = torch.sigmoid(logits)
    biased = score + router_bias.float().view(1, -1)
    indices = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
    topk_vals = torch.gather(score, dim=-1, index=indices.long())
    weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)) * 3.0
    weights_bf = weights.to(torch.bfloat16).float()

    routed_acc = torch.zeros(t, HIDDEN, dtype=torch.float32)
    for ti in range(t):
        for k in range(TOPK):
            eid = int(indices[ti, k].item())
            x_row = x[ti : ti + 1, :].float()
            gate_a = x_row @ w_gate_r[eid].float()
            up_a = x_row @ w_up_r[eid].float()
            if routed_lim > 0.0:
                silu_g = F.silu(gate_a).clamp(max=routed_lim)
                up_c = up_a.clamp(min=-routed_lim, max=routed_lim)
                h = silu_g * up_c
            else:
                h = F.silu(gate_a) * up_a
            h_bf = h.to(torch.bfloat16).float()
            y = h_bf @ w_down_r[eid].float()
            routed_acc[ti, :] += weights_bf[ti, k] * y[0]

    sh_gate = x.float() @ w_gate_s.float()
    sh_up = x.float() @ w_up_s.float()
    if shared_lim > 0.0:
        sh_silu = F.silu(sh_gate).clamp(max=shared_lim)
        sh_up_c = sh_up.clamp(min=-shared_lim, max=shared_lim)
        sh_h = sh_silu * sh_up_c
    else:
        sh_h = F.silu(sh_gate) * sh_up
    sh_h_bf = sh_h.to(torch.bfloat16).float()
    sh_out = sh_h_bf @ w_down_s.float()

    return (sh_out + routed_acc).to(torch.bfloat16)


def _distributed_mock_check(layer_idx: int = 3, seed: int = 0):
    """Mock prefill-T MoE; verify tile-by-tile equivalence to a full pass.

    The test runs a torch reference for the full ``PREFILL_T`` input,
    then runs the same reference tile-by-tile on ``PREFILL_TILE_COUNT``
    chunks of ``BATCH`` rows each and concatenates. The per-element
    pass_rate is reported.
    """
    import torch

    routed_lim = float(SWIGLU_LIMITS[layer_idx])
    shared_lim = float(SWIGLU_LIMITS_SHARED[layer_idx])

    gen = torch.Generator().manual_seed(seed)
    inter_mock = 256
    sh_lane_mock = 64
    sh_total = sh_lane_mock * TP_WORLD_SIZE

    x = (torch.randn(PREFILL_T, HIDDEN, generator=gen) * 0.3).to(
        torch.bfloat16,
    )
    gate_w = (
        torch.randn(HIDDEN, N_EXPERTS, generator=gen) / HIDDEN ** 0.5
    ).float()
    router_bias = (
        torch.randn(N_EXPERTS, generator=gen) * 0.05
    ).float()
    w_gate_r = (
        torch.randn(N_EXPERTS, HIDDEN, inter_mock, generator=gen)
        / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_up_r = (
        torch.randn(N_EXPERTS, HIDDEN, inter_mock, generator=gen)
        / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_down_r = (
        torch.randn(N_EXPERTS, inter_mock, HIDDEN, generator=gen)
        / inter_mock ** 0.5
    ).to(torch.bfloat16)
    w_gate_s = (
        torch.randn(HIDDEN, sh_total, generator=gen) / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_up_s = (
        torch.randn(HIDDEN, sh_total, generator=gen) / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_down_s = (
        torch.randn(sh_total, HIDDEN, generator=gen) / sh_total ** 0.5
    ).to(torch.bfloat16)

    full_ref = _torch_moe_ref(
        routed_lim, shared_lim,
        x, gate_w, router_bias,
        w_gate_r, w_up_r, w_down_r,
        w_gate_s, w_up_s, w_down_s,
    )

    tiled = torch.zeros_like(full_ref)
    for tile_idx in range(PREFILL_TILE_COUNT):
        lo = tile_idx * BATCH
        hi = lo + BATCH
        tile_y = _torch_moe_ref(
            routed_lim, shared_lim,
            x[lo:hi, :], gate_w, router_bias,
            w_gate_r, w_up_r, w_down_r,
            w_gate_s, w_up_s, w_down_s,
        )
        tiled[lo:hi, :] = tile_y

    diff = (tiled.float() - full_ref.float()).abs()
    tol = 5e-2 + 5e-2 * full_ref.float().abs()
    matches = int((diff <= tol).sum().item())
    return matches / (PREFILL_T * HIDDEN)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 prefill MoE driver — TP=EP=8. Adapter that runs the "
            "Wave-2 EpTpMoE pipeline tile-by-tile on the prefill T="
            f"{PREFILL_T} token shape (chunked into "
            f"{PREFILL_TILE_COUNT} BATCH={BATCH}-sized tiles)."
        ),
    )
    parser.add_argument("--layers", default="3,4,44",
                        help="Comma-separated layer indices.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    worst = 1.0
    for li in layers:
        _ = select_prefill_moe_block(li)
        pass_rate = _distributed_mock_check(li, seed=args.seed + li)
        print(
            f"[prefill_moe.py] layer {li}: "
            f"routed={SWIGLU_LIMITS[li]}, shared={SWIGLU_LIMITS_SHARED[li]}, "
            f"pass_rate={pass_rate:.4f}"
        )
        worst = min(worst, pass_rate)
    if worst < 0.97:
        raise SystemExit(1)


__all__ = [
    "PrefillMoE_silu_silu",
    "PrefillMoE_swiglu7_silu",
    "PrefillMoE_swiglu7_swiglu16",
    "select_prefill_moe_block",
    "_build_prefill_moe_program",
    "PREFILL_BATCH",
    "PREFILL_SEQ",
    "PREFILL_T",
    "PREFILL_TILE_COUNT",
    "N_RANKS",
    "N_LOCAL_EXPERTS",
    "N_EXPERTS",
    "TOPK",
    "INTER",
    "SH_INTER_LOCAL",
    "LOCAL_RECV_MAX",
    "N_ROUTES_PER_RANK",
    "SH_TP_CHUNK",
    "EP_WORLD_SIZE",
]
