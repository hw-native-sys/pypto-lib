# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI marker: run on >=2 NPUs via $DEVICE_RANGE instead of single $DEVICE_ID
"""DeepSeek-V4 MoE end-to-end (decode, 2-rank EP single-layer).

Mirrors moe.py but assembles the 7 stages as a ``@pl.program`` with explicit
cross-rank dispatch/combine over an HCCL window scratch, following the
``test_l3_ep_dispatch_combine`` reference protocol. The compute kernels
(hc_pre / gate / expert_shared / expert_routed / hc_post) are reused as
``@pl.jit.inline`` calls inside ``InCore`` methods.

Demo sizing (set in the module preamble below):
  N_RANKS = 2, DEMO preset, EP_WORLD_SIZE = 2, EP_ROUTING_GLOBAL = True
  T = 128, D = 4096, N_LOCAL = 4, RECV_MAX = 256, TOPK = 2,
  N_EXPERTS_GLOBAL = 8, MOE_INTER = 4096
"""


# === Module preamble: override config BEFORE importing sub-kernels ==========
# Sub-kernels (hc_pre / gate / expert_routed / ...) bind preset constants at
# module-import time via ``from config import FLASH as M``. By the time those
# modules execute their first line, the overrides below must already be in
# place — otherwise they'd capture FLASH and EP_WORLD_SIZE=16 instead.
import dataclasses

import config

# Use DEMO sizing instead of FLASH:
#   FLASH's n_routed_experts=256 combined with EP_ROUTING_GLOBAL=True makes
#   gate.py's matmul allocate a single gate_w[:, 0:GATE_D_CHUNK=512] slice of
#   256 × 512 × 4 = 512 KB, which saturates the cube Mat buffer once the LHS
#   x slice is added. DEMO's n_routed_experts=16 keeps the slices comfortable.
#   gate.py does not chunk along the N_EXPERTS dim today; supporting FLASH-EP
#   would require that change.
#
# Override num_hash_layers 0 -> 1:
#   DEMO ships with num_hash_layers=0. gate.py picks the hash routing branch
#   when ``layer_id < N_HASH_LAYERS``, so with num_hash_layers=0 every
#   non-negative layer_id (including the CLI default of 0) falls into the
#   ELSE branch — the sort routing path. That path has an independent
#   precision regression (single-card ``python moe.py --layer-id 3`` reproduces
#   the same x_next mismatch, ratio_reldiff ≈ 7%), unrelated to the EP
#   changes. Bumping num_hash_layers to 1 makes layer_id=0 satisfy 0 < 1 and
#   pick hash, so EpMoE runs the validated route end-to-end. Pass
#   ``--layer-id 1`` (or any layer_id >= num_hash_layers) to exercise the sort
#   path explicitly once that regression is investigated.
#
# dataclasses.replace creates a fresh DeepSeekV4Config copy, so the DEMO
# preset in config.py stays untouched for any other importer.
config.FLASH = dataclasses.replace(config.DEMO, num_hash_layers=1)
config.EP_WORLD_SIZE = 2
config.EP_ROUTING_GLOBAL = True
config.RECV_MAX = (
    config.DECODE_BATCH * config.DECODE_SEQ * config.FLASH.num_experts_per_tok
    // (config.FLASH.n_routed_experts // config.EP_WORLD_SIZE)
) * config.RECV_SAFETY

# Now safe to import the compute sub-kernels.
import pypto.language as pl  # noqa: E402
import pypto.language.distributed as pld  # noqa: E402
from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: E402

from hc_pre import hc_pre_inline as hc_pre  # noqa: E402
from hc_post import hc_post_inline as hc_post  # noqa: E402
from gate import gate_inline as gate  # noqa: E402
from expert_shared import expert_shared_inline as expert_shared  # noqa: E402
from expert_routed import expert_routed_inline as expert_routed  # noqa: E402


# === Demo / EP constants ====================================================
M = config.FLASH  # alias (now DEMO after the override above)
N_RANKS = config.EP_WORLD_SIZE
B = config.DECODE_BATCH
S = config.DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MOE_INTER = M.moe_intermediate_size
N_EXPERTS_GLOBAL = M.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
RECV_MAX = config.RECV_MAX
N_ROUTES = T * TOPK

# Padding widths required by tile vector ops (32 B minimum tile).
W_PAD = 8  # FP32 weight/scale tile width
IDX_PAD = 8  # INT32 r_route tile width

# Single-program sanity asserts catch preset mismatches early.
assert N_RANKS == 2, "moe_ep demo is wired for 2 ranks"
assert TOPK == 2, "moe_ep demo assumes TOPK == 2 for combine reduce"
assert N_EXPERTS_GLOBAL == N_RANKS * N_LOCAL


# === Program ================================================================
def _build_ep_moe_program():
    """Deferred-build pattern (matches test_l3_ep_dispatch_combine): keeps the
    module importable even if the embedded body trips parser at collection."""

    @pl.program
    class EpMoE:
        # --- Steps 1a..1d: pre-dispatch local compute -------------------
        # Each sub-kernel gets its own InCore method to avoid leaking inline
        # local vars across kernels (e.g. both hc_pre and gate use ``sq_sum``,
        # which trips the strict-SSA InCore parser if inlined back-to-back in
        # the same scope).
        @pl.function(type=pl.FunctionType.Inline)
        def hc_pre_step(
            self,
            x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
            hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
            hc_ffn_scale: pl.Tensor[[3], pl.FP32],
            hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
            x_mixed: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
            post_ffn: pl.Out[pl.Tensor[[B, S, HC_MULT], pl.FP32]],
            comb_ffn: pl.Out[pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[B, S, D], pl.BF16],
            pl.Tensor[[B, S, HC_MULT], pl.FP32],
            pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
        ]:
            x_mixed = hc_pre(
                x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                x_mixed, post_ffn, comb_ffn,
            )
            return x_mixed, post_ffn, comb_ffn

        @pl.function(type=pl.FunctionType.Inline)
        def gate_step(  # noqa: PLR0913
            self,
            x_mixed: pl.Tensor[[B, S, D], pl.BF16],
            norm_w: pl.Tensor[[D], pl.FP32],
            gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
            gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
            tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
            input_ids: pl.Tensor[[B, S], pl.INT64],
            x_norm: pl.Out[pl.Tensor[[T, D], pl.BF16]],
            x_norm_i8: pl.Out[pl.Tensor[[T, D], pl.INT8]],
            x_norm_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
            indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
            weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
            layer_id: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[T, D], pl.BF16],
            pl.Tensor[[T, D], pl.INT8],
            pl.Tensor[[T, 1], pl.FP32],
            pl.Tensor[[T, TOPK], pl.INT32],
            pl.Tensor[[T, TOPK], pl.FP32],
        ]:
            weights = gate(
                x_mixed,
                norm_w, gate_w, gate_bias,
                layer_id,
                tid2eid, input_ids,
                x_norm, x_norm_i8, x_norm_scale, indices, weights,
            )
            return x_norm, x_norm_i8, x_norm_scale, indices, weights

        @pl.function(type=pl.FunctionType.Inline)
        def expert_shared_step(
            self,
            x_norm_i8: pl.Tensor[[T, D], pl.INT8],
            x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
            shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
            shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
            shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
            shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
            shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
            shared_w2_scale: pl.Tensor[[D], pl.FP32],
            sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
        ) -> pl.Tensor[[T, D], pl.BF16]:
            sh = expert_shared(
                x_norm_i8, x_norm_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                sh,
            )
            return sh

        @pl.function(type=pl.FunctionType.Inline)
        def pack_step(
            self,
            x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
            weights: pl.Tensor[[T, TOPK], pl.FP32],
            scale_padded: pl.Out[pl.Tensor[[T, W_PAD], pl.FP32]],
            weight_padded: pl.Out[pl.Tensor[[N_ROUTES, W_PAD], pl.FP32]],
            r_route_padded: pl.Out[pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32]],
        ) -> tuple[
            pl.Tensor[[T, W_PAD], pl.FP32],
            pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
        ]:
            # Pad scale/weight/r_route to 32B-aligned tile widths so dispatch
            # can use single TILE remote_store pushes. Columns 1..PAD-1 stay 0
            # so the stage-out row_sum trick recovers column 0 exactly.
            for t in pl.range(T):
                s_val = pl.read(x_norm_scale, [t, 0])
                pl.write(scale_padded, [t, 0], s_val)
                for sp in pl.range(1, W_PAD):
                    pl.write(scale_padded, [t, sp], 0.0)
                for k in pl.range(TOPK):
                    w_val = pl.read(weights, [t, k])
                    r_route = pl.cast(t * TOPK + k, pl.INT32)
                    pl.write(weight_padded, [t * TOPK + k, 0], w_val)
                    for wp in pl.range(1, W_PAD):
                        pl.write(weight_padded, [t * TOPK + k, wp], 0.0)
                    pl.write(r_route_padded, [t * TOPK + k, 0], r_route)
                    for ip in pl.range(1, IDX_PAD):
                        pl.write(
                            r_route_padded,
                            [t * TOPK + k, ip],
                            pl.cast(0, pl.INT32),
                        )
            return scale_padded, weight_padded, r_route_padded

        # --- Step 2: dispatch (cross-rank #1) ----------------------------
        # 1:1 of test_l3_ep_dispatch_combine.dispatch_step but with FOUR push
        # channels: x_i8 INT8, scale FP32, weight FP32, r_route INT32.
        @pl.function(type=pl.FunctionType.InCore)
        def dispatch_step(  # noqa: PLR0913, PLR0915
            self,
            indices: pl.Tensor[[T, TOPK], pl.INT32],
            x_norm_i8: pl.Tensor[[T, D], pl.INT8],
            scale_padded: pl.Tensor[[T, W_PAD], pl.FP32],
            weight_padded: pl.Tensor[[N_ROUTES, W_PAD], pl.FP32],
            r_route_padded: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
            recv_x_out: pl.Out[pl.Tensor[[N_LOCAL * RECV_MAX, D], pl.INT8]],
            recv_scale_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32]],
            recv_w_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32]],
            recv_r_route_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32]],
            recv_count_out: pl.Out[pl.Tensor[[N_LOCAL, 1], pl.INT32]],
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
            count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
            recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
            recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
            recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> tuple[
            pl.Tensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
            pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
            pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
            pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
            pl.Tensor[[N_LOCAL, 1], pl.INT32],
        ]:
            # ---------- histogram: scalar histogram on indices ----------
            send_counts = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
            for d in pl.range(N_RANKS):
                for e in pl.range(N_LOCAL):
                    send_counts[d * N_LOCAL + e] = 0

            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    d = eid // N_LOCAL
                    e = eid - d * N_LOCAL
                    cur = send_counts[d * N_LOCAL + e]
                    send_counts[d * N_LOCAL + e] = cur + 1

            # ---------- publish: TNOTIFY(AtomicAdd) ----------
            for peer in pl.range(N_RANKS):
                for d in pl.range(N_RANKS):
                    for e in pl.range(N_LOCAL):
                        v = send_counts[d * N_LOCAL + e]
                        if v != 0:
                            # Single-writer cell (each (src, d, e) is touched by
                            # exactly src=my_rank), so Set is sufficient.
                            pld.system.notify(
                                target=pub_counts,
                                peer=peer,
                                offsets=[my_rank * N_RANKS + d, e],
                                value=v,
                                op=pld.NotifyOp.Set,
                            )

            # ---------- count_done barrier ----------
            # First notify on this per-src cell; Set since only my_rank writes
            # offsets=[my_rank, 0] on each peer.
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=count_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=count_done,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # ---------- prefix_sum: my slot offset + total recv_count ----------
            my_slot_at_dst = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
            for d in pl.range(N_RANKS):
                for e in pl.range(N_LOCAL):
                    acc = pl.const(0, pl.INT32)
                    for s in pl.range(N_RANKS):
                        if s < my_rank:
                            acc = acc + pl.read(pub_counts, [s * N_RANKS + d, e])
                    my_slot_at_dst[d * N_LOCAL + e] = acc

            for e in pl.range(N_LOCAL):
                acc = pl.const(0, pl.INT32)
                for s in pl.range(N_RANKS):
                    acc = acc + pl.read(pub_counts, [s * N_RANKS + my_rank, e])
                pl.write(recv_count_out, [e, 0], acc)

            # ---------- payload_push: 4 channels per (t, k) ----------
            cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
            for d in pl.range(N_RANKS):
                for e in pl.range(N_LOCAL):
                    cursor[d * N_LOCAL + e] = 0

            for t in pl.range(T):
                for k in pl.range(TOPK):
                    eid = pl.read(indices, [t, k])
                    dst = eid // N_LOCAL
                    loc_e = eid - dst * N_LOCAL
                    bucket = dst * N_LOCAL + loc_e
                    cur_val = cursor[bucket]
                    slot_off = my_slot_at_dst[bucket]
                    slot = slot_off + cur_val
                    row = loc_e * RECV_MAX + slot
                    cursor[bucket] = cur_val + 1
                    r_route = t * TOPK + k

                    x_tile = pl.load(x_norm_i8, [t, 0], [1, D])
                    pld.tile.remote_store(x_tile, target=recv_x, peer=dst, offsets=[row, 0])

                    scale_tile = pl.load(scale_padded, [t, 0], [1, W_PAD])
                    pld.tile.remote_store(scale_tile, target=recv_scale, peer=dst, offsets=[row, 0])

                    w_tile = pl.load(weight_padded, [r_route, 0], [1, W_PAD])
                    pld.tile.remote_store(w_tile, target=recv_w, peer=dst, offsets=[row, 0])

                    idx_tile = pl.load(r_route_padded, [r_route, 0], [1, IDX_PAD])
                    pld.tile.remote_store(idx_tile, target=recv_r_route, peer=dst, offsets=[row, 0])

            # ---------- data_done barrier ----------
            # Reuse count_done signal cells: count phase bumps to 1, data
            # phase bumps to 2 (per-src cumulative count via AtomicAdd).
            # Avoids a separate window and keeps dispatch_step under the
            # MAX_TENSOR_ARGS=16 InCore limit.
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=count_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=count_done,
                        offsets=[src, 0],
                        expected=2,
                        cmp=pld.WaitCmp.Ge,
                    )

            # ---------- stage_out: window → host-backed ----------
            # recv_x: per-row [1, D] INT8 tile copy.
            for e in pl.range(N_LOCAL):
                for slot in pl.range(RECV_MAX):
                    row = e * RECV_MAX + slot
                    x_tile = pl.load(recv_x, [row, 0], [1, D])
                    pl.store(x_tile, [row, 0], recv_x_out)

            # recv_scale / recv_w: per-expert TROWSUM trick on [R, W_PAD] →
            # [R, 1] (column 0 is the real value; rest are zero), reshape and
            # store as [1, R].
            for e in pl.range(N_LOCAL):
                w_wide: pl.Tile[[RECV_MAX, W_PAD], pl.FP32] = pl.load(
                    recv_scale, [e * RECV_MAX, 0], [RECV_MAX, W_PAD]
                )
                tmp: pl.Tile[[RECV_MAX, 1], pl.FP32] = pl.tile.create(
                    [RECV_MAX, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                w_sum: pl.Tile[[RECV_MAX, 1], pl.FP32] = pl.tile.row_sum(w_wide, tmp)
                w_row: pl.Tile[[1, RECV_MAX], pl.FP32] = pl.tile.reshape(w_sum, [1, RECV_MAX])
                pl.store(w_row, [e, 0], recv_scale_out)

            for e in pl.range(N_LOCAL):
                w_wide2: pl.Tile[[RECV_MAX, W_PAD], pl.FP32] = pl.load(
                    recv_w, [e * RECV_MAX, 0], [RECV_MAX, W_PAD]
                )
                tmp2: pl.Tile[[RECV_MAX, 1], pl.FP32] = pl.tile.create(
                    [RECV_MAX, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                w_sum2: pl.Tile[[RECV_MAX, 1], pl.FP32] = pl.tile.row_sum(w_wide2, tmp2)
                w_row2: pl.Tile[[1, RECV_MAX], pl.FP32] = pl.tile.reshape(w_sum2, [1, RECV_MAX])
                pl.store(w_row2, [e, 0], recv_w_out)

            # recv_r_route: INT32 — scalar copy fallback (TROWSUM hangs on a2a3).
            for e in pl.range(N_LOCAL):
                for slot in pl.range(RECV_MAX):
                    r_val = pl.read(recv_r_route, [e * RECV_MAX + slot, 0])
                    pl.write(recv_r_route_out, [e, slot], r_val)

            return recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out

        # --- Step 3: expert_routed (local) -------------------------------
        @pl.function(type=pl.FunctionType.Inline)
        def expert_routed_step(  # noqa: PLR0913
            self,
            recv_x_out: pl.Tensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
            recv_scale_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
            recv_w_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32],
            recv_count_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
            routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
            routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
            routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
            routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
            routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
            routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
            recv_y: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16]],
        ) -> pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16]:
            # expert_routed takes [N_LOCAL, RECV_MAX, D]; reshape the staged
            # [N_LOCAL*RECV_MAX, D] flat view of recv_x_out to match its contract.
            recv_x_3d = pl.reshape(recv_x_out, [N_LOCAL, RECV_MAX, D])
            recv_y = expert_routed(
                recv_x_3d, recv_scale_out, recv_w_out, recv_count_out,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                recv_y,
            )
            return recv_y

        # --- Step 4a: combine (cross-rank #2 + reduce → ffn_out) ----------
        @pl.function(type=pl.FunctionType.InCore)
        def combine_step(  # noqa: PLR0913
            self,
            recv_y: pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.BF16],
            recv_r_route_out: pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32],
            sh: pl.Tensor[[T, D], pl.BF16],
            ffn_out: pl.Out[pl.Tensor[[B, S, D], pl.BF16]],
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
            routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
            combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[B, S, D], pl.BF16]:
            # ---------- push: TPUT recv_y rows to peer's routed_y_buf ----
            # For each dst rank and each local expert e on MY rank, the n rows
            # I sent to dst's expert e originally landed at slots
            # [src_off, src_off + n) on dst (src_off = Σ_{s<dst} counts).
            # Per-row pl.load uses 3D indexing on recv_y directly — DON'T
            # reshape recv_y to 2D first, that would tile.load the full
            # [N_LOCAL, RECV_MAX, D] tensor into Vec (≫ 192 KB).
            for dst in pl.range(N_RANKS):
                for e in pl.range(N_LOCAL):
                    n = pl.cast(pl.read(pub_counts, [dst * N_RANKS + my_rank, e]), pl.INDEX)
                    src_off = pl.const(0, pl.INT32)
                    for s in pl.range(N_RANKS):
                        if s < dst:
                            src_off = src_off + pl.read(pub_counts, [s * N_RANKS + my_rank, e])
                    src_off_idx = pl.cast(src_off, pl.INDEX)
                    for row in pl.range(n):
                        slot = src_off_idx + row
                        r_route = pl.read(recv_r_route_out, [e, slot])
                        y_tile_3d = pl.load(recv_y, [e, slot, 0], [1, 1, D])
                        y_tile = pl.reshape(y_tile_3d, [1, D])
                        pld.tile.remote_store(
                            y_tile, target=routed_y_buf, peer=dst, offsets=[r_route, 0]
                        )

            # ---------- combine_done barrier ----------
            # Single-writer per-src cell — Set, not Add.
            for peer in pl.range(N_RANKS):
                if peer != my_rank:
                    pld.system.notify(
                        target=combine_done,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=1,
                        op=pld.NotifyOp.Set,
                    )
            for src in pl.range(N_RANKS):
                if src != my_rank:
                    pld.system.wait(
                        signal=combine_done,
                        offsets=[src, 0],
                        expected=1,
                        cmp=pld.WaitCmp.Ge,
                    )

            # ---------- reduce: ffn_out[t] = sh[t] + Σ_k routed_y_buf[t*TOPK+k] ----
            # Per-row tile load + accumulate; don't reshape ffn_out [B,S,D]
            # to [T,D] up-front — that would tile.load the full 1 MB tensor
            # into Vec memory at once.
            for t in pl.range(T):
                b = t // S
                s_idx = t - b * S
                sh_tile = pl.load(sh, [t, 0], [1, D])
                acc = pl.cast(sh_tile, target_type=pl.FP32)
                for k in pl.range(TOPK):
                    y_k = pl.load(routed_y_buf, [t * TOPK + k, 0], [1, D])
                    y_k_fp = pl.cast(y_k, target_type=pl.FP32)
                    acc = pl.add(acc, y_k_fp)
                acc_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
                pl.store(acc_bf16, [b, s_idx, 0], ffn_out, shapes=[1, 1, D])
            return ffn_out

        # --- Step 4b: hc_post writes back the 4-stream hc residual --------
        @pl.function(type=pl.FunctionType.Inline)
        def hc_post_step(
            self,
            ffn_out: pl.Tensor[[B, S, D], pl.BF16],
            x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
            post_ffn: pl.Tensor[[B, S, HC_MULT], pl.FP32],
            comb_ffn: pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32],
            x_next: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
        ) -> pl.Tensor[[B, S, HC_MULT, D], pl.BF16]:
            x_next = hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
            return x_next

        # --- chip_orch: thread per-rank tensors through the 4 steps ------
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913
            self,
            # model inputs
            x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
            hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
            hc_ffn_scale: pl.Tensor[[3], pl.FP32],
            hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
            norm_w: pl.Tensor[[D], pl.FP32],
            gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
            gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
            tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
            input_ids: pl.Tensor[[B, S], pl.INT64],
            routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
            routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
            routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
            routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
            routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
            routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
            shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
            shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
            shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
            shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
            shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
            shared_w2_scale: pl.Tensor[[D], pl.FP32],
            # final output
            x_next: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
            # windows
            pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
            count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
            recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
            recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
            recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
            routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
            combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
            # scalars trailing — runtime TaskArgs requires all tensor args
            # before any scalar args (#1603-adjacent constraint).
            layer_id: pl.Scalar[pl.INT32],
            my_rank: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[B, S, HC_MULT, D], pl.BF16]:
            # All non-output intermediates allocate locally so the convert
            # pass sees them in the same scope as their producer / consumer,
            # mirroring single-card moe.py's @pl.jit.inline composition.
            x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
            post_ffn = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
            comb_ffn = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
            x_mixed, post_ffn, comb_ffn = self.hc_pre_step(
                x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                x_mixed, post_ffn, comb_ffn,
            )

            x_norm = pl.create_tensor([T, D], dtype=pl.BF16)
            x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
            x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
            indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
            weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
            x_norm, x_norm_i8, x_norm_scale, indices, weights = self.gate_step(
                x_mixed, norm_w, gate_w, gate_bias,
                tid2eid, input_ids,
                x_norm, x_norm_i8, x_norm_scale, indices, weights,
                layer_id,
            )

            sh = pl.create_tensor([T, D], dtype=pl.BF16)
            sh = self.expert_shared_step(
                x_norm_i8, x_norm_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                sh,
            )

            scale_padded = pl.create_tensor([T, W_PAD], dtype=pl.FP32)
            weight_padded = pl.create_tensor([N_ROUTES, W_PAD], dtype=pl.FP32)
            r_route_padded = pl.create_tensor([N_ROUTES, IDX_PAD], dtype=pl.INT32)
            scale_padded, weight_padded, r_route_padded = self.pack_step(
                x_norm_scale, weights,
                scale_padded, weight_padded, r_route_padded,
            )

            recv_x_out = pl.create_tensor([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
            recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
            recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
            recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32)
            recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
            (
                recv_x_out, recv_scale_out, recv_w_out,
                recv_r_route_out, recv_count_out,
            ) = self.dispatch_step(
                indices, x_norm_i8, scale_padded, weight_padded, r_route_padded,
                recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out,
                pub_counts, count_done,
                recv_x, recv_scale, recv_w, recv_r_route,
                my_rank,
            )

            recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
            recv_y = self.expert_routed_step(
                recv_x_out, recv_scale_out, recv_w_out, recv_count_out,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                recv_y,
            )

            ffn_out = pl.create_tensor([B, S, D], dtype=pl.BF16)
            ffn_out = self.combine_step(
                recv_y, recv_r_route_out, sh,
                ffn_out,
                pub_counts, routed_y_buf, combine_done,
                my_rank,
            )
            return self.hc_post_step(
                ffn_out, x_hc, post_ffn, comb_ffn, x_next,
            )

        # --- host_orch: allocate windows + loop world ranks --------------
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            x_hc: pl.Tensor[[N_RANKS, B, S, HC_MULT, D], pl.BF16],
            hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
            hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
            hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
            norm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
            gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
            gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
            tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
            input_ids: pl.Tensor[[N_RANKS, B, S], pl.INT64],
            routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
            routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
            routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
            routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
            routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
            routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
            shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
            shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
            shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
            shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
            shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
            shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
            x_next: pl.Out[pl.Tensor[[N_RANKS, B, S, HC_MULT, D], pl.BF16]],
            layer_id: pl.Scalar[pl.INT32],
        ):
            pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * N_LOCAL * 4)
            count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
            recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D * 1)  # INT8
            recv_scale_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)  # FP32
            recv_w_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)  # FP32
            recv_r_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)  # INT32
            # count_done window is reused for the data-phase barrier (AtomicAdd
            # bumps per-src cell from 1 to 2; wait expected=2 in data phase).
            routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)  # BF16
            combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)

            for r in pl.range(pld.world_size()):
                pub_counts = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
                count_done = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
                recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
                recv_scale = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
                recv_w = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
                recv_r_route = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
                routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
                combine_done = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
                self.chip_orch(
                    x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
                    norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
                    routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
                    routed_w2[r], routed_w2_scale[r],
                    shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
                    shared_w2[r], shared_w2_scale[r],
                    x_next[r],
                    pub_counts, count_done,
                    recv_x, recv_scale, recv_w, recv_r_route,
                    routed_y_buf, combine_done,
                    layer_id, r,
                    device=r,
                )

    return EpMoE


# === Golden + test ==========================================================
def golden_moe_ep(tensors):
    """Per-rank torch reference. Replays the 4 stages on host. Each rank's
    output depends only on its own inputs because the dispatch+combine round-
    trip is r_route-keyed and shape-preserving (test_l3 pattern)."""
    import torch

    from hc_pre import golden_hc_pre
    from hc_post import golden_hc_post
    from gate import golden_gate_core
    from expert_shared import golden_expert_shared
    from expert_routed import golden_expert_routed

    x_next_out = torch.zeros(N_RANKS, B, S, HC_MULT, D, dtype=torch.bfloat16)

    for r in range(N_RANKS):
        # Stage 1: hc_pre
        x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
        post_t = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
        comb_t = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
        golden_hc_pre({
            "x":        tensors["x_hc"][r],
            "hc_fn":    tensors["hc_ffn_fn"][r],
            "hc_scale": tensors["hc_ffn_scale"][r],
            "hc_base":  tensors["hc_ffn_base"][r],
            "x_mixed":  x_mixed,
            "post":     post_t,
            "comb":     comb_t,
        })

        # Stage 2: gate (global routing)
        x_norm = torch.zeros(T, D, dtype=torch.bfloat16)
        x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
        x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
        indices = torch.zeros(T, TOPK, dtype=torch.int32)
        weights = torch.zeros(T, TOPK, dtype=torch.float32)
        golden_gate_core({
            "x_mixed":      x_mixed,
            "norm_w":       tensors["norm_w"][r],
            "gate_w":       tensors["gate_w"][r],
            "gate_bias":    tensors["gate_bias"][r],
            "layer_id":     tensors["layer_id"],
            "tid2eid":      tensors["tid2eid"][r],
            "input_ids":    tensors["input_ids"][r],
            "x_norm":       x_norm,
            "x_norm_i8":    x_norm_i8,
            "x_norm_scale": x_norm_scale,
            "indices":      indices,
            "weights":      weights,
        })

        # Stage 3: expert_shared (local)
        sh = torch.zeros(T, D, dtype=torch.bfloat16)
        golden_expert_shared({
            "x_local_i8":       x_norm_i8,
            "x_local_scale_dq": x_norm_scale,
            "shared_w1":        tensors["shared_w1"][r],
            "shared_w1_scale":  tensors["shared_w1_scale"][r],
            "shared_w3":        tensors["shared_w3"][r],
            "shared_w3_scale":  tensors["shared_w3_scale"][r],
            "shared_w2":        tensors["shared_w2"][r],
            "shared_w2_scale":  tensors["shared_w2_scale"][r],
            "sh":               sh,
        })

        # Stage 4: host-side dispatch simulation across all ranks for this dst=r.
        # Collect all (src, t, k) routes that land on rank r.
        recv_x = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.int8)
        recv_scale = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
        recv_w = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
        recv_r_route = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.int32)
        recv_count = torch.zeros(N_LOCAL, 1, dtype=torch.int32)

        # Compute slot offsets the way dispatch_step does (rank-major within
        # each local expert), so the order matches the on-device run.
        send_counts = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
        all_indices = []
        all_x_i8 = []
        all_scale = []
        all_weights = []
        for src in range(N_RANKS):
            src_x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
            src_post = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
            src_comb = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
            golden_hc_pre({
                "x":        tensors["x_hc"][src],
                "hc_fn":    tensors["hc_ffn_fn"][src],
                "hc_scale": tensors["hc_ffn_scale"][src],
                "hc_base":  tensors["hc_ffn_base"][src],
                "x_mixed":  src_x_mixed,
                "post":     src_post,
                "comb":     src_comb,
            })
            src_x_norm = torch.zeros(T, D, dtype=torch.bfloat16)
            src_x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
            src_x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
            src_indices = torch.zeros(T, TOPK, dtype=torch.int32)
            src_weights = torch.zeros(T, TOPK, dtype=torch.float32)
            golden_gate_core({
                "x_mixed":      src_x_mixed,
                "norm_w":       tensors["norm_w"][src],
                "gate_w":       tensors["gate_w"][src],
                "gate_bias":    tensors["gate_bias"][src],
                "layer_id":     tensors["layer_id"],
                "tid2eid":      tensors["tid2eid"][src],
                "input_ids":    tensors["input_ids"][src],
                "x_norm":       src_x_norm,
                "x_norm_i8":    src_x_norm_i8,
                "x_norm_scale": src_x_norm_scale,
                "indices":      src_indices,
                "weights":      src_weights,
            })
            all_indices.append(src_indices)
            all_x_i8.append(src_x_norm_i8)
            all_scale.append(src_x_norm_scale)
            all_weights.append(src_weights)
            for t in range(T):
                for k in range(TOPK):
                    eid = int(src_indices[t, k].item())
                    dst = eid // N_LOCAL
                    loc_e = eid % N_LOCAL
                    send_counts[src, dst, loc_e] += 1

        # Pack onto rank r in src-major (rank 0 first, then rank 1) within each
        # local expert — same convention as dispatch_step's prefix_sum offsets.
        slot_offsets = torch.zeros(N_RANKS, N_LOCAL, dtype=torch.int32)
        running = torch.zeros(N_LOCAL, dtype=torch.int32)
        for src in range(N_RANKS):
            slot_offsets[src] = running.clone()
            running = running + send_counts[src, r]
        for e in range(N_LOCAL):
            recv_count[e, 0] = int(running[e].item())

        for src in range(N_RANKS):
            cursor = torch.zeros(N_LOCAL, dtype=torch.int32)
            for t in range(T):
                for k in range(TOPK):
                    eid = int(all_indices[src][t, k].item())
                    if eid // N_LOCAL != r:
                        continue
                    loc_e = eid % N_LOCAL
                    slot = int(slot_offsets[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    recv_x[loc_e, slot, :] = all_x_i8[src][t, :]
                    recv_scale[loc_e, slot] = float(all_scale[src][t, 0].item())
                    recv_w[loc_e, slot] = float(all_weights[src][t, k].item())
                    recv_r_route[loc_e, slot] = t * TOPK + k

        # Stage 5: routed expert (local, weighted)
        recv_y = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.bfloat16)
        golden_expert_routed({
            "recv_x":            recv_x,
            "recv_scale_dq":     recv_scale,
            "recv_weights":      recv_w,
            "recv_expert_count": recv_count,
            "routed_w1":         tensors["routed_w1"][r],
            "routed_w1_scale":   tensors["routed_w1_scale"][r],
            "routed_w3":         tensors["routed_w3"][r],
            "routed_w3_scale":   tensors["routed_w3_scale"][r],
            "routed_w2":         tensors["routed_w2"][r],
            "routed_w2_scale":   tensors["routed_w2_scale"][r],
            "recv_y":            recv_y,
        })

        # Stage 6: combine — for each (src, t, k) that originated on this
        # rank, find the (loc_e, slot) on rank dst where the SwiGLU result
        # landed, then accumulate by r_route = t*TOPK+k.
        # Recreate the slot bookkeeping for each dst from this rank r's POV.
        my_routes = []
        for t in range(T):
            for k in range(TOPK):
                eid = int(all_indices[r][t, k].item())
                dst = eid // N_LOCAL
                loc_e = eid % N_LOCAL
                my_routes.append((t, k, dst, loc_e))

        # For each dst, dst-side has packing where rank-r's contribution lives
        # at slot offset = Σ_{s<r} send_counts[s, dst, loc_e].
        dst_recv_y = {}
        dst_recv_count = {}
        for dst in range(N_RANKS):
            # Replay dispatch from ALL src ranks to dst, then expert_routed,
            # then pull out per-route results.
            d_recv_x = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.int8)
            d_recv_scale = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
            d_recv_w = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
            d_recv_r_route = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.int32)
            d_recv_count = torch.zeros(N_LOCAL, 1, dtype=torch.int32)
            d_slot_offsets = torch.zeros(N_RANKS, N_LOCAL, dtype=torch.int32)
            d_running = torch.zeros(N_LOCAL, dtype=torch.int32)
            for src in range(N_RANKS):
                d_slot_offsets[src] = d_running.clone()
                d_running = d_running + send_counts[src, dst]
            for e in range(N_LOCAL):
                d_recv_count[e, 0] = int(d_running[e].item())
            for src in range(N_RANKS):
                cursor = torch.zeros(N_LOCAL, dtype=torch.int32)
                for t in range(T):
                    for k in range(TOPK):
                        eid = int(all_indices[src][t, k].item())
                        if eid // N_LOCAL != dst:
                            continue
                        loc_e = eid % N_LOCAL
                        slot = int(d_slot_offsets[src, loc_e].item() + cursor[loc_e].item())
                        cursor[loc_e] += 1
                        d_recv_x[loc_e, slot, :] = all_x_i8[src][t, :]
                        d_recv_scale[loc_e, slot] = float(all_scale[src][t, 0].item())
                        d_recv_w[loc_e, slot] = float(all_weights[src][t, k].item())
                        d_recv_r_route[loc_e, slot] = t * TOPK + k
            d_recv_y = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.bfloat16)
            golden_expert_routed({
                "recv_x":            d_recv_x,
                "recv_scale_dq":     d_recv_scale,
                "recv_weights":      d_recv_w,
                "recv_expert_count": d_recv_count,
                "routed_w1":         tensors["routed_w1"][dst],
                "routed_w1_scale":   tensors["routed_w1_scale"][dst],
                "routed_w3":         tensors["routed_w3"][dst],
                "routed_w3_scale":   tensors["routed_w3_scale"][dst],
                "routed_w2":         tensors["routed_w2"][dst],
                "routed_w2_scale":   tensors["routed_w2_scale"][dst],
                "recv_y":            d_recv_y,
            })
            dst_recv_y[dst] = d_recv_y
            dst_recv_count[dst] = d_recv_count

        # Now combine — per-route reverse lookup of dst's slot for THIS rank
        # r's (t, k):
        routed_y_buf_r = torch.zeros(N_ROUTES, D, dtype=torch.bfloat16)
        for (t, k, dst, loc_e) in my_routes:
            # Find this rank r's slot inside dst.recv_x: src=r block,
            # cursor = how many of r's (t', k' <= (t, k)) so far targeted this loc_e.
            src_off = 0
            for s in range(r):
                src_off += int(send_counts[s, dst, loc_e].item())
            # Count how many earlier (t', k') from rank r targeted (dst, loc_e).
            cursor = 0
            for (tt, kk, dd, ll) in my_routes:
                if (tt, kk) == (t, k):
                    break
                if dd == dst and ll == loc_e:
                    cursor += 1
            slot = src_off + cursor
            r_route = t * TOPK + k
            routed_y_buf_r[r_route, :] = dst_recv_y[dst][loc_e, slot, :]

        # Stage 7: reduce + sh + hc_post
        acc = sh.float().clone()
        for k in range(TOPK):
            for t in range(T):
                acc[t, :] += routed_y_buf_r[t * TOPK + k, :].float()
        ffn_out = acc.to(torch.bfloat16).reshape(B, S, D)
        x_next_r = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
        golden_hc_post({
            "x":        ffn_out,
            "residual": tensors["x_hc"][r],
            "post":     post_t,
            "comb":     comb_t,
            "y":        x_next_r,
        })
        x_next_out[r] = x_next_r

    tensors["x_next"][:] = x_next_out


def _int8_amax_per_row(x_bf16):
    return x_bf16.float().abs().amax(dim=-1, keepdim=True).clamp_min(config.INT8_AMAX_EPS)


def _quant_w_per_channel(w_bf16):
    import torch
    amax = w_bf16.float().abs().amax(dim=-1).clamp_min(config.INT8_AMAX_EPS)
    scale_quant = config.INT8_SCALE_MAX / amax
    scaled = w_bf16.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def build_tensor_specs(layer_id=0):
    import torch
    from golden import ScalarSpec, TensorSpec

    # Shared (replicated) weights are broadcast across ranks; the routed
    # weights are per-rank shards.
    def init_x_hc():
        return torch.randn(N_RANKS, B, S, HC_MULT, D)

    def init_hc_ffn_fn():
        x = torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_hc_ffn_scale():
        x = torch.ones(3) * 0.5
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_hc_ffn_base():
        x = torch.zeros(MIX_HC)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_norm_w():
        x = torch.ones(D)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_gate_w():
        x = torch.randn(N_EXPERTS_GLOBAL, D) / D ** 0.5
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_gate_bias():
        x = torch.zeros(N_EXPERTS_GLOBAL)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_tid2eid():
        x = torch.randint(0, N_EXPERTS_GLOBAL, (VOCAB, TOPK), dtype=torch.int32)
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_input_ids():
        # Distinct per-rank token streams.
        return torch.randint(0, VOCAB, (N_RANKS, B, S), dtype=torch.int64)

    # Per-rank routed expert weights (different shards).
    routed_w1_i8_list = []
    routed_w1_s_list = []
    routed_w3_i8_list = []
    routed_w3_s_list = []
    routed_w2_i8_list = []
    routed_w2_s_list = []
    for _ in range(N_RANKS):
        w1_bf16 = (torch.randn(N_LOCAL, MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
        w3_bf16 = (torch.randn(N_LOCAL, MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
        w2_bf16 = (torch.randn(N_LOCAL, D, MOE_INTER) / MOE_INTER ** 0.5).to(torch.bfloat16)
        w1_i8, w1_s = _quant_w_per_channel(w1_bf16)
        w3_i8, w3_s = _quant_w_per_channel(w3_bf16)
        w2_i8, w2_s = _quant_w_per_channel(w2_bf16)
        routed_w1_i8_list.append(w1_i8)
        routed_w1_s_list.append(w1_s)
        routed_w3_i8_list.append(w3_i8)
        routed_w3_s_list.append(w3_s)
        routed_w2_i8_list.append(w2_i8)
        routed_w2_s_list.append(w2_s)

    rw1_i8 = torch.stack(routed_w1_i8_list)
    rw1_s = torch.stack(routed_w1_s_list)
    rw3_i8 = torch.stack(routed_w3_i8_list)
    rw3_s = torch.stack(routed_w3_s_list)
    rw2_i8 = torch.stack(routed_w2_i8_list)
    rw2_s = torch.stack(routed_w2_s_list)

    # Shared expert weights — replicated across ranks.
    sw1_bf16 = (torch.randn(MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    sw3_bf16 = (torch.randn(MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    sw2_bf16 = (torch.randn(D, MOE_INTER) / MOE_INTER ** 0.5).to(torch.bfloat16)
    sw1_i8, sw1_s = _quant_w_per_channel(sw1_bf16)
    sw3_i8, sw3_s = _quant_w_per_channel(sw3_bf16)
    sw2_i8, sw2_s = _quant_w_per_channel(sw2_bf16)
    sw1_i8 = sw1_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw1_s = sw1_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    sw3_i8 = sw3_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw3_s = sw3_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    sw2_i8 = sw2_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw2_s = sw2_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    return [
        TensorSpec("x_hc",          [N_RANKS, B, S, HC_MULT, D],     torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",     [N_RANKS, MIX_HC, HC_DIM],       torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale",  [N_RANKS, 3],                    torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",   [N_RANKS, MIX_HC],               torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",        [N_RANKS, D],                    torch.float32,  init_value=init_norm_w),
        TensorSpec("gate_w",        [N_RANKS, N_EXPERTS_GLOBAL, D],  torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",     [N_RANKS, N_EXPERTS_GLOBAL],     torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",       [N_RANKS, VOCAB, TOPK],          torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids",     [N_RANKS, B, S],                 torch.int64,    init_value=init_input_ids),
        TensorSpec("routed_w1",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: rw1_i8),
        TensorSpec("routed_w1_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: rw1_s),
        TensorSpec("routed_w3",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: rw3_i8),
        TensorSpec("routed_w3_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: rw3_s),
        TensorSpec("routed_w2",        [N_RANKS, N_LOCAL, D, MOE_INTER], torch.int8,    init_value=lambda: rw2_i8),
        TensorSpec("routed_w2_scale",  [N_RANKS, N_LOCAL, D],            torch.float32, init_value=lambda: rw2_s),
        TensorSpec("shared_w1",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2",        [N_RANKS, D, MOE_INTER],          torch.int8,    init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale",  [N_RANKS, D],                     torch.float32, init_value=lambda: sw2_s),
        TensorSpec("x_next",           [N_RANKS, B, S, HC_MULT, D],      torch.bfloat16, is_output=True),
        ScalarSpec("layer_id",         torch.int32,                      layer_id),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1",
                        help="comma-separated device ids; need at least 2")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    program = _build_ep_moe_program()
    result = run(
        program=program,
        specs=build_tensor_specs(layer_id=args.layer_id),
        golden_fn=golden_moe_ep,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
