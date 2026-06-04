# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE packed dispatch (decode, single-card EP).

Pure scatter from token-major router outputs to the per-local-expert layout
consumed by ``expert_routed``. The INT8 quant of ``x_norm`` already happened in
``gate``; this kernel only moves the pre-quantized rows and dequant
scales into the recv buffers.
"""


import pypto.language as pl
import pypto.language.distributed as pld

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ,
                    EP_WORLD_SIZE, EP_RANK, RECV_MAX)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS

# EP (cross-rank) layout — used only by dispatch_ep below, which moe_ep.py
# imports after overriding config to the 2-rank DEMO preset. In the single-card
# moe.py process these bind to the default config and dispatch_ep is defined but
# never called/compiled (lazy JIT), so the values are harmless there.
N_RANKS = EP_WORLD_SIZE
N_LOCAL = N_LOCAL_EXPERTS  # all-rank naming used by the EP kernel
W_PAD = 8  # FP32 scale/weight tile pad (32B min tile)
IDX_PAD = 8  # INT32 r_route tile pad


@pl.jit.inline
def dispatch(
    x_norm_i8:       pl.Tensor[[T, D],    pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1],    pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq:     pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_weights:      pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_token:        pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1],           pl.INT32],
):
    # recv_x still uses a flat row view because its destination row is
    # data-dependent. Small route metadata uses the natural 2-D layout.
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch"):
        # Zero-init tail slots so downstream consumers (expert_routed / combine)
        # can rely on slot >= recv_expert_count[e] being neutral:
        #   - recv_scale_dq = 0 -> dequant of any recv_x tail row yields 0
        #   - recv_weights  = 0 -> combine's weighted reduction skips tail rows
        #   - recv_token    = 0 -> safe scatter target (combine ignores via count)
        # The recv_x INT8 tail is left uninitialized; pairing it with scale_dq=0
        # is sufficient to neutralize its contribution after dequant.
        for e in pl.range(N_LOCAL_EXPERTS):
            pl.write(recv_expert_count, [e, 0], pl.cast(0, pl.INT32))
            for s in pl.range(RECV_MAX):
                pl.write(recv_scale_dq, [e, s], 0.0)
                pl.write(recv_weights, [e, s], 0.0)
                pl.write(recv_token, [e, s], pl.cast(0, pl.INT32))

        for t in pl.range(T):
            for k in pl.range(TOPK):
                e_global = pl.read(indices, [t, k])
                e = pl.cast(e_global - EXPERTS_START_IDX, pl.INDEX)
                slot_i32 = pl.read(recv_expert_count, [e, 0])
                slot = pl.cast(slot_i32, pl.INDEX)
                dst = e * RECV_MAX + slot

                recv_x_flat = pl.assemble(recv_x_flat, pl.slice(x_norm_i8, [1, D], [t, 0]), [dst, 0])
                pl.write(recv_scale_dq, [e, slot], pl.read(x_norm_scale, [t, 0]))
                pl.write(recv_weights, [e, slot], pl.read(weights, [t, k]))
                pl.write(recv_token, [e, slot], pl.cast(t, pl.INT32))
                pl.write(recv_expert_count, [e, 0], pl.cast(slot_i32 + 1, pl.INT32))


@pl.jit.incore
def dispatch_ep(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8]],
    recv_scale_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32]],
    recv_w_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX], pl.FP32]],
    recv_r_route_out: pl.Out[pl.Tensor[[N_LOCAL, RECV_MAX], pl.INT32]],
    recv_count_out: pl.Out[pl.Tensor[[N_LOCAL, 1], pl.INT32]],
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # ``recv_x`` is widened to FP16 to work around a b8 SDMA bug on a2a3 (see
    # x_tile cast in payload_push below). ``recv_x_out`` stays INT8 — the narrow
    # cast happens in stage_out.
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.FP16],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor[[N_LOCAL, RECV_MAX, D], pl.INT8],
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

    # Allocate the three pad tiles once, zero-initialised. Columns 1..PAD-1 stay
    # 0 across every push so the stage_out row_sum at the receiver recovers
    # column 0 exactly. The loop body only overwrites column 0.
    scale_tile: pl.Tile[[1, W_PAD], pl.FP32] = pl.tile.full(
        [1, W_PAD], dtype=pl.FP32, value=0.0
    )
    w_tile: pl.Tile[[1, W_PAD], pl.FP32] = pl.tile.full(
        [1, W_PAD], dtype=pl.FP32, value=0.0
    )
    idx_tile: pl.Tile[[1, IDX_PAD], pl.INT32] = pl.tile.full(
        [1, IDX_PAD], dtype=pl.INT32, value=0
    )

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
            r_route = pl.cast(t * TOPK + k, pl.INT32)

            # Widen INT8 → FP16 before the cross-rank push: the b8 SDMA burst
            # path (``copy_ubuf_to_gm_align_b8``) is broken on a2a3 — recv data
            # lands stale on the peer. b16 is fine, and FP16 is the only floating
            # dtype that the a2a3 ``TCVT`` table reaches from INT8 directly (see
            # ``TCvt.hpp``: ``I8 -> FP16`` is listed as a direct path; INT8 →
            # BF16 / INT8 → INT32 are not). FP16's 10-bit mantissa exactly
            # represents every INT8 value in [-128, 127], so the round-trip is
            # lossless. INT8 is recovered via the symmetric FP16 → INT8 cast in
            # stage_out below (also a documented direct path).
            x_tile_i8 = pl.load(x_norm_i8, [t, 0], [1, D])
            x_tile = pl.cast(x_tile_i8, target_type=pl.FP16)
            pld.tile.remote_store(x_tile, target=recv_x, peer=dst, offsets=[row, 0])

            s_val = pl.read(x_norm_scale, [t, 0])
            pl.tile.write(scale_tile, [0, 0], s_val)
            pld.tile.remote_store(scale_tile, target=recv_scale, peer=dst, offsets=[row, 0])

            w_val = pl.read(weights, [t, k])
            pl.tile.write(w_tile, [0, 0], w_val)
            pld.tile.remote_store(w_tile, target=recv_w, peer=dst, offsets=[row, 0])

            pl.tile.write(idx_tile, [0, 0], r_route)
            pld.tile.remote_store(idx_tile, target=recv_r_route, peer=dst, offsets=[row, 0])

    # ---------- data_done barrier ----------
    # Reuse count_done signal cells: count phase bumps to 1, data phase bumps to
    # 2 (per-src cumulative count via AtomicAdd). Avoids a separate window and
    # keeps dispatch_ep under the MAX_TENSOR_ARGS=16 InCore limit.
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
    # Runs after the data_done barrier, in this same InCore kernel: a separate
    # downstream kernel reading the recv_x window would NOT be ordered after the
    # cross-rank push (remote_store targets aren't tracked as a window write) —
    # pypto#1670.
    # recv_x: per-row [1, D] FP16 → narrow back to INT8 host tensor. The window
    # is read with pl.load (tensor.slice rejects a DistributedTensor — pypto#1672)
    # then cast; recv_x_out is the 3D [N_LOCAL, RECV_MAX, D] view expert_routed
    # consumes directly.
    for e in pl.range(N_LOCAL):
        for slot in pl.range(RECV_MAX):
            row = e * RECV_MAX + slot
            stage_x_fp16 = pl.load(recv_x, [row, 0], [1, D])
            stage_x_i8 = pl.cast(stage_x_fp16, target_type=pl.INT8)
            pl.store(stage_x_i8, [e, slot, 0], recv_x_out, shapes=[1, 1, D])

    # recv_scale / recv_w: per-expert TROWSUM trick on [R, W_PAD] → [R, 1]
    # (column 0 is the real value; rest are zero), reshape and store as [1, R].
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


@pl.jit
def dispatch_test(
    x_norm_i8:       pl.Tensor[[T, D],    pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1],    pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8]],
    recv_scale_dq:     pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_weights:      pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_token:        pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32]],
    recv_expert_count: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]],
):
    dispatch(
        x_norm_i8, x_norm_scale, indices, weights,
        recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count,
    )
    return recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count


def golden_dispatch(tensors):
    """Torch reference for the packed dispatch contract (pure scatter)."""
    import torch

    x_norm_i8       = tensors["x_norm_i8"]        # [T, D]    int8 (pre-quantized in router)
    x_norm_scale = tensors["x_norm_scale"]  # [T, 1]    fp32 per-token dequant scale
    indices = tensors["indices"]   # [T, TOPK] int32
    weights = tensors["weights"]   # [T, TOPK] fp32

    recv_x        = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.int8)
    recv_scale_dq = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights  = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_token    = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
    cursor = [0] * N_LOCAL_EXPERTS
    for t in range(T):
        for k in range(TOPK):
            e = int(indices[t, k].item()) - EXPERTS_START_IDX
            s = cursor[e]
            assert 0 <= e < N_LOCAL_EXPERTS
            assert s < RECV_MAX, f"expert {e} received > RECV_MAX={RECV_MAX} rows"
            recv_x[e, s, :]      = x_norm_i8[t, :]
            recv_scale_dq[e, s]  = float(x_norm_scale[t, 0].item())
            recv_weights[e, s]   = float(weights[t, k].item())
            recv_token[e, s]     = t
            cursor[e] = s + 1

    recv_count = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
    for e in range(N_LOCAL_EXPERTS):
        recv_count[e, 0] = cursor[e]

    tensors["recv_x"][:]            = recv_x
    tensors["recv_scale_dq"][:]     = recv_scale_dq
    tensors["recv_weights"][:]      = recv_weights
    tensors["recv_token"][:]        = recv_token
    tensors["recv_expert_count"][:] = recv_count


def _valid_rows_compare(is_3d: bool = False):
    """Comparator that checks only the valid packed rows (slot < count).

    Dispatch leaves tail rows (slot >= recv_expert_count[e]) uninitialized --
    they carry no contract because every downstream consumer is count-bounded.
    Build the valid mask from the golden recv_expert_count and compare only the
    masked entries.
    """
    import torch

    def cmp(actual, expected, *, expected_outputs, rtol, atol, **_):
        count = expected_outputs["recv_expert_count"].cpu().reshape(-1, 1)        # [E, 1]
        valid = torch.arange(RECV_MAX).reshape(1, RECV_MAX) < count               # [E, RECV_MAX]
        a = actual.cpu()
        e = expected.cpu()
        if is_3d:
            valid = valid.unsqueeze(-1).expand_as(a)
        a_v = a[valid].to(torch.float32)
        e_v = e[valid].to(torch.float32)
        if torch.allclose(a_v, e_v, rtol=rtol, atol=atol):
            return True, ""
        diff = (a_v - e_v).abs()
        n_bad = int((diff > atol + rtol * e_v.abs()).sum().item())
        worst = float(diff.max().item()) if diff.numel() else 0.0
        return False, (
            f"    valid-row mismatch (rtol={rtol} atol={atol}): "
            f"{n_bad}/{a_v.numel()} bad, worst_diff={worst:.6g}"
        )

    cmp.__name__ = "valid_rows_compare"
    return cmp


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_norm_i8():
        return torch.randint(-128, 128, (T, D), dtype=torch.int8)

    def init_x_norm_scale():
        # Per-token dequant scale (strictly positive, as produced by the router).
        return (torch.rand(T, 1) + 0.01).float()

    def init_indices():
        # Each token picks TOPK distinct experts.
        rows = [torch.randperm(N_LOCAL_EXPERTS)[:TOPK] for _ in range(T)]
        return torch.stack(rows).to(torch.int32)

    def init_weights():
        # Per-row weights normalized to sum=routed_scaling_factor.
        w = torch.rand(T, TOPK) + 0.1
        w = w / w.sum(dim=-1, keepdim=True) * M.routed_scaling_factor
        return w.float()

    return [
        TensorSpec("x_norm_i8",       [T, D], torch.int8,    init_value=init_x_norm_i8),
        TensorSpec("x_norm_scale", [T, 1], torch.float32, init_value=init_x_norm_scale),
        TensorSpec("indices", [T, TOPK], torch.int32,    init_value=init_indices),
        TensorSpec("weights", [T, TOPK], torch.float32,  init_value=init_weights),
        TensorSpec("recv_x",            [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8,     is_output=True),
        TensorSpec("recv_scale_dq",     [N_LOCAL_EXPERTS, RECV_MAX],    torch.float32,  is_output=True),
        TensorSpec("recv_weights",      [N_LOCAL_EXPERTS, RECV_MAX],    torch.float32,  is_output=True),
        TensorSpec("recv_token",        [N_LOCAL_EXPERTS, RECV_MAX],    torch.int32,    is_output=True),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1],           torch.int32,    is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=dispatch_test,
        specs=build_tensor_specs(),
        golden_fn=golden_dispatch,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Tail rows (slot >= count) are uninitialized by design; only the
            # valid packed rows carry a contract.
            "recv_x":        _valid_rows_compare(is_3d=True),
            "recv_scale_dq": _valid_rows_compare(),
            "recv_weights":  _valid_rows_compare(),
            "recv_token":    _valid_rows_compare(),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
