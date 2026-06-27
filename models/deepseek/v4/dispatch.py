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


@pl.jit.inline
def dispatch_ep1(
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
        # Vectorized tail zero-init: one fill per buffer instead of
        # N_LOCAL_EXPERTS * RECV_MAX scalar writes. Same neutral tails, far less
        # scalar-store time on this serial CORE_GROUP kernel (which sits on the
        # decode critical path between gate and expert_routed).
        recv_scale_dq[:, :] = pl.full([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32, value=0.0)
        recv_weights[:, :] = pl.full([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32, value=0.0)
        recv_token[:, :] = pl.full([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32, value=0)
        # recv_expert_count is [E, 1] -- a 4-byte row that fails the 32B tile
        # alignment for a vector fill, so zero its E entries with scalar writes.
        for e in pl.range(N_LOCAL_EXPERTS):
            pl.write(recv_expert_count, [e, 0], pl.cast(0, pl.INT32))

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


W_PAD = 8  # FP32 scale/weight tile pad (32B min tile)
IDX_PAD = 8  # INT32 r_route tile pad
X_STAGE_ROWS = 8  # recv_x stage chunk rows (8 x D INT8 = 32KB UB tile)
N_ROUTES = T * TOPK


@pl.jit.inline
def dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_w_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_r_route_out: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_count_out: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    pub_counts: pld.DistributedTensor[[EP_WORLD_SIZE * EP_WORLD_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    count_done: pld.DistributedTensor[[EP_WORLD_SIZE, 1], pl.INT32],
    data_done: pld.DistributedTensor[[EP_WORLD_SIZE, 1], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, IDX_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based ordinal of this MoE call within the current shared-window epoch.
    # The done windows are monotonic counters, so waits use `>= moe_epoch`
    # instead of a fixed `>= 1` and cannot be satisfied by a previous MoE call.
    moe_epoch: pl.Scalar[pl.INT32],
):
    # Flat 2-D view for the stage_out row stores; reshape kept outside the scope
    # so it stays a tensor view, not a tile.
    recv_x_out_flat = pl.reshape(recv_x_out, [N_LOCAL_EXPERTS * RECV_MAX, D])

    # Route + push + stage_out in one pl.at(CORE_GROUP) (= InCore) so the scalar
    # histogram/prefix-sum, notify/wait barriers and remote_store run as one task.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch"):
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        # ---------- histogram: scalar histogram on indices ----------
        send_counts = pl.array.create(EP_WORLD_SIZE * N_LOCAL_EXPERTS, pl.INT32)
        for d in pl.range(EP_WORLD_SIZE):
            for e in pl.range(N_LOCAL_EXPERTS):
                send_counts[d * N_LOCAL_EXPERTS + e] = 0

        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                d = eid // N_LOCAL_EXPERTS
                e = eid - d * N_LOCAL_EXPERTS
                cur = send_counts[d * N_LOCAL_EXPERTS + e]
                send_counts[d * N_LOCAL_EXPERTS + e] = cur + 1

        # ---------- publish: overwrite this epoch's exact counts ----------
        # `pub_counts` stores the current MoE's exact routing counts, not a
        # monotonic counter. Publish all buckets, including zeros, so reusing the
        # same window across serial MoE calls cannot leak stale counts.
        for peer in pl.range(EP_WORLD_SIZE):
            for d in pl.range(EP_WORLD_SIZE):
                for e in pl.range(N_LOCAL_EXPERTS):
                    v = send_counts[d * N_LOCAL_EXPERTS + e]
                    pld.system.notify(
                        target=pub_counts,
                        peer=peer,
                        offsets=[my_rank * EP_WORLD_SIZE + d, e],
                        value=v,
                        op=pld.NotifyOp.Set,
                    )

        # ---------- count_done barrier ----------
        # count_done/data_done are separate per-stage monotonic counters. Each
        # MoE bumps the local source cell once; receivers wait for this call's
        # epoch so older stages from earlier MoE calls do not pass the barrier.
        for peer in pl.range(EP_WORLD_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=count_done,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(EP_WORLD_SIZE):
            if src != my_rank:
                pld.system.wait(
                    signal=count_done,
                    offsets=[src, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        # ---------- prefix_sum: my slot offset + total recv_count ----------
        my_slot_at_dst = pl.array.create(EP_WORLD_SIZE * N_LOCAL_EXPERTS, pl.INT32)
        for d in pl.range(EP_WORLD_SIZE):
            for e in pl.range(N_LOCAL_EXPERTS):
                acc = pl.const(0, pl.INT32)
                for s in pl.range(EP_WORLD_SIZE):
                    if s < my_rank:
                        acc = acc + pl.read(pub_counts, [s * EP_WORLD_SIZE + d, e])
                my_slot_at_dst[d * N_LOCAL_EXPERTS + e] = acc

        for e in pl.range(N_LOCAL_EXPERTS):
            acc = pl.const(0, pl.INT32)
            for s in pl.range(EP_WORLD_SIZE):
                acc = acc + pl.read(pub_counts, [s * EP_WORLD_SIZE + my_rank, e])
            pl.write(recv_count_out, [e, 0], acc)

        # ---------- payload_push: 4 channels per (t, k) ----------
        cursor = pl.array.create(EP_WORLD_SIZE * N_LOCAL_EXPERTS, pl.INT32)
        for d in pl.range(EP_WORLD_SIZE):
            for e in pl.range(N_LOCAL_EXPERTS):
                cursor[d * N_LOCAL_EXPERTS + e] = 0

        # Pad tiles, zero-initialised once; only column 0 is overwritten per
        # push (UB tile + remote_store is the proven path for runtime-computed
        # scalars — a GM pack table written by scalar pl.write corrupts).
        scale_tile = pl.tile.full([1, W_PAD], dtype=pl.FP32, value=0.0)
        w_tile = pl.tile.full([1, W_PAD], dtype=pl.FP32, value=0.0)
        idx_tile = pl.tile.full([1, IDX_PAD], dtype=pl.INT32, value=0)

        for t in pl.range(active_tokens):
            for k in pl.range(TOPK):
                eid = pl.read(indices, [t, k])
                dst = eid // N_LOCAL_EXPERTS
                loc_e = eid - dst * N_LOCAL_EXPERTS
                bucket = dst * N_LOCAL_EXPERTS + loc_e
                cur_val = cursor[bucket]
                slot_off = my_slot_at_dst[bucket]
                slot = slot_off + cur_val
                row = loc_e * RECV_MAX + slot
                cursor[bucket] = cur_val + 1
                r_route = t * TOPK + k

                pld.tensor.put(
                    dst=recv_x,
                    peer=dst,
                    src=x_norm_i8,
                    dst_offsets=[row, 0],
                    src_offsets=[t, 0],
                    shape=[1, D],
                )
                pl.tile.write(scale_tile, [0, 0], pl.read(x_norm_scale, [t, 0]))
                pld.tile.remote_store(scale_tile, target=recv_scale, peer=dst, offsets=[row, 0])
                pl.tile.write(w_tile, [0, 0], pl.read(weights, [t, k]))
                pld.tile.remote_store(w_tile, target=recv_w, peer=dst, offsets=[row, 0])
                pl.tile.write(idx_tile, [0, 0], pl.cast(r_route, pl.INT32))
                pld.tile.remote_store(idx_tile, target=recv_r_route, peer=dst, offsets=[row, 0])

        # ---------- data_done barrier — per-src signal cells ----------
        # Same epoch discipline as count_done: value is only a counter bump, and
        # the phase meaning comes from the dedicated data_done window.
        for peer in pl.range(EP_WORLD_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=data_done,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
        for src in pl.range(EP_WORLD_SIZE):
            if src != my_rank:
                pld.system.wait(
                    signal=data_done,
                    offsets=[src, 0],
                    expected=moe_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        # stage_out: copy the windows the push filled into the host outputs.
        # recv_x: INT8 window copied in X_STAGE_ROWS-row chunks through the
        # flat view (per-row copies cost ~16x more MTE transfers).
        for row in pl.range(0, N_LOCAL_EXPERTS * RECV_MAX, X_STAGE_ROWS):
            recv_x_out_flat[row:row + X_STAGE_ROWS, :] = recv_x[row:row + X_STAGE_ROWS, :]

        # scale / w / r_route: scalar copy of window column 0, bounded to the
        # valid rows (downstream consumers are count-bounded, so tail slots
        # carry no contract).
        # WORKAROUND pypto#1693: writing these via tile pl.store made them
        # add_inout and the orchestration scrambled their post-write SSA aliases;
        # scalar pl.write keeps them add_output.
        for e in pl.range(N_LOCAL_EXPERTS):
            n_rows = pl.cast(pl.read(recv_count_out, [e, 0]), pl.INDEX)
            for slot in pl.range(n_rows):
                row = e * RECV_MAX + slot
                pl.write(recv_scale_out, [e, slot], pl.read(recv_scale, [row, 0]))
                pl.write(recv_w_out, [e, slot], pl.read(recv_w, [row, 0]))
                pl.write(recv_r_route_out, [e, slot], pl.read(recv_r_route, [row, 0]))



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
    dispatch_ep1(
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
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=dispatch_test,
        specs=build_tensor_specs(),
        golden_fn=golden_dispatch,
        compile_cfg=dict(dump_passes=args.dump_passes),
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
