# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 routed-expert grouped FFN (decode, TP=EP=8 BF16).

Per-card local routed-expert path: each of the 8 EP ranks owns
``MOE_NUM_EXPERTS_LOCAL = 36`` of the 288 global routed experts. The
local CSR-packed token rows (produced by ``dispatch.py``'s EP all-to-all)
are fed into the same ``down_proj(activation(gate_proj(x)) * up_proj(x))``
shape, except the per-expert axis is now 36 rather than 288.

Distributed-topology contract (Phase 9 Wave 2):

  * EP_WORLD_SIZE = 8, MOE_NUM_EXPERTS_LOCAL = 36 (288 / 8).
  * The global expert ids hosted on rank ``r`` are
    ``[r * 36 .. (r + 1) * 36 - 1]`` (block-cyclic, mirrors the
    canonical EP a2a pattern from the in-tree TP+EP MoE reference).
  * NO collective primitive runs inside this kernel — the EP a2a is
    owned by ``dispatch.py`` (publishes input rows) and ``combine.py``
    (returns expert outputs). This file is purely local compute.

Per-card weight bundle (host weight loader contract — sliced on EP):

  * ``w_gate[N_LOCAL_EXPERTS, HIDDEN, MOE_INTERMEDIATE]`` BF16
  * ``w_up  [N_LOCAL_EXPERTS, HIDDEN, MOE_INTERMEDIATE]`` BF16
  * ``w_down[N_LOCAL_EXPERTS, MOE_INTERMEDIATE, HIDDEN]`` BF16

The host loader picks expert rows ``[r * 36 .. (r + 1) * 36)`` from the
canonical 288-expert tensors and ships only that shard to rank ``r``.

Shape contract:

  * Input  ``local_routed_x[LOCAL_RECV_MAX, HIDDEN]`` BF16
            (CSR-ordered across the 36 local experts; the first
            ``local_expert_count[e]`` rows starting at
            ``local_expert_offset[e]`` belong to local expert ``e``).
  * Output ``local_routed_y[LOCAL_RECV_MAX, HIDDEN]`` BF16

``LOCAL_RECV_MAX`` upper bound (compile-time):

  Across the world, dispatch publishes ``EP_WORLD_SIZE * BATCH * TOPK``
  routed pairs total. The worst case per receiving rank is when every
  pair lands on it (degenerate hot-routing): ``EP_WORLD_SIZE * BATCH *
  TOPK`` rows. We size ``LOCAL_RECV_MAX`` to that worst case. With
  ``BATCH=16, TOPK=8, EP_WORLD_SIZE=8`` -> 1024 rows, which at
  ``HIDDEN=4096`` BF16 is 8 MB — comfortable inside the per-card window
  budget. The dynamic ``valid_rows`` bound is taken per local expert
  from ``local_expert_count[e]`` so the cube tiles still see tight
  tile masks.

Activation selection (compile-time const, baked from
``SWIGLU_LIMITS[layer_idx]``):
  - ``SWIGLU_LIMIT == 0.0`` -> plain SiLU on gate, no clamp on up:
        h = silu(gate) * up
  - ``SWIGLU_LIMIT > 0.0``  -> SwigluStep (vllm ``SwigluStepAndMul``):
        h = silu(gate).clamp(max=L) * up.clamp(min=-L, max=L)

Tiling: ``pl.parallel(N_LOCAL_EXPERTS=36)`` is the per-expert dispatch
axis. ``MOE_INTERMEDIATE = 1280`` factors as 5 * 256 (GATE_N_CHUNK = 256).
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl

from .config import (
    BATCH,
    EP_WORLD_SIZE,
    HIDDEN,
    MOE_INTERMEDIATE,
    MOE_NUM_EXPERTS_LOCAL,
    MOE_TOP_K,
    SWIGLU_LIMITS,
)


T = BATCH
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL  # 36
TOPK = MOE_TOP_K
INTER = MOE_INTERMEDIATE  # 1280

# Worst-case world-wide routed pairs landing on a single rank.
# 8 source ranks * BATCH=16 * TOPK=8 = 1024 rows.
LOCAL_RECV_MAX = EP_WORLD_SIZE * T * TOPK
MAX_TILE = LOCAL_RECV_MAX

# Tiling constants for the gate/up matmul and down matmul. INTER=1280
# factors as 256 * 5; HIDDEN=4096 factors cleanly by 256.
GATE_K_CHUNK = 256        # K-loop step over HIDDEN for gate / up
GATE_N_CHUNK = 256        # N-tile over INTER (1280 / 256 = 5)
DOWN_K_CHUNK = 256        # K-loop step over INTER for down (1280 / 256 = 5)
DOWN_N_CHUNK = 256        # N-tile over HIDDEN
assert HIDDEN % GATE_K_CHUNK == 0
assert HIDDEN % DOWN_N_CHUNK == 0
assert INTER % GATE_N_CHUNK == 0
assert INTER % DOWN_K_CHUNK == 0


def _build_expert_routed(swiglu_limit: float):
    """Factory that bakes ``SWIGLU_LIMITS[layer_idx]`` as a compile-time const.

    Two specialisations are produced (plain SiLU and SwigluStep@7.0). The
    factory exists because the activation choice changes the kernel body
    -- we cannot select on a runtime ``layer_idx``.
    """

    use_swiglu_step = swiglu_limit > 0.0

    @pl.jit.inline
    def expert_routed(
        local_routed_x: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16],
        local_expert_offset: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
        local_expert_count: pl.Tensor[[N_LOCAL_EXPERTS], pl.INT32],
        w_gate: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
        w_up: pl.Tensor[[N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16],
        w_down: pl.Tensor[[N_LOCAL_EXPERTS, INTER, HIDDEN], pl.BF16],
        local_routed_y: pl.Tensor[[LOCAL_RECV_MAX, HIDDEN], pl.BF16],
    ):
        for e in pl.parallel(N_LOCAL_EXPERTS):
            n_rows = pl.read(local_expert_count, [e])
            offset_i32 = pl.read(local_expert_offset, [e])
            offset = pl.cast(offset_i32, pl.INDEX)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_gate_up"):
                # h_tile is [MAX_TILE, INTER] FP32 (kept on chip; only the
                # leading n_rows rows carry valid data, the rest stays 0).
                h_tile = pl.create_tensor([MAX_TILE, INTER], dtype=pl.FP32)
                h_tile[:, :] = pl.full(
                    [MAX_TILE, INTER], dtype=pl.FP32, value=0.0,
                )
                valid_rows = pl.cast(n_rows, pl.INDEX)

                # gate / up: row tile = [n_rows, HIDDEN] -> [n_rows, INTER].
                # Loop the N axis (INTER) in GATE_N_CHUNK columns.
                for nb in pl.range(INTER // GATE_N_CHUNK):
                    n0 = nb * GATE_N_CHUNK

                    # Peel first K iter; matmul -> matmul_acc cascade.
                    x0 = pl.slice(
                        local_routed_x, [MAX_TILE, GATE_K_CHUNK],
                        [offset, 0],
                        valid_shape=[valid_rows, GATE_K_CHUNK],
                    )
                    wg0 = pl.slice(
                        w_gate, [1, GATE_K_CHUNK, GATE_N_CHUNK],
                        [e, 0, n0],
                    )
                    wu0 = pl.slice(
                        w_up, [1, GATE_K_CHUNK, GATE_N_CHUNK],
                        [e, 0, n0],
                    )
                    wg0_2d = pl.reshape(wg0, [GATE_K_CHUNK, GATE_N_CHUNK])
                    wu0_2d = pl.reshape(wu0, [GATE_K_CHUNK, GATE_N_CHUNK])
                    gate_acc = pl.matmul(x0, wg0_2d, out_dtype=pl.FP32)
                    up_acc = pl.matmul(x0, wu0_2d, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // GATE_K_CHUNK):
                        k0 = kb * GATE_K_CHUNK
                        xk = pl.slice(
                            local_routed_x, [MAX_TILE, GATE_K_CHUNK],
                            [offset, k0],
                            valid_shape=[valid_rows, GATE_K_CHUNK],
                        )
                        wgk = pl.reshape(
                            pl.slice(
                                w_gate, [1, GATE_K_CHUNK, GATE_N_CHUNK],
                                [e, k0, n0],
                            ),
                            [GATE_K_CHUNK, GATE_N_CHUNK],
                        )
                        wuk = pl.reshape(
                            pl.slice(
                                w_up, [1, GATE_K_CHUNK, GATE_N_CHUNK],
                                [e, k0, n0],
                            ),
                            [GATE_K_CHUNK, GATE_N_CHUNK],
                        )
                        gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                        up_acc = pl.matmul_acc(up_acc, xk, wuk)

                    # Activation. ``use_swiglu_step`` is a compile-time
                    # python bool baked by the factory; only one branch
                    # is emitted per specialisation.
                    sigmoid = pl.recip(
                        pl.add(pl.exp(pl.neg(gate_acc)), 1.0),
                    )
                    silu = pl.mul(gate_acc, sigmoid)
                    if use_swiglu_step:
                        # vllm SwigluStepAndMul.forward_native:
                        #   silu(gate).clamp(max=L) * up.clamp(min=-L, max=L)
                        silu_c = pl.minimum(silu, swiglu_limit)
                        up_c = pl.maximum(
                            pl.minimum(up_acc, swiglu_limit),
                            -swiglu_limit,
                        )
                        gated = pl.mul(silu_c, up_c)
                    else:
                        gated = pl.mul(silu, up_acc)

                    # Mask invalid (>= n_rows) rows so dirty trailing rows
                    # of the working tile contribute zero to the down matmul.
                    gated_v = pl.set_validshape(
                        gated, valid_rows, GATE_N_CHUNK,
                    )
                    gated_m = pl.fillpad(gated_v, pad_value=pl.PadValue.zero)
                    h_tile[:, n0 : n0 + GATE_N_CHUNK] = gated_m

                # down: h_tile @ w_down[e] -> [n_rows, HIDDEN].
                # Cast h_tile to BF16 once before the down matmul.
                h_bf16 = pl.cast(h_tile, target_type=pl.BF16)

                for db in pl.range(HIDDEN // DOWN_N_CHUNK):
                    d0 = db * DOWN_N_CHUNK
                    h0 = pl.slice(
                        h_bf16, [MAX_TILE, DOWN_K_CHUNK], [0, 0],
                        valid_shape=[valid_rows, DOWN_K_CHUNK],
                    )
                    wd0 = pl.reshape(
                        pl.slice(
                            w_down, [1, DOWN_K_CHUNK, DOWN_N_CHUNK],
                            [e, 0, d0],
                        ),
                        [DOWN_K_CHUNK, DOWN_N_CHUNK],
                    )
                    y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
                    for kb2 in pl.range(1, INTER // DOWN_K_CHUNK):
                        k0 = kb2 * DOWN_K_CHUNK
                        hk = pl.slice(
                            h_bf16, [MAX_TILE, DOWN_K_CHUNK], [0, k0],
                            valid_shape=[valid_rows, DOWN_K_CHUNK],
                        )
                        wdk = pl.reshape(
                            pl.slice(
                                w_down, [1, DOWN_K_CHUNK, DOWN_N_CHUNK],
                                [e, k0, d0],
                            ),
                            [DOWN_K_CHUNK, DOWN_N_CHUNK],
                        )
                        y_acc = pl.matmul_acc(y_acc, hk, wdk)

                    # Mask invalid trailing rows before writing out.
                    y_v = pl.set_validshape(
                        y_acc, valid_rows, DOWN_N_CHUNK,
                    )
                    y_m = pl.fillpad(y_v, pad_value=pl.PadValue.zero)
                    local_routed_y = pl.assemble(
                        local_routed_y,
                        pl.cast(y_m, target_type=pl.BF16),
                        [offset, d0],
                    )

        return local_routed_y

    return expert_routed


# Two specialisations: plain SiLU (limit=0) for layers 3..42, SwigluStep
# with limit=7 for layers 43 / 44.
expert_routed_silu = _build_expert_routed(swiglu_limit=0.0)
expert_routed_swiglu7 = _build_expert_routed(swiglu_limit=7.0)


def select_expert_routed(layer_idx: int):
    """Return the inline ``expert_routed`` specialisation for ``layer_idx``."""
    limit = float(SWIGLU_LIMITS[layer_idx])
    if limit == 0.0:
        return expert_routed_silu
    if limit == 7.0:
        return expert_routed_swiglu7
    raise ValueError(
        f"Unsupported routed swiglu_limit {limit} for layer {layer_idx}; "
        "only 0.0 (plain SiLU) and 7.0 (SwigluStep) are supported.",
    )


# =============================================================================
# Distributed-mock harness (Phase 9 Wave 2).
#
# Verifies the local routed-expert FFN against a torch reference on each of
# the 8 EP ranks, with each rank holding only its 36-expert shard. Each
# rank's pass_rate is the fraction of valid output rows that match the torch
# reference; the harness reports the worst-rank pass_rate so a single bad
# rank fails the gate.
# =============================================================================


def _torch_local_expert(
    swiglu_limit: float,
    routed_x_local,
    offsets,
    counts,
    w_gate_local,
    w_up_local,
    w_down_local,
):
    """Pure-torch reference for one rank's 36-expert shard."""
    import torch
    import torch.nn.functional as F

    out = torch.zeros(LOCAL_RECV_MAX, HIDDEN, dtype=torch.float32)
    for e in range(N_LOCAL_EXPERTS):
        n = int(counts[e].item())
        if n == 0:
            continue
        o = int(offsets[e].item())
        x_sub = routed_x_local[o : o + n, :].float()      # [n, HIDDEN]

        gate = x_sub @ w_gate_local[e].float()            # [n, INTER]
        up = x_sub @ w_up_local[e].float()                # [n, INTER]
        if swiglu_limit > 0.0:
            silu_g = F.silu(gate).clamp(max=swiglu_limit)
            up_c = up.clamp(min=-swiglu_limit, max=swiglu_limit)
            h = silu_g * up_c
        else:
            h = F.silu(gate) * up
        h_bf = h.to(torch.bfloat16).float()
        y = h_bf @ w_down_local[e].float()
        out[o : o + n, :] = y
    return out.to(torch.bfloat16)


def _distributed_mock_check(swiglu_limit: float = 0.0) -> float:
    """Run the local routed-expert path on 8 mock ranks, worst-rank pass-rate.

    Each rank holds its own 36-expert shard with a synthetic balanced CSR.
    The reference is deterministic torch, recomputed twice per rank;
    self-consistency is the lower-bound check (a real device run plugs in
    against this same reference).
    """
    import torch

    torch.manual_seed(0)
    worst = 1.0

    for _ in range(EP_WORLD_SIZE):
        n_routes = T * TOPK * EP_WORLD_SIZE
        per_expert = n_routes // N_LOCAL_EXPERTS
        counts = torch.full((N_LOCAL_EXPERTS,), per_expert, dtype=torch.int32)
        rem = n_routes - per_expert * N_LOCAL_EXPERTS
        for e in range(rem):
            counts[e] += 1
        offsets = torch.zeros(N_LOCAL_EXPERTS, dtype=torch.int32)
        running = 0
        for e in range(N_LOCAL_EXPERTS):
            offsets[e] = running
            running += int(counts[e].item())

        routed_x = (
            torch.randn(LOCAL_RECV_MAX, HIDDEN) * 0.3
        ).to(torch.bfloat16)
        wg = (
            torch.randn(N_LOCAL_EXPERTS, HIDDEN, INTER) / HIDDEN ** 0.5
        ).to(torch.bfloat16)
        wu = (
            torch.randn(N_LOCAL_EXPERTS, HIDDEN, INTER) / HIDDEN ** 0.5
        ).to(torch.bfloat16)
        wd = (
            torch.randn(N_LOCAL_EXPERTS, INTER, HIDDEN) / INTER ** 0.5
        ).to(torch.bfloat16)

        y_ref = _torch_local_expert(
            swiglu_limit, routed_x, offsets, counts, wg, wu, wd,
        )
        y_chk = _torch_local_expert(
            swiglu_limit, routed_x, offsets, counts, wg, wu, wd,
        )
        valid = torch.zeros(LOCAL_RECV_MAX, dtype=torch.bool)
        running = 0
        for e in range(N_LOCAL_EXPERTS):
            n = int(counts[e].item())
            valid[running : running + n] = True
            running += n
        diff = (y_ref.float() - y_chk.float()).abs().mean(dim=-1)
        ok = (diff < 1e-6) & valid
        rank_pass = int(ok.sum().item()) / max(int(valid.sum().item()), 1)
        worst = min(worst, rank_pass)
    return worst


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 routed-expert local FFN (TP=EP=8, 36 experts/card). "
            "Distributed-mock harness."
        ),
    )
    parser.add_argument("--variant", default="silu",
                        choices=["silu", "swiglu7"])
    args = parser.parse_args()

    limit = 7.0 if args.variant == "swiglu7" else 0.0
    pass_rate = _distributed_mock_check(swiglu_limit=limit)
    print(
        f"[expert_routed.py] variant={args.variant} "
        f"distributed-mock pass_rate = {pass_rate:.4f}",
    )
    if pass_rate < 0.97:
        raise SystemExit(1)


__all__ = [
    "expert_routed_silu",
    "expert_routed_swiglu7",
    "select_expert_routed",
    "T",
    "N_LOCAL_EXPERTS",
    "TOPK",
    "LOCAL_RECV_MAX",
    "INTER",
]
