# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 shared-expert MLP (decode, TP=EP=8 BF16).

A 1280-dim FFN sliced across the TP group: each of the 8 cards owns
``SHARE_EXPERT_DIM_LOCAL = 1280 / 8 = 160`` of the intermediate
features. The shard runs on the full ``[T, HIDDEN]`` token batch
(replicated input) and produces the partial ``sh_y_shard[T, HIDDEN]``
contribution; the per-card shards are then summed via
``collectives.tp_all_reduce`` so every rank ends up with the same
fully-reduced shared-expert output.

Distributed-topology contract (Phase 9 Wave 2):

  * TP_WORLD_SIZE = 8, SHARE_EXPERT_DIM_LOCAL = 160 (1280 / 8).
  * Input ``x[T, HIDDEN]`` is REPLICATED across all 8 ranks (no token
    sharding on the shared expert).
  * Output ``sh_y[T, HIDDEN]`` is FULLY REDUCED (identical across
    ranks) after the trailing ``tp_all_reduce`` step.
  * The local-compute body is ``@pl.jit.inline``; the
    ``tp_all_reduce`` is driven by the caller (``moe.py``'s
    ``@pl.program`` class) from an InCore context so the window-buffer
    contract matches the in-tree TP+EP MoE reference's pattern.

Per-card weight bundle (host weight loader contract — sliced on TP):

  * ``w_gate[HIDDEN, SHARE_EXPERT_DIM_LOCAL]`` BF16
  * ``w_up  [HIDDEN, SHARE_EXPERT_DIM_LOCAL]`` BF16
  * ``w_down[SHARE_EXPERT_DIM_LOCAL, HIDDEN]`` BF16

The host loader splits the canonical 1280-dim shared-expert tensors
along the intermediate axis: rank ``r`` gets columns
``[r * 160 .. (r + 1) * 160)`` of ``w_gate`` / ``w_up`` and rows
``[r * 160 .. (r + 1) * 160)`` of ``w_down``. The local matmul chain
is identical to the single-card path with INTER=160 substituted for
INTER=1280; ``w_gate``/``w_up`` produce the local 160-dim
intermediate, ``w_down`` reduces the local 160 lanes into a
partial ``[T, HIDDEN]`` shard, and the sum across the TP group is
the true 1280 = 8 * 160 reduction.

Pipeline per token tile (per rank, before the TP all-reduce):

    h_local = activation(x @ w_gate_local) * (x @ w_up_local)
            # [T, 160] -- per-card share of the intermediate
    sh_y_shard = h_local @ w_down_local
            # [T, HIDDEN] -- per-card share of the output

Then ``tp_all_reduce`` sums the 8 shards into a per-rank-identical
``sh_y[T, HIDDEN]``.

Activation selection (compile-time const, baked from
``SWIGLU_LIMITS_SHARED[layer_idx]``):
  - ``0.0``  -> plain SiLU on gate, no clamp on up.
  - ``16.0`` -> SwigluStep with limit=16 (only at layer 44).

The clamp formula matches vllm ``SwigluStepAndMul.forward_native``:
    silu(gate).clamp(max=L) * up.clamp(min=-L, max=L)
i.e. silu first, then clamp the silu output and clamp ``up`` to
``[-L, L]``. Per-card semantics are preserved by the activation: the
``silu * up`` product is computed pointwise on each card's 160-lane
intermediate slice; the TP reduce sums the 8 sliced outputs to recover
the true 1280-wide product without an extra cross-rank exchange on
the intermediate.

Distinguishing properties of this shared-expert path:
  - No INT8 quantization or per-channel dequant scales -- BF16
    weights consumed directly.
  - Single dense-tile path per rank; no per-token amax / requant pass.
  - The tp_all_reduce is performed by the caller; this file's inline
    body returns the un-reduced per-rank shard.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl

from .config import (
    BATCH,
    HIDDEN,
    SHARE_EXPERT_DIM_LOCAL,
    SWIGLU_LIMITS_SHARED,
    TP_WORLD_SIZE,
)


T = BATCH
INTER = SHARE_EXPERT_DIM_LOCAL  # 160 (== SHARE_EXPERT_DIM / TP_WORLD_SIZE)

# Tiling. INTER=160 is below the canonical GATE_N_CHUNK=256, so we set the
# N-tile equal to the full per-card intermediate slice and the K-tile to 256
# (HIDDEN=4096 factors as 16 * 256). The down-direction K-tile equals INTER
# in a single chunk, and the N-tile is 256 (HIDDEN / 256 = 16).
GATE_K_CHUNK = 256
GATE_N_CHUNK = INTER      # 160 — one N tile covers the whole local slice
DOWN_K_CHUNK = INTER      # 160 — one K tile covers the whole local slice
DOWN_N_CHUNK = 256
assert HIDDEN % GATE_K_CHUNK == 0
assert HIDDEN % DOWN_N_CHUNK == 0


def _build_expert_shared(swiglu_limit: float):
    """Factory baking ``SWIGLU_LIMITS_SHARED[layer_idx]`` as a const."""
    use_swiglu_step = swiglu_limit > 0.0

    @pl.jit.inline
    def expert_shared_local(
        x: pl.Tensor[[T, HIDDEN], pl.BF16],
        w_gate: pl.Tensor[[HIDDEN, INTER], pl.BF16],
        w_up: pl.Tensor[[HIDDEN, INTER], pl.BF16],
        w_down: pl.Tensor[[INTER, HIDDEN], pl.BF16],
        sh_y_shard: pl.Tensor[[T, HIDDEN], pl.BF16],
    ):
        """Per-rank local shared-expert FFN; output is the per-card shard.

        The returned ``sh_y_shard`` is the un-reduced partial
        ``[T, HIDDEN]`` contribution. The caller is responsible for
        running ``tp_all_reduce`` across the TP group so every rank
        ends up with the summed result.
        """
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up"):
            h_tile = pl.create_tensor([T, INTER], dtype=pl.BF16)

            # Only one N-tile because INTER (160) <= GATE_N_CHUNK (160).
            x0 = pl.slice(x, [T, GATE_K_CHUNK], [0, 0])
            wg0 = pl.slice(
                w_gate, [GATE_K_CHUNK, GATE_N_CHUNK], [0, 0],
            )
            wu0 = pl.slice(
                w_up, [GATE_K_CHUNK, GATE_N_CHUNK], [0, 0],
            )
            gate_acc = pl.matmul(x0, wg0, out_dtype=pl.FP32)
            up_acc = pl.matmul(x0, wu0, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN // GATE_K_CHUNK):
                k0 = kb * GATE_K_CHUNK
                xk = pl.slice(x, [T, GATE_K_CHUNK], [0, k0])
                wgk = pl.slice(
                    w_gate, [GATE_K_CHUNK, GATE_N_CHUNK], [k0, 0],
                )
                wuk = pl.slice(
                    w_up, [GATE_K_CHUNK, GATE_N_CHUNK], [k0, 0],
                )
                gate_acc = pl.matmul_acc(gate_acc, xk, wgk)
                up_acc = pl.matmul_acc(up_acc, xk, wuk)

            sigmoid = pl.recip(
                pl.add(pl.exp(pl.neg(gate_acc)), 1.0),
            )
            silu = pl.mul(gate_acc, sigmoid)
            if use_swiglu_step:
                silu_c = pl.minimum(silu, swiglu_limit)
                up_c = pl.maximum(
                    pl.minimum(up_acc, swiglu_limit),
                    -swiglu_limit,
                )
                gated = pl.mul(silu_c, up_c)
            else:
                gated = pl.mul(silu, up_acc)

            h_tile[:, 0:GATE_N_CHUNK] = pl.cast(
                gated, target_type=pl.BF16,
            )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_down"):
            # Single K-tile because DOWN_K_CHUNK == INTER == 160.
            for db in pl.range(HIDDEN // DOWN_N_CHUNK):
                d0 = db * DOWN_N_CHUNK
                h0 = pl.slice(h_tile, [T, DOWN_K_CHUNK], [0, 0])
                wd0 = pl.slice(
                    w_down, [DOWN_K_CHUNK, DOWN_N_CHUNK], [0, d0],
                )
                y_acc = pl.matmul(h0, wd0, out_dtype=pl.FP32)
                sh_y_shard = pl.assemble(
                    sh_y_shard,
                    pl.cast(y_acc, target_type=pl.BF16),
                    [0, d0],
                )

        return sh_y_shard

    return expert_shared_local


expert_shared_silu = _build_expert_shared(swiglu_limit=0.0)
expert_shared_swiglu16 = _build_expert_shared(swiglu_limit=16.0)


def select_expert_shared(layer_idx: int):
    """Return the ``expert_shared_local`` specialisation for ``layer_idx``."""
    limit = float(SWIGLU_LIMITS_SHARED[layer_idx])
    if limit == 0.0:
        return expert_shared_silu
    if limit == 16.0:
        return expert_shared_swiglu16
    raise ValueError(
        f"Unsupported shared swiglu_limit {limit} for layer {layer_idx}; "
        "only 0.0 (plain SiLU) and 16.0 (SwigluStep) are supported.",
    )


# =============================================================================
# Distributed-mock harness (Phase 9 Wave 2).
#
# Verifies the TP-sliced shared expert: per-rank local compute over
# 160-dim slices, then sum-reduce across the 8 ranks should reproduce the
# full 1280-dim reference output (within BF16 rtol/atol).
# =============================================================================


def _torch_shared_local(swiglu_limit: float, x, w_gate, w_up, w_down):
    """One rank's torch reference: 160-dim FFN; returns the unreduced shard."""
    import torch
    import torch.nn.functional as F

    gate = x.float() @ w_gate.float()             # [T, 160]
    up = x.float() @ w_up.float()                 # [T, 160]
    if swiglu_limit > 0.0:
        silu_g = F.silu(gate).clamp(max=swiglu_limit)
        up_c = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        h = silu_g * up_c
    else:
        h = F.silu(gate) * up
    h_bf = h.to(torch.bfloat16).float()
    y_shard = h_bf @ w_down.float()               # [T, HIDDEN]
    return y_shard.to(torch.bfloat16)


def _distributed_mock_check(swiglu_limit: float = 0.0) -> float:
    """Sum the 8 rank-local shards and compare to a full-dim reference.

    Pass-rate = fraction of (T * HIDDEN) elements that match the
    full-1280-dim reference within rtol=5e-3, atol=5e-3 BF16 tolerance.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    full_inter = INTER * TP_WORLD_SIZE  # 1280
    x = (torch.randn(T, HIDDEN) * 0.3).to(torch.bfloat16)
    w_gate_full = (
        torch.randn(HIDDEN, full_inter) / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_up_full = (
        torch.randn(HIDDEN, full_inter) / HIDDEN ** 0.5
    ).to(torch.bfloat16)
    w_down_full = (
        torch.randn(full_inter, HIDDEN) / full_inter ** 0.5
    ).to(torch.bfloat16)

    # Full reference (single-card semantics) for comparison.
    gate_full = x.float() @ w_gate_full.float()
    up_full = x.float() @ w_up_full.float()
    if swiglu_limit > 0.0:
        silu_g = F.silu(gate_full).clamp(max=swiglu_limit)
        up_c = up_full.clamp(min=-swiglu_limit, max=swiglu_limit)
        h = silu_g * up_c
    else:
        h = F.silu(gate_full) * up_full
    h_bf = h.to(torch.bfloat16).float()
    y_ref = (h_bf @ w_down_full.float()).to(torch.bfloat16)

    # Per-rank shards, then sum-reduce.
    reduced = torch.zeros(T, HIDDEN, dtype=torch.float32)
    for r in range(TP_WORLD_SIZE):
        col_lo = r * INTER
        col_hi = (r + 1) * INTER
        wg = w_gate_full[:, col_lo:col_hi].contiguous()
        wu = w_up_full[:, col_lo:col_hi].contiguous()
        wd = w_down_full[col_lo:col_hi, :].contiguous()
        shard = _torch_shared_local(swiglu_limit, x, wg, wu, wd)
        reduced += shard.float()
    reduced_bf = reduced.to(torch.bfloat16).float()

    diff = (reduced_bf - y_ref.float()).abs()
    tol = 5e-3 + 5e-3 * y_ref.float().abs()
    matches = int((diff <= tol).sum().item())
    return matches / (T * HIDDEN)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 shared-expert (TP=EP=8, 160 lanes/card + "
            "tp_all_reduce). Distributed-mock harness."
        ),
    )
    parser.add_argument("--variant", default="silu",
                        choices=["silu", "swiglu16"])
    args = parser.parse_args()

    limit = 16.0 if args.variant == "swiglu16" else 0.0
    pass_rate = _distributed_mock_check(swiglu_limit=limit)
    print(
        f"[expert_shared.py] variant={args.variant} "
        f"distributed-mock pass_rate = {pass_rate:.4f}",
    )
    if pass_rate < 0.97:
        raise SystemExit(1)


__all__ = [
    "expert_shared_silu",
    "expert_shared_swiglu16",
    "select_expert_shared",
    "T",
    "INTER",
    "GATE_K_CHUNK",
    "GATE_N_CHUNK",
    "DOWN_K_CHUNK",
    "DOWN_N_CHUNK",
]
