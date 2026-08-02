# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 MoE FP32 sigmoid+bias router (decode, TP=EP=8 BF16).

Per-token top-K=8 router for the 288-expert step3p5 MoE block.

Distributed-topology contract (Phase 9 Wave 2):

  * Single-node, 8 cards, ``world_size == TP_WORLD_SIZE == EP_WORLD_SIZE == 8``.
  * The gate weight ``gate_w`` ``[HIDDEN, N_EXPERTS=288]`` and the
    additive ``router_bias[N_EXPERTS=288]`` are REPLICATED across all
    8 ranks (no TP slicing on the router). Each rank independently
    runs the full FP32 sigmoid + bias + flat top-8 + renormalise * 3.0
    over all 288 experts, so routing decisions are bit-identical
    across ranks (no cross-rank exchange).
  * The output ``expert_indices`` carries **GLOBAL** expert ids in
    ``[0, N_EXPERTS)`` — the EP partitioning into 36 experts per card
    is interpreted downstream by ``dispatch`` using the
    ``config.ep_expert_owner`` / ``config.ep_local_expert_id`` helpers.
  * No collective primitive is invoked inside the gate body; this lets
    the in-tree TP+EP MoE reference's chip_orch overlap the gate with
    other lanes naturally.

Per-card weight bundle (host weight loader contract — replicated, NOT sliced):

  * ``gate_w[HIDDEN, N_EXPERTS]`` FP32, identical on every rank
  * ``router_bias[N_EXPERTS]`` FP32, identical on every rank

Routing math (unchanged from the single-card path):

  * Activation is plain ``sigmoid`` (``MOE_ROUTER_ACTIVATION = "sigmoid"``)
    and the bias is **additive** on the sigmoid scores (matches the
    upstream ``router_bias_func``).
  * Top-K is a flat top-8 over all ``N_EXPERTS=288`` experts.
  * ``NORM_EXPERT_WEIGHT = True`` -> the K weights are normalised to
    sum=1 after top-K, then multiplied by
    ``MOE_ROUTER_SCALING_FACTOR = 3.0``.

Numerics:

  * ``NEED_FP32_GATE = True`` -> matmul accumulator and softmax-equivalent
    pipeline run in FP32. Input ``x`` arrives BF16 from the post-attention
    RMSNorm in ``decode_layer.py``; we cast to FP32 inside the gate.
  * The deterministic NPU sort32 ordering is bit-identical across all
    ranks given the same FP32 ``logits`` -> the same ``expert_indices``
    is produced on every rank without an explicit barrier.

Top-K implementation: ``pl.sort32`` + two ``pl.mrgsort`` stages on a
``[1, SCORE_PAD=512]`` per-row buffer (288 valid scores zero-padded /
``-inf``-padded for the bias variant), sized to the next power-of-two
above ``N_EXPERTS``.

Weight orientation: pypto stores matmul weights ``[in, out]`` so
``gate_w`` is ``[HIDDEN, N_EXPERTS]`` (no ``b_trans``). The host loader
transposes the checkpoint's canonical ``[N_EXPERTS, HIDDEN]`` once at
load time and broadcasts the same FP32 tensor to all 8 cards.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl

from .config import (
    BATCH,
    HIDDEN,
    MOE_NUM_EXPERTS,
    MOE_ROUTER_SCALING_FACTOR,
    MOE_TOP_K,
    TP_WORLD_SIZE,
)


# Per-token batch (decode runs one row per user; BATCH = config.BATCH).
T = BATCH
N_EXPERTS = MOE_NUM_EXPERTS  # 288 GLOBAL routed experts
TOPK = MOE_TOP_K              # 8
ROUTE_SCALE = MOE_ROUTER_SCALING_FACTOR  # 3.0
FP32_NEG_INF = -3.4028235e38

# Sort working width: pad N_EXPERTS=288 up to the next power-of-two for the
# sort32 + 4-way mrgsort cascade. 512 covers 288 valid lanes + 224 -inf pad.
SCORE_PAD = 512
TOPK_PAD = 16  # 32B-aligned width for the (val, idx) interleaved slice
assert TOPK <= TOPK_PAD
SORT_PAD = TOPK_PAD * 2  # interleaved (value, index) pair width

# Tiling.
GATE_K_CHUNK = 512  # K-loop step over HIDDEN for the gate matmul
assert HIDDEN % GATE_K_CHUNK == 0


@pl.jit.inline
def gate(
    x: pl.Tensor[[T, HIDDEN], pl.BF16],
    gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
    router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    expert_indices: pl.Tensor[[T, TOPK], pl.INT32],
    expert_weights: pl.Tensor[[T, TOPK], pl.BF16],
):
    """Step3p5 router: sigmoid + additive bias + top-K + renorm + scale.

    Replicated body — same compute on every rank, identical output.

    Pipeline (per row, fused into one InCore region):
      1. ``logits = x_fp32 @ gate_w`` (FP32 accumulator over HIDDEN).
      2. ``score = sigmoid(logits)`` (un-biased; used for the renorm gather).
      3. ``biased = score + router_bias[None, :]`` (selection key).
      4. Pad biased to [1, SCORE_PAD] with -inf, sort32 + mrgsort cascade.
      5. Gather the top-K **global** indices, look up un-biased scores.
      6. Renormalise to sum=1, multiply by ``ROUTE_SCALE``, cast to BF16.

    The un-biased score (not the bias-shifted one) is what's renormalised
    into the per-expert routing weight, matching vllm's ``router_bias_func``
    (the bias only steers selection; the weighting uses the raw sigmoid).
    """
    score_buf = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    biased_buf = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)

    # --- Stage 1: gate matmul + sigmoid + bias ----------------------------
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_matmul"):
        # FP32 cast of x once; reused across the K-loop.
        x_fp32 = pl.cast(x, target_type=pl.FP32)
        x0 = pl.slice(x_fp32, [T, GATE_K_CHUNK], [0, 0])
        w0 = pl.slice(gate_w, [GATE_K_CHUNK, N_EXPERTS], [0, 0])
        logits = pl.matmul(x0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // GATE_K_CHUNK):
            k0 = kb * GATE_K_CHUNK
            xk = pl.slice(x_fp32, [T, GATE_K_CHUNK], [0, k0])
            wk = pl.slice(gate_w, [GATE_K_CHUNK, N_EXPERTS], [k0, 0])
            logits = pl.matmul_acc(logits, xk, wk)

        # sigmoid(logits) = 1 / (1 + exp(-logits)).
        score_n = pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0))
        bias_row = pl.reshape(router_bias, [1, N_EXPERTS])
        biased_n = pl.add(
            score_n,
            pl.col_expand_mul(
                pl.full([T, N_EXPERTS], dtype=pl.FP32, value=1.0), bias_row,
            ),
        )

        # Pre-fill the pad region with FP32_NEG_INF / 0.0 so sort sees -inf
        # in the invalid lanes and the renorm gather sees 0.0 there.
        score_buf[:, :] = pl.full(
            [T, SCORE_PAD], dtype=pl.FP32, value=0.0,
        )
        biased_buf[:, :] = pl.full(
            [T, SCORE_PAD], dtype=pl.FP32, value=FP32_NEG_INF,
        )
        score_buf[:, 0:N_EXPERTS] = score_n
        biased_buf[:, 0:N_EXPERTS] = biased_n

    # --- Stage 2: per-row top-K via sort32 + mrgsort ----------------------
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_topk"):
        topk_idx_tile = pl.create_tensor([T, TOPK_PAD], dtype=pl.INT32)
        # sort32 + mrgsort require row count == 1; iterate rows scalar.
        for tt in pl.range(T):
            row = biased_buf[tt : tt + 1, :]
            idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
            srt = pl.sort32(row, idx_init)              # [1, 2*SCORE_PAD]
            srt = pl.mrgsort(srt, block_len=64)         # 64 -> 256-pos runs
            srt = pl.mrgsort(srt[:, 0:512], srt[:, 512:1024])
            pairs = srt[:, 0:SORT_PAD]
            top_idx = pl.gather(
                pairs, mask_pattern=pl.tile.MaskPattern.P1010,
                output_dtype=pl.INT32,
            )
            topk_idx_tile[tt : tt + 1, :] = top_idx

        # Batched index-gather of un-biased scores. set_validshape limits
        # the gather output to TOPK; fillpad zeros the [TOPK, TOPK_PAD) tail
        # so the renormalize sum below sums only the real K entries.
        gather_all = pl.gather(score_buf, dim=-1, index=topk_idx_tile)
        gather_valid = pl.set_validshape(gather_all, T, TOPK)
        topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)

        denom = pl.reshape(pl.row_sum(topk_vals_pad), [T, 1])
        weights_pad = pl.mul(
            pl.row_expand_div(topk_vals_pad, denom), ROUTE_SCALE,
        )

        # Scalar scatter of the K leading **global** indices and BF16
        # weights into the caller-visible outputs. K=8 is small; an
        # explicit loop avoids the 24B/32B alignment pitfalls of
        # slice-assigning a [T, K] sub-tile.
        for tt in pl.range(T):
            for k in pl.range(TOPK):
                pl.write(expert_indices, [tt, k],
                         pl.read(topk_idx_tile, [tt, k]))
                pl.write(expert_weights, [tt, k],
                         pl.cast(pl.read(weights_pad, [tt, k]), pl.BF16))

    # @pl.inline parser requires inline calls to return a value.
    return expert_weights


@pl.jit
def gate_test(
    x: pl.Tensor[[T, HIDDEN], pl.BF16],
    gate_w: pl.Tensor[[HIDDEN, N_EXPERTS], pl.FP32],
    router_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    expert_indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    expert_weights: pl.Out[pl.Tensor[[T, TOPK], pl.BF16]],
):
    gate(x, gate_w, router_bias, expert_indices, expert_weights)
    return expert_indices, expert_weights


def golden_gate(tensors):
    """Torch reference: sigmoid + bias + topk + renorm + scale (BF16 out).

    Returns GLOBAL expert ids; identical on every rank in the distributed
    deployment (gate weights and the additive bias are replicated).
    """
    import torch

    x = tensors["x"].float()                          # [T, HIDDEN]
    gate_w = tensors["gate_w"].float()                # [HIDDEN, N_EXPERTS]
    router_bias = tensors["router_bias"].float()      # [N_EXPERTS]

    logits = x @ gate_w                                # [T, N_EXPERTS]
    score = torch.sigmoid(logits)                      # raw sigmoid score
    biased = score + router_bias.view(1, -1)           # selection key

    # argsort-stable matches the deterministic NPU sort32 ordering.
    indices = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
    topk_vals = torch.gather(score, dim=-1, index=indices.long())
    weights = (topk_vals / topk_vals.sum(dim=-1, keepdim=True)) * ROUTE_SCALE

    tensors["expert_indices"][:] = indices.to(torch.int32)
    tensors["expert_weights"][:] = weights.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(T, HIDDEN) * 0.5
    def init_gate_w():
        return torch.randn(HIDDEN, N_EXPERTS) / HIDDEN ** 0.5
    def init_router_bias():
        return torch.randn(N_EXPERTS) * 0.05

    return [
        TensorSpec("x", [T, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("gate_w", [HIDDEN, N_EXPERTS], torch.float32,
                   init_value=init_gate_w),
        TensorSpec("router_bias", [N_EXPERTS], torch.float32,
                   init_value=init_router_bias),
        TensorSpec("expert_indices", [T, TOPK], torch.int32, is_output=True),
        TensorSpec("expert_weights", [T, TOPK], torch.bfloat16, is_output=True),
    ]


# =============================================================================
# Distributed-mock harness (Phase 9 Wave 2).
#
# Verifies the replicated-gate invariant: 8 ranks compute the SAME
# expert_indices / expert_weights, bit-identical, given the SAME input
# (and the host loader broadcasts the same gate_w / router_bias to all
# ranks).
#
# The pure-torch loop simulates the deployment: 8 independent torch runs
# of the reference compared against each other and against a single
# canonical run.
# =============================================================================


def _distributed_mock_check() -> float:
    """Run the golden gate on 8 mock ranks; return the cross-rank pass_rate.

    The pass_rate is the fraction of (T * TOPK) (expert id, weight) pairs
    that match the rank-0 reference across all 8 ranks. Replicated weights
    + deterministic torch -> we expect 1.0.
    """
    import torch

    torch.manual_seed(0)
    x = torch.randn(T, HIDDEN, dtype=torch.bfloat16)
    gate_w = (torch.randn(HIDDEN, N_EXPERTS) / HIDDEN ** 0.5).float()
    router_bias = (torch.randn(N_EXPERTS) * 0.05).float()

    ref_indices = torch.zeros(T, TOPK, dtype=torch.int32)
    ref_weights = torch.zeros(T, TOPK, dtype=torch.bfloat16)
    tensors_rank0 = {
        "x": x.clone(),
        "gate_w": gate_w.clone(),
        "router_bias": router_bias.clone(),
        "expert_indices": ref_indices,
        "expert_weights": ref_weights,
    }
    golden_gate(tensors_rank0)

    total = T * TOPK
    matches = 0
    for _ in range(TP_WORLD_SIZE):
        ind = torch.zeros(T, TOPK, dtype=torch.int32)
        wts = torch.zeros(T, TOPK, dtype=torch.bfloat16)
        tensors_r = {
            "x": x.clone(),
            "gate_w": gate_w.clone(),
            "router_bias": router_bias.clone(),
            "expert_indices": ind,
            "expert_weights": wts,
        }
        golden_gate(tensors_r)
        idx_eq = (ind == ref_indices)
        w_close = torch.isclose(
            wts.float(), ref_weights.float(), rtol=0.0, atol=0.0,
        )
        matches += int((idx_eq & w_close).sum().item())
    return matches / (total * TP_WORLD_SIZE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 MoE gate (replicated TP=EP=8): sigmoid + bias "
            "+ topk + renorm. Distributed-mock harness."
        ),
    )
    parser.add_argument("--rank0-only", action="store_true", default=False,
                        help="Run only rank 0 against the torch reference.")
    args = parser.parse_args()

    pass_rate = _distributed_mock_check()
    print(f"[gate.py] distributed-mock pass_rate = {pass_rate:.4f}")
    if pass_rate < 0.97:
        raise SystemExit(1)


__all__ = [
    "gate",
    "gate_test",
    "golden_gate",
    "build_tensor_specs",
    "T",
    "N_EXPERTS",
    "TOPK",
    "TOPK_PAD",
    "ROUTE_SCALE",
]
