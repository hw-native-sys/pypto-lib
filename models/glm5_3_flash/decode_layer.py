# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""The sparse-layer MLP, assembled: routed experts (EP) + shared expert (TP) + reduce.

A GLM-5.3-Flash MoE layer's feed-forward half is two branches that meet once:

* the **routed** branch is expert-parallel — the router's top-8 sends each token's
  INT8 payload to the ranks that own its experts, the experts run, and every result
  comes back to the token's rank. ``decode_moe.py`` owns that whole path
  (pack -> arrive -> regroup -> ``expert_routed`` -> return -> combine) and leaves
  the completed FP32 sum in ``routed``.
* the **shared** expert runs on every rank over its own tokens but is sharded on
  the intermediate axis, so each rank holds one row-parallel summand of the full
  hidden width. ``expert_shared.py`` computes the shard; ``attention_tp.py``'s
  ``tp_all_reduce`` sums the group's shards into BF16.

``moe_layer_add`` then adds the two, with the inactive capacity rows written as
zeros. The residual / hyper-connection path is downstream and out of scope here: this
file ends at the MLP contribution to the sublayer. The router is the head of the
block, so the entry runs ``moe_gate`` (deferred norm + router + INT8 view) and
then the two branches over its outputs.

Scope policy: the hyper-connection coefficients, ``hc_pre`` / ``hc_post``,
attention and the INT8-emitting RMSNorm are **not** pypto work items here — the
test sources their outputs from the torch references in ``golden.py`` (the mHC
stream is folded to ``x_mixed`` with ``hc_seed`` / ``hc_mixes`` / ``hc_pre`` in
the fixture), so no non-MoE kernel has to exist for the block to be verified.

The L3 entry runs the assembly twice over one set of windows, so the epoch-keyed
EP and TP barriers are exercised for the multi-layer reuse they exist for, and
compares both passes against ``golden_decode_layer``.
"""

import sys
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
import torch

# ``--tp`` / ``--ep`` are read from argv by config.py at import time and every
# sub-kernel inherits the frozen shapes. config.py's own default is the 16-die
# deployment shape, while the CI sweeps run each entry at its default world size
# (ep2 / 2-card, see ``# ci: devices=2``), so pin that bring-up default before
# the first ``models.glm5_3_flash`` import when the command line pins neither axis.
if not any(tok in ("--tp", "--ep") or tok.startswith(("--tp=", "--ep=")) for tok in sys.argv):
    sys.argv += ["--tp", "2", "--ep", "2"]

from models.glm5_3_flash.attention_tp import tp_all_reduce
from models.glm5_3_flash.config import D, EP_SIZE, FLASH, HC_DIM, MIX_HC, N_EXPERTS, TOPK
from models.glm5_3_flash.gate import moe_gate
from models.glm5_3_flash.decode_moe import (
    AUX_PAD,
    COUNT_PAD,
    LANE_ROWS,
    MOE_INTER,
    N_LOCAL_EXPERTS,
    N_RANKS,
    N_ROUTES,
    T,
    moe_arrive,
    moe_combine,
    moe_dispatch,
    moe_pack,
    moe_return,
    moe_return_notify,
    gen_routed_weight,
)
from models.glm5_3_flash.expert_routed import expert_routed
from models.glm5_3_flash.expert_shared import LOCAL_MOE_INTER, expert_shared
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX

# This directory owns a ``golden.py`` reference module, so the repository-root
# ``golden`` harness package must come first on the path before any harness import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pl.jit.inline
def moe_layer_add(
    routed: pl.Tensor[[T, D], pl.FP32],
    shared_full: pl.Tensor[[T, D], pl.BF16],
    y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Add the routed sum and the TP-reduced shared expert, capacity rows zeroed."""
    active = pl.cast(num_tokens, pl.INDEX)
    if active < 0:
        active = pl.cast(0, pl.INDEX)
    if active > T:
        active = pl.cast(T, pl.INDEX)

    with pl.spmd(T, name_hint="moe_layer_add") as _add_tid:
        t = pl.tile.get_block_idx()
        acc = pl.full([1, D], dtype=pl.FP32, value=0.0)
        if t < active:
            acc = pl.add(routed[t : t + 1, :], pl.cast(shared_full[t : t + 1, :], pl.FP32, mode="none"))
        y[t : t + 1, :] = acc

    return y


@pl.jit(auto_scope=False)
def l2_decode_layer(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    w_gate_up: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w_gate_up: pl.Tensor[[2 * LOCAL_MOE_INTER, D], pl.INT8],
    shared_w_gate_up_scale: pl.Tensor[[2 * LOCAL_MOE_INTER], pl.FP32],
    shared_w_down: pl.Tensor[[D, LOCAL_MOE_INTER], pl.INT8],
    shared_w_down_scale: pl.Tensor[[D], pl.FP32],
    y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    exchange: pld.DistributedTensor[[T, D], pl.FP32],
    tp_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """One rank's orchestration of both branches over a sparse layer."""
    arrive_x = pl.create_tensor([LANE_ROWS, D], dtype=pl.INT8)
    arrive_scale = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    arrive_weight = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    arrive_route = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    arrive_expert = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    arrive_src = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    arrive_total = pl.create_tensor([1], dtype=pl.INT32)
    recv_x = pl.create_tensor([LANE_ROWS, D], dtype=pl.INT8)
    recv_scale = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    recv_weight = pl.create_tensor([LANE_ROWS, 1], dtype=pl.FP32)
    recv_src = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    recv_route = pl.create_tensor([LANE_ROWS, 1], dtype=pl.INT32)
    expert_offsets = pl.create_tensor([N_LOCAL_EXPERTS + 1], dtype=pl.INT32)
    expert_out = pl.create_tensor([LANE_ROWS, D], dtype=pl.FP32)
    routed = pl.create_tensor([T, D], dtype=pl.FP32)
    shared_out = pl.create_tensor([T, D], dtype=pl.FP32)
    shared_full = pl.create_tensor([T, D], dtype=pl.BF16)
    route_weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    route_indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    x_int8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_scale = pl.create_tensor([T, 1], dtype=pl.FP32)

    # Block head: deferred norm + router + the INT8 view both branches reuse.
    moe_gate(
        x_mixed, norm_weight, gate_weight, correction_bias,
        route_weights, route_indices, x_int8, x_scale, num_tokens,
    )

    # Routed branch: dispatch this rank's routes, run its experts, return and sum.
    pack_tid = moe_pack(
        route_indices, route_weights, x_int8, x_scale,
        lane_x, lane_aux, lane_count, lane_ready,
        num_tokens, num_ranks, my_rank, epoch,
    )
    moe_arrive(
        lane_x, lane_aux, lane_count, lane_ready,
        arrive_x, arrive_scale, arrive_weight, arrive_route, arrive_expert, arrive_src,
        arrive_total, num_ranks, my_rank, epoch, pack_tid,
    )
    moe_dispatch(
        arrive_expert, arrive_src, arrive_route, arrive_x, arrive_scale, arrive_weight,
        arrive_total,
        recv_x, recv_scale, recv_weight, recv_src, recv_route, expert_offsets,
    )
    expert_routed(
        recv_x, recv_scale,
        w_gate_up, w_gate_up_scale,
        w_down, w_down_scale,
        expert_offsets,
        recv_weight,
        expert_out,
    )
    return_tid = moe_return(expert_out, recv_src, recv_route, slot_out, arrive_total)
    sync_tid = moe_return_notify(return_ready, num_ranks, my_rank, epoch, return_tid)
    moe_combine(
        slot_out, return_ready, routed, num_tokens, num_ranks, my_rank, epoch, sync_tid
    )

    # Shared branch: the local TP shard, then the group reduction.
    expert_shared(
        x_int8, x_scale,
        shared_w_gate_up, shared_w_gate_up_scale,
        shared_w_down, shared_w_down_scale,
        shared_out,
    )
    tp_all_reduce(
        shared_out, exchange, tp_arrived, shared_full,
        num_tokens, my_rank, 0, epoch,
    )

    moe_layer_add(routed, shared_full, y, num_tokens)
    return y


@pl.jit.host
def l3_decode_layer(
    x_mixed: pl.Tensor[[N_RANKS, T, D], pl.BF16],
    norm_weight: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_weight: pl.Tensor[[N_RANKS, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_RANKS, N_EXPERTS], pl.FP32],
    w_gate_up: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w_gate_up: pl.Tensor[[N_RANKS, 2 * LOCAL_MOE_INTER, D], pl.INT8],
    shared_w_gate_up_scale: pl.Tensor[[N_RANKS, 2 * LOCAL_MOE_INTER], pl.FP32],
    shared_w_down: pl.Tensor[[N_RANKS, D, LOCAL_MOE_INTER], pl.INT8],
    shared_w_down_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
):
    """Launch one rank's layer orchestration per chip, sharing the window buffers."""
    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([N_RANKS, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    exchange_buf = pld.alloc_window_buffer([T, D], dtype=pl.FP32)
    tp_arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)

    # Two calls over the same windows: the epoch-keyed EP and TP barriers must
    # order call N's readers before call N+1's writers.
    for call in pl.range(2):
        for r in pl.range(pld.world_size()):
            lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
            lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
            lane_count = pld.window(lane_count_buf, [N_RANKS, COUNT_PAD], dtype=pl.INT32)
            lane_ready = pld.window(lane_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            slot_out = pld.window(slot_out_buf, [N_ROUTES, D], dtype=pl.FP32)
            return_ready = pld.window(return_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            exchange = pld.window(exchange_buf, [T, D], dtype=pl.FP32)
            tp_arrived = pld.window(tp_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
            l2_decode_layer(
                x_mixed[r], norm_weight[r], gate_weight[r], correction_bias[r],
                w_gate_up[r], w_gate_up_scale[r], w_down[r], w_down_scale[r],
                shared_w_gate_up[r], shared_w_gate_up_scale[r],
                shared_w_down[r], shared_w_down_scale[r],
                y[r],
                lane_x, lane_aux, lane_count, lane_ready, slot_out, return_ready,
                exchange, tp_arrived,
                num_tokens, num_ranks, r, pl.cast(call + 1, pl.INT32),
                device=r,
            )


def golden_decode_layer(tensors):
    """Routed branch + shared shard + TP reduce, per rank.

    The routed half is ``decode_moe_ep``'s golden; the shared half reuses
    ``expert_shared``'s golden per rank and then ``attention_tp``'s stack
    reduction, so every number has exactly one reference implementation.
    """
    from models.glm5_3_flash.attention_tp import golden_tp_all_reduce
    from models.glm5_3_flash.decode_moe import golden_routed_ep
    from models.glm5_3_flash.expert_shared import golden_expert_shared_fn
    from models.glm5_3_flash.golden import gate as gate_ref, rms_norm
    from models.glm5_3_flash.quantization import quantize_per_token_int8

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # Block head, per rank, with the same torch references gate.py's test uses.
    route_indices = torch.zeros(N_RANKS, T, TOPK, dtype=torch.int32)
    route_weights = torch.zeros(N_RANKS, T, TOPK, dtype=torch.float32)
    x_int8 = torch.zeros(N_RANKS, T, D, dtype=torch.int8)
    x_scale = torch.zeros(N_RANKS, T, 1, dtype=torch.float32)
    for r in range(N_RANKS):
        x_norm = rms_norm(tensors["x_mixed"][r], tensors["norm_weight"][r])
        weights, indices = gate_ref(x_norm, tensors["gate_weight"][r], tensors["correction_bias"][r])
        route_weights[r], route_indices[r] = weights, indices
        # Kernel-faithful INT8 view: gate.py quantizes ``xg = x * gamma`` in FP32
        # (the deferred norm rides the dequant scale), while the natural reference
        # quantizes the BF16-rounded normalized activation. The two differ by
        # +/-1 LSB on rounding boundaries — the gate test tolerates that, but a
        # composed ``y`` comparison amplifies it. Mirror the kernel so this test
        # isolates the composition, not the quant boundary.
        x_f = tensors["x_mixed"][r].float()
        xg = x_f * tensors["norm_weight"][r].float()
        inverse_rms = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + FLASH.rms_norm_eps)
        x_i8, dequant = quantize_per_token_int8(xg)
        x_int8[r], x_scale[r] = x_i8, dequant * inverse_rms
    if num_tokens < T:
        route_weights[:, num_tokens:] = 0
        route_indices[:, num_tokens:] = 0
        x_scale[:, num_tokens:] = 0

    routed = golden_routed_ep({
        "route_indices": route_indices,
        "route_weights": route_weights,
        "x_int8": x_int8,
        "x_scale": x_scale,
        "w_gate_up": tensors["w_gate_up"],
        "w_gate_up_scale": tensors["w_gate_up_scale"],
        "w_down": tensors["w_down"],
        "w_down_scale": tensors["w_down_scale"],
        "num_tokens": num_tokens,
    })

    shared_partials = torch.zeros(N_RANKS, T, D, dtype=torch.float32)
    for r in range(N_RANKS):
        work = {
            "x_int8": x_int8[r],
            "x_scale": x_scale[r],
            "w_gate_up": tensors["shared_w_gate_up"][r],
            "w_gate_up_scale": tensors["shared_w_gate_up_scale"][r],
            "w_down": tensors["shared_w_down"][r],
            "w_down_scale": tensors["shared_w_down_scale"][r],
            "output": torch.zeros(T, D, dtype=torch.float32),
        }
        golden_expert_shared_fn(work)
        shared_partials[r] = work["output"]

    reduced = golden_tp_all_reduce(shared_partials[:, :num_tokens]).float()
    out = torch.zeros_like(tensors["y"])
    out[:, :num_tokens] = routed[:, :num_tokens] + reduced.unsqueeze(0)
    tensors["y"][:] = out


def build_tensor_specs(num_tokens=T, *, num_ranks=EP_SIZE):
    from golden import ScalarSpec, TensorSpec

    num_tokens = max(0, min(T, int(num_tokens)))

    def gen_shared(shape, dequant_std, chan_cv):
        w = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))
        amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        s = amax / INT8_SCALE_MAX
        w_i8 = torch.round(w / s).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        s = (s.squeeze(-1) * (dequant_std / (w_i8.float() * s).std())).float()
        return w_i8, s

    gen = torch.Generator().manual_seed(7)
    # The hyper-connection side is torch-only by design: fold a seeded BF16
    # stream with the golden's mHC references into the sublayer input, so no
    # non-MoE kernel has to exist for the block to be testable.
    from models.glm5_3_flash.golden import hc_mixes, hc_pre, hc_seed

    emb = torch.randn(N_RANKS, T, D, generator=gen).to(torch.bfloat16)
    hc_fn = torch.randn(N_RANKS, MIX_HC, HC_DIM, generator=gen).to(torch.bfloat16)
    hc_scale = torch.rand(N_RANKS, 3, generator=gen)
    hc_base = torch.randn(N_RANKS, MIX_HC, generator=gen)
    with torch.no_grad():
        x_hc = hc_seed(emb)
        pre_mixes = torch.stack(
            [hc_mixes(x_hc[r], hc_fn[r], hc_scale[r], hc_base[r])[0] for r in range(N_RANKS)]
        )
        x_mixed = torch.stack([hc_pre(x_hc[r], pre_mixes[r]) for r in range(N_RANKS)])

    def init_correction_bias():
        bias = torch.full((N_RANKS, N_EXPERTS), -5.0)
        for r in range(N_RANKS):
            for loc in range(TOPK // N_RANKS):
                bias[r, r * N_LOCAL_EXPERTS + loc] = 5.0
        return bias

    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2}
    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2}
    w_gu_i8, w_gu_s = gen_routed_weight(
        (N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D), ROUTED_DEQUANT_STD["w1"]
    )
    w2_i8, w2_s = gen_routed_weight(
        (N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"]
    )
    shared_gu_i8, shared_gu_s = gen_shared(
        (N_RANKS, 2 * LOCAL_MOE_INTER, D), SHARED_DEQUANT_STD["w1"], 0.50
    )
    shared_w2_i8, shared_w2_s = gen_shared(
        (N_RANKS, D, LOCAL_MOE_INTER), SHARED_DEQUANT_STD["w2"], 0.33
    )

    return [
        TensorSpec("x_mixed", [N_RANKS, T, D], torch.bfloat16, init_value=lambda: x_mixed),
        TensorSpec("norm_weight", [N_RANKS, D], torch.bfloat16,
                   init_value=lambda: torch.ones(N_RANKS, D, dtype=torch.bfloat16)),
        TensorSpec("gate_weight", [N_RANKS, N_EXPERTS, D], torch.bfloat16,
                   init_value=lambda: torch.randn(N_RANKS, N_EXPERTS, D, generator=gen) / (D ** 0.5)),
        # Zero the router's top-k ambiguity: four experts per rank carry a +10
        # score gap, so kernel and golden must select the same eight. Otherwise a
        # legal near-tie swap changes which experts contribute and masks the
        # composition under test behind router noise.
        TensorSpec("correction_bias", [N_RANKS, N_EXPERTS], torch.float32,
                   init_value=init_correction_bias),
        TensorSpec("w_gate_up", [N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], torch.int8,
                   init_value=lambda: w_gu_i8),
        TensorSpec("w_gate_up_scale", [N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER], torch.float32,
                   init_value=lambda: w_gu_s),
        TensorSpec("w_down", [N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8,
                   init_value=lambda: w2_i8),
        TensorSpec("w_down_scale", [N_RANKS, N_LOCAL_EXPERTS, D], torch.float32,
                   init_value=lambda: w2_s),
        TensorSpec("shared_w_gate_up", [N_RANKS, 2 * LOCAL_MOE_INTER, D], torch.int8,
                   init_value=lambda: shared_gu_i8),
        TensorSpec("shared_w_gate_up_scale", [N_RANKS, 2 * LOCAL_MOE_INTER], torch.float32,
                   init_value=lambda: shared_gu_s),
        TensorSpec("shared_w_down", [N_RANKS, D, LOCAL_MOE_INTER], torch.int8,
                   init_value=lambda: shared_w2_i8),
        TensorSpec("shared_w_down_scale", [N_RANKS, D], torch.float32,
                   init_value=lambda: shared_w2_s),
        TensorSpec("y", [N_RANKS, T, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("num_ranks", torch.int32, num_ranks),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=2, help="tensor-parallel size (config.py reads argv)")
    parser.add_argument("--ep", type=int, default=2, help="expert-parallel size / rank count")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"
    num_ranks = len(device_ids)

    result = run(
        fn=l3_decode_layer,
        specs=build_tensor_specs(args.num_tokens, num_ranks=num_ranks),
        golden_fn=golden_decode_layer,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # The block entry starts at the router, so its error floor is the
            # gate's INT8 boundary: the kernel quantizes the FP32 ``xg`` while the
            # golden reproduces it in torch FP32, and real hardware flips a small
            # fraction of elements by +/-1 LSB at rounding boundaries. That is
            # tolerated by gate.py's own test but amplified into ``y`` through the
            # experts, so the block bar is 1e-2 / 5% (the sibling EP MoE's bar)
            # while the branch-level entries keep 3e-3 / 1%: structural faults
            # (route, slot, barrier) still land far outside either.
            "y": ratio_reldiff(diff_thd=1e-2, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "golden_decode_layer",
    "l2_decode_layer",
    "l3_decode_layer",
    "moe_layer_add",
]
