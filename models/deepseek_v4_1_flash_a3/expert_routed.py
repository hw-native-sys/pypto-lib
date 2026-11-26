# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 routed expert compute with distributed scatter."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"


import argparse
import json

import pypto.language as pl
import pypto.language.distributed as pld
import torch
import torch.nn.functional as F

from .config import FLASH as M
from .config import N_LOCAL_EXPERTS
from .dsl_types import ROUTE_T_DYN
from .expert_prefill import PREFILL_FUSED_MAX_ROWS, prefill_expert_compute
from .gemm import EXPERT_K_TILE, EXPERT_N_TILE, expert_down, expert_swiglu, golden_dynamic_int8


# Dynamic shape variables.
EXPERT_ROWS_DYN = pl.dynamic("A3_EXPERT_ROWS_DYN")

# model config
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# tiling
PREFILL_ROUTED_DOWN_ROW_TILE = 128
PREFILL_ROUTED_GATE_ROW_TILE = 128
PREFILL_ROUTED_GATE_N_TILE = 64


@pl.jit.inline(auto_scope=False)
def expert_routed_compute(
    source: pl.Tensor[[EXPERT_ROWS_DYN, D], pl.INT8],
    source_scale: pl.Tensor[[EXPERT_ROWS_DYN, 1], pl.FP32],
    route_weights: pl.Tensor[[1, EXPERT_ROWS_DYN], pl.FP32],
    w1: pl.Tensor,
    scale1: pl.Tensor[[1, MOE_INTER], pl.FP32],
    w3: pl.Tensor,
    scale3: pl.Tensor[[1, MOE_INTER], pl.FP32],
    w2: pl.Tensor,
    scale2: pl.Tensor[[1, D], pl.FP32],
    output: pl.Tensor[[EXPERT_ROWS_DYN, D], pl.BF16],
    count: pl.Scalar[pl.INT32],
    prefill: pl.constexpr,
):
    """Apply a routed W8A8 expert with route weighting before hidden quantization."""
    if prefill and count <= PREFILL_FUSED_MAX_ROWS:
        prefill_expert_compute(
            source, source_scale, route_weights, w1, scale1, w3, scale3, w2, scale2,
            output, count, True,
        )
    else:
        capacity = pl.max(16, (count + 15) // 16 * 16)
        hidden = pl.create_tensor([capacity, MOE_INTER], dtype=pl.BF16)
        expert_swiglu(
            source, source_scale, w1, scale1, w3, scale3, route_weights, hidden, count,
            D, MOE_INTER, EXPERT_N_TILE, EXPERT_K_TILE, SWIGLU_LIMIT, True, "routed_swiglu", True, prefill,
            PREFILL_ROUTED_GATE_ROW_TILE, PREFILL_ROUTED_GATE_N_TILE,
        )
        expert_down(
            hidden, w2, scale2, output, count, MOE_INTER, D, "routed_down", True, prefill, PREFILL_ROUTED_DOWN_ROW_TILE,
        )
    return output


@pl.jit.inline(auto_scope=False)
def expert_routed_scatter(
    recv_x: pl.Tensor,
    recv_mx_scale: pl.Tensor,
    recv_weights: pl.Tensor,
    recv_expert_count: pl.Tensor,
    routed_w1: pl.Tensor,
    routed_w1_scale: pl.Tensor,
    routed_w3: pl.Tensor,
    routed_w3_scale: pl.Tensor,
    routed_w2: pl.Tensor,
    routed_w2_scale: pl.Tensor,
    recv_route: pl.Tensor,
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    input_ready: pl.Scalar[pl.TASK_ID],
    prefill: pl.constexpr,
    routes_per_source: pl.constexpr,
    recv_capacity: pl.constexpr,
):
    """Compute compact INT8 expert rows and return weighted outputs to their owners."""
    scales = pl.reshape(recv_mx_scale, [recv_capacity, 1])
    completion = pl.array.create(N_LOCAL_EXPERTS, pl.TASK_ID)
    for expert in pl.parallel(N_LOCAL_EXPERTS):
        completion[expert] = pl.system.task_dummy(deps=[input_ready])
        count = pl.read(recv_expert_count, [expert, 0])
        if count > 0:
            padded_count = (count + 15) // 16 * 16
            offset = pl.cast(pl.read(recv_expert_count, [expert, 1]), pl.INDEX)
            source = pl.slice(recv_x, [padded_count, D], [offset, 0])
            source_scale = pl.slice(scales, [padded_count, 1], [offset, 0])
            scale1 = pl.slice(routed_w1_scale, [1, MOE_INTER], [expert, 0])
            scale3 = pl.slice(routed_w3_scale, [1, MOE_INTER], [expert, 0])
            output = pl.create_tensor([padded_count, D], dtype=pl.BF16)
            scale2 = pl.slice(routed_w2_scale, [1, D], [expert, 0])
            route_weights = pl.slice(recv_weights, [1, padded_count], [0, offset])
            expert_routed_compute(
                source, source_scale, route_weights,
                routed_w1[expert], scale1, routed_w3[expert], scale3, routed_w2[expert], scale2,
                output, count, prefill,
            )
            workers = pl.min(pl.min(32, (count + 3) // 4), pl.system.available_aiv_count())
            with pl.spmd(workers, name_hint="routed_return") as returned:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, count, workers):
                    address = pl.cast(pl.read(recv_route, [0, offset + row]), pl.INDEX)
                    route = address % routes_per_source
                    source_rank = address // routes_per_source
                    pld.tensor.put(
                        dst=routed_output, peer=source_rank, src=output,
                        dst_offsets=[route, 0], src_offsets=[row, 0], shape=[1, D],
                    )
            completion[expert] = returned
    return pl.system.task_dummy(deps=[completion])


def golden_expert_routed(source, source_scale, route_weights, w1, scale1, w3, scale3, w2, scale2):
    """Evaluate weighted routed SwiGLU with exact integer dot products."""
    source = source.long()
    factor = source_scale.float().reshape(-1, 1)
    gate = ((source @ w1.t().long()).float() * factor * scale1).bfloat16().float()
    up = ((source @ w3.t().long()).float() * factor * scale3).bfloat16().float()
    hidden = F.silu(gate.clamp(max=SWIGLU_LIMIT)) * up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    hidden = (hidden * route_weights.float().reshape(-1, 1)).bfloat16()
    return golden_dynamic_int8(hidden, w2.t(), scale2)


def golden_l2_expert_routed(tensors):
    """Compute valid routed expert rows and preserve inactive rows."""
    count = int(tensors["counts"][0])
    projected = golden_expert_routed(
        tensors["source"][:count], tensors["source_scale"][:count], tensors["route_weights"][:, :count],
        tensors["w1"], tensors["scale1"], tensors["w3"], tensors["scale3"], tensors["w2"], tensors["scale2"],
    )
    tensors["output"][:count].copy_(projected)


VALIDATION_ROWS = PREFILL_FUSED_MAX_ROWS + 16


@pl.jit
def l2_expert_routed(
    source: pl.Tensor[[VALIDATION_ROWS, D], pl.INT8],
    source_scale: pl.Tensor[[VALIDATION_ROWS, 1], pl.FP32],
    route_weights: pl.Tensor[[1, VALIDATION_ROWS], pl.FP32],
    w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    scale1: pl.Tensor[[1, MOE_INTER], pl.FP32],
    w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    scale3: pl.Tensor[[1, MOE_INTER], pl.FP32],
    w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    scale2: pl.Tensor[[1, D], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    prefill: pl.Scalar[pl.INT32],
    output: pl.InOut[pl.Tensor[[VALIDATION_ROWS, D], pl.BF16]],
):
    count = pl.read(counts, [0])
    if prefill != 0:
        expert_routed_compute(
            source, source_scale, route_weights, w1, scale1, w3, scale3, w2, scale2,
            output, count, True,
        )
    else:
        expert_routed_compute(
            source, source_scale, route_weights, w1, scale1, w3, scale3, w2, scale2,
            output, count, False,
        )
    return output


if __name__ == '__main__':
    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from .gemm import golden_activation_int8

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--platform', choices=['a2a3'], default='a2a3')
    parser.add_argument("--sim", action="store_true")
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--layer', type=int, choices=range(M.num_hidden_layers), default=0)
    parser.add_argument('--expert', type=int, choices=range(M.n_routed_experts))
    parser.add_argument('--num-tokens', type=int, choices=range(1, VALIDATION_ROWS + 1))
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(41)
    hidden = torch.randn(VALIDATION_ROWS, D, generator=gen).bfloat16()
    hidden[0].zero_()
    source = golden_activation_int8(hidden)
    factor = (hidden.float().abs().amax(-1) / 127).clamp_min(1e-30).reshape(VALIDATION_ROWS, 1)
    reports = []
    if args.checkpoint:
        from .weights import Checkpoint, ResidentWeights

        weights = ResidentWeights(Checkpoint(args.checkpoint))
    experts = (0, M.n_routed_experts - 1) if args.checkpoint else (0,)
    if args.expert is not None:
        experts = (args.expert,)
    for expert in experts:
        matrices = {}
        for number, width, columns in ((1, D, MOE_INTER), (3, D, MOE_INTER), (2, MOE_INTER, D)):
            if args.checkpoint:
                stem = f'layers.{args.layer}.ffn.experts.{expert}.w{number}'
                weight, scale = weights.projection(stem, output_major=True)
            else:
                weight = torch.randint(-127, 128, (columns, width), dtype=torch.int8, generator=gen)
                scale = torch.full((1, columns), 0.02 / 127)
            matrices[f'w{number}'], matrices[f'scale{number}'] = weight, scale
        cases = [("decode", args.num_tokens or 17), ("prefill", args.num_tokens or 17)]
        if args.num_tokens is None:
            cases.append(("prefill", PREFILL_FUSED_MAX_ROWS + 1))
        for phase, count in cases:
            route_weights = torch.zeros(1, VALIDATION_ROWS)
            route_weights[0, :count] = torch.linspace(0, 1.5, count)
            if count == 1:
                route_weights[0, 0] = 1.0
                source[0].fill_(1)
                factor[0] = 0.01
            values = dict(source=source, source_scale=factor, route_weights=route_weights, **matrices)
            values["counts"] = torch.tensor([count], dtype=torch.int32)
            inputs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
            specs = [
                *inputs,
                ScalarSpec("prefill", torch.int32, int(phase == "prefill")),
                TensorSpec("output", [VALIDATION_ROWS, D], torch.bfloat16, init_value=-1.0),
            ]
            print(f"A3_ROUTED_EXPERT layer={args.layer} expert={expert} phase={phase}", flush=True)
            result = run(
                fn=l2_expert_routed,
                specs=specs,
                golden_fn=golden_l2_expert_routed,
                compare_fn={"output": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.005)},
                config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
                compile_only=args.compile_only,
            )
            reports.append(dict(layer=args.layer, expert=expert, phase=phase, count=count, passed=result.passed,
                                compile_only=args.compile_only, build=str(result.work_dir)))
            if args.report:
                args.report.write_text(json.dumps(reports, indent=2) + "\n")
            if not result.passed:
                raise SystemExit(result.error or 1)
