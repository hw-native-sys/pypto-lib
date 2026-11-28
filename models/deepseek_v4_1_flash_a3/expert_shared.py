# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 shared W8A8 SwiGLU expert."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import argparse
import json

import pypto.language as pl
import torch
import torch.nn.functional as F

from .activation_quant import quantize_projection_activation
from .config import FLASH as M, SHARED_NZ_PANEL
from .dsl_types import SLAB_T_DYN
from .gemm import EXPERT_K_TILE, PREFILL_EXPERT_LARGE_M_TILE, gemm_int8_tiled_nz


# model config
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# tiling
SHARED_ROWS = pl.dynamic("SHARED_ROWS")


@pl.jit.inline(auto_scope=False)
def _shared_swiglu_nz(
    source: pl.Tensor[[SLAB_T_DYN, D], pl.INT8],
    activation_scale: pl.Tensor[[1, SLAB_T_DYN], pl.FP32],
    paired: pl.Tensor[[D, 2 * MOE_INTER], pl.INT8, pl.NZ],
    s1: pl.Tensor[[1, MOE_INTER], pl.FP32],
    s3: pl.Tensor[[1, MOE_INTER], pl.FP32],
    output: pl.Tensor[[SHARED_ROWS, MOE_INTER], pl.BF16],
    count: pl.Scalar[pl.INT32], row_tile: pl.constexpr,
):
    """Share each input tile across paired gate and up panels."""
    panels = MOE_INTER // SHARED_NZ_PANEL
    row_blocks = (count + row_tile - 1) // row_tile
    workers = pl.min(pl.system.available_cluster_count(), panels * row_blocks)
    for worker in pl.spmd(workers, name_hint="shared_swiglu_nz", optimizations=[pl.cross_core_slot(slot_num=1)]):
        for panel in pl.range(0, panels):
            first_row = (worker + workers - (panel * row_blocks) % workers) % workers
            for row_index in pl.range(first_row, row_blocks, workers):
                row = row_index * row_tile
                valid_rows = pl.min(row_tile, count - row)
                col = panel * SHARED_NZ_PANEL
                acc_seed = pl.tile.create([row_tile, 2 * SHARED_NZ_PANEL], dtype=pl.INT32, target_memory=pl.Mem.Acc, compact=True)
                acc = pl.set_validshape(acc_seed, valid_rows, 2 * SHARED_NZ_PANEL)
                for k in pl.pipeline(0, D, EXPERT_K_TILE, stage=2):
                    x = pl.load(source, [row, k], [row_tile, EXPERT_K_TILE], valid_shape=[valid_rows, EXPERT_K_TILE])
                    matrix = pl.load(paired, [k, panel * (2 * SHARED_NZ_PANEL)], [EXPERT_K_TILE, 2 * SHARED_NZ_PANEL], target_memory=pl.Mem.Mat)
                    acc = pl.matmul_acc(acc, x, matrix, init_cond=(k == 0))
                value = pl.cast(acc, pl.FP32)
                gate_product = pl.tile.slice(value, [row_tile, SHARED_NZ_PANEL], [0, 0], valid_shape=[valid_rows, SHARED_NZ_PANEL])
                up_product = pl.tile.slice(value, [row_tile, SHARED_NZ_PANEL], [0, SHARED_NZ_PANEL], valid_shape=[valid_rows, SHARED_NZ_PANEL])
                scale_row = pl.load(activation_scale, [0, row], [1, row_tile], valid_shape=[1, valid_rows])
                scale = pl.reshape(scale_row, [row_tile, 1])
                g_scale = pl.load(s1, [0, col], [1, SHARED_NZ_PANEL])
                u_scale = pl.load(s3, [0, col], [1, SHARED_NZ_PANEL])
                gate_fp = pl.col_expand_mul(pl.row_expand_mul(gate_product, scale), g_scale)
                up_fp = pl.col_expand_mul(pl.row_expand_mul(up_product, scale), u_scale)
                gate = pl.minimum(pl.cast(pl.cast(gate_fp, pl.BF16, mode="rint"), pl.FP32), SWIGLU_LIMIT)
                up = pl.cast(pl.cast(up_fp, pl.BF16, mode="rint"), pl.FP32)
                up = pl.minimum(pl.maximum(up, -SWIGLU_LIMIT), SWIGLU_LIMIT)
                sigma = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
                value = pl.mul(pl.mul(gate, sigma), up)
                result = pl.cast(value, pl.BF16, mode="rint")
                pl.store(pl.set_validshape(result, valid_rows, SHARED_NZ_PANEL), [row, col], output)
    return output



@pl.jit.inline(auto_scope=False)
def _run_shared_nz(
    x_local: pl.Tensor[[SLAB_T_DYN, D], pl.INT8],
    source_scales: pl.Tensor[[1, SLAB_T_DYN], pl.FP32],
    gate_up: pl.Tensor[[D, 2 * MOE_INTER], pl.INT8, pl.NZ],
    gate_scale: pl.Tensor[[1, MOE_INTER], pl.FP32],
    up_scale: pl.Tensor[[1, MOE_INTER], pl.FP32],
    down: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    down_scale: pl.Tensor[[1, D], pl.FP32],
    output: pl.Tensor[[SHARED_ROWS, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32], prefill: pl.constexpr,
    row_tile: pl.constexpr, down_tile: pl.constexpr,
):
    padded_rows = pl.max(16, (num_tokens + 15) // 16 * 16)
    hidden = pl.create_tensor([padded_rows, MOE_INTER], dtype=pl.BF16)
    _shared_swiglu_nz(x_local, source_scales, gate_up, gate_scale, up_scale, hidden, num_tokens, row_tile)
    payload, scales = quantize_projection_activation(hidden, num_tokens, MOE_INTER, True, prefill)
    return gemm_int8_tiled_nz(payload, scales, down, down_scale, output, num_tokens,
                             MOE_INTER, D, "shared_down_nz", down_tile, 256, row_tile)


@pl.jit.inline(auto_scope=False)
def expert_shared_nz(
    x_local: pl.Tensor, x_local_scale: pl.Tensor,
    gate_up: pl.Tensor, gate_scale: pl.Tensor, up_scale: pl.Tensor,
    down: pl.Tensor, down_scale: pl.Tensor, output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], prefill: pl.constexpr,
):
    """Run the shared expert against the two resident Cube weight banks."""
    if prefill and num_tokens > PREFILL_EXPERT_LARGE_M_TILE:
        output = _run_shared_nz(x_local, x_local_scale, gate_up, gate_scale, up_scale,
                                down, down_scale, output, num_tokens, prefill, 64, 128)
    elif prefill and num_tokens > 16:
        output = _run_shared_nz(x_local, x_local_scale, gate_up, gate_scale, up_scale,
                                down, down_scale, output, num_tokens, prefill, 32, 256)
    else:
        output = _run_shared_nz(x_local, x_local_scale, gate_up, gate_scale, up_scale,
                                down, down_scale, output, num_tokens, prefill, 16, 256)
    return output


def golden_expert_shared(payload, activation_scale, w1, scale1, w3, scale3, w2, scale2):
    """Evaluate the checkpoint W8A8 shared expert with BF16 activation boundaries."""
    from .gemm import golden_dynamic_int8

    source = payload.long()
    factor = activation_scale.float().reshape(-1, 1)
    gate = (((source @ w1.long()).float()) * factor * scale1.float()).bfloat16().float()
    up = (((source @ w3.long()).float()) * factor * scale3.float()).bfloat16().float()
    hidden = F.silu(gate.clamp(max=SWIGLU_LIMIT)) * up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    return golden_dynamic_int8(hidden.bfloat16(), w2, scale2)


VALIDATION_ROWS = 80


@pl.jit
def l2_expert_shared(
    source: pl.Tensor[[VALIDATION_ROWS, D], pl.INT8],
    scale: pl.Tensor[[1, VALIDATION_ROWS], pl.FP32],
    gate_up: pl.Tensor[[D, 2 * MOE_INTER], pl.INT8, pl.NZ],
    scale1: pl.Tensor[[1, MOE_INTER], pl.FP32],
    scale3: pl.Tensor[[1, MOE_INTER], pl.FP32],
    down: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    scale2: pl.Tensor[[1, D], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.InOut[pl.Tensor[[VALIDATION_ROWS, D], pl.BF16]],
):
    count = pl.read(counts, [0])
    if count > 1:
        expert_shared_nz(source, scale, gate_up, scale1, scale3, down, scale2, output, count, True)
    else:
        expert_shared_nz(source, scale, gate_up, scale1, scale3, down, scale2, output, count, False)
    return output


def golden_l2_expert_shared(tensors, weights):
    """Project live rows and preserve the output tail."""
    count = int(tensors["counts"][0])
    w1, w3, w2 = weights
    projected = golden_expert_shared(
        tensors["source"][:count], tensors["scale"][:, :count],
        w1, tensors["scale1"], w3, tensors["scale3"], w2, tensors["scale2"],
    )
    tensors["output"][:count].copy_(projected)


if __name__ == "__main__":
    from functools import partial

    from golden import TensorSpec, ratio_allclose, run

    from .gemm import golden_activation_int8
    from .weights import Checkpoint, ResidentWeights, pack_projection_int8_nz, pack_shared_gate_up

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, choices=range(M.num_hidden_layers), default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(41)
    hidden = torch.randn(VALIDATION_ROWS, D, generator=generator).bfloat16()
    hidden[1].zero_()
    source = golden_activation_int8(hidden)
    factor = (hidden.float().abs().amax(-1) / 127).clamp_min(1e-30).reshape(1, VALIDATION_ROWS)
    checkpoint = ResidentWeights(Checkpoint(args.checkpoint)) if args.checkpoint else None
    matrices = []
    scales = []
    for number, width, columns in ((1, D, MOE_INTER), (3, D, MOE_INTER), (2, MOE_INTER, D)):
        if checkpoint is not None:
            stem = f"layers.{args.layer}.ffn.shared_experts.w{number}"
            weight, scale = checkpoint.projection(stem)
        else:
            weight = torch.randint(-127, 128, (width, columns), dtype=torch.int8, generator=generator)
            scale = torch.full((1, columns), 0.02 / 127)
        matrices.append(weight)
        scales.append(scale)
    w1, w3, w2 = matrices
    scale1, scale3, scale2 = scales
    gate_up = pack_shared_gate_up(w1.t(), w3.t()).view(D, 2 * MOE_INTER)
    down = pack_projection_int8_nz(w2).view(MOE_INTER, D)
    values = {
        "source": source, "scale": factor,
        "gate_up": gate_up, "scale1": scale1, "scale3": scale3,
        "down": down, "scale2": scale2,
    }
    base_specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
    reports = []
    passed = True
    for count in (1, 17, 65):
        counts = torch.tensor([count], dtype=torch.int32)
        specs = [
            *base_specs,
            TensorSpec("counts", [1], torch.int32, init_value=counts),
            TensorSpec("output", [VALIDATION_ROWS, D], torch.bfloat16, init_value=-1.0),
        ]
        result = run(
            fn=l2_expert_shared,
            specs=specs,
            golden_fn=partial(golden_l2_expert_shared, weights=matrices),
            compare_fn={"output": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.005)},
            config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
            compile_only=args.compile_only,
        )
        passed &= result.passed
        reports.append({"count": count, "passed": result.passed, "build": str(result.work_dir)})
    if args.report:
        args.report.write_text(json.dumps({"layer": args.layer, "cases": reports}, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)
