# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Biased BF16 vision MLP and the clamped SwiGLU used by its spatial merger."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pypto.language as pl
import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import T_DYN
from models.glm5_3_flash.elementwise import accurate_divide, accurate_exp
from models.glm5_3_flash.linear import bf16_projection, golden_bf16_linear
from models.glm5_3_flash.vision.config import VISION_DEPTH, VISION_HIDDEN, VISION_INTER, VISION_SWIGLU_LIMIT

VISION_MLP_WIDTH = pl.dynamic("GLM53_VISION_MLP_WIDTH")
ACT_M_TILE = 8
ACT_N_TILE = 128


def golden_vision_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    b_gate: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor,
    w_down: torch.Tensor,
    b_down: torch.Tensor,
) -> torch.Tensor:
    gate = (golden_bf16_linear(x, w_gate) + b_gate.float()).bfloat16().clamp(max=VISION_SWIGLU_LIMIT)
    up = (golden_bf16_linear(x, w_up) + b_up.float()).bfloat16()
    up = up.clamp(min=-VISION_SWIGLU_LIMIT, max=VISION_SWIGLU_LIMIT)
    activated = (F.silu(gate) * up).bfloat16()
    return (golden_bf16_linear(activated, w_down) + b_down.float()).bfloat16()


@pl.jit.inline
def vision_swiglu(
    gate: pl.Tensor[[T_DYN, VISION_MLP_WIDTH], pl.BF16],
    up: pl.Tensor[[T_DYN, VISION_MLP_WIDTH], pl.BF16],
    output: pl.Tensor[[T_DYN, VISION_MLP_WIDTH], pl.BF16],
):
    rows = pl.tensor.dim(gate, 0)
    width = pl.tensor.dim(gate, 1)
    for block in pl.spmd(((rows + ACT_M_TILE - 1) // ACT_M_TILE) * (width // ACT_N_TILE), name_hint="vision_swiglu"):
        row0 = block // (width // ACT_N_TILE) * ACT_M_TILE
        col0 = block % (width // ACT_N_TILE) * ACT_N_TILE
        valid_rows = pl.min(ACT_M_TILE, rows - row0)
        gate_values = pl.slice(gate, [ACT_M_TILE, ACT_N_TILE], [row0, col0], valid_shape=[valid_rows, ACT_N_TILE])
        up_values = pl.slice(up, [ACT_M_TILE, ACT_N_TILE], [row0, col0], valid_shape=[valid_rows, ACT_N_TILE])
        gate_fp32 = pl.minimum(pl.cast(gate_values, pl.FP32), VISION_SWIGLU_LIMIT)
        up_fp32 = pl.minimum(pl.cast(up_values, pl.FP32), VISION_SWIGLU_LIMIT)
        up_fp32 = pl.maximum(up_fp32, -VISION_SWIGLU_LIMIT)
        gate_buffer = pl.create_tensor([ACT_M_TILE, ACT_N_TILE], dtype=pl.FP32)
        exponent = pl.create_tensor([ACT_M_TILE, ACT_N_TILE], dtype=pl.FP32)
        denominator = pl.create_tensor([ACT_M_TILE, ACT_N_TILE], dtype=pl.FP32)
        gate_buffer[:, :] = gate_fp32
        exponent[:, :] = pl.neg(gate_fp32)
        exponential = accurate_exp(exponent)
        denominator[:, :] = pl.add(exponential, 1.0)
        silu = accurate_divide(gate_buffer, denominator)
        silu_bf16 = pl.cast(silu, pl.BF16, mode="rint")
        product = pl.mul(pl.cast(silu_bf16, pl.FP32), up_fp32)
        result = pl.cast(product, pl.BF16, mode="rint")
        output[row0:row0 + ACT_M_TILE, col0:col0 + ACT_N_TILE] = pl.set_validshape(result, valid_rows, ACT_N_TILE)
    return output


@pl.jit.inline
def vision_mlp(
    x: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_gate: pl.Tensor[[VISION_INTER], pl.BF16],
    w_up: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_up: pl.Tensor[[VISION_INTER], pl.BF16],
    w_down: pl.Tensor[[VISION_HIDDEN, VISION_INTER], pl.BF16],
    b_down: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    gate = pl.create_tensor([rows, VISION_INTER], dtype=pl.BF16)
    up = pl.create_tensor([rows, VISION_INTER], dtype=pl.BF16)
    bf16_projection(x, w_gate, b_gate, gate, VISION_HIDDEN, VISION_INTER, True)
    bf16_projection(x, w_up, b_up, up, VISION_HIDDEN, VISION_INTER, True)
    activated = pl.create_tensor([rows, VISION_INTER], dtype=pl.BF16)
    vision_swiglu(gate, up, activated)
    bf16_projection(activated, w_down, b_down, output, VISION_INTER, VISION_HIDDEN, True)
    return output


@pl.jit
def vision_mlp_test(
    x: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_gate: pl.Tensor[[VISION_INTER], pl.BF16],
    w_up: pl.Tensor[[VISION_INTER, VISION_HIDDEN], pl.BF16],
    b_up: pl.Tensor[[VISION_INTER], pl.BF16],
    w_down: pl.Tensor[[VISION_HIDDEN, VISION_INTER], pl.BF16],
    b_down: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    vision_mlp(x, w_gate, b_gate, w_up, b_up, w_down, b_down, output)
    return output


def main():
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run
    from models.glm5_3_flash.utils import component_hidden, component_options
    from models.glm5_3_flash.weights import Checkpoint

    options = component_options(__doc__, layers=VISION_DEPTH)
    shapes = {
        "x": [options.tokens, VISION_HIDDEN],
        "w_gate": [VISION_INTER, VISION_HIDDEN], "b_gate": [VISION_INTER],
        "w_up": [VISION_INTER, VISION_HIDDEN], "b_up": [VISION_INTER],
        "w_down": [VISION_HIDDEN, VISION_INTER], "b_down": [VISION_HIDDEN],
    }
    names = {"w_gate": "gate_proj.weight", "b_gate": "gate_proj.bias",
             "w_up": "up_proj.weight", "b_up": "up_proj.bias",
             "w_down": "down_proj.weight", "b_down": "down_proj.bias"}

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5351)
        checkpoint = Checkpoint(options.checkpoint) if options.checkpoint is not None else None
        values = {"x": component_hidden(options, VISION_HIDDEN, generator)}
        for name, suffix in names.items():
            shape = shapes[name]
            if checkpoint is not None:
                values[name] = checkpoint.vision_tensor(f"blocks.{options.layer}.mlp.{suffix}", shape)
            else:
                scale = shape[-1] ** -0.5 if len(shape) == 2 else 0.1
                values[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()
        return values

    specs = [TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
             for name, shape in shapes.items()]
    specs.append(TensorSpec("output", [options.tokens, VISION_HIDDEN], torch.bfloat16))

    def golden(values):
        values["output"].copy_(golden_vision_mlp(*(values[name] for name in shapes)))

    result = run(
        fn=vision_mlp_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"output": ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
