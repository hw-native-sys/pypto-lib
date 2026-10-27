# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Spatial downsampling, LayerNorm-GELU and clamped SwiGLU into the text width."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pypto.language as pl
import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import D, T_DYN
from models.glm5_3_flash.elementwise import accurate_reciprocal, accurate_rsqrt
from models.glm5_3_flash.linear import bf16_projection, golden_bf16_linear
from models.glm5_3_flash.vision.config import VISION_HIDDEN, VISION_MERGE, VISION_PROJ_INTER, VISION_SWIGLU_LIMIT
from models.glm5_3_flash.vision.mlp import vision_swiglu

MERGE_N_TILE = 128
MERGE_M_TILE = 8
LAYER_NORM_EPS = 1e-5


def golden_vision_merger(
    x: torch.Tensor,
    downsample_weight: torch.Tensor,
    downsample_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    post_norm_bias: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    packed = x.transpose(1, 2).reshape(-1, VISION_HIDDEN * VISION_MERGE ** 2)
    downsampled = (golden_bf16_linear(packed, downsample_weight) + downsample_bias.float()).bfloat16()
    projected = golden_bf16_linear(downsampled, proj_weight).bfloat16()
    normalized = F.layer_norm(projected.float(), (D,), post_norm_weight.float(), post_norm_bias.float(), LAYER_NORM_EPS)
    activated = F.gelu(normalized.bfloat16()).bfloat16()
    gate = golden_bf16_linear(activated, w_gate).bfloat16().clamp(max=VISION_SWIGLU_LIMIT)
    up = golden_bf16_linear(activated, w_up).bfloat16().clamp(-VISION_SWIGLU_LIMIT, VISION_SWIGLU_LIMIT)
    intermediate = (F.silu(gate) * up).bfloat16()
    return golden_bf16_linear(intermediate, w_down).bfloat16()


@pl.jit.inline
def vision_merger_activation(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    norm_bias: pl.Tensor[[D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    for block in pl.spmd((rows + MERGE_M_TILE - 1) // MERGE_M_TILE, name_hint="vision_merger_layernorm_gelu"):
        row0 = block * MERGE_M_TILE
        valid_rows = pl.min(MERGE_M_TILE, rows - row0)
        total = pl.full([1, MERGE_M_TILE], dtype=pl.FP32, value=0.0)
        for col0 in pl.range(0, D, MERGE_N_TILE):
            source = pl.slice(x, [MERGE_M_TILE, MERGE_N_TILE], [row0, col0], valid_shape=[valid_rows, MERGE_N_TILE])
            value = pl.cast(source, pl.FP32)
            partial_sum = pl.reshape(pl.row_sum(value), [1, MERGE_M_TILE])
            total = pl.add(total, partial_sum)
        mean_row = pl.mul(total, 1.0 / D)
        mean = pl.reshape(mean_row, [MERGE_M_TILE, 1])
        squared_total = pl.full([1, MERGE_M_TILE], dtype=pl.FP32, value=0.0)
        for col0 in pl.range(0, D, MERGE_N_TILE):
            source = pl.slice(x, [MERGE_M_TILE, MERGE_N_TILE], [row0, col0], valid_shape=[valid_rows, MERGE_N_TILE])
            centered = pl.row_expand_sub(pl.cast(source, pl.FP32), mean)
            partial_square_sum = pl.reshape(pl.row_sum(pl.mul(centered, centered)), [1, MERGE_M_TILE])
            squared_total = pl.add(squared_total, partial_square_sum)
        variance = pl.add(pl.mul(squared_total, 1.0 / D), LAYER_NORM_EPS)
        variance_buffer = pl.create_tensor([1, MERGE_M_TILE], dtype=pl.FP32)
        variance_buffer[:, :] = variance
        inverse_row = accurate_rsqrt(variance_buffer)
        inverse = pl.reshape(inverse_row, [MERGE_M_TILE, 1])
        for col0 in pl.range(0, D, MERGE_N_TILE):
            source = pl.slice(x, [MERGE_M_TILE, MERGE_N_TILE], [row0, col0], valid_shape=[valid_rows, MERGE_N_TILE])
            centered = pl.row_expand_sub(pl.cast(source, pl.FP32), mean)
            unit = pl.row_expand_mul(centered, inverse)
            gamma = pl.reshape(pl.cast(norm_weight[col0:col0 + MERGE_N_TILE], pl.FP32), [1, MERGE_N_TILE])
            beta = pl.reshape(pl.cast(norm_bias[col0:col0 + MERGE_N_TILE], pl.FP32), [1, MERGE_N_TILE])
            affine = pl.col_expand_add(pl.col_expand_mul(unit, gamma), beta)
            normalized = pl.cast(affine, pl.BF16, mode="rint")
            normalized_fp32 = pl.cast(normalized, pl.FP32)
            z = pl.mul(normalized_fp32, 0.7071067811865476)
            gelu_denominator = pl.create_tensor([MERGE_M_TILE, MERGE_N_TILE], dtype=pl.FP32)
            gelu_denominator[:, :] = pl.add(pl.mul(pl.abs(z), 0.3275911), 1.0)
            t = accurate_reciprocal(gelu_denominator)
            polynomial = pl.add(pl.mul(t, 1.061405429), -1.453152027)
            polynomial = pl.add(pl.mul(t, polynomial), 1.421413741)
            polynomial = pl.add(pl.mul(t, polynomial), -0.284496736)
            polynomial = pl.add(pl.mul(t, polynomial), 0.254829592)
            gaussian = pl.exp(pl.neg(pl.mul(z, z)))
            erf_error = pl.mul(pl.mul(t, polynomial), gaussian)
            erf_absolute = pl.add(pl.neg(erf_error), 1.0)
            positive = pl.cast(pl.cmp(z, 0.0, cmp_type=5), pl.INT32)
            sign = pl.add(pl.mul(pl.cast(positive, pl.FP32), 2.0), -1.0)
            erf_value = pl.mul(sign, erf_absolute)
            gelu = pl.mul(pl.mul(normalized_fp32, 0.5), pl.add(erf_value, 1.0))
            result = pl.cast(gelu, pl.BF16, mode="rint")
            output[row0:row0 + MERGE_M_TILE, col0:col0 + MERGE_N_TILE] = pl.set_validshape(result, valid_rows, MERGE_N_TILE)
    return output


@pl.jit.inline
def vision_merger(
    x: pl.Tensor[[T_DYN, VISION_MERGE * VISION_MERGE, VISION_HIDDEN], pl.BF16],
    downsample_weight: pl.Tensor[[D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    downsample_bias: pl.Tensor[[D], pl.BF16],
    proj_weight: pl.Tensor[[D, D], pl.BF16],
    post_norm_weight: pl.Tensor[[D], pl.BF16],
    post_norm_bias: pl.Tensor[[D], pl.BF16],
    w_gate: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_up: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_down: pl.Tensor[[D, VISION_PROJ_INTER], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    rows = pl.tensor.dim(x, 0)
    packed_channels = pl.create_tensor([rows * VISION_HIDDEN, VISION_MERGE * VISION_MERGE], dtype=pl.BF16)
    for block in pl.spmd(rows * (VISION_HIDDEN // MERGE_N_TILE), name_hint="vision_merge_channel_order"):
        row = block // (VISION_HIDDEN // MERGE_N_TILE)
        col0 = block % (VISION_HIDDEN // MERGE_N_TILE) * MERGE_N_TILE
        spatial_rows = pl.full([16, MERGE_N_TILE], dtype=pl.BF16, value=0.0)
        for position in pl.unroll(VISION_MERGE * VISION_MERGE):
            spatial = x[row:row + 1, position:position + 1, col0:col0 + MERGE_N_TILE]
            spatial_row = pl.reshape(spatial, [1, MERGE_N_TILE])
            spatial_rows[position:position + 1, :] = spatial_row
        channel_major = pl.transpose(spatial_rows, axis1=0, axis2=1)
        valid_channels = pl.set_validshape(channel_major, MERGE_N_TILE, VISION_MERGE * VISION_MERGE)
        packed_channels[row * VISION_HIDDEN + col0:row * VISION_HIDDEN + col0 + MERGE_N_TILE, :] = valid_channels
    packed = pl.reshape(packed_channels, [rows, VISION_MERGE * VISION_MERGE * VISION_HIDDEN])
    downsampled = pl.create_tensor([rows, D], dtype=pl.BF16)
    bf16_projection(packed, downsample_weight, downsample_bias, downsampled, VISION_HIDDEN * VISION_MERGE * VISION_MERGE, D, True)
    projected = pl.create_tensor([rows, D], dtype=pl.BF16)
    bf16_projection(downsampled, proj_weight, post_norm_weight, projected, D, D, False)
    activated = pl.create_tensor([rows, D], dtype=pl.BF16)
    vision_merger_activation(projected, post_norm_weight, post_norm_bias, activated)
    gate = pl.create_tensor([rows, VISION_PROJ_INTER], dtype=pl.BF16)
    up = pl.create_tensor([rows, VISION_PROJ_INTER], dtype=pl.BF16)
    bf16_projection(activated, w_gate, post_norm_weight, gate, D, VISION_PROJ_INTER, False)
    bf16_projection(activated, w_up, post_norm_weight, up, D, VISION_PROJ_INTER, False)
    intermediate = pl.create_tensor([rows, VISION_PROJ_INTER], dtype=pl.BF16)
    vision_swiglu(gate, up, intermediate)
    bf16_projection(intermediate, w_down, post_norm_weight, output, VISION_PROJ_INTER, D, False)
    return output


@pl.jit
def vision_merger_test(
    x: pl.Tensor[[T_DYN, VISION_MERGE * VISION_MERGE, VISION_HIDDEN], pl.BF16],
    downsample_weight: pl.Tensor[[D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    downsample_bias: pl.Tensor[[D], pl.BF16],
    proj_weight: pl.Tensor[[D, D], pl.BF16],
    post_norm_weight: pl.Tensor[[D], pl.BF16],
    post_norm_bias: pl.Tensor[[D], pl.BF16],
    w_gate: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_up: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    w_down: pl.Tensor[[D, VISION_PROJ_INTER], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    vision_merger(x, downsample_weight, downsample_bias, proj_weight, post_norm_weight, post_norm_bias,
                  w_gate, w_up, w_down, output)
    return output


def main():
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run
    from models.glm5_3_flash.utils import component_hidden, component_options
    from models.glm5_3_flash.weights import Checkpoint

    options = component_options(__doc__, tokens=3)
    input_width = VISION_MERGE ** 2 * VISION_HIDDEN
    shapes = {
        "x": [options.tokens, VISION_MERGE ** 2, VISION_HIDDEN],
        "downsample_weight": [D, input_width], "downsample_bias": [D], "proj_weight": [D, D],
        "post_norm_weight": [D], "post_norm_bias": [D],
        "w_gate": [VISION_PROJ_INTER, D], "w_up": [VISION_PROJ_INTER, D], "w_down": [D, VISION_PROJ_INTER],
    }
    names = {"downsample_weight": "downsample.weight", "downsample_bias": "downsample.bias",
             "proj_weight": "merger.proj.weight", "post_norm_weight": "merger.post_projection_norm.weight",
             "post_norm_bias": "merger.post_projection_norm.bias", "w_gate": "merger.gate_proj.weight",
             "w_up": "merger.up_proj.weight", "w_down": "merger.down_proj.weight"}

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5353)
        checkpoint = Checkpoint(options.checkpoint) if options.checkpoint is not None else None
        values = {"x": component_hidden(options, input_width, generator).reshape(shapes["x"])}
        for name, suffix in names.items():
            shape = shapes[name]
            if checkpoint is not None:
                source_shape = [D, VISION_HIDDEN, VISION_MERGE, VISION_MERGE] if name == "downsample_weight" else shape
                values[name] = checkpoint.vision_tensor(suffix, source_shape).reshape(shape)
            elif name == "post_norm_weight":
                values[name] = torch.ones(shape, dtype=torch.bfloat16)
            else:
                scale = shape[-1] ** -0.5 if len(shape) == 2 else 0.1
                values[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()
        return values

    specs = [TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
             for name, shape in shapes.items()]
    specs.append(TensorSpec("output", [options.tokens, D], torch.bfloat16))

    def golden(values):
        values["output"].copy_(golden_vision_merger(*(values[name] for name in shapes)))

    result = run(
        fn=vision_merger_test,
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
