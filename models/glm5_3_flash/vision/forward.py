# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The complete 24-block BF16 vision tower with packed image and video-frame segments."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pypto.language as pl
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig
import torch

from models.glm5_3_flash.config import D, T_DYN, VISION_ROWS_DYN
from models.glm5_3_flash.golden import rms_norm
from models.glm5_3_flash.rmsnorm import rmsnorm
from models.glm5_3_flash.vision.attention import VISION_STARTS_DYN, VISION_POSITIONS_DYN
from models.glm5_3_flash.vision.attention import golden_vision_attention, vision_attention, vision_axial_rope_table, vision_rope_tables
from models.glm5_3_flash.vision.config import VISION_DEPTH, VISION_HEAD_DIM, VISION_HIDDEN, VISION_INTER
from models.glm5_3_flash.vision.config import VISION_MERGE, VISION_PATCH_IN, VISION_PROJ_INTER, VISION_RMS_NORM_EPS
from models.glm5_3_flash.vision.merger import golden_vision_merger, vision_merger
from models.glm5_3_flash.vision.mlp import golden_vision_mlp, vision_mlp
from models.glm5_3_flash.vision.patch_embed import golden_vision_patch_embed, vision_patch_embed
from models.glm5_3_flash.weights import Checkpoint, vision_weight_layouts

RESIDUAL_M_TILE = 8
RESIDUAL_N_TILE = 128


def vision_metadata(grid_thw):
    """Build merge-group patch positions and one attention segment per temporal patch."""
    grid = torch.as_tensor(grid_thw, device="cpu")
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
        raise ValueError("grid_thw must have shape [images_or_videos, 3]")
    if grid.is_floating_point() and not torch.equal(grid, grid.round()):
        raise ValueError("grid_thw must contain integers")
    grid = grid.long()
    if (grid <= 0).any() or (grid[:, 1:] % VISION_MERGE).any():
        raise ValueError("grid dimensions must be positive with spatial sizes divisible by the merge factor")
    positions, starts = [], [0]
    for frames, height, width in grid.tolist():
        h = torch.arange(height)[:, None].expand(height, width)
        w = torch.arange(width)[None, :].expand(height, width)
        shape = (height // VISION_MERGE, VISION_MERGE, width // VISION_MERGE, VISION_MERGE)
        h = h.reshape(shape).permute(0, 2, 1, 3).reshape(-1)
        w = w.reshape(shape).permute(0, 2, 1, 3).reshape(-1)
        frame_positions = torch.stack((h, w), dim=-1).int()
        for _ in range(frames):
            positions.append(frame_positions)
            starts.append(starts[-1] + height * width)
    return torch.cat(positions), torch.tensor(starts, dtype=torch.int32)


@pl.jit
def vision_stem(
    patches: pl.Tensor[[T_DYN, VISION_PATCH_IN], pl.BF16],
    patch_weight: pl.Tensor[[VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    patch_bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    position_ids: pl.Tensor[[T_DYN, 2], pl.INT32],
    cosine_table: pl.Tensor[[VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    sine_table: pl.Tensor[[VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    hidden: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
    cosine: pl.Out[pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32]],
    sine: pl.Out[pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32]],
):
    vision_patch_embed(patches, patch_weight, patch_bias, hidden)
    vision_axial_rope_table(position_ids, cosine_table, sine_table, cosine, sine)
    return hidden, cosine, sine


@pl.jit
def vision_block(
    hidden: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    norm1_weight: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[VISION_DEPTH, 3 * VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    qkv_bias: pl.Tensor[[VISION_DEPTH, 3 * VISION_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[VISION_DEPTH, VISION_HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[VISION_DEPTH, VISION_HEAD_DIM], pl.BF16],
    out_weight: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    out_bias: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    norm2_weight: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    gate_weight: pl.Tensor[[VISION_DEPTH, VISION_INTER, VISION_HIDDEN], pl.BF16],
    gate_bias: pl.Tensor[[VISION_DEPTH, VISION_INTER], pl.BF16],
    up_weight: pl.Tensor[[VISION_DEPTH, VISION_INTER, VISION_HIDDEN], pl.BF16],
    up_bias: pl.Tensor[[VISION_DEPTH, VISION_INTER], pl.BF16],
    down_weight: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN, VISION_INTER], pl.BF16],
    down_bias: pl.Tensor[[VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    cosine: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    sine: pl.Tensor[[T_DYN, VISION_HEAD_DIM], pl.FP32],
    segment_start_loc: pl.Tensor[[VISION_STARTS_DYN], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
    layer: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(hidden, 0)
    normalized = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    norm1 = norm1_weight[layer]
    rmsnorm(hidden, norm1, normalized)
    attended = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    qkv_w = qkv_weight[layer]
    qkv_b = qkv_bias[layer]
    q_norm = q_norm_weight[layer]
    k_norm = k_norm_weight[layer]
    out_w = out_weight[layer]
    out_b = out_bias[layer]
    vision_attention(normalized, qkv_w, qkv_b, q_norm, k_norm, cosine, sine,
                     out_w, out_b, segment_start_loc, attended)
    residual = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    for attention_block in pl.spmd(((rows + RESIDUAL_M_TILE - 1) // RESIDUAL_M_TILE) * (VISION_HIDDEN // RESIDUAL_N_TILE), name_hint="vision_attention_residual"):
        attention_row0 = attention_block // (VISION_HIDDEN // RESIDUAL_N_TILE) * RESIDUAL_M_TILE
        attention_col0 = attention_block % (VISION_HIDDEN // RESIDUAL_N_TILE) * RESIDUAL_N_TILE
        attention_rows = pl.min(RESIDUAL_M_TILE, rows - attention_row0)
        previous = pl.slice(hidden, [RESIDUAL_M_TILE, RESIDUAL_N_TILE], [attention_row0, attention_col0], valid_shape=[attention_rows, RESIDUAL_N_TILE])
        update = pl.slice(attended, [RESIDUAL_M_TILE, RESIDUAL_N_TILE], [attention_row0, attention_col0], valid_shape=[attention_rows, RESIDUAL_N_TILE])
        combined = pl.add(pl.cast(previous, pl.FP32), pl.cast(update, pl.FP32))
        narrowed = pl.cast(combined, pl.BF16, mode="rint")
        residual[attention_row0:attention_row0 + RESIDUAL_M_TILE, attention_col0:attention_col0 + RESIDUAL_N_TILE] = pl.set_validshape(narrowed, attention_rows, RESIDUAL_N_TILE)
    mlp_input = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    norm2 = norm2_weight[layer]
    rmsnorm(residual, norm2, mlp_input)
    mlp_output = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    gate_w = gate_weight[layer]
    gate_b = gate_bias[layer]
    up_w = up_weight[layer]
    up_b = up_bias[layer]
    down_w = down_weight[layer]
    down_b = down_bias[layer]
    vision_mlp(mlp_input, gate_w, gate_b, up_w, up_b, down_w, down_b, mlp_output)
    for mlp_block in pl.spmd(((rows + RESIDUAL_M_TILE - 1) // RESIDUAL_M_TILE) * (VISION_HIDDEN // RESIDUAL_N_TILE), name_hint="vision_mlp_residual"):
        mlp_row0 = mlp_block // (VISION_HIDDEN // RESIDUAL_N_TILE) * RESIDUAL_M_TILE
        mlp_col0 = mlp_block % (VISION_HIDDEN // RESIDUAL_N_TILE) * RESIDUAL_N_TILE
        mlp_rows = pl.min(RESIDUAL_M_TILE, rows - mlp_row0)
        previous_mlp = pl.slice(residual, [RESIDUAL_M_TILE, RESIDUAL_N_TILE], [mlp_row0, mlp_col0], valid_shape=[mlp_rows, RESIDUAL_N_TILE])
        update_mlp = pl.slice(mlp_output, [RESIDUAL_M_TILE, RESIDUAL_N_TILE], [mlp_row0, mlp_col0], valid_shape=[mlp_rows, RESIDUAL_N_TILE])
        combined_mlp = pl.add(pl.cast(previous_mlp, pl.FP32), pl.cast(update_mlp, pl.FP32))
        narrowed_mlp = pl.cast(combined_mlp, pl.BF16, mode="rint")
        output[mlp_row0:mlp_row0 + RESIDUAL_M_TILE, mlp_col0:mlp_col0 + RESIDUAL_N_TILE] = pl.set_validshape(narrowed_mlp, mlp_rows, RESIDUAL_N_TILE)
    return output


@pl.jit
def vision_finish(
    hidden: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
    post_norm_weight: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    downsample_weight: pl.Tensor[[D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    downsample_bias: pl.Tensor[[D], pl.BF16],
    merger_proj_weight: pl.Tensor[[D, D], pl.BF16],
    merger_norm_weight: pl.Tensor[[D], pl.BF16],
    merger_norm_bias: pl.Tensor[[D], pl.BF16],
    merger_gate_weight: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    merger_up_weight: pl.Tensor[[VISION_PROJ_INTER, D], pl.BF16],
    merger_down_weight: pl.Tensor[[D, VISION_PROJ_INTER], pl.BF16],
    features: pl.Out[pl.Tensor[[VISION_ROWS_DYN, D], pl.BF16]],
):
    rows = pl.tensor.dim(hidden, 0)
    output_rows = pl.tensor.dim(features, 0)
    normalized = pl.create_tensor([rows, VISION_HIDDEN], dtype=pl.BF16)
    rmsnorm(hidden, post_norm_weight, normalized)
    grouped = pl.reshape(normalized, [output_rows, VISION_MERGE * VISION_MERGE, VISION_HIDDEN])
    vision_merger(grouped, downsample_weight, downsample_bias, merger_proj_weight,
                  merger_norm_weight, merger_norm_bias, merger_gate_weight, merger_up_weight, merger_down_weight, features)
    return features


@pl.jit.host
def vision_forward(
    patches: pl.Tensor[[1, T_DYN, VISION_PATCH_IN], pl.BF16],
    patch_weight: pl.Tensor[[1, VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    patch_bias: pl.Tensor[[1, VISION_HIDDEN], pl.BF16],
    norm1_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    qkv_weight: pl.Tensor[[1, VISION_DEPTH, 3 * VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    qkv_bias: pl.Tensor[[1, VISION_DEPTH, 3 * VISION_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HEAD_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HEAD_DIM], pl.BF16],
    out_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN, VISION_HIDDEN], pl.BF16],
    out_bias: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    norm2_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    gate_weight: pl.Tensor[[1, VISION_DEPTH, VISION_INTER, VISION_HIDDEN], pl.BF16],
    gate_bias: pl.Tensor[[1, VISION_DEPTH, VISION_INTER], pl.BF16],
    up_weight: pl.Tensor[[1, VISION_DEPTH, VISION_INTER, VISION_HIDDEN], pl.BF16],
    up_bias: pl.Tensor[[1, VISION_DEPTH, VISION_INTER], pl.BF16],
    down_weight: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN, VISION_INTER], pl.BF16],
    down_bias: pl.Tensor[[1, VISION_DEPTH, VISION_HIDDEN], pl.BF16],
    post_norm_weight: pl.Tensor[[1, VISION_HIDDEN], pl.BF16],
    downsample_weight: pl.Tensor[[1, D, VISION_MERGE * VISION_MERGE * VISION_HIDDEN], pl.BF16],
    downsample_bias: pl.Tensor[[1, D], pl.BF16],
    merger_proj_weight: pl.Tensor[[1, D, D], pl.BF16],
    merger_norm_weight: pl.Tensor[[1, D], pl.BF16],
    merger_norm_bias: pl.Tensor[[1, D], pl.BF16],
    merger_gate_weight: pl.Tensor[[1, VISION_PROJ_INTER, D], pl.BF16],
    merger_up_weight: pl.Tensor[[1, VISION_PROJ_INTER, D], pl.BF16],
    merger_down_weight: pl.Tensor[[1, D, VISION_PROJ_INTER], pl.BF16],
    position_ids: pl.Tensor[[1, T_DYN, 2], pl.INT32],
    segment_start_loc: pl.Tensor[[1, VISION_STARTS_DYN], pl.INT32],
    cosine_table: pl.Tensor[[1, VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    sine_table: pl.Tensor[[1, VISION_POSITIONS_DYN, VISION_HEAD_DIM // 4], pl.FP32],
    features: pl.Out[pl.Tensor[[1, VISION_ROWS_DYN, D], pl.BF16]],
    workspace_a: pl.InOut[pl.Tensor[[1, T_DYN, VISION_HIDDEN], pl.BF16]],
    workspace_b: pl.InOut[pl.Tensor[[1, T_DYN, VISION_HIDDEN], pl.BF16]],
    cosine: pl.Out[pl.Tensor[[1, T_DYN, VISION_HEAD_DIM], pl.FP32]],
    sine: pl.Out[pl.Tensor[[1, T_DYN, VISION_HEAD_DIM], pl.FP32]],
    num_layers: pl.Scalar[pl.INT32],
):
    patches.bind_dynamic(1, T_DYN)
    features.bind_dynamic(1, VISION_ROWS_DYN)
    workspace_a.bind_dynamic(1, T_DYN)
    workspace_b.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    segment_start_loc.bind_dynamic(1, VISION_STARTS_DYN)
    cosine.bind_dynamic(1, T_DYN)
    sine.bind_dynamic(1, T_DYN)
    cosine_table.bind_dynamic(1, VISION_POSITIONS_DYN)
    sine_table.bind_dynamic(1, VISION_POSITIONS_DYN)
    vision_stem(
        patches[0], patch_weight[0], patch_bias[0], position_ids[0], cosine_table[0], sine_table[0],
        workspace_a[0], cosine[0], sine[0], device=0,
    )
    for layer in pl.range(num_layers):
        if layer % 2 == 0:
            vision_block(
                workspace_a[0], norm1_weight[0],
                qkv_weight[0], qkv_bias[0], q_norm_weight[0], k_norm_weight[0], out_weight[0], out_bias[0],
                norm2_weight[0], gate_weight[0], gate_bias[0], up_weight[0], up_bias[0],
                down_weight[0], down_bias[0], cosine[0], sine[0], segment_start_loc[0],
                workspace_b[0], layer, device=0,
            )
        else:
            vision_block(
                workspace_b[0], norm1_weight[0],
                qkv_weight[0], qkv_bias[0], q_norm_weight[0], k_norm_weight[0], out_weight[0], out_bias[0],
                norm2_weight[0], gate_weight[0], gate_bias[0], up_weight[0], up_bias[0],
                down_weight[0], down_bias[0], cosine[0], sine[0], segment_start_loc[0],
                workspace_a[0], layer, device=0,
            )
    if num_layers % 2 == 0:
        vision_finish(
            workspace_a[0], post_norm_weight[0], downsample_weight[0], downsample_bias[0],
            merger_proj_weight[0], merger_norm_weight[0], merger_norm_bias[0],
            merger_gate_weight[0], merger_up_weight[0], merger_down_weight[0],
            features[0], device=0,
        )
    else:
        vision_finish(
            workspace_b[0], post_norm_weight[0], downsample_weight[0], downsample_bias[0],
            merger_proj_weight[0], merger_norm_weight[0], merger_norm_bias[0],
            merger_gate_weight[0], merger_up_weight[0], merger_down_weight[0],
            features[0], device=0,
        )


def golden_vision_single(values):
    hidden = golden_vision_patch_embed(values["patches"], values["patch_weight"], values["patch_bias"])
    values["workspace_a"].copy_(hidden)
    height, width = values["position_ids"].long().unbind(dim=-1)
    cosine_hw = torch.cat((values["cosine_table"][height], values["cosine_table"][width]), dim=-1)
    sine_hw = torch.cat((values["sine_table"][height], values["sine_table"][width]), dim=-1)
    cosine, sine = cosine_hw.repeat(1, 2), sine_hw.repeat(1, 2)
    values["cosine"].copy_(cosine)
    values["sine"].copy_(sine)
    for layer in range(int(values["num_layers"])):
        normalized = rms_norm(hidden, values["norm1_weight"][layer], VISION_RMS_NORM_EPS)
        attended = golden_vision_attention(
            normalized, values["qkv_weight"][layer], values["qkv_bias"][layer],
            values["q_norm_weight"][layer], values["k_norm_weight"][layer], cosine, sine,
            values["out_weight"][layer], values["out_bias"][layer], values["segment_start_loc"],
        )
        residual = (hidden.float() + attended.float()).bfloat16()
        normalized_mlp = rms_norm(residual, values["norm2_weight"][layer], VISION_RMS_NORM_EPS)
        mlp_output = golden_vision_mlp(
            normalized_mlp, values["gate_weight"][layer], values["gate_bias"][layer],
            values["up_weight"][layer], values["up_bias"][layer], values["down_weight"][layer], values["down_bias"][layer],
        )
        hidden = (residual.float() + mlp_output.float()).bfloat16()
        values["workspace_b" if layer % 2 == 0 else "workspace_a"].copy_(hidden)
    post_norm = rms_norm(hidden, values["post_norm_weight"], VISION_RMS_NORM_EPS)
    grouped = post_norm.reshape(-1, VISION_MERGE ** 2, VISION_HIDDEN)
    features = golden_vision_merger(
        grouped, values["downsample_weight"], values["downsample_bias"], values["merger_proj_weight"],
        values["merger_norm_weight"], values["merger_norm_bias"], values["merger_gate_weight"],
        values["merger_up_weight"], values["merger_down_weight"],
    )
    values["features"].copy_(features)


def golden_vision_forward(values):
    inner = {name: value[0] if isinstance(value, torch.Tensor) and value.ndim else value
             for name, value in values.items()}
    golden_vision_single(inner)


class VisionEncoder:
    """Execute the checkpoint's BF16 vision tower on one allocated device."""

    def __init__(self, checkpoint, device_id, *, platform="a2a3", compiled=None):
        self.checkpoint = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
        self.weights = self.checkpoint.vision_weights()
        self.config = RunConfig(platform=platform, distributed_config=DistributedConfig(device_ids=[device_id], num_sub_workers=0))
        self.program = vision_forward.compile(config=self.config) if compiled is None else compiled
        config = self.checkpoint._json("config.json")["vision_config"]
        rope = config.get("rope_parameters") or {"rope_type": "axial", "rope_theta": 10000.0}
        if rope.get("rope_type") != "axial" or float(rope.get("rope_theta", 0)) <= 0:
            raise ValueError("vision RoPE must use an axial positive-frequency base")
        powers = torch.arange(VISION_HEAD_DIM // 4).float() * 4 / VISION_HEAD_DIM
        self.inv_freq = float(rope["rope_theta"]) ** (-powers)

    def __call__(self, pixel_values, grid_thw):
        positions, starts = vision_metadata(grid_thw)
        if tuple(pixel_values.shape) != (positions.shape[0], VISION_PATCH_IN):
            raise ValueError("pixel_values rows must match grid_thw and the flattened patch width")
        if not torch.isfinite(pixel_values).all():
            raise ValueError("pixel_values must be finite")
        patches = pixel_values.cpu().bfloat16().contiguous()
        features = torch.empty((patches.shape[0] // VISION_MERGE ** 2, D), dtype=torch.bfloat16)
        rows = patches.shape[0]
        workspace_a = torch.zeros((1, rows, VISION_HIDDEN), dtype=torch.bfloat16)
        workspace_b = torch.zeros_like(workspace_a)
        cosine = torch.empty((1, rows, VISION_HEAD_DIM), dtype=torch.float32)
        sine = torch.empty_like(cosine)
        cosine_table, sine_table = vision_rope_tables(positions, self.inv_freq)
        arguments = [patches.unsqueeze(0), *(self.weights[name].unsqueeze(0) for name in vision_weight_layouts()),
                     positions.unsqueeze(0), starts.unsqueeze(0), cosine_table.unsqueeze(0), sine_table.unsqueeze(0),
                     features.unsqueeze(0),
                     workspace_a, workspace_b, cosine, sine, torch.tensor(VISION_DEPTH, dtype=torch.int32)]
        self.program(*arguments, config=self.config)
        if not torch.isfinite(features).all():
            raise FloatingPointError("vision encoder produced nonfinite features")
        return features


def main():
    import argparse
    from functools import lru_cache
    import json

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--pixels", type=Path)
    parser.add_argument("--grid", default="[[1, 2, 2], [1, 2, 6]]")
    parser.add_argument("--layers", type=int, default=VISION_DEPTH)
    parser.add_argument("--compile-only", action="store_true")
    options = parser.parse_args()
    if not 0 <= options.layers <= VISION_DEPTH:
        parser.error(f"--layers must be in [0, {VISION_DEPTH}]")
    positions, starts = vision_metadata(json.loads(options.grid))
    rows = positions.shape[0]
    layouts = vision_weight_layouts()

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5355)
        if options.checkpoint is not None:
            weights = Checkpoint(options.checkpoint).vision_weights()
        else:
            weights = {}
            for name, shape in layouts.items():
                if "norm" in name and "bias" not in name:
                    weights[name] = torch.ones(shape, dtype=torch.bfloat16)
                else:
                    scale = 0.1 if "bias" in name else shape[-1] ** -0.5
                    weights[name] = (torch.randn(shape, generator=generator) * scale).bfloat16()
        if options.pixels is None:
            patches = torch.randn(rows, VISION_PATCH_IN, generator=generator).bfloat16()
        else:
            patches = torch.load(options.pixels, map_location="cpu", weights_only=True)
            if tuple(patches.shape) != (rows, VISION_PATCH_IN) or not torch.isfinite(patches).all():
                raise ValueError("--pixels must contain finite patches matching --grid")
            patches = patches.bfloat16()
        return {"patches": patches, **weights}

    shapes = {"patches": (rows, VISION_PATCH_IN), **layouts}
    specs = [TensorSpec(name, (1, *shape), torch.bfloat16, init_value=lambda key=name: inputs()[key].unsqueeze(0))
             for name, shape in shapes.items()]
    inverse = 10000.0 ** (-torch.arange(VISION_HEAD_DIM // 4).float() * 4 / VISION_HEAD_DIM)
    cosine_table, sine_table = vision_rope_tables(positions, inverse)
    specs.extend([
        TensorSpec("position_ids", (1, *positions.shape), torch.int32, init_value=positions.unsqueeze(0)),
        TensorSpec("segment_start_loc", (1, *starts.shape), torch.int32, init_value=starts.unsqueeze(0)),
        TensorSpec("cosine_table", (1, *cosine_table.shape), torch.float32, init_value=cosine_table.unsqueeze(0)),
        TensorSpec("sine_table", (1, *sine_table.shape), torch.float32, init_value=sine_table.unsqueeze(0)),
        TensorSpec("features", [1, rows // VISION_MERGE ** 2, D], torch.bfloat16),
        TensorSpec("workspace_a", [1, rows, VISION_HIDDEN], torch.bfloat16, init_value=torch.zeros),
        TensorSpec("workspace_b", [1, rows, VISION_HIDDEN], torch.bfloat16, init_value=torch.zeros),
        TensorSpec("cosine", [1, rows, VISION_HEAD_DIM], torch.float32),
        TensorSpec("sine", [1, rows, VISION_HEAD_DIM], torch.float32),
        ScalarSpec("num_layers", torch.int32, options.layers),
    ])
    result = run(
        fn=vision_forward,
        specs=specs,
        golden_fn=golden_vision_forward,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, distributed_config=DistributedConfig(device_ids=[options.device], num_sub_workers=0)),
        compare_fn={"features": ratio_allclose(atol=0.02, rtol=0.02, max_error_ratio=0.05),
                    "workspace_a": ratio_allclose(atol=0.02, rtol=0.02, max_error_ratio=0.05),
                    "workspace_b": ratio_allclose(atol=0.02, rtol=0.02, max_error_ratio=0.05),
                    "cosine": ratio_allclose(atol=1e-5, rtol=1e-5, max_error_ratio=0),
                    "sine": ratio_allclose(atol=1e-5, rtol=1e-5, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
