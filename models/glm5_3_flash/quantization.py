# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash dynamic INT8 quantization and W8A8 references."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import FLASH
from models.glm5_3_flash.elementwise import accurate_divide, accurate_reciprocal
from models.glm5_3_flash.utils import _check, _report

INT8_K_ALIGN = 32
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
QUANT_ROWS = pl.dynamic("GLM53_QUANT_ROWS")
QUANT_WIDTH = pl.dynamic("GLM53_QUANT_WIDTH")
QUANT_M_TILE = 8
QUANT_N_TILE = 128


@pl.jit.inline
def quantize_bf16_rows(
    x: pl.Tensor[[QUANT_ROWS, QUANT_WIDTH], pl.BF16],
    quantized: pl.Tensor[[QUANT_ROWS, QUANT_WIDTH], pl.INT8],
    scales: pl.Tensor[[QUANT_ROWS, 8], pl.FP32],
):
    """Quantize finite BF16 rows and replicate each dequant scale across eight lanes."""
    rows = pl.tensor.dim(x, 0)
    width = pl.tensor.dim(x, 1)
    for block in pl.spmd((rows + QUANT_M_TILE - 1) // QUANT_M_TILE, name_hint="glm53_dynamic_quant"):
        row0 = block * QUANT_M_TILE
        valid_rows = pl.min(QUANT_M_TILE, rows - row0)
        maximum = pl.full([1, QUANT_M_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for col0 in pl.range(0, width, QUANT_N_TILE):
            valid_columns = pl.min(QUANT_N_TILE, width - col0)
            source = pl.slice(x, [QUANT_M_TILE, QUANT_N_TILE], [row0, col0], valid_shape=[valid_rows, valid_columns])
            padded = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), QUANT_M_TILE, QUANT_N_TILE)
            value = pl.cast(padded, pl.FP32)
            part_maximum = pl.reshape(pl.row_max(pl.abs(value)), [1, QUANT_M_TILE])
            maximum = pl.maximum(maximum, part_maximum)
        numerator = pl.create_tensor([1, QUANT_M_TILE], dtype=pl.FP32)
        denominator = pl.create_tensor([1, QUANT_M_TILE], dtype=pl.FP32)
        scale_buffer = pl.create_tensor([1, QUANT_M_TILE], dtype=pl.FP32)
        numerator[:, :] = pl.full([1, QUANT_M_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        denominator[:, :] = maximum
        scale_buffer[:, :] = accurate_divide(numerator, denominator)
        inverse_scale = accurate_reciprocal(scale_buffer)
        inverse_column = pl.reshape(inverse_scale, [QUANT_M_TILE, 1])
        scale_column = pl.reshape(scale_buffer, [QUANT_M_TILE, 1])
        scale_lanes = pl.full([QUANT_M_TILE, 8], dtype=pl.FP32, value=0.0)
        scale_values = pl.row_expand(scale_lanes, inverse_column)
        scales[row0 : row0 + QUANT_M_TILE, :] = pl.set_validshape(scale_values, valid_rows, 8)
        for col0 in pl.range(0, width, QUANT_N_TILE):
            valid_columns = pl.min(QUANT_N_TILE, width - col0)
            source = pl.slice(x, [QUANT_M_TILE, QUANT_N_TILE], [row0, col0], valid_shape=[valid_rows, valid_columns])
            value = pl.cast(source, pl.FP32)
            scaled = pl.row_expand_mul(value, scale_column)
            integer = pl.cast(scaled, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(integer, -127), 127)
            half = pl.cast(bounded, pl.FP16, mode="rint")
            result = pl.cast(half, pl.INT8, mode="trunc")
            quantized[row0:row0 + QUANT_M_TILE, col0:col0 + QUANT_N_TILE] = pl.set_validshape(result, valid_rows, valid_columns)
    return quantized, scales


def _round_to_int8(scaled: torch.Tensor) -> torch.Tensor:
    """Round the way the device does: int32, then through fp16, then int8.

    The clamp bounds the result at +/-127 rather than the int8 minimum of -128.
    That is deliberate and symmetric, but it means this reference cannot reproduce
    a checkpoint weight bit-exactly: about 0.003% of the elements of
    ``layers.0.mlp.gate_proj.weight`` are exactly -128. It does not affect the
    dequant path, since the stored ``weight_scale`` is what the kernel multiplies by.
    """
    rounded = torch.round(scaled).to(torch.int32)
    rounded = torch.clamp(rounded, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return rounded.to(torch.float16).to(torch.int8)


def quantize_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 weight quantization.

    Args:
        weight: ``[out_features, in_features]`` in any float dtype.

    Returns:
        The int8 weight and its ``[out_features, 1]`` FP32 **dequant** scale.
    """
    amax = weight.float().abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = torch.div(torch.full_like(amax, INT8_SCALE_MAX), amax)
    return _round_to_int8(weight.float() * scale_quant), 1.0 / scale_quant


def quantize_per_token_int8(
    x: torch.Tensor,
    smooth_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token int8 activation quantization.

    Mirrors ``torch_npu.npu_dynamic_quant``. ``smooth_scale`` is the optional
    per-input-channel SmoothQuant vector folded in before the row maximum.

    Args:
        x: ``[..., in_features]``.
        smooth_scale: ``[in_features]`` or ``None``.

    Returns:
        The int8 activation and its ``[..., 1]`` FP32 **dequant** scale.
    """
    value = x.float()
    if smooth_scale is not None:
        value = value * smooth_scale.float()
    amax = value.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = torch.div(torch.full_like(amax, INT8_SCALE_MAX), amax)
    return _round_to_int8(value * scale_quant), 1.0 / scale_quant


def int8_matmul(x_int8: torch.Tensor, weight_int8: torch.Tensor) -> torch.Tensor:
    """``[..., K] x [N, K] -> [..., N]`` int8 matmul accumulating in int32.

    The a2a3 int8 cube fractal is MKN ``16, 32, 16``, so a kernel's K tile must be a
    multiple of :data:`INT8_K_ALIGN`; the reference itself is layout-free.
    """
    if x_int8.shape[-1] % INT8_K_ALIGN:
        raise ValueError(f"the a2a3 int8 cube needs K aligned to {INT8_K_ALIGN}, got {x_int8.shape[-1]}")
    return torch.matmul(x_int8.to(torch.int32), weight_int8.to(torch.int32).transpose(-2, -1))


def w8a8_dynamic_linear(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Per-token activation quant, int8 matmul, per-channel dequant.

    Mirrors ``torch_npu.npu_dynamic_quant`` followed by ``torch_npu.npu_quant_matmul``
    with a ``pertoken_scale``.

    Args:
        x: ``[..., in_features]`` activation.
        weight_int8: ``[out_features, in_features]``.
        weight_scale: ``[out_features]`` or ``[out_features, 1]``.
    """
    x_int8, token_scale = quantize_per_token_int8(x, smooth_scale)
    accumulator = int8_matmul(x_int8, weight_int8).float()
    result = accumulator * token_scale * weight_scale.float().reshape(-1)
    if bias is not None:
        result = result + bias.float()
    return result.to(out_dtype)


def dequant_swiglu_quant(
    gate_up_accumulator: torch.Tensor,
    gate_up_scale: torch.Tensor,
    token_scale: torch.Tensor,
    swiglu_limit: float,
    down_smooth_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the expert epilogue: dequant, clamped SwiGLU, then re-quantize for ``down_proj``.

    This is the reference for ``torch_npu.npu_dequant_swiglu_quant``, which the
    Ascend-native GLM-5.2 recipe calls between the two grouped matmuls so the
    ``2 * moe_intermediate_size`` intermediate never materialises in BF16.

    Args:
        gate_up_accumulator: ``[tokens, 2 * moe_intermediate_size]`` int32 from the
            fused gate/up matmul, gate first.
        gate_up_scale: ``[2 * moe_intermediate_size]`` per-channel weight scale.
        token_scale: ``[tokens, 1]`` per-token activation scale.
        down_smooth_scale: optional SmoothQuant vector for the ``down_proj`` input.

    Returns:
        The int8 activation for ``down_proj`` and its ``[tokens, 1]`` FP32 scale.
    """
    value = gate_up_accumulator.float() * token_scale * gate_up_scale.float()
    gate_value, up_value = value.chunk(2, dim=-1)
    gate_value = gate_value.clamp(max=swiglu_limit)
    up_value = up_value.clamp(-swiglu_limit, swiglu_limit)
    hidden = torch.nn.functional.silu(gate_value) * up_value
    return quantize_per_token_int8(hidden, down_smooth_scale)


def swiglu(gate_value: torch.Tensor, up_value: torch.Tensor) -> torch.Tensor:
    """Clamp-stabilised SwiGLU shared by the dense MLP, shared expert and routed experts.

    ``gate`` is clamped from above only; ``up`` is clamped on both sides.
    """
    limit = FLASH.swiglu_limit
    gate_value = gate_value.float().clamp(max=limit)
    up_value = up_value.float().clamp(-limit, limit)
    return F.silu(gate_value) * up_value


def expert(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    route_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate one clamp-stabilised SwiGLU expert (routed, shared, or dense)."""
    hidden = swiglu(F.linear(x, w_gate), F.linear(x, w_up))
    if route_weight is not None:
        hidden = hidden * route_weight
    return F.linear(hidden.to(x.dtype), w_down)


def run_swiglu_golden(golden_swiglu) -> None:
    """Check the clamped SwiGLU's saturation behaviour at the configured limit."""
    torch.manual_seed(17)
    limit = FLASH.swiglu_limit
    gate_value = torch.tensor([[-100.0, 0.0, 100.0]])
    up_value = torch.tensor([[-100.0, 1.0, 100.0]])

    hidden = golden_swiglu(gate_value, up_value)
    _check(hidden.shape == gate_value.shape, f"swiglu shape {tuple(hidden.shape)}")
    # `up` is clamped on both sides, `gate` only from above.
    expected_high = torch.nn.functional.silu(torch.tensor(limit)) * limit
    _check(
        torch.allclose(hidden[0, 2], expected_high, atol=1e-05),
        f"swiglu did not clamp the upper tail: {hidden[0, 2]} vs {expected_high}",
    )
    expected_low = torch.nn.functional.silu(torch.tensor(-100.0)) * -limit
    _check(
        torch.allclose(hidden[0, 0], expected_low, atol=1e-05),
        f"swiglu clamped the gate from below: {hidden[0, 0]} vs {expected_low}",
    )

    _report("swiglu")


def run_expert_golden(golden_expert) -> None:
    """Check one clamped-SwiGLU expert FFN: routed, shared and dense share it."""
    torch.manual_seed(29)
    tokens, hidden, inter = 6, 64, 128
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    w_gate = torch.randn(inter, hidden, dtype=torch.bfloat16)
    w_up = torch.randn(inter, hidden, dtype=torch.bfloat16)
    w_down = torch.randn(hidden, inter, dtype=torch.bfloat16)

    out = golden_expert(x, w_gate, w_up, w_down)
    _check(out.shape == (tokens, hidden), f"expert output shape {tuple(out.shape)}")

    route_weight = torch.rand(tokens, 1, dtype=torch.bfloat16)
    out_routed = golden_expert(x, w_gate, w_up, w_down, route_weight)
    _check(out_routed.shape == out.shape, f"routed expert output shape {tuple(out_routed.shape)}")
    # The routing weight is folded into the BF16 hidden before ``down``, so the
    # comparison to ``out * weight`` carries a few BF16 ULPs of requant noise
    # amplified by the down projection. A per-row relative check admits that.
    expected = out.float() * route_weight.float()
    row_denom = expected.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    row_reldiff = (out_routed.float() - expected).abs().amax(dim=-1) / row_denom.squeeze(-1)
    _check(
        bool((row_reldiff < 0.01).all()),
        f"routed expert must fold the per-row weight into its output (max row reldiff {row_reldiff.max().item():.5f})",
    )
    # The plain path must reproduce the same maths with a unit weight.
    _check(
        torch.allclose(golden_expert(x, w_gate, w_up, w_down, torch.ones(tokens, 1)).float(), out.float(), atol=0.001),
        "expert must be invariant to a unit routing weight",
    )

    _report("expert")


def run_quantization_goldens(quantize_per_channel, quantize_per_token, dynamic_linear) -> None:
    """Check the W8A8 references reconstruct a float matmul within int8 error."""
    torch.manual_seed(23)
    tokens, in_features, out_features = 6, 128, 64
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    weight_int8, weight_scale = quantize_per_channel(weight)
    _check(weight_int8.dtype is torch.int8, f"weight dtype {weight_int8.dtype}")
    _check(weight_scale.shape == (out_features, 1), f"per-channel scale shape {tuple(weight_scale.shape)}")

    x_int8, token_scale = quantize_per_token(x)
    _check(x_int8.dtype is torch.int8, f"activation dtype {x_int8.dtype}")
    _check(token_scale.shape == (tokens, 1), f"per-token scale shape {tuple(token_scale.shape)}")

    reference = torch.nn.functional.linear(x.float(), weight.float())
    quantized = dynamic_linear(x, weight_int8, weight_scale, out_dtype=torch.float32)
    error = (quantized - reference).abs().max() / reference.abs().max()
    _check(float(error) < 0.05, f"W8A8 reconstruction error is too large: {float(error)}")

    _report("quantization")


if __name__ == "__main__":
    run_swiglu_golden(swiglu)
    run_expert_golden(expert)

    run_quantization_goldens(quantize_per_channel_int8, quantize_per_token_int8, w8a8_dynamic_linear)
