# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch references for the W8A8 checkpoint format used on a2a3.

The a2a3 cube has no fp8 or MX ``mmad`` — ``Intrinsic_mmad`` in
``Ascend910_9392.ini`` lists only ``s32s8s8`` / ``u32u8u8`` / ``u8s8`` and the
fp16/fp32 forms — so the released FP8-blockwise checkpoint cannot execute here and
int8 is the only quantized path. The deployment weights are therefore the
msmodelslim W8A8 conversion that vLLM Ascend serves on a 16-card A3 node.

The deployment checkpoint is ``Eco-Tech/GLM-5.3-Flash-w8a8`` and its
``quant_model_description.json`` declares one scheme and one only:

    {"group_size": 0, "model_quant_type": "W8A8_DYNAMIC", "is_rot_used": true}

So there is a single flavour: int8 weights with an FP32 per-output-channel
``weight_scale``, and the activation quantized per token at run time. ``group_size:
0`` means per-channel rather than grouped. Every quantized weight also ships a
``weight_offset``, but it is **all zero** — measured across three tensors of 12288,
2048 and 4096 channels — so the weight quantization is symmetric and the loader
reads and discards the offset. There is no static-activation path and no
SmoothQuant vector in this checkpoint.
"""

import torch

from models.glm5_3_flash.config import INT8_K_ALIGN


# Shared with models/deepseek_v4_flash_mtp/config.py:282-283, the a2a3 W8A8 sibling
# port. The amax floor avoids 127/0 on an all-zero row.
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4


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
        raise ValueError(
            f"the a2a3 int8 cube needs K aligned to {INT8_K_ALIGN}, got {x_int8.shape[-1]}"
        )
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


__all__ = [
    "INT8_AMAX_EPS",
    "INT8_SCALE_MAX",
    "dequant_swiglu_quant",
    "int8_matmul",
    "quantize_per_channel_int8",
    "quantize_per_token_int8",
    "w8a8_dynamic_linear",
]


if __name__ == "__main__":
    from models.glm5_3_flash._golden_smoke import run_quantization_goldens

    run_quantization_goldens(
        quantize_per_channel_int8, quantize_per_token_int8, w8a8_dynamic_linear
    )
