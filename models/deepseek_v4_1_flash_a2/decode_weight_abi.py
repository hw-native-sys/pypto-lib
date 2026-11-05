# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only adapter from Aurora safetensors to the PyPTO EP2 weight ABI.

The checkpoint stores logical row-major W8A8 matrices.  The routed-expert
kernel consumes NZ matrices and an aligned W2 reduction width, while the
shared-expert kernel consumes logical matrices and performs its own packing.
This module owns that representation change and is intentionally outside the
PyPTO product import closure.  It never executes model computation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from .decode_weights import DecodeSafetensorsCheckpoint
from .utils import pack_nz


HIDDEN = 5120
INTERMEDIATE = 2304
INTERMEDIATE_PADDED = 2560
EXPERTS = 384
LOCAL_EXPERTS = 192
TOP_K = 6


def _name(layer: int, suffix: str) -> str:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    return f"layers.{layer}.ffn.{suffix}"


def _expert_name(layer: int, expert: int, suffix: str) -> str:
    if not 0 <= expert < EXPERTS:
        raise ValueError(f"expert must be in [0, {EXPERTS}), got {expert}")
    return _name(layer, f"experts.{expert}.{suffix}")


def _require(tensor: torch.Tensor, name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    if tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
            f"expected dtype={dtype}, shape={shape}"
        )
    return tensor.contiguous()


def _require_zero_offset(tensor: torch.Tensor, name: str, shape: tuple[int, ...]) -> None:
    offset = _require(tensor, name, torch.float32, shape)
    if bool(torch.count_nonzero(offset).item()):
        raise ValueError(
            f"{name} contains non-zero affine offsets; the current PyPTO W8A8 ABI "
            "supports the verified symmetric zero-offset export only"
        )


def _scale(tensor: torch.Tensor, name: str, channels: int) -> torch.Tensor:
    value = _require(tensor, name, torch.float32, (channels, 1))
    return value.reshape(channels).contiguous()


def _pad_w2(weight: torch.Tensor, name: str) -> torch.Tensor:
    logical = _require(weight, name, torch.int8, (HIDDEN, INTERMEDIATE))
    padded = torch.zeros((HIDDEN, INTERMEDIATE_PADDED), dtype=torch.int8)
    padded[:, :INTERMEDIATE].copy_(logical)
    return padded


def pack_routed_expert(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    expert: int,
) -> dict[str, torch.Tensor]:
    """Load and pack one routed expert for ``decode_ep2_rank``.

    W1/W3 are already in the kernel's logical ``[out, in]`` orientation.
    W2 is also ``[hidden, intermediate]``; only its reduction tail is padded
    from 2304 to 2560 for the A3 INT8 matmul tile contract.
    """

    w1 = _require(
        checkpoint.tensor(_expert_name(layer, expert, "w1.weight")),
        _expert_name(layer, expert, "w1.weight"),
        torch.int8,
        (INTERMEDIATE, HIDDEN),
    )
    w3 = _require(
        checkpoint.tensor(_expert_name(layer, expert, "w3.weight")),
        _expert_name(layer, expert, "w3.weight"),
        torch.int8,
        (INTERMEDIATE, HIDDEN),
    )
    w2_name = _expert_name(layer, expert, "w2.weight")
    w2 = _pad_w2(checkpoint.tensor(w2_name), w2_name)

    w1_scale_name = _expert_name(layer, expert, "w1.weight_scale")
    w3_scale_name = _expert_name(layer, expert, "w3.weight_scale")
    w2_scale_name = _expert_name(layer, expert, "w2.weight_scale")
    w1_offset_name = _expert_name(layer, expert, "w1.weight_offset")
    w3_offset_name = _expert_name(layer, expert, "w3.weight_offset")
    w2_offset_name = _expert_name(layer, expert, "w2.weight_offset")
    _require_zero_offset(checkpoint.tensor(w1_offset_name), w1_offset_name, (INTERMEDIATE, 1))
    _require_zero_offset(checkpoint.tensor(w3_offset_name), w3_offset_name, (INTERMEDIATE, 1))
    _require_zero_offset(checkpoint.tensor(w2_offset_name), w2_offset_name, (HIDDEN, 1))

    return {
        "w1": pack_nz(w1),
        "w1_scale": _scale(checkpoint.tensor(w1_scale_name), w1_scale_name, INTERMEDIATE),
        "w3": pack_nz(w3),
        "w3_scale": _scale(checkpoint.tensor(w3_scale_name), w3_scale_name, INTERMEDIATE),
        "w2": pack_nz(w2),
        "w2_scale": _scale(checkpoint.tensor(w2_scale_name), w2_scale_name, HIDDEN),
    }


def iter_routed_experts(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """Yield the 192 owner experts without retaining a second full shard."""

    if rank not in (0, 1):
        raise ValueError(f"EP2 rank must be 0 or 1, got {rank}")
    first = rank * LOCAL_EXPERTS
    for expert in range(first, first + LOCAL_EXPERTS):
        yield expert, pack_routed_expert(checkpoint, layer, expert)


def load_routed_shard(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Materialize one complete 192-expert EP2 shard in external NZ layout.

    The ``pl.NZ`` annotation is logical in the Python signature, but the
    distributed host ABI passes the compiler's physical five-dimensional
    view.  The first dimension is the local expert, followed by
    ``[C / C0, R / 16, 16, C0]``.  Keeping that shape here prevents the
    validation runner from silently treating packed bytes as row-major data.
    """

    entries = [packed for _, packed in iter_routed_experts(checkpoint, layer, rank)]

    def stack_nz(name: str, rows: int, cols: int) -> torch.Tensor:
        value = torch.stack([entry[name] for entry in entries])
        return value.reshape(LOCAL_EXPERTS, cols // 32, rows // 16, 16, 32).contiguous()

    return {
        "routed_w1": stack_nz("w1", INTERMEDIATE, HIDDEN),
        "routed_w1_scale": torch.stack([entry["w1_scale"] for entry in entries]),
        "routed_w3": stack_nz("w3", INTERMEDIATE, HIDDEN),
        "routed_w3_scale": torch.stack([entry["w3_scale"] for entry in entries]),
        "routed_w2": stack_nz("w2", HIDDEN, INTERMEDIATE_PADDED),
        "routed_w2_scale": torch.stack([entry["w2_scale"] for entry in entries]),
    }


def load_router(checkpoint: DecodeSafetensorsCheckpoint, layer: int) -> dict[str, torch.Tensor]:
    """Load the replicated FP32 router operands for one physical layer."""

    gate_name = _name(layer, "gate.weight")
    text_name = _name(layer, "gate.bias")
    vision_name = _name(layer, "gate.bias_vl")
    gate = _require(checkpoint.tensor(gate_name), gate_name, torch.bfloat16, (EXPERTS, HIDDEN))
    text = _require(checkpoint.tensor(text_name), text_name, torch.float32, (EXPERTS,))
    vision = _require(checkpoint.tensor(vision_name), vision_name, torch.float32, (EXPERTS,))
    return {
        "gate_weight": gate.float().contiguous(),
        "text_bias": text,
        "vision_bias": vision,
    }


def load_shared_expert(checkpoint: DecodeSafetensorsCheckpoint, layer: int) -> dict[str, torch.Tensor]:
    """Load shared W8A8 weights in the logical ABI consumed by ``moe_shared``."""

    w1_name = _name(layer, "shared_experts.w1.weight")
    w3_name = _name(layer, "shared_experts.w3.weight")
    w2_name = _name(layer, "shared_experts.w2.weight")
    w1 = _require(checkpoint.tensor(w1_name), w1_name, torch.int8, (INTERMEDIATE, HIDDEN))
    w3 = _require(checkpoint.tensor(w3_name), w3_name, torch.int8, (INTERMEDIATE, HIDDEN))
    w2 = _require(checkpoint.tensor(w2_name), w2_name, torch.int8, (HIDDEN, INTERMEDIATE))
    _require_zero_offset(
        checkpoint.tensor(_name(layer, "shared_experts.w1.weight_offset")),
        _name(layer, "shared_experts.w1.weight_offset"),
        (INTERMEDIATE, 1),
    )
    _require_zero_offset(
        checkpoint.tensor(_name(layer, "shared_experts.w3.weight_offset")),
        _name(layer, "shared_experts.w3.weight_offset"),
        (INTERMEDIATE, 1),
    )
    _require_zero_offset(
        checkpoint.tensor(_name(layer, "shared_experts.w2.weight_offset")),
        _name(layer, "shared_experts.w2.weight_offset"),
        (HIDDEN, 1),
    )
    w1_scale_name = _name(layer, "shared_experts.w1.weight_scale")
    w3_scale_name = _name(layer, "shared_experts.w3.weight_scale")
    w2_scale_name = _name(layer, "shared_experts.w2.weight_scale")
    return {
        "shared_gate_up": torch.cat((w1.transpose(0, 1), w3.transpose(0, 1)), dim=1).contiguous(),
        "shared_gate_up_scale": torch.cat(
            (
                _scale(checkpoint.tensor(w1_scale_name), w1_scale_name, INTERMEDIATE),
                _scale(checkpoint.tensor(w3_scale_name), w3_scale_name, INTERMEDIATE),
            )
        ).contiguous(),
        "shared_down": w2.transpose(0, 1).contiguous(),
        "shared_down_scale": _scale(checkpoint.tensor(w2_scale_name), w2_scale_name, HIDDEN),
    }


def load_ep2_rank_weights(
    checkpoint: DecodeSafetensorsCheckpoint,
    layer: int,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Materialize all real checkpoint operands required by one EP2 rank."""

    result: dict[str, torch.Tensor] = {}
    result.update(load_router(checkpoint, layer))
    result.update(load_routed_shard(checkpoint, layer, rank))
    result.update(load_shared_expert(checkpoint, layer))
    return result


def abi_summary() -> dict[str, Any]:
    """Return the fixed host-to-PyPTO layout without reading checkpoint data."""

    return {
        "source": "Aurora safetensors W8A8_DYNAMIC",
        "offset_policy": "reject_nonzero",
        "routed": {
            "w1_w3_logical": [INTERMEDIATE, HIDDEN],
            "w1_w3_device": [LOCAL_EXPERTS, HIDDEN // 32, INTERMEDIATE // 16, 16, 32],
            "w2_logical": [HIDDEN, INTERMEDIATE],
            "w2_device": [LOCAL_EXPERTS, INTERMEDIATE_PADDED // 32, HIDDEN // 16, 16, 32],
            "scale_w1_w3": [LOCAL_EXPERTS, INTERMEDIATE],
            "scale_w2": [LOCAL_EXPERTS, HIDDEN],
            "layout": "FRACTAL_NZ for all routed matrices; EP host shape flattens to [EP*LOCAL_EXPERTS, ...]",
        },
        "shared": {
            "gate_up": [HIDDEN, INTERMEDIATE * 2],
            "down": [INTERMEDIATE, HIDDEN],
            "layout": "logical row-major; moe_shared packs at kernel entry",
        },
        "router": {"gate_weight": [EXPERTS, HIDDEN], "bias": [EXPERTS]},
    }


__all__ = [
    "abi_summary",
    "iter_routed_experts",
    "load_ep2_rank_weights",
    "load_router",
    "load_routed_shard",
    "load_shared_expert",
    "pack_routed_expert",
]
