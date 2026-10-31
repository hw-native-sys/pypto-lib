"""Load and validate the prepared A2/A3 W8A8 tensor bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
from contextlib import contextmanager

import torch


def load_tensor_bundle(path: str | Path, required: Mapping[str, tuple[torch.dtype, tuple[int, ...]]] | None = None):
    """Load a ``torch.save`` bundle and validate the A2/A3 tensor contract.

    The prepared layer-0 and layer-2 assets under the checkpoint workspace are
    already converted to contiguous ``[K, N]`` INT8 weights with FP32 channel
    scales.  This loader deliberately refuses silent dtype or shape changes.
    """

    bundle_path = Path(path)
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a tensor dictionary in {bundle_path}, got {type(payload).__name__}")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()):
        raise TypeError(f"all entries in {bundle_path} must be named torch.Tensor values")
    if required is not None:
        missing = sorted(set(required) - set(payload))
        if missing:
            raise KeyError(f"missing tensors in {bundle_path}: {missing}")
        for name, (dtype, shape) in required.items():
            tensor = payload[name]
            if tensor.dtype != dtype or tuple(tensor.shape) != tuple(shape):
                raise ValueError(
                    f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                    f"expected dtype={dtype}, shape={shape}"
                )
    return payload


def load_layer0_qkv(path: str | Path):
    """Load the prepared layer-0 QKV bundle used by the first decoder step."""

    return load_tensor_bundle(
        path,
        {
            "wq_a": (torch.int8, (5120, 1280)),
            "wq_a_scale": (torch.float32, (1280,)),
            "wq_b": (torch.int8, (1280, 32768)),
            "wq_b_scale": (torch.float32, (32768,)),
            "wkv": (torch.int8, (5120, 512)),
            "wkv_scale": (torch.float32, (512,)),
            "q_norm_weight": (torch.bfloat16, (1280,)),
            "kv_norm_weight": (torch.bfloat16, (512,)),
        },
    )


def load_layer0_shared_expert(path: str | Path):
    """Load the prepared layer-0 shared W8A8 expert bundle."""

    return load_tensor_bundle(
        path,
        {
            "gate_up_weight": (torch.int8, (5120, 4608)),
            "gate_up_scale": (torch.float32, (4608,)),
            "down_weight": (torch.int8, (2304, 5120)),
            "down_scale": (torch.float32, (5120,)),
        },
    )


def load_layer0_output(path: str | Path):
    """Load the prepared layer-0 grouped output projection bundle."""

    return load_tensor_bundle(
        path,
        {
            "wo_a": (torch.bfloat16, (8, 4096, 1024)),
            "wo_b": (torch.int8, (8192, 5120)),
            "wo_b_scale": (torch.float32, (5120,)),
        },
    )


def load_layer0_router(path: str | Path):
    """Load router tensors from the converted layer-0 MoE safetensors."""

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - only hit outside pto
        raise RuntimeError("router safetensors require the safetensors package") from exc
    required = {
        "gate.weight": (torch.bfloat16, (384, 5120)),
        "gate.bias": (torch.float32, (384,)),
    }
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        result = {}
        for name, (dtype, shape) in required.items():
            if name not in handle.keys():
                raise KeyError(f"missing tensors in {path}: {name}")
            tensor = handle.get_tensor(name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                    f"expected dtype={dtype}, shape={shape}"
                )
            result[name] = tensor
        if "gate.bias_vl" in handle.keys():
            vision_bias = handle.get_tensor("gate.bias_vl")
            if vision_bias.dtype != torch.float32 or tuple(vision_bias.shape) != (384,):
                raise ValueError(
                    f"gate.bias_vl contract mismatch: got dtype={vision_bias.dtype}, "
                    f"shape={tuple(vision_bias.shape)}; expected dtype={torch.float32}, shape=(384,)"
                )
            result["gate.bias_vl"] = vision_bias
    return result


@contextmanager
def open_layer0_moe(path: str | Path):
    """Open the converted layer-0 MoE and expose a lazy expert provider.

    The routed checkpoint is about 13 GiB.  Keeping the ``safe_open`` handle
    alive lets a validator load only the six tensors for experts selected by
    the router while retaining the exact INT8/FP32 tensors used by A2/A3.
    """

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - only hit outside pto
        raise RuntimeError("MoE safetensors require the safetensors package") from exc

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        required_router = {
            "gate.weight": (torch.bfloat16, (384, 5120)),
            "gate.bias": (torch.float32, (384,)),
        }
        router = {}
        for name, (dtype, shape) in required_router.items():
            if name not in handle.keys():
                raise KeyError(f"missing tensors in {path}: {name}")
            tensor = handle.get_tensor(name)
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                    f"expected dtype={dtype}, shape={shape}"
                )
            router[name] = tensor
        if "gate.bias_vl" in handle.keys():
            vision_bias = handle.get_tensor("gate.bias_vl")
            if vision_bias.dtype != torch.float32 or tuple(vision_bias.shape) != (384,):
                raise ValueError("gate.bias_vl must be FP32[384]")
            router["gate.bias_vl"] = vision_bias

        def expert_provider(expert_id: int) -> dict[str, torch.Tensor]:
            expert = int(expert_id)
            if not 0 <= expert < 384:
                raise IndexError(f"expert id must be in [0, 384), got {expert}")
            names = {
                "w1": f"experts.{expert}.w1.weight",
                "w1_scale": f"experts.{expert}.w1.scale",
                "w2": f"experts.{expert}.w2.weight",
                "w2_scale": f"experts.{expert}.w2.scale",
                "w3": f"experts.{expert}.w3.weight",
                "w3_scale": f"experts.{expert}.w3.scale",
            }
            result = {}
            expected = {
                "w1": (torch.int8, (5120, 2304)),
                "w1_scale": (torch.float32, (2304,)),
                "w2": (torch.int8, (2304, 5120)),
                "w2_scale": (torch.float32, (5120,)),
                "w3": (torch.int8, (5120, 2304)),
                "w3_scale": (torch.float32, (2304,)),
            }
            for key, name in names.items():
                if name not in handle.keys():
                    raise KeyError(f"missing tensors in {path}: {name}")
                tensor = handle.get_tensor(name)
                dtype, shape = expected[key]
                if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                    raise ValueError(
                        f"{name} contract mismatch: got dtype={tensor.dtype}, shape={tuple(tensor.shape)}; "
                        f"expected dtype={dtype}, shape={shape}"
                    )
                result[key] = tensor
            return result

        yield router, expert_provider


__all__ = [
    "load_layer0_output",
    "load_layer0_qkv",
    "load_layer0_router",
    "load_layer0_shared_expert",
    "load_tensor_bundle",
    "open_layer0_moe",
]
