# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash indexed W8A8 checkpoint loading, conversion and per-rank weight preparation."""

import json
import sys
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

from models.glm5_3_flash.config import (
    D,
    DENSE_INTER,
    FLASH,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    SUPPORTED_EP_SIZES,
    SUPPORTED_TP_SIZES,
    TP_SIZE,
)
from models.glm5_3_flash.indexer_cache import sylvester_hadamard
from models.glm5_3_flash.golden import golden_absorb_output
import models.glm5_3_flash.vision.config as V

# This directory owns a ``golden.py`` reference module, so the repository-root
# ``golden`` harness package must come first on the path before any harness import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# INT8 FRACTAL_NZ storage: 16 rows and one 32-byte C0 line per inner block.
_NZ_ROWS = 16
_NZ_INT8_C0 = 32


def vision_weight_layouts() -> dict[str, tuple[int, ...]]:
    """Return the complete released BF16 vision-tower tensor layout."""
    depth, hidden, intermediate = V.VISION_DEPTH, V.VISION_HIDDEN, V.VISION_INTER
    return {
        "patch_weight": (hidden, V.VISION_PATCH_IN), "patch_bias": (hidden,),
        "norm1_weight": (depth, hidden), "qkv_weight": (depth, 3 * hidden, hidden),
        "qkv_bias": (depth, 3 * hidden), "q_norm_weight": (depth, V.VISION_HEAD_DIM),
        "k_norm_weight": (depth, V.VISION_HEAD_DIM), "out_weight": (depth, hidden, hidden),
        "out_bias": (depth, hidden), "norm2_weight": (depth, hidden),
        "gate_weight": (depth, intermediate, hidden), "gate_bias": (depth, intermediate),
        "up_weight": (depth, intermediate, hidden), "up_bias": (depth, intermediate),
        "down_weight": (depth, hidden, intermediate), "down_bias": (depth, hidden),
        "post_norm_weight": (hidden,), "downsample_weight": (D, V.VISION_MERGE ** 2 * hidden),
        "downsample_bias": (D,), "merger_proj_weight": (D, D), "merger_norm_weight": (D,),
        "merger_norm_bias": (D,), "merger_gate_weight": (V.VISION_PROJ_INTER, D),
        "merger_up_weight": (V.VISION_PROJ_INTER, D), "merger_down_weight": (D, V.VISION_PROJ_INTER),
    }


def pack_int8_nz(logical: torch.Tensor) -> torch.Tensor:
    """Pack a CPU INT8 tensor's trailing matrix into NZ order, preserving its shape."""
    if not isinstance(logical, torch.Tensor) or logical.dtype != torch.int8 or logical.device.type != "cpu":
        raise TypeError("NZ shared weights must be CPU INT8 tensors before packing")
    if logical.ndim < 2 or any(extent <= 0 for extent in logical.shape):
        raise ValueError("NZ weights require positive dimensions and a trailing matrix")
    rows, cols = logical.shape[-2:]
    if rows % _NZ_ROWS or cols % _NZ_INT8_C0:
        raise ValueError("INT8 NZ weights require 16-row fractals and 32-element C0 lines")
    blocks = logical.reshape(-1, rows // _NZ_ROWS, _NZ_ROWS, cols // _NZ_INT8_C0, _NZ_INT8_C0)
    return blocks.permute(0, 3, 1, 2, 4).contiguous().reshape(logical.shape)


def unpack_int8_nz(packed: torch.Tensor) -> torch.Tensor:
    """Restore a CPU INT8 NZ tensor's trailing matrix to logical row-major order."""
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.int8 or packed.device.type != "cpu":
        raise TypeError("NZ shared weights must be CPU INT8 tensors before unpacking")
    if packed.ndim < 2 or any(extent <= 0 for extent in packed.shape):
        raise ValueError("NZ weights require positive dimensions and a trailing matrix")
    rows, cols = packed.shape[-2:]
    if rows % _NZ_ROWS or cols % _NZ_INT8_C0:
        raise ValueError("INT8 NZ weights require 16-row fractals and 32-element C0 lines")
    blocks = packed.reshape(-1, cols // _NZ_INT8_C0, rows // _NZ_ROWS, _NZ_ROWS, _NZ_INT8_C0)
    return blocks.permute(0, 2, 3, 1, 4).contiguous().reshape(packed.shape)


# Candidate checkpoint spellings, most specific first. ``{e}`` is the global
# expert id; templates without it are for the single shared/dense FFN.
ROLES: dict[str, tuple[str, ...]] = {
    "routed_gate": ("experts.{e}.gate_proj.weight", "experts.{e}.gate_proj"),
    "routed_up": ("experts.{e}.up_proj.weight", "experts.{e}.up_proj"),
    "routed_down": ("experts.{e}.down_proj.weight", "experts.{e}.down_proj"),
    "routed_fused": ("experts.{e}.w_gate_up", "experts.{e}.w_gate_up.weight"),
    "routed_fused_down": ("experts.{e}.w_down", "experts.{e}.w_down.weight"),
    "shared_gate": ("shared_experts.gate_proj.weight", "shared_experts.gate_proj", "shared_experts.w1.weight"),
    "shared_up": ("shared_experts.up_proj.weight", "shared_experts.up_proj", "shared_experts.w3.weight"),
    "shared_down": ("shared_experts.down_proj.weight", "shared_experts.down_proj", "shared_experts.w2.weight"),
    "shared_fused": ("shared_experts.w_gate_up",),
    "shared_fused_down": ("shared_experts.w_down",),
    "dense_gate": ("gate_proj.weight", "gate_proj"),
    "dense_up": ("up_proj.weight", "up_proj"),
    "dense_down": ("down_proj.weight", "down_proj"),
    "dense_fused": ("w_gate_up",),
    "dense_fused_down": ("w_down",),
    "router_weight": ("gate.weight",),
    "router_bias": ("gate.e_score_correction_bias", "e_score_correction_bias"),
}

SCALE_SUFFIXES = (".weight_scale", "_scale")
OFFSET_SUFFIXES = (".weight_offset", "_offset")


# The released checkpoint nests the text backbone under ``language_model``
# (``model.language_model.layers.*``); converted builds keep the scaffold's
# root spelling. Every lookup accepts both, canonical first, so neither the
# loader nor its fixtures depend on which spelling is on disk.
TEXT_LAYER_ROOTS = ("model.layers.", "model.language_model.layers.")


def layer_prefix(layer_id: int) -> str:
    """Canonical layer prefix inside the text backbone."""
    return f"{TEXT_LAYER_ROOTS[0]}{layer_id}.mlp."


def layer_prefixes(layer_id: int) -> tuple[str, ...]:
    """Every accepted layer prefix, canonical spelling first."""
    return tuple(f"{root}{layer_id}.mlp." for root in TEXT_LAYER_ROOTS)


def _prefixed(prefixes: Sequence[str], templates: Sequence[str], **fmt: object) -> list[str]:
    """All ``prefix + template`` lookups, template priority inside a prefix."""
    return [prefix + template.format(**fmt) for prefix in prefixes for template in templates]


@dataclass(frozen=True)
class QuantDescription:
    """The deployment checkpoint's ``quant_model_description.json``, in part."""

    model_quant_type: str
    group_size: int
    is_rot_used: bool = False

    @classmethod
    def parse(cls, description: Mapping[str, object]) -> "QuantDescription":
        return cls(
            model_quant_type=str(description["model_quant_type"]),
            group_size=int(description["group_size"]),
            is_rot_used=bool(description.get("is_rot_used", False)),
        )

    def require_dynamic_symmetric(self) -> None:
        """The only scheme this loader implements, and the only one shipped."""
        if self.model_quant_type != "W8A8_DYNAMIC":
            raise ValueError(f"expected W8A8_DYNAMIC, got {self.model_quant_type!r}")
        if self.group_size != 0:
            raise ValueError(f"expected group_size 0 (per-channel), got {self.group_size}")


@dataclass(frozen=True)
class FfnShard:
    """One FFN's INT8 weights and dequant scales, full or TP-sliced."""

    gate_up: torch.Tensor        # [2 * inter, D] INT8, gate rows then up rows
    gate_up_scale: torch.Tensor  # [2 * inter] FP32
    down: torch.Tensor           # [D, inter] INT8
    down_scale: torch.Tensor     # [D] FP32


@dataclass(frozen=True)
class RoutedShard:
    """This EP rank's routed experts, one FFN per expert."""

    w_gate_up: torch.Tensor        # [n_local, 2 * MOE_INTER, D] INT8
    w_gate_up_scale: torch.Tensor  # [n_local, 2 * MOE_INTER] FP32
    w_down: torch.Tensor           # [n_local, D, MOE_INTER] INT8
    w_down_scale: torch.Tensor     # [n_local, D] FP32


@dataclass(frozen=True)
class RouterTensors:
    """The router's BF16 weight and FP32 correction bias (replicated, not sharded)."""

    weight: torch.Tensor
    correction_bias: torch.Tensor


def load_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint file or directory into one name -> tensor mapping.

    ``*.safetensors`` is used when the package is available; otherwise a
    directory of ``*.pt`` shards, or a single ``.pt`` mapping, is read. The
    layout contract above is about names and dtypes, so any reader that produces
    the same mapping works.
    """
    root = Path(path)
    if root.is_dir():
        files = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.pt"))
    else:
        files = [root]
    tensors: dict[str, torch.Tensor] = {}
    for file in files:
        if file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - exercised only with real shards
                raise ImportError(
                    "reading .safetensors needs the safetensors package; convert to .pt or install it"
                ) from exc
            tensors.update(load_file(str(file)))
        else:
            payload = torch.load(str(file), map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError(f"{file} is not a tensor mapping")
            tensors.update(payload)
    return tensors


def _find(tensors: Mapping[str, torch.Tensor], candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in tensors:
            return name
    raise KeyError(f"none of {list(candidates)} is in the checkpoint")


def _scale_of(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    for suffix in SCALE_SUFFIXES:
        key = name + suffix
        if key in tensors:
            scale = tensors[key]
            if not scale.is_floating_point() or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"{key} must contain finite positive floating-point scales")
            scale = scale.to(torch.float32)
            # [out, 1] -> [out]; already-1-D stays put.
            return scale.reshape(-1).contiguous()
    raise KeyError(f"{name} has no weight_scale ({list(SCALE_SUFFIXES)})")


def _weight_shape(tensors: Mapping[str, torch.Tensor], name: str) -> tuple[int, ...]:
    return tensors.shape(name) if isinstance(tensors, Checkpoint) else tuple(tensors[name].shape)


def _read_quant(
    tensors: Mapping[str, torch.Tensor], name: str,
    rows: slice = slice(None), cols: slice = slice(None),
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(tensors, Checkpoint):
        return tensors.quant(name, rows, cols)
    weight = tensors[name]
    scale = _scale_of(tensors, name)
    if weight.dtype != torch.int8 or weight.ndim != 2 or scale.numel() != weight.shape[0]:
        raise ValueError(f"{name} must be an INT8 matrix with one scale per output row")
    for suffix in OFFSET_SUFFIXES:
        offset = tensors.get(name + suffix)
        if offset is not None and bool(offset.any()):
            raise ValueError(f"{name}{suffix} must be zero for symmetric W8A8")
    return weight[rows, cols].contiguous(), scale[rows].contiguous()


def fuse_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_scale: torch.Tensor,
    up_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse separate gate/up projections into the kernels' layout.

    The kernels read ``w_gate_up`` with gate rows first and up rows second, and
    one scale vector in the same order.
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shapes differ: {tuple(gate.shape)} vs {tuple(up.shape)}")
    return (
        torch.cat([gate, up], dim=0).contiguous(),
        torch.cat([gate_scale.reshape(-1), up_scale.reshape(-1)]).contiguous(),
    )


def _try_find(tensors: Mapping[str, torch.Tensor], candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in tensors:
            return name
    return None


def _fused_ffn(
    tensors: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
    roles: tuple[str, str, str, str, str],
    *,
    expert: int | None = None,
    intermediate: int | None = None,
    channels: slice = slice(None),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read one FFN's gate/up halves, down projection and scales for selected channels.

    ``roles`` is ``(fused_gate_up, fused_down, gate, up, down)``: a checkpoint
    that stores the fused pair names its down projection ``w_down``, a separate
    one names it ``down_proj``, and either spelling is accepted. ``prefixes``
    carries every accepted layer-root spelling.
    """
    fmt = {"e": expert} if expert is not None else {}
    fused_role, fused_down_role, gate_role, up_role, down_role = roles
    fused_name = _try_find(tensors, _prefixed(prefixes, ROLES[fused_role], **fmt))
    down_name = _find(tensors, _prefixed(prefixes, ROLES[fused_down_role] + ROLES[down_role], **fmt))
    if fused_name is not None:
        shape = _weight_shape(tensors, fused_name)
        if len(shape) != 2:
            raise ValueError(f"{fused_name} must be a matrix, got shape {shape}")
        intermediate = intermediate if intermediate is not None else shape[0] // 2
        if shape != (2 * intermediate, D):
            raise ValueError(f"{fused_name} must have shape {(2 * intermediate, D)}, got {shape}")
        if channels == slice(None):
            gate_up, gate_up_scale = _read_quant(tensors, fused_name)
        else:
            gate, gate_scale = _read_quant(tensors, fused_name, channels)
            up_rows = slice(intermediate + channels.start, intermediate + channels.stop)
            up, up_scale = _read_quant(tensors, fused_name, up_rows)
            gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    else:
        gate_name = _find(tensors, _prefixed(prefixes, ROLES[gate_role], **fmt))
        up_name = _find(tensors, _prefixed(prefixes, ROLES[up_role], **fmt))
        shape = _weight_shape(tensors, gate_name)
        if len(shape) != 2:
            raise ValueError(f"{gate_name} must be a matrix, got shape {shape}")
        intermediate = intermediate if intermediate is not None else shape[0]
        if shape != (intermediate, D) or _weight_shape(tensors, up_name) != shape:
            raise ValueError(f"{gate_name} and {up_name} must have shape {(intermediate, D)}")
        gate, gate_scale = _read_quant(tensors, gate_name, channels)
        up, up_scale = _read_quant(tensors, up_name, channels)
        gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    if _weight_shape(tensors, down_name) != (D, intermediate):
        raise ValueError(f"{down_name} must have shape {(D, intermediate)}")
    down, down_scale = _read_quant(tensors, down_name, cols=channels)
    return gate_up, gate_up_scale, down, down_scale


def shard_routed_experts(
    tensors: Mapping[str, torch.Tensor],
    layer_id: int,
    ep_rank: int,
    ep_size: int,
    *,
    local_experts: int = N_LOCAL_EXPERTS,
    global_experts: int = N_EXPERTS,
) -> RoutedShard:
    """Stack this EP rank's experts into the batched kernel tensors."""
    if ep_size not in SUPPORTED_EP_SIZES:
        raise ValueError(f"ep_size must be one of {SUPPORTED_EP_SIZES}, got {ep_size}")
    if global_experts <= 0 or local_experts <= 0:
        raise ValueError("global_experts and local_experts must be positive")
    if global_experts % ep_size:
        raise ValueError(f"{global_experts} experts cannot be split across EP{ep_size}")
    if local_experts != global_experts // ep_size:
        raise ValueError(f"local_experts {local_experts} != {global_experts} / {ep_size}")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")

    prefixes = layer_prefixes(layer_id)
    first = ep_rank * local_experts
    stacked = []
    for local, expert in enumerate(range(first, first + local_experts)):
        values = _fused_ffn(
            tensors, prefixes, ("routed_fused", "routed_fused_down", "routed_gate", "routed_up", "routed_down"),
            expert=expert, intermediate=FLASH.moe_intermediate_size,
        )
        if local == 0:
            stacked = [torch.empty((local_experts, *value.shape), dtype=value.dtype) for value in values]
        for target, value in zip(stacked, values):
            target[local].copy_(value)
    return RoutedShard(*stacked)


def load_shared_expert(
    tensors: Mapping[str, torch.Tensor],
    layer_id: int,
) -> FfnShard:
    """Load the complete shared expert replicated on every EP rank."""
    prefixes = layer_prefixes(layer_id)
    intermediate = FLASH.moe_intermediate_size
    gate_up, gate_up_scale, down, down_scale = _fused_ffn(
        tensors, prefixes, ("shared_fused", "shared_fused_down", "shared_gate", "shared_up", "shared_down"),
        intermediate=intermediate,
    )
    return FfnShard(gate_up, gate_up_scale, down, down_scale)


def shard_dense_mlp(
    tensors: Mapping[str, torch.Tensor],
    layer_id: int,
    tp_rank: int,
    tp_size: int,
    *,
    intermediate: int = DENSE_INTER,
    local_inter: int | None = None,
) -> FfnShard:
    """TP-slice one dense layer's MLP for one rank (layers 0-2)."""
    if tp_size not in SUPPORTED_TP_SIZES:
        raise ValueError(f"tp_size must be one of {SUPPORTED_TP_SIZES}, got {tp_size}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    if intermediate % tp_size:
        raise ValueError(f"dense intermediate {intermediate} does not divide TP{tp_size}")
    local_inter = local_inter if local_inter is not None else intermediate // tp_size
    if local_inter != intermediate // tp_size:
        raise ValueError(f"local_inter must be {intermediate // tp_size}, got {local_inter}")
    channels = slice(tp_rank * local_inter, (tp_rank + 1) * local_inter)
    prefixes = layer_prefixes(layer_id)
    gate_up, gate_up_scale, down, down_scale = _fused_ffn(
        tensors, prefixes, ("dense_fused", "dense_fused_down", "dense_gate", "dense_up", "dense_down"),
        intermediate=intermediate, channels=channels,
    )
    return FfnShard(gate_up, gate_up_scale, down, down_scale)


def layer_router(tensors: Mapping[str, torch.Tensor], layer_id: int) -> RouterTensors:
    """Read the layer's router: BF16 weights, FP32 correction bias."""
    prefixes = layer_prefixes(layer_id)
    weight_name = _find(tensors, _prefixed(prefixes, ROLES["router_weight"]))
    bias_name = _find(tensors, _prefixed(prefixes, ROLES["router_bias"]))
    if isinstance(tensors, Checkpoint):
        tensors._require_float_metadata(weight_name)
        tensors._require_float_metadata(bias_name)
    weight, bias = tensors[weight_name], tensors[bias_name]
    if weight.dtype not in (torch.bfloat16, torch.float32) or bias.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("router weights and correction bias must be BF16 or FP32")
    if weight.shape != (FLASH.n_routed_experts, D) or bias.shape != (FLASH.n_routed_experts,):
        raise ValueError(f"router must cover all {FLASH.n_routed_experts} experts")
    return RouterTensors(
        weight=weight.to(torch.bfloat16).contiguous(),
        correction_bias=bias.to(torch.float32).contiguous(),
    )


def check_ffn_scales(tensors: Mapping[str, torch.Tensor], names: Sequence[str]) -> None:
    """Every quantized FFN tensor must carry a scale.

    The checkpoint's ``weight_offset`` is ignored (nothing consumes it), but a
    non-zero one would invalidate the symmetric-quantizer assumption, so it is
    checked rather than silently dropped.
    """
    for name in names:
        _scale_of(tensors, name)
        for suffix in OFFSET_SUFFIXES:
            offset = tensors.get(name + suffix)
            if offset is not None and bool(offset.any()):
                raise ValueError(f"{name}{suffix} is not all zero; symmetric dequant assumes it is")


class Checkpoint(Mapping[str, torch.Tensor]):
    """Read indexed W8A8 tensors and prepare the 45-layer text backbone on CPU.

    ``layer`` stages one rank's layer. ``iter_layers`` leaves layer placement and
    lifetime to the caller; ``model`` retains every prepared layer in host memory.
    Shared experts are complete replicas; routed experts use EP and dense FFNs use
    TP. Attention and vocabulary sharding use the explicitly supplied TP geometry.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.root}")
        config = self._json("config.json")
        self.config = config.get("text_config", config)
        self.description = self._json("quant_model_description.json")
        QuantDescription.parse(self.description).require_dynamic_symmetric()
        self._validate_config()
        index_name = "quant_model_weights.safetensors.index.json"
        self.index = self._json(index_name).get("weight_map")
        if not isinstance(self.index, dict) or not self.index:
            raise ValueError(f"{index_name} must contain a nonempty weight_map")
        for key, filename in self.index.items():
            if not isinstance(key, str) or not isinstance(filename, str):
                raise ValueError("checkpoint weight_map must map tensor names to shard filenames")
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
                raise ValueError(f"invalid checkpoint shard path for {key}: {filename}")
        self.used: set[str] = set()
        self.dequantized: set[str] = set()
        self._shapes: dict[str, tuple[int, ...]] = {}

    def _json(self, name: str) -> dict:
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint metadata is missing: {path}")
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint metadata must be a JSON object: {path}")
        return value

    def _validate_config(self) -> None:
        fields = (
            "vocab_size", "hidden_size", "num_hidden_layers", "first_k_dense_replace", "intermediate_size",
            "rms_norm_eps", "swiglu_limit", "tie_word_embeddings",
            "num_attention_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "index_n_heads", "index_head_dim", "index_topk", "index_kpool",
            "index_kpool_compress", "index_kpool_always_select_tail",
            "n_routed_experts", "n_shared_experts", "num_experts_per_tok", "moe_intermediate_size",
            "scoring_func", "topk_method", "norm_topk_prob", "routed_scaling_factor", "n_group", "topk_group",
            "mhc", "hc_mult", "hc_sinkhorn_iters", "hc_eps",
        )
        for name in fields:
            expected = getattr(FLASH, name)
            if self.config.get(name) != expected:
                raise ValueError(f"checkpoint {name} must be {expected!r}, got {self.config.get(name)!r}")
        schedule = [FLASH.layer_config(i).attention_kind.value for i in range(FLASH.num_hidden_layers)]
        if self.config.get("layer_types") != schedule:
            raise ValueError("checkpoint layer_types must match the GLM-5.3-Flash KDA/MLA schedule")
        mlp_schedule = [FLASH.layer_config(i).mlp_kind.value for i in range(FLASH.num_hidden_layers)]
        if "mlp_layer_types" in self.config and self.config["mlp_layer_types"] != mlp_schedule:
            raise ValueError("checkpoint mlp_layer_types must match the GLM-5.3-Flash dense/MoE schedule")
        linear = self.config.get("linear_attn_config", {})
        linear_fields = {
            "num_heads": FLASH.linear_num_heads, "head_dim": FLASH.linear_head_dim,
            "short_conv_kernel_size": FLASH.linear_conv_kernel_dim, "gate_lower_bound": FLASH.linear_lower_bound,
        }
        for name, expected in linear_fields.items():
            if linear.get(name) != expected:
                raise ValueError(f"checkpoint linear_attn_config.{name} must be {expected!r}")

    def __contains__(self, key: object) -> bool:
        return key in self.index

    def __iter__(self) -> Iterator[str]:
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.read(key)

    def _open(self, key: str):
        if key not in self.index:
            raise KeyError(f"tensor is absent from checkpoint index: {key}")
        path = (self.root / self.index[key]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard escapes its directory: {self.index[key]}")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint shard for {key} is missing: {path}")
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("checkpoint loading requires the safetensors package") from exc
        try:
            return safe_open(str(path), framework="pt", device="cpu")
        except OSError:
            # Surface native permission errors hidden by the safetensors reader.
            with path.open("rb"):
                pass
            raise

    def shape(self, key: str) -> tuple[int, ...]:
        """Read a tensor's dimensions from its shard header without loading its payload."""
        if key not in self._shapes:
            with self._open(key) as source:
                self._shapes[key] = tuple(source.get_slice(key).get_shape())
        return self._shapes[key]

    def read(self, key: str, selection=None) -> torch.Tensor:
        """Read only the requested tensor or slice; retain names rather than tensor copies."""
        with self._open(key) as source:
            value = source.get_tensor(key) if selection is None else source.get_slice(key)[selection]
        self.used.add(key)
        return value.contiguous()

    def vision_tensor(self, suffix: str, shape: Sequence[int]) -> torch.Tensor:
        """Load one unquantized vision tensor from either supported checkpoint root."""
        candidates = [root + suffix for root in ("model.visual.", "visual.")]
        names = [name for name in candidates if name in self]
        if len(names) != 1:
            raise ValueError(f"vision tensor must resolve exactly once: {suffix}")
        value = self.read(names[0])
        if tuple(value.shape) != tuple(shape) or value.dtype != torch.bfloat16:
            raise ValueError(f"vision {suffix} must have BF16 shape {tuple(shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"vision tensor contains nonfinite values: {suffix}")
        return value

    def vision_weights(self) -> dict[str, torch.Tensor]:
        """Stage all 24 vision blocks and the spatial merger without quantization."""
        config = self._json("config.json").get("vision_config", {})
        required = {
            "depth": V.VISION_DEPTH, "hidden_size": V.VISION_HIDDEN, "num_heads": V.VISION_HEADS,
            "intermediate_size": V.VISION_INTER, "out_hidden_size": D, "patch_size": V.VISION_PATCH,
            "temporal_patch_size": V.VISION_TEMPORAL_PATCH, "spatial_merge_size": V.VISION_MERGE,
            "projection_intermediate_size": V.VISION_PROJ_INTER, "rms_norm_eps": V.VISION_RMS_NORM_EPS,
            "swiglu_limit": V.VISION_SWIGLU_LIMIT, "attention_bias": True,
        }
        for name, expected in required.items():
            if config.get(name) != expected:
                raise ValueError(f"vision_config.{name} must be {expected!r}")
        blocks = {
            "norm1_weight": "norm1.weight", "qkv_weight": "attn.qkv.weight",
            "qkv_bias": "attn.qkv.bias", "q_norm_weight": "attn.q_norm.weight",
            "k_norm_weight": "attn.k_norm.weight", "out_weight": "attn.proj.weight",
            "out_bias": "attn.proj.bias", "norm2_weight": "norm2.weight",
            "gate_weight": "mlp.gate_proj.weight", "gate_bias": "mlp.gate_proj.bias",
            "up_weight": "mlp.up_proj.weight", "up_bias": "mlp.up_proj.bias",
            "down_weight": "mlp.down_proj.weight", "down_bias": "mlp.down_proj.bias",
        }
        singles = {
            "patch_weight": "patch_embed.proj.weight", "patch_bias": "patch_embed.proj.bias",
            "post_norm_weight": "post_layernorm.weight", "downsample_weight": "downsample.weight",
            "downsample_bias": "downsample.bias", "merger_proj_weight": "merger.proj.weight",
            "merger_norm_weight": "merger.post_projection_norm.weight",
            "merger_norm_bias": "merger.post_projection_norm.bias",
            "merger_gate_weight": "merger.gate_proj.weight", "merger_up_weight": "merger.up_proj.weight",
            "merger_down_weight": "merger.down_proj.weight",
        }
        values = {}
        for name, shape in vision_weight_layouts().items():
            if name in blocks:
                values[name] = torch.stack([
                    self.vision_tensor(f"blocks.{layer}.{blocks[name]}", shape[1:])
                    for layer in range(V.VISION_DEPTH)
                ])
            else:
                source_shape = shape
                if name == "patch_weight":
                    source_shape = (V.VISION_HIDDEN, V.VISION_IN_CHANNELS, V.VISION_TEMPORAL_PATCH,
                                    V.VISION_PATCH, V.VISION_PATCH)
                elif name == "downsample_weight":
                    source_shape = (D, V.VISION_HIDDEN, V.VISION_MERGE, V.VISION_MERGE)
                values[name] = self.vision_tensor(singles[name], source_shape).reshape(shape)
        return values

    def layer_key(self, layer: int, suffix: str) -> str:
        """Resolve a tensor name in a text-backbone layer, excluding the MTP layer."""
        if not 0 <= layer < FLASH.num_hidden_layers:
            raise ValueError(f"layer must be in [0, {FLASH.num_hidden_layers}), got {layer}")
        candidates = [f"{root}{layer}.{suffix}" for root in TEXT_LAYER_ROOTS]
        if suffix.endswith("."):
            for root in TEXT_LAYER_ROOTS:
                if f"{root}{layer}.input_layernorm.weight" in self.index:
                    return f"{root}{layer}.{suffix}"
        return _find(self, candidates)

    def _require_shape(self, key: str, expected: tuple[int, ...]) -> None:
        actual = self.shape(key)
        if actual != expected:
            raise ValueError(f"{key} must have shape {expected}, got {actual}")

    def _require_float_metadata(self, key: str) -> None:
        kind = self.description.get(key)
        if kind != "FLOAT":
            raise ValueError(f"{key} must be declared FLOAT, got {kind!r}")

    def _float(
        self, key: str, shape: tuple[int, ...], selection=None,
        *, dtype: torch.dtype = torch.bfloat16, allow_fp32: bool = False,
    ) -> torch.Tensor:
        self._require_float_metadata(key)
        self._require_shape(key, shape)
        value = self.read(key, selection)
        allowed = (torch.bfloat16, torch.float32) if allow_fp32 else (torch.bfloat16,)
        if value.dtype not in allowed:
            raise ValueError(f"{key} must have dtype {allowed}, got {value.dtype}")
        return value.to(dtype).contiguous()

    def quant(
        self, key: str, rows: slice = slice(None), cols: slice = slice(None),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep INT8 values and FP32 per-output scales; reject asymmetric offsets."""
        if self.description.get(key) != "W8A8_DYNAMIC":
            raise ValueError(f"{key} is not declared W8A8_DYNAMIC in quant_model_description.json")
        shape = self.shape(key)
        if len(shape) != 2:
            raise ValueError(f"quantized weight must be a matrix: {key}, shape {shape}")
        scale_name = _find(self, [key + suffix for suffix in SCALE_SUFFIXES])
        offset_name = _find(self, [key + suffix for suffix in OFFSET_SUFFIXES])
        scale = self.read(scale_name)
        offset = self.read(offset_name)
        allowed_shapes = ((shape[0],), (shape[0], 1))
        if tuple(scale.shape) not in allowed_shapes or tuple(offset.shape) not in allowed_shapes:
            raise ValueError(f"{key} needs one scale and offset per output row")
        if not scale.is_floating_point() or not offset.is_floating_point():
            raise ValueError(f"{key} scales and offsets must be floating point")
        if offset.any() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"{key} has nonzero offsets or nonpositive/nonfinite scales")
        weight = self.read(key, (rows, cols))
        scale = scale.reshape(-1)[rows].float().contiguous()
        if weight.dtype != torch.int8 or weight.ndim != 2 or scale.numel() != weight.shape[0]:
            raise ValueError(f"{key} must be INT8 with one scale per output row")
        return weight, scale

    def matrix(
        self, key: str, rows: slice = slice(None), cols: slice = slice(None), *, mode: str = "native",
    ) -> dict[str, torch.Tensor]:
        """Preserve native INT8/scales or explicitly dequantize a matrix into BF16."""
        if mode not in ("native", "bf16"):
            raise ValueError("matrix mode must be native or bf16")
        kind = self.description.get(key)
        if kind == "W8A8_DYNAMIC":
            weight, scale = self.quant(key, rows, cols)
            if mode == "native":
                return {"weight": weight, "scale": scale}
            self.dequantized.add(key)
            return {"weight": (weight.float() * scale[:, None]).bfloat16()}
        if kind != "FLOAT":
            raise ValueError(f"unsupported or missing quantization description for {key}: {kind!r}")
        weight = self.read(key, (rows, cols))
        if weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError(f"{key} must be a BF16 matrix, got {weight.dtype} {tuple(weight.shape)}")
        return {"weight": weight}

    @staticmethod
    def _partition(size: int, rank: int, world: int) -> slice:
        if world not in SUPPORTED_TP_SIZES or not 0 <= rank < world or size % world:
            raise ValueError(f"cannot partition {size} over rank {rank}/{world}")
        width = size // world
        return slice(rank * width, (rank + 1) * width)

    @staticmethod
    def _ffn_dict(shard: FfnShard) -> dict[str, torch.Tensor]:
        return {
            "w_gate_up": shard.gate_up, "w_gate_up_scale": shard.gate_up_scale,
            "w_down": shard.down, "w_down_scale": shard.down_scale,
        }

    def ffn(
        self, layer: int, *, tp_rank: int = 0, tp: int = 16, ep_rank: int = 0, ep: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Load dense TP channels or complete EP experts with a replicated shared FFN."""
        norm = self._float(self.layer_key(layer, "post_attention_layernorm.weight"), (D,))
        if layer < FLASH.first_k_dense_replace:
            return {"norm_weight": norm, **self._ffn_dict(shard_dense_mlp(self, layer, tp_rank, tp))}
        experts = self._partition(FLASH.n_routed_experts, ep_rank, ep)
        routed = shard_routed_experts(
            self, layer, ep_rank, ep,
            local_experts=experts.stop - experts.start, global_experts=FLASH.n_routed_experts,
        )
        shared = self._ffn_dict(load_shared_expert(self, layer))
        router = layer_router(self, layer)
        return {
            "norm_weight": norm, "gate_weight": router.weight, "correction_bias": router.correction_bias,
            "w_gate_up": routed.w_gate_up, "w_gate_up_scale": routed.w_gate_up_scale,
            "w_down": routed.w_down, "w_down_scale": routed.w_down_scale,
            **{"shared_" + name: value for name, value in shared.items()},
        }

    def mhc(self, layer: int, side: str) -> dict[str, torch.Tensor]:
        """Load the attention or FFN hyper-connection function, scales and base."""
        if side not in ("attn", "ffn"):
            raise ValueError("side must be attn or ffn")
        layout = (
            ("function", "fn", (FLASH.mix_hc, FLASH.hc_dim), torch.bfloat16),
            ("scale", "scale", (3,), torch.float32),
            ("base", "base", (FLASH.mix_hc,), torch.float32),
        )
        out = {}
        for name, suffix, shape, dtype in layout:
            key = self.layer_key(layer, f"{side}_hc.{suffix}")
            out[name] = self._float(key, shape, dtype=dtype, allow_fp32=True)
        return out

    def attention(
        self, layer: int, *, tp_rank: int = 0, tp: int = 16, mla_mode: str = "bf16",
    ) -> dict[str, torch.Tensor]:
        """Load KDA weights or MLA weights with explicit BF16/native quantization mode."""
        if mla_mode not in ("native", "bf16"):
            raise ValueError("mla_mode must be native or bf16")
        prefix = self.layer_key(layer, "self_attn.")
        if layer in FLASH.kda_layer_ids:
            heads = self._partition(FLASH.linear_num_heads, tp_rank, tp)
            width = FLASH.linear_head_dim
            full_width = FLASH.linear_num_heads * width
            rows = slice(heads.start * width, heads.stop * width)
            out = {}
            for name in ("q", "k", "v"):
                key = prefix + name + "_proj.weight"
                out["w_" + name] = self._float(key, (full_width, D), (rows, slice(None)))
            layout = (
                ("w_f_a", "forget_gate.f_a_proj.weight", (width, D), None, torch.bfloat16),
                ("w_f_b", "forget_gate.f_b_proj.weight", (full_width, width), (rows, slice(None)), torch.bfloat16),
                ("dt_bias", "forget_gate.dt_bias", (full_width,), rows, torch.float32),
                ("a_log", "forget_gate.A_log", (FLASH.linear_num_heads,), heads, torch.float32),
                ("w_b", "b_proj.weight", (FLASH.linear_num_heads, D), (heads, slice(None)), torch.bfloat16),
                ("w_g_a", "g_a_proj.weight", (width, D), None, torch.bfloat16),
                ("w_g_b", "g_b_proj.weight", (full_width, width), (rows, slice(None)), torch.bfloat16),
                ("norm_weight", "o_norm.weight", (width,), None, torch.bfloat16),
                ("w_o", "o_proj.weight", (D, full_width), (slice(None), rows), torch.bfloat16),
            )
            for target, name, shape, selection, dtype in layout:
                allow_fp32 = dtype == torch.float32
                out[target] = self._float(prefix + name, shape, selection, dtype=dtype, allow_fp32=allow_fp32)
            conv_shape = (full_width, 1, FLASH.linear_conv_kernel_dim)
            conv = [self._float(prefix + q + "_conv1d.weight", conv_shape, (rows, 0)) for q in ("q", "k", "v")]
            out["weight"] = torch.cat(conv).T.contiguous()
            return out
        heads = self._partition(FLASH.num_attention_heads, tp_rank, tp)
        qdim, vdim, latent = FLASH.qk_nope_head_dim, FLASH.v_head_dim, FLASH.kv_lora_rank
        q_rows = slice(heads.start * qdim, heads.stop * qdim)
        o_cols = slice(heads.start * vdim, heads.stop * vdim)
        projections = (
            ("w_q_a", "q_a_proj", (FLASH.q_lora_rank, D), slice(None), slice(None)),
            ("w_q_b", "q_b_proj", (FLASH.num_attention_heads * qdim, FLASH.q_lora_rank), q_rows, slice(None)),
            ("w_kv_a", "kv_a_proj_with_mqa", (latent, D), slice(None), slice(None)),
            ("w_o", "o_proj", (D, FLASH.num_attention_heads * vdim), slice(None), o_cols),
        )
        out = {}
        for target, source, shape, rows, cols in projections:
            key = prefix + source + ".weight"
            self._require_shape(key, shape)
            pair = self.matrix(key, rows, cols, mode=mla_mode)
            out[target] = pair["weight"]
            if "scale" in pair:
                out[target + "_scale"] = pair["scale"]
        kv_rows = slice(heads.start * (qdim + vdim), heads.stop * (qdim + vdim))
        kv_shape = (FLASH.num_attention_heads * (qdim + vdim), latent)
        kv = self._float(prefix + "kv_b_proj.weight", kv_shape, (kv_rows, slice(None)))
        kv = kv.reshape(heads.stop - heads.start, qdim + vdim, latent)
        out.update(w_k=kv[:, :qdim].contiguous(), w_v=kv[:, qdim:].contiguous())
        out["q_a_norm_weight"] = self._float(prefix + "q_a_layernorm.weight", (FLASH.q_lora_rank,))
        out["kv_a_norm_weight"] = self._float(prefix + "kv_a_layernorm.weight", (latent,))
        if mla_mode == "bf16":
            out["w_o_absorbed"] = golden_absorb_output(out["w_v"], out["w_o"])
        return out

    def indexer(self, layer: int) -> dict[str, torch.Tensor]:
        """Load replicated Indexer projections, normalization and pooling parameters."""
        prefix = self.layer_key(layer, "self_attn.indexer.")
        if layer not in FLASH.dsa_layer_ids:
            raise ValueError(f"layer {layer} is a KDA layer and has no Indexer")
        ih, width = FLASH.index_n_heads, FLASH.index_head_dim
        projections = (
            ("w_q_b", "wq_b.weight", (ih * width, FLASH.q_lora_rank)),
            ("w_k", "wk.weight", (width, D)),
            ("w_weights", "weights_proj.weight", (ih, D)),
            ("w_compress_gate", "index_kpool_compress_gate", (width, D)),
        )
        out = {name: self._float(prefix + key, shape).T.contiguous() for name, key, shape in projections}
        vectors = (
            ("k_norm_weight", "k_norm.weight", (width,)),
            ("k_norm_bias", "k_norm.bias", (width,)),
            ("compress_ape", "index_kpool_compress_ape", (FLASH.index_kpool, width)),
        )
        out.update({name: self._float(prefix + key, shape) for name, key, shape in vectors})
        out["hadamard"] = sylvester_hadamard(width)
        return out

    def layer(
        self, layer: int, *, tp_rank: int = 0, tp: int = 16, ep_rank: int = 0, ep: int = 16,
        mla_mode: str = "bf16",
    ) -> dict:
        """Prepare one rank's complete backbone layer for prefill and decode."""
        self._partition(FLASH.num_attention_heads, tp_rank, tp)
        self._partition(FLASH.n_routed_experts, ep_rank, ep)
        if ep % tp:
            raise ValueError(f"complete-layer EP size {ep} must be divisible by TP size {tp}")
        out = {
            "input_norm": self._float(self.layer_key(layer, "input_layernorm.weight"), (D,)),
            "attn_hc": self.mhc(layer, "attn"), "ffn_hc": self.mhc(layer, "ffn"),
            "attention": self.attention(layer, tp_rank=tp_rank, tp=tp, mla_mode=mla_mode),
            "ffn": self.ffn(layer, tp_rank=tp_rank, tp=tp, ep_rank=ep_rank, ep=ep),
        }
        if layer in FLASH.dsa_layer_ids:
            out["indexer"] = self.indexer(layer)
        return out

    def _vocab_key(self, suffix: str) -> str:
        return _find(self, [f"{root.removesuffix('layers.')}{suffix}" for root in TEXT_LAYER_ROOTS])

    def embedding(self, token_ids) -> torch.Tensor:
        """Look up selected token rows without materializing the complete embedding table."""
        ids = torch.as_tensor(token_ids)
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token ids must be integers")
        ids = ids.flatten().tolist()
        if not ids or min(ids) < 0 or max(ids) >= FLASH.vocab_size:
            raise ValueError("token ids must be nonempty and within the vocabulary")
        key = self._vocab_key("embed_tokens.weight")
        shape = (FLASH.vocab_size, D)
        unique = {i: self._float(key, shape, (slice(i, i + 1), slice(None))) for i in set(ids)}
        return torch.cat([unique[i] for i in ids])

    def vocabulary(self, tp_rank: int = 0, tp: int = 16) -> dict:
        """Prepare vocabulary TP shards and the complete final RMSNorm vector."""
        rows = self._partition(FLASH.vocab_size, tp_rank, tp)
        shape = (FLASH.vocab_size, D)
        return {
            "vocab_start": rows.start, "vocab_stop": rows.stop,
            "embedding": self._float(self._vocab_key("embed_tokens.weight"), shape, (rows, slice(None))),
            "lm_head": self._float("lm_head.weight", shape, (rows, slice(None))),
            "final_norm": self._float(self._vocab_key("norm.weight"), (D,)),
        }

    def rotation(self) -> torch.Tensor:
        """Expose the saved QuaRot artifact without applying an additional rotation."""
        return self.read("rot.weight")

    def iter_layers(
        self, *, tp_rank: int, tp: int, ep_rank: int, ep: int, mla_mode: str = "bf16",
    ) -> Iterator[tuple[int, dict]]:
        """Yield backbone layers in order; the caller owns placement and tensor lifetime."""
        for layer_id in range(FLASH.num_hidden_layers):
            yield layer_id, self.layer(layer_id, tp_rank=tp_rank, tp=tp, ep_rank=ep_rank, ep=ep, mla_mode=mla_mode)

    def model(self, *, tp_rank: int, tp: int, ep_rank: int, ep: int, mla_mode: str = "bf16") -> dict:
        """Materialize all 45 layers and vocabulary weights for one rank in CPU memory."""
        layers = dict(self.iter_layers(tp_rank=tp_rank, tp=tp, ep_rank=ep_rank, ep=ep, mla_mode=mla_mode))
        return {"layers": layers, **self.vocabulary(tp_rank, tp)}




# -----------------------------------------------------------------------------------------------------------
# Resident prefill staging
# -----------------------------------------------------------------------------------------------------------
"""Layer-stacked device residency for the prefill composition, on the loader above.

:class:`Checkpoint` (and any fixture implementing the same narrow read protocol)
supplies the raw tensors; this section turns them into the exact stacked layout
the composed prefill programs consume: one device tensor per (family, kernel
name) with the family's layers on a leading axis, because every GLM kernel keeps
its single-layer ABI and slices by family ordinal.

The staging geometry is parameterized by :class:`~models.glm5_3_flash.config.Glm53FlashConfig`
so the CPU self-test can run a scaled fixture; at the deployment geometry the
FFN assembly below is the same ``_read_quant``/``fuse_gate_up`` pipeline
:func:`shard_dense_mlp`, :func:`shard_routed_experts` and :func:`load_shared_expert`
run, carried over the same checkpoint view.
"""

import math

from models.glm5_3_flash.config import KV_LORA, QK_DIM, V_DIM, Glm53FlashConfig

# The staging side of the checkpoint read protocol: ``name in checkpoint``,
# ``checkpoint.description`` (per-tensor quantization labels), ``checkpoint.read(name,
# selection=None)`` and the Mapping interface the module-level FFN helpers already
# consume. :class:`Checkpoint` implements it natively; the CPU self-test and the
# CI fixtures provide in-memory stand-ins.


@dataclass(frozen=True)
class _StackMember:
    """One kernel-side stack member: checkpoint spellings, view dtype and shard."""

    member: str
    suffixes: tuple[str, ...]
    dtype: torch.dtype
    transform: str = "identity"  # identity | transpose | split_k | split_v
    shard: str = "replicate"     # replicate | rows | columns


_FAMILY_PREFIX = {
    "kda": "self_attn.",
    "mla": "self_attn.",
    "indexer": "self_attn.indexer.",
    "mhc": "",
    "norm": "",
    "router": "mlp.",
}

_KDA_MEMBERS = (
    _StackMember("w_q", ("q_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_k", ("k_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_v", ("v_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_q", ("q_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_k", ("k_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_v", ("v_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_f_a", ("forget_gate.f_a_proj.weight", "f_a_proj.weight"), torch.bfloat16),
    _StackMember("w_f_b", ("forget_gate.f_b_proj.weight", "f_b_proj.weight"), torch.bfloat16, shard="rows"),
    _StackMember("w_g_a", ("g_a_proj.weight",), torch.bfloat16),
    _StackMember("w_g_b", ("g_b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_b", ("b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("dt_bias", ("forget_gate.dt_bias", "dt_bias"), torch.float32, shard="rows"),
    _StackMember("a_log", ("forget_gate.A_log", "A_log"), torch.float32, shard="rows"),
    _StackMember("o_norm", ("o_norm.weight",), torch.bfloat16),
    _StackMember("w_o", ("o_proj.weight",), torch.bfloat16, shard="columns"),
)

_MLA_MEMBERS = (
    _StackMember("w_q_a", ("q_a_proj.weight",), torch.bfloat16),
    _StackMember("q_a_norm", ("q_a_layernorm.weight",), torch.bfloat16),
    _StackMember("w_q_b", ("q_b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_kv_a", ("kv_a_proj_with_mqa.weight",), torch.bfloat16),
    _StackMember("kv_a_norm", ("kv_a_layernorm.weight",), torch.bfloat16),
    _StackMember("w_k", ("kv_b_proj.weight",), torch.bfloat16, transform="split_k"),
    _StackMember("w_v", ("kv_b_proj.weight",), torch.bfloat16, transform="split_v"),
    _StackMember("w_o", ("o_proj.weight",), torch.bfloat16, shard="columns"),
)

_INDEXER_MEMBERS = (
    _StackMember("w_q_b", ("wq_b.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("w_k", ("wk.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("k_norm_w", ("k_norm.weight",), torch.bfloat16),
    _StackMember("k_norm_b", ("k_norm.bias",), torch.bfloat16),
    _StackMember("w_weights", ("weights_proj.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("w_compress_gate", ("index_kpool_compress_gate",), torch.bfloat16, transform="transpose"),
    _StackMember("compress_ape", ("index_kpool_compress_ape",), torch.bfloat16),
)

_MHC_MEMBERS = (
    _StackMember("attn_fn", ("attn_hc.fn", "hc_attn_fn"), torch.bfloat16),
    _StackMember("attn_scale", ("attn_hc.scale", "hc_attn_scale"), torch.float32),
    _StackMember("attn_base", ("attn_hc.base", "hc_attn_base"), torch.float32),
    _StackMember("ffn_fn", ("ffn_hc.fn", "hc_ffn_fn"), torch.bfloat16),
    _StackMember("ffn_scale", ("ffn_hc.scale", "hc_ffn_scale"), torch.float32),
    _StackMember("ffn_base", ("ffn_hc.base", "hc_ffn_base"), torch.float32),
)

_NORM_MEMBERS = (
    _StackMember("input_norm", ("input_layernorm.weight",), torch.bfloat16),
    _StackMember("post_norm", ("post_attention_layernorm.weight",), torch.bfloat16),
)

_ROUTER_MEMBERS = (
    _StackMember("gate_w", ("gate.weight",), torch.bfloat16),
    _StackMember("gate_bias", ("gate.e_score_correction_bias", "e_score_correction_bias"), torch.float32),
)

_FAMILY_MEMBERS = {
    "kda": _KDA_MEMBERS,
    "mla": _MLA_MEMBERS,
    "indexer": _INDEXER_MEMBERS,
    "mhc": _MHC_MEMBERS,
    "norm": _NORM_MEMBERS,
    "router": _ROUTER_MEMBERS,
}

# Top-level table spellings: the deployment nests the text backbone under
# ``model.language_model.``; converted builds keep the scaffold's root.
_TEXT_MODEL_ROOTS = ("model.language_model.", "model.")


def _source_shape(model: Glm53FlashConfig, family: str, member: str) -> tuple[int, ...]:
    """The checkpoint-side shape a member is read from (before any transform)."""
    hidden = model.hidden_size
    qkv = model.linear_num_heads * model.linear_head_dim
    if family == "kda":
        shapes = {
            "w_q": (qkv, hidden), "w_k": (qkv, hidden), "w_v": (qkv, hidden),
            "conv_q": (qkv, 1, model.linear_conv_kernel_dim),
            "conv_k": (qkv, 1, model.linear_conv_kernel_dim),
            "conv_v": (qkv, 1, model.linear_conv_kernel_dim),
            "w_f_a": (model.linear_head_dim, hidden), "w_f_b": (qkv, model.linear_head_dim),
            "w_g_a": (model.linear_head_dim, hidden), "w_g_b": (qkv, model.linear_head_dim),
            "w_b": (model.linear_num_heads, hidden), "dt_bias": (qkv,),
            "a_log": (model.linear_num_heads,), "o_norm": (model.linear_head_dim,),
            "w_o": (hidden, qkv),
        }
    elif family == "mla":
        shapes = {
            "w_q_a": (model.q_lora_rank, hidden), "q_a_norm": (model.q_lora_rank,),
            "w_q_b": (model.num_attention_heads * model.qk_head_dim, model.q_lora_rank),
            "w_kv_a": (model.kv_lora_rank, hidden), "kv_a_norm": (model.kv_lora_rank,),
            "w_k": (model.num_attention_heads * (model.qk_head_dim + model.v_head_dim), model.kv_lora_rank),
            "w_v": (model.num_attention_heads * (model.qk_head_dim + model.v_head_dim), model.kv_lora_rank),
            "w_o": (hidden, model.num_attention_heads * model.v_head_dim),
        }
    elif family == "indexer":
        shapes = {
            "w_q_b": (model.index_n_heads * model.index_head_dim, model.q_lora_rank),
            "w_k": (model.index_head_dim, hidden),
            "k_norm_w": (model.index_head_dim,), "k_norm_b": (model.index_head_dim,),
            "w_weights": (model.index_n_heads, hidden),
            "w_compress_gate": (model.index_head_dim, hidden),
            "compress_ape": (model.index_kpool, model.index_head_dim),
        }
    elif family == "mhc":
        shapes = {
            "attn_fn": (model.mix_hc, model.hc_dim), "attn_scale": (3,), "attn_base": (model.mix_hc,),
            "ffn_fn": (model.mix_hc, model.hc_dim), "ffn_scale": (3,), "ffn_base": (model.mix_hc,),
        }
    elif family == "norm":
        shapes = {"input_norm": (hidden,), "post_norm": (hidden,)}
    else:
        shapes = {"gate_w": (model.n_routed_experts, hidden), "gate_bias": (model.n_routed_experts,)}
    return shapes[member]


@dataclass(frozen=True)
class StackPlan:
    """One device-resident stacked tensor as the runner must allocate it."""

    name: str                 # e.g. "kda.w_q"
    family: str               # kda | mla | indexer | mhc | norm | router | moe | dense | mtp | single
    layers: tuple[int, ...]   # source layer ids on the stack axis, ascending
    shape: tuple[int, ...]    # per-rank stacked shape (leading axis = family ordinal)
    dtype: torch.dtype

    @property
    def bytes(self) -> int:
        count = math.prod(self.shape)
        return count * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class RankPlan:
    """Pure-metadata view of one rank's resident weights (no payload reads)."""

    rank: int
    tp_rank: int
    ep_rank: int
    local_expert_ids: tuple[int, ...]
    stacks: tuple[StackPlan, ...]
    device_bytes: int
    host_tables: tuple[str, ...]  # ("embed_tokens", "lm_head"), host-side aliases included


def split_kv_b(
    w_kv_b: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    *,
    qk_dim: int = QK_DIM,
    v_dim: int = V_DIM,
    kv_lora: int = KV_LORA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into its key/value halves and keep one TP rank's heads.

    Mirrors ``golden_split_kv_b`` (``golden.py``): the checkpoint stores
    ``[H * (QK_DIM + V_DIM), KV_LORA]``; the halves come back per head as
    ``[LOCAL_H, QK_DIM, KV_LORA]`` and ``[LOCAL_H, V_DIM, KV_LORA]`` — reduction
    dim first, the layout ``absorb_query`` consumes without a transpose. The
    dims default to the deployment geometry and are overridable for scaled
    fixtures.
    """
    heads = w_kv_b.shape[0] // (qk_dim + v_dim) if w_kv_b.shape[0] % (qk_dim + v_dim) == 0 else -1
    if w_kv_b.dim() != 2 or heads <= 0:
        raise ValueError(f"kv_b_proj must be 2-D with rows a multiple of QK+V, got {tuple(w_kv_b.shape)}")
    local_h = heads // tp_size
    reshaped = w_kv_b.reshape(heads, qk_dim + v_dim, kv_lora)
    key, value = reshaped[:, :qk_dim], reshaped[:, qk_dim:]
    start = tp_rank * local_h
    return key[start : start + local_h].contiguous(), value[start : start + local_h].contiguous()


def gather_rows(checkpoint, name: str, rows: torch.Tensor) -> torch.Tensor:
    """Read selected rows of one tensor through the checkpoint read protocol.

    ``Checkpoint.read`` selects through the shard's slice view, which takes
    slices rather than an index tensor, so the rows are read one at a time and
    concatenated (deduplicated, in ``token_ids`` order). Fixture stand-ins may
    gather directly.
    """
    ids = torch.as_tensor(rows, device="cpu")
    flat = ids.reshape(-1)
    gather = getattr(checkpoint, "gather", None)
    if gather is not None:
        gathered = gather(name, flat)
    else:
        unique = {
            int(row): checkpoint.read(name, slice(row, row + 1))
            for row in sorted({int(r) for r in flat.tolist()})
        }
        gathered = torch.cat([unique[int(row)] for row in flat.tolist()], dim=0)
    return gathered.reshape(*ids.shape, *gathered.shape[1:])


def _fused_ffn_at(
    tensors: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
    roles: tuple[str, str, str, str, str],
    *,
    expert: int | None,
    intermediate: int,
    hidden: int,
    channels: slice = slice(None),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_fused_ffn`` carried over an explicit geometry: hidden and intermediate
    come from the staging model instead of the module constants, so scaled
    fixtures load through the same read/fuse pipeline as the deployment."""
    fmt = {"e": expert} if expert is not None else {}
    fused_role, fused_down_role, gate_role, up_role, down_role = roles
    fused_name = _try_find(tensors, _prefixed(prefixes, ROLES[fused_role], **fmt))
    down_name = _find(tensors, _prefixed(prefixes, ROLES[fused_down_role] + ROLES[down_role], **fmt))
    if fused_name is not None:
        if _weight_shape(tensors, fused_name) != (2 * intermediate, hidden):
            raise ValueError(f"{fused_name} must have shape {(2 * intermediate, hidden)}")
        if channels == slice(None):
            gate_up, gate_up_scale = _read_quant(tensors, fused_name)
        else:
            gate, gate_scale = _read_quant(tensors, fused_name, channels)
            up_rows = slice(intermediate + channels.start, intermediate + channels.stop)
            up, up_scale = _read_quant(tensors, fused_name, up_rows)
            gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    else:
        gate_name = _find(tensors, _prefixed(prefixes, ROLES[gate_role], **fmt))
        up_name = _find(tensors, _prefixed(prefixes, ROLES[up_role], **fmt))
        if _weight_shape(tensors, gate_name) != (intermediate, hidden) or _weight_shape(tensors, up_name) != (
            intermediate, hidden,
        ):
            raise ValueError(f"{gate_name} and {up_name} must have shape {(intermediate, hidden)}")
        gate, gate_scale = _read_quant(tensors, gate_name, channels)
        up, up_scale = _read_quant(tensors, up_name, channels)
        gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    if _weight_shape(tensors, down_name) != (hidden, intermediate):
        raise ValueError(f"{down_name} must have shape {(hidden, intermediate)}")
    down, down_scale = _read_quant(tensors, down_name, cols=channels)
    return gate_up, gate_up_scale, down, down_scale


def _ffn_roles(kind: str) -> tuple[str, str, str, str, str]:
    return {
        "routed": ("routed_fused", "routed_fused_down", "routed_gate", "routed_up", "routed_down"),
        "shared": ("shared_fused", "shared_fused_down", "shared_gate", "shared_up", "shared_down"),
        "dense": ("dense_fused", "dense_fused_down", "dense_gate", "dense_up", "dense_down"),
    }[kind]


class ResidentPrefillWeights:
    """Plan and stage the resident prefill weights for the TP/EP ranks.

    Construction goes through :meth:`load`, which reads headers only: it
    resolves every stack member's source name (both layer-root spellings, fused
    or separate gate/up, nested or flat mHC/KDA-gate spellings), validates the
    header shapes against the geometry, and builds the full stack plan. Payloads
    move only in :meth:`upload`, one staged layer-slice at a time, so host
    staging never exceeds a single layer of one family.

    Rank mapping: ranks are EP ranks; the TP rank is ``rank % tp_size``. In the
    deployment topology (TP16/EP16/DP1) the two coincide with the rank id, and
    TP1/EP16 collapses to tp_rank 0 for every rank.
    """

    def __init__(self, checkpoint, *, tp_size: int, ep_size: int, model: Glm53FlashConfig | None = None, mla_mode: str = "bf16"):
        model = FLASH if model is None else model
        if isinstance(checkpoint, Checkpoint) and model is not FLASH:
            raise ValueError("Checkpoint validates the deployment geometry; pass model only for fixtures")
        if tp_size not in SUPPORTED_TP_SIZES:
            raise ValueError(f"tp_size must be one of {SUPPORTED_TP_SIZES}, got {tp_size}")
        if ep_size not in SUPPORTED_EP_SIZES:
            raise ValueError(f"ep_size must be one of {SUPPORTED_EP_SIZES}, got {ep_size}")
        if ep_size % tp_size:
            raise ValueError(f"EP{ep_size} must be divisible by TP{tp_size}")
        if model.num_attention_heads % tp_size:
            raise ValueError(f"{model.num_attention_heads} MLA heads do not divide TP{tp_size}")
        if model.linear_num_heads % tp_size:
            raise ValueError(f"{model.linear_num_heads} KDA heads do not divide TP{tp_size}")
        if model.intermediate_size % tp_size:
            raise ValueError(f"{model.intermediate_size} dense channels do not divide TP{tp_size}")
        if model.n_routed_experts % ep_size:
            raise ValueError(f"{model.n_routed_experts} routed experts do not divide EP{ep_size}")
        if mla_mode not in ("bf16", "native"):
            raise ValueError("mla_mode must be bf16 or native")
        self.mla_mode = mla_mode
        self._family_members = dict(_FAMILY_MEMBERS)
        if mla_mode == "native":
            native = []
            for spec in _MLA_MEMBERS:
                if spec.member in ("w_q_a", "w_q_b", "w_kv_a", "w_o"):
                    native.append(dataclasses.replace(spec, dtype=torch.int8))
                    native.append(dataclasses.replace(spec, member=spec.member + "_scale", dtype=torch.float32))
                else:
                    native.append(spec)
            self._family_members["mla"] = tuple(native)
        self.checkpoint = checkpoint
        self.model = model
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.ranks = ep_size
        self.local_h = model.num_attention_heads // tp_size
        self.local_kda_h = model.linear_num_heads // tp_size
        self.local_kda_qkv = model.linear_num_heads * model.linear_head_dim // tp_size
        self.n_local_experts = model.n_routed_experts // ep_size
        self.local_dense_inter = model.intermediate_size // tp_size
        mtp = model.mtp_layer_id
        self._family_layers = {
            "kda": tuple(model.kda_layer_ids),
            "mla": tuple(model.indexer_layer_ids),
            "indexer": tuple(model.indexer_layer_ids),
            "mhc": tuple(range(model.num_hidden_layers)),
            "norm": tuple(range(model.num_checkpoint_layers)),
            "router": tuple(range(model.first_k_dense_replace, model.num_checkpoint_layers)),
            "moe": tuple(range(model.first_k_dense_replace, model.num_checkpoint_layers)),
            "dense": tuple(range(model.first_k_dense_replace)),
        }
        self._mtp_layer = mtp
        self._lm_head_cache: torch.Tensor | None = None
        self._source: dict[tuple[str, int], str] = {}
        self.stacks: dict[str, StackPlan] = {}

    # -- construction ----------------------------------------------------------------

    @classmethod
    def load(
        cls, checkpoint, *, tp_size: int = 16, ep_size: int = 16, model: Glm53FlashConfig | None = None,
        mla_mode: str = "bf16",
    ) -> "ResidentPrefillWeights":
        """Build the full stack plan from checkpoint headers, reading no payloads."""
        self = cls(checkpoint, tp_size=tp_size, ep_size=ep_size, model=model, mla_mode=mla_mode)
        self._resolve_sources()
        self._build_stacks()
        return self

    def _resolve(self, candidates: Sequence[str], what: str) -> str:
        for name in candidates:
            if name in self.checkpoint:
                return name
        raise ValueError(f"checkpoint is missing {what} (tried {list(candidates)})")

    def _label(self, name: str) -> str:
        description = getattr(self.checkpoint, "description", None)
        label = None if description is None else description.get(name)
        if label not in ("FLOAT", "W8A8_DYNAMIC"):
            raise ValueError(f"{name} has unsupported quantization label {label!r}")
        return label

    def _resolve_sources(self) -> None:
        """Resolve and header-validate every stack member's source tensor."""
        model = self.model
        for family, members in self._family_members.items():
            prefix = _FAMILY_PREFIX[family]
            for layer in self._family_layers[family]:
                for spec in members:
                    candidates = [
                        root + f"{layer}." + prefix + suffix
                        for root in TEXT_LAYER_ROOTS
                        for suffix in spec.suffixes
                    ]
                    name = self._resolve(candidates, f"{family}.{spec.member} on layer {layer}")
                    self._source[(f"{family}.{spec.member}", layer)] = name
                    member = spec.member.removesuffix("_scale") if family == "mla" else spec.member
                    native = family == "mla" and self.mla_mode == "native" and member in ("w_q_a", "w_q_b", "w_kv_a", "w_o")
                    if native and self._label(name) != "W8A8_DYNAMIC":
                        raise ValueError(f"native MLA projection must be W8A8_DYNAMIC: {name}")
                    dtype = torch.bfloat16 if native else spec.dtype
                    self._validate_member_header(name, _source_shape(model, family, member), dtype)
        hidden = model.hidden_size
        for member, suffix, shape in (
            ("eh_proj", "eh_proj.weight", (hidden, 2 * hidden)),
            ("enorm", "enorm.weight", (hidden,)),
            ("hnorm", "hnorm.weight", (hidden,)),
            ("final_norm", "shared_head.norm.weight", (hidden,)),
        ):
            candidates = [root + f"{self._mtp_layer}." + suffix for root in TEXT_LAYER_ROOTS]
            name = self._resolve(candidates, f"mtp.{member}")
            self._source[(f"mtp.{member}", self._mtp_layer)] = name
            self._validate_member_header(name, shape, torch.bfloat16)
        self._source[("norm.final_norm", -1)] = self._resolve(
            [root + "norm.weight" for root in _TEXT_MODEL_ROOTS], "norm.final_norm"
        )
        self._validate_member_header(
            self._source[("norm.final_norm", -1)], (hidden,), torch.bfloat16
        )
        self._embed_name = self._resolve(
            [root + "embed_tokens.weight" for root in _TEXT_MODEL_ROOTS], "embed_tokens"
        )
        self._lm_head_name = self._resolve(("lm_head.weight",), "lm_head")

    def _validate_member_header(self, name: str, expected_shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Header-only shape and label gate; skipped for header-less fixture views.

        ``Checkpoint`` validates dtype at read time (``_float`` accepts only
        BF16/FP32 storage, ``quant`` only INT8); here the shape is checked from
        the shard header so a layout mismatch fails before any payload moves.
        """
        shape_of = getattr(self.checkpoint, "shape", None)
        if shape_of is None:
            return
        shape = shape_of(name)
        if tuple(shape) != expected_shape:
            raise ValueError(f"{name} has shape {tuple(shape)}, expected {expected_shape}")
        label = self._label(name)
        if label == "W8A8_DYNAMIC" and dtype != torch.bfloat16:
            # Only the MLA projections are INT8 in storage; every other member
            # must be FLOAT, and dequantizing one would silently change its ABI.
            raise ValueError(f"{name} is {label}; only BF16 members may dequantize, view is {dtype}")

    def _build_stacks(self) -> None:
        """Compute every stack's per-rank shape from the geometry alone."""
        model = self.model
        hidden = model.hidden_size
        stacks: dict[str, StackPlan] = {}
        for family, members in self._family_members.items():
            layers = self._family_layers[family]
            for spec in members:
                name = f"{family}.{spec.member}"
                shape = (len(layers), *self._member_shape(family, spec))
                stacks[name] = StackPlan(name, family, layers, shape, spec.dtype)
        moe_layers = self._family_layers["moe"]
        inter, local_experts = model.moe_intermediate_size, self.n_local_experts
        for member, shape, dtype in (
            ("w_gate_up", (local_experts, 2 * inter, hidden), torch.int8),
            ("w_gate_up_scale", (local_experts, 2 * inter), torch.float32),
            ("w_down", (local_experts, hidden, inter), torch.int8),
            ("w_down_scale", (local_experts, hidden), torch.float32),
            ("shared_w_gate_up", (2 * inter, hidden), torch.int8),
            ("shared_w_gate_up_scale", (2 * inter,), torch.float32),
            ("shared_w_down", (hidden, inter), torch.int8),
            ("shared_w_down_scale", (hidden,), torch.float32),
        ):
            name = f"moe.{member}"
            stacks[name] = StackPlan(name, "moe", moe_layers, (len(moe_layers), *shape), dtype)
        dense_layers = self._family_layers["dense"]
        local = self.local_dense_inter
        for member, shape, dtype in (
            ("w_gate_up", (2 * local, hidden), torch.int8),
            ("w_gate_up_scale", (2 * local,), torch.float32),
            ("w_down", (hidden, local), torch.int8),
            ("w_down_scale", (hidden,), torch.float32),
        ):
            name = f"dense.{member}"
            stacks[name] = StackPlan(name, "dense", dense_layers, (len(dense_layers), *shape), dtype)
        for name, family, shape in (
            ("norm.final_norm", "single", (hidden,)),
            ("mtp.eh_proj", "mtp", (hidden, 2 * hidden)),
            ("mtp.enorm", "mtp", (hidden,)),
            ("mtp.hnorm", "mtp", (hidden,)),
            ("mtp.final_norm", "mtp", (hidden,)),
            ("hadamard", "single", (model.index_head_dim, model.index_head_dim)),
        ):
            layers = (self._mtp_layer,) if family == "mtp" else ()
            stacks[name] = StackPlan(name, family, layers, shape, torch.bfloat16)
        self.stacks = stacks

    def _member_shape(self, family: str, spec: _StackMember) -> tuple[int, ...]:
        """One layer's rank-local shape for a stack member."""
        model = self.model
        hidden, qkv = model.hidden_size, self.local_kda_qkv
        if family == "kda":
            shapes = {
                "w_q": (qkv, hidden), "w_k": (qkv, hidden), "w_v": (qkv, hidden),
                "conv_q": (qkv, 1, model.linear_conv_kernel_dim),
                "conv_k": (qkv, 1, model.linear_conv_kernel_dim),
                "conv_v": (qkv, 1, model.linear_conv_kernel_dim),
                "w_f_a": (model.linear_head_dim, hidden), "w_f_b": (qkv, model.linear_head_dim),
                "w_g_a": (model.linear_head_dim, hidden), "w_g_b": (qkv, model.linear_head_dim),
                "w_b": (self.local_kda_h, hidden), "dt_bias": (qkv,),
                "a_log": (self.local_kda_h,), "o_norm": (model.linear_head_dim,),
                "w_o": (hidden, qkv),
            }
        elif family == "mla":
            shapes = {
                "w_q_a": (model.q_lora_rank, hidden), "q_a_norm": (model.q_lora_rank,),
                "w_q_b": (self.local_h * model.qk_head_dim, model.q_lora_rank),
                "w_kv_a": (model.kv_lora_rank, hidden), "kv_a_norm": (model.kv_lora_rank,),
                "w_k": (self.local_h, model.qk_head_dim, model.kv_lora_rank),
                "w_v": (self.local_h, model.v_head_dim, model.kv_lora_rank),
                "w_o": (hidden, self.local_h * model.v_head_dim),
            }
        elif family == "indexer":
            shapes = {
                "w_q_b": (model.q_lora_rank, model.index_n_heads * model.index_head_dim),
                "w_k": (hidden, model.index_head_dim),
                "k_norm_w": (model.index_head_dim,), "k_norm_b": (model.index_head_dim,),
                "w_weights": (hidden, model.index_n_heads),
                "w_compress_gate": (hidden, model.index_head_dim),
                "compress_ape": (model.index_kpool, model.index_head_dim),
            }
        elif family == "mhc":
            shapes = {
                "attn_fn": (model.mix_hc, model.hc_dim), "attn_scale": (3,), "attn_base": (model.mix_hc,),
                "ffn_fn": (model.mix_hc, model.hc_dim), "ffn_scale": (3,), "ffn_base": (model.mix_hc,),
            }
        elif family == "norm":
            shapes = {"input_norm": (hidden,), "post_norm": (hidden,)}
        else:
            shapes = {"gate_w": (model.n_routed_experts, hidden), "gate_bias": (model.n_routed_experts,)}
        if family == "mla" and spec.member.endswith("_scale"):
            return (shapes[spec.member.removesuffix("_scale")][0],)
        return shapes[spec.member]

    # -- public interface --------------------------------------------------------------

    @property
    def tensor_layouts(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Device stack name -> stacked shape/dtype, for ``runtime.alloc_tensor``."""
        return {plan.name: (plan.shape, plan.dtype) for plan in self.stacks.values()}

    @property
    def bytes_per_rank(self) -> int:
        """Device-resident bytes every rank allocates (identical across ranks)."""
        return sum(plan.bytes for plan in self.stacks.values())

    @property
    def host_bytes(self) -> int:
        """Host-side table bytes: the full LM head plus the embedding table.

        The backbone head is cached and embedding rows are gathered per prompt.
        MTP rows and head shards are read separately without retaining full tables.
        """
        return 2 * self.model.vocab_size * self.model.hidden_size * 2

    def plan(self, rank: int) -> RankPlan:
        """The rank's slice of every device stack plus byte totals; no payload reads."""
        if rank not in range(self.ranks):
            raise ValueError(f"rank must be in [0, {self.ranks}), got {rank}")
        first = rank * self.n_local_experts
        return RankPlan(
            rank=rank,
            tp_rank=rank % self.tp_size,
            ep_rank=rank,
            local_expert_ids=tuple(range(first, first + self.n_local_experts)),
            stacks=tuple(self.stacks.values()),
            device_bytes=self.bytes_per_rank,
            host_tables=("embed_tokens", "lm_head"),
        )

    def lookup_rows(self, token_ids: torch.Tensor, *, mtp: bool = False) -> torch.Tensor:
        """Gather one prompt's BF16 embedding rows from the host table."""
        ids = torch.as_tensor(token_ids, device="cpu")
        if ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("token ids must have an integer dtype")
        if bool(((ids < 0) | (ids >= self.model.vocab_size)).any()):
            raise ValueError("token id is outside the checkpoint vocabulary")
        name = self._embed_name
        if mtp:
            name = self._resolve([root + f"{self._mtp_layer}.embed_tokens.weight" for root in TEXT_LAYER_ROOTS],
                                 "MTP embedding")
        return gather_rows(self.checkpoint, name, ids).to(torch.bfloat16)

    def mtp_head_shard(self, rank: int) -> torch.Tensor:
        """Read the MTP-owned vocabulary shard without assuming tied tables."""
        if rank not in range(self.tp_size):
            raise ValueError("MTP head rank is outside the TP group")
        name = self._resolve([root + f"{self._mtp_layer}.shared_head.head.weight" for root in TEXT_LAYER_ROOTS],
                             "MTP shared head")
        width = self.model.vocab_size // self.tp_size
        return self.checkpoint.read(name, slice(rank * width, (rank + 1) * width)).bfloat16()

    def lm_head(self) -> torch.Tensor:
        """The full ``[vocab, hidden]`` BF16 head matrix, read once and cached."""
        if self._lm_head_cache is None:
            head = self.checkpoint.read(self._lm_head_name)
            expected = (self.model.vocab_size, self.model.hidden_size)
            if tuple(head.shape) != expected:
                raise ValueError(f"lm_head has shape {tuple(head.shape)}, expected {expected}")
            self._lm_head_cache = head.to(torch.bfloat16).contiguous()
        return self._lm_head_cache

    # -- materialization ----------------------------------------------------------------

    def _read_source(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Authority-driven read: FLOAT casts, W8A8_DYNAMIC dequantizes to BF16.

        Dequant is one FP32 scale multiply and one BF16 rounding, so the INT8
        storage of the four MLA projections never crosses the host/device
        boundary; a non-zero ``weight_offset`` would invalidate the symmetric
        assumption and is rejected instead of silently dropped.
        """
        label = self._label(name)
        if label == "FLOAT":
            return self.checkpoint.read(name).to(dtype)
        if dtype != torch.bfloat16:
            raise ValueError(f"{name} is {label}; only BF16 consumers may dequantize, asked for {dtype}")
        weight, scale = _read_quant(self.checkpoint, name)
        product = weight.to(torch.float32) * scale.reshape(-1, *([1] * (weight.dim() - 1)))
        return product.to(torch.bfloat16)

    def _shard(self, tensor: torch.Tensor, shard: str, tp_rank: int) -> torch.Tensor:
        """Keep one TP rank's row/column block (heads sit contiguously)."""
        if shard == "replicate":
            return tensor
        axis = 0 if shard == "rows" else 1
        size = tensor.shape[axis]
        if size % self.tp_size:
            raise ValueError(f"{shard} axis {size} does not divide TP{self.tp_size}")
        width, start = size // self.tp_size, tp_rank * (size // self.tp_size)
        return tensor[start : start + width] if axis == 0 else tensor[:, start : start + width]

    def _member_tensor(self, family: str, spec: _StackMember, layer: int, tp_rank: int) -> torch.Tensor:
        name = self._source[(f"{family}.{spec.member}", layer)]
        value = self._read_source(name, spec.dtype)
        if spec.transform == "transpose":
            value = value.t()
        return self._shard(value, spec.shard, tp_rank).contiguous()

    def _mla_layer(self, layer: int, tp_rank: int) -> dict[str, torch.Tensor]:
        """One MLA layer's eight rank slices; ``kv_b_proj`` is read once."""
        out: dict[str, torch.Tensor] = {}
        kv_b = None
        for spec in self._family_members["mla"]:
            if spec.member.endswith("_scale"):
                continue
            if spec.dtype == torch.int8:
                name = self._source[(f"mla.{spec.member}", layer)]
                weight, scale = _read_quant(self.checkpoint, name)
                out[spec.member] = self._shard(weight, spec.shard, tp_rank).contiguous()
                if spec.shard == "rows":
                    scale = self._shard(scale, "rows", tp_rank)
                out[spec.member + "_scale"] = scale.contiguous()
            elif spec.transform in ("split_k", "split_v"):
                if kv_b is None:
                    kv_b = self._read_source(self._source[("mla.w_k", layer)], torch.bfloat16)
                key, value = split_kv_b(
                    kv_b, tp_rank, self.tp_size,
                    qk_dim=self.model.qk_head_dim, v_dim=self.model.v_head_dim,
                    kv_lora=self.model.kv_lora_rank,
                )
                out[spec.member] = key if spec.transform == "split_k" else value
            else:
                out[spec.member] = self._member_tensor("mla", spec, layer, tp_rank)
        return out

    def _stack_family(self, family: str, tp_rank: int) -> dict[str, torch.Tensor]:
        """Materialize one non-FFN family as its stacked rank tensor per member."""
        collected: dict[str, list[torch.Tensor]] = {spec.member: [] for spec in self._family_members[family]}
        for layer in self._family_layers[family]:
            per_layer = (
                self._mla_layer(layer, tp_rank)
                if family == "mla"
                else {
                    spec.member: self._member_tensor(family, spec, layer, tp_rank)
                    for spec in self._family_members[family]
                }
            )
            for member, tensor in per_layer.items():
                collected[member].append(tensor)
        return {member: torch.stack(slices) for member, slices in collected.items()}

    def _stack_dense(self, tp_rank: int) -> dict[str, torch.Tensor]:
        """The dense layers' TP-sliced MLPs, stacked on the family axis."""
        model = self.model
        channels = slice(tp_rank * self.local_dense_inter, (tp_rank + 1) * self.local_dense_inter)
        shards = [
            _fused_ffn_at(
                self.checkpoint, layer_prefixes(layer), _ffn_roles("dense"),
                expert=None, intermediate=model.intermediate_size, hidden=model.hidden_size,
                channels=channels,
            )
            for layer in self._family_layers["dense"]
        ]
        return {
            "w_gate_up": torch.stack([shard[0] for shard in shards]),
            "w_gate_up_scale": torch.stack([shard[1] for shard in shards]),
            "w_down": torch.stack([shard[2] for shard in shards]),
            "w_down_scale": torch.stack([shard[3] for shard in shards]),
        }

    def _moe_layer(self, layer: int, ep_rank: int) -> dict[str, torch.Tensor]:
        """One MoE layer's local experts and replicated shared expert."""
        model = self.model
        prefixes = layer_prefixes(layer)
        first = ep_rank * self.n_local_experts
        stacked = None
        for expert in range(first, first + self.n_local_experts):
            values = _fused_ffn_at(
                self.checkpoint, prefixes, _ffn_roles("routed"),
                expert=expert, intermediate=model.moe_intermediate_size, hidden=model.hidden_size,
            )
            if stacked is None:
                stacked = [torch.empty((self.n_local_experts, *value.shape), dtype=value.dtype) for value in values]
            for target, value in zip(stacked, values):
                target[expert - first].copy_(value)
        shared = _fused_ffn_at(
            self.checkpoint, prefixes, _ffn_roles("shared"),
            expert=None, intermediate=model.moe_intermediate_size, hidden=model.hidden_size,
        )
        return {
            "w_gate_up": stacked[0], "w_gate_up_scale": stacked[1],
            "w_down": stacked[2], "w_down_scale": stacked[3],
            "shared_w_gate_up": shared[0], "shared_w_gate_up_scale": shared[1],
            "shared_w_down": shared[2], "shared_w_down_scale": shared[3],
        }

    def _single_tensors(self) -> dict[str, torch.Tensor]:
        """The unstacked residents: final norm, MTP extras, the Hadamard table."""
        from models.glm5_3_flash.indexer_cache import sylvester_hadamard  # noqa: PLC0415

        model = self.model
        hadamard = sylvester_hadamard(model.index_head_dim)
        return {
            "norm.final_norm": self._read_source(self._source[("norm.final_norm", -1)], torch.bfloat16),
            "mtp.eh_proj": self._read_source(self._source[("mtp.eh_proj", self._mtp_layer)], torch.bfloat16),
            "mtp.enorm": self._read_source(self._source[("mtp.enorm", self._mtp_layer)], torch.bfloat16),
            "mtp.hnorm": self._read_source(self._source[("mtp.hnorm", self._mtp_layer)], torch.bfloat16),
            "mtp.final_norm": self._read_source(self._source[("mtp.final_norm", self._mtp_layer)], torch.bfloat16),
            "hadamard": hadamard.contiguous(),
        }

    # -- staged upload --------------------------------------------------------------------

    def upload(self, runtime, rank: int, handles) -> None:
        """Stage one rank's resident stacks to its preallocated device tensors.

        Every stacked family stages one layer-slice at a time: the member
        slice is host-materialized, copied at its layer offset and dropped
        before the next layer, so peak host memory is one layer of one family
        (about 20 MB for a KDA layer at TP16, 280 MB at TP1) or one MoE
        layer-slice (the model's largest single buffer), never the whole
        stacked family doubled while ``torch.stack`` assembled it. Each
        per-layer ``put`` at ``layer_index`` preserves the logical layer slice
        ``torch.stack`` produced before. The two shared INT8
        matrices use NZ bytes; checkpoint-reader tensors and all other stacks
        retain logical row-major order.
        ``handles`` maps every layout name from :attr:`tensor_layouts` to a
        device handle carrying ``data_ptr``.
        """
        if rank not in range(self.ranks):
            raise ValueError(f"rank must be in [0, {self.ranks}), got {rank}")
        missing = [name for name in self.stacks if name not in handles]
        if missing:
            raise ValueError(f"handles are missing the stacks {missing}")
        tp_rank = rank % self.tp_size

        def put(name: str, data: torch.Tensor, layer_index: int | None = None) -> None:
            plan = self.stacks[name]
            if name in ("moe.shared_w_gate_up", "moe.shared_w_down"):
                data = pack_int8_nz(data)
            data = data.contiguous()
            expected = plan.shape if layer_index is None else plan.shape[1:]
            if tuple(data.shape) != expected or data.dtype != plan.dtype:
                raise ValueError(
                    f"staged {name} is {tuple(data.shape)}/{data.dtype}, planned {expected}/{plan.dtype}"
                )
            offset = 0 if layer_index is None else layer_index * data.numel() * data.element_size()
            runtime.copy_to(
                handles[name].data_ptr, data.data_ptr(), data.numel() * data.element_size(),
                dst_offset=offset, worker_id=rank,
            )

        for name, value in self._single_tensors().items():
            put(name, value)
        for family in ("norm", "mhc", "router", "kda", "mla", "indexer"):
            for index, layer in enumerate(self._family_layers[family]):
                per_layer = (
                    self._mla_layer(layer, tp_rank)
                    if family == "mla"
                    else {
                        spec.member: self._member_tensor(family, spec, layer, tp_rank)
                        for spec in self._family_members[family]
                    }
                )
                for member, value in per_layer.items():
                    put(f"{family}.{member}", value, layer_index=index)
        for member, stack in self._stack_dense(tp_rank).items():
            put(f"dense.{member}", stack)
        for index, layer in enumerate(self._family_layers["moe"]):
            for member, value in self._moe_layer(layer, rank).items():
                put(f"moe.{member}", value, layer_index=index)
if __name__ == "__main__":
    """CPU golden: layout, fusion and sharding against direct slicing.

    Host-side only, but the a2a3 sweep invokes every model file as
    ``python <file> -p a2a3 -d <ids>``, so the same switches are accepted and
    ignored. ``--tp`` / ``--ep`` are read by ``config.py`` at import time.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default="0")
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--ep", type=int, default=16)
    parser.parse_args()
    torch.manual_seed(23)

    def pieces(rows: int, width: int) -> tuple[torch.Tensor, ...]:
        """One FFN's canonical INT8 weights and per-channel scales."""
        return (
            torch.randint(-127, 128, (rows, width), dtype=torch.int8),
            torch.randint(-127, 128, (rows, width), dtype=torch.int8),
            torch.randint(-127, 128, (width, rows), dtype=torch.int8),
            torch.rand(rows, 1) * 1e-2,
            torch.rand(rows, 1) * 1e-2,
            torch.rand(width, 1) * 1e-2,
        )

    routed_global, routed_ep, routed_local = 32, 2, 16
    routed_pieces = {expert: pieces(2048, D) for expert in range(routed_global)}
    shared_pieces = pieces(2048, D)
    dense_pieces = pieces(DENSE_INTER, D)
    router_weight = torch.randn(N_EXPERTS, D, dtype=torch.bfloat16)
    router_bias = torch.randn(N_EXPERTS)

    def materialize(fused: bool, root: str = TEXT_LAYER_ROOTS[0]) -> dict[str, torch.Tensor]:
        """Write one canonical weight set out in either checkpoint spelling."""
        def prefix(layer: int) -> str:
            return f"{root}{layer}.mlp."

        out: dict[str, torch.Tensor] = {
            prefix(0) + "gate.weight": router_weight,
            prefix(0) + "gate.e_score_correction_bias": router_bias,
            prefix(3) + "gate.weight": router_weight,
            prefix(3) + "gate.e_score_correction_bias": router_bias,
        }

        def put(prefix: str, base: str, piece: tuple[torch.Tensor, ...]) -> None:
            gate, up, down, gate_scale, up_scale, down_scale = piece
            if fused:
                out[prefix + base + "w_gate_up"] = torch.cat([gate, up], dim=0)
                out[prefix + base + "w_gate_up.weight_scale"] = torch.cat([gate_scale, up_scale], dim=0)
                out[prefix + base + "w_down"] = down
                out[prefix + base + "w_down.weight_scale"] = down_scale
            else:
                out[prefix + base + "gate_proj.weight"] = gate
                out[prefix + base + "gate_proj.weight_scale"] = gate_scale
                out[prefix + base + "up_proj.weight"] = up
                out[prefix + base + "up_proj.weight_scale"] = up_scale
                out[prefix + base + "down_proj.weight"] = down
                out[prefix + base + "down_proj.weight_scale"] = down_scale

        for expert, piece in routed_pieces.items():
            put(prefix(3), f"experts.{expert}.", piece)
        put(prefix(3), "shared_experts.", shared_pieces)
        put(prefix(0), "", dense_pieces)
        return out

    sparse_fused = materialize(fused=True)
    sparse_separate = materialize(fused=False)

    def assert_close(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
        if a.shape != b.shape:
            raise AssertionError(f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        if not torch.equal(a, b):
            raise AssertionError(f"{what}: values differ")


    # Fusing separate gate/up must reproduce the fused checkpoint bit-exactly,
    # and EP must pick whole experts in the kernel's [local, 2*inter, D] layout.
    for ep_rank in (0, 1):
        shard_args = dict(local_experts=routed_local, global_experts=routed_global)
        separate = shard_routed_experts(sparse_separate, 3, ep_rank, routed_ep, **shard_args)
        fused = shard_routed_experts(sparse_fused, 3, ep_rank, routed_ep, **shard_args)
        assert_close(separate.w_gate_up, fused.w_gate_up, f"ep{ep_rank} routed gate_up")
        assert_close(separate.w_gate_up_scale, fused.w_gate_up_scale, f"ep{ep_rank} routed gate_up_scale")
        assert_close(separate.w_down, fused.w_down, f"ep{ep_rank} routed down")
        assert_close(separate.w_down_scale, fused.w_down_scale, f"ep{ep_rank} routed down_scale")
        expected = torch.stack(
            [torch.cat([routed_pieces[e][0], routed_pieces[e][1]], dim=0)
             for e in range(ep_rank * routed_local, (ep_rank + 1) * routed_local)]
        )
        assert_close(separate.w_gate_up, expected, f"ep{ep_rank} routed layout")
        assert separate.w_gate_up.dtype == torch.int8
        assert separate.w_gate_up_scale.dtype == torch.float32
        assert separate.w_gate_up_scale.shape == (routed_local, 4096)

    # Shared expert: every EP rank keeps complete projections and scales.
    shared = load_shared_expert(sparse_fused, 3)
    shared_separate = load_shared_expert(sparse_separate, 3)
    inter = FLASH.moe_intermediate_size
    assert_close(shared.gate_up[:inter], shared_pieces[0], "shared gate")
    assert_close(shared.gate_up[inter:], shared_pieces[1], "shared up")
    assert_close(shared.down, shared_pieces[2], "shared down")
    assert_close(
        shared.gate_up_scale,
        torch.cat([shared_pieces[3].reshape(-1), shared_pieces[4].reshape(-1)]),
        "shared gate/up scale",
    )
    assert_close(shared.down_scale, shared_pieces[5].reshape(-1), "shared down scale")
    for field in ("gate_up", "gate_up_scale", "down", "down_scale"):
        assert_close(getattr(shared_separate, field), getattr(shared, field), f"shared fused {field}")

    # Dense layer: TP slices the intermediate width, and the router passes through
    # with the kernels' dtypes.
    dense = shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE)
    local_dense = DENSE_INTER // TP_SIZE
    assert dense.gate_up.shape == (2 * local_dense, D)
    assert dense.down.shape == (D, local_dense)
    assert dense.gate_up_scale.shape == (2 * local_dense,)
    assert dense.down_scale.shape == (D,)
    assert_close(dense.gate_up[:local_dense], dense_pieces[0][:local_dense], "dense gate slice")
    router = layer_router(sparse_fused, 0)
    assert router.weight.dtype == torch.bfloat16 and router.weight.shape == (N_EXPERTS, D)
    assert router.correction_bias.dtype == torch.float32 and router.correction_bias.shape == (N_EXPERTS,)
    assert_close(router.correction_bias, router_bias, "router bias")

    # The released checkpoint nests the text backbone under ``language_model``
    # (``model.language_model.layers.*``); both roots load the same tensors.
    nested = materialize(fused=True, root="model.language_model.layers.")
    ep_rank = 1
    nested_routed = shard_routed_experts(
        nested, 3, ep_rank, routed_ep, local_experts=routed_local, global_experts=routed_global
    )
    canonical_routed = shard_routed_experts(
        sparse_fused, 3, ep_rank, routed_ep, local_experts=routed_local, global_experts=routed_global
    )
    for field in ("w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale"):
        assert_close(getattr(nested_routed, field), getattr(canonical_routed, field), f"nested routed {field}")
    nested_shared = load_shared_expert(nested, 3)
    assert_close(nested_shared.gate_up, shared.gate_up, "nested shared gate_up")
    assert_close(nested_shared.down, shared.down, "nested shared down")
    nested_dense = shard_dense_mlp(nested, 0, 0, TP_SIZE)
    assert_close(nested_dense.gate_up, dense.gate_up, "nested dense gate_up")
    nested_router = layer_router(nested, 3)
    canonical_router = layer_router(sparse_fused, 3)
    assert_close(nested_router.weight, canonical_router.weight, "nested router weight")
    assert_close(nested_router.correction_bias, canonical_router.correction_bias, "nested router bias")
    try:
        load_shared_expert(materialize(fused=True, root="model.text."), 3)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown layer root should have been rejected")

    # Boundary validation: the one shipped scheme, and a scale is mandatory.
    QuantDescription.parse(
        {"model_quant_type": "W8A8_DYNAMIC", "group_size": 0, "is_rot_used": True}
    ).require_dynamic_symmetric()
    for bad in (
        {"model_quant_type": "FP8", "group_size": 0},
        {"model_quant_type": "W8A8_DYNAMIC", "group_size": 128},
    ):
        try:
            QuantDescription.parse(bad).require_dynamic_symmetric()
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should have been rejected")
    try:
        check_ffn_scales(sparse_fused, [layer_prefix(0) + "missing.weight"])
    except KeyError:
        pass
    else:
        raise AssertionError("a missing scale should have been rejected")

    # Shard geometry is validated rather than silently slicing short.
    for bad_call in (
        lambda: shard_dense_mlp(sparse_fused, 0, -1, TP_SIZE),
        lambda: shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE, intermediate=DENSE_INTER + 32),
        lambda: shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE, local_inter=DENSE_INTER // TP_SIZE + 1),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid shard geometry should have been rejected")

    # The .safetensors branch of ``load_tensors`` is the one loader path with no
    # coverage when the optional package is absent; round-trip a small shard
    # through it whenever the package is present.
    try:
        from safetensors.torch import save_file  # noqa: PLC0415
    except ImportError:
        print("[GOLDEN] skip  weights safetensors reader (safetensors not installed)")
    else:
        import tempfile  # noqa: PLC0415

        probe = {
            "probe.weight": torch.arange(24, dtype=torch.int8).reshape(4, 6),
            "probe.weight_scale": torch.linspace(0.1, 1.0, 4).reshape(4, 1),
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_file(probe, str(Path(tmp) / "shard-0.safetensors"))
            loaded = load_tensors(tmp)
            if not torch.equal(loaded["probe.weight"], probe["probe.weight"]):
                raise AssertionError("safetensors reader lost the weight payload")
            if not torch.equal(
                _scale_of(loaded, "probe.weight"), probe["probe.weight_scale"].reshape(-1)
            ):
                raise AssertionError("safetensors reader did not flatten the [out, 1] scale")
        print("[GOLDEN] PASS weights (safetensors round-trip)")

    print(
        f"[GOLDEN] PASS weights (routed 32 experts over EP{routed_ep}, shared replicated, dense TP{TP_SIZE})"
    )

    # ------------------- resident prefill staging (in-memory fixture) -------------------
    # Everything below runs against a scaled-geometry synthetic checkpoint held
    # in memory through the staging read protocol: never /data/models, never a
    # device, and never the optional safetensors package. ``Checkpoint`` parsing
    # and metadata validation are covered by the loader tests above; these cases
    # cover the staging itself - resolution, fusion, sharding, splitting,
    # stacking and the staged upload.
    import ctypes  # noqa: PLC0415

    from models.glm5_3_flash.indexer_cache import sylvester_hadamard  # noqa: PLC0415
    from models.glm5_3_flash.golden import golden_split_kv_b  # noqa: PLC0415

    TINY = Glm53FlashConfig(
        name="glm5_3_flash_tiny", vocab_size=128, hidden_size=64, num_hidden_layers=4,
        first_k_dense_replace=1, intermediate_size=32, rms_norm_eps=1e-5,
        max_position_embeddings=1024, swiglu_limit=10.0, tie_word_embeddings=False,
        num_attention_heads=4, q_lora_rank=12, kv_lora_rank=6, qk_nope_head_dim=8,
        qk_rope_head_dim=0, v_head_dim=8, index_n_heads=2, index_head_dim=8,
        index_topk=4, index_kpool=2, index_kpool_compress=True,
        index_kpool_always_select_tail=True, index_share_for_mtp_iteration=True,
        linear_num_heads=4, linear_head_dim=8, linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0, n_routed_experts=16, n_shared_experts=1,
        num_experts_per_tok=2, moe_intermediate_size=32, scoring_func="sigmoid",
        topk_method="noaux_tc", norm_topk_prob=True, routed_scaling_factor=2.5,
        n_group=1, topk_group=1, mhc=True, hc_mult=2, hc_sinkhorn_iters=20,
        hc_eps=1e-6, num_nextn_predict_layers=1, dtype="bfloat16", quant_method="w8a8",
        activation_scheme="dynamic",
    )

    def tiny_root(layer: int) -> str:
        """The fixture's dual-spelling rule: layers 0-1 canonical, 2-4 nested."""
        return "model.layers." if layer <= 1 else "model.language_model.layers."

    class _FixtureWeights(Mapping[str, torch.Tensor]):
        """In-memory stand-in for :class:`Checkpoint` (the staging read protocol)."""

        def __init__(self, tensors: dict[str, torch.Tensor], labels: dict[str, str]):
            self.tensors, self.description = tensors, labels

        def __contains__(self, name: object) -> bool:
            return name in self.tensors

        def __iter__(self):
            return iter(self.tensors)

        def __len__(self) -> int:
            return len(self.tensors)

        def __getitem__(self, name: str) -> torch.Tensor:
            return self.tensors[name]

        def read(self, name: str, selection=None) -> torch.Tensor:
            return self.tensors[name] if selection is None else self.tensors[name][selection]

        def shape(self, name: str) -> tuple[int, ...]:
            return tuple(self.tensors[name].shape)

        def gather(self, name: str, rows: torch.Tensor) -> torch.Tensor:
            return self.tensors[name][rows]

    def build_tiny_fixture(mutate=None) -> _FixtureWeights:
        """Materialize the scaled-geometry deployment checkpoint the tests run against.

        Covers both layer-root spellings, nested and flat KDA-gate/mHC spellings,
        fused and separate gate/up layouts, the quantized MLA projections with
        scales and zero offsets, and the MTP duplicates of the host tables.
        ``mutate(tensors, labels)`` exists for the negative tests.
        """
        generator = torch.Generator().manual_seed(101)
        hidden = TINY.hidden_size
        qkv = TINY.linear_num_heads * TINY.linear_head_dim
        tensors: dict[str, torch.Tensor] = {}
        labels: dict[str, str] = {}

        def put(name: str, tensor: torch.Tensor, label: str = "FLOAT") -> None:
            tensors[name] = tensor
            labels[name] = label

        def bf16(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator).bfloat16()

        def int8_linear(stem: str, out: int, inn: int, *, bare: bool = False) -> None:
            name = stem if bare else stem + ".weight"
            weight = torch.randint(-127, 128, (out, inn), dtype=torch.int8, generator=generator)
            scale = torch.rand(out, 1, generator=generator).float() * 1e-2
            put(name, weight, "W8A8_DYNAMIC")
            put(stem + ".weight_scale", scale, "W8A8_DYNAMIC")
            put(stem + ".weight_offset", torch.zeros(out, 1), "W8A8_DYNAMIC")

        def ffn(base: str, inter: int, *, fused: bool, dotted: bool = False) -> None:
            # The canonical fused spelling is bare (``w_gate_up``); only the
            # routed expert entries also accept the ``.weight`` variant.
            if fused:
                int8_linear(base + "w_gate_up", 2 * inter, hidden, bare=not dotted)
                int8_linear(base + "w_down", hidden, inter, bare=not dotted)
            else:
                for part in ("gate", "up"):
                    int8_linear(base + part + "_proj", inter, hidden)
                int8_linear(base + "down_proj", hidden, inter)

        for layer in range(TINY.num_checkpoint_layers):
            base = tiny_root(layer) + f"{layer}."
            if layer in TINY.kda_layer_ids:
                attn = base + "self_attn."
                for part in ("q", "k", "v"):
                    put(attn + f"{part}_proj.weight", bf16(qkv, hidden))
                    put(attn + f"{part}_conv1d.weight", bf16(qkv, 1, TINY.linear_conv_kernel_dim))
                gates = attn if layer == 0 else attn + "forget_gate."  # flat vs nested spelling
                put(gates + "dt_bias", torch.randn(qkv, generator=generator).float())
                put(gates + "A_log", torch.randn(TINY.linear_num_heads, generator=generator).float())
                put(gates + "f_a_proj.weight", bf16(TINY.linear_head_dim, hidden))
                put(gates + "f_b_proj.weight", bf16(qkv, TINY.linear_head_dim))
                put(attn + "g_a_proj.weight", bf16(TINY.linear_head_dim, hidden))
                put(attn + "g_b_proj.weight", bf16(qkv, TINY.linear_head_dim))
                put(attn + "b_proj.weight", bf16(TINY.linear_num_heads, hidden))
                put(attn + "o_norm.weight", bf16(TINY.linear_head_dim))
                put(attn + "o_proj.weight", bf16(hidden, qkv))
            else:  # MLA + indexer (layers 3 and 4, including the MTP layer)
                attn = base + "self_attn."
                int8_linear(attn + "q_a_proj", TINY.q_lora_rank, hidden)
                put(attn + "q_a_layernorm.weight", bf16(TINY.q_lora_rank))
                int8_linear(attn + "q_b_proj", TINY.num_attention_heads * TINY.qk_head_dim, TINY.q_lora_rank)
                int8_linear(attn + "kv_a_proj_with_mqa", TINY.kv_lora_rank, hidden)
                put(attn + "kv_a_layernorm.weight", bf16(TINY.kv_lora_rank))
                put(
                    attn + "kv_b_proj.weight",
                    bf16(TINY.num_attention_heads * (TINY.qk_head_dim + TINY.v_head_dim), TINY.kv_lora_rank),
                )
                int8_linear(attn + "o_proj", hidden, TINY.num_attention_heads * TINY.v_head_dim)
                indexer = attn + "indexer."
                put(indexer + "wq_b.weight", bf16(TINY.index_n_heads * TINY.index_head_dim, TINY.q_lora_rank))
                put(indexer + "wk.weight", bf16(TINY.index_head_dim, hidden))
                put(indexer + "k_norm.weight", bf16(TINY.index_head_dim))
                put(indexer + "k_norm.bias", bf16(TINY.index_head_dim))
                put(indexer + "weights_proj.weight", bf16(TINY.index_n_heads, hidden))
                put(indexer + "index_kpool_compress_gate", bf16(TINY.index_head_dim, hidden))
                put(indexer + "index_kpool_compress_ape", bf16(TINY.index_kpool, TINY.index_head_dim))
            if layer < TINY.num_hidden_layers:  # mHC is backbone-only
                for kind in ("attn", "ffn"):
                    if layer == 0:  # flat spelling on one layer, canonical elsewhere
                        put(base + f"hc_{kind}_fn", bf16(TINY.mix_hc, TINY.hc_dim))
                        put(base + f"hc_{kind}_scale", torch.rand(3, generator=generator).float())
                        put(base + f"hc_{kind}_base", torch.rand(TINY.mix_hc, generator=generator).float())
                    else:
                        put(base + f"{kind}_hc.fn", bf16(TINY.mix_hc, TINY.hc_dim))
                        put(base + f"{kind}_hc.scale", torch.rand(3, generator=generator).float())
                        put(base + f"{kind}_hc.base", torch.rand(TINY.mix_hc, generator=generator).float())
            put(base + "input_layernorm.weight", bf16(hidden))
            put(base + "post_attention_layernorm.weight", bf16(hidden))
            if layer >= TINY.first_k_dense_replace:
                put(base + "mlp.gate.weight", bf16(TINY.n_routed_experts, hidden))
                flat_bias = layer == 1  # flat spelling on one layer, canonical elsewhere
                bias_stem = "mlp.e_score_correction_bias" if flat_bias else "mlp.gate.e_score_correction_bias"
                put(base + bias_stem, torch.randn(TINY.n_routed_experts, generator=generator).float())
                for expert in range(TINY.n_routed_experts):
                    dotted = layer == 4 and expert % 2 == 0  # the routed ``.weight`` variant
                    fused = layer == 2 or dotted
                    ffn(base + f"mlp.experts.{expert}.", TINY.moe_intermediate_size, fused=fused, dotted=dotted)
                ffn(base + "mlp.shared_experts.", TINY.moe_intermediate_size, fused=layer >= 3)
            else:
                ffn(base + "mlp.", TINY.intermediate_size, fused=False)
        embed = bf16(TINY.vocab_size, hidden)
        head = bf16(TINY.vocab_size, hidden)
        put("model.language_model.embed_tokens.weight", embed)
        put("lm_head.weight", head)
        put("model.language_model.norm.weight", bf16(hidden))
        mtp = tiny_root(TINY.mtp_layer_id) + f"{TINY.mtp_layer_id}."
        put(mtp + "eh_proj.weight", bf16(hidden, 2 * hidden))
        put(mtp + "enorm.weight", bf16(hidden))
        put(mtp + "hnorm.weight", bf16(hidden))
        put(mtp + "embed_tokens.weight", embed.clone())          # MTP duplicates alias the
        put(mtp + "shared_head.head.weight", head.clone())       # top-level host tables
        put(mtp + "shared_head.norm.weight", bf16(hidden))       # ignored, like rot.weight
        if mutate is not None:
            mutate(tensors, labels)
        return _FixtureWeights(tensors, labels)

    # Resident test 1: the fixture loads through the staging protocol and the
    # plan arithmetic holds.
    checkpoint = build_tiny_fixture()
    assert checkpoint.description["model.language_model.layers.3.self_attn.q_b_proj.weight"] == "W8A8_DYNAMIC"
    assert checkpoint.description["model.language_model.layers.3.self_attn.kv_b_proj.weight"] == "FLOAT"
    weights = ResidentPrefillWeights.load(checkpoint, tp_size=2, ep_size=2, model=TINY)
    assert weights.bytes_per_rank == sum(
        math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        for shape, dtype in weights.tensor_layouts.values()
    )
    assert weights.plan(1).device_bytes == weights.bytes_per_rank
    assert weights.plan(1).tp_rank == 1 and weights.plan(1).local_expert_ids == tuple(range(8, 16))
    try:
        weights.plan(2)
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-range rank should have been rejected")

    # Resident test 3: authority-driven dequant of the quantized MLA
    # projections equals int8 * scale with exactly one BF16 rounding, and
    # survives a matmul within that rounding.
    q_b_name = "model.language_model.layers.3.self_attn.q_b_proj.weight"
    q_b_stem = "model.language_model.layers.3.self_attn.q_b_proj"
    raw, scale = checkpoint[q_b_name], checkpoint[q_b_stem + ".weight_scale"].reshape(-1)
    dequant = weights._read_source(q_b_name, torch.bfloat16)
    assert dequant.dtype == torch.bfloat16
    assert_close(dequant, (raw.float() * scale.reshape(-1, 1)).to(torch.bfloat16), "mla q_b dequant")
    mla_layer = weights._mla_layer(3, tp_rank=1)
    assert_close(mla_layer["w_q_b"], dequant[16:], "mla q_b tp slice")  # 4 heads x 8 rows, head-major
    probe = torch.randn(6, TINY.q_lora_rank).bfloat16()
    got = (probe.float() @ mla_layer["w_q_b"].float().t()).to(torch.bfloat16)
    reference = (probe.float() @ (raw[16:].float() * scale[16:].reshape(-1, 1)).t()).to(torch.bfloat16)
    # One BF16 rounding of the weight matrix: the error scales with the
    # products' magnitude, so bound it by the reference's span, not an absolute.
    bound = reference.abs().max().item() / 64
    if not torch.allclose(got.float(), reference.float(), rtol=2.0 / 128, atol=bound):
        raise AssertionError("dequantized q_b disagrees with the reference within one BF16 rounding")
    kv_b = checkpoint["model.language_model.layers.3.self_attn.kv_b_proj.weight"]
    kv_b_passthrough = weights._read_source(
        "model.language_model.layers.3.self_attn.kv_b_proj.weight", torch.bfloat16
    )
    assert_close(kv_b_passthrough, kv_b, "float passthrough")
    bad_offset = build_tiny_fixture(
        mutate=lambda tensors, labels: tensors.update(
            {"model.language_model.layers.3.self_attn.q_b_proj.weight_offset": torch.ones(32, 1)}
        )
    )
    bad_offset_weights = ResidentPrefillWeights.load(bad_offset, tp_size=2, ep_size=2, model=TINY)
    try:
        bad_offset_weights._read_source(q_b_name, torch.bfloat16)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-zero MLA weight_offset should have been rejected")
    bad_ffn_offset = build_tiny_fixture(
        mutate=lambda tensors, labels: tensors.update(
            {"model.layers.1.mlp.experts.0.gate_proj.weight_offset": torch.ones(16, 1)}
        )
    )
    bad_ffn_weights = ResidentPrefillWeights.load(bad_ffn_offset, tp_size=2, ep_size=2, model=TINY)
    try:
        bad_ffn_weights._moe_layer(1, ep_rank=0)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-zero FFN weight_offset should have been rejected")

    # Resident test 4: every family's rank slices are disjoint and cover the
    # source exactly once, across the bring-up geometries.
    for tp_size, ep_size in ((1, 1), (2, 2), (4, 16)):
        sharded = ResidentPrefillWeights.load(checkpoint, tp_size=tp_size, ep_size=ep_size, model=TINY)
        kda_stacks = [sharded._stack_family("kda", rank % tp_size) for rank in range(tp_size)]
        for stack in kda_stacks:
            if not all(tensor.is_contiguous() for tensor in stack.values()):
                raise AssertionError("every staged family tensor must be contiguous")
        for index, layer in enumerate(TINY.kda_layer_ids):
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.q_proj.weight"]
            rebuilt = torch.cat([stack["w_q"][index] for stack in kda_stacks], dim=0)
            assert_close(rebuilt, source, f"kda w_q coverage tp{tp_size} layer {layer}")
        mla_stacks = [sharded._stack_family("mla", rank % tp_size) for rank in range(tp_size)]
        heads = TINY.num_attention_heads
        for index, layer in enumerate(TINY.indexer_layer_ids):
            prefix = tiny_root(layer) + f"{layer}.self_attn."
            kv_source = checkpoint[prefix + "kv_b_proj.weight"]
            reshaped = kv_source.reshape(
                heads, TINY.qk_head_dim + TINY.v_head_dim, TINY.kv_lora_rank
            )
            key, value = reshaped[:, : TINY.qk_head_dim], reshaped[:, TINY.qk_head_dim :]
            assert_close(
                torch.cat([s["w_k"][index] for s in mla_stacks], dim=0),
                key,
                f"kv_b key coverage tp{tp_size}",
            )
            assert_close(
                torch.cat([s["w_v"][index] for s in mla_stacks], dim=0),
                value,
                f"kv_b value coverage tp{tp_size}",
            )
            # o_proj is W8A8 on the MLA layers: coverage is against the
            # dequantized source, which is what the ranks actually hold.
            o_source = sharded._read_source(prefix + "o_proj.weight", torch.bfloat16)
            assert_close(
                torch.cat([s["w_o"][index] for s in mla_stacks], dim=1),
                o_source,
                f"mla o_proj column coverage tp{tp_size}",
            )
        expert_ids = [expert for rank in range(ep_size) for expert in sharded.plan(rank).local_expert_ids]
        assert sorted(expert_ids) == list(range(TINY.n_routed_experts))
        layer_one = 1
        moe_per_rank = [sharded._moe_layer(layer_one, rank) for rank in range(ep_size)]
        for piece in moe_per_rank:
            if not all(tensor.is_contiguous() for tensor in piece.values()):
                raise AssertionError("every staged moe tensor must be contiguous")
        rebuilt_experts = torch.cat([piece["w_gate_up"] for piece in moe_per_rank], dim=0)

        def expert_tensor(expert: int, part: str) -> torch.Tensor:
            stem = tiny_root(layer_one) + f"{layer_one}.mlp.experts.{expert}.{part}_proj.weight"
            return checkpoint[stem]

        expected_experts = torch.stack([
            torch.cat([expert_tensor(expert, "gate"), expert_tensor(expert, "up")], dim=0)
            for expert in range(TINY.n_routed_experts)
        ])
        assert_close(rebuilt_experts, expected_experts, f"routed expert coverage ep{ep_size}")
    for bad_call in (
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=8, ep_size=8, model=TINY),   # 4 MLA heads
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=3, ep_size=3, model=TINY),   # unsupported
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=2, ep_size=32, model=TINY),  # unsupported EP
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=4, ep_size=2, model=TINY),   # EP % TP
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid TP/EP geometry should have been rejected")

    # Resident test 5: the kv_b halves land reduction-first ([QK, KV_LORA]),
    # match the inline split sliced per head, and the full-geometry helper is
    # bit-equal to the shared projection golden with complete head coverage.
    kv_source = checkpoint["model.language_model.layers.3.self_attn.kv_b_proj.weight"]
    reshaped = kv_source.reshape(
        TINY.num_attention_heads, TINY.qk_head_dim + TINY.v_head_dim, TINY.kv_lora_rank
    )
    expected_key, expected_value = reshaped[:, : TINY.qk_head_dim], reshaped[:, TINY.qk_head_dim :]
    assert mla_layer["w_k"].shape == (2, TINY.qk_head_dim, TINY.kv_lora_rank)  # local_h = 4/2
    assert mla_layer["w_v"].shape == (2, TINY.v_head_dim, TINY.kv_lora_rank)
    assert_close(mla_layer["w_k"], expected_key[2:], "reduction-first w_k head slice")
    assert_close(mla_layer["w_v"], expected_value[2:], "reduction-first w_v head slice")
    generator = torch.Generator().manual_seed(202)
    full_kv_b = torch.randn(
        FLASH.num_attention_heads * (FLASH.qk_head_dim + FLASH.v_head_dim),
        FLASH.kv_lora_rank, generator=generator,
    ).bfloat16()
    golden_key, golden_value = golden_split_kv_b(full_kv_b)
    rank_keys = [split_kv_b(full_kv_b, rank, 16)[0] for rank in range(16)]
    rank_values = [split_kv_b(full_kv_b, rank, 16)[1] for rank in range(16)]
    assert_close(torch.cat(rank_keys, dim=0), golden_key, "split_kv_b covers every head exactly once")
    assert_close(torch.cat(rank_values, dim=0), golden_value, "split_kv_b value coverage")
    assert rank_keys[3].shape == (FLASH.num_attention_heads // 16, FLASH.qk_head_dim, FLASH.kv_lora_rank)

    # Resident test 6: the Hadamard table is the scaled Sylvester form and
    # orthonormal at BF16 tolerance.
    singles = weights._single_tensors()
    hadamard = singles["hadamard"]
    matrix = torch.ones((1, 1))
    while matrix.shape[0] < TINY.index_head_dim:
        top = torch.cat([matrix, matrix], dim=1)
        bottom = torch.cat([matrix, -matrix], dim=1)
        matrix = torch.cat([top, bottom], dim=0)
    local_build = (matrix * TINY.index_head_dim ** -0.5).to(torch.bfloat16)
    assert hadamard.dtype == torch.bfloat16 and torch.equal(hadamard, local_build)
    # The full-width table is the sylvester_hadamard the loader imports,
    # checked orthonormal at INDEX_DIM = 128.
    big = sylvester_hadamard(128)
    if not torch.allclose(big.float() @ big.float().t(), torch.eye(128), atol=2e-2):
        raise AssertionError("the Hadamard table is not orthonormal at BF16 tolerance")

    # Resident test 7: the indexer projections are transposed once into the
    # kernels' K-major [in, out] layout.
    indexer_stack = weights._stack_family("indexer", tp_rank=0)
    transposed = (("w_q_b", "wq_b.weight"), ("w_k", "wk.weight"),
                  ("w_weights", "weights_proj.weight"), ("w_compress_gate", "index_kpool_compress_gate"))
    passthrough = (("k_norm_w", "k_norm.weight"), ("k_norm_b", "k_norm.bias"),
                   ("compress_ape", "index_kpool_compress_ape"))
    for index, layer in enumerate(TINY.indexer_layer_ids):
        for member, stem in transposed:
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.indexer.{stem}"]
            assert_close(
                indexer_stack[member][index], source.t(), f"indexer {member} transpose layer {layer}"
            )
        for member, stem in passthrough:
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.indexer.{stem}"]
            assert_close(
                indexer_stack[member][index], source, f"indexer {member} passthrough layer {layer}"
            )

    # Resident test 8: backbone tables and independently addressed MTP tables.
    assert weights.host_bytes == 2 * TINY.vocab_size * TINY.hidden_size * 2
    assert not any("embed" in name or "head" in name for name in weights.tensor_layouts)

    def nudge_mtp_embed(tensors, labels):
        tensors["model.language_model.layers.4.embed_tokens.weight"][0].add_(1.0)
        tensors["model.language_model.layers.4.shared_head.head.weight"][0].add_(2.0)

    untied_weights = ResidentPrefillWeights.load(
        build_tiny_fixture(mutate=nudge_mtp_embed), tp_size=2, ep_size=2, model=TINY
    )
    mtp_rows = untied_weights.lookup_rows(torch.tensor([0]), mtp=True)
    backbone_rows = untied_weights.lookup_rows(torch.tensor([0]))
    assert_close(mtp_rows, backbone_rows + 1, "MTP keeps its untied embedding values")
    mtp_head = untied_weights.mtp_head_shard(0)
    assert_close(mtp_head[0], untied_weights.lm_head()[0] + 2, "MTP keeps its untied head values")
    rows = weights.lookup_rows(torch.tensor([[3, 0], [127, 5]]))
    assert rows.shape == (2, 2, TINY.hidden_size) and rows.dtype == torch.bfloat16
    embed_table = checkpoint["model.language_model.embed_tokens.weight"]
    assert_close(rows[1, 0], embed_table[127], "lookup_rows gathers the host embed table")
    assert_close(rows[0, 1], embed_table[0], "lookup_rows keeps the request shape")
    assert weights.lm_head().shape == (TINY.vocab_size, TINY.hidden_size)
    try:
        weights.lookup_rows(torch.tensor([TINY.vocab_size]))
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-vocabulary token id should have been rejected")

    # Resident test 10: the staged upload - order, per-layer MoE staging, byte
    # coverage, and rank tagging - against a recording runtime.
    class _Handle:
        def __init__(self, pointer: int):
            self.data_ptr = pointer

    class _RecordingRuntime:
        def __init__(self):
            self.copies = []

        def copy_to(self, dst, src, nbytes, dst_offset=0, worker_id=None):
            self.copies.append((dst, dst_offset, nbytes, worker_id))

    runtime = _RecordingRuntime()
    handles = {name: _Handle(index) for index, name in enumerate(weights.tensor_layouts)}
    weights.upload(runtime, rank=1, handles=handles)
    names_by_pointer = {index: name for index, name in enumerate(weights.tensor_layouts)}
    per_name: dict[str, list[tuple[int, int]]] = {}
    for dst, offset, nbytes, worker in runtime.copies:
        assert worker == 1, "every copy must be tagged with the uploading rank"
        per_name.setdefault(names_by_pointer[dst], []).append((offset, nbytes))
    for name, plan in weights.stacks.items():
        copies = per_name.get(name, [])
        if not copies:
            raise AssertionError(f"stack {name} was never staged")
        if sum(nbytes for _, nbytes in copies) != plan.bytes:
            raise AssertionError(f"stack {name} staged {sum(n for _, n in copies)} of {plan.bytes} bytes")
        if plan.family in ("norm", "mhc", "router", "kda", "mla", "indexer", "moe"):
            layer_bytes = math.prod(plan.shape[1:]) * torch.empty((), dtype=plan.dtype).element_size()
            offsets = [offset for offset, _ in copies]
            if offsets != [index * layer_bytes for index in range(len(plan.layers))]:
                raise AssertionError(f"stack {name} was not staged one layer at a time in order")
            if any(nbytes != layer_bytes for _, nbytes in copies):
                raise AssertionError(f"stack {name} staged a partial layer slice")
        else:
            if copies != [(0, plan.bytes)]:
                raise AssertionError(f"unstacked resident {name} must be one whole copy")
    first_seen = list(dict.fromkeys(name.split(".")[0] for name in
                                    (names_by_pointer[dst] for dst, _, _, _ in runtime.copies)))
    expected_order = ["norm", "mtp", "hadamard", "mhc", "router", "kda", "mla", "indexer", "dense", "moe"]
    if first_seen != expected_order:
        raise AssertionError(f"staging order {first_seen} != {expected_order}")

    # Logical byte parity of per-layer staging versus whole-stack materialization:
    # decode the two shared NZ stacks before comparing to the logical
    # ``torch.stack`` reference. All other staged bytes stay unchanged.
    class _ByteRuntime(_RecordingRuntime):
        """Capture the staged source bytes at copy time (pointer valid there)."""

        def __init__(self):
            super().__init__()
            self.blobs: dict[str, bytearray] = {}

        def copy_to(self, dst, src, nbytes, dst_offset=0, worker_id=None):
            super().copy_to(dst, src, nbytes, dst_offset, worker_id)
            name = names_by_pointer[dst]
            blob = self.blobs.setdefault(name, bytearray(weights.stacks[name].bytes))
            blob[dst_offset : dst_offset + nbytes] = ctypes.string_at(src, nbytes)

    def raw_bytes(tensor: torch.Tensor) -> bytes:
        return bytes(tensor.contiguous().flatten().view(torch.uint8).numpy())

    byte_runtime = _ByteRuntime()
    weights.upload(byte_runtime, rank=1, handles=handles)
    reference: dict[str, torch.Tensor] = {}
    for family in ("norm", "mhc", "router", "kda", "mla", "indexer"):
        for member, stack in weights._stack_family(family, tp_rank=1).items():
            reference[f"{family}.{member}"] = stack
    for member, stack in weights._stack_dense(tp_rank=1).items():
        reference[f"dense.{member}"] = stack
    for member in ("w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale",
                   "shared_w_gate_up", "shared_w_gate_up_scale",
                   "shared_w_down", "shared_w_down_scale"):
        reference[f"moe.{member}"] = torch.stack(
            [weights._moe_layer(layer, 1)[member] for layer in weights._family_layers["moe"]]
        )
    for name, tensor in weights._single_tensors().items():
        reference[name] = tensor
    if set(reference) != set(byte_runtime.blobs):
        raise AssertionError("the parity reference and the staged stacks disagree on names")
    for name, tensor in reference.items():
        staged = bytes(byte_runtime.blobs[name])
        if name in ("moe.shared_w_gate_up", "moe.shared_w_down"):
            physical = torch.frombuffer(byte_runtime.blobs[name], dtype=tensor.dtype).reshape(tensor.shape)
            staged = raw_bytes(unpack_int8_nz(physical))
        if staged != raw_bytes(tensor):
            raise AssertionError(f"per-layer staging of {name} is not byte-identical to the stack")
    print(f"[GOLDEN] PASS weights upload byte parity ({len(reference)} stacks, rank 1 of TP2/EP2)")

    fresh = _RecordingRuntime()
    try:
        weights.upload(fresh, rank=0, handles={})
    except ValueError:
        pass
    else:
        raise AssertionError("upload with missing handles should have been rejected")
    try:
        weights.upload(fresh, rank=2, handles=handles)
    except ValueError:
        pass
    else:
        raise AssertionError("upload with an out-of-range rank should have been rejected")
    copies = len(runtime.copies)
    message = f"[GOLDEN] PASS weights resident loader (tiny checkpoint, TP1-4/EP1-16, {copies} copies)"
    print(message)

    # Resident test 9: the full-geometry byte budget, reproduced from headers
    # only and pinned to the design arithmetic (weights_resident_design_1015.md
    # section 6). TP16/EP16 must land on the 22.05 GB headline; the retired TP1
    # appendix (34.98 GB) predates the BF16-resident MLA and host-side head
    # decisions, so the design-consistent TP1/EP16 value is the same component
    # arithmetic carried to TP1.
    class _FullGeometryCheckpoint(Mapping[str, torch.Tensor]):
        """Geometry-only checkpoint view: canonical names, no files, no payloads."""

        def __init__(self):
            model = FLASH
            names: dict[str, str] = {}
            for layer in range(model.num_checkpoint_layers):
                base = f"model.layers.{layer}."
                if layer in model.kda_layer_ids:
                    for part in ("q", "k", "v"):
                        names[base + f"self_attn.{part}_proj.weight"] = "FLOAT"
                        names[base + f"self_attn.{part}_conv1d.weight"] = "FLOAT"
                    for stem in ("dt_bias", "A_log", "f_a_proj.weight", "f_b_proj.weight"):
                        names[base + "self_attn.forget_gate." + stem] = "FLOAT"
                    kda_stems = ("g_a_proj.weight", "g_b_proj.weight", "b_proj.weight",
                                 "o_norm.weight", "o_proj.weight")
                    for stem in kda_stems:
                        names[base + "self_attn." + stem] = "FLOAT"
                else:
                    quant_stems = ("q_a_proj.weight", "q_b_proj.weight",
                                   "kv_a_proj_with_mqa.weight", "o_proj.weight")
                    for stem in quant_stems:
                        names[base + "self_attn." + stem] = "W8A8_DYNAMIC"
                        names[base + "self_attn." + stem + ".weight_scale"] = "W8A8_DYNAMIC"
                        names[base + "self_attn." + stem + ".weight_offset"] = "W8A8_DYNAMIC"
                    for stem in ("q_a_layernorm.weight", "kv_a_layernorm.weight", "kv_b_proj.weight"):
                        names[base + "self_attn." + stem] = "FLOAT"
                    for stem in ("wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias",
                                 "weights_proj.weight", "index_kpool_compress_gate",
                                 "index_kpool_compress_ape"):
                        names[base + "self_attn.indexer." + stem] = "FLOAT"
                if layer < model.num_hidden_layers:
                    for kind in ("attn", "ffn"):
                        names[base + f"{kind}_hc.fn"] = "FLOAT"
                        names[base + f"{kind}_hc.scale"] = "FLOAT"
                        names[base + f"{kind}_hc.base"] = "FLOAT"
                for stem in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                    names[base + stem] = "FLOAT"
                if layer >= model.first_k_dense_replace:
                    names[base + "mlp.gate.weight"] = "FLOAT"
                    names[base + "mlp.gate.e_score_correction_bias"] = "FLOAT"
                    bases = [f"mlp.experts.{expert}." for expert in range(model.n_routed_experts)]
                    bases.append("mlp.shared_experts.")
                else:
                    bases = ["mlp."]
                for expert_base in bases:
                    for part in ("gate_proj", "up_proj", "down_proj"):
                        names[base + expert_base + part + ".weight"] = "W8A8_DYNAMIC"
                        names[base + expert_base + part + ".weight_scale"] = "W8A8_DYNAMIC"
                        names[base + expert_base + part + ".weight_offset"] = "W8A8_DYNAMIC"
            mtp = f"model.layers.{model.mtp_layer_id}."
            for stem in ("eh_proj.weight", "enorm.weight", "hnorm.weight", "embed_tokens.weight",
                         "shared_head.head.weight", "shared_head.norm.weight"):
                names[mtp + stem] = "FLOAT"
            names["model.language_model.norm.weight"] = "FLOAT"
            names["model.language_model.embed_tokens.weight"] = "FLOAT"
            names["lm_head.weight"] = "FLOAT"
            self._names = names

        def __contains__(self, name: object) -> bool:
            return name in self._names

        def __iter__(self):
            return iter(self._names)

        def __len__(self) -> int:
            return len(self._names)

        def __getitem__(self, name: str) -> torch.Tensor:
            raise AssertionError("the budget gate must not read payloads")

        @property
        def description(self) -> dict[str, str]:
            return self._names

        def read(self, name: str, selection=None) -> torch.Tensor:
            raise AssertionError("the budget gate must not read payloads")

    def family_bytes(resident, family: str, *, shared: bool | None = None) -> int:
        total = 0
        for plan in resident.stacks.values():
            if plan.family != family:
                continue
            if shared is True and not plan.name.startswith("moe.shared"):
                continue
            if shared is False and plan.name.startswith("moe.shared"):
                continue
            total += plan.bytes
        return total

    # Hand-computed from the design's section 2.2 shapes and section 4 mappings
    # (INT8 = 1 B + 4 B/output-channel scale, BF16 = 2 B, F32 = 4 B, decimal GB).
    TP16_COMPONENTS = {
        "kda": 652_252_192, "mla": 364_953_600, "indexer": 179_324_928, "mhc": 70_788_600,
        "norm": 753_664, "router": 101_499_264, "routed": 19_503_710_208, "shared": 1_083_539_456,
        "dense": 28_379_136, "mtp": 67_133_440, "single": 40_960,
    }
    TP16_TOTAL = 22_052_375_448   # 22.05 GB per rank, the design's headline
    TP1_OVERRIDES = {"kda": 9_366_356_992, "mla": 2_818_621_440, "dense": 453_328_896}
    TP1_TOTAL = 33_645_097_848
    for tp_size, overrides, total in ((16, {}, TP16_TOTAL), (1, TP1_OVERRIDES, TP1_TOTAL)):
        resident = ResidentPrefillWeights.load(_FullGeometryCheckpoint(), tp_size=tp_size, ep_size=16)
        components = {**TP16_COMPONENTS, **overrides}
        for family in ("kda", "mla", "indexer", "mhc", "norm", "router", "dense", "mtp", "single"):
            got = family_bytes(resident, family)
            if got != components[family]:
                raise AssertionError(
                    f"TP{tp_size} {family} budget {got} != design value {components[family]}"
                )
        routed = family_bytes(resident, "moe", shared=False)
        shared = family_bytes(resident, "moe", shared=True)
        if (routed, shared) != (components["routed"], components["shared"]):
            raise AssertionError(f"TP{tp_size} moe budget {(routed, shared)} != design values")
        if resident.bytes_per_rank != total:
            raise AssertionError(f"TP{tp_size}/EP16 budget {resident.bytes_per_rank} != design total {total}")
        if resident.host_bytes != 2 * FLASH.vocab_size * FLASH.hidden_size * 2:
            raise AssertionError("host bytes must be the head plus the embedding table")
        assert resident.plan(0).device_bytes == resident.bytes_per_rank
    if abs(TP16_TOTAL / 1e9 - 22.05) > 0.01:
        raise AssertionError("the TP16 budget drifted off the design's 22.05 GB headline")
    print("[GOLDEN] PASS weights resident budget (TP16/EP16 22.05 GB, TP1/EP16 33.65 GB, headers only)")
