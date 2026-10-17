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
from models.glm5_3_flash.mla_prolog import golden_absorb_output

# This directory owns a ``golden.py`` reference module, so the repository-root
# ``golden`` harness package must come first on the path before any harness import.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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


if __name__ == "__main__":
    """CPU golden: layout, fusion and sharding against direct slicing.

    Host-side only, but the a2a3 sweep invokes every model file as
    ``python <file> -p a2a3 -d <ids>``, so the same switches are accepted and
    ignored. ``--tp`` / ``--ep`` are read by ``config.py`` at import time.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3")
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
