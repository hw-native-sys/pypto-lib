# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""W8A8 deployment-checkpoint loading and TP/EP sharding.

Host-side only: this file turns a checkpoint tensor mapping into the exact tensor
set the FFN kernels take. It owns no device code, so it is unit-testable on CPU
(``python models/glm5_3_flash/weights.py``).

What the deployment checkpoint is
---------------------------------
``Eco-Tech/GLM-5.3-Flash-w8a8`` declares one scheme, ``W8A8_DYNAMIC`` with
``group_size: 0`` (per-output-channel), and its ``quant_model_description.json``
is the authority on the boundary: **the FFN is INT8 and everything else is
BF16**. Each quantized tensor ships a ``weight_scale`` (FP32 per-output-channel)
and a ``weight_offset`` that measures all-zero, so the loader reads the scale and
discards the offset.

Layout
------
The kernels consume fused gate/up halves, so a checkpoint that stores separate
``gate_proj``/``up_proj`` (the released naming) is fused here into
``w_gate_up``: gate rows first, then up rows, with the two scale vectors
concatenated in the same order. A checkpoint that already stores the fused
tensor is read as-is.

Sharding
--------
* **Routed experts are EP-sharded**: rank ``ep_rank`` of ``ep_size`` owns experts
  ``[ep_rank * N_LOCAL, (ep_rank + 1) * N_LOCAL)``, whole tensors — no weight
  slicing inside an expert.
* **Shared and dense FFNs are TP-sharded on the intermediate axis**: each rank
  keeps ``intermediate / tp_size`` channels of the *fused* layout, i.e. the gate
  head ``[0, local_inter)`` and the up head
  ``[intermediate, intermediate + local_inter)``, and ``w_down``'s matching
  columns. The routing/attention tensors are not covered here yet; they keep
  their checkpoint layout until their kernels land.

Naming
------
The released checkpoint nests the text backbone under ``language_model``
(``model.language_model.layers.*``) while converted builds keep the scaffold's
root spelling; every lookup accepts both, canonical first.

Scale shapes are normalized: a checkpoint's ``[out, 1]`` becomes the kernels'
``[out]``.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
    """One FFN's INT8 weights and dequant scales, in kernel layout."""

    gate_up: torch.Tensor        # [2 * local_inter, D] INT8, gate rows then up rows
    gate_up_scale: torch.Tensor  # [2 * local_inter] FP32
    down: torch.Tensor           # [D, local_inter] INT8
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
            scale = tensors[key].to(torch.float32)
            # [out, 1] -> [out]; already-1-D stays put.
            return scale.reshape(-1).contiguous()
    raise KeyError(f"{name} has no weight_scale ({list(SCALE_SUFFIXES)})")


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read one FFN as (gate_up, gate_up_scale, down, down_scale) in full shape.

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
        return (
            tensors[fused_name].to(torch.int8).contiguous(),
            _scale_of(tensors, fused_name),
            tensors[down_name].to(torch.int8).contiguous(),
            _scale_of(tensors, down_name),
        )
    gate_name = _find(tensors, _prefixed(prefixes, ROLES[gate_role], **fmt))
    up_name = _find(tensors, _prefixed(prefixes, ROLES[up_role], **fmt))
    gate_up, gate_up_scale = fuse_gate_up(
        tensors[gate_name].to(torch.int8),
        tensors[up_name].to(torch.int8),
        _scale_of(tensors, gate_name),
        _scale_of(tensors, up_name),
    )
    return gate_up, gate_up_scale, tensors[down_name].to(torch.int8).contiguous(), _scale_of(tensors, down_name)


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
    if global_experts % ep_size:
        raise ValueError(f"{global_experts} experts cannot be split across EP{ep_size}")
    if local_experts != global_experts // ep_size:
        raise ValueError(f"local_experts {local_experts} != {global_experts} / {ep_size}")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")

    prefixes = layer_prefixes(layer_id)
    first = ep_rank * local_experts
    gate_ups, gate_up_scales, downs, down_scales = [], [], [], []
    for expert in range(first, first + local_experts):
        gate_up, gate_up_scale, down, down_scale = _fused_ffn(
            tensors, prefixes, ("routed_fused", "routed_fused_down", "routed_gate", "routed_up", "routed_down"),
            expert=expert
        )
        gate_ups.append(gate_up)
        gate_up_scales.append(gate_up_scale)
        downs.append(down)
        down_scales.append(down_scale)
    return RoutedShard(
        w_gate_up=torch.stack(gate_ups).contiguous(),
        w_gate_up_scale=torch.stack(gate_up_scales).contiguous(),
        w_down=torch.stack(downs).contiguous(),
        w_down_scale=torch.stack(down_scales).contiguous(),
    )


def _local_intermediate_slice(
    gate_up: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down: torch.Tensor,
    down_scale: torch.Tensor,
    *,
    intermediate: int,
    local_inter: int,
    tp_rank: int,
) -> FfnShard:
    """Keep one TP rank's channels of the fused gate/up layout and of ``down``."""
    if gate_up.shape[0] != 2 * intermediate:
        raise ValueError(f"fused gate/up height {gate_up.shape[0]} != 2 * {intermediate}")
    if not 0 <= tp_rank or local_inter <= 0 or intermediate % local_inter or (tp_rank + 1) * local_inter > intermediate:
        raise ValueError(f"tp_rank {tp_rank} with local_inter {local_inter} does not fit {intermediate}")
    start = tp_rank * local_inter
    gate_rows = gate_up[start : start + local_inter]
    up_rows = gate_up[intermediate + start : intermediate + start + local_inter]
    fused_rows = torch.cat([gate_rows, up_rows], dim=0).contiguous()
    fused_scale = torch.cat(
        [
            gate_up_scale[start : start + local_inter],
            gate_up_scale[intermediate + start : intermediate + start + local_inter],
        ]
    ).contiguous()
    # `w_down` is [hidden, local_inter]: TP slices its *input* axis, so every
    # hidden row survives and the per-output-channel scale is kept whole.
    return FfnShard(
        gate_up=fused_rows,
        gate_up_scale=fused_scale,
        down=down[:, start : start + local_inter].contiguous(),
        down_scale=down_scale.contiguous(),
    )


def shard_shared_expert(
    tensors: Mapping[str, torch.Tensor],
    layer_id: int,
    tp_rank: int,
    tp_size: int,
    *,
    intermediate: int = FLASH.moe_intermediate_size,
    local_inter: int | None = None,
) -> FfnShard:
    """TP-slice the shared expert's fused layout for one rank."""
    if tp_size not in SUPPORTED_TP_SIZES:
        raise ValueError(f"tp_size must be one of {SUPPORTED_TP_SIZES}, got {tp_size}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    local = local_inter if local_inter is not None else intermediate // tp_size
    if intermediate % tp_size:
        raise ValueError(f"shared intermediate {intermediate} does not divide TP{tp_size}")
    prefixes = layer_prefixes(layer_id)
    gate_up, gate_up_scale, down, down_scale = _fused_ffn(
        tensors, prefixes, ("shared_fused", "shared_fused_down", "shared_gate", "shared_up", "shared_down")
    )
    return _local_intermediate_slice(
        gate_up, gate_up_scale, down, down_scale,
        intermediate=intermediate, local_inter=local, tp_rank=tp_rank,
    )


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
    prefixes = layer_prefixes(layer_id)
    gate_up, gate_up_scale, down, down_scale = _fused_ffn(
        tensors, prefixes, ("dense_fused", "dense_fused_down", "dense_gate", "dense_up", "dense_down")
    )
    return _local_intermediate_slice(
        gate_up, gate_up_scale, down, down_scale,
        intermediate=intermediate, local_inter=local_inter, tp_rank=tp_rank,
    )


def layer_router(tensors: Mapping[str, torch.Tensor], layer_id: int) -> RouterTensors:
    """Read the layer's router: BF16 weights, FP32 correction bias."""
    prefixes = layer_prefixes(layer_id)
    weight_name = _find(tensors, _prefixed(prefixes, ROLES["router_weight"]))
    bias_name = _find(tensors, _prefixed(prefixes, ROLES["router_bias"]))
    return RouterTensors(
        weight=tensors[weight_name].to(torch.bfloat16).contiguous(),
        correction_bias=tensors[bias_name].to(torch.float32).reshape(-1).contiguous(),
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

    # Shared expert: the TP rank keeps the gate head and the up head of its own
    # channels, and `down` keeps the matching columns.
    tp_rank = 1
    shared = shard_shared_expert(sparse_fused, 3, tp_rank, TP_SIZE)
    local = 2048 // TP_SIZE
    assert_close(shared.gate_up[:local], shared_pieces[0][local : 2 * local], "shared gate slice")
    assert_close(shared.gate_up[local:], shared_pieces[1][local : 2 * local], "shared up slice")
    assert_close(shared.down, shared_pieces[2][:, local : 2 * local], "shared down slice")
    assert_close(shared.down_scale, shared_pieces[5].reshape(-1), "shared down scale")

    # Dense layer: same rules at the dense width, and the router passes through
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
    # (``model.language_model.layers.*``); the nested spelling must shard
    # identically to the canonical one, and an unknown root must still fail.
    nested = materialize(fused=True, root="model.language_model.layers.")
    nested_routed = shard_routed_experts(
        nested, 3, tp_rank, routed_ep, local_experts=routed_local, global_experts=routed_global
    )
    canonical_routed = shard_routed_experts(
        sparse_fused, 3, tp_rank, routed_ep, local_experts=routed_local, global_experts=routed_global
    )
    for field in ("w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale"):
        assert_close(getattr(nested_routed, field), getattr(canonical_routed, field), f"nested routed {field}")
    nested_shared = shard_shared_expert(nested, 3, tp_rank, TP_SIZE)
    assert_close(nested_shared.gate_up, shared.gate_up, "nested shared gate_up")
    assert_close(nested_shared.down, shared.down, "nested shared down")
    nested_dense = shard_dense_mlp(nested, 0, 0, TP_SIZE)
    assert_close(nested_dense.gate_up, dense.gate_up, "nested dense gate_up")
    nested_router = layer_router(nested, 3)
    canonical_router = layer_router(sparse_fused, 3)
    assert_close(nested_router.weight, canonical_router.weight, "nested router weight")
    assert_close(nested_router.correction_bias, canonical_router.correction_bias, "nested router bias")
    try:
        shard_shared_expert(materialize(fused=True, root="model.text."), 3, tp_rank, TP_SIZE)
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
    moe_inter = FLASH.moe_intermediate_size
    fused_gu = torch.cat([shared_pieces[0], shared_pieces[1]], dim=0)
    fused_gu_scale = torch.cat([shared_pieces[3].reshape(-1), shared_pieces[4].reshape(-1)])
    for bad_call in (
        lambda: shard_shared_expert(sparse_fused, 3, TP_SIZE, TP_SIZE),
        lambda: shard_dense_mlp(sparse_fused, 0, -1, TP_SIZE),
        lambda: shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE, intermediate=DENSE_INTER + 32),
        # The private helper is reachable directly and carries its own guard.
        lambda: _local_intermediate_slice(
            fused_gu, fused_gu_scale, shared_pieces[2], shared_pieces[5].reshape(-1),
            intermediate=moe_inter, local_inter=moe_inter // TP_SIZE, tp_rank=-1,
        ),
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
        f"[GOLDEN] PASS weights (routed 32 experts over EP{routed_ep}, dense/shared TP{TP_SIZE})"
    )
