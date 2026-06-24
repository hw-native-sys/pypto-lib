# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 real-weight adapters for the packed prefill bring-up scripts.

The standalone kernels keep synthetic fixture generation as their default.  This
module only rewrites selected ``TensorSpec.init_value`` callables so the same
single-layer kernel can be launched with the quantized FLASH safetensors under
``/data/models/dsv4-flash-w8a8``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from config import FLASH as MODEL_CONFIG
from golden import ScalarSpec, TensorSpec
from moe import (
    D,
    HC_DIM,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    TOPK,
    VOCAB,
)
from prefill_attention_csa import (
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    H,
    HEAD_DIM,
    INNER_OUT_DIM,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
)
from prefill_attention_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
)
from rope_tables import build_deepseek_v4_rope_tables

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - only exercised on hosts without safetensors.
    safe_open = None


MODEL_INDEX = "model.safetensors.index.json"


class DeepSeekV4WeightStore:
    """Small safetensors index/cache wrapper for DeepSeek-V4 FLASH weights."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        index_path = self.root / MODEL_INDEX
        if not index_path.is_file():
            raise FileNotFoundError(f"missing safetensors index: {index_path}")
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._handles: dict[str, Any] = {}

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def _handle(self, key: str) -> Any:
        if safe_open is None:
            raise ImportError("safetensors is required to load DeepSeek-V4 real weights")
        try:
            shard = self.weight_map[key]
        except KeyError:
            raise KeyError(f"weight key not found in {MODEL_INDEX}: {key}") from None
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.root / shard), framework="pt", device="cpu")
        return self._handles[shard]

    def load(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)

    def shape_dtype(self, key: str) -> tuple[list[int], str]:
        view = self._handle(key).get_slice(key)
        return list(view.get_shape()), str(view.get_dtype())


def attention_kind_for_layer(layer_id: int) -> str:
    ratio = int(MODEL_CONFIG.compress_ratios[layer_id])
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported compress ratio {ratio} at layer {layer_id}")


def _ranked(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.unsqueeze(0).expand(N_RANKS, *tensor.shape).contiguous()


def _i8_transpose(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.t().contiguous().to(torch.int8)


def _bf16_transpose(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.t().contiguous().to(torch.bfloat16)


def _fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().to(torch.float32)


def _bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().to(torch.bfloat16)


def _reshape_wo_a(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(O_GROUPS, O_LORA, O_GROUP_IN).to(torch.bfloat16)


def _reshape_or_check(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _compress_ratio_for_layer(layer_id: int) -> int:
    return int(MODEL_CONFIG.compress_ratios[layer_id])


def _rope_tables_for_layer(layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    return build_deepseek_v4_rope_tables(
        MODEL_CONFIG,
        _compress_ratio_for_layer(layer_id),
        max_seq_len=MAX_SEQ_LEN,
        rope_dim=ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
    )


def _lower_tid2eid(tid2eid: torch.Tensor) -> torch.Tensor:
    # EP bring-up intentionally uses only the first N_EXPERTS_GLOBAL real
    # experts.  Keep hash-gate routing in-bounds by folding official expert IDs
    # into that local bring-up expert range.
    table = tid2eid[:, :TOPK].to(torch.int64) % int(N_EXPERTS_GLOBAL)
    return table.contiguous().to(torch.int32)


def _expert_prefix() -> range:
    return range(int(N_EXPERTS_GLOBAL))


def _load_routed_experts(store: DeepSeekV4WeightStore, layer_id: int) -> dict[str, torch.Tensor]:
    base = f"layers.{layer_id}.ffn.experts"
    w1 = torch.empty((N_RANKS, N_LOCAL, MOE_INTER, D), dtype=torch.int8)
    w3 = torch.empty_like(w1)
    w2 = torch.empty((N_RANKS, N_LOCAL, D, MOE_INTER), dtype=torch.int8)
    s1 = torch.empty((N_RANKS, N_LOCAL, MOE_INTER), dtype=torch.float32)
    s3 = torch.empty_like(s1)
    s2 = torch.empty((N_RANKS, N_LOCAL, D), dtype=torch.float32)

    for global_eid in _expert_prefix():
        rank = global_eid // N_LOCAL
        local_eid = global_eid - rank * N_LOCAL
        prefix = f"{base}.{global_eid}"
        w1[rank, local_eid] = store.load(f"{prefix}.w1.weight").contiguous().to(torch.int8)
        w3[rank, local_eid] = store.load(f"{prefix}.w3.weight").contiguous().to(torch.int8)
        w2[rank, local_eid] = store.load(f"{prefix}.w2.weight").contiguous().to(torch.int8)
        s1[rank, local_eid] = _fp32(store.load(f"{prefix}.w1.scale"))
        s3[rank, local_eid] = _fp32(store.load(f"{prefix}.w3.scale"))
        s2[rank, local_eid] = _fp32(store.load(f"{prefix}.w2.scale"))

    return {
        "routed_w1": w1,
        "routed_w1_scale": s1,
        "routed_w3": w3,
        "routed_w3_scale": s3,
        "routed_w2": w2,
        "routed_w2_scale": s2,
    }


def _load_common_attention(store: DeepSeekV4WeightStore, layer_id: int) -> dict[str, torch.Tensor]:
    base = f"layers.{layer_id}"
    attn = f"{base}.attn"
    cos, sin = _rope_tables_for_layer(layer_id)
    return {
        "hc_attn_fn": _ranked(_fp32(store.load(f"{base}.hc_attn_fn"))),
        "hc_attn_scale": _ranked(_fp32(store.load(f"{base}.hc_attn_scale"))),
        "hc_attn_base": _ranked(_fp32(store.load(f"{base}.hc_attn_base"))),
        "attn_norm_w": _ranked(_bf16(store.load(f"{base}.attn_norm.weight"))),
        "wq_a": _ranked(_bf16_transpose(store.load(f"{attn}.wq_a.weight"))),
        "wq_b": _ranked(_i8_transpose(store.load(f"{attn}.wq_b.weight"))),
        "wq_b_scale": _ranked(_fp32(store.load(f"{attn}.wq_b.scale"))),
        "wkv": _ranked(_bf16_transpose(store.load(f"{attn}.wkv.weight"))),
        "gamma_cq": _ranked(_bf16(store.load(f"{attn}.q_norm.weight"))),
        "gamma_ckv": _ranked(_bf16(store.load(f"{attn}.kv_norm.weight"))),
        "freqs_cos": _ranked(cos),
        "freqs_sin": _ranked(sin),
        "attn_sink": _ranked(_fp32(store.load(f"{attn}.attn_sink"))),
        "wo_a": _ranked(_reshape_wo_a(store.load(f"{attn}.wo_a.weight"))),
        "wo_b": _ranked(store.load(f"{attn}.wo_b.weight").contiguous().to(torch.int8)),
        "wo_b_scale": _ranked(_fp32(store.load(f"{attn}.wo_b.scale"))),
    }


def _load_hca_compressor(store: DeepSeekV4WeightStore, layer_id: int) -> dict[str, torch.Tensor]:
    prefix = f"layers.{layer_id}.attn.compressor"
    return {
        "hca_cmp_wkv": _ranked(_bf16_transpose(store.load(f"{prefix}.wkv.weight"))),
        "hca_cmp_wgate": _ranked(_bf16_transpose(store.load(f"{prefix}.wgate.weight"))),
        "hca_cmp_ape": _ranked(
            _reshape_or_check(
                f"{prefix}.ape",
                _fp32(store.load(f"{prefix}.ape")),
                (HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM),
            )
        ),
        "hca_cmp_norm_w": _ranked(_bf16(store.load(f"{prefix}.norm.weight"))),
    }


def _load_csa_compressor(store: DeepSeekV4WeightStore, layer_id: int) -> dict[str, torch.Tensor]:
    cmp_prefix = f"layers.{layer_id}.attn.compressor"
    idx_prefix = f"layers.{layer_id}.attn.indexer"
    inner_prefix = f"{idx_prefix}.compressor"
    out = {
        "csa_cmp_wkv": _ranked(_bf16_transpose(store.load(f"{cmp_prefix}.wkv.weight"))),
        "csa_cmp_wgate": _ranked(_bf16_transpose(store.load(f"{cmp_prefix}.wgate.weight"))),
        "csa_cmp_ape": _ranked(
            _reshape_or_check(
                f"{cmp_prefix}.ape",
                _fp32(store.load(f"{cmp_prefix}.ape")),
                (CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM),
            )
        ),
        "csa_cmp_norm_w": _ranked(_bf16(store.load(f"{cmp_prefix}.norm.weight"))),
        "csa_inner_wkv": _ranked(_bf16_transpose(store.load(f"{inner_prefix}.wkv.weight"))),
        "csa_inner_wgate": _ranked(_bf16_transpose(store.load(f"{inner_prefix}.wgate.weight"))),
        "csa_inner_ape": _ranked(
            _reshape_or_check(
                f"{inner_prefix}.ape",
                _fp32(store.load(f"{inner_prefix}.ape")),
                (CSA_COMPRESS_RATIO, INNER_OUT_DIM),
            )
        ),
        "csa_inner_norm_w": _ranked(_bf16(store.load(f"{inner_prefix}.norm.weight"))),
    }
    hadamard_key = f"{idx_prefix}.hadamard_idx"
    if store.has(hadamard_key):
        out["csa_hadamard_idx"] = _ranked(_bf16(store.load(hadamard_key)))
    return out


def _load_moe(store: DeepSeekV4WeightStore, layer_id: int) -> dict[str, torch.Tensor]:
    base = f"layers.{layer_id}"
    ffn = f"{base}.ffn"
    gate = f"{ffn}.gate"
    gate_w = _fp32(store.load(f"{gate}.weight")[:N_EXPERTS_GLOBAL])
    if store.has(f"{gate}.bias"):
        gate_bias = _fp32(store.load(f"{gate}.bias")[:N_EXPERTS_GLOBAL])
    else:
        gate_bias = torch.zeros((N_EXPERTS_GLOBAL,), dtype=torch.float32)

    out = {
        "hc_ffn_fn": _ranked(_fp32(store.load(f"{base}.hc_ffn_fn"))),
        "hc_ffn_scale": _ranked(_fp32(store.load(f"{base}.hc_ffn_scale"))),
        "hc_ffn_base": _ranked(_fp32(store.load(f"{base}.hc_ffn_base"))),
        "norm_w": _ranked(_bf16(store.load(f"{base}.ffn_norm.weight"))),
        "gate_w": _ranked(gate_w),
        "gate_bias": _ranked(gate_bias),
        "shared_w1": _ranked(store.load(f"{ffn}.shared_experts.w1.weight").contiguous().to(torch.int8)),
        "shared_w1_scale": _ranked(_fp32(store.load(f"{ffn}.shared_experts.w1.scale"))),
        "shared_w3": _ranked(store.load(f"{ffn}.shared_experts.w3.weight").contiguous().to(torch.int8)),
        "shared_w3_scale": _ranked(_fp32(store.load(f"{ffn}.shared_experts.w3.scale"))),
        "shared_w2": _ranked(store.load(f"{ffn}.shared_experts.w2.weight").contiguous().to(torch.int8)),
        "shared_w2_scale": _ranked(_fp32(store.load(f"{ffn}.shared_experts.w2.scale"))),
    }
    tid2eid_key = f"{gate}.tid2eid"
    if store.has(tid2eid_key):
        out["tid2eid"] = _ranked(_lower_tid2eid(store.load(tid2eid_key)))
    out.update(_load_routed_experts(store, layer_id))
    return out


def load_layer_overrides(root: str | Path, layer_id: int) -> dict[str, torch.Tensor]:
    """Return real-weight tensors keyed by ``prefill_layer.build_tensor_specs`` names."""
    store = DeepSeekV4WeightStore(root)
    overrides: dict[str, torch.Tensor] = {}
    overrides.update(_load_common_attention(store, layer_id))
    kind = attention_kind_for_layer(layer_id)
    if kind == "hca":
        overrides.update(_load_hca_compressor(store, layer_id))
    elif kind == "csa":
        overrides.update(_load_csa_compressor(store, layer_id))
    overrides.update(_load_moe(store, layer_id))
    return overrides


def apply_real_weight_overrides(
    specs: list[TensorSpec | ScalarSpec],
    root: str | Path,
    layer_id: int,
) -> list[TensorSpec | ScalarSpec]:
    """Replace synthetic model-weight TensorSpecs with real-weight init callables."""
    overrides = load_layer_overrides(root, layer_id)
    out: list[TensorSpec | ScalarSpec] = []
    for spec in specs:
        if isinstance(spec, ScalarSpec) or spec.name not in overrides:
            out.append(spec)
            continue
        tensor = overrides[spec.name]
        if list(tensor.shape) != list(spec.shape):
            raise ValueError(
                f"{spec.name}: real-weight shape {list(tensor.shape)} does not match spec {spec.shape}"
            )

        def init_value(t=tensor):
            return t.clone()

        out.append(replace(spec, init_value=init_value))
    return out


def layer_inventory(root: str | Path, layer_id: int) -> list[tuple[str, list[int], str]]:
    """Return source safetensors keys for a layer with their indexed shape/dtype."""
    store = DeepSeekV4WeightStore(root)
    keys = [
        f"layers.{layer_id}.hc_attn_fn",
        f"layers.{layer_id}.hc_attn_scale",
        f"layers.{layer_id}.hc_attn_base",
        f"layers.{layer_id}.attn_norm.weight",
        f"layers.{layer_id}.attn.wq_a.weight",
        f"layers.{layer_id}.attn.wq_b.weight",
        f"layers.{layer_id}.attn.wq_b.scale",
        f"layers.{layer_id}.attn.wkv.weight",
        f"layers.{layer_id}.attn.q_norm.weight",
        f"layers.{layer_id}.attn.kv_norm.weight",
        f"layers.{layer_id}.attn.attn_sink",
        f"layers.{layer_id}.attn.wo_a.weight",
        f"layers.{layer_id}.attn.wo_b.weight",
        f"layers.{layer_id}.attn.wo_b.scale",
        f"layers.{layer_id}.ffn_norm.weight",
        f"layers.{layer_id}.ffn.gate.weight",
        f"layers.{layer_id}.ffn.gate.bias",
        f"layers.{layer_id}.ffn.gate.tid2eid",
        f"layers.{layer_id}.hc_ffn_fn",
        f"layers.{layer_id}.hc_ffn_scale",
        f"layers.{layer_id}.hc_ffn_base",
        f"layers.{layer_id}.ffn.shared_experts.w1.weight",
        f"layers.{layer_id}.ffn.shared_experts.w2.weight",
        f"layers.{layer_id}.ffn.shared_experts.w3.weight",
    ]
    kind = attention_kind_for_layer(layer_id)
    if kind == "hca":
        keys += [
            f"layers.{layer_id}.attn.compressor.wkv.weight",
            f"layers.{layer_id}.attn.compressor.wgate.weight",
            f"layers.{layer_id}.attn.compressor.ape",
            f"layers.{layer_id}.attn.compressor.norm.weight",
        ]
    elif kind == "csa":
        keys += [
            f"layers.{layer_id}.attn.compressor.wkv.weight",
            f"layers.{layer_id}.attn.compressor.wgate.weight",
            f"layers.{layer_id}.attn.compressor.ape",
            f"layers.{layer_id}.attn.compressor.norm.weight",
            f"layers.{layer_id}.attn.indexer.compressor.wkv.weight",
            f"layers.{layer_id}.attn.indexer.compressor.wgate.weight",
            f"layers.{layer_id}.attn.indexer.compressor.ape",
            f"layers.{layer_id}.attn.indexer.compressor.norm.weight",
            f"layers.{layer_id}.attn.indexer.hadamard_idx",
        ]
    rows = []
    for key in keys:
        if not store.has(key):
            rows.append((key, [], "MISSING"))
            continue
        shape, dtype = store.shape_dtype(key)
        rows.append((key, shape, dtype))
    return rows


def _parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DeepSeek-V4 FLASH safetensors for prefill.")
    parser.add_argument("--weight-root", type=str, required=True)
    parser.add_argument("--layers", type=str, default="0,2,3",
                        help="Comma/range layer list, e.g. 0,2,3 or 0-9.")
    parser.add_argument("--check-load", action="store_true",
                        help="Actually materialize real-weight overrides for each layer.")
    args = parser.parse_args()

    for layer_id in _parse_layers(args.layers):
        print(f"\n[layer {layer_id}] kind={attention_kind_for_layer(layer_id)}")
        for key, shape, dtype in layer_inventory(args.weight_root, layer_id):
            print(f"  {key}: {shape or '-'} {dtype}")
        if args.check_load:
            overrides = load_layer_overrides(args.weight_root, layer_id)
            print(f"  loaded overrides: {len(overrides)} tensors")


if __name__ == "__main__":
    main()
