# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay one complete physical Decode layer on two A3 ranks.

The input boundary is frozen by the independent vLLM-Ascend capture.  This
process only imports PyPTO and the host checkpoint reader; vLLM and torch-npu
remain confined to the reference capture process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from golden import ScalarSpec, TensorSpec, run
from pypto.ir import DistributedConfig

from models.deepseek_v4_1_flash_a2.decode_layer_ep2 import (
    CHANNELS,
    D,
    EP_SIZE,
    HC,
    HIDDEN,
    LATENT,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    O_RANK,
    ROPE_PAIRS,
    T,
    TOPK,
    WIDTH,
    WINDOW,
    decode_layer_ep2,
)
from models.deepseek_v4_1_flash_a2.decode_weight_abi import (
    load_router,
    load_routed_shard,
    load_shared_expert,
)
from models.deepseek_v4_1_flash_a2.decode_weights import DecodeSafetensorsCheckpoint
from models.deepseek_v4_1_flash_a2.utils import unpack_nz


DEFAULT_ASSETS = Path(".pypto-a3-state/reference/decode_ep2_l0_20261106_v26")
DEFAULT_CACHE_ASSETS = Path(".pypto-a3-state/reference/decode_ep2_l0_20261106_v24")
DEFAULT_FIXTURE = Path(".pypto-a3-state/product/decode-layer-ep2-l0-fixture-v27")
TARGET_LAYER_IDS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)


def _bf16_storage(value: np.ndarray) -> torch.Tensor:
    if value.dtype != np.int16:
        raise ValueError(f"expected BF16 int16 storage, got {value.dtype}")
    return torch.from_numpy(np.ascontiguousarray(value)).view(torch.bfloat16)


def _capture(asset_root: Path, rank: int, name: str) -> dict[str, np.ndarray]:
    path = asset_root / f"rank{rank}" / name
    with np.load(path) as archive:
        return {key: np.ascontiguousarray(archive[key]) for key in archive.files}


def _decode_boundary(root: Path) -> dict[str, Any]:
    path = root / "decode_input.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Decode input contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "checkpoint",
        "layer_id",
        "decode_token",
        "history_token_length",
        "position_ids",
        "block_table",
        "slot_mapping",
        "valid_lengths",
        "active_finished_request_state",
        "topology",
    )
    missing = [name for name in required if name not in contract]
    if missing:
        raise ValueError(f"Decode input contract is incomplete: {missing}")
    return contract


def _validate_reference_boundary(asset_root: Path, cache_root: Path) -> dict[str, Any]:
    asset = _decode_boundary(asset_root)
    cache = _decode_boundary(cache_root)
    fields = (
        "checkpoint",
        "layer_id",
        "decode_token",
        "history_token_length",
        "position_ids",
        "block_table",
        "slot_mapping",
        "valid_lengths",
        "active_finished_request_state",
        "topology",
    )
    mismatches = {
        name: {"assets": asset[name], "cache_assets": cache[name]}
        for name in fields
        if asset[name] != cache[name]
    }
    if mismatches:
        raise ValueError(f"reference/cache Decode boundary mismatch: {mismatches}")
    reference_path = asset_root / "reference_capture.json"
    if not reference_path.is_file():
        raise FileNotFoundError(f"missing reference capture contract: {reference_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("input_contract") != "decode-token-cache-request-state-v1":
        raise ValueError(
            "reference capture is diagnostic-only or has no cache state: "
            f"{reference.get('input_contract')!r}"
        )
    if not all(reference.get(name) is True for name in ("same_inputs", "same_layer", "same_topology", "same_state_boundary")):
        raise ValueError("reference capture did not assert the complete state boundary")
    return {"decode_input": asset, "reference_capture": reference}


def _read_npz_prefix(path: Path, member: str, shape: tuple[int, ...], dtype: np.dtype[Any], rows: int) -> np.ndarray:
    """Read only the first *rows* of a huge compressed NPY member.

    The reference cache has a 43 GiB logical array dominated by zero pages.
    NPY members are row-major, so streaming the prefix avoids materializing the
    complete cache merely to validate the two pages used by this request.
    """

    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        major, minor = np.lib.format.read_magic(stream)
        if (major, minor) == (1, 0):
            header = np.lib.format.read_array_header_1_0(stream)
        elif (major, minor) == (2, 0):
            header = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {(major, minor)} in {path}:{member}")
        actual_shape, fortran_order, actual_dtype = header
        if tuple(actual_shape) != shape or fortran_order or np.dtype(actual_dtype) != dtype:
            raise ValueError(f"unexpected cache member header for {member}: {header}")
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        raw = stream.read(rows * row_bytes)
        if len(raw) != rows * row_bytes:
            raise ValueError(f"short cache prefix for {member}: {len(raw)} bytes")
    return np.frombuffer(raw, dtype=dtype).reshape((rows, *shape[1:])).copy()


def _cache_pages(cache_root: Path, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    path = cache_root / f"rank{rank}" / "history_cache.npz"
    capture_path = cache_root / f"rank{rank}" / "capture.json"
    if not capture_path.is_file():
        raise FileNotFoundError(f"missing cache capture contract: {capture_path}")
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    cache_meta = capture.get("cache_arrays", {}).get("before_cache_0", {})
    shape = tuple(cache_meta.get("shape", ()))
    if len(shape) != 4 or tuple(shape[1:]) != (128, 1, 512):
        raise ValueError(f"unexpected cache shape in {capture_path}: {shape}")
    # The first dimension is the physical block.  Keep blocks 0 and 1; page 1
    # contains the eight history rows and the decode slot 136.
    before = _read_npz_prefix(path, "before_cache_0.npy", shape, np.dtype("<i2"), 2 * 128)
    after = _read_npz_prefix(path, "after_cache_0.npy", shape, np.dtype("<i2"), 2 * 128)
    return _bf16_storage(before), _bf16_storage(after)


def _tensor(checkpoint: DecodeSafetensorsCheckpoint, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = checkpoint.tensor(name).contiguous()
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} has dtype {value.dtype}, expected {dtype}")
    return value


def _attention_weights(checkpoint: DecodeSafetensorsCheckpoint, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"layers.{layer}.attn"
    result: dict[str, torch.Tensor] = {}
    for name, expected in (("wq_a", (1280, 5120)), ("wq_b", (32768, 1280)), ("wkv", (512, 5120))):
        raw = _tensor(checkpoint, f"{prefix}.{name}.weight", torch.int8)
        if tuple(raw.shape) != expected:
            raise ValueError(f"unexpected {name} shape: {tuple(raw.shape)}")
        offset = _tensor(checkpoint, f"{prefix}.{name}.weight_offset", torch.float32)
        if torch.count_nonzero(offset).item():
            raise ValueError(f"{name} has nonzero asymmetric offset")
        scale = _tensor(checkpoint, f"{prefix}.{name}.weight_scale", torch.float32).reshape(-1)
        result[name] = raw.transpose(0, 1).contiguous()
        result[f"{name}_scale"] = scale
    result["q_norm_weight"] = _tensor(checkpoint, f"{prefix}.q_norm.weight", torch.bfloat16)
    result["kv_norm_weight"] = _tensor(checkpoint, f"{prefix}.kv_norm.weight", torch.bfloat16)
    wo_a = _tensor(checkpoint, f"{prefix}.wo_a.weight", torch.bfloat16)
    wo_b = _tensor(checkpoint, f"{prefix}.wo_b.weight", torch.bfloat16)
    result["wo_a"] = wo_a.reshape(8, O_RANK, WIDTH // 8).permute(0, 2, 1).contiguous()
    result["wo_b"] = wo_b.transpose(0, 1).contiguous()
    result["sink"] = _tensor(checkpoint, f"{prefix}.attn_sink", torch.float32)
    return result


def _layer_inputs(asset_root: Path, cache_root: Path, checkpoint_root: Path, layer: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    boundary = _validate_reference_boundary(asset_root, cache_root)
    decode_input = boundary["decode_input"]
    position_ids = [int(value) for value in decode_input["position_ids"]]
    if len(position_ids) != 1:
        raise ValueError(f"the current EP2 runner expects one decode row, got {position_ids}")
    decode_position = position_ids[0]
    decode_token = int(decode_input["decode_token"])
    history_length = int(decode_input["history_token_length"])
    slot_mapping = decode_input["slot_mapping"]
    if len(slot_mapping) != EP_SIZE:
        raise ValueError(f"slot_mapping must contain {EP_SIZE} rank records")
    slots_from_contract: list[tuple[int, int]] = []
    for rank in range(EP_SIZE):
        rank_slots = slot_mapping[rank]
        if not rank_slots or not rank_slots[0]:
            raise ValueError(f"rank {rank} has no decode slot mapping")
        physical_slot = int(rank_slots[0][0])
        slots_from_contract.append((physical_slot // 128, physical_slot % 128))
    if any(page < 0 or page >= 2 for page, _ in slots_from_contract):
        raise ValueError(f"EP2 fixture only supports the two captured cache pages: {slots_from_contract}")
    checkpoint = DecodeSafetensorsCheckpoint(checkpoint_root)
    attention = _attention_weights(checkpoint, layer)
    router = load_router(checkpoint, layer)
    shared = load_shared_expert(checkpoint, layer)
    captures = [_capture(asset_root, rank, "layer_capture.npz") for rank in range(EP_SIZE)]
    moes = [_capture(asset_root, rank, "moe_capture.npz") for rank in range(EP_SIZE)]
    cache_before, cache_after = zip(*(_cache_pages(cache_root, rank) for rank in range(EP_SIZE)))

    def stacked(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0).expand(EP_SIZE, *value.shape).contiguous()

    inputs: dict[str, torch.Tensor] = {
        "x_hc": torch.zeros((EP_SIZE, T, HC, HIDDEN), dtype=torch.bfloat16),
        "attn_hc_function": stacked(_tensor(checkpoint, f"layers.{layer}.hc_attn_fn", torch.float32)),
        "attn_hc_scale": stacked(_tensor(checkpoint, f"layers.{layer}.hc_attn_scale", torch.float32)),
        "attn_hc_base": stacked(_tensor(checkpoint, f"layers.{layer}.hc_attn_base", torch.float32)),
        "attn_incoming": torch.zeros((EP_SIZE, T, HC), dtype=torch.float32),
        "attn_norm_weight": stacked(_tensor(checkpoint, f"layers.{layer}.attn_norm.weight", torch.bfloat16)),
        "wq_a": stacked(attention["wq_a"]), "wq_a_scale": stacked(attention["wq_a_scale"]),
        "q_norm_weight": stacked(attention["q_norm_weight"]), "wq_b": stacked(attention["wq_b"]),
        "wq_b_scale": stacked(attention["wq_b_scale"]), "wkv": stacked(attention["wkv"]),
        "wkv_scale": stacked(attention["wkv_scale"]), "kv_norm_weight": stacked(attention["kv_norm_weight"]),
        "wo_a": stacked(attention["wo_a"]), "wo_b": stacked(attention["wo_b"]),
        "rope_cos": torch.zeros((EP_SIZE, T, ROPE_PAIRS), dtype=torch.float32),
        "rope_sin": torch.zeros((EP_SIZE, T, ROPE_PAIRS), dtype=torch.float32),
        "slots": torch.zeros((EP_SIZE, T, 2), dtype=torch.int64),
        "indices": torch.full((EP_SIZE, T, WINDOW), -1, dtype=torch.int32),
        "sink": stacked(attention["sink"]),
        "positions": torch.zeros((EP_SIZE, T), dtype=torch.int64),
        "query_latent": torch.zeros((EP_SIZE, T, LATENT), dtype=torch.bfloat16),
        "query": torch.zeros((EP_SIZE, T, WIDTH), dtype=torch.bfloat16),
        "window_kv": torch.zeros((EP_SIZE, T, 512), dtype=torch.bfloat16),
        "cache": torch.stack((cache_before[0], cache_before[1])),
        "cache_lens": torch.zeros((EP_SIZE, 2), dtype=torch.int32),
        "attn_output": torch.zeros((EP_SIZE, T, HIDDEN), dtype=torch.bfloat16),
        "attn_next_pre": torch.zeros((EP_SIZE, T, HC), dtype=torch.float32),
        "attn_post": torch.zeros((EP_SIZE, T, HC), dtype=torch.float32),
        "attn_comb": torch.zeros((EP_SIZE, T, HC, HC), dtype=torch.float32),
        "attn_collapsed": torch.zeros((EP_SIZE, T, HIDDEN), dtype=torch.bfloat16),
        "attn_normed": torch.zeros((EP_SIZE, T, HIDDEN), dtype=torch.bfloat16),
        "after_attention": torch.zeros((EP_SIZE, T, HC, HIDDEN), dtype=torch.bfloat16),
        "ffn_hc_function": stacked(_tensor(checkpoint, f"layers.{layer}.hc_ffn_fn", torch.float32)),
        "ffn_hc_scale": stacked(_tensor(checkpoint, f"layers.{layer}.hc_ffn_scale", torch.float32)),
        "ffn_hc_base": stacked(_tensor(checkpoint, f"layers.{layer}.hc_ffn_base", torch.float32)),
        "ffn_norm_weight": stacked(_tensor(checkpoint, f"layers.{layer}.ffn_norm.weight", torch.bfloat16)),
        "ffn_next_pre": torch.zeros((EP_SIZE, T, HC), dtype=torch.float32),
        "ffn_post": torch.zeros((EP_SIZE, T, HC), dtype=torch.float32),
        "ffn_comb": torch.zeros((EP_SIZE, T, HC, HC), dtype=torch.float32),
        "ffn_collapsed": torch.zeros((EP_SIZE, T, HIDDEN), dtype=torch.bfloat16),
        "ffn_normed": torch.zeros((EP_SIZE, T, HIDDEN), dtype=torch.bfloat16),
        "gate_weight": stacked(router["gate_weight"]), "text_bias": stacked(router["text_bias"]),
        "vision_bias": stacked(router["vision_bias"]), "input_ids": torch.zeros((EP_SIZE, T), dtype=torch.int64),
        "routed_w1": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D), dtype=torch.int8),
        "routed_w1_scale": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER), dtype=torch.float32),
        "routed_w3": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D), dtype=torch.int8),
        "routed_w3_scale": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER), dtype=torch.float32),
        "routed_w2": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, D, 2560), dtype=torch.int8),
        "routed_w2_scale": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, D), dtype=torch.float32),
        "shared_gate_up": stacked(shared["shared_gate_up"]), "shared_gate_up_scale": stacked(shared["shared_gate_up_scale"]),
        "shared_down": stacked(shared["shared_down"]), "shared_down_scale": stacked(shared["shared_down_scale"]),
        "logits": torch.zeros((EP_SIZE, T, N_EXPERTS), dtype=torch.float32),
        "route_ids": torch.zeros((EP_SIZE, T, TOPK), dtype=torch.int32),
        "route_weights": torch.zeros((EP_SIZE, T, TOPK), dtype=torch.float32),
        "moe_output": torch.zeros((EP_SIZE, T, D), dtype=torch.bfloat16),
        "output": torch.zeros((EP_SIZE, T, HC, HIDDEN), dtype=torch.bfloat16),
    }
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64.0))
    angle = decode_position * frequencies
    rope_cos = torch.cos(angle)
    rope_sin = torch.sin(angle)
    for rank in range(EP_SIZE):
        cap = captures[rank]
        moe = moes[rank]
        inputs["x_hc"][rank, 0] = _bf16_storage(cap["input_bf16"])[0]
        inputs["attn_incoming"][rank, 0] = torch.from_numpy(cap["pre_mix_fp32"])[0]
        inputs["positions"][rank, 0] = decode_position
        inputs["slots"][rank, 0] = torch.tensor(slots_from_contract[rank], dtype=torch.int64)
        physical_slot = slots_from_contract[rank][0] * 128 + slots_from_contract[rank][1]
        first_history = max(0, physical_slot - history_length)
        inputs["indices"][rank, 0, : history_length + 1] = torch.arange(
            first_history, physical_slot + 1, dtype=torch.int32
        )
        inputs["input_ids"][rank, 0] = decode_token
        shard = load_routed_shard(checkpoint, layer, rank)
        inputs["routed_w1"][rank] = unpack_nz(shard["routed_w1"].reshape(N_LOCAL_EXPERTS, MOE_INTER, D))
        inputs["routed_w3"][rank] = unpack_nz(shard["routed_w3"].reshape(N_LOCAL_EXPERTS, MOE_INTER, D))
        inputs["routed_w2"][rank] = unpack_nz(shard["routed_w2"].reshape(N_LOCAL_EXPERTS, D, 2560))
        inputs["routed_w1_scale"][rank] = shard["routed_w1_scale"]
        inputs["routed_w3_scale"][rank] = shard["routed_w3_scale"]
        inputs["routed_w2_scale"][rank] = shard["routed_w2_scale"]
        del shard
        inputs["rope_cos"][rank, 0] = rope_cos
        inputs["rope_sin"][rank, 0] = rope_sin

    expected = {
        "cache": torch.stack((cache_after[0], cache_after[1])),
        "positions": inputs["positions"].clone(),
        "cache_lens": inputs["cache_lens"].clone(),
    }
    for rank in range(EP_SIZE):
        cap = captures[rank]
        moe = moes[rank]
        expected.setdefault("output", torch.zeros_like(inputs["output"]))[rank, 0] = _bf16_storage(cap["output_bf16"])[0]
        expected.setdefault("ffn_next_pre", torch.zeros_like(inputs["ffn_next_pre"]))[rank, 0] = torch.from_numpy(cap["next_pre_fp32"])[0]
        expected.setdefault("moe_output", torch.zeros_like(inputs["moe_output"]))[rank, 0] = _bf16_storage(moe["output_bf16"])[0]
        expected.setdefault("route_ids", torch.zeros_like(inputs["route_ids"]))[rank, 0] = torch.from_numpy(moe["route_ids"])[0]
        expected.setdefault("route_weights", torch.zeros_like(inputs["route_weights"]))[rank, 0] = torch.from_numpy(moe["route_weights"])[0]
        expected.setdefault("logits", torch.zeros_like(inputs["logits"]))[rank, 0] = torch.from_numpy(moe["router_logits"])[0]
        expected.setdefault("ffn_normed", torch.zeros_like(inputs["ffn_normed"]))[rank, 0] = torch.from_numpy(
            moe["router_input_fp32"]
        )[0].to(torch.bfloat16)
    # The composition exposes diagnostic InOut buffers whose reference capture
    # is not part of this layer contract.  Keep complete golden files for the
    # harness and mark those buffers as observationally out of scope below.
    for name, value in inputs.items():
        expected.setdefault(name, value.clone())
    return inputs, expected


def _specs() -> list[TensorSpec | ScalarSpec]:
    shapes = {
        "x_hc": ([EP_SIZE, T, HC, HIDDEN], torch.bfloat16), "attn_hc_function": ([EP_SIZE, CHANNELS, HC * HIDDEN], torch.float32),
        "attn_hc_scale": ([EP_SIZE, 3], torch.float32), "attn_hc_base": ([EP_SIZE, CHANNELS], torch.float32), "attn_incoming": ([EP_SIZE, T, HC], torch.float32),
        "attn_norm_weight": ([EP_SIZE, HIDDEN], torch.bfloat16), "wq_a": ([EP_SIZE, HIDDEN, LATENT], torch.int8), "wq_a_scale": ([EP_SIZE, LATENT], torch.float32),
        "q_norm_weight": ([EP_SIZE, LATENT], torch.bfloat16), "wq_b": ([EP_SIZE, LATENT, WIDTH], torch.int8), "wq_b_scale": ([EP_SIZE, WIDTH], torch.float32),
        "wkv": ([EP_SIZE, HIDDEN, 512], torch.int8), "wkv_scale": ([EP_SIZE, 512], torch.float32), "kv_norm_weight": ([EP_SIZE, 512], torch.bfloat16),
        "wo_a": ([EP_SIZE, 8, WIDTH // 8, O_RANK], torch.bfloat16), "wo_b": ([EP_SIZE, 8 * O_RANK, HIDDEN], torch.bfloat16),
        "rope_cos": ([EP_SIZE, T, ROPE_PAIRS], torch.float32), "rope_sin": ([EP_SIZE, T, ROPE_PAIRS], torch.float32), "slots": ([EP_SIZE, T, 2], torch.int64),
        "indices": ([EP_SIZE, T, WINDOW], torch.int32), "sink": ([EP_SIZE, 64], torch.float32), "positions": ([EP_SIZE, T], torch.int64),
        "query_latent": ([EP_SIZE, T, LATENT], torch.bfloat16), "query": ([EP_SIZE, T, WIDTH], torch.bfloat16), "window_kv": ([EP_SIZE, T, 512], torch.bfloat16),
        "cache": ([EP_SIZE, 2, 128, 1, 512], torch.bfloat16), "cache_lens": ([EP_SIZE, 2], torch.int32), "attn_output": ([EP_SIZE, T, HIDDEN], torch.bfloat16),
        "attn_next_pre": ([EP_SIZE, T, HC], torch.float32), "attn_post": ([EP_SIZE, T, HC], torch.float32), "attn_comb": ([EP_SIZE, T, HC, HC], torch.float32),
        "attn_collapsed": ([EP_SIZE, T, HIDDEN], torch.bfloat16), "attn_normed": ([EP_SIZE, T, HIDDEN], torch.bfloat16), "after_attention": ([EP_SIZE, T, HC, HIDDEN], torch.bfloat16),
        "ffn_hc_function": ([EP_SIZE, CHANNELS, HC * HIDDEN], torch.float32), "ffn_hc_scale": ([EP_SIZE, 3], torch.float32), "ffn_hc_base": ([EP_SIZE, CHANNELS], torch.float32),
        "ffn_norm_weight": ([EP_SIZE, HIDDEN], torch.bfloat16), "ffn_next_pre": ([EP_SIZE, T, HC], torch.float32), "ffn_post": ([EP_SIZE, T, HC], torch.float32),
        "ffn_comb": ([EP_SIZE, T, HC, HC], torch.float32), "ffn_collapsed": ([EP_SIZE, T, HIDDEN], torch.bfloat16), "ffn_normed": ([EP_SIZE, T, HIDDEN], torch.bfloat16),
        "gate_weight": ([EP_SIZE, N_EXPERTS, D], torch.float32), "text_bias": ([EP_SIZE, N_EXPERTS], torch.float32), "vision_bias": ([EP_SIZE, N_EXPERTS], torch.float32),
        "input_ids": ([EP_SIZE, T], torch.int64), "routed_w1": ([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8), "routed_w1_scale": ([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], torch.float32),
        "routed_w3": ([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8), "routed_w3_scale": ([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], torch.float32),
        "routed_w2": ([EP_SIZE, N_LOCAL_EXPERTS, D, 2560], torch.int8), "routed_w2_scale": ([EP_SIZE, N_LOCAL_EXPERTS, D], torch.float32),
        "shared_gate_up": ([EP_SIZE, D, MOE_INTER * 2], torch.int8), "shared_gate_up_scale": ([EP_SIZE, MOE_INTER * 2], torch.float32),
        "shared_down": ([EP_SIZE, MOE_INTER, D], torch.int8), "shared_down_scale": ([EP_SIZE, D], torch.float32), "logits": ([EP_SIZE, T, N_EXPERTS], torch.float32),
        "route_ids": ([EP_SIZE, T, TOPK], torch.int32), "route_weights": ([EP_SIZE, T, TOPK], torch.float32), "moe_output": ([EP_SIZE, T, D], torch.bfloat16), "output": ([EP_SIZE, T, HC, HIDDEN], torch.bfloat16),
    }
    return [TensorSpec(name, shape, dtype) for name, (shape, dtype) in shapes.items()] + [ScalarSpec("active", torch.int32, 1), ScalarSpec("epoch", torch.int32, 1)]


def _compare(name: str, expected: torch.Tensor):
    def compare(actual: torch.Tensor, _expected: torch.Tensor, **_: Any) -> tuple[bool, str]:
        if name == "route_ids":
            ok = torch.equal(actual[:, 0], expected[:, 0])
        else:
            ok = bool(torch.isclose(actual, expected, rtol=0.05, atol=0.05).all())
        return (ok, "" if ok else f"{name} mismatch at rtol=0.05 atol=0.05")
    return compare


def _write_fixture(
    root: Path,
    inputs: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    layer_id: int,
) -> None:
    if root.exists():
        import shutil
        shutil.rmtree(root)
    (root / "in").mkdir(parents=True)
    (root / "out").mkdir(parents=True)
    for name, value in inputs.items():
        torch.save(value, root / "in" / f"{name}.pt")
    torch.save(torch.tensor(1, dtype=torch.int32), root / "in" / "active.pt")
    torch.save(torch.tensor(1, dtype=torch.int32), root / "in" / "epoch.pt")
    for name, value in expected.items():
        torch.save(value, root / "out" / f"{name}.pt")
    (root / "contract.json").write_text(
        json.dumps(
            {
                "status": "frozen_full_layer_inputs",
                "product_compute": "PyPTO only",
                "formal_accuracy": "not_run",
                "layer_id": layer_id,
                "topology": {"dp": 2, "tp": 1, "ep": 2},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--cache-assets", type=Path, default=DEFAULT_CACHE_ASSETS)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("DEEPSEEK_V41_CHECKPOINT", "/mnt/data/models/DeepSeek-V4.1-Flash-w8a8")))
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--layer-id", type=int, choices=TARGET_LAYER_IDS, default=0)
    parser.add_argument("--device", default=os.environ.get("TASK_DEVICE", "0,1"))
    parser.add_argument("--platform", default="a2a3", choices=("a2a3", "a5"))
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    assets, cache_assets, checkpoint = args.assets.resolve(), args.cache_assets.resolve(), args.checkpoint.resolve()
    inputs, expected = _layer_inputs(assets, cache_assets, checkpoint, args.layer_id)
    _write_fixture(args.fixture.resolve(), inputs, expected, layer_id=args.layer_id)
    if args.prepare_only:
        print(
            json.dumps(
                {"status": "fixture_ready", "layer_id": args.layer_id, "fixture": str(args.fixture.resolve())},
                indent=2,
            )
        )
        return 0
    if args.layer_id != 0:
        report = {
            "status": "not_run",
            "formal_accuracy": "not_run",
            "scope": "independent physical Decode layer",
            "layer_id": args.layer_id,
            "topology": {"dp": 2, "tp": 1, "ep": 2},
            "same_input": True,
            "same_layer": True,
            "same_topology": True,
            "same_state_boundary": True,
            "product_compute": "PyPTO only",
            "reference_assets": {"root": str(assets), "cache_root": str(cache_assets)},
            "blocker": (
                "decode_layer_ep2 is the L0 SWA composition. The selected physical layer "
                "requires its C1A/C2A cache/index path and, for L1/L14, Engram state; "
                "running the L0 kernel here would compare the wrong semantics."
            ),
            "recovery": "Add a layer-specific PyPTO composition and capture all owned cache/index/Engram state before NPU submission.",
        }
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2
    devices = [int(item) for item in args.device.split(",") if item]
    if devices != [0, 1] and len(devices) != 2:
        raise SystemExit(f"EP2 needs two allocated devices, got {devices}")
    checked = {
        "output",
        "moe_output",
        "logits",
        "route_ids",
        "route_weights",
        "cache",
        "positions",
        "cache_lens",
        "ffn_next_pre",
        "ffn_normed",
    }
    compare = {name: _compare(name, expected[name]) for name in checked}
    compare.update({name: (lambda _actual, _expected, **_kwargs: (True, "")) for name in inputs if name not in checked})
    result = run(fn=decode_layer_ep2, specs=_specs(), golden_data=str(args.fixture.resolve()), compare_fn=compare, config={"backend_type": args.platform, "execution_mode": args.sim, "ring_heap": (2 * 1024**3, 2 * 1024**3, 1024**3, 512 * 1024**2), "ring_dep_pool": 131072, "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0)}, rtol=0.05, atol=0.05)
    boundary = _validate_reference_boundary(assets, cache_assets)
    report = {
        "status": "pass" if result.passed else "fail",
        "formal_accuracy": "not_run",
        "scope": f"complete physical L{args.layer_id} layer EP2 representative case",
        "layer_id": args.layer_id,
        "topology": {"dp": 2, "tp": 1, "ep": 2, "devices": devices},
        "same_input": True,
        "same_layer": True,
        "same_topology": True,
        "same_state_boundary": True,
        "cache_state_comparison": "pass" if result.passed else "fail",
        "comparison": {
            "rtol": 0.05,
            "atol": 0.05,
            "floating_outputs": ["cache", "ffn_next_pre", "ffn_normed", "logits", "moe_output", "output", "positions", "route_weights"],
            "exact_outputs": ["route_ids"],
            "cache_scope": "rank-local pages 0 and 1, including slot row 136 and untouched rows",
        },
        "reference_assets": {
            "root": str(assets),
            "cache_root": str(cache_assets),
            "input_contract_sha256": hashlib.sha256((assets / "decode_input.json").read_bytes()).hexdigest(),
            "reference_capture_contract": boundary["reference_capture"].get("input_contract"),
        },
        "result": str(result),
        "work_dir": str(result.work_dir) if result.work_dir else None,
        "product_import_forbidden": ["vllm", "vllm_ascend", "torch_npu"],
    }
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
