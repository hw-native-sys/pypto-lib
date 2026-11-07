# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the real PyPTO DeepSeek V4.1 EP2 MoE boundary on two A3 devices.

The input boundary is frozen by the independent vLLM-Ascend capture.  This
entry loads only the host-side checkpoint ABI, dispatches ``decode_ep2`` through
the PyPTO distributed runtime, and compares route/output tensors with the
captured NPU result.  It never imports vLLM, vLLM-Ascend, or torch-npu.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from golden import ScalarSpec, TensorSpec, run
from pypto.ir import DistributedConfig

from models.deepseek_v4_1_flash_a2.decode_moe_ep2 import (
    D,
    EP_SIZE,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    T,
    TOPK,
    decode_ep2,
)
from models.deepseek_v4_1_flash_a2.decode_weight_abi import (
    load_router,
    load_routed_shard,
    load_shared_expert,
)
from models.deepseek_v4_1_flash_a2.decode_weights import DecodeSafetensorsCheckpoint
from models.deepseek_v4_1_flash_a2.utils import unpack_nz


DEFAULT_ASSETS = Path(
    ".pypto-a3-state/reference/decode_ep2_l0_20261105_v19"
)
DEFAULT_FIXTURE_ROOT = Path(".pypto-a3-state/product")
TARGET_LAYERS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)


def _bind_user_pto_isa() -> None:
    """Keep device-side runtime assembly on the writable pinned checkout.

    The queue's L3 worker can import the shared ``simpler_setup`` package even
    when the host process sees the per-session overlay.  Its resolver then
    tries to create a lock beside the read-only shared wheel.  The activation
    helper has already verified the user checkout; bind both resolver entry
    points to that checkout before the distributed dispatch is assembled.
    """

    state = os.environ.get("PYPTO_A3_STATE_DIR")
    if not state:
        return
    overlay = Path(state).resolve() / "python" / "simpler_setup" / "_assets" / "build" / "pto-isa"
    if not overlay.is_dir():
        return
    resolved = str(overlay.resolve())
    import pypto.runtime.pto_isa as pypto_pto_isa
    import simpler_setup.pto_isa as simpler_pto_isa

    pypto_pto_isa._resolve_pinned_pto_isa_root.cache_clear()
    pypto_pto_isa._resolve_pinned_pto_isa_root = lambda: resolved
    simpler_pto_isa.ensure_pto_isa_root = lambda verbose=False: resolved
    os.environ["PTO_ISA_ROOT"] = resolved


def _bf16_from_storage(value: np.ndarray) -> torch.Tensor:
    if value.dtype != np.int16:
        raise ValueError(f"BF16 capture storage must be int16, got {value.dtype}")
    return torch.from_numpy(np.ascontiguousarray(value)).view(torch.bfloat16)


def _read_moe_capture(asset_root: Path, rank: int) -> dict[str, torch.Tensor]:
    record = json.loads((asset_root / f"rank{rank}" / "capture.json").read_text(encoding="utf-8"))
    metadata = record.get("moe_capture")
    if not isinstance(metadata, dict) or not metadata.get("file"):
        raise RuntimeError(
            f"rank {rank} capture has no MoE boundary; rerun the independent capture "
            "with the current capture driver"
        )
    path = asset_root / str(metadata["file"]).split(f"rank{rank}/", 1)[-1]
    if not path.is_file():
        path = asset_root / f"rank{rank}" / Path(str(metadata["file"])).name
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        required = {
            "input_bf16",
            "router_input_fp32",
            "output_bf16",
            "route_ids",
            "route_weights",
            "router_logits",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"{path} is missing capture arrays: {missing}")
        return {
            "value": _bf16_from_storage(archive["input_bf16"]),
            "router_value": torch.from_numpy(np.ascontiguousarray(archive["router_input_fp32"])).float(),
            "reference_output": _bf16_from_storage(archive["output_bf16"]),
            "reference_route_ids": torch.from_numpy(np.ascontiguousarray(archive["route_ids"])).to(torch.int32),
            "reference_route_weights": torch.from_numpy(
                np.ascontiguousarray(archive["route_weights"])
            ).float(),
            "reference_logits": torch.from_numpy(np.ascontiguousarray(archive["router_logits"])).float(),
            "input_ids": torch.tensor(record.get("input_ids", ()), dtype=torch.int64),
        }


def _load_inputs(asset_root: Path, checkpoint_root: Path, layer: int) -> dict[str, torch.Tensor]:
    checkpoint = DecodeSafetensorsCheckpoint(checkpoint_root)
    captured = [_read_moe_capture(asset_root, rank) for rank in range(EP_SIZE)]
    value = torch.zeros((EP_SIZE, T, D), dtype=torch.bfloat16)
    router_value = torch.zeros((EP_SIZE, T, D), dtype=torch.bfloat16)
    input_ids = torch.zeros((EP_SIZE, T), dtype=torch.int64)
    route_ids = torch.zeros((EP_SIZE, T, TOPK), dtype=torch.int32)
    route_weights = torch.zeros((EP_SIZE, T, TOPK), dtype=torch.float32)
    reference_logits = torch.zeros((EP_SIZE, T, N_EXPERTS), dtype=torch.float32)
    reference_output = torch.zeros((EP_SIZE, T, D), dtype=torch.bfloat16)

    input_contract = json.loads((asset_root / "decode_input.json").read_text(encoding="utf-8"))
    token = int(input_contract["decode_token"])
    active_counts: list[int] = []
    for rank, item in enumerate(captured):
        rows = int(item["value"].reshape(-1, D).shape[0])
        if rows < 1 or rows > T:
            raise ValueError(f"rank {rank} MoE input row count must be in [1,{T}], got {rows}")
        active_counts.append(rows)
        value[rank, :rows] = item["value"].reshape(-1, D)[:rows]
        router_value[rank, :rows] = item["router_value"].reshape(-1, D)[:rows].to(torch.bfloat16)
        reference_output[rank, :rows] = item["reference_output"].reshape(-1, D)[:rows]
        route_ids[rank, :rows] = item["reference_route_ids"].reshape(-1, TOPK)[:rows]
        route_weights[rank, :rows] = item["reference_route_weights"].reshape(-1, TOPK)[:rows]
        reference_logits[rank, :rows] = item["reference_logits"].reshape(-1, N_EXPERTS)[:rows]
        captured_ids = item["input_ids"].reshape(-1)
        if captured_ids.numel() >= rows:
            input_ids[rank, :rows] = captured_ids[:rows]
        else:
            input_ids[rank, :rows] = token

    router = load_router(checkpoint, layer)
    routed_target = {
        "routed_w1": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D), dtype=torch.int8),
        "routed_w1_scale": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER), dtype=torch.float32),
        "routed_w3": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D), dtype=torch.int8),
        "routed_w3_scale": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER), dtype=torch.float32),
        "routed_w2": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, D, 2560), dtype=torch.int8),
        "routed_w2_scale": torch.empty((EP_SIZE, N_LOCAL_EXPERTS, D), dtype=torch.float32),
    }
    for rank in range(EP_SIZE):
        shard = load_routed_shard(checkpoint, layer, rank)
        for name, target in routed_target.items():
            shard_value = shard[name]
            if name in {"routed_w1", "routed_w3"}:
                shard_value = unpack_nz(shard_value.reshape(N_LOCAL_EXPERTS, MOE_INTER, D))
            elif name == "routed_w2":
                shard_value = unpack_nz(shard_value.reshape(N_LOCAL_EXPERTS, D, 2560))
            target[rank].copy_(shard_value)
        del shard

    shared = load_shared_expert(checkpoint, layer)
    if len(set(active_counts)) != 1:
        raise ValueError(
            "the current EP2 product ABI uses one shared active row count; "
            f"reference ranks disagree: {active_counts}"
        )
    result = {
        "value": value,
        "router_value": router_value,
        "gate_weight": router["gate_weight"].unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous(),
        "text_bias": router["text_bias"].unsqueeze(0).expand(EP_SIZE, -1).contiguous(),
        "vision_bias": router["vision_bias"].unsqueeze(0).expand(EP_SIZE, -1).contiguous(),
        "input_ids": input_ids,
        **routed_target,
        "shared_gate_up": shared["shared_gate_up"].unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous(),
        "shared_gate_up_scale": shared["shared_gate_up_scale"].unsqueeze(0).expand(EP_SIZE, -1).contiguous(),
        "shared_down": shared["shared_down"].unsqueeze(0).expand(EP_SIZE, -1, -1).contiguous(),
        "shared_down_scale": shared["shared_down_scale"].unsqueeze(0).expand(EP_SIZE, -1).contiguous(),
        "logits": reference_logits,
        "route_ids": route_ids,
        "route_weights": route_weights,
        "output": reference_output,
        "active": torch.tensor(active_counts[0], dtype=torch.int32),
        "epoch": torch.tensor(1, dtype=torch.int32),
    }
    return result


def _write_fixture(root: Path, values: dict[str, torch.Tensor]) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "in").mkdir(parents=True)
    (root / "out").mkdir(parents=True)
    input_names = {
        "value",
        "router_value",
        "gate_weight",
        "text_bias",
        "vision_bias",
        "input_ids",
        "routed_w1",
        "routed_w1_scale",
        "routed_w3",
        "routed_w3_scale",
        "routed_w2",
        "routed_w2_scale",
        "shared_gate_up",
        "shared_gate_up_scale",
        "shared_down",
        "shared_down_scale",
    }
    for name, value in values.items():
        if name in input_names or name in {"active", "epoch"}:
            torch.save(value, root / "in" / f"{name}.pt")
    for name in ("logits", "route_ids", "route_weights", "output"):
        torch.save(values[name], root / "out" / f"{name}.pt")
    (root / "contract.json").write_text(
        json.dumps(
            {
                "status": "frozen_reference_inputs",
                "source": "independent_vllm_ascend_npu_capture",
                "product_compute": "PyPTO only",
                "active_rows": int(values["active"].item()),
                "topology": {"dp": 2, "tp": 1, "ep": 2},
                "outputs": ["logits", "route_ids", "route_weights", "output"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _output_compare(actual: torch.Tensor, expected: torch.Tensor, **kwargs: Any) -> tuple[bool, str]:
    """Persist the device output while retaining the formal allclose gate."""

    del kwargs
    dump_root = os.environ.get("PYPTO_EP2_DUMP_ACTUAL")
    if dump_root:
        dump_path = Path(dump_root)
        dump_path.mkdir(parents=True, exist_ok=True)
        torch.save(actual, dump_path / "output.pt")
    close = torch.isclose(actual, expected, rtol=0.05, atol=0.05)
    mismatch = int((~close).sum().item())
    if mismatch == 0:
        return True, ""
    return False, f"    Mismatched elements: {mismatch}/{actual.numel()} rtol=0.05 atol=0.05"


def _specs() -> list[TensorSpec | ScalarSpec]:
    return [
        TensorSpec("value", [EP_SIZE, T, D], torch.bfloat16),
        TensorSpec("router_value", [EP_SIZE, T, D], torch.bfloat16),
        TensorSpec("gate_weight", [EP_SIZE, N_EXPERTS, D], torch.float32),
        TensorSpec("text_bias", [EP_SIZE, N_EXPERTS], torch.float32),
        TensorSpec("vision_bias", [EP_SIZE, N_EXPERTS], torch.float32),
        TensorSpec("input_ids", [EP_SIZE, T], torch.int64),
        TensorSpec("routed_w1", [EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8),
        TensorSpec("routed_w1_scale", [EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], torch.float32),
        TensorSpec("routed_w3", [EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8),
        TensorSpec("routed_w3_scale", [EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], torch.float32),
        TensorSpec("routed_w2", [EP_SIZE, N_LOCAL_EXPERTS, D, 2560], torch.int8),
        TensorSpec("routed_w2_scale", [EP_SIZE, N_LOCAL_EXPERTS, D], torch.float32),
        TensorSpec("shared_gate_up", [EP_SIZE, D, MOE_INTER * 2], torch.int8),
        TensorSpec("shared_gate_up_scale", [EP_SIZE, MOE_INTER * 2], torch.float32),
        TensorSpec("shared_down", [EP_SIZE, MOE_INTER, D], torch.int8),
        TensorSpec("shared_down_scale", [EP_SIZE, D], torch.float32),
        TensorSpec("logits", [EP_SIZE, T, N_EXPERTS], torch.float32),
        TensorSpec("route_ids", [EP_SIZE, T, TOPK], torch.int32),
        TensorSpec("route_weights", [EP_SIZE, T, TOPK], torch.float32),
        TensorSpec("output", [EP_SIZE, T, D], torch.bfloat16),
        ScalarSpec("active", torch.int32, 1),
        ScalarSpec("epoch", torch.int32, 1),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("DEEPSEEK_V41_CHECKPOINT", "/mnt/data/models/DeepSeek-V4.1-Flash-w8a8")))
    parser.add_argument(
        "--fixture",
        type=Path,
        help="per-layer golden fixture; defaults to decode_ep2_l<layer>_moe_fixture_v1",
    )
    parser.add_argument("--layer", type=int, choices=TARGET_LAYERS, default=0)
    parser.add_argument(
        "--device",
        default=os.environ.get("TASK_DEVICE", "0,1"),
        help="comma-separated logical devices; defaults to the queue's TASK_DEVICE",
    )
    parser.add_argument("-p", "--platform", default="a2a3", choices=("a2a3", "a5"))
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)

    _bind_user_pto_isa()

    assets = args.assets.resolve()
    checkpoint = args.checkpoint.resolve()
    boundary = json.loads((assets / "decode_input.json").read_text(encoding="utf-8"))
    scenario = str(boundary.get("scenario", "single"))
    fixture = (args.fixture or DEFAULT_FIXTURE_ROOT / f"decode_ep2_l{args.layer}_moe_fixture_v1").resolve()
    values = _load_inputs(assets, checkpoint, args.layer)
    _write_fixture(fixture, values)
    del values
    if args.prepare_only:
        print(json.dumps({"status": "fixture_ready", "fixture": str(fixture)}, indent=2))
        return 0

    devices = [int(item) for item in args.device.split(",") if item]
    if len(devices) != EP_SIZE:
        raise SystemExit(f"EP2 needs exactly two devices, got {devices}")
    specs = _specs()
    result = run(
        fn=decode_ep2,
        specs=specs,
        golden_data=str(fixture),
        compare_fn={"output": _output_compare},
        config={
            "backend_type": args.platform,
            "execution_mode": args.sim,
            # The EP2 routed expert live-set is larger than the runtime's
            # default 256 MiB ring.  Match the A3 Decode runtime allocation
            # used by the sibling DeepSeek entry while keeping the product
            # path PyPTO-only.
            "ring_heap": (2 * 1024**3, 2 * 1024**3, 1024**3, 512 * 1024**2),
            "ring_dep_pool": 131072,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
        },
        rtol=0.05,
        atol=0.05,
    )
    report = {
        "status": "pass" if result.passed else "fail",
        "formal_accuracy": "not_run",
        "scope": f"MoE boundary only for physical L{args.layer}; complete physical layer remains not_run",
        "layer_id": args.layer,
        "scenario": scenario,
        "topology": {"dp": 2, "tp": 1, "ep": 2, "devices": devices},
        "assets": str(assets),
        "checkpoint": str(checkpoint),
        "fixture": str(fixture),
        "same_input": True,
        "same_layer": True,
        "same_topology": True,
        "route_and_output_comparison": "pass" if result.passed else "fail",
        "cache_state_comparison": "not_applicable_at_moe_boundary",
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
