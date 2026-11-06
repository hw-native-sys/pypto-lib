# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inventory independent Decode stage factories for the eleven target layers.

This verifier is intentionally a no-device preparation step.  It constructs
the available PyPTO stage factories at the production Decode capacity and
records which layers still lack a complete mHC/attention/cache/MoE entrypoint.
It never executes a model and never imports vLLM, vLLM-Ascend, or torch-npu.
The resulting report is setup evidence only; all reference, NPU, and accuracy
fields remain ``not_run`` until a same-input device replay is completed.
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


TARGET_LAYERS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)
CAPACITY = 16


def _factory_specs() -> dict[int, dict[str, Any]]:
    """Return lazy module/factory descriptions without importing PyPTO yet."""

    return {
        0: {
            "mode": "swa_mhc_moe",
            "module": "models.deepseek_v4_1_flash_a2.decode_layer_ep2",
            "factory": None,
            "full_entrypoint": "models.deepseek_v4_1_flash_a2.decode_layer_ep2.decode_layer_ep2",
            "compile_entrypoint": "models.deepseek_v4_1_flash_a2.decode_layer_ep2_compile",
        },
        1: {
            "mode": "swa_engram_moe",
            "module": "models.deepseek_v4_1_flash_a2.engram",
            "factory": None,
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        2: {
            "mode": "c2a_full",
            "module": "models.deepseek_v4_1_flash_a2.layer2_c1c2",
            "factory": "make_c2_index_publish_inline",
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        3: {
            "mode": "c2a_reuse",
            "module": "models.deepseek_v4_1_flash_a2.layers19_25",
            "factory": None,
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        14: {
            "mode": "c2a_full",
            "module": "models.deepseek_v4_1_flash_a2.decode_c2a_full",
            "factory": "make_stages",
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        15: {
            "mode": "c2a_reuse",
            "module": "models.deepseek_v4_1_flash_a2.decode_c2a_reuse",
            "factory": "make_stages",
            "factory_args": (15,),
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        19: {
            "mode": "c2a_reuse",
            "module": "models.deepseek_v4_1_flash_a2.decode_c2a_reuse",
            "factory": "make_stages",
            "factory_args": (19,),
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        20: {
            "mode": "c1a_full",
            "module": "models.deepseek_v4_1_flash_a2.decode_c1a_full",
            "factory": "make_stages",
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        21: {
            "mode": "c1a_reuse",
            "module": "models.deepseek_v4_1_flash_a2.decode_c1a_reuse",
            "factory": "make_stages",
            "factory_args": (21,),
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        24: {
            "mode": "c1a_reindex",
            "module": "models.deepseek_v4_1_flash_a2.decode_c1a_reindex",
            "factory": "make_stages",
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
        25: {
            "mode": "c1a_reuse",
            "module": "models.deepseek_v4_1_flash_a2.decode_c1a_reuse",
            "factory": "make_stages",
            "factory_args": (25,),
            "full_entrypoint": None,
            "compile_entrypoint": None,
        },
    }


def _construct(spec: dict[str, Any], capacity: int) -> tuple[str, list[str], str | None]:
    module = importlib.import_module(str(spec["module"]))
    factory_name = spec.get("factory")
    if factory_name is None:
        return "not_applicable", [], None
    factory = getattr(module, factory_name)
    args = tuple(spec.get("factory_args", ()))
    if spec["mode"] == "c2a_full" and spec["module"].endswith("layer2_c1c2"):
        from models.deepseek_v4_1_flash_a2.config import FLASH

        args = (asdict(FLASH), *args)
    stages = factory(capacity, *args)
    if callable(stages):
        return "pass", ["publish"], spec["mode"]
    if not isinstance(stages, dict):
        raise TypeError(f"{spec['module']}.{factory_name} returned {type(stages).__name__}")
    names = sorted(str(name) for name in stages if name != "contract")
    contract = stages.get("contract")
    mode = getattr(contract, "mode", None) or spec["mode"]
    return "pass", names, str(mode)


def inventory(capacity: int = CAPACITY) -> dict[str, Any]:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    rows: dict[str, Any] = {}
    for layer in TARGET_LAYERS:
        spec = _factory_specs()[layer]
        row: dict[str, Any] = {
            "layer_id": layer,
            "mode": spec["mode"],
            "capacity": capacity,
            "module": spec["module"],
            "full_entrypoint": spec["full_entrypoint"],
            "compile_entrypoint": spec["compile_entrypoint"],
            "stage_factory": "not_configured" if spec["factory"] is None else f"{spec['module']}.{spec['factory']}",
            "stage_construct": "not_run",
            "stage_names": [],
            "resolved_mode": None,
            "full_layer_status": "not_run",
            "reference_status": "not_run",
            "npu_status": "not_run",
            "accuracy_status": "not_run",
        }
        try:
            status, names, resolved_mode = _construct(spec, capacity)
            row["stage_construct"] = status
            row["stage_names"] = names
            row["resolved_mode"] = resolved_mode
            if spec["full_entrypoint"] is None:
                row["full_layer_status"] = "missing_full_layer_entrypoint"
        except Exception as error:  # report construction failures without hiding the layer row
            row["stage_construct"] = "fail"
            row["construction_error"] = f"{type(error).__name__}: {error}"
        rows[str(layer)] = row
    return {
        "status": "setup_only",
        "scope": "independent physical Decode layer stages",
        "capacity": capacity,
        "target_layers": list(TARGET_LAYERS),
        "product_compute": "PyPTO only",
        "reference_process": "independent vLLM-Ascend process; not imported",
        "npu_execution": "not_run",
        "accuracy": "not_run",
        "layers": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    result = inventory(args.capacity)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if all(row["stage_construct"] in ("pass", "not_applicable") for row in result["layers"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
