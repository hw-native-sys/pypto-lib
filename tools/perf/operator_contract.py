# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only workload, allocation and result contracts for operator tracking."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).parent / "suites/dsv4_operators.json"
CASE_IDS = {
    f"{model}-{op}" for model in ("mtp", "dspark")
    for op in ("attention-csa", "attention-hca", "attention-swa", "moe-ep8", "lm-head")
}
SAMPLING = {"seed": 1807, "rounds": 100, "warmup": 5, "raw": True}


class ContractError(ValueError):
    """Evidence is missing or incompatible with the selected workload."""


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def devices(value):
    try:
        parsed = [int(item) for item in value.split(",")] if isinstance(value, str) else value
        if not isinstance(parsed, list) or not parsed:
            raise ValueError
        if any(type(item) is not int or item < 0 or item % 2 for item in parsed):
            raise ValueError
        if len(set(parsed)) != len(parsed) or parsed != sorted(parsed):
            raise ValueError
    except (TypeError, ValueError):
        raise ContractError("devices must be distinct, ascending, non-negative even host IDs") from None
    return parsed


def load_profile(path):
    profile = read_json(path)
    if profile.get("schema_version") != 1 or profile.get("device_id_domain") != "host":
        raise ContractError("only schema 1 host device IDs are supported; remapping is not supported")
    epoch = profile.get("device_epoch")
    if not isinstance(epoch, str) or not epoch.strip():
        raise ContractError("a device epoch is required")
    for count in (1, 4, 8):
        selected = devices(profile.get("sequences", {}).get(str(count)))
        if len(selected) != count:
            raise ContractError(f"the {count}-device sequence has the wrong size")
    if not set(profile["sequences"]["1"] + profile["sequences"]["4"]) <= set(profile["sequences"]["8"]):
        raise ContractError("single-card and TP4 devices must be subsets of the EP8 allocation")
    return profile


def validate_allocation(profile, allocated, selected, environ=None):
    environ = os.environ if environ is None else environ
    allocation = devices(allocated)
    if allocation != profile["sequences"]["8"]:
        raise ContractError("the suite requires the exact eight-device allocation from its profile")
    if devices(environ.get("TASK_DEVICE", "")) != allocation:
        raise ContractError("TASK_DEVICE differs from the requested allocation")
    if environ.get("TASKQUEUE_INSIDE") != "1":
        raise ContractError("run inside a task-submit allocation")
    for key in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        if environ.get(key, "").strip():
            raise ContractError(f"{key} remaps host IDs; this runner requires an unmasked host")
    if selected != profile["sequences"].get(str(len(selected))) or not set(selected) <= set(allocation):
        raise ContractError("case devices differ from the fixed profile or escape the allocation")


def load_manifest(path=MANIFEST):
    manifest = read_json(path)
    cases = manifest.get("cases", [])
    if manifest.get("schema_version") != 1 or manifest.get("sampling") != SAMPLING:
        raise ContractError("unsupported manifest or sampling contract")
    if len(cases) != 10 or {case.get("case_id") for case in cases} != CASE_IDS:
        raise ContractError("the suite must contain exactly the ten MTP/DSpark cases")
    if manifest.get("platform") != "a2a3":
        raise ContractError("official performance requires a2a3")
    for case in cases:
        entry = (ROOT / case["entrypoint"]).resolve()
        if not entry.is_relative_to(ROOT / "models") or not entry.is_file():
            raise ContractError("entry point must be an existing model in this checkout")
        if case["device_count"] not in (1, 4, 8) or not case.get("required_outputs"):
            raise ContractError("invalid case devices or output contract")
    return manifest


def case_arguments(case, selected):
    return ["-p", "a2a3", "-d", ",".join(map(str, selected)),
            *case["arguments"], "--enable-chip-swimlane", "0"]


def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ContractError(f"{label} must be a finite positive number")
    return float(value)


def check_specs(case, specs, *, require_outputs=False):
    by_name = {spec["name"]: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ContractError("duplicate tensor specifications")
    for name, expected in case["required_specs"].items():
        actual = by_name.get(name, {})
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ContractError(f"{name} shape/dtype differs from the workload contract")
    if require_outputs:
        outputs = {spec["name"] for spec in specs if spec.get("direction") in ("out", "inout")}
        if not set(case["required_outputs"]) <= outputs:
            raise ContractError("required results are not validated outputs")
