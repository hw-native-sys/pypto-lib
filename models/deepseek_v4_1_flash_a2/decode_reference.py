# Copyright (c) PyPTO Contributors.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Describe an independent vLLM-Ascend Decode capture without importing it.

The product process never calls this module.  It records the command and the
identity fields that a separately queued reference process must publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from models.deepseek_v4_1_flash_a2.decode_acceptance import CONTRACT, TOPOLOGIES


REQUIRED_ASSETS = tuple(CONTRACT["required_assets"])


def reference_manifest(topology: str, asset_root: Path | None = None) -> dict[str, Any]:
    """Build a fail-closed manifest for the external reference process."""

    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown Decode topology: {topology}")
    root_value = os.environ.get("DEEPSEEK_V41_DECODE_ASSETS")
    root = asset_root or (Path(root_value) if root_value else None)
    assets = {
        name: str(root / name) if root else "not_configured"
        for name in REQUIRED_ASSETS
    }
    present = {name: bool(root and (root / name).is_file()) for name in REQUIRED_ASSETS}
    topology_config = TOPOLOGIES[topology]
    devices = topology_config["devices"]
    source_root = os.environ.get("VLLM_ASCEND_SOURCE", "not_exported")
    pythonpath = os.environ.get("VLLM_PYTHONPATH", "not_exported")
    source_digest = "unavailable"
    if source_root != "not_exported":
        source_path = Path(source_root)
        if source_path.is_file():
            source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return {
        "status": "not_run",
        "topology": topology,
        "topology_config": topology_config,
        "commit": CONTRACT["runtime"]["vllm_reference_commit"],
        "assets": assets,
        "asset_presence": present,
        "missing_assets": [name for name, exists in present.items() if not exists],
        "python": os.environ.get("VLLM_PYTHON", "not_exported"),
        "pythonpath": pythonpath,
        "source": source_root,
        "source_sha256": source_digest,
        "loaded_shared_objects": os.environ.get("VLLM_REFERENCE_SO", "not_exported"),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "not_exported"),
        "runtime_library_path": os.environ.get("LD_LIBRARY_PATH", "not_exported"),
        "child_processes": "must_be_false",
        "product_imported": False,
        "queue_command": (
            f"task-submit --device auto --device-num {devices} --run "
            f"'<independent-vllm-ascend-capture> --commit {CONTRACT['runtime']['vllm_reference_commit']} "
            f"--topology {topology}'"
        ),
        "recovery": "Run the reference command in its own process, then record module paths, mapped libraries, source digest, environment and cache/output capture.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES), default="main")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = reference_manifest(args.topology, args.asset_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(encoded, encoding="utf-8")
    if args.json or args.record:
        print(encoded, end="")
    else:
        print(f"[decode-reference] {args.topology}: not_run")
        print(f"[decode-reference] queue: {result['queue_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
