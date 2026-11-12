# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fail-closed entry point for the independent A3 Decode topology gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from models.deepseek_v4_1_flash_a2.decode_acceptance import (
    CONTRACT,
    TOPOLOGIES,
    checkpoint_asset_report,
    runtime_identity,
)
from models.deepseek_v4_1_flash_a2.decode_asset_contract import validate_assets


REQUIRED_ASSETS = (
    "checkpoint.json",
    "decode_input.json",
    "history_cache.npz",
    "reference_capture.json",
)


def gate_report(topology: str, asset_root: Path | None = None) -> dict[str, Any]:
    """Describe one queued gate without claiming execution.

    The product and vLLM reference are deliberately separate processes. This
    entry point only checks that the frozen inputs exist and emits the exact
    queue contract; it never imports or launches vLLM.
    """

    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown Decode topology: {topology}")
    topology_config = TOPOLOGIES[topology]
    configured_root = os.environ.get("DEEPSEEK_V41_DECODE_ASSETS")
    root = asset_root or (Path(configured_root) if configured_root else None)
    assets = {
        name: (str(root / name) if root else "not_configured")
        for name in REQUIRED_ASSETS
    }
    present = {
        name: bool(root and (root / name).is_file())
        for name in REQUIRED_ASSETS
    }
    missing = [name for name, exists in present.items() if not exists]
    checkpoint = checkpoint_asset_report()
    asset_validation = validate_assets(
        root,
        topology_config,
        checkpoint_root=checkpoint.get("root"),
        expected_reference_commit=CONTRACT["runtime"]["vllm_reference_commit"],
    )
    implementation = CONTRACT["implementation"]
    devices = topology_config["devices"]
    ready = (
        not missing
        and checkpoint.get("status") == "metadata_verified"
        and implementation["formal_ready"]
        and asset_validation["status"] == "pass"
    )
    command = (
        f"task-submit --device auto --device-num {devices} --run "
        f"'PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_moe "
        "--contract'" if topology == "ep2_gate" else
        f"'PYTHONPATH=$PWD python -m models.deepseek_v4_1_flash_a2.decode_fwd "
        f"-p a2a3 --ep {topology_config['ep']} --dp {topology_config['dp']} "
        f"--tp {topology_config['tp']}'"
    )
    reference_command = (
        "task-submit --device auto --device-num "
        f"{devices} --run '<independent-vllm-ascend-reference> "
        f"--commit {CONTRACT['runtime']['vllm_reference_commit']} "
        f"--topology {topology}'"
    )
    return {
        "topology": topology,
        "topology_config": topology_config,
        "status": "not_run",
        "ready_for_submission": ready,
        "assets": assets,
        "asset_presence": present,
        "missing_assets": missing,
        "asset_validation": asset_validation,
        "checkpoint_asset": checkpoint,
        "implementation": implementation,
        "required_state_inputs": list(CONTRACT["state_inputs"]),
        "required_state_outputs": list(CONTRACT["state_outputs"]),
        "queue_command": command,
        "reference_command": reference_command,
        "product_entrypoint": (
            "models.deepseek_v4_1_flash_a2.decode_moe.decode_ep2"
            if topology == "ep2_gate"
            else "models.deepseek_v4_1_flash_a2.decode_fwd"
        ),
        "reference_process": {
            "status": "not_run",
            "commit": CONTRACT["runtime"]["vllm_reference_commit"],
            "separate_process": True,
            "product_imported": False,
        },
        "comparison": {
            "status": "not_run",
            "same_inputs": False,
            "same_layer": False,
            "same_topology": False,
            "same_state_boundary": False,
        },
        "runtime_identity": runtime_identity(),
        "recovery": (
            "Provide identical decode input and historical cache, and frozen "
            "vLLM-Ascend reference capture; set DEEPSEEK_V41_CHECKPOINT to the "
            "verified Aurora export, submit "
            "this topology through task-submit, and record per-rank state."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=tuple(TOPOLOGIES), default="ep2_gate")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    result = gate_report(args.topology, args.asset_root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(encoded, encoding="utf-8")
    if args.json or args.record:
        print(encoded, end="")
    else:
        print(f"[decode-gate] {args.topology}: not_run")
        print(f"[decode-gate] missing assets: {', '.join(result['missing_assets']) or 'none'}")
        print(f"[decode-gate] queue: {result['queue_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
