# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent DeepSeek V4.1 A3 Decode contract and product dependency scan."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from models.deepseek_v4_1_flash_a2.decode_candidates import CANDIDATES, SEARCH_K, validate_candidates
from models.deepseek_v4_1_flash_a2.decode_checkpoint_manifest import inspect_checkpoint
from models.deepseek_v4_1_flash_a2.decode_shape_contract import shape_matrix
from models.deepseek_v4_1_flash_a2.runtime_config import DECODE_RUNTIME_CONTRACT


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parent
TARGET_LAYERS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)
PRODUCT_ENTRYPOINTS = (
    "decode_fwd.py",
    "decode_replay.py",
    "decode_candidates.py",
    "decode_moe.py",
    "decode_layer_l0.py",
)
VALIDATION_ENTRYPOINTS = (
    "decode_acceptance.py",
    "decode_asset_contract.py",
    "decode_gate.py",
    "decode_reference.py",
    "decode_moe_compile.py",
    "decode_layer_l0_compile.py",
)
REFERENCE_ONLY_SUFFIXES = ("_runtime.py",)
FORBIDDEN_PRODUCT_IMPORTS = ("vllm", "vllm_ascend", "torch_npu")
DECODE_SESSION = "deepseek-v4.1-flash-a3-decode"
PYPTO_COMMIT = "72a62d13efabbb25d240e35944f68ba5bcc22b7f"
VLLM_REFERENCE_COMMIT = "23d1d9055f2161b3c15613c7fc309f7d5d2ef629"
CHECKPOINT_ENV = "DEEPSEEK_V41_CHECKPOINT"
CHECKPOINT_MANIFEST_ENV = "DEEPSEEK_V41_CHECKPOINT_MANIFEST"

SOURCE_COMMITS = (
    {
        "commit": "500858a9abd6ce1d3796a5e26a4bb597cdf622e4",
        "author": "Adelaide999",
        "email": "73272686+Adelaide999@users.noreply.github.com",
        "author_time": "2026-10-31T10:13:30+08:00",
    },
    {
        "commit": "d401f4e82e6348aee1566118c0c768b66e70504d",
        "author": "sunhaolun",
        "email": "sunhaolun3@huawei.com",
        "author_time": "2026-10-31T17:51:06+08:00",
    },
    {
        "commit": "c0f23d56800fbd34d4b3442d5df9565be387a955",
        "author": "Lun7777",
        "email": "1055318345@qq.com",
        "author_time": "2026-11-02T10:00:49+08:00",
    },
    {
        "commit": "addb655f7b4f4a4ce56af875724a04d79d3fe325",
        "author": "Lun7777",
        "email": "1055318345@qq.com",
        "author_time": "2026-11-02T14:36:58+08:00",
    },
)

TOPOLOGIES = {
    "ep2_gate": {"dp": 2, "tp": 1, "ep": 2, "pp": 1, "pcp": 1, "dcp": 1, "devices": 2},
    "main": {"dp": 8, "tp": 2, "ep": 16, "pp": 1, "pcp": 1, "dcp": 1, "devices": 16},
    "tp1_compat": {"dp": 16, "tp": 1, "ep": 16, "pp": 1, "pcp": 1, "dcp": 1, "devices": 16},
    "tp4_compat": {"dp": 4, "tp": 4, "ep": 16, "pp": 1, "pcp": 1, "dcp": 1, "devices": 16},
}

CONTRACT: dict[str, Any] = {
    "session": DECODE_SESSION,
    "product": "DeepSeek V4.1 Flash Decode single physical layer",
    "state": "independent_decode_not_run",
    "search_k": SEARCH_K,
    "baseline": {
        "repository": "https://github.com/Adelaide999/pypto-lib",
        "target": "addb655f7b4f4a4ce56af875724a04d79d3fe325",
        "base": "eed3280972f6743fc46d68219bcb108596e64578",
        "cherry_pick_order": [item["commit"] for item in SOURCE_COMMITS],
        "cherry_pick_x": {
            "required": True,
            "execution_evidence": "verified_in_temporary_clean_clone",
            "generated_commits": [
                "b76f321946c620cd4507e965ed54a9c0d93ec4d4",
                "286300578cecd2b1589407e273bb6e9a515defe1",
                "2f8671af7bc9d1c3d5a3468f5c2f22b073726c21",
                "56ef874b66126e3710758a7f9f717305a6e278cd",
            ],
            "result_tree": "116493ca2094d9237868be8abfabf9d8c51e1227",
            "source_checkout": "unavailable_in_current_workspace",
            "provided_receipt": "unavailable_in_current_workspace",
            "recovery": "Recreate a clean worktree from the base and run the four ordered git cherry-pick -x commands before promotion.",
        },
    },
    "model": {
        "hidden_size": 5120,
        "attention_heads": 64,
        "kv_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1280,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "moe_intermediate": 2304,
        "routed_experts": 384,
        "top_k": 6,
        "hc_mult": 4,
    },
    "shape_contract": shape_matrix(),
    "constructed_runtime_contract": DECODE_RUNTIME_CONTRACT,
    "quantization": {
        "model": "W8A8 dynamic product boundary",
        "engram": "INT8",
        "weight_scale": "FP32 channel scale",
        "expert_semantics": "FP4 checkpoint conversion remains an input contract",
        "block_size": 128,
    },
    "runtime": {
        "soc": "Ascend910_9382",
        "soc_version": "ascend910_9382",
        "platform": "a2a3",
        "pytorch_product_compute": False,
        "vllm_product_compute": False,
        "vllm_reference_commit": VLLM_REFERENCE_COMMIT,
        "vllm_reference_process": "separate process required",
        "enable_expert_parallel": True,
        "enable_fused_mc2": True,
        "enable_dsa_cp": True,
        "enable_flashcomm1": True,
        "enable_shared_expert_dp": True,
        "vllm_use_v2_model_runner": 0,
    },
    "state_inputs": [
        "decode_token",
        "history_token_length",
        "position_ids",
        "initial_kv_cache",
        "block_table",
        "slot_mapping",
        "valid_lengths",
        "mask",
        "active_finished_request_state",
        "engram_rows",
        "cache_source_ids",
        "ratio1_ratio2_state",
    ],
    "state_outputs": [
        "layer_output",
        "logits_or_hidden",
        "cache_delta",
        "index_and_candidate_state",
        "route_ids_and_gate_weights",
        "dispatch_combine_state",
        "untouched_cache_rows",
        "inactive_rows",
    ],
    "tolerances": {"rtol": 0.05, "atol": 0.05, "discrete": "exact"},
    "formal_scope": {
        "product_compute": "PyPTO only",
        "reference_compute": "independent vLLM-Ascend process",
        "cpu_precision": "out_of_scope",
        "prefill": "not_run",
        "continuous_network": "not_run",
        "mtp": "not_run",
        "dspark": "not_run",
        "dataset": "not_run",
        "http_service": "not_run",
    },
    "implementation": {
        "status": "ep2_composition_compile_pass_npu_not_run",
        "formal_ready": False,
        "formal_entrypoint": "models.deepseek_v4_1_flash_a2.decode_moe.decode_ep2",
        "blockers": [
            "EP2 driver NPU execution gate",
            "same-input EP2 vLLM-Ascend reference capture",
            "C1A/C2A compression and index refresh",
            "Engram state and cache ownership",
            "TP/DP/CP/SP runtime driver and state outputs",
        ],
    },
    "required_assets": (
        "checkpoint.json",
        "decode_input.json",
        "history_cache.npz",
        "reference_capture.json",
    ),
    "checkpoint_asset": {
        "path_env": CHECKPOINT_ENV,
        "manifest_env": CHECKPOINT_MANIFEST_ENV,
        "status": "not_configured",
        "payload_hashing": "omitted; external safetensors payloads are not copied into the repository",
    },
    "memory_contract": {
        "single_machine_capacity": "not_run",
        "per_rank_peak": "not_run",
        "required_inputs": ["checkpoint.json", "decode_input.json", "history_cache.npz"],
        "recovery": "Measure allocated bytes and peak live buffers in each queued EP2 and EP16 rank after the checkpoint and cache assets are supplied.",
    },
    "layers": {str(layer): "not_run" for layer in TARGET_LAYERS},
    "topologies": {name: "not_run" for name in TOPOLOGIES},
    "candidates": [candidate.__dict__ for candidate in CANDIDATES],
}


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    return ".".join(relative.parts)


def _local_module(path: Path, module: str, level: int = 0) -> Path | None:
    if level:
        package = path.parent
        for _ in range(level - 1):
            package = package.parent
        candidate = package / (module.replace(".", "/") if module else path.stem)
    else:
        candidate = ROOT / module.replace(".", "/")
    py_file = candidate.with_suffix(".py")
    package_file = candidate / "__init__.py"
    if py_file.is_file():
        return py_file
    if package_file.is_file():
        return package_file
    return None


def _import_targets(path: Path) -> list[tuple[str, Path | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: list[tuple[str, Path | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.append((alias.name, _local_module(path, alias.name)))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                local = _local_module(path, module, node.level)
                targets.append((module, local))
            else:
                local = _local_module(path, module)
                targets.append((module, local))
    return targets


def product_dependency_scan() -> dict[str, Any]:
    """Walk the product entrypoint closure and reject forbidden native bridges."""

    visited: set[Path] = set()
    queue = [MODEL_ROOT / name for name in PRODUCT_ENTRYPOINTS]
    forbidden: list[dict[str, str]] = []
    host_reference: list[dict[str, str]] = []
    subprocess_edges: list[str] = []
    reference_edges: list[str] = []
    while queue:
        path = queue.pop()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        for imported, local in _import_targets(path):
            root = imported.split(".", 1)[0]
            if root in FORBIDDEN_PRODUCT_IMPORTS:
                forbidden.append({"file": str(path.relative_to(ROOT)), "import": imported})
            if root == "torch":
                host_reference.append({"file": str(path.relative_to(ROOT)), "import": imported})
            if root == "subprocess":
                subprocess_edges.append(str(path.relative_to(ROOT)))
            if path.name.endswith(REFERENCE_ONLY_SUFFIXES):
                reference_edges.append(str(path.relative_to(ROOT)))
            if local is not None and local.is_relative_to(MODEL_ROOT):
                queue.append(local)
    return {
        "status": "pass" if not forbidden else "failed",
        "visited_files": sorted(str(path.relative_to(ROOT)) for path in visited),
        "forbidden_imports": forbidden,
        "host_reference_imports": host_reference,
        "subprocess_edges": subprocess_edges,
        "reference_edges": sorted(set(reference_edges)),
        "product_compute_scope": "complete_l0_swa_mhc_and_ep2_moe_driver",
        "reference_only_files": sorted(
            str(path.relative_to(ROOT))
            for path in MODEL_ROOT.glob("*_runtime.py")
        ),
    }


def runtime_identity() -> dict[str, Any]:
    """Capture source and process identity without importing product kernels."""

    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
    def module_origin(name: str) -> str:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            return "unavailable"
        if spec is None:
            return "unavailable"
        return spec.origin if spec.origin else "namespace"

    loaded_shared_objects: set[str] = set()
    maps_path = Path("/proc/self/maps")
    if maps_path.is_file():
        for line in maps_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"(/[^ ]+\.so(?:\.[^ ]+)?)", line)
            if match:
                loaded_shared_objects.add(match.group(1))

    model_files = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(MODEL_ROOT.glob("decode_*.py"))
        if path.is_file()
    }
    forbidden_loaded = sorted(
        name for name in sys.modules if name == "vllm" or name.startswith(("vllm.", "vllm_ascend", "torch_npu"))
    )
    child_pids: list[int] = []
    children_path = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    if children_path.is_file():
        try:
            child_pids = [int(value) for value in children_path.read_text().split()]
        except (OSError, ValueError):
            child_pids = []
    pypto_origin = module_origin("pypto")
    pypto_root = str(Path(pypto_origin).parents[2]) if pypto_origin.endswith("/pypto/__init__.py") else "unavailable"
    vllm_origin = module_origin("vllm")
    vllm_ascend_origin = module_origin("vllm_ascend")
    torch_npu_origin = module_origin("torch_npu")
    source_root = os.environ.get("PYPTO_ROOT")
    release_root = os.environ.get("PYPTO_A3_RELEASE")
    if not source_root and release_root:
        source_root = str(Path(release_root) / "source/pypto")
    if not source_root:
        source_root = pypto_root
    return {
        "session": DECODE_SESSION,
        "git_head": git_head,
        "baseline_target": CONTRACT["baseline"]["target"],
        "python": sys.executable,
        "python_version": platform.python_version(),
        "pypto_source": source_root,
        "pypto_python": pypto_origin,
        "pypto_commit": PYPTO_COMMIT,
        "golden_python": module_origin("golden"),
        "soc_version": os.environ.get("SOC_VERSION", "not_exported"),
        "task_device": os.environ.get("TASK_DEVICE", "not_allocated"),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "not_exported"),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "cann_root": os.environ.get("ASCEND_HOME_PATH", os.environ.get("CANN_HOME", "not_exported")),
        "library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "loaded_shared_objects": sorted(loaded_shared_objects),
        "forbidden_modules_loaded": forbidden_loaded,
        "child_processes": bool(child_pids),
        "child_pids": child_pids,
        "module_origins": {
            "pypto": pypto_origin,
            "golden": module_origin("golden"),
            "vllm": vllm_origin,
            "vllm_ascend": vllm_ascend_origin,
            "torch_npu": torch_npu_origin,
        },
        "model_files": model_files,
        "vllm_reference_commit": VLLM_REFERENCE_COMMIT,
        "reference_process": "not_started",
    }


def source_hashes() -> dict[str, str]:
    """Return hashes for Decode-owned source files."""

    files = [
        MODEL_ROOT / name
        for name in PRODUCT_ENTRYPOINTS + VALIDATION_ENTRYPOINTS
    ]
    files.extend(
        [
            MODEL_ROOT / "__init__.py",
            MODEL_ROOT / "runtime_config.py",
            MODEL_ROOT / "source_manifest.json",
            MODEL_ROOT / "decode_cherry_pick_receipt.json",
            MODEL_ROOT / "decode_checkpoint_manifest.py",
            MODEL_ROOT / "decode_weights.py",
            MODEL_ROOT / "decode_weight_abi.py",
        ]
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if path.is_file()
    }


def checkpoint_asset_report() -> dict[str, Any]:
    """Report the supplied checkpoint without treating it as Decode evidence."""

    root_value = os.environ.get(CHECKPOINT_ENV)
    manifest_value = os.environ.get(CHECKPOINT_MANIFEST_ENV)
    if manifest_value:
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.is_file():
            return {
                "status": "invalid_manifest",
                "path_env": CHECKPOINT_ENV,
                "manifest": str(manifest_path),
                "error": "manifest does not exist",
            }
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "status": payload.get("status", "unknown"),
            "root": payload.get("root", root_value or "not_configured"),
            "manifest": str(manifest_path),
            "index": payload.get("index", {}),
            "config": payload.get("config", {}),
            "target_layer_files": payload.get("target_layer_files", {}),
            "engram_files": payload.get("engram_files", []),
            "payload_hashing": payload.get("payload_hashing"),
        }
    if not root_value:
        return {"status": "not_configured", "path_env": CHECKPOINT_ENV, "manifest_env": CHECKPOINT_MANIFEST_ENV}
    try:
        payload = inspect_checkpoint(root_value)
    except (FileNotFoundError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "root": str(Path(root_value).expanduser()), "error": str(exc)}
    return {
        "status": payload["status"],
        "root": payload["root"],
        "index": payload["index"],
        "config": payload["config"],
        "target_layer_files": payload["target_layer_files"],
        "engram_files": payload["engram_files"],
        "payload_hashing": payload["payload_hashing"],
    }


def report() -> dict[str, Any]:
    validate_candidates()
    result = dict(CONTRACT)
    result["runtime_identity"] = runtime_identity()
    result["product_dependency_scan"] = product_dependency_scan()
    result["source_hashes"] = source_hashes()
    result["checkpoint_asset"] = checkpoint_asset_report()
    result["formal_gate"] = "not_run"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete contract as JSON")
    parser.add_argument("--static-scan", action="store_true", help="scan the PyPTO product import closure")
    parser.add_argument("--record", type=Path, help="write the report to a JSON file")
    args = parser.parse_args(argv)
    result = report()
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json or args.record:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        scan = result["product_dependency_scan"]
        print(f"[decode-contract] static product scan={scan['status']}; files={len(scan['visited_files'])}")
        print(f"[decode-contract] Search K={SEARCH_K}; formal_gate=not_run")
        print("[decode-contract] NPU checkpoint/reference gates remain not_run until assets are supplied")
    return 0 if result["product_dependency_scan"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
