# Copyright (c) PyPTO Contributors.
# This program is free software; you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0.
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare independent Decode replay plans for the retained physical layers.

This is a no-device setup tool.  It validates the external checkpoint metadata,
records one isolated work directory per physical layer, and emits commands for
the later queued reference capture.  It deliberately does not create a
``decode_input.json``, cache, or numerical output: those files are produced by
the independent vLLM-Ascend capture and are required before a PyPTO replay can
start.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


TARGET_LAYERS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)
SESSION = "deepseek-v4.1-flash-a3-decode"
CHECKPOINT = Path("/mnt/data/models/DeepSeek-V4.1-Flash-w8a8")
REFERENCE_ROOT = Path("/mnt/data/pyptouser/liufei-shared-envs/vllm-a3-0.31.0-20261031")
REFERENCE_PYTHON = REFERENCE_ROOT / "vllm-python.sh"
PYPTO_ACTIVATE = Path(
    "/mnt/data/pyptouser/liufei-shared-envs/pypto-a3-72a62d13-20261031/activate-shared-pypto.sh"
)
REQUIRED_REFERENCE_FILES = (
    "checkpoint.json",
    "decode_input.json",
    "history_cache.npz",
    "reference_capture.json",
    "rank0/capture.json",
    "rank1/capture.json",
    "rank0/layer_capture.npz",
    "rank1/layer_capture.npz",
    "rank0/moe_capture.npz",
    "rank1/moe_capture.npz",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _text_config(checkpoint: Path) -> dict[str, Any]:
    config = _read_json(checkpoint / "config.json")
    value = config.get("text_config", config)
    if not isinstance(value, dict):
        raise ValueError("checkpoint text_config must be an object")
    return value


def _latest_source(layer: int, ratio: int, sources: list[int], ratios: list[int]) -> int | None:
    matches = [source for source in sources if source <= layer and ratios[source] == ratio]
    return matches[-1] if matches else None


def _layer_shards(weight_map: dict[str, str], layer: int) -> list[str]:
    prefix = f"layers.{layer}."
    return sorted({shard for name, shard in weight_map.items() if name.startswith(prefix)})


def _mode(layer: int, ratio: int, text: dict[str, Any]) -> str:
    if ratio == 0:
        return "swa_engram" if layer in {int(item) for item in text.get("engram_layer_ids", ())} else "swa"
    if ratio == 2:
        return "c2a_full" if layer == 2 or layer == 14 else "c2a_reuse"
    if ratio == 1:
        return "c1a_full" if layer == 20 else ("c1a_reindex" if layer == 24 else "c1a_reuse")
    return f"ratio_{ratio}_unsupported"


def _layer_state_requirements(layer: int, mode: str) -> tuple[str, ...]:
    """Return state planes that must be captured for this physical layer."""

    if mode == "swa_engram":
        return ("engram_rows",)
    if mode == "swa":
        return ()
    if mode.startswith("c2a"):
        return ("compressed_kv_cache", "index_cache", "index_scale")
    if mode == "c1a_full":
        return ("compressed_kv_cache", "index_cache", "index_scale", "candidate_pool")
    if mode in {"c1a_reuse", "c1a_reindex"}:
        return ("compressed_kv_cache", "index_cache", "index_scale", "candidate_pool")
    raise ValueError(f"unsupported layer mode for L{layer}: {mode}")


def _capture_state_missing(candidate: Path, layer: int, mode: str) -> list[str]:
    """Reject generic layer/MoE captures that lack physical cache state."""

    requirements = _layer_state_requirements(layer, mode)
    if not requirements:
        return []
    missing: list[str] = []
    try:
        reference = _read_json(candidate / "reference_capture.json")
    except (OSError, ValueError):
        return ["reference_capture.json.layer_state_capture"]
    state = reference.get("layer_state_capture")
    if not isinstance(state, dict):
        return ["reference_capture.json.layer_state_capture"]
    if state.get("complete") is not True:
        missing.append("reference_capture.json.layer_state_capture.complete")
    for name in requirements:
        value = state.get(name)
        if not isinstance(value, dict) or value.get("captured") is not True:
            missing.append(f"reference_capture.json.layer_state_capture.{name}")
    if "engram_rows" in requirements:
        try:
            decode_input = _read_json(candidate / "decode_input.json")
        except (OSError, ValueError):
            decode_input = {}
        rows = decode_input.get("engram_rows")
        if not isinstance(rows, dict) or rows.get("captured") is not True:
            missing.append("decode_input.json.engram_rows.captured")
    return missing


def _reference_candidate(assets_root: Path, layer: int, mode: str) -> tuple[Path, str, list[str]]:
    """Find a complete immutable capture without treating partial output as ready."""

    canonical = assets_root / f"decode_ep2_l{layer}"
    candidates = [canonical]
    candidates.extend(sorted(assets_root.glob(f"decode_ep2_l{layer}_*"), reverse=True))
    pending_state: tuple[Path, list[str]] | None = None
    for candidate in candidates:
        file_missing = [name for name in REQUIRED_REFERENCE_FILES if not (candidate / name).is_file()]
        if file_missing:
            continue
        missing = _capture_state_missing(candidate, layer, mode)
        if missing:
            if pending_state is None:
                pending_state = (candidate, missing)
            continue
        if not missing:
            return candidate, "ready", []
    if pending_state is not None:
        return pending_state[0], "pending_capture", pending_state[1]
    missing = [name for name in REQUIRED_REFERENCE_FILES if not (canonical / name).is_file()]
    if not missing and canonical.is_dir():
        missing.extend(_capture_state_missing(canonical, layer, mode))
    return canonical, "pending_capture", missing


def _capture_command(checkpoint: Path, output: Path, layer: int) -> str:
    # Keep TASK_DEVICE queue-owned.  TASK_ID is optional on queue wrappers and
    # is only evidence metadata, so it must not abort a capture before launch.
    command = [
        str(REFERENCE_PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "tools/deepseek_v41_vllm_capture.py",
        "--release-root",
        str(REFERENCE_ROOT),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--layer-id",
        str(layer),
        "--history-length",
        "8",
        "--max-model-len",
        "128",
        "--task-device",
        '"${TASK_DEVICE:?TASK_DEVICE is required}"',
        "--capture-task",
        '"${TASK_ID:-not_recorded}"',
    ]
    argv = " ".join(shlex.quote(item) if not item.startswith('"${') else item for item in command)
    return (
        "task-submit --device auto --device-num 2 --run "
        + shlex.quote(
            "bash -lc 'set -euo pipefail; "
            "unset PYTHONPATH; unset ASCEND_HOME_PATH; "
            "export VLLM_USE_V2_MODEL_RUNNER=0; "
            "export SOC_VERSION=ascend910_9382; "
            + argv
            + "'"
        )
    )


def _product_command(checkpoint: Path, assets: Path, product: Path, layer: int) -> str | None:
    if layer != 0:
        return None
    command = (
        "source "
        + shlex.quote(str(PYPTO_ACTIVATE))
        + "; export SOC_VERSION=ascend910_9382; "
        + "export PYPTO_A3_STATE_DIR=\"$PWD/.pypto-a3-state\"; "
        + "PYTHONPATH=\"$PWD\" python tools/deepseek_v41_pypto_layer_ep2_runner.py "
        + f"--layer-id {layer} --assets {shlex.quote(str(assets))} "
        + f"--cache-assets {shlex.quote(str(assets))} --record {shlex.quote(str(product))}"
    )
    return "task-submit --device auto --device-num 2 --run " + shlex.quote(command)


def _validate_checkpoint(checkpoint: Path, target_layers: tuple[int, ...]) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {checkpoint}")
    text = _text_config(checkpoint)
    ratios = [int(value) for value in text["compress_ratios"]]
    kv_sources = [int(value) for value in text["kv_source_layer_ids"]]
    index_sources = [int(value) for value in text["index_source_layer_ids"]]
    if max(target_layers) >= len(ratios):
        raise ValueError("checkpoint compression plan does not cover every target layer")
    index = _read_json(checkpoint / "quant_model_weights.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint weight_map is missing")
    missing_shards = sorted(
        {
            shard
            for layer in target_layers
            for shard in _layer_shards(weight_map, layer)
            if not (checkpoint / shard).is_file()
        }
    )
    if missing_shards:
        raise FileNotFoundError(f"checkpoint shards are missing: {missing_shards}")
    engram = {
        layer: [
            str(checkpoint / f"engram_embed_{kind}_l{layer}.safetensors")
            for kind in ("weight", "scale")
            if (checkpoint / f"engram_embed_{kind}_l{layer}.safetensors").is_file()
        ]
        for layer in target_layers
        if layer in {int(item) for item in text.get("engram_layer_ids", ())}
    }
    missing_engram = [layer for layer, paths in engram.items() if len(paths) != 2]
    if missing_engram:
        raise FileNotFoundError(f"Engram weight/scale pair is missing for layers: {missing_engram}")
    return {
        "text_config": text,
        "ratios": ratios,
        "kv_sources": kv_sources,
        "index_sources": index_sources,
        "weight_map": weight_map,
        "engram": engram,
        "missing_shards": missing_shards,
    }


def prepare(
    checkpoint: Path,
    state_root: Path,
    assets_root: Path,
    product_root: Path,
) -> dict[str, Any]:
    metadata = _validate_checkpoint(checkpoint, TARGET_LAYERS)
    text = metadata["text_config"]
    ratios = metadata["ratios"]
    kv_sources = metadata["kv_sources"]
    index_sources = metadata["index_sources"]
    state_root.mkdir(parents=True, exist_ok=True)
    layers: list[dict[str, Any]] = []
    for layer in TARGET_LAYERS:
        ratio = ratios[layer]
        work = state_root / "layers" / f"l{layer}"
        work.mkdir(parents=True, exist_ok=True)
        assets = assets_root / f"decode_ep2_l{layer}"
        product = product_root / f"decode-layer-ep2-l{layer}.json"
        mode = _mode(layer, ratio, text)
        observed_assets, reference_status, missing = _reference_candidate(assets_root, layer, mode)
        complete_entrypoint = layer == 0
        plan = {
            "session": SESSION,
            "status": "prepared_not_run",
            "layer_id": layer,
            "mode": mode,
            "reference_state_requirements": list(_layer_state_requirements(layer, mode)),
            "compression_ratio": ratio,
            "kv_source_layer_id": _latest_source(layer, ratio, kv_sources, ratios),
            "index_source_layer_id": _latest_source(layer, ratio, index_sources, ratios),
            "is_kv_source": layer in kv_sources,
            "is_index_source": layer in index_sources,
            "engram_required": layer in metadata["engram"],
            "weight_shards": _layer_shards(metadata["weight_map"], layer),
            "engram_assets": metadata["engram"].get(layer, []),
            "topology": {"dp": 2, "tp": 1, "ep": 2, "pp": 1, "pcp": 1, "dcp": 1, "devices": 2},
            "input_contract": {
                "required": [
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
                "source": "independent vLLM-Ascend capture",
                "fixture_status": "not_created_until_reference_capture",
            },
            "reference": {
                "status": reference_status,
                "root": str(observed_assets),
                "capture_output": str(assets),
                "missing_files": missing,
                "command": _capture_command(checkpoint, assets, layer),
                "required_files": list(REQUIRED_REFERENCE_FILES),
                "reference_only": True,
            },
            "product": {
                "status": "ready_to_submit" if complete_entrypoint else "blocked_missing_complete_layer_entrypoint",
                "root": str(product),
                "command": _product_command(checkpoint, assets, product, layer),
                "entrypoint": (
                    "models.deepseek_v4_1_flash_a2.decode_layer_ep2.decode_layer_ep2"
                    if complete_entrypoint
                    else None
                ),
                "compute": "PyPTO only",
                "blocker": None
                if complete_entrypoint
                else "A complete A2/A3 mHC/attention/cache/index/Engram/MoE layer entrypoint is not present.",
            },
            "accuracy": {
                "reference": "not_run",
                "npu": "not_run",
                "cross_check": "not_run",
                "formal": "not_run",
                "rtol": 0.05,
                "atol": 0.05,
                "discrete": "exact",
            },
            "do_not_create": ["decode_input.json", "history_cache.npz", "layer_capture.npz", "moe_capture.npz"],
            "recovery": (
                "Submit the reference command when two cards are available, validate all required "
                "files, then add the complete PyPTO layer runner."
            ),
        }
        (work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        layers.append(plan)
    report = {
        "session": SESSION,
        "status": "setup_only",
        "checkpoint": str(checkpoint),
        "target_layers": list(TARGET_LAYERS),
        "product_compute": "PyPTO only",
        "reference_compute": "independent vLLM-Ascend process",
        "npu_execution": "not_run",
        "accuracy": "not_run",
        "formal_tolerance": {"rtol": 0.05, "atol": 0.05, "discrete": "exact"},
        "layers": layers,
    }
    output = state_root / "layer-preflight.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--state-root", type=Path, default=Path(".pypto-a3-state/decode-assets"))
    parser.add_argument("--assets-root", type=Path, default=Path(".pypto-a3-state/reference"))
    parser.add_argument("--product-root", type=Path, default=Path(".pypto-a3-state/product"))
    args = parser.parse_args(argv)
    report = prepare(
        args.checkpoint.resolve(),
        args.state_root.resolve(),
        args.assets_root.resolve(),
        args.product_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str((args.state_root / "layer-preflight.json").resolve()),
                "layers": len(report["layers"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
