# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Emit the independent per-layer Decode contract without loading weights.

The report is an environment preparation artifact.  It reads only checkpoint
JSON metadata and existing reference/product report files.  It never imports
PyPTO, vLLM, Torch-NPU, or a model computation path, and it never promotes a
layer from ``not_run`` based on metadata or stage construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_LAYERS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)
SESSION = "deepseek-v4.1-flash-a3-decode"
TOPOLOGY = {"dp": 2, "tp": 1, "ep": 2, "pp": 1, "pcp": 1, "dcp": 1}
REFERENCE_COMMIT = "23d1d9055f2161b3c15613c7fc309f7d5d2ef629"
REFERENCE_ROOT = "/mnt/data/pyptouser/liufei-shared-envs/vllm-a3-0.31.0-20261031"
REFERENCE_PYTHON = f"{REFERENCE_ROOT}/vllm-python.sh"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _latest_source(layer: int, ratio: int, sources: list[int], ratios: list[int]) -> int | None:
    candidates = [source for source in sources if source <= layer and ratios[source] == ratio]
    return candidates[-1] if candidates else None


def _layer_shards(weight_map: dict[str, str], layer: int) -> list[str]:
    prefix = f"layers.{layer}."
    return sorted({shard for name, shard in weight_map.items() if name.startswith(prefix)})


def _stage_entry(layer: int, ratio: int, engram: bool) -> dict[str, Any]:
    if layer == 0:
        return {"module": "decode_layer_ep2", "status": "implemented_l0_only"}
    if layer == 1:
        return {"module": "decode_swa + engram", "status": "composition_required", "engram": engram}
    if layer == 2:
        return {"module": "decode_c2a_full", "status": "stage_factory_only"}
    if layer in (3, 15, 19):
        return {"module": "decode_c2a_reuse", "status": "stage_factory_only"}
    if layer == 14:
        return {"module": "decode_c2a_full", "status": "stage_factory_only", "engram": engram}
    if layer == 20:
        return {"module": "decode_c1a_full", "status": "stage_factory_only"}
    if layer == 24:
        return {"module": "decode_c1a_reindex", "status": "stage_factory_only"}
    return {"module": "decode_c1a_reuse", "status": "stage_factory_only"}


def _execution_plan(
    layer: int,
    *,
    checkpoint: Path,
    assets_root: Path,
    product_root: Path,
) -> dict[str, Any]:
    """Return fixed, non-running commands for one independent layer replay."""

    reference_output = assets_root / f"decode_ep2_l{layer}"
    product_report = product_root / f"decode-layer-ep2-l{layer}.json"
    # Keep shell variables literal in the recorded command.  They are
    # resolved only inside the queue job after task-submit allocates devices.
    capture = (
        "task-submit --device auto --device-num 2 --run "
        "\"bash -lc 'set -euo pipefail; unset PYTHONPATH; unset ASCEND_HOME_PATH; "
        "export VLLM_USE_V2_MODEL_RUNNER=0; "
        "export SOC_VERSION=ascend910_9382; "
        "export ASCEND_RT_VISIBLE_DEVICES=\\\"\\${TASK_DEVICE:?TASK_DEVICE is required}\\\"; "
        f"{REFERENCE_PYTHON} -m torch.distributed.run --standalone "
        "--nnodes=1 --nproc_per_node=2 tools/deepseek_v41_vllm_capture.py "
        f"--release-root {REFERENCE_ROOT} --checkpoint {checkpoint} "
        f"--output {reference_output} --layer-id {layer} "
        "--history-length 8 --max-model-len 128 "
        "--task-device \\\"\\${TASK_DEVICE:?TASK_DEVICE is required}\\\" "
        "--capture-task \\\"\\${TASK_ID:-not_recorded}\\\"'\""
    )
    if layer == 0:
        replay_status = "ready_to_submit"
        replay = (
            "task-submit --device auto --device-num 2 --run "
            "\"source /mnt/data/pyptouser/liufei-shared-envs/"
            "pypto-a3-72a62d13-20261031/activate-shared-pypto.sh; "
            "export PYPTO_A3_STATE_DIR=\\$PWD/.pypto-a3-state; "
            "export SOC_VERSION=ascend910_9382; "
            "export DEEPSEEK_V41_CHECKPOINT="
            f"{checkpoint}; "
            "PYTHONPATH=\\$PWD python tools/deepseek_v41_pypto_layer_ep2_runner.py "
            f"--layer-id {layer} --assets {reference_output} "
            f"--cache-assets {reference_output} --record {product_report}\""
        )
        replay_blocker = None
    else:
        replay_status = "blocked_missing_complete_layer_entrypoint"
        replay = None
        replay_blocker = (
            "The current A2/A3 checkout exposes only stage factories for this layer; "
            "a complete PyPTO mHC/attention/cache/index/Engram/MoE composition is required."
        )
    return {
        "reference": {
            "status": "ready_to_submit" if checkpoint.is_dir() else "blocked_missing_checkpoint",
            "commit": REFERENCE_COMMIT,
            "python": REFERENCE_PYTHON,
            "topology": {**TOPOLOGY, "devices": 2},
            "output": str(reference_output),
            "command": capture,
            "required_files": [
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
            ],
        },
        "product": {
            "status": replay_status,
            "topology": {**TOPOLOGY, "devices": 2},
            "record": str(product_report),
            "command": replay,
            "blocker": replay_blocker,
            "requires": [
                "reference_capture.json",
                "decode_input.json",
                "history_cache.npz",
                "complete layer-specific PyPTO entrypoint",
            ],
        },
    }


def build_report(
    checkpoint: str | Path,
    *,
    assets_root: str | Path | None = None,
    product_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed contract matrix for the retained physical layers."""

    root = Path(checkpoint).expanduser().resolve()
    config = _read_json(root / "config.json")
    text = config.get("text_config", config)
    ratios = [int(value) for value in text["compress_ratios"]]
    kv_sources = [int(value) for value in text["kv_source_layer_ids"]]
    index_sources = [int(value) for value in text["index_source_layer_ids"]]
    weight_index = _read_json(root / "quant_model_weights.safetensors.index.json")
    weight_map = weight_index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint weight_map is missing")
    assets = Path(assets_root).expanduser().resolve() if assets_root else None
    product = Path(product_root).expanduser().resolve() if product_root else None
    plan_assets = assets or (root.parent / ".pypto-a3-state" / "reference")
    plan_product = product or (root.parent / ".pypto-a3-state" / "product")
    layers: list[dict[str, Any]] = []
    for layer in TARGET_LAYERS:
        ratio = ratios[layer]
        layer_assets = assets / f"l{layer}" if assets else None
        # The capture driver uses a timestamped directory per invocation;
        # accepting a direct layer directory keeps manually staged fixtures
        # discoverable as well.
        candidates = [layer_assets] if layer_assets else []
        if assets:
            candidates.append(assets)
            candidates.extend(sorted(assets.glob(f"decode_ep2_l{layer}_*"), reverse=True))
        asset_dir = next((item for item in candidates if item and (item / "decode_input.json").is_file()), None)
        report_candidates = []
        if product:
            report_candidates.extend(sorted(product.glob(f"decode-layer-ep2-l{layer}-*.json"), reverse=True))
            report_candidates.extend(sorted(product.glob(f"decode-layer-ep2-l{layer}.json"), reverse=True))
        product_report = next((item for item in report_candidates if item.name != "decode-acceptance.json"), None)
        component_candidates = []
        if product:
            component_candidates.extend(sorted(product.glob(f"decode-ep2-l{layer}-moe-*.json"), reverse=True))
        component_report = next((item for item in component_candidates if item.name != "decode-acceptance.json"), None)
        product_status = "not_run"
        if product_report:
            try:
                payload = _read_json(product_report)
                product_status = str(payload.get("status", "not_run"))
            except (OSError, ValueError):
                product_status = "invalid_report"
        component_status = "not_run"
        if component_report:
            try:
                component_status = str(_read_json(component_report).get("status", "not_run"))
            except (OSError, ValueError):
                component_status = "invalid_report"
        layers.append(
            {
                "layer_id": layer,
                "compression_ratio": ratio,
                "mode": "swa" if ratio == 0 else ("c2a" if ratio == 2 else "c1a"),
                "kv_source_layer_id": _latest_source(layer, ratio, kv_sources, ratios),
                "index_source_layer_id": _latest_source(layer, ratio, index_sources, ratios),
                "is_kv_source": layer in kv_sources,
                "is_index_source": layer in index_sources,
                "engram_required": layer in {int(value) for value in text.get("engram_layer_ids", ())},
                "weight_shards": _layer_shards(weight_map, layer),
                "weight_tensor_count": sum(1 for name in weight_map if name.startswith(f"layers.{layer}.")),
                "engram_assets": [
                    str(root / f"engram_embed_{kind}_l{layer}.safetensors")
                    for kind in ("weight", "scale")
                    if (root / f"engram_embed_{kind}_l{layer}.safetensors").is_file()
                ],
                "reference_assets": {
                    "root": str(asset_dir) if asset_dir else "not_ready",
                    "status": "ready" if asset_dir else "not_ready",
                    "required": ["checkpoint.json", "decode_input.json", "history_cache.npz", "reference_capture.json"],
                },
                "product_report": str(product_report) if product_report else "not_run",
                "product_accuracy": product_status,
                "moe_component_report": str(component_report) if component_report else "not_run",
                "moe_component_accuracy": component_status,
                "representative_status": "pass" if product_status == "pass" and asset_dir else "not_run",
                "stage_entry": _stage_entry(layer, ratio, layer in {1, 14}),
                "execution_plan": _execution_plan(
                    layer,
                    checkpoint=root,
                    assets_root=plan_assets,
                    product_root=plan_product,
                ),
                # A single active-request EP2 replay is not the formal gate.
                # Keep this independent from the representative report so a
                # report with ``status=pass`` cannot promote a layer.
                "formal_status": "not_run",
                "recovery": (
                    "Run an independent vLLM-Ascend capture and a complete PyPTO layer runner "
                    "with the same token/cache/state boundary."
                ),
            }
        )
    return {
        "session": SESSION,
        "status": "metadata_ready_not_npu_run",
        "checkpoint": str(root),
        "topology": TOPOLOGY,
        "target_layers": list(TARGET_LAYERS),
        "formal_tolerance": {"rtol": 0.05, "atol": 0.05, "discrete": "exact"},
        "layers": layers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--product-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.checkpoint, assets_root=args.assets_root, product_root=args.product_root)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
