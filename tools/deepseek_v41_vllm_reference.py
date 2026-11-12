# Copyright (c) PyPTO Contributors.
# This program is free software: you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. This software is provided without warranties.
# -----------------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
"""Record independent vLLM-Ascend DeepSeek V4.1 reference identity.

This script is run with the pinned vLLM environment, never from the PyPTO
product process.  It intentionally performs only runtime/config inspection
and a small NPU RMSNorm smoke; it does not start vLLM serve or load weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _library_identity(root: Path) -> dict[str, str]:
    relative_paths = (
        "source/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so",
        "source/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so",
        "env/lib/python3.12/site-packages/torch_npu/_C.cpython-312-aarch64-linux-gnu.so",
        "env/lib/python3.12/site-packages/torch_npu/lib/libtorch_npu.so",
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        result[relative] = _sha256(path) if path.is_file() else "missing"
    return result


def _module_origins() -> dict[str, str]:
    import torch_npu
    import vllm
    import vllm_ascend
    import vllm_ascend.vllm_ascend_C

    return {
        "vllm": str(vllm.__file__),
        "vllm_ascend": str(vllm_ascend.__file__),
        "vllm_ascend_C": str(vllm_ascend.vllm_ascend_C.__file__),
        "torch_npu": str(torch_npu.__file__),
    }


def _config_snapshot(model: Path, *, tensor_parallel: int, data_parallel: int) -> dict[str, Any]:
    from vllm.plugins import load_general_plugins
    import vllm_ascend.quantization.configs  # noqa: F401
    from vllm.config import ModelConfig, ParallelConfig

    load_general_plugins()
    model_config = ModelConfig(
        model=str(model),
        tokenizer=str(model),
        dtype="bfloat16",
        quantization="ascend",
        trust_remote_code=False,
        max_model_len=128,
    )
    text = getattr(model_config.hf_config, "text_config", model_config.hf_config)
    parallel = ParallelConfig(
        tensor_parallel_size=tensor_parallel,
        data_parallel_size=data_parallel,
        data_parallel_size_local=data_parallel,
        enable_expert_parallel=True,
        is_moe_model=True,
    )
    keys = (
        "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
        "q_lora_rank", "o_lora_rank", "o_groups", "n_routed_experts",
        "num_experts_per_tok", "moe_intermediate_size", "compress_ratios",
        "kv_source_layer_ids", "index_source_layer_ids", "index_topk",
        "engram_layer_ids", "block_size", "enable_dsa_cp",
    )
    return {
        "model": str(model),
        "model_type": getattr(text, "model_type", None),
        "text_config": {key: getattr(text, key, None) for key in keys},
        "parallel": {
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "data_parallel_size": parallel.data_parallel_size,
            "enable_expert_parallel": parallel.enable_expert_parallel,
        },
        "config_only": True,
        "weights_loaded": False,
    }


def probe(root: Path, model: Path | None, *, tensor_parallel: int, data_parallel: int, npu_smoke: bool) -> dict[str, Any]:
    import torch
    import torch_npu

    result: dict[str, Any] = {
        "status": "pass",
        "role": "independent_reference_only",
        "python": sys.executable,
        "python_version": sys.version,
        "soc_version": os.environ.get("SOC_VERSION", "not_exported"),
        "task_device": os.environ.get("TASK_DEVICE", "not_allocated"),
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "not_exported"),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "module_origins": _module_origins(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "vllm-ascend", "torch-npu")
        },
        "native_sha256": _library_identity(root),
        "child_processes": False,
        "serve_started": False,
        "weights_loaded": False,
    }
    if npu_smoke:
        torch.npu.set_device(0)
        torch.manual_seed(7)
        source = torch.randn(1, 7168, dtype=torch.float16, device="npu:0")
        gamma = torch.randn(7168, dtype=torch.float16, device="npu:0")
        output, _ = torch_npu.npu_rms_norm(source, gamma, 1e-6)
        torch.npu.synchronize()
        result["npu_smoke"] = {
            "operation": "torch_npu.npu_rms_norm",
            "shape": [1, 7168],
            "dtype": "float16",
            "device": "npu:0",
            "device_mapping": [
                {"logical_device": 0, "physical_device": os.environ.get("TASK_DEVICE", "unknown")}
            ],
            "status": "pass",
            "output_dtype": str(output.dtype),
        }
    if model is not None:
        result["config_snapshot"] = _config_snapshot(
            model, tensor_parallel=tensor_parallel, data_parallel=data_parallel
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--data-parallel", type=int, default=2)
    parser.add_argument("--npu-smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    result = probe(
        args.release_root,
        args.model,
        tensor_parallel=args.tensor_parallel,
        data_parallel=args.data_parallel,
        npu_smoke=args.npu_smoke,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
