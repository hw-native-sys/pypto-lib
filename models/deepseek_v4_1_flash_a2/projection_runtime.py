# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ascend FP32 linear custom operator for actual Prefill compressor operands.

The boundary widens BF16 storage exactly and invokes aten.linear on the NPU.
Pooling, normalization, index projection, RoPE and publication remain in DSL.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from .runtime_utils import validate_device, file_sha as _sha, tensor_sha as _tensor_sha


def fp32_compressor_projection_host(
    source, kv_weight, gate_weight, *, expected_runtime, runtime_config, output_dir,
):
    """Stage actual BF16 operands and return the two full FP32 projections."""
    settings = json.loads(Path(runtime_config).read_text())
    validate_device(settings)
    directory = Path(output_dir)
    directory.mkdir(exist_ok=False)
    values = dict(source=source, kv_weight=kv_weight, gate_weight=gate_weight)
    if any(value.device.type != "cpu" for value in values.values()):
        raise ValueError("The projection boundary stages actual CPU tensors")
    values = {name: value.contiguous() for name, value in values.items()}
    runtime = {key: expected_runtime[key] for key in ("python", "torch", "torch_npu", "cxx11_abi")}
    inputs = directory / "inputs.pt"
    torch.save(dict(tensors=values, expected_runtime=runtime), inputs)
    product_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    env["PYTHONPATH"] = ":".join([str(product_root), *settings["vllm_pythonpath"]])
    command = [settings["vllm_python"], "-u", "-m", __name__, "--native", "--runtime-config",
               str(Path(runtime_config).resolve()), "--directory", str(directory.resolve())]
    with (directory / "run.log").open("x") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"FP32 projection exited {result.returncode}; inspect {directory / 'run.log'}")
    report = json.loads((directory / "report.json").read_text())
    outputs = directory / "outputs.pt"
    if (report["status"] != "pass" or report["runtime"] != runtime or report["reference_operands"]
            or not report["readonly_inputs_exact"] or report["inputs_sha256"] != _sha(inputs)
            or report["outputs_sha256"] != _sha(outputs)):
        raise RuntimeError("FP32 projection integrity/runtime check failed")
    for name, value in values.items():
        if report["actual_inputs"][name]["sha256"] != _tensor_sha(value):
            raise RuntimeError("Projection consumed a different product operand: " + name)
    actual = torch.load(outputs, map_location="cpu", weights_only=True)
    expected_shape = (source.shape[0], kv_weight.shape[1])
    if set(actual) != {"kv_proj", "score_proj"} or any(
        value.shape != expected_shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
        for value in actual.values()
    ):
        raise RuntimeError("Projection output ABI/finite check failed")
    return actual, report


def _native_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.runtime_config.read_text())
    invocation_pid = validate_device(settings)
    inputs = args.directory / "inputs.pt"
    payload = torch.load(inputs, map_location="cpu", weights_only=True)
    values = payload["tensors"]
    source, kv_weight, gate_weight = (values[name] for name in ("source", "kv_weight", "gate_weight"))
    if (source.ndim != 2 or source.shape[0] < 1 or kv_weight.shape != gate_weight.shape
            or kv_weight.shape != (source.shape[1], 512)
            or any(value.dtype != torch.bfloat16 or value.device.type != "cpu"
                   or not bool(torch.isfinite(value).all()) for value in values.values())):
        raise ValueError("Expected actual finite BF16 token rows and both complete width512 matrices")
    import torch_npu

    runtime = dict(python=sys.version, torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                   cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    if runtime != payload["expected_runtime"]:
        raise RuntimeError("FP32 projection Python/Torch/NPU ABI differs")
    torch.npu.set_device(0)
    device_source = source.to(device="npu:0", dtype=torch.float32)
    weights = {name: value.T.contiguous().to(device="npu:0", dtype=torch.float32)
               for name, value in (("kv_proj", kv_weight), ("score_proj", gate_weight))}
    actual = {name: torch.ops.aten.linear.default(device_source, weight) for name, weight in weights.items()}
    torch.npu.synchronize()
    readonly = torch.equal(device_source.cpu(), source.float()) and all(
        torch.equal(weights[name].cpu(), value.T.contiguous().float())
        for name, value in (("kv_proj", kv_weight), ("score_proj", gate_weight))
    )
    actual = {name: value.cpu().contiguous() for name, value in actual.items()}
    valid = all(value.shape == (source.shape[0], 512) and value.dtype == torch.float32
                and bool(torch.isfinite(value).all()) for value in actual.values())
    outputs = args.directory / "outputs.pt"
    torch.save(actual, outputs)
    report = dict(status="pass" if readonly and valid else "failed", boundary="aten.linear Ascend FP32 dispatch",
        reference_operands=False, physical_device=int(os.environ["ASCEND_RT_VISIBLE_DEVICES"]), logical_device=0, pid=os.getpid(),
        invocation_pid=invocation_pid, runtime=runtime, readonly_inputs_exact=readonly, finite_shape_dtype=valid,
        operator_schema=str(torch.ops.aten.linear.default._schema),
        input_storage="BF16 KN weights, exact FP32 widening, native contiguous NK operand",
        full_compute=dict(rows=source.shape[0], reduction=source.shape[1], columns=512, projections=2),
        actual_inputs={name: dict(sha256=_tensor_sha(value), shape=list(value.shape), dtype=str(value.dtype))
                       for name, value in values.items()},
        extension=str(Path(torch_npu._C.__file__).resolve()), extension_sha256=_sha(torch_npu._C.__file__),
        inputs_sha256=_sha(inputs), outputs_sha256=_sha(outputs),
        runtime_config_sha256=_sha(args.runtime_config), product_boundary_sha256=_sha(Path(__file__)))
    (args.directory / "report.json").open("x").write(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("FP32 projection readonly/output check failed")


if __name__ == "__main__":
    _native_main()
