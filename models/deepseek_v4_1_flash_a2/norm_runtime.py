# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Registered A2 RMSNorm boundary on actual product operands."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from .runtime_utils import validate_device, file_sha as _sha, tensor_sha as _tensor_sha


def rms_norm_host(source, weight, *, eps, expected_runtime, runtime_config, output_dir):
    """Stage BF16 inputs, execute npu_rms_norm and return its BF16 output."""
    directory = Path(output_dir)
    directory.mkdir(exist_ok=False)
    settings = json.loads(Path(runtime_config).read_text())
    validate_device(settings)
    values = {"source": source, "weight": weight}
    if any(value.device.type != "cpu" for value in values.values()):
        raise ValueError("RMSNorm requires CPU staging of actual product operands")
    values = {name: value.contiguous() for name, value in values.items()}
    runtime = {key: expected_runtime[key] for key in ("python", "torch", "torch_npu", "cxx11_abi")}
    input_path = directory / "inputs.pt"
    torch.save(dict(tensors=values, eps=float(eps), expected_runtime=runtime), input_path)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    product_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = ":".join([str(product_root), *settings["vllm_pythonpath"]])
    command = [settings["vllm_python"], "-u", "-m", __name__, "--native", "--runtime-config",
               str(Path(runtime_config).resolve()), "--directory", str(directory.resolve())]
    with (directory / "run.log").open("x") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Native RMSNorm exited {result.returncode}; inspect {directory / 'run.log'}")
    report = json.loads((directory / "report.json").read_text())
    if (report["status"] != "pass" or not report["readonly_inputs_exact"]
            or report["inputs_sha256"] != _sha(input_path) or report["runtime"] != runtime
            or report["eps"] != float(eps) or report["reference_operands"]):
        raise RuntimeError("Actual-operand RMSNorm runtime or integrity differs")
    for name, value in values.items():
        if report["actual_inputs"][name]["sha256"] != _tensor_sha(value):
            raise RuntimeError("RMSNorm consumed a different product operand: " + name)
    output_path = directory / "outputs.pt"
    if report["outputs_sha256"] != _sha(output_path):
        raise RuntimeError("RMSNorm output identity differs")
    outputs = torch.load(output_path, map_location="cpu", weights_only=True)
    output = outputs["normed"]
    if output.shape != source.shape or output.dtype != source.dtype or not bool(torch.isfinite(output).all()):
        raise RuntimeError("Actual RMSNorm output ABI or finite check failed")
    return output, report


def _native_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.runtime_config.read_text())
    invocation_pid = validate_device(settings)
    payload = torch.load(args.directory / "inputs.pt", map_location="cpu", weights_only=True)
    source, weight = payload["tensors"]["source"], payload["tensors"]["weight"]
    if (source.ndim != 2 or source.shape[0] < 1 or weight.shape != source.shape[1:]
            or any(value.dtype != torch.bfloat16 or value.device.type != "cpu"
                   or not bool(torch.isfinite(value).all()) for value in (source, weight))
            or not 0 < payload["eps"] < 1):
        raise ValueError("Expected finite BF16 active rows/gamma and positive RMSNorm epsilon")
    import torch_npu

    runtime = dict(python=sys.version, torch=str(torch.__version__),
                   torch_npu=str(torch_npu.__version__), cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    if runtime != payload["expected_runtime"]:
        raise RuntimeError(f"RMSNorm ABI differs: actual={runtime!r}; expected={payload['expected_runtime']!r}")
    torch.npu.set_device(0)
    device_source, device_weight = source.to("npu:0"), weight.to("npu:0")
    normed, _ = torch_npu.npu_rms_norm(device_source, device_weight, payload["eps"])
    torch.npu.synchronize()
    output = normed.cpu().contiguous()
    readonly = torch.equal(device_source.cpu(), source) and torch.equal(device_weight.cpu(), weight)
    valid = output.shape == source.shape and output.dtype == source.dtype and bool(torch.isfinite(output).all())
    output_path = args.directory / "outputs.pt"
    torch.save(dict(normed=output), output_path)
    extension = Path(torch_npu._C.__file__).resolve()
    report = dict(status="pass" if readonly and valid else "failed", boundary="torch_npu.npu_rms_norm",
        reference_operands=False, physical_device=int(os.environ["ASCEND_RT_VISIBLE_DEVICES"]), logical_device=0, invocation_pid=invocation_pid,
        pid=os.getpid(), runtime=runtime, operator_schema=str(torch.ops.npu.npu_rms_norm.default._schema),
        extension=str(extension), extension_sha256=_sha(extension), eps=payload["eps"],
        readonly_inputs_exact=readonly, finite_shape_dtype=valid,
        actual_inputs={name: dict(sha256=_tensor_sha(value), shape=list(value.shape), dtype=str(value.dtype))
                       for name, value in payload["tensors"].items()},
        inputs_sha256=_sha(args.directory / "inputs.pt"), outputs_sha256=_sha(output_path),
        runtime_config_sha256=_sha(args.runtime_config), product_boundary_sha256=_sha(Path(__file__)))
    (args.directory / "report.json").open("x").write(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Actual RMSNorm readonly/output integrity failed")


if __name__ == "__main__":
    _native_main()
