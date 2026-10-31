# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stage actual PyPTO operands through the native A2 MoE dispatch runtime."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from .moe_boundary import dispatch_full384
from .runtime_utils import validate_device, file_sha as _sha, tensor_sha as _tensor_sha



def dispatch_host_full384(hidden, route_ids, *, runtime_config, output_dir):
    """Execute the registered native operator on actual CPU-staged operands."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    settings = json.loads(Path(runtime_config).read_text())
    validate_device(settings)
    if hidden.device.type != "cpu" or route_ids.device.type != "cpu":
        raise ValueError("Expected CPU staging of actual PyPTO output buffers")
    inputs = {"hidden": hidden.contiguous(), "route_ids": route_ids.contiguous()}
    input_path = output_dir / "inputs.pt"
    torch.save(inputs, input_path)
    env = dict(os.environ)
    product_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = ":".join([str(product_root), *settings["vllm_pythonpath"]])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    command = [settings["vllm_python"], "-u", "-m", __name__, "--native", "--runtime-config",
               str(Path(runtime_config).resolve()), "--directory", str(output_dir.resolve())]
    with (output_dir / "run.log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Native dispatch exited {result.returncode}; inspect {output_dir / 'run.log'}")
    report = json.loads((output_dir / "report.json").read_text())
    if report["status"] != "pass" or not report["readonly_inputs_exact"]:
        raise RuntimeError("Native dispatch runtime integrity failed")
    if report["inputs_sha256"] != _sha(input_path):
        raise RuntimeError("Actual dispatch inputs changed during native execution")
    for name, value in inputs.items():
        if report["actual_input_" + name + "_sha256"] != _tensor_sha(value):
            raise RuntimeError("Native dispatch consumed different actual product operands")
    output_path = output_dir / "outputs.pt"
    if _sha(output_path) != report["outputs_sha256"]:
        raise RuntimeError("Native dispatch output identity differs")
    outputs = torch.load(output_path, map_location="cpu", weights_only=True)
    return tuple(outputs[key] for key in ("payload", "expanded", "counts", "scale")), report


def _native_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.runtime_config.read_text())
    invocation_pid = validate_device(settings)
    values = torch.load(args.directory / "inputs.pt", map_location="cpu", weights_only=True)
    hidden, route_ids = values["hidden"], values["route_ids"]
    if hidden.dtype != torch.bfloat16 or route_ids.dtype != torch.int32:
        raise ValueError("Expected actual BF16 hidden and INT32 routes")
    if not bool(torch.isfinite(hidden).all()) or not bool(((route_ids >= 0) & (route_ids < 384)).all()):
        raise ValueError("Actual dispatch operands are nonfinite or have illegal expert IDs")
    import torch_npu

    torch.npu.set_device(0)
    device_hidden, device_routes = hidden.to("npu:0"), route_ids.to("npu:0")
    dispatched = dispatch_full384(device_hidden, device_routes)
    torch.npu.synchronize()
    outputs = {key: value.cpu().contiguous() for key, value in
               zip(("payload", "expanded", "counts", "scale"), dispatched)}
    readonly = torch.equal(device_hidden.cpu(), hidden) and torch.equal(device_routes.cpu(), route_ids)
    output_path = args.directory / "outputs.pt"
    torch.save(outputs, output_path)
    extension = Path(torch_npu._C.__file__).resolve()
    report = {
        "status": "pass" if readonly else "failed", "boundary": "torch_npu.npu_moe_init_routing_v2",
        "reference_operands": False, "physical_device": int(os.environ["ASCEND_RT_VISIBLE_DEVICES"]), "logical_device": 0,
        "invocation_pid": invocation_pid, "pid": os.getpid(), "python": platform.python_version(),
        "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
        "CXX11_ABI": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "extension": str(extension), "extension_sha256": _sha(extension),
        "operator_schema": str(torch.ops.npu.npu_moe_init_routing_v2.default._schema),
        "readonly_inputs_exact": readonly,
        "actual_input_hidden_sha256": _tensor_sha(hidden), "actual_input_route_ids_sha256": _tensor_sha(route_ids),
        "inputs_sha256": _sha(args.directory / "inputs.pt"), "outputs_sha256": _sha(output_path),
        "runtime_config_sha256": _sha(args.runtime_config), "product_boundary_sha256": _sha(Path(__file__)),
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
    }
    (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not readonly:
        raise RuntimeError("Native dispatch mutated actual product operands")


if __name__ == "__main__":
    _native_main()
