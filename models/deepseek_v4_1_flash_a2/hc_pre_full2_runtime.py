# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stage actual Prefill operands through the registered A2 HC-pre operator."""

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from .runtime_utils import validate_device, file_sha as _sha
from .full2_readback_runtime import admit_readonly, save_readonly_device_inputs, tensor_sha as _tensor_sha


OUTPUT_NAMES = ("collapsed", "post", "comb", "next_pre")


def hc_pre_host(hidden, function, scale, base, incoming, *, config, runtime_config, output_dir):
    """Execute HC-pre on actual CPU-staged product tensors on one explicitly selected visible device."""
    directory = Path(output_dir)
    directory.mkdir(exist_ok=False)
    settings = json.loads(Path(runtime_config).read_text())
    validate_device(settings)
    values = dict(hidden=hidden, function=function, scale=scale, base=base, incoming=incoming)
    if any(value.device.type != "cpu" for value in values.values()):
        raise ValueError("HC-pre requires CPU staging of actual product buffers")
    values = {name: value.contiguous() for name, value in values.items()}
    parameters = {name: config[name] for name in ("hc_mult", "hc_sinkhorn_iters", "rms_norm_eps", "hc_eps")}
    input_path = directory / "inputs.pt"
    torch.save(dict(tensors=values, parameters=parameters), input_path)
    env = dict(os.environ)
    vendor = Path(settings["hc_vendor"]).resolve()
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + ":" + env.get("ASCEND_CUSTOM_OPP_PATH", "")
    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    product_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = ":".join([str(product_root), *settings["vllm_pythonpath"]])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    command = [settings["vllm_python"], "-u", "-m", __name__, "--native", "--runtime-config",
               str(Path(runtime_config).resolve()), "--directory", str(directory.resolve())]
    with (directory / "run.log").open("x") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Native HC-pre exited {result.returncode}; inspect {directory / 'run.log'}")
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "pass" or not report["readonly_inputs_exact"]:
        raise RuntimeError("Actual-operand HC-pre integrity failed")
    if report["inputs_sha256"] != _sha(input_path) or report["parameters"] != parameters:
        raise RuntimeError("Actual HC-pre input contract changed")
    admit_readonly(report, directory, values, _sha)
    for name, value in values.items():
        if report["actual_inputs"][name]["sha256"] != _tensor_sha(value):
            raise RuntimeError("HC-pre consumed different actual product operands: " + name)
    output_path = directory / "outputs.pt"
    if report["outputs_sha256"] != _sha(output_path):
        raise RuntimeError("HC-pre output identity differs")
    outputs = torch.load(output_path, map_location="cpu", weights_only=True)
    return outputs, report


def _native_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.runtime_config.read_text())
    invocation_pid = validate_device(settings)
    payload = torch.load(args.directory / "inputs.pt", map_location="cpu", weights_only=True)
    values, parameters = payload["tensors"], payload["parameters"]
    hidden = values["hidden"]
    tokens, hc, width = hidden.shape
    expected = {
        "hidden": ((tokens, hc, width), torch.bfloat16),
        "function": (((hc + 2) * hc, hc * width), torch.float32),
        "scale": ((3,), torch.float32), "base": (((hc + 2) * hc,), torch.float32),
        "incoming": ((tokens, hc), torch.float32),
    }
    if tokens < 1 or hc != parameters["hc_mult"] or hc != 4:
        raise ValueError("HC-pre input dimensions differ from the four-stream contract")
    for name, (shape, dtype) in expected.items():
        value = values[name]
        if tuple(value.shape) != shape or value.dtype != dtype or not bool(torch.isfinite(value).all()):
            raise ValueError("Invalid actual HC-pre operand: " + name)
    import torch_npu

    torch.npu.set_device(0)
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        # Import directly to retain the loader's actual exception.
        importlib.import_module("vllm_ascend.vllm_ascend_C")
        raise RuntimeError("Upstream enable_custom_op returned False")
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    operator = torch.ops._C_ascend.npu_hc_pre_v3
    device_values = {name: value.to("npu:0") for name, value in values.items()}
    native_outputs = operator(
        device_values["hidden"], device_values["function"], device_values["scale"], device_values["base"],
        device_values["incoming"], hc_mult=hc, hc_sinkhorn_iters=parameters["hc_sinkhorn_iters"],
        norm_eps=parameters["rms_norm_eps"], hc_eps=parameters["hc_eps"],
    )
    torch.npu.synchronize()
    outputs = {name: value.cpu().contiguous() for name, value in zip(OUTPUT_NAMES, native_outputs)}
    output_shapes = {"collapsed": (tokens, width), "post": (tokens, hc),
                     "comb": (tokens, hc, hc), "next_pre": (tokens, hc)}
    valid = all(tuple(value.shape) == output_shapes[name] and bool(torch.isfinite(value).all())
                and value.dtype == (torch.bfloat16 if name == "collapsed" else torch.float32)
                for name, value in outputs.items())
    actual_readbacks, readonly = save_readonly_device_inputs(device_values, values, args.directory)
    output_path = args.directory / "outputs.pt"
    torch.save(outputs, output_path)
    mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if "/" in line and "libcust_op" in line}
    vendor = Path(settings["hc_vendor"]).resolve()
    manifest_path = Path(settings["hc_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    mapped_receipt = [{"path": path, "sha256": _sha(path)} for path in sorted(mapped)]
    (args.directory / "mapped-libraries.json").write_text(json.dumps(mapped_receipt, indent=2) + "\n")
    libraries = []
    for name in ("op_api/lib/libcust_opapi.so", "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"):
        path = (vendor / name).resolve()
        digest = _sha(path)
        if str(path) not in mapped or manifest["files"][name] != digest:
            raise RuntimeError("HC-pre loaded a different configured native library: " + str(path))
        libraries.append(dict(path=str(path), sha256=digest))
    extension_path = Path(extension.__file__).resolve()
    report = {
        "status": "pass" if readonly and valid else "failed", "boundary": "_C_ascend.npu_hc_pre_v3",
        "reference_operands": False, "physical_device": int(os.environ["ASCEND_RT_VISIBLE_DEVICES"]), "logical_device": 0,
        "invocation_pid": invocation_pid, "pid": os.getpid(), "python": platform.python_version(),
        "torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
        "CXX11_ABI": bool(torch._C._GLIBCXX_USE_CXX11_ABI), "operator_schema": str(operator.default._schema),
        "extension": str(extension_path), "extension_sha256": _sha(extension_path),
        "readonly_inputs_exact": readonly, "finite_shape_dtype": valid, "parameters": parameters,
        "actual_inputs": {name: dict(sha256=_tensor_sha(value), shape=list(value.shape), dtype=str(value.dtype))
                          for name, value in actual_readbacks.items()},
        "actual_consumer_readback": True, "readonly_actual_inputs_sha256": _sha(args.directory / "readonly-actual.pt"),
        "inputs_sha256": _sha(args.directory / "inputs.pt"), "outputs_sha256": _sha(output_path),
        "runtime_config_sha256": _sha(args.runtime_config), "product_boundary_sha256": _sha(Path(__file__)),
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"), "loaded_hc_libraries": libraries,
        "native_manifest_sha256": _sha(manifest_path),
    }
    (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Actual-operand HC-pre output or readonly gate failed")


if __name__ == "__main__":
    _native_main()
