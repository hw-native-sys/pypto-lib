# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stage actual product index operands through registered A2 QLI operators."""

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace

import torch

from .runtime_utils import validate_device, file_sha as _sha
from .qli_boundary import QLIContract, run_qli_native_boundary
from .full2_readback_runtime import admit_readonly, save_readonly_device_inputs


def _tensor_sha(value):
    flat = torch.empty(value.numel(), dtype=value.dtype, device="cpu")
    flat.copy_(value.detach().cpu().reshape(-1))
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def qli_host(
    query_i8, weights_fp16, query_scale_fp16, key_i8, key_scale_fp16,
    *, scheduler, expected_libraries, runtime_config, output_dir, contract=QLIContract(), candidates=None,
):
    """Execute metadata and sparse selection using only actual staged product operands."""
    settings = json.loads(Path(runtime_config).read_text())
    validate_device(settings)
    directory = Path(output_dir)
    directory.mkdir(exist_ok=False)
    tensors = dict(query_i8=query_i8, weights_fp16=weights_fp16, query_scale_fp16=query_scale_fp16,
                   key_i8=key_i8, key_scale_fp16=key_scale_fp16)
    if "cmp_residual" not in scheduler:
        raise ValueError("Explicit residual presence is required, including None for ratio1")
    residual = scheduler["cmp_residual"]
    if contract.compress_ratio == 1 and residual is not None:
        raise ValueError("Source ratio1 requires cmp_residual=None; a zero tensor is not its ABI")
    if contract.compress_ratio == 2 and residual is None:
        raise ValueError("Source ratio2 requires its actual compression residual tensor")
    tensors.update({name: scheduler[name] for name in ("block_table", "query_start_loc", "cache_seq_lens")})
    if residual is not None:
        tensors["cmp_residual"] = residual
    if candidates is not None:
        tensors["candidates"] = candidates
    if any(value.device.type != "cpu" for value in tensors.values()):
        raise ValueError("QLI requires CPU staging of actual product tensors")
    tensors = {name: value.contiguous() for name, value in tensors.items()}
    if len(expected_libraries) < 2:
        raise ValueError("Frozen QLI library identities are required")
    for path, digest in expected_libraries.items():
        if _sha(path) != digest:
            raise RuntimeError("QLI runtime dependency changed: " + path)
    scalar_names = ("num_reqs", "max_query_len", "max_cache_seq_len")
    scalar = {name: int(scheduler[name]) for name in scalar_names}
    input_path = directory / "inputs.pt"
    torch.save(dict(tensors=tensors, scheduler=dict(scalar, residual_is_none=residual is None), contract=asdict(contract),
                    expected_libraries=expected_libraries), input_path)
    env = dict(os.environ)
    product_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = ":".join([str(product_root), *settings["vllm_pythonpath"]])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    command = [settings["vllm_python"], "-u", "-m", __name__, "--native", "--runtime-config",
               str(Path(runtime_config).resolve()), "--directory", str(directory.resolve())]
    with (directory / "run.log").open("x") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Native QLI exited {result.returncode}; inspect {directory / 'run.log'}")
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "pass" or report["inputs_sha256"] != _sha(input_path):
        raise RuntimeError("QLI actual-operand integrity or runtime gate failed")
    admit_readonly(report, directory, tensors, _sha)
    for name, value in tensors.items():
        if report["actual_inputs"][name]["sha256"] != _tensor_sha(value):
            raise RuntimeError("QLI consumed different actual operands: " + name)
    path = directory / "outputs.pt"
    if report["outputs_sha256"] != _sha(path):
        raise RuntimeError("QLI actual output identity changed")
    return torch.load(path, map_location="cpu", weights_only=True), report


def _native_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.runtime_config.read_text())
    invocation_pid = validate_device(settings)
    payload = torch.load(args.directory / "inputs.pt", map_location="cpu", weights_only=True)
    values, scalar = payload["tensors"], payload["scheduler"]
    contract = QLIContract(**payload["contract"])
    active = values["query_i8"].shape[0]
    requests = scalar["num_reqs"]
    for name in ("block_table", "query_start_loc", "cache_seq_lens") + (() if scalar["residual_is_none"] else ("cmp_residual",)):
        if values[name].dtype != torch.int32:
            raise ValueError("QLI scheduler tensors must be INT32: " + name)
    starts, lengths = (values[name] for name in ("query_start_loc", "cache_seq_lens"))
    residual = None if scalar["residual_is_none"] else values["cmp_residual"]
    if scalar["residual_is_none"] != (contract.compress_ratio == 1):
        raise ValueError("Residual nullability differs from the original compression ratio")
    if ("cmp_residual" in values) == scalar["residual_is_none"]:
        raise ValueError("Residual tensor/scalar partition differs")
    if starts.shape != (requests + 1,) or int(starts[0]) != 0 or int(starts[-1]) != active:
        raise ValueError("QLI query starts do not partition the actual token rows")
    if lengths.shape != (requests,) or residual is not None and residual.shape != (requests,) or bool((lengths < 0).any()):
        raise ValueError("QLI compressed lengths or residual ABI differs")
    if residual is not None and not bool(((residual >= 0) & (residual < contract.compress_ratio)).all()):
        raise ValueError("QLI compression residual is outside the source ratio")
    if scalar["max_query_len"] != int((starts[1:] - starts[:-1]).max()):
        raise ValueError("QLI maximum query length differs from actual request partition")
    if scalar["max_cache_seq_len"] != int(lengths.max()) or scalar["max_cache_seq_len"] < 1:
        raise ValueError("QLI requires actual nonempty compressed cache lengths")
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    if not enable_custom_op():
        importlib.import_module("vllm_ascend.vllm_ascend_C")
        raise RuntimeError("Upstream custom operator initialization returned False")
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    device = {name: value.to("npu:0") for name, value in values.items()}
    if residual is None:
        device["cmp_residual"] = None
    metadata_op = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata
    metadata = metadata_op(
        32, 1, 128, contract.topk, 2,
        cu_seqlens_q=device["query_start_loc"], seqused_k=device["cache_seq_lens"],
        cmp_residual_k=device["cmp_residual"], batch_size=requests,
        max_seqlen_q=scalar["max_query_len"], max_seqlen_k=scalar["max_cache_seq_len"],
        layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=contract.compress_ratio,
    )
    source_metadata = SimpleNamespace(**scalar, **{name: device[name] for name in (
        "block_table", "query_start_loc", "cache_seq_lens", "cmp_residual")}, qli_metadata=metadata)
    selected, scores, candidate_out = run_qli_native_boundary(
        device["query_i8"], device["weights_fp16"], device["query_scale_fp16"],
        device["key_i8"], device["key_scale_fp16"], source_metadata,
        contract=contract, candidates=device.get("candidates"),
    )
    torch.npu.synchronize()
    actual_readbacks, readonly = save_readonly_device_inputs(device, values, args.directory)
    outputs = dict(raw_selected=selected.cpu().contiguous(), scores=scores.cpu().contiguous(),
                   candidates=candidate_out.cpu().contiguous() if isinstance(candidate_out, torch.Tensor) else candidate_out,
                   metadata=metadata.cpu().contiguous())
    valid = tuple(selected.shape) == (active, contract.topk) and selected.dtype == torch.int32
    mapped = {str(Path(line.split()[-1]).resolve()) for line in Path("/proc/self/maps").read_text().splitlines()
              if line.split()[-1].startswith("/")}
    libraries = []
    for path, digest in payload["expected_libraries"].items():
        path = str(Path(path).resolve())
        if path not in mapped or _sha(path) != digest:
            raise RuntimeError("QLI loaded a different native dependency: " + path)
        libraries.append(dict(path=path, sha256=digest))
    output_path = args.directory / "outputs.pt"
    torch.save(outputs, output_path)
    report = dict(
        status="pass" if readonly and valid else "failed", boundary="_C_ascend.npu_quant_lightning_indexer_v3",
        metadata_boundary="_C_ascend.npu_quant_lightning_indexer_v2_metadata",
        metadata_built_from_actual_scheduler=True, reference_operands=False,
        physical_device=int(os.environ["ASCEND_RT_VISIBLE_DEVICES"]), logical_device=0, invocation_pid=invocation_pid, pid=os.getpid(),
        python=platform.python_version(), torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
        CXX11_ABI=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        extension=str(Path(extension.__file__).resolve()), extension_sha256=_sha(extension.__file__),
        operator_schema=str(torch.ops._C_ascend.npu_quant_lightning_indexer_v3.default._schema),
        metadata_schema=str(metadata_op.default._schema), loaded_libraries=libraries,
        readonly_inputs_exact=readonly, output_shape_dtype=valid, scheduler=scalar, contract=asdict(contract),
        actual_inputs={name: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=_tensor_sha(value))
                       for name, value in actual_readbacks.items()},
        actual_consumer_readback=True, readonly_actual_inputs_sha256=_sha(args.directory / "readonly-actual.pt"),
        inputs_sha256=_sha(args.directory / "inputs.pt"), outputs_sha256=_sha(output_path),
        product_boundary_sha256=_sha(Path(__file__)), runtime_config_sha256=_sha(args.runtime_config),
        custom_opp_path=os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
    )
    (args.directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("QLI actual-operand readonly or output ABI check failed")


if __name__ == "__main__":
    _native_main()
