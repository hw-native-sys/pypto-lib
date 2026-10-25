# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export original CI CPU inputs and goldens without native assembly or dispatch."""

import argparse
import ast
import concurrent.futures
from contextlib import ExitStack
import dataclasses
import enum
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
import sysconfig
from unittest.mock import patch


PYPTO_COMMIT = "afe81773c33ef041d97f81e473c06618b961ecda"
SIMPLER_COMMIT = "d8f997fabd1b6988b69a94feef61ab669dda0eea"
ISA_COMMIT = "b1d460ee8c001585ede36b68ba3678dc5a226bbf"
SOURCE_DIGEST = "9f0dbb5730c04b2a26c84c30dde63ee0a00ab62c0c0425579c88a73458a232dd"
SEQUENCES = {
    "c1a": [("l3_decode_c1a_full", 1), ("l3_decode_c1a_full", 72)],
    "c2a": [("l3_decode_c2a_full", 24), ("l3_decode_c2a_full", 72)],
    "sparse": [
        (name, rows)
        for rows in (1, 7, 12, 13)
        for name in ("l2_sparse_attention", "l2_sparse_attention_bf16", "l2_c2a_attention")
    ] + [("l2_c2a_attention", rows) for rows in (32, 33, 80)],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    text = str(value)
    assert not re.search(r"0x[0-9a-fA-F]{6,}", text), type(value)
    return {"type": type(value).__module__ + "." + type(value).__qualname__, "value": text}


def write(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--entry", choices=tuple(SEQUENCES), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
output = args.output.resolve()
assert not os.environ.get("TASK_DEVICE"), "CPU export must run outside device allocation"
assert platform.machine() == "x86_64", "This diagnostic specifically captures CI x86 references"
assert not output.exists()
output.mkdir(parents=True)


def source_inventory():
    names = git(repo, "ls-files", "-z", "*.py").split("\0")
    names = sorted(n for n in names if n and n != ".github/scripts/export_ci_fixtures.py")
    values = {name: sha(repo / name) for name in names}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len(values) == 246 and digest == SOURCE_DIGEST
    return values


initial_source = source_inventory()
pypto_source = Path(os.environ["PYPTO_SRC"]).resolve()
assert git(pypto_source, "rev-parse", "HEAD") == PYPTO_COMMIT
assert git(pypto_source / "runtime", "rev-parse", "HEAD") == SIMPLER_COMMIT
assert (pypto_source / "runtime/pto_isa.pin").read_text().strip() == ISA_COMMIT
assert os.environ["PTOAS_VERSION"] == "v0.67"
for name in ("ThreadPoolExecutor", "ProcessPoolExecutor"):
    original = getattr(concurrent.futures, name)

    def bounded(max_workers=None, *positional, _original=original, **kwargs):
        return _original(min(2, max_workers) if max_workers is not None else 2, *positional, **kwargs)

    setattr(concurrent.futures, name, bounded)

filename = "sparse_attention.py" if args.entry == "sparse" else f"decode_{args.entry}_full.py"
source = repo / "models/deepseek_v4_1_flash" / filename
sys.argv = [str(source), "-p", "a5", "-d", "0", "--compile-only"]
sys.path[:0] = [str(source.parent), str(repo)]
sys.dont_write_bytecode = True
os.environ["PYPTO_BENCH"] = "0"
os.environ.pop("SIMPLER_PMU_EVENT_TYPE", None)

import torch
import golden
from golden import runner
from pypto.ir import CompiledProgram, DistributedCompiledProgram
from pypto.runtime import device_runner
from pypto.runtime.worker import ChipWorker
from simpler_setup.runtime_compiler import RuntimeCompiler

assert torch.__version__.split("+")[0] == "2.10.0"
venv_roots = {Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib")}
module_hashes = {}
for name in ("pypto", "pypto.pypto_core", "pypto.jit.decorator", "simpler"):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert any(path.is_relative_to(parent) for parent in venv_roots), name
    module_hashes[name] = sha(path)
for relative in ("__init__.py", "jit/decorator.py"):
    installed = Path(importlib.import_module("pypto").__file__).parent / relative
    assert sha(installed) == sha(pypto_source / "python/pypto" / relative)

host = dict(
    machine=platform.machine(), platform=platform.platform(), python=sys.version,
    torch_version=torch.__version__, torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
    torch_threads=torch.get_num_threads(),
    torch_interop_threads=torch.get_num_interop_threads(), torch_config=torch.__config__.show(),
    thread_environment={name: os.environ.get(name) for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "OMP_DYNAMIC", "MKL_DYNAMIC", "ATEN_CPU_CAPABILITY",
    )},
    pypto_commit=PYPTO_COMMIT, pypto_tree=git(pypto_source, "rev-parse", "HEAD^{tree}"),
    simpler_commit=SIMPLER_COMMIT, pto_isa_commit=ISA_COMMIT, ptoas_version="0.67",
    module_sha256=module_hashes, source_sha256=initial_source,
    workflow_sha256=sha(repo / ".github/workflows/ci.yml"), driver_sha256=sha(__file__),
    main_ast_sha256=hashlib.sha256(ast.dump(next(
        n for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)
    )).encode()).hexdigest(),
    compile_workers=2, torch_thread_count_not_overridden=True,
)
write(output / "host.json", host)
original_run = golden.run
calls, forbidden = [], []
completed = False


class SequenceComplete(Exception):
    pass


def forbid(*unused, **ignored):
    forbidden.append("native assembly or device boundary")
    raise AssertionError(forbidden[-1])


def tensor_record(tensor, path):
    value = tensor.detach().cpu().contiguous()
    raw = memoryview(value.reshape(-1).view(torch.uint8).numpy())
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.numel() * value.element_size(),
                raw_sha256=hashlib.sha256(raw).hexdigest(), file_sha256=sha(path))


def saved_input_tensors(tensor_specs, scalar_specs_eff, values):
    inputs = {s.name: values[s.name] for s in tensor_specs if s.is_input}
    inputs.update({name: spec.value for name, spec in scalar_specs_eff.items()})
    return inputs


def intercept(*positional, **kwargs):
    if positional:
        assert len(positional) == 2 and "fn" not in kwargs and "specs" not in kwargs
        kwargs = dict(kwargs, fn=positional[0], specs=positional[1])
    ordinal = len(calls)
    expected_fn, rows = SEQUENCES[args.entry][ordinal]
    fn, specs = kwargs["fn"], kwargs["specs"]
    assert fn.__name__ == expected_fn and list(specs[0].shape)[0] == rows
    assert kwargs["compile_only"] and not kwargs.get("runtime_dir") and not kwargs.get("golden_data")
    tensor_specs = [s for s in specs if isinstance(s, golden.TensorSpec)]
    scalar_specs = [s for s in specs if isinstance(s, golden.ScalarSpec)]
    call_dir = output / f"{ordinal + 1:02d}-{expected_fn}-{rows}"
    call_dir.mkdir()
    compiled = original_run(**dict(kwargs, compile_only=True, save_data=False))
    assert compiled.passed, compiled.error
    values, scalars = runner._prepare_inputs(specs, tensor_specs, scalar_specs, None, call_dir, True)
    expected = runner._compute_golden(specs, tensor_specs, scalars, values, call_dir, None, kwargs["golden_fn"], True)
    runner._validate(tensor_specs, {**values, **expected}, expected,
                     kwargs.get("rtol", 1e-5), kwargs.get("atol", 1e-5), kwargs["compare_fn"], scalars)
    fields = {}
    for phase, tensors in (("in", saved_input_tensors(tensor_specs, scalars, values)), ("out", expected)):
        fields[phase] = {name: tensor_record(value, call_dir / "data" / phase / (name + ".pt"))
                         for name, value in tensors.items()}
    record = dict(
        function=expected_fn, rows=rows, fields=fields,
        specs=[dict(name=s.name, shape=list(s.shape), dtype=str(s.dtype), direction=s.direction)
               if isinstance(s, golden.TensorSpec) else dict(
                   name=s.name, dtype=str(s.dtype), value=s.to_python(), compile_runtime=s.compile_runtime)
               for s in specs],
        original_comparators=list(kwargs["compare_fn"]),
        rtol=kwargs.get("rtol", 1e-5), atol=kwargs.get("atol", 1e-5),
        config=runner.normalize_config(kwargs["config"]),
        compiled_directory_name=Path(compiled.work_dir).name,
        cpu_reference_self_check_passed=True, native_execution=False,
    )
    write(call_dir / "manifest.json", record)
    calls.append(dict(directory=call_dir.name, manifest_sha256=sha(call_dir / "manifest.json")))
    if len(calls) == len(SEQUENCES[args.entry]):
        raise SequenceComplete
    return compiled


error = None
try:
    with ExitStack() as stack:
        targets = [(runner, "_dispatch"), (device_runner, "_execute_on_device"),
                   (device_runner, "_compile_and_assemble"), (ChipWorker, "__init__"),
                   (CompiledProgram, "__call__"), (DistributedCompiledProgram, "__call__"),
                   (RuntimeCompiler, "compile"), (RuntimeCompiler, "compile_sim_context")]
        targets += [(device_runner.KernelCompiler, name) for name in dir(device_runner.KernelCompiler)
                    if name.startswith("compile") and callable(getattr(device_runner.KernelCompiler, name))]
        for target, name in targets:
            stack.enter_context(patch.object(target, name, forbid))
        stack.enter_context(patch.object(golden, "run", intercept))
        try:
            runpy.run_path(str(source), run_name="__main__")
        except SequenceComplete:
            completed = True
        assert completed and len(calls) == len(SEQUENCES[args.entry])
except BaseException as exc:
    error = repr(exc)
    raise
finally:
    unchanged = source_inventory() == initial_source
    unchanged = unchanged and sha(__file__) == host["driver_sha256"]
    unchanged = unchanged and sha(repo / ".github/workflows/ci.yml") == host["workflow_sha256"]
    write(output / "summary.json", dict(
        passed=completed and unchanged and not forbidden, error=error, calls=calls,
        expected_sequence=SEQUENCES[args.entry], forbidden_calls=forbidden,
        source_unchanged=unchanged, host_sha256=sha(output / "host.json"),
        scope="Original CPU inputs/goldens only, original compile-only ABI and callback self-check; no native assembly or device execution.",
    ))
