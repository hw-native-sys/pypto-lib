# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pass one prepared golden dispatch to the PR/nightly device allocator."""

import argparse
import fcntl
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from golden.runner import (
    _dispatch,
    _is_l3,
    _packed_resident_inputs,
    _reload_from_dir,
    _run_l3_resident,
    _Stage,
    _validate_compiled_spec_abi,
    normalize_config,
)
from golden.spec import ScalarSpec, SparseZero, TensorSpec


def _stage_windows(sources, directory):
    if sources is None:
        return None
    from pypto.runtime.window_sources import normalize_window_sources

    windows = {}
    for index, (key, ranks) in enumerate(normalize_window_sources(sources).items()):
        windows[key] = []
        for rank, source in enumerate(ranks):
            if isinstance(source, torch.Tensor):
                windows[key].append(source)
                continue
            path = directory / f"window-{index}-{rank}"
            position = source.tell()
            try:
                source.seek(0)
                with path.open("wb") as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
            finally:
                source.seek(position)
            windows[key].append(path.name)
    return windows


@contextmanager
def _slot(directory):
    slots = list(directory.iterdir())
    while True:
        for path in slots:
            with path.open("rb") as slot:
                try:
                    fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                yield slot
                return
        time.sleep(0.1)


def _assemble(compiled, cfg):
    work_dir = Path(compiled.output_dir).resolve()
    # Compile the binaries in the directory the allocated process will reload.
    with _Stage("assemble binaries"):
        replay = _reload_from_dir(work_dir, cfg)
        if _is_l3(compiled):
            from pypto.runtime.distributed_runner import _assemble_chip_callables

            _assemble_chip_callables(replay)
            return asdict(compiled._distributed_config)
        replay.load()
    return None


@contextmanager
def prepare_dispatch(count):
    """Bound compilation, golden generation, and the lifetime of prepared tensors separately."""
    slots = Path(os.environ["PYPTO_CI_PREPARE_SLOTS"])
    with _slot(slots / "compile") as compiler, ExitStack() as live:
        def prepare_golden(compiled, cfg):
            distributed = _assemble(compiled, cfg)
            compiler.close()
            # Take capacity for the tensors before generating any harness inputs.
            live.enter_context(_slot(slots / "inflight"))
            golden = live.enter_context(_slot(slots / "golden"))
            return partial(submit_dispatch, count, golden.close, distributed)

        yield prepare_golden


def submit_dispatch(
    count, release_golden, distributed, compiled, specs, tensors, scalars, cfg, *,
    readback_outputs, window_sources, persistent, reset_persistent_windows,
):
    work_dir = Path(compiled.output_dir).resolve()
    packed = _packed_resident_inputs(compiled, specs, tensors)
    args = []
    for spec in specs:
        if isinstance(spec, ScalarSpec):
            args.append((spec.name, None, spec.dtype, None, scalars[spec.name].value))
        else:
            value = packed.get(spec.name, tensors[spec.name])
            initial = value if spec.is_input else None
            initializer = spec.init_value
            if spec.direction == "in" and isinstance(initializer, SparseZero) and initializer.owns(value):
                initial = "sparse_zero"
            args.append((spec.name, list(value.shape), spec.dtype, spec.resident, initial))
    options = asdict(cfg.run_options())
    options.update(options.pop("dfx"))
    # Keep files visible to both the CI runner and task-submit's checkout.
    scratch = Path.cwd() / "build_output"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ci-dispatch-", dir=scratch) as temp:
        directory = Path(temp)
        request = {
            "work_dir": str(work_dir),
            "args": args,
            "options": options,
            "distributed": distributed,
            "count": count,
            "readback_outputs": readback_outputs,
            "windows": _stage_windows(window_sources, directory),
            "persistent": persistent,
            "reset_persistent_windows": reset_persistent_windows,
        }
        torch.save(request, directory / "request.pt")
        release_golden()
        print(f"[device] {work_dir.name} ready, requesting {count} device(s)", flush=True)
        subprocess.run([
            "bash", ".github/scripts/run_device_tests.sh", "--submit", str(count),
            ".github/scripts/ci_device.py", str(directory),
        ], check=True)
        # Require the allocated worker's actual outputs before reporting success.
        outputs = torch.load(directory / "outputs.pt", weights_only=True, mmap=True)
        prepared_worker = window_sources is not None or any(
            isinstance(s, TensorSpec) and s.is_resident for s in specs
        )
        for spec in specs:
            if isinstance(spec, TensorSpec) and spec.is_output:
                # The prepared worker uses contiguous host buffers for strided inputs and outputs.
                target = tensors[spec.name]
                if prepared_worker:
                    target = tensors[spec.name] = target.contiguous()
                target.copy_(outputs[spec.name])


def execute(directory, devices):
    request = torch.load(directory / "request.pt", weights_only=True, mmap=True)
    if len(devices) != request["count"] or len(set(devices)) != len(devices) or any(d < 0 for d in devices):
        raise ValueError(f"expected {request['count']} allocated devices, got {devices}")

    # Logical ranks come from the entry's CLI and may select a subset of its allocation.
    def physical(rank):
        if not 0 <= rank < len(devices):
            raise ValueError(f"logical device {rank} is outside the allocation {devices}")
        return devices[rank]

    options = request["options"]
    options["device_id"] = physical(options["device_id"])
    options["backend_type"] = options.pop("platform")
    options["execution_mode"] = False
    if request["distributed"] is not None:
        from pypto.ir import DistributedConfig

        dc = request["distributed"]
        dc["device_ids"] = [physical(rank) for rank in dc["device_ids"]]
        options["distributed_config"] = DistributedConfig(**dc)
    cfg = normalize_config(options)
    compiled = _reload_from_dir(Path(request["work_dir"]), cfg)
    specs, tensors, scalars = [], {}, {}
    for name, shape, dtype, resident, value in request["args"]:
        if shape is None:
            spec = scalars[name] = ScalarSpec(name, dtype, value)
        else:
            spec = TensorSpec(name, shape, dtype, resident=resident)
            if value is None:
                value = torch.zeros(shape, dtype=dtype)
            elif isinstance(value, str) and value == "sparse_zero":
                value = SparseZero(shape, dtype)()
            tensors[name] = value
        specs.append(spec)
    _validate_compiled_spec_abi(compiled, specs)
    windows = request["windows"]
    if windows is not None or any(isinstance(s, TensorSpec) and s.is_resident for s in specs):
        with ExitStack() as files:
            if windows is not None:
                for key, ranks in windows.items():
                    windows[key] = [
                        files.enter_context((directory / src).open("rb")) if isinstance(src, str) else src
                        for src in ranks
                    ]
            _run_l3_resident(
                compiled, specs, tensors, scalars, cfg, None, 0, 0, {}, benchmark_enabled=False,
                window_sources=windows, persistent=request["persistent"],
                reset_persistent_windows=request["reset_persistent_windows"],
                readback_outputs=request["readback_outputs"],
            )
    else:
        _dispatch(compiled, specs, tensors, scalars, cfg)
    outputs = {s.name: tensors[s.name] for s in specs if isinstance(s, TensorSpec) and s.is_output}
    torch.save(outputs, directory / "outputs.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("-d", required=True)
    args = parser.parse_args()
    execute(args.directory, [int(device) for device in args.d.split(",")])
