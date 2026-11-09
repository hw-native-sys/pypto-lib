# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load the DSpark chunked prefill forward for batch-one TP8 serving."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import threading


_IMPORT_LOCK = threading.RLock()


@contextmanager
def _dspark_prefill_import():
    """Isolate the upstream model's import-time TP4/EP8 configuration."""
    source = Path(__file__).resolve().parent.parent / "deepseek_v4_flash_dspark"
    names = tuple(path.stem for path in source.glob("*.py"))
    missing = object()
    with _IMPORT_LOCK:
        old_modules = {name: sys.modules.get(name, missing) for name in names}
        old_path, old_argv = list(sys.path), list(sys.argv)
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(source))
        sys.argv = ["lowlat-prefill", "--tp", "4", "--ep", "8", "--dp", "2", "--lm-head-tp", "4"]
        try:
            yield
        finally:
            sys.argv = old_argv
            sys.path[:] = old_path
            for name, module in old_modules.items():
                if module is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


def load_prefill_program():
    """Return the existing chunked 43-layer prefill JIT entry and its module.

    Load during single-threaded setup, before creating Workers. The two CP4
    groups must each receive the same batch-one prompt; EP8 still spans both
    groups. Compile or register the returned ``l3_prefill_fwd`` normally.
    """
    with _dspark_prefill_import():
        module = importlib.import_module("prefill_fwd")
        if module.N_RANKS != 8 or module.TP_SIZE != 4 or module.LM_HEAD_TP_SIZE != 4:
            raise RuntimeError("Unexpected DSpark prefill parallel geometry")
        if module.N_EXPERTS_GLOBAL != 256 or module.FWD_NUM_LAYERS != 43:
            raise RuntimeError("Unexpected DSpark prefill checkpoint geometry")
        return module.l3_prefill_fwd, module


def prefill_arguments(metadata, bindings):
    """Order actual generated prefill arguments by their metadata names."""
    rows = metadata.get("params")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Actual generated prefill parameter metadata is required")
    names = [row["name"].split("__ssa", 1)[0] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Generated prefill parameter names are not unique")
    missing = [name for name in names if name not in bindings]
    if missing:
        raise ValueError(f"Missing prefill arguments: {', '.join(missing)}")
    return tuple(bindings[name] for name in names)


def prepare_prefill_bindings(worker, metadata, caches, weights, inputs, scratch):
    """Bind the real prefill ABI to shared decode caches and prefill inputs."""
    from .prefill_cache import prefill_cache_views

    bindings = {}
    for group in (prefill_cache_views(caches, worker=worker), weights, inputs, scratch):
        overlap = bindings.keys() & group.keys()
        if overlap:
            raise ValueError(f"Duplicate prefill bindings: {', '.join(sorted(overlap))}")
        bindings.update(group)
    prefill_arguments(metadata, bindings)
    return bindings


def submit_prefill(worker, program, metadata, bindings, *, config=None):
    """Submit a complete prefill chunk and wait for its real device result."""
    arguments = prefill_arguments(metadata, bindings)
    return worker.submit(program, *arguments, config=config).result()
