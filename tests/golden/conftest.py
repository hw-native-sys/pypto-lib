# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pytest conftest for golden tests — adds repo root to sys.path.

Also installs stub ``pypto`` / ``pypto.ir`` / ``pypto.runtime`` modules when
the real pypto is not importable, so the golden unit tests can run in a
CPU-only CI job without building the compiler. The stubs expose the
attributes the tests patch and the side-effect helpers ``runner.py`` calls
on the runtime_dir replay path (``invalidate_binary_cache``,
``rebuild_kernel_cpp_from_pto``, ``configure_log``). If real pypto is
installed it is used as-is.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _install_pypto_stubs() -> None:
    if importlib.util.find_spec("pypto") is not None:
        return

    import enum

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "stub pypto: this function must be patched in tests"
        )

    class BackendType(enum.Enum):
        Ascend910B = "Ascend910B"
        Ascend950 = "Ascend950"

    pypto = types.ModuleType("pypto")
    pypto.__path__ = []  # mark as package so submodule imports resolve
    ir = types.ModuleType("pypto.ir")
    runtime = types.ModuleType("pypto.runtime")
    runtime.__path__ = []
    log_config = types.ModuleType("pypto.runtime.log_config")
    debug = types.ModuleType("pypto.runtime.debug")
    debug.__path__ = []
    replay = types.ModuleType("pypto.runtime.debug.replay")
    pto_rebuild = types.ModuleType("pypto.runtime.debug.pto_rebuild")
    backend = types.ModuleType("pypto.backend")

    # Tests that observe these patch them; the stub defaults are silent
    # no-ops so the runtime_dir replay path can flow through without
    # exploding when a test doesn't care.
    ir.compile = _unavailable
    runtime.execute_compiled = _unavailable
    log_config.configure_log = lambda *_a, **_k: None
    replay.invalidate_binary_cache = lambda *_a, **_k: None
    pto_rebuild.rebuild_kernel_cpp_from_pto = lambda *_a, **_k: []
    backend.BackendType = BackendType

    pypto.ir = ir
    pypto.runtime = runtime
    pypto.backend = backend
    runtime.log_config = log_config
    runtime.debug = debug
    debug.replay = replay
    debug.pto_rebuild = pto_rebuild

    sys.modules["pypto"] = pypto
    sys.modules["pypto.ir"] = ir
    sys.modules["pypto.runtime"] = runtime
    sys.modules["pypto.runtime.log_config"] = log_config
    sys.modules["pypto.runtime.debug"] = debug
    sys.modules["pypto.runtime.debug.replay"] = replay
    sys.modules["pypto.runtime.debug.pto_rebuild"] = pto_rebuild
    sys.modules["pypto.backend"] = backend


_install_pypto_stubs()


# Bench knobs golden.runner reads from the environment. An exported PYPTO_BENCH
# sends every run() down the benchmark path — which the stub pypto cannot serve —
# and fails a couple of dozen unrelated tests, so clear them for the whole suite;
# the tests that exercise these knobs set them explicitly via monkeypatch.
_BENCH_ENV = ("PYPTO_BENCH", "PYPTO_BENCH_RAW", "PYPTO_BENCH_ROUNDS", "PYPTO_BENCH_WARMUP")


@pytest.fixture(autouse=True)
def _isolate_bench_env(monkeypatch):
    for name in _BENCH_ENV:
        monkeypatch.delenv(name, raising=False)
