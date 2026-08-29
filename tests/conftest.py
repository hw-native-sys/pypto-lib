# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Repo-wide conftest for the test tree — pypto is optional.

Installs stub ``pypto`` / ``pypto.ir`` / ``pypto.runtime`` modules when the
real pypto is not importable, so the unit tests can run in a CPU-only CI job
without building the compiler. The stubs expose the attributes the tests patch,
the side-effect helpers ``golden/runner.py`` calls on the runtime_dir replay
path (``invalidate_binary_cache``, ``rebuild_kernel_cpp_from_pto``,
``configure_log``), and the ``@pl.jit.host`` decorator
``models/qwen3_14b/contract.py`` applies to its serving wrappers. If real pypto
is installed it is used as-is.

This lives at the tree root because more than one subdirectory needs it, and
two competing stubs cannot coexist: whichever conftest ran first would win
``sys.modules`` and starve the other of the attributes it added.
"""

import importlib.machinery
import importlib.util
import sys
import types


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

    class RunConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _TypeSpec:
        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    def _identity_jit(fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    _identity_jit.inline = _identity_jit
    _identity_jit.host = _identity_jit

    pypto = types.ModuleType("pypto")
    pypto.__path__ = []  # mark as package so submodule imports resolve
    pypto.__pypto_stub__ = True  # lets callers tell the stand-in from a real install
    language = types.ModuleType("pypto.language")
    ir = types.ModuleType("pypto.ir")
    runtime = types.ModuleType("pypto.runtime")
    runtime.__path__ = []
    log_config = types.ModuleType("pypto.runtime.log_config")
    debug = types.ModuleType("pypto.runtime.debug")
    debug.__path__ = []
    replay = types.ModuleType("pypto.runtime.debug.replay")
    pto_rebuild = types.ModuleType("pypto.runtime.debug.pto_rebuild")
    backend = types.ModuleType("pypto.backend")

    for name in ("Tensor", "Scalar", "Array", "InOut", "Out"):
        setattr(language, name, _TypeSpec)
    for name in (
        "BF16",
        "FP32",
        "INT8",
        "INT32",
        "INT64",
        "TASK_ID",
        "UINT32",
    ):
        setattr(language, name, object())
    language.RUNTIME = object()
    language.dynamic = lambda name: name
    language.jit = _identity_jit

    # Tests that observe these patch them; the stub defaults are silent
    # no-ops so the runtime_dir replay path can flow through without
    # exploding when a test doesn't care.
    ir.compile = _unavailable
    runtime.execute_compiled = _unavailable
    runtime.RunConfig = RunConfig
    log_config.configure_log = lambda *_a, **_k: None
    replay.invalidate_binary_cache = lambda *_a, **_k: None
    pto_rebuild.rebuild_kernel_cpp_from_pto = lambda *_a, **_k: []
    backend.BackendType = BackendType

    pypto.ir = ir
    pypto.language = language
    pypto.runtime = runtime
    pypto.backend = backend
    runtime.log_config = log_config
    runtime.debug = debug
    debug.replay = replay
    debug.pto_rebuild = pto_rebuild

    sys.modules["pypto"] = pypto
    sys.modules["pypto.language"] = language
    sys.modules["pypto.ir"] = ir
    sys.modules["pypto.runtime"] = runtime
    sys.modules["pypto.runtime.log_config"] = log_config
    sys.modules["pypto.runtime.debug"] = debug
    sys.modules["pypto.runtime.debug.replay"] = replay
    sys.modules["pypto.runtime.debug.pto_rebuild"] = pto_rebuild
    sys.modules["pypto.backend"] = backend

    for name in [n for n in sys.modules if n == "pypto" or n.startswith("pypto.")]:
        sys.modules[name].__spec__ = importlib.machinery.ModuleSpec(name, loader=None)


_install_pypto_stubs()
