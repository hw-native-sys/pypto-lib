# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise environment rejection using tiny CPU-only fake installations."""

import json
import sys
from types import SimpleNamespace

import pytest

from tools.perf import operator_provenance as provenance
from tools.perf.operator_contract import ContractError


@pytest.mark.parametrize("defect", [None, "import", "gitlink", "isa", "build", "ptoas", "native"])
def test_toolchain_pin_and_binary_identity(tmp_path, monkeypatch, defect):
    pypto = tmp_path / "pypto"
    runtime = pypto / "runtime"
    isa = tmp_path / "isa"
    cann = tmp_path / "cann"
    for path in (runtime / "build/lib", pypto / "toolchain", isa, cann / "arch"):
        path.mkdir(parents=True)
    monkeypatch.setenv("PYPTO_ROOT", str(pypto))
    monkeypatch.setenv("PTO_ISA_ROOT", str(isa))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann))
    for name, path in (("pypto", pypto / "python/pypto/__init__.py"),
                       ("simpler", runtime / "python/simpler/__init__.py")):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(path)))
    if defect == "import":
        monkeypatch.setitem(sys.modules, "pypto", SimpleNamespace(__file__="/outside/pypto.py"))
    for name in ("torch", "numpy"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__="test"))
    heads = {pypto: "compiler", runtime: "runtime", isa: "wrong" if defect == "isa" else "isa"}
    monkeypatch.setattr(provenance, "clean_head", lambda root: heads[root])
    assembler = tmp_path / "ptoas"
    assembler.write_bytes(b"assembler")
    monkeypatch.setattr(provenance.shutil, "which", lambda _: str(assembler))
    def command(argv, cwd=None):
        if argv[0] == "git":
            return "wrong" if defect == "gitlink" else "runtime"
        return "PTOAS 0.60" if defect == "ptoas" else "PTOAS 0.61"
    monkeypatch.setattr(provenance, "command", command)
    (runtime / "pto_isa.pin").write_text("isa\n")
    (pypto / "toolchain/versions.env").write_text("PTOAS_VERSION=v0.61\n")
    stamp = {"actual_checkout_commit": "wrong" if defect == "build" else "isa",
             "required_commit_from_pin": "isa"}
    build = {**stamp, "runtime_artifacts": {
        "a2a3/onboard/host_build_graph": stamp, "a2a3/onboard/tensormap_and_ringbuffer": stamp}}
    (runtime / "build/lib/pto_isa_build.json").write_text(json.dumps(build))
    if defect != "native":
        (runtime / "build/lib/libruntime.so").write_bytes(b"runtime")
    (cann / "arch/ascend_toolkit_install.info").write_text("version=test")
    driver = tmp_path / "driver-version.info"
    driver.write_text("version=test")
    if defect:
        with pytest.raises(ContractError):
            provenance.capture_toolchain(driver)
    else:
        result = provenance.capture_toolchain(driver)
        assert result["pypto"] == "compiler" and result["runtime"] == "runtime"
        assert result["native_runtime_sha256"]["libruntime.so"] == provenance.file_hash(runtime / "build/lib/libruntime.so")
        assert result["ptoas_sha256"] == provenance.file_hash(assembler)
