# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate installed CI wheels using temporary distributions, without hardware."""

import base64
import csv
from importlib.metadata import PathDistribution
import json
import sys
from types import SimpleNamespace

import pytest

from tools.perf import operator_provenance as provenance
from tools.perf.operator_contract import ContractError


@pytest.fixture
def installation(tmp_path, monkeypatch):
    source = tmp_path / "source"
    runtime = source / "runtime"
    site = tmp_path / "venv/site-packages"
    cache = tmp_path / "cache"
    cann = tmp_path / "cann/arch"
    ptoas = tmp_path / "ptoas/bin/ptoas"
    for path in (runtime, source / "toolchain", site, cache, cann, ptoas.parent):
        path.mkdir(parents=True, exist_ok=True)
    for name, value in {"PYPTO_SRC": source, "CI_CACHE_ROOT": cache, "PTO_ISA_COMMIT": "isa",
                        "PTOAS_ROOT": ptoas.parent.parent, "ASCEND_HOME_PATH": cann.parent,
                        "PYPTO_TOOLCHAIN": tmp_path / "bundle/v1"}.items():
        monkeypatch.setenv(name, str(value))
    ptoas.write_bytes(b"assembler")
    (runtime / "pto_isa.pin").write_text("isa")
    (source / "toolchain/versions.env").write_text("PTOAS_VERSION=v0.61")
    (cann / "ascend_toolkit_install.info").write_text("CANN=test")
    driver = tmp_path / "driver.info"
    driver.write_text("driver=test")
    monkeypatch.setattr(provenance.sysconfig, "get_paths", lambda: {"purelib": str(site), "platlib": str(site)})
    monkeypatch.setattr(provenance, "clean_head", lambda root: "runtime" if root == runtime else "compiler")
    def command(argv, cwd=None):
        if argv[0] != "git":
            return "PTOAS 0.61"
        return "runtime" if argv[-1] == "HEAD:runtime" else "tree"
    monkeypatch.setattr(provenance, "command", command)
    distributions = {}
    for name, root in (("pypto", source), ("simpler", runtime)):
        module = site / name / "__init__.py"
        module.parent.mkdir()
        module.write_text("# installed module")
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(module)))
        library = module.parent / "libnative.so"
        library.write_bytes(b"native")
        files = [module, library]
        if name == "simpler":
            stamp = {"actual_checkout_commit": "isa", "required_commit_from_pin": "isa"}
            build = {**stamp, "runtime_artifacts": {key: stamp for key in (
                "a2a3/onboard/host_build_graph", "a2a3/onboard/tensormap_and_ringbuffer")}}
            metadata_file = module.parent / "pto_isa_build.json"
            metadata_file.write_text(json.dumps(build))
            files.append(metadata_file)
        distinfo = site / f"{name}-1.dist-info"
        distinfo.mkdir()
        (distinfo / "METADATA").write_text(f"Name: {name}\nVersion: 1\n")
        slot = cache / "wheels" / (f"{name}-tree-" + ("isa-" if name == "simpler" else "") + "abi-bundle")
        slot.mkdir(parents=True)
        wheel = slot / f"{name}.whl"
        wheel.write_bytes(b"wheel")
        (distinfo / "direct_url.json").write_text(json.dumps({"url": wheel.as_uri(),
            "archive_info": {"hashes": {"sha256": provenance.file_hash(wheel)}}}))
        with (distinfo / "RECORD").open("w") as stream:
            writer = csv.writer(stream)
            for file in files:
                checksum = base64.urlsafe_b64encode(bytes.fromhex(provenance.file_hash(file))).decode().rstrip("=")
                writer.writerow([str(file.relative_to(site)), "sha256=" + checksum, file.stat().st_size])
        distributions[name] = PathDistribution(distinfo)
    monkeypatch.setattr(provenance.metadata, "distribution", lambda name: distributions[name])
    for name in ("torch", "numpy"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__version__="test"))
    return SimpleNamespace(source=source, site=site, cache=cache, driver=driver, distributions=distributions)


def test_ci_wheels_without_source_build_directory(installation):
    assert not (installation.source / "runtime/build").exists()
    result = provenance.capture_toolchain(installation.driver)
    assert result["pypto"] == "compiler" and result["runtime"] == "runtime"
    assert result["packages"]["simpler"]["installation"]["kind"] == "wheel"
    assert result["packages"]["pypto"]["native_sha256"]


@pytest.mark.parametrize("defect", ["import", "isa", "native", "record", "wheel", "origin", "tree",
                                   "missing_stamp", "changed_stamp", "ptoas", "gitlink", "driver"])
def test_reject_mixed_or_changed_installation(installation, monkeypatch, defect):
    site = installation.site
    if defect == "import":
        monkeypatch.setitem(sys.modules, "pypto", SimpleNamespace(__file__="/outside/pypto.py"))
    elif defect == "isa":
        monkeypatch.setenv("PTO_ISA_COMMIT", "wrong")
    elif defect == "native":
        (site / "pypto/libnative.so").write_bytes(b"changed")
    elif defect == "record":
        (site / "pypto-1.dist-info/RECORD").write_text("")
    elif defect == "wheel":
        next(installation.cache.rglob("pypto.whl")).write_bytes(b"changed")
    elif defect in ("origin", "tree"):
        path = site / "pypto-1.dist-info/direct_url.json"
        value = json.loads(path.read_text())
        value["url"] = "https://example.invalid/pypto.whl" if defect == "origin" else value["url"].replace("tree", "wrong")
        path.write_text(json.dumps(value))
    elif defect == "missing_stamp":
        (site / "simpler/pto_isa_build.json").unlink()
    elif defect == "changed_stamp":
        (site / "simpler/pto_isa_build.json").write_text("{}")
    elif defect in ("ptoas", "gitlink"):
        original = provenance.command
        monkeypatch.setattr(provenance, "command", lambda argv, cwd=None:
            "wrong" if (argv[0] != "git" if defect == "ptoas" else argv[-1] == "HEAD:runtime")
            else original(argv, cwd))
    elif defect == "driver":
        installation.driver.unlink()
    with pytest.raises((ContractError, OSError)):
        provenance.capture_toolchain(installation.driver)


def test_ci_source_install_fallback(installation):
    path = installation.site / "pypto-1.dist-info/direct_url.json"
    path.write_text(json.dumps({"url": installation.source.as_uri(), "dir_info": {}}))
    result = provenance.capture_toolchain(installation.driver)
    assert result["packages"]["pypto"]["installation"] == {"kind": "source", "tree": "tree"}
