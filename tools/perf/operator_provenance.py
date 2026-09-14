# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read toolchain pins and host-device identity without changing an environment."""

from __future__ import annotations

import base64
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
import sysconfig
from urllib.parse import unquote, urlparse

from tools.perf.operator_contract import ContractError, ROOT, read_json


def command(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.PIPE).strip()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def clean_head(root):
    if command(["git", "status", "--porcelain", "--untracked-files=no"], root):
        raise ContractError(f"source checkout is dirty: {root}")
    return command(["git", "rev-parse", "HEAD"], root)


def installed_package(name, module, source_root, *, isa_pin=None):
    """Bind the installed CI distribution to its selected source and RECORD."""
    roots = {Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib")}
    module_path = Path(module.__file__).resolve()
    if not any(module_path.is_relative_to(root) for root in roots):
        raise ContractError(f"{name} is imported outside the CI venv")
    dist = metadata.distribution(name)
    origin = json.loads(dist.read_text("direct_url.json") or "{}")
    url = urlparse(origin.get("url", ""))
    if url.scheme != "file" or url.netloc not in ("", "localhost"):
        raise ContractError(f"{name} has no local CI installation provenance")
    installed_from = Path(unquote(url.path)).resolve()
    tree = command(["git", "rev-parse", "HEAD^{tree}"], source_root)
    if "archive_info" in origin:
        expected = f"{name}-{tree}-" + (f"{isa_pin}-" if isa_pin else "")
        cache = Path(os.environ["CI_CACHE_ROOT"]).resolve() / "wheels"
        if (installed_from.parent.parent != cache or
                not installed_from.parent.name.startswith(expected)):
            raise ContractError(f"{name} wheel cache source/ISA differs from the selected checkout")
        checksum = origin["archive_info"].get("hashes", {}).get("sha256")
        if not checksum or file_hash(installed_from) != checksum:
            raise ContractError(f"{name} wheel hash is missing or changed")
        installation = {"kind": "wheel", "sha256": checksum, "cache_key": installed_from.parent.name}
    elif "dir_info" in origin and not origin["dir_info"].get("editable"):
        if installed_from != source_root.resolve():
            raise ContractError(f"{name} was installed from another source checkout")
        installation = {"kind": "source", "tree": tree}
    else:
        raise ContractError(f"{name} must use the CI wheel or source installer")
    files = dist.files or []
    native = {}
    owned_module = False
    isa_build = None
    for entry in files:
        path = Path(dist.locate_file(entry)).resolve()
        owned_module |= path == module_path
        if path.suffix != ".so" and path.name != "pto_isa_build.json" and path != module_path:
            continue
        if not any(path.is_relative_to(root) for root in roots):
            raise ContractError(f"{name} installation escapes the CI venv")
        checksum = file_hash(path)
        recorded = entry.hash
        encoded = base64.urlsafe_b64encode(bytes.fromhex(checksum)).decode().rstrip("=")
        if recorded is None or recorded.mode != "sha256" or recorded.value != encoded:
            raise ContractError(f"{name} installed file differs from wheel RECORD: {entry}")
        if path.suffix == ".so":
            native[str(entry)] = checksum
        if path.name == "pto_isa_build.json":
            isa_build = read_json(path)
    if not owned_module or not native:
        raise ContractError(f"{name} module or native libraries are missing from the distribution")
    if isa_pin:
        build = isa_build or {}
        required = [build, *(build.get("runtime_artifacts", {}).get(key, {}) for key in (
            "a2a3/onboard/host_build_graph", "a2a3/onboard/tensormap_and_ringbuffer",
        ))]
        if any(item.get("actual_checkout_commit") != isa_pin or
               item.get("required_commit_from_pin") != isa_pin for item in required):
            raise ContractError("installed runtime ISA metadata differs from the selected pin")
    return {"version": dist.version, "installation": installation, "native_sha256": native}


def capture_toolchain(driver_info=Path("/usr/local/Ascend/driver/version.info")):
    import pypto
    import torch
    import numpy
    import simpler

    pypto_root = Path(os.environ["PYPTO_SRC"]).resolve()
    runtime_root = pypto_root / "runtime"
    pypto_sha = clean_head(pypto_root)
    runtime_sha = clean_head(runtime_root)
    if command(["git", "rev-parse", "HEAD:runtime"], pypto_root) != runtime_sha:
        raise ContractError("runtime checkout differs from the PyPTO gitlink")
    pin = (runtime_root / "pto_isa.pin").read_text().strip()
    if os.environ.get("PTO_ISA_COMMIT") != pin:
        raise ContractError("CI's resolved ISA revision differs from the runtime pin")
    packages = {"pypto": installed_package("pypto", pypto, pypto_root),
                "simpler": installed_package("simpler", simpler, runtime_root, isa_pin=pin)}
    versions = (pypto_root / "toolchain/versions.env").read_text()
    match = re.search(r"^PTOAS_VERSION=['\"]?v?([\d.]+)", versions, re.M)
    # setup-ci-job installs PTOAS in its own venv, exposed through PTOAS_ROOT.
    ptoas = Path(os.environ["PTOAS_ROOT"]) / "bin/ptoas"
    if match is None or not ptoas.is_file():
        raise ContractError("PTOAS pin or CI executable is missing")
    version = command([str(ptoas), "--version"])
    if not re.search(r"(?<![\d.])" + re.escape(match[1]) + r"(?![\d.])", version):
        raise ContractError("PTOAS executable does not match the PyPTO pin")
    cann_root = Path(os.environ["ASCEND_HOME_PATH"])
    cann_info = sorted(cann_root.glob("*/ascend_toolkit_install.info"))
    if not cann_info or not driver_info.is_file():
        raise ContractError("CANN or driver version metadata is missing")
    return {"pypto": pypto_sha, "runtime": runtime_sha, "pto_isa": pin,
            "packages": packages, "ptoas_version": version, "ptoas_sha256": file_hash(ptoas),
            "bundle": Path(os.environ["PYPTO_TOOLCHAIN"]).name,
            "python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
            "cann_sha256": file_hash(cann_info[0]), "driver_sha256": file_hash(driver_info)}


def capture_host(profile, output_dir):
    inventory = command(["npu-smi", "info"])
    (Path(output_dir) / "npu-smi-info.txt").write_text(inventory + "\n")
    mapping = []
    for device in profile["sequences"]["8"]:
        node = Path(f"/dev/davinci{device}").stat()
        if not stat.S_ISCHR(node.st_mode):
            raise ContractError("allocated device is not a host character device")
        mapping.append({"host_device_id": device, "major": os.major(node.st_rdev),
                        "minor": os.minor(node.st_rdev)})
    return {"hostname": socket.gethostname(), "device_epoch": profile["device_epoch"],
            "device_id_domain": "host", "devices": mapping, "kernel": platform.release()}


def source_identity():
    return clean_head(ROOT)
