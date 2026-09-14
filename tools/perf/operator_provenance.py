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

import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import stat
import subprocess

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
    if command(["git", "status", "--porcelain", "--untracked-files=normal"], root):
        raise ContractError(f"source checkout is dirty: {root}")
    return command(["git", "rev-parse", "HEAD"], root)


def capture_toolchain(driver_info=Path("/usr/local/Ascend/driver/version.info")):
    import pypto
    import torch
    import numpy
    import simpler

    pypto_root = Path(os.environ["PYPTO_ROOT"]).resolve()
    isa_root = Path(os.environ["PTO_ISA_ROOT"]).resolve()
    runtime_root = pypto_root / "runtime"
    if not Path(pypto.__file__).resolve().is_relative_to(pypto_root):
        raise ContractError("imported pypto does not match PYPTO_ROOT")
    if not Path(simpler.__file__).resolve().is_relative_to(runtime_root):
        raise ContractError("imported simpler does not match the PyPTO runtime submodule")
    pypto_sha = clean_head(pypto_root)
    runtime_sha = clean_head(runtime_root)
    isa_sha = clean_head(isa_root)
    if command(["git", "rev-parse", "HEAD:runtime"], pypto_root) != runtime_sha:
        raise ContractError("runtime checkout differs from the PyPTO gitlink")
    pin = (runtime_root / "pto_isa.pin").read_text().strip()
    if isa_sha != pin:
        raise ContractError("PTO ISA checkout differs from the runtime pin")
    build = read_json(runtime_root / "build/lib/pto_isa_build.json")
    required = [build, *(build.get("runtime_artifacts", {}).get(key, {}) for key in (
        "a2a3/onboard/host_build_graph", "a2a3/onboard/tensormap_and_ringbuffer",
    ))]
    if any(item.get("actual_checkout_commit") != pin or item.get("required_commit_from_pin") != pin
           for item in required):
        raise ContractError("runtime build contains a different PTO ISA revision")
    versions = (pypto_root / "toolchain/versions.env").read_text()
    match = re.search(r"^PTOAS_VERSION=['\"]?v?([\d.]+)", versions, re.M)
    ptoas = shutil.which("ptoas")
    if match is None or ptoas is None:
        raise ContractError("PTOAS pin or executable is missing")
    version = command([ptoas, "--version"])
    if not re.search(r"(?<![\d.])" + re.escape(match[1]) + r"(?![\d.])", version):
        raise ContractError("PTOAS executable does not match the PyPTO pin")
    native = {str(path.relative_to(runtime_root / "build/lib")): file_hash(path)
              for path in sorted((runtime_root / "build/lib").glob("*.so"))}
    if not native:
        raise ContractError("native runtime libraries are missing")
    cann_root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    cann_info = sorted(cann_root.glob("*/ascend_toolkit_install.info"))
    if not cann_info or not driver_info.is_file():
        raise ContractError("CANN or driver version metadata is missing")
    return {"pypto": pypto_sha, "runtime": runtime_sha, "pto_isa": isa_sha,
            "ptoas_version": version, "ptoas_sha256": file_hash(ptoas),
            "native_runtime_sha256": native, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": numpy.__version__,
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
