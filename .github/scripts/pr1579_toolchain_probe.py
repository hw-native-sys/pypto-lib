# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare the cache writers using frozen inputs and CPU-prebuilt binaries."""
import json
import os
from pathlib import Path
import shutil
import sys

import golden
from pypto.runtime.device_runner import _compile_and_assemble


def main():
    """Compile or replay both cache-writer cases for one toolchain."""
    label, mode = sys.argv[1:3]
    root = Path(os.environ["GITHUB_WORKSPACE"]) / "diagnostics"
    original = golden.run
    index = 0

    def wrapped(*args, **kwargs):
        nonlocal index
        case = index
        index += 1
        work = root / label / str(case)
        frozen = root / "frozen" / str(case)
        kwargs["compile_only"] = mode == "compile"
        kwargs["save_data"] = True
        if mode == "compile":
            kwargs["config"]["save_kernels_dir"] = str(work)
        else:
            kwargs["runtime_dir"] = str(work)
            if frozen.exists():
                kwargs["golden_data"] = str(frozen)
        result = original(*args, **kwargs)
        if mode == "compile":
            if not result.passed:
                raise RuntimeError(result.error)
            _compile_and_assemble(Path(result.work_dir), "a2a3")
            print("CPU_BUILD_READY", label, case, result.work_dir, flush=True)
        else:
            if not frozen.exists() and (work / "data" / "in").exists():
                frozen.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(work / "data", frozen)
            verdict = {"toolchain": label, "case": case, "passed": result.passed, "error": result.error}
            print("VERDICT", json.dumps(verdict), flush=True)
        return result

    golden.run = wrapped
    from models.glm5_3_flash.indexer import run_cache_case
    run_cache_case(["--case", "both", "-p", "a2a3", "-d", os.environ.get("TASK_DEVICE", "0")])


if __name__ == "__main__":
    main()
