# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare both native Kernel callable caches before device allocation."""

import argparse
import json
import runpy
from pathlib import Path

from simpler.task_interface import ArgDirection
from simpler_setup.compile_pool import compile_worker_budget
from simpler_setup.scene_test import compile_chip_callable_spec, l3_compile_cache_key


def prepare_artifact(artifact):
    artifact = Path(artifact).resolve()
    params = json.loads((artifact / "compiled_meta.json").read_text())["params"]
    config = runpy.run_path(str(artifact / "kernel_config.py"))
    runtime = config["RUNTIME_CONFIG"]["runtime"]
    orchestration = config["ORCHESTRATION"]
    directions = {"In": ArgDirection.IN, "InOut": ArgDirection.INOUT, "Out": ArgDirection.OUT}
    if orchestration.get("host_tensor_indices") or orchestration["signature"] != [
        directions[param["direction"]] for param in params
    ]:
        raise ValueError("expected the generated device-only decode signature")
    spec = {"orchestration": orchestration, "incores": config["KERNELS"], "runtime_config": config["RUNTIME_CONFIG"]}
    key = l3_compile_cache_key("qwen_kernel_worker", str(artifact), "decode_fwd", "a2a3", runtime)
    callable_ = compile_chip_callable_spec(spec, "a2a3", runtime, key)
    return runtime, callable_, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with compile_worker_budget(args.workers):
        for runtime in ("tmr", "hbg"):
            prepare_artifact(args.output / runtime)


if __name__ == "__main__":
    main()
