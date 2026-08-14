# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a Python entry point after applying the official performance seed."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("entrypoint", type=Path)
    parser.add_argument("entrypoint_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != str(args.seed):
        raise SystemExit(
            f"PYTHONHASHSEED must be {args.seed} before Python starts, got {hash_seed!r}"
        )

    entrypoint = args.entrypoint.resolve()
    if not entrypoint.is_file():
        raise SystemExit(f"entry point does not exist: {entrypoint}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(
        f"[PERF] deterministic_seed seed={args.seed} python_hash_seed={hash_seed}",
        flush=True,
    )
    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *args.entrypoint_args]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
