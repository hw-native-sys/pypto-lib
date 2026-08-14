# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a Python script after deterministically seeding its host fixture generators."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _seed_type(value: str) -> int:
    seed = int(value)
    if not 0 <= seed < 2**32:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**32)")
    return seed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=_seed_type)
    parser.add_argument("script", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != str(args.seed):
        parser.error(
            f"PYTHONHASHSEED must be {args.seed} before interpreter startup, got {hash_seed!r}"
        )
    script = args.script.resolve()
    if not script.is_file():
        parser.error(f"script does not exist: {script}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(
        f"[RUN] fixture seed={args.seed} python_hash_seed={hash_seed}",
        flush=True,
    )

    sys.argv = [str(script), *args.script_args]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
