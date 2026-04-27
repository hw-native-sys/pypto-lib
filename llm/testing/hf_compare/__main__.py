# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CLI for the HF comparison framework.

Usage:
    python -m llm.testing.hf_compare list
    python -m llm.testing.hf_compare run qwen3_14b.decode \\
        --hf-model-path /path/to/Qwen3-14B --platform a2a3 [--cpu-only]
"""
from __future__ import annotations

import argparse
import json
import sys

from .base import get_case, list_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hf_compare")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List registered cases.")

    run = sub.add_parser("run", help="Run one case.")
    run.add_argument("case", help="Registered case name (see `list`).")
    run.add_argument("--json", action="store_true", help="Print JSON report.")
    run.add_argument(
        "--kwarg", "-k", action="append", default=[],
        help="Extra kwarg for the case factory, e.g. -k hf_model_path=/data/...",
    )

    args = parser.parse_args(argv)
    # Autodiscovery is lazy: list_cases() / get_case() trigger it internally.

    if args.cmd == "list":
        for name in list_cases():
            print(name)
        return 0

    if args.cmd == "run":
        kwargs: dict[str, str] = {}
        for kv in args.kwarg:
            if "=" not in kv:
                parser.error(f"--kwarg must be key=value, got {kv!r}")
            k, v = kv.split("=", 1)
            kwargs[k] = v
        case = get_case(args.case, **kwargs)
        report = case.run()
        if args.json:
            print(json.dumps(report.to_json(), indent=2))
        else:
            print(report.summary())
        return 0 if report.passed else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
