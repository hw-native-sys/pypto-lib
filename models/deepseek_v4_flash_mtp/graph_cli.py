# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select the explicit standalone HBG entry before loading model modules."""

import argparse
from pathlib import Path
import runpy
import sys


_PROVIDERS = {
    "decode_swa.py": "graph_decode_swa.py",
    "decode_hca.py": "graph_decode_hca.py",
    "decode_csa.py": "decode_csa.py",
    "decode_mtp.py": "decode_mtp.py",
}


def select_graph_cli(entry_file, argv):
    """Resolve explicit mode flags without importing PyPTO or a model."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--runtime", choices=("tmr", "hbg"), default=None)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--hbg-graph", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if args.hbg_graph:
        if args.runtime not in (None, "hbg"):
            parser.error("--hbg-graph conflicts with --runtime tmr")
        args.runtime = "hbg"
        args.graph = True
    if args.graph and args.runtime != "hbg":
        parser.error("--graph requires --runtime hbg; TMR Graph is unsupported")
    if args.runtime is None:
        return None
    entry = Path(entry_file).resolve()
    if entry.name not in _PROVIDERS:
        raise ValueError(f"Unsupported standalone Graph entry: {entry.name}")
    if args.runtime == "tmr":
        return entry, remaining, "tensormap_and_ringbuffer"
    target = entry.parent / "_hbg_prefill_compat" / _PROVIDERS[entry.name]
    if not target.is_file():
        raise FileNotFoundError(f"Missing explicit HBG provider: {target}")
    forwarded = [*remaining, "--runtime", "hbg"]
    if args.graph:
        forwarded.append("--graph")
    return target, forwarded, "host_build_graph"


def dispatch_graph_cli(entry_file):
    """Run the selected source-local CLI with an isolated import search order."""
    selected = select_graph_cli(entry_file, sys.argv[1:])
    if selected is None:
        return
    target, arguments, runtime_name = selected
    old_argv = sys.argv
    old_path = sys.path[:]
    try:
        sys.argv = [str(target), *arguments]
        sys.path.insert(0, str(target.parent.parent))
        sys.path.insert(0, str(target.parent))
        if runtime_name == "tensormap_and_ringbuffer":
            from pypto.ir import PassContext
            from pypto.pypto_core.passes import runtime_kind_from_name

            with PassContext([], runtime=runtime_kind_from_name(runtime_name)):
                runpy.run_path(str(target), run_name="__main__")
        else:
            runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
    raise SystemExit(0)
