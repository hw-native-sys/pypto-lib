# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the migrated sources and prefill controls without allocating an NPU."""

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path

from .smla_scheduler import canonical_selection, from_controls
from .swa_scheduler import build


def check_sources():
    """Check source receipts, local import closure, and dependency direction."""
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "source_manifest.json").read_text())
    for row in manifest["files"]:
        digest = hashlib.sha256((directory / row["file"]).read_bytes()).hexdigest()
        if digest != row["migrated_sha256"]:
            raise RuntimeError("Migrated source identity changed: " + row["file"])
    modules = {path.stem: ast.parse(path.read_text()) for path in directory.glob("*.py")}
    graph = {name: set() for name in modules}
    for name, tree in modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if node.level != 1:
                raise RuntimeError("Model imports must stay in the flat directory: " + name)
            targets = [node.module.split(".")[0]] if node.module else [alias.name for alias in node.names]
            for target in targets:
                if target not in modules:
                    raise RuntimeError("Missing local dependency: " + target)
                graph[name].add(target)

    visited, active = set(), set()

    def visit(name):
        if name in active:
            raise RuntimeError("Cyclic model dependency: " + name)
        if name in visited:
            return
        active.add(name)
        for child in graph[name]:
            visit(child)
        active.remove(name)
        visited.add(name)

    for name in modules:
        visit(name)
    return len(modules)


def check_controls():
    """Exercise page crossing, separate requests, causal selection, and rejection."""
    plan = build([3, 1], [127, 0], [[2, 0], [1, -1]], cache_blocks=3, capacity=16)
    if plan["positions"] != [127, 128, 129, 0] or plan["flat_slot_mapping"] != [383, 0, 1, 128]:
        raise RuntimeError("Page crossing or request publication differs")
    if plan["physical_window_indices"][1] != list(range(257, 384)) + [0]:
        raise RuntimeError("Causal window lost the preceding physical page")
    raw = [[128, 3, -1] + [-1] * 509, [0] + [-1] * 511]
    selected = canonical_selection(raw, [127, 0])
    if selected != [[3] + [-1] * 511, [0] + [-1] * 511]:
        raise RuntimeError("Causal QLI conversion differs")
    ratio1 = from_controls([0, 1, 2], [128, 1], [[2], [0]], [[1], [2]], selected,
                           ori_blocks=3, cmp_blocks=3, capacity=16)
    if ratio1["physical_compressed_indices"][0][0] != 131 or ratio1["cmp_slots"] != [[1, 127], [2, 0]]:
        raise RuntimeError("Ratio1 lanes did not retain independent page tables")
    invalid = (
        lambda: build([1, 1], [0, 0], [[0], [0]], cache_blocks=1, capacity=16),
        lambda: build([2], [127], [[0]], cache_blocks=1, capacity=16),
        lambda: build([2], [0], [[0]], cache_blocks=1, capacity=1),
        lambda: canonical_selection([[0, 0] + [-1] * 510], [1]),
    )
    for case in invalid:
        try:
            case()
        except RuntimeError:
            continue
        raise RuntimeError("Invalid prefill controls were accepted")


def check_factories():
    """Import every module and construct the principal bounded DSL entries."""
    directory = Path(__file__).resolve().parent
    package = __package__
    for path in sorted(directory.glob("*.py")):
        importlib.import_module(package + "." + path.stem)
    from .ecotech_full2_prefill import make_entries
    from .ecotech_swa_prefill import GEOMETRY, make_builders
    from .engram_ecotech_observed import make_engram_ecotech_observed
    from .layers19_25 import make_decoder_layer_stages
    from .ratio1_prefill import make_ratio1_compress_publish
    from .smla_staging import make_import, make_pack

    make_builders()
    for active in (1, 7, 16):
        make_entries(active)
        make_entries(active, compressor_projected=True)
        make_pack(active, 2, 2)
        make_import(active, 2, 2)
    make_ratio1_compress_publish(16, dict(GEOMETRY, index_head_dim=128))
    make_engram_ecotech_observed(16, 4096)
    for layer_id in (19, 20, 21, 24, 25):
        make_decoder_layer_stages(layer_id, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factories", action="store_true", help="Also import Torch/PyPTO and construct DSL entries")
    args = parser.parse_args()
    count = check_sources()
    check_controls()
    if args.factories:
        check_factories()
    print(f"PASS: {count} modules, source receipts, acyclic imports, and prefill controls")
    if args.factories:
        print("PASS: module imports and SWA/Full2/Ratio1/Engram/decoder factories; no NPU execution")


if __name__ == "__main__":
    main()
