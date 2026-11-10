# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build both Qwen decode artifacts from the Lib checkout under test."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from pypto._artifact_contract import ArtifactExecutionMode
from pypto.ir.compile import _compile_artifact
from pypto.pypto_core.passes import RuntimeKind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root), str(root / "models/qwen3_14b")]
    from decode_fwd import decode_fwd
    from models.qwen3_14b.contract import build_decode_compile_args

    model = SimpleNamespace(**json.loads((args.model / "config.json").read_text()))
    runtime = SimpleNamespace(max_batch_size=16, max_seq_len=4096, page_size=128)
    with torch.device("meta"):
        samples = build_decode_compile_args(model, runtime)
    for name, kind in (
        ("tmr", RuntimeKind.TENSORMAP_AND_RINGBUFFER),
        ("hbg", RuntimeKind.HOST_BUILD_GRAPH),
    ):
        artifact_dir = args.output / name
        _compile_artifact(
            decode_fwd.specialize(*samples),
            output_dir=str(artifact_dir),
            platform="a2a3",
            runtime=kind,
            contract=ArtifactExecutionMode.KERNEL,
        )
        meta = json.loads((artifact_dir / "compiled_meta.json").read_text())
        directions = {param["name"]: param["direction"] for param in meta["params"]}
        if any(directions.get(name) != "InOut" for name in ("k_cache", "v_cache")):
            raise RuntimeError(f"{kind}: generated K/V cache parameters must be InOut")


if __name__ == "__main__":
    main()
