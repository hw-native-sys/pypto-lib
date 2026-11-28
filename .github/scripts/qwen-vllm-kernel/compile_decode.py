# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare Lib source JIT specializations for the A3 Qwen Kernel smoke."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pypto
import torch
from pypto._artifact_contract import ArtifactExecutionMode
from pypto.jit._artifact_manifest import ArtifactState
from pypto.runtime import CompileOptions
from pypto.runtime.kernel.session import KernelConfig
from decode_fwd import decode_fwd
from models.qwen3_14b.contract import build_decode_compile_args, get_qwen3_14b_contract
from qwen_model import BATCH, HEAD_DIM, HEADS, LAYERS, MAX_SEQ_LEN, PAGE, SMOKE_CACHE_PAGES, normalize_library_path

RUNTIMES = {"tmr": "tensormap_and_ringbuffer", "hbg": "host_build_graph"}


def decode_samples(model_config, pages):
    runtime = SimpleNamespace(max_batch_size=BATCH, max_seq_len=MAX_SEQ_LEN, page_size=PAGE)
    specs = get_qwen3_14b_contract().kernels["decode"].args
    with torch.device("meta"):
        samples = build_decode_compile_args(model_config, runtime)
        kv = torch.empty((LAYERS * pages * PAGE * HEADS, HEAD_DIM), dtype=torch.bfloat16)
    return tuple(
        kv if spec.name in ("k_cache", "v_cache") else sample
        for spec, sample in zip(specs, samples, strict=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    normalize_library_path()
    model = SimpleNamespace(**json.loads((args.model / "config.json").read_text()))
    samples = decode_samples(model, SMOKE_CACHE_PAGES)
    pypto.configure_cache(pypto.CacheConfig(enabled=True, root=args.output.resolve() / "jit-cache"))
    for name, runtime in RUNTIMES.items():
        # The stock JIT spelling for compile() selects Program artifacts.
        # Resolve the Kernel entry on the CPU for this isolated experiment.
        artifact, _, _ = decode_fwd._resolve_compiled(
            samples,
            {"config": CompileOptions()},
            _contract=ArtifactExecutionMode.KERNEL,
            _kernel_config=KernelConfig("a2a3", runtime, 0),
        )
        if artifact._artifact_runtime is None:
            raise RuntimeError(f"{runtime}: the stock PyPTO cache could not publish the specialization")
        artifact.load()
        handle = artifact._artifact_runtime.handle
        if handle.spec.state is not ArtifactState.BINARY_READY:
            raise RuntimeError(f"{runtime}: the original PyPTO cache did not publish device binaries")
        metadata = handle.directory / "compiled_meta.json"
        directions = {p["name"]: p["direction"] for p in json.loads(metadata.read_text())["params"]}
        if any(directions.get(k) != "InOut" for k in ("k_cache", "v_cache")):
            raise RuntimeError(f"{runtime}: generated K/V cache parameters must be InOut")
        (args.output / f"{name}-artifact.txt").write_text(str(metadata) + "\n")
        print(f"{runtime}: validated binary-ready Kernel cache")


if __name__ == "__main__":
    main()
