# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check native vLLM prefill followed by Lib decode in eager or capture."""

import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

from torch_npu.npu import graphs
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from qwen_model import BATCH, LAYERS, SMOKE_CACHE_PAGES

PROMPT = [785, 6722, 315, 9625, 374]
EXPECTED = [12095, 13, 3555, 374, 279, 6722, 315, 279]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", choices=("tmr", "hbg"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "capture"), required=True)
    args = parser.parse_args()
    # Native prefill keeps the validated vendor stack; only decode is replaced.
    packages = {
        "vllm": "0.20.2",
        "vllm-ascend": "0.20.2rc1",
        "torch": "2.10.0",
        "torch-npu": "2.10.0.post2",
        "triton-ascend": "3.2.1",
    }
    for package, expected in packages.items():
        actual = version(package).split("+", 1)[0]
        if actual != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {actual}")
    capture = args.mode == "capture"
    metadata_path = Path((args.prepared / f"{args.runtime}-artifact.txt").read_text().strip())
    metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if capture and hasattr(graphs, "_current_native_graph"):
        raise RuntimeError("the capture smoke requires stock Torch-NPU")
    compilation = CompilationConfig(mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE)
    if capture:
        compilation = CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[BATCH],
            max_cudagraph_capture_size=BATCH,
        )
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        enforce_eager=not capture,
        max_model_len=4096,
        max_num_seqs=BATCH,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        block_size=128,
        num_gpu_blocks_override=SMOKE_CACHE_PAGES,
        worker_cls="qwen_kernel_worker.CaptureWorker" if capture else "qwen_kernel_worker.KernelWorker",
        compilation_config=compilation,
        additional_config={
            "kernel": {"runtime": args.runtime, "cache": str(args.prepared.resolve() / "jit-cache")},
            "ascend_compilation_config": {"enable_npugraph_ex": False},
        },
    )
    engine = llm.llm_engine.engine_core
    processes = tuple(engine.resources.engine_manager.processes)
    cleanup = None
    try:
        sampling = SamplingParams(temperature=0, max_tokens=len(EXPECTED), ignore_eos=True)
        llm.sleep(level=0, mode="keep")
        llm.enqueue([{"prompt_token_ids": PROMPT} for _ in range(BATCH)], sampling, use_tqdm=False)
        llm.wake_up(tags=["scheduling"])
        results = llm.wait_for_completion(use_tqdm=False)
        outputs = [list(result.outputs[0].token_ids) for result in results]
        (stats,) = llm.collective_rpc("kernel_stats")
        if stats["device_id"] != int(os.environ["TASK_DEVICE"]):
            raise RuntimeError(f"worker did not use the allocated physical device: {stats}")
        if outputs != [EXPECTED] * BATCH:
            raise RuntimeError(f"Qwen token mismatch: expected {EXPECTED}, got {outputs}")
        if stats["native_prefills"] != BATCH:
            raise RuntimeError(f"expected {BATCH} native prefilled requests, got {stats}")
        steps = len(EXPECTED) - 1
        if stats["failed"] or stats["steps"] != steps or stats["audited_steps"] != steps:
            raise RuntimeError(f"expected {steps} successful audited decode steps, got {stats}")
        # Each short request uses one KV page per layer for both K and V.
        if stats["audited_address_ranges"] != steps * BATCH * LAYERS * 2:
            raise RuntimeError(f"incomplete per-layer device address audit: {stats}")
        if capture and (stats["captures"], stats["replays"], stats["record_only_verified"]) != (
            1,
            steps,
            True,
        ):
            raise RuntimeError(
                f"expected one recording after eager warmup and {steps} actual replays, got {stats}"
            )
    finally:
        try:
            if not engine.resources.engine_dead:
                (cleanup,) = llm.collective_rpc("close_kernel")
        finally:
            engine.shutdown(timeout=60)
    if len(processes) != 1 or any(process.exitcode != 0 for process in processes):
        raise RuntimeError(f"unclean vLLM engine exit: {[process.exitcode for process in processes]}")
    if capture and cleanup != {"graph_reset": True, "pypto_reentered": True, "torch_usable": True}:
        raise RuntimeError(f"missing capture teardown receipt: {cleanup}")
    result = {
        "mode": args.mode,
        "runtime": args.runtime,
        "dispatch": "pypto.torch",
        "artifact_metadata_sha256": metadata_sha256,
        "requests": BATCH,
        "input_tokens_per_request": len(PROMPT),
        "output_tokens_per_request": len(EXPECTED),
        "tokens": outputs,
        **stats,
        "engine_exit_code": processes[0].exitcode,
        "cleanup": cleanup,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
