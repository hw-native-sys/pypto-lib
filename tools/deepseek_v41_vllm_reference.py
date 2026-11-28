# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture native vLLM-Ascend raw logits with fixed TP4/DP4/EP16 trace replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import torch

from tools.deepseek_v41_accuracy import (
    MAX_CONTEXT, MAX_PREFILL_CHUNK_SIZE, TOPOLOGY, allocated_devices, load_cases, new_capture, save_case, write_json,
)


def engine_kwargs(spec, *, chunk_size=16):
    return dict(
        model=spec["checkpoint"], tokenizer=spec["checkpoint"], tokenizer_mode="deepseek_v41",
        dtype="bfloat16", quantization="ascend", trust_remote_code=False, generation_config="vllm",
        tensor_parallel_size=4, data_parallel_size=4, data_parallel_size_local=4,
        enable_expert_parallel=True, distributed_executor_backend="external_launcher",
        max_model_len=MAX_CONTEXT, max_num_batched_tokens=chunk_size, max_num_seqs=1,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        gpu_memory_utilization=0.90, block_size=128, safetensors_load_strategy="lazy",
        engram_config=dict(cpu_offload=True, dp_shared_memory=True),
        enforce_eager=True, swa_bounded_replay=False, async_scheduling=False, seed=0,
        max_logprobs=-1, logprobs_mode="raw_logits", enable_trace_replay=True,
    )


def sampling_params(case):
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, logprobs=-1, detokenize=False, max_tokens=len(case["trace_token_ids"]),
                          trace_decode_token_ids=list(case["trace_token_ids"]))


def extract_logits(request, case, vocabulary):
    if list(request.prompt_token_ids) != case["prompt_token_ids"] or len(request.outputs) != 1:
        raise ValueError(f"{case['id']}: vLLM returned a different prompt or multiple completions")
    completion = request.outputs[0]
    if list(completion.token_ids) != case["trace_token_ids"]:
        raise ValueError(f"{case['id']}: native vLLM trace replay returned a different Decode history")
    rows = completion.logprobs
    if rows is None or len(rows) != len(case["trace_token_ids"]):
        raise ValueError(f"{case['id']}: missing Prefill/Decode score distributions")
    logits = torch.empty((len(rows), vocabulary), dtype=torch.float32)
    expected_ids = set(range(vocabulary))
    for step, scores in enumerate(rows):
        if scores is None or scores.keys() != expected_ids:
            raise ValueError(f"{case['id']}: require the full vocabulary; set max_logprobs=-1 and logprobs=-1")
        # vLLM retains the Logprob container name in raw_logits mode.
        logits[step] = torch.tensor([scores[token].logprob for token in range(vocabulary)], dtype=torch.float32)
    if not torch.isfinite(logits).all():
        raise ValueError(f"{case['id']}: vLLM returned nonfinite logits")
    return logits


def verify_engine(config, spec):
    model, parallel, engram = config.model_config, config.parallel_config, config.engram_config
    if (parallel.tensor_parallel_size != 4 or parallel.data_parallel_size != 4
            or parallel.data_parallel_size_local != 4 or not parallel.enable_expert_parallel
            or parallel.pipeline_parallel_size != 1 or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.distributed_executor_backend != "external_launcher"):
        raise ValueError("Native reference requires single-node TP4/DP4/EP16, PP1/PCP1/DCP1")
    if engram is None or not engram.cpu_offload or not engram.dp_shared_memory:
        raise ValueError("Native reference requires Engram CPU offload with DP shared memory")
    if model.max_logprobs != -1 or model.logprobs_mode != "raw_logits" or not model.enable_trace_replay:
        raise ValueError("Native reference requires unlimited raw logits and native trace replay")
    if model.get_vocab_size() != spec["vocab_size"] or model.quantization != "ascend":
        raise ValueError("Native reference checkpoint vocabulary or quantization differs")
    if config.speculative_config is not None:
        raise ValueError("Trace replay does not support speculative decoding")
    for case in spec["cases"]:
        params = sampling_params(case)
        params._validate_logprobs(model)
        params._validate_trace_replay(model, config.speculative_config)


def check_config(spec, *, chunk_size=16):
    from vllm.engine.arg_utils import EngineArgs

    # External-launcher configuration reads RANK without creating any workers.
    previous_rank = os.environ.get("RANK")
    os.environ.setdefault("RANK", "0")
    try:
        config = EngineArgs(**engine_kwargs(spec, chunk_size=chunk_size)).create_engine_config()
        verify_engine(config, spec)
        initialized = hasattr(torch, "npu") and torch.npu.is_initialized()
        if initialized:
            raise RuntimeError("Configuration check unexpectedly initialized the NPU")
        return dict(status="config_checked", topology=TOPOLOGY, weights_loaded=False,
                    npu_initialized=False, native_trace_replay=True, max_logprobs=config.model_config.max_logprobs,
                    logprobs_mode=config.model_config.logprobs_mode,
                    engram_cpu_offload=config.engram_config.cpu_offload,
                    engram_dp_shared_memory=config.engram_config.dp_shared_memory,
                    chunk_size=chunk_size, cases=len(spec["cases"]))
    finally:
        if previous_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = previous_rank


def capture_reference(spec, cases_path, output, *, chunk_size=16):
    devices = allocated_devices()
    rank = int(os.environ.get("RANK", "-1"))
    if (os.environ.get("WORLD_SIZE") != "16" or os.environ.get("LOCAL_WORLD_SIZE") != "16"
            or not 0 <= rank < 16 or int(os.environ.get("LOCAL_RANK", "-1")) != rank):
        raise ValueError("Use torchrun --standalone --nnodes=1 --nproc-per-node=16 on the allocated devices")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))

    from vllm import LLM
    from vllm.distributed.parallel_state import get_world_group

    output, llm, report = Path(output), None, None
    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            report = new_capture(spec, cases_path, "vllm-ascend")
            report.update(devices=devices, chunk_size=chunk_size, engine=engine_kwargs(spec, chunk_size=chunk_size),
                          native_trace_replay=True)
            write_json(output / "manifest.json", report)
        llm = LLM(**engine_kwargs(spec, chunk_size=chunk_size))
        config = llm.llm_engine.vllm_config
        verify_engine(config, spec)
        dp_rank = config.parallel_config.data_parallel_rank
        if dp_rank != rank // TOPOLOGY["tp"]:
            raise RuntimeError("Native vLLM rank mapping differs from the PyPTO DP/TP mapping")
        for case in spec["cases"]:
            # Empty DP engines participate in native dummy batches for EP collectives.
            prompts = [dict(prompt_token_ids=case["prompt_token_ids"])] if dp_rank == 0 else []
            requests = llm.generate(prompts, sampling_params(case), use_tqdm=False)
            if rank == 0:
                if len(requests) != 1:
                    raise RuntimeError("Expected exactly one result from DP rank 0")
                logits = extract_logits(requests[0], case, spec["vocab_size"])
                report["cases"].append(save_case(output, case, logits, spec["vocab_size"]))
                write_json(output / "manifest.json", report)
            torch.distributed.barrier(group=get_world_group().cpu_group)
        if rank == 0:
            report["status"] = "complete"
    except BaseException as error:
        if report is not None:
            report.update(status="failed", error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
        raise
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        except BaseException as error:
            if report is not None:
                report.update(status="failed", error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc())
            raise
        finally:
            if report is not None:
                write_json(output / "manifest.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel", type=int, choices=(4,), default=4)
    parser.add_argument("--data-parallel", type=int, choices=(4,), default=4)
    parser.add_argument("--expert-parallel", type=int, choices=(16,), default=16)
    parser.add_argument("--chunk-size", type=int, choices=range(1, MAX_PREFILL_CHUNK_SIZE + 1), default=16)
    parser.add_argument("--check-config", action="store_true",
                        help="Inspect native config without loading weights or NPU")
    args = parser.parse_args(argv)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    try:
        spec = load_cases(args.cases)
        if args.check_config:
            print(json.dumps(check_config(spec, chunk_size=args.chunk_size)))
        else:
            result = capture_reference(spec, args.cases, args.output, chunk_size=args.chunk_size)
            if result is not None:
                print(json.dumps(dict(status=result["status"], accuracy="not_compared", output=str(args.output))))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
