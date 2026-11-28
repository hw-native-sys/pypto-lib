# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the A3 prefill/decode ABI and validate artifact compatibility."""

import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import torch
from golden import ScalarSpec, TensorSpec, run
from pypto.ir import DistributedConfig

from . import config as C
from .weights import bf16_nz_shape, engram_wkv_nz_shape, projection_int8_nz_shape


TOPOLOGY = {"world": 16, "dp": 4, "tp": 4, "ep": 16, "local_experts": 24}
HISTORY_CAPACITY = 8200
PREFILL_MOE_SLAB_ROWS = (64, 128, 256, 512, 1024, 2048)


def default_prefill_slab_rows(capacity):
    """Select the MoE slab size for the Prefill artifact."""
    return 2048 if capacity == C.PREFILL_MAX_TOKENS else 64


def build_specs(function, *, mode, capacity, history_capacity=HISTORY_CAPACITY, prepare_windows=False, request_slots=1):
    """Construct compile shapes without allocating the model's weight tensors."""
    dtype_map = {"FP32": torch.float32, "BF16": torch.bfloat16, "INT8": torch.int8,
                 "UINT8": torch.uint8, "INT32": torch.int32, "INT64": torch.int64}
    if type(request_slots) is not int or not 1 <= request_slots <= C.MAX_BATCH_PER_DP:
        raise ValueError("Invalid per-DP request capacity")
    table_pages = (history_capacity + 127) // 128
    pages = table_pages * request_slots + 1
    c2a_pages = max(1, (history_capacity // 2 + 127) // 128)
    dimensions = {
        "FWD_ORI_BLOCKS_DYN": 40 * pages, "FWD_CMP_BLOCKS_DYN": 4 * pages,
        "FWD_INDEX_BLOCKS_DYN": 4 * pages, "FWD_STATE_BLOCKS_DYN": 3 * request_slots,
        "FWD_C2A_TABLE_DYN": c2a_pages, "FWD_C1A_TABLE_DYN": table_pages,
        "PREFILL_HISTORY_PAGES": pages, "PREFILL_C2A_TABLE": c2a_pages, "PREFILL_C1A_TABLE": table_pages,
        "ROPE_ROWS_DYN": history_capacity, "LOCAL_T_DYN": capacity, "T_DYN": capacity,
        "Q_START_DYN": request_slots + 1, "STATE_BLOCKS_DYN": request_slots, "B_DYN": request_slots,
        "INDEX_POOL_BLOCKS_DYN": min(C.FLASH.candidate_topk_blocks, ((table_pages + 1) * 16 + 31) // 32 * 32),
    }

    def extent(dim):
        if isinstance(dim, int):
            return dim
        label = repr(dim)
        for marker, value in sorted(dimensions.items(), key=lambda item: len(item[0]), reverse=True):
            if marker in label:
                return value
        raise ValueError(f"unknown {mode} dimension {dim!r}")

    specs = []
    for name, parameter in inspect.signature(function._func).parameters.items():
        if name == "execute_layers":
            if prepare_windows:
                specs.append(ScalarSpec(name, torch.int32, 1, compile_runtime=True))
            continue
        annotation = parameter.annotation
        dtype_name = str(annotation.dtype).split(".")[-1].upper().replace("BFLOAT16", "BF16").replace("FLOAT32", "FP32")
        if dtype_name not in dtype_map:
            raise ValueError(f"unsupported A3 dtype {annotation.dtype} for {name}")
        shape = [extent(dim) for dim in annotation.shape]
        if name == "engram_wkv":
            # The host orchestration slices rank-local matrices from a flat batch of NZ blocks.
            ranks, layers, _, _ = shape
            shape = list(engram_wkv_nz_shape(ranks * layers))
        elif name in ("wq_a", "wq_b", "wkv", "shared_w13", "shared_w2"):
            ranks, layers, rows, columns = shape
            shape = list(projection_int8_nz_shape(ranks * layers, rows, columns))
        elif name in ('wo_a', 'wo_b'):
            ranks, matrices, rows, columns = shape
            shape = list(bf16_nz_shape(ranks * matrices, rows, columns))
        specs.append(TensorSpec(name, shape, dtype_map[dtype_name]))
    return specs


def manifest(mode, capacity, history_capacity=HISTORY_CAPACITY, *, slab_rows=64, request_slots=1):
    """Version the shared weight, cache and quantization ABI."""
    root = Path(__file__).parent
    pending = ["prefill_fwd", "decode_fwd", "driver"]
    sources = {}
    while pending:
        name = pending.pop()
        path = root / f"{name}.py"
        if path.name in sources:
            continue
        source = path.read_bytes()
        sources[path.name] = hashlib.sha256(source).hexdigest()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module:
                    pending.append(node.module)
                else:
                    pending.extend(alias.name for alias in node.names)
    return {"format": "deepseek-v41-a3-artifact-v1", "platform": "a2a3", "mode": mode,
            "topology": TOPOLOGY, "moe_slab_rows": slab_rows, "capacity": capacity, "history_capacity": history_capacity,
            "request_slots_per_dp": request_slots, "weight_layout": C.WEIGHT_LAYOUT_VERSION, "cache_layout": C.CACHE_LAYOUT_VERSION,
            "quantization": "W8A8_DYNAMIC", "sources": sources}


def load_manifest(directory, *, mode, capacity=None, request_slots=1):
    """Reject artifacts from a different topology, dtype ABI or source snapshot."""
    path = Path(directory) / "a3_manifest.json"
    if not path.is_file():
        raise ValueError(f"Missing {path}. Compile this phase with {mode}_fwd --compile-only")
    actual = json.loads(path.read_text())
    slab_rows = actual.get("moe_slab_rows")
    if mode == "decode" and actual["capacity"] not in (16, C.DECODE_INPUT_MAX_TOKENS):
        raise ValueError(f"artifact {path} has incompatible Decode capacity")
    supported_slabs = PREFILL_MOE_SLAB_ROWS if mode == "prefill" else (min(64, actual["capacity"]),)
    if slab_rows not in supported_slabs:
        raise ValueError(f"artifact {path} has incompatible MoE slab geometry")
    expected = manifest(mode, actual["capacity"] if capacity is None else capacity, actual["history_capacity"],
                        slab_rows=slab_rows, request_slots=request_slots)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"artifact {path} has incompatible {key}")
    return actual


def compile_generation_graphs(prefill_dir, decode_dir, *, devices, context, prefill_capacity,
                              prefill_slab_rows, decode_capacity=16, request_slots=1):
    """Compile each phase in its own process with the matching runtime ABI."""
    for phase, directory in (("prefill", prefill_dir), ("decode", decode_dir)):
        cmd = [sys.executable, str(Path(__file__).with_name(f"{phase}_fwd.py")), "--compile-only", "--prepare-windows",
               "--output", str(directory), "--context-capacity", str(context), "--devices", devices,
               "--request-slots-per-dp", str(request_slots)]
        if phase == "prefill":
            cmd.extend(("--capacity", str(prefill_capacity), "--prefill-moe-slab-rows", str(prefill_slab_rows)))
        else:
            cmd.extend(("--decode-capacity", str(decode_capacity)))
        subprocess.run(cmd, check=True)


def _device_parser():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", "--devices", dest="devices", default=",".join(map(str, range(16))))
    return parser


def forward_validation_argv(argv, *, mode):
    """Use the numerical Prefill smoke case by default and bound explicit CI compilations."""
    _, remaining = _device_parser().parse_known_args(argv)
    if mode == "prefill" and remaining in ([], ["--sim"]):
        return [*argv, "--validate-replay"]
    if remaining:
        return argv
    return [*argv, "--compile-only", "--prepare-windows", "--context-capacity", "256",
            "--request-slots-per-dp", "2", "--decode-capacity", "16"]


def main(function, *, mode, capacity, slab_rows=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__, parents=[_device_parser()])
    parser.add_argument("--tp", type=int, choices=[4], default=4)
    parser.add_argument("--dp", type=int, choices=[4], default=4)
    parser.add_argument("--ep", type=int, choices=[16], default=16)
    parser.add_argument("--experts-per-rank", type=int, choices=[24], default=24)
    parser.add_argument("--decode-capacity", type=int, choices=[16, C.DECODE_INPUT_MAX_TOKENS],
                        help="Physical rows per DP group in the Decode graph")
    parser.add_argument("--prefill-moe-slab-rows", type=int, choices=PREFILL_MOE_SLAB_ROWS)
    parser.add_argument("--capacity", type=int, default=capacity)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--prepare-windows", action="store_true",
                        help="Compile the application entry with startup communication-window reservation")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--decode-runtime-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--chunk-size", type=int)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-ids-json", type=Path, help="Experiment request replay")
    inputs.add_argument("--tokens-file", type=Path, help="JSON array of token IDs")
    inputs.add_argument("--tokens", help="Comma-separated token IDs")
    inputs.add_argument("--prompt")
    parser.add_argument("--context-capacity", type=int)
    parser.add_argument("--output", type=Path, help="Fresh compile or generation directory")
    parser.add_argument("--result", type=Path, help="Generation output directory")
    parser.add_argument("--graph-dir", type=Path, help="Directory containing native prefill/ and decode/ artifacts")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--dump-logits", action="store_true")
    parser.add_argument("--full-graph", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--logits-mode", choices=["last"], default="last")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--teacher-forced", action="store_true")
    parser.add_argument("--audit-caches", action="store_true")
    parser.add_argument("--save-logits", type=Path)
    parser.add_argument("--validate-layers", default="")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validate-teacher-forced", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--measured-requests", type=int, default=1)
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--profile-last-decode", action="store_true")
    profiling.add_argument("--profile-all-steps", type=Path, metavar="DIR",
                           help="capture Level 1 swimlanes for prefill and all seven decode steps in a new directory")
    parser.add_argument("--execution-log", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--request-slots-per-dp", type=int, choices=range(1, C.MAX_BATCH_PER_DP + 1), default=1)
    args = parser.parse_args(argv)
    if args.request_slots_per_dp != 1 and not args.compile_only:
        parser.error("Use the packed session API to execute multi-slot artifacts")
    app_request = args.prompt is not None or args.tokens is not None or args.tokens_file is not None
    if args.max_new_tokens is None:
        args.max_new_tokens = 3 if app_request else 8
    chunk_default = 128 if app_request else C.PREFILL_MAX_TOKENS
    requested_chunk = args.chunk_size
    if requested_chunk is None:
        args.chunk_size = min(chunk_default, args.capacity or chunk_default)
    if args.profile_all_steps is not None and (args.checkpoint is None or args.validation_only or args.compile_only):
        parser.error("--profile-all-steps requires full checkpoint generation")
    if args.warmup_requests < 0 or args.measured_requests < 1:
        parser.error("warmup requests must be nonnegative; measured requests must be positive")
    if args.validation_only and not args.validate_layers:
        parser.error("--validation-only requires --validate-layers")
    if (args.teacher_forced or args.validate_teacher_forced or args.validate_layers) and not args.reference_dir:
        parser.error("numerical validation requires --reference-dir")
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) != 16 or len(set(devices)) != 16:
        parser.error("exactly 16 distinct allocated devices are required")
    if not 1 <= args.chunk_size <= C.PREFILL_MAX_TOKENS or not 1 <= args.max_new_tokens <= 256:
        parser.error("Require a chunk size in 1..8192 and a generation limit in 1..256")
    if args.context_capacity is not None and not 1 <= args.context_capacity <= 16384:
        parser.error("Context capacity must be in 1..16384")
    if args.compile_only:
        if function is None or args.capacity != capacity:
            parser.error("Compile with the phase's forward entry and its matching capacity")
        context = args.context_capacity or HISTORY_CAPACITY
        config = dict(backend_type="a2a3", execution_mode=False, dump_passes=args.dump_passes,
                      distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
                      ring_dep_pool=131072, ring_heap=(4 * 1024**3, 4 * 1024**3, 2 * 1024**3, 1024**3))
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=False)
            config.update(save_kernels=True, save_kernels_dir=str(args.output.resolve()))
        if args.prefill_moe_slab_rows is not None and (mode != "prefill" or args.prefill_moe_slab_rows != slab_rows):
            parser.error("MoE slab size must match the imported Prefill entry")
        snapshot = manifest(mode, capacity, context, slab_rows=slab_rows, request_slots=args.request_slots_per_dp)
        specs = build_specs(function, mode=mode, capacity=capacity, history_capacity=context,
                            prepare_windows=args.prepare_windows, request_slots=args.request_slots_per_dp)
        result = run(fn=function, specs=specs, config=config, compile_only=True)
        if not result.passed:
            raise SystemExit(result.error or 1)
        if manifest(mode, capacity, context, slab_rows=slab_rows, request_slots=args.request_slots_per_dp) != snapshot:
            raise RuntimeError("Model sources changed during compilation")
        path = Path(result.work_dir) / "a3_manifest.json"
        path.write_text(json.dumps(snapshot, indent=2) + "\n")
        print(f"A3_ARTIFACT {result.work_dir}", flush=True)
        return
    if mode != "prefill" or args.checkpoint is None:
        parser.error("Run generation through prefill_fwd or generate with --checkpoint")
    if args.graph_dir is not None:
        if args.runtime_dir or args.decode_runtime_dir:
            parser.error("Choose --graph-dir or the two per-phase directories")
        args.runtime_dir, args.decode_runtime_dir = args.graph_dir / "prefill", args.graph_dir / "decode"
    if (args.runtime_dir is None) != (args.decode_runtime_dir is None):
        parser.error("Supply both phase artifact directories")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.runtime_dir is None:
        if args.output is None:
            parser.error("Provide --output to build both graphs or supply compiled artifacts")
        args.runtime_dir, args.decode_runtime_dir = args.output / "prefill", args.output / "decode"
        context = args.context_capacity or (16384 if app_request else HISTORY_CAPACITY)
        prefill_capacity = args.capacity or ((args.chunk_size + 15) // 16 * 16)
        compile_generation_graphs(
            args.runtime_dir, args.decode_runtime_dir, devices=args.devices, context=context,
            prefill_capacity=prefill_capacity, decode_capacity=args.decode_capacity or 16,
            prefill_slab_rows=args.prefill_moe_slab_rows or default_prefill_slab_rows(prefill_capacity),
        )
    manifests = (load_manifest(args.runtime_dir, mode="prefill"),
                 load_manifest(args.decode_runtime_dir, mode="decode", capacity=args.decode_capacity))
    if args.capacity is not None and function is None and args.capacity != manifests[0]["capacity"]:
        parser.error("Requested capacity differs from the compiled Prefill artifact")
    slab_rows = manifests[0]["moe_slab_rows"]
    if args.prefill_moe_slab_rows is not None and args.prefill_moe_slab_rows != slab_rows:
        parser.error("Requested MoE slab size differs from the compiled Prefill artifact")
    args.capacity = manifests[0]["capacity"]
    args.decode_capacity = manifests[1]["capacity"]
    if requested_chunk is None:
        args.chunk_size = min(chunk_default, args.capacity)
    context = manifests[0]["history_capacity"]
    if context != manifests[1]["history_capacity"] or (args.context_capacity and context != args.context_capacity):
        parser.error("Both programs must share the requested history capacity")
    args.context_capacity = context
    if args.chunk_size > args.capacity:
        parser.error("Chunk size exceeds compiled Prefill capacity")
    from .generation import generate

    result_dir = args.result or (args.output / "generation" if args.output else None)
    if result_dir:
        result_dir.mkdir(parents=True, exist_ok=True)
        args.report = args.report or result_dir / "result.json"
        if args.dump_logits and args.save_logits is None:
            args.save_logits = result_dir / "logits.pt"
    report = generate(args, devices, manifests=manifests)
    report["devices"] = devices
    if args.execution_log:
        from .telemetry import rank_timings

        report["device_timings"] = rank_timings(args.execution_log, devices, dp_rank=report["dp_rank"])
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if result_dir and report["output_sequences"]:
        tokens = report["output_sequences"][0]["token_ids"]
        (result_dir / "tokens.json").write_text(json.dumps(tokens) + "\n")
        (result_dir / "answer.txt").write_text(report["text"] + "\n")
        if args.dump_logits:
            logits = torch.load(args.save_logits, map_location="cpu", weights_only=True)["logits"]
            for step, row in enumerate(logits):
                torch.save(row.clone(), result_dir / f"logits-{step}.pt")
    print(json.dumps(report, indent=2), flush=True)
