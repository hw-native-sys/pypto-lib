# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate checkpoint-sized batch-one programs without allocating a device.

Run from the repository root as a module. Native compilation and request
execution are subsequent stages, not side effects of this command.
"""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace

from . import native_source


@contextmanager
def ptoas_deadline(seconds):
    """Bound this build's PTOAS calls without changing the installed backend.

    The backend currently fixes each assembler subprocess to 60 seconds.
    Large compensated vector functions need a longer resource budget; keep
    its command construction, version checks and failure handling intact.
    Build profiles run in separate processes, as required by their frozen
    import-time shapes.
    """
    if type(seconds) is not int or not 1 <= seconds <= 600:
        raise ValueError("PTOAS timeout must be an integer in [1, 600] seconds")
    from pypto.backend import pto_backend

    original = pto_backend._run_ptoas
    original_subprocess = original.__globals__["subprocess"]

    def run(*args, **kwargs):
        if kwargs.get("timeout") != 60:
            raise RuntimeError("The backend PTOAS subprocess contract changed")
        kwargs["timeout"] = seconds
        return original_subprocess.run(*args, **kwargs)

    globals_ = dict(original.__globals__)
    globals_["subprocess"] = SimpleNamespace(run=run, TimeoutExpired=original_subprocess.TimeoutExpired)
    bounded = FunctionType(
        original.__code__, globals_, original.__name__, original.__defaults__, original.__closure__
    )
    bounded.__kwdefaults__ = original.__kwdefaults__
    bounded.__annotations__ = original.__annotations__
    pto_backend._run_ptoas = bounded
    try:
        yield
    finally:
        pto_backend._run_ptoas = original


def artifact_directory(compiled):
    """Return the directory owned by the public compiled-program result."""
    return Path(compiled.output_dir).resolve()


def source_manifest():
    root = Path(__file__).resolve().parent
    sources = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.suffix in (".py", ".cpp", ".inc")
    }
    return sources


def _validate_target_capacity(metadata, profile, max_seq_len):
    """Reject generated verification ABIs that truncate decode cache metadata."""
    from .native_decode import (
        CSA_CMP_MAX_BLOCKS, CSA_IDX_CACHE_MAX_BLOCKS, CSA_INNER_STATE_MAX_BLOCKS,
        CSA_MAIN_STATE_MAX_BLOCKS, HCA_CMP_MAX_BLOCKS, HCA_COMPRESS_STATE_MAX_BLOCKS,
        MAX_SEQ_LEN, ORI_TABLE_MAX_BLOCKS, ROPE_HEAD_DIM,
    )

    if MAX_SEQ_LEN != max_seq_len:
        raise RuntimeError("Decode shapes are already frozen at another maximum sequence length")
    prefix = "" if profile == "single" else "s_"
    expected = {
        "freqs_cos": [8, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM],
        "freqs_sin": [8, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM],
        "block_table": [8, 1, ORI_TABLE_MAX_BLOCKS],
        "hca_cmp_block_table": [8, 1, HCA_CMP_MAX_BLOCKS],
        "csa_cmp_block_table": [8, 1, CSA_CMP_MAX_BLOCKS],
        "idx_block_table": [8, 1, CSA_IDX_CACHE_MAX_BLOCKS],
        "hca_compress_state_block_table": [8, 1, HCA_COMPRESS_STATE_MAX_BLOCKS],
        "csa_compress_state_block_table": [8, 1, CSA_MAIN_STATE_MAX_BLOCKS],
        "csa_inner_compress_state_block_table": [8, 1, CSA_INNER_STATE_MAX_BLOCKS],
    }
    actual = {row["name"].split("__ssa", 1)[0]: row.get("shape") for row in metadata["params"]}
    for name, shape in expected.items():
        if actual.get(prefix + name) != shape:
            raise RuntimeError(f"Generated target capacity differs for {prefix + name}: {actual.get(prefix + name)} != {shape}")


def compile_program(profile, *, cache_root, device_ids, ptoas_timeout_seconds=180, max_seq_len=1048576,
                    prefill_indexer_arenas=1, projection_degrees=None, prefill_moe_ready=None,
                    prefill_moe_wave_tokens=None, prefill_moe_owner_reduce=False, prefill_moe_expert_affinity=False,

                    prefill_causal_ready=False, prefill_causal_full_wait=False,
                    prefill_echo_prefetch=False, prefill_echo_diagnostics=False, prefill_mechanisms=None):
    """Compile one immutable source profile; device IDs describe topology only."""
    if profile not in ("single", "one", "two", "rsag", "prefill"):
        raise ValueError("Unknown request-pipeline source profile")
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise ValueError("Maximum sequence length must be a positive integer")
    if type(prefill_indexer_arenas) is not int or prefill_indexer_arenas not in (1, 2):
        raise ValueError("Prefill indexer arenas must be one or two")
    if profile != "prefill" and prefill_indexer_arenas != 1:
        raise ValueError("Indexer arena selection applies only to prefill")
    if type(prefill_moe_expert_affinity) is not bool:
        raise ValueError("Prefill expert affinity must be a boolean compile option")
    if profile != "prefill" and prefill_moe_expert_affinity:
        raise ValueError("Expert affinity applies only to prefill")
    if type(prefill_moe_owner_reduce) is not bool:
        raise ValueError("Prefill owner reduction must be a boolean compile option")
    if profile != "prefill" and prefill_moe_owner_reduce:
        raise ValueError("Owner reduction applies only to prefill")
    if prefill_moe_ready is None:
        prefill_moe_ready = profile == "prefill"
    if type(prefill_moe_ready) is not bool:
        raise ValueError("Prefill MoE readiness must be a boolean compile option")
    if profile != "prefill" and prefill_moe_ready:
        raise ValueError("MoE readiness selection applies only to prefill")
    if prefill_moe_wave_tokens is not None:
        if type(prefill_moe_wave_tokens) is not int or prefill_moe_wave_tokens not in (8, 32, 64, 128, 1024, 2048, 8192):
            raise ValueError("Prefill MoE wave tokens must be 8, 32, 64, 128, 1024, 2048 or 8192")
        if profile != "prefill":
            raise ValueError("MoE wave geometry selection applies only to prefill")
    if profile == "prefill" and prefill_moe_wave_tokens is None:
        prefill_moe_wave_tokens = 2048
    from .prefill_fwd import resolve_prefill_mechanisms
    selected_mechanisms = resolve_prefill_mechanisms(prefill_mechanisms, echo_prefetch=prefill_echo_prefetch, causal_ready=prefill_causal_ready)
    prefill_echo_prefetch = selected_mechanisms in ("echo-only", "both")
    prefill_causal_ready = selected_mechanisms in ("causal-only", "both")
    if type(prefill_causal_ready) is not bool or type(prefill_causal_full_wait) is not bool:
        raise ValueError("Causal readiness options must be boolean compile options")
    if profile != "prefill" and (prefill_causal_ready or prefill_causal_full_wait):
        raise ValueError("Causal cache readiness applies only to prefill")
    if prefill_causal_full_wait and not prefill_causal_ready:
        raise ValueError("Full-prefix diagnostics require causal cache readiness")
    if type(prefill_echo_prefetch) is not bool or type(prefill_echo_diagnostics) is not bool:
        raise ValueError("ECHO options must be booleans")
    if prefill_echo_prefetch and (profile != "prefill" or prefill_indexer_arenas != 2):
        raise ValueError("ECHO requires prefill with two indexer arenas")
    if prefill_echo_diagnostics and not prefill_echo_prefetch:
        raise ValueError("ECHO diagnostics require ECHO prefetch")
    if prefill_mechanisms is not None and (profile != "prefill" or prefill_indexer_arenas != 2 or not prefill_moe_ready):
        raise ValueError("Mechanism comparisons require prefill exact2 and MoE-ready")
    degrees = {family: 8 for family in ("swa", "csa", "hca")}
    degrees.update(projection_degrees or {})
    if set(degrees) != {"swa", "csa", "hca"} or any(type(d) is not int or d not in (1, 2, 4, 8) for d in degrees.values()):
        raise ValueError("Projection degrees require swa/csa/hca and d=1/2/4/8")
    if profile != "prefill" and any(d != 8 for d in degrees.values()):
        raise ValueError("Projection degree selection applies only to prefill")
    if prefill_mechanisms is not None and any(d != 8 for d in degrees.values()):
        raise ValueError("Mechanism comparisons require eight projection replicas")
    projection_degrees = degrees
    if (
        len(device_ids) != 8
        or len(set(device_ids)) != 8
        or any(type(d) is not int or d < 0 for d in device_ids)
    ):
        raise ValueError("Eight distinct logical device IDs are required")
    from pypto.backend import BackendType
    from pypto.ir.distributed_compiled_program import DistributedConfig
    from pypto.jit import CacheConfig, configure_cache
    from pypto.runtime import ExecutionMode, RunConfig

    configure_cache(CacheConfig(enabled=True, root=Path(cache_root), readonly=False))
    before = source_manifest()
    saved_argv = sys.argv
    try:
        # The single-query and verification profiles freeze the shared operators
        # in separate processes. Drafter/Markov retain their own TP4 profile.
        sys.argv = [
            "dspark-bs1-build",
            "--tp",
            "8",
            "--dp",
            "1",
            "--spec-tokens",
            "0" if profile == "single" else "5",
            "--max-seq-len",
            str(max_seq_len),
            "--moe-banks",
            "43",
        ]
        if profile == "prefill":
            from .prefill_fwd import load_prefill_programs

            function = load_prefill_programs(
                indexer_arenas=prefill_indexer_arenas, projection_degrees=projection_degrees,
                moe_ready=prefill_moe_ready, moe_wave_tokens=prefill_moe_wave_tokens, moe_owner_reduce=prefill_moe_owner_reduce, moe_expert_affinity=prefill_moe_expert_affinity,
                causal_ready=prefill_causal_ready,
                causal_full_wait=prefill_causal_full_wait,
                echo_prefetch=prefill_echo_prefetch, echo_diagnostics=prefill_echo_diagnostics,
                mechanisms=prefill_mechanisms,
            )["chunk"]
            kwargs = {}
            selected_source = None
        elif profile == "single":
            module = importlib.import_module(".native_single", __package__)
            function = module.l3_decode_fwd
            kwargs = {}
            selected_source = None
        else:
            module = importlib.import_module(
                ".native_rsag_pipeline" if profile == "rsag" else ".native_pipeline", __package__
            )
            function = module.native_cycles
            selected_source = native_source.fingerprint()
            kwargs = dict(cycle_count=1 if profile == "one" else 2, native_w2_fingerprint=selected_source)
    finally:
        sys.argv = saved_argv
    config = RunConfig(
        backend_type=BackendType.Ascend910B,
        execution_mode=ExecutionMode.ONBOARD,
        device_id=device_ids[0],
        codegen_only=True,
        distributed_config=DistributedConfig(device_ids=device_ids, aicpu_thread_num=4),
    )
    context = native_source.use_native_w2(selected_source) if selected_source else nullcontext()
    with context, ptoas_deadline(ptoas_timeout_seconds):
        compiled = function.compile(config=config, **kwargs)
    directory = artifact_directory(compiled)
    metadata = json.loads((directory / "distributed_meta.json").read_text())
    expected_parameters = len(function.param_names) if profile == "prefill" else (81 if profile == "single" else 153)
    if len(metadata["params"]) != expected_parameters:
        raise RuntimeError("The actual public parameter contract changed")
    if profile != "prefill":
        _validate_target_capacity(metadata, profile, max_seq_len)
    if selected_source:
        # A validated GENERATED cache hit does not re-enter the generation hook.
        # Validate its source choice instead of mistaking the hit for new work.
        selection = json.loads((directory / "report/native-w2-source-selection.json").read_text())
        if selection["source_sha256"] != selected_source or len(selection["selected"]) not in (2, 10):
            raise RuntimeError("Cached program has a different native implementation")
        for item in selection["selected"]:
            original = directory / "report/original-native-source" / item["source"]
            actual = (directory / item["source"]).read_text()
            expected = native_source.replace_cube_source(original.read_text(), selected_source)
            if (
                actual != expected
                or hashlib.sha256(actual.encode()).hexdigest() != item["selected_source_sha256"]
            ):
                raise RuntimeError("Selected native source differs from the reviewed graft")
    if before != source_manifest():
        raise RuntimeError("Pipeline sources changed during compilation")
    report = dict(
        status="GENERATED_NOT_NATIVE_OR_DEVICE_TESTED",
        profile=profile,
        artifact_dir=str(directory),
        cache=str(Path(cache_root).resolve()),
        metadata_sha256=hashlib.sha256((directory / "distributed_meta.json").read_bytes()).hexdigest(),
        public_parameters=len(metadata["params"]),
        sources=before,
        native_w2_source_sha256=selected_source,
        prefill_moe_ready=bool(prefill_moe_ready) if profile == "prefill" else False,
        old_binaries_loaded=False,
        ptoas_timeout_seconds=ptoas_timeout_seconds,
        max_seq_len=max_seq_len,
        model_weights_loaded=False,
        Worker=0,
        NPU=0,
        native=0,
        arithmetic_contract="historical_fast_approximation",
        reference_bitwise_guaranteed=False,
    )
    if profile == "prefill":
        report["prefill_plan"] = dict(
            chunk_tokens=8192, kv_tile_tokens=512, query_block_tokens=128,
            attention_cp=8, moe_tp=8, lm_head_tp=8,
            projection_replicas=projection_degrees, prefetch="exact", transfer_slots=2,
            indexer_arenas=prefill_indexer_arenas,
            moe_ready=prefill_moe_ready,
            moe_wave_tokens=prefill_moe_wave_tokens,
            moe_owner_reduce=prefill_moe_owner_reduce, moe_expert_affinity=prefill_moe_expert_affinity,
            moe_expert_tile_tokens=16,
            moe_copy_tile=[4, 2048], moe_reduce_tile=[8, 512],
            moe_waves_per_chunk=(8192 + prefill_moe_wave_tokens - 1) // prefill_moe_wave_tokens,
            causal_ready=prefill_causal_ready,
            causal_full_wait=prefill_causal_full_wait,
            echo_prefetch=prefill_echo_prefetch, echo_diagnostics=prefill_echo_diagnostics,
            mechanisms=selected_mechanisms, primary_control="exact2",
            echo_bins=256, echo_ema_alpha=0.5, echo_query_slots=2,
            resident_projection_layout="ND",
            dispatches_per_chunk=1, tp8_baseline_bitwise_verified=False,
        )
    return compiled, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["single", "one", "two", "rsag", "prefill"], required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=1048576)
    parser.add_argument("--prefill-indexer-arenas", type=int, choices=(1, 2), default=1)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--ptoas-timeout-seconds",
        type=int,
        default=180,
        help="Per-kernel assembler deadline, between 1 and 600 seconds",
    )
    for family in ("swa", "csa", "hca"):
        parser.add_argument("--projection-" + family + "-d", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--prefill-moe-ready", action=argparse.BooleanOptionalAction, default=None, help="Bounded prefill compute/publish overlap")
    parser.add_argument("--prefill-moe-wave-tokens", type=int, choices=(8, 32, 64, 128, 1024, 2048, 8192), default=None)
    parser.add_argument("--prefill-causal-ready", action="store_true", help="Experimental per-query causal cache readiness")
    parser.add_argument("--prefill-causal-full-wait", action="store_true", help="Diagnostic full-prefix gate with causal control tasks")
    parser.add_argument("--prefill-echo-prefetch", action="store_true", help="Experimental coarse-bin KV prefetch")
    parser.add_argument("--prefill-echo-diagnostics", action="store_true", help="Collect ECHO recall counters")
    parser.add_argument("--prefill-mechanisms", choices=("none", "echo-only", "causal-only", "both"))
    parser.add_argument("--prefill-moe-owner-reduce", action="store_true", help="Experimental ordered owner reduction")
    parser.add_argument("--prefill-moe-expert-affinity", action="store_true", help="Experimental expert work locality")
    args = parser.parse_args()
    if args.report.exists():
        parser.error("The report already exists; preserve it and choose a new run directory")
    devices = [int(v) for v in args.devices.split(",")]
    started = time.monotonic()
    _, report = compile_program(
        args.profile,
        cache_root=args.cache,
        device_ids=devices,
        ptoas_timeout_seconds=args.ptoas_timeout_seconds,
        max_seq_len=args.max_seq_len,
        prefill_indexer_arenas=args.prefill_indexer_arenas,
        prefill_moe_ready=args.prefill_moe_ready,
        prefill_moe_wave_tokens=args.prefill_moe_wave_tokens,
        prefill_moe_owner_reduce=args.prefill_moe_owner_reduce,
        prefill_moe_expert_affinity=args.prefill_moe_expert_affinity,
        prefill_causal_ready=args.prefill_causal_ready,
        prefill_causal_full_wait=args.prefill_causal_full_wait,
        prefill_echo_prefetch=args.prefill_echo_prefetch, prefill_echo_diagnostics=args.prefill_echo_diagnostics,
        prefill_mechanisms=args.prefill_mechanisms,
        projection_degrees={family: getattr(args, "projection_" + family + "_d") for family in ("swa", "csa", "hca")},
    )
    report["generation_seconds"] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
