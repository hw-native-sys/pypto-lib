# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim

"""Validate the production fused CSA kernels against independent Torch math.

Run from the repository root with PYTHONPATH set to that root. This native
runtime driver preserves NZ tensors and workspace aliases; it uses TensorSpec
fixtures and golden.validate_golden for the observable operator contract.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from golden import ratio_allclose, ratio_reldiff, validate_golden
from models.vllm_ascend.deepseek_v4_flash.csa.fixtures import Case, make_fixture, outputs, to_device
from models.vllm_ascend.deepseek_v4_flash.csa.reference import golden_csa


# Same composed-output gate as models/deepseek_v4_flash_mtp/decode_csa.py.
# Cache comparisons cover written rows only; all other storage is checked exactly.
def compare_outputs(actual, expected, native, groups, case, full_scores):
    actual = dict(actual)
    expected = dict(expected)
    valid = sum(case.queries)
    for name in ("x_out", "idx_topk", "idx_topk_scores"):
        actual[name] = actual[name][:valid]
        expected[name] = expected[name][:valid]
    group_names = {
        "swa": "kv_cache",
        "compressed": "cmp_kv",
        "state": "compress_state",
        "indexer_state": "inner_compress_state",
    }
    for name, key in group_names.items():
        itemsize = native[key].element_size()
        mask = groups[key]["allowed"][128:-128].reshape(-1, itemsize).any(1)
        actual[name] = actual[name].flatten()[mask]
        expected[name] = expected[name].flatten()[mask]
    mask = groups["idx_native_kv_cache"]["allowed"][128:-128].reshape(-1, 4160)
    for name, select in [
        ("indexer_key", mask[:, :4096]),
        ("indexer_scale", mask[:, 4096:].reshape(-1, 32, 2).any(-1)),
    ]:
        actual[name] = actual[name].flatten()[select.flatten()]
        expected[name] = expected[name].flatten()[select.flatten()]

    def ranked(actual_indices, expected_indices, **kwargs):
        if actual_indices.shape != expected_indices.shape:
            return False, "Top-K shape mismatch"
        for token, row in enumerate(actual_indices.long()):
            bound = (int(native["position_ids"][token]) + 1) // 4
            if (row < 0).any() or (row >= bound).any() or row.unique().numel() != 512:
                return False, f"Invalid, duplicate, or noncausal Top-K indices at token {token}"
        selected = full_scores[:valid].gather(1, actual_indices.long())
        reported = kwargs["actual_outputs"]["idx_topk_scores"]
        gate = ratio_allclose(atol=1e-5, rtol=0.02, max_error_ratio=0.005)
        ok, detail = gate(reported, selected, **kwargs)
        if not ok:
            return False, "Selected scores disagree with Torch: " + detail
        cutoff = expected["idx_topk_scores"][:, -1:]
        if (selected < cutoff - 0.02 * cutoff.abs() - 1e-5).any():
            return False, "Selected indices fall below the Torch Top-K cutoff"
        if (reported[:, 1:] > reported[:, :-1] + 1e-6).any():
            return False, "Top-K scores are not descending"
        return True, ""

    comparisons = {
        "x_out": ratio_reldiff(diff_thd=4e-3, pct_thd=0.03, max_diff_hd=2),
        "idx_topk": ranked,
        "idx_topk_scores": ratio_allclose(atol=1e-5, rtol=0.02, max_error_ratio=0.005),
        "swa": ratio_allclose(atol=1e-4, rtol=1 / 128),
        "compressed": ratio_allclose(atol=1e-4, rtol=1 / 128),
        "state": ratio_allclose(atol=1e-4, rtol=1e-3),
        "indexer_state": ratio_allclose(atol=1e-4, rtol=1e-3),
        "indexer_key": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.005),
        "indexer_scale": ratio_allclose(atol=1e-5, rtol=1 / 128),
    }
    validate_golden(actual, expected, rtol=1e-2, atol=1e-2, compare_fn=comparisons)


def exact(actual, expected, label):
    if actual.keys() != expected.keys():
        raise AssertionError(label + ": output names differ")
    for name, value in actual.items():
        other = expected[name]
        if (
            value.shape != other.shape
            or value.dtype != other.dtype
            or not torch.equal(value.contiguous().view(torch.uint8), other.contiguous().view(torch.uint8))
        ):
            raise AssertionError(f"{label}: {name} is not bit-identical")
    print(f"[RUN] {label}: all nine states bitwise PASS", flush=True)


def run_case(case, args, case_dir):
    import torch_npu  # noqa: F401 - register the torch.npu backend
    import pypto.torch
    from pypto.runtime import CompileOptions, RunConfig
    from pypto.runtime.kernel.context import get_process_kernel_state

    modes = ("spmd", "mpmd") if args.mode == "both" else (args.mode,)
    case_dir.mkdir(parents=True, exist_ok=False)
    config = CompileOptions(dump_passes=args.dump_passes)
    if args.compile_only:
        for mode in modes:
            if mode == "spmd":
                from models.vllm_ascend.deepseek_v4_flash.csa.spmd import entry, policy
            else:
                from models.vllm_ascend.deepseek_v4_flash.csa.mpmd import entry, policy
            selected = policy.select_policy(
                tokens=case.tokens,
                requests=case.requests,
                max_query_tokens=case.query,
                max_seq_len=case.max_length,
            )
            function = getattr(entry, "decode_csa_spmd" if mode == "spmd" else "decode_csa_tnd")
            function.warmup(
                **selected.constexprs(),
                config=RunConfig(
                    platform=args.platform,
                    dump_passes=args.dump_passes,
                    save_kernels_dir=str(case_dir / mode),
                ),
            )
        return {"compile_passed": True, "accuracy": "NOT_RUN"}

    print("[RUN] generate TensorSpec inputs", flush=True)
    native, groups = make_fixture(case)
    print("[RUN] independent Torch golden", flush=True)
    reference = {name: value.clone() for name, value in native.items()}
    full_scores = golden_csa(reference, case)
    expected = outputs(reference)
    del reference
    if args.save_data:
        torch.save({"case": asdict(case), "inputs": native, "expected": expected}, case_dir / "golden.pt")
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    pypto.torch.init(
        device=args.device,
        platform=args.platform,
        runtime="tensormap_and_ringbuffer",
        enable_chip_swimlane=args.enable_chip_swimlane,
        output_dir=str(args.output / "dfx") if args.enable_chip_swimlane else None,
        ring_heap=(256 * 1024 * 1024,) * 4,
        ring_task_window=16384,
    )
    first = None
    report = {}
    for mode in modes:
        package = f"models.vllm_ascend.deepseek_v4_flash.csa.{mode}"
        if mode == "spmd":
            from models.vllm_ascend.deepseek_v4_flash.csa.spmd.binding import PreparedDynamicCSA
        else:
            from models.vllm_ascend.deepseek_v4_flash.csa.mpmd.binding import PreparedDynamicCSA
        device_native, cache_guards = to_device(native, groups, torch.device(f"npu:{args.device}"))
        workspaces = {}

        def allocate(name, shape, dtype, device):
            size = torch.empty((), dtype=dtype).element_size()
            count = 1
            for dim in shape:
                count *= dim
            allocation = torch.full((count * size + 1024,), 37, dtype=torch.uint8, device=device)
            workspaces[name] = allocation
            return allocation[512:-512].view(dtype).view(shape)

        prepared = PreparedDynamicCSA(
            device_native, max_query_tokens=case.query, max_seq_len=case.max_length, allocator=allocate
        )
        previous = None
        for repeat in range(2):
            for name, group in groups.items():
                cache_guards[name].copy_(group["allocation"])
            device_native["x_out"].zero_()
            device_native["idx_topk"].fill_(-1)
            device_native["idx_topk_scores"].zero_()
            print(f"[RUN] {mode} execute repeat={repeat}", flush=True)
            prepared(config=config)
            torch.npu.synchronize()
            # The pinned eager bridge retains native RunHandles after Torch
            # synchronization. Retire them before registering another variant.
            get_process_kernel_state().drain()
            actual = outputs(device_native)
            if args.save_data:
                torch.save(actual, case_dir / (mode + "_outputs.pt"))
            for name, allocation in cache_guards.items():
                mask = ~groups[name]["allowed"]
                if not torch.equal(allocation.cpu()[mask], groups[name]["allocation"][mask]):
                    raise AssertionError(f"{mode}: out-of-slot cache write: {name}")
            for name, allocation in workspaces.items():
                if not (allocation[:512].cpu() == 37).all() or not (allocation[-512:].cpu() == 37).all():
                    raise AssertionError(f"{mode}: workspace guard corrupted: {name}")
            for name, value in device_native.items():
                if (
                    any(
                        part in name
                        for part in ("slot_mapping", "block_table", "seq_lens", "query_start_loc", "freqs_")
                    )
                    or name == "position_ids"
                ):
                    if not torch.equal(value.cpu(), native[name]):
                        raise AssertionError(f"Read-only metadata changed: {name}")
            if case.mixed:
                valid = sum(case.queries)
                if (
                    torch.count_nonzero(actual["x_out"][valid:])
                    or not (actual["idx_topk"][valid:] == -1).all()
                    or not (actual["idx_topk_scores"][valid:] == torch.finfo(torch.float32).min).all()
                ):
                    raise AssertionError("Inactive token outputs were not masked")
            if repeat == 0:
                print(f"[RUN] {mode} validate against Torch", flush=True)
                compare_outputs(actual, expected, native, groups, case, full_scores)
                previous = actual
            else:
                exact(actual, previous, mode + " repeat")
        if first is None:
            first = actual
        else:
            exact(actual, first, "SPMD/MPMD")
        report[mode] = {
            "passed": True,
            "kernel_module": package,
            "torch_golden": True,
            "guards": True,
            "repeat_bit_equal": True,
        }
        del prepared, device_native, cache_guards
        workspaces.clear()
    if "vllm" in sys.modules or "vllm_ascend" in sys.modules:
        raise AssertionError("Standalone validation imported the serving framework")
    report["mpmd_spmd_bit_equal"] = True if args.mode == "both" else "NOT_RUN"
    report["standalone_without_vllm"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["spmd", "mpmd", "both"], default="both")
    parser.add_argument("--case", choices=["uniform", "mixed", "single", "all"], default="all")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--query", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "2")
    if os.environ["VLLM_ASCEND_ENABLE_NZ"] != "2":
        raise ValueError("Standalone TND CSA requires NZ2")
    if args.output is None:
        from datetime import datetime

        args.output = Path("build_output") / ("csa_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "passed": False,
        "scope": "fused CSA operator; synthetic weights and inputs; independent Torch reference",
        "accuracy": "NOT_RUN" if args.compile_only else "PENDING",
        "cases": {},
    }
    try:
        cases = ["uniform", "mixed", "single"] if args.case == "all" else [args.case]
        for name in cases:
            mixed = name == "mixed"
            case = Case(
                batch=max(2, args.batch) if mixed else args.batch,
                history=args.history,
                query=1 if name == "single" else args.query,
                mixed=mixed,
                seed=args.seed,
            )
            report["cases"][name] = {
                "configuration": asdict(case),
                "results": run_case(case, args, args.output / name),
            }
        report["passed"] = True
        if not args.compile_only:
            report["accuracy"] = "PASS"
    except Exception:
        report["error"] = traceback.format_exc()
        if not args.compile_only:
            report["accuracy"] = "FAIL"
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(
        "[RUN] "
        + (
            "COMPILE PASS (accuracy not run)"
            if args.compile_only and report["passed"]
            else "PASS"
            if report["passed"]
            else "FAIL"
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
