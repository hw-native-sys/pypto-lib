# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Freeze KDA inputs and compare PTO3 and vLLM-Ascend core computation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(samples):
    return {"rounds": len(samples), "median_us": statistics.median(samples),
            "mean_us": statistics.mean(samples), "min_us": min(samples),
            "max_us": max(samples), "stddev_us": statistics.stdev(samples) if len(samples) > 1 else 0,
            "samples_us": samples}


def fixture(config):
    generator = torch.Generator().manual_seed(config["seed"])
    requests, tokens, heads = (config[n] for n in ("requests", "tokens", "heads"))
    length = tokens // requests
    shape = (tokens, heads, 128)
    beta_width = (heads + 15) // 16 * 16
    starts = torch.arange(requests + 1, dtype=torch.int32) * length
    columns = 2 + (length if config["phase"] == "decode" else (length + 15) // 16)
    rows = torch.full((requests, columns), -1, dtype=torch.int32)
    rows[:, 0] = torch.arange(requests, dtype=torch.int32).roll(1)
    rows[:, 1] = requests + torch.arange(requests, dtype=torch.int32)
    state = torch.randn(2 * requests + 1, heads, 128, 128, generator=generator) * 0.1
    if config["initial_state"] == "zero":
        state.zero_()
    return {
        "query": torch.randn(shape, generator=generator).bfloat16(),
        "key": (torch.randn(shape, generator=generator) * 0.4 + torch.randn(1, heads, 128, generator=generator)).bfloat16(),
        "value": torch.randn(shape, generator=generator).bfloat16(),
        "decay": -5 * torch.randn(shape, generator=generator).sigmoid(),
        "beta": torch.randn(tokens, beta_width, generator=generator).sigmoid(),
        "recurrent_state": state,
        "query_start_loc": starts,
        "state_rows": rows,
        "output": torch.zeros(shape, dtype=torch.bfloat16),
    }


def read_case(directory, config):
    manifest = json.loads((directory / "inputs.json").read_text())
    if manifest["config"] != config or file_hash(directory / "inputs.pt") != manifest["sha256"]:
        raise ValueError("frozen case configuration or input hash differs")
    return torch.load(directory / "inputs.pt", map_location="cpu", weights_only=True)


def pto_run(args, config, directory):
    # Select module-level TP shapes before importing the operator.
    sys.argv += ["--tp", str(config["tp"]), "--ep", str(config["tp"])]
    from golden import TensorSpec, ratio_allclose, run
    from models.glm5_3_flash import kda

    if not (directory / "inputs.json").exists():
        values = fixture(config)
        torch.save(values, directory / "inputs.pt")
        write_json(directory / "inputs.json", {"config": config, "sha256": file_hash(directory / "inputs.pt")})
    values = read_case(directory, config)
    names = ["query", "key", "value", "decay", "beta", "recurrent_state", "query_start_loc", "state_rows"]
    if config["phase"] == "prefill":
        values.update({name.removeprefix("kda_"): value for name, value in kda.kda_host_constants().items()})
        names += ["tri_incl", "tri_strict", "head_onehot", "eye_wide"]
    names += ["output"]
    specs = [TensorSpec(name, list(values[name].shape), values[name].dtype,
                        init_value=lambda name=name: values[name].clone()) for name in names]
    common_path = directory / "fp32-reference.pt"
    implementation_path = directory / "pto-reference.pt"
    if not common_path.exists() or not implementation_path.exists():
        arguments = [values[name] for name in names[:8]]
        output, state = kda.golden_prefill_kda_fp32(*arguments)
        torch.save({"output": output, "recurrent_state": state}, common_path)
        if config["phase"] == "prefill":
            output, state = kda.golden_prefill_kda_mixed_precision(*arguments)
        torch.save({"output": output, "recurrent_state": state}, implementation_path)
    expected = torch.load(implementation_path, weights_only=True)

    def golden(tensors):
        for name, value in expected.items():
            tensors[name].copy_(value)

    actuals = {}
    compare = ratio_allclose(max_error_ratio=0)
    destinations = set(values["state_rows"][:, 1].tolist())
    untouched = [n for n in range(values["recurrent_state"].shape[0]) if n not in destinations]

    def capture(name):
        def comparator(actual, reference, **kwargs):
            actuals[name] = actual.clone()
            if name == "recurrent_state" and not torch.equal(actual[untouched], values[name][untouched]):
                return False, "read-only seeds or unused state rows changed"
            return compare(actual, reference, **kwargs)
        return comparator

    os.environ.update(PYPTO_BENCH="1", PYPTO_BENCH_ROUNDS=str(args.rounds),
                      PYPTO_BENCH_WARMUP=str(args.warmup), PYPTO_BENCH_RAW="1")
    fn = kda.prefill_kda_test if config["phase"] == "prefill" else kda.recurrent_kda_test
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden,
        compare_fn={name: capture(name) for name in expected},
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode,
                "device_id": args.device, "ring_heap": 1 << 30},
        rtol=0.02,
        atol=0.02,
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise RuntimeError(result.error)
    if not actuals:
        # CPU-only CI preparation does not execute numerical comparisons.
        if args.compile_only:
            from pypto.runtime.device_runner import _compile_and_assemble

            _compile_and_assemble(result.work_dir, args.backend_type)
        return
    torch.save(actuals, directory / "pto-actual.pt")
    if args.execution_mode:
        print("Simulator numerical validation passed; device timing unavailable")
        return
    if result.bench is None or result.bench.all_zero_device:
        raise RuntimeError("real device-domain benchmark samples are required")
    write_json(directory / "pto-result.json", {
        "engine": "pto", "config": config, "inputs_sha256": file_hash(directory / "inputs.pt"),
        "passed": result.passed, "warmup": result.bench.warmup, "device": args.device,
        "timing": "single-device effective wall time; immutable seed; no restore kernel",
        "benchmark": stats(result.bench.per_round("effective")),
        "source_sha256": file_hash(Path(kda.__file__)), "torch_version": torch.__version__,
    })


def vllm_run(args, config, directory):
    import torch_npu

    torch.npu.set_device(args.device)
    from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
    import vllm_ascend.ops.kda as reference_module

    values = read_case(directory, config)
    device = torch.device(f"npu:{args.device}")
    seed_rows = values["state_rows"][:, 0].long()
    output_rows = values["state_rows"][:, 1].long()
    initial = values["recurrent_state"][seed_rows].contiguous().to(device)
    state = initial.clone()
    immutable = {name: values[name].unsqueeze(0).to(device) for name in ("query", "key", "value", "decay")}
    work = {name: immutable[name].clone() for name in ("query", "key", "value")}
    beta = values["beta"][:, :config["heads"]].unsqueeze(0).contiguous().to(device)
    starts = values["query_start_loc"].to(device)
    indices = torch.arange(config["requests"], dtype=torch.int32, device=device)
    host_starts = values["query_start_loc"].tolist()
    chunks = [item for n, (begin, end) in enumerate(zip(host_starts, host_starts[1:]))
              for block in range((end - begin + 63) // 64) for item in (n, block)]

    def reset():
        state.copy_(initial)
        for name, value in work.items():
            value.copy_(immutable[name])

    def invoke():
        if config["phase"] == "decode":
            output, _ = recurrent_kda(
                work["query"], work["key"], work["value"], immutable["decay"], beta, state,
                cu_seqlens=starts, ssm_state_indices=indices, layout="BSND", scale=128**-0.5,
                output_final_state=False, inplace_final_state=True, state_v_first=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, safe_gate=False,
            )
            return output, state
        output, final, *_ = chunk_kda_fwd(
            l2norm_fwd(work["query"]), l2norm_fwd(work["key"]), work["value"], immutable["decay"],
            beta, 128**-0.5, 64, layout="BSND", initial_state=state, output_final_state=True,
            cu_seqlens=host_starts, chunk_indices=chunks, state_v_first=True,
            use_gate_in_kernel=False, safe_gate=False, disable_recompute=False,
            return_intermediate_states=False,
        )
        return output, final

    with torch.inference_mode():
        reset()
        output, final = invoke()
        torch.npu.synchronize()
        if args.vllm_mode == "graph":
            reset()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output, final = invoke()
            torch.npu.synchronize()
            step = graph.replay
        else:
            def step():
                nonlocal output, final
                output, final = invoke()
        samples = []
        for n in range(args.warmup + args.rounds):
            reset()
            # Events are recorded outside graph capture; reset copies precede start.
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            step()
            end.record()
            end.synchronize()
            if n >= args.warmup:
                samples.append(float(begin.elapsed_time(end)) * 1000)
        if not all(value > 0 for value in samples):
            raise RuntimeError("reference device timing is unavailable")
        actual_state = values["recurrent_state"].clone()
        actual_state[output_rows] = final.cpu().reshape_as(initial.cpu())
        actual = {"output": output.cpu().reshape_as(values["output"]), "recurrent_state": actual_state}
        if not all(torch.isfinite(value).all() for value in actual.values()):
            raise RuntimeError("reference produced nonfinite results")
        if not torch.equal(initial.cpu(), values["recurrent_state"][seed_rows]):
            raise RuntimeError("reference mutated its immutable initial-state source")
        if not torch.equal(immutable["decay"].cpu().squeeze(0), values["decay"]):
            raise RuntimeError("reference mutated the shared preprocessed decay")
        if not torch.equal(beta.cpu().squeeze(0), values["beta"][:, :config["heads"]]):
            raise RuntimeError("reference mutated the shared preprocessed beta")
    torch.save(actual, directory / "vllm-actual.pt")
    versions = {}
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "fla-npu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    write_json(directory / "vllm-result.json", {
        "engine": "vllm-ascend/fla-npu-ascendc", "config": config,
        "inputs_sha256": file_hash(directory / "inputs.pt"), "warmup": args.warmup,
        "device": args.device, "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "timing": f"NPU Event device elapsed, {args.vllm_mode}; events outside capture; resets excluded",
        "mode": args.vllm_mode, "benchmark": stats(samples), "versions": versions,
        "torch_npu_version": torch_npu.__version__,
        "source_sha256": file_hash(Path(reference_module.__file__)),
        "chunk_size": 64 if config["phase"] == "prefill" else None,
        "normalization": "BF16 normalized operands for prefill; fused normalization for decode",
        "gate_beta": "shared preprocessed FP32 gate/beta; gate/sigmoid fusion disabled",
        "state_layout": "V,K", "state_dtype": "float32",
    })


def compare_report(config, directory):
    from golden import ratio_allclose

    values = read_case(directory, config)
    rows = values["state_rows"][:, 1].long()
    fp32 = torch.load(directory / "fp32-reference.pt", weights_only=True)
    engines = {}
    for engine in ("pto", "vllm"):
        result = json.loads((directory / f"{engine}-result.json").read_text())
        if result["config"] != config or result["inputs_sha256"] != file_hash(directory / "inputs.pt"):
            raise ValueError("implementation reports do not share the frozen workload")
        actual = torch.load(directory / f"{engine}-actual.pt", weights_only=True)
        engines[engine] = (result, actual)
    sampling = {(data[0]["benchmark"]["rounds"], data[0]["warmup"]) for data in engines.values()}
    if len(sampling) != 1:
        raise ValueError("implementation reports use different sampling counts")
    report = {"config": config, "comparisons": {}}
    gate = ratio_allclose(rtol=0.02, atol=0.02, max_error_ratio=0)
    for name in ("output", "recurrent_state"):
        selection = rows if name == "recurrent_state" else slice(None)
        reference = fp32[name][selection]
        for label, actual in [("pto_vs_fp32", engines["pto"][1][name][selection]),
                              ("vllm_vs_fp32", engines["vllm"][1][name][selection]),
                              ("pto_vs_vllm", engines["pto"][1][name][selection])]:
            expected = engines["vllm"][1][name][selection] if label == "pto_vs_vllm" else reference
            passed, details = gate(actual, expected, rtol=0.02, atol=0.02,
                                   actual_outputs={}, expected_outputs={}, inputs=values)
            difference = actual.double() - expected.double()
            denominator = expected.double().square().sum().sqrt().item()
            relative_l2 = difference.square().sum().sqrt().item() / max(denominator, 1e-30) * 100
            report["comparisons"][f"{name}:{label}"] = {
                "passed": passed, "relative_l2_percent": relative_l2,
                "max_absolute_error": difference.abs().max().item(),
                "details": details,
            }
    report["performance"] = {engine: data[0]["benchmark"] for engine, data in engines.items()}
    report["implementation_contracts"] = {engine: {k: v for k, v in data[0].items() if k != "benchmark"}
                                          for engine, data in engines.items()}
    write_json(directory / "comparison.json", report)
    lines = ["# KDA core comparison", "", f"Case: `{json.dumps(config)}`", "",
             "| Engine | Median us | Mean us | Stddev us |", "|---|---:|---:|---:|"]
    for engine, data in engines.items():
        timing = data[0]["benchmark"]
        lines.append(f"| {engine} | {timing['median_us']:.3f} | {timing['mean_us']:.3f} | {timing['stddev_us']:.3f} |")
    for engine, data in engines.items():
        lines += ["", f"{engine}: {data[0]['timing']}; "
                  f"{data[0]['benchmark']['rounds']} samples, {data[0]['warmup']} warmups."]
    lines += ["", "| Tensor / comparison | Relative L2 % | Max abs | Allclose |", "|---|---:|---:|---|"]
    for name, metric in report["comparisons"].items():
        lines.append(f"| {name} | {metric['relative_l2_percent']:.6f} | {metric['max_absolute_error']:.6g} | {metric['passed']} |")
    lines += ["", "Inputs and initial states are identical. Q/K normalization is included; projections,",
              "convolution, gate/beta preprocessing, layout conversion and reset copies are excluded.",
              "PTO3 prefill uses chunk16 and its existing BF16 boundaries. Reference chunk size and",
              "normalization dtypes are recorded separately; cross-engine equality is not assumed.",
              "Relative L2 describes the whole tensor; allclose uses rtol=0.02, atol=0.02, zero outliers."]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"case": config, "medians_us": {k: v[0]["benchmark"]["median_us"] for k, v in engines.items()},
                      "comparisons": report["comparisons"]}), flush=True)
    return all(x["passed"] for x in report["comparisons"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["pto", "vllm-ascend", "report"], default="pto")
    parser.add_argument("--phase", choices=["prefill", "decode"])
    parser.add_argument("--tp", type=int, choices=[8, 16])
    parser.add_argument("--ep", type=int, choices=[8, 16])
    parser.add_argument("--requests", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-state", choices=["zero", "nonzero"])
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--vllm-mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    directory = args.case_dir
    if directory is None:
        if args.engine != "pto":
            parser.error("reference and report require --case-dir from a PTO3 case")
        (ROOT / "build_output").mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="kda-case-", dir=ROOT / "build_output"))
    directory.mkdir(parents=True, exist_ok=True)
    previous = json.loads((directory / "inputs.json").read_text())["config"] if (directory / "inputs.json").exists() else {}
    defaults = {"phase": "decode", "tp": 16, "requests": 1, "seed": 1024, "initial_state": "nonzero"}
    config = {key: getattr(args, key) if getattr(args, key) is not None else previous.get(key, value)
              for key, value in defaults.items()}
    config["tokens"] = args.tokens if args.tokens is not None else previous.get("tokens", config["requests"] if config["phase"] == "decode" else 128)
    config["heads"] = 64 // config["tp"]
    if args.ep is not None and args.ep != config["tp"]:
        parser.error("local-shard cases require --ep to match --tp")
    if config["requests"] < 1 or config["tokens"] < config["requests"] or config["tokens"] % config["requests"]:
        parser.error("tokens must be positive and divisible by requests")
    if config["phase"] == "decode" and config["tokens"] != config["requests"]:
        parser.error("decode comparison is one token per request; use --component verify for MTP")
    if args.rounds < 1 or args.warmup < 0:
        parser.error("invalid sampling counts")
    if args.engine != "pto" and (args.execution_mode or args.compile_only):
        parser.error("reference/report do not support simulator or compile-only execution")
    if args.engine == "pto":
        pto_run(args, config, directory)
    elif args.engine == "vllm-ascend":
        vllm_run(args, config, directory)
    elif not compare_report(config, directory):
        return 1
    print(f"KDA_CASE_DIR={directory.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
