# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare fixed token traces, capture PyPTO logits, and compare DP4/TP4/EP16 outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

import torch

TOPOLOGY = dict(dp=4, tp=4, ep=16, pp=1, pcp=1, dcp=1)
MAX_CONTEXT = 16384
MAX_PREFILL_CHUNK_SIZE = 8192
MAX_TRACE = 256
CHECKPOINT_METADATA = (
    "config.json", "quant_model_weights.safetensors.index.json", "quant_model_description.json", "tokenizer.json",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_identity(checkpoint):
    root = Path(checkpoint).resolve()
    def checkpoint_file(name):
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError(f"File outside the checkpoint directory: {name}")
        return path

    identity = {name: file_sha256(checkpoint_file(name)) for name in CHECKPOINT_METADATA}
    index = read_json(root / "quant_model_weights.safetensors.index.json")
    description = read_json(checkpoint_file("quant_model_description.json"))
    rotation = description.get("optional", {}).get("quarot", {}).get("rotation_map", {}).get("global_rotation")
    if rotation is not None:
        identity["global_rotation_sha256"] = file_sha256(checkpoint_file(rotation))
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        path = checkpoint_file(name)
        stat = path.stat()
        shards[name] = dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    identity["weight_shard_stats"] = shards
    return identity


def validate_cases(spec):
    if spec.get("schema_version") != 1 or spec.get("topology") != TOPOLOGY:
        raise ValueError("Require schema_version=1 and DP4/TP4/EP16, PP1/PCP1/DCP1")
    vocabulary = spec.get("vocab_size")
    if type(vocabulary) is not int or vocabulary <= 0:
        raise ValueError("Require a positive checkpoint vocabulary size")
    if not isinstance(spec.get("cases"), list) or not spec["cases"]:
        raise ValueError("Require at least one accuracy case")
    names = set()
    for case in spec["cases"]:
        name = case.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("Case IDs must be unique nonempty letters, numbers, underscores, or hyphens")
        names.add(name)
        for field in ("prompt_token_ids", "trace_token_ids"):
            ids = case.get(field)
            if not isinstance(ids, list) or not ids or any(type(t) is not int or not 0 <= t < vocabulary for t in ids):
                raise ValueError(f"{name}: {field} must contain nonempty, in-vocabulary integer IDs")
        if len(case["trace_token_ids"]) > MAX_TRACE:
            raise ValueError(f"{name}: trace exceeds {MAX_TRACE} tokens")
        if len(case["prompt_token_ids"]) + len(case["trace_token_ids"]) > MAX_CONTEXT:
            raise ValueError(f"{name}: prompt plus trace exceeds {MAX_CONTEXT} positions")
    return spec


def load_cases(path, *, verify_checkpoint=True):
    spec = validate_cases(read_json(path))
    if verify_checkpoint and checkpoint_identity(spec["checkpoint"]) != spec.get("checkpoint_identity"):
        raise ValueError("Checkpoint metadata differs from the prepared cases; prepare a new case file")
    return spec


def prepare_cases(checkpoint, source, output):
    checkpoint = Path(checkpoint).resolve()
    model = read_json(checkpoint / "config.json")
    text = model["text_config"]
    quantization = model.get("quantization_config", {})
    if model.get("architectures") != ["DeepseekV41ForCausalLM"] or text.get("num_hidden_layers") != 40:
        raise ValueError("Require the original 40-layer DeepseekV41ForCausalLM checkpoint")
    if quantization.get("quant_method") != "ascend" or quantization.get("model_quant_type") != "W8A8_DYNAMIC":
        raise ValueError("Require an Ascend W8A8_DYNAMIC checkpoint")
    raw = read_json(source)
    if raw.get("topology", TOPOLOGY) != TOPOLOGY:
        raise ValueError("Accuracy verification is fixed to DP4/TP4/EP16")
    cases, tokenizer = [], None
    for item in raw["cases"]:
        case = {"id": item["id"]}
        for ids_field, text_field in (("prompt_token_ids", "prompt"), ("trace_token_ids", "trace")):
            if (ids_field in item) == (text_field in item):
                raise ValueError(f"{item['id']}: supply exactly one of {ids_field} and {text_field}")
            if ids_field in item:
                case[ids_field] = item[ids_field]
            else:
                if not isinstance(item[text_field], str):
                    raise ValueError(f"{item['id']}: {text_field} must be text")
                if tokenizer is None:
                    from tokenizers import Tokenizer

                    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
                    tokenizer.no_padding()
                    tokenizer.no_truncation()
                case[ids_field] = tokenizer.encode(item[text_field], add_special_tokens=False).ids
        cases.append(case)
    spec = dict(schema_version=1, topology=TOPOLOGY, checkpoint=str(checkpoint),
                checkpoint_identity=checkpoint_identity(checkpoint), vocab_size=text["vocab_size"], cases=cases)
    validate_cases(spec)
    if Path(output).exists():
        raise FileExistsError("Prepared cases already exist; use a new output path")
    write_json(output, spec)
    return spec


def allocated_devices():
    assigned = os.environ.get("TASK_DEVICE", "").split(",")
    if len(assigned) != TOPOLOGY["ep"] or any(not value.isdecimal() for value in assigned):
        raise ValueError("Execution requires task-submit --device auto --device-num 16 and TASK_DEVICE")
    devices = [int(value) for value in assigned]
    if len(set(devices)) != TOPOLOGY["ep"]:
        raise ValueError("Require 16 distinct allocated physical devices")
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible is not None and visible != ",".join(assigned):
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES differs from TASK_DEVICE")
    return devices


def new_capture(spec, cases_path, backend):
    versions = {}
    for name in ("torch", "pypto") if backend == "pypto" else ("torch", "torch-npu", "vllm", "vllm-ascend"):
        versions[name] = importlib.metadata.version(name)
    return dict(schema_version=1, status="running", backend=backend, topology=TOPOLOGY,
                case_spec_sha256=file_sha256(cases_path), checkpoint_identity=spec["checkpoint_identity"],
                checkpoint=spec["checkpoint"], vocab_size=spec["vocab_size"], output_kind="raw_logits",
                active_dp_rank=0, python=sys.executable, versions=versions,
                expected_case_ids=[case["id"] for case in spec["cases"]], cases=[])


def save_case(root, case, logits, vocabulary, **metadata):
    tensor = torch.as_tensor(logits).detach().cpu().to(torch.float32).contiguous()
    expected = (len(case["trace_token_ids"]), vocabulary)
    if tuple(tensor.shape) != expected or not torch.isfinite(tensor).all():
        raise ValueError(f"{case['id']}: require finite full-vocabulary logits with shape {expected}")
    filename = case["id"] + ".pt"
    torch.save(tensor, Path(root) / filename)
    trace, prompt = case["trace_token_ids"], case["prompt_token_ids"]
    return dict(case, contexts=[prompt + trace[:step] for step in range(len(trace))],
                phases=["prefill"] + ["decode"] * (len(trace) - 1), logits_file=filename,
                logits_sha256=file_sha256(Path(root) / filename), logits_shape=list(tensor.shape),
                argmax_token_ids=tensor.argmax(dim=-1).tolist(), **metadata)


def capture_pypto(spec, cases_path, output, *, chunk_size=16, graph_dir=None):
    """Replay each fixed history through the same Prefill and Decode artifacts."""
    devices = allocated_devices()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = new_capture(spec, cases_path, "pypto")
    report.update(devices=devices, chunk_size=chunk_size)
    write_json(output / "manifest.json", report)
    try:
        for case in spec["cases"]:
            directory = output / "requests" / case["id"]
            trace = case["trace_token_ids"]
            request_path = directory / "input.json"
            write_json(request_path, dict(request_id=case["id"], dp_rank=0,
                                          input_ids=case["prompt_token_ids"], trace_token_ids=trace))
            report_path = directory / "result.json"
            logits_path = directory / "logits.pt"
            cmd = [sys.executable, "-m", "models.deepseek_v4_1_flash_a3.generate",
                   "--checkpoint", spec["checkpoint"], "--devices", ",".join(map(str, devices)),
                   "--input-ids-json", str(request_path), "--chunk-size", str(chunk_size),
                   "--max-new-tokens", str(len(trace)), "--ignore-eos",
                   "--report", str(report_path), "--save-logits", str(logits_path)]
            if graph_dir is None:
                graph_dir = output / "graphs"
                longest = max(len(c["prompt_token_ids"]) + len(c["trace_token_ids"]) for c in spec["cases"])
                cmd.extend(("--output", str(graph_dir), "--context-capacity", str(max(128, chunk_size, longest))))
            else:
                cmd.extend(("--graph-dir", str(graph_dir)))
            log_path = directory / "execution.log"
            with log_path.open("w") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise ValueError(f"{case['id']}: full graph exited with {completed.returncode}. See {log_path}")
            execution = read_json(report_path)
            if (not execution["teacher_forced"] or execution["decode_input_ids"] != trace[:-1]
                    or execution["prefill_calls"] != math.ceil(len(case["prompt_token_ids"]) / chunk_size)
                    or execution["decode_calls"] != len(trace) - 1):
                raise ValueError(f"{case['id']}: full graph executed a different token history")
            result = torch.load(logits_path, map_location="cpu", weights_only=True)
            record = save_case(output, case, result["logits"], spec["vocab_size"],
                               execution_report=str(report_path.relative_to(output)))
            if result["token_ids"] != record["argmax_token_ids"]:
                raise ValueError(f"{case['id']}: emitted tokens differ from the captured logits")
            logits_path.unlink()
            report["cases"].append(record)
            write_json(output / "manifest.json", report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "manifest.json", report)
    return report


def capture_manifest_path(path):
    path = Path(path)
    return path / "manifest.json" if path.is_dir() else path


def load_capture(path, backend):
    path = capture_manifest_path(path)
    capture = read_json(path)
    if capture.get("schema_version") != 1 or capture.get("status") != "complete":
        raise ValueError(f"{backend}: capture is incomplete or has an unsupported schema")
    if capture.get("backend") != backend or capture.get("topology") != TOPOLOGY:
        raise ValueError(f"{backend}: wrong backend or topology")
    if capture.get("output_kind") != "raw_logits" or capture.get("active_dp_rank") != 0:
        raise ValueError(f"{backend}: require raw logits from DP rank 0")
    validate_cases(dict(capture, schema_version=1))
    if capture.get("expected_case_ids") != [case["id"] for case in capture["cases"]]:
        raise ValueError(f"{backend}: capture does not contain all prepared cases in order")
    return path.parent.resolve(), capture


def load_logits(root, record, vocabulary):
    path = (root / record["logits_file"]).resolve()
    if root not in path.parents or file_sha256(path) != record["logits_sha256"]:
        raise ValueError("Logits file is outside the capture or has changed")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    expected = (len(record["trace_token_ids"]), vocabulary)
    if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected
            or tensor.dtype != torch.float32):
        raise ValueError(f"{record['id']}: logits shape or dtype does not match the FP32 full-vocabulary contract")
    if record.get("logits_shape") != list(expected) or not torch.isfinite(tensor).all():
        raise ValueError(f"{record['id']}: nonfinite logits or inconsistent shape metadata")
    return tensor.to(torch.float32)


def step_metrics(actual, reference, trace_token, topk):
    actual, reference = actual.to(torch.float64), reference.to(torch.float64)
    delta = actual - reference
    absolute_error = delta.abs()
    worst_token = int(absolute_error.argmax())
    actual_logprob = actual.log_softmax(dim=-1)
    reference_logprob = reference.log_softmax(dim=-1)
    actual_topk = actual.topk(topk).indices.tolist()
    reference_topk = reference.topk(topk).indices.tolist()
    actual_norm, reference_norm = actual.norm(), reference.norm()
    norm = actual_norm * reference_norm
    cosine = float(torch.dot(actual, reference) / norm) if float(norm) > 0 else None
    relative_l2 = float(delta.norm() / reference_norm.clamp_min(1e-12))
    kl = float((reference_logprob.exp() * (reference_logprob - actual_logprob)).sum())
    reference_top2 = reference.topk(min(2, reference.numel())).values
    reference_margin = float(reference_top2[0] - reference_top2[1]) if len(reference_top2) == 2 else None
    return dict(max_abs_error=float(absolute_error[worst_token]), mean_abs_error=float(absolute_error.mean()),
                p99_abs_error=float(absolute_error.quantile(0.99)), max_abs_error_token_id=worst_token,
                worst_token_actual_logit=float(actual[worst_token]),
                worst_token_reference_logit=float(reference[worst_token]),
                worst_token_signed_error=float(delta[worst_token]),
                rmse=float(delta.square().mean().sqrt()), relative_l2=relative_l2,
                actual_l2_norm=float(actual_norm), reference_l2_norm=float(reference_norm),
                reference_near_zero_norm=bool(reference_norm < 1e-12),
                cosine_similarity=cosine, logprob_max_abs_error=float((actual_logprob - reference_logprob).abs().max()),
                kl_reference_to_actual=max(0.0, kl),
                top1_match=int(actual.argmax()) == int(reference.argmax()),
                actual_top1=int(actual.argmax()), reference_top1=int(reference.argmax()),
                reference_top1_top2_margin=reference_margin,
                actual_topk=actual_topk, reference_topk=reference_topk,
                topk_overlap=len(set(actual_topk).intersection(reference_topk)) / topk,
                trace_token_id=trace_token, actual_trace_logprob=float(actual_logprob[trace_token]),
                reference_trace_logprob=float(reference_logprob[trace_token]))


def summarize(rows):
    summary = dict(steps=len(rows), top1_agreement=sum(row["top1_match"] for row in rows) / len(rows),
                   mean_topk_overlap=sum(row["topk_overlap"] for row in rows) / len(rows),
                   max_abs_error=max(row["max_abs_error"] for row in rows),
                   max_logprob_abs_error=max(row["logprob_max_abs_error"] for row in rows),
                   reference_near_zero_norm_steps=sum(row["reference_near_zero_norm"] for row in rows))
    for metric in ("rmse", "relative_l2", "p99_abs_error"):
        values = torch.tensor([row[metric] for row in rows], dtype=torch.float64)
        summary[metric] = dict(mean=float(values.mean()), p95=float(values.quantile(0.95)), max=float(values.max()))
    for metric in ("rmse", "relative_l2", "max_abs_error"):
        worst = max(rows, key=lambda row: row[metric])
        summary["worst_" + metric] = {key: worst[key] for key in (
            "case_id", "phase", "step", "context_length", "max_abs_error_token_id",
            "worst_token_actual_logit", "worst_token_reference_logit", "worst_token_signed_error",
        )}
        summary["worst_" + metric]["value"] = worst[metric]
    return summary


def compare_captures(actual_path, reference_path, *, topk=10, rtol=None, atol=None,
                     max_error_ratio=0.0, require_top1=False):
    return _compare_captures(actual_path, reference_path, actual_backend="pypto", reference_backend="vllm-ascend",
                             topk=topk, rtol=rtol, atol=atol, max_error_ratio=max_error_ratio,
                             require_top1=require_top1)


def measure_repeatability(first_path, second_path, *, backend, topk=10):
    if backend not in ("pypto", "vllm-ascend"):
        raise ValueError("Repeatability requires the pypto or vllm-ascend backend")
    if capture_manifest_path(first_path).resolve() == capture_manifest_path(second_path).resolve():
        raise ValueError("Repeatability requires two independent captures, not the same manifest")
    report = _compare_captures(first_path, second_path, actual_backend=backend, reference_backend=backend,
                               topk=topk)
    report.update(comparison="repeatability", backend=backend)
    return report


def _compare_captures(actual_path, reference_path, *, actual_backend, reference_backend,
                      topk=10, rtol=None, atol=None, max_error_ratio=0.0, require_top1=False):
    if (rtol is None) != (atol is None):
        raise ValueError("Supply both --rtol and --atol, or omit both for measurement only")
    for name, value in (("rtol", rtol), ("atol", atol), ("max_error_ratio", max_error_ratio)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if max_error_ratio > 1 or (rtol is None and max_error_ratio != 0):
        raise ValueError("An outlier ratio in [0,1] requires explicit tolerances")
    actual_root, actual = load_capture(actual_path, actual_backend)
    reference_root, reference = load_capture(reference_path, reference_backend)
    for key in ("case_spec_sha256", "checkpoint", "checkpoint_identity", "vocab_size", "chunk_size"):
        if not actual.get(key) or actual[key] != reference.get(key):
            raise ValueError(f"Capture {key} differs; the comparison would use different inputs or weights")
    vocabulary = actual["vocab_size"]
    if not 1 <= topk <= vocabulary:
        raise ValueError("TopK must be between 1 and the checkpoint vocabulary size")
    if [case["id"] for case in actual["cases"]] != [case["id"] for case in reference["cases"]]:
        raise ValueError("Capture case IDs or order differ")
    comparator = None
    if rtol is not None:
        from golden import ratio_allclose

        comparator = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=max_error_ratio)
    rows, passed = [], True
    for observed, expected in zip(actual["cases"], reference["cases"]):
        prompt, trace = expected["prompt_token_ids"], expected["trace_token_ids"]
        contexts = [prompt + trace[:step] for step in range(len(trace))]
        phases = ["prefill"] + ["decode"] * (len(trace) - 1)
        for case in (observed, expected):
            if (case["prompt_token_ids"] != prompt or case["trace_token_ids"] != trace
                    or case.get("contexts") != contexts or case.get("phases") != phases):
                raise ValueError(f"{case['id']}: prompt, Decode history, position, or phase differs")
        a = load_logits(actual_root, observed, vocabulary)
        e = load_logits(reference_root, expected, vocabulary)
        for step, (a_row, e_row) in enumerate(zip(a, e)):
            metrics = step_metrics(a_row, e_row, trace[step], topk)
            metrics.update(case_id=expected["id"], step=step, phase=phases[step], context_length=len(contexts[step]))
            if comparator is not None:
                ok, detail = comparator(a_row, e_row, actual_outputs={}, expected_outputs={}, inputs={},
                                        rtol=rtol, atol=atol)
                metrics.update(logits_pass=ok, logits_failure=detail)
                passed = passed and ok
            if require_top1:
                passed = passed and metrics["top1_match"]
            rows.append(metrics)
    gated = comparator is not None or require_top1
    by_phase = {phase: summarize([row for row in rows if row["phase"] == phase])
                for phase in ("prefill", "decode") if any(row["phase"] == phase for row in rows)}
    by_case = {}
    for case in actual["cases"]:
        case_rows = [row for row in rows if row["case_id"] == case["id"]]
        phases = {phase: summarize([row for row in case_rows if row["phase"] == phase])
                  for phase in ("prefill", "decode") if any(row["phase"] == phase for row in case_rows)}
        by_case[case["id"]] = dict(summary=summarize(case_rows), by_phase=phases)
    capture_hashes = {}
    for name, path in (("actual", actual_path), ("reference", reference_path)):
        capture_hashes[name + "_manifest_sha256"] = file_sha256(capture_manifest_path(path))
    return dict(schema_version=1, status=("pass" if passed else "fail") if gated else "measured",
                topology=TOPOLOGY, output_kind="raw_logits", case_spec_sha256=actual["case_spec_sha256"],
                checkpoint=actual["checkpoint"], checkpoint_identity=actual["checkpoint_identity"],
                chunk_size=actual["chunk_size"],
                captures=capture_hashes,
                policy=dict(rtol=rtol, atol=atol, max_error_ratio=max_error_ratio,
                            require_top1=require_top1, topk=topk),
                summary=summarize(rows), by_phase=by_phase, by_case=by_case, steps=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze checkpoint identity and explicit prompt/trace IDs")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--cases", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    pypto = commands.add_parser("pypto", help="Capture existing Prefill/Decode with one persistent session per case")
    pypto.add_argument("--cases", type=Path, required=True)
    pypto.add_argument("--output", type=Path, required=True)
    pypto.add_argument("--chunk-size", type=int, choices=range(1, MAX_PREFILL_CHUNK_SIZE + 1), default=16)
    pypto.add_argument("--graph-dir", type=Path, help="Reuse compiled A3 prefill/ and decode/ graphs")
    pypto.add_argument("--check-config", action="store_true",
                       help="Validate cases without loading weights or using NPU")
    compare = commands.add_parser("compare", help="Compare raw full-vocabulary logits on CPU")
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--topk", type=int, default=10)
    compare.add_argument("--rtol", type=float)
    compare.add_argument("--atol", type=float)
    compare.add_argument("--max-error-ratio", type=float, default=0.0)
    compare.add_argument("--require-top1", action="store_true")
    repeat = commands.add_parser("repeatability", help="Measure two same-backend captures without accuracy gates")
    repeat.add_argument("--first", type=Path, required=True)
    repeat.add_argument("--second", type=Path, required=True)
    repeat.add_argument("--backend", choices=("pypto", "vllm-ascend"), required=True)
    repeat.add_argument("--output", type=Path, required=True)
    repeat.add_argument("--topk", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            spec = prepare_cases(args.checkpoint, args.cases, args.output)
            print(json.dumps(dict(status="prepared", topology=TOPOLOGY, cases=len(spec["cases"]),
                                  output=str(args.output))))
        elif args.command == "pypto":
            spec = load_cases(args.cases)
            if args.check_config:
                print(json.dumps(dict(status="config_checked", topology=TOPOLOGY, cases=spec["cases"], npu_run=False)))
            else:
                result = capture_pypto(spec, args.cases, args.output, chunk_size=args.chunk_size, graph_dir=args.graph_dir)
                print(json.dumps(dict(status=result["status"], accuracy="not_compared", output=str(args.output))))
        else:
            if args.output.exists():
                raise FileExistsError("Comparison report already exists; use a new output path")
            if args.command == "repeatability":
                result = measure_repeatability(args.first, args.second, backend=args.backend, topk=args.topk)
            else:
                result = compare_captures(args.actual, args.reference, topk=args.topk, rtol=args.rtol, atol=args.atol,
                                          max_error_ratio=args.max_error_ratio, require_top1=args.require_top1)
            write_json(args.output, result)
            print(json.dumps(dict(status=result["status"], **result["summary"], output=str(args.output))))
            return 1 if result["status"] == "fail" else 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
