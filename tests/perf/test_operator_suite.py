# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise allocation, raw evidence and failure reporting without a device."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.perf.operator_contract import (
    ContractError, ROOT, SAMPLING, case_arguments, devices, load_manifest,
    load_profile, validate_allocation, write_json,
)
from tools.perf.operator_results import benchmark_payload, comparison, validate_result
from tools.perf import run_operator_suite as runner


@pytest.fixture
def manifest():
    return load_manifest()


@pytest.fixture
def profile():
    value = load_profile(ROOT / "tools/perf/suites/even_devices.example.json")
    value["device_epoch"] = "test-host-epoch-1"
    return value


def raw_result(case, *, offset=0):
    return {
        "case_id": case["case_id"], "passed": True, "sampling": SAMPLING,
        "run_config": case["run_config"],
        "fixture_sha256": "a" * 64, "golden_sha256": "b" * 64,
        "specs": [{"name": name, **spec, "direction": "out"}
                  for name, spec in case["required_specs"].items()],
        "benchmark": {
            "rounds": 100, "warmup": 5, "fallback_flattened": False,
            "unstable_dispatch_slots": False, "all_zero_device": False,
            "distributed_grid": case["device_count"] > 1,
            "samples": [{str(pid): [
                {"inv": round_id * 2, "task": "compute", "effective_us": 100 + pid + offset},
                {"inv": round_id * 2 + 1, "task": "clear", "effective_us": 7},
            ] for pid in range(10, 10 + case["device_count"])} for round_id in range(100)],
        },
    }


def test_manifest_and_fixed_workloads(manifest, profile):
    assert len(manifest["cases"]) == 10
    for case in manifest["cases"]:
        selected = profile["sequences"][str(case["device_count"])]
        arguments = case_arguments(case, selected)
        assert arguments[arguments.index("-d") + 1] == ",".join(map(str, selected))
        if case["case_id"].startswith("dspark-attention"):
            assert arguments[arguments.index("--start-pos") + 1].split(",") == ["256"] * 16
            assert case["required_specs"]["x_out"]["shape"] == [4, 128, 4, 4096]
        if case["case_id"] == "mtp-moe-ep8":
            assert arguments[arguments.index("--experts-per-rank") + 1] == "16"
        if case["case_id"] == "dspark-moe-ep8":
            assert "--tp" not in arguments
        if case["case_id"].endswith("lm-head"):
            assert arguments[arguments.index("--entry") + 1] == "projection"


@pytest.mark.parametrize("value", ["0,1,2,3", "0,2,2,4", "2,0,4,6", "", "auto", [-2], [True], None])
def test_invalid_devices(value):
    with pytest.raises(ContractError):
        devices(value)


def test_allocation_subsets_preserve_task_device(profile):
    allocation = profile["sequences"]["8"]
    env = {"TASKQUEUE_INSIDE": "1", "TASK_DEVICE": ",".join(map(str, allocation))}
    before = env.copy()
    for selected in profile["sequences"].values():
        validate_allocation(profile, allocation, selected, env)
    assert env == before
    for key, value in (("TASK_DEVICE", "0,2,4,6"), ("TASKQUEUE_INSIDE", "0"),
                       ("ASCEND_RT_VISIBLE_DEVICES", "0,2,4,6,8,10,12,14")):
        with pytest.raises(ContractError):
            validate_allocation(profile, allocation, [4], {**env, key: value})
    with pytest.raises(ContractError):
        validate_allocation(profile, allocation, [0], env)


def test_dry_run_without_site_packages():
    command = [sys.executable, "-S", str(ROOT / "tools/perf/run_operator_suite.py"),
               "--device-profile", str(ROOT / "tools/perf/suites/even_devices.example.json"), "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert len(plan["cases"]) == 10
    assert plan["queue_prefix"] == ["task-submit", "--device", "0,2,4,6,8,10,12,14"]
    assert "--device-num" not in plan["queue_prefix"]


def test_raw_metric_sums_cleanup_then_takes_minimum_rank_median(manifest):
    case = next(case for case in manifest["cases"] if case["case_id"] == "mtp-moe-ep8")
    payload = raw_result(case)
    result = validate_result(payload, case)
    assert result["metric_us"] == 117
    assert result["max_rank_median_us"] == 124
    assert result["rank_spread_us"] == 7
    assert len(result["ranks"]) == 8
    assert len(result["ranks"][0]["dispatches"]) == 2
    assert all(len(rank["samples_us"]) == 100 for rank in result["ranks"])


@pytest.mark.parametrize("defect", ["failed", "round", "rank", "slot", "duplicate", "nan", "zero", "bool",
                                   "shape", "direction", "no_fingerprint", "flat", "unstable", "no_grid"])
def test_reject_invalid_evidence(manifest, defect):
    case = manifest["cases"][3]
    payload = raw_result(case)
    bench = payload["benchmark"]
    if defect == "failed":
        payload["passed"] = False
    elif defect == "round":
        bench["samples"].pop()
    elif defect == "rank":
        bench["samples"][50].pop("10")
    elif defect == "slot":
        bench["samples"][50]["10"][0]["task"] = "other"
    elif defect == "duplicate":
        bench["samples"][50]["10"][0]["inv"] = 0
    elif defect in ("nan", "zero", "bool"):
        bench["samples"][50]["10"][0]["effective_us"] = {"nan": float("nan"), "zero": 0, "bool": True}[defect]
    elif defect == "shape":
        payload["specs"][0]["shape"] = [1]
    elif defect == "direction":
        payload["specs"][0]["direction"] = "in"
    elif defect == "no_fingerprint":
        payload.pop("fixture_sha256")
    elif defect == "flat":
        bench["fallback_flattened"] = True
    elif defect == "unstable":
        bench["unstable_dispatch_slots"] = True
    elif defect == "no_grid":
        bench["distributed_grid"] = False
    with pytest.raises(ContractError):
        validate_result(payload, case)


def test_nonzero_process_rejects_even_a_passing_payload(manifest):
    case = manifest["cases"][0]
    with pytest.raises(ContractError):
        validate_result(raw_result(case), case, process_rc=1)


def test_public_benchmark_grid_preserves_invocations():
    item = SimpleNamespace(pid=9, inv=5, task="abc", task_name="operator", effective_us=12.25)
    stats = SimpleNamespace(rounds=1, warmup=5, rounds_dispatches=[{9: [item]}], invocations=[item],
                            fallback_flattened=False, unstable_dispatch_slots=False, all_zero_device=False)
    payload = benchmark_payload(stats)
    assert payload["samples"] == [{"9": [{"inv": 5, "task": "abc", "task_name": "operator", "effective_us": 12.25}]}]
    assert payload["distributed_grid"] is True
    stats.rounds_dispatches = []
    assert benchmark_payload(stats)["distributed_grid"] is False


def test_comparison_identity_and_sign():
    baseline = {"status": "pass", "metric_us": 100, "run_id": "prior", "case_contract": "v1",
                "device_identity": {"epoch": 1}, "toolchain": {"pypto": "a"}, "source_sha": "lib1",
                "fixture_sha256": "input", "golden_sha256": "output"}
    current = {**baseline, "metric_us": 110, "source_sha": "lib2"}
    assert comparison(current, baseline)["delta_pct"] == pytest.approx(10)
    for key in ("case_contract", "device_identity", "toolchain", "fixture_sha256", "golden_sha256"):
        assert comparison({**current, key: "different"}, baseline)["delta_pct"] is None
    assert comparison(current, baseline, mode="toolchain")["delta_pct"] is None
    assert comparison({**current, "source_sha": "lib1", "toolchain": "new"}, baseline,
                      mode="toolchain")["delta_pct"] == pytest.approx(10)
    assert comparison(current, None)["delta_pct"] is None


@pytest.mark.parametrize("failure", [None, "precision", "device"])
def test_runner_partial_results_and_device_fault_stop(tmp_path, monkeypatch, manifest, profile, failure):
    monkeypatch.setenv("TASKQUEUE_INSIDE", "1")
    monkeypatch.setenv("TASK_DEVICE", "0,2,4,6,8,10,12,14")
    for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner, "source_identity", lambda: "source")
    monkeypatch.setattr(runner, "capture_toolchain", lambda: {"pypto": "test"})
    monkeypatch.setattr(runner, "capture_host", lambda *args: {"hostname": "test-host", "epoch": 1})
    calls = []

    def process(command, log, cwd, environ, timeout):
        case_id = command[command.index("--case") + 1]
        case = next(case for case in manifest["cases"] if case["case_id"] == case_id)
        calls.append(case_id)
        assert environ["TASK_DEVICE"] == "0,2,4,6,8,10,12,14"
        assert environ["PYPTO_BENCH_ROUNDS"] == "100"
        raw = raw_result(case)
        fail = len(calls) == 2 and failure
        raw["passed"] = not fail
        log.write_text("SDMA error 507034" if fail and failure == "device" else "correctness result")
        write_json(Path(command[command.index("--output") + 1]), raw)
        return (1 if fail else 0), False

    monkeypatch.setattr(runner, "run_process", process)
    args = SimpleNamespace(output_dir=tmp_path / "run", date="2026-09-14", variant="candidate",
                           model=None, case=None, baseline=None, week_baseline=None, case_timeout=1,
                           keep_builds=False, save_data=False)
    rc = runner.execute(args, manifest, profile)
    suite = json.loads((args.output_dir / "suite-result.json").read_text())
    assert len(suite["cases"]) == 10
    assert len(calls) == (2 if failure == "device" else 10)
    assert suite["coverage_complete"] is (failure is None)
    assert rc == (0 if failure is None else 1)
    if failure:
        assert suite["cases"][0]["status"] == "pass"
        assert suite["cases"][1]["status"] == "fail"
        assert "metric_us" not in suite["cases"][1]
        assert suite["cases"][2]["status"] == ("not_run" if failure == "device" else "pass")
    with pytest.raises(FileExistsError):
        runner.execute(args, manifest, profile)


def test_baseline_dates(tmp_path):
    path = tmp_path / "baseline.json"
    write_json(path, {"logical_date": "2026-09-07", "cases": []})
    assert runner.load_baseline(path, "2026-09-14", weekly=True) == {}
    with pytest.raises(ContractError):
        runner.load_baseline(path, "2026-09-13", weekly=True)
    with pytest.raises(ContractError):
        runner.load_baseline(path, "2026-09-07")


def test_timeout_is_reported_without_another_attempt(tmp_path):
    rc, timed_out = runner.run_process([sys.executable, "-c", "import time; time.sleep(10)"],
                                      tmp_path / "log", tmp_path, os.environ.copy(), 0.1)
    assert (rc, timed_out) == (124, True)
