# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise allocation, raw evidence and failure reporting without a device."""

import ast
from contextlib import nullcontext
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

from golden import TensorSpec
from run_operator_case import benchmark_payload, make_capture
from run_operator_suite import (
    ContractError, ROOT, SAMPLING, case_arguments, devices, load_manifest,
    comparison, load_profile, validate_allocation, validate_result, write_json,
)
import run_operator_suite as runner


@pytest.fixture
def manifest():
    return load_manifest()


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "test-host")
    value = load_profile()
    value["hostname"] = "test-host"
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
    assert len(manifest["cases"]) == 20
    assert set(profile["sequences"]) == {"1", "8"}
    for case in manifest["cases"]:
        label = case["case_id"].rsplit("-", 1)[1]
        context = {"8k": 8192, "128k": 131072}[label]
        assert case["workload"]["scenario_context"] == context
        selected = profile["sequences"][str(case["device_count"])]
        arguments = case_arguments(case, selected)
        assert arguments[arguments.index("-d") + 1] == ",".join(map(str, selected))
        tokens = 8 if case["model"] == "mtp" else 128
        if "attention" in case["case_id"]:
            batch, sequence = (4, 2) if case["model"] == "mtp" else (16, 8)
            assert case["workload"]["local_batch"] == batch
            assert case["workload"]["sequence"] == sequence
            assert case["workload"]["local_tokens"] == tokens
            assert case["workload"]["tp"] == case["device_count"] == 1
            expected = [str(context)] if case["model"] == "mtp" else [str(context)] * batch
            assert arguments[arguments.index("--start-pos") + 1].split(",") == expected
            assert case["required_specs"]["x_out"]["shape"] == [tokens, 4, 4096]
        elif "moe-ep8" in case["case_id"]:
            assert case["device_count"] == case["workload"]["ep"] == 8
            assert case["workload"]["experts_per_rank"] == 16
            assert case["workload"]["routing"] == "balanced"
            assert arguments[arguments.index("--num-tokens") + 1] == str(tokens)
            assert "--tp" not in arguments and "--start-pos" not in arguments
        else:
            assert case["device_count"] == case["workload"]["tp"] == case["workload"]["dp"] == 1
            for option in ("--tp", "--dp"):
                assert arguments[arguments.index(option) + 1] == "1"
            assert arguments[arguments.index("--entry") + 1] == "projection"
            assert arguments[arguments.index("--num-tokens") + 1] == str(tokens)
            assert case["required_specs"]["logits"]["shape"] == [1, tokens, 129280]
            assert "--start-pos" not in arguments


@pytest.mark.parametrize("value", ["0,1,2,3", "0,2,2,4", "2,0,4,6", "", "auto", [-2], [True], None])
def test_invalid_devices(value):
    with pytest.raises(ContractError):
        devices(value)


def test_allocation_matches_each_case_exactly(profile):
    for allocation in profile["sequences"].values():
        env = {"TASKQUEUE_INSIDE": "1", "TASK_DEVICE": ",".join(map(str, allocation))}
        validate_allocation(profile, allocation, allocation, env)
        for key, value in (("TASK_DEVICE", "0,2,4,6"), ("TASKQUEUE_INSIDE", "0"),
                           ("ASCEND_RT_VISIBLE_DEVICES", "0,2,4,6,8,10,12,14")):
            with pytest.raises(ContractError):
                validate_allocation(profile, allocation, allocation, {**env, key: value})
    with pytest.raises(ContractError):
        validate_allocation(profile, profile["sequences"]["8"], [4], env)


def test_dry_run_without_site_packages(tmp_path):
    command = [sys.executable, "-S", str(ROOT / ".github/scripts/run_operator_suite.py"), "--dry-run"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["report_rows"] == 20
    assert len(plan["cases"]) == 16
    assert sum(len(case["devices"]) == 1 for case in plan["cases"]) == 14
    assert sum(len(case["devices"]) == 8 for case in plan["cases"]) == 2
    for case in plan["cases"]:
        assert case["queue_prefix"] == ["task-submit", "--device", ",".join(map(str, case["devices"]))]
    assert sorted(row for case in plan["cases"] for row in case["report_cases"]) == sorted(runner.CASE_IDS)


@pytest.mark.parametrize("change", ["arguments", "run_config", "workload", "chain", "attention"])
def test_sharing_rejects_different_or_context_dependent_workloads(manifest, change):
    alias = next(case for case in manifest["cases"] if case.get("measurement_case"))
    if change == "arguments":
        alias["arguments"].append("--changed")
    elif change == "run_config":
        alias["run_config"]["ring_heap"] += 1
    elif change == "workload":
        alias["workload"]["experts_per_rank"] += 1
    elif change == "chain":
        alias["measurement_case"] = alias["case_id"]
    else:
        manifest["cases"][0]["measurement_case"] = manifest["cases"][1]["case_id"]
    with pytest.raises(ContractError):
        runner.measurement_groups(manifest)


def test_raw_metric_sums_cleanup_then_takes_minimum_rank_median(manifest):
    case = next(case for case in manifest["cases"] if case["case_id"] == "mtp-moe-ep8-8k")
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


@pytest.mark.parametrize("failure", [None, "precision", "shared_precision", "device", "timeout", "unhealthy_timeout", "cleanup_timeout", "queue"])
def test_runner_partial_results_and_device_fault_stop(tmp_path, monkeypatch, manifest, profile, failure):
    monkeypatch.delenv("TASKQUEUE_INSIDE", raising=False)
    for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    for key in ("PYPTO_SRC", "PTOAS_ROOT", "PTO_ISA_COMMIT"):
        monkeypatch.setenv(key, "/ci path/'quoted'")
    monkeypatch.setenv("GH_TOKEN", "must-not-forward")
    monkeypatch.setattr(runner, "source_identity", lambda: "source")
    monkeypatch.setattr(runner, "capture_toolchain", lambda: {"pypto": "test"})
    monkeypatch.setattr(runner, "capture_host", lambda *args: {"hostname": "test-host", "epoch": 1})
    monkeypatch.setattr(runner, "check_recovery", lambda *args: failure != "unhealthy_timeout")
    calls = []

    def process(command, log, cwd, environ, timeout):
        case_id = command[command.index("--case") + 1]
        case = next(case for case in manifest["cases"] if case["case_id"] == case_id)
        calls.append(case_id)
        assert environ["TASK_DEVICE"] == ",".join(map(str, profile["sequences"][str(case["device_count"])]))
        assert environ["PYPTO_BENCH_ROUNDS"] == "100"
        assert environ["PYPTO_BENCH"] == environ["PYPTO_BENCH_RAW"] == "1"
        raw = raw_result(case)
        fail = (case_id == "mtp-moe-ep8-8k" if failure == "shared_precision" else len(calls) == 2) and failure
        if fail and failure == "cleanup_timeout":
            raise runner.TimeoutCleanupError("process group cleanup failed")
        raw["passed"] = not fail
        log.write_text("SDMA error 507034" if fail and failure == "device" else "correctness result")
        write_json(Path(command[command.index("--output") + 1]), raw)
        return (124 if fail and "timeout" in failure else 1 if fail else 0), bool(fail and "timeout" in failure)

    def queue(command, **kwargs):
        assert command[:2] == ["task-submit", "--device"]
        assert "GH_TOKEN" not in kwargs["env"]
        tokens = shlex.split(command[-1])
        assert "PYPTO_SRC=/ci path/'quoted'" in tokens
        case_id = tokens[tokens.index("--worker-case") + 1]
        worker = SimpleNamespace(worker_case=case_id, output_dir=Path(tokens[tokens.index("--output-dir") + 1]),
                                 case_timeout=1)
        with monkeypatch.context() as child:
            child.setenv("TASKQUEUE_INSIDE", "1")
            child.setenv("TASK_DEVICE", command[2])
            rc = runner.run_worker(worker, manifest, profile)
        return SimpleNamespace(returncode=7 if failure == "queue" and len(calls) == 1 else rc)

    monkeypatch.setattr(runner, "run_process", process)
    monkeypatch.setattr(runner.subprocess, "run", queue)
    args = SimpleNamespace(output_dir=tmp_path / "run", date="2026-09-14", baseline=None,
                           week_baseline=None, case_timeout=1, queue_timeout=600, suite_timeout=10800)
    rc = runner.execute(args, manifest, profile)
    suite = json.loads((args.output_dir / "suite-result.json").read_text())
    assert len(suite["cases"]) == 20
    stops = failure in ("device", "unhealthy_timeout", "cleanup_timeout", "queue")
    assert len(calls) == (1 if failure == "queue" else 2 if stops else 16)
    assert suite["coverage_complete"] is (failure is None)
    assert rc == (0 if failure is None else 1)
    assert suite["cases"][0]["status"] == "pass"
    if failure == "queue":
        assert suite["cases"][0]["metric_us"] > 0
        assert "queue_error" in suite["cases"][0]
    elif failure == "shared_precision":
        assert suite["cases"][3]["status"] == "fail"
        assert "metric_us" not in suite["cases"][3]
    elif failure:
        expected = "device_fault" if failure == "device" else "timeout" if "timeout" in failure else "fail"
        assert suite["cases"][1]["status"] == expected
        assert "metric_us" not in suite["cases"][1]
        assert suite["cases"][2]["status"] == ("not_run" if stops else "pass")
    if not stops:
        records = {case["case_id"]: case for case in suite["cases"]}
        for case in manifest["cases"]:
            if case.get("measurement_case"):
                shared, source = records[case["case_id"]], records[case["measurement_case"]]
                assert shared["shared_measurement"] is True
                for key in ("status", "metric_us", "measurement_id", "fixture_sha256", "golden_sha256", "wall_seconds"):
                    assert shared.get(key) == source.get(key)
                assert shared["case_contract"] != source["case_contract"]
                assert not (args.output_dir / case["case_id"]).exists()
    report = (args.output_dir / "report.md").read_text()
    assert "warmup=5, rounds=100; golden replayed=False" in report
    assert "Rank spread us" in report and "Wall s" in report and "shared:" in report
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


def test_timeout_kills_children_after_leader_exits_and_retains_queue_session(tmp_path):
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    script = (
        "import os,subprocess,sys,time,json; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.PIPE,text=True); "
        "p.stdout.readline(); "
        "print(json.dumps({'group':os.getpgrp(),'session':os.getsid(0)}),flush=True); time.sleep(60)"
    )
    log = tmp_path / "log"
    assert runner.run_process([sys.executable, "-c", script], log, tmp_path, os.environ.copy(), 3) == (124, True)
    identity = json.loads(log.read_text())
    assert identity["session"] == os.getsid(0)
    assert not runner.group_running(identity["group"])


def inventory(health="OK", busy="0"):
    return f"| 2 Ascend910 | {health} | power temp |\n| 0 4 | 0000:95:00.0 | {busy} 0 / 0 |\n"


@pytest.mark.parametrize("text,selected,expected", [
    (inventory(), [4], True), (inventory(), [4, 6], False),
    (inventory("Warning"), [4], False), (inventory(busy="1"), [4], False),
    (inventory(busy="N/A"), [4], False), ("new unknown table", [4], False),
    (inventory() + inventory(), [4], False),
])
def test_recovery_requires_matching_healthy_idle_physical_devices(text, selected, expected):
    assert runner.healthy_idle_inventory(text, selected) is expected


@pytest.mark.parametrize("second", [inventory(), inventory(busy="100"), None])
def test_recovery_requires_two_successful_observations(tmp_path, monkeypatch, second):
    replies = [SimpleNamespace(stdout=inventory()),
               SimpleNamespace(stdout=second) if second is not None else subprocess.TimeoutExpired("npu-smi", 15)]
    mock = Mock(side_effect=replies)
    monkeypatch.setattr(runner.subprocess, "run", mock)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    assert runner.check_recovery([4], tmp_path) is (second == inventory())
    assert mock.call_count == 2
    assert (tmp_path / "recovery-0.txt").is_file()


def test_submission_budget_accounts_for_execution_not_only_queue_wait(tmp_path, manifest, profile):
    args = SimpleNamespace(case_timeout=600, queue_timeout=1200)
    env = {key: "ci-value" for key in ("PYPTO_SRC", "PTOAS_ROOT", "PTO_ISA_COMMIT")}
    command = runner.queue_command(manifest["cases"][0], profile, tmp_path / "profile",
                                   tmp_path / "out", args, env, remaining=1000)
    assert command[command.index("--timeout") + 1] == "970"
    assert command[command.index("--max-time") + 1] == "690"
    assert int(command[command.index("--timeout") + 1]) + 30 <= 1000
    with pytest.raises(ContractError, match="budget exhausted"):
        runner.queue_command(manifest["cases"][0], profile, tmp_path / "profile",
                             tmp_path / "out", args, env, remaining=700)


@pytest.mark.parametrize("kind", ["client_timeout", "missing_result", "budget"])
def test_unknown_queue_completion_or_exhausted_budget_never_submits_another_case(
    tmp_path, monkeypatch, manifest, profile, kind,
):
    monkeypatch.delenv("TASKQUEUE_INSIDE", raising=False)
    for key in ("PYPTO_SRC", "PTOAS_ROOT", "PTO_ISA_COMMIT"):
        monkeypatch.setenv(key, "ci")
    monkeypatch.setattr(runner, "source_identity", lambda: "source")
    monkeypatch.setattr(runner, "capture_toolchain", lambda: {})
    monkeypatch.setattr(runner, "capture_host", lambda *args: {})
    process = Mock(side_effect=subprocess.TimeoutExpired("queue", 1)) if kind == "client_timeout" else Mock(
        return_value=SimpleNamespace(returncode=1))
    monkeypatch.setattr(runner.subprocess, "run", process)
    args = SimpleNamespace(output_dir=tmp_path / "run", date="2026-09-15", baseline=None,
                           week_baseline=None, case_timeout=600, queue_timeout=1200,
                           suite_timeout=700 if kind == "budget" else 7200)
    assert runner.execute(args, manifest, profile) == 1
    assert process.call_count == (0 if kind == "budget" else 1)
    suite = runner.read_json(args.output_dir / "suite-result.json")
    assert suite["cases"][0]["status"] == ("not_run" if kind == "budget" else "queue_failed")
    assert all(case["status"] == "not_run" for case in suite["cases"][1:])
    assert "budget exhausted" in suite["stop_reason"] if kind == "budget" else "queue" in suite["stop_reason"]


def test_ci_history_exposes_stack_changes_but_rejects_system_or_device_changes():
    system = {key: "fixed" for key in ("python", "torch", "numpy", "cann_sha256", "driver_sha256", "bundle")}
    baseline = {"status": "pass", "metric_us": 100, "run_id": "previous", "source_sha": "old-lib",
                "case_contract": "case", "fixture_sha256": "input", "golden_sha256": "golden",
                "device_identity": {"hostname": "host1"}, "toolchain": {**system, "pypto": "old"}}
    current = {**baseline, "metric_us": 110, "source_sha": "new-lib", "toolchain": {**system, "pypto": "new"}}
    result = comparison(current, baseline)
    assert result["delta_pct"] == pytest.approx(10)
    assert result["scope"] == "ci_stack"
    assert result["changed_components"] == ["pypto-lib", "pypto"]
    assert comparison(current, None)["delta_pct"] is None
    assert comparison({**current, "status": "fail"}, baseline)["delta_pct"] is None
    for key in ("case_contract", "fixture_sha256", "golden_sha256"):
        assert comparison({**current, key: "different"}, baseline)["delta_pct"] is None
    for key in system:
        changed = {**current, "toolchain": {**current["toolchain"], key: "other"}}
        assert comparison(changed, baseline)["delta_pct"] is None
    current["device_identity"] = {"hostname": "host2"}
    assert comparison(current, baseline)["delta_pct"] is None


def test_ci_versions_need_no_source_build_or_wheel_metadata(tmp_path, monkeypatch):
    cann = tmp_path / "cann"
    info = cann / "aarch64-linux/ascend_toolkit_install.info"
    info.parent.mkdir(parents=True)
    info.write_text("version=9.0\n")
    driver = tmp_path / "driver.info"
    driver.write_text("driver=v1\n")
    source = tmp_path / "source"
    for key, value in {"PYPTO_SRC": str(source), "PTO_ISA_COMMIT": "isa-revision",
                       "PTOAS_ROOT": str(tmp_path / "assembler"), "ASCEND_HOME_PATH": str(cann),
                       "PYPTO_TOOLCHAIN": "/opt/pypto/toolchain/test-bundle"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(runner, "command", lambda argv, cwd=None:
                        "ptoas 0.61" if cwd is None else "runtime-revision" if cwd.name == "runtime" else "pypto-revision")
    versions = runner.capture_toolchain(driver)
    assert versions["pypto"] == "pypto-revision"
    assert versions["runtime"] == "runtime-revision"
    assert versions["pto_isa"] == "isa-revision"
    assert versions["ptoas_version"] == "ptoas 0.61"
    assert versions["bundle"] == "test-bundle"
    assert versions["torch"] and versions["numpy"]
    assert not source.exists()
    driver.write_text("driver=v2\n")
    assert runner.capture_toolchain(driver)["driver_sha256"] != versions["driver_sha256"]


def function_from_source(path, name, namespace):
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("ep", [2, 4, 8, 16])
@pytest.mark.parametrize("local", [None, 16, 32])
def test_moe_specializes_before_subkernel_imports(ep, local, monkeypatch):
    path = ROOT / "models/deepseek_v4_flash_mtp/decode_moe.py"
    tree = ast.parse(path.read_text())
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.Import) and node.names[0].name.startswith("pypto"):
            break
        prefix.append(node)
    @dataclasses.dataclass(frozen=True)
    class Config:
        n_routed_experts: int = 256
    import sys
    cfg = SimpleNamespace(FLASH=Config(), MOE_TOKENS=8, EP_WORLD_SIZE=8)
    monkeypatch.setitem(sys.modules, "config", cfg)
    args = [str(path), "--ep", str(ep)]
    if local is not None:
        args += [f"--experts-per-rank={local}"]
    monkeypatch.setattr(sys, "argv", args)
    namespace = {}
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec"), namespace)
    assert cfg.EP_WORLD_SIZE == ep
    assert cfg.FLASH.n_routed_experts == ep * (32 if local is None else local)
    assert cfg.RECV_MAX == ep * 8


@pytest.mark.parametrize("tp,dp", [(1, 1), (1, 2), (4, 2)])
def test_projection_host_preserves_tp_groups(tp, dp):
    world = tp * dp
    calls = Mock()
    namespace = {
        "WORLD_SIZE": world, "TEST_TOKENS": 2, "D": 4, "VOCAB_PER_TP": 3,
        "MAX_LOGIT_ROWS": 2, "VOCAB": 3 * tp, "GROUP_LOGIT_ROWS": 2 * tp, "TP_SIZE": tp, "DONE_VALUE": 1,
        "pl": SimpleNamespace(range=range, BF16="bf16", FP32="fp32", INT32="int32"),
        "pld": SimpleNamespace(alloc_window_buffer=lambda *args: object(), world_size=lambda: world,
                               window=lambda *args, **kwargs: object()),
        "lm_head_test": calls,
    }
    fn = function_from_source(ROOT / "models/deepseek_v4_flash_mtp/lm_head.py",
                              "l3_lm_head_projection", namespace)
    fn(torch.zeros(world, 2, 4), torch.zeros(world, 3, 4), torch.zeros(world, 2, 3 * tp), torch.zeros(world, 2))
    assert calls.call_count == world
    for rank, call in enumerate(calls.call_args_list):
        assert call.kwargs == {"device": rank}
        assert call.args[-3:] == (rank // tp * tp, rank % tp, 1)


@pytest.mark.parametrize("active", [0, 1, 8])
def test_tp1_projection_handles_padding_tail_and_retained_windows(active):
    """Execute the kernel's Python body on CPU tensors, without a compiler/runtime."""
    path = ROOT / "models/deepseek_v4_flash_mtp/lm_head.py"
    lane = 0

    class Spmd:
        def __init__(self, count, **_):
            self.count = count
        def __iter__(self):
            return iter(range(self.count))
        def __enter__(self):
            assert self.count == 1
        def __exit__(self, *_):
            return False

    def split_aiv(count, **_):
        nonlocal lane
        for lane in range(count):
            yield lane

    def tile_slice(value, shape, offsets, valid_shape):
        tile = torch.zeros(shape, dtype=value.dtype)
        rows, cols = valid_shape
        row, col = offsets
        tile[:rows, :cols] = value[row:row + rows, col:col + cols]
        return SimpleNamespace(value=tile, valid_shape=valid_shape)

    def put(*, dst, peer, src, dst_offsets, src_offsets, shape):
        assert peer == 0
        dr, dc = dst_offsets
        sr, sc = src_offsets
        rows, cols = shape
        dst[dr:dr + rows, dc:dc + cols] = src[sr:sr + rows, sc:sc + cols]

    def remote_store(tile, dst, peer, offsets):
        assert peer == 0, "TP1 must never publish its padding to a second owner"
        rows, cols = tile.valid_shape
        put(dst=dst, peer=peer, src=tile.value, dst_offsets=offsets, src_offsets=[0, 0], shape=[rows, cols])

    def matmul_acc(acc, lhs, rhs, *, b_trans, init_cond):
        assert b_trans
        product = lhs.float() @ rhs.value.float().T
        return product if init_cond else acc + product

    no_peer = Mock(side_effect=AssertionError("TP1 has no peer to notify or wait for"))
    pl = SimpleNamespace(
        dynamic=lambda _: 1, BF16=torch.bfloat16, FP32=torch.float32, INT32=torch.int32, INDEX=int,
        Level=SimpleNamespace(CORE_GROUP=0), SplitMode=SimpleNamespace(UP_DOWN=0, NONE=1),
        spmd=Spmd, at=lambda **_: nullcontext(), range=range, pipeline=lambda *args, **_: range(*args),
        tensor=SimpleNamespace(dim=lambda value, axis: value.shape[axis]),
        tile=SimpleNamespace(get_block_idx=lambda: 0), min=min, max=max,
        read=lambda value, indexes: value[tuple(indexes)].item(),
        write=lambda value, indexes, item: value.__setitem__(tuple(indexes), item),
        cast=lambda value, *args, **kwargs: value,
        create_tensor=lambda shape, dtype: torch.full(shape, float("nan"), dtype=dtype),
        full=lambda shape, dtype, value: torch.full(shape, value, dtype=dtype),
        slice=tile_slice, matmul_acc=matmul_acc, set_validshape=lambda value, *args: value,
        split_aiv=split_aiv, aiv_shard=lambda value: value.chunk(2, dim=0)[lane],
        cross_core_slot=lambda **_: None,
    )
    namespace = {"sys": SimpleNamespace(argv=[str(path), "--tp", "1"]), "pl": pl,
                 "DECODE_TOKENS": 8, "M": SimpleNamespace(hidden_size=512, vocab_size=320),
                 "pld": SimpleNamespace(tensor=SimpleNamespace(put=put, remote_store=remote_store),
                                         system=SimpleNamespace(notify=no_peer, wait=no_peer))}
    prefix = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == "lm_head":
            break
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            prefix.append(node)
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["MATMUL_ROWS"] == 16 and namespace["GROUP_LOGIT_ROWS"] == 8
    # One logical core visits every vocab tile, including the 64-column tail.
    namespace["FUSED_LM_HEAD_CORES"] = 1
    fn = function_from_source(path, "lm_head", namespace)
    reference = function_from_source(path, "golden_lm_head", namespace)
    generator = torch.Generator().manual_seed(1807)
    weight = torch.randn(320, 512, generator=generator).to(torch.bfloat16)
    indices = torch.tensor([15, -1, 0, 4, 30, 1, 1, 2], dtype=torch.int32)
    indices[active:] = -1
    hidden_window = torch.full((8, 512), float("nan"), dtype=torch.bfloat16)
    logits_window = torch.full((8, 320), float("nan"))
    hidden_done = torch.full((1, 1), 7, dtype=torch.int32)
    logits_done = torch.full((1, 1), 9, dtype=torch.int32)
    for _ in range(2):
        hidden = torch.randn(16, 512, generator=generator).to(torch.bfloat16)
        logits = torch.full((8, 320), float("nan"))
        fn(hidden, weight, indices, logits, hidden_window, hidden_done, logits_window, logits_done, 0, 0, 1)
        values = {"hidden_states": hidden[None], "lm_head_weight": weight[None],
                  "logit_row_indices": indices[None], "logits": torch.zeros(1, 8, 320)}
        reference(values)
        torch.testing.assert_close(logits, values["logits"][0], rtol=1e-3, atol=1e-3)
        assert not hidden_done.any() and not logits_done.any()
    no_peer.assert_not_called()


def test_projection_golden_does_not_require_sampling_inputs():
    sample = Mock(side_effect=AssertionError("sampling must not run"))
    namespace = {"TP_SIZE": 2, "MAX_LOGIT_ROWS": 2, "D": 3, "golden_sample": sample}
    fn = function_from_source(ROOT / "models/deepseek_v4_flash_mtp/lm_head.py", "golden_lm_head", namespace)
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    weight = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    values = {"hidden_states": hidden, "lm_head_weight": weight,
              "logit_row_indices": torch.tensor([[1, -1], [0, 1]]), "logits": torch.zeros(2, 2, 4)}
    fn(values)
    full_weight = weight.reshape(4, 3)
    torch.testing.assert_close(values["logits"][0, 0], hidden[0, 1] @ full_weight.T)
    assert torch.equal(values["logits"][0, 1], torch.zeros(4))
    sample.assert_not_called()


def test_capture_preserves_numerics_and_hashes_actual_inputs(tmp_path):
    case = {"case_id": "small", "required_specs": {"out": {"shape": [2], "dtype": "torch.float32"}}}
    cfg = {"device_id": 4, "platform": "a2a3"}
    compare = {"out": object()}
    def original(**kwargs):
        assert kwargs["rtol"] == 0.001
        assert kwargs["atol"] == 0.002
        assert kwargs["compare_fn"] is compare
        assert kwargs["config"] == cfg
        assert kwargs["save_data"] is False
        specs = kwargs["specs"]
        specs[0].direction = "in"
        specs[1].direction = "out"
        values = {"inp": specs[0].create_tensor(), "out": torch.zeros(2)}
        kwargs["golden_fn"](values)
        torch.testing.assert_close(values["out"], values["inp"] * 2)
        # Failed correctness retains hashes but cannot gain a valid metric.
        return SimpleNamespace(passed=False, error="deliberate", work_dir=None)

    hashes = []
    for value in (1, 1, 2):
        specs = [TensorSpec("inp", [2], torch.float32, init_value=value),
                 TensorSpec("out", [2], torch.float32)]
        output = tmp_path / f"capture-{len(hashes)}.json"
        observer = make_capture(original, case, output, [4])
        kwargs = dict(specs=specs, config=cfg, rtol=0.001, atol=0.002, compare_fn=compare,
                      golden_fn=lambda values: values["out"].copy_(values["inp"] * 2))
        observer(**kwargs)
        payload = json.loads(output.read_text())
        assert payload["passed"] is False and "benchmark" not in payload
        hashes.append(payload["fixture_sha256"])
        with pytest.raises(ContractError):
            observer(**kwargs)
        assert not output.exists()
    assert hashes[0] == hashes[1] != hashes[2]
