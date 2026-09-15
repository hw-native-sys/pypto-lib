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
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from golden import TensorSpec
from tools.perf.deterministic_run import benchmark_payload, make_capture
from tools.perf.run_operator_suite import (
    ContractError, ROOT, SAMPLING, case_arguments, devices, load_manifest,
    comparison, load_profile, validate_allocation, validate_result, write_json,
)
from tools.perf import run_operator_suite as runner


@pytest.fixture
def manifest():
    return load_manifest()


@pytest.fixture
def profile():
    value = load_profile()
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
               "--dry-run"]
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
    args = SimpleNamespace(output_dir=tmp_path / "run", date="2026-09-14",
                           baseline=None, week_baseline=None, case_timeout=1)
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


def test_projection_host_uses_two_tp4_groups():
    calls = Mock()
    namespace = {
        "WORLD_SIZE": 8, "TEST_TOKENS": 2, "D": 4, "VOCAB_PER_TP": 3,
        "MAX_LOGIT_ROWS": 2, "VOCAB": 12, "GROUP_LOGIT_ROWS": 8, "TP_SIZE": 4, "DONE_VALUE": 1,
        "pl": SimpleNamespace(range=range, BF16="bf16", FP32="fp32", INT32="int32"),
        "pld": SimpleNamespace(alloc_window_buffer=lambda *args: object(), world_size=lambda: 8,
                               window=lambda *args, **kwargs: object()),
        "lm_head_test": calls,
    }
    fn = function_from_source(ROOT / "models/deepseek_v4_flash_mtp/lm_head.py",
                              "l3_lm_head_projection", namespace)
    fn(torch.zeros(8, 2, 4), torch.zeros(8, 3, 4), torch.zeros(8, 2, 12), torch.zeros(8, 2))
    assert calls.call_count == 8
    for rank, call in enumerate(calls.call_args_list):
        assert call.kwargs == {"device": rank}
        assert call.args[-3:] == (rank // 4 * 4, rank % 4, 1)


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
