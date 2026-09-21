# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks for implementation-local precision tests and queue dispatch."""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/deepseek_v4_1_flash"
ENTRIES = [path for path in sorted(MODEL.glob("*.py")) if "# ci: a5" in path.read_text().splitlines()]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f".github/scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", ENTRIES, ids=lambda path: path.stem)
def test_precision_asserts_golden_and_main_preserves_exit_status(path):
    tree = ast.parse(path.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name in ("test_precision", "main")]
    assert len(functions) == 2
    state = SimpleNamespace(passed=False, error="golden mismatch")
    namespace = {"pytest": pytest, "validate": lambda *args: state}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    test = namespace["test_precision"]
    arguments = {name: 1 for name in test.__code__.co_varnames[:test.__code__.co_argcount]
                 if name != "a5_args"}
    arguments["a5_args"] = lambda **kwargs: ["-p", "a5", "-d", "0"]
    with pytest.raises(AssertionError, match="golden mismatch"):
        test(**arguments)
    with pytest.raises(SystemExit, match="golden mismatch"):
        namespace["main"]()
    state.passed = True
    test(**arguments)
    namespace["main"]()


@pytest.fixture
def sample(tmp_path):
    shutil.copyfile(MODEL / "conftest.py", tmp_path / "conftest.py")
    entry = tmp_path / "precision_case.py"
    entry.write_text(
        'import pytest\n'
        '@pytest.mark.parametrize("tp,dp", [(1, 1), (4, 2)])\n'
        'def test_precision(tp, dp, a5_args):\n'
        '    args = a5_args(tp=tp, dp=dp)\n'
        '    assert args[:2] == ["-p", "a5"]\n'
    )
    return entry


def run_pytest(sample, *args):
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args, "--rootdir", str(sample.parent), "-q"],
        cwd=sample.parent, env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        capture_output=True, text=True, check=False,
    )


def test_real_pytest_collection_exports_only_declared_cases(sample):
    manifest = sample.parent / "cases.json"
    result = run_pytest(sample, str(sample), "--collect-only", "--a5-case-list", str(manifest))
    assert result.returncode == 0, result.stdout + result.stderr
    cases = json.loads(manifest.read_text())
    assert cases == [
        {"nodeid": "precision_case.py::test_precision[1-1]", "axes": {"tp": 1, "dp": 1}, "cards": 1},
        {"nodeid": "precision_case.py::test_precision[4-2]", "axes": {"tp": 4, "dp": 2}, "cards": 8},
    ]


@pytest.mark.parametrize("options,expected", [
    (["--tp", "4", "--dp", "2", "--device", "0,1,2,3,4,5,6,7"], 0),
    (["--tp", "1", "--dp", "2", "--device", "0,1,2,3,4,5,6,7"], 1),
    (["--tp", "4", "--dp", "2", "--device", "0,1"], 1),
    (["--tp", "4", "--dp", "2", "--device", "0,1,2,3,4,5,6,6"], 1),
    (["--tp", "4", "--dp", "2"], 1),
])
def test_pytest_case_checks_process_axes_and_devices(sample, options, expected):
    result = run_pytest(sample, str(sample) + "::test_precision[4-2]", *options)
    assert result.returncode == expected, result.stdout + result.stderr
    assert ("1 passed" if expected == 0 else "1 failed") in result.stdout


def test_queue_runs_pytest_node_with_matching_cards():
    runner = load_script("run_a5_pytest")
    command = runner.queue_command({"nodeid": "precision_case.py::test_precision[4-2]",
                                    "axes": {"tp": 4, "dp": 2}, "cards": 8})
    assert command[:6] == ["task-submit", "--device", "auto", "--device-num", "8", "--ignore-whitelist"]
    assert "python -m pytest 'precision_case.py::test_precision[4-2]'" in command[-1]
    assert '--tp 4 --dp 2 --device "$TASK_DEVICE"' in command[-1]


def test_runner_continues_and_reports_collection_and_precision_failures(monkeypatch, tmp_path):
    runner = load_script("run_a5_pytest")
    report = tmp_path / "results.tsv"
    monkeypatch.setattr(sys, "argv", ["runner", "bad.py", "good.py", "--results", str(report)])

    def collect(entry, output):
        if entry == "bad.py":
            raise ValueError("collection error")
        return [{"nodeid": f"good.py::test_precision[{i}]", "axes": {}, "cards": 1} for i in (1, 2)]

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(runner, "collect", collect)
    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main() == 1
    assert len(calls) == 2
    assert report.read_text().splitlines() == [
        "bad.py\tfail", "good.py::test_precision[1]\tfail", "good.py::test_precision[2]\tpass",
    ]


@pytest.mark.parametrize("changed", [
    "models/deepseek_v4_1_flash/conftest.py",
    ".github/scripts/run_a5_pytest.py",
])
def test_pytest_infrastructure_change_selects_all_a5_entries(monkeypatch, changed):
    monkeypatch.chdir(ROOT)
    detector = load_script("detect_changes")
    assert detector.select_a5([changed]) == [
        str(path.relative_to(ROOT)) for path in ENTRIES
    ]


@pytest.mark.parametrize("filename,runner_name", [
    ("decode_attn_c2a_full.py", "run_c2a"),
    ("decode_attn_c2a_reuse.py", "run_c2a_reuse"),
    ("decode_attn_swa.py", "run_swa"),
])
@pytest.mark.parametrize("argv", [
    ["--tp", "1", "--device", "not-a-device"],
    ["--tp", "4"],
    [],
    None,
])
def test_runner_checks_tp_before_device_setup(filename, runner_name, argv, monkeypatch, capsys):
    path = MODEL / filename
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == runner_name)

    class ReachedExecution(Exception):
        pass

    def stop_before_execution(*args):
        raise ReachedExecution

    namespace = {
        "argparse": argparse,
        "Path": Path,
        "os": os,
        "TP_SIZE": 4,
        "C": SimpleNamespace(EP_SIZE=8, DECODE_MAX_TOKENS=32, PREFILL_MAX_TOKENS=257),
        "MAX_BATCH_PER_DP": 32,
        "torch": SimpleNamespace(set_num_threads=stop_before_execution),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setattr(sys, "argv", [filename])
    if argv and argv[1] == "1":
        with pytest.raises(SystemExit) as error:
            namespace[runner_name](None, "decode", argv)
        assert error.value.code == 2
        assert "--tp 1 does not match import-time TP_SIZE 4" in capsys.readouterr().err
    else:
        with pytest.raises(ReachedExecution):
            namespace[runner_name](None, "decode", argv)


@pytest.fixture
def ep_sample(sample):
    sample.write_text(
        'import pytest\n'
        '@pytest.mark.parametrize("tp,ep", [(2, 4), (4, 4)])\n'
        'def test_precision(tp, ep, a5_args):\n'
        '    args = a5_args(tp=tp, ep=ep)\n'
        '    assert "--ep" in args and "--dp" not in args\n'
    )
    return sample


def test_ep_collection_and_queue_request_ep_cards(ep_sample):
    manifest = ep_sample.parent / "cases.json"
    result = run_pytest(ep_sample, str(ep_sample), "--collect-only", "--a5-case-list", str(manifest))
    assert result.returncode == 0, result.stdout + result.stderr
    cases = json.loads(manifest.read_text())
    assert [case["cards"] for case in cases] == [4, 4]
    assert [case["axes"] for case in cases] == [{"tp": 2, "ep": 4}, {"tp": 4, "ep": 4}]
    runner = load_script("run_a5_pytest")
    for case in cases:
        command = runner.queue_command(case)
        assert command[command.index("--device-num") + 1] == "4"
        assert f'--tp {case["axes"]["tp"]} --ep 4 --device "$TASK_DEVICE"' in command[-1]


@pytest.mark.parametrize("options,expected", [
    (["--tp", "2", "--ep", "4", "--device", "0,1,2,3"], 0),
    (["--tp", "2", "--ep", "4", "--device", "0,1"], 1),
    (["--tp", "2", "--ep", "8", "--device", "0,1,2,3"], 1),
])
def test_ep_fixture_checks_world_size_and_import_options(ep_sample, options, expected):
    result = run_pytest(ep_sample, str(ep_sample) + "::test_precision[2-4]", *options)
    assert result.returncode == expected, result.stdout + result.stderr
    assert ("1 passed" if expected == 0 else "1 failed") in result.stdout


@pytest.mark.parametrize("axes", [
    {"tp": 4, "ep": 2},
    {"tp": 2, "ep": 4, "dp": 1},
    {"tp": 2, "ep": 0},
    {"tp": 2, "ep": True},
])
def test_invalid_ep_topology_rejected_at_collection(sample, axes):
    sample.write_text(
        'import pytest\n'
        f'@pytest.mark.parametrize({",".join(axes)!r}, {[tuple(axes.values())]!r})\n'
        f'def test_precision({", ".join(axes)}):\n'
        '    pass\n'
    )
    result = run_pytest(sample, str(sample), "--collect-only", "--a5-case-list", str(sample.parent / "cases.json"))
    assert result.returncode == 4, result.stdout + result.stderr
    assert "ERROR:" in result.stderr
