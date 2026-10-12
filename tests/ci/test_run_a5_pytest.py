# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for TP1 smoke filtering and complete daily A5 coverage."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/run_a5_pytest.py"
SPEC = importlib.util.spec_from_file_location("run_a5_pytest", SCRIPT)
run_a5_pytest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_a5_pytest)


def test_tp1_collection_retains_dp_ep_and_unparametrized_cases(monkeypatch, tmp_path):
    (tmp_path / "conftest.py").write_text(
        (ROOT / "models/deepseek_v4_1_flash/conftest.py").read_text()
    )
    monkeypatch.setattr(run_a5_pytest, "ROOT", tmp_path)
    entry = tmp_path / "test_matrix.py"
    entry.write_text('''import pytest

@pytest.mark.parametrize("tp,dp", [(1, 1), (1, 2), (2, 2), (4, 1)])
def test_attention(tp, dp):
    pass

@pytest.mark.parametrize("tp,ep", [(1, 2), (2, 4), (4, 4)])
def test_moe(tp, ep):
    pass

def test_unparametrized():
    pass
''')
    output = tmp_path / "cases.json"
    all_cases = run_a5_pytest.collect(str(entry), output)
    smoke = run_a5_pytest.collect(str(entry), output, tp1_only=True)
    assert len(all_cases) == 8
    assert [case["axes"] for case in smoke] == [
        {"tp": 1, "dp": 1}, {"tp": 1, "dp": 2}, {"tp": 1, "ep": 2}, {},
    ]
    assert [case["cards"] for case in smoke] == [1, 2, 2, 1]
    assert all(case in all_cases for case in smoke)


def test_tp1_collection_fails_when_an_entry_has_no_tp1_case(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        (tmp_path / "cases.json").write_text(json.dumps([
            {"nodeid": "model.py::test_precision[4]", "axes": {"tp": 4}, "cards": 4},
        ]))
        return subprocess.CompletedProcess(args[0], 0, "")

    monkeypatch.setattr(run_a5_pytest.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="no TP1 pytest cases collected"):
        run_a5_pytest.collect("model.py", tmp_path / "cases.json", tp1_only=True)


def test_tp1_script_collection_preserves_complete_daily_matrix(monkeypatch, tmp_path):
    entry = tmp_path / "model.py"
    entry.write_text("")
    matrix = [
        {"cards": 2, "arguments": ["--tp", "1", "--ep", "2"]},
        {"cards": 4, "arguments": ["--tp", "2", "--ep", "4"]},
        {"cards": 4, "arguments": ["--tp", "4", "--ep", "4"]},
    ]
    monkeypatch.setattr(run_a5_pytest, "V41_ROOT", tmp_path)

    def fake_run(command, **kwargs):
        Path(command[-1]).write_text(json.dumps(matrix))

    monkeypatch.setattr(run_a5_pytest.subprocess, "run", fake_run)
    daily = run_a5_pytest.script_cases(str(entry))
    smoke = run_a5_pytest.script_cases(str(entry), tp1_only=True)
    assert len(daily) == 3
    assert smoke == daily[:1]
    assert smoke[0]["command"][3:6] == ["--device-num", "2", "--ignore-whitelist"]


def test_pr_card_limit_leaves_large_worlds_in_daily(monkeypatch, tmp_path):
    entry = "models/deepseek_v4_1_flash/decode_layer.py"
    cases = [
        {"nodeid": f"ep{cards}", "cards": cards, "command": [f"ep{cards}"]}
        for cards in (2, 4, 8)
    ]
    submitted = []
    monkeypatch.setattr(run_a5_pytest, "has_pytest_entry", lambda path: False)
    monkeypatch.setattr(run_a5_pytest, "script_cases", lambda *args, **kwargs: cases)

    def fake_run(command, **kwargs):
        submitted.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run_a5_pytest.subprocess, "run", fake_run)
    output = tmp_path / "results.tsv"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), entry, "--tp1-only", "--max-cards", "2",
                                      "--results", str(output)])
    assert run_a5_pytest.main() == 0
    assert submitted == [["ep2"]]
    assert output.read_text() == "ep2\tpass\nep4\tdaily-only\nep8\tdaily-only\n"


@pytest.mark.parametrize("axes,cards", [({}, 1), ({"tp": 1, "dp": 2}, 2), ({"tp": 1, "ep": 2}, 2)])
def test_queue_passes_tp1_before_model_import(axes, cards):
    command = run_a5_pytest.queue_command({"nodeid": "model.py::test_precision", "axes": axes,
                                          "cards": cards})
    assert command[:3] == ["task-submit", "--device", "auto"]
    assert "--tp 1" in command[-1]
    assert '--device "$TASK_DEVICE"' in command[-1]
    if cards > 1:
        assert command[3:6] == ["--device-num", str(cards), "--ignore-whitelist"]


def test_device_matrix_retains_multicard_cases_and_has_tp1_for_every_entry():
    model = ROOT / "models/deepseek_v4_1_flash"
    count = smoke_count = 0
    entries = [ROOT / entry for entry in run_a5_pytest.all_entries("a5", root="models")
               if (ROOT / entry).parent == model and run_a5_pytest.has_pytest_entry(entry)]
    assert entries
    for path in entries:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            parameters = [
                decorator for decorator in node.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
            ]
            if not parameters:
                count += 1
                smoke_count += 1
                continue
            names = ast.literal_eval(parameters[0].args[0]).split(",")
            values = ast.literal_eval(parameters[0].args[1])
            axes = [dict(zip(names, value if isinstance(value, tuple) else (value,))) for value in values]
            tp1_cases = [case for case in axes if case.get("tp", 1) == 1]
            assert tp1_cases, f"{path.name}::{node.name} has no TP1 case"
            count += len(axes)
            smoke_count += len(tp1_cases)
    assert count > smoke_count > 0


def test_parallel_cases_overlap_and_report_every_result(monkeypatch, tmp_path, capsys):
    cases = [{"nodeid": label, "cards": 1, "command": [label]}
             for label in ("first", "failure", "third", "fourth")]
    monkeypatch.setattr(run_a5_pytest, "has_pytest_entry", lambda path: False)
    monkeypatch.setattr(run_a5_pytest, "script_cases", lambda *args, **kwargs: cases)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = peak = 0

    def fake_run(command, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        # Neither case can finish until a second case is executing.
        barrier.wait(timeout=5)
        with lock:
            active -= 1
        return subprocess.CompletedProcess(command, int(command == ["failure"]),
                                           stdout=f"output for {command[0]}\n")

    monkeypatch.setattr(run_a5_pytest.subprocess, "run", fake_run)
    results = tmp_path / "results.tsv"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "model.py", "--jobs", "2",
                                      "--results", str(results)])
    assert run_a5_pytest.main() == 1
    assert peak == 2
    assert results.read_text() == "first\tpass\nfailure\tfail\nthird\tpass\nfourth\tpass\n"
    log = capsys.readouterr().out
    for case in cases:
        label = case["nodeid"]
        group = log.split(f"::group::{label}\n", 1)[1].split("::endgroup::", 1)[0]
        assert f"output for {label}\n" in group
        assert "::group::" not in group
