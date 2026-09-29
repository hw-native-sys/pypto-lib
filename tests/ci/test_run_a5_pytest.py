# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the queued A5 pytest runner's log and result reporting."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/run_a5_pytest.py"
SPEC = importlib.util.spec_from_file_location("run_a5_pytest", SCRIPT)
run_a5_pytest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_a5_pytest)


def test_collect_suppresses_successful_pytest_output(monkeypatch, tmp_path, capsys):
    cases = [{"nodeid": "model.py::test_precision[2-2]", "axes": {"tp": 2, "dp": 2}, "cards": 4}]
    output = tmp_path / "cases.json"

    def fake_run(*args, **kwargs):
        output.write_text(json.dumps(cases))
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["text"] is True
        return subprocess.CompletedProcess(args[0], 0, "duplicate pytest collection output\n")

    monkeypatch.setattr(run_a5_pytest.subprocess, "run", fake_run)

    assert run_a5_pytest.collect("model.py", output) == cases
    assert capsys.readouterr().out == ""


def test_collect_replays_failed_pytest_output(monkeypatch, tmp_path, capsys):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 4, "collection diagnostic\n")

    monkeypatch.setattr(run_a5_pytest.subprocess, "run", fake_run)

    try:
        run_a5_pytest.collect("model.py", tmp_path / "cases.json")
    except subprocess.CalledProcessError as error:
        assert error.returncode == 4
    else:
        raise AssertionError("collection failure did not raise")
    assert capsys.readouterr().out == "collection diagnostic\n"


def test_main_prints_final_failed_case_summary(monkeypatch, capsys):
    cases = [
        {"nodeid": "model.py::test_precision[1-1]", "axes": {"tp": 1, "dp": 1}, "cards": 1},
        {"nodeid": "model.py::test_precision[2-2]", "axes": {"tp": 2, "dp": 2}, "cards": 4},
    ]
    results = iter([subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 1)])
    monkeypatch.setattr(run_a5_pytest, "collect", lambda entry, output: cases)
    monkeypatch.setattr(run_a5_pytest, "queue_command", lambda case: [case["nodeid"]])
    monkeypatch.setattr(run_a5_pytest.subprocess, "run", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "model.py"])

    assert run_a5_pytest.main() == 1
    output = capsys.readouterr().out
    assert output.count("::group::model.py::test_precision[1-1]") == 1
    assert output.endswith("FAILED CASES:\n  model.py::test_precision[2-2]\n")


def test_pr_device_matrix_has_41_cases():
    model = ROOT / "models/deepseek_v4_1_flash"
    count = 0
    for path in model.glob("*.py"):
        if "# ci: a5" not in path.read_text().splitlines():
            continue
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
            count += len(ast.literal_eval(parameters[0].args[1])) if parameters else 1
    assert count == 41
