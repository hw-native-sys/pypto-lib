# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the actual CI queue wrapper and artifact history without devices."""

from datetime import date
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from urllib.error import HTTPError
import zipfile

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import ci_operator_perf as ci
from run_operator_suite import ContractError, ROOT, load_manifest, write_json


REPO = "example/lib"


@pytest.mark.parametrize("script,arguments", [
    ("run_operator_case.py", ["--help"]),
    ("ci_operator_perf.py", ["--finalize"]),
])
def test_ci_scripts_start_outside_checkout_without_site_packages(tmp_path, script, arguments):
    env = {"RUNNER_TEMP": str(tmp_path), "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1"}
    subprocess.run([sys.executable, "-S", str(ROOT / ".github/scripts" / script), *arguments],
                   cwd=tmp_path, env=env, capture_output=True, text=True, check=True)


def run(run_id=1, day=13, attempt=1, **changes):
    return {"id": run_id, "run_attempt": attempt, "created_at": f"2026-09-{day:02d}T01:00:00Z",
            "head_branch": "main", "head_sha": f"commit{run_id}", "event": "schedule",
            "path": ci.WORKFLOW, "status": "completed", "conclusion": "failure", **changes}


def suite(record):
    cases = [{"case_id": case["case_id"], "devices": [4], "status": "pass", "metric_us": 123}
             for case in load_manifest()["cases"]]
    return {"ci": {"repository": REPO, "workflow_run_id": str(record["id"]),
                   "run_attempt": str(record["run_attempt"]), "head_sha": record["head_sha"],
                   "head_branch": "main", "workflow_path": ci.WORKFLOW},
            "logical_date": ci.logical_date(record["created_at"]).isoformat(), "variant": "ci",
            "suite_id": "dsv4-operators", "run_id": "unique", "status": "pass",
            "coverage_complete": True, "cases": cases}


def archive(value):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as output:
        output.writestr("suite-result.json", json.dumps(value))
    return buf.getvalue()


def test_dates_and_selection_keep_failed_latest_attempt():
    assert ci.logical_date("2026-09-13T20:51:00Z") == date(2026, 9, 14)
    rows = [run(1, 7), run(2, 12), run(3, 13), run(3, 13, 2), run(4, 14),
            run(5, 13, head_branch="feature"), run(6, 13, event="pull_request")]
    selected = ci.select_runs(rows, date(2026, 9, 14))
    assert selected["previous"]["id"] == 3
    assert selected["previous"]["run_attempt"] == 2
    assert selected["week"]["id"] == 1


@pytest.mark.parametrize("defect", ["attempt", "sha", "date", "variant", "repository"])
def test_artifact_identity_rejects_stale_or_foreign_data(defect):
    record = run()
    data = suite(record)
    if defect in ("date", "variant"):
        data["logical_date" if defect == "date" else "variant"] = "wrong"
    else:
        data["ci"][{"attempt": "run_attempt", "sha": "head_sha", "repository": "repository"}[defect]] = "wrong"
    with pytest.raises(ContractError):
        ci.suite_from_archive(archive(data), record, REPO)


@pytest.mark.parametrize("failure", [None, "missing", "running", "expired", "bad_zip"])
def test_history_never_substitutes_an_older_success(tmp_path, failure):
    records = [run(1, 12), run(2, 13, 2, status="in_progress" if failure == "running" else "completed")]
    class Client:
        def pages(self, path, key):
            if key == "workflow_runs":
                return records
            assert "/runs/2/" in path
            return [] if failure == "missing" else [{"name": "operator-perf-2", "id": 22,
                                                      "expired": failure == "expired"}]
        def archive(self, artifact_id):
            assert artifact_id == 22
            return b"broken" if failure == "bad_zip" else archive(suite(records[-1]))
    selected = ci.retrieve_history(Client(), REPO, date(2026, 9, 14), tmp_path)
    assert selected["previous"]["run_id"] == 2
    assert bool(selected["previous"]["path"]) is (failure is None)
    assert selected["week"]["path"] is None
    if failure is None:
        assert json.loads((tmp_path / "previous.json").read_text())["ci"]["run_attempt"] == "2"


def test_artifact_redirect_does_not_forward_token(monkeypatch):
    client = ci.GitHub(REPO, "private-token")
    def request(path):
        raise HTTPError("https://api.github.com", 302, "redirect", {"Location": "https://blob.example/file?signature=abc"}, None)
    monkeypatch.setattr(client, "request", request)
    def download(url, timeout):
        assert isinstance(url, str) and "private-token" not in url
        assert timeout == 45
        return io.BytesIO(b"archive")
    monkeypatch.setattr(ci, "urlopen", download)
    assert client.archive(1) == b"archive"


@pytest.mark.parametrize("publish_failure", [False, True])
@pytest.mark.parametrize("suite_rc", [0, 1])
def test_ci_wrapper_runs_once_publishes_small_files_and_drops_token(tmp_path, monkeypatch, capsys, suite_rc, publish_failure):
    checkout = tmp_path / "checkout with spaces"
    monkeypatch.setattr(ci, "ROOT", checkout)
    current = run(9, 14)
    class Client:
        def __init__(self, *args):
            pass
        def json(self, path):
            return current
        def pages(self, *args):
            return []
    monkeypatch.setattr(ci, "GitHub", Client)
    values = {"GITHUB_RUN_ID": "9", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_REPOSITORY": REPO,
              "GITHUB_SHA": "commit9", "GITHUB_REF_NAME": "main", "RUNNER_TEMP": str(tmp_path),
              "GH_TOKEN": "private-token", "GITHUB_TOKEN": "another-token", "PYPTO_SRC": "/source with spaces",
              "PTOAS_ROOT": "/assembler", "PTO_ISA_COMMIT": "isa",
              "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"), "OPERATOR_PERF_HOST": socket.gethostname()}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    calls = []
    def process(command, **kwargs):
        calls.append(command)
        assert command[:2] == [sys.executable, str(checkout / ".github/scripts/run_operator_suite.py")]
        assert "task-submit" not in command
        assert command[command.index("--case-timeout") + 1] == "1800"
        assert command[command.index("--queue-timeout") + 1] == "3600"
        assert command[command.index("--suite-timeout") + 1] == "10800"
        assert "GH_TOKEN" not in kwargs["env"] and "GITHUB_TOKEN" not in kwargs["env"]
        work = tmp_path / "operator-perf-9-1/work"
        work.mkdir()
        write_json(work / "suite-result.json", suite(current))
        case = work / "mtp-attention-csa-8k"
        (case / "build_output").mkdir(parents=True)
        (case / "build_output/weights.bin").write_bytes(b"must not upload")
        (case / "run.log").write_text("case log")
        return SimpleNamespace(returncode=suite_rc)
    monkeypatch.setattr(ci.subprocess, "run", process)
    original_copy = ci.shutil.copyfile
    if publish_failure:
        def fail_copy(*args, **kwargs):
            raise OSError("temporary publication failure")
        monkeypatch.setattr(ci.shutil, "copyfile", fail_copy)
    expected_rc = 1 if publish_failure else suite_rc
    assert ci.main() == expected_rc
    assert len(calls) == 1
    console = capsys.readouterr().out
    assert "[perf] Starting suite" in console and f"Suite exited {suite_rc}" in console
    assert "private-token" not in console and "another-token" not in console
    monkeypatch.setattr(ci.shutil, "copyfile", original_copy)
    ci.finalize({**values, "PERF_STEP_OUTCOME": "failure" if expected_rc else "success"})
    published = tmp_path / "operator-perf-9-1/published"
    output = json.loads((published / "suite-result.json").read_text())
    assert output["coverage_complete"] is True
    assert output["execution"]["suite_exit_code"] == suite_rc
    if publish_failure:
        assert output["execution"]["errors"] == [{"phase": "publish", "error": "temporary publication failure"}]
    assert (published / "mtp-attention-csa-8k/run.log").is_file()
    assert not list(published.rglob("weights.bin"))
    assert "Operator performance" in (tmp_path / "summary.md").read_text()
    assert all(case["metric_us"] == 123 and case["status"] == "pass" for case in output["cases"])


def test_workflow_has_isolated_ci_environment_queue_budget_and_artifact_gates():
    workflow = yaml.safe_load((ROOT / ".github/workflows/daily_ci.yml").read_text())
    job = workflow["jobs"]["operator-performance"]
    steps = job["steps"]
    setup = next(step for step in steps if step.get("uses", "").endswith("setup-ci-job"))
    assert setup["with"]["toolchain"] == "bundle"
    assert setup["with"]["checkout-path"] == "operator-checkout"
    assert setup["with"]["build-namespace"] == "operator-performance-a2a3"
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    assert job["concurrency"]["cancel-in-progress"] is False
    assert job["timeout-minutes"] * 60 > 3600 + 10800
    execute = next(step for step in steps if step.get("id") == "perf")
    assert "source activate.sh" in execute["run"]
    assert "python .github/scripts/ci_operator_perf.py" in execute["run"]
    upload = next(step for step in steps if step.get("id") == "upload")
    assert "steps.perf.outcome != 'skipped'" in upload["if"]
    assert upload["with"]["path"].endswith("/published")
    assert "operator-performance" not in workflow["jobs"]["summary"]["needs"]
    assert job["runs-on"] == ["self-hosted", "linux", "arm64", "npu"]
    host_check = steps[0]
    assert "OPERATOR_PERF_HOST" in host_check["run"]
    assert "socket.gethostname()" in host_check["run"]
    assert steps.index(host_check) < steps.index(setup)


def test_manual_performance_run_skips_other_jobs_and_keeps_schedule_default():
    workflow = yaml.safe_load((ROOT / ".github/workflows/daily_ci.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    setting = triggers["workflow_dispatch"]["inputs"]["performance_only"]
    assert setting["type"] == "boolean" and setting["default"] is False
    assert triggers["schedule"]
    for name, job in workflow["jobs"].items():
        if name == "operator-performance":
            assert "if" not in job and "needs" not in job
        else:
            assert "!inputs.performance_only" in job["if"]
    assert "always()" in workflow["jobs"]["summary"]["if"]


@pytest.mark.parametrize("expected", ["", socket.gethostname(), "wrong-performance-host"])
def test_workflow_records_assigned_host_and_rejects_explicit_mismatch(tmp_path, expected):
    workflow = yaml.safe_load((ROOT / ".github/workflows/daily_ci.yml").read_text())
    script = workflow["jobs"]["operator-performance"]["steps"][0]["run"]
    github_env = tmp_path / "github-env"
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                            env={**os.environ, "OPERATOR_PERF_HOST": expected,
                                 "GITHUB_ENV": str(github_env)})
    if expected and expected != socket.gethostname():
        assert result.returncode != 0
        assert "does not match OPERATOR_PERF_HOST" in result.stderr
        assert not github_env.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert github_env.read_text() == f"OPERATOR_PERF_HOST={socket.gethostname()}\n"


def test_finalize_salvages_cancelled_checkpoint(tmp_path):
    root = tmp_path / "operator-perf-1-1"
    published = root / "published"
    published.mkdir(parents=True)
    (root / "work").mkdir()
    data = suite(run())
    data.update(status="running", coverage_complete=False)
    data["cases"][1].update(status="running")
    write_json(root / "work/suite-result.json", data)
    write_json(published / "submission.json", {"ci": data["ci"]})
    write_json(published / "history-selection.json", {})
    assert ci.finalize({"RUNNER_TEMP": str(tmp_path), "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1",
                        "PERF_STEP_OUTCOME": "cancelled"}) == 0
    result = json.loads((published / "suite-result.json").read_text())
    assert result["status"] == "interrupted"
    assert result["cases"][0]["status"] == "pass"
    assert "**interrupted**" in (published / "report.md").read_text()


def test_invalid_suite_structure_is_a_history_error():
    for data in ([], {}, {"cases": []}):
        with pytest.raises(ContractError):
            ci.suite_from_archive(archive(data), run(), REPO)


def test_history_rejects_the_previous_ten_case_suite():
    record = run()
    data = suite(record)
    data["cases"] = [case for case in data["cases"] if case["case_id"].endswith("-8k")]
    for case in data["cases"]:
        case["case_id"] = case["case_id"].removesuffix("-8k")
    assert len(data["cases"]) == 10
    with pytest.raises(ContractError, match="twenty official cases"):
        ci.suite_from_archive(archive(data), record, REPO)


@pytest.mark.parametrize("failure_phase", ["publish", "suite"])
def test_finalize_preserves_passing_measurements_after_wrapper_failure(tmp_path, failure_phase):
    root = tmp_path / "operator-perf-1-1"
    published, work = root / "published", root / "work"
    published.mkdir(parents=True)
    work.mkdir()
    data = suite(run())
    write_json(work / "suite-result.json", data)
    write_json(published / "submission.json", {"ci": data["ci"]})
    write_json(published / "history-selection.json", {})
    write_json(published / "execution.json", {
        "suite_exit_code": 0 if failure_phase == "publish" else 7,
        "errors": [{"phase": failure_phase, "error": "test failure"}],
    })
    summary = tmp_path / "summary.md"
    assert ci.finalize({"RUNNER_TEMP": str(tmp_path), "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1",
                        "PERF_STEP_OUTCOME": "failure", "GITHUB_STEP_SUMMARY": str(summary)}) == 0
    result = json.loads((published / "suite-result.json").read_text())
    assert result["status"] == "pass" and result["coverage_complete"] is True
    assert result["cases"] == data["cases"]
    assert result["execution"]["step_outcome"] == "failure"
    assert f"{failure_phase}: test failure" in summary.read_text()
    assert "queue exited 1" not in summary.read_text()


def test_missing_hostname_is_rejected_before_submission(tmp_path, monkeypatch):
    monkeypatch.setenv("OPERATOR_PERF_HOST", "")
    monkeypatch.setattr(ci, "validate_host", __import__("run_operator_suite").validate_host)
    with pytest.raises(ContractError, match="OPERATOR_PERF_HOST"):
        ci.validate_host({})


def test_ci_timeout_overrides_are_positive_and_explicit(tmp_path):
    env = {"PERF_LOGICAL_DATE": "2026-09-15", "OPERATOR_PERF_CASE_TIMEOUT": "600",
           "OPERATOR_PERF_QUEUE_TIMEOUT": "1200", "OPERATOR_PERF_SUITE_TIMEOUT": "7200"}
    command = ci.suite_command(tmp_path / "profile", tmp_path / "work", {}, env)
    assert command[command.index("--case-timeout") + 1] == "600"
    assert command[command.index("--queue-timeout") + 1] == "1200"
    assert command[command.index("--suite-timeout") + 1] == "7200"
    with pytest.raises(ContractError):
        ci.suite_command(tmp_path / "profile", tmp_path / "work", {}, {**env, "OPERATOR_PERF_CASE_TIMEOUT": "0"})


def test_progress_distinguishes_queue_worker_and_completed_unique_measurements(tmp_path):
    assert "waiting for suite checkpoint" in ci.progress_snapshot(tmp_path)
    case = {"case_id": "mtp-attention-csa-8k", "status": "running", "devices": [4]}
    data = {"status": "running", "cases": [case, {"case_id": "mtp-moe-ep8-8k", "status": "pass"},
                                              {"case_id": "mtp-moe-ep8-128k", "status": "pass",
                                               "shared_measurement": True}]}
    write_json(tmp_path / "suite-result.json", data)
    queued = ci.progress_snapshot(tmp_path)
    assert "1/2 measurements completed" in queued
    assert "case=mtp-attention-csa-8k devices=[4]" in queued
    assert "queue/worker startup" in queued and "worker=running" not in queued
    case_dir = tmp_path / case["case_id"]
    case_dir.mkdir()
    write_json(case_dir / "measurement.json", {"status": "running"})
    (case_dir / "run.log").write_text("[RUN] compile done\n[RUN] compute golden ...\n")
    running = ci.progress_snapshot(tmp_path)
    assert "worker=running" in running and "last stage: [RUN] compute golden ..." in running
    case["status"] = "fail"
    data["status"] = "incomplete"
    write_json(tmp_path / "suite-result.json", data)
    finished = ci.progress_snapshot(tmp_path)
    assert "2/2 measurements completed; suite=incomplete" in finished
    assert "case=" not in finished


def test_progress_does_not_fail_the_workload_on_unreadable_checkpoint(tmp_path):
    (tmp_path / "suite-result.json").write_text("incomplete JSON")
    assert "progress checkpoint unavailable" in ci.progress_snapshot(tmp_path)
