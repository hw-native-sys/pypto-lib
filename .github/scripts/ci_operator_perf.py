# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the operator suite in Daily CI and retrieve its dated result artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_operator_suite import (
    CASE_IDS, ContractError, ROOT, load_profile, read_json, render_report, validate_host, write_json,
)


ZONE = timezone(timedelta(hours=8))
WORKFLOW = ".github/workflows/daily_ci.yml"
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_RESULT = 16 * 1024 * 1024


def logical_date(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(ZONE).date()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class GitHub:
    def __init__(self, repository, token):
        self.base = f"https://api.github.com/repos/{repository}"
        self.headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"}

    def request(self, path):
        return build_opener(NoRedirect).open(Request(self.base + path, headers=self.headers), timeout=45)

    def json(self, path):
        with self.request(path) as response:
            return json.load(response)

    def archive(self, artifact_id):
        try:
            response = self.request(f"/actions/artifacts/{int(artifact_id)}/zip")
        except HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise
            location = error.headers["Location"]
            if urlparse(location).scheme != "https":
                raise ContractError("artifact download must use HTTPS") from error
            # The signed blob URL needs no GitHub token. Never forward it there.
            response = urlopen(location, timeout=45)
        with response:
            data = response.read(MAX_ARCHIVE + 1)
        if len(data) > MAX_ARCHIVE:
            raise ContractError("historical artifact exceeds the size limit")
        return data

    def pages(self, path, key):
        items = []
        for page in range(1, 5):
            separator = "&" if "?" in path else "?"
            batch = self.json(f"{path}{separator}per_page=100&page={page}")[key]
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise ContractError("history pagination limit reached; no partial history will be selected")


def select_runs(runs, today):
    by_date = {}
    for run in runs:
        if (run.get("head_branch") != "main" or run.get("event") not in ("schedule", "workflow_dispatch")
                or run.get("path") != WORKFLOW):
            continue
        day = logical_date(run["created_at"])
        if day >= today:
            continue
        prior = by_date.get(day)
        if prior is None or (run["id"], run["run_attempt"]) > (prior["id"], prior["run_attempt"]):
            by_date[day] = run
    return {"previous": by_date[max(by_date)] if by_date else None,
            "week": by_date.get(today - timedelta(days=7))}


def suite_from_archive(data, run, repository):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        candidates = [item for item in archive.infolist() if item.filename == "suite-result.json"]
        if len(candidates) != 1 or candidates[0].file_size > MAX_RESULT:
            raise ContractError("artifact must contain one bounded suite-result.json")
        suite = json.loads(archive.read(candidates[0]))
    if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list):
        raise ContractError("artifact has no suite/case structure")
    cases = suite["cases"]
    if (len(cases) != len(CASE_IDS) or any(not isinstance(case, dict) for case in cases) or
            {case.get("case_id") for case in cases} != CASE_IDS):
        raise ContractError("artifact does not describe the twenty official cases")
    expected = {"repository": repository, "workflow_run_id": str(run["id"]),
                "run_attempt": str(run["run_attempt"]), "head_sha": run["head_sha"],
                "head_branch": "main", "workflow_path": WORKFLOW}
    if (suite.get("ci") != expected or suite.get("logical_date") != logical_date(run["created_at"]).isoformat()
            or suite.get("suite_id") != "dsv4-operators" or suite.get("variant") != "ci"):
        raise ContractError("artifact identity does not match its workflow run")
    return suite


def retrieve_history(client, repository, today, destination):
    query = urlencode({"branch": "main", "created": ">=" + (today - timedelta(days=14)).isoformat()})
    runs = client.pages(f"/actions/workflows/daily_ci.yml/runs?{query}", "workflow_runs")
    selected = select_runs(runs, today)
    history = {}
    for name, run in selected.items():
        info = {"run_id": run["id"] if run else None, "path": None}
        history[name] = info
        if run is None:
            info["reason"] = "no workflow run for the selected date"
            continue
        info.update(date=logical_date(run["created_at"]).isoformat(), attempt=run["run_attempt"])
        try:
            if run["status"] != "completed":
                raise ContractError("latest attempt is still running; an older attempt is not substituted")
            artifacts = client.pages(f"/actions/runs/{run['id']}/artifacts", "artifacts")
            matching = [item for item in artifacts if item["name"] == f"operator-perf-{run['run_attempt']}"
                        and not item["expired"]]
            if len(matching) != 1:
                raise ContractError("latest attempt has no unique retained performance artifact")
            suite = suite_from_archive(client.archive(matching[0]["id"]), run, repository)
            path = destination / f"{name}.json"
            write_json(path, suite)
            info["path"] = str(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            info["reason"] = str(error)
    return history


def suite_command(profile_path, output, history, env):
    command = [sys.executable, str(ROOT / ".github/scripts/run_operator_suite.py"),
               "--device-profile", str(profile_path), "--output-dir", str(output),
               "--date", env["PERF_LOGICAL_DATE"]]
    for name, default in (("CASE", 1800), ("QUEUE", 3600), ("SUITE", 10800)):
        value = int(env.get(f"OPERATOR_PERF_{name}_TIMEOUT") or default)
        if value <= 0:
            raise ContractError("performance timeout settings must be positive")
        command += [f"--{name.lower()}-timeout", str(value)]
    for name, option in (("previous", "--baseline"), ("week", "--week-baseline")):
        if history.get(name, {}).get("path"):
            command += [option, history[name]["path"]]
    return command


def publish(output, published, context, history, execution):
    source = output / "suite-result.json"
    if source.is_file():
        suite = read_json(source)
        suite["ci"] = context
        suite["execution"] = execution
        if suite.get("status") == "running":
            suite.update(status="interrupted", coverage_complete=False)
            for case in suite["cases"]:
                if case.get("status") == "running":
                    case.update(status="interrupted", error="wrapper ended before measurement completion")
        write_json(published / "suite-result.json", suite)
        report = render_report(suite)
        # Publish only the small evidence files, never builds or tensor dumps.
        for case in suite["cases"]:
            for name in ("raw-result.json", "measurement.json", "run.log", "queue.log",
                         "recovery-0.txt", "recovery-1.txt"):
                path = output / case["case_id"] / name
                if path.is_file() and not path.is_symlink():
                    target = published / case["case_id"] / name
                    target.parent.mkdir(exist_ok=True)
                    if path.stat().st_size <= MAX_RESULT:
                        shutil.copyfile(path, target)
                    else:
                        report += f"\n- {case['case_id']}/{name}: too large to upload.\n"
        inventory = output / "npu-smi-info.txt"
        if inventory.is_file():
            shutil.copyfile(inventory, published / inventory.name)
        complete = suite.get("status") == "pass" and suite.get("coverage_complete") is True
    else:
        report = "Operator performance failed before producing results.\n"
        for error in execution.get("errors", []):
            report += f"- {error['phase']}: {error['error']}\n"
        complete = False
    report += "\nHistory selection:\n"
    for name, info in history.items():
        report += f"- {name}: run {info.get('run_id')}, {info.get('date')}; {info.get('reason', 'artifact loaded')}\n"
    (published / "report.md").write_text(report)
    return complete


def finalize(env):
    """Salvage checkpoints after a failed/cancelled wrapper, after queue cleanup."""
    root = Path(env["RUNNER_TEMP"]) / f"operator-perf-{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}"
    published = root / "published"
    if (root / "work/suite-result.json").is_file():
        submission = read_json(published / "submission.json")
        history = read_json(published / "history-selection.json")
        state_path = published / "execution.json"
        execution = read_json(state_path) if state_path.is_file() else {}
        execution["step_outcome"] = env.get("PERF_STEP_OUTCOME", "unknown")
        write_json(state_path, execution)
        publish(root / "work", published, submission["ci"], history, execution)
    if (published / "report.md").is_file() and env.get("GITHUB_STEP_SUMMARY"):
        with Path(env["GITHUB_STEP_SUMMARY"]).open("a") as summary:
            summary.write((published / "report.md").read_text())
    return 0


def progress_snapshot(work):
    """Read bounded, existing checkpoints without touching the running workload."""
    try:
        suite = read_json(work / "suite-result.json")
        unique = [case for case in suite["cases"] if not case.get("shared_measurement")]
        completed = sum(case["status"] not in ("not_run", "running") for case in unique)
        message = f"{completed}/{len(unique)} measurements completed; suite={suite['status']}"
        active = next((case for case in unique if case["status"] == "running"), None)
        if active is None:
            return message
        case_dir = work / active["case_id"]
        message += f"; case={active['case_id']} devices={active['devices']}"
        measurement = case_dir / "measurement.json"
        if not measurement.is_file():
            return message + "; waiting for worker checkpoint (queue/worker startup)"
        message += f"; worker={read_json(measurement)['status']}"
        log = case_dir / "run.log"
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                lines = stream.read(8192).decode(errors="replace").splitlines()
            stage = next((line for line in reversed(lines) if line.startswith("[RUN] ")), None)
            if stage:
                message += f"; last stage: {stage[:240]}"
        return message
    except FileNotFoundError:
        return "waiting for suite checkpoint (initialization/preflight)"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f"progress checkpoint unavailable: {type(error).__name__}"


def report_progress(work, stopped):
    while not stopped.is_set():
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[perf] {timestamp} {progress_snapshot(work)}", flush=True)
        stopped.wait(30)


def main():
    env = os.environ.copy()
    run_id, attempt = env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"]
    root = Path(env["RUNNER_TEMP"]) / f"operator-perf-{run_id}-{attempt}"
    published = root / "published"
    published.mkdir(parents=True, exist_ok=False)
    repository = env["GITHUB_REPOSITORY"]
    context = {"repository": repository, "workflow_run_id": run_id, "run_attempt": attempt,
               "head_sha": env["GITHUB_SHA"], "head_branch": env["GITHUB_REF_NAME"], "workflow_path": WORKFLOW}
    history = {}
    execution = {"phase": "setup", "errors": []}
    write_json(published / "execution.json", execution)
    try:
        print(f"[perf] Resolving run {run_id} and historical baselines", flush=True)
        client = GitHub(repository, env["GH_TOKEN"])
        # A logical date belongs to the workflow, even when the queue runs past midnight.
        current = client.json(f"/actions/runs/{run_id}")
        today = logical_date(current["created_at"])
        env["PERF_LOGICAL_DATE"] = today.isoformat()
        (root / "history").mkdir()
        try:
            history = retrieve_history(client, repository, today, root / "history")
        except (OSError, ValueError, KeyError) as error:
            history = {"lookup": {"reason": str(error)}}
        write_json(published / "history-selection.json", history)
        print("[perf] History lookup finished; validating the host and preparing the suite", flush=True)
        profile = load_profile()
        profile["device_epoch"] = env.get("OPERATOR_PERF_DEVICE_EPOCH") or profile["device_epoch"]
        profile["hostname"] = env.get("OPERATOR_PERF_HOST", "")
        validate_host(profile)
        profile_path = root / "device-profile.json"
        write_json(profile_path, profile)
        command = suite_command(profile_path, root / "work", history, env)
        write_json(published / "submission.json", {"command": command, "ci": context, "logical_date": today.isoformat()})
        # Tokens are for artifact retrieval only, not for the queued model processes.
        for key in ("GH_TOKEN", "GITHUB_TOKEN"):
            env.pop(key, None)
        execution["phase"] = "suite"
        write_json(published / "execution.json", execution)
        print(f"[perf] Starting suite on {profile['hostname']}; progress every 30 seconds", flush=True)
        stopped = threading.Event()
        monitor = threading.Thread(target=report_progress, args=(root / "work", stopped), daemon=True)
        monitor.start()
        try:
            with (published / "suite.log").open("w") as log:
                rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
        finally:
            stopped.set()
            monitor.join(timeout=2)
        print(f"[perf] Suite exited {rc}; {progress_snapshot(root / 'work')}; publishing results", flush=True)
        execution.update(suite_exit_code=rc, phase="publish")
        write_json(published / "execution.json", execution)
        complete = publish(root / "work", published, context, history, execution)
        execution["phase"] = "complete"
        write_json(published / "execution.json", execution)
        print(f"[perf] Results published; complete={complete}; suite_exit_code={rc}", flush=True)
        return 0 if rc == 0 and complete else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"[perf] {execution['phase']} failed: {error}", flush=True)
        execution["errors"].append({"phase": execution["phase"], "error": str(error)})
        write_json(published / "execution.json", execution)
        (published / "report.md").write_text(f"Operator performance {execution['phase']} failed: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(finalize(os.environ) if sys.argv[1:] == ["--finalize"] else main())
