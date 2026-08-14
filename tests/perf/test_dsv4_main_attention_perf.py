# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free tests for the DeepSeek-V4 main-attention performance contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.perf.dsv4_main_attention_metrics import (
    MetricParseError,
    assemble_suite_result,
    build_case_result,
    load_manifest,
    parse_case_log,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tools/perf/suites/dsv4_main_attention.json"
PARSER_PATH = REPO_ROOT / "tools/perf/dsv4_main_attention_metrics.py"
RUNNER_PATH = REPO_ROOT / "tools/perf/run_dsv4_main_attention_perf.sh"
LAUNCHER_PATH = REPO_ROOT / "tools/perf/deterministic_run.py"


@pytest.fixture
def manifest() -> dict:
    return load_manifest(MANIFEST_PATH)


def _valid_log(*, case_id: str = "attention-csa", samples: list[float] | None = None) -> str:
    samples = samples or [float(value) for value in range(1, 101)]
    minimum = min(samples)
    ordered = sorted(samples)
    median = (ordered[49] + ordered[50]) / 2
    mean = sum(samples) / len(samples)
    maximum = max(samples)
    threshold = "0.004" if case_id == "attention-csa" else "0.003"
    return "\n".join(
        (
            "[PERF] deterministic_seed seed=1807 python_hash_seed=1807",
            "[RUN] compile ...",
            "[RUN] compile done (1.00s)",
            "[RUN] benchmark ...",
            "[RUN] benchmark done (1.00s)",
            f"[RUN]   effective_us (100 rounds) min={minimum:.1f} "
            f"median={median:.1f} mean={mean:.1f} max={maximum:.1f}",
            "[RUN]   raw samples: ranks=1 rounds=100 warmup=5",
            f"[RUN]     rank 1234 raw n=100 eff_us={samples!r}",
            "[RUN] validate ...",
            "[RUN]   'kv_cache' PASS  shape=(128, 128, 1, 512) dtype=torch.bfloat16 "
            "(ratio_allclose(atol=0.0001, rtol=0.0078125))",
            "[RUN]   'x_out' PASS  shape=(8, 4, 4096) dtype=torch.float32 "
            f"(ratio_reldiff(diff_thd={threshold}, pct_thd=0.008, max_diff_hd=1))",
            "[RUN] validate done (0.05s)",
            "[RUN] PASS (4.00s)",
        )
    )


def test_manifest_pins_official_contract() -> None:
    manifest = load_manifest(MANIFEST_PATH)

    assert manifest["contract_id"] == "dsv4-main-attention-v1"
    assert manifest["ordered_devices"] == [4]
    assert manifest["sampling"] == {"seed": 1807, "rounds": 100, "warmup": 5, "raw": True}
    assert [case["case_id"] for case in manifest["cases"]] == [
        "attention-csa",
        "attention-hca",
        "attention-swa",
    ]
    assert all(case["metric"]["aggregation"] == "median" for case in manifest["cases"])


@pytest.mark.parametrize("case_id", ["attention-csa", "attention-hca", "attention-swa"])
def test_parse_case_log_accepts_complete_contract(manifest: dict, case_id: str) -> None:
    result = parse_case_log(
        _valid_log(case_id=case_id),
        manifest=manifest,
        case_id=case_id,
        device_id=4,
    )

    assert result["status"] == "pass"
    assert result["metric_us"] == 50.5
    assert result["metric"]["aggregation"] == "median"
    assert result["metric"]["sample_count"] == 100
    assert result["validation"]["status"] == "pass"
    assert [output["name"] for output in result["validation"]["outputs"]] == [
        "kv_cache",
        "x_out",
    ]
    assert result["rank_diagnostics"][0]["device_id"] == 4
    assert len(result["rank_diagnostics"][0]["samples_us"]) == 100


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (lambda log: log.replace("seed=1807", "seed=9", 1), "deterministic seed"),
        (lambda log: log.replace("rounds=100 warmup=5", "rounds=99 warmup=5"), "header mismatch"),
        (lambda log: log.replace("median=50.5", "median=60.5"), "median mismatch"),
        (lambda log: log.replace("'x_out' PASS", "'x_out' FAIL"), "validation failed"),
        (lambda log: log.replace("shape=(8, 4, 4096)", "shape=(8, 4, 1)"), "shape mismatch"),
        (lambda log: log.replace("dtype=torch.float32", "dtype=torch.float16"), "dtype mismatch"),
        (lambda log: log.replace("[RUN] PASS (4.00s)", ""), "final PASS"),
        (
            lambda log: log + "\n[RUN] benchmark unavailable: late failure",
            "forbidden marker",
        ),
        (lambda log: log + "\n[RUN] late diagnostic", r"last \[RUN\] line"),
    ],
)
def test_parse_case_log_rejects_invalid_or_truncated_logs(
    manifest: dict,
    replacement,
    match: str,
) -> None:
    with pytest.raises(MetricParseError, match=match):
        parse_case_log(
            replacement(_valid_log()),
            manifest=manifest,
            case_id="attention-csa",
            device_id=4,
        )


def test_parse_case_log_rejects_wrong_device_and_process_failure(manifest: dict) -> None:
    with pytest.raises(MetricParseError, match="ordered devices"):
        parse_case_log(
            _valid_log(),
            manifest=manifest,
            case_id="attention-csa",
            device_id=2,
        )
    with pytest.raises(MetricParseError, match="exited with status 7"):
        parse_case_log(
            _valid_log(),
            manifest=manifest,
            case_id="attention-csa",
            device_id=4,
            process_rc=7,
        )


def test_build_case_result_preserves_explicit_failure(manifest: dict) -> None:
    result, error = build_case_result(
        _valid_log().replace("[RUN] PASS (4.00s)", ""),
        manifest=manifest,
        case_id="attention-csa",
        device_id=4,
        process_rc=0,
    )

    assert error is not None
    assert result["status"] == "invalid_metric"
    assert result["metric"] is None
    assert result["validation"]["status"] == "fail"


def test_suite_requires_complete_case_coverage(manifest: dict, tmp_path: Path) -> None:
    case = parse_case_log(
        _valid_log(),
        manifest=manifest,
        case_id="attention-csa",
        device_id=4,
    )
    result = assemble_suite_result(
        manifest=manifest,
        repo_root=REPO_ROOT,
        device_id=4,
        started_at_utc="2026-08-13T00:00:00+00:00",
        finished_at_utc="2026-08-13T00:01:00+00:00",
        cases=[case],
    )

    assert result["status"] == "fail"
    assert result["coverage"]["complete"] is False
    assert result["source"]["commit"]
    assert "status_porcelain" in result["source"]
    assert result["device"]["ordered_devices"] == [4]
    assert result["device"]["physical_mapping"] == []
    assert result["device"]["physical_mapping_source"] == "unavailable"
    assert set(result["toolchain"]["host"]) == {"hostname", "kernel", "architecture"}


def test_suite_uses_explicit_ordered_physical_mapping(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = parse_case_log(
        _valid_log(), manifest=manifest, case_id="attention-csa", device_id=4
    )
    mapping = [
        {
            "logical_rank": 0,
            "requested_device_id": 4,
            "physical_device_id": 104,
            "serial": "synthetic-card-4",
        }
    ]
    monkeypatch.setenv("PYPTO_DEVICE_MAPPING_JSON", json.dumps(mapping))
    result = assemble_suite_result(
        manifest=manifest,
        repo_root=REPO_ROOT,
        device_id=4,
        started_at_utc="2026-08-13T00:00:00+00:00",
        finished_at_utc="2026-08-13T00:01:00+00:00",
        cases=[case],
    )

    assert result["device"]["physical_mapping"] == mapping
    assert result["device"]["physical_mapping_source"] == "PYPTO_DEVICE_MAPPING_JSON"


def test_suite_rejects_mismatched_physical_mapping(
    manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = parse_case_log(
        _valid_log(), manifest=manifest, case_id="attention-csa", device_id=4
    )
    monkeypatch.setenv(
        "PYPTO_DEVICE_MAPPING_JSON",
        json.dumps(
            [
                {
                    "logical_rank": 0,
                    "requested_device_id": 2,
                    "physical_device_id": 102,
                    "serial": "wrong-card",
                }
            ]
        ),
    )
    with pytest.raises(MetricParseError, match="requested_device_id=4"):
        assemble_suite_result(
            manifest=manifest,
            repo_root=REPO_ROOT,
            device_id=4,
            started_at_utc="2026-08-13T00:00:00+00:00",
            finished_at_utc="2026-08-13T00:01:00+00:00",
            cases=[case],
        )


def test_parse_cli_writes_failure_case_and_crash_safe_journal(tmp_path: Path) -> None:
    log_path = tmp_path / "case.log"
    output_path = tmp_path / "case.json"
    journal_path = tmp_path / "case-results.jsonl"
    log_path.write_text(_valid_log().replace("[RUN] PASS (4.00s)", ""), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PARSER_PATH),
            "parse",
            "--manifest",
            str(MANIFEST_PATH),
            "--case",
            "attention-csa",
            "--log",
            str(log_path),
            "--device",
            "4",
            "--process-rc",
            "0",
            "--output",
            str(output_path),
            "--journal",
            str(journal_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    case_result = json.loads(output_path.read_text(encoding="utf-8"))
    journal_result = json.loads(journal_path.read_text(encoding="utf-8"))
    assert case_result == journal_result
    assert case_result["status"] == "invalid_metric"


def test_deterministic_launcher_replays_python_numpy_and_torch(tmp_path: Path) -> None:
    entrypoint = tmp_path / "random_values.py"
    entrypoint.write_text(
        "import random\n"
        "import numpy as np\n"
        "import torch\n"
        "print(random.random(), np.random.rand(), torch.rand(1).item())\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "1807"
    command = [sys.executable, str(LAUNCHER_PATH), "--seed", "1807", str(entrypoint)]

    first = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)
    second = subprocess.run(command, env=environment, capture_output=True, text=True, check=True)

    assert first.stdout == second.stdout
    assert first.stdout.startswith(
        "[PERF] deterministic_seed seed=1807 python_hash_seed=1807\n"
    )


def test_runner_dry_run_enforces_card_four_without_device_work() -> None:
    rejected = subprocess.run(
        [str(RUNNER_PATH), "--device", "2", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    accepted = subprocess.run(
        [str(RUNNER_PATH), "--device", "4", "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert rejected.returncode == 2
    assert "require --device 4" in rejected.stderr
    assert accepted.returncode == 0
    assert "suite-result.json" in accepted.stdout
    assert accepted.stdout.count("PYTHONHASHSEED=1807") == 3


def test_runner_pins_the_archived_repo_before_inherited_pythonpath() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in runner
    assert "export PYTHONDONTWRITEBYTECODE=1" in runner
