# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify the document-aligned DeepSeek-V4 EPLB performance parser."""

from __future__ import annotations

import ast
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dsv4_eplb_perf_metrics import MetricParseError, parse_perf_log


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARSER = _REPO_ROOT / "tools" / "dsv4_eplb_perf_metrics.py"
_RUNNER = _REPO_ROOT / "tools" / "run_dsv4_eplb_perf.sh"
_SUITE_RUNNER = _REPO_ROOT / "tools" / "run_dsv4_eplb_suite.sh"
_SUITE_RESULT_HELPER = _REPO_ROOT / "tools" / "dsv4_eplb_suite_results.py"
_SEED_LAUNCHER = _REPO_ROOT / "tools" / "run_seeded_python.py"
_MOE_SCRIPT = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp" / "moe.py"
_DEVICE_SET = "0,2,4,6,8,10,12,14"
_PIDS = tuple(range(100, 108))
_ROUNDS = 100
_WARMUP = 5
_SEED = 1807


def _make_clean_suite_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "suite-repo"
    tool_names = (
        "dsv4_eplb_perf_metrics.py",
        "dsv4_eplb_suite_results.py",
        "run_dsv4_eplb_perf.sh",
        "run_dsv4_eplb_suite.sh",
        "run_seeded_python.py",
    )
    (repo / "tools").mkdir(parents=True)
    for name in tool_names:
        shutil.copy2(_REPO_ROOT / "tools" / name, repo / "tools" / name)
    model_dir = repo / "models" / "deepseek_v4_flash_mtp"
    model_dir.mkdir(parents=True)
    for name in ("moe.py", "eplb_decode_logits.py", "eplb_mtp_core.py"):
        (model_dir / name).touch()
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "test suite runner"], cwd=repo, check=True)
    return repo


def _physical_mapping_json() -> str:
    return json.dumps(
        [
            {
                "logical_rank": rank,
                "requested_device_id": device,
                "physical_device_id": device,
                "serial": f"test-serial-{device}",
            }
            for rank, device in enumerate((0, 2, 4, 6, 8, 10, 12, 14))
        ]
    )


def _stats_text(samples: list[float]) -> str:
    return (
        f"min={min(samples):.1f} median={statistics.median(samples):.1f} "
        f"mean={statistics.fmean(samples):.1f} max={max(samples):.1f}"
    )


def _rank_summary(pid: int, samples: list[float]) -> str:
    return f"[RUN]     rank {pid}: eff_us {_stats_text(samples)}"


def _slot_summary(slot: int, task: str, samples: list[float]) -> str:
    return f"[RUN]       slot {slot} ({task}): eff_us {_stats_text(samples)}"


def _raw_line(pid: int, samples: list[float]) -> str:
    return f"[RUN]     rank {pid} raw n={len(samples)} eff_us={samples}"


def _write_log(tmp_path: Path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _decode_log(tmp_path: Path) -> Path:
    samples = {pid: [float(pid - 90)] * _ROUNDS for pid in _PIDS}
    samples[100] = [1.0] * (_ROUNDS - 1) + [100.0]
    samples[101] = [2.0] * _ROUNDS
    lines = [f"[RUN] fixture seed={_SEED} python_hash_seed={_SEED}"]
    lines.extend(_rank_summary(pid, samples[pid]) for pid in _PIDS)
    lines.append(f"[RUN]   raw samples: ranks=8 rounds={_ROUNDS} warmup={_WARMUP}")
    lines.extend(_raw_line(pid, samples[pid]) for pid in _PIDS)
    lines.append(
        "[RUN] benchmark kernel=_jit_l3_eplb_decode_logits "
        f"l3_resident=1 rounds={_ROUNDS} ranks=8 host_union_mean_us=100"
    )
    lines.append("[RUN] PASS (1.00s, validation skipped: no golden_fn or golden_data)")
    return _write_log(tmp_path, "decode.log", lines)


def _mtp_log(tmp_path: Path) -> Path:
    lines = [
        f"[RUN] fixture seed={_SEED} python_hash_seed={_SEED}",
        "[RUN]   'kv_cache' PASS  shape=(8,) dtype=torch.bfloat16 (finite_tensor_compare)",
        "[RUN]   'hidden_out' PASS  shape=(8,) dtype=torch.bfloat16 (finite_tensor_compare)",
        "[RUN]   'next_pre_hc_hidden' PASS  shape=(8,) dtype=torch.float32 (finite_tensor_compare)",
        "[RUN]   'logits' PASS  shape=(8,) dtype=torch.float32 (finite_tensor_compare)",
    ]
    raw = {}
    for logical_rank, pid in enumerate(_PIDS):
        compute = [float(logical_rank + 10)] * _ROUNDS
        cleanup = [float(logical_rank + 1)] * _ROUNDS
        if logical_rank == 6:
            compute = [2.0] * _ROUNDS
        if logical_rank == 7:
            compute = [1.0] * (_ROUNDS - 1) + [100.0]
            cleanup = [1000.0] * _ROUNDS
        raw[pid] = [value for pair in zip(compute, cleanup, strict=True) for value in pair]
        combined = [left + right for left, right in zip(compute, cleanup, strict=True)]
        lines.extend(
            [
                _rank_summary(pid, combined),
                _slot_summary(0, "eplb_mtp_core_logits", compute),
                _slot_summary(1, "eplb_mtp_core_cleanup", cleanup),
            ]
        )
    lines.append(f"[RUN]   raw samples: ranks=8 rounds={_ROUNDS} warmup={_WARMUP}")
    lines.extend(_raw_line(pid, raw[pid]) for pid in _PIDS)
    lines.append(
        "[RUN] benchmark kernel=_jit_l3_eplb_mtp_core "
        f"l3_resident=1 rounds={_ROUNDS} ranks=8 host_union_mean_us=100"
    )
    lines.append("[RUN] PASS (1.00s)")
    return _write_log(tmp_path, "mtp.log", lines)


def _moe_log(tmp_path: Path) -> Path:
    samples = {pid: [float(pid - 90)] * _ROUNDS for pid in _PIDS}
    samples[100] = [3.0] * _ROUNDS
    samples[101] = [2.0] * _ROUNDS
    samples[107] = [1.0] * (_ROUNDS - 1) + [100.0]
    lines = [
        f"[RUN] fixture seed={_SEED} python_hash_seed={_SEED}",
        "[RUN]   'x_next' PASS  shape=(8,) dtype=torch.float32 (ratio_reldiff)",
        "[RUN]   effective_us (100 rounds) min=100.0 median=900.0 mean=901.0 max=1000.0",
    ]
    lines.extend(_rank_summary(pid, samples[pid]) for pid in _PIDS)
    lines.append(f"[RUN]   raw samples: ranks=8 rounds={_ROUNDS} warmup={_WARMUP}")
    lines.extend(_raw_line(pid, samples[pid]) for pid in _PIDS)
    lines.append(
        "[RUN] benchmark kernel=_jit_l3_moe "
        f"l3_resident=1 rounds={_ROUNDS} ranks=8 host_union_mean_us=100"
    )
    lines.append("[RUN] PASS (1.00s)")
    return _write_log(tmp_path, "moe.log", lines)


def test_compare3_selects_the_minimum_rank_median_and_its_mean(tmp_path: Path) -> None:
    result = parse_perf_log(
        "decode-logits",
        _decode_log(tmp_path),
        rounds=_ROUNDS,
        warmup=_WARMUP,
        device_set=_DEVICE_SET,
    )

    assert result.metric_scope == "compare3_fastest_rank"
    assert result.selected_rank == 0
    assert result.selected_device == 0
    assert result.selected_pid == 100
    assert result.selected_stats.median == 1.0
    assert result.selected_stats.mean == 1.99
    assert result.cleanup_median_us is None
    assert len(result.rank_metrics) == 8


def test_moe_ep8_selects_the_minimum_rank_median_not_the_top_level_summary(
    tmp_path: Path,
) -> None:
    result = parse_perf_log(
        "moe-ep8",
        _moe_log(tmp_path),
        rounds=_ROUNDS,
        warmup=_WARMUP,
        device_set=_DEVICE_SET,
        seed=_SEED,
    )

    assert result.metric_contract_version == "dsv4-moe-ep8-v1"
    assert result.metric_scope == "moe_ep8_fastest_rank"
    assert result.selected_rank == 7
    assert result.selected_device == 14
    assert result.selected_pid == 107
    assert result.selected_stats.median == 1.0
    assert result.selected_stats.mean == 1.99
    assert result.baseline_median_us is None
    assert result.validation_mode == "numeric_golden"
    assert result.validation_outputs == (("x_next", "PASS"),)


def test_compare4_selects_compute_only_and_reports_cleanup_separately(tmp_path: Path) -> None:
    result = parse_perf_log(
        "mtp-core",
        _mtp_log(tmp_path),
        rounds=_ROUNDS,
        warmup=_WARMUP,
        device_set=_DEVICE_SET,
    )

    assert result.metric_scope == "compare4_fastest_rank_compute_only"
    assert result.selected_rank == 7
    assert result.selected_device == 14
    assert result.selected_pid == 107
    assert result.selected_stats.median == 1.0
    assert result.selected_stats.mean == 1.99
    assert result.cleanup_median_us == 1000.0
    assert len(result.rank_metrics) == 16
    assert [metric.task for metric in result.rank_metrics[-2:]] == [
        "eplb_mtp_core_logits",
        "eplb_mtp_core_cleanup",
    ]


def test_moe_parser_requires_the_seed_contract_in_cli_and_log(tmp_path: Path) -> None:
    log_path = _moe_log(tmp_path)
    with pytest.raises(MetricParseError, match="require seed=1807"):
        parse_perf_log(
            "moe-ep8",
            log_path,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
            seed=7,
        )

    log_path.write_text(
        log_path.read_text(encoding="utf-8").replace(
            "fixture seed=1807 python_hash_seed=1807",
            "fixture seed=1807 python_hash_seed=unset",
        ),
        encoding="utf-8",
    )
    with pytest.raises(MetricParseError, match="fixture-seed mismatch"):
        parse_perf_log(
            "moe-ep8",
            log_path,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
            seed=_SEED,
        )


def test_seed_launcher_replays_python_numpy_and_torch_rngs(tmp_path: Path) -> None:
    probe = tmp_path / "rng_probe.py"
    probe.write_text(
        """import json
import random
import numpy as np
import torch
print(json.dumps([random.random(), np.random.random(), torch.rand(1).item()]))
""",
        encoding="utf-8",
    )
    command = [sys.executable, str(_SEED_LAUNCHER), "--seed", str(_SEED), "--", str(probe)]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(_SEED)

    first = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    second = subprocess.run(command, env=env, capture_output=True, text=True, check=False)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    assert first.stdout.startswith("[RUN] fixture seed=1807 python_hash_seed=1807\n")
    json_values = []
    for line in first.stdout.splitlines():
        try:
            json_values.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    assert len(json_values) == 1

    del env["PYTHONHASHSEED"]
    invalid = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert invalid.returncode == 2
    assert "PYTHONHASHSEED must be 1807" in invalid.stderr


def test_moe_cli_seed_helper_is_deterministic_and_runs_before_fixture_build() -> None:
    source = _MOE_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_seed_fixture_generators"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(_MOE_SCRIPT), "exec"), namespace)
    seed_fn = namespace["_seed_fixture_generators"]

    import random

    import numpy as np
    import torch

    seed_fn(_SEED)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    seed_fn(_SEED)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second
    assert 'parser.add_argument("--seed"' in source
    assert source.index("_seed_fixture_generators(args.seed)") < source.index(
        "specs=build_tensor_specs("
    )


def test_cli_emits_summary_and_rank_tsv_rows(tmp_path: Path) -> None:
    rank_output = tmp_path / "rank-results.tsv"
    result = subprocess.run(
        [
            sys.executable,
            str(_PARSER),
            "--case",
            "mtp-core",
            "--log",
            str(_mtp_log(tmp_path)),
            "--rounds",
            str(_ROUNDS),
            "--warmup",
            str(_WARMUP),
            "--device",
            _DEVICE_SET,
            "--rank-output",
            str(rank_output),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    fields = result.stdout.strip().split("\t")
    assert fields[:3] == [
        "dsv4-eplb-v2",
        "compare4_fastest_rank_compute_only",
        "minimum_rank_median",
    ]
    assert fields[7:10] == ["7", "14", "107"]
    rank_rows = rank_output.read_text(encoding="utf-8").splitlines()
    assert len(rank_rows) == 16
    assert rank_rows[-2].split("\t")[7] == "1"
    assert rank_rows[-1].split("\t")[7] == "0"


def test_measured_runner_writes_valid_case_and_rank_tsvs(tmp_path: Path) -> None:
    decode_log = _decode_log(tmp_path)
    mtp_log = _mtp_log(tmp_path)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

if sys.argv[1:] == ["--version"]:
    print("Python fake-runner")
elif sys.argv[1].endswith("dsv4_eplb_perf_metrics.py"):
    if os.environ.get("FAKE_PARSER_STDOUT_DIAGNOSTIC"):
        print("fake stdout diagnostic", flush=True)
    else:
        print("fake stderr diagnostic", file=sys.stderr, flush=True)
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
elif sys.argv[1].endswith("run_seeded_python.py"):
    target = sys.argv[sys.argv.index("--") + 1]
    if target.endswith("eplb_decode_logits.py"):
        print(Path(os.environ["FAKE_DECODE_LOG"]).read_text(encoding="utf-8"), end="")
    elif target.endswith("eplb_mtp_core.py"):
        print(Path(os.environ["FAKE_MTP_LOG"]).read_text(encoding="utf-8"), end="")
    else:
        raise SystemExit(f"unexpected seeded target: {target}")
elif sys.argv[1].endswith("eplb_decode_logits.py"):
    print(Path(os.environ["FAKE_DECODE_LOG"]).read_text(encoding="utf-8"), end="")
elif sys.argv[1].endswith("eplb_mtp_core.py"):
    print(Path(os.environ["FAKE_MTP_LOG"]).read_text(encoding="utf-8"), end="")
else:
    raise SystemExit(f"unexpected fake-python arguments: {sys.argv[1:]}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_dir = tmp_path / "runner-output"
    env = os.environ.copy()
    env["FAKE_DECODE_LOG"] = str(decode_log)
    env["FAKE_MTP_LOG"] = str(mtp_log)

    result = subprocess.run(
        [
            str(_RUNNER),
            "--device",
            _DEVICE_SET,
            "--case",
            "all",
            "--python",
            str(fake_python),
            "--output-dir",
            str(output_dir),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    results = [line.split("\t") for line in (output_dir / "results.tsv").read_text().splitlines()]
    assert all(len(row) == 25 for row in results)
    header = results[0]
    by_case = {row[0]: dict(zip(header, row, strict=True)) for row in results[1:]}
    assert by_case["decode-logits"]["status"] == "pass"
    assert by_case["decode-logits"]["median_us"] == "1.000"
    assert by_case["mtp-core"]["status"] == "pass"
    assert by_case["mtp-core"]["median_us"] == "1.000"
    assert by_case["mtp-core"]["cleanup_median_us"] == "1000.000"
    rank_results = (output_dir / "rank-results.tsv").read_text().splitlines()
    assert len(rank_results) == 25
    assert all(len(line.split("\t")) == 13 for line in rank_results)

    invalid_output_dir = tmp_path / "invalid-runner-output"
    env["FAKE_PARSER_STDOUT_DIAGNOSTIC"] = "1"
    invalid_result = subprocess.run(
        [
            str(_RUNNER),
            "--device",
            _DEVICE_SET,
            "--case",
            "decode-logits",
            "--python",
            str(fake_python),
            "--output-dir",
            str(invalid_output_dir),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert invalid_result.returncode == 1
    assert "metric parser emitted more than one summary line" in invalid_result.stderr
    invalid_rows = (invalid_output_dir / "results.tsv").read_text().splitlines()
    assert all(len(line.split("\t")) == 25 for line in invalid_rows)
    assert invalid_rows[1].split("\t")[1] == "invalid_metric"


def test_runner_dry_run_uses_the_official_contract() -> None:
    result = subprocess.run(
        [str(_RUNNER), "--device", _DEVICE_SET, "--case", "all", "--dry-run"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Metric contract: dsv4-eplb-v2" in result.stdout
    assert "PYTHONHASHSEED=1807" in result.stdout
    assert "run_seeded_python.py" in result.stdout
    assert "PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_WARMUP=5" in result.stdout
    assert "eplb_decode_logits.py" in result.stdout
    assert "eplb_mtp_core.py" in result.stdout
    assert "--finite-only" in result.stdout


def test_suite_runner_dry_run_freezes_order_seed_and_contracts() -> None:
    result = subprocess.run(
        [str(_SUITE_RUNNER), "--device", _DEVICE_SET, "--dry-run"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Suite contract: dsv4-eplb-suite-v2" in result.stdout
    assert "moe-ep8=dsv4-moe-ep8-v1" in result.stdout
    assert "PYTHONDONTWRITEBYTECODE=1" in result.stdout
    assert "PYTHONHASHSEED=1807" in result.stdout
    assert "--balanced-routing --seed 1807" in result.stdout
    assert result.stdout.index("moe-ep8:") < result.stdout.index("decode-logits:")
    assert result.stdout.index("decode-logits:") < result.stdout.index("mtp-core:")


def test_suite_result_helper_is_location_independent(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_SUITE_RESULT_HELPER), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "create crash-safe machine-readable results" in result.stdout.lower()


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        (None, "PYPTO_DEVICE_MAPPING_JSON is required"),
        ("[]", "must be an ordered list with 8 entries"),
    ],
)
def test_suite_runner_requires_a_valid_physical_mapping(
    tmp_path: Path,
    mapping: str | None,
    error: str,
) -> None:
    env = os.environ.copy()
    if mapping is None:
        env.pop("PYPTO_DEVICE_MAPPING_JSON", None)
    else:
        env["PYPTO_DEVICE_MAPPING_JSON"] = mapping
    result = subprocess.run(
        [
            str(_SUITE_RUNNER),
            "--device",
            _DEVICE_SET,
            "--python",
            sys.executable,
            "--output-dir",
            str(tmp_path / "invalid-mapping-output"),
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert error in result.stderr


def test_suite_finalize_refreshes_and_rejects_dirty_source(tmp_path: Path) -> None:
    suite_repo = _make_clean_suite_repo(tmp_path)
    helper = suite_repo / "tools" / "dsv4_eplb_suite_results.py"
    output_dir = tmp_path / "source-mutation-output"
    context = output_dir / "context.json"
    journal = output_dir / "cases.jsonl"
    suite_output = output_dir / "suite.json"
    env = os.environ.copy()
    env["PYPTO_DEVICE_MAPPING_JSON"] = _physical_mapping_json()
    init_result = subprocess.run(
        [
            sys.executable,
            str(helper),
            "init",
            "--repo",
            str(suite_repo),
            "--output-dir",
            str(output_dir),
            "--context",
            str(context),
            "--suite-output",
            str(suite_output),
            "--device",
            _DEVICE_SET,
            "--rounds",
            "100",
            "--warmup",
            "5",
            "--seed",
            "1807",
            "--ring-task-window",
            "262144",
            "--ring-dep-pool",
            "262144",
            "--ring-heap",
            "2147483648",
            "--started-at",
            "2026-01-01T00:00:00Z",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert init_result.returncode == 0, init_result.stderr
    journal.write_text(
        "".join(
            json.dumps({"case": {"case_id": case_id, "status": "pass"}}) + "\n"
            for case_id in ("moe-ep8", "decode-logits", "mtp-core")
        ),
        encoding="utf-8",
    )
    tracked_source = suite_repo / "tools" / "dsv4_eplb_perf_metrics.py"
    tracked_source.write_text(
        tracked_source.read_text(encoding="utf-8") + "\n# Test source mutation.\n",
        encoding="utf-8",
    )

    finalize_result = subprocess.run(
        [
            sys.executable,
            str(helper),
            "finalize",
            "--repo",
            str(suite_repo),
            "--context",
            str(context),
            "--journal",
            str(journal),
            "--suite-output",
            str(suite_output),
            "--finished-at",
            "2026-01-01T01:00:00Z",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert finalize_result.returncode == 1, finalize_result.stderr
    suite = json.loads(suite_output.read_text(encoding="utf-8"))
    assert suite["status"] == "fail"
    assert suite["source"]["dirty"] is True


def test_suite_runner_writes_crash_safe_structured_results(tmp_path: Path) -> None:
    moe_log = _moe_log(tmp_path)
    decode_log = _decode_log(tmp_path)
    mtp_log = _mtp_log(tmp_path)
    fake_python = tmp_path / "fake-suite-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

if sys.argv[1:] == ["--version"]:
    print("Python fake-suite")
elif sys.argv[1].endswith("dsv4_eplb_suite_results.py"):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
elif sys.argv[1].endswith("dsv4_eplb_perf_metrics.py"):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
elif sys.argv[1].endswith("run_seeded_python.py"):
    target = sys.argv[sys.argv.index("--") + 1]
    if target.endswith("eplb_decode_logits.py"):
        source = "FAKE_DECODE_LOG"
    elif target.endswith("eplb_mtp_core.py"):
        source = "FAKE_MTP_LOG"
    else:
        raise SystemExit(f"unexpected seeded target: {target}")
    print(Path(os.environ[source]).read_text(encoding="utf-8"), end="")
elif sys.argv[1].endswith("moe.py"):
    print(Path(os.environ["FAKE_MOE_LOG"]).read_text(encoding="utf-8"), end="")
else:
    raise SystemExit(f"unexpected fake-suite-python arguments: {sys.argv[1:]}")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    suite_repo = _make_clean_suite_repo(tmp_path)
    suite_runner = suite_repo / "tools" / "run_dsv4_eplb_suite.sh"
    output_dir = tmp_path / "suite-output"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_MOE_LOG": str(moe_log),
            "FAKE_DECODE_LOG": str(decode_log),
            "FAKE_MTP_LOG": str(mtp_log),
            "PYPTO_TOOLCHAIN_EPOCH": "test-toolchain-epoch",
            "PYPTO_DEVICE_EPOCH": "test-device-epoch",
            "PYPTO_DEVICE_MAPPING_JSON": _physical_mapping_json(),
        }
    )

    result = subprocess.run(
        [
            str(suite_runner),
            "--device",
            _DEVICE_SET,
            "--python",
            str(fake_python),
            "--output-dir",
            str(output_dir),
        ],
        cwd=suite_repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in (output_dir / "case-results.jsonl").read_text().splitlines()]
    assert [record["case"]["case_id"] for record in records] == [
        "moe-ep8",
        "decode-logits",
        "mtp-core",
    ]
    assert [record["case"]["status"] for record in records] == ["pass", "pass", "pass"]
    assert [record["case"]["metric"]["summary"]["median_us"] for record in records] == [
        1.0,
        1.0,
        1.0,
    ]
    assert records[0]["case"]["spec_hash"] == "dsv4-moe-ep8-v1"
    assert records[1]["case"]["spec_hash"] == "dsv4-eplb-v2:decode-logits"
    assert records[2]["case"]["spec_hash"] == "dsv4-eplb-v2:mtp-core"
    assert "models/deepseek_v4_flash_mtp/eplb_mtp_core.py" in records[2]["case"]["command"]
    assert "models/deepseek_v4_flash_mtp/eplb_decode_logits.py" not in records[2]["case"]["command"]
    assert records[1]["case"]["validation"]["status"] == "runtime_only"
    assert records[2]["case"]["validation"]["status"] == "pass"

    suite = json.loads((output_dir / "suite-result.json").read_text(encoding="utf-8"))
    assert suite["schema_version"] == 1
    assert suite["suite_id"] == "dsv4-eplb-8card"
    assert suite["suite_contract_id"] == "dsv4-eplb-suite-v2"
    assert suite["lane_id"] == "dsv4-eplb-branch"
    assert suite["status"] == "pass"
    assert suite["coverage"] == {
        "required_case_ids": ["moe-ep8", "decode-logits", "mtp-core"],
        "selected_case_ids": ["moe-ep8", "decode-logits", "mtp-core"],
        "complete": True,
    }
    assert suite["source"]["commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=suite_repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert suite["provenance"]["epoch"] == "test-toolchain-epoch"
    assert suite["devices"]["epoch"] == "test-device-epoch"
    assert suite["devices"]["platform"] == "a2a3"
    assert suite["devices"]["ordered"] == [0, 2, 4, 6, 8, 10, 12, 14]
    assert suite["devices"]["physical_mapping_available"] is True
    assert [entry["serial"] for entry in suite["devices"]["physical_mapping"]] == [
        f"test-serial-{device}" for device in (0, 2, 4, 6, 8, 10, 12, 14)
    ]
    assert suite["sampling"] == {"rounds": 100, "seed": 1807, "warmup": 5}
    assert suite["provenance"]["torch"]["version"]
    assert suite["provenance"]["torch_npu"]["version"]
    assert suite["provenance"]["driver"]["available"]
    assert [case["case_id"] for case in suite["cases"]] == [
        "moe-ep8",
        "decode-logits",
        "mtp-core",
    ]


@pytest.mark.parametrize(
    ("extra_args", "error"),
    [
        (["--rounds", "99"], "requires --rounds 100"),
        (["--warmup", "4"], "requires --warmup 5"),
        (["--seed", "7"], "requires --seed 1807"),
        (["--platform", "a2a3sim"], "requires --platform a2a3"),
        (["--device", "0,1,2,3,4,5,6,7"], "requires --device 0,2,4,6,8,10,12,14"),
    ],
)
def test_suite_runner_rejects_nonofficial_config(extra_args: list[str], error: str) -> None:
    result = subprocess.run(
        [str(_SUITE_RUNNER), "--device", _DEVICE_SET, "--dry-run", *extra_args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert error in result.stderr


@pytest.mark.parametrize(
    ("extra_args", "error"),
    [
        (["--rounds", "99"], "require --rounds 100"),
        (["--warmup", "4"], "require --warmup 5"),
        (["--platform", "a2a3sim"], "require --platform a2a3"),
        (["--device", "0,1,2,3,4,5,6,7"], "require --device 0,2,4,6,8,10,12,14"),
    ],
)
def test_runner_rejects_nonofficial_measured_config(
    extra_args: list[str],
    error: str,
) -> None:
    result = subprocess.run(
        [str(_RUNNER), "--device", _DEVICE_SET, "--dry-run", *extra_args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert error in result.stderr


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        (
            f"raw samples: ranks=8 rounds={_ROUNDS} warmup={_WARMUP}",
            f"raw samples: ranks=8 rounds={_ROUNDS} warmup={_WARMUP} fallback_flattened=1",
            "fallback_flattened=1",
        ),
        ("rank 100 raw n=100", "rank 100 raw n=99", "declares n=99"),
        ("[1.0, 1.0, 1.0", "[0.0, 1.0, 1.0", "finite and positive"),
        ("rounds=100 ranks=8", "rounds=99 ranks=8", "completed benchmark context"),
    ],
)
def test_parser_rejects_unofficial_decode_logs(
    tmp_path: Path,
    old: str,
    new: str,
    error: str,
) -> None:
    path = _decode_log(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    with pytest.raises(MetricParseError, match=error):
        parse_perf_log(
            "decode-logits",
            path,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
        )


def test_parser_rejects_a_missing_rank(tmp_path: Path) -> None:
    path = _decode_log(tmp_path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if "rank 107 raw" not in line]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(MetricParseError, match="8 unique rank sample lines"):
        parse_perf_log(
            "decode-logits",
            path,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("remove_pass", "completed RUN PASS"),
        ("append_run", "RUN PASS must be the final RUN line"),
        ("append_traceback", "Python traceback"),
        ("replace_with_fail", "failed RUN line"),
    ],
)
def test_parser_rejects_incomplete_or_failed_runs(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    path = _decode_log(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if mutation == "remove_pass":
        lines.pop()
    elif mutation == "append_run":
        lines.append("[RUN] runtime done (1.00s)")
    elif mutation == "append_traceback":
        lines.append("Traceback (most recent call last):")
    elif mutation == "replace_with_fail":
        lines[-1] = "[RUN] FAIL (1.00s)"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(MetricParseError, match=error):
        parse_perf_log(
            "decode-logits",
            path,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
        )


def test_parser_rejects_rank_summary_and_raw_pid_mismatch(tmp_path: Path) -> None:
    path = _decode_log(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("rank 107 raw", "rank 999 raw", 1),
        encoding="utf-8",
    )

    with pytest.raises(MetricParseError, match="rank-summary/raw PID mismatch"):
        parse_perf_log(
            "decode-logits",
            path,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
        )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("slot 0 (eplb_mtp_core_logits)", "slot 0 (renamed_compute)"),
        ("slot 1 (eplb_mtp_core_cleanup)", "slot 2 (eplb_mtp_core_cleanup)"),
    ],
)
def test_parser_rejects_mtp_slot_contract_drift(tmp_path: Path, old: str, new: str) -> None:
    path = _mtp_log(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    with pytest.raises(MetricParseError, match="slot contract mismatch"):
        parse_perf_log(
            "mtp-core",
            path,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
        )


def test_parser_rejects_swapped_mtp_raw_interleave(tmp_path: Path) -> None:
    path = _mtp_log(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if "rank 100 raw" not in line:
            continue
        samples = ast.literal_eval(line.partition("eff_us=")[2])
        swapped = [value for pair in zip(samples[1::2], samples[0::2], strict=True) for value in pair]
        lines[index] = _raw_line(100, swapped)
        break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(MetricParseError, match="slot 0 .* mismatch"):
        parse_perf_log(
            "mtp-core",
            path,
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
        )


def test_parser_rejects_noncanonical_devices(tmp_path: Path) -> None:
    with pytest.raises(MetricParseError, match="ordered devices"):
        parse_perf_log(
            "decode-logits",
            _decode_log(tmp_path),
            rounds=_ROUNDS,
            warmup=_WARMUP,
            device_set="0,1,2,3,4,5,6,7",
        )


def test_parser_rejects_nonofficial_sampling_config(tmp_path: Path) -> None:
    with pytest.raises(MetricParseError, match="rounds/warmup=100/5"):
        parse_perf_log(
            "decode-logits",
            _decode_log(tmp_path),
            rounds=99,
            warmup=_WARMUP,
            device_set=_DEVICE_SET,
        )
