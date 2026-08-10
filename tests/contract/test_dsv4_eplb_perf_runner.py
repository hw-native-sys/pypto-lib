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
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dsv4_eplb_perf_metrics import MetricParseError, parse_perf_log


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARSER = _REPO_ROOT / "tools" / "dsv4_eplb_perf_metrics.py"
_RUNNER = _REPO_ROOT / "tools" / "run_dsv4_eplb_perf.sh"
_DEVICE_SET = "0,2,4,6,8,10,12,14"
_PIDS = tuple(range(100, 108))
_ROUNDS = 100
_WARMUP = 5


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
    lines = [_rank_summary(pid, samples[pid]) for pid in _PIDS]
    lines.append(f"[RUN]   raw samples: ranks=8 rounds={_ROUNDS} warmup={_WARMUP}")
    lines.extend(_raw_line(pid, samples[pid]) for pid in _PIDS)
    lines.append(
        "[RUN] benchmark kernel=_jit_l3_eplb_decode_logits "
        f"l3_resident=1 rounds={_ROUNDS} ranks=8 host_union_mean_us=100"
    )
    return _write_log(tmp_path, "decode.log", lines)


def _mtp_log(tmp_path: Path) -> Path:
    lines = [
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
    return _write_log(tmp_path, "mtp.log", lines)


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
        "dsv4-eplb-v1",
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
    assert "Metric contract: dsv4-eplb-v1" in result.stdout
    assert "PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_WARMUP=5" in result.stdout
    assert "eplb_decode_logits.py" in result.stdout
    assert "eplb_mtp_core.py" in result.stdout
    assert "--finite-only" in result.stdout


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
