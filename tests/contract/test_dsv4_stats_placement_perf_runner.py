# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _REPO_ROOT / "tools" / "run_dsv4_stats_placement_perf.sh"
_MANIFEST = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp" / "stats_placement_decode_manifest.json"
_DEVICE_SET = "0,2,4,6,8,10,12,14"
_ALLOCATOR_DEVICE_SET = "1,3,5,7,9,11,13,15"
_LEGACY_COMPARISON_HEADER = "\t".join(
    [
        "case",
        "contiguous_fastest_rank_median_us",
        "stats_fastest_rank_median_us",
        "fastest_stats_minus_contiguous_us",
        "fastest_stats_minus_contiguous_pct",
        "contiguous_max_rank_median_us",
        "stats_max_rank_median_us",
        "max_rank_stats_minus_contiguous_us",
        "max_rank_stats_minus_contiguous_pct",
        "contiguous_rank_median_spread_us",
        "stats_rank_median_spread_us",
        "spread_stats_minus_contiguous_us",
        "winner_by_max_rank_median",
        "validation_mode",
        "golden_replayed",
    ]
)
_STATS_VS_EPLB_HEADER = "\t".join(
    [
        "case",
        "eplb_fastest_rank_median_us",
        "stats_fastest_rank_median_us",
        "fastest_stats_minus_eplb_us",
        "fastest_stats_minus_eplb_pct",
        "eplb_max_rank_median_us",
        "stats_max_rank_median_us",
        "max_rank_stats_minus_eplb_us",
        "max_rank_stats_minus_eplb_pct",
        "eplb_rank_median_spread_us",
        "stats_rank_median_spread_us",
        "spread_stats_minus_eplb_us",
        "winner_by_max_rank_median",
        "validation_mode",
        "golden_replayed",
    ]
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_RUNNER), "--device", _DEVICE_SET, *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _command_lines(output: str) -> list[str]:
    return [
        line for line in output.splitlines() if "run_seeded_python.py" in line and "stats_placement_" in line
    ]


def test_dry_run_pairs_identical_ep8x32_workloads_for_all_three_entrypoints() -> None:
    result = _run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert "Comparison contract: dsv4-stats-placement-compare-v2" in result.stdout
    assert "Stats-vs-EPLB contract: dsv4-stats-vs-eplb-compare-v2" in result.stdout
    assert "Metric contract: dsv4-stats-placement-numeric-v4" in result.stdout
    assert "MoE metric contract: dsv4-stats-placement-moe-ep8-v3" in result.stdout
    assert "MoE workload contract: dsv4-stats-placement-moe-ep8x32-v1" in result.stdout
    assert "same EP8x32 stats-shaped routes; only expert placement changes" in result.stdout
    assert "same routes, weights, and inputs; physical expert IDs and tensors follow placement" in result.stdout
    assert "EPLB algorithm: deepseek-eplb-balanced-packing-no-redundancy" in result.stdout
    assert "EPLB expert order: upstream-fp32-torch-sort-descending" in result.stdout
    assert "EPLB timing: solve_in_timed_region=false weight_migration_in_timed_region=false" in result.stdout
    assert "EPLB control kind: placement-quality-oracle" in result.stdout
    assert (
        "moe-ep8=numeric_golden decode-logits=numeric_golden "
        "mtp-core=numeric_golden golden_replayed=false"
    ) in result.stdout
    assert f"Manifest: {_MANIFEST}" in result.stdout

    headings = [
        "moe-ep8-contiguous:",
        "moe-ep8-stats:",
        "moe-ep8-eplb:",
        "decode-logits-contiguous:",
        "decode-logits-stats:",
        "decode-logits-eplb:",
        "mtp-core-contiguous:",
        "mtp-core-stats:",
        "mtp-core-eplb:",
    ]
    assert all(heading in result.stdout for heading in headings)
    assert [result.stdout.index(heading) for heading in headings] == sorted(
        result.stdout.index(heading) for heading in headings
    )

    commands = _command_lines(result.stdout)
    assert len(commands) == 9
    for command in commands:
        assert "--ep 8" in command
        assert "--experts-per-rank 32" in command
        assert "--num-tokens 8" in command
        assert str(_MANIFEST) in command
        assert command.count("--seed 1807") == 1
    assert sum("--expert-placement contiguous" in command for command in commands) == 3
    assert sum("--expert-placement stats" in command for command in commands) == 3
    assert sum("--expert-placement eplb" in command for command in commands) == 3
    assert all("--placement-manifest" in command for command in commands)
    assert all("stats_placement_moe.py" in command for command in commands[:3])
    assert all("--layer-id 0" in command for command in commands[:3])
    assert all("--tp 4" not in command for command in commands[:3])
    assert all("--start-pos 8192" not in command for command in commands[:3])
    assert all("--balanced-routing" not in command for command in commands[:3])
    assert all("--finite-only" not in command for command in commands[:3])
    assert all("stats_placement_decode_logits.py" in command for command in commands[3:6])
    assert all("--tp 4" in command for command in commands[3:])
    assert all("--start-pos 8192" in command for command in commands[3:])
    assert all("--finite-only" not in command for command in commands[3:6])
    assert all("stats_placement_mtp_core.py" in command for command in commands[6:])
    assert all("--finite-only" not in command for command in commands[6:])


def test_dry_run_filters_one_case_and_one_placement() -> None:
    result = _run("--case", "mtp-core", "--placement", "eplb", "--dry-run")

    assert result.returncode == 0, result.stderr
    commands = _command_lines(result.stdout)
    assert len(commands) == 1
    assert "mtp-core-eplb:" in result.stdout
    assert "--expert-placement eplb" in commands[0]
    assert "stats_placement_mtp_core.py" in commands[0]
    assert "decode-logits" not in result.stdout
    assert "contiguous:" not in result.stdout
    assert "mtp-core-stats:" not in result.stdout


def test_dry_run_selects_only_the_stats_shaped_moe_variant() -> None:
    result = _run("--case", "moe-ep8", "--placement", "stats", "--dry-run")

    assert result.returncode == 0, result.stderr
    commands = _command_lines(result.stdout)
    assert len(commands) == 1
    assert "moe-ep8-stats:" in result.stdout
    assert "stats_placement_moe.py" in commands[0]
    assert "--expert-placement stats" in commands[0]
    assert commands[0].count("--seed 1807") == 1
    assert "--balanced-routing" not in commands[0]
    assert "decode-logits" not in result.stdout
    assert "mtp-core" not in result.stdout


def test_runner_accepts_the_allocator_device_mapping() -> None:
    result = subprocess.run(
        [str(_RUNNER), "--device", _ALLOCATOR_DEVICE_SET, "--dry-run"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = _command_lines(result.stdout)
    assert len(commands) == 9
    escaped_device_set = _ALLOCATOR_DEVICE_SET.replace(",", r"\,")
    assert all(f"-d {escaped_device_set}" in command for command in commands)


def test_runner_rejects_an_unsupported_device_mapping() -> None:
    result = subprocess.run(
        [str(_RUNNER), "--device", "0,1,2,3,4,5,6,7", "--dry-run"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert (
        f"comparison metric requires --device {_DEVICE_SET} or {_ALLOCATOR_DEVICE_SET}"
        in result.stderr
    )


@pytest.mark.parametrize(
    ("option", "value", "expected"),
    [
        ("--rounds", "1", "requires --rounds 100"),
        ("--warmup", "0", "requires --warmup 5"),
    ],
)
def test_measured_runner_rejects_nonofficial_sample_counts(
    option: str,
    value: str,
    expected: str,
) -> None:
    result = _run(option, value, "--dry-run")

    assert result.returncode == 2
    assert expected in result.stderr


def test_compile_only_execution_records_every_selected_variant(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "--version" ]]; then\n'
        "  printf 'Python fake\\n'\n"
        'elif [[ "${1:-}" == "-c" ]]; then\n'
        "  printf 'torch-fake\\n'\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_dir = tmp_path / "results"

    result = _run(
        "--compile-only",
        "--python",
        str(fake_python),
        "--rounds",
        "1",
        "--warmup",
        "0",
        "--seed",
        "7",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    result_rows = (output_dir / "results.tsv").read_text(encoding="utf-8").splitlines()
    assert len(result_rows) == 10
    assert all(len(row.split("\t")) == 29 for row in result_rows)
    assert [row.split("\t")[0] for row in result_rows[1:]] == [
        "moe-ep8-contiguous",
        "moe-ep8-stats",
        "moe-ep8-eplb",
        "decode-logits-contiguous",
        "decode-logits-stats",
        "decode-logits-eplb",
        "mtp-core-contiguous",
        "mtp-core-stats",
        "mtp-core-eplb",
    ]
    assert all(row.split("\t")[3] == "pass_compile" for row in result_rows[1:])
    assert [row.split("\t")[4] for row in result_rows[1:]] == [
        "numeric_golden",
        "numeric_golden",
        "numeric_golden",
        "numeric_golden",
        "numeric_golden",
        "numeric_golden",
        "numeric_golden",
        "numeric_golden",
        "numeric_golden",
    ]
    assert all(row.split("\t")[5] == "false" for row in result_rows[1:])
    comparison_rows = (output_dir / "comparison.tsv").read_text(encoding="utf-8").splitlines()
    assert comparison_rows == [_LEGACY_COMPARISON_HEADER]
    stats_vs_eplb_rows = (output_dir / "stats-vs-eplb.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert stats_vs_eplb_rows == [_STATS_VS_EPLB_HEADER]
    balance_rows = (output_dir / "rank-balance.tsv").read_text(encoding="utf-8").splitlines()
    assert len(balance_rows) == 1
    metadata = dict(
        line.split("\t", maxsplit=1)
        for line in (output_dir / "metadata.tsv").read_text(encoding="utf-8").splitlines()
    )
    assert metadata["comparison_contract_version"] == "dsv4-stats-placement-compare-v2"
    assert metadata["stats_vs_eplb_contract_version"] == "dsv4-stats-vs-eplb-compare-v2"
    assert metadata["metric_contract_version"] == "dsv4-stats-placement-numeric-v4"
    assert metadata["moe_metric_contract_version"] == "dsv4-stats-placement-moe-ep8-v3"
    assert metadata["moe_workload_contract_version"] == (
        "dsv4-stats-placement-moe-ep8x32-v1"
    )
    assert metadata["comparison_case_set"] == "moe-ep8,decode-logits,mtp-core"
    assert metadata["moe_tp_size"] == "not_applicable"
    assert metadata["moe_layer_id"] == "0"
    assert metadata["moe_validation_mode"] == "numeric_golden"
    assert metadata["decode_validation_mode"] == "numeric_golden"
    assert metadata["mtp_validation_mode"] == "numeric_golden"
    assert metadata["eplb_algorithm"] == "deepseek-eplb-balanced-packing-no-redundancy"
    assert metadata["eplb_algorithm_version"] == "d52c72d5b2f2fb4c41afbf8eb21366820239913d"
    assert metadata["eplb_load_source"] == "replayed-logical-route-counts"
    assert metadata["eplb_mapping_basis"] == "current-replay-histogram"
    assert metadata["eplb_expert_order"] == "upstream-fp32-torch-sort-descending"
    assert metadata["eplb_solver_torch_version"] == "torch-fake"
    assert metadata["eplb_redundant_experts"] == "0"
    assert metadata["eplb_solve_in_timed_region"] == "false"
    assert metadata["eplb_weight_migration_in_timed_region"] == "false"
    assert metadata["eplb_control_scope"] == (
        "EP8x32 placement-only, not legacy EP8x16 branch baseline"
    )
    assert metadata["eplb_control_kind"] == "placement-quality-oracle"
    assert (output_dir / "source-status.txt").is_file()
    assert all(
        (output_dir / f"{variant}.log").is_file()
        for variant in (
            "moe-ep8-contiguous",
            "moe-ep8-stats",
            "moe-ep8-eplb",
            "decode-logits-contiguous",
            "decode-logits-stats",
            "decode-logits-eplb",
            "mtp-core-contiguous",
            "mtp-core-stats",
            "mtp-core-eplb",
        )
    )


def test_comparison_winner_uses_max_rank_median_and_reports_fastest_separately(
    tmp_path: Path,
) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "--version" ]]; then\n'
        "  printf 'Python fake\\n'\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "${1:-}" == "-c" ]]; then\n'
        "  printf 'torch-fake\\n'\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "${1:-}" == */dsv4_eplb_perf_metrics.py ]]; then\n'
        "  shift\n"
        "  case_name=''\n"
        "  log_file=''\n"
        "  rank_output=''\n"
        "  validation_profile=''\n"
        '  while [[ "$#" -gt 0 ]]; do\n'
        '    case "$1" in\n'
        '      --case) case_name="$2"; shift 2 ;;\n'
        '      --log) log_file="$2"; shift 2 ;;\n'
        '      --rank-output) rank_output="$2"; shift 2 ;;\n'
        '      --validation-profile) validation_profile="$2"; shift 2 ;;\n'
        "      *) shift ;;\n"
        "    esac\n"
        "  done\n"
        '  [[ "$validation_profile" == "stats-placement-numeric" ]] || exit 9\n'
        '  if [[ "$log_file" == *-contiguous.log ]]; then\n'
        "    medians=(10 11 12 13 14 15 16 17)\n"
        '  elif [[ "$log_file" == *-eplb.log ]]; then\n'
        "    medians=(8 10 11 12 13 14 15 20)\n"
        "  else\n"
        "    medians=(9 10 11 12 13 14 15 19)\n"
        "  fi\n"
        "  scope='compare3_fastest_rank'\n"
        "  task='eplb_decode_logits'\n"
        "  dispatches=1\n"
        '  if [[ "$case_name" == "moe-ep8" ]]; then\n'
        "    scope='moe_ep8_fastest_rank'\n"
        "    task='moe'\n"
        "  fi\n"
        '  if [[ "$case_name" == "mtp-core" ]]; then\n'
        "    scope='compare4_fastest_rank_compute_only'\n"
        "    task='eplb_mtp_core_logits'\n"
        "    dispatches=2\n"
        "  fi\n"
        '  for index in "${!medians[@]}"; do\n'
        "    selected=0\n"
        '    [[ "$index" -eq 0 ]] && selected=1\n'
        '    printf \'%s\\t%s\\t%s\\t%s\\t%s\\t0\\t%s\\t%s\\t100\\t%s\\t%s\\t%s\\t%s\\n\' "$case_name" "$scope" "$index" "$((index * 2))" "$((1000 + index))" "$task" "$selected" "${medians[$index]}" "${medians[$index]}" "${medians[$index]}" "${medians[$index]}" >>"$rank_output"\n'
        '    if [[ "$case_name" == "mtp-core" ]]; then\n'
        '      printf \'%s\\t%s\\t%s\\t%s\\t%s\\t1\\teplb_mtp_core_cleanup\\t0\\t100\\t1000\\t1000\\t1000\\t1000\\n\' "$case_name" "$scope" "$index" "$((index * 2))" "$((1000 + index))" >>"$rank_output"\n'
        "    fi\n"
        "  done\n"
        '  selected_median="${medians[0]}"\n'
        "  metric_version='dsv4-stats-placement-numeric-v4'\n"
        '  [[ "$case_name" == "moe-ep8" ]] && metric_version="dsv4-stats-placement-moe-ep8-v3"\n'
        '  printf \'%s\\t%s\\tminimum_rank_median\\t100\\t5\\t8\\t%s\\t0\\t0\\t1000\\t100\\t%s\\t%s\\t%s\\t%s\\t-\\t-\\t-\\t-\\traw_all\\tordered_pid_to_ordered_device_set\\n\' "$metric_version" "$scope" "$dispatches" "$selected_median" "$selected_median" "$selected_median" "$selected_median"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_dir = tmp_path / "measured-results"

    result = _run(
        "--python",
        str(fake_python),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    balance_rows = (output_dir / "rank-balance.tsv").read_text(encoding="utf-8").splitlines()
    assert balance_rows[1:] == [
        "moe-ep8\tcontiguous\t10.000\t17.000\t7.000",
        "moe-ep8\tstats\t9.000\t19.000\t10.000",
        "moe-ep8\teplb\t8.000\t20.000\t12.000",
        "decode-logits\tcontiguous\t10.000\t17.000\t7.000",
        "decode-logits\tstats\t9.000\t19.000\t10.000",
        "decode-logits\teplb\t8.000\t20.000\t12.000",
        "mtp-core\tcontiguous\t10.000\t17.000\t7.000",
        "mtp-core\tstats\t9.000\t19.000\t10.000",
        "mtp-core\teplb\t8.000\t20.000\t12.000",
    ]
    comparison_rows = (output_dir / "comparison.tsv").read_text(encoding="utf-8").splitlines()
    assert comparison_rows[0] == _LEGACY_COMPARISON_HEADER
    assert "fastest_rank_median_us" in comparison_rows[0]
    assert "winner_by_max_rank_median" in comparison_rows[0]
    expected_metrics = [
        "10.000",
        "9.000",
        "-1.000",
        "-10.000",
        "17.000",
        "19.000",
        "2.000",
        "11.765",
        "7.000",
        "10.000",
        "3.000",
        "contiguous",
    ]
    assert comparison_rows[1].split("\t") == [
        "moe-ep8",
        *expected_metrics,
        "numeric_golden",
        "false",
    ]
    assert comparison_rows[2].split("\t") == [
        "decode-logits",
        *expected_metrics,
        "numeric_golden",
        "false",
    ]
    assert comparison_rows[3].split("\t") == [
        "mtp-core",
        *expected_metrics,
        "numeric_golden",
        "false",
    ]

    stats_vs_eplb_rows = (output_dir / "stats-vs-eplb.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert stats_vs_eplb_rows[0] == _STATS_VS_EPLB_HEADER
    stats_vs_eplb_metrics = [
        "8.000",
        "9.000",
        "1.000",
        "12.500",
        "20.000",
        "19.000",
        "-1.000",
        "-5.000",
        "12.000",
        "10.000",
        "-2.000",
        "stats",
    ]
    assert stats_vs_eplb_rows[1].split("\t") == [
        "moe-ep8",
        *stats_vs_eplb_metrics,
        "numeric_golden",
        "false",
    ]
    assert stats_vs_eplb_rows[2].split("\t") == [
        "decode-logits",
        *stats_vs_eplb_metrics,
        "numeric_golden",
        "false",
    ]
    assert stats_vs_eplb_rows[3].split("\t") == [
        "mtp-core",
        *stats_vs_eplb_metrics,
        "numeric_golden",
        "false",
    ]
