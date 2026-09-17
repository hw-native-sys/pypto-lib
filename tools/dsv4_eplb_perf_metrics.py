# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Extract the official DeepSeek-V4 EPLB performance metrics from benchmark logs."""

from __future__ import annotations

import argparse
import ast
import math
import re
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence


CANONICAL_DEVICE_IDS = (0, 2, 4, 6, 8, 10, 12, 14)
ALLOCATOR_DEVICE_IDS = (1, 3, 5, 7, 9, 11, 13, 15)
SUPPORTED_DEVICE_ID_SETS = (CANONICAL_DEVICE_IDS, ALLOCATOR_DEVICE_IDS)
OFFICIAL_ROUNDS = 100
OFFICIAL_WARMUP = 5
OFFICIAL_SEED = 1807
SELECTION_POLICY = "minimum_rank_median"
MAPPING_BASIS = "ordered_pid_to_ordered_device_set"

_STATS_DECODE_VALIDATION_COMPARATORS = (
    (
        "kv_cache",
        "stacked_mapped_pool_cosine_rel_l2(kv_cache, min_cosine=0.99, "
        "max_rel_l2=0.1, max_bad_row_ratio=0.02, hard_min_cosine=0.98, "
        "hard_max_rel_l2=0.5, strict_layers=(0,), strict_atol=0.001, "
        "strict_rtol=0.03, strict_max_error_ratio=0.01, "
        "expected_active_rows=2752)",
    ),
    (
        "hca_compress_state",
        "stacked_mapped_pool_cosine_rel_l2(hca_compress_state, "
        "min_cosine=0.99, max_rel_l2=0.1, max_bad_row_ratio=0.02, "
        "hard_min_cosine=0.98, hard_max_rel_l2=0.5, strict_layers=(), "
        "strict_atol=0.005, strict_rtol=0.02, "
        "strict_max_error_ratio=0.01, expected_active_rows=1280)",
    ),
    (
        "csa_compress_state",
        "stacked_mapped_pool_cosine_rel_l2(csa_compress_state, "
        "min_cosine=0.99, max_rel_l2=0.1, max_bad_row_ratio=0.02, "
        "hard_min_cosine=0.98, hard_max_rel_l2=0.5, strict_layers=(), "
        "strict_atol=0.005, strict_rtol=0.02, "
        "strict_max_error_ratio=0.01, expected_active_rows=1344)",
    ),
    (
        "csa_inner_compress_state",
        "stacked_mapped_pool_cosine_rel_l2(csa_inner_compress_state, "
        "min_cosine=0.99, max_rel_l2=0.1, max_bad_row_ratio=0.02, "
        "hard_min_cosine=0.98, hard_max_rel_l2=0.5, strict_layers=(), "
        "strict_atol=0.005, strict_rtol=0.02, "
        "strict_max_error_ratio=0.01, expected_active_rows=1344)",
    ),
    (
        "hca_cmp_kv",
        "stacked_mapped_pool_cosine_rel_l2(hca_cmp_kv, min_cosine=0.99, "
        "max_rel_l2=0.1, max_bad_row_ratio=0.02, hard_min_cosine=0.98, "
        "hard_max_rel_l2=0.5, strict_layers=(), strict_atol=0.001, "
        "strict_rtol=0.03, strict_max_error_ratio=0.01, "
        "expected_active_rows=0)",
    ),
    (
        "csa_cmp_kv",
        "stacked_mapped_pool_cosine_rel_l2(csa_cmp_kv, min_cosine=0.99, "
        "max_rel_l2=0.1, max_bad_row_ratio=0.02, hard_min_cosine=0.98, "
        "hard_max_rel_l2=0.5, strict_layers=(), strict_atol=0.001, "
        "strict_rtol=0.03, strict_max_error_ratio=0.01, "
        "expected_active_rows=0)",
    ),
    (
        "idx_kv_cache",
        "stacked_mapped_pool_ratio_allclose(idx_kv_cache, atol=1, rtol=0, "
        "max_error_ratio=0.02, expected_active_rows=0)",
    ),
    (
        "idx_kv_scale",
        "stacked_mapped_pool_cosine_rel_l2(idx_kv_scale, min_cosine=0.99, "
        "max_rel_l2=0.1, max_bad_row_ratio=0.02, hard_min_cosine=0.98, "
        "hard_max_rel_l2=0.5, strict_layers=(), strict_atol=0.001, "
        "strict_rtol=0.01, strict_max_error_ratio=0.01, "
        "expected_active_rows=0)",
    ),
    (
        "pre_hc_hidden_out",
        "rank_token_cosine_rel_l2(pre_hc_hidden_out, min_cosine=0.99, "
        "max_rel_l2=0.1, expected_active_rows=64)",
    ),
    (
        "hidden_out",
        "rank_token_cosine_rel_l2(hidden_out, min_cosine=0.99, "
        "max_rel_l2=0.1, expected_active_rows=64)",
    ),
    (
        "logits",
        "logits_cosine_compare(min_cosine=0.99, max_rel_l2=0.1, "
        "expected_active_rows=64)",
    ),
)

_STATS_DECODE_GOLDEN_METRIC_ROWS = (
    ("kv_cache", 2752),
    ("hca_compress_state", 1280),
    ("csa_compress_state", 1344),
    ("csa_inner_compress_state", 1344),
    ("hca_cmp_kv", 0),
    ("csa_cmp_kv", 0),
    ("idx_kv_scale", 0),
    ("pre_hc_hidden_out", 64),
    ("hidden_out", 64),
    ("logits", 64),
)

_STATS_MTP_VALIDATION_COMPARATORS = (
    (
        "kv_cache",
        "mapped_pool_ratio_reldiff(mapping=swa_slot_mapping, shape=(8, 8), "
        "block_size=128, leading_rank_axis=True, diff_thd=0.01, "
        "pct_thd=0.05, max_diff_hd=inf, expected_mapped_rows=64, "
        "compare_mapped_rows_individually=True)",
    ),
    (
        "hidden_out",
        "rowwise_ratio_reldiff(output=hidden_out, row_shape=(8, 8), "
        "mapping=None, diff_thd=0.02, pct_thd=0.1, max_diff_hd=inf, "
        "expected_active_rows=64)",
    ),
    (
        "next_pre_hc_hidden",
        "rowwise_ratio_reldiff(output=next_pre_hc_hidden, "
        "row_shape=(8, 8), mapping=None, diff_thd=0.02, pct_thd=0.05, "
        "max_diff_hd=inf, expected_active_rows=64)",
    ),
    (
        "logits",
        "rowwise_ratio_reldiff(output=logits, row_shape=(8, 8), "
        "mapping=logit_row_indices, diff_thd=0.02, pct_thd=0.1, "
        "max_diff_hd=inf, expected_active_rows=64)",
    ),
)

_STATS_MOE_VALIDATION_COMPARATORS = (
    (
        "x_next",
        "rowwise_ratio_reldiff(output=x_next, row_shape=(8, 8), "
        "mapping=None, diff_thd=0.003, pct_thd=0.1, max_diff_hd=inf, "
        "expected_active_rows=64, aggregate_pct_thd=0.05)",
    ),
)


@dataclass(frozen=True)
class CaseConfig:
    metric_contract_version: str
    metric_scope: str
    kernel: str
    dispatches_per_round: int
    baseline_median_us: float | None
    metric_source: str
    primary_task: str
    validation_mode: str
    slot_tasks: tuple[str, ...]
    required_validation_outputs: tuple[str, ...] = ()
    required_validation_comparators: tuple[tuple[str, str], ...] = ()
    validation_reference: str | None = None
    required_seed: int | None = None
    exact_validation_comparators: bool = False
    required_golden_metric_rows: tuple[tuple[str, int], ...] = ()


CASE_CONFIGS = {
    "moe-ep8": CaseConfig(
        metric_contract_version="dsv4-moe-ep8-v1",
        metric_scope="moe_ep8_fastest_rank",
        kernel="_jit_l3_moe",
        dispatches_per_round=1,
        baseline_median_us=None,
        metric_source="raw_all",
        primary_task="moe",
        validation_mode="numeric_golden",
        slot_tasks=(),
        required_validation_outputs=("x_next",),
        required_validation_comparators=(("x_next", "ratio_reldiff"),),
        required_seed=OFFICIAL_SEED,
    ),
    "decode-logits": CaseConfig(
        metric_contract_version="dsv4-eplb-v2",
        metric_scope="compare3_fastest_rank",
        kernel="_jit_l3_eplb_decode_logits",
        dispatches_per_round=1,
        baseline_median_us=None,
        metric_source="raw_all",
        primary_task="eplb_decode_logits",
        validation_mode="runtime_only",
        slot_tasks=(),
        required_seed=OFFICIAL_SEED,
    ),
    "mtp-core": CaseConfig(
        metric_contract_version="dsv4-eplb-v2",
        metric_scope="compare4_fastest_rank_compute_only",
        kernel="_jit_l3_eplb_mtp_core",
        dispatches_per_round=2,
        baseline_median_us=None,
        metric_source="raw_even_with_slot_validation",
        primary_task="eplb_mtp_core_logits",
        validation_mode="finite_only",
        slot_tasks=("eplb_mtp_core_logits", "eplb_mtp_core_cleanup"),
        required_validation_outputs=("kv_cache", "hidden_out", "next_pre_hc_hidden", "logits"),
        required_validation_comparators=(
            ("kv_cache", "finite_tensor_compare"),
            ("hidden_out", "finite_tensor_compare"),
            ("next_pre_hc_hidden", "finite_tensor_compare"),
            ("logits", "finite_tensor_compare"),
        ),
        required_seed=OFFICIAL_SEED,
    ),
}

STATS_NUMERIC_CASE_CONFIGS = {
    "moe-ep8": replace(
        CASE_CONFIGS["moe-ep8"],
        metric_contract_version="dsv4-stats-placement-moe-ep8-v3",
        required_validation_comparators=_STATS_MOE_VALIDATION_COMPARATORS,
        validation_reference="golden_moe",
        exact_validation_comparators=True,
        required_golden_metric_rows=(("x_next", 64),),
    ),
    "decode-logits": replace(
        CASE_CONFIGS["decode-logits"],
        metric_contract_version="dsv4-stats-placement-numeric-v4",
        validation_mode="numeric_golden",
        required_validation_outputs=(
            "kv_cache",
            "hca_compress_state",
            "csa_compress_state",
            "csa_inner_compress_state",
            "hca_cmp_kv",
            "csa_cmp_kv",
            "idx_kv_cache",
            "idx_kv_scale",
            "pre_hc_hidden_out",
            "hidden_out",
            "logits",
        ),
        required_validation_comparators=_STATS_DECODE_VALIDATION_COMPARATORS,
        validation_reference="golden_eplb_decode_logits",
        exact_validation_comparators=True,
        required_golden_metric_rows=_STATS_DECODE_GOLDEN_METRIC_ROWS,
    ),
    "mtp-core": replace(
        CASE_CONFIGS["mtp-core"],
        metric_contract_version="dsv4-stats-placement-numeric-v4",
        validation_mode="numeric_golden",
        required_validation_comparators=_STATS_MTP_VALIDATION_COMPARATORS,
        validation_reference="golden_eplb_mtp_core",
        exact_validation_comparators=True,
        required_golden_metric_rows=(
            ("kv_cache", 64),
            ("hidden_out", 64),
            ("next_pre_hc_hidden", 64),
            ("logits", 64),
        ),
    ),
}

VALIDATION_PROFILES = {
    "legacy": CASE_CONFIGS,
    "stats-placement-numeric": STATS_NUMERIC_CASE_CONFIGS,
}


class MetricParseError(ValueError):
    """Raised when a log cannot prove the official metric contract."""


@dataclass(frozen=True)
class SampleStats:
    samples: tuple[float, ...]

    @property
    def minimum(self) -> float:
        return min(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def maximum(self) -> float:
        return max(self.samples)


@dataclass(frozen=True)
class RankMetric:
    logical_rank: int
    device_id: int
    pid: int
    slot: int
    task: str
    selected: bool
    stats: SampleStats


@dataclass(frozen=True)
class MetricResult:
    case: str
    metric_contract_version: str
    rounds: int
    warmup: int
    dispatches_per_round: int
    selected_rank: int
    selected_device: int
    selected_pid: int
    selected_stats: SampleStats
    cleanup_median_us: float | None
    baseline_median_us: float | None
    metric_scope: str
    metric_source: str
    validation_mode: str
    validation_outputs: tuple[tuple[str, str], ...]
    rank_metrics: tuple[RankMetric, ...]

    @property
    def delta_us(self) -> float | None:
        if self.baseline_median_us is None:
            return None
        return self.selected_stats.median - self.baseline_median_us

    @property
    def delta_pct(self) -> float | None:
        if self.delta_us is None or self.baseline_median_us is None:
            return None
        return self.delta_us / self.baseline_median_us * 100.0

    def summary_tsv_fields(self) -> str:
        cleanup = "-" if self.cleanup_median_us is None else f"{self.cleanup_median_us:.3f}"
        baseline = (
            "-" if self.baseline_median_us is None else f"{self.baseline_median_us:.3f}"
        )
        delta_us = "-" if self.delta_us is None else f"{self.delta_us:.3f}"
        delta_pct = "-" if self.delta_pct is None else f"{self.delta_pct:.3f}"
        fields = (
            self.metric_contract_version,
            self.metric_scope,
            SELECTION_POLICY,
            str(self.rounds),
            str(self.warmup),
            str(len(CANONICAL_DEVICE_IDS)),
            str(self.dispatches_per_round),
            str(self.selected_rank),
            str(self.selected_device),
            str(self.selected_pid),
            str(len(self.selected_stats.samples)),
            f"{self.selected_stats.minimum:.3f}",
            f"{self.selected_stats.median:.3f}",
            f"{self.selected_stats.mean:.3f}",
            f"{self.selected_stats.maximum:.3f}",
            cleanup,
            baseline,
            delta_us,
            delta_pct,
            self.metric_source,
            MAPPING_BASIS,
        )
        return "\t".join(fields)

    def rank_tsv_rows(self) -> list[str]:
        rows = []
        for metric in self.rank_metrics:
            stats = metric.stats
            fields = (
                self.case,
                self.metric_scope,
                str(metric.logical_rank),
                str(metric.device_id),
                str(metric.pid),
                str(metric.slot),
                metric.task,
                "1" if metric.selected else "0",
                str(len(stats.samples)),
                f"{stats.minimum:.3f}",
                f"{stats.median:.3f}",
                f"{stats.mean:.3f}",
                f"{stats.maximum:.3f}",
            )
            rows.append("\t".join(fields))
        return rows


_RAW_HEADER_RE = re.compile(
    r"^\[RUN\]\s+raw samples: ranks=(?P<ranks>\d+) rounds=(?P<rounds>\d+) "
    r"warmup=(?P<warmup>\d+)(?P<suffix>.*)$"
)
_RAW_RANK_RE = re.compile(
    r"^\[RUN\]\s+rank (?P<pid>\d+) raw n=(?P<count>\d+) eff_us=(?P<samples>\[.*\])$"
)
_STATS_PATTERN = (
    r"eff_us min=(?P<minimum>[0-9]+(?:\.[0-9]+)?) "
    r"median=(?P<median>[0-9]+(?:\.[0-9]+)?) "
    r"mean=(?P<mean>[0-9]+(?:\.[0-9]+)?) "
    r"max=(?P<maximum>[0-9]+(?:\.[0-9]+)?)"
)
_RANK_SUMMARY_RE = re.compile(rf"^\[RUN\]\s+rank (?P<pid>\d+): {_STATS_PATTERN}$")
_SLOT_SUMMARY_RE = re.compile(
    rf"^\[RUN\]\s+slot (?P<slot>\d+) \((?P<task>[^)]+)\): {_STATS_PATTERN}$"
)
_VALIDATION_RE = re.compile(
    r"^\[RUN\]\s+'(?P<name>[^']+)' (?P<status>PASS|FAIL)\b.* "
    r"\((?P<comparator>.+)\)$"
)
_VALIDATION_CONTRACT_RE = re.compile(
    r"^\[VALIDATION\] mode=(?P<mode>\S+) reference=(?P<reference>\S+)$"
)
_GOLDEN_METRIC_RE = re.compile(
    r"^\[GOLDEN METRIC\] output=(?P<name>\S+) "
    r"active_rows=(?P<active_rows>\d+)\b"
)
_FIXTURE_SEED_RE = re.compile(
    r"^\[RUN\]\s+fixture seed=(?P<seed>-?\d+) python_hash_seed=(?P<hash_seed>\S+)$"
)
_RUN_PASS_RE = re.compile(r"^\[RUN\]\s+PASS\s+\(.*\)$")


def _parse_device_ids(device_set: str) -> tuple[int, ...]:
    try:
        device_ids = tuple(int(part) for part in device_set.split(","))
    except ValueError as error:
        raise MetricParseError(f"invalid device set: {device_set}") from error
    if device_ids not in SUPPORTED_DEVICE_ID_SETS:
        expected = " or ".join(
            ",".join(str(device_id) for device_id in supported)
            for supported in SUPPORTED_DEVICE_ID_SETS
        )
        raise MetricParseError(
            f"official EPLB metrics require ordered devices {expected}, got {device_set}"
        )
    return device_ids


def _parse_sample_list(payload: str, *, pid: int, declared_count: int) -> tuple[float, ...]:
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError) as error:
        raise MetricParseError(f"rank PID {pid} has a malformed raw sample list") from error
    if not isinstance(parsed, list):
        raise MetricParseError(f"rank PID {pid} raw samples are not a list")
    if len(parsed) != declared_count:
        raise MetricParseError(
            f"rank PID {pid} declares n={declared_count} but contains {len(parsed)} samples"
        )
    samples = []
    for index, value in enumerate(parsed):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricParseError(f"rank PID {pid} sample {index} is not numeric")
        sample = float(value)
        if not math.isfinite(sample) or sample <= 0.0:
            raise MetricParseError(
                f"rank PID {pid} sample {index} must be finite and positive, got {value!r}"
            )
        samples.append(sample)
    return tuple(samples)


def _require_single_raw_header(lines: Sequence[str], *, rounds: int, warmup: int) -> None:
    headers = [match for line in lines if (match := _RAW_HEADER_RE.match(line))]
    if len(headers) != 1:
        raise MetricParseError(f"expected exactly one raw-sample header, found {len(headers)}")
    header = headers[0]
    actual = (
        int(header.group("ranks")),
        int(header.group("rounds")),
        int(header.group("warmup")),
    )
    expected = (len(CANONICAL_DEVICE_IDS), rounds, warmup)
    if actual != expected:
        raise MetricParseError(
            "raw-sample header mismatch: "
            f"expected ranks/rounds/warmup={expected}, got {actual}"
        )
    if header.group("suffix").strip():
        raise MetricParseError(f"unexpected raw-sample header suffix: {header.group('suffix').strip()}")


def _require_context(lines: Sequence[str], config: CaseConfig, *, rounds: int) -> None:
    prefix = f"[RUN] benchmark kernel={config.kernel} l3_resident=1 rounds={rounds} ranks=8"
    contexts = [line for line in lines if line.startswith("[RUN] benchmark kernel=")]
    if len(contexts) != 1 or not (
        contexts[0] == prefix or contexts[0].startswith(f"{prefix} ")
    ):
        raise MetricParseError(
            f"expected exactly one completed benchmark context beginning with {prefix!r}"
        )


def _reported_stats(match: re.Match[str]) -> tuple[float, float, float, float]:
    return tuple(
        float(match.group(name)) for name in ("minimum", "median", "mean", "maximum")
    )


def _parse_summaries(
    lines: Sequence[str],
) -> tuple[
    dict[int, tuple[float, float, float, float]],
    dict[int, list[tuple[int, str, tuple[float, float, float, float]]]],
]:
    ranks: dict[int, tuple[float, float, float, float]] = {}
    slots: dict[int, list[tuple[int, str, tuple[float, float, float, float]]]] = {}
    current_pid: int | None = None
    for line in lines:
        if rank_match := _RANK_SUMMARY_RE.match(line):
            current_pid = int(rank_match.group("pid"))
            if current_pid in ranks:
                raise MetricParseError(f"duplicate rank summary for PID {current_pid}")
            ranks[current_pid] = _reported_stats(rank_match)
            continue
        if slot_match := _SLOT_SUMMARY_RE.match(line):
            if current_pid is None:
                raise MetricParseError("slot summary appears before its rank summary")
            slots.setdefault(current_pid, []).append(
                (
                    int(slot_match.group("slot")),
                    slot_match.group("task"),
                    _reported_stats(slot_match),
                )
            )
    return ranks, slots


def _require_reported_stats(
    label: str,
    actual: SampleStats,
    reported: tuple[float, float, float, float],
) -> None:
    actual_values = (actual.minimum, actual.median, actual.mean, actual.maximum)
    names = ("minimum", "median", "mean", "maximum")
    for name, actual_value, reported_value in zip(names, actual_values, reported, strict=True):
        if not math.isclose(actual_value, reported_value, rel_tol=0.0, abs_tol=0.11):
            raise MetricParseError(
                f"{label} {name} mismatch: raw={actual_value:.3f}, summary={reported_value:.3f}"
            )


def _require_completed_run(lines: Sequence[str], config: CaseConfig) -> None:
    if any("Traceback (most recent call last):" in line for line in lines):
        raise MetricParseError("benchmark log contains a Python traceback")
    failed_run_lines = [line for line in lines if line.startswith("[RUN]") and "FAIL" in line]
    if failed_run_lines:
        raise MetricParseError(f"benchmark log contains a failed RUN line: {failed_run_lines[0]}")
    passes = [line for line in lines if _RUN_PASS_RE.match(line)]
    if len(passes) != 1:
        raise MetricParseError(f"expected exactly one completed RUN PASS line, found {len(passes)}")
    run_lines = [line for line in lines if line.startswith("[RUN]")]
    if not run_lines or run_lines[-1] != passes[0]:
        raise MetricParseError("RUN PASS must be the final RUN line")
    has_runtime_only_note = "validation skipped: no golden_fn or golden_data" in passes[0]
    if config.validation_mode == "runtime_only" and not has_runtime_only_note:
        raise MetricParseError("runtime-only case must report validation skipped")
    if config.validation_mode != "runtime_only" and has_runtime_only_note:
        raise MetricParseError("validated case unexpectedly reports validation skipped")


def _require_validation(
    lines: Sequence[str], config: CaseConfig
) -> tuple[tuple[str, str], ...]:
    validation: dict[str, list[tuple[str, str]]] = {}
    for line in lines:
        if match := _VALIDATION_RE.match(line):
            validation.setdefault(match.group("name"), []).append(
                (match.group("status"), match.group("comparator"))
            )
    failures = sorted(
        name
        for name, records in validation.items()
        if any(status == "FAIL" for status, _comparator in records)
    )
    if failures:
        raise MetricParseError(f"validation failed for outputs: {', '.join(failures)}")
    required_comparators = dict(config.required_validation_comparators)
    for name in config.required_validation_outputs:
        records = validation.get(name, [])
        if len(records) != 1 or records[0][0] != "PASS":
            raise MetricParseError(
                f"expected exactly one validation PASS for {name!r}, found {records}"
            )
        expected_prefix = required_comparators.get(name)
        comparator = records[0][1]
        comparator_matches = (
            comparator == expected_prefix
            if config.exact_validation_comparators
            else (
                comparator == expected_prefix
                or comparator.startswith(f"{expected_prefix}(")
            )
        )
        if expected_prefix is not None and not comparator_matches:
            raise MetricParseError(
                f"validation comparator mismatch for {name!r}: "
                f"expected {expected_prefix!r}, got {comparator!r}"
            )
    return tuple(
        (name, records[0][0])
        for name, records in sorted(validation.items())
    )


def _require_golden_metrics(lines: Sequence[str], config: CaseConfig) -> None:
    expected_rows = dict(config.required_golden_metric_rows)
    if not expected_rows:
        return

    observed: dict[str, list[int]] = {}
    for line in lines:
        if match := _GOLDEN_METRIC_RE.match(line):
            observed.setdefault(match.group("name"), []).append(
                int(match.group("active_rows"))
            )
    for name, expected in expected_rows.items():
        rows = observed.get(name, [])
        if rows != [expected]:
            raise MetricParseError(
                f"Golden metric coverage mismatch for {name!r}: "
                f"expected active_rows={expected}, found {rows}"
            )


def _require_validation_contract(lines: Sequence[str], config: CaseConfig) -> None:
    if config.validation_reference is None:
        return
    matches = [
        match
        for line in lines
        if (match := _VALIDATION_CONTRACT_RE.match(line))
    ]
    if len(matches) != 1:
        raise MetricParseError(
            "expected exactly one stats validation-contract line, "
            f"found {len(matches)}"
        )
    actual = (matches[0].group("mode"), matches[0].group("reference"))
    expected = (config.validation_mode, config.validation_reference)
    if actual != expected:
        raise MetricParseError(
            f"validation contract mismatch: expected {expected}, got {actual}"
        )


def _require_fixture_seed(lines: Sequence[str], config: CaseConfig, seed: int | None) -> None:
    if config.required_seed is None:
        if seed is not None:
            raise MetricParseError("--seed is only valid for a seeded metric contract")
        return
    if seed != config.required_seed:
        raise MetricParseError(
            f"official {config.metric_contract_version} metrics require "
            f"seed={config.required_seed}, got {seed}"
        )
    matches = [match for line in lines if (match := _FIXTURE_SEED_RE.match(line))]
    if len(matches) != 1:
        raise MetricParseError(
            f"expected exactly one fixture-seed line, found {len(matches)}"
        )
    actual_seed = int(matches[0].group("seed"))
    actual_hash_seed = matches[0].group("hash_seed")
    if actual_seed != config.required_seed or actual_hash_seed != str(config.required_seed):
        raise MetricParseError(
            "fixture-seed mismatch: "
            f"expected seed/python_hash_seed={config.required_seed}, "
            f"got {actual_seed}/{actual_hash_seed}"
        )


def parse_perf_log(
    case: str,
    log_path: Path,
    *,
    rounds: int,
    warmup: int,
    device_set: str,
    seed: int | None = OFFICIAL_SEED,
    validation_profile: str = "legacy",
) -> MetricResult:
    """Parse one completed EPLB benchmark log using the official case contract."""

    if validation_profile not in VALIDATION_PROFILES:
        raise MetricParseError(
            f"unsupported validation profile: {validation_profile}"
        )
    case_configs = VALIDATION_PROFILES[validation_profile]
    if case not in case_configs:
        raise MetricParseError(f"unsupported EPLB performance case: {case}")
    if (rounds, warmup) != (OFFICIAL_ROUNDS, OFFICIAL_WARMUP):
        raise MetricParseError(
            "official EPLB metrics require "
            f"rounds/warmup={OFFICIAL_ROUNDS}/{OFFICIAL_WARMUP}, got {rounds}/{warmup}"
        )
    device_ids = _parse_device_ids(device_set)
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise MetricParseError(f"cannot read benchmark log {log_path}: {error}") from error
    if any("fallback_flattened=1" in line for line in lines):
        raise MetricParseError("fallback_flattened=1 cannot produce an official EPLB metric")

    config = case_configs[case]
    _require_completed_run(lines, config)
    _require_single_raw_header(lines, rounds=rounds, warmup=warmup)
    _require_context(lines, config, rounds=rounds)
    _require_validation_contract(lines, config)
    validation_outputs = _require_validation(lines, config)
    _require_golden_metrics(lines, config)
    _require_fixture_seed(lines, config, seed)

    raw_by_pid: dict[int, tuple[float, ...]] = {}
    for line in lines:
        match = _RAW_RANK_RE.match(line)
        if match is None:
            continue
        pid = int(match.group("pid"))
        if pid in raw_by_pid:
            raise MetricParseError(f"duplicate raw samples for rank PID {pid}")
        raw_by_pid[pid] = _parse_sample_list(
            match.group("samples"),
            pid=pid,
            declared_count=int(match.group("count")),
        )
    if len(raw_by_pid) != len(device_ids):
        raise MetricParseError(
            f"expected {len(device_ids)} unique rank sample lines, found {len(raw_by_pid)}"
        )

    rank_summaries, slots_by_pid = _parse_summaries(lines)
    if set(rank_summaries) != set(raw_by_pid):
        raise MetricParseError(
            "rank-summary/raw PID mismatch: "
            f"summaries={sorted(rank_summaries)}, raw={sorted(raw_by_pid)}"
        )
    expected_slots = list(enumerate(config.slot_tasks))
    if config.slot_tasks:
        for pid in raw_by_pid:
            actual_slots = [
                (slot, task) for slot, task, _stats in slots_by_pid.get(pid, [])
            ]
            if actual_slots != expected_slots:
                raise MetricParseError(
                    f"rank PID {pid} slot contract mismatch: "
                    f"expected {expected_slots}, got {actual_slots}"
                )
        unexpected_slot_pids = sorted(set(slots_by_pid) - set(raw_by_pid))
        if unexpected_slot_pids:
            raise MetricParseError(f"slot summaries have no raw rank: {unexpected_slot_pids}")
    elif slots_by_pid:
        raise MetricParseError("single-dispatch cases must not contain slot summaries")

    ordered_pids = sorted(raw_by_pid)
    primary_by_pid: dict[int, SampleStats] = {}
    cleanup_by_pid: dict[int, SampleStats] = {}
    expected_count = rounds * config.dispatches_per_round
    for pid in ordered_pids:
        raw = raw_by_pid[pid]
        if len(raw) != expected_count:
            raise MetricParseError(
                f"rank PID {pid} requires {expected_count} raw samples, found {len(raw)}"
            )
        if config.dispatches_per_round == 1:
            primary_by_pid[pid] = SampleStats(raw)
            _require_reported_stats(
                f"rank PID {pid}", primary_by_pid[pid], rank_summaries[pid]
            )
        else:
            primary_by_pid[pid] = SampleStats(raw[0::2])
            cleanup_by_pid[pid] = SampleStats(raw[1::2])
            _require_reported_stats(
                f"rank PID {pid} slot 0", primary_by_pid[pid], slots_by_pid[pid][0][2]
            )
            _require_reported_stats(
                f"rank PID {pid} slot 1", cleanup_by_pid[pid], slots_by_pid[pid][1][2]
            )

    selected_pid = min(ordered_pids, key=lambda pid: (primary_by_pid[pid].median, pid))
    selected_rank = ordered_pids.index(selected_pid)
    rank_metrics = []
    for logical_rank, (device_id, pid) in enumerate(zip(device_ids, ordered_pids, strict=True)):
        rank_metrics.append(
            RankMetric(
                logical_rank=logical_rank,
                device_id=device_id,
                pid=pid,
                slot=0,
                task=config.primary_task,
                selected=pid == selected_pid,
                stats=primary_by_pid[pid],
            )
        )
        if config.dispatches_per_round == 2:
            rank_metrics.append(
                RankMetric(
                    logical_rank=logical_rank,
                    device_id=device_id,
                    pid=pid,
                    slot=1,
                    task=config.slot_tasks[1],
                    selected=False,
                    stats=cleanup_by_pid[pid],
                )
            )

    cleanup_median = None
    if selected_pid in cleanup_by_pid:
        cleanup_median = cleanup_by_pid[selected_pid].median
    return MetricResult(
        case=case,
        metric_contract_version=config.metric_contract_version,
        rounds=rounds,
        warmup=warmup,
        dispatches_per_round=config.dispatches_per_round,
        selected_rank=selected_rank,
        selected_device=device_ids[selected_rank],
        selected_pid=selected_pid,
        selected_stats=primary_by_pid[selected_pid],
        cleanup_median_us=cleanup_median,
        baseline_median_us=config.baseline_median_us,
        metric_scope=config.metric_scope,
        metric_source=config.metric_source,
        validation_mode=config.validation_mode,
        validation_outputs=validation_outputs,
        rank_metrics=tuple(rank_metrics),
    )


def _append_rank_rows(path: Path, result: MetricResult) -> None:
    try:
        with path.open("a", encoding="utf-8") as output:
            for row in result.rank_tsv_rows():
                output.write(row + "\n")
    except OSError as error:
        raise MetricParseError(f"cannot append rank metrics to {path}: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASE_CONFIGS))
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--rounds", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=OFFICIAL_SEED)
    parser.add_argument(
        "--validation-profile",
        choices=sorted(VALIDATION_PROFILES),
        default="legacy",
    )
    parser.add_argument("--rank-output", type=Path)
    args = parser.parse_args(argv)

    try:
        result = parse_perf_log(
            args.case,
            args.log,
            rounds=args.rounds,
            warmup=args.warmup,
            device_set=args.device,
            seed=args.seed,
            validation_profile=args.validation_profile,
        )
        if args.rank_output is not None:
            _append_rank_rows(args.rank_output, result)
    except MetricParseError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(result.summary_tsv_fields())
    return 0


if __name__ == "__main__":
    sys.exit(main())
