# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate fused_pre_norm atomic soft-SYNCALL workspace argument dumps.

Example:

    python check_sync_dump.py /path/to/dfx_outputs/args_dump \
        --expect-b1 4 --expect-b2 8
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_MANIFEST_NAME = "args_dump.json"
_SYNC_ARG_INDEX = 12
_SYNC_WORDS = 32
_COUNTER_WORDS = (0, 16)
_PAYLOAD_BYTES = _SYNC_WORDS * struct.calcsize("<i")


class DumpValidationError(RuntimeError):
    """Raised when an argument dump cannot prove the sync-workspace contract."""


@dataclass(frozen=True)
class TaskSummary:
    """Validated record counts for one fused task."""

    task_id: str
    before_records: int
    after_records: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DumpValidationError(message)


def _load_manifest(dump_dir: Path) -> tuple[dict[str, Any], bytes]:
    manifest_path = dump_dir / _MANIFEST_NAME
    _require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DumpValidationError(
            f"cannot read manifest {manifest_path}: {error}",
        ) from error
    _require(isinstance(manifest, dict), "manifest root must be a JSON object")

    for field in ("dropped_records", "dropped_overwrite"):
        _require(field in manifest, f"manifest is missing {field}")
        value = manifest[field]
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"manifest {field} must be an integer, got {value!r}",
        )
        _require(value == 0, f"manifest reports {field}={value}")

    bin_format = manifest.get("bin_format")
    _require(isinstance(bin_format, dict), "manifest is missing bin_format")
    _require(
        bin_format.get("type") == "logical_contiguous",
        "manifest payload must use logical_contiguous format",
    )
    _require(
        bin_format.get("byte_order") == "little_endian",
        "manifest payload must use little_endian byte order",
    )

    bin_file = manifest.get("bin_file")
    _require(
        isinstance(bin_file, str) and bool(bin_file),
        "manifest has no binary payload file",
    )
    relative_bin_path = Path(bin_file)
    _require(
        not relative_bin_path.is_absolute() and ".." not in relative_bin_path.parts,
        f"manifest bin_file must stay under dump_dir, got {bin_file!r}",
    )
    resolved_dump_dir = dump_dir.resolve()
    payload_path = (dump_dir / relative_bin_path).resolve()
    _require(
        payload_path.is_relative_to(resolved_dump_dir),
        f"manifest bin_file escapes dump_dir: {bin_file!r}",
    )
    _require(payload_path.is_file(), f"missing binary payload: {payload_path}")
    try:
        payload = payload_path.read_bytes()
    except OSError as error:
        raise DumpValidationError(
            f"cannot read binary payload {payload_path}: {error}",
        ) from error
    _require(payload, f"binary payload is empty: {payload_path}")
    return manifest, payload


def _sync_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_records = manifest.get("args", manifest.get("tensors"))
    _require(isinstance(raw_records, list), "manifest has no argument record list")
    _require(
        all(isinstance(record, dict) for record in raw_records),
        "manifest argument records must be JSON objects",
    )
    matching_records = [
        record
        for record in raw_records
        if record.get("kind", "tensor") == "tensor"
        and record.get("role") == "inout"
        and record.get("arg_index") == _SYNC_ARG_INDEX
        and isinstance(record.get("dtype"), str)
        and record["dtype"].upper() == "INT32"
        and record.get("shape") == [_SYNC_WORDS]
    ]
    _require(
        bool(matching_records),
        "no fused sync_workspace arg12 INOUT INT32[32] records",
    )
    target_task_ids = set()
    for record in matching_records:
        task_id = record.get("task_id")
        _require(
            isinstance(task_id, str) and bool(task_id),
            "matching fused sync_workspace record has no task_id",
        )
        target_task_ids.add(task_id)

    records = []
    for record in raw_records:
        task_id = record.get("task_id")
        if (
            not isinstance(task_id, str)
            or task_id not in target_task_ids
            or record.get("arg_index") != _SYNC_ARG_INDEX
        ):
            continue
        _require(
            record.get("kind", "tensor") == "tensor",
            f"task {task_id} arg12 is not a tensor",
        )
        _require(
            record.get("role") == "inout",
            f"task {task_id} arg12 role is not inout",
        )
        dtype = record.get("dtype")
        _require(
            isinstance(dtype, str) and dtype.upper() == "INT32",
            f"task {task_id} arg12 dtype is not INT32",
        )
        _require(
            record.get("shape") == [_SYNC_WORDS],
            f"task {task_id} arg12 shape is not [32]",
        )
        records.append(record)
    return records


def _record_values(
    record: dict[str, Any],
    payload: bytes,
    *,
    record_index: int,
) -> tuple[int, ...]:
    prefix = f"record {record_index}"
    _require(
        record.get("is_contiguous") is True,
        f"{prefix} is not logically contiguous",
    )
    _require(record.get("start_offset") == 0, f"{prefix} start_offset is not zero")
    _require(record.get("numel") == _SYNC_WORDS, f"{prefix} numel is not 32")
    _require(record.get("strides") == [1], f"{prefix} strides are not [1]")
    _require(record.get("truncated") is False, f"{prefix} is truncated")
    _require(record.get("overwritten") is False, f"{prefix} is overwritten")

    bin_offset = record.get("bin_offset")
    bin_size = record.get("bin_size")
    _require(
        isinstance(bin_offset, int)
        and not isinstance(bin_offset, bool)
        and bin_offset >= 0
        and bin_offset % struct.calcsize("<i") == 0,
        f"{prefix} has invalid bin_offset={bin_offset!r}",
    )
    _require(
        isinstance(bin_size, int)
        and not isinstance(bin_size, bool)
        and bin_size >= _PAYLOAD_BYTES,
        f"{prefix} has invalid bin_size={bin_size!r}",
    )
    _require(
        bin_offset + bin_size <= len(payload),
        f"{prefix} payload range exceeds binary payload size",
    )
    return struct.unpack_from(f"<{_SYNC_WORDS}i", payload, bin_offset)


def _validate_expected_count(value: int, name: str) -> None:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= 8,
        f"{name} must be an integer in [0, 8], got {value!r}",
    )


def validate_sync_dump(
    dump_dir: str | Path,
    expected_b1: int,
    expected_b2: int,
) -> tuple[TaskSummary, ...]:
    """Validate all fused sync-workspace tasks in one args-dump directory."""
    _validate_expected_count(expected_b1, "expected_b1")
    _validate_expected_count(expected_b2, "expected_b2")
    dump_path = Path(dump_dir)
    manifest, payload = _load_manifest(dump_path)
    records = _sync_records(manifest)

    by_task: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        task_id = record.get("task_id")
        _require(
            isinstance(task_id, str) and bool(task_id),
            f"record {index} has no task_id",
        )
        by_task[task_id].append((index, record))

    zero_values = (0,) * _SYNC_WORDS
    expected_values = [0] * _SYNC_WORDS
    expected_values[_COUNTER_WORDS[0]] = expected_b1
    expected_values[_COUNTER_WORDS[1]] = expected_b2
    expected_after = tuple(expected_values)
    summaries = []

    for task_id in sorted(by_task):
        task_records = by_task[task_id]
        before = [
            (index, record)
            for index, record in task_records
            if record.get("stage") == "before_dispatch"
        ]
        after = [
            (index, record)
            for index, record in task_records
            if record.get("stage") == "after_completion"
        ]
        known_stages = {"before_dispatch", "after_completion"}
        unexpected_stages = {
            record.get("stage")
            for _, record in task_records
            if record.get("stage") not in known_stages
        }
        _require(
            not unexpected_stages,
            f"task {task_id} has unexpected stages: {sorted(map(str, unexpected_stages))}",
        )
        _require(bool(before), f"task {task_id} has no before_dispatch record")
        _require(bool(after), f"task {task_id} has no after_completion record")

        for index, record in before:
            values = _record_values(record, payload, record_index=index)
            _require(
                values == zero_values,
                f"task {task_id} before_dispatch record {index} is not all zero",
            )

        for index, record in after:
            values = _record_values(record, payload, record_index=index)
            actual_b1 = values[_COUNTER_WORDS[0]]
            actual_b2 = values[_COUNTER_WORDS[1]]
            _require(
                (actual_b1, actual_b2) == (expected_b1, expected_b2),
                f"task {task_id} after_completion record {index} has "
                f"B1/B2={actual_b1}/{actual_b2}, "
                f"expected {expected_b1}/{expected_b2}",
            )
            bad_padding = [
                word
                for word, value in enumerate(values)
                if word not in _COUNTER_WORDS and value != 0
            ]
            _require(
                not bad_padding,
                f"task {task_id} after_completion record {index} has "
                f"nonzero padding words {bad_padding}",
            )
            _require(
                values == expected_after,
                f"task {task_id} after_completion record {index} "
                "does not match the expected workspace",
            )

        summaries.append(
            TaskSummary(
                task_id=task_id,
                before_records=len(before),
                after_records=len(after),
            ),
        )

    return tuple(summaries)


def _participant_count(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from error
    if not 0 <= value <= 8:
        raise argparse.ArgumentTypeError(f"must be in [0, 8], got {value}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--expect-b1", type=_participant_count, required=True)
    parser.add_argument("--expect-b2", type=_participant_count, required=True)
    args = parser.parse_args(argv)

    try:
        summaries = validate_sync_dump(
            args.dump_dir,
            args.expect_b1,
            args.expect_b2,
        )
    except DumpValidationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    for summary in summaries:
        print(
            f"PASS {summary.task_id}: "
            f"before={summary.before_records} after={summary.after_records} "
            f"B1={args.expect_b1} B2={args.expect_b2}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
