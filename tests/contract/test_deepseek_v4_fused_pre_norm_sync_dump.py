# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-independent tests for the fused_pre_norm sync-workspace dump checker."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_PATH = (
    _REPO_ROOT
    / "models"
    / "deepseek"
    / "v4-flash"
    / "kernels"
    / "fused_pre_norm_cce"
    / "check_sync_dump.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "fused_pre_norm_sync_dump_checker",
    _CHECKER_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_CHECKER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CHECKER
_SPEC.loader.exec_module(_CHECKER)

_TASK_ID = "0x0000000100000002"
_EXPECTED_B1 = 4
_EXPECTED_B2 = 8


def _workspace_values(b1: int = 0, b2: int = 0) -> list[int]:
    values = [0] * 32
    values[0] = b1
    values[16] = b2
    return values


def _write_dump(
    tmp_path: Path,
    *,
    before_values: list[list[int]] | None = None,
    after_values: list[list[int]] | None = None,
) -> Path:
    dump_dir = tmp_path / "args_dump"
    dump_dir.mkdir()
    before_values = (
        [_workspace_values(), _workspace_values()]
        if before_values is None
        else before_values
    )
    after_values = (
        [_workspace_values(_EXPECTED_B1, _EXPECTED_B2)]
        if after_values is None
        else after_values
    )

    payload = bytearray()
    records = []
    for stage, snapshots in (
        ("before_dispatch", before_values),
        ("after_completion", after_values),
    ):
        for values in snapshots:
            assert len(values) == 32
            offset = len(payload)
            payload.extend(struct.pack("<32i", *values))
            records.append(
                {
                    "task_id": _TASK_ID,
                    "func_id": [0],
                    "arg_index": 12,
                    "role": "inout",
                    "stage": stage,
                    "kind": "tensor",
                    "dtype": "INT32",
                    "is_contiguous": True,
                    "shape": [32],
                    "strides": [1],
                    "start_offset": 0,
                    "numel": 32,
                    "bin_offset": offset,
                    "bin_size": 128,
                    "truncated": False,
                    "overwritten": False,
                },
            )

    manifest = {
        "run_dir": "args_dump",
        "bin_format": {
            "type": "logical_contiguous",
            "byte_order": "little_endian",
        },
        "total_args": len(records),
        "before_dispatch": len(before_values),
        "after_completion": len(after_values),
        "inout_args": len(records),
        "truncated_args": 0,
        "dropped_records": 0,
        "dropped_overwrite": 0,
        "bin_file": "args.bin",
        "args": records,
    }
    (dump_dir / "args.bin").write_bytes(payload)
    (dump_dir / "args_dump.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return dump_dir


def _read_manifest(dump_dir: Path) -> dict:
    return json.loads((dump_dir / "args_dump.json").read_text(encoding="utf-8"))


def _write_manifest(dump_dir: Path, manifest: dict) -> None:
    (dump_dir / "args_dump.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_checker_accepts_repeated_zero_before_records_and_exact_after(
    tmp_path: Path,
) -> None:
    dump_dir = _write_dump(tmp_path)
    summaries = _CHECKER.validate_sync_dump(
        dump_dir,
        _EXPECTED_B1,
        _EXPECTED_B2,
    )
    assert summaries == (
        _CHECKER.TaskSummary(
            task_id=_TASK_ID,
            before_records=2,
            after_records=1,
        ),
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(_CHECKER_PATH),
            str(dump_dir),
            "--expect-b1",
            str(_EXPECTED_B1),
            "--expect-b2",
            str(_EXPECTED_B2),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PASS" in completed.stdout
    assert "B1=4 B2=8" in completed.stdout


@pytest.mark.parametrize("field", ("dropped_records", "dropped_overwrite"))
def test_checker_rejects_dropped_dump_records(
    tmp_path: Path,
    field: str,
) -> None:
    dump_dir = _write_dump(tmp_path)
    manifest = _read_manifest(dump_dir)
    manifest[field] = 1
    _write_manifest(dump_dir, manifest)

    with pytest.raises(_CHECKER.DumpValidationError, match=field):
        _CHECKER.validate_sync_dump(dump_dir, _EXPECTED_B1, _EXPECTED_B2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("is_contiguous", False, "not logically contiguous"),
        ("start_offset", 1, "start_offset is not zero"),
        ("truncated", True, "is truncated"),
        ("overwritten", True, "is overwritten"),
    ),
)
def test_checker_rejects_invalid_record_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    dump_dir = _write_dump(tmp_path)
    manifest = _read_manifest(dump_dir)
    manifest["args"][0][field] = value
    _write_manifest(dump_dir, manifest)

    with pytest.raises(_CHECKER.DumpValidationError, match=message):
        _CHECKER.validate_sync_dump(dump_dir, _EXPECTED_B1, _EXPECTED_B2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("role", "input", "role is not inout"),
        ("kind", "scalar", "is not a tensor"),
        ("dtype", "FP32", "dtype is not INT32"),
        ("shape", [64], r"shape is not \[32\]"),
    ),
)
def test_checker_rejects_wrong_arg12_signature(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    dump_dir = _write_dump(tmp_path)
    manifest = _read_manifest(dump_dir)
    manifest["args"][0][field] = value
    _write_manifest(dump_dir, manifest)

    with pytest.raises(_CHECKER.DumpValidationError, match=message):
        _CHECKER.validate_sync_dump(dump_dir, _EXPECTED_B1, _EXPECTED_B2)


def test_checker_rejects_nonzero_before_dispatch_workspace(
    tmp_path: Path,
) -> None:
    invalid_before = _workspace_values()
    invalid_before[7] = 1
    dump_dir = _write_dump(tmp_path, before_values=[invalid_before])

    with pytest.raises(_CHECKER.DumpValidationError, match="is not all zero"):
        _CHECKER.validate_sync_dump(dump_dir, _EXPECTED_B1, _EXPECTED_B2)


@pytest.mark.parametrize(
    ("word", "value", "message"),
    (
        (0, 3, "B1/B2=3/8"),
        (16, 7, "B1/B2=4/7"),
        (1, 1, "nonzero padding words"),
        (31, -1, "nonzero padding words"),
    ),
)
def test_checker_rejects_wrong_after_counters_or_padding(
    tmp_path: Path,
    word: int,
    value: int,
    message: str,
) -> None:
    invalid_after = _workspace_values(_EXPECTED_B1, _EXPECTED_B2)
    invalid_after[word] = value
    dump_dir = _write_dump(tmp_path, after_values=[invalid_after])

    with pytest.raises(_CHECKER.DumpValidationError, match=message):
        _CHECKER.validate_sync_dump(dump_dir, _EXPECTED_B1, _EXPECTED_B2)


def test_checker_requires_both_dump_stages(tmp_path: Path) -> None:
    dump_dir = _write_dump(tmp_path, after_values=[])

    with pytest.raises(
        _CHECKER.DumpValidationError,
        match="has no after_completion record",
    ):
        _CHECKER.validate_sync_dump(dump_dir, _EXPECTED_B1, _EXPECTED_B2)


def test_checker_rejects_missing_binary_payload(tmp_path: Path) -> None:
    dump_dir = _write_dump(tmp_path)
    (dump_dir / "args.bin").unlink()

    with pytest.raises(_CHECKER.DumpValidationError, match="missing binary payload"):
        _CHECKER.validate_sync_dump(dump_dir, _EXPECTED_B1, _EXPECTED_B2)
