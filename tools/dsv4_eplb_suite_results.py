# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Create crash-safe machine-readable results for the DSV4 EPLB performance suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.dsv4_eplb_perf_metrics import (
    CASE_CONFIGS,
    MAPPING_BASIS,
    SELECTION_POLICY,
    MetricParseError,
    MetricResult,
    parse_perf_log,
)


SCHEMA_VERSION = 1
SUITE_ID = "dsv4-eplb-8card"
SUITE_CONTRACT_ID = "dsv4-eplb-suite-v2"
LANE_ID = "dsv4-eplb-branch"
CASE_ORDER = ("moe-ep8", "decode-logits", "mtp-core")
CASE_SPEC_HASHES = {
    "moe-ep8": "dsv4-moe-ep8-v1",
    "decode-logits": "dsv4-eplb-v2:decode-logits",
    "mtp-core": "dsv4-eplb-v2:mtp-core",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 15,
) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _output(command: Sequence[str], *, cwd: Path | None = None) -> str | None:
    result = _run(command, cwd=cwd)
    if result is None or result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace").strip() or None


def _git(repo: Path, *args: str) -> str | None:
    return _output(("git", "-C", str(repo), *args))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_journal(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError as error:
        raise ValueError(f"cannot read case journal {path}: {error}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL record at {path}:{line_number}: {error}") from error
        if not isinstance(record, dict) or not isinstance(record.get("case"), dict):
            raise ValueError(f"invalid case record at {path}:{line_number}")
        records.append(record)
    return records


def _parse_key_values(path: Path) -> dict[str, str] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    values = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values or None


def _direct_url_info(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    try:
        raw = distribution.read_text("direct_url.json")
        value = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return {"kind": "unavailable"}
    if not isinstance(value, dict):
        return {"kind": "unavailable"}
    parsed = urlparse(str(value.get("url", "")))
    artifact = Path(unquote(parsed.path)).name or None
    archive_info = value.get("archive_info")
    if isinstance(archive_info, dict):
        hashes = archive_info.get("hashes")
        sha256 = hashes.get("sha256") if isinstance(hashes, dict) else None
        return {"kind": "archive", "artifact": artifact, "sha256": sha256}
    vcs_info = value.get("vcs_info")
    if isinstance(vcs_info, dict):
        return {
            "kind": "vcs",
            "vcs": vcs_info.get("vcs"),
            "commit": vcs_info.get("commit_id"),
        }
    if "dir_info" in value:
        source_path = Path(unquote(parsed.path)) if parsed.scheme == "file" else None
        source_commit = _git(source_path, "rev-parse", "HEAD") if source_path else None
        source_tree = _git(source_path, "rev-parse", "HEAD^{tree}") if source_path else None
        return {"kind": "directory", "commit": source_commit, "tree": source_tree}
    return {"kind": "other", "artifact": artifact}


def _distribution_info(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"available": False}
    record = distribution.read_text("RECORD") or ""
    return {
        "available": True,
        "version": distribution.version,
        "record_sha256": _sha256_bytes(record.encode()),
        "source": _direct_url_info(distribution),
    }


def _selected_pypto_info() -> dict[str, Any]:
    installed = _distribution_info("pypto")
    root_value = os.environ.get("PYPTO_ROOT")
    if not root_value:
        return {"installed": installed, "checkout": {"available": False}}
    root = Path(root_value)
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all") or ""
    versions = _parse_key_values(root / "toolchain" / "versions.env") or {}
    try:
        pto_isa_pin = (root / "runtime" / "pto_isa.pin").read_text(encoding="utf-8").strip()
    except OSError:
        pto_isa_pin = None
    return {
        "installed": installed,
        "checkout": {
            "available": root.is_dir(),
            "commit": _git(root, "rev-parse", "HEAD"),
            "tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "status_sha256": _sha256_bytes(status.encode()),
            "dirty": bool(status),
            "simpler_pin": _git(root, "rev-parse", "HEAD:runtime"),
            "ptoas_version_pin": versions.get("PTOAS_VERSION"),
            "pto_isa_pin": pto_isa_pin,
        },
    }


def _resolve_ptoas() -> Path | None:
    root = os.environ.get("PTOAS_ROOT")
    if root:
        for relative in ("ptoas", "bin/ptoas"):
            candidate = Path(root) / relative
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
    executable = shutil.which("ptoas")
    return Path(executable).resolve() if executable else None


def _ptoas_info() -> dict[str, Any]:
    executable = _resolve_ptoas()
    if executable is None:
        return {"available": False}
    return {
        "available": True,
        "executable": executable.name,
        "version": _output((str(executable), "--version")),
        "sha256": _sha256_file(executable),
    }


def _pto_isa_info() -> dict[str, Any]:
    root = os.environ.get("PTO_ISA_ROOT")
    if not root:
        return {"available": False}
    path = Path(root)
    return {
        "available": True,
        "commit": _git(path, "rev-parse", "HEAD"),
        "tree": _git(path, "rev-parse", "HEAD^{tree}"),
        "status_sha256": _sha256_bytes((_git(path, "status", "--porcelain=v1") or "").encode()),
    }


def _cann_info() -> dict[str, Any]:
    root_value = next(
        (
            os.environ[name]
            for name in ("ASCEND_HOME_PATH", "CANN_ROOT", "ASCEND_HOME")
            if os.environ.get(name)
        ),
        None,
    )
    candidates = []
    if root_value is not None:
        root = Path(root_value)
        candidates.extend(
            (
                root / f"{platform.machine()}-linux" / "ascend_toolkit_install.info",
                root / "ascend_toolkit_install.info",
            )
        )
    candidates.extend(
        Path("/usr/local/Ascend").glob(
            f"cann-*/{platform.machine()}-linux/ascend_toolkit_install.info"
        )
    )
    candidates.extend(Path("/usr/local/Ascend").glob("cann-*/ascend_toolkit_install.info"))
    install_info = next((candidate for candidate in candidates if candidate.is_file()), None)
    values = _parse_key_values(install_info) if install_info else None
    return {
        "available": install_info is not None,
        "version": values.get("version") if values else None,
        "inner_version": values.get("innerversion") if values else None,
        "install_info_sha256": _sha256_file(install_info) if install_info else None,
    }


def _system_component(path: Path) -> dict[str, Any]:
    values = _parse_key_values(path)
    if values is None:
        return {"available": False}
    return {
        "available": True,
        "version": values.get("Version") or values.get("version"),
        "inner_version": values.get("Innerversion") or values.get("innerversion"),
        "sha256": _sha256_file(path),
    }


def _capture_npu_smi(output_dir: Path) -> dict[str, Any]:
    result = _run(("npu-smi", "info"), timeout=30)
    if result is None:
        return {"available": False, "returncode": None, "snapshot_sha256": None}
    snapshot = output_dir / "npu-smi-info.txt"
    snapshot_written = False
    try:
        with snapshot.open("wb") as output:
            output.write(result.stdout)
            output.flush()
            os.fsync(output.fileno())
        snapshot_written = True
    except OSError:
        pass
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "snapshot_sha256": _sha256_bytes(result.stdout),
        "snapshot": "npu-smi-info.txt" if snapshot_written else None,
    }


def _physical_mapping(device_ids: list[int]) -> tuple[list[dict[str, Any]], str]:
    raw = os.environ.get("PYPTO_DEVICE_MAPPING_JSON")
    if not raw:
        raise ValueError(
            "PYPTO_DEVICE_MAPPING_JSON is required for a measured suite"
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"PYPTO_DEVICE_MAPPING_JSON is not valid JSON: {error}") from error
    if not isinstance(value, list) or len(value) != len(device_ids):
        raise ValueError(
            "PYPTO_DEVICE_MAPPING_JSON must be an ordered list with "
            f"{len(device_ids)} entries"
        )
    entries = []
    for logical_rank, (requested_device_id, entry) in enumerate(
        zip(device_ids, value, strict=True)
    ):
        if not isinstance(entry, dict):
            raise ValueError(
                f"PYPTO_DEVICE_MAPPING_JSON entry {logical_rank} must be an object"
            )
        expected = {
            "logical_rank": logical_rank,
            "requested_device_id": requested_device_id,
        }
        for field, expected_value in expected.items():
            if entry.get(field) != expected_value:
                raise ValueError(
                    f"PYPTO_DEVICE_MAPPING_JSON entry {logical_rank} requires "
                    f"{field}={expected_value}"
                )
        physical_device_id = entry.get("physical_device_id")
        serial = entry.get("serial")
        if isinstance(physical_device_id, bool) or not isinstance(physical_device_id, int):
            raise ValueError(
                f"PYPTO_DEVICE_MAPPING_JSON entry {logical_rank} requires an integer "
                "physical_device_id"
            )
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError(
                f"PYPTO_DEVICE_MAPPING_JSON entry {logical_rank} requires a non-empty serial"
            )
        entries.append(
            {
                "logical_rank": logical_rank,
                "requested_device_id": requested_device_id,
                "physical_device_id": physical_device_id,
                "serial": serial,
            }
        )
    physical_ids = [entry["physical_device_id"] for entry in entries]
    serials = [entry["serial"] for entry in entries]
    if len(set(physical_ids)) != len(physical_ids) or len(set(serials)) != len(serials):
        raise ValueError("PYPTO_DEVICE_MAPPING_JSON physical IDs and serials must be unique")
    return entries, "PYPTO_DEVICE_MAPPING_JSON"


def _source_info(repo: Path) -> dict[str, Any]:
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    branch = _git(repo, "branch", "--show-current")
    tracking_ref = _git(repo, "rev-parse", "--abbrev-ref", "@{upstream}")
    upstream_ref = next(
        (candidate for candidate in ("upstream/main", "origin/main") if _git(repo, "rev-parse", candidate)),
        None,
    )
    upstream_commit = _git(repo, "rev-parse", upstream_ref) if upstream_ref else None
    merge_base = _git(repo, "merge-base", "HEAD", upstream_ref) if upstream_ref else None
    ahead = None
    behind = None
    if upstream_ref:
        counts = _git(repo, "rev-list", "--left-right", "--count", f"HEAD...{upstream_ref}")
        if counts:
            parts = counts.split()
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                ahead, behind = (int(part) for part in parts)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all") or ""
    patch = b""
    if merge_base:
        result = _run(("git", "-C", str(repo), "diff", "--binary", merge_base, "HEAD"))
        if result is not None and result.returncode == 0:
            patch = result.stdout
    return {
        "commit": commit,
        "tip_sha": commit,
        "tree": tree,
        "tree_sha": tree,
        "branch": branch,
        "branch_ref": branch,
        "tracking_ref": tracking_ref,
        "tracking_commit": _git(repo, "rev-parse", tracking_ref) if tracking_ref else None,
        "upstream_main_ref": upstream_ref,
        "upstream_main_commit": upstream_commit,
        "merge_base": merge_base,
        "ahead": ahead,
        "behind": behind,
        "patchset_sha256": _sha256_bytes(patch) if merge_base else None,
        "status_sha256": _sha256_bytes(status.encode()),
        "status_porcelain": status.splitlines(),
        "dirty": bool(status),
    }


def _contract_spec(rounds: int, warmup: int, seed: int, devices: list[int]) -> dict[str, Any]:
    cases = {}
    for case_id in CASE_ORDER:
        config = CASE_CONFIGS[case_id]
        cases[case_id] = {
            "contract_id": config.metric_contract_version,
            "spec_hash": CASE_SPEC_HASHES[case_id],
            "kernel": config.kernel,
            "metric_scope": config.metric_scope,
            "selection_policy": SELECTION_POLICY,
            "dispatches_per_round": config.dispatches_per_round,
            "validation_mode": config.validation_mode,
            "required_outputs": list(config.required_validation_outputs),
        }
    return {
        "suite_contract_id": SUITE_CONTRACT_ID,
        "case_order": list(CASE_ORDER),
        "devices": devices,
        "rounds": rounds,
        "warmup": warmup,
        "seed": seed,
        "cases": cases,
    }


def _build_context(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    device_ids = [int(part) for part in args.device.split(",")]
    physical_mapping, physical_mapping_source = _physical_mapping(device_ids)
    spec = _contract_spec(args.rounds, args.warmup, args.seed, device_ids)
    canonical_spec = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    context = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "suite_contract_id": SUITE_CONTRACT_ID,
        "lane_id": LANE_ID,
        "suite_contract": {
            "id": SUITE_CONTRACT_ID,
            "spec_sha256": _sha256_bytes(canonical_spec),
            "cases": spec["cases"],
        },
        "started_at": args.started_at,
        "source": _source_info(repo),
        "provenance": {
            "epoch": os.environ.get("PYPTO_TOOLCHAIN_EPOCH") or "unassigned",
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": Path(sys.executable).name,
                "environment": Path(sys.prefix).name,
            },
            "pypto": _selected_pypto_info(),
            "simpler": _distribution_info("simpler"),
            "torch": _distribution_info("torch"),
            "torch_npu": _distribution_info("torch-npu"),
            "ptoas": _ptoas_info(),
            "pto_isa": _pto_isa_info(),
            "cann": _cann_info(),
            "driver": _system_component(Path("/usr/local/Ascend/driver/version.info")),
            "firmware": _system_component(Path("/usr/local/Ascend/firmware/version.info")),
            "host": {
                "hostname": socket.gethostname(),
                "kernel": platform.release(),
                "architecture": platform.machine(),
            },
            "npu_smi": _capture_npu_smi(output_dir),
        },
        "devices": {
            "epoch": os.environ.get("PYPTO_DEVICE_EPOCH") or "unassigned",
            "platform": "a2a3",
            "ordered": device_ids,
            "mapping_basis": MAPPING_BASIS,
            "task_device": os.environ.get("TASK_DEVICE"),
            "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "physical_mapping": physical_mapping,
            "physical_mapping_available": bool(physical_mapping),
            "physical_mapping_source": physical_mapping_source,
        },
        "sampling": {
            "rounds": args.rounds,
            "warmup": args.warmup,
            "seed": args.seed,
        },
        "benchmark_environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYPTO_BENCH": "1",
            "PYPTO_BENCH_RAW": "1",
            "PYPTO_BENCH_ROUNDS": str(args.rounds),
            "PYPTO_BENCH_WARMUP": str(args.warmup),
            "PYPTO_RUNTIME_LOG": "error",
            "PYTHONHASHSEED": str(args.seed),
            "SIMPLER_DEVICE_STRACE_ENABLE": "1",
            "PTO2_RING_TASK_WINDOW": str(args.ring_task_window),
            "PTO2_RING_DEP_POOL": str(args.ring_dep_pool),
            "PTO2_RING_HEAP": str(args.ring_heap),
        },
    }
    return context


def _validation_from_result(result: MetricResult) -> dict[str, Any]:
    status = "runtime_only" if result.validation_mode == "runtime_only" else "pass"
    config = CASE_CONFIGS[result.case]
    return {
        "status": status,
        "mode": result.validation_mode,
        "required_outputs": list(config.required_validation_outputs),
        "outputs": {name: value.lower() for name, value in result.validation_outputs},
    }


def _metric_from_result(result: MetricResult) -> dict[str, Any]:
    ranks = []
    for rank in result.rank_metrics:
        ranks.append(
            {
                "logical_rank": rank.logical_rank,
                "device_id": rank.device_id,
                "pid": rank.pid,
                "slot": rank.slot,
                "task": rank.task,
                "selected": rank.selected,
                "samples": len(rank.stats.samples),
                "min_us": rank.stats.minimum,
                "median_us": rank.stats.median,
                "mean_us": rank.stats.mean,
                "max_us": rank.stats.maximum,
            }
        )
    return {
        "status": "valid",
        "scope": result.metric_scope,
        "selection_policy": SELECTION_POLICY,
        "source": result.metric_source,
        "mapping_basis": MAPPING_BASIS,
        "dispatches_per_round": result.dispatches_per_round,
        "selected_rank": result.selected_rank,
        "selected_device": result.selected_device,
        "selected_pid": result.selected_pid,
        "summary": {
            "samples": len(result.selected_stats.samples),
            "min_us": result.selected_stats.minimum,
            "median_us": result.selected_stats.median,
            "mean_us": result.selected_stats.mean,
            "max_us": result.selected_stats.maximum,
        },
        "cleanup_median_us": result.cleanup_median_us,
        "baseline_median_us": result.baseline_median_us,
        "delta_us": result.delta_us,
        "delta_pct": result.delta_pct,
        "ranks": ranks,
    }


def _failed_validation(case_id: str, log_path: Path, process_rc: int) -> dict[str, Any]:
    config = CASE_CONFIGS[case_id]
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    outputs = {}
    for line in lines:
        if not line.startswith("[RUN]") or "'" not in line:
            continue
        for status in ("PASS", "FAIL"):
            marker = f"' {status}"
            if marker in line:
                name = line.split("'", 2)[1]
                outputs[name] = status.lower()
    if any(value == "fail" for value in outputs.values()) or process_rc != 0:
        status = "fail"
    else:
        status = "not_evaluated"
    return {
        "status": status,
        "mode": config.validation_mode,
        "required_outputs": list(config.required_validation_outputs),
        "outputs": outputs,
    }


def _relative_log_info(log_path: Path, output_dir: Path) -> dict[str, Any]:
    try:
        relative = str(log_path.resolve().relative_to(output_dir.resolve()))
    except (OSError, ValueError):
        relative = log_path.name
    try:
        size = log_path.stat().st_size
    except OSError:
        size = None
    return {"path": relative, "sha256": _sha256_file(log_path), "bytes": size}


def _case_result(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    config = CASE_CONFIGS[args.case]
    output_dir = args.context.parent
    seed = context["sampling"]["seed"] if config.required_seed is not None else None
    metric_result = None
    metric_error = None
    try:
        metric_result = parse_perf_log(
            args.case,
            args.log,
            rounds=context["sampling"]["rounds"],
            warmup=context["sampling"]["warmup"],
            device_set=",".join(str(value) for value in context["devices"]["ordered"]),
            seed=seed,
        )
    except MetricParseError as error:
        metric_error = str(error)

    if metric_result is not None and args.process_rc == 0:
        status = "pass"
    elif metric_result is not None:
        status = "metric_valid_execution_failed"
    elif args.process_rc == 0:
        status = "invalid_metric"
    else:
        status = "fail"
    validation = (
        _validation_from_result(metric_result)
        if metric_result is not None
        else _failed_validation(args.case, args.log, args.process_rc)
    )
    metric = (
        _metric_from_result(metric_result)
        if metric_result is not None
        else {"status": "invalid", "error": metric_error}
    )
    return {
        "case_id": args.case,
        "contract_id": config.metric_contract_version,
        "spec_hash": CASE_SPEC_HASHES[args.case],
        "status": status,
        "process_rc": args.process_rc,
        "started_at": args.started_at,
        "finished_at": args.finished_at,
        "command": args.command_arg,
        "validation": validation,
        "metric": metric,
        "log": _relative_log_info(args.log, output_dir),
    }


def _latest_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = {record["case"]["case_id"]: record["case"] for record in records}
    return [latest[case_id] for case_id in CASE_ORDER if case_id in latest]


def _write_suite_result(
    context: dict[str, Any],
    records: list[dict[str, Any]],
    output: Path,
    *,
    status: str,
    finished_at: str | None,
) -> None:
    value = dict(context)
    cases = _latest_cases(records)
    selected_case_ids = [case["case_id"] for case in cases]
    value["status"] = status
    value["finished_at"] = finished_at
    value["coverage"] = {
        "required_case_ids": list(CASE_ORDER),
        "selected_case_ids": selected_case_ids,
        "complete": selected_case_ids == list(CASE_ORDER),
    }
    value["cases"] = cases
    _atomic_json(output, value)


def _init(args: argparse.Namespace) -> int:
    context = _build_context(args)
    _atomic_json(args.context, context)
    _write_suite_result(context, [], args.suite_output, status="running", finished_at=None)
    return 0


def _record(args: argparse.Namespace) -> int:
    context = _read_json(args.context)
    case = _case_result(args, context)
    record = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": context["suite_id"],
        "suite_contract_id": context["suite_contract_id"],
        "lane_id": context["lane_id"],
        "source": context["source"],
        "provenance": context["provenance"],
        "devices": context["devices"],
        "sampling": context["sampling"],
        "observed_at": args.finished_at,
        "case": case,
    }
    _append_jsonl(args.journal, record)
    records = _read_journal(args.journal)
    _write_suite_result(context, records, args.suite_output, status="running", finished_at=None)
    median = case["metric"].get("summary", {}).get("median_us")
    print(f"case={case['case_id']} status={case['status']} median_us={median}")
    return 0 if case["status"] == "pass" else 1


def _finalize(args: argparse.Namespace) -> int:
    context = _read_json(args.context)
    records = _read_journal(args.journal)
    cases = _latest_cases(records)
    complete = [case["case_id"] for case in cases] == list(CASE_ORDER)
    source_at_start = context["source"]
    source_at_finish = _source_info(args.repo.resolve())
    source_stable = all(
        source_at_finish.get(field) == source_at_start.get(field)
        for field in ("commit", "tree")
    )
    source_clean = not source_at_finish["dirty"]
    context["source"] = source_at_finish
    passed = (
        complete
        and all(case["status"] == "pass" for case in cases)
        and source_stable
        and source_clean
    )
    status = "pass" if passed else "fail" if complete else "incomplete"
    _write_suite_result(
        context,
        records,
        args.suite_output,
        status=status,
        finished_at=args.finished_at,
    )
    return 0 if passed else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--repo", required=True, type=Path)
    init_parser.add_argument("--output-dir", required=True, type=Path)
    init_parser.add_argument("--context", required=True, type=Path)
    init_parser.add_argument("--suite-output", required=True, type=Path)
    init_parser.add_argument("--device", required=True)
    init_parser.add_argument("--rounds", required=True, type=int)
    init_parser.add_argument("--warmup", required=True, type=int)
    init_parser.add_argument("--seed", required=True, type=int)
    init_parser.add_argument("--ring-task-window", required=True, type=int)
    init_parser.add_argument("--ring-dep-pool", required=True, type=int)
    init_parser.add_argument("--ring-heap", required=True, type=int)
    init_parser.add_argument("--started-at", required=True)
    init_parser.set_defaults(handler=_init)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--context", required=True, type=Path)
    record_parser.add_argument("--journal", required=True, type=Path)
    record_parser.add_argument("--suite-output", required=True, type=Path)
    record_parser.add_argument("--case", required=True, choices=CASE_ORDER)
    record_parser.add_argument("--log", required=True, type=Path)
    record_parser.add_argument("--process-rc", required=True, type=int)
    record_parser.add_argument("--started-at", required=True)
    record_parser.add_argument("--finished-at", required=True)
    record_parser.add_argument("--command-arg", action="append", default=[])
    record_parser.set_defaults(handler=_record)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--repo", required=True, type=Path)
    finalize_parser.add_argument("--context", required=True, type=Path)
    finalize_parser.add_argument("--journal", required=True, type=Path)
    finalize_parser.add_argument("--suite-output", required=True, type=Path)
    finalize_parser.add_argument("--finished-at", required=True)
    finalize_parser.set_defaults(handler=_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
