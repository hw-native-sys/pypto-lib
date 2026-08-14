# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parse and assemble the DeepSeek-V4 main-attention performance suite."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse


SCHEMA_VERSION = 1
_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
_SEED_RE = re.compile(
    r"^\[PERF\] deterministic_seed seed=(?P<seed>\d+) "
    r"python_hash_seed=(?P<hash_seed>\d+)$"
)
_HEADLINE_RE = re.compile(
    rf"^\[RUN\]\s+effective_us \((?P<rounds>\d+) rounds\) "
    rf"min=(?P<minimum>{_NUMBER}) median=(?P<median>{_NUMBER}) "
    rf"mean=(?P<mean>{_NUMBER}) max=(?P<maximum>{_NUMBER})$"
)
_RAW_HEADER_RE = re.compile(
    r"^\[RUN\]\s+raw samples: ranks=(?P<ranks>\d+) "
    r"rounds=(?P<rounds>\d+) warmup=(?P<warmup>\d+)$"
)
_RAW_RANK_RE = re.compile(
    r"^\[RUN\]\s+rank (?P<pid>\d+) raw n=(?P<count>\d+) "
    r"eff_us=(?P<samples>\[.*\])$"
)
_VALIDATION_RE = re.compile(
    r"^\[RUN\]\s+'(?P<name>[^']+)' (?P<status>PASS|FAIL)\s+"
    r"shape=(?P<shape>\([^)]*\)) dtype=(?P<dtype>[^\s]+)(?: \(.*\))?$"
)
_FINAL_PASS_RE = re.compile(rf"^\[RUN\] PASS \({_NUMBER}s\)$")
_FORBIDDEN_FRAGMENTS = (
    "fallback_flattened=1",
    "effective_us unavailable",
    "benchmark unavailable",
    "benchmark skipped",
    "validation skipped",
    "Traceback (most recent call last)",
)


class MetricParseError(ValueError):
    """Raised when a log cannot prove the official metric contract."""


@dataclass(frozen=True)
class SampleStats:
    samples: tuple[float, ...]

    def as_dict(self) -> dict[str, float | int]:
        return {
            "sample_count": len(self.samples),
            "minimum": min(self.samples),
            "median": statistics.median(self.samples),
            "mean": statistics.fmean(self.samples),
            "maximum": max(self.samples),
        }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate the checked-in suite contract."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricParseError(f"cannot load suite manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise MetricParseError("suite manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise MetricParseError(
            f"unsupported manifest schema_version={manifest.get('schema_version')!r}"
        )
    for key in ("suite_id", "lane_id", "contract_id", "platform", "ordered_devices"):
        if key not in manifest:
            raise MetricParseError(f"suite manifest is missing {key!r}")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise MetricParseError("suite manifest sampling must be an object")
    for key in ("seed", "rounds", "warmup", "raw"):
        if key not in sampling:
            raise MetricParseError(f"suite manifest sampling is missing {key!r}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise MetricParseError("suite manifest cases must be a non-empty list")
    case_ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or len(set(case_ids)) != len(case_ids):
        raise MetricParseError("suite manifest case IDs must be present and unique")
    return manifest


def _case_config(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [case for case in manifest["cases"] if case["case_id"] == case_id]
    if len(matches) != 1:
        raise MetricParseError(f"unknown suite case: {case_id}")
    return matches[0]


def _single_match(lines: Sequence[str], pattern: re.Pattern[str], label: str) -> tuple[int, re.Match[str]]:
    matches = [(index, match) for index, line in enumerate(lines) if (match := pattern.match(line))]
    if len(matches) != 1:
        raise MetricParseError(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


def _parse_samples(payload: str, declared_count: int) -> tuple[float, ...]:
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError) as error:
        raise MetricParseError("raw effective-time samples are malformed") from error
    if not isinstance(parsed, list):
        raise MetricParseError("raw effective-time samples must be a list")
    if len(parsed) != declared_count:
        raise MetricParseError(
            f"raw sample line declares n={declared_count} but contains {len(parsed)} samples"
        )
    samples = []
    for index, value in enumerate(parsed):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricParseError(f"raw sample {index} is not numeric")
        sample = float(value)
        if not math.isfinite(sample) or sample <= 0.0:
            raise MetricParseError(f"raw sample {index} must be finite and positive")
        samples.append(sample)
    return tuple(samples)


def _require_reported_stats(stats: SampleStats, match: re.Match[str]) -> None:
    actual = stats.as_dict()
    for name in ("minimum", "median", "mean", "maximum"):
        reported = float(match.group(name))
        if not math.isclose(float(actual[name]), reported, rel_tol=0.0, abs_tol=0.11):
            raise MetricParseError(
                f"effective_us {name} mismatch: raw={actual[name]:.3f}, summary={reported:.3f}"
            )


def parse_case_log(
    log_text: str,
    *,
    manifest: dict[str, Any],
    case_id: str,
    device_id: int,
    process_rc: int = 0,
) -> dict[str, Any]:
    """Parse one case log and return a contract-valid case result."""

    case = _case_config(manifest, case_id)
    sampling = manifest["sampling"]
    expected_devices = manifest["ordered_devices"]
    if expected_devices != [device_id]:
        raise MetricParseError(
            f"official metrics require ordered devices {expected_devices}, got {[device_id]}"
        )
    if process_rc != 0:
        raise MetricParseError(f"benchmark process exited with status {process_rc}")

    lines = log_text.splitlines()
    for fragment in _FORBIDDEN_FRAGMENTS:
        if any(fragment in line for line in lines):
            raise MetricParseError(f"benchmark log contains forbidden marker {fragment!r}")
    if any(line.startswith("[RUN] FAIL") for line in lines):
        raise MetricParseError("benchmark log contains a final FAIL result")

    seed_index, seed_match = _single_match(lines, _SEED_RE, "deterministic-seed sentinel")
    expected_seed = int(sampling["seed"])
    if (int(seed_match.group("seed")), int(seed_match.group("hash_seed"))) != (
        expected_seed,
        expected_seed,
    ):
        raise MetricParseError(f"deterministic seed must be {expected_seed}")

    headline_index, headline = _single_match(lines, _HEADLINE_RE, "effective_us headline")
    raw_header_index, raw_header = _single_match(lines, _RAW_HEADER_RE, "raw-sample header")
    raw_rank_index, raw_rank = _single_match(lines, _RAW_RANK_RE, "raw rank-sample line")
    final_index, _final = _single_match(lines, _FINAL_PASS_RE, "final PASS line")
    run_line_indices = [index for index, line in enumerate(lines) if line.startswith("[RUN]")]
    if not run_line_indices or final_index != run_line_indices[-1]:
        raise MetricParseError("final PASS must be the last [RUN] line")

    expected_rounds = int(sampling["rounds"])
    expected_warmup = int(sampling["warmup"])
    headline_rounds = int(headline.group("rounds"))
    raw_contract = (
        int(raw_header.group("ranks")),
        int(raw_header.group("rounds")),
        int(raw_header.group("warmup")),
    )
    if headline_rounds != expected_rounds:
        raise MetricParseError(
            f"effective_us headline requires {expected_rounds} rounds, got {headline_rounds}"
        )
    if raw_contract != (1, expected_rounds, expected_warmup):
        raise MetricParseError(
            "raw-sample header mismatch: expected ranks/rounds/warmup="
            f"{(1, expected_rounds, expected_warmup)}, got {raw_contract}"
        )

    samples = _parse_samples(raw_rank.group("samples"), int(raw_rank.group("count")))
    if len(samples) != expected_rounds:
        raise MetricParseError(f"expected {expected_rounds} raw samples, found {len(samples)}")
    stats = SampleStats(samples)
    _require_reported_stats(stats, headline)

    validation_matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := _VALIDATION_RE.match(line))
    ]
    expected_outputs = case["validation_outputs"]
    expected_by_name = {output["name"]: output for output in expected_outputs}
    actual_by_name: dict[str, tuple[int, re.Match[str]]] = {}
    for index, match in validation_matches:
        name = match.group("name")
        if name in actual_by_name:
            raise MetricParseError(f"duplicate validation result for {name!r}")
        actual_by_name[name] = (index, match)
    if set(actual_by_name) != set(expected_by_name):
        raise MetricParseError(
            "validation output mismatch: "
            f"expected={sorted(expected_by_name)}, got={sorted(actual_by_name)}"
        )

    validation_outputs = []
    validation_indices = []
    for expected in expected_outputs:
        index, match = actual_by_name[expected["name"]]
        validation_indices.append(index)
        if match.group("status") != "PASS":
            raise MetricParseError(f"validation failed for {expected['name']!r}")
        try:
            shape = ast.literal_eval(match.group("shape"))
        except (SyntaxError, ValueError) as error:
            raise MetricParseError(f"invalid validation shape for {expected['name']!r}") from error
        if tuple(shape) != tuple(expected["shape"]):
            raise MetricParseError(
                f"validation shape mismatch for {expected['name']!r}: "
                f"expected={tuple(expected['shape'])}, got={shape}"
            )
        if match.group("dtype") != expected["dtype"]:
            raise MetricParseError(
                f"validation dtype mismatch for {expected['name']!r}: "
                f"expected={expected['dtype']}, got={match.group('dtype')}"
            )
        validation_outputs.append(
            {
                "name": expected["name"],
                "status": "pass",
                "shape": list(shape),
                "dtype": match.group("dtype"),
            }
        )

    if not (
        seed_index < headline_index < raw_header_index < raw_rank_index
        < min(validation_indices) <= max(validation_indices) < final_index
    ):
        raise MetricParseError("benchmark sections are out of contract order")

    metric = {
        "name": case["metric"]["name"],
        "unit": case["metric"]["unit"],
        "aggregation": case["metric"]["aggregation"],
        **stats.as_dict(),
    }
    rank_diagnostic = {
        "logical_rank": 0,
        "device_id": device_id,
        "pid": int(raw_rank.group("pid")),
        **stats.as_dict(),
        "samples_us": list(samples),
    }
    return {
        "case_id": case_id,
        "status": "pass",
        "process_rc": process_rc,
        "contract_id": f"{manifest['contract_id']}:{case_id}",
        "entrypoint": case["entrypoint"],
        "metric_us": metric["median"],
        "metric": metric,
        "validation": {"status": "pass", "outputs": validation_outputs},
        "rank_diagnostics": [rank_diagnostic],
    }


def build_case_result(
    log_text: str,
    *,
    manifest: dict[str, Any],
    case_id: str,
    device_id: int,
    process_rc: int,
) -> tuple[dict[str, Any], MetricParseError | None]:
    """Return a success or explicit failure record for one case."""

    try:
        return (
            parse_case_log(
                log_text,
                manifest=manifest,
                case_id=case_id,
                device_id=device_id,
                process_rc=process_rc,
            ),
            None,
        )
    except MetricParseError as error:
        case = _case_config(manifest, case_id)
        status = "fail" if process_rc else "invalid_metric"
        return (
            {
                "case_id": case_id,
                "status": status,
                "process_rc": process_rc,
                "contract_id": f"{manifest['contract_id']}:{case_id}",
                "entrypoint": case["entrypoint"],
                "metric_us": None,
                "metric": None,
                "validation": {"status": "fail", "outputs": [], "error": str(error)},
                "rank_diagnostics": [],
            },
            error,
        )


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, value: Any) -> None:
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


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _repo_for_path(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _module_sha(module_name: str) -> str:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return "unavailable"
    if spec is None:
        return "unavailable"
    locations = list(spec.submodule_search_locations or ())
    origin = None if spec.origin in (None, "built-in") else Path(spec.origin)
    probe = origin.parent if origin is not None else (Path(locations[0]) if locations else None)
    if probe is None or (repo := _repo_for_path(probe.resolve())) is None:
        return "unavailable"
    return _git(repo, "rev-parse", "HEAD") or "unavailable"


def _path_repo_sha(path_value: str | None) -> str:
    if not path_value:
        return "unavailable"
    path = Path(path_value)
    if not path.exists() or (repo := _repo_for_path(path.resolve())) is None:
        return "unavailable"
    return _git(repo, "rev-parse", "HEAD") or "unavailable"


def _pto_isa_provenance() -> dict[str, Any]:
    root_value = os.environ.get("PTO_ISA_ROOT")
    if not root_value:
        return {"available": False}
    root = Path(root_value)
    status = _git(root, "status", "--porcelain=v1") or ""
    return {
        "available": root.is_dir(),
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "clean": status == "",
    }


def _direct_url_info(distribution: importlib.metadata.Distribution) -> dict[str, Any]:
    try:
        raw = distribution.read_text("direct_url.json")
        value = json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return {"kind": "unavailable"}
    if not isinstance(value, dict):
        return {"kind": "unavailable"}
    parsed = urlparse(str(value.get("url", "")))
    archive_info = value.get("archive_info")
    if isinstance(archive_info, dict):
        hashes = archive_info.get("hashes")
        return {
            "kind": "archive",
            "artifact": Path(unquote(parsed.path)).name or None,
            "sha256": hashes.get("sha256") if isinstance(hashes, dict) else None,
        }
    vcs_info = value.get("vcs_info")
    if isinstance(vcs_info, dict):
        return {
            "kind": "vcs",
            "vcs": vcs_info.get("vcs"),
            "commit": vcs_info.get("commit_id"),
        }
    if "dir_info" in value and parsed.scheme == "file":
        source_path = Path(unquote(parsed.path))
        return {
            "kind": "directory",
            "commit": _git(source_path, "rev-parse", "HEAD"),
            "tree": _git(source_path, "rev-parse", "HEAD^{tree}"),
        }
    return {"kind": "unavailable"}


def _distribution_info(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"available": False}
    record = distribution.read_text("RECORD") or ""
    return {
        "available": True,
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        "source": _direct_url_info(distribution),
    }


def _distribution_commit(info: dict[str, Any], module_name: str) -> str:
    source = info.get("source")
    if isinstance(source, dict) and source.get("commit"):
        return str(source["commit"])
    return _module_sha(module_name)


def _ptoas_provenance() -> dict[str, Any]:
    candidates = []
    if root := os.environ.get("PTOAS_ROOT"):
        candidates.extend((Path(root) / "ptoas", Path(root) / "bin" / "ptoas"))
    if executable := shutil.which("ptoas"):
        candidates.append(Path(executable))
    for candidate in candidates:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        try:
            version = subprocess.run(
                [str(candidate), "--version"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0]
            checksum = hashlib.sha256(candidate.read_bytes()).hexdigest()
            return {
                "available": True,
                "executable": candidate.name,
                "version": version,
                "sha256": checksum,
            }
        except (OSError, subprocess.CalledProcessError, IndexError):
            continue
    return {"available": False}


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


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _cann_provenance() -> dict[str, Any]:
    root_value = next(
        (
            os.environ[name]
            for name in ("ASCEND_HOME_PATH", "CANN_ROOT", "ASCEND_HOME")
            if os.environ.get(name)
        ),
        None,
    )
    candidates = []
    if root_value:
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
    install_info = next((path for path in candidates if path.is_file()), None)
    values = _parse_key_values(install_info) if install_info else None
    return {
        "available": install_info is not None,
        "version": values.get("version") if values else None,
        "inner_version": values.get("innerversion") if values else None,
        "install_info_sha256": _sha256_file(install_info) if install_info else None,
    }


def _system_component(path: Path) -> dict[str, Any]:
    values = _parse_key_values(path)
    return {
        "available": values is not None,
        "version": (values.get("Version") or values.get("version")) if values else None,
        "inner_version": (
            values.get("Innerversion") or values.get("innerversion")
        ) if values else None,
        "sha256": _sha256_file(path),
    }


def _capture_npu_smi(output_dir: Path | None) -> dict[str, Any]:
    if output_dir is None:
        return {"available": False, "returncode": None, "snapshot_sha256": None}
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "returncode": None, "snapshot_sha256": None}
    snapshot = output_dir / "npu-smi-info.txt"
    snapshot_written = False
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
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
        "snapshot_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "snapshot": snapshot.name if snapshot_written else None,
    }


def _physical_mapping(device_ids: list[int]) -> tuple[list[dict[str, Any]], str]:
    raw = os.environ.get("PYPTO_DEVICE_MAPPING_JSON")
    if not raw:
        return [], "unavailable"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise MetricParseError(
            f"PYPTO_DEVICE_MAPPING_JSON is not valid JSON: {error}"
        ) from error
    if not isinstance(value, list) or len(value) != len(device_ids):
        raise MetricParseError(
            "PYPTO_DEVICE_MAPPING_JSON must be an ordered list with "
            f"{len(device_ids)} entries"
        )
    entries = []
    for logical_rank, (requested_device_id, entry) in enumerate(
        zip(device_ids, value, strict=True)
    ):
        if not isinstance(entry, dict):
            raise MetricParseError(
                f"PYPTO_DEVICE_MAPPING_JSON entry {logical_rank} must be an object"
            )
        expected = {
            "logical_rank": logical_rank,
            "requested_device_id": requested_device_id,
        }
        for field, expected_value in expected.items():
            if entry.get(field) != expected_value:
                raise MetricParseError(
                    f"PYPTO_DEVICE_MAPPING_JSON entry {logical_rank} requires "
                    f"{field}={expected_value}"
                )
        physical_device_id = entry.get("physical_device_id")
        serial = entry.get("serial")
        if isinstance(physical_device_id, bool) or not isinstance(
            physical_device_id, int
        ):
            raise MetricParseError(
                f"PYPTO_DEVICE_MAPPING_JSON entry {logical_rank} requires an integer "
                "physical_device_id"
            )
        if not isinstance(serial, str) or not serial.strip():
            raise MetricParseError(
                f"PYPTO_DEVICE_MAPPING_JSON entry {logical_rank} requires a non-empty "
                "serial"
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
        raise MetricParseError(
            "PYPTO_DEVICE_MAPPING_JSON physical IDs and serials must be unique"
        )
    return entries, "PYPTO_DEVICE_MAPPING_JSON"


def _source_provenance(repo_root: Path) -> dict[str, Any]:
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all") or ""
    status_bytes = status.encode("utf-8")
    branch = _git(repo_root, "branch", "--show-current")
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "branch": branch or None,
        "clean": status == "",
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "status_porcelain": status.splitlines(),
    }


def _toolchain_provenance() -> dict[str, Any]:
    pypto_info = _distribution_info("pypto")
    simpler_info = _distribution_info("simpler")
    return {
        "epoch": os.environ.get("PYPTO_TOOLCHAIN_EPOCH", "unassigned"),
        "python": sys.version.split()[0],
        "pypto_sha": _distribution_commit(pypto_info, "pypto"),
        "simpler_sha": _distribution_commit(simpler_info, "simpler"),
        "pto_isa_sha": _path_repo_sha(os.environ.get("PTO_ISA_ROOT")),
        "pypto": pypto_info,
        "simpler": simpler_info,
        "torch": _distribution_info("torch"),
        "torch_npu": _distribution_info("torch-npu"),
        "pto_isa": _pto_isa_provenance(),
        "ptoas": _ptoas_provenance(),
        "cann": _cann_provenance(),
        "driver": _system_component(Path("/usr/local/Ascend/driver/version.info")),
        "firmware": _system_component(Path("/usr/local/Ascend/firmware/version.info")),
        "host": {
            "hostname": socket.gethostname(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
        },
    }


def assemble_suite_result(
    *,
    manifest: dict[str, Any],
    repo_root: Path,
    device_id: int,
    started_at_utc: str,
    finished_at_utc: str,
    cases: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Assemble the stable tracker-facing suite result."""

    expected_devices = manifest["ordered_devices"]
    if expected_devices != [device_id]:
        raise MetricParseError(
            f"official metrics require ordered devices {expected_devices}, got {[device_id]}"
        )
    known_case_ids = [case["case_id"] for case in manifest["cases"]]
    actual_case_ids = [case.get("case_id") for case in cases]
    if not actual_case_ids or len(set(actual_case_ids)) != len(actual_case_ids):
        raise MetricParseError("suite result case IDs must be present and unique")
    if any(case_id not in known_case_ids for case_id in actual_case_ids):
        raise MetricParseError(f"suite result contains unknown cases: {actual_case_ids}")

    canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    complete = set(actual_case_ids) == set(known_case_ids)
    suite_status = (
        "pass" if complete and all(case.get("status") == "pass" for case in cases) else "fail"
    )
    physical_mapping, physical_mapping_source = _physical_mapping([device_id])
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": manifest["suite_id"],
        "lane_id": manifest["lane_id"],
        "contract_id": manifest["contract_id"],
        "contract": {
            "id": manifest["contract_id"],
            "spec_hash": hashlib.sha256(canonical_manifest).hexdigest(),
        },
        "status": suite_status,
        "coverage": {
            "complete": complete,
            "required_case_ids": known_case_ids,
            "selected_case_ids": actual_case_ids,
        },
        "source": _source_provenance(repo_root),
        "toolchain": _toolchain_provenance(),
        "device": {
            "epoch": os.environ.get("PYPTO_DEVICE_EPOCH", "unassigned"),
            "platform": manifest["platform"],
            "ordered_devices": [device_id],
            "mapping_basis": "ordered_logical_rank_to_ordered_device_set",
            "task_device": os.environ.get("TASK_DEVICE"),
            "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "npu_smi": _capture_npu_smi(output_dir),
            "physical_mapping": physical_mapping,
            "physical_mapping_available": bool(physical_mapping),
            "physical_mapping_source": physical_mapping_source,
        },
        "sampling": {
            **manifest["sampling"],
            "python_hash_seed": manifest["sampling"]["seed"],
        },
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "cases": cases,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetricParseError(f"cannot load JSON {path}: {error}") from error


def _parse_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    try:
        log_text = args.log.read_text(encoding="utf-8")
    except OSError as error:
        log_text = ""
        process_rc = args.process_rc or 1
        read_error = MetricParseError(f"cannot read benchmark log {args.log}: {error}")
    else:
        process_rc = args.process_rc
        read_error = None
    result, error = build_case_result(
        log_text,
        manifest=manifest,
        case_id=args.case,
        device_id=args.device,
        process_rc=process_rc,
    )
    if read_error is not None:
        result["validation"]["error"] = str(read_error)
        error = read_error
    _json_dump(args.output, result)
    if args.journal is not None:
        _append_jsonl(args.journal, result)
    if error is not None:
        print(f"ERROR: {args.case}: {error}", file=sys.stderr)
        return 1
    return 0


def _suite_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    cases = [_read_json(path) for path in args.case_result]
    result = assemble_suite_result(
        manifest=manifest,
        repo_root=args.repo_root.resolve(),
        device_id=args.device,
        started_at_utc=args.started_at_utc,
        finished_at_utc=args.finished_at_utc,
        cases=cases,
        output_dir=args.output.parent.resolve(),
    )
    _json_dump(args.output, result)
    return 0 if result["status"] == "pass" else 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="parse one case log")
    parse_parser.add_argument("--manifest", type=Path, required=True)
    parse_parser.add_argument("--case", required=True)
    parse_parser.add_argument("--log", type=Path, required=True)
    parse_parser.add_argument("--device", type=int, required=True)
    parse_parser.add_argument("--process-rc", type=int, required=True)
    parse_parser.add_argument("--output", type=Path, required=True)
    parse_parser.add_argument("--journal", type=Path)
    parse_parser.set_defaults(handler=_parse_command)

    suite_parser = subparsers.add_parser("suite", help="assemble suite-result.json")
    suite_parser.add_argument("--manifest", type=Path, required=True)
    suite_parser.add_argument("--repo-root", type=Path, required=True)
    suite_parser.add_argument("--device", type=int, required=True)
    suite_parser.add_argument("--started-at-utc", default=_utc_now())
    suite_parser.add_argument("--finished-at-utc", default=_utc_now())
    suite_parser.add_argument("--case-result", type=Path, action="append", required=True)
    suite_parser.add_argument("--output", type=Path, required=True)
    suite_parser.set_defaults(handler=_suite_command)

    args = parser.parse_args()
    try:
        return args.handler(args)
    except MetricParseError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
