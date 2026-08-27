# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CLI plumbing for post-Golden placement device-output A/B checks."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from device_output_ab import (
    ArtifactMetadata,
    CaptureSpec,
    DeviceOutputArtifactComparer,
    DeviceOutputArtifactWriter,
    sha256_file,
)


_PROGRAM_SOURCE_SCHEMA = b"dsv4-program-source-v1"
_PROGRAM_SOURCE_ROOTS = (
    Path("models/deepseek_v4_flash_mtp"),
    Path("golden"),
)
_IGNORED_SOURCE_DIRECTORIES = frozenset({"__pycache__", "build_output"})
_IGNORED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _update_source_hash(
    digest: "hashlib._Hash",
    identity: str,
    content: bytes,
) -> None:
    identity_bytes = identity.encode("utf-8")
    digest.update(len(identity_bytes).to_bytes(8, byteorder="big"))
    digest.update(identity_bytes)
    digest.update(len(content).to_bytes(8, byteorder="big"))
    digest.update(content)


def _program_source_files(repo_root: Path) -> tuple[Path, ...]:
    files = []
    for relative_root in _PROGRAM_SOURCE_ROOTS:
        source_root = repo_root / relative_root
        if source_root.is_symlink() or not source_root.is_dir():
            raise ValueError(f"program source root must be a non-symlink directory: {source_root}")
        root_files = []
        for path in source_root.rglob("*"):
            relative_path = path.relative_to(source_root)
            if any(part in _IGNORED_SOURCE_DIRECTORIES for part in relative_path.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"program source contains a symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"program source contains a non-regular entry: {path}")
            if path.suffix in _IGNORED_SOURCE_SUFFIXES:
                continue
            root_files.append(path)
        if not root_files:
            raise ValueError(f"program source root contains no source files: {source_root}")
        files.extend(root_files)
    return tuple(sorted(files, key=lambda path: path.relative_to(repo_root).as_posix()))


def program_source_sha256(
    entry_identity: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> str:
    """Hash exact program bytes and the selected repository entrypoint."""
    root_candidate = Path(repo_root) if repo_root is not None else Path(__file__).parents[2]
    if root_candidate.is_symlink() or not root_candidate.is_dir():
        raise ValueError(f"repository root must be a non-symlink directory: {root_candidate}")
    root = root_candidate.resolve(strict=True)

    entry_candidate = Path(entry_identity)
    if not entry_candidate.is_absolute():
        if ".." in entry_candidate.parts:
            raise ValueError(f"program entry identity must not escape the repository: {entry_identity}")
        entry_candidate = root / entry_candidate
    if entry_candidate.is_symlink() or not entry_candidate.is_file():
        raise ValueError(
            f"program entry must be an existing non-symlink file: {entry_candidate}"
        )
    entry_path = entry_candidate.resolve(strict=True)
    model_root = root / _PROGRAM_SOURCE_ROOTS[0]
    try:
        relative_entry = entry_path.relative_to(root)
        entry_path.relative_to(model_root)
    except ValueError as error:
        raise ValueError(f"program entry must be inside {model_root}: {entry_identity}") from error

    source_files = _program_source_files(root)
    if entry_path not in source_files:
        raise ValueError(f"program entry is not part of the source fingerprint: {entry_path}")

    digest = hashlib.sha256()
    _update_source_hash(digest, "schema", _PROGRAM_SOURCE_SCHEMA)
    _update_source_hash(
        digest,
        "entry",
        relative_entry.as_posix().encode("utf-8"),
    )
    for source_path in source_files:
        source_identity = source_path.relative_to(root).as_posix()
        _update_source_hash(digest, source_identity, source_path.read_bytes())
    return digest.hexdigest()


def add_device_output_arguments(
    parser: argparse.ArgumentParser,
    *,
    visible: bool = True,
) -> None:
    """Add optional immutable-reference artifact flags to one stats entrypoint."""
    parser.add_argument(
        "--save-device-output",
        default=None,
        help=(
            "write a post-Golden canonical device-output reference artifact"
            if visible
            else argparse.SUPPRESS
        ),
    )
    parser.add_argument(
        "--compare-device-output",
        default=None,
        help=(
            "compare post-Golden device outputs against a reference artifact"
            if visible
            else argparse.SUPPRESS
        ),
    )
    parser.add_argument(
        "--device-output-seed",
        type=int,
        default=None,
        help=(
            "fixture seed recorded by a device-output A/B artifact"
            if visible
            else argparse.SUPPRESS
        ),
    )


def build_device_output_callback(
    args: argparse.Namespace,
    *,
    case: str,
    placement: str,
    placement_manifest: str | Path,
    entry_identity: str | Path,
    topology: Mapping[str, int],
    captures: Sequence[CaptureSpec],
):
    """Return a write or compare callback, or ``None`` when A/B is disabled."""
    write_path = args.save_device_output
    compare_path = args.compare_device_output
    if write_path is None and compare_path is None:
        return None
    if write_path is not None and compare_path is not None:
        raise ValueError("--save-device-output and --compare-device-output are mutually exclusive")
    if args.device_output_seed is None:
        raise ValueError("--device-output-seed is required with a device-output A/B flag")
    if args.device_output_seed < 0:
        raise ValueError("--device-output-seed must be non-negative")

    manifest_path = Path(placement_manifest)
    manifest_digest = sha256_file(manifest_path)
    metadata = ArtifactMetadata(
        case=case,
        seed=args.device_output_seed,
        topology=topology,
        placement=placement,
        placement_manifest_sha256=manifest_digest,
        program_source_sha256=program_source_sha256(entry_identity),
    )
    if write_path is not None:
        return DeviceOutputArtifactWriter(
            write_path,
            metadata=metadata,
            captures=captures,
        )

    reference_path = Path(compare_path)
    if not reference_path.is_dir() or reference_path.is_symlink():
        raise ValueError(
            "--compare-device-output must name an existing non-symlink artifact directory"
        )
    return DeviceOutputArtifactComparer(
        reference_path,
        metadata=metadata,
        captures=captures,
    )
