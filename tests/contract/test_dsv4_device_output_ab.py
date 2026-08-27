# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek_v4_flash_mtp"
sys.path.insert(0, str(_MODEL_DIR))

from device_output_ab import (  # noqa: E402
    ArtifactMetadata,
    DenseOutputCapture,
    DeviceOutputArtifactComparer,
    DeviceOutputArtifactError,
    DeviceOutputArtifactWriter,
    DeviceOutputMismatchError,
    MappedPoolCapture,
    RowwiseThresholds,
    compare_device_output_artifact,
    write_device_output_artifact,
)
from stats_placement_device_output import (  # noqa: E402
    add_device_output_arguments,
    build_device_output_callback,
    program_source_sha256,
)


_MANIFEST_DIGEST = "a" * 64
_PROGRAM_SOURCE_DIGEST = "c" * 64
_THRESHOLDS = RowwiseThresholds(
    min_cosine=0.999,
    max_rel_l2=0.01,
    max_abs=0.05,
)


def _metadata(placement: str = "contiguous", **overrides) -> ArtifactMetadata:
    values = {
        "case": "decode-logits",
        "seed": 1807,
        "topology": {
            "ep_size": 8,
            "tp_size": 4,
            "experts_per_rank": 32,
        },
        "placement": placement,
        "placement_manifest_sha256": _MANIFEST_DIGEST,
        "program_source_sha256": _PROGRAM_SOURCE_DIGEST,
    }
    values.update(overrides)
    return ArtifactMetadata(**values)


def _captures() -> tuple[DenseOutputCapture, MappedPoolCapture]:
    return (
        DenseOutputCapture(
            logical_key="decode.hidden",
            source_name="hidden_out",
            row_axes=2,
            expected_rows=4,
            thresholds=_THRESHOLDS,
        ),
        MappedPoolCapture(
            logical_key="decode.compressed-cache",
            source_name="cmp_kv",
            block_size=2,
            layer_ids=(2, 3),
            mapping_names=("csa_mapping", "hca_mapping"),
            expected_rows=8,
            thresholds=_THRESHOLDS,
        ),
    )


def _tensors() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    outputs = {
        "hidden_out": torch.arange(24, dtype=torch.float32).reshape(2, 2, 6) / 10,
        "cmp_kv": torch.arange(48, dtype=torch.float32).reshape(2, 4, 2, 3) / 10,
        "ignored_large_output": torch.zeros(100_000, dtype=torch.float32),
        # Validation inputs take priority over same-named entries here.
        "csa_mapping": torch.full((2, 3), -1, dtype=torch.int64),
    }
    validation_inputs = {
        "csa_mapping": torch.tensor([[1, -1, 3], [0, 2, -1]], dtype=torch.int64),
        "hca_mapping": torch.tensor([[-1, 0, 2], [3, -1, 1]], dtype=torch.int64),
    }
    return outputs, validation_inputs


def _write_reference(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    path = tmp_path / "reference-ab"
    outputs, validation_inputs = _tensors()
    writer = DeviceOutputArtifactWriter(
        path,
        metadata=_metadata(),
        captures=_captures(),
        emit_report=False,
    )
    summary = writer(outputs, validation_inputs)
    assert summary.placement == "contiguous"
    assert dict(summary.row_counts) == {
        "decode.compressed-cache": 8,
        "decode.hidden": 4,
    }
    return path, outputs, validation_inputs


def test_cli_builder_requires_seed_and_builds_immutable_writer(tmp_path: Path) -> None:
    parser = ArgumentParser()
    add_device_output_arguments(parser)
    args = parser.parse_args(
        [
            "--save-device-output",
            str(tmp_path / "reference"),
            "--device-output-seed",
            "1807",
        ]
    )
    manifest_path = tmp_path / "placement.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    callback = build_device_output_callback(
        args,
        case="decode-logits",
        placement="contiguous",
        placement_manifest=manifest_path,
        entry_identity="models/deepseek_v4_flash_mtp/stats_placement_decode_logits.py",
        topology={"ep_size": 8, "tp_size": 4},
        captures=_captures(),
    )

    assert isinstance(callback, DeviceOutputArtifactWriter)
    assert callback.metadata.seed == 1807
    assert callback.metadata.placement_manifest_sha256 == hashlib.sha256(
        b"{}\n"
    ).hexdigest()
    assert callback.metadata.program_source_sha256 == program_source_sha256(
        "models/deepseek_v4_flash_mtp/stats_placement_decode_logits.py"
    )

    missing_seed = parser.parse_args(["--save-device-output", str(tmp_path / "missing-seed")])
    with pytest.raises(ValueError, match="device-output-seed"):
        build_device_output_callback(
            missing_seed,
            case="decode-logits",
            placement="contiguous",
            placement_manifest=manifest_path,
            entry_identity="models/deepseek_v4_flash_mtp/stats_placement_decode_logits.py",
            topology={"ep_size": 8, "tp_size": 4},
            captures=_captures(),
        )


def _rewrite_manifest(path: Path, mutate) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    data = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(data)
    (path / "manifest.sha256").write_text(
        hashlib.sha256(data).hexdigest() + "\n",
        encoding="ascii",
    )


def test_writer_saves_only_selected_dense_and_active_cache_rows(tmp_path: Path) -> None:
    path, _, _ = _write_reference(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["format"] == "dsv4-device-output-ab"
    assert manifest["version"] == 2
    assert manifest["metadata"]["placement"] == "contiguous"
    assert manifest["metadata"]["program_source_sha256"] == _PROGRAM_SOURCE_DIGEST
    assert manifest["logical_keys"] == [
        "decode.compressed-cache",
        "decode.hidden",
    ]
    by_key = {entry["logical_key"]: entry for entry in manifest["tensors"]}
    assert by_key["decode.compressed-cache"]["values"]["shape"] == [8, 3]
    assert by_key["decode.hidden"]["values"]["shape"] == [4, 6]
    assert "ignored_large_output" not in (path / "manifest.json").read_text(encoding="utf-8")
    cache_payload = path / by_key["decode.compressed-cache"]["values"]["file"]
    assert cache_payload.stat().st_size == 8 * 3 * torch.tensor([], dtype=torch.float32).element_size()


def test_identical_candidate_returns_rowwise_metrics_and_both_placements(tmp_path: Path) -> None:
    path, outputs, validation_inputs = _write_reference(tmp_path)
    comparer = DeviceOutputArtifactComparer(
        path,
        metadata=_metadata("stats"),
        captures=_captures(),
        emit_report=False,
    )

    result = comparer(outputs, validation_inputs)

    assert result.passed
    assert result.reference_placement == "contiguous"
    assert result.candidate_placement == "stats"
    assert [tensor.logical_key for tensor in result.tensors] == [
        "decode.compressed-cache",
        "decode.hidden",
    ]
    assert all(row.cosine == pytest.approx(1.0) for tensor in result.tensors for row in tensor.rows)
    assert all(row.rel_l2 == pytest.approx(0.0) for tensor in result.tensors for row in tensor.rows)
    assert all(row.max_abs == pytest.approx(0.0) for tensor in result.tensors for row in tensor.rows)
    report = result.format_report()
    assert "reference=contiguous candidate=stats status=PASS" in report
    assert "key=decode.compressed-cache rows=8 status=PASS" in report


def test_fixed_threshold_failure_is_reported_and_callback_raises(tmp_path: Path) -> None:
    path, outputs, validation_inputs = _write_reference(tmp_path)
    candidate = dict(outputs)
    candidate["hidden_out"] = outputs["hidden_out"].clone()
    candidate["hidden_out"][1, 0, 2] += 1.0

    result = compare_device_output_artifact(
        path,
        metadata=_metadata("stats"),
        captures=_captures(),
        device_outputs=candidate,
        validation_inputs=validation_inputs,
    )

    assert not result.passed
    hidden = next(tensor for tensor in result.tensors if tensor.logical_key == "decode.hidden")
    failed = [row for row in hidden.rows if not row.passed]
    assert [row.row_key for row in failed] == [(1, 0)]
    assert failed[0].cosine is not None
    assert failed[0].rel_l2 > _THRESHOLDS.max_rel_l2
    assert failed[0].max_abs == pytest.approx(1.0)
    assert "max_abs" in failed[0].detail

    comparer = DeviceOutputArtifactComparer(
        path,
        metadata=_metadata("stats"),
        captures=_captures(),
        emit_report=False,
    )
    with pytest.raises(DeviceOutputMismatchError, match="decode.hidden"):
        comparer(candidate, validation_inputs)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_writer_and_comparer_reject_nonfinite_rows(tmp_path: Path, nonfinite: float) -> None:
    outputs, validation_inputs = _tensors()
    bad_reference = dict(outputs)
    bad_reference["hidden_out"] = outputs["hidden_out"].clone()
    bad_reference["hidden_out"][0, 0, 0] = nonfinite
    with pytest.raises(DeviceOutputArtifactError, match="contains NaN or Inf"):
        write_device_output_artifact(
            tmp_path / "bad-reference",
            metadata=_metadata(),
            captures=_captures(),
            device_outputs=bad_reference,
            validation_inputs=validation_inputs,
        )

    path, outputs, validation_inputs = _write_reference(tmp_path)
    candidate = dict(outputs)
    candidate["hidden_out"] = outputs["hidden_out"].clone()
    candidate["hidden_out"][0, 0, 0] = nonfinite
    result = compare_device_output_artifact(
        path,
        metadata=_metadata("stats"),
        captures=_captures(),
        device_outputs=candidate,
        validation_inputs=validation_inputs,
    )
    assert not result.passed
    failed = result.tensors[1].rows[0]
    assert failed.cosine is None
    assert failed.rel_l2 is None
    assert failed.max_abs is None
    assert failed.detail == "candidate row contains NaN or Inf"


@pytest.mark.parametrize(
    "overrides",
    [
        {"case": "mtp-core"},
        {"seed": 1808},
        {"topology": {"ep_size": 8, "tp_size": 8, "experts_per_rank": 32}},
        {"placement_manifest_sha256": "b" * 64},
        {"program_source_sha256": "b" * 64},
    ],
)
def test_comparer_allows_only_placement_metadata_to_differ(tmp_path: Path, overrides) -> None:
    path, outputs, validation_inputs = _write_reference(tmp_path)
    with pytest.raises(DeviceOutputArtifactError, match="metadata mismatch"):
        compare_device_output_artifact(
            path,
            metadata=_metadata("stats", **overrides),
            captures=_captures(),
            device_outputs=outputs,
            validation_inputs=validation_inputs,
        )


def test_comparer_rejects_stale_program_source_before_capture(tmp_path: Path) -> None:
    path, _, _ = _write_reference(tmp_path)
    with pytest.raises(DeviceOutputArtifactError, match="program_source_sha256"):
        compare_device_output_artifact(
            path,
            metadata=_metadata("stats", program_source_sha256="b" * 64),
            captures=_captures(),
            device_outputs={},
        )


def test_program_source_digest_binds_dirty_bytes_and_entry_identity(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "deepseek_v4_flash_mtp"
    golden_dir = tmp_path / "golden"
    model_dir.mkdir(parents=True)
    golden_dir.mkdir()
    first_entry = model_dir / "first.py"
    second_entry = model_dir / "second.py"
    implementation = model_dir / "implementation.py"
    first_entry.write_text("from implementation import run\n", encoding="utf-8")
    second_entry.write_text("from implementation import run\n", encoding="utf-8")
    implementation.write_text("def run():\n    return 1\n", encoding="utf-8")
    golden_runner = golden_dir / "runner.py"
    golden_runner.write_text("def validate():\n    return True\n", encoding="utf-8")

    first_identity = "models/deepseek_v4_flash_mtp/first.py"
    initial = program_source_sha256(first_identity, repo_root=tmp_path)
    assert program_source_sha256(first_entry, repo_root=tmp_path) == initial

    generated_dir = model_dir / "build_output" / "generated"
    generated_dir.mkdir(parents=True)
    (generated_dir / "kernel.py").write_text("generated = True\n", encoding="utf-8")
    assert program_source_sha256(first_identity, repo_root=tmp_path) == initial

    implementation.write_text("def run():\n    return 2\n", encoding="utf-8")
    dirty_source = program_source_sha256(first_identity, repo_root=tmp_path)
    assert dirty_source != initial
    golden_runner.write_text("def validate():\n    return False\n", encoding="utf-8")
    dirty_golden = program_source_sha256(first_identity, repo_root=tmp_path)
    assert dirty_golden != dirty_source
    assert program_source_sha256(
        "models/deepseek_v4_flash_mtp/second.py",
        repo_root=tmp_path,
    ) != dirty_golden


def test_program_source_digest_fails_closed_when_a_source_root_is_missing(tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "deepseek_v4_flash_mtp"
    model_dir.mkdir(parents=True)
    (model_dir / "entry.py").write_text("pass\n", encoding="utf-8")

    with pytest.raises(ValueError, match="program source root"):
        program_source_sha256(
            "models/deepseek_v4_flash_mtp/entry.py",
            repo_root=tmp_path,
        )


def test_comparer_rejects_missing_or_extra_logical_capture_keys(tmp_path: Path) -> None:
    path, outputs, validation_inputs = _write_reference(tmp_path)
    dense, mapped = _captures()
    extra = DenseOutputCapture(
        logical_key="decode.extra",
        source_name="hidden_out",
        row_axes=2,
        expected_rows=4,
        thresholds=_THRESHOLDS,
    )

    with pytest.raises(DeviceOutputArtifactError, match="logical keys mismatch"):
        compare_device_output_artifact(
            path,
            metadata=_metadata("stats"),
            captures=(dense,),
            device_outputs=outputs,
            validation_inputs=validation_inputs,
        )
    with pytest.raises(DeviceOutputArtifactError, match="logical keys mismatch"):
        compare_device_output_artifact(
            path,
            metadata=_metadata("stats"),
            captures=(dense, mapped, extra),
            device_outputs=outputs,
            validation_inputs=validation_inputs,
        )


def test_comparer_rejects_changed_logical_mapping_rows(tmp_path: Path) -> None:
    path, outputs, validation_inputs = _write_reference(tmp_path)
    changed = dict(validation_inputs)
    changed["csa_mapping"] = torch.tensor(
        [[-1, 1, 3], [0, 2, -1]],
        dtype=torch.int64,
    )

    with pytest.raises(DeviceOutputArtifactError, match="row keys changed"):
        compare_device_output_artifact(
            path,
            metadata=_metadata("stats"),
            captures=_captures(),
            device_outputs=outputs,
            validation_inputs=changed,
        )


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        (torch.tensor([[1, -2, 3], [0, 2, -1]]), "only -1"),
        (torch.tensor([[1, -1, 1], [0, 2, -1]]), "duplicate active rows"),
        (torch.tensor([[1, -1, 4], [0, 2, -1]]), "out of range"),
    ],
)
def test_mapped_capture_rejects_invalid_mapping_contract(
    tmp_path: Path,
    mapping: torch.Tensor,
    message: str,
) -> None:
    outputs, validation_inputs = _tensors()
    validation_inputs["csa_mapping"] = mapping
    with pytest.raises(DeviceOutputArtifactError, match=message):
        write_device_output_artifact(
            tmp_path / "invalid-mapping",
            metadata=_metadata(),
            captures=_captures(),
            device_outputs=outputs,
            validation_inputs=validation_inputs,
        )


def test_static_mapping_fallback_keeps_callback_single_argument(tmp_path: Path) -> None:
    outputs, validation_inputs = _tensors()
    outputs = {key: value for key, value in outputs.items() if key != "csa_mapping"}
    writer = DeviceOutputArtifactWriter(
        tmp_path / "static-mapping",
        metadata=_metadata(),
        captures=_captures(),
        static_tensors=validation_inputs,
        emit_report=False,
    )

    summary = writer(outputs)

    assert dict(summary.row_counts)["decode.compressed-cache"] == 8


def test_writer_refuses_to_overwrite_an_existing_artifact(tmp_path: Path) -> None:
    path, outputs, validation_inputs = _write_reference(tmp_path)
    with pytest.raises(DeviceOutputArtifactError, match="already exists"):
        write_device_output_artifact(
            path,
            metadata=_metadata(),
            captures=_captures(),
            device_outputs=outputs,
            validation_inputs=validation_inputs,
        )


def test_payload_digest_and_exact_file_set_are_fail_closed(tmp_path: Path) -> None:
    path, outputs, validation_inputs = _write_reference(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    payload = path / manifest["tensors"][0]["values"]["file"]
    data = bytearray(payload.read_bytes())
    data[0] ^= 1
    payload.write_bytes(data)
    with pytest.raises(DeviceOutputArtifactError, match="payload digest mismatch"):
        compare_device_output_artifact(
            path,
            metadata=_metadata("stats"),
            captures=_captures(),
            device_outputs=outputs,
            validation_inputs=validation_inputs,
        )

    path2, outputs, validation_inputs = _write_reference(tmp_path / "second")
    (path2 / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(DeviceOutputArtifactError, match="file set mismatch"):
        compare_device_output_artifact(
            path2,
            metadata=_metadata("stats"),
            captures=_captures(),
            device_outputs=outputs,
            validation_inputs=validation_inputs,
        )


def test_unknown_artifact_version_is_rejected_even_with_valid_digest(tmp_path: Path) -> None:
    path, outputs, validation_inputs = _write_reference(tmp_path)
    _rewrite_manifest(path, lambda manifest: manifest.__setitem__("version", 3))

    with pytest.raises(DeviceOutputArtifactError, match="version must be 2"):
        compare_device_output_artifact(
            path,
            metadata=_metadata("stats"),
            captures=_captures(),
            device_outputs=outputs,
            validation_inputs=validation_inputs,
        )
