# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host checks for the Qwen3.8 whole-checkpoint loader and text contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.qwen3_8_27b import reference, weights
from models.qwen3_8_27b.config import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    QWEN3_8_27B,
    Qwen38Config,
)
from models.qwen3_8_27b.contract import QWEN3_8_27B_TEXT_CONTRACT, make_text_contract


def tiny_config() -> Qwen38Config:
    """Two architecture-faithful layers small enough for real safetensors I/O."""
    return Qwen38Config(
        name="Qwen3.8-weight-test",
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        full_attention_interval=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        attention_bias=False,
    )


def _write_fixture(root: Path, contract) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """The two-shard fixture checkpoint, written by the module under test's own writer.

    `gdn_block` writes one the same way to feed a kernel checkpoint-native weights, so
    the layout a load is checked against is the layout a device case is checked against.
    """
    return weights.write_fixture_checkpoint(root, contract)


def check_contract() -> None:
    contract = QWEN3_8_27B_TEXT_CONTRACT
    assert contract.layer_types == (LINEAR_ATTENTION, LINEAR_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION) * 16
    assert contract.dims.rotary_dim == 64
    assert len(contract.weight_specs) == 1555
    assert len(contract.prepared_weight_specs) == 1203
    assert len(contract.layer_specs(0)) == 24
    assert len(contract.layer_specs(3)) == 25
    assert contract.memory.checkpoint_weight_bytes == 31_004_212_224
    assert contract.memory.resident_weight_bytes == 30_989_597_696
    assert contract.memory.kv_cache_bytes_per_token == 64 * 2**10
    assert contract.memory.max_kv_cache_bytes == 16 * 2**30
    assert contract.memory.gdn_state_bytes_per_sequence == 144 * 2**20
    assert contract.memory.conv_state_bytes_per_sequence == 48 * 3 * 10240 * 2
    assert contract.memory.max_context_headroom_bytes > 18 * 2**30
    assert QWEN3_8_27B.num_linear_layers == 48
    assert QWEN3_8_27B.num_full_attention_layers == 16


def check_quantization_recipe() -> None:
    source = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    actual_q, actual_scale, actual_offset = weights.modelslim_quantize_rows(source)
    assert actual_q.tolist() == [[-127, -64, 0, 64], [0, 0, 0, 0]]
    assert actual_scale[:, 0].tolist() == [0.0157470703125, 7.897615432739258e-07]
    assert actual_scale.shape == (2, 1)
    assert torch.count_nonzero(actual_offset) == 0

    # Q4 predates the deployment checkpoint and deliberately computes its
    # synthetic scale in FP32. Keep the two recipes visibly distinct.
    _q4_q, q4_scale = reference.quantize_rows(source)
    assert not torch.equal(actual_scale[:, 0], q4_scale)


def check_loader() -> None:
    from safetensors.torch import save_file

    contract = make_text_contract(tiny_config())
    with tempfile.TemporaryDirectory(prefix="qwen38_weights_") as directory:
        root = Path(directory)
        source, weight_map = _write_fixture(root, contract)
        summary = weights.checkpoint_summary(root, contract)
        assert summary["text_tensors"] == len(contract.weight_specs)
        assert summary["prepared_tensors"] == len(contract.prepared_weight_specs)
        assert summary["shards"] == 2

        prepared = weights.prepare_qwen38_weights(root, lambda tensor: tensor.clone(), contract=contract)
        assert prepared.layer_types == (LINEAR_ATTENTION, FULL_ATTENTION)
        assert len(prepared.tensors) == len(contract.prepared_weight_specs)
        assert len(prepared.layer(0)) == len(contract.layer_specs(0)) - 5
        assert len(prepared.layer(1)) == len(contract.layer_specs(1)) - 7
        assert not any(name.endswith("weight_offset") for name in prepared.tensors)
        assert not any(name.startswith("model.visual") for name in prepared.tensors)

        for spec in contract.prepared_weight_specs:
            actual = prepared.tensors[spec.name]
            expected = source[spec.name]
            if spec.role == "weight_scale":
                expected = expected.reshape(-1)
            assert torch.equal(actual, expected), spec.name

        # The loader must reject an asymmetric checkpoint instead of silently
        # treating a non-zero offset as the symmetric kernel format.
        offset_name = next(spec.name for spec in contract.weight_specs if spec.is_zero_offset)
        source[offset_name] = torch.ones_like(source[offset_name])
        shard_name = weight_map[offset_name]
        shard_names = [name for name, shard in weight_map.items() if shard == shard_name]
        tensors = {name: source[name] for name in shard_names if name in source}
        if shard_name.endswith("00001-of-00002.safetensors"):
            tensors["model.visual.blocks.0.norm1.weight"] = torch.ones(4, dtype=torch.bfloat16)
        save_file(tensors, root / shard_name)
        try:
            weights.prepare_qwen38_weights(root, lambda tensor: tensor, contract=contract)
        except ValueError as error:
            assert "offset is not zero" in str(error)
        else:
            raise AssertionError("non-zero symmetric weight_offset was accepted")


def check_resumed_fetch_verifies_content() -> None:
    """A resumed fetch must judge an existing file by content, not by length.

    Only a real 29 GiB download exercises this path otherwise, and it fails in
    both directions: trusting a corrupted shard of the right size, or rejecting
    a good one and re-downloading it. Both are expensive to discover for real.
    """
    body = b"safetensors payload" * 256
    digest = hashlib.sha256(body).hexdigest()
    with tempfile.TemporaryDirectory(prefix="qwen38_fetch_") as directory:
        shard = Path(directory) / "quant_model_weights-00001-of-00010.safetensors"
        shard.write_bytes(body)
        marker = weights._digest_marker(shard)

        assert weights._verified_locally(shard, len(body), digest)
        assert marker.is_file(), "a verified file must record its digest so the next run need not rehash"
        assert weights._verified_locally(shard, len(body), digest)

        assert not weights._verified_locally(shard, len(body) - 1, digest)
        assert not weights._verified_locally(shard, len(body), "de" * 32)
        assert not weights._verified_locally(Path(directory) / "absent", len(body), digest)

        # A stale or hand-written marker must not be believed over the bytes.
        marker.write_text("0" * 64 + "\n")
        assert weights._verified_locally(shard, len(body), digest)

        # Rewriting the file invalidates the record even at the same length,
        # so a corrupted shard is re-hashed and rejected rather than trusted.
        before = shard.stat()
        corrupted = bytearray(body)
        corrupted[0] ^= 0xFF
        shard.write_bytes(bytes(corrupted))
        os.utime(shard, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        assert not weights._verified_locally(shard, len(body), digest), "same length, different content"

        # The same rule must not cost a re-download when the content is fine:
        # a byte-identical rewrite is re-hashed once and accepted.
        shard.write_bytes(body)
        assert weights._verified_locally(shard, len(body), digest)
        assert weights._verified_locally(shard, len(body), digest)

        # Replace the path between hashing and marker publication.
        replacement = shard.with_suffix(".new")
        replacement.write_bytes(corrupted)
        marker.unlink()
        write_marker = weights._write_digest_marker

        def replace_before_marking(destination, stamp):
            replacement.replace(destination)
            write_marker(destination, stamp)

        with patch.object(weights, "_write_digest_marker", replace_before_marking):
            assert not weights._verified_locally(shard, len(body), digest)
        assert not weights._verified_locally(shard, len(body), digest)


def check_fetch_requires_indexed_shards() -> None:
    """A manifest missing an indexed shard must fail before fetching the weights."""
    with tempfile.TemporaryDirectory(prefix="qwen38_fetch_") as directory:
        root = Path(directory)
        names = dict.fromkeys((spec.name for spec in QWEN3_8_27B_TEXT_CONTRACT.weight_specs), "missing.safetensors")
        index = json.dumps({"weight_map": names}).encode()
        (root / weights.ECO_INDEX).write_bytes(index)
        manifest = dict.fromkeys(weights.ECO_METADATA_FILES, {})
        manifest[weights.ECO_INDEX] = {"Size": len(index), "Sha256": hashlib.sha256(index).hexdigest()}
        with patch.object(weights, "_modelscope_manifest", return_value=manifest):
            try:
                weights.fetch_checkpoint(root)
            except RuntimeError as error:
                assert "missing.safetensors" in str(error)
            else:
                raise AssertionError("fetch accepted a manifest missing an indexed shard")


def check_fetch_publishes_only_verified_index() -> None:
    """Reject incomplete and misassigned indexes even when manifest digests match."""
    contract = make_text_contract(tiny_config())
    with tempfile.TemporaryDirectory(prefix="qwen38_fetch_") as directory:
        root = Path(directory)
        source, destination = root / "remote", root / "fetch"
        source.mkdir()
        _, weight_map = _write_fixture(source, contract)
        (source / "config.json").write_text("{}")
        (source / "quant_model_description.json").write_text("{}")
        index = source / weights.ECO_INDEX

        def manifest(_repo, _revision):
            entries = {}
            for path in source.iterdir():
                body = path.read_bytes()
                entries[path.name] = {"Size": len(body), "Sha256": hashlib.sha256(body).hexdigest()}
            return entries

        first = contract.weight_specs[0].name
        last = contract.weight_specs[-1].name
        assert weight_map[first] != weight_map[last]
        incomplete = {name: shard for name, shard in weight_map.items() if name != first}
        misassigned = {**weight_map, last: weight_map[first]}
        with (
            patch.object(weights, "QWEN3_8_27B_TEXT_CONTRACT", contract),
            patch.object(weights, "_modelscope_manifest", manifest),
            patch.object(weights, "_modelscope_url", lambda _repo, _revision, name: (source / name).as_uri()),
        ):
            for invalid, message in ((incomplete, "missing 1 tensors"), (misassigned, "absent from safetensors")):
                index.write_text(json.dumps({"weight_map": invalid}))
                try:
                    weights.fetch_checkpoint(destination, log=lambda _: None)
                except ValueError as error:
                    assert message in str(error)
                else:
                    raise AssertionError("fetch accepted an invalid checkpoint")
                assert not (destination / weights.ECO_INDEX).exists()
                assert not (destination / weight_map[first]).exists()

            index.write_text(json.dumps({"weight_map": weight_map}))
            assert weights.fetch_checkpoint(destination, log=lambda _: None) == destination
            assert (destination / weights.ECO_INDEX).read_bytes() == index.read_bytes()
            assert weights.checkpoint_summary(destination, contract)["shards"] == 2


def check_loader_excludes_a_concurrent_fetch() -> None:
    """A fetch must not replace shards under a load that already passed the stamp check.

    The revision stamp is read once, before the first shard is opened, so on its own it
    cannot see a fetch that starts while the remaining shards are still streaming. The
    loader holds the fetch lock shared for the whole stream; this releases a would-be
    fetcher after the first tensor is exported and checks it takes the exclusive lock
    only once the load has returned.
    """
    import fcntl

    contract = make_text_contract(tiny_config())
    with tempfile.TemporaryDirectory(prefix="qwen38_race_") as directory:
        root = Path(directory)
        _write_fixture(root, contract)
        weights._write_revision_stamp(root, weights.ECO_REVISION)

        refused: list[bool] = []

        def exporter(tensor):
            if not refused:
                with (root / weights._FETCH_LOCK).open("a+b") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        refused.append(False)
                    except BlockingIOError:
                        refused.append(True)
            return tensor.clone()

        weights.prepare_qwen38_weights(root, exporter, contract=contract)
        assert refused == [True], "a fetch could have replaced the shards still to be read"

        # The lock is an exclusion, not a requirement: a checkpoint nothing can fetch
        # into still loads, and so does one no fetch ever stamped.
        (root / weights._REVISION_STAMP).unlink()
        (root / weights._FETCH_LOCK).unlink()
        os.chmod(root, 0o555)
        try:
            prepared = weights.prepare_qwen38_weights(root, lambda t: t.clone(), contract=contract)
        finally:
            os.chmod(root, 0o755)
        assert len(prepared.tensors) == len(contract.prepared_weight_specs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3",
                        help="accepted for CI's uniform invocation; nothing here runs on device")
    parser.add_argument("--sim", dest="execution_mode", action="store_true",
                        help="accepted for CI's uniform invocation; unused")
    parser.add_argument("-d", "--device", type=int, default=0, help="accepted and unused")
    parser.parse_args()

    check_contract()
    check_quantization_recipe()
    check_resumed_fetch_verifies_content()
    check_fetch_requires_indexed_shards()
    if importlib.util.find_spec("safetensors") is None:
        print("SKIP: loader check requires safetensors")
    else:
        check_loader()
        check_loader_excludes_a_concurrent_fetch()
        check_fetch_publishes_only_verified_index()
    print("PASS: Qwen3.8 weight loader and text contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
