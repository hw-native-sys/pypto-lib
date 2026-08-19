# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side contracts for the DeepSeek-V4 decode RoPE profile ABI."""

import ast
import inspect
import os
import sys
from pathlib import Path

import pytest
import torch

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"
os.environ.setdefault("DEEPSEEK_V4_VARIANT", "flash")
sys.path.insert(0, str(_MODEL_DIR))

import decode_fwd  # noqa: E402
import decode_mtp  # noqa: E402
import prefill_mtp  # noqa: E402
from golden import TensorSpec  # noqa: E402


def _raw_function(function):
    return getattr(function, "_func", function)


@pytest.mark.parametrize(
    ("function", "ranked"),
    [
        (decode_fwd.decode_fwd, False),
        (decode_fwd.l3_decode_fwd, True),
        (decode_mtp.mtp_decode_layer, False),
        (decode_mtp.l3_mtp_decode_layer, True),
        (prefill_mtp.mtp_prefill_fwd, False),
        (prefill_mtp.l3_mtp_prefill_fwd, True),
    ],
)
@pytest.mark.parametrize("name", ["freqs_cos", "freqs_sin"])
def test_decode_entries_use_two_profile_rope_abi(function, ranked, name):
    annotation = inspect.signature(_raw_function(function)).parameters[name].annotation
    prefix = [decode_fwd.N_RANKS] if ranked else []

    assert list(annotation.shape) == [
        *prefix,
        2,
        decode_fwd.MAX_SEQ_LEN,
        decode_fwd.ROPE_HEAD_DIM,
    ]


def _leaf_spec(name, value):
    return TensorSpec(
        name,
        list(value.shape),
        value.dtype,
        init_value=lambda value=value: value.clone(),
    )


@pytest.mark.parametrize(
    "profile_builder",
    [decode_fwd._make_rope_profile_spec, decode_mtp._make_rope_profile_spec],
)
def test_decode_profile_builders_stack_nonzero_distinct_tables(profile_builder):
    swa = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16) + 1
    compressed = swa + 7
    spec = profile_builder(
        "freqs_cos",
        _leaf_spec("freqs_cos", swa),
        _leaf_spec("freqs_cos", compressed),
    )
    actual = spec.create_tensor()

    assert spec.shape == [2, 6, 4]
    assert torch.count_nonzero(actual[0]) > 0
    assert torch.count_nonzero(actual[1]) > 0
    assert not torch.equal(actual[0], actual[1])
    assert torch.equal(actual[0], swa)
    assert torch.equal(actual[1], compressed)


def test_prefill_mtp_preserves_the_ranked_two_profile_spec():
    profiles = torch.arange(
        prefill_mtp.N_RANKS * 2 * 6 * 4, dtype=torch.float32
    ).reshape(prefill_mtp.N_RANKS, 2, 6, 4).to(torch.bfloat16) + 1
    source = TensorSpec(
        "freqs_cos",
        list(profiles.shape),
        profiles.dtype,
        init_value=lambda: profiles.clone(),
    )
    copied = prefill_mtp._ranked(source, torch)

    assert copied.shape == [prefill_mtp.N_RANKS, 2, 6, 4]
    assert torch.equal(copied.create_tensor(), profiles)


def _function_node(path, name):
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _calls(function, name):
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _referenced_names(call):
    return {
        node.id
        for arg in call.args
        for node in ast.walk(arg)
        if isinstance(node, ast.Name)
    }


def test_full_decode_routes_swa_and_compressed_attention_profiles():
    function = _function_node(_MODEL_DIR / "decode_fwd.py", "decode_fwd")
    swa_calls = _calls(function, "attention_swa")
    csa_calls = _calls(function, "attention_csa")
    hca_calls = _calls(function, "attention_hca")

    assert len(swa_calls) == 2
    assert all(
        {"swa_freqs_cos", "swa_freqs_sin"} <= _referenced_names(call)
        for call in swa_calls
    )
    assert len(csa_calls) == 2
    assert len(hca_calls) == 3
    assert all(
        {"compressed_freqs_cos", "compressed_freqs_sin"}
        <= _referenced_names(call)
        for call in [*csa_calls, *hca_calls]
    )


@pytest.mark.parametrize(
    ("filename", "runtime_name", "golden_name"),
    [
        ("decode_mtp.py", "mtp_decode_layer", "golden_mtp_decode_layer"),
        ("prefill_mtp.py", "mtp_prefill_fwd", "golden_mtp_prefill_fwd"),
    ],
)
def test_mtp_runtime_and_golden_both_select_swa_profile(
    filename, runtime_name, golden_name
):
    path = _MODEL_DIR / filename
    runtime = _function_node(path, runtime_name)
    golden = _function_node(path, golden_name)
    attention_calls = _calls(runtime, "attention_swa") + _calls(
        runtime, "prefill_attention_swa"
    )

    assert len(attention_calls) == 1
    assert {"swa_freqs_cos", "swa_freqs_sin"} <= _referenced_names(
        attention_calls[0]
    )

    profile_values = {}
    for mapping in (node for node in ast.walk(golden) if isinstance(node, ast.Dict)):
        for key, value in zip(mapping.keys, mapping.values):
            if isinstance(key, ast.Constant) and key.value in {
                "freqs_cos",
                "freqs_sin",
            }:
                profile_values[key.value] = ast.unparse(value)

    assert profile_values == {
        "freqs_cos": "tensors['freqs_cos'][rank, 0]",
        "freqs_sin": "tensors['freqs_sin'][rank, 0]",
    }
