# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-only contracts for importing and selecting the vLLM Ascend CSA service."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.vllm_ascend.deepseek_v4_flash.csa import csa_package
from models.vllm_ascend.deepseek_v4_flash.csa.common.quantization_contract import validate_quantization_contract
from models.vllm_ascend.deepseek_v4_flash.csa.common.graph_contract import TNDGraphContract
from models.vllm_ascend.deepseek_v4_flash.csa.common.tnd_config import supported_descriptor


@pytest.mark.parametrize("variant,suffix", [
    ("spmd_tnd", "spmd"), ("mpmd_tnd", "mpmd"),
    ("performance", "legacy"), ("perf", "legacy"), (" SPMD_TND ", "spmd"),
])
def test_package_selection(variant, suffix):
    package = csa_package(variant)
    assert package == f"models.vllm_ascend.deepseek_v4_flash.csa.{suffix}"
    assert importlib.import_module(package).__file__


@pytest.mark.parametrize("variant", ["", "spmd", "precision", "pkg:other", "../other"])
def test_unknown_variant_is_not_silently_replaced(variant):
    with pytest.raises(ValueError, match="Unsupported CSA variant"):
        csa_package(variant)


def test_namespace_import_does_not_load_device_stack():
    # A fresh process avoids the test suite's optional PyPTO stubs.
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib
import sys
from models.vllm_ascend.deepseek_v4_flash.csa import csa_package
for variant in ('spmd_tnd', 'mpmd_tnd', 'performance'):
    importlib.import_module(csa_package(variant))
for dependency in ('torch', 'torch_npu', 'pypto', 'vllm', 'vllm_ascend'):
    assert dependency not in sys.modules, dependency
"""
    subprocess.run([sys.executable, "-c", code], cwd=root,
                   env={**os.environ, "PYTHONPATH": str(root)}, check=True)


def descriptor(**overrides):
    return dict(tokens=96, requests=16, actual_tokens=96, max_query_tokens=6,
                max_seq_len=131072, batch_capacity=16) | overrides


def test_native_descriptor_accepts_padding_and_mixed_query_lengths():
    assert supported_descriptor(**descriptor())
    assert supported_descriptor(**descriptor(actual_tokens=81))
    assert supported_descriptor(**descriptor(actual_tokens=16, max_query_tokens=1))


@pytest.mark.parametrize("overrides", [
    {"requests": 17}, {"actual_tokens": 97}, {"actual_tokens": 0},
    {"max_query_tokens": 7}, {"max_seq_len": 8191}, {"max_seq_len": 1048577},
    {"requests": True}, {"tokens": 96.0},
])
def test_unsupported_descriptors_fall_back_to_native(overrides):
    assert not supported_descriptor(**descriptor(**overrides))


def test_graph_replay_requires_same_runtime_signature():
    signature = ("device-address", "shape", "policy")
    context = SimpleNamespace(signature=signature)
    runtime = SimpleNamespace(graph_signature=lambda ctx, capacity: ctx.signature)
    contract = TNDGraphContract(runtime, 96, signature)
    assert contract.allows_replay(context)
    context.signature = ("different-address", "shape", "policy")
    assert not contract.allows_replay(context)
    assert not TNDGraphContract(runtime, 96, None).allows_replay(context)


@pytest.mark.parametrize("mode", ["spmd_tnd", "mpmd_tnd"])
def test_compressed_tensor_quantization_rejects_asymmetric_weights(mode):
    group = {
        "targets": ["Linear"],
        "weights": dict(num_bits=8, type="int", symmetric=True, dynamic=False, strategy="channel"),
        "input_activations": dict(num_bits=8, type="int", symmetric=True, dynamic=True, strategy="token"),
    }
    config = dict(quant_method="compressed-tensors", format="int-quantized", config_groups={"g": group})
    validate_quantization_contract("compressed-tensors", config, mode)
    group["weights"]["symmetric"] = False
    with pytest.raises(ValueError, match="symmetric dynamic W8A8"):
        validate_quantization_contract("compressed-tensors", config, mode)
