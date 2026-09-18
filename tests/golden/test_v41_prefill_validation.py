# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C1A reference rounding and output validation regression tests."""

import pytest
import torch

from models.deepseek_v4_1_flash import prefill_c1a_test_utils as utils
from models.deepseek_v4_1_flash.quantization import mxfp8_linear, pack_mx_b_scale


@pytest.mark.parametrize("quantized", [False, True])
def test_output_projection_preserves_fp32_partial(quantized):
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, 0] = 1
    x[0, 1] = 1 / 64
    weights = torch.zeros(64, 16)
    weights[0] = 1
    weights[1] = 0.25
    dtype = torch.float8_e4m3fn if quantized else torch.bfloat16
    weights = weights.to(dtype)
    scales = pack_mx_b_scale(torch.full((2, 16), 127, dtype=torch.uint8)) if quantized else None
    partial = mxfp8_linear(x, weights, scales, output_dtype=torch.float32)
    rounded = mxfp8_linear(x, weights, scales)
    torch.testing.assert_close(partial, torch.full((1, 16), 1 + 1 / 256), rtol=0, atol=0)
    torch.testing.assert_close(rounded, torch.ones(1, 16, dtype=torch.bfloat16), rtol=0, atol=0)


def test_tp_reduction_rounds_after_sum():
    partials = [torch.tensor([[1 + 1 / 256]]), torch.tensor([[-1.0]])]
    actual = utils._reduce_tp_partials(partials)
    torch.testing.assert_close(actual, torch.tensor([[1 / 256]], dtype=torch.bfloat16), rtol=0, atol=0)


def test_prefill_attention_probability_rounding():
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 2, 32, generator=generator).to(torch.bfloat16)
    cache = torch.randn(1, 128, 1, 32, generator=generator).to(torch.bfloat16)
    sink = torch.tensor([0.5, -0.25])
    actual = utils.golden_prefill_c1a_attention(
        query, cache, torch.tensor([[0, 1]]), cache, torch.tensor([[-1, -1]]), sink,
    )
    # A single two-key tile has a closed-form denominator including the sink.
    q = query[0].double()
    kv = cache[0, :2, 0].double()
    logits = (q[:, None, :] * kv[None, :, :]).sum(-1) * 32**-0.5
    maximum = logits.amax(-1)
    probability = (logits - maximum[:, None]).exp()
    denominator = probability.sum(-1) + (sink.double() - maximum).exp()
    numerator = (probability.to(torch.bfloat16).double()[:, :, None] * kv[None, :, :]).sum(1)
    numerator[0, :16] = (probability[0, :, None] * kv[:, :16]).sum(0)
    expected = (numerator / denominator[:, None]).to(torch.bfloat16).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    fp32_pv = ((probability[:, :, None] * kv[None, :, :]).sum(1) / denominator[:, None])
    assert not torch.equal(actual[0, 1], fp32_pv[1].to(torch.bfloat16))


def test_prefill_attention_empty_sources():
    query = torch.ones(1, 2, 32, dtype=torch.bfloat16)
    cache = torch.ones(1, 128, 1, 32, dtype=torch.bfloat16)
    indices = torch.full((1, 32), -1)
    actual = utils.golden_prefill_c1a_attention(query, cache, indices, cache, indices, torch.zeros(2))
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("corruption", ["zero", "gain", "single_outlier", "one_bad_row"])
def test_output_gate_rejects_bad_low_magnitude_output(corruption):
    expected = torch.full((4, 64, 256), 0.005, dtype=torch.bfloat16)
    actual = expected.clone()
    if corruption == "zero":
        actual.zero_()
    elif corruption == "gain":
        actual = (actual.float() * 1.5).to(actual.dtype)
    elif corruption == "single_outlier":
        actual[0, 0, 0] = 1e6
    else:
        actual[0, 0] = (actual[0, 0].float() * 1.1).to(actual.dtype)
    passed, _ = utils._compare_attention_rows(actual, expected)
    assert not passed


def test_output_peak_bound_catches_outlier_below_rms_bound():
    expected = torch.ones(1, 1, 256, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, 0, 0] += 0.125
    assert float((actual.float() - expected.float()).square().mean().sqrt()) < utils.OUTPUT_RMS_RTOL
    passed, _ = utils._compare_attention_rows(actual, expected)
    assert not passed


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_output_gate_rejects_nonfinite(value):
    expected = torch.ones(1, 1, 256)
    actual = expected.clone()
    actual[0, 0, 0] = value
    assert not utils._compare_attention_rows(actual, expected)[0]


def test_output_gate_accepts_small_error_and_zero_reference():
    expected = torch.full((1, 2, 256), 0.005, dtype=torch.bfloat16)
    actual = (expected.float() * 1.006).to(expected.dtype)
    assert bool((actual != expected).any())
    assert utils._compare_attention_rows(actual, expected)[0]
    zeros = torch.zeros_like(expected)
    assert utils._compare_attention_rows(zeros, zeros)[0]


def test_output_gate_does_not_overflow_on_large_finite_reference():
    expected = torch.full((1, 1, 256), 1e30)
    assert not utils._compare_attention_rows(torch.zeros_like(expected), expected)[0]


def test_output_gate_rejects_dtype_mismatch():
    expected = torch.ones(1, 1, 256, dtype=torch.bfloat16)
    assert not utils._compare_attention_rows(expected.float(), expected)[0]


def topk_inputs():
    return {
        "request_ids": torch.tensor([[0]], dtype=torch.int32),
        "compressed_lens": torch.tensor([[3]], dtype=torch.int32),
        "index_block_table": torch.tensor([[[1]]], dtype=torch.int32),
    }


@pytest.mark.parametrize("indices", [
    [129, 128, 130, -1], [128, 129, 130, -7], [128, -1, 129, 130],
    [128, 128, 130, -1], [128, 129, 999, -1],
])
def test_topk_abi_checks(indices):
    actual = torch.tensor([[indices]], dtype=torch.int32)
    expected = torch.tensor([[[128, 129, 130, -1]]], dtype=torch.int32)
    passed, _ = utils.topk_indices_compare("full")(
        actual, expected, actual_outputs={}, expected_outputs={}, inputs=topk_inputs(), rtol=0, atol=0,
    )
    assert not passed


def test_topk_order_uses_logical_positions_not_physical_addresses(monkeypatch):
    inputs = {
        "request_ids": torch.tensor([[0]], dtype=torch.int32),
        "compressed_lens": torch.tensor([[129]], dtype=torch.int32),
        "index_block_table": torch.tensor([[[1, 0]]], dtype=torch.int32),
    }
    expected = torch.tensor([[[129, 0, -1]]], dtype=torch.int32)
    actual = torch.tensor([[[128, 0, -1]]], dtype=torch.int32)
    monkeypatch.setattr(utils, "_golden_index_scores", lambda *args: torch.ones(1, 129))
    compare = utils.topk_indices_compare("full")
    context = dict(actual_outputs={}, expected_outputs={}, inputs=inputs, rtol=0, atol=0)
    assert compare(actual, expected, **context)[0]
    assert not compare(actual[..., [1, 0, 2]], expected, **context)[0]


def test_invalid_topk_cannot_redefine_output_reference(monkeypatch):
    def forbidden(*args):
        pytest.fail("Invalid selection reached the conditional output reference")
    monkeypatch.setattr(utils, "_selected_output_reference", forbidden)
    nominal = torch.tensor([[[128, 129, 130, -1]]], dtype=torch.int32)
    selected = torch.tensor([[[128, 129, 999, -1]]], dtype=torch.int32)
    output = torch.ones(1, 1, 256)
    passed, _ = utils.attention_output_compare("full")(
        output, output, actual_outputs={"topk_indices": selected},
        expected_outputs={"topk_indices": nominal, "output": output},
        inputs=topk_inputs(), rtol=0, atol=0,
    )
    assert not passed


@pytest.mark.parametrize("output_value,passed", [(2.0, True), (2.5, False)])
def test_tied_selection_still_checks_independent_output(monkeypatch, output_value, passed):
    nominal = torch.tensor([[[128, -1]]], dtype=torch.int32)
    selected = torch.tensor([[[129, -1]]], dtype=torch.int32)
    expected = {
        "topk_indices": nominal, "output": torch.ones(1, 1, 256),
        "window_cache": torch.ones(1, 1, 4),
    }
    monkeypatch.setattr(utils, "_golden_index_scores", lambda *args: torch.tensor([[1.0, 1.0, 0.0]]))
    def reference(inputs, reference_outputs, selection):
        assert reference_outputs is expected
        assert torch.equal(selection, selected)
        assert torch.equal(reference_outputs["window_cache"], torch.ones(1, 1, 4))
        return torch.full_like(expected["output"], 2.0)
    monkeypatch.setattr(utils, "_selected_output_reference", reference)
    result, _ = utils.attention_output_compare("full")(
        torch.full_like(expected["output"], output_value), expected["output"],
        actual_outputs={"topk_indices": selected, "window_cache": torch.full((1, 1, 4), 99.0)},
        expected_outputs=expected,
        inputs=topk_inputs(), rtol=0, atol=0,
    )
    assert result is passed
