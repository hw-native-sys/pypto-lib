# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reference and precision checks for token-sharded prefill SWA boundaries."""

import pytest
import torch

@pytest.fixture
def swa():
    pypto = pytest.importorskip("pypto")
    if getattr(pypto, "__pypto_stub__", False):
        pytest.skip("model imports require real PyPTO")
    from models.deepseek_v4_1_flash import prefill_swa

    return prefill_swa


@pytest.mark.parametrize("tokens", [1, 3, 7])
def test_two_swa_boundaries_match_dense_mhc(monkeypatch, tokens, swa):
    """Distinct rank rows survive gather, local residual updates, and a second layer."""
    tp, dp, width, streams = 4, 2, 8, 4
    monkeypatch.setattr(swa, "TP_SIZE", tp)
    local = (tokens + tp - 1) // tp
    generator = torch.Generator().manual_seed(41)
    full = torch.randn(dp, tokens, streams, width, generator=generator)
    # The second DP group includes inactive rows, including an entirely empty group at T=1.
    full[1, tokens // 2:] = 0
    function = torch.randn(dp, 24, streams * width, generator=generator) / (streams * width) ** 0.5
    scale = torch.randn(dp, 3, generator=generator)
    bias = torch.randn(dp, 24, generator=generator)
    projection = torch.randn(width, width, generator=generator)
    incoming = torch.sigmoid(torch.randn(dp, tokens, streams, generator=generator))
    norm_weight = torch.linspace(0.5, 1.5, width).bfloat16().repeat(dp, 1)
    tensors = {
        "x_hc": torch.cat([swa.token_shards(x, local) for x in full]),
        "incoming_pre_mix": torch.cat([swa.token_shards(x, local) for x in incoming]),
        "attn_norm_weight": norm_weight.repeat_interleave(tp, dim=0),
        "next_pre_mix": torch.empty(tp * dp, local, streams),
        "hc_attn_fn": function.repeat_interleave(tp, dim=0),
        "hc_attn_scale": scale.repeat_interleave(tp, dim=0),
        "hc_attn_base": bias.repeat_interleave(tp, dim=0),
        "hidden": torch.empty(tp * dp, tokens, width, dtype=torch.bfloat16),
        "attn_out": torch.empty(tp * dp, local, width, dtype=torch.bfloat16),
        "output": torch.empty(tp * dp, local, streams, width),
        "window_cache": torch.zeros(tp * dp, 1),
        "window_cache_scale": torch.zeros(tp * dp, 1),
        "num_tokens": torch.tensor([tokens, tokens // 2]).repeat_interleave(tp).reshape(tp * dp, 1),
    }
    for group, active in enumerate([tokens, tokens // 2]):
        tensors["x_hc"][group * tp:(group + 1) * tp].flatten(0, 1)[active:] = 17

    def reference_attention(inputs, hidden, base):
        output = (hidden.float() @ projection).bfloat16()
        caches = [(torch.tensor([rank]), torch.tensor([rank + 1])) for rank in range(base, base + tp)]
        return output, caches

    monkeypatch.setattr(swa, "reference_attention", reference_attention)
    for _ in range(2):
        expected = []
        expected_next = []
        for group in range(dp):
            pre, post, mix = swa.golden_mhc_mixes(full[group], function[group], scale[group], bias[group])
            hidden = swa.golden_mhc_pre(full[group], incoming[group]).float()
            inv = torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-20)
            hidden = (hidden * inv * norm_weight[group].float()).bfloat16()
            attention = (hidden.float() @ projection).bfloat16()
            expected.append(swa.golden_mhc_post(attention, full[group], post, mix).float())
            pre[int(tensors["num_tokens"][group * tp, 0]):] = 0
            expected_next.append(pre)
        swa.golden_prefill_swa_case(tensors)
        for group in range(dp):
            base = group * tp
            actual = tensors["output"][base:base + tp].flatten(0, 1)
            torch.testing.assert_close(actual[:tokens], expected[group], rtol=0, atol=0)
            assert torch.count_nonzero(actual[tokens:]) == 0
            actual_next = tensors["next_pre_mix"][base:base + tp].flatten(0, 1)
            torch.testing.assert_close(actual_next[:tokens], expected_next[group], rtol=0, atol=0)
            assert torch.count_nonzero(actual_next[tokens:]) == 0
            for rank in range(base, base + tp):
                torch.testing.assert_close(tensors["hidden"][rank], tensors["hidden"][base], rtol=0, atol=0)
                assert tensors["window_cache"][rank].item() == rank
        full = torch.stack(expected)
        tensors["x_hc"] = tensors["output"].clone()
        incoming = torch.stack(expected_next)
        tensors["incoming_pre_mix"] = tensors["next_pre_mix"].clone()


def test_hc_comparator_checks_every_shard_and_padding(monkeypatch, swa):
    """A wrong nonzero-rank residual or a padded-row write must fail validation."""
    monkeypatch.setattr(swa, "TP_SIZE", 4)
    x = torch.randn(4, 2, 4, 8)
    x[-1, -1] = 0
    function = torch.zeros(4, 24, 32)
    scale = torch.zeros(4, 3)
    bias = torch.zeros(4, 24)
    attention = torch.randn(4, 2, 8, dtype=torch.bfloat16)
    attention[-1, -1] = 0
    output = torch.empty_like(x)
    for rank in range(4):
        _, post, mix = swa.golden_mhc_mixes(x[rank], function[rank], scale[rank], bias[rank])
        output[rank] = swa.golden_mhc_post(attention[rank], x[rank], post, mix)
    inputs = {"x_hc": x, "hc_attn_fn": function, "hc_attn_scale": scale, "hc_attn_base": bias}
    check = swa.make_staged_compare()["output"]
    kwargs = {"inputs": inputs, "actual_outputs": {"attn_out": attention},
              "expected_outputs": {}, "rtol": 1e-3, "atol": 1e-3}
    assert check(output, output, **kwargs)[0]
    for rank, row in [(1, 0), (3, 1)]:
        wrong = output.clone()
        wrong[rank, row] += 1
        assert not check(wrong, output, **kwargs)[0]


@pytest.mark.parametrize("active_rows, outliers, passes", [(2, 10, True), (2, 11, False), (1, 5, True), (1, 6, False), (0, 0, True)])
def test_output_tolerance_counts_only_active_elements(monkeypatch, swa, active_rows, outliers, passes):
    """The output budget counts outliers per active shard without padding dilution."""
    monkeypatch.setattr(swa, "TP_SIZE", 1)
    x = torch.zeros(1, 2, 4, 125)
    attention = torch.ones(1, 2, 125, dtype=torch.bfloat16)
    attention[:, active_rows:] = 0
    output = attention.unsqueeze(2).expand(-1, -1, 4, -1).float().clone()
    expected = output.clone()
    expected.reshape(-1)[:outliers] = 2
    inputs = {
        "x_hc": x, "hc_attn_fn": torch.zeros(1, 24, 500),
        "hc_attn_scale": torch.zeros(1, 3), "hc_attn_base": torch.zeros(1, 24),
        "num_tokens": torch.tensor([[active_rows]]),
    }
    kwargs = {
        "inputs": inputs, "actual_outputs": {"attn_out": attention},
        "expected_outputs": {}, "rtol": 1e-3, "atol": 1e-3,
    }
    check = swa.make_staged_compare()["output"]
    assert check(output, expected, **kwargs)[0] == passes
    for value in [float("nan"), float("inf")]:
        nonfinite = expected.clone()
        nonfinite[0, 0, 0, 0] = value
        if active_rows:
            assert not check(output, nonfinite, **kwargs)[0]
    if active_rows < 2:
        dirty_padding = output.clone()
        dirty_padding[0, -1, 0, 0] = 1e-5
        assert not check(dirty_padding, expected, **kwargs)[0]
