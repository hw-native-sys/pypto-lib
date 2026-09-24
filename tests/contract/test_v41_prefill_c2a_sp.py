# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sequence-parallel C2A row mapping and validation regressions."""

import pytest
import torch


@pytest.fixture
def c2a():
    pypto = pytest.importorskip("pypto")
    if getattr(pypto, "__pypto_stub__", False):
        pytest.skip("model imports require real PyPTO")
    from models.deepseek_v4_1_flash import prefill_c2a_full

    return prefill_c2a_full


@pytest.mark.parametrize("tokens,counts", [(1, (1, 0)), (3, (3, 0)), (7, (7, 3)), (9, (9, 5))])
def test_two_sublayers_keep_contiguous_rows(monkeypatch, c2a, tokens, counts):
    tp, dp, streams, width = 4, 2, 4, 8
    monkeypatch.setattr(c2a, "TP_SIZE", tp)
    local = (tokens + tp - 1) // tp
    gen = torch.Generator().manual_seed(1275)
    full = torch.randn(dp, tokens, streams, width, generator=gen)
    incoming = torch.sigmoid(torch.randn(dp, tokens, streams, generator=gen))
    fn = torch.randn(dp, 24, streams * width, generator=gen) / (streams * width) ** 0.5
    scale = torch.randn(dp, 3, generator=gen)
    bias = torch.randn(dp, 24, generator=gen)
    norm = torch.linspace(0.5, 1.5, width).bfloat16().repeat(dp, 1)
    projection = torch.randn(width, width, generator=gen)
    tensors = {
        "x_hc": torch.cat([c2a.token_shards(value, local) for value in full]),
        "pre_mix": torch.cat([c2a.token_shards(value, local) for value in incoming]),
        "hc_attn_fn": fn.repeat_interleave(tp, 0),
        "hc_attn_scale": scale.repeat_interleave(tp, 0),
        "hc_attn_base": bias.repeat_interleave(tp, 0),
        "attn_norm_weight": norm.repeat_interleave(tp, 0),
        "attn_input": torch.empty(tp * dp, tokens, width, dtype=torch.bfloat16),
        "attn_output": torch.empty(tp * dp, local, width, dtype=torch.bfloat16),
        "next_pre_mix": torch.empty(tp * dp, local, streams),
        "x_hc_out": torch.empty(tp * dp, local, streams, width),
        "num_tokens": torch.tensor(counts).repeat_interleave(tp).reshape(-1, 1),
    }
    monkeypatch.setattr(c2a, "ATTENTION_STATE", {"reuse": ()})

    def attention(mode, epochs, values, x, ranks):
        assert mode == "reuse"
        active = int(values["num_tokens"][ranks[0], 0])
        assert torch.count_nonzero(x[active:]) == 0
        return (x.float() @ projection).bfloat16(), [{} for _ in ranks]

    monkeypatch.setattr(c2a, "reference_attention", attention)
    for _ in range(2):
        expected, next_mix = [], []
        for group in range(dp):
            active = counts[group]
            full[group, active:] = 0
            tensors["x_hc"][group * tp : (group + 1) * tp].flatten(0, 1)[active:] = 17
            pre, post, comb = c2a.hc_mixes(full[group], fn[group], scale[group], bias[group])
            collapsed = c2a.hc_pre(full[group], incoming[group]).bfloat16().float()
            normalized = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-20)
            normalized = (normalized * norm[group].float()).bfloat16()
            output = c2a.hc_post(
                (normalized.float() @ projection).bfloat16(), full[group], post, comb
            ).float()
            output[active:] = 0
            pre[active:] = 0
            expected.append(output)
            next_mix.append(pre)
        c2a.make_golden("reuse", 1)(tensors)
        for group in range(dp):
            ranks = slice(group * tp, (group + 1) * tp)
            actual = tensors["x_hc_out"][ranks].flatten(0, 1)
            torch.testing.assert_close(actual[:tokens], expected[group], rtol=0, atol=0)
            assert torch.count_nonzero(actual[tokens:]) == 0
            torch.testing.assert_close(tensors["next_pre_mix"][ranks].flatten(0, 1)[:tokens], next_mix[group])
            for rank in range(group * tp + 1, (group + 1) * tp):
                assert torch.equal(tensors["attn_input"][group * tp], tensors["attn_input"][rank])
        full, incoming = torch.stack(expected), torch.stack(next_mix)
        tensors["x_hc"], tensors["pre_mix"] = tensors["x_hc_out"].clone(), tensors["next_pre_mix"].clone()


def test_comparisons_check_nonleader_and_empty_shards(monkeypatch, c2a):
    monkeypatch.setattr(c2a, "TP_SIZE", 4)
    inputs = {"num_tokens": torch.tensor([[3], [3], [3], [3]])}
    expected = torch.ones(4, 1, 8)
    expected[3] = 0
    check = c2a.compare_shards("output", lambda a, b, **kw: (torch.equal(a, b), ""))
    assert check(expected, expected, inputs=inputs)[0]
    for rank in range(4):
        bad = expected.clone()
        bad[rank, 0, 0] += 1
        assert not check(bad, expected, inputs=inputs)[0]
    bad = expected.clone()
    bad[3, 0, 0] = float("nan")
    assert not check(bad, expected, inputs=inputs)[0]


def test_c1_reference_excludes_padding_and_preserves_empty_groups(monkeypatch, c2a):
    from types import SimpleNamespace
    from models.deepseek_v4_1_flash import prefill_c1a_sp as c1a

    monkeypatch.setattr(c1a, "TP_SIZE", 4)
    tokens, width, active = 7, 8, 3
    x = torch.arange(tokens * width).reshape(tokens, width).bfloat16()
    calls = []

    def reference(x, rope_cos, window_slots, compressed_slots, window_cache):
        calls.append(x.shape[0])
        assert x.shape[0] == rope_cos.shape[0] == window_slots.shape[0] == compressed_slots.shape[0] == active
        marker = window_cache[0].item()
        return SimpleNamespace(
            output=x.float() * (marker + 1),
            window_cache=window_cache + 1,
            window_cache_scale=window_cache.clone(),
            compressed_cache=window_cache.clone(),
            compressed_cache_scale=window_cache.clone(),
            index_cache=window_cache.clone(),
            index_cache_scale=window_cache.clone(),
            topk_indices=torch.zeros(active, c1a.C.INDEX_TOPK, dtype=torch.int32),
            candidate_mask=torch.zeros(active, 128, dtype=torch.uint8),
        )

    monkeypatch.setattr(c1a.full, "golden_prefill_attn_c1a_full", reference)
    tensors = {
        "num_tokens": torch.tensor([3] * 4 + [0] * 4).reshape(8, 1),
        "rope_cos": torch.ones(8, tokens, 2),
        "window_slots": torch.zeros(8, tokens, dtype=torch.int64),
        "compressed_slots": torch.zeros(8, tokens, dtype=torch.int64),
        "topk_indices": torch.full((8, tokens, c1a.C.INDEX_TOPK), 13, dtype=torch.int32),
        "candidate_mask": torch.ones(8, tokens, 128, dtype=torch.uint8),
    }
    for name in c1a.STATE_NAMES["full"]:
        if name not in tensors:
            tensors[name] = torch.arange(8).float().reshape(8, 1)
    reduced, results = c1a.reference_attention("full", 1, tensors, x, range(4))
    torch.testing.assert_close(reduced[:active], (x[:active].float() * 10).bfloat16(), rtol=0, atol=0)
    assert torch.count_nonzero(reduced[active:]) == 0
    assert all(bool((result["topk_indices"][active:] == -1).all()) for result in results)
    empty, results = c1a.reference_attention("full", 1, tensors, x, range(4, 8))
    assert calls == [active] * 4
    assert torch.count_nonzero(empty) == 0
    for rank, result in enumerate(results, 4):
        assert torch.equal(result["window_cache"], tensors["window_cache"][rank])
        assert torch.count_nonzero(result["candidate_mask"]) == 0
        assert bool((result["topk_indices"] == -1).all())


@pytest.mark.parametrize("replicated", [False, True])
def test_reference_sums_fp32_partials_before_narrowing(monkeypatch, c2a, replicated):
    monkeypatch.setattr(c2a, "TP_SIZE", 2)
    values = torch.tensor([1.00390625, -1.0]).reshape(2, 1, 1)

    def reference(inputs, *, fp32_output=False):
        output = inputs["partial"]
        return {"output": output if fp32_output else output.bfloat16()}

    monkeypatch.setattr(c2a, "MODES", {"reuse": (("x", "partial"), (), (), reference)})
    counts = torch.tensor(1) if replicated else torch.ones(2, 1, dtype=torch.int32)
    inputs = {"num_tokens": counts, "partial": values}
    result, _ = c2a.reference_attention("reuse", 1, inputs, torch.zeros(1, 1), range(2))
    assert result.item() == 0.00390625
    staged = c2a.StagedAttentionReference("reuse", 1, {})
    actual = {
        "attn_input": torch.zeros(2, 1, 1),
        "attn_output": torch.empty(2, 1, 1, dtype=torch.bfloat16),
    }
    monkeypatch.setattr(staged, "state_names", ())
    output = staged(inputs, actual)["attn_output"]
    assert output[0].item() == 0.00390625
    assert output[1].item() == (0.00390625 if replicated else 0.0)


def test_full_rope_reference_uses_global_active_positions(monkeypatch, c2a):
    monkeypatch.setattr(c2a, "TP_SIZE", 1)
    calls = []

    def reference(inputs, *, fp32_output=False):
        calls.append(inputs)
        assert fp32_output
        assert inputs["rope_cos"].flatten().tolist() == [13, 11]
        assert inputs["compressed_rope_cos"].flatten().tolist() == [1, 12]
        return {"output": inputs["x"].float(), "topk_indices": torch.zeros(2, c2a.C.INDEX_TOPK)}

    names = ("x", "rope_cos", "compressed_rope_cos")
    monkeypatch.setattr(c2a, "MODES", {"full": (names, (), (), reference)})
    monkeypatch.setattr(c2a, "FULL_ROPE_NAMES", {
        "rope_cos": "freqs_cos", "compressed_rope_cos": "compressed_freqs_cos",
    })
    tensors = {
        "num_tokens": torch.tensor([[2]]),
        "position_ids": torch.tensor([[3, 1, 999]]),
        "compressed_rope_positions": torch.tensor([[-1, 2, 999]]),
        "freqs_cos": torch.arange(10, 14).reshape(1, 4, 1),
        "compressed_freqs_cos": torch.arange(10, 14).reshape(1, 4, 1),
    }
    output, results = c2a.reference_attention("full", 1, tensors, torch.ones(3, 8), range(1))
    assert len(calls) == 1
    assert output.shape == (3, 8) and not output[2].any()
    assert bool((results[0]["topk_indices"][2] == -1).all())
