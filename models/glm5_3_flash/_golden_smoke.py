# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small deterministic CPU fixtures for directly executing operator goldens.

Every operator file's script-entry guard calls one runner from here, so that
``python models/glm5_3_flash/<file>.py`` proves the reference before any kernel body
exists. The a2a3 daily CI selects a file purely by grepping for that guard, so a
file gains one only once its golden is real — a file whose golden is still a stub
must stay without it, or the sweep runs an operator that cannot pass. This module
deliberately spells the guard nowhere, so it is not selected as a case itself.

The fixtures are deliberately tiny and shape-reduced. They check that a reference
runs, keeps its declared shapes and dtypes, and satisfies the invariants an
operator owner can rely on. They are not accuracy tests — those belong to the
golden harness in ``golden/`` once the kernel exists.
"""

import torch

from models.glm5_3_flash.config import BLOCK_SIZE, FLASH


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _report(name: str) -> None:
    print(f"[GOLDEN] PASS {name}")


def run_mhc_goldens(golden_mixes, golden_pre, golden_post, golden_head) -> None:
    """Exercise the four mHC references on one reduced hyper-connection site."""
    torch.manual_seed(11)
    tokens, width = 6, FLASH.hc_mult
    hidden = 64
    hc_dim = width * hidden
    mix_hc = FLASH.mix_hc

    x_hc = torch.randn(tokens, width, hidden, dtype=torch.bfloat16)
    function = torch.randn(mix_hc, hc_dim) * 0.02
    scale = torch.randn(3)
    base = torch.randn(mix_hc) * 0.1

    pre_mix, post_mix, residual_mix = golden_mixes(x_hc, function, scale, base)
    _check(pre_mix.shape == (tokens, width), f"pre_mix shape {tuple(pre_mix.shape)}")
    _check(post_mix.shape == (tokens, width), f"post_mix shape {tuple(post_mix.shape)}")
    _check(
        residual_mix.shape == (tokens, width, width),
        f"residual_mix shape {tuple(residual_mix.shape)}",
    )
    # Sinkhorn must leave the residual mix doubly stochastic.
    for axis in (-1, -2):
        deviation = (residual_mix.sum(dim=axis) - 1.0).abs().max()
        _check(deviation < 1e-3, f"residual_mix is not doubly stochastic on axis {axis}: {deviation}")
    _check(bool((pre_mix > 0).all()), "pre_mix must be positive")

    sublayer_input = golden_pre(x_hc, pre_mix)
    _check(
        sublayer_input.shape == (tokens, hidden),
        f"mhc_pre shape {tuple(sublayer_input.shape)}",
    )
    _check(sublayer_input.dtype is torch.bfloat16, f"mhc_pre dtype {sublayer_input.dtype}")

    streams = golden_post(sublayer_input, x_hc, post_mix, residual_mix)
    _check(streams.shape == x_hc.shape, f"mhc_post shape {tuple(streams.shape)}")
    _check(streams.dtype is torch.bfloat16, f"mhc_post dtype {streams.dtype}")

    collapsed = golden_head(x_hc)
    _check(collapsed.shape == (tokens, hidden), f"mhc_head shape {tuple(collapsed.shape)}")
    _check(collapsed.dtype is torch.bfloat16, f"mhc_head dtype {collapsed.dtype}")
    # GLM-5.3-Flash collapses with an unweighted mean, not a learned head.
    expected = x_hc.float().mean(dim=-2).to(torch.bfloat16)
    _check(torch.equal(collapsed, expected), "mhc_head must be the unweighted stream mean")

    _report("mhc")


def run_moe_gate_golden(golden_gate) -> None:
    """Exercise the sigmoid ``noaux_tc`` router on a reduced expert count."""
    torch.manual_seed(13)
    tokens, hidden, experts = 8, 64, 32
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    weight = torch.randn(experts, hidden)
    correction_bias = torch.randn(experts)

    weights, indices = golden_gate(x, weight, correction_bias)
    topk = FLASH.num_experts_per_tok
    _check(weights.shape == (tokens, topk), f"router weight shape {tuple(weights.shape)}")
    _check(indices.shape == (tokens, topk), f"router index shape {tuple(indices.shape)}")
    _check(indices.dtype is torch.int32, f"router index dtype {indices.dtype}")
    _check(bool((indices >= 0).all() and (indices < experts).all()), "router index out of range")
    for row in indices.tolist():
        _check(len(set(row)) == topk, f"router selected a duplicate expert: {row}")
    if FLASH.norm_topk_prob:
        total = weights.sum(dim=-1) / FLASH.routed_scaling_factor
        _check((total - 1.0).abs().max() < 1e-4, f"router weights are not normalised: {total}")

    _report("moe_gate")


def run_swiglu_golden(golden_swiglu) -> None:
    """Check the clamped SwiGLU's saturation behaviour at the configured limit."""
    torch.manual_seed(17)
    limit = FLASH.swiglu_limit
    gate_value = torch.tensor([[-100.0, 0.0, 100.0]])
    up_value = torch.tensor([[-100.0, 1.0, 100.0]])

    hidden = golden_swiglu(gate_value, up_value)
    _check(hidden.shape == gate_value.shape, f"swiglu shape {tuple(hidden.shape)}")
    # `up` is clamped on both sides, `gate` only from above.
    expected_high = torch.nn.functional.silu(torch.tensor(limit)) * limit
    _check(
        torch.allclose(hidden[0, 2], expected_high, atol=1e-5),
        f"swiglu did not clamp the upper tail: {hidden[0, 2]} vs {expected_high}",
    )
    expected_low = torch.nn.functional.silu(torch.tensor(-100.0)) * -limit
    _check(
        torch.allclose(hidden[0, 0], expected_low, atol=1e-5),
        f"swiglu clamped the gate from below: {hidden[0, 0]} vs {expected_low}",
    )

    _report("swiglu")


def run_norm_goldens(golden_rms_norm, golden_rms_norm_gated, golden_l2norm) -> None:
    """Exercise the three normalisations the backbone shares."""
    torch.manual_seed(19)
    tokens, hidden = 5, 64
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    weight = torch.randn(hidden, dtype=torch.bfloat16)

    normalized = golden_rms_norm(x, weight)
    _check(normalized.shape == x.shape, f"rms_norm shape {tuple(normalized.shape)}")
    _check(normalized.dtype is x.dtype, f"rms_norm dtype {normalized.dtype}")

    heads, head_dim = 4, 16
    value = torch.randn(tokens, heads, head_dim)
    gate_value = torch.randn(tokens, heads, head_dim)
    gated = golden_rms_norm_gated(value, torch.ones(head_dim), gate_value)
    _check(gated.shape == value.shape, f"rms_norm_gated shape {tuple(gated.shape)}")
    ungated = golden_rms_norm_gated(value, torch.ones(head_dim), torch.full_like(gate_value, 1e9))
    _check(
        torch.allclose(gated, ungated * torch.sigmoid(gate_value), atol=1e-4),
        "rms_norm_gated must apply a sigmoid gate after the norm",
    )

    unit = golden_l2norm(torch.randn(tokens, head_dim))
    _check(
        (unit.square().sum(dim=-1) - 1.0).abs().max() < 1e-4,
        "l2norm must return unit rows",
    )

    _report("norms")


def run_quantization_goldens(quantize_per_channel, quantize_per_token, dynamic_linear) -> None:
    """Check the W8A8 references reconstruct a float matmul within int8 error."""
    torch.manual_seed(23)
    tokens, in_features, out_features = 6, 128, 64
    x = torch.randn(tokens, in_features, dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    weight_int8, weight_scale = quantize_per_channel(weight)
    _check(weight_int8.dtype is torch.int8, f"weight dtype {weight_int8.dtype}")
    _check(
        weight_scale.shape == (out_features, 1),
        f"per-channel scale shape {tuple(weight_scale.shape)}",
    )

    x_int8, token_scale = quantize_per_token(x)
    _check(x_int8.dtype is torch.int8, f"activation dtype {x_int8.dtype}")
    _check(token_scale.shape == (tokens, 1), f"per-token scale shape {tuple(token_scale.shape)}")

    reference = torch.nn.functional.linear(x.float(), weight.float())
    quantized = dynamic_linear(x, weight_int8, weight_scale, out_dtype=torch.float32)
    error = (quantized - reference).abs().max() / reference.abs().max()
    _check(float(error) < 0.05, f"W8A8 reconstruction error is too large: {float(error)}")

    _report("quantization")




def run_mla_prolog_goldens(golden_prolog, golden_split, golden_absorb_query, golden_absorb_output) -> None:
    """Exercise the MLA prolog and both weight-time absorptions on reduced shapes.

    ``qk_head_dim``, ``v_head_dim`` and ``kv_lora_rank`` stay at their real values
    because :func:`golden_split_kv_b` reads them from the config; the hidden size, the
    query rank, the head count and the token count are reduced.
    """
    torch.manual_seed(23)
    tokens, hidden, q_lora, heads = 4, 64, 48, 2
    qk, v_dim, kv_lora = FLASH.qk_head_dim, FLASH.v_head_dim, FLASH.kv_lora_rank

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    w_q_a = torch.randn(q_lora, hidden, dtype=torch.bfloat16) * 0.05
    q_a_norm = torch.randn(q_lora, dtype=torch.bfloat16)
    w_q_b = torch.randn(heads * qk, q_lora, dtype=torch.bfloat16) * 0.05
    w_kv_a = torch.randn(kv_lora, hidden, dtype=torch.bfloat16) * 0.05
    kv_a_norm = torch.randn(kv_lora, dtype=torch.bfloat16)

    q_resid, query, kv_latent = golden_prolog(x, w_q_a, q_a_norm, w_q_b, w_kv_a, kv_a_norm)
    _check(q_resid.shape == (tokens, q_lora), f"q_resid shape {tuple(q_resid.shape)}")
    _check(query.shape == (tokens, heads, qk), f"query shape {tuple(query.shape)}")
    _check(kv_latent.shape == (tokens, kv_lora), f"kv_latent shape {tuple(kv_latent.shape)}")
    for name, value in (("q_resid", q_resid), ("query", query), ("kv_latent", kv_latent)):
        _check(value.dtype is torch.bfloat16, f"{name} dtype {value.dtype}")
    # The query must be the projection of the published q_resid.
    replayed = torch.nn.functional.linear(q_resid.float(), w_q_b.float())
    replayed = replayed.unflatten(-1, (heads, qk)).to(torch.bfloat16)
    _check(torch.equal(query, replayed), "query must be projected from the published q_resid")

    w_kv_b = torch.randn(heads * (qk + v_dim), kv_lora, dtype=torch.bfloat16) * 0.02
    w_k, w_v = golden_split(w_kv_b)
    _check(w_k.shape == (heads, qk, kv_lora), f"w_k shape {tuple(w_k.shape)}")
    _check(w_v.shape == (heads, v_dim, kv_lora), f"w_v shape {tuple(w_v.shape)}")

    # Absorbing the key half must leave every score unchanged.
    latent = torch.randn(5, kv_lora, dtype=torch.bfloat16)
    expanded_key = torch.einsum("nk,hdk->nhd", latent.float(), w_k.float())
    scores_expanded = torch.einsum("thd,nhd->thn", query.float(), expanded_key)
    absorbed_query = golden_absorb_query(query, w_k)
    _check(
        absorbed_query.shape == (tokens, heads, kv_lora),
        f"absorbed query shape {tuple(absorbed_query.shape)}",
    )
    scores_absorbed = torch.einsum("thk,nk->thn", absorbed_query.float(), latent.float())
    deviation = (scores_absorbed - scores_expanded).abs().max() / scores_expanded.abs().max()
    _check(float(deviation) < 2e-2, f"query absorption changed the scores: {float(deviation)}")

    # Absorbing the value half must leave the projected output unchanged.
    model_dim = 32
    context = torch.randn(tokens, heads, kv_lora, dtype=torch.bfloat16)
    w_o = torch.randn(model_dim, heads * v_dim, dtype=torch.bfloat16) * 0.05
    expanded_out = torch.einsum("thk,hvk->thv", context.float(), w_v.float())
    reference = torch.nn.functional.linear(expanded_out.flatten(-2), w_o.float())
    w_o_absorbed = golden_absorb_output(w_v, w_o)
    _check(
        w_o_absorbed.shape == (model_dim, heads * kv_lora),
        f"absorbed o_proj shape {tuple(w_o_absorbed.shape)}",
    )
    projected = torch.nn.functional.linear(context.float().flatten(-2), w_o_absorbed.float())
    deviation = (projected - reference).abs().max() / reference.abs().max()
    _check(float(deviation) < 2e-2, f"output absorption changed the projection: {float(deviation)}")

    _report("mla_prolog")


def run_mla_cache_golden(golden_cache_write) -> None:
    """Check that the latent cache write scatters by slot and touches nothing else."""
    torch.manual_seed(29)
    rows, kv_lora = 16, 8
    cache = torch.randn(rows, kv_lora, dtype=torch.bfloat16)
    original = cache.clone()
    latent = torch.randn(4, kv_lora, dtype=torch.bfloat16)
    # Row 2 owns no cache position.
    slots = torch.tensor([5, 0, -1, 11], dtype=torch.int32)

    updated = golden_cache_write(cache, latent, slots)
    _check(updated.shape == cache.shape, f"cache shape {tuple(updated.shape)}")
    _check(updated.dtype is cache.dtype, f"cache dtype {updated.dtype}")
    _check(torch.equal(cache, original), "the reference must not write the cache in place")
    written = [slot for slot in slots.tolist() if slot >= 0]
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        _check(torch.equal(updated[slot], latent[row]), f"slot {slot} did not receive its row")
    untouched = [index for index in range(rows) if index not in written]
    _check(
        torch.equal(updated[untouched], original[untouched]),
        "the write disturbed rows outside its slot mapping",
    )
    # A -1 slot must not wrap into the tail of the pool.
    _check(torch.equal(updated[-1], original[-1]), "a -1 slot wrapped into the last row")

    _report("mla_cache")


def run_indexer_cache_golden(golden_cache_write) -> None:
    """Check that the indexer state write packs both halves and skips ``-1`` slots."""
    torch.manual_seed(31)
    rows, dim = 16, 8
    cache = torch.randn(rows, 2 * dim, dtype=torch.float32)
    original = cache.clone()
    index_k = torch.randn(4, dim, dtype=torch.bfloat16)
    gate_scores = torch.randn(4, dim, dtype=torch.float32)
    # Row 2 owns no cache position.
    slots = torch.tensor([5, 0, -1, 11], dtype=torch.int32)

    updated = golden_cache_write(cache, index_k, gate_scores, slots)
    _check(updated.shape == cache.shape, f"cache shape {tuple(updated.shape)}")
    _check(updated.dtype is cache.dtype, f"cache dtype {updated.dtype}")
    _check(torch.equal(cache, original), "the reference must not write the cache in place")
    written = [slot for slot in slots.tolist() if slot >= 0]
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        # The key half is widened from BF16, which is exact; the gate half is copied.
        _check(
            torch.equal(updated[slot, :dim], index_k[row].to(torch.float32)),
            f"slot {slot} did not receive its key half",
        )
        _check(
            torch.equal(updated[slot, dim:], gate_scores[row]),
            f"slot {slot} did not receive its gate half",
        )
    untouched = [index for index in range(rows) if index not in written]
    _check(
        torch.equal(updated[untouched], original[untouched]),
        "the write disturbed rows outside its slot mapping",
    )
    # A -1 slot must not wrap into the tail of the pool.
    _check(torch.equal(updated[-1], original[-1]), "a -1 slot wrapped into the last row")

    _report("indexer_cache")


def _sparse_attention_fixture(seed: int):
    """Build one small paged selection: cache, block table, indices and query."""
    torch.manual_seed(seed)
    tokens, heads, requests, width = 5, 2, 2, 6
    qk, kv_lora, v_dim = FLASH.qk_head_dim, 24, 16
    block_table = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    latent_cache = torch.randn(4 * BLOCK_SIZE, kv_lora, dtype=torch.bfloat16)
    request_ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    # Logical positions inside the two pages each request owns, with a padded tail.
    topk_indices = torch.tensor(
        [
            [0, 1, 5, 130, -1, -1],
            [0, 3, 129, 200, 7, -1],
            [2, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4, 5],
            [131, 6, 0, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    query = torch.randn(tokens, heads, qk, dtype=torch.bfloat16)
    w_k = torch.randn(heads, qk, kv_lora, dtype=torch.bfloat16) * 0.05
    w_v = torch.randn(heads, v_dim, kv_lora, dtype=torch.bfloat16) * 0.05
    return latent_cache, block_table, request_ids, topk_indices, query, w_k, w_v


def _selected_rows(topk_row, request_id, block_table):
    """Resolve one row's valid logical positions into physical cache rows."""
    rows = []
    for position in topk_row.tolist():
        if position < 0:
            continue
        page = block_table[request_id, position // BLOCK_SIZE].item()
        rows.append(int(page) * BLOCK_SIZE + position % BLOCK_SIZE)
    return rows


def run_prefill_sparse_attn_golden(golden_prefill) -> None:
    """Check the head-space sparse attention against an independent loop reference."""
    cache, block_table, request_ids, indices, query, w_k, w_v = _sparse_attention_fixture(31)
    tokens, heads, _ = query.shape
    v_dim = w_v.shape[1]
    scale = query.shape[-1] ** -0.5

    out = golden_prefill(query, cache, w_k, w_v, indices, block_table, request_ids)
    _check(out.shape == (tokens, heads, v_dim), f"prefill output shape {tuple(out.shape)}")
    _check(out.dtype is torch.bfloat16, f"prefill output dtype {out.dtype}")

    for token in range(tokens):
        rows = _selected_rows(indices[token], int(request_ids[token]), block_table)
        for head in range(heads):
            keys = torch.stack([w_k[head].float() @ cache[row].float() for row in rows])
            values = torch.stack([w_v[head].float() @ cache[row].float() for row in rows])
            weights = torch.softmax(keys @ query[token, head].float() * scale, dim=0)
            expected = (weights.unsqueeze(-1) * values).sum(dim=0)
            deviation = (out[token, head].float() - expected).abs().max()
            _check(
                float(deviation) < 5e-2 * max(float(expected.abs().max()), 1e-3),
                f"prefill row {token} head {head} deviates by {float(deviation)}",
            )

    # Padded slots must not reach the softmax.
    trimmed = indices[:, :4].contiguous()
    padded = torch.full((indices.shape[0], 2), -1, dtype=torch.int32)
    same = golden_prefill(
        query, cache, w_k, w_v, torch.cat([trimmed, padded], dim=1), block_table, request_ids
    )
    reference = golden_prefill(query, cache, w_k, w_v, trimmed, block_table, request_ids)
    _check(torch.equal(same, reference), "padded -1 slots changed the prefill result")

    _report("prefill_sparse_attn")


def run_decode_sparse_attn_golden(golden_decode) -> None:
    """Check the latent-space sparse attention and its scale against a loop reference."""
    cache, block_table, request_ids, indices, query, w_k, _ = _sparse_attention_fixture(37)
    tokens, heads, _ = query.shape
    kv_lora = cache.shape[-1]
    # The scale belongs to the un-absorbed head dim.
    absorbed = torch.einsum("thd,hdk->thk", query.float(), w_k.float()).to(torch.bfloat16)
    scale = query.shape[-1] ** -0.5

    out = golden_decode(absorbed, cache, indices, block_table, request_ids)
    _check(out.shape == (tokens, heads, kv_lora), f"decode output shape {tuple(out.shape)}")
    _check(out.dtype is torch.bfloat16, f"decode output dtype {out.dtype}")

    for token in range(tokens):
        rows = _selected_rows(indices[token], int(request_ids[token]), block_table)
        latent = torch.stack([cache[row].float() for row in rows])
        for head in range(heads):
            weights = torch.softmax(latent @ absorbed[token, head].float() * scale, dim=0)
            expected = (weights.unsqueeze(-1) * latent).sum(dim=0)
            deviation = (out[token, head].float() - expected).abs().max()
            _check(
                float(deviation) < 5e-2 * max(float(expected.abs().max()), 1e-3),
                f"decode row {token} head {head} deviates by {float(deviation)}",
            )

    trimmed = indices[:, :4].contiguous()
    padded = torch.full((indices.shape[0], 2), -1, dtype=torch.int32)
    same = golden_decode(
        absorbed, cache, torch.cat([trimmed, padded], dim=1), block_table, request_ids
    )
    reference = golden_decode(absorbed, cache, trimmed, block_table, request_ids)
    _check(torch.equal(same, reference), "padded -1 slots changed the decode result")

    _report("decode_sparse_attn")


def run_mla_epilog_goldens(golden_epilog_prefill, golden_epilog_decode) -> None:
    """Check both output projections keep their row-parallel FP32 partial-sum contract."""
    torch.manual_seed(41)
    tokens, heads, model_dim = 4, 2, 32
    v_dim, kv_lora = 16, 24

    attn_out = torch.randn(tokens, heads, v_dim, dtype=torch.bfloat16)
    w_o = torch.randn(model_dim, heads * v_dim, dtype=torch.bfloat16) * 0.05
    prefill = golden_epilog_prefill(attn_out, w_o)
    _check(prefill.shape == (tokens, model_dim), f"prefill epilog shape {tuple(prefill.shape)}")
    _check(prefill.dtype is torch.float32, f"prefill epilog dtype {prefill.dtype}")
    expected = torch.nn.functional.linear(attn_out.float().flatten(-2), w_o.float())
    _check(torch.allclose(prefill, expected, atol=1e-5), "prefill epilog is not a plain projection")

    latent_out = torch.randn(tokens, heads, kv_lora, dtype=torch.bfloat16)
    w_o_absorbed = torch.randn(model_dim, heads * kv_lora, dtype=torch.bfloat16) * 0.05
    decode = golden_epilog_decode(latent_out, w_o_absorbed)
    _check(decode.shape == (tokens, model_dim), f"decode epilog shape {tuple(decode.shape)}")
    _check(decode.dtype is torch.float32, f"decode epilog dtype {decode.dtype}")

    _report("mla_epilog")


__all__ = [
    "run_decode_sparse_attn_golden",
    "run_indexer_cache_golden",
    "run_mhc_goldens",
    "run_mla_cache_golden",
    "run_mla_epilog_goldens",
    "run_mla_prolog_goldens",
    "run_moe_gate_golden",
    "run_norm_goldens",
    "run_prefill_sparse_attn_golden",
    "run_quantization_goldens",
    "run_swiglu_golden",
]
