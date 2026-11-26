# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay representative layers through the production resident prefill graph."""

import dataclasses
import json
import time

import torch
import torch.nn.functional as F
from golden import ratio_allclose

from .config import FLASH as M
from .host import prepare_prefill_inputs, prepare_resident_inputs
from .hc import golden_hc_collapse, golden_mhc_mixes
from .engram import golden_engram
from .gate import golden_gate_reference
from .reference import W8A8Reference
from .rmsnorm import golden_rms_norm
from .quant import dequantize_mxfp4_cache, dequantize_mxfp8_cache


class LayerValidationError(ValueError):
    """Report completed layer comparisons that did not satisfy acceptance."""


def reset_caches(runtime, ranks, host_cache):
    """Reset the same seven resident allocations for the next request-slot reuse."""
    for rank in range(16):
        for name, value in host_cache[rank].items():
            runtime.copy_to(ranks[0][rank][name].data_ptr, value.data_ptr(),
                            value.numel() * value.element_size(), worker_id=rank)


def residual_metrics(actual, expected):
    """Accumulate residual norms and the dot product in FP64."""
    a = actual.flatten().double()
    b = expected.flatten().double()
    relative = float((a - b).norm() / b.norm().clamp_min(1e-30))
    cosine = float(F.cosine_similarity(a, b, dim=0))
    return relative, cosine


def validate_mhc(runtime, ranks, checkpoint, layer, owner, actual, *, diagnostics_path=None):
    """Compare FFN pre-mix coefficients on the captured production attention residual."""
    source = torch.empty((8192, M.hc_mult, M.hidden_size), dtype=torch.float32)
    runtime.copy_from(source.data_ptr(), ranks[0][owner]["attn_hc_out"].data_ptr,
                      source.numel() * source.element_size(), worker_id=owner)
    prefix = f"layers.{layer}."
    expected, _, _ = golden_mhc_mixes(source, checkpoint.tensor(prefix + "hc_ffn_fn"),
                                      checkpoint.tensor(prefix + "hc_ffn_scale"),
                                      checkpoint.tensor(prefix + "hc_ffn_base"))
    checker = ratio_allclose(atol=2.5e-5, rtol=5e-3)
    passed, detail = checker(actual, expected, inputs={}, actual_outputs={}, expected_outputs={}, rtol=0, atol=0)
    relative, cosine = residual_metrics(actual, expected)
    if diagnostics_path is not None:
        torch.save(dict(source=source, actual_pre_mix=actual, expected_pre_mix=expected), diagnostics_path)
    return dict(passed=bool(passed), detail=detail, relative_l2=relative, cosine=cosine,
                reference="independent mHC golden on captured production FP32 attention residual")


def validate_routes(runtime, ranks, checkpoint, layer, owner, received, *, bias=None, diagnostics_dir=None):
    """Check captured production gate inputs, six routes and weights against the CPU golden."""
    matrix = checkpoint.tensor(f"layers.{layer}.ffn.gate.weight").float()
    if bias is None:
        bias = checkpoint.tensor(f"layers.{layer}.ffn.gate.bias")
    route_counts = torch.zeros(M.n_routed_experts, dtype=torch.int64)
    records = []
    for rank in range(16):
        indices = torch.empty((40, 2048, 6), dtype=torch.int32)
        weights = torch.empty((40, 2048, 6), dtype=torch.float32)
        for name, value in (("moe_route_indices", indices), ("moe_route_weights", weights)):
            runtime.copy_from(value.data_ptr(), ranks[0][rank][name].data_ptr,
                              value.numel() * value.element_size(), worker_id=rank)
        if rank // 4 == owner // 4:
            source = torch.empty((2048, M.hidden_size), dtype=torch.bfloat16)
            runtime.copy_from(source.data_ptr(), ranks[0][rank]["moe_gate_inputs"].data_ptr,
                              source.numel() * source.element_size(), worker_id=rank)
            expected_weights, expected_indices = golden_gate_reference(source, matrix, bias)
            actual_ids, actual_order = indices[layer].sort(dim=1)
            expected_ids, expected_order = expected_indices.sort(dim=1)
            indices_ok = torch.equal(actual_ids, expected_ids)
            checker = ratio_allclose(atol=1e-6, rtol=3e-5, max_error_ratio=0.0)
            actual_paired = weights[layer].gather(1, actual_order)
            expected_paired = expected_weights.gather(1, expected_order)
            weights_ok, detail = checker(actual_paired, expected_paired, inputs={}, actual_outputs={},
                                         expected_outputs={}, rtol=0, atol=0)
            unique = bool((indices[layer].sort(dim=1).values.diff(dim=1) > 0).all())
            route_counts += torch.bincount(indices[layer].long().flatten(), minlength=M.n_routed_experts)
            records.append(dict(rank=rank, indices_exact=torch.equal(indices[layer], expected_indices),
                                expert_set_exact=indices_ok, weights_passed=bool(weights_ok),
                                weights_detail=detail, six_distinct_experts=unique,
                                passed=indices_ok and bool(weights_ok) and unique))
            if not records[-1]["passed"] and diagnostics_dir is not None:
                path = diagnostics_dir / f"failed_gate_layer_{layer:02d}_dp{owner // 4}_rank{rank:02d}.pt"
                torch.save(dict(source=source, gate_weight=matrix, gate_bias=bias,
                                actual_indices=indices[layer], actual_weights=weights[layer],
                                expected_indices=expected_indices, expected_weights=expected_weights), path)
                records[-1]["diagnostics"] = path.name
        else:
            empty = not bool(indices[layer].count_nonzero() or weights[layer].count_nonzero())
            records.append(dict(rank=rank, empty_routes=empty, passed=empty))
    counts_match = torch.equal(route_counts.reshape(16, 24).sum(1), received)
    return dict(passed=all(record["passed"] for record in records) and counts_match,
                reference="independent gate golden on captured production BF16 normalized inputs",
                received_counts_match_routes=counts_match, expert_counts=route_counts.tolist(), ranks=records)


def concentrated_fixture(fixture, target, reference_dir, weights, tokenizer, request, rope):
    """Freeze layer zero with all six experts on one remote EP rank."""
    path = reference_dir / f"layer_00_concentrated_rank{target:02d}.pt"
    if path.exists():
        return {**fixture, **torch.load(path, map_location="cpu", weights_only=True)}
    bias = torch.full((M.n_routed_experts,), -1e6, dtype=torch.float32)
    bias[target * 24:target * 24 + 6] = 1e6
    reference = W8A8Reference(weights.checkpoint, tokenizer)
    reference.overrides["layers.0.ffn.gate.bias"] = bias
    cache = {name: value.clone() for name, value in fixture["cache"].items()}
    embedding = weights.lookup_rows(torch.tensor(request["input_ids"]))
    values = prepare_prefill_inputs(embedding, 8192, max_seq_len=8200, cache_state=cache, rope_tables=rope)
    reference.cache, reference.rope = cache, rope
    output, pre, counts = reference.layer_forward(fixture["hidden"], fixture["pre_mix"], 0, values)
    if int(counts[target * 24:target * 24 + 6].sum()) != 49152:
        raise ValueError("concentrated golden did not select the six remote experts")
    frozen = dict(output=output, output_pre_mix=pre, correction_bias=bias)
    torch.save(frozen, path)
    return {**fixture, **frozen}


def _engram_input(reference, weights, request, layer, hidden):
    prefix = f"layers.{layer}.engram."
    lookup = reference.engram.lookup(torch.tensor(request["input_ids"]), 8192, request_id=0)[M.engram_layer_ids.index(layer)]
    gate = reference.tensor(prefix + "q_weight").float() * reference.tensor(prefix + "k_weight").float()
    engrammed = torch.empty_like(hidden)
    golden_engram(dict(source=hidden, lookup=lookup, output=engrammed, weight=gate,
                       wkv_weight=reference.tensor(prefix + "wkv.weight").t(),
                       rotation=weights.checkpoint.rotation_blocks()))
    return engrammed


def freeze_cache_reference(layer, fixture, reference_dir, weights, tokenizer, request, rope):
    """Freeze independent post-attention caches for each representative residual fixture."""
    path = reference_dir / f"layer_{layer:02d}_cache_reference_v2.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    reference = W8A8Reference(weights.checkpoint, tokenizer)
    embedding = weights.lookup_rows(torch.tensor(request["input_ids"]))
    cache = {name: value.clone() for name, value in fixture["cache"].items()}
    values = prepare_prefill_inputs(embedding, 8192, max_seq_len=8200, cache_state=cache, rope_tables=rope)
    reference.cache, reference.rope = cache, rope
    hidden, pre = fixture["hidden"], fixture["pre_mix"]
    if layer in M.engram_layer_ids:
        hidden = _engram_input(reference, weights, request, layer, hidden)
    plan = M.layer_config(layer)
    if plan.mode in ("reuse", "reindex"):
        reference.selected = fixture["topk_indices"]
        reference.pool = fixture["pool_block_ids"], fixture["pool_block_counts"]
    collapsed = golden_hc_collapse(hidden, pre).bfloat16()
    norm = weights.checkpoint.tensor(f"layers.{layer}.attn_norm.weight")
    reference.attention(golden_rms_norm(collapsed, norm), layer, values)
    frozen = dict(cache=cache, policy=dict(relative_l2_max=0.03, cosine_min=0.999,
                                         state_relative_l2_max=0.01, state_cosine_min=0.9999))
    torch.save(frozen, path)
    print("A3_CACHE_REFERENCE_READY", layer, flush=True)
    return frozen


def validate_cache_values(runtime, ranks, fixture, layer, owner):
    """Compare decoded native cache values to the frozen independent attention result."""
    frozen = fixture["cache_reference"]
    actual = {}
    for name, expected in frozen["cache"].items():
        value = torch.empty_like(expected)
        runtime.copy_from(value.data_ptr(), ranks[0][owner][name].data_ptr,
                          value.numel() * value.element_size(), worker_id=owner)
        actual[name] = value
    families = [("window", layer, 32, "bf16")]
    plan = M.layer_config(layer)
    if layer >= 2:
        source = M.kv_source_layer_ids.index(plan.kv_source_layer_id)
        families += [("compressed", source, 16, "bf16"), ("index", source, 32, "e8m0")]
    records = []
    for name, slot, group, encoding in families:
        decoded = []
        for caches in (actual, frozen["cache"]):
            payload, scale = caches[name + "_cache"][slot], caches[name + "_scales"][slot]
            if name == "window":
                value = dequantize_mxfp8_cache(payload.view(torch.float8_e4m3fn), scale)
            else:
                value = dequantize_mxfp4_cache(payload, scale, group, encoding)
            decoded.append(value)
        relative, cosine = residual_metrics(*decoded)
        if not bool(decoded[1].count_nonzero()):
            cosine = float(torch.equal(*decoded))
        passed = relative <= frozen["policy"]["relative_l2_max"] and cosine >= frozen["policy"]["cosine_min"]
        records.append(dict(cache=name, source=slot, relative_l2=relative, cosine=cosine, passed=passed))
    if layer >= 2 and plan.compression_ratio == 2:
        relative, cosine = residual_metrics(actual["state_cache"], frozen["cache"]["state_cache"])
        if not bool(frozen["cache"]["state_cache"].count_nonzero()):
            cosine = float(torch.equal(actual["state_cache"], frozen["cache"]["state_cache"]))
        passed = relative <= frozen["policy"]["state_relative_l2_max"]
        passed &= cosine >= frozen["policy"]["state_cosine_min"]
        records.append(dict(cache="state", relative_l2=relative, cosine=cosine, passed=passed))
    return dict(passed=all(record["passed"] for record in records), policy=frozen["policy"], values=records)


def make_layer_fixture(layer, reference_dir, weights, tokenizer, request, rope):
    """Load frozen residuals and independently regenerate the reused source caches."""
    snapshot = torch.load(reference_dir / f"layer_{layer:02d}.pt", map_location="cpu", weights_only=True)
    hidden, pre = snapshot["hidden"], snapshot["pre_mix"]
    if layer in M.engram_layer_ids and snapshot.get("hidden_stage") != "layer_input":
        previous_path = reference_dir / f"layer_{layer - 1:02d}.pt"
        if not previous_path.exists():
            raise ValueError(f"Regenerate layer {layer} reference with a pre-Engram layer input")
        previous = torch.load(previous_path, map_location="cpu", weights_only=True)
        hidden, pre = previous["output"], previous["output_pre_mix"]
    reference = W8A8Reference(weights.checkpoint, tokenizer)
    embedding = weights.lookup_rows(torch.tensor(request["input_ids"]))
    values = prepare_prefill_inputs(embedding, 8192, max_seq_len=8200, rope_tables=rope)
    reference.cache = {name: value for name, value in values.items() if name.endswith(("_cache", "_scales"))}
    reference.rope = rope
    plan = M.layer_config(layer)
    if plan.mode in ("reuse", "reindex"):
        source = plan.kv_source_layer_id
        initial = torch.load(reference_dir / f"layer_{source:02d}.pt", map_location="cpu", weights_only=True)
        source_hidden = initial["hidden"]
        if source in M.engram_layer_ids and initial.get("hidden_stage") == "layer_input":
            source_hidden = _engram_input(reference, weights, request, source, source_hidden)
        collapsed = golden_hc_collapse(source_hidden, initial["pre_mix"]).bfloat16()
        norm = weights.checkpoint.tensor(f"layers.{source}.attn_norm.weight")
        reference.attention(golden_rms_norm(collapsed, norm), source, values)
    topk = torch.full((8192, M.index_topk), -1, dtype=torch.int32)
    pool_ids = values["pool_block_ids"].clone()
    pool_counts = values["pool_block_counts"].clone()
    if plan.mode in ("reuse", "reindex"):
        topk.copy_(reference.selected)
        if plan.compression_ratio == 1:
            ids, counts = reference.pool
            pool_ids[:, :ids.shape[1]].copy_(ids)
            pool_counts.copy_(counts)
    return dict(hidden=hidden, pre_mix=pre, output=snapshot["output"],
                output_pre_mix=snapshot["output_pre_mix"], cache=reference.cache,
                topk_indices=topk, pool_block_ids=pool_ids, pool_block_counts=pool_counts)


def validate_layers(runtime, programs, arguments, names, ranks, weights, engram, tokenizer,
                    request, host_cache, rope, args, config, memory, audit, *, prefill_slab_rows, records=None):
    """Run one exact layer on DP0 and DP3, then compare residuals and route conservation."""
    if not args.reference_dir:
        raise ValueError("single-layer replay requires --reference-dir")
    selected = [int(value) for value in args.validate_layers.split(",") if value]
    if any(layer not in (0, 1, 2, 3, 14, 20, 21, 24) for layer in selected):
        raise ValueError("representative layer replay supports 0, 1, 2, 3, 14, 20, 21 and 24")
    if records is None:
        records = []
    for layer in selected:
        base_fixture = make_layer_fixture(layer, args.reference_dir, weights, tokenizer, request, rope)
        base_fixture["cache_reference"] = freeze_cache_reference(
            layer, base_fixture, args.reference_dir, weights, tokenizer, request, rope,
        )
        cases = [(0, None), (3, None)]
        if layer == 0:
            cases += [(0, 8), (3, 4)]
        for dp, target in cases:
            fixture = base_fixture
            if target is not None:
                fixture = concentrated_fixture(base_fixture, target, args.reference_dir, weights, tokenizer, request, rope)
            owner = dp * 4
            memory.phase = f"layer_{layer}_dp{dp}"
            for rank in range(16):
                active = rank // 4 == dp
                if target is not None:
                    bias = fixture["correction_bias"]
                    runtime.copy_to(ranks[0][rank]["correction_bias"].data_ptr, bias.data_ptr(),
                                    bias.numel() * bias.element_size(), worker_id=rank)
                values = prepare_resident_inputs(weights, request["input_ids"] if active else [], 8192,
                                                 max_seq_len=8200, cache_state=host_cache[rank],
                                                 rope_tables=rope, engram=engram, request_id=rank,
                                                 moe_slab_rows=prefill_slab_rows)
                values["layer_range"] = torch.tensor([layer, layer + 1], dtype=torch.int32)
                values["logit_rows"].fill_(-1)
                values["x_hc_out"] = fixture["hidden"] if active else torch.zeros_like(fixture["hidden"])
                if active:
                    values["initial_pre"] = fixture["pre_mix"]
                    for name in ("topk_indices", "pool_block_ids", "pool_block_counts"):
                        values[name] = fixture[name]
                for name, value in (fixture["cache"] if active else host_cache[rank]).items():
                    values[name] = value
                for name, value in values.items():
                    if name not in ranks[0][rank] or name in weights.tensor_layouts or name in ("freqs_cos", "freqs_sin"):
                        continue
                    runtime.copy_to(ranks[0][rank][name].data_ptr, value.data_ptr(),
                                    value.numel() * value.element_size(), worker_id=rank)
            started = time.monotonic()
            print("A3_LAYER_BEGIN", json.dumps(dict(layer=layer, dp_rank=dp, concentrated_rank=target)), flush=True)
            try:
                measured = dataclasses.replace(config, enable_scope_stats=True) if layer == 24 else config
                runtime.run(programs[0], *[arguments[0][name] for name in names[0]], config=measured)
            except Exception as error:
                records.append(dict(layer=layer, dp_rank=dp, passed=False, error=str(error)))
                raise
            elapsed = time.monotonic() - started
            actual = torch.empty_like(fixture["output"])
            mix = torch.empty_like(fixture["output_pre_mix"])
            for name, value in (("x_hc_out", actual), ("pre_mix_out", mix)):
                runtime.copy_from(value.data_ptr(), ranks[0][owner][name].data_ptr,
                                  value.numel() * value.element_size(), worker_id=owner)
            relative, cosine = residual_metrics(actual, fixture["output"])
            gate = ratio_allclose(atol=2.5e-5, rtol=5e-3)
            mix_ok, mix_detail = gate(mix, fixture["output_pre_mix"], inputs={}, actual_outputs={},
                                       expected_outputs={}, rtol=0, atol=0)
            totals = torch.zeros(16, 3, dtype=torch.int64)
            for rank in range(16):
                device_counts = ranks[0][rank]["moe_statistics"]
                counters = torch.empty(device_counts.shape, dtype=torch.int32)
                runtime.copy_from(counters.data_ptr(), device_counts.data_ptr,
                                  counters.numel() * counters.element_size(), worker_id=rank)
                totals[rank] = counters[layer].sum(0)
            global_counts = totals.sum(0).tolist()
            conserved = global_counts == [49152, 49152, 49152]
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(mix).all())
            record = dict(layer=layer, dp_rank=dp, concentrated_rank=target,
                          seconds=elapsed, relative_l2=relative, cosine=cosine,
                          pre_mix_passed=bool(mix_ok), pre_mix_detail=mix_detail, finite=finite,
                          route_counts=global_counts, route_counts_per_rank=totals.tolist(),
                          passed=finite and relative <= 0.01 and cosine >= 0.9999 and bool(mix_ok) and conserved)
            record["routes"] = validate_routes(runtime, ranks, weights.checkpoint, layer, owner, totals[:, 1],
                                               bias=fixture.get("correction_bias"), diagnostics_dir=args.reference_dir)
            record["passed"] &= record["routes"]["passed"]
            diagnostics_path = None
            if not record["pre_mix_passed"]:
                diagnostics_path = args.reference_dir / f"failed_mhc_layer_{layer:02d}_dp{dp}.pt"
            record["mhc_operator"] = validate_mhc(runtime, ranks, weights.checkpoint, layer, owner, mix,
                                                  diagnostics_path=diagnostics_path)
            record["passed"] &= record["mhc_operator"]["passed"]
            record["cache_values"] = validate_cache_values(runtime, ranks, fixture, layer, owner)
            record["passed"] &= record["cache_values"]["passed"]
            record["cache_checks"] = audit(runtime, ranks, host_cache, owner, 0)
            if target is not None:
                expected = [0] * 16
                expected[target] = 49152
                record["concentrated_receiver_passed"] = totals[:, 1].tolist() == expected
                record["passed"] &= record["concentrated_receiver_passed"]
                bias = weights.checkpoint.tensor("layers.0.ffn.gate.bias")
                for rank in range(16):
                    runtime.copy_to(ranks[0][rank]["correction_bias"].data_ptr, bias.data_ptr(),
                                    bias.numel() * bias.element_size(), worker_id=rank)
            records.append(record)
            print("A3_LAYER_VALIDATION", json.dumps(record), flush=True)
            if not record["passed"]:
                torch.save(dict(actual=actual, pre_mix=mix, expected=fixture),
                           args.reference_dir / f"failed_layer_{layer:02d}_dp{dp}_rank{target}.pt")
    reset_caches(runtime, ranks, host_cache)
    failed = [record for record in records if not record["passed"]]
    if failed:
        raise LayerValidationError(f"{len(failed)} representative layer cases failed; see layer_validation")
    return records
