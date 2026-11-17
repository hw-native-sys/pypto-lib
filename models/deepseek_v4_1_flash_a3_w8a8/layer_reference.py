# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared host suite code for the prefill/decode layer assembly.

The layer suites validate one decoder layer built on the A5-shaped paged
interface: the mode chains publish into the window/compressed/index caches and
the layer adds the FFN sublayer.  This module owns the pieces both phases share:

* the bounded fixtures (one paged request plan, per-source paged caches, the
  MoE windows sized for one ``MOE_TILE`` tile),
* the composed Torch golden for the whole layer, built from the shared operator
  goldens (``mla_proj``, ``compressor``, ``indexer``, ``paged_cache``,
  ``attention``, ``o_proj``, ``hc``, ``engram``, ``moe_exchange``),
* the staged references and comparators.

Every staged reference re-runs one operator on the device's own upstream
intermediate, so its residual is that operator's arithmetic floor rather than
the accumulated error of the whole layer.
"""

import torch

from utils import compressed_source_positions

from config import (
    BLOCK_SIZE, CANDIDATE_POOL_BLOCKS, FLASH as M, LOCAL_HEADS, TP_SIZE, WINDOW_KEYS,
)


# Mode codes shared with the layer dispatch.
MODE_SWA = 0
MODE_C2A_FULL = 1
MODE_C2A_REUSE = 2
MODE_C1A_FULL = 3
MODE_C1A_REUSE = 4
MODE_C1A_REINDEX = 5

D = M.hidden_size
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
INDEX_TOPK = M.index_topk
IDX_DIM = M.index_head_dim
IDX_HEADS = M.index_n_heads
TOPK = M.num_experts_per_tok
INTER = M.moe_intermediate_size
POOL_BLOCKS = CANDIDATE_POOL_BLOCKS
# The pool cut is gated with the same element tolerance as the scores it
# derives from (``indexer.index_score_compare``): near-tie maxima may swap
# order within that accepted arithmetic floor.
POOL_CUT_ATOL = 5e-2
POOL_CUT_RTOL = 2e-2

# Exact suite parameter order: the phase host entries declare their parameters
# in this order and the harness rejects any mismatch with the spec list.
# Padded staging rows for the host-driven EP16 phase.  The bounded layer
# suite carries at most 16 rows per rank; a wider token range compiles the
# tile at its own bound.
MOE_TILE = 16

LAYER_SPEC_ORDER = (
    "x_hc", "pre_mix",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale",
    "wkv", "wkv_scale", "kv_norm_w", "sink", "wo_a", "wo_b",
    "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "compressed_cache",
    "request_ids", "compressed_lens", "index_cache", "index_block_table",
    "compressed_cos", "compressed_sin", "compressed_slots",
    "position_ids", "state_block_table", "state_cache",
    "compressor_wkv", "compressor_wgate", "compressor_norm",
    "index_wq_b", "index_wq_b_scale", "index_weights_proj", "index_wk", "index_norm_w",
    "topk_indices", "topk_values", "pool_block_ids", "pool_block_counts",
    "engram_lookup", "engram_wkv", "engram_qk", "engram_rotation",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm_w",
    "gate_weight", "gate_bias",
    "w1", "w1_scale", "w3", "w3_scale", "w2", "w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
    "hidden", "attn_input", "attn_out", "x_hc_mid", "ffn_input", "ffn_owned",
    "ffn_normed", "route_indices", "route_weights", "x_hc_out", "next_pre_mix",
    "ffn_post_mix", "ffn_residual_mix",
    "moe_x", "moe_indices", "moe_weights", "routed_pad",
)

# Modes that own the compressor and publish compressed rows for this layer.
COMPRESS_MODES = (MODE_C2A_FULL, MODE_C1A_FULL)
# Modes that score the index cache in this layer.
SCORE_MODES = (MODE_C2A_FULL, MODE_C1A_FULL, MODE_C1A_REINDEX)
# Modes that publish index keys (sources only).
INDEX_PUBLISH_MODES = (MODE_C2A_FULL, MODE_C1A_FULL)

# InOut suite buffers: the harness stamps them as outputs, so the comparator
# side captures their pristine fixture values while the specs are built.
_FIXTURE_NAMES = (
    "window_cache", "compressed_cache", "index_cache", "state_cache",
    "topk_indices", "topk_values", "pool_block_ids", "pool_block_counts",
)

_FIXTURE = {}
_STAGED_CACHE = {}


def capture_fixture(values) -> None:
    """Snapshot the InOut fixture values while the specs are built."""
    _FIXTURE.clear()
    _STAGED_CACHE.clear()
    _FIXTURE.update(values)


def _fixture(name, inputs):
    value = _FIXTURE.get(name)
    return inputs[name] if value is None else value


_MOE_WEIGHT_RANK_NDIM = {
    "w1": 3, "w3": 3, "w2": 3,
    "w1_scale": 2, "w3_scale": 2, "w2_scale": 2,
    "shared_w1": 2, "shared_w3": 2, "shared_w2": 2,
    "shared_w1_scale": 1, "shared_w3_scale": 1, "shared_w2_scale": 1,
}


def _moe_weight_shard(values):
    """Preserve destination-rank expert ownership in the shared Torch golden."""
    return {name: values[name] for name in _MOE_WEIGHT_RANK_NDIM}


def _rms_reference(value, gamma):
    """FP32 RMSNorm reference with the model epsilon; BF16 output."""
    value = value.float()
    inverse = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
    return (value * inverse * gamma.float()).to(torch.bfloat16)


def _hc_post_reference(sublayer, residual, post, residual_mix):
    update = post.float().unsqueeze(-1) * sublayer.float().unsqueeze(-2)
    skip = (residual_mix.float().unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(dim=-3)
    return (update + skip).to(torch.bfloat16).float()


def _precision_report(name, actual, expected):
    """Print the end-to-end rel-L2; the staged checks are the actual gates."""
    diff = actual.double() - expected.double()
    relative = (diff.norm() / expected.double().norm().clamp_min(1e-12)).item()
    print(
        f"[PRECISION] {name} end-to-end rel_l2={relative:.4%} "
        f"max_abs={diff.abs().max().item():.4g}",
        flush=True,
    )
    return relative


# ---------------------------------------------------------------------------
# Per-operator golden steps shared by the composed golden and the staged refs
# ---------------------------------------------------------------------------


def _mla_rows(x, tensors, rank, tokens):
    """Golden MLA projection for one rank; returns its output dict."""
    from mla_proj import _golden_mla_proj

    mla = {
        "x": x,
        "wq_a": tensors["wq_a"][rank], "wq_a_scale": tensors["wq_a_scale"][rank],
        "q_norm_w": tensors["q_norm_w"][rank],
        "wq_b": tensors["wq_b"][rank], "wq_b_scale": tensors["wq_b_scale"][rank],
        "wkv": tensors["wkv"][rank], "wkv_scale": tensors["wkv_scale"][rank],
        "kv_norm_w": tensors["kv_norm_w"][rank],
        "rope_cos": tensors["rope_cos"][rank], "rope_sin": tensors["rope_sin"][rank],
        "q": torch.zeros(tokens, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        "kv": torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16),
        "qr": torch.zeros(tokens, Q_LORA, dtype=torch.int8),
        "qr_scale": torch.zeros(tokens, 1, dtype=torch.float32),
    }
    _golden_mla_proj(mla)
    return mla


def _compressor_rows(mode, x, tensors, rank, positions, requests, state_input):
    """Golden compressor for one rank; returns ``(latent, updated_state)``."""
    from compressor import golden_c1a_compressor, golden_c2a_compressor

    if mode == MODE_C2A_FULL:
        comp = {
            "x": x, "wkv": tensors["compressor_wkv"][rank],
            "wgate": tensors["compressor_wgate"][rank],
            "norm_w": tensors["compressor_norm"][rank],
            "position_ids": positions, "request_ids": requests,
            "state_block_table": tensors["state_block_table"][rank],
            "state_cache": state_input.clone(),
            "output": torch.zeros(x.shape[0], HEAD_DIM, dtype=torch.bfloat16),
        }
        golden_c2a_compressor(comp)
        return comp["output"], comp["state_cache"]
    comp = {
        "x": x, "wkv": tensors["compressor_wkv"][rank],
        "norm_w": tensors["compressor_norm"][rank],
        "output": torch.zeros(x.shape[0], HEAD_DIM, dtype=torch.bfloat16),
    }
    golden_c1a_compressor(comp)
    return comp["output"], None


def _score_rows(mode, x, mla, tensors, rank, latent, index_cache):
    """Golden paged index scoring on ``index_cache``; returns ``(scores, cache)``."""
    from indexer import golden_index_scores

    positions = tensors["position_ids"][rank]
    requests = tensors["request_ids"][rank]
    slots = tensors["compressed_slots"][rank]
    if mode == MODE_C1A_REINDEX:
        # The reindex layers rescore the source basis and publish no keys: a
        # negative slot keeps the reference publish a no-op.
        latent = torch.zeros_like(latent)
        slots = torch.full_like(slots, -1)
    index = {
        "x": x, "qr": mla["qr"], "qr_scale": mla["qr_scale"],
        "wq_b": tensors["index_wq_b"][rank], "wq_b_scale": tensors["index_wq_b_scale"][rank],
        "weights_proj": tensors["index_weights_proj"][rank],
        "wk": tensors["index_wk"][rank], "k_norm": tensors["index_norm_w"][rank],
        "latent": latent, "compressed_slots": slots,
        "position_ids": positions, "request_ids": requests,
        "compressed_lens": tensors["compressed_lens"][rank],
        "index_block_table": tensors["index_block_table"][rank],
        "cos": tensors["rope_cos"][rank], "sin": tensors["rope_sin"][rank],
        "compressed_cos": tensors["compressed_cos"][rank],
        "compressed_sin": tensors["compressed_sin"][rank],
        "index_cache": index_cache.clone(),
        "scores": torch.zeros(
            x.shape[0], tensors["index_block_table"].shape[-1] * BLOCK_SIZE, dtype=torch.float32,
        ),
    }
    scores = golden_index_scores(index)
    return scores, index["index_cache"]


def _selection_rows(tensors, rank, scores, mode):
    """Golden Top-K physical rows and candidate block pool for one rank."""
    from indexer import golden_index_select, golden_reindex_select

    selection = {
        "compressed_lens": tensors["compressed_lens"][rank],
        "request_ids": tensors["request_ids"][rank],
        "index_block_table": tensors["index_block_table"][rank],
        "indices": torch.full((scores.shape[0], INDEX_TOPK), -1, dtype=torch.int32),
        "values": torch.zeros(scores.shape[0], INDEX_TOPK, dtype=torch.float32),
        "pool_ids": torch.full((scores.shape[0], POOL_BLOCKS), -1, dtype=torch.int32),
        "pool_counts": torch.zeros(scores.shape[0], dtype=torch.int32),
    }
    if mode == MODE_C1A_REINDEX:
        selection["pool_ids"] = tensors["pool_block_ids"][rank].clone()
        selection["pool_counts"] = tensors["pool_block_counts"][rank].clone()
        golden_reindex_select(selection, scores)
    else:
        golden_index_select(selection, scores, POOL_BLOCKS)
    return selection


def _gather_rows(mode, window_cache, compressed_cache, window_indices, topk_indices):
    """Golden packed key gather from the paged caches for one rank."""
    from paged_cache import golden_gather_selected_keys, golden_gather_window_keys

    tokens = window_indices.shape[0]
    if mode == MODE_SWA:
        gather = {
            "window_cache": window_cache, "window_indices": window_indices,
            "selected": torch.zeros(tokens, WINDOW_KEYS, HEAD_DIM, dtype=torch.bfloat16),
            "combined": torch.zeros(tokens, WINDOW_KEYS, dtype=torch.int32),
        }
        golden_gather_window_keys(gather)
        return gather["selected"], gather["combined"]
    gather = {
        "window_cache": window_cache, "window_indices": window_indices,
        "compressed_cache": compressed_cache, "topk_indices": topk_indices,
        "selected": torch.zeros(tokens, WINDOW_KEYS + INDEX_TOPK, HEAD_DIM, dtype=torch.bfloat16),
        "combined": torch.zeros(tokens, WINDOW_KEYS + INDEX_TOPK, dtype=torch.int32),
    }
    golden_gather_selected_keys(gather)
    return gather["selected"], gather["combined"]


def _partial_rows(q, selected, combined, sink, wo_a, wo_b, cos, sin):
    """Golden MLA core plus output projection for one TP rank (FP32 rows)."""
    from attention import golden_attention_core
    from o_proj import golden_o_proj

    tokens = q.shape[0]
    attention = {
        "q": q, "selected": selected, "combined": combined, "sink": sink,
        "output": torch.zeros(tokens, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16),
    }
    golden_attention_core(attention)
    out = {
        "attended": attention["output"].reshape(tokens, LOCAL_HEADS * HEAD_DIM).float(),
        "wo_a": wo_a, "wo_b": wo_b,
        "output": torch.zeros(tokens, D, dtype=torch.bfloat16),
    }
    out["cos"] = cos
    out["sin"] = sin
    golden_o_proj(out)
    return out["output"].float()


def _tp_sum(partials, ranks):
    """Sum the output-projection partials of each TP group (BF16 result)."""
    summed = torch.zeros_like(partials)
    for base in range(0, ranks, TP_SIZE):
        group = partials[base:base + TP_SIZE].sum(dim=0).to(torch.bfloat16)
        for rank in range(base, base + TP_SIZE):
            summed[rank] = group
    return summed


# ---------------------------------------------------------------------------
# Composed layer golden
# ---------------------------------------------------------------------------


def reference_active_counts(tensors):
    """Validate independently supplied per-rank active prefixes."""
    ranks, tokens = tensors["x_hc"].shape[:2]
    counts = tensors.get("token_counts")
    if counts is None:
        return torch.full((ranks,), tokens, dtype=torch.int64)
    if not isinstance(counts, torch.Tensor) or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("active counts must be an integer tensor")
    if list(counts.shape) != [ranks] or bool(((counts < 0) | (counts > tokens)).any()):
        raise ValueError("active counts must fit the physical token capacity")
    for base in range(0, ranks, TP_SIZE):
        group = counts[base:base + TP_SIZE]
        if bool((group != group[0]).any()):
            raise ValueError("TP peers must share the active query prefix")
    return counts.to(dtype=torch.int64, device="cpu")


def active_output_rows(value, tensors, token_axis):
    """Select valid query outputs; pure-output inactive tails are undefined."""
    counts = reference_active_counts(tensors)
    if value.shape[0] != counts.numel() or value.shape[token_axis] != tensors["x_hc"].shape[1]:
        raise ValueError("query output does not match the active-count geometry")
    if bool((counts == value.shape[token_axis]).all()):
        return value
    moved = value.movedim(token_axis, 1)
    mask = torch.arange(moved.shape[1]).unsqueeze(0) < counts.unsqueeze(1)
    return moved[mask]


def _active_reference_inputs(tensors, counts):
    """Per-rank token views for the existing projection/cache reference owners."""
    if bool((counts == tensors["x_hc"].shape[1]).all()):
        return tensors
    result = dict(tensors)
    row_names = (
        "x_hc", "pre_mix", "rope_cos", "rope_sin", "compressed_cos", "compressed_sin",
        "position_ids", "request_ids", "window_slots", "window_indices", "compressed_slots", "compressed_lens",
        "topk_indices", "topk_values", "pool_block_ids", "pool_block_counts", "engram_lookup",
    )
    for name in row_names:
        if name in tensors:
            result[name] = [tensors[name][rank, :int(counts[rank])] for rank in range(counts.numel())]
    return result


def make_layer_golden(mode, has_engram, ratio):
    """Return the composed Torch reference for one bounded layer case."""

    def golden(tensors):
        from engram import golden_engram
        from hc import golden_hc_collapse, golden_mhc_mixes
        from moe_exchange import golden_moe_ep
        from moe_gate import golden_moe_gate
        from utils import scatter_cache_rows

        ranks = tensors["x_hc"].shape[0]
        tokens = tensors["x_hc"].shape[1]
        counts = reference_active_counts(tensors)
        rows = _active_reference_inputs(tensors, counts)
        compress = mode in COMPRESS_MODES
        current_scores = mode in SCORE_MODES

        window_cache = tensors["window_cache"].clone()
        compressed_cache = tensors["compressed_cache"].clone()
        index_cache = tensors["index_cache"].clone()
        state_cache = tensors["state_cache"].clone()
        topk_all = tensors["topk_indices"].clone()
        topk_values_all = tensors["topk_values"].clone()
        attention_sources = torch.zeros_like(tensors["x_hc"])
        hidden_all = torch.zeros(ranks, tokens, D, dtype=torch.bfloat16)
        attn_input_all = torch.zeros(ranks, tokens, D, dtype=torch.bfloat16)
        partial_all = torch.zeros(ranks, tokens, D)
        pre_attn_all = torch.zeros(ranks, tokens, HC_MULT)
        post_attn_all = torch.zeros(ranks, tokens, HC_MULT)
        res_attn_all = torch.zeros(ranks, tokens, HC_MULT, HC_MULT)

        for rank in range(ranks):
            active = int(counts[rank])
            if active == 0:
                continue
            x_hc = tensors["x_hc"][rank, :active]
            if has_engram:
                engram = {"lookup": rows["engram_lookup"][rank], "wkv_weight": tensors["engram_wkv"][0], "weight": tensors["engram_qk"][0], "rotation": tensors["engram_rotation"][rank], "x": x_hc, "output": torch.zeros_like(x_hc)}
                golden_engram(engram)
                x_hc = engram["output"]
            attention_sources[rank, :active] = x_hc
            pre, post, residual = golden_mhc_mixes(
                x_hc, tensors["hc_attn_fn"][rank], tensors["hc_attn_scale"][rank],
                tensors["hc_attn_base"][rank],
            )
            pre_attn_all[rank, :active] = pre
            post_attn_all[rank, :active] = post
            res_attn_all[rank, :active] = residual
            hidden = golden_hc_collapse(x_hc, rows["pre_mix"][rank])
            hidden_all[rank, :active] = hidden
            attn_input = _rms_reference(hidden, tensors["attn_norm_w"][rank])
            attn_input_all[rank, :active] = attn_input

            mla = _mla_rows(attn_input, rows, rank, active)
            scatter_cache_rows(
                window_cache[rank], rows["window_slots"][rank],
                mla["kv"].reshape(-1, HEAD_DIM),
            )
            latent = torch.zeros(active, HEAD_DIM, dtype=torch.bfloat16)
            if compress:
                latent, updated_state = _compressor_rows(
                    mode, attn_input, rows, rank, rows["position_ids"][rank],
                    rows["request_ids"][rank], state_cache[rank],
                )
                if updated_state is not None:
                    state_cache[rank] = updated_state
                from rope import golden_rope_tail_bf16

                attention_latent = golden_rope_tail_bf16(latent, rows["compressed_cos"][rank], rows["compressed_sin"][rank])
                scatter_cache_rows(
                    compressed_cache[rank], rows["compressed_slots"][rank], attention_latent,
                )
            if current_scores:
                scores, index_cache[rank] = _score_rows(
                    mode, attn_input, mla, rows, rank, latent, index_cache[rank],
                )
                selection = _selection_rows(rows, rank, scores, mode)
                topk_all[rank, :active] = selection["indices"]
                topk_values_all[rank, :active] = selection["values"]
                tensors["pool_block_ids"][rank, :active] = selection["pool_ids"]
                tensors["pool_block_counts"][rank, :active] = selection["pool_counts"]

            selected, combined = _gather_rows(
                mode, window_cache[rank], compressed_cache[rank],
                rows["window_indices"][rank], topk_all[rank, :active],
            )
            partial_all[rank, :active] = _partial_rows(
                mla["q"], selected, combined, tensors["sink"][rank],
                tensors["wo_a"][rank], tensors["wo_b"][rank],
                rows["rope_cos"][rank], rows["rope_sin"][rank],
            )

        attn_output = _tp_sum(partial_all, ranks)
        x_hc_mid = torch.zeros(ranks, tokens, HC_MULT, D)
        pre_ffn_all = torch.zeros(ranks, tokens, HC_MULT)
        post_ffn_all = torch.zeros(ranks, tokens, HC_MULT)
        res_ffn_all = torch.zeros(ranks, tokens, HC_MULT, HC_MULT)
        ffn_input = torch.zeros(ranks, tokens, D, dtype=torch.bfloat16)
        ffn_normed = torch.zeros(ranks, tokens, D, dtype=torch.bfloat16)
        for rank in range(ranks):
            active = int(counts[rank])
            if active == 0:
                continue
            mid = _hc_post_reference(
                attn_output[rank, :active], attention_sources[rank, :active],
                post_attn_all[rank, :active], res_attn_all[rank, :active],
            )
            x_hc_mid[rank, :active] = mid
            pre, post, residual = golden_mhc_mixes(
                mid, tensors["hc_ffn_fn"][rank], tensors["hc_ffn_scale"][rank],
                tensors["hc_ffn_base"][rank],
            )
            pre_ffn_all[rank, :active] = pre
            post_ffn_all[rank, :active] = post
            res_ffn_all[rank, :active] = residual
            collapsed = golden_hc_collapse(mid, pre_attn_all[rank, :active])
            ffn_input[rank, :active] = collapsed
            ffn_normed[rank, :active] = _rms_reference(collapsed, tensors["ffn_norm_w"][rank])

        indices_all = torch.zeros(ranks, tokens, TOPK, dtype=torch.int32)
        weights_all = torch.zeros(ranks, tokens, TOPK, dtype=torch.float32)
        for rank in range(ranks):
            active = int(counts[rank])
            if active == 0:
                continue
            golden_moe_gate({
                "x_normed": ffn_normed[rank, :active], "gate_weight": tensors["gate_weight"][rank],
                "gate_bias": tensors["gate_bias"][rank], "indices": indices_all[rank, :active],
                "weights": weights_all[rank, :active],
            })
        moe = {
            "x": ffn_normed, "indices": indices_all, "weights": weights_all,
            "routed": torch.zeros(ranks, tokens, D, dtype=torch.bfloat16),
            "active_counts": counts,
        }
        moe.update(_moe_weight_shard(tensors))
        golden_moe_ep(moe)
        ffn_owned = moe["routed"]
        x_hc_out = torch.zeros(ranks, tokens, HC_MULT, D)
        for rank in range(ranks):
            active = int(counts[rank])
            if active == 0:
                continue
            x_hc_out[rank, :active] = _hc_post_reference(
                ffn_owned[rank, :active], x_hc_mid[rank, :active], post_ffn_all[rank, :active], res_ffn_all[rank, :active],
            )

        tensors["hidden"][:] = hidden_all
        tensors["attn_input"][:] = attn_input_all
        tensors["attn_out"][:] = attn_output
        tensors["x_hc_mid"][:] = x_hc_mid
        tensors["ffn_input"][:] = ffn_input
        tensors["ffn_owned"][:] = ffn_owned
        tensors["ffn_normed"][:] = ffn_normed
        tensors["route_indices"][:] = indices_all
        tensors["route_weights"][:] = weights_all
        tensors["x_hc_out"][:] = x_hc_out
        tensors["next_pre_mix"][:] = pre_ffn_all
        tensors["window_cache"][:] = window_cache
        tensors["compressed_cache"][:] = compressed_cache
        tensors["index_cache"][:] = index_cache
        tensors["state_cache"][:] = state_cache
        tensors["topk_indices"][:] = topk_all
        tensors["topk_values"][:] = topk_values_all

    return golden


# ---------------------------------------------------------------------------
# Staged references
# ---------------------------------------------------------------------------
#
# Each reference consumes the device value of the preceding stage, so the
# residual under its gate is the operator's own device arithmetic floor:
#
#   hidden          <- golden hc_pre collapse on the device x_hc
#   attn_input      <- golden hc collapse + RMSNorm on the device x_hc
#   window_cache    <- golden mla_proj on the device attn_input, published
#   compressed_cache<- golden compressor on the device attn_input, published
#   index_cache     <- golden index keys published from the device compressor
#   topk/pool       <- golden selection on the staged scores
#   attn_out        <- golden gather + MLA core + o_proj + TP sum on the
#                      staged caches and selection
#   x_hc_mid        <- golden hc_post (+ Engram) on the device attn_out
#   ffn_input       <- golden FFN collapse on the device x_hc_mid
#   next_pre_mix    <- golden FFN hc_pre coefficients on the device x_hc_mid
#   state_cache     <- C2A: golden compressor update on the device attn_input;
#                      other modes: the untouched input contract


def _build_staged_references(actual_outputs, inputs):
    """Re-run each layer-side operator on the device's own intermediates."""
    from utils import scatter_cache_rows

    mode = int(inputs["mode"])
    has_engram = int(inputs["has_engram"]) == 1
    ranks = actual_outputs["attn_input"].shape[0]
    tokens = actual_outputs["attn_input"].shape[1]
    if bool((reference_active_counts(inputs) != tokens).any()):
        raise ValueError("active-S standalone staged layer reference is not implemented in this decode draft")
    compress = mode in COMPRESS_MODES
    current_scores = mode in SCORE_MODES

    ref = {}
    for name in (
        "hidden", "attn_input", "attn_out", "x_hc_mid", "ffn_input", "next_pre_mix",
        "window_cache", "compressed_cache", "index_cache", "topk_indices", "topk_values",
        "pool_block_ids", "pool_block_counts", "state_cache",
    ):
        ref[name] = torch.zeros_like(actual_outputs[name])

    pre_mix_in = _fixture("pre_mix", inputs)
    window_pristine = _fixture("window_cache", inputs)
    compressed_pristine = _fixture("compressed_cache", inputs)
    index_pristine = _fixture("index_cache", inputs)
    state_pristine = _fixture("state_cache", inputs)
    topk_pristine = _fixture("topk_indices", inputs)
    values_pristine = _fixture("topk_values", inputs)
    pool_ids_pristine = _fixture("pool_block_ids", inputs)
    pool_counts_pristine = _fixture("pool_block_counts", inputs)

    from hc import golden_hc_collapse, golden_mhc_mixes

    partial = torch.zeros(ranks, tokens, D)
    pre_attn = []
    post_attn = []
    res_attn = []
    attention_sources = []
    from engram import golden_engram
    for rank in range(ranks):
        x_hc = inputs["x_hc"][rank]
        if has_engram:
            engram = {"lookup": inputs["engram_lookup"][rank], "wkv_weight": inputs["engram_wkv"][0], "weight": inputs["engram_qk"][0], "rotation": inputs["engram_rotation"][rank], "x": x_hc, "output": torch.zeros_like(x_hc)}
            golden_engram(engram)
            x_hc = engram["output"]
        attention_sources.append(x_hc)
        pre, post, residual = golden_mhc_mixes(
            x_hc, inputs["hc_attn_fn"][rank], inputs["hc_attn_scale"][rank],
            inputs["hc_attn_base"][rank],
        )
        pre_attn.append(pre)
        post_attn.append(post)
        res_attn.append(residual)
        hidden = golden_hc_collapse(x_hc, pre_mix_in[rank])
        ref["hidden"][rank] = hidden
        ref["attn_input"][rank] = _rms_reference(hidden, inputs["attn_norm_w"][rank])

        device_input = actual_outputs["attn_input"][rank]
        mla = _mla_rows(device_input, inputs, rank, tokens)
        ref["window_cache"][rank] = scatter_cache_rows(
            window_pristine[rank].clone(), inputs["window_slots"][rank],
            mla["kv"].reshape(-1, HEAD_DIM),
        )

        latent = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
        if compress:
            latent, updated_state = _compressor_rows(
                mode, device_input, inputs, rank, inputs["position_ids"][rank],
                inputs["request_ids"][rank], state_pristine[rank],
            )
            if updated_state is not None:
                ref["state_cache"][rank] = updated_state
            else:
                ref["state_cache"][rank] = state_pristine[rank]
            from rope import golden_rope_tail_bf16

            attention_latent = golden_rope_tail_bf16(latent, inputs["compressed_cos"][rank], inputs["compressed_sin"][rank])
            ref["compressed_cache"][rank] = scatter_cache_rows(
                compressed_pristine[rank].clone(), inputs["compressed_slots"][rank], attention_latent,
            )
        else:
            ref["compressed_cache"][rank] = compressed_pristine[rank]
            ref["state_cache"][rank] = state_pristine[rank]

        if current_scores:
            scores, staged_index = _score_rows(
                mode, device_input, mla, inputs, rank, latent, index_pristine[rank],
            )
            if ref.get("index_scores") is None:
                ref["index_scores"] = torch.zeros(ranks, tokens, scores.shape[1])
            ref["index_scores"][rank] = scores
            ref["index_cache"][rank] = staged_index
            selection_inputs = dict(inputs)
            selection_inputs["pool_block_ids"] = pool_ids_pristine
            selection_inputs["pool_block_counts"] = pool_counts_pristine
            selection = _selection_rows(selection_inputs, rank, scores, mode)
            ref["topk_indices"][rank] = selection["indices"]
            ref["topk_values"][rank] = selection["values"]
            ref["pool_block_ids"][rank] = selection["pool_ids"]
            ref["pool_block_counts"][rank] = selection["pool_counts"]
        else:
            ref["index_cache"][rank] = index_pristine[rank]
            ref["topk_indices"][rank] = topk_pristine[rank]
            ref["topk_values"][rank] = values_pristine[rank]
            ref["pool_block_ids"][rank] = pool_ids_pristine[rank]
            ref["pool_block_counts"][rank] = pool_counts_pristine[rank]

        selected, combined = _gather_rows(
            mode, ref["window_cache"][rank], ref["compressed_cache"][rank],
            inputs["window_indices"][rank], ref["topk_indices"][rank],
        )
        partial[rank] = _partial_rows(
            mla["q"], selected, combined, inputs["sink"][rank],
            inputs["wo_a"][rank], inputs["wo_b"][rank],
            inputs["rope_cos"][rank], inputs["rope_sin"][rank],
        )

    ref["attn_out"] = _tp_sum(partial, ranks)

    for rank in range(ranks):
        mid = _hc_post_reference(
            actual_outputs["attn_out"][rank], attention_sources[rank],
            post_attn[rank], res_attn[rank],
        )
        ref["x_hc_mid"][rank] = mid
        ref["ffn_input"][rank] = golden_hc_collapse(mid, pre_attn[rank])
        pre, _, _ = golden_mhc_mixes(
            mid, inputs["hc_ffn_fn"][rank], inputs["hc_ffn_scale"][rank],
            inputs["hc_ffn_base"][rank],
        )
        ref["next_pre_mix"][rank] = pre
    return ref


def _staged_references(actual_outputs, inputs):
    # Cache per run: every comparator callback receives the same output dict,
    # and the fixture capture clears the cache before the next case.
    cached = _STAGED_CACHE.get("value")
    if cached is not None and cached[0] is actual_outputs:
        return cached[1]
    values = _build_staged_references(actual_outputs, inputs)
    _STAGED_CACHE["value"] = (actual_outputs, values)
    return values


def _report_staged(name, actual, staged):
    diff = actual.double() - staged.double()
    relative = (diff.norm() / staged.double().norm().clamp_min(1e-12)).item()
    print(
        f"[PRECISION] {name} staged rel_l2={relative:.4%} "
        f"max_abs={diff.abs().max().item():.4g}",
        flush=True,
    )
    return relative


def _staged_ratio_compare(name, *, atol, rtol, max_error_ratio, max_rel_l2=None):
    """Gate ``name`` against the staged reference with a calibrated band.

    ``max_rel_l2`` is the device-floor bound measured on the sensitivity run
    recorded next to the comparator table; an elementwise outlier allowance
    alone cannot hide a uniformly wrong result.
    """
    from golden import ratio_allclose

    gate = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=max_error_ratio)

    def compare(actual, expected, *, actual_outputs, inputs, rtol=None, atol=None, **_kwargs):
        staged = _staged_references(actual_outputs, inputs)[name]
        ok, detail = gate(
            actual, staged, actual_outputs=actual_outputs,
            expected_outputs=staged, inputs=inputs, rtol=rtol, atol=atol,
        )
        relative = _report_staged(name, actual, staged)
        if max_rel_l2 is not None and relative > max_rel_l2:
            ok = False
            detail += f"\n    staged rel_l2={relative:.4%} exceeds {max_rel_l2:.2%}"
        return ok, detail

    compare.__name__ = f"staged_{name}"
    return compare


def _staged_cache_compare(name, published_modes):
    """Gate a published paged cache on the child band; untouched caches exactly.

    A published cache is a device-computed projection whose rows may differ from
    the reference by one BF16 ULP at larger token counts, so it keeps the
    accepted child band.  Modes that do not publish into a cache must leave it
    untouched, so there the contract is exact equality.
    """
    from golden import ratio_allclose

    published_gate = ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2)
    preserved_gate = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, *, actual_outputs, inputs, rtol=None, atol=None, **_kwargs):
        mode = int(inputs["mode"])
        staged = _staged_references(actual_outputs, inputs)[name]
        gate = published_gate if mode in published_modes else preserved_gate
        ok, detail = gate(
            actual, staged, actual_outputs=actual_outputs,
            expected_outputs=staged, inputs=inputs, rtol=rtol, atol=atol,
        )
        _report_staged(name, actual, staged)
        if ok and name == "window_cache" and int(inputs["has_engram"]) == 1:
            for rank in range(actual.shape[0]):
                slots = inputs["window_slots"][rank].long()
                slots = slots[slots >= 0].unique()
                actual_rows = actual[rank].reshape(-1, HEAD_DIM)[slots]
                expected_rows = expected[rank].reshape(-1, HEAD_DIM)[slots]
                row_ok, row_detail = published_gate(actual_rows, expected_rows, actual_outputs=actual_outputs, expected_outputs={}, inputs=inputs, rtol=rtol, atol=atol)
                if not row_ok:
                    return False, detail + f"\n    Engram input/order cache boundary rank {rank}: " + row_detail
        return ok, detail

    compare.__name__ = f"staged_{name}"
    return compare


def _staged_index_score_compare(name):
    """Gate index scores/values with the indexer's own composed-chain comparator.

    ``index_score_compare`` is the project calibration for a device-produced
    query/latent (1 % rel-L2 bound plus its elementwise band); reusing it keeps
    the layer and the phase entries on the same accepted arithmetic floor.
    """

    def compare(actual, expected, *, actual_outputs, inputs, rtol=None, atol=None, **_kwargs):
        from indexer import index_score_compare

        staged = _staged_references(actual_outputs, inputs)[name]
        ok, detail = index_score_compare(
            actual, staged, actual_outputs=actual_outputs,
            expected_outputs=staged, inputs=inputs, rtol=rtol, atol=atol,
        )
        _report_staged(name, actual, staged)
        return ok, detail

    compare.__name__ = f"staged_{name}"
    return compare


def _staged_topk_indices_compare():
    """Tie-tolerant Top-K indices against the selection on the staged scores."""
    from indexer import index_topk_compare

    gate = index_topk_compare("topk_values")

    def compare(actual, expected, *, actual_outputs, inputs, rtol=None, atol=None, **_kwargs):
        staged = _staged_references(actual_outputs, inputs)["topk_indices"]
        return gate(
            actual, staged, actual_outputs=actual_outputs,
            expected_outputs=staged, inputs=inputs, rtol=rtol, atol=atol,
        )

    compare.__name__ = "staged_topk_indices"
    return compare


def _staged_pool_compare():
    """Candidate pool with set semantics against the staged selection.

    The staged selection runs the shared indexer golden on the staged scores, so
    the accepted score arithmetic floor can reorder near-tie blocks at the
    diagnostic cut.  A one-for-one membership swap is admitted only when both
    blocks' maxima lie inside the accepted index-score relative band
    (``POOL_CUT_ATOL`` + ``POOL_CUT_RTOL``, the accepted index-score element
    tolerance); equal sizes and a banded pair keep a pool wiring fault failing
    while an arithmetic-floor tie flip passes.
    """
    from indexer import pool_set_compare

    def _row_maxima(scores, inputs, rank, token):
        row_scores = scores[rank, token]
        candidate_block = M.candidate_block_size
        visible = min(int(inputs["compressed_lens"][rank, token]), row_scores.numel())
        blocks = (visible + candidate_block - 1) // candidate_block
        return row_scores[: blocks * candidate_block].view(blocks, candidate_block).max(dim=-1).values

    def compare(actual, expected, *, actual_outputs, inputs, rtol=None, atol=None, **_kwargs):
        staged = _staged_references(actual_outputs, inputs)["pool_block_ids"]
        if int(inputs["mode"]) == MODE_C1A_REINDEX:
            from golden import ratio_allclose

            pristine = _fixture("pool_block_ids", inputs)
            gate = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
            return gate(
                actual, pristine, actual_outputs=actual_outputs, expected_outputs=staged,
                inputs=inputs, rtol=rtol, atol=atol,
            )
        flat_actual = actual.reshape(-1, actual.shape[-1])
        flat_staged = staged.reshape(-1, staged.shape[-1])
        exact, detail = pool_set_compare()(
            flat_actual, flat_staged,
            actual_outputs=actual_outputs,
            expected_outputs={**actual_outputs, "pool_ids": staged},
            inputs=inputs, rtol=rtol, atol=atol,
        )
        if exact:
            return True, detail
        scores = _staged_references(actual_outputs, inputs).get("index_scores")
        if scores is None:
            return False, detail
        tokens = actual.shape[1]
        swaps = 0
        for row in range(flat_actual.shape[0]):
            a = sorted(int(v) for v in flat_actual[row] if v >= 0)
            e = sorted(int(v) for v in flat_staged[row] if v >= 0)
            if a == e:
                continue
            if len(a) != len(e):
                return False, detail + (
                    f"\n    pool row {row}: sizes=({len(a)},{len(e)}) differ"
                )
            only_actual = sorted(set(a) - set(e))
            only_expected = sorted(set(e) - set(a))
            if len(only_actual) != len(only_expected):
                return False, detail + f"\n    pool row {row}: unbalanced swap"
            rank, token = divmod(row, tokens)
            maxima = _row_maxima(scores, inputs, rank, token)
            for left, right in zip(only_actual, only_expected):
                if left >= maxima.numel() or right >= maxima.numel():
                    return False, detail + f"\n    pool row {row}: block id out of range"
                left_value = float(maxima[left])
                right_value = float(maxima[right])
                gap = abs(left_value - right_value)
                tolerance = POOL_CUT_ATOL + POOL_CUT_RTOL * max(
                    abs(left_value), abs(right_value)
                )
                if gap > tolerance:
                    return False, detail + (
                        f"\n    pool row {row}: cut maxima "
                        f"{left_value:.6g} vs {right_value:.6g} "
                        f"gap {gap:.4g} exceeds tolerance {tolerance:.4g}"
                    )
                swaps += 1
        detail += f"\n    admitted {swaps} cut swap(s) inside the accepted score tolerance"
        return True, detail

    compare.__name__ = "staged_pool_compare"
    return compare


def _staged_state_compare():
    """C2A compares the updated compressor state; other modes assert its contract.

    Only the C2A compressor carries state; every other mode must leave the
    per-request state blocks untouched.
    """
    from golden import ratio_allclose

    updated_gate = ratio_allclose(atol=1e-5, rtol=1e-5, max_error_ratio=5e-3)
    preserved_gate = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, *, actual_outputs, inputs, rtol=None, atol=None, **_kwargs):
        mode = int(inputs["mode"])
        staged = _staged_references(actual_outputs, inputs)["state_cache"]
        gate = updated_gate if mode == MODE_C2A_FULL else preserved_gate
        ok, detail = gate(
            actual, staged, actual_outputs=actual_outputs,
            expected_outputs=staged, inputs=inputs, rtol=rtol, atol=atol,
        )
        _report_staged("state_cache", actual, staged)
        return ok, detail

    compare.__name__ = "staged_state"
    return compare


def staged_moe_compare(actual, expected, *, actual_outputs, inputs, rtol, atol, **_kwargs):
    """Gate the MoE sublayer against a reference driven by the device's own inputs.

    One INT8 per-token activation code can change on a single BF16 ULP of the
    FFN input, which moves the composed expert output by one to two percent end
    to end.  The gate therefore holds ``ffn_owned`` to the standalone MoE budget
    against a reference re-run on the device's ``ffn_normed`` and selected
    routes; the end-to-end difference is reported as a diagnostic.
    """
    from moe_exchange import golden_moe_ep
    from moe_experts import expert_output_compare

    forced = {
        "x": actual_outputs["ffn_normed"],
        "indices": actual_outputs["route_indices"],
        "weights": actual_outputs["route_weights"],
        "routed": torch.zeros_like(actual),
    }
    forced.update(_moe_weight_shard({name: inputs[name] for name in _MOE_WEIGHT_RANK_NDIM}))
    golden_moe_ep(forced)
    _precision_report("ffn_owned", actual, expected)
    # Device-calibrated expert gate (see moe_experts.expert_output_compare):
    # the second per-token INT8 quantization amplifies one BF16 ULP of
    # activation rounding beyond the plain (2e-2, 2e-2) band.
    return expert_output_compare(
        actual, forced["routed"], actual_outputs=actual_outputs,
        expected_outputs=forced, inputs=inputs, rtol=rtol, atol=atol,
    )


def staged_hc_post_compare(actual, expected, *, actual_outputs, inputs, rtol, atol, **_kwargs):
    """Gate XHC post against a reference driven by the device's own sublayer results."""
    from golden import ratio_allclose
    from hc import golden_mhc_mixes

    mid = actual_outputs["x_hc_mid"]
    sublayer = actual_outputs["ffn_owned"]
    forced = torch.zeros_like(actual)
    for rank in range(mid.shape[0]):
        _, post, residual = golden_mhc_mixes(
            mid[rank], inputs["hc_ffn_fn"][rank], inputs["hc_ffn_scale"][rank],
            inputs["hc_ffn_base"][rank],
        )
        update = post.float().unsqueeze(-1) * sublayer[rank].float().unsqueeze(-2)
        skip = (residual.float().unsqueeze(-1) * mid[rank].float().unsqueeze(-2)).sum(dim=-3)
        forced[rank] = update + skip
    _precision_report("x_hc_out", actual, expected)
    return ratio_allclose(atol=1e-3, rtol=1e-2, max_error_ratio=5e-3)(
        actual, forced, actual_outputs=actual_outputs, expected_outputs=forced,
        inputs=inputs, rtol=rtol, atol=atol,
    )


def staged_rms_compare(actual, expected, *, actual_outputs, inputs, rtol, atol, **_kwargs):
    """Gate the FFN RMSNorm against a reference driven by the device's own input."""
    from golden import ratio_allclose

    forced = torch.zeros_like(actual)
    for rank in range(actual.shape[0]):
        forced[rank] = _rms_reference(
            actual_outputs["ffn_input"][rank], inputs["ffn_norm_w"][rank],
        )
    _precision_report("ffn_normed", actual, expected)
    return ratio_allclose(atol=1e-3, rtol=1e-2, max_error_ratio=5e-3)(
        actual, forced, actual_outputs=actual_outputs, expected_outputs=forced,
        inputs=inputs, rtol=rtol, atol=atol,
    )


def staged_gate_compare(name):
    """Validate each rank's biased Top-K membership and selected unbiased route weights."""
    def compare(actual, expected, *, actual_outputs, inputs, rtol, atol, **_kwargs):
        from moe_gate import compare_moe_indices, compare_moe_weights

        comparator = compare_moe_indices if name == "route_indices" else compare_moe_weights
        for rank in range(actual.shape[0]):
            local_inputs = {
                "x_normed": actual_outputs["ffn_normed"][rank],
                "gate_weight": inputs["gate_weight"][rank],
                "gate_bias": inputs["gate_bias"][rank],
            }
            local_outputs = {
                "indices": actual_outputs["route_indices"][rank],
                "weights": actual_outputs["route_weights"][rank],
            }
            ok, detail = comparator(actual[rank], expected[rank], actual_outputs=local_outputs, expected_outputs={}, inputs=local_inputs, rtol=rtol, atol=atol)
            if not ok:
                return False, f"rank {rank}: {detail}"
        return True, "all ranks satisfy routing selection and weight contracts"

    compare.__name__ = f"staged_gate_compare({name})"
    return compare


def build_compare() -> dict:
    """Comparators for the shared layer-suite outputs.

    Bands are the attention-side device-arithmetic floors measured by the host
    sensitivity run and the device ``[PRECISION]`` statistics recorded in
    TASKS.md; FFN-side staged gates keep their own calibrated helpers.
    """
    return {
        "hidden": _staged_ratio_compare(
            "hidden", atol=1e-2, rtol=1e-2, max_error_ratio=5e-2,
        ),
        "attn_input": _staged_ratio_compare(
            "attn_input", atol=1e-2, rtol=1e-2, max_error_ratio=5e-2,
        ),
        "attn_out": _staged_ratio_compare(
            "attn_out", atol=2e-2, rtol=2e-2, max_error_ratio=2e-2,
        ),
        "x_hc_mid": _staged_ratio_compare(
            "x_hc_mid", atol=1e-3, rtol=1e-2, max_error_ratio=5e-3,
        ),
        "ffn_input": _staged_ratio_compare(
            "ffn_input", atol=1e-2, rtol=1e-2, max_error_ratio=1e-2,
        ),
        "next_pre_mix": _staged_ratio_compare(
            "next_pre_mix", atol=2.5e-5, rtol=5e-3, max_error_ratio=5e-3,
        ),
        "ffn_normed": staged_rms_compare,
        "route_indices": staged_gate_compare("route_indices"),
        "route_weights": staged_gate_compare("route_weights"),
        "ffn_owned": staged_moe_compare,
        "x_hc_out": staged_hc_post_compare,
        "window_cache": _staged_cache_compare("window_cache", (MODE_SWA,) + COMPRESS_MODES + (MODE_C2A_REUSE, MODE_C1A_REUSE, MODE_C1A_REINDEX)),
        "compressed_cache": _staged_cache_compare("compressed_cache", COMPRESS_MODES),
        "index_cache": _staged_cache_compare("index_cache", INDEX_PUBLISH_MODES),
        "topk_indices": _staged_topk_indices_compare(),
        "topk_values": _staged_index_score_compare("topk_values"),
        "pool_block_ids": _staged_pool_compare(),
        "pool_block_counts": _staged_ratio_compare(
            "pool_block_counts", atol=0.0, rtol=0.0, max_error_ratio=0.0,
        ),
        "state_cache": _staged_state_compare(),
    }


# ---------------------------------------------------------------------------
# Bounded layer fixtures
# ---------------------------------------------------------------------------
#
# One paged request plan per case: two requests over the same block table, both
# spanning several logical pages so the paged addressing (not just the first
# page) is exercised.  Every rank of a DP group sees the same plan and the same
# activations; the TP ranks of a group carry their own weight shards.

PREFILL_REQUESTS = ((1240, 8), (1256, 8))
DECODE_REQUESTS = ((1250, 1), (1253, 1))

_EXPERT_CACHE = {}


def _rope_tables(count, start=0, theta=None):
    half = 32
    theta = M.rope_theta if theta is None else theta
    inv_freq = 1.0 / torch.pow(
        torch.tensor(theta, dtype=torch.float32),
        torch.arange(0, half, dtype=torch.float32) * 2.0 / (2 * half),
    )
    angle = torch.arange(start, start + count, dtype=torch.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
    return torch.cos(angle), torch.sin(angle)


def _expert_fixture(seed):
    """One routed/shared expert shard shared by every rank (bounded memory)."""
    from config import N_LOCAL_EXPERTS
    from utils import pack_nz, quant_w_per_channel

    if seed in _EXPERT_CACHE:
        return _EXPERT_CACHE[seed]
    torch.manual_seed(seed)

    def stacked(n, k):
        packed, scales = [], []
        for _ in range(N_LOCAL_EXPERTS):
            logical, scale = quant_w_per_channel(torch.randn(n, k) * 0.05)
            packed.append(pack_nz(logical.transpose(0, 1).contiguous()))
            scales.append(scale)
        return torch.stack(packed), torch.stack(scales)

    def single(n, k):
        logical, scale = quant_w_per_channel(torch.randn(n, k) * 0.05)
        return pack_nz(logical.transpose(0, 1).contiguous()), scale

    values = {
        "w1": stacked(INTER, D), "w3": stacked(INTER, D), "w2": stacked(D, INTER),
        "shared_w1": single(INTER, D), "shared_w3": single(INTER, D), "shared_w2": single(D, INTER),
    }
    _EXPERT_CACHE[seed] = values
    return values


def _selection_input(plan, tokens: int):
    """Physical compressed rows of every visible row, as the source published them."""
    indices = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    table = plan["index_block_table"]
    for token in range(tokens):
        visible = int(plan["compressed_lens"][token])
        request = int(plan["request_ids"][token])
        for logical in range(min(visible, INDEX_TOPK)):
            physical = int(table[request, logical // BLOCK_SIZE])
            indices[token, logical] = physical * BLOCK_SIZE + logical % BLOCK_SIZE
    return indices


def _dispatch_prepared(compiled, specs, tensors, scalar_specs, cfg):
    """Dispatch one stage through a freshly prepared worker.

    ``compiled(...)`` one-shot dispatch cannot repeat a distributed program in
    the same process (the CommDomain window state is not reclaimed), so each
    stage is dispatched through its own prepared worker: per-call buffers are
    made shared before the fork, the worker runs the program once, and the
    written outputs are copied back into the caller-owned tensors.
    """
    from golden.runner import _ordered_args
    from golden.spec import TensorSpec

    originals = {}
    for spec in specs:
        if not isinstance(spec, TensorSpec):
            continue
        tensor = tensors[spec.name]
        if tensor.is_shared() and tensor.is_contiguous():
            continue
        tensors[spec.name] = tensor.detach().cpu().contiguous().share_memory_()
        originals[spec.name] = tensor
    args = _ordered_args(specs, tensors, scalar_specs, ctypes_scalars=False)
    # ``reset_persistent_windows`` restores every retained CommDomain window to
    # zero before the dispatch, so the exchange/TP protocols always start from
    # epoch 1 no matter how many programs already ran in this process.
    with compiled.prepare(persistent=True, reset_persistent_windows=True) as rt:
        rt(*args, config=cfg)
    by_name = {spec.name: spec for spec in specs if isinstance(spec, TensorSpec)}
    for name, original in originals.items():
        if by_name[name].is_output:
            original.copy_(tensors[name])
        tensors[name] = original


def run_staged_programs(
    entries, stages, specs, golden_fn, compare, config,
    extra_inputs=None, compile_only=False, validate=True, stage_epochs=None,
):
    """Compile and run several host programs over one shared tensor set.

    The golden harness runs one compiled program per call.  A layer whose MoE
    exchange cannot share a compilation unit with the dynamic typed chain (see
    KNOWN_PYPTO_ISSUES.md) needs several programs over the same host buffers, so
    this reuses the harness's compile/dispatch/validate plumbing directly: the
    buffers are built once, each stage is compiled and dispatched in order, and
    the final device outputs are gated by the same calibrated comparators.

    ``stages`` is a sequence of ``(entry_key, parameter_names, with_epoch)``;
    ``entries`` maps the key to a ``@pl.jit.host`` function.  ``extra_inputs``
    are added to the comparator's input mapping (e.g. the mode selector).
    ``validate=False`` is the forward-driver mode: the staged programs are
    compiled and dispatched in place over the caller-owned tensors (specs use
    those tensors as ``init_value``) and no golden/comparison runs binding a
    scalar/state value to one compile specialization.

    Scalar specs keep ``compile_runtime`` so a per-layer epoch stays a runtime
    ABI parameter and every layer reuses one compiled artifact.
    """
    from golden.runner import (
        _dispatch, _jit_entry, _validate_compiled_spec_abi, normalize_config,
    )
    from golden.spec import ScalarSpec, TensorSpec
    from golden.validation import validate_golden

    if isinstance(config, dict):
        config = dict(config)
        fresh_worker = bool(config.pop("fresh_worker", False))
    else:
        fresh_worker = False
    cfg = normalize_config(config)
    tensor_specs = [s for s in specs if isinstance(s, TensorSpec)]
    scalar_specs = [s for s in specs if isinstance(s, ScalarSpec)]
    by_name = {s.name: s for s in tensor_specs}
    tensors = {s.name: s.create_tensor() for s in tensor_specs}
    scalar_map = {s.name: s for s in scalar_specs}

    compiled_stages = []
    for key, names, with_epoch in stages:
        stage = [
            TensorSpec(name, by_name[name].shape, by_name[name].dtype, init_value=tensors[name])
            for name in names
            if isinstance(by_name[name], TensorSpec)
        ]
        # Scalar parameters listed in ``names`` are taken from the spec set too.
        stage.extend(
            ScalarSpec(
                by_name[name].name, by_name[name].dtype, by_name[name].value,
                compile_runtime=by_name[name].compile_runtime,
                benchmark_step=by_name[name].benchmark_step,
            )
            for name in names
            if isinstance(by_name[name], ScalarSpec)
        )
        if with_epoch:
            stage.extend(
                ScalarSpec(
                    spec.name, spec.dtype,
                    (stage_epochs or {}).get(key, spec.value) if spec.name == "epoch" else spec.value,
                    compile_runtime=spec.compile_runtime,
                    benchmark_step=spec.benchmark_step,
                )
                for spec in scalar_specs
            )
        compile_step, prologue, _ = _jit_entry(entries[key], stage)
        compiled = compile_step(
            cfg, prologue([s for s in stage if isinstance(s, ScalarSpec)], None),
        )
        _validate_compiled_spec_abi(compiled, stage)
        compiled_stages.append((key, compiled, stage))
        if validate and key == stages[0][0]:
            # The golden runs on a clone taken before any device write.
            scratch = {name: value.clone() for name, value in tensors.items()}
            golden_fn(scratch)
            golden_outputs = {name: scratch[name] for name in compare}

    if compile_only:
        return True, None

    # A one-shot ``compiled(...)`` call cannot be repeated for the same
    # compiled distributed program: the second call's CommDomain state is not
    # reclaimed and the shared exchange window times out.  ``fresh_worker``
    # dispatches each stage through ``prepare()`` (a freshly forked worker and
    # domain) and closes it, which is the supported one-dispatch path for a
    # program that was already dispatched once in this process.
    for key, compiled, stage in compiled_stages:
        if not validate:
            print(f"[FWD-STAGE] {key}", flush=True)
        # Dispatch with this stage's own scalar specs so a per-program epoch
        # override reaches the runtime ABI (the compiled scalar list is the
        # stage's scalar list).
        stage_scalars = {spec.name: spec for spec in stage if isinstance(spec, ScalarSpec)}
        if fresh_worker:
            _dispatch_prepared(compiled, stage, tensors, stage_scalars or scalar_map, cfg)
        else:
            _dispatch(compiled, stage, tensors, stage_scalars or scalar_map, cfg)

    if not validate:
        return True, None

    stamped = {}
    for _, _, stage in compiled_stages:
        for spec in stage:
            if isinstance(spec, TensorSpec):
                stamped.setdefault(spec.name, spec)
    device_outputs = {name: tensors[name] for name in compare}
    validation_inputs = {
        name: tensors[name] for name in stamped if name not in compare
    }
    if extra_inputs:
        validation_inputs.update(extra_inputs)
    validate_golden(
        device_outputs, golden_outputs, rtol=2e-2, atol=2e-2,
        compare_fn=compare, inputs=validation_inputs,
    )
    return True, None


def build_layer_specs(phase, mode, has_engram, seed=0, moe_tile=MOE_TILE):
    """Stacked 16-rank specs for one bounded layer case.

    ``phase`` selects the bounded token shape (prefill multi-token vs decode
    active tokens); the plan, cache geometry and MoE windows are the same for
    both phases.
    """
    from config import EP_SIZE
    from golden import ScalarSpec, TensorSpec
    from utils import build_request_plan, pack_nz, quant_w_per_channel

    requests = PREFILL_REQUESTS if phase == "prefill" else DECODE_REQUESTS
    ratio = 2 if mode in (MODE_C2A_FULL, MODE_C2A_REUSE) else 1
    torch.manual_seed(seed)
    plan = build_request_plan(list(requests), ratio=ratio)
    tokens = plan["tokens"]
    rows = EP_SIZE
    pages = plan["pages"]
    table = plan["index_block_table"]
    batch = table.shape[0]
    positions = plan["position_ids"]

    cos, sin = _rope_tables(int(positions.max()) + 2)
    compressed_cos, compressed_sin = _rope_tables(
        int(positions.max()) + 2, theta=M.compress_rope_theta,
    )
    compressed_positions = compressed_source_positions(positions, ratio=ratio)

    def quantized(n, k):
        logical, scale = quant_w_per_channel(torch.randn(n, k) * 0.05)
        return pack_nz(logical.transpose(0, 1).contiguous()), scale

    generator = torch.Generator().manual_seed(seed)
    rep = {}
    rep["hc_attn_fn"] = torch.cat(
        [torch.randn(MIX_HC, HC_DIM) / HC_DIM**0.5, torch.zeros(32 - MIX_HC, HC_DIM)], dim=0
    )
    rep["hc_ffn_fn"] = torch.cat(
        [torch.randn(MIX_HC, HC_DIM) / HC_DIM**0.5, torch.zeros(32 - MIX_HC, HC_DIM)], dim=0
    )
    rep["hc_attn_scale"] = torch.randn(3)
    rep["hc_ffn_scale"] = torch.randn(3)
    rep["hc_attn_base"] = torch.randn(MIX_HC)
    rep["hc_ffn_base"] = torch.randn(MIX_HC)
    rep["attn_norm_w"] = (torch.randn(D) * 0.1 + 1.0).to(torch.bfloat16)
    rep["ffn_norm_w"] = (torch.randn(D) * 0.1 + 1.0).to(torch.bfloat16)
    rep["wq_a"], rep["wq_a_scale"] = quantized(Q_LORA, D)
    rep["q_norm_w"] = (torch.randn(Q_LORA) * 0.1 + 1.0).to(torch.bfloat16)
    rep["wkv"], rep["wkv_scale"] = quantized(HEAD_DIM, D)
    rep["kv_norm_w"] = (torch.randn(HEAD_DIM) * 0.1 + 1.0).to(torch.bfloat16)
    rep["compressor_wkv"] = (torch.randn(HEAD_DIM, D) * 0.02).to(torch.bfloat16)
    rep["compressor_wgate"] = (torch.randn(HEAD_DIM, D) * 0.02).to(torch.bfloat16)
    rep["compressor_norm"] = (torch.randn(HEAD_DIM) * 0.05 + 1.0).to(torch.bfloat16)
    rep["index_weights_proj"] = (torch.randn(D, IDX_HEADS) * 0.02).to(torch.bfloat16)
    rep["index_wk"] = (torch.randn(IDX_DIM, HEAD_DIM) * 0.02).to(torch.bfloat16)
    rep["index_norm_w"] = (torch.randn(IDX_DIM) * 0.05 + 1.0).to(torch.bfloat16)
    rep["gate_weight"] = torch.randn(384, D) * 0.01
    rep["gate_bias"] = torch.randn(384) * 0.01
    rep["rope_cos"] = cos[positions]
    rep["rope_sin"] = sin[positions]
    rep["compressed_cos"] = compressed_cos[compressed_positions]
    rep["compressed_sin"] = compressed_sin[compressed_positions]
    rep["window_slots"] = plan["window_slots"]
    rep["window_indices"] = plan["window_indices"]
    rep["request_ids"] = plan["request_ids"]
    rep["compressed_lens"] = plan["compressed_lens"]
    rep["compressed_slots"] = plan["compressed_slots"]
    rep["position_ids"] = plan["position_ids"]
    rep["index_block_table"] = table

    per_rank = {
        name: [] for name in (
            "wq_b", "wq_b_scale", "sink", "wo_a", "wo_b", "index_wq_b", "index_wq_b_scale",
        )
    }
    for _ in range(rows):
        wq_b, wq_b_scale = quantized(LOCAL_HEADS * HEAD_DIM, Q_LORA)
        index_wq_b, index_wq_b_scale = quantized(IDX_HEADS * IDX_DIM, Q_LORA)
        per_rank["wq_b"].append(wq_b)
        per_rank["wq_b_scale"].append(wq_b_scale)
        per_rank["sink"].append(torch.randn(LOCAL_HEADS) * 0.5)
        per_rank["wo_a"].append(torch.randn(2, 1024, 4096).to(torch.bfloat16) * 0.02)
        per_rank["wo_b"].append(torch.randn(D, 2048).to(torch.bfloat16) * 0.02)
        per_rank["index_wq_b"].append(index_wq_b)
        per_rank["index_wq_b_scale"].append(index_wq_b_scale)

    reuse = mode in (MODE_C2A_REUSE, MODE_C1A_REUSE)
    group = {}
    for base in range(0, rows, TP_SIZE):
        group[base] = {
            "x_hc": torch.randn(tokens, HC_MULT, D),
            "pre_mix": torch.sigmoid(torch.randn(tokens, HC_MULT)) + M.hc_eps,
            "engram_lookup": (
                (torch.randn(tokens, 6144) * 0.02).to(torch.bfloat16)
                if has_engram else torch.zeros(tokens, 6144, dtype=torch.bfloat16)
            ),
            "topk_indices": (
                _selection_input(plan, tokens) if reuse
                else torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
            ),
            "window_cache": (
                torch.randn(pages, BLOCK_SIZE, 1, HEAD_DIM, generator=generator) * 0.1
            ).to(torch.bfloat16),
            "compressed_cache": (
                torch.randn(pages, BLOCK_SIZE, 1, HEAD_DIM, generator=generator) * 0.1
            ).to(torch.bfloat16),
            "index_cache": (
                torch.randn(pages, BLOCK_SIZE, 1, IDX_DIM, generator=generator) * 0.1
            ).to(torch.bfloat16),
            "state_cache": torch.randn(batch, 2, 2 * HEAD_DIM),
            "state_block_table": plan["state_block_table"],
        }

    expert = _expert_fixture(seed + 977)
    engram_wkv = (
        (torch.randn(25600, 6144) * 0.02).to(torch.bfloat16)
        if has_engram else torch.zeros(25600, 6144, dtype=torch.bfloat16)
    )
    engram_qk = torch.randn(4, D) * 0.1 if has_engram else torch.zeros(4, D)

    def stacked_rep(value):
        return value.unsqueeze(0).expand(rows, *value.shape).contiguous()

    def stacked_group(name):
        return torch.stack([group[r // TP_SIZE * TP_SIZE][name] for r in range(rows)])

    values = {}
    for name, value in rep.items():
        values[name] = stacked_rep(value)
    for name, entries in per_rank.items():
        values[name] = torch.stack(entries)
    for name in ("x_hc", "pre_mix", "engram_lookup", "window_cache", "compressed_cache",
                 "index_cache", "state_cache", "state_block_table", "topk_indices"):
        values[name] = stacked_group(name)
    values["index_block_table"] = stacked_rep(table)
    values["topk_values"] = torch.zeros(rows, tokens, INDEX_TOPK, dtype=torch.float32)
    values["pool_block_ids"] = torch.full((rows, tokens, POOL_BLOCKS), -1, dtype=torch.int32)
    values["pool_block_counts"] = torch.zeros(rows, tokens, dtype=torch.int32)
    if mode == MODE_C1A_REINDEX:
        from indexer import candidate_pool_fixture

        pool_ids, pool_counts = candidate_pool_fixture(plan["compressed_lens"], POOL_BLOCKS)
        values["pool_block_ids"] = stacked_rep(pool_ids)
        values["pool_block_counts"] = stacked_rep(pool_counts)
    for name, value in (
        ("w1", expert["w1"][0]), ("w1_scale", expert["w1"][1]),
        ("w3", expert["w3"][0]), ("w3_scale", expert["w3"][1]),
        ("w2", expert["w2"][0]), ("w2_scale", expert["w2"][1]),
        ("shared_w1", expert["shared_w1"][0]), ("shared_w1_scale", expert["shared_w1"][1]),
        ("shared_w3", expert["shared_w3"][0]), ("shared_w3_scale", expert["shared_w3"][1]),
        ("shared_w2", expert["shared_w2"][0]), ("shared_w2_scale", expert["shared_w2"][1]),
    ):
        values[name] = stacked_rep(value)
    values["engram_wkv"] = stacked_rep(engram_wkv)
    values["engram_qk"] = stacked_rep(engram_qk)
    from engram import fixture_rotation

    values["engram_rotation"] = stacked_rep(fixture_rotation())
    for name, shape, dtype, fill in (
        ("hidden", [rows, tokens, D], torch.bfloat16, 0.0),
        ("attn_input", [rows, tokens, D], torch.bfloat16, 0.0),
        ("attn_out", [rows, tokens, D], torch.bfloat16, 0.0),
        ("x_hc_mid", [rows, tokens, HC_MULT, D], torch.float32, 0.0),
        ("ffn_input", [rows, tokens, D], torch.bfloat16, 0.0),
        ("ffn_owned", [rows, tokens, D], torch.bfloat16, 0.0),
        ("ffn_normed", [rows, tokens, D], torch.bfloat16, 0.0),
        ("route_indices", [rows, tokens, TOPK], torch.int32, -1),
        ("route_weights", [rows, tokens, TOPK], torch.float32, 0.0),
        ("x_hc_out", [rows, tokens, HC_MULT, D], torch.float32, 0.0),
        ("next_pre_mix", [rows, tokens, HC_MULT], torch.float32, 0.0),
        ("ffn_post_mix", [rows, tokens, HC_MULT], torch.float32, 0.0),
        ("ffn_residual_mix", [rows, tokens, HC_MULT, HC_MULT], torch.float32, 0.0),
        ("moe_x", [rows, moe_tile, D], torch.bfloat16, 0.0),
        ("moe_indices", [rows, moe_tile, TOPK], torch.int32, 0),
        ("moe_weights", [rows, moe_tile, TOPK], torch.float32, 0.0),
        ("routed_pad", [rows, moe_tile, D], torch.bfloat16, 0.0),
    ):
        values[name] = torch.full(shape, fill, dtype=dtype)

    specs = [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name])
        for name in LAYER_SPEC_ORDER
    ]
    specs.extend([
        TensorSpec("token_counts", [EP_SIZE], torch.int32, init_value=torch.full((EP_SIZE,), tokens, dtype=torch.int32)),
        ScalarSpec("epoch", torch.int32, 1),
    ])
    # InOut inputs are stamped as outputs, so the comparators cannot read them
    # back from ``inputs``; keep a pristine copy of the values they started from.
    capture_fixture({
        "pre_mix": stacked_group("pre_mix").clone(),
        "window_cache": stacked_group("window_cache").clone(),
        "compressed_cache": stacked_group("compressed_cache").clone(),
        "index_cache": stacked_group("index_cache").clone(),
        "state_cache": stacked_group("state_cache").clone(),
        "topk_indices": stacked_group("topk_indices").clone(),
        "topk_values": values["topk_values"].clone(),
        "pool_block_ids": values["pool_block_ids"].clone(),
        "pool_block_counts": values["pool_block_counts"].clone(),
    })
    return specs
