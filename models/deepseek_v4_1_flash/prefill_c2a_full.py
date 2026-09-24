# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C2A Full wired through mHC, and the validation shared with Reuse.

One attention sublayer of the text backbone, in the order the official ``Block.forward`` runs it::

    attn_pre, attn_post, attn_comb = hc_mixes(x_hc)   # coefficients from this sublayer's input
    x = rms_norm(hc_pre(x_hc, pre_mix))              # pre_mix: the previous sublayer's delayed mix
    x = attention(x)                                  # prefill C2A Full or Reuse, TP-reduced
    x_hc = hc_post(x, x_hc, attn_post, attn_comb)

``attn_pre`` comes back as ``next_pre_mix``; the FFN sublayer of the same layer collapses its input
with it.
"""

import argparse
import math
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from golden import ScalarSpec, TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.config import (
    B_DYN,
    CMP_BLOCKS_DYN,
    D,
    FLASH,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    INDEX_H,
    INDEX_DIM,
    LOCAL_H,
    LOCAL_O_WIDTH,
    Q_LORA,
    ROPE_DIM,
    INDEX_BLOCKS_DYN,
    MIX_HC,
    ORI_BLOCKS_DYN,
    T_DYN,
    TABLE_DYN,
    TP_SIZE,
)
from models.deepseek_v4_1_flash.decode_attn_c2a_full import (
    CACHE_SLOTS,
    CMP_PACKED,
    CMP_SCALES,
    IDX_PACKED,
    IDX_SCALES,
    INPUT_NAMES,
    STATE_METADATA,
    FULL_ROPE_NAMES,
    PROGRAM_INPUT_NAMES,
    MUTABLE_NAMES,
    SHARDED_NAMES,
    compare_cache,
    compare_output,
    compare_per_rank,
    compare_replicated,
    compare_state,
    compare_topk,
    make_c2a_inputs,
    official_reference_c2a,
)
from models.deepseek_v4_1_flash.decode_attn_c2a_reuse import (
    REUSE_INPUT_NAMES,
    REUSE_MUTABLE_NAMES,
    REUSE_SHARDED_NAMES,
    make_c2a_reuse_inputs,
    official_reference_c2a_reuse,
)
from models.deepseek_v4_1_flash.golden import hc_mixes, hc_post, hc_pre, rms_norm as golden_rms_norm
from models.deepseek_v4_1_flash.hc_mixes import mhc_mixes
from models.deepseek_v4_1_flash.hc_post import mhc_post
from models.deepseek_v4_1_flash.hc_pre import mhc_pre
from models.deepseek_v4_1_flash.prefill_attn_c2a_full import prefill_attn_c2a_full
from models.deepseek_v4_1_flash.rope_tables import ROPE_ROWS_DYN, materialize_rope_rows
from models.deepseek_v4_1_flash.rmsnorm import rms_norm


@pl.jit.inline
def attention_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    x: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Derive this sublayer's mixes, then collapse with the delayed pre-mix and normalize."""
    tokens = pl.tensor.dim(x_hc, 0)
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    mhc_pre(x_hc, pre_mix, collapsed)
    rms_norm(collapsed, attn_norm_weight, x)
    return x


@pl.jit.inline
def prefill_c2a_full(
    x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0],
    compressed_cache: pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ],
    token_to_req_indices: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    index_cache: pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[C.Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[C.B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[C.STATE_BLOCKS_DYN, C.STATE_CAPACITY, C.STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    attn_input: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    attn_output: pl.Tensor[[C.T_DYN, C.D], pl.BF16],
    next_pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    x_hc_out: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Full attention sublayer between mHC pre and post.

    Every row of x_hc is active, so ``num_tokens`` equals its row count. ``attn_input`` receives
    the normalized attention input and ``attn_output`` the TP-reduced attention output; both are
    caller workspace, exposed so each stage can be checked. Outputs must not alias inputs.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    attention_hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, next_pre_mix,
        post_mix, residual_mix, attn_input,
    )
    prefill_attn_c2a_full(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
        attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices,
        compressed_lens, index_cache, index_cache_scale, index_block_table, position_ids,
        compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate,
        query_start_loc, state_block_table, state_cache, compressor_norm_weight, compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, output_window, output_arrived, attn_output, group_base, tp_rank,
        num_tokens, attention_epoch,
    )
    mhc_post(attn_output, x_hc, post_mix, residual_mix, x_hc_out)
    return x_hc_out


def golden_attention_input(
    x_hc: torch.Tensor,
    pre_mix: torch.Tensor,
    attn_norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Collapse the streams with the delayed pre-mix, round to BF16, then apply RMSNorm."""
    return golden_rms_norm(hc_pre(x_hc, pre_mix).to(torch.bfloat16), attn_norm_weight)


HC_INPUT_NAMES = ("x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight")
MODES = {
    "full": (INPUT_NAMES, MUTABLE_NAMES, SHARDED_NAMES, official_reference_c2a),
    "reuse": (REUSE_INPUT_NAMES, REUSE_MUTABLE_NAMES, REUSE_SHARDED_NAMES, official_reference_c2a_reuse),
}
# Per-rank attention results besides the TP-reduced output.
ATTENTION_STATE = {"full": MUTABLE_NAMES + ("topk_indices",), "reuse": REUSE_MUTABLE_NAMES}


def make_hc_inputs(tokens, seed, case):
    """mHC inputs with distinct per-token magnitudes so both RMS statistics stay observable."""
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    # The residual stream holds BF16 values; FP32 storage only keeps mhc_post's single rounding.
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()
    values = {
        "x_hc": x_hc,
        "pre_mix": torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen)) + FLASH.hc_eps,
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM),
        "hc_attn_scale": torch.randn(3, generator=gen),
        "hc_attn_base": torch.randn(MIX_HC, generator=gen),
        "attn_norm_weight": (torch.randn(D, generator=gen) * 0.1 + 1).bfloat16(),
    }
    if case == "zero":
        values["x_hc"] = torch.zeros_like(x_hc)
    return values


def make_attention_inputs(mode, tokens, requests, seed, case):
    """Attention fixture of the chained mode, without ``x``: the chain derives it from x_hc."""
    make = make_c2a_inputs if mode == "full" else make_c2a_reuse_inputs
    options = {"full_rope_tables": True} if mode == "full" else {}
    values = make(tokens=tokens, requests=requests, seed=seed, case=case, mode="prefill", **options)
    return {name: value for name, value in values.items() if name != "x"}


def reference_attention(mode, epochs, tensors, x, ranks):
    """Reference one TP group's attention on ``x``: the FP32 TP sum and each rank's last result."""
    names, mutable_names, _, reference = MODES[mode]
    partials, results = [], []
    for rank in ranks:
        inputs = {
            name: x if name == "x" else tensors[name][rank]
            for name in names if mode != "full" or name not in FULL_ROPE_NAMES
        }
        if mode == "full":
            for row_name, table_name in FULL_ROPE_NAMES.items():
                positions_name = "compressed_rope_positions" if row_name.startswith("compressed_") else "position_ids"
                positions = tensors[positions_name][rank].to(torch.long)
                identity = 1.0 if row_name.endswith("cos") else 0.0
                inputs[row_name] = tensors[table_name][rank][positions.clamp_min(0)].masked_fill(
                    positions[:, None] < 0, identity
                )
        for _ in range(epochs):
            result = reference(inputs)
            for name in mutable_names:
                inputs[name] = result[name]
        partials.append(result["output"].float())
        results.append(result)
    return sum(partials).bfloat16(), results


def make_golden(mode, epochs):
    """Reference each TP group: mHC pre on the leader, per-rank attention, FP32 TP sum, mHC post."""

    def golden(tensors):
        """Fill every chained output and the attention state each rank publishes."""
        for base in range(0, tensors["x_hc"].shape[0], TP_SIZE):
            ranks = range(base, base + TP_SIZE)
            x_hc = tensors["x_hc"][base]
            pre, post, comb = hc_mixes(
                x_hc, tensors["hc_attn_fn"][base], tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base]
            )
            x = golden_attention_input(x_hc, tensors["pre_mix"][base], tensors["attn_norm_weight"][base])
            attention, results = reference_attention(mode, epochs, tensors, x, ranks)
            for rank, result in zip(ranks, results):
                for name in ATTENTION_STATE[mode]:
                    tensors[name][rank].copy_(result[name])
            outputs = {
                "attn_input": x,
                "attn_output": attention,
                "next_pre_mix": pre,
                "x_hc_out": hc_post(attention, x_hc, post, comb).float(),
            }
            for name, value in outputs.items():
                tensors[name][base : base + TP_SIZE].copy_(value.expand_as(tensors[name][base : base + TP_SIZE]))

    return golden


class StagedAttentionReference:
    """The attention reference re-run on the device's own ``attn_input`` (teacher forcing).

    One BF16 ULP of the attention input can move an MXFP8 group scale or an E2M1 code, and with
    it the attention output by close to 1%. The device's mHC pre and RMSNorm reduce in another
    order than torch and flip a handful of such ULPs, so the attention stage keeps its standalone
    budget against this reference while ``attn_input`` itself is checked against the golden input.
    """

    def __init__(self, mode, epochs, initial_state):
        """Keep the mode, the epoch count and the pre-run cache and state of every rank."""
        self.mode = mode
        self.epochs = epochs
        self.initial_state = initial_state
        self.outputs = None

    def __call__(self, inputs, actual_outputs):
        """Reference every rank's attention on the device's ``attn_input``, once per run."""
        if self.outputs is None:
            tensors = {**inputs, **self.initial_state}
            x = actual_outputs["attn_input"]
            names = ("attn_output",) + ATTENTION_STATE[self.mode]
            outputs = {name: torch.empty_like(actual_outputs[name]) for name in names}
            for base in range(0, x.shape[0], TP_SIZE):
                ranks = range(base, base + TP_SIZE)
                attention, results = reference_attention(self.mode, self.epochs, tensors, x[base], ranks)
                outputs["attn_output"][base : base + TP_SIZE] = attention
                for rank, result in zip(ranks, results):
                    for name in ATTENTION_STATE[self.mode]:
                        outputs[name][rank] = result[name]
            self.outputs = outputs
        return self.outputs

    def compare(self, name, check):
        """Hold one attention output to ``check`` against the teacher-forced reference."""

        def staged(actual, expected, *, inputs, actual_outputs, expected_outputs, **kwargs):
            """Report the end-to-end difference, then compare against the forced reference."""
            if name == "attn_output":
                for base in range(0, actual.shape[0], TP_SIZE):
                    _report("attn_output(end-to-end, not gated)", actual[base], expected[base])
            forced = self(inputs, actual_outputs)
            return check(
                actual, forced[name], inputs=inputs, actual_outputs=actual_outputs, expected_outputs=forced, **kwargs
            )

        return staged


# A stage that ends in BF16 rounding may flip isolated elements by one ULP; a token row as a whole
# moving by 2**-8 (one ULP everywhere) is a real error that the outlier allowance of
# ratio_allclose would still admit once the row is under 0.5% of the tensor.
ROW_BUDGET = 2.0**-8


def _report(name, actual, expected):
    """Print and return (rel_l2, worst token-row rel_l2); rows are the leading axis."""
    diff = actual.double() - expected.double()
    rel_l2 = (diff.norm() / expected.double().norm().clamp_min(1e-12)).item()
    rows = diff.flatten(1).norm(dim=-1) / expected.double().flatten(1).norm(dim=-1).clamp_min(1e-12)
    print(
        f"[PRECISION] {name} rel_l2={rel_l2:.6g} max_abs={diff.abs().max().item():.6g} "
        f"max_row_rel_l2={rows.max().item():.6g}"
    )
    return rel_l2, rows.max().item()


def compare_group_leaders(name, check, max_row_rel_l2):
    """Hold every TP group leader to ``check`` and a per-token bound; replicas must be byte identical."""

    def compare(actual, expected, **kwargs):
        """Check each leader, its worst token row, and that the replicas are byte identical."""
        passed = True
        for base in range(0, actual.shape[0], TP_SIZE):
            _, worst_row = _report(name, actual[base], expected[base])
            valid, _ = check(actual[base], expected[base], **kwargs)
            passed &= valid and worst_row <= max_row_rel_l2
            passed &= all(torch.equal(actual[base], actual[rank]) for rank in range(base + 1, base + TP_SIZE))
        return passed, f"allclose budget, every token row <= {max_row_rel_l2:.3g} rel L2, replicas identical"

    return compare


def compare_x_hc_out(actual, expected, *, inputs, actual_outputs, **kwargs):
    """The new residual stream.

    hc_post is replayed on the device's own attention output and held to hc_post's budget, which
    isolates the mHC stages from the FP8 attention error ``attn_output`` is already held to. The
    end-to-end bound is only a sanity check: the residual term dilutes attention error about 4x.
    """
    check = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    passed = True
    for base in range(0, actual.shape[0], TP_SIZE):
        x_hc = inputs["x_hc"][base]
        _, post, comb = hc_mixes(
            x_hc, inputs["hc_attn_fn"][base], inputs["hc_attn_scale"][base], inputs["hc_attn_base"][base]
        )
        replay = hc_post(actual_outputs["attn_output"][base], x_hc, post, comb).float()
        _, worst_row = _report("x_hc_out(replay)", actual[base], replay)
        valid, _ = check(actual[base], replay, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
        passed &= valid and worst_row <= ROW_BUDGET
        passed &= _report("x_hc_out(end-to-end)", actual[base], expected[base])[0] <= 0.01
        passed &= bool(torch.isfinite(actual[base]).all())
        passed &= all(torch.equal(actual[base], actual[rank]) for rank in range(base + 1, base + TP_SIZE))
    return passed, "hc_post replay within hc_post's budget per token row; end-to-end rel L2 <= 1%"


def make_compare(mode, epochs, initial_state):
    """Stage-wise comparators: mHC pre vs golden, attention teacher-forced, hc_post replayed."""
    staged = StagedAttentionReference(mode, epochs, initial_state)
    compare = {
        # attn_input ends in BF16 rounding (collapse, then norm): hc_pre's budget per element.
        "attn_input": compare_group_leaders(
            "attn_input", ratio_allclose(atol=1e-4, rtol=1.0 / 128), ROW_BUDGET
        ),
        # next_pre_mix is hc_mixes' FP32 pre coefficient: hc_mixes' budget.
        "next_pre_mix": compare_group_leaders(
            "next_pre_mix", ratio_allclose(atol=2.5e-5, rtol=5e-3), 5e-3
        ),
        "attn_output": staged.compare("attn_output", compare_replicated(compare_output)),
        "x_hc_out": compare_x_hc_out,
    }
    if mode == "full":
        compare["topk_indices"] = staged.compare("topk_indices", compare_per_rank(compare_topk))
        compare["state_cache"] = staged.compare("state_cache", compare_per_rank(compare_state, STATE_METADATA))
        for name, slots in CACHE_SLOTS.items():
            compare[name] = staged.compare(name, compare_per_rank(compare_cache(name), slots))
    else:
        for name in REUSE_MUTABLE_NAMES:
            compare[name] = staged.compare(name, compare_per_rank(compare_cache(name), "window_slots"))
    return compare


def make_hc_program(capacity, world_size, epochs):
    """Build the L3 group entry that runs one mHC-wrapped C2A Full sublayer on every rank."""

    @pl.jit
    def c2a_full_rank(
        x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
        pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // 32, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // 32, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0, pl.MX_B_NN],
        freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, C.ROPE_DIM // 2], pl.FP32],
        freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.InOut[pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
        compressed_cache_scale: pl.InOut[pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN]],
        token_to_req_indices: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
        index_cache_scale: pl.InOut[pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0]],
        index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
        position_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_rope_positions: pl.Tensor[[C.T_DYN], pl.INT32],
        compressor_wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
        compressor_wgate: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
        query_start_loc: pl.Tensor[[C.Q_START_DYN], pl.INT32],
        state_block_table: pl.Tensor[[C.B_DYN, 1], pl.INT32],
        state_cache: pl.InOut[pl.Tensor[[C.STATE_BLOCKS_DYN, C.STATE_CAPACITY, C.STATE_WIDTH], pl.FP32]],
        compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
        index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.Out[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        attn_input: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[C.T_DYN, C.D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
        output_window: pld.DistributedTensor[[capacity, C.D], pl.FP32],
        output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Bind the runtime shapes and run the sublayer once per epoch on one rank."""
        freqs_cos.bind_dynamic(0, ROPE_ROWS_DYN)
        x_hc.bind_dynamic(0, T_DYN)
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
        index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
        index_block_table.bind_dynamic(0, B_DYN)
        index_block_table.bind_dynamic(1, TABLE_DYN)
        query_start_loc.bind_dynamic(0, C.Q_START_DYN)
        state_block_table.bind_dynamic(0, C.B_DYN)
        state_cache.bind_dynamic(0, C.STATE_BLOCKS_DYN)
        if num_tokens > 0:
            tokens = pl.tensor.dim(x_hc, 0)
            rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
            rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
            compressed_rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
            compressed_rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
            materialize_rope_rows(freqs_cos, freqs_sin, position_ids, num_tokens, rope_cos, rope_sin)
            materialize_rope_rows(
                compressed_freqs_cos, compressed_freqs_sin, compressed_rope_positions, num_tokens,
                compressed_rope_cos, compressed_rope_sin,
            )
            for step in pl.range(epochs):
                prefill_c2a_full(
                    x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
                    wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
                    kv_norm_weight, attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
                    window_slots, window_indices, window_cache, window_cache_scale,
                    compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
                    index_cache, index_cache_scale, index_block_table, position_ids,
                    compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate,
                    query_start_loc, state_block_table, state_cache, compressor_norm_weight,
                    compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
                    index_weights_proj, topk_indices, output_window, output_arrived, attn_input,
                    attn_output, next_pre_mix, x_hc_out, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE,
                    num_tokens, attention_epoch + step,
                )
        return (
            x_hc_out, next_pre_mix, attn_output, attn_input, topk_indices, window_cache, window_cache_scale,
            compressed_cache, compressed_cache_scale, index_cache, index_cache_scale, state_cache,
        )

    @pl.jit.host
    def c2a_full_group(
        x_hc: pl.Tensor[[world_size, C.T_DYN, C.HC_MULT, C.D], pl.FP32],
        pre_mix: pl.Tensor[[world_size, C.T_DYN, C.HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[world_size, C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[world_size, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[world_size, C.MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[world_size, C.D], pl.BF16],
        wq_a: pl.Tensor[[world_size, C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[world_size, C.D // 32, C.Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[world_size, C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[world_size, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[world_size, C.Q_LORA // 32, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[world_size, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[world_size, C.D // 32, C.HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[world_size, C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[world_size, C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[world_size, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[world_size, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[world_size, C.LOCAL_O_WIDTH // 32, C.D], pl.FP8E8M0],
        freqs_cos: pl.Tensor[[world_size, ROPE_ROWS_DYN, C.ROPE_DIM // 2], pl.FP32],
        freqs_sin: pl.Tensor[[world_size, ROPE_ROWS_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[world_size, C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[world_size, C.T_DYN, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[world_size, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[world_size, C.ORI_BLOCKS_DYN, 128, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.InOut[pl.Tensor[[world_size, C.CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[[world_size, C.CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN]
        ],
        token_to_req_indices: pl.Tensor[[world_size, C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[world_size, C.T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor[[world_size, C.INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
        index_cache_scale: pl.InOut[pl.Tensor[[world_size, C.INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0]],
        index_block_table: pl.Tensor[[world_size, C.B_DYN, C.TABLE_DYN], pl.INT32],
        position_ids: pl.Tensor[[world_size, C.T_DYN], pl.INT32],
        compressed_freqs_cos: pl.Tensor[[world_size, ROPE_ROWS_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_freqs_sin: pl.Tensor[[world_size, ROPE_ROWS_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_rope_positions: pl.Tensor[[world_size, C.T_DYN], pl.INT32],
        compressor_wkv: pl.Tensor[[world_size, C.D, C.HEAD_DIM], pl.FP32],
        compressor_wgate: pl.Tensor[[world_size, C.D, C.HEAD_DIM], pl.FP32],
        query_start_loc: pl.Tensor[[world_size, C.Q_START_DYN], pl.INT32],
        state_block_table: pl.Tensor[[world_size, C.B_DYN, 1], pl.INT32],
        state_cache: pl.InOut[
            pl.Tensor[[world_size, C.STATE_BLOCKS_DYN, C.STATE_CAPACITY, C.STATE_WIDTH], pl.FP32]
        ],
        compressor_norm_weight: pl.Tensor[[world_size, C.HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[world_size, C.T_DYN], pl.INT64],
        index_wk: pl.Tensor[[world_size, C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[world_size, C.INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[world_size, C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[world_size, C.Q_LORA // 32, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0],
        index_weights_proj: pl.Tensor[[world_size, C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        attn_input: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[world_size, C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Allocate the TP communication windows and launch one rank entry per device."""
        freqs_cos.bind_dynamic(1, ROPE_ROWS_DYN)
        x_hc.bind_dynamic(1, T_DYN)
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
        index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
        index_block_table.bind_dynamic(1, B_DYN)
        index_block_table.bind_dynamic(2, TABLE_DYN)
        query_start_loc.bind_dynamic(1, C.Q_START_DYN)
        state_block_table.bind_dynamic(1, C.B_DYN)
        state_cache.bind_dynamic(1, C.STATE_BLOCKS_DYN)
        data_buffer = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            data = pld.window(data_buffer, [capacity, D], dtype=pl.FP32)
            signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            # Each rank consumes packed MX_B_NN scale rows.
            wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
            wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
            wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
            wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
            index_wq_b_scale_r: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale[rank]
            c2a_full_rank(
                x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
                attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank],
                wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank],
                attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale_r, freqs_cos[rank],
                freqs_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank],
                window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
                token_to_req_indices[rank], compressed_lens[rank], index_cache[rank],
                index_cache_scale[rank], index_block_table[rank], position_ids[rank],
                compressed_freqs_cos[rank], compressed_freqs_sin[rank], compressed_rope_positions[rank], compressor_wkv[rank],
                compressor_wgate[rank], query_start_loc[rank], state_block_table[rank], state_cache[rank],
                compressor_norm_weight[rank], compressed_slots[rank], index_wk[rank],
                index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r,
                index_weights_proj[rank], topk_indices[rank], attn_input[rank], attn_output[rank],
                next_pre_mix[rank], x_hc_out[rank], data, signal, rank, num_tokens, attention_epoch,
                device=rank,
            )

    return c2a_full_group


def build_specs(args, mode, initial_state):
    """Stacked specs whose per-rank fixtures are drawn once, on first use.

    The comparators see pure inputs only, so the initial cache and state are kept here for the
    staged attention reference.
    """
    world_size = TP_SIZE * args.dp
    names, mutable_names, sharded_names, _ = MODES[mode]
    program_names = PROGRAM_INPUT_NAMES if mode == "full" else names
    attention_names = tuple(name for name in program_names if name != "x")
    ranks = {}

    def initialize(name):
        """Draw every rank's fixture on the first spec that needs it, then stack one tensor."""
        if not ranks:
            for rank in range(world_size):
                ranks[rank] = make_attention_inputs(
                    mode, args.tokens, args.requests, args.seed + rank, args.case
                )
            for rank in range(world_size):
                leader = rank // TP_SIZE * TP_SIZE
                # Every rank of a TP group sees the same tokens, residual stream, metadata and caches.
                ranks[rank].update(
                    make_hc_inputs(args.tokens, args.seed + 7919 * (leader + 1), args.case)
                    if rank == leader else {key: ranks[leader][key] for key in HC_INPUT_NAMES}
                )
                for key in attention_names:
                    if key not in sharded_names:
                        ranks[rank][key] = ranks[leader][key]
        column = [ranks[rank][name] for rank in range(world_size)]
        if column[0].dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            stacked = torch.stack([value.view(torch.uint8) for value in column]).view(column[0].dtype)
        else:
            stacked = torch.stack(column)
        if name in mutable_names:
            initial_state[name] = stacked.clone()
        return stacked

    shapes = make_attention_inputs(mode, args.tokens, args.requests, args.seed, args.case)
    shapes.update(make_hc_inputs(args.tokens, args.seed, args.case))
    specs = [
        TensorSpec(
            name,
            [world_size, *shapes[name].shape],
            shapes[name].dtype,
            init_value=(lambda name=name: initialize(name)),
            resident="stacked",
        )
        for name in HC_INPUT_NAMES + attention_names
    ]
    if mode == "full":
        specs.append(
            TensorSpec("topk_indices", [world_size, args.tokens, C.INDEX_TOPK], torch.int32, resident="stacked")
        )
    specs += [
        TensorSpec("attn_input", [world_size, args.tokens, D], torch.bfloat16, resident="stacked"),
        TensorSpec("attn_output", [world_size, args.tokens, D], torch.bfloat16, resident="stacked"),
        TensorSpec("next_pre_mix", [world_size, args.tokens, HC_MULT], torch.float32, resident="stacked"),
        TensorSpec("x_hc_out", [world_size, args.tokens, HC_MULT, D], torch.float32, resident="stacked"),
        ScalarSpec("num_tokens", torch.int32, args.tokens, compile_runtime=True),
        ScalarSpec(
            "attention_epoch",
            torch.int32,
            1,
            compile_runtime=True,
            benchmark_step=args.epochs if args.bench else None,
        ),
    ]
    return specs


def run_prefill_c2a(make_program, mode, argv=None):
    """Validate an mHC-wrapped prefill C2A sublayer on A5; ``make_program`` builds the L3 entry."""
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description=f"DeepSeek V4.1 prefill C2A {mode} with mHC: A5 precision")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--case", default="mixed", choices=["mixed", "long", "masked", "zero"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1, help="sublayer calls per dispatch; timing includes all epochs")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="official DeepSeek-V4.1-Flash checkpoint root")
    parser.add_argument("--checkpoint-layer", type=int, default=None)
    args = parser.parse_args(argv)

    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    devices = list(range(TP_SIZE * args.dp))
    if args.device:
        devices = [int(value) for value in args.device.split(",")]
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"--device must name {TP_SIZE * args.dp} distinct non-negative device IDs")
    if not 1 <= args.tokens <= C.PREFILL_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {C.PREFILL_MAX_TOKENS}]")
    if not 1 <= args.requests <= min(C.MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"--requests must be in [1, {min(C.MAX_BATCH_PER_DP, args.tokens)}]")
    if not 1 <= args.epochs <= 1000:
        parser.error("--epochs must be in [1, 1000]")
    torch.set_num_threads(8)

    print(
        f"[C2A-HC] mode={mode} tokens={args.tokens} requests={args.requests} TP={TP_SIZE} "
        f"DP={args.dp} case={args.case} seed={args.seed} epochs/dispatch={args.epochs} devices={devices}"
    )
    initial_state = {}
    specs = build_specs(args, mode, initial_state)
    if args.checkpoint:
        from models.deepseek_v4_1_flash.prefill_checkpoint_weights import (
            PrefillCheckpoint,
            bind_checkpoint_weights,
        )

        layer_id = args.checkpoint_layer
        if layer_id is None:
            layer_id = 2 if mode == "full" else 3
        checkpoint = PrefillCheckpoint(args.checkpoint)
        weights = (
            checkpoint.c2a_full_weights(layer_id, len(devices))
            if mode == "full" else checkpoint.c2a_reuse_weights(layer_id, len(devices))
        )
        bind_checkpoint_weights(specs, weights)
        print(f"[C2A-HC] checkpoint layer={layer_id} root={args.checkpoint}")
    result = run(
        fn=make_program(C.PREFILL_MAX_TOKENS, len(devices), args.epochs),
        specs=specs,
        golden_fn=make_golden(mode, args.epochs),
        compile_only=args.compile_only,
        config=dict(
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
        ),
        compare_fn=make_compare(mode, args.epochs, initial_state),
    )
    print(f"[C2A-HC] work_dir={result.work_dir}")
    if args.compile_only and result.passed:
        print("[C2A-HC] Compilation passed; device accuracy was NOT validated.")
    return result


def validate(argv=None):
    """Validate packed prefill C2A Full wired through mHC on A5."""
    return run_prefill_c2a(make_hc_program, "full", argv=argv)


__all__ = [
    "attention_hc_pre",
    "golden_attention_input",
    "prefill_c2a_full",
    "run_prefill_c2a",
]


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,dp", [(1, 1), (2, 2), (4, 1)])
    def test_precision(tp, dp, a5_args):
        """Validate the operator against its golden reference on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
