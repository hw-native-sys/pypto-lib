# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One complete packed-prefill text-backbone layer: attention and MoE sublayers through mHC.

The official ``Block.forward`` runs a layer as two mHC sublayers::

    attn_pre, attn_post, attn_comb = hc_mixes(x_hc)     # this sublayer's coefficients
    x     = attn_norm(hc_pre(x_hc, pre_mix))            # pre_mix: the previous sublayer's delayed mix
    x_hc  = hc_post(attention(x), x_hc, attn_post, attn_comb)
    ffn_pre, ffn_post, ffn_comb = hc_mixes(x_hc)
    y     = moe(hc_pre(x_hc, attn_pre))                 # ffn_norm lives inside the MoE gate
    x_hc  = hc_post(y, x_hc, ffn_post, ffn_comb)

``ffn_pre`` leaves the layer as ``next_pre_mix``: the next layer's attention sublayer collapses
its input with it. Attention is replicated over a TP group and reduced there; the EP MoE gives
each token row one owner inside that group, so the routed result comes back partitioned and one
FP32 sum over the group restores the replicated residual stream.

Every sublayer boundary is exposed as caller workspace (``attn_input``, ``attn_output``,
``x_hc_mid``, ``ffn_input``, ``ffn_output``), which is what lets the validation force each stage
on the device's own input instead of diluting one stage's error into the next. The layer runs
as three device entries - attention, MoE, TP restore - and ``make_layer_program`` says why they
carry the packed token extent statically and why each runs the layer exactly once per launch.
"""

import argparse
import functools
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# A5-only; intentionally excluded from the A2/A3 device sweep. The complete-layer
# entry exercises the packed MxFp4 routed-weight ABI and the TP-to-EP local-token path.
# ci: no-sim
# ci: a5

import torch

import pypto.language as pl
import pypto.language.distributed as pld


def _command_line_int(flag, default):
    """Read one integer flag before argparse; config and the MoE shapes freeze at import."""
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1])
    return default


LAYER_TP = _command_line_int("--tp", 1)
LAYER_DP = _command_line_int("--dp", 2)
LAYER_TOKENS = _command_line_int("--tokens", 48)
# One layer owns both sublayers, so the EP world is the attention world: config derives
# DP = EP / TP, and the MoE shapes below derive from EP. config reads the command line
# itself and has its own defaults, so hand it this entry's before it is imported.
if not any(argument == "--tp" or argument.startswith("--tp=") for argument in sys.argv):
    sys.argv += ["--tp", str(LAYER_TP)]
if not any(argument == "--ep" or argument.startswith("--ep=") for argument in sys.argv):
    sys.argv += ["--ep", str(LAYER_TP * LAYER_DP)]

from models.deepseek_v4_1_flash import config as C

# gate.py and the EP transport freeze their token extent at import; the layer routes the
# packed prefill tokens of one call.
C.MOE_TOKENS = LAYER_TOKENS

from models.deepseek_v4_1_flash.config import (  # noqa: E402
    CMP_BLOCKS_DYN,
    D,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    LOCAL_H,
    LOCAL_O_WIDTH,
    MIX_HC,
    ORI_BLOCKS_DYN,
    Q_LORA,
    T_DYN,
    TP_SIZE,
)
from models.deepseek_v4_1_flash.attention_tp import prefill_tp_output_all_reduce  # noqa: E402
from models.deepseek_v4_1_flash.decode_attn_c2a_full import (  # noqa: E402
    CMP_PACKED,
    CMP_SCALES,
    compare_cache,
    compare_output,
    compare_per_rank,
    compare_replicated,
)
from models.deepseek_v4_1_flash.decode_attn_c2a_reuse import REUSE_MUTABLE_NAMES  # noqa: E402
from models.deepseek_v4_1_flash.golden import gate, hc_mixes, hc_post, hc_pre, rms_norm  # noqa: E402
from models.deepseek_v4_1_flash.hc_mixes import mhc_mixes  # noqa: E402
from models.deepseek_v4_1_flash.hc_post import mhc_post  # noqa: E402
from models.deepseek_v4_1_flash.hc_pre import mhc_pre  # noqa: E402
from models.deepseek_v4_1_flash.prefill_c2a_full import (  # noqa: E402
    ATTENTION_STATE,
    HC_INPUT_NAMES,
    MODES,
    ROW_BUDGET,
    StagedAttentionReference,
    _report as report_stage,
    compare_group_leaders,
    golden_attention_input,
    make_attention_inputs,
    make_hc_inputs,
    reference_attention,
)
from models.deepseek_v4_1_flash.prefill_c2a_reuse import prefill_c2a_reuse  # noqa: E402
from models.deepseek_v4_1_flash.quantization import build_mxfp4_pair_lut  # noqa: E402
from models.deepseek_v4_1_flash.rmsnorm import rms_norm as npu_rms_norm  # noqa: E402
# Import moe before expert_routed: moe freezes C.RECV_MAX to this layer's token extent
# before expert_routed captures the transport buffer shape in its annotations.
from models.deepseek_v4_1_flash.moe import (  # noqa: E402
    AUX_WIDTH,
    EP_SIZE,
    MX_GROUP,
    N_LOCAL_EXPERTS,
    RECV_MAX,
    ROUTE_WIDTH,
    _golden_moe_core as golden_moe_core,
    _moe_core as moe_core,
)
from models.deepseek_v4_1_flash.expert_routed import (  # noqa: E402
    MX_PACKED_LANE_COLS,
    MX_W1_PACKED_ROWS,
    MX_W2_PACKED_ROWS,
    MX_W3_PACKED_ROWS,
    ROUTED_DEQUANT_STD,
    gen_routed_mxfp4_weights,
)

from golden import ScalarSpec, TensorSpec, ratio_allclose, run  # noqa: E402
from golden.validation import ratio_reldiff  # noqa: E402


MOE_INTER = C.MOE_INTER
N_EXPERTS = C.N_EXPERTS
# The packed prefill token extent of one layer call.
TOKENS = C.MOE_TOKENS
# The routed-result window holds one row per token per selected expert.
ROUTE_ROWS = C.MOE_TOKENS * C.TOPK
# End-to-end bound for the stages the MoE feeds. It is loose on purpose: the MoE quantizes
# its input to MXFP8 per group of 32, and that step amplifies about tenfold - measured on
# this fixture, a 0.19% difference in the collapsed stream becomes 1.9% once quantized,
# because an E4M3 code is coarse enough that a small shift moves many elements of a group
# a whole code. The per-stage forced checks are the acceptance; this bound only has to
# catch a miswired chain.
CHAIN_BUDGET = 0.15
# Attention plus a four-stream mHC and the EP MoE dispatch buffers overflow the default
# 256 MiB ring heap; one GiB per ring is the prefill attention and MoE bring-up budget.
LAYER_RING_HEAP = (1024 * 1024 * 1024,) * 4


@pl.jit.inline
def moe_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    ffn_input: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Derive the FFN sublayer's mixes, then collapse with the attention sublayer's pre-mix.

    The collapsed stream stays in BF16 here; ``prefill_moe_sublayer`` applies RMSNorm
    immediately before the packed MoE core consumes it.
    """
    mhc_mixes(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, next_pre_mix, post_mix, residual_mix)
    mhc_pre(x_hc, pre_mix, ffn_input)
    return ffn_input


@pl.jit.inline
def widen_to_fp32(
    source: pl.Tensor[[T_DYN, D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """Widen a routed result to the FP32 partial layout the TP reduction consumes."""
    t_dim = pl.tensor.dim(source, 0)
    for row in pl.spmd(t_dim, name_hint="ffn_partial_widen"):
        for d0 in pl.pipeline(0, D, 512, stage=2):
            output[row : row + 1, d0 : d0 + 512] = pl.cast(
                source[row : row + 1, d0 : d0 + 512], target_type=pl.FP32
            )
    return output


@pl.jit.inline
def _local_token_count(num_tokens: pl.Scalar[pl.INT32], tp_rank: pl.Scalar[pl.INT32]):
    """Map the replicated TP token prefix to this rank's compact local prefix."""
    # The runtime validates 1 <= num_tokens <= TOKENS and tp_rank is a valid TP lane,
    # so the integer ceiling is already in [0, TOKENS].  Keep the native INDEX scalar
    # type produced by the division for comparisons and row addressing.
    return (num_tokens + TP_SIZE - 1 - tp_rank) // TP_SIZE


@pl.jit.inline
def _compact_local_tokens(
    source: pl.Tensor[[T_DYN, D], pl.BF16],
    compact: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Pack this TP rank's interleaved owner rows into the local-token ABI prefix."""
    tokens = pl.tensor.dim(source, 0)
    local_tokens = _local_token_count(num_tokens, tp_rank)
    for row in pl.spmd(tokens, name_hint="prefill_moe_compact"):
        value = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
        if row < local_tokens:
            source_row = row * TP_SIZE + tp_rank
            value = pl.tile.load(source, [source_row, 0], [1, D])
        pl.tile.store(value, [row, 0], compact, shapes=[1, D])
    return compact


@pl.jit.inline
def _scatter_local_tokens(
    compact: pl.Tensor[[T_DYN, D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Scatter compact local MoE rows back into the original TP token positions."""
    tokens = pl.tensor.dim(output, 0)
    local_tokens = _local_token_count(num_tokens, tp_rank)
    with pl.spmd(tokens, name_hint="prefill_moe_scatter_zero") as zero_tid:
        row = pl.tile.get_block_idx()
        zero = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
        pl.tile.store(zero, [row, 0], output, shapes=[1, D])
    with pl.spmd(tokens, name_hint="prefill_moe_scatter", deps=[zero_tid]):
        row = pl.tile.get_block_idx()
        if row < local_tokens:
            dst_row = row * TP_SIZE + tp_rank
            value = pl.tile.load(compact, [row, 0], [1, D])
            pl.tile.store(value, [dst_row, 0], output, shapes=[1, D])
    return output


@pl.jit.inline(auto_scope=False)
def prefill_moe_sublayer(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    ffn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    ffn_owned: pl.Tensor[[T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Derive the FFN sublayer's mixes, collapse the stream, and run the EP MoE on it.

    Each token row has exactly one owner inside the TP group, so ``ffn_owned`` comes back
    holding only this rank's share, with every other row at zero; ``prefill_ffn_restore``
    puts the group back together. ``post_mix`` and ``residual_mix`` are this sublayer's own
    coefficients, handed on so the restore can expand the residual without re-deriving them.
    """
    moe_hc_pre(
        x_hc, pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        next_pre_mix, post_mix, residual_mix, ffn_input,
    )
    # The current EP ABI consumes one compact local-token prefix per rank.  The
    # prefill residual stream still stores rows in the replicated TP layout, so
    # compact this rank's interleaved owner rows before MoE and scatter them back.
    local_ffn_input = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
    local_ffn_normed = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
    local_ffn_owned = pl.create_tensor([TOKENS, D], dtype=pl.BF16)
    _compact_local_tokens(ffn_input, local_ffn_input, num_tokens, tp_rank)
    npu_rms_norm(local_ffn_input, ffn_norm_weight, local_ffn_normed)
    local_num_tokens = _local_token_count(num_tokens, tp_rank)
    with pl.scope():
        moe_core(
            local_ffn_normed, gate_weight, correction_bias,
            routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
            mxfp4_pair_lut, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale,
            shared_w3, shared_w3_scale, recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived, local_ffn_owned,
            local_num_tokens, ep_rank, moe_epoch,
        )
        _scatter_local_tokens(local_ffn_owned, ffn_owned, num_tokens, tp_rank)
    return ffn_owned


@pl.jit.inline(auto_scope=False)
def prefill_ffn_restore(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    ffn_owned: pl.Tensor[[T_DYN, D], pl.BF16],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    ffn_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.FP32],
    ffn_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    ffn_output: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ffn_epoch: pl.Scalar[pl.INT32],
):
    """Restore the replicated routed rows inside the TP group, then expand the residual.

    The MoE leaves every row it does not own at zero, so summing the group's partials is
    the all-gather that puts the attention layout back. At TP1 the group is this rank alone
    and the BF16 values survive the FP32 round trip unchanged. This is the layer's second
    collective, in a device entry of its own.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
    widen_to_fp32(ffn_owned, partial)
    prefill_tp_output_all_reduce(
        partial, ffn_window, ffn_arrived, ffn_output, group_base, tp_rank, num_tokens, ffn_epoch,
    )
    mhc_post(ffn_output, x_hc, post_mix, residual_mix, x_hc_out)
    return x_hc_out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FFN_HC_NAMES = ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm_weight")
# Gate, shared expert and FFN mHC weights are one model copy on every rank; only the routed
# experts are sharded, and each rank owns a different slice of the global expert ids.
MOE_GATE_NAMES = ("gate_weight", "correction_bias")
MOE_SHARED_NAMES = (
    "shared_w1", "shared_w1_scale", "shared_w2", "shared_w2_scale", "shared_w3", "shared_w3_scale",
)
MOE_RANK_NAMES = (
    "routed_w1", "routed_w1_scale", "routed_w2", "routed_w2_scale", "routed_w3", "routed_w3_scale",
    "mxfp4_pair_lut",
)
MOE_REPLICATED_NAMES = MOE_GATE_NAMES + MOE_SHARED_NAMES


# Drawing the routed shards costs minutes, and the spec shapes need the same draw the
# stacked fixtures use; cache each rank's fixture so it is built once.
@functools.lru_cache(maxsize=None)
def make_ffn_hc_inputs(seed):
    """FFN-sublayer mHC coefficients and the RMSNorm weight the MoE gate applies."""
    import math

    gen = torch.Generator().manual_seed(seed)
    return {
        "hc_ffn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM),
        "hc_ffn_scale": torch.randn(3, generator=gen),
        "hc_ffn_base": torch.randn(MIX_HC, generator=gen),
        "ffn_norm_weight": (torch.randn(D, generator=gen) * 0.1 + 1).to(torch.bfloat16),
    }


@functools.lru_cache(maxsize=None)
def make_moe_replicated_inputs(seed):
    """Gate and shared-expert weights at the deployment magnitudes, identical on every rank."""
    from models.deepseek_v4_1_flash.moe import _route_bias
    from models.deepseek_v4_1_flash.quantization import gen_mxfp8_weight_kn_v41

    gen = torch.Generator().manual_seed(seed)
    shared_std = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    shared_w1, shared_w1_scale = gen_mxfp8_weight_kn_v41(MOE_INTER, D, shared_std["w1"], chan_cv=0.50, seed=seed + 1)
    shared_w3, shared_w3_scale = gen_mxfp8_weight_kn_v41(MOE_INTER, D, shared_std["w3"], chan_cv=0.50, seed=seed + 2)
    shared_w2, shared_w2_scale = gen_mxfp8_weight_kn_v41(D, MOE_INTER, shared_std["w2"], chan_cv=0.33, seed=seed + 3)
    return {
        "gate_weight": torch.randn(N_EXPERTS, D, generator=gen) / D ** 0.5,
        # The bias offset dominates the O(1) score, which keeps top-k off the boundary where an
        # FP32 reduction-order difference could swap an expert and change the output.
        "correction_bias": _route_bias(),
        "shared_w1": shared_w1, "shared_w1_scale": shared_w1_scale,
        "shared_w2": shared_w2, "shared_w2_scale": shared_w2_scale,
        "shared_w3": shared_w3, "shared_w3_scale": shared_w3_scale,
    }


@functools.lru_cache(maxsize=None)
def make_moe_rank_inputs(rank):
    """This rank's routed-expert shard, drawn through the checkpoint conversion path."""
    w1, w1_scale, w3, w3_scale, w2, w2_scale = gen_routed_mxfp4_weights(
        N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=rank * N_LOCAL_EXPERTS * 3
    )
    return {
        "routed_w1": w1, "routed_w1_scale": w1_scale,
        "routed_w2": w2, "routed_w2_scale": w2_scale,
        "routed_w3": w3, "routed_w3_scale": w3_scale,
        "mxfp4_pair_lut": build_mxfp4_pair_lut(),
    }


def make_token_owners(tokens, rank):
    """Give every replicated token row exactly one owner inside this rank's TP group."""
    group_base = rank // TP_SIZE * TP_SIZE
    return (torch.arange(tokens, dtype=torch.int32) % TP_SIZE + group_base).to(torch.int32)


# ---------------------------------------------------------------------------
# Golden
# ---------------------------------------------------------------------------


def golden_moe_sublayer(tensors, x_hc_mid, attn_pre_mix, tokens):
    """Reference the FFN sublayer: mixes, collapse, EP MoE, TP sum, residual expansion."""
    world = x_hc_mid.shape[0]
    ffn_input = torch.empty(world, tokens, D, dtype=torch.bfloat16)
    mixes = {}
    for base in range(0, world, TP_SIZE):
        mid = x_hc_mid[base]
        pre, post, comb = hc_mixes(
            mid, tensors["hc_ffn_fn"][base], tensors["hc_ffn_scale"][base], tensors["hc_ffn_base"][base]
        )
        hidden = hc_pre(mid, attn_pre_mix[base]).to(torch.bfloat16)
        for rank in range(base, base + TP_SIZE):
            ffn_input[rank] = hidden
        mixes[base] = (pre, post, comb)
    ffn_owned = golden_moe_rows(tensors, ffn_input, tokens)
    outputs = {
        "ffn_input": ffn_input,
        "ffn_owned": ffn_owned,
        "ffn_output": torch.empty_like(ffn_owned),
        "next_pre_mix": torch.empty(world, tokens, HC_MULT, dtype=torch.float32),
        "ffn_post_mix": torch.empty(world, tokens, HC_MULT, dtype=torch.float32),
        "ffn_residual_mix": torch.empty(world, tokens, HC_MULT, HC_MULT, dtype=torch.float32),
        "x_hc_out": torch.empty_like(x_hc_mid),
    }
    for base in range(0, world, TP_SIZE):
        pre, post, comb = mixes[base]
        # Each row has one owner in the group and the rest are zero, so the group sum is the
        # complete routed result.
        group = ffn_owned[base : base + TP_SIZE].float().sum(dim=0).to(torch.bfloat16)
        expanded = hc_post(group, x_hc_mid[base], post, comb).float()
        for rank in range(base, base + TP_SIZE):
            outputs["ffn_output"][rank] = group
            outputs["next_pre_mix"][rank] = pre
            outputs["ffn_post_mix"][rank] = post
            outputs["ffn_residual_mix"][rank] = comb
            outputs["x_hc_out"][rank] = expanded
    return outputs


def golden_moe_rows(tensors, ffn_input, tokens):
    """Run the EP MoE on each rank's compact local-token prefix, then scatter owners back."""
    world = ffn_input.shape[0]
    counts = torch.zeros(world, dtype=torch.int32)
    local_input = torch.zeros_like(ffn_input)
    for rank in range(world):
        tp_rank = rank % TP_SIZE
        count = max(0, min(tokens, (tokens + TP_SIZE - 1 - tp_rank) // TP_SIZE))
        counts[rank] = count
        if count:
            source_rows = torch.arange(count, dtype=torch.long) * TP_SIZE + tp_rank
            local_input[rank, :count] = ffn_input[rank, source_rows]
    moe_tensors = {
        "x": local_input,
        "norm_weight": tensors["ffn_norm_weight"],
        "mxfp4_pair_lut": tensors["mxfp4_pair_lut"],
        "output": torch.zeros_like(local_input),
        "num_tokens": counts,
    }
    for name in MOE_REPLICATED_NAMES + MOE_RANK_NAMES:
        moe_tensors[name] = tensors[name]
    golden_moe_core(moe_tensors)
    scattered = torch.zeros_like(ffn_input)
    for rank in range(world):
        count = int(counts[rank])
        if count:
            destination_rows = torch.arange(count, dtype=torch.long) * TP_SIZE + (rank % TP_SIZE)
            scattered[rank, destination_rows] = moe_tensors["output"][rank, :count]
    return scattered


def make_layer_golden(mode):
    """Reference the whole layer: attention sublayer per TP group, then the MoE sublayer."""

    def golden(tensors):
        """Fill every exposed boundary and the attention state each rank publishes."""
        world = tensors["x_hc"].shape[0]
        tokens = tensors["x_hc"].shape[1]
        attn_input = torch.empty(world, tokens, D, dtype=torch.bfloat16)
        attn_output = torch.empty(world, tokens, D, dtype=torch.bfloat16)
        attn_pre_mix = torch.empty(world, tokens, HC_MULT, dtype=torch.float32)
        x_hc_mid = torch.empty(world, tokens, HC_MULT, D, dtype=torch.float32)
        for base in range(0, world, TP_SIZE):
            ranks = range(base, base + TP_SIZE)
            x_hc = tensors["x_hc"][base]
            pre, post, comb = hc_mixes(
                x_hc, tensors["hc_attn_fn"][base], tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base]
            )
            x = golden_attention_input(x_hc, tensors["pre_mix"][base], tensors["attn_norm_weight"][base])
            attention, results = reference_attention(mode, 1, tensors, x, ranks)
            for rank, result in zip(ranks, results):
                for name in ATTENTION_STATE[mode]:
                    tensors[name][rank].copy_(result[name])
            mid = hc_post(attention, x_hc, post, comb).float()
            for rank in ranks:
                attn_input[rank] = x
                attn_output[rank] = attention
                attn_pre_mix[rank] = pre
                x_hc_mid[rank] = mid
        outputs = {
            "attn_input": attn_input, "attn_output": attn_output,
            "attn_pre_mix": attn_pre_mix, "x_hc_mid": x_hc_mid,
            **golden_moe_sublayer(tensors, x_hc_mid, attn_pre_mix, tokens),
        }
        for name, value in outputs.items():
            tensors[name].copy_(value)

    return golden


# ---------------------------------------------------------------------------
# Stage-wise comparison
# ---------------------------------------------------------------------------


class StagedMoeReference:
    """The MoE reference re-run on the device's own ``ffn_input`` (teacher forcing).

    The MoE quantizes its input to MXFP8 per group of 32, so one BF16 ULP out of the mHC
    collapse can move a group scale and with it a routed row; the sublayer therefore keeps its
    standalone budget against this reference while ``ffn_input`` is checked against the golden.
    """

    def __init__(self, tokens):
        """Remember the active token count and cache the reference across comparators."""
        self.tokens = tokens
        self.outputs = None

    def __call__(self, inputs, actual_outputs):
        """Route the device's collapsed stream through the reference once per run."""
        if self.outputs is None:
            owned = golden_moe_rows(inputs, actual_outputs["ffn_input"], self.tokens)
            full = torch.empty_like(owned)
            for base in range(0, owned.shape[0], TP_SIZE):
                full[base : base + TP_SIZE] = owned[base : base + TP_SIZE].float().sum(dim=0).to(torch.bfloat16)
            self.outputs = {"ffn_owned": owned, "ffn_output": full}
        return self.outputs


def compare_ffn_input(actual, expected, *, inputs, actual_outputs, **kwargs):
    """Collapse the device's own residual stream with its own pre-mix and compare."""
    check = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    passed = True
    for base in range(0, actual.shape[0], TP_SIZE):
        replay = hc_pre(actual_outputs["x_hc_mid"][base], actual_outputs["attn_pre_mix"][base]).to(torch.bfloat16)
        _, worst_row = report_stage("ffn_input(replay)", actual[base], replay)
        valid, _ = check(actual[base], replay, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
        passed &= valid and worst_row <= ROW_BUDGET
        passed &= report_stage("ffn_input(end-to-end)", actual[base], expected[base])[0] <= 0.01
        passed &= all(torch.equal(actual[base], actual[rank]) for rank in range(base + 1, base + TP_SIZE))
    return passed, "mHC collapse replayed on the device's residual stream, per token row"


def compare_hc_mixes_output(name, source_name, weight_prefix, max_row_rel_l2, coefficient=0):
    """Replay hc_mixes on the device's own residual stream and check one coefficient."""
    check = ratio_allclose(atol=2.5e-5, rtol=5e-3)

    def compare(actual, expected, *, inputs, actual_outputs, **kwargs):
        """Compare the coefficient against the replay, and report the end-to-end difference."""
        passed = True
        for base in range(0, actual.shape[0], TP_SIZE):
            stream = inputs["x_hc"][base] if source_name is None else actual_outputs[source_name][base]
            replay = hc_mixes(
                stream, inputs[f"{weight_prefix}_fn"][base], inputs[f"{weight_prefix}_scale"][base],
                inputs[f"{weight_prefix}_base"][base],
            )[coefficient]
            _, worst_row = report_stage(f"{name}(replay)", actual[base], replay)
            valid, _ = check(actual[base], replay, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
            passed &= valid and worst_row <= max_row_rel_l2
            passed &= all(torch.equal(actual[base], actual[rank]) for rank in range(base + 1, base + TP_SIZE))
        return passed, f"hc_mixes replay within its budget, every token row <= {max_row_rel_l2:.3g} rel L2"

    return compare


def compare_expanded_stream(name, sublayer_name, source_name, weight_prefix, end_to_end_bound):
    """Replay hc_post on the device's own sublayer result and check the new residual stream."""
    check = ratio_allclose(atol=1e-4, rtol=1.0 / 128)

    def compare(actual, expected, *, inputs, actual_outputs, **kwargs):
        """Hold the replay to hc_post's budget; the end-to-end bound is the sanity check."""
        passed = True
        for base in range(0, actual.shape[0], TP_SIZE):
            stream = inputs["x_hc"][base] if source_name is None else actual_outputs[source_name][base]
            _, post, comb = hc_mixes(
                stream, inputs[f"{weight_prefix}_fn"][base], inputs[f"{weight_prefix}_scale"][base],
                inputs[f"{weight_prefix}_base"][base],
            )
            replay = hc_post(actual_outputs[sublayer_name][base], stream, post, comb).float()
            _, worst_row = report_stage(f"{name}(replay)", actual[base], replay)
            valid, _ = check(actual[base], replay, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
            passed &= valid and worst_row <= ROW_BUDGET
            passed &= report_stage(f"{name}(end-to-end)", actual[base], expected[base])[0] <= end_to_end_bound
            passed &= bool(torch.isfinite(actual[base]).all())
            passed &= all(torch.equal(actual[base], actual[rank]) for rank in range(base + 1, base + TP_SIZE))
        return passed, f"hc_post replay within hc_post's budget per token row; end-to-end rel L2 <= {end_to_end_bound:.3g}"

    return compare


def routing_flips(inputs, rank, actual_input, expected_input):
    """Count token rows whose top-k selection moves when the gate sees the device's input."""
    norm_weight = inputs["ffn_norm_weight"][rank]
    gate_weight = inputs["gate_weight"][rank]
    correction_bias = inputs["correction_bias"][rank]
    _, actual_indices = gate(rms_norm(actual_input, norm_weight), gate_weight, correction_bias)
    _, expected_indices = gate(rms_norm(expected_input, norm_weight), gate_weight, correction_bias)
    flipped = (actual_indices.sort(dim=-1).values != expected_indices.sort(dim=-1).values).any(dim=-1)
    return int(flipped.sum().item()), flipped.numel()


def compare_ffn_owned(staged, tokens):
    """Hold this rank's share of the routed rows to the MoE budget; the rest must be zero."""
    check = ratio_reldiff(diff_thd=3e-3, pct_thd=0.02)

    def compare(actual, expected, *, inputs, actual_outputs, **kwargs):
        """Check every rank on its own rows, then that it left the other rows alone."""
        forced = staged(inputs, actual_outputs)["ffn_owned"]
        passed = bool(torch.isfinite(actual.float()).all())
        for rank in range(actual.shape[0]):
            owners = inputs["token_owners"][rank].to(torch.int64)
            mine = (owners == rank)[:tokens]
            report_stage(f"ffn_owned(rank {rank}, owned rows)", actual[rank, :tokens][mine],
                         forced[rank, :tokens][mine])
            others = actual[rank, :tokens][~mine]
            intruded = int(others.abs().amax(-1).gt(0).sum()) if others.numel() else 0
            print(
                f"[PRECISION] ffn_owned rank {rank} owns {int(mine.sum())}/{tokens} rows, "
                f"wrote {intruded} of the {int((~mine).sum())} rows it does not own"
            )
            valid, _ = check(
                actual[rank, :tokens][mine], forced[rank, :tokens][mine],
                inputs=inputs, actual_outputs=actual_outputs, **kwargs
            )
            passed &= valid and intruded == 0
        return passed, "owned rows within the MoE budget of the forced reference; other rows zero"

    return compare


def compare_ffn_output(staged, tokens):
    """Hold the routed result to the MoE budget against the teacher-forced reference.

    The end-to-end difference carries the chain budget instead of the stage budget, because
    the input quantization amplifies (see ``CHAIN_BUDGET``). The printed flip count separates
    that from the other thing a perturbed input could do here: move the top-k selection, which
    would change a row by a whole expert's contribution. It has measured zero so far.
    """
    check = ratio_reldiff(diff_thd=3e-3, pct_thd=0.02)

    def compare(actual, expected, *, inputs, actual_outputs, expected_outputs, **kwargs):
        """Compare active rows on every rank; TP replicas must agree after the group sum."""
        forced = staged(inputs, actual_outputs)["ffn_output"]
        passed = bool(torch.isfinite(actual.float()).all())
        for base in range(0, actual.shape[0], TP_SIZE):
            flipped, rows = routing_flips(
                inputs, base, actual_outputs["ffn_input"][base], expected_outputs["ffn_input"][base]
            )
            print(f"[PRECISION] ffn_output routing rows_with_moved_topk={flipped}/{rows}")
            owners = inputs["token_owners"][base].to(torch.int64)
            lanes = " ".join(
                f"owner{int(owner)}:{actual[base][owners == owner].abs().max().item():.4g}"
                for owner in owners[:tokens].unique()
            )
            print(f"[PRECISION] ffn_output max_abs by owning rank: {lanes}")
            end_to_end, _ = report_stage(
                "ffn_output(end-to-end)", actual[base, :tokens], expected[base, :tokens]
            )
            report_stage("ffn_output(forced)", actual[base, :tokens], forced[base, :tokens])
            valid, _ = check(
                actual[base, :tokens], forced[base, :tokens],
                inputs=inputs, actual_outputs=actual_outputs, expected_outputs=expected_outputs, **kwargs
            )
            passed &= valid and end_to_end <= CHAIN_BUDGET
            passed &= all(torch.equal(actual[base], actual[rank]) for rank in range(base + 1, base + TP_SIZE))
        return passed, "routed rows within the MoE budget of the forced reference; replicas identical"

    return compare


def make_layer_compare(mode, tokens, initial_state):
    """Stage-wise comparators: every sublayer boundary forced on the device's own input."""
    attention = StagedAttentionReference(mode, 1, initial_state)
    moe = StagedMoeReference(tokens)
    compare = {
        "attn_input": compare_group_leaders("attn_input", ratio_allclose(atol=1e-4, rtol=1.0 / 128), ROW_BUDGET),
        "attn_pre_mix": compare_hc_mixes_output("attn_pre_mix", None, "hc_attn", 5e-3),
        "attn_output": attention.compare("attn_output", compare_replicated(compare_output)),
        "x_hc_mid": compare_expanded_stream("x_hc_mid", "attn_output", None, "hc_attn", 0.01),
        "ffn_input": compare_ffn_input,
        "next_pre_mix": compare_hc_mixes_output("next_pre_mix", "x_hc_mid", "hc_ffn", 5e-3),
        "ffn_post_mix": compare_hc_mixes_output("ffn_post_mix", "x_hc_mid", "hc_ffn", 5e-3, coefficient=1),
        "ffn_residual_mix": compare_hc_mixes_output(
            "ffn_residual_mix", "x_hc_mid", "hc_ffn", 5e-3, coefficient=2
        ),
        "ffn_owned": compare_ffn_owned(moe, tokens),
        "ffn_output": compare_ffn_output(moe, tokens),
        # This stream is the MoE result expanded, so it carries the MoE's end-to-end budget
        # rather than the attention sublayer's.
        "x_hc_out": compare_expanded_stream("x_hc_out", "ffn_output", "x_hc_mid", "hc_ffn", CHAIN_BUDGET),
    }
    for name in REUSE_MUTABLE_NAMES:
        compare[name] = attention.compare(name, compare_per_rank(compare_cache(name), "window_slots"))
    return compare


# ---------------------------------------------------------------------------
# Device program
# ---------------------------------------------------------------------------


def make_layer_program(capacity, world_size):
    """Build the L3 group entry that runs one complete prefill layer on every rank.

    The layer is three device entries launched back to back on each rank: the attention
    sublayer, the FFN sublayer's mixes and EP MoE, and the TP restore that puts the routed
    rows back together, so each entry carries exactly one collective.

    Each entry runs its sublayer once per launch. The EP transport has no consumed window: it
    leaves the lifetime of its receive and result windows to the caller, and a second round
    inside the same launch lets a fast rank overwrite windows a slower peer is still reading.
    Repeated calls - a benchmark - are separate dispatches, which advance the epoch.

    All three carry the packed token extent statically: the MoE gate and the EP transport
    freeze it at import anyway, and a static extent unifies with the ``T_DYN`` parameters of
    every sublayer operator, including across the boundary where one sublayer's result feeds
    the next - a result carries the dim expression of the operator that wrote it, which no
    longer matches the callee's type variable.
    """

    @pl.jit
    def attention_rank(
        x_hc: pl.Tensor[[TOKENS, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[TOKENS, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[TOKENS, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[TOKENS, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[TOKENS], pl.INT64],
        window_indices: pl.Tensor[[TOKENS, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[TOKENS, C.INDEX_TOPK], pl.INT32],
        attn_input: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
        attn_pre_mix: pl.Out[pl.Tensor[[TOKENS, HC_MULT], pl.FP32]],
        x_hc_mid: pl.Out[pl.Tensor[[TOKENS, HC_MULT, D], pl.FP32]],
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Bind the runtime shapes and run the attention sublayer."""
        window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
        prefill_c2a_reuse(
            x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            compressed_indices, output_window, output_arrived, attn_input, attn_output,
            attn_pre_mix, x_hc_mid, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, num_tokens,
            attention_epoch,
        )
        return x_hc_mid, attn_pre_mix, attn_output, attn_input, window_cache, window_cache_scale

    @pl.jit
    def moe_rank(
        x_hc_mid: pl.Tensor[[TOKENS, HC_MULT, D], pl.FP32],
        attn_pre_mix: pl.Tensor[[TOKENS, HC_MULT], pl.FP32],
        hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[3], pl.FP32],
        hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
        ffn_norm_weight: pl.Tensor[[D], pl.BF16],
        gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
        correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
        routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
        routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
        routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN],
        routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
        routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
        shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
        shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
        shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
        shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
        shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        ffn_input: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
        ffn_owned: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[TOKENS, HC_MULT], pl.FP32]],
        ffn_post_mix: pl.Out[pl.Tensor[[TOKENS, HC_MULT], pl.FP32]],
        ffn_residual_mix: pl.Out[pl.Tensor[[TOKENS, HC_MULT, HC_MULT], pl.FP32]],
        recv_meta: pld.DistributedTensor[[EP_SIZE, N_LOCAL_EXPERTS], pl.INT32],
        recv_x: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
        recv_scale: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], pl.UINT8],
        recv_weights: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], pl.FP32],
        recv_routes: pld.DistributedTensor[[N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], pl.INT32],
        arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        data_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        routed_output: pld.DistributedTensor[[C.ROUTE_T_DYN, D], pl.BF16],
        combine_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ):
        """Run the FFN sublayer's mixes, collapse and EP MoE."""
        prefill_moe_sublayer(
            x_hc_mid, attn_pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            ffn_norm_weight, gate_weight, correction_bias,
            routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
            mxfp4_pair_lut, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale,
            shared_w3, shared_w3_scale, recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            ffn_input, ffn_owned, next_pre_mix, ffn_post_mix, ffn_residual_mix,
            num_tokens, rank, rank % TP_SIZE, moe_epoch,
        )
        return ffn_owned, next_pre_mix, ffn_post_mix, ffn_residual_mix, ffn_input

    @pl.jit
    def restore_rank(
        x_hc_mid: pl.Tensor[[TOKENS, HC_MULT, D], pl.FP32],
        ffn_owned: pl.Tensor[[TOKENS, D], pl.BF16],
        ffn_post_mix: pl.Tensor[[TOKENS, HC_MULT], pl.FP32],
        ffn_residual_mix: pl.Tensor[[TOKENS, HC_MULT, HC_MULT], pl.FP32],
        ffn_output: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
        x_hc_out: pl.Out[pl.Tensor[[TOKENS, HC_MULT, D], pl.FP32]],
        ffn_window: pld.DistributedTensor[[capacity, D], pl.FP32],
        ffn_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        ffn_epoch: pl.Scalar[pl.INT32],
    ):
        """Put the TP group's routed rows back together and expand the residual."""
        prefill_ffn_restore(
            x_hc_mid, ffn_owned, ffn_post_mix, ffn_residual_mix, ffn_window, ffn_arrived,
            ffn_output, x_hc_out, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, num_tokens,
            ffn_epoch,
        )
        return x_hc_out, ffn_output

    @pl.jit.host
    def layer_group(
        x_hc: pl.Tensor[[world_size, TOKENS, HC_MULT, D], pl.FP32],
        pre_mix: pl.Tensor[[world_size, TOKENS, HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[world_size, MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[world_size, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[world_size, MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[world_size, D], pl.BF16],
        wq_a: pl.Tensor[[world_size, D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[world_size, D // 32, Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[world_size, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[world_size, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[world_size, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[world_size, D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[world_size, D // 32, HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[world_size, HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[world_size, LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[world_size, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[world_size, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[world_size, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[world_size, TOKENS, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[world_size, TOKENS, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[world_size, TOKENS], pl.INT64],
        window_indices: pl.Tensor[[world_size, TOKENS, 128], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[world_size, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[world_size, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.Tensor[[world_size, CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8],
        compressed_cache_scale: pl.Tensor[[world_size, CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.FP8E4M3FN],
        compressed_indices: pl.Tensor[[world_size, TOKENS, C.INDEX_TOPK], pl.INT32],
        hc_ffn_fn: pl.Tensor[[world_size, MIX_HC, HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[world_size, 3], pl.FP32],
        hc_ffn_base: pl.Tensor[[world_size, MIX_HC], pl.FP32],
        ffn_norm_weight: pl.Tensor[[world_size, D], pl.BF16],
        gate_weight: pl.Tensor[[world_size, N_EXPERTS, D], pl.FP32],
        correction_bias: pl.Tensor[[world_size, N_EXPERTS], pl.FP32],
        routed_w1: pl.Tensor[[world_size, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
        routed_w1_scale: pl.Tensor[[world_size, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
        routed_w2: pl.Tensor[[world_size, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
        routed_w2_scale: pl.Tensor[[world_size, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
        routed_w3: pl.Tensor[[world_size, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
        routed_w3_scale: pl.Tensor[[world_size, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
        mxfp4_pair_lut: pl.Tensor[[world_size, 2, 256], pl.INT16],
        shared_w1: pl.Tensor[[world_size, D, MOE_INTER], pl.FP8E4M3FN],
        shared_w1_scale: pl.Tensor[[world_size, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
        shared_w2: pl.Tensor[[world_size, MOE_INTER, D], pl.FP8E4M3FN],
        shared_w2_scale: pl.Tensor[[world_size, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
        shared_w3: pl.Tensor[[world_size, D, MOE_INTER], pl.FP8E4M3FN],
        shared_w3_scale: pl.Tensor[[world_size, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
        token_owners: pl.Tensor[[world_size, TOKENS], pl.INT32],
        attn_input: pl.Out[pl.Tensor[[world_size, TOKENS, D], pl.BF16]],
        attn_output: pl.Out[pl.Tensor[[world_size, TOKENS, D], pl.BF16]],
        attn_pre_mix: pl.Out[pl.Tensor[[world_size, TOKENS, HC_MULT], pl.FP32]],
        x_hc_mid: pl.Out[pl.Tensor[[world_size, TOKENS, HC_MULT, D], pl.FP32]],
        ffn_input: pl.Out[pl.Tensor[[world_size, TOKENS, D], pl.BF16]],
        ffn_owned: pl.Out[pl.Tensor[[world_size, TOKENS, D], pl.BF16]],
        ffn_output: pl.Out[pl.Tensor[[world_size, TOKENS, D], pl.BF16]],
        next_pre_mix: pl.Out[pl.Tensor[[world_size, TOKENS, HC_MULT], pl.FP32]],
        ffn_post_mix: pl.Out[pl.Tensor[[world_size, TOKENS, HC_MULT], pl.FP32]],
        ffn_residual_mix: pl.Out[pl.Tensor[[world_size, TOKENS, HC_MULT, HC_MULT], pl.FP32]],
        x_hc_out: pl.Out[pl.Tensor[[world_size, TOKENS, HC_MULT, D], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
        layer_epoch: pl.Scalar[pl.INT32],
    ):
        """Allocate the TP and EP communication windows and launch both sublayers per device."""
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
        attention_buffer = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        attention_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        ffn_buffer = pld.alloc_window_buffer([capacity, D], dtype=pl.FP32)
        ffn_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
        recv_weights_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes_buffer = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        data_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        routed_output_buffer = pld.alloc_window_buffer([ROUTE_ROWS, D], dtype=pl.BF16)
        combine_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            attention_data = pld.window(attention_buffer, [capacity, D], dtype=pl.FP32)
            attention_signal = pld.window(attention_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            # Each rank consumes packed MX_B_NN scale rows; a bare slice is ND.
            wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
            wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
            wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
            wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
            attention_rank(
                x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
                attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank],
                wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank],
                wo_a[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank],
                window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank],
                compressed_cache[rank], compressed_cache_scale[rank], compressed_indices[rank],
                attn_input[rank], attn_output[rank], attn_pre_mix[rank], x_hc_mid[rank],
                attention_data, attention_signal, rank, num_tokens, layer_epoch, device=rank,
            )
        for rank in pl.range(pld.world_size()):
            recv_meta = pld.window(recv_meta_buffer, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
            recv_x = pld.window(recv_x_buffer, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
            recv_scale = pld.window(recv_scale_buffer, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
            recv_weights = pld.window(recv_weights_buffer, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
            recv_routes = pld.window(recv_routes_buffer, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
            arrived = pld.window(arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            data_arrived = pld.window(data_arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            routed_output = pld.window(routed_output_buffer, [ROUTE_ROWS, D], dtype=pl.BF16)
            combine_arrived = pld.window(combine_arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            routed_w1_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w1_scale[rank]
            routed_w2_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w2_scale[rank]
            routed_w3_scale_r: pl.Tensor[
                [N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN
            ] = routed_w3_scale[rank]
            shared_w1_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[rank]
            shared_w2_scale_r: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[rank]
            shared_w3_scale_r: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[rank]
            moe_rank(
                x_hc_mid[rank], attn_pre_mix[rank], hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
                ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank],
                routed_w1[rank], routed_w1_scale_r, routed_w2[rank], routed_w2_scale_r,
                routed_w3[rank], routed_w3_scale_r, mxfp4_pair_lut[rank],
                shared_w1[rank], shared_w1_scale_r, shared_w2[rank], shared_w2_scale_r,
                shared_w3[rank], shared_w3_scale_r,
                ffn_input[rank], ffn_owned[rank], next_pre_mix[rank], ffn_post_mix[rank],
                ffn_residual_mix[rank],
                recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
                arrived, data_arrived, routed_output, combine_arrived,
                rank, num_tokens, layer_epoch, device=rank,
            )
        for rank in pl.range(pld.world_size()):
            ffn_data = pld.window(ffn_buffer, [capacity, D], dtype=pl.FP32)
            ffn_signal = pld.window(ffn_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            restore_rank(
                x_hc_mid[rank], ffn_owned[rank], ffn_post_mix[rank], ffn_residual_mix[rank],
                ffn_output[rank], x_hc_out[rank], ffn_data, ffn_signal,
                rank, num_tokens, layer_epoch, device=rank,
            )

    return layer_group


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def build_specs(args, mode, initial_state):
    """Stacked specs whose per-rank fixtures are drawn once, on first use."""
    world_size = EP_SIZE
    names, mutable_names, sharded_names, _ = MODES[mode]
    attention_names = tuple(name for name in names if name != "x")
    ranks = {}

    def initialize(name):
        """Draw every rank's fixture on the first spec that needs it, then stack one tensor."""
        if not ranks:
            replicated = {**make_moe_replicated_inputs(args.seed + 5003), **make_ffn_hc_inputs(args.seed + 3001)}
            for rank in range(world_size):
                ranks[rank] = make_attention_inputs(
                    mode, args.tokens, args.requests, args.seed + rank, args.case
                )
                ranks[rank].update(make_moe_rank_inputs(rank))
                ranks[rank].update(replicated)
                ranks[rank]["token_owners"] = make_token_owners(args.tokens, rank)
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
    shapes.update(make_ffn_hc_inputs(args.seed + 3001))
    shapes.update(make_moe_replicated_inputs(args.seed + 5003))
    shapes.update(make_moe_rank_inputs(0))
    shapes["token_owners"] = make_token_owners(args.tokens, 0)
    # The spec order is the device entry's parameter order.
    input_names = (
        HC_INPUT_NAMES + attention_names + FFN_HC_NAMES
        + MOE_GATE_NAMES + MOE_RANK_NAMES + MOE_SHARED_NAMES + ("token_owners",)
    )
    specs = [
        TensorSpec(
            name,
            [world_size, *shapes[name].shape],
            shapes[name].dtype,
            init_value=(lambda name=name: initialize(name)),
            resident="stacked",
        )
        for name in input_names
    ]
    specs += [
        TensorSpec("attn_input", [world_size, args.tokens, D], torch.bfloat16, resident="stacked"),
        TensorSpec("attn_output", [world_size, args.tokens, D], torch.bfloat16, resident="stacked"),
        TensorSpec("attn_pre_mix", [world_size, args.tokens, HC_MULT], torch.float32, resident="stacked"),
        TensorSpec("x_hc_mid", [world_size, args.tokens, HC_MULT, D], torch.float32, resident="stacked"),
        TensorSpec("ffn_input", [world_size, args.tokens, D], torch.bfloat16, resident="stacked"),
        TensorSpec("ffn_owned", [world_size, args.tokens, D], torch.bfloat16, resident="stacked"),
        TensorSpec("ffn_output", [world_size, args.tokens, D], torch.bfloat16, resident="stacked"),
        TensorSpec("next_pre_mix", [world_size, args.tokens, HC_MULT], torch.float32, resident="stacked"),
        TensorSpec("ffn_post_mix", [world_size, args.tokens, HC_MULT], torch.float32, resident="stacked"),
        TensorSpec(
            "ffn_residual_mix", [world_size, args.tokens, HC_MULT, HC_MULT], torch.float32, resident="stacked"
        ),
        TensorSpec("x_hc_out", [world_size, args.tokens, HC_MULT, D], torch.float32, resident="stacked"),
        ScalarSpec("num_tokens", torch.int32, args.tokens, compile_runtime=True),
        ScalarSpec(
            "layer_epoch",
            torch.int32,
            1,
            compile_runtime=True,
            # Every dispatch runs the layer once; a benchmark advances the epoch per dispatch.
            benchmark_step=1 if args.bench else None,
        ),
    ]
    return specs


def run_prefill_layer(make_program, mode, argv=None):
    """Validate one complete prefill layer on A5; ``make_program`` builds the L3 entry."""
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(
        description=f"DeepSeek V4.1 prefill layer ({mode} attention + EP MoE): A5 precision"
    )
    parser.add_argument("-p", "--platform", default="a5", choices=["a5"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through EP-1")
    # The A5 job runs one case per TP/DP combination declared here. TP4 needs eight cards and
    # has not been run; widen the choice once it has.
    parser.add_argument("--tp", type=int, default=LAYER_TP, choices=[1, 2])
    parser.add_argument("--dp", type=int, default=LAYER_DP, choices=[2])
    parser.add_argument("--tokens", type=int, default=LAYER_TOKENS)
    # config reads --ep from the command line (the import above supplies TP x DP when it is
    # absent). Declare it so argparse consumes it as itself instead of prefix-matching it to
    # another long option.
    parser.add_argument("--ep", type=int, default=EP_SIZE, help="set from --tp x --dp; read by config")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--case", default="mixed", choices=["mixed", "long", "masked", "zero"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)

    # config and the MoE shapes were frozen from the same command line before argparse ran.
    if (args.tp, args.dp, args.ep, args.tokens) != (TP_SIZE, C.DP_SIZE, EP_SIZE, C.MOE_TOKENS):
        parser.error("--tp/--dp/--ep/--tokens are read before argparse; pass them once on the command line")
    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    devices = list(range(EP_SIZE))
    if args.device:
        devices = [int(value) for value in args.device.split(",")]
    if len(devices) != EP_SIZE or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"--device must name {EP_SIZE} distinct non-negative device IDs")
    if not 1 <= args.tokens <= C.PREFILL_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {C.PREFILL_MAX_TOKENS}]")
    if not 1 <= args.requests <= min(C.MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"--requests must be in [1, {min(C.MAX_BATCH_PER_DP, args.tokens)}]")
    torch.set_num_threads(8)

    print(
        f"[LAYER] mode={mode} tokens={args.tokens} requests={args.requests} TP={TP_SIZE} DP={C.DP_SIZE} "
        f"EP={EP_SIZE} experts/rank={N_LOCAL_EXPERTS} case={args.case} seed={args.seed} devices={devices}"
    )
    initial_state = {}
    result = run(
        fn=make_program(C.PREFILL_MAX_TOKENS, EP_SIZE),
        specs=build_specs(args, mode, initial_state),
        golden_fn=make_layer_golden(mode),
        compile_only=args.compile_only,
        config=dict(
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            ring_heap=LAYER_RING_HEAP,
        ),
        compare_fn=make_layer_compare(mode, args.tokens, initial_state),
    )
    print(f"[LAYER] work_dir={result.work_dir}")
    if args.compile_only and result.passed:
        print("[LAYER] Compilation passed; device accuracy was NOT validated.")
    return result


def validate(argv=None):
    """Validate the complete packed prefill layer against its golden reference on A5."""
    return run_prefill_layer(make_layer_program, "reuse", argv=argv)


def main():
    """Validate one complete prefill layer, C2A Reuse attention followed by the EP MoE."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "moe_hc_pre",
    "prefill_ffn_restore",
    "prefill_moe_sublayer",
    "run_prefill_layer",
    "validate",
    "widen_to_fp32",
]


if "pytest" in sys.modules:
    import pytest

    @pytest.mark.parametrize("tp,dp", [(1, 2), (2, 2)])
    def test_precision(tp, dp, a5_args):
        """Validate TP1/DP2 and TP2/DP2 complete prefill layer cases on A5."""
        result = validate(a5_args(tp=tp, dp=dp))
        assert result.passed, result.error


# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"
if __name__ == _SCRIPT_ENTRY_POINT:
    main()
