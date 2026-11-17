# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Complete-model prefill: one device program per rank, driven by one host program.

This is the A5 single-program shape (``models/deepseek_v4_1_flash/decode_fwd.py``
``l2_decode_fwd`` / ``l3_decode_fwd``): :func:`prefill_fwd` is one
``@pl.jit.host`` entry that allocates the EP16 exchange and TP output windows
once and calls the per-rank device program ``l2_prefill_fwd`` inside
``for rank in pl.range(pld.world_size())``.  The rank program runs the
TP-sharded embedding, :func:`prefill_encoder` (layers 0-19),
:func:`prefill_decoder` (layers 20-39) and the final HC collapse + BF16 head.
Both encoder and decoder are ``@pl.jit.inline`` bodies with a ``pl.range`` layer
loop and runtime per-mode dispatch, exactly like A5 ``decode_decoder``.

Per layer: optional Engram at layers 1/14 -> the mode chain (``prefill_swa`` / ``prefill_c2a_full`` /
``prefill_c2a_reuse`` / ``prefill_c1a_full`` / ``prefill_c1a_reindex`` /
``prefill_c1a_reuse``; each owns attention HC pre, RMSNorm, the paged attention
chain, the TP output reduce and HC post) ->
``moe_exchange.moe_ffn`` (FFN HC pre, RMSNorm, gate, inline static-tile EP16
``moe_ep``, HC post).  ``moe_ffn`` receives ``rows`` = the compiled token
bound (2/16/128) as its constexpr, so the exchange never sees pad rows, and
``epoch = layer + 1``: the exchange windows are allocated once by the host entry
and reused across all forty layers, so every call must advance the monotonic
window counters.  The TP output window is shared by the embedding and every
layer for the same reason; its epoch is 1 for the embedding and ``layer + 2``
afterwards, which keeps the folded ``2 * epoch`` counter strictly monotonic.

Attention families from ``config.FLASH.layer_config``:

* layers 0-1: ``swa``, no cache owner (layer 1 also runs Engram).
* layers 2, 8 (and 14 with Engram): ``c2a_full``.
* layers 3-7, 9-13, 15-19: ``c2a_reuse`` of the preceding c2a source slot.
* layer 20: ``c1a_full``, owner of the compressed KV consumed by layers 21-39.
* layers 24, 28, 32, 36: ``c1a_reindex`` (fresh selection on the source index basis).
* layers 21-23, 25-27, 29-31, 33-35, 37-39: ``c1a_reuse``.

Weight arguments carry a leading layer axis ``[N_LAYERS, ...]`` inside the rank
program and are sliced at runtime (``wq_a[layer]``, A5 pattern); the host entry
passes ``param[rank]``, so the host tensors are stacked ``[EP_SIZE, N_LAYERS,
...]``.  Cache-owner weights keep their source axis ``[3, ...]`` (C2A) or
``[4, ...]`` (C1A reindex).

The routed/shared expert shards are the one exception: they keep only the rank
axis (``[EP_SIZE, N_LOCAL_EXPERTS, ...]``) because the bounded fixture
rank-broadcasts a single shard and a faithful ``[EP_SIZE, N_LAYERS, ...]``
expert stack is 543 GB of host memory (40 layers x 16 ranks x 24 experts x
5120 x 2304 x 3 tensors).  ``prepare_fwd_weights`` already materializes the real
per-layer shards; wiring the layer axis into the entry is a memory-bound
follow-up recorded in the experiment log, not a change of the compute contract.

Validation is staged per layer, unchanged: the rank program writes
``layer_x_hc`` / ``layer_pre_mix`` after every layer, and the composed golden is
re-run from the device's own previous-layer values, so each layer's residual is
the model's own arithmetic floor rather than a forty-layer accumulation.  All
comparator thresholds are the ones carried over from the validated entry.

Sublayer order is the frozen paged contract (``hc_pre -> norm -> operator ->
hc_post`` with delayed pre-mix, Engram at layers 1/14).
"""

import argparse
import os

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from utils import compressed_source_positions

from config import (
    B_DYN,
    BLOCK_SIZE,
    CANDIDATE_POOL_BLOCKS,
    CMP_BLOCKS_DYN,
    EP_SIZE,
    FLASH as M,
    INDEX_BLOCKS_DYN,
    LOCAL_HEADS,
    N_LOCAL_EXPERTS,
    ORI_BLOCKS_DYN,
    POOL_BLOCKS_DYN,
    PREFILL_MAX_TOKENS,
    STATE_BLOCKS_DYN,
    STATE_CAPACITY,
    TABLE_DYN,
    T_DYN,
    TP_SIZE,
    WINDOW_KEYS,
)
from embedding import embedding_lookup
from engram import engram_from_rows, fixture_rotation
from golden import ScalarSpec
from hc import hc_head
from layer_reference import (
    MODE_C1A_FULL,
    MODE_C1A_REINDEX,
    MODE_C1A_REUSE,
    MODE_C2A_FULL,
    MODE_C2A_REUSE,
    MODE_SWA,
    make_layer_golden,
    reference_active_counts,
    active_output_rows,
)
from lm_head import lm_head
from moe_exchange import COUNT_PAD, moe_ffn
from prefill_c1a_full import prefill_c1a_full
from prefill_c1a_reindex import prefill_c1a_reindex
from prefill_c1a_reuse import prefill_c1a_reuse
from prefill_c2a_full import prefill_c2a_full
from prefill_c2a_reuse import prefill_c2a_reuse
from prefill_swa import prefill_swa
from tp_reduce import tp_output_reduce
from utils import pack_nz


# model config
N_LAYERS = M.num_hidden_layers
D = M.hidden_size
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
LOCAL_Q = LOCAL_HEADS * HEAD_DIM
O_GROUPS = M.o_groups // (M.num_attention_heads // LOCAL_HEADS)
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
O_LORA = M.o_lora_rank
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
MIX_PAD = 32
INDEX_TOPK = M.index_topk
IDX_DIM = M.index_head_dim
IDX_HEADS = M.index_n_heads
LOCAL_INDEX_DIM = IDX_HEADS * IDX_DIM
TOPK = M.num_experts_per_tok
N_EXPERTS = M.n_routed_experts
INTER = M.moe_intermediate_size

# Generation ABI switch (driver process only).  With DEEPSEEK_PER_LAYER_EXPERTS=1
# the routed-expert arguments carry the real 40-layer axis; the bounded entries
# use a size-1 layer axis (contiguous, cheap) and clamp the layer index, because
# the harness requires contiguous host tensors and a contiguous 40-layer expert
# stack is ~181 GiB per tensor (INTERFACE.md).
PER_LAYER_EXPERTS = os.environ.get("DEEPSEEK_PER_LAYER_EXPERTS") == "1"
EXPERT_LAYERS = N_LAYERS if PER_LAYER_EXPERTS else 1
_LOCAL_W1 = [N_LOCAL_EXPERTS, *(D, INTER)]
_RANK_W1 = [EXPERT_LAYERS, N_LOCAL_EXPERTS, *(D, INTER)]
_HOST_W1 = [EP_SIZE, EXPERT_LAYERS, N_LOCAL_EXPERTS, *(D, INTER)]
_LOCAL_W1_SCALE = [N_LOCAL_EXPERTS, *(INTER,)]
_RANK_W1_SCALE = [EXPERT_LAYERS, N_LOCAL_EXPERTS, *(INTER,)]
_HOST_W1_SCALE = [EP_SIZE, EXPERT_LAYERS, N_LOCAL_EXPERTS, *(INTER,)]
_LOCAL_W3 = [N_LOCAL_EXPERTS, *(D, INTER)]
_RANK_W3 = [EXPERT_LAYERS, N_LOCAL_EXPERTS, *(D, INTER)]
_HOST_W3 = [EP_SIZE, EXPERT_LAYERS, N_LOCAL_EXPERTS, *(D, INTER)]
_LOCAL_W3_SCALE = [N_LOCAL_EXPERTS, *(INTER,)]
_RANK_W3_SCALE = [EXPERT_LAYERS, N_LOCAL_EXPERTS, *(INTER,)]
_HOST_W3_SCALE = [EP_SIZE, EXPERT_LAYERS, N_LOCAL_EXPERTS, *(INTER,)]
_LOCAL_W2 = [N_LOCAL_EXPERTS, *(INTER, D)]
_RANK_W2 = [EXPERT_LAYERS, N_LOCAL_EXPERTS, *(INTER, D)]
_HOST_W2 = [EP_SIZE, EXPERT_LAYERS, N_LOCAL_EXPERTS, *(INTER, D)]
_LOCAL_W2_SCALE = [N_LOCAL_EXPERTS, *(D,)]
_RANK_W2_SCALE = [EXPERT_LAYERS, N_LOCAL_EXPERTS, *(D,)]
_HOST_W2_SCALE = [EP_SIZE, EXPERT_LAYERS, N_LOCAL_EXPERTS, *(D,)]

ENGRAM_K = (M.engram_max_ngram_size - 1) * M.engram_n_heads * M.engram_head_dim
ENGRAM_OUT = (HC_MULT + 1) * D
STATE_WIDTH = 2 * HEAD_DIM
LOCAL_VOCAB = M.vocab_size // TP_SIZE
N_ENGRAM = len(M.engram_layer_ids)
DEFAULT_T = 16
SUPPORTED_TILES = (2, 16, 128)

C2A_SOURCES = tuple(source for source in M.kv_source_layer_ids if M.compress_ratios[source] == 2)
C1A_SOURCES = tuple(source for source in M.kv_source_layer_ids if M.compress_ratios[source] == 1)
REINDEX_LAYERS = tuple(
    layer for layer in M.index_source_layer_ids if M.layer_config(layer).mode == "c1a_reindex"
)
assert C2A_SOURCES == (2, 8, 14)
assert C1A_SOURCES == (20,)
assert REINDEX_LAYERS == (24, 28, 32, 36)

MODE_CODE = {
    "swa": MODE_SWA,
    "c2a_full": MODE_C2A_FULL,
    "c2a_reuse": MODE_C2A_REUSE,
    "c1a_full": MODE_C1A_FULL,
    "c1a_reuse": MODE_C1A_REUSE,
    "c1a_reindex": MODE_C1A_REINDEX,
}
C2A_MODES = (MODE_SWA, MODE_C2A_FULL, MODE_C2A_REUSE)
N_C1A_SLOTS = 1 + len(REINDEX_LAYERS)

# Layer-scoped weight names carried with a leading ``[N_LAYERS, ...]`` axis.
DENSE_LAYER_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_w", "sink", "wo_a", "wo_b",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm_w", "gate_weight", "gate_bias",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
)
# Routed expert shards carry the rank axis only (see the module docstring).
EXPERT_NAMES = ("w1", "w1_scale", "w3", "w3_scale", "w2", "w2_scale")


def _layer_family(layer):
    """Return ``(mode, ratio, has_engram)`` for one text layer."""
    config = M.layer_config(layer)
    return MODE_CODE[config.mode], (config.compression_ratio or 1), (layer in M.engram_layer_ids)


def _c2a_slot(layer):
    """Compressed/index/state source slot consumed by a C2A layer."""
    source = M.layer_config(layer).kv_source_layer_id
    return C2A_SOURCES.index(source) if source in C2A_SOURCES else 0


def _c1a_slot(layer):
    """Selection slot of a C1A layer: 0 for layer 20, ``1 + (layer-24)//4`` after."""
    return (layer - C1A_SOURCES[0]) // 4


# ---------------------------------------------------------------------------
# Per-rank device program pieces (inlined into ``l2_prefill_fwd``)
# ---------------------------------------------------------------------------


@pl.jit.inline(auto_scope=False)
def _prefill_embed(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
    initial_pre: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """TP-sharded embedding lookup, TP-group reduce, HC seeding (one rank).

    The checkpoint stores ``embed.weight`` TP-sharded: each rank holds the
    contiguous global rows ``[tp_rank * LOCAL_VOCAB, (tp_rank + 1) * LOCAL_VOCAB)``.
    A rank looks up only the tokens it owns (masked, zero for non-owners), and
    ``tp_output_reduce`` sums the four TP peers' windows into the full embedding
    row.  The reduced rows are then replicated across the HC streams and the
    delayed pre-mix is seeded with the identity pre-mix.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    local_ids = pl.create_tensor([tokens], dtype=pl.INT32)
    owners = pl.create_tensor([tokens], dtype=pl.INT32)
    partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
    # One writer: INT32 ids are 4 bytes, so concurrent blocks would share a
    # 64-byte line and lose each other's scalar stores.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embed_shard_ids"):
        for token in pl.range(tokens):
            local = pl.cast(pl.read(input_ids, [token]), pl.INT32) - tp_rank * LOCAL_VOCAB
            clamped = pl.min(pl.max(local, 0), LOCAL_VOCAB - 1)
            # owner = 1 exactly when ``local`` lies inside this rank's shard;
            # (local - clamped)^2 is 0 in range and >= 1 on either side.
            gap = local - clamped
            owner = 1 - pl.min(pl.mul(gap, gap), 1)
            pl.write(local_ids, [token], pl.cast(pl.mul(clamped, owner), pl.INT32))
            pl.write(owners, [token], pl.cast(owner, pl.INT32))
    embedding_lookup(local_ids, embed_weight, partial, width=D)
    for token in pl.spmd(tokens, name_hint="embed_shard_mask"):
        if pl.read(owners, [token]) == 0:
            pl.store(pl.tile.full([1, D], dtype=pl.BF16, value=0.0), [token, 0], partial)
    hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    tp_output_reduce(
        partial, output_window, output_arrived, hidden,
        group_base, tp_rank, tokens, epoch,
    )
    flat = pl.reshape(x_hc, [tokens * HC_MULT, D])
    for token in pl.spmd(tokens, name_hint="embed_to_hc"):
        row = pl.tile.load(hidden, [token, 0], [1, D])
        value = pl.cast(row, target_type=pl.FP32)
        for lane in pl.range(HC_MULT):
            pl.store(value, [token * HC_MULT + lane, 0], flat)
    # One writer: HC_MULT FP32 values per token (16 B) share 64-byte cache
    # lines, so concurrent blocks would lose each other's scalar stores.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="seed_pre_mix"):
        for token in pl.range(tokens):
            for lane in pl.range(HC_MULT):
                pl.write(pre_mix, [token, lane], pl.read(initial_pre, [token, lane]))


@pl.jit.inline(auto_scope=False)
def _prefill_layer_snapshot(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    layer_x_hc: pl.Tensor,
    layer_pre_mix: pl.Tensor,
    layer: pl.Scalar[pl.INT32],
):
    """Record the layer output into the per-layer validation snapshots.

    ``layer_x_hc``/``layer_pre_mix`` are the flat ``[N_LAYERS * tokens * HC_MULT,
    D]`` / ``[N_LAYERS * tokens, HC_MULT]`` views of the per-rank snapshot
    tensors; the staged golden consumes them to isolate each layer's error.
    """
    tokens = pl.tensor.dim(pre_mix, 0)
    x_flat = pl.reshape(x_hc, [tokens * HC_MULT, D])
    with pl.spmd(tokens * HC_MULT, name_hint="layer_snapshot_x"):
        row = pl.tile.get_block_idx()
        value = pl.tile.load(x_flat, [row, 0], [1, D])
        pl.store(value, [layer * tokens * HC_MULT + row, 0], layer_x_hc)
    # One writer: HC_MULT FP32 values per row share a 64-byte cache line.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="layer_snapshot_pre_mix"):
        for row in pl.range(tokens):
            for lane in pl.range(HC_MULT):
                pl.write(
                    layer_pre_mix, [layer * tokens + row, lane],
                    pl.read(pre_mix, [row, lane]),
                )


@pl.jit.inline(auto_scope=False)
def _prefill_encoder_layer(
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc_mid: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    attn_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_cos_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
    index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    topk_values: pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    engram_lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[ENGRAM_OUT, ENGRAM_K], pl.BF16],
    engram_qk: pl.Tensor[[4, D], pl.FP32],
    engram_rotation: pl.Tensor[[32, 32], pl.FP32],
    engram_enabled: pl.Scalar[pl.INT32],
    hc_ffn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    w1: pl.Tensor[[N_LOCAL_EXPERTS, D, INTER], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w3: pl.Tensor[[N_LOCAL_EXPERTS, D, INTER], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w2: pl.Tensor[[N_LOCAL_EXPERTS, INTER, D], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[INTER], pl.FP32],
    shared_w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[INTER], pl.FP32],
    shared_w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_x: pld.DistributedTensor,
    recv_aux: pld.DistributedTensor,
    recv_meta: pld.DistributedTensor,
    meta_ready: pld.DistributedTensor,
    data_ready: pld.DistributedTensor,
    return_slots: pld.DistributedTensor,
    return_ready: pld.DistributedTensor,
    consumed: pld.DistributedTensor,
    return_consumed: pld.DistributedTensor,
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    layer: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rows: pl.constexpr,
):
    """One encoder layer: Engram input update, attention chain and FFN half.

    The mode dispatch is the A5 ``decode_decoder`` form: a runtime ``if/elif`` on
    the layer id, so every variant is compiled into the loop body once.  The
    attention chain writes ``x_hc_mid`` and the delayed FFN pre-mix; Engram
    gates the four residual streams at layers 1/14; ``moe_ffn`` consumes
    ``x_hc_mid`` and writes the next ``x_hc``/``pre_mix``.  The exchange epoch is
    ``layer + 1`` (windows are shared by all forty layers); the TP output epoch
    is ``layer + 2`` because the embedding consumed epoch 1 of the same window.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    attention_x = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    if (layer == 1 or layer == 14) and engram_enabled != 0:
        engram_slot = layer // 14
        _engram_ready = engram_from_rows(engram_lookup[engram_slot], engram_wkv[engram_slot], engram_qk[engram_slot], engram_rotation, x_hc, attention_x)
    else:
        source_hc = pl.reshape(x_hc, [tokens * HC_MULT, D])
        target_hc = pl.reshape(attention_x, [tokens * HC_MULT, D])
        for hc_row in pl.spmd(tokens * HC_MULT, name_hint="attention_input_copy"):
            row_value = source_hc[hc_row : hc_row + 1, 0:D]
            target_hc[hc_row : hc_row + 1, 0:D] = row_value
    if layer < 2:
        _swa_out, _swa_attn_in = prefill_swa(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos_swa, rope_sin_swa, window_slots, window_indices,
            window_cache, output_window, output_arrived, hidden, attn_input, attn_out,
            x_hc_mid, attn_pre_mix, group_base, tp_rank, num_tokens, layer + 1,
        )
    elif (layer - 2) % 6 == 0:
        _c2af_out, _c2af_topk, _c2af_vals, _c2af_pool, _c2af_counts, _c2af_attn_in = prefill_c2a_full(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, compressed_cache, request_ids, compressed_lens, index_cache,
            index_block_table, compressed_cos, compressed_sin, compressed_slots,
            position_ids, state_block_table, state_cache, compressor_wkv,
            compressor_wgate, compressor_norm, index_wq_b, index_wq_b_scale,
            index_weights_proj, index_wk, index_norm_w, topk_indices, topk_values,
            pool_block_ids, pool_block_counts, output_window, output_arrived, hidden,
            attn_input, attn_out, x_hc_mid, attn_pre_mix, group_base, tp_rank,
            num_tokens, layer + 1,
        )
    else:
        _c2ar_out, _c2ar_attn_in = prefill_c2a_reuse(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, compressed_cache, topk_indices, output_window,
            output_arrived, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
            group_base, tp_rank, num_tokens, layer + 1,
        )
    _ffn_out, _ffn_done = moe_ffn(
        x_hc_mid, attn_pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
        gate_weight, gate_bias,
        w1, w1_scale, w3, w3_scale, w2, w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        recv_x, recv_aux, recv_meta, meta_ready, data_ready,
        return_slots, return_ready, consumed, return_consumed,
        x_hc, pre_mix, my_rank, layer + 1, num_tokens, rows=rows,
    )


@pl.jit.inline(auto_scope=False)
def _prefill_decoder_layer(
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc_mid: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    attn_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_cos_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressor_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
    index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16],
    reindex_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    reindex_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    reindex_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    topk_values: pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32],
    pool_block_ids: pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    engram_lookup: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[ENGRAM_OUT, ENGRAM_K], pl.BF16],
    engram_qk: pl.Tensor[[4, D], pl.FP32],
    engram_rotation: pl.Tensor[[32, 32], pl.FP32],
    engram_enabled: pl.Scalar[pl.INT32],
    hc_ffn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    w1: pl.Tensor[[N_LOCAL_EXPERTS, D, INTER], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w3: pl.Tensor[[N_LOCAL_EXPERTS, D, INTER], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, INTER], pl.FP32],
    w2: pl.Tensor[[N_LOCAL_EXPERTS, INTER, D], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[INTER], pl.FP32],
    shared_w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[INTER], pl.FP32],
    shared_w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_x: pld.DistributedTensor,
    recv_aux: pld.DistributedTensor,
    recv_meta: pld.DistributedTensor,
    meta_ready: pld.DistributedTensor,
    data_ready: pld.DistributedTensor,
    return_slots: pld.DistributedTensor,
    return_ready: pld.DistributedTensor,
    consumed: pld.DistributedTensor,
    return_consumed: pld.DistributedTensor,
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    layer: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rows: pl.constexpr,
):
    """One decoder layer: Engram input update, attention chain and FFN half.

    Layer 20 owns the C1A compressed/index caches; layers 24/28/32/36 re-run the
    indexer selection on the source basis (``reindex_wq_b*``); every other layer
    reuses the selection of the nearest source (``topk_indices`` of its slot).
    The epoch contract is the encoder's: exchange ``layer + 1``, TP ``layer + 2``.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    attention_x = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    if (layer == 1 or layer == 14) and engram_enabled != 0:
        engram_slot = layer // 14
        _engram_ready = engram_from_rows(engram_lookup[engram_slot], engram_wkv[engram_slot], engram_qk[engram_slot], engram_rotation, x_hc, attention_x)
    else:
        source_hc = pl.reshape(x_hc, [tokens * HC_MULT, D])
        target_hc = pl.reshape(attention_x, [tokens * HC_MULT, D])
        for hc_row in pl.spmd(tokens * HC_MULT, name_hint="attention_input_copy"):
            row_value = source_hc[hc_row : hc_row + 1, 0:D]
            target_hc[hc_row : hc_row + 1, 0:D] = row_value
    if layer == 20:
        _c1af_out, _c1af_topk, _c1af_vals, _c1af_pool, _c1af_counts, _c1af_attn_in = prefill_c1a_full(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, compressed_cache, request_ids, compressed_lens, index_cache,
            index_block_table, compressed_cos, compressed_sin, compressed_slots,
            compressor_wkv, compressor_norm, index_wq_b, index_wq_b_scale,
            index_weights_proj, index_wk, index_norm_w, topk_indices, topk_values,
            pool_block_ids, pool_block_counts, output_window, output_arrived, hidden,
            attn_input, attn_out, x_hc_mid, attn_pre_mix, group_base, tp_rank,
            num_tokens, layer + 1,
        )
    elif (layer - 20) % 4 == 0:
        _c1ar_out, _c1ar_topk, _c1ar_vals, _c1ar_pool, _c1ar_counts, _c1ar_attn_in = prefill_c1a_reindex(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, compressed_cache, request_ids, compressed_lens, index_cache,
            index_block_table, reindex_wq_b, reindex_wq_b_scale, reindex_weights_proj,
            topk_indices, topk_values, pool_block_ids, pool_block_counts,
            output_window, output_arrived, hidden, attn_input, attn_out, x_hc_mid,
            attn_pre_mix, group_base, tp_rank, num_tokens, layer + 1,
        )
    else:
        _c1au_out, _c1au_attn_in = prefill_c1a_reuse(
            attention_x, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices,
            window_cache, compressed_cache, topk_indices, output_window,
            output_arrived, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
            group_base, tp_rank, num_tokens, layer + 1,
        )
    _ffn_out, _ffn_done = moe_ffn(
        x_hc_mid, attn_pre_mix, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w,
        gate_weight, gate_bias,
        w1, w1_scale, w3, w3_scale, w2, w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        recv_x, recv_aux, recv_meta, meta_ready, data_ready,
        return_slots, return_ready, consumed, return_consumed,
        x_hc, pre_mix, my_rank, layer + 1, num_tokens, rows=rows,
    )


@pl.jit.inline(auto_scope=False)
def prefill_encoder(
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    layer_x_hc: pl.Tensor,
    layer_pre_mix: pl.Tensor,
    hc_attn_fn: pl.Tensor[[N_LAYERS, MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_LAYERS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.INT8],
    wq_a_scale: pl.Tensor[[N_LAYERS, Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[N_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_Q], pl.INT8],
    wq_b_scale: pl.Tensor[[N_LAYERS, LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.INT8],
    wkv_scale: pl.Tensor[[N_LAYERS, HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[N_LAYERS, HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[N_LAYERS, LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[N_LAYERS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_LAYERS, D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_cos_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[N_LAYERS, ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[3, CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    index_cache: pl.Tensor[[3, INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    state_cache: pl.Tensor[[3, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    compressor_wkv: pl.Tensor[[3, HEAD_DIM, D], pl.BF16],
    compressor_wgate: pl.Tensor[[3, HEAD_DIM, D], pl.BF16],
    compressor_norm: pl.Tensor[[3, HEAD_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[3, Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
    index_wq_b_scale: pl.Tensor[[3, IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[3, D, IDX_HEADS], pl.BF16],
    index_wk: pl.Tensor[[3, IDX_DIM, HEAD_DIM], pl.BF16],
    index_norm_w: pl.Tensor[[3, IDX_DIM], pl.BF16],
    topk_indices: pl.Tensor[[3, T_DYN, INDEX_TOPK], pl.INT32],
    topk_values: pl.Tensor[[3, T_DYN, INDEX_TOPK], pl.FP32],
    pool_block_ids: pl.Tensor[[3, T_DYN, POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[3, T_DYN], pl.INT32],
    engram_lookup: pl.Tensor[[N_ENGRAM, T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[N_ENGRAM, ENGRAM_OUT, ENGRAM_K], pl.BF16],
    engram_qk: pl.Tensor[[N_ENGRAM, 4, D], pl.FP32],
    engram_rotation: pl.Tensor[[32, 32], pl.FP32],
    engram_enabled: pl.Scalar[pl.INT32],
    hc_ffn_fn: pl.Tensor[[N_LAYERS, MIX_PAD, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[N_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[N_LAYERS, N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_LAYERS, N_EXPERTS], pl.FP32],
    shared_w1: pl.Tensor[[N_LAYERS, D, INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[N_LAYERS, INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_LAYERS, D, INTER], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[N_LAYERS, INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_LAYERS, INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[N_LAYERS, D], pl.FP32],
    w1: pl.Tensor[_RANK_W1, pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[_RANK_W1_SCALE, pl.FP32],
    w3: pl.Tensor[_RANK_W3, pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[_RANK_W3_SCALE, pl.FP32],
    w2: pl.Tensor[_RANK_W2, pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[_RANK_W2_SCALE, pl.FP32],
    recv_x: pld.DistributedTensor,
    recv_aux: pld.DistributedTensor,
    recv_meta: pld.DistributedTensor,
    meta_ready: pld.DistributedTensor,
    data_ready: pld.DistributedTensor,
    return_slots: pld.DistributedTensor,
    return_ready: pld.DistributedTensor,
    consumed: pld.DistributedTensor,
    return_consumed: pld.DistributedTensor,
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rows: pl.constexpr,
):
    """Text layers 0-19 as one dynamic layer loop with runtime mode dispatch.

    Two conventions are required by the front end at this inline depth:
    the layer step's inline results are bound to throwaway names (a bare call
    trips the "called for its side effects only" verifier), and every scratch
    tensor is created with the ``pl.tensor.dim`` token expression rather than
    the ``T_DYN`` symbol (the symbol form breaks memref type symmetry inside
    ``hc_coefficients``).
    """
    tokens = pl.tensor.dim(x_hc, 0)
    hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attn_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    x_hc_mid = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    attn_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    for layer in pl.range(0, 20):
        with pl.scope():
            c2a_slot = pl.max(layer - 2, pl.const(0, pl.INT32)) // 6
            layer_hc_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32] = hc_attn_fn[layer]
            layer_hc_scale: pl.Tensor[[3], pl.FP32] = hc_attn_scale[layer]
            layer_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = hc_attn_base[layer]
            layer_norm_w: pl.Tensor[[D], pl.BF16] = attn_norm_w[layer]
            layer_wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ] = wq_a[layer]
            layer_wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32] = wq_a_scale[layer]
            layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = q_norm_w[layer]
            layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ] = wq_b[layer]
            layer_wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32] = wq_b_scale[layer]
            layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ] = wkv[layer]
            layer_wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32] = wkv_scale[layer]
            layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = kv_norm_w[layer]
            layer_sink: pl.Tensor[[LOCAL_HEADS], pl.FP32] = sink[layer]
            layer_wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = wo_a[layer]
            layer_wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16] = wo_b[layer]
            layer_window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = (
                window_cache[layer]
            )
            layer_compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = (
                compressed_cache[c2a_slot]
            )
            layer_index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16] = (
                index_cache[c2a_slot]
            )
            layer_state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32] = (
                state_cache[c2a_slot]
            )
            layer_topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32] = topk_indices[c2a_slot]
            layer_topk_values: pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32] = topk_values[c2a_slot]
            layer_pool_ids: pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32] = pool_block_ids[c2a_slot]
            layer_pool_counts: pl.Tensor[[T_DYN], pl.INT32] = pool_block_counts[c2a_slot]
            layer_index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ] = (
                index_wq_b[c2a_slot]
            )
            layer_index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32] = (
                index_wq_b_scale[c2a_slot]
            )
            layer_index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16] = (
                index_weights_proj[c2a_slot]
            )
            layer_index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16] = index_wk[c2a_slot]
            layer_index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16] = index_norm_w[c2a_slot]
            layer_hc_ffn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32] = hc_ffn_fn[layer]
            layer_hc_ffn_scale: pl.Tensor[[3], pl.FP32] = hc_ffn_scale[layer]
            layer_hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = hc_ffn_base[layer]
            layer_ffn_norm: pl.Tensor[[D], pl.BF16] = ffn_norm_w[layer]
            layer_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32] = gate_weight[layer]
            layer_gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32] = gate_bias[layer]
            layer_shared_w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ] = shared_w1[layer]
            layer_shared_w1_scale: pl.Tensor[[INTER], pl.FP32] = shared_w1_scale[layer]
            layer_shared_w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ] = shared_w3[layer]
            layer_shared_w3_scale: pl.Tensor[[INTER], pl.FP32] = shared_w3_scale[layer]
            layer_shared_w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ] = shared_w2[layer]
            layer_shared_w2_scale: pl.Tensor[[D], pl.FP32] = shared_w2_scale[layer]
            _prefill_encoder_layer(
                x_hc, pre_mix, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
                layer_hc_fn, layer_hc_scale, layer_hc_base, layer_norm_w,
                layer_wq_a, layer_wq_a_scale, layer_q_norm, layer_wq_b, layer_wq_b_scale,
                layer_wkv, layer_wkv_scale, layer_kv_norm, layer_sink, layer_wo_a, layer_wo_b,
                rope_cos, rope_sin, rope_cos_swa, rope_sin_swa, window_slots, window_indices, layer_window_cache,
                layer_compressed_cache, request_ids, compressed_lens, layer_index_cache,
                index_block_table, compressed_cos, compressed_sin, compressed_slots,
                position_ids, state_block_table, layer_state_cache,
                compressor_wkv[c2a_slot], compressor_wgate[c2a_slot], compressor_norm[c2a_slot],
                layer_index_wq_b, layer_index_wq_b_scale, layer_index_weights_proj,
                layer_index_wk, layer_index_norm_w,
                layer_topk_indices, layer_topk_values, layer_pool_ids, layer_pool_counts,
                engram_lookup, engram_wkv, engram_qk, engram_rotation, engram_enabled,
                layer_hc_ffn_fn, layer_hc_ffn_scale, layer_hc_ffn_base, layer_ffn_norm,
                layer_gate_weight, layer_gate_bias,
                w1[pl.min(layer, EXPERT_LAYERS - 1)],
                w1_scale[pl.min(layer, EXPERT_LAYERS - 1)],
                w3[pl.min(layer, EXPERT_LAYERS - 1)],
                w3_scale[pl.min(layer, EXPERT_LAYERS - 1)],
                w2[pl.min(layer, EXPERT_LAYERS - 1)],
                w2_scale[pl.min(layer, EXPERT_LAYERS - 1)],
                layer_shared_w1, layer_shared_w1_scale,
                layer_shared_w3, layer_shared_w3_scale,
                layer_shared_w2, layer_shared_w2_scale,
                recv_x, recv_aux, recv_meta, meta_ready, data_ready,
                return_slots, return_ready, consumed, return_consumed,
                output_window, output_arrived,
                layer, my_rank, group_base, tp_rank, num_tokens, rows=rows,
            )
            _prefill_layer_snapshot(
                x_hc, pre_mix, layer_x_hc, layer_pre_mix, layer,
            )


@pl.jit.inline(auto_scope=False)
def prefill_decoder(
    x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    layer_x_hc: pl.Tensor,
    layer_pre_mix: pl.Tensor,
    hc_attn_fn: pl.Tensor[[N_LAYERS, MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_LAYERS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.INT8],
    wq_a_scale: pl.Tensor[[N_LAYERS, Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[N_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_Q], pl.INT8],
    wq_b_scale: pl.Tensor[[N_LAYERS, LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.INT8],
    wkv_scale: pl.Tensor[[N_LAYERS, HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[N_LAYERS, HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[N_LAYERS, LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[N_LAYERS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_LAYERS, D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_cos_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[N_LAYERS, ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressor_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
    index_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
    index_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
    index_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
    index_norm_w: pl.Tensor[[IDX_DIM], pl.BF16],
    reindex_wq_b: pl.Tensor[[N_C1A_SLOTS - 1, Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
    reindex_wq_b_scale: pl.Tensor[[N_C1A_SLOTS - 1, IDX_HEADS * IDX_DIM], pl.FP32],
    reindex_weights_proj: pl.Tensor[[N_C1A_SLOTS - 1, D, IDX_HEADS], pl.BF16],
    topk_indices: pl.Tensor[[N_C1A_SLOTS, T_DYN, INDEX_TOPK], pl.INT32],
    topk_values: pl.Tensor[[N_C1A_SLOTS, T_DYN, INDEX_TOPK], pl.FP32],
    pool_block_ids: pl.Tensor[[N_C1A_SLOTS, T_DYN, POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[N_C1A_SLOTS, T_DYN], pl.INT32],
    engram_lookup: pl.Tensor[[N_ENGRAM, T_DYN, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[N_ENGRAM, ENGRAM_OUT, ENGRAM_K], pl.BF16],
    engram_qk: pl.Tensor[[N_ENGRAM, 4, D], pl.FP32],
    engram_rotation: pl.Tensor[[32, 32], pl.FP32],
    engram_enabled: pl.Scalar[pl.INT32],
    hc_ffn_fn: pl.Tensor[[N_LAYERS, MIX_PAD, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[N_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[N_LAYERS, N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_LAYERS, N_EXPERTS], pl.FP32],
    shared_w1: pl.Tensor[[N_LAYERS, D, INTER], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[N_LAYERS, INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_LAYERS, D, INTER], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[N_LAYERS, INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_LAYERS, INTER, D], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[N_LAYERS, D], pl.FP32],
    w1: pl.Tensor[_RANK_W1, pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[_RANK_W1_SCALE, pl.FP32],
    w3: pl.Tensor[_RANK_W3, pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[_RANK_W3_SCALE, pl.FP32],
    w2: pl.Tensor[_RANK_W2, pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[_RANK_W2_SCALE, pl.FP32],
    recv_x: pld.DistributedTensor,
    recv_aux: pld.DistributedTensor,
    recv_meta: pld.DistributedTensor,
    meta_ready: pld.DistributedTensor,
    data_ready: pld.DistributedTensor,
    return_slots: pld.DistributedTensor,
    return_ready: pld.DistributedTensor,
    consumed: pld.DistributedTensor,
    return_consumed: pld.DistributedTensor,
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rows: pl.constexpr,
):
    """Text layers 20-39 as one dynamic layer loop with runtime mode dispatch.

    Same front-end conventions as :func:`prefill_encoder`.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attn_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    x_hc_mid = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    attn_pre_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    for layer in pl.range(20, 40):
        with pl.scope():
            c1a_slot = (layer - 20) // 4
            reindex_slot = pl.max((layer - 20) // 4 - 1, pl.const(0, pl.INT32))
            layer_hc_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32] = hc_attn_fn[layer]
            layer_hc_scale: pl.Tensor[[3], pl.FP32] = hc_attn_scale[layer]
            layer_hc_base: pl.Tensor[[MIX_HC], pl.FP32] = hc_attn_base[layer]
            layer_norm_w: pl.Tensor[[D], pl.BF16] = attn_norm_w[layer]
            layer_wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ] = wq_a[layer]
            layer_wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32] = wq_a_scale[layer]
            layer_q_norm: pl.Tensor[[Q_LORA], pl.BF16] = q_norm_w[layer]
            layer_wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ] = wq_b[layer]
            layer_wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32] = wq_b_scale[layer]
            layer_wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ] = wkv[layer]
            layer_wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32] = wkv_scale[layer]
            layer_kv_norm: pl.Tensor[[HEAD_DIM], pl.BF16] = kv_norm_w[layer]
            layer_sink: pl.Tensor[[LOCAL_HEADS], pl.FP32] = sink[layer]
            layer_wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = wo_a[layer]
            layer_wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16] = wo_b[layer]
            layer_window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = (
                window_cache[layer]
            )
            layer_topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32] = topk_indices[c1a_slot]
            layer_topk_values: pl.Tensor[[T_DYN, INDEX_TOPK], pl.FP32] = topk_values[c1a_slot]
            layer_pool_ids: pl.Tensor[[T_DYN, POOL_BLOCKS_DYN], pl.INT32] = pool_block_ids[0]
            layer_pool_counts: pl.Tensor[[T_DYN], pl.INT32] = pool_block_counts[0]
            layer_reindex_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ] = (
                reindex_wq_b[reindex_slot]
            )
            layer_reindex_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32] = (
                reindex_wq_b_scale[reindex_slot]
            )
            layer_reindex_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16] = (
                reindex_weights_proj[reindex_slot]
            )
            layer_hc_ffn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32] = hc_ffn_fn[layer]
            layer_hc_ffn_scale: pl.Tensor[[3], pl.FP32] = hc_ffn_scale[layer]
            layer_hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32] = hc_ffn_base[layer]
            layer_ffn_norm: pl.Tensor[[D], pl.BF16] = ffn_norm_w[layer]
            layer_gate_weight: pl.Tensor[[N_EXPERTS, D], pl.FP32] = gate_weight[layer]
            layer_gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32] = gate_bias[layer]
            layer_shared_w1: pl.Tensor[[D, INTER], pl.INT8, pl.NZ] = shared_w1[layer]
            layer_shared_w1_scale: pl.Tensor[[INTER], pl.FP32] = shared_w1_scale[layer]
            layer_shared_w3: pl.Tensor[[D, INTER], pl.INT8, pl.NZ] = shared_w3[layer]
            layer_shared_w3_scale: pl.Tensor[[INTER], pl.FP32] = shared_w3_scale[layer]
            layer_shared_w2: pl.Tensor[[INTER, D], pl.INT8, pl.NZ] = shared_w2[layer]
            layer_shared_w2_scale: pl.Tensor[[D], pl.FP32] = shared_w2_scale[layer]
            _prefill_decoder_layer(
                x_hc, pre_mix, hidden, attn_input, attn_out, x_hc_mid, attn_pre_mix,
                layer_hc_fn, layer_hc_scale, layer_hc_base, layer_norm_w,
                layer_wq_a, layer_wq_a_scale, layer_q_norm, layer_wq_b, layer_wq_b_scale,
                layer_wkv, layer_wkv_scale, layer_kv_norm, layer_sink, layer_wo_a, layer_wo_b,
                rope_cos, rope_sin, rope_cos_swa, rope_sin_swa, window_slots, window_indices, layer_window_cache,
                compressed_cache, request_ids, compressed_lens, index_cache,
                index_block_table, compressed_cos, compressed_sin, compressed_slots,
                position_ids, compressor_wkv, compressor_norm,
                index_wq_b, index_wq_b_scale, index_weights_proj, index_wk, index_norm_w,
                layer_reindex_wq_b, layer_reindex_wq_b_scale, layer_reindex_weights_proj,
                layer_topk_indices, layer_topk_values, layer_pool_ids, layer_pool_counts,
                engram_lookup, engram_wkv, engram_qk, engram_rotation, engram_enabled,
                layer_hc_ffn_fn, layer_hc_ffn_scale, layer_hc_ffn_base, layer_ffn_norm,
                layer_gate_weight, layer_gate_bias,
                w1[pl.min(layer, EXPERT_LAYERS - 1)],
                w1_scale[pl.min(layer, EXPERT_LAYERS - 1)],
                w3[pl.min(layer, EXPERT_LAYERS - 1)],
                w3_scale[pl.min(layer, EXPERT_LAYERS - 1)],
                w2[pl.min(layer, EXPERT_LAYERS - 1)],
                w2_scale[pl.min(layer, EXPERT_LAYERS - 1)],
                layer_shared_w1, layer_shared_w1_scale,
                layer_shared_w3, layer_shared_w3_scale,
                layer_shared_w2, layer_shared_w2_scale,
                recv_x, recv_aux, recv_meta, meta_ready, data_ready,
                return_slots, return_ready, consumed, return_consumed,
                output_window, output_arrived,
                layer, my_rank, group_base, tp_rank, num_tokens, rows=rows,
            )
            _prefill_layer_snapshot(
                x_hc, pre_mix, layer_x_hc, layer_pre_mix, layer,
            )


# ---------------------------------------------------------------------------
# Program entries: one host program, one device program per rank
# ---------------------------------------------------------------------------


def _build_prefill_entry(tokens):
    """Build the phase entry for one compiled token bound.

    ``tokens`` is the compile-time bound of the token axis; it fixes the EP16
    exchange capacity (``rows``) and the exchange window sizes, so the bound has
    to be a Python constant at trace time.  One entry object is built per tile
    (2/16/128) and cached by the driver.

    ``prefill_fwd`` (the ``@pl.jit.host`` program) allocates the EP16 exchange
    windows and the TP output window once and dispatches ``l2_prefill_fwd`` on
    every rank, slicing ``param[rank]`` (the A5 ``l3`` pattern);
    ``l2_prefill_fwd`` (the ``@pl.jit`` device program) owns the embedding, the
    two layer halves and the head for one rank.
    """

    @pl.jit
    def l2_prefill_fwd(
        input_ids: pl.Tensor[[T_DYN], pl.INT64],
        embed_weight: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
        initial_pre: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
        final_norm: pl.Tensor[[D], pl.BF16],
        lm_head_weight: pl.Tensor[[LOCAL_VOCAB, D], pl.BF16],
        position_ids: pl.Tensor[[T_DYN], pl.INT32],
        request_ids: pl.Tensor[[T_DYN], pl.INT32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
        index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
        state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
        rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
        rope_cos_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
        rope_sin_swa: pl.Tensor[[T_DYN, 32], pl.FP32],
        c2a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
        c2a_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        c2a_compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
        c2a_compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
        c1a_compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
        c1a_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
        c1a_compressed_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
        c1a_compressed_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
        hc_attn_fn: pl.Tensor[[N_LAYERS, MIX_PAD, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[N_LAYERS, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[N_LAYERS, MIX_HC], pl.FP32],
        attn_norm_w: pl.Tensor[[N_LAYERS, D], pl.BF16],
        wq_a: pl.Tensor[[N_LAYERS, D, Q_LORA], pl.INT8],
        wq_a_scale: pl.Tensor[[N_LAYERS, Q_LORA], pl.FP32],
        q_norm_w: pl.Tensor[[N_LAYERS, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[N_LAYERS, Q_LORA, LOCAL_Q], pl.INT8],
        wq_b_scale: pl.Tensor[[N_LAYERS, LOCAL_Q], pl.FP32],
        wkv: pl.Tensor[[N_LAYERS, D, HEAD_DIM], pl.INT8],
        wkv_scale: pl.Tensor[[N_LAYERS, HEAD_DIM], pl.FP32],
        kv_norm_w: pl.Tensor[[N_LAYERS, HEAD_DIM], pl.BF16],
        sink: pl.Tensor[[N_LAYERS, LOCAL_HEADS], pl.FP32],
        wo_a: pl.Tensor[[N_LAYERS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[N_LAYERS, D, O_GROUPS * O_LORA], pl.BF16],
        hc_ffn_fn: pl.Tensor[[N_LAYERS, MIX_PAD, HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[N_LAYERS, 3], pl.FP32],
        hc_ffn_base: pl.Tensor[[N_LAYERS, MIX_HC], pl.FP32],
        ffn_norm_w: pl.Tensor[[N_LAYERS, D], pl.BF16],
        gate_weight: pl.Tensor[[N_LAYERS, N_EXPERTS, D], pl.FP32],
        gate_bias: pl.Tensor[[N_LAYERS, N_EXPERTS], pl.FP32],
        shared_w1: pl.Tensor[[N_LAYERS, D, INTER], pl.INT8, pl.NZ],
        shared_w1_scale: pl.Tensor[[N_LAYERS, INTER], pl.FP32],
        shared_w3: pl.Tensor[[N_LAYERS, D, INTER], pl.INT8, pl.NZ],
        shared_w3_scale: pl.Tensor[[N_LAYERS, INTER], pl.FP32],
        shared_w2: pl.Tensor[[N_LAYERS, INTER, D], pl.INT8, pl.NZ],
        shared_w2_scale: pl.Tensor[[N_LAYERS, D], pl.FP32],
        w1: pl.Tensor[_RANK_W1, pl.INT8, pl.NZ],
        w1_scale: pl.Tensor[_RANK_W1_SCALE, pl.FP32],
        w3: pl.Tensor[_RANK_W3, pl.INT8, pl.NZ],
        w3_scale: pl.Tensor[_RANK_W3_SCALE, pl.FP32],
        w2: pl.Tensor[_RANK_W2, pl.INT8, pl.NZ],
        w2_scale: pl.Tensor[_RANK_W2_SCALE, pl.FP32],
        c2a_comp_wkv: pl.Tensor[[3, HEAD_DIM, D], pl.BF16],
        c2a_comp_wgate: pl.Tensor[[3, HEAD_DIM, D], pl.BF16],
        c2a_comp_norm: pl.Tensor[[3, HEAD_DIM], pl.BF16],
        c2a_idx_wq_b: pl.Tensor[[3, Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
        c2a_idx_wq_b_scale: pl.Tensor[[3, IDX_HEADS * IDX_DIM], pl.FP32],
        c2a_idx_weights_proj: pl.Tensor[[3, D, IDX_HEADS], pl.BF16],
        c2a_idx_wk: pl.Tensor[[3, IDX_DIM, HEAD_DIM], pl.BF16],
        c2a_idx_k_norm: pl.Tensor[[3, IDX_DIM], pl.BF16],
        c1a_comp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
        c1a_comp_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
        c1a_idx_wq_b: pl.Tensor[[Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8, pl.NZ],
        c1a_idx_wq_b_scale: pl.Tensor[[IDX_HEADS * IDX_DIM], pl.FP32],
        c1a_idx_weights_proj: pl.Tensor[[D, IDX_HEADS], pl.BF16],
        c1a_idx_wk: pl.Tensor[[IDX_DIM, HEAD_DIM], pl.BF16],
        c1a_idx_k_norm: pl.Tensor[[IDX_DIM], pl.BF16],
        re_idx_wq_b: pl.Tensor[[N_C1A_SLOTS - 1, Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
        re_idx_wq_b_scale: pl.Tensor[[N_C1A_SLOTS - 1, IDX_HEADS * IDX_DIM], pl.FP32],
        re_idx_weights_proj: pl.Tensor[[N_C1A_SLOTS - 1, D, IDX_HEADS], pl.BF16],
        engram_lookup: pl.Tensor[[N_ENGRAM, T_DYN, ENGRAM_K], pl.BF16],
        engram_wkv: pl.Tensor[[N_ENGRAM, ENGRAM_OUT, ENGRAM_K], pl.BF16],
        engram_qk: pl.Tensor[[N_ENGRAM, 4, D], pl.FP32],
        engram_rotation: pl.Tensor[[32, 32], pl.FP32],
        engram_enabled: pl.Scalar[pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[N_LAYERS, ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        compressed_cache_c2a: pl.InOut[
            pl.Tensor[[3, CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
        ],
        index_cache_c2a: pl.InOut[pl.Tensor[[3, INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16]],
        state_cache_c2a: pl.InOut[
            pl.Tensor[[3, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]
        ],
        compressed_cache_c1a: pl.InOut[
            pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
        ],
        index_cache_c1a: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16]],
        topk_indices_c2a: pl.InOut[pl.Tensor[[3, T_DYN, INDEX_TOPK], pl.INT32]],
        topk_values_c2a: pl.InOut[pl.Tensor[[3, T_DYN, INDEX_TOPK], pl.FP32]],
        pool_block_ids_c2a: pl.InOut[pl.Tensor[[3, T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
        pool_block_counts_c2a: pl.InOut[pl.Tensor[[3, T_DYN], pl.INT32]],
        topk_indices_c1a: pl.InOut[pl.Tensor[[N_C1A_SLOTS, T_DYN, INDEX_TOPK], pl.INT32]],
        topk_values_c1a: pl.InOut[pl.Tensor[[N_C1A_SLOTS, T_DYN, INDEX_TOPK], pl.FP32]],
        pool_block_ids_c1a: pl.InOut[pl.Tensor[[N_C1A_SLOTS, T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
        pool_block_counts_c1a: pl.InOut[pl.Tensor[[N_C1A_SLOTS, T_DYN], pl.INT32]],
        token_counts: pl.Tensor[[EP_SIZE], pl.INT32],
        x_hc: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
        pre_mix: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
        layer_x_hc: pl.Out[pl.Tensor[[N_LAYERS, T_DYN, HC_MULT, D], pl.FP32]],
        layer_pre_mix: pl.Out[pl.Tensor[[N_LAYERS, T_DYN, HC_MULT], pl.FP32]],
        final_hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
        logits: pl.Out[pl.Tensor[[T_DYN, LOCAL_VOCAB], pl.FP32]],
        recv_x: pld.DistributedTensor,
        recv_aux: pld.DistributedTensor,
        recv_meta: pld.DistributedTensor,
        meta_ready: pld.DistributedTensor,
        data_ready: pld.DistributedTensor,
        return_slots: pld.DistributedTensor,
        return_ready: pld.DistributedTensor,
        consumed: pld.DistributedTensor,
        return_consumed: pld.DistributedTensor,
        output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        embed_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
        embed_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        """One rank: embedding, encoder layers, decoder layers, final head."""
        input_ids.bind_dynamic(0, T_DYN)
        initial_pre.bind_dynamic(0, T_DYN)
        position_ids.bind_dynamic(0, T_DYN)
        request_ids.bind_dynamic(0, T_DYN)
        window_slots.bind_dynamic(0, T_DYN)
        window_indices.bind_dynamic(0, T_DYN)
        index_block_table.bind_dynamic(0, B_DYN)
        index_block_table.bind_dynamic(1, TABLE_DYN)
        state_block_table.bind_dynamic(0, B_DYN)
        rope_cos.bind_dynamic(0, T_DYN)
        rope_cos_swa.bind_dynamic(0, T_DYN)
        rope_sin_swa.bind_dynamic(0, T_DYN)
        rope_sin.bind_dynamic(0, T_DYN)
        c2a_compressed_slots.bind_dynamic(0, T_DYN)
        c2a_compressed_lens.bind_dynamic(0, T_DYN)
        c2a_compressed_cos.bind_dynamic(0, T_DYN)
        c2a_compressed_sin.bind_dynamic(0, T_DYN)
        c1a_compressed_slots.bind_dynamic(0, T_DYN)
        c1a_compressed_lens.bind_dynamic(0, T_DYN)
        c1a_compressed_cos.bind_dynamic(0, T_DYN)
        c1a_compressed_sin.bind_dynamic(0, T_DYN)
        engram_lookup.bind_dynamic(1, T_DYN)
        window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
        compressed_cache_c2a.bind_dynamic(1, CMP_BLOCKS_DYN)
        compressed_cache_c1a.bind_dynamic(0, CMP_BLOCKS_DYN)
        index_cache_c2a.bind_dynamic(1, INDEX_BLOCKS_DYN)
        index_cache_c1a.bind_dynamic(0, INDEX_BLOCKS_DYN)
        state_cache_c2a.bind_dynamic(1, STATE_BLOCKS_DYN)
        topk_indices_c2a.bind_dynamic(1, T_DYN)
        topk_values_c2a.bind_dynamic(1, T_DYN)
        pool_block_ids_c2a.bind_dynamic(1, T_DYN)
        pool_block_ids_c2a.bind_dynamic(2, POOL_BLOCKS_DYN)
        pool_block_counts_c2a.bind_dynamic(1, T_DYN)
        topk_indices_c1a.bind_dynamic(1, T_DYN)
        topk_values_c1a.bind_dynamic(1, T_DYN)
        pool_block_ids_c1a.bind_dynamic(1, T_DYN)
        pool_block_ids_c1a.bind_dynamic(2, POOL_BLOCKS_DYN)
        pool_block_counts_c1a.bind_dynamic(1, T_DYN)
        x_hc.bind_dynamic(0, T_DYN)
        pre_mix.bind_dynamic(0, T_DYN)
        layer_x_hc.bind_dynamic(1, T_DYN)
        layer_pre_mix.bind_dynamic(1, T_DYN)
        final_hidden.bind_dynamic(0, T_DYN)
        logits.bind_dynamic(0, T_DYN)
        tokens_rt = pl.tensor.dim(x_hc, 0)
        num_tokens = pl.read(token_counts, [rank])
        group_base = rank // TP_SIZE * TP_SIZE
        tp_rank = rank % TP_SIZE
        layer_x_hc_flat = pl.reshape(layer_x_hc, [N_LAYERS * tokens_rt * HC_MULT, D])
        layer_pre_mix_flat = pl.reshape(layer_pre_mix, [N_LAYERS * tokens_rt, HC_MULT])
        _prefill_embed(
            input_ids, embed_weight, initial_pre, x_hc, pre_mix,
            embed_window, embed_arrived, group_base, tp_rank,
            pl.const(1, pl.INT32),
        )
        prefill_encoder(
            x_hc, pre_mix, layer_x_hc_flat, layer_pre_mix_flat,
            hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, rope_cos_swa, rope_sin_swa, window_slots, window_indices,
            window_cache, compressed_cache_c2a, index_cache_c2a, state_cache_c2a,
            request_ids, c2a_compressed_lens, index_block_table,
            c2a_compressed_cos, c2a_compressed_sin, c2a_compressed_slots,
            position_ids, state_block_table,
            c2a_comp_wkv, c2a_comp_wgate, c2a_comp_norm,
            c2a_idx_wq_b, c2a_idx_wq_b_scale, c2a_idx_weights_proj,
            c2a_idx_wk, c2a_idx_k_norm,
            topk_indices_c2a, topk_values_c2a, pool_block_ids_c2a, pool_block_counts_c2a,
            engram_lookup, engram_wkv, engram_qk, engram_rotation, engram_enabled,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w, gate_weight, gate_bias,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            w1, w1_scale, w3, w3_scale, w2, w2_scale,
            recv_x, recv_aux, recv_meta, meta_ready, data_ready,
            return_slots, return_ready, consumed, return_consumed,
            output_window, output_arrived,
            rank, group_base, tp_rank, num_tokens, rows=tokens,
        )
        prefill_decoder(
            x_hc, pre_mix, layer_x_hc_flat, layer_pre_mix_flat,
            hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
            sink, wo_a, wo_b, rope_cos, rope_sin, rope_cos_swa, rope_sin_swa, window_slots, window_indices,
            window_cache, compressed_cache_c1a, index_cache_c1a,
            request_ids, c1a_compressed_lens, index_block_table,
            c1a_compressed_cos, c1a_compressed_sin, c1a_compressed_slots,
            position_ids, c1a_comp_wkv, c1a_comp_norm,
            c1a_idx_wq_b, c1a_idx_wq_b_scale, c1a_idx_weights_proj,
            c1a_idx_wk, c1a_idx_k_norm,
            re_idx_wq_b, re_idx_wq_b_scale, re_idx_weights_proj,
            topk_indices_c1a, topk_values_c1a, pool_block_ids_c1a, pool_block_counts_c1a,
            engram_lookup, engram_wkv, engram_qk, engram_rotation, engram_enabled,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base, ffn_norm_w, gate_weight, gate_bias,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            w1, w1_scale, w3, w3_scale, w2, w2_scale,
            recv_x, recv_aux, recv_meta, meta_ready, data_ready,
            return_slots, return_ready, consumed, return_consumed,
            output_window, output_arrived,
            rank, group_base, tp_rank, num_tokens, rows=tokens,
        )
        hc_head(x_hc, pre_mix, final_hidden)
        lm_head(final_hidden, final_norm, lm_head_weight, logits, vocab=LOCAL_VOCAB)

    @pl.jit.host
    def prefill_fwd(
        input_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
        embed_weight: pl.Tensor[[EP_SIZE, LOCAL_VOCAB, D], pl.BF16],
        initial_pre: pl.Tensor[[EP_SIZE, T_DYN, HC_MULT], pl.FP32],
        final_norm: pl.Tensor[[EP_SIZE, D], pl.BF16],
        lm_head_weight: pl.Tensor[[EP_SIZE, LOCAL_VOCAB, D], pl.BF16],
        position_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
        request_ids: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
        window_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
        window_indices: pl.Tensor[[EP_SIZE, T_DYN, WINDOW_KEYS], pl.INT32],
        index_block_table: pl.Tensor[[EP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
        state_block_table: pl.Tensor[[EP_SIZE, B_DYN, 1], pl.INT32],
        rope_cos: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        rope_sin: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        rope_cos_swa: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        rope_sin_swa: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        c2a_compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
        c2a_compressed_lens: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
        c2a_compressed_cos: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        c2a_compressed_sin: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        c1a_compressed_slots: pl.Tensor[[EP_SIZE, T_DYN], pl.INT64],
        c1a_compressed_lens: pl.Tensor[[EP_SIZE, T_DYN], pl.INT32],
        c1a_compressed_cos: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        c1a_compressed_sin: pl.Tensor[[EP_SIZE, T_DYN, 32], pl.FP32],
        hc_attn_fn: pl.Tensor[[EP_SIZE, N_LAYERS, MIX_PAD, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[EP_SIZE, N_LAYERS, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[EP_SIZE, N_LAYERS, MIX_HC], pl.FP32],
        attn_norm_w: pl.Tensor[[EP_SIZE, N_LAYERS, D], pl.BF16],
        wq_a: pl.Tensor[[EP_SIZE, N_LAYERS, D, Q_LORA], pl.INT8],
        wq_a_scale: pl.Tensor[[EP_SIZE, N_LAYERS, Q_LORA], pl.FP32],
        q_norm_w: pl.Tensor[[EP_SIZE, N_LAYERS, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[EP_SIZE, N_LAYERS, Q_LORA, LOCAL_Q], pl.INT8],
        wq_b_scale: pl.Tensor[[EP_SIZE, N_LAYERS, LOCAL_Q], pl.FP32],
        wkv: pl.Tensor[[EP_SIZE, N_LAYERS, D, HEAD_DIM], pl.INT8],
        wkv_scale: pl.Tensor[[EP_SIZE, N_LAYERS, HEAD_DIM], pl.FP32],
        kv_norm_w: pl.Tensor[[EP_SIZE, N_LAYERS, HEAD_DIM], pl.BF16],
        sink: pl.Tensor[[EP_SIZE, N_LAYERS, LOCAL_HEADS], pl.FP32],
        wo_a: pl.Tensor[[EP_SIZE, N_LAYERS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[EP_SIZE, N_LAYERS, D, O_GROUPS * O_LORA], pl.BF16],
        hc_ffn_fn: pl.Tensor[[EP_SIZE, N_LAYERS, MIX_PAD, HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[EP_SIZE, N_LAYERS, 3], pl.FP32],
        hc_ffn_base: pl.Tensor[[EP_SIZE, N_LAYERS, MIX_HC], pl.FP32],
        ffn_norm_w: pl.Tensor[[EP_SIZE, N_LAYERS, D], pl.BF16],
        gate_weight: pl.Tensor[[EP_SIZE, N_LAYERS, N_EXPERTS, D], pl.FP32],
        gate_bias: pl.Tensor[[EP_SIZE, N_LAYERS, N_EXPERTS], pl.FP32],
        shared_w1: pl.Tensor[[EP_SIZE, N_LAYERS, D, INTER], pl.INT8],
        shared_w1_scale: pl.Tensor[[EP_SIZE, N_LAYERS, INTER], pl.FP32],
        shared_w3: pl.Tensor[[EP_SIZE, N_LAYERS, D, INTER], pl.INT8],
        shared_w3_scale: pl.Tensor[[EP_SIZE, N_LAYERS, INTER], pl.FP32],
        shared_w2: pl.Tensor[[EP_SIZE, N_LAYERS, INTER, D], pl.INT8],
        shared_w2_scale: pl.Tensor[[EP_SIZE, N_LAYERS, D], pl.FP32],
        w1: pl.Tensor[_HOST_W1, pl.INT8],
        w1_scale: pl.Tensor[_HOST_W1_SCALE, pl.FP32],
        w3: pl.Tensor[_HOST_W3, pl.INT8],
        w3_scale: pl.Tensor[_HOST_W3_SCALE, pl.FP32],
        w2: pl.Tensor[_HOST_W2, pl.INT8],
        w2_scale: pl.Tensor[_HOST_W2_SCALE, pl.FP32],
        c2a_comp_wkv: pl.Tensor[[EP_SIZE, 3, HEAD_DIM, D], pl.BF16],
        c2a_comp_wgate: pl.Tensor[[EP_SIZE, 3, HEAD_DIM, D], pl.BF16],
        c2a_comp_norm: pl.Tensor[[EP_SIZE, 3, HEAD_DIM], pl.BF16],
        c2a_idx_wq_b: pl.Tensor[[EP_SIZE, 3, Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
        c2a_idx_wq_b_scale: pl.Tensor[[EP_SIZE, 3, IDX_HEADS * IDX_DIM], pl.FP32],
        c2a_idx_weights_proj: pl.Tensor[[EP_SIZE, 3, D, IDX_HEADS], pl.BF16],
        c2a_idx_wk: pl.Tensor[[EP_SIZE, 3, IDX_DIM, HEAD_DIM], pl.BF16],
        c2a_idx_k_norm: pl.Tensor[[EP_SIZE, 3, IDX_DIM], pl.BF16],
        c1a_comp_wkv: pl.Tensor[[EP_SIZE, HEAD_DIM, D], pl.BF16],
        c1a_comp_norm: pl.Tensor[[EP_SIZE, HEAD_DIM], pl.BF16],
        c1a_idx_wq_b: pl.Tensor[[EP_SIZE, Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
        c1a_idx_wq_b_scale: pl.Tensor[[EP_SIZE, IDX_HEADS * IDX_DIM], pl.FP32],
        c1a_idx_weights_proj: pl.Tensor[[EP_SIZE, D, IDX_HEADS], pl.BF16],
        c1a_idx_wk: pl.Tensor[[EP_SIZE, IDX_DIM, HEAD_DIM], pl.BF16],
        c1a_idx_k_norm: pl.Tensor[[EP_SIZE, IDX_DIM], pl.BF16],
        re_idx_wq_b: pl.Tensor[[EP_SIZE, N_C1A_SLOTS - 1, Q_LORA, IDX_HEADS * IDX_DIM], pl.INT8],
        re_idx_wq_b_scale: pl.Tensor[[EP_SIZE, N_C1A_SLOTS - 1, IDX_HEADS * IDX_DIM], pl.FP32],
        re_idx_weights_proj: pl.Tensor[[EP_SIZE, N_C1A_SLOTS - 1, D, IDX_HEADS], pl.BF16],
        engram_lookup: pl.Tensor[[EP_SIZE, N_ENGRAM, T_DYN, ENGRAM_K], pl.BF16],
        engram_wkv: pl.Tensor[[EP_SIZE, N_ENGRAM, ENGRAM_OUT, ENGRAM_K], pl.BF16],
        engram_qk: pl.Tensor[[EP_SIZE, N_ENGRAM, 4, D], pl.FP32],
        engram_rotation: pl.Tensor[[EP_SIZE, 32, 32], pl.FP32],
        engram_enabled: pl.Scalar[pl.INT32],
        window_cache: pl.InOut[
            pl.Tensor[[EP_SIZE, N_LAYERS, ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
        ],
        compressed_cache_c2a: pl.InOut[
            pl.Tensor[[EP_SIZE, 3, CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
        ],
        index_cache_c2a: pl.InOut[
            pl.Tensor[[EP_SIZE, 3, INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16]
        ],
        state_cache_c2a: pl.InOut[
            pl.Tensor[[EP_SIZE, 3, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]
        ],
        compressed_cache_c1a: pl.InOut[
            pl.Tensor[[EP_SIZE, CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
        ],
        index_cache_c1a: pl.InOut[
            pl.Tensor[[EP_SIZE, INDEX_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_DIM], pl.BF16]
        ],
        topk_indices_c2a: pl.InOut[pl.Tensor[[EP_SIZE, 3, T_DYN, INDEX_TOPK], pl.INT32]],
        topk_values_c2a: pl.InOut[pl.Tensor[[EP_SIZE, 3, T_DYN, INDEX_TOPK], pl.FP32]],
        pool_block_ids_c2a: pl.InOut[pl.Tensor[[EP_SIZE, 3, T_DYN, POOL_BLOCKS_DYN], pl.INT32]],
        pool_block_counts_c2a: pl.InOut[pl.Tensor[[EP_SIZE, 3, T_DYN], pl.INT32]],
        topk_indices_c1a: pl.InOut[pl.Tensor[[EP_SIZE, N_C1A_SLOTS, T_DYN, INDEX_TOPK], pl.INT32]],
        topk_values_c1a: pl.InOut[pl.Tensor[[EP_SIZE, N_C1A_SLOTS, T_DYN, INDEX_TOPK], pl.FP32]],
        pool_block_ids_c1a: pl.InOut[
            pl.Tensor[[EP_SIZE, N_C1A_SLOTS, T_DYN, POOL_BLOCKS_DYN], pl.INT32]
        ],
        pool_block_counts_c1a: pl.InOut[pl.Tensor[[EP_SIZE, N_C1A_SLOTS, T_DYN], pl.INT32]],
        token_counts: pl.Tensor[[EP_SIZE], pl.INT32],
        x_hc: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
        pre_mix: pl.InOut[pl.Tensor[[EP_SIZE, T_DYN, HC_MULT], pl.FP32]],
        layer_x_hc: pl.Out[pl.Tensor[[EP_SIZE, N_LAYERS, T_DYN, HC_MULT, D], pl.FP32]],
        layer_pre_mix: pl.Out[pl.Tensor[[EP_SIZE, N_LAYERS, T_DYN, HC_MULT], pl.FP32]],
        final_hidden: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, D], pl.BF16]],
        logits: pl.Out[pl.Tensor[[EP_SIZE, T_DYN, LOCAL_VOCAB], pl.FP32]],
    ):
        """Run the complete prefill program once on every rank.

        All EP16 exchange windows and the TP output window are allocated here,
        once, and handed to each rank's device program with ``pld.window(...)``.
        They are reused across the forty layers; the per-layer epochs keep the
        monotonic counters consistent (see the layer wrappers).
        """
        input_ids.bind_dynamic(1, T_DYN)
        initial_pre.bind_dynamic(1, T_DYN)
        position_ids.bind_dynamic(1, T_DYN)
        request_ids.bind_dynamic(1, T_DYN)
        window_slots.bind_dynamic(1, T_DYN)
        window_indices.bind_dynamic(1, T_DYN)
        index_block_table.bind_dynamic(1, B_DYN)
        index_block_table.bind_dynamic(2, TABLE_DYN)
        state_block_table.bind_dynamic(1, B_DYN)
        rope_cos.bind_dynamic(1, T_DYN)
        rope_cos_swa.bind_dynamic(1, T_DYN)
        rope_sin_swa.bind_dynamic(1, T_DYN)
        rope_sin.bind_dynamic(1, T_DYN)
        c2a_compressed_slots.bind_dynamic(1, T_DYN)
        c2a_compressed_lens.bind_dynamic(1, T_DYN)
        c2a_compressed_cos.bind_dynamic(1, T_DYN)
        c2a_compressed_sin.bind_dynamic(1, T_DYN)
        c1a_compressed_slots.bind_dynamic(1, T_DYN)
        c1a_compressed_lens.bind_dynamic(1, T_DYN)
        c1a_compressed_cos.bind_dynamic(1, T_DYN)
        c1a_compressed_sin.bind_dynamic(1, T_DYN)
        engram_lookup.bind_dynamic(2, T_DYN)
        window_cache.bind_dynamic(2, ORI_BLOCKS_DYN)
        compressed_cache_c2a.bind_dynamic(2, CMP_BLOCKS_DYN)
        compressed_cache_c1a.bind_dynamic(1, CMP_BLOCKS_DYN)
        index_cache_c2a.bind_dynamic(2, INDEX_BLOCKS_DYN)
        index_cache_c1a.bind_dynamic(1, INDEX_BLOCKS_DYN)
        state_cache_c2a.bind_dynamic(2, STATE_BLOCKS_DYN)
        topk_indices_c2a.bind_dynamic(2, T_DYN)
        topk_values_c2a.bind_dynamic(2, T_DYN)
        pool_block_ids_c2a.bind_dynamic(2, T_DYN)
        pool_block_ids_c2a.bind_dynamic(3, POOL_BLOCKS_DYN)
        pool_block_counts_c2a.bind_dynamic(2, T_DYN)
        topk_indices_c1a.bind_dynamic(2, T_DYN)
        topk_values_c1a.bind_dynamic(2, T_DYN)
        pool_block_ids_c1a.bind_dynamic(2, T_DYN)
        pool_block_ids_c1a.bind_dynamic(3, POOL_BLOCKS_DYN)
        pool_block_counts_c1a.bind_dynamic(2, T_DYN)
        x_hc.bind_dynamic(1, T_DYN)
        pre_mix.bind_dynamic(1, T_DYN)
        layer_x_hc.bind_dynamic(2, T_DYN)
        layer_pre_mix.bind_dynamic(2, T_DYN)
        final_hidden.bind_dynamic(1, T_DYN)
        logits.bind_dynamic(1, T_DYN)
        recv_rows = N_LOCAL_EXPERTS * EP_SIZE * tokens
        output_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        output_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        embed_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        embed_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        recv_x_buffer = pld.alloc_window_buffer([recv_rows, D], dtype=pl.BF16)
        recv_aux_buffer = pld.alloc_window_buffer([recv_rows, 8], dtype=pl.FP32)
        recv_meta_buffer = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        meta_ready_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        data_ready_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        return_buffer = pld.alloc_window_buffer([tokens * TOPK, D], dtype=pl.BF16)
        return_ready_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        consumed_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        return_consumed_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(pld.world_size()):
            output_window = pld.window(output_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
            output_arrived = pld.window(output_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            embed_window = pld.window(embed_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
            embed_arrived = pld.window(embed_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            recv_x = pld.window(recv_x_buffer, [recv_rows, D], dtype=pl.BF16)
            recv_aux = pld.window(recv_aux_buffer, [recv_rows, 8], dtype=pl.FP32)
            recv_meta = pld.window(recv_meta_buffer, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
            meta_ready = pld.window(meta_ready_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            data_ready = pld.window(data_ready_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            return_slots = pld.window(return_buffer, [tokens * TOPK, D], dtype=pl.BF16)
            return_ready = pld.window(return_ready_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            consumed = pld.window(consumed_buffer, [EP_SIZE, 1], dtype=pl.INT32)
            return_consumed = pld.window(
                return_consumed_buffer, [EP_SIZE, 1], dtype=pl.INT32,
            )
            l2_prefill_fwd(
                input_ids[rank], embed_weight[rank], initial_pre[rank],
                final_norm[rank], lm_head_weight[rank],
                position_ids[rank], request_ids[rank], window_slots[rank],
                window_indices[rank], index_block_table[rank], state_block_table[rank],
                rope_cos[rank], rope_sin[rank], rope_cos_swa[rank], rope_sin_swa[rank],
                c2a_compressed_slots[rank], c2a_compressed_lens[rank],
                c2a_compressed_cos[rank], c2a_compressed_sin[rank],
                c1a_compressed_slots[rank], c1a_compressed_lens[rank],
                c1a_compressed_cos[rank], c1a_compressed_sin[rank],
                hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_w[rank],
                wq_a[rank], wq_a_scale[rank], q_norm_w[rank], wq_b[rank], wq_b_scale[rank],
                wkv[rank], wkv_scale[rank], kv_norm_w[rank], sink[rank],
                wo_a[rank], wo_b[rank],
                hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], ffn_norm_w[rank],
                gate_weight[rank], gate_bias[rank],
                shared_w1[rank], shared_w1_scale[rank], shared_w3[rank],
                shared_w3_scale[rank], shared_w2[rank], shared_w2_scale[rank],
                w1[rank], w1_scale[rank], w3[rank], w3_scale[rank],
                w2[rank], w2_scale[rank],
                c2a_comp_wkv[rank], c2a_comp_wgate[rank], c2a_comp_norm[rank],
                c2a_idx_wq_b[rank], c2a_idx_wq_b_scale[rank], c2a_idx_weights_proj[rank],
                c2a_idx_wk[rank], c2a_idx_k_norm[rank],
                c1a_comp_wkv[rank], c1a_comp_norm[rank], c1a_idx_wq_b[rank],
                c1a_idx_wq_b_scale[rank], c1a_idx_weights_proj[rank],
                c1a_idx_wk[rank], c1a_idx_k_norm[rank],
                re_idx_wq_b[rank], re_idx_wq_b_scale[rank], re_idx_weights_proj[rank],
                engram_lookup[rank], engram_wkv[rank], engram_qk[rank], engram_rotation[rank], engram_enabled,
                window_cache[rank], compressed_cache_c2a[rank], index_cache_c2a[rank],
                state_cache_c2a[rank], compressed_cache_c1a[rank], index_cache_c1a[rank],
                topk_indices_c2a[rank], topk_values_c2a[rank],
                pool_block_ids_c2a[rank], pool_block_counts_c2a[rank],
                topk_indices_c1a[rank], topk_values_c1a[rank],
                pool_block_ids_c1a[rank], pool_block_counts_c1a[rank],
                token_counts, x_hc[rank], pre_mix[rank],
                layer_x_hc[rank], layer_pre_mix[rank], final_hidden[rank], logits[rank],
                recv_x, recv_aux, recv_meta, meta_ready, data_ready,
                return_slots, return_ready, consumed, return_consumed,
                output_window, output_arrived,
                embed_window, embed_arrived,
                rank, device=rank,
            )

    return prefill_fwd


_ENTRY_CACHE = {}


def _prefill_entry(tokens):
    """Cached entry object for one compiled token bound."""
    entry = _ENTRY_CACHE.get(tokens)
    if entry is None:
        entry = _ENTRY_CACHE[tokens] = _build_prefill_entry(tokens)
    return entry


def _entry_specs(entry, tensors):
    """Build the harness specs in the entry's declared parameter order."""
    from golden import ScalarSpec, TensorSpec

    declared = list(entry.param_names)
    missing = [name for name in declared if name not in tensors]
    extra = [name for name in tensors if name not in declared]
    if missing or extra:
        raise SystemExit(f"fixture/spec mismatch: missing={missing} extra={extra}")
    return [
        ScalarSpec(name, torch.int32, tensors[name], compile_runtime=True) if name == "engram_enabled"
        else TensorSpec(name, list(tensors[name].shape), tensors[name].dtype, init_value=tensors[name])
        for name in declared
    ]


def run_prefill_fwd(tensors, config, compile_only=False):
    """Compile and dispatch the single-program prefill entry.

    ``prefill_fwd`` is the only compiled host program: ``run`` compiles it,
    checks the spec ABI and (unless ``compile_only``) dispatches it once,
    writing the device results back into the caller's tensors.
    """
    from golden import run

    tokens = _token_tile(tensors)
    entry = _prefill_entry(tokens)
    if not compile_only:
        _share_base_tensors(tensors)
    result = run(fn=entry, specs=_entry_specs(entry, tensors), config=config, compile_only=compile_only)
    if not result.passed:
        raise SystemExit(result.error or 1)
    return tensors["logits"]


# ---------------------------------------------------------------------------
# Host weight materialization (real checkpoint path)
# ---------------------------------------------------------------------------

DENSE_STACK_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_w", "sink", "wo_a", "wo_b", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    "ffn_norm_w", "gate_w", "gate_b",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
)
EXPERT_STACK_NAMES = ("w1", "w1_scale", "w3", "w3_scale", "w2", "w2_scale")
C2A_SOURCE_NAMES = (
    "comp_wkv", "comp_wgate", "comp_norm", "idx_wq_b", "idx_wq_b_scale",
    "idx_weights_proj", "idx_wk", "idx_k_norm",
)
C1A_SOURCE_NAMES = (
    "comp_wkv", "comp_norm", "idx_wq_b", "idx_wq_b_scale",
    "idx_weights_proj", "idx_wk", "idx_k_norm",
)
REINDEX_SOURCE_NAMES = ("idx_wq_b", "idx_wq_b_scale", "idx_weights_proj")
TOP_LEVEL_CONTRACT_NAMES = ("embed_weight", "lm_head_weight", "final_norm")


def _stack_source(loader, layers, names):
    collected = {name: [] for name in names}
    for layer in layers:
        dense = loader.layer_dense(layer)
        for name in names:
            collected[name].append(dense[name])
    return {name: torch.stack(values) for name, values in collected.items()}


def prepare_fwd_weights(loader):
    """Materialize one rank's complete argument set from ``WeightLoader``.

    Dense stacks come from ``WeightLoader.stacked`` (``[N_LAYERS, ...]``);
    routed experts from the rank's per-layer EP16 shard
    (``[N_LAYERS, N_LOCAL_EXPERTS, ...]``); cache-owner weights from the source
    layers.  The caller stacks the per-rank dictionaries along the rank axis
    (``[EP_SIZE, N_LAYERS, ...]``) and supplies the host Engram rows via
    ``engram_table(layer)``.  The bounded fixture does not use this.

    The bounded entry currently declares the routed-expert shards with the rank
    axis only (see the module docstring); a real per-layer expert stack is
    materialized here but needs the layer axis in the entry ABI to be wired.
    """
    arguments = loader.top_level_arguments()
    arguments.update(loader.stacked(DENSE_STACK_NAMES))
    experts = {name: [] for name in EXPERT_STACK_NAMES}
    for layer in range(N_LAYERS):
        shard = loader.expert_shard(layer)
        for name in EXPERT_STACK_NAMES:
            experts[name].append(shard[name])
    for name in EXPERT_STACK_NAMES:
        arguments[name] = torch.stack(experts[name])
    for name, value in _stack_source(loader, C2A_SOURCES, C2A_SOURCE_NAMES).items():
        arguments["c2a_" + name] = value
    for name, value in _stack_source(loader, C1A_SOURCES, C1A_SOURCE_NAMES).items():
        arguments["c1a_" + name] = value
    for name, value in _stack_source(loader, REINDEX_LAYERS, REINDEX_SOURCE_NAMES).items():
        arguments["re_" + name] = value
    arguments["engram_tables"] = {layer: loader.engram_table(layer) for layer in M.engram_layer_ids}
    return arguments


def check_weight_contract(loader, layers=(0, 1, 2, 14, 20, 24, 39)):
    """Compare sampled checkpoint materializations with the entry's contracts."""
    layers = list(layers)
    arguments = loader.top_level_arguments()
    arguments.update(loader.stacked(DENSE_STACK_NAMES, layers=layers))
    problems = []
    for name in TOP_LEVEL_CONTRACT_NAMES:
        if name not in arguments:
            problems.append(f"top-level argument {name} missing")
    for name, value in arguments.items():
        if name in TOP_LEVEL_CONTRACT_NAMES:
            continue
        if value.shape[0] != len(layers):
            problems.append(f"{name}: leading axis {value.shape[0]} != {len(layers)}")
    for layer in layers:
        dense = loader.layer_dense(layer)
        optional = loader.optional_arguments(layer)
        print(f"[PREFILL_FWD] layer {layer}: dense={len(dense)} optional={sorted(optional)}")
        if layer in REINDEX_LAYERS:
            if "idx_wq_b" not in optional or "idx_weights_proj" not in optional:
                problems.append(f"layer {layer}: reindex query weights are missing")
            for name in ("idx_wk", "idx_k_norm"):
                if name in optional:
                    problems.append(f"layer {layer}: unexpected {name}")
        if "engram_wkv" in optional:
            table = loader.engram_table(layer)
            print(f"[PREFILL_FWD] layer {layer}: Engram table rows={table.rows} width={table.width}")
    for layer in layers:
        shard = loader.expert_shard(layer)
        for name in EXPERT_STACK_NAMES:
            expected = (N_LOCAL_EXPERTS,) + tuple(shard[name].shape[1:])
            if tuple(shard[name].shape) != expected:
                problems.append(f"layer {layer}.{name}: {tuple(shard[name].shape)} != {expected}")
    if problems:
        for problem in problems:
            print(f"[PREFILL_FWD] CONTRACT FAIL {problem}")
        raise SystemExit(1)
    print("[PREFILL_FWD] weight contract PASS")
    return 0


# ---------------------------------------------------------------------------
# Bounded synthetic fixture (paged request plan)
# ---------------------------------------------------------------------------

# Immutable fixture tensors and pristine copies of the mutable (InOut) tensors.
_FIXTURE = {}
_FIXTURE_INITIAL = {}
_MUTABLE_NAMES = []

MUTABLE_NAMES = (
    "window_cache", "compressed_cache_c2a", "index_cache_c2a", "state_cache_c2a",
    "compressed_cache_c1a", "index_cache_c1a",
    "topk_indices_c2a", "topk_values_c2a", "pool_block_ids_c2a", "pool_block_counts_c2a",
    "topk_indices_c1a", "topk_values_c1a", "pool_block_ids_c1a", "pool_block_counts_c1a",
    "x_hc", "pre_mix",
)


def _token_tile(tensors):
    return tensors["position_ids"].shape[1]


def _share_base_tensors(tensors):
    """Move the fixture's base buffers into shared memory before planning."""
    for value in list(tensors.values()):
        if isinstance(value, torch.Tensor) and not value.is_shared() and value.is_contiguous():
            value.share_memory_()


def _rope_tables(count, theta=None):
    half = 32
    theta = M.rope_theta if theta is None else theta
    inv_freq = 1.0 / torch.pow(
        torch.tensor(theta, dtype=torch.float32),
        torch.arange(0, half, dtype=torch.float32) * 2.0 / (2 * half),
    )
    angle = torch.arange(0, count, dtype=torch.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
    return torch.cos(angle), torch.sin(angle)


def _rank_repeat(value):
    """Replicate a ``[tokens, ...]`` fixture tensor across the EP rank axis."""
    return value.unsqueeze(0).expand(EP_SIZE, *value.shape).contiguous()


def _layer_weight(shape, dtype, pattern, packed=False, zero_tail=None, compile_only=False):
    """``[EP_SIZE, N_LAYERS, *shape]`` with a distinct small constant per layer."""
    value = torch.empty((EP_SIZE, N_LAYERS, *shape), dtype=dtype)
    if compile_only:
        return value
    for layer in range(N_LAYERS):
        value[:, layer].fill_(pattern[layer % len(pattern)])
        if zero_tail is not None:
            value[:, layer, zero_tail:].zero_()
    return pack_nz(value) if packed else value


def _layer_expert_weight(shape, dtype, pattern, packed=True, compile_only=False):
    """``[EP_SIZE, *shape]`` routed/shared expert shards (rank axis only).

    The bounded fixture rank-broadcasts one EP16 shard for every layer, which
    keeps the compile fixture inside host memory.  ``prepare_fwd_weights``
    supplies the real per-layer shards for the checkpoint path.
    """
    value = torch.empty((EP_SIZE, *shape), dtype=dtype)
    if not compile_only:
        value.fill_(pattern[0])
        if packed:
            value = pack_nz(value)
    return value.unsqueeze(1).expand(EP_SIZE, EXPERT_LAYERS, *shape)


def _source_weight(count, shape, dtype, pattern, packed=False, compile_only=False):
    """``[EP_SIZE, count, *shape]`` per source slot, replicated over ranks."""
    value = torch.empty((EP_SIZE, count, *shape), dtype=dtype)
    if compile_only:
        return value
    for slot in range(count):
        value[:, slot].fill_(pattern[slot % len(pattern)])
    return pack_nz(value) if packed else value


def build_fwd_specs(tokens=DEFAULT_T, seed=0, compile_only=False):
    """Bounded, checkpoint-free fixture for one prefill tile and sixteen ranks.

    Shapes follow the single-program ABI: weights are stacked ``[EP_SIZE,
    N_LAYERS, ...]`` (so ``param[rank]`` is the per-rank layer stack and the
    layer loop slices ``[layer]``), caches are pooled on the layer/source axis,
    and the two layer halves share one exchange window set.
    """
    from golden import ScalarSpec, TensorSpec
    from utils import build_request_plan

    global _FIXTURE, _FIXTURE_INITIAL, _MUTABLE_NAMES

    torch.manual_seed(seed)
    start = 1248
    requests = [(start, tokens)]
    plan1 = build_request_plan(requests, ratio=1)
    plan2 = build_request_plan(requests, ratio=2)
    positions = plan1["position_ids"]
    max_position = int(positions.max())
    pages = plan1["pages"]
    batch = plan1["index_block_table"].shape[0]

    cos, sin = _rope_tables(max_position + 2)
    comp_cos, comp_sin = _rope_tables(max_position + 2, theta=M.compress_rope_theta)
    # Per-profile rope ABI: layers 0-1 (SWA) use the base profile, layers
    # 2-39 use the compressed profile; both tables are per token.
    rope_cos = comp_cos[positions]
    rope_sin = comp_sin[positions]
    rope_cos_swa = cos[positions]
    rope_sin_swa = sin[positions]
    c2a_positions = compressed_source_positions(positions, ratio=2)
    c1a_positions = (plan1["compressed_lens"].long() - 1).clamp_min(0)

    int_pattern = (1, 2, 3, 4, 5)
    bf_pattern = (0.002, 0.003, 0.004, 0.005, 0.006)
    f32_pattern = (0.02, -0.015, 0.01, -0.008, 0.005)
    scale_pattern = (1e-5, 1.1e-5, 1.2e-5, 1.3e-5, 1.4e-5)
    norm_pattern = (1.0, 1.01, 0.99, 1.02, 0.98)

    fixture = {}

    def add(name, value):
        fixture[name] = value

    # --- plan / metadata (identical on every rank) -------------------------
    # TP peers carry the same batch tokens (DP4 x TP4), so the sharded embedding
    # lookup + TP-group reduce reproduces the per-rank golden row.
    group_ids = torch.randint(0, LOCAL_VOCAB, (EP_SIZE // TP_SIZE, tokens), dtype=torch.int64)
    add("input_ids", group_ids.repeat_interleave(TP_SIZE, dim=0))
    initial_pre = torch.zeros((tokens, HC_MULT), dtype=torch.float32)
    initial_pre[:, 0] = 1.0
    add("initial_pre", _rank_repeat(initial_pre))
    embed = (
        torch.empty(LOCAL_VOCAB, D, dtype=torch.bfloat16)
        if compile_only
        else (torch.randn(LOCAL_VOCAB, D, dtype=torch.float32) * 0.02).bfloat16()
    )
    add("embed_weight", _rank_repeat(embed))
    add("lm_head_weight", fixture["embed_weight"])
    add("final_norm", _rank_repeat(
        torch.empty(D, dtype=torch.bfloat16) if compile_only
        else (torch.randn(D) * 0.1 + 1.0).bfloat16()
    ))
    for name, value in (
        ("position_ids", positions),
        ("request_ids", plan1["request_ids"]),
        ("window_slots", plan1["window_slots"]),
        ("window_indices", plan1["window_indices"]),
        ("c2a_compressed_slots", plan2["compressed_slots"]),
        ("c2a_compressed_lens", plan2["compressed_lens"]),
        ("c1a_compressed_slots", plan1["compressed_slots"]),
        ("c1a_compressed_lens", plan1["compressed_lens"]),
        ("index_block_table", plan1["index_block_table"]),
        ("state_block_table", plan1["state_block_table"]),
        ("rope_cos", rope_cos),
        ("rope_sin", rope_sin),
        ("rope_cos_swa", rope_cos_swa),
        ("rope_sin_swa", rope_sin_swa),
        ("c2a_compressed_cos", comp_cos[c2a_positions]),
        ("c2a_compressed_sin", comp_sin[c2a_positions]),
        ("c1a_compressed_cos", comp_cos[c1a_positions]),
        ("c1a_compressed_sin", comp_sin[c1a_positions]),
    ):
        add(name, _rank_repeat(value))

    # --- per-layer dense weights -------------------------------------------
    add("hc_attn_fn", _layer_weight((MIX_PAD, HC_DIM), torch.float32, f32_pattern,
                                    zero_tail=MIX_HC, compile_only=compile_only))
    add("hc_attn_scale", _layer_weight((3,), torch.float32, (0.05, 0.06, 0.07, 0.08, 0.09),
                                       compile_only=compile_only))
    add("hc_attn_base", _layer_weight((MIX_HC,), torch.float32, f32_pattern, compile_only=compile_only))
    add("attn_norm_w", _layer_weight((D,), torch.bfloat16, norm_pattern, compile_only=compile_only))
    add("wq_a", _layer_weight((D, Q_LORA), torch.int8, int_pattern, packed=True,
                              compile_only=compile_only))
    add("wq_a_scale", _layer_weight((Q_LORA,), torch.float32, scale_pattern, compile_only=compile_only))
    add("q_norm_w", _layer_weight((Q_LORA,), torch.bfloat16, norm_pattern, compile_only=compile_only))
    add("wq_b", _layer_weight((Q_LORA, LOCAL_Q), torch.int8, int_pattern, packed=True,
                              compile_only=compile_only))
    add("wq_b_scale", _layer_weight((LOCAL_Q,), torch.float32, scale_pattern,
                                    compile_only=compile_only))
    add("wkv", _layer_weight((D, HEAD_DIM), torch.int8, int_pattern, packed=True,
                             compile_only=compile_only))
    add("wkv_scale", _layer_weight((HEAD_DIM,), torch.float32, scale_pattern,
                                   compile_only=compile_only))
    add("kv_norm_w", _layer_weight((HEAD_DIM,), torch.bfloat16, norm_pattern,
                                   compile_only=compile_only))
    add("sink", _layer_weight((LOCAL_HEADS,), torch.float32, f32_pattern,
                              compile_only=compile_only))
    add("wo_a", _layer_weight((O_GROUPS, O_LORA, O_GROUP_IN), torch.bfloat16, bf_pattern,
                              compile_only=compile_only))
    add("wo_b", _layer_weight((D, O_GROUPS * O_LORA), torch.bfloat16, bf_pattern,
                              compile_only=compile_only))
    add("hc_ffn_fn", _layer_weight((MIX_PAD, HC_DIM), torch.float32, f32_pattern,
                                   zero_tail=MIX_HC, compile_only=compile_only))
    add("hc_ffn_scale", _layer_weight((3,), torch.float32, (0.05, 0.06, 0.07, 0.08, 0.09),
                                      compile_only=compile_only))
    add("hc_ffn_base", _layer_weight((MIX_HC,), torch.float32, f32_pattern,
                                     compile_only=compile_only))
    add("ffn_norm_w", _layer_weight((D,), torch.bfloat16, norm_pattern, compile_only=compile_only))
    add("gate_weight", _layer_weight((N_EXPERTS, D), torch.float32,
                                     (0.001, -0.002, 0.0015, -0.0005, 0.002),
                                     compile_only=compile_only))
    add("gate_bias", _layer_weight((N_EXPERTS,), torch.float32,
                                   (0.01, -0.02, 0.015, -0.005, 0.02), compile_only=compile_only))
    for prefix, shape, scale_shape, dtype, pattern in (
        ("shared_w1", (D, INTER), (INTER,), torch.int8, int_pattern),
        ("shared_w3", (D, INTER), (INTER,), torch.int8, int_pattern),
        ("shared_w2", (INTER, D), (D,), torch.int8, int_pattern),
    ):
        add(prefix, _layer_weight(shape, dtype, pattern, packed=True, compile_only=compile_only))
        add(prefix + "_scale", _layer_weight(scale_shape, torch.float32, scale_pattern,
                                             compile_only=compile_only))

    # --- routed experts (one rank-broadcast shard per layer) ---------------
    add("w1", _layer_expert_weight((N_LOCAL_EXPERTS, D, INTER), torch.int8, int_pattern,
                                   compile_only=compile_only))
    add("w1_scale", _layer_expert_weight((N_LOCAL_EXPERTS, INTER), torch.float32, scale_pattern,
                                         packed=False, compile_only=compile_only))
    add("w3", _layer_expert_weight((N_LOCAL_EXPERTS, D, INTER), torch.int8, int_pattern,
                                   compile_only=compile_only))
    add("w3_scale", _layer_expert_weight((N_LOCAL_EXPERTS, INTER), torch.float32, scale_pattern,
                                         packed=False, compile_only=compile_only))
    add("w2", _layer_expert_weight((N_LOCAL_EXPERTS, INTER, D), torch.int8, int_pattern,
                                   compile_only=compile_only))
    add("w2_scale", _layer_expert_weight((N_LOCAL_EXPERTS, D), torch.float32, scale_pattern,
                                         packed=False, compile_only=compile_only))

    # --- source-owned weights ---------------------------------------------
    for prefix, count, names in (
        ("c2a", len(C2A_SOURCES), (
            ("comp_wkv", (HEAD_DIM, D), torch.bfloat16, bf_pattern, False),
            ("comp_wgate", (HEAD_DIM, D), torch.bfloat16, bf_pattern, False),
            ("comp_norm", (HEAD_DIM,), torch.bfloat16, (1.0, 1.01, 0.99), False),
            ("idx_wq_b", (Q_LORA, LOCAL_INDEX_DIM), torch.int8, int_pattern, True),
            ("idx_wq_b_scale", (LOCAL_INDEX_DIM,), torch.float32, scale_pattern, False),
            ("idx_weights_proj", (D, IDX_HEADS), torch.bfloat16, bf_pattern, False),
            ("idx_wk", (IDX_DIM, HEAD_DIM), torch.bfloat16, bf_pattern, False),
            ("idx_k_norm", (IDX_DIM,), torch.bfloat16, (1.0, 1.01, 0.99), False),
        )),
        ("c1a", len(C1A_SOURCES), (
            ("comp_wkv", (HEAD_DIM, D), torch.bfloat16, bf_pattern, False),
            ("comp_norm", (HEAD_DIM,), torch.bfloat16, (1.0,), False),
            ("idx_wq_b", (Q_LORA, LOCAL_INDEX_DIM), torch.int8, int_pattern, True),
            ("idx_wq_b_scale", (LOCAL_INDEX_DIM,), torch.float32, (1.1e-5,), False),
            ("idx_weights_proj", (D, IDX_HEADS), torch.bfloat16, (0.03,), False),
            ("idx_wk", (IDX_DIM, HEAD_DIM), torch.bfloat16, (0.03,), False),
            ("idx_k_norm", (IDX_DIM,), torch.bfloat16, (1.0,), False),
        )),
        ("re", len(REINDEX_LAYERS), (
            ("idx_wq_b", (Q_LORA, LOCAL_INDEX_DIM), torch.int8, int_pattern, True),
            ("idx_wq_b_scale", (LOCAL_INDEX_DIM,), torch.float32, (1.2e-5,), False),
            ("idx_weights_proj", (D, IDX_HEADS), torch.bfloat16, (0.04,), False),
        )),
    ):
        for name, shape, dtype, pattern, packed in names:
            value = _source_weight(count, shape, dtype, pattern, packed=packed,
                                   compile_only=compile_only)
            # A single-source family (C1A) carries no source axis: the entry
            # declares ``[EP_SIZE, *shape]`` and the chain receives ``[shape]``.
            add(f"{prefix}_{name}", value.squeeze(1) if count == 1 else value)

    # --- Engram rows (host-gathered in the real path) ----------------------
    engram_lookup = (
        torch.empty(EP_SIZE, N_ENGRAM, tokens, ENGRAM_K, dtype=torch.bfloat16) if compile_only
        else (torch.randn(EP_SIZE, N_ENGRAM, tokens, ENGRAM_K) * 0.02).bfloat16()
    )
    engram_wkv = (
        torch.empty(EP_SIZE, N_ENGRAM, ENGRAM_OUT, ENGRAM_K, dtype=torch.bfloat16) if compile_only
        else (torch.randn(EP_SIZE, N_ENGRAM, ENGRAM_OUT, ENGRAM_K) * 0.02).bfloat16()
    )
    engram_qk = (
        torch.empty(EP_SIZE, N_ENGRAM, 4, D, dtype=torch.float32) if compile_only
        else torch.randn(EP_SIZE, N_ENGRAM, 4, D) * 0.1
    )
    add("engram_lookup", engram_lookup)
    add("engram_wkv", engram_wkv)
    add("engram_qk", engram_qk)
    add("engram_rotation", _rank_repeat(fixture_rotation()))
    add("engram_enabled", torch.tensor(1, dtype=torch.int32))

    # --- pooled caches and selection slots --------------------------------
    add("window_cache",
        torch.zeros(EP_SIZE, N_LAYERS, pages, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16))
    add("compressed_cache_c2a",
        torch.zeros(EP_SIZE, len(C2A_SOURCES), pages, BLOCK_SIZE, 1, HEAD_DIM,
                    dtype=torch.bfloat16))
    add("index_cache_c2a",
        torch.zeros(EP_SIZE, len(C2A_SOURCES), pages, BLOCK_SIZE, 1, IDX_DIM,
                    dtype=torch.bfloat16))
    add("state_cache_c2a",
        torch.zeros(EP_SIZE, len(C2A_SOURCES), batch, STATE_CAPACITY, STATE_WIDTH,
                    dtype=torch.float32))
    add("compressed_cache_c1a",
        torch.zeros(EP_SIZE, pages, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16))
    add("index_cache_c1a",
        torch.zeros(EP_SIZE, pages, BLOCK_SIZE, 1, IDX_DIM, dtype=torch.bfloat16))
    for family, slots in (("c2a", len(C2A_SOURCES)), ("c1a", N_C1A_SLOTS)):
        add(f"topk_indices_{family}",
            torch.full((EP_SIZE, slots, tokens, INDEX_TOPK), -1, dtype=torch.int32))
        add(f"topk_values_{family}",
            torch.zeros(EP_SIZE, slots, tokens, INDEX_TOPK, dtype=torch.float32))
        add(f"pool_block_ids_{family}",
            torch.full((EP_SIZE, slots, tokens, CANDIDATE_POOL_BLOCKS), -1, dtype=torch.int32))
        add(f"pool_block_counts_{family}",
            torch.zeros(EP_SIZE, slots, tokens, dtype=torch.int32))

    # --- running / output buffers -----------------------------------------
    add("token_counts", torch.full((EP_SIZE,), tokens, dtype=torch.int32))
    add("x_hc", torch.zeros(EP_SIZE, tokens, HC_MULT, D, dtype=torch.float32))
    add("pre_mix", torch.zeros(EP_SIZE, tokens, HC_MULT, dtype=torch.float32))
    add("layer_x_hc", torch.zeros(EP_SIZE, N_LAYERS, tokens, HC_MULT, D, dtype=torch.float32))
    add("layer_pre_mix", torch.zeros(EP_SIZE, N_LAYERS, tokens, HC_MULT, dtype=torch.float32))
    add("final_hidden", torch.zeros(EP_SIZE, tokens, D, dtype=torch.bfloat16))
    add("logits", torch.zeros(EP_SIZE, tokens, LOCAL_VOCAB, dtype=torch.float32))

    _FIXTURE = fixture
    _MUTABLE_NAMES = tuple(name for name in MUTABLE_NAMES if name in fixture)
    _FIXTURE_INITIAL = {
        name: (value.clone() if name in _MUTABLE_NAMES else value) for name, value in fixture.items()
    }
    return [ScalarSpec(name, torch.int32, value, compile_runtime=True) if name == "engram_enabled"
            else TensorSpec(name, list(value.shape), value.dtype, init_value=value)
            for name, value in fixture.items()]


# ---------------------------------------------------------------------------
# Composed golden and comparators
# ---------------------------------------------------------------------------


def _layer_source_arguments(inputs, layer, mode):
    """Source-owned weights and compressed metadata for one layer (rank axis)."""
    if mode in C2A_MODES:
        slot = _c2a_slot(layer)
        return {
            "compressor_wkv": inputs["c2a_comp_wkv"][:, slot],
            "compressor_wgate": inputs["c2a_comp_wgate"][:, slot],
            "compressor_norm": inputs["c2a_comp_norm"][:, slot],
            "index_wq_b": inputs["c2a_idx_wq_b"][:, slot],
            "index_wq_b_scale": inputs["c2a_idx_wq_b_scale"][:, slot],
            "index_weights_proj": inputs["c2a_idx_weights_proj"][:, slot],
            "index_wk": inputs["c2a_idx_wk"][:, slot],
            "index_norm_w": inputs["c2a_idx_k_norm"][:, slot],
            "compressed_lens": inputs["c2a_compressed_lens"],
            "compressed_slots": inputs["c2a_compressed_slots"],
            "compressed_cos": inputs["c2a_compressed_cos"],
            "compressed_sin": inputs["c2a_compressed_sin"],
        }
    arguments = {
        "compressor_wkv": inputs["c1a_comp_wkv"],
        # The c1a compressor has no separate gate; the prolog passes wkv twice.
        "compressor_wgate": inputs["c1a_comp_wkv"],
        "compressor_norm": inputs["c1a_comp_norm"],
        "index_wk": inputs["c1a_idx_wk"],
        "index_norm_w": inputs["c1a_idx_k_norm"],
        "compressed_lens": inputs["c1a_compressed_lens"],
        "compressed_slots": inputs["c1a_compressed_slots"],
        "compressed_cos": inputs["c1a_compressed_cos"],
        "compressed_sin": inputs["c1a_compressed_sin"],
    }
    if mode == MODE_C1A_REINDEX:
        slot = REINDEX_LAYERS.index(layer)
        arguments["index_wq_b"] = inputs["re_idx_wq_b"][:, slot]
        arguments["index_wq_b_scale"] = inputs["re_idx_wq_b_scale"][:, slot]
        arguments["index_weights_proj"] = inputs["re_idx_weights_proj"][:, slot]
    else:
        arguments["index_wq_b"] = inputs["c1a_idx_wq_b"]
        arguments["index_wq_b_scale"] = inputs["c1a_idx_wq_b_scale"]
        arguments["index_weights_proj"] = inputs["c1a_idx_weights_proj"]
    return arguments


def _layer_reference_dict(inputs, layer):
    """Per-layer reference dict assembled from the pooled fixture tensors."""
    mode, ratio, has_engram = _layer_family(layer)
    has_engram = has_engram and int(inputs.get("engram_enabled", 1)) != 0
    values = {name: inputs[name][:, layer] for name in DENSE_LAYER_NAMES}
    values["token_counts"] = inputs.get("token_counts")
    values.update({name: inputs[name][:, min(layer, EXPERT_LAYERS - 1)] for name in EXPERT_NAMES})
    values.update(_layer_source_arguments(inputs, layer, mode))
    engram_slot = M.engram_layer_ids.index(layer) if has_engram else 0
    values["engram_lookup"] = inputs["engram_lookup"][:, engram_slot]
    values["engram_wkv"] = inputs["engram_wkv"][:, engram_slot]
    values["engram_qk"] = inputs["engram_qk"][:, engram_slot]
    values["engram_rotation"] = inputs["engram_rotation"]
    values["window_cache"] = inputs["window_cache"][:, layer]
    if mode in C2A_MODES:
        slot = _c2a_slot(layer)
        values["compressed_cache"] = inputs["compressed_cache_c2a"][:, slot]
        values["index_cache"] = inputs["index_cache_c2a"][:, slot]
        values["state_cache"] = inputs["state_cache_c2a"][:, slot]
        family, selection_slot = "c2a", slot
    else:
        values["compressed_cache"] = inputs["compressed_cache_c1a"]
        values["index_cache"] = inputs["index_cache_c1a"]
        # The C1A chains carry no compressor state; slot 0 keeps the golden's
        # contract identical to the validated layer suite.
        values["state_cache"] = inputs["state_cache_c2a"][:, 0]
        family, selection_slot = "c1a", _c1a_slot(layer)
    for name in ("topk_indices", "topk_values", "pool_block_ids", "pool_block_counts"):
        source_slot = 0 if mode == MODE_C1A_REINDEX and name.startswith("pool_") else selection_slot
        values[name] = inputs[f"{name}_{family}"][:, source_slot]
    values["window_slots"] = inputs["window_slots"]
    values["window_indices"] = inputs["window_indices"]
    values["request_ids"] = inputs["request_ids"]
    values["position_ids"] = inputs["position_ids"]
    values["index_block_table"] = inputs["index_block_table"]
    values["state_block_table"] = inputs["state_block_table"]
    if layer < 2:
        values["rope_cos"] = inputs["rope_cos_swa"]
        values["rope_sin"] = inputs["rope_sin_swa"]
    else:
        values["rope_cos"] = inputs["rope_cos"]
        values["rope_sin"] = inputs["rope_sin"]
    return values, mode, ratio, has_engram


def _group_outputs(values, tokens):
    """Regroup the pooled caches/slots into the validated comparison shapes."""
    outputs = {}
    outputs["window_cache"] = values["window_cache"].reshape(EP_SIZE, -1, BLOCK_SIZE, 1, HEAD_DIM)
    outputs["compressed_cache_c2a"] = values["compressed_cache_c2a"].reshape(
        EP_SIZE, -1, BLOCK_SIZE, 1, HEAD_DIM)
    outputs["index_cache_c2a"] = values["index_cache_c2a"].reshape(
        EP_SIZE, -1, BLOCK_SIZE, 1, IDX_DIM)
    outputs["state_cache_c2a"] = values["state_cache_c2a"].reshape(
        EP_SIZE, -1, STATE_CAPACITY, STATE_WIDTH)
    outputs["compressed_cache_c1a"] = values["compressed_cache_c1a"]
    outputs["index_cache_c1a"] = values["index_cache_c1a"]
    for family in ("c2a", "c1a"):
        for name in ("topk_indices", "topk_values", "pool_block_ids", "pool_block_counts"):
            outputs[f"{name}_{family}"] = values[f"{name}_{family}"]
    return outputs


def _run_composed_golden(inputs, tokens, staged_x_hc=None, staged_pre_mix=None):
    """Chain the per-layer goldens over all forty layers on the paged plan."""
    from embedding import golden_embedding
    from hc import golden_hc_head
    from lm_head import golden_lm_head

    counts = reference_active_counts(inputs)
    if bool((counts != tokens).any()):
        raise ValueError("active-S prefill reference is outside the bounded decode draft")
    state = dict(inputs)
    for name in _MUTABLE_NAMES:
        state[name] = inputs[name].clone()

    embedding = torch.zeros((EP_SIZE, tokens, D), dtype=torch.bfloat16)
    for rank in range(EP_SIZE):
        golden_embedding({
            "input_ids": state["input_ids"][rank].clone(),
            "embed_weight": state["embed_weight"][rank],
            "output": embedding[rank],
        })
    x_hc = embedding.float().reshape(EP_SIZE, tokens, 1, D).repeat(1, 1, HC_MULT, 1).contiguous()
    pre_mix = state["initial_pre"].clone()

    snapshots = torch.zeros((EP_SIZE, N_LAYERS, tokens, HC_MULT, D), dtype=torch.float32)
    pre_snapshots = torch.zeros((EP_SIZE, N_LAYERS, tokens, HC_MULT), dtype=torch.float32)
    for layer in range(N_LAYERS):
        if staged_x_hc is not None and layer > 0:
            x_hc = staged_x_hc[:, layer - 1].clone()
            pre_mix = staged_pre_mix[:, layer - 1].clone()
        values, mode, ratio, has_engram = _layer_reference_dict(state, layer)
        values["x_hc"] = x_hc
        values["pre_mix"] = pre_mix
        values["hidden"] = torch.zeros(EP_SIZE, tokens, D, dtype=torch.bfloat16)
        values["attn_input"] = torch.zeros(EP_SIZE, tokens, D, dtype=torch.bfloat16)
        values["attn_out"] = torch.zeros(EP_SIZE, tokens, D, dtype=torch.bfloat16)
        values["x_hc_mid"] = torch.zeros(EP_SIZE, tokens, HC_MULT, D, dtype=torch.float32)
        values["ffn_input"] = torch.zeros(EP_SIZE, tokens, D, dtype=torch.bfloat16)
        values["ffn_owned"] = torch.zeros(EP_SIZE, tokens, D, dtype=torch.bfloat16)
        values["ffn_normed"] = torch.zeros(EP_SIZE, tokens, D, dtype=torch.bfloat16)
        values["route_indices"] = torch.full((EP_SIZE, tokens, TOPK), -1, dtype=torch.int32)
        values["route_weights"] = torch.zeros(EP_SIZE, tokens, TOPK, dtype=torch.float32)
        values["x_hc_out"] = torch.zeros(EP_SIZE, tokens, HC_MULT, D, dtype=torch.float32)
        values["next_pre_mix"] = torch.zeros(EP_SIZE, tokens, HC_MULT, dtype=torch.float32)
        make_layer_golden(mode, has_engram, ratio)(values)
        x_hc = values["x_hc_out"]
        pre_mix = values["next_pre_mix"]
        snapshots[:, layer] = x_hc
        pre_snapshots[:, layer] = pre_mix

    final_hidden = torch.zeros((EP_SIZE, tokens, D), dtype=torch.bfloat16)
    for rank in range(EP_SIZE):
        golden_hc_head({
            "x_hc": x_hc[rank],
            "pre_mix": pre_mix[rank],
            "output": final_hidden[rank],
        })
    logits = torch.zeros((EP_SIZE, tokens, LOCAL_VOCAB), dtype=torch.float32)
    for rank in range(EP_SIZE):
        golden_lm_head({
            "hidden": final_hidden[rank],
            "norm_w": state["final_norm"][rank],
            "head_weight": state["lm_head_weight"][rank],
            "logits": logits[rank],
        })

    outputs = {
        "layer_x_hc": snapshots,
        "layer_pre_mix": pre_snapshots,
        "logits": logits,
    }
    outputs.update(_group_outputs(state, tokens))
    return outputs


def make_fwd_comparators(tokens=DEFAULT_T):
    """Staged per-layer gates plus the paged cache/state handoff checks."""
    from golden import ratio_allclose
    from indexer import index_topk_compare, pool_set_compare

    cache = {}

    def staged(actual_outputs):
        # Every comparator receives the same output dict; recompute once.
        if cache.get("actual") is not actual_outputs:
            cache["value"] = _run_composed_golden(
                _FIXTURE_INITIAL, tokens,
                staged_x_hc=actual_outputs["layer_x_hc"],
                staged_pre_mix=actual_outputs["layer_pre_mix"],
            )
            cache["actual"] = actual_outputs
        return cache["value"]

    def staged_compare(name, gate):
        def compare(actual, expected, *, actual_outputs, inputs, rtol, atol, **_kwargs):
            reference = staged(actual_outputs)[name]
            return gate(
                actual, reference, actual_outputs=actual_outputs,
                expected_outputs={name: reference}, inputs=inputs, rtol=rtol, atol=atol,
            )

        compare.__name__ = f"staged_{name}"
        return compare

    def direct(gate):
        def compare(actual, expected, **kwargs):
            return gate(actual, expected, **kwargs)

        return compare

    def pool_compare(actual, expected, **_kwargs):
        return pool_set_compare()(
            actual.reshape(-1, actual.shape[-1]), expected.reshape(-1, expected.shape[-1]))

    cache_gate = ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2)
    comparator = {
        "layer_x_hc": staged_compare(
            "layer_x_hc", ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2)),
        "layer_pre_mix": staged_compare(
            "layer_pre_mix", ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=5e-3)),
        "logits": staged_compare(
            "logits", ratio_allclose(atol=2e-2, rtol=2e-2, max_error_ratio=5e-3)),
        "window_cache": staged_compare("window_cache", cache_gate),
        "compressed_cache_c2a": staged_compare("compressed_cache_c2a", cache_gate),
        "compressed_cache_c1a": staged_compare("compressed_cache_c1a", cache_gate),
        "index_cache_c2a": staged_compare("index_cache_c2a", cache_gate),
        "index_cache_c1a": staged_compare("index_cache_c1a", cache_gate),
        # Calibrated to the child c2a_compressor's own state gate (atol/rtol
        # 1e-4); the composed reference recomputes the compressor input through
        # the full golden, so it cannot be tighter than the accepted child.
        # Independent floor: a 1-BF16-ULP compressor-input perturbation moves the
        # state by ~3.2e-4 mean / 1.5e-3 max at state scale 0.055 (~6e-3
        # relative), far above the observed 2.3e-5 relative residual.
        "state_cache_c2a": staged_compare(
            "state_cache_c2a", ratio_allclose(atol=1e-4, rtol=1e-4, max_error_ratio=5e-3)),
        "topk_values_c2a": staged_compare(
            "topk_values_c2a", ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2)),
        "topk_values_c1a": staged_compare(
            "topk_values_c1a", ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2)),
        "pool_block_ids_c2a": pool_compare,
        "pool_block_ids_c1a": pool_compare,
        "pool_block_counts_c2a": direct(ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)),
        "pool_block_counts_c1a": direct(ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)),
    }
    for name, paired in (("topk_indices_c2a", "topk_values_c2a"),
                         ("topk_indices_c1a", "topk_values_c1a")):
        gate = index_topk_compare(paired)

        def compare(actual, expected, *, actual_outputs, inputs, rtol, atol,
                    _gate=gate, _paired=paired, **_kwargs):
            reference = staged(actual_outputs)[
                "topk_indices_c2a" if _paired == "topk_values_c2a" else "topk_indices_c1a"]
            return _gate(
                actual, reference, actual_outputs=actual_outputs,
                expected_outputs={_paired: staged(actual_outputs)[_paired],
                                  "topk_indices_c2a" if _paired == "topk_values_c2a"
                                  else "topk_indices_c1a": reference},
                inputs=inputs, rtol=rtol, atol=atol,
            )

        compare.__name__ = f"staged_{name}"
        comparator[name] = compare
    return comparator


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _device_outputs(tensors, tokens):
    outputs = {
        "layer_x_hc": tensors["layer_x_hc"],
        "layer_pre_mix": tensors["layer_pre_mix"],
        "logits": tensors["logits"],
    }
    outputs.update(_group_outputs(tensors, tokens))
    return outputs


def main() -> int:
    from golden.validation import validate_golden
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description=__doc__)
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(EP_SIZE)))
    parser.add_argument("--tokens", type=int, default=DEFAULT_T)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engram-disabled", action="store_true", help="validate the temporary noEngram path")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--weight-check", action="store_true", help="host-only checkpoint contract check")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--check-layers", default="0,1,2,14,20,24,39")
    args = parser.parse_args()
    if args.tokens not in SUPPORTED_TILES:
        raise SystemExit(f"supported bounded tiles are {SUPPORTED_TILES}, got {args.tokens}")
    if args.weight_check:
        from weight_loader import WeightLoader

        loader = WeightLoader(args.checkpoint) if args.checkpoint else WeightLoader()
        layers = [int(item) for item in args.check_layers.split(",") if item]
        raise SystemExit(check_weight_contract(loader, layers))
    device_ids = [int(value) for value in args.device.split(",") if value != ""]
    if len(device_ids) != EP_SIZE:
        raise SystemExit(f"need exactly {EP_SIZE} device ids for EP{EP_SIZE}, got {device_ids}")

    torch.manual_seed(args.seed)
    specs = build_fwd_specs(args.tokens, args.seed, compile_only=args.compile_only)
    tensors = {spec.name: (spec.value if isinstance(spec, ScalarSpec) else spec.create_tensor()) for spec in specs}
    if args.engram_disabled:
        tensors["engram_enabled"] = torch.tensor(0, dtype=torch.int32)
        _FIXTURE_INITIAL["engram_enabled"] = torch.tensor(0, dtype=torch.int32)
    config = dict(
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        backend_type=args.backend_type,
        execution_mode=args.execution_mode,
    )
    if args.compile_only:
        run_prefill_fwd(tensors, config, compile_only=True)
        print("[PREFILL_FWD] single-program compile PASS")
        return 0

    golden = _run_composed_golden(_FIXTURE_INITIAL, args.tokens)
    run_prefill_fwd(tensors, config)
    validate_golden(
        _device_outputs(tensors, args.tokens), golden,
        rtol=2e-2, atol=2e-2, compare_fn=make_fwd_comparators(args.tokens),
        inputs=_FIXTURE_INITIAL,
    )
    print("[PREFILL_FWD] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
