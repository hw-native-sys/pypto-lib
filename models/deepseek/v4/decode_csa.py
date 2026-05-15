# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 decode layer for the CSA (`compress_ratio == 4`) path.
Composes ``attention_csa`` (hc_pre + qkv_proj + main compressor + indexer +
sparse_attn + hc_post) with the MoE stack, matching ``Block.forward``
(model.py:688-700) for layers whose ``compress_ratio == 4`` (the ratio=4
layers in the demo / flash compress_ratios tuple). Inherits the CSA caller
contract: runtime ``start_pos`` MUST equal compile-time ``START_POS`` AND
``(START_POS + 1) % COMPRESS_RATIO == 0`` AND ``START_POS >= WIN - 1`` —
see attention_csa.py."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    EP_WORLD_SIZE,
    RECV_MAX,
)
from attention_csa import attention_csa
from moe import moe


# ---- shared model/decode constants (mirror attention_csa.py + moe.py) ----
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# ---- attention_csa-local constants ----
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)

IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk

MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO

SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK
SPARSE_TOPK = WIN + IDX_TOPK

ORI_MAX_BLOCKS = 1
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 64
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

START_POS = 127

Q_PROJ_OUT_CHUNK = 128
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK

# Caller contract — same as attention_csa.py.
SHOULD_COMPRESS = ((START_POS + 1) % COMPRESS_RATIO) == 0
assert SHOULD_COMPRESS and START_POS >= WIN - 1, (
    f"CSA fixture requires a full-window compression step; got START_POS={START_POS}, "
    f"COMPRESS_RATIO={COMPRESS_RATIO}, WIN={WIN}."
)

# ---- moe-local constants ----
N_EXPERTS = M.n_routed_experts
TOPK_E = M.num_experts_per_tok
VOCAB = M.vocab_size
MOE_INTER = M.moe_intermediate_size
N_LOCAL_EXPERTS = N_EXPERTS // EP_WORLD_SIZE
assert RECV_MAX >= T * TOPK_E, "packed layout needs RECV_MAX >= T * TOPK_E"


@pl.jit.inline
def decode_csa(
    # ---- layer input (HC stack) ----
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    # ---- indexer-side residual stream (model.py: indexer reads pre-norm activations) ----
    x_indexer: pl.Tensor[[B, S, D], pl.BF16],
    # ---- attention hc_pre weights ----
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # ---- attention qkv_proj_rope weights ----
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[HALF_ROPE, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[HALF_ROPE, ROPE_HEAD_DIM], pl.BF16],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, HALF_ROPE], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, HALF_ROPE], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    # ---- indexer's pre-computed qr (model.py: q-side latent that drives index scoring) ----
    qr_indexer: pl.Tensor[[B, S, Q_LORA], pl.BF16],
    # ---- main compressor (ratio=4, overlap=True, rotate=False) ----
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_kv_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    # ---- indexer weights + hadamard ----
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    # ---- inner compressor (ratio=4, overlap=True, rotate=True) ----
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    # ---- attention KV cache (split into ori + cmp pools) ----
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    # ---- indexer-cached compressed KV (written by inner compressor, read for scoring) ----
    idx_kv_cache: pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16],
    # ---- sparse_attn ----
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    # ---- o_proj weights ----
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    # ---- moe hc_pre + router weights ----
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.FP32],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK_E], pl.INT32],
    input_ids: pl.Tensor[[B, S], pl.INT64],
    # ---- moe expert weights ----
    expert_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    expert_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_expert_count_full: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    # ---- output ----
    x_next: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    # ---- scalars ----
    start_pos: pl.Scalar[pl.INT32],   # MUST equal compile-time START_POS — see header
    layer_id: pl.Scalar[pl.INT32],
):
    # Attention sub-block (CSA): hc_pre + attention(+main compressor + indexer) + hc_post → x_attn.
    x_attn = pl.create_tensor([B, S, HC_MULT, D], dtype=pl.BF16)
    x_attn = attention_csa(
        x_hc, x_indexer,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, even_select_t, odd_select_t,
        even_select, odd_select,
        even_select_local, odd_select_local,
        qr_indexer,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_kv_state, cmp_score_state,
        idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_kv_state, inner_score_state,
        kv_cache, ori_block_table, cmp_kv, cmp_block_table,
        idx_kv_cache,
        attn_sink, seqused_kv,
        wo_a, wo_b, wo_b_scale,
        x_attn,
        start_pos,
    )

    # MoE sub-block.
    x_next = moe(
        x_attn,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias,
        tid2eid, input_ids,
        expert_w1, expert_w1_scale, expert_w3, expert_w3_scale,
        expert_w2, expert_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        recv_expert_count_full,
        x_next,
        layer_id,
    )
    return x_next


@pl.jit
def decode_csa_test(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    x_indexer: pl.Tensor[[B, S, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[HALF_ROPE, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[HALF_ROPE, ROPE_HEAD_DIM], pl.BF16],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, HALF_ROPE], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, HALF_ROPE], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    qr_indexer: pl.Tensor[[B, S, Q_LORA], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_kv_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.FP32],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK_E], pl.INT32],
    input_ids: pl.Tensor[[B, S], pl.INT64],
    expert_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    expert_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    expert_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    expert_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    recv_expert_count_full: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    x_next: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
):
    x_next = decode_csa(
        x_hc, x_indexer,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, even_select_t, odd_select_t,
        even_select, odd_select,
        even_select_local, odd_select_local,
        qr_indexer,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_kv_state, cmp_score_state,
        idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_kv_state, inner_score_state,
        kv_cache, ori_block_table, cmp_kv, cmp_block_table,
        idx_kv_cache,
        attn_sink, seqused_kv,
        wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias,
        tid2eid, input_ids,
        expert_w1, expert_w1_scale, expert_w3, expert_w3_scale,
        expert_w2, expert_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        recv_expert_count_full,
        x_next,
        start_pos, layer_id,
    )
    return x_next


def golden_decode_csa(tensors):
    """Chains golden_attention_csa and golden_moe (decode branch)."""
    import torch

    from attention_csa import golden_attention_csa
    from moe import golden_moe

    x_attn = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    attn_tensors = dict(tensors)
    attn_tensors["x_out"] = x_attn
    # ``cmp_sparse_indices`` is consumed inside golden_attention_csa as both
    # an indexer output and a sparse_attn input; provide a fresh buffer.
    attn_tensors["cmp_sparse_indices"] = torch.full(
        (T, SPARSE_TOPK), -1, dtype=torch.int32,
    )
    golden_attention_csa(attn_tensors)

    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    golden_moe(moe_tensors)


def build_tensor_specs(layer_id: int = 0):
    """Merges attention_csa and moe specs and reorders them to match the
    positional parameter order of ``decode_csa_test``. The harness binds
    dummy_args positionally, so spec order is load-bearing.

    Drops:
      - attn ``x_out``               (intermediate, not exposed)
      - attn ``cmp_sparse_indices``  (internal indexer<->sparse_attn handoff)
      - moe  ``x_hc``                (provided by attn output)
    """
    from attention_csa import build_tensor_specs as build_attn_specs
    from moe import build_tensor_specs as build_moe_specs

    by_name = {}
    for s in build_attn_specs():
        by_name[s.name] = s
    for s in build_moe_specs(layer_id=layer_id):
        by_name.setdefault(s.name, s)

    order = [
        # ---- attention inputs ----
        "x_hc", "x_indexer",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv",
        "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin", "even_select_t", "odd_select_t",
        "even_select", "odd_select",
        "even_select_local", "odd_select_local",
        "qr_indexer",
        "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
        "cmp_kv_state", "cmp_score_state",
        "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
        "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
        "inner_kv_state", "inner_score_state",
        "kv_cache", "ori_block_table", "cmp_kv", "cmp_block_table",
        "idx_kv_cache",
        "attn_sink", "seqused_kv",
        "wo_a", "wo_b", "wo_b_scale",
        # ---- moe inputs ----
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
        "norm_w", "gate_w", "gate_bias",
        "tid2eid", "input_ids",
        "expert_w1", "expert_w1_scale",
        "expert_w3", "expert_w3_scale",
        "expert_w2", "expert_w2_scale",
        "shared_w1", "shared_w1_scale",
        "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
        "recv_expert_count_full",
        # ---- output ----
        "x_next",
        # ---- scalars ----
        "start_pos", "layer_id",
    ]

    missing = [n for n in order if n not in by_name]
    if missing:
        raise RuntimeError(f"build_tensor_specs: missing specs for {missing}")
    return [by_name[n] for n in order]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import RunConfig, ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=decode_csa_test,
        specs=build_tensor_specs(layer_id=args.layer_id),
        golden_fn=golden_decode_csa,
        config=RunConfig(
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                # Two-stage composition (attention_csa + moe) accumulates BF16 error.
                "x_next": ratio_allclose(atol=5e-2, rtol=2.0 / 128, max_error_ratio=0.02),
            },
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
