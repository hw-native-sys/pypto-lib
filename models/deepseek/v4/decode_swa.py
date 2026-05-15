# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 decode layer for the SWA (`compress_ratio == 0`) path.
Composes ``attention_swa`` (hc_pre + qkv_proj + sparse_attn + hc_post) with
the MoE stack (hc_pre + router + dispatch + expert + combine + hc_post),
matching ``Block.forward`` (model.py:688-700) for layers whose
``compress_ratio == 0`` (layers 0/1/7 in the demo)."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    EP_WORLD_SIZE,
    RECV_MAX,
)
from attention_swa import attention_swa
from moe import moe


# ---- shared model/decode constants (mirror attention_swa.py + moe.py) ----
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# ---- attention_swa-local constants ----
ORI_MAX_BLOCKS = 1
MAX_BLOCKS = ORI_MAX_BLOCKS
BLOCK_NUM = B * MAX_BLOCKS
START_POS = 127

Q_PROJ_OUT_CHUNK = 128
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK

# ---- moe-local constants ----
N_EXPERTS = M.n_routed_experts
TOPK_E = M.num_experts_per_tok          # router topk (experts/token)
VOCAB = M.vocab_size
MOE_INTER = M.moe_intermediate_size
N_LOCAL_EXPERTS = N_EXPERTS // EP_WORLD_SIZE
assert RECV_MAX >= T * TOPK_E, "packed layout needs RECV_MAX >= T * TOPK_E"


@pl.jit.inline
def decode_swa(
    # ---- layer input (HC stack) ----
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
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
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    # ---- attention KV cache (sliding-window only) ----
    kv_cache: pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
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
    start_pos: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
):
    # Attention sub-block: hc_pre + attention + hc_post → x_attn (HC stack).
    x_attn = pl.create_tensor([B, S, HC_MULT, D], dtype=pl.BF16)
    x_attn = attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, even_select_t, odd_select_t,
        even_select_local, odd_select_local,
        kv_cache, block_table,
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
def decode_swa_test(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
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
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    kv_cache: pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
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
    x_next = decode_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, even_select_t, odd_select_t,
        even_select_local, odd_select_local,
        kv_cache, block_table,
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


def golden_decode_swa(tensors):
    """Chains golden_attention_swa and golden_moe (decode branch)."""
    import torch

    from attention_swa import golden_attention_swa
    from moe import golden_moe

    # Stage A: attention_swa writes its HC output to a local intermediate.
    x_attn = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    attn_tensors = dict(tensors)
    attn_tensors["x_out"] = x_attn
    golden_attention_swa(attn_tensors)

    # Stage B: feed x_attn into MoE; final layer output goes to tensors["x_next"].
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    golden_moe(moe_tensors)


def build_tensor_specs(layer_id: int = 0):
    """Merges attention_swa and moe specs and reorders them to match the
    positional parameter order of ``decode_swa_test``. The harness binds
    dummy_args positionally, so spec order is load-bearing.

    Drops:
      - attn ``x_out``  (intermediate, not exposed)
      - moe ``x_hc``    (provided by attn output)
    """
    from attention_swa import build_tensor_specs as build_attn_specs
    from moe import build_tensor_specs as build_moe_specs

    by_name = {}
    for s in build_attn_specs():
        by_name[s.name] = s
    for s in build_moe_specs(layer_id=layer_id):
        by_name.setdefault(s.name, s)

    order = [
        # ---- attention inputs ----
        "x_hc",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv",
        "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin", "even_select_t", "odd_select_t",
        "even_select_local", "odd_select_local",
        "kv_cache", "block_table",
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
        fn=decode_swa_test,
        specs=build_tensor_specs(layer_id=args.layer_id),
        golden_fn=golden_decode_swa,
        config=RunConfig(
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                # Two-stage composition (attention_swa + moe) accumulates BF16 error;
                # bump max_error_ratio above the per-stage default.
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
