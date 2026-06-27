# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI marker: run on >=2 NPUs via $DEVICE_RANGE instead of single $DEVICE_ID
"""DeepSeek-V4 MoE single-layer (decode), FLASH preset. --ep picks the EP world
size: 1 runs single-card (in-card scatter), 2/4/8 run N-rank distributed; each
rank keeps 32 experts."""


# Sub-kernels freeze EP_WORLD_SIZE / n_routed_experts into their shapes at import
# time, so read --ep from argv and override config before importing them below.
import dataclasses
import sys

import config

_EP_CHOICES = (1, 2, 4, 8)
_EP_DEFAULT = 2


def _parse_ep_argv():
    for i, tok in enumerate(sys.argv):
        if tok == "--ep" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith("--ep="):
            return int(tok.split("=", 1)[1])
    return _EP_DEFAULT


EP = _parse_ep_argv()
# EP==1 is a single-card degenerate world: routing is global-but-local (one rank
# owns every expert), so the cross-rank HCCL dispatch/combine collapse to in-card
# scatter/gather. That path uses the local dispatch/combine and a plain single-card
# run instead of the @pl.jit.host distributed orchestration (the *_ep1 defs below).
config.EP_WORLD_SIZE = EP
config.EP_ROUTING_GLOBAL = True
config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=config.FLASH.n_routed_experts // 8 * EP)  # 32 experts/rank

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX
from hc_pre import hc_pre
from hc_post import hc_post
from gate import gate
from expert_shared import expert_shared
from expert_routed import expert_routed
from dispatch import dispatch, dispatch_ep1
from combine import combine, combine_ep1


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size

HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
MOE_INTER = M.moe_intermediate_size

N_RANKS = EP_WORLD_SIZE
N_EXPERTS_GLOBAL = M.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS
N_ROUTES = T * TOPK

# Padding widths required by tile vector ops (32 B minimum tile).
W_PAD = 8   # FP32 weight/scale tile width
IDX_PAD = 8  # INT32 r_route tile width

assert N_RANKS in _EP_CHOICES, f"--ep must be one of {_EP_CHOICES} (got {N_RANKS})"
assert N_EXPERTS_GLOBAL == N_RANKS * N_LOCAL


@pl.jit.inline
def moe(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # final output
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    # windows
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # Caller-maintained 1-based MoE invocation id for shared done windows.
    # It is independent from `layer_id`: layer_id selects weights/routing, while
    # moe_epoch orders the communication protocol over reused windows.
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    # All non-output intermediates allocate locally so the convert pass sees
    # them in the same scope as their producer / consumer, mirroring the
    # single-card moe_ep1 @pl.jit.inline composition.
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn, comb_ffn,
    )

    x_norm = pl.create_tensor([T, D], dtype=pl.BF16)
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, tid2eid, input_ids,
        x_norm, x_norm_i8, x_norm_scale, indices, weights,
    )

    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    # Pass the same epoch through dispatch and combine so their stage-specific
    # done counters describe the same MoE invocation.
    dispatch(
        indices, x_norm_i8, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out, recv_count_out,
        pub_counts, count_done, data_done,
        recv_x, recv_scale, recv_w, recv_r_route,
        num_tokens, my_rank, moe_epoch,
    )

    recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
    expert_routed(
        recv_x_out, recv_scale_out, recv_w_out, recv_count_out,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )

    ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    combine(
        recv_y, recv_r_route_out, sh,
        ffn_out,
        pub_counts, routed_y_buf, combine_done,
        num_tokens, my_rank, moe_epoch,
    )

    hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next


@pl.jit
def moe_test(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # final output
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    # windows
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # Single-layer tests pass 1; multi-layer callers must pass a monotonically
    # increasing MoE call id when they reuse the same done windows.
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    moe(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next,
        pub_counts, count_done, data_done,
        recv_x, recv_scale, recv_w, recv_r_route,
        routed_y_buf, combine_done,
        layer_id, pl.const(T, pl.INT32), my_rank, moe_epoch,
    )
    return x_next


@pl.jit.host
def l3_moe(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16]],
    layer_id: pl.Scalar[pl.INT32],
):
    pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * N_LOCAL * 4)
    count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    data_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)  # INT8 (b8 fixed in ptoas v0.45)
    recv_scale_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)  # FP32
    recv_w_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)  # FP32
    recv_r_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)  # INT32
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)  # BF16
    combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)

    for r in pl.range(pld.world_size()):
        pub_counts = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
        count_done = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_done = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_w = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_r_route = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_done = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        moe_test(
            x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            x_next[r],
            pub_counts, count_done, data_done,
            recv_x, recv_scale, recv_w, recv_r_route,
            routed_y_buf, combine_done,
            layer_id, r, pl.const(1, pl.INT32),
            device=r,
        )


# === Golden + test ==========================================================
def golden_moe(tensors):
    """Per-rank torch reference. Replays the 4 stages on host. Each rank's
    output depends only on its own inputs because the dispatch+combine round-
    trip is r_route-keyed and shape-preserving (test_l3 pattern)."""
    import torch

    from hc_pre import golden_hc_pre
    from hc_post import golden_hc_post
    from gate import golden_gate_core
    from expert_shared import golden_expert_shared
    from expert_routed import golden_expert_routed

    x_next_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.bfloat16)
    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    for r in range(N_RANKS):
        # Stage 1: hc_pre
        x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
        post_t = torch.zeros(T, HC_MULT, dtype=torch.float32)
        comb_t = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
        golden_hc_pre({
            "x":        tensors["x_hc"][r],
            "hc_fn":    tensors["hc_ffn_fn"][r],
            "hc_scale": tensors["hc_ffn_scale"][r],
            "hc_base":  tensors["hc_ffn_base"][r],
            "x_mixed":  x_mixed,
            "post":     post_t,
            "comb":     comb_t,
        })

        # Stage 2: gate (global routing)
        x_norm = torch.zeros(T, D, dtype=torch.bfloat16)
        x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
        x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
        indices = torch.zeros(T, TOPK, dtype=torch.int32)
        weights = torch.zeros(T, TOPK, dtype=torch.float32)
        golden_gate_core({
            "x_mixed":      x_mixed,
            "norm_w":       tensors["norm_w"][r],
            "gate_w":       tensors["gate_w"][r],
            "gate_bias":    tensors["gate_bias"][r],
            "layer_id":     tensors["layer_id"],
            "tid2eid":      tensors["tid2eid"][r],
            "input_ids":    tensors["input_ids"][r],
            "x_norm":       x_norm,
            "x_norm_i8":    x_norm_i8,
            "x_norm_scale": x_norm_scale,
            "indices":      indices,
            "weights":      weights,
        })

        # Stage 3: expert_shared (local)
        sh = torch.zeros(T, D, dtype=torch.bfloat16)
        golden_expert_shared({
            "x_local_i8":       x_norm_i8,
            "x_local_scale_dq": x_norm_scale,
            "shared_w1":        tensors["shared_w1"][r],
            "shared_w1_scale":  tensors["shared_w1_scale"][r],
            "shared_w3":        tensors["shared_w3"][r],
            "shared_w3_scale":  tensors["shared_w3_scale"][r],
            "shared_w2":        tensors["shared_w2"][r],
            "shared_w2_scale":  tensors["shared_w2_scale"][r],
            "sh":               sh,
        })

        # Stage 4: host-side dispatch simulation across all ranks for this dst=r.
        # Collect all (src, t, k) routes that land on rank r.
        recv_x = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.int8)
        recv_scale = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
        recv_w = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
        recv_r_route = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.int32)
        recv_count = torch.zeros(N_LOCAL, 1, dtype=torch.int32)

        # Compute slot offsets the way dispatch does (rank-major within
        # each local expert), so the order matches the on-device run.
        send_counts = torch.zeros(N_RANKS, N_RANKS, N_LOCAL, dtype=torch.int32)
        all_indices = []
        all_x_i8 = []
        all_scale = []
        all_weights = []
        for src in range(N_RANKS):
            src_x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
            src_post = torch.zeros(T, HC_MULT, dtype=torch.float32)
            src_comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
            golden_hc_pre({
                "x":        tensors["x_hc"][src],
                "hc_fn":    tensors["hc_ffn_fn"][src],
                "hc_scale": tensors["hc_ffn_scale"][src],
                "hc_base":  tensors["hc_ffn_base"][src],
                "x_mixed":  src_x_mixed,
                "post":     src_post,
                "comb":     src_comb,
            })
            src_x_norm = torch.zeros(T, D, dtype=torch.bfloat16)
            src_x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
            src_x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
            src_indices = torch.zeros(T, TOPK, dtype=torch.int32)
            src_weights = torch.zeros(T, TOPK, dtype=torch.float32)
            golden_gate_core({
                "x_mixed":      src_x_mixed,
                "norm_w":       tensors["norm_w"][src],
                "gate_w":       tensors["gate_w"][src],
                "gate_bias":    tensors["gate_bias"][src],
                "layer_id":     tensors["layer_id"],
                "tid2eid":      tensors["tid2eid"][src],
                "input_ids":    tensors["input_ids"][src],
                "x_norm":       src_x_norm,
                "x_norm_i8":    src_x_norm_i8,
                "x_norm_scale": src_x_norm_scale,
                "indices":      src_indices,
                "weights":      src_weights,
            })
            all_indices.append(src_indices)
            all_x_i8.append(src_x_norm_i8)
            all_scale.append(src_x_norm_scale)
            all_weights.append(src_weights)
            for t in range(num_tokens):
                for k in range(TOPK):
                    eid = int(src_indices[t, k].item())
                    dst = eid // N_LOCAL
                    loc_e = eid % N_LOCAL
                    send_counts[src, dst, loc_e] += 1

        # Pack onto rank r in src-major (rank 0 first, then rank 1) within each
        # local expert — same convention as dispatch's prefix_sum offsets.
        slot_offsets = torch.zeros(N_RANKS, N_LOCAL, dtype=torch.int32)
        running = torch.zeros(N_LOCAL, dtype=torch.int32)
        for src in range(N_RANKS):
            slot_offsets[src] = running.clone()
            running = running + send_counts[src, r]
        for e in range(N_LOCAL):
            recv_count[e, 0] = int(running[e].item())

        for src in range(N_RANKS):
            cursor = torch.zeros(N_LOCAL, dtype=torch.int32)
            for t in range(num_tokens):
                for k in range(TOPK):
                    eid = int(all_indices[src][t, k].item())
                    if eid // N_LOCAL != r:
                        continue
                    loc_e = eid % N_LOCAL
                    slot = int(slot_offsets[src, loc_e].item() + cursor[loc_e].item())
                    cursor[loc_e] += 1
                    recv_x[loc_e, slot, :] = all_x_i8[src][t, :]
                    recv_scale[loc_e, slot] = float(all_scale[src][t, 0].item())
                    recv_w[loc_e, slot] = float(all_weights[src][t, k].item())
                    recv_r_route[loc_e, slot] = t * TOPK + k

        # Stage 5: routed expert (local, weighted)
        recv_y = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.bfloat16)
        golden_expert_routed({
            "recv_x":            recv_x,
            "recv_scale_dq":     recv_scale,
            "recv_weights":      recv_w,
            "recv_expert_count": recv_count,
            "routed_w1":         tensors["routed_w1"][r],
            "routed_w1_scale":   tensors["routed_w1_scale"][r],
            "routed_w3":         tensors["routed_w3"][r],
            "routed_w3_scale":   tensors["routed_w3_scale"][r],
            "routed_w2":         tensors["routed_w2"][r],
            "routed_w2_scale":   tensors["routed_w2_scale"][r],
            "recv_y":            recv_y,
        })

        # Stage 6: combine — for each (src, t, k) that originated on this
        # rank, find the (loc_e, slot) on rank dst where the SwiGLU result
        # landed, then accumulate by r_route = t*TOPK+k.
        # Recreate the slot bookkeeping for each dst from this rank r's POV.
        my_routes = []
        for t in range(num_tokens):
            for k in range(TOPK):
                eid = int(all_indices[r][t, k].item())
                dst = eid // N_LOCAL
                loc_e = eid % N_LOCAL
                my_routes.append((t, k, dst, loc_e))

        # For each dst, dst-side has packing where rank-r's contribution lives
        # at slot offset = Σ_{s<r} send_counts[s, dst, loc_e].
        dst_recv_y = {}
        dst_recv_count = {}
        for dst in range(N_RANKS):
            # Replay dispatch from ALL src ranks to dst, then expert_routed,
            # then pull out per-route results.
            d_recv_x = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.int8)
            d_recv_scale = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
            d_recv_w = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.float32)
            d_recv_r_route = torch.zeros(N_LOCAL, RECV_MAX, dtype=torch.int32)
            d_recv_count = torch.zeros(N_LOCAL, 1, dtype=torch.int32)
            d_slot_offsets = torch.zeros(N_RANKS, N_LOCAL, dtype=torch.int32)
            d_running = torch.zeros(N_LOCAL, dtype=torch.int32)
            for src in range(N_RANKS):
                d_slot_offsets[src] = d_running.clone()
                d_running = d_running + send_counts[src, dst]
            for e in range(N_LOCAL):
                d_recv_count[e, 0] = int(d_running[e].item())
            for src in range(N_RANKS):
                cursor = torch.zeros(N_LOCAL, dtype=torch.int32)
                for t in range(num_tokens):
                    for k in range(TOPK):
                        eid = int(all_indices[src][t, k].item())
                        if eid // N_LOCAL != dst:
                            continue
                        loc_e = eid % N_LOCAL
                        slot = int(d_slot_offsets[src, loc_e].item() + cursor[loc_e].item())
                        cursor[loc_e] += 1
                        d_recv_x[loc_e, slot, :] = all_x_i8[src][t, :]
                        d_recv_scale[loc_e, slot] = float(all_scale[src][t, 0].item())
                        d_recv_w[loc_e, slot] = float(all_weights[src][t, k].item())
                        d_recv_r_route[loc_e, slot] = t * TOPK + k
            d_recv_y = torch.zeros(N_LOCAL, RECV_MAX, D, dtype=torch.bfloat16)
            golden_expert_routed({
                "recv_x":            d_recv_x,
                "recv_scale_dq":     d_recv_scale,
                "recv_weights":      d_recv_w,
                "recv_expert_count": d_recv_count,
                "routed_w1":         tensors["routed_w1"][dst],
                "routed_w1_scale":   tensors["routed_w1_scale"][dst],
                "routed_w3":         tensors["routed_w3"][dst],
                "routed_w3_scale":   tensors["routed_w3_scale"][dst],
                "routed_w2":         tensors["routed_w2"][dst],
                "routed_w2_scale":   tensors["routed_w2_scale"][dst],
                "recv_y":            d_recv_y,
            })
            dst_recv_y[dst] = d_recv_y
            dst_recv_count[dst] = d_recv_count

        # Now combine — per-route reverse lookup of dst's slot for THIS rank
        # r's (t, k):
        routed_y_buf_r = torch.zeros(N_ROUTES, D, dtype=torch.bfloat16)
        for (t, k, dst, loc_e) in my_routes:
            # Find this rank r's slot inside dst.recv_x: src=r block,
            # cursor = how many of r's (t', k' <= (t, k)) so far targeted this loc_e.
            src_off = 0
            for s in range(r):
                src_off += int(send_counts[s, dst, loc_e].item())
            # Count how many earlier (t', k') from rank r targeted (dst, loc_e).
            cursor = 0
            for (tt, kk, dd, ll) in my_routes:
                if (tt, kk) == (t, k):
                    break
                if dd == dst and ll == loc_e:
                    cursor += 1
            slot = src_off + cursor
            r_route = t * TOPK + k
            routed_y_buf_r[r_route, :] = dst_recv_y[dst][loc_e, slot, :]

        # Stage 7: reduce + sh + hc_post
        acc = sh.float().clone()
        for k in range(TOPK):
            for t in range(num_tokens):
                acc[t, :] += routed_y_buf_r[t * TOPK + k, :].float()
        ffn_out = acc.to(torch.bfloat16)
        x_next_r = torch.zeros(T, HC_MULT, D, dtype=torch.bfloat16)
        golden_hc_post({
            "x":        ffn_out,
            "residual": tensors["x_hc"][r],
            "post":     post_t,
            "comb":     comb_t,
            "y":        x_next_r,
        })
        x_next_out[r] = x_next_r

    tensors["x_next"][:] = x_next_out


def build_tensor_specs(layer_id=0):
    import torch
    from golden import ScalarSpec, TensorSpec
    from expert_routed import gen_routed_weight
    from expert_shared import gen_shared_weight

    # Routed = MXFP4 (gen_routed_weight), shared = MXFP8 (gen_shared_weight). This
    # is an integration test whose x_next-equivalent output is dominated by near-zero
    # residual+FFN cancellations, so it keeps the smaller *behaviorally-calibrated* magnitude
    # (random fixtures blow up the relative metric at the real ~2.5e-2 magnitude); only the
    # grid SHAPE (FP4/FP8 discreteness, scale CV) matches the real distribution.
    ROUTED_DEQUANT_STD = {"w1": 1.08e-2, "w2": 2.54e-2, "w3": 1.10e-2}
    SHARED_DEQUANT_STD = {"w1": 7.65e-3, "w2": 2.39e-2, "w3": 7.39e-3}

    # Shared (replicated) weights are broadcast across ranks; the routed
    # weights are per-rank shards.
    def init_x_hc():
        return torch.randn(N_RANKS, T, HC_MULT, D)

    # Real layer-0 hc_ffn scale/base (fn synthetic at real magnitude). A synthetic
    # scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling the FFN output and
    # hc residual to near-zero in x_next where W8A8 noise blows up the relative tail.
    def init_hc_ffn_fn():
        x = torch.randn(MIX_HC, HC_DIM) * 0.0635
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_hc_ffn_scale():
        x = torch.tensor([0.11334, 0.035901, 0.058183])
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_hc_ffn_base():
        x = torch.tensor([
            2.4153, -2.0252, -2.0019, -2.1947,
            -1.5430, -3.0228, -6.8248, 0.5894,
            2.1916, -7.2132, -3.0938, -2.1119,
            -3.0161, 3.3293, -3.2224, -4.0226,
            -2.0428, -3.3478, 3.0893, -3.4166,
            -1.8144, -3.8147, -3.1307, 1.7862,
        ])
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_norm_w():
        x = torch.ones(D)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_gate_w():
        x = torch.randn(N_EXPERTS_GLOBAL, D) / D ** 0.5
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_gate_bias():
        x = torch.zeros(N_EXPERTS_GLOBAL)
        return x.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_tid2eid():
        # Distinct experts per token (sample without replacement) like real top-k,
        # so token-keyed combine_ep1 == route-keyed distributed combine.
        x = torch.argsort(torch.rand(VOCAB, N_EXPERTS_GLOBAL), dim=1)[:, :TOPK].to(torch.int32)
        return x.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

    def init_input_ids():
        # Distinct per-rank token streams.
        return torch.randint(0, VOCAB, (N_RANKS, T), dtype=torch.int64)

    # Per-rank routed expert weights (different shards).
    routed_w1_i8_list = []
    routed_w1_s_list = []
    routed_w3_i8_list = []
    routed_w3_s_list = []
    routed_w2_i8_list = []
    routed_w2_s_list = []
    for _ in range(N_RANKS):
        w1_i8, w1_s = gen_routed_weight((N_LOCAL, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
        w3_i8, w3_s = gen_routed_weight((N_LOCAL, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
        w2_i8, w2_s = gen_routed_weight((N_LOCAL, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])
        routed_w1_i8_list.append(w1_i8)
        routed_w1_s_list.append(w1_s)
        routed_w3_i8_list.append(w3_i8)
        routed_w3_s_list.append(w3_s)
        routed_w2_i8_list.append(w2_i8)
        routed_w2_s_list.append(w2_s)

    rw1_i8 = torch.stack(routed_w1_i8_list)
    rw1_s = torch.stack(routed_w1_s_list)
    rw3_i8 = torch.stack(routed_w3_i8_list)
    rw3_s = torch.stack(routed_w3_s_list)
    rw2_i8 = torch.stack(routed_w2_i8_list)
    rw2_s = torch.stack(routed_w2_s_list)

    # Shared expert weights — replicated across ranks.
    sw1_i8, sw1_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3_i8, sw3_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2_i8, sw2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)
    sw1_i8 = sw1_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw1_s = sw1_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    sw3_i8 = sw3_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw3_s = sw3_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()
    sw2_i8 = sw2_i8.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
    sw2_s = sw2_s.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    return [
        TensorSpec("x_hc",          [N_RANKS, T, HC_MULT, D],     torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",     [N_RANKS, MIX_HC, HC_DIM],       torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale",  [N_RANKS, 3],                    torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",   [N_RANKS, MIX_HC],               torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",        [N_RANKS, D],                    torch.bfloat16,  init_value=init_norm_w),
        TensorSpec("gate_w",        [N_RANKS, N_EXPERTS_GLOBAL, D],  torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",     [N_RANKS, N_EXPERTS_GLOBAL],     torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",       [N_RANKS, VOCAB, TOPK],          torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids",     [N_RANKS, T],                 torch.int64,    init_value=init_input_ids),
        TensorSpec("routed_w1",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: rw1_i8),
        TensorSpec("routed_w1_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: rw1_s),
        TensorSpec("routed_w3",        [N_RANKS, N_LOCAL, MOE_INTER, D], torch.int8,    init_value=lambda: rw3_i8),
        TensorSpec("routed_w3_scale",  [N_RANKS, N_LOCAL, MOE_INTER],    torch.float32, init_value=lambda: rw3_s),
        TensorSpec("routed_w2",        [N_RANKS, N_LOCAL, D, MOE_INTER], torch.int8,    init_value=lambda: rw2_i8),
        TensorSpec("routed_w2_scale",  [N_RANKS, N_LOCAL, D],            torch.float32, init_value=lambda: rw2_s),
        TensorSpec("shared_w1",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3",        [N_RANKS, MOE_INTER, D],          torch.int8,    init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale",  [N_RANKS, MOE_INTER],             torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2",        [N_RANKS, D, MOE_INTER],          torch.int8,    init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale",  [N_RANKS, D],                     torch.float32, init_value=lambda: sw2_s),
        TensorSpec("x_next",           [N_RANKS, T, HC_MULT, D],      torch.bfloat16, is_output=True),
        ScalarSpec("layer_id",         torch.int32,                      layer_id),
    ]


# === Single-card (EP=1) path ================================================
# Local scatter dispatch/combine, no HCCL windows. With one rank the global
# router degenerates to local routing (every expert is owned here), so this is
# the standalone single-card path composed against the same sub-kernels. __main__
# dispatches to these *_ep1 defs directly when EP == 1.
@pl.jit.inline
def moe_ep1(
    x_hc:            pl.Tensor[[T, HC_MULT, D],          pl.BF16],
    hc_ffn_fn:       pl.Tensor[[MIX_HC, HC_DIM],         pl.FP32],
    hc_ffn_scale:    pl.Tensor[[3],                      pl.FP32],
    hc_ffn_base:     pl.Tensor[[MIX_HC],                 pl.FP32],
    norm_w:          pl.Tensor[[D],                      pl.BF16],
    gate_w:          pl.Tensor[[N_EXPERTS_GLOBAL, D],    pl.FP32],
    gate_bias:       pl.Tensor[[N_EXPERTS_GLOBAL],       pl.FP32],
    tid2eid:         pl.Tensor[[VOCAB, TOPK],            pl.INT32],
    input_ids:       pl.Tensor[[T],                      pl.INT64],
    routed_w1:       pl.Tensor[[N_LOCAL, MOE_INTER, D],  pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER],     pl.FP32],
    routed_w3:       pl.Tensor[[N_LOCAL, MOE_INTER, D],  pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER],     pl.FP32],
    routed_w2:       pl.Tensor[[N_LOCAL, D, MOE_INTER],  pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D],             pl.FP32],
    shared_w1:       pl.Tensor[[MOE_INTER, D],           pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER],              pl.FP32],
    shared_w3:       pl.Tensor[[MOE_INTER, D],           pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER],              pl.FP32],
    shared_w2:       pl.Tensor[[D, MOE_INTER],           pl.INT8],
    shared_w2_scale: pl.Tensor[[D],                      pl.FP32],
    x_next:          pl.Out[pl.Tensor[[T, HC_MULT, D],   pl.BF16]],
    layer_id:        pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed, post_ffn, comb_ffn)

    x_norm = pl.create_tensor([T, D], dtype=pl.BF16)
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, tid2eid, input_ids,
        x_norm, x_norm_i8, x_norm_scale, indices, weights,
    )

    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )

    recv_x = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_dq = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
    recv_weights = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32)
    recv_token = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32)
    recv_count = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    dispatch_ep1(
        x_norm_i8, x_norm_scale, indices, weights,
        recv_x, recv_scale_dq, recv_weights, recv_token, recv_count,
    )

    recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
    expert_routed(
        recv_x, recv_scale_dq, recv_weights, recv_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )

    ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    combine_ep1(recv_y, recv_token, recv_count, sh, ffn_out)

    hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next


@pl.jit
def moe_ep1_test(
    x_hc:            pl.Tensor[[T, HC_MULT, D],          pl.BF16],
    hc_ffn_fn:       pl.Tensor[[MIX_HC, HC_DIM],         pl.FP32],
    hc_ffn_scale:    pl.Tensor[[3],                      pl.FP32],
    hc_ffn_base:     pl.Tensor[[MIX_HC],                 pl.FP32],
    norm_w:          pl.Tensor[[D],                      pl.BF16],
    gate_w:          pl.Tensor[[N_EXPERTS_GLOBAL, D],    pl.FP32],
    gate_bias:       pl.Tensor[[N_EXPERTS_GLOBAL],       pl.FP32],
    tid2eid:         pl.Tensor[[VOCAB, TOPK],            pl.INT32],
    input_ids:       pl.Tensor[[T],                      pl.INT64],
    routed_w1:       pl.Tensor[[N_LOCAL, MOE_INTER, D],  pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER],     pl.FP32],
    routed_w3:       pl.Tensor[[N_LOCAL, MOE_INTER, D],  pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER],     pl.FP32],
    routed_w2:       pl.Tensor[[N_LOCAL, D, MOE_INTER],  pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D],             pl.FP32],
    shared_w1:       pl.Tensor[[MOE_INTER, D],           pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER],              pl.FP32],
    shared_w3:       pl.Tensor[[MOE_INTER, D],           pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER],              pl.FP32],
    shared_w2:       pl.Tensor[[D, MOE_INTER],           pl.INT8],
    shared_w2_scale: pl.Tensor[[D],                      pl.FP32],
    x_next:          pl.Out[pl.Tensor[[T, HC_MULT, D],   pl.BF16]],
    layer_id:        pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.BF16]:
    x_next = moe_ep1(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next,
        layer_id,
    )
    return x_next


# EP1 golden/fixtures reuse the distributed builders: build_tensor_specs_ep1 projects
# the rank-0 slice; golden_moe_ep1 wraps a rank-0 axis and replays golden_moe.
def golden_moe_ep1(tensors):
    """EP1 golden = the 1-rank distributed reference, wrapped/unwrapped on the rank axis."""
    import torch

    wrapped = {
        k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1 else v)
        for k, v in tensors.items()
    }
    golden_moe(wrapped)
    tensors["x_next"][:] = wrapped["x_next"][0]


def build_tensor_specs_ep1(layer_id=0):
    """EP1 fixtures = the distributed specs with the rank-0 leading dim dropped."""
    from golden import ScalarSpec, TensorSpec

    flat = []
    for s in build_tensor_specs(layer_id=layer_id):
        if isinstance(s, ScalarSpec):
            flat.append(s)
        elif s.is_output:
            flat.append(TensorSpec(s.name, s.shape[1:], s.dtype, is_output=True))
        else:
            flat.append(TensorSpec(
                s.name, s.shape[1:], s.dtype,
                init_value=(lambda iv: (lambda: iv()[0]))(s.init_value),
            ))
    return flat


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=_EP_DEFAULT, choices=list(_EP_CHOICES),
                        help="EP world size / rank count")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None,
                        help="dir with cached in/{name}.pt + out/{name}.pt; reuses them "
                             "instead of regenerating inputs + recomputing golden.")
    parser.add_argument("--log-level", type=str, default=None,
                        help="runtime log threshold: debug, v0..v9, info, warn, error, null")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    golden_data = args.golden_data

    if EP == 1:
        # One rank: plain single-card run (no DistributedConfig / HCCL windows).
        result = run_jit(
            fn=moe_ep1_test,
            specs=build_tensor_specs_ep1(layer_id=args.layer_id),
            golden_fn=golden_moe_ep1,
            golden_data=golden_data,
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=device_ids[0],
                enable_l2_swimlane=args.enable_l2_swimlane,
                log_level=args.log_level,
            ),
            rtol=1e-3,
            atol=1e-3,
            # BF16 x_next, same FFN floor as the distributed path. 3e-3 with the real
            # layer-0 hc_ffn gate; no max_diff_hd (near-zero cancellations blow up relatively).
            compare_fn={"x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.05)},
        )
    else:
        result = run_jit(
            fn=l3_moe,
            specs=build_tensor_specs(layer_id=args.layer_id),
            golden_fn=golden_moe,
            golden_data=golden_data,
            compile_only=args.compile_only,
            runtime_dir=args.runtime_dir,
            compile_cfg=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(
                    device_ids=device_ids,
                    num_sub_workers=0,
                ),
            ),
            runtime_cfg=dict(
                platform=args.platform,
                enable_l2_swimlane=args.enable_l2_swimlane,
                log_level=args.log_level,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                # BF16 x_next, same FFN floor as the single-card path. Tightened 5e-3 -> 3e-3 with
                # the real layer-0 hc_ffn gate (~2.1% of points > 3e-3). No max_diff_hd
                # (near-zero residual/FFN cancellations blow up relatively).
                "x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.05),
            },
        )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
