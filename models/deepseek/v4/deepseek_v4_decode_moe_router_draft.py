# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE FFN router decode orchestration.
Composes Block.hc_pre (ffn) + ffn_norm + Gate.forward (model.py:697-698, 564-584). Outputs the
post-norm hidden state `x_norm` for downstream EP dispatch / shared-expert input, the per-token
top-k expert indices+weights, and the post/comb tensors required by hc_post (ffn).
Companion file: deepseek_v4_decode_hc_pre.py (reused via the same Compute, weights swapped to hc_ffn_*)."""


import pypto.language as pl


B           = 16               # demo 4
S           = 1
T           = B * S
D           = 4096             # flash:4096 pro:7168
NORM_EPS    = 1e-6

N_EXPERTS   = 8                # flash:256 pro:384
TOPK        = 2                # flash:6 pro:6 (n_activated_experts)
ROUTE_SCALE = 1.0              # flash:1.5 pro:2.5
VOCAB       = 129280

# Per-layer routing mode is fixed at build time: layers with LAYER_ID < N_HASH_LAYERS
# do tid2eid lookup (no scores, no bias, no topk); the rest do learned-score + bias + topk.
# Mirrors `Gate.hash = layer_id < args.n_hash_layers` (model.py:556).
LAYER_ID      = 1               # this layer's index in the Transformer stack
N_HASH_LAYERS = 0               # demo 0; flash:3 pro:3 (raise to make the first few layers hash-routed)

# hc_pre (ffn)
HC_MULT          = 4
MIX_HC           = (2 + HC_MULT) * HC_MULT
HC_DIM           = HC_MULT * D


def build_deepseek_v4_decode_moe_router_program():
    @pl.program
    class DeepSeekV4DecodeMoERouter:
        @pl.function(type=pl.FunctionType.Orchestration)
        def deepseek_v4_decode_moe_router(
            self,
            x_hc:         pl.Tensor[[B, S, HC_MULT, D],            pl.BF16],
            # hc_pre (ffn) weights
            hc_ffn_fn:    pl.Tensor[[MIX_HC, HC_DIM],              pl.FP32],
            hc_ffn_scale: pl.Tensor[[3],                           pl.FP32],
            hc_ffn_base:  pl.Tensor[[MIX_HC],                      pl.FP32],
            # ffn_norm + gate weights
            norm_w:       pl.Tensor[[D],                           pl.FP32],
            gate_w:       pl.Tensor[[N_EXPERTS, D],                pl.FP32],
            gate_bias:    pl.Tensor[[N_EXPERTS],                   pl.FP32],
            tid2eid:      pl.Tensor[[VOCAB, TOPK],                 pl.INT32],
            input_ids:    pl.Tensor[[B, S],                        pl.INT64],
            x_norm:       pl.Out[pl.Tensor[[T, D],                 pl.BF16]],
            indices:      pl.Out[pl.Tensor[[T, TOPK],              pl.INT32]],
            weights:      pl.Out[pl.Tensor[[T, TOPK],              pl.FP32]],
            post_ffn:     pl.Out[pl.Tensor[[B, S, HC_MULT],        pl.FP32]],
            comb_ffn:     pl.Out[pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32]],
        ):
            # TODO: orchestration body (dispatches the per-step kernels)
            return x_norm, indices, weights, post_ffn, comb_ffn

    return DeepSeekV4DecodeMoERouter


def golden_deepseek_v4_decode_moe_router(tensors):
    """End-to-end orchestration: Block.hc_pre (ffn) + ffn_norm + Gate.forward.
    Mirrors model.py:697-698 (Block.forward ffn half) + 191-196 (RMSNorm) + 564-584 (Gate.forward)."""
    import torch
    import torch.nn.functional as F

    from deepseek_v4_decode_hc_pre import golden_deepseek_v4_decode_hc_pre

    # ---- Block.hc_pre (model.py:697); reuses attn-side hc_pre kernel with hc_ffn_* weights. ----
    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT)
    golden_deepseek_v4_decode_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_ffn_fn"],
        "hc_scale": tensors["hc_ffn_scale"],
        "hc_base": tensors["hc_ffn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # ---- ffn_norm (RMSNorm, model.py:191-196 fused into Block at line 698) ----
    norm_w = tensors["norm_w"].float()
    x_f = x_mixed.float()
    var = x_f.square().mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + NORM_EPS)
    # RMSNorm returns the original dtype (bf16); preserves the cast that downstream gate/expert see.
    x_normalized = (norm_w * x_n).to(torch.bfloat16)         # [B, S, D]

    x_flat = x_normalized.view(T, D)                          # [T, D] bf16

    # ---- Gate.forward (model.py:564-584); sqrtsoftplus path. ----
    gate_w = tensors["gate_w"].float()
    gate_bias = tensors["gate_bias"].float()
    scores = F.softplus(x_flat.float() @ gate_w.T).sqrt()    # [T, N_EXPERTS]
    original_scores = scores

    if LAYER_ID >= N_HASH_LAYERS:
        biased = scores + gate_bias
        indices = biased.topk(TOPK, dim=-1).indices           # [T, TOPK]
    else:                                                     # hash-routed layer
        tid2eid = tensors["tid2eid"]
        input_ids = tensors["input_ids"]
        indices = tid2eid[input_ids.flatten().long()]         # [T, TOPK]

    weights = original_scores.gather(1, indices.long())       # [T, TOPK]
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE

    tensors["x_norm"][:]   = x_flat
    tensors["indices"][:]  = indices.to(torch.int32)
    tensors["weights"][:]  = weights.to(torch.float32)
    tensors["post_ffn"][:] = post_t
    tensors["comb_ffn"][:] = comb_t


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_hc():
        return torch.randn(B, S, HC_MULT, D) * 0.1
    def init_hc_ffn_fn():
        return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    def init_hc_ffn_scale():
        return torch.ones(3) * 0.5
    def init_hc_ffn_base():
        return torch.zeros(MIX_HC)
    def init_norm_w():
        return torch.ones(D)
    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():
        return torch.zeros(N_EXPERTS)
    def init_tid2eid():
        return torch.randint(0, N_EXPERTS, (VOCAB, TOPK), dtype=torch.int32)
    def init_input_ids():
        return torch.randint(0, VOCAB, (B, S), dtype=torch.int64)

    return [
        TensorSpec("x_hc",         [B, S, HC_MULT, D],         torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",    [MIX_HC, HC_DIM],           torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale", [3],                        torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",  [MIX_HC],                   torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",       [D],                        torch.float32,  init_value=init_norm_w),
        TensorSpec("gate_w",       [N_EXPERTS, D],             torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",    [N_EXPERTS],                torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",      [VOCAB, TOPK],              torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids",    [B, S],                     torch.int64,    init_value=init_input_ids),
        TensorSpec("x_norm",       [T, D],                     torch.bfloat16, is_output=True),
        TensorSpec("indices",      [T, TOPK],                  torch.int32,    is_output=True),
        TensorSpec("weights",      [T, TOPK],                  torch.float32,  is_output=True),
        TensorSpec("post_ffn",     [B, S, HC_MULT],            torch.float32,  is_output=True),
        TensorSpec("comb_ffn",     [B, S, HC_MULT, HC_MULT],   torch.float32,  is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v4_decode_moe_router_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_moe_router,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
