# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE FFN norm + router (decode): RMSNorm then sqrt-softplus gate score, with a hash
branch (tid2eid lookup) for the first n_hash_layers. Outputs per-token expert indices + weights."""


import pypto.language as pl


B           = 16               # demo 4
S           = 1
T           = B * S
D           = 4096             # v4-pro 7168
NORM_EPS    = 1e-6

N_EXPERTS   = 8                # v4-pro 384
TOPK        = 2                # v4-pro 6
ROUTE_SCALE = 1.0              # v4-pro 2.5
VOCAB       = 129280

# "score" for most layers (learned gate scores + bias + topk).
# "hash" for the first n_hash_layers (tid2eid lookup, no topk).
MODE        = "score"


def build_deepseek_v4_decode_moe_router_program():
    @pl.program
    class DeepSeekV4DecodeMoERouter:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_moe_router(
            self,
            x:         pl.Tensor[[B, S, D],            pl.BF16],
            norm_w:    pl.Tensor[[D],                  pl.FP32],
            gate_w:    pl.Tensor[[N_EXPERTS, D],       pl.FP32],
            gate_bias: pl.Tensor[[N_EXPERTS],          pl.FP32],
            tid2eid:   pl.Tensor[[VOCAB, TOPK],        pl.INT32],
            input_ids: pl.Tensor[[B, S],               pl.INT64],
            indices:   pl.Out[pl.Tensor[[T, TOPK],     pl.INT32]],
            weights:   pl.Out[pl.Tensor[[T, TOPK],     pl.FP32]],
        ):
            # TODO: kernel implementation
            return indices, weights

    return DeepSeekV4DecodeMoERouter


def golden_deepseek_v4_decode_moe_router(tensors):
    """Torch reference: RMSNorm then Gate.forward (model.py 191-196, 564-584)."""
    import torch
    import torch.nn.functional as F

    x         = tensors["x"].float()              # [B, S, D]
    norm_w    = tensors["norm_w"].float()          # [D]
    gate_w    = tensors["gate_w"].float()          # [N_EXPERTS, D]
    gate_bias = tensors["gate_bias"].float()       # [N_EXPERTS]
    tid2eid   = tensors["tid2eid"]                 # [VOCAB, TOPK]
    input_ids = tensors["input_ids"]               # [B, S]

    # ffn_norm: RMSNorm (model.py 191-196)
    var = x.square().mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + NORM_EPS)
    x_norm = (norm_w * x_norm).to(torch.float32)  # [B, S, D]

    x_flat = x_norm.view(T, D)                     # [T, D]

    # Gate.forward (model.py 564-584)
    scores = F.softplus(x_flat @ gate_w.T).sqrt()  # [T, N_EXPERTS]
    original_scores = scores

    if MODE == "score":
        biased  = scores + gate_bias
        indices = biased.topk(TOPK, dim=-1).indices                   # [T, TOPK]
    else:  # "hash"
        indices = tid2eid[input_ids.flatten().long()]                 # [T, TOPK]

    weights = original_scores.gather(1, indices.long())               # [T, TOPK]
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE

    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights.to(torch.float32)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.1
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
        TensorSpec("x",         [B, S, D],         torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w",    [D],                torch.float32,  init_value=init_norm_w),
        TensorSpec("gate_w",    [N_EXPERTS, D],     torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias", [N_EXPERTS],        torch.float32,  init_value=init_gate_bias),
        TensorSpec("tid2eid",   [VOCAB, TOPK],      torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids", [B, S],             torch.int64,    init_value=init_input_ids),
        TensorSpec("indices",   [T, TOPK],          torch.int32,    is_output=True),
        TensorSpec("weights",   [T, TOPK],          torch.float32,  is_output=True),
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
