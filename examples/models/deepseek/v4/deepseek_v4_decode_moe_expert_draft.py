# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE local expert + shared expert compute (decode, EP single-card): runs SwiGLU FFN
on dispatched tokens for this card's local routed experts plus the replicated shared expert."""


import pypto.language as pl


B            = 16                 # demo 4
S            = 1
T            = B * S

D            = 4096               # v4-pro 7168
MOE_INTER    = 4096               # v4-pro 3072
TOPK         = 2                  # v4-pro 6
SWIGLU_LIMIT = 0.0                # v4-pro 10.0

# EP sharding (compile-time constants; one binary per EP rank)
EP_WORLD_SIZE     = 1                                 # v4-pro 16
EP_RANK           = 0
N_EXPERTS         = 8                                 # v4-pro 384
N_LOCAL_EXPERTS   = N_EXPERTS // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS

# Static upper bound for dispatch token count (dynamic at runtime; average T*TOPK/EP_WORLD_SIZE).
RECV_TOTAL   = T * TOPK // EP_WORLD_SIZE


def build_deepseek_v4_decode_moe_expert_program():
    @pl.program
    class DeepSeekV4DecodeMoEExpert:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_moe_expert(
            self,
            recv_x:        pl.Tensor[[RECV_TOTAL, D],                        pl.BF16],
            recv_expert_id: pl.Tensor[[RECV_TOTAL],                          pl.INT32],
            recv_weights:  pl.Tensor[[RECV_TOTAL],                           pl.FP32],
            x_local:       pl.Tensor[[T, D],                                 pl.BF16],
            expert_w1:     pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],        pl.BF16],
            expert_w3:     pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],        pl.BF16],
            expert_w2:     pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER],        pl.BF16],
            shared_w1:     pl.Tensor[[MOE_INTER, D],                         pl.BF16],
            shared_w3:     pl.Tensor[[MOE_INTER, D],                         pl.BF16],
            shared_w2:     pl.Tensor[[D, MOE_INTER],                         pl.BF16],
            recv_y:        pl.Out[pl.Tensor[[RECV_TOTAL, D],                 pl.BF16]],
            sh:            pl.Out[pl.Tensor[[T, D],                          pl.BF16]],
        ):
            # TODO: kernel implementation
            return recv_y, sh

    return DeepSeekV4DecodeMoEExpert


def golden_deepseek_v4_decode_moe_expert(tensors):
    """Torch reference: local grouped SwiGLU + shared expert (model.py 596-644).

    recv_y is the partial routed contribution for tokens dispatched to this
    card; it is NOT the final MoE output (AllToAllv combine and +sh happen
    outside this kernel in the host orchestrator).
    """
    import torch
    import torch.nn.functional as F

    recv_x        = tensors["recv_x"].float()          # [RECV_TOTAL, D]
    recv_expert_id = tensors["recv_expert_id"]         # [RECV_TOTAL]  values 0..N_LOCAL_EXPERTS-1
    recv_weights  = tensors["recv_weights"].float()    # [RECV_TOTAL]
    x_local       = tensors["x_local"].float()         # [T, D]
    w1 = tensors["expert_w1"].float()                  # [N_LOCAL_EXPERTS, MOE_INTER, D]
    w3 = tensors["expert_w3"].float()                  # [N_LOCAL_EXPERTS, MOE_INTER, D]
    w2 = tensors["expert_w2"].float()                  # [N_LOCAL_EXPERTS, D, MOE_INTER]
    sw1 = tensors["shared_w1"].float()                 # [MOE_INTER, D]
    sw3 = tensors["shared_w3"].float()                 # [MOE_INTER, D]
    sw2 = tensors["shared_w2"].float()                 # [D, MOE_INTER]

    # Local routed experts (model.py 636-641)
    recv_y = torch.zeros(RECV_TOTAL, D)
    for local_i in range(N_LOCAL_EXPERTS):
        mask = (recv_expert_id == local_i)
        if mask.sum() == 0:
            continue
        x_sub = recv_x[mask]                                             # [n, D]
        w_sub = recv_weights[mask]                                       # [n]
        gate = x_sub @ w1[local_i].T                                     # [n, MOE_INTER]
        up   = x_sub @ w3[local_i].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up   = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h    = F.silu(gate) * up                                         # [n, MOE_INTER]
        h    = h * w_sub.unsqueeze(-1)
        recv_y[mask] = h @ w2[local_i].T                                 # [n, D]

    # Shared expert: no clamp, no routing weight (model.py 644)
    sh_gate = x_local @ sw1.T                                            # [T, MOE_INTER]
    sh_up   = x_local @ sw3.T
    sh      = F.silu(sh_gate) * sh_up @ sw2.T                           # [T, D]

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)
    tensors["sh"][:]     = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_recv_x():
        return torch.randn(RECV_TOTAL, D) * 0.05
    def init_recv_expert_id():
        # Distribute dispatched tokens uniformly across local experts
        ids = torch.arange(RECV_TOTAL, dtype=torch.int32) % N_LOCAL_EXPERTS
        return ids[torch.randperm(RECV_TOTAL)]
    def init_recv_weights():
        w = torch.rand(RECV_TOTAL) + 0.1
        return (w / w.sum() * TOPK).float()
    def init_x_local():
        return torch.randn(T, D) * 0.05
    def init_w1():
        return torch.randn(N_LOCAL_EXPERTS, MOE_INTER, D) / D ** 0.5
    def init_w3():
        return torch.randn(N_LOCAL_EXPERTS, MOE_INTER, D) / D ** 0.5
    def init_w2():
        return torch.randn(N_LOCAL_EXPERTS, D, MOE_INTER) / MOE_INTER ** 0.5
    def init_sw1():
        return torch.randn(MOE_INTER, D) / D ** 0.5
    def init_sw3():
        return torch.randn(MOE_INTER, D) / D ** 0.5
    def init_sw2():
        return torch.randn(D, MOE_INTER) / MOE_INTER ** 0.5

    return [
        TensorSpec("recv_x",        [RECV_TOTAL, D],                   torch.bfloat16, init_value=init_recv_x),
        TensorSpec("recv_expert_id",[RECV_TOTAL],                       torch.int32,    init_value=init_recv_expert_id),
        TensorSpec("recv_weights",  [RECV_TOTAL],                       torch.float32,  init_value=init_recv_weights),
        TensorSpec("x_local",       [T, D],                            torch.bfloat16, init_value=init_x_local),
        TensorSpec("expert_w1",     [N_LOCAL_EXPERTS, MOE_INTER, D],   torch.bfloat16, init_value=init_w1),
        TensorSpec("expert_w3",     [N_LOCAL_EXPERTS, MOE_INTER, D],   torch.bfloat16, init_value=init_w3),
        TensorSpec("expert_w2",     [N_LOCAL_EXPERTS, D, MOE_INTER],   torch.bfloat16, init_value=init_w2),
        TensorSpec("shared_w1",     [MOE_INTER, D],                    torch.bfloat16, init_value=init_sw1),
        TensorSpec("shared_w3",     [MOE_INTER, D],                    torch.bfloat16, init_value=init_sw3),
        TensorSpec("shared_w2",     [D, MOE_INTER],                    torch.bfloat16, init_value=init_sw2),
        TensorSpec("recv_y",        [RECV_TOTAL, D],                   torch.bfloat16, is_output=True),
        TensorSpec("sh",            [T, D],                            torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_moe_expert_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_moe_expert,
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
