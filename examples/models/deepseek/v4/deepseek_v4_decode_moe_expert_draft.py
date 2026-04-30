# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE local expert + shared expert compute (decode, EP single-card)."""


import pypto.language as pl


B = 16  # demo 4
S = 1
T = B * S

D = 4096            # v4-pro 7168
MOE_INTER = 4096    # v4-pro 3072
TOPK = 2            # v4-pro 6
SWIGLU_LIMIT = 0.0  # v4-pro 10.0

EP_WORLD_SIZE = 1   # v4-pro 16
EP_RANK = 0
N_EXPERTS = 8       # v4-pro 384
N_LOCAL_EXPERTS = N_EXPERTS // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS  # global id offset; recv_expert_id carries global ids

RECV_TOTAL_MAX = 32  # v4-pro 32 (avg T*TOPK/EP=1.5, padded with imbalance headroom)

K_CHUNK = 256
INTER_K = 256
INTER_CHUNK = 64
D_OUT_CHUNK = 64
RECV_Y_INIT_CHUNK = 512


def build_deepseek_v4_decode_moe_expert_program():
    @pl.program
    class DeepSeekV4DecodeMoEExpert:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_moe_expert(
            self,
            recv_x: pl.Tensor[[RECV_TOTAL_MAX, D], pl.BF16],
            # Global expert id per row; padding rows must carry an id outside
            # [EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS) (e.g. -1).
            recv_expert_id: pl.Tensor[[RECV_TOTAL_MAX], pl.INT32],
            recv_weights: pl.Tensor[[RECV_TOTAL_MAX], pl.FP32],
            x_local: pl.Tensor[[T, D], pl.BF16],
            expert_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.BF16],
            expert_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.BF16],
            expert_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.BF16],
            shared_w1: pl.Tensor[[MOE_INTER, D], pl.BF16],
            shared_w3: pl.Tensor[[MOE_INTER, D], pl.BF16],
            shared_w2: pl.Tensor[[D, MOE_INTER], pl.BF16],
            recv_y: pl.Out[pl.Tensor[[RECV_TOTAL_MAX, D], pl.BF16]],
            sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
        ):
            # Stage 0: zero-init recv_y so cross-expert accumulation is exact.
            for d0 in pl.parallel(0, D, RECV_Y_INIT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="recv_y_zero"):
                    zero_chunk = pl.full([RECV_TOTAL_MAX, RECV_Y_INIT_CHUNK], dtype=pl.FP32, value=0.0)
                    recv_y[:, d0 : d0 + RECV_Y_INIT_CHUNK] = pl.cast(zero_chunk, target_type=pl.BF16)

            # Stage 1: routed local experts.
            #   - local_i iterates GLOBAL expert ids (matches recv_expert_id values).
            #   - expert_w* are local-only, indexed by local_offset.
            #   - Each row matches at most one local_i, so mask zeros out non-matching
            #     rows and cross-expert `recv_y +=` reduces to scatter-by-mask.
            for local_i in pl.range(EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS):
                local_offset = local_i - EXPERTS_START_IDX
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_mask"):
                    mask_fp32 = pl.cast(pl.cmps(recv_expert_id, local_i, 0), target_type=pl.FP32)
                    scale_col = pl.reshape(pl.mul(recv_weights, mask_fp32), [RECV_TOTAL_MAX, 1])

                h_tile = pl.create_tensor([RECV_TOTAL_MAX, MOE_INTER], dtype=pl.BF16)

                for n0 in pl.parallel(0, MOE_INTER, INTER_CHUNK):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_gate_up_matmul"):
                        x_init = recv_x[:, 0 : K_CHUNK]
                        w1_init = expert_w1[local_offset, n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                        w3_init = expert_w3[local_offset, n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                        gate_acc = pl.matmul(x_init, w1_init, b_trans=True, out_dtype=pl.FP32)
                        up_acc = pl.matmul(x_init, w3_init, b_trans=True, out_dtype=pl.FP32)
                        for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                            x_k = recv_x[:, k0 : k0 + K_CHUNK]
                            w1_k = expert_w1[local_offset, n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                            w3_k = expert_w3[local_offset, n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                            gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)
                            up_acc = pl.matmul_acc(up_acc, x_k, w3_k, b_trans=True)

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_swiglu"):
                        if SWIGLU_LIMIT > 0:
                            gate_acc = pl.mins(gate_acc, SWIGLU_LIMIT)
                            up_acc = pl.maxs(pl.mins(up_acc, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        silu = pl.mul(gate_acc, sigmoid)
                        gated = pl.mul(silu, up_acc)
                        h_chunk = pl.row_expand_mul(gated, scale_col)
                        h_tile[:, n0 : n0 + INTER_CHUNK] = pl.cast(h_chunk, target_type=pl.BF16)

                for d0 in pl.parallel(0, D, D_OUT_CHUNK):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_w2_matmul"):
                        h_init = h_tile[:, 0 : INTER_K]
                        w2_init = expert_w2[local_offset, d0 : d0 + D_OUT_CHUNK, 0 : INTER_K]
                        y_acc = pl.matmul(h_init, w2_init, b_trans=True, out_dtype=pl.FP32)
                        for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                            h_k = h_tile[:, k0 : k0 + INTER_K]
                            w2_k = expert_w2[local_offset, d0 : d0 + D_OUT_CHUNK, k0 : k0 + INTER_K]
                            y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="exp_recv_y_accum"):
                        existing = pl.cast(recv_y[:, d0 : d0 + D_OUT_CHUNK], target_type=pl.FP32)
                        summed = pl.add(existing, y_acc)
                        recv_y[:, d0 : d0 + D_OUT_CHUNK] = pl.cast(summed, target_type=pl.BF16)

            # Stage 2: shared expert
            sh_tile = pl.create_tensor([T, MOE_INTER], dtype=pl.BF16)

            for n0 in pl.parallel(0, MOE_INTER, INTER_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_up_matmul"):
                    xs_init = x_local[:, 0 : K_CHUNK]
                    sw1_init = shared_w1[n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                    sw3_init = shared_w3[n0 : n0 + INTER_CHUNK, 0 : K_CHUNK]
                    sh_gate_acc = pl.matmul(xs_init, sw1_init, b_trans=True, out_dtype=pl.FP32)
                    sh_up_acc = pl.matmul(xs_init, sw3_init, b_trans=True, out_dtype=pl.FP32)
                    for k0 in pl.range(K_CHUNK, D, K_CHUNK):
                        xs_k = x_local[:, k0 : k0 + K_CHUNK]
                        sw1_k = shared_w1[n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                        sw3_k = shared_w3[n0 : n0 + INTER_CHUNK, k0 : k0 + K_CHUNK]
                        sh_gate_acc = pl.matmul_acc(sh_gate_acc, xs_k, sw1_k, b_trans=True)
                        sh_up_acc = pl.matmul_acc(sh_up_acc, xs_k, sw3_k, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_swiglu"):
                    sh_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(sh_gate_acc)), 1.0))
                    sh_silu = pl.mul(sh_gate_acc, sh_sigmoid)
                    sh_gated = pl.mul(sh_silu, sh_up_acc)
                    sh_tile[:, n0 : n0 + INTER_CHUNK] = pl.cast(sh_gated, target_type=pl.BF16)

            for d0 in pl.parallel(0, D, D_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_w2_matmul"):
                    hs_init = sh_tile[:, 0 : INTER_K]
                    sw2_init = shared_w2[d0 : d0 + D_OUT_CHUNK, 0 : INTER_K]
                    sh_y_acc = pl.matmul(hs_init, sw2_init, b_trans=True, out_dtype=pl.FP32)
                    for k0 in pl.range(INTER_K, MOE_INTER, INTER_K):
                        hs_k = sh_tile[:, k0 : k0 + INTER_K]
                        sw2_k = shared_w2[d0 : d0 + D_OUT_CHUNK, k0 : k0 + INTER_K]
                        sh_y_acc = pl.matmul_acc(sh_y_acc, hs_k, sw2_k, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_write"):
                    sh[:, d0 : d0 + D_OUT_CHUNK] = pl.cast(sh_y_acc, target_type=pl.BF16)

            return recv_y, sh

    return DeepSeekV4DecodeMoEExpert


def golden_deepseek_v4_decode_moe_expert(tensors):
    """Torch reference (model.py 596-644). recv_y is the partial routed contribution
    only; AllToAllv combine and `+sh` happen in the host orchestrator."""
    import torch
    import torch.nn.functional as F

    recv_x = tensors["recv_x"].float()
    recv_expert_id = tensors["recv_expert_id"]
    recv_weights = tensors["recv_weights"].float()
    x_local = tensors["x_local"].float()
    w1 = tensors["expert_w1"].float()
    w3 = tensors["expert_w3"].float()
    w2 = tensors["expert_w2"].float()
    sw1 = tensors["shared_w1"].float()
    sw3 = tensors["shared_w3"].float()
    sw2 = tensors["shared_w2"].float()

    recv_y = torch.zeros(RECV_TOTAL_MAX, D)
    for local_i in range(EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS):
        local_offset = local_i - EXPERTS_START_IDX
        mask = (recv_expert_id == local_i)
        if mask.sum() == 0:
            continue
        x_sub = recv_x[mask]
        w_sub = recv_weights[mask]
        gate = x_sub @ w1[local_offset].T
        up = x_sub @ w3[local_offset].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        h = h * w_sub.unsqueeze(-1)
        h = h.to(torch.bfloat16).float()
        recv_y[mask] = h @ w2[local_offset].T

    sh_gate = x_local @ sw1.T
    sh_up = x_local @ sw3.T
    sh_h = (F.silu(sh_gate) * sh_up).to(torch.bfloat16).float()
    sh = sh_h @ sw2.T

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)
    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_recv_x():
        return torch.randn(RECV_TOTAL_MAX, D) * 0.05

    def init_recv_expert_id():
        # Global expert ids in [EXPERTS_START_IDX, EXPERTS_START_IDX + N_LOCAL_EXPERTS).
        ids = torch.arange(RECV_TOTAL_MAX, dtype=torch.int32) % N_LOCAL_EXPERTS + EXPERTS_START_IDX
        return ids[torch.randperm(RECV_TOTAL_MAX)]

    def init_recv_weights():
        w = torch.rand(RECV_TOTAL_MAX) + 0.1
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
        TensorSpec("recv_x", [RECV_TOTAL_MAX, D], torch.bfloat16, init_value=init_recv_x),
        TensorSpec("recv_expert_id", [RECV_TOTAL_MAX], torch.int32, init_value=init_recv_expert_id),
        TensorSpec("recv_weights", [RECV_TOTAL_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("x_local", [T, D], torch.bfloat16, init_value=init_x_local),
        TensorSpec("expert_w1", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.bfloat16, init_value=init_w1),
        TensorSpec("expert_w3", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.bfloat16, init_value=init_w3),
        TensorSpec("expert_w2", [N_LOCAL_EXPERTS, D, MOE_INTER], torch.bfloat16, init_value=init_w2),
        TensorSpec("shared_w1", [MOE_INTER, D], torch.bfloat16, init_value=init_sw1),
        TensorSpec("shared_w3", [MOE_INTER, D], torch.bfloat16, init_value=init_sw3),
        TensorSpec("shared_w2", [D, MOE_INTER], torch.bfloat16, init_value=init_sw2),
        TensorSpec("recv_y", [RECV_TOTAL_MAX, D], torch.bfloat16, is_output=True),
        TensorSpec("sh", [T, D], torch.bfloat16, is_output=True),
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
