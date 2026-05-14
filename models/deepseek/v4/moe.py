# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE end-to-end path (decode, EP single-card).

Connects ``hc_pre`` + router + dispatch + expert + combine + ``hc_post`` inside
one ``@pl.jit`` orchestration.
The local EP=1 dispatch contract is:

    for p = t * TOPK + k:
        e = indices_flat[p]
        slot = recv_expert_count[e]
        recv_x[e, slot, :] = x_norm[t, :]
        recv_weights[e, slot] = weights_flat[p]
        recv_token[e, slot] = t
        recv_expert_count[e] += 1

``moe_expert`` multiplies ``recv_weights`` into the routed expert activation.
The current integration deliberately lets the expert process all ``RECV_MAX``
rows per expert, with non-routed tail rows initialized to zero. The actual
``recv_expert_count`` is still used by combine, so only valid packed rows are
scattered back to tokens.
"""


import pypto.language as pl

from config import (
    DEMO as M,
    DECODE_BATCH,
    DECODE_SEQ,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from hc_pre import hc_pre
from hc_post import hc_post
from moe_router import moe_router
from moe_dispatch import moe_dispatch
from moe_expert import moe_expert
from moe_combine import moe_combine


# --- Shared shape constants. Must agree with the imported inlines. ---
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size

# hc_pre (ffn) weights
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim

# Router
N_EXPERTS = M.n_routed_experts
TOPK = M.num_experts_per_tok

# Expert (must match moe_expert.py)
MOE_INTER = M.moe_intermediate_size
EP_WORLD_SIZE = 1
N_LOCAL_EXPERTS = N_EXPERTS // EP_WORLD_SIZE
RECV_MAX = 32

# Sanity: chosen layout requires RECV_MAX >= T * TOPK.
assert RECV_MAX >= T * TOPK, "packed layout needs RECV_MAX >= T * TOPK"


@pl.jit
def moe(
    # ---- router (hc_pre + ffn_norm + gate) inputs ----
    x_hc:           pl.Tensor[[B, S, HC_MULT, D],            pl.BF16],
    hc_ffn_fn:      pl.Tensor[[MIX_HC, HC_DIM],              pl.FP32],
    hc_ffn_scale:   pl.Tensor[[3],                           pl.FP32],
    hc_ffn_base:    pl.Tensor[[MIX_HC],                      pl.FP32],
    norm_w:         pl.Tensor[[D],                           pl.FP32],
    gate_w:         pl.Tensor[[N_EXPERTS, D],                pl.FP32],
    gate_bias:      pl.Tensor[[N_EXPERTS],                   pl.FP32],
    # ---- expert weights ----
    expert_w1:      pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],  pl.INT8],
    expert_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER],    pl.FP32],
    expert_w3:      pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],  pl.INT8],
    expert_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER],    pl.FP32],
    expert_w2:      pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER],  pl.INT8],
    expert_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D],            pl.FP32],
    shared_w1:      pl.Tensor[[MOE_INTER, D],                pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER],                  pl.FP32],
    shared_w3:      pl.Tensor[[MOE_INTER, D],                pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER],                  pl.FP32],
    shared_w2:      pl.Tensor[[D, MOE_INTER],                pl.INT8],
    shared_w2_scale: pl.Tensor[[D],                          pl.FP32],
    recv_expert_count_full: pl.Tensor[[N_LOCAL_EXPERTS, 1],  pl.INT32],
    # ---- outputs ----
    x_next:         pl.Out[pl.Tensor[[B, S, HC_MULT, D],     pl.BF16]],
    ffn_out:        pl.Out[pl.Tensor[[T, D],                 pl.BF16]],
    post_ffn:       pl.Out[pl.Tensor[[B, S, HC_MULT],        pl.FP32]],
    comb_ffn:       pl.Out[pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32]],
):
    # Stage 1: hc_pre(ffn) -> x_mixed, post_ffn, comb_ffn
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post_ffn_tmp = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb_ffn_tmp = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn_tmp, comb_ffn_tmp,
    )

    # Stage 2: router kernel -> x_norm and compact route tables.
    x_norm = pl.create_tensor([T, D], dtype=pl.BF16)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    moe_router(
        x_mixed,
        norm_w, gate_w, gate_bias,
        x_norm, indices, weights,
    )

    # Stage 3: packed dispatch.
    recv_x = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.BF16)
    recv_weights = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
    recv_token = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
    recv_expert_count = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
    moe_dispatch(
        x_norm, indices, weights,
        recv_x, recv_weights, recv_token, recv_expert_count,
    )

    # Stage 4: routed local experts + shared expert. Use a full count for the
    # expert loop so `moe_expert` does not need to derive its dynamic tile count
    # from an internal dispatch-produced tensor. Tail rows are zero weighted.
    recv_y = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.BF16)
    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    moe_expert(
        recv_x, recv_weights, recv_expert_count_full, x_norm,
        expert_w1, expert_w1_scale, expert_w3, expert_w3_scale,
        expert_w2, expert_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        recv_y, sh,
    )

    # Stage 5: combine routed and shared expert outputs.
    moe_combine(recv_y, recv_token, recv_expert_count, sh, ffn_out)

    # Stage 6: hc_post(ffn) merges the FFN output back into the HC stack.
    ffn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    ffn_out_3d = pl.reshape(ffn_out, [B, S, D])
    x_next = hc_post(ffn_out_3d, x_hc, post_ffn_tmp, comb_ffn_tmp, x_next)
    post_ffn[:, :, :] = post_ffn_tmp
    comb_ffn[:, :, :, :] = comb_ffn_tmp
    return x_next, ffn_out, post_ffn, comb_ffn


# =============================================================================
# Golden (torch reference): mirrors the DSL stages.
# =============================================================================
def golden_moe(tensors):
    import torch

    from hc_pre import golden_hc_pre
    from hc_post import golden_hc_post
    from moe_router import golden_moe_router_core
    from moe_dispatch import golden_moe_dispatch
    from moe_expert import golden_moe_expert
    from moe_combine import golden_moe_combine

    # Stage 1: hc_pre.
    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x":        tensors["x_hc"],
        "hc_fn":    tensors["hc_ffn_fn"],
        "hc_scale": tensors["hc_ffn_scale"],
        "hc_base":  tensors["hc_ffn_base"],
        "x_mixed":  x_mixed,
        "post":     post_t,
        "comb":     comb_t,
    })

    # Stage 2: router.
    x_norm = torch.zeros(T, D, dtype=torch.bfloat16)
    indices = torch.zeros(T, TOPK, dtype=torch.int32)
    weights = torch.zeros(T, TOPK, dtype=torch.float32)
    golden_moe_router_core({
        "x_mixed":      x_mixed,
        "norm_w":       tensors["norm_w"],
        "gate_w":       tensors["gate_w"],
        "gate_bias":    tensors["gate_bias"],
        "x_norm":       x_norm,
        "indices":      indices,
        "weights":      weights,
    })

    # Stage 3: packed dispatch.
    recv_x = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    recv_weights = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_token = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
    recv_expert_count_actual = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
    golden_moe_dispatch({
        "x_norm":            x_norm,
        "indices":           indices,
        "weights":           weights,
        "recv_x":            recv_x,
        "recv_weights":      recv_weights,
        "recv_token":        recv_token,
        "recv_expert_count": recv_expert_count_actual,
    })

    # Stage 4: experts.
    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    sh = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_moe_expert({
        "recv_x":           recv_x,
        "recv_weights":     recv_weights,
        "recv_expert_count": tensors["recv_expert_count_full"],
        "x_local":          x_norm,
        "expert_w1":        tensors["expert_w1"],
        "expert_w1_scale":  tensors["expert_w1_scale"],
        "expert_w3":        tensors["expert_w3"],
        "expert_w3_scale":  tensors["expert_w3_scale"],
        "expert_w2":        tensors["expert_w2"],
        "expert_w2_scale":  tensors["expert_w2_scale"],
        "shared_w1":        tensors["shared_w1"],
        "shared_w1_scale":  tensors["shared_w1_scale"],
        "shared_w3":        tensors["shared_w3"],
        "shared_w3_scale":  tensors["shared_w3_scale"],
        "shared_w2":        tensors["shared_w2"],
        "shared_w2_scale":  tensors["shared_w2_scale"],
        "recv_y":           recv_y,
        "sh":               sh,
    })

    # Stage 5: combine routed and shared expert outputs.
    ffn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_moe_combine({
        "recv_y":            recv_y,
        "recv_token":        recv_token,
        "recv_expert_count": recv_expert_count_actual,
        "sh":                sh,
        "ffn_out":           ffn_out,
    })

    # Stage 6: hc_post(ffn).
    x_next = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x":        ffn_out.reshape(B, S, D),
        "residual": tensors["x_hc"],
        "post":     post_t,
        "comb":     comb_t,
        "y":        x_next,
    })

    tensors["x_next"][:]  = x_next
    tensors["ffn_out"][:] = ffn_out
    tensors["post_ffn"][:] = post_t
    tensors["comb_ffn"][:] = comb_t


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def round_haz(x):
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    def quant_w_per_channel_last(w_bf16):
        amax = w_bf16.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w_bf16.float() * scale_quant.unsqueeze(-1)
        w_i8 = round_haz(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x_hc():           return torch.randn(B, S, HC_MULT, D) * 0.05
    def init_hc_ffn_fn():      return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    def init_hc_ffn_scale():   return torch.ones(3) * 0.5
    def init_hc_ffn_base():    return torch.zeros(MIX_HC)
    def init_norm_w():         return torch.ones(D)
    def init_gate_w():         return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():      return torch.zeros(N_EXPERTS)
    def init_recv_expert_count_full():
        return torch.full((N_LOCAL_EXPERTS, 1), RECV_MAX, dtype=torch.int32)

    w1_bf16 = (torch.randn(N_LOCAL_EXPERTS, MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    w3_bf16 = (torch.randn(N_LOCAL_EXPERTS, MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    w2_bf16 = (torch.randn(N_LOCAL_EXPERTS, D, MOE_INTER) / MOE_INTER ** 0.5).to(torch.bfloat16)
    sw1_bf16 = (torch.randn(MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    sw3_bf16 = (torch.randn(MOE_INTER, D) / D ** 0.5).to(torch.bfloat16)
    sw2_bf16 = (torch.randn(D, MOE_INTER) / MOE_INTER ** 0.5).to(torch.bfloat16)
    w1_i8, w1_s = quant_w_per_channel_last(w1_bf16)
    w3_i8, w3_s = quant_w_per_channel_last(w3_bf16)
    w2_i8, w2_s = quant_w_per_channel_last(w2_bf16)
    sw1_i8, sw1_s = quant_w_per_channel_last(sw1_bf16)
    sw3_i8, sw3_s = quant_w_per_channel_last(sw3_bf16)
    sw2_i8, sw2_s = quant_w_per_channel_last(sw2_bf16)

    return [
        TensorSpec("x_hc",          [B, S, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",     [MIX_HC, HC_DIM],   torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale",  [3],                torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",   [MIX_HC],           torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",        [D],                torch.float32,  init_value=init_norm_w),
        TensorSpec("gate_w",        [N_EXPERTS, D],     torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",     [N_EXPERTS],        torch.float32,  init_value=init_gate_bias),
        TensorSpec("expert_w1",        [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8,    init_value=lambda: w1_i8),
        TensorSpec("expert_w1_scale",  [N_LOCAL_EXPERTS, MOE_INTER],    torch.float32, init_value=lambda: w1_s),
        TensorSpec("expert_w3",        [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8,    init_value=lambda: w3_i8),
        TensorSpec("expert_w3_scale",  [N_LOCAL_EXPERTS, MOE_INTER],    torch.float32, init_value=lambda: w3_s),
        TensorSpec("expert_w2",        [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8,    init_value=lambda: w2_i8),
        TensorSpec("expert_w2_scale",  [N_LOCAL_EXPERTS, D],            torch.float32, init_value=lambda: w2_s),
        TensorSpec("shared_w1",        [MOE_INTER, D],                  torch.int8,    init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale",  [MOE_INTER],                     torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3",        [MOE_INTER, D],                  torch.int8,    init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale",  [MOE_INTER],                     torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2",        [D, MOE_INTER],                  torch.int8,    init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale",  [D],                             torch.float32, init_value=lambda: sw2_s),
        TensorSpec("recv_expert_count_full", [N_LOCAL_EXPERTS, 1],       torch.int32,   init_value=init_recv_expert_count_full),
        TensorSpec("x_next",        [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
        TensorSpec("ffn_out",       [T, D],             torch.bfloat16, is_output=True),
        TensorSpec("post_ffn",      [B, S, HC_MULT],    torch.float32,  is_output=True),
        TensorSpec("comb_ffn",      [B, S, HC_MULT, HC_MULT], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import RunConfig, run_jit

    def allclose_with_tolerance(rel_tol, abs_tol):
        def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            actual_f = actual.float()
            expected_f = expected.float()
            close = torch.isclose(actual_f, expected_f, rtol=rel_tol, atol=abs_tol)
            if bool(close.all().item()):
                return True, ""

            mismatch_indices = torch.where(~close.flatten())[0]
            flat_actual = actual.flatten()
            flat_expected = expected.flatten()
            n_show = min(20, mismatch_indices.numel())
            idx = mismatch_indices[:n_show]
            lines = [
                f"    [{i.item()}] actual={flat_actual[i].item()}, expected={flat_expected[i].item()}"
                for i in idx
            ]
            detail = (
                f"    Mismatched elements: {mismatch_indices.numel()}/{actual.numel()}  "
                f"rtol={rel_tol} atol={abs_tol}\n"
                f"    first {n_show} mismatches:\n" + "\n".join(lines)
            )
            return False, detail

        cmp.__name__ = f"allclose_with_tolerance(rtol={rel_tol},atol={abs_tol})"
        return cmp

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=moe,
        specs=build_tensor_specs(),
        golden_fn=golden_moe,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
            compare_fn={
                "ffn_out": allclose_with_tolerance(1e-2, 5e-2),
                "x_next": allclose_with_tolerance(1e-2, 5e-2),
            },
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
