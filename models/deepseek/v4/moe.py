# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE end-to-end (decode, single-card EP): hc_pre + router +
dispatch + expert + combine + hc_post in one @pl.jit orchestration."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    EP_WORLD_SIZE,
    EP_RANK,
    RECV_MAX,
)
from hc_pre import hc_pre
from hc_post import hc_post
from gate import gate
from dispatch import dispatch
from expert_routed import expert_routed
from expert_shared import expert_shared
from combine import combine


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size

HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim

N_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE   # single-card: router routes over local shard only
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size

MOE_INTER = M.moe_intermediate_size
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS


@pl.jit.inline
def moe(
    x_hc:           pl.Tensor[[T, HC_MULT, D],               pl.BF16],
    hc_ffn_fn:      pl.Tensor[[MIX_HC, HC_DIM],              pl.FP32],
    hc_ffn_scale:   pl.Tensor[[3],                           pl.FP32],
    hc_ffn_base:    pl.Tensor[[MIX_HC],                      pl.FP32],
    norm_w:         pl.Tensor[[D],                           pl.FP32],
    gate_w:         pl.Tensor[[N_EXPERTS, D],                pl.FP32],
    gate_bias:      pl.Tensor[[N_EXPERTS],                   pl.FP32],
    tid2eid:        pl.Tensor[[VOCAB, TOPK],                 pl.INT32],
    input_ids:      pl.Tensor[[T],                           pl.INT64],
    routed_w1:      pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],  pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER],    pl.FP32],
    routed_w3:      pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],  pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER],    pl.FP32],
    routed_w2:      pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER],  pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D],            pl.FP32],
    shared_w1:      pl.Tensor[[MOE_INTER, D],                pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER],                  pl.FP32],
    shared_w3:      pl.Tensor[[MOE_INTER, D],                pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER],                  pl.FP32],
    shared_w2:      pl.Tensor[[D, MOE_INTER],                pl.INT8],
    shared_w2_scale: pl.Tensor[[D],                          pl.FP32],
    x_next:         pl.Tensor[[T, HC_MULT, D],               pl.BF16],
    layer_id:       pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn, comb_ffn,
    )

    # Router emits x_norm plus its per-token INT8 quant so dispatch is a pure scatter.
    x_norm = pl.create_tensor([T, D], dtype=pl.BF16)
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed,
        norm_w, gate_w, gate_bias,
        layer_id,
        tid2eid, input_ids,
        x_norm, x_norm_i8, x_norm_scale, indices, weights,
    )

    sh = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )

    recv_x = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_dq = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
    recv_weights = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.FP32)
    recv_token = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX], dtype=pl.INT32)
    recv_expert_count = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
    dispatch(
        x_norm_i8, x_norm_scale, indices, weights,
        recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count,
    )

    recv_y = pl.create_tensor([N_LOCAL_EXPERTS, RECV_MAX, D], dtype=pl.BF16)
    expert_routed(
        recv_x, recv_scale_dq, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )

    ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    combine(recv_y, recv_token, recv_expert_count, sh, ffn_out)

    x_next = hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next


@pl.jit
def moe_test(
    x_hc:           pl.Tensor[[T, HC_MULT, D],               pl.BF16],
    hc_ffn_fn:      pl.Tensor[[MIX_HC, HC_DIM],              pl.FP32],
    hc_ffn_scale:   pl.Tensor[[3],                           pl.FP32],
    hc_ffn_base:    pl.Tensor[[MIX_HC],                      pl.FP32],
    norm_w:         pl.Tensor[[D],                           pl.FP32],
    gate_w:         pl.Tensor[[N_EXPERTS, D],                pl.FP32],
    gate_bias:      pl.Tensor[[N_EXPERTS],                   pl.FP32],
    layer_id:       pl.Scalar[pl.INT32],
    tid2eid:        pl.Tensor[[VOCAB, TOPK],                 pl.INT32],
    input_ids:      pl.Tensor[[T],                           pl.INT64],
    routed_w1:      pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],  pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER],    pl.FP32],
    routed_w3:      pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D],  pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER],    pl.FP32],
    routed_w2:      pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER],  pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D],            pl.FP32],
    shared_w1:      pl.Tensor[[MOE_INTER, D],                pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER],                  pl.FP32],
    shared_w3:      pl.Tensor[[MOE_INTER, D],                pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER],                  pl.FP32],
    shared_w2:      pl.Tensor[[D, MOE_INTER],                pl.INT8],
    shared_w2_scale: pl.Tensor[[D],                          pl.FP32],
    x_next:         pl.Out[pl.Tensor[[T, HC_MULT, D],        pl.BF16]],
):
    x_next = moe(
        x_hc,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias,
        tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next,
        layer_id,
    )
    return x_next


def golden_moe(tensors):
    """Torch reference: mirrors the DSL stages."""
    import torch

    from hc_pre import golden_hc_pre
    from hc_post import golden_hc_post
    from gate import golden_gate_core
    from dispatch import golden_dispatch
    from expert_routed import golden_expert_routed
    from expert_shared import golden_expert_shared
    from combine import golden_combine

    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x":        tensors["x_hc"],
        "hc_fn":    tensors["hc_ffn_fn"],
        "hc_scale": tensors["hc_ffn_scale"],
        "hc_base":  tensors["hc_ffn_base"],
        "x_mixed":  x_mixed,
        "post":     post_t,
        "comb":     comb_t,
    })

    x_norm = torch.zeros(T, D, dtype=torch.bfloat16)
    x_norm_i8 = torch.zeros(T, D, dtype=torch.int8)
    x_norm_scale = torch.zeros(T, 1, dtype=torch.float32)
    indices = torch.zeros(T, TOPK, dtype=torch.int32)
    weights = torch.zeros(T, TOPK, dtype=torch.float32)
    golden_gate_core({
        "x_mixed":         x_mixed,
        "norm_w":          tensors["norm_w"],
        "gate_w":          tensors["gate_w"],
        "gate_bias":       tensors["gate_bias"],
        "layer_id":        tensors["layer_id"],
        "tid2eid":         tensors["tid2eid"],
        "input_ids":       tensors["input_ids"],
        "x_norm":          x_norm,
        "x_norm_i8":       x_norm_i8,
        "x_norm_scale": x_norm_scale,
        "indices":         indices,
        "weights":         weights,
    })

    sh = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_expert_shared({
        "x_local_i8":       x_norm_i8,
        "x_local_scale_dq": x_norm_scale,
        "shared_w1":        tensors["shared_w1"],
        "shared_w1_scale":  tensors["shared_w1_scale"],
        "shared_w3":        tensors["shared_w3"],
        "shared_w3_scale":  tensors["shared_w3_scale"],
        "shared_w2":        tensors["shared_w2"],
        "shared_w2_scale":  tensors["shared_w2_scale"],
        "sh":               sh,
    })

    recv_x = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.int8)
    recv_scale_dq = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_token = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
    recv_expert_count_actual = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
    golden_dispatch({
        "x_norm_i8":         x_norm_i8,
        "x_norm_scale":   x_norm_scale,
        "indices":           indices,
        "weights":           weights,
        "recv_x":            recv_x,
        "recv_scale_dq":     recv_scale_dq,
        "recv_weights":      recv_weights,
        "recv_token":        recv_token,
        "recv_expert_count": recv_expert_count_actual,
    })

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    golden_expert_routed({
        "recv_x":           recv_x,
        "recv_scale_dq":    recv_scale_dq,
        "recv_weights":     recv_weights,
        "recv_expert_count": recv_expert_count_actual,
        "routed_w1":        tensors["routed_w1"],
        "routed_w1_scale":  tensors["routed_w1_scale"],
        "routed_w3":        tensors["routed_w3"],
        "routed_w3_scale":  tensors["routed_w3_scale"],
        "routed_w2":        tensors["routed_w2"],
        "routed_w2_scale":  tensors["routed_w2_scale"],
        "recv_y":           recv_y,
    })

    ffn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_combine({
        "recv_y":            recv_y,
        "recv_token":        recv_token,
        "recv_expert_count": recv_expert_count_actual,
        "sh":                sh,
        "ffn_out":           ffn_out,
    })

    x_next = torch.zeros(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x":        ffn_out,
        "residual": tensors["x_hc"],
        "post":     post_t,
        "comb":     comb_t,
        "y":        x_next,
    })

    tensors["x_next"][:] = x_next


def build_tensor_specs(layer_id=0):
    import torch
    from golden import ScalarSpec, TensorSpec

    def round_haz(x):
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    def quant_w_per_channel_last(w_bf16):
        amax = w_bf16.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w_bf16.float() * scale_quant.unsqueeze(-1)
        w_i8 = round_haz(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x_hc():           return torch.randn(T, HC_MULT, D)
    def init_hc_ffn_fn():      return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    def init_hc_ffn_scale():   return torch.ones(3) * 0.5
    def init_hc_ffn_base():    return torch.zeros(MIX_HC)
    def init_norm_w():         return torch.ones(D)
    def init_gate_w():         return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():      return torch.zeros(N_EXPERTS)
    def init_tid2eid():
        return torch.randint(0, N_EXPERTS, (VOCAB, TOPK), dtype=torch.int32)
    def init_input_ids():
        return torch.randint(0, VOCAB, (T,), dtype=torch.int64)

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
        TensorSpec("x_hc",          [T, HC_MULT, D],    torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_ffn_fn",     [MIX_HC, HC_DIM],   torch.float32,  init_value=init_hc_ffn_fn),
        TensorSpec("hc_ffn_scale",  [3],                torch.float32,  init_value=init_hc_ffn_scale),
        TensorSpec("hc_ffn_base",   [MIX_HC],           torch.float32,  init_value=init_hc_ffn_base),
        TensorSpec("norm_w",        [D],                torch.float32,  init_value=init_norm_w),
        TensorSpec("gate_w",        [N_EXPERTS, D],     torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",     [N_EXPERTS],        torch.float32,  init_value=init_gate_bias),
        ScalarSpec("layer_id",      torch.int32,        layer_id),
        TensorSpec("tid2eid",       [VOCAB, TOPK],      torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids",     [T],                torch.int64,    init_value=init_input_ids),
        TensorSpec("routed_w1",        [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8,    init_value=lambda: w1_i8),
        TensorSpec("routed_w1_scale",  [N_LOCAL_EXPERTS, MOE_INTER],    torch.float32, init_value=lambda: w1_s),
        TensorSpec("routed_w3",        [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8,    init_value=lambda: w3_i8),
        TensorSpec("routed_w3_scale",  [N_LOCAL_EXPERTS, MOE_INTER],    torch.float32, init_value=lambda: w3_s),
        TensorSpec("routed_w2",        [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8,    init_value=lambda: w2_i8),
        TensorSpec("routed_w2_scale",  [N_LOCAL_EXPERTS, D],            torch.float32, init_value=lambda: w2_s),
        TensorSpec("shared_w1",        [MOE_INTER, D],                  torch.int8,    init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale",  [MOE_INTER],                     torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3",        [MOE_INTER, D],                  torch.int8,    init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale",  [MOE_INTER],                     torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2",        [D, MOE_INTER],                  torch.int8,    init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale",  [D],                             torch.float32, init_value=lambda: sw2_s),
        TensorSpec("x_next",        [T, HC_MULT, D],    torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0,
                        help="layer_id < num_hash_layers picks the hash route; "
                             "≥ num_hash_layers picks the sort route")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the torch RNG for reproducible inputs/routing "
                             "(default: keep random behavior)")
    args = parser.parse_args()

    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)

    result = run_jit(
        fn=moe_test,
        specs=build_tensor_specs(layer_id=args.layer_id),
        golden_fn=golden_moe,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
