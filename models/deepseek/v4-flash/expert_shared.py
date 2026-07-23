# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE shared expert compute (decode, EP single-card).

Split out of ``expert_routed.py``: only the shared-expert FFN path lives here.
The routed local experts are computed by ``expert_routed.py``; both kernels
are composed inside ``moe.py``.

The shared expert reuses the per-token INT8 quant already produced by
``gate`` (``x_norm_i8`` + ``x_norm_scale``) — the same INT8 view
that ``dispatch`` packs for the routed path. This avoids a second
amax+rescale of the same tokens.
"""


import pypto.language as pl

from config import (FLASH as M, MOE_TOKENS, INT8_SCALE_MAX, INT8_AMAX_EPS)


T_DYN = pl.dynamic("EXPERT_SHARED_TOKENS_DYN")


# model config
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# tiling
SH_M_TILE = 16
K_TILE = 512
INTER_K = 512
MM_INTER_TILE = 256
ACT_INTER_TILE = 128
ACT_GATE_INNER = 4
D_OUT_TILE = 256
# h_tile_i8 store innermost = QUANT_TILE bytes (int8); 512 hits the a2a3 L2 cache
# line (perf_hint PH001 flagged the prior 256B store as sub-line).
QUANT_TILE = 512
D_OUT_TILE_ACT = 512
W2_ACT_INNER = 8


@pl.jit.inline
def expert_shared(
    x_local_i8: pl.Tensor,
    x_local_scale_dq: pl.Tensor,
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Tensor,
    late_dep: pl.Scalar[pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x_local_i8, 0)
    n_mtiles = (t_dim + SH_M_TILE - 1) // SH_M_TILE
    t_pad = n_mtiles * SH_M_TILE
    x_local_i8_pad = pl.create_tensor([t_pad, D], dtype=pl.INT8)
    x_local_scale_dq_pad = pl.create_tensor([t_pad, 1], dtype=pl.FP32)
    with pl.spmd(n_mtiles, name_hint="sh_dynamic_pad", deps=[late_dep]) as pad_tid:
        pad_mt = pl.tile.get_block_idx()
        pad_t0 = pad_mt * SH_M_TILE
        pad_rows = pl.min(SH_M_TILE, t_dim - pad_t0)
        for pad_k0 in pl.pipeline(0, D, K_TILE, stage=2):
            pad_x = pl.load(
                x_local_i8,
                [pad_t0, pad_k0],
                [SH_M_TILE, K_TILE],
                valid_shapes=[pad_rows, K_TILE],
            )
            pl.store(pad_x, [pad_t0, pad_k0], x_local_i8_pad)
        for pad_row in pl.range(SH_M_TILE):
            pad_scale = pl.cast(0.0, pl.FP32)
            if pad_t0 + pad_row < t_dim:
                pad_scale = pl.read(x_local_scale_dq, [pad_t0 + pad_row, 0])
            pl.write(x_local_scale_dq_pad, [pad_t0 + pad_row, 0], pad_scale)

    # One M-tile of SH_M_TILE rows per iteration (decode: 1 tile, T<=16 rows valid;
    # prefill: T_PAD/SH_M_TILE fully-valid tiles).
    for mt in pl.parallel(n_mtiles):
        ts0 = mt * SH_M_TILE
        valid_rows = pl.min(SH_M_TILE, t_dim - ts0)

        h_tile_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        gate_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)
        up_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)
        sh_tile_buf = pl.create_tensor([SH_M_TILE, D], dtype=pl.BF16)

        # gate (w1) cube matmul -> INT32 GM accumulator.
        with pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="sh_gate_mm", deps=[pad_tid]) as gate_tid:
            nb_idx = pl.tile.get_block_idx()
            n0 = nb_idx * MM_INTER_TILE
            gate_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = x_local_i8_pad[ts0 : ts0 + SH_M_TILE, k0 : k0 + K_TILE]
                sw1_k = shared_w1[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    gate_acc = pl.matmul(xs_k, sw1_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    gate_acc = pl.matmul_acc(gate_acc, xs_k, sw1_k, b_trans=True)
            gate_i32[:, n0 : n0 + MM_INTER_TILE] = gate_acc

        # up (w3) cube matmul -> INT32 GM accumulator.
        with pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="sh_up_mm", deps=[pad_tid]) as up_tid:
            nb_idx = pl.tile.get_block_idx()
            n0 = nb_idx * MM_INTER_TILE
            up_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = x_local_i8_pad[ts0 : ts0 + SH_M_TILE, k0 : k0 + K_TILE]
                sw3_k = shared_w3[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    up_acc = pl.matmul(xs_k, sw3_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    up_acc = pl.matmul_acc(up_acc, xs_k, sw3_k, b_trans=True)
            up_i32[:, n0 : n0 + MM_INTER_TILE] = up_acc

        # SwiGLU activation (dequant gate/up, clamp, silu*up) -> FP32 GM.
        with pl.spmd(
            MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE),
            name_hint="sh_gate_up_act",
            deps=[pad_tid, gate_tid, up_tid],
        ) as _act_tid:
            nb_idx = pl.tile.get_block_idx()
            n_base = nb_idx * (ACT_GATE_INNER * ACT_INTER_TILE)
            for ng in pl.pipeline(ACT_GATE_INNER, stage=2):
                n0 = n_base + ng * ACT_INTER_TILE
                gate_2d_i32 = gate_i32[:, n0 : n0 + ACT_INTER_TILE]
                up_2d_i32 = up_i32[:, n0 : n0 + ACT_INTER_TILE]
                x_local_scale_dq_tile = x_local_scale_dq_pad[ts0 : ts0 + SH_M_TILE, 0:1]
                w1_scale_chunk = pl.reshape(shared_w1_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                w3_scale_chunk = pl.reshape(shared_w3_scale[n0 : n0 + ACT_INTER_TILE], [1, ACT_INTER_TILE])
                gate_2d = pl.cast(gate_2d_i32, target_type=pl.FP32, mode="none")
                up_2d = pl.cast(up_2d_i32, target_type=pl.FP32, mode="none")
                gate_2d = pl.col_expand_mul(pl.row_expand_mul(gate_2d, x_local_scale_dq_tile), w1_scale_chunk)
                up_2d = pl.col_expand_mul(pl.row_expand_mul(up_2d, x_local_scale_dq_tile), w3_scale_chunk)
                if SWIGLU_LIMIT > 0.0:
                    gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                    up_2d = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                silu = pl.mul(gate_2d, sigmoid)
                gated = pl.mul(silu, up_2d)
                h_tile_fp32[:, n0 : n0 + ACT_INTER_TILE] = gated

        # Per-row A8 requant of h_tile (amax across full MOE_INTER row).
        h_tile_i8 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT8)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_h_q"):
            eh_amax = pl.full([1, SH_M_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                eh_a_f32 = h_tile_fp32[:, k0 : k0 + QUANT_TILE]
                eh_a_abs = pl.maximum(eh_a_f32, pl.neg(eh_a_f32))
                eh_a_max = pl.reshape(pl.row_max(eh_a_abs), [1, SH_M_TILE])
                eh_amax = pl.maximum(eh_amax, eh_a_max)
            eh_sq_row = pl.div(pl.full([1, SH_M_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), eh_amax)
            h_tile_scale_dq = pl.reshape(pl.recip(eh_sq_row), [SH_M_TILE, 1])
            eh_sq_col = pl.reshape(eh_sq_row, [SH_M_TILE, 1])
            for k1 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                eh_q_f32 = h_tile_fp32[:, k1 : k1 + QUANT_TILE]
                eh_q_scaled = pl.row_expand_mul(eh_q_f32, eh_sq_col)
                eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode="rint")
                eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode="round")
                h_tile_i8[:, k1 : k1 + QUANT_TILE] = pl.cast(eh_q_half, target_type=pl.INT8, mode="trunc")

        # w2 (down) cube matmul -> INT32 GM accumulator.
        y_i32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.INT32)
        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="sh_w2_mm"):
            d0 = db_idx * D_OUT_TILE
            y_acc = pl.create_tensor([SH_M_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                hs_k = h_tile_i8[:, k0 : k0 + INTER_K]
                sw2_k = shared_w2[d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                if k0 == 0:
                    y_acc = pl.matmul(hs_k, sw2_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    y_acc = pl.matmul_acc(y_acc, hs_k, sw2_k, b_trans=True)
            y_i32[:, d0 : d0 + D_OUT_TILE] = y_acc

        # Dequant w2 output (per-row h scale x per-channel w2 scale) -> BF16.
        for db_idx in pl.spmd(D // (W2_ACT_INNER * D_OUT_TILE_ACT), name_hint="sh_w2_act"):
            d_base = db_idx * (W2_ACT_INNER * D_OUT_TILE_ACT)
            for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                d0 = d_base + dg * D_OUT_TILE_ACT
                y_2d_i32 = y_i32[:, d0 : d0 + D_OUT_TILE_ACT]
                w2_scale_chunk = pl.reshape(shared_w2_scale[d0 : d0 + D_OUT_TILE_ACT], [1, D_OUT_TILE_ACT])
                y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, h_tile_scale_dq), w2_scale_chunk)
                # Materialize the Tensor-world activation into explicit scratch before the dynamic
                # Tile writeback. This prevents tensor-to-tile conversion from folding the load back
                # into y_bf16 and changing its type across lowering passes.
                y_bf16 = pl.cast(y_2d, target_type=pl.BF16, mode="rint")
                sh_tile_buf = pl.assemble(sh_tile_buf, y_bf16, [0, d0])
                y_bf16_tile = pl.load(
                    sh_tile_buf,
                    [0, d0],
                    [SH_M_TILE, D_OUT_TILE_ACT],
                    valid_shapes=[valid_rows, D_OUT_TILE_ACT],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(y_bf16_tile, [ts0, d0], sh)

    # The @pl.inline parser requires inline call expressions to have a return
    # value; sh is convenient because it's already pl.Out.
    return sh


@pl.jit
def expert_shared_test(
    x_local_i8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[T_DYN, 1], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    sh: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x_local_i8.bind_dynamic(0, T_DYN)
    x_local_scale_dq.bind_dynamic(0, T_DYN)
    sh.bind_dynamic(0, T_DYN)

    late_dep = pl.system.task_dummy(deps=[])
    expert_shared(
        x_local_i8, x_local_scale_dq,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh, late_dep,
    )
    return sh


def _int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def golden_expert_shared(tensors):
    """Torch reference for the shared expert.

    Input is the per-token INT8 quant produced by gate (shared with
    dispatch / routed expert); we dequant inside to match the kernel's
    dequant-then-matmul pattern."""
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)

    x_local_i8 = tensors["x_local_i8"]                       # [T, D] int8
    x_local_scale_dq = tensors["x_local_scale_dq"].float()   # [T, 1]
    x_local = x_local_i8.float() * x_local_scale_dq
    sw1 = dequant_w(tensors["shared_w1"], tensors["shared_w1_scale"].float())
    sw3 = dequant_w(tensors["shared_w3"], tensors["shared_w3_scale"].float())
    sw2 = dequant_w(tensors["shared_w2"], tensors["shared_w2_scale"].float())

    sh_gate = x_local @ sw1.T
    sh_up = x_local @ sw3.T
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up
    sh_h_i8, sh_h_sd = _int8_quant_per_row(sh_h)
    sh_h = sh_h_i8.float() * sh_h_sd
    sh = sh_h @ sw2.T

    tensors["sh"][:] = sh.to(torch.bfloat16)


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize a shared-expert per-channel-symmetric INT8 weight + FP32 scale by
    simulating the real DeepSeek-V4-Flash MXFP8 shared-expert quant grid (e4m3, 128x128-block
    E8M0 scale), then re-quantizing per-output-channel. Unlike routed (MXFP4 -> ~37 discrete
    levels), shared stays near-Gaussian (~200 levels). The coarse 128-block scale does NOT
    flatten the real per-output-channel magnitude spread, so ``chan_cv`` (log-space source-gain
    std) injects it to reproduce the real INT8 scale CV (~0.5 gate/up, ~0.35 down). Per-output-
    channel INT8 is scale-invariant, so the grid sets the level shape and ``dequant_std`` only
    sets the absolute scale magnitude. (routed experts use a different grid -- see
    expert_routed.gen_routed_weight.)

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel
    scale shape ([out, in] -> scale [out]).
    """
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):   # e4m3 + 128x128-block E8M0 (round-up) scale on (out, in)
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))  # per-channel gain
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def build_tensor_specs(num_tokens=MOE_TOKENS):
    import torch
    from golden import TensorSpec

    # Pre-quantize x_local once so the i8 / scale specs see consistent values
    # (mirrors what gate produces in the full pipeline).
    x_local_bf16 = torch.randn(num_tokens, D, dtype=torch.bfloat16)
    x_local_i8_pre, x_local_sd_pre = _int8_quant_per_row(x_local_bf16)

    # Synthesize (int8, per-channel scale) by simulating the real MXFP8 shared-expert
    # quant grid (gen_shared_weight). chan_cv reproduces the real per-output-channel scale
    # CV (~0.5 gate/up, ~0.35 down) the coarse FP8 block scale leaves behind.
    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_i8, sw1_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3_i8, sw3_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2_i8, sw2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], chan_cv=0.33)

    return [
        TensorSpec("x_local_i8", [num_tokens, D], torch.int8, init_value=lambda: x_local_i8_pre),
        TensorSpec("x_local_scale_dq", [num_tokens, 1], torch.float32, init_value=lambda: x_local_sd_pre.float()),
        TensorSpec("shared_w1", [MOE_INTER, D], torch.int8, init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale", [MOE_INTER], torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3", [MOE_INTER, D], torch.int8, init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale", [MOE_INTER], torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2", [D, MOE_INTER], torch.int8, init_value=lambda: sw2_i8),
        TensorSpec("shared_w2_scale", [D], torch.float32, init_value=lambda: sw2_s),
        TensorSpec("sh", [num_tokens, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=MOE_TOKENS)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive")

    result = run_jit(
        fn=expert_shared_test,
        specs=build_tensor_specs(num_tokens=args.num_tokens),
        golden_fn=golden_expert_shared,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 sh, ~1 ULP. Gen weights reproduce real(L21): 0.01% vs 0.004% of points > 1e-3.
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
