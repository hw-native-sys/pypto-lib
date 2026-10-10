# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sigmoid ``noaux_tc`` router over 288 experts, fused with the deferred norm and quant.

``scores = sigmoid(x @ gate_weight.T)`` in FP32; selection ranks
``scores + e_score_correction_bias`` and takes the top 8; the returned weights are
the *unbiased* scores, renormalised (``norm_topk_prob``) and scaled by
``routed_scaling_factor = 2.5``. ``n_group`` and ``topk_group`` are both 1, so the
grouped masking of ``DeepseekV3TopkRouter`` degenerates to a plain top-k.

``mlp.gate.weight`` is stored BF16 ``[288, 4096]`` and ``e_score_correction_bias``
is FP32 ``[288]``; ``moe_router_dtype`` is ``float32``, so the matmul accumulates
and everything downstream of it stays in FP32.

Like the a2a3 sibling's ``models/deepseek_v4_flash_mtp/gate.py``, this kernel also
owns the deferred RMSNorm and the per-token INT8 quantization, so the INT8 view is
produced once and reused by both the shared expert and the EP dispatch payload.

The sort is the one place the 288-expert row departs from the sibling. That
sibling pads to ``SCORE_PAD = 256`` and stops after two ``pl.mrgsort`` stages;
here the row is padded to 512 and the ``pl.sort32`` output (16 runs of 64 lanes)
is merged 4-way twice, ``block_len=64`` then ``block_len=256``, ending as one
1024-lane run.
"""

import sys
from pathlib import Path

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, FLASH, FP32_NEG_INF, N_EXPERTS, T_DYN, TOPK
from models.glm5_3_flash.golden import gate, rms_norm
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX

# This directory owns a ``golden.py`` reference module, so the repository-root
# ``golden`` harness package must come first on the path before any harness import.
# The script directory sits at sys.path[0], ahead of PYTHONPATH, so an insert is
# required even when the root already appears there.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCORE_PAD = 512  # 288 experts padded to the next sort32-friendly width

# tiling
GATE_T_TILE = 8         # router sort / write-back token tile
GATE_M_TILE = 16        # cube M-tile: matmul rows must be a multiple of 16 (fractal)
GATE_N_TILE = 16        # expert columns per gate spmd block
assert N_EXPERTS % GATE_N_TILE == 0
GATE_D_TILE = 512
assert D % GATE_D_TILE == 0
ROW_PAD = 8
FFN_REDUCE_TILE = D // ROW_PAD
QUANT_TILE = 256
TOPK_PAD = 8            # TOPK padded to 32B-aligned width
SORT_PAD = TOPK_PAD * 2  # (val, idx) interleaved slice width
assert TOPK <= TOPK_PAD

NORM_EPS = FLASH.rms_norm_eps
ROUTE_SCALE = FLASH.routed_scaling_factor

T_DYN_TEST = 64


def x_rows(tensors):
    return int(tensors["x"].shape[0])


def gate_active_rows(num_tokens):
    """Rows the kernel must produce: the clamped active token count.

    The 8-row quant tiles leave rows past ``num_tokens`` untouched, so the
    comparison must stop at the active count rather than the padded tile width.
    """
    return max(0, int(num_tokens))


def golden_gate(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gate(rms_norm(x, norm_weight), gate_weight, correction_bias)


@pl.jit.inline
def moe_gate(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    route_weights: pl.Tensor[[T_DYN, TOPK], pl.FP32],
    route_indices: pl.Tensor[[T_DYN, TOPK], pl.INT32],
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    token_rows = pl.tensor.dim(x, 0)
    padded_rows = ((token_rows + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
    # Deferred RMSNorm (qwen3-style): store xg = x*gamma (NOT *inv_rms), because
    # the per-token positive scalar inv_rms factors out of everything downstream:
    #   - gate logits: inv_rms * (xg @ gate_w.T)  -> applied as a [16,1] row-scale
    #   - int8 quant : symmetric per-token quant of xg CANCELS inv_rms exactly;
    #                  inv_rms rides only x_scale (= inv_rms * amax(xg)/127).
    # This lets sq_sum (needs raw x) and xg share ONE pass over x instead of
    # a sqsum pass followed by a separate normalize pass.
    xg_buf = pl.create_tensor([padded_rows, D], dtype=pl.FP32)
    inv_rms_buf = pl.create_tensor([padded_rows, 1], dtype=pl.FP32)
    # per-token int8 quant scale (= INT8_SCALE_MAX / amax(xg)), computed in ffn_norm
    # and consumed by x_quant so quant skips its own amax pass.
    xn_scale_buf = pl.create_tensor([padded_rows, 1], dtype=pl.FP32)
    route_scores_buf = pl.create_tensor([padded_rows, SCORE_PAD], dtype=pl.FP32)
    biased_scores_buf = pl.create_tensor([padded_rows, SCORE_PAD], dtype=pl.FP32)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > token_rows:
        active_tokens = pl.cast(token_rows, pl.INDEX)
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    active_gate_tokens = active_gate_tiles * GATE_M_TILE
    if active_gate_tokens > token_rows:
        active_gate_tokens = pl.cast(token_rows, pl.INDEX)

    # One token per core with two-level full-row reductions.
    norm_w_2d = pl.reshape(norm_weight, [1, D])
    for tok in pl.spmd(active_gate_tokens, name_hint="ffn_norm", allow_early_resolve=True):
        rms_x = pl.cast(pl.tile.load(x, [tok, 0], [1, D]), pl.FP32)
        rms_w = pl.cast(pl.tile.load(norm_w_2d, [0, 0], [1, D]), pl.FP32)
        xg = pl.mul(rms_x, rms_w)
        pl.tile.store(xg, [tok, 0], xg_buf, shapes=[1, D])

        sq_rows = pl.reshape(pl.mul(rms_x, rms_x), [ROW_PAD, FFN_REDUCE_TILE])
        sq_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
        sq_partial = pl.row_sum(sq_rows, sq_partial_tmp)
        sq_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        sq_reduce[0:1, :] = pl.reshape(sq_partial, [1, ROW_PAD])
        sq_reduce = pl.set_validshape(sq_reduce, 1, ROW_PAD)
        sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        sq_sum = pl.row_sum(sq_reduce, sq_sum_tmp)
        sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
        inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
        pl.tile.store(inv_rms, [tok, 0], inv_rms_buf, shapes=[1, 1])

        xg_abs_rows = pl.reshape(pl.abs(xg), [ROW_PAD, FFN_REDUCE_TILE])
        amax_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
        amax_partial = pl.row_max(xg_abs_rows, amax_partial_tmp)
        amax_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        amax_reduce[0:1, :] = pl.reshape(amax_partial, [1, ROW_PAD])
        amax_reduce = pl.set_validshape(amax_reduce, 1, ROW_PAD)
        amax_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
        xg_amax = pl.row_max(amax_reduce, amax_tmp)
        xg_amax = pl.set_validshape(pl.reshape(xg_amax, [1, ROW_PAD]), 1, 1)
        amax_eps = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_AMAX_EPS)
        amax_eps = pl.set_validshape(amax_eps, 1, 1)
        xg_amax = pl.maximum(xg_amax, amax_eps)
        # quant scale = INT8_SCALE_MAX / amax(xg); dequant scale rides inv_rms.
        scale_max = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX)
        scale_max = pl.set_validshape(scale_max, 1, 1)
        xg_sq = pl.div(scale_max, xg_amax)
        xg_dequant_scale = pl.mul(xg_amax, 1.0 / INT8_SCALE_MAX)
        x_dequant_scale = pl.mul(xg_dequant_scale, inv_rms)
        pl.tile.store(x_dequant_scale, [tok, 0], x_scale, shapes=[1, 1])
        pl.tile.store(xg_sq, [tok, 0], xn_scale_buf, shapes=[1, 1])

    seed_dummy = pl.system.task_dummy(deps=[])

    # Per-token symmetric INT8 quant of xg: scale precomputed in ffn_norm. inv_rms
    # cancels here (symmetric quant is invariant to a positive per-token scalar),
    # so x_int8 = quant(xg). Early resolution lets the shared-expert chain
    # pre-stage before dispatch_push.
    for quant_block in pl.spmd(
        (active_gate_tokens + GATE_T_TILE - 1) // GATE_T_TILE,
        name_hint="x_quant",
        deps=[seed_dummy],
        allow_early_resolve=True,
    ):
        t0 = quant_block * GATE_T_TILE
        # The last block may cover fewer than GATE_T_TILE active rows; writing the
        # full tile would run past the T_DYN row bound of x_int8.
        quant_rows = pl.min(pl.cast(GATE_T_TILE, pl.INDEX), active_gate_tokens - t0)
        xn_sq_col = xn_scale_buf[t0 : t0 + GATE_T_TILE, 0:1]
        for xq_b_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
            xn_q_scaled = pl.row_expand_mul(xg_buf[t0 : t0 + GATE_T_TILE, xq_b_k : xq_b_k + QUANT_TILE], xn_sq_col)
            xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
            xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
            xn_q_i8 = pl.cast(xn_q_half, pl.INT8, mode="trunc")
            x_int8 = pl.assemble(x_int8, pl.set_validshape(xn_q_i8, quant_rows, QUANT_TILE), [t0, xq_b_k])

    # Pre-route setup: zero the inactive-token outputs and NEG_INF the biased pad
    # columns so the sort ranks pad experts last. Route write-backs are guarded to
    # active tokens, so the inactive-zero can run here rather than post-route.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
        for zt in pl.range(token_rows):
            if zt >= active_tokens:
                pl.write(x_scale, [zt, 0], pl.cast(0.0, pl.FP32))
                for zk in pl.range(TOPK):
                    pl.write(route_indices, [zt, zk], pl.cast(0, pl.INT32))
                    pl.write(route_weights, [zt, zk], pl.cast(0.0, pl.FP32))
        if N_EXPERTS < SCORE_PAD:
            # A one-shot pl.full over all padded rows materializes the whole tile in
            # the Vec buffer: at the CP forward's token rows a wide pad can exceed
            # the Vec budget. Fill the same columns a row block at a time; the
            # bytes stored are the same.
            for pad_block in pl.range(padded_rows // GATE_M_TILE):
                pad_t0 = pad_block * GATE_M_TILE
                biased_scores_buf[pad_t0 : pad_t0 + GATE_M_TILE, N_EXPERTS:SCORE_PAD] = pl.full(
                    [GATE_M_TILE, SCORE_PAD - N_EXPERTS],
                    dtype=pl.FP32,
                    value=FP32_NEG_INF,
                )

    # Gate matmul: xg @ gate_w.T → FP32 logits, then sigmoid scores. The matmul
    # input is the BF16 view of xg (gate_w is BF16 too); inv_rms rides the logits
    # as a per-token row scale, and only the *biased* scores feed the sort.
    for gb_idx in pl.spmd(active_gate_tiles * (N_EXPERTS // GATE_N_TILE), name_hint="gate", allow_early_resolve=True):
        tg = gb_idx // (N_EXPERTS // GATE_N_TILE)
        nb = gb_idx % (N_EXPERTS // GATE_N_TILE)
        t1 = tg * GATE_M_TILE
        n0 = nb * GATE_N_TILE
        gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // GATE_D_TILE, stage=2):
            gd_kd = kb * GATE_D_TILE
            gd_x = pl.cast(xg_buf[t1 : t1 + GATE_M_TILE, gd_kd : gd_kd + GATE_D_TILE], pl.BF16, mode="rint")
            gd_w = gate_weight[n0 : n0 + GATE_N_TILE, gd_kd : gd_kd + GATE_D_TILE]
            if gd_kd == 0:
                gate_logits_tile = pl.matmul(gd_x, gd_w, out_dtype=pl.FP32, b_trans=True)
            else:
                gate_logits_tile = pl.matmul_acc(gate_logits_tile, gd_x, gd_w, b_trans=True)
        # xg omitted inv_rms; logits = inv_rms * (xg @ gate_w.T).
        gate_logits_tile = pl.row_expand_mul(gate_logits_tile, inv_rms_buf[t1 : t1 + GATE_M_TILE, 0:1])
        # Sigmoid router score: 1 / (1 + exp(-logits)).
        gp_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_logits_tile)), 1.0))
        route_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_sigmoid
        gp_bias_row = pl.reshape(correction_bias[n0 : n0 + GATE_N_TILE], [1, GATE_N_TILE])
        gp_biased = pl.col_expand_add(gp_sigmoid, gp_bias_row)
        biased_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_biased

    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    for ts_idx in pl.spmd(active_route_tiles, name_hint="route_sort", allow_early_resolve=True):
        t1 = ts_idx * GATE_T_TILE
        # topk_idx_tile stays Tensor (created here, not a pl.full Tile) so
        # the batched pl.gather below accepts it — Tile-against-Tensor src
        # is rejected.
        topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
        # ptoas pto.tmrgsort requires src rows == 1; sort path iterates
        # row-by-row. sort32: [1,512] → [1,1024] (16 runs of 64). mrgsort
        # format1 4-way: 16 → 4 runs of 256. mrgsort format2 4-way: 4 runs
        # of 256 → 1 run of 1024. The GLM-5.3-Flash row is 288 experts, so
        # the pad grows to 512 and the merge tree gains a level over the
        # DeepSeek-V4 sibling (which stops at two mrgsort stages for 256).
        for sr_tt in pl.range(GATE_T_TILE):
            sr_row = biased_scores_buf[t1 + sr_tt : t1 + sr_tt + 1, :]
            sr_idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
            sr_sorted = pl.sort32(sr_row, sr_idx_init)
            sr_sorted = pl.mrgsort(sr_sorted, block_len=64)
            sr_sorted = pl.mrgsort(sr_sorted, block_len=256)
            sr_pairs = sr_sorted[:, 0:SORT_PAD]
            sr_i = pl.gather(sr_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i
        # Batched gather of the *unbiased* scores at the sorted ids; a
        # set_validshape + fillpad zeros the [TOPK, TOPK_PAD) tail so the
        # normalize sum below sees only real TOPK entries.
        local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
        local_scores[:, :] = route_scores_buf[t1 : t1 + GATE_T_TILE, :]
        gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
        gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
        topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
        # Copy to dodge the tensor_view-vs-ptr SSA conflict between the gather
        # and the scalar pl.read below (pypto #1493).
        topk_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
        topk_idx_read[:, :] = topk_idx_tile[:, :]
        topk_sum = pl.row_sum(topk_vals_pad)
        denom = pl.reshape(topk_sum, [GATE_T_TILE, 1])
        topk_normalized = pl.row_expand_div(topk_vals_pad, denom)
        normalized_weights = pl.mul(topk_normalized, ROUTE_SCALE)
        for wt_tt in pl.range(GATE_T_TILE):
            wt_out_t = t1 + wt_tt
            if wt_out_t < active_tokens:
                for wt_k in pl.range(TOPK):
                    wt_out_idx = pl.read(topk_idx_read, [wt_tt, wt_k])
                    pl.write(route_indices, [wt_out_t, wt_k], wt_out_idx)
                    wt_out_weight = pl.read(normalized_weights, [wt_tt, wt_k])
                    pl.write(route_weights, [wt_out_t, wt_k], wt_out_weight)

    # The @pl.inline parser requires inline call expressions to have a return
    # value; route_weights is convenient because it's already the write-back
    # target and reads as the same SSA name on the caller side.
    return route_weights


@pl.jit
def moe_gate_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    route_weights: pl.Tensor[[T_DYN, TOPK], pl.FP32],
    route_indices: pl.Tensor[[T_DYN, TOPK], pl.INT32],
    x_int8: pl.Tensor[[T_DYN, D], pl.INT8],
    x_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
):
    x.bind_dynamic(0, T_DYN)
    route_weights.bind_dynamic(0, T_DYN)
    route_indices.bind_dynamic(0, T_DYN)
    x_int8.bind_dynamic(0, T_DYN)
    x_scale.bind_dynamic(0, T_DYN)

    moe_gate(
        x,
        norm_weight,
        gate_weight,
        correction_bias,
        route_weights,
        route_indices,
        x_int8,
        x_scale,
        num_tokens,
    )
    return route_weights, route_indices, x_int8, x_scale


def golden_moe_gate(tensors):
    """Fill outputs in-place from the CPU router reference.

    route_weights/route_indices come from the sigmoid top-k over the biased
    scores; x_int8/x_scale are the per-token INT8 view of the normalized xg,
    exactly what the shared expert and the EP dispatch payload consume.
    """
    from models.glm5_3_flash.quantization import quantize_per_token_int8

    num_tokens = max(0, min(int(tensors.get("num_tokens", x_rows(tensors))), x_rows(tensors)))

    x = tensors["x"]
    x_norm = rms_norm(x, tensors["norm_weight"])
    weights, indices = gate(x_norm, tensors["gate_weight"], tensors["correction_bias"])
    x_int8, x_scale = quantize_per_token_int8(x_norm)

    tensors["route_weights"][:] = weights
    tensors["route_indices"][:] = indices
    tensors["x_int8"][:] = x_int8
    tensors["x_scale"][:] = x_scale
    if num_tokens < x_rows(tensors):
        tensors["route_weights"][num_tokens:] = 0
        tensors["route_indices"][num_tokens:] = 0
        tensors["x_scale"][num_tokens:] = 0


def build_tensor_specs(num_tokens=T_DYN_TEST):
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.randn(num_tokens, D, dtype=torch.bfloat16)

    def init_norm_weight():
        return torch.ones(D, dtype=torch.bfloat16)

    def init_gate_weight():
        return torch.randn(N_EXPERTS, D, dtype=torch.bfloat16) / (D ** 0.5)

    def init_correction_bias():
        return torch.randn(N_EXPERTS) * 0.1

    return [
        TensorSpec("x", [num_tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_weight", [D], torch.bfloat16, init_value=init_norm_weight),
        TensorSpec("gate_weight", [N_EXPERTS, D], torch.bfloat16, init_value=init_gate_weight),
        TensorSpec("correction_bias", [N_EXPERTS], torch.float32, init_value=init_correction_bias),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("route_weights", [num_tokens, TOPK], torch.float32),
        TensorSpec("route_indices", [num_tokens, TOPK], torch.int32),
        TensorSpec("x_int8", [num_tokens, D], torch.int8),
        TensorSpec("x_scale", [num_tokens, 1], torch.float32),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, ratio_reldiff, run, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T_DYN_TEST)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=moe_gate_test,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_moe_gate,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # x_int8 keeps its LSB rule: +/-1 on a bounded fraction of elements.
            "x_int8": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.001,
                                     valid_rows=gate_active_rows(args.num_tokens)),
            # FP32 router output: 3e-3 per point, 1% of points.
            "route_weights": ratio_reldiff(diff_thd=3e-3, pct_thd=0.01),
            "route_indices": topk_pair_compare("route_weights"),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "SCORE_PAD",
    "golden_gate",
    "moe_gate",
]
