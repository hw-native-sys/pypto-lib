# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash sigmoid routing and activation quantization."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import dataclass

import pypto.language as pl
import torch

from models.glm5_3_flash.config import FLASH, N_EXPERTS, TOPK, D, Glm53FlashConfig
from models.glm5_3_flash.elementwise import FP32_NEG_INF, accurate_divide, accurate_sigmoid
from models.glm5_3_flash.linear import bf16_linear
from models.glm5_3_flash.rmsnorm import golden_rmsnorm_quant, rms_norm, rmsnorm_quant
from models.glm5_3_flash.utils import _check, _report

T_DYN = pl.dynamic('GLM53_T_DYN')

ROUTE_SCALE = FLASH.routed_scaling_factor
SCORE_PAD = 512  # 288 experts padded to the next sort32-friendly width
GATE_T_TILE = 8  # router sort / write-back token tile
GATE_M_TILE = 16  # cube M-tile: matmul rows must be a multiple of 16 (fractal)
GATE_N_TILE = 16  # expert columns per gate spmd block
TOPK_PAD = 8  # TOPK padded to 32B-aligned width
SORT_PAD = TOPK_PAD * 2  # (val, idx) interleaved slice width
T_DYN_TEST = 64


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
    # Routing and quantization consume the same BF16 RMSNorm result.
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > token_rows:
        active_tokens = pl.cast(token_rows, pl.INDEX)
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE

    normalized_input = pl.create_tensor([token_rows, D], dtype=pl.BF16)
    if active_tokens > 0:
        rmsnorm_quant(x, norm_weight, normalized_input, x_int8, x_scale, num_tokens)

    # Pre-route setup: zero the inactive-token outputs and NEG_INF the biased pad
    # columns so the sort ranks pad experts last. Route write-backs are guarded to
    # active tokens, so the inactive-zero can run here rather than post-route.
    biased_scores_buf = pl.create_tensor([padded_rows, SCORE_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
        for zt in pl.range(token_rows):
            if zt >= active_tokens:
                zero = pl.full([1, D], dtype=pl.FP16, value=0.0)
                x_int8[zt : zt + 1, :] = pl.cast(zero, pl.INT8, mode="trunc")
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
                biased_scores_buf[pad_t0:pad_t0 + GATE_M_TILE, N_EXPERTS:SCORE_PAD] = pl.full([GATE_M_TILE, SCORE_PAD - N_EXPERTS], dtype=pl.FP32, value=FP32_NEG_INF)

    route_scores_buf = pl.create_tensor([padded_rows, SCORE_PAD], dtype=pl.FP32)
    if active_tokens > 0:
        # BF16 normalized routing inputs produce FP32 logits and sigmoid scores.
        for gb_idx in pl.spmd(active_gate_tiles * (N_EXPERTS // GATE_N_TILE), name_hint="gate", allow_early_resolve=True):
            tg = gb_idx // (N_EXPERTS // GATE_N_TILE)
            nb = gb_idx % (N_EXPERTS // GATE_N_TILE)
            t1 = tg * GATE_M_TILE
            n0 = nb * GATE_N_TILE
            rows = pl.min(GATE_M_TILE, token_rows - t1)
            logits = bf16_linear(
                normalized_input, gate_weight, t1, n0, rows, pl.cast(GATE_N_TILE, pl.INDEX), GATE_M_TILE,
                GATE_N_TILE, D, True,
            )
            gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
            gate_logits_tile[:, :] = logits
            gp_sigmoid = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
            gp_sigmoid[:, :] = accurate_sigmoid(gate_logits_tile, GATE_M_TILE, GATE_N_TILE)
            route_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_sigmoid
            gp_bias_row = pl.reshape(correction_bias[n0 : n0 + GATE_N_TILE], [1, GATE_N_TILE])
            gp_biased = pl.col_expand_add(gp_sigmoid, gp_bias_row)
            biased_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_biased

    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    if active_tokens > 0:
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
            # Gather unbiased scores; explicitly zero any padded top-k lanes
            # before their normalization sum.
            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + GATE_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
            # Copy to dodge the tensor_view-vs-ptr SSA conflict between the gather
            # and the scalar pl.read below (pypto #1493).
            topk_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            topk_idx_read[:, :] = topk_idx_tile[:, :]
            topk_values = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.FP32)
            topk_values[:, :] = gather_all
            if TOPK < TOPK_PAD:
                topk_values[:, TOPK:TOPK_PAD] = pl.full([GATE_T_TILE, TOPK_PAD - TOPK], dtype=pl.FP32, value=0.0)
            split = pl.mul(topk_values, 4097.0)
            score_high = pl.sub(split, pl.sub(split, topk_values))
            score_low = pl.sub(topk_values, score_high)
            topk_sum = pl.add(pl.row_sum(score_high), pl.row_sum(score_low))
            denom = pl.reshape(topk_sum, [GATE_T_TILE, 1])
            denominator = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.FP32)
            denominator_value_full = pl.full([GATE_T_TILE, TOPK_PAD], dtype=pl.FP32, value=1.0)
            denominator_value_add = pl.add(denom, 1e-20)
            denominator[:, :] = pl.row_expand_mul(denominator_value_full, denominator_value_add)
            normalized = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.FP32)
            normalized[:, :] = accurate_divide(topk_values, denominator)
            normalized_weights = pl.mul(normalized, ROUTE_SCALE)
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
    route_weights: pl.Out[pl.Tensor[[T_DYN, TOPK], pl.FP32]],
    route_indices: pl.Out[pl.Tensor[[T_DYN, TOPK], pl.INT32]],
    x_int8: pl.Out[pl.Tensor[[T_DYN, D], pl.INT8]],
    x_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    route_weights.bind_dynamic(0, T_DYN)
    route_indices.bind_dynamic(0, T_DYN)
    x_int8.bind_dynamic(0, T_DYN)
    x_scale.bind_dynamic(0, T_DYN)

    moe_gate(x, norm_weight, gate_weight, correction_bias, route_weights, route_indices, x_int8, x_scale, num_tokens)
    return route_weights, route_indices, x_int8, x_scale


@dataclass
class RouterTensors:
    weight: torch.Tensor  # [n_experts, D]
    bias: torch.Tensor  # [n_experts] FP32


def router_forward(
    flat: torch.Tensor, weights: RouterTensors, model: Glm53FlashConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """noaux-tc sigmoid router (port of ``Glm5NextTextTopkRouter.forward``).

    Returns ``(topk_weights [T, k], topk_indices [T, k])``; weights are the
    bias-free sigmoid scores, top-k normalized and scaled by
    ``routed_scaling_factor``.
    """
    num_experts = model.n_routed_experts
    router_logits = flat @ weights.weight.t()
    scores = router_logits.sigmoid()
    scores_for_choice = scores + weights.bias
    group_scores = scores_for_choice.view(-1, model.n_group, num_experts // model.n_group).topk(2, dim=-1)[0].sum(
        dim=-1,
    )
    group_idx = torch.topk(group_scores, k=model.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, model.n_group, num_experts // model.n_group).reshape(
        -1, num_experts,
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=model.num_experts_per_tok, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if model.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * model.routed_scaling_factor
    return topk_weights, topk_indices


def x_rows(tensors):
    return int(tensors["x"].shape[0])


def gate_active_rows(num_tokens):
    """Rows the kernel must produce: the clamped active token count.

    The 8-row quant tiles leave rows past ``num_tokens`` untouched, so the
    comparison must stop at the active count rather than the padded tile width.
    """
    return max(0, int(num_tokens))


def make_correction_bias(num_ranks: int, num_experts: int, topk: int) -> torch.Tensor:
    """Build a replicated fixture bias with fixed winners spread across EP ranks."""
    if num_ranks < 1 or num_experts < 1 or num_experts % num_ranks:
        raise ValueError("num_experts must be positive and divisible by a positive num_ranks")
    if not 1 <= topk <= num_experts:
        raise ValueError("topk must be between 1 and num_experts")
    slots = torch.arange(topk)
    winners = (slots % num_ranks) * (num_experts // num_ranks) + slots // num_ranks
    bias = torch.full((num_experts,), -5.0, dtype=torch.float32)
    bias[winners] = 5.0
    return bias.unsqueeze(0).expand(num_ranks, -1).contiguous()


def gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid ``noaux_tc`` routing; the correction bias affects selection only.

    ``n_group`` and ``topk_group`` are both 1 for GLM-5.3-Flash, so the grouped
    masking of ``DeepseekV3TopkRouter`` degenerates to a plain top-k.
    """
    logits = (x.double() @ weight.double().t()).float()
    scores = (1.0 / (1.0 + torch.exp(-logits.double()).float()).double()).float()
    biased = scores + correction_bias.float()
    # Exact corrected-score ties keep ascending expert ids.
    indices = torch.argsort(-biased, dim=-1, stable=True)[..., : FLASH.num_experts_per_tok]
    weights = scores.gather(-1, indices)
    if FLASH.norm_topk_prob:
        denominator = weights.double().sum(dim=-1, keepdim=True).float() + 1e-20
        weights = (weights.double() / denominator.double()).float()
    return weights * FLASH.routed_scaling_factor, indices.to(torch.int32)


def golden_gate(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return gate(rms_norm(x, norm_weight), gate_weight, correction_bias)


def golden_moe_gate(tensors):
    """Route and quantize the BF16 RMSNorm result."""
    num_tokens = max(0, min(int(tensors.get("num_tokens", x_rows(tensors))), x_rows(tensors)))

    x = tensors["x"]
    x_norm, x_int8, x_scale = golden_rmsnorm_quant(x, tensors["norm_weight"])
    weights, indices = gate(x_norm, tensors["gate_weight"], tensors["correction_bias"])

    tensors["route_weights"][:] = weights
    tensors["route_indices"][:] = indices
    tensors["x_int8"][:] = x_int8
    tensors["x_scale"][:] = x_scale
    if num_tokens < x_rows(tensors):
        tensors["route_weights"][num_tokens:] = 0
        tensors["route_indices"][num_tokens:] = 0
        tensors["x_scale"][num_tokens:] = 0
        tensors["x_int8"][num_tokens:] = 0


def compare_route_indices(actual, expected, **kwargs):
    """Require the same selected experts, independently of biased-score ordering."""
    if not torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values):
        return False, "selected expert sets differ from CPU golden"
    return True, ""


def compare_route_weights(actual, expected, *, actual_outputs, expected_outputs, **kwargs):
    """Align weights by expert ID before applying the existing error budget."""
    from golden import ratio_reldiff

    actual_order = actual_outputs["route_indices"].argsort(dim=-1)
    expected_order = expected_outputs["route_indices"].argsort(dim=-1)
    actual_aligned = actual.gather(-1, actual_order)
    expected_aligned = expected.gather(-1, expected_order)
    compare = ratio_reldiff(diff_thd=3e-3, pct_thd=0.01)
    return compare(
        actual_aligned, expected_aligned, actual_outputs=actual_outputs, expected_outputs=expected_outputs,
        **kwargs,
    )


def build_tensor_specs(num_tokens=T_DYN_TEST):
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(533)

    def init_x():
        return torch.randn(num_tokens, D, dtype=torch.bfloat16, generator=generator)

    def init_norm_weight():
        return torch.ones(D, dtype=torch.bfloat16)

    def init_gate_weight():
        return torch.randn(N_EXPERTS, D, dtype=torch.bfloat16, generator=generator) / (D**0.5)

    def init_correction_bias():
        return torch.randn(N_EXPERTS, generator=generator) * 0.1

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


def run_moe_gate_golden(golden_gate) -> None:
    """Exercise the sigmoid ``noaux_tc`` router on a reduced expert count."""
    torch.manual_seed(13)
    tokens, hidden, experts = 8, 64, 32
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    weight = torch.randn(experts, hidden)
    correction_bias = torch.randn(experts)

    weights, indices = golden_gate(x, weight, correction_bias)
    topk = FLASH.num_experts_per_tok
    _check(weights.shape == (tokens, topk), f"router weight shape {tuple(weights.shape)}")
    _check(indices.shape == (tokens, topk), f"router index shape {tuple(indices.shape)}")
    _check(indices.dtype is torch.int32, f"router index dtype {indices.dtype}")
    _check(bool((indices >= 0).all() and (indices < experts).all()), "router index out of range")
    for row in indices.tolist():
        _check(len(set(row)) == topk, f"router selected a duplicate expert: {row}")
    if FLASH.norm_topk_prob:
        total = weights.sum(dim=-1) / FLASH.routed_scaling_factor
        _check((total - 1.0).abs().max() < 1e-4, f"router weights are not normalised: {total}")

    _report("moe_gate")


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T_DYN_TEST)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=moe_gate_test,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_moe_gate,
        config=dict(dump_passes=args.dump_passes, backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device, enable_chip_swimlane=args.enable_chip_swimlane),
        rtol=0.001,
        atol=0.001,
        compare_fn={"x_int8": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.001, valid_rows=gate_active_rows(args.num_tokens)), "route_weights": compare_route_weights, "route_indices": compare_route_indices},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
