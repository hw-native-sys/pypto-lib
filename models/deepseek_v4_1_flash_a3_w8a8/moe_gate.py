# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 sqrtsoftplus router: biased top-k selection and normalized weights."""

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN


# Dynamic shape variable.

# model config
D = M.hidden_size
N_EXPERTS = M.n_routed_experts
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor
TOPK_PAD = 8
SORT_PAD = 16

# tiling
T_TILE = 16
D_TILE = 128


@pl.jit.inline(auto_scope=False)
def moe_gate(
    x_normed: pl.Tensor,
    gate_weight: pl.Tensor,
    gate_bias: pl.Tensor,
    indices: pl.Tensor,
    weights: pl.Tensor,
    topk: pl.constexpr,
    topk_pad: pl.constexpr,
):
    """Score every expert with sqrtsoftplus and select the biased top-k per token."""
    t_dim = pl.tensor.dim(x_normed, 0)
    blocks = (t_dim + T_TILE - 1) // T_TILE
    for block in pl.spmd(blocks, name_hint="moe_gate"):
        t0 = block * T_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        logits = pl.create_tensor([N_EXPERTS, T_TILE], dtype=pl.FP32)
        for k0 in pl.pipeline(0, D, D_TILE, stage=2):
            x_tile = pl.slice(x_normed, [T_TILE, D_TILE], [t0, k0], valid_shape=[rows, D_TILE])
            x_tile = pl.set_validshape(pl.fillpad(x_tile, pad_value=pl.PadValue.zero), T_TILE, D_TILE)
            w_tile = pl.slice(gate_weight, [N_EXPERTS, D_TILE], [0, k0])
            logits = pl.matmul_acc(
                logits, w_tile, pl.cast(x_tile, pl.FP32), b_trans=True, init_cond=(k0 == 0)
            )
        positive = pl.maximum(logits, 0.0)
        tail = pl.log(pl.add(pl.exp(pl.neg(pl.abs(logits))), 1.0))
        softplus = pl.add(positive, tail)
        floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(logits), 10.0), 0.0), 1.0)
        floor_value = pl.mul(floor_mask, pl.exp(pl.minimum(logits, 0.0)))
        score = pl.sqrt(pl.maximum(softplus, floor_value))
        bias_col = pl.reshape(gate_bias, [N_EXPERTS, 1])
        biased = pl.row_expand_add(score, bias_col)
        index_row = pl.arange(0, [1, N_EXPERTS], dtype=pl.INT32)
        index_col = pl.cast(pl.reshape(index_row, [N_EXPERTS, 1]), target_type=pl.FP32)
        topk_idx_slot = pl.create_tensor([TOPK_PAD, T_TILE], dtype=pl.FP32)
        topk_val_slot = pl.create_tensor([TOPK_PAD, T_TILE], dtype=pl.FP32)
        work = biased
        # Static unroll: the per-slot row store below slices a tile at the loop
        # index, and a runtime slice start lowers to a dynamic-offset tile view
        # whose emitted valid-region mask is not static on the onboard backend.
        for slot in pl.unroll(topk):
            max_row = pl.col_max(work)
            zeros = pl.full([N_EXPERTS, T_TILE], dtype=pl.FP32, value=0.0)
            max_bcast = pl.col_expand_add(zeros, max_row)
            is_max = pl.cmp(work, max_bcast, cmp_type=0)
            reversed_indices = pl.add(pl.neg(index_col), N_EXPERTS)
            candidate_indices = pl.mul(is_max, reversed_indices)
            selected_index = pl.add(pl.neg(pl.col_max(candidate_indices)), N_EXPERTS)
            selected_bcast = pl.col_expand_add(zeros, selected_index)
            index_bcast = pl.row_expand_add(zeros, index_col)
            selected_mask = pl.cmp(index_bcast, selected_bcast, cmp_type=0)
            selected_value = pl.col_sum(pl.mul(score, selected_mask))
            # Store the FP32 index row: the int32 row-slice store lowers to a
            # dynamic subview whose emitted valid-region mask is not static on
            # the onboard backend; expert ids 0..383 are exact in FP32.
            topk_idx_slot[slot : slot + 1, :] = selected_index
            topk_val_slot[slot : slot + 1, :] = selected_value
            work = pl.add(work, pl.mul(selected_mask, -1.0e30))
        valid_values = pl.set_validshape(topk_val_slot, topk, T_TILE)
        padded_values = pl.fillpad(valid_values, pad_value=pl.PadValue.zero)
        total = pl.col_sum(padded_values)
        normalized = pl.col_expand_div(padded_values, total)
        normalized = pl.mul(normalized, ROUTE_SCALE)
        for token in pl.range(T_TILE):
            out_token = t0 + token
            if out_token < t_dim:
                for slot in pl.range(topk):
                    top_idx = pl.cast(pl.read(topk_idx_slot, [slot, token]), pl.INT32)
                    top_weight = pl.read(normalized, [slot, token])
                    pl.write(indices, [out_token, slot], top_idx)
                    pl.write(weights, [out_token, slot], top_weight)
    return weights


@pl.jit
def moe_gate_test(
    x_normed: pl.Tensor[[T_DYN, 5120], pl.BF16],
    gate_weight: pl.Tensor[[384, 5120], pl.FP32],
    gate_bias: pl.Tensor[[384], pl.FP32],
    indices: pl.Out[pl.Tensor[[T_DYN, 6], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T_DYN, 6], pl.FP32]],
):
    x_normed.bind_dynamic(0, T_DYN)
    indices.bind_dynamic(0, T_DYN)
    weights.bind_dynamic(0, T_DYN)
    moe_gate(x_normed, gate_weight, gate_bias, indices, weights, topk=6, topk_pad=8)
    return indices, weights


def golden_moe_gate(tensors):
    """Torch reference for sqrtsoftplus scoring, biased top-k, and normalization."""
    x = tensors["x_normed"].float()
    logits = torch.matmul(x, tensors["gate_weight"].float().t())
    softplus = torch.clamp(logits, min=0) + torch.log1p(torch.exp(-logits.abs()))
    score = torch.sqrt(softplus)
    biased = score + tensors["gate_bias"].float().reshape(1, -1)
    top_ids = torch.topk(biased, 6, dim=-1, largest=True, sorted=True).indices
    top_values = torch.gather(score, -1, top_ids)
    normalized = top_values / top_values.sum(dim=-1, keepdim=True)
    tensors["indices"][:] = top_ids.to(torch.int32)
    tensors["weights"][:] = normalized * 1.5


def build_tensor_specs(tokens: int = 16):
    from golden import TensorSpec

    return [
        TensorSpec("x_normed", [tokens, 5120], torch.bfloat16, init_value=torch.randn),
        TensorSpec("gate_weight", [384, 5120], torch.float32, init_value=lambda: torch.randn(384, 5120) * 0.01),
        TensorSpec("gate_bias", [384], torch.float32, init_value=lambda: torch.randn(384) * 0.01),
        TensorSpec("indices", [tokens, 6], torch.int32),
        TensorSpec("weights", [tokens, 6], torch.float32),
    ]


def build_tie_specs(kind, tokens):
    """Zero, equal nonzero and partially tied expert scores."""
    from golden import TensorSpec

    x = torch.zeros(tokens, D, dtype=torch.bfloat16)
    gate = torch.zeros(N_EXPERTS, D)
    bias = torch.zeros(N_EXPERTS)
    if kind == "equal":
        x[:, 0] = 1
        gate[:, 0] = 0.75
    elif kind == "partial":
        x[:, 0] = 1
        gate[:, 0] = -2
        gate[0:2, 0] = 2
        gate[2:9, 0] = 0.75
        bias[9] = 2.0
        bias[10] = 1.5
    elif kind != "zero":
        raise ValueError(kind)
    return [
        TensorSpec("x_normed", [tokens, D], torch.bfloat16, init_value=x),
        TensorSpec("gate_weight", [N_EXPERTS, D], torch.float32, init_value=gate),
        TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=bias),
        TensorSpec("indices", [tokens, TOPK], torch.int32),
        TensorSpec("weights", [tokens, TOPK], torch.float32),
    ]


def _reference_scores(inputs):
    logits = inputs["x_normed"].float() @ inputs["gate_weight"].float().t()
    scores = torch.nn.functional.softplus(logits).sqrt()
    return scores, scores + inputs["gate_bias"].float()


def compare_moe_indices(actual, expected, *, inputs, rtol, atol, **kwargs):
    """Check legal distinct experts and their ordered biased Top-K scores."""
    from golden import ratio_allclose

    ids = actual.long()
    if bool(((ids < 0) | (ids >= N_EXPERTS)).any()):
        return False, f"expert IDs outside [0,{N_EXPERTS}): {ids.tolist()}"
    ordered = ids.sort(dim=-1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        return False, f"duplicate expert IDs: {ids.tolist()}"
    _, biased = _reference_scores(inputs)
    selected = biased.gather(-1, ids)
    reference = biased.topk(TOPK, dim=-1).values
    compare = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=0)
    return compare(selected, reference, inputs=inputs, rtol=rtol, atol=atol, **kwargs)


def compare_moe_weights(actual, expected, *, actual_outputs, inputs, rtol, atol, **kwargs):
    """Check normalized unbiased weights for the actual selected legal experts."""
    from golden import ratio_allclose

    ids = actual_outputs["indices"].long()
    if bool(((ids < 0) | (ids >= N_EXPERTS)).any()):
        return False, "cannot normalize invalid expert IDs"
    scores, _ = _reference_scores(inputs)
    selected = scores.gather(-1, ids)
    reference = selected / selected.sum(-1, keepdim=True) * ROUTE_SCALE
    compare = ratio_allclose(atol=1e-3, rtol=1e-2, max_error_ratio=1e-2)
    return compare(actual, reference, actual_outputs=actual_outputs, inputs=inputs, rtol=rtol, atol=atol, **kwargs)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    result = run(
        fn=moe_gate_test,
        specs=build_tensor_specs(args.tokens),
        golden_fn=golden_moe_gate,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "indices": compare_moe_indices,
            "weights": compare_moe_weights,
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

    for kind, tokens in (("zero", 1), ("equal", 7), ("partial", 17)):
        print(f"[CASE] MoE tied routes {kind} T={tokens}", flush=True)
        result = run(
            fn=moe_gate_test,
            specs=build_tie_specs(kind, tokens),
            golden_fn=golden_moe_gate,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={"indices": compare_moe_indices, "weights": compare_moe_weights},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
