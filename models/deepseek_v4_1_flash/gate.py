# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 MoE router: normalized BF16 input, quantization, projection, and top-k weights."""

from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import torch
import torch.nn.functional as F

from config import FLASH as M

from config import MOE_TOKENS, EP_SIZE, N_LOCAL_EXPERTS


# model config
T = MOE_TOKENS
D = M.hidden_size
BIAS_DYN = pl.dynamic("GATE_BIAS_DYN")
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor

# tiling
GATE_T_TILE = 4
assert T % GATE_T_TILE == 0
GATE_M_TILE = 16
GATE_N_TILE = 16
T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
QUANT_TILE = 256
QUANT_TASK_TILE = QUANT_TILE
assert D % QUANT_TASK_TILE == 0
MX_GROUP = 32
MX_SCALE_GROUPS = D // MX_GROUP
QUANT_WORKER_TILE = 24 if T > 80 else 64
ROUTE_WORKER_TILE = 24 if T > 80 else 64
TOPK_PAD = 8
SORT_PAD = TOPK_PAD * 2


@pl.jit.extern(
    core_type="mixed",
    aic_source=Path(__file__).parent / "kernels" / "gate_scores_vf.cpp",
    aiv_source=Path(__file__).parent / "kernels" / "gate_scores_vf.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _gate_scores_vf(
    x: pl.Tensor,
    gate_w: pl.Tensor,
    gate_bias: pl.Tensor,
    route_scores: pl.Out[pl.Tensor],
    biased_scores: pl.InOut[pl.Tensor],
    tasks: pl.Scalar[pl.INDEX],
    bias_stride: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Project router inputs and compute sqrt-softplus scores with selection bias."""
    return route_scores, biased_scores


@pl.jit.inline
def gather_routing_bias(
    source: pl.Tensor,
    row_ids: pl.Tensor,
    output: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID],
    n_experts: pl.constexpr,
):
    """Gather token routing biases or copy one broadcast expert vector."""
    source_rows = pl.tensor.dim(source, 0) // n_experts
    output_rows = pl.tensor.dim(output, 0) // n_experts
    source_matrix = pl.reshape(source, [source_rows, n_experts])
    output_matrix = pl.reshape(output, [output_rows, n_experts])
    for row in pl.spmd(output_rows, name_hint="gate_bias_gather", deps=[ready]):
        source_row = pl.cast(0, pl.INT32)
        if source_rows > 1:
            source_row = pl.read(row_ids, [row])
        for column in pl.range(0, n_experts, 32):
            valid_columns = pl.min(32, n_experts - column)
            if source_row >= 0 and source_row < source_rows:
                value = pl.load(source_matrix, [source_row, column], [1, 32], valid_shape=[1, valid_columns])
                pl.store(value, [row, column], output_matrix)
            else:
                zero = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
                pl.store(pl.set_validshape(zero, 1, valid_columns), [row, column], output_matrix)
    return output


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "gate_route_sort_vf.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _gate_route_sort_vf(
    biased_scores: pl.Tensor,
    route_scores: pl.Tensor,
    indices: pl.InOut[pl.Tensor],
    weights: pl.InOut[pl.Tensor],
    active_tokens: pl.Scalar[pl.INDEX],
    route_scale: pl.Scalar[pl.FP32],
    topk: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Sort router scores, normalize selected weights, and store active rows."""
    return indices, weights


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "gate_quant_vf.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _gate_quant_vf(
    x: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
) -> tuple[pl.Tensor, pl.Tensor]:
    """Quantize normalized input into group-32 FP8 and MX_A_ZZ scale bytes."""
    return payload, scales


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "gate_pre_route_vf.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _gate_pre_route_vf(
    indices: pl.Out[pl.Tensor],
    weights: pl.Out[pl.Tensor],
    biased_scores: pl.Out[pl.Tensor],
    active_tokens: pl.Scalar[pl.INDEX],
    experts: pl.Scalar[pl.INDEX],
    topk: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    """Zero inactive routes and initialize padded expert scores."""
    return indices, weights, biased_scores


def golden_gate_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sqrt-softplus routing, with bias affecting selection only."""
    scores = F.softplus(F.linear(x.float(), weight.to(torch.bfloat16).float()) / M.gate_temperature).sqrt()
    bias = correction_bias.float().reshape(-1, weight.shape[0])[:x.shape[0]]
    indices = (scores + bias).topk(M.num_experts_per_tok, dim=-1).indices
    weights = scores.gather(-1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * M.routed_scaling_factor, indices.to(torch.int32)


def prepare_routing_bias(text_bias, vision_bias, image_mask, capacity):
    """Flatten per-layer token biases in encoder input order, padding with text bias."""
    if text_bias.ndim not in (1, 2) or text_bias.shape != vision_bias.shape:
        raise ValueError("text and vision biases must have matching [experts] or [layers, experts] shapes")
    if text_bias.dtype != torch.float32 or vision_bias.dtype != torch.float32:
        raise ValueError("routing biases must have FP32 dtype")
    if text_bias.device.type != "cpu" or vision_bias.device.type != "cpu":
        raise ValueError("routing bias preparation requires CPU weights")
    mask = torch.as_tensor(image_mask, device="cpu")
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("image_mask must be a one-dimensional boolean sequence")
    if type(capacity) is not int or capacity < max(1, mask.numel()):
        raise ValueError("capacity must be a positive integer covering the image mask")
    padded_mask = torch.zeros(capacity, dtype=torch.bool)
    padded_mask[:mask.numel()] = mask
    text = text_bias.reshape(-1, 1, text_bias.shape[-1])
    vision = vision_bias.reshape_as(text)
    return torch.where(padded_mask[None, :, None], vision, text).flatten().contiguous()


@pl.jit.inline
def gate_from_bf16(
    xg_buf: pl.Tensor[[T_PAD, D], pl.BF16],
    gate_w: pl.Tensor,
    gate_bias: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    x_norm_mx: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_norm_scale: pl.Tensor[[1, T_PAD * MX_SCALE_GROUPS], pl.FP8E8M0],
    indices: pl.Tensor,
    weights: pl.Tensor,
    n_experts: pl.constexpr = N_EXPERTS,
    topk: pl.constexpr = TOPK,
):
    """Route normalized BF16 rows with caller-zeroed inactive and padded rows."""
    if n_experts <= 32:
        score_pad = 32
    elif n_experts <= 256:
        score_pad = 256
    else:
        score_pad = 384
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    task_tokens = pl.max(active_tokens, pl.cast(1, pl.INDEX))
    task_gate_tiles = (task_tokens + GATE_M_TILE - 1) // GATE_M_TILE

    with pl.spmd(pl.min((T_PAD // GATE_M_TILE) * (D // QUANT_TASK_TILE), QUANT_WORKER_TILE), name_hint="x_norm_mx_quant"):
        x_norm_mx, x_norm_scale = _gate_quant_vf(xg_buf, x_norm_mx, x_norm_scale)

    biased_scores_buf = pl.create_tensor([T_PAD, score_pad], dtype=pl.FP32)
    with pl.spmd(1, name_hint="gate_pre_route"):
        indices, weights, biased_scores_buf = _gate_pre_route_vf(
            indices, weights, biased_scores_buf, active_tokens, n_experts, topk,
        )

    route_scores_buf = pl.create_tensor([T_PAD, score_pad], dtype=pl.FP32)
    # Reserve both engines before any local Cube-to-Vector transfer.
    # Bound synchronous cohorts to the target A5 core count.
    gate_tasks = task_gate_tiles * ((n_experts + GATE_N_TILE - 1) // GATE_N_TILE)
    bias_stride = pl.cast(0, pl.INDEX)
    if pl.tensor.dim(gate_bias, 0) > n_experts:
        bias_stride = pl.cast(n_experts, pl.INDEX)
    gate_worker_limit = pl.min(16, pl.system.available_cluster_count())
    gate_workers = pl.max(1, pl.min(gate_tasks, gate_worker_limit))
    with pl.spmd(gate_workers, name_hint="gate", sync_start=True):
        route_scores_buf, biased_scores_buf = _gate_scores_vf(
            xg_buf, gate_w, gate_bias, route_scores_buf, biased_scores_buf, gate_tasks, bias_stride,
        )

    # V4.1 uses score-only top-k: the V4 hash-route loop is dropped so that PTOAS
    # does not emit an invalid function for a route_hash path that does no work.
    # The native route publisher stores six columns; odd Top-K uses aligned DSL tasks.
    if topk == 6:
        task_route_tiles = (task_tokens + GATE_T_TILE - 1) // GATE_T_TILE
        with pl.spmd(pl.max(pl.min(task_route_tiles, ROUTE_WORKER_TILE), 1), name_hint="route_sort"):
            indices, weights = _gate_route_sort_vf(
                biased_scores_buf, route_scores_buf, indices, weights, active_tokens, ROUTE_SCALE, topk,
            )
    else:
        # Scalar route stores own complete 64-byte cache lines.
        if topk % 2 != 0:
            route_rows = 16
        elif topk % 4 != 0:
            route_rows = 8
        else:
            route_rows = GATE_T_TILE
        task_route_tiles = (task_tokens + route_rows - 1) // route_rows
        for ts_idx in pl.spmd(task_route_tiles, name_hint="route_sort"):
            t1 = ts_idx * route_rows
            # ptoas pto.tmrgsort requires a single source row.
            topk_idx_tile = pl.create_tensor([route_rows, TOPK_PAD], dtype=pl.INT32)
            sr_idx_init = pl.create_tensor([1, score_pad], dtype=pl.UINT32)
            sr_index_range = pl.arange(0, [1, score_pad], dtype=pl.UINT32)
            sr_idx_init[:, :] = sr_index_range
            for sr_tt in pl.range(route_rows):
                sr_t = t1 + sr_tt
                sr_row = pl.slice(biased_scores_buf, [1, score_pad], [sr_t, 0])
                sorted_32 = pl.sort32(sr_row, sr_idx_init)
                if n_experts <= 32:
                    sorted_pairs = sorted_32
                elif n_experts <= 256:
                    sorted_64 = pl.mrgsort(sorted_32, block_len=64)
                    sorted_pairs = pl.mrgsort(sorted_64[:, 0:256], sorted_64[:, 256:512])
                else:
                    sorted_64 = pl.mrgsort(sorted_32, block_len=64)
                    first = sorted_64[:, 0:256]
                    second = sorted_64[:, 256:512]
                    third = sorted_64[:, 512:768]
                    sorted_pairs = pl.mrgsort(first, second, third)
                selected_pairs = sorted_pairs[:, 0:SORT_PAD]
                sr_i = pl.gather(selected_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i

            local_scores = pl.create_tensor([route_rows, score_pad], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + route_rows, :]
            gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
            gather_valid = pl.set_validshape(gather_all, route_rows, topk)
            topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            topk_idx_read = pl.create_tensor([route_rows, TOPK_PAD], dtype=pl.INT32)
            topk_idx_read[:, :] = topk_idx_tile[:, :]
            topk_sum = pl.row_sum(topk_vals_pad)
            denom = pl.reshape(topk_sum, [route_rows, 1])
            topk_normalized = pl.row_expand_div(topk_vals_pad, denom)
            normalized_weights = pl.mul(topk_normalized, ROUTE_SCALE)
            for wt_tt in pl.range(route_rows):
                wt_out_t = t1 + wt_tt
                if wt_out_t < active_tokens:
                    for wt_k in pl.range(topk):
                        wt_out_idx = pl.read(topk_idx_read, [wt_tt, wt_k])
                        pl.write(indices, [wt_out_t, wt_k], wt_out_idx)
                        wt_out_weight = pl.read(normalized_weights, [wt_tt, wt_k])
                        pl.write(weights, [wt_out_t, wt_k], wt_out_weight)

    return weights


@pl.jit
def l2_gate(
    x_normed: pl.Tensor[[T_PAD, D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    gate_bias: pl.Tensor[[BIAS_DYN], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    x_norm_mx: pl.Out[pl.Tensor[[T_PAD, D], pl.FP8E4M3FN]],
    x_norm_scale: pl.Out[pl.Tensor[[1, T_PAD * MX_SCALE_GROUPS], pl.FP8E8M0]],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
):
    gate_bias.bind_dynamic(0, BIAS_DYN)
    gate_from_bf16(
        x_normed, gate_w, gate_bias,
        num_tokens,
        x_norm_mx, x_norm_scale, indices, weights, n_experts=N_EXPERTS, topk=TOPK,
    )
    return x_norm_mx, x_norm_scale, indices, weights


def _golden_gate_scores(tensors):
    """Recompute the host router scores from the immutable gate inputs."""
    import torch

    x_norm = tensors["x_normed"].cpu().to(torch.bfloat16).float()[:T].view(T, D)

    gate_w = tensors["gate_w"].cpu().to(torch.bfloat16).float()
    gate_bias = tensors["gate_bias"].cpu().float()
    logits = x_norm @ gate_w.T
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.reshape(-1, N_EXPERTS)[:T]
    return x_norm, scores, biased


def _golden_gate_core(tensors, x_norm, scores, biased):
    import torch

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    from quant import quantize_mxfp8_upward_e8m0, pack_mx_a_scale

    def host_mxfp8_activation(value):
        # The released activation quantizer rounds E8M0 scales upward.
        quantized, codes = quantize_mxfp8_upward_e8m0(value)
        packed = pack_mx_a_scale(codes)
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is not None:
            packed = packed.contiguous().view(e8m0)
        return quantized, packed

    x_norm[num_tokens:] = 0
    x_norm_padded = torch.zeros(T_PAD, D, dtype=torch.float32)
    x_norm_padded[:T] = x_norm
    x_norm_mx, x_norm_scale = host_mxfp8_activation(x_norm_padded)

    indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]

    topk_vals = torch.gather(scores, dim=-1, index=indices.long())
    denom = topk_vals.sum(dim=-1, keepdim=True)
    weights = (topk_vals / denom) * ROUTE_SCALE
    if num_tokens < T:
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0

    tensors["x_norm_mx"][:] = x_norm_mx
    tensors["x_norm_scale"][:] = x_norm_scale.reshape(1, -1)
    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights.to(torch.float32)


def golden_gate_normalized_core(tensors):
    """Reference routing from normalized BF16 input with a zero inactive tail."""
    x_norm, scores, biased = _golden_gate_scores(tensors)
    _golden_gate_core(tensors, x_norm, scores, biased)


def build_tensor_specs(num_tokens=T, *, token_bias=False):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_normed():
        from rmsnorm import golden_rms_norm_tiled

        value = golden_rms_norm_tiled(torch.randn(T_PAD, D).to(torch.bfloat16), torch.ones(D))
        value[max(0, min(T, num_tokens)):] = 0
        return value

    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5

    def init_gate_bias():
        if not token_bias:
            return torch.randn(N_EXPERTS) * 0.1
        bias = torch.full((T_PAD, N_EXPERTS), -10.0)
        for row in range(T_PAD):
            selected = (torch.arange(TOPK) + row * TOPK) % N_EXPERTS
            bias[row, selected] = 10.0
        return bias.flatten()

    # PyTorch CPU has incomplete float8 support for zeros/fill; initialise through
    # a uint8 carrier and view it as FP8 so the harness does not fail before NPU
    # compilation.
    def init_fp8(shape):
        return torch.zeros(shape, dtype=torch.uint8).view(torch.float8_e4m3fn)

    def init_e8m0(shape):
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if e8m0 is None:
            return torch.zeros(shape, dtype=torch.uint8)
        return torch.zeros(shape, dtype=torch.uint8).view(e8m0)

    return [
        TensorSpec("x_normed", [T_PAD, D], torch.bfloat16, init_value=init_x_normed),
        TensorSpec("gate_w", [N_EXPERTS, D], torch.bfloat16, init_value=init_gate_w),
        TensorSpec("gate_bias", [T_PAD * N_EXPERTS if token_bias else N_EXPERTS], torch.float32, init_value=init_gate_bias),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("x_norm_mx", [T_PAD, D], torch.float8_e4m3fn, init_value=lambda: init_fp8((T_PAD, D))),
        TensorSpec("x_norm_scale", [1, T_PAD * MX_SCALE_GROUPS], getattr(torch, "float8_e8m0fnu", torch.uint8), init_value=lambda: init_e8m0((1, T_PAD * MX_SCALE_GROUPS))),
        TensorSpec("indices", [T, TOPK], torch.int32),
        TensorSpec("weights", [T, TOPK], torch.float32),
    ]


def gate_indices_compare(
    num_tokens,
    *,
    score_atol=1e-4,
    score_rtol=2e-5,
    max_show=10,
):
    """Validate router ids against bounded score routes.

    ``topk_pair_compare`` cannot stand in for this: it adjudicates an index
    swap by checking whether the paired *weights* field stays locally
    monotonic, but selection is driven by the biased score
    (``scores + gate_bias``), not by the unbiased ``weights`` the device
    reports. Only the host-recomputed biased score tells us whether a swapped
    pair is a genuine tie at the top-k cutoff.
    """
    import torch

    active_tokens = max(0, min(T, int(num_tokens)))

    def cmp(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, rtol, atol
        actual = actual.cpu()
        expected = expected.cpu()
        if actual.shape != expected.shape:
            return False, (
                f"    index shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )

        inactive = actual[active_tokens:]
        inactive_nonzero = int(inactive.count_nonzero().item())
        if inactive_nonzero:
            return False, (
                f"    inactive index tail contains {inactive_nonzero} nonzero values"
            )

        actual = actual[:active_tokens].to(torch.int64)
        expected = expected[:active_tokens].to(torch.int64)
        if actual.numel() == 0:
            return True, ""

        invalid = (actual < 0) | (actual >= N_EXPERTS)
        if invalid.any().item():
            bad = invalid.nonzero(as_tuple=False)
            lines = [
                f"    score route contains {bad.shape[0]} out-of-range id(s); "
                f"valid range is [0, {N_EXPERTS})"
            ]
            for row, pos in bad[:max_show].tolist():
                lines.append(f"      [{row},{pos}] actual={int(actual[row, pos])}")
            return False, "\n".join(lines)

        sorted_ids = torch.sort(actual, dim=-1).values
        duplicate = sorted_ids[:, 1:] == sorted_ids[:, :-1]
        if duplicate.any().item():
            rows = duplicate.any(dim=-1).nonzero(as_tuple=False).flatten()
            lines = [f"    score route contains duplicate ids in {rows.numel()} row(s)"]
            for row in rows[:max_show].tolist():
                lines.append(f"      row {row}: ids={actual[row].tolist()}")
            return False, "\n".join(lines)

        _, scores, biased = _golden_gate_scores(inputs)
        scores = scores[:active_tokens]
        biased = biased[:active_tokens]
        if not torch.isfinite(biased).all().item():
            return False, "    CPU router reference contains NaN or Inf"

        score_error = score_atol + score_rtol * scores.abs()
        actual_biased = torch.gather(biased, dim=-1, index=actual)
        actual_error = torch.gather(score_error, dim=-1, index=actual)
        selected_mask = torch.zeros_like(biased, dtype=torch.bool)
        selected_mask.scatter_(dim=-1, index=actual, value=True)
        omitted_lower = (biased - score_error).masked_fill(
            selected_mask,
            float("-inf"),
        )
        best_omitted_lower, best_omitted_id = omitted_lower.max(dim=-1, keepdim=True)
        selected_upper = actual_biased + actual_error
        selected_floor_upper, selected_floor_pos = selected_upper.min(
            dim=-1,
            keepdim=True,
        )
        omitted_better = best_omitted_lower > selected_floor_upper

        # Top-k denotes an expert set, not an order: device sort may differ from
        # CPU stable sort on near-equal scores, but combine pairs (expert_id,
        # weight), so an identical set must not be reported as a mismatch.
        actual_set = torch.sort(actual, dim=-1).values
        expected_set = torch.sort(expected, dim=-1).values
        set_mismatch = actual_set != expected_set
        if not omitted_better.any().item() and not set_mismatch.any().item():
            return True, ""

        lines = [
            "    score route exceeds the calibrated FP32 top-k ambiguity band "
            f"(score_atol={score_atol:g}, score_rtol={score_rtol:g})"
        ]
        bad_rows = omitted_better.nonzero(as_tuple=False)[:max_show, 0].tolist()
        set_rows = set_mismatch.any(dim=-1).nonzero(as_tuple=False)[:max_show, 0].tolist()
        diagnostic_rows = sorted(set(bad_rows + set_rows))
        cpu_topk = torch.argsort(-biased, dim=-1, stable=True)[:, :TOPK]
        for row in diagnostic_rows[:max_show]:
            actual_ids = actual[row].tolist()
            cpu_ids = cpu_topk[row].tolist()
            actual_scores = [float(biased[row, i]) for i in actual_ids]
            cpu_scores = [float(biased[row, i]) for i in cpu_ids]
            lines.append(
                f"      diagnostic row {row}: device_ids={actual_ids} "
                f"device_scores={[round(v, 7) for v in actual_scores]} "
                f"cpu_ids={cpu_ids} cpu_scores={[round(v, 7) for v in cpu_scores]}"
            )
        for row in bad_rows:
            omitted_id = int(best_omitted_id[row, 0])
            selected_pos = int(selected_floor_pos[row, 0])
            selected_id = int(actual[row, selected_pos])
            regret = float(biased[row, omitted_id] - biased[row, selected_id])
            budget = float(score_error[row, omitted_id] + actual_error[row, selected_pos])
            lines.append(
                f"      row {row}: omitted id={omitted_id} beats selected "
                f"id={selected_id}; raw_regret={regret:.8g} budget={budget:.8g}"
            )
        shown = len(bad_rows)
        remaining = max_show - shown
        if remaining:
            for row in set_rows[:remaining]:
                lines.append(
                    f"      row {row} selected expert set differs: "
                    f"device={actual[row].tolist()} cpu={expected[row].tolist()}"
                )
        return False, "\n".join(lines)

    cmp.__name__ = (
        f"gate_indices_compare(score_atol={score_atol},score_rtol={score_rtol})"
    )
    return cmp


def mxfp8_activation_value_compare(actual, expected, *, actual_outputs, expected_outputs, **kwargs):
    """Compare MXFP8 activations by dequantised value, not by non-unique bytes.

    quant_mx may scale the payload and the E8M0 scale by the same power of two
    while keeping the dequantised value identical, so the V4.1 gate cannot reuse
    the V4 dual-bitwise comparison.
    """
    import torch

    from quant import decode_e8m0, unpack_mx_a_scale

    def decode(outputs, payload_name):
        payload = outputs[payload_name].cpu().float()
        scale_bytes = outputs["x_norm_scale"].cpu().contiguous().view(torch.uint8)
        scale = unpack_mx_a_scale(scale_bytes.reshape(T_PAD, MX_SCALE_GROUPS))
        scale = decode_e8m0(scale).repeat_interleave(MX_GROUP, dim=-1)
        return payload * scale

    actual_value = decode(actual_outputs, "x_norm_mx")
    expected_value = decode(expected_outputs, "x_norm_mx")
    from golden import ratio_allclose

    check = ratio_allclose(atol=2e-3, rtol=2e-3, max_error_ratio=0.0, max_show=8)
    return check(
        actual_value, expected_value, actual_outputs=actual_outputs, expected_outputs=expected_outputs, **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import torch
    from golden.runner import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-bias", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run(
        fn=l2_gate,
        specs=build_tensor_specs(num_tokens=args.num_tokens, token_bias=args.token_bias),
        golden_fn=golden_gate_normalized_core,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # payload/scale may use a different but equivalent E8M0 representation, so
            # compare dequantised values.
            "x_norm_mx": mxfp8_activation_value_compare,
            # The scale is already covered by the combined payload/scale comparator and
            # must not be compared bitwise on its own.
            "x_norm_scale": mxfp8_activation_value_compare,
            "indices": gate_indices_compare(args.num_tokens),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
