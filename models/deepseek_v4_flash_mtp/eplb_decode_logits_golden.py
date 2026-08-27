# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-network Torch Golden for the DeepSeek-V4 EP8 decode-logits benchmark."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from config import FLASH as MODEL_CONFIG
from decode_csa import (
    CMP_STORAGE_BLOCK_SIZE as CSA_CMP_STORAGE_BLOCK_SIZE,
    INNER_STATE_BLOCK_SIZE as CSA_INNER_STATE_BLOCK_SIZE,
    MAIN_STATE_BLOCK_SIZE as CSA_MAIN_STATE_BLOCK_SIZE,
    golden_attention_csa,
)
from decode_hca import (
    CMP_STORAGE_BLOCK_SIZE as HCA_CMP_STORAGE_BLOCK_SIZE,
    COMPRESS_STATE_BLOCK_SIZE as HCA_COMPRESS_STATE_BLOCK_SIZE,
    golden_attention_hca,
)
from decode_swa import B, BLOCK_SIZE, golden_attention_swa
from golden import ratio_allclose
from hc_head import golden_hc_head
from lm_head import TP_SIZE as LM_HEAD_TP_SIZE, golden_lm_head
from moe import N_RANKS, T, golden_moe
from rmsnorm import golden_rms_norm


_KIND_BY_RATIO = {0: "swa", 4: "csa", 128: "hca"}
_ROPE_PROFILE_BY_KIND = {"swa": 0, "csa": 1, "hca": 1}
FWD_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
LAYER_KINDS = tuple(
    _KIND_BY_RATIO[ratio]
    for ratio in MODEL_CONFIG.compress_ratios[:FWD_NUM_LAYERS]
)
CSA_NUM_LAYERS = LAYER_KINDS.count("csa")
HCA_NUM_LAYERS = LAYER_KINDS.count("hca")


def _kind_orders() -> dict[int, int]:
    orders = {}
    counters = {"swa": 0, "csa": 0, "hca": 0}
    for layer, kind in enumerate(LAYER_KINDS):
        orders[layer] = counters[kind]
        counters[kind] += 1
    return orders


KIND_ORDER = _kind_orders()


def _layer_rows(stacked: torch.Tensor, count: int, index: int) -> torch.Tensor:
    """Return one packed-layer view along the rank-stacked tensor's dim 1."""
    if count <= 0 or stacked.shape[1] % count != 0:
        raise ValueError(
            f"cannot split dim 1 of shape {tuple(stacked.shape)} into {count} layers"
        )
    unit = stacked.shape[1] // count
    return stacked[:, index * unit:(index + 1) * unit]


def _base_attention_views(
    tensors: dict[str, torch.Tensor],
    rank: int,
    layer: int,
    x_hc: torch.Tensor,
    x_out: torch.Tensor,
) -> dict[str, torch.Tensor]:
    def fwd(name: str) -> torch.Tensor:
        return _layer_rows(tensors[name], FWD_NUM_LAYERS, layer)[rank]

    kind = LAYER_KINDS[layer]
    rope_profile = _ROPE_PROFILE_BY_KIND[kind]
    return {
        "x_hc": x_hc,
        "hc_attn_fn": fwd("hc_attn_fn"),
        "hc_attn_scale": fwd("hc_attn_scale"),
        "hc_attn_base": fwd("hc_attn_base"),
        "attn_norm_w": fwd("attn_norm_w"),
        "wq_a": fwd("wq_a"),
        "wq_b": fwd("wq_b"),
        "wq_b_scale": fwd("wq_b_scale"),
        "wkv": fwd("wkv"),
        "gamma_cq": fwd("gamma_cq"),
        "gamma_ckv": fwd("gamma_ckv"),
        "freqs_cos": tensors["freqs_cos"][rank, rope_profile],
        "freqs_sin": tensors["freqs_sin"][rank, rope_profile],
        "kv_cache": fwd("kv_cache"),
        "position_ids": tensors["position_ids"][rank],
        "attn_sink": fwd("attn_sink"),
        "wo_a": fwd("wo_a"),
        "wo_b": fwd("wo_b"),
        "wo_b_scale": fwd("wo_b_scale"),
        "x_out": x_out,
    }


def _swa_attention_views(
    tensors: dict[str, torch.Tensor],
    rank: int,
    layer: int,
    x_hc: torch.Tensor,
    x_out: torch.Tensor,
) -> dict[str, torch.Tensor]:
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out)
    views.update({
        "swa_slot_mapping": tensors["swa_slot_mapping"][rank],
        "swa_indices": tensors["swa_indices"][rank],
        "swa_lens": tensors["swa_lens"][rank],
    })
    return views


def _hca_attention_views(
    tensors: dict[str, torch.Tensor],
    rank: int,
    layer: int,
    x_hc: torch.Tensor,
    x_out: torch.Tensor,
) -> dict[str, torch.Tensor]:
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out)
    order = KIND_ORDER[layer]

    def hca(name: str) -> torch.Tensor:
        return _layer_rows(tensors[name], HCA_NUM_LAYERS, order)[rank]

    views.update({
        "cmp_wkv": hca("hca_cmp_wkv"),
        "cmp_wgate": hca("hca_cmp_wgate"),
        "cmp_ape": hca("hca_cmp_ape"),
        "cmp_norm_w": hca("hca_cmp_norm_w"),
        "compress_state": hca("hca_compress_state"),
        "compress_state_block_table": tensors[
            "hca_compress_state_block_table"
        ][rank],
        "cmp_kv": _layer_rows(
            tensors["hca_cmp_kv"], FWD_NUM_LAYERS, layer
        )[rank],
        "cmp_block_table": tensors["cmp_block_table"][rank],
        "ori_slot_mapping": tensors["ori_slot_mapping"][rank],
        "window_swa_indices": tensors["window_swa_indices"][rank],
        "window_swa_lens": tensors["window_swa_lens"][rank],
        "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"][rank],
        "state_slot_mapping": tensors["hca_state_slot_mapping"][rank],
        "kv_seq_lens": tensors["kv_seq_lens"][rank],
    })
    return views


def _csa_attention_views(
    tensors: dict[str, torch.Tensor],
    rank: int,
    layer: int,
    x_hc: torch.Tensor,
    x_out: torch.Tensor,
) -> dict[str, torch.Tensor]:
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out)
    order = KIND_ORDER[layer]

    def csa(name: str) -> torch.Tensor:
        return _layer_rows(tensors[name], CSA_NUM_LAYERS, order)[rank]

    if layer == FWD_NUM_LAYERS - 1:
        window_swa_indices = tensors["swa_indices"][rank]
        window_swa_lens = tensors["swa_lens"][rank]
    else:
        window_swa_indices = tensors["window_swa_indices"][rank]
        window_swa_lens = tensors["window_swa_lens"][rank]

    views.update({
        "cmp_wkv": csa("csa_cmp_wkv"),
        "cmp_wgate": csa("csa_cmp_wgate"),
        "cmp_ape": csa("csa_cmp_ape"),
        "cmp_norm_w": csa("csa_cmp_norm_w"),
        "compress_state": csa("csa_compress_state"),
        "compress_state_block_table": tensors[
            "csa_compress_state_block_table"
        ][rank],
        "idx_wq_b": csa("csa_idx_wq_b"),
        "idx_wq_b_scale": csa("csa_idx_wq_b_scale"),
        "weights_proj": csa("csa_weights_proj"),
        "hadamard_idx": csa("csa_hadamard_idx"),
        "inner_wkv": csa("csa_inner_wkv"),
        "inner_wgate": csa("csa_inner_wgate"),
        "inner_ape": csa("csa_inner_ape"),
        "inner_norm_w": csa("csa_inner_norm_w"),
        "inner_compress_state": csa("csa_inner_compress_state"),
        "inner_compress_state_block_table": tensors[
            "csa_inner_compress_state_block_table"
        ][rank],
        "cmp_kv": _layer_rows(
            tensors["csa_cmp_kv"], FWD_NUM_LAYERS, layer
        )[rank],
        "cmp_block_table": tensors["cmp_block_table"][rank],
        "idx_kv_cache": csa("idx_kv_cache"),
        "idx_kv_scale": csa("idx_kv_scale"),
        "idx_block_table": tensors["idx_block_table"][rank],
        "ori_slot_mapping": tensors["ori_slot_mapping"][rank],
        "window_swa_indices": window_swa_indices,
        "window_swa_lens": window_swa_lens,
        "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"][rank],
        "idx_slot_mapping": tensors["csa_idx_slot_mapping"][rank],
        "state_slot_mapping": tensors["csa_state_slot_mapping"][rank],
        "inner_state_slot_mapping": tensors[
            "csa_inner_state_slot_mapping"
        ][rank],
        "kv_seq_lens": tensors["kv_seq_lens"][rank],
    })
    return views


_ATTENTION_VIEWS = {
    "swa": _swa_attention_views,
    "hca": _hca_attention_views,
    "csa": _csa_attention_views,
}
_ATTENTION_GOLDENS = {
    "swa": golden_attention_swa,
    "hca": golden_attention_hca,
    "csa": golden_attention_csa,
}

_MOE_LAYER_STACKED = (
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
)


def _moe_views(
    tensors: dict[str, torch.Tensor],
    layer: int,
    x_hc: torch.Tensor,
    x_next: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    views: dict[str, torch.Tensor | int] = {
        name: _layer_rows(tensors[name], FWD_NUM_LAYERS, layer)
        for name in _MOE_LAYER_STACKED
    }
    views.update({
        "x_hc": x_hc,
        "input_ids": tensors["input_ids"],
        "layer_id": 0,
        "num_tokens": T,
        "x_next": x_next,
    })
    return views


def _golden_lm_head_groups(tensors: dict[str, torch.Tensor]) -> None:
    for group_base in range(0, N_RANKS, LM_HEAD_TP_SIZE):
        group_end = group_base + LM_HEAD_TP_SIZE
        golden_lm_head({
            "hidden_states": tensors["hidden_out"][group_base:group_end],
            "lm_head_weight": tensors["lm_head_weight"][group_base:group_end],
            "logit_row_indices": tensors[
                "logit_row_indices"
            ][group_base:group_end],
            "logits": tensors["logits"][group_base:group_end],
        })


def golden_eplb_decode_logits(tensors: dict[str, torch.Tensor]) -> None:
    """Replay the 43-layer decode stack and fill all benchmark outputs."""
    x_hc = tensors["x_hc"]
    for layer, kind in enumerate(LAYER_KINDS):
        x_attn = torch.zeros_like(x_hc)
        for rank in range(N_RANKS):
            views = _ATTENTION_VIEWS[kind](
                tensors,
                rank,
                layer,
                x_hc[rank],
                x_attn[rank],
            )
            _ATTENTION_GOLDENS[kind](views)

        x_next = (
            tensors["pre_hc_hidden_out"]
            if layer == FWD_NUM_LAYERS - 1
            else torch.zeros_like(x_hc)
        )
        golden_moe(_moe_views(tensors, layer, x_attn, x_next))
        x_hc = x_next

    for rank in range(N_RANKS):
        x_head = torch.empty_like(tensors["hidden_out"][rank])
        golden_hc_head({
            "x_hc": tensors["pre_hc_hidden_out"][rank],
            "hc_head_fn": tensors["hc_head_fn"][rank],
            "hc_head_scale": tensors["hc_head_scale"][rank],
            "hc_head_base": tensors["hc_head_base"][rank],
            "y": x_head,
        })
        tensors["hidden_out"][rank].copy_(
            golden_rms_norm(x_head, tensors["final_norm_w"][rank])
        )

    _golden_lm_head_groups(tensors)


def _cosine_rel_l2_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[bool, str, float, float]:
    """Measure one logical output row with finite-value hard checks."""
    if actual.shape != expected.shape:
        return False, (
            f"shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        ), float("nan"), float("nan")

    actual_fp64 = actual.double().reshape(-1)
    expected_fp64 = expected.double().reshape(-1)
    if not bool(torch.isfinite(actual_fp64).all()):
        return False, "actual row has non-finite values", float("nan"), float("nan")
    if not bool(torch.isfinite(expected_fp64).all()):
        return False, "Golden row has non-finite values", float("nan"), float("nan")

    actual_norm = actual_fp64.norm()
    expected_norm = expected_fp64.norm()
    denominator = float(actual_norm * expected_norm)
    if denominator > 0.0:
        cosine = float(actual_fp64 @ expected_fp64) / denominator
    elif float(actual_norm) == 0.0 and float(expected_norm) == 0.0:
        cosine = 1.0
    else:
        cosine = 0.0
    rel_l2 = float(
        (actual_fp64 - expected_fp64).norm()
        / expected_norm.clamp_min(1e-12)
    )
    if not math.isfinite(cosine) or not math.isfinite(rel_l2):
        return False, (
            f"derived metric is non-finite: cosine={cosine} "
            f"rel_l2={rel_l2}"
        ), cosine, rel_l2
    return True, "", cosine, rel_l2


def _stacked_mapped_pool_compare(
    layer_mapping_names: tuple[str | None, ...],
    *,
    layer_labels: tuple[int, ...],
    block_size: int,
    pool_name: str,
    atol: float,
    rtol: float,
    max_error_ratio: float,
    min_cosine: float | None = None,
    max_rel_l2: float | None = None,
    max_bad_row_ratio: float = 0.0,
    hard_min_cosine: float | None = None,
    hard_max_rel_l2: float | None = None,
    strict_layer_labels: tuple[int, ...] = (),
    expected_active_rows: int | None = None,
    emit_point_metrics: bool = False,
) -> Callable:
    strict_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )
    semantic_compare = min_cosine is not None or max_rel_l2 is not None
    if semantic_compare and (min_cosine is None or max_rel_l2 is None):
        raise ValueError("min_cosine and max_rel_l2 must be set together")
    if semantic_compare:
        hard_min_cosine = (
            min_cosine if hard_min_cosine is None else hard_min_cosine
        )
        hard_max_rel_l2 = (
            max_rel_l2 if hard_max_rel_l2 is None else hard_max_rel_l2
        )
    if semantic_compare and not (
        math.isfinite(min_cosine)
        and -1.0 <= min_cosine <= 1.0
        and math.isfinite(max_rel_l2)
        and max_rel_l2 >= 0.0
        and math.isfinite(max_bad_row_ratio)
        and 0.0 <= max_bad_row_ratio <= 1.0
        and math.isfinite(hard_min_cosine)
        and -1.0 <= hard_min_cosine <= min_cosine
        and math.isfinite(hard_max_rel_l2)
        and hard_max_rel_l2 >= max_rel_l2
    ):
        raise ValueError(
            "semantic thresholds require finite cosine values in [-1, 1], "
            "hard_min_cosine <= min_cosine, finite relative-L2 values >= 0, "
            "hard_max_rel_l2 >= max_rel_l2, and max_bad_row_ratio in [0, 1]"
        )
    if not semantic_compare and (
        max_bad_row_ratio != 0.0
        or hard_min_cosine is not None
        or hard_max_rel_l2 is not None
    ):
        raise ValueError("semantic row budgets require cosine/relative-L2 thresholds")
    if expected_active_rows is not None and expected_active_rows < 0:
        raise ValueError("expected_active_rows must be non-negative")
    point_atol = strict_compare.atol_override
    point_rtol = strict_compare.rtol_override
    if emit_point_metrics and (point_atol is None or point_rtol is None):
        raise ValueError("point metrics require explicit strict tolerances")
    strict_layers = frozenset(strict_layer_labels)
    unknown_strict_layers = strict_layers - frozenset(layer_labels)
    if unknown_strict_layers:
        raise ValueError(
            f"strict layers are absent from {pool_name}: "
            f"{sorted(unknown_strict_layers)}"
        )

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        del actual_outputs, expected_outputs
        if actual.shape != expected.shape:
            return False, (
                f"    {pool_name} shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        layer_count = len(layer_mapping_names)
        if (
            actual.ndim < 4
            or actual.shape[0] != N_RANKS
            or actual.shape[1] % layer_count != 0
            or actual.shape[2] != block_size
        ):
            return False, (
                f"    invalid stacked {pool_name} layout: {tuple(actual.shape)}"
            )

        blocks_per_layer = actual.shape[1] // layer_count
        rows_per_layer = blocks_per_layer * block_size
        actual_rows = actual.reshape(N_RANKS, layer_count, rows_per_layer, -1)
        expected_rows = expected.reshape(N_RANKS, layer_count, rows_per_layer, -1)
        failures = []
        failure_count = 0
        measured_rows = 0
        bad_rows = 0
        bad_row_samples = []
        worst_cosine = 1.0
        worst_cosine_label = ""
        worst_rel_l2 = 0.0
        worst_rel_l2_label = ""
        point_error_count = 0
        point_value_count = 0
        point_max_abs = 0.0
        point_max_abs_label = ""

        def record_failure(message: str) -> None:
            nonlocal failure_count
            failure_count += 1
            if len(failures) < 16:
                failures.append(message)

        for layer_index, (layer_label, mapping_name) in enumerate(
            zip(layer_labels, layer_mapping_names, strict=True)
        ):
            for rank in range(N_RANKS):
                if mapping_name is None:
                    mapping = torch.empty(0, dtype=torch.int64)
                else:
                    if mapping_name not in inputs:
                        return False, (
                            f"    missing mapping input {mapping_name!r}"
                        )
                    mapping_input = inputs[mapping_name]
                    if mapping_input.dtype not in (
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    ):
                        return False, (
                            f"    mapping input {mapping_name!r} must have an "
                            f"integer dtype, got {mapping_input.dtype}"
                        )
                    if (
                        mapping_input.ndim != 2
                        or mapping_input.shape[0] != N_RANKS
                        or mapping_input.shape[1] < T
                    ):
                        return False, (
                            f"    mapping input {mapping_name!r} must have shape "
                            f"[{N_RANKS}, >= {T}], got {tuple(mapping_input.shape)}"
                        )
                    mapping = mapping_input[rank, :T].to(torch.int64)
                    invalid_negative = mapping < -1
                    if invalid_negative.any().item():
                        token = int(
                            invalid_negative.nonzero(as_tuple=False)[0].item()
                        )
                        value = int(mapping[token].item())
                        return False, (
                            f"    mapping input {mapping_name!r}[{rank}, {token}]="
                            f"{value} is invalid; only -1 is a negative sentinel"
                        )
                    mapping = mapping[mapping >= 0]
                if bool((mapping >= rows_per_layer).any()):
                    record_failure(
                        f"    {pool_name} layer={layer_label} rank={rank} "
                        "has an out-of-range mapping"
                    )
                    continue
                if mapping.numel() != torch.unique(mapping).numel():
                    record_failure(
                        f"    {pool_name} layer={layer_label} rank={rank} "
                        "has duplicate mapped rows"
                    )
                    continue

                written_rows = torch.zeros(rows_per_layer, dtype=torch.bool)
                written_rows[mapping] = True
                actual_layer = actual_rows[rank, layer_index]
                expected_layer = expected_rows[rank, layer_index]
                if not torch.equal(
                    actual_layer[~written_rows],
                    expected_layer[~written_rows],
                ):
                    record_failure(
                        f"    {pool_name} layer={layer_label} rank={rank} "
                        "changed outside mapped rows"
                    )
                    continue
                if mapping.numel() == 0:
                    continue

                measured_rows += int(mapping.numel())

                if emit_point_metrics:
                    actual_mapped = actual_layer[mapping].to(torch.float32)
                    expected_mapped = expected_layer[mapping].to(torch.float32)
                    point_diff = (actual_mapped - expected_mapped).abs()
                    point_tolerance = (
                        float(point_atol)
                        + float(point_rtol) * expected_mapped.abs()
                    )
                    point_error_count += int(
                        (point_diff > point_tolerance).sum().item()
                    )
                    point_value_count += point_diff.numel()
                    point_max, point_flat = torch.max(
                        point_diff.flatten(),
                        dim=0,
                    )
                    point_max_value = float(point_max.item())
                    if point_max_value > point_max_abs:
                        point_flat_index = int(point_flat.item())
                        point_width = actual_mapped[0].numel()
                        mapped_row = point_flat_index // point_width
                        component = point_flat_index % point_width
                        point_max_abs = point_max_value
                        point_max_abs_label = (
                            f"{pool_name} layer={layer_label} rank={rank} "
                            f"row={int(mapping[mapped_row].item())} "
                            f"flat_component={component}"
                        )

                if not semantic_compare or layer_label in strict_layers:
                    ok, detail = strict_compare(
                        actual_layer[mapping],
                        expected_layer[mapping],
                        actual_outputs={},
                        expected_outputs={},
                        inputs=inputs,
                        rtol=rtol,
                        atol=atol,
                    )
                    if not ok:
                        record_failure(
                            f"    {pool_name} layer={layer_label} "
                            f"rank={rank}\n{detail}"
                        )

                if not semantic_compare:
                    continue
                for physical_row in mapping.tolist():
                    label = (
                        f"{pool_name} layer={layer_label} rank={rank} "
                        f"row={physical_row}"
                    )
                    finite, detail, cosine, rel_l2 = _cosine_rel_l2_metrics(
                        actual_layer[physical_row],
                        expected_layer[physical_row],
                    )
                    if finite:
                        if cosine < worst_cosine:
                            worst_cosine = cosine
                            worst_cosine_label = label
                        if rel_l2 > worst_rel_l2:
                            worst_rel_l2 = rel_l2
                            worst_rel_l2_label = label
                    if not finite:
                        record_failure(f"    {label}: {detail}")
                        bad_rows += 1
                    elif cosine < min_cosine or rel_l2 > max_rel_l2:
                        bad_rows += 1
                        if len(bad_row_samples) < 16:
                            bad_row_samples.append(
                                f"    {label}: cosine={cosine:.6f} "
                                f"(soft min {min_cosine}) "
                                f"rel_l2={rel_l2:.6f} "
                                f"(soft max {max_rel_l2})"
                            )
                    if finite and (
                        cosine < hard_min_cosine
                        or rel_l2 > hard_max_rel_l2
                    ):
                        record_failure(
                            f"    {label}: cosine={cosine:.6f} "
                            f"(hard min {hard_min_cosine}) "
                            f"rel_l2={rel_l2:.6f} "
                            f"(hard max {hard_max_rel_l2})"
                        )

        if (
            expected_active_rows is not None
            and measured_rows != expected_active_rows
        ):
            record_failure(
                f"    {pool_name} active-row coverage {measured_rows} "
                f"does not match expected {expected_active_rows}"
            )
        if emit_point_metrics:
            point_error_ratio = (
                point_error_count / point_value_count
                if point_value_count
                else 0.0
            )
            print(
                f"[GOLDEN METRIC] output={pool_name} "
                f"active_rows={measured_rows} "
                f"error_points={point_error_count} "
                f"error_point_ratio={point_error_ratio:.6f} "
                f"max_abs={point_max_abs:.6f} "
                f"max_abs_at=\"{point_max_abs_label}\""
            )
        if semantic_compare:
            bad_row_ratio = bad_rows / measured_rows if measured_rows else 0.0
            if measured_rows:
                print(
                    f"[GOLDEN METRIC] output={pool_name} "
                    f"active_rows={measured_rows} "
                    f"bad_rows={bad_rows} "
                    f"bad_row_ratio={bad_row_ratio:.6f} "
                    f"min_cosine={worst_cosine:.6f} "
                    f"min_cosine_at=\"{worst_cosine_label}\" "
                    f"max_rel_l2={worst_rel_l2:.6f} "
                    f"max_rel_l2_at=\"{worst_rel_l2_label}\""
                )
            else:
                print(
                    f"[GOLDEN METRIC] output={pool_name} active_rows=0 "
                    "bad_rows=0 bad_row_ratio=0.000000 "
                    "min_cosine=NA max_rel_l2=NA"
                )
            if bad_row_ratio > max_bad_row_ratio:
                record_failure(
                    f"    {pool_name} semantic bad-row ratio "
                    f"{bad_row_ratio:.6f} exceeds {max_bad_row_ratio} "
                    f"({bad_rows}/{measured_rows})"
                )
                failures.extend(
                    bad_row_samples[: max(0, 16 - len(failures))]
                )
        if failure_count > len(failures):
            failures.append(
                f"    ... omitted {failure_count - len(failures)} "
                f"additional {pool_name} failures"
            )
        return failure_count == 0, "\n".join(failures)

    if semantic_compare:
        compare.__name__ = (
            f"stacked_mapped_pool_cosine_rel_l2({pool_name}, "
            f"min_cosine={min_cosine}, max_rel_l2={max_rel_l2}, "
            f"max_bad_row_ratio={max_bad_row_ratio}, "
            f"hard_min_cosine={hard_min_cosine}, "
            f"hard_max_rel_l2={hard_max_rel_l2}, "
            f"strict_layers={tuple(sorted(strict_layers))}, "
            f"strict_atol={atol}, strict_rtol={rtol}, "
            f"strict_max_error_ratio={max_error_ratio}, "
            f"expected_active_rows={expected_active_rows})"
        )
    else:
        compare.__name__ = (
            f"stacked_mapped_pool_ratio_allclose({pool_name}, atol={atol}, "
            f"rtol={rtol}, max_error_ratio={max_error_ratio}, "
            f"expected_active_rows={expected_active_rows})"
        )
    return compare


def _rank_token_cosine_compare(
    output_name: str,
    *,
    min_cosine: float = 0.99,
    max_rel_l2: float = 0.10,
    expected_active_rows: int | None = None,
) -> Callable:
    """Compare every rank/token row without diluting one bad token globally."""
    if not (
        math.isfinite(min_cosine)
        and -1.0 <= min_cosine <= 1.0
        and math.isfinite(max_rel_l2)
        and max_rel_l2 >= 0.0
    ):
        raise ValueError(
            "row thresholds require finite min_cosine in [-1, 1] "
            "and finite max_rel_l2 >= 0"
        )
    if expected_active_rows is not None and expected_active_rows < 0:
        raise ValueError("expected_active_rows must be non-negative")

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        **_kwargs,
    ) -> tuple[bool, str]:
        if actual.shape != expected.shape:
            return False, (
                f"    {output_name} shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if actual.ndim < 3 or actual.shape[0] != N_RANKS:
            return False, f"    invalid {output_name} layout: {tuple(actual.shape)}"

        failures = []
        measured_rows = actual.shape[0] * actual.shape[1]
        if (
            expected_active_rows is not None
            and measured_rows != expected_active_rows
        ):
            failures.append(
                f"    {output_name} active-row coverage {measured_rows} "
                f"does not match expected {expected_active_rows}"
            )
        worst_cosine = 1.0
        worst_cosine_label = ""
        worst_rel_l2 = 0.0
        worst_rel_l2_label = ""
        for rank in range(actual.shape[0]):
            for token in range(actual.shape[1]):
                label = f"{output_name} rank={rank} token={token}"
                finite, detail, cosine, rel_l2 = _cosine_rel_l2_metrics(
                    actual[rank, token],
                    expected[rank, token],
                )
                if finite:
                    if cosine < worst_cosine:
                        worst_cosine = cosine
                        worst_cosine_label = label
                    if rel_l2 > worst_rel_l2:
                        worst_rel_l2 = rel_l2
                        worst_rel_l2_label = label
                if not finite:
                    failures.append(f"    {label}: {detail}")
                elif cosine < min_cosine or rel_l2 > max_rel_l2:
                    failures.append(
                        f"    {label}: cosine={cosine:.6f} "
                        f"(min {min_cosine}) rel_l2={rel_l2:.6f} "
                        f"(max {max_rel_l2})"
                    )

        print(
            f"[GOLDEN METRIC] output={output_name} "
            f"active_rows={measured_rows} "
            f"min_cosine={worst_cosine:.6f} "
            f"min_cosine_at=\"{worst_cosine_label}\" "
            f"max_rel_l2={worst_rel_l2:.6f} "
            f"max_rel_l2_at=\"{worst_rel_l2_label}\""
        )
        if len(failures) > 16:
            failures = [
                *failures[:16],
                f"    ... omitted {len(failures) - 16} additional "
                f"{output_name} failures",
            ]
        return not failures, "\n".join(failures)

    compare.__name__ = (
        f"rank_token_cosine_rel_l2({output_name}, min_cosine={min_cosine}, "
        f"max_rel_l2={max_rel_l2}, "
        f"expected_active_rows={expected_active_rows})"
    )
    return compare


def _logits_cosine_compare(
    min_cosine: float = 0.99,
    max_rel_l2: float = 0.10,
    expected_active_rows: int | None = None,
) -> Callable:
    if not (
        math.isfinite(min_cosine)
        and -1.0 <= min_cosine <= 1.0
        and math.isfinite(max_rel_l2)
        and max_rel_l2 >= 0.0
    ):
        raise ValueError(
            "logits thresholds require finite min_cosine in [-1, 1] "
            "and finite max_rel_l2 >= 0"
        )
    if expected_active_rows is not None and expected_active_rows < 0:
        raise ValueError("expected_active_rows must be non-negative")

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        del actual_outputs, expected_outputs, rtol, atol
        if actual.shape != expected.shape:
            return False, (
                f"    logits shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )

        row_indices = inputs.get("logit_row_indices")
        if row_indices is None:
            return False, "    missing mapping input 'logit_row_indices'"
        if row_indices.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            return False, (
                "    'logit_row_indices' must have an integer dtype, "
                f"got {row_indices.dtype}"
            )
        if tuple(row_indices.shape) != tuple(actual.shape[:2]):
            return False, (
                "    'logit_row_indices' must match the leading logits shape "
                f"{tuple(actual.shape[:2])}, got {tuple(row_indices.shape)}"
            )
        invalid_negative = row_indices < -1
        if invalid_negative.any().item():
            first = invalid_negative.nonzero(as_tuple=False)[0]
            rank = int(first[0].item())
            row = int(first[1].item())
            value = int(row_indices[rank, row].item())
            return False, (
                f"    'logit_row_indices'[{rank}, {row}]={value} is invalid; "
                "only -1 is a negative sentinel"
            )
        actual_fp32 = actual.float()
        expected_fp32 = expected.float()
        failures = []
        measured_rows = 0
        worst_cosine = 1.0
        worst_cosine_label = ""
        worst_rel_l2 = 0.0
        worst_rel_l2_label = ""
        active_rows = int((row_indices >= 0).sum().item())
        if expected_active_rows is not None and active_rows != expected_active_rows:
            failures.append(
                f"    logits active-row coverage {active_rows} "
                f"does not match expected {expected_active_rows}"
            )
        for rank in range(actual.shape[0]):
            for row in range(actual.shape[1]):
                if int(row_indices[rank, row]) < 0:
                    if not torch.equal(actual[rank, row], expected[rank, row]):
                        failures.append(
                            f"    inactive logits[{rank},{row}] changed"
                        )
                    continue
                finite, detail, cosine, rel_l2 = _cosine_rel_l2_metrics(
                    actual_fp32[rank, row],
                    expected_fp32[rank, row],
                )
                if not finite:
                    failures.append(f"    logits[{rank},{row}]: {detail}")
                    continue
                measured_rows += 1
                label = f"logits rank={rank} row={row}"
                if cosine < worst_cosine:
                    worst_cosine = cosine
                    worst_cosine_label = label
                if rel_l2 > worst_rel_l2:
                    worst_rel_l2 = rel_l2
                    worst_rel_l2_label = label
                if cosine < min_cosine or rel_l2 > max_rel_l2:
                    failures.append(
                        f"    logits[{rank},{row}] cosine={cosine:.6f} "
                        f"rel_l2={rel_l2:.6f}"
                    )
        print(
            f"[GOLDEN METRIC] output=logits active_rows={measured_rows} "
            f"min_cosine={worst_cosine:.6f} "
            f"min_cosine_at=\"{worst_cosine_label}\" "
            f"max_rel_l2={worst_rel_l2:.6f} "
            f"max_rel_l2_at=\"{worst_rel_l2_label}\""
        )
        return not failures, "\n".join(failures)

    compare.__name__ = (
        f"logits_cosine_compare(min_cosine={min_cosine}, "
        f"max_rel_l2={max_rel_l2}, "
        f"expected_active_rows={expected_active_rows})"
    )
    return compare


def build_eplb_decode_logits_compare_fn(
    *,
    compressed_cache_active_write: bool = False,
) -> dict[str, Callable]:
    """Build full-network comparators for all 11 decode-logits outputs."""
    all_layers = tuple(range(FWD_NUM_LAYERS))
    csa_layers = tuple(
        layer for layer, kind in enumerate(LAYER_KINDS) if kind == "csa"
    )
    hca_layers = tuple(
        layer for layer, kind in enumerate(LAYER_KINDS) if kind == "hca"
    )
    hca_compressed_active_rows = (
        N_RANKS * B * HCA_NUM_LAYERS
        if compressed_cache_active_write
        else 0
    )
    csa_compressed_active_rows = (
        N_RANKS * B * CSA_NUM_LAYERS
        if compressed_cache_active_write
        else 0
    )

    def pool(
        mapping_names: tuple[str | None, ...],
        *,
        layer_labels: tuple[int, ...],
        block_size: int,
        pool_name: str,
        atol: float,
        rtol: float,
        max_error_ratio: float,
        expected_active_rows: int,
        semantic: bool = True,
        strict_layer_labels: tuple[int, ...] = (),
        emit_point_metrics: bool = False,
        min_cosine: float = 0.99,
        max_rel_l2: float = 0.10,
        max_bad_row_ratio: float = 0.02,
        hard_min_cosine: float = 0.98,
        hard_max_rel_l2: float = 0.50,
    ) -> Callable:
        return _stacked_mapped_pool_compare(
            mapping_names,
            layer_labels=layer_labels,
            block_size=block_size,
            pool_name=pool_name,
            atol=atol,
            rtol=rtol,
            max_error_ratio=max_error_ratio,
            min_cosine=min_cosine if semantic else None,
            max_rel_l2=max_rel_l2 if semantic else None,
            max_bad_row_ratio=max_bad_row_ratio if semantic else 0.0,
            hard_min_cosine=hard_min_cosine if semantic else None,
            hard_max_rel_l2=hard_max_rel_l2 if semantic else None,
            strict_layer_labels=strict_layer_labels,
            expected_active_rows=expected_active_rows,
            emit_point_metrics=emit_point_metrics,
        )

    return {
        "pre_hc_hidden_out": _rank_token_cosine_compare(
            "pre_hc_hidden_out",
            max_rel_l2=0.11 if compressed_cache_active_write else 0.10,
            expected_active_rows=N_RANKS * T,
        ),
        "hidden_out": _rank_token_cosine_compare(
            "hidden_out", expected_active_rows=N_RANKS * T
        ),
        "logits": _logits_cosine_compare(expected_active_rows=N_RANKS * T),
        "kv_cache": pool(
            tuple(
                "swa_slot_mapping" if kind == "swa" else "ori_slot_mapping"
                for kind in LAYER_KINDS
            ),
            layer_labels=all_layers,
            block_size=BLOCK_SIZE,
            pool_name="kv_cache",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
            expected_active_rows=N_RANKS * T * FWD_NUM_LAYERS,
            strict_layer_labels=(0,),
        ),
        "hca_cmp_kv": pool(
            tuple(
                "hca_cmp_slot_mapping" if kind == "hca" else None
                for kind in LAYER_KINDS
            ),
            layer_labels=all_layers,
            block_size=HCA_CMP_STORAGE_BLOCK_SIZE,
            pool_name="hca_cmp_kv",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
            expected_active_rows=hca_compressed_active_rows,
            max_bad_row_ratio=0.03 if compressed_cache_active_write else 0.02,
            hard_min_cosine=0.97 if compressed_cache_active_write else 0.98,
        ),
        "csa_cmp_kv": pool(
            tuple(
                "csa_cmp_slot_mapping" if kind == "csa" else None
                for kind in LAYER_KINDS
            ),
            layer_labels=all_layers,
            block_size=CSA_CMP_STORAGE_BLOCK_SIZE,
            pool_name="csa_cmp_kv",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
            expected_active_rows=csa_compressed_active_rows,
            max_bad_row_ratio=0.025 if compressed_cache_active_write else 0.02,
            hard_min_cosine=0.96 if compressed_cache_active_write else 0.98,
        ),
        "hca_compress_state": pool(
            ("hca_state_slot_mapping",) * HCA_NUM_LAYERS,
            layer_labels=hca_layers,
            block_size=HCA_COMPRESS_STATE_BLOCK_SIZE,
            pool_name="hca_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
            expected_active_rows=N_RANKS * T * HCA_NUM_LAYERS,
            hard_max_rel_l2=0.60 if compressed_cache_active_write else 0.50,
        ),
        "csa_compress_state": pool(
            ("csa_state_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layers,
            block_size=CSA_MAIN_STATE_BLOCK_SIZE,
            pool_name="csa_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
            expected_active_rows=N_RANKS * T * CSA_NUM_LAYERS,
        ),
        "csa_inner_compress_state": pool(
            ("csa_inner_state_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layers,
            block_size=CSA_INNER_STATE_BLOCK_SIZE,
            pool_name="csa_inner_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
            expected_active_rows=N_RANKS * T * CSA_NUM_LAYERS,
        ),
        "idx_kv_cache": pool(
            ("csa_idx_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layers,
            block_size=CSA_CMP_STORAGE_BLOCK_SIZE,
            pool_name="idx_kv_cache",
            atol=1,
            rtol=0,
            max_error_ratio=0.02,
            expected_active_rows=csa_compressed_active_rows,
            # An active compressed index cache is quantized after values have
            # accumulated through earlier layers. Its contract therefore uses
            # the same per-logical-row quality gate as the other active cache
            # pools, while preserving unmapped rows byte-for-byte. The default
            # 8192 profile has no active rows and retains its exact pointwise
            # comparator name and behavior.
            semantic=compressed_cache_active_write,
            emit_point_metrics=compressed_cache_active_write,
            max_bad_row_ratio=0.035 if compressed_cache_active_write else 0.02,
            hard_min_cosine=0.96 if compressed_cache_active_write else 0.98,
            hard_max_rel_l2=0.35 if compressed_cache_active_write else 0.50,
        ),
        "idx_kv_scale": pool(
            ("csa_idx_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layers,
            block_size=CSA_CMP_STORAGE_BLOCK_SIZE,
            pool_name="idx_kv_scale",
            atol=1e-3,
            rtol=1e-2,
            max_error_ratio=0.01,
            expected_active_rows=csa_compressed_active_rows,
        ),
    }
