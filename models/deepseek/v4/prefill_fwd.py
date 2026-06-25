# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI marker: run on >=2 NPUs via $DEVICE_RANGE instead of single $DEVICE_ID
# ci: no-sim    # CI marker: full multi-layer / multi-card forward — device-only, skip on *sim
"""DeepSeek-V4 packed prefill multi-layer bring-up driver.

This driver intentionally keeps the layer kernel in ``prefill_layer.py`` as the
single source of truth.  It stacks synthetic layer fixtures and threads the
``x_next`` output from one layer into the next layer's ``x_hc`` input.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

import pypto.language as pl
import pypto.language.distributed as pld
from config import FLASH as MODEL_CONFIG
from golden import ScalarSpec, TensorSpec
from golden.runner import (
    RunResult,
    _Stage,
    _compute_golden,
    _consume_runtime_harness_keys,
    _execute_via_runner,
    _prepare_inputs,
    _try_l3_dispatch,
    _validate,
)
from pypto.ir.distributed_compiled_program import DistributedConfig

from prefill_layer import (
    BLOCK_SIZE,
    CSA_CMP_BLOCK_NUM,
    CSA_COMPRESS_RATIO,
    CSA_MAIN_OUT_DIM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    D,
    H,
    HEAD_DIM,
    HCA_COMPRESS_RATIO,
    HCA_MAIN_OUT_DIM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    HC_DIM,
    HC_MULT,
    HOST_TENSOR_ORDER,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_PAD,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAX_SEQ_LEN,
    MIX_HC,
    MOE_INTER,
    N_RANKS,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_ROUTES,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    RECV_MAX,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    SPARSE_TOPK,
    T,
    TOPK,
    VOCAB,
    W_PAD,
    build_tensor_specs,
    golden_prefill_layer,
    prefill_layer_core,
    valid_ratio_reldiff,
)
X_NEXT_DIFF_THD = 0.01
X_NEXT_PCT_THD = 0.05
PREFILL_FWD_LAYER_CHOICES = (1, 2, 4, 10, 20, 43)
FWD_KV_CACHE_ROWS = CSA_ORI_BLOCK_NUM * BLOCK_SIZE
FWD_CMP_CACHE_ROWS = CSA_CMP_BLOCK_NUM * BLOCK_SIZE
FWD_X_ROWS = T * HC_MULT
FWD_CACHE_COPY_ROWS = 16
FWD_X_COPY_ROWS = 4
FWD_RANK_SLICED_CACHE_NAMES = {"kv_cache", "cmp_kv", "idx_kv_cache"}
assert N_RANKS == 2, "rank-split prefill_fwd bring-up currently targets EP2"


def attention_kind_for_layer(layer_id: int) -> str:
    ratio = int(MODEL_CONFIG.compress_ratios[layer_id])
    if ratio == 0:
        return "swa"
    if ratio == 4:
        return "csa"
    if ratio == 128:
        return "hca"
    raise ValueError(f"unsupported compress ratio {ratio} at layer {layer_id}")


def _parse_num_layers_argv() -> int:
    for i, tok in enumerate(sys.argv):
        if tok == "--num-layers" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith("--num-layers="):
            return int(tok.split("=", 1)[1])
    return 4


PREFILL_FWD_ACTIVE_LAYERS = _parse_num_layers_argv()
if PREFILL_FWD_ACTIVE_LAYERS not in PREFILL_FWD_LAYER_CHOICES:
    raise ValueError(f"--num-layers must be one of {PREFILL_FWD_LAYER_CHOICES}, got {PREFILL_FWD_ACTIVE_LAYERS}")
FWD_COMPRESS_RATIOS = tuple(int(MODEL_CONFIG.compress_ratios[i]) for i in range(PREFILL_FWD_ACTIVE_LAYERS))
FWD_HCA_LAYER_IDS = tuple(i for i, ratio in enumerate(FWD_COMPRESS_RATIOS) if ratio == 128)
FWD_CSA_LAYER_IDS = tuple(i for i, ratio in enumerate(FWD_COMPRESS_RATIOS) if ratio == 4)
FWD_HCA_LAYER_COUNT = max(1, len(FWD_HCA_LAYER_IDS))
FWD_CSA_LAYER_COUNT = max(1, len(FWD_CSA_LAYER_IDS))
FWD_HCA_INDEX_BY_LAYER = tuple(
    FWD_HCA_LAYER_IDS.index(i) if i in FWD_HCA_LAYER_IDS else 0
    for i in range(PREFILL_FWD_ACTIVE_LAYERS)
)
FWD_CSA_INDEX_BY_LAYER = tuple(
    FWD_CSA_LAYER_IDS.index(i) if i in FWD_CSA_LAYER_IDS else 0
    for i in range(PREFILL_FWD_ACTIVE_LAYERS)
)
FWD_HCA_COMPACT_NAMES = {
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_cmp_kv_state",
    "hca_cmp_score_state",
    "hca_compress_state_block_table",
}
FWD_CSA_COMPACT_NAMES = {
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_cmp_kv_state",
    "csa_cmp_score_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_kv_state",
    "csa_inner_score_state",
    "csa_inner_compress_state_block_table",
}


_FWD_FLAT_STRIDE_CACHE: tuple[dict[str, int], dict[str, int], dict[str, int]] | None = None


def _flat_stride_maps() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return per-layer flat strides for rank-leading tensors in the golden path."""
    global _FWD_FLAT_STRIDE_CACHE
    if _FWD_FLAT_STRIDE_CACHE is None:
        sample_specs = {
            spec.name: spec
            for spec in build_tensor_specs(start_pos=0, num_tokens=T, layer_id=0)
            if isinstance(spec, TensorSpec)
        }

        def stride(name: str) -> int:
            shape = sample_specs[name].shape
            if len(shape) < 2 or int(shape[0]) != N_RANKS:
                raise ValueError(f"{name}: expected rank-leading TensorSpec, got {shape}")
            return int(shape[1])

        hca_stride = {name: stride(name) for name in FWD_HCA_COMPACT_NAMES}
        csa_stride = {name: stride(name) for name in FWD_CSA_COMPACT_NAMES}
        layer_stride = {
            name: stride(name)
            for name in HOST_TENSOR_ORDER
            if name not in {"x_hc", "x_next"}
            and name not in FWD_RANK_SLICED_CACHE_NAMES
            and name not in FWD_HCA_COMPACT_NAMES
            and name not in FWD_CSA_COMPACT_NAMES
        }
        _FWD_FLAT_STRIDE_CACHE = layer_stride, hca_stride, csa_stride
    return _FWD_FLAT_STRIDE_CACHE


def _slice_rank_flat(value: torch.Tensor, layer_index: int, stride: int) -> torch.Tensor:
    start = layer_index * stride
    return value[:, start : start + stride].clone()


def _run_jit_return_tensors(
    *,
    fn: Any,
    specs: list[TensorSpec | ScalarSpec],
    golden_fn: Any,
    compile_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    compare_fn: dict[str, Any],
    compile_only: bool,
    save_data: bool,
    kernel_only: bool = False,
    allow_mismatch: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> tuple[RunResult, dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
    """Local run_jit variant that returns runtime tensors for layer chaining."""
    from pypto.runtime import RunConfig

    runtime_cfg = dict(runtime_cfg)
    _consume_runtime_harness_keys(runtime_cfg)
    tensor_specs = [s for s in specs if isinstance(s, TensorSpec)]
    scalar_specs = [s for s in specs if isinstance(s, ScalarSpec)]
    start = time.time()
    work_dir: Path | None = None

    def fail(error: str) -> tuple[RunResult, None, None]:
        return RunResult(False, error=error, execution_time=time.time() - start, work_dir=work_dir), None, None

    with _Stage("compile"):
        dummy_args = [
            spec.value.item() if isinstance(spec, ScalarSpec)
            else torch.empty(spec.shape, dtype=spec.dtype)
            for spec in specs
        ]
        cfg = dict(compile_cfg)
        platform = runtime_cfg.get("platform")
        if platform is not None:
            cfg["platform"] = platform
        compiled = fn.compile(*dummy_args, config=RunConfig(**cfg))
        work_dir = Path(compiled.output_dir)
        print(f"[RUN] build path: {work_dir}", flush=True)
    if compile_only:
        total = time.time() - start
        print(f"[RUN] PASS ({total:.2f}s)", flush=True)
        return RunResult(True, execution_time=total, work_dir=work_dir), None, None

    try:
        with _Stage("generate inputs"):
            tensors, scalar_specs_eff, input_snapshot = _prepare_inputs(
                specs, tensor_specs, scalar_specs, None, work_dir, save_data,
            )
    except ValueError as exc:
        return fail(str(exc))

    golden_outputs = None
    if not kernel_only:
        golden_outputs = _compute_golden(
            specs, tensor_specs, scalar_specs_eff, input_snapshot, work_dir, None, golden_fn, save_data,
        )

    with _Stage("runtime"):
        if not _try_l3_dispatch(compiled, specs, tensors, scalar_specs_eff, runtime_cfg):
            _execute_via_runner(work_dir, specs, tensors, scalar_specs_eff, runtime_cfg)

    if kernel_only:
        try:
            _check_kernel_only_outputs(tensors, scalar_specs_eff)
        except AssertionError as exc:
            return fail(str(exc))
        total = time.time() - start
        print(f"[RUN] PASS kernel-only ({total:.2f}s)", flush=True)
        return RunResult(True, execution_time=total, work_dir=work_dir), tensors, None

    try:
        _validate(tensor_specs, tensors, golden_outputs, rtol, atol, compare_fn)
    except AssertionError as exc:
        if not allow_mismatch:
            return fail(str(exc))
        print("[RUN] validation mismatch ignored for chaining", flush=True)
        total = time.time() - start
        return (
            RunResult(False, error=str(exc), execution_time=total, work_dir=work_dir),
            tensors,
            golden_outputs,
        )

    total = time.time() - start
    print(f"[RUN] PASS ({total:.2f}s)", flush=True)
    return RunResult(True, execution_time=total, work_dir=work_dir), tensors, golden_outputs


def _check_kernel_only_outputs(
    tensors: dict[str, torch.Tensor],
    scalar_specs_eff: dict[str, ScalarSpec],
) -> None:
    """Cheap runtime sanity for kernel-only bring-up.

    This intentionally does not compare against golden.  It only catches the
    common broken-kernel symptoms: no active output, NaN/Inf, or an all-zero
    active ``x_out``.
    """
    if "x_out" in tensors:
        x_out = tensors["x_out"]
    elif "x_out_rank0" in tensors and "x_out_rank1" in tensors:
        x_out = torch.stack([tensors["x_out_rank0"], tensors["x_out_rank1"]], dim=0)
    else:
        return
    num_tokens = int(scalar_specs_eff["num_tokens"].value.item())
    if num_tokens <= 0:
        raise AssertionError("kernel-only sanity failed: num_tokens must be > 0")
    active = x_out[:, :num_tokens].detach().to(torch.float32)
    if active.numel() == 0:
        raise AssertionError("kernel-only sanity failed: active x_out is empty")
    if not torch.isfinite(active).all():
        raise AssertionError("kernel-only sanity failed: active x_out has NaN/Inf")
    max_abs = float(active.abs().max().item())
    mean_abs = float(active.abs().mean().item())
    nonzero = int(torch.count_nonzero(active).item())
    print(
        f"[RUN] kernel-only sanity: x_out active nonzero={nonzero}/{active.numel()} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}",
        flush=True,
    )
    if nonzero == 0 or max_abs == 0.0:
        raise AssertionError("kernel-only sanity failed: active x_out is all zero")



@pl.jit.inline
def prefill_fwd_rank_core(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[FWD_HCA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[FWD_CSA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[FWD_CSA_LAYER_COUNT * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    attn_sink: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    routed_w1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    hc_attn_fn_flat = pl.reshape(hc_attn_fn, [PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM])
    hc_attn_scale_flat = pl.reshape(hc_attn_scale, [PREFILL_FWD_ACTIVE_LAYERS * 3])
    hc_attn_base_flat = pl.reshape(hc_attn_base, [PREFILL_FWD_ACTIVE_LAYERS * MIX_HC])
    attn_norm_w_flat = pl.reshape(attn_norm_w, [PREFILL_FWD_ACTIVE_LAYERS * D])
    wq_a_flat = pl.reshape(wq_a, [PREFILL_FWD_ACTIVE_LAYERS * D, Q_LORA])
    wq_b_flat = pl.reshape(wq_b, [PREFILL_FWD_ACTIVE_LAYERS * Q_LORA, H * HEAD_DIM])
    wq_b_scale_flat = pl.reshape(wq_b_scale, [PREFILL_FWD_ACTIVE_LAYERS * H * HEAD_DIM])
    wkv_flat = pl.reshape(wkv, [PREFILL_FWD_ACTIVE_LAYERS * D, HEAD_DIM])
    gamma_cq_flat = pl.reshape(gamma_cq, [PREFILL_FWD_ACTIVE_LAYERS * Q_LORA])
    gamma_ckv_flat = pl.reshape(gamma_ckv, [PREFILL_FWD_ACTIVE_LAYERS * HEAD_DIM])
    freqs_cos_flat = pl.reshape(freqs_cos, [PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM])
    freqs_sin_flat = pl.reshape(freqs_sin, [PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM])
    hca_cmp_wkv_flat = pl.reshape(hca_cmp_wkv, [FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM])
    hca_cmp_wgate_flat = pl.reshape(hca_cmp_wgate, [FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM])
    hca_cmp_ape_flat = pl.reshape(hca_cmp_ape, [FWD_HCA_LAYER_COUNT * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM])
    hca_cmp_norm_w_flat = pl.reshape(hca_cmp_norm_w, [FWD_HCA_LAYER_COUNT * HEAD_DIM])
    hca_cmp_kv_state_flat = pl.reshape(
        hca_cmp_kv_state,
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
    )
    hca_cmp_score_state_flat = pl.reshape(
        hca_cmp_score_state,
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
    )
    hca_compress_state_block_table_flat = pl.reshape(
        hca_compress_state_block_table,
        [FWD_HCA_LAYER_COUNT * HCA_STATE_MAX_BLOCKS],
    )
    csa_cmp_wkv_flat = pl.reshape(csa_cmp_wkv, [FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM])
    csa_cmp_wgate_flat = pl.reshape(csa_cmp_wgate, [FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM])
    csa_cmp_ape_flat = pl.reshape(csa_cmp_ape, [FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM])
    csa_cmp_norm_w_flat = pl.reshape(csa_cmp_norm_w, [FWD_CSA_LAYER_COUNT * HEAD_DIM])
    csa_cmp_kv_state_flat = pl.reshape(
        csa_cmp_kv_state,
        [FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM],
    )
    csa_cmp_score_state_flat = pl.reshape(
        csa_cmp_score_state,
        [FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM],
    )
    csa_compress_state_block_table_flat = pl.reshape(
        csa_compress_state_block_table,
        [FWD_CSA_LAYER_COUNT * CSA_STATE_MAX_BLOCKS],
    )
    csa_hadamard_idx_flat = pl.reshape(csa_hadamard_idx, [FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM, IDX_HEAD_DIM])
    csa_idx_wq_b_flat = pl.reshape(csa_idx_wq_b, [FWD_CSA_LAYER_COUNT * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM])
    csa_idx_wq_b_scale_flat = pl.reshape(csa_idx_wq_b_scale, [FWD_CSA_LAYER_COUNT * IDX_N_HEADS * IDX_HEAD_DIM])
    csa_weights_proj_flat = pl.reshape(csa_weights_proj, [FWD_CSA_LAYER_COUNT * D, IDX_N_HEADS])
    csa_inner_wkv_flat = pl.reshape(csa_inner_wkv, [FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM])
    csa_inner_wgate_flat = pl.reshape(csa_inner_wgate, [FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM])
    csa_inner_ape_flat = pl.reshape(csa_inner_ape, [FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, INNER_OUT_DIM])
    csa_inner_norm_w_flat = pl.reshape(csa_inner_norm_w, [FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM])
    csa_inner_kv_state_flat = pl.reshape(
        csa_inner_kv_state,
        [FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM],
    )
    csa_inner_score_state_flat = pl.reshape(
        csa_inner_score_state,
        [FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM],
    )
    csa_inner_compress_state_block_table_flat = pl.reshape(
        csa_inner_compress_state_block_table,
        [FWD_CSA_LAYER_COUNT * INNER_STATE_MAX_BLOCKS],
    )
    kv_cache_rows = pl.reshape(kv_cache, [PREFILL_FWD_ACTIVE_LAYERS * FWD_KV_CACHE_ROWS, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [PREFILL_FWD_ACTIVE_LAYERS * SPARSE_ORI_MAX_BLOCKS])
    ori_slot_mapping_flat = pl.reshape(ori_slot_mapping, [PREFILL_FWD_ACTIVE_LAYERS * T])
    cmp_kv_rows = pl.reshape(cmp_kv, [PREFILL_FWD_ACTIVE_LAYERS * FWD_CMP_CACHE_ROWS, HEAD_DIM])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [PREFILL_FWD_ACTIVE_LAYERS * SPARSE_CMP_MAX_BLOCKS])
    cmp_sparse_indices_flat = pl.reshape(cmp_sparse_indices, [PREFILL_FWD_ACTIVE_LAYERS * T, SPARSE_TOPK])
    cmp_sparse_lens_flat = pl.reshape(cmp_sparse_lens, [PREFILL_FWD_ACTIVE_LAYERS * T])
    idx_kv_cache_rows = pl.reshape(idx_kv_cache, [PREFILL_FWD_ACTIVE_LAYERS * FWD_CMP_CACHE_ROWS, IDX_HEAD_DIM])
    idx_block_table_flat = pl.reshape(idx_block_table, [PREFILL_FWD_ACTIVE_LAYERS * IDX_CACHE_MAX_BLOCKS])
    position_ids_flat = pl.reshape(position_ids, [PREFILL_FWD_ACTIVE_LAYERS * T])
    hca_cmp_slot_mapping_flat = pl.reshape(hca_cmp_slot_mapping, [PREFILL_FWD_ACTIVE_LAYERS * T])
    hca_state_slot_mapping_flat = pl.reshape(hca_state_slot_mapping, [PREFILL_FWD_ACTIVE_LAYERS * T])
    csa_cmp_slot_mapping_flat = pl.reshape(csa_cmp_slot_mapping, [PREFILL_FWD_ACTIVE_LAYERS * T])
    csa_idx_slot_mapping_flat = pl.reshape(csa_idx_slot_mapping, [PREFILL_FWD_ACTIVE_LAYERS * T])
    csa_state_slot_mapping_flat = pl.reshape(csa_state_slot_mapping, [PREFILL_FWD_ACTIVE_LAYERS * T])
    csa_inner_state_slot_mapping_flat = pl.reshape(csa_inner_state_slot_mapping, [PREFILL_FWD_ACTIVE_LAYERS * T])
    attn_sink_flat = pl.reshape(attn_sink, [PREFILL_FWD_ACTIVE_LAYERS * H])
    wo_a_flat = pl.reshape(wo_a, [PREFILL_FWD_ACTIVE_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN])
    wo_b_flat = pl.reshape(wo_b, [PREFILL_FWD_ACTIVE_LAYERS * D, O_GROUPS * O_LORA])
    wo_b_scale_flat = pl.reshape(wo_b_scale, [PREFILL_FWD_ACTIVE_LAYERS * D])
    hc_ffn_fn_flat = pl.reshape(hc_ffn_fn, [PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM])
    hc_ffn_scale_flat = pl.reshape(hc_ffn_scale, [PREFILL_FWD_ACTIVE_LAYERS * 3])
    hc_ffn_base_flat = pl.reshape(hc_ffn_base, [PREFILL_FWD_ACTIVE_LAYERS * MIX_HC])
    norm_w_flat = pl.reshape(norm_w, [PREFILL_FWD_ACTIVE_LAYERS * D])
    gate_w_flat = pl.reshape(gate_w, [PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL, D])
    gate_bias_flat = pl.reshape(gate_bias, [PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL])
    tid2eid_flat = pl.reshape(tid2eid, [PREFILL_FWD_ACTIVE_LAYERS * VOCAB, TOPK])
    input_ids_flat = pl.reshape(input_ids, [PREFILL_FWD_ACTIVE_LAYERS * T])
    routed_w1_flat = pl.reshape(routed_w1, [PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D])
    routed_w1_scale_flat = pl.reshape(routed_w1_scale, [PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER])
    routed_w3_flat = pl.reshape(routed_w3, [PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D])
    routed_w3_scale_flat = pl.reshape(routed_w3_scale, [PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER])
    routed_w2_flat = pl.reshape(routed_w2, [PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D, MOE_INTER])
    routed_w2_scale_flat = pl.reshape(routed_w2_scale, [PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D])
    shared_w1_flat = pl.reshape(shared_w1, [PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D])
    shared_w1_scale_flat = pl.reshape(shared_w1_scale, [PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER])
    shared_w3_flat = pl.reshape(shared_w3, [PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D])
    shared_w3_scale_flat = pl.reshape(shared_w3_scale, [PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER])
    shared_w2_flat = pl.reshape(shared_w2, [PREFILL_FWD_ACTIVE_LAYERS * D, MOE_INTER])
    shared_w2_scale_flat = pl.reshape(shared_w2_scale, [PREFILL_FWD_ACTIVE_LAYERS * D])

    hidden_a = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    hidden_b = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    hidden_a_flat = pl.reshape(hidden_a, [FWD_X_ROWS, D])
    hidden_b_flat = pl.reshape(hidden_b, [FWD_X_ROWS, D])
    x_out_flat = pl.reshape(x_out, [FWD_X_ROWS, D])
    if PREFILL_FWD_ACTIVE_LAYERS > 0:
        for layer_i in pl.range(PREFILL_FWD_ACTIVE_LAYERS):
            hca_compact_i = 0
            csa_compact_i = 0
            if layer_i >= 3:
                hca_compact_i = (layer_i - 3) // 2
            if layer_i >= 2:
                csa_compact_i = (layer_i - 2) // 2
            layer_src_buf = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
            layer_src_flat = pl.reshape(layer_src_buf, [T * HC_MULT, D])
            if layer_i == 0:
                x_hc_flat = pl.reshape(x_hc, [T * HC_MULT, D])
                for x_row0 in pl.parallel(0, FWD_X_ROWS, FWD_X_COPY_ROWS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_copy_x_hc_input"):
                        layer_src_flat[x_row0 : x_row0 + FWD_X_COPY_ROWS, 0:D] = x_hc_flat[
                            x_row0 : x_row0 + FWD_X_COPY_ROWS,
                            0:D,
                        ]
            elif layer_i % 2 == 1:
                for x_row0 in pl.parallel(0, FWD_X_ROWS, FWD_X_COPY_ROWS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_copy_hidden_a_input"):
                        layer_src_flat[x_row0 : x_row0 + FWD_X_COPY_ROWS, 0:D] = hidden_a_flat[
                            x_row0 : x_row0 + FWD_X_COPY_ROWS,
                            0:D,
                        ]
            else:
                for x_row0 in pl.parallel(0, FWD_X_ROWS, FWD_X_COPY_ROWS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_copy_hidden_b_input"):
                        layer_src_flat[x_row0 : x_row0 + FWD_X_COPY_ROWS, 0:D] = hidden_b_flat[
                            x_row0 : x_row0 + FWD_X_COPY_ROWS,
                            0:D,
                        ]
            hc_attn_fn_i: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn_flat, [MIX_HC, HC_DIM], [layer_i * MIX_HC, 0])
            hc_attn_scale_i: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale_flat, [3], [layer_i * 3])
            hc_attn_base_i: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base_flat, [MIX_HC], [layer_i * MIX_HC])
            attn_norm_w_i: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w_flat, [D], [layer_i * D])
            wq_a_i: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a_flat, [D, Q_LORA], [layer_i * D, 0])
            wq_b_i: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b_flat, [Q_LORA, H * HEAD_DIM], [layer_i * Q_LORA, 0])
            wq_b_scale_i: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale_flat, [H * HEAD_DIM], [layer_i * H * HEAD_DIM])
            wkv_i: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv_flat, [D, HEAD_DIM], [layer_i * D, 0])
            gamma_cq_i: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq_flat, [Q_LORA], [layer_i * Q_LORA])
            gamma_ckv_i: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv_flat, [HEAD_DIM], [layer_i * HEAD_DIM])
            freqs_cos_i: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(freqs_cos_flat, [MAX_SEQ_LEN, ROPE_HEAD_DIM], [layer_i * MAX_SEQ_LEN, 0])
            freqs_sin_i: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(freqs_sin_flat, [MAX_SEQ_LEN, ROPE_HEAD_DIM], [layer_i * MAX_SEQ_LEN, 0])
            hca_cmp_wkv_i: pl.Tensor[[D, HCA_MAIN_OUT_DIM], pl.BF16] = pl.slice(hca_cmp_wkv_flat, [D, HCA_MAIN_OUT_DIM], [hca_compact_i * D, 0])
            hca_cmp_wgate_i: pl.Tensor[[D, HCA_MAIN_OUT_DIM], pl.BF16] = pl.slice(hca_cmp_wgate_flat, [D, HCA_MAIN_OUT_DIM], [hca_compact_i * D, 0])
            hca_cmp_ape_i: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(hca_cmp_ape_flat, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [hca_compact_i * HCA_COMPRESS_RATIO, 0])
            hca_cmp_norm_w_i: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(hca_cmp_norm_w_flat, [HEAD_DIM], [hca_compact_i * HEAD_DIM])
            hca_cmp_kv_state_i: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(hca_cmp_kv_state_flat, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], [hca_compact_i * HCA_STATE_BLOCK_NUM, 0, 0])
            hca_cmp_score_state_i: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(hca_cmp_score_state_flat, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], [hca_compact_i * HCA_STATE_BLOCK_NUM, 0, 0])
            hca_compress_state_block_table_i: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32] = pl.slice(hca_compress_state_block_table_flat, [HCA_STATE_MAX_BLOCKS], [hca_compact_i * HCA_STATE_MAX_BLOCKS])
            csa_cmp_wkv_i: pl.Tensor[[D, CSA_MAIN_OUT_DIM], pl.BF16] = pl.slice(csa_cmp_wkv_flat, [D, CSA_MAIN_OUT_DIM], [csa_compact_i * D, 0])
            csa_cmp_wgate_i: pl.Tensor[[D, CSA_MAIN_OUT_DIM], pl.BF16] = pl.slice(csa_cmp_wgate_flat, [D, CSA_MAIN_OUT_DIM], [csa_compact_i * D, 0])
            csa_cmp_ape_i: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_ape_flat, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [csa_compact_i * CSA_COMPRESS_RATIO, 0])
            csa_cmp_norm_w_i: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(csa_cmp_norm_w_flat, [HEAD_DIM], [csa_compact_i * HEAD_DIM])
            csa_cmp_kv_state_i: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_kv_state_flat, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], [csa_compact_i * CSA_STATE_BLOCK_NUM, 0, 0])
            csa_cmp_score_state_i: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_score_state_flat, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], [csa_compact_i * CSA_STATE_BLOCK_NUM, 0, 0])
            csa_compress_state_block_table_i: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32] = pl.slice(csa_compress_state_block_table_flat, [CSA_STATE_MAX_BLOCKS], [csa_compact_i * CSA_STATE_MAX_BLOCKS])
            csa_hadamard_idx_i: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_hadamard_idx_flat, [IDX_HEAD_DIM, IDX_HEAD_DIM], [csa_compact_i * IDX_HEAD_DIM, 0])
            csa_idx_wq_b_i: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8] = pl.slice(csa_idx_wq_b_flat, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [csa_compact_i * Q_LORA, 0])
            csa_idx_wq_b_scale_i: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = pl.slice(csa_idx_wq_b_scale_flat, [IDX_N_HEADS * IDX_HEAD_DIM], [csa_compact_i * IDX_N_HEADS * IDX_HEAD_DIM])
            csa_weights_proj_i: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(csa_weights_proj_flat, [D, IDX_N_HEADS], [csa_compact_i * D, 0])
            csa_inner_wkv_i: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16] = pl.slice(csa_inner_wkv_flat, [D, INNER_OUT_DIM], [csa_compact_i * D, 0])
            csa_inner_wgate_i: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16] = pl.slice(csa_inner_wgate_flat, [D, INNER_OUT_DIM], [csa_compact_i * D, 0])
            csa_inner_ape_i: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_ape_flat, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [csa_compact_i * CSA_COMPRESS_RATIO, 0])
            csa_inner_norm_w_i: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_inner_norm_w_flat, [IDX_HEAD_DIM], [csa_compact_i * IDX_HEAD_DIM])
            csa_inner_kv_state_i: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_kv_state_flat, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], [csa_compact_i * INNER_STATE_BLOCK_NUM, 0, 0])
            csa_inner_score_state_i: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_score_state_flat, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], [csa_compact_i * INNER_STATE_BLOCK_NUM, 0, 0])
            csa_inner_compress_state_block_table_i: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32] = pl.slice(csa_inner_compress_state_block_table_flat, [INNER_STATE_MAX_BLOCKS], [csa_compact_i * INNER_STATE_MAX_BLOCKS])
            ori_block_table_i: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32] = pl.slice(ori_block_table_flat, [SPARSE_ORI_MAX_BLOCKS], [layer_i * SPARSE_ORI_MAX_BLOCKS])
            ori_slot_mapping_i: pl.Tensor[[T], pl.INT64] = pl.slice(ori_slot_mapping_flat, [T], [layer_i * T])
            cmp_block_table_i: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32] = pl.slice(cmp_block_table_flat, [SPARSE_CMP_MAX_BLOCKS], [layer_i * SPARSE_CMP_MAX_BLOCKS])
            cmp_sparse_indices_i: pl.Tensor[[T, SPARSE_TOPK], pl.INT32] = pl.slice(cmp_sparse_indices_flat, [T, SPARSE_TOPK], [layer_i * T, 0])
            cmp_sparse_lens_i: pl.Tensor[[T], pl.INT32] = pl.slice(cmp_sparse_lens_flat, [T], [layer_i * T])
            idx_block_table_i: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32] = pl.slice(idx_block_table_flat, [IDX_CACHE_MAX_BLOCKS], [layer_i * IDX_CACHE_MAX_BLOCKS])
            position_ids_i: pl.Tensor[[T], pl.INT32] = pl.slice(position_ids_flat, [T], [layer_i * T])
            hca_cmp_slot_mapping_i: pl.Tensor[[T], pl.INT64] = pl.slice(hca_cmp_slot_mapping_flat, [T], [layer_i * T])
            hca_state_slot_mapping_i: pl.Tensor[[T], pl.INT64] = pl.slice(hca_state_slot_mapping_flat, [T], [layer_i * T])
            csa_cmp_slot_mapping_i: pl.Tensor[[T], pl.INT64] = pl.slice(csa_cmp_slot_mapping_flat, [T], [layer_i * T])
            csa_idx_slot_mapping_i: pl.Tensor[[T], pl.INT64] = pl.slice(csa_idx_slot_mapping_flat, [T], [layer_i * T])
            csa_state_slot_mapping_i: pl.Tensor[[T], pl.INT64] = pl.slice(csa_state_slot_mapping_flat, [T], [layer_i * T])
            csa_inner_state_slot_mapping_i: pl.Tensor[[T], pl.INT64] = pl.slice(csa_inner_state_slot_mapping_flat, [T], [layer_i * T])
            attn_sink_i: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink_flat, [H], [layer_i * H])
            wo_a_i: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a_flat, [O_GROUPS, O_LORA, O_GROUP_IN], [layer_i * O_GROUPS, 0, 0])
            wo_b_i: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b_flat, [D, O_GROUPS * O_LORA], [layer_i * D, 0])
            wo_b_scale_i: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale_flat, [D], [layer_i * D])
            hc_ffn_fn_i: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn_flat, [MIX_HC, HC_DIM], [layer_i * MIX_HC, 0])
            hc_ffn_scale_i: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale_flat, [3], [layer_i * 3])
            hc_ffn_base_i: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base_flat, [MIX_HC], [layer_i * MIX_HC])
            norm_w_i: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w_flat, [D], [layer_i * D])
            gate_w_i: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w_flat, [N_EXPERTS_GLOBAL, D], [layer_i * N_EXPERTS_GLOBAL, 0])
            gate_bias_i: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias_flat, [N_EXPERTS_GLOBAL], [layer_i * N_EXPERTS_GLOBAL])
            tid2eid_i: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid_flat, [VOCAB, TOPK], [layer_i * VOCAB, 0])
            input_ids_i: pl.Tensor[[T], pl.INT64] = pl.slice(input_ids_flat, [T], [layer_i * T])
            routed_w1_i: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1_flat, [N_LOCAL, MOE_INTER, D], [layer_i * N_LOCAL, 0, 0])
            routed_w1_scale_i: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale_flat, [N_LOCAL, MOE_INTER], [layer_i * N_LOCAL, 0])
            routed_w3_i: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3_flat, [N_LOCAL, MOE_INTER, D], [layer_i * N_LOCAL, 0, 0])
            routed_w3_scale_i: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale_flat, [N_LOCAL, MOE_INTER], [layer_i * N_LOCAL, 0])
            routed_w2_i: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2_flat, [N_LOCAL, D, MOE_INTER], [layer_i * N_LOCAL, 0, 0])
            routed_w2_scale_i: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale_flat, [N_LOCAL, D], [layer_i * N_LOCAL, 0])
            shared_w1_i: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1_flat, [MOE_INTER, D], [layer_i * MOE_INTER, 0])
            shared_w1_scale_i: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale_flat, [MOE_INTER], [layer_i * MOE_INTER])
            shared_w3_i: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3_flat, [MOE_INTER, D], [layer_i * MOE_INTER, 0])
            shared_w3_scale_i: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale_flat, [MOE_INTER], [layer_i * MOE_INTER])
            shared_w2_i: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2_flat, [D, MOE_INTER], [layer_i * D, 0])
            shared_w2_scale_i: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale_flat, [D], [layer_i * D])
            kv_cache_i = pl.create_tensor([CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
            cmp_kv_i = pl.create_tensor([CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
            idx_kv_cache_i = pl.create_tensor([CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], dtype=pl.BF16)
            kv_cache_i_rows = pl.reshape(kv_cache_i, [FWD_KV_CACHE_ROWS, HEAD_DIM])
            cmp_kv_i_rows = pl.reshape(cmp_kv_i, [FWD_CMP_CACHE_ROWS, HEAD_DIM])
            idx_kv_cache_i_rows = pl.reshape(idx_kv_cache_i, [FWD_CMP_CACHE_ROWS, IDX_HEAD_DIM])
            for cache_row0 in pl.parallel(0, FWD_KV_CACHE_ROWS, FWD_CACHE_COPY_ROWS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_load_kv_cache"):
                    kv_cache_i_rows[cache_row0 : cache_row0 + FWD_CACHE_COPY_ROWS, 0:HEAD_DIM] = kv_cache_rows[
                        layer_i * FWD_KV_CACHE_ROWS + cache_row0 : layer_i * FWD_KV_CACHE_ROWS + cache_row0 + FWD_CACHE_COPY_ROWS,
                        0:HEAD_DIM,
                    ]
            for cache_row0 in pl.parallel(0, FWD_CMP_CACHE_ROWS, FWD_CACHE_COPY_ROWS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_load_cmp_cache"):
                    cmp_kv_i_rows[cache_row0 : cache_row0 + FWD_CACHE_COPY_ROWS, 0:HEAD_DIM] = cmp_kv_rows[
                        layer_i * FWD_CMP_CACHE_ROWS + cache_row0 : layer_i * FWD_CMP_CACHE_ROWS + cache_row0 + FWD_CACHE_COPY_ROWS,
                        0:HEAD_DIM,
                    ]
                    idx_kv_cache_i_rows[cache_row0 : cache_row0 + FWD_CACHE_COPY_ROWS, 0:IDX_HEAD_DIM] = idx_kv_cache_rows[
                        layer_i * FWD_CMP_CACHE_ROWS + cache_row0 : layer_i * FWD_CMP_CACHE_ROWS + cache_row0 + FWD_CACHE_COPY_ROWS,
                        0:IDX_HEAD_DIM,
                    ]
            if layer_i == PREFILL_FWD_ACTIVE_LAYERS - 1:
                layer_out = prefill_layer_core(
                    layer_src_buf, hc_attn_fn_i, hc_attn_scale_i, hc_attn_base_i, attn_norm_w_i,
                    wq_a_i, wq_b_i, wq_b_scale_i, wkv_i, gamma_cq_i, gamma_ckv_i,
                    freqs_cos_i, freqs_sin_i, hca_cmp_wkv_i, hca_cmp_wgate_i, hca_cmp_ape_i,
                    hca_cmp_norm_w_i, hca_cmp_kv_state_i, hca_cmp_score_state_i,
                    hca_compress_state_block_table_i, csa_cmp_wkv_i, csa_cmp_wgate_i,
                    csa_cmp_ape_i, csa_cmp_norm_w_i, csa_cmp_kv_state_i, csa_cmp_score_state_i,
                    csa_compress_state_block_table_i, csa_hadamard_idx_i, csa_idx_wq_b_i,
                    csa_idx_wq_b_scale_i, csa_weights_proj_i, csa_inner_wkv_i,
                    csa_inner_wgate_i, csa_inner_ape_i, csa_inner_norm_w_i, csa_inner_kv_state_i,
                    csa_inner_score_state_i, csa_inner_compress_state_block_table_i, kv_cache_i,
                    ori_block_table_i, ori_slot_mapping_i, cmp_kv_i, cmp_block_table_i,
                    cmp_sparse_indices_i, cmp_sparse_lens_i, idx_kv_cache_i, idx_block_table_i,
                    position_ids_i, hca_cmp_slot_mapping_i, hca_state_slot_mapping_i,
                    csa_cmp_slot_mapping_i, csa_idx_slot_mapping_i, csa_state_slot_mapping_i,
                    csa_inner_state_slot_mapping_i, attn_sink_i, wo_a_i, wo_b_i, wo_b_scale_i,
                    hc_ffn_fn_i, hc_ffn_scale_i, hc_ffn_base_i, norm_w_i, gate_w_i, gate_bias_i,
                    tid2eid_i, input_ids_i, routed_w1_i, routed_w1_scale_i, routed_w3_i,
                    routed_w3_scale_i, routed_w2_i, routed_w2_scale_i, shared_w1_i,
                    shared_w1_scale_i, shared_w3_i, shared_w3_scale_i, shared_w2_i,
                    shared_w2_scale_i,
                    x_out,
                    pub_counts, count_done, data_done,
                    recv_x, recv_scale, recv_w, recv_r_route,
                    routed_y_buf, combine_done,
                    num_tokens, pl.cast(layer_i, pl.INT32), my_rank, pl.cast(layer_i + 1, pl.INT32),
                )
            else:
                layer_out_buf = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
                layer_out = prefill_layer_core(
                    layer_src_buf, hc_attn_fn_i, hc_attn_scale_i, hc_attn_base_i, attn_norm_w_i,
                    wq_a_i, wq_b_i, wq_b_scale_i, wkv_i, gamma_cq_i, gamma_ckv_i,
                    freqs_cos_i, freqs_sin_i, hca_cmp_wkv_i, hca_cmp_wgate_i, hca_cmp_ape_i,
                    hca_cmp_norm_w_i, hca_cmp_kv_state_i, hca_cmp_score_state_i,
                    hca_compress_state_block_table_i, csa_cmp_wkv_i, csa_cmp_wgate_i,
                    csa_cmp_ape_i, csa_cmp_norm_w_i, csa_cmp_kv_state_i, csa_cmp_score_state_i,
                    csa_compress_state_block_table_i, csa_hadamard_idx_i, csa_idx_wq_b_i,
                    csa_idx_wq_b_scale_i, csa_weights_proj_i, csa_inner_wkv_i,
                    csa_inner_wgate_i, csa_inner_ape_i, csa_inner_norm_w_i, csa_inner_kv_state_i,
                    csa_inner_score_state_i, csa_inner_compress_state_block_table_i, kv_cache_i,
                    ori_block_table_i, ori_slot_mapping_i, cmp_kv_i, cmp_block_table_i,
                    cmp_sparse_indices_i, cmp_sparse_lens_i, idx_kv_cache_i, idx_block_table_i,
                    position_ids_i, hca_cmp_slot_mapping_i, hca_state_slot_mapping_i,
                    csa_cmp_slot_mapping_i, csa_idx_slot_mapping_i, csa_state_slot_mapping_i,
                    csa_inner_state_slot_mapping_i, attn_sink_i, wo_a_i, wo_b_i, wo_b_scale_i,
                    hc_ffn_fn_i, hc_ffn_scale_i, hc_ffn_base_i, norm_w_i, gate_w_i, gate_bias_i,
                    tid2eid_i, input_ids_i, routed_w1_i, routed_w1_scale_i, routed_w3_i,
                    routed_w3_scale_i, routed_w2_i, routed_w2_scale_i, shared_w1_i,
                    shared_w1_scale_i, shared_w3_i, shared_w3_scale_i, shared_w2_i,
                    shared_w2_scale_i,
                    layer_out_buf,
                    pub_counts, count_done, data_done,
                    recv_x, recv_scale, recv_w, recv_r_route,
                    routed_y_buf, combine_done,
                    num_tokens, pl.cast(layer_i, pl.INT32), my_rank, pl.cast(layer_i + 1, pl.INT32),
                )
            for cache_row0 in pl.parallel(0, FWD_KV_CACHE_ROWS, FWD_CACHE_COPY_ROWS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_copy_kv_cache"):
                    kv_cache_rows[
                        layer_i * FWD_KV_CACHE_ROWS + cache_row0 : layer_i * FWD_KV_CACHE_ROWS + cache_row0 + FWD_CACHE_COPY_ROWS,
                        0:HEAD_DIM,
                    ] = kv_cache_i_rows[cache_row0 : cache_row0 + FWD_CACHE_COPY_ROWS, 0:HEAD_DIM]
            for cache_row0 in pl.parallel(0, FWD_CMP_CACHE_ROWS, FWD_CACHE_COPY_ROWS):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_copy_cmp_cache"):
                    cmp_kv_rows[
                        layer_i * FWD_CMP_CACHE_ROWS + cache_row0 : layer_i * FWD_CMP_CACHE_ROWS + cache_row0 + FWD_CACHE_COPY_ROWS,
                        0:HEAD_DIM,
                    ] = cmp_kv_i_rows[cache_row0 : cache_row0 + FWD_CACHE_COPY_ROWS, 0:HEAD_DIM]
                    idx_kv_cache_rows[
                        layer_i * FWD_CMP_CACHE_ROWS + cache_row0 : layer_i * FWD_CMP_CACHE_ROWS + cache_row0 + FWD_CACHE_COPY_ROWS,
                        0:IDX_HEAD_DIM,
                    ] = idx_kv_cache_i_rows[cache_row0 : cache_row0 + FWD_CACHE_COPY_ROWS, 0:IDX_HEAD_DIM]
            layer_out_flat = pl.reshape(layer_out, [FWD_X_ROWS, D])
            if layer_i == PREFILL_FWD_ACTIVE_LAYERS - 1:
                for x_row0 in pl.parallel(0, FWD_X_ROWS, FWD_X_COPY_ROWS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_copy_x_out"):
                        x_out_flat[x_row0 : x_row0 + FWD_X_COPY_ROWS, 0:D] = layer_out_flat[
                            x_row0 : x_row0 + FWD_X_COPY_ROWS,
                            0:D,
                        ]
            elif layer_i % 2 == 0:
                for x_row0 in pl.parallel(0, FWD_X_ROWS, FWD_X_COPY_ROWS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_copy_hidden_a"):
                        hidden_a_flat[x_row0 : x_row0 + FWD_X_COPY_ROWS, 0:D] = layer_out_flat[
                            x_row0 : x_row0 + FWD_X_COPY_ROWS,
                            0:D,
                        ]
            else:
                for x_row0 in pl.parallel(0, FWD_X_ROWS, FWD_X_COPY_ROWS):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_copy_hidden_b"):
                        hidden_b_flat[x_row0 : x_row0 + FWD_X_COPY_ROWS, 0:D] = layer_out_flat[
                            x_row0 : x_row0 + FWD_X_COPY_ROWS,
                            0:D,
                        ]
    return x_out





@pl.jit
def prefill_fwd_rank0(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[FWD_HCA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[FWD_CSA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[FWD_CSA_LAYER_COUNT * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    attn_sink: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    routed_w1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    return prefill_fwd_rank_core(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        hca_cmp_wkv,
        hca_cmp_wgate,
        hca_cmp_ape,
        hca_cmp_norm_w,
        hca_cmp_kv_state,
        hca_cmp_score_state,
        hca_compress_state_block_table,
        csa_cmp_wkv,
        csa_cmp_wgate,
        csa_cmp_ape,
        csa_cmp_norm_w,
        csa_cmp_kv_state,
        csa_cmp_score_state,
        csa_compress_state_block_table,
        csa_hadamard_idx,
        csa_idx_wq_b,
        csa_idx_wq_b_scale,
        csa_weights_proj,
        csa_inner_wkv,
        csa_inner_wgate,
        csa_inner_ape,
        csa_inner_norm_w,
        csa_inner_kv_state,
        csa_inner_score_state,
        csa_inner_compress_state_block_table,
        kv_cache,
        ori_block_table,
        ori_slot_mapping,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        cmp_sparse_lens,
        idx_kv_cache,
        idx_block_table,
        position_ids,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        x_out,
        pub_counts,
        count_done,
        data_done,
        recv_x,
        recv_scale,
        recv_w,
        recv_r_route,
        routed_y_buf,
        combine_done,
        num_tokens,
        pl.const(0, pl.INT32),
    )


@pl.jit
def prefill_fwd_rank1(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[FWD_HCA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[FWD_CSA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[FWD_CSA_LAYER_COUNT * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    attn_sink: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    routed_w1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    pub_counts: pld.DistributedTensor[[N_RANKS * N_RANKS, N_LOCAL], pl.INT32],
    count_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_w: pld.DistributedTensor[[N_LOCAL * RECV_MAX, W_PAD], pl.FP32],
    recv_r_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    return prefill_fwd_rank_core(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        hca_cmp_wkv,
        hca_cmp_wgate,
        hca_cmp_ape,
        hca_cmp_norm_w,
        hca_cmp_kv_state,
        hca_cmp_score_state,
        hca_compress_state_block_table,
        csa_cmp_wkv,
        csa_cmp_wgate,
        csa_cmp_ape,
        csa_cmp_norm_w,
        csa_cmp_kv_state,
        csa_cmp_score_state,
        csa_compress_state_block_table,
        csa_hadamard_idx,
        csa_idx_wq_b,
        csa_idx_wq_b_scale,
        csa_weights_proj,
        csa_inner_wkv,
        csa_inner_wgate,
        csa_inner_ape,
        csa_inner_norm_w,
        csa_inner_kv_state,
        csa_inner_score_state,
        csa_inner_compress_state_block_table,
        kv_cache,
        ori_block_table,
        ori_slot_mapping,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        cmp_sparse_lens,
        idx_kv_cache,
        idx_block_table,
        position_ids,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        input_ids,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w3,
        shared_w3_scale,
        shared_w2,
        shared_w2_scale,
        x_out,
        pub_counts,
        count_done,
        data_done,
        recv_x,
        recv_scale,
        recv_w,
        recv_r_route,
        routed_y_buf,
        combine_done,
        num_tokens,
        pl.const(1, pl.INT32),
    )


@pl.jit.host
def l3_prefill_fwd_rank_specific(
    x_hc_rank0: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_attn_base_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    wq_a_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, Q_LORA], pl.BF16],
    wq_b_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv_rank0: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate_rank0: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape_rank0: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w_rank0: pl.Tensor[[FWD_HCA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    hca_cmp_kv_state_rank0: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state_rank0: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table_rank0: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    csa_cmp_kv_state_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table_rank0: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache_rank0: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    cmp_kv_rank0: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    idx_kv_cache_rank0: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    hca_cmp_slot_mapping_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    hca_state_slot_mapping_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_cmp_slot_mapping_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_idx_slot_mapping_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_state_slot_mapping_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_inner_state_slot_mapping_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    attn_sink_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H], pl.FP32],
    wo_a_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    hc_ffn_fn_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_ffn_base_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    norm_w_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    gate_w_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    routed_w1_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w3_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w2_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale_rank0: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    x_out_rank0: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    x_hc_rank1: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_attn_base_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    wq_a_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, Q_LORA], pl.BF16],
    wq_b_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv_rank1: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_wgate_rank1: pl.Tensor[[FWD_HCA_LAYER_COUNT * D, HCA_MAIN_OUT_DIM], pl.BF16],
    hca_cmp_ape_rank1: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w_rank1: pl.Tensor[[FWD_HCA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    hca_cmp_kv_state_rank1: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_score_state_rank1: pl.Tensor[
        [FWD_HCA_LAYER_COUNT * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_compress_state_block_table_rank1: pl.Tensor[[FWD_HCA_LAYER_COUNT * HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_wgate_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, CSA_MAIN_OUT_DIM], pl.BF16],
    csa_cmp_ape_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * HEAD_DIM], pl.BF16],
    csa_cmp_kv_state_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_compress_state_block_table_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_wgate_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * D, INNER_OUT_DIM], pl.BF16],
    csa_inner_ape_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_compress_state_block_table_rank1: pl.Tensor[[FWD_CSA_LAYER_COUNT * INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache_rank1: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    cmp_kv_rank1: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    idx_kv_cache_rank1: pl.Out[pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT32],
    hca_cmp_slot_mapping_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    hca_state_slot_mapping_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_cmp_slot_mapping_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_idx_slot_mapping_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_state_slot_mapping_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    csa_inner_state_slot_mapping_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    attn_sink_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * H], pl.FP32],
    wo_a_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    hc_ffn_fn_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * 3], pl.FP32],
    hc_ffn_base_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MIX_HC], pl.FP32],
    norm_w_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.BF16],
    gate_w_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * T], pl.INT64],
    routed_w1_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w3_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * MOE_INTER], pl.FP32],
    shared_w2_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale_rank1: pl.Tensor[[PREFILL_FWD_ACTIVE_LAYERS * D], pl.FP32],
    x_out_rank1: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * N_LOCAL * 4)
    count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    data_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_scale_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_w_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_r_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)

    pub_counts0 = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
    count_done0 = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
    data_done0 = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
    recv_x0 = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_scale0 = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
    recv_w0 = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
    recv_r_route0 = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    routed_y_buf0 = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
    combine_done0 = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)

    prefill_fwd_rank0(
        x_hc_rank0,
        hc_attn_fn_rank0,
        hc_attn_scale_rank0,
        hc_attn_base_rank0,
        attn_norm_w_rank0,
        wq_a_rank0,
        wq_b_rank0,
        wq_b_scale_rank0,
        wkv_rank0,
        gamma_cq_rank0,
        gamma_ckv_rank0,
        freqs_cos_rank0,
        freqs_sin_rank0,
        hca_cmp_wkv_rank0,
        hca_cmp_wgate_rank0,
        hca_cmp_ape_rank0,
        hca_cmp_norm_w_rank0,
        hca_cmp_kv_state_rank0,
        hca_cmp_score_state_rank0,
        hca_compress_state_block_table_rank0,
        csa_cmp_wkv_rank0,
        csa_cmp_wgate_rank0,
        csa_cmp_ape_rank0,
        csa_cmp_norm_w_rank0,
        csa_cmp_kv_state_rank0,
        csa_cmp_score_state_rank0,
        csa_compress_state_block_table_rank0,
        csa_hadamard_idx_rank0,
        csa_idx_wq_b_rank0,
        csa_idx_wq_b_scale_rank0,
        csa_weights_proj_rank0,
        csa_inner_wkv_rank0,
        csa_inner_wgate_rank0,
        csa_inner_ape_rank0,
        csa_inner_norm_w_rank0,
        csa_inner_kv_state_rank0,
        csa_inner_score_state_rank0,
        csa_inner_compress_state_block_table_rank0,
        kv_cache_rank0,
        ori_block_table_rank0,
        ori_slot_mapping_rank0,
        cmp_kv_rank0,
        cmp_block_table_rank0,
        cmp_sparse_indices_rank0,
        cmp_sparse_lens_rank0,
        idx_kv_cache_rank0,
        idx_block_table_rank0,
        position_ids_rank0,
        hca_cmp_slot_mapping_rank0,
        hca_state_slot_mapping_rank0,
        csa_cmp_slot_mapping_rank0,
        csa_idx_slot_mapping_rank0,
        csa_state_slot_mapping_rank0,
        csa_inner_state_slot_mapping_rank0,
        attn_sink_rank0,
        wo_a_rank0,
        wo_b_rank0,
        wo_b_scale_rank0,
        hc_ffn_fn_rank0,
        hc_ffn_scale_rank0,
        hc_ffn_base_rank0,
        norm_w_rank0,
        gate_w_rank0,
        gate_bias_rank0,
        tid2eid_rank0,
        input_ids_rank0,
        routed_w1_rank0,
        routed_w1_scale_rank0,
        routed_w3_rank0,
        routed_w3_scale_rank0,
        routed_w2_rank0,
        routed_w2_scale_rank0,
        shared_w1_rank0,
        shared_w1_scale_rank0,
        shared_w3_rank0,
        shared_w3_scale_rank0,
        shared_w2_rank0,
        shared_w2_scale_rank0,
        x_out_rank0,
        pub_counts0,
        count_done0,
        data_done0,
        recv_x0,
        recv_scale0,
        recv_w0,
        recv_r_route0,
        routed_y_buf0,
        combine_done0,
        num_tokens,
        device=0,
    )

    pub_counts1 = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
    count_done1 = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
    data_done1 = pld.window(data_done_buf, [N_RANKS, 1], dtype=pl.INT32)
    recv_x1 = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_scale1 = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
    recv_w1 = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
    recv_r_route1 = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    routed_y_buf1 = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
    combine_done1 = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)

    prefill_fwd_rank1(
        x_hc_rank1,
        hc_attn_fn_rank1,
        hc_attn_scale_rank1,
        hc_attn_base_rank1,
        attn_norm_w_rank1,
        wq_a_rank1,
        wq_b_rank1,
        wq_b_scale_rank1,
        wkv_rank1,
        gamma_cq_rank1,
        gamma_ckv_rank1,
        freqs_cos_rank1,
        freqs_sin_rank1,
        hca_cmp_wkv_rank1,
        hca_cmp_wgate_rank1,
        hca_cmp_ape_rank1,
        hca_cmp_norm_w_rank1,
        hca_cmp_kv_state_rank1,
        hca_cmp_score_state_rank1,
        hca_compress_state_block_table_rank1,
        csa_cmp_wkv_rank1,
        csa_cmp_wgate_rank1,
        csa_cmp_ape_rank1,
        csa_cmp_norm_w_rank1,
        csa_cmp_kv_state_rank1,
        csa_cmp_score_state_rank1,
        csa_compress_state_block_table_rank1,
        csa_hadamard_idx_rank1,
        csa_idx_wq_b_rank1,
        csa_idx_wq_b_scale_rank1,
        csa_weights_proj_rank1,
        csa_inner_wkv_rank1,
        csa_inner_wgate_rank1,
        csa_inner_ape_rank1,
        csa_inner_norm_w_rank1,
        csa_inner_kv_state_rank1,
        csa_inner_score_state_rank1,
        csa_inner_compress_state_block_table_rank1,
        kv_cache_rank1,
        ori_block_table_rank1,
        ori_slot_mapping_rank1,
        cmp_kv_rank1,
        cmp_block_table_rank1,
        cmp_sparse_indices_rank1,
        cmp_sparse_lens_rank1,
        idx_kv_cache_rank1,
        idx_block_table_rank1,
        position_ids_rank1,
        hca_cmp_slot_mapping_rank1,
        hca_state_slot_mapping_rank1,
        csa_cmp_slot_mapping_rank1,
        csa_idx_slot_mapping_rank1,
        csa_state_slot_mapping_rank1,
        csa_inner_state_slot_mapping_rank1,
        attn_sink_rank1,
        wo_a_rank1,
        wo_b_rank1,
        wo_b_scale_rank1,
        hc_ffn_fn_rank1,
        hc_ffn_scale_rank1,
        hc_ffn_base_rank1,
        norm_w_rank1,
        gate_w_rank1,
        gate_bias_rank1,
        tid2eid_rank1,
        input_ids_rank1,
        routed_w1_rank1,
        routed_w1_scale_rank1,
        routed_w3_rank1,
        routed_w3_scale_rank1,
        routed_w2_rank1,
        routed_w2_scale_rank1,
        shared_w1_rank1,
        shared_w1_scale_rank1,
        shared_w3_rank1,
        shared_w3_scale_rank1,
        shared_w2_rank1,
        shared_w2_scale_rank1,
        x_out_rank1,
        pub_counts1,
        count_done1,
        data_done1,
        recv_x1,
        recv_scale1,
        recv_w1,
        recv_r_route1,
        routed_y_buf1,
        combine_done1,
        num_tokens,
        device=1,
    )
    return


def _build_prefill_fwd_rank_specific_specs(
    *,
    start_pos: int,
    num_tokens: int,
) -> list[TensorSpec | ScalarSpec]:
    base_specs = {
        spec.name: spec
        for spec in build_tensor_specs(start_pos=start_pos, num_tokens=num_tokens, layer_id=0)
        if isinstance(spec, TensorSpec)
    }
    fallback_specs_cache: dict[int, dict[str, TensorSpec]] = {}

    def fast_fallback(layer_id: int, name: str) -> torch.Tensor | None:
        if name not in base_specs:
            return None
        spec = base_specs[name]
        if not spec.shape or int(spec.shape[0]) != N_RANKS:
            return None
        shape = list(spec.shape[1:])
        dtype = spec.dtype
        kind = attention_kind_for_layer(layer_id)
        pos = torch.arange(start_pos, start_pos + T, dtype=torch.int64)

        if name == "position_ids":
            return pos.to(torch.int32)
        if name == "input_ids":
            return (torch.arange(T, dtype=torch.int64) % VOCAB).contiguous()
        if name in {
            "ori_block_table",
            "cmp_block_table",
            "idx_block_table",
            "hca_compress_state_block_table",
            "csa_compress_state_block_table",
            "csa_inner_compress_state_block_table",
        }:
            return torch.zeros(shape, dtype=dtype)
        if name == "ori_slot_mapping":
            return (pos % BLOCK_SIZE).to(torch.int64)
        if name in {"hca_state_slot_mapping", "csa_state_slot_mapping", "csa_inner_state_slot_mapping"}:
            return pos.to(torch.int64)
        if name in {"hca_cmp_slot_mapping", "csa_cmp_slot_mapping", "csa_idx_slot_mapping"}:
            out = torch.full(shape, -1, dtype=torch.int64)
            if kind == "hca" and name == "hca_cmp_slot_mapping":
                mask = ((pos + 1) % HCA_COMPRESS_RATIO) == 0
                out[mask] = ((pos[mask] + 1) // HCA_COMPRESS_RATIO) - 1
            elif kind == "csa" and name in {"csa_cmp_slot_mapping", "csa_idx_slot_mapping"}:
                mask = ((pos + 1) % CSA_COMPRESS_RATIO) == 0
                out[mask] = ((pos[mask] + 1) // CSA_COMPRESS_RATIO) - 1
            return out
        if name == "cmp_sparse_lens":
            return torch.clamp(torch.arange(1, T + 1, dtype=torch.int32), max=SPARSE_TOPK)
        if name == "cmp_sparse_indices":
            out = torch.full(shape, -1, dtype=torch.int32)
            for t in range(T):
                valid = min(t + 1, SPARSE_TOPK)
                first = t + 1 - valid
                for k in range(valid):
                    out[t, k] = BLOCK_SIZE + first + k
            return out
        if name in FWD_RANK_SLICED_CACHE_NAMES or name in {
            "hca_cmp_kv_state",
            "hca_cmp_score_state",
            "csa_cmp_kv_state",
            "csa_cmp_score_state",
            "csa_inner_kv_state",
            "csa_inner_score_state",
            "csa_hadamard_idx",
        }:
            return torch.zeros(shape, dtype=dtype)
        return None

    def fallback(layer_id: int, name: str, rank: int) -> torch.Tensor:
        fast = fast_fallback(layer_id, name)
        if fast is not None:
            return fast.contiguous()
        # Synthetic metadata/cache fixtures only need to match the attention
        # kind. Building them through the standalone layer fixture is relatively
        # expensive, so avoid repeating that work for every real layer.
        kind = attention_kind_for_layer(layer_id)
        fixture_layer_id = 0 if kind == "swa" else 2 if kind == "csa" else 3
        if fixture_layer_id not in fallback_specs_cache:
            fallback_specs_cache[fixture_layer_id] = {
                spec.name: spec
                for spec in build_tensor_specs(start_pos=start_pos, num_tokens=num_tokens, layer_id=fixture_layer_id)
                if isinstance(spec, TensorSpec)
            }
        spec = fallback_specs_cache[fixture_layer_id][name]
        value = spec.create_tensor()
        if not value.shape or int(value.shape[0]) != N_RANKS:
            raise ValueError(f"{name}: expected leading rank dimension {N_RANKS}, got {list(value.shape)}")
        return value[rank].clone().contiguous()

    out: list[TensorSpec | ScalarSpec] = []
    for rank in range(N_RANKS):
        for name in HOST_TENSOR_ORDER:
            if name == "x_next":
                continue
            if name == "x_hc":
                base = base_specs[name]
                if not base.shape or int(base.shape[0]) != N_RANKS:
                    raise ValueError(f"{name}: expected leading rank dimension {N_RANKS}, got {base.shape}")

                def init_x_hc(src=base, rank_i=rank):
                    return src.create_tensor()[rank_i].clone().contiguous()

                out.append(
                    TensorSpec(
                        f"{name}_rank{rank}",
                        list(base.shape[1:]),
                        base.dtype,
                        init_value=init_x_hc,
                        is_output=False,
                    )
                )
                continue
            if name in FWD_HCA_COMPACT_NAMES:
                layer_ids = list(FWD_HCA_LAYER_IDS) or [0]
            elif name in FWD_CSA_COMPACT_NAMES:
                layer_ids = list(FWD_CSA_LAYER_IDS) or [0]
            else:
                layer_ids = list(range(PREFILL_FWD_ACTIVE_LAYERS))
            base = base_specs[name]
            if not base.shape or int(base.shape[0]) != N_RANKS:
                raise ValueError(f"{name}: expected leading rank dimension {N_RANKS}, got {base.shape}")
            if name in FWD_RANK_SLICED_CACHE_NAMES:
                shape = [len(layer_ids), *base.shape[1:]]
                def init_value(layer_ids_eff=layer_ids, name_eff=name, rank_eff=rank):
                    values = [fallback(layer_id, name_eff, rank_eff).contiguous() for layer_id in layer_ids_eff]
                    return torch.stack(values, dim=0).contiguous()
            else:
                shape = [len(layer_ids) * base.shape[1], *base.shape[2:]]
                def init_value(layer_ids_eff=layer_ids, name_eff=name, rank_eff=rank):
                    values = [fallback(layer_id, name_eff, rank_eff).contiguous() for layer_id in layer_ids_eff]
                    return torch.cat(values, dim=0).contiguous()
            out.append(
                TensorSpec(
                    f"{name}_rank{rank}",
                    shape,
                    base.dtype,
                    init_value=init_value,
                    is_output=base.is_output,
                )
            )
        out.append(TensorSpec(f"x_out_rank{rank}", [T, HC_MULT, D], torch.bfloat16, is_output=True))
    out.append(ScalarSpec("num_tokens", torch.int32, num_tokens))
    return out


def golden_prefill_fwd_rank_specific(tensors: dict[str, Any]) -> None:
    combined: dict[str, Any] = {"num_tokens": tensors["num_tokens"]}
    for name in HOST_TENSOR_ORDER:
        if name == "x_next":
            continue
        rank_values = []
        for rank in range(N_RANKS):
            rank_name = f"{name}_rank{rank}"
            if rank_name in tensors:
                rank_values.append(tensors[rank_name])
        if len(rank_values) == N_RANKS:
            combined[name] = torch.stack(rank_values, dim=0).contiguous()
    combined["x_out"] = torch.empty(
        [N_RANKS, T, HC_MULT, D],
        dtype=tensors["x_out_rank0"].dtype,
        device=tensors["x_out_rank0"].device,
    )
    golden_prefill_fwd(combined)
    for rank in range(N_RANKS):
        tensors[f"x_out_rank{rank}"][:] = combined["x_out"][rank]
        for name in FWD_RANK_SLICED_CACHE_NAMES:
            rank_name = f"{name}_rank{rank}"
            if rank_name in tensors and name in combined:
                tensors[rank_name][:] = combined[name][rank]



def golden_prefill_fwd(tensors: dict[str, Any]) -> None:
    """Golden for rank-first, layer-stacked prefill forward."""
    x_hc = tensors["x_hc"]
    if x_hc.ndim == 3:
        x_hc = x_hc.unsqueeze(0).expand(N_RANKS, *x_hc.shape).contiguous()
    num_tokens = int(tensors["num_tokens"])
    layer_stride, hca_stride, csa_stride = _flat_stride_maps()
    for layer_i in range(PREFILL_FWD_ACTIVE_LAYERS):
        layer_tensors: dict[str, Any] = {}
        for name, value in tensors.items():
            if name == "x_out":
                continue
            if name == "x_hc":
                layer_tensors[name] = x_hc
            elif name == "num_tokens":
                layer_tensors[name] = num_tokens
            elif name in hca_stride and isinstance(value, torch.Tensor):
                layer_tensors[name] = _slice_rank_flat(
                    value,
                    FWD_HCA_INDEX_BY_LAYER[layer_i],
                    hca_stride[name],
                )
            elif name in csa_stride and isinstance(value, torch.Tensor):
                layer_tensors[name] = _slice_rank_flat(
                    value,
                    FWD_CSA_INDEX_BY_LAYER[layer_i],
                    csa_stride[name],
                )
            elif name in layer_stride and isinstance(value, torch.Tensor):
                layer_tensors[name] = _slice_rank_flat(value, layer_i, layer_stride[name])
            elif isinstance(value, torch.Tensor) and value.ndim > 1 and value.shape[0] == N_RANKS and value.shape[1] == PREFILL_FWD_ACTIVE_LAYERS:
                layer_tensors[name] = value[:, layer_i].clone()
            elif name in FWD_HCA_COMPACT_NAMES and isinstance(value, torch.Tensor):
                layer_value = value[FWD_HCA_INDEX_BY_LAYER[layer_i]].clone()
                layer_tensors[name] = layer_value.unsqueeze(0).expand(N_RANKS, *layer_value.shape).contiguous()
            elif name in FWD_CSA_COMPACT_NAMES and isinstance(value, torch.Tensor):
                layer_value = value[FWD_CSA_INDEX_BY_LAYER[layer_i]].clone()
                layer_tensors[name] = layer_value.unsqueeze(0).expand(N_RANKS, *layer_value.shape).contiguous()
            elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == PREFILL_FWD_ACTIVE_LAYERS:
                layer_value = value[layer_i].clone()
                layer_tensors[name] = layer_value.unsqueeze(0).expand(N_RANKS, *layer_value.shape).contiguous()
            else:
                layer_tensors[name] = value
        layer_tensors["x_next"] = torch.empty_like(x_hc)
        layer_tensors["layer_id"] = layer_i
        golden_prefill_layer(layer_tensors)
        x_hc = layer_tensors["x_next"].detach().clone()
        for name, value in layer_tensors.items():
            if name in {"x_hc", "x_next", "num_tokens", "layer_id"}:
                continue
            if name in tensors and isinstance(tensors[name], torch.Tensor):
                target = tensors[name]
                if name in FWD_RANK_SLICED_CACHE_NAMES:
                    target[:, layer_i] = value
                elif name in hca_stride:
                    hca_idx = FWD_HCA_INDEX_BY_LAYER[layer_i]
                    stride = hca_stride[name]
                    target[:, hca_idx * stride : (hca_idx + 1) * stride] = value
                elif name in csa_stride:
                    csa_idx = FWD_CSA_INDEX_BY_LAYER[layer_i]
                    stride = csa_stride[name]
                    target[:, csa_idx * stride : (csa_idx + 1) * stride] = value
                elif name in layer_stride:
                    stride = layer_stride[name]
                    target[:, layer_i * stride : (layer_i + 1) * stride] = value
    tensors["x_out"][:] = x_hc


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 synthetic packed prefill forward driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a5"])  # device-only: full multi-layer/-card forward, no *sim
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size. Import-time moe config must see this argv value.")
    parser.add_argument("-d", "--device", type=str,
                        default="13,15",
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--num-layers", type=int, default=PREFILL_FWD_ACTIVE_LAYERS, choices=PREFILL_FWD_LAYER_CHOICES)
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--kernel-only", action="store_true", default=False,
                        help="Run compile/runtime only, skip golden generation and compare, then sanity-check x_out.")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Run golden generation and compare. By default this multi-layer bring-up uses kernel-only sanity.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    if args.num_layers != PREFILL_FWD_ACTIVE_LAYERS:
        raise ValueError(
            "prefill_fwd --num-layers must match the import-time static value "
            f"{PREFILL_FWD_ACTIVE_LAYERS}; got {args.num_layers}"
        )
    specs = _build_prefill_fwd_rank_specific_specs(
        start_pos=args.start_pos,
        num_tokens=args.num_tokens,
    )
    compare_fn = {
        f"x_out_rank{rank}": valid_ratio_reldiff(args.num_tokens, diff_thd=X_NEXT_DIFF_THD, pct_thd=X_NEXT_PCT_THD)
        for rank in range(N_RANKS)
    }
    kernel_only = args.kernel_only or not args.compare
    result, runtime_tensors, golden_outputs = _run_jit_return_tensors(
        fn=l3_prefill_fwd_rank_specific,
        specs=specs,
        golden_fn=golden_prefill_fwd_rank_specific,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        compare_fn=compare_fn,
        save_data=False,
        kernel_only=kernel_only,
        allow_mismatch=False,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
    return



if __name__ == "__main__":
    main()
