# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 AIV bridge for fused split_pre_post + mix_x + ffn_norm."""

import os
from pathlib import Path

import pypto.language as pl

from config import (
    DECODE_BATCH,
    DECODE_SEQ,
    FLASH as M,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    PREFILL_BATCH,
    PREFILL_SEQ,
)


_KERNEL_DIR = Path(__file__).parent / "kernels" / "fused_pre_norm_cce"
_ENTRY = _KERNEL_DIR / "entry.cpp"
_DEBUG_ENTRY = _KERNEL_DIR / "debug" / "entry.cpp"
_BASELINE_ENTRY = _KERNEL_DIR / "baseline" / "entry.cpp"
_BASELINE_DEBUG_ENTRY = _KERNEL_DIR / "baseline_debug" / "entry.cpp"


def _cann_include_dirs() -> tuple[Path, ...]:
    cann_root = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"),
    )
    devkit = cann_root / "aarch64-linux"
    candidates = (
        devkit / "include",
        devkit / "asc" / "impl" / "adv_api",
        devkit / "asc" / "impl" / "basic_api",
        devkit / "asc" / "impl" / "c_api",
        devkit / "asc" / "impl" / "basic_api" / "reg_compute",
        devkit / "asc" / "impl" / "simt_api",
        devkit / "asc" / "impl" / "utils",
        devkit / "asc",
        devkit / "asc" / "include",
        devkit / "asc" / "include" / "adv_api",
        devkit / "asc" / "include" / "basic_api",
        devkit / "asc" / "include" / "aicpu_api",
        devkit / "asc" / "include" / "c_api",
        devkit / "asc" / "include" / "interface",
        devkit / "asc" / "include" / "basic_api" / "reg_compute",
        devkit / "asc" / "include" / "simt_api",
        devkit / "asc" / "include" / "utils",
        devkit / "tikcpp" / "tikcfw",
        devkit / "tikcpp" / "tikcfw" / "interface",
        devkit / "tikcpp" / "tikcfw" / "impl",
    )
    return tuple(path for path in candidates if path.is_dir())


_PTO_ISA_INCLUDE = Path(os.environ.get("PTO_ISA_ROOT", "")) / "include"
_INCLUDE_DIRS = _cann_include_dirs() + (
    (_PTO_ISA_INCLUDE,) if _PTO_ISA_INCLUDE.is_dir() else ()
)

SUPPORTED_PLATFORMS = ("a2a3", "a2a3sim")
FUSED_AIV_CORES = 8
SOFT_SYNC_COUNTER_INT32 = 16
FUSED_SOFT_SYNC_COUNTERS = 2
FUSED_SOFT_SYNC_WORDS = (
    FUSED_SOFT_SYNC_COUNTERS * SOFT_SYNC_COUNTER_INT32
)
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps
MIX_PAD = 32
HC_PAD = 8
LINEAR_T_TILE = 16
GATE_M_TILE = 16

T_DYN = pl.dynamic("FUSED_PRE_NORM_T_DYN")
T_PAD_DYN = pl.dynamic("FUSED_PRE_NORM_T_PAD_DYN")

# Debug entry stop points. Every logical AIV takes the same branch.
STOP_SPLIT_BEFORE_BARRIER1 = 0
STOP_AFTER_BARRIER1 = 1
STOP_MIX_BEFORE_BARRIER2 = 2
STOP_AFTER_BARRIER2 = 3
STOP_FULL = 4


@pl.jit.extern(
    core_type="aiv",
    source=_ENTRY,
    include_dirs=_INCLUDE_DIRS,
)
def fused_pre_norm_cce(
    # Direct spmd_submit maps these returns to Out params in declaration order.
    x_mixed: pl.Out[pl.Tensor],
    x_flat: pl.Tensor,
    inv_rms: pl.Tensor,
    mixes_raw: pl.Tensor,
    hc_base: pl.Tensor,
    norm_w: pl.Tensor,
    pre_val_store: pl.Out[pl.Tensor],
    post: pl.Out[pl.Tensor],
    xg_buf: pl.Out[pl.Tensor],
    ffn_inv_rms_buf: pl.Out[pl.Tensor],
    xn_scale_buf: pl.Out[pl.Tensor],
    x_norm_scale: pl.Out[pl.Tensor],
    sync_workspace: pl.InOut[
        pl.Tensor[[FUSED_SOFT_SYNC_WORDS], pl.INT32]
    ],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INDEX],
) -> tuple[
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
]:
    return (
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )


@pl.jit.extern(
    core_type="aiv",
    source=_DEBUG_ENTRY,
    include_dirs=_INCLUDE_DIRS,
)
def fused_pre_norm_debug_cce(
    x_mixed: pl.Out[pl.Tensor],
    x_flat: pl.Tensor,
    inv_rms: pl.Tensor,
    mixes_raw: pl.Tensor,
    hc_base: pl.Tensor,
    norm_w: pl.Tensor,
    pre_val_store: pl.Out[pl.Tensor],
    post: pl.Out[pl.Tensor],
    xg_buf: pl.Out[pl.Tensor],
    ffn_inv_rms_buf: pl.Out[pl.Tensor],
    xn_scale_buf: pl.Out[pl.Tensor],
    x_norm_scale: pl.Out[pl.Tensor],
    sync_workspace: pl.InOut[
        pl.Tensor[[FUSED_SOFT_SYNC_WORDS], pl.INT32]
    ],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INDEX],
    stop_after: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
]:
    return (
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )


@pl.jit.extern(
    core_type="aiv",
    source=_BASELINE_ENTRY,
    include_dirs=_INCLUDE_DIRS,
)
def fused_pre_norm_baseline_cce(
    x_mixed: pl.Out[pl.Tensor],
    x_flat: pl.Tensor,
    inv_rms: pl.Tensor,
    mixes_raw: pl.Tensor,
    hc_base: pl.Tensor,
    norm_w: pl.Tensor,
    pre_val_store: pl.Out[pl.Tensor],
    post: pl.Out[pl.Tensor],
    xg_buf: pl.Out[pl.Tensor],
    ffn_inv_rms_buf: pl.Out[pl.Tensor],
    xn_scale_buf: pl.Out[pl.Tensor],
    x_norm_scale: pl.Out[pl.Tensor],
    sync_workspace: pl.InOut[
        pl.Tensor[[FUSED_SOFT_SYNC_WORDS], pl.INT32]
    ],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INDEX],
) -> tuple[
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
]:
    return (
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )


@pl.jit.extern(
    core_type="aiv",
    source=_BASELINE_DEBUG_ENTRY,
    include_dirs=_INCLUDE_DIRS,
)
def fused_pre_norm_baseline_debug_cce(
    x_mixed: pl.Out[pl.Tensor],
    x_flat: pl.Tensor,
    inv_rms: pl.Tensor,
    mixes_raw: pl.Tensor,
    hc_base: pl.Tensor,
    norm_w: pl.Tensor,
    pre_val_store: pl.Out[pl.Tensor],
    post: pl.Out[pl.Tensor],
    xg_buf: pl.Out[pl.Tensor],
    ffn_inv_rms_buf: pl.Out[pl.Tensor],
    xn_scale_buf: pl.Out[pl.Tensor],
    x_norm_scale: pl.Out[pl.Tensor],
    sync_workspace: pl.InOut[
        pl.Tensor[[FUSED_SOFT_SYNC_WORDS], pl.INT32]
    ],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INDEX],
    stop_after: pl.Scalar[pl.INT32],
) -> tuple[
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
    pl.Tensor,
]:
    return (
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )


@pl.jit.inline
def _tag_test_tensors(
    x_flat: pl.Tensor,
    inv_rms: pl.Tensor,
    mixes_raw: pl.Tensor,
    hc_base: pl.Tensor,
    norm_w: pl.Tensor,
    x_mixed: pl.Tensor,
    pre_val_store: pl.Tensor,
    post: pl.Tensor,
    xg_buf: pl.Tensor,
    ffn_inv_rms_buf: pl.Tensor,
    xn_scale_buf: pl.Tensor,
    x_norm_scale: pl.Tensor,
):
    """Tag both sides of the standalone extern boundary for partial dump."""
    pl.dump_tag(x_flat)
    pl.dump_tag(inv_rms)
    pl.dump_tag(mixes_raw)
    pl.dump_tag(hc_base)
    pl.dump_tag(norm_w)
    pl.dump_tag(x_mixed)
    pl.dump_tag(pre_val_store)
    pl.dump_tag(post)
    pl.dump_tag(xg_buf)
    pl.dump_tag(ffn_inv_rms_buf)
    pl.dump_tag(xn_scale_buf)
    pl.dump_tag(x_norm_scale)
    return x_mixed


@pl.jit
def fused_pre_norm_test(
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    inv_rms: pl.Tensor[[T_PAD_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[T_PAD_DYN, MIX_PAD], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_mixed: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_val_store: pl.InOut[pl.Tensor[[T_PAD_DYN, HC_PAD], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    xg_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, D], pl.FP32]],
    ffn_inv_rms_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, 1], pl.FP32]],
    xn_scale_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, 1], pl.FP32]],
    x_norm_scale: pl.InOut[pl.Tensor[[T_DYN, 1], pl.FP32]],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INDEX],
):
    """Standalone production-entry test; output buffers are poison-initialized."""
    x_flat.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    x_norm_scale.bind_dynamic(0, T_DYN)
    inv_rms.bind_dynamic(0, T_PAD_DYN)
    mixes_raw.bind_dynamic(0, T_PAD_DYN)
    pre_val_store.bind_dynamic(0, T_PAD_DYN)
    xg_buf.bind_dynamic(0, T_PAD_DYN)
    ffn_inv_rms_buf.bind_dynamic(0, T_PAD_DYN)
    xn_scale_buf.bind_dynamic(0, T_PAD_DYN)

    sync_workspace = pl.create_tensor(
        [FUSED_SOFT_SYNC_WORDS],
        dtype=pl.INT32,
        init_value=0,
    )
    _tag_test_tensors(
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )
    # dump_tag is forward-sticky and marks both the before-dispatch and
    # after-completion snapshots of this InOut task argument. Do not reference
    # the consumed InOut value after the extern submit.
    pl.dump_tag(sync_workspace)
    ready_tid = pl.system.task_dummy(deps=[])
    # A direct call lets @pl.jit specialize the external callable referenced by
    # the spmd_submit below. The constant-false branch is removed before codegen.
    if False:
        _sync_workspace_specialize = pl.create_tensor(
            [FUSED_SOFT_SYNC_WORDS],
            dtype=pl.INT32,
            init_value=0,
        )
        (
            x_mixed,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
        ) = fused_pre_norm_cce(
            x_mixed,
            x_flat,
            inv_rms,
            mixes_raw,
            hc_base,
            norm_w,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
            _sync_workspace_specialize,
            scale0,
            scale1,
            num_tokens,
        )
    (
        (
            x_mixed,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
        ),
        _fused_tid,
    ) = pl.spmd_submit(
        self.fused_pre_norm_cce,  # noqa: F821 - materialized as a @pl.program method by @pl.jit
        x_mixed,
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
        sync_workspace,
        scale0,
        scale1,
        num_tokens,
        core_num=FUSED_AIV_CORES,
        sync_start=True,
        deps=[ready_tid],
    )
    _tag_test_tensors(
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )
    return (
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )


@pl.jit
def fused_pre_norm_debug_test(
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    inv_rms: pl.Tensor[[T_PAD_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[T_PAD_DYN, MIX_PAD], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_mixed: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_val_store: pl.InOut[pl.Tensor[[T_PAD_DYN, HC_PAD], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    xg_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, D], pl.FP32]],
    ffn_inv_rms_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, 1], pl.FP32]],
    xn_scale_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, 1], pl.FP32]],
    x_norm_scale: pl.InOut[pl.Tensor[[T_DYN, 1], pl.FP32]],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INDEX],
    stop_after: pl.Scalar[pl.INT32],
):
    """Debug-entry test that brackets each generated body and soft barrier."""
    x_flat.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    x_norm_scale.bind_dynamic(0, T_DYN)
    inv_rms.bind_dynamic(0, T_PAD_DYN)
    mixes_raw.bind_dynamic(0, T_PAD_DYN)
    pre_val_store.bind_dynamic(0, T_PAD_DYN)
    xg_buf.bind_dynamic(0, T_PAD_DYN)
    ffn_inv_rms_buf.bind_dynamic(0, T_PAD_DYN)
    xn_scale_buf.bind_dynamic(0, T_PAD_DYN)

    sync_workspace = pl.create_tensor(
        [FUSED_SOFT_SYNC_WORDS],
        dtype=pl.INT32,
        init_value=0,
    )
    _tag_test_tensors(
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )
    pl.dump_tag(sync_workspace)
    ready_tid = pl.system.task_dummy(deps=[])
    if False:
        _sync_workspace_specialize = pl.create_tensor(
            [FUSED_SOFT_SYNC_WORDS],
            dtype=pl.INT32,
            init_value=0,
        )
        (
            x_mixed,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
        ) = fused_pre_norm_debug_cce(
            x_mixed,
            x_flat,
            inv_rms,
            mixes_raw,
            hc_base,
            norm_w,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
            _sync_workspace_specialize,
            scale0,
            scale1,
            num_tokens,
            stop_after,
        )
    (
        (
            x_mixed,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
        ),
        _fused_tid,
    ) = pl.spmd_submit(
        self.fused_pre_norm_debug_cce,  # noqa: F821 - materialized as a @pl.program method by @pl.jit
        x_mixed,
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
        sync_workspace,
        scale0,
        scale1,
        num_tokens,
        stop_after,
        core_num=FUSED_AIV_CORES,
        sync_start=True,
        deps=[ready_tid],
    )
    _tag_test_tensors(
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )
    return (
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )


@pl.jit
def fused_pre_norm_baseline_test(
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    inv_rms: pl.Tensor[[T_PAD_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[T_PAD_DYN, MIX_PAD], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_mixed: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_val_store: pl.InOut[pl.Tensor[[T_PAD_DYN, HC_PAD], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    xg_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, D], pl.FP32]],
    ffn_inv_rms_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, 1], pl.FP32]],
    xn_scale_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, 1], pl.FP32]],
    x_norm_scale: pl.InOut[pl.Tensor[[T_DYN, 1], pl.FP32]],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INDEX],
):
    """Standalone test-only atomic 8/8 correctness baseline."""
    x_flat.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    x_norm_scale.bind_dynamic(0, T_DYN)
    inv_rms.bind_dynamic(0, T_PAD_DYN)
    mixes_raw.bind_dynamic(0, T_PAD_DYN)
    pre_val_store.bind_dynamic(0, T_PAD_DYN)
    xg_buf.bind_dynamic(0, T_PAD_DYN)
    ffn_inv_rms_buf.bind_dynamic(0, T_PAD_DYN)
    xn_scale_buf.bind_dynamic(0, T_PAD_DYN)

    sync_workspace = pl.create_tensor(
        [FUSED_SOFT_SYNC_WORDS],
        dtype=pl.INT32,
        init_value=0,
    )
    _tag_test_tensors(
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )
    pl.dump_tag(sync_workspace)
    ready_tid = pl.system.task_dummy(deps=[])
    if False:
        _sync_workspace_specialize = pl.create_tensor(
            [FUSED_SOFT_SYNC_WORDS],
            dtype=pl.INT32,
            init_value=0,
        )
        (
            x_mixed,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
        ) = fused_pre_norm_baseline_cce(
            x_mixed,
            x_flat,
            inv_rms,
            mixes_raw,
            hc_base,
            norm_w,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
            _sync_workspace_specialize,
            scale0,
            scale1,
            num_tokens,
        )
    (
        (
            x_mixed,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
        ),
        _fused_tid,
    ) = pl.spmd_submit(
        self.fused_pre_norm_baseline_cce,  # noqa: F821 - materialized as a @pl.program method by @pl.jit
        x_mixed,
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
        sync_workspace,
        scale0,
        scale1,
        num_tokens,
        core_num=FUSED_AIV_CORES,
        sync_start=True,
        deps=[ready_tid],
    )
    _tag_test_tensors(
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )
    return (
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )


@pl.jit
def fused_pre_norm_baseline_debug_test(
    x_flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    inv_rms: pl.Tensor[[T_PAD_DYN, 1], pl.FP32],
    mixes_raw: pl.Tensor[[T_PAD_DYN, MIX_PAD], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_mixed: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    pre_val_store: pl.InOut[pl.Tensor[[T_PAD_DYN, HC_PAD], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    xg_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, D], pl.FP32]],
    ffn_inv_rms_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, 1], pl.FP32]],
    xn_scale_buf: pl.InOut[pl.Tensor[[T_PAD_DYN, 1], pl.FP32]],
    x_norm_scale: pl.InOut[pl.Tensor[[T_DYN, 1], pl.FP32]],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INDEX],
    stop_after: pl.Scalar[pl.INT32],
):
    """Debug test-only atomic 8/8 baseline with uniform phase stops."""
    x_flat.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    x_norm_scale.bind_dynamic(0, T_DYN)
    inv_rms.bind_dynamic(0, T_PAD_DYN)
    mixes_raw.bind_dynamic(0, T_PAD_DYN)
    pre_val_store.bind_dynamic(0, T_PAD_DYN)
    xg_buf.bind_dynamic(0, T_PAD_DYN)
    ffn_inv_rms_buf.bind_dynamic(0, T_PAD_DYN)
    xn_scale_buf.bind_dynamic(0, T_PAD_DYN)

    sync_workspace = pl.create_tensor(
        [FUSED_SOFT_SYNC_WORDS],
        dtype=pl.INT32,
        init_value=0,
    )
    _tag_test_tensors(
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )
    pl.dump_tag(sync_workspace)
    ready_tid = pl.system.task_dummy(deps=[])
    if False:
        _sync_workspace_specialize = pl.create_tensor(
            [FUSED_SOFT_SYNC_WORDS],
            dtype=pl.INT32,
            init_value=0,
        )
        (
            x_mixed,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
        ) = fused_pre_norm_baseline_debug_cce(
            x_mixed,
            x_flat,
            inv_rms,
            mixes_raw,
            hc_base,
            norm_w,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
            _sync_workspace_specialize,
            scale0,
            scale1,
            num_tokens,
            stop_after,
        )
    (
        (
            x_mixed,
            pre_val_store,
            post,
            xg_buf,
            ffn_inv_rms_buf,
            xn_scale_buf,
            x_norm_scale,
        ),
        _fused_tid,
    ) = pl.spmd_submit(
        self.fused_pre_norm_baseline_debug_cce,  # noqa: F821 - materialized as a @pl.program method by @pl.jit
        x_mixed,
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
        sync_workspace,
        scale0,
        scale1,
        num_tokens,
        stop_after,
        core_num=FUSED_AIV_CORES,
        sync_start=True,
        deps=[ready_tid],
    )
    _tag_test_tensors(
        x_flat,
        inv_rms,
        mixes_raw,
        hc_base,
        norm_w,
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )
    return (
        x_mixed,
        pre_val_store,
        post,
        xg_buf,
        ffn_inv_rms_buf,
        xn_scale_buf,
        x_norm_scale,
    )


def golden_fused_pre_norm(tensors):
    """Independent Torch reference that preserves poison in unwritten rows."""
    import torch

    x_flat = tensors["x_flat"].float()
    inv_rms = tensors["inv_rms"].float()
    mixes_raw = tensors["mixes_raw"].float()
    hc_base = tensors["hc_base"].float()
    norm_w = tensors["norm_w"].float()
    scale0 = float(tensors["scale0"])
    scale1 = float(tensors["scale1"])
    t_dim = x_flat.shape[0]
    stop_after = int(tensors.get("stop_after", STOP_FULL))

    pre_scaled = mixes_raw[:t_dim, :HC_PAD] * inv_rms[:t_dim] * scale0
    pre_val = torch.sigmoid(pre_scaled + hc_base[:HC_PAD]) + HC_EPS
    tensors["pre_val_store"][:t_dim] = pre_val
    post_scaled = (
        mixes_raw[:t_dim, HC_MULT:HC_MULT + HC_PAD]
        * inv_rms[:t_dim]
        * scale1
    )
    post_pad = 2.0 * torch.sigmoid(
        post_scaled + hc_base[HC_MULT:HC_MULT + HC_PAD],
    )
    tensors["post"][:t_dim] = post_pad[:, :HC_MULT]

    if stop_after < STOP_MIX_BEFORE_BARRIER2:
        return

    x_hc = x_flat.reshape(t_dim, HC_MULT, D)
    y0 = x_hc[:, 0, :] * pre_val[:, 0:1]
    y1 = x_hc[:, 1, :] * pre_val[:, 1:2]
    y2 = x_hc[:, 2, :] * pre_val[:, 2:3]
    y3 = x_hc[:, 3, :] * pre_val[:, 3:4]
    mixed_fp32 = (y0 + y1) + (y2 + y3)
    # Match PTO BF16 rint before ffn_norm reloads the buffer.
    mixed_bits = (mixed_fp32.contiguous().view(torch.int32) + 0x8000) & -0x10000
    mixed_bf16 = mixed_bits.view(torch.float32).to(torch.bfloat16)
    tensors["x_mixed"][:t_dim] = mixed_bf16

    if stop_after < STOP_FULL:
        return

    active_tokens = max(0, min(t_dim, int(tensors["num_tokens"])))
    active_gate_tokens = min(
        t_dim,
        ((active_tokens + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE,
    )
    if active_gate_tokens == 0:
        return

    rms_x = mixed_bf16[:active_gate_tokens].float()
    xg = rms_x * norm_w.reshape(1, D)
    sq_sum = (rms_x * rms_x).sum(dim=-1, keepdim=True)
    ffn_inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)
    xg_amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    xn_scale = INT8_SCALE_MAX / xg_amax
    x_norm_scale = (xg_amax / INT8_SCALE_MAX) * ffn_inv_rms
    tensors["xg_buf"][:active_gate_tokens] = xg
    tensors["ffn_inv_rms_buf"][:active_gate_tokens] = ffn_inv_rms
    tensors["xn_scale_buf"][:active_gate_tokens] = xn_scale
    tensors["x_norm_scale"][:active_gate_tokens] = x_norm_scale


def build_tensor_specs(t_dim, num_tokens, *, debug_stop=None):
    """Poison-initialized direct-input specs for decode or prefill."""
    import torch
    from golden import ScalarSpec, TensorSpec

    t_pad = (
        (t_dim + LINEAR_T_TILE - 1)
        // LINEAR_T_TILE
        * LINEAR_T_TILE
    )

    def fp32_poison(*shape):
        raw = torch.full(shape, 0x7FC0D00D, dtype=torch.int32)
        return raw.view(torch.float32)

    def bf16_poison(*shape):
        raw = torch.full(shape, 0x7FC1, dtype=torch.int16)
        return raw.view(torch.bfloat16)

    specs = [
        TensorSpec(
            "x_flat",
            [t_dim, HC_DIM],
            torch.float32,
            init_value=lambda: torch.randn(t_dim, HC_DIM) * 0.05,
        ),
        TensorSpec(
            "inv_rms",
            [t_pad, 1],
            torch.float32,
            init_value=lambda: torch.rand(t_pad, 1) * 0.5 + 0.75,
        ),
        TensorSpec(
            "mixes_raw",
            [t_pad, MIX_PAD],
            torch.float32,
            init_value=lambda: torch.randn(t_pad, MIX_PAD),
        ),
        TensorSpec(
            "hc_base",
            [MIX_HC],
            torch.float32,
            init_value=lambda: torch.randn(MIX_HC),
        ),
        TensorSpec(
            "norm_w",
            [D],
            torch.bfloat16,
            init_value=lambda: torch.ones(D, dtype=torch.bfloat16),
        ),
        TensorSpec(
            "x_mixed",
            [t_dim, D],
            torch.bfloat16,
            init_value=lambda: bf16_poison(t_dim, D),
            is_output=True,
        ),
        TensorSpec(
            "pre_val_store",
            [t_pad, HC_PAD],
            torch.float32,
            init_value=lambda: fp32_poison(t_pad, HC_PAD),
            is_output=True,
        ),
        TensorSpec(
            "post",
            [t_dim, HC_MULT],
            torch.float32,
            init_value=lambda: fp32_poison(t_dim, HC_MULT),
            is_output=True,
        ),
        TensorSpec(
            "xg_buf",
            [t_pad, D],
            torch.float32,
            init_value=lambda: fp32_poison(t_pad, D),
            is_output=True,
        ),
        TensorSpec(
            "ffn_inv_rms_buf",
            [t_pad, 1],
            torch.float32,
            init_value=lambda: fp32_poison(t_pad, 1),
            is_output=True,
        ),
        TensorSpec(
            "xn_scale_buf",
            [t_pad, 1],
            torch.float32,
            init_value=lambda: fp32_poison(t_pad, 1),
            is_output=True,
        ),
        TensorSpec(
            "x_norm_scale",
            [t_dim, 1],
            torch.float32,
            init_value=lambda: fp32_poison(t_dim, 1),
            is_output=True,
        ),
        ScalarSpec("scale0", torch.float32, 0.076099),
        ScalarSpec("scale1", torch.float32, 0.032597),
        ScalarSpec("num_tokens", torch.int64, num_tokens),
    ]
    if debug_stop is not None:
        specs.append(ScalarSpec("stop_after", torch.int32, debug_stop))
    return specs


def poison_aware_allclose(actual, expected, *, rtol=1e-3, atol=1e-3, **_):
    """Numerical comparison plus bit-exact validation of untouched poison."""
    import torch

    poison = torch.isnan(expected)
    if actual.dtype == torch.bfloat16:
        actual_bits = actual.contiguous().view(torch.int16)
        expected_bits = expected.contiguous().view(torch.int16)
    elif actual.dtype == torch.float32:
        actual_bits = actual.contiguous().view(torch.int32)
        expected_bits = expected.contiguous().view(torch.int32)
    else:
        actual_bits = actual
        expected_bits = expected
    if not torch.equal(actual_bits[poison], expected_bits[poison]):
        changed = int((actual_bits[poison] != expected_bits[poison]).sum())
        return False, f"{changed} poison elements were unexpectedly overwritten"
    valid = ~poison
    if not bool(valid.any()):
        return True, ""
    close = torch.isclose(actual[valid].float(), expected[valid].float(), rtol=rtol, atol=atol)
    if bool(close.all()):
        return True, ""
    bad = int((~close).sum())
    max_abs = float((actual[valid].float() - expected[valid].float()).abs().max())
    return False, f"{bad} finite elements differ; max_abs={max_abs}"


__all__ = [
    "FUSED_AIV_CORES",
    "FUSED_SOFT_SYNC_COUNTERS",
    "FUSED_SOFT_SYNC_WORDS",
    "SOFT_SYNC_COUNTER_INT32",
    "STOP_AFTER_BARRIER1",
    "STOP_AFTER_BARRIER2",
    "STOP_FULL",
    "STOP_MIX_BEFORE_BARRIER2",
    "STOP_SPLIT_BEFORE_BARRIER1",
    "SUPPORTED_PLATFORMS",
    "fused_pre_norm_baseline_cce",
    "fused_pre_norm_baseline_debug_cce",
    "fused_pre_norm_baseline_debug_test",
    "fused_pre_norm_baseline_test",
    "fused_pre_norm_cce",
    "fused_pre_norm_debug_cce",
    "fused_pre_norm_debug_test",
    "fused_pre_norm_test",
]


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    modes = {
        "decode": DECODE_BATCH * DECODE_SEQ,
        "prefill": PREFILL_BATCH * PREFILL_SEQ,
    }
    debug_stops = {
        "split": STOP_SPLIT_BEFORE_BARRIER1,
        "barrier1": STOP_AFTER_BARRIER1,
        "mix": STOP_MIX_BEFORE_BARRIER2,
        "barrier2": STOP_AFTER_BARRIER2,
        "full": STOP_FULL,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=SUPPORTED_PLATFORMS,
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=("decode", "prefill", "all"), default="all")
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="defaults to all tokens in each selected mode",
    )
    parser.add_argument(
        "--debug-stop",
        choices=tuple(debug_stops),
        default=None,
        help="use the debug entry and stop at the selected phase boundary",
    )
    parser.add_argument(
        "--barrier-policy",
        choices=("dense", "baseline"),
        default="dense",
        help=(
            "select the experimental dense 4/8 target or the test-only "
            "atomic 8/8 baseline"
        ),
    )
    parser.add_argument(
        "--dump-args",
        nargs="?",
        const=1,
        default=0,
        type=int,
        choices=(0, 1, 2, 3),
    )
    parser.add_argument("--enable-l2-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    selected = tuple(modes) if args.mode == "all" else (args.mode,)
    for mode in selected:
        t_dim = modes[mode]
        num_tokens = t_dim if args.num_tokens is None else args.num_tokens
        debug_stop = (
            None if args.debug_stop is None else debug_stops[args.debug_stop]
        )
        if args.barrier_policy == "baseline":
            test_fn = (
                fused_pre_norm_baseline_test
                if debug_stop is None
                else fused_pre_norm_baseline_debug_test
            )
        else:
            test_fn = (
                fused_pre_norm_test
                if debug_stop is None
                else fused_pre_norm_debug_test
            )
        result = run_jit(
            fn=test_fn,
            specs=build_tensor_specs(
                t_dim,
                num_tokens,
                debug_stop=debug_stop,
            ),
            golden_fn=golden_fused_pre_norm,
            golden_data=args.golden_data,
            save_data=args.save_data,
            compile_only=args.compile_only,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_dump_args=args.dump_args,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "x_mixed": poison_aware_allclose,
                "pre_val_store": poison_aware_allclose,
                "post": poison_aware_allclose,
                "xg_buf": poison_aware_allclose,
                "ffn_inv_rms_buf": poison_aware_allclose,
                "xn_scale_buf": poison_aware_allclose,
                "x_norm_scale": poison_aware_allclose,
            },
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
