# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AIV bridge for the fused DeepSeek-V4 MoE preprocessing phases."""


import os
from pathlib import Path

import pypto.language as pl

from config import FLASH as M, MOE_TOKENS


_KERNEL_DIR = Path(__file__).parent / "kernels" / "moe_fused_cce"
_ENTRY = _KERNEL_DIR / "entry.cpp"


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

T = MOE_TOKENS
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
MIX_PAD = 32
HC_PAD = 8
LINEAR_T_TILE = 16
GATE_M_TILE = 16
GATE_T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
FUSED_AIV_CORES = 8
SYNCALL_SOFT_SLOT_INT32 = 8
SYNC_WORKSPACE_SLOTS = FUSED_AIV_CORES * SYNCALL_SOFT_SLOT_INT32


@pl.jit.extern(
    core_type="aiv",
    source=_ENTRY,
    include_dirs=_INCLUDE_DIRS,
)
def split_mix_ffn_norm_cce(
    x_mixed: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    inv_rms: pl.Tensor[[LINEAR_T_TILE, 1], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    mixes_raw: pl.Tensor[[LINEAR_T_TILE, MIX_PAD], pl.FP32],
    pre_val_store: pl.Tensor[[LINEAR_T_TILE, HC_PAD], pl.FP32],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    x_flat: pl.Tensor[[T, HC_DIM], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    xg_buf: pl.Tensor[[GATE_T_PAD, D], pl.FP32],
    ffn_inv_rms_buf: pl.Tensor[[GATE_T_PAD, 1], pl.FP32],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    xn_scale_buf: pl.Tensor[[GATE_T_PAD, 1], pl.FP32],
    sync_workspace: pl.Tensor[[SYNC_WORKSPACE_SLOTS], pl.INT32],
    scale0: pl.Scalar[pl.FP32],
    scale1: pl.Scalar[pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, D], pl.BF16]: ...
