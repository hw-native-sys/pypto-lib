# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 CANN FusedInferAttentionScore bridge for Qwen3 decode attention."""

import os
from pathlib import Path

import pypto.language as pl


_KERNEL_DIR = Path(__file__).parent / "kernels" / "paged_attention_cce"
_ATTENTION_ENTRY = _KERNEL_DIR / "attention" / "entry.cpp"
_TILING_ENTRY = _KERNEL_DIR / "tiling" / "entry.cpp"


def _cann_include_dirs() -> tuple[Path, ...]:
    cann_root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"))
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


_CANN_INCLUDE_DIRS = _cann_include_dirs()

SUPPORTED_PLATFORMS = ("a2a3", "a2a3sim")
BATCH = 16
DEFAULT_BLOCK_DIM = 24
BLOCK_SIZE = 128
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

TILING_BYTES = 2488
CUMULATIVE_Q_OFFSET = TILING_BYTES
KV_LENGTHS_OFFSET = CUMULATIVE_Q_OFFSET + BATCH * 8
METADATA_PREFIX_BYTES = KV_LENGTHS_OFFSET + BATCH * 8
BARRIER_SLOT_BYTES = 512
BARRIER_PHYSICAL_LANES = DEFAULT_BLOCK_DIM * 2
# The CCE wrapper aligns the barrier start at runtime, so reserve one slot of
# alignment slack before the maximum 48 single-writer barrier slots.
METADATA_BYTES = (
    (METADATA_PREFIX_BYTES + BARRIER_SLOT_BYTES - 1 + BARRIER_PHYSICAL_LANES * BARRIER_SLOT_BYTES + 31)
    // 32
    * 32
)
WORKSPACE_BYTES = 66_132_544

NUM_BLOCKS_DYN = pl.dynamic("PA_NUM_BLOCKS_DYN")
MAX_BLOCKS_DYN = pl.dynamic("PA_MAX_BLOCKS_DYN")


@pl.jit.extern(
    core_type="mixed",
    aic_source=_ATTENTION_ENTRY,
    aiv_source=_ATTENTION_ENTRY,
    include_dirs=_CANN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def paged_attention_cce(
    query: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    block_table: pl.Tensor,
    out: pl.Out[pl.Tensor],
    workspace: pl.InOut[pl.Tensor],
    metadata: pl.InOut[pl.Tensor],
    cache_row_offset: pl.Scalar[pl.INDEX],
) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="aiv",
    source=_TILING_ENTRY,
    include_dirs=_CANN_INCLUDE_DIRS,
)
def paged_attention_tiling_cce(
    seq_lens: pl.Tensor,
    metadata: pl.Out[pl.Tensor],
    max_blocks_per_seq: pl.Scalar[pl.INT32],
    num_blocks: pl.Scalar[pl.INT32],
) -> pl.Tensor: ...


@pl.jit.inline(auto_scope=False)
def build_paged_attention_metadata(
    seq_lens: pl.Tensor,
    max_blocks_per_seq: pl.Scalar[pl.INT32],
    num_blocks: pl.Scalar[pl.INT32],
    metadata: pl.Tensor[[METADATA_BYTES], pl.UINT8],
):
    """Build runtime FAI metadata and return its scheduler dependency."""
    with pl.spmd(1, name_hint="pa_tiling", allow_early_resolve=True) as tiling_tid:
        metadata = paged_attention_tiling_cce(
            seq_lens,
            metadata,
            max_blocks_per_seq,
            num_blocks,
        )
    return tiling_tid


@pl.jit
def qwen_decode_attention_cce(
    query: pl.Tensor[[BATCH, NUM_HEADS, HEAD_DIM], pl.BF16],
    key_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    value_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    block_table: pl.Tensor[[BATCH, MAX_BLOCKS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[BATCH], pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH, NUM_HEADS, HEAD_DIM], pl.BF16]],
) -> pl.Tensor[[BATCH, NUM_HEADS, HEAD_DIM], pl.BF16]:
    """Standalone B16 attention with vLLM's active-TND and paged-BSND ABI."""
    key_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    value_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    block_table.bind_dynamic(1, MAX_BLOCKS_DYN)

    metadata = pl.create_tensor([METADATA_BYTES], dtype=pl.UINT8)
    workspace = pl.create_tensor([WORKSPACE_BYTES], dtype=pl.UINT8)
    max_blocks_per_seq = pl.cast(pl.tensor.dim(block_table, 1), pl.INT32)
    num_blocks = pl.cast(pl.tensor.dim(key_cache, 0), pl.INT32)
    tiling_tid = build_paged_attention_metadata(
        seq_lens,
        max_blocks_per_seq,
        num_blocks,
        metadata,
    )
    attention_core_num = DEFAULT_BLOCK_DIM
    with pl.spmd(
        attention_core_num,
        name_hint="fa_fused",
        sync_start=True,
        deps=[tiling_tid],
    ) as _attention_tid:
        out = paged_attention_cce(
            query,
            key_cache,
            value_cache,
            block_table,
            out,
            workspace,
            metadata,
            0,
        )
    return out


@pl.jit
def qwen_decode_attention_cache_offset_test(
    query: pl.Tensor[[BATCH, NUM_HEADS, HEAD_DIM], pl.BF16],
    key_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    value_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    block_table: pl.Tensor[[BATCH, MAX_BLOCKS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[BATCH], pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH, NUM_HEADS, HEAD_DIM], pl.BF16]],
) -> pl.Tensor[[BATCH, NUM_HEADS, HEAD_DIM], pl.BF16]:
    """Read the second layer from a two-layer paged KV pool."""
    key_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    value_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    block_table.bind_dynamic(1, MAX_BLOCKS_DYN)

    metadata = pl.create_tensor([METADATA_BYTES], dtype=pl.UINT8)
    workspace = pl.create_tensor([WORKSPACE_BYTES], dtype=pl.UINT8)
    max_blocks_per_seq = pl.tensor.dim(block_table, 1)
    layer_num_blocks = pl.tensor.dim(block_table, 0) * max_blocks_per_seq
    tiling_tid = build_paged_attention_metadata(
        seq_lens,
        pl.cast(max_blocks_per_seq, pl.INT32),
        pl.cast(layer_num_blocks, pl.INT32),
        metadata,
    )
    cache_row_offset = layer_num_blocks * BLOCK_SIZE * NUM_KV_HEADS
    attention_core_num = DEFAULT_BLOCK_DIM
    with pl.spmd(
        attention_core_num,
        name_hint="fa_fused",
        sync_start=True,
        deps=[tiling_tid],
    ) as _attention_tid:
        out = paged_attention_cce(
            query,
            key_cache,
            value_cache,
            block_table,
            out,
            workspace,
            metadata,
            cache_row_offset,
        )
    return out


__all__ = [
    "BATCH",
    "BLOCK_SIZE",
    "DEFAULT_BLOCK_DIM",
    "HEAD_DIM",
    "KV_HIDDEN",
    "METADATA_BYTES",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "SUPPORTED_PLATFORMS",
    "WORKSPACE_BYTES",
    "build_paged_attention_metadata",
    "paged_attention_cce",
    "paged_attention_tiling_cce",
    "qwen_decode_attention_cache_offset_test",
    "qwen_decode_attention_cce",
]
