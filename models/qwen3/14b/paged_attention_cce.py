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
_ATTENTION_ROPE_ENTRY = _KERNEL_DIR / "attention_rope" / "entry.cpp"
_PREFILL_ATTENTION_ENTRY = _KERNEL_DIR / "attention_prefill" / "entry.cpp"
_TILING_ENTRY = _KERNEL_DIR / "tiling" / "entry.cpp"
_PREFILL_TILING_ENTRY = _KERNEL_DIR / "tiling_prefill" / "entry.cpp"


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

# The fused rope+attention extern embeds the pypto-generated rope_qkv kernel,
# which includes <pto/pto-inst.hpp>; add the pto-isa include root for it.
_PTO_ISA_INCLUDE = Path(os.environ.get("PTO_ISA_ROOT", "")) / "include"
_ROPE_INCLUDE_DIRS = _CANN_INCLUDE_DIRS + (
    (_PTO_ISA_INCLUDE,) if _PTO_ISA_INCLUDE.is_dir() else ()
)

SUPPORTED_PLATFORMS = ("a2a3", "a2a3sim")
BATCH = 16
DEFAULT_BLOCK_DIM = 24
BLOCK_SIZE = 128
CAUSAL_MASK_SIZE = 2048
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
PREFILL_Q_TOKENS_DYN = pl.dynamic("PREFILL_PA_Q_TOKENS_DYN")


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
    core_type="mixed",
    aic_source=_ATTENTION_ROPE_ENTRY,
    aiv_source=_ATTENTION_ROPE_ENTRY,
    include_dirs=_ROPE_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def paged_attention_rope_cce(
    # This single-result extern binds its return to the first Out/InOut
    # parameter. Keep the real FAI output first instead of returning query.
    out: pl.Out[pl.Tensor],
    query: pl.InOut[pl.Tensor],
    key_cache: pl.InOut[pl.Tensor],
    value_cache: pl.InOut[pl.Tensor],
    block_table: pl.Tensor,
    workspace: pl.InOut[pl.Tensor],
    metadata: pl.InOut[pl.Tensor],
    q_proj: pl.Tensor,
    k_proj: pl.Tensor,
    v_proj: pl.Tensor,
    q_norm_w: pl.Tensor,
    k_norm_w: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    inv_rms_states: pl.Tensor,
    slot_mapping: pl.Tensor,
    seq_lens: pl.Tensor,
    cache_row_offset: pl.Scalar[pl.INDEX],
) -> pl.Tensor: ...


@pl.jit.extern(
    core_type="mixed",
    aic_source=_PREFILL_ATTENTION_ENTRY,
    aiv_source=_PREFILL_ATTENTION_ENTRY,
    include_dirs=_CANN_INCLUDE_DIRS,
    dual_aiv_dispatch=True,
)
def paged_prefill_attention_cce(
    query: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    block_table: pl.Tensor,
    out: pl.Out[pl.Tensor],
    workspace: pl.InOut[pl.Tensor],
    metadata: pl.InOut[pl.Tensor],
    causal_mask: pl.Tensor,
    cache_row_offset: pl.Scalar[pl.INDEX],
    block_table_offset: pl.Scalar[pl.INDEX],
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


@pl.jit.extern(
    core_type="aiv",
    source=_PREFILL_TILING_ENTRY,
    include_dirs=_CANN_INCLUDE_DIRS,
)
def paged_prefill_attention_tiling_cce(
    q_lens: pl.Tensor,
    kv_lens: pl.Tensor,
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


@pl.jit.inline(auto_scope=False)
def build_paged_prefill_attention_metadata(
    q_lens: pl.Tensor,
    kv_lens: pl.Tensor,
    max_blocks_per_seq: pl.Scalar[pl.INT32],
    num_blocks: pl.Scalar[pl.INT32],
    metadata: pl.Tensor[[METADATA_BYTES], pl.UINT8],
):
    """Build runtime FAI metadata for a causal prefill query tile."""
    with pl.spmd(1, name_hint="prefill_pa_tiling", allow_early_resolve=True) as tiling_tid:
        metadata = paged_prefill_attention_tiling_cce(
            q_lens,
            kv_lens,
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
def qwen_prefill_attention_cce(
    query: pl.Tensor[[PREFILL_Q_TOKENS_DYN, NUM_HEADS, HEAD_DIM], pl.BF16],
    key_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    value_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    block_table: pl.Tensor[[BATCH, MAX_BLOCKS_DYN], pl.INT32],
    q_lens: pl.Tensor[[BATCH], pl.INT32],
    kv_lens: pl.Tensor[[BATCH], pl.INT32],
    causal_mask: pl.Tensor[[CAUSAL_MASK_SIZE, CAUSAL_MASK_SIZE], pl.INT8],
    out: pl.Out[pl.Tensor[[PREFILL_Q_TOKENS_DYN, NUM_HEADS, HEAD_DIM], pl.BF16]],
) -> pl.Tensor[[PREFILL_Q_TOKENS_DYN, NUM_HEADS, HEAD_DIM], pl.BF16]:
    """Standalone causal prefill paged attention over a packed TND query tile."""
    query.bind_dynamic(0, PREFILL_Q_TOKENS_DYN)
    out.bind_dynamic(0, PREFILL_Q_TOKENS_DYN)
    key_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    value_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    block_table.bind_dynamic(1, MAX_BLOCKS_DYN)

    metadata = pl.create_tensor([METADATA_BYTES], dtype=pl.UINT8)
    workspace = pl.create_tensor([WORKSPACE_BYTES], dtype=pl.UINT8)
    max_blocks_per_seq = pl.cast(pl.tensor.dim(block_table, 1), pl.INT32)
    num_blocks = pl.cast(pl.tensor.dim(key_cache, 0), pl.INT32)
    tiling_tid = build_paged_prefill_attention_metadata(
        q_lens,
        kv_lens,
        max_blocks_per_seq,
        num_blocks,
        metadata,
    )
    with pl.spmd(
        DEFAULT_BLOCK_DIM,
        name_hint="prefill_fa_fused",
        sync_start=True,
        deps=[tiling_tid],
    ) as _attention_tid:
        out = paged_prefill_attention_cce(
            query,
            key_cache,
            value_cache,
            block_table,
            out,
            workspace,
            metadata,
            causal_mask,
            0,
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
    "CAUSAL_MASK_SIZE",
    "DEFAULT_BLOCK_DIM",
    "HEAD_DIM",
    "KV_HIDDEN",
    "METADATA_BYTES",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "PREFILL_Q_TOKENS_DYN",
    "SUPPORTED_PLATFORMS",
    "WORKSPACE_BYTES",
    "build_paged_attention_metadata",
    "build_paged_prefill_attention_metadata",
    "paged_attention_cce",
    "paged_attention_rope_cce",
    "paged_prefill_attention_cce",
    "paged_prefill_attention_tiling_cce",
    "paged_attention_tiling_cce",
    "qwen_decode_attention_cache_offset_test",
    "qwen_decode_attention_cce",
    "qwen_prefill_attention_cce",
]
