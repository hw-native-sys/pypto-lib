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
from pypto.runtime import pto_isa_include_dir


_KERNEL_DIR = Path(__file__).parent / "kernels" / "paged_attention_cce"
_ATTENTION_ENTRY = _KERNEL_DIR / "attention" / "entry.cpp"
_ATTENTION_ROPE_ENTRY = _KERNEL_DIR / "attention_rope" / "entry.cpp"
_TILING_ENTRY = _KERNEL_DIR / "tiling" / "entry.cpp"


_CANN_SUBDIRS = (
    "include",
    "asc/impl/adv_api",
    "asc/impl/basic_api",
    "asc/impl/c_api",
    "asc/impl/basic_api/reg_compute",
    "asc/impl/simt_api",
    "asc/impl/utils",
    "asc",
    "asc/include",
    "asc/include/adv_api",
    "asc/include/basic_api",
    "asc/include/aicpu_api",
    "asc/include/c_api",
    "asc/include/interface",
    "asc/include/basic_api/reg_compute",
    "asc/include/simt_api",
    "asc/include/utils",
    "tikcpp/tikcfw",
    "tikcpp/tikcfw/interface",
    "tikcpp/tikcfw/impl",
)


def _cann_include_dirs() -> tuple[Path, ...]:
    cann_root = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"))
    # CANN puts the devkit under a host-architecture directory on some installs
    # and flat at the toolkit root on others, so probe both. Pinning one layout
    # leaves an x86_64 host with no CANN include dir at all, and ccec then dies
    # on a missing basic_api/kernel_basic_intf.h far inside the vendor headers.
    devkits = (cann_root / f"{os.uname().machine}-linux", cann_root)
    resolved = tuple(devkit / sub for devkit in devkits for sub in _CANN_SUBDIRS if (devkit / sub).is_dir())
    if not resolved:
        raise RuntimeError(
            f"no CANN devkit include directories found under ASCEND_HOME_PATH={cann_root}. "
            f"Expected asc/include or tikcpp/tikcfw below {cann_root} or below "
            f"{cann_root / f'{os.uname().machine}-linux'}; source the CANN set_env.sh of a "
            "toolkit that ships the AscendC devkit headers."
        )
    return resolved


_CANN_INCLUDE_DIRS = _cann_include_dirs()

# The fused rope+attention extern embeds the pypto-generated rope_qkv kernel,
# which includes <pto/pto-inst.hpp>; add the pto-isa include root for it.
# pto_isa_include_dir() resolves runtime/pto_isa.pin -- an ambient PTO_ISA_ROOT
# is not the pin and pypto no longer reads it.
_ROPE_INCLUDE_DIRS = _CANN_INCLUDE_DIRS + (pto_isa_include_dir(),)

SUPPORTED_PLATFORMS = ("a2a3", "a2a3sim")
# METADATA_BATCH_SLOTS is the PHYSICAL slot count of the metadata length arrays.
# It must stay in lockstep with kernel/metadata_layout.h's
# `kLengthArrayBytes = 16 * sizeof(int64_t)` and with qwen_fai_tiler::kMaxBatch,
# because tiling/entry.cpp unconditionally writes all 16 slots (zeros past the
# actual batch) and the barrier region starts right after them. It is NOT the
# model's batch: shrinking it to track a smaller runtime batch would move
# kKvLengthsOffset / kBarrierAlignmentOffset and run the barrier off the end of
# the allocation.
METADATA_BATCH_SLOTS = 16
# BATCH_PAD is the padded batch of the STANDALONE test kernels below
# (qwen_decode_attention_cce / _cache_offset_test). The fused decode path does
# not use it -- paged_attention_rope_cce takes whatever shapes decode_fwd passes,
# and the row window each call serves is passed explicitly (batch_offset /
# batch_count), so a public batch above METADATA_BATCH_SLOTS is served as
# consecutive windows rather than rejected.
BATCH_PAD = 16
DEFAULT_BLOCK_DIM = 24
BLOCK_SIZE = 128
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

TILING_BYTES = 2488
CUMULATIVE_Q_OFFSET = TILING_BYTES
KV_LENGTHS_OFFSET = CUMULATIVE_Q_OFFSET + METADATA_BATCH_SLOTS * 8
METADATA_PREFIX_BYTES = KV_LENGTHS_OFFSET + METADATA_BATCH_SLOTS * 8
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
    # First public batch row served by this call. The whole-batch inputs
    # (seq_lens, slot_mapping, block_table) are indexed from here; the per-row
    # buffers the caller slices itself stay 0-based. Scalars pack after every
    # tensor, so this lands at args[18] -- see kernel/fai_body.hpp.
    batch_offset: pl.Scalar[pl.INDEX],
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
    batch_offset: pl.Scalar[pl.INT32],
    batch_count: pl.Scalar[pl.INT32],
) -> pl.Tensor: ...


@pl.jit.inline(auto_scope=False)
def build_paged_attention_metadata(
    seq_lens: pl.Tensor,
    max_blocks_per_seq: pl.Scalar[pl.INT32],
    num_blocks: pl.Scalar[pl.INT32],
    metadata: pl.Tensor[[METADATA_BYTES], pl.UINT8],
    batch_offset: pl.Scalar[pl.INT32],
    batch_count: pl.Scalar[pl.INT32],
    prev_reader_tid: pl.Array[1, pl.TASK_ID],
):
    """Build runtime FAI metadata for one row window and return its dependency.

    ``batch_count`` rows starting at ``batch_offset`` are tiled into a 0-based
    tiling, so a caller serving more than METADATA_BATCH_SLOTS rows repeats the
    call per window. ``prev_reader_tid`` orders this rebuild after the readers of
    the PREVIOUS window's metadata -- the buffer is reused, and the attention
    tasks that read it live in a manual scope with no auto-tracked WAR edge.
    A length-one Array (not a plain TASK_ID Scalar) because an explicit dep may
    only name a tid the enclosing scope captured; callers that build the metadata
    once seed it with a fresh ``pl.system.task_dummy``.
    """
    with pl.spmd(1, name_hint="pa_tiling", allow_early_resolve=True, deps=[prev_reader_tid[0]]) as tiling_tid:
        metadata = paged_attention_tiling_cce(
            seq_lens,
            metadata,
            max_blocks_per_seq,
            num_blocks,
            batch_offset,
            batch_count,
        )
    return tiling_tid


@pl.jit
def qwen_decode_attention_cce(
    query: pl.Tensor[[BATCH_PAD, NUM_HEADS, HEAD_DIM], pl.BF16],
    key_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    value_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    block_table: pl.Tensor[[BATCH_PAD, MAX_BLOCKS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[BATCH_PAD], pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH_PAD, NUM_HEADS, HEAD_DIM], pl.BF16]],
) -> pl.Tensor[[BATCH_PAD, NUM_HEADS, HEAD_DIM], pl.BF16]:
    """Standalone B16 attention with vLLM's active-TND and paged-BSND ABI."""
    key_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    value_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    block_table.bind_dynamic(1, MAX_BLOCKS_DYN)

    metadata = pl.create_tensor([METADATA_BYTES], dtype=pl.UINT8)
    workspace = pl.create_tensor([WORKSPACE_BYTES], dtype=pl.UINT8)
    max_blocks_per_seq = pl.cast(pl.tensor.dim(block_table, 1), pl.INT32)
    num_blocks = pl.cast(pl.tensor.dim(key_cache, 0), pl.INT32)
    pa_tiling_seed = pl.array.create(1, pl.TASK_ID)
    pa_tiling_seed[0] = pl.system.task_dummy(deps=[])
    # Bound to its own SSA value: an unbound expression reaching an extern arg is
    # not a variable the orchestration codegen can pack.
    batch_count = pl.cast(pl.tensor.dim(seq_lens, 0), pl.INT32)
    tiling_tid = build_paged_attention_metadata(
        seq_lens,
        max_blocks_per_seq,
        num_blocks,
        metadata,
        0,
        batch_count,
        pa_tiling_seed,
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
    query: pl.Tensor[[BATCH_PAD, NUM_HEADS, HEAD_DIM], pl.BF16],
    key_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    value_cache: pl.Tensor[[NUM_BLOCKS_DYN, BLOCK_SIZE, KV_HIDDEN], pl.BF16],
    block_table: pl.Tensor[[BATCH_PAD, MAX_BLOCKS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[BATCH_PAD], pl.INT32],
    out: pl.Out[pl.Tensor[[BATCH_PAD, NUM_HEADS, HEAD_DIM], pl.BF16]],
) -> pl.Tensor[[BATCH_PAD, NUM_HEADS, HEAD_DIM], pl.BF16]:
    """Read the second layer from a two-layer paged KV pool."""
    key_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    value_cache.bind_dynamic(0, NUM_BLOCKS_DYN)
    block_table.bind_dynamic(1, MAX_BLOCKS_DYN)

    metadata = pl.create_tensor([METADATA_BYTES], dtype=pl.UINT8)
    workspace = pl.create_tensor([WORKSPACE_BYTES], dtype=pl.UINT8)
    max_blocks_per_seq = pl.tensor.dim(block_table, 1)
    layer_num_blocks = pl.tensor.dim(block_table, 0) * max_blocks_per_seq
    pa_tiling_seed = pl.array.create(1, pl.TASK_ID)
    pa_tiling_seed[0] = pl.system.task_dummy(deps=[])
    # Bound to its own SSA value: an unbound expression reaching an extern arg is
    # not a variable the orchestration codegen can pack.
    batch_count = pl.cast(pl.tensor.dim(seq_lens, 0), pl.INT32)
    tiling_tid = build_paged_attention_metadata(
        seq_lens,
        pl.cast(max_blocks_per_seq, pl.INT32),
        pl.cast(layer_num_blocks, pl.INT32),
        metadata,
        0,
        batch_count,
        pa_tiling_seed,
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
    "BATCH_PAD",
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
    "paged_attention_rope_cce",
    "paged_attention_tiling_cce",
    "qwen_decode_attention_cache_offset_test",
    "qwen_decode_attention_cce",
]
