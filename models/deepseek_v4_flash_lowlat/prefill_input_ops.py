# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Populate prefill input and fresh-cache buffers on the owning Worker."""

import pypto.language as pl
import pypto.language.distributed as pld
from .config import FLASH as M


INPUT_ROWS = pl.dynamic("LOWLAT_COMPACT_INPUT_ROWS")
INPUT_RANKS = 8
INPUT_D = M.hidden_size
INPUT_HC = M.hc_mult


@pl.jit(auto_scope=False)
def l2_expand_compact_input(
    embedding: pl.Tensor[[INPUT_ROWS, INPUT_D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[INPUT_ROWS, INPUT_HC, INPUT_D], pl.FP32]],
):
    embedding.bind_dynamic(0, INPUT_ROWS)
    x_hc.bind_dynamic(0, INPUT_ROWS)
    rows = pl.tensor.dim(embedding, 0)
    flat = pl.reshape(x_hc, [rows * INPUT_HC, INPUT_D])
    with pl.spmd(rows, name_hint="prefill_compact_bf16_to_hc"):
        row = pl.tile.get_block_idx()
        source = pl.load(embedding, [row, 0], [1, INPUT_D])
        # BF16 is the upper half of FP32. Preserve every bit, including NaN
        # payloads, instead of relying on a numeric converter's NaN policy.
        halfwords = pl.reinterpret_view(source, pl.INT16)
        # The pinned A3 backend has no S16->S32 cast. Byte-sized integers
        # survive its S16->F16->F32 conversion exactly, then widen to S32.
        high_byte = pl.tile.shrs(halfwords, 8)
        low_byte = pl.tile.ands(halfwords, 255)
        high_fp32 = pl.cast(high_byte, target_type=pl.FP32)
        low_fp32 = pl.cast(low_byte, target_type=pl.FP32)
        high_i32 = pl.cast(high_fp32, target_type=pl.INT32)
        low_i32 = pl.cast(low_fp32, target_type=pl.INT32)
        high_bits = pl.tile.shls(high_i32, 24)
        low_bits = pl.tile.shls(low_i32, 16)
        shifted = pl.add(high_bits, low_bits)
        expanded = pl.reinterpret_view(shifted, pl.FP32)
        for channel in pl.range(INPUT_HC):
            pl.store(expanded, [row * INPUT_HC + channel, 0], flat)
    return x_hc


@pl.jit.host
def l3_expand_compact_input(
    embedding: pl.Tensor[[INPUT_RANKS, INPUT_ROWS, INPUT_D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[INPUT_RANKS, INPUT_ROWS, INPUT_HC, INPUT_D], pl.FP32]],
):
    for rank in pl.range(pld.world_size()):
        l2_expand_compact_input(embedding[rank], x_hc[rank], device=rank)
    return x_hc


INIT_KV_CACHE_BLOCKS = pl.dynamic("LOWLAT_INIT_KV_CACHE_BLOCKS")
INIT_HCA_CMP_KV_BLOCKS = pl.dynamic("LOWLAT_INIT_HCA_CMP_KV_BLOCKS")
INIT_CSA_CMP_KV_BLOCKS = pl.dynamic("LOWLAT_INIT_CSA_CMP_KV_BLOCKS")
INIT_IDX_KV_CACHE_BLOCKS = pl.dynamic("LOWLAT_INIT_IDX_KV_CACHE_BLOCKS")
INIT_IDX_KV_SCALE_BLOCKS = pl.dynamic("LOWLAT_INIT_IDX_KV_SCALE_BLOCKS")
INIT_HCA_COMPRESS_STATE_BLOCKS = pl.dynamic("LOWLAT_INIT_HCA_COMPRESS_STATE_BLOCKS")
INIT_CSA_COMPRESS_STATE_BLOCKS = pl.dynamic("LOWLAT_INIT_CSA_COMPRESS_STATE_BLOCKS")
INIT_CSA_INNER_COMPRESS_STATE_BLOCKS = pl.dynamic("LOWLAT_INIT_CSA_INNER_COMPRESS_STATE_BLOCKS")


@pl.jit(auto_scope=False)
def l2_initialize_prefill_caches(
    kv_cache: pl.Out[pl.Tensor[[INIT_KV_CACHE_BLOCKS, 128, 1, 512], pl.BF16]],
    hca_cmp_kv: pl.Out[pl.Tensor[[INIT_HCA_CMP_KV_BLOCKS, 1, 1, 512], pl.BF16]],
    csa_cmp_kv: pl.Out[pl.Tensor[[INIT_CSA_CMP_KV_BLOCKS, 32, 1, 512], pl.BF16]],
    idx_kv_cache: pl.Out[pl.Tensor[[INIT_IDX_KV_CACHE_BLOCKS, 32, 1, 128], pl.INT8]],
    idx_kv_scale: pl.Out[pl.Tensor[[INIT_IDX_KV_SCALE_BLOCKS, 32, 1, 1], pl.FP32]],
    hca_compress_state: pl.Out[pl.Tensor[[INIT_HCA_COMPRESS_STATE_BLOCKS, 8, 1024], pl.FP32]],
    csa_compress_state: pl.Out[pl.Tensor[[INIT_CSA_COMPRESS_STATE_BLOCKS, 4, 2048], pl.FP32]],
    csa_inner_compress_state: pl.Out[pl.Tensor[[INIT_CSA_INNER_COMPRESS_STATE_BLOCKS, 4, 512], pl.FP32]],
    hca_compress_state_pattern: pl.Tensor[[1, 1024], pl.FP32],
    csa_compress_state_pattern: pl.Tensor[[1, 2048], pl.FP32],
    csa_inner_compress_state_pattern: pl.Tensor[[1, 512], pl.FP32],
):
    kv_cache.bind_dynamic(0, INIT_KV_CACHE_BLOCKS)
    hca_cmp_kv.bind_dynamic(0, INIT_HCA_CMP_KV_BLOCKS)
    csa_cmp_kv.bind_dynamic(0, INIT_CSA_CMP_KV_BLOCKS)
    idx_kv_cache.bind_dynamic(0, INIT_IDX_KV_CACHE_BLOCKS)
    idx_kv_scale.bind_dynamic(0, INIT_IDX_KV_SCALE_BLOCKS)
    hca_compress_state.bind_dynamic(0, INIT_HCA_COMPRESS_STATE_BLOCKS)
    csa_compress_state.bind_dynamic(0, INIT_CSA_COMPRESS_STATE_BLOCKS)
    csa_inner_compress_state.bind_dynamic(0, INIT_CSA_INNER_COMPRESS_STATE_BLOCKS)
    kv_cache_elements = pl.tensor.dim(kv_cache, 0) * 65536
    kv_cache_flat = pl.reshape(kv_cache, [1, kv_cache_elements])
    with pl.spmd((kv_cache_elements + 8191) // 8192, name_hint="prefill_init_kv_cache"):
        kv_cache_offset = pl.tile.get_block_idx() * 8192
        kv_cache_valid = pl.min(8192, kv_cache_elements - kv_cache_offset)
        kv_cache_zero = pl.tile.full([1, 8192], dtype=pl.FP32, value=0.0)
        kv_cache_converted = pl.cast(kv_cache_zero, target_type=pl.BF16)
        kv_cache_tile = pl.set_validshape(kv_cache_converted, 1, kv_cache_valid)
        pl.store(kv_cache_tile, [0, kv_cache_offset], kv_cache_flat)
    hca_cmp_kv_elements = pl.tensor.dim(hca_cmp_kv, 0) * 512
    hca_cmp_kv_flat = pl.reshape(hca_cmp_kv, [1, hca_cmp_kv_elements])
    with pl.spmd((hca_cmp_kv_elements + 8191) // 8192, name_hint="prefill_init_hca_cmp_kv"):
        hca_cmp_kv_offset = pl.tile.get_block_idx() * 8192
        hca_cmp_kv_valid = pl.min(8192, hca_cmp_kv_elements - hca_cmp_kv_offset)
        hca_cmp_kv_zero = pl.tile.full([1, 8192], dtype=pl.FP32, value=0.0)
        hca_cmp_kv_converted = pl.cast(hca_cmp_kv_zero, target_type=pl.BF16)
        hca_cmp_kv_tile = pl.set_validshape(hca_cmp_kv_converted, 1, hca_cmp_kv_valid)
        pl.store(hca_cmp_kv_tile, [0, hca_cmp_kv_offset], hca_cmp_kv_flat)
    csa_cmp_kv_elements = pl.tensor.dim(csa_cmp_kv, 0) * 16384
    csa_cmp_kv_flat = pl.reshape(csa_cmp_kv, [1, csa_cmp_kv_elements])
    with pl.spmd((csa_cmp_kv_elements + 8191) // 8192, name_hint="prefill_init_csa_cmp_kv"):
        csa_cmp_kv_offset = pl.tile.get_block_idx() * 8192
        csa_cmp_kv_valid = pl.min(8192, csa_cmp_kv_elements - csa_cmp_kv_offset)
        csa_cmp_kv_zero = pl.tile.full([1, 8192], dtype=pl.FP32, value=0.0)
        csa_cmp_kv_converted = pl.cast(csa_cmp_kv_zero, target_type=pl.BF16)
        csa_cmp_kv_tile = pl.set_validshape(csa_cmp_kv_converted, 1, csa_cmp_kv_valid)
        pl.store(csa_cmp_kv_tile, [0, csa_cmp_kv_offset], csa_cmp_kv_flat)
    idx_kv_cache_elements = pl.tensor.dim(idx_kv_cache, 0) * 4096
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [1, idx_kv_cache_elements])
    with pl.spmd((idx_kv_cache_elements + 8191) // 8192, name_hint="prefill_init_idx_kv_cache"):
        idx_kv_cache_offset = pl.tile.get_block_idx() * 8192
        idx_kv_cache_valid = pl.min(8192, idx_kv_cache_elements - idx_kv_cache_offset)
        idx_kv_cache_zero = pl.tile.full([1, 8192], dtype=pl.FP16, value=0.0)
        idx_kv_cache_converted = pl.cast(idx_kv_cache_zero, target_type=pl.INT8)
        idx_kv_cache_tile = pl.set_validshape(idx_kv_cache_converted, 1, idx_kv_cache_valid)
        pl.store(idx_kv_cache_tile, [0, idx_kv_cache_offset], idx_kv_cache_flat)
    idx_kv_scale_elements = pl.tensor.dim(idx_kv_scale, 0) * 32
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [1, idx_kv_scale_elements])
    with pl.spmd((idx_kv_scale_elements + 8191) // 8192, name_hint="prefill_init_idx_kv_scale"):
        idx_kv_scale_offset = pl.tile.get_block_idx() * 8192
        idx_kv_scale_valid = pl.min(8192, idx_kv_scale_elements - idx_kv_scale_offset)
        idx_kv_scale_zero = pl.tile.full([1, 8192], dtype=pl.FP32, value=0.0)
        idx_kv_scale_tile = pl.set_validshape(idx_kv_scale_zero, 1, idx_kv_scale_valid)
        pl.store(idx_kv_scale_tile, [0, idx_kv_scale_offset], idx_kv_scale_flat)
    hca_compress_state_elements = pl.tensor.dim(hca_compress_state, 0) * 8192
    hca_compress_state_flat = pl.reshape(hca_compress_state, [1, hca_compress_state_elements])
    with pl.spmd(hca_compress_state_elements // 1024, name_hint="prefill_init_hca_compress_state"):
        hca_compress_state_row = pl.tile.get_block_idx()
        hca_compress_state_value = pl.load(hca_compress_state_pattern, [0, 0], [1, 1024])
        pl.store(hca_compress_state_value, [0, hca_compress_state_row * 1024], hca_compress_state_flat)
    csa_compress_state_elements = pl.tensor.dim(csa_compress_state, 0) * 8192
    csa_compress_state_flat = pl.reshape(csa_compress_state, [1, csa_compress_state_elements])
    with pl.spmd(csa_compress_state_elements // 2048, name_hint="prefill_init_csa_compress_state"):
        csa_compress_state_row = pl.tile.get_block_idx()
        csa_compress_state_value = pl.load(csa_compress_state_pattern, [0, 0], [1, 2048])
        pl.store(csa_compress_state_value, [0, csa_compress_state_row * 2048], csa_compress_state_flat)
    csa_inner_compress_state_elements = pl.tensor.dim(csa_inner_compress_state, 0) * 2048
    csa_inner_compress_state_flat = pl.reshape(csa_inner_compress_state, [1, csa_inner_compress_state_elements])
    with pl.spmd(csa_inner_compress_state_elements // 512, name_hint="prefill_init_csa_inner_compress_state"):
        csa_inner_compress_state_row = pl.tile.get_block_idx()
        csa_inner_compress_state_value = pl.load(csa_inner_compress_state_pattern, [0, 0], [1, 512])
        pl.store(csa_inner_compress_state_value, [0, csa_inner_compress_state_row * 512], csa_inner_compress_state_flat)
    return kv_cache, hca_cmp_kv, csa_cmp_kv, idx_kv_cache, idx_kv_scale, hca_compress_state, csa_compress_state, csa_inner_compress_state


@pl.jit.host
def l3_initialize_prefill_caches(
    kv_cache: pl.Out[pl.Tensor[[INPUT_RANKS, INIT_KV_CACHE_BLOCKS, 128, 1, 512], pl.BF16]],
    hca_cmp_kv: pl.Out[pl.Tensor[[INPUT_RANKS, INIT_HCA_CMP_KV_BLOCKS, 1, 1, 512], pl.BF16]],
    csa_cmp_kv: pl.Out[pl.Tensor[[INPUT_RANKS, INIT_CSA_CMP_KV_BLOCKS, 32, 1, 512], pl.BF16]],
    idx_kv_cache: pl.Out[pl.Tensor[[INPUT_RANKS, INIT_IDX_KV_CACHE_BLOCKS, 32, 1, 128], pl.INT8]],
    idx_kv_scale: pl.Out[pl.Tensor[[INPUT_RANKS, INIT_IDX_KV_SCALE_BLOCKS, 32, 1, 1], pl.FP32]],
    hca_compress_state: pl.Out[pl.Tensor[[INPUT_RANKS, INIT_HCA_COMPRESS_STATE_BLOCKS, 8, 1024], pl.FP32]],
    csa_compress_state: pl.Out[pl.Tensor[[INPUT_RANKS, INIT_CSA_COMPRESS_STATE_BLOCKS, 4, 2048], pl.FP32]],
    csa_inner_compress_state: pl.Out[pl.Tensor[[INPUT_RANKS, INIT_CSA_INNER_COMPRESS_STATE_BLOCKS, 4, 512], pl.FP32]],
    hca_compress_state_pattern: pl.Tensor[[INPUT_RANKS, 1, 1024], pl.FP32],
    csa_compress_state_pattern: pl.Tensor[[INPUT_RANKS, 1, 2048], pl.FP32],
    csa_inner_compress_state_pattern: pl.Tensor[[INPUT_RANKS, 1, 512], pl.FP32],
):
    for rank in pl.range(pld.world_size()):
        l2_initialize_prefill_caches(
            kv_cache[rank],
            hca_cmp_kv[rank],
            csa_cmp_kv[rank],
            idx_kv_cache[rank],
            idx_kv_scale[rank],
            hca_compress_state[rank],
            csa_compress_state[rank],
            csa_inner_compress_state[rank],
            hca_compress_state_pattern[rank],
            csa_compress_state_pattern[rank],
            csa_inner_compress_state_pattern[rank],
            device=rank,
        )
    return kv_cache, hca_cmp_kv, csa_cmp_kv, idx_kv_cache, idx_kv_scale, hca_compress_state, csa_compress_state, csa_inner_compress_state


# Dynamic shape variables.
LOOKUP_ROWS = pl.dynamic("LOWLAT_LOOKUP_INPUT_ROWS")
LOOKUP_VOCAB = pl.dynamic("LOWLAT_LOOKUP_INPUT_VOCAB")

# Model configuration.
RANKS = 8
HIDDEN = M.hidden_size
HC_MULT = M.hc_mult


@pl.jit(auto_scope=False)
def l2_lookup_prefill_input(
    embedding_weight: pl.Tensor[[LOOKUP_VOCAB, HIDDEN], pl.BF16],
    input_ids: pl.Tensor[[LOOKUP_ROWS], pl.INT64],
    x_hc: pl.Out[pl.Tensor[[LOOKUP_ROWS, HC_MULT, HIDDEN], pl.FP32]],
):
    embedding_weight.bind_dynamic(0, LOOKUP_VOCAB)
    input_ids.bind_dynamic(0, LOOKUP_ROWS)
    x_hc.bind_dynamic(0, LOOKUP_ROWS)
    rows = pl.tensor.dim(input_ids, 0)
    flat_rows = rows * HC_MULT
    flat = pl.reshape(x_hc, [flat_rows, HIDDEN])
    with pl.spmd(rows, name_hint="prefill_device_embedding_to_hc"):
        row = pl.tile.get_block_idx()
        token = pl.read(input_ids, [row])
        token_index = pl.cast(token, pl.INDEX)
        source = pl.load(embedding_weight, [token_index, 0], [1, HIDDEN])
        halfwords = pl.reinterpret_view(source, pl.INT16)
        high_byte = pl.tile.shrs(halfwords, 8)
        low_byte = pl.tile.ands(halfwords, 255)
        high_fp32 = pl.cast(high_byte, target_type=pl.FP32)
        low_fp32 = pl.cast(low_byte, target_type=pl.FP32)
        high_i32 = pl.cast(high_fp32, target_type=pl.INT32)
        low_i32 = pl.cast(low_fp32, target_type=pl.INT32)
        high_bits = pl.tile.shls(high_i32, 24)
        low_bits = pl.tile.shls(low_i32, 16)
        shifted = pl.add(high_bits, low_bits)
        expanded = pl.reinterpret_view(shifted, pl.FP32)
        for channel in pl.range(HC_MULT):
            output_row = row * HC_MULT + channel
            pl.store(expanded, [output_row, 0], flat)
    return x_hc


@pl.jit.host
def l3_lookup_prefill_input(
    embedding_weight: pl.Tensor[[RANKS, LOOKUP_VOCAB, HIDDEN], pl.BF16],
    input_ids: pl.Tensor[[RANKS, LOOKUP_ROWS], pl.INT64],
    x_hc: pl.Out[pl.Tensor[[RANKS, LOOKUP_ROWS, HC_MULT, HIDDEN], pl.FP32]],
):
    embedding_weight.bind_dynamic(1, LOOKUP_VOCAB)
    input_ids.bind_dynamic(1, LOOKUP_ROWS)
    x_hc.bind_dynamic(1, LOOKUP_ROWS)
    for rank in pl.range(pld.world_size()):
        l2_lookup_prefill_input(embedding_weight[rank], input_ids[rank], x_hc[rank], device=rank)
    return x_hc
