# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared decode execution shapes and allocator pool geometry."""

import os

from config import FLASH, _parse_int_argv


# Deployment constants
DECODE_BATCH = 1                     # B: requests per decode step
DSPARK_SPEC_TOKENS = 7               # drafts the DSpark drafter proposes per request
DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS  # S: tokens the target model verifies per step
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ

# Parallelism constants. TP shards the MoE intermediate dim and the embedding /
# LM-head vocab. Attention uses head-group sharding and CSA token sharding;
# its public per-rank decode inputs still contain DECODE_TOKENS rows.
TP = _parse_int_argv("--tp", 8)
if TP <= 0:
    raise ValueError(f"--tp must be a positive integer, got {TP}")
WORLD_SIZE = TP
DP = WORLD_SIZE
MOE_TOKENS = DECODE_TOKENS

# Implementation constants
BLOCK_SIZE = 128                          # paged-KV page size / weight-quant block size
C4A_COMPRESSOR_BLOCK_SIZE = 4             # ratio-4 compressor state page size
C128_COMPRESSOR_BLOCK_SIZE = 8            # ratio-128 compressor state page size

# Static paged-cache pools. ``*_BLOCK_NUM`` is the global physical-pool capacity
# and is deliberately independent from the compute batch. Per-request ownership
# is expressed only by block tables.
KV_ORI_TABLE_MAX_BLOCKS = (FLASH.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_ORI_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
# Scheduler-visible tables use the same 128-source-token block granularity as
# the uncompressed KV cache. Compressed pools store fewer rows per table entry.
KV_CMP_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
IDX_CACHE_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
ORI_KV_BLOCK_NUM = 128
# The default compressed pool holds the complete context for every request.
# CTX_POOL_SCALE preserves the existing frozen-fixture pool overrides.
CAPACITY_SCALE = int(os.environ.get("CTX_POOL_SCALE", max(1, (DECODE_BATCH * KV_ORI_TABLE_MAX_BLOCKS + 31) // 32)))
if CAPACITY_SCALE <= 0:
    raise ValueError(f"CTX_POOL_SCALE must be a positive integer, got {CAPACITY_SCALE}")
CMP_KV_BLOCK_NUM = 32 * CAPACITY_SCALE
IDX_KV_BLOCK_NUM = 64 * CAPACITY_SCALE
HCA_STATE_PHYSICAL_BLOCKS = 64
CSA_STATE_PHYSICAL_BLOCKS = 65
CSA_INNER_STATE_PHYSICAL_BLOCKS = 65
DECODE_ORI_BLOCK_NUM = ORI_KV_BLOCK_NUM
DECODE_CMP_BLOCK_NUM = CMP_KV_BLOCK_NUM
DECODE_IDX_BLOCK_NUM = IDX_KV_BLOCK_NUM

