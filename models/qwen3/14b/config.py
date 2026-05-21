# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B model and decode-kernel configuration."""

import pypto.language as pl

# Dynamic dimensions used by the JIT/program signatures.
USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
KV_CACHE_ROWS_DYN = pl.dynamic("KV_CACHE_ROWS_DYN")
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")
ROPE_SEQ_DYN = pl.dynamic("ROPE_SEQ_DYN")
LAYER_DYN = pl.dynamic("LAYER_DYN")
LAYER_HIDDEN_ROWS_DYN = pl.dynamic("LAYER_HIDDEN_ROWS_DYN")
LAYER_INTER_ROWS_DYN = pl.dynamic("LAYER_INTER_ROWS_DYN")

# Model shape.
BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
INTERMEDIATE = 17408
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
VOCAB = 152064
NUM_LAYERS = 40

# Numeric constants.
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)
HALF_DIM = HEAD_DIM // 2

# Scope 1 tiling constants.
INPUT_PROJ_K_CHUNK = 256
KV_PROJ_K_CHUNK = 256
Q_OUT_CHUNK = 256
KV_OUT_CHUNK = 256
BATCH_TILE = 16

# Scope 2 tiling constants.
Q_HEAD_BATCH = 5
Q_HEAD_PAD = 16
# SEQ_TILE = 128 keeps each K/V tile at 32 KB (BLOCK_SIZE * HEAD_DIM * BF16),
# letting the cube L0B fit two tiles simultaneously (64 KB platform limit).
# This is required by the fa_fused mixed root in decode_layer.py; raising
# SEQ_TILE to 256 makes a single tile fill L0B and breaks the cube/vec fuse.
SEQ_TILE = 128
SB_BATCH = 128
BLOCK_SIZE = SEQ_TILE

# Scope 3 tiling constants.
K_CHUNK = 256
OUT_PROJ_K_CHUNK = 256
OUT_PROJ_N_CHUNK = 256
MLP_OUT_CHUNK = 256
DOWN_MLP_CHUNK = 256
DOWN_OUT_CHUNK = 256
FINAL_RMS_K_CHUNK = 128
LM_HEAD_K_CHUNK = 128
VOCAB_CHUNK = 64

# Decode grouping.
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
TOTAL_Q_GROUPS = NUM_KV_HEADS * Q_GROUPS
MAX_BLOCKS_PER_SEQ = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
