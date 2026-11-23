# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill cache page geometry and retained decode transaction extents."""

# Deployment constants
DECODE_BATCH = 64                 # B: requests per decode step, per DP rank
DSPARK_SPEC_TOKENS = 7            # drafts the DSpark drafter proposes per request
DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS  # S: tokens the target model verifies per step
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ

# Paging constants
BLOCK_SIZE = 32                           # paged-KV page size / weight-quant block size
COMPRESSED_BLOCK_TOKENS = 128            # source tokens per compressed cache page (same as MTP)
HCA_CMP_STORAGE_BLOCK_SIZE = COMPRESSED_BLOCK_TOKENS // 128
C4A_COMPRESSOR_BLOCK_SIZE = 2             # ratio-4 compressor state page size
C128_COMPRESSOR_BLOCK_SIZE = 8            # ratio-128 compressor state page size
KV_ORI_BLOCK_NUM = 512
KV_CMP_BLOCK_NUM = 256

# Persistent compressor state pool capacities shared by prefill and decode.
# HCA state includes 128 historical rows and one decode transaction.
HCA_STATE_BLOCKS_PER_REQUEST = (128 + DECODE_SEQ + C128_COMPRESSOR_BLOCK_SIZE - 1) // C128_COMPRESSOR_BLOCK_SIZE
HCA_STATE_PHYSICAL_BLOCKS = DECODE_BATCH * HCA_STATE_BLOCKS_PER_REQUEST
# Ratio-4 compressor state: eight history rows plus one decode transaction.
CSA_STATE_BLOCKS_PER_REQUEST = (8 + DECODE_SEQ + C4A_COMPRESSOR_BLOCK_SIZE - 1) // C4A_COMPRESSOR_BLOCK_SIZE
CSA_STATE_PHYSICAL_BLOCKS = DECODE_BATCH * CSA_STATE_BLOCKS_PER_REQUEST
# Inner compressor state uses the same request-local geometry.
CSA_INNER_STATE_BLOCKS_PER_REQUEST = CSA_STATE_BLOCKS_PER_REQUEST
CSA_INNER_STATE_PHYSICAL_BLOCKS = DECODE_BATCH * CSA_INNER_STATE_BLOCKS_PER_REQUEST

TP = 4
MOE_TOKENS = DECODE_TOKENS // TP
