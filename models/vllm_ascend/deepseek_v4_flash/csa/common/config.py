# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA execution capacities and numerical constants."""

from ...config import FLASH as FLASH

# Compile capacity covers all six requested batch sizes (up to B=40).
# 64 * 6 = 384 also meets the reference output projection's 128-row tile.
TP = 1
DECODE_BATCH = 64
DSPARK_SPEC_TOKENS = 5
DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
BLOCK_SIZE = 32
C4A_COMPRESSOR_BLOCK_SIZE = 2
KV_ORI_BLOCK_NUM = 512
KV_CMP_BLOCK_NUM = 256
IDX_CACHE_BLOCK_NUM = 256
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
FP32_NEG_INF = -3.4028234663852886e38

# 候选：压缩历史达到2048行时使用Native片上Score，覆盖8K对照档位。
INDEXER_NATIVE_CUBE_MIN_ROWS = 2048
