# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 DSpark draft configuration."""

from models.deepseek_v4_1_flash.config import DSPARK_SPEC_TOKENS, FLASH


DSPARK_DRAFT_LAYERS = 3
DSPARK_QUERY_WIDTH = DSPARK_SPEC_TOKENS
DSPARK_SWA_INDEX_WIDTH = (FLASH.sliding_window + DSPARK_QUERY_WIDTH + 63) // 64 * 64
DSPARK_MARKOV_RANK = 256
DSPARK_NOISE_TOKEN_ID = 128799
