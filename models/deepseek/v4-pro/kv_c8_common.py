# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared KV Cache C8 constants for main compressors (device quant is inlined per file)."""

from __future__ import annotations

from config import FLASH as M
from mx_quant_common import MX_KV_GROUP

HEAD_DIM = M.head_dim
KV_SCALE_COLS = HEAD_DIM // MX_KV_GROUP
KV_C8_AMAX_EPS = 1e-12
LN2_F32 = 0.6931471805599453
KV_C8_QUANT_TILE = 16
KV_C8_SCALE_TILE_COLS = 32

assert HEAD_DIM % MX_KV_GROUP == 0
