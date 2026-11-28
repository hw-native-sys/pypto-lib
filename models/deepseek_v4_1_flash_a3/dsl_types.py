# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Symbolic extents and physical PyPTO types shared by the A3 kernels."""

import pypto.language as pl


# Dynamic shape variables.
T_DYN = pl.dynamic("V41_T_DYN")
LOCAL_T_DYN = pl.dynamic("V41_LOCAL_T_DYN")
SP_T_DYN = pl.dynamic("V41_SP_T_DYN")
ROUTE_T_DYN = pl.dynamic("V41_ROUTE_T_DYN")
SLAB_T_DYN = pl.dynamic("V41_SLAB_T_DYN")
B_DYN = pl.dynamic("V41_B_DYN")
STATE_BLOCKS_DYN = pl.dynamic("V41_STATE_BLOCKS_DYN")
Q_START_DYN = pl.dynamic("V41_Q_START_DYN")
TABLE_DYN = pl.dynamic("V41_TABLE_DYN")
ORI_BLOCKS_DYN = pl.dynamic("V41_ORI_BLOCKS_DYN")
CMP_BLOCKS_DYN = pl.dynamic("V41_CMP_BLOCKS_DYN")
INDEX_BLOCKS_DYN = pl.dynamic("V41_INDEX_BLOCKS_DYN")
CMP_POSITIONS_DYN = pl.dynamic("V41_CMP_POSITIONS_DYN")
INDEX_POOL_BLOCKS_DYN = pl.dynamic("V41_INDEX_POOL_BLOCKS_DYN")
INDEX_POOL_POSITIONS_DYN = pl.dynamic("V41_INDEX_POOL_POSITIONS_DYN")

# Encoded cache and projection layouts.
FP8_DTYPE = pl.UINT8
SCALE_DTYPE = pl.UINT8
MX_A_LAYOUT = pl.ND
MX_B_LAYOUT = pl.ND
PROJECTION_LAYOUT = pl.ND
