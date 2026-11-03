# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""TND MPMD 的动态 Tensor 容量与 A3 核组常量。

T 是输入 token 容量，B 是请求槽容量；实际有效范围只由设备 Q/L 决定。
main/Indexer Q/L 独立保留。跨 task 完成条件由 TMR 数据依赖表达，
本包不保留单 SPMD 的 GM phase 编号、代际计数或软件轮询。
"""

import pypto.language as pl

GROUPS = 24
VECTORS = 48
CORES = 72
ATTN_PLAN_ROWS = pl.dynamic("CSA_FULL_ATTN_PLAN_ROWS")
B = pl.dynamic("CSA_FULL_B")
CACHE_ROWS = pl.dynamic("CSA_FULL_CACHE_ROWS")
COEFF_COLS = pl.dynamic("CSA_FULL_COEFF_COLS")
COEFF_ROWS = pl.dynamic("CSA_FULL_COEFF_ROWS")
COMPRESSED_COLS = pl.dynamic("CSA_FULL_COMPRESSED_COLS")
GROUP_OFFSET_ROWS = pl.dynamic("CSA_FULL_GROUP_OFFSET_ROWS")
HEAD_RING_ROWS = pl.dynamic("CSA_FULL_HEAD_RING_ROWS")
IC = pl.dynamic("CSA_FULL_IC")
ICACHE_BYTES = pl.dynamic("CSA_FULL_ICACHE_BYTES")
ICACHE_PAGES = pl.dynamic("CSA_FULL_ICACHE_PAGES")
ICPAD = pl.dynamic("CSA_FULL_ICPAD")
IDX_HEAD_ROWS = pl.dynamic("CSA_FULL_IDX_HEAD_ROWS")
IDX_PROJ_ROWS = pl.dynamic("CSA_FULL_IDX_PROJ_ROWS")
IDX_ROPE_HEAD_ROWS = pl.dynamic("CSA_FULL_IDX_ROPE_HEAD_ROWS")
IDX_ROPE_ROWS = pl.dynamic("CSA_FULL_IDX_ROPE_ROWS")
IDX_SCALE_T_ROWS = pl.dynamic("CSA_FULL_IDX_SCALE_T_ROWS")
INDEXER_COLS = pl.dynamic("CSA_FULL_INDEXER_COLS")
INDEX_TRANSFER_COLS = pl.dynamic("CSA_FULL_INDEX_TRANSFER_COLS")
INDEX_TRANSFER_ROWS = pl.dynamic("CSA_FULL_INDEX_TRANSFER_ROWS")
IPOOL = pl.dynamic("CSA_FULL_IPOOL")
IPROJ = pl.dynamic("CSA_FULL_IPROJ")
IQ = pl.dynamic("CSA_FULL_IQ")
ISCALES = pl.dynamic("CSA_FULL_ISCALES")
ISLOTS = pl.dynamic("CSA_FULL_ISLOTS")
ISTATE_ELEMS = pl.dynamic("CSA_FULL_ISTATE_ELEMS")
ISTATE_PAGES = pl.dynamic("CSA_FULL_ISTATE_PAGES")
ITABLE_COLS = pl.dynamic("CSA_FULL_ITABLE_COLS")
KEY0_ROWS = pl.dynamic("CSA_FULL_KEY0_ROWS")
KEY64_ROWS = pl.dynamic("CSA_FULL_KEY64_ROWS")
KV_RING_ROWS = pl.dynamic("CSA_FULL_KV_RING_ROWS")
KV_ROWS = pl.dynamic("CSA_FULL_KV_ROWS")
MC = pl.dynamic("CSA_FULL_MC")
MCACHE_ROWS = pl.dynamic("CSA_FULL_MCACHE_ROWS")
MNORM = pl.dynamic("CSA_FULL_MNORM")
MPOOL = pl.dynamic("CSA_FULL_MPOOL")
MPROJ = pl.dynamic("CSA_FULL_MPROJ")
MQ = pl.dynamic("CSA_FULL_MQ")
MSLOTS = pl.dynamic("CSA_FULL_MSLOTS")
MSTATE_ELEMS = pl.dynamic("CSA_FULL_MSTATE_ELEMS")
MSTATE_PAGES = pl.dynamic("CSA_FULL_MSTATE_PAGES")
MTABLE_COLS = pl.dynamic("CSA_FULL_MTABLE_COLS")
ORIGINAL_COLS = pl.dynamic("CSA_FULL_ORIGINAL_COLS")
O_MM_ROWS = pl.dynamic("CSA_FULL_O_MM_ROWS")
O_SCALE_ROWS = pl.dynamic("CSA_FULL_O_SCALE_ROWS")
P16 = pl.dynamic("CSA_FULL_P16")
P8 = pl.dynamic("CSA_FULL_P8")
PACK_ROWS = pl.dynamic("CSA_FULL_PACK_ROWS")
PAIR_ROWS = pl.dynamic("CSA_FULL_PAIR_ROWS")
PARTIAL = pl.dynamic("CSA_FULL_PARTIAL")
QPROJ_ROWS = pl.dynamic("CSA_FULL_QPROJ_ROWS")
QR_ROWS = pl.dynamic("CSA_FULL_QR_ROWS")
QUERY_HEAD_ROWS = pl.dynamic("CSA_FULL_QUERY_HEAD_ROWS")
SCORE_ROWS = pl.dynamic("CSA_FULL_SCORE_ROWS")
T = pl.dynamic("CSA_FULL_T")
WEIGHTS_PARTIAL_ROWS = pl.dynamic("CSA_FULL_WEIGHTS_PARTIAL_ROWS")
WEIGHTS_ROWS = pl.dynamic("CSA_FULL_WEIGHTS_ROWS")
