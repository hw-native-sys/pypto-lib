# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态CSA统一ABI与阶段编号：不含具体batch、历史长度或冻结执行图。

T为实际输入行容量（允许Native dummy行），B为Native请求槽数。MQ/IQ是两套
query前缀Tensor的独立长度，实际有效边界由device Q/L/position解释。其余
dynamic轴描述独立GM容量；不能把较大的容量当成有效行数。详细绑定合同在
entry.py；各数学文件说明最小容量和尾行。核组数为A3启动配置，不是shape档位。

阶段0保存每核调用代际；其余阶段一个核每调用只发布一次。阶段编号只标识数据，
不强制数字顺序。META由一个AIV写，其他完成槽归各核；512B隔离步长见phase_sync。
同一signal工作区必须固定本文件阶段含义、72参与者、单流顺序生命周期。
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
WORDS = pl.dynamic("CSA_FULL_WORDS")

PHASE_INPUT = 1
PHASE_PARTIAL = 2
PHASE_RAW = 3
PHASE_GATES = 4
PHASE_COMB = 5
PHASE_NORM = 6
PHASE_QA = 7
PHASE_QR = 8
PHASE_ROPE = 9
PHASE_QB = 10
PHASE_QUERY = 11
PHASE_KV_PROJECT = 12
PHASE_KV_NORM = 13
PHASE_ORIGINAL_CACHE = 14
PHASE_META = 15
PHASE_IDX_QUERY_PROJECT = 16
PHASE_IDX_QUERY_ROPE = 17
PHASE_IDX_QUERY_HAD = 18
PHASE_IDX_QUERY_QUANT = 19
PHASE_WEIGHTS_PROJECT = 20
PHASE_WEIGHTS = 21
PHASE_MAIN_PROJECT = 22
PHASE_IDX_PROJECT = 23
PHASE_MAIN_POOL = 24
PHASE_MAIN_COMMIT = 25
PHASE_MAIN_OUTPUT = 26
PHASE_IDX_POOL = 27
PHASE_IDX_COMMIT = 28
PHASE_IDX_NORM = 29
PHASE_IDX_HADAMARD = 30
PHASE_IDX_QUANT = 31
PHASE_IDX_SCALE = 32
PHASE_COEFF = 33
PHASE_SCORE_CUBE = 34
PHASE_SCORE_VECTOR = 35
PHASE_TOPK = 36
PHASE_ATTN_PLAN = 37
PHASE_ATTN_CUBE = 38
PHASE_ATTN_VECTOR = 39
PHASE_OA = 40
PHASE_O_QUANT = 41
PHASE_OB = 42
PHASE_OUT = 43
# 输出组的数学维度固定为8，但各组行任务数来自本次T。每组单独发布，
# 同一(组phase,核)每代仍只写一次；旧整阶段40/41供非流水策略保留。
# 新绑定必须重新分配清零信号区，不可沿用44阶段版本的进度内容。
PHASE_OA_GROUP_BASE = 44
PHASE_O_QUANT_GROUP_BASE = 52
PHASE_COUNT = 60
