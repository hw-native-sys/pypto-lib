# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent prefill MoE slabs, expert tiles and communication geometry."""

from .config import FLASH as M, _parse_int_argv

# Model configuration.
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
N_EXPERTS = M.n_routed_experts
N_BANK = N_EXPERTS + 1
SHARED_EID = N_EXPERTS
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size
N_RANKS = _parse_int_argv("--tp", 8)
if N_RANKS != 8:
    raise ValueError("Prefill MoE requires TP8")
MOE_INTER = M.moe_intermediate_size // N_RANKS
SWIGLU_LIMIT = M.swiglu_limit

# Prefill plan.
T = _parse_int_argv("--prefill-moe-wave-tokens", 2048)
if T not in (8, 32, 64, 128, 1024, 2048, 8192):
    raise ValueError("Prefill MoE wave tokens must be 8, 32, 64, 128, 1024, 2048 or 8192")
if MOE_INTER != 256:
    raise ValueError("Prefill MoE requires TP8 intermediate shards of 256 columns")

PREFILL_MOE_OWNER_REDUCE = _parse_int_argv("--prefill-moe-owner-reduce", 0)
if PREFILL_MOE_OWNER_REDUCE not in (0, 1):
    raise ValueError("Prefill owner reduction must be zero or one")

PREFILL_MOE_EXPERT_AFFINITY = _parse_int_argv("--prefill-moe-expert-affinity", 0)
if PREFILL_MOE_EXPERT_AFFINITY not in (0, 1):
    raise ValueError("Prefill expert affinity must be zero or one")
PREFILL_MOE_BATCHED_GATHER = _parse_int_argv("--prefill-moe-batched-gather", 0)
if PREFILL_MOE_BATCHED_GATHER not in (0, 1):
    raise ValueError("Prefill batched gather must be zero or one")
PREFILL_MOE_DIRECT_POST = _parse_int_argv("--prefill-moe-direct-post", 0)
if PREFILL_MOE_DIRECT_POST not in (0, 1):
    raise ValueError("Prefill direct post must be zero or one")
if PREFILL_MOE_DIRECT_POST and not PREFILL_MOE_OWNER_REDUCE:
    raise ValueError("Prefill direct post requires owner reduction")
PREFILL_MOE_FUSED_NORM_QUANT = _parse_int_argv("--prefill-moe-fused-norm-quant", 0)
if PREFILL_MOE_FUSED_NORM_QUANT not in (0, 1):
    raise ValueError("Prefill fused norm quant must be zero or one")
PREFILL_MOE_COMBINE_ROWS = _parse_int_argv("--prefill-moe-combine-rows", 1)
if PREFILL_MOE_COMBINE_ROWS not in (1, 4, 8):
    raise ValueError("Prefill combine rows must be 1, 4 or 8")
PREFILL_MOE_QUANT_GRID = _parse_int_argv("--prefill-moe-quant-grid", 0)
if PREFILL_MOE_QUANT_GRID not in (0, 1):
    raise ValueError("Quantization grid must be zero or one")
if PREFILL_MOE_QUANT_GRID and PREFILL_MOE_FUSED_NORM_QUANT:
    raise ValueError("Quantization grid requires separate original quantization")

PREFILL_MOE_OWNER_PREPARE = _parse_int_argv("--prefill-moe-owner-prepare", 0)
if PREFILL_MOE_OWNER_PREPARE not in (0, 1):
    raise ValueError("Owner preparation must be zero or one")
if PREFILL_MOE_OWNER_PREPARE and (T < 128 or T % 128):
    raise ValueError("Owner preparation requires waves divisible by128")

PREFILL_MOE_GATE_COPY_ROWS = _parse_int_argv("--prefill-moe-gate-copy-rows", 16)
if PREFILL_MOE_GATE_COPY_ROWS not in (16, 64, 128):
    raise ValueError("Prefill gate copy rows must be16,64 or128")

PREFILL_MOE_SHARE_GATE = _parse_int_argv("--prefill-moe-share-gate", 0)
if PREFILL_MOE_SHARE_GATE not in (0, 1):
    raise ValueError("Prefill gate sharing must be zero or one")
if PREFILL_MOE_OWNER_PREPARE and (PREFILL_MOE_SHARE_GATE or not PREFILL_MOE_QUANT_GRID or PREFILL_MOE_FUSED_NORM_QUANT):
    raise ValueError("Owner preparation requires unshared gate and grouped original quantization")

PREFILL_MOE_LOCAL_ROUTE_STATE = _parse_int_argv("--prefill-moe-local-route-state", 0)
if PREFILL_MOE_LOCAL_ROUTE_STATE not in (0, 1):
    raise ValueError("Prefill local route state must be zero or one")
EXPERT_IDS_PAD = (N_BANK + 7) // 8 * 8

# Expert and transport tiling.
RECV_TILE = 16
GATHER_TOKEN_TILE = 8
SHARED_GATHER_T_TILE = 16
RECV_MAX = RECV_TILE
IDX_PAD = 8
N_ROUTES = T * TOPK
N_EXPERTS_LIVE = min(N_EXPERTS, N_ROUTES)
N_SLOTS = N_EXPERTS_LIVE + (N_ROUTES - N_EXPERTS_LIVE) // RECV_TILE
SH_SLOT = N_SLOTS
SHARED_TILES = (T + RECV_TILE - 1) // RECV_TILE
N_SLOTS_B = N_SLOTS + SHARED_TILES
WORK_SLOTS_PAD = (N_SLOTS_B + IDX_PAD - 1) // IDX_PAD * IDX_PAD
T_PAD = (T + RECV_TILE - 1) // RECV_TILE * RECV_TILE
OWNER_ROWS = T // N_RANKS
OWNER_REDUCE_T_TILE = min(8, OWNER_ROWS)
OWNER_COPY_T_TILE = min(4, OWNER_ROWS)
RESULT_WINDOW_ROWS = 2 * T_PAD
REDUCE_LANE_ROWS = T_PAD if PREFILL_MOE_OWNER_REDUCE else N_RANKS * T_PAD
REDUCE_WINDOW_ROWS = 2 * REDUCE_LANE_ROWS
COPY_T_TILE = 4
COPY_D_TILE = 2048
REDUCE_T_TILE = 8
REDUCE_D_TILE = 512

GATE_WINDOW_ROWS = 2 * T_PAD

OWNER_WINDOW_ROWS = 2 * T_PAD
PREP_OWNER_ROWS = T // N_RANKS
