# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 configuration"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Follows HuggingFace config.json"""

    name: str

    # ---- attention / hidden ----
    hidden_size: int
    num_attention_heads: int
    head_dim: int                  # MLA value-head dim
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int                  # grouped output projection
    sliding_window: int
    rms_norm_eps: float
    vocab_size: int

    # ---- MoE ----
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]
    routed_scaling_factor: float
    swiglu_limit: float

    # ---- layers ----
    num_hidden_layers: int
    num_hash_layers: int           # first layers use hash routing
    num_nextn_predict_layers: int  # multi-token-prediction layers
    compress_ratios: Tuple[int, ...]   # per-layer KV compression ratio (0 / 4 / 128)

    # ---- lightning indexer ----
    index_n_heads: int
    index_head_dim: int
    index_topk: int                # compressed positions kept by the indexer

    # ---- hyper-connections (HC) ----
    hc_mult: int                   # hc-stack width
    hc_sinkhorn_iters: int
    hc_eps: float

    # ---- context length / RoPE (YaRN; rope_scaling.* flattened) ----
    max_position_embeddings: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float                       # rope_scaling.factor
    beta_fast: int                           # rope_scaling.beta_fast
    beta_slow: int                           # rope_scaling.beta_slow
    original_max_position_embeddings: int    # rope_scaling.original_max_position_embeddings

    # ---- precision / quantization (quantization_config.* flattened; unused by decode kernels) ----
    dtype: Literal["bf16", "fp8"]              # quantization_config.quant_method
    scale_fmt: Optional[Literal["ue8m0"]]     # quantization_config.scale_fmt
    expert_dtype: Optional[Literal["fp4"]]    # MoE-expert weight dtype (None = same as `dtype`)
    scale_dtype: Literal["fp32", "fp8"]       # dequant-scale storage dtype

    # ---- deployment (not consumed by the decode kernels) ----
    max_batch_size: int            # max supported batch size (cache sizing)

    # ---- derived ----
    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def index_nope_head_dim(self) -> int:
        return self.index_head_dim - self.qk_rope_head_dim

    @property
    def index_weights_scale(self) -> float:
        return self.index_head_dim ** -0.5 * self.index_n_heads ** -0.5

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


DEMO = DeepSeekV4Config(
    name="demo",
    hidden_size=4096,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1024,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=4096,
    n_routed_experts=16,
    n_shared_experts=1,
    num_experts_per_tok=2,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.0,
    swiglu_limit=0.0,
    num_hidden_layers=8,
    num_hash_layers=0,
    num_nextn_predict_layers=1,
    compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=4096,
    rope_theta=10000.0,
    compress_rope_theta=40000.0,
    rope_factor=40.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=0,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype=None,
    scale_dtype="fp8",
    max_batch_size=4,
)

FLASH = DeepSeekV4Config(
    name="flash",
    hidden_size=4096,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1024,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=2048,
    n_routed_experts=256,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    num_hidden_layers=43,
    num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
    ),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=512,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=1_048_576,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype="fp4",
    scale_dtype="fp8",
    max_batch_size=4,
)

PRO = DeepSeekV4Config(
    name="pro",
    hidden_size=7168,
    num_attention_heads=128,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1536,
    o_lora_rank=1024,
    o_groups=16,
    sliding_window=128,
    rms_norm_eps=1e-6,
    vocab_size=129280,
    moe_intermediate_size=3072,
    n_routed_experts=384,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=2.5,
    swiglu_limit=10.0,
    num_hidden_layers=61,
    num_hash_layers=3,
    num_nextn_predict_layers=1,
    compress_ratios=(
        128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
    ),
    index_n_heads=64,
    index_head_dim=128,
    index_topk=1024,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    max_position_embeddings=1048576,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    dtype="fp8",
    scale_fmt="ue8m0",
    expert_dtype=None,
    scale_dtype="fp8",
    max_batch_size=4,
)

PRESETS = {p.name: p for p in (DEMO, FLASH, PRO)}


# Deployment constants
DECODE_BATCH = 64                 # B: logical requests per decode step
DSPARK_SPEC_TOKENS = 7            # drafts the DSpark drafter proposes per request
DECODE_SEQ = 1 + DSPARK_SPEC_TOKENS  # S: tokens the target model verifies per step
DECODE_TOKENS = DECODE_BATCH * DECODE_SEQ
DECODE_START_POS = 8192
PREFILL_BATCH = 1                 # B: prefill batch for the current kernel programs
PREFILL_SEQ = 512                 # S: prefill sequence for the current kernel programs
PREFILL_TOKENS = PREFILL_BATCH * PREFILL_SEQ

# Paging constants
BLOCK_SIZE = 32                           # paged-KV page size / weight-quant block size
C4A_COMPRESSOR_BLOCK_SIZE = 2             # ratio-4 compressor state page size
C128_COMPRESSOR_BLOCK_SIZE = 8            # ratio-128 compressor state page size
KV_ORI_MAX_BLOCKS = (FLASH.max_position_embeddings + BLOCK_SIZE - 1) // BLOCK_SIZE
KV_CMP_MAX_BLOCKS = (FLASH.max_position_embeddings // 4 + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_CACHE_MAX_BLOCKS = 2 * KV_CMP_MAX_BLOCKS
KV_ORI_BLOCK_NUM = 512
KV_CMP_BLOCK_NUM = 256
IDX_CACHE_BLOCK_NUM = 256
HCA_STATE_PHYSICAL_BLOCKS = 256
CSA_STATE_PHYSICAL_BLOCKS = 260
CSA_INNER_STATE_PHYSICAL_BLOCKS = 260

# Int8 quantization constants
INT8_SCALE_MAX = 127.0                    # per-row INT8 quant: clamp scale so |q| <= 127
INT8_AMAX_EPS = 1e-4                      # amax floor: avoids 127/0 on all-zero rows
FP32_NEG_INF = -3.4028234663852886e38     # most-negative finite fp32 (softmax masking)

# Parallelism constants
TP = 4    # tensor-parallel ranks per DP group
DP = 4    # DP groups per node
EP = 16   # expert-parallel world size (moe overrides it from --ep)

# A DP group owns one disjoint request partition.
DECODE_LOCAL_REQUESTS = DECODE_BATCH // DP

assert DECODE_BATCH % DP == 0
assert DECODE_LOCAL_REQUESTS == 16
assert DECODE_SEQ == 8

# MoE constants
MOE_TOKENS = DECODE_TOKENS * DP // EP
DECODE_RECV_MAX = DP * DECODE_TOKENS
PREFILL_RECV_MAX = DP * PREFILL_TOKENS
RECV_MAX = DECODE_RECV_MAX


# ---------------------------------------------------------------------------
# Canonical 1M context geometry
#
# The single source of truth for the 1M capacity ceiling and the fixed
# micro-tile/shard granularity. SWA/HCA/CSA metadata and kernels consume these
# names without expanding runtime work to the capacity ceiling.
#
# 1M is a *capacity ceiling*, not a per-step execution length. The ceiling must
# not appear as any runtime loop/task count — shard/leaf *counts* are derived
# per request from runtime visible length (see context_geometry.py). 128/12K/
# 16K/1M are test points, not length profiles or buckets. No runtime
# shard-granularity self-adaptation is introduced in this run.
# ---------------------------------------------------------------------------

# Storage ABI.  Main now uses 32-row paged KV blocks; semantic windows and
# attention shards remain independent compile-time quantities below.
CACHE_BLOCK_SIZE = BLOCK_SIZE

# Model-semantic ratios (match compress_ratios entries / sliding_window).
SWA_WINDOW_ROWS = 128
HCA_COMPRESS_RATIO = 128
CSA_COMPRESS_RATIO = 4

# Fixed compile-time micro-tile granularity. The *granularity* is fixed; the
# *number* of shards/leaves is derived at runtime from visible length and is
# NOT a user tunable. These must not enter the device ABI as runtime inputs
# (no rows_per_shard / candidates_per_leaf / pages_per_task parameters).
HCA_ROWS_PER_SHARD = 128          # four 32-row compressed-KV pages
CSA_CANDIDATES_PER_LEAF = 2048    # one local selector (reuses the 2K selector)

# Fixed downstream output width: the K is fixed, never the candidate-scan width.
CSA_TOPK = 512
CSA_MERGE_ARITY = 2

# Bounded ready frontier for the CSA Top-512 merge DAG (Phase D.1). First
# version is a platform constant; it only tunes scheduling fairness and never
# changes shard granularity or top-k semantics. Not consumed in Phase A.
CSA_TOPK_READY_FRONTIER_W = 8
TOPK_READY_FRONTIER_W = CSA_TOPK_READY_FRONTIER_W

# Max logical rows/candidates/leaves derived from the ceiling — used only for
# capacity accounting / admission, never as a runtime task count and never
# materialized as a dense per-B table (no [B, max_logical_pages] layout).
MAX_HCA_ROWS = FLASH.max_position_embeddings // HCA_COMPRESS_RATIO           # 8192
MAX_HCA_SHARDS = MAX_HCA_ROWS // HCA_ROWS_PER_SHARD              # 64
MAX_CSA_CANDIDATES = FLASH.max_position_embeddings // CSA_COMPRESS_RATIO    # 262144
MAX_CSA_LEAVES = MAX_CSA_CANDIDATES // CSA_CANDIDATES_PER_LEAF   # 128

# Phase D.1 exact Top-K forest capacity.  ``CSA_MAX_QUERIES`` is the fixed
# local launch upper bound, not a logical-context or granularity setting.
# A non-empty binary forest has one leaf task per packed leaf and one merge
# task per merge node, or at most 255 tasks per query at the 1M limit.
CSA_MAX_CANDIDATES = MAX_CSA_CANDIDATES
CSA_MAX_LEAVES_PER_QUERY = MAX_CSA_LEAVES
CSA_MAX_NODES_PER_QUERY = 2 * CSA_MAX_LEAVES_PER_QUERY - 1
CSA_MAX_QUERIES = 16
CSA_MAX_TOPK_TASKS = CSA_MAX_QUERIES * CSA_MAX_NODES_PER_QUERY
CSA_TOPK_INVALID_INDEX = -1
CSA_TOPK_INVALID_TASK_SLOT = CSA_MAX_TOPK_TASKS
CSA_LOGICAL_INDEX_MIN = 0
CSA_LOGICAL_INDEX_MAX = CSA_MAX_CANDIDATES - 1

# Every arena row contains fixed Top-512 score/index pairs packed as FP32
# lanes: 512 score lanes followed by 512 index-bit lanes.  The explicit byte
# size makes the bounded workspace accounting unambiguous.
CSA_PAIR_COMPONENTS = 2
CSA_PAIR_WIDTH = CSA_PAIR_COMPONENTS * CSA_TOPK
CSA_SCORE_INDEX_PAIR_BYTES = CSA_PAIR_COMPONENTS * 4
CSA_PAIR_BYTES = CSA_PAIR_WIDTH * 4
CSA_TOPK_PAIR_BYTES = CSA_PAIR_BYTES

# Minimum global physical-pool capacities for one request at the 1M ceiling.
# These are page budgets, not submitted-work counts. A serving allocator may
# provision more pages and admit multiple requests when their aggregate demand
# fits, but it must never pre-split these capacities by a fixed decode batch.
SWA_KV_POOL_BLOCKS = (SWA_WINDOW_ROWS + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE
HCA_KV_POOL_BLOCKS = (MAX_HCA_ROWS + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE
CSA_KV_POOL_BLOCKS = (MAX_CSA_CANDIDATES + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE
CSA_IDX_POOL_BLOCKS = CSA_KV_POOL_BLOCKS

# Phase D.1 owns separate paired main/index page maps.  Both have enough
# pages for exactly one 1M candidate stream, but they are never aliases for
# the same physical map or allocation group.
CSA_MAIN_POOL_PAGES = CSA_KV_POOL_BLOCKS
CSA_IDX_POOL_PAGES = CSA_IDX_POOL_BLOCKS
CSA_INDEX_POOL_PAGES = CSA_IDX_POOL_PAGES
CSA_MAIN_PAGES_AT_1M = MAX_CSA_CANDIDATES // CACHE_BLOCK_SIZE
CSA_IDX_PAGES_AT_1M = MAX_CSA_CANDIDATES // CACHE_BLOCK_SIZE
CSA_INDEX_PAGES_AT_1M = CSA_IDX_PAGES_AT_1M

# Phase C HCA persistent-state ABI.  The semantic ring retains one complete
# ratio-128 window, while main stores it in sixteen 8-row state pages.
HCA_MAIN_PAGES_AT_1M = HCA_KV_POOL_BLOCKS
HCA_MAIN_ROW_WIDTH = FLASH.head_dim
HCA_MAIN_BYTES_PER_ROW = HCA_MAIN_ROW_WIDTH * 2
HCA_MAIN_BYTES_AT_1M_PER_REQUEST_PER_LAYER = MAX_HCA_ROWS * HCA_MAIN_BYTES_PER_ROW
HCA_STATE_ROWS_PER_REQUEST = HCA_COMPRESS_RATIO
HCA_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
HCA_STATE_PAGES_PER_REQUEST = (
    HCA_STATE_ROWS_PER_REQUEST + HCA_STATE_BLOCK_SIZE - 1
) // HCA_STATE_BLOCK_SIZE
HCA_STATE_ROW_WIDTH = 2 * FLASH.head_dim
HCA_STATE_BYTES_PER_ROW = HCA_STATE_ROW_WIDTH * 4
HCA_STATE_BYTES_PER_REQUEST_PER_LAYER = HCA_STATE_ROWS_PER_REQUEST * HCA_STATE_BYTES_PER_ROW
HCA_STATE_POOL_BLOCKS = HCA_STATE_PHYSICAL_BLOCKS

# Phase D.1 persistent CSA ABI.  Main candidate rows are 512-wide BF16;
# index rows are 128 INT8 values with one FP32 scale.  Each compressor retains
# the previous/current ratio-4 window (eight rows) in four 2-row state pages.
CSA_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
CSA_INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
CSA_STATE_ROWS_PER_REQUEST = 2 * CSA_COMPRESS_RATIO
CSA_INNER_STATE_ROWS_PER_REQUEST = 2 * CSA_COMPRESS_RATIO
CSA_STATE_PAGES_PER_REQUEST = (
    CSA_STATE_ROWS_PER_REQUEST + CSA_STATE_BLOCK_SIZE - 1
) // CSA_STATE_BLOCK_SIZE
CSA_INNER_STATE_PAGES_PER_REQUEST = (
    CSA_INNER_STATE_ROWS_PER_REQUEST + CSA_INNER_STATE_BLOCK_SIZE - 1
) // CSA_INNER_STATE_BLOCK_SIZE
CSA_STATE_POOL_PAGES = CSA_STATE_PHYSICAL_BLOCKS
CSA_INNER_STATE_POOL_PAGES = CSA_INNER_STATE_PHYSICAL_BLOCKS
CSA_MAIN_STATE_ROW_WIDTH = 2 * FLASH.q_lora_rank
CSA_INNER_STATE_ROW_WIDTH = FLASH.head_dim
CSA_MAIN_STATE_BYTES_PER_ROW = CSA_MAIN_STATE_ROW_WIDTH * 4
CSA_INNER_STATE_BYTES_PER_ROW = CSA_INNER_STATE_ROW_WIDTH * 4
CSA_MAIN_STATE_BYTES_PER_REQUEST_PER_LAYER = (
    CSA_STATE_ROWS_PER_REQUEST * CSA_MAIN_STATE_BYTES_PER_ROW
)
CSA_INNER_STATE_BYTES_PER_REQUEST_PER_LAYER = (
    CSA_INNER_STATE_ROWS_PER_REQUEST * CSA_INNER_STATE_BYTES_PER_ROW
)

CSA_MAIN_ROW_WIDTH = FLASH.head_dim
CSA_MAIN_BYTES_PER_ROW = CSA_MAIN_ROW_WIDTH * 2
CSA_INDEX_ROW_WIDTH = FLASH.index_head_dim
CSA_INDEX_BYTES_PER_ROW = CSA_INDEX_ROW_WIDTH
CSA_INDEX_SCALE_BYTES_PER_ROW = 4
CSA_MAIN_BYTES_AT_1M_PER_REQUEST_PER_LAYER = (
    MAX_CSA_CANDIDATES * CSA_MAIN_BYTES_PER_ROW
)
CSA_INDEX_BYTES_AT_1M_PER_REQUEST_PER_LAYER = (
    MAX_CSA_CANDIDATES * CSA_INDEX_BYTES_PER_ROW
)
CSA_INDEX_SCALE_BYTES_AT_1M_PER_REQUEST_PER_LAYER = (
    MAX_CSA_CANDIDATES * CSA_INDEX_SCALE_BYTES_PER_ROW
)
CSA_PERSISTENT_BYTES_AT_1M_PER_REQUEST_PER_LAYER = (
    CSA_MAIN_BYTES_AT_1M_PER_REQUEST_PER_LAYER
    + CSA_INDEX_BYTES_AT_1M_PER_REQUEST_PER_LAYER
    + CSA_INDEX_SCALE_BYTES_AT_1M_PER_REQUEST_PER_LAYER
    + CSA_MAIN_STATE_BYTES_PER_REQUEST_PER_LAYER
    + CSA_INNER_STATE_BYTES_PER_REQUEST_PER_LAYER
)

# Phase B SWA ABI: four allocator-owned 32-row pages per admitted request form
# the semantic 128-row ring. Logical context length changes descriptor state
# only; it never changes the four-page capacity.
SWA_PERSISTENT_PAGES_PER_REQUEST = (
    SWA_WINDOW_ROWS + CACHE_BLOCK_SIZE - 1
) // CACHE_BLOCK_SIZE
SWA_PERSISTENT_ROWS_PER_REQUEST = SWA_PERSISTENT_PAGES_PER_REQUEST * CACHE_BLOCK_SIZE
SWA_ATTENTION_K_TILE = SWA_WINDOW_ROWS
SWA_KV_ROW_WIDTH = FLASH.head_dim
SWA_KV_BYTES_PER_ROW = SWA_KV_ROW_WIDTH * 2
SWA_PERSISTENT_BYTES_PER_REQUEST_PER_LAYER = (
    SWA_PERSISTENT_ROWS_PER_REQUEST * SWA_KV_BYTES_PER_ROW
)
CSA_PERSISTENT_BYTES_WITH_SWA_AT_1M_PER_REQUEST_PER_LAYER = (
    CSA_PERSISTENT_BYTES_AT_1M_PER_REQUEST_PER_LAYER
    + SWA_PERSISTENT_BYTES_PER_REQUEST_PER_LAYER
)

# SWA source ABI. Non-negative values are flattened persistent-cache rows,
# -1 is invalid/padding, and values <= -2 encode current-step KV rows.
SWA_SOURCE_INVALID = -1
SWA_SOURCE_OVERLAY_BASE = -2
SWA_SOURCE_INT32_MIN = -(1 << 31)
SWA_SOURCE_INT32_MAX = (1 << 31) - 1
SWA_SOURCE_MAX_OVERLAY_QUERY = -SWA_SOURCE_INT32_MIN + SWA_SOURCE_OVERLAY_BASE

def encode_swa_overlay_source(query_index: int) -> int:
    """Encode one current-step query index in the negative SWA source space."""
    if query_index < 0 or query_index > SWA_SOURCE_MAX_OVERLAY_QUERY:
        raise ValueError(
            "overlay query index must fit the SWA INT32 source ABI, got "
            f"{query_index}"
        )
    return SWA_SOURCE_OVERLAY_BASE - query_index


def decode_swa_overlay_source(source: int) -> Optional[int]:
    """Return the current-step query index encoded by ``source``, if any."""
    if source < SWA_SOURCE_INT32_MIN or source > SWA_SOURCE_OVERLAY_BASE:
        return None
    return SWA_SOURCE_OVERLAY_BASE - source


def is_swa_persistent_source(source: int) -> bool:
    """Whether ``source`` is a non-negative INT32 persistent-cache row."""
    return 0 <= source <= SWA_SOURCE_INT32_MAX


def is_swa_overlay_source(source: int) -> bool:
    """Whether ``source`` encodes a current-step KV overlay row."""
    return SWA_SOURCE_INT32_MIN <= source <= SWA_SOURCE_OVERLAY_BASE

# Physical pool capacity is a global property of each cache group and must not
# be derived from DECODE_BATCH. The legacy pool constants above
# (KV_ORI_BLOCK_NUM, KV_CMP_BLOCK_NUM, IDX_CACHE_BLOCK_NUM, ...) are literal
# globals and already B-independent; Phase A does not redefine them. From
# Phase B/C/D the leaves stop deriving work shape from
# FLASH.max_position_embeddings and the legacy pools are superseded by the 1M
# capacity contract.

# Invariants — fail fast at import if the geometry is violated.
assert FLASH.max_position_embeddings == 1_048_576, "FLASH context ceiling must be 1M"
assert FLASH.max_position_embeddings % HCA_COMPRESS_RATIO == 0, "1M must divide by HCA ratio"
assert FLASH.max_position_embeddings % CSA_COMPRESS_RATIO == 0, "1M must divide by CSA ratio"
assert MAX_HCA_ROWS % HCA_ROWS_PER_SHARD == 0, "HCA rows must fill whole shards"
assert MAX_CSA_CANDIDATES % CSA_CANDIDATES_PER_LEAF == 0, "CSA candidates must fill whole leaves"
assert CSA_TOPK <= CSA_CANDIDATES_PER_LEAF, "Top-K must fit in one leaf"
assert CSA_MERGE_ARITY == 2
assert CSA_TOPK_READY_FRONTIER_W == TOPK_READY_FRONTIER_W == 8
assert CSA_MAX_CANDIDATES == MAX_CSA_CANDIDATES == 262144
assert CSA_MAX_LEAVES_PER_QUERY == MAX_CSA_LEAVES == 128
assert CSA_MAX_NODES_PER_QUERY == 255
assert CSA_MAX_TOPK_TASKS == CSA_MAX_QUERIES * CSA_MAX_NODES_PER_QUERY
assert CSA_TOPK_INVALID_INDEX == -1
assert CSA_TOPK_INVALID_TASK_SLOT == CSA_MAX_TOPK_TASKS
assert CSA_PAIR_WIDTH == 1024 and CSA_PAIR_BYTES == 4096
assert CSA_SCORE_INDEX_PAIR_BYTES == 8
assert CSA_LOGICAL_INDEX_MIN == 0
assert CSA_LOGICAL_INDEX_MAX == MAX_CSA_CANDIDATES - 1
assert CACHE_BLOCK_SIZE == BLOCK_SIZE, "canonical cache block must match storage ABI"
assert SWA_WINDOW_ROWS == FLASH.sliding_window, "SWA window must match model semantic"
assert HCA_COMPRESS_RATIO == 128 and CSA_COMPRESS_RATIO == 4, "compress ratios are model semantics"
assert SWA_KV_POOL_BLOCKS * CACHE_BLOCK_SIZE >= SWA_WINDOW_ROWS
assert SWA_KV_POOL_BLOCKS == SWA_PERSISTENT_PAGES_PER_REQUEST
assert SWA_PERSISTENT_ROWS_PER_REQUEST == SWA_WINDOW_ROWS == SWA_ATTENTION_K_TILE
assert SWA_KV_ROW_WIDTH == FLASH.head_dim == 512
assert SWA_PERSISTENT_BYTES_PER_REQUEST_PER_LAYER == 128 * 512 * 2
assert SWA_SOURCE_INVALID == -1 and SWA_SOURCE_OVERLAY_BASE == -2
assert SWA_SOURCE_INVALID < 0 and SWA_SOURCE_OVERLAY_BASE < SWA_SOURCE_INVALID
assert encode_swa_overlay_source(0) == SWA_SOURCE_OVERLAY_BASE
assert decode_swa_overlay_source(SWA_SOURCE_OVERLAY_BASE) == 0
assert encode_swa_overlay_source(SWA_SOURCE_MAX_OVERLAY_QUERY) >= SWA_SOURCE_INT32_MIN
assert is_swa_persistent_source(0)
assert not is_swa_persistent_source(SWA_SOURCE_INVALID)
assert is_swa_overlay_source(SWA_SOURCE_OVERLAY_BASE)
assert not is_swa_overlay_source(SWA_SOURCE_INVALID)
assert HCA_KV_POOL_BLOCKS * CACHE_BLOCK_SIZE >= MAX_HCA_ROWS
assert HCA_MAIN_PAGES_AT_1M == 256
assert HCA_MAIN_BYTES_AT_1M_PER_REQUEST_PER_LAYER == 8 * 1024 * 1024
assert HCA_STATE_PAGES_PER_REQUEST == 16
assert HCA_STATE_ROWS_PER_REQUEST == 128 and HCA_STATE_BLOCK_SIZE == 8
assert HCA_STATE_ROW_WIDTH == 1024
assert HCA_STATE_BYTES_PER_REQUEST_PER_LAYER == 512 * 1024
assert isinstance(HCA_STATE_POOL_BLOCKS, int) and not callable(HCA_STATE_POOL_BLOCKS)
assert CSA_KV_POOL_BLOCKS * CACHE_BLOCK_SIZE >= MAX_CSA_CANDIDATES
assert CSA_IDX_POOL_BLOCKS * CACHE_BLOCK_SIZE >= MAX_CSA_CANDIDATES
assert CSA_MAIN_POOL_PAGES == CSA_MAIN_PAGES_AT_1M == 8192
assert CSA_IDX_POOL_PAGES == CSA_INDEX_POOL_PAGES == CSA_IDX_PAGES_AT_1M == 8192
assert CSA_STATE_PAGES_PER_REQUEST == CSA_INNER_STATE_PAGES_PER_REQUEST == 4
assert CSA_STATE_BLOCK_SIZE == CSA_INNER_STATE_BLOCK_SIZE == 2
assert CSA_STATE_ROWS_PER_REQUEST == CSA_INNER_STATE_ROWS_PER_REQUEST == 8
assert CSA_MAIN_STATE_ROW_WIDTH == 2048
assert CSA_INNER_STATE_ROW_WIDTH == 512
assert CSA_MAIN_STATE_BYTES_PER_REQUEST_PER_LAYER == 64 * 1024
assert CSA_INNER_STATE_BYTES_PER_REQUEST_PER_LAYER == 16 * 1024
assert CSA_MAIN_BYTES_AT_1M_PER_REQUEST_PER_LAYER == 256 * 1024 * 1024
assert CSA_INDEX_BYTES_AT_1M_PER_REQUEST_PER_LAYER == 32 * 1024 * 1024
assert CSA_INDEX_SCALE_BYTES_AT_1M_PER_REQUEST_PER_LAYER == 1 * 1024 * 1024
assert CSA_PERSISTENT_BYTES_AT_1M_PER_REQUEST_PER_LAYER == 303120384
assert CSA_PERSISTENT_BYTES_WITH_SWA_AT_1M_PER_REQUEST_PER_LAYER == 303251456
assert isinstance(CSA_MAIN_POOL_PAGES, int) and not callable(CSA_MAIN_POOL_PAGES)
assert isinstance(CSA_INDEX_POOL_PAGES, int) and not callable(CSA_INDEX_POOL_PAGES)
assert isinstance(CSA_STATE_POOL_PAGES, int) and not callable(CSA_STATE_POOL_PAGES)
assert isinstance(CSA_INNER_STATE_POOL_PAGES, int) and not callable(CSA_INNER_STATE_POOL_PAGES)
# Pool constants must stay B-independent (literal globals, not B-derived).
assert isinstance(KV_ORI_BLOCK_NUM, int) and not callable(KV_ORI_BLOCK_NUM)
assert isinstance(KV_CMP_BLOCK_NUM, int) and not callable(KV_CMP_BLOCK_NUM)
assert isinstance(IDX_CACHE_BLOCK_NUM, int) and not callable(IDX_CACHE_BLOCK_NUM)
