# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1-Flash A3 W8A8 model, deployment, and quantization configuration."""

from dataclasses import dataclass
import sys
from typing import Literal, Optional, Tuple

import pypto.language as pl


@dataclass(frozen=True)
class LayerConfig:
    """Resolved cache owners and attention family for one text-backbone layer."""

    layer_id: int
    compression_ratio: int
    kv_source_layer_id: Optional[int]
    index_source_layer_id: Optional[int]
    is_kv_source: bool
    mode: Literal["swa", "c1a_full", "c1a_reindex", "c1a_reuse", "c2a_full", "c2a_reuse"]


@dataclass(frozen=True)
class ModelConfig:
    """Text-backbone configuration mirrored from the released checkpoint."""

    name: str
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    sliding_window: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_max_position_embeddings: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: Literal["sqrtsoftplus"]
    norm_topk_prob: bool
    routed_scaling_factor: float
    swiglu_limit: float
    compress_ratios: Tuple[int, ...]
    kv_source_layer_ids: Tuple[int, ...]
    index_source_layer_ids: Tuple[int, ...]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: Tuple[int, ...]
    engram_num_embeddings: Tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_token_id: int
    engram_compressed_vocab_size: int

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    def layer_config(self, layer_id: int) -> LayerConfig:
        if not 0 <= layer_id < self.num_hidden_layers:
            raise ValueError(f"layer_id must be in [0, {self.num_hidden_layers - 1}], got {layer_id}")
        ratio = self.compress_ratios[layer_id]
        kv_source = self._active_source(layer_id, ratio, self.kv_source_layer_ids)
        index_source = self._active_source(layer_id, ratio, self.index_source_layer_ids)
        if ratio == 0:
            mode = "swa"
        elif layer_id in self.kv_source_layer_ids:
            mode = "c1a_full" if ratio == 1 else "c2a_full"
        elif layer_id in self.index_source_layer_ids:
            mode = "c1a_reindex"
        else:
            mode = "c1a_reuse" if ratio == 1 else "c2a_reuse"
        return LayerConfig(layer_id, ratio, kv_source, index_source, layer_id in self.kv_source_layer_ids, mode)

    def _active_source(self, layer_id: int, ratio: int, sources: Tuple[int, ...]) -> Optional[int]:
        if ratio == 0:
            return None
        matching = [source for source in sources if source <= layer_id and self.compress_ratios[source] == ratio]
        return matching[-1] if matching else None


FLASH = ModelConfig(
    name="deepseek_v4_1_flash_a3_w8a8",
    vocab_size=129280,
    hidden_size=5120,
    moe_intermediate_size=2304,
    num_hidden_layers=40,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1280,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    rms_norm_eps=1e-20,
    max_position_embeddings=1048576,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=16.0,
    beta_fast=32,
    beta_slow=1,
    original_max_position_embeddings=65536,
    n_routed_experts=384,
    n_shared_experts=1,
    num_experts_per_tok=6,
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20,
    kv_source_layer_ids=(2, 8, 14, 20),
    index_source_layer_ids=(2, 8, 14, 20, 24, 28, 32, 36),
    index_n_heads=32,
    index_head_dim=128,
    index_topk=512,
    candidate_source_layer_id=20,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=(1, 14),
    engram_num_embeddings=(384006168, 384016682),
    engram_max_ngram_size=4,
    engram_vocab_size=16000000,
    engram_n_heads=8,
    engram_head_dim=256,
    engram_pad_token_id=2,
    engram_compressed_vocab_size=99092,
)

# Colocated tensor/data parallel groups and whole-expert parallel ranks.
SUPPORTED_TP_SIZES = (1, 2, 4, 8)
SUPPORTED_EP_SIZES = (2, 4, 8, 16)


def _parse_parallel_size(name: str, default: int) -> int:
    flag = f"--{name}"
    values = []
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument == flag:
            if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
                raise ValueError(f"{flag} requires an integer value")
            value = sys.argv[index + 1]
        elif argument.startswith(f"{flag}="):
            value = argument.split("=", 1)[1]
        else:
            continue
        try:
            values.append(int(value))
        except ValueError as error:
            raise ValueError(f"{flag} requires an integer value, got {value!r}") from error
    if values and any(value != values[0] for value in values):
        raise ValueError(f"conflicting repeated {flag} values: {values}")
    return values[0] if values else default


TP_SIZE = _parse_parallel_size("tp", 4)
EP_SIZE = _parse_parallel_size("ep", 16)
if TP_SIZE not in SUPPORTED_TP_SIZES:
    raise ValueError(f"--tp must be one of {SUPPORTED_TP_SIZES}, got {TP_SIZE}")
if EP_SIZE not in SUPPORTED_EP_SIZES:
    raise ValueError(f"--ep must be one of {SUPPORTED_EP_SIZES}, got {EP_SIZE}")
if EP_SIZE < TP_SIZE or EP_SIZE % TP_SIZE:
    raise ValueError(f"--ep {EP_SIZE} must be divisible by --tp {TP_SIZE} and at least TP")
if any(width % TP_SIZE for width in (FLASH.num_attention_heads, FLASH.o_groups, FLASH.index_n_heads, FLASH.vocab_size)):
    raise ValueError(f"attention/index heads, output groups and vocabulary must divide TP{TP_SIZE}")
if FLASH.n_routed_experts % EP_SIZE:
    raise ValueError(f"routed experts must divide EP{EP_SIZE}")
WORLD_SIZE = EP_SIZE
DP_SIZE = EP_SIZE // TP_SIZE
N_LOCAL_EXPERTS = FLASH.n_routed_experts // EP_SIZE


def add_parallel_arguments(parser):
    """Register the import-time topology flags on an executable parser."""
    parser.add_argument("--tp", type=int, choices=SUPPORTED_TP_SIZES, default=TP_SIZE)
    parser.add_argument("--ep", type=int, choices=SUPPORTED_EP_SIZES, default=EP_SIZE)
    return parser


def parallel_topology() -> dict:
    """Return the resolved colocated rank topology."""
    return {"tp": TP_SIZE, "ep": EP_SIZE, "dp": DP_SIZE, "world": WORLD_SIZE, "local_experts": N_LOCAL_EXPERTS}


# Attention tensor-parallel shards.
LOCAL_HEADS = FLASH.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = FLASH.o_groups // TP_SIZE
LOCAL_KV_ROWS = FLASH.head_dim

# Quantization: W8A8 dynamic, per-output-channel weights, per-token activations.
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
WEIGHT_GROUP_NONE = 0

# Paged state.
BLOCK_SIZE = 128
MAX_BATCH_PER_DP = 32
PREFILL_MAX_TOKENS = 8192
DECODE_MAX_TOKENS = MAX_BATCH_PER_DP
WINDOW_KEYS = FLASH.sliding_window
SELECTED_KEYS = WINDOW_KEYS + FLASH.index_topk
STATE_CAPACITY = 2
CANDIDATE_POOL_BLOCKS = FLASH.candidate_topk_blocks

# KV/index/state cache owners (one cache slice per source).
KV_SOURCES: Tuple[int, ...] = (2, 8, 14, 20)
C2A_SOURCES: Tuple[int, ...] = (2, 8, 14)
C1A_SOURCES: Tuple[int, ...] = (20,)
REINDEX_LAYERS: Tuple[int, ...] = (24, 28, 32, 36)

# Dynamic shape symbols shared by every migrated module.
T_DYN = pl.dynamic("V41_A3_T_DYN")
B_DYN = pl.dynamic("V41_A3_B_DYN")
TABLE_DYN = pl.dynamic("V41_A3_TABLE_DYN")
ORI_BLOCKS_DYN = pl.dynamic("V41_A3_ORI_BLOCKS_DYN")
CMP_BLOCKS_DYN = pl.dynamic("V41_A3_CMP_BLOCKS_DYN")
INDEX_BLOCKS_DYN = pl.dynamic("V41_A3_INDEX_BLOCKS_DYN")
STATE_BLOCKS_DYN = pl.dynamic("V41_A3_STATE_BLOCKS_DYN")
POOL_BLOCKS_DYN = pl.dynamic("V41_A3_POOL_BLOCKS_DYN")
R_DYN = pl.dynamic("V41_A3_R_DYN")
SCORE_DYN = pl.dynamic("V41_A3_IDX_SCORE_DYN")

# Engram storage (ground truth: quant_model_description.json embedding_storage).
ENGRAM_GROUP_SIZE = 32
ENGRAM_OUTPUT_DTYPE = "bfloat16"

assert FLASH.num_attention_heads % TP_SIZE == 0
assert FLASH.o_groups % TP_SIZE == 0
assert FLASH.n_routed_experts % EP_SIZE == 0
assert len(FLASH.compress_ratios) == FLASH.num_hidden_layers
assert FLASH.head_dim > FLASH.qk_rope_head_dim
assert FLASH.engram_head_dim % ENGRAM_GROUP_SIZE == 0
