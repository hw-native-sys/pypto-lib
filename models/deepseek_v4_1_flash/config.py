# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1-Flash model, deployment, and kernel-shape configuration."""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple

import pypto.language as pl


class AttentionKind(str, Enum):
    """Attention families required by the text backbone."""

    SLIDING_WINDOW = "sliding_window"
    COMPRESSED_SPARSE = "compressed_sparse"


class AttentionMode(str, Enum):
    """Static attention implementation selected for one backbone layer."""

    SWA = "swa"
    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


@dataclass(frozen=True)
class DeepSeekV41LayerConfig:
    """Resolved ownership and attention mode for one text-backbone layer."""

    layer_id: int
    attention_kind: AttentionKind
    compression_ratio: int
    kv_source_layer_id: Optional[int]
    index_source_layer_id: Optional[int]
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    mode: AttentionMode


@dataclass(frozen=True)
class DeepSeekV41Config:
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
    scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"]
    gate_temperature: float
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
    dtype: Literal["bfloat16"]
    quant_method: Literal["fp8"]
    activation_scheme: Literal["dynamic"]
    weight_block_size: Tuple[int, int]
    expert_dtype: Optional[Literal["fp4"]]
    scale_fmt: Optional[Literal["ue8m0"]]

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_layer_schedule()

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    def layer_config(self, layer_id: int) -> DeepSeekV41LayerConfig:
        """Resolve the cache owners and attention family visible to one layer."""
        if not 0 <= layer_id < self.num_hidden_layers:
            raise ValueError(f"layer_id must be in [0, {self.num_hidden_layers - 1}], got {layer_id}")

        ratio = self.compress_ratios[layer_id]
        if ratio == 0:
            attention_kind = AttentionKind.SLIDING_WINDOW
        else:
            attention_kind = AttentionKind.COMPRESSED_SPARSE

        kv_source = self._active_source(layer_id, ratio, self.kv_source_layer_ids)
        index_source = self._active_source(layer_id, ratio, self.index_source_layer_ids)
        if ratio == 0:
            mode = AttentionMode.SWA
        elif layer_id in self.kv_source_layer_ids:
            mode = AttentionMode.FULL
        elif layer_id in self.index_source_layer_ids:
            mode = AttentionMode.REINDEX
        else:
            mode = AttentionMode.REUSE
        return DeepSeekV41LayerConfig(
            layer_id=layer_id,
            attention_kind=attention_kind,
            compression_ratio=ratio,
            kv_source_layer_id=kv_source,
            index_source_layer_id=index_source,
            is_kv_source=layer_id in self.kv_source_layer_ids,
            is_index_source=layer_id in self.index_source_layer_ids,
            is_candidate_source=layer_id == self.candidate_source_layer_id,
            mode=mode,
        )

    def backbone_layers(self) -> Tuple[DeepSeekV41LayerConfig, ...]:
        return tuple(self.layer_config(layer_id) for layer_id in range(self.num_hidden_layers))

    def _active_source(self, layer_id: int, ratio: int, sources: Tuple[int, ...]) -> Optional[int]:
        if ratio == 0:
            return None
        matching = [
            source for source in sources if source <= layer_id and self.compress_ratios[source] == ratio
        ]
        return matching[-1] if matching else None

    def _validate_dimensions(self) -> None:
        if self.hidden_size <= 0 or self.vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if self.head_dim <= self.qk_rope_head_dim:
            raise ValueError("head_dim must be larger than qk_rope_head_dim")
        if self.index_head_dim < self.qk_rope_head_dim:
            raise ValueError("index_head_dim must be at least qk_rope_head_dim")
        if self.n_routed_experts < self.num_experts_per_tok:
            raise ValueError("n_routed_experts must cover num_experts_per_tok")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("engram layers and embedding tables must have the same length")
        if any(not 0 <= layer_id < self.num_hidden_layers for layer_id in self.engram_layer_ids):
            raise ValueError("engram layer ids must refer to backbone layers")

    def _validate_layer_schedule(self) -> None:
        if len(self.compress_ratios) != self.num_hidden_layers:
            raise ValueError("compress_ratios must contain every backbone layer")
        if set(self.compress_ratios) - {0, 1, 2}:
            raise ValueError("DeepSeek-V4.1-Flash supports compression ratios 0, 1, and 2")

        for source in self.kv_source_layer_ids + self.index_source_layer_ids:
            if not 0 <= source < self.num_hidden_layers:
                raise ValueError(f"cache source layer {source} is outside the backbone")
            if self.compress_ratios[source] == 0:
                raise ValueError(f"cache source layer {source} has no compressed attention")

        for layer_id in range(self.num_hidden_layers):
            ratio = self.compress_ratios[layer_id]
            if ratio == 0:
                continue
            if self._active_source(layer_id, ratio, self.kv_source_layer_ids) is None:
                raise ValueError(f"compressed layer {layer_id} has no matching KV source")
            if self._active_source(layer_id, ratio, self.index_source_layer_ids) is None:
                raise ValueError(f"compressed layer {layer_id} has no matching index source")

        if self.candidate_source_layer_id not in self.index_source_layer_ids:
            raise ValueError("candidate_source_layer_id must also be an index source")
        if self.candidate_source_layer_id not in self.kv_source_layer_ids:
            raise ValueError("candidate_source_layer_id must share the compressed cache it filters")


FLASH = DeepSeekV41Config(
    name="deepseek_v4_1_flash",
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
    gate_temperature=1.0,
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
    dtype="bfloat16",
    quant_method="fp8",
    activation_scheme="dynamic",
    weight_block_size=(32, 32),
    expert_dtype="fp4",
    scale_fmt="ue8m0",
)


T_DYN = pl.dynamic("V41_T_DYN")
ROUTE_T_DYN = pl.dynamic("V41_ROUTE_T_DYN")
B_DYN = pl.dynamic("V41_B_DYN")
Q_START_DYN = pl.dynamic("V41_Q_START_DYN")
TABLE_DYN = pl.dynamic("V41_TABLE_DYN")
ORI_BLOCKS_DYN = pl.dynamic("V41_ORI_BLOCKS_DYN")
CMP_BLOCKS_DYN = pl.dynamic("V41_CMP_BLOCKS_DYN")
INDEX_BLOCKS_DYN = pl.dynamic("V41_INDEX_BLOCKS_DYN")
CMP_POSITIONS_DYN = pl.dynamic("V41_CMP_POSITIONS_DYN")
RECV_DYN = pl.dynamic("V41_RECV_DYN")
LOGIT_ROWS_DYN = pl.dynamic("V41_LOGIT_ROWS_DYN")

D = FLASH.hidden_size
H = FLASH.num_attention_heads
HEAD_DIM = FLASH.head_dim
ROPE_DIM = FLASH.qk_rope_head_dim
NOPE_DIM = FLASH.nope_head_dim
Q_LORA = FLASH.q_lora_rank
O_LORA = FLASH.o_lora_rank
O_GROUPS = FLASH.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
HC_MULT = FLASH.hc_mult
HC_DIM = FLASH.hc_dim
MIX_HC = FLASH.mix_hc
INDEX_H = FLASH.index_n_heads
INDEX_DIM = FLASH.index_head_dim
INDEX_TOPK = FLASH.index_topk
N_EXPERTS = FLASH.n_routed_experts
TOPK = FLASH.num_experts_per_tok
MOE_INTER = FLASH.moe_intermediate_size
VOCAB = FLASH.vocab_size
MX_GROUP = FLASH.weight_block_size[0]

SUPPORTED_TP_SIZES = (1, 2, 4, 8)
SUPPORTED_EP_SIZES = (2, 4, 8)


def _parse_parallel_size(name: str, default: int) -> int:
    flag = f"--{name}"
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1])
    return default


TP_SIZE = _parse_parallel_size("tp", 4)
EP_SIZE = _parse_parallel_size("ep", 8)
if TP_SIZE not in SUPPORTED_TP_SIZES:
    raise ValueError(f"--tp must be one of {SUPPORTED_TP_SIZES}, got {TP_SIZE}")
if EP_SIZE not in SUPPORTED_EP_SIZES:
    raise ValueError(f"--ep must be one of {SUPPORTED_EP_SIZES}, got {EP_SIZE}")
if EP_SIZE % TP_SIZE:
    raise ValueError(f"EP{EP_SIZE} must be divisible by TP{TP_SIZE}")
if H % TP_SIZE:
    raise ValueError(f"{H} attention heads cannot be evenly sharded across TP{TP_SIZE}")
if O_GROUPS % TP_SIZE:
    raise ValueError(f"{O_GROUPS} output groups cannot be evenly sharded across TP{TP_SIZE}")
if N_EXPERTS % EP_SIZE:
    raise ValueError(f"{N_EXPERTS} routed experts cannot be evenly sharded across EP{EP_SIZE}")

DP_SIZE = EP_SIZE // TP_SIZE
LOCAL_H = H // TP_SIZE
LOCAL_O_GROUPS = O_GROUPS // TP_SIZE
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA
N_LOCAL_EXPERTS = N_EXPERTS // EP_SIZE

BLOCK_SIZE = 128
RATIO2_STORAGE_BLOCK_SIZE = BLOCK_SIZE
RATIO1_STORAGE_BLOCK_SIZE = BLOCK_SIZE
STATE_HEADS = 2
MAX_BATCH_PER_DP = 32
DSPARK_SPEC_TOKENS = 5
DECODE_ROWS_PER_REQUEST = DSPARK_SPEC_TOKENS + 1
DECODE_MAX_TOKENS = MAX_BATCH_PER_DP * DECODE_ROWS_PER_REQUEST
PREFILL_MAX_TOKENS = 4096
DECODE_RECV_MAX = DP_SIZE * DECODE_MAX_TOKENS
PREFILL_RECV_MAX = DP_SIZE * PREFILL_MAX_TOKENS
MOE_TOKENS = 16
# Each EP source contributes at most one block to a given expert per round.
RECV_MAX = EP_SIZE * MOE_TOKENS
AUX_WIDTH = 8
ROUTE_WIDTH = 8
WINDOW_CACHE_GROUP = 32
COMPRESSED_CACHE_GROUP = 16
INDEX_CACHE_GROUP = 32
