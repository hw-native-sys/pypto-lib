# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash model, deployment, and kernel-shape configuration."""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Tuple

import pypto.language as pl


class AttentionKind(str, Enum):
    """Attention families in the GLM-5.3-Flash hybrid backbone."""

    LINEAR = "linear_attention"
    SPARSE = "deepseek_sparse_attention"


class MlpKind(str, Enum):
    """Feed-forward families selected per layer."""

    DENSE = "dense"
    SPARSE = "sparse"


@dataclass(frozen=True)
class Glm53LayerConfig:
    """Resolved attention family, MLP family and residual style for one layer."""

    layer_id: int
    attention_kind: AttentionKind
    mlp_kind: MlpKind
    has_indexer: bool
    has_hyper_connections: bool
    is_mtp: bool


@dataclass(frozen=True)
class Glm53FlashConfig:
    """Text-backbone configuration mirrored from the released checkpoint.

    Field values come from ``zai-org/GLM-5.3-Flash`` ``config.json`` (``text_config``)
    and were cross-checked against ``model.safetensors.index.json``. The checkpoint
    carries 46 layers: ``num_hidden_layers`` backbone layers plus one MTP layer.
    """

    name: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    first_k_dense_replace: int
    intermediate_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    swiglu_limit: float
    tie_word_embeddings: bool

    # NoPE MLA (the deepseek_sparse_attention layers and the MTP layer).
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    # kpool DSA indexer.
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_compress: bool
    index_kpool_always_select_tail: bool
    index_share_for_mtp_iteration: bool

    # KDA linear attention (``linear_attn_config``).
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float

    # MoE.
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    scoring_func: Literal["sigmoid"]
    topk_method: Literal["noaux_tc"]
    norm_topk_prob: bool
    routed_scaling_factor: float
    n_group: int
    topk_group: int

    # Manifold-constrained hyper-connections.
    mhc: bool
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float

    # Multi-token prediction.
    num_nextn_predict_layers: int

    # Deployment dtype and quantization. The a2a3 cube has no fp8/MX mmad
    # (``Intrinsic_mmad`` offers only ``s32s8s8`` and the fp16/fp32 forms), so the
    # released FP8-blockwise checkpoint cannot run here: the deployment target is
    # the msmodelslim W8A8 checkpoint, matching vLLM Ascend's 16-card A3 recipe.
    dtype: Literal["bfloat16"]
    quant_method: Literal["w8a8"]
    activation_scheme: Literal["dynamic"]

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_layer_schedule()

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def mix_hc(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @property
    def mtp_layer_id(self) -> int:
        return self.num_hidden_layers

    @property
    def num_checkpoint_layers(self) -> int:
        return self.num_hidden_layers + self.num_nextn_predict_layers

    @property
    def kda_layer_ids(self) -> Tuple[int, ...]:
        return tuple(
            layer_id
            for layer_id in range(self.num_hidden_layers)
            if layer_id % 4 != 3
        )

    @property
    def dsa_layer_ids(self) -> Tuple[int, ...]:
        """Backbone layers with NoPE MLA and a kpool indexer (3, 7, 11, ...)."""
        return tuple(
            layer_id
            for layer_id in range(self.num_hidden_layers)
            if layer_id % 4 == 3
        )

    @property
    def indexer_layer_ids(self) -> Tuple[int, ...]:
        """Every layer that owns indexer weights, including the MTP layer."""
        return self.dsa_layer_ids + (self.mtp_layer_id,)

    @property
    def kpool_select_k(self) -> int:
        """Pools selected per query. Raw indices are ``kpool_select_k * index_kpool``."""
        return self.index_topk // self.index_kpool

    @property
    def topk_index_width(self) -> int:
        """Width of the emitted index list: the expanded pools plus the tail."""
        return self.index_topk + self.index_kpool - 1

    def layer_config(self, layer_id: int) -> Glm53LayerConfig:
        """Resolve the attention family, MLP family and residual style of one layer."""
        if not 0 <= layer_id <= self.mtp_layer_id:
            raise ValueError(f"layer_id must be in [0, {self.mtp_layer_id}], got {layer_id}")

        is_mtp = layer_id == self.mtp_layer_id
        if is_mtp:
            # The MTP layer is an MLA layer with its own indexer and a sparse MoE,
            # and it carries no hc_* weights, so it uses a plain residual.
            return Glm53LayerConfig(
                layer_id=layer_id,
                attention_kind=AttentionKind.SPARSE,
                mlp_kind=MlpKind.SPARSE,
                has_indexer=True,
                has_hyper_connections=False,
                is_mtp=True,
            )

        attention_kind = (
            AttentionKind.SPARSE if layer_id in self.dsa_layer_ids else AttentionKind.LINEAR
        )
        mlp_kind = MlpKind.DENSE if layer_id < self.first_k_dense_replace else MlpKind.SPARSE
        return Glm53LayerConfig(
            layer_id=layer_id,
            attention_kind=attention_kind,
            mlp_kind=mlp_kind,
            has_indexer=attention_kind is AttentionKind.SPARSE,
            has_hyper_connections=True,
            is_mtp=False,
        )

    def backbone_layers(self) -> Tuple[Glm53LayerConfig, ...]:
        return tuple(self.layer_config(layer_id) for layer_id in range(self.num_hidden_layers))

    def _validate_dimensions(self) -> None:
        if self.hidden_size <= 0 or self.vocab_size <= 0:
            raise ValueError("hidden_size and vocab_size must be positive")
        if self.qk_rope_head_dim:
            raise ValueError(
                "GLM-5.3-Flash MLA is NoPE; qk_rope_head_dim must be 0, "
                f"got {self.qk_rope_head_dim}"
            )
        if self.kv_lora_rank <= 0 or self.q_lora_rank <= 0:
            raise ValueError("both LoRA ranks are required for the DSA attention layers")
        if self.n_routed_experts < self.num_experts_per_tok:
            raise ValueError("n_routed_experts must cover num_experts_per_tok")
        if self.linear_num_heads * self.linear_head_dim <= 0:
            raise ValueError("the linear-attention head layout must be positive")
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be at least 1")

    def _validate_layer_schedule(self) -> None:
        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}")
        if self.index_topk % self.index_kpool:
            raise ValueError(
                f"index_topk ({self.index_topk}) must be divisible by "
                f"index_kpool ({self.index_kpool})"
            )
        if self.first_k_dense_replace > self.num_hidden_layers:
            raise ValueError("first_k_dense_replace exceeds the backbone depth")
        if len(self.kda_layer_ids) + len(self.dsa_layer_ids) != self.num_hidden_layers:
            raise ValueError("the hybrid attention schedule must cover every backbone layer")
        if self.num_nextn_predict_layers != 1:
            raise ValueError("only a single MTP layer is supported")


FLASH = Glm53FlashConfig(
    name="glm5_3_flash",
    vocab_size=154880,
    hidden_size=4096,
    num_hidden_layers=45,
    first_k_dense_replace=3,
    intermediate_size=12288,
    rms_norm_eps=1e-5,
    max_position_embeddings=1048576,
    swiglu_limit=10.0,
    tie_word_embeddings=False,
    num_attention_heads=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=256,
    qk_rope_head_dim=0,
    v_head_dim=256,
    index_n_heads=32,
    index_head_dim=128,
    index_topk=2048,
    index_kpool=4,
    index_kpool_compress=True,
    index_kpool_always_select_tail=True,
    index_share_for_mtp_iteration=True,
    linear_num_heads=64,
    linear_head_dim=128,
    linear_conv_kernel_dim=4,
    linear_lower_bound=-5.0,
    n_routed_experts=288,
    n_shared_experts=1,
    num_experts_per_tok=8,
    moe_intermediate_size=2048,
    scoring_func="sigmoid",
    topk_method="noaux_tc",
    norm_topk_prob=True,
    routed_scaling_factor=2.5,
    n_group=1,
    topk_group=1,
    mhc=True,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    num_nextn_predict_layers=1,
    dtype="bfloat16",
    quant_method="w8a8",
    activation_scheme="dynamic",
)


T_DYN = pl.dynamic("GLM53_T_DYN")
B_DYN = pl.dynamic("GLM53_B_DYN")
Q_START_DYN = pl.dynamic("GLM53_Q_START_DYN")
# Total physical blocks in one paged pool. The per-request block table is a
# different quantity and gets its own symbol, or a signature carrying both
# would assert they are equal.
TABLE_DYN = pl.dynamic("GLM53_TABLE_DYN")
BLOCK_TABLE_DYN = pl.dynamic("GLM53_BLOCK_TABLE_DYN")
KV_BLOCKS_DYN = pl.dynamic("GLM53_KV_BLOCKS_DYN")
INDEX_BLOCKS_DYN = pl.dynamic("GLM53_INDEX_BLOCKS_DYN")
POOLS_DYN = pl.dynamic("GLM53_POOLS_DYN")
# Merged vision tokens in one batch: one row per placeholder, unrelated to the
# text token count.
VISION_ROWS_DYN = pl.dynamic("GLM53_VISION_ROWS_DYN")
RECV_DYN = pl.dynamic("GLM53_RECV_DYN")
LOGIT_ROWS_DYN = pl.dynamic("GLM53_LOGIT_ROWS_DYN")

D = FLASH.hidden_size
VOCAB = FLASH.vocab_size
DENSE_INTER = FLASH.intermediate_size

# NoPE MLA.
H = FLASH.num_attention_heads
QK_DIM = FLASH.qk_head_dim
V_DIM = FLASH.v_head_dim
Q_LORA = FLASH.q_lora_rank
KV_LORA = FLASH.kv_lora_rank

# kpool DSA indexer.
INDEX_H = FLASH.index_n_heads
INDEX_DIM = FLASH.index_head_dim
INDEX_TOPK = FLASH.index_topk
INDEX_KPOOL = FLASH.index_kpool
KPOOL_SELECT_K = FLASH.kpool_select_k
TOPK_INDEX_WIDTH = FLASH.topk_index_width
# One indexer cache row is [key(128), gate_scores(128)]. The reference packs a
# third `valid` channel, but that channel only exists to let pooling start at the
# first real token of a left-padded dense batch; a paged cache has no left padding,
# so validity is a function of the request's length and is not stored. This matches
# vLLM Ascend's AscendIndexerKPoolStateSpec, whose head_size is 2 * 128.
INDEX_STATE_WIDTH = 2 * INDEX_DIM

# KDA linear attention.
KDA_H = FLASH.linear_num_heads
KDA_DIM = FLASH.linear_head_dim
KDA_QKV_DIM = KDA_H * KDA_DIM
KDA_CONV_K = FLASH.linear_conv_kernel_dim
KDA_GATE_LOWER_BOUND = FLASH.linear_lower_bound

# MoE.
N_EXPERTS = FLASH.n_routed_experts
TOPK = FLASH.num_experts_per_tok
MOE_INTER = FLASH.moe_intermediate_size

# Hyper-connections.
HC_MULT = FLASH.hc_mult
HC_DIM = FLASH.hc_dim
MIX_HC = FLASH.mix_hc

# The a2a3 int8 cube fractal is MKN 16,32,16, so every quantized K tile must be a
# multiple of 32 (Ascend910_9392.ini, DtypeMKN).
INT8_K_ALIGN = 32

# Most-negative finite FP32, matching models/deepseek_v4_flash_mtp/config.py:284.
# Used to floor masked scores in the sort and softmax paths: a genuine -inf would
# propagate NaN through the sort, and a select cannot be used where an unmasked
# score may legitimately be negative.
FP32_NEG_INF = -3.4028234663852886e38

SUPPORTED_TP_SIZES = (1, 2, 4, 8, 16)
SUPPORTED_EP_SIZES = (1, 2, 4, 8, 16)


def _parse_parallel_size(name: str, default: int) -> int:
    flag = f"--{name}"
    for index, argument in enumerate(sys.argv):
        if argument == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if argument.startswith(f"{flag}="):
            return int(argument.split("=", 1)[1])
    return default


# The deployment target is one Atlas 800 A3 node: 16 dies, TP16 with expert
# parallelism over the same 16 ranks (vLLM Ascend's `--tensor-parallel-size 16
# --enable-expert-parallel` recipe). Smaller shapes stay available for bring-up.
TP_SIZE = _parse_parallel_size("tp", 16)
EP_SIZE = _parse_parallel_size("ep", 16)
if TP_SIZE not in SUPPORTED_TP_SIZES:
    raise ValueError(f"--tp must be one of {SUPPORTED_TP_SIZES}, got {TP_SIZE}")
if EP_SIZE not in SUPPORTED_EP_SIZES:
    raise ValueError(f"--ep must be one of {SUPPORTED_EP_SIZES}, got {EP_SIZE}")
if EP_SIZE % TP_SIZE:
    raise ValueError(f"EP{EP_SIZE} must be divisible by TP{TP_SIZE}")
if H % TP_SIZE:
    raise ValueError(f"{H} attention heads cannot be evenly sharded across TP{TP_SIZE}")
if KDA_H % TP_SIZE:
    raise ValueError(f"{KDA_H} KDA heads cannot be evenly sharded across TP{TP_SIZE}")
if INDEX_H % TP_SIZE:
    raise ValueError(f"{INDEX_H} indexer heads cannot be evenly sharded across TP{TP_SIZE}")
if N_EXPERTS % EP_SIZE:
    raise ValueError(f"{N_EXPERTS} routed experts cannot be evenly sharded across EP{EP_SIZE}")

DP_SIZE = EP_SIZE // TP_SIZE
LOCAL_H = H // TP_SIZE
LOCAL_KDA_H = KDA_H // TP_SIZE
# The indexer is REPLICATED, not head-sharded. Its score sums over all 32 heads
# before the top-k, so sharding heads would make every query need a cross-rank
# reduction of partial scores on every DSA layer. Replicating a kernel this small
# is far cheaper than 12 collectives per token, and it keeps the selection local.
# LOCAL_INDEX_H stays defined for a future sharded variant; kernels use INDEX_H.
LOCAL_INDEX_H = INDEX_H
LOCAL_KDA_QKV_DIM = LOCAL_KDA_H * KDA_DIM
N_LOCAL_EXPERTS = N_EXPERTS // EP_SIZE
LOCAL_VOCAB = VOCAB // TP_SIZE if VOCAB % TP_SIZE == 0 else -1

BLOCK_SIZE = 128
# The indexer's pooled state is addressed in groups of index_kpool tokens, which
# vLLM Ascend models as its own small page class (AscendIndexerKPoolStateSpec).
INDEX_STATE_BLOCK_SIZE = INDEX_KPOOL
MAX_BATCH_PER_DP = 32
MTP_SPEC_TOKENS = 3
DECODE_ROWS_PER_REQUEST = MTP_SPEC_TOKENS + 1
DECODE_MAX_TOKENS = MAX_BATCH_PER_DP * DECODE_ROWS_PER_REQUEST
PREFILL_MAX_TOKENS = 8192
DECODE_RECV_MAX = DP_SIZE * DECODE_MAX_TOKENS
PREFILL_RECV_MAX = DP_SIZE * PREFILL_MAX_TOKENS
RECV_MAX = PREFILL_RECV_MAX
AUX_WIDTH = 8
ROUTE_WIDTH = 8
