# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B model and external ABI constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3TilingConfig:
    """Qwen3-14B kernel tiling constants that are part of the external ABI."""

    seq_tile: int
    vocab_chunk: int

    @property
    def block_size(self) -> int:
        return self.seq_tile


@dataclass(frozen=True)
class Qwen3Config:
    """Single-source structured constants for the Qwen3-14B kernels."""

    name: str
    family: str
    variant: str

    # Model shape.
    batch: int
    max_seq: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden: int
    intermediate: int
    vocab: int
    real_vocab: int
    num_layers: int

    sampled_ids_pad: int

    # Shared numeric/model invariants.
    eps: float

    # Qwen3-14B attention grouping invariant.
    q_head_batch: int
    q_head_pad: int

    # Quantization constants.
    int8_scale_max: float
    int8_amax_eps: float

    @property
    def kv_hidden(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def hidden_inv(self) -> float:
        return 1.0 / self.hidden

    @property
    def head_dim_inv(self) -> float:
        return 1.0 / self.head_dim

    @property
    def attn_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def half_dim(self) -> int:
        return self.head_dim // 2

    @property
    def q_per_kv(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def q_groups(self) -> int:
        return self.q_per_kv // self.q_head_batch

    @property
    def total_q_groups(self) -> int:
        return self.num_kv_heads * self.q_groups


QWEN3_14B = Qwen3Config(
    name="qwen3-14b",
    family="qwen3",
    variant="14b",
    batch=16,
    max_seq=4096,
    num_heads=40,
    num_kv_heads=8,
    head_dim=128,
    hidden=40 * 128,
    intermediate=17408,
    vocab=152064,
    real_vocab=151936,
    num_layers=40,
    sampled_ids_pad=8,
    eps=1e-6,
    q_head_batch=5,
    q_head_pad=16,
    int8_scale_max=127.0,
    int8_amax_eps=1e-4,
)

QWEN3_14B_TILING = Qwen3TilingConfig(
    seq_tile=128,
    vocab_chunk=512,
)

# fa_fused groups attention work by (KV head, Q-head batch). qk_norm and
# rope_kv_cache currently loop over NUM_KV_HEADS only (one Q-head batch per
# KV head), so the Q heads per KV head must equal Q_HEAD_BATCH exactly --
# supporting Q_GROUPS > 1 would require also iterating the inner Q groups
# in those two regions.
assert QWEN3_14B.q_per_kv == QWEN3_14B.q_head_batch, (
    f"q_per_kv ({QWEN3_14B.q_per_kv}) must equal q_head_batch ({QWEN3_14B.q_head_batch}) "
    f"(qk_norm / rope_kv_cache assume one Q group per KV head)"
)
# Q_HEAD_PAD is the padded Q row count fa_fused operates on. fa_fused does
# set_validshape(scores, Q_HEAD_PAD // 2, ...) on the vec-side scores tile
# and then trims oi/li to Q_HEAD_BATCH rows, so the *half* must (a) be even
# (an odd valid_row without an explicit operand hits pypto#1031) and
# (b) be >= Q_HEAD_BATCH so the trim is fully covered. Both reduce to
# Q_HEAD_PAD % 4 == 0 and Q_HEAD_PAD // 2 >= Q_HEAD_BATCH. (Q_HEAD_PAD = 16
# here -> //2 = 8 >= 5; fa_fused runs SplitMode=None / dual-AIV no-op
# replay, not row halving -- see the module docstring.)
assert QWEN3_14B.q_head_pad % 4 == 0 and QWEN3_14B.q_head_pad // 2 >= QWEN3_14B.q_head_batch, (
    f"q_head_pad ({QWEN3_14B.q_head_pad}) must be a multiple of 4 with "
    f"q_head_pad // 2 ({QWEN3_14B.q_head_pad // 2}) >= q_head_batch ({QWEN3_14B.q_head_batch})"
)
# fa_fused dispatches via pl.spmd(TOTAL_Q_GROUPS // 2) with an inner
# pl.pipeline(2, stage=2) over the Q-group pair; that requires an even count.
assert QWEN3_14B.total_q_groups % 2 == 0, (
    f"total_q_groups ({QWEN3_14B.total_q_groups}) must be even (fa_fused pairs Q groups)"
)
