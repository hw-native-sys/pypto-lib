# Copyright (c) PyPTO Contributors.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Machine-readable Decode shapes and sharding boundaries.

The table is a contract for the A3 gate.  It deliberately keeps the single KV
head explicit: a KV head is replicated across TP ranks until the constructed
runtime proves a legal split.  The table is not an accuracy result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecodeGlobalShape:
    """Physical model dimensions read from the validated text configuration."""

    hidden_size: int = 5120
    attention_heads: int = 64
    kv_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1280
    output_groups: int = 8
    output_lora_rank: int = 1024
    moe_intermediate: int = 2304
    routed_experts: int = 384
    top_k: int = 6
    hc_mult: int = 4
    cache_block_size: int = 128


@dataclass(frozen=True)
class DecodeRankShape:
    """Rank-local dimensions plus the axis ownership that must be captured."""

    dp: int
    tp: int
    ep: int
    q_heads_local: int
    query_width_local: int
    kv_heads_local: int
    kv_head_replicated: bool
    routed_experts_local: int
    expert_shard_axis: str
    q_shard_axis: str
    kv_shard_axis: str
    cache_layout: str
    cache_source_id: str
    dispatch_contract: str
    output_reduce_contract: str
    runtime_shape_status: str = "not_run"


GLOBAL = DecodeGlobalShape()


def rank_shape(dp: int, tp: int, ep: int) -> DecodeRankShape:
    """Return the conservative local shape for one supported topology."""

    if min(dp, tp, ep) <= 0 or dp * tp != ep:
        raise ValueError(f"invalid Decode topology DP={dp}, TP={tp}, EP={ep}")
    if GLOBAL.attention_heads % tp or GLOBAL.routed_experts % ep:
        raise ValueError("attention heads and routed experts must divide the requested TP/EP")
    return DecodeRankShape(
        dp=dp,
        tp=tp,
        ep=ep,
        q_heads_local=GLOBAL.attention_heads // tp,
        query_width_local=(GLOBAL.attention_heads // tp) * GLOBAL.head_dim,
        kv_heads_local=GLOBAL.kv_heads,
        kv_head_replicated=tp > 1,
        routed_experts_local=GLOBAL.routed_experts // ep,
        expert_shard_axis="expert_id over EP",
        q_shard_axis="query_head over TP",
        kv_shard_axis="replicated KV head; runtime capture required for any split",
        cache_layout="[page, row, kv_head, head_dim]",
        cache_source_id="physical source layer plus request slot",
        dispatch_contract="all-to-all EP dispatch, local expert, all-to-all combine",
        output_reduce_contract="TP grouped output reduction followed by shared/routed sum",
    )


def shape_matrix() -> dict[str, dict[str, object]]:
    """Return the required TP1/TP2/TP4 compatibility shape matrix."""

    rows: dict[str, dict[str, object]] = {}
    for name, dp, tp in (("DP16_TP1_EP16", 16, 1), ("DP8_TP2_EP16", 8, 2), ("DP4_TP4_EP16", 4, 4)):
        rows[name] = asdict(rank_shape(dp, tp, 16))
    rows["DP2_TP1_EP2"] = asdict(rank_shape(2, 1, 2))
    return rows


__all__ = ["GLOBAL", "DecodeGlobalShape", "DecodeRankShape", "rank_shape", "shape_matrix"]
