# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 decode routing identities and trace-hash fixtures."""

ROUTING_MODEL = 0
ROUTING_TRACE_HASH = 1
ROUTING_MODES = {
    "model": ROUTING_MODEL,
    "trace-hash": ROUTING_TRACE_HASH,
}


def routing_mode_value(name):
    try:
        return ROUTING_MODES[name]
    except KeyError:
        raise ValueError(
            f"routing mode must be one of {sorted(ROUTING_MODES)}, got {name!r}"
        ) from None


def resolve_routing_layer_id(actual_layer_id, mode):
    mode_value = routing_mode_value(mode) if isinstance(mode, str) else mode
    if mode_value == ROUTING_TRACE_HASH:
        return 0
    if mode_value == ROUTING_MODEL:
        return actual_layer_id
    raise ValueError(f"unsupported routing mode value {mode_value}")


def target_expert_topology(ep):
    if ep != 8:
        raise ValueError(f"the comparison topology requires ep=8, got {ep}")
    return 128, 16


def build_trace_hash_tid2eid(
    *,
    num_layers,
    first_layer_id,
    n_ranks,
    tokens_per_rank,
    vocab_size,
    topk,
    n_experts,
):
    import torch

    rank = torch.arange(n_ranks, dtype=torch.int64).reshape(n_ranks, 1, 1)
    token = torch.arange(vocab_size, dtype=torch.int64).reshape(1, vocab_size, 1)
    route = torch.arange(topk, dtype=torch.int64).reshape(1, 1, topk)
    layers = []
    routes_per_layer = n_ranks * tokens_per_rank * topk
    for layer_id in range(first_layer_id, first_layer_id + num_layers):
        layer_routes = (
            layer_id * routes_per_layer
            + rank * (tokens_per_rank * topk)
            + token * topk
            + route
        ) % n_experts
        layers.append(layer_routes.to(torch.int32))
    return torch.cat(layers, dim=1).contiguous()
