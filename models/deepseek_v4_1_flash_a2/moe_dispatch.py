# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stage native dispatched tokens into expert slabs and restore weighted routes."""

import pypto.language as pl


# model config
EXPERTS = 384
TOPK = 6
D = 5120
RECV_MAX = 64

# tiling
D_TILE = 512
MAP_PAD = 8


def make_dispatch_layout(tokens):
    """Map native count-mode dispatch into the full384 expert ABI."""
    routes = tokens * TOPK

    @pl.jit
    def dispatch_layout(
        payload: pl.Tensor[[routes, D], pl.INT8],
        scales: pl.Tensor[[routes], pl.FP32],
        counts: pl.Tensor[[EXPERTS], pl.INT64],
        expanded: pl.Tensor[[routes], pl.INT32],
        weights: pl.Tensor[[tokens, TOPK], pl.FP32],
        recv_x: pl.InOut[pl.Tensor[[EXPERTS, RECV_MAX, D], pl.INT8]],
        recv_scale: pl.InOut[pl.Tensor[[EXPERTS, RECV_MAX], pl.FP32]],
        recv_weights: pl.InOut[pl.Tensor[[EXPERTS, RECV_MAX], pl.FP32]],
        recv_count: pl.InOut[pl.Tensor[[EXPERTS, 1], pl.INT32]],
        route_map: pl.InOut[pl.Tensor[[routes, MAP_PAD], pl.INT32]],
    ):
        # Serialize narrow scalar stores that can share a cache line.
        for _block in pl.spmd(1, name_hint="a2_moe_dispatch_metadata"):
            begin = pl.cast(0, pl.INT32)
            for expert in pl.range(EXPERTS):
                count = pl.cast(pl.read(counts, [expert]), pl.INT32)
                pl.write(recv_count, [expert, 0], count)
                for original in pl.range(routes):
                    dispatched = pl.read(expanded, [original])
                    if dispatched >= begin and dispatched < begin + count:
                        row = dispatched - begin
                        route_weight = pl.read(weights, [original // TOPK, original % TOPK])
                        token_scale = pl.read(scales, [dispatched])
                        pl.write(recv_scale, [expert, row], token_scale)
                        pl.write(recv_weights, [expert, row], route_weight)
                        target = pl.cast(expert * RECV_MAX, pl.INT32) + row
                        pl.write(route_map, [original, 0], target)
                begin = begin + count

        recv_flat = pl.reshape(recv_x, [EXPERTS * RECV_MAX, D])
        for block in pl.spmd(routes * (D // D_TILE), name_hint="a2_moe_dispatch_payload"):
            original = block // (D // D_TILE)
            column = block % (D // D_TILE) * D_TILE
            dispatched = pl.read(expanded, [original])
            target = pl.read(route_map, [original, 0])
            value = payload[dispatched:dispatched + 1, column:column + D_TILE]
            recv_flat[target:target + 1, column:column + D_TILE] = value
        return recv_x, recv_scale, recv_weights, recv_count, route_map

    return dispatch_layout


def make_route_restore(tokens):
    """Gather every actual weighted expert result in original token/top6 order."""
    routes = tokens * TOPK

    @pl.jit
    def route_restore(
        recv_y: pl.Tensor[[EXPERTS, RECV_MAX, D], pl.FP32],
        route_map: pl.Tensor[[routes, MAP_PAD], pl.INT32],
        dispatch_y: pl.Out[pl.Tensor[[routes, D], pl.FP32]],
    ):
        recv_flat = pl.reshape(recv_y, [EXPERTS * RECV_MAX, D])
        for block in pl.spmd(routes * (D // D_TILE), name_hint="a2_moe_route_restore"):
            original = block // (D // D_TILE)
            column = block % (D // D_TILE) * D_TILE
            source = pl.read(route_map, [original, 0])
            value = recv_flat[source:source + 1, column:column + D_TILE]
            dispatch_y[original:original + 1, column:column + D_TILE] = value
        return dispatch_y

    return route_restore
