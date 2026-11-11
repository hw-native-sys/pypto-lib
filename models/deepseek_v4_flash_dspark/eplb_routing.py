# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""Map logical top-k expert IDs to physical replicas with explicit valid-row masking."""

from math import gcd

import pypto.language as pl


def make_eplb_routing(*, logical_experts: int, ranks: int, tokens: int, top_k: int,
                      bank_layers: int = 1, boolean_mask: bool = True):
    """Specialize replica selection independently from the gate's score/hash policy."""
    if min(logical_experts, ranks, tokens, top_k, bank_layers) <= 0 or top_k > 8:
        raise ValueError("EPLB routing requires positive dimensions and cache-line aligned route rows")
    E = logical_experts
    R = ranks
    T = tokens
    K = top_k
    BANK_ROWS = bank_layers * E
    # tiling
    TOKEN_TILE = min(T, 16 // gcd(K, 16))
    K_TILE = 8
    if T % TOKEN_TILE:
        raise ValueError("EPLB token blocks must own complete 64-byte route spans")

    @pl.jit.inline
    def route_experts(
        logical_ids: pl.Tensor[[T, K], pl.INT32],
        logical_to_physical: pl.Tensor[[BANK_ROWS, R], pl.INT32],
        replica_count: pl.Tensor[[BANK_ROWS], pl.INT32],
        valid_rows: pl.Tensor[[T], pl.INT32],
        physical_ids: pl.Tensor[[T, K], pl.INT32],
        my_rank: pl.Scalar[pl.INT32],
        route_epoch: pl.Scalar[pl.INT32],
        layer_id: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for token_block in pl.spmd(T // TOKEN_TILE, name_hint="eplb_select_replica"):
            token0 = token_block * TOKEN_TILE
            logical_tile = pl.tile.load(logical_ids, [token0, 0], [TOKEN_TILE, K_TILE], valid_shape=[TOKEN_TILE, K])
            physical_tile = pl.tile.full([TOKEN_TILE, K_TILE], dtype=pl.INT32, value=-1)
            for row in pl.range(TOKEN_TILE):
                token = token0 + row
                row_marker = pl.read(valid_rows, [token])
                if boolean_mask:
                    valid = row_marker != 0
                else:
                    valid = row_marker >= 0
                if token < num_tokens and valid:
                    for route in pl.range(K):
                        expert = pl.tile.read(logical_tile, [row, route])
                        bank_expert = (layer_id % bank_layers) * E + expert
                        count = pl.read(replica_count, [bank_expert])
                        seed = token + my_rank * T + route + route_epoch
                        replica = seed % count
                        physical = pl.read(logical_to_physical, [bank_expert, replica])
                        pl.tile.write(physical_tile, [row, route], physical)
            physical_valid = pl.set_validshape(physical_tile, TOKEN_TILE, K)
            pl.tile.store(physical_valid, [token0, 0], physical_ids)
        return physical_ids

    return route_experts


def make_moe_eplb(*, enabled: bool, logical_experts: int, ranks: int, tokens: int,
                  top_k: int, local_slots: int, bank_layers: int):
    """Bind route control and padded load publication for one MoE capacity."""
    E, R, T, K = logical_experts, ranks, tokens, top_k
    C = local_slots
    C_PAD = (C + 15) // 16 * 16
    BANK_ROWS = bank_layers * E
    route_experts = make_eplb_routing(
        logical_experts=E, ranks=R, tokens=T, top_k=K,
        bank_layers=bank_layers, boolean_mask=False,
    ) if enabled else None

    @pl.jit.inline
    def mapped_routes(
        indices: pl.Tensor[[T, K], pl.INT32],
        mapping: pl.Tensor[[BANK_ROWS, R], pl.INT32],
        counts: pl.Tensor[[BANK_ROWS], pl.INT32],
        source_rows: pl.Tensor[[T], pl.INT32],
        control: pl.Tensor[[2], pl.INT32],
        layer: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        physical = pl.create_tensor([T, K], dtype=pl.INT32)
        epoch = pl.read(control, [0])
        route_experts(indices, mapping, counts, source_rows, physical, rank, epoch, layer, num_tokens)
        return physical

    @pl.jit.inline
    def identity_routes(
        indices: pl.Tensor[[T, K], pl.INT32],
        mapping: pl.Tensor[[BANK_ROWS, R], pl.INT32],
        counts: pl.Tensor[[BANK_ROWS], pl.INT32],
        source_rows: pl.Tensor[[T], pl.INT32],
        control: pl.Tensor[[2], pl.INT32],
        layer: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        return indices

    @pl.jit.inline
    def publish_counts(
        counts: pl.Tensor[[C, 1], pl.INT32],
        output: pl.Tensor[[bank_layers, C_PAD], pl.INT32],
        control: pl.Tensor[[2], pl.INT32],
        layer: pl.Scalar[pl.INT32],
        accumulate: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="eplb_export_load", deps=[ready]):
            bank_layer = layer % bank_layers
            row = pl.tile.full([1, C_PAD], dtype=pl.INT32, value=0)
            if pl.read(control, [1]) != 0:
                for expert in pl.range(C):
                    value = pl.read(counts, [expert, 0])
                    if accumulate != 0:
                        value = value + pl.read(output, [bank_layer, expert])
                    pl.tile.write(row, [0, expert], value)
            pl.tile.store(row, [bank_layer, 0], output)
        return counts

    @pl.jit.inline
    def discard_counts(
        counts: pl.Tensor[[C, 1], pl.INT32],
        output: pl.Tensor[[bank_layers, C_PAD], pl.INT32],
        control: pl.Tensor[[2], pl.INT32],
        layer: pl.Scalar[pl.INT32],
        accumulate: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        return counts

    return (mapped_routes, publish_counts) if enabled else (identity_routes, discard_counts)


def make_routing_test(*, logical_experts: int, ranks: int, tokens: int = 8, top_k: int = 6):
    """Build a distributed golden entry for replica IDs and non-contiguous invalid rows."""
    E, R, T, K = logical_experts, ranks, tokens, top_k
    route_experts = make_eplb_routing(logical_experts=E, ranks=R, tokens=T, top_k=K)

    @pl.jit
    def routing_test(
        logical_ids: pl.Tensor[[T, K], pl.INT32],
        logical_to_physical: pl.Tensor[[E, R], pl.INT32],
        replica_count: pl.Tensor[[E], pl.INT32],
        valid_rows: pl.Tensor[[T], pl.INT32],
        physical_ids: pl.Out[pl.Tensor[[T, K], pl.INT32]],
        rank: pl.Scalar[pl.INT32],
        epoch: pl.Scalar[pl.INT32],
    ):
        route_experts(logical_ids, logical_to_physical, replica_count, valid_rows, physical_ids, rank, epoch, 0, T)
        return physical_ids

    @pl.jit.host
    def l3_routing_test(
        logical_ids: pl.Tensor[[R, T, K], pl.INT32],
        logical_to_physical: pl.Tensor[[R, E, R], pl.INT32],
        replica_count: pl.Tensor[[R, E], pl.INT32],
        valid_rows: pl.Tensor[[R, T], pl.INT32],
        physical_ids: pl.Out[pl.Tensor[[R, T, K], pl.INT32]],
        epoch: pl.Scalar[pl.INT32],
    ):
        for rank in pl.range(R):
            routing_test(
                logical_ids[rank], logical_to_physical[rank], replica_count[rank], valid_rows[rank],
                physical_ids[rank],
                rank, epoch, device=rank,
            )
        return physical_ids

    return l3_routing_test


def golden_routing(tensors):
    import torch

    logical = tensors["logical_ids"].long()
    inverse = tensors["logical_to_physical"]
    counts = tensors["replica_count"]
    valid = tensors["valid_rows"]
    epoch = int(tensors["epoch"])
    result = torch.full_like(logical, -1, dtype=torch.int32)
    ranks, tokens, top_k = logical.shape
    for rank in range(ranks):
        for token in range(tokens):
            if valid[rank, token]:
                for route in range(top_k):
                    expert = int(logical[rank, token, route])
                    replica = (token + rank * tokens + route + epoch) % int(counts[rank, expert])
                    result[rank, token, route] = inverse[rank, expert, replica]
    tensors["physical_ids"] = result


def build_tensor_specs(*, logical_experts: int, ranks: int, redundant_experts: int, tokens: int = 8, top_k: int = 6):
    import torch

    from golden import ScalarSpec, TensorSpec

    physical = logical_experts + redundant_experts
    if logical_experts < 8 or physical % ranks or physical // ranks > logical_experts:
        raise ValueError("Invalid routing fixture capacity")
    mapping = torch.arange(physical) % logical_experts
    inverse = torch.full((logical_experts, ranks), -1, dtype=torch.int32)
    counts = torch.bincount(mapping, minlength=logical_experts).to(torch.int32)
    for expert in range(logical_experts):
        locations = torch.nonzero(mapping == expert).flatten().to(torch.int32)
        inverse[expert, :len(locations)] = locations
    ids = torch.arange(ranks * tokens * top_k).reshape(ranks, tokens, top_k).remainder(logical_experts).int()
    valid = torch.ones((ranks, tokens), dtype=torch.int32)
    valid[:, 1::3] = 0
    inverse_stacked = inverse.unsqueeze(0).repeat(ranks, 1, 1)
    counts_stacked = counts.unsqueeze(0).repeat(ranks, 1)
    return [
        TensorSpec("logical_ids", [ranks, tokens, top_k], torch.int32, init_value=lambda: ids),
        TensorSpec("logical_to_physical", list(inverse_stacked.shape), torch.int32, init_value=lambda: inverse_stacked),
        TensorSpec("replica_count", list(counts_stacked.shape), torch.int32, init_value=lambda: counts_stacked),
        TensorSpec("valid_rows", [ranks, tokens], torch.int32, init_value=lambda: valid),
        TensorSpec("physical_ids", [ranks, tokens, top_k], torch.int32),
        ScalarSpec("epoch", torch.int32, 5),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a5"), default="a2a3")
    parser.add_argument("--ep", type=int, choices=(2, 4, 8, 16), default=2)
    parser.add_argument("--logical-experts", type=int, default=256)
    parser.add_argument("--redundant-experts", type=int, default=16)
    parser.add_argument("-d", "--devices", required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(device) for device in args.devices.split(",")]
    if len(devices) != args.ep:
        parser.error("Device count must match EP")
    fn = make_routing_test(logical_experts=args.logical_experts, ranks=args.ep)
    result = run(
        fn=fn,
        specs=build_tensor_specs(logical_experts=args.logical_experts, ranks=args.ep,
                                 redundant_experts=args.redundant_experts),
        golden_fn=golden_routing,
        compile_only=args.compile_only,
        config=dict(backend_type=args.platform, execution_mode=False,
                    distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0)),
        rtol=0,
        atol=0,
        compare_fn={"physical_ids": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
    )
    if not result.passed:
        raise SystemExit(result.error or "EPLB routing validation failed")
