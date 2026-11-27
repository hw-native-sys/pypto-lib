# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One EP2 cold-cache decode MoE numerical check on the A2/A3 CPU simulator."""

import argparse
import dataclasses

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig

import config

# Model types specialize at import time; this executable owns its mapped configuration.
config.EP_WORLD_SIZE = 2
config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=16)
config.EXPERT_SLOTS = 8

from decode_moe import (
    AUX_PAD, BANK_ROWS, D, HC_DIM, HC_MULT, MIX_HC, MOE_INTER, N_EXPERTS_GLOBAL,
    N_LOCAL, N_RANKS, N_ROUTES, RECV_MAX, T, TOPK, VOCAB,
    build_tensor_specs, clear_moe_signals, golden_moe, moe,
)
from expert_slots import BANK_DIRECTIONS, EXPERT_SLOTS, RankDescriptors, RankState, WorkerDescriptors, WorkerState
from golden import TensorSpec, ratio_reldiff, run
from models.common.expert_cache import initial_expert_cache


@pl.jit(**BANK_DIRECTIONS)
def mapped_moe_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[BANK_ROWS, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[BANK_ROWS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[BANK_ROWS, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[BANK_ROWS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[BANK_ROWS, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[BANK_ROWS, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    expert_state: pl.InOut[RankState],
    expert_descriptors: RankDescriptors,
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    moe(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next,
        recv_meta, recv_x, recv_aux, arrived, data_arrived,
        routed_y_buf, combine_arrived,
        layer_id, num_tokens, my_rank, pl.const(1, pl.INT32),
        expert_state=expert_state, expert_descriptors=expert_descriptors,
    )
    clear_moe_signals(x_next, arrived, data_arrived, combine_arrived)
    return x_next


@pl.jit.host(**BANK_DIRECTIONS)
def l3_mapped_moe(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, BANK_ROWS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, BANK_ROWS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, BANK_ROWS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, BANK_ROWS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, BANK_ROWS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, BANK_ROWS, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    expert_state: pl.InOut[WorkerState],
    expert_descriptors: WorkerDescriptors,
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mapped_moe_test(
            x_hc[r], hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], input_ids[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            x_next[r],
            recv_meta, recv_x, recv_aux, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            expert_state[r], expert_descriptors[r], layer_id, num_tokens, r,
            device=r,
        )


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    specs, golden = mapped_fixture()
    result = run(
        fn=l3_mapped_moe,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type="a2a3", execution_mode=True,
                    distributed_config=DistributedConfig(device_ids=[0, 1], num_sub_workers=0)),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"x_next": ratio_reldiff(diff_thd=3e-3, pct_thd=0.05)},
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)


def mapped_fixture():
    torch.manual_seed(0)
    specs = build_tensor_specs()
    by_name = {spec.name: spec for spec in specs}
    experts = torch.tensor([1, 3, 5], dtype=torch.int32)
    by_name["input_ids"].init_value = 0
    route_table = torch.zeros(by_name["tid2eid"].shape, dtype=torch.int32)
    route_table[:, 0] = torch.cat([experts, experts + N_LOCAL])
    by_name["tid2eid"].init_value = route_table

    fields = tuple(f"routed_w{i}{suffix}" for i in (1, 2, 3) for suffix in ("", "_scale"))
    # The golden closure keeps shared source storage alive for the simulator workers.
    sources = {name: by_name[name].create_tensor().clone().contiguous().share_memory_() for name in fields}
    descriptors = torch.zeros(N_RANKS, config.FLASH.num_hidden_layers, 6, 4, dtype=torch.int64)
    for field, name in enumerate(fields):
        for rank in range(N_RANKS):
            shard = sources[name][rank]
            stride = shard[0].numel() * shard.element_size()
            descriptors[rank, 0, field] = torch.tensor([shard.data_ptr(), stride, stride, stride])
        by_name[name].init_value = 0
    state = torch.stack([
        initial_expert_cache(EXPERT_SLOTS, config.FLASH.num_hidden_layers, N_LOCAL) for _ in range(N_RANKS)
    ])
    # Keep the existing layer/token scalars after the mapped tensor resources.
    specs[-2:-2] = [
        TensorSpec("expert_state", list(state.shape), torch.int32, init_value=state, resident="stacked"),
        TensorSpec("expert_descriptors", list(descriptors.shape), torch.int64, init_value=descriptors, resident="stacked"),
    ]

    def golden_mapped(values):
        golden_moe(values | sources)
        for name in fields:
            values[name][:, :3] = sources[name][:, experts.long()]
        expected = values["expert_state"]
        expected[:, 1:4, 0] = 0
        expected[:, 1:4, 1] = experts
        expected[:, 1:4, 5] = 1
        expected[:, EXPERT_SLOTS + 1:].flatten(1)[:, experts.long()] = torch.arange(3, dtype=torch.int32)
        # Three cold misses move cache rows 1, 2, 3 to the LRU tail in that order.
        expected[:, 0, 2:4] = torch.tensor([3, 4])
        expected[:, 4, 2] = 0
        expected[:, EXPERT_SLOTS, 3] = 1
        expected[:, 1, 2] = EXPERT_SLOTS
        expected[:, 3, 3] = 0

    return specs, golden_mapped


if __name__ == "__main__":
    main()
