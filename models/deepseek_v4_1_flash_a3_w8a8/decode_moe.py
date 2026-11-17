# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode MoE over EP16: one active token per rank through the shared exchange.

The active-token tile is one, so the worst-case receive capacity for a local
expert is ``N_RANKS * 1`` rows.  The fixture splits the ranks over two
destinations to keep local, remote, skewed, zero-receive, and idle-rank
participation covered while every receiving rank stays a real EP16 rank.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import EP_SIZE, N_LOCAL_EXPERTS, FLASH as M
from moe_exchange import COUNT_PAD, golden_moe_ep, TOPK
from moe_exchange import moe_ep


T = 1
CAP = T
RECV_MAX = EP_SIZE * CAP
RECV_ROWS = N_LOCAL_EXPERTS * RECV_MAX
SLOT_ROWS = T * TOPK
D = M.hidden_size
INTER = M.moe_intermediate_size


@pl.jit
def decode_moe_rank(
    x: pl.Tensor[[1, 5120], pl.BF16],
    indices: pl.Tensor[[1, 6], pl.INT32],
    weights: pl.Tensor[[1, 6], pl.FP32],
    w1: pl.Tensor[[N_LOCAL_EXPERTS, 5120, 2304], pl.INT8, pl.NZ],
    w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2304], pl.FP32],
    w3: pl.Tensor[[N_LOCAL_EXPERTS, 5120, 2304], pl.INT8, pl.NZ],
    w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2304], pl.FP32],
    w2: pl.Tensor[[N_LOCAL_EXPERTS, 2304, 5120], pl.INT8, pl.NZ],
    w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, 5120], pl.FP32],
    shared_w1: pl.Tensor[[5120, 2304], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[2304], pl.FP32],
    shared_w3: pl.Tensor[[5120, 2304], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[2304], pl.FP32],
    shared_w2: pl.Tensor[[2304, 5120], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[5120], pl.FP32],
    recv_x: pld.DistributedTensor[[RECV_ROWS, 5120], pl.BF16],
    recv_aux: pld.DistributedTensor[[RECV_ROWS, 8], pl.FP32],
    recv_meta: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    meta_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    data_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    return_slots: pld.DistributedTensor[[SLOT_ROWS, 5120], pl.BF16],
    return_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    return_consumed: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    routed: pl.Out[pl.Tensor[[1, 5120], pl.BF16]],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    routed_value, _moe_done = moe_ep(
        x, indices, weights,
        w1, w1_scale, w3, w3_scale, w2, w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
        recv_x, recv_aux, recv_meta, meta_ready, data_ready, return_slots, return_ready,
        consumed, return_consumed,
        routed, my_rank, epoch, capacity=T, rows=T,
    )
    return routed_value


@pl.jit.host
def decode_moe(
    x: pl.Tensor[[EP_SIZE, T, 5120], pl.BF16],
    indices: pl.Tensor[[EP_SIZE, T, 6], pl.INT32],
    weights: pl.Tensor[[EP_SIZE, T, 6], pl.FP32],
    w1: pl.Tensor[[N_LOCAL_EXPERTS, 5120, 2304], pl.INT8],
    w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2304], pl.FP32],
    w3: pl.Tensor[[N_LOCAL_EXPERTS, 5120, 2304], pl.INT8],
    w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2304], pl.FP32],
    w2: pl.Tensor[[N_LOCAL_EXPERTS, 2304, 5120], pl.INT8],
    w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, 5120], pl.FP32],
    shared_w1: pl.Tensor[[5120, 2304], pl.INT8],
    shared_w1_scale: pl.Tensor[[2304], pl.FP32],
    shared_w3: pl.Tensor[[5120, 2304], pl.INT8],
    shared_w3_scale: pl.Tensor[[2304], pl.FP32],
    shared_w2: pl.Tensor[[2304, 5120], pl.INT8],
    shared_w2_scale: pl.Tensor[[5120], pl.FP32],
    routed: pl.Out[pl.Tensor[[EP_SIZE, T, 5120], pl.BF16]],
    epoch: pl.Scalar[pl.INT32],
):
    recv_x_buf = pld.alloc_window_buffer([RECV_ROWS, 5120], dtype=pl.BF16)
    recv_aux_buf = pld.alloc_window_buffer([RECV_ROWS, 8], dtype=pl.FP32)
    recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
    meta_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    data_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    return_buf = pld.alloc_window_buffer([SLOT_ROWS, 5120], dtype=pl.BF16)
    return_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    consumed_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    return_consumed_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_x = pld.window(recv_x_buf, [RECV_ROWS, 5120], dtype=pl.BF16)
        recv_aux = pld.window(recv_aux_buf, [RECV_ROWS, 8], dtype=pl.FP32)
        recv_meta = pld.window(recv_meta_buf, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
        meta_ready = pld.window(meta_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        data_ready = pld.window(data_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        return_slots = pld.window(return_buf, [SLOT_ROWS, 5120], dtype=pl.BF16)
        return_ready = pld.window(return_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
        consumed = pld.window(consumed_buf, [EP_SIZE, 1], dtype=pl.INT32)
        return_consumed = pld.window(return_consumed_buf, [EP_SIZE, 1], dtype=pl.INT32)
        decode_moe_rank(
            x[rank], indices[rank], weights[rank],
            w1, w1_scale, w3, w3_scale, w2, w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
            recv_x, recv_aux, recv_meta, meta_ready, data_ready, return_slots, return_ready,
            consumed, return_consumed,
            routed[rank], rank, epoch,
            device=rank,
        )


def _stacked_experts(n: int, k: int):
    from utils import pack_nz, quant_w_per_channel

    packed_rows, scale_rows = [], []
    for _ in range(N_LOCAL_EXPERTS):
        logical, scale = quant_w_per_channel(torch.randn(n, k) * 0.05)
        packed_rows.append(pack_nz(logical.transpose(0, 1).contiguous()))
        scale_rows.append(scale)
    return torch.stack(packed_rows), torch.stack(scale_rows)


def _shared_expert(n: int, k: int):
    from utils import pack_nz, quant_w_per_channel

    logical, scale = quant_w_per_channel(torch.randn(n, k) * 0.05)
    return pack_nz(logical.transpose(0, 1).contiguous()), scale


def build_tensor_specs():
    from golden import ScalarSpec, TensorSpec

    w1, w1_scale = _stacked_experts(INTER, D)
    w3, w3_scale = _stacked_experts(INTER, D)
    w2, w2_scale = _stacked_experts(D, INTER)
    shared_w1, shared_w1_scale = _shared_expert(INTER, D)
    shared_w3, shared_w3_scale = _shared_expert(INTER, D)
    shared_w2, shared_w2_scale = _shared_expert(D, INTER)
    # Every rank's single active token routes to three experts on rank 0 and three
    # on rank 1; both destinations own local routes, the other 14 ranks receive
    # nothing but still take part in every exchange barrier.
    routes = torch.tensor([0, 1, 2, N_LOCAL_EXPERTS, N_LOCAL_EXPERTS + 1, N_LOCAL_EXPERTS + 2], dtype=torch.int32)
    indices = routes.reshape(1, 1, TOPK).expand(EP_SIZE, T, TOPK).contiguous()
    return [
        TensorSpec("x", [EP_SIZE, T, D], torch.bfloat16, init_value=lambda: torch.randn(EP_SIZE, T, D) * 0.1),
        TensorSpec("indices", [EP_SIZE, T, TOPK], torch.int32, init_value=indices),
        TensorSpec("weights", [EP_SIZE, T, TOPK], torch.float32, init_value=lambda: torch.rand(EP_SIZE, T, TOPK)),
        TensorSpec("w1", [N_LOCAL_EXPERTS, D, INTER], torch.int8, init_value=w1),
        TensorSpec("w1_scale", [N_LOCAL_EXPERTS, INTER], torch.float32, init_value=w1_scale),
        TensorSpec("w3", [N_LOCAL_EXPERTS, D, INTER], torch.int8, init_value=w3),
        TensorSpec("w3_scale", [N_LOCAL_EXPERTS, INTER], torch.float32, init_value=w3_scale),
        TensorSpec("w2", [N_LOCAL_EXPERTS, INTER, D], torch.int8, init_value=w2),
        TensorSpec("w2_scale", [N_LOCAL_EXPERTS, D], torch.float32, init_value=w2_scale),
        TensorSpec("shared_w1", [D, INTER], torch.int8, init_value=shared_w1),
        TensorSpec("shared_w1_scale", [INTER], torch.float32, init_value=shared_w1_scale),
        TensorSpec("shared_w3", [D, INTER], torch.int8, init_value=shared_w3),
        TensorSpec("shared_w3_scale", [INTER], torch.float32, init_value=shared_w3_scale),
        TensorSpec("shared_w2", [INTER, D], torch.int8, init_value=shared_w2),
        TensorSpec("shared_w2_scale", [D], torch.float32, init_value=shared_w2_scale),
        TensorSpec("routed", [EP_SIZE, T, D], torch.bfloat16),
        ScalarSpec("epoch", torch.int32, 1),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(EP_SIZE)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device_ids = [int(value) for value in args.device.split(",") if value != ""]
    if len(device_ids) != EP_SIZE:
        raise SystemExit(f"need exactly {EP_SIZE} device ids for EP{EP_SIZE}, got {device_ids}")
    result = run(
        fn=decode_moe,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_ep,
        compile_only=args.compile_only,
        config=dict(
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
        ),
        rtol=2e-2,
        atol=2e-2,
        compare_fn={"routed": ratio_allclose(atol=2e-2, rtol=2e-2, max_error_ratio=1e-2)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
