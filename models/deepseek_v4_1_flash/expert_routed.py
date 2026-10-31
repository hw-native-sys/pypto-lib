# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 routed expert compute with distributed scatter."""

import argparse

import torch
import pypto.language as pl
import pypto.language.distributed as pld


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--case", choices=["all", "bounds", "tails", "zero", "full"], default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    from dataclasses import replace

    import config

    test_experts = args.ep * args.experts_per_rank
    if args.experts_per_rank <= 0 or not config.FLASH.num_experts_per_tok <= test_experts <= 384:
        parser.error("--experts-per-rank must give between TopK and 384 total routed experts")
    config.FLASH = replace(config.FLASH, n_routed_experts=test_experts)
    config.N_LOCAL_EXPERTS = args.experts_per_rank


from config import FLASH as M
from config import EP_SIZE as EP_WORLD_SIZE, MOE_RECV_MAX as RECV_MAX
from config import ROUTE_T_DYN, N_LOCAL_EXPERTS
from moe_ffn import moe_ffn_experts
from quant import pack_mxfp4_weight_tiles, unpack_mxfp4_weight_tiles


# model config
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
MX_GROUP = 32
K_SCALE = D // MX_GROUP
H_SCALE = MOE_INTER // MX_GROUP

# Routed-expert across-layer mean dequantization standard deviations.
ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

# tiling
MX_K_TILE = 256
MX_MM_INTER_TILE = 256
MX_W2_D_OUT_TILE = 256
MX_PACKED_LANE_ROWS = MX_K_TILE // 2
MX_PACKED_LANE_COLS = MX_MM_INTER_TILE // 2
MX_W1_PACKED_ROWS = (2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE)) * MX_PACKED_LANE_ROWS
MX_W3_PACKED_ROWS = (2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE)) * MX_PACKED_LANE_ROWS
MX_W2_PACKED_ROWS = (2 * (MOE_INTER // MX_K_TILE) * (D // MX_W2_D_OUT_TILE)) * MX_PACKED_LANE_ROWS
ROUTED_ROW_CAPACITY = min(N_LOCAL_EXPERTS, M.num_experts_per_tok) * RECV_MAX


@pl.jit.inline(auto_scope=False)
def expert_routed_compute(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.BF16]],
    input_ready: pl.Scalar[pl.TASK_ID],
    partial_rows: pl.constexpr,
):
    """Project, activate, and down-project the active fixed-capacity expert rows."""
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    route_weights = pl.reshape(recv_weights, [N_LOCAL_EXPERTS * RECV_MAX, 1])
    recv_y, down_ready = moe_ffn_experts(
        recv_x_flat, recv_mx_scale, route_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale, recv_y, input_ready,
        N_LOCAL_EXPERTS, RECV_MAX, D, MOE_INTER, partial_rows,
    )
    return recv_y, down_ready


@pl.jit.inline(auto_scope=False)
def expert_routed_scatter(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    recv_route: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_meta: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    input_ready: pl.Scalar[pl.TASK_ID],
):
    """Compute active expert rows and scatter each source interval to its route destinations."""
    recv_y = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.BF16)
    recv_y, down_ready = expert_routed_compute(
        recv_x, recv_mx_scale, recv_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale,
        routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale,
        recv_y, input_ready, partial_rows=ROUTED_ROW_CAPACITY,
    )
    scatter_tids = pl.array.create(N_LOCAL_EXPERTS, pl.TASK_ID)
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="expert_scatter", deps=[down_ready], no_dep_args=[routed_output]) as scatter_tid:
            count_read = pl.read(recv_expert_count, [local_e, 0])
            count = pl.cast(count_read, pl.INDEX)
            source_begin = pl.cast(0, pl.INDEX)
            for src in pl.range(EP_WORLD_SIZE):
                source_rows_read = pl.read(recv_meta, [src, local_e])
                source_rows = pl.cast(source_rows_read, pl.INDEX)
                source_end = source_begin + source_rows
                valid_end = pl.min(source_end, count)
                for row in pl.range(source_begin, valid_end):
                    route_read = pl.read(recv_route, [local_e, row])
                    route = pl.cast(route_read, pl.INDEX)
                    pld.tensor.put(dst=routed_output, peer=src, src=recv_y, dst_offsets=[route, 0], src_offsets=[local_e * RECV_MAX + row, 0], shape=[1, D])
                source_begin = source_end
        scatter_tids[local_e] = scatter_tid
    scatter_done = pl.system.task_dummy(deps=[scatter_tids[expert] for expert in range(N_LOCAL_EXPERTS)])
    return scatter_done


@pl.jit(auto_scope=False)
def local_compute_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    routed_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.BF16]],
):
    input_ready = pl.system.task_dummy(deps=[])
    routed_y, down_ready = expert_routed_compute(
        recv_x, recv_mx_scale, recv_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale,
        routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale,
        routed_y, input_ready, partial_rows=N_LOCAL_EXPERTS * RECV_MAX,
    )
    return routed_y


def golden_expert_routed(tensors):
    """Torch reference for MXFP4-grid W1/W3/W2 with MXFP8 activations."""
    import torch

    import torch.nn.functional as F

    from quant import unpack_mx_scale_codes as decode_e8m0_codes, quantize_mxfp8_upward_e8m0
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized as matmul_mx_golden

    recv_x = tensors["recv_x"]
    recv_scale_codes = tensors["recv_mx_scale"].reshape(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE)
    recv_scale_decoded = decode_e8m0_codes(recv_scale_codes, side="a")
    recv_mx_scale = recv_scale_decoded.reshape(N_LOCAL_EXPERTS, RECV_MAX, K_SCALE)
    recv_weights = tensors["recv_weights"].float()
    recv_expert_count = tensors["recv_expert_count"]
    w1_scale = decode_e8m0_codes(tensors['routed_w1_scale'], side='b').reshape(N_LOCAL_EXPERTS, K_SCALE, MOE_INTER)
    w3_scale = decode_e8m0_codes(tensors['routed_w3_scale'], side='b').reshape(N_LOCAL_EXPERTS, K_SCALE, MOE_INTER)
    w2_scale = decode_e8m0_codes(tensors['routed_w2_scale'], side='b').reshape(N_LOCAL_EXPERTS, H_SCALE, D)

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub = recv_x[e, :n_rows, :]
        x_scale = recv_mx_scale[e, :n_rows, :]
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)
        w1_packed = tensors['routed_w1_packed'][e].reshape(
            2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE), MX_K_TILE * MX_MM_INTER_TILE // 4,
        )
        w1_fp8 = unpack_mxfp4_weight_tiles(w1_packed, D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE)
        w3_packed = tensors['routed_w3_packed'][e].reshape(
            2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE), MX_K_TILE * MX_MM_INTER_TILE // 4,
        )
        w3_fp8 = unpack_mxfp4_weight_tiles(w3_packed, D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE)
        w2_packed = tensors['routed_w2_packed'][e].reshape(
            2 * (MOE_INTER // MX_K_TILE) * (D // MX_W2_D_OUT_TILE), MX_K_TILE * MX_W2_D_OUT_TILE // 4,
        )
        w2_fp8 = unpack_mxfp4_weight_tiles(w2_packed, MOE_INTER, D, MX_K_TILE, MX_W2_D_OUT_TILE)

        gate = matmul_mx_golden(x_sub, x_scale, w1_fp8, w1_scale[e])
        up = matmul_mx_golden(x_sub, x_scale, w3_fp8, w3_scale[e])
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up * w_per_row
        h_fp8, h_scale_codes = quantize_mxfp8_upward_e8m0(h)
        y = matmul_mx_golden(h_fp8, h_scale_codes, w2_fp8, w2_scale[e]).to(torch.bfloat16).float()
        recv_y[e, :n_rows, :] = y

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_mxfp4_weights(n_experts, dequant_std, seed_base=0):
    """Generate checkpoint-style packed MXFP4 routed weights and scales."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from quant import pack_mx_b_scale, quantize_mxfp4_weight

    w1_weights = torch.empty((n_experts, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS), dtype=torch.uint8)
    w3_weights = torch.empty_like(w1_weights)
    w2_weights = torch.empty((n_experts, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS), dtype=torch.uint8)
    w1_scales = torch.empty((n_experts, D // MX_GROUP, MOE_INTER), dtype=torch.uint8)
    w3_scales = torch.empty_like(w1_scales)
    w2_scales = torch.empty((n_experts, MOE_INTER // MX_GROUP, D), dtype=torch.uint8)

    def generate_expert(e):
        seed = seed_base + e * 3
        generator = torch.Generator().manual_seed(seed)
        w1, w1s = quantize_mxfp4_weight(torch.randn(MOE_INTER, D, generator=generator) * dequant_std['w1'])
        w3, w3s = quantize_mxfp4_weight(torch.randn(MOE_INTER, D, generator=generator) * dequant_std['w3'])
        w2, w2s = quantize_mxfp4_weight(torch.randn(D, MOE_INTER, generator=generator) * dequant_std['w2'])
        w1_tiles = pack_mxfp4_weight_tiles(w1, MX_K_TILE, MX_MM_INTER_TILE)
        w1_weights[e] = w1_tiles.reshape(MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w3_tiles = pack_mxfp4_weight_tiles(w3, MX_K_TILE, MX_MM_INTER_TILE)
        w3_weights[e] = w3_tiles.reshape(MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w2_tiles = pack_mxfp4_weight_tiles(w2, MX_K_TILE, MX_W2_D_OUT_TILE)
        w2_weights[e] = w2_tiles.reshape(MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w1_scales[e] = w1s.transpose(0, 1).contiguous().view(torch.uint8)
        w3_scales[e] = w3s.transpose(0, 1).contiguous().view(torch.uint8)
        w2_scales[e] = w2s.transpose(0, 1).contiguous().view(torch.uint8)

    with ThreadPoolExecutor(max_workers=min(4, n_experts)) as pool:
        list(pool.map(generate_expert, range(n_experts)))

    def pack_expert_scales(scales):
        packed = pack_mx_b_scale(scales.flatten(0, 1))
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        return packed.view(e8m0) if e8m0 is not None else packed

    return (
        w1_weights, pack_expert_scales(w1_scales),
        w3_weights, pack_expert_scales(w3_scales),
        w2_weights, pack_expert_scales(w2_scales),
    )


def golden_local_compute(tensors):
    """Fill the flat local output through the production expert golden."""
    values = dict(tensors)
    values["recv_y"] = tensors["routed_y"].reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
    golden_expert_routed(values)


def compare_local_compute(actual, expected, *, inputs, **kwargs):
    """Compare each active expert independently with the routed-row numerical contract."""
    from golden import ratio_reldiff

    actual_rows = actual.reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
    expected_rows = expected.reshape_as(actual_rows)
    comparator = ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)
    for expert, count in enumerate(inputs["recv_expert_count"].reshape(-1).tolist()):
        passed, message = comparator(actual_rows[expert, :count], expected_rows[expert, :count], inputs=inputs, **kwargs)
        if not passed:
            return False, f"expert {expert}, count {count}: {message}"
    return True, ""


def validation_counts(case="bounds", *, tail_offset=0):
    """Return bounded empty, 16-row-tail, and receive-capacity workloads."""
    if case == "zero":
        pattern = (0,)
    elif case == "full":
        pattern = (RECV_MAX,)
    elif case == "tails":
        pattern = (15, 16, 17, 63, 64, 65)
    elif case == "bounds":
        pattern = (0, 1, RECV_MAX - 1, RECV_MAX)
    else:
        raise ValueError(f"Unknown local compute case: {case}")
    values = [min(RECV_MAX, pattern[(expert + tail_offset) % len(pattern)]) for expert in range(N_LOCAL_EXPERTS)]
    return torch.tensor(values, dtype=torch.int32).reshape(N_LOCAL_EXPERTS, 1)


def build_tensor_specs(case="bounds", *, tail_offset=0):
    """Build local compute tensors with poisoned inactive capacity and lazy packed weights."""
    from functools import cache

    from golden import TensorSpec
    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    counts = validation_counts(case, tail_offset=tail_offset)

    @cache
    def receive_inputs():
        generator = torch.Generator().manual_seed(20261028)
        source = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16, generator=generator)
        payload, codes = quantize_mxfp8_upward_e8m0(source.reshape(-1, D).float())
        payload = payload.reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
        codes = codes.reshape(N_LOCAL_EXPERTS, RECV_MAX, K_SCALE).view(torch.uint8)
        route_weights = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, generator=generator)
        for expert, count in enumerate(counts.reshape(-1).tolist()):
            payload.view(torch.uint8)[expert, count:] = 127
            codes[expert, count:] = 255
            route_weights[expert, count:] = float("nan")
        packed_codes = pack_mx_a_scale(codes.reshape(-1, K_SCALE)).view(torch.float8_e8m0fnu)
        return payload, packed_codes.reshape(1, -1), route_weights

    @cache
    def routed_weights():
        return gen_routed_mxfp4_weights(N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=10)

    fp8 = torch.float8_e4m3fn
    e8m0 = torch.float8_e8m0fnu
    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], fp8, init_value=lambda: receive_inputs()[0]),
        TensorSpec("recv_mx_scale", [1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], e8m0, init_value=lambda: receive_inputs()[1]),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=lambda: receive_inputs()[2]),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=counts),
        TensorSpec("routed_w1_packed", [N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[0]),
        TensorSpec("routed_w1_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], e8m0, init_value=lambda: routed_weights()[1]),
        TensorSpec("routed_w3_packed", [N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[2]),
        TensorSpec("routed_w3_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], e8m0, init_value=lambda: routed_weights()[3]),
        TensorSpec("routed_w2_packed", [N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[4]),
        TensorSpec("routed_w2_scale", [N_LOCAL_EXPERTS * H_SCALE, D], e8m0, init_value=lambda: routed_weights()[5]),
        TensorSpec("routed_y", [N_LOCAL_EXPERTS * RECV_MAX, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    from golden import run

    cases = ("bounds", "tails", "zero") if args.case == "all" else (args.case,)
    for case in cases:
        tail_rounds = (6 + N_LOCAL_EXPERTS - 1) // N_LOCAL_EXPERTS if case == "tails" else 1
        for tail_round in range(tail_rounds):
            print(f"Local routed compute: case={case}, tail_round={tail_round}, experts={N_LOCAL_EXPERTS}, stride={RECV_MAX}")
            result = run(
                fn=local_compute_test,
                specs=build_tensor_specs(case, tail_offset=tail_round * N_LOCAL_EXPERTS),
                golden_fn=golden_local_compute,
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                runtime_dir=args.runtime_dir,
                config=dict(
                    dump_passes=args.dump_passes,
                    backend_type=args.backend_type,
                    execution_mode=args.execution_mode,
                    device_id=args.device,
                    enable_chip_swimlane=args.enable_chip_swimlane,
                ),
                rtol=1e-3,
                atol=1e-3,
                compare_fn={"routed_y": compare_local_compute},
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
            if args.compile_only or args.golden_data:
                break
        if args.compile_only or args.golden_data:
            break
