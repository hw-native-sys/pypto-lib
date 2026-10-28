# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Grouped MXFP8 by tiled MXFP4 projection, host fixtures, and validation."""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
import pypto.language as pl
from pypto.runtime import pto_isa_include_dir

from config import FLASH as M
from quant import pack_mx_a_scale, pack_mx_b_scale, quantize_mxfp4_weight
from quant import quantize_mxfp8_upward_e8m0, unpack_mx_scale_codes


# model config
D = M.hidden_size
H = M.moe_intermediate_size
TOPK = M.num_experts_per_tok
GLOBAL_EXPERTS = M.n_routed_experts
EP = 8
LOCAL_EXPERTS = GLOBAL_EXPERTS // EP
ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

# tiling
M_TILE = 64
K_TILE = 256
N_TILE = 256


@pl.jit.extern(
    core_type="mixed",
    aic_source="cpp/grouped_mxfp8_mxfp4.cpp",
    aiv_source="cpp/grouped_mxfp8_mxfp4.cpp",
    dual_aiv_dispatch=True,
    include_dirs=[str(pto_isa_include_dir())],
)
def grouped_mxfp8_mxfp4_extern(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    chunks: pl.Tensor,
    out: pl.Out[pl.Tensor],
    k: pl.Scalar[pl.INDEX],
    n: pl.Scalar[pl.INDEX],
    experts: pl.Scalar[pl.INDEX],
    chunk_count: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    ...


@pl.jit.inline(auto_scope=False)
def grouped_mxfp8_mxfp4(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    chunks: pl.Tensor,
    out: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID],
    k: pl.constexpr,
    n: pl.constexpr,
    experts: pl.constexpr,
    chunk_count: pl.constexpr,
    rows: pl.constexpr,
):
    """Schedule ordinary-NZ LUT projection over staged 64-row chunks.

    K and N are positive multiples of 256. Each metadata row is
    (expert, chunk_index * 64, valid_rows, compact_row_start); input padding is zero.
    Weights are 256x256 tiled packed FP4, with globally packed MX_B_NN scales.
    """
    with pl.spmd(32, name_hint="grouped_fp4_lut_nz", deps=[ready]) as done:
        # Keep the explicit Out buffer: generic extern return inference caches the first call's shape.
        _extern_result = grouped_mxfp8_mxfp4_extern(
            x, x_scale, weight, weight_scale, chunks, out,
            k, n, experts, chunk_count,
        )
    return out, done


@pl.jit.inline(auto_scope=False)
def grouped_mxfp8_mxfp4_dsl(
    x: pl.Tensor,
    x_scale: pl.Tensor,
    weight: pl.Tensor,
    weight_scale: pl.Tensor,
    chunks: pl.Tensor,
    out: pl.Tensor,
    ready: pl.Scalar[pl.TASK_ID],
    k: pl.constexpr,
    n: pl.constexpr,
    experts: pl.constexpr,
    chunk_count: pl.constexpr,
    rows: pl.constexpr,
):
    """Run the mainline multi-hop DSL projection used as the benchmark reference."""
    x_scale_mx = pl.tensor.view(x_scale, [rows, k // 32], layout=pl.MX_A_ZZ)
    weight_scale_mx = pl.tensor.view(weight_scale, [experts * (k // 32), n], layout=pl.MX_B_NN)
    weight_flat = pl.reshape(weight, [experts * (k * n // 256), 128])
    with pl.spmd(
        32, name_hint="grouped_fp4_multihop", deps=[ready],
        optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=4)],
    ) as done:
        core = pl.tile.get_block_idx()
        for chunk in pl.range(0, chunk_count):
            for col in pl.range(0, n, N_TILE):
                work = chunk * (n // N_TILE) + col // N_TILE
                if work % 32 == core:
                    selected = pl.read(chunks, [chunk, 0])
                    for expert in pl.range(0, experts):
                        if expert == selected:
                            row = chunk * M_TILE
                            packed_row = expert * (k * n // 256) + (col // N_TILE) * k
                            packed = pl.load(weight_flat, [packed_row, 0], [K_TILE, 128], target_memory=pl.Mem.Vec)
                            fp4 = pl.reinterpret_view(packed, pl.FP4E2M1X2)
                            fp8 = pl.cast(fp4, pl.FP8E4M3FN)
                            rhs = pl.move(fp8, pl.Mem.Mat)
                            lhs = pl.load(x, [row, 0], [M_TILE, K_TILE])
                            lhs_scale = pl.load(x_scale_mx, [row, 0], [M_TILE, K_TILE // 32], target_memory=pl.Mem.Mat)
                            scale_row = expert * (k // 32)
                            rhs_scale = pl.load(weight_scale_mx, [scale_row, col], [K_TILE // 32, N_TILE], target_memory=pl.Mem.Mat)
                            acc = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                            for k0 in pl.pipeline(K_TILE, k, K_TILE, stage=2):
                                packed_next = pl.load(weight_flat, [packed_row + k0, 0], [K_TILE, 128], target_memory=pl.Mem.Vec)
                                fp4_next = pl.reinterpret_view(packed_next, pl.FP4E2M1X2)
                                fp8_next = pl.cast(fp4_next, pl.FP8E4M3FN)
                                rhs_next = pl.move(fp8_next, pl.Mem.Mat)
                                lhs_next = pl.load(x, [row, k0], [M_TILE, K_TILE])
                                scale_col = k0 // 32
                                lhs_scale_next = pl.load(x_scale_mx, [row, scale_col], [M_TILE, K_TILE // 32], target_memory=pl.Mem.Mat)
                                scale_row_next = scale_row + scale_col
                                rhs_scale_next = pl.load(weight_scale_mx, [scale_row_next, col], [K_TILE // 32, N_TILE], target_memory=pl.Mem.Mat)
                                acc = pl.matmul_mx_acc(acc, lhs_next, lhs_scale_next, rhs_next, rhs_scale_next)
                            out = pl.store(acc, [row, col], out)
    return out, done


def make_grouped_entry(k, n, experts, chunk_count, rows, *, use_cpp):
    """Bind projection dimensions before the frontend resolves inline variants."""
    projection = grouped_mxfp8_mxfp4 if use_cpp else grouped_mxfp8_mxfp4_dsl

    @pl.jit(auto_scope=False)
    def grouped_test(
        x: pl.Tensor,
        x_scale: pl.Tensor,
        weight: pl.Tensor,
        weight_scale: pl.Tensor,
        chunks: pl.Tensor,
        out: pl.Out[pl.Tensor],
    ):
        ready = pl.system.task_dummy(deps=[])
        out, done = projection(
            x, x_scale, weight, weight_scale, chunks, out, ready,
            k, n, experts, chunk_count, rows,
        )
        return out

    return grouped_test


def tensor_checksum(value):
    """Hash tensor storage bytes without numerically converting FP8/E8M0."""
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def group_host_routes(source_x, global_ids, route_weights, *, shard_rank=0, experts=LOCAL_EXPERTS):
    """Stage frozen global routes by local expert, then by source/token/top-k slot."""
    if source_x.ndim != 3 or global_ids.shape != route_weights.shape:
        raise ValueError("Expected source_x[EP,T,K] and matching global_ids/route_weights[EP,T,topk]")
    owner = global_ids // experts
    count_e = torch.zeros(experts, dtype=torch.int32)
    routes = []
    chunks = []
    staged_x = []
    staged_weights = []
    compact_start = 0
    for expert in range(experts):
        hits = torch.nonzero((owner == shard_rank) & (global_ids % experts == expert), as_tuple=False)
        count_e[expert] = hits.shape[0]
        routes.append(hits.to(torch.int32))
        for offset in range(0, hits.shape[0], M_TILE):
            selected = hits[offset:offset + M_TILE]
            valid = selected.shape[0]
            stage_row = len(chunks) * M_TILE
            chunks.append([expert, stage_row, valid, compact_start + offset])
            block = torch.zeros(M_TILE, source_x.shape[-1], dtype=torch.bfloat16)
            block[:valid] = source_x[selected[:, 0], selected[:, 1]]
            staged_x.append(block)
            weights = torch.zeros(M_TILE, 1, dtype=torch.float32)
            weights[:valid, 0] = route_weights[selected[:, 0], selected[:, 1], selected[:, 2]]
            staged_weights.append(weights)
        compact_start += hits.shape[0]
    if not chunks:
        raise ValueError("An all-zero local workload has no supported chunk")
    staged_bf16 = torch.cat(staged_x)
    x, x_codes = quantize_mxfp8_upward_e8m0(staged_bf16)
    active = torch.zeros(x.shape[0], dtype=torch.bool)
    for _, row, valid, _ in chunks:
        active[row:row + valid] = True
    x_codes[~active] = 127
    return dict(
        x=x, x_scale=pack_mx_a_scale(x_codes).reshape(1, -1).view(torch.float8_e8m0fnu),
        chunks=torch.tensor(chunks, dtype=torch.int32), count_e=count_e,
        route_weights=torch.cat(staged_weights), compact_routes=torch.cat(routes),
    )


def build_route_fixture(scenario="decode", *, seed=20261028, shard_rank=0):
    """Freeze model-shaped synthetic global384 gate choices outside the device invocation."""
    token_counts = {"decode": 72, "prefill-full": 80, "prefill-tail": 48}
    if scenario not in token_counts:
        raise ValueError(f"Unsupported model routing scenario: {scenario}")
    if not 0 <= shard_rank < EP:
        raise ValueError("shard_rank must be within EP8")
    tokens = token_counts[scenario]
    generator = torch.Generator().manual_seed(seed)
    # One maximal source slab keeps decode/prefill source prefixes and the gate identical.
    source = torch.randn(EP, 80, D, generator=generator)
    variance = source.square().mean(dim=-1, keepdim=True)
    source = (source * torch.rsqrt(variance + M.rms_norm_eps)).to(torch.bfloat16)
    gate_weight = torch.randn(GLOBAL_EXPERTS, D, generator=generator) / D**0.5
    correction_bias = torch.randn(GLOBAL_EXPERTS, generator=generator) * 0.05
    source = source[:, :tokens].contiguous()
    logits = F.linear(source.float(), gate_weight.float()) / M.gate_temperature
    scores = F.softplus(logits).sqrt()
    global_ids = (scores + correction_bias).topk(TOPK, dim=-1).indices.to(torch.int32)
    chosen = scores.gather(-1, global_ids.long())
    route_weights = chosen / (chosen.sum(-1, keepdim=True) + 1e-20) * M.routed_scaling_factor
    fixture = group_host_routes(source, global_ids, route_weights, shard_rank=shard_rank)
    fixture.update(
        global_ids=global_ids, global_route_weights=route_weights, source_x=source,
        metadata=dict(
            label="model-shaped synthetic; not checkpoint data", scenario=scenario, seed=seed,
            ep=EP, global_experts=GLOBAL_EXPERTS, local_experts=LOCAL_EXPERTS, topk=TOPK,
            source_tokens_per_rank=tokens, shard_rank=shard_rank,
            gate_weight_sha256=tensor_checksum(gate_weight), bias_sha256=tensor_checksum(correction_bias),
            global_ids_sha256=tensor_checksum(global_ids), count_e=fixture["count_e"].tolist(),
            active_rows=int(fixture["count_e"].sum()), chunk_count=fixture["chunks"].shape[0],
        ),
    )
    return fixture


@lru_cache(maxsize=3)
def generate_projection_weights(experts, k, n, *, projection="w1", seed_base=0):
    """Generate one selected projection with the existing per-expert RNG and tiled/NN packing."""
    from expert_routed import pack_mxfp4_weight_tiles

    if projection not in ROUTED_DEQUANT_STD:
        raise ValueError("projection must be w1, w3, or w2")
    if (k, n) != ((H, D) if projection == "w2" else (D, H)):
        raise ValueError("Model projection dimensions do not match the selected weight")
    packed = torch.empty(experts, k * n // 256, 128, dtype=torch.uint8)
    logical_codes = torch.empty(experts, k // 32, n, dtype=torch.uint8)
    for expert in range(experts):
        generator = torch.Generator().manual_seed(seed_base + expert * 3)
        if projection != "w1":
            torch.randn(H, D, generator=generator)
        if projection == "w2":
            torch.randn(H, D, generator=generator)
        value = torch.randn(n, k, generator=generator) * ROUTED_DEQUANT_STD[projection]
        payload, codes = quantize_mxfp4_weight(value)
        packed[expert] = pack_mxfp4_weight_tiles(payload, K_TILE, N_TILE).reshape(k * n // 256, 128)
        logical_codes[expert] = codes.t().contiguous()
    scales = pack_mx_b_scale(logical_codes.flatten(0, 1)).view(torch.float8_e8m0fnu)
    return packed, scales


def build_smoke_fixture(*, seed=20261028):
    """Build G3 K=N256 with counts[0,1,65], all FP4 codes and varying MX scales."""
    generator = torch.Generator().manual_seed(seed)
    counts = [0, 1, 65]
    source = torch.randn(1, sum(counts), 256, generator=generator).to(torch.bfloat16)
    source *= torch.exp2(torch.arange(256).div(32, rounding_mode="floor").remainder(4) - 2).bfloat16()
    ids = torch.tensor([1] + [2] * 65, dtype=torch.int32).reshape(1, -1, 1)
    weights = torch.linspace(0.1, 1.5, sum(counts)).reshape(1, -1, 1)
    fixture = group_host_routes(source, ids, weights, experts=3)
    packed = torch.arange(3 * 256 * 128, dtype=torch.int64).reshape(3, 256, 128).to(torch.uint8)
    packed = packed + torch.arange(3, dtype=torch.uint8).reshape(3, 1, 1) * 17
    expert = torch.arange(3).reshape(3, 1, 1)
    group = torch.arange(8).reshape(1, 8, 1)
    column = torch.arange(256).reshape(1, 1, 256)
    codes = (123 + (expert * 3 + group + column // 16) % 7).to(torch.uint8)
    fixture["weight"] = packed
    fixture["weight_scale"] = pack_mx_b_scale(codes.flatten(0, 1)).view(torch.float8_e8m0fnu)
    fixture["metadata"] = dict(label="bounded synthetic correctness", scenario="smoke", seed=seed, count_e=counts)
    return fixture


def golden_grouped(tensors):
    """Decode global NN scales and tiled weights one active expert at a time."""
    from expert_routed import unpack_mxfp4_weight_tiles
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized

    k = tensors["x"].shape[1]
    experts = tensors["weight"].shape[0]
    n = tensors["out"].shape[1]
    x_codes = unpack_mx_scale_codes(tensors["x_scale"].reshape(-1, k // 32), side="a")
    b_codes = unpack_mx_scale_codes(tensors["weight_scale"], side="b").reshape(experts, k // 32, n)
    tensors["out"].zero_()
    for expert in range(experts):
        expert_chunks = tensors["chunks"][tensors["chunks"][:, 0] == expert]
        if not expert_chunks.numel():
            continue
        packed = tensors["weight"][expert].reshape(2 * (k // K_TILE) * (n // N_TILE), K_TILE * N_TILE // 4)
        decoded = unpack_mxfp4_weight_tiles(packed, k, n, K_TILE, N_TILE)
        for _, row, valid, _ in expert_chunks.tolist():
            x = tensors["x"][row:row + valid]
            codes = x_codes[row:row + valid]
            result = golden_gemm_mxfp8_to_fp32_dequantized(x, codes, decoded, b_codes[expert])
            tensors["out"][row:row + valid] = result.to(torch.bfloat16)


def compare_grouped(actual, expected, *, inputs, **kwargs):
    """Compare active chunk rows separately and require zero in every chunk tail."""
    from golden.validation import ratio_reldiff

    compare = ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)
    for index, (_, row, valid, _) in enumerate(inputs["chunks"].tolist()):
        passed, message = compare(actual[row:row + valid], expected[row:row + valid], inputs=inputs, **kwargs)
        if not passed:
            return False, f"chunk {index}: {message}"
        if torch.count_nonzero(actual[row + valid:row + M_TILE]).item():
            return False, f"chunk {index}: nonzero padding rows"
    return True, ""


def build_tensor_specs(scenario="smoke", projection="gate", *, golden_data=None):
    """Build lazy model-weight specs or replay all tensor payloads from a captured data directory."""
    from golden.spec import TensorSpec

    if golden_data:
        data = Path(golden_data)
        names = ("x", "x_scale", "chunks")
        tensors = {name: torch.load(data / "in" / f"{name}.pt", weights_only=True) for name in names}
        weight = torch.load(data / "in" / "weight.pt", mmap=True, weights_only=True)
        n = torch.load(data / "out" / "out.pt", mmap=True, weights_only=True).shape[1]
        shapes = dict(x=tensors["x"].shape, x_scale=tensors["x_scale"].shape, chunks=tensors["chunks"].shape)
        shapes.update(weight=weight.shape, weight_scale=(weight.shape[0] * (tensors["x"].shape[1] // 32), n))
        return [TensorSpec(name, list(shapes[name]), dtype) for name, dtype in (
            ("x", torch.float8_e4m3fn), ("x_scale", torch.float8_e8m0fnu), ("weight", torch.uint8),
            ("weight_scale", torch.float8_e8m0fnu), ("chunks", torch.int32),
        )] + [TensorSpec("out", [tensors["x"].shape[0], n], torch.bfloat16)], None
    fixture = build_smoke_fixture() if scenario == "smoke" else build_route_fixture(scenario)
    k, n = (256, 256) if scenario == "smoke" else ((H, D) if projection == "down" else (D, H))
    if scenario != "smoke" and projection == "down":
        generator = torch.Generator().manual_seed(20261029)
        source = torch.randn(EP, fixture["metadata"]["source_tokens_per_rank"], H, generator=generator)
        staged = group_host_routes(source.bfloat16(), fixture["global_ids"], fixture["global_route_weights"])
        fixture.update({name: staged[name] for name in ("x", "x_scale")})
    cache = {}

    def get_weights():
        if "pair" not in cache:
            if scenario == "smoke":
                cache["pair"] = fixture["weight"], fixture["weight_scale"]
            else:
                which = "w2" if projection == "down" else "w1"
                cache["pair"] = generate_projection_weights(LOCAL_EXPERTS, k, n, projection=which)
        return cache["pair"]

    experts = 3 if scenario == "smoke" else LOCAL_EXPERTS
    specs = [
        TensorSpec("x", list(fixture["x"].shape), torch.float8_e4m3fn, init_value=lambda: fixture["x"]),
        TensorSpec("x_scale", list(fixture["x_scale"].shape), torch.float8_e8m0fnu, init_value=lambda: fixture["x_scale"]),
        TensorSpec("weight", [experts, k * n // 256, 128], torch.uint8, init_value=lambda: get_weights()[0]),
        TensorSpec("weight_scale", [experts * (k // 32), n], torch.float8_e8m0fnu, init_value=lambda: get_weights()[1]),
        TensorSpec("chunks", list(fixture["chunks"].shape), torch.int32, init_value=lambda: fixture["chunks"]),
        TensorSpec("out", [fixture["x"].shape[0], n], torch.bfloat16),
    ]
    return specs, fixture


def validate(argv=None):
    """Run bounded correctness by default, or an explicitly selected model-shaped projection."""
    from golden.runner import run
    from pypto import passes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=["cpp", "dsl"], default="cpp")
    parser.add_argument("--scenario", choices=["smoke", "decode", "prefill-full", "prefill-tail"], default="smoke")
    parser.add_argument("--projection", choices=["gate", "down"], default="gate")
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--memory-planner", choices=["pypto", "dsa-rp"], default="pypto")
    args = parser.parse_args(argv)
    specs, fixture = build_tensor_specs(args.scenario, args.projection, golden_data=args.golden_data)
    if fixture is not None:
        metadata = dict(fixture["metadata"])
        names = ("x", "x_scale", "chunks", "route_weights")
        metadata["checksums"] = {name: tensor_checksum(fixture[name]) for name in names}
        print(json.dumps(metadata, sort_keys=True), flush=True)
    planner = passes.MemoryPlanner.PYPTO if args.memory_planner == "pypto" else passes.MemoryPlanner.DSA_RP
    entry = make_grouped_entry(
        specs[0].shape[1], specs[-1].shape[1], specs[2].shape[0],
        specs[4].shape[0], specs[0].shape[0], use_cpp=args.impl == "cpp",
    )
    return run(
        fn=entry,
        specs=specs,
        golden_fn=golden_grouped,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        config=dict(
            backend_type=args.platform,
            execution_mode=False,
            device_id=args.device,
            memory_planner=planner,
            dump_passes=args.dump_passes,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        compare_fn={"out": compare_grouped},
    )


if __name__ == "__main__":
    result = validate()
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
