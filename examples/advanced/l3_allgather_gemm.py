# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""N-rank chunked L3 allgather + GEMM example.

This keeps the example intentionally small and explicit:

1. Specialize the distributed setup to the requested rank count.
2. Split each rank-local ``A`` shard into ``M_TILE`` row tiles.
3. Publish each chunk with explicit ``pld.tensor.put``.
4. Run one merged AIC kernel via ``pl.spmd_submit`` so multiple cores:
   - split the local chunk GEMMs across cores first
   - then wait on the exact remote ``[src_rank, chunk_idx]`` signal cell
     and compute the matching remote chunk.

The structure stays close to the earlier version: manual submit for
communication and one merged ``gemm_from_gathered`` compute kernel.
"""

from __future__ import annotations

import argparse
import statistics

import pypto.language as pl
import pypto.language.distributed as pld
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig
from pypto.runtime import RunConfig, benchmark as runtime_benchmark

import torch

MAT_K = 4096
MAT_N = 4096
MAT_M_LOCAL = 2048

M_TILE = 128
K_TILE = 128
N_TILE = 256
M_TILES = MAT_M_LOCAL // M_TILE
K_TILES = MAT_K // K_TILE
N_TILES = MAT_N // N_TILE

if MAT_M_LOCAL % M_TILE != 0:
    raise ValueError(f"MAT_M_LOCAL={MAT_M_LOCAL} must be divisible by M_TILE={M_TILE}")


def make_program(world_size: int, comm_cores: int, gemm_cores: int):
    MAT_M = world_size * MAT_M_LOCAL

    @pl.program
    class L3AllGatherGemm:

        @pl.function(type=pl.FunctionType.InCore)
        def comm_local_shard(
            self,
            local_a: pl.Tensor[[MAT_M_LOCAL, MAT_K], pl.FP16],
            gathered: pl.InOut[pld.DistributedTensor[[MAT_M, MAT_K], pl.FP16]],
            signal: pl.InOut[pld.DistributedTensor[[world_size, M_TILES], pl.INT32]],
        ) -> pld.DistributedTensor[[MAT_M, MAT_K], pl.FP16]:
            ctx = pld.get_comm_ctx(gathered)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)
            rank_row_offset = my_rank * MAT_M_LOCAL

            block_idx = pl.tile.get_block_idx()
            block_num = pl.tile.get_block_num()

            for chunk_idx in pl.range(block_idx, M_TILES, block_num):
                local_row_offset = chunk_idx * M_TILE
                global_row_offset = rank_row_offset + local_row_offset

                for kb in pl.range(K_TILES):
                    k0 = kb * K_TILE
                    local_chunk = pl.load(local_a, [local_row_offset, k0], [M_TILE, K_TILE])
                    gathered = pl.store(local_chunk, [global_row_offset, k0], gathered)

                for peer in pl.range(nranks):
                    if peer != my_rank:
                        pld.tensor.put(
                            gathered,
                            peer=peer,
                            src=local_a,
                            dst_offsets=[global_row_offset, 0],
                            src_offsets=[local_row_offset, 0],
                            shape=[M_TILE, MAT_K],
                            atomic=pld.AtomicType.None_,
                            chunk_rows=M_TILE,
                            chunk_cols=K_TILE,
                        )
                        pld.system.notify(
                            target=signal,
                            peer=peer,
                            offsets=[my_rank, chunk_idx],
                            value=1,
                            op=pld.NotifyOp.AtomicAdd,
                        )

            return gathered

        @pl.function(type=pl.FunctionType.AIC)
        def gemm_from_gathered(
            self,
            local_a: pl.Tensor[[MAT_M_LOCAL, MAT_K], pl.FP16],
            gathered: pl.InOut[pld.DistributedTensor[[MAT_M, MAT_K], pl.FP16]],
            signal: pl.InOut[pld.DistributedTensor[[world_size, M_TILES], pl.INT32]],
            weight: pl.Tensor[[MAT_K, MAT_N], pl.FP16],
            out: pl.Out[pl.Tensor[[MAT_M, MAT_N], pl.FP32]],
        ) -> pl.Tensor[[MAT_M, MAT_N], pl.FP32]:
            ctx = pld.get_comm_ctx(gathered)
            my_rank = pld.rank(ctx)
            nranks = pld.nranks(ctx)

            block_idx = pl.tile.get_block_idx()
            block_num = pl.tile.get_block_num()

            gemm_tasks = M_TILES * N_TILES

            # local chunk GEMM first, partitioned across (chunk_idx, nb) tasks.
            for task_idx in pl.range(block_idx, gemm_tasks, block_num):
                chunk_idx = task_idx // N_TILES
                nb = task_idx % N_TILES
                local_row_offset = chunk_idx * M_TILE
                global_row_offset = my_rank * MAT_M_LOCAL + local_row_offset
                n0 = nb * N_TILE
                local_acc = pl.create_tile([M_TILE, N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)

                for kb in pl.pipeline(K_TILES, stage=2):
                    k0 = kb * K_TILE
                    ak_mat: pl.Tile[[M_TILE, K_TILE], pl.FP16, pl.Mem.Mat] = pl.load(
                        local_a,
                        [local_row_offset, k0],
                        [M_TILE, K_TILE],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    ak_left: pl.Tile[[M_TILE, K_TILE], pl.FP16, pl.Mem.Left] = pl.move(
                        ak_mat,
                        target_memory=pl.Mem.Left,
                    )
                    wk_mat: pl.Tile[[K_TILE, N_TILE], pl.FP16, pl.Mem.Mat] = pl.load(
                        weight,
                        [k0, n0],
                        [K_TILE, N_TILE],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    wk_right: pl.Tile[[K_TILE, N_TILE], pl.FP16, pl.Mem.Right] = pl.move(
                        wk_mat,
                        target_memory=pl.Mem.Right,
                    )
                    if kb == 0:
                        local_acc = pl.matmul(ak_left, wk_right)
                    else:
                        local_acc = pl.matmul_acc(local_acc, ak_left, wk_right)

                out = pl.store(local_acc, [global_row_offset, n0], out)

            # remote chunk GEMMs: wait on the M-only signal cell, then compute the
            # assigned remote (chunk_idx, nb) tile.
            for src_rank in pl.range(nranks):
                if src_rank != my_rank:
                    for task_idx in pl.range(block_idx, gemm_tasks, block_num):
                        chunk_idx = task_idx // N_TILES
                        nb = task_idx % N_TILES
                        remote_row_offset = chunk_idx * M_TILE
                        remote_global_row_offset = src_rank * MAT_M_LOCAL + remote_row_offset
                        n0 = nb * N_TILE

                        pld.system.wait(
                            signal=signal,
                            offsets=[src_rank, chunk_idx],
                            expected=1,
                            cmp=pld.WaitCmp.Ge,
                        )

                        remote_acc = pl.create_tile([M_TILE, N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)

                        for kb in pl.pipeline(K_TILES, stage=2):
                            k0 = kb * K_TILE
                            ak_mat: pl.Tile[[M_TILE, K_TILE], pl.FP16, pl.Mem.Mat] = pl.load(
                                gathered,
                                [remote_global_row_offset, k0],
                                [M_TILE, K_TILE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            ak_left: pl.Tile[[M_TILE, K_TILE], pl.FP16, pl.Mem.Left] = pl.move(
                                ak_mat,
                                target_memory=pl.Mem.Left,
                            )
                            wk_mat: pl.Tile[[K_TILE, N_TILE], pl.FP16, pl.Mem.Mat] = pl.load(
                                weight,
                                [k0, n0],
                                [K_TILE, N_TILE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            wk_right: pl.Tile[[K_TILE, N_TILE], pl.FP16, pl.Mem.Right] = pl.move(
                                wk_mat,
                                target_memory=pl.Mem.Right,
                            )
                            if kb == 0:
                                remote_acc = pl.matmul(ak_left, wk_right)
                            else:
                                remote_acc = pl.matmul_acc(remote_acc, ak_left, wk_right)

                        out = pl.store(remote_acc, [remote_global_row_offset, n0], out)

            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            local_a: pl.Tensor[[MAT_M_LOCAL, MAT_K], pl.FP16],
            weight: pl.Tensor[[MAT_K, MAT_N], pl.FP16],
            out: pl.Out[pl.Tensor[[MAT_M, MAT_N], pl.FP32]],
            gathered: pl.InOut[pld.DistributedTensor[[MAT_M, MAT_K], pl.FP16]],
            signal: pl.InOut[pld.DistributedTensor[[world_size, M_TILES], pl.INT32]],
        ) -> pl.Tensor[[MAT_M, MAT_N], pl.FP32]:
            with pl.manual_scope():
                _gathered, _ = pl.spmd_submit(
                    self.comm_local_shard,
                    local_a,
                    gathered,
                    signal,
                    core_num=comm_cores,
                )
                out, _ = pl.spmd_submit(
                    self.gemm_from_gathered,
                    local_a,
                    gathered,
                    signal,
                    weight,
                    out,
                    core_num=gemm_cores,
                )

            return out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[world_size, MAT_M_LOCAL, MAT_K], pl.FP16],
            weight: pl.Tensor[[MAT_K, MAT_N], pl.FP16],
            outputs: pl.Out[pl.Tensor[[world_size, MAT_M, MAT_N], pl.FP32]],
        ) -> pl.Tensor[[world_size, MAT_M, MAT_N], pl.FP32]:
            gathered_buf = pld.alloc_window_buffer([MAT_M, MAT_K], dtype=pl.FP16)
            signal_buf = pld.alloc_window_buffer([world_size, M_TILES], dtype=pl.INT32)

            for rank in pl.range(pld.world_size()):
                gathered = pld.window(gathered_buf, [MAT_M, MAT_K], dtype=pl.FP16)
                signal = pld.window(signal_buf, [world_size, M_TILES], dtype=pl.INT32)
                self.chip_orch(inputs[rank], weight, outputs[rank], gathered, signal, device=rank)

            return outputs

    return L3AllGatherGemm


def build_inputs(world_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    rank_inputs = []
    for _rank in range(world_size):
        shard = torch.randn((MAT_M_LOCAL, MAT_K), dtype=torch.float32) * 0.1
        rank_inputs.append(shard.to(torch.float16))

    weight = (torch.randn((MAT_K, MAT_N), dtype=torch.float32) * 0.1).to(torch.float16)
    return torch.stack(rank_inputs), weight


def expected_outputs(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    world_size = inputs.shape[0]
    gathered = inputs.reshape(world_size * MAT_M_LOCAL, MAT_K).to(torch.float32)
    expected = torch.matmul(gathered, weight.to(torch.float32))
    return expected.unsqueeze(0).expand(world_size, -1, -1).contiguous()


def summarize_us(samples: list[float]) -> str:
    if not samples:
        return "n/a"
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return (
        f"min={min(samples):.1f} median={statistics.median(samples):.1f} "
        f"mean={statistics.fmean(samples):.1f} max={max(samples):.1f} stdev={stdev:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an N-rank chunked TPUT allgather + GEMM example.")
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="0,1",
        help="comma-separated device IDs; rank count is inferred from the list length",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument(
        "--comm-cores",
        type=int,
        default=4,
        help="SPMD cores for comm_local_shard",
    )
    parser.add_argument(
        "--gemm-cores",
        type=int,
        default=8,
        help="SPMD cores for gemm_from_gathered",
    )
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument(
        "--benchmark-rounds",
        type=int,
        default=20,
        help="measured benchmark rounds (warmup excluded)",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=5,
        help="warmup launches discarded before benchmark measurement",
    )
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",") if device]
    world_size = len(device_ids)
    if world_size < 2:
        parser.error(f"need at least 2 devices for this example, got {device_ids}")
    if args.comm_cores <= 0:
        parser.error(f"--comm-cores must be > 0, got {args.comm_cores}")
    if args.gemm_cores <= 0:
        parser.error(f"--gemm-cores must be > 0, got {args.gemm_cores}")
    if args.benchmark_rounds <= 0:
        parser.error(f"--benchmark-rounds must be > 0, got {args.benchmark_rounds}")
    if args.benchmark_warmup < 0:
        parser.error(f"--benchmark-warmup must be >= 0, got {args.benchmark_warmup}")

    if args.comm_cores > M_TILES:
        print(
            f"warning: --comm-cores={args.comm_cores} > M_TILES={M_TILES}; "
            "extra comm cores will be idle",
        )
    gemm_tasks = M_TILES * N_TILES
    if args.gemm_cores > gemm_tasks:
        print(
            f"warning: --gemm-cores={args.gemm_cores} > gemm_tasks={gemm_tasks}; "
            "extra gemm cores will be idle",
        )

    program = make_program(world_size, args.comm_cores, args.gemm_cores)
    compiled = ir.compile(
        program,
        platform=args.platform,
        distributed_config=DistributedConfig(
            device_ids=device_ids,
            num_sub_workers=0,
        ),
    )

    if args.compile_only:
        return 0

    MAT_M = world_size * MAT_M_LOCAL
    inputs, weight = build_inputs(world_size)
    outputs = torch.zeros((world_size, MAT_M, MAT_N), dtype=torch.float32)
    if args.benchmark:
        inputs.share_memory_()
        weight.share_memory_()
        outputs.share_memory_()
    run_config = RunConfig(
        platform=args.platform,
        enable_l2_swimlane=args.enable_l2_swimlane,
    )
    compiled(inputs, weight, outputs, config=run_config)

    expected = expected_outputs(inputs, weight)
    max_diff = (outputs - expected).abs().max().item()
    passed = torch.allclose(outputs, expected, rtol=1e-2, atol=1e-2)

    print(f"max diff vs torch reference: {max_diff:.6f}")

    if not passed:
        print("chunked TPUT allgather GEMM mismatch")
        return 1

    print("chunked TPUT allgather GEMM passed")

    if args.benchmark:
        outputs.zero_()
        try:
            stats = runtime_benchmark(
                compiled,
                (inputs, weight, outputs),
                rounds=args.benchmark_rounds,
                warmup=args.benchmark_warmup,
                config=run_config,
            )
        except RuntimeError as exc:
            print(f"benchmark unavailable: {exc}")
            return 1

        print(
            f"benchmark rounds={stats.rounds} warmup={stats.warmup}"
        )
        print(f"benchmark device_wall_us: {summarize_us(stats.device_wall_us)}")

        host_round_us = stats.per_round("host")
        if host_round_us:
            print(f"benchmark host_round_us: {summarize_us(host_round_us)}")

        union_round_us = stats.per_round("union")
        if union_round_us:
            print(f"benchmark union_round_us: {summarize_us(union_round_us)}")

        if stats.all_zero_device:
            print(
                "benchmark note: device_wall_us is all zero; on *sim platforms or non-STRACE runtimes, "
                "fall back to host/union timing"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
