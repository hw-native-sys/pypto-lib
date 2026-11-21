# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""DeepSeek-V4.1 fused shared and routed experts with distributed transport."""

import argparse
import hashlib
import os
from pathlib import Path
from functools import lru_cache

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--case", choices=["all", "bounds", "tails", "zero", "full", "balanced"], default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()
    import config

    test_experts = args.ep * args.experts_per_rank
    if args.experts_per_rank <= 0 or not config.FLASH.num_experts_per_tok <= test_experts <= 384:
        parser.error("--experts-per-rank must give between TopK and 384 total routed experts")
    device_ids = list(range(args.ep)) if args.device is None else [int(value) for value in args.device.split(",")]
    if len(device_ids) != args.ep or len(set(device_ids)) != args.ep:
        parser.error("--device must name one distinct device per EP rank")


from config import FLASH as M
from config import EP_SIZE as EP_WORLD_SIZE, MOE_RECV_MAX as RECV_MAX
from config import MOE_TOKENS

from expert_routed import (
    H_SCALE, K_SCALE, MX_GROUP, MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS,
    MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS, OUTPUT_SENTINEL, build_tensor_specs,
    compare_local_compute, golden_local_compute,
)


N_LOCAL_EXPERTS = args.experts_per_rank if __name__ == "__main__" else M.n_routed_experts // EP_WORLD_SIZE
D = M.hidden_size
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size


def workspace_bytes(ep, local_experts, tokens, topk, hidden, intermediate):
    """Size the symmetric workspace for the routed experts and one shared expert."""
    if ep not in (2, 4, 8) or local_experts <= 0 or tokens <= 0 or tokens % 16:
        raise ValueError("Invalid EP, expert count or physical token capacity")
    if hidden % 256 or intermediate % 256 or topk <= 0:
        raise ValueError("Invalid persistent MegaMoE shape")
    experts = ep * local_experts
    if not topk <= experts <= 384:
        raise ValueError("TopK/expert count is outside the model contract")
    offset = 0

    def align(value, multiple):
        return (value + multiple - 1) // multiple * multiple

    def allocate(count, stride, alignment=None):
        nonlocal offset
        offset = align(offset, stride if alignment is None else alignment)
        offset += count * stride

    def fp8(rows, columns):
        allocate(rows * columns, 1, 512)
        allocate(rows * ((columns + 63) // 64), 2, 512)

    max_blocks = (ep * tokens * topk + 255) // 256 + local_experts
    ring_blocks = min(max_blocks, 128 if hidden // 256 <= 16 else 32)
    shared_blocks = (tokens + 255) // 256
    counters = shared_blocks + max_blocks
    full_rows, ring_rows = max_blocks * 256, ring_blocks * 256
    src_indices = experts * tokens * topk
    allocate(3, 4, 512)  # sync.grid_sync_count
    allocate(2, 16, 512)  # sync.rank_barriers
    allocate(32 * experts, 4, 512)  # metadata.worker_expert_count
    allocate(experts, 4)  # metadata.total_expert_count
    allocate(ep, 128)  # metadata.combine_ready_prefix
    allocate(experts, 4)  # metadata.expert_recv_count
    allocate(local_experts, 512)  # metadata.expert_recv_count_sum
    allocate(1, 4, 128)  # metadata.num_m_blocks_certificate
    allocate(src_indices, 8, 512)  # metadata.src_token_metadata
    allocate(src_indices, 4, 512)  # metadata.staged_token_topk_idx
    fp8(tokens, hidden)  # input.acts
    allocate(tokens * topk, 8, 512)  # input.topk_idx
    allocate(tokens * topk, 4, 512)  # input.topk_weights
    allocate(4 * counters, 128)  # progress.blocks
    allocate(32, 128)  # progress.num_issued_gemms
    allocate(32, 128)  # progress.metadata_ready
    fp8(ring_rows, hidden)  # routed.l1_acts
    allocate(full_rows, 4, 512)  # routed.topk_weights
    allocate(full_rows, 4, 512)  # routed.src_metadata
    fp8(ring_rows, intermediate)  # routed.l2_acts
    allocate(ring_rows * hidden, 2, 512)  # routed.output
    allocate((topk + 1) * tokens * hidden, 2, 512)  # combine_input
    fp8(align(tokens, 256), intermediate)  # shared_l2_acts
    allocate(tokens * topk, 4, 512)  # route_m_blocks
    return align(offset, 2 * 1024 * 1024)


def _fused_external_kernel(local_experts, topk):
    """Prepare a content-addressed, shape-specialized native entry."""
    kernels = Path(__file__).parent / "kernels"
    values = dict(MOE_EP=EP_WORLD_SIZE, MOE_EXPERTS=local_experts, MOE_TOKENS=MOE_TOKENS,
                  MOE_TOPK=topk, MOE_HIDDEN=D, MOE_INTER=MOE_INTER)
    text = "".join(f"#define {name} {value}\n" for name, value in values.items())
    text += '#include "moe_fused.cpp"\n'
    digest = hashlib.sha256(text.encode())
    for name in ("moe_fused.cpp", "mx_quant.hpp"):
        digest.update((kernels / name).read_bytes())
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    directory = cache / "pypto-lib" / "moe-fused" / digest.hexdigest()
    wrapper = directory / "entry.cpp"
    if not wrapper.exists():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f"entry.{os.getpid()}.tmp"
        temporary.write_text(text)
        temporary.replace(wrapper)
    elif wrapper.read_text() != text:
        raise RuntimeError("Fused MoE source cache mismatch")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/latest"))
    include_dirs = [kernels]
    for base in (cann / "compiler", cann, cann / (os.uname().machine + "-linux")):
        for relative in ("asc", "asc/include", "include"):
            path = (base / relative).resolve()
            if path.is_dir() and path not in include_dirs:
                include_dirs.append(path)
    return pl.jit.extern(core_type="mixed", aic_source=wrapper, aiv_source=wrapper,
                         dual_aiv_dispatch=True, include_dirs=tuple(include_dirs))


@lru_cache(maxsize=None)
def specialize_mega_moe(local_experts, topk):
    """Bind the model entry's local expert count and route width to its mixed-core kernel."""
    @_fused_external_kernel(local_experts, topk)
    def _fused_experts(
        indices: pl.Tensor, x: pl.Tensor, weights: pl.Tensor,
        w1: pl.Tensor, w1_scale: pl.Tensor, w3: pl.Tensor, w3_scale: pl.Tensor,
        w2: pl.Tensor, w2_scale: pl.Tensor,
        shared_w1: pl.Tensor, shared_w1_scale: pl.Tensor,
        shared_w3: pl.Tensor, shared_w3_scale: pl.Tensor,
        shared_w2: pl.Tensor, shared_w2_scale: pl.Tensor,
        output: pl.Out[pl.Tensor], workspace: pl.InOut[pld.DistributedTensor],
        active_tokens: pl.Scalar[pl.INT32],
    ) -> pl.Tensor:
        return output

    @pl.jit.inline(auto_scope=False)
    def mega_moe(
        indices: pl.Tensor,
        normalized: pl.Tensor,
        weights: pl.Tensor,
        output: pl.Tensor,
        w1: pl.Tensor,
        w1_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
        w3: pl.Tensor,
        w3_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
        w2: pl.Tensor,
        w2_scale: pl.Tensor[None, pl.FP8E8M0, pl.MX_B_NN],
        shared_w1: pl.Tensor,
        shared_w1_scale: pl.Tensor,
        shared_w3: pl.Tensor,
        shared_w3_scale: pl.Tensor,
        shared_w2: pl.Tensor,
        shared_w2_scale: pl.Tensor,
        workspace: pld.DistributedTensor,
        num_tokens: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(pl.system.available_cluster_count(), sync_start=True, name_hint="moe_fused") as ready:
            output = _fused_experts(
                indices, normalized, weights,
                w1, w1_scale, w3, w3_scale, w2, w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                output, workspace, num_tokens,
            )
        return ready

    return mega_moe


# Kernel used by the independent, guarded expert-row validation below.
_validation_moe = specialize_mega_moe(N_LOCAL_EXPERTS, TOPK)


@pl.jit(auto_scope=False)
def local_compute_test(
    normalized: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    routed_y: pl.InOut[pl.Tensor[[N_LOCAL_EXPERTS * RECV_MAX, D], pl.BF16]],
    workspace: pld.DistributedTensor,
    rank: pl.Scalar[pl.INT32],
):
    """Validate expert rows through the production EP pipeline."""
    recv_x_flat = pl.reshape(normalized, [N_LOCAL_EXPERTS * RECV_MAX, D])
    indices = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
    route_weights = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.FP32)
    output = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    completed = pl.array.create(1, pl.TASK_ID)
    completed[0] = pl.system.task_dummy(deps=[])
    peer = (rank + 1) % EP_WORLD_SIZE
    for expert in pl.range(N_LOCAL_EXPERTS):
        count = pl.read(recv_expert_count, [expert, 0])
        for begin in pl.range(0, RECV_MAX, MOE_TOKENS):
            active = pl.max(0, pl.min(MOE_TOKENS, count - begin))
            source_row = expert * RECV_MAX + begin
            source = pl.slice(recv_x_flat, [MOE_TOKENS, D], [source_row, 0])
            with pl.spmd(1, name_hint="expert_test_routes", deps=[completed[0]]):
                writer = pl.tile.get_block_idx()
                for row in pl.range(writer, MOE_TOKENS):
                    for slot in pl.range(TOPK):
                        index = pl.cast(-1, pl.INT32)
                        weight = pl.cast(0.0, pl.FP32)
                        if row < active and slot == 0:
                            index = pl.cast(peer * N_LOCAL_EXPERTS + expert, pl.INT32)
                            weight = pl.read(recv_weights, [expert, begin + row])
                        pl.write(indices, [row, slot], index)
                        pl.write(route_weights, [row, slot], weight)
            down_ready = _validation_moe(
                indices, source, route_weights, output,
                routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
                routed_w2_packed, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                workspace, active,
            )
            with pl.spmd(64, name_hint="ffn_copy", deps=[down_ready]) as copied:
                core = pl.tile.get_block_idx()
                for work in pl.range(core, (active + 1) // 2, 64):
                    local_row = work * 2
                    valid = pl.min(2, active - local_row)
                    result = pl.load(output, [local_row, 0], [2, D], valid_shape=[valid, D])
                    masked = pl.set_validshape(result, valid, D)
                    routed_y = pl.store(masked, [source_row + local_row, 0], routed_y)
            completed[0] = copied
    return routed_y


_LOCAL_COMPUTE_WINDOW_BYTES = workspace_bytes(EP_WORLD_SIZE, N_LOCAL_EXPERTS, MOE_TOKENS, TOPK, D, MOE_INTER)


@pl.jit.host
def distributed_compute_test(
    normalized: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
    recv_weights: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w3_packed: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w2_packed: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0],
    shared_w1: pl.Tensor[[EP_WORLD_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[EP_WORLD_SIZE, K_SCALE, MOE_INTER], pl.FP8E8M0],
    shared_w3: pl.Tensor[[EP_WORLD_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[EP_WORLD_SIZE, K_SCALE, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[EP_WORLD_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[EP_WORLD_SIZE, H_SCALE, D], pl.FP8E8M0],
    routed_y: pl.InOut[pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS * RECV_MAX, D], pl.BF16]],
):
    """Run guarded expert-row validation across EP peers."""
    storage = pld.alloc_window_buffer([_LOCAL_COMPUTE_WINDOW_BYTES], dtype=pl.UINT8)
    for rank in pl.range(pld.world_size()):
        workspace = pld.window(storage, [_LOCAL_COMPUTE_WINDOW_BYTES], dtype=pl.UINT8)
        w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w1_scale[rank]
        w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w3_scale[rank]
        w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN] = routed_w2_scale[rank]
        sw1_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[rank]
        sw3_scale: pl.Tensor[[K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[rank]
        sw2_scale: pl.Tensor[[H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[rank]
        local_compute_test(
            normalized[rank], recv_weights[rank], recv_expert_count[rank],
            routed_w1_packed[rank], w1_scale, routed_w3_packed[rank], w3_scale, routed_w2_packed[rank], w2_scale,
            shared_w1[rank], sw1_scale, shared_w3[rank], sw3_scale, shared_w2[rank], sw2_scale,
            routed_y[rank], workspace, rank, device=rank,
        )


def golden_distributed_compute(tensors):
    """Apply the unchanged guarded expert golden to every source rank."""
    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    for rank in range(EP_WORLD_SIZE):
        local = {name: value[rank] for name, value in tensors.items()}
        payload, codes = quantize_mxfp8_upward_e8m0(local["normalized"].reshape(-1, D).float())
        local["recv_x"] = payload.reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
        local["recv_mx_scale"] = pack_mx_a_scale(codes).reshape(1, -1)
        golden_local_compute(local)


def compare_distributed_compute(actual, expected, *, inputs, **kwargs):
    """Check each source rank with the original expert-row comparator."""
    for rank in range(EP_WORLD_SIZE):
        local_inputs = {name: value[rank] for name, value in inputs.items()}
        passed, message = compare_local_compute(actual[rank], expected[rank], inputs=local_inputs, **kwargs)
        if not passed:
            return False, f"rank {rank}: {message}"
    return True, ""


def build_distributed_tensor_specs(case="bounds", *, tail_offset=0):
    """Build independent peer inputs, replicated expert weights, and neutral shared weights."""
    from golden import TensorSpec

    local_specs = [
        build_tensor_specs(case, tail_offset=tail_offset + rank, seed=20261028 + rank, local_experts=N_LOCAL_EXPERTS)
        for rank in range(EP_WORLD_SIZE)
    ]

    def normalized():
        rows = []
        for rank in range(EP_WORLD_SIZE):
            g = torch.Generator().manual_seed(20261028 + rank)
            x = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16, generator=g)
            counts = next(s for s in local_specs[rank] if s.name == "recv_expert_count").create_tensor()
            for e, count in enumerate(counts.flatten().tolist()):
                x[e, count:] = float("nan")
            rows.append(x)
        return torch.stack(rows)

    specs = [TensorSpec("normalized", [EP_WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16,
                        init_value=normalized)]
    for index, local in enumerate(local_specs[0][:-1]):
        if local.name in ("recv_x", "recv_mx_scale"):
            continue
        shape = [EP_WORLD_SIZE, *local.shape]
        resident = "stacked" if local.name.startswith("routed_") else None
        initializers = tuple(local_specs[rank][index].init_value for rank in range(EP_WORLD_SIZE))
        if resident:
            initializers = (local.init_value,) * EP_WORLD_SIZE

        def initialize(values=initializers):
            return torch.stack([value() if callable(value) else value for value in values])

        specs.append(TensorSpec(local.name, shape, local.dtype, init_value=initialize, resident=resident))
    for layer, rows, columns in (("w1", D, MOE_INTER), ("w3", D, MOE_INTER), ("w2", MOE_INTER, D)):
        shape = [EP_WORLD_SIZE, rows, columns]
        scale_shape = [EP_WORLD_SIZE, rows // MX_GROUP, columns]

        def unit_scale(shape=scale_shape):
            return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        specs.append(TensorSpec("shared_" + layer, shape, torch.float8_e4m3fn, init_value=0, resident="stacked"))
        specs.append(TensorSpec("shared_" + layer + "_scale", scale_shape, torch.float8_e8m0fnu, init_value=unit_scale, resident="stacked"))
    specs.append(TensorSpec("routed_y", [EP_WORLD_SIZE, N_LOCAL_EXPERTS * RECV_MAX, D], torch.bfloat16, init_value=OUTPUT_SENTINEL))
    return specs


if __name__ == "__main__":
    from golden import run

    cases = ("bounds", "tails", "zero") if args.case == "all" else (args.case,)
    for case in cases:
        tail_rounds = (6 + N_LOCAL_EXPERTS - 1) // N_LOCAL_EXPERTS if case == "tails" else 1
        for tail_round in range(tail_rounds):
            print(f"Fused routed compute: case={case}, tail_round={tail_round}, ep={EP_WORLD_SIZE}, experts={N_LOCAL_EXPERTS}, stride={RECV_MAX}")
            result = run(
                fn=distributed_compute_test,
                specs=build_distributed_tensor_specs(case, tail_offset=tail_round * N_LOCAL_EXPERTS),
                golden_fn=golden_distributed_compute,
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                runtime_dir=args.runtime_dir,
                config=dict(
                    dump_passes=args.dump_passes,
                    backend_type=args.backend_type,
                    execution_mode=args.execution_mode,
                    distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
                    enable_chip_swimlane=args.enable_chip_swimlane,
                ),
                rtol=1e-3,
                atol=1e-3,
                compare_fn={"routed_y": compare_distributed_compute},
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
            if args.compile_only or args.golden_data:
                break
        if args.compile_only or args.golden_data:
            break
