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
"""DeepSeek-V4.1 decode expert return and shared-plus-routed reduction."""

import argparse
from pathlib import Path

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.runtime import pto_isa_include_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=2)
    parser.add_argument("-d", "--device", type=str, default="0,1")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument(
        "--benchmark-fixture", action="store_true", help="use matched EP2 M48 or M72 zero-shared fixtures",
    )
    parser.add_argument(
        "--case", choices=["all", "pair", "uniform", "remote", "hot", "mixed", "zero"], default="all",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()
    if args.execution_mode:
        parser.error("the packed return kernel requires A5 device execution (VF, direct DMA, and cache control)")
    import config

    if args.tokens not in (48, 72):
        parser.error("--tokens must be 48 or 72")
    config.set_moe_tokens((args.tokens + 15) // 16 * 16)


from config import EP_SIZE, FLASH as M, MOE_RECV_MAX, MOE_TOKENS, N_LOCAL_EXPERTS, ROUTE_T_DYN


# model config
D = M.hidden_size
TOPK = M.num_experts_per_tok

# EP packet layout
PACKED_WIDTH = ((D * 2 + 479) // 480) * 256

# Validation dimensions.
VALIDATION_TOKENS = args.tokens if __name__ == "__main__" else MOE_TOKENS
VALIDATION_EXPERTS = 12
VALIDATION_ROWS = TOPK * EP_SIZE * MOE_TOKENS


@pl.jit.extern(
    core_type="aiv", source=Path(__file__).parent / "kernels" / "expert_routed_combine.cpp",
    include_dirs=(pto_isa_include_dir(),),
)
def _combine_routed(
    counts: pl.Tensor,
    metadata: pl.Tensor,
    routes: pl.Tensor,
    output: pl.Tensor,
    shared: pl.Tensor,
    result: pl.InOut[pl.Tensor],
    target: pl.InOut[pld.DistributedTensor[[ROUTE_T_DYN, PACKED_WIDTH], pl.BF16]],
    signals: pld.DistributedTensor,
    experts: pl.Scalar[pl.INDEX],
    stride: pl.Scalar[pl.INDEX],
    ranks: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
    active: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    # Distinguish return contracts when target and drafter share a compiled program.
    local_experts: pl.constexpr,
    route_stride: pl.constexpr,
) -> tuple[pl.Tensor, pld.DistributedTensor]:
    """Send packed BF16 rows and consume packet arrivals in TopK order."""
    return result, target


@pl.jit.inline(auto_scope=False)
def combine_routed(
    output: pl.Tensor,
    counts: pl.Tensor,
    routes: pl.Tensor,
    metadata: pl.Tensor,
    shared: pl.Tensor,
    result: pl.Tensor,
    target: pld.DistributedTensor[[ROUTE_T_DYN, PACKED_WIDTH], pl.BF16],
    signals: pld.DistributedTensor,
    active: pl.Scalar[pl.INT32],
    input_ready: pl.Scalar[pl.TASK_ID],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    stride: pl.constexpr = MOE_RECV_MAX,
):
    """Write active compact weighted expert rows and return their completion dependency.

    Each active token has six unique destination slots. Counts are bounded
    by stride, and output is compact in expert/source order. Result rows beyond
    active are unchanged. The caller allocates target first at the 512-byte-aligned
    communication domain base and
    fences prior peer consumption before reusing target; decode's next
    input dispatch provides that fence. Signals is an unused ABI slot.
    """
    workers = pl.system.available_aiv_count()
    with pl.spmd(workers, name_hint="routed_combine", deps=[input_ready], sync_start=True) as ready:
        result, target = _combine_routed(
            counts, metadata, routes, output, shared, result, target, signals,
            local_experts, stride, EP_SIZE, result.shape[0], active, 1, local_experts, stride,
        )
    return result, target, ready


@pl.jit.inline(auto_scope=False)
def _release_round(
    signals: pld.DistributedTensor,
    ready: pl.Scalar[pl.TASK_ID],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Fence peer consumption between validation rounds."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="return_release", deps=[ready]) as sent:
        for peer in pl.range(EP_SIZE):
            if peer != rank:
                pld.system.notify(target=signals, peer=peer, offsets=[rank, 0, 0], value=epoch, op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="return_release_wait", deps=[sent], no_dep_args=[signals]) as waited:
        for peer in pl.range(EP_SIZE):
            if peer != rank:
                pld.system.defer_wait(signal=signals, offsets=[peer, 0, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
    return waited


@pl.jit(auto_scope=False)
def _validate_rank(
    first: pl.Tensor[[VALIDATION_ROWS, D], pl.BF16],
    second: pl.Tensor[[VALIDATION_ROWS, D], pl.BF16],
    counts: pl.Tensor[[VALIDATION_EXPERTS, 1], pl.INT32],
    routes: pl.Tensor[[VALIDATION_EXPERTS, MOE_RECV_MAX], pl.INT32],
    metadata: pl.Tensor[[EP_SIZE, VALIDATION_EXPERTS], pl.INT32],
    shared_first: pl.Tensor[[MOE_TOKENS, D], pl.BF16],
    shared_second: pl.Tensor[[MOE_TOKENS, D], pl.BF16],
    active_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    result_first: pl.InOut[pl.Tensor[[MOE_TOKENS, D], pl.BF16]],
    result_second: pl.InOut[pl.Tensor[[MOE_TOKENS, D], pl.BF16]],
    target: pld.DistributedTensor[[MOE_TOKENS * TOPK, PACKED_WIDTH], pl.BF16],
    signals: pld.DistributedTensor[[EP_SIZE, VALIDATION_EXPERTS, 128], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    active = pl.read(active_tokens, [rank])
    anchor = pl.system.task_dummy(deps=[])
    started = _release_round(signals, anchor, rank, 1)
    result_first, target, first_ready = combine_routed(
        first, counts, routes, metadata, shared_first, result_first, target, signals, active, started,
        local_experts=VALIDATION_EXPERTS, stride=MOE_RECV_MAX,
    )
    consumed = _release_round(signals, first_ready, rank, 2)
    result_second, target, second_ready = combine_routed(
        second, counts, routes, metadata, shared_second, result_second, target, signals, active, consumed,
        local_experts=VALIDATION_EXPERTS, stride=MOE_RECV_MAX,
    )
    consumed_second = _release_round(signals, second_ready, rank, 3)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="return_test_reset", deps=[consumed_second]):
        anchor_value = pl.read(result_second, [0, 0])
        for peer in pl.range(EP_SIZE):
            pl.write(signals, [peer, 0, 0], pl.cast(0, pl.INT32))
        pl.write(result_second, [0, 0], anchor_value)
    return result_first, result_second


@pl.jit.host
def validate_return(
    first: pl.Tensor[[EP_SIZE, VALIDATION_ROWS, D], pl.BF16],
    second: pl.Tensor[[EP_SIZE, VALIDATION_ROWS, D], pl.BF16],
    counts: pl.Tensor[[EP_SIZE, VALIDATION_EXPERTS, 1], pl.INT32],
    routes: pl.Tensor[[EP_SIZE, VALIDATION_EXPERTS, MOE_RECV_MAX], pl.INT32],
    metadata: pl.Tensor[[EP_SIZE, EP_SIZE, VALIDATION_EXPERTS], pl.INT32],
    shared_first: pl.Tensor[[EP_SIZE, MOE_TOKENS, D], pl.BF16],
    shared_second: pl.Tensor[[EP_SIZE, MOE_TOKENS, D], pl.BF16],
    active_tokens: pl.Tensor[[EP_SIZE], pl.INT32],
    result_first: pl.InOut[pl.Tensor[[EP_SIZE, MOE_TOKENS, D], pl.BF16]],
    result_second: pl.InOut[pl.Tensor[[EP_SIZE, MOE_TOKENS, D], pl.BF16]],
):
    target_buffer = pld.alloc_window_buffer([MOE_TOKENS * TOPK, PACKED_WIDTH], dtype=pl.BF16)
    signal_buffer = pld.alloc_window_buffer([EP_SIZE, VALIDATION_EXPERTS, 128], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        target = pld.window(target_buffer, [MOE_TOKENS * TOPK, PACKED_WIDTH], dtype=pl.BF16)
        signals = pld.window(signal_buffer, [EP_SIZE, VALIDATION_EXPERTS, 128], dtype=pl.INT32)
        _validate_rank(
            first[rank], second[rank], counts[rank], routes[rank], metadata[rank],
            shared_first[rank], shared_second[rank], active_tokens, result_first[rank], result_second[rank],
            target, signals, rank, device=rank,
        )


def build_case(name, benchmark=False):
    """Build deterministic compact rows, poisoned capacity, and repeated-round outputs."""
    import torch

    tokens = VALIDATION_TOKENS
    active = torch.full((EP_SIZE,), tokens, dtype=torch.int32)
    if name == "mixed":
        active[0] = 0
        active[1:] = max(1, tokens - 11)
    elif name == "zero":
        active.zero_()
    slots = torch.arange(tokens * TOPK)
    expert_ids = torch.empty((EP_SIZE, tokens, TOPK), dtype=torch.int32)
    if benchmark and (EP_SIZE != 2 or tokens not in (48, 72) or name not in {"pair", "uniform", "remote"}):
        raise ValueError("the benchmark fixture requires EP2, 48 or 72 tokens, and a recorded routing case")
    if name in {"pair", "remote"} or (name == "uniform" and EP_SIZE == 2):
        if EP_SIZE != 2 or tokens not in (48, 72):
            raise ValueError("pair and remote require EP2 and 48 or 72 tokens")
        pair = round(tokens * TOPK / 32)
        remote = {"pair": pair, "uniform": tokens * TOPK // 2, "remote": tokens * TOPK - pair}[name]
        rows_per_expert = tokens * TOPK // VALIDATION_EXPERTS
        mask = torch.zeros(tokens * TOPK, dtype=torch.bool)
        for expert in range(VALIDATION_EXPERTS):
            count = remote // VALIDATION_EXPERTS + int(expert < remote % VALIDATION_EXPERTS)
            positions = [
                expert + VALIDATION_EXPERTS * ((j * 5 + expert * 7) % rows_per_expert)
                for j in range(count)
            ]
            mask[positions] = True
        for rank in range(EP_SIZE):
            ids = (rank ^ mask.long()) * VALIDATION_EXPERTS + slots % VALIDATION_EXPERTS
            expert_ids[rank] = ids.reshape(tokens, TOPK)
    else:
        for rank in range(EP_SIZE):
            ids = slots % TOPK if name == "hot" else (slots + rank) % (EP_SIZE * VALIDATION_EXPERTS)
            expert_ids[rank] = ids.reshape(tokens, TOPK)

    generator = torch.Generator().manual_seed(202611061448)
    samples = torch.randint(
        -32, 33, (EP_SIZE, MOE_TOKENS, TOPK, D), generator=generator, dtype=torch.int16,
    )
    values = (samples.float() / 16).bfloat16()
    next_values = (-values.float() + 0.125).bfloat16()
    shared_samples = torch.randint(-16, 17, (EP_SIZE, MOE_TOKENS, D), generator=generator)
    shared = (shared_samples.float() / 16).bfloat16()
    next_shared = (-shared.float() - 0.25).bfloat16()
    if benchmark:
        shared.zero_()
        next_shared.zero_()
    first = torch.full((EP_SIZE, VALIDATION_ROWS, D), float("nan"), dtype=torch.bfloat16)
    second = torch.full_like(first, float("nan"))
    counts = torch.zeros((EP_SIZE, VALIDATION_EXPERTS, 1), dtype=torch.int32)
    routes = torch.full((EP_SIZE, VALIDATION_EXPERTS, MOE_RECV_MAX), -1, dtype=torch.int32)
    metadata = torch.zeros((EP_SIZE, EP_SIZE, VALIDATION_EXPERTS), dtype=torch.int32)
    if benchmark:
        first.zero_()
        second.zero_()
        routes.zero_()
    for destination in range(EP_SIZE):
        compact = 0
        for expert in range(VALIDATION_EXPERTS):
            local_row = 0
            for source in range(EP_SIZE):
                ids = expert_ids[source, :int(active[source])].flatten()
                selected = torch.where(ids == destination * VALIDATION_EXPERTS + expert)[0]
                metadata[destination, source, expert] = len(selected)
                for route in selected.tolist():
                    token, slot = divmod(route, TOPK)
                    routes[destination, expert, local_row] = route
                    first[destination, compact] = values[source, token, slot]
                    second[destination, compact] = next_values[source, token, slot]
                    local_row += 1
                    compact += 1
            counts[destination, expert, 0] = local_row

    def reference(routed, shared_value, guard):
        result = shared_value.float().clone()
        for slot in range(TOPK):
            result.add_(routed[:, :, slot].float())
        for rank in range(EP_SIZE):
            result[rank, int(active[rank]):] = guard
        return result.bfloat16()

    return {
        "first": first, "second": second, "counts": counts, "routes": routes, "metadata": metadata,
        "shared_first": shared, "shared_second": next_shared, "active_tokens": active,
        "result_first": reference(values, shared, 127), "result_second": reference(next_values, next_shared, -127),
        "expert_ids": expert_ids, "route_values": values,
    }


if __name__ == "__main__":
    from functools import partial

    import torch
    from golden import TensorSpec, run
    from pypto.ir import DistributedConfig

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != EP_SIZE:
        parser.error("the device list must contain exactly --ep devices")
    cases = ["uniform", "hot", "mixed", "zero"] if args.case == "all" else [args.case]
    if args.case == "all" and EP_SIZE == 2:
        cases = ["pair", "uniform", "remote", "hot", "mixed", "zero"]
    if args.benchmark_fixture and args.case == "all":
        cases = ["pair", "uniform", "remote"]
    runtime_dir = args.runtime_dir
    for case_name in cases:
        values = build_case(case_name, benchmark=args.benchmark_fixture)
        initial = dict(values)
        initial["result_first"] = torch.full_like(values["result_first"], 127)
        initial["result_second"] = torch.full_like(values["result_second"], -127)
        specs = [
            TensorSpec(
                name, list(values[name].shape), values[name].dtype,
                init_value=partial(torch.clone, initial[name]),
            )
            for name in validate_return.param_names
        ]

        def golden(tensors):
            tensors["result_first"][:] = values["result_first"]
            tensors["result_second"][:] = values["result_second"]

        result = run(
            fn=validate_return,
            specs=specs,
            golden_fn=golden,
            rtol=0,
            atol=0,
            compile_only=args.compile_only,
            runtime_dir=runtime_dir,
            save_data=args.save_data,
            golden_data=args.golden_data,
            config={
                "backend_type": args.backend_type,
                "execution_mode": args.execution_mode,
                "enable_chip_swimlane": args.enable_chip_swimlane,
                "distributed_config": DistributedConfig(device_ids=device_ids),
            },
        )
        if not result.passed:
            raise RuntimeError(f"{case_name}: {result.error}")
        print(f"{case_name}: {result}", flush=True)
        runtime_dir = str(result.work_dir)
        if args.compile_only or args.golden_data:
            break
