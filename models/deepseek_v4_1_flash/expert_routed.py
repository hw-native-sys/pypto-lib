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

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from pypto.runtime import pto_isa_include_dir
from simpler_setup.environment import PROJECT_ROOT as SIMPLER_ROOT


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
    from dataclasses import replace

    import config

    test_experts = args.ep * args.experts_per_rank
    if args.experts_per_rank <= 0 or not config.FLASH.num_experts_per_tok <= test_experts <= 384:
        parser.error("--experts-per-rank must give between TopK and 384 total routed experts")
    config.FLASH = replace(config.FLASH, n_routed_experts=test_experts)
    config.N_LOCAL_EXPERTS = args.experts_per_rank
    device_ids = list(range(args.ep)) if args.device is None else [int(value) for value in args.device.split(",")]
    if len(device_ids) != args.ep or len(set(device_ids)) != args.ep:
        parser.error("--device must name one distinct device per EP rank")


from config import FLASH as M
from config import EP_SIZE as EP_WORLD_SIZE, MOE_RECV_MAX as RECV_MAX
from config import N_LOCAL_EXPERTS, MOE_TOKENS
from config import DSPARK_N_ROUTED_EXPERTS, DSPARK_NUM_EXPERTS_PER_TOK

from quant import pack_mxfp4_weight_tiles, unpack_mxfp4_weight_tiles


# model config
D = M.hidden_size
TOPK = M.num_experts_per_tok
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


def _pipeline_workspace_layout(ep, local_experts, tokens, topk, hidden, intermediate, shared=1):
    """Describe the private fused MoE communication workspace."""
    if ep not in (2, 4, 8) or local_experts <= 0 or tokens <= 0 or tokens % 16:
        raise ValueError("Invalid EP, expert count or physical token capacity")
    if hidden % 256 or intermediate % 256 or shared not in (0, 1, 2) or topk <= 0:
        raise ValueError("Invalid persistent MegaMoE shape")
    experts = ep * local_experts
    if not topk <= experts <= 384:
        raise ValueError("TopK/expert count is outside the model contract")
    result = {}
    offset = 0

    def align(value, multiple):
        return (value + multiple - 1) // multiple * multiple

    def allocate(name, count, stride, alignment=None):
        nonlocal offset
        offset = align(offset, stride if alignment is None else alignment)
        result[name] = offset
        offset += count * stride

    def fp8(name, rows, columns):
        allocate(name + ".data", rows * columns, 1, 512)
        allocate(name + ".sf", rows * ((columns + 63) // 64), 2, 512)

    max_blocks = (ep * tokens * topk + 255) // 256 + local_experts
    ring_blocks = min(max_blocks, 128 if hidden // 256 <= 16 else 32)
    shared_blocks = (tokens + 255) // 256 if shared else 0
    counters = shared_blocks + max_blocks
    full_rows, ring_rows = max_blocks * 256, ring_blocks * 256
    src_indices = experts * tokens * topk
    allocate("sync.grid_sync_count", 3, 4, 512)
    allocate("sync.rank_barriers", 2, 16, 512)
    allocate("metadata.worker_expert_count", 32 * experts, 4, 512)
    allocate("metadata.total_expert_count", experts, 4)
    allocate("metadata.combine_ready_prefix", ep, 128)
    allocate("metadata.expert_recv_count", experts, 4)
    allocate("metadata.expert_recv_count_sum", local_experts, 512)
    allocate("metadata.num_m_blocks_certificate", 1, 4, 128)
    allocate("metadata.src_token_metadata", src_indices, 8, 512)
    allocate("metadata.staged_token_topk_idx", src_indices, 4, 512)
    fp8("input.acts", tokens, hidden)
    allocate("input.topk_idx", tokens * topk, 8, 512)
    allocate("input.topk_weights", tokens * topk, 4, 512)
    allocate("progress.blocks", 4 * counters, 128)
    allocate("progress.num_issued_gemms", 32, 128)
    allocate("progress.metadata_ready", 32, 128)
    result["num_progress_bytes"] = offset - result["progress.blocks"]
    fp8("routed.l1_acts", ring_rows, hidden)
    allocate("routed.topk_weights", full_rows, 4, 512)
    allocate("routed.src_metadata", full_rows, 4, 512)
    fp8("routed.l2_acts", ring_rows, intermediate)
    allocate("routed.output", ring_rows * hidden, 2, 512)
    allocate("combine_input", (topk + bool(shared)) * tokens * hidden, 2, 512)
    fp8("shared_l2_acts", align(tokens, 256) if shared else 0, shared * intermediate)
    allocate("route_m_blocks", tokens * topk, 4, 512)
    result.update(logical_bytes=offset, bytes=align(offset, 2 * 1024 * 1024),
                  max_blocks=max_blocks, ring_blocks=ring_blocks, ring_rows=ring_rows)
    return result


def fused_workspace_bytes(ep, local_experts, tokens, topk, hidden, intermediate):
    return _pipeline_workspace_layout(ep, local_experts, tokens, topk, hidden, intermediate)["bytes"]


def _fused_external_kernel(local_experts, topk):
    """Prepare a content-addressed, shape-specialized native entry."""
    kernels = Path(__file__).parent / "kernels"
    entry = kernels / "moe_fused.cpp"
    values = dict(MOE_EP=EP_WORLD_SIZE, MOE_EXPERTS=local_experts, MOE_TOKENS=MOE_TOKENS,
                  MOE_TOPK=topk, MOE_HIDDEN=D, MOE_INTER=MOE_INTER)
    text = "".join(f"#define {name} {value}\n" for name, value in values.items())
    text += f'#include "{entry.as_posix()}"\n'
    digest = hashlib.sha256(text.encode())
    for name in ("moe_fused.cpp", "moe_pipeline.hpp", "moe_pipeline_compute.hpp"):
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
    include_dirs = [pto_isa_include_dir(), kernels, SIMPLER_ROOT / "src/common/platform_comm"]
    for base in (cann / "compiler", cann, cann / (os.uname().machine + "-linux")):
        for relative in ("asc", "asc/include", "include"):
            if (base / relative).is_dir():
                include_dirs.append(base / relative)
    return pl.jit.extern(core_type="mixed", aic_source=wrapper, aiv_source=wrapper,
                         dual_aiv_dispatch=True, include_dirs=tuple(include_dirs))


@_fused_external_kernel(N_LOCAL_EXPERTS, TOPK)
def _fused_experts(
    indices: pl.Tensor, x: pl.Tensor, x_scale: pl.Tensor, weights: pl.Tensor,
    w1: pl.Tensor, w1_scale: pl.Tensor, w3: pl.Tensor, w3_scale: pl.Tensor,
    w2: pl.Tensor, w2_scale: pl.Tensor,
    shared_w1: pl.Tensor, shared_w1_scale: pl.Tensor,
    shared_w3: pl.Tensor, shared_w3_scale: pl.Tensor,
    shared_w2: pl.Tensor, shared_w2_scale: pl.Tensor,
    output: pl.Out[pl.Tensor], workspace: pl.InOut[pld.DistributedTensor],
    active_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    return output


DRAFT_LOCAL_EXPERTS = DSPARK_N_ROUTED_EXPERTS // EP_WORLD_SIZE
DRAFT_TOPK = DSPARK_NUM_EXPERTS_PER_TOK


@_fused_external_kernel(DRAFT_LOCAL_EXPERTS, DRAFT_TOPK)
def _fused_experts_draft(
    indices: pl.Tensor, x: pl.Tensor, x_scale: pl.Tensor, weights: pl.Tensor,
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
def routed_moe(
    indices: pl.Tensor,
    x_norm_mx: pl.Tensor,
    x_norm_scale: pl.Tensor,
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
    recv_x: pld.DistributedTensor,
    num_tokens: pl.Scalar[pl.INT32],
    local_experts: pl.constexpr = N_LOCAL_EXPERTS,
    topk: pl.constexpr = TOPK,
):
    """Execute shared/routed experts with persistent native tile pipelines."""
    if local_experts == DRAFT_LOCAL_EXPERTS and topk == DRAFT_TOPK:
        with pl.spmd(pl.system.available_cluster_count(), sync_start=True, name_hint="moe_fused_draft") as draft_ready:
            output = _fused_experts_draft(
                indices, x_norm_mx, x_norm_scale, weights,
                w1, w1_scale, w3, w3_scale, w2, w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                output, recv_x, num_tokens,
            )
        ready = draft_ready
    else:
        with pl.spmd(pl.system.available_cluster_count(), sync_start=True, name_hint="moe_fused") as main_ready:
            output = _fused_experts(
                indices, x_norm_mx, x_norm_scale, weights,
                w1, w1_scale, w3, w3_scale, w2, w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                output, recv_x, num_tokens,
            )
        ready = main_ready
    return ready


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
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
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
            source_scale = pl.slice(recv_mx_scale, [1, MOE_TOKENS * K_SCALE], [0, source_row * K_SCALE])
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
            down_ready = routed_moe(
                indices, source, source_scale, route_weights, output,
                routed_w1_packed, routed_w1_scale, routed_w3_packed, routed_w3_scale,
                routed_w2_packed, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
                workspace, active, local_experts=N_LOCAL_EXPERTS, topk=TOPK,
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


_LOCAL_COMPUTE_WINDOW_BYTES = fused_workspace_bytes(EP_WORLD_SIZE, N_LOCAL_EXPERTS, MOE_TOKENS, TOPK, D, MOE_INTER)


@pl.jit.host
def distributed_compute_test(
    recv_x: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[EP_WORLD_SIZE, 1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
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
            recv_x[rank], recv_mx_scale[rank], recv_weights[rank], recv_expert_count[rank],
            routed_w1_packed[rank], w1_scale, routed_w3_packed[rank], w3_scale, routed_w2_packed[rank], w2_scale,
            shared_w1[rank], sw1_scale, shared_w3[rank], sw3_scale, shared_w2[rank], sw2_scale,
            routed_y[rank], workspace, rank, device=rank,
        )


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

        gate = matmul_mx_golden(x_sub, x_scale, w1_fp8, w1_scale[e]).bfloat16().float()
        up = matmul_mx_golden(x_sub, x_scale, w3_fp8, w3_scale[e]).bfloat16().float()
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


# validation
OUTPUT_SENTINEL = 37.0


def golden_local_compute(tensors):
    """Fill active local output rows through the production expert golden."""
    values = dict(tensors)
    values["recv_y"] = torch.empty_like(tensors["routed_y"]).reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
    golden_expert_routed(values)
    output = tensors["routed_y"].reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
    for expert, count in enumerate(tensors["recv_expert_count"].reshape(-1).tolist()):
        output[expert, :count] = values["recv_y"][expert, :count]


def compare_local_compute(actual, expected, *, inputs, **kwargs):
    """Compare active expert rows and require exact inactive output guards."""
    from golden import ratio_reldiff

    actual_rows = actual.reshape(N_LOCAL_EXPERTS, RECV_MAX, D)
    expected_rows = expected.reshape_as(actual_rows)
    comparator = ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)
    for expert, count in enumerate(inputs["recv_expert_count"].reshape(-1).tolist()):
        if not torch.equal(actual_rows[expert, count:], expected_rows[expert, count:]):
            return False, f"expert {expert}: inactive output rows changed"
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
    elif case == "balanced":
        pattern = (36,)
    elif case == "tails":
        pattern = (15, 16, 17, 63, 64, 65)
    elif case == "bounds":
        pattern = (0, 1, RECV_MAX - 1, RECV_MAX)
    else:
        raise ValueError(f"Unknown local compute case: {case}")
    values = [min(RECV_MAX, pattern[(expert + tail_offset) % len(pattern)]) for expert in range(N_LOCAL_EXPERTS)]
    return torch.tensor(values, dtype=torch.int32).reshape(N_LOCAL_EXPERTS, 1)


def build_tensor_specs(case="bounds", *, tail_offset=0, seed=20261028):
    """Build local compute tensors with poisoned inactive capacity and lazy packed weights."""
    from functools import cache

    from golden import TensorSpec
    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    counts = validation_counts(case, tail_offset=tail_offset)

    @cache
    def receive_inputs():
        generator = torch.Generator().manual_seed(seed)
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
        TensorSpec("routed_y", [N_LOCAL_EXPERTS * RECV_MAX, D], torch.bfloat16, init_value=OUTPUT_SENTINEL),
    ]


def golden_distributed_compute(tensors):
    """Apply the unchanged guarded expert golden to every source rank."""
    for rank in range(EP_WORLD_SIZE):
        golden_local_compute({name: value[rank] for name, value in tensors.items()})


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
        build_tensor_specs(case, tail_offset=tail_offset + rank, seed=20261028 + rank)
        for rank in range(EP_WORLD_SIZE)
    ]
    specs = []
    for index, local in enumerate(local_specs[0][:-1]):
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
