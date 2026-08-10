# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
"""DeepSeek-V4 decode DSA-CP communication and grouped output-projection layouts."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import DECODE_TOKENS, FLASH as M, SP, TP_O_A, TP_O_B


# parallel layout
SP_SIZE = SP
O_A_SHARDS = TP_O_A
O_B_SHARDS = TP_O_B

# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
LOCAL_T = DECODE_TOKENS // SP_SIZE
GROUP_T = SP_SIZE * LOCAL_T
LOCAL_O_GROUPS = O_GROUPS // O_A_SHARDS

# tiling and collective-native layouts
TOKEN_TILE = 16
COMM_ROW_TILE = 1
LOCAL_T_PAD = (LOCAL_T + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE
GROUP_T_PAD = SP_SIZE * LOCAL_T_PAD
ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD
O_WINDOW_ROWS = SP_SIZE * LOCAL_T_PAD

if DECODE_TOKENS % SP_SIZE != 0:
    raise ValueError(f"decode tokens {DECODE_TOKENS} must be divisible by SP {SP_SIZE}")
if O_GROUPS % O_A_SHARDS != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by O-A shards {O_A_SHARDS}")
if O_A_SHARDS != SP_SIZE:
    raise ValueError(f"decode DSA all-to-all requires O-A shards {O_A_SHARDS} to equal SP {SP_SIZE}")
if O_B_SHARDS != SP_SIZE:
    raise ValueError(f"decode O-B reduce-scatter requires O-B shards {O_B_SHARDS} to equal SP {SP_SIZE}")


@pl.jit.inline
def sp_group_barrier(
    signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    sp_rank: pl.Scalar[pl.INT32],
    expected: pl.Scalar[pl.INT32],
):
    """Synchronize one contiguous physical SP group."""
    for peer_sp in pl.range(SP_SIZE):
        if peer_sp != sp_rank:
            pld.system.notify(
                target=signal, peer=group_base + peer_sp,
                offsets=[sp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    for source_sp in pl.range(SP_SIZE):
        if source_sp != sp_rank:
            pld.system.wait(signal=signal, offsets=[source_sp, 0], expected=expected, cmp=pld.WaitCmp.Ge)
    return signal


@pl.jit.inline
def reset_sp_group_signal(
    signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    sp_rank: pl.Scalar[pl.INT32],
):
    """Clear this rank's peer-credit cells after a collective completes."""
    reset_value = pl.cast(-2, pl.INT32)
    self_rank = group_base + sp_rank
    for source_sp in pl.range(SP_SIZE):
        if source_sp != sp_rank:
            pld.system.notify(
                target=signal, peer=self_rank,
                offsets=[source_sp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
            )
    return signal


@pl.jit.incore
def kv_token_allgather_step(
    kv_local: pl.Tensor[[LOCAL_T, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[GROUP_T, D], pl.BF16]],
    gather_window: pld.DistributedTensor[[GROUP_T, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    sp_rank: pl.Scalar[pl.INT32],
):
    """Gather rank-major token shards for the replicated KV projections."""
    target_row = sp_rank * LOCAL_T
    for peer_sp in pl.range(SP_SIZE):
        pld.tensor.put(
            dst=gather_window, peer=group_base + peer_sp, src=kv_local,
            dst_offsets=[target_row, 0], src_offsets=[0, 0], shape=[LOCAL_T, D],
            chunk_rows=COMM_ROW_TILE, chunk_cols=D, pipeline=True,
        )

    expected_one = pl.cast(1, pl.INT32)
    gather_signal = sp_group_barrier(gather_signal, group_base, sp_rank, expected_one)
    for group_t in pl.range(GROUP_T):
        group_out[group_t : group_t + 1, 0:D] = gather_window[group_t : group_t + 1, 0:D]
    expected_two = pl.cast(2, pl.INT32)
    gather_signal = sp_group_barrier(gather_signal, group_base, sp_rank, expected_two)
    gather_signal = reset_sp_group_signal(gather_signal, group_base, sp_rank)
    return group_out, gather_signal


@pl.jit.incore
def attention_token_head_all_to_all_step(
    attention_grouped: pl.Tensor[[O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], pl.BF16],
    local_groups_out: pl.Out[pl.Tensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16]],
    exchange_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    exchange_signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    sp_rank: pl.Scalar[pl.INT32],
):
    """Exchange contiguous output-group slabs into local-group, group-token order."""
    for destination_rank in pl.range(SP_SIZE):
        for local_group in pl.range(LOCAL_O_GROUPS):
            global_group = destination_rank * LOCAL_O_GROUPS + local_group
            source_row = global_group * LOCAL_T_PAD
            target_row = local_group * GROUP_T_PAD + sp_rank * LOCAL_T_PAD
            pld.tensor.put(
                dst=exchange_window, peer=group_base + destination_rank, src=attention_grouped,
                dst_offsets=[target_row, 0], src_offsets=[source_row, 0], shape=[LOCAL_T_PAD, O_GROUP_IN],
                chunk_rows=COMM_ROW_TILE, chunk_cols=O_GROUP_IN, pipeline=True,
            )

    expected_one = pl.cast(1, pl.INT32)
    exchange_signal = sp_group_barrier(exchange_signal, group_base, sp_rank, expected_one)
    for copy_row in pl.range(ATTENTION_WINDOW_ROWS):
        window_row = exchange_window[copy_row : copy_row + 1, 0:O_GROUP_IN]
        local_groups_out[copy_row : copy_row + 1, 0:O_GROUP_IN] = window_row
    expected_two = pl.cast(2, pl.INT32)
    exchange_signal = sp_group_barrier(exchange_signal, group_base, sp_rank, expected_two)
    exchange_signal = reset_sp_group_signal(exchange_signal, group_base, sp_rank)
    return local_groups_out, exchange_signal


@pl.jit.incore
def o_projection_reduce_scatter_step(
    o_partial: pl.Tensor[[GROUP_T_PAD, D], pl.FP32],
    local_out: pl.Out[pl.Tensor[[LOCAL_T_PAD, D], pl.BF16]],
    reduce_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    sp_rank: pl.Scalar[pl.INT32],
):
    """Sum O-B rank partials and scatter contiguous token-owner rows."""
    for owner_rank in pl.range(SP_SIZE):
        owner_source_row = owner_rank * LOCAL_T_PAD
        target_row = sp_rank * LOCAL_T_PAD
        pld.tensor.put(
            dst=reduce_window, peer=group_base + owner_rank, src=o_partial,
            dst_offsets=[target_row, 0], src_offsets=[owner_source_row, 0], shape=[LOCAL_T_PAD, D],
            chunk_rows=COMM_ROW_TILE, chunk_cols=D, pipeline=True,
        )

    expected_one = pl.cast(1, pl.INT32)
    reduce_signal = sp_group_barrier(reduce_signal, group_base, sp_rank, expected_one)
    for local_t in pl.range(LOCAL_T_PAD):
        acc = pl.load(reduce_window, [local_t, 0], [1, D])
        for source_rank in pl.range(1, SP_SIZE):
            source_partial_row = source_rank * LOCAL_T_PAD + local_t
            source_partial = pl.load(reduce_window, [source_partial_row, 0], [1, D])
            acc = pl.add(acc, source_partial)
        reduced_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
        pl.store(reduced_bf16, [local_t, 0], local_out)
    expected_two = pl.cast(2, pl.INT32)
    reduce_signal = sp_group_barrier(reduce_signal, group_base, sp_rank, expected_two)
    reduce_signal = reset_sp_group_signal(reduce_signal, group_base, sp_rank)
    return local_out, reduce_signal


@pl.jit
def decode_attention_cp_layout(
    kv_local: pl.Tensor[[LOCAL_T, D], pl.BF16],
    attention_grouped: pl.Tensor[[O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], pl.BF16],
    o_partial: pl.Tensor[[GROUP_T_PAD, D], pl.FP32],
    kv_group: pl.Out[pl.Tensor[[GROUP_T, D], pl.BF16]],
    attention_local_groups: pl.Out[pl.Tensor[[LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN], pl.BF16]],
    o_local: pl.Out[pl.Tensor[[LOCAL_T_PAD, D], pl.BF16]],
    kv_window: pld.DistributedTensor[[GROUP_T, D], pl.BF16],
    kv_signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[SP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    sp_rank: pl.Scalar[pl.INT32],
):
    """Apply the three decode attention communication seams on one rank."""
    kv_group, kv_signal = kv_token_allgather_step(
        kv_local, kv_group, kv_window, kv_signal, group_base, sp_rank,
    )
    attention_local_groups, attention_signal = attention_token_head_all_to_all_step(
        attention_grouped, attention_local_groups, attention_window, attention_signal, group_base, sp_rank,
    )
    o_local, o_signal = o_projection_reduce_scatter_step(
        o_partial, o_local, o_window, o_signal, group_base, sp_rank,
    )
    return kv_group, attention_local_groups, o_local, kv_signal, attention_signal, o_signal


@pl.jit.host
def l3_decode_attention_cp_layout(
    kv_local: pl.Tensor[[SP_SIZE, LOCAL_T, D], pl.BF16],
    attention_grouped: pl.Tensor[[SP_SIZE, O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], pl.BF16],
    o_partial: pl.Tensor[[SP_SIZE, GROUP_T_PAD, D], pl.FP32],
    kv_group: pl.Out[pl.Tensor[[SP_SIZE, GROUP_T, D], pl.BF16]],
    attention_local_groups: pl.Out[pl.Tensor[[SP_SIZE, ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16]],
    o_local: pl.Out[pl.Tensor[[SP_SIZE, LOCAL_T_PAD, D], pl.BF16]],
):
    """Launch the decode DSA-CP layout fixture on one physical SP group."""
    kv_window_buf = pld.alloc_window_buffer([GROUP_T, D], dtype=pl.BF16)
    kv_signal_buf = pld.alloc_window_buffer([SP_SIZE, 1], dtype=pl.INT32)
    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([SP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([SP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        kv_window = pld.window(kv_window_buf, [GROUP_T, D], dtype=pl.BF16)
        kv_signal = pld.window(kv_signal_buf, [SP_SIZE, 1], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [SP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [SP_SIZE, 1], dtype=pl.INT32)
        decode_attention_cp_layout(
            kv_local[rank], attention_grouped[rank], o_partial[rank],
            kv_group[rank], attention_local_groups[rank], o_local[rank],
            kv_window, kv_signal, attention_window, attention_signal, o_window, o_signal,
            0, rank, device=rank,
        )


def build_tensor_specs():
    """Build deterministic four-rank layout inputs."""
    import torch

    from golden import TensorSpec

    def init_kv_local():
        values = torch.arange(SP_SIZE * LOCAL_T * D, dtype=torch.int32)
        return values.remainder(251).reshape(SP_SIZE, LOCAL_T, D).to(torch.bfloat16)

    def init_attention_grouped():
        shape = (SP_SIZE, O_GROUPS * LOCAL_T_PAD, O_GROUP_IN)
        values = torch.arange(SP_SIZE * O_GROUPS * LOCAL_T_PAD * O_GROUP_IN, dtype=torch.int32)
        return values.remainder(127).reshape(shape).to(torch.bfloat16)

    def init_o_partial():
        shape = (SP_SIZE, GROUP_T_PAD, D)
        values = torch.arange(SP_SIZE * GROUP_T_PAD * D, dtype=torch.int32)
        return values.remainder(17).reshape(shape).to(torch.float32)

    attention_grouped_shape = [SP_SIZE, O_GROUPS * LOCAL_T_PAD, O_GROUP_IN]
    attention_local_shape = [SP_SIZE, LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN]
    return [
        TensorSpec("kv_local", [SP_SIZE, LOCAL_T, D], torch.bfloat16, init_value=init_kv_local),
        TensorSpec("attention_grouped", attention_grouped_shape, torch.bfloat16, init_value=init_attention_grouped),
        TensorSpec("o_partial", [SP_SIZE, GROUP_T_PAD, D], torch.float32, init_value=init_o_partial),
        TensorSpec("kv_group", [SP_SIZE, GROUP_T, D], torch.bfloat16, is_output=True),
        TensorSpec("attention_local_groups", attention_local_shape, torch.bfloat16, is_output=True),
        TensorSpec("o_local", [SP_SIZE, LOCAL_T_PAD, D], torch.bfloat16, is_output=True),
    ]


def golden_decode_attention_cp_layout(tensors):
    """Assemble the rank-major gather, grouped exchange, and reduced token shards."""
    import torch

    gathered_kv = tensors["kv_local"].reshape(GROUP_T, D)
    tensors["kv_group"][:] = gathered_kv.unsqueeze(0).expand_as(tensors["kv_group"])

    grouped = tensors["attention_grouped"].reshape(SP_SIZE, O_GROUPS, LOCAL_T_PAD, O_GROUP_IN)
    exchanged = torch.empty_like(tensors["attention_local_groups"])
    for destination_rank in range(SP_SIZE):
        for local_group in range(LOCAL_O_GROUPS):
            global_group = destination_rank * LOCAL_O_GROUPS + local_group
            target_row = local_group * GROUP_T_PAD
            group_rows = grouped[:, global_group].reshape(GROUP_T_PAD, O_GROUP_IN)
            exchanged[destination_rank, target_row : target_row + GROUP_T_PAD] = group_rows
    tensors["attention_local_groups"][:] = exchanged

    reduced = tensors["o_partial"].sum(dim=0)
    for rank in range(SP_SIZE):
        token_start = rank * LOCAL_T_PAD
        token_end = token_start + LOCAL_T_PAD
        tensors["o_local"][rank] = reduced[token_start:token_end].to(torch.bfloat16)


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(SP_SIZE)))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != SP_SIZE:
        parser.error(f"need exactly {SP_SIZE} devices, got {device_ids}")

    result = run_jit(
        fn=l3_decode_attention_cp_layout,
        specs=build_tensor_specs(),
        golden_fn=golden_decode_attention_cp_layout,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(platform=args.platform),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
