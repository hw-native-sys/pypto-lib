# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Flash TP vocabulary-sharded embedding for packed prefill and decode IDs."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld

from config import DECODE_TOKENS, FLASH as M, PREFILL_TOKENS, TP_VOCAB


# Dynamic shape variables.
T_DYN = pl.dynamic("LOOKUP_EMBEDDING_T_DYN")
VOCAB_DYN = pl.dynamic("LOOKUP_EMBEDDING_VOCAB_DYN")

# model config
D = M.hidden_size
VOCAB = M.vocab_size

# parallelism
TP_CHOICES = (1, 2, 4, 8)
GROUP_T_MAX = max(DECODE_TOKENS, PREFILL_TOKENS)


def _parse_tp_vocab_argv():
    for index, token in enumerate(sys.argv):
        if token == "--tp-vocab" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if token.startswith("--tp-vocab="):
            return int(token.split("=", 1)[1])
    for index, token in enumerate(sys.argv):
        if token == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if token.startswith("--tp="):
            return int(token.split("=", 1)[1])
    return TP_VOCAB


TP_SIZE = _parse_tp_vocab_argv()
if TP_SIZE not in TP_CHOICES:
    raise ValueError(f"vocabulary TP must be one of {TP_CHOICES}, got {TP_SIZE}")
SP_T = GROUP_T_MAX // TP_SIZE

# tiling
HIDDEN_TILE = 512
SPMD_BLOCKS = 48
COMM_D_TILE = 4096
COMM_WORKERS = 24

# communication
ID_PUBLISH_TASKS = 1

if VOCAB % TP_SIZE != 0:
    raise ValueError(f"vocabulary size {VOCAB} must be divisible by TP {TP_SIZE}")


@pl.jit.inline
def lookup_embedding(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
) -> pl.Tensor[[T_DYN, D], pl.BF16]:
    token_count = pl.tensor.dim(input_ids, 0)
    work_items = token_count * (D // HIDDEN_TILE)
    for block in pl.spmd(SPMD_BLOCKS, name_hint="lookup_embedding"):
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            token_idx = work_idx // (D // HIDDEN_TILE)
            hidden_block = work_idx % (D // HIDDEN_TILE)
            hidden_offset = hidden_block * HIDDEN_TILE
            token_id = pl.tensor.read(input_ids, [token_idx])
            token_row = pl.cast(token_id, target_type=pl.INDEX)
            hidden_chunk = embed_weight[token_row : token_row + 1, hidden_offset : hidden_offset + HIDDEN_TILE]
            hidden_states[token_idx : token_idx + 1, hidden_offset : hidden_offset + HIDDEN_TILE] = hidden_chunk

    return hidden_states


@pl.jit.inline(auto_scope=False)
def gather_sp_ids(
    input_ids_local: pl.Tensor,
    input_ids_group: pl.Tensor,
    gather_window: pld.DistributedTensor[[1, GROUP_T_MAX], pl.INT64],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    producer_dep: pl.Scalar[pl.TASK_ID],
):
    """All-gather one contiguous token-ID shard inside a TP group."""
    group_base = my_rank // TP_SIZE * TP_SIZE
    tp_rank = my_rank % TP_SIZE
    input_ids_local_row = pl.reshape(input_ids_local, [1, SP_T])

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="embedding_tp_id_publish",
        deps=[producer_dep],
    ) as publish_tid:
        dst_col = tp_rank * SP_T
        for peer_tp in pl.range(TP_SIZE):
            pld.tensor.put(
                dst=gather_window, peer=group_base + peer_tp, src=input_ids_local_row,
                dst_offsets=[0, dst_col], src_offsets=[0, 0], shape=[1, SP_T],
            )

        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embedding_tp_id_wait", deps=[publish_tid]) as wait_tid:
        expected = pl.cast(ID_PUBLISH_TASKS, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=expected, cmp=pld.WaitCmp.Ge,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embedding_tp_id_copy", deps=[wait_tid]) as copy_tid:
        input_ids_group[0:1, 0:GROUP_T_MAX] = gather_window[0:1, 0:GROUP_T_MAX]

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embedding_tp_id_complete", deps=[copy_tid]):
        _completion_anchor = pl.read(input_ids_group, [0, 0])
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(ID_PUBLISH_TASKS + 1, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=gather_signal, offsets=[source_tp, 0],
                    expected=completion_expected, cmp=pld.WaitCmp.Ge,
                )

        reset_value = pl.cast(-(ID_PUBLISH_TASKS + 1), pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal, peer=my_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
    return input_ids_group


@pl.jit.inline(auto_scope=False)
def reduce_scatter_embedding_bf16(
    partial: pl.Tensor,
    local_out: pl.Tensor,
    scatter_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.BF16],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    producer_dep: pl.Scalar[pl.TASK_ID],
):
    """Sum vocabulary-shard contributions and scatter contiguous token rows."""
    group_base = my_rank // TP_SIZE * TP_SIZE
    tp_rank = my_rank % TP_SIZE

    with pl.spmd(COMM_WORKERS, name_hint="embedding_tp_scatter_publish", deps=[producer_dep]) as publish_tid:
        comm_core = pl.tile.get_block_idx()
        for owner_tp in pl.range(TP_SIZE):
            for block in pl.range(comm_core, SP_T * (D // COMM_D_TILE), COMM_WORKERS):
                owner_row = block // (D // COMM_D_TILE)
                col = block % (D // COMM_D_TILE) * COMM_D_TILE
                source_row = owner_tp * SP_T + owner_row
                dst_row = tp_rank * SP_T + owner_row
                peer = group_base + owner_tp
                pld.tensor.put(
                    dst=scatter_window,
                    peer=peer,
                    src=partial,
                    dst_offsets=[dst_row, col],
                    src_offsets=[source_row, col],
                    shape=[1, COMM_D_TILE],
                )

        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embedding_tp_scatter_wait", deps=[publish_tid]) as wait_tid:
        expected = pl.cast(COMM_WORKERS, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=scatter_signal,
                    offsets=[source_tp, 0],
                    expected=expected,
                    cmp=pld.WaitCmp.Ge,
                )

    with pl.spmd(COMM_WORKERS, name_hint="embedding_tp_scatter_reduce", deps=[wait_tid]) as reduce_tid:
        comm_core = pl.tile.get_block_idx()
        for block in pl.range(comm_core, SP_T * (D // COMM_D_TILE), COMM_WORKERS):
            local_row = block // (D // COMM_D_TILE)
            col = block % (D // COMM_D_TILE) * COMM_D_TILE
            acc_bf16 = pl.load(scatter_window, [local_row, col], [1, COMM_D_TILE])
            acc = pl.cast(acc_bf16, target_type=pl.FP32, mode="none")
            for source_tp in pl.range(1, TP_SIZE):
                source_row = source_tp * SP_T + local_row
                source_partial_bf16 = pl.load(scatter_window, [source_row, col], [1, COMM_D_TILE])
                source_partial = pl.cast(source_partial_bf16, target_type=pl.FP32, mode="none")
                acc = pl.add(acc, source_partial)
            reduced = pl.cast(acc, target_type=pl.BF16, mode="rint")
            pl.store(reduced, [local_row, col], local_out)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embedding_tp_scatter_complete", deps=[reduce_tid]):
        _completion_anchor = pl.read(local_out, [0, 0])
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(COMM_WORKERS + 1, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=scatter_signal,
                    offsets=[source_tp, 0],
                    expected=completion_expected,
                    cmp=pld.WaitCmp.Ge,
                )

        reset_value = pl.cast(-(COMM_WORKERS + 1), pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=scatter_signal,
                    peer=my_rank,
                    offsets=[source_tp, 0],
                    value=reset_value,
                    op=pld.NotifyOp.AtomicAdd,
                )
    return local_out


@pl.jit.inline(auto_scope=False)
def lookup_embedding_tp(
    input_ids_local: pl.Tensor,
    embed_weight_local: pl.Tensor,
    hidden_states_local: pl.Tensor,
    gather_window: pld.DistributedTensor[[1, GROUP_T_MAX], pl.INT64],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    scatter_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.BF16],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    producer_dep: pl.Scalar[pl.TASK_ID],
):
    """Look up one vocabulary shard and reduce-scatter the group token rows."""
    input_ids_group = pl.create_tensor([1, GROUP_T_MAX], dtype=pl.INT64)
    input_ids_group = gather_sp_ids(
        input_ids_local,
        input_ids_group,
        gather_window,
        gather_signal,
        my_rank,
        producer_dep,
    )

    local_vocab = pl.tensor.dim(embed_weight_local, 0)
    tp_rank = my_rank % TP_SIZE
    tp_rank_i64 = pl.cast(tp_rank, pl.INT64)
    vocab_start = tp_rank_i64 * local_vocab
    vocab_end = vocab_start + local_vocab
    work_items = GROUP_T_MAX * (D // HIDDEN_TILE)
    hidden_partial = pl.create_tensor([GROUP_T_MAX, D], dtype=pl.BF16)
    with pl.spmd(SPMD_BLOCKS, name_hint="lookup_embedding_tp_partial") as partial_tid:
        block = pl.tile.get_block_idx()
        for work_idx in pl.range(block, work_items, SPMD_BLOCKS):
            token_idx = work_idx // (D // HIDDEN_TILE)
            hidden_block = work_idx % (D // HIDDEN_TILE)
            hidden_offset = hidden_block * HIDDEN_TILE
            hidden_chunk = pl.full([1, HIDDEN_TILE], dtype=pl.BF16, value=0.0)
            token_id = pl.tensor.read(input_ids_group, [0, token_idx])
            if token_id >= vocab_start:
                if token_id < vocab_end:
                    local_token_id_i64 = token_id - vocab_start
                    local_token_id = pl.cast(local_token_id_i64, target_type=pl.INDEX)
                    local_chunk = embed_weight_local[
                        local_token_id : local_token_id + 1,
                        hidden_offset : hidden_offset + HIDDEN_TILE,
                    ]
                    hidden_chunk = local_chunk
            hidden_partial[token_idx : token_idx + 1, hidden_offset : hidden_offset + HIDDEN_TILE] = hidden_chunk

    return reduce_scatter_embedding_bf16(
        hidden_partial, hidden_states_local,
        scatter_window, scatter_signal,
        my_rank, partial_tid,
    )


@pl.jit
def lookup_embedding_test(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    hidden_states: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
) -> pl.Tensor[[T_DYN, D], pl.BF16]:
    input_ids.bind_dynamic(0, T_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    hidden_states.bind_dynamic(0, T_DYN)

    lookup_embedding(input_ids, embed_weight, hidden_states)
    return hidden_states


@pl.jit
def lookup_embedding_tp_test(
    input_ids_local: pl.Tensor[[SP_T], pl.INT64],
    embed_weight_local: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    hidden_states_local: pl.Out[pl.Tensor[[SP_T, D], pl.BF16]],
    gather_window: pld.DistributedTensor[[1, GROUP_T_MAX], pl.INT64],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    scatter_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.BF16],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    embed_weight_local.bind_dynamic(0, VOCAB_DYN)
    input_ids_ready = pl.system.task_dummy(deps=[])
    return lookup_embedding_tp(
        input_ids_local, embed_weight_local, hidden_states_local,
        gather_window, gather_signal, scatter_window, scatter_signal,
        my_rank, input_ids_ready,
    )


@pl.jit.host
def l3_lookup_embedding_tp(
    input_ids_local: pl.Tensor[[TP_SIZE, SP_T], pl.INT64],
    embed_weight_local: pl.Tensor[[TP_SIZE, VOCAB_DYN, D], pl.BF16],
    hidden_states_local: pl.Out[pl.Tensor[[TP_SIZE, SP_T, D], pl.BF16]],
):
    gather_window_buffer = pld.alloc_window_buffer([1, GROUP_T_MAX], dtype=pl.INT64)
    gather_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    scatter_window_buffer = pld.alloc_window_buffer([GROUP_T_MAX, D], dtype=pl.BF16)
    scatter_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(TP_SIZE):
        gather_window = pld.window(gather_window_buffer, [1, GROUP_T_MAX], dtype=pl.INT64)
        gather_signal = pld.window(gather_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        scatter_window = pld.window(scatter_window_buffer, [GROUP_T_MAX, D], dtype=pl.BF16)
        scatter_signal = pld.window(scatter_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        lookup_embedding_tp_test(
            input_ids_local[rank], embed_weight_local[rank], hidden_states_local[rank],
            gather_window, gather_signal, scatter_window, scatter_signal,
            rank,
            device=rank,
        )
    return hidden_states_local


def golden_lookup_embedding_test(tensors):
    tensors["hidden_states"][:] = tensors["embed_weight"].index_select(0, tensors["input_ids"].long())


def golden_lookup_embedding_tp(tensors):
    import torch

    input_ids_group = tensors["input_ids_local"].reshape(GROUP_T_MAX).long()
    embed_weight = torch.cat([tensors["embed_weight_local"][rank] for rank in range(TP_SIZE)], dim=0)
    hidden_states = embed_weight.index_select(0, input_ids_group)
    tensors["hidden_states_local"][:] = hidden_states.reshape(TP_SIZE, SP_T, D)


def build_tensor_specs(token_count, vocab_size):
    import torch
    from golden import TensorSpec

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    def init_input_ids():
        base_ids = [0, 1, 17, vocab_size - 1, 17, 2, vocab_size // 2, 1]
        sample_ids = torch.tensor([token_id % vocab_size for token_id in base_ids], dtype=torch.int64)
        repeats = (token_count + sample_ids.numel() - 1) // sample_ids.numel()
        return sample_ids.repeat(repeats)[:token_count].contiguous()

    def init_embed_weight():
        return torch.randn(vocab_size, D, dtype=torch.bfloat16)

    return [
        TensorSpec("input_ids", [token_count], torch.int64, init_value=init_input_ids),
        TensorSpec("embed_weight", [vocab_size, D], torch.bfloat16, init_value=init_embed_weight),
        TensorSpec("hidden_states", [token_count, D], torch.bfloat16, is_output=True),
    ]


def build_tp_tensor_specs(vocab_size):
    import torch
    from golden import TensorSpec

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if vocab_size % TP_SIZE != 0:
        raise ValueError(f"vocab_size {vocab_size} must be divisible by TP {TP_SIZE}")
    local_vocab = vocab_size // TP_SIZE

    def init_input_ids_local():
        samples = []
        for rank in range(TP_SIZE):
            shard_start = rank * local_vocab
            shard_end = shard_start + local_vocab
            samples.extend([shard_start, shard_end - 1])
        samples.extend([0, vocab_size - 1, vocab_size // 2, 1])
        sample_ids = torch.tensor(samples, dtype=torch.int64)
        repeats = (GROUP_T_MAX + sample_ids.numel() - 1) // sample_ids.numel()
        return sample_ids.repeat(repeats)[:GROUP_T_MAX].reshape(TP_SIZE, SP_T).contiguous()

    def init_embed_weight_local():
        embed_weight = torch.randn(vocab_size, D, dtype=torch.bfloat16)
        return embed_weight.reshape(TP_SIZE, local_vocab, D).contiguous()

    return [
        TensorSpec("input_ids_local", [TP_SIZE, SP_T], torch.int64, init_value=init_input_ids_local),
        TensorSpec(
            "embed_weight_local", [TP_SIZE, local_vocab, D], torch.bfloat16,
            init_value=init_embed_weight_local, resident="stacked",
        ),
        TensorSpec("hidden_states_local", [TP_SIZE, SP_T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    MODES = {"decode": DECODE_TOKENS, "prefill": PREFILL_TOKENS}
    TEST_VOCAB_SIZE = 256

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash embedding lookup validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp-vocab", "--tp", dest="tp_vocab", type=int, default=TP_SIZE, choices=list(TP_CHOICES))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(rank) for rank in range(TP_SIZE)))
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--vocab-size", type=int, default=TEST_VOCAB_SIZE)
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.tp_vocab != TP_SIZE:
        parser.error(f"import-time vocabulary TP {TP_SIZE} does not match --tp-vocab {args.tp_vocab}")

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < TP_SIZE:
        parser.error(f"need at least {TP_SIZE} devices, got {device_ids}")
    if args.vocab_size <= 0:
        parser.error(f"--vocab-size must be positive, got {args.vocab_size}")
    if TP_SIZE > 1 and args.vocab_size % TP_SIZE != 0:
        parser.error(f"--vocab-size {args.vocab_size} must be divisible by TP {TP_SIZE}")

    modes_to_run = list(MODES) if args.mode == "all" else [args.mode]
    for mode_name in modes_to_run:
        token_count = MODES[mode_name]
        if TP_SIZE == 1:
            fn = lookup_embedding_test
            specs = build_tensor_specs(token_count, args.vocab_size)
            golden_fn = golden_lookup_embedding_test
            compile_cfg = dict(dump_passes=args.dump_passes)
            runtime_cfg = dict(
                platform=args.platform,
                device_id=device_ids[0],
                enable_l2_swimlane=args.enable_l2_swimlane,
            )
        else:
            if token_count != GROUP_T_MAX:
                parser.error(f"distributed embedding requires {GROUP_T_MAX} group rows, got {token_count}")
            fn = l3_lookup_embedding_tp
            specs = build_tp_tensor_specs(args.vocab_size)
            golden_fn = golden_lookup_embedding_tp
            compile_cfg = dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(
                    device_ids=device_ids[:TP_SIZE],
                    num_sub_workers=0,
                ),
            )
            runtime_cfg = dict(
                platform=args.platform,
                enable_l2_swimlane=args.enable_l2_swimlane,
            )

        print(f"--- lookup_embedding TP={TP_SIZE} {mode_name}: T={token_count} ---")
        result = run_jit(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            save_data=args.save_data,
            compile_only=args.compile_only,
            compile_cfg=compile_cfg,
            runtime_cfg=runtime_cfg,
            rtol=0.0,
            atol=0.0,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
