# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token embedding lookup and tensor-parallel exchange."""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig


if __package__:
    from .config import FLASH as M
else:
    from config import FLASH as M
if __package__:
    from .decode_shapes import DECODE_TOKENS
else:
    from decode_shapes import DECODE_TOKENS


# Dynamic shape variables.
T_DYN = pl.dynamic('LOWLAT_TARGET_LOOKUP_EMBEDDING_T_DYN')
VOCAB_DYN = pl.dynamic('LOWLAT_TARGET_LOOKUP_EMBEDDING_VOCAB_DYN')

# model config
D = M.hidden_size

# tiling
HIDDEN_TILE = 512
SPMD_BLOCKS = 48


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


# ci: devices=2


if __package__:
    from .config import FLASH as M
else:
    from config import FLASH as M
if __package__:
    from .decode_shapes import DECODE_TOKENS, TP
else:
    from decode_shapes import DECODE_TOKENS, TP


# model config
T = DECODE_TOKENS
VOCAB = M.vocab_size
N_RANKS = TP
VOCAB_PER_TP = VOCAB // N_RANKS

# tiling


@pl.jit.inline
def embed_shard_gather(
    input_ids: pl.Tensor[[T], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    owned_rows: pl.Tensor[[T, D], pl.BF16],
    vocab_base: pl.Scalar[pl.INT32],
):
    """Copy the rows this rank owns; rows it does not own are left untouched."""
    for block in pl.spmd(T * (D // HIDDEN_TILE), name_hint="embed_shard_gather", allow_early_resolve=True):
        token = block // (D // HIDDEN_TILE)
        hidden_offset = (block % (D // HIDDEN_TILE)) * HIDDEN_TILE
        token_id = pl.cast(pl.tensor.read(input_ids, [token]), pl.INT32)
        local_id = token_id - vocab_base
        if local_id >= 0:
            if local_id < VOCAB_PER_TP:
                shard_row = pl.cast(local_id, pl.INDEX)
                shard_tile = embed_weight[shard_row : shard_row + 1, hidden_offset : hidden_offset + HIDDEN_TILE]
                owned_rows[token : token + 1, hidden_offset : hidden_offset + HIDDEN_TILE] = shard_tile


@pl.jit.inline
def all_gather_embedding(input_ids: pl.Tensor[[T], pl.INT64], owned_rows: pl.Tensor[[T, D], pl.BF16], hidden_states: pl.Tensor[[T, D], pl.BF16], gather_window: pld.DistributedTensor[[T, D], pl.BF16], gather_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], vocab_base: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32], gather_epoch: pl.Scalar[pl.INT32], defer_clear: pl.constexpr):
    """Publish, wait, read, and clear counters before the invocation completes.

    Reuse this single window only after all ranks finish the previous call.
    """
    with pl.spmd(T, name_hint='embed_publish', allow_early_resolve=True) as publish_tid:
        token = pl.tile.get_block_idx()
        token_id = pl.cast(pl.tensor.read(input_ids, [token]), pl.INT32)
        local_id = token_id - vocab_base
        if local_id >= 0:
            if local_id < VOCAB_PER_TP:
                for peer in pl.range(N_RANKS):
                    pld.tensor.put(dst=gather_window, peer=peer, src=owned_rows, dst_offsets=[token, 0], src_offsets=[token, 0], shape=[1, D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint='embed_barrier', allow_early_resolve=True, deps=[publish_tid]) as barrier_tid:
        for peer in pl.range(N_RANKS):
            pld.system.notify(target=gather_signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        for src in pl.range(N_RANKS):
            pld.system.wait(signal=gather_signal, offsets=[src, 0], expected=gather_epoch, cmp=pld.WaitCmp.Ge)
    with pl.spmd(T * (D // HIDDEN_TILE), name_hint='embed_readback', allow_early_resolve=True, deps=[barrier_tid]) as _readback_tid:
        block = pl.tile.get_block_idx()
        token = block // (D // HIDDEN_TILE)
        hidden_offset = block % (D // HIDDEN_TILE) * HIDDEN_TILE
        gathered = pl.load(gather_window, [token, hidden_offset], [1, HIDDEN_TILE])
        pl.store(gathered, [token, hidden_offset], hidden_states)
    if not defer_clear:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint='embed_signal_clear', deps=[_readback_tid]):
            zero = pl.cast(0, pl.INT32)
            for src in pl.range(N_RANKS):
                pl.write(gather_signal, [src, 0], zero)


@pl.jit.inline
def clear_embedding_signals(
    completion_anchor: pl.Tensor[[T, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's embedding counters after its forward completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embed_signal_clear") as s6_embed_cleanup_tid:
        _completion_anchor = pl.read(completion_anchor, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(gather_signal, [src, 0], zero)
    return (completion_anchor, s6_embed_cleanup_tid)

@pl.jit.inline
def tp_lookup_embedding(input_ids: pl.Tensor[[T], pl.INT64], embed_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16], hidden_states: pl.Tensor[[T, D], pl.BF16], gather_window: pld.DistributedTensor[[T, D], pl.BF16], gather_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], my_rank: pl.Scalar[pl.INT32], gather_epoch: pl.Scalar[pl.INT32], defer_clear: pl.constexpr) -> pl.Tensor[[T, D], pl.BF16]:
    vocab_base = pl.cast(my_rank, pl.INT32) * VOCAB_PER_TP
    if N_RANKS == 1:
        embed_shard_gather(input_ids, embed_weight, hidden_states, vocab_base)
    else:
        owned_rows = pl.create_tensor([T, D], dtype=pl.BF16)
        embed_shard_gather(input_ids, embed_weight, owned_rows, vocab_base)
        all_gather_embedding(input_ids, owned_rows, hidden_states, gather_window, gather_signal, vocab_base, my_rank, gather_epoch, defer_clear=defer_clear)
    return hidden_states


@pl.jit
def tp_embedding_test(input_ids: pl.Tensor[[T], pl.INT64], embed_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16], hidden_states: pl.Out[pl.Tensor[[T, D], pl.BF16]], gather_window: pld.DistributedTensor[[T, D], pl.BF16], gather_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32], my_rank: pl.Scalar[pl.INT32], gather_epoch: pl.Scalar[pl.INT32]) -> pl.Tensor[[T, D], pl.BF16]:
    tp_lookup_embedding(input_ids, embed_weight, hidden_states, gather_window, gather_signal, my_rank, gather_epoch, defer_clear=False)
    return hidden_states


@pl.jit.host
def l3_tp_embedding(
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    embed_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    hidden_states: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
):
    gather_window_buf = pld.alloc_window_buffer([T, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [T, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        tp_embedding_test(
            input_ids[r], embed_weight[r], hidden_states[r],
            gather_window, gather_signal,
            r, pl.const(1, pl.INT32),
            device=r,
        )


def golden_lookup_embedding_test(tensors):
    tensors["hidden_states"][:] = tensors["embed_weight"].index_select(0, tensors["input_ids"].long())


def build_tensor_specs_lookup(token_count, vocab_size):
    import torch
    from golden import TensorSpec

    def init_input_ids():
        sample_ids = torch.tensor([0, 1, 17, vocab_size - 1, 17, 2, vocab_size // 2, 1], dtype=torch.int64)
        repeats = (token_count + sample_ids.numel() - 1) // sample_ids.numel()
        return sample_ids.repeat(repeats)[:token_count].contiguous()

    def init_embed_weight():
        return torch.randn(vocab_size, D, dtype=torch.bfloat16)

    return [
        TensorSpec("input_ids", [token_count], torch.int64, init_value=init_input_ids),
        TensorSpec("embed_weight", [vocab_size, D], torch.bfloat16, init_value=init_embed_weight),
        TensorSpec("hidden_states", [token_count, D], torch.bfloat16),
    ]


def golden_tp_embedding(tensors):
    """Concatenating the rank shards in index order reproduces the global table."""
    import torch

    full_table = torch.cat([tensors["embed_weight"][r] for r in range(N_RANKS)], dim=0)
    rows = full_table.index_select(0, tensors["input_ids"][0].long())
    tensors["hidden_states"][:] = rows.unsqueeze(0).expand(N_RANKS, -1, -1)


def build_tensor_specs_tp():
    import torch
    from golden import TensorSpec

    # Distinct owners across the fixture: ids spread over every shard plus both ends.
    def init_input_ids():
        stride = VOCAB // T
        ids = torch.arange(T, dtype=torch.int64) * stride
        ids[0] = 0
        ids[-1] = VOCAB - 1
        return ids.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_embed_weight():
        table = torch.randn(VOCAB, D, dtype=torch.bfloat16)
        return torch.stack([table[r * VOCAB_PER_TP : (r + 1) * VOCAB_PER_TP] for r in range(N_RANKS)])

    specs = [
        TensorSpec("input_ids", [N_RANKS, T], torch.int64, init_value=init_input_ids),
        TensorSpec("embed_weight", [N_RANKS, VOCAB_PER_TP, D], torch.bfloat16, init_value=init_embed_weight),
        TensorSpec("hidden_states", [N_RANKS, T, D], torch.bfloat16),
    ]
    for spec in specs:
        if spec.name == "embed_weight":
            spec.resident = "stacked"
    return specs


def validate_lookup():
    import argparse
    from golden import run

    MODES = {"decode": DECODE_TOKENS}
    TEST_VOCAB_SIZE = 256

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash embedding lookup validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode"], default="decode")
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES) if args.mode == "all" else [args.mode]
    for mode_name in modes_to_run:
        token_count = MODES[mode_name]
        print(f"--- lookup_embedding_test {mode_name}: T={token_count} ---")
        result = run(
            compile_only=args.compile_only,
            fn=lookup_embedding_test,
            specs=build_tensor_specs_lookup(token_count, TEST_VOCAB_SIZE),
            golden_fn=golden_lookup_embedding_test,
            config=dict(
                backend_type=args.platform.removesuffix("sim"),
                execution_mode=args.sim or args.platform.endswith("sim"),
                device_id=args.device,
            ),
            rtol=0.0,
            atol=0.0,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)



def validate_tp():
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash TP vocab-sharded embedding.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP, choices=[1, 2, 4, 8],
                        help="tensor-parallel degree / rank count; config freezes it at import")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids (need {N_RANKS})")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    result = run(
        fn=l3_tp_embedding,
        specs=build_tensor_specs_tp(),
        golden_fn=golden_tp_embedding,
        compile_only=args.compile_only,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            backend_type=args.platform.removesuffix("sim"),
            execution_mode=args.sim or args.platform.endswith("sim"),
        ),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    from models.deepseek_v4_flash_lowlat.utils import run_operator_validations

    run_operator_validations(
        {
            'lookup': validate_lookup,
            'tp': validate_tp,
        },
        single_device=('lookup',),
        automatic=('tp', 'lookup'),
        standalone_options={"lookup": {"--mode"}, "tp": {"--tp", "--dump-passes"}},
    )
