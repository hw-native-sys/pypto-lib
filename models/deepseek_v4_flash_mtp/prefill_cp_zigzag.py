# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek V4 context-parallel zigzag ownership and projected-KV tail exchange."""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import FLASH as M

# model config
TAIL_ROWS = 128
HEAD_DIM = M.head_dim

# CP layout
CP_CHOICES = (2, 4, 8)
CP_DEFAULT = 2
EPOCHS = 1

# tiling
ROW_TILE = 8


def _parse_static_int(name: str, default: int) -> int:
    flag = f"--{name}"
    for i, token in enumerate(sys.argv):
        if token == flag and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if token.startswith(f"{flag}="):
            return int(token.split("=", 1)[1])
    return default


CP_SIZE = _parse_static_int("cp", CP_DEFAULT)
NUM_SEGMENTS = 2 * CP_SIZE

# Rank-major tail-window rows.
CP_TAIL_WINDOW_ROWS = NUM_SEGMENTS * TAIL_ROWS


def cp_owner_rank(segment: int, cp_size: int = CP_SIZE) -> int:
    """Return the physical zigzag owner rank for a logical segment."""
    if segment < cp_size:
        return segment
    return 2 * cp_size - 1 - segment


def cp_owner_part(segment: int, cp_size: int = CP_SIZE) -> int:
    """Return the owner-local part for a logical segment."""
    return 0 if segment < cp_size else 1

def cp_reverse_index(cp_size: int = CP_SIZE):
    """Return rank-major payload indices in logical segment order."""
    import torch

    idx = torch.empty(2 * cp_size, dtype=torch.int64)
    for s in range(2 * cp_size):
        if s < cp_size:
            idx[s] = 2 * s
        else:
            idx[s] = 2 * (2 * cp_size - 1 - s) + 1
    return idx

def cp_final_window_sources(segment_lens):
    """Return final raw-window logical-segment and tail-row sources."""
    import torch

    total = sum(segment_lens)
    seg_src = torch.full((TAIL_ROWS,), -1, dtype=torch.int64)
    row_src = torch.full((TAIL_ROWS,), -1, dtype=torch.int64)
    cum = 0
    boundaries = [0]
    for length in segment_lens:
        cum += length
        boundaries.append(cum)
    for j in range(TAIL_ROWS):
        abs_pos = total - TAIL_ROWS + j
        if abs_pos < 0:
            continue
        for s in range(len(segment_lens)):
            if boundaries[s] <= abs_pos < boundaries[s + 1]:
                seg_src[j] = s
                # Tail-relative row in the segment's final block.
                seg_local = abs_pos - boundaries[s]
                tail_start = max(0, segment_lens[s] - TAIL_ROWS)
                row_src[j] = seg_local - tail_start
                break
    return seg_src, row_src


def cp_owner_tables(cp_size: int = CP_SIZE):
    """Build logical-segment owner-rank and owner-part tables."""
    import torch

    owner_rank = torch.tensor([cp_owner_rank(s, cp_size) for s in range(2 * cp_size)], dtype=torch.int32)
    owner_part = torch.tensor([cp_owner_part(s, cp_size) for s in range(2 * cp_size)], dtype=torch.int32)
    return owner_rank, owner_part



@pl.jit.inline
def _prefill_cp_zigzag_kv_tail_exchange_wave(
    local_kv_tail: pl.Tensor[[2 * TAIL_ROWS, HEAD_DIM], pl.BF16],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    logical_tails_out: pl.Out[pl.Tensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]:
    """Exchange one local tail wave into logical segment order."""
    epoch_value = pl.cast(epoch + 1, pl.INT32)

    # Wait for prior window consumption.
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=consumed, offsets=[peer, 0],
                expected=epoch, cmp=pld.WaitCmp.Ge,
            )

    # Publish both local tails.
    for peer in pl.range(CP_SIZE):
        for part in pl.range(2):
            rank_major_pos = my_rank * 2 + part
            dst_row_base = rank_major_pos * TAIL_ROWS
            src_row_base = part * TAIL_ROWS
            pld.tensor.put(
                dst=kv_tail_window, peer=peer, src=local_kv_tail,
                dst_offsets=[dst_row_base, 0], src_offsets=[src_row_base, 0], shape=[TAIL_ROWS, HEAD_DIM],
                chunk_rows=ROW_TILE, chunk_cols=HEAD_DIM, pipeline=True,
            )
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=ready, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    # Gather logical-segment tails.
    for seg in pl.range(NUM_SEGMENTS):
        rm_pos = reverse_index[seg]
        owner = owner_rank_table[seg]
        if owner != my_rank:
            pld.system.wait(
                signal=ready, offsets=[owner, 0],
                expected=epoch_value, cmp=pld.WaitCmp.Ge,
            )
        rm_row_base = rm_pos * TAIL_ROWS
        out_row_base = seg * TAIL_ROWS
        for t0 in pl.range(0, TAIL_ROWS, ROW_TILE):
            win_tile = pl.load(
                kv_tail_window,
                [rm_row_base + t0, 0],
                [ROW_TILE, HEAD_DIM],
            )
            pl.store(win_tile, [out_row_base + t0, 0], logical_tails_out)

    # Acknowledge window consumption.
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=consumed, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    return logical_tails_out


@pl.jit.incore
def prefill_cp_zigzag_kv_tail_exchange_core(
    local_kv_tail: pl.Tensor[[EPOCHS * 2 * TAIL_ROWS, HEAD_DIM], pl.BF16],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    logical_tails_out: pl.Out[pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]],
    decode_raw_window_out: pl.Out[pl.Tensor[[TAIL_ROWS, HEAD_DIM], pl.BF16]],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]:
    """Exchange two local tails and assemble the final decode window."""
    for epoch in pl.range(EPOCHS):
        epoch_value = pl.cast(epoch + 1, pl.INT32)
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.wait(
                    signal=consumed,
                    offsets=[peer, 0],
                    expected=epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        for peer in pl.range(CP_SIZE):
            for part in pl.range(2):
                rank_major_pos = my_rank * 2 + part
                pld.tensor.put(
                    dst=kv_tail_window,
                    peer=peer,
                    src=local_kv_tail,
                    dst_offsets=[rank_major_pos * TAIL_ROWS, 0],
                    src_offsets=[(epoch * 2 + part) * TAIL_ROWS, 0],
                    shape=[TAIL_ROWS, HEAD_DIM],
                    chunk_rows=ROW_TILE,
                    chunk_cols=HEAD_DIM,
                    pipeline=True,
                )
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=ready,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        for seg in pl.range(NUM_SEGMENTS):
            rm_pos = reverse_index[seg]
            owner = owner_rank_table[seg]
            if owner != my_rank:
                pld.system.wait(
                    signal=ready,
                    offsets=[owner, 0],
                    expected=epoch_value,
                    cmp=pld.WaitCmp.Ge,
                )
            source_row = rm_pos * TAIL_ROWS
            destination_row = epoch * CP_TAIL_WINDOW_ROWS + seg * TAIL_ROWS
            for t0 in pl.range(0, TAIL_ROWS, ROW_TILE):
                win_tile = pl.load(
                    kv_tail_window,
                    [source_row + t0, 0],
                    [ROW_TILE, HEAD_DIM],
                )
                pl.store(
                    win_tile,
                    [destination_row + t0, 0],
                    logical_tails_out,
                )

        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=consumed,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    last_epoch_base = (EPOCHS - 1) * CP_TAIL_WINDOW_ROWS
    for j in pl.range(TAIL_ROWS):
        seg_src = final_win_seg_src[j]
        row_src = final_win_row_src[j]
        if seg_src >= 0:
            source_row = last_epoch_base + seg_src * TAIL_ROWS + row_src
            win_row = pl.load(logical_tails_out, [source_row, 0], [1, HEAD_DIM])
            pl.store(win_row, [j, 0], decode_raw_window_out)

    return logical_tails_out


@pl.jit
def prefill_cp_zigzag_kv_tail_exchange(
    local_kv_tail: pl.Tensor[[EPOCHS * 2 * TAIL_ROWS, HEAD_DIM], pl.BF16],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    logical_tails_out: pl.Out[pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]],
    decode_raw_window_out: pl.Out[pl.Tensor[[TAIL_ROWS, HEAD_DIM], pl.BF16]],
    my_rank: pl.Scalar[pl.INT32],
):
    return prefill_cp_zigzag_kv_tail_exchange_core(
        local_kv_tail, reverse_index, owner_rank_table,
        final_win_seg_src, final_win_row_src,
        kv_tail_window, ready, consumed,
        logical_tails_out, decode_raw_window_out, my_rank,
    )


@pl.jit.host
def prefill_cp_zigzag_kv_tail_exchange_test(
    local_kv_tail: pl.Tensor[[CP_SIZE, EPOCHS * 2 * TAIL_ROWS, HEAD_DIM], pl.BF16],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    logical_tails_out: pl.Out[pl.Tensor[[CP_SIZE, EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16]],
    decode_raw_window_out: pl.Out[pl.Tensor[[CP_SIZE, TAIL_ROWS, HEAD_DIM], pl.BF16]],
):
    """Launch one zigzag exchange child per rank."""
    kv_tail_window_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        kv_tail_window = pld.window(kv_tail_window_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
        ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        consumed = pld.window(consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        prefill_cp_zigzag_kv_tail_exchange(
            local_kv_tail[rank], reverse_index, owner_rank_table,
            final_win_seg_src, final_win_row_src,
            kv_tail_window, ready, consumed,
            logical_tails_out[rank], decode_raw_window_out[rank], rank,
            device=rank,
        )


def _segment_abs_starts(segment_lens):
    starts = []
    position = 0
    for length in segment_lens:
        starts.append(position)
        position += length
    return starts


def _encode_marker(abs_pos: int):
    import torch

    marker = torch.zeros(HEAD_DIM, dtype=torch.bfloat16)
    marker[1] = float(abs_pos // TAIL_ROWS)
    marker[2] = float(abs_pos % TAIL_ROWS)
    return marker


def golden_prefill_cp_zigzag_kv_tail_exchange(tensors):
    import torch

    segment_lens = [TAIL_ROWS] * NUM_SEGMENTS
    starts = _segment_abs_starts(segment_lens)
    logical = torch.zeros(CP_TAIL_WINDOW_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    for segment in range(NUM_SEGMENTS):
        for row in range(TAIL_ROWS):
            logical[segment * TAIL_ROWS + row] = _encode_marker(starts[segment] + row)
    tensors["logical_tails_out"][:] = logical.unsqueeze(0).expand(CP_SIZE, -1, -1)

    final_start = NUM_SEGMENTS * TAIL_ROWS - TAIL_ROWS
    decode_window = torch.stack([_encode_marker(final_start + row) for row in range(TAIL_ROWS)])
    tensors["decode_raw_window_out"][:] = decode_window.unsqueeze(0).expand(CP_SIZE, -1, -1)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    segment_lens = [TAIL_ROWS] * NUM_SEGMENTS
    starts = _segment_abs_starts(segment_lens)
    reverse_index = cp_reverse_index(CP_SIZE).to(torch.int32)
    owner_rank, _ = cp_owner_tables(CP_SIZE)
    final_segment, final_row = cp_final_window_sources(segment_lens)

    local_tail = torch.zeros(CP_SIZE, 2 * TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    for rank in range(CP_SIZE):
        for part in range(2):
            segment = rank if part == 0 else NUM_SEGMENTS - 1 - rank
            for row in range(TAIL_ROWS):
                local_tail[rank, part * TAIL_ROWS + row] = _encode_marker(starts[segment] + row)

    return [
        TensorSpec("local_kv_tail", [CP_SIZE, 2 * TAIL_ROWS, HEAD_DIM], torch.bfloat16, init_value=local_tail),
        TensorSpec("reverse_index", [NUM_SEGMENTS], torch.int32, init_value=reverse_index),
        TensorSpec("owner_rank_table", [NUM_SEGMENTS], torch.int32, init_value=owner_rank),
        TensorSpec("final_win_seg_src", [TAIL_ROWS], torch.int32, init_value=final_segment.to(torch.int32)),
        TensorSpec("final_win_row_src", [TAIL_ROWS], torch.int32, init_value=final_row.to(torch.int32)),
        TensorSpec("logical_tails_out", [CP_SIZE, CP_TAIL_WINDOW_ROWS, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("decode_raw_window_out", [CP_SIZE, TAIL_ROWS, HEAD_DIM], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(description="Standalone context-parallel zigzag exchange test.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(CP_SIZE)))
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=CP_CHOICES)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    if args.cp != CP_SIZE:
        raise SystemExit(f"--cp={args.cp} does not match import-time CP_SIZE={CP_SIZE}")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < args.cp:
        raise SystemExit(f"CP{args.cp} requires {args.cp} devices, got {device_ids}")

    result = run_jit(
        fn=prefill_cp_zigzag_kv_tail_exchange_test,
        specs=build_tensor_specs(),
        golden_fn=golden_prefill_cp_zigzag_kv_tail_exchange,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(device_ids=device_ids[:args.cp], num_sub_workers=0),
        ),
        runtime_cfg=dict(platform=args.platform),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
