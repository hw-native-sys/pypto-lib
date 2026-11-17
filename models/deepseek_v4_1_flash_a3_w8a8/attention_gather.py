# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Assemble the sparse attention key matrix and visibility mask.

The union key set is [raw sliding window | compressed rows].  The raw part is a
contiguous absolute-position range covering the history rows and the current
chunk; the compressed part covers compressed positions ``0 .. compressed_len``
(history plus rows emitted by the current chunk).  Visibility combines the
sliding window (128 positions) with the indexer Top-K selection: the compressed
slots are enabled only where the indexer emitted an index.
"""

import pypto.language as pl
import torch

from config import FLASH as M


# Dynamic shape variable.

# model config
HEAD_DIM = M.head_dim
WINDOW = M.sliding_window

# tiling
TILE = 16


@pl.jit.inline(auto_scope=False)
def assemble_key_set(
    raw_history: pl.Tensor,
    kv: pl.Tensor,
    comp_history: pl.Tensor,
    latent: pl.Tensor,
    history_positions: pl.Tensor,
    position_ids: pl.Tensor,
    topk_indices: pl.Tensor,
    keys: pl.Tensor,
    visible: pl.Tensor,
    comp_rows: pl.Scalar[pl.INT32],
    ratio: pl.constexpr,
    window: pl.constexpr,
    key_max: pl.constexpr,
    local_cols: pl.constexpr,
):
    """Copy the union rows and publish the window + Top-K visibility mask.

    ``comp_rows`` is the number of valid compressed history rows (the history
    buffer may be larger than the valid prefix, e.g. a chunk starting at
    position 0 in a full-size cache).
    """
    raw_rows = pl.tensor.dim(raw_history, 0)
    t_dim = pl.tensor.dim(kv, 0)

    # Clear the union buffer so alignment gaps are deterministic.
    for block in pl.spmd((key_max + TILE - 1) // TILE, name_hint="key_zero"):
        pl.store(
            pl.tile.full([TILE, HEAD_DIM], dtype=pl.BF16, value=0.0),
            [block * TILE, 0],
            keys,
        )
    # Raw history rows occupy the leading slots.
    for block in pl.spmd((raw_rows + TILE - 1) // TILE, name_hint="key_raw_history"):
        row = block * TILE
        rows = pl.min(TILE, raw_rows - row)
        value = pl.tile.load(raw_history, [row, 0], [TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        pl.store(pl.set_validshape(value, rows, HEAD_DIM), [row, 0], keys)
    # Current chunk rows follow the history.
    for block in pl.spmd((t_dim + TILE - 1) // TILE, name_hint="key_current"):
        row = block * TILE
        rows = pl.min(TILE, t_dim - row)
        value = pl.tile.load(kv, [row, 0], [TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        pl.store(pl.set_validshape(value, rows, HEAD_DIM), [raw_rows + row, 0], keys)
    # Compressed history rows.
    for block in pl.spmd((comp_rows + TILE - 1) // TILE, name_hint="key_comp_history"):
        row = block * TILE
        rows = pl.min(TILE, comp_rows - row)
        value = pl.tile.load(comp_history, [row, 0], [TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        pl.store(pl.set_validshape(value, rows, HEAD_DIM), [local_cols + row, 0], keys)
    # Rows emitted by the current chunk (token order; non-emitting rows are never visible).
    for block in pl.spmd((t_dim + TILE - 1) // TILE, name_hint="key_comp_current"):
        row = block * TILE
        rows = pl.min(TILE, t_dim - row)
        value = pl.tile.load(latent, [row, 0], [TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        pl.store(pl.set_validshape(value, rows, HEAD_DIM), [local_cols + comp_rows + row, 0], keys)

    start = pl.read(history_positions, [0])
    for token in pl.spmd(t_dim, name_hint="key_visible"):
        position = pl.read(position_ids, [token])
        # Contiguous raw range: history positions followed by current chunk positions.
        lane = pl.tile.arange(0, [1, local_cols], dtype=pl.INT32)
        absolute = pl.add(pl.cast(lane, target_type=pl.FP32), pl.cast(start, target_type=pl.FP32))
        current = pl.cast(position, target_type=pl.FP32)
        earliest = pl.sub(current, (window - 1) * 1.0)
        upper = pl.tile.cmps(absolute, current, cmp_type=3)
        lower = pl.tile.cmps(absolute, earliest, cmp_type=5)
        select_tmp = pl.tile.create([1, local_cols], dtype=pl.FP32)
        ones = pl.tile.full([1, local_cols], dtype=pl.FP32, value=1.0)
        zeros = pl.tile.full([1, local_cols], dtype=pl.FP32, value=0.0)
        window_mask = pl.tile.sel(lower, ones, zeros, select_tmp)
        window_mask = pl.tile.sel(upper, window_mask, zeros, select_tmp)
        # Positions before the start of the sequence are never visible even
        # though the fixed-size raw window has slots for them.
        nonneg = pl.tile.cmps(absolute, 0.0, cmp_type=5)
        window_mask = pl.tile.sel(nonneg, window_mask, zeros, select_tmp)
        mask_row = pl.tile.assemble(
            pl.tile.full([1, key_max], dtype=pl.INT32, value=0),
            pl.cast(window_mask, target_type=pl.INT32, mode="rint"),
            [0, 0],
        )
        # Top-K compressed indices map to history or current-latent slots.
        keep = pl.min(pl.tensor.dim(topk_indices, 1), pl.max(position + 1, 0) // ratio)
        first_current = pl.read(history_positions, [pl.tensor.dim(history_positions, 0) - 1]) + 1
        for lane_index in pl.range(keep):
            compressed = pl.read(topk_indices, [token, lane_index])
            if compressed >= 0:
                if compressed < comp_rows:
                    slot = local_cols + compressed
                else:
                    source_token = compressed * ratio + (ratio - 1) - first_current
                    slot = local_cols + comp_rows + source_token
                if slot < key_max:
                    pl.tile.write(mask_row, [0, slot], pl.cast(1, pl.INT32))
        pl.store(mask_row, [token, 0], visible)
    return keys, visible


@pl.jit.inline(auto_scope=False)
def assemble_swa_key_set(
    raw_history: pl.Tensor,
    kv: pl.Tensor,
    history_positions: pl.Tensor,
    position_ids: pl.Tensor,
    keys: pl.Tensor,
    visible: pl.Tensor,
    window: pl.constexpr,
    key_max: pl.constexpr,
    local_cols: pl.constexpr,
):
    """Copy the raw window plus the current chunk and publish the window-only mask.

    Uncompressed layers keep no compressed rows: the union set is exactly the
    contiguous raw range and visibility is the sliding window.
    """
    raw_rows = pl.tensor.dim(raw_history, 0)
    t_dim = pl.tensor.dim(kv, 0)
    for block in pl.spmd((key_max + TILE - 1) // TILE, name_hint="key_zero"):
        pl.store(
            pl.tile.full([TILE, HEAD_DIM], dtype=pl.BF16, value=0.0),
            [block * TILE, 0],
            keys,
        )
    for block in pl.spmd((raw_rows + TILE - 1) // TILE, name_hint="key_raw_history"):
        row = block * TILE
        rows = pl.min(TILE, raw_rows - row)
        value = pl.tile.load(raw_history, [row, 0], [TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        pl.store(pl.set_validshape(value, rows, HEAD_DIM), [row, 0], keys)
    for block in pl.spmd((t_dim + TILE - 1) // TILE, name_hint="key_current"):
        row = block * TILE
        rows = pl.min(TILE, t_dim - row)
        value = pl.tile.load(kv, [row, 0], [TILE, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
        pl.store(pl.set_validshape(value, rows, HEAD_DIM), [raw_rows + row, 0], keys)
    start = pl.read(history_positions, [0])
    for token in pl.spmd(t_dim, name_hint="key_visible"):
        position = pl.read(position_ids, [token])
        lane = pl.tile.arange(0, [1, local_cols], dtype=pl.INT32)
        absolute = pl.add(pl.cast(lane, target_type=pl.FP32), pl.cast(start, target_type=pl.FP32))
        current = pl.cast(position, target_type=pl.FP32)
        earliest = pl.sub(current, (window - 1) * 1.0)
        upper = pl.tile.cmps(absolute, current, cmp_type=3)
        lower = pl.tile.cmps(absolute, earliest, cmp_type=5)
        select_tmp = pl.tile.create([1, local_cols], dtype=pl.FP32)
        ones = pl.tile.full([1, local_cols], dtype=pl.FP32, value=1.0)
        zeros = pl.tile.full([1, local_cols], dtype=pl.FP32, value=0.0)
        window_mask = pl.tile.sel(lower, ones, zeros, select_tmp)
        window_mask = pl.tile.sel(upper, window_mask, zeros, select_tmp)
        nonneg = pl.tile.cmps(absolute, 0.0, cmp_type=5)
        window_mask = pl.tile.sel(nonneg, window_mask, zeros, select_tmp)
        mask_row = pl.tile.assemble(
            pl.tile.full([1, key_max], dtype=pl.INT32, value=0),
            pl.cast(window_mask, target_type=pl.INT32, mode="rint"),
            [0, 0],
        )
        pl.store(mask_row, [token, 0], visible)
    return keys, visible


@pl.jit
def prefill_key_set_test(
    raw_history: pl.Tensor[[128, 512], pl.BF16],
    kv: pl.Tensor[[16, 512], pl.BF16],
    comp_history: pl.Tensor[[624, 512], pl.BF16],
    latent: pl.Tensor[[16, 512], pl.BF16],
    history_positions: pl.Tensor[[128], pl.INT32],
    position_ids: pl.Tensor[[16], pl.INT32],
    topk_indices: pl.Tensor[[16, 512], pl.INT32],
    keys: pl.Out[pl.Tensor[[784, 512], pl.BF16]],
    visible: pl.Out[pl.Tensor[[16, 784], pl.INT32]],
):
    result_keys, result_visible = assemble_key_set(
        raw_history, kv, comp_history, latent, history_positions, position_ids, topk_indices,
        keys, visible, 624, ratio=2, window=WINDOW, key_max=784, local_cols=128 + 16,
    )
    return keys, visible


@pl.jit
def c1a_key_set_test(
    raw_history: pl.Tensor[[128, 512], pl.BF16],
    kv: pl.Tensor[[16, 512], pl.BF16],
    comp_history: pl.Tensor[[1248, 512], pl.BF16],
    latent: pl.Tensor[[16, 512], pl.BF16],
    history_positions: pl.Tensor[[128], pl.INT32],
    position_ids: pl.Tensor[[16], pl.INT32],
    topk_indices: pl.Tensor[[16, 512], pl.INT32],
    keys: pl.Out[pl.Tensor[[1408, 512], pl.BF16]],
    visible: pl.Out[pl.Tensor[[16, 1408], pl.INT32]],
):
    result_keys, result_visible = assemble_key_set(
        raw_history, kv, comp_history, latent, history_positions, position_ids, topk_indices,
        keys, visible, 1248, ratio=1, window=WINDOW, key_max=1408, local_cols=128 + 16,
    )
    return keys, visible


@pl.jit
def decode_key_set_test(
    raw_history: pl.Tensor[[128, 512], pl.BF16],
    kv: pl.Tensor[[1, 512], pl.BF16],
    comp_history: pl.Tensor[[625, 512], pl.BF16],
    latent: pl.Tensor[[1, 512], pl.BF16],
    history_positions: pl.Tensor[[128], pl.INT32],
    position_ids: pl.Tensor[[1], pl.INT32],
    topk_indices: pl.Tensor[[1, 512], pl.INT32],
    keys: pl.Out[pl.Tensor[[768, 512], pl.BF16]],
    visible: pl.Out[pl.Tensor[[1, 768], pl.INT32]],
):
    result_keys, result_visible = assemble_key_set(
        raw_history, kv, comp_history, latent, history_positions, position_ids, topk_indices,
        keys, visible, 625, ratio=2, window=WINDOW, key_max=768, local_cols=136,
    )
    return keys, visible


@pl.jit
def prefill_swa_key_set_test(
    raw_history: pl.Tensor[[128, 512], pl.BF16],
    kv: pl.Tensor[[16, 512], pl.BF16],
    history_positions: pl.Tensor[[128], pl.INT32],
    position_ids: pl.Tensor[[16], pl.INT32],
    keys: pl.Out[pl.Tensor[[144, 512], pl.BF16]],
    visible: pl.Out[pl.Tensor[[16, 144], pl.INT32]],
):
    assemble_swa_key_set(
        raw_history, kv, history_positions, position_ids, keys, visible,
        window=WINDOW, key_max=144, local_cols=144,
    )
    return keys, visible


@pl.jit
def decode_swa_key_set_test(
    raw_history: pl.Tensor[[128, 512], pl.BF16],
    kv: pl.Tensor[[1, 512], pl.BF16],
    history_positions: pl.Tensor[[128], pl.INT32],
    position_ids: pl.Tensor[[1], pl.INT32],
    keys: pl.Out[pl.Tensor[[144, 512], pl.BF16]],
    visible: pl.Out[pl.Tensor[[1, 144], pl.INT32]],
):
    assemble_swa_key_set(
        raw_history, kv, history_positions, position_ids, keys, visible,
        window=WINDOW, key_max=144, local_cols=144,
    )
    return keys, visible


def golden_swa_key_set(tensors):
    """Torch reference for the raw-window-only union and its visibility mask."""
    raw = tensors["raw_history"]
    kv = tensors["kv"]
    positions = tensors["position_ids"]
    raw_rows, t_dim = raw.shape[0], kv.shape[0]
    local = int(tensors["visible"].shape[1])
    keys = torch.zeros(tensors["keys"].shape, dtype=torch.bfloat16)
    keys[0:raw_rows] = raw
    keys[raw_rows : raw_rows + t_dim] = kv
    tensors["keys"][:] = keys
    rows = tensors["visible"].shape[0]
    visible = torch.zeros(rows, tensors["visible"].shape[1], dtype=torch.int32)
    start = int(tensors["history_positions"][0])
    for token in range(min(rows, t_dim)):
        position = int(positions[token])
        for slot in range(local):
            absolute = start + slot
            if absolute >= 0 and absolute <= position and position - absolute <= WINDOW - 1:
                visible[token, slot] = 1
    tensors["visible"][:] = visible


def _golden_key_set(tensors, ratio, comp_rows=None):
    """Torch reference for union-row copying and window + Top-K visibility."""
    raw = tensors["raw_history"]
    kv = tensors["kv"]
    comp = tensors["comp_history"]
    latent = tensors["latent"]
    positions = tensors["position_ids"]
    topk = tensors["topk_indices"]
    window = WINDOW
    raw_rows, t_dim = raw.shape[0], kv.shape[0]
    comp_rows = comp.shape[0] if comp_rows is None else comp_rows
    local = (raw_rows + t_dim + 7) // 8 * 8
    keys = torch.zeros(tensors["keys"].shape, dtype=torch.bfloat16)
    keys[0:raw_rows] = raw
    keys[raw_rows : raw_rows + t_dim] = kv
    keys[local : local + comp_rows] = comp
    keys[local + comp_rows : local + comp_rows + t_dim] = latent
    tensors["keys"][:] = keys
    rows = tensors["visible"].shape[0]
    visible = torch.zeros(rows, tensors["visible"].shape[1], dtype=torch.int32)
    start = int(tensors["history_positions"][0])
    first_current = int(tensors["history_positions"][-1]) + 1
    for token in range(min(rows, t_dim)):
        position = int(positions[token])
        for slot in range(local):
            absolute = start + slot
            if absolute >= 0 and absolute <= position and position - absolute <= window - 1:
                visible[token, slot] = 1
        keep = min(topk.shape[1], max(position + 1, 0) // ratio)
        for lane in range(keep):
            compressed = int(topk[token, lane])
            if compressed < 0:
                continue
            if compressed < comp_rows:
                slot = local + compressed
            else:
                slot = local + comp_rows + compressed * ratio + (ratio - 1) - first_current
            if 0 <= slot < visible.shape[1]:
                visible[token, slot] = 1
    tensors["visible"][:] = visible


def golden_assemble_key_set(tensors):
    _golden_key_set(tensors, 2)


def golden_c1a_key_set(tensors):
    _golden_key_set(tensors, 1)


def build_swa_specs(tokens: int, start: int, seed: int = 0):
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(128, HEAD_DIM, generator=generator).to(torch.bfloat16)
    kv = torch.randn(tokens, HEAD_DIM, generator=generator).to(torch.bfloat16)
    positions = torch.arange(start, start + tokens, dtype=torch.int32)
    local = (128 + tokens + 15) // 16 * 16
    return [
        TensorSpec("raw_history", [128, HEAD_DIM], torch.bfloat16, init_value=raw),
        TensorSpec("kv", [tokens, HEAD_DIM], torch.bfloat16, init_value=kv),
        TensorSpec("history_positions", [128], torch.int32, init_value=torch.arange(start - 128, start, dtype=torch.int32)),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=positions),
        TensorSpec("keys", [local, HEAD_DIM], torch.bfloat16),
        TensorSpec("visible", [tokens, local], torch.int32),
    ]


def build_tensor_specs(tokens: int = 16, start: int = 1248, comp_rows: int = 624, ratio: int = 2, seed: int = 0):
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(128, HEAD_DIM, generator=generator).to(torch.bfloat16)
    kv = torch.randn(tokens, HEAD_DIM, generator=generator).to(torch.bfloat16)
    comp = torch.randn(comp_rows, HEAD_DIM, generator=generator).to(torch.bfloat16)
    latent = torch.randn(tokens, HEAD_DIM, generator=generator).to(torch.bfloat16)
    positions = torch.arange(start, start + tokens, dtype=torch.int32)
    topk = torch.full((tokens, 512), -1, dtype=torch.int32)
    for token in range(tokens):
        available = (int(positions[token]) + 1) // ratio
        visible = min(512, available)
        chosen = torch.randperm(available, generator=generator)[:visible]
        topk[token, :visible] = chosen.to(torch.int32)
    local_cols = (128 + tokens + 7) // 8 * 8
    key_max = (local_cols + comp_rows + tokens + 7) // 8 * 8
    return [
        TensorSpec("raw_history", [128, HEAD_DIM], torch.bfloat16, init_value=raw),
        TensorSpec("kv", [tokens, HEAD_DIM], torch.bfloat16, init_value=kv),
        TensorSpec("comp_history", [comp_rows, HEAD_DIM], torch.bfloat16, init_value=comp),
        TensorSpec("latent", [tokens, HEAD_DIM], torch.bfloat16, init_value=latent),
        TensorSpec("history_positions", [128], torch.int32, init_value=torch.arange(start - 128, start, dtype=torch.int32)),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=positions),
        TensorSpec("topk_indices", [tokens, 512], torch.int32, init_value=topk),
        TensorSpec("keys", [key_max, HEAD_DIM], torch.bfloat16),
        TensorSpec("visible", [tokens, key_max], torch.int32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--start", type=int, default=1248)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    compare = {
        "keys": ratio_allclose(atol=0.0, rtol=0.0),
        "visible": ratio_allclose(atol=0.0, rtol=0.0),
    }
    cases = [
        ("c2a_prefill", prefill_key_set_test, golden_assemble_key_set, build_tensor_specs(16, 1248, 624, 2)),
        ("c1a_prefill", c1a_key_set_test, golden_c1a_key_set, build_tensor_specs(16, 1248, 1248, 1)),
        ("c2a_decode", decode_key_set_test, golden_assemble_key_set, build_tensor_specs(1, 1251, 625, 2)),
        ("swa_prefill", prefill_swa_key_set_test, golden_swa_key_set, build_swa_specs(16, 1248)),
        ("swa_decode", decode_swa_key_set_test, golden_swa_key_set, build_swa_specs(1, 1251)),
    ]
    failures = []
    for name, fn, golden_fn, specs in cases:
        print(f"[GATHER] {name}", flush=True)
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=0.0,
            atol=0.0,
            compare_fn=compare,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
