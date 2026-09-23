# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sparse NoPE MLA attention, decode path, over the absorbed latent query.

Decode attends in **latent space**: the query has already had the key half of
``kv_b_proj`` folded in, so it is ``[LOCAL_H, KV_LORA]`` and multiplies the cached
512-wide rows directly, with no per-token key expansion. The value half comes back
out through the absorbed ``o_proj``, so this kernel's output is ``[T, LOCAL_H,
KV_LORA]``, not ``[T, LOCAL_H, V_DIM]``.

Each of the ``1 + MTP_SPEC_TOKENS`` rows per request carries its own index list, so
the gather is ragged across rows even within one request. FULL_DECODE_ONLY graph
capture needs the shapes static, so the index list stays padded to
``TOPK_INDEX_WIDTH`` and the ``-1`` sentinel does the masking.

**``topk_indices`` is front packed, and this kernel requires it.** Every valid
position precedes every ``-1``, so a row's padding is one suffix and a ``-1`` never
sits between two live entries. That is the indexer's ABI — see
:func:`models.glm5_3_flash.decode_indexer.indexer_expand` — not an observation about
the current reference. The kernel reads one lane to decide that a 128-wide block, or
a whole row, selects nothing; against an interleaved list those tests would drop the
live entries behind the first ``-1``. Honouring ``-1`` per lane inside a block is
unconditional and stays correct either way; only the block-level and row-level
short-circuits depend on the packing.

**The indexer emits logical per-request positions, not physical cache rows**, so the
block table is part of this kernel's ABI rather than something the host resolves.
Resolving it host-side would push a per-layer index translation into the captured
decode graph, which is exactly what FULL_DECODE_ONLY cannot absorb.

Donor: ``models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py`` (827 lines,
a2a3-tuned — its comments call out the AIC-count dispatch lanes and the L0C wall).
Its KV row width is 512, the same as ``kv_lora_rank``, so the 128x512 L1 tile, the
per-row block-table gather, the additive ``NEG_INF`` validity bias, the flash
partials and the online-softmax merge all transfer. Delete from it: the
sliding-window half of the gather, the ratio-4 compressed-slot rewrite (GLM's
indices are raw positions, already causality- and padding-filtered by the indexer),
the attention sink, the whole inverse-RoPE block, and the fused ``o_proj`` tail —
that last one belongs to :mod:`models.glm5_3_flash.mla_epilog`.

Watch the plan-stage compile: the donor unrolls ``T x SPARSE_BLOCKS`` work items,
which is 40 iterations there and 2176 here.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, B_DYN, KV_LORA, LOCAL_H, QK_DIM
from models.glm5_3_flash.config import BLOCK_TABLE_DYN, TABLE_DYN, TOPK_INDEX_WIDTH, T_DYN
from models.glm5_3_flash.metadata import paged_slots

SOFTMAX_SCALE = QK_DIM**-0.5
NEG_INF = -1.0e20       # finite floor; a true -inf would make an all-masked row NaN
ATTN_K_TILE = 128       # selected rows per sparse block
NUM_QK_CORES = 24       # qk_pv dispatch lanes = a2a3 AIC count
H_TILE = min(LOCAL_H, 16)
if H_TILE != LOCAL_H:
    raise ValueError(f"sparse MLA attention needs TP >= 4 (LOCAL_H={LOCAL_H} > 16 heads per tile)")
H_PAD = ((H_TILE + 15) // 16) * 16
SPARSE_BLOCKS = (TOPK_INDEX_WIDTH + ATTN_K_TILE - 1) // ATTN_K_TILE


def golden_decode_sparse_attn(
    absorbed_query: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """Attend in latent space over the indexer's selected rows.

    Selection, resolution through ``block_table`` and ``-1`` masking follow the prefill
    path exactly. What differs is that the query already carries the key half of
    ``kv_b_proj`` (see
    :func:`models.glm5_3_flash.mla_prolog.golden_absorb_query`), so the cached 512-wide
    rows serve as both keys and values and nothing is expanded per token. The value half
    comes back out through the absorbed ``o_proj``.

    The softmax scale still belongs to the **un-absorbed** head dim, ``QK_DIM ** -0.5``:
    absorption is an exact rewrite of the score, not a change of its space, so the scale
    must not be derived from the query's latent width.

    Each request contributes ``1 + MTP_SPEC_TOKENS`` rows and every row carries its own
    index list, so the gather is ragged even within one request and ``request_ids`` is
    per row rather than per request.

    Args:
        absorbed_query: ``[T, H, KV_LORA]`` query folded into latent space.
        latent_cache: ``[TABLE * BLOCK_SIZE, KV_LORA]`` paged 512-latent pool.
        topk_indices: ``[T, W]`` int32 logical positions, ``-1`` padded.
        block_table: ``[B, BLOCK_TABLE]`` int32 logical-to-physical page map.
        request_ids: ``[T]`` int32 owning request of each decode row.

    Returns:
        ``[T, H, KV_LORA]`` latent-space context in ``absorbed_query``'s dtype.
    """
    valid = topk_indices >= 0
    positions = topk_indices.clamp(min=0).to(torch.int64)
    rows = paged_slots(positions, request_ids.to(torch.int64).unsqueeze(-1), block_table)
    latent = latent_cache[rows].float()
    scores = torch.einsum("thk,twk->thw", absorbed_query.float(), latent) * (QK_DIM**-0.5)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
    return torch.einsum("thw,twk->thk", weights, latent).to(absorbed_query.dtype)


@pl.jit.inline
def decode_sparse_attn(
    absorbed_query: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
):
    """Flash-decode over the indexer's selected latent rows.

    One work item per (token, sparse block) across ``NUM_QK_CORES`` lanes, each item
    gathering ``ATTN_K_TILE`` selected rows into L1 and emitting that block's
    ``(max, sum, weighted rows)`` partial; a second scope merges the
    ``SPARSE_BLOCKS`` partials per token with the usual alpha/beta rescale.

    The absorbed query makes the cached row serve as both key and value, so the same
    L1 tile feeds the QK matmul and the PV matmul and the gather happens once.

    Selected positions are **logical and per-request**, so each row costs two scalar
    reads — the index, then its page in ``block_table``. A ``-1`` lane gathers row 0
    to keep the tile finite and takes a ``NEG_INF`` bias, which is also what the
    ``SPARSE_BLOCKS * ATTN_K_TILE - TOPK_INDEX_WIDTH`` padding lanes of the last
    block get. Causality and request padding are already folded into the index list,
    so no causal mask is applied here.

    A block whose lane 0 is ``-1`` is skipped outright and emits ``(NEG_INF, 0, 0)``,
    which the merge's ``beta`` discards. This reads the indexer's front-packing ABI
    (module docstring): lane 0 being ``-1`` means the whole block is padding, because
    padding is a suffix. The skip is what keeps a short request cheap — the 2,176
    lanes cover a 1 M context, so without it a 40-selection row would gather 2,176
    rows to use 40 of them. ``NEG_INF`` is a finite floor precisely so that
    ``NEG_INF - NEG_INF`` is 0 rather than NaN when every block of a row is skipped.

    **Precondition**: every row has at least one valid lane, which
    ``index_kpool_always_select_tail`` guarantees. An all-masked row would leave the
    softmax denominator at ``ATTN_K_TILE`` rather than zero, so it is not defined
    here; the reference returns a zero row for that case as a host-side convenience.

    Head rows are padded to ``H_PAD`` = 16. The QK/PV matmuls are only split into L0
    tiles once M reaches the 16-row cube minimum — below it the 128x512 KV operand stays
    whole and overflows the 64 KB right buffer — and 16 FP32 rows also give the 32-byte
    column alignment the merge's reductions need. The padded query rows are zero-filled
    and never written out, which leaves the cube three quarters idle at TP16. Recovering
    that quarter is the open tuning question for this scope, and batching the heads of
    several decode rows behind one gather is not the answer, since each row carries its
    own index list.

    Returns the attention output so the caller can chain it.
    """
    t_dim = pl.tensor.dim(absorbed_query, 0)
    q_flat = pl.reshape(absorbed_query, [t_dim * LOCAL_H, KV_LORA])
    out_flat = pl.reshape(output, [t_dim * LOCAL_H, KV_LORA])

    blk_mi = pl.create_tensor([t_dim * SPARSE_BLOCKS * H_PAD, 1], dtype=pl.FP32)
    blk_li = pl.create_tensor([t_dim * SPARSE_BLOCKS * H_PAD, 1], dtype=pl.FP32)
    blk_oi = pl.create_tensor([t_dim * SPARSE_BLOCKS * H_PAD, KV_LORA], dtype=pl.FP32)
    with pl.spmd(NUM_QK_CORES, name_hint="mla_decode_qk_pv") as qk_tid:
        lane = pl.tile.get_block_idx()
        items = t_dim * SPARSE_BLOCKS
        for it in pl.range((items - lane + NUM_QK_CORES - 1) // NUM_QK_CORES):
            item = lane + it * NUM_QK_CORES
            token = item // SPARSE_BLOCKS
            sb = item - token * SPARSE_BLOCKS
            w0 = sb * ATTN_K_TILE
            request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
            blk_row = (token * SPARSE_BLOCKS + sb) * H_PAD
            # Lane 0 is -1 only when the whole block is padding.
            if pl.read(topk_indices, [token, w0]) < 0:
                blk_mi[blk_row : blk_row + H_PAD, 0:1] = pl.reshape(
                    pl.full([1, H_PAD], dtype=pl.FP32, value=NEG_INF), [H_PAD, 1]
                )
                blk_li[blk_row : blk_row + H_PAD, 0:1] = pl.reshape(
                    pl.full([1, H_PAD], dtype=pl.FP32, value=0.0), [H_PAD, 1]
                )
                blk_oi[blk_row : blk_row + H_PAD, 0:KV_LORA] = pl.full(
                    [H_PAD, KV_LORA], dtype=pl.FP32, value=0.0
                )
            else:
                kv_tile = pl.create_l1([ATTN_K_TILE, KV_LORA], pl.BF16)
                bias_row = pl.full([1, ATTN_K_TILE], dtype=pl.FP32, value=NEG_INF)
                for r in pl.range(ATTN_K_TILE):
                    if w0 + r < TOPK_INDEX_WIDTH:
                        position = pl.read(topk_indices, [token, w0 + r])
                        if position >= 0:
                            logical = pl.cast(position, pl.INDEX)
                            page = pl.cast(
                                pl.read(block_table, [request, logical // BLOCK_SIZE]), pl.INDEX
                            )
                            source = page * BLOCK_SIZE + logical % BLOCK_SIZE
                            kv_tile = pl.gather_row(
                                kv_tile, latent_cache, [r, 0], [source, 0], [1, KV_LORA]
                            )
                            pl.write(bias_row, [0, r], 0.0)
                        else:
                            kv_tile = pl.gather_row(
                                kv_tile, latent_cache, [r, 0], [0, 0], [1, KV_LORA]
                            )
                    else:
                        kv_tile = pl.gather_row(kv_tile, latent_cache, [r, 0], [0, 0], [1, KV_LORA])

                q_tile = pl.fillpad(
                    pl.slice(
                        q_flat,
                        [H_PAD, KV_LORA],
                        [token * LOCAL_H, 0],
                        valid_shape=[H_TILE, KV_LORA],
                    ),
                    pad_value=pl.PadValue.zero,
                )
                raw = pl.matmul(q_tile, kv_tile, b_trans=True, out_dtype=pl.FP32)
                scores = pl.col_expand_add(pl.mul(raw, SOFTMAX_SCALE), bias_row)
                mi = pl.row_max(scores)
                weights = pl.exp(pl.row_expand_sub(scores, mi))
                li = pl.row_sum(weights)
                oi = pl.matmul(
                    pl.cast(weights, target_type=pl.BF16, mode="rint"), kv_tile, out_dtype=pl.FP32
                )
                blk_mi[blk_row : blk_row + H_PAD, 0:1] = mi
                blk_li[blk_row : blk_row + H_PAD, 0:1] = li
                blk_oi[blk_row : blk_row + H_PAD, 0:KV_LORA] = oi

    with pl.spmd(t_dim, name_hint="mla_decode_merge", deps=[qk_tid]):
        token = pl.tile.get_block_idx()
        base = (token * SPARSE_BLOCKS) * H_PAD
        merged_mi = blk_mi[base : base + H_PAD, 0:1]
        merged_li = blk_li[base : base + H_PAD, 0:1]
        merged_oi = blk_oi[base : base + H_PAD, 0:KV_LORA]
        for sb in pl.pipeline(1, SPARSE_BLOCKS, stage=2):
            row = base + sb * H_PAD
            cur_mi = blk_mi[row : row + H_PAD, 0:1]
            cur_li = blk_li[row : row + H_PAD, 0:1]
            cur_oi = blk_oi[row : row + H_PAD, 0:KV_LORA]
            new_mi = pl.maximum(merged_mi, cur_mi)
            alpha = pl.exp(pl.sub(merged_mi, new_mi))
            beta = pl.exp(pl.sub(cur_mi, new_mi))
            merged_li = pl.add(pl.mul(alpha, merged_li), pl.mul(beta, cur_li))
            merged_oi = pl.add(
                pl.row_expand_mul(merged_oi, alpha), pl.row_expand_mul(cur_oi, beta)
            )
            merged_mi = new_mi
        # Lane 0 of the row is -1 only when the row selects nothing at all; every block
        # is then neutral and ``merged_li`` is zero, which a division would turn to NaN.
        if pl.read(topk_indices, [token, 0]) >= 0:
            normalized = pl.row_expand_div(merged_oi, merged_li)
            out_bf16 = pl.cast(normalized, target_type=pl.BF16, mode="rint")
            out_flat[token * LOCAL_H : token * LOCAL_H + H_TILE, 0:KV_LORA] = out_bf16[
                0:H_TILE, 0:KV_LORA
            ]
        else:
            out_flat[token * LOCAL_H : token * LOCAL_H + H_TILE, 0:KV_LORA] = pl.full(
                [H_TILE, KV_LORA], dtype=pl.BF16, value=0.0
            )
    return output


@pl.jit
def decode_sparse_attn_test(
    absorbed_query: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16]],
):
    """Run one sparse decode step for golden.run validation."""
    absorbed_query.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    block_table.bind_dynamic(1, BLOCK_TABLE_DYN)
    decode_sparse_attn(
        absorbed_query, latent_cache, block_table, request_ids, topk_indices, output
    )
    return output


def build_decode_sparse_attn_specs(requests: int = 2, pages_per_request: int = 4):
    """Build one deterministic selection: ragged valid counts, padded tails.

    Every row selects a different number of positions and always keeps its own tail,
    which is what ``index_kpool_always_select_tail`` guarantees, so the ``-1`` lanes
    and the ``SPARSE_BLOCKS`` padding are both exercised. The counts ramp past
    ``ATTN_K_TILE`` so rows span one to four sparse blocks and the merge sees more
    than one live partial.

    Every row is front packed, which is the indexer's ABI: the ``-1`` lanes are one
    suffix per row. A fixture that interleaved them would be invalid input, not a
    harder case, so none is built here.
    """
    from golden import TensorSpec

    from models.glm5_3_flash.config import DECODE_ROWS_PER_REQUEST

    generator = torch.Generator().manual_seed(83)
    tokens = requests * DECODE_ROWS_PER_REQUEST
    cache_rows = requests * pages_per_request * BLOCK_SIZE
    visible = pages_per_request * BLOCK_SIZE

    def init_query():
        return torch.randn(tokens, LOCAL_H, KV_LORA, generator=generator).bfloat16()

    def init_cache():
        return torch.randn(cache_rows, KV_LORA, generator=generator).bfloat16()

    def init_block_table():
        pages = torch.randperm(requests * pages_per_request, generator=generator)
        return pages.reshape(requests, pages_per_request).to(torch.int32)

    def init_request_ids():
        return torch.arange(tokens, dtype=torch.int32) // DECODE_ROWS_PER_REQUEST

    def init_indices():
        indices = torch.full((tokens, TOPK_INDEX_WIDTH), -1, dtype=torch.int32)
        for token in range(tokens):
            if token == tokens - 1:
                continue
            count = min(37 + 61 * token, visible - 1)
            chosen = torch.randperm(visible - 1, generator=generator)[: count - 1]
            selected = torch.cat([chosen, torch.tensor([visible - 1])]).to(torch.int32)
            indices[token, : selected.numel()] = selected
        return indices

    return [
        TensorSpec(
            "absorbed_query", [tokens, LOCAL_H, KV_LORA], torch.bfloat16, init_value=init_query
        ),
        TensorSpec("latent_cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=init_cache),
        TensorSpec(
            "block_table", [requests, pages_per_request], torch.int32, init_value=init_block_table
        ),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=init_request_ids),
        TensorSpec(
            "topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32, init_value=init_indices
        ),
        TensorSpec("output", [tokens, LOCAL_H, KV_LORA], torch.bfloat16),
    ]


def golden_decode_sparse_attn_case(tensors):
    """Fill the expected output for :func:`build_decode_sparse_attn_specs`."""
    tensors["output"][:] = golden_decode_sparse_attn(
        tensors["absorbed_query"],
        tensors["latent_cache"],
        tensors["topk_indices"],
        tensors["block_table"],
        tensors["request_ids"],
    )


def main():
    """Prove the golden on CPU, then validate the sparse decode on device.

    The BF16 output carries the flash block split and the merge rescale on top of the
    input rounding, so its budget is looser than a plain projection's.
    """
    import argparse

    from golden import ratio_allclose, run
    from models.glm5_3_flash._golden_smoke import run_decode_sparse_attn_golden

    run_decode_sparse_attn_golden(golden_decode_sparse_attn)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run(
        fn=decode_sparse_attn_test,
        specs=build_decode_sparse_attn_specs(args.requests, args.pages),
        golden_fn=golden_decode_sparse_attn_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1.0 / 64,
        atol=1e-3,
        compare_fn={"output": ratio_allclose(atol=1e-3, rtol=1.0 / 64)},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "build_decode_sparse_attn_specs",
    "decode_sparse_attn",
    "decode_sparse_attn_test",
    "golden_decode_sparse_attn",
    "golden_decode_sparse_attn_case",
]


if __name__ == "__main__":
    main()
