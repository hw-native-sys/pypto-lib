# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sparse NoPE MLA attention over the indexer's selected rows, prefill path.

Each query row carries up to ``TOPK_INDEX_WIDTH = 2051`` int32 cache rows, padded
with ``-1``. The reference turns them into a dense boolean mask and runs SDPA, which
is only tractable because the mask is built with ``scatter_add``; on device the
kernel gathers the selected latent rows instead and runs an online-softmax
attention over them.

Scale is ``qk_head_dim ** -0.5`` = ``256 ** -0.5``; ``v_head_dim`` is 256, so the
per-head output is 256 wide and the heads flatten to ``[T, LOCAL_H * 256]``.

Causality and padding are already folded into the index list by the indexer — a row
that should not be visible is ``-1`` — so this kernel must not re-apply a causal
mask, only honour the ``-1`` sentinel.

**``topk_indices`` is front packed, and this kernel requires it**, exactly as the
decode path does. Every valid position precedes every ``-1``, so a row's padding is
one suffix — the indexer's ABI, see
:func:`models.glm5_3_flash.prefill_indexer.indexer_expand`. This kernel tests lane 0
of a 128-wide block to treat it as all padding, and lane 0 of the row to emit a zero
row; against an interleaved list both would drop the live entries behind the first
``-1``. The per-lane ``-1`` handling inside a gathered block is unconditional and
does not depend on the packing.

Like the decode path, the indexer emits **logical per-request positions**, so the
block table is part of this kernel's ABI: resolving it host-side would mean a
per-layer index translation outside the kernel.

**Prior art.** ``ops/pypto_python/impl/sparse_compress_flash_attention_pypto.py`` in
cann-recipes-infer implements sparse flash attention over selected rows on this
hardware generation. It targets the older ``pypto.Tensor`` / ``pypto_impl``
frontend, which this repo's pinned pypto does not export, so treat it as an
algorithm and tiling reference rather than code.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, B_DYN, KV_LORA, LOCAL_H, QK_DIM
from models.glm5_3_flash.config import BLOCK_TABLE_DYN, TABLE_DYN, TOPK_INDEX_WIDTH, T_DYN, V_DIM
from models.glm5_3_flash.metadata import paged_slots
from models.glm5_3_flash.mla_prolog import absorb_query

SOFTMAX_SCALE = QK_DIM**-0.5
NEG_INF = -1.0e20       # finite floor; a true -inf would make an all-masked row NaN
ATTN_K_TILE = 128       # selected rows per sparse block
H_TILE = min(LOCAL_H, 16)
if H_TILE != LOCAL_H:
    raise ValueError(f"sparse MLA attention needs TP >= 4 (LOCAL_H={LOCAL_H} > 16 heads per tile)")
H_PAD = ((H_TILE + 15) // 16) * 16
SPARSE_BLOCKS = (TOPK_INDEX_WIDTH + ATTN_K_TILE - 1) // ATTN_K_TILE
MM_T_TILE = 16          # cube M tile of the value expansion
EXPAND_N_TILE = 128     # value-expansion output-column tile over V_DIM
EXPAND_K_TILE = 256     # value-expansion KV_LORA reduction tile


def golden_prefill_sparse_attn(
    query: torch.Tensor,
    latent_cache: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """Attend over the indexer's selected rows in full head space.

    ``topk_indices`` holds **logical per-request positions**, so they are resolved
    through ``block_table`` with :func:`models.glm5_3_flash.metadata.paged_slots` — the
    same mapping the kernel performs on device. A ``-1`` entry marks a padded slot and
    is the *only* masking this kernel may apply: the indexer has already filtered for
    causality and request padding, so re-applying a causal mask here would drop rows the
    reference keeps. A row whose slots are all ``-1`` contributes nothing and yields a
    zero output row rather than a NaN.

    Keys and values are expanded from the gathered 512-wide latent rows, which is what
    prefill does; decode absorbs ``w_k`` / ``w_v`` into the query and ``o_proj``
    instead. This reference materialises ``[T, W, H, *]``, so it is meant for reduced
    fixtures and small validation batches, not for a full 2051-wide production step.

    Args:
        query: ``[T, H, QK_DIM]`` from :func:`models.glm5_3_flash.mla_prolog.golden_mla_prolog`.
        latent_cache: ``[TABLE * BLOCK_SIZE, KV_LORA]`` paged 512-latent pool.
        w_k: ``[H, QK_DIM, KV_LORA]`` key half of ``kv_b_proj``.
        w_v: ``[H, V_DIM, KV_LORA]`` value half of ``kv_b_proj``.
        topk_indices: ``[T, W]`` int32 logical positions, ``-1`` padded.
        block_table: ``[B, BLOCK_TABLE]`` int32 logical-to-physical page map.
        request_ids: ``[T]`` int32 owning request of each query row.

    Returns:
        ``[T, H, V_DIM]`` attention output in ``query``'s dtype.
    """
    valid = topk_indices >= 0
    positions = topk_indices.clamp(min=0).to(torch.int64)
    rows = paged_slots(positions, request_ids.to(torch.int64).unsqueeze(-1), block_table)
    latent = latent_cache[rows].float()
    key = torch.einsum("twk,hdk->twhd", latent, w_k.float())
    value = torch.einsum("twk,hvk->twhv", latent, w_v.float())
    scale = query.shape[-1] ** -0.5
    scores = torch.einsum("thd,twhd->thw", query.float(), key) * scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
    return torch.einsum("thw,twhv->thv", weights, value).to(query.dtype)


@pl.jit.inline
def prefill_sparse_attn(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    w_v: pl.Tensor[[LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
):
    """Sparse attention over the selected latent rows, prefill path.

    **This attends in latent space, not head space.** The reference expands ``w_k``
    and ``w_v`` per gathered row, which is the definition but not a tractable
    schedule: every query token owns its own index list, so an expanded tile could
    never be reused across tokens and each block would cost ``ATTN_K_TILE x KV_LORA x
    QK_DIM`` twice over. Absorbing instead — ``q . K^T`` as ``(q @ w_k) . latent^T``,
    and ``P @ V`` as ``(P @ latent) @ w_v^T`` — is the same value with the two
    expansions moved out of the block loop and onto the token, where ``w_v`` is
    applied once to the finished context. The cost of the identity is one extra BF16
    rounding of the query and of the context, so the kernel is close to, but not
    bit-comparable with, the expanded reference.

    One work item per token: the online softmax runs over ``SPARSE_BLOCKS`` blocks
    inside the item with ``pl.yield_``-carried ``(max, sum, context)``, so no
    per-block partials are materialised. At 8,192 prefill tokens a partial buffer
    would have been ``T x SPARSE_BLOCKS x H_TILE x KV_LORA`` FP32, which is why the
    decode scope's split-and-merge shape is not reused here.

    Gather, ``-1`` masking, the ``NEG_INF`` floor and the padding of head rows to
    ``H_PAD`` = 16 all follow :mod:`models.glm5_3_flash.decode_sparse_attn`; causality
    and request padding are already folded into the index list.

    A block whose lane 0 is ``-1`` is all padding — the front-packing ABI is what
    makes that one lane conclusive — and takes one wide gather of the pool's first
    ``ATTN_K_TILE`` rows, then runs like any other block rather than skipping the
    merge: with the
    bias row left at ``NEG_INF`` its ``beta`` is zero, so it contributes nothing. The
    branch must not decide whether the carried state is updated. A branch the carried
    tiles flow through forces the partitioner to materialise their pre-loop
    initialisation in the cube half of this mixed cube/vector scope, where
    ``vector_dup`` does not exist and ``ccec`` rejects the kernel -- ``a2a3sim``
    compiles both halves with ``g++`` and does not see it. The filler rows keep the
    second matmul finite, which a skip over uninitialised L1 would not.

    Returns the attention output so the caller can chain it.
    """
    t_dim = pl.tensor.dim(query, 0)
    absorbed = pl.create_tensor([t_dim, LOCAL_H, KV_LORA], dtype=pl.BF16)
    absorb_query(query, w_k, absorbed)
    absorbed_rows = pl.reshape(absorbed, [t_dim * LOCAL_H, KV_LORA])

    # Head-major: the value expansion needs its M rows contiguous per head.
    context = pl.create_tensor([LOCAL_H * t_dim, KV_LORA], dtype=pl.BF16)
    with pl.spmd(t_dim, name_hint="prefill_sparse_flash") as flash_tid:
        token = pl.tile.get_block_idx()
        request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
        q_tile = pl.fillpad(
            pl.slice(
                absorbed_rows,
                [H_PAD, KV_LORA],
                [token * LOCAL_H, 0],
                valid_shape=[H_TILE, KV_LORA],
            ),
            pad_value=pl.PadValue.zero,
        )
        running_m = pl.reshape(pl.full([1, H_PAD], dtype=pl.FP32, value=NEG_INF), [H_PAD, 1])
        running_l = pl.reshape(pl.full([1, H_PAD], dtype=pl.FP32, value=0.0), [H_PAD, 1])
        running_o = pl.full([H_PAD, KV_LORA], dtype=pl.FP32, value=0.0)
        for sb, (m_iter, l_iter, o_iter) in pl.range(
            SPARSE_BLOCKS, init_values=(running_m, running_l, running_o)
        ):
            w0 = sb * ATTN_K_TILE
            kv_tile = pl.create_l1([ATTN_K_TILE, KV_LORA], pl.BF16)
            bias_row = pl.full([1, ATTN_K_TILE], dtype=pl.FP32, value=NEG_INF)
            # Lane 0 is -1 only when the whole block is padding.
            if pl.read(topk_indices, [token, w0]) >= 0:
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
                        kv_tile = pl.gather_row(
                            kv_tile, latent_cache, [r, 0], [0, 0], [1, KV_LORA]
                        )
            else:
                kv_tile = pl.gather_row(
                    kv_tile, latent_cache, [0, 0], [0, 0], [ATTN_K_TILE, KV_LORA]
                )

            raw = pl.matmul(q_tile, kv_tile, b_trans=True, out_dtype=pl.FP32)
            scores = pl.col_expand_add(pl.mul(raw, SOFTMAX_SCALE), bias_row)
            block_m = pl.row_max(scores)
            weights = pl.exp(pl.row_expand_sub(scores, block_m))
            block_l = pl.row_sum(weights)
            block_o = pl.matmul(
                pl.cast(weights, target_type=pl.BF16, mode="rint"),
                kv_tile,
                out_dtype=pl.FP32,
            )
            next_m = pl.maximum(m_iter, block_m)
            alpha = pl.exp(pl.sub(m_iter, next_m))
            beta = pl.exp(pl.sub(block_m, next_m))
            next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, block_l))
            next_o = pl.add(
                pl.row_expand_mul(o_iter, alpha), pl.row_expand_mul(block_o, beta)
            )
            running_m, running_l, running_o = pl.yield_(next_m, next_l, next_o)

        # Lane 0 of the row is -1 only when the row selects nothing at all.
        if pl.read(topk_indices, [token, 0]) >= 0:
            normalized = pl.cast(
                pl.row_expand_div(running_o, running_l), target_type=pl.BF16, mode="rint"
            )
            for head in pl.unroll(H_TILE):
                ctx_row = head * t_dim + token
                context[ctx_row : ctx_row + 1, 0:KV_LORA] = normalized[head : head + 1, 0:KV_LORA]
        else:
            empty_row = pl.full([1, KV_LORA], dtype=pl.BF16, value=0.0)
            for head in pl.unroll(H_TILE):
                ctx_row = head * t_dim + token
                context[ctx_row : ctx_row + 1, 0:KV_LORA] = empty_row

    output_flat = pl.reshape(output, [t_dim, LOCAL_H * V_DIM])
    w_v_flat = pl.reshape(w_v, [LOCAL_H * V_DIM, KV_LORA])
    t_mm = ((t_dim + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
    with pl.spmd(
        LOCAL_H * (V_DIM // EXPAND_N_TILE), name_hint="prefill_value_expand", deps=[flash_tid]
    ):
        expand_idx = pl.tile.get_block_idx()
        head = expand_idx // (V_DIM // EXPAND_N_TILE)
        n0 = (expand_idx % (V_DIM // EXPAND_N_TILE)) * EXPAND_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            valid_rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = pl.create_tensor([MM_T_TILE, EXPAND_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(KV_LORA // EXPAND_K_TILE, stage=2):
                k0 = kb * EXPAND_K_TILE
                ctx_tile = pl.slice(
                    context,
                    [MM_T_TILE, EXPAND_K_TILE],
                    [head * t_dim + t0, k0],
                    valid_shape=[valid_rows, EXPAND_K_TILE],
                )
                w_tile = pl.slice(
                    w_v_flat,
                    [EXPAND_N_TILE, EXPAND_K_TILE],
                    [head * V_DIM + n0, k0],
                )
                acc = pl.matmul_acc(acc, ctx_tile, w_tile, b_trans=True, init_cond=(kb == 0))
            expanded = pl.cast(acc, target_type=pl.BF16, mode="rint")
            output_flat = pl.assemble(
                output_flat,
                pl.set_validshape(expanded, valid_rows, EXPAND_N_TILE),
                [t0, head * V_DIM + n0],
            )
    return output


@pl.jit
def prefill_sparse_attn_test(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    w_v: pl.Tensor[[LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16]],
):
    """Run one sparse prefill step for golden.run validation."""
    query.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    block_table.bind_dynamic(1, BLOCK_TABLE_DYN)
    prefill_sparse_attn(
        query, latent_cache, block_table, request_ids, w_k, w_v, topk_indices, output
    )
    return output


def build_prefill_sparse_attn_specs(requests: int = 2, rows: int = 6, pages_per_request: int = 4):
    """Build one deterministic prefill selection with ragged per-token counts.

    The counts ramp past ``ATTN_K_TILE`` so rows span one to four sparse blocks, which
    is what exercises the ``pl.yield_`` merge rather than just the single-block path.
    Row ``rows`` selects nothing, which is the padded-batch row the kernel must answer
    with zeros rather than a division by zero; the one-block path it would otherwise
    have covered is still covered by row 0.

    Every row is front packed, which is the indexer's ABI: the ``-1`` lanes are one
    suffix per row. A fixture that interleaved them would be invalid input, not a
    harder case, so none is built here.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(89)
    tokens = requests * rows
    cache_rows = requests * pages_per_request * BLOCK_SIZE
    visible = pages_per_request * BLOCK_SIZE

    def init_query():
        return torch.randn(tokens, LOCAL_H, QK_DIM, generator=generator).bfloat16()

    def init_cache():
        return torch.randn(cache_rows, KV_LORA, generator=generator).bfloat16()

    def init_w_k():
        return (torch.randn(LOCAL_H, QK_DIM, KV_LORA, generator=generator) * 0.02).bfloat16()

    def init_w_v():
        return (torch.randn(LOCAL_H, V_DIM, KV_LORA, generator=generator) * 0.02).bfloat16()

    def init_block_table():
        pages = torch.randperm(requests * pages_per_request, generator=generator)
        return pages.reshape(requests, pages_per_request).to(torch.int32)

    def init_request_ids():
        return (torch.arange(tokens, dtype=torch.int32) // rows).to(torch.int32)

    def init_indices():
        indices = torch.full((tokens, TOPK_INDEX_WIDTH), -1, dtype=torch.int32)
        for token in range(tokens):
            if token == rows:
                continue
            count = min(37 + 113 * (token % rows), visible - 1)
            chosen = torch.randperm(visible - 1, generator=generator)[: count - 1]
            selected = torch.cat([chosen, torch.tensor([visible - 1])]).to(torch.int32)
            indices[token, : selected.numel()] = selected
        return indices

    return [
        TensorSpec("query", [tokens, LOCAL_H, QK_DIM], torch.bfloat16, init_value=init_query),
        TensorSpec("latent_cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=init_cache),
        TensorSpec(
            "block_table", [requests, pages_per_request], torch.int32, init_value=init_block_table
        ),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=init_request_ids),
        TensorSpec("w_k", [LOCAL_H, QK_DIM, KV_LORA], torch.bfloat16, init_value=init_w_k),
        TensorSpec("w_v", [LOCAL_H, V_DIM, KV_LORA], torch.bfloat16, init_value=init_w_v),
        TensorSpec(
            "topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32, init_value=init_indices
        ),
        TensorSpec("output", [tokens, LOCAL_H, V_DIM], torch.bfloat16),
    ]


def golden_prefill_sparse_attn_case(tensors):
    """Fill the expected output for :func:`build_prefill_sparse_attn_specs`."""
    tensors["output"][:] = golden_prefill_sparse_attn(
        tensors["query"],
        tensors["latent_cache"],
        tensors["w_k"],
        tensors["w_v"],
        tensors["topk_indices"],
        tensors["block_table"],
        tensors["request_ids"],
    )


def main():
    """Prove the golden on CPU, then validate the sparse prefill on device.

    The kernel absorbs where the reference expands, so on top of the flash block split it
    carries two extra BF16 roundings and its budget is a rounding wider than the decode
    scope's.
    """
    import argparse

    from golden import ratio_allclose, run
    from models.glm5_3_flash._golden_smoke import run_prefill_sparse_attn_golden

    run_prefill_sparse_attn_golden(golden_prefill_sparse_attn)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run(
        fn=prefill_sparse_attn_test,
        specs=build_prefill_sparse_attn_specs(args.requests, args.rows, args.pages),
        golden_fn=golden_prefill_sparse_attn_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1.0 / 32,
        atol=2e-3,
        compare_fn={"output": ratio_allclose(atol=2e-3, rtol=1.0 / 32)},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "build_prefill_sparse_attn_specs",
    "golden_prefill_sparse_attn",
    "golden_prefill_sparse_attn_case",
    "prefill_sparse_attn",
    "prefill_sparse_attn_test",
]


if __name__ == "__main__":
    main()
