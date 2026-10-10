# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The kpool DSA indexer, decode path: one continuous-batch step, end to end.

Decode runs the same per-query math as prefill, so the kernels live in
:mod:`models.glm5_3_flash.prefill_indexer` and this file contributes the
composition: one dispatch of ``DECODE_ROWS_PER_REQUEST`` = 4 packed rows per
request (one live token plus three MTP spec tokens) drives the whole pipeline,

1. :func:`~models.glm5_3_flash.prefill_indexer.indexer_proj` projects the 4
   rows' queries, keys, head weights and gate scores;
2. :func:`~models.glm5_3_flash.indexer_cache.indexer_cache_write` scatters the
   raw ``[key, gate]`` rows into the support table;
3. :func:`~models.glm5_3_flash.indexer_cache.indexer_pool_write` closes the one
   pool this step completes per request — four new rows always finish exactly
   one pool window, whatever the request's length alignment, so the pooled
   table grows by exactly one row per request per step and the incomplete tail
   pool is never stored (``index_kpool_always_select_tail`` force-selects its
   raw positions in the expansion);
4. :func:`~models.glm5_3_flash.prefill_indexer.indexer_score` scores the
   queries against the pooled table, this time including the just-closed row;
5. :func:`~models.glm5_3_flash.prefill_indexer.indexer_topk` selects the top
   512 pools per query row;
6. :func:`~models.glm5_3_flash.prefill_indexer.indexer_expand` emits the
   front-packed 2051-wide index list both sparse attention kernels consume.

The four rows of one request share nothing below the selection:
``selected_pools``/``selected_valid`` are the per-request shareable surface
(the settled ``index_share_for_mtp_iteration`` ABI, issue #1267), while each
row's expansion is rebuilt from its own tail window. The MTP sharing path
itself stays out of this phase — every row projects, scores, selects and
expands independently.

Everything the composition needs beyond the tensors is host-lowered metadata —
physical cache rows and segment offsets per :mod:`models.glm5_3_flash.metadata`
— the same way the standalone kernel tests consume it. The projection stage
needs a token count that is a multiple of 64 (``PROJ_T_TILE``), which a decode
batch reaches by padding to whole 16-request groups (4 rows per request);
padded rows carry ordinary metadata and simply compute selections nobody
gathers.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, D, INDEX_DIM, INDEX_H
from models.glm5_3_flash.config import INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE
from models.glm5_3_flash.config import INDEX_STATE_WIDTH, KPOOL_SELECT_K, KV_LORA, LOCAL_H, POOLS_DYN
from models.glm5_3_flash.config import Q_LORA, TABLE_DYN, TOPK_INDEX_WIDTH, T_DYN
from models.glm5_3_flash.decode_sparse_attn import decode_sparse_attn_test
from models.glm5_3_flash.decode_sparse_attn import golden_decode_sparse_attn_case
from models.glm5_3_flash.indexer_cache import POOL_SCALE_WIDTH
from models.glm5_3_flash.indexer_cache import POOL_VALID_WIDTH
from models.glm5_3_flash.indexer_cache import indexer_cache_write, indexer_pool_write
from models.glm5_3_flash.prefill_indexer import LEAF
from models.glm5_3_flash.prefill_indexer import golden_indexer_expand, golden_indexer_proj
from models.glm5_3_flash.prefill_indexer import golden_indexer_score, golden_indexer_topk
from models.glm5_3_flash.prefill_indexer import indexer_expand, indexer_proj
from models.glm5_3_flash.prefill_indexer import indexer_score_token, indexer_topk
from models.glm5_3_flash.prefill_indexer import sylvester_hadamard
from models.glm5_3_flash.indexer_cache import golden_indexer_cache_write, golden_indexer_pool_write
from models.glm5_3_flash.quantization import quantize_per_token_int8


@pl.jit
def decode_indexer_step_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    index_slots: pl.Tensor[[T_DYN], pl.INT32],
    pool_token_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_blocks: pl.Tensor[[POOLS_DYN], pl.INT32],
    seg_start: pl.Tensor[[T_DYN], pl.INT32],
    pool_count: pl.Tensor[[T_DYN], pl.INT32],
    tail_start: pl.Tensor[[T_DYN], pl.INT32],
    tail_count: pl.Tensor[[T_DYN], pl.INT32],
    kv_len: pl.Tensor[[T_DYN], pl.INT32],
    raw_cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32]],
    pool_cache: pl.InOut[pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.INT8]],
    pool_scale: pl.InOut[pl.Tensor[[POOLS_DYN, POOL_SCALE_WIDTH], pl.FP32]],
    pool_valid: pl.Out[pl.Tensor[[POOLS_DYN, POOL_VALID_WIDTH], pl.INT32]],
    index_q: pl.Out[pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16]],
    head_weights: pl.Out[pl.Tensor[[T_DYN, INDEX_H], pl.FP32]],
    gate_scores: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32]],
    index_scores: pl.Out[pl.Tensor[[T_DYN, POOLS_DYN], pl.FP32]],
    selected_pools: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.Out[pl.Tensor[[T_DYN, KPOOL_SELECT_K], pl.INT32]],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32]],
):
    """Run one whole decode step for golden.run validation.

    The raw scatter, the pool close, the score, the selection and the expansion
    order themselves through their shared tensors; every stage's output is
    exposed so the harness can validate the composition, not just the tail.
    """
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    index_slots.bind_dynamic(0, T_DYN)
    seg_start.bind_dynamic(0, T_DYN)
    pool_count.bind_dynamic(0, T_DYN)
    tail_start.bind_dynamic(0, T_DYN)
    tail_count.bind_dynamic(0, T_DYN)
    kv_len.bind_dynamic(0, T_DYN)
    pool_token_slots.bind_dynamic(0, POOLS_DYN)
    pool_slots.bind_dynamic(0, POOLS_DYN)
    pool_blocks.bind_dynamic(0, POOLS_DYN)
    pool_valid.bind_dynamic(0, POOLS_DYN)
    index_q.bind_dynamic(0, T_DYN)
    index_k.bind_dynamic(0, T_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(0, T_DYN)
    selected_pools.bind_dynamic(0, T_DYN)
    selected_valid.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    index_scores.bind_dynamic(1, POOLS_DYN)

    indexer_proj(
        x,
        q_resid,
        w_q_b,
        w_k,
        k_norm_weight,
        k_norm_bias,
        w_weights,
        w_compress_gate,
        index_q,
        index_k,
        head_weights,
        gate_scores,
    )
    indexer_cache_write(index_k, gate_scores, index_slots, raw_cache)
    indexer_pool_write(
        raw_cache, compress_ape, hadamard, pool_token_slots, pool_slots, pool_cache, pool_scale, pool_valid
    )
    indexer_score_token(
        index_q,
        hadamard,
        pool_cache,
        pool_scale,
        pool_blocks,
        head_weights,
        seg_start,
        pool_count,
        index_scores,
    )
    indexer_topk(index_scores, seg_start, pool_count, selected_pools, selected_valid)
    indexer_expand(selected_pools, selected_valid, tail_start, tail_count, kv_len, topk_indices)
    return (
        index_q,
        index_k,
        head_weights,
        gate_scores,
        raw_cache,
        pool_cache,
        pool_scale,
        pool_valid,
        index_scores,
        selected_pools,
        selected_valid,
        topk_indices,
    )


def build_decode_indexer_step_specs(requests: int = 16, prior_counts: int | tuple[int, ...] | None = None):
    """Build one deterministic decode step: sixteen requests, one closing pool each.

    The requests carry prior contexts of different lengths, so one closing pool
    is made of four fresh rows, another of fresh rows mixed with rows cached by
    earlier steps, and the pooled table starts with rows only earlier steps
    could have written. Every request contributes exactly
    ``DECODE_ROWS_PER_REQUEST`` query rows. ``prior_counts`` overrides the
    fixture's per-request pool history for the business-shape benchmark points;
    an int means the same history for every request.
    """
    from golden import TensorSpec

    from models.glm5_3_flash.config import DECODE_ROWS_PER_REQUEST

    rows_per_request = DECODE_ROWS_PER_REQUEST
    tokens = requests * rows_per_request
    if tokens % 64:
        raise ValueError("the projection stage needs a token count that is a multiple of 64")
    if prior_counts is None:
        prior_counts = (13, 5, 40, 2, 9, 21, 60, 3, 33, 7, 50, 17, 26, 11, 44, 6)[:requests]
    elif isinstance(prior_counts, int):
        prior_counts = (prior_counts,) * requests
    if len(prior_counts) != requests:
        raise ValueError("prior_counts must hold one pool count per request")
    new_counts = tuple(count + 1 for count in prior_counts)
    pools_true = sum(new_counts)
    width = ((pools_true + 2 * LEAF - 2) // LEAF) * LEAF

    generator = torch.Generator().manual_seed(89)
    prior_tokens = [4 * count for count in prior_counts]

    def row_map():
        """Physical raw row of every request's logical token, past and new.

        Pages come from one global permutation, partitioned request by
        request: per-request draws from a shared page pool can hand two
        requests the same physical slots, which makes the cache writer race
        and lets a pool close read another request's keys and gates.
        """
        rows = {}
        pages_per_request = [
            (prior_tokens[request] + rows_per_request + 127) // 128 + 1 for request in range(requests)
        ]
        total_pages = sum(pages_per_request)
        page_pool = max(8, total_pages + 4)
        permutation = torch.randperm(page_pool, generator=generator)
        cursor = 0
        for request in range(requests):
            take = pages_per_request[request]
            pages = permutation[cursor : cursor + take]
            cursor += take
            total = prior_tokens[request] + rows_per_request
            for position in range(total):
                rows[(request, position)] = int(pages[position // 128]) * 128 + position % 128
        all_rows = list(rows.values())
        if len(set(all_rows)) != len(all_rows):
            raise AssertionError("raw cache rows collide across requests")
        return rows

    mapping = row_map()

    def raw_rows_for(request, first, count):
        return [mapping[(request, position)] for position in range(first, first + count)]

    # The pooled table is paged at INDEX_STATE_BLOCK_SIZE rows like the prefill
    # fixtures: each compacted page lands at its own physical base, and a
    # pool's physical row derives from its compacted id.
    pages = width // INDEX_STATE_BLOCK_SIZE
    pool_table_rows = pages * INDEX_STATE_BLOCK_SIZE
    page_bases = torch.randperm(pages, generator=generator).to(torch.int32) * INDEX_STATE_BLOCK_SIZE
    seg_offsets = []
    offset = 0
    for count in new_counts:
        seg_offsets.append(offset)
        offset += count

    def pool_row_of_id(compacted):
        return int(page_bases[compacted // INDEX_STATE_BLOCK_SIZE]) + compacted % INDEX_STATE_BLOCK_SIZE

    pool_row_of = {}
    for request in range(requests):
        for pool in range(new_counts[request]):
            pool_row_of[(request, pool)] = pool_row_of_id(seg_offsets[request] + pool)

    seg_start = torch.zeros(tokens, dtype=torch.int32)
    pool_count = torch.zeros(tokens, dtype=torch.int32)
    tail_start = torch.zeros(tokens, dtype=torch.int32)
    tail_count = torch.zeros(tokens, dtype=torch.int32)
    kv_len = torch.zeros(tokens, dtype=torch.int32)
    for request in range(requests):
        for row in range(rows_per_request):
            token = request * rows_per_request + row
            position = prior_tokens[request] + row
            length = position + 1
            seg_start[token] = sum(new_counts[:request])
            pool_count[token] = min(length // INDEX_KPOOL, new_counts[request])
            tail_count[token] = length % INDEX_KPOOL
            tail_start[token] = length - int(tail_count[token])
            kv_len[token] = length

    closing_slots = torch.full((requests, INDEX_KPOOL), -1, dtype=torch.int32)
    closing_dest = torch.zeros(requests, dtype=torch.int32)
    for request in range(requests):
        closing = new_counts[request] - 1
        first = closing * INDEX_KPOOL
        closing_slots[request] = torch.tensor(raw_rows_for(request, first, INDEX_KPOOL), dtype=torch.int32)
        closing_dest[request] = pool_row_of[(request, closing)]

    index_slots = torch.zeros(tokens, dtype=torch.int32)
    for request in range(requests):
        for row in range(rows_per_request):
            index_slots[request * rows_per_request + row] = mapping[(request, prior_tokens[request] + row)]

    def init_x():
        return torch.randn(tokens, D, generator=generator, dtype=torch.float32).bfloat16()

    def init_q_resid():
        return torch.randn(tokens, Q_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_raw_cache():
        rows = max(mapping.values()) + 1
        cache = torch.randn(rows, INDEX_STATE_WIDTH, generator=generator)
        cache[:, :INDEX_DIM] = cache[:, :INDEX_DIM].bfloat16().float()
        return cache

    # The pooled table starts as the quantized form earlier steps would have
    # left behind: INT8 rows already rotated into the Hadamard basis, paired
    # with their dequant scales, one draw feeding both so the pairs stay
    # consistent.
    prior_i8, prior_scale_col = quantize_per_token_int8(
        torch.randn(pool_table_rows, INDEX_DIM, generator=generator, dtype=torch.float32)
    )

    def init_pool_cache():
        return prior_i8

    def init_pool_scale():
        # The writer replicates the scale across all POOL_SCALE_WIDTH lanes and
        # the scorer reads the row sum scaled by 1/8, so the prior table must
        # carry the replication too.
        return prior_scale_col.expand(-1, POOL_SCALE_WIDTH).contiguous().clone()

    shapes = {
        "w_q_b": (Q_LORA, INDEX_H * INDEX_DIM),
        "w_k": (D, INDEX_DIM),
        "w_weights": (D, INDEX_H),
        "w_compress_gate": (D, INDEX_DIM),
    }

    def init_weights(name, scale):
        def build():
            return (torch.randn(*shapes[name], generator=generator) * scale).bfloat16()

        return build

    specs = [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=init_x),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16, init_value=init_q_resid),
        TensorSpec(
            "w_q_b",
            [Q_LORA, INDEX_H * INDEX_DIM],
            torch.bfloat16,
            init_value=init_weights("w_q_b", 0.02),
        ),
        TensorSpec("w_k", [D, INDEX_DIM], torch.bfloat16, init_value=init_weights("w_k", 0.02)),
        TensorSpec(
            "k_norm_weight",
            [INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: (1.0 + 0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16(),
        ),
        TensorSpec(
            "k_norm_bias",
            [INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: (0.1 * torch.randn(INDEX_DIM, generator=generator)).bfloat16(),
        ),
        TensorSpec("w_weights", [D, INDEX_H], torch.bfloat16, init_value=init_weights("w_weights", 0.05)),
        TensorSpec(
            "w_compress_gate",
            [D, INDEX_DIM],
            torch.bfloat16,
            init_value=init_weights("w_compress_gate", 0.02),
        ),
        TensorSpec("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard),
        TensorSpec(
            "compress_ape",
            [INDEX_KPOOL, INDEX_DIM],
            torch.bfloat16,
            init_value=lambda: torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16(),
        ),
        TensorSpec("index_slots", [tokens], torch.int32, init_value=lambda: index_slots),
        TensorSpec(
            "pool_token_slots", [requests, INDEX_KPOOL], torch.int32, init_value=lambda: closing_slots
        ),
        TensorSpec("pool_slots", [requests], torch.int32, init_value=lambda: closing_dest),
        TensorSpec("pool_blocks", [pages], torch.int32, init_value=lambda: page_bases),
        TensorSpec("seg_start", [tokens], torch.int32, init_value=lambda: seg_start),
        TensorSpec("pool_count", [tokens], torch.int32, init_value=lambda: pool_count),
        TensorSpec("tail_start", [tokens], torch.int32, init_value=lambda: tail_start),
        TensorSpec("tail_count", [tokens], torch.int32, init_value=lambda: tail_count),
        TensorSpec("kv_len", [tokens], torch.int32, init_value=lambda: kv_len),
        TensorSpec(
            "raw_cache",
            [max(mapping.values()) + 1, INDEX_STATE_WIDTH],
            torch.float32,
            init_value=init_raw_cache,
        ),
        TensorSpec("pool_cache", [pool_table_rows, INDEX_DIM], torch.int8, init_value=init_pool_cache),
        TensorSpec(
            "pool_scale", [pool_table_rows, POOL_SCALE_WIDTH], torch.float32, init_value=init_pool_scale
        ),
        TensorSpec("pool_valid", [requests, POOL_VALID_WIDTH], torch.int32),
        TensorSpec("index_q", [tokens, INDEX_H, INDEX_DIM], torch.bfloat16),
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16),
        TensorSpec("head_weights", [tokens, INDEX_H], torch.float32),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32),
        TensorSpec("index_scores", [tokens, width], torch.float32),
        TensorSpec("selected_pools", [tokens, KPOOL_SELECT_K], torch.int32),
        TensorSpec("selected_valid", [tokens, KPOOL_SELECT_K], torch.int32),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32),
    ]
    return specs


def golden_decode_indexer_step_case(tensors):
    """Fill every expected output of one decode step by composing the goldens."""
    index_q, index_k, head_weights, gate_scores = golden_indexer_proj(
        tensors["x"],
        tensors["q_resid"],
        tensors["w_q_b"],
        tensors["w_k"],
        tensors["k_norm_weight"],
        tensors["k_norm_bias"],
        tensors["w_weights"],
        tensors["w_compress_gate"],
    )
    tensors["index_q"][:] = index_q
    tensors["index_k"][:] = index_k
    tensors["head_weights"][:] = head_weights
    tensors["gate_scores"][:] = gate_scores
    tensors["raw_cache"][:] = golden_indexer_cache_write(
        tensors["raw_cache"], index_k, gate_scores, tensors["index_slots"]
    )
    pool_cache, pool_scale, pool_valid = golden_indexer_pool_write(
        tensors["pool_cache"],
        tensors["pool_scale"],
        tensors["raw_cache"],
        tensors["compress_ape"],
        tensors["hadamard"],
        tensors["pool_token_slots"],
        tensors["pool_slots"],
    )
    tensors["pool_cache"][:] = pool_cache
    tensors["pool_scale"][:] = pool_scale
    tensors["pool_valid"][:] = pool_valid
    tensors["index_scores"][:] = golden_indexer_score(
        tensors["index_q"],
        tensors["hadamard"],
        tensors["pool_cache"],
        tensors["pool_scale"],
        tensors["pool_blocks"],
        tensors["head_weights"],
        tensors["seg_start"],
        tensors["pool_count"],
    )
    selected, valid = golden_indexer_topk(
        tensors["index_scores"], tensors["seg_start"], tensors["pool_count"]
    )
    tensors["selected_pools"][:] = selected
    tensors["selected_valid"][:] = valid
    tensors["topk_indices"][:] = golden_indexer_expand(
        tensors["selected_pools"],
        tensors["selected_valid"],
        tensors["tail_start"],
        tensors["tail_count"],
        tensors["kv_len"],
    )
    for token in range(tensors["topk_indices"].shape[0]):
        row = tensors["topk_indices"][token]
        live = int((row >= 0).sum())
        assert (row[:live] >= 0).all() and (row[live:] == -1).all(), (
            f"row {token} violates the front-packed index ABI"
        )


def build_decode_indexer_abi_specs(topk_indices: torch.Tensor, kv_len: torch.Tensor):
    """Build the sparse-attention consumer specs around a real index list.

    Takes the decode step's actual front-packed ``topk_indices`` together with
    the per-row ``kv_len`` it was produced under, and pages each request's
    logical positions into a fresh latent pool the way the MLA cache would, so
    the stream B gather resolves exactly the rows serving will resolve. The
    latent pool is its own table: the indexer's raw row map and the attention
    page map are independent layouts over the same logical positions, which is
    the point of the resolution contract.
    """
    from golden import TensorSpec

    from models.glm5_3_flash.config import DECODE_ROWS_PER_REQUEST

    tokens = topk_indices.shape[0]
    requests = tokens // DECODE_ROWS_PER_REQUEST
    pages_per_request = (int(kv_len.max()) + BLOCK_SIZE - 1) // BLOCK_SIZE

    generator = torch.Generator().manual_seed(101)
    pages = torch.randperm(requests * pages_per_request, generator=generator)
    block_table = pages.reshape(requests, pages_per_request).to(torch.int32)
    cache_rows = requests * pages_per_request * BLOCK_SIZE
    request_ids = torch.arange(tokens, dtype=torch.int32) // DECODE_ROWS_PER_REQUEST

    return [
        TensorSpec(
            "absorbed_query",
            [tokens, LOCAL_H, KV_LORA],
            torch.bfloat16,
            init_value=lambda: torch.randn(
                tokens, LOCAL_H, KV_LORA, generator=generator, dtype=torch.float32
            ).bfloat16(),
        ),
        TensorSpec(
            "latent_cache",
            [cache_rows, KV_LORA],
            torch.bfloat16,
            init_value=lambda: torch.randn(cache_rows, KV_LORA, generator=generator).bfloat16(),
        ),
        TensorSpec(
            "block_table",
            [requests, pages_per_request],
            torch.int32,
            init_value=lambda: block_table,
        ),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=lambda: request_ids),
        TensorSpec(
            "topk_indices",
            [tokens, TOPK_INDEX_WIDTH],
            torch.int32,
            init_value=lambda: topk_indices.clone(),
        ),
        TensorSpec("output", [tokens, LOCAL_H, KV_LORA], torch.bfloat16),
    ]


def main():
    """Prove the composed golden on CPU, then validate one decode step on device."""
    import argparse

    from golden import ratio_allclose, run, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument(
        "--case",
        default="all",
        choices=["step", "abi", "all"],
        help="step: the indexer composition alone; abi: the composition with the sparse "
        "attention consuming its index list; all: both",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="run the step at a business shape without golden validation; the shape "
        "comes from --requests/--prior-tokens, timing from PYPTO_BENCH=1",
    )
    parser.add_argument(
        "--prior-tokens",
        type=int,
        default=8192,
        help="per-request history length in tokens; defines the bench geometry",
    )
    args = parser.parse_args()

    def print_bench(label: str, result) -> None:
        stats = result.bench
        if stats is None:
            print(f"[BENCH] {label}: no timing (run with PYPTO_BENCH=1)")
            return
        print(
            f"[BENCH] {label}: device_us median={stats.device_us_median:.1f} "
            f"min={stats.device_us_min:.1f} mean={stats.device_us_mean:.1f} "
            f"max={stats.device_us_max:.1f} rounds={stats.rounds}"
        )

    if args.bench:
        if args.requests % 16:
            parser.error("--bench needs a multiple of 16 requests (projection tile multiple of 64)")
        prior_pools = args.prior_tokens // INDEX_KPOOL
        specs = build_decode_indexer_step_specs(args.requests, prior_pools)
        pools_true = args.requests * (prior_pools + 1)
        width = ((pools_true + 2 * LEAF - 2) // LEAF) * LEAF
        label = (
            f"decode_step requests={args.requests} prior_tokens={args.prior_tokens} "
            f"T={args.requests * 4} pools={pools_true} width={width}"
        )
        result = run(
            fn=decode_indexer_step_test,
            specs=specs,
            golden_fn=None,
            config={"platform": args.platform, "device_id": args.device},
            compile_only=args.compile_only,
        )
        print_bench(label, result)
        if not result.passed:
            raise SystemExit(result.error or 1)
        return

    def selected_pools_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        """Tie-aware selection compare that also audits the picked scores.

        :func:`topk_pair_compare` adjudicates index mismatches through the
        actual ordering alone, so a sorted strictly-worse subset would pass
        it. The picked scores must therefore also match the scores the golden
        picked, under the same tolerance: a legal tie swap keeps the paired
        values equal, a real miss breaks them.
        """
        scores = actual_outputs["index_scores"].float()
        seg0 = inputs["seg_start"].long().unsqueeze(1)

        def paired_of(selected):
            cols = (selected.long() + seg0).clamp(0, scores.shape[-1] - 1)
            paired = torch.gather(scores, 1, cols)
            return torch.where(selected < 0, torch.full_like(paired, -torch.inf), paired)

        paired = paired_of(actual)
        synth_outputs = {**actual_outputs, "_selected_paired_scores": paired}
        ok, msg = topk_pair_compare("_selected_paired_scores")(
            actual,
            expected,
            actual_outputs=synth_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if not ok:
            return ok, msg
        expected_paired = paired_of(expected)
        if not torch.allclose(paired, expected_paired, rtol=rtol, atol=atol):
            diff = (paired - expected_paired).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, torch.inf))
            return False, (
                "    selected scores differ from the golden-selected scores "
                f"(max |diff|={diff.max().item():.6g})"
            )
        return True, ""

    def exact_compare(actual, expected, **_kwargs):
        exact = torch.equal(actual.cpu(), expected.cpu())
        return exact, "" if exact else "    integer output differs from golden"

    def score_window_compare(actual, expected, *, inputs, rtol, atol, **_kwargs):
        """Compare only each query's visible window: lanes outside it are
        undefined scratch under the scorer's contract (the top-k clamps
        through its valid shapes), so they carry nothing to check."""

        seg = inputs["seg_start"].long().unsqueeze(1)
        end = seg + inputs["pool_count"].long().clamp(min=0).unsqueeze(1)
        lanes = torch.arange(actual.shape[-1]).unsqueeze(0)
        window = (lanes >= seg) & (lanes < end)
        picked_a = actual.float()[window]
        picked_e = expected.float()[window]
        if picked_a.numel() == 0:
            return True, ""
        bad = ~torch.isfinite(picked_a)
        bad |= (picked_a - picked_e).abs() > (atol + rtol * picked_e.abs())
        ratio = bad.float().mean().item()
        ok = ratio <= 0.01
        return ok, "" if ok else f"    in-window mismatch ratio {ratio:.4f}"

    def topk_indices_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        """Tie-aware position compare for the expanded front-packed rows.

        A near-boundary score cluster can reshuffle two pools whose scores each
        sit inside the scorer's own tolerance; the swap is legal exactly when
        the swapped positions' paired scores agree under the same allclose rule
        that certified ``index_scores``. Tail and padding lanes are deterministic
        given the inputs, so any mismatch there fails outright, and every actual
        row must keep the front-packed ABI.
        """
        a = actual.cpu()
        e = expected.cpu()
        if torch.equal(a, e):
            return True, ""
        scores = actual_outputs["index_scores"].float().cpu()
        seg0 = inputs["seg_start"].long().unsqueeze(1)
        kv_len = inputs["kv_len"].long().unsqueeze(1)
        hist_len = (kv_len // INDEX_KPOOL) * INDEX_KPOOL
        live = (a >= 0).sum(dim=1)
        rows = a.shape[0]
        for row in range(rows):
            if not ((a[row, : live[row]] >= 0).all() and (a[row, live[row] :] == -1).all()):
                return False, f"    row {row} violates the front-packed ABI"

        def lane_scores(x):
            pool = x.clamp(min=0) // INDEX_KPOOL
            cols = (pool + seg0).clamp(0, scores.shape[1] - 1)
            paired = torch.gather(scores, 1, cols)
            is_hist = (x >= 0) & (x < hist_len)
            return torch.where(is_hist, paired, torch.full_like(paired, -torch.inf))

        mismatch = a != e
        both_hist = (a >= 0) & (a < hist_len) & (e >= 0) & (e < hist_len)
        sa, se = lane_scores(a), lane_scores(e)
        tie_legal = (sa - se).abs() <= atol + rtol * torch.maximum(sa.abs(), se.abs())
        illegal = mismatch & ~(both_hist & tie_legal)
        if illegal.any():
            n = int(illegal.sum())
            first = illegal.nonzero()[0].tolist()
            return False, f"    {n} lane(s) differ beyond the score tolerance, first at {first}"
        return True, ""

    step_compare = {
        "index_q": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
        "index_k": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
        "head_weights": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
        "gate_scores": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.01),
        # The raw rows carry the projection rounding of the composed step,
        # unlike the standalone scatter, which moves bytes at zero tolerance.
        "raw_cache": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.01),
        # The pooled table is quantized on write; the softmax transcendentals
        # can flip a rint boundary, so allow a tiny tail of one-quantum flips.
        "pool_cache": ratio_allclose(atol=1.0, rtol=0.0, max_error_ratio=0.005),
        "pool_scale": ratio_allclose(atol=1e-6, rtol=1e-3, max_error_ratio=0.005),
        "pool_valid": exact_compare,
        "index_scores": score_window_compare,
        "selected_pools": selected_pools_compare,
        "selected_valid": exact_compare,
        "topk_indices": topk_indices_compare,
    }

    results = []
    captured = {}

    def capture_topk_indices(actual, expected, **kwargs):
        # Stash the device's own expanded list: the ABI stage replays it as an
        # input to the sparse-attention kernel, the way the serving graph
        # would hand the buffer over.
        captured["topk_indices"] = actual.detach().cpu().clone()
        return topk_indices_compare(actual, expected, **kwargs)

    if args.case in ("step", "all"):
        results.append(
            run(
                fn=decode_indexer_step_test,
                specs=build_decode_indexer_step_specs(args.requests),
                golden_fn=golden_decode_indexer_step_case,
                config={"platform": args.platform, "device_id": args.device},
                rtol=1.0 / 64,
                atol=1e-3,
                compare_fn=step_compare,
                compile_only=args.compile_only,
            )
        )
    if args.case in ("abi", "all"):
        step_specs = build_decode_indexer_step_specs(args.requests)
        kv_len = {spec.name: spec for spec in step_specs}["kv_len"].init_value()
        results.append(
            run(
                fn=decode_indexer_step_test,
                specs=step_specs,
                golden_fn=golden_decode_indexer_step_case,
                config={"platform": args.platform, "device_id": args.device},
                rtol=1.0 / 64,
                atol=1e-3,
                compare_fn={**step_compare, "topk_indices": capture_topk_indices},
                compile_only=args.compile_only,
            )
        )
        if results[-1].passed and not args.compile_only:
            indices = captured["topk_indices"]
            live = (indices >= 0).sum(dim=1)
            for row in range(indices.shape[0]):
                assert (indices[row, : live[row]] >= 0).all() and (indices[row, live[row] :] == -1).all(), (
                    f"row {row} violates the front-packed index ABI"
                )
                assert live[row] == 0 or int(indices[row, : live[row]].max()) < int(kv_len[row]), (
                    f"row {row} selects a position past its causal length"
                )
            results.append(
                run(
                    fn=decode_sparse_attn_test,
                    specs=build_decode_indexer_abi_specs(indices, kv_len),
                    golden_fn=golden_decode_sparse_attn_case,
                    config={"platform": args.platform, "device_id": args.device},
                    rtol=1.0 / 64,
                    atol=1e-3,
                    compare_fn={"output": ratio_allclose(atol=1e-3, rtol=1.0 / 64)},
                    compile_only=args.compile_only,
                )
            )
    for result in results:
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)


__all__ = [
    "build_decode_indexer_abi_specs",
    "build_decode_indexer_step_specs",
    "decode_indexer_step_test",
    "golden_decode_indexer_step_case",
    "indexer_cache_write",
    "indexer_expand",
    "indexer_proj",
    "indexer_score_token",
    "indexer_topk",
    "indexer_pool_write",
]


if __name__ == "__main__":
    main()
