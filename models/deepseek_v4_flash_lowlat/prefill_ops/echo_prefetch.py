# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded coarse-score histogram, exact-order KV completion and chunk EMA."""

import pypto.language as pl
from causal_cache_ready import CAUSAL_READY
import pypto.language.distributed as pld

from config import BLOCK_SIZE, FLASH as M
from prefill_cp_token_allgather import _parse_int_argv

# Model configuration.
ECHO_TOPK = M.index_topk
ECHO_HEAD_DIM = M.head_dim
ECHO_MAX_CANDIDATES = M.max_position_embeddings // 4
ECHO_RANKS = _parse_int_argv("--tp", 2)
ECHO_LAYERS = 21
ECHO_STATE_ROWS = ECHO_LAYERS * (ECHO_RANKS + 1)
ECHO_SIGNAL_COLS = ECHO_RANKS * 16
ECHO_PREFETCH = _parse_int_argv("--prefill-echo-prefetch", 0)
ECHO_DIAGNOSTICS = _parse_int_argv("--prefill-echo-diagnostics", 0)

# Tiling.
ECHO_QUERY_TILE = 128
ECHO_SCORE_TILE = 512
ECHO_BIN_TILE = 16


@pl.jit.inline

def _echo_ordered_bins(value: pl.Tile[[1, ECHO_SCORE_TILE], pl.FP32], shift: pl.Scalar[pl.FP32]):
    shifted = pl.sub(value, shift)
    halfwords = pl.reinterpret_view(shifted, pl.INT16)
    high_word = pl.tile.gather_mask(halfwords, mask_pattern=pl.tile.MaskPattern.P1010)
    magnitude_word = pl.tile.ands(high_word, 32767)
    high_byte = pl.tile.shrs(magnitude_word, 8)
    magnitude = pl.cast(high_byte, target_type=pl.FP32)
    positive = pl.add(magnitude, 128.0)
    negative = pl.add(pl.neg(magnitude), 127.0)
    is_negative = pl.tile.cmps(shifted, 0.0, cmp_type=2)
    ordered = pl.tile.select(is_negative, negative, positive)
    is_zero = pl.tile.cmps(shifted, 0.0, cmp_type=0)
    canonical = pl.tile.select(is_zero, 128.0, ordered)
    is_number = pl.tile.cmp(value, value, cmp_type=0)
    return pl.tile.select(is_number, canonical, -1.0)


@pl.jit.inline(auto_scope=False)
def echo_plan_from_histogram(
    scores: pl.Tensor,
    position_ids: pl.Tensor,
    request_ids: pl.Tensor,
    estimate: pl.Tensor,
    histogram: pl.Tensor,
    indices: pl.Tensor,
    logical_map: pl.Tensor,
    counts: pl.Tensor,
    score_ready: pl.Scalar[pl.TASK_ID],
    slot_retired: pl.Scalar[pl.TASK_ID],
    query_base: pl.Scalar[pl.INDEX],
    query_rows: pl.Scalar[pl.INDEX],
    epoch: pl.Scalar[pl.INT32],
):
    with pl.spmd(query_rows, name_hint="echo_complete_bins_plan", deps=[score_ready, slot_retired]) as planned:
        query = pl.tile.get_block_idx()
        output_query = query_base + query
        request = pl.read(request_ids, [output_query])
        position = pl.read(position_ids, [output_query])
        visible = 0
        if request == 0:
            visible = pl.max(pl.min((position + 1) // 4, ECHO_MAX_CANDIDATES), 0)
        pl.write(counts, [query, 0], epoch)
        pl.write(counts, [query, 1], pl.const(0, pl.INT32))
        pl.write(counts, [query, 2], pl.const(0, pl.INT32))
        pl.write(counts, [query, 3], pl.const(0, pl.INT32))
        pl.write(counts, [query, 4], pl.const(0, pl.INT32))
        pl.write(counts, [query, 5], pl.const(0, pl.INT32))
        pl.write(counts, [query, 6], pl.cast(visible, pl.INT32))
        pl.write(counts, [query, 7], pl.cast(output_query, pl.INT32))
        estimate_valid = pl.read(estimate, [1])
        if visible > ECHO_TOPK and estimate_valid > 0.0:
            bins = pl.load(histogram, [query, 0], [1, 256])
            cumulative = pl.const(0, pl.INT32)
            threshold = pl.const(256, pl.INT32)
            safe_count = pl.const(0, pl.INT32)
            for descending in pl.range(256):
                bin_id = 255 - descending
                bin_count = pl.cast(pl.tile.read(bins, [0, bin_id]), pl.INT32)
                cumulative = cumulative + bin_count
                if cumulative <= ECHO_TOPK:
                    threshold = pl.cast(bin_id, pl.INT32)
                    safe_count = cumulative
            # A NaN query uses the exact gather fallback without a partial plan.
            if cumulative != visible:
                threshold = pl.const(256, pl.INT32)
                safe_count = pl.const(0, pl.INT32)
                pl.write(counts, [query, 5], pl.const(1, pl.INT32))
            if safe_count > 0:
                written = pl.const(0, pl.INT32)
                shift = pl.read(estimate, [0])
                scratch = pl.tile.create([1, 8192], dtype=pl.UINT8)
                for source_base in pl.range(0, visible, ECHO_SCORE_TILE):
                    valid = pl.min(ECHO_SCORE_TILE, visible - source_base)
                    empty = pl.tile.full([1, ECHO_SCORE_TILE], dtype=pl.INT32, value=-1)
                    pl.store(empty, [query, source_base], logical_map)
                    plan_source = pl.load(scores, [query, source_base], [1, ECHO_SCORE_TILE], valid_shape=[1, valid])
                    plan_padded = pl.tile.fillpad(plan_source, pad_value=pl.PadValue.min)
                    plan_full = pl.set_validshape(plan_padded, 1, ECHO_SCORE_TILE)
                    keys = _echo_ordered_bins(plan_full, shift)
                    valid_keys = pl.set_validshape(keys, 1, valid)
                    padded = pl.tile.fillpad(valid_keys, pad_value=pl.PadValue.min)
                    key_fp32 = padded
                    threshold_fp32 = pl.cast(threshold, pl.FP32) - 0.5
                    chosen, number = pl.tile.gather_compare(key_fp32, threshold_fp32, scratch, cmp_mode="gt", out_cols=ECHO_SCORE_TILE)
                    chunk_count = pl.tile.read(number, [0, 0])
                    for entry in pl.range(pl.cast(chunk_count, pl.INDEX)):
                        slot = pl.cast(written, pl.INDEX) + entry
                        if slot < ECHO_TOPK:
                            offset = pl.tile.read(chosen, [0, entry])
                            logical = pl.cast(source_base, pl.INT32) + offset
                            pl.write(indices, [query, slot], logical)
                        else:
                            pl.write(counts, [query, 5], pl.const(2, pl.INT32))
                    written = written + chunk_count
                pl.write(counts, [query, 1], written)
    return planned




@pl.jit.inline(auto_scope=False)
def echo_coarse_plan(
    scores: pl.Tensor,
    position_ids: pl.Tensor,
    request_ids: pl.Tensor,
    estimate: pl.Tensor,
    histogram: pl.Tensor,
    indices: pl.Tensor,
    logical_map: pl.Tensor,
    counts: pl.Tensor,
    score_ready: pl.Scalar[pl.TASK_ID],
    slot_retired: pl.Scalar[pl.TASK_ID],
    query_base: pl.Scalar[pl.INDEX],
    query_rows: pl.Scalar[pl.INDEX],
    epoch: pl.Scalar[pl.INT32],
):
    with pl.spmd(query_rows, name_hint="echo_histogram256", deps=[score_ready, slot_retired]) as histogram_ready:
        query = pl.tile.get_block_idx()
        output_query = query_base + query
        request = pl.read(request_ids, [output_query])
        position = pl.read(position_ids, [output_query])
        visible = 0
        if request == 0:
            visible = pl.max(pl.min((position + 1) // 4, ECHO_MAX_CANDIDATES), 0)
        shift = pl.read(estimate, [0])
        estimate_valid = pl.read(estimate, [1])
        if visible > ECHO_TOPK and estimate_valid > 0.0:
            totals = pl.tile.full([1, 256], dtype=pl.FP32, value=0.0)
            for source_base in pl.range(0, visible, ECHO_SCORE_TILE):
                valid = pl.min(ECHO_SCORE_TILE, visible - source_base)
                hist_source = pl.load(scores, [query, source_base], [1, ECHO_SCORE_TILE], valid_shape=[1, valid])
                hist_padded = pl.tile.fillpad(hist_source, pad_value=pl.PadValue.min)
                hist_full = pl.set_validshape(hist_padded, 1, ECHO_SCORE_TILE)
                values = _echo_ordered_bins(hist_full, shift)
                fp32 = values
                for bin_base in pl.range(0, 256, ECHO_BIN_TILE):
                    bin_zero = pl.tile.arange(0, [1, ECHO_BIN_TILE], dtype=pl.INT32)
                    bin_ramp = pl.add(bin_zero, pl.cast(bin_base, pl.INT32))
                    bin_ids = pl.reshape(bin_ramp, [ECHO_BIN_TILE, 1])
                    bin_fp32 = pl.cast(bin_ids, target_type=pl.FP32)
                    zeros = pl.tile.full([ECHO_BIN_TILE, ECHO_SCORE_TILE], dtype=pl.FP32, value=0.0)
                    expanded = pl.col_expand_add(zeros, fp32)
                    difference = pl.row_expand_sub(expanded, bin_fp32)
                    absolute = pl.maximum(difference, pl.neg(difference))
                    match = pl.add(pl.neg(pl.minimum(absolute, 1.0)), 1.0)
                    valid_match = pl.set_validshape(match, ECHO_BIN_TILE, valid)
                    padded_match = pl.fillpad(valid_match, pad_value=pl.PadValue.zero)
                    sum_scratch = pl.tile.create([ECHO_BIN_TILE, ECHO_SCORE_TILE], dtype=pl.FP32)
                    tile_count = pl.reshape(pl.row_sum(padded_match, tmp_tile=sum_scratch), [1, ECHO_BIN_TILE])
                    prior = pl.tile.slice(totals, [1, ECHO_BIN_TILE], [0, bin_base])
                    updated = pl.add(prior, tile_count)
                    totals = pl.tile.assemble(totals, updated, [0, bin_base])
            pl.store(totals, [query, 0], histogram)

    planned = echo_plan_from_histogram(scores, position_ids, request_ids, estimate, histogram, indices, logical_map, counts, histogram_ready, slot_retired, query_base, query_rows, epoch)
    return planned



@pl.jit.inline(auto_scope=False)
def echo_prefetch_kv(
    indices: pl.Tensor,
    logical_map: pl.Tensor,
    counts: pl.Tensor,
    prefetched: pl.Tensor,
    cmp_kv: pl.Tensor,
    cmp_table: pl.Tensor,
    request_ids: pl.Tensor,
    query_base: pl.Scalar[pl.INDEX],
    query_rows: pl.Scalar[pl.INDEX],
    plan_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cmp_kv, 0)
    cache = pl.reshape(cmp_kv, [blocks * BLOCK_SIZE, ECHO_HEAD_DIM])
    causal_ready_choices = pl.array.create(1, pl.TASK_ID)
    if CAUSAL_READY:
        if False:
            causal_specialize_echo_prefetch_paged_kv_prefetched = pl.create_tensor([ECHO_QUERY_TILE * ECHO_TOPK, ECHO_HEAD_DIM], dtype=pl.BF16)
            causal_specialize_echo_prefetch_paged_kv_logical_map = pl.create_tensor([ECHO_QUERY_TILE, ECHO_MAX_CANDIDATES], dtype=pl.INT32)
            causal_specialize_echo_prefetch_paged_kv_counts = pl.create_tensor([ECHO_QUERY_TILE, 16], dtype=pl.INT32)
            _causal_reader_echo_prefetch_paged_kv(request_ids, query_base, causal_specialize_echo_prefetch_paged_kv_counts, indices, cmp_table, blocks, cache, causal_specialize_echo_prefetch_paged_kv_prefetched, causal_specialize_echo_prefetch_paged_kv_logical_map)
        (prefetched, logical_map, counts), ready_causal = pl.spmd_submit(
            self._causal_reader_echo_prefetch_paged_kv,  # noqa: F821 - materialized by @pl.jit
            request_ids, query_base, counts, indices, cmp_table, blocks, pl.no_dep(cache), prefetched, logical_map,
            core_num=query_rows, deps=[plan_ready],
        )
        causal_ready_choices[0] = ready_causal
    else:
        with pl.spmd(query_rows, name_hint="echo_prefetch_paged_kv", deps=[plan_ready]) as ready:
            _causal_echo_prefetch_paged_kv_body(request_ids, query_base, counts, indices, cmp_table, blocks, cache, prefetched, logical_map)
        causal_ready_choices[0] = ready
    ready = causal_ready_choices[0]
    return ready


@pl.jit.inline(auto_scope=False)
def echo_load_estimate(
    state: pld.DistributedTensor[[ECHO_STATE_ROWS, 16], pl.FP32],
    estimate: pl.Tensor,
    diagnostics: pl.Tensor,
    layer: pl.Scalar[pl.INT32],
    first_chunk: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    prior: pl.Scalar[pl.TASK_ID],
):
    diagnostic_row = layer * 2 + epoch % 2
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="echo_chunk_estimate", deps=[prior]) as ready:
        shift = pl.const(0.0, pl.FP32)
        valid = pl.const(0.0, pl.FP32)
        row = layer * (ECHO_RANKS + 1)
        if first_chunk == 0:
            saved_epoch = pl.read(state, [row, 3])
            if saved_epoch == pl.cast(epoch, pl.FP32):
                valid = pl.read(state, [row, 5])
                if valid > 0.0:
                    shift = pl.read(state, [row, 4])
            else:
                valid = pl.read(state, [row, 1])
                if valid > 0.0:
                    shift = pl.read(state, [row, 0])
        pl.write(estimate, [0], shift)
        pl.write(estimate, [1], valid)
        zero = pl.tile.full([1, 16], dtype=pl.FP32, value=0.0)
        pl.tile.write(zero, [0, 4], shift)
        pl.store(zero, [diagnostic_row, 0], diagnostics)
    return ready


@pl.jit.inline(auto_scope=False)
def echo_record_exact_end(
    pairs: pl.Tensor,
    position_ids: pl.Tensor,
    request_ids: pl.Tensor,
    record: pl.Tensor,
    query_base: pl.Scalar[pl.INDEX],
    query_rows: pl.Scalar[pl.INDEX],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    groups_per_query: pl.Scalar[pl.INDEX],
    selected: pl.Scalar[pl.TASK_ID],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="echo_exact_ending_score", deps=[selected]) as recorded:
        local_rows = pl.tensor.dim(position_ids, 0)
        physical_last = pl.const(-1.0, pl.FP32)
        kth = pl.const(0.0, pl.FP32)
        valid = pl.const(0.0, pl.FP32)
        found = pl.const(0, pl.INT32)
        for reverse in pl.range(query_rows):
            if found == 0:
                query = query_rows - 1 - reverse
                source = query_base + query
                request = pl.read(request_ids, [source])
                if request == 0:
                    found = pl.const(1, pl.INT32)
                    position = pl.read(position_ids, [source])
                    physical_row = pl.cast(pl.cast(rank, pl.INDEX) * local_rows + source, pl.INT32)
                    physical_last = pl.cast(physical_row, pl.FP32)
                    valid = pl.const(0.0, pl.FP32)
                    if (position + 1) // 4 >= ECHO_TOPK:
                        kth = pl.read(pairs, [query * groups_per_query, 2 * (ECHO_TOPK - 1)])
                        valid = pl.const(1.0, pl.FP32)
        pl.write(record, [0, 0], physical_last)
        pl.write(record, [0, 1], kth)
        pl.write(record, [0, 2], valid)
        pl.write(record, [0, 3], pl.cast(epoch, pl.FP32))
    return recorded


@pl.jit.inline(auto_scope=False)
def echo_commit_chunk(
    state: pld.DistributedTensor[[ECHO_STATE_ROWS, 16], pl.FP32],
    signal: pld.DistributedTensor[[ECHO_LAYERS, ECHO_SIGNAL_COLS], pl.INT32],
    record: pl.Tensor,
    estimate: pl.Tensor,
    diagnostics: pl.Tensor,
    layer: pl.Scalar[pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    first_chunk: pl.Scalar[pl.INT32],
    record_ready: pl.Scalar[pl.TASK_ID],
):
    diagnostic_row = layer * 2 + epoch % 2
    state_row = layer * (ECHO_RANKS + 1)
    published = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.spmd(ECHO_RANKS, name_hint="echo_publish_exact_end", deps=[record_ready]) as sent:
            peer = pl.tile.get_block_idx()
            destination = state_row + 1 + rank
            pld.tensor.put(dst=state, peer=peer, src=record, dst_offsets=[destination, 0], src_offsets=[0, 0], shape=[1, 16])
            pld.system.notify(signal, peer, [layer, rank * 16], value=epoch, op=pld.NotifyOp.Set)
        published[0] = sent
    waited = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="echo_wait_exact_end", deps=[published[0]]) as received:
            for source in pl.range(ECHO_RANKS):
                pld.system.defer_wait(signal, [layer, source * 16], epoch, cmp=pld.WaitCmp.Ge)
        waited[0] = received
    completed = pl.array.create(1, pl.TASK_ID)
    with pl.manual_scope():
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="echo_global_last_ema", deps=[waited[0]]) as updated:
            last = pl.const(-1.0, pl.FP32)
            exact = pl.const(0.0, pl.FP32)
            valid = pl.const(0.0, pl.FP32)
            bad_epoch = pl.const(0, pl.INT32)
            for source in pl.range(ECHO_RANKS):
                row = state_row + 1 + source
                position = pl.read(state, [row, 0])
                generation = pl.read(state, [row, 3])
                if generation != pl.cast(epoch, pl.FP32):
                    bad_epoch = pl.const(1, pl.INT32)
                if generation == pl.cast(epoch, pl.FP32) and position > last:
                    last = position
                    exact = pl.read(state, [row, 1])
                    valid = pl.read(state, [row, 2])
            predicted = exact
            old_valid = pl.read(estimate, [1])
            if first_chunk == 0 and old_valid > 0.0 and valid > 0.0:
                previous = pl.read(estimate, [0])
                predicted = previous * 0.5 + exact * 0.5
            finite = predicted <= 3.4028234663852886e38 and predicted >= -3.4028234663852886e38
            if not finite:
                valid = pl.const(0.0, pl.FP32)
            pl.write(state, [state_row, 0], predicted)
            pl.write(state, [state_row, 1], valid)
            pl.write(state, [state_row, 2], exact)
            pl.write(state, [state_row, 3], pl.cast(epoch, pl.FP32))
            pl.write(state, [state_row, 4], pl.read(estimate, [0]))
            pl.write(state, [state_row, 5], old_valid)
            pl.write(diagnostics, [diagnostic_row, 0], predicted)
            pl.write(diagnostics, [diagnostic_row, 1], valid)
            pl.write(diagnostics, [diagnostic_row, 2], exact)
            pl.write(diagnostics, [diagnostic_row, 3], pl.cast(epoch, pl.FP32))
            pl.write(diagnostics, [diagnostic_row, 15], pl.cast(bad_epoch, pl.FP32))
            # All publishers have delivered to this receiver. Chunks are
            # globally joined before reusing the domain on the next submit.
            for source in pl.range(ECHO_RANKS):
                pl.write(signal, [layer, source * 16], pl.const(0, pl.INT32))
        completed[0] = updated
    return completed[0]


@pl.jit.inline(auto_scope=False)
def echo_account_query(
    counts: pl.Tensor,
    diagnostics: pl.Tensor,
    layer: pl.Scalar[pl.INT32],
    query_rows: pl.Scalar[pl.INDEX],
    retired: pl.Scalar[pl.TASK_ID],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="echo_account_query", deps=[retired]) as accounted:
        planned = pl.const(0, pl.INT32)
        fetched = pl.const(0, pl.INT32)
        hits = pl.const(0, pl.INT32)
        misses = pl.const(0, pl.INT32)
        no_plan = pl.const(0, pl.INT32)
        nonfinite = pl.const(0, pl.INT32)
        overflow = pl.const(0, pl.INT32)
        all_visible = pl.const(0, pl.INT32)
        for query in pl.range(query_rows):
            planned = planned + pl.read(counts, [query, 1])
            fetched = fetched + pl.read(counts, [query, 2])
            hits = hits + pl.read(counts, [query, 3])
            misses = misses + pl.read(counts, [query, 4])
            if pl.read(counts, [query, 1]) == 0:
                no_plan = no_plan + pl.const(1, pl.INT32)
            status = pl.read(counts, [query, 5])
            if status == 1:
                nonfinite = nonfinite + pl.const(1, pl.INT32)
            if status == 2:
                overflow = overflow + pl.const(1, pl.INT32)
            if pl.read(counts, [query, 6]) <= ECHO_TOPK:
                all_visible = all_visible + pl.const(1, pl.INT32)
        pl.write(diagnostics, [layer, 5], pl.read(diagnostics, [layer, 5]) + pl.cast(planned, pl.FP32))
        pl.write(diagnostics, [layer, 6], pl.read(diagnostics, [layer, 6]) + pl.cast(fetched, pl.FP32))
        pl.write(diagnostics, [layer, 7], pl.read(diagnostics, [layer, 7]) + pl.cast(hits, pl.FP32))
        pl.write(diagnostics, [layer, 8], pl.read(diagnostics, [layer, 8]) + pl.cast(misses, pl.FP32))
        pl.write(diagnostics, [layer, 9], pl.read(diagnostics, [layer, 9]) + pl.cast(no_plan, pl.FP32))
        pl.write(diagnostics, [layer, 10], pl.read(diagnostics, [layer, 10]) + pl.cast(nonfinite, pl.FP32))
        pl.write(diagnostics, [layer, 11], pl.read(diagnostics, [layer, 11]) + pl.cast(overflow, pl.FP32))
        pl.write(diagnostics, [layer, 12], pl.read(diagnostics, [layer, 12]) + pl.cast(pl.cast(query_rows, pl.INT32), pl.FP32))
        pl.write(diagnostics, [layer, 13], pl.read(diagnostics, [layer, 13]) + pl.cast(all_visible, pl.FP32))
    return accounted


@pl.jit.inline(auto_scope=False)
def _causal_echo_prefetch_paged_kv_body(
    request_ids: pl.Tensor,
    query_base: pl.Scalar[pl.INDEX],
    counts: pl.Tensor,
    indices: pl.Tensor,
    cmp_table: pl.Tensor,
    blocks: pl.Scalar[pl.INDEX],
    cache: pl.Tensor,
    prefetched: pl.Tensor,
    logical_map: pl.Tensor,
):
    """Compute the unchanged echo_prefetch_paged_kv reader body."""
    query = pl.tile.get_block_idx()
    request = pl.read(request_ids, [query_base + query])
    selected = pl.read(counts, [query, 1])
    fetched = pl.const(0, pl.INT32)
    if selected <= ECHO_TOPK and request >= 0:
        for slot in pl.range(pl.cast(selected, pl.INDEX)):
            logical = pl.read(indices, [query, slot])
            page = logical // BLOCK_SIZE
            block = pl.read(cmp_table, [request, page])
            if block >= 0 and block < blocks:
                source = pl.cast(block, pl.INDEX) * BLOCK_SIZE + logical % BLOCK_SIZE
                destination = query * ECHO_TOPK + slot
                value = cache[source:source + 1, :]
                prefetched[destination:destination + 1, :] = value
                pl.write(logical_map, [query, logical], pl.cast(slot, pl.INT32))
                fetched = fetched + pl.const(1, pl.INT32)
    pl.write(counts, [query, 2], fetched)


@pl.jit.incore
def _causal_reader_echo_prefetch_paged_kv(
    request_ids: pl.Tensor,
    query_base: pl.Scalar[pl.INDEX],
    counts: pl.InOut[pl.Tensor[[ECHO_QUERY_TILE, 16], pl.INT32]],
    indices: pl.Tensor,
    cmp_table: pl.Tensor,
    blocks: pl.Scalar[pl.INDEX],
    cache: pl.Tensor,
    prefetched: pl.InOut[pl.Tensor[[ECHO_QUERY_TILE * ECHO_TOPK, ECHO_HEAD_DIM], pl.BF16]],
    logical_map: pl.InOut[pl.Tensor[[ECHO_QUERY_TILE, ECHO_MAX_CANDIDATES], pl.INT32]],
):
    """Read persistent caches with explicit NoDep directions."""
    _causal_echo_prefetch_paged_kv_body(request_ids, query_base, counts, indices, cmp_table, blocks, cache, prefetched, logical_map)
    return prefetched, logical_map, counts
