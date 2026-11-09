# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v4_flash_lowlat.prefill_cache import (
    CACHE_LAYOUTS, initialize_cache_pages, prefill_block_tables, prefill_cache_views,
)
from models.deepseek_v4_flash_lowlat.prefill_fwd import prefill_arguments
from models.deepseek_v4_flash_lowlat.prefill_metadata import (
    prefill_tap_positions, prepare_prefill_hidden, prepare_prefill_metadata,
)

def _tables():
    counts = {
        "block_table": 8, "hca_cmp_block_table": 8, "csa_cmp_block_table": 8,
        "idx_block_table": 8, "hca_compress_state_block_table": 128,
        "csa_compress_state_block_table": 256, "csa_inner_compress_state_block_table": 256,
    }
    return {name: torch.arange(count, dtype=torch.int32).reshape(1, 1, -1).repeat(8, 1, 1)
            for name, count in counts.items()}


def _cache_metadata():
    pages = {"kv_cache": 2, "hca_cmp_kv": 8, "csa_cmp_kv": 2, "idx_kv_cache": 2, "idx_kv_scale": 2,
             "hca_compress_state": 17, "csa_compress_state": 33, "csa_inner_compress_state": 33}
    return {name: SimpleNamespace(shape=(8, layers * pages[name], decode_page, *tail), dtype=dtype)
            for name, (layers, decode_page, _, tail, dtype) in CACHE_LAYOUTS.items()}


def test_cache_views_share_storage_and_page_tables_preserve_rows():
    caches = {name: torch.empty((8, layers, decode_page, *tail), dtype=dtype)
              for name, (layers, decode_page, _, tail, dtype) in CACHE_LAYOUTS.items()}
    views = prefill_cache_views(caches)
    for name, value in views.items():
        assert value.untyped_storage().data_ptr() == caches[name].untyped_storage().data_ptr()
        assert value.numel() == caches[name].numel()
    assert views["kv_cache"].shape == (8, 43 * 4, 32, 1, 512)
    assert views["csa_compress_state"].shape == (8, 21 * 2, 2, 2048)
    views["kv_cache"][0, 4 * 5 + 2, 7, 0, 0] = 9
    assert caches["kv_cache"][0, 5, 71, 0, 0] == 9
    tables = _tables()
    tables["block_table"][:, :, 1] = -1
    converted = prefill_block_tables(tables)
    assert converted["ori_block_table"][0, 0, :12].tolist() == [0, 1, 2, 3, -1, -1, -1, -1, 8, 9, 10, 11]
    assert converted["csa_compress_state_block_table"][0, 0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    tables["block_table"][2, 0, 0] = 3
    with pytest.raises(ValueError, match="agree within"):
        prefill_block_tables(tables)


def test_only_owned_cache_pages_are_initialized():
    values = torch.full((8, 40, 8, 1024), 7.0)
    root = SimpleNamespace(shape=values.shape, dtype=values.dtype, worker_ids=tuple(range(8)),
                           shards=[SimpleNamespace(data_ptr=values[rank].data_ptr()) for rank in range(8)])

    class HostCopy:
        def copy_to(self, dst, src, nbytes, *, dst_offset, worker_id):
            ctypes.memmove(dst + dst_offset, src, nbytes)

    initialize_cache_pages(HostCopy(), "hca_compress_state", root, [[1]] + [[] for _ in range(7)])
    for layer in range(20):
        assert bool((values[0, layer * 2] == 7).all())
        assert bool((values[0, layer * 2 + 1, :, :512] == 0).all())
        assert bool((values[0, layer * 2 + 1, :, 512:] == torch.finfo(torch.float32).min).all())
    assert bool((values[1:] == 7).all())


def test_prefill_metadata_matches_page_geometry_and_padding():
    tables = _tables()
    cos = torch.arange(2 * 256 * 64, dtype=torch.float32).reshape(2, 256, 64).bfloat16()
    sin = -cos
    tokens = torch.tensor([3, 9, 2, 5, 7], dtype=torch.int64)
    meta = prepare_prefill_metadata(tokens, start_pos=125, tables=tables, caches=_cache_metadata(), rope_cos=cos, rope_sin=sin,
                                    max_logit_rows=128, terminal=True)
    assert meta["query_start_loc"][0].tolist() == [0, 5]
    assert meta["input_ids"][:4].tolist() == [[3, 9], [2, 5], [7, 0], [0, 0]]
    assert prefill_tap_positions(125, 5).tolist() == [[125, 126], [127, 128], [129, -1], [-1, -1]]
    assert meta["ori_slot_mapping_full"][0].tolist() == [125, 126, 127, 128, 129, -1, -1, -1]
    assert meta["hca_cmp_slot_mapping_full"][0].tolist() == [-1, -1, 0, -1, -1, -1, -1, -1]
    assert meta["csa_cmp_slot_mapping_full"][0].tolist() == [-1, -1, 31, -1, -1, -1, -1, -1]
    assert meta["csa_state_slot_mapping_full"][0, :5].tolist() == [125, 126, 127, 128, 129]
    assert meta["logit_row_indices"][0, 0] == 4 and meta["logit_row_indices"][4, 0] == 4
    assert torch.equal(meta["hca_cmp_freqs_cos"][0, 2], cos[1, 0])
    assert torch.equal(meta["csa_cmp_freqs_cos"][0, 2], cos[1, 124])
    hidden = prepare_prefill_hidden(torch.ones((5, 4096), dtype=torch.bfloat16))
    assert hidden.shape == (8, 8, 4, 4096)
    assert bool((hidden[:, :5] == 1).all()) and bool((hidden[:, 5:] == 0).all())
    short = prepare_prefill_metadata(tokens[:1], start_pos=0, tables=tables, caches=_cache_metadata(),
                                     rope_cos=cos, rope_sin=sin, max_logit_rows=128, terminal=False)
    assert short["hca_cmp_slot_mapping_full"][0].tolist() == [-1, -1, -1, -1]
    assert bool((short["logit_row_indices"] == -1).all())
    tables["block_table"][:, :, 1] = 0
    with pytest.raises(ValueError, match="overwrite its live SWA history"):
        prepare_prefill_metadata(tokens, start_pos=125, tables=tables, caches=_cache_metadata(),
                                 rope_cos=cos, rope_sin=sin, max_logit_rows=128, terminal=True)
    tables["block_table"][:, :, 0] = -1
    with pytest.raises(ValueError, match="no cache page"):
        prepare_prefill_metadata(tokens, start_pos=125, tables=tables, caches=_cache_metadata(), rope_cos=cos, rope_sin=sin,
                                 max_logit_rows=128, terminal=True)


def test_prefill_arguments_follow_the_generated_metadata():
    weights = object()
    prompt = object()
    metadata = {"params": [{"name": "prompt__ssa_v0"}, {"name": "weights__ssa_v0"}]}
    assert prefill_arguments(metadata, {"weights": weights, "prompt": prompt}) == (prompt, weights)
    with pytest.raises(ValueError, match="Missing prefill arguments"):
        prefill_arguments(metadata, {"prompt": prompt})
