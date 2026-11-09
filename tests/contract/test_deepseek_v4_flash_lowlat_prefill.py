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

from models.deepseek_v4_flash_lowlat.prefill_bootstrap_metadata import build_drafter_metadata_banks
from models.deepseek_v4_flash_lowlat.prefill_bootstrap import prepare_bootstrap
from models.deepseek_v4_flash_lowlat.prefill_cache import (
    CACHE_LAYOUTS, initialize_cache_pages, prefill_block_tables, prefill_cache_views,
)
from models.deepseek_v4_flash_lowlat.prefill_metadata import (
    prefill_tap_positions, prepare_prefill_hidden, prepare_prefill_metadata,
)
from models.deepseek_v4_flash_lowlat.prefill_sequential import prefill_with_single
from models.deepseek_v4_flash_lowlat.prefill_state import update_prefill_tail
from models.deepseek_v4_flash_lowlat.prefill_weights import (
    attention_layer_weights, lm_head_weight, moe_layer_weights, prompt_embeddings,
)


def _unpack_nz(value):
    rows, columns = value.shape[-2:]
    c0 = 32 // value.element_size()
    blocked = value.reshape(-1, columns // c0, rows // 16, 16, c0)
    return blocked.permute(0, 2, 3, 1, 4).contiguous().reshape(value.shape)


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


def test_prefill_tail_and_first_decode_state_across_ring_wrap():
    history = torch.zeros((128, 12288), dtype=torch.bfloat16)
    positions = torch.full((128,), -1, dtype=torch.int64)
    for start, count in ((0, 130), (130, 7)):
        local_positions = prefill_tap_positions(start, count)
        features = local_positions.clamp(min=0).to(torch.bfloat16).unsqueeze(-1).expand(-1, -1, 12288)
        update_prefill_tail(history, positions, features, local_positions)
    ring, state = prepare_bootstrap(history, positions, prompt_length=137, first_token=42,
                                    max_new_tokens=2048, epoch=9)
    assert torch.equal(state, torch.tensor([[9, 136, 2047, 42]]).expand(8, 4))
    assert positions[136 % 128] == 136 and ring[136 % 128, 0] == 136
    assert not bool(ring[128:].any())
    positions[9] = -1
    with pytest.raises(ValueError, match="last prompt positions"):
        prepare_bootstrap(history, positions, prompt_length=137, first_token=42, max_new_tokens=2048, epoch=9)


def test_attention_and_lm_head_weight_conversion():
    wqa = torch.arange(32 * 16).reshape(32, 16).bfloat16()
    wqb = torch.arange(16 * 32).reshape(16, 32).to(torch.int8)
    woa = torch.arange(8 * 16 * 16).reshape(8, 16, 16).bfloat16()
    wob = torch.arange(32 * 8 * 16).reshape(32, 8 * 16).to(torch.int8)
    out = attention_layer_weights(wq_a=wqa, wq_b=wqb, wo_a=woa, wo_b=wob, rank=5)
    assert torch.equal(_unpack_nz(out["wq_a"]), wqa)
    assert torch.equal(_unpack_nz(out["wq_b"]), wqb)
    assert torch.equal(out["wo_a"], woa[2:4])
    assert torch.equal(out["wo_b"], wob.reshape(32, 8, 16).permute(1, 0, 2)[2:4])
    shards = [torch.full((16, 16), rank, dtype=torch.bfloat16) for rank in range(8)]
    assert torch.equal(_unpack_nz(lm_head_weight(shards, 6)), torch.cat(shards[4:6]))
    embedded = prompt_embeddings(torch.tensor([0, 17, 127], dtype=torch.int64), shards)
    assert embedded[:, 0].tolist() == [0, 1, 7]


def test_tp_expert_weights_become_ep_and_shared_weights():
    shards = []
    for rank in range(8):
        shards.append({
            "routed_w1": torch.full((9, 16, 32), rank + 1, dtype=torch.int8),
            "routed_w3": torch.full((9, 16, 32), rank + 11, dtype=torch.int8),
            "routed_w2": torch.full((9, 32, 16), rank + 21, dtype=torch.int8),
            "routed_w1_scale": torch.full((9, 16), rank + 0.5),
            "routed_w3_scale": torch.full((9, 16), rank + 1.5),
            "routed_w2_scale": torch.arange(9 * 32, dtype=torch.float32).reshape(9, 32),
        })
    out = moe_layer_weights(shards, 3)
    assert torch.equal(_unpack_nz(out["routed_w1"]), torch.cat([s["routed_w1"][3:4] for s in shards], dim=1))
    assert torch.equal(_unpack_nz(out["shared_w2"]), torch.cat([s["routed_w2"][-1] for s in shards], dim=1))
    assert torch.equal(out["routed_w2_scale"], shards[0]["routed_w2_scale"][3:4])
    shards[7]["routed_w2_scale"][3, 0] += 1
    with pytest.raises(ValueError, match="scales differ"):
        moe_layer_weights(shards, 3)


def test_drafter_banks_use_absolute_anchor_and_physical_pages():
    tables = torch.arange(32768, dtype=torch.int32).reshape(1, 1, 1, -1).expand(4, 3, 1, -1).contiguous()
    rope = torch.arange(40 * 64, dtype=torch.float32).reshape(40, 64).bfloat16()
    banks = build_drafter_metadata_banks(tables, rope, -rope, max_anchor=10)
    assert banks["context_group_position_ids_bank"].shape == (4, 11, 512)
    assert banks["context_group_position_ids_bank"][0, 10, :12].tolist() == list(range(11)) + [-1]
    assert banks["context_group_slot_mapping_bank"][0, 10, 0, :11].tolist() == list(range(11))
    assert banks["query_group_position_ids_bank"][0, 10, :5].tolist() == [11, 12, 13, 14, 15]
    assert torch.equal(banks["query_freqs_cos_bank"][0, 10, 0], rope[11])


def test_single_weight_prefill_builds_the_last_128_rows_and_state():
    inputs = {
        "position_ids": torch.zeros((8, 1), dtype=torch.int32).share_memory_(),
        "kv_seq_lens": torch.zeros((8, 1), dtype=torch.int32).share_memory_(),
        "input_ids": torch.zeros((8, 1), dtype=torch.int64).share_memory_(),
        "sampling_positions": torch.zeros((8, 1), dtype=torch.int32).share_memory_(),
        "logit_row_indices": torch.zeros((8, 1), dtype=torch.int32).share_memory_(),
        "num_tokens_per_owner": torch.zeros(8, dtype=torch.int32).share_memory_(),
    }
    feature = torch.empty((1, 12288), dtype=torch.bfloat16)
    sample = torch.empty((1, 8), dtype=torch.int32)
    feature_shard = SimpleNamespace(data_ptr=feature.data_ptr())
    sample_shard = SimpleNamespace(data_ptr=sample.data_ptr())
    inputs["dspark_target_hidden_out"] = SimpleNamespace(
        shape=(8, 1, 12288), dtype=torch.bfloat16, worker_ids=tuple(range(8)), shards=[feature_shard] * 8,
    )
    inputs["sampled_ids"] = SimpleNamespace(
        shape=(8, 1, 8), dtype=torch.int32, worker_ids=tuple(range(8)), shards=[sample_shard] * 8,
    )
    metadata = {"params": [{"name": name + "__ssa_v0"} for name in inputs]}

    class Worker:
        def __init__(self):
            self.seen = []
            self.reads = 0

        def submit(self, program, *args, config):
            position = len(self.seen)
            assert program == "single" and len(args) == len(inputs) and config is None
            assert bool((inputs["position_ids"] == position).all())
            assert bool((inputs["kv_seq_lens"] == position + 1).all())
            assert bool((inputs["input_ids"] == position).all())
            self.seen.append(position)
            feature.fill_(position)
            sample.fill_(position + 1)
            return SimpleNamespace(result=lambda: None)

        def copy_from(self, dst, src, size, *, worker_id):
            assert worker_id == 0
            self.reads += 1
            ctypes.memmove(dst, src, size)

    worker = Worker()
    first, ring, state = prefill_with_single(worker, "single", metadata, inputs, torch.arange(130),
                                            max_new_tokens=32, epoch=9)
    assert worker.seen == list(range(130))
    assert worker.reads == 129
    assert first == 130 and state[0].tolist() == [9, 129, 31, 130]
    assert torch.equal(state, state[:1].expand(8, -1))
    assert ring[0, 0] == 128 and ring[1, 0] == 129 and ring[2, 0] == 2
    assert not bool(ring[128:].any())
