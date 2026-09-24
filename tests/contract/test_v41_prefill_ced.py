# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CED prefill contracts at the encoder-to-decoder cache handoff."""

import json
import struct
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v4_1_flash.config import FLASH
from models.deepseek_v4_1_flash.metadata import build_forward_metadata, paged_slots
from models.deepseek_v4_1_flash.prefill_checkpoint_weights import PrefillCheckpoint
from models.deepseek_v4_1_flash.quantization import unpack_mx_b_scale
from models.deepseek_v4_1_flash.prefill_decoder import (
    CachedEncoderTail,
    run_prefill_decoder,
    select_decoder_replay,
)
from models.deepseek_v4_1_flash.prefill_encoder import (
    CachedEncoderInputTail,
    PrefillState,
    run_prefill_encoder,
    run_prefill_encoder_replay,
)
from models.deepseek_v4_1_flash.prefill_layer_plan import (
    load_prefill_attention,
    load_prefill_attention_host,
    load_prefill_ffn_host,
    load_prefill_global_publisher,
    resolve_prefill_layer_plan,
)


def test_ced_plan_selects_all_checkpoint_modes_and_seeded_decoder_full(monkeypatch):
    plans = [resolve_prefill_layer_plan(layer_id) for layer_id in range(40)]
    assert Counter((plan.stage, plan.attention_symbol) for plan in plans) == {
        ("encoder", "prefill_swa"): 2,
        ("encoder", "prefill_c2a_full"): 3,
        ("encoder", "prefill_c2a_reuse"): 15,
        ("decoder", "prefill_c1a_seeded"): 1,
        ("decoder", "prefill_c1a_reindex"): 4,
        ("decoder", "prefill_c1a_reuse"): 15,
    }
    assert [plan.layer_id for plan in plans if plan.global_publisher_symbol] == [20]
    from models.deepseek_v4_1_flash import prefill_layer_plan

    sentinel = object()
    loaded = []

    def import_stub(name):
        loaded.append(name)
        return SimpleNamespace(
            prefill_c1a_seeded=sentinel,
            l3_prefill_c1a_seeded=sentinel,
            make_ffn_program=sentinel,
            publish_decoder_global_from_encoder_rank=sentinel,
        )

    monkeypatch.setattr(prefill_layer_plan, "import_module", import_stub)
    assert load_prefill_attention(plans[20]) is sentinel
    assert load_prefill_attention_host(plans[20]) is sentinel
    assert load_prefill_ffn_host() is sentinel
    assert load_prefill_global_publisher(plans[20]) is sentinel
    assert load_prefill_global_publisher(plans[21]) is None
    assert loaded == [
        "models.deepseek_v4_1_flash.prefill_c1a_full",
        "models.deepseek_v4_1_flash.prefill_c1a_full",
        "models.deepseek_v4_1_flash.prefill_layer",
        "models.deepseek_v4_1_flash.prefill_decoder_kv",
    ]


def metadata_for(lengths, prefixes=None):
    """Allocate independent physical pages for each request and cache family."""
    prefixes = prefixes or [0] * len(lengths)
    starts = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    old = torch.tensor(prefixes, dtype=torch.int32)
    pages = 4
    physical = torch.arange(len(lengths) * pages, dtype=torch.int32).reshape(len(lengths), pages)
    window_table = physical + 100
    compressed_tables = {source: physical + (0 if source < 20 else 200) for source in (2, 8, 14, 20)}
    state_tables = {
        source: torch.arange(len(lengths), dtype=torch.int32).unsqueeze(1) + source * 100
        for source in (2, 8, 14)
    }
    metadata = build_forward_metadata(starts, old, window_table, compressed_tables, state_tables)
    return metadata, window_table


def test_decoder_replay_keeps_request_boundaries_and_truncates_swa():
    lengths = (0, 1, 127, 128, 129)
    metadata, table = metadata_for(lengths)
    replay = select_decoder_replay(metadata, table)
    assert replay.query_lens.tolist() == [0, 1, 127, 128, 128]
    assert replay.replay_starts.tolist() == [0, 0, 0, 0, 1]
    assert replay.logit_row_indices.tolist() == [-1, 0, 127, 255, 383]
    assert replay.source_rows.tolist()[-128:] == list(range(257, 385))
    for request, count in enumerate(replay.query_lens.tolist()):
        first = int(replay.query_start_loc[request])
        for row in range(first, first + count):
            position = int(replay.position_ids[row])
            visible = list(range(int(replay.replay_starts[request]), position + 1))[-FLASH.sliding_window:]
            expected = paged_slots(
                torch.tensor(visible, dtype=torch.int32),
                torch.full((len(visible),), request, dtype=torch.int32),
                table,
            )
            assert replay.window_lens[row] == len(visible)
            assert torch.equal(replay.window_indices[row, :len(visible)], expected.to(torch.int32))
            assert bool((replay.window_indices[row, len(visible):] == -1).all())


def test_decoder_replay_rejects_missing_cached_encoder_rows():
    metadata, table = metadata_for((24,), (200,))
    with pytest.raises(ValueError, match="needs encoder replay rows"):
        select_decoder_replay(metadata, table)


def test_cached_encoder_tails_complete_mixed_request_replay():
    metadata, table = metadata_for((2, 24), (5, 200))
    cached_positions = torch.tensor([*range(5), *range(96, 200)], dtype=torch.float32)
    new_positions = metadata.position_ids.to(torch.float32)

    def state_with_positions(positions):
        hidden = torch.zeros((positions.numel(), FLASH.hc_mult, FLASH.hidden_size))
        hidden[:, 0, 0] = positions
        mix = torch.zeros((positions.numel(), FLASH.hc_mult))
        mix[:, 0] = positions
        return PrefillState(hidden, mix)

    cached = CachedEncoderTail(
        state_with_positions(cached_positions),
        torch.tensor([0, 5, 109], dtype=torch.int32),
    )
    replay = select_decoder_replay(metadata, table, cached_prefix=cached)
    assert replay.query_lens.tolist() == [7, 128]
    assert replay.replay_starts.tolist() == [0, 96]
    assert replay.position_ids.tolist() == [*range(7), *range(96, 224)]
    assert replay.compressed_lens.tolist() == [*range(1, 8), *range(97, 225)]
    assert replay.source_rows.tolist() == [*range(5), 109, 110, *range(5, 109), *range(111, 135)]
    events = []

    def publish_global(layer, layer_state, full_metadata):
        assert layer.layer_id == 20 and layer_state.num_tokens == 26
        assert full_metadata is metadata
        events.append("publish")

    def decoder_layer(layer, layer_state, replay_info, full_metadata):
        assert layer_state.num_tokens == 135
        assert torch.equal(layer_state.x_hc[:, 0, 0], replay_info.position_ids.to(torch.float32))
        assert torch.equal(layer_state.pre_mix[:, 0], replay_info.position_ids.to(torch.float32))
        assert full_metadata is metadata
        events.append(layer.layer_id)
        return layer_state

    result = run_prefill_decoder(
        state_with_positions(new_positions), metadata, table,
        publish_global, decoder_layer, cached_prefix=cached,
    )
    assert events == ["publish", *range(20, 40)]
    assert result.replay.logit_row_indices.tolist() == [6, 134]


def test_encoder_prefix_replay_truncates_swa_and_marks_only_new_global_rows():
    metadata, table = metadata_for((2, 24), (5, 200))
    cached_positions = torch.tensor([*range(5), *range(72, 200)], dtype=torch.float32)

    def state_with_positions(positions):
        hidden = torch.zeros((positions.numel(), FLASH.hc_mult, FLASH.hidden_size))
        hidden[:, 0, 0] = positions
        mix = torch.zeros((positions.numel(), FLASH.hc_mult))
        mix[:, 0] = positions
        return PrefillState(hidden, mix)

    cached = CachedEncoderInputTail(
        state_with_positions(cached_positions), torch.tensor([0, 5, 133], dtype=torch.int32)
    )
    seen_layers = []

    def replay_layer(layer, state, replay, new_metadata):
        assert new_metadata is metadata
        assert state.num_tokens == 159
        assert replay.replay_starts.tolist() == [0, 72]
        assert replay.position_ids.tolist() == [*range(7), *range(72, 224)]
        assert replay.position_ids[replay.is_new].tolist() == [5, 6, *range(200, 224)]
        assert replay.window_lens[7] == 1
        assert replay.window_lens[7 + 127] == 128
        assert replay.window_lens[7 + 128] == 128
        assert int(replay.window_indices[7, 0]) == int(paged_slots(
            torch.tensor([72], dtype=torch.int32), torch.tensor([1], dtype=torch.int32), table
        )[0])
        seen_layers.append(layer.layer_id)
        updated = state.x_hc.clone()
        updated[:, 0, 0] += 1
        return PrefillState(updated, state.pre_mix)

    result = run_prefill_encoder_replay(
        state_with_positions(metadata.position_ids.float()), metadata, table, cached, replay_layer
    )
    assert seen_layers == list(range(20))
    assert result.new_state.num_tokens == 26
    assert result.new_state.x_hc[:, 0, 0].tolist() == [25, 26, *range(220, 244)]
    assert result.cached_query_start_loc.tolist() == [0, 5, 133]
    assert result.cached_state.num_tokens == 133
    assert result.cached_state.x_hc[5, 0, 0] == 92
    publications = []

    def publish_global(layer, state, new_metadata):
        assert layer.layer_id == 20 and state.num_tokens == 26
        assert new_metadata is metadata
        publications.append(layer.layer_id)

    decoder = run_prefill_decoder(
        result.new_state, metadata, table, publish_global,
        lambda layer, state, replay, _: state,
        cached_prefix=CachedEncoderTail(result.cached_state, result.cached_query_start_loc),
    )
    assert publications == [20]
    assert decoder.replay.position_ids.tolist() == [*range(7), *range(96, 224)]
    assert decoder.state.x_hc[:, 0, 0].tolist() == [*range(20, 27), *range(116, 244)]


def test_ced_publishes_full_encoder_stream_before_bounded_decoder_layers():
    metadata, table = metadata_for((129,))
    hidden = torch.zeros((129, FLASH.hc_mult, FLASH.hidden_size), dtype=torch.float32)
    hidden[:, 0, 0] = torch.arange(129)
    state = PrefillState(hidden, torch.zeros((129, FLASH.hc_mult)))
    events = []

    def encoder_layer(layer, layer_state, full_metadata):
        assert layer_state.num_tokens == 129
        assert full_metadata is metadata
        events.append(("encoder", layer.layer_id))
        return layer_state

    def publish_global(layer, layer_state, full_metadata):
        assert layer.layer_id == 20 and layer_state.num_tokens == 129
        assert full_metadata is metadata
        events.append(("publish", layer.layer_id))

    def decoder_layer(layer, layer_state, replay, full_metadata):
        assert layer_state.num_tokens == 128
        assert replay.position_ids[0] == 1
        assert full_metadata is metadata
        assert layer_state.x_hc[0, 0, 0] == 1
        events.append(("decoder", layer.layer_id))
        return layer_state

    final_encoder = run_prefill_encoder(state, metadata, encoder_layer)
    result = run_prefill_decoder(final_encoder, metadata, table, publish_global, decoder_layer)
    assert events == [
        *( ("encoder", layer_id) for layer_id in range(20) ),
        ("publish", 20),
        *( ("decoder", layer_id) for layer_id in range(20, 40) ),
    ]
    assert result.state.num_tokens == 128
    assert result.replay.logit_row_indices.tolist() == [127]


def test_checkpoint_reader_transposes_fp8_blocks_and_reorders_scales(tmp_path):
    """A safetensors shard uses output-major weights and 32x32 E8M0 blocks."""
    stem = "layers.0.attn.wq_a"
    weight = (torch.arange(32 * 64).reshape(32, 64).remainder(7) - 3).to(torch.float8_e4m3fn)
    scales = torch.tensor([[127, 128]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    payloads = {
        stem + ".weight": ("F8_E4M3", bytes(weight.contiguous().view(torch.uint8).flatten().tolist()), [32, 64]),
        stem + ".scale": ("F8_E8M0", bytes(scales.view(torch.uint8).flatten().tolist()), [1, 2]),
    }
    header = {}
    body = bytearray()
    for name, (dtype, payload, shape) in payloads.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(body), len(body) + len(payload)]}
        body.extend(payload)
    encoded_header = json.dumps(header).encode()
    shard = "model-00001-of-00001.safetensors"
    (tmp_path / shard).write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + body)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in payloads}})
    )
    (tmp_path / "config.json").write_text(json.dumps({
        "text_config": {
            "hidden_size": FLASH.hidden_size,
            "num_hidden_layers": FLASH.num_hidden_layers,
            "num_attention_heads": FLASH.num_attention_heads,
            "head_dim": FLASH.head_dim,
            "sliding_window": FLASH.sliding_window,
            "kv_source_layer_ids": list(FLASH.kv_source_layer_ids),
            "index_source_layer_ids": list(FLASH.index_source_layer_ids),
        },
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32]},
    }))
    reader = PrefillCheckpoint(tmp_path)
    converted, packed_scales = reader.mx_linear(stem)
    assert torch.equal(converted.T.contiguous().view(torch.uint8), weight.view(torch.uint8))
    expected_scales = scales.T.contiguous().view(torch.uint8).repeat_interleave(32, dim=1)
    assert torch.equal(unpack_mx_b_scale(packed_scales).view(torch.uint8), expected_scales)
