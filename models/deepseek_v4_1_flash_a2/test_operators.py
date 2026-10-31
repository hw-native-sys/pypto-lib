"""CPU/PyTorch precision checks for the A2/A3 leaf operator contracts."""

import pytest
import torch

from .index_cache import dequantize_index_cache, quantize_index_cache
from .moe import routed_expert_reference, shared_expert_reference
from .norm_rope import rms_norm_reference, rope_reference
from .precision import compare_dump, compare_tensors
from .quant import (
    dequantize_int32,
    int8_matmul,
    quantize_per_channel_int8,
    quantize_per_token_int8,
    w8a8_linear,
)
from .swa_cache import gather_swa_cache_reference, write_swa_cache_reference
from .sparse_attention import sparse_attention_reference
from .weights import load_tensor_bundle
from .qkv import qkv_reference, qkv_shape
from .indexer_query import quantize_indexer_query_reference
from .indexer_select import indexer_select_reference
from .indexer_prepare import prepare_indexer_indices_reference


def test_dynamic_quant_and_int32_gemm_match_explicit_reference():
    torch.manual_seed(0)
    value = torch.randn(3, 64, dtype=torch.bfloat16)
    weight_fp = torch.randn(5, 64, dtype=torch.float32)
    weight, weight_scale = quantize_per_channel_int8(weight_fp)
    payload, token_scale = quantize_per_token_int8(value)
    accumulator = int8_matmul(payload, weight)
    expected = dequantize_int32(accumulator, token_scale, weight_scale)
    actual = w8a8_linear(value, weight, weight_scale)
    assert compare_tensors(actual, expected).max_abs == 0.0


def test_zero_rows_keep_vllm_scale_floors():
    value = torch.zeros((2, 64), dtype=torch.bfloat16)
    _, token_scale = quantize_per_token_int8(value)
    assert torch.equal(token_scale, torch.full_like(token_scale, 1.0e-30 / 127.0))
    index_payload, index_scale = quantize_index_cache(torch.zeros((1, 1, 128), dtype=torch.bfloat16))
    assert torch.count_nonzero(index_payload) == 0
    assert torch.equal(index_scale, torch.full_like(index_scale, 2.0 ** -24))


def test_bf16_rmsnorm_and_rope_boundaries():
    torch.manual_seed(1)
    value = torch.randn(2, 2 * 8, dtype=torch.bfloat16)
    weight = torch.randn(16, dtype=torch.bfloat16)
    normalized = rms_norm_reference(value, weight)
    cosine = torch.randn(2, 2, dtype=torch.float32)
    sine = torch.randn(2, 2, dtype=torch.float32)
    rotated = rope_reference(normalized, cosine, sine, heads=2, head_dim=8, rope_dim=4)
    assert normalized.dtype == torch.bfloat16
    assert rotated.shape == value.shape


def test_bf16_swa_cache_round_trip():
    cache = torch.zeros((2, 128, 1, 4), dtype=torch.bfloat16)
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
    slots = torch.tensor([[0, 2], [1, 4], [-1, 0]], dtype=torch.int64)
    updated = write_swa_cache_reference(cache, values, slots)
    indices = torch.tensor([[2, 132, -1]], dtype=torch.int32)
    gathered = gather_swa_cache_reference(updated, indices)
    assert torch.equal(gathered[0, 0], values[0])
    assert torch.equal(gathered[0, 1], values[1])
    assert torch.count_nonzero(gathered[0, 2]) == 0


def test_index_cache_is_int8_with_fp16_scale():
    torch.manual_seed(2)
    value = torch.randn(2, 3, 128, dtype=torch.bfloat16)
    payload, scale = quantize_index_cache(value)
    restored = dequantize_index_cache(payload, scale)
    assert payload.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert restored.shape == value.shape
    assert compare_tensors(restored, value).max_abs < 0.04


def test_indexer_query_uses_rounded_fp16_scale_before_payload_quantization():
    query = torch.zeros((1, 2, 128), dtype=torch.float32)
    query[0, 0, :6] = torch.tensor([127.0, 63.5, -63.5, 1.5, -1.5, 2.5])
    query[0, 1, :2] = torch.tensor([0.0, -0.0])
    payload, scales = quantize_indexer_query_reference(query)
    assert payload.dtype == torch.int8
    assert scales.dtype == torch.float16
    assert scales[0, 0] == torch.tensor(1.0, dtype=torch.float16)
    assert torch.equal(payload[0, 0, :6], torch.tensor([127, 64, -64, 2, -2, 2], dtype=torch.int8))
    assert torch.equal(payload[0, 1], torch.zeros(128, dtype=torch.int8))


def test_ratio2_sparse_attention_masks_both_bf16_cache_planes():
    torch.manual_seed(23)
    query = torch.randn(2, 16, 32, dtype=torch.bfloat16)
    window = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    compressed = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    window_indices = torch.tensor([[0, 1, -1, -1], [0, -1, 2, -1]], dtype=torch.int32)
    compressed_indices = torch.tensor([[0, -1, 2, -1, -1], [1, 2, 3, 4, -1]], dtype=torch.int32)
    sink = torch.zeros(16, dtype=torch.float32)
    output = sparse_attention_reference(
        query, window, window_indices, compressed, compressed_indices, sink
    )
    assert output.dtype == torch.bfloat16
    assert output.shape == query.shape
    assert torch.isfinite(output.float()).all()


def test_qli_int8_score_and_topk_reference():
    torch.manual_seed(24)
    tokens, keys, topk = 3, 96, 32
    query = torch.randint(-127, 128, (tokens, 32, 128), dtype=torch.int8)
    key = torch.randint(-127, 128, (keys, 128), dtype=torch.int8)
    query_scale = torch.rand(tokens, 32, dtype=torch.float16) * 0.02
    key_scale = torch.rand(keys, dtype=torch.float16) * 0.02
    weights = torch.rand(tokens, 32, dtype=torch.float16)
    positions = torch.tensor([31, 95, 191], dtype=torch.int32)
    result = indexer_select_reference(
        query, query_scale, key, key_scale, weights, positions, topk=topk, compress_ratio=2
    )
    assert result["scores"].shape == (tokens, keys)
    assert result["indices"].shape == (tokens, topk)
    assert result["indices"].dtype == torch.int32
    assert torch.equal(result["indices"][0, 16:], torch.full((16,), -1, dtype=torch.int32))
    assert result["indices"][1].ge(0).all() and result["indices"][1].lt(48).all()
    assert result["indices"][1].unique().numel() == topk
    assert result["indices"][2].ge(0).all() and result["indices"][2].lt(keys).all()
    assert result["indices"][2].unique().numel() == topk


def test_prepare_indexer_indices_filters_and_sorts_visible_prefix():
    selected = torch.tensor(
        [[7, 1, -1, 3, 2, 99, 0, -1], [5, 4, 3, 2, 1, 0, -1, -1]],
        dtype=torch.int32,
    )
    positions = torch.tensor([7, 5], dtype=torch.int32)
    result = prepare_indexer_indices_reference(selected, positions, compress_ratio=2)
    assert torch.equal(
        result["indices"],
        torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1], [0, 1, 2, -1, -1, -1, -1, -1]], dtype=torch.int32),
    )
    assert torch.equal(result["lengths"].flatten(), torch.tensor([4, 3], dtype=torch.int32))


def test_shared_and_routed_w8a8_experts():
    torch.manual_seed(3)
    tokens, hidden, intermediate, experts = 2, 32, 32, 2
    value = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    gate_up_fp = torch.randn(2 * intermediate, hidden)
    down_fp = torch.randn(hidden, intermediate)
    gate_up, gate_up_scale = quantize_per_channel_int8(gate_up_fp)
    down, down_scale = quantize_per_channel_int8(down_fp)
    shared = shared_expert_reference(value, gate_up, gate_up_scale, down, down_scale)
    assert shared.shape == value.shape

    w1_fp = torch.randn(experts, intermediate, hidden)
    w2_fp = torch.randn(experts, hidden, intermediate)
    w3_fp = torch.randn(experts, intermediate, hidden)
    w1, w1_scale = quantize_per_channel_int8(w1_fp)
    w2, w2_scale = quantize_per_channel_int8(w2_fp)
    w3, w3_scale = quantize_per_channel_int8(w3_fp)
    routed = routed_expert_reference(
        value,
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32),
        w1,
        w1_scale,
        w2,
        w2_scale,
        w3,
        w3_scale,
    )
    assert routed.shape == value.shape


def test_prepared_bundle_contract(tmp_path):
    bundle = {"weight": torch.zeros((64, 32), dtype=torch.int8), "scale": torch.ones(64, dtype=torch.float32)}
    path = tmp_path / "bundle.pt"
    torch.save(bundle, path)
    loaded = load_tensor_bundle(path, {"weight": (torch.int8, (64, 32)), "scale": (torch.float32, (64,))})
    assert loaded["weight"].dtype == torch.int8


def test_nested_vllm_case_dump_can_be_selected(tmp_path):
    expected = {
        "cases": {
            "layer0-cap16-active7": {
                "query": torch.ones(2, 3),
                "diagnostics": {"wkv": torch.zeros(2, 4)},
                "metadata": {"active": 2},
            }
        }
    }
    path = tmp_path / "vllm.pt"
    torch.save(expected, path)
    metrics = compare_dump(
        {"query": torch.ones(2, 3), "diagnostics": {"wkv": torch.zeros(2, 4)}},
        path,
        case="layer0-cap16-active7",
    )
    assert metrics["query"].exact
    assert metrics["diagnostics/wkv"].exact


def test_qkv_reference_uses_canonical_k_by_n_weights():
    torch.manual_seed(46)
    config = {
        "hidden_size": 512,
        "q_lora_rank": 256,
        "num_attention_heads": 2,
        "head_dim": 256,
        "qk_rope_head_dim": 64,
        "rms_norm_eps": 1.0e-20,
    }
    value = torch.randn(4, 512, dtype=torch.bfloat16)
    weights = {}
    for name, shape in {
        "wq_a": (512, 256),
        "wq_b": (256, 512),
        "wkv": (512, 256),
    }.items():
        quantized, weights[name + "_scale"] = quantize_per_channel_int8(torch.randn(shape[1], shape[0]))
        weights[name] = quantized.transpose(0, 1).contiguous()
    weights["q_norm_weight"] = torch.randn(256, dtype=torch.bfloat16)
    weights["kv_norm_weight"] = torch.randn(256, dtype=torch.bfloat16)
    cosine = torch.randn(4, 32)
    sine = torch.randn(4, 32)
    result = qkv_reference(value, weights, cosine, sine, config)
    assert qkv_shape(config).query_width == 512
    assert result["query_latent"].shape == (4, 256)
    assert result["query"].shape == (4, 512)
    assert result["window_kv"].shape == (4, 256)
