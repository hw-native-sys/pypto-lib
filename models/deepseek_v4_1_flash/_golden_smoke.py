# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Small deterministic CPU fixtures for directly executing operator goldens."""

import inspect
from collections.abc import Callable

import torch

from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale
from models.deepseek_v4_1_flash.quantization import quantize_mxfp4_cache
from models.deepseek_v4_1_flash.quantization import quantize_mxfp4_weight
from models.deepseek_v4_1_flash.quantization import quantize_mxfp8_cache


def _mxfp8_weight(input_dim: int, output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(input_dim, output_dim).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    scale = pack_mx_b_scale(torch.full((input_dim // 32, output_dim), 127, dtype=torch.uint8))
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is not None:
        scale = scale.view(e8m0_dtype)
    return weight, scale


def _attention_values() -> dict[str, torch.Tensor | None]:
    """Build small deterministic inputs for all attention golden modes."""
    torch.manual_seed(7)
    wq_a, wq_a_scale = _mxfp8_weight(64, 64)
    wq_b, wq_b_scale = _mxfp8_weight(64, 128)
    wkv, wkv_scale = _mxfp8_weight(64, 64)
    wo_b, wo_b_scale = _mxfp8_weight(64, 64)
    index_wq_b, index_wq_b_scale = _mxfp8_weight(64, 64)
    window_cache, window_cache_scale = quantize_mxfp8_cache(torch.zeros(1, 128, 1, 64))
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(
        torch.randn(1, 128, 1, 64), group_size=16, scale_format="e4m3"
    )
    index_cache, index_cache_scale = quantize_mxfp4_cache(
        torch.randn(1, 128, 1, 32), group_size=32, scale_format="e8m0"
    )
    return {
        "x": torch.randn(2, 64, dtype=torch.bfloat16),
        "wq_a": wq_a,
        "wq_a_scale": wq_a_scale,
        "q_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": wkv,
        "wkv_scale": wkv_scale,
        "kv_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "attn_sink": torch.randn(2),
        "wo_a": torch.randn(2, 32, 64, dtype=torch.bfloat16),
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
        "rope_cos": torch.ones(2, 1),
        "rope_sin": torch.zeros(2, 1),
        "window_slots": torch.tensor([0, 1]),
        "window_indices": torch.tensor([[0, -1], [0, 1]], dtype=torch.int32),
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
        "compressed_cache": compressed_cache,
        "compressed_cache_scale": compressed_cache_scale,
        "compressed_indices": torch.tensor([[-1, -1], [0, -1]], dtype=torch.int32),
        "request_ids": torch.tensor([0, 0], dtype=torch.int32),
        "position_ids": torch.tensor([0, 1], dtype=torch.int32),
        "compressed_lens": torch.tensor([0, 1], dtype=torch.int32),
        "compressed_rope_cos": torch.ones(2, 1),
        "compressed_rope_sin": torch.zeros(2, 1),
        "index_cache": index_cache,
        "index_cache_scale": index_cache_scale,
        "index_block_table": torch.tensor([[0]], dtype=torch.int32),
        "candidate_mask": torch.ones(2, 2, dtype=torch.bool),
        "compressor_wkv": torch.randn(64, 64),
        "compressor_wgate": torch.randn(64, 64),
        "query_start_loc": torch.tensor([0, 2], dtype=torch.int32),
        "token_to_req_indices": torch.tensor([0, 0], dtype=torch.int32),
        "state_block_table": torch.tensor([[1]], dtype=torch.int32),
        "state_cache": torch.zeros(2, 4, 128),
        "compressor_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "compressed_slots": torch.tensor([-1, 0]),
        "index_wk": torch.randn(64, 32, dtype=torch.bfloat16),
        "index_norm_weight": torch.ones(32, dtype=torch.bfloat16),
        "index_wq_b": index_wq_b,
        "index_wq_b_scale": index_wq_b_scale,
        "index_weights_proj": torch.randn(64, 2, dtype=torch.bfloat16),
    }


def run_attention_golden(golden_fn: Callable[..., object], ratio: int, mode: str) -> None:
    """Execute one attention golden with a small mode-valid fixture."""
    values = _attention_values()
    if ratio == 1:
        values["compressed_lens"] = torch.tensor([1, 2], dtype=torch.int32)
        values["compressed_slots"] = torch.tensor([0, 1])
        values["compressor_wkv"] = values["compressor_wkv"].to(torch.bfloat16)
    if mode == "reuse":
        values["compressed_indices"] = torch.tensor([[0, -1], [0, 1]], dtype=torch.int32)
    parameters = inspect.signature(golden_fn).parameters
    kwargs = {name: values[name] for name in parameters}
    result = golden_fn(**kwargs)
    if not bool(torch.isfinite(result.output).all()):
        raise RuntimeError("attention golden produced non-finite output")
    if result.output.dtype is not torch.bfloat16:
        raise RuntimeError(f"attention golden produced {result.output.dtype}, expected torch.bfloat16")
    if (
        result.window_cache.dtype is not torch.float8_e4m3fn
        or result.window_cache_scale.dtype is not torch.uint8
    ):
        raise RuntimeError("attention golden did not preserve the MXFP8 window-cache ABI")
    if ratio:
        if result.compressed_cache is None or result.compressed_cache.dtype is not torch.uint8:
            raise RuntimeError("attention golden did not preserve the packed MXFP4 compressed-cache ABI")
        if (
            result.compressed_cache_scale is None
            or result.compressed_cache_scale.dtype is not torch.float8_e4m3fn
        ):
            raise RuntimeError("attention golden did not preserve the E4M3 compressed-cache scale ABI")
    if mode in ("full", "reindex"):
        if result.index_cache is None or result.index_cache.dtype is not torch.uint8:
            raise RuntimeError("attention golden did not preserve the packed MXFP4 index-cache ABI")
        if result.index_cache_scale is None or result.index_cache_scale.dtype is not torch.uint8:
            raise RuntimeError("attention golden did not preserve the E8M0 index-cache scale ABI")
    if ratio and mode != "full":
        if not torch.equal(result.compressed_cache, values["compressed_cache"]):
            raise RuntimeError("a non-owner attention mode modified the compressed cache")
        if not torch.equal(
            result.compressed_cache_scale.view(torch.uint8),
            values["compressed_cache_scale"].view(torch.uint8),
        ):
            raise RuntimeError("a non-owner attention mode modified compressed-cache scales")
    if mode == "reindex":
        if not torch.equal(result.index_cache, values["index_cache"]):
            raise RuntimeError("reindex attention modified the source index cache")
        if not torch.equal(result.index_cache_scale, values["index_cache_scale"]):
            raise RuntimeError("reindex attention modified source index-cache scales")
    print(f"[GOLDEN] PASS {golden_fn.__name__} output={tuple(result.output.shape)}")


def run_hierarchical_indexer_golden(golden_fn: Callable[..., torch.Tensor]) -> None:
    width = 2050 * 8
    scores = -torch.arange(width, dtype=torch.float32).reshape(1, width)
    candidate_mask = golden_fn(scores, torch.tensor([width]))
    if not bool(candidate_mask[..., -8:].all()) or int(candidate_mask.sum()) > 2048 * 8:
        raise RuntimeError("candidate selector did not pin the newest partial block")
    print(f"[GOLDEN] PASS {golden_fn.__name__} output={tuple(candidate_mask.shape)}")


def run_mhc_goldens(
    mixes_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pre_fn: Callable[..., torch.Tensor],
    post_fn: Callable[..., torch.Tensor],
    head_fn: Callable[..., torch.Tensor],
) -> None:
    torch.manual_seed(3)
    x_hc = torch.randn(2, 4, 3)
    pre_mix, post_mix, residual_mix = mixes_fn(x_hc, torch.randn(24, 12), torch.randn(3), torch.randn(24))
    sublayer_input = pre_fn(x_hc, pre_mix)
    output = post_fn(sublayer_input, x_hc, post_mix, residual_mix)
    collapsed = head_fn(output, pre_mix)
    if sublayer_input.dtype is not torch.bfloat16:
        raise RuntimeError("mHC pre golden did not match the BF16 kernel ABI")
    if output.dtype is not torch.float32:
        raise RuntimeError("mHC post golden did not match the FP32 kernel ABI")
    if not bool(torch.isfinite(collapsed).all()):
        raise RuntimeError("mHC golden produced non-finite output")
    print(f"[GOLDEN] PASS mHC output={tuple(collapsed.shape)}")


def run_moe_golden(golden_fn: Callable[..., torch.Tensor]) -> None:
    torch.manual_seed(11)
    routed_w1, routed_w1_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    routed_w2, routed_w2_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    routed_w3, routed_w3_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    shared_up_scale = pack_mx_b_scale(torch.full((2, 128), 127, dtype=torch.uint8))
    shared_down_scale = pack_mx_b_scale(torch.full((4, 64), 127, dtype=torch.uint8))
    output = golden_fn(
        torch.randn(3, 64),
        torch.ones(64),
        torch.randn(8, 64),
        torch.randn(8),
        routed_w1,
        routed_w1_scale,
        routed_w2,
        routed_w2_scale,
        routed_w3,
        routed_w3_scale,
        torch.randn(64, 128).to(torch.float8_e4m3fn),
        shared_up_scale,
        torch.randn(128, 64).to(torch.float8_e4m3fn),
        shared_down_scale,
        torch.randn(64, 128).to(torch.float8_e4m3fn),
        shared_up_scale,
        token_owners=torch.tensor([0, 1, 2], dtype=torch.int32),
        tp_size=4,
    )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("MoE golden produced non-finite output")
    print(f"[GOLDEN] PASS {golden_fn.__name__} output={tuple(output.shape)}")


def make_decode_layer_golden_inputs(layer_id: int) -> dict:
    """Build a deterministic small Block fixture for one representative layer."""
    torch.manual_seed(17 + layer_id)
    attention = _attention_values()
    if layer_id >= 20:
        attention["compressed_lens"] = torch.tensor([1, 2], dtype=torch.int32)
        attention["compressed_slots"] = torch.tensor([0, 1], dtype=torch.int64)
        attention["compressor_wkv"] = attention["compressor_wkv"].to(torch.bfloat16)
    if layer_id in (3, 21):
        attention["compressed_indices"] = torch.tensor([[0, -1], [0, 1]], dtype=torch.int32)

    routed_w1, routed_w1_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    routed_w2, routed_w2_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    routed_w3, routed_w3_scale = quantize_mxfp4_weight(torch.randn(8, 64, 64))
    shared_scale = pack_mx_b_scale(torch.full((2, 64), 127, dtype=torch.uint8))
    moe = {
        "gate_weight": torch.randn(8, 64),
        "correction_bias": torch.randn(8),
        "routed_w1": routed_w1,
        "routed_w1_scale": routed_w1_scale,
        "routed_w2": routed_w2,
        "routed_w2_scale": routed_w2_scale,
        "routed_w3": routed_w3,
        "routed_w3_scale": routed_w3_scale,
        "shared_w1": torch.randn(64, 64).to(torch.float8_e4m3fn),
        "shared_w1_scale": shared_scale,
        "shared_w2": torch.randn(64, 64).to(torch.float8_e4m3fn),
        "shared_w2_scale": shared_scale,
        "shared_w3": torch.randn(64, 64).to(torch.float8_e4m3fn),
        "shared_w3_scale": shared_scale,
        "token_owners": torch.tensor([0, 1], dtype=torch.int32),
        "tp_size": 4,
    }
    incoming_pre_mix = torch.zeros(2, 4)
    incoming_pre_mix[:, 0] = 1.0
    return {
        "layer_id": layer_id,
        "x_hc": torch.randn(2, 4, 64),
        "incoming_pre_mix": incoming_pre_mix,
        "hc_attn_fn": torch.randn(24, 256) / 16,
        "hc_attn_scale": torch.randn(3),
        "hc_attn_base": torch.randn(24),
        "attn_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "hc_ffn_fn": torch.randn(24, 256) / 16,
        "hc_ffn_scale": torch.randn(3),
        "hc_ffn_base": torch.randn(24),
        "ffn_norm_weight": torch.ones(64, dtype=torch.bfloat16),
        "attention_inputs": attention,
        "moe_inputs": moe,
    }


def run_decode_layer_goldens(golden_fn: Callable[..., object], layer_ids) -> None:
    """Execute the six decode Block goldens without compiling unfinished kernels."""
    for layer_id in layer_ids:
        result = golden_fn(**make_decode_layer_golden_inputs(layer_id))
        if result.output.dtype is not torch.float32:
            raise RuntimeError(f"layer {layer_id} produced {result.output.dtype}, expected torch.float32")
        if not bool(torch.isfinite(result.output).all()):
            raise RuntimeError(f"layer {layer_id} produced non-finite output")
        print(
            f"[GOLDEN] PASS decode_layer layer={layer_id} "
            f"output={tuple(result.output.shape)} next_pre_mix={tuple(result.next_pre_mix.shape)}"
        )


def run_two_layer_decode_chain(golden_fn: Callable[..., object], *, tp_size: int = 4) -> None:
    """Run two consecutive Block goldens across the sequence-parallel boundary.

    The device Block still depends on the EP8 MoE implementation.  This CPU
    check nevertheless exercises the contract owned by Decode: the first layer's
    residual output and its delayed mHC pre-mix are split into owner slabs, are
    handed to the second layer through the same gather the collectives perform,
    and have to come back unchanged.  Slabs are compared against
    ``sequence_parallel_bounds``, the ownership rule the kernels use, so a rank
    order or padding mistake fails here instead of cancelling out in an identity
    round trip.
    """
    from models.deepseek_v4_1_flash.decode_sp_integration import (
        gather_decode_batch,
        sequence_parallel_bounds,
        shard_decode_batch,
    )

    first_inputs = make_decode_layer_golden_inputs(0)
    first = golden_fn(**first_inputs)
    tokens = first.output.shape[0]
    token_ids = torch.arange(1000, 1000 + tokens, dtype=torch.int64)
    position_ids = torch.arange(7, 7 + tokens, dtype=torch.int32)

    hidden_slabs = shard_decode_batch(first.output, token_ids, position_ids, tp_size=tp_size)
    pre_mix_slabs = shard_decode_batch(first.next_pre_mix, token_ids, position_ids, tp_size=tp_size)
    for rank, (hidden_slab, pre_mix_slab) in enumerate(zip(hidden_slabs, pre_mix_slabs)):
        first_row, count, width = sequence_parallel_bounds(tokens, tp_size, rank)
        if hidden_slab.hidden.shape[0] != width or pre_mix_slab.hidden.shape[0] != width:
            raise RuntimeError(f"rank {rank} slab width does not match the ownership rule")
        if not torch.equal(hidden_slab.hidden[:count], first.output[first_row : first_row + count]):
            raise RuntimeError(f"rank {rank} owns the wrong residual rows")
        if not torch.equal(pre_mix_slab.hidden[:count], first.next_pre_mix[first_row : first_row + count]):
            raise RuntimeError(f"rank {rank} owns the wrong delayed mHC rows")
        if count < width and not bool((hidden_slab.hidden[count:] == 0).all()):
            raise RuntimeError(f"rank {rank} leaves non-zero residual padding")

    gathered = gather_decode_batch(hidden_slabs, tokens)
    gathered_pre_mix = gather_decode_batch(pre_mix_slabs, tokens)
    if not torch.equal(gathered.hidden, first.output) or not torch.equal(
        gathered_pre_mix.hidden, first.next_pre_mix
    ):
        raise RuntimeError("the Attention boundary changed the residual stream or the delayed mHC rows")

    # The second layer consumes what the boundary published, not the pristine
    # global tensor.
    second_inputs = make_decode_layer_golden_inputs(2)
    second_inputs["x_hc"] = gathered.hidden
    second_inputs["incoming_pre_mix"] = gathered_pre_mix.hidden
    second = golden_fn(**second_inputs)

    if second.output.shape != first.output.shape or second.next_pre_mix.shape != first.next_pre_mix.shape:
        raise RuntimeError("consecutive Decode layers changed the residual boundary shape")
    if not bool(torch.isfinite(second.output).all()):
        raise RuntimeError("the second Decode layer produced non-finite hidden state")
    print(
        f"[GOLDEN] PASS decode_layer chain layers=(0,2) TP={tp_size} "
        f"output={tuple(second.output.shape)} tokens={tokens}"
    )
