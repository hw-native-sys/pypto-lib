# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash A2 prefill and A2/A3 decoder operators."""

from importlib import import_module


_EXPORTS = {
    "A2A3OperatorConfig": ("operator_config", "A2A3OperatorConfig"),
    "A2A3_DTYPE_CONTRACT": ("operator_config", "A2A3_DTYPE_CONTRACT"),
    "make_candidate_block_pool": ("decode_indexer", "make_candidate_block_pool"),
    "get_decode_layer_entry": ("decode_layer", "get_layer_entry"),
    "decode_layer_entries": ("decode_layer", "LAYER_ENTRIES"),
    "dequantize_int32": ("quant", "dequantize_int32"),
    "int8_matmul": ("quant", "int8_matmul"),
    "quantize_per_channel_int8": ("quant", "quantize_per_channel_int8"),
    "quantize_per_token_int8": ("quant", "quantize_per_token_int8"),
    "w8a8_linear": ("quant", "w8a8_linear"),
    "load_layer0_qkv": ("weights", "load_layer0_qkv"),
    "load_layer0_output": ("weights", "load_layer0_output"),
    "load_layer0_router": ("weights", "load_layer0_router"),
    "load_layer0_shared_expert": ("weights", "load_layer0_shared_expert"),
    "load_tensor_bundle": ("weights", "load_tensor_bundle"),
    "open_layer0_moe": ("weights", "open_layer0_moe"),
    "QKVShape": ("qkv", "QKVShape"),
    "make_qkv_w8a8": ("qkv", "make_qkv_w8a8"),
    "qkv_reference": ("qkv", "qkv_reference"),
    "qkv_shape": ("qkv", "qkv_shape"),
    "make_routed_expert_w8a8": ("moe", "make_routed_expert_w8a8"),
    "make_shared_expert_w8a8": ("moe", "make_shared_expert_w8a8"),
    "layer0_reference": ("layer0", "layer0_reference"),
    "route_reference": ("layer0", "route_reference"),
    "make_decode_step": ("decode_fwd", "make_decode_step"),
    "make_decode_replay": ("decode_replay", "make_decode_replay"),
    "validate_decode_contract": ("decode_fwd", "validate_decode_contract"),
    "DecodeCandidate": ("decode_candidates", "DecodeCandidate"),
    "DECODE_CANDIDATES": ("decode_candidates", "CANDIDATES"),
    "validate_decode_candidates": ("decode_candidates", "validate_candidates"),
    "decode_acceptance_report": ("decode_acceptance", "report"),
    "decode_gate_report": ("decode_gate", "gate_report"),
    "validate_decode_assets": ("decode_asset_contract", "validate_assets"),
    "decode_ep2": ("decode_moe", "decode_ep2"),
    "decode_ep2_contract": ("decode_moe", "product_contract"),
    "decode_layer_ep2": ("decode_layer", "decode_layer_ep2"),
    "decode_layer_ep2_contract": ("decode_layer", "product_contract"),
    "decode_layer_ep2_engram": ("decode_layer", "decode_layer_ep2_engram"),
    "decode_layer_ep2_sparse": ("decode_layer", "decode_layer_ep2_sparse"),
    "decode_layer_ep2_c1a_full": ("decode_layer", "decode_layer_ep2_c1a_full"),
    "decode_layer_ep2_c1a_reindex": ("decode_layer", "decode_layer_ep2_c1a_reindex"),
    "decode_layer_ep2_c1a_reuse": ("decode_layer", "decode_layer_ep2_c1a_reuse"),
    "decode_reference_manifest": ("decode_reference", "reference_manifest"),
    "inspect_decode_checkpoint": ("decode_checkpoint_manifest", "inspect_checkpoint"),
    "DecodeSafetensorsCheckpoint": ("decode_weights", "DecodeSafetensorsCheckpoint"),
    "decode_weight_abi_summary": ("decode_weight_abi", "abi_summary"),
    "pack_decode_routed_expert": ("decode_weight_abi", "pack_routed_expert"),
    "load_decode_ep2_rank_weights": ("decode_weight_abi", "load_ep2_rank_weights"),
    "DecodeGlobalShape": ("decode_shape_contract", "DecodeGlobalShape"),
    "DecodeRankShape": ("decode_shape_contract", "DecodeRankShape"),
    "decode_shape_matrix": ("decode_shape_contract", "shape_matrix"),
    "decode_rank_shape": ("decode_shape_contract", "rank_shape"),
    "DECODE_RUNTIME_CONTRACT": ("runtime_config", "DECODE_RUNTIME_CONTRACT"),
    "engram_reference": ("engram", "engram_reference"),
    "make_gate": ("engram", "make_gate"),
    "make_lookup": ("engram", "make_lookup"),
    "make_bf16_linear": ("staged", "make_bf16_linear"),
    "make_fp32_linear": ("staged", "make_fp32_linear"),
    "make_grouped_wo_a": ("staged", "make_grouped_wo_a"),
    "make_index_quantize": ("staged", "make_index_quantize"),
    "make_linear": ("staged", "make_linear"),
    "make_norm": ("staged", "make_norm"),
    "make_rope_kernel": ("staged", "make_rope_kernel"),
    "make_swa": ("staged", "make_swa"),
    "make_swiglu": ("staged", "make_swiglu"),
    "make_sparse_attention": ("sparse_attention", "make_sparse_attention"),
    "sparse_attention_reference": ("sparse_attention", "sparse_attention_reference"),
    "indexer_query_reference": ("decode_indexer", "indexer_query_reference"),
    "make_indexer_query_quantize": ("decode_indexer", "make_indexer_query_quantize"),
    "make_l3_indexer_query": ("decode_indexer", "make_l3_indexer_query"),
    "quantize_indexer_query_reference": ("decode_indexer", "quantize_indexer_query_reference"),
    "indexer_select_reference": ("decode_indexer", "indexer_select_reference"),
    "make_indexer_select": ("decode_indexer", "make_indexer_select"),
    "make_indexer_select_restricted": ("decode_indexer", "make_indexer_select_restricted"),
    "make_indexer_prepare": ("decode_indexer", "make_indexer_prepare"),
    "prepare_indexer_indices_reference": ("decode_indexer", "prepare_indexer_indices_reference"),
    "make_prefill_swa": ("prefill_fwd", "make_prefill_swa"),
    "make_prefill_l0_attention": ("prefill_layer", "make_l0_attention"),
    "make_prefill_layer2_producer": ("prefill_layer", "make_layer2_prefill_producer"),
    "LAYER14": ("decode_c2a_full", "LAYER14"),
    "LAYER15": ("decode_c2a_reuse", "LAYER15"),
    "Layer14A2A3Contract": ("decode_c2a_full", "Layer14A2A3Contract"),
    "Layer15A2A3Contract": ("decode_c2a_reuse", "Layer15A2A3Contract"),
    "index_key_reference": ("decode_c2a_full", "index_key_reference"),
    "dequantize_engram_rows_reference": ("decode_c2a_full", "dequantize_engram_rows_reference"),
    "engram_a2a3_reference": ("decode_c2a_full", "engram_a2a3_reference"),
    "layer14_source_reference": ("decode_c2a_full", "layer14_source_reference"),
    "layer15_reuse_reference": ("decode_c2a_reuse", "layer15_reuse_reference"),
    "load_a2a3_layer14_15_bundle": ("decode_c2a_full", "load_a2a3_layer14_15_bundle"),
    "make_layer14_stages": ("decode_c2a_full", "make_layer14_stages"),
    "make_layer15_stages": ("decode_c2a_reuse", "make_layer15_stages"),
    "make_ratio2_projection": ("decode_c2a_full", "make_ratio2_projection"),
    "quantize_engram_rows_reference": ("decode_c2a_full", "quantize_engram_rows_reference"),
    "ratio2_pool_reference": ("decode_c2a_full", "ratio2_pool_reference"),
    "update_ratio2_state_reference": ("decode_c2a_full", "update_ratio2_state_reference"),
    "validate_layer14_15_contract": ("decode_c2a_full", "validate_layer14_15_contract"),
    "RawCheckpoint": ("raw_checkpoint", "RawCheckpoint"),
    "flatten_scale": ("raw_checkpoint", "flatten_scale"),
    "load_attention_weights": ("raw_checkpoint", "load_attention_weights"),
    "load_decoder_layer_weights": ("raw_checkpoint", "load_decoder_layer_weights"),
    "LAYER19": ("decode_c2a_reuse", "LAYER19"),
    "LAYER20": ("decode_c1a_full", "LAYER20"),
    "LAYER21": ("decode_c1a_reuse", "LAYER21"),
    "LAYER24": ("decode_c1a_reindex", "LAYER24"),
    "LAYER25": ("decode_c1a_reuse", "LAYER25"),
    "TARGET_LAYERS": ("decode_layer_stages", "TARGET_LAYERS"),
    "c1a_full_reference": ("decode_c1a_full", "c1a_full_reference"),
    "c1a_reindex_reference": ("decode_c1a_reindex", "c1a_reindex_reference"),
    "c1a_reuse_reference": ("decode_c1a_reuse", "c1a_reuse_reference"),
    "c2a_reuse_reference": ("decode_c2a_reuse", "c2a_reuse_reference"),
    "layer_contract": ("decode_layer_stages", "layer_contract"),
    "make_decoder_layer_stages": ("decode_layer_stages", "make_decoder_layer_stages"),
    "select_candidate_block_pool_reference": ("decode_layer_stages", "select_candidate_block_pool_reference"),
    "validate_layer19_20_21_24_25_contracts": (
        "decode_layer_stages",
        "validate_layer19_20_21_24_25_contracts",
    ),
}


__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public operator only when a caller requests it."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return eager package names and lazy public operator names."""

    return sorted(set(globals()) | set(__all__))
