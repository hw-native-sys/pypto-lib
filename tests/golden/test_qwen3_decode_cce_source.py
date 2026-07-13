# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source-level checks for the Qwen3 direct CANN attention integration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DECODE = ROOT / "models" / "qwen3" / "14b" / "decode_fwd.py"


def _source() -> str:
    return DECODE.read_text(encoding="utf-8")


def test_decode_uses_vllm_attention_layout_without_materialized_cache_conversion() -> None:
    source = _source()

    assert "q_tnd_flat = pl.create_tensor([BATCH * NUM_HEADS, HEAD_DIM]" in source
    assert "attn_out_tnd = pl.reshape(attn_out, [BATCH, NUM_HEADS, HEAD_DIM])" in source
    assert "attn_out_tnd = paged_attention_cce(" in source
    assert "+ (wr_slot_block * BLOCK_SIZE + wr_slot_offset) * NUM_KV_HEADS" in source
    assert "(wr_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE" not in source
    assert "pl.create_tensor([KV_CACHE_ROWS_DYN" not in source


def test_decode_replaces_only_attention_and_keeps_main_numeric_path() -> None:
    source = _source()

    assert "# ci: no-dep-gen" in source
    assert "QWEN3_14B_DIMS as D" in source
    assert "QWEN3_14B_TILING as T" in source
    assert "QWEN3_14B as M" in source
    assert "rms_lm_head_fp32(cur, final_norm_weight, lm_head_weight, seq_lens, out)" in source
    assert "rope_dep_tids = pl.array.create(ROPE_NDEPS, pl.TASK_ID)" in source
    assert "name_hint=\"rope_qkv\"" in source
    assert "q_raw = pl.mul(" in source
    assert "k_raw = pl.mul(" in source
    assert "name_hint=\"out_proj\"" in source
    assert "name_hint=\"dcr_xgamma\"" in source
    assert "fa_work_build" not in source
    assert "online_softmax" not in source
    assert "pl.system.syncall" not in source
    assert "all_q_padded" not in source
    assert "allow_early_resolve" not in source


def test_decode_public_batch_is_active_while_internal_rows_stay_padded() -> None:
    source = _source()

    assert "seq_lens: pl.Tensor[[D.user_batch], pl.INT32]" in source
    assert "block_table: pl.Tensor[[D.block_table_flat], pl.INT32]" in source
    assert "for b in pl.range(user_batch, BATCH):" in source
    assert "valid_shape=[user_batch, RMSNORM_K_CHUNK]" in source
    assert "pad_value=pl.PadValue.zero" in source
    assert "if g_idx < NUM_KV_HEADS * user_batch:" in source
    assert "q_row = b * NUM_HEADS + q_base" in source


def test_attention_workspace_and_metadata_are_shared_across_layers() -> None:
    source = _source()

    assert source.count("pa_metadata = pl.create_tensor([PA_METADATA_BYTES]") == 2
    assert source.count("pa_workspace = pl.create_tensor([PA_WORKSPACE_BYTES]") == 2
    assert "pa_metadata: pl.Tensor[[PA_METADATA_BYTES], pl.UINT8]" in source
    assert "pa_workspace: pl.Tensor[[PA_WORKSPACE_BYTES], pl.UINT8]" in source
