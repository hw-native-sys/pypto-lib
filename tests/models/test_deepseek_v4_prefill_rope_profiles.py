# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side contracts for the DeepSeek-V4 prefill RoPE profile ABI."""

import os
import sys
from pathlib import Path

import pytest
import torch

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"
os.environ.setdefault("DEEPSEEK_V4_VARIANT", "flash")
sys.path.insert(0, str(_MODEL_DIR))

from golden import TensorSpec  # noqa: E402
from golden_fwd import _rope_profile_for_kind  # noqa: E402
from prefill_fwd import (  # noqa: E402
    COMPRESSED_ROPE_PROFILE,
    N_RANKS,
    SWA_ROPE_PROFILE,
    _make_rope_profile_spec,
    _make_shared_spec,
    _rope_profile_for_compress_ratio,
)
from rope_tables import build_deepseek_v4_rope_tables  # noqa: E402
from config import ACTIVE as MODEL_CONFIG  # noqa: E402


def _leaf_spec(name, value):
    return TensorSpec(
        name,
        list(value.shape),
        value.dtype,
        init_value=lambda value=value: value.clone(),
    )


@pytest.mark.parametrize("name", ["freqs_cos", "freqs_sin"])
def test_rope_leaf_profiles_stack_as_nonzero_distinct_tables(name):
    base = build_deepseek_v4_rope_tables(
        MODEL_CONFIG, 0, max_seq_len=32, dtype=torch.bfloat16
    )
    compressed = build_deepseek_v4_rope_tables(
        MODEL_CONFIG, 4, max_seq_len=32, dtype=torch.bfloat16
    )
    table_index = 0 if name == "freqs_cos" else 1

    spec = _make_rope_profile_spec(
        name,
        _leaf_spec(name, base[table_index]),
        _leaf_spec(name, compressed[table_index]),
    )
    stacked = spec.init_value()

    assert spec.shape == [2, 32, MODEL_CONFIG.qk_rope_head_dim]
    assert torch.count_nonzero(stacked[SWA_ROPE_PROFILE]) > 0
    assert torch.count_nonzero(stacked[COMPRESSED_ROPE_PROFILE]) > 0
    assert not torch.equal(
        stacked[SWA_ROPE_PROFILE], stacked[COMPRESSED_ROPE_PROFILE]
    )


def test_ranked_shared_rope_spec_preserves_both_profiles():
    base = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16) + 1
    compressed = base + 7
    profiles = torch.stack((base, compressed), dim=0)
    ranked = profiles.unsqueeze(0).expand(N_RANKS, -1, -1, -1).contiguous()
    source = TensorSpec(
        "freqs_cos",
        list(ranked.shape),
        ranked.dtype,
        init_value=lambda: ranked.clone(),
    )

    shared = _make_shared_spec("freqs_cos", {"freqs_cos": source}, start_pos=0)
    actual = shared.init_value()

    assert shared.shape == [N_RANKS, 2, 6, 4]
    assert torch.equal(actual, ranked)
    assert torch.count_nonzero(actual[:, SWA_ROPE_PROFILE]) > 0
    assert torch.count_nonzero(actual[:, COMPRESSED_ROPE_PROFILE]) > 0


def test_runtime_and_golden_select_profiles_by_attention_kind():
    base = torch.full((3, 4), 11, dtype=torch.bfloat16)
    compressed = torch.full((3, 4), 29, dtype=torch.bfloat16)
    stacked = torch.stack((base, compressed), dim=0)

    assert _rope_profile_for_compress_ratio(0) == SWA_ROPE_PROFILE
    assert _rope_profile_for_compress_ratio(4) == COMPRESSED_ROPE_PROFILE
    assert _rope_profile_for_compress_ratio(128) == COMPRESSED_ROPE_PROFILE
    assert torch.equal(_rope_profile_for_kind(stacked, "swa"), base)
    assert torch.equal(_rope_profile_for_kind(stacked, "csa"), compressed)
    assert torch.equal(_rope_profile_for_kind(stacked, "hca"), compressed)
    with pytest.raises(ValueError, match="unsupported"):
        _rope_profile_for_compress_ratio(7)
    with pytest.raises(ValueError, match="unsupported"):
        _rope_profile_for_kind(stacked, "unknown")
