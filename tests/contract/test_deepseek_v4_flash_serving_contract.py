# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest

from models.deepseek_v4_flash_mtp import config
from models.deepseek_v4_flash_mtp.serving_contract import (
    DEEPSEEK_V4_FLASH_SERVING_CONTRACT as CONTRACT,
)


def test_deepseek_v4_flash_serving_capabilities() -> None:
    assert CONTRACT.schema_version == "1"
    assert CONTRACT.prefill_tile_tokens == 128
    assert CONTRACT.max_prefill_tokens_per_request == 8192
    assert CONTRACT.max_prefill_requests_per_partition == 1
    assert CONTRACT.requires_homogeneous_prefill_decode is True


@pytest.mark.parametrize(
    ("active_tokens", "expected"),
    [(1, 128), (128, 128), (129, 256), (8191, 8192), (8192, 8192)],
)
def test_deepseek_v4_flash_prefill_padding(
    active_tokens: int,
    expected: int,
) -> None:
    assert CONTRACT.padded_prefill_tokens(active_tokens) == expected


@pytest.mark.parametrize("active_tokens", [0, -1, 8193])
def test_deepseek_v4_flash_prefill_padding_rejects_invalid_extents(
    active_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        CONTRACT.padded_prefill_tokens(active_tokens)


@pytest.mark.parametrize("active_tokens", [True, 128.5, "128"])
def test_deepseek_v4_flash_prefill_padding_rejects_non_integer_extents(
    active_tokens: object,
) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        CONTRACT.padded_prefill_tokens(active_tokens)


def test_deepseek_v4_flash_contract_matches_kernel_prefill_shape() -> None:
    assert config.PREFILL_BATCH == CONTRACT.max_prefill_requests_per_partition
    assert config.PREFILL_SEQ == CONTRACT.prefill_tile_tokens
