# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the checkpoint-independent numeric helpers of weights_flash.

Covers the MXFP4/MXFP8 dequantization grids and the fixture-equivalent INT8
per-output-channel requantization chain. The checkpoint-driven conversion
paths (layer stacking, EP/TP sharding) need the real DeepSeek-V4-Flash
checkpoint and are exercised by the driver ``--weights`` flow instead.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"
os.environ.setdefault("DEEPSEEK_V4_VARIANT", "flash")
sys.path.insert(0, str(_MODEL_DIR))

from weights_flash import (  # noqa: E402
    _e8m0_to_fp32,
    dequant_fp4,
    dequant_fp8_block,
    quant_int8_per_out_channel,
)


class TestE8M0Decode:
    def test_powers_of_two(self):
        """UE8M0 bytes decode to 2^(x - 127)."""
        s = _e8m0_to_fp32(torch.tensor([0, 126, 127, 128, 130], dtype=torch.uint8))
        assert torch.equal(s, torch.tensor([2.0**-127, 0.5, 1.0, 2.0, 8.0]))


class TestDequantFP4:
    def test_nibble_order_and_value_table(self):
        """Low nibble is the first element; e2m1 table with sign in bit 3."""
        # 0x21 -> low 1 (0.5), high 2 (1.0); 0x9F -> low 15 (-6.0), high 9 (-0.5)
        packed = torch.tensor([[0x21, 0x9F] + [0] * 14], dtype=torch.uint8).view(torch.int8)
        scale = torch.tensor([[129]], dtype=torch.uint8)  # 2^2 = 4.0 for the 32-group
        out = dequant_fp4(packed, scale)
        expect = torch.zeros(1, 32)
        expect[0, :4] = torch.tensor([2.0, 4.0, -24.0, -2.0])
        assert torch.equal(out, expect)

    def test_per_32_group_scales(self):
        """Each group of 32 unpacked elements uses its own scale."""
        packed = torch.full((1, 32), 0x22, dtype=torch.uint8).view(torch.int8)  # all 1.0
        scale = torch.tensor([[127, 128]], dtype=torch.uint8)  # groups: 1.0, 2.0
        out = dequant_fp4(packed, scale)
        assert torch.equal(out[0, :32], torch.ones(32))
        assert torch.equal(out[0, 32:], torch.full((32,), 2.0))

    def test_batched_expert_shape(self):
        """Batched [E, out, in/2] unpacks to [E, out, in]."""
        packed = torch.zeros(3, 4, 16, dtype=torch.int8)
        scale = torch.full((3, 4, 1), 127, dtype=torch.uint8)
        assert dequant_fp4(packed, scale).shape == (3, 4, 32)


class TestDequantFP8Block:
    def test_matches_manual_expansion(self):
        """128x128-block scales multiply the e4m3 payload."""
        torch.manual_seed(0)
        w = (torch.randn(256, 384) * 0.1).to(torch.float8_e4m3fn)
        scale = torch.tensor([[127, 128, 129], [130, 126, 127]], dtype=torch.uint8)
        out = dequant_fp8_block(w, scale)
        for i, j in [(0, 0), (127, 127), (128, 0), (255, 383), (100, 300)]:
            manual = float(w[i, j].float()) * 2.0 ** (int(scale[i // 128, j // 128]) - 127)
            assert float(out[i, j]) == manual


class TestQuantInt8PerOutChannel:
    def test_matches_fixture_chain(self):
        """Reproduces the fixture helpers: amax/127 scale, round->clamp->fp16->int8."""
        torch.manual_seed(1)
        w = torch.randn(16, 64)
        w_i8, scale = quant_int8_per_out_channel(w)
        amax = w.abs().amax(dim=1).clamp_min(1e-4)
        ref = torch.clamp(
            torch.round(w * (127.0 / amax).unsqueeze(1)).to(torch.int32), -127, 127
        ).to(torch.float16).to(torch.int8)
        assert torch.equal(w_i8, ref)
        assert torch.allclose(scale, amax / 127.0)
        assert w_i8.dtype == torch.int8 and scale.dtype == torch.float32

    def test_round_trip_error_bound(self):
        """Dequantized int8 stays within half a quant step of the input."""
        torch.manual_seed(2)
        w = torch.randn(8, 128) * 0.02
        w_i8, scale = quant_int8_per_out_channel(w)
        err = (w_i8.float() * scale.unsqueeze(1) - w).abs()
        assert (err <= scale.unsqueeze(1) * 0.5 + 1e-7).all()

    def test_zero_row_uses_eps_floor(self):
        """An all-zero output channel quantizes to zeros with the eps-floored scale."""
        w = torch.zeros(2, 32)
        w_i8, scale = quant_int8_per_out_channel(w)
        assert torch.equal(w_i8, torch.zeros(2, 32, dtype=torch.int8))
        assert torch.allclose(scale, torch.full((2,), 1e-4 / 127.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
