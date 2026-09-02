# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[2] / "models" / "deepseek_v4_pro" / "mx_utils.py"
_SPEC = importlib.util.spec_from_file_location("deepseek_v4_pro_mx_utils", _MODULE_PATH)
mx = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(mx)


def _pack_checkpoint_nibbles(indices_nk):
    low = indices_nk[..., 0::2] & 0x0F
    high = indices_nk[..., 1::2] & 0x0F
    return (low | (high << 4)).to(torch.uint8)


def test_mxfp4_bridge_preserves_all_nibble_codes_and_signed_zero():
    indices = torch.arange(16, dtype=torch.uint8).repeat(4).reshape(1, 64).repeat(16, 1)
    packed = _pack_checkpoint_nibbles(indices)
    scale = torch.full((16, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

    data, packed_scale = mx.mxfp4_to_mxfp8_weight_kn(packed, scale)

    assert data.shape == (64, 16)
    assert data.contiguous().view(torch.uint8)[:, 0].tolist() == mx.NIBBLE_LUT * 4
    assert data.contiguous().view(torch.uint8)[8, 0].item() == 0x80
    assert torch.all(mx.unpack_b_scale(packed_scale.view(torch.uint8)) == 127)


def test_mxfp4_bridge_matches_checkpoint_dequantization_for_w13_and_w2():
    for n, k in ((32, 64), (64, 128)):
        indices = torch.arange(n * k, dtype=torch.int64).reshape(n, k) % 16
        packed = _pack_checkpoint_nibbles(indices)
        scale_codes = 100 + (torch.arange(n * (k // 32), dtype=torch.uint8).reshape(n, k // 32) % 40)

        data_kn, packed_scale = mx.mxfp4_to_mxfp8_weight_kn(
            packed,
            scale_codes.view(torch.float8_e8m0fnu),
        )
        logical_scale = mx.unpack_b_scale(packed_scale.view(torch.uint8))

        values_nk = mx.nibble_indices_to_fp8(indices).float()
        expected = values_nk * mx.e8m0_codes_to_fp32(scale_codes).repeat_interleave(32, dim=-1)
        actual = data_kn.float() * mx.e8m0_codes_to_fp32(logical_scale).repeat_interleave(32, dim=0)
        torch.testing.assert_close(actual, expected.transpose(0, 1), rtol=0, atol=0)
        assert torch.equal(logical_scale, scale_codes.transpose(0, 1))


def test_mxfp4_bridge_keeps_leading_expert_dimensions_independent():
    indices = torch.arange(2 * 16 * 64, dtype=torch.int64).reshape(2, 16, 64) % 16
    packed = _pack_checkpoint_nibbles(indices)
    scale_codes = torch.tensor([[[121, 122]] * 16, [[137, 138]] * 16], dtype=torch.uint8)

    data, packed_scale = mx.mxfp4_to_mxfp8_weight_kn(
        packed,
        scale_codes.view(torch.float8_e8m0fnu),
    )

    assert data.shape == (2, 64, 16)
    assert packed_scale.shape == (2, 2, 16)
    for expert in range(2):
        assert torch.equal(mx.unpack_b_scale(packed_scale[expert].view(torch.uint8)), scale_codes[expert].T)


def test_mx_scale_pack_round_trips():
    a = torch.arange(32 * 4, dtype=torch.uint8).reshape(32, 4)
    b = torch.arange(4 * 32, dtype=torch.uint8).reshape(4, 32)
    batched_b = torch.arange(3 * 4 * 32, dtype=torch.int32).to(torch.uint8).reshape(3, 4, 32)
    assert torch.equal(mx.unpack_a_scale(mx.pack_a_scale(a)), a)
    assert torch.equal(mx.unpack_b_scale(mx.pack_b_scale(b)), b)
    assert torch.equal(
        mx.unpack_b_scale_batched(mx.pack_b_scale_batched(batched_b)), batched_b
    )


def test_host_weight_quant_returns_cube_layout_and_packed_e8m0_scale():
    weight_nk = torch.linspace(-2.0, 2.0, 32 * 64).reshape(32, 64)

    data_kn, packed_scale = mx.host_quant_mxfp8_weight_kn(weight_nk)

    logical_scale = mx.unpack_b_scale(packed_scale.view(torch.uint8))
    restored = data_kn.float() * mx.e8m0_codes_to_fp32(logical_scale).repeat_interleave(
        mx.MX_GROUP, dim=0
    )
    expected_data, expected_scale = mx.host_quant_mxfp8(weight_nk, return_e8m0=True)
    expected = expected_data.float() * mx.e8m0_codes_to_fp32(
        expected_scale.view(torch.uint8)
    ).repeat_interleave(mx.MX_GROUP, dim=-1)
    assert data_kn.shape == (64, 32)
    assert packed_scale.shape == (2, 32)
    torch.testing.assert_close(restored, expected.T, rtol=0, atol=0)


def test_host_quant_uses_ascend_ocp_shared_exponents():
    x = torch.zeros(1, 160)
    for group, value in enumerate((0.0, 0.99, 1.0, 1.99, 2.0)):
        x[:, group * mx.MX_GROUP : (group + 1) * mx.MX_GROUP] = value

    _, scale = mx.host_quant_mxfp8(x, return_e8m0=True)

    assert scale.view(torch.uint8).tolist() == [[0, 118, 119, 119, 120]]
