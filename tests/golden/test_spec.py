# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for golden.spec — TensorSpec and ScalarSpec."""

import ctypes

import pytest
import torch
from golden.spec import SUPPORTED_SCALAR_DTYPES, ScalarSpec, TensorSpec


class TestTensorSpecCreateTensor:
    """Tests for TensorSpec.create_tensor() with various init_value strategies."""

    def test_none_init_creates_zeros(self):
        """init_value=None produces a zero-filled tensor."""
        spec = TensorSpec("x", [4, 8], torch.float32)
        t = spec.create_tensor()
        assert t.shape == (4, 8)
        assert t.dtype == torch.float32
        assert torch.equal(t, torch.zeros(4, 8, dtype=torch.float32))

    def test_int_init_creates_full(self):
        """init_value=int/float fills every element with that constant."""
        spec = TensorSpec("x", [3, 5], torch.float32, init_value=7)
        t = spec.create_tensor()
        assert torch.equal(t, torch.full((3, 5), 7, dtype=torch.float32))

    def test_tensor_init_uses_directly(self):
        """init_value=torch.Tensor uses the tensor directly, casting dtype."""
        data = torch.arange(0, 6, dtype=torch.float64).reshape(2, 3)
        spec = TensorSpec("x", [2, 3], torch.float32, init_value=data)
        t = spec.create_tensor()
        assert t.dtype == torch.float32
        assert torch.allclose(t, data.float())

    @pytest.mark.parametrize("factory", [torch.randn, torch.rand, torch.zeros, torch.ones])
    def test_torch_factory_callables(self, factory):
        """Each torch factory callable produces a tensor with correct shape/dtype."""
        spec = TensorSpec("x", [4, 4], torch.float32, init_value=factory)
        t = spec.create_tensor()
        assert t.shape == (4, 4)
        assert t.dtype == torch.float32

    def test_custom_callable(self):
        """init_value=custom_fn calls with no args and casts dtype."""
        def make_data():
            return torch.arange(0, 4, dtype=torch.float64)

        spec = TensorSpec("x", [4], torch.float32, init_value=make_data)
        t = spec.create_tensor()
        assert t.dtype == torch.float32
        assert torch.allclose(t, torch.arange(0, 4, dtype=torch.float32))

    def test_unsupported_init_value_raises(self):
        """Unsupported init_value type raises TypeError."""
        spec = TensorSpec("x", [4], torch.float32, init_value="invalid")
        with pytest.raises(TypeError, match="Unsupported init_value type"):
            spec.create_tensor()

    def test_is_output_flag(self):
        """is_output flag is stored correctly and defaults to False."""
        spec_in = TensorSpec("a", [4], torch.float32)
        spec_out = TensorSpec("b", [4], torch.float32, is_output=True)
        assert spec_in.is_output is False
        assert spec_out.is_output is True

    def test_tensor_init_ignores_spec_shape(self):
        """Pin current behavior: when init_value is a Tensor, spec.shape is NOT enforced.

        The spec says [2, 2] but the supplied tensor is [3]. ``create_tensor``
        returns the supplied tensor cast to dtype — shape mismatch is not
        validated. Kept as a regression pin; revisit if we decide to enforce
        shape matching in ``TensorSpec.create_tensor``.
        """
        data = torch.tensor([1.0, 2.0, 3.0])
        spec = TensorSpec("x", [2, 2], torch.float32, init_value=data)
        t = spec.create_tensor()
        assert t.shape == (3,)
        assert t.shape != tuple(spec.shape)


# ---------------------------------------------------------------------------
# ScalarSpec
# ---------------------------------------------------------------------------


class TestScalarSpecConstruction:
    """Constructor-time validation: dtype and value compatibility."""

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported dtype"):
            ScalarSpec("x", torch.float64, 1.0)

    def test_int_dtype_rejects_float(self):
        with pytest.raises(ValueError, match="requires int value"):
            ScalarSpec("x", torch.int32, 1.5)

    def test_int_dtype_rejects_bool(self):
        # bool is a subclass of int — must be rejected so dtype is unambiguous
        with pytest.raises(ValueError, match="requires int value"):
            ScalarSpec("x", torch.int32, True)

    def test_int_dtype_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            ScalarSpec("x", torch.int8, 200)

    def test_uint8_negative_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            ScalarSpec("x", torch.uint8, -1)

    def test_bool_dtype_requires_bool(self):
        with pytest.raises(ValueError, match="requires bool value"):
            ScalarSpec("flag", torch.bool, 1)

    def test_fp32_rejects_bool(self):
        with pytest.raises(ValueError, match="requires int or float"):
            ScalarSpec("x", torch.float32, True)

    def test_fp32_accepts_int_and_float(self):
        ScalarSpec("a", torch.float32, 1)
        ScalarSpec("b", torch.float32, 1.5)

    def test_supported_dtypes_complete(self):
        """All advertised dtypes can be constructed with a representative value."""
        sample_value = {
            torch.int8: 1, torch.int32: 1, torch.int64: 1, torch.uint8: 1,
            torch.bool: True,
            torch.float32: 1.0, torch.float16: 1.0, torch.bfloat16: 1.0,
        }
        assert set(SUPPORTED_SCALAR_DTYPES) == set(sample_value)
        for dtype, value in sample_value.items():
            ScalarSpec("x", dtype, value)


class TestScalarSpecToCtypes:
    """ScalarSpec.to_ctypes returns a ctypes._SimpleCData instance with the
    right type and bit-pattern for each supported dtype."""

    @pytest.mark.parametrize("dtype, ctype, value", [
        (torch.int8,    ctypes.c_int8,  -42),
        (torch.int32,   ctypes.c_int32, 123456),
        (torch.int64,   ctypes.c_int64, 1 << 40),
        (torch.uint8,   ctypes.c_uint8, 200),
        (torch.float32, ctypes.c_float, 0.125),
    ])
    def test_direct_dtypes(self, dtype, ctype, value):
        c = ScalarSpec("x", dtype, value).to_ctypes()
        assert isinstance(c, ctype)
        if dtype is torch.float32:
            assert c.value == pytest.approx(value)
        else:
            assert c.value == value

    def test_bool(self):
        c_true = ScalarSpec("f", torch.bool, True).to_ctypes()
        c_false = ScalarSpec("f", torch.bool, False).to_ctypes()
        assert isinstance(c_true, ctypes.c_bool)
        assert c_true.value is True
        assert c_false.value is False

    def test_fp16_bit_pattern(self):
        # 1.0 in IEEE-754 fp16 is 0x3C00
        c = ScalarSpec("a", torch.float16, 1.0).to_ctypes()
        assert isinstance(c, ctypes.c_uint16)
        assert c.value == 0x3C00

    def test_bf16_bit_pattern(self):
        # 1.0 in bf16 is 0x3F80 (top 16 bits of fp32 1.0 = 0x3F800000)
        c = ScalarSpec("a", torch.bfloat16, 1.0).to_ctypes()
        assert isinstance(c, ctypes.c_uint16)
        assert c.value == 0x3F80

    def test_fp16_negative(self):
        # -2.0 fp16 is 0xC000 (sign=1, exponent=0x10, mantissa=0)
        c = ScalarSpec("a", torch.float16, -2.0).to_ctypes()
        assert c.value == 0xC000

    def test_bf16_negative(self):
        # -2.0 bf16 = top16(0xC0000000) = 0xC000
        c = ScalarSpec("a", torch.bfloat16, -2.0).to_ctypes()
        assert c.value == 0xC000


class TestScalarSpecValueIsTensor:
    """After construction, ``spec.value`` is always a 0-dim torch.Tensor with
    dtype matching ``spec.dtype`` — even when the user passed a python scalar."""

    @pytest.mark.parametrize("dtype, value", [
        (torch.int8,     -42),
        (torch.int32,    1234),
        (torch.int64,    1 << 40),
        (torch.uint8,    200),
        (torch.bool,     True),
        (torch.float32,  0.125),
        (torch.float16,  0.5),
        (torch.bfloat16, -1.0),
    ])
    def test_python_value_coerced_to_tensor(self, dtype, value):
        s = ScalarSpec("x", dtype, value)
        assert isinstance(s.value, torch.Tensor)
        assert s.value.ndim == 0
        assert s.value.dtype == dtype

    def test_constructor_accepts_tensor(self):
        t = torch.tensor(2.5, dtype=torch.float32)
        s = ScalarSpec("x", torch.float32, t)
        # Stored tensor must be the one passed in (no implicit copy).
        assert s.value is t

    def test_constructor_rejects_non_zero_dim_tensor(self):
        with pytest.raises(ValueError, match="must be 0-dim"):
            ScalarSpec("x", torch.float32, torch.tensor([1.0, 2.0]))

    def test_constructor_rejects_dtype_mismatch_tensor(self):
        with pytest.raises(ValueError, match="does not match dtype"):
            ScalarSpec("x", torch.float32, torch.tensor(1.0, dtype=torch.float16))


class TestScalarSpecToPython:
    """ScalarSpec.to_python returns a value usable inside golden_fn."""

    def test_int_returns_int(self):
        assert ScalarSpec("x", torch.int32, 42).to_python() == 42

    def test_bool_returns_bool(self):
        assert ScalarSpec("x", torch.bool, True).to_python() is True

    def test_fp32_returns_float(self):
        v = ScalarSpec("x", torch.float32, 0.125).to_python()
        assert isinstance(v, float)
        assert v == pytest.approx(0.125)

    def test_fp16_round_trips_through_half_precision(self):
        # 1/3 is not exactly representable in fp16; to_python must return the
        # rounded value so golden math sees what the device sees.
        spec = ScalarSpec("x", torch.float16, 1.0 / 3.0)
        v = spec.to_python()
        expected = float(torch.tensor([1.0 / 3.0], dtype=torch.float16).item())
        assert v == expected

    def test_bf16_round_trips_through_bf16(self):
        spec = ScalarSpec("x", torch.bfloat16, 1.0 / 3.0)
        v = spec.to_python()
        expected = float(torch.tensor([1.0 / 3.0], dtype=torch.bfloat16).item())
        assert v == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
