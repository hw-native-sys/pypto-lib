# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for golden.validation."""

import pytest

import torch
from golden.validation import validate_golden


class TestValidateGolden:
    """Tests for validate_golden() comparison logic."""

    def test_matching_tensors_pass(self):
        """Identical tensors should not raise."""
        t = torch.tensor([1.0, 2.0, 3.0])
        validate_golden({"out": t}, {"out": t.clone()})

    def test_within_tolerance_passes(self):
        """Tensors within rtol/atol tolerance should not raise."""
        actual = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([1.001, 2.002, 3.003])
        validate_golden({"out": actual}, {"out": expected}, rtol=1e-2, atol=1e-2)

    def test_exceeding_tolerance_raises(self):
        """Tensors exceeding tolerance should raise AssertionError."""
        actual = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([2.0, 3.0, 4.0])
        with pytest.raises(AssertionError, match="does not match golden"):
            validate_golden({"out": actual}, {"out": expected}, rtol=1e-5, atol=1e-5)

    def test_error_message_contains_details(self):
        """Error message should contain mismatch count and sample values."""
        actual = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([1.0, 200.0, 3.0, 400.0])
        with pytest.raises(AssertionError, match=r"Mismatched elements: 2/4") as exc_info:
            validate_golden({"out": actual}, {"out": expected}, rtol=1e-5, atol=1e-5)
        assert "actual=" in str(exc_info.value)
        assert "expected=" in str(exc_info.value)

    def test_multiple_outputs(self):
        """Multiple output tensors are all validated."""
        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0, 4.0])
        # Both match
        validate_golden(
            {"a": t1, "b": t2},
            {"a": t1.clone(), "b": t2.clone()},
        )

    def test_multiple_outputs_one_fails(self):
        """If one of multiple outputs fails, AssertionError is raised."""
        t1 = torch.tensor([1.0, 2.0])
        t2_actual = torch.tensor([3.0, 4.0])
        t2_expected = torch.tensor([30.0, 40.0])
        with pytest.raises(AssertionError, match="'b'"):
            validate_golden(
                {"a": t1, "b": t2_actual},
                {"a": t1.clone(), "b": t2_expected},
            )

    def test_zero_tensor_matches(self):
        """Zero tensors should match."""
        z = torch.zeros(10)
        validate_golden({"out": z}, {"out": z.clone()})

    def test_tolerance_boundary(self):
        """Test the exact boundary of tolerance."""
        actual = torch.tensor([1.0])
        # atol=0.1 means values within 0.1 of each other pass
        close_enough = torch.tensor([1.09])
        validate_golden({"out": actual}, {"out": close_enough}, rtol=0, atol=0.1)

        too_far = torch.tensor([1.11])
        with pytest.raises(AssertionError):
            validate_golden({"out": actual}, {"out": too_far}, rtol=0, atol=0.1)

    def test_bfloat16_tensors(self):
        """bfloat16 tensors should be comparable."""
        actual = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        validate_golden({"out": actual}, {"out": expected})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
