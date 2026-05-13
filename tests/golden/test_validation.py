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
from golden.validation import bf16_allclose_or_ulp, topk_pair_compare, validate_golden


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

    def test_missing_golden_key_raises_keyerror(self):
        """If golden lacks a key present in outputs, KeyError surfaces directly."""
        actual = torch.tensor([1.0])
        with pytest.raises(KeyError):
            validate_golden({"missing": actual}, {"other": actual})

    def test_shape_mismatch_raises(self):
        """Shape mismatch (non-broadcastable) raises."""
        actual = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises((RuntimeError, AssertionError)):
            validate_golden({"out": actual}, {"out": expected})

    def test_nan_values_fail(self):
        """NaN values should fail comparison (allclose treats NaN != NaN)."""
        actual = torch.tensor([1.0, float("nan"), 3.0])
        expected = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(AssertionError, match="does not match golden"):
            validate_golden({"out": actual}, {"out": expected})

    def test_default_tolerances_catch_large_diff(self):
        """Default rtol/atol=1e-5 should reject clearly different values."""
        actual = torch.tensor([1.0])
        expected = torch.tensor([1.1])
        with pytest.raises(AssertionError, match="does not match golden"):
            validate_golden({"out": actual}, {"out": expected})


class TestCompareFnDispatch:
    """Tests for the compare_fn override path in validate_golden."""

    def test_custom_pass_skips_default(self):
        """A compare_fn returning True bypasses the default allclose check."""
        actual = torch.tensor([1.0, 2.0])
        expected = torch.tensor([100.0, 200.0])  # would fail under allclose

        def always_pass(a, e, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            return True, ""

        validate_golden(
            {"out": actual}, {"out": expected},
            compare_fn={"out": always_pass},
        )

    def test_custom_fail_raises_with_detail(self):
        """A compare_fn returning False raises AssertionError carrying the detail."""
        t = torch.tensor([1.0])

        def always_fail(a, e, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            return False, "    custom-detail-marker"

        with pytest.raises(AssertionError, match="custom-detail-marker"):
            validate_golden(
                {"out": t}, {"out": t.clone()},
                compare_fn={"out": always_fail},
            )

    def test_custom_receives_full_context(self):
        """The compare_fn receives all outputs, golden, inputs, and tolerances."""
        captured = {}

        def capture(a, e, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            captured["actual_outputs"] = set(actual_outputs)
            captured["expected_outputs"] = set(expected_outputs)
            captured["inputs"] = set(inputs)
            captured["rtol"] = rtol
            captured["atol"] = atol
            return True, ""

        validate_golden(
            {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
            {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
            rtol=1e-2, atol=1e-3,
            compare_fn={"a": capture},
            inputs={"x": torch.tensor([0.0])},
        )
        assert captured["actual_outputs"] == {"a", "b"}
        assert captured["expected_outputs"] == {"a", "b"}
        assert captured["inputs"] == {"x"}
        assert captured["rtol"] == 1e-2
        assert captured["atol"] == 1e-3

    def test_partial_override_other_uses_default(self):
        """Names not in compare_fn still go through the default allclose path."""
        ok = torch.tensor([1.0])
        bad_actual = torch.tensor([1.0])
        bad_expected = torch.tensor([5.0])

        def always_pass(a, e, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            return True, ""

        # 'a' overridden to pass, 'b' uses default and fails -> overall fail.
        with pytest.raises(AssertionError, match="'b'"):
            validate_golden(
                {"a": bad_actual, "b": bad_actual},
                {"a": bad_expected, "b": bad_expected},
                compare_fn={"a": always_pass},
            )
        # Sanity: with both overridden it passes.
        validate_golden(
            {"a": bad_actual, "b": bad_actual},
            {"a": bad_expected, "b": bad_expected},
            compare_fn={"a": always_pass, "b": always_pass},
        )
        # Sanity: defaults pass when tensors match.
        validate_golden({"a": ok}, {"a": ok.clone()})


class TestBf16AllcloseOrUlp:
    """Tests for the BF16 ULP comparator helper."""

    def test_default_one_ulp_passes(self):
        """Adjacent BF16 values pass even when normal tolerance fails."""
        actual = torch.tensor([1.0078125], dtype=torch.bfloat16)
        expected = torch.tensor([1.0], dtype=torch.bfloat16)
        validate_golden(
            {"out": actual},
            {"out": expected},
            rtol=0.0,
            atol=0.0,
            compare_fn={"out": bf16_allclose_or_ulp()},
        )

    def test_max_ulp_parameter_controls_allowance(self):
        """A two-ULP difference fails with max_ulp=1 and passes with max_ulp=2."""
        actual = torch.tensor([1.015625], dtype=torch.bfloat16)
        expected = torch.tensor([1.0], dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="after 1-ULP allowance"):
            validate_golden(
                {"out": actual},
                {"out": expected},
                rtol=0.0,
                atol=0.0,
                compare_fn={"out": bf16_allclose_or_ulp(max_ulp=1)},
            )
        validate_golden(
            {"out": actual},
            {"out": expected},
            rtol=0.0,
            atol=0.0,
            compare_fn={"out": bf16_allclose_or_ulp(max_ulp=2)},
        )

    def test_non_bf16_tensors_fail_with_clear_message(self):
        """The helper is only for BF16 tensors."""
        actual = torch.tensor([1.0], dtype=torch.float32)
        expected = torch.tensor([1.0], dtype=torch.float32)
        with pytest.raises(AssertionError, match="requires BF16 tensors"):
            validate_golden(
                {"out": actual},
                {"out": expected},
                compare_fn={"out": bf16_allclose_or_ulp()},
            )

    def test_nan_values_do_not_pass_via_ulp_fallback(self):
        """NaN bit patterns should not bypass torch.isclose semantics."""
        actual = torch.tensor([float("nan")], dtype=torch.bfloat16)
        expected = torch.tensor([float("nan")], dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="after 1-ULP allowance"):
            validate_golden(
                {"out": actual},
                {"out": expected},
                compare_fn={"out": bf16_allclose_or_ulp()},
            )

    def test_negative_max_ulp_rejected(self):
        """Negative ULP allowance is invalid."""
        with pytest.raises(ValueError, match="max_ulp must be non-negative"):
            bf16_allclose_or_ulp(max_ulp=-1)


class TestTopkPairCompare:
    """Tests for the topk_pair_compare helper."""

    def test_legal_tie_break_passes(self):
        """Same picked-score set with different idx ordering passes."""
        idx_actual = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        idx_expected = torch.tensor([[2, 1, 0]], dtype=torch.int32)
        # Both sides report the same set of picked vals (sorted desc).
        vals = torch.tensor([[3.0, 2.0, 1.0]])

        cmp = topk_pair_compare("vals")
        ok, _ = cmp(
            idx_actual, idx_expected,
            actual_outputs={"vals": vals},
            expected_outputs={"vals": vals.clone()},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert ok

    def test_real_miss_fails(self):
        """If one side picked a strictly lower-scoring candidate, fail."""
        idx_actual = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        idx_expected = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        vals_actual = torch.tensor([[3.0, 2.0, 0.5]])    # picked a 0.5
        vals_expected = torch.tensor([[3.0, 2.0, 1.0]])  # had a 1.0 there

        cmp = topk_pair_compare("vals")
        ok, detail = cmp(
            idx_actual, idx_expected,
            actual_outputs={"vals": vals_actual},
            expected_outputs={"vals": vals_expected},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert not ok
        assert "top-k pair mismatch" in detail
        assert "0.5" in detail and "1.0" in detail

    def test_within_tolerance_passes(self):
        """Score sets within rtol/atol pass even if not bit-exact."""
        idx = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        vals_a = torch.tensor([[3.0, 2.0, 1.0]])
        vals_b = torch.tensor([[3.0005, 2.0005, 1.0005]])

        cmp = topk_pair_compare("vals")
        ok, _ = cmp(
            idx, idx,
            actual_outputs={"vals": vals_a},
            expected_outputs={"vals": vals_b},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert ok

    def test_multi_batch_isolated(self):
        """Per-batch sort: a swap inside one batch should not contaminate another."""
        idx_actual = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        idx_expected = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        vals = torch.tensor([[5.0, 5.0], [2.0, 1.0]])  # batch 0 has a tie

        cmp = topk_pair_compare("vals")
        ok, _ = cmp(
            idx_actual, idx_expected,
            actual_outputs={"vals": vals},
            expected_outputs={"vals": vals.clone()},
            inputs={},
            rtol=1e-5, atol=1e-5,
        )
        assert ok

    def test_function_name_for_logging(self):
        """The returned cmp exposes __name__ for log labelling."""
        cmp = topk_pair_compare("vals")
        assert cmp.__name__ == "topk_pair_compare"

    def test_integrated_with_validate_golden(self):
        """End-to-end: validate_golden uses the helper via compare_fn."""
        idx_actual = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        idx_expected = torch.tensor([[2, 1, 0]], dtype=torch.int32)
        vals = torch.tensor([[3.0, 2.0, 1.0]])
        validate_golden(
            {"idx": idx_actual, "vals": vals},
            {"idx": idx_expected, "vals": vals.clone()},
            rtol=1e-3, atol=1e-3,
            compare_fn={"idx": topk_pair_compare("vals")},
        )

    def test_misconfigured_vals_name_returns_friendly_error(self):
        """A typo in vals_name should yield a clear failure, not a KeyError."""
        idx = torch.tensor([[0, 1]], dtype=torch.int32)
        vals = torch.tensor([[2.0, 1.0]])
        cmp = topk_pair_compare("typo_vals")
        ok, detail = cmp(
            idx, idx,
            actual_outputs={"vals": vals},
            expected_outputs={"vals": vals.clone()},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert not ok
        assert "misconfigured" in detail
        assert "typo_vals" in detail

    def test_ndim_greater_than_two(self):
        """Helper handles vals with leading rank > 1 (e.g. [B, H, K])."""
        idx = torch.zeros((2, 3, 4), dtype=torch.int32)
        vals_a = torch.arange(24.0).reshape(2, 3, 4)
        # Permute the last dim per row — same set, different order, must pass.
        vals_b = vals_a.flip(dims=[-1]).contiguous()
        cmp = topk_pair_compare("vals")
        ok, _ = cmp(
            idx, idx,
            actual_outputs={"vals": vals_a},
            expected_outputs={"vals": vals_b},
            inputs={},
            rtol=1e-5, atol=1e-5,
        )
        assert ok

    def test_bfloat16_vals(self):
        """BF16 vals work — helper promotes to float32 internally."""
        idx = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        vals = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.bfloat16)
        cmp = topk_pair_compare("vals")
        ok, _ = cmp(
            idx, idx,
            actual_outputs={"vals": vals},
            expected_outputs={"vals": vals.clone()},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert ok


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
