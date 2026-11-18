# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for golden validation helpers."""

import pytest

import torch
from golden.validation import (
    mapped_pool_ratio_allclose,
    mapped_pool_ratio_reldiff,
    ratio_allclose,
    ratio_reldiff,
    topk_pair_compare,
    validate_golden,
)


class TestMappedPoolRatioAllclose:
    """Tests for allocator-mapped pool validation."""

    @staticmethod
    def _call(comparator, actual, expected, mapping):
        return comparator(
            actual,
            expected,
            actual_outputs={"pool": actual},
            expected_outputs={"pool": expected},
            inputs={"mapping": mapping},
            rtol=0.0,
            atol=0.0,
        )

    def test_ignores_matching_nonfinite_unmapped_rows(self):
        expected = torch.zeros(2, 2, 2, dtype=torch.float32)
        expected[1] = float("-inf")
        actual = expected.clone()
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(1,),
            block_size=2,
            atol=0.0,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([0], dtype=torch.int64),
        )

        assert ok, detail

    def test_rejects_nonfinite_mapped_rows(self):
        expected = torch.zeros(2, 2, 2, dtype=torch.float32)
        actual = expected.clone()
        actual[0, 0, 0] = float("nan")
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(1,),
            block_size=2,
            atol=0.0,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([0], dtype=torch.int64),
        )

        assert not ok
        assert "mapped rows" in detail
        assert "non-finite" in detail

    def test_accepts_fp8_pool(self):
        expected = torch.zeros(2, 2, 2, dtype=torch.float32).to(torch.float8_e4m3fn)
        actual = expected.clone()
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(1,),
            block_size=2,
            atol=0.0,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([0], dtype=torch.int64),
        )

        assert ok, detail

    def test_rejects_nonfinite_fp8_mapped_rows(self):
        expected = torch.zeros(2, 2, 2, dtype=torch.float32).to(torch.float8_e4m3fn)
        actual = expected.clone()
        actual.view(torch.uint8)[0, 0, 0] = 0x7F
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(1,),
            block_size=2,
            atol=0.0,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([0], dtype=torch.int64),
        )

        assert not ok
        assert "non-finite" in detail

    def test_rejects_rank_local_unmapped_write(self):
        expected = torch.zeros(2, 2, 2, 1, dtype=torch.float32)
        actual = expected.clone()
        actual[1, 1, 1, 0] = 1.0
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(2, 1),
            block_size=2,
            leading_rank_axis=True,
            atol=0.0,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([[0], [1]], dtype=torch.int64),
        )

        assert not ok
        assert "rank=1" in detail
        assert "row=3" in detail

    @pytest.mark.parametrize(
        ("mapping", "detail_fragment"),
        [
            (torch.tensor([-2], dtype=torch.int64), "only -1 is a negative sentinel"),
            (torch.tensor([4], dtype=torch.int64), "physical row range [0, 4)"),
            (torch.tensor([0.0], dtype=torch.float32), "must have an integer dtype"),
        ],
    )
    def test_rejects_invalid_mapping_values_and_dtype(self, mapping, detail_fragment):
        expected = torch.zeros(2, 2, 1, dtype=torch.float32)
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(1,),
            block_size=2,
            atol=0.0,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(comparator, expected.clone(), expected, mapping)

        assert not ok
        assert detail_fragment in detail

    def test_rejects_mapping_shape_mismatch(self):
        expected = torch.zeros(2, 2, 1, dtype=torch.float32)
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(2,),
            block_size=2,
            atol=0.0,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(
            comparator,
            expected.clone(),
            expected,
            torch.tensor([0], dtype=torch.int64),
        )

        assert not ok
        assert "must have shape (2,)" in detail

    def test_rejects_duplicate_rows_within_rank(self):
        expected = torch.zeros(2, 2, 1, dtype=torch.float32)
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(2,),
            block_size=2,
            atol=0.0,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(
            comparator,
            expected.clone(),
            expected,
            torch.tensor([1, 1], dtype=torch.int64),
        )

        assert not ok
        assert "duplicate physical row 1" in detail

    def test_allows_same_physical_row_on_different_ranks(self):
        expected = torch.zeros(2, 2, 2, 1, dtype=torch.float32)
        actual = expected.clone()
        actual[:, 0, 0, 0] = 0.5
        comparator = mapped_pool_ratio_allclose(
            "mapping",
            mapping_shape=(2, 1),
            block_size=2,
            leading_rank_axis=True,
            atol=0.5,
            rtol=0.0,
            max_error_ratio=0.0,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([[0], [0]], dtype=torch.int64),
        )

        assert ok, detail

    def test_reldiff_allows_bounded_bfloat16_steps_on_mapped_rows(self):
        expected = torch.zeros(2, 2, 4, dtype=torch.bfloat16)
        expected[0, 0] = torch.tensor([1.0, 0.5, 0.25, 0.125])
        actual = expected.clone()
        actual[0, 0] = torch.tensor([1.0078125, 0.50390625, 0.251953125, 0.1259765625])
        comparator = mapped_pool_ratio_reldiff(
            "mapping", mapping_shape=(1,), block_size=2,
            diff_thd=0.01, pct_thd=0.05,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([0], dtype=torch.int64),
        )

        assert ok, detail

    def test_reldiff_rejects_excess_mapped_outliers(self):
        expected = torch.ones(1, 1, 100, dtype=torch.float32)
        actual = expected.clone()
        actual[0, 0, :6] = 1.1
        comparator = mapped_pool_ratio_reldiff(
            "mapping", mapping_shape=(1,), block_size=1,
            diff_thd=0.01, pct_thd=0.05,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([0], dtype=torch.int64),
        )

        assert not ok
        assert "error_count=6/100" in detail

    def test_reldiff_rejects_unmapped_mutation(self):
        expected = torch.zeros(2, 2, 1, dtype=torch.float32)
        actual = expected.clone()
        actual[1, 1, 0] = 1.0
        comparator = mapped_pool_ratio_reldiff(
            "mapping", mapping_shape=(1,), block_size=2,
            diff_thd=0.01, pct_thd=0.05,
        )

        ok, detail = self._call(
            comparator,
            actual,
            expected,
            torch.tensor([0], dtype=torch.int64),
        )

        assert not ok
        assert "unmapped physical pool row changed" in detail


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
        """Mismatch at a position where a_vals breaks descending order → fail."""
        idx_actual   = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        idx_expected = torch.tensor([[0, 1, 3]], dtype=torch.int32)
        # Pair (1, 2): 0.5 < 1.0 — kernel's own output is not descending at the
        # mismatched position, so the pick at pos 2 cannot be a legal tie-swap.
        a_vals = torch.tensor([[3.0, 0.5, 1.0]])

        cmp = topk_pair_compare("vals")
        ok, detail = cmp(
            idx_actual, idx_expected,
            actual_outputs={"vals": a_vals},
            expected_outputs={"vals": a_vals.clone()},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert not ok
        assert "top-k idx mismatch" in detail
        assert "actual_idx=2" in detail
        assert "expected_idx=3" in detail
        assert "[0,2]" in detail  # multi-dim coord

    def test_idx_match_short_circuits_without_vals_check(self):
        """When idx matches position-wise, vals are not consulted at all."""
        idx = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        # vals differ wildly, but the comparator never looks at vals when idx matches.
        vals_a = torch.tensor([[3.0, 2.0, 1.0]])
        vals_b = torch.tensor([[100.0, -1.0, 50.0]])

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

    def test_ndim_greater_than_two_uses_multi_dim_coord(self):
        """Failure diagnostics use original tensor axes for the coordinate."""
        # Shape [2, 1, 3]: batch 1 has a mismatch at last-dim position 2
        # with a_vals broken across that position.
        idx_actual   = torch.tensor([[[0, 1, 2]], [[0, 1, 2]]], dtype=torch.int32)
        idx_expected = torch.tensor([[[0, 1, 2]], [[0, 1, 3]]], dtype=torch.int32)
        a_vals = torch.tensor([[[3.0, 2.0, 1.0]], [[3.0, 0.5, 1.0]]])

        cmp = topk_pair_compare("vals")
        ok, detail = cmp(
            idx_actual, idx_expected,
            actual_outputs={"vals": a_vals},
            expected_outputs={"vals": a_vals.clone()},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert not ok
        assert "[1,0,2]" in detail  # original-axis coord

    def test_dim_parameter(self):
        """Top-k sorted along a non-last axis is handled via ``dim``."""
        # Shape [3, 4]: top-k along dim=0 with descending order per column.
        # Column 0 has a tie 8.0 == 8.0 between rows 1 and 2 — kernel may
        # swap their idx legally.
        idx_actual = torch.tensor(
            [[0, 0, 0, 0],
             [1, 1, 1, 1],
             [2, 2, 2, 2]], dtype=torch.int32)
        idx_expected = torch.tensor(
            [[0, 0, 0, 0],
             [2, 1, 1, 1],
             [1, 2, 2, 2]], dtype=torch.int32)
        a_vals = torch.tensor(
            [[9.0, 9.0, 9.0, 9.0],
             [8.0, 8.0, 8.0, 8.0],
             [8.0, 7.0, 7.0, 7.0]])

        cmp = topk_pair_compare("vals", dim=0)
        ok, _ = cmp(
            idx_actual, idx_expected,
            actual_outputs={"vals": a_vals},
            expected_outputs={"vals": a_vals.clone()},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert ok

    def test_descending_false_passes_on_ascending_tie(self):
        """Ascending top-k passes when a_vals is ascending across mismatches."""
        idx_actual   = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        idx_expected = torch.tensor([[2, 1, 0]], dtype=torch.int32)
        a_vals = torch.tensor([[1.0, 2.0, 3.0]])  # ascending

        cmp = topk_pair_compare("vals", descending=False)
        ok, _ = cmp(
            idx_actual, idx_expected,
            actual_outputs={"vals": a_vals},
            expected_outputs={"vals": a_vals.clone()},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert ok

    def test_descending_false_broken_fails(self):
        """Ascending top-k with order broken at a mismatch fails."""
        idx_actual   = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        idx_expected = torch.tensor([[0, 1, 3]], dtype=torch.int32)
        # Pair (1, 2): 3.0 > 2.0 — ascending broken at the mismatch.
        a_vals = torch.tensor([[1.0, 3.0, 2.0]])

        cmp = topk_pair_compare("vals", descending=False)
        ok, detail = cmp(
            idx_actual, idx_expected,
            actual_outputs={"vals": a_vals},
            expected_outputs={"vals": a_vals.clone()},
            inputs={},
            rtol=1e-3, atol=1e-3,
        )
        assert not ok
        assert "ascending" in detail

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


class TestRatioAllclose:
    """Tests for the ratio_allclose comparator."""

    @staticmethod
    def _call(cmp, actual, expected, rtol=1e-5, atol=1e-5):
        return cmp(
            actual, expected,
            actual_outputs={"out": actual},
            expected_outputs={"out": expected},
            inputs={},
            rtol=rtol, atol=atol,
        )

    def test_within_tolerance_passes(self):
        """All points within atol+rtol*|expected| pass."""
        actual = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([1.001, 2.002, 3.003])
        cmp = ratio_allclose(atol=1e-2, rtol=1e-2)
        ok, _ = self._call(cmp, actual, expected)
        assert ok

    def test_outliers_within_ratio_pass(self):
        """A small fraction of outliers is tolerated up to max_error_ratio."""
        # 1 outlier out of 100 = 1% ; max_error_ratio=0.05 allows it.
        actual = torch.zeros(100)
        expected = torch.zeros(100)
        actual[0] = 10.0  # one big outlier
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.05)
        ok, _ = self._call(cmp, actual, expected)
        assert ok

    def test_outliers_exceed_ratio_fail(self):
        """Too many outliers fail and the message names ratio_allclose."""
        actual = torch.zeros(100)
        expected = torch.zeros(100)
        actual[:10] = 10.0  # 10% outliers, threshold is 5%
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.05)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "ratio_allclose fail" in detail
        assert "error_count=10/100" in detail

    @pytest.mark.parametrize("side", ["actual", "expected"])
    @pytest.mark.parametrize(
        ("value", "nan_count", "inf_count"),
        [(float("nan"), 1, 0), (float("inf"), 0, 1)],
    )
    def test_nan_inf_in_either_side_always_fails(
        self, side, value, nan_count, inf_count
    ):
        """NaN or Inf on either side is a hard fail, independent of ratio."""
        cmp = ratio_allclose(atol=1.0, rtol=1.0, max_error_ratio=1.0)
        actual = torch.tensor([0.0, 0.0])
        expected = torch.tensor([0.0, 0.0])
        if side == "actual":
            actual[0] = value
        else:
            expected[0] = value
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "illegal values in comparison" in detail
        assert f"{side}: NaN={nan_count} Inf={inf_count}" in detail

    def test_invalid_max_error_ratio_rejected(self):
        """max_error_ratio outside [0, 1] raises at factory time."""
        with pytest.raises(ValueError, match="max_error_ratio"):
            ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=1.5)

    def test_atol_rtol_override(self):
        """Factory-supplied atol/rtol override validate_golden's defaults."""
        # validate_golden defaults rtol=atol=1e-5 would fail this, but the
        # comparator's own atol=1.0 should allow it.
        actual = torch.tensor([1.0])
        expected = torch.tensor([1.5])
        cmp = ratio_allclose(atol=1.0, rtol=0.0)
        ok, _ = self._call(cmp, actual, expected, rtol=1e-5, atol=1e-5)
        assert ok


class TestIgnoreNan:
    """Tests for ratio_allclose's ignore_nan mask."""

    @staticmethod
    def _call(cmp, actual, expected, rtol=1e-5, atol=1e-5):
        return cmp(
            actual, expected,
            actual_outputs={"out": actual},
            expected_outputs={"out": expected},
            inputs={},
            rtol=rtol, atol=atol,
        )

    @pytest.mark.parametrize("residue", [1e9, float("nan"), float("inf")])
    def test_nan_golden_positions_ignored(self, residue):
        """Any actual value passes where the golden is NaN, garbage included."""
        actual = torch.tensor([1.0, residue, 3.0])
        expected = torch.tensor([1.0, float("nan"), 3.0])
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0,
                             ignore_nan=True)
        ok, detail = self._call(cmp, actual, expected)
        assert ok, detail

    def test_nan_in_actual_still_fails_inside_care_region(self):
        """A NaN the golden does define is still a hard fail."""
        actual = torch.tensor([float("nan"), 2.0])
        expected = torch.tensor([1.0, float("nan")])
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0,
                             ignore_nan=True)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "illegal values in comparison" in detail

    def test_mismatch_inside_care_region_still_fails(self):
        """Masking the ignored points does not mask a real mismatch."""
        actual = torch.tensor([1.0, 9.0, 3.0])
        expected = torch.tensor([1.0, float("nan"), 3.5])
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0,
                             ignore_nan=True)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "error_count=1/2" in detail
        assert "1/3 NaN golden pts ignored" in detail

    def test_denominator_excludes_ignored_points(self):
        """The error ratio is taken over the compared region, not the whole tensor."""
        # 4 bad of 10 compared = 40% > 25%, so this must fail. Counting the 10
        # ignored NaN points as compared would dilute it to 20% and wrongly pass.
        actual = torch.zeros(20)
        actual[:4] = 9.0
        expected = torch.zeros(20)
        expected[10:] = float("nan")
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.25,
                             ignore_nan=True)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "error_count=4/10" in detail
        assert "10/20 NaN golden pts ignored" in detail

    def test_all_nan_golden_fails(self):
        """An entirely NaN golden fails instead of passing on an empty compare."""
        actual = torch.tensor([1.0, 2.0])
        expected = torch.tensor([float("nan"), float("nan")])
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0,
                             ignore_nan=True)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "entirely NaN" in detail

    def test_disabled_by_default(self):
        """Without the flag a NaN golden stays a hard fail."""
        actual = torch.tensor([1.0, 2.0])
        expected = torch.tensor([1.0, float("nan")])
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "illegal values in comparison" in detail

    def test_composes_with_valid_rows(self):
        """valid_rows drops the tail first, ignore_nan masks what remains."""
        actual = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]])
        expected = torch.tensor([[1.0, float("nan")], [3.0, 4.0], [0.0, 0.0]])
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0,
                             valid_rows=2, ignore_nan=True)
        ok, detail = self._call(cmp, actual, expected)
        assert ok, detail

    def test_name_reports_the_flag(self):
        """The comparator label shows ignore_nan so PASS lines are explicit."""
        cmp = ratio_allclose(atol=1e-3, rtol=1e-3, ignore_nan=True)
        assert "ignore_nan=True" in cmp.__name__


class TestRatioReldiff:
    """Tests for the ratio_reldiff comparator."""

    @staticmethod
    def _call(cmp, actual, expected):
        return cmp(
            actual, expected,
            actual_outputs={"out": actual},
            expected_outputs={"out": expected},
            inputs={},
            rtol=1e-5, atol=1e-5,
        )

    def test_within_thd_passes(self):
        """Relative diff below diff_thd passes."""
        actual = torch.tensor([100.0, 200.0])
        expected = torch.tensor([100.5, 201.0])  # rel diff ~0.005
        cmp = ratio_reldiff(diff_thd=0.01, pct_thd=0.0)
        ok, _ = self._call(cmp, actual, expected)
        assert ok

    def test_small_abs_diff_shortcircuits(self):
        """Points with |a-e|<diff_thd pass even with large relative diff (near zero)."""
        # |a-e|=0.005 < diff_thd=0.01, but |a-e|/max(|a|,|e|) would be huge.
        actual = torch.tensor([1e-6])
        expected = torch.tensor([5e-3])
        cmp = ratio_reldiff(diff_thd=0.01, pct_thd=0.0)
        ok, _ = self._call(cmp, actual, expected)
        assert ok

    def test_outliers_exceed_pct_fail(self):
        """Too many bad points fail; message names ratio_reldiff."""
        actual = torch.full((100,), 100.0)
        expected = torch.full((100,), 100.0)
        actual[:10] = 200.0  # 10 bad points; pct_thd=0.05 allows 5
        cmp = ratio_reldiff(diff_thd=0.01, pct_thd=0.05)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "ratio_reldiff fail" in detail
        assert "error_count=10/100" in detail

    def test_max_diff_hd_caps_single_point(self):
        """A single point exceeding max_diff_hd fails even when count is fine."""
        actual = torch.full((100,), 100.0)
        expected = torch.full((100,), 100.0)
        actual[0] = 10000.0  # 1 bad point, rdiff ~0.99 > max_diff_hd=0.1
        cmp = ratio_reldiff(diff_thd=0.01, pct_thd=0.05, max_diff_hd=0.1)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "max_diff_hd" in detail

    def test_symmetric_denominator(self):
        """Denominator uses max(|a|,|e|): tolerates actual >> expected."""
        # a=2, e=1: |a-e|/max(|a|,|e|) = 0.5 ; |a-e|/|e| would be 1.0.
        # diff_thd=0.6 passes only under symmetric-max denominator.
        actual = torch.tensor([2.0])
        expected = torch.tensor([1.0])
        cmp = ratio_reldiff(diff_thd=0.6, pct_thd=0.0)
        ok, _ = self._call(cmp, actual, expected)
        assert ok

    @pytest.mark.parametrize("side", ["actual", "expected"])
    @pytest.mark.parametrize(
        ("value", "nan_count", "inf_count"),
        [(float("nan"), 1, 0), (float("inf"), 0, 1)],
    )
    def test_nan_inf_in_either_side_always_fails(
        self, side, value, nan_count, inf_count
    ):
        """NaN/Inf on either side is a hard fail."""
        cmp = ratio_reldiff(diff_thd=1.0, pct_thd=1.0)
        actual = torch.tensor([0.0, 0.0])
        expected = torch.tensor([0.0, 0.0])
        if side == "actual":
            actual[0] = value
        else:
            expected[0] = value
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "illegal values in comparison" in detail
        assert f"{side}: NaN={nan_count} Inf={inf_count}" in detail

    def test_invalid_params_rejected(self):
        """Out-of-range factory params raise immediately."""
        with pytest.raises(ValueError, match="diff_thd"):
            ratio_reldiff(diff_thd=0.0)
        with pytest.raises(ValueError, match="pct_thd"):
            ratio_reldiff(diff_thd=0.01, pct_thd=1.5)
        with pytest.raises(ValueError, match="max_diff_hd"):
            ratio_reldiff(diff_thd=0.01, max_diff_hd=0.0)


class TestValidRows:
    """Tests for the valid_rows / valid_axis / zero_tail restriction."""

    @staticmethod
    def _call(cmp, actual, expected):
        return cmp(
            actual, expected,
            actual_outputs={"out": actual},
            expected_outputs={"out": expected},
            inputs={},
            rtol=1e-5, atol=1e-5,
        )

    def test_inactive_tail_excluded_from_ratio(self):
        """Garbage past valid_rows is ignored when valid_rows trims it off."""
        actual = torch.zeros(10, 4)
        expected = torch.zeros(10, 4)
        actual[5:] = 99.0                      # inactive rows diverge wildly
        cmp_all = ratio_allclose(atol=1e-3, rtol=0.0)
        assert not self._call(cmp_all, actual, expected)[0]
        cmp_valid = ratio_allclose(atol=1e-3, rtol=0.0, valid_rows=5)
        assert self._call(cmp_valid, actual, expected)[0]

    def test_zero_tail_rejects_nonzero_padding(self):
        """zero_tail turns a written-past-the-active-count tail into a failure."""
        actual = torch.zeros(10, 4)
        expected = torch.zeros(10, 4)
        actual[7, 2] = 1.0
        cmp = ratio_allclose(atol=1e-3, rtol=0.0, valid_rows=5, zero_tail=True)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "inactive tail contains 1 nonzero" in detail

    def test_zero_tail_accepts_zero_padding(self):
        """A genuinely zero tail passes the zero_tail check."""
        actual = torch.zeros(10, 4)
        expected = torch.zeros(10, 4)
        cmp = ratio_allclose(atol=1e-3, rtol=0.0, valid_rows=5, zero_tail=True)
        assert self._call(cmp, actual, expected)[0]

    def test_valid_axis_one(self):
        """valid_axis=1 slices the token axis behind a leading rank axis."""
        actual = torch.zeros(2, 10, 4)
        expected = torch.zeros(2, 10, 4)
        actual[:, 5:] = 99.0
        cmp = ratio_allclose(atol=1e-3, rtol=0.0, valid_rows=5, valid_axis=1)
        assert self._call(cmp, actual, expected)[0]

    def test_reldiff_honours_valid_rows(self):
        """ratio_reldiff takes the same restriction."""
        actual = torch.ones(10, 4)
        expected = torch.ones(10, 4)
        actual[5:] = 99.0
        assert not self._call(ratio_reldiff(diff_thd=1e-3, pct_thd=0.0), actual, expected)[0]
        assert self._call(ratio_reldiff(diff_thd=1e-3, pct_thd=0.0, valid_rows=5), actual, expected)[0]

    def test_zero_valid_rows_passes_vacuously(self):
        """valid_rows=0 leaves nothing to compare and passes."""
        actual = torch.full((4, 2), 99.0)
        expected = torch.zeros(4, 2)
        assert self._call(ratio_allclose(atol=1e-3, rtol=0.0, valid_rows=0), actual, expected)[0]
        assert self._call(ratio_reldiff(diff_thd=1e-3, pct_thd=0.0, valid_rows=0), actual, expected)[0]

    def test_out_of_range_valid_rows_fails(self):
        """A valid_rows beyond the axis length is reported, not silently clamped."""
        actual = torch.zeros(4, 2)
        expected = torch.zeros(4, 2)
        cmp = ratio_allclose(atol=1e-3, rtol=0.0, valid_rows=9)
        ok, detail = self._call(cmp, actual, expected)
        assert not ok
        assert "valid_rows=9 out of range" in detail

    def test_name_records_every_param(self):
        """The comparator label spells out every knob, defaults included."""
        assert ratio_allclose(valid_rows=5, valid_axis=1).__name__ == (
            "ratio_allclose(atol=None, rtol=None, max_error_ratio=0.005, "
            "valid_rows=5, valid_axis=1, zero_tail=False, ignore_nan=False)"
        )
        assert ratio_reldiff(valid_rows=5, zero_tail=True).__name__ == (
            "ratio_reldiff(diff_thd=0.01, pct_thd=0.05, max_diff_hd=inf, "
            "valid_rows=5, valid_axis=0, zero_tail=True)"
        )
        assert "valid_rows=None" in ratio_allclose().__name__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
