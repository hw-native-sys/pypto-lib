# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for golden validation helpers and model comparator contracts."""

import importlib.util
from pathlib import Path
import sys

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


_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro"
_MISSING = object()


def _load_model_module(module_name, filename, dependencies=()):
    """Load a script-style model module without leaking its flat imports."""
    managed_modules = ("config", *dependencies, module_name)
    saved_modules = {
        name: sys.modules.get(name, _MISSING)
        for name in managed_modules
    }
    saved_path = list(sys.path)
    loaded_dependencies = {}

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    try:
        sys.path.insert(0, str(_MODEL_DIR))
        for name in managed_modules:
            sys.modules.pop(name, None)
        for dependency in dependencies:
            loaded_dependencies[dependency] = load(
                dependency,
                _MODEL_DIR / f"{dependency}.py",
            )
        module = load(module_name, _MODEL_DIR / filename)
        return module, loaded_dependencies
    finally:
        sys.path[:] = saved_path
        for name in managed_modules:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not _MISSING:
                sys.modules[name] = module


_DECODE_INDEXER, _ = _load_model_module(
    "_test_deepseek_v4_pro_decode_indexer",
    "decode_indexer.py",
    dependencies=("decode_indexer_compressor",),
)
_GATE, _ = _load_model_module(
    "_test_deepseek_v4_pro_gate",
    "gate.py",
)
_PREFILL_INDEXER, _PREFILL_DEPS = _load_model_module(
    "_test_deepseek_v4_pro_prefill_indexer",
    "prefill_indexer.py",
    dependencies=("prefill_indexer_compressor",),
)
_PREFILL_COMPRESSOR = _PREFILL_DEPS["prefill_indexer_compressor"]
_QKV, _ = _load_model_module(
    "_test_deepseek_v4_pro_qkv_proj_rope",
    "qkv_proj_rope.py",
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


def _decode_indexer_fixture(visible=2048):
    position = visible * _DECODE_INDEXER.COMPRESS_RATIO - 1
    position_ids = torch.full(
        (_DECODE_INDEXER.B, _DECODE_INDEXER.S),
        position,
        dtype=torch.int32,
    )
    kv_seq_lens = torch.full(
        (_DECODE_INDEXER.B,),
        visible * _DECODE_INDEXER.COMPRESS_RATIO,
        dtype=torch.int32,
    )
    indices = torch.full(
        (
            _DECODE_INDEXER.B,
            _DECODE_INDEXER.S,
            _DECODE_INDEXER.SCORE_LEN,
        ),
        -1,
        dtype=torch.int32,
    )
    indices[..., : _DECODE_INDEXER.IDX_TOPK] = (
        torch.arange(_DECODE_INDEXER.IDX_TOPK, dtype=torch.int32)
        + _DECODE_INDEXER.OFFSET
    )
    scores = torch.full(
        (
            _DECODE_INDEXER.B,
            _DECODE_INDEXER.S,
            _DECODE_INDEXER.SCORE_LEN,
        ),
        _DECODE_INDEXER.FP32_NEG_INF,
        dtype=torch.float32,
    )
    scores[..., :visible] = torch.arange(
        visible,
        0,
        -1,
        dtype=torch.float32,
    )
    return {
        "position_ids": position_ids,
        "kv_seq_lens": kv_seq_lens,
    }, indices, scores


def _call_decode_indexer_comparator(
    comparator,
    actual,
    expected,
    inputs,
    actual_score,
    expected_score,
):
    return comparator(
        actual,
        expected,
        actual_outputs={"topk_idxs": actual, "score": actual_score},
        expected_outputs={"topk_idxs": expected, "score": expected_score},
        inputs=inputs,
        rtol=1e-3,
        atol=1e-3,
    )


class TestDeepSeekV4ProDecodeIndexerValidation:
    """CPU regressions for the decode-indexer output comparators."""

    def test_rejects_out_of_range_active_prefix_entry(self):
        inputs, expected, scores = _decode_indexer_fixture()
        actual = expected.clone()
        actual[0, 0, 0] = _DECODE_INDEXER.OFFSET + 2048

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            scores.clone(),
            scores,
        )

        assert not ok
        assert "entries outside" in detail

    def test_rejects_duplicate_active_prefix_entry(self):
        inputs, expected, scores = _decode_indexer_fixture()
        actual = expected.clone()
        actual[0, 0, 1] = actual[0, 0, 0]

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            scores.clone(),
            scores,
        )

        assert not ok
        assert "active prefix has" in detail
        assert "unique entries" in detail

    def test_rejects_non_padding_tail_entry(self):
        inputs, expected, scores = _decode_indexer_fixture()
        actual = expected.clone()
        actual[0, 0, _DECODE_INDEXER.IDX_TOPK] = _DECODE_INDEXER.OFFSET

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            scores.clone(),
            scores,
        )

        assert not ok
        assert "tail contains" in detail
        assert "non--1 entries" in detail

    def test_rejects_internally_sorted_inferior_selection(self):
        inputs, expected, scores = _decode_indexer_fixture()
        actual = expected.clone()
        actual[0, 0, : _DECODE_INDEXER.IDX_TOPK] = (
            torch.arange(
                _DECODE_INDEXER.IDX_TOPK,
                2 * _DECODE_INDEXER.IDX_TOPK,
                dtype=torch.int32,
            )
            + _DECODE_INDEXER.OFFSET
        )

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            scores.clone(),
            scores,
        )

        assert not ok
        assert "clear top-k boundary miss" in detail

    def test_allows_score_bounded_selection_boundary_swap(self):
        inputs, expected, expected_scores = _decode_indexer_fixture()
        boundary = _DECODE_INDEXER.IDX_TOPK - 1
        expected_scores[..., boundary] = 1.0
        expected_scores[..., boundary + 1] = 0.9999
        expected_scores[..., boundary - 1] = 1.1
        actual = expected.clone()
        actual[0, 0, boundary] = boundary + 1 + _DECODE_INDEXER.OFFSET
        actual_scores = expected_scores.clone()
        actual_scores[0, 0, boundary] = 0.9998
        actual_scores[0, 0, boundary + 1] = 1.0001

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            actual_scores,
            expected_scores,
        )

        assert ok, detail

    def test_rejects_clear_selection_boundary_replacement(self):
        inputs, expected, scores = _decode_indexer_fixture()
        scores[..., _DECODE_INDEXER.IDX_TOPK - 1] = 1.0
        scores[..., _DECODE_INDEXER.IDX_TOPK] = 0.1
        actual = expected.clone()
        actual[0, 0, _DECODE_INDEXER.IDX_TOPK - 1] = (
            _DECODE_INDEXER.IDX_TOPK + _DECODE_INDEXER.OFFSET
        )

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            scores.clone(),
            scores,
        )

        assert not ok
        assert "clear top-k boundary miss" in detail

    def test_rejects_displaced_candidate_score_outlier(self):
        inputs, expected, expected_scores = _decode_indexer_fixture()
        boundary = _DECODE_INDEXER.IDX_TOPK - 1
        expected_scores[..., boundary] = 1.0
        expected_scores[..., boundary + 1] = 0.9999
        actual = expected.clone()
        actual[0, 0, boundary] = boundary + 1 + _DECODE_INDEXER.OFFSET
        actual_scores = expected_scores.clone()
        actual_scores[0, 0, boundary] = 0.0
        actual_scores[0, 0, boundary + 1] = 2.0

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            actual_scores,
            expected_scores,
        )

        assert not ok
        assert "candidate hard bound" in detail

    def test_rejects_clear_selected_set_inversion(self):
        inputs, expected, scores = _decode_indexer_fixture()
        scores[..., :2048] = torch.linspace(0.9, 0.0, 2048)
        scores[..., 0] = 10.0
        scores[..., 1] = 1.0
        actual = expected.clone()
        actual[0, 0, :2] = actual[0, 0, :2].flip(0)

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            scores.clone(),
            scores,
        )

        assert not ok
        assert "clear inversion" in detail

    def test_allows_score_bounded_selected_set_inversion(self):
        inputs, expected, expected_scores = _decode_indexer_fixture()
        expected_scores[..., 0] = 2048.0
        expected_scores[..., 1] = 2047.9999
        actual = expected.clone()
        actual[0, 0, :2] = actual[0, 0, :2].flip(0)
        actual_scores = expected_scores.clone()
        actual_scores[0, 0, 0] = 2047.9998
        actual_scores[0, 0, 1] = 2048.0001

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.decode_topk_compare,
            actual,
            expected,
            inputs,
            actual_scores,
            expected_scores,
        )

        assert ok, detail

    def test_score_rejects_large_visible_outlier(self):
        inputs, _indices, expected_scores = _decode_indexer_fixture()
        actual_scores = expected_scores.clone()
        actual_scores[0, 0, 100] += 100.0

        ok, detail = _call_decode_indexer_comparator(
            _DECODE_INDEXER.score_valid_compare,
            actual_scores,
            expected_scores,
            inputs,
            actual_scores,
            expected_scores,
        )

        assert not ok
        assert "score hard bound" in detail


def _gate_inputs():
    return {
        "x_mixed": torch.zeros(_GATE.T, _GATE.D, dtype=torch.bfloat16),
        "norm_w": torch.ones(_GATE.D, dtype=torch.bfloat16),
        "gate_w": torch.zeros(
            _GATE.N_EXPERTS,
            _GATE.D,
            dtype=torch.float32,
        ),
        "gate_bias": torch.zeros(_GATE.N_EXPERTS, dtype=torch.float32),
    }


def _call_gate_comparator(comparator, actual, expected, inputs, actual_outputs):
    return comparator(
        actual,
        expected,
        actual_outputs=actual_outputs,
        expected_outputs={},
        inputs=inputs,
        rtol=1e-3,
        atol=1e-3,
    )


def _gate_routes():
    row = torch.arange(_GATE.TOPK, dtype=torch.int32)
    return row.view(1, -1).expand(_GATE.T, -1).clone()


class TestDeepSeekV4ProGateValidation:
    """CPU regressions for expert-routing comparator semantics."""

    def test_accepts_score_bounded_topk_boundary_swap(self, monkeypatch):
        scores = torch.zeros(_GATE.T, _GATE.N_EXPERTS)
        scores[:, : _GATE.TOPK + 1] = torch.tensor(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.50000, 0.49995]
        )
        monkeypatch.setattr(
            _GATE,
            "_golden_gate_scores",
            lambda inputs: (None, None, scores, scores),
        )
        expected = _gate_routes()
        actual = expected.clone()
        actual[:, -1] = _GATE.TOPK
        comparator = _GATE.gate_indices_compare(_GATE.N_HASH_LAYERS, _GATE.T)

        ok, detail = _call_gate_comparator(
            comparator,
            actual,
            expected,
            _gate_inputs(),
            {"indices": actual},
        )

        assert ok, detail

    def test_rejects_clear_topk_miss(self, monkeypatch):
        scores = torch.zeros(_GATE.T, _GATE.N_EXPERTS)
        scores[:, : _GATE.TOPK + 1] = torch.tensor(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
        )
        monkeypatch.setattr(
            _GATE,
            "_golden_gate_scores",
            lambda inputs: (None, None, scores, scores),
        )
        expected = _gate_routes()
        actual = expected.clone()
        actual[:, -1] = _GATE.TOPK
        comparator = _GATE.gate_indices_compare(_GATE.N_HASH_LAYERS, _GATE.T)

        ok, _ = _call_gate_comparator(
            comparator,
            actual,
            expected,
            _gate_inputs(),
            {"indices": actual},
        )

        assert not ok

    @pytest.mark.parametrize("bad_id", [0, -1, _GATE.N_EXPERTS])
    def test_rejects_duplicate_or_invalid_id(self, bad_id):
        actual = _gate_routes()
        if bad_id == 0:
            actual[:, 1] = 0
        else:
            actual[:, -1] = bad_id
        comparator = _GATE.gate_indices_compare(_GATE.N_HASH_LAYERS, _GATE.T)

        ok, _ = _call_gate_comparator(
            comparator,
            actual,
            actual.clone(),
            _gate_inputs(),
            {"indices": actual},
        )

        assert not ok

    def test_rejects_unambiguous_reverse_order(self, monkeypatch):
        scores = torch.zeros(_GATE.T, _GATE.N_EXPERTS)
        scores[:, : _GATE.TOPK] = torch.tensor(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        )
        monkeypatch.setattr(
            _GATE,
            "_golden_gate_scores",
            lambda inputs: (None, None, scores, scores),
        )
        expected = _gate_routes()
        actual = expected.clone()
        actual[:, [0, 1]] = actual[:, [1, 0]]
        comparator = _GATE.gate_indices_compare(_GATE.N_HASH_LAYERS, _GATE.T)

        ok, _ = _call_gate_comparator(
            comparator,
            actual,
            expected,
            _gate_inputs(),
            {"indices": actual},
        )

        assert not ok

    def test_weights_follow_actual_route(self, monkeypatch):
        scores = torch.ones(_GATE.T, _GATE.N_EXPERTS)
        scores[:, : _GATE.TOPK] = torch.tensor(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        )
        monkeypatch.setattr(
            _GATE,
            "_golden_gate_scores",
            lambda inputs: (None, None, scores, scores),
        )
        indices = _gate_routes()
        selected = torch.gather(scores, -1, indices.long())
        weights = (
            selected / selected.sum(dim=-1, keepdim=True) * _GATE.ROUTE_SCALE
        )
        comparator = _GATE.gate_weights_compare(_GATE.T)

        ok, detail = _call_gate_comparator(
            comparator,
            weights,
            weights,
            _gate_inputs(),
            {"indices": indices},
        )
        assert ok, detail

        wrong = weights.clone()
        wrong[:, [0, 1]] = wrong[:, [1, 0]]
        ok, _ = _call_gate_comparator(
            comparator,
            wrong,
            weights,
            _gate_inputs(),
            {"indices": indices},
        )
        assert not ok

    def test_inactive_route_and_weight_tails_are_zero(self, monkeypatch):
        scores = torch.ones(_GATE.T, _GATE.N_EXPERTS)
        monkeypatch.setattr(
            _GATE,
            "_golden_gate_scores",
            lambda inputs: (None, None, scores, scores),
        )
        active = _GATE.T - 1
        indices = _gate_routes()
        indices[active:] = 1
        index_comparator = _GATE.gate_indices_compare(
            _GATE.N_HASH_LAYERS,
            active,
        )
        ok, _ = _call_gate_comparator(
            index_comparator,
            indices,
            indices,
            _gate_inputs(),
            {"indices": indices},
        )
        assert not ok

        indices[active:] = 0
        weights = torch.full(
            (_GATE.T, _GATE.TOPK),
            _GATE.ROUTE_SCALE / _GATE.TOPK,
        )
        weight_comparator = _GATE.gate_weights_compare(active)
        ok, _ = _call_gate_comparator(
            weight_comparator,
            weights,
            weights,
            _gate_inputs(),
            {"indices": indices},
        )
        assert not ok

    def test_x_norm_scale_requires_exact_zero_inactive_tail(self):
        active = _GATE.T - 1
        expected = torch.ones(_GATE.T, 1)
        expected[active:] = 0
        actual = expected.clone()
        comparator = _GATE.gate_x_norm_scale_compare(active)

        actual[0, 0] += 1e-3
        ok, detail = _call_gate_comparator(
            comparator,
            actual,
            expected,
            {},
            {},
        )
        assert ok, detail

        actual = expected.clone()
        actual[active, 0] = 5e-4
        ok, _ = _call_gate_comparator(
            comparator,
            actual,
            expected,
            {},
            {},
        )
        assert not ok


def _call_prefill_comparator(
    comparator,
    actual,
    expected,
    *,
    inputs,
    actual_outputs,
    expected_outputs,
):
    return comparator(
        actual,
        expected,
        actual_outputs=actual_outputs,
        expected_outputs=expected_outputs,
        inputs=inputs,
        rtol=1e-3,
        atol=1e-3,
    )


def _prefill_index_mapping_inputs(num_tokens):
    position_ids = torch.arange(_PREFILL_COMPRESSOR.T, dtype=torch.int32)
    mapping = torch.full((_PREFILL_COMPRESSOR.T,), -1, dtype=torch.int64)
    for token in range(num_tokens):
        position = int(position_ids[token].item())
        if (position + 1) % _PREFILL_COMPRESSOR.COMPRESS_RATIO == 0:
            mapping[token] = (
                (position + 1) // _PREFILL_COMPRESSOR.COMPRESS_RATIO - 1
            )
    return {
        "position_ids": position_ids,
        "idx_slot_mapping": mapping,
        "idx_block_table": torch.tensor([0], dtype=torch.int32),
    }


def _prefill_topk_fixture(num_tokens):
    position_ids = torch.arange(_PREFILL_INDEXER.T, dtype=torch.int32)
    indices = torch.full(
        (_PREFILL_INDEXER.T, _PREFILL_INDEXER.INDEXER_TOPK_CAP),
        -1,
        dtype=torch.int32,
    )
    scores = torch.zeros(
        _PREFILL_INDEXER.T,
        _PREFILL_INDEXER.INDEXER_SCORE_CAP,
    )
    for token in range(num_tokens):
        visible = min(
            (int(position_ids[token].item()) + 1)
            // _PREFILL_INDEXER.COMPRESS_RATIO,
            _PREFILL_INDEXER.INDEXER_SCORE_CAP,
        )
        if visible:
            indices[token, :visible] = torch.arange(
                visible,
                dtype=torch.int32,
            )
            scores[token, :visible] = torch.arange(
                visible,
                0,
                -1,
                dtype=torch.float32,
            )
    return position_ids, indices, scores


class TestDeepSeekV4ProPrefillIndexerValidation:
    """CPU regressions for packed-prefill indexer comparators."""

    def test_topk_matches_runner_abi_and_rejects_inactive_tail(self):
        num_tokens = 8
        position_ids, expected, scores = _prefill_topk_fixture(num_tokens)
        actual = expected.clone()
        inputs = {"position_ids": position_ids}
        actual_outputs = {"score": scores, "topk_idxs": actual}
        expected_outputs = {
            "score": scores.clone(),
            "topk_idxs": expected,
        }
        comparator = _PREFILL_INDEXER.prefill_topk_compare(
            num_tokens=num_tokens
        )

        assert "num_tokens" not in inputs
        ok, detail = _call_prefill_comparator(
            comparator,
            actual,
            expected,
            inputs=inputs,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
        )
        assert ok, detail

        bad_actual = actual.clone()
        bad_actual[num_tokens, 0] = 0
        bad_outputs = {**actual_outputs, "topk_idxs": bad_actual}
        ok, detail = _call_prefill_comparator(
            comparator,
            bad_actual,
            expected,
            inputs=inputs,
            actual_outputs=bad_outputs,
            expected_outputs=expected_outputs,
        )
        assert not ok
        assert "inactive top-k row" in detail

    def test_topk_allows_score_bounded_near_tie(self):
        num_tokens = 8
        position_ids, expected, expected_scores = _prefill_topk_fixture(
            num_tokens
        )
        token = num_tokens - 1
        expected_scores[token, 0] = 1.0
        expected_scores[token, 1] = 0.9999
        actual = expected.clone()
        actual[token, :2] = torch.tensor([1, 0], dtype=torch.int32)
        actual_scores = expected_scores.clone()
        actual_scores[token, 0] = 0.9998
        actual_scores[token, 1] = 1.0001

        validate_golden(
            {"topk_idxs": actual, "score": actual_scores},
            {"topk_idxs": expected, "score": expected_scores},
            rtol=1.0 / 128,
            atol=1e-4,
            compare_fn={
                "topk_idxs": _PREFILL_INDEXER.prefill_topk_compare(
                    num_tokens=num_tokens
                ),
                "score": ratio_allclose(
                    atol=1e-4,
                    rtol=1.0 / 128,
                    valid_rows=num_tokens,
                ),
            },
            inputs={"position_ids": position_ids},
        )

    def test_topk_rejects_clear_inversion(self):
        num_tokens = 8
        position_ids, expected, scores = _prefill_topk_fixture(num_tokens)
        token = num_tokens - 1
        actual = expected.clone()
        actual[token, :2] = torch.tensor([1, 0], dtype=torch.int32)

        with pytest.raises(AssertionError, match="clear inversion"):
            validate_golden(
                {"topk_idxs": actual, "score": scores.clone()},
                {"topk_idxs": expected, "score": scores},
                rtol=1.0 / 128,
                atol=1e-4,
                compare_fn={
                    "topk_idxs": _PREFILL_INDEXER.prefill_topk_compare(
                        num_tokens=num_tokens
                    ),
                    "score": ratio_allclose(
                        atol=1e-4,
                        rtol=1.0 / 128,
                        valid_rows=num_tokens,
                    ),
                },
                inputs={"position_ids": position_ids},
            )

    @pytest.mark.parametrize(
        ("score_gap", "expected_ok"),
        ((13.7e-3, True), (14.1e-3, False)),
    )
    def test_topk_respects_joint_hard_boundary(self, score_gap, expected_ok):
        num_tokens = 8
        position_ids, expected, scores = _prefill_topk_fixture(num_tokens)
        token = num_tokens - 1
        scores[token, 0] = score_gap
        scores[token, 1] = 0.0
        actual = expected.clone()
        actual[token, :2] = torch.tensor([1, 0], dtype=torch.int32)
        comparator = _PREFILL_INDEXER.prefill_topk_compare(
            num_tokens=num_tokens
        )

        ok, detail = _call_prefill_comparator(
            comparator,
            actual,
            expected,
            inputs={"position_ids": position_ids},
            actual_outputs={"topk_idxs": actual, "score": scores.clone()},
            expected_outputs={"topk_idxs": expected, "score": scores},
        )

        assert ok is expected_ok, detail

    def test_topk_rejects_displaced_candidate_score_outlier(self):
        num_tokens = 8
        position_ids, expected, expected_scores = _prefill_topk_fixture(
            num_tokens
        )
        token = num_tokens - 1
        actual = expected.clone()
        actual[token, :2] = torch.tensor([1, 0], dtype=torch.int32)
        actual_scores = expected_scores.clone()
        actual_scores[token, 0] = 0.0
        actual_scores[token, 1] = 3.0

        with pytest.raises(AssertionError, match="candidate=0 score error"):
            validate_golden(
                {"topk_idxs": actual, "score": actual_scores},
                {"topk_idxs": expected, "score": expected_scores},
                rtol=1.0 / 128,
                atol=1e-4,
                compare_fn={
                    "topk_idxs": _PREFILL_INDEXER.prefill_topk_compare(
                        num_tokens=num_tokens
                    ),
                    "score": ratio_allclose(
                        atol=1e-4,
                        rtol=1.0 / 128,
                        max_error_ratio=0.005,
                        valid_rows=num_tokens,
                    ),
                },
                inputs={"position_ids": position_ids},
            )

    def test_score_rejects_ratio_budget_catastrophic_outlier(self):
        num_tokens = 8
        _, _, expected = _prefill_topk_fixture(num_tokens)
        actual = expected.clone()
        actual[num_tokens - 1, 0] += 1.0
        comparator = _PREFILL_INDEXER.prefill_score_compare(
            num_tokens=num_tokens,
            max_error_ratio=0.005,
            hard_multiplier=4.0,
        )

        ok, detail = _call_prefill_comparator(
            comparator,
            actual,
            expected,
            inputs={},
            actual_outputs={"score": actual},
            expected_outputs={"score": expected},
        )

        assert not ok
        assert "hard bound exceeded" in detail

    def test_score_keeps_bounded_near_zero_hard_ceiling(self):
        num_tokens = 8
        _, _, expected = _prefill_topk_fixture(num_tokens)
        expected[num_tokens - 1, 0] = 0.0
        comparator = _PREFILL_INDEXER.prefill_score_compare(
            num_tokens=num_tokens
        )
        accepted = expected.clone()
        accepted[num_tokens - 1, 0] = 6.9e-3

        ok, detail = _call_prefill_comparator(
            comparator,
            accepted,
            expected,
            inputs={},
            actual_outputs={"score": accepted},
            expected_outputs={"score": expected},
        )
        assert ok, detail

        rejected = expected.clone()
        rejected[num_tokens - 1, 0] = 7.1e-3
        ok, detail = _call_prefill_comparator(
            comparator,
            rejected,
            expected,
            inputs={},
            actual_outputs={"score": rejected},
            expected_outputs={"score": expected},
        )
        assert not ok
        assert "hard bound exceeded" in detail

    def test_score_rejects_finite_inactive_write(self):
        num_tokens = 8
        _, _, expected = _prefill_topk_fixture(num_tokens)
        expected[num_tokens:] = _PREFILL_INDEXER.FP32_NEG_INF
        actual = expected.clone()
        actual[num_tokens, 0] = 0.0
        comparator = _PREFILL_INDEXER.prefill_score_compare(
            num_tokens=num_tokens
        )

        ok, detail = _call_prefill_comparator(
            comparator,
            actual,
            expected,
            inputs={},
            actual_outputs={"score": actual},
            expected_outputs={"score": expected},
        )

        assert not ok
        assert "inactive score rows" in detail

    @pytest.mark.parametrize("pool_kind", ["index", "inner_state"])
    def test_mapped_pool_rejects_unmapped_row(self, pool_kind):
        num_tokens = 8 if pool_kind == "index" else 4
        position_ids = torch.arange(
            _PREFILL_COMPRESSOR.T,
            dtype=torch.int32,
        )
        if pool_kind == "index":
            inputs = _prefill_index_mapping_inputs(num_tokens)
            actual = torch.zeros(1, _PREFILL_COMPRESSOR.BLOCK_SIZE, 1, 1)
            comparator = (
                _PREFILL_COMPRESSOR.mapped_idx_cache_ratio_allclose(
                    num_tokens=num_tokens,
                    atol=0,
                    rtol=0,
                    max_error_ratio=0,
                )
            )
            output_name = "idx_kv_cache"
            unmapped_row = 2
        else:
            mapping = torch.full(
                (_PREFILL_COMPRESSOR.T,),
                -1,
                dtype=torch.int64,
            )
            mapping[:num_tokens] = torch.arange(
                _PREFILL_COMPRESSOR.INNER_STATE_BLOCK_SIZE,
                _PREFILL_COMPRESSOR.INNER_STATE_BLOCK_SIZE + num_tokens,
                dtype=torch.int64,
            )
            inputs = {
                "position_ids": position_ids,
                "inner_state_slot_mapping": mapping,
                "inner_compress_state_block_table": torch.tensor(
                    [1],
                    dtype=torch.int32,
                ),
            }
            actual = torch.zeros(
                2,
                _PREFILL_COMPRESSOR.INNER_STATE_BLOCK_SIZE,
                1,
            )
            comparator = (
                _PREFILL_COMPRESSOR.mapped_inner_state_ratio_allclose(
                    num_tokens=num_tokens,
                    atol=0,
                    rtol=0,
                    max_error_ratio=0,
                )
            )
            output_name = "compress_state"
            unmapped_row = 0
        expected = actual.clone()
        actual_outputs = {output_name: actual}
        expected_outputs = {output_name: expected}

        ok, detail = _call_prefill_comparator(
            comparator,
            actual,
            expected,
            inputs=inputs,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
        )
        assert ok, detail

        bad_actual = actual.clone()
        bad_actual.reshape(-1, 1)[unmapped_row, 0] = 1
        ok, detail = _call_prefill_comparator(
            comparator,
            bad_actual,
            expected,
            inputs=inputs,
            actual_outputs={output_name: bad_actual},
            expected_outputs=expected_outputs,
        )
        assert not ok
        assert "unmapped physical" in detail

    def test_compact_kv_rejects_inactive_tail(self):
        num_tokens = 8
        inputs = _prefill_index_mapping_inputs(num_tokens)
        expected = torch.zeros(
            _PREFILL_COMPRESSOR.MAX_CMP_WRITES,
            _PREFILL_COMPRESSOR.HEAD_DIM,
            dtype=torch.int8,
        )
        actual = expected.clone()
        actual_cache = torch.zeros(
            1,
            _PREFILL_COMPRESSOR.BLOCK_SIZE,
            1,
            _PREFILL_COMPRESSOR.HEAD_DIM,
            dtype=torch.int8,
        )
        expected_cache = actual_cache.clone()
        actual_outputs = {"kv": actual, "idx_kv_cache": actual_cache}
        expected_outputs = {
            "kv": expected,
            "idx_kv_cache": expected_cache,
        }
        comparator = (
            _PREFILL_COMPRESSOR.active_compressed_rows_ratio_allclose(
                num_tokens=num_tokens,
                atol=1,
                rtol=0,
                max_error_ratio=0.01,
            )
        )

        ok, detail = _call_prefill_comparator(
            comparator,
            actual,
            expected,
            inputs=inputs,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
        )
        assert ok, detail

        active_rows = int(
            (inputs["idx_slot_mapping"] >= 0).count_nonzero().item()
        )
        bad_actual = actual.clone()
        bad_actual[active_rows, 0] = 1
        ok, detail = _call_prefill_comparator(
            comparator,
            bad_actual,
            expected,
            inputs=inputs,
            actual_outputs={**actual_outputs, "kv": bad_actual},
            expected_outputs=expected_outputs,
        )
        assert not ok
        assert "inactive compact KV rows" in detail


def _call_qkv_comparator(
    comparator,
    actual,
    expected,
    *,
    actual_outputs=None,
    inputs=None,
):
    return comparator(
        actual,
        expected,
        actual_outputs=actual_outputs or {},
        expected_outputs={},
        inputs=inputs or {},
        rtol=1e-3,
        atol=1e-3,
    )


def _small_conditioned_q_fixture():
    generator = torch.Generator().manual_seed(7)
    tokens = 2
    qr = torch.randint(
        -8,
        9,
        (tokens, _QKV.Q_LORA),
        dtype=torch.int8,
        generator=generator,
    )
    qr_scale = torch.rand(tokens, 1, generator=generator) * 0.01 + 0.001
    wq_b = torch.randint(
        -8,
        9,
        (_QKV.Q_LORA, _QKV.H * _QKV.HEAD_DIM),
        dtype=torch.int8,
        generator=generator,
    )
    wq_b_scale = (
        torch.rand(_QKV.H * _QKV.HEAD_DIM, generator=generator) * 0.01
        + 0.001
    )
    rope_cos = torch.rand(
        tokens,
        _QKV.ROPE_DIM,
        dtype=torch.bfloat16,
        generator=generator,
    )
    rope_sin = torch.rand(
        tokens,
        _QKV.ROPE_DIM,
        dtype=torch.bfloat16,
        generator=generator,
    )
    inputs = {
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
    }
    reference = _QKV._reference_q_from_quantized_qr(
        qr,
        qr_scale,
        wq_b,
        wq_b_scale,
        rope_cos,
        rope_sin,
    )
    return qr, qr_scale, inputs, reference


class TestDeepSeekV4ProQkvValidation:
    """CPU regressions for staged QKV and RoPE output validation."""

    def test_quantized_qr_accepts_sparse_one_step_change(self):
        expected = torch.zeros(2, 1024, dtype=torch.int8)
        actual = expected.clone()
        actual[1, 17] = 1
        comparator = _QKV.quantized_qr_compare(
            max_code_step=1,
            max_changed_ratio=0.005,
        )

        ok, detail = _call_qkv_comparator(comparator, actual, expected)

        assert ok, detail

    def test_quantized_qr_rejects_large_step_and_population(self):
        expected = torch.zeros(2, 1024, dtype=torch.int8)
        comparator = _QKV.quantized_qr_compare(
            max_code_step=1,
            max_changed_ratio=0.005,
        )
        large_step = expected.clone()
        large_step[0, 0] = 2

        ok, detail = _call_qkv_comparator(comparator, large_step, expected)
        assert not ok
        assert "code_step>1" in detail

        too_many = expected.clone()
        too_many.view(-1)[:20] = 1
        ok, detail = _call_qkv_comparator(comparator, too_many, expected)
        assert not ok
        assert "changed=20/2048" in detail

    def test_quantized_qr_rejects_row_concentrated_changes(self):
        expected = torch.zeros(8, _QKV.Q_LORA, dtype=torch.int8)
        actual = expected.clone()
        actual[3, :8] = 1
        comparator = _QKV.quantized_qr_compare(
            max_code_step=1,
            max_changed_ratio=0.005,
            max_changed_per_row_ratio=0.005,
        )

        ok, detail = _call_qkv_comparator(comparator, actual, expected)

        assert not ok
        # The per-row floor is Q_LORA-derived so the assertion holds under any
        # variant geometry the driver module was imported with (pro: rows>7,
        # flash: rows>5); eight changed codes in one row exceed both.
        per_row_floor = int(_QKV.Q_LORA * 0.005)
        assert f"rows>{per_row_floor} changed codes: 1" in detail

    def test_quantized_qr_zero_per_row_ratio_is_strict(self):
        expected = torch.zeros(2, _QKV.Q_LORA, dtype=torch.int8)
        actual = expected.clone()
        actual[0, 0] = 1
        comparator = _QKV.quantized_qr_compare(
            max_code_step=1,
            max_changed_ratio=1.0,
            max_changed_per_row_ratio=0.0,
        )

        ok, detail = _call_qkv_comparator(comparator, actual, expected)

        assert not ok
        assert "rows>0 changed codes: 1" in detail

    def test_qr_scale_rejects_nonpositive_and_large_error(self):
        expected = torch.ones(128, 1)
        comparator = _QKV.qr_scale_compare(
            atol=2.5e-5,
            rtol=5e-3,
            max_error_ratio=0.0,
        )
        nonpositive = expected.clone()
        nonpositive[0] = 0

        ok, detail = _call_qkv_comparator(comparator, nonpositive, expected)
        assert not ok
        assert "positive" in detail

        large_error = expected.clone()
        large_error[0] = 2
        ok, detail = _call_qkv_comparator(comparator, large_error, expected)
        assert not ok
        assert "hard_bad=1" in detail

    def test_quantized_q_reference_matches_independent_formula(self):
        qr, qr_scale, inputs, reference = _small_conditioned_q_fixture()
        q_i32 = qr.to(torch.int32) @ inputs["wq_b"].to(torch.int32)
        q_full = (
            q_i32.float()
            * qr_scale
            * inputs["wq_b_scale"].view(1, -1)
        ).view(qr.shape[0], _QKV.H, _QKV.HEAD_DIM)
        q_full *= torch.rsqrt(
            q_full.square().mean(-1, keepdim=True) + _QKV.EPS
        )
        pair = q_full[..., _QKV.NOPE_DIM :].reshape(
            qr.shape[0],
            _QKV.H,
            _QKV.ROPE_HALF,
            2,
        )
        cos = inputs["rope_cos"].float()[:, None, : _QKV.ROPE_HALF]
        sin = inputs["rope_sin"].float()[:, None, : _QKV.ROPE_HALF]
        independently_rotated = torch.empty_like(pair, dtype=torch.bfloat16)
        independently_rotated[..., 0] = (
            pair[..., 0] * cos - pair[..., 1] * sin
        ).to(torch.bfloat16)
        independently_rotated[..., 1] = (
            pair[..., 0] * sin + pair[..., 1] * cos
        ).to(torch.bfloat16)
        independent = torch.cat(
            (
                q_full[..., : _QKV.NOPE_DIM],
                independently_rotated.flatten(-2),
            ),
            dim=-1,
        ).to(torch.bfloat16)

        assert torch.equal(reference, independent)

    def test_conditioned_q_uses_emitted_qr_and_detects_corruption(self):
        qr, qr_scale, inputs, reference = _small_conditioned_q_fixture()
        actual_outputs = {"qr": qr, "qr_scale": qr_scale}
        comparator = _QKV.q_from_runtime_qr_compare(max_error_ratio=0.0)
        unrelated_cpu_q = torch.zeros_like(reference)

        ok, detail = _call_qkv_comparator(
            comparator,
            reference,
            unrelated_cpu_q,
            actual_outputs=actual_outputs,
            inputs=inputs,
        )
        assert ok, detail

        corrupted = reference.clone()
        corrupted.view(-1)[0] += torch.tensor(1.0, dtype=torch.bfloat16)
        ok, detail = _call_qkv_comparator(
            comparator,
            corrupted,
            unrelated_cpu_q,
            actual_outputs=actual_outputs,
            inputs=inputs,
        )
        assert not ok
        assert "downstream of emitted QR" in detail

    def test_conditioned_q_rejects_missing_runner_input(self):
        qr, qr_scale, inputs, reference = _small_conditioned_q_fixture()
        del inputs["rope_sin"]
        comparator = _QKV.q_from_runtime_qr_compare()

        ok, detail = _call_qkv_comparator(
            comparator,
            reference,
            reference,
            actual_outputs={"qr": qr, "qr_scale": qr_scale},
            inputs=inputs,
        )

        assert not ok
        assert "rope_sin" in detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
