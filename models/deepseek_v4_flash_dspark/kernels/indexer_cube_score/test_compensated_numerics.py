# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Portable CPU checks for the compensated Indexer score contraction.

Run directly, or collect with pytest/unittest. Requires only PyTorch; no PyPTO,
model fixture, NPU, or runtime compilation. This validates arithmetic properties
and CPU examples, not the device instruction's accumulation order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import unittest

import torch

A = 1 << 14
DOT_BOUND = 1 << 21
HEADS = 64
ATOL = 1e-4
RTOL = 1 / 128
MAX_ERROR_RATIO = 0.001
HALF_MIN_NORMAL = 2.0**-14
HALF_MIN_SUBNORMAL = 2.0**-24
MAGIC_BITS = 0x44400000 - 1024
REPORT = {}


def split_relu_dot(dot):
    """Two nonnegative half limbs; all calculations here are CPU FP32."""
    z = dot.float()
    high = ((z - 1024) / A).clamp_min(0).half().float()
    low = (z / A - high).clamp_min(0).half().float()
    return high, low


def split_scaled_coefficient(scaled):
    """Nearest-half expansion; require finite inputs inside the guarded range."""
    scaled = scaled.float()
    if not bool(torch.isfinite(scaled).all()) or bool((scaled.abs() > 65504).any()):
        raise ValueError("A * coefficient must be finite and |A * coefficient| <= 65504")
    high = scaled.half().float()
    residual = scaled - high
    low = residual.half().float()
    tail = (residual - low).half().float()
    return high, low, tail


def original_score(keys, query, coefficient, key_scale):
    """Independent golden: exact INT32 QK, then the original FP32 expression.

    No scaling by A, half conversion, operand decomposition, or GEMV appears
    in this golden. keys=[N,128], query=[64,128], coefficient=[64].
    """
    dot = keys.to(torch.int32) @ query.to(torch.int32).T
    return (dot.float().clamp_min(0) * coefficient.float()[None, :]).sum(dim=1) * key_scale.float()


def compensated_score(keys, query, coefficient, key_scale, *, third_limb=True, flush_subnormals=False):
    dot = keys.to(torch.int32) @ query.to(torch.int32).T
    r_high, r_low = split_relu_dot(dot)
    limbs = split_scaled_coefficient(coefficient.float() * A)
    if not third_limb:
        limbs = limbs[:2]
    if flush_subnormals:
        limbs = tuple(torch.where(w.abs() < HALF_MIN_NORMAL, 0.0, w) for w in limbs)
    # Mirrors the intended high-then-low product order; each CPU matvec's
    # internal FP32 reduction order is not a model of the Cube instruction.
    products = [r @ w for r in (r_high, r_low) for w in limbs]
    total = products[0]
    for product in products[1:]:
        total = total + product
    return total * key_scale.float()


def score_metrics(actual, expected):
    error = (actual.double() - expected.double()).abs()
    tolerance = ATOL + RTOL * expected.double().abs()
    bad = ~torch.isfinite(actual) | (error > tolerance)
    return {
        "elements": actual.numel(),
        "outliers": int(bad.sum()),
        "max_abs_error": float(error.max()),
        "max_error_over_tolerance": float((error / tolerance).max()),
        "passed": bool(torch.isfinite(actual).all()) and int(bad.sum()) / actual.numel() <= MAX_ERROR_RATIO,
    }


class CompensatedNumericsTest(unittest.TestCase):
    def test_relu_split_entire_conservative_int8_dot_range(self):
        """All 4,194,305 integers; bounded temporary storage per chunk."""
        count = 0
        max_low = 0.0
        min_positive = float("inf")
        for begin in range(-DOT_BOUND, DOT_BOUND + 1, 65536):
            z = torch.arange(begin, min(begin + 65536, DOT_BOUND + 1), dtype=torch.int32)
            high, low = split_relu_dot(z)
            # Float64 comparison prevents an FP32 add from hiding lost bits.
            expected = z.double().clamp_min(0) / A
            self.assertTrue(torch.equal(high.double() + low.double(), expected), f"chunk {begin}")
            self.assertTrue(bool((high >= 0).all() and (low >= 0).all()))
            self.assertTrue(bool(torch.isfinite(high).all() and torch.isfinite(low).all()))
            max_low = max(max_low, float(low.max()))
            for limb in (high, low):
                nonzero = limb[limb != 0]
                if nonzero.numel():
                    min_positive = min(min_positive, float(nonzero.min()))
            # Check the separate hardware-oriented INT32 -> FP32 bit alias.
            # This is a bit view, not an INT32-to-FP32 numeric cast.
            biased = (z.to(torch.int64) + MAGIC_BITS).to(torch.int32).view(torch.float32)
            recovered = biased - 768.0
            self.assertTrue(torch.equal(recovered, (z.float() - 1024) / A))
            count += z.numel()
        self.assertEqual(count, 2 * DOT_BOUND + 1)
        self.assertLessEqual(max_low, 1536 / A)
        self.assertGreaterEqual(min_positive, HALF_MIN_NORMAL)
        REPORT["relu_exhaustive"] = dict(integers=count, interval=[-DOT_BOUND, DOT_BOUND],
                                          max_low=max_low, min_nonzero_limb=min_positive)

    def test_three_coefficient_limb_absolute_error_bound(self):
        # Every nonnegative finite half value, both FP32 neighbors, every
        # adjacent-half midpoint and its FP32 neighbors, and exponent-wide
        # FP32 samples. Include both signs. This is not an exhaustive FP32 scan.
        half_values = torch.arange(0x7c00, dtype=torch.int16).view(torch.float16).float()
        midpoint = (half_values[:-1] + half_values[1:]) * 0.5
        anchors = torch.cat((half_values, midpoint))
        rng = torch.Generator().manual_seed(20260917)
        random_bits = torch.randint(0, 0x477fe001, (32768,), dtype=torch.int32, generator=rng)
        positive = torch.cat((anchors, torch.nextafter(anchors, torch.full_like(anchors, float("inf"))),
                              torch.nextafter(anchors, torch.full_like(anchors, float("-inf"))),
                              random_bits.view(torch.float32)))
        positive = positive[positive.abs() <= 65504]
        scaled = torch.cat((positive, -positive)).contiguous()
        limbs = split_scaled_coefficient(scaled)
        reconstructed = sum(w.double() for w in limbs)
        delta = (reconstructed - scaled.double()).abs()
        self.assertLessEqual(float(delta.max()), 2.0**-25)
        REPORT["coefficient_sample"] = dict(values=scaled.numel(), max_error_on_scaled_coefficient=float(delta.max()),
            bound_scaled=2.0**-25, bound_coefficient=2.0**-39,
            nonexact_values=int((delta != 0).sum()), third_half_subnormals=int(((limbs[2] != 0) & (limbs[2].abs() < HALF_MIN_NORMAL)).sum()))

    def test_out_of_range_coefficients_require_fallback(self):
        for value in (float("nan"), float("inf"), float("-inf"), 65505.0, -65505.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                split_scaled_coefficient(torch.tensor([value], dtype=torch.float32))
        split_scaled_coefficient(torch.tensor([-65504.0, 0.0, 65504.0]))

    def test_original_expression_random_target_magnitude(self):
        rows = []
        for seed, count in ((17, 1), (19, 127), (23, 128), (29, 129), (31, 257)):
            rng = torch.Generator().manual_seed(seed)
            query = torch.randint(-128, 128, (HEADS, 128), dtype=torch.int8, generator=rng)
            keys = torch.randint(-128, 128, (count, 128), dtype=torch.int8, generator=rng)
            coefficient = (torch.rand(HEADS, generator=rng) * 2 - 1) * 0.012
            key_scale = 0.004 + torch.rand(count, generator=rng) * 0.006
            expected = original_score(keys, query, coefficient, key_scale)
            actual = compensated_score(keys, query, coefficient, key_scale)
            metric = score_metrics(actual, expected)
            self.assertTrue(metric["passed"], (seed, metric))
            # Selection check is separate from the original numerical tolerance.
            top = min(64, count)
            self.assertEqual(set(actual.topk(top).indices.tolist()), set(expected.topk(top).indices.tolist()))
            rows.append(dict(seed=seed, candidates=count, **metric))
        REPORT["random_original_expression"] = rows

    def test_target_magnitude_cancellation_requires_third_limb(self):
        query = torch.full((HEADS, 128), 127, dtype=torch.int8)
        keys = torch.full((1, 128), 127, dtype=torch.int8)
        coefficient = torch.cat((torch.full((32,), -0.010233103297650814),
                                 torch.full((32,), 0.010233104228973389)))
        scale = torch.tensor([0.00784325785934925], dtype=torch.float32)
        expected = original_score(keys, query, coefficient, scale)
        two = compensated_score(keys, query, coefficient, scale, third_limb=False)
        three = compensated_score(keys, query, coefficient, scale)
        flushed = compensated_score(keys, query, coefficient, scale, flush_subnormals=True)
        old_metric = score_metrics(two, expected)
        new_metric = score_metrics(three, expected)
        ftz_metric = score_metrics(flushed, expected)
        # Each is a one-element test: an aggregate error allowance cannot hide it.
        self.assertFalse(old_metric["passed"])
        self.assertTrue(new_metric["passed"])
        self.assertFalse(ftz_metric["passed"])
        REPORT["target_magnitude_cancellation"] = dict(dot=128 * 127**2,
            coefficients=[float(coefficient[0]), float(coefficient[-1])], original=float(expected),
            two_limb=float(two), three_limb=float(three), flushed_three_limb=float(flushed),
            two_limb_metrics=old_metric, three_limb_metrics=new_metric, flushed_metrics=ftz_metric)

    def test_exact_coefficient_encoding_is_not_bitwise_fp32_reduction(self):
        # Deliberately outside the measured target coefficient magnitude:
        # this documents the limit of an arbitrary-input equivalence claim.
        query = torch.full((HEADS, 128), 127, dtype=torch.int8)
        keys = torch.full((1, 128), 127, dtype=torch.int8)
        coefficient = torch.zeros(HEADS, dtype=torch.float32)
        coefficient[0] = 1 + 7.998046875 / A
        coefficient[-1] = -(1 + 8 / A)
        scale = torch.tensor([0.00784325785934925], dtype=torch.float32)
        limbs = split_scaled_coefficient(coefficient * A)
        self.assertTrue(torch.equal(sum(w.double() for w in limbs) / A, coefficient.double()))
        expected = original_score(keys, query, coefficient, scale)
        actual = compensated_score(keys, query, coefficient, scale)
        self.assertFalse(torch.equal(actual, expected))
        REPORT["reassociation_limit"] = dict(original=float(expected), compensated=float(actual),
            exact_coefficient_reconstruction=True, metrics=score_metrics(actual, expected),
            scope="CPU example with |coefficient| about 1; not a target-magnitude acceptance test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CompensatedNumericsTest))
    output = dict(cpu_only=True, torch_version=torch.__version__, threads=args.threads,
                  elapsed_seconds=time.perf_counter() - start, passed=result.wasSuccessful(),
                  tests_run=result.testsRun, thresholds=dict(atol=ATOL, rtol=RTOL, max_error_ratio=MAX_ERROR_RATIO),
                  results=REPORT)
    if args.report:
        args.report.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
