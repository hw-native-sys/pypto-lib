# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


ci = runpy.run_path(str(Path(__file__).resolve().parents[1] / ".github/scripts/ci_perf.py"))
sampling = {"rounds": 2, "warmup": 1}
case = {"device_count": 2, "case_id": "example"}


def payload():
    def dispatch(wall):
        return SimpleNamespace(device_wall_us=wall, effective_us=wall / 2)

    stats = SimpleNamespace(
        **sampling, fallback_flattened=False, all_zero_device=False,
        rounds_dispatches=[{1: [dispatch(10), dispatch(20)], 2: [dispatch(28)]},
                           {1: [dispatch(12), dispatch(24)], 2: [dispatch(30)]}],
    )
    return {"passed": True, "benchmark": ci["benchmark_payload"](stats)}


@pytest.mark.parametrize(("platform", "expected", "heading"), [("a5", 29, "Device wall"), ("a2a3", 14.5, "Effective")])
def test_per_platform_dispatch_median(platform, expected, heading):
    assert ci["timing_median_us"](payload(), case, sampling, platform) == expected
    assert f"{heading} median (us)" in ci["render_report"]({"cases": [case]}, {}, platform)


@pytest.mark.parametrize("bad_dispatches", [[], [18.0, 0.0], [float("nan")]])
def test_missing_device_window_fails_the_a5_case(bad_dispatches):
    data = payload()
    data["benchmark"]["device_wall_samples"][1]["2"] = bad_dispatches
    with pytest.raises(ValueError, match="device wall timing"):
        ci["timing_median_us"](data, case, sampling, "a5")
