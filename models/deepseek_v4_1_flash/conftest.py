# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 precision test options and collection metadata for the CI device queue."""

import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("v41-a5")
    group.addoption("--tp", type=int, help="TP size, fixed before model import")
    group.addoption("--dp", type=int, help="DP size for this process")
    group.addoption("--ep", type=int, help="EP world size, fixed before model import")
    group.addoption("--device", help="comma-separated device IDs allocated by task-submit")
    group.addoption("--a5-case-list", help="write collected node IDs and card requirements as JSON")


def case_card_count(axes):
    """Size a TP/DP world or an EP world with implicit DP = EP / TP."""
    if any(type(value) is not int or value < 1 for value in axes.values()):
        raise pytest.UsageError("tp/dp/ep must be positive integers")
    tp = axes.get("tp", 1)
    if "ep" in axes:
        ep = axes["ep"]
        if ep % tp or ("dp" in axes and axes["dp"] != ep // tp):
            raise pytest.UsageError("EP must be divisible by TP and equal TP x DP when DP is explicit")
        return ep
    return tp * axes.get("dp", 1)


def pytest_collection_finish(session):
    output = session.config.getoption("--a5-case-list")
    if not output:
        return
    cases = []
    for item in session.items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        axes = {name: params[name] for name in ("tp", "dp", "ep") if name in params}
        cases.append({"nodeid": item.nodeid, "axes": axes, "cards": case_card_count(axes)})
    Path(output).write_text(json.dumps(cases))


@pytest.fixture
def a5_args(request):
    """Build validation arguments for a case with matching import-time shapes."""
    def build(*, tp=None, dp=None, ep=None):
        args = ["-p", "a5"]
        axes = {name: value for name, value in (("tp", tp), ("dp", dp), ("ep", ep)) if value is not None}
        cards = case_card_count(axes)
        for name, value in axes.items():
            if request.config.getoption(f"--{name}") != value:
                pytest.fail(
                    f"Run this node in a fresh pytest process with --{name} {value}; "
                    "model shapes are fixed at import time"
                )
            args += [f"--{name}", str(value)]
        device = request.config.getoption("--device")
        if device is None:
            pytest.fail("Pass --device with the IDs allocated by task-submit")
        devices = device.split(",")
        if (len(devices) != cards
                or len(set(devices)) != len(devices)
                or any(not part.isdigit() for part in devices)):
            pytest.fail(f"Allocated device IDs must be distinct and match the {cards}-card world")
        return args + ["-d", device]
    return build
