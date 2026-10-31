# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Save and verify exact readonly consumer bytes at Full2 native boundaries."""

import hashlib

import torch


def canonical_bytes(value):
    if value.device.type != "cpu":
        raise ValueError("CPU device readback required")
    flat = torch.empty(value.numel(), dtype=value.dtype, device="cpu")
    flat.copy_(value.reshape(-1))
    return flat.view(torch.uint8)


def tensor_sha(value):
    return hashlib.sha256(canonical_bytes(value).numpy().tobytes()).hexdigest()


def exact(actual, submitted):
    return actual.dtype == submitted.dtype and actual.shape == submitted.shape and torch.equal(
        canonical_bytes(actual), canonical_bytes(submitted)
    )


def save_readonly_device_inputs(device, submitted, directory):
    actual = {name: device[name].cpu().contiguous() for name in submitted}
    passed = all(exact(actual[name], value) for name, value in submitted.items())
    with (directory / "readonly-actual.pt").open("xb") as stream:
        torch.save(actual, stream)
    return actual, passed


def admit_readonly(report, directory, submitted, file_sha):
    path = directory / "readonly-actual.pt"
    if not report.get("actual_consumer_readback") or report["readonly_actual_inputs_sha256"] != file_sha(path):
        raise RuntimeError("Actual readonly consumer readback identity differs")
    actual = torch.load(path, map_location="cpu", weights_only=True)
    if set(actual) != set(submitted):
        raise RuntimeError("Actual readonly consumer coverage differs")
    for name, value in submitted.items():
        if not exact(actual[name], value) or report["actual_inputs"][name]["sha256"] != tensor_sha(actual[name]):
            raise RuntimeError("Actual readonly consumer bytes differ: " + name)
