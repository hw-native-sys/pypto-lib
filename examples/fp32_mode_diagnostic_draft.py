# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Temporary fixed-input FP32 matrix-mode diagnostics for device CI."""

import json
from pathlib import Path
import shutil

import torch

from golden import run


def round_mantissa(tensor, bits, nearest):
    shift = 23 - bits
    raw = tensor.view(torch.int32)
    if nearest:
        raw = raw + (1 << (shift - 1)) - 1 + ((raw >> shift) & 1)
    return ((raw >> shift) << shift).view(torch.float32)


def diagnose(*, fn, specs, golden_fn, config, rtol, atol):
    if config["platform"] != "a2a3":
        return run(fn=fn, specs=specs, golden_fn=golden_fn, config=config, rtol=rtol, atol=atol)
    torch.manual_seed(0)
    phase = "baseline"

    def compare(actual, expected, **kwargs):
        inputs = kwargs["inputs"]
        references = {"fp32": expected}
        for bits in (10, 11):
            for nearest in (False, True):
                name = f"hf32_m{bits}_{'nearest' if nearest else 'truncate'}"
                references[name] = (round_mantissa(inputs["a"], bits, nearest)
                                    @ round_mantissa(inputs["b"], bits, nearest))
        references["fp16_operands"] = inputs["a"].half().float() @ inputs["b"].half().float()
        report = {"case": getattr(fn, "__name__", str(fn)), "phase": phase,
                  "device": config["device_id"], "seed": 0}
        for name, reference in references.items():
            delta = actual - reference
            report[name] = {"mismatches": int((~torch.isclose(actual, reference, rtol=rtol, atol=atol)).sum()),
                            "max_abs": delta.abs().max().item(),
                            "rmse": delta.square().mean().sqrt().item()}
        print("FP32_DIAGNOSTIC " + json.dumps(report), flush=True)
        return torch.allclose(actual, expected, rtol=rtol, atol=atol), json.dumps(report)

    result = run(fn=fn, specs=specs, golden_fn=golden_fn, config=config,
                 rtol=rtol, atol=atol, compare_fn={"c": compare}, save_data=True)
    if result.work_dir is None:
        return result
    source = Path(result.work_dir)
    target = source.with_name(source.name + "_hf32_off")
    shutil.copytree(source, target)
    for path in target.joinpath("kernels/aic").glob("*.cpp"):
        text = path.read_text()
        marker = "  #if defined(__DAV_CUBE__)\n"
        if marker not in text:
            raise RuntimeError(f"No cube body found in {path.name}")
        path.write_text(text.replace(marker, marker + "  set_ctrl(sbitset0(get_ctrl(), 46));\n", 1))
    phase = "hf32_off"
    replay = run(fn=fn, specs=specs, golden_data=str(source / "data"), runtime_dir=str(target),
                 config=config, rtol=rtol, atol=atol, compare_fn={"c": compare})
    print("FP32_DIAGNOSTIC_RESULT " + json.dumps({"baseline": result.passed, "hf32_off": replay.passed}),
          flush=True)
    return result
