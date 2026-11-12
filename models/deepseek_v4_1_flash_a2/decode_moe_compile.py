# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the independent EP2 Decode MoE ABI without allocating device inputs.

The routed expert weights are declared in logical FRACTAL_NZ dimensions by the
product kernel.  PyPTO lowers those dimensions to physical NZ shapes in the
compiled artifact, so this entry intentionally uses the direct compiler API
instead of the Golden runner's logical-shape ABI comparator.  It proves code
generation only; it does not execute the kernel or establish numerical
accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from pypto.ir import DistributedConfig
from golden import normalize_config

from models.deepseek_v4_1_flash_a2.decode_moe import (
    D,
    EP_SIZE,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    T,
    decode_ep2,
)


def _dummy_args() -> list[torch.Tensor | int]:
    """Build shape-only arguments for signature-driven PyPTO compilation."""

    return [
        torch.empty([EP_SIZE, T, D], dtype=torch.bfloat16),
        torch.empty([EP_SIZE, T, D], dtype=torch.float32),
        torch.empty([EP_SIZE, N_EXPERTS, D], dtype=torch.float32),
        torch.empty([EP_SIZE, N_EXPERTS], dtype=torch.float32),
        torch.empty([EP_SIZE, N_EXPERTS], dtype=torch.float32),
        torch.empty([EP_SIZE, T], dtype=torch.int64),
        torch.empty([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], dtype=torch.int8),
        torch.empty([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], dtype=torch.float32),
        torch.empty([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], dtype=torch.int8),
        torch.empty([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], dtype=torch.float32),
        torch.empty([EP_SIZE, N_LOCAL_EXPERTS, D, 2560], dtype=torch.int8),
        torch.empty([EP_SIZE, N_LOCAL_EXPERTS, D], dtype=torch.float32),
        torch.empty([EP_SIZE, D, MOE_INTER * 2], dtype=torch.int8),
        torch.empty([EP_SIZE, MOE_INTER * 2], dtype=torch.float32),
        torch.empty([EP_SIZE, MOE_INTER, D], dtype=torch.int8),
        torch.empty([EP_SIZE, D], dtype=torch.float32),
        torch.empty([EP_SIZE, T, N_EXPERTS], dtype=torch.float32),
        torch.empty([EP_SIZE, T, 6], dtype=torch.int32),
        torch.empty([EP_SIZE, T, 6], dtype=torch.float32),
        torch.empty([EP_SIZE, T, D], dtype=torch.bfloat16),
        T,
        1,
    ]


def compile_ep2(device_ids: list[int] | None = None, *, backend_type: str = "a2a3", sim: bool = True):
    """Compile EP2 and return the PyPTO compiled artifact."""

    devices = list(range(EP_SIZE)) if device_ids is None else device_ids
    if len(devices) != EP_SIZE or len(set(devices)) != EP_SIZE:
        raise ValueError(f"EP2 compilation needs two distinct devices, got {devices}")
    config = dict(
        backend_type=backend_type,
        execution_mode=sim,
        distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
    )
    return decode_ep2.compile(*_dummy_args(), config=normalize_config(config))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a5"), default="a2a3")
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--sim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    devices = [int(value) for value in args.device.split(",") if value]
    compiled = compile_ep2(devices, backend_type=args.platform, sim=args.sim)
    result = {
        "status": "compile_pass",
        "execution": "not_run",
        "accuracy": "not_run",
        "topology": {"dp": 2, "tp": 1, "ep": 2, "devices": devices},
        "output_dir": str(compiled.output_dir),
        "physical_nz_lowering": "compiler artifact owns physical NZ shape",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
