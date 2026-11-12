# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the complete L0 EP2 Decode composition without device execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from golden import normalize_config
from pypto.ir import DistributedConfig

from .decode_layer import (
    CHANNELS,
    D,
    EP_SIZE,
    GROUP_INPUT,
    HC,
    HIDDEN,
    LATENT,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    O_RANK,
    ROPE_PAIRS,
    T,
    TOPK,
    WIDTH,
    WINDOW,
    decode_layer_ep2,
)


def _dummy_args() -> list[torch.Tensor | int]:
    """Build logical-shape operands for signature-driven PyPTO compilation."""

    args: list[torch.Tensor | int] = []

    def add(shape: list[int], dtype: torch.dtype) -> None:
        args.append(torch.empty(shape, dtype=dtype))

    f32, bf16, i8, i32, i64 = torch.float32, torch.bfloat16, torch.int8, torch.int32, torch.int64
    add([EP_SIZE, T, HC, HIDDEN], bf16)
    add([EP_SIZE, CHANNELS, HC * HIDDEN], f32)
    add([EP_SIZE, 3], f32)
    add([EP_SIZE, CHANNELS], f32)
    add([EP_SIZE, T, HC], f32)
    add([EP_SIZE, HIDDEN], bf16)
    add([EP_SIZE, HIDDEN, LATENT], i8)
    add([EP_SIZE, LATENT], f32)
    add([EP_SIZE, LATENT], bf16)
    add([EP_SIZE, LATENT, WIDTH], i8)
    add([EP_SIZE, WIDTH], f32)
    add([EP_SIZE, HIDDEN, 512], i8)
    add([EP_SIZE, 512], f32)
    add([EP_SIZE, 512], bf16)
    add([EP_SIZE, 8, GROUP_INPUT, O_RANK], bf16)
    add([EP_SIZE, 8 * O_RANK, HIDDEN], bf16)
    add([EP_SIZE, T, ROPE_PAIRS], f32)
    add([EP_SIZE, T, ROPE_PAIRS], f32)
    add([EP_SIZE, T, 2], i64)
    add([EP_SIZE, T, WINDOW], i32)
    add([EP_SIZE, 64], f32)
    add([EP_SIZE, T], i64)
    add([EP_SIZE, T, LATENT], bf16)
    add([EP_SIZE, T, WIDTH], bf16)
    add([EP_SIZE, T, 512], bf16)
    add([EP_SIZE, 2, 128, 1, 512], bf16)
    add([EP_SIZE, 2], i32)
    add([EP_SIZE, T, HIDDEN], bf16)
    add([EP_SIZE, T, HC], f32)
    add([EP_SIZE, T, HC], f32)
    add([EP_SIZE, T, HC, HC], f32)
    add([EP_SIZE, T, HIDDEN], bf16)
    add([EP_SIZE, T, HIDDEN], bf16)
    add([EP_SIZE, T, HC, HIDDEN], bf16)
    add([EP_SIZE, CHANNELS, HC * HIDDEN], f32)
    add([EP_SIZE, 3], f32)
    add([EP_SIZE, CHANNELS], f32)
    add([EP_SIZE, HIDDEN], bf16)
    add([EP_SIZE, T, HC], f32)
    add([EP_SIZE, T, HC], f32)
    add([EP_SIZE, T, HC, HC], f32)
    add([EP_SIZE, T, HIDDEN], bf16)
    add([EP_SIZE, T, HIDDEN], bf16)
    add([EP_SIZE, N_EXPERTS, D], f32)
    add([EP_SIZE, N_EXPERTS], f32)
    add([EP_SIZE, N_EXPERTS], f32)
    add([EP_SIZE, T], i64)
    add([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], i8)
    add([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], f32)
    add([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER, D], i8)
    add([EP_SIZE, N_LOCAL_EXPERTS, MOE_INTER], f32)
    add([EP_SIZE, N_LOCAL_EXPERTS, D, 2560], i8)
    add([EP_SIZE, N_LOCAL_EXPERTS, D], f32)
    add([EP_SIZE, D, MOE_INTER * 2], i8)
    add([EP_SIZE, MOE_INTER * 2], f32)
    add([EP_SIZE, MOE_INTER, D], i8)
    add([EP_SIZE, D], f32)
    add([EP_SIZE, T, N_EXPERTS], f32)
    add([EP_SIZE, T, TOPK], i32)
    add([EP_SIZE, T, TOPK], f32)
    add([EP_SIZE, T, D], bf16)
    add([EP_SIZE, T, HC, HIDDEN], bf16)
    args.extend([T, 1])
    return args


def compile_layer_ep2(*, backend_type: str = "a2a3", sim: bool = True):
    """Compile the complete L0 composition for two logical devices."""

    config = {
        "backend_type": backend_type,
        "execution_mode": sim,
        "distributed_config": DistributedConfig(device_ids=[0, 1], num_sub_workers=0),
    }
    return decode_layer_ep2.compile(*_dummy_args(), config=normalize_config(config))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a5"), default="a2a3")
    parser.add_argument("--sim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    compiled = compile_layer_ep2(backend_type=args.platform, sim=args.sim)
    result = {
        "status": "compile_pass",
        "execution": "not_run",
        "accuracy": "not_run",
        "layer_id": 0,
        "topology": {"dp": 2, "tp": 1, "ep": 2, "devices": [0, 1]},
        "output_dir": str(compiled.output_dir),
        "product_compute": "PyPTO only",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
