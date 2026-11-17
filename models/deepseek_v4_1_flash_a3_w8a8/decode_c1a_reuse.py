# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode C1A reuse attention: thin ratio-1 entry over the shared reuse chain.

`decode_c2a_reuse.py` owns `decode_reuse_attention`, `decode_reuse`, the Torch
golden, the fixture helpers and the bounded entry.  This file only runs the same
paged chain against a ratio-1 (C1A source) fixture; nothing differs in the
kernel because the paged chain reads the source rows through the cache.
"""

import argparse

import torch

from decode_c2a_reuse import (
    build_layer_specs, build_reuse_specs, decode_reuse_attention_test,
    decode_reuse_layer_entry, golden_decode_reuse,
)


if __name__ == "__main__":
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.compile_only:
        config = dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                      device_id=args.device)
        compile_cases = (
            ("operator chain", decode_reuse_attention_test,
             build_reuse_specs(1, handoff=False, seed=args.seed)),
            ("layer chain", decode_reuse_layer_entry, build_layer_specs(seed=args.seed)),
        )
        for label, fn, specs in compile_cases:
            print(f"[DECODE] host compile: {label}", flush=True)
            result = run(fn=fn, specs=specs, golden_fn=None, config=config, compare_fn={},
                         compile_only=True)
            print(result)
            if not result.passed:
                raise SystemExit(result.error or 1)
        raise SystemExit(0)
    compare = {
        "partial": ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2),
        "window_cache": ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2),
    }
    cases = [
        ("reuse_c1a", build_reuse_specs(1, handoff=False, seed=args.seed)),
        ("reuse_c1a_handoff", build_reuse_specs(1, handoff=True, requests=[(1240, 9), (1252, 9)], seed=args.seed)),
    ]
    failures = []
    for name, specs in cases:
        print(f"[DECODE] {name}", flush=True)
        result = run(
            fn=decode_reuse_attention_test,
            specs=specs,
            golden_fn=golden_decode_reuse,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-2,
            atol=1e-2,
            compare_fn=compare,
            compile_only=args.compile_only,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
