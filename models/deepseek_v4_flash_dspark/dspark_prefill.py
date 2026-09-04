# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
"""Validate DSpark prompt-context KV insertion followed by the seven-query drafter."""

from pypto.ir.distributed_compiled_program import DistributedConfig

from dspark_drafter import (
    DSPARK_SUPPORTED_BATCHES,
    TP_SIZE,
    _DSPARK_RING_HEAP,
    _dspark_kv_cache_compare,
    build_tensor_specs,
    golden_dspark_drafter,
    l3_dspark_drafter,
)
from moe import N_RANKS


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description="Validate DeepSeek V4 DSpark prompt prefill and drafting.")
    parser.add_argument("--batch", type=int, choices=DSPARK_SUPPORTED_BATCHES, default=4)
    parser.add_argument("--tp", type=int, choices=(4,), default=TP_SIZE)
    parser.add_argument("--ep", type=int, choices=(4, 8, 16), default=N_RANKS)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    assert args.tp == TP_SIZE
    assert args.ep == N_RANKS
    assert len(device_ids) >= N_RANKS
    result = run(
        fn=l3_dspark_drafter,
        specs=build_tensor_specs(args.batch, mode="prefill"),
        golden_fn=golden_dspark_drafter,
        compile_only=args.compile_only,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
            platform=args.platform,
            ring_heap=_DSPARK_RING_HEAP,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv_caches": _dspark_kv_cache_compare(),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
