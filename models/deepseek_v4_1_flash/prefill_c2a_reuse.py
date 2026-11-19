# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""C2A reuse prefill."""


from config import CP_SIZE, PREFILL_MAX_TOKENS
from prefill_c1a_reuse import l3_prefill_c1a_reuse, l3_prefill_c1a_reuse_cp


PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4


if __name__ == "__main__":
    import argparse
    import functools
    import torch

    from config import SUPPORTED_CP_SIZES
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run
    from prefill_c2a_full import golden_c2a_prefill, build_c2a_specs
    from quant import packed_cache_compare
    from pypto.ir import DistributedConfig

    from utils import parse_prefill_groups

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C2A reuse")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=lambda value: [int(part) for part in value.split(",")], help="comma-separated device IDs")
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=SUPPORTED_CP_SIZES)
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--dp-tokens", type=lambda value: [int(part) for part in value.split(",")], help="active token counts per DP group")
    parser.add_argument("--requests", type=int, default=1, help="request count per DP when query-lens is omitted")
    parser.add_argument("--query-lens", type=parse_prefill_groups, help="request lengths, e.g. '17,15;32'")
    parser.add_argument("--start-pos", type=parse_prefill_groups, help="request prefix lengths, e.g. '127,0;256'")
    parser.add_argument("--case", default="normal", choices=["normal", "prefix"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args()
    devices = list(range(CP_SIZE * args.dp)) if args.device is None else args.device
    if len(devices) != CP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("device IDs must be distinct nonnegative integers, with count CP * DP")
    if not 1 <= args.tokens <= PREFILL_MAX_TOKENS:
        parser.error(f"tokens must be in [1, {PREFILL_MAX_TOKENS}]")
    if args.dp_tokens is not None:
        if len(args.dp_tokens) != args.dp or any(n < 0 or n > args.tokens for n in args.dp_tokens):
            parser.error("dp-tokens must contain one count per DP group, each in [0, tokens]")
    torch.set_num_threads(8)
    entry = l3_prefill_c1a_reuse if CP_SIZE == 1 else l3_prefill_c1a_reuse_cp
    result = run(
        fn=entry,
        specs=build_c2a_specs(args, entry, mode="reuse"),
        golden_fn=functools.partial(golden_c2a_prefill, mode="reuse"),
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            ring_heap=PREFILL_ATTN_RING_HEAP,
        ),
        compare_fn={
            "kv_hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "hc_next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "hc_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": packed_cache_compare(
                mapped_pool_ratio_allclose(
                    "kv_write_slots", pool_name="window_cache",
                    mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.01,
                ),
                mapped_pool_ratio_allclose(
                    "kv_write_slots", pool_name="window_cache scale",
                    mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.0,
                ),
                fp4=False,
            ),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
