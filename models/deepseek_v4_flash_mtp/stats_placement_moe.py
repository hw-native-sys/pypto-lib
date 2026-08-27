# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=8
"""DeepSeek-V4 standalone MoE replay with offline expert placement."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from eplb_fixture import (
    EPLB_EP_SIZE,
    EPLB_TOKENS,
    EPLB_TOPK,
    EPLB_TP_SIZE,
    EXPERT_PLACEMENT_CHOICES,
    EXPERT_PLACEMENT_FLAG,
    PLACEMENT_MANIFEST_FLAG,
    STATIC_STATS_PLACEMENT,
    STATS_PLACEMENT_EXPERTS_PER_RANK,
    STATS_PLACEMENT_FLAG,
    configure_eplb_argv,
    make_eplb_input_ids_spec,
    validate_eplb_topology,
)


if STATS_PLACEMENT_FLAG not in sys.argv:
    sys.argv.append(STATS_PLACEMENT_FLAG)
configure_eplb_argv()

import moe
from stats_placement_fixture import (
    DEFAULT_MANIFEST_PATH,
    adapt_single_layer_stats_placement_specs,
)


LAYER_ID = 0


def build_compare_fn():
    """Build the versioned numeric Golden comparator for the MoE output."""
    from golden import rowwise_ratio_reldiff

    return {
        "x_next": rowwise_ratio_reldiff(
            "x_next",
            row_shape=(moe.N_RANKS, EPLB_TOKENS),
            diff_thd=3e-3,
            pct_thd=0.10,
            expected_active_rows=moe.N_RANKS * EPLB_TOKENS,
            aggregate_pct_thd=0.05,
        ),
    }


def build_tensor_specs(
    *,
    placement: str = STATIC_STATS_PLACEMENT,
    manifest_path=DEFAULT_MANIFEST_PATH,
):
    """Build the matched EP8x32 layer-0 MoE workload for one placement."""
    specs = moe.build_tensor_specs(
        layer_id=LAYER_ID,
        num_tokens=EPLB_TOKENS,
        balanced_routing=False,
    )
    specs = [
        make_eplb_input_ids_spec(spec, active_tokens=EPLB_TOKENS)
        if spec.name == "input_ids"
        else spec
        for spec in specs
    ]
    return adapt_single_layer_stats_placement_specs(
        specs,
        layer_id=LAYER_ID,
        placement=placement,
        manifest_path=manifest_path,
    )


def main() -> None:
    """Run the placement-aware standalone MoE benchmark."""
    from golden import run

    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 standalone MoE stats-shaped route replay.",
    )
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("--ep", type=int, default=EPLB_EP_SIZE, choices=[EPLB_EP_SIZE])
    parser.add_argument("--tp", type=int, default=EPLB_TP_SIZE, choices=[EPLB_TP_SIZE])
    parser.add_argument(
        "--experts-per-rank",
        type=int,
        default=STATS_PLACEMENT_EXPERTS_PER_RANK,
        choices=[STATS_PLACEMENT_EXPERTS_PER_RANK],
    )
    parser.add_argument(STATS_PLACEMENT_FLAG, action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument(
        EXPERT_PLACEMENT_FLAG,
        choices=EXPERT_PLACEMENT_CHOICES,
        default=STATIC_STATS_PLACEMENT,
        help="physical expert layout for the matched EP8x32 workload",
    )
    parser.add_argument(
        PLACEMENT_MANIFEST_FLAG,
        default=str(DEFAULT_MANIFEST_PATH),
        help="static placement manifest",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=",".join(str(device_id) for device_id in range(EPLB_EP_SIZE)),
    )
    parser.add_argument("--layer-id", type=int, default=LAYER_ID, choices=[LAYER_ID])
    parser.add_argument("--num-tokens", type=int, default=EPLB_TOKENS, choices=[EPLB_TOKENS])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--enable-chip-swimlane",
        "--enable-l2-swimlane",
        dest="enable_chip_swimlane",
        type=int,
        nargs="?",
        const=1,
        default=0,
        choices=range(5),
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", default=None)
    parser.add_argument("--log-level", default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    from stats_placement_device_output import add_device_output_arguments

    add_device_output_arguments(parser)
    args = parser.parse_args()

    device_ids = [int(device_id) for device_id in args.device.split(",")]
    if len(device_ids) != EPLB_EP_SIZE:
        raise ValueError(
            f"stats-placement MoE needs exactly {EPLB_EP_SIZE} devices, got {device_ids}"
        )
    validate_eplb_topology(
        ep_size=args.ep,
        tp_size=args.tp,
        experts_per_rank=args.experts_per_rank,
        num_experts=moe.N_EXPERTS_GLOBAL,
        tokens=args.num_tokens,
        topk=EPLB_TOPK,
    )

    if args.seed is not None:
        moe._seed_fixture_generators(args.seed)
        print(
            f"[RUN] fixture seed={args.seed} "
            f"python_hash_seed={os.environ.get('PYTHONHASHSEED', 'unset')}",
            flush=True,
        )

    specs = build_tensor_specs(
        placement=args.expert_placement,
        manifest_path=args.placement_manifest,
    )
    device_output_capture = None
    if args.save_device_output is not None or args.compare_device_output is not None:
        from device_output_ab import DenseOutputCapture, RowwiseThresholds
        from stats_placement_device_output import build_device_output_callback

        captures = (
            DenseOutputCapture(
                logical_key="moe.x_next",
                source_name="x_next",
                row_axes=2,
                expected_rows=moe.N_RANKS * EPLB_TOKENS,
                thresholds=RowwiseThresholds(
                    min_cosine=0.99,
                    max_rel_l2=0.10,
                    max_abs=10.0,
                ),
            ),
        )
        try:
            device_output_capture = build_device_output_callback(
                args,
                case="moe-ep8",
                placement=args.expert_placement,
                placement_manifest=args.placement_manifest,
                entry_identity=Path(sys.argv[0]).absolute(),
                topology={
                    "ep_size": args.ep,
                    "experts_per_rank": args.experts_per_rank,
                    "num_experts": moe.N_EXPERTS_GLOBAL,
                    "placement_layer": LAYER_ID,
                    "tokens": args.num_tokens,
                },
                captures=captures,
            )
        except ValueError as error:
            parser.error(str(error))
    print(
        "[VALIDATION] mode=numeric_golden reference=golden_moe",
        flush=True,
    )
    result = run(
        fn=moe.l3_moe,
        specs=specs,
        golden_fn=moe.golden_moe,
        golden_data=args.golden_data,
        save_data=args.save_data,
        share_readonly_golden_inputs=args.golden_data is None,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=moe.DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            log_level=args.log_level,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=build_compare_fn(),
        device_output_capture=device_output_capture,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
