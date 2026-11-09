# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Calculate one dense GEMM's effective throughput with explicit timing and peaks."""

import argparse
import json
import math


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for dim in ("m", "n", "k"):
        parser.add_argument(f"--{dim}", type=positive_int, required=True)
    parser.add_argument("--time-us", type=positive_float, required=True)
    parser.add_argument(
        "--time-kind", required=True,
        choices=("kernel-span", "visible-span", "profiler-duration", "mean-core-hypothesis"),
    )
    parser.add_argument("--peak-tflops", type=positive_float, required=True)
    parser.add_argument("--peak-tb-s", type=positive_float, help="Verified bandwidth peak; omit if unknown")
    for operand in ("a", "b", "out"):
        parser.add_argument(f"--{operand}-bytes", type=positive_float, required=True)
    parser.add_argument("--scale-group", type=positive_int)
    parser.add_argument("--scale-bytes", type=positive_float)
    parser.add_argument("--hbm-bytes", type=nonnegative_int, help="Measured bytes on the same device and timing window")
    parser.add_argument(
        "--hbm-scope",
        choices=("kernel-attributed", "exclusive-window", "device-wide"),
        help="Traffic attribution; required with --hbm-bytes",
    )
    args = parser.parse_args()
    if (args.scale_group is None) != (args.scale_bytes is None):
        parser.error("--scale-group and --scale-bytes must be supplied together")
    if (args.hbm_bytes is None) != (args.hbm_scope is None):
        parser.error("--hbm-bytes and --hbm-scope must be supplied together")
    if args.hbm_bytes is not None and args.time_kind == "mean-core-hypothesis":
        parser.error("measured HBM utilization requires an observed timing window, not a mean-core hypothesis")

    flops = 2 * args.m * args.n * args.k
    traffic = {
        "a_read": args.m * args.k * args.a_bytes,
        "b_read": args.k * args.n * args.b_bytes,
        "output_write": args.m * args.n * args.out_bytes,
    }
    if args.scale_group is not None:
        groups = (args.k + args.scale_group - 1) // args.scale_group
        traffic["a_scale_read"] = args.m * groups * args.scale_bytes
        traffic["b_scale_read"] = args.n * groups * args.scale_bytes
    total_bytes = sum(traffic.values())
    tflops = flops / args.time_us / 1e6
    tb_s = total_bytes / args.time_us / 1e6
    result = {
        "shape_mnk": [args.m, args.n, args.k],
        "time_us": args.time_us,
        "time_kind": args.time_kind,
        "effective_flops": flops,
        "effective_tflops": tflops,
        "peak_tflops": args.peak_tflops,
        "effective_compute_utilization_pct": 100 * tflops / args.peak_tflops,
        "traffic_assumption": "Logical data read once, output written once; no bias, input C, or padding",
        "algorithmic_bytes_breakdown": traffic,
        "algorithmic_bytes": total_bytes,
        "effective_tb_s": tb_s,
        "peak_tb_s": args.peak_tb_s,
        "effective_bandwidth_pct": 100 * tb_s / args.peak_tb_s if args.peak_tb_s is not None else None,
        "measured_hbm_bytes": args.hbm_bytes,
        "hbm_scope": args.hbm_scope,
        "actual_hbm_mbu_pct": None,
        "device_hbm_mbu_pct": None,
    }
    if args.hbm_bytes is not None and args.peak_tb_s is not None:
        field = "device_hbm_mbu_pct" if args.hbm_scope == "device-wide" else "actual_hbm_mbu_pct"
        result[field] = 100 * args.hbm_bytes / args.time_us / 1e6 / args.peak_tb_s
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
