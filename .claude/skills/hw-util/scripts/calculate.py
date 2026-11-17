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
    parser.add_argument("--peak-tb-s", type=positive_float, help="Bandwidth peak used for each direction; omit if unknown")
    for operand in ("a", "b", "out"):
        parser.add_argument(f"--{operand}-bytes", type=positive_float, required=True)
    parser.add_argument("--scale-group", type=positive_int)
    parser.add_argument("--scale-bytes", type=positive_float)
    parser.add_argument("--hbm-read-bytes", type=nonnegative_int, help="Measured HBM reads in the selected window")
    parser.add_argument("--hbm-write-bytes", type=nonnegative_int, help="Measured HBM writes in the same window")
    parser.add_argument(
        "--hbm-scope",
        choices=("kernel-attributed", "exclusive-window", "device-wide"),
        help="Traffic attribution; required with both measured HBM byte counts",
    )
    args = parser.parse_args()
    if (args.scale_group is None) != (args.scale_bytes is None):
        parser.error("--scale-group and --scale-bytes must be supplied together")
    hbm_inputs = (args.hbm_read_bytes, args.hbm_write_bytes, args.hbm_scope)
    if any(value is not None for value in hbm_inputs) and any(value is None for value in hbm_inputs):
        parser.error("--hbm-read-bytes, --hbm-write-bytes and --hbm-scope must be supplied together")
    if args.hbm_read_bytes is not None and args.time_kind == "mean-core-hypothesis":
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
    write_bytes = traffic["output_write"]
    read_bytes = traffic["a_read"] + traffic["b_read"]
    if args.scale_group is not None:
        read_bytes += traffic["a_scale_read"] + traffic["b_scale_read"]
    total_bytes = read_bytes + write_bytes
    tflops = flops / args.time_us / 1e6
    read_tb_s = read_bytes / args.time_us / 1e6
    write_tb_s = write_bytes / args.time_us / 1e6
    tb_s = max(read_tb_s, write_tb_s)
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
        "algorithmic_read_bytes": read_bytes,
        "algorithmic_write_bytes": write_bytes,
        "bandwidth_convention": "max(read bandwidth, write bandwidth); total bytes are diagnostic only",
        "effective_read_tb_s": read_tb_s,
        "effective_write_tb_s": write_tb_s,
        "effective_tb_s": tb_s,
        "peak_tb_s": args.peak_tb_s,
        "effective_read_bandwidth_pct": 100 * read_tb_s / args.peak_tb_s if args.peak_tb_s is not None else None,
        "effective_write_bandwidth_pct": 100 * write_tb_s / args.peak_tb_s if args.peak_tb_s is not None else None,
        "effective_bandwidth_pct": 100 * tb_s / args.peak_tb_s if args.peak_tb_s is not None else None,
        "measured_hbm_bytes": None,
        "measured_hbm_read_bytes": args.hbm_read_bytes,
        "measured_hbm_write_bytes": args.hbm_write_bytes,
        "measured_hbm_read_tb_s": None,
        "measured_hbm_write_tb_s": None,
        "measured_hbm_read_utilization_pct": None,
        "measured_hbm_write_utilization_pct": None,
        "hbm_scope": args.hbm_scope,
        "actual_hbm_mbu_pct": None,
        "device_hbm_mbu_pct": None,
    }
    if args.hbm_read_bytes is not None:
        result["measured_hbm_bytes"] = args.hbm_read_bytes + args.hbm_write_bytes
        hbm_read_tb_s = args.hbm_read_bytes / args.time_us / 1e6
        hbm_write_tb_s = args.hbm_write_bytes / args.time_us / 1e6
        result["measured_hbm_read_tb_s"] = hbm_read_tb_s
        result["measured_hbm_write_tb_s"] = hbm_write_tb_s
        if args.peak_tb_s is not None:
            read_pct = 100 * hbm_read_tb_s / args.peak_tb_s
            write_pct = 100 * hbm_write_tb_s / args.peak_tb_s
            result["measured_hbm_read_utilization_pct"] = read_pct
            result["measured_hbm_write_utilization_pct"] = write_pct
            field = "device_hbm_mbu_pct" if args.hbm_scope == "device-wide" else "actual_hbm_mbu_pct"
            result[field] = max(read_pct, write_pct)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
