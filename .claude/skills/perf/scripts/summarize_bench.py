# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Summarize pypto-lib PYPTO_BENCH raw samples per selected device."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from pathlib import Path
from typing import TextIO


# Older headers may omit fields; keep their samples without inferring round timing.
HEADER_RE = re.compile(r"^\[RUN\]\s+raw samples:(?P<fields>.*)$")
SAMPLE_RE = re.compile(
    r"^\[RUN\]\s+rank\s+(?P<rank>\S+)\s+raw\s+n=(?P<n>\d+)\s+"
    r"eff_us=\[(?P<samples>[^\]]*)\]\s*$"
)
# The harness's own per-rank summary, printed just above the raw block for an
# L3 run. Each entry sums that rank's dispatches within a round, so it is the
# per-round rank latency the raw per-dispatch samples cannot give.
PER_RANK_RE = re.compile(
    r"^\[RUN\]\s+rank\s+(?P<rank>\S+):\s+eff_us\s+.*?\bmedian=(?P<median>[0-9.]+)"
)


def _parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("provide at least one device ID")
    return devices


def _parse_samples(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",")]


class Block:
    """One raw benchmark block: its samples plus the context to read them."""

    def __init__(
        self, rounds: int | None, ranks: int | None = None, *,
        warmup: int | None = None, flattened: bool = False,
    ) -> None:
        self.rounds = rounds
        self.ranks = ranks
        self.warmup = warmup
        self.flattened = flattened
        self.rows: list[tuple[str, int, list[float]]] = []
        # rank -> median of that rank's per-round totals, from the harness's own
        # per-rank summary lines. Empty for L2 and the flatten fallback.
        self.per_rank_median: dict[str, float] = {}


def _read_blocks(stream: TextIO) -> list[Block]:
    blocks: list[Block] = []
    current: Block | None = None
    pending_median: dict[str, float] = {}

    for raw_line in stream:
        line = raw_line.rstrip("\n")

        header = HEADER_RE.match(line)
        if header is not None:
            fields = dict(re.findall(r"(\w+)=(\d+)", header.group("fields")))
            current = Block(
                int(fields["rounds"]) if "rounds" in fields else None,
                int(fields["ranks"]) if "ranks" in fields else None,
                warmup=int(fields["warmup"]) if "warmup" in fields else None,
                flattened=fields.get("fallback_flattened", "0") != "0",
            )
            current.per_rank_median = pending_median
            pending_median = {}
            blocks.append(current)
            continue

        # Per-rank summaries precede their raw block, so buffer them until the
        # block opens.
        per_rank = PER_RANK_RE.match(line)
        if per_rank is not None:
            pending_median[per_rank.group("rank")] = float(per_rank.group("median"))
            continue

        match = SAMPLE_RE.match(line)
        if match is None:
            continue
        if current is None:
            current = Block(None)
            current.per_rank_median = pending_median
            pending_median = {}
            blocks.append(current)
        current.rows.append(
            (
                match.group("rank"),
                int(match.group("n")),
                _parse_samples(match.group("samples")),
            )
        )

    return blocks


def _open_input(path: Path | None) -> TextIO:
    if path is None:
        return sys.stdin
    return path.open(encoding="utf-8", errors="replace")


def _print_headline(
    block: Block,
    device_of: dict[str, str],
    raw_median: dict[str, float],
    per_dispatch: bool,
) -> None:
    """Print the block's headline: the fastest card's median.

    Prefer the harness's per-rank medians, which sum each round's dispatches;
    the raw samples are per dispatch, so their median is not a round latency
    whenever a rank issues more than one dispatch per round.
    """
    if block.flattened or block.rounds is None or block.ranks is None:
        print(
            "warning: flattened samples or missing round/rank metadata; "
            "no per-round headline is available",
            file=sys.stderr,
        )
        return

    if set(raw_median) != set(device_of):
        print("warning: a rank has no valid timing samples; no headline is available", file=sys.stderr)
        return
    if block.per_rank_median and set(block.per_rank_median) != set(raw_median):
        print("warning: incomplete per-rank summary; no headline is available", file=sys.stderr)
        return
    if per_dispatch and not block.per_rank_median:
        print(
            "warning: raw samples are per dispatch (n > rounds) without a "
            "complete per-rank summary; no per-round headline is available",
            file=sys.stderr,
        )
        return

    medians = block.per_rank_median or raw_median
    source = "per-rank summary" if block.per_rank_median else "raw samples"
    rank = min(medians, key=lambda key: medians[key])
    device = device_of.get(rank, "unknown")
    print()
    if len(medians) == 1:
        print(
            f"**headline** {medians[rank]:.2f} us — median of device {device} "
            f"(rank {rank}), from the {source}."
        )
        return
    print(
        f"**headline** {medians[rank]:.2f} us — fastest-card median: device "
        f"{device} (rank {rank}) of {len(medians)}, from the {source}. "
        "This is not end-to-end step latency. Inspect persistent rank gaps "
        "for load imbalance as well as start skew."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parse raw benchmark log lines and print per-device timing "
            "statistics as Markdown tables."
        )
    )
    parser.add_argument(
        "--devices",
        required=True,
        type=_parse_devices,
        help="Comma-separated device IDs in the order passed to the target",
    )
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        help="Task log path; read standard input when omitted",
    )
    args = parser.parse_args()

    with _open_input(args.log) as stream:
        blocks = _read_blocks(stream)

    if not blocks:
        print("No PYPTO_BENCH raw samples found.", file=sys.stderr)
        return 1

    for block_index, block in enumerate(blocks, start=1):
        ranks = [rank for rank, _, _ in block.rows]
        if not ranks or len(set(ranks)) != len(ranks):
            print("error: empty block or duplicate rank samples", file=sys.stderr)
            return 1
        if block.ranks is not None and len(ranks) != block.ranks:
            print(
                f"error: block declares {block.ranks} ranks, but {len(ranks)} were parsed",
                file=sys.stderr,
            )
            return 1
        if len(blocks) > 1:
            print(f"### Benchmark block {block_index}")
            print()
        rounds = block.rounds if block.rounds is not None else "unknown"
        warmup = block.warmup if block.warmup is not None else "unknown"
        print(f"rounds={rounds} warmup={warmup}")
        print()
        print("| device | rank | n | min_us | max_us | median_us | mean_us | stdev_us |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")

        mapping_is_exact = len(args.devices) == len(block.rows)
        device_of: dict[str, str] = {}
        raw_median: dict[str, float] = {}
        per_dispatch = False
        for rank_index, (rank, declared_n, samples) in enumerate(block.rows):
            device = args.devices[rank_index] if mapping_is_exact else "unknown"
            device_of[rank] = device
            if declared_n != len(samples):
                print(
                    f"error: rank {rank} declares n={declared_n}, "
                    f"but {len(samples)} samples were parsed",
                    file=sys.stderr,
                )
                return 1
            if not samples or any(not math.isfinite(value) or value < 0 for value in samples):
                print(f"error: rank {rank} has empty or invalid samples", file=sys.stderr)
                return 1
            if not block.flattened and block.rounds is not None and (
                block.rounds <= 0 or len(samples) % block.rounds != 0
            ):
                print(f"error: rank {rank} has incomplete rounds", file=sys.stderr)
                return 1

            if block.rounds is not None and len(samples) > block.rounds:
                per_dispatch = True
            valid_samples = [sample for sample in samples if sample > 0.0]
            missing = len(samples) - len(valid_samples)
            if missing:
                print(
                    f"warning: rank {rank}: excluded {missing} zero-valued missing timing samples; "
                    "n counts valid samples",
                    file=sys.stderr,
                )
            samples = valid_samples
            if not samples:
                print(f"| {device} | {rank} | 0 | N/A | N/A | N/A | N/A | N/A |")
                continue
            raw_median[rank] = statistics.median(samples)
            stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
            print(
                f"| {device} | {rank} | {len(samples)} | "
                f"{min(samples):.2f} | {max(samples):.2f} | "
                f"{statistics.median(samples):.2f} | "
                f"{statistics.fmean(samples):.2f} | {stdev:.2f} |"
            )

        _print_headline(block, device_of, raw_median, per_dispatch)
        if block_index != len(blocks):
            print()

    if any(len(block.rows) != len(args.devices) for block in blocks):
        print(
            "warning: runtime rank count does not match the selected device "
            "count; device labels are unknown",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
