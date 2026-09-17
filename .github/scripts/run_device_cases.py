# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run CI's selected kernel files, computing their goldens off the device queue.

Only the runtime phase of a case needs a card. An entry exposing
``--golden-only`` (see ``docs/run-and-validate/save-and-replay.md``) computes
its torch golden and persists it without opening a device; the leased run then
replays that snapshot through ``--golden-data`` instead of recomputing it. The
golden is the longest CPU phase of a model run -- 110s of a 122s pre-runtime
stretch for ``prefill_swa``, on the two cards it borrows -- so moving it out of
the lease is most of the card time the job holds.

``detect-changes`` hands the whole file list over before anything runs, so the
goldens do not have to be produced one case ahead of the card: a pool computes
them concurrently, and each case is submitted the moment *its* golden is ready,
in completion order rather than in list order. Cases needing no golden are
ready immediately and fill the card while the pool works. The card therefore
idles only when nothing at all is ready, and the job's wall clock is bounded by
whichever of the two streams is longer instead of by their sum.

Concurrency is bounded by disk, not by CPU: ``--golden-only`` forces the input
and golden snapshot to disk and a full-model fixture reaches ~1GB, so a
producer holds its slot until the dispatcher has consumed *and deleted* its
snapshot. At most ``--golden-workers`` snapshots exist at once. Concurrent
compiles are safe -- each lands in its own ``build_output`` directory and
pypto's JIT artifact store is flock-guarded.

A golden that fails to produce is never a failure of the case: the leased run
recomputes the same golden itself, so a broken split costs card time and
nothing else. It is reported as a warning and the case runs whole on the card.

Usage:
    python .github/scripts/run_device_cases.py --platform a2a3 \
        --device-id auto --run-timeout 600 [--golden-workers 4] <file>...
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

#: `# ci: devices=N` near the top of a file. Same marker pypto's own model
#: matrix greps for, so both repos read one spelling the same way.
_DEVICE_MARKER = re.compile(r"#\s*ci:\s*devices=(\d+)")

#: The argparse declaration an entry carries when it can produce a golden
#: without a device. Matched in the source rather than by running `--help`,
#: which would cost a torch import per file before anything is scheduled.
_GOLDEN_ONLY_MARKER = '"--golden-only"'

#: `[RUN] PASS (12.34s, golden saved to <dir>)` -- golden/runner.py's
#: golden_only exit. The directory is what --golden-data consumes.
_SNAPSHOT_LINE = re.compile(r"golden saved to (.+)\)$")


@dataclass
class Case:
    """One CI leg: a file, the cards it needs, and its golden once produced."""

    path: str
    cards: int
    #: Snapshot directory to replay, None when this case has no golden to reuse.
    snapshot: str | None = None
    #: Producer output, replayed inside this case's log group.
    log: list[str] = field(default_factory=list)


def read_device_marker(source: str) -> int:
    """Cards the `# ci: devices=N` marker asks for; 1 when a file carries none."""
    match = _DEVICE_MARKER.search(source)
    return int(match.group(1)) if match else 1


def parse_snapshot(output: str) -> str | None:
    """The snapshot directory a `--golden-only` run reported, or None."""
    for line in reversed(output.splitlines()):
        match = _SNAPSHOT_LINE.search(line.strip())
        if match:
            return match.group(1)
    return None


def build_cases(paths: list[str]) -> tuple[list[Case], list[Case]]:
    """Split *paths* into the cases needing a golden and the ones that do not.

    An unreadable path is not decided here: it becomes a plain case, so the
    lease reports it the way it reports any other broken file rather than
    taking the whole job down before a single case has run.
    """
    needs, plain = [], []
    for path in paths:
        try:
            source = Path(path).read_text(errors="replace")
        except OSError:
            plain.append(Case(path=path, cards=1))
            continue
        case = Case(path=path, cards=read_device_marker(source))
        (needs if _GOLDEN_ONLY_MARKER in source else plain).append(case)
    return needs, plain


def golden_command(case: Case, platform: str, repo_root: str) -> list[str]:
    """The card-free producing run for *case*.

    ``-d`` opens no device on this path, but an entry whose fixture is selected
    by world size reads that size off the argument, so hand it the same
    cardinality the lease will.
    """
    devices = ",".join(str(i) for i in range(case.cards))
    return [
        sys.executable, os.path.join(repo_root, case.path),
        "-p", platform, "-d", devices, "--golden-only",
    ]


def lease_command(case: Case, args: argparse.Namespace) -> list[str]:
    """The `task-submit` lease for *case*, replaying its snapshot when it has one."""
    run = (
        f"cd {args.repo_root} && source activate.sh && "
        f"PYTHONPATH={args.repo_root} python {case.path} "
        f"-p {args.platform} -d $TASK_DEVICE"
    )
    if case.snapshot:
        run += f" --golden-data {case.snapshot}"
    command = ["task-submit", "--device", args.device_id]
    if case.cards > 1:
        command += ["--device-num", str(case.cards)]
    command += [
        "--timeout", str(args.queue_timeout),
        "--max-time", str(args.run_timeout),
        "--run", run,
    ]
    return command


def produce_golden(case: Case, args: argparse.Namespace) -> Case:
    """Compute and persist *case*'s golden; annotate it with the result."""
    # Prepended, not assigned: the repo root is what the lease's own
    # PYTHONPATH carries, but anything the runner's activate.sh put there has
    # to survive too, or the producing run resolves a different pypto than the
    # leased one.
    inherited = os.environ.get("PYTHONPATH")
    path = f"{args.repo_root}{os.pathsep}{inherited}" if inherited else args.repo_root
    env = dict(os.environ, PYTHONPATH=path)
    result = subprocess.run(
        golden_command(case, args.platform, args.repo_root),
        cwd=args.repo_root, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    case.log.append(result.stdout)
    case.snapshot = parse_snapshot(result.stdout) if result.returncode == 0 else None
    if case.snapshot is None:
        case.log.append(
            f"::warning file={case.path}::golden-only split unavailable; "
            "running whole case on the card"
        )
    return case


def dispatch(case: Case, args: argparse.Namespace) -> int:
    """Run *case* on a borrowed card and report it as one log group."""
    print(f"::group::{case.path}", flush=True)
    for chunk in case.log:
        print(chunk, end="" if chunk.endswith("\n") else "\n", flush=True)
    returncode = subprocess.run(lease_command(case, args)).returncode
    if returncode != 0:
        print(f"::error file={case.path}::FAIL", flush=True)
    print("::endgroup::", flush=True)
    return returncode


def run(paths: list[str], args: argparse.Namespace) -> int:
    """Produce goldens concurrently, dispatching each case as its golden lands."""
    needs_golden, plain = build_cases(paths)
    ready: queue.Queue[Case] = queue.Queue()
    for case in plain:
        ready.put(case)

    # One slot per in-flight snapshot, held until the dispatcher deletes it:
    # the bound is disk, not CPU, so a producer that has finished still blocks
    # its successor until the card has consumed what it wrote.
    slots = threading.Semaphore(args.golden_workers)

    def produce(case: Case) -> None:
        slots.acquire()
        try:
            produce_golden(case, args)
        except BaseException as error:  # never strand the dispatcher
            case.log.append(
                f"::warning file={case.path}::golden-only split failed: {error}"
            )
            case.snapshot = None
        # The slot stands for a snapshot on disk, so it is released by whoever
        # ends its life: the dispatcher once it has deleted one, or right here
        # when there is none to delete. Exactly one of the two, always.
        if case.snapshot is None:
            slots.release()
        ready.put(case)

    failed = []
    with ThreadPoolExecutor(max_workers=args.golden_workers) as pool:
        for case in needs_golden:
            pool.submit(produce, case)
        for _ in range(len(needs_golden) + len(plain)):
            case = ready.get()
            if dispatch(case, args) != 0:
                failed.append(case.path)
            if case.snapshot:
                # Consumed: free the disk and let the next producer start. The
                # build the producer compiled around it is dead weight too --
                # the lease compiled its own -- so drop the whole work dir,
                # keeping the delete to the `<work_dir>/data` shape run()
                # itself reported.
                snapshot = Path(case.snapshot)
                doomed = snapshot.parent if snapshot.name == "data" else snapshot
                shutil.rmtree(doomed, ignore_errors=True)
                slots.release()

    if failed:
        print("FAILED: " + " ".join(failed), flush=True)
        return 1
    print("All selected files passed.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="runnable files, in report order")
    parser.add_argument("--platform", required=True)
    parser.add_argument("--device-id", required=True,
                        help="task-submit --device value (`auto` or a fixed card id)")
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--queue-timeout", type=int, default=3600,
                        help="task-submit --timeout: how long to wait for cards")
    parser.add_argument("--run-timeout", type=int, default=600,
                        help="task-submit --max-time: how long a case may hold them")
    parser.add_argument("--golden-workers", type=int, default=4,
                        help="concurrent goldens, and so snapshots on disk at once")
    args = parser.parse_args()

    if not args.files:
        print("No runnable changed files; nothing to test.", flush=True)
        return 0
    return run(args.files, args)


if __name__ == "__main__":
    sys.exit(main())
