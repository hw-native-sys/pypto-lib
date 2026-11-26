# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run selected device entries concurrently, borrowing cards at the dispatch boundary."""

import ast
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path


def run_case(file, platform, log, slots):
    source = Path(file).read_text()
    count = re.search(r"#\s*ci:\s*devices=(\d+)", source)
    devices = int(count[1]) if count else 1
    if devices < 1:
        raise ValueError(f"invalid device count in {file}")
    # Entries that drive the runtime themselves need the allocation for their whole CLI.
    uses_harness = any(
        isinstance(node, ast.ImportFrom) and node.module in ("golden", "golden.runner")
        and any(name.name == "run" for name in node.names)
        for node in ast.walk(ast.parse(source, filename=file))
    )
    args = [file, "-p", platform]
    env = os.environ.copy()
    build_root = Path(env.get("PYPTO_PROG_BUILD_DIR") or "build_output").resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    env["PYPTO_PROG_BUILD_DIR"] = tempfile.mkdtemp(prefix=f"{Path(file).stem}-", dir=build_root)
    admission = nullcontext()
    if uses_harness and not re.search(r"^#\s*ci:\s*direct-device\b", source, re.MULTILINE):
        env["PYPTO_CI_DEVICE_COUNT"] = str(devices)
        env["PYPTO_CI_PREPARE_SLOTS"] = str(slots)
        logical_devices = ",".join(str(rank) for rank in range(devices))
        cmd = [sys.executable, "-u", *args, "-d", logical_devices]
    else:
        from ci_device import _slot

        admission = _slot(slots / "inflight")
        cmd = ["bash", ".github/scripts/run_device_tests.sh", "--submit", str(devices), *args]
    with admission, log.open("wb") as stream:
        # Bound the whole entry, including host preparation and time in the device queue.
        return subprocess.run(
            ["timeout", "-k", "10s", "4500s", *cmd], env=env, stdout=stream, stderr=subprocess.STDOUT,
        ).returncode


def write_results(results):
    target = Path("results.tsv")
    tmp = target.with_suffix(".tmp")
    tmp.write_text("".join(f"{file}\t{status}\n" for file, status in results.items()))
    tmp.replace(target)


def print_case_log(path):
    """Keep internal spacing and print each case transcript on whole lines."""
    if not path.is_file():
        return
    started = False
    blank_lines = 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                blank_lines += 1
                continue
            if started and blank_lines:
                sys.stdout.buffer.write(b"\n" * blank_lines)
            sys.stdout.buffer.write(line.rstrip(b"\r\n") + b"\n")
            started = True
            blank_lines = 0


def main():
    platform = os.environ["PLATFORM"]
    if platform not in ("a2a3", "a5"):
        raise ValueError(f"unsupported device platform: {platform}")
    results = dict.fromkeys(os.environ["FILES"].split(), "missing")
    write_results(results)
    compile_workers = int(os.environ.get("PYPTO_CI_COMPILE_WORKERS", "8"))
    golden_workers = int(os.environ.get("PYPTO_CI_DEVICE_WORKERS", "2"))
    if compile_workers < 1 or golden_workers < 1:
        raise ValueError("PYPTO_CI_COMPILE_WORKERS and PYPTO_CI_DEVICE_WORKERS must be positive")
    inflight = 2 * golden_workers
    total = len(results)
    width = len(str(total))
    print(
        f"[device][{platform}] cases={total} compile={compile_workers} "
        f"golden={golden_workers} in-flight={inflight}", flush=True,
    )
    with tempfile.TemporaryDirectory() as temp, ThreadPoolExecutor(max_workers=compile_workers + inflight) as pool:
        slots = Path(temp)
        for name, limit in (("compile", compile_workers), ("golden", golden_workers), ("inflight", inflight)):
            (slots / name).mkdir()
            for index in range(limit):
                (slots / name / str(index)).touch()
        tasks = {}
        for index, file in enumerate(results):
            log = Path(temp) / f"{index}.log"
            tasks[pool.submit(run_case, file, platform, log, slots)] = (file, log)
        for done, future in enumerate(as_completed(tasks), 1):
            file, log = tasks[future]
            detail = None
            try:
                rc = future.result()
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                rc = 1
            results[file] = "pass" if rc == 0 else "fail"
            write_results(results)
            print(f"::group::[{done:>{width}}/{total}] {results[file].upper()} {file}", flush=True)
            print_case_log(log)
            if detail:
                print(detail)
            if rc:
                print(f"::error file={file}::FAIL (exit {rc})")
            print("::endgroup::", flush=True)
    return int("fail" in results.values())


if __name__ == "__main__":
    sys.exit(main())
