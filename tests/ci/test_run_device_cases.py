# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the a2a3 case driver's scheduling and its disk bound.

``task-submit`` and the producing run are stubbed, so these exercise the
scheduler itself: dispatch order, the snapshot slot accounting, and what
reaches the lease command line.
"""

import argparse
import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

_DRIVER = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "run_device_cases.py"
)


def _load_driver():
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules[cls.__module__], which is None for an unregistered module.
    spec = importlib.util.spec_from_file_location("run_device_cases", _DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


def _args(tmp_path, **overrides):
    values = dict(
        platform="a2a3",
        device_id="auto",
        repo_root=str(tmp_path),
        queue_timeout=3600,
        run_timeout=600,
        golden_workers=4,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _write(tmp_path, name, *, golden_only=True, cards=1):
    """A stand-in entry file carrying the markers the driver greps for."""
    body = f"# ci: devices={cards}\n" if cards > 1 else ""
    if golden_only:
        body += 'parser.add_argument("--golden-only", action="store_true")\n'
    else:
        body += 'parser.add_argument("--compile-only", action="store_true")\n'
    path = tmp_path / name
    path.write_text(body)
    return name


class TestMarkers:
    def test_device_marker_defaults_to_one_card(self):
        assert driver.read_device_marker("# nothing here\n") == 1

    def test_device_marker_reads_the_count(self):
        assert driver.read_device_marker("# ci: devices=4\n") == 4

    def test_snapshot_line_is_parsed_from_the_pass_line(self):
        out = "[RUN] compute golden ...\n[RUN] PASS (3.0s, golden saved to /b/x_1/data)\n"
        assert driver.parse_snapshot(out) == "/b/x_1/data"

    def test_snapshot_absent_when_the_run_printed_no_pass_line(self):
        assert driver.parse_snapshot("[RUN] compile ...\nTraceback\n") is None

    def test_cases_split_on_the_golden_only_declaration(self, tmp_path):
        a = _write(tmp_path, "a.py", golden_only=True)
        b = _write(tmp_path, "b.py", golden_only=False)
        needs, plain = driver.build_cases([str(tmp_path / a), str(tmp_path / b)])
        assert [c.path for c in needs] == [str(tmp_path / a)]
        assert [c.path for c in plain] == [str(tmp_path / b)]

    def test_unreadable_path_becomes_a_plain_case(self, tmp_path):
        needs, plain = driver.build_cases([str(tmp_path / "gone.py")])
        assert not needs
        assert [c.path for c in plain] == [str(tmp_path / "gone.py")]


class TestCommands:
    def test_golden_run_gets_the_world_size_the_lease_will_hand_it(self, tmp_path):
        case = driver.Case(path="models/m.py", cards=2)
        command = driver.golden_command(case, "a2a3", str(tmp_path))
        assert command[-5:] == ["-p", "a2a3", "-d", "0,1", "--golden-only"]

    def test_lease_replays_the_snapshot_and_borrows_the_cards(self, tmp_path):
        case = driver.Case(path="models/m.py", cards=2, snapshot="/b/m_1/data")
        command = driver.lease_command(case, _args(tmp_path))
        assert "--device-num" in command and command[command.index("--device-num") + 1] == "2"
        run = command[-1]
        assert run.endswith("-p a2a3 -d $TASK_DEVICE --golden-data /b/m_1/data")

    def test_lease_without_a_snapshot_runs_the_whole_case(self, tmp_path):
        case = driver.Case(path="models/m.py", cards=1)
        command = driver.lease_command(case, _args(tmp_path))
        assert "--golden-data" not in command[-1]
        assert "--device-num" not in command


class _Harness:
    """Stubs the producing run and the lease, recording what the driver did."""

    def __init__(self, monkeypatch, tmp_path, *, golden_seconds=None, lease_fail=()):
        self.golden_seconds = golden_seconds or {}
        self.lease_fail = set(lease_fail)
        self.dispatched = []
        self.live_snapshots = 0
        self.peak_snapshots = 0
        self.lock = threading.Lock()
        self.tmp_path = tmp_path
        monkeypatch.setattr(driver, "produce_golden", self._produce)
        monkeypatch.setattr(driver.subprocess, "run", self._lease)

    def _produce(self, case, args):
        time.sleep(self.golden_seconds.get(case.path, 0))
        snapshot = self.tmp_path / f"{Path(case.path).stem}_build" / "data"
        snapshot.mkdir(parents=True, exist_ok=True)
        case.snapshot = str(snapshot)
        case.log.append(f"[RUN] PASS (golden saved to {snapshot})")
        with self.lock:
            self.live_snapshots += 1
            self.peak_snapshots = max(self.peak_snapshots, self.live_snapshots)
        return case

    def _lease(self, command, **_kwargs):
        run = command[-1]
        path = run.split(" python ")[1].split(" ")[0]
        self.dispatched.append(path)
        if "--golden-data" in run:
            with self.lock:
                self.live_snapshots -= 1

        class _Result:
            returncode = 1 if path in self.lease_fail else 0

        return _Result()


class TestScheduling:
    def test_cases_dispatch_in_golden_completion_order(self, monkeypatch, tmp_path):
        """A slow golden must not hold the card behind it: whoever is ready goes."""
        slow = _write(tmp_path, "slow.py")
        fast = _write(tmp_path, "fast.py")
        harness = _Harness(
            monkeypatch, tmp_path,
            golden_seconds={str(tmp_path / slow): 0.4, str(tmp_path / fast): 0.0},
        )

        rc = driver.run([str(tmp_path / slow), str(tmp_path / fast)], _args(tmp_path))

        assert rc == 0
        assert harness.dispatched == [str(tmp_path / fast), str(tmp_path / slow)]

    def test_cases_without_a_golden_fill_the_card_first(self, monkeypatch, tmp_path):
        """They are ready at once, so they run while the pool is still working."""
        needs = _write(tmp_path, "needs.py", golden_only=True)
        plain = _write(tmp_path, "plain.py", golden_only=False)
        harness = _Harness(
            monkeypatch, tmp_path, golden_seconds={str(tmp_path / needs): 0.3},
        )

        driver.run([str(tmp_path / needs), str(tmp_path / plain)], _args(tmp_path))

        assert harness.dispatched[0] == str(tmp_path / plain)

    def test_snapshots_on_disk_never_exceed_the_worker_bound(self, monkeypatch, tmp_path):
        """The slot stands for a snapshot, so the bound is disk, not throughput."""
        paths = [str(tmp_path / _write(tmp_path, f"m{i}.py")) for i in range(8)]
        harness = _Harness(
            monkeypatch, tmp_path, golden_seconds={p: 0.05 for p in paths},
        )

        driver.run(paths, _args(tmp_path, golden_workers=2))

        assert len(harness.dispatched) == 8
        assert harness.peak_snapshots <= 2

    def test_consumed_snapshot_takes_the_producing_build_with_it(
        self, monkeypatch, tmp_path,
    ):
        """The lease compiles its own, so the producer's build dir is dead weight."""
        path = str(tmp_path / _write(tmp_path, "m.py"))
        _Harness(monkeypatch, tmp_path)

        driver.run([path], _args(tmp_path))

        assert not (tmp_path / "m_build" / "data").exists()
        assert not (tmp_path / "m_build").exists()

    def test_only_a_data_shaped_snapshot_takes_its_parent(self, monkeypatch, tmp_path):
        """An unexpected path deletes itself and nothing above it."""
        path = str(tmp_path / _write(tmp_path, "m.py"))
        harness = _Harness(monkeypatch, tmp_path)
        odd = tmp_path / "keep" / "elsewhere"
        odd.mkdir(parents=True)

        def _produce(case, _args):
            case.snapshot = str(odd)
            with harness.lock:
                harness.live_snapshots += 1
            return case

        monkeypatch.setattr(driver, "produce_golden", _produce)
        driver.run([path], _args(tmp_path))

        assert not odd.exists()
        assert (tmp_path / "keep").exists()

    def test_failed_lease_is_reported_and_exits_nonzero(self, monkeypatch, tmp_path):
        good = str(tmp_path / _write(tmp_path, "good.py"))
        bad = str(tmp_path / _write(tmp_path, "bad.py"))
        _Harness(monkeypatch, tmp_path, lease_fail=[bad])

        assert driver.run([good, bad], _args(tmp_path)) == 1

    def test_failed_golden_falls_back_to_a_whole_case_run(self, monkeypatch, tmp_path):
        """A broken split costs card time, never a FAIL the device cannot reproduce."""
        path = str(tmp_path / _write(tmp_path, "m.py"))
        harness = _Harness(monkeypatch, tmp_path)

        def _explode(case, _args):
            raise RuntimeError("producer died")

        monkeypatch.setattr(driver, "produce_golden", _explode)
        rc = driver.run([path], _args(tmp_path))

        assert rc == 0
        assert harness.dispatched == [path]

    def test_failed_golden_does_not_strand_later_cases(self, monkeypatch, tmp_path):
        """The released slot is what lets the next producer start."""
        paths = [str(tmp_path / _write(tmp_path, f"m{i}.py")) for i in range(4)]
        harness = _Harness(monkeypatch, tmp_path)

        def _explode(case, _args):
            raise RuntimeError("producer died")

        monkeypatch.setattr(driver, "produce_golden", _explode)
        rc = driver.run(paths, _args(tmp_path, golden_workers=1))

        assert rc == 0
        assert sorted(harness.dispatched) == sorted(paths)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
