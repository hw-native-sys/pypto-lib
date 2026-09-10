# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Profiler process isolation and runtime constructor-order regressions."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / ".claude/skills/incore-profiling/incore_profile.py"
_SPEC = importlib.util.spec_from_file_location("incore_profile_under_test", _SCRIPT)
profile = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profile)


def test_simulator_runtime_initializes_before_transitive_callback(tmp_path):
    """A dependent DSO constructor must initialize the interposed runtime first."""
    cc = shutil.which("cc")
    if not cc or sys.platform != "linux":
        pytest.skip("ELF constructor-order regression requires a Linux C compiler")
    toolkit = tmp_path / "toolkit"
    libs = toolkit / "lib64"
    sim = toolkit / "simulator/TestSoc/lib"
    build = tmp_path / "build"
    for directory in (libs, sim, build):
        directory.mkdir(parents=True)
    sources = {
        "camodel": """
            #include <unistd.h>
            static int ready;
            __attribute__((constructor)) static void init(void) { ready = 1; }
            void register_callback(void) { if (!ready) _exit(91); }
            int simulator_marker(void) { return 42; }
        """,
        "runtime": "void register_callback(void) {}",
        "dump": """
            extern void register_callback(void);
            __attribute__((constructor)) static void init(void) { register_callback(); }
            int dump_marker(void) { return 0; }
        """,
        "main": """
            extern int simulator_marker(void), dump_marker(void);
            int main(void) { return simulator_marker() == 42 ? dump_marker() : 92; }
        """,
    }
    for name, source in sources.items():
        (build / f"{name}.c").write_text(source)
    commands = [
        ["-shared", "-fPIC", "camodel.c", "-Wl,-soname,libruntime_camodel.so",
         "-o", str(sim / "libruntime_camodel.so")],
        ["-shared", "-fPIC", "runtime.c", "-Wl,-soname,libruntime.so", "-o", str(libs / "libruntime.so")],
        ["-shared", "-fPIC", "dump.c", f"-L{libs}", "-lruntime", "-o", str(libs / "libdump.so")],
        ["main.c", f"-L{sim}", f"-L{libs}", "-lruntime_camodel", "-ldump",
         f"-Wl,-rpath-link,{libs}", "-o", "app"],
    ]
    for command in commands:
        subprocess.run([cc, *command], cwd=build, check=True, capture_output=True)
    env = {**os.environ, "ASCEND_HOME_PATH": str(toolkit)}
    env["LD_LIBRARY_PATH"] = profile.make_ld_library_path(build, env, "TestSoc") + ":" + str(sim)
    before = subprocess.run([str(build / "app")], env=env, capture_output=True)
    assert before.returncode == 91
    isolated = profile.make_simulator_env(build, env, "TestSoc")
    after = subprocess.run([str(build / "app")], env=isolated, capture_output=True)
    assert after.returncode == 0, after.stderr
    assert not (libs / "libruntime.so").is_symlink()
    assert profile.make_simulator_env(build, env, "TestSoc") == isolated


@pytest.mark.parametrize("launcher_waits", [False, True])
def test_timeout_terminates_descendants_and_preserves_log(tmp_path, launcher_waits):
    log = tmp_path / "timeout.log"
    code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(p.pid,flush=True)"
    )
    if launcher_waits:
        code += "; time.sleep(60)"
    result = profile.run_cmd([sys.executable, "-c", code], timeout=1, log_path=log, check=False)
    assert result.returncode != 0
    pid = int(result.stdout.strip())
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        assert "Z (zombie)" in status.read_text()
    assert "process group terminated" in log.read_text()
    assert str(pid) in log.read_text()


def test_source_env_preserves_selected_python_environment(tmp_path):
    script = tmp_path / "set env.sh"
    script.write_text('export PROFILE_TEST_VALUE="$PROFILE_TEST_BASE/selected"\nunset PROFILE_TEST_REMOVED\n')
    env = {**os.environ, "PROFILE_TEST_BASE": "conda", "PROFILE_TEST_REMOVED": "old"}
    sourced = profile.source_env(script, env)
    assert sourced["PROFILE_TEST_VALUE"] == "conda/selected"
    assert "PROFILE_TEST_REMOVED" not in sourced
    assert sourced["PATH"] == env["PATH"]


@pytest.mark.parametrize("message", [
    "The timeout has reached and the application will be forcibly killed.",
    "Child process exited with status 139",
    "Instr info list is empty",
    "terminate called after throwing an instance of 'std::bad_alloc'",
])
def test_failure_overrides_profiler_success_banner(message):
    assert profile.profiler_failed(message + "\n" + profile.SUCCESS_TEXT)


def test_normal_child_exit_is_not_failure():
    assert not profile.profiler_failed("Child process exited with status 0\n" + profile.SUCCESS_TEXT)


@pytest.mark.parametrize("seconds,minutes", [(1, 1), (60, 1), (61, 2), (180, 3)])
def test_simulator_timeout_units(seconds, minutes):
    assert profile.simulator_timeout_minutes(seconds) == minutes


def test_invalid_timeout_rejected_before_toolchain_setup():
    with pytest.raises(SystemExit):
        profile.parse_args(["--msprof-timeout", "0"])
