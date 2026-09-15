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


_SKILL = Path(__file__).resolve().parents[1] / ".claude/skills/incore-profiling"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SKILL / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = _load("incore_profile_under_test", "incore_profile.py")
gen = _load("gen_profiling_case_under_test", "gen_profiling_case.py")

_MIXED_PTO = """\
module {
  func.func @k_aic(%arg0: !pto.ptr<i8>, %arg1: !pto.ptr<f32>, %arg2: index, %__pypto_spmd_block_idx: i32, %__pypto_spmd_block_num: i32) {
  %c32_index = arith.constant 32 : index
  %c128_index = arith.constant 128 : index
  %c1_index = arith.constant 1 : index
  %7 = arith.muli %arg2, %c32_index : index
  %a_view = pto.make_tensor_view %arg0, shape = [%7, %c128_index], strides = [%c128_index, %c1_index] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xi8>
  }
  func.func @k_aiv(%arg0: !pto.ptr<i8>, %arg1: !pto.ptr<f32>, %arg2: index, %__pypto_spmd_block_idx: i32, %__pypto_spmd_block_num: i32, %__pypto_spmd_subblock_idx: i32) {
  %c16_index = arith.constant 16 : index
  %c1_index = arith.constant 1 : index
  %c32_index = arith.constant 32 : index
  %7 = arith.muli %c32_index, %arg2 : index
  %b_view = pto.make_tensor_view %arg1, shape = [%7, %c1_index], strides = [%c1_index, %7] {layout = #pto.layout<dn>} : !pto.tensor_view<?x?xf32>
  %7 = arith.muli %arg2, %c16_index : index
  }
}
"""
_MIXED_CPP = """\
AICORE void k_aic(__gm__ int8_t* v1, __gm__ float* v2, int64_t v3, int32_t v4, int32_t v5) {}
AICORE void k_aiv(__gm__ int8_t* v1, __gm__ float* v2, int64_t v3, int32_t v4, int32_t v5, int32_t v6) {}
"""


def test_arg_times_constant_extent_is_bounded_by_dynamic_dim():
    sizes, dynamic_args = gen.parse_pto_sizes(_MIXED_PTO, dynamic_dim=4096)
    assert sizes == {0: 4096 * 32 * 128, 1: 4096 * 32}
    assert dynamic_args == {2}


def test_other_computed_extent_is_still_rejected():
    pto = _MIXED_PTO.replace("arith.muli %arg2, %c32_index", "arith.addi %arg2, %c32_index")
    with pytest.raises(ValueError, match="cannot safely bound"):
        gen.parse_pto_sizes(pto)


def test_ssa_product_does_not_leak_across_functions():
    pto = "func.func @a(%arg0: index) {\n%7 = arith.muli %arg0, %c4_index : index\n}\n" \
        "func.func @b(%arg0: !pto.ptr<i8>) {\n" \
        "%v = pto.make_tensor_view %arg0, shape = [%7], strides = [%c1_index]\n}\n"
    with pytest.raises(ValueError, match="%7"):
        gen.parse_pto_sizes(pto)


def test_mixed_dispatcher_forwards_each_vector_lane():
    name, is_mixed, params = gen.parse_cpp(_MIXED_CPP)
    aiv_names = gen.parse_pto_param_names(_MIXED_PTO, "k_aiv")
    text = gen.emit_kernel_cpp(_MIXED_CPP, name, is_mixed, params, aiv_names)
    assert (
        "k_aiv(v1, v2, v3, v4, v5, static_cast<int32_t>(get_subblockid()) /* __pypto_spmd_subblock_idx */);"
        in text
    )


def test_spmd_scalar_defaults_run_block_zero_of_one():
    name, _, params = gen.parse_cpp(_MIXED_CPP)
    names = gen.parse_pto_param_names(_MIXED_PTO, "k_aic")
    counts = {p.name: 1 for p in params if p.is_ptr}
    text = gen.emit_main_cpp(name, params, counts, {2}, 4096, names)
    assert "int64_t v3 = 1;  // %arg2" in text
    assert "int32_t v4 = 0;  // %__pypto_spmd_block_idx" in text
    assert "int32_t v5 = 1;  // %__pypto_spmd_block_num" in text
    assert "> 4096.0L" in text


_NPU_SMI = """\
+------------------------------------------------------------------------------------------------+
| npu-smi 26.0.rc1                            Version: 26.0.rc1                                  |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                       |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)              |
+===========================+===============+====================================================+
| 0     910B1               | OK            | 98.1                 48                            |
| 0                         | 0000:C1:00.0  | 100                  0    / 0                      |
+===========================+===============+====================================================+
| 1     910B1               | OK            | 102.8                48                            |
| 0                         | 0000:01:00.0  | 100                  0    / 0                      |
+===========================+===============+====================================================+
"""
_SOCS = ["Ascend910B", "Ascend910B1", "Ascend910B2", "Ascend950"]


def test_npu_smi_variant_selects_exact_camodel_soc():
    chips = profile.parse_npu_smi_chip_names(_NPU_SMI)
    assert chips == ["910B1", "910B1"]
    assert profile.select_soc_version("a2a3", _SOCS, None, chips) == "Ascend910B1"


def test_explicit_soc_version_wins_over_npu_smi():
    assert profile.select_soc_version("a2a3", _SOCS, "Ascend910B2", ["910B1"]) == "Ascend910B2"


@pytest.mark.parametrize("chips", [[], ["910B4"]])
def test_ambiguous_soc_without_device_variant_requires_override(chips):
    with pytest.raises(profile.StepError, match="--soc-version"):
        profile.select_soc_version("a2a3", _SOCS, None, chips)


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
