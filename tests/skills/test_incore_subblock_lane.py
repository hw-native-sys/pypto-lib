# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The standalone profiler must preserve per-AIV branch selection."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_GENERATOR = Path(__file__).resolve().parents[2] / ".claude/skills/incore-profiling/gen_profiling_case.py"
_SPEC = importlib.util.spec_from_file_location("gen_profiling_case", _GENERATOR)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
wire_subblock_lane = _MODULE.wire_subblock_lane

_PTO = """\
module {
  func.func @sample_aiv(%arg0: !pto.ptr<f32>, %arg1: i32, %__pypto_spmd_subblock_idx: i32)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    return
  }
}
"""
_AIV = "AICORE void sample_aiv(__gm__ float* v1, int32_t v2, int32_t v3) {}\n"
_ENTRY = 'extern "C" __global__ AICORE void sample(__gm__ float* v1, int32_t v2, int32_t v3) {\n'


@pytest.mark.parametrize(
    ("wrapper", "expected"),
    [
        (_ENTRY + "  sample_aiv(v1, v2, v3);\n}\n", "v3 = get_subblockid();"),
        (
            'extern "C" __global__ AICORE void sample(__gm__ float* v1, int32_t v2) {\n'
            "#if defined(__DAV_VEC__)\n  sample_aiv(v1, v2, 0);\n#endif\n}\n",
            "sample_aiv(v1, v2, get_subblockid());",
        ),
    ],
)
def test_wires_each_vector_core_to_its_lane(tmp_path: Path, wrapper: str, expected: str) -> None:
    pto = tmp_path / "sample.pto"
    kernel = tmp_path / "sample_kernel.cpp"
    pto.write_text(_PTO, encoding="utf-8")
    kernel.write_text(_AIV + wrapper, encoding="utf-8")

    assert wire_subblock_lane(kernel, pto, "sample") == "get_subblockid()"
    patched = kernel.read_text(encoding="utf-8")
    assert expected in patched
    assert wire_subblock_lane(kernel, pto, "sample") == "get_subblockid()"
    assert kernel.read_text(encoding="utf-8") == patched


def test_leaves_unrelated_scalar_alone(tmp_path: Path) -> None:
    pto = tmp_path / "sample.pto"
    kernel = tmp_path / "sample_kernel.cpp"
    pto.write_text(_PTO.replace("__pypto_spmd_subblock_idx", "arg2"), encoding="utf-8")
    original = _AIV + _ENTRY + "  sample_aiv(v1, v2, v3);\n}\n"
    kernel.write_text(original, encoding="utf-8")

    assert wire_subblock_lane(kernel, pto, "sample") == ""
    assert kernel.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "source_text",
    [
        _AIV + _ENTRY + "  sample_aiv(v1, v2, v3);\n}\n",
        "AICORE void sample_aic(__gm__ float* v1, int32_t v2) {}\n" + _AIV,
    ],
)
def test_profiler_uses_declared_name_when_source_filename_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_text: str
) -> None:
    monkeypatch.syspath_prepend(str(_GENERATOR.parent))
    import incore_profile

    source = tmp_path / "renamed.cpp"
    cpp = _AIV + _ENTRY + "  sample_aiv(v1, v2, v3);\n}\n"
    source.write_text(source_text, encoding="utf-8")
    source.with_suffix(".pto").write_text(_PTO, encoding="utf-8")

    def fake_run_cmd(cmd: list[str], **_kwargs: object) -> None:
        if cmd[0] == sys.executable:
            case = tmp_path / "run/cases/ptoas/renamed_msprof/renamed_msprof_kernel.cpp"
            case.parent.mkdir(parents=True)
            case.write_text(cpp, encoding="utf-8")
            return
        raise incore_profile.StepError("stop after lane wiring")

    monkeypatch.setattr(incore_profile, "run_cmd", fake_run_cmd)
    args = SimpleNamespace(
        generate_testcase=incore_profile.GEN_PROFILING_CASE,
        soc_version="Ascend910B1",
        aicore_arch="dav-c220",
        dynamic_dim=256,
        step_timeout=60,
        pto_isa_root=tmp_path,
    )
    result = incore_profile.export_one(source, args, {}, tmp_path / "run", 1, 1)

    assert result["subblock_lane"] == "get_subblockid()"
    assert "v3 = get_subblockid();" in (
        tmp_path / "run/cases/ptoas/renamed_msprof/renamed_msprof_kernel.cpp"
    ).read_text(encoding="utf-8")
