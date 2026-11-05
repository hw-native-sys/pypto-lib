# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep standalone profiling allocations bounded and SDK types compatible."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


_GENERATOR = Path(__file__).resolve().parents[2] / ".claude/skills/incore-profiling/gen_profiling_case.py"
_SPEC = importlib.util.spec_from_file_location("incore_generator_bounds", _GENERATOR)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize(
    ("expression", "shape", "strides"),
    [
        ("%length = arith.addi %arg1, %c16_index : index", "%length", "%c1_index"),
        ("%length = arith.muli %arg1, %c1024_index : index", "%length", "%c1_index"),
        (
            "%scaled = arith.muli %arg1, %c1024_index : index\n"
            "%length = arith.addi %scaled, %c1_index : index",
            "%length", "%c1_index",
        ),
        ("%stride = arith.addi %arg1, %c16_index : index", "%c2_index, %c2_index", "%stride, %c1_index"),
    ],
)
def test_rejects_computed_dimensions_and_strides(expression: str, shape: str, strides: str) -> None:
    pto = (
        expression + "\n"
        f"%view = pto.make_tensor_view %arg0, shape = [{shape}], strides = [{strides}]"
    )

    with pytest.raises(ValueError, match="cannot safely bound computed PTO dimension"):
        _MODULE.parse_pto_sizes(pto, dynamic_dim=256)


@pytest.mark.parametrize("sdk_provides_type", [False, True])
def test_merge_sort_type_is_defined_once(tmp_path: Path, sdk_provides_type: bool) -> None:
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A host C++ compiler is unavailable")

    common = tmp_path / "pto/common"
    common.mkdir(parents=True)
    type_header = "#pragma once\n"
    if sdk_provides_type:
        type_header += (
            "#define _PTO_INCLUDE_NPU_TYPE_H_\n"
            "#include <stdint.h>\n"
            "namespace pto { struct MrgSortExecutedNumList {\n"
            "uint16_t mrgSortList0, mrgSortList1, mrgSortList2, mrgSortList3;\n"
            "}; }\n"
        )
    (common / "type.hpp").write_text(type_header, encoding="utf-8")
    (common / "constants.hpp").write_text("#pragma once\n", encoding="utf-8")
    (tmp_path / "pto/pto-inst.hpp").write_text('#include "common/type.hpp"\n', encoding="utf-8")
    source = tmp_path / "preamble.cpp"
    source.write_text(
        _MODULE._PREAMBLE + "\nint main() { pto::MrgSortExecutedNumList counts{}; return counts.mrgSortList0; }\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [compiler, "-std=c++17", "-D__CPU_SIM=1", "-fsyntax-only", "-I", str(tmp_path), str(source)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
