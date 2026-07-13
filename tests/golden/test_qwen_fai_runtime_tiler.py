# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side coverage for the Qwen FAI runtime tiler."""

import ast
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QWEN = ROOT / "models" / "qwen3" / "14b"


def _int_constant(name: str) -> int:
    tree = ast.parse((QWEN / "paged_attention_cce.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not defined")


def test_runtime_tiler_fits_launch_and_workspace_for_all_b1_lengths(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    assert compiler is not None, "a host C++ compiler is required for the runtime tiler test"

    source = tmp_path / "runtime_tiler_test.cpp"
    binary = tmp_path / "runtime_tiler_test"
    source.write_text(
        f"""
#include <cstdint>

#include "models/qwen3/14b/kernels/paged_attention_cce/tiling/qwen_fai_runtime_tiler.hpp"

int main() {{
    constexpr uint64_t kWorkspaceBytes = {_int_constant("WORKSPACE_BYTES")};
    static_assert(qwen_fai_tiler::kPlanningCoreNum == {_int_constant("DEFAULT_BLOCK_DIM")});
    static_assert(qwen_fai_tiler::kFlashDecodeLaunchCoreNum == {_int_constant("B1_BLOCK_DIM")});
    static_assert(
        qwen_fai_tiler::kFlashDecodeMinKvSeqlen == {_int_constant("FLASH_DECODE_MIN_KV_SEQLEN")}
    );
    uint32_t seq_lens[16] = {{}};
    int64_t cumulative_q[16] = {{}};
    int64_t kv_lengths[16] = {{}};

    for (uint32_t seq_len = 1; seq_len <= 4096; ++seq_len) {{
        seq_lens[0] = seq_len;
        FAInferTilingData tiling{{}};
        if (!qwen_fai_tiler::build(seq_lens, 1, 32, 32, tiling, cumulative_q, kv_lengths)) return 1;
        if (tiling.needCoreNum > qwen_fai_tiler::kFlashDecodeLaunchCoreNum) return 2;
        if (tiling.workSpaceSize > kWorkspaceBytes) return 3;
        if (seq_len < qwen_fai_tiler::kFlashDecodeMinKvSeqlen &&
            (tiling.needCoreNum != 0 || tiling.totalSplitNodeNum != 0)) return 4;
        if (seq_len >= qwen_fai_tiler::kFlashDecodeMinKvSeqlen && tiling.needCoreNum == 0) return 5;
        if (seq_len == 3338 &&
            (tiling.needCoreNum != 19 || tiling.totalSplitNodeNum != 8 || tiling.workSpaceSize != 66122208)) {{
            return 6;
        }}
    }}

    for (uint32_t batch_idx = 0; batch_idx < 16; ++batch_idx) seq_lens[batch_idx] = 3338;
    FAInferTilingData tiling{{}};
    if (!qwen_fai_tiler::build(seq_lens, 16, 32, 512, tiling, cumulative_q, kv_lengths)) return 7;
    if (tiling.batch != 16 || tiling.needCoreNum != 0 || tiling.totalTaskNum != 128 ||
        tiling.workSpaceSize != 66060288) {{
        return 8;
    }}

    return 0;
}}
""",
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(source), "-o", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run([binary], check=False, capture_output=True, text=True)
    assert run_result.returncode == 0, run_result.stderr


def test_attention_entry_selects_flash_decode_from_tiler_metadata() -> None:
    source = (
        QWEN / "kernels" / "paged_attention_cce" / "attention" / "entry.cpp"
    ).read_text(encoding="utf-8")

    assert "if (tiling->needCoreNum != 0)" in source
    assert "if (tiling->batch == 1)" not in source
