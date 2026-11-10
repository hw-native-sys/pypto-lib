# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execute the migrated model epilogues against torch without model-sized weights.

Read the actual lane bodies from the model so this checks its addresses and
math rather than a second implementation. Small matmuls supply their cube inputs;
nonzero row/column bases expose accidental writes to another lane's output.
"""

import ast
import textwrap
from pathlib import Path

import pytest
import pypto
import torch

pytestmark = pytest.mark.skipif(
    getattr(pypto, "__pypto_stub__", False), reason="requires the real PyPTO compiler"
)


def _model_region(shard_name: str) -> str:
    path = Path(__file__).resolve().parents[1] / "models/qwen3_14b/decode_tq_draft.py"
    tree = ast.parse(path.read_text())
    regions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and node.iter.func.attr == "split_aiv"
        and any(
            isinstance(stmt, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == shard_name for target in stmt.targets)
            for stmt in node.body
        )
    ]
    assert len(regions) == 1
    return ast.unparse(regions[0])


@pytest.mark.parametrize("phase", ["out", "gate_up", "down"])
@pytest.mark.parametrize("backend_name", ["Ascend910B", "Ascend950"])
def test_tq_explicit_epilogue_matches_torch(phase, backend_name, tmp_path):
    from pypto import backend
    from pypto.backend import pto_backend
    from pypto.backend._ptoas_locate import find_ptoas_binary
    from pypto.debug import torch_codegen
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager
    from pypto.language.parser import parse

    torch.manual_seed(7)
    x = torch.randn(16, 64).to(torch.bfloat16)
    width = 256 if phase == "gate_up" else 64
    weight = torch.randn(64, width).to(torch.bfloat16)
    acc = x.float() @ weight.float()
    constants = "BATCH_TILE = 16\nQ_OUT_CHUNK = 64\nK_CHUNK = 64\nb0 = 16\no0 = 32\nd0 = 32\nmlp_o0 = 128\n"
    if phase == "out":
        bias = torch.randn(32, 128).to(torch.bfloat16)
        output = torch.full((16, 128), -123.0)
        expected = output.clone()
        expected[:, 32:96] = acc + bias[16:32, 32:96].float()
        extra = "current_hidden: pl.Tensor[[32, 128], pl.BF16], resid1_tile: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]"
        prefix, name, result = "o_acc = pl.matmul(x, weight, out_dtype=pl.FP32)", "o_half", "resid1_tile"
        args = [x, weight, bias, output]
    elif phase == "down":
        bias = torch.randn(16, 128)
        output = torch.full((32, 128), -123.0, dtype=torch.bfloat16)
        expected = output.clone()
        expected[16:32, 32:96] = (acc + bias[:, 32:96]).to(torch.bfloat16)
        extra = (
            "resid1_tile: pl.Tensor[[16, 128], pl.FP32], next_hidden: pl.InOut[pl.Tensor[[32, 128], pl.BF16]]"
        )
        prefix, name, result = (
            "down_acc = pl.matmul(x, weight, out_dtype=pl.FP32)",
            "down_half",
            "next_hidden",
        )
        args = [x, weight, bias, output]
    else:
        up_weight = torch.randn(64, 256).to(torch.bfloat16)
        up = x.float() @ up_weight.float()
        output = torch.full((16, 512), -123.0, dtype=torch.bfloat16)
        expected = output.clone()
        expected[:, 128:384] = (acc * torch.sigmoid(acc) * up).to(torch.bfloat16)
        extra = "up_weight: pl.Tensor[[64, 256], pl.BF16], mlp_tile: pl.InOut[pl.Tensor[[16, 512], pl.BF16]]"
        prefix = "gate_acc = pl.matmul(x, weight, out_dtype=pl.FP32)\nup_acc = pl.matmul(x, up_weight, out_dtype=pl.FP32)"
        name, result = "gate_half", "mlp_tile"
        args = [x, weight, up_weight, output]
    source = constants + (
        "@pl.program\nclass Epilogue:\n    @pl.function\n"
        f"    def main(self, x: pl.Tensor[[16, 64], pl.BF16], weight: pl.Tensor[[64, {width}], pl.BF16], {extra}):\n"
        "        with pl.at(level=pl.Level.CORE_GROUP):\n"
        + textwrap.indent(prefix + "\n" + _model_region(name), "            ")
        + f"\n        return {result}\n"
    )
    backend.reset_for_testing()
    backend.set_backend_type(getattr(backend.BackendType, backend_name))
    try:
        program = parse(source)
        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
        files = pto_backend.generate(lowered, str(tmp_path), skip_ptoas=True)
        pto = "\n".join(content for name, content in files.items() if name.endswith(".pto"))
        assert "pto.tpush_to_aiv" in pto and "pto.tpop_from_aic" in pto
        assert "__FREE_VAR" not in pto
        if find_ptoas_binary() is not None:
            native = pto_backend.generate(lowered, str(tmp_path), skip_ptoas=False)
            assert any(name.endswith(".cpp") for name in native)
        code = torch_codegen(lowered, check_shapes=True)
        namespace = {}
        exec(code, namespace)
        namespace["main"](*args)
    finally:
        backend.reset_for_testing()
    torch.testing.assert_close(output, expected, rtol=5e-2, atol=5e-2)
