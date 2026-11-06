# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify the real compressor references keep boundary results at token indices."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


def _load_golden(filename):
    path = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_lowlat" / filename
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "golden_compressor")
    namespace = {
        "__package__": "", "S": 8, "HEAD_DIM": 8, "OUT_DIM": 16, "COMPRESS_RATIO": 4,
        "ROPE_HEAD_DIM": 2, "COMPRESS_STATE_BLOCK_SIZE": 8, "CMP_STORAGE_BLOCK_SIZE": 8,
        "IDX_STORAGE_BLOCK_SIZE": 8, "FP32_NEG_INF": float("-inf"), "EPS": 1e-6,
        "INT8_AMAX_EPS": 1e-8, "INT8_SCALE_MAX": 127.0,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["golden_compressor"]


class CompressorBoundaryGoldenTests(unittest.TestCase):
    def test_two_boundaries_use_token_indexed_rope_and_output(self):
        x = (torch.arange(64, dtype=torch.float32).reshape(1, 8, 8) + 1) / 10
        identity = torch.eye(8, dtype=torch.float32)
        positions = torch.arange(8, dtype=torch.int32).reshape(1, 8)
        mapping = torch.tensor([[-1, -1, -1, 0, -1, -1, -1, 1]], dtype=torch.int64)
        cosine = torch.arange(8, dtype=torch.float32).reshape(8, 1) / 10 + 0.1
        sine = torch.arange(8, dtype=torch.float32).reshape(8, 1) / 20 + 0.05
        fake_utils = SimpleNamespace(bf16_matmul_fp32=lambda left, right: left @ right)
        for filename, is_indexer in (("decode_compressor_ratio4.py", False), ("decode_indexer_compressor.py", True)):
            with self.subTest(filename=filename), patch.dict(sys.modules, {"utils": fake_utils}):
                tensors = {
                    "x": x, "kv": torch.empty(1, 8, 8), "wkv": torch.cat((identity, identity * 2)),
                    "wgate": torch.zeros(16, 8), "ape": torch.zeros(4, 16), "norm_w": torch.ones(8),
                    "cos": cosine, "sin": sine, "compress_state": torch.zeros(1, 8, 32),
                    "compress_state_block_table": torch.zeros(1, 1, dtype=torch.int32),
                    "position_ids": positions,
                }
                if is_indexer:
                    tensors.update(hadamard=identity, idx_kv_cache=torch.zeros(1, 8, 1, 8, dtype=torch.int8),
                                   idx_kv_scale=torch.zeros(1, 8, 1, 1), idx_slot_mapping=mapping,
                                   inner_state_slot_mapping=torch.full_like(mapping, -1))
                else:
                    tensors.update(cmp_kv_cache=torch.zeros(1, 8, 1, 8, dtype=torch.bfloat16),
                                   cmp_slot_mapping=mapping, state_slot_mapping=torch.full_like(mapping, -1))
                _load_golden(filename)(tensors)
                self.assertTrue(torch.isnan(tensors["kv"][0, [0, 1, 2, 4, 5, 6]]).all())
                for token in (3, 7):
                    pooled = 2 * x[0, :4].mean(dim=0) if token == 3 else (
                        x[0, :4].sum(dim=0) + 2 * x[0, 4:8].sum(dim=0)
                    ) / 8
                    expected = pooled * torch.rsqrt(pooled.square().mean() + 1e-6)
                    last = expected[-2:].clone()
                    expected[-2] = last[0] * cosine[token, 0] - last[1] * sine[token, 0]
                    expected[-1] = last[0] * sine[token, 0] + last[1] * cosine[token, 0]
                    if is_indexer:
                        expected = expected.to(torch.bfloat16).float()
                    torch.testing.assert_close(tensors["kv"][0, token], expected, rtol=1e-5, atol=1e-5)
                    if not is_indexer:
                        torch.testing.assert_close(tensors["cmp_kv_cache"][0, token // 4, 0],
                                                   expected.to(torch.bfloat16), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
