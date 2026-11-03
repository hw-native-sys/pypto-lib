# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash multi-token prediction input projection."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import pypto.language as pl
import torch

from models.glm5_3_flash.config import FLASH, D
from models.glm5_3_flash.linear import bf16_projection, golden_bf16_linear
from models.glm5_3_flash.rmsnorm import rms_norm, rmsnorm
from models.glm5_3_flash.utils import Checkpoint

T_DYN = pl.dynamic('GLM53_T_DYN')


@pl.jit.inline
def mtp_projection(
    embedding: pl.Tensor[[T_DYN, D], pl.BF16],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    enorm_weight: pl.Tensor[[D], pl.BF16],
    hnorm_weight: pl.Tensor[[D], pl.BF16],
    w_eh: pl.Tensor[[D, 2 * D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    rows = pl.tensor.dim(embedding, 0)
    embedding_norm = pl.create_tensor([rows, D], dtype=pl.BF16)
    rmsnorm(embedding, enorm_weight, embedding_norm)
    hidden_norm = pl.create_tensor([rows, D], dtype=pl.BF16)
    rmsnorm(hidden, hnorm_weight, hidden_norm)
    combined = pl.create_tensor([rows, 2 * D], dtype=pl.BF16)
    for block in pl.spmd(((rows + 7) // 8) * (D // 128), name_hint="mtp_projection_concat"):
        row0 = block // (D // 128) * 8
        col0 = block % (D // 128) * 128
        valid_rows = pl.min(8, rows - row0)
        embedding_tile = pl.slice(embedding_norm, [8, 128], [row0, col0], valid_shape=[valid_rows, 128])
        hidden_tile = pl.slice(hidden_norm, [8, 128], [row0, col0], valid_shape=[valid_rows, 128])
        combined[row0 : row0 + 8, col0 : col0 + 128] = embedding_tile
        combined[row0 : row0 + 8, D + col0 : D + col0 + 128] = hidden_tile
    bf16_projection(combined, w_eh, enorm_weight, output, 2 * D, D, False)
    return output


@pl.jit
def mtp_projection_test(
    embedding: pl.Tensor[[T_DYN, D], pl.BF16],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    enorm_weight: pl.Tensor[[D], pl.BF16],
    hnorm_weight: pl.Tensor[[D], pl.BF16],
    w_eh: pl.Tensor[[D, 2 * D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    embedding.bind_dynamic(0, T_DYN)
    hidden.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mtp_projection(embedding, hidden, enorm_weight, hnorm_weight, w_eh, output)
    return output


def golden_mtp_projection(
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    w_eh: torch.Tensor,
) -> torch.Tensor:
    normalized_embedding = rms_norm(embedding, enorm_weight)
    normalized_hidden = rms_norm(hidden, hnorm_weight)
    combined = torch.cat((normalized_embedding, normalized_hidden), dim=-1)
    return golden_bf16_linear(combined, w_eh).bfloat16()


def main():
    import argparse
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hidden-input", type=Path)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5345)
        embedding = torch.randn(args.tokens, D, generator=generator).bfloat16()
        hidden = torch.randn(args.tokens, D, generator=generator).bfloat16()
        if args.checkpoint is None:
            values = {
                "embedding": embedding,
                "hidden": hidden,
                "enorm_weight": torch.ones(D, dtype=torch.bfloat16),
                "hnorm_weight": torch.ones(D, dtype=torch.bfloat16),
                "w_eh": (torch.randn(D, 2 * D, generator=generator) / (2 * D) ** 0.5).bfloat16(),
            }
        else:
            checkpoint = Checkpoint(args.checkpoint)
            prefix = f"model.language_model.layers.{FLASH.mtp_layer_id}."
            values = {
                name: checkpoint.read(prefix + suffix).bfloat16()
                for name, suffix in (
                    ("enorm_weight", "enorm.weight"),
                    ("hnorm_weight", "hnorm.weight"),
                    ("w_eh", "eh_proj.weight"),
                )
            }
            embedding_name = prefix + "embed_tokens.weight"
            values["embedding"] = checkpoint.read(embedding_name, slice(2000, 2000 + args.tokens)).bfloat16()
            values["hidden"] = hidden
        if args.hidden_input is not None:
            value = torch.load(args.hidden_input, map_location="cpu", weights_only=True)
            if value.shape != (args.tokens, D) or not torch.isfinite(value).all():
                raise ValueError("--hidden-input must contain finite [tokens, hidden_size] rows")
            values["hidden"] = value.bfloat16()
        values["embedding"][0].zero_()
        return values

    shapes = {
        "embedding": [args.tokens, D],
        "hidden": [args.tokens, D],
        "enorm_weight": [D],
        "hnorm_weight": [D],
        "w_eh": [D, 2 * D],
    }
    specs = [
        TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
        for name, shape in shapes.items()
    ]
    specs.append(TensorSpec("output", [args.tokens, D], torch.bfloat16))

    def golden(values):
        result = golden_mtp_projection(*(values[name] for name in shapes))
        values["output"].copy_(result)

    result = run(
        fn=mtp_projection_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        compare_fn={"output": ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0)},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
