# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scatter merged vision embeddings into all hyper-connection lanes of selected text rows."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, HC_MULT, T_DYN, VISION_ROWS_DYN

FUSION_D_TILE = 128


def golden_vision_fusion(
    inputs_embeds: torch.Tensor,
    vision_embeds: torch.Tensor,
    placeholder_mask: torch.Tensor,
) -> torch.Tensor:
    if placeholder_mask.dtype != torch.bool or placeholder_mask.shape != inputs_embeds.shape[:1]:
        raise ValueError("placeholder mask must be boolean with one entry per text row")
    if int(placeholder_mask.sum()) != vision_embeds.shape[0]:
        raise ValueError("placeholder count does not match the merged vision token count")
    fused = inputs_embeds.clone()
    fused[placeholder_mask] = vision_embeds.to(inputs_embeds.dtype)
    return fused


@pl.jit.inline
def vision_fusion(
    vision_embeds: pl.Tensor[[VISION_ROWS_DYN, D], pl.BF16],
    placeholder_rows: pl.Tensor[[VISION_ROWS_DYN], pl.INT32],
    hidden_streams: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    rows = pl.tensor.dim(vision_embeds, 0)
    for block in pl.spmd(rows * (D // FUSION_D_TILE), name_hint="vision_placeholder_scatter"):
        source_row = block // (D // FUSION_D_TILE)
        col0 = block % (D // FUSION_D_TILE) * FUSION_D_TILE
        target_row = pl.cast(pl.read(placeholder_rows, [source_row]), pl.INDEX)
        values = vision_embeds[source_row:source_row + 1, col0:col0 + FUSION_D_TILE]
        stream_values = pl.reshape(values, [1, 1, FUSION_D_TILE])
        for lane in pl.unroll(HC_MULT):
            hidden_streams[target_row:target_row + 1, lane:lane + 1, col0:col0 + FUSION_D_TILE] = stream_values
    return hidden_streams


@pl.jit
def vision_fusion_test(
    vision_embeds: pl.Tensor[[VISION_ROWS_DYN, D], pl.BF16],
    placeholder_rows: pl.Tensor[[VISION_ROWS_DYN], pl.INT32],
    hidden_streams: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    vision_embeds.bind_dynamic(0, VISION_ROWS_DYN)
    placeholder_rows.bind_dynamic(0, VISION_ROWS_DYN)
    hidden_streams.bind_dynamic(0, T_DYN)
    vision_fusion(vision_embeds, placeholder_rows, hidden_streams)
    return hidden_streams


def main():
    import argparse
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run
    from models.glm5_3_flash.weights import Checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--vision-rows", type=int, default=5)
    parser.add_argument("--vision-input", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    options = parser.parse_args()
    if not 1 <= options.vision_rows <= options.tokens:
        parser.error("require 1 <= --vision-rows <= --tokens")

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5354)
        streams = torch.randn(options.tokens, HC_MULT, D, generator=generator).bfloat16()
        vision = torch.randn(options.vision_rows, D, generator=generator).bfloat16()
        if options.checkpoint is not None:
            checkpoint = Checkpoint(options.checkpoint)
            embedding = checkpoint.read("model.language_model.embed_tokens.weight", slice(2000, 2000 + options.tokens))
            streams = embedding[:, None, :].expand(-1, HC_MULT, -1).clone()
        if options.vision_input is not None:
            vision = torch.load(options.vision_input, map_location="cpu", weights_only=True)
            if tuple(vision.shape) != (options.vision_rows, D) or not torch.isfinite(vision).all():
                raise ValueError("--vision-input must contain finite [vision_rows, hidden_size] features")
            vision = vision.bfloat16()
        rows = torch.randperm(options.tokens, generator=generator)[:options.vision_rows].int()
        return {"vision_embeds": vision, "placeholder_rows": rows, "hidden_streams": streams}

    shapes = {"vision_embeds": [options.vision_rows, D], "placeholder_rows": [options.vision_rows],
              "hidden_streams": [options.tokens, HC_MULT, D]}
    specs = [TensorSpec(name, shape, torch.int32 if name == "placeholder_rows" else torch.bfloat16,
                        init_value=lambda key=name: inputs()[key]) for name, shape in shapes.items()]

    def golden(values):
        rows = values["placeholder_rows"].long()
        mask = torch.zeros(options.tokens, dtype=torch.bool)
        mask[rows] = True
        ordered_features = values["vision_embeds"][rows.argsort()]
        for lane in range(HC_MULT):
            fused = golden_vision_fusion(values["hidden_streams"][:, lane], ordered_features, mask)
            values["hidden_streams"][:, lane].copy_(fused)

    result = run(
        fn=vision_fusion_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"hidden_streams": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
