# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""BF16 projection of flattened temporal image patches with a learned bias."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import T_DYN
from models.glm5_3_flash.linear import bf16_projection, golden_bf16_linear
from models.glm5_3_flash.vision.config import VISION_HIDDEN, VISION_PATCH_IN


def golden_vision_patch_embed(patches: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return (golden_bf16_linear(patches, weight) + bias.float()).bfloat16()


@pl.jit.inline
def vision_patch_embed(
    patches: pl.Tensor[[T_DYN, VISION_PATCH_IN], pl.BF16],
    weight: pl.Tensor[[VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16],
):
    bf16_projection(patches, weight, bias, output, VISION_PATCH_IN, VISION_HIDDEN, True)
    return output


@pl.jit
def vision_patch_embed_test(
    patches: pl.Tensor[[T_DYN, VISION_PATCH_IN], pl.BF16],
    weight: pl.Tensor[[VISION_HIDDEN, VISION_PATCH_IN], pl.BF16],
    bias: pl.Tensor[[VISION_HIDDEN], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, VISION_HIDDEN], pl.BF16]],
):
    patches.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    vision_patch_embed(patches, weight, bias, output)
    return output


def main():
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run
    from models.glm5_3_flash.utils import component_hidden, component_options
    from models.glm5_3_flash.weights import Checkpoint

    options = component_options(__doc__)

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5350)
        values = {"patches": component_hidden(options, VISION_PATCH_IN, generator)}
        if options.checkpoint is None:
            values["weight"] = (torch.randn(VISION_HIDDEN, VISION_PATCH_IN, generator=generator)
                                / VISION_PATCH_IN ** 0.5).bfloat16()
            values["bias"] = (torch.randn(VISION_HIDDEN, generator=generator) * 0.1).bfloat16()
        else:
            checkpoint = Checkpoint(options.checkpoint)
            shape = (VISION_HIDDEN, 3, 2, 14, 14)
            values["weight"] = checkpoint.vision_tensor("patch_embed.proj.weight", shape).flatten(1)
            values["bias"] = checkpoint.vision_tensor("patch_embed.proj.bias", (VISION_HIDDEN,))
        return values

    shapes = {"patches": [options.tokens, VISION_PATCH_IN],
              "weight": [VISION_HIDDEN, VISION_PATCH_IN], "bias": [VISION_HIDDEN]}
    specs = [TensorSpec(name, shape, torch.bfloat16, init_value=lambda key=name: inputs()[key])
             for name, shape in shapes.items()]
    specs.append(TensorSpec("output", [options.tokens, VISION_HIDDEN], torch.bfloat16))

    def golden(values):
        values["output"].copy_(golden_vision_patch_embed(*(values[name] for name in shapes)))

    result = run(
        fn=vision_patch_embed_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"output": ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
