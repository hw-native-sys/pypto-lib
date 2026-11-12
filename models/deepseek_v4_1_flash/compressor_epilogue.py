# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fused RMSNorm, pre-RoPE latent, RoPE, MXFP4 quantization, and cache scatter."""

from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "compressor_publish.cpp",
               include_dirs=(pto_isa_include_dir(), str(Path(__file__).parent / "kernels")))
def _compressor_epilogue(
    source: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    norm_weight: pl.Tensor,
    latent: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    round_even: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Normalize, retain unrotated latent, and publish E2M1 rows."""
    ...


@pl.jit.inline(auto_scope=False)
def compressor_epilogue(
    source: pl.Tensor,
    norm_weight: pl.Tensor,
    latent: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    source_ready: pl.Scalar[pl.TASK_ID],
    round_even: pl.constexpr,
):
    """Retain normalized latent and publish rows with one owner per cache slot."""
    active = pl.cast(num_tokens, pl.INT32)
    workers = pl.cast(64, pl.INT32)
    if active <= 128:
        workers = pl.cast(16, pl.INT32)
    with pl.spmd(workers, name_hint="compressor_epilogue", deps=[cache_ready, source_ready]) as published:
        _compressor_epilogue(
            source, slots, cache, cosine, sine, norm_weight, latent, active, pl.cast(round_even, pl.INT32),
        )
    return published


@pl.jit
def compressor_epilogue_test(
    source: pl.Tensor[[pl.dynamic("rows"), 512], pl.BF16],
    norm_weight: pl.Tensor[[512], pl.BF16],
    cosine: pl.Tensor[[pl.dynamic("rows"), 32], pl.FP32],
    sine: pl.Tensor[[pl.dynamic("rows"), 32], pl.FP32],
    slots: pl.Tensor[[pl.dynamic("rows")], pl.INT64],
    latent: pl.InOut[pl.Tensor[[pl.dynamic("rows"), 512], pl.BF16]],
    cache: pl.InOut[pl.Tensor[[2, 128, 1, 320], pl.UINT8]],
    num_tokens: pl.Scalar[pl.INT32],
    round_even: pl.Scalar[pl.INT32],
):
    ready = pl.system.task_dummy(deps=[])
    active = pl.cast(num_tokens, pl.INT32)
    workers = pl.cast(64, pl.INT32)
    if active <= 128:
        workers = pl.cast(16, pl.INT32)
    with pl.spmd(workers, name_hint="compressor_epilogue", deps=[ready]):
        _compressor_epilogue(source, slots, cache, cosine, sine, norm_weight, latent, active, round_even)
    return latent, cache


def golden_compressor_epilogue(tensors):
    from qkv_proj_rope import golden_rope_tail
    from quant import pack_quantized_cache, quantize_mxfp4_cache
    from rmsnorm import golden_rms_norm

    active = int(tensors["num_tokens"])
    normalized = golden_rms_norm(tensors["source"][:active], tensors["norm_weight"])
    tensors["latent"][:active].copy_(normalized)
    rotated = golden_rope_tail(normalized, tensors["cosine"][:active], tensors["sine"][:active])
    payload, scales = quantize_mxfp4_cache(
        rotated, 16, "bf16", round_to_even=bool(tensors["round_even"]),
    )
    packed = pack_quantized_cache(payload, scales)
    for row in range(active):
        slot = int(tensors["slots"][row])
        if slot >= 0:
            tensors["cache"].view(-1, 320)[slot].copy_(packed[row])


if __name__ == "__main__":
    import argparse

    import torch

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from utils import packed_cache_compare

    parser = argparse.ArgumentParser(description="Fused compressor epilogue validation")
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--round-even", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(42)
    for rows, active in ((97, 94), (1103, 1100)):
        source = torch.randn(rows, 512, generator=generator).bfloat16()
        source[0].zero_()
        slots = torch.arange(rows, dtype=torch.int64) % 256
        slots[1::4] = -1
        slots[8] = slots[0]
        slots[72] = slots[8]
        angles = torch.randn(rows, 32, generator=generator)
        if active > 1024:
            # Distinct writes to four slots straddle the metadata chunk boundary.
            slots[1032:1040:2] = slots[1000:1008:2]
            source[1032:1040:2] = -source[1000:1008:2]
            angles[1032:1040:2] = angles[1000:1008:2]
        specs = [
            TensorSpec("source", [rows, 512], torch.bfloat16, init_value=source),
            TensorSpec("norm_weight", [512], torch.bfloat16, init_value=1.0),
            TensorSpec("cosine", [rows, 32], torch.float32, init_value=angles.cos()),
            TensorSpec("sine", [rows, 32], torch.float32, init_value=angles.sin()),
            TensorSpec("slots", [rows], torch.int64, init_value=slots),
            TensorSpec("latent", [rows, 512], torch.bfloat16, init_value=0.25),
            TensorSpec("cache", [2, 128, 1, 320], torch.uint8, init_value=37),
            ScalarSpec("num_tokens", torch.int32, active),
            ScalarSpec("round_even", torch.int32, args.round_even),
        ]
        print(f"Validating rows={rows}, active={active}, round_even={args.round_even}", flush=True)
        result = run(
            fn=compressor_epilogue_test, specs=specs, golden_fn=golden_compressor_epilogue,
            compare_fn={
                "latent": ratio_allclose(atol=1e-3, rtol=1.0 / 128),
                "cache": packed_cache_compare(
                    ratio_allclose(atol=0.0, rtol=0.0), ratio_allclose(atol=1e-3, rtol=1.0 / 128), fp4=True,
                ),
            },
            config={"backend_type": "a5", "device_id": args.device,
                    "execution_mode": args.execution_mode or args.platform == "a5sim"},
            compile_only=args.compile_only,
        )
        print(result)
        if not result.passed:
            raise SystemExit(1)
