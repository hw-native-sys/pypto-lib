# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC coefficient generation, stream collapse, residual expansion, and final collapse.

GLM-5.3-Flash carries **two** hyper-connection sites per backbone layer — one around
attention (``hc_attn_fn`` / ``hc_attn_scale`` / ``hc_attn_base``) and one around the
MLP (``hc_ffn_*``) — for 90 sites over 45 layers. The MTP layer has no ``hc_*``
weights and uses a plain residual instead.

``Glm5NextTextHyperConnection`` subclasses ``DeepseekV4HyperConnection`` with no
changes, so the coefficient algebra is the same as ``models/deepseek_v4_1_flash/mhc.py``.
Two things differ and both change the ABI:

* **The stream is BF16, not FP32.** It is seeded by replicating the BF16 embedding
  (``inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)``) and the decoder layer
  casts the mixes into the stream dtype. DeepSeek-V4 keeps its HC stream in FP32, so
  a kernel cannot be lifted across without changing the tile dtypes. At width 4 this
  is 32 KB of live residual per token instead of 64 KB.
* **The head is an unweighted mean** (``Glm5NextTextHyperHead``), where DeepSeek-V4
  applies the last layer's delayed pre-mix, so ``mhc_head`` takes no mix input. The
  model tail is ``norm(hc_head(streams))``.

The coefficients themselves stay FP32: the projection accumulates in FP32 and the
two sigmoids, the softmax and all 20 Sinkhorn iterations must not be computed in
BF16. All three weights are stored BF16 in the deployment checkpoint, read from the
shard headers: ``fn`` ``[24, 16384]``, ``base`` ``[24]`` and ``scale`` ``[3]``, so
the loader upcasts ``scale`` and ``base`` before the coefficient maths. Note the
checkpoint spells them ``layers.N.attn_hc.*`` and ``layers.N.ffn_hc.*``, not the
``hc_attn_*`` / ``hc_ffn_*`` parameter names the reference implementation uses.

**Prior art.** ``ops/pypto_python/impl/hc_pre_pypto.py`` in cann-recipes-infer is a
pypto implementation of the hyper-connection pre-collapse on this hardware
generation. It targets the older ``pypto.Tensor`` / ``pypto_impl`` frontend, which
this repo's pinned pypto does not export, so treat it as an algorithm and tiling
reference rather than code.

**Donors, in the right order.** ``models/deepseek_v4_flash_mtp/hc_pre.py`` (434
lines), ``hc_post.py`` (237) and ``hc_head.py`` (288) are implemented, carry no
``# ci:`` tag — so the a2a3 daily sweep runs them — and are the ones to port.
``models/deepseek_v4_flash_dspark/`` has a second implemented set (573 / 239 / 386)
for the larger batch. ``models/deepseek_v4_1_flash/mhc.py`` (559 lines, landed as
#1247) consolidates the four into one file and is the cleanest read, **but it is
a5-only** (``# ci: no-sim`` + ``# ci: a5``, and it splits the entry sentinel so the
a2a3 sweep cannot pick it up), so take its structure, not its tiling.

Against any of them the port is: the FP32 stream becomes BF16, the learned head
becomes an unweighted mean, and the router-side scoring is unrelated.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, FLASH, HC_DIM, HC_MULT, MIX_HC, T_DYN
from models.glm5_3_flash.golden import hc_head, hc_mixes, hc_post, hc_pre

MIX_D_TILE = 256        # D-wise tile of the three stream-collapse scopes
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = FLASH.hc_sinkhorn_iters
HC_EPS = FLASH.hc_eps
NORM_EPS = FLASH.rms_norm_eps
MIX_PAD = 32            # MIX_HC = 24 padded to the cube's N granularity
HC_PAD = 8              # HC_MULT = 4 padded so an FP32 row is 32 bytes
RMS_T_TILE = 8          # token tile of the sum-of-squares scope
LINEAR_T_TILE = 16      # cube M tile; below 16 the matmul is not L0 split
COMB_T_TILE = 8         # token tile of the softmax / Sinkhorn scope
RMS_K_TILE = 512        # HC_DIM reduction chunk of the sum of squares
LINEAR_K_TILE = 256     # HC_DIM reduction chunk of the projection
LINEAR_OK = 4           # split-K fan-out of the projection
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK


def golden_mhc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return hc_mixes(x_hc, function, scale, base)


def golden_mhc_pre(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return hc_pre(x_hc, pre_mix).to(torch.bfloat16)


def golden_mhc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    return hc_post(sublayer, residual, post_mix, residual_mix).to(torch.bfloat16)


def golden_mhc_head(x_hc: torch.Tensor) -> torch.Tensor:
    return hc_head(x_hc)


@pl.jit.inline
def mhc_mixes(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
):
    raise NotImplementedError("mHC coefficient kernel body is assigned independently")


@pl.jit.inline
def mhc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Collapse the four streams into one sublayer input, one block per token.

    ``sum_m pre_mix[t, m] * x_hc[t, m, :]`` accumulated in FP32 and rounded once, which
    is what the reference does. The mix is a per-token scalar, so it is read rather
    than broadcast from a tile: the four streams of a token are four ``D``-wide rows of
    the flattened stream, and the D loop pipelines over them.

    The stream is BF16 here where DeepSeek-V4 keeps it FP32, so every stream tile is
    widened before the multiply and only the result is rounded back.

    Returns the collapsed rows so the caller can chain them.
    """
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for t in pl.spmd(t_dim, name_hint="mhc_pre"):
        for d0 in pl.pipeline(0, D, MIX_D_TILE, stage=2):
            value = pl.mul(
                pl.cast(x_flat[t : t + 1, d0 : d0 + MIX_D_TILE], target_type=pl.FP32),
                pl.read(pre_mix, [t, 0]),
            )
            for stream in pl.unroll(HC_MULT - 1):
                head = stream + 1
                tile = x_flat[t : t + 1, head * D + d0 : head * D + d0 + MIX_D_TILE]
                value = pl.add(
                    value,
                    pl.mul(pl.cast(tile, target_type=pl.FP32), pl.read(pre_mix, [t, head])),
                )
            output[t : t + 1, d0 : d0 + MIX_D_TILE] = pl.cast(
                value, target_type=pl.BF16, mode="rint"
            )
    return output


@pl.jit.inline
def mhc_post(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    """Expand the sublayer result into the streams and mix them, one block per stream.

    ``output[t, j, :] = post_mix[t, j] * sublayer[t, :] + sum_i residual_mix[t, i, j] *
    residual[t, i, :]``. Both mixes are per-token scalars, so a block owns one output
    stream of one token and reads its four coefficients; the D loop pipelines and the
    stream loop unrolls, so the four residual rows are read once per D tile.

    Donor: ``models/deepseek_v4_1_flash/hc_post.py``. What changes is the dtype — the
    GLM stream is BF16, so the residual tiles are widened on the way in and the result
    is rounded once on the way out, where the donor carries FP32 throughout and has to
    round twice to match its reference.

    Returns the mixed streams so the caller can chain them.
    """
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    for block in pl.spmd(t_dim * HC_MULT, name_hint="mhc_post"):
        t = block // HC_MULT
        out_h = block % HC_MULT
        for d0 in pl.pipeline(0, D, MIX_D_TILE, stage=2):
            x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + MIX_D_TILE], target_type=pl.FP32)
            value = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
            for in_h in pl.unroll(HC_MULT):
                residual_tile = residual_flat[
                    t : t + 1, in_h * D + d0 : in_h * D + d0 + MIX_D_TILE
                ]
                value = pl.add(
                    value,
                    pl.mul(
                        pl.cast(residual_tile, target_type=pl.FP32),
                        pl.read(residual_mix_flat, [t, in_h * HC_MULT + out_h]),
                    ),
                )
            output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + MIX_D_TILE] = pl.cast(
                value, target_type=pl.BF16, mode="rint"
            )
    return output


@pl.jit.inline
def mhc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Collapse the final streams by an unweighted mean, one block per token.

    This is where GLM is cheapest against DeepSeek-V4: ``Glm5NextTextHyperHead`` is
    ``hidden_streams.mean(dim=2)``, so there is no delayed pre-mix to compute and the
    donor's coefficient machinery — its own rms, projection and sigmoid — has no
    counterpart here. The kernel is :func:`mhc_pre` with the mixes replaced by the
    constant ``1 / HC_MULT``, folded in after the FP32 sum so the division rounds once.

    Returns the collapsed rows so the caller can chain them.
    """
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for t in pl.spmd(t_dim, name_hint="mhc_head"):
        for d0 in pl.pipeline(0, D, MIX_D_TILE, stage=2):
            value = pl.cast(x_flat[t : t + 1, d0 : d0 + MIX_D_TILE], target_type=pl.FP32)
            for stream in pl.unroll(HC_MULT - 1):
                head = stream + 1
                tile = x_flat[t : t + 1, head * D + d0 : head * D + d0 + MIX_D_TILE]
                value = pl.add(value, pl.cast(tile, target_type=pl.FP32))
            output[t : t + 1, d0 : d0 + MIX_D_TILE] = pl.cast(
                pl.mul(value, 1.0 / HC_MULT), target_type=pl.BF16, mode="rint"
            )
    return output


@pl.jit
def mhc_pre_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run one stream collapse for golden.run validation."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_pre(x_hc, pre_mix, output)
    return output


@pl.jit
def mhc_post_test(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
):
    """Run one residual expansion for golden.run validation."""
    sublayer.bind_dynamic(0, T_DYN)
    residual.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_post(sublayer, residual, post_mix, residual_mix, output)
    return output


@pl.jit
def mhc_head_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run one final collapse for golden.run validation."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_head(x_hc, output)
    return output


def _mix_generator(seed: int):
    return torch.Generator().manual_seed(seed)


def build_mhc_pre_tensor_specs(tokens: int = 20):
    """Build one deterministic stream collapse at the real hidden size."""
    from golden import TensorSpec

    generator = _mix_generator(97)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16()

    def init_pre_mix():
        # The reference mixes are sigmoid outputs, so they are positive and O(1).
        return torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator)).float()

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


def build_mhc_post_tensor_specs(tokens: int = 20):
    """Build one deterministic residual expansion at the real hidden size.

    ``residual_mix`` is drawn doubly stochastic, which is what the 20 Sinkhorn
    iterations in :func:`golden_mhc_mixes` produce, so the fixture exercises the
    magnitudes the kernel actually sees.
    """
    from golden import TensorSpec

    generator = _mix_generator(101)

    def init_sublayer():
        return torch.randn(tokens, D, generator=generator).bfloat16()

    def init_residual():
        return torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16()

    def init_post_mix():
        return (2 * torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))).float()

    def init_residual_mix():
        mix = torch.softmax(torch.randn(tokens, HC_MULT, HC_MULT, generator=generator), dim=-1)
        for _ in range(8):
            mix = mix / mix.sum(dim=-1, keepdim=True)
            mix = mix / mix.sum(dim=-2, keepdim=True)
        return mix.float()

    return [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16, init_value=init_sublayer),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_residual),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32, init_value=init_post_mix),
        TensorSpec(
            "residual_mix",
            [tokens, HC_MULT, HC_MULT],
            torch.float32,
            init_value=init_residual_mix,
        ),
        TensorSpec("output", [tokens, HC_MULT, D], torch.bfloat16),
    ]


def build_mhc_head_tensor_specs(tokens: int = 20):
    """Build one deterministic final collapse at the real hidden size."""
    from golden import TensorSpec

    generator = _mix_generator(103)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16()

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


def golden_mhc_pre_case(tensors):
    """Fill the expected collapse for :func:`build_mhc_pre_tensor_specs`."""
    tensors["output"][:] = golden_mhc_pre(tensors["x_hc"], tensors["pre_mix"])


def golden_mhc_post_case(tensors):
    """Fill the expected streams for :func:`build_mhc_post_tensor_specs`."""
    tensors["output"][:] = golden_mhc_post(
        tensors["sublayer"],
        tensors["residual"],
        tensors["post_mix"],
        tensors["residual_mix"],
    )


def golden_mhc_head_case(tensors):
    """Fill the expected collapse for :func:`build_mhc_head_tensor_specs`."""
    tensors["output"][:] = golden_mhc_head(tensors["x_hc"])


def main():
    """Prove the goldens on CPU, then validate the three stream scopes on device.

    Each output is one BF16 rounding of an FP32 sum over ``HC_MULT`` = 4 terms, so the
    budget is the rounding plus the accumulation order. ``mhc_mixes`` has no kernel
    body yet and so has no device case here.
    """
    import argparse

    from golden import ratio_allclose, run
    from models.glm5_3_flash._golden_smoke import run_mhc_goldens

    run_mhc_goldens(golden_mhc_mixes, golden_mhc_pre, golden_mhc_post, golden_mhc_head)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    compare = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    cases = (
        ("pre", mhc_pre_test, build_mhc_pre_tensor_specs(args.tokens), golden_mhc_pre_case),
        ("post", mhc_post_test, build_mhc_post_tensor_specs(args.tokens), golden_mhc_post_case),
        ("head", mhc_head_test, build_mhc_head_tensor_specs(args.tokens), golden_mhc_head_case),
    )
    for name, fn, specs, golden_fn in cases:
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            config={"platform": args.platform, "device_id": args.device},
            rtol=1.0 / 128,
            atol=1e-4,
            compare_fn={"output": compare},
            compile_only=args.compile_only,
        )
        print(f"{name}: {result}")
        if not result.passed:
            raise SystemExit(result.error or 1)


__all__ = [
    "build_mhc_head_tensor_specs",
    "build_mhc_post_tensor_specs",
    "build_mhc_pre_tensor_specs",
    "golden_mhc_head",
    "golden_mhc_head_case",
    "golden_mhc_post_case",
    "golden_mhc_pre_case",
    "mhc_head_test",
    "mhc_post_test",
    "mhc_pre_test",
    "golden_mhc_mixes",
    "golden_mhc_post",
    "golden_mhc_pre",
    "mhc_head",
    "mhc_mixes",
    "mhc_post",
    "mhc_pre",
]


if __name__ == "__main__":
    main()
