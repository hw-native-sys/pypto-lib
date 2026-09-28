# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Boundary composition: embedding → HC head → final RMSNorm (backbone skipped).

Validates the Yang-Haoran ABI pieces that a future decode_fwd can consume.
Does not run decode_layer / prefill_layer (owned by other bring-up owners).
LM head + greedy sampling are exercised via ``lm_head.validate`` from ``main``.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

from models.deepseek_v4_1_flash.config import D, HC_MULT, T_DYN
from models.deepseek_v4_1_flash.golden import identity_pre_mix
from models.deepseek_v4_1_flash.hc_head import golden_hc_head, hc_head
from models.deepseek_v4_1_flash.input_pack import pack_x_hc
from models.deepseek_v4_1_flash.rmsnorm import golden_rms_norm, rms_norm


VOCAB_DYN = pl.dynamic("BOUNDARY_VOCAB_DYN")
TEST_VOCAB_SIZE = 256
DEFAULT_TOKENS = 8


@pl.jit.inline(auto_scope=False)
def boundary_embed_to_norm(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Pack embeddings into HC streams, collapse with delayed pre_mix, then final RMSNorm.

    The transformer backbone is intentionally skipped: ``pre_mix`` is a fixture
    (typically ``identity_pre_mix``) standing in for the last layer's delayed mix.
    """
    pack_x_hc(input_ids, embed_weight, x_hc)
    hc_head(x_hc, pre_mix, hidden)
    rms_norm(hidden, final_norm_w, x_normed)
    return x_normed


@pl.jit
def boundary_embed_to_norm_test(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_DYN, D], pl.BF16],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Standalone entry for the embed → HC head → final RMSNorm chain."""
    input_ids.bind_dynamic(0, T_DYN)
    embed_weight.bind_dynamic(0, VOCAB_DYN)
    x_hc.bind_dynamic(0, T_DYN)
    hidden.bind_dynamic(0, T_DYN)
    x_normed.bind_dynamic(0, T_DYN)
    return boundary_embed_to_norm(
        input_ids, embed_weight, pre_mix, final_norm_w, x_hc, hidden, x_normed,
    )


def golden_boundary_embed_to_norm(tensors):
    """Torch reference for the boundary chain (no backbone)."""
    vocab_size = tensors["embed_weight"].shape[0]
    safe_ids = tensors["input_ids"].long().clamp(0, vocab_size - 1)
    hidden_f = tensors["embed_weight"].index_select(0, safe_ids).float()
    tensors["x_hc"][:] = hidden_f.unsqueeze(1).expand(-1, HC_MULT, -1)
    tensors["hidden"][:] = golden_hc_head(tensors["x_hc"], tensors["pre_mix"])
    tensors["x_normed"][:] = golden_rms_norm(tensors["hidden"], tensors["final_norm_w"])


def build_boundary_tensor_specs(token_count: int = DEFAULT_TOKENS, vocab_size: int = TEST_VOCAB_SIZE):
    """Build fixtures for the embed → HC head → RMSNorm boundary chain."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(11)

    def init_input_ids():
        samples = torch.tensor(
            [0, 1, 17, vocab_size - 1, vocab_size, vocab_size + 3, 2, 5],
            dtype=torch.int64,
        )
        repeats = (token_count + samples.numel() - 1) // samples.numel()
        return samples.repeat(repeats)[:token_count].contiguous()

    def init_embed():
        return torch.randn(vocab_size, D, generator=generator, dtype=torch.bfloat16)

    def init_pre_mix():
        # Identity lane-0 mix stands in for the delayed last-layer pre_mix.
        x_shape = torch.empty(token_count, HC_MULT, D)
        return identity_pre_mix(x_shape)

    def init_norm_w():
        return torch.randn(D, generator=generator, dtype=torch.bfloat16)

    return [
        TensorSpec("input_ids", [token_count], torch.int64, init_value=init_input_ids),
        TensorSpec("embed_weight", [vocab_size, D], torch.bfloat16, init_value=init_embed),
        TensorSpec("pre_mix", [token_count, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("final_norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("x_hc", [token_count, HC_MULT, D], torch.float32),
        TensorSpec("hidden", [token_count, D], torch.bfloat16),
        TensorSpec("x_normed", [token_count, D], torch.bfloat16),
    ]


def validate_embed_to_norm(argv=None):
    """Validate pack → hc_head → rms_norm on A5 (backbone skipped)."""
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="V4.1 Flash boundary: embed → HC head → RMSNorm")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--vocab", type=int, default=TEST_VOCAB_SIZE)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        fn=boundary_embed_to_norm_test,
        specs=build_boundary_tensor_specs(args.tokens, args.vocab),
        golden_fn=golden_boundary_embed_to_norm,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_hc": ratio_allclose(atol=0.0, rtol=0.0),
            "hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "x_normed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
        compile_only=args.compile_only,
    )
    return result


def validate(argv=None):
    """Run the boundary chain, then LM head + greedy sampling (TP1)."""
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 Flash boundary operators bring-up")
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    known, remaining = parser.parse_known_args(argv)

    chain_argv = list(remaining)
    if known.compile_only and "--compile-only" not in chain_argv:
        chain_argv.append("--compile-only")
    chain = validate_embed_to_norm(chain_argv)
    if not chain.passed:
        return chain

    if known.skip_lm_head:
        return chain

    lm_argv = list(remaining)
    if "--tp" not in lm_argv:
        lm_argv.extend(["--tp", "1"])
    if "--dp" not in lm_argv:
        lm_argv.extend(["--dp", "1"])
    if known.compile_only and "--compile-only" not in lm_argv:
        lm_argv.append("--compile-only")

    # Import lm_head only after argv carries the intended --tp/--dp so its
    # static TP_SIZE / DP_SIZE match the harness.
    saved = sys.argv
    try:
        sys.argv = [saved[0], *lm_argv]
        result = _run_lm_head(lm_argv)
    finally:
        sys.argv = saved
    return result if not result.passed else chain


def _run_lm_head(argv):
    """Import and run the LM-head harness with the current process argv."""
    import argparse

    from golden import run
    from models.deepseek_v4_1_flash.lm_head import (
        DP_SIZE,
        TP_SIZE,
        WORLD_SIZE,
        build_tensor_specs,
        compare_logits,
        compare_sampled_ids,
        golden_lm_head,
        l3_lm_head,
    )
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--dp", type=int, default=DP_SIZE)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(WORLD_SIZE)))
    parser.add_argument("--compile-only", action="store_true", default=False)
    args, _unknown = parser.parse_known_args(argv)
    if args.tp != TP_SIZE or args.dp != DP_SIZE:
        raise SystemExit(
            f"lm_head was imported with TP={TP_SIZE} DP={DP_SIZE}; "
            f"got --tp {args.tp} --dp {args.dp}"
        )
    device_ids = [int(d) for d in args.device.split(",")]
    return run(
        fn=l3_lm_head,
        specs=build_tensor_specs(args.num_tokens),
        golden_fn=golden_lm_head,
        compare_fn={"logits": compare_logits, "sampled_ids": compare_sampled_ids},
        compile_only=args.compile_only,
        config=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:WORLD_SIZE], num_sub_workers=0
            ),
            platform=args.platform,
        ),
        rtol=1e-3,
        atol=1e-3,
    )


__all__ = [
    "boundary_embed_to_norm",
    "boundary_embed_to_norm_test",
    "build_boundary_tensor_specs",
    "golden_boundary_embed_to_norm",
    "validate",
    "validate_embed_to_norm",
]


_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run boundary chain then LM head; exit nonzero on failure."""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skip-lm-head", action="store_true")
    known, _ = parser.parse_known_args()
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)
    if known.skip_lm_head:
        print("[BOUNDARY] embed→hc_head→rms_norm PASS")
    else:
        print("[BOUNDARY] embed→hc_head→rms_norm and lm_head+greedy PASS")


def test_precision(a5_args):
    """Validate the embed → HC head → RMSNorm chain against its golden on A5."""
    result = validate_embed_to_norm(a5_args())
    assert result.passed, result.error


if __name__ == _SCRIPT_ENTRY_POINT:
    main()
