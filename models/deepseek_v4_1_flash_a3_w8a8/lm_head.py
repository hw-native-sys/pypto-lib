# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Final RMSNorm and untied BF16 head projection."""

import pypto.language as pl
import torch

from config import FLASH as M, TP_SIZE, T_DYN
from rmsnorm import rms_norm


# Dynamic shape variable.

# model config
D = M.hidden_size
VOCAB = M.vocab_size
LOCAL_VOCAB = VOCAB // TP_SIZE

# tiling
T_TILE = 16
K_TILE = 64
# A [T_TILE, vocab] FP32 accumulator cannot hold the real TP4 vocabulary
# (16 x 32320 x 4 B ~ 2.0 MB > the 512 KB on-chip budget), so the vocabulary
# axis is tiled: [16, 1024] FP32 (64 KB) accumulator and [1024, 64] BF16
# (128 KB) weight tile per step.
V_TILE = 1024


@pl.jit.inline(auto_scope=False)
def lm_head(
    hidden: pl.Tensor,
    norm_w: pl.Tensor,
    head_weight: pl.Tensor,
    logits: pl.Tensor,
    vocab: pl.constexpr,
):
    """Normalize hidden rows and project the vocabulary (two-level V/K tiling)."""
    t_dim = pl.tensor.dim(hidden, 0)
    normed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_norm(hidden, norm_w, normed, width=D, output_fp32=False, column_tile=256)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    vocab_tiles = (vocab + V_TILE - 1) // V_TILE
    # Token tiles are the parallel loop; vocabulary tiles are the SPMD block
    # index (static count).  A dynamic SPMD count carries a T_DYN ForStmt
    # argument that the combined-program front end rejects.
    for t_block in pl.parallel(token_tiles):
        t0 = t_block * T_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        for v_block in pl.spmd(vocab_tiles, name_hint="lm_head"):
            v0 = v_block * V_TILE
            vcols = pl.min(V_TILE, vocab - v0)
            acc = pl.create_tensor([T_TILE, V_TILE], dtype=pl.FP32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                x_tile = pl.slice(normed, [T_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
                x_tile = pl.set_validshape(pl.fillpad(x_tile, pad_value=pl.PadValue.zero), T_TILE, K_TILE)
                w_tile = pl.slice(
                    head_weight, [V_TILE, K_TILE], [v0, k0], valid_shape=[vcols, K_TILE]
                )
                acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True, init_cond=(k0 == 0))
            valid_logits = pl.set_validshape(acc, rows, vcols)
            logits[t0 : t0 + T_TILE, v0 : v0 + V_TILE] = valid_logits
    return logits


@pl.jit
def lm_head_test(
    hidden: pl.Tensor[[T_DYN, 5120], pl.BF16],
    norm_w: pl.Tensor[[5120], pl.BF16],
    head_weight: pl.Tensor[[1024, 5120], pl.BF16],
    logits: pl.Out[pl.Tensor[[T_DYN, 1024], pl.FP32]],
):
    hidden.bind_dynamic(0, T_DYN)
    logits.bind_dynamic(0, T_DYN)
    lm_head(hidden, norm_w, head_weight, logits, vocab=1024)
    return logits


@pl.jit
def lm_head_full_test(
    hidden: pl.Tensor[[T_DYN, 5120], pl.BF16],
    norm_w: pl.Tensor[[5120], pl.BF16],
    head_weight: pl.Tensor[[LOCAL_VOCAB, 5120], pl.BF16],
    logits: pl.Out[pl.Tensor[[T_DYN, LOCAL_VOCAB], pl.FP32]],
):
    """Real TP4 vocabulary width: exercises the vocabulary tiling."""
    hidden.bind_dynamic(0, T_DYN)
    logits.bind_dynamic(0, T_DYN)
    lm_head(hidden, norm_w, head_weight, logits, vocab=LOCAL_VOCAB)
    return logits


def golden_lm_head(tensors):
    """Torch reference: final norm, head matmul, and first-occurrence argmax."""
    value = tensors["hidden"].float()
    inv_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + M.rms_norm_eps)
    normed = (value * inv_rms * tensors["norm_w"].float()).to(torch.bfloat16)
    logits = torch.matmul(normed.float(), tensors["head_weight"].float().t())
    tensors["logits"][:] = logits


def build_tensor_specs(tokens: int = 16, vocab: int = 1024):
    from golden import TensorSpec

    return [
        TensorSpec("hidden", [tokens, 5120], torch.bfloat16, init_value=torch.randn),
        TensorSpec("norm_w", [5120], torch.bfloat16, init_value=lambda: torch.randn(5120) * 0.1 + 1.0),
        TensorSpec("head_weight", [vocab, 5120], torch.bfloat16, init_value=torch.randn),
        TensorSpec("logits", [tokens, vocab], torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    failures = []
    for name, fn, vocab in (
        ("bounded_1024", lm_head_test, 1024),
        ("real_tp4_vocab", lm_head_full_test, LOCAL_VOCAB),
    ):
        print(f"[LM_HEAD] {name} (vocab={vocab})", flush=True)
        torch.manual_seed(args.seed)
        result = run(
            fn=fn,
            specs=build_tensor_specs(args.tokens, vocab),
            golden_fn=golden_lm_head,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=2e-2,
            atol=2e-2,
            compare_fn={"logits": ratio_allclose(atol=2e-2, rtol=2e-2, max_error_ratio=5e-3)},
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
