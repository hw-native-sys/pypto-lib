# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KDA front end: the q/k/v projections and every gate the delta rule consumes.

For each of the 34 linear-attention layers (64 heads x 128, so 4 heads per rank at
TP16):

* ``q_proj`` / ``k_proj`` / ``v_proj``: ``[D] -> [KDA_H * KDA_DIM]``, head-sharded.
* forget gate: ``g = gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``
  with ``gate_lower_bound = -5.0``. ``f_a_proj`` is a rank-128 bottleneck and stays
  replicated; ``f_b_proj``, ``dt_bias`` and ``A_log`` are head-sharded.
* input gate: ``beta = sigmoid(b_proj(x))``, one scalar per head.
* output gate: ``g_b(g_a(x))``, the same 128-wide bottleneck shape as the forget gate.

The checkpoint keeps all of these BF16 — none of the KDA projections carry a
``weight_scale_inv``, and the vLLM Ascend port nulls the quant config for the whole
KDA block for the same reason.

There is no ``sigmoid`` tile op in the pypto DSL, so every gate here is spelled out as
``pl.recip(pl.add(pl.exp(pl.neg(x)), 1.0))`` at the call site — the traced body admits
only ``pl.*`` calls, so a Python helper cannot factor it out.

Two details decide whether this matches the deployment numerics:

* ``A_log`` is **per head** and broadcasts across all 128 channels, while ``dt_bias``
  is **per channel**. Swapping them is the classic fused-gate bug and no shape check
  catches it, because both flatten to the same 512 values per rank.
* Both bottleneck projections are BF16 ``nn.Linear`` layers upstream, so their outputs
  are rounded to BF16 *before* the FP32 gate arithmetic begins. Keeping them in FP32
  here would be more accurate than the reference and would show up as a systematic
  golden mismatch, so the rounding is reproduced deliberately.

``beta`` leaves this kernel already sigmoided in FP32. vLLM-Ascend carries the raw
logit into its recurrent kernel and sigmoids it there, but pre-computes the FP32
sigmoid for its chunked path; doing it once here gives both paths the more accurate
FP32 form and costs nothing.
"""

import sys
from pathlib import Path

# Run directly and the script's own directory leads sys.path, where this model's
# ``golden.py`` shadows the repo-root ``golden`` harness package. Put the repo root
# first, exactly as models/deepseek_v4_1_flash does.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, KDA_DIM, KDA_GATE_LOWER_BOUND, LOCAL_KDA_H
from models.glm5_3_flash.config import LOCAL_KDA_QKV_DIM, T_DYN

# tiling
T_TILE = 16  # tokens per task
# L1 holds 512 KiB with the first 64 KiB reserved for the cross-core ring, and a
# weight tile is double-buffered by the pipeline. At K_TILE 128 one [512, 128] BF16
# tile is 128 KiB, so a single 512-wide lane fits with room to spare; three lanes in
# one pipeline, or K_TILE 512, do not. q/k/v therefore run as three separate
# pipelines, and the three narrow projections share a fourth.
K_TILE = 128
# beta is one value per head, and LOCAL_KDA_H is 4 at TP16. Both the cube's N axis
# and the store path want 16, so beta carries that padding in its own shape rather
# than being narrowed on the way out: narrowing a wider tile down to four FP32
# columns does not work here in any of its spellings -- a slice's valid_shape, a
# pl.set_validshape and pl.store's explicit extent all leave the store writing the
# tile's full row, which spills each token's gate into the next three. Columns at and
# above LOCAL_KDA_H are unused; at TP4, where LOCAL_KDA_H is already 16, there is no
# padding at all.
BETA_PAD = 16
# Padding rows to zero. Folded here rather than in the traced body, which takes only
# pl.* calls; it floors at one because a zero-task region is not dispatchable, and at
# TP4 LOCAL_KDA_H already reaches BETA_PAD so that one task writes nothing.
B_ZERO_TASKS = max(BETA_PAD - LOCAL_KDA_H, 1)


def golden_kda_projection(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_f_a: torch.Tensor,
    w_f_b: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    w_b: torch.Tensor,
    w_g_a: torch.Tensor,
    w_g_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the four KDA front-end outputs.

    Returns ``(mixed_qkv, decay, beta, out_gate)``. ``decay`` is the **log** decay:
    the delta rule exponentiates it, this function does not.
    """
    tokens = x.shape[0]
    xf = x.float()

    def linear_bf16(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # BF16 nn.Linear: FP32 accumulate, BF16 result.
        return (inp.float() @ weight.float().t()).to(torch.bfloat16)

    q = linear_bf16(x, w_q)
    k = linear_bf16(x, w_k)
    v = linear_bf16(x, w_v)
    mixed_qkv = torch.cat([q, k, v], dim=-1)

    f_a = linear_bf16(x, w_f_a)
    g_raw = linear_bf16(f_a, w_f_b).float().view(tokens, LOCAL_KDA_H, KDA_DIM)
    decay_rate = torch.exp(a_log.float()).view(1, LOCAL_KDA_H, 1)
    decay = KDA_GATE_LOWER_BOUND * torch.sigmoid(
        decay_rate * (g_raw + dt_bias.float().view(1, LOCAL_KDA_H, KDA_DIM))
    )

    # beta carries the cube's 16-column padding. The padded rows of b_proj are zero,
    # so their gate is sigmoid(0) = 0.5 rather than 0; the reference reproduces that
    # instead of leaving a mismatch the caller would have to know to ignore.
    beta = torch.full((tokens, BETA_PAD), 0.5, dtype=torch.float32)
    beta[:, :LOCAL_KDA_H] = torch.sigmoid(linear_bf16(x, w_b).float())

    g_a = linear_bf16(x, w_g_a)
    out_gate = linear_bf16(g_a, w_g_b).view(tokens, LOCAL_KDA_H, KDA_DIM)

    del xf
    return mixed_qkv, decay.float(), beta, out_gate


@pl.jit.inline
def kda_projection(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    mixed_qkv: pl.Tensor[[T_DYN, 3 * LOCAL_KDA_QKV_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(x, 0)
    decay_flat = pl.reshape(decay, [t_dim, LOCAL_KDA_QKV_DIM])
    out_gate_flat = pl.reshape(out_gate, [t_dim, LOCAL_KDA_QKV_DIM])
    dt_bias_row = pl.reshape(dt_bias, [1, LOCAL_KDA_QKV_DIM])

    # b_proj emits one value per head, four at TP16, and the cube's N axis wants 16.
    # Padding it with valid_shape is not enough: the accumulator then carries a valid
    # width of 4 and is read back at that pitch, so each token's gates land four
    # columns further along than they should. Materialising a zero-padded weight makes
    # all 16 accumulator columns real, and the store is width-matched.
    w_b_pad = pl.create_tensor([BETA_PAD, D], dtype=pl.BF16)
    with pl.spmd(BETA_PAD, name_hint="kda_projection_b_pad") as b_pad_tid:
        r = pl.tile.get_block_idx()
        w_b_pad[r : r + 1, 0:D] = pl.slice(w_b, [1, D], [pl.min(r, LOCAL_KDA_H - 1), 0])
    with pl.spmd(B_ZERO_TASKS, name_hint="kda_projection_b_zero",
                 deps=[b_pad_tid]) as b_zero_tid:
        r = LOCAL_KDA_H + pl.tile.get_block_idx()
        if r < BETA_PAD:
            w_b_pad[r : r + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)

    tiles = (t_dim + T_TILE - 1) // T_TILE

    with pl.spmd(tiles, name_hint="kda_projection", deps=[b_zero_tid]):
        task = pl.tile.get_block_idx()
        t0 = task * T_TILE
        # The last block spills past t_dim; valid_shape zero-fills what it reads and
        # set_validshape keeps the write inside the tensor.
        rows = pl.min(T_TILE, t_dim - t0)

        # --- q | k | v: one pipeline each, so only one 512-wide weight is in flight ---
        # Each lane seeds with pl.matmul rather than matmul_acc into a create_tensor.
        # With a narrowed valid row count and N above one 16-wide fractal, mad writes
        # at a pitch of ceil(rows/16)*16 while a non-compact accumulator is read back
        # at its physical height, which skews every fractal past the first. Seeding
        # with pl.matmul stamps the accumulator compact and keeps the two agreed.
        x_head = pl.slice(x, [T_TILE, K_TILE], [t0, 0], valid_shape=[rows, K_TILE])

        acc_q = pl.matmul(x_head, w_q[0:LOCAL_KDA_QKV_DIM, 0:K_TILE], pl.FP32, b_trans=True)
        for kb in pl.pipeline(1, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            acc_q = pl.matmul_acc(
                acc_q, pl.slice(x, [T_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE]),
                w_q[0:LOCAL_KDA_QKV_DIM, k0 : k0 + K_TILE], b_trans=True)
        mixed_qkv[t0 : t0 + T_TILE, 0:LOCAL_KDA_QKV_DIM] = pl.set_validshape(
            pl.cast(acc_q, pl.BF16, mode="rint"), rows, LOCAL_KDA_QKV_DIM)

        acc_k = pl.matmul(x_head, w_k[0:LOCAL_KDA_QKV_DIM, 0:K_TILE], pl.FP32, b_trans=True)
        for kb in pl.pipeline(1, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            acc_k = pl.matmul_acc(
                acc_k, pl.slice(x, [T_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE]),
                w_k[0:LOCAL_KDA_QKV_DIM, k0 : k0 + K_TILE], b_trans=True)
        mixed_qkv[t0 : t0 + T_TILE, LOCAL_KDA_QKV_DIM : 2 * LOCAL_KDA_QKV_DIM] = pl.set_validshape(
            pl.cast(acc_k, pl.BF16, mode="rint"), rows, LOCAL_KDA_QKV_DIM)

        acc_v = pl.matmul(x_head, w_v[0:LOCAL_KDA_QKV_DIM, 0:K_TILE], pl.FP32, b_trans=True)
        for kb in pl.pipeline(1, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            acc_v = pl.matmul_acc(
                acc_v, pl.slice(x, [T_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE]),
                w_v[0:LOCAL_KDA_QKV_DIM, k0 : k0 + K_TILE], b_trans=True)
        mixed_qkv[t0 : t0 + T_TILE, 2 * LOCAL_KDA_QKV_DIM : 3 * LOCAL_KDA_QKV_DIM] = pl.set_validshape(
            pl.cast(acc_v, pl.BF16, mode="rint"), rows, LOCAL_KDA_QKV_DIM)

        # --- the three narrow projections: both 128-wide bottlenecks and b_proj ---
        f_a = pl.matmul(x_head, w_f_a[0:KDA_DIM, 0:K_TILE], pl.FP32, b_trans=True)
        g_a = pl.matmul(x_head, w_g_a[0:KDA_DIM, 0:K_TILE], pl.FP32, b_trans=True)
        acc_b = pl.matmul(x_head, w_b_pad[0:BETA_PAD, 0:K_TILE], pl.FP32, b_trans=True)
        for kb in pl.pipeline(1, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_slice = pl.slice(x, [T_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
            f_a = pl.matmul_acc(f_a, x_slice, w_f_a[0:KDA_DIM, k0 : k0 + K_TILE], b_trans=True)
            g_a = pl.matmul_acc(g_a, x_slice, w_g_a[0:KDA_DIM, k0 : k0 + K_TILE], b_trans=True)
            acc_b = pl.matmul_acc(acc_b, x_slice, w_b_pad[0:BETA_PAD, k0 : k0 + K_TILE],
                                  b_trans=True)

        # Both bottlenecks are BF16 Linears upstream: round here, not after the second GEMM.
        f_a_bf = pl.cast(f_a, pl.BF16, mode="rint")
        g_a_bf = pl.cast(g_a, pl.BF16, mode="rint")

        # --- forget gate: -5.0 * sigmoid(exp(A_log) * (f_b(f_a) + dt_bias)) ---
        g_raw = pl.matmul(f_a_bf, w_f_b[0:LOCAL_KDA_QKV_DIM, 0:KDA_DIM], pl.FP32, b_trans=True)
        # dt_bias is per channel, so it broadcasts down the token axis.
        biased = pl.col_expand_add(pl.cast(pl.cast(g_raw, pl.BF16, mode="rint"), pl.FP32),
                                   dt_bias_row[0:1, 0:LOCAL_KDA_QKV_DIM])
        # exp(A_log) is per head, so it scales a contiguous 128-channel block.
        for h in pl.unroll(LOCAL_KDA_H):
            c0 = h * KDA_DIM
            # An explicit slice: `biased` carries `rows` valid rows, so a plain
            # subview asking for T_TILE contradicts the inferred valid_row.
            biased_h = pl.slice(biased, [T_TILE, KDA_DIM], [0, c0], valid_shape=[rows, KDA_DIM])
            # exp(A_log[h]) as a tile. A_log has one value per head, and four FP32
            # columns is a 16-byte tile row ptoas rejects -- widening the load past the
            # tensor's 4 elements is rejected too, and pl.exp takes no Scalar. Multiply
            # by zero and add the scalar to fill a correctly shaped tile instead.
            a_fill = pl.add(pl.mul(biased_h, 0.0), pl.read(a_log, [h]))
            scaled = pl.mul(biased_h, pl.exp(a_fill))
            # sigmoid, spelled out: the traced body takes only pl.* calls.
            gate_h = pl.recip(pl.add(pl.exp(pl.neg(scaled)), 1.0))
            decay_flat[t0 : t0 + T_TILE, c0 : c0 + KDA_DIM] = pl.set_validshape(
                pl.mul(gate_h, KDA_GATE_LOWER_BOUND), rows, KDA_DIM
            )

        # --- output gate: the raw g_b(g_a) logit; o_norm applies the sigmoid ---
        out_raw = pl.matmul(g_a_bf, w_g_b[0:LOCAL_KDA_QKV_DIM, 0:KDA_DIM], pl.FP32, b_trans=True)
        out_gate_flat[t0 : t0 + T_TILE, 0:LOCAL_KDA_QKV_DIM] = pl.set_validshape(
            pl.cast(out_raw, pl.BF16, mode="rint"), rows, LOCAL_KDA_QKV_DIM
        )

        # --- input gate: sigmoid(b_proj(x)), one value per head ---
        # The upstream b_proj is a BF16 Linear, so its output rounds before the gate.
        logits = pl.cast(pl.cast(acc_b, pl.BF16, mode="rint"), pl.FP32)
        beta[t0 : t0 + T_TILE, 0:BETA_PAD] = pl.set_validshape(
            pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), rows, BETA_PAD)

    # beta is four FP32 columns, and a tile row must be 32-byte aligned, so the gate
    # is computed eight columns wide and narrowed on the way out. The valid shape of
    # that narrowing has to be *static*: with a runtime row count the allocator
    # instantiates the tile at its valid shape, which is the illegal 16-byte row. Full
    # tiles and the ragged tail are therefore written by two passes, each with a
    # compile-time shape. models/deepseek_v4_flash_mtp/hc_pre.py writes its own 4-wide
    # gate the same way, and sidesteps the tail by requiring an aligned token count.
    return mixed_qkv, decay, beta, out_gate


__all__ = [
    "KDA_GATE_LOWER_BOUND",
    "golden_kda_projection",
    "kda_projection",
]


@pl.jit
def kda_projection_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    mixed_qkv: pl.Out[pl.Tensor[[T_DYN, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.Out[pl.Tensor[[T_DYN, BETA_PAD], pl.FP32]],
    out_gate: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    mixed_qkv, decay, beta, out_gate = kda_projection(
        x, w_q, w_k, w_v, w_f_a, w_f_b, dt_bias, a_log, w_b, w_g_a, w_g_b,
        mixed_qkv, decay, beta, out_gate,
    )
    return mixed_qkv, decay, beta, out_gate


def golden_kda_projection_tensors(tensors) -> None:
    """Harness adapter: run the reference and write the four outputs in place."""
    mixed_qkv, decay, beta, out_gate = golden_kda_projection(
        tensors["x"], tensors["w_q"], tensors["w_k"], tensors["w_v"],
        tensors["w_f_a"], tensors["w_f_b"], tensors["dt_bias"], tensors["a_log"],
        tensors["w_b"], tensors["w_g_a"], tensors["w_g_b"],
    )
    tensors["mixed_qkv"][:] = mixed_qkv
    tensors["decay"][:] = decay
    tensors["beta"][:] = beta
    tensors["out_gate"][:] = out_gate


def build_tensor_specs(tokens: int = 67):
    """A deliberately ragged token count, so the tail block is always exercised."""
    from golden import TensorSpec

    bf, f32 = torch.bfloat16, torch.float32

    def w(rows, cols, scale=0.02):
        return lambda: (torch.randn(rows, cols) * scale).to(bf)

    return [
        TensorSpec("x", [tokens, D], bf, init_value=lambda: (torch.randn(tokens, D) * 0.5).to(bf)),
        TensorSpec("w_q", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_k", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_v", [LOCAL_KDA_QKV_DIM, D], bf, init_value=w(LOCAL_KDA_QKV_DIM, D)),
        TensorSpec("w_f_a", [KDA_DIM, D], bf, init_value=w(KDA_DIM, D)),
        TensorSpec("w_f_b", [LOCAL_KDA_QKV_DIM, KDA_DIM], bf,
                   init_value=w(LOCAL_KDA_QKV_DIM, KDA_DIM, 0.08)),
        # dt_bias spans a wide range in the real checkpoint; a narrow fixture would
        # hide a per-head / per-channel mix-up behind the sigmoid's flat tails.
        TensorSpec("dt_bias", [LOCAL_KDA_QKV_DIM], f32,
                   init_value=lambda: torch.randn(LOCAL_KDA_QKV_DIM) * 2.0),
        # Distinct per-head values, so a head-axis broadcast error is visible. Spread
        # across however many heads the rank owns: four at TP16, sixteen at TP4.
        TensorSpec("a_log", [LOCAL_KDA_H], f32,
                   init_value=lambda: torch.linspace(-0.7, 1.1, LOCAL_KDA_H)),
        TensorSpec("w_b", [LOCAL_KDA_H, D], bf, init_value=w(LOCAL_KDA_H, D, 0.05)),
        TensorSpec("w_g_a", [KDA_DIM, D], bf, init_value=w(KDA_DIM, D)),
        TensorSpec("w_g_b", [LOCAL_KDA_QKV_DIM, KDA_DIM], bf,
                   init_value=w(LOCAL_KDA_QKV_DIM, KDA_DIM, 0.08)),
        TensorSpec("mixed_qkv", [tokens, 3 * LOCAL_KDA_QKV_DIM], bf),
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32),
        TensorSpec("beta", [tokens, BETA_PAD], f32),
        TensorSpec("out_gate", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged",
                        choices=["ragged", "aligned", "single"])
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    tokens = {"ragged": 67, "aligned": 64, "single": 1}[args.case]
    result = run(
        fn=kda_projection_test,
        specs=build_tensor_specs(tokens),
        golden_fn=golden_kda_projection_tensors,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
