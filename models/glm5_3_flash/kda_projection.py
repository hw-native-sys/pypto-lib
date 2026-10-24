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
sigmoid for its chunked path. Both paths here consume the same FP32 beta values.
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

from models.glm5_3_flash.config import KDA_DIM, KDA_GATE_LOWER_BOUND, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM, T_DYN, D
from models.glm5_3_flash.linear import bf16_linear

# tiling
T_TILE = 16  # tokens per task
# Per-head gates use cube-aligned storage at every supported TP degree.
BETA_PAD = ((LOCAL_KDA_H + 15) // 16) * 16
N_TILE = min(256, LOCAL_KDA_QKV_DIM)


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

    def linear_bf16(inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # FP32 dot products followed by the BF16 projection boundary.
        return (inp.float() @ weight.float().t()).to(torch.bfloat16)

    q = linear_bf16(x, w_q)
    k = linear_bf16(x, w_k)
    v = linear_bf16(x, w_v)
    mixed_qkv = torch.cat([q, k, v], dim=-1)

    f_a = linear_bf16(x, w_f_a)
    g_raw = linear_bf16(f_a, w_f_b).float().view(tokens, LOCAL_KDA_H, KDA_DIM)
    decay_rate = torch.exp(a_log.float()).view(1, LOCAL_KDA_H, 1)
    decay_logits = decay_rate * (g_raw + dt_bias.float().view(1, LOCAL_KDA_H, KDA_DIM))
    decay = KDA_GATE_LOWER_BOUND * torch.reciprocal(1.0 + torch.exp(-decay_logits))

    # beta carries the cube's 16-column padding. The padded rows of b_proj are zero,
    # so their gate is sigmoid(0) = 0.5 rather than 0; the reference reproduces that
    # instead of leaving a mismatch the caller would have to know to ignore.
    beta = torch.full((tokens, BETA_PAD), 0.5, dtype=torch.float32)
    beta[:, :LOCAL_KDA_H] = torch.reciprocal(1.0 + torch.exp(-linear_bf16(x, w_b).float()))

    g_a = linear_bf16(x, w_g_a)
    out_gate = linear_bf16(g_a, w_g_b).view(tokens, LOCAL_KDA_H, KDA_DIM)

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
    tiles = (t_dim + T_TILE - 1) // T_TILE
    t_pad = tiles * T_TILE
    f_a_gm = pl.create_tensor([t_pad, KDA_DIM], dtype=pl.BF16)
    g_a_gm = pl.create_tensor([t_pad, KDA_DIM], dtype=pl.BF16)
    b_raw_gm = pl.create_tensor([t_pad, BETA_PAD], dtype=pl.BF16)
    g_raw_gm = pl.create_tensor([t_pad, LOCAL_KDA_QKV_DIM], dtype=pl.BF16)
    decay_flat = pl.reshape(decay, [t_dim, LOCAL_KDA_QKV_DIM])
    out_gate_flat = pl.reshape(out_gate, [t_dim, LOCAL_KDA_QKV_DIM])
    dt_bias_row = pl.reshape(dt_bias, [1, LOCAL_KDA_QKV_DIM])
    w_b_pad = pl.create_tensor([BETA_PAD, D], dtype=pl.BF16)
    # composition-compat store form, numerics unchanged: one task builds every pad
    # row inside a pl.unroll body, so each w_b load carries a constant row offset.
    # A plain-spmd load of a raw weight at a traced offset blocks the downstream
    # dim folding that composed entries rely on (logs_t16/relay_m.py: s04 form
    # poisons, s05/s12/s13 unrolled form is clean); kda_conv's unrolled weight
    # taps are the in-tree precedent. Rows at and above LOCAL_KDA_H take the same
    # zeros the b_zero pass used to write.
    with pl.spmd(1, name_hint="kda_projection_b_pad") as pad_tid:
        zero = pl.tile.get_block_idx() * 0  # single task, block index zero
        for r in pl.unroll(BETA_PAD):
            if r < LOCAL_KDA_H + zero:
                w_b_pad[r : r + 1, 0:D] = pl.slice(w_b, [1, D], [r, 0])
            else:
                w_b_pad[r : r + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)

    with pl.spmd(tiles * (LOCAL_KDA_QKV_DIM // N_TILE), name_hint='kda_projection_qkv'):
        task = pl.tile.get_block_idx()
        t0 = task // (LOCAL_KDA_QKV_DIM // N_TILE) * T_TILE
        n0 = task % (LOCAL_KDA_QKV_DIM // N_TILE) * N_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        acc_q = bf16_linear(x, w_q, t0, n0, rows, pl.cast(N_TILE, pl.INDEX), T_TILE, N_TILE, D)
        part_v1 = pl.cast(acc_q, pl.BF16, mode='rint')
        mixed_qkv[t0:t0 + T_TILE, n0:n0 + N_TILE] = pl.set_validshape(part_v1, rows, N_TILE)
        acc_k = bf16_linear(x, w_k, t0, n0, rows, pl.cast(N_TILE, pl.INDEX), T_TILE, N_TILE, D)
        part_v2 = pl.cast(acc_k, pl.BF16, mode='rint')
        mixed_qkv[t0:t0 + T_TILE, LOCAL_KDA_QKV_DIM + n0:LOCAL_KDA_QKV_DIM + n0 + N_TILE] = pl.set_validshape(part_v2, rows, N_TILE)
        acc_v = bf16_linear(x, w_v, t0, n0, rows, pl.cast(N_TILE, pl.INDEX), T_TILE, N_TILE, D)
        part_v3 = pl.cast(acc_v, pl.BF16, mode='rint')
        mixed_qkv[t0:t0 + T_TILE, 2 * LOCAL_KDA_QKV_DIM + n0:2 * LOCAL_KDA_QKV_DIM + n0 + N_TILE] = pl.set_validshape(part_v3, rows, N_TILE)
    with pl.spmd(tiles, name_hint='kda_projection_bottleneck', deps=[pad_tid]) as bottleneck_tid:
        t0 = pl.tile.get_block_idx() * T_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        f_a_v4 = pl.cast(0, pl.INDEX)
        f_a_v5 = pl.cast(KDA_DIM, pl.INDEX)
        f_a = bf16_linear(x, w_f_a, t0, f_a_v4, rows, f_a_v5, T_TILE, KDA_DIM, D)
        f_a_gm[t0:t0 + T_TILE, :] = pl.set_validshape(pl.cast(f_a, pl.BF16, mode='rint'), rows, KDA_DIM)
        g_a_v6 = pl.cast(0, pl.INDEX)
        g_a_v7 = pl.cast(KDA_DIM, pl.INDEX)
        g_a = bf16_linear(x, w_g_a, t0, g_a_v6, rows, g_a_v7, T_TILE, KDA_DIM, D)
        g_a_gm[t0:t0 + T_TILE, :] = pl.set_validshape(pl.cast(g_a, pl.BF16, mode='rint'), rows, KDA_DIM)
        b_raw_v8 = pl.cast(0, pl.INDEX)
        b_raw_v9 = pl.cast(BETA_PAD, pl.INDEX)
        b_raw = bf16_linear(x, w_b_pad, t0, b_raw_v8, rows, b_raw_v9, T_TILE, BETA_PAD, D)
        part_v10 = pl.cast(b_raw, pl.BF16, mode='rint')
        b_raw_gm[t0:t0 + T_TILE, :] = pl.set_validshape(part_v10, rows, BETA_PAD)
    with pl.spmd(tiles * (LOCAL_KDA_QKV_DIM // N_TILE), name_hint='kda_projection_output', deps=[bottleneck_tid]) as output_tid:
        task = pl.tile.get_block_idx()
        t0 = task // (LOCAL_KDA_QKV_DIM // N_TILE) * T_TILE
        n0 = task % (LOCAL_KDA_QKV_DIM // N_TILE) * N_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        g_raw_v11 = pl.cast(N_TILE, pl.INDEX)
        g_raw = bf16_linear(f_a_gm, w_f_b, t0, n0, rows, g_raw_v11, T_TILE, N_TILE, KDA_DIM)
        part_v12 = pl.cast(g_raw, pl.BF16, mode='rint')
        g_raw_gm[t0:t0 + T_TILE, n0:n0 + N_TILE] = pl.set_validshape(part_v12, rows, N_TILE)
        out_raw_v13 = pl.cast(N_TILE, pl.INDEX)
        out_raw = bf16_linear(g_a_gm, w_g_b, t0, n0, rows, out_raw_v13, T_TILE, N_TILE, KDA_DIM)
        part_v14 = pl.cast(out_raw, pl.BF16, mode='rint')
        out_gate_flat[t0:t0 + T_TILE, n0:n0 + N_TILE] = pl.set_validshape(part_v14, rows, N_TILE)
    with pl.spmd((t_dim + 3) // 4 * (LOCAL_KDA_QKV_DIM // KDA_DIM), name_hint='kda_projection_decay', deps=[output_tid]):
        task = pl.tile.get_block_idx()
        t0 = task // (LOCAL_KDA_QKV_DIM // KDA_DIM) * 4
        n0 = task % (LOCAL_KDA_QKV_DIM // KDA_DIM) * KDA_DIM
        rows = pl.min(4, t_dim - t0)
        g_part_v15 = pl.slice(g_raw_gm, [4, KDA_DIM], [t0, n0], valid_shape=[rows, KDA_DIM])
        g_part = pl.cast(g_part_v15, pl.FP32)
        biased = pl.col_expand_add(g_part, dt_bias_row[0:1, n0:n0 + KDA_DIM])
        a_fill = pl.create_tensor([4, KDA_DIM], dtype=pl.FP32)
        a_fill[:, :] = pl.add(pl.mul(biased, 0.0), pl.read(a_log, [n0 // KDA_DIM]))
        rate = pl.exp(a_fill)
        scaled = pl.create_tensor([4, KDA_DIM], dtype=pl.FP32)
        scaled[:, :] = pl.mul(biased, rate)
        gate_h = pl.recip(pl.add(pl.exp(pl.neg(scaled)), 1.0))
        part_v16 = pl.mul(gate_h, KDA_GATE_LOWER_BOUND)
        decay_flat[t0:t0 + 4, n0:n0 + KDA_DIM] = pl.set_validshape(part_v16, rows, KDA_DIM)
    with pl.spmd((t_dim + 7) // 8, name_hint='kda_projection_beta', deps=[bottleneck_tid]):
        t0 = pl.tile.get_block_idx() * 8
        rows = pl.min(8, t_dim - t0)
        logits = pl.create_tensor([8, BETA_PAD], dtype=pl.FP32)
        part_v17 = pl.slice(b_raw_gm, [8, BETA_PAD], [t0, 0], valid_shape=[rows, BETA_PAD])
        logits[:, :] = pl.cast(part_v17, pl.FP32)
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0))
        beta[t0:t0 + 8, :] = pl.set_validshape(sigmoid, rows, BETA_PAD)
    return (mixed_qkv, decay, beta, out_gate)


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
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
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
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
