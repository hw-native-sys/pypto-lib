# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The attention output gate and the int8 hand-off to `o_proj`:

    a = attended * sigmoid(gate) ;  act_q[t] = rint(a * 127 / amax_t) ,  act_scale[t] = amax_t / 127

`gate` is the second half of each head's `2 * D` columns of the `q_proj` output.
One token at a time, its row viewed as `[HQ, D]`, one head per row.
"""
import pypto.language as pl

from config import GDN_QUANT, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
HQ = QWEN3_8_27B.num_attention_heads    # query heads
D = QWEN3_8_27B.head_dim                # head dimension

# quantization
SCALE_MAX = GDN_QUANT.scale_max
AMAX_EPS = GDN_QUANT.amax_eps

# gate
MIN_EXACT_VS_GOLDEN = 0.95      # of act_q elements identical to the float64 reference's quantisation
# validation
DEFAULT_CASES = (16, 32)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
TOK_TILE = 16           # tokens per block; 16 scalar scale writes fill one 64-byte line
LANES = 16              # amax lanes the decode form publishes; 16 FP32 is one 64-byte line

# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True


ATT_WIDTH = HQ * D
QG_WIDTH = 2 * HQ * D


def _qwen3_attn_gate_quant(
    attended: pl.Tensor[[T_DYN, ATT_WIDTH], pl.BF16],
    qg: pl.Tensor[[T_DYN, QG_WIDTH], pl.BF16],
    act_q: pl.Out[pl.Tensor[[T_DYN, ATT_WIDTH], pl.INT8]],
    act_scale: pl.Out[pl.Tensor[[1, T_DYN], pl.FP32]],
):
    """The token count must be a multiple of TOK_TILE; a block walks one whole tile."""
    attended.bind_dynamic(0, T_DYN)
    qg.bind_dynamic(0, T_DYN)
    act_q.bind_dynamic(0, T_DYN)
    act_scale.bind_dynamic(1, T_DYN)
    t_dim = pl.tensor.dim(attended, 0)
    head_rows = t_dim * HQ
    att_heads = pl.reshape(attended, [head_rows, D])
    qg_heads = pl.reshape(qg, [head_rows, 2 * D])
    q_heads = pl.reshape(act_q, [head_rows, D])
    for blk in pl.spmd(t_dim // TOK_TILE, name_hint="attn_gate_quant"):
        t0 = blk * TOK_TILE
        ones_col = pl.full([HQ, 8], dtype=pl.FP32, value=1.0)
        for i in pl.pipeline(TOK_TILE, stage=1):
            r0 = t0 + i
            h0 = r0 * HQ
            att = pl.cast(att_heads[h0 : h0 + HQ, :], target_type=pl.FP32)
            gate = pl.cast(qg_heads[h0 : h0 + HQ, D : 2 * D], target_type=pl.FP32)
            # sigmoid as a divide: the reciprocal would be used once
            act = pl.div(att, pl.add(pl.exp(pl.neg(gate)), 1.0))

            # fold the token's maximum to [1, 8], clamp it there, then
            # broadcast down the row axis so row_max lands it in every row
            col_amax = pl.reshape(pl.col_max(pl.abs(act)), [D // 8, 8])
            part_8 = pl.maximum(pl.col_max(col_amax), pl.full([1, 8], dtype=pl.FP32, value=AMAX_EPS))
            amax = pl.row_max(pl.col_expand_mul(ones_col, part_8))
            pl.write(act_scale, [0, r0], pl.read(amax, [0, 0]) / SCALE_MAX)

            # no clamp: amax is the row's own maximum, so |a| * SCALE_MAX / amax <= 127
            q = pl.row_expand_mul(act, pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX))
            # INT32 then INT8, not FP32 straight to INT8: the direct cast rounds through
            # FP16, whose spacing at 127 is a sixteenth of an INT8 step, and that moves
            # 0.9% of the output by one step. `quant_x` carries the measurement.
            q32 = pl.cast(q, target_type=pl.INT32, mode="rint")
            q_heads[h0 : h0 + HQ, :] = pl.cast(q32, target_type=pl.INT8, mode="trunc")
    return act_q, act_scale


qwen3_attn_gate_quant = pl.jit.inline(_qwen3_attn_gate_quant)
qwen3_attn_gate_quant_test = pl.jit(_qwen3_attn_gate_quant)


def build_fanned_kernel(m: int, inline: bool = False):
    """The decode form: one task per token, and the scale written by a task that owns its line.

    The body above gives one task a whole token tile and walks its 16 rows, which is right at
    prefill and is **one task** at a decode batch.

    The token axis is the one to split, and the only thing stopping a plain `pl.spmd(m)` is
    the scale: 16 FP32 scalars are exactly one 64-byte cache line, and a line written from two
    blocks loses one of them silently. So each task publishes its token's amax as a
    `[1, LANES]` tile into its own row of a `[m, LANES]` scratch -- 16 FP32 being the whole
    line -- and one `pl.spmd(1)` folds the scratch into `act_scale`.

    **Bit-identical to the body above**: the same operations per token in the same order, with
    the amax reassociated from 8 lanes to 16, and a maximum is exact under any association.
    """
    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_attn_gate_quant_fanned(
        attended: pl.Tensor[[m, ATT_WIDTH], pl.BF16],
        qg: pl.Tensor[[m, QG_WIDTH], pl.BF16],
        act_q: pl.Out[pl.Tensor[[m, ATT_WIDTH], pl.INT8]],
        act_scale: pl.Out[pl.Tensor[[1, m], pl.FP32]],
    ):
        """One task per token; `m` is the padded decode batch."""
        head_rows = m * HQ
        att_heads = pl.reshape(attended, [head_rows, D])
        qg_heads = pl.reshape(qg, [head_rows, 2 * D])
        q_heads = pl.reshape(act_q, [head_rows, D])
        part = pl.create_tensor([m, LANES], dtype=pl.FP32)

        for r0 in pl.spmd(m, name_hint="attn_gate_quant_token",
                          allow_early_resolve=EARLY_RESOLVE):
            h0 = r0 * HQ
            ones_h = pl.full([HQ, LANES], dtype=pl.FP32, value=1.0)
            att = pl.cast(att_heads[h0 : h0 + HQ, :], target_type=pl.FP32)
            gate = pl.cast(qg_heads[h0 : h0 + HQ, D : 2 * D], target_type=pl.FP32)
            act = pl.div(att, pl.add(pl.exp(pl.neg(gate)), 1.0))

            lane_max = pl.col_max(pl.reshape(pl.col_max(pl.abs(act)), [D // LANES, LANES]))
            lanes = pl.maximum(lane_max, pl.full([1, LANES], dtype=pl.FP32, value=AMAX_EPS))
            part[r0 : r0 + 1, 0:LANES] = lanes
            amax = pl.row_max(pl.col_expand_mul(ones_h, lanes))

            q = pl.row_expand_mul(act, pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX))
            q32 = pl.cast(q, target_type=pl.INT32, mode="rint")
            q_heads[h0 : h0 + HQ, :] = pl.cast(q32, target_type=pl.INT8, mode="trunc")

        for _solo in pl.spmd(1, name_hint="attn_gate_quant_scale",
                             allow_early_resolve=EARLY_RESOLVE):
            folded = pl.row_max(part[0:m, 0:LANES])
            act_scale[0:1, 0:m] = pl.div(pl.reshape(folded, [1, m]), SCALE_MAX)
        return act_q, act_scale

    return _qwen3_attn_gate_quant_fanned


def build_tensor_specs(t: int = BENCH_SEQ, hq: int = HQ, d: int = D):
    import torch
    from golden import TensorSpec

    def draw(seed, cols):
        def make():
            gen = torch.Generator().manual_seed(seed)
            return torch.randn(t, cols, generator=gen).to(torch.bfloat16)
        return make

    return [
        TensorSpec("attended", [t, hq * d], torch.bfloat16, init_value=draw(42, hq * d)),
        TensorSpec("qg", [t, 2 * hq * d], torch.bfloat16, init_value=draw(43, 2 * hq * d)),
        TensorSpec("act_q", [t, hq * d], torch.int8),
        TensorSpec("act_scale", [1, t], torch.float32),
    ]


def golden_qwen3_attn_gate_quant(tensors):
    import reference

    hq = tensors["qg"].shape[1] // (2 * D)
    act = reference.attn_gate(tensors["attended"], tensors["qg"], hq, D)
    act_q, scale = reference.quantize_rows(act)
    tensors["act_q"].copy_(act_q)
    tensors["act_scale"].copy_(scale.reshape(1, -1))


if __name__ == "__main__":
    import argparse

    from golden import run

    from utils import int8_step_and_exact

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--fanned", action="store_true", default=False,
                        help="the decode form: one task per token")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    if args.seq_len and args.seq_len % TOK_TILE:
        parser.error(f"--seq-len must be a multiple of TOK_TILE={TOK_TILE}")

    for t in cases:
        print(f"[case] attn_gate_quant T={t}", flush=True)
        result = run(
            fn=build_fanned_kernel(t) if args.fanned else qwen3_attn_gate_quant_test,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_qwen3_attn_gate_quant,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            compare_fn={"act_q": int8_step_and_exact(MIN_EXACT_VS_GOLDEN)},
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=1e-2,
            atol=1e-5,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={t}")
