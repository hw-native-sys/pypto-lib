# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The per-token INT8 quantization `down_proj` reads, over the MLP's activation:

    act_q[t] = rint(act[t] * 127 / amax_t) ,  act_scale[t] = amax_t / 127

One token at a time, the row viewed as `[ROW_TILE, COL_TILE]`: a `[1, 1]`
reduction result is a 4-byte tile ptoas will not allocate.
"""
import pypto.language as pl

from config import GDN_QUANT, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
INTER = QWEN3_8_27B.intermediate_size       # the axis the quantiser reduces over

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
COL_TILE = 128          # the activation row viewed as (INTER // COL_TILE) x COL_TILE


ROW_TILE = INTER // COL_TILE            # the activation row's other axis


def _qwen3_act_quant(
    act: pl.Tensor[[T_DYN, INTER], pl.BF16],
    act_q: pl.Out[pl.Tensor[[T_DYN, INTER], pl.INT8]],
    act_scale: pl.Out[pl.Tensor[[1, T_DYN], pl.FP32]],
):
    """Any token count. A block walks one whole tile; the last one repeats its last token.

    **The block count is a ceiling and the row index is clamped**, which is what makes a
    ragged token count correct. A floor division dropped the final partial tile silently --
    up to `TOK_TILE - 1` tokens left with `act_q` and `act_scale` untouched, no error, which
    is the hazard `decode_projection.check_tile` was added for on the other path. The
    clamp costs the last tile up to 15 redundant single-token passes, 0.18% at T = 8192, and
    every one of them recomputes a row it already wrote with the identical value: the body is
    a pure function of its own row, so repeating it is idempotent.

    A full tile is untouched by this -- `t0 + i < t_dim` throughout, so the clamp is a no-op
    and the `pl.pipeline` trip count stays compile-time, which is why the fast path keeps its
    overlap.
    """
    act.bind_dynamic(0, T_DYN)
    act_q.bind_dynamic(0, T_DYN)
    act_scale.bind_dynamic(1, T_DYN)
    t_dim = pl.tensor.dim(act, 0)
    for blk in pl.spmd((t_dim + TOK_TILE - 1) // TOK_TILE, name_hint="act_quant"):
        t0 = blk * TOK_TILE
        ones_col = pl.full([ROW_TILE, 8], dtype=pl.FP32, value=1.0)
        for i in pl.pipeline(TOK_TILE, stage=1):
            r0 = pl.min(t0 + i, t_dim - 1)
            act_run = pl.cast(act[r0 : r0 + 1, :], target_type=pl.FP32)
            act_tile = pl.reshape(act_run, [ROW_TILE, COL_TILE])

            # fold the token's maximum to [1, 8], clamp it there, then
            # broadcast down the row axis so row_max lands it in every row
            col_amax = pl.reshape(pl.col_max(pl.abs(act_tile)), [COL_TILE // 8, 8])
            part_8 = pl.maximum(pl.col_max(col_amax), pl.full([1, 8], dtype=pl.FP32, value=AMAX_EPS))
            amax = pl.row_max(pl.col_expand_mul(ones_col, part_8))
            pl.write(act_scale, [0, r0], pl.read(amax, [0, 0]) / SCALE_MAX)

            # no clamp: amax is the row's own maximum, so |a| * SCALE_MAX / amax <= 127
            q = pl.row_expand_mul(act_tile, pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX))
            # INT32 then INT8, not FP32 straight to INT8: the direct cast rounds through
            # FP16, whose spacing at 127 is a sixteenth of an INT8 step, and that moves
            # 0.9% of the output by one step. `quant_x` carries the measurement.
            q32 = pl.cast(q, target_type=pl.INT32, mode="rint")
            q_i8 = pl.cast(q32, target_type=pl.INT8, mode="trunc")
            act_q[r0 : r0 + 1, :] = pl.reshape(q_i8, [1, INTER])
    return act_q, act_scale


# Column groups the decode form splits the reduction over; see `build_fanned_kernel`.
COL_GROUPS = 17
VEC_BUFFER_BYTES = 188416       # a2a3's vector buffer, which bounds the column range
STORE_GROUP = 128               # elements per INT8 store group; the guard keeps every store whole
# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True



def build_fanned_kernel(m: int, col_groups: int = COL_GROUPS, tok_tile: int = TOK_TILE,
                        inline: bool = False):
    """The decode form: the per-token reduction split across the column axis.

    The body above gives one task a whole token tile and walks its 16 rows with `stage=1`, so
    nothing overlaps. At prefill that is right; at a decode batch there is **one** task, and
    its 16 rows each reduce and rescale a `[136, 128]` FP32 tile, 69632 bytes. Two of those
    live at once, which is why the token loop cannot pipeline: `stage=2` would need 278 KB of
    a 188416 B vector buffer.

    The token axis is gone at decode, so the fan-out moves to the columns, and that splits
    the per-token amax into a partial pass and a combine:

    1. `pl.spmd(col_groups)` -- each task reduces its own column range over every token and
       writes one row of a `[col_groups, m]` FP32 scratch.
    2. `pl.spmd(1)` -- one task folds the scratch to the per-token amax and writes
       `act_scale`. **It has to be one task**: 16 FP32 scales are exactly one 64-byte line,
       and a line written from two blocks loses one of them silently (the
       `ScalarWriteLineShared` diagnostic).
    3. `pl.spmd(col_groups)` -- each task re-folds the scratch, then rescales and stores its
       own column range.

    **This is bit-identical to the body above**, which is what makes an exactness gate the
    right one. Maximum is associative and exact, so splitting the reduction cannot move
    `amax`; the clamp commutes with it; and the rescale applies the same
    `recip(amax) * SCALE_MAX` to the same values.

    **`col_groups` must divide `INTER // 128 = 136`, so that every INT8 store covers whole
    128-element groups.** A partial group was observed to write wrong values into the leading
    half of the remainder, deterministically and with no compiler diagnostic, with the errors
    at or past the last whole group in its leading `(width % 128) / 2` columns; only the INT8
    cast and store were affected, an FP32 store of the same tile being exact at the same
    width.

    **That no longer reproduces on the pinned toolchain**: this kernel passes its own golden
    on device at `col_groups` 16 and 32, widths 1088 and 544. The guard is kept because the
    failure it covers is silent and two widths are not the twelve the original report
    measured. Lifting it needs its own sweep, and would open `col_groups` values the tile
    sweep has never scored.

    `col_groups` also sets the per-task tile: the apply pass holds `[tok_tile, width]` as
    FP32 twice plus the INT8 result, so 8 groups overflows the vector buffer and is refused
    at build rather than left to die in `MemoryReuse`.
    """
    if m % tok_tile:
        raise ValueError(f"m={m} must be a multiple of tok_tile={tok_tile}")
    if INTER % col_groups:
        raise ValueError(f"col_groups={col_groups} must divide INTER={INTER}")
    width = INTER // col_groups
    if width % STORE_GROUP:
        raise ValueError(
            f"col_groups={col_groups} leaves a {width}-column range, which is not a multiple "
            f"of {STORE_GROUP}; a partial group was observed to write wrong values into the "
            f"leading half of the remainder, and the guard is kept although that no longer "
            f"reproduces (see the docstring). col_groups must divide {INTER // STORE_GROUP}")
    # The apply pass holds the column range three times over: FP32 in, FP32 rescaled, INT8
    # out. Measured against the refusal: col_groups=8 (2176 columns) overflows and 16 (1088)
    # does not, which this estimate reproduces. Without the check the compiler accepts the
    # build and dies in MemoryReuse with "also overflows -- reduce tile size or stage count".
    live = tok_tile * width * (4 + 4 + 1)
    if live > VEC_BUFFER_BYTES:
        raise ValueError(f"col_groups={col_groups} needs about {live} B of the {VEC_BUFFER_BYTES} B "
                         f"vector buffer in the apply pass; use at least "
                         f"{INTER * tok_tile * 9 // VEC_BUFFER_BYTES + 1} groups")

    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_act_quant_fanned(
        act: pl.Tensor[[m, INTER], pl.BF16],
        act_q: pl.Out[pl.Tensor[[m, INTER], pl.INT8]],
        act_scale: pl.Out[pl.Tensor[[1, m], pl.FP32]],
    ):
        """One task per column group; `m` is the padded decode batch."""
        part = pl.create_tensor([col_groups, m], dtype=pl.FP32)

        for cg in pl.spmd(col_groups, name_hint="act_quant_amax",
                          allow_early_resolve=EARLY_RESOLVE):
            c0 = cg * width
            for t0 in pl.range(0, m, tok_tile):
                a = pl.cast(act[t0 : t0 + tok_tile, c0 : c0 + width], target_type=pl.FP32)
                row_amax = pl.row_max(pl.abs(a))
                part[cg : cg + 1, t0 : t0 + tok_tile] = pl.reshape(row_amax, [1, tok_tile])

        for _solo in pl.spmd(1, name_hint="act_quant_scale",
                             allow_early_resolve=EARLY_RESOLVE):
            for t0 in pl.range(0, m, tok_tile):
                eps_row = pl.full([1, tok_tile], dtype=pl.FP32, value=AMAX_EPS)
                amax = pl.maximum(pl.col_max(part[0:col_groups, t0 : t0 + tok_tile]), eps_row)
                act_scale[0:1, t0 : t0 + tok_tile] = pl.div(amax, SCALE_MAX)

        for cg in pl.spmd(col_groups, name_hint="act_quant_apply",
                          allow_early_resolve=EARLY_RESOLVE):
            c0 = cg * width
            for t0 in pl.range(0, m, tok_tile):
                eps_row = pl.full([1, tok_tile], dtype=pl.FP32, value=AMAX_EPS)
                amax = pl.maximum(pl.col_max(part[0:col_groups, t0 : t0 + tok_tile]), eps_row)
                amax_col = pl.reshape(amax, [tok_tile, 1])
                inv = pl.mul(pl.recip(amax_col, high_precision=True), SCALE_MAX)
                a = pl.cast(act[t0 : t0 + tok_tile, c0 : c0 + width], target_type=pl.FP32)
                q = pl.row_expand_mul(a, inv)
                q32 = pl.cast(q, target_type=pl.INT32, mode="rint")
                act_q[t0 : t0 + tok_tile, c0 : c0 + width] = pl.cast(q32, target_type=pl.INT8,
                                                                     mode="trunc")
        return act_q, act_scale

    return _qwen3_act_quant_fanned


qwen3_act_quant = pl.jit.inline(_qwen3_act_quant)
qwen3_act_quant_test = pl.jit(_qwen3_act_quant)


def build_tensor_specs(t: int = BENCH_SEQ, inter: int = INTER, weights: str | None = None):
    import torch
    from golden import TensorSpec

    import reference

    def init_act():
        names = ("x_q", "x_scale", "w_gate", "w_gate_scale", "w_up", "w_up_scale")
        a = [reference.mlp_inputs(n, t, QWEN3_8_27B.hidden_size, inter, weights=weights) for n in names]
        return reference.gate_up_swiglu(*a).to(torch.bfloat16)

    return [
        TensorSpec("act", [t, inter], torch.bfloat16, init_value=init_act),
        TensorSpec("act_q", [t, inter], torch.int8),
        TensorSpec("act_scale", [1, t], torch.float32),
    ]


def golden_qwen3_act_quant(tensors):
    import reference

    act_q, scale = reference.quantize_rows(tensors["act"])
    tensors["act_q"].copy_(act_q)
    tensors["act_scale"].copy_(scale.reshape(1, -1))


if __name__ == "__main__":
    import argparse

    from golden import run

    from utils import int8_step_and_exact

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--weights", type=str, default=None,
                        help="one layer's BF16 mlp tensors from weights.py; default: random weights")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % TOK_TILE:
        parser.error(f"--seq-len must be a multiple of {TOK_TILE}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        print(f"[case] act_quant T={t}", flush=True)
        result = run(
            fn=qwen3_act_quant_test,
            specs=build_tensor_specs(t=t, weights=args.weights),
            golden_fn=golden_qwen3_act_quant,
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
