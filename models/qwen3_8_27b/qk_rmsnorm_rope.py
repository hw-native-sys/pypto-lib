# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B attention q/k norm and partial RoPE, per head:

    y = rope(x * rsqrt(mean(x^2) + eps) * (1 + w)) ,  q additionally times head_dim ** -0.5

`q` is read from the `q_proj` output, whose head `h` holds `[q_h | gate_h]` in
columns `[2 * D * h, 2 * D * (h + 1))`. RoPE rotates the first ROT dimensions as
two halves (`rotate_half`) and passes the rest through.

The tile is a `build_kernel` argument because the task count is a quotient of it, and
at a decode batch the prefill value makes that quotient zero. `decode_gated_attention`
passes the decode value.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
HQ = QWEN3_8_27B.num_attention_heads    # query heads
HKV = QWEN3_8_27B.num_key_value_heads   # KV heads
D = QWEN3_8_27B.head_dim                # head dimension, the norm's axis
ROT = QWEN3_8_27B.rotary_dim            # leading dimensions RoPE rotates
EPS = QWEN3_8_27B.rms_norm_eps
# validation
DEFAULT_CASES = (32, 64)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
TOK_TILE = 32           # tokens per work item; the RoPE tables are read once per item
SPMD_BLOCKS = 48        # one per vector core
# The decode form fans out over heads instead of tokens; see `build_fused_kernel`.
Q_BLOCKS = 6            # tasks over the (token tile, query head) axis
K_BLOCKS = 2            # tasks over the (token tile, KV head) axis
HEAD_STAGE = 2          # pipeline depth of a task's run of heads


QG_WIDTH = 2 * HQ * D
Q_WIDTH = HQ * D
KV_WIDTH = HKV * D
HALF = ROT // 2
D_INV = 1.0 / D
SCALE = D ** -0.5                       # the score scale, folded into q; a power of two, so exact


# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True


def build_kernel(tok_tile: int = TOK_TILE, spmd_blocks: int = SPMD_BLOCKS,
                 inline: bool = False):
    """Norm and RoPE at a given token tile; `t` must be a multiple of `tok_tile`."""

    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_qk_rmsnorm_rope(
        qg: pl.Tensor[[T_DYN, QG_WIDTH], pl.BF16],
        k: pl.Tensor[[T_DYN, KV_WIDTH], pl.BF16],
        q_norm_w: pl.Tensor[[D], pl.BF16],
        k_norm_w: pl.Tensor[[D], pl.BF16],
        rope_cos: pl.Tensor[[T_DYN, ROT], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROT], pl.FP32],
        q_rot: pl.Out[pl.Tensor[[T_DYN, Q_WIDTH], pl.BF16]],
        k_rot: pl.Out[pl.Tensor[[T_DYN, KV_WIDTH], pl.BF16]],
    ):
        """The token count must be a multiple of tok_tile; a step walks one whole tile."""
        qg.bind_dynamic(0, T_DYN)
        k.bind_dynamic(0, T_DYN)
        rope_cos.bind_dynamic(0, T_DYN)
        rope_sin.bind_dynamic(0, T_DYN)
        q_rot.bind_dynamic(0, T_DYN)
        k_rot.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(qg, 0)
        q_w_row = pl.reshape(q_norm_w[:], [1, D])
        k_w_row = pl.reshape(k_norm_w[:], [1, D])
        for blk in pl.spmd(spmd_blocks, name_hint="qk_rmsnorm_rope"):
            for ti in pl.range(blk, t_dim // tok_tile, spmd_blocks):
                t0 = ti * tok_tile
                cos_lo = rope_cos[t0 : t0 + tok_tile, 0:HALF]
                cos_hi = rope_cos[t0 : t0 + tok_tile, HALF:ROT]
                sin_lo = rope_sin[t0 : t0 + tok_tile, 0:HALF]
                sin_hi = rope_sin[t0 : t0 + tok_tile, HALF:ROT]
                q_gamma = pl.mul(pl.add(pl.cast(q_w_row, target_type=pl.FP32), 1.0), SCALE)
                k_gamma = pl.add(pl.cast(k_w_row, target_type=pl.FP32), 1.0)

                for hh in pl.range(HQ):
                    q_c0 = hh * 2 * D
                    q_x = pl.cast(qg[t0 : t0 + tok_tile, q_c0 : q_c0 + D], target_type=pl.FP32)
                    q_ss = pl.row_sum(pl.mul(q_x, q_x))
                    q_inv = pl.rsqrt(pl.add(pl.mul(q_ss, D_INV), EPS), high_precision=True)
                    q_y = pl.col_expand_mul(pl.row_expand_mul(q_x, q_inv), q_gamma)
                    q_lo = q_y[:, 0:HALF]
                    q_hi = q_y[:, HALF:ROT]
                    q_rot_lo = pl.sub(pl.mul(q_lo, cos_lo), pl.mul(q_hi, sin_lo))
                    q_rot_hi = pl.add(pl.mul(q_hi, cos_hi), pl.mul(q_lo, sin_hi))
                    q_out = pl.concat(pl.concat(q_rot_lo, q_rot_hi), q_y[:, ROT:D])
                    q_rot[t0 : t0 + tok_tile, hh * D : hh * D + D] = pl.cast(q_out, target_type=pl.BF16, mode="rint")

                for kh in pl.range(HKV):
                    k_c0 = kh * D
                    k_x = pl.cast(k[t0 : t0 + tok_tile, k_c0 : k_c0 + D], target_type=pl.FP32)
                    k_ss = pl.row_sum(pl.mul(k_x, k_x))
                    k_inv = pl.rsqrt(pl.add(pl.mul(k_ss, D_INV), EPS), high_precision=True)
                    k_y = pl.col_expand_mul(pl.row_expand_mul(k_x, k_inv), k_gamma)
                    k_lo = k_y[:, 0:HALF]
                    k_hi = k_y[:, HALF:ROT]
                    k_rot_lo = pl.sub(pl.mul(k_lo, cos_lo), pl.mul(k_hi, sin_lo))
                    k_rot_hi = pl.add(pl.mul(k_hi, cos_hi), pl.mul(k_lo, sin_hi))
                    k_out = pl.concat(pl.concat(k_rot_lo, k_rot_hi), k_y[:, ROT:D])
                    k_rot[t0 : t0 + tok_tile, k_c0 : k_c0 + D] = pl.cast(k_out, target_type=pl.BF16, mode="rint")
        return q_rot, k_rot

    return _qwen3_qk_rmsnorm_rope


def build_fused_kernel(tok_tile: int = TOK_TILE, q_blocks: int = Q_BLOCKS,
                       k_blocks: int = K_BLOCKS, stage: int = HEAD_STAGE,
                       inline: bool = False):
    """The decode form: q and k out of one fused projection output, fanned over heads.

    The decode projection emits q, its gate, k and v as one `[T, ATTN_QKV_WIDTH]` tensor. q
    and its gate occupy the leading columns, so every q offset here is the one `build_kernel`
    uses; k sits at `QKV_K_BASE` and v above it, unread. The offsets come from
    `decode_projection` so that one module names the layout.

    **The fan-out is over heads, because at a decode batch the token axis is gone.**
    `build_kernel` spreads token tiles across 48 blocks with the heads serial inside; here the
    `(token tile, head)` axis is flattened and split over two `pl.spmd` loops, the form
    `qk_norm_gate` uses. The two groups cannot be one: telling a query head from a KV head
    would be a branch on the block index, which is a runtime scalar.

    Every structure writes each head's output tile from exactly one task and the arithmetic
    per head is untouched, so this is **bit-identical** to `build_kernel` on the same inputs,
    which is what catches an index slip in the flattening or in the pipelined loop's tail
    dispatch.
    """
    from decode_projection import ATTN_QKV_WIDTH, QKV_K_BASE

    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_qk_rmsnorm_rope_fused(
        qkv: pl.Tensor[[T_DYN, ATTN_QKV_WIDTH], pl.BF16],
        q_norm_w: pl.Tensor[[D], pl.BF16],
        k_norm_w: pl.Tensor[[D], pl.BF16],
        rope_cos: pl.Tensor[[T_DYN, ROT], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROT], pl.FP32],
        q_rot: pl.Out[pl.Tensor[[T_DYN, Q_WIDTH], pl.BF16]],
        k_rot: pl.Out[pl.Tensor[[T_DYN, KV_WIDTH], pl.BF16]],
    ):
        """The token count must be a multiple of tok_tile; one task walks whole heads."""
        qkv.bind_dynamic(0, T_DYN)
        rope_cos.bind_dynamic(0, T_DYN)
        rope_sin.bind_dynamic(0, T_DYN)
        q_rot.bind_dynamic(0, T_DYN)
        k_rot.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(qkv, 0)
        q_w_row = pl.reshape(q_norm_w[:], [1, D])
        k_w_row = pl.reshape(k_norm_w[:], [1, D])
        tiles = t_dim // tok_tile

        for qb in pl.spmd(q_blocks, name_hint="qk_rmsnorm_rope_fused_q",
                          allow_early_resolve=EARLY_RESOLVE):
            for idx in pl.pipeline(qb, tiles * HQ, q_blocks, stage=stage):
                ti = idx // HQ
                hh = idx - ti * HQ
                t0 = ti * tok_tile
                q_gamma = pl.mul(pl.add(pl.cast(q_w_row, target_type=pl.FP32), 1.0), SCALE)
                q_c0 = hh * 2 * D
                q_x = pl.cast(qkv[t0 : t0 + tok_tile, q_c0 : q_c0 + D], target_type=pl.FP32)
                q_ss = pl.row_sum(pl.mul(q_x, q_x))
                q_inv = pl.rsqrt(pl.add(pl.mul(q_ss, D_INV), EPS), high_precision=True)
                q_y = pl.col_expand_mul(pl.row_expand_mul(q_x, q_inv), q_gamma)
                q_lo = q_y[:, 0:HALF]
                q_hi = q_y[:, HALF:ROT]
                q_rot_lo = pl.sub(pl.mul(q_lo, rope_cos[t0 : t0 + tok_tile, 0:HALF]),
                                  pl.mul(q_hi, rope_sin[t0 : t0 + tok_tile, 0:HALF]))
                q_rot_hi = pl.add(pl.mul(q_hi, rope_cos[t0 : t0 + tok_tile, HALF:ROT]),
                                  pl.mul(q_lo, rope_sin[t0 : t0 + tok_tile, HALF:ROT]))
                q_out = pl.concat(pl.concat(q_rot_lo, q_rot_hi), q_y[:, ROT:D])
                q_rot[t0 : t0 + tok_tile, hh * D : hh * D + D] = pl.cast(q_out, target_type=pl.BF16, mode="rint")

        for kb in pl.spmd(k_blocks, name_hint="qk_rmsnorm_rope_fused_k",
                          allow_early_resolve=EARLY_RESOLVE):
            for idx in pl.pipeline(kb, tiles * HKV, k_blocks, stage=stage):
                ti = idx // HKV
                kh = idx - ti * HKV
                t0 = ti * tok_tile
                k_gamma = pl.add(pl.cast(k_w_row, target_type=pl.FP32), 1.0)
                k_c0 = QKV_K_BASE + kh * D
                k_x = pl.cast(qkv[t0 : t0 + tok_tile, k_c0 : k_c0 + D], target_type=pl.FP32)
                k_ss = pl.row_sum(pl.mul(k_x, k_x))
                k_inv = pl.rsqrt(pl.add(pl.mul(k_ss, D_INV), EPS), high_precision=True)
                k_y = pl.col_expand_mul(pl.row_expand_mul(k_x, k_inv), k_gamma)
                k_lo = k_y[:, 0:HALF]
                k_hi = k_y[:, HALF:ROT]
                k_rot_lo = pl.sub(pl.mul(k_lo, rope_cos[t0 : t0 + tok_tile, 0:HALF]),
                                  pl.mul(k_hi, rope_sin[t0 : t0 + tok_tile, 0:HALF]))
                k_rot_hi = pl.add(pl.mul(k_hi, rope_cos[t0 : t0 + tok_tile, HALF:ROT]),
                                  pl.mul(k_lo, rope_sin[t0 : t0 + tok_tile, HALF:ROT]))
                k_out = pl.concat(pl.concat(k_rot_lo, k_rot_hi), k_y[:, ROT:D])
                k_rot[t0 : t0 + tok_tile, kh * D : kh * D + D] = pl.cast(k_out, target_type=pl.BF16, mode="rint")
        return q_rot, k_rot

    return _qwen3_qk_rmsnorm_rope_fused


qwen3_qk_rmsnorm_rope = build_kernel(inline=True)
qwen3_qk_rmsnorm_rope_test = build_kernel()


def build_tensor_specs(t: int = BENCH_SEQ, hq: int = HQ, hkv: int = HKV, d: int = D, rot: int = ROT):
    import torch
    from golden import TensorSpec

    import reference

    def draw(seed, shape, scale=1.0):
        def make():
            gen = torch.Generator().manual_seed(seed)
            return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16)
        return make

    def tables():
        cos, sin = reference.rotary_cos_sin(torch.arange(t), QWEN3_8_27B)
        return cos.float(), sin.float()

    return [
        TensorSpec("qg", [t, 2 * hq * d], torch.bfloat16, init_value=draw(42, (t, 2 * hq * d), 3.0)),
        TensorSpec("k", [t, hkv * d], torch.bfloat16, init_value=draw(43, (t, hkv * d), 3.0)),
        TensorSpec("q_norm_w", [d], torch.bfloat16, init_value=draw(44, (d,), 0.5)),
        TensorSpec("k_norm_w", [d], torch.bfloat16, init_value=draw(45, (d,), 0.5)),
        TensorSpec("rope_cos", [t, rot], torch.float32, init_value=lambda: tables()[0]),
        TensorSpec("rope_sin", [t, rot], torch.float32, init_value=lambda: tables()[1]),
        TensorSpec("q_rot", [t, hq * d], torch.bfloat16),
        TensorSpec("k_rot", [t, hkv * d], torch.bfloat16),
    ]


# acceptance, per BF16 output against the float64 reference rounded the same
# way. A last-bit difference in an fp32 rsqrt or rotation moves an element
# across a BF16 rounding boundary, so one BF16 step is the tolerance -- except
# where the rotation cancels, `x_lo cos - x_hi sin` landing near zero: there
# the step is tiny and the fp32 error of the two products is not. The bound is
# therefore one step of the element OR a fixed fraction of its row's RMS,
# whichever is larger. The exact-match rate is what would expose a wrong
# rounding mode or a lost term.
MIN_EXACT = 0.99
ROW_FLOOR = 2.0 ** -16          # of the row's RMS; fp32 noise is ~2^-24, one BF16 step 2^-8


def compare_bf16(actual, expected, **_kwargs):
    """Every element within one BF16 step (or the row floor) of the golden, and most exact."""
    import torch

    a, e = actual.double(), expected.double()
    # Non-finite first. `diff > bound` is False when `diff` is NaN, so a NaN output would
    # clear the bound and cost only one element of the exact rate -- a single NaN passes.
    # `golden.validation` rejects non-finite values for the comparators that delegate to
    # it; this one computes its own bound, so it has to reject them itself.
    finite = torch.isfinite(a) & torch.isfinite(e)
    if not bool(finite.all()):
        bad = (~finite).nonzero()
        r, c = int(bad[0][0]), int(bad[0][1])
        return False, (f"[stats] {bad.shape[0]} non-finite element(s); first at [{r}, {c}]: "
                       f"actual {float(a[r, c])}, golden {float(e[r, c])}")
    diff = (a - e).abs()
    step = torch.exp2(torch.floor(torch.log2(e.abs().clamp_min(1e-30))) - 7)
    row_rms = e.pow(2).mean(dim=-1, keepdim=True).sqrt()
    bound = torch.maximum(step, ROW_FLOOR * row_rms)
    exact = float((diff == 0).double().mean())
    over = diff > bound
    steps = diff / step
    worst = int(steps.flatten().argmax())
    r, c = divmod(worst, e.shape[-1])
    detail = (f"[stats] exact {exact:.5f} (gate {MIN_EXACT}), {int(over.sum())} outside one step "
              f"or {ROW_FLOOR:.1e} of the row RMS; worst {float(steps.max()):.0f} steps at [{r}, {c}], "
              f"|golden| {float(e[r, c].abs()):.3e} against row RMS {float(row_rms[r]):.3e}, "
              f"diff {float(diff[r, c]):.3e} = {float(diff[r, c] / row_rms[r]):.1e} of the RMS")
    ok = exact >= MIN_EXACT and not bool(over.any())
    return ok, "" if ok else detail


def golden_qwen3_qk_rmsnorm_rope(eps: float = EPS):
    """The oracle at one `eps`, which must be the `eps` the kernel was built with."""
    import torch

    import reference

    def golden(tensors):
        q, k = reference.attn_qk_norm_rope(
            tensors["qg"], tensors["k"], tensors["q_norm_w"], tensors["k_norm_w"],
            tensors["rope_cos"], tensors["rope_sin"], HQ, HKV, D, eps,
        )
        tensors["q_rot"].copy_(q.to(torch.bfloat16))
        tensors["k_rot"].copy_(k.to(torch.bfloat16))

    return golden


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--eps", type=float, default=EPS)
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
        print(f"[case] qk_rmsnorm_rope T={t}", flush=True)
        result = run(
            fn=qwen3_qk_rmsnorm_rope_test,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_qwen3_qk_rmsnorm_rope(args.eps),
            compare_fn={"q_rot": compare_bf16, "k_rot": compare_bf16},
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=1e-2,
            atol=1e-2,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={t}")
