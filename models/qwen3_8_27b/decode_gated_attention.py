# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Qwen3.8-27B's gated-attention mixer at a decode batch: one token per sequence.

    x_q, x_scale   = quant_x(x)                     per-token int8 of the hidden states
    qkv            = qkv_proj(x_q, x_scale)         5120 -> 14336, A8W8, q|gate|k|v fused
    q_rot, k_rot   = qk_rmsnorm_rope(qkv)           per-head norm and partial RoPE
    k_pages,v_pages += kv_append(k_rot, qkv)        this token into the paged cache
    attended       = attention(q_rot, pages)        paged GQA over the whole context
    act_q,act_scale = attn_gate_quant(attended, qg) times sigmoid(gate), int8 epilogue
    out            = o_proj(act_q, act_scale)       6144 -> 5120, A8W8

Seven operators as one program. The three q/k/v projections are fused into one, which is
why the RoPE and the cache append read their inputs as channel ranges of a single
`[m, 14336]` tensor rather than as separate tensors.

**`attn_gate_quant` reads a narrowed copy of the fused tensor.** Its body views `qg` as
`[t * HQ, 2 * D]` by a flat reshape, which is a valid reinterpretation only when `qg` is
exactly `[t, 2 * HQ * D]`; against the 14336-wide fused tensor the same reshape would
stride wrongly. A declared-width argument therefore cannot serve it the way it serves the
RoPE, whose per-head offsets are unchanged because q leads the layout.

**The cache grows by one row per sequence per step.** `kv_append` writes before the
attention reads, and both derive the slot the same way from `block_table` and `seq_lens`,
where `seq_lens[b]` counts the token being decoded. Passing a length that excluded it
would make the reader skip the row just written and nothing in either signature would
notice.

The batch must be a positive multiple of 16 (`decode_projection.check_batch`).
"""
import pypto.language as pl

import attn_gate_quant
import decode_attention
import decode_kv_append
import decode_projection
import qk_rmsnorm_rope
import quant_x
from config import QWEN3_8_27B
from decode_projection import check_batch, check_tile, padded_batch

# model config
C = QWEN3_8_27B.hidden_size                 # the residual stream
QKV_WIDTH = decode_projection.ATTN_QKV_WIDTH    # q and gate, then k, then v
QG_WIDTH = attn_gate_quant.QG_WIDTH             # q and its gate, the leading columns
ATT_WIDTH = attn_gate_quant.ATT_WIDTH           # one token's attention output
KV_WIDTH = decode_kv_append.KV_WIDTH
HQ = QWEN3_8_27B.num_attention_heads
D = QWEN3_8_27B.head_dim
ROT = QWEN3_8_27B.rotary_dim

# validation
# one padded decode batch, then two; the inlined tiles are checked against it.
DEFAULT_CASES = (16, 32)

# tiling
QUANT_GROUPS = 4        # quant_x column groups
ROPE_TOK_TILE = 16      # qk_rmsnorm_rope at a padded batch (decode_projection)
QG_COPY_TILE = 2048     # columns per task of the narrowing copy, at the smallest batches
VEC_BUFFER_BYTES = 188416   # a2a3's vector buffer, which bounds that tile as the batch grows
# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True


# `attn_gate_quant`'s decode form, one task per token. A module constant rather than a
# build argument so both schedules can be built in one process; the two are bit-identical.
GATE_QUANT_FANNED = True
PAGE = decode_kv_append.PAGE


def _qg_copy_tile(m: int) -> int:
    """The widest column tile of `qg` the narrowing copy can hold at this batch.

    The copy keeps one `[m, tile]` BF16 tile live, so the bound moves with the batch rather
    than being a property of the stage: 2048 columns are 64 KiB at batch 16 and 192 KiB at
    batch 48, where the compiler refuses the build -- *"Vec buffer usage (196608 bytes)
    exceeds platform limit (188416 bytes)"*. Halving the tile costs nothing, the
    stage being a copy whose task count simply doubles, so the tile is derived here instead
    of pinned.
    """
    tile = QG_COPY_TILE
    while tile > 64 and m * tile * 2 > VEC_BUFFER_BYTES:
        tile //= 2
    if QG_WIDTH % tile or m * tile * 2 > VEC_BUFFER_BYTES:
        raise ValueError(f"batch {m} leaves no column tile of {QG_WIDTH} that fits the "
                         f"{VEC_BUFFER_BYTES} B vector buffer")
    return tile


def run_config(backend_type: str = "a2a3", sim: bool = False, device: int = 0,
               swimlane: int = 0) -> dict:
    """The config `golden.run` needs for this mixer."""
    return dict(backend_type=backend_type, execution_mode=sim, device_id=device,
                enable_chip_swimlane=swimlane)


def build_kernel(batch: int, max_pages: int, splits: int | None = None,
                 inline: bool = False):
    """The mixer at one decode batch and cache depth, composing seven operators."""
    m = padded_batch(batch)
    check_batch(m)
    check_tile("ROPE_TOK_TILE", ROPE_TOK_TILE, m)

    qg_copy_tile = _qg_copy_tile(m)
    quant = quant_x.build_fanned_kernel(m, col_groups=QUANT_GROUPS, inline=True)
    op_qkv = decode_projection.build_kernel("qkv_proj", m, inline=True)
    rope = qk_rmsnorm_rope.build_fused_kernel(tok_tile=ROPE_TOK_TILE, inline=True)
    append = decode_kv_append.build_fused_kernel(max_pages=max_pages, inline=True)
    attend_kwargs = dict(max_pages=max_pages, inline=True)
    if splits is not None:
        attend_kwargs["splits"] = splits
    attend = decode_attention.build_kernel(**attend_kwargs)
    # `o_proj`, not `out_proj`: the same matmul shape, but this one is int8 in the
    # checkpoint while the GDN projection is not.
    op_out = decode_projection.build_kernel("o_proj", m, inline=True)
    gate_quant = (attn_gate_quant.build_fanned_kernel(m, inline=True) if GATE_QUANT_FANNED
                  else attn_gate_quant.qwen3_attn_gate_quant)

    @(pl.jit.inline if inline else pl.jit)
    def decode_gated_attention(
        x: pl.Tensor[[m, C], pl.BF16],
        w_qkv: pl.Tensor[[QKV_WIDTH, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[QKV_WIDTH], pl.FP32],
        q_norm_w: pl.Tensor[[D], pl.BF16],
        k_norm_w: pl.Tensor[[D], pl.BF16],
        rope_cos: pl.Tensor[[m, ROT], pl.FP32],
        rope_sin: pl.Tensor[[m, ROT], pl.FP32],
        block_table: pl.Tensor[[m, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[m], pl.INT32],
        tail_mask: pl.Tensor[[PAGE + 1, PAGE], pl.FP32],
        w_out: pl.Tensor[[C, ATT_WIDTH], pl.INT8],
        w_out_scale: pl.Tensor[[C], pl.FP32],
        k_pages: pl.InOut[pl.Tensor[[decode_kv_append.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[decode_kv_append.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        out: pl.Out[pl.Tensor[[m, C], pl.BF16]],
    ):
        """`seq_lens[b]` counts the token being decoded; the append writes before the read."""
        x_q = pl.create_tensor([m, C], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, m], dtype=pl.FP32)
        quant(x, x_q, x_scale)

        qkv = pl.create_tensor([m, QKV_WIDTH], dtype=pl.BF16)
        op_qkv(x_q, x_scale, w_qkv, w_qkv_scale, qkv)

        q_rot = pl.create_tensor([m, ATT_WIDTH], dtype=pl.BF16)
        k_rot = pl.create_tensor([m, KV_WIDTH], dtype=pl.BF16)
        rope(qkv, q_norm_w, k_norm_w, rope_cos, rope_sin, q_rot, k_rot)

        append(k_rot, qkv, block_table, seq_lens, k_pages, v_pages)

        attended = pl.create_tensor([m, ATT_WIDTH], dtype=pl.BF16)
        attend(q_rot, k_pages, v_pages, block_table, seq_lens, tail_mask, attended)

        # the narrowed view `attn_gate_quant`'s flat reshape needs; see the header
        qg = pl.create_tensor([m, QG_WIDTH], dtype=pl.BF16)
        for blk in pl.spmd(QG_WIDTH // qg_copy_tile, name_hint="qg_narrow",
                           allow_early_resolve=EARLY_RESOLVE):
            c0 = blk * qg_copy_tile
            qg[0:m, c0 : c0 + qg_copy_tile] = qkv[0:m, c0 : c0 + qg_copy_tile]

        act_q = pl.create_tensor([m, ATT_WIDTH], dtype=pl.INT8)
        act_scale = pl.create_tensor([1, m], dtype=pl.FP32)
        gate_quant(attended, qg, act_q, act_scale)

        op_out(act_q, act_scale, w_out, w_out_scale, out)
        return out

    return decode_gated_attention


__all__ = ["attn_weights", "build_kernel", "run_config"]


def build_tensor_specs(batch: int = 16, context: int = 320, max_pages: int = 8,
                       pool_pages: int | None = None, weights: str | None = None):
    """Every weight the mixer reads, plus a paged cache whose pages are scattered.

    The block table strides through the pool rather than running, so a kernel that derived
    a slot from the sequence index would land on another sequence's pages. `context` counts
    the token being decoded and is deliberately not a multiple of `PAGE`, so the last page
    is partial and the reader's tail mask has to bind.
    """
    import torch
    from golden import TensorSpec

    import reference

    m = padded_batch(batch)
    lens, table_rows, pool, owned = decode_attention.page_layout(m, context, max_pages, pool_pages)

    def attn(key):
        return reference.attn_inputs(key, m, QWEN3_8_27B, weights=weights)

    def fused_weight():
        w, _ = decode_projection.fuse_qkv_weight(
            attn("w_q"), attn("w_k"), attn("w_v"),
            attn("w_q_scale"), attn("w_k_scale"), attn("w_v_scale"))
        return w

    def fused_scale():
        _, s = decode_projection.fuse_qkv_weight(
            attn("w_q"), attn("w_k"), attn("w_v"),
            attn("w_q_scale"), attn("w_k_scale"), attn("w_v_scale"))
        return s

    def tables_cache(seed):
        # Rows no sequence owns are zero: see the contract in `decode_attention`.
        def make():
            t = torch.randn(pool * PAGE, KV_WIDTH, generator=torch.Generator().manual_seed(seed)) * 0.5
            t[~owned] = 0.0
            return t.to(torch.bfloat16)
        return make

    return [
        TensorSpec("x", [m, C], torch.bfloat16, init_value=lambda: attn("x")),
        TensorSpec("w_qkv", [QKV_WIDTH, C], torch.int8, init_value=fused_weight),
        TensorSpec("w_qkv_scale", [QKV_WIDTH], torch.float32, init_value=fused_scale),
        TensorSpec("q_norm_w", [D], torch.bfloat16, init_value=lambda: attn("q_norm_w")),
        TensorSpec("k_norm_w", [D], torch.bfloat16, init_value=lambda: attn("k_norm_w")),
        TensorSpec("rope_cos", [m, ROT], torch.float32,
                   init_value=lambda: reference.rotary_cos_sin(
                       torch.full((m,), context - 1), QWEN3_8_27B)[0].float()),
        TensorSpec("rope_sin", [m, ROT], torch.float32,
                   init_value=lambda: reference.rotary_cos_sin(
                       torch.full((m,), context - 1), QWEN3_8_27B)[1].float()),
        TensorSpec("block_table", [m, max_pages], torch.int32, init_value=lambda: table_rows),
        TensorSpec("seq_lens", [m], torch.int32, init_value=lambda: lens),
        TensorSpec("tail_mask", [PAGE + 1, PAGE], torch.float32,
                   init_value=lambda: decode_attention.build_tail_mask()),
        TensorSpec("w_out", [C, ATT_WIDTH], torch.int8, init_value=lambda: attn("w_o")),
        TensorSpec("w_out_scale", [C], torch.float32, init_value=lambda: attn("w_o_scale")),
        TensorSpec("k_pages", [pool * PAGE, KV_WIDTH], torch.bfloat16,
                   init_value=tables_cache(31)),
        TensorSpec("v_pages", [pool * PAGE, KV_WIDTH], torch.bfloat16,
                   init_value=tables_cache(32)),
        TensorSpec("out", [m, C], torch.bfloat16),
    ]


def attn_weights(tensors):
    """The flat `attn_inputs` dict the oracles take, rebuilt from the fused spec.

    `w_q`, `w_k` and `w_v` are the row ranges of the fused weight, which is how it was
    assembled, so splitting it back is exact rather than approximate.
    """
    q_rows = QG_WIDTH
    kv = KV_WIDTH
    w_qkv, w_scale = tensors["w_qkv"], tensors["w_qkv_scale"]
    return {
        "w_q": w_qkv[0:q_rows], "w_q_scale": w_scale[0:q_rows],
        "w_k": w_qkv[q_rows : q_rows + kv], "w_k_scale": w_scale[q_rows : q_rows + kv],
        "w_v": w_qkv[q_rows + kv : q_rows + 2 * kv],
        "w_v_scale": w_scale[q_rows + kv : q_rows + 2 * kv],
        "w_o": tensors["w_out"], "w_o_scale": tensors["w_out_scale"],
        "q_norm_w": tensors["q_norm_w"], "k_norm_w": tensors["k_norm_w"],
    }


_TRUTH: dict[str, object] = {}


def golden_decode_gated_attention(tensors):
    """`reference.attn_w8a8_decode`, and the float64 truth stashed for the gate.

    **Both oracles append to the cache in place, and by the time a comparator runs the
    kernel has overwritten the real one**, so the truth cannot be computed there -- it is
    computed here, from clones, while the starting cache still exists. The fake-quant chain
    then runs on the harness's own tensors, which is what the `k_pages` and `v_pages` gates
    compare against.
    """
    import torch

    import reference

    w = attn_weights(tensors)
    _TRUTH["out"] = reference.attn_truth_decode(
        tensors["x"], w, tensors["rope_cos"], tensors["rope_sin"], QWEN3_8_27B,
        tensors["k_pages"].clone(), tensors["v_pages"].clone(), tensors["block_table"],
        tensors["seq_lens"], PAGE)
    out = reference.attn_w8a8_decode(
        tensors["x"], w, tensors["rope_cos"], tensors["rope_sin"], QWEN3_8_27B,
        tensors["k_pages"], tensors["v_pages"], tensors["block_table"],
        tensors["seq_lens"], PAGE)
    tensors["out"].copy_(out.to(torch.bfloat16))


# **`MAX_OVER_FLOOR` and `MAX_VS_GOLDEN` cannot both bind.** If the
# kernel may sit a whole floor away from the fake-quant chain, its distance to the truth can
# reach sqrt(2) floors when the two errors are orthogonal. The 1.05 this started at held only
# while the floor was dominated by a term the kernel reproduces exactly; taking the GDN tail to
# BF16 shrank that floor 2.8x, the kernel's own rounding is 0.36 to 0.42 of what is left, and
# the expected ratio is therefore sqrt(1 + 0.42^2) = 1.08. Three of four configurations read
# 1.042x to 1.050x against a 1.05 bound, which is a mis-calibrated gate rather than a kernel
# that moved. The binding gate is the distance to the chain the kernel implements; the truth
# ratio is a sanity bound, and a real regression shows up there as a multiple, not a percent.
MAX_OVER_FLOOR = 1.15
MAX_ROW_OVER_FLOOR = 1.25
MAX_VS_GOLDEN = 1.00
MAX_ROW_VS_GOLDEN = 1.50


def compare_out():
    """Ratios against the quantisation floor, as `gated_attention` and the GDN block use."""
    def compare(actual, expected, **_kwargs):
        truth = _TRUTH["out"].double()
        fq = expected.double()
        dev = actual.double()

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).max())

        floor, floor_row = rel(fq, truth), worst_row(fq, truth)
        got, got_row = rel(dev, truth), worst_row(dev, truth)
        vs_fq, vs_fq_row = rel(dev, fq), worst_row(dev, fq)
        ratio = got / floor if floor else float("inf")
        ratio_row = got_row / floor_row if floor_row else float("inf")
        fq_ratio = vs_fq / floor if floor else float("inf")
        fq_ratio_row = vs_fq_row / floor_row if floor_row else float("inf")
        line = (f"[stats] out: rel frob {got:.3e} vs the float64 truth against a "
                f"quantisation floor of {floor:.3e} ({ratio:.3f}x); worst token row "
                f"{got_row:.3e} against {floor_row:.3e} ({ratio_row:.3f}x); {vs_fq:.3e} vs "
                f"the fake-quant chain ({fq_ratio:.3f} of the floor, worst row "
                f"{fq_ratio_row:.3f})")
        ok = (ratio <= MAX_OVER_FLOOR and ratio_row <= MAX_ROW_OVER_FLOOR
              and fq_ratio <= MAX_VS_GOLDEN and fq_ratio_row <= MAX_ROW_VS_GOLDEN)
        return ok, "" if ok else line

    return compare


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--context", type=int, default=320)
    parser.add_argument("--max-pages", type=int, default=0,
                        help="0 derives it from --context")
    parser.add_argument("--splits", type=int, default=None)
    parser.add_argument("--weights", type=str, default=None,
                        help="one layer's BF16 self_attn tensors; default: random weights")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=range(5))
    args = parser.parse_args()
    cases = [args.batch] if args.batch else list(DEFAULT_CASES)

    max_pages = args.max_pages or (args.context + PAGE - 1) // PAGE

    for batch in cases:
        print(f"[case] decode_gated_attention batch={batch}", flush=True)
        result = run(
            fn=build_kernel(batch, max_pages=max_pages, splits=args.splits),
            specs=build_tensor_specs(batch=batch, context=args.context,
                                     max_pages=max_pages, weights=args.weights),
            golden_fn=golden_decode_gated_attention,
            compare_fn={"out": compare_out(),
                        "k_pages": decode_attention.compare_pages("k_pages"),
                        "v_pages": decode_attention.compare_pages("v_pages")},
            config=run_config(args.backend_type, args.execution_mode, args.device,
                              args.enable_chip_swimlane),
            rtol=1e-2,
            atol=1e-2,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at batch={batch}")
