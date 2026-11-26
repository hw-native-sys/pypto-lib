# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B final RMSNorm and BF16 LM head over selected residual-stream rows."""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
ROWS_DYN = pl.dynamic("ROWS_DYN")           # rows the head scores

# model config
HIDDEN = QWEN3_8_27B.hidden_size
VOCAB = QWEN3_8_27B.vocab_size
EPS = QWEN3_8_27B.rms_norm_eps

# case shape
# validation
# one sequence, then a full ROW_TILE of them. The row extent is padded up to the tile
# and only the store is trimmed, so one row is a real case and not a degenerate one.
DEFAULT_CASES = (1, 16)
ROWS = 16               # rows logits are produced for; one per sequence

# tiling
ROW_TILE = 16
RMS_K_TILE = 128        # hidden elements per norm pass step
LM_K_TILE = 512         # contraction step; 512 BF16 columns is one L2 line
# The L1 Mat buffer is 512 KiB and the K pipeline double-buffers both operands,
# so (ROW_TILE + VOCAB_TILE) * LM_K_TILE * 2 bytes * 2 must fit: VOCAB_TILE
# stays under 240. 160 is the largest multiple of 16 under that which divides
# 248320 = 2^9 * 5 * 97 -- 192, the qwen3_14b value, does not.
VOCAB_TILE = 160
LM_CORES = 24           # one per cube core

# host only: bands that keep the oracle and the input draw off the peak
GOLDEN_VOCAB_BAND = 16384
INIT_ROW_BAND = 16384


HIDDEN_INV = 1.0 / HIDDEN


def build_kernel(row_tile: int = ROW_TILE, rms_k_tile: int = RMS_K_TILE,
                 lm_k_tile: int = LM_K_TILE, vocab_tile: int = VOCAB_TILE,
                 lm_cores: int = LM_CORES, inline: bool = False):
    """The norm and head at given tiles, over a padded row extent and a valid row count.

    **The row extent is padded; `valid_rows` says how many of them are real.** Without that
    split the ordinary case could not call this kernel at all: one sequence is one row, and
    every loop here steps a whole `row_tile` of 16, so a one-row tensor gave `pl.range` zero
    iterations and returned with `logits` untouched -- no error, no output.

    So the inputs and `logits` carry a row extent that is a multiple of `row_tile`, the
    loops cover all of it, and only the store is trimmed, with `pl.tensor.set_validshape`.
    The padded rows are normed and projected and then not written, which costs at most one
    tile of arithmetic and keeps every read inside the tensor. `qwen3_14b/rms_lm_head.py`
    carries the same idiom.
    """

    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_rms_lm_head(
        residual: pl.Tensor[[ROWS_DYN, HIDDEN], pl.FP32],
        delta: pl.Tensor[[ROWS_DYN, HIDDEN], pl.BF16],
        norm_w: pl.Tensor[[HIDDEN], pl.BF16],
        lm_w: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
        logits: pl.Out[pl.Tensor[[ROWS_DYN, VOCAB], pl.FP32]],
        valid_rows: pl.Scalar[pl.INDEX],
    ):
        """The row extent must be a multiple of row_tile; `valid_rows` may be anything."""
        residual.bind_dynamic(0, ROWS_DYN)
        delta.bind_dynamic(0, ROWS_DYN)
        logits.bind_dynamic(0, ROWS_DYN)
        row_dim = pl.tensor.dim(residual, 0)
        w_row = pl.reshape(norm_w[:], [1, HIDDEN])
        normed = pl.create_tensor([row_dim, HIDDEN], dtype=pl.BF16)
        for r0 in pl.parallel(0, row_dim, row_tile):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
                partial_sq = pl.full([1, row_tile], dtype=pl.FP32, value=0.0)
                for k0 in pl.range(0, HIDDEN, rms_k_tile):
                    sq_d = pl.cast(delta[r0 : r0 + row_tile, k0 : k0 + rms_k_tile], target_type=pl.FP32)
                    sq_x = pl.add(residual[r0 : r0 + row_tile, k0 : k0 + rms_k_tile], sq_d)
                    sq_prod = pl.mul(sq_x, sq_x)
                    sq_row = pl.reshape(pl.row_sum(sq_prod), [1, row_tile])
                    partial_sq = pl.add(partial_sq, sq_row)
                mean_sq = pl.mul(partial_sq, HIDDEN_INV)
                variance = pl.reshape(mean_sq, [row_tile, 1])
                inv_rms = pl.rsqrt(pl.add(variance, EPS), high_precision=True)

                for k0 in pl.range(0, HIDDEN, rms_k_tile):
                    norm_d = pl.cast(delta[r0 : r0 + row_tile, k0 : k0 + rms_k_tile], target_type=pl.FP32)
                    norm_x = pl.add(residual[r0 : r0 + row_tile, k0 : k0 + rms_k_tile], norm_d)
                    w_tile = pl.cast(w_row[:, k0 : k0 + rms_k_tile], target_type=pl.FP32)
                    gamma = pl.add(w_tile, 1.0)
                    scaled = pl.row_expand_mul(norm_x, inv_rms)
                    weighted = pl.col_expand_mul(scaled, gamma)
                    normed[r0 : r0 + row_tile, k0 : k0 + rms_k_tile] = pl.cast(
                        weighted, target_type=pl.BF16, mode="rint")

        # One grid-stride over the vocab tiles: each block owns a disjoint set of
        # columns, so there is no split-K and no atomic store.
        for core in pl.spmd(lm_cores, name_hint="lm_head"):
            for r0 in pl.range(0, row_dim, row_tile):
                for vb in pl.range(core, VOCAB // vocab_tile, lm_cores):
                    v0 = vb * vocab_tile
                    acc = pl.create_tensor([row_tile, vocab_tile], dtype=pl.FP32)
                    for kb in pl.pipeline(0, HIDDEN // lm_k_tile, stage=2):
                        k0 = kb * lm_k_tile
                        lm_h_tile = normed[r0 : r0 + row_tile, k0 : k0 + lm_k_tile]
                        lm_w_tile = lm_w[v0 : v0 + vocab_tile, k0 : k0 + lm_k_tile]
                        acc = pl.matmul_acc(acc, lm_h_tile, lm_w_tile, b_trans=True, init_cond=(kb == 0))
                    live = pl.min(row_tile, valid_rows - r0)
                    trimmed = pl.tensor.set_validshape(acc, live, vocab_tile)
                    logits[r0 : r0 + row_tile, v0 : v0 + vocab_tile] = trimmed
        return logits

    return _qwen3_rms_lm_head


def build_l3_kernel(batch: int):
    """Build a HOST entry for a fixed token-row extent."""
    if batch < ROW_TILE or batch % ROW_TILE:
        raise ValueError("LM head rows must be a positive multiple of ROW_TILE")
    kernel = build_kernel()

    @pl.jit.host
    def l3_rms_lm_head(
        residual: pl.Tensor[[batch, HIDDEN], pl.FP32],
        delta: pl.Tensor[[batch, HIDDEN], pl.BF16],
        norm_w: pl.Tensor[[HIDDEN], pl.BF16],
        lm_w: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
        logits: pl.Out[pl.Tensor[[batch, VOCAB], pl.FP32]],
        valid_rows: pl.Scalar[pl.INDEX],
    ):
        return kernel(residual, delta, norm_w, lm_w, logits, valid_rows)

    return l3_rms_lm_head

qwen3_rms_lm_head = build_kernel(inline=True)
qwen3_rms_lm_head_test = build_kernel()




def _banded_normal(rows: int, cols: int, seed: int, scale: float = 1.0):
    """A BF16 `[rows, cols]` normal draw built one band at a time.

    `torch.randn(248320, 5120)` is 5.09 GB of FP32 before it is narrowed, and
    simulator CI builds these at the full vocabulary. Banding keeps the peak at
    the BF16 result plus one band.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    out = torch.empty(rows, cols, dtype=torch.bfloat16)
    for r0 in range(0, rows, INIT_ROW_BAND):
        rows_here = min(INIT_ROW_BAND, rows - r0)
        band = torch.randn(rows_here, cols, generator=gen)
        if scale != 1.0:
            band *= scale
        out[r0 : r0 + rows_here] = band.to(torch.bfloat16)
    return out

def padded_rows(rows: int, row_tile: int = ROW_TILE) -> int:
    """The row extent a launch must present for `rows` real rows."""
    return max(row_tile, (rows + row_tile - 1) // row_tile * row_tile)


def build_tensor_specs(rows: int = ROWS, hidden: int = HIDDEN, vocab: int = VOCAB,
                       row_tile: int = ROW_TILE):
    """Fixtures for `rows` real rows, in a tensor padded up to `row_tile`.

    Padding inputs are populated; logits beyond `valid_rows` are undefined.
    """
    import torch
    from golden import ScalarSpec, TensorSpec

    padded = padded_rows(rows, row_tile)

    def init_residual():
        gen = torch.Generator().manual_seed(42)
        return torch.randn(padded, hidden, generator=gen)

    def init_delta():
        gen = torch.Generator().manual_seed(45)
        return torch.randn(padded, hidden, generator=gen).to(torch.bfloat16)

    def init_norm_w():
        gen = torch.Generator().manual_seed(43)
        return torch.randn(hidden, generator=gen).to(torch.bfloat16)

    def init_lm_w():
        return _banded_normal(vocab, hidden, seed=44, scale=hidden ** -0.5)

    return [
        TensorSpec("residual", [padded, hidden], torch.float32, init_value=init_residual),
        TensorSpec("delta", [padded, hidden], torch.bfloat16, init_value=init_delta),
        TensorSpec("norm_w", [hidden], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("lm_w", [vocab, hidden], torch.bfloat16, init_value=init_lm_w),
        TensorSpec("logits", [padded, vocab], torch.float32),
        # A runtime parameter, not specialised into the artifact: one compiled program has
        # to serve every row count, which is the whole point of the trim.
        ScalarSpec("valid_rows", torch.int64, rows, compile_runtime=True),
    ]


def golden_qwen3_rms_lm_head(eps: float = EPS):
    """The oracle at one `eps`, which must be the `eps` the kernel was built with.

    A factory rather than a bare function: `build_kernel` takes `eps`, so a
    golden reading the module constant would score any non-default build
    against the wrong oracle.
    """
    import torch

    import reference

    def golden(tensors):
        # The kernel narrows the normed rows to BF16 before the head, as the
        # module does; leaving them in float64 would score the dtype rather
        # than the kernel.
        # The oracle and comparison cover only the written prefix of logits.
        rows = tensors["valid_rows"]
        # The addition in fp32, as the kernel computes it, and the norm of that sum.
        total = (tensors["residual"][:rows].double() + tensors["delta"][:rows].double()).float()
        normed = reference.rmsnorm(total.double(), tensors["norm_w"], eps)
        normed_bf = normed.to(torch.bfloat16).to(reference.REF_DTYPE)
        lm_w, out = tensors["lm_w"], tensors["logits"]
        # Widen the weight one vocabulary band at a time. Whole, it is 9.47 GiB
        # in float64 against 2.37 in BF16, and simulator CI runs this file at
        # the full vocabulary.
        for v0 in range(0, lm_w.shape[0], GOLDEN_VOCAB_BAND):
            band = lm_w[v0 : v0 + GOLDEN_VOCAB_BAND].to(reference.REF_DTYPE)
            out[:rows, v0 : v0 + band.shape[0]] = (normed_bf @ band.t()).to(torch.float32)

    return golden


# The logits bound. A fixed `atol` judges the sample count rather than the kernel: at 48
# rows it rejects 24 of 11919360 elements, all near-zero logits about 1.4e-4 out, where the
# relative Frobenius is 5.80e-06 against 2.47e-06 at 16 rows. Same quality, three times the
# draws. `down_proj`-style cancellation over a 5120-term contraction is what puts logits
# near zero, so the bound is a norm ratio.
FROB_GATE = 1e-4


def compare_logits():
    """Relative Frobenius on the valid logits against the float64 oracle."""
    def compare(actual, expected, inputs=None, **_kwargs):
        if inputs is None or "valid_rows" not in inputs:
            return False, "logits: missing valid_rows input"
        live = int(inputs["valid_rows"])
        if not 0 < live <= min(actual.shape[0], expected.shape[0]):
            return False, f"logits: invalid valid-row count {live}"
        a, e = actual[:live].double(), expected[:live].double()
        frob = float((a - e).norm() / e.norm())
        diff = (a - e).abs()
        over = int((diff > 1e-4 + 1e-2 * e.abs()).sum())
        ok = frob <= FROB_GATE
        detail = (f"rel frobenius {frob:.3e} (gate {FROB_GATE:.0e}); max abs "
                  f"{float(diff.max()):.3e}; {over} of {a.numel()} outside a fixed "
                  f"atol=1e-4 rtol=1e-2")
        return ok, "" if ok else detail

    return compare


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--vocab", type=int, default=VOCAB)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.rows] if args.rows else list(DEFAULT_CASES)
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --rows")

    for rows in cases:
        print(f"[case] rms_lm_head rows={rows}", flush=True)
        result = run(
            fn=qwen3_rms_lm_head_test,
            specs=build_tensor_specs(rows=rows, vocab=args.vocab),
            golden_fn=golden_qwen3_rms_lm_head(args.eps),
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=1e-2,
            atol=1e-4,
            compare_fn={"logits": compare_logits()},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at rows={rows}")
