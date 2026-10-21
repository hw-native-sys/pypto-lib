# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet scaled_dot_kkt: the gated intra-chunk key-key matrix,
A[i, j] = (k_i . k_j) * exp(min(g_i - g_j, 0)) * beta_i for j < i, else 0."""
import pypto.language as pl
import torch

from chunk_cumsum import CHUNK_TILE, cumsum_ref
from config import QWEN3_8_27B
from reference import REF_DTYPE, make_inputs, to_hT

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D

# tiling
A_WIDTH = H * CHUNK_TILE
COL_TILE = 128          # columns of A per matmul
SLOT_NUM = 1            # cross-core ring depth; the default cannot hold a
                        # [CHUNK_TILE, COL_TILE] FP32 crossing tile (pypto#2903)

# validation
# One chunk, the cross-chunk recurrence, then eight chunks. The token count must be a
# multiple of CHUNK_TILE.
DEFAULT_CASES = (128, 256, 1024)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len


def _gdn_scaled_dot_kkt(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    beta: pl.Tensor[[H, T_DYN], pl.FP32],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    mask: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    a_out: pl.Out[pl.Tensor[[T_DYN, H, CHUNK_TILE], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE; a block walks one whole chunk."""
    k.bind_dynamic(0, T_DYN)
    beta.bind_dynamic(1, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    a_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(k, 0)
    # BSND [T, Hg, D] viewed as [T, Hg*D]: a per-head slice is a strided 2D window
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    a_flat = pl.reshape(a_out, [t_dim, A_WIDTH])
    for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="scaled_dot_kkt",
                      optimizations=[pl.cross_core_slot(slot_num=SLOT_NUM),
                                     pl.split(pl.SplitMode.UP_DOWN)]):
        t0 = c0 * CHUNK_TILE
        msk = mask[:, :]                       # constant, held for the whole scope
        for hh in pl.range(H):
            # GQA: value head hh reads key head hh // grp, the same mapping the
            # model reaches by repeat_interleave. GRP == 1 leaves this an identity.
            d0 = (hh // GRP) * D
            kc = k_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            # g_sum and beta are head-major [H, T]: one head's chunk views as
            # [1, CHUNK_TILE] for a row broadcast and [CHUNK_TILE, 1] for a column one.
            g_col = pl.reshape(g_sum[hh : hh + 1, t0 : t0 + CHUNK_TILE], [CHUNK_TILE, 1])
            beta_col = pl.reshape(beta[hh : hh + 1, t0 : t0 + CHUNK_TILE], [CHUNK_TILE, 1])
            for j0 in pl.unroll(0, CHUNK_TILE, COL_TILE):
                kj = k_flat[t0 + j0 : t0 + j0 + COL_TILE, d0 : d0 + D]
                scores = pl.matmul(kc, kj, b_trans=True)
                g_row = g_sum[hh : hh + 1, t0 + j0 : t0 + j0 + COL_TILE]
                diff = pl.full([CHUNK_TILE, COL_TILE], dtype=pl.FP32, value=0.0)
                diff = pl.row_expand_add(diff, g_col)
                diff = pl.col_expand_sub(diff, g_row)
                decay = pl.exp(pl.minimum(diff, 0.0))
                gated = pl.mul(scores, decay)
                gated = pl.row_expand_mul(gated, beta_col)
                masked = pl.mul(gated, msk[:, j0 : j0 + COL_TILE])
                masked16 = pl.cast(masked, target_type=pl.FP16, mode="rint")
                col = hh * CHUNK_TILE + j0
                a_flat[t0 : t0 + CHUNK_TILE, col : col + COL_TILE] = masked16
    return a_out


gdn_scaled_dot_kkt = pl.jit.inline(_gdn_scaled_dot_kkt)
gdn_scaled_dot_kkt_test = pl.jit(_gdn_scaled_dot_kkt)


def kkt_ref(k: torch.Tensor, beta: torch.Tensor, g_sum: torch.Tensor, chunk: int) -> torch.Tensor:
    """S3: the gated key-key matrix, strictly lower triangular. -> [T, H, chunk].

    The decay is `exp(g_i - g_j)` clamped to zero above the diagonal rather than
    `exp(min(g_i - g_j, 0))`: on the strict lower triangle the two agree, and the
    masked-out entries are discarded either way.
    """
    t, hg, _ = k.shape
    h = beta.shape[1]
    grp = h // hg
    kf, bf, gf = k.to(REF_DTYPE), beta.to(REF_DTYPE), g_sum.to(REF_DTYPE)
    out = torch.zeros(t, h, chunk, dtype=REF_DTYPE)
    rows = torch.arange(chunk)[:, None]
    cols = torch.arange(chunk)[None, :]
    strict_lower = (rows > cols).to(REF_DTYPE)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            kc = kf[t0 : t0 + chunk, hh // grp, :]
            gc = gf[t0 : t0 + chunk, hh]
            diff = gc[:, None] - gc[None, :]
            decay = torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
            out[t0 : t0 + chunk, hh, :] = (
                (kc @ kc.T) * decay * bf[t0 : t0 + chunk, hh, None] * strict_lower)
    return out


_INPUTS: dict = {}


def _inputs(t: int) -> dict:
    """The chain this stage consumes, memoised so the several specs pay for it once."""
    key = t
    if key not in _INPUTS:
        st = make_inputs(t, H, D, HG)
        st["g_sum"] = cumsum_ref(st["g"], CHUNK_TILE)
        _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ):
    from golden import TensorSpec

    def draw(key, transform=None):
        def make():
            value = _inputs(t)[key]
            return transform(value) if transform is not None else value

        return make

    def init_mask():
        rows = torch.arange(CHUNK_TILE)[:, None]
        cols = torch.arange(CHUNK_TILE)[None, :]
        return (rows > cols).float()

    return [
        TensorSpec("k", [t, HG, D], torch.float16,
                   init_value=draw("k")),
        TensorSpec("beta", [H, t], torch.float32,
                   init_value=draw("beta", to_hT)),
        TensorSpec("g_sum", [H, t], torch.float32,
                   init_value=draw("g_sum", to_hT)),
        TensorSpec("mask", [CHUNK_TILE, CHUNK_TILE], torch.float32, init_value=init_mask),
        TensorSpec("a_out", [t, H, CHUNK_TILE], torch.float16),
    ]


def golden_gdn_scaled_dot_kkt(tensors):
    chunk = tensors["mask"].shape[0]
    ref = kkt_ref(tensors["k"], tensors["beta"].t(), tensors["g_sum"].t(), chunk)
    tensors["a_out"].copy_(ref)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        print(f"[case] scaled_dot_kkt T={t}", flush=True)
        result = run(
            fn=gdn_scaled_dot_kkt_test,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_gdn_scaled_dot_kkt,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-2,
            atol=1e-5,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
