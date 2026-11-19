# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet unit-triangular inverse with FP32 block substitution and an FP16 output.

Diagonal blocks use nilpotent doubling. Lower blocks satisfy
T[i,j] = -T[i,i] @ sum(A[i,k] @ T[k,j], k=j..i-1).
"""
import pypto.language as pl
import torch

from chunk_cumsum import cumsum_ref
from config import QWEN3_8_27B
from delta_rule_reference import solve_tril_ref
from reference import REF_DTYPE, frob_ok, make_inputs
from scaled_dot_kkt import CHUNK_TILE, kkt_ref

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension; unused here, drawn inputs match the pipeline

# tiling
A_WIDTH = H * CHUNK_TILE
BLOCK_TILE = 16         # diagonal inversion and substitution tile

# validation
# One chunk, the cross-chunk recurrence, then eight chunks. The token count must be a
# multiple of CHUNK_TILE.
DEFAULT_CASES = (128, 256, 1024)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len


# Diagonal doubling requires a power-of-two block that exactly partitions the chunk.
if BLOCK_TILE & (BLOCK_TILE - 1) or BLOCK_TILE < 4 or CHUNK_TILE % BLOCK_TILE or BLOCK_TILE >= CHUNK_TILE:
    raise ValueError(f"BLOCK_TILE must be a power of two in [4, {CHUNK_TILE}), got {BLOCK_TILE}")
ND_IN = BLOCK_TILE.bit_length() - 2          # X updates for (I + D)^-1, since D^BLOCK_TILE = 0


def _gdn_solve_tril(
    a_in: pl.Tensor[[T_DYN, H, CHUNK_TILE], pl.FP16],
    eye: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    m_diag: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    m_low: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    t_out: pl.Out[pl.Tensor[[T_DYN, H, CHUNK_TILE], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE; a block walks one whole chunk."""
    a_in.bind_dynamic(0, T_DYN)
    t_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(a_in, 0)
    a_flat = pl.reshape(a_in, [t_dim, A_WIDTH])
    t_flat = pl.reshape(t_out, [t_dim, A_WIDTH])
    # FP32 scratch holds the diagonal inverses and successive lower block distances.
    inverse = pl.create_tensor([t_dim, A_WIDTH], dtype=pl.FP32)
    for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="solve_zero"):
        t0 = c0 * CHUNK_TILE
        for hh in pl.range(H):
            col = hh * CHUNK_TILE
            zeros = pl.full([CHUNK_TILE, CHUNK_TILE], dtype=pl.FP32, value=0.0)
            inverse[t0 : t0 + CHUNK_TILE, col : col + CHUNK_TILE] = zeros
    for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="solve_diag", optimizations=[pl.cross_core_slot(slot_num=32)]):
        t0 = c0 * CHUNK_TILE
        for hh in pl.range(H):
            col = hh * CHUNK_TILE
            for b0 in pl.range(0, CHUNK_TILE, BLOCK_TILE):
                ident32 = pl.cast(eye[:BLOCK_TILE, :BLOCK_TILE], target_type=pl.FP32)
                row0 = t0 + b0
                col0 = col + b0
                block16 = a_flat[row0 : row0 + BLOCK_TILE, col0 : col0 + BLOCK_TILE]
                dm16 = pl.mul(block16, m_diag[:BLOCK_TILE, :BLOCK_TILE])
                dm = pl.cast(dm16, target_type=pl.FP32)
                nd = pl.matmul(dm, ident32, out_dtype=pl.FP32)
                xa = pl.matmul_acc(nd, ident32, ident32)
                xc = pl.add(xa, 0.0)
                yb = pl.matmul(dm, dm, out_dtype=pl.FP32)
                yc = pl.add(yb, 0.0)
                for _ in pl.unroll(ND_IN - 1):
                    xa = pl.matmul_acc(xa, xc, yc)
                    xc = pl.add(xa, 0.0)
                    yb = pl.matmul(yc, yc, out_dtype=pl.FP32)
                    yc = pl.add(yb, 0.0)
                xd = pl.matmul_acc(xa, xc, yc)
                inverse[row0 : row0 + BLOCK_TILE, col0 : col0 + BLOCK_TILE] = xd
    # Block distances are expanded before SPMD dispatch.
    for distance in pl.unroll(1, CHUNK_TILE // BLOCK_TILE):
        for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="solve_lower", optimizations=[pl.cross_core_slot(slot_num=4)]):
            t0 = c0 * CHUNK_TILE
            for hh in pl.range(H):
                col = hh * CHUNK_TILE
                for jj in pl.range(CHUNK_TILE // BLOCK_TILE - distance):
                    ii = jj + distance
                    i0 = ii * BLOCK_TILE
                    j0 = jj * BLOCK_TILE
                    row0 = t0 + i0
                    diag_col = col + i0
                    target_col = col + j0
                    a_row_half = a_flat[row0 : row0 + BLOCK_TILE, col : col + CHUNK_TILE]
                    a_row = pl.cast(a_row_half, target_type=pl.FP32)
                    inv_col = inverse[t0 : t0 + CHUNK_TILE, target_col : target_col + BLOCK_TILE]
                    # Unwritten lower blocks and block-upper entries are zero.
                    accum = pl.matmul(a_row, inv_col, out_dtype=pl.FP32)
                    neg_sum = pl.mul(accum, -1.0)
                    inv_ii = inverse[row0 : row0 + BLOCK_TILE, diag_col : diag_col + BLOCK_TILE]
                    inv_ij = pl.matmul(inv_ii, neg_sum, out_dtype=pl.FP32)
                    inverse[row0 : row0 + BLOCK_TILE, target_col : target_col + BLOCK_TILE] = inv_ij
    for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="solve_store"):
        t0 = c0 * CHUNK_TILE
        for hh in pl.range(H):
            col = hh * CHUNK_TILE
            inv32 = inverse[t0 : t0 + CHUNK_TILE, col : col + CHUNK_TILE]
            inv16 = pl.cast(inv32, target_type=pl.FP16, mode="rint")
            t_flat[t0 : t0 + CHUNK_TILE, col : col + CHUNK_TILE] = inv16
    return t_out


gdn_solve_tril = pl.jit.inline(_gdn_solve_tril)
gdn_solve_tril_test = pl.jit(_gdn_solve_tril)


def blk_masks(chunk: int = CHUNK_TILE, block: int = BLOCK_TILE):
    """The negated diagonal-block and lower-block masks in the shared stage interface."""
    i = torch.arange(chunk)[:, None] // block
    j = torch.arange(chunk)[None, :] // block
    return -(i == j).to(torch.float16), -(i > j).to(torch.float16)


def eye_block(chunk: int):
    """The chunk identity; its leading block seeds diagonal inversion."""
    return torch.eye(chunk, dtype=torch.float16)


_INPUTS: dict = {}


def _inputs(t: int, input_case: str = "random") -> dict:
    """Random or correlated-key validation inputs, cached by case and token extent."""
    if input_case not in ("random", "correlated"):
        raise ValueError(f"unknown solve input case: {input_case}")
    key = (t, input_case)
    if key not in _INPUTS:
        if input_case == "correlated":
            rows = torch.arange(CHUNK_TILE, dtype=REF_DTYPE)
            lag = rows[:, None] - rows[None, :]
            decay = torch.linspace(0.001, 0.08, H, dtype=REF_DTYPE)[:, None, None]
            beta = torch.linspace(0.5, 0.99, H, dtype=REF_DTYPE)[:, None, None]
            a = beta * torch.exp(-decay * lag)
            a = torch.where(lag > 0, a, 0.0).permute(1, 0, 2)
            a = a.repeat(t // CHUNK_TILE, 1, 1).to(torch.float16)
            _INPUTS[key] = {"a16": a}
        else:
            st = make_inputs(t, H, D, HG)
            g_sum = cumsum_ref(st["g"], CHUNK_TILE)
            st["a16"] = kkt_ref(st["k"], st["beta"], g_sum, CHUNK_TILE).to(torch.float16)
            _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = BENCH_SEQ, input_case: str = "random"):
    from golden import TensorSpec

    def draw(key, transform=None):
        def make():
            value = _inputs(t, input_case)[key]
            return transform(value) if transform is not None else value

        return make

    return [
        TensorSpec("a_in", [t, H, CHUNK_TILE], torch.float16,
                   init_value=draw("a16")),
        TensorSpec("eye", [CHUNK_TILE, CHUNK_TILE], torch.float16,
                   init_value=lambda: eye_block(CHUNK_TILE)),
        TensorSpec("m_diag", [CHUNK_TILE, CHUNK_TILE], torch.float16,
                   init_value=lambda: blk_masks()[0]),
        TensorSpec("m_low", [CHUNK_TILE, CHUNK_TILE], torch.float16,
                   init_value=lambda: blk_masks()[1]),
        TensorSpec("t_out", [t, H, CHUNK_TILE], torch.float16),
    ]


def golden_gdn_solve_tril(tensors):
    chunk = tensors["a_in"].shape[-1]
    tensors["t_out"].copy_(solve_tril_ref(tensors["a_in"], chunk))


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--input-case", choices=("random", "correlated"), default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()
    if args.seq_len:
        cases = [(args.input_case or "random", args.seq_len)]
    elif args.input_case:
        cases = [(args.input_case, t) for t in DEFAULT_CASES]
    else:
        cases = [("random", t) for t in DEFAULT_CASES] + [("correlated", 256)]
    if args.seq_len and args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for input_case, t in cases:
        print(f"[case] solve_tril {input_case} T={t}", flush=True)
        result = run(
            fn=gdn_solve_tril_test,
            specs=build_tensor_specs(t=t, input_case=input_case),
            golden_fn=golden_gdn_solve_tril,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compare_fn={"t_out": frob_ok()},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")
