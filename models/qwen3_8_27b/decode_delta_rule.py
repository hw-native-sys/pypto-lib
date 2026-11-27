# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet decode: the single-token recurrence, per value head, with the
state `S` in FP32 and the value axis leading,

    S      <- S * exp(g)                    # g is per head: a scalar
    kv_mem <- sum_k S[v, k] * k[k]
    delta  <- (value - kv_mem) * beta
    S[v,k] <- S[v, k] + delta[v] * k[k]
    out    <- sum_k S[v, k] * q[k]

q and k are L2-normalised in FP32 with `eps = 1e-6` and q carries the `D ** -0.5`
scale, both inside the kernel, matching `use_qk_l2norm_in_kernel=True`. GQA: value
head `h` reads QK head `h // GRP`.

**The state's last two axes are `[V, K]`, not `[K, V]`.** That puts the `sum_k`
reduction along the contiguous axis, and it is the layout the AscendC operator's ABI
fixes. `D` is 128 on both axes, so **no shape check can ever catch a transposed state**
-- it surfaces as wrong output from the first decode step onward.
`reference.recurrent_delta_rule` and `chunk_h`'s `final_state` are both `[K, V]`, so the
transpose belongs at that boundary.

One token per state row is assumed, which is what a non-speculative decode step
produces. Two tokens sharing a row would be a sequential dependency, and the tasks here
are independent.

`out` is computed as a vector reduction rather than a matmul; the column it produces is
turned back into a row by broadcasting across `COL_REP` columns, transposing, and taking
row 0. Each task walks `HEAD_TILE` head-states, and the inner loop is `pl.unroll`
because the trip count is a compile-time constant.

`V_TILE` is 64: a whole head's `[128, 128]` FP32 state is 64 KiB, and the live set at
`V_TILE = 128` needs 280064 bytes of a 188416-byte vector buffer.

The gate is exponentiated and q and k normalised in a prologue pass rather than in the
main task. `exp` has no scalar form, and reducing a single `[1, 128]` row inside the main
task would make a `[1, 1]` FP32 result -- four bytes, which is not a legal tile column.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                     # decode tokens, one per sequence
DSTATE_DYN = pl.dynamic("DSTATE_DYN")           # rows of the state pool

# model config
H = QWEN3_8_27B.linear_num_value_heads          # value heads: v, beta, g and the state
HG = QWEN3_8_27B.linear_num_key_heads           # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim           # head dimension, both state axes
GRP = H // HG                                   # value heads sharing one key head
L2_EPS = 1e-6                                   # inside the sqrt, not a clamp
Q_SCALE = D ** -0.5

# validation
# one token, then several; a token is one state row of the pool.
DEFAULT_CASES = (1, 5)

# tiling
V_TILE = 64             # value rows of one head's state per inner step
HEAD_TILE = 8           # head-states one task walks; see the table above
COL_REP = 16            # replication width that makes a row-to-column turn legal

LANES = D // V_TILE                     # inner steps per head-state
# Arithmetic on module constants inside a jit body is parsed into an IR expression,
# not folded to a Python number, and `pl.full` takes a number -- so the reciprocal
# is formed here. 1/16 is exact in binary, so the round trip is lossless.
COL_REP_INV = 1.0 / COL_REP


# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True


def build_kernel(head_tile: int = HEAD_TILE):
    """The decode recurrence at a given `HEAD_TILE`; returns `(inline, jit)` forms."""
    if H % head_tile:
        raise ValueError(f"HEAD_TILE={head_tile} must divide H={H}")
    inner = head_tile * LANES
    head_tasks = H // head_tile

    def _gdn_decode_delta_rule(
        q: pl.Tensor[[T_DYN, HG, D], pl.BF16],
        k: pl.Tensor[[T_DYN, HG, D], pl.BF16],
        v: pl.Tensor[[T_DYN, H, D], pl.BF16],
        beta: pl.Tensor[[T_DYN, H], pl.FP32],
        g: pl.Tensor[[T_DYN, H], pl.FP32],
        state: pl.InOut[pl.Tensor[[DSTATE_DYN, H, D, D], pl.FP32]],
        state_rows: pl.Tensor[[T_DYN], pl.INT32],
        o: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
    ):
        """`state_rows` must name a distinct pool row per token; the tasks are independent."""
        q.bind_dynamic(0, T_DYN)
        k.bind_dynamic(0, T_DYN)
        v.bind_dynamic(0, T_DYN)
        beta.bind_dynamic(0, T_DYN)
        g.bind_dynamic(0, T_DYN)
        state.bind_dynamic(0, DSTATE_DYN)
        state_rows.bind_dynamic(0, T_DYN)
        o.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(q, 0)

        qk_rows = t_dim * HG
        q_rows = pl.reshape(q, [qk_rows, D])
        k_rows = pl.reshape(k, [qk_rows, D])
        qn_gm = pl.create_tensor([qk_rows, D], dtype=pl.FP32)
        kn_gm = pl.create_tensor([qk_rows, D], dtype=pl.FP32)
        decay_gm = pl.create_tensor([t_dim, H], dtype=pl.FP32)
        # One task per token: a token's HG q rows and HG k rows are contiguous, so each
        # norm is a [HG, D] tile, and the token's whole gate row is [1, H].
        with pl.spmd(t_dim, name_hint="decode_qk_norm",
                     allow_early_resolve=EARLY_RESOLVE) as norm_tid:
            tn = pl.tile.get_block_idx()
            r0 = tn * HG
            qt = pl.cast(q_rows[r0 : r0 + HG, 0:D], target_type=pl.FP32)
            kt = pl.cast(k_rows[r0 : r0 + HG, 0:D], target_type=pl.FP32)
            # The eps sits inside the square root; transformers adds it rather than
            # clamping, and at this model's q row norms the two differ by about 1%.
            q_inv = pl.rsqrt(pl.add(pl.row_sum(pl.mul(qt, qt)), L2_EPS), high_precision=True)
            k_inv = pl.rsqrt(pl.add(pl.row_sum(pl.mul(kt, kt)), L2_EPS), high_precision=True)
            qn_gm[r0 : r0 + HG, 0:D] = pl.mul(pl.row_expand_mul(qt, q_inv), Q_SCALE)
            kn_gm[r0 : r0 + HG, 0:D] = pl.row_expand_mul(kt, k_inv)
            # g holds the log decay; the recurrence wants exp(g), and there is no
            # scalar exp.
            decay_gm[tn : tn + 1, 0:H] = pl.exp(g[tn : tn + 1, 0:H])

        # A store whose source is a tile reshaped to two leading unit axes lowers to a
        # tile with a single-element row, which ptoas rejects on its 32-byte row
        # alignment. Addressing the pool through a flat view of the same buffer keeps
        # the stored tile the shape the arithmetic already produced.
        state_2d = pl.reshape(
            state, [pl.tensor.dim(state, 0) * H * D, D])
        out_rows = pl.reshape(o, [t_dim, H * D])
        with pl.spmd(t_dim * head_tasks, name_hint="decode_delta_rule", deps=[norm_tid],
                     allow_early_resolve=EARLY_RESOLVE):
            task = pl.tile.get_block_idx()
            t = task // head_tasks
            h0 = (task % head_tasks) * head_tile
            row = pl.cast(pl.read(state_rows, [t]), pl.INDEX)
            # Loop-invariant constant tiles, materialised once per task rather than
            # once per head-state.
            spread = pl.full([COL_REP, V_TILE], dtype=pl.FP32, value=COL_REP_INV)
            wide = pl.full([V_TILE, COL_REP], dtype=pl.FP32, value=1.0)
            ones = pl.full([V_TILE, D], dtype=pl.FP32, value=1.0)
            for i in pl.unroll(inner):
                h = h0 + i // LANES
                v0 = (i % LANES) * V_TILE
                qk_row = t * HG + h // GRP
                q_row = qn_gm[qk_row : qk_row + 1, 0:D]
                k_row = kn_gm[qk_row : qk_row + 1, 0:D]
                decay_s = pl.read(decay_gm, [t, h])
                beta_s = pl.read(beta, [t, h])

                # This inner step owns value rows [v0, v0 + V_TILE) of one head's state.
                srow = (row * H + h) * D + v0
                state_tile = pl.slice(state_2d, [V_TILE, D], [srow, 0])
                # GDN's gate is one scalar per head, where KDA's is a row over the key
                # channels; the decay is a scalar multiply, not a column broadcast.
                decayed = pl.mul(state_tile, decay_s)
                # kv_mem is built as a column by a reduction, which is the shape the
                # delta and the rank-1 update both want.
                kv_col = pl.row_sum(pl.col_expand_mul(decayed, k_row))      # [V_TILE, 1]
                v_row = pl.cast(pl.reshape(
                    pl.slice(v, [1, 1, V_TILE], [t, h, v0], drop_dims=[0]),
                    [1, V_TILE]), target_type=pl.FP32)
                # value arrives as a row and is needed as a column. Transposing a row
                # into a column is not expressible -- the result is row-major with a
                # four-byte row -- but a column is legal when a reduction produces it,
                # so the row is broadcast down COL_REP rows, transposed the legal way
                # round and summed. The 1/COL_REP rides on `spread`, where it is exact.
                v_col = pl.row_sum(pl.transpose(pl.col_expand_mul(spread, v_row), 0, 1))
                delta_col = pl.mul(pl.sub(v_col, kv_col), beta_s)           # [V_TILE, 1]
                # The rank-1 update delta (x) k, from the two broadcasts: a width-one
                # cube contraction is not addressable.
                updated = pl.add(decayed, pl.col_expand_mul(
                    pl.row_expand_mul(ones, delta_col), k_row))
                state_2d[srow : srow + V_TILE, 0:D] = updated

                # The output is the same reduction as kv_mem against q, and the column
                # it gives is turned back into a row by the reverse of the v_col turn:
                # COL_REP copies transpose into rows, and row 0 is the answer.
                out_col = pl.row_sum(pl.col_expand_mul(updated, q_row))
                rowed = pl.transpose(pl.row_expand_mul(wide, out_col), 0, 1)
                ocol = h * D + v0
                out_rows[t : t + 1, ocol : ocol + V_TILE] = pl.cast(
                    pl.slice(rowed, [1, V_TILE], [0, 0]), target_type=pl.FP16, mode="rint")
        return o, state

    return pl.jit.inline(_gdn_decode_delta_rule), pl.jit(_gdn_decode_delta_rule)


gdn_decode_delta_rule, gdn_decode_delta_rule_test = build_kernel()


def golden_state_from_kv(initial_state):
    """`[H, K, V]`, the layout the oracle and `chunk_h` use, to this kernel's `[H, V, K]`."""
    return initial_state.transpose(-1, -2).contiguous()


def golden_state_to_kv(state):
    """This kernel's `[..., V, K]` back to the oracle's `[..., K, V]`."""
    return state.transpose(-1, -2).contiguous()


def golden_step(q, k, v, beta, g, state, rows):
    """One decode launch, in float64, against a `[rows, H, V, K]` pool.

    `q` and `k` are raw: the L2 norm and the `D ** -0.5` scale are the kernel's, so
    they are applied here too. Returns `(o, state)` with `state` updated in place for
    the rows `rows` names and untouched elsewhere.
    """
    import torch

    import reference

    tokens = q.shape[0]
    s = state.double().clone()
    out = torch.zeros(tokens, H, D, dtype=torch.float64)
    for t in range(tokens):
        r = int(rows[t])
        qn = reference.l2norm(q[t], eps=L2_EPS) * Q_SCALE        # [HG, D], float64
        kn = reference.l2norm(k[t], eps=L2_EPS)
        for h in range(H):
            kh = h // GRP
            st = s[r, h] * torch.exp(g[t, h].double())           # [V, K]
            kv = (st * kn[kh].unsqueeze(0)).sum(dim=-1)          # [V]
            delta = (v[t, h].double() - kv) * beta[t, h].double()
            st = st + delta.unsqueeze(-1) * kn[kh].unsqueeze(0)
            s[r, h] = st
            out[t, h] = (st * qn[kh].unsqueeze(0)).sum(dim=-1)
    return out, s


def golden_gdn_decode_delta_rule(tensors):
    out, s = golden_step(tensors["q"], tensors["k"], tensors["v"], tensors["beta"],
                         tensors["g"], tensors["state"], tensors["state_rows"])
    tensors["o"].copy_(out.to(tensors["o"].dtype))
    tensors["state"].copy_(s.to(tensors["state"].dtype))


POOL_ROWS = 6


def build_tensor_specs(tokens: int = 5, pool_rows: int = POOL_ROWS):
    import torch
    from golden import TensorSpec

    bf, f16, f32, i32 = torch.bfloat16, torch.float16, torch.float32, torch.int32

    def gen(offset):
        return torch.Generator().manual_seed(42 + offset)

    # Distinct rows, and deliberately not the identity mapping: a kernel that ignored
    # state_rows and used the token index would pass an identity fixture. Distinctness
    # is the kernel's contract and this test cannot prove it -- at these sizes the
    # blocks happen to run in task-index order, which is token order, so a fixture
    # that mapped two tokens to one row would agree with a sequential golden by
    # accident. The caller owes the guarantee; the fixture only avoids relying on it.
    rows = torch.randperm(pool_rows, generator=gen(7))[:tokens].to(i32)
    if len(set(rows.tolist())) != tokens:
        raise ValueError("state_rows must name a distinct pool row per token")

    return [
        # q and k are raw here -- the conv output, before the norm this kernel does.
        TensorSpec("q", [tokens, HG, D], bf,
                   init_value=lambda: torch.randn(tokens, HG, D, generator=gen(1)).to(bf)),
        TensorSpec("k", [tokens, HG, D], bf,
                   init_value=lambda: torch.randn(tokens, HG, D, generator=gen(2)).to(bf)),
        TensorSpec("v", [tokens, H, D], bf,
                   init_value=lambda: torch.randn(tokens, H, D, generator=gen(3)).to(bf)),
        TensorSpec("beta", [tokens, H], f32,
                   init_value=lambda: torch.rand(tokens, H, generator=gen(4))),
        # The model's own gate distribution: log-sigmoid, so g <= 0 and exp(g) spans
        # the decay range. A near-zero fixture would make exp(g) ~ 1 and hide it.
        TensorSpec("g", [tokens, H], f32,
                   init_value=lambda: torch.nn.functional.logsigmoid(
                       torch.randn(tokens, H, generator=gen(5)))),
        # A non-zero initial state, so a kernel that silently starts from zero fails,
        # at the magnitude a real rollout reaches rather than unit noise.
        TensorSpec("state", [pool_rows, H, D, D], f32,
                   init_value=lambda: torch.randn(pool_rows, H, D, D, generator=gen(6)) * 0.3),
        TensorSpec("state_rows", [tokens], i32, init_value=lambda: rows),
        TensorSpec("o", [tokens, H, D], f16),
    ]


if __name__ == "__main__":
    import argparse

    import torch
    from golden import run

    from utils import rel_frobenius

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--pool-rows", type=int, default=POOL_ROWS)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.tokens] if args.tokens else list(DEFAULT_CASES)
    if max(cases) > args.pool_rows:
        parser.error(f"--tokens {max(cases)} must not exceed --pool-rows {args.pool_rows}; "
                     "one token per state row")
    torch.set_num_threads(min(torch.get_num_threads(), 32))
    torch.backends.mkldnn.enabled = False

    for tokens in cases:
        print(f"[case] decode_delta_rule tokens={tokens}", flush=True)
        result = run(
            fn=gdn_decode_delta_rule_test,
            specs=build_tensor_specs(tokens=tokens, pool_rows=args.pool_rows),
            golden_fn=golden_gdn_decode_delta_rule,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            compile_only=args.compile_only,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=2e-3,
            atol=2e-3,
            # The state is FP32 and feeds itself, so it is held to the dtype rather than to
            # the FP16 output's tolerance.
            compare_fn={"state": rel_frobenius(2e-6, "state"), "o": rel_frobenius(2e-3, "o")},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at tokens={tokens}")
