# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_o: the chunk output,

    inter = exp(g_i) * (Q @ S)
    intra = (Q @ K^T * exp(min(g_i - g_j, 0)) * causal) @ V_new
    O     = inter + intra

S is the state snapshot entering the chunk and V_new the residual-corrected
values, both from chunk_h. The causal mask includes the diagonal, unlike
scaled_dot_kkt's strictly-lower one: a token attends to itself in the output but
not in the key-key matrix.
"""
import pypto.language as pl

from models.qwen3_8_27b.config import GDN_TILING, QWEN3_8_27B

# model shape
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
CHUNK = GDN_TILING.chunk                    # chunk size in tokens, our tiling choice

# case shape
T = 8192                # tokens (single sequence, B = 1)


def build_kernel(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK,
                 hg: int = HG, inline: bool = False):
    """The stage kernel at one shape.

    `hg` is the number of QK heads, `h` the number of value heads; they differ
    under GQA. Pass `hg=h` for an ungrouped shape. `inline` makes the kernel a
    callee for `gdn_layer` rather than a program of its own.
    """
    nchunk = t // chunk                    # state snapshots, one per chunk
    grp = h // hg

    @(pl.jit.inline if inline else pl.jit)
    def gdn_chunk_o(
        q: pl.Tensor[[t, hg, d], pl.FP16],
        k: pl.Tensor[[t, hg, d], pl.FP16],
        v: pl.Tensor[[t, h, d], pl.FP16],
        state: pl.Tensor[[nchunk * h * d, d], pl.FP16],
        g_sum: pl.Tensor[[h, t], pl.FP32],
        mask: pl.Tensor[[chunk, chunk], pl.FP32],
        o_out: pl.Out[pl.Tensor[[t, h, d], pl.FP16]],
    ):
        q_flat = pl.reshape(q, [t, hg * d])
        k_flat = pl.reshape(k, [t, hg * d])
        v_flat = pl.reshape(v, [t, h * d])
        o_flat = pl.reshape(o_out, [t, h * d])
        for c0 in pl.spmd(t // chunk, name_hint="chunk_o",
                          optimizations=[pl.cross_core_slot(slot_num=1),
                                         pl.split(pl.SplitMode.UP_DOWN)]):
            t0 = c0 * chunk
            for hh in pl.range(h):
                d0 = hh * d
                # GQA: Q and K come from key head hh // grp; V, O and the state
                # snapshot are per value head.
                dg0 = (hh // grp) * d
                qc = q_flat[t0 : t0 + chunk, dg0 : dg0 + d]
                g_row = g_sum[hh : hh + 1, t0 : t0 + chunk]
                g_col = pl.reshape(g_row, [chunk, 1])
                # exp on the ROW vector, reshaped after -- NOT pl.exp(g_col). Under
                # pl.split a view op reading the per-lane slice used to fold onto the
                # sliced buffer's base, so lane 1 applied lane 0's gate. Fixed upstream
                # 2026-09-02; this spelling still builds against stock pypto and
                # generates the same code.
                eg = pl.reshape(pl.exp(g_row), [chunk, 1])

                row = c0 * (h * d) + d0
                s_blk = state[row : row + d, 0 : d]
                inter = pl.row_expand_mul(pl.matmul(qc, s_blk), eg)

                # The accumulator has to be a matmul result -- a tile the matrix unit
                # owns, private to this core group. pl.create_tensor allocates a runtime
                # TENSOR instead: one buffer shared by every chunk running in parallel,
                # which corrupts intermittently.
                kc = k_flat[t0 : t0 + chunk, dg0 : dg0 + d]
                qk = pl.matmul(qc, kc, b_trans=True)
                diff = pl.full([chunk, chunk], dtype=pl.FP32, value=0.0)
                diff = pl.row_expand_add(diff, g_col)
                diff = pl.col_expand_sub(diff, g_row)
                decay = pl.exp(pl.minimum(diff, 0.0))
                gate = pl.mul(decay, mask[:, :])
                # a matmul's result dtype follows its CONSUMER, not its operands
                gated = pl.cast(pl.mul(qk, gate), target_type=pl.FP16, mode="rint")
                vc = v_flat[t0 : t0 + chunk, d0 : d0 + d]
                intra = pl.matmul(gated, vc, out_dtype=pl.FP32)

                o_flat[t0 : t0 + chunk, d0 : d0 + d] = pl.cast(pl.add(inter, intra),
                                                               target_type=pl.FP16, mode="rint")
        return o_out

    return gdn_chunk_o


gdn_chunk_o = build_kernel()


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK,
                       hg: int = HG):
    import torch
    from golden import TensorSpec

    from models.qwen3_8_27b import reference

    nc = t // chunk

    def init_mask():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows >= cols).float()      # inclusive diagonal

    return [
        TensorSpec("q", [t, hg, d], torch.float16,
                   init_value=reference.lazy("chunk_o", "q", t, h, d, chunk, hg=hg)),
        TensorSpec("k", [t, hg, d], torch.float16,
                   init_value=reference.lazy("chunk_o", "k", t, h, d, chunk, hg=hg)),
        TensorSpec("v", [t, h, d], torch.float16,
                   init_value=reference.lazy("chunk_o", "v_new16", t, h, d, chunk, hg=hg)),
        TensorSpec("state", [nc * h * d, d], torch.float16,
                   init_value=reference.lazy("chunk_o", "state", t, h, d, chunk, reference.flat_state, hg=hg)),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=reference.lazy("chunk_o", "g_sum", t, h, d, chunk, reference.to_hT, hg=hg)),
        TensorSpec("mask", [chunk, chunk], torch.float32, init_value=init_mask),
        TensorSpec("o_out", [t, h, d], torch.float16),
    ]


def golden_gdn_chunk_o(tensors):
    from models.qwen3_8_27b import reference

    t, _, d = tensors["q"].shape          # q and k have Hg heads under GQA
    h = tensors["v"].shape[1]             # V, O and the state are per value head
    chunk = tensors["mask"].shape[0]
    state = tensors["state"].reshape(t // chunk, h, d, d)
    tensors["o_out"].copy_(reference.chunk_o(
        tensors["q"], tensors["k"], tensors["v"], state, tensors["g_sum"].t(), chunk))


def _stats_ok(actual, expected, **_kwargs):
    """megagdn-pto's criterion for this stage (tests/utils.py: NumericalAccuracy)."""
    from models.qwen3_8_27b import reference

    ok, detail = reference.stats_ok(actual, expected, chunk=CHUNK)
    print(f"[stats] {detail}", flush=True)
    return ok, detail


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run(
        fn=gdn_chunk_o,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_chunk_o,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"o_out": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
