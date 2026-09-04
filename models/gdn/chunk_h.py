# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_h: the inter-chunk state recurrence. Per chunk c, per
head, with S the state entering the chunk,

    snapshot_c = S
    V_new      = U - W @ S
    S          = exp(g_last) * S + K^T (V_new * exp(g_last - g))

It produces both inputs chunk_o consumes: the per-chunk state snapshots and
V_new. Chunks carry state, so they cannot run in parallel; work is parallel over
heads and the chunks run sequentially within a head.

The decay is rebased on the chunk's last gate, exp(g_last - g_i) <= 1, because
the algebraically equal exp(g_last) * exp(-g_i) reaches e^90 on a 128-token chunk
and overflows FP32. The state decay therefore cannot be factored out of the
contraction, and g_last is read as a scalar.
"""
import pypto.language as pl

# model config
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # value heads (= key heads; no GQA in the default build)
D = 128                 # head dimension
CHUNK = 128             # chunk size in tokens


def build_kernel(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    """The stage kernel at one shape."""
    nchunk = t // chunk                    # sequential steps per head

    @pl.jit
    def gdn_chunk_h(
        k: pl.Tensor[[t, h, d], pl.FP16],
        w: pl.Tensor[[t, h, d], pl.FP16],
        u: pl.Tensor[[t, h, d], pl.FP16],
        g_sum: pl.Tensor[[h, t], pl.FP32],
        state: pl.Out[pl.Tensor[[nchunk * h * d, d], pl.FP16]],
        v_new: pl.Out[pl.Tensor[[t, h, d], pl.FP16]],
    ):
        k_flat = pl.reshape(k, [t, h * d])
        w_flat = pl.reshape(w, [t, h * d])
        u_flat = pl.reshape(u, [t, h * d])
        v_flat = pl.reshape(v_new, [t, h * d])
        # pl.split is required, not a tuning choice: without it the kernel needs
        # 197632 B of a 188416 B vector buffer and does not build.
        for hh in pl.spmd(h, name_hint="chunk_h",
                          optimizations=[pl.cross_core_slot(slot_num=1),
                                         pl.split(pl.SplitMode.UP_DOWN)]):
            s0 = pl.full([d, d], dtype=pl.FP32, value=0.0)
            for c, (s_cur, st, vf) in pl.range(nchunk, init_values=(s0, state, v_flat)):
                t0 = c * chunk
                row = (c * h + hh) * d
                d0 = hh * d

                s16 = pl.cast(s_cur, target_type=pl.FP16, mode="rint")
                st_next = pl.assemble(st, s16, [row, 0])        # state ENTERING this chunk

                # coeff[i] = exp(g_last - g_i), decay = exp(g_last); both arguments <= 0.
                # The unary exp comes BEFORE the [1,C]->[C,1] reshape: an elementwise op
                # after that reshape loses pl.split's tracking.
                g_last = pl.read(g_sum, [hh, t0 + chunk - 1])
                g_row = g_sum[hh : hh + 1, t0 : t0 + chunk]
                zero_row = pl.full([1, chunk], dtype=pl.FP32, value=0.0)
                neg_g = pl.sub(zero_row, g_row)
                coeff = pl.reshape(pl.exp(pl.add(neg_g, g_last)), [chunk, 1])
                zero_d = pl.full([1, d], dtype=pl.FP32, value=0.0)
                decay = pl.reshape(pl.exp(pl.add(zero_d, g_last)), [d, 1])

                wc = w_flat[t0 : t0 + chunk, d0 : d0 + d]
                ws = pl.matmul(wc, s16, out_dtype=pl.FP32)
                uc = u_flat[t0 : t0 + chunk, d0 : d0 + d]
                vc = pl.sub(pl.cast(uc, target_type=pl.FP32), ws)   # V_new = U - W @ S
                vc16 = pl.cast(vc, target_type=pl.FP16, mode="rint")

                # S = exp(g_last) * S + K^T @ (coeff * V_new). The decay goes on V, not
                # on K: K^T (V c) == (K c)^T V, and under pl.split a matmul whose
                # TRANSPOSED operand was produced on the vector unit cannot be lowered
                # (ptoas: "'pto.tmov' op expects a supported tmov address-space pair").
                kc = k_flat[t0 : t0 + chunk, d0 : d0 + d]
                vs = pl.row_expand_mul(vc, coeff)
                vs16 = pl.cast(vs, target_type=pl.FP16, mode="rint")
                kv = pl.matmul(kc, vs16, a_trans=True, out_dtype=pl.FP32)
                s_next = pl.add(pl.row_expand_mul(s_cur, decay), kv)

                vf_next = pl.assemble(vf, vc16, [t0, d0])
                s_end, st_end, vf_end = pl.yield_(s_next, st_next, vf_next)
            state = st_end
            v_flat = vf_end
        return state, v_new

    return gdn_chunk_h


gdn_chunk_h = build_kernel()


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    from models.gdn import reference

    nc = t // chunk

    return [
        TensorSpec("k", [t, h, d], torch.float16, init_value=reference.lazy("chunk_h", "k", t, h, d, chunk)),
        TensorSpec("w", [t, h, d], torch.float16,
                   init_value=reference.lazy("chunk_h", "w16", t, h, d, chunk)),
        TensorSpec("u", [t, h, d], torch.float16,
                   init_value=reference.lazy("chunk_h", "u16", t, h, d, chunk)),
        TensorSpec("g_sum", [h, t], torch.float32,
                   init_value=reference.lazy("chunk_h", "g_sum", t, h, d, chunk, reference.to_hT)),
        TensorSpec("state", [nc * h * d, d], torch.float16, is_output=True),
        TensorSpec("v_new", [t, h, d], torch.float16, is_output=True),
    ]


def golden_gdn_chunk_h(tensors):
    from models.gdn import reference

    t, h, d = tensors["k"].shape
    chunk = t // (tensors["state"].shape[0] // (h * d))
    state, v_new = reference.chunk_h(tensors["k"], tensors["w"], tensors["u"],
                                     tensors["g_sum"].t(), chunk)
    tensors["state"].copy_(reference.flat_state(state))
    tensors["v_new"].copy_(v_new)


def _stats_ok(actual, expected, **_kwargs):
    """megagdn-pto's criterion for this stage (tests/utils.py: NumericalAccuracy)."""
    from models.gdn import reference

    ok, detail = reference.stats_ok(actual, expected, chunk=CHUNK)
    print(f"[stats] {detail}", flush=True)
    return ok, detail


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=gdn_chunk_h,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_chunk_h,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"state": _stats_ok, "v_new": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
