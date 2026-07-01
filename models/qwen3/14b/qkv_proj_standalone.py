# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone Q/K/V projection for Qwen3-14B — BATCH=1, V can be Cube or AIV (VECTOR row_sum).

    normed_in [BATCH=1, HIDDEN=5120] (BF16)
    q_proj = normed_in @ wq  — always Cube (SPMD split-K)
    k_proj = normed_in @ wk  — Cube or AIV row_sum  (K_PROJ_ON_AIV=1)
    v_proj = normed_in @ wv  — Cube or AIV row_sum  (V_PROJ_ON_AIV=1)

Usage::

    # All cube (default)
    python qkv_proj_standalone.py -p a2a3 -d 0

    # V on AIV (row_sum via VECTOR unit)
    V_PROJ_ON_AIV=1 python qkv_proj_standalone.py -p a2a3 -d 0

    # K and V both on AIV
    K_PROJ_ON_AIV=1 V_PROJ_ON_AIV=1 python qkv_proj_standalone.py -p a2a3 -d 0

    # Compile-only smoke
    python qkv_proj_standalone.py --smoke
"""

import argparse
import os
from pathlib import Path

import pypto.language as pl
import torch
from pypto.backend import BackendType, set_backend_type
from pypto.runtime import RunConfig

# ══════════════════════════════════════════════════════════════════════════════
# Model architecture (Qwen3-14B, fixed).
# ══════════════════════════════════════════════════════════════════════════════
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM       # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM # 1024

# ══════════════════════════════════════════════════════════════════════════════
# Workload — BATCH=1 for decode (single token, single request).
# ══════════════════════════════════════════════════════════════════════════════
BATCH = 1

# ══════════════════════════════════════════════════════════════════════════════
# Cube matmul tiling (shared by Q / K / V cube paths).
# ══════════════════════════════════════════════════════════════════════════════
# TM is NOT asserted == BATCH here (BATCH=1 but TM stays 16; ISA promotes
# m=1 → m=16 on A2A3 so the hardware always computes a 16-row tile).
TN = 128           # inner N sub-tile
TK = 128           # inner K chunk
QKV_N_TILE = 512   # outer N-tile = N_SUB inner TN subtiles
N_SUB = QKV_N_TILE // TN              # 4
Q_ON = HIDDEN // QKV_N_TILE           # 10 (Q: 5120/512)
KV_ON = KV_HIDDEN // QKV_N_TILE       # 2  (K,V: 1024/512)
QKV_OK = 5                            # split-K slices (atomic-add)
QKV_K_SLICE = HIDDEN // QKV_OK        # 1024 K per split
QKV_K_CHUNKS = QKV_K_SLICE // TK      # 8 inner TK chunks per split
# Cube matmul requires M-dimension >= 16 (ISA promotes m=1→16 but ptoas
# codegen asserts dstRow%16==0). Pad BATCH to the minimum for the cube paths;
# the AIV paths use separate SAFE_BATCH=8 for blayout-safe accumulator shapes.
PAD_BATCH = max(BATCH, 16)  # 16

# ══════════════════════════════════════════════════════════════════════════════
# AIV (VECTOR) row_sum tiling — V / K projections.
# BATCH=1 allows col_expand + mul + row_sum + reshape without per-column
# loop and without SAFE_BATCH padding.
#
# NV=16 is the minimum: rs_wblk [KC, NV] is cast from BF16 to FP32, and the
# source BF16 tile must satisfy PTO's row-major row-byte alignment (>= 32
# bytes).  NV*sizeof(BF16) = NV*2 >= 32 → NV >= 16.  Smaller NV values
# (8/4/2/1) all trigger `pto.alloc_tile` alignment errors.
#
# KC=1024 is the maximum that fits UB: the FP32 weight transposed tile
# [NV, KC] = [16, 1024] occupies 64 KB, exactly at the A2/A3 VEC buffer
# limit.  KC=2048 overflows (270 KB > 188 KB).
V_RS_NV = 16
V_RS_NTILES = KV_HIDDEN // V_RS_NV    # 8
V_RS_KC = 1024

K_RS_NV = 16
K_RS_NTILES = KV_HIDDEN // K_RS_NV    # 8
K_RS_KC = 1024

# ══════════════════════════════════════════════════════════════════════════════
# Env toggles — resolved at import (trace) time so the JIT picks one branch.
# ══════════════════════════════════════════════════════════════════════════════
_K_PROJ_ON_AIV = os.environ.get("K_PROJ_ON_AIV", "0") == "1"
_V_PROJ_ON_AIV = os.environ.get("V_PROJ_ON_AIV", "0") == "1"


@pl.jit
def qkv_proj(
    normed_in: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    q_proj: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.FP32]],
    k_proj: pl.Out[pl.Tensor[[BATCH, KV_HIDDEN], pl.FP32]],
    v_proj: pl.Out[pl.Tensor[[BATCH, KV_HIDDEN], pl.FP32]],
):
    # ── Pad normed_in for cube alignment (cube matmul requires M >= 16). ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="pad_normed") as pad_tid:
        padded_normed = pl.full([PAD_BATCH, HIDDEN], dtype=pl.BF16, value=0.0)
        padded_normed = pl.assemble(padded_normed, normed_in, [0, 0])

    # Intermediate padded accumulators and SPMD TaskId arrays — allocated
    # BEFORE manual_scope. Inside manual_scope the SPMD blocks fill the arrays
    # and accumulate; slice tasks gate on them for proper ordering.
    q_padded = pl.create_tensor([PAD_BATCH, HIDDEN], dtype=pl.FP32)
    k_padded = pl.create_tensor([PAD_BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_padded = pl.create_tensor([PAD_BATCH, KV_HIDDEN], dtype=pl.FP32)
    q_tids = pl.array.create(Q_ON * QKV_OK, pl.TASK_ID)
    k_tids = pl.array.create(KV_ON * QKV_OK, pl.TASK_ID)
    v_tids = pl.array.create(KV_ON * QKV_OK, pl.TASK_ID)

    with pl.manual_scope():
        # ── Q projection — always Cube, SPLIT-K + atomic-add. ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_seed", deps=[pad_tid]) as q_seed_tid:
            for snb in pl.pipeline(Q_ON, stage=2):
                q_padded = pl.assemble(
                    q_padded,
                    pl.full([PAD_BATCH, QKV_N_TILE], dtype=pl.FP32, value=0.0),
                    [0, snb * QKV_N_TILE],
                )
        for q_nt in pl.parallel(Q_ON):
            q_n_region = q_nt * QKV_N_TILE
            for q_ks in pl.range(QKV_OK):
                q_k_base = q_ks * QKV_K_SLICE
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="q_proj",
                    deps=[q_seed_tid],
                ) as q_tid:
                    for n_sub in pl.range(N_SUB):
                        n0 = q_n_region + n_sub * TN
                        q_acc = pl.matmul(
                            padded_normed[:, q_k_base : q_k_base + TK],
                            wq[q_k_base : q_k_base + TK, n0 : n0 + TN],
                            out_dtype=pl.FP32,
                        )
                        for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                            kk = q_k_base + kc * TK
                            q_acc = pl.matmul_acc(
                                q_acc,
                                padded_normed[:, kk : kk + TK],
                                wq[kk : kk + TK, n0 : n0 + TN],
                            )
                        q_padded = pl.assemble(q_padded, q_acc, [0, n0], atomic=pl.AtomicType.Add)
                q_tids[q_nt * QKV_OK + q_ks] = q_tid
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_slice",
                   deps=[q_tids[i] for i in range(Q_ON * QKV_OK)]):
            q_proj = pl.assemble(q_proj, pl.slice(q_padded, [BATCH, HIDDEN], [0, 0]), [0, 0])

        # ── K projection — Cube or AIV row_sum. ──
        if _K_PROJ_ON_AIV:
            for ks_vn in pl.parallel(K_RS_NTILES):
                ks_n0 = ks_vn * K_RS_NV
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj_vec_rs") as ks_tid:
                    ks_acc = pl.full([BATCH, K_RS_NV], dtype=pl.FP32, value=0.0)
                    for ks_kb in pl.pipeline(HIDDEN // K_RS_KC, stage=1):
                        ks_k0 = ks_kb * K_RS_KC
                        ks_nblk = pl.cast(
                            normed_in[:, ks_k0 : ks_k0 + K_RS_KC], target_type=pl.FP32
                        )  # [1, KC]
                        ks_wblk = pl.cast(
                            wk[ks_k0 : ks_k0 + K_RS_KC, ks_n0 : ks_n0 + K_RS_NV],
                            target_type=pl.FP32,
                        )  # [KC, NV]
                        ks_wblk_t = pl.transpose(ks_wblk, 0, 1)  # [NV, KC]
                        ks_exp = pl.col_expand(ks_wblk_t, ks_nblk)  # [NV, KC]
                        ks_prod = pl.mul(ks_exp, ks_wblk_t)  # [NV, KC]
                        ks_partial = pl.row_sum(ks_prod)  # [NV, 1]
                        ks_partial_r = pl.reshape(ks_partial, [1, K_RS_NV])  # [1, NV]
                        ks_acc = pl.add(ks_acc, ks_partial_r)
                    k_proj = pl.assemble(k_proj, ks_acc, [0, ks_n0])
        else:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_seed", deps=[pad_tid]) as k_seed_tid:
                k_padded = pl.assemble(
                    k_padded,
                    pl.full([PAD_BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0),
                    [0, 0],
                )
            for k_nt in pl.parallel(KV_ON):
                k_n_region = k_nt * QKV_N_TILE
                for k_ks in pl.range(QKV_OK):
                    k_k_base = k_ks * QKV_K_SLICE
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="k_proj",
                        deps=[k_seed_tid],
                    ) as k_tid:
                        for n_sub in pl.range(N_SUB):
                            n0 = k_n_region + n_sub * TN
                            k_acc = pl.matmul(
                                padded_normed[:, k_k_base : k_k_base + TK],
                                wk[k_k_base : k_k_base + TK, n0 : n0 + TN],
                                out_dtype=pl.FP32,
                            )
                            for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                                kk = k_k_base + kc * TK
                                k_acc = pl.matmul_acc(
                                    k_acc,
                                    padded_normed[:, kk : kk + TK],
                                    wk[kk : kk + TK, n0 : n0 + TN],
                                )
                            k_padded = pl.assemble(k_padded, k_acc, [0, n0], atomic=pl.AtomicType.Add)
                    k_tids[k_nt * QKV_OK + k_ks] = k_tid
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_slice",
                       deps=[k_tids[i] for i in range(KV_ON * QKV_OK)]):
                k_proj = pl.assemble(k_proj, pl.slice(k_padded, [BATCH, KV_HIDDEN], [0, 0]), [0, 0])

        # ── V projection — Cube or AIV row_sum. ──
        if _V_PROJ_ON_AIV:
            for rs_vn in pl.parallel(V_RS_NTILES):
                rs_n0 = rs_vn * V_RS_NV
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj_vec_rs") as rs_tid:
                    rs_acc = pl.full([BATCH, V_RS_NV], dtype=pl.FP32, value=0.0)
                    for rs_kb in pl.pipeline(HIDDEN // V_RS_KC, stage=1):
                        rs_k0 = rs_kb * V_RS_KC
                        rs_nblk = pl.cast(
                            normed_in[:, rs_k0 : rs_k0 + V_RS_KC], target_type=pl.FP32
                        )  # [1, KC]
                        rs_wblk = pl.cast(
                            wv[rs_k0 : rs_k0 + V_RS_KC, rs_n0 : rs_n0 + V_RS_NV],
                            target_type=pl.FP32,
                        )  # [KC, NV]
                        rs_wblk_t = pl.transpose(rs_wblk, 0, 1)  # [NV, KC]
                        rs_exp = pl.col_expand(rs_wblk_t, rs_nblk)  # [NV, KC]
                        rs_prod = pl.mul(rs_exp, rs_wblk_t)  # [NV, KC]
                        rs_partial = pl.row_sum(rs_prod)  # [NV, 1]
                        rs_partial_r = pl.reshape(rs_partial, [1, V_RS_NV])  # [1, NV]
                        rs_acc = pl.add(rs_acc, rs_partial_r)
                    v_proj = pl.assemble(v_proj, rs_acc, [0, rs_n0])
        else:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_seed", deps=[pad_tid]) as v_seed_tid:
                v_padded = pl.assemble(
                    v_padded,
                    pl.full([PAD_BATCH, KV_HIDDEN], dtype=pl.FP32, value=0.0),
                    [0, 0],
                )
            for v_nt in pl.parallel(KV_ON):
                v_n_region = v_nt * QKV_N_TILE
                for v_ks in pl.range(QKV_OK):
                    v_k_base = v_ks * QKV_K_SLICE
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="v_proj",
                        deps=[v_seed_tid],
                    ) as v_tid:
                        for n_sub in pl.range(N_SUB):
                            n0 = v_n_region + n_sub * TN
                            v_acc = pl.matmul(
                                padded_normed[:, v_k_base : v_k_base + TK],
                                wv[v_k_base : v_k_base + TK, n0 : n0 + TN],
                                out_dtype=pl.FP32,
                            )
                            for kc in pl.pipeline(1, QKV_K_CHUNKS, stage=2):
                                kk = v_k_base + kc * TK
                                v_acc = pl.matmul_acc(
                                    v_acc,
                                    padded_normed[:, kk : kk + TK],
                                    wv[kk : kk + TK, n0 : n0 + TN],
                                )
                            v_padded = pl.assemble(v_padded, v_acc, [0, n0], atomic=pl.AtomicType.Add)
                    v_tids[v_nt * QKV_OK + v_ks] = v_tid
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_slice",
                       deps=[v_tids[i] for i in range(KV_ON * QKV_OK)]):
                v_proj = pl.assemble(v_proj, pl.slice(v_padded, [BATCH, KV_HIDDEN], [0, 0]), [0, 0])


def build_tensor_specs():
    from golden import TensorSpec

    def rn(shape, dtype):
        return torch.randn(shape, dtype=dtype)

    return [
        TensorSpec("normed_in", [BATCH, HIDDEN], torch.bfloat16, init_value=rn([BATCH, HIDDEN], torch.bfloat16)),
        TensorSpec("wq", [HIDDEN, HIDDEN], torch.bfloat16, init_value=rn([HIDDEN, HIDDEN], torch.bfloat16) * 0.02),
        TensorSpec("wk", [HIDDEN, KV_HIDDEN], torch.bfloat16, init_value=rn([HIDDEN, KV_HIDDEN], torch.bfloat16) * 0.02),
        TensorSpec("wv", [HIDDEN, KV_HIDDEN], torch.bfloat16, init_value=rn([HIDDEN, KV_HIDDEN], torch.bfloat16) * 0.02),
        TensorSpec("q_proj", [BATCH, HIDDEN], torch.float32, is_output=True),
        TensorSpec("k_proj", [BATCH, KV_HIDDEN], torch.float32, is_output=True),
        TensorSpec("v_proj", [BATCH, KV_HIDDEN], torch.float32, is_output=True),
    ]


def golden_qkv_proj(tensors):
    normed = tensors["normed_in"].float()
    tensors["q_proj"][:] = normed @ tensors["wq"].float()
    tensors["k_proj"][:] = normed @ tensors["wk"].float()
    tensors["v_proj"][:] = normed @ tensors["wv"].float()


def _backend_type(platform: str) -> BackendType:
    return BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="compile-only (no device)")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    set_backend_type(_backend_type(args.platform))

    if args.smoke:
        smoke_out = [torch.empty([BATCH, HIDDEN], dtype=torch.float32),
                     torch.empty([BATCH, KV_HIDDEN], dtype=torch.float32),
                     torch.empty([BATCH, KV_HIDDEN], dtype=torch.float32)]
        smoke_in = [torch.randn([BATCH, HIDDEN], dtype=torch.bfloat16),
                    torch.randn([HIDDEN, HIDDEN], dtype=torch.bfloat16) * 0.02,
                    torch.randn([HIDDEN, KV_HIDDEN], dtype=torch.bfloat16) * 0.02,
                    torch.randn([HIDDEN, KV_HIDDEN], dtype=torch.bfloat16) * 0.02]
        qkv_proj.compile_for_test(*smoke_in, *smoke_out)
        raise SystemExit(0)

    from golden import run_jit, ratio_allclose

    result = run_jit(
        fn=qkv_proj,
        specs=build_tensor_specs(),
        golden_fn=golden_qkv_proj,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=3e-3,
        atol=3e-3,
        compare_fn={
            "q_proj": ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02),
            "k_proj": ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02),
            "v_proj": ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
