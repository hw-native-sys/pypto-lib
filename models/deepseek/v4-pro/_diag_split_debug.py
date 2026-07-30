"""Debug AIV-only / AIC-only split without ExpandMixed."""
from __future__ import annotations

import torch

from _diag_prefill_qr_proj_instr import (
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    Q_LORA,
    Q_OUT_TILE,
    Q_TILE,
    T,
    _QR_KS,
    _WQ_B_SCALE_ROWS,
    _acc_stats,
    _shared_inputs,
    aic_matmul_from_gm,
    aiv_quant_store,
)
from golden import TensorSpec, run_jit
from mx_quant_common import ATOL_RTOL

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    data = _shared_inputs()
    tol = ATOL_RTOL["indexer_fp8"]
    cfg = dict(platform=args.platform, device_id=args.device)
    captured: dict = {}

    def golden_aiv(t):
        t["out_scale_nd"][:] = data["xs"]
        t["out_act_fp8"][:] = data["xq"]

    def cmp(name):
        def f(actual, expected, **_kw):
            captured[name] = (actual.cpu().clone(), expected.cpu().clone())
            return True, ""

        f.__name__ = name
        return f

    print("=== AIV-only quant/store ===")
    run_jit(
        fn=aiv_quant_store,
        specs=[
            TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: data["qr"]),
            TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: data["qr_scale"]),
            TensorSpec("out_scale_nd", [T, _QR_KS], torch.float8_e8m0fnu, is_output=True),
            TensorSpec("out_act_fp8", [T, Q_TILE], torch.float8_e4m3fn, is_output=True),
        ],
        golden_fn=golden_aiv,
        runtime_cfg=cfg,
        rtol=0.0,
        atol=0.0,
        compare_fn={"out_scale_nd": cmp("s"), "out_act_fp8": cmp("a")},
    )
    sa, se = captured["s"]
    aa, ae = captured["a"]
    su, eu = sa.view(torch.uint8), se.view(torch.uint8)
    au, eu2 = aa.view(torch.uint8), ae.view(torch.uint8)
    print(
        f"scale nz={int((su != 0).sum())} exp_nz={int((eu != 0).sum())} "
        f"eq={int((su == eu).sum())}/{su.numel()}"
    )
    print(
        f"act nz={int((au != 0).sum())} exp_nz={int((eu2 != 0).sum())} "
        f"eq={int((au == eu2).sum())}/{au.numel()}"
    )
    print("scale sample", su.reshape(-1)[:8].tolist(), "exp", eu.reshape(-1)[:8].tolist())
    print("act sample", au[0, :8].tolist(), "exp", eu2[0, :8].tolist())

    print("=== AIC-only with HOST act/scale ===")
    captured2: dict = {}

    def golden_aic(t):
        t["out_acc_k0"][:] = data["acc"]

    def cmp_aic(actual, expected, **_kw):
        captured2["acc"] = (actual.cpu().clone(), expected.cpu().clone())
        return True, ""

    cmp_aic.__name__ = "acc"
    run_jit(
        fn=aic_matmul_from_gm,
        specs=[
            TensorSpec("act_fp8", [T, Q_TILE], torch.float8_e4m3fn, init_value=lambda: data["xq"]),
            TensorSpec(
                "scale_nd", [T, _QR_KS], torch.float8_e8m0fnu, init_value=lambda: data["scale_nd"]
            ),
            TensorSpec(
                "wq_b",
                [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM],
                torch.float8_e4m3fn,
                init_value=lambda: data["wq_b"],
            ),
            TensorSpec(
                "wq_b_scale",
                [_WQ_B_SCALE_ROWS, Q_OUT_TILE],
                torch.float8_e8m0fnu,
                init_value=lambda: data["wq_b_scale"],
            ),
            TensorSpec("out_acc_k0", [T, Q_OUT_TILE], torch.float32, is_output=True),
        ],
        golden_fn=golden_aic,
        runtime_cfg=cfg,
        rtol=tol["rtol"],
        atol=tol["atol"],
        compare_fn={"out_acc_k0": cmp_aic},
    )
    bad, zero, ratio = _acc_stats(captured2["acc"][0], captured2["acc"][1], tol)
    print(f"AIC-only host-fed: bad={bad}/{captured2['acc'][0].numel()} zero_act={zero} ratio={ratio:.4f}")

    if int((su != 0).sum()) > 0:
        print("=== AIC-only with DEVICE act/scale from AIV ===")
        captured3: dict = {}

        def cmp3(actual, expected, **_kw):
            captured3["acc"] = (actual.cpu().clone(), expected.cpu().clone())
            return True, ""

        cmp3.__name__ = "acc_dev"
        run_jit(
            fn=aic_matmul_from_gm,
            specs=[
                TensorSpec("act_fp8", [T, Q_TILE], torch.float8_e4m3fn, init_value=lambda: aa),
                TensorSpec("scale_nd", [T, _QR_KS], torch.float8_e8m0fnu, init_value=lambda: sa),
                TensorSpec(
                    "wq_b",
                    [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM],
                    torch.float8_e4m3fn,
                    init_value=lambda: data["wq_b"],
                ),
                TensorSpec(
                    "wq_b_scale",
                    [_WQ_B_SCALE_ROWS, Q_OUT_TILE],
                    torch.float8_e8m0fnu,
                    init_value=lambda: data["wq_b_scale"],
                ),
                TensorSpec("out_acc_k0", [T, Q_OUT_TILE], torch.float32, is_output=True),
            ],
            golden_fn=golden_aic,
            runtime_cfg=cfg,
            rtol=tol["rtol"],
            atol=tol["atol"],
            compare_fn={"out_acc_k0": cmp3},
        )
        bad3, zero3, ratio3 = _acc_stats(captured3["acc"][0], captured3["acc"][1], tol)
        print(
            f"AIC-only device-fed: bad={bad3}/{captured3['acc'][0].numel()} "
            f"zero_act={zero3} ratio={ratio3:.4f}"
        )
