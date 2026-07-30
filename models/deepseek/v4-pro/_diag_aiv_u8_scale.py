"""Store mx_quant scale as UINT8 to isolate e8m0 D2H vs TQUANT/TSTORE."""
from __future__ import annotations

import os

import numpy as np
import pypto.language as pl
import torch

from _diag_prefill_qr_proj_instr import Q_LORA, Q_TILE, T, _QR_KS, _shared_inputs
from golden import TensorSpec, run_jit


@pl.jit
def aiv_u8(
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    out_scale_u8: pl.Out[pl.Tensor[[T, _QR_KS], pl.UINT8]],
    out_act_fp8: pl.Out[pl.Tensor[[T, Q_TILE], pl.FP8E4M3FN]],
):
    for _ in pl.spmd(1, name_hint="aiv_u8"):
        qr_tile = pl.load(qr, [0, 0], [T, Q_TILE], target_memory=pl.Mem.Vec)
        qr_f = pl.cast(
            pl.cast(qr_tile, target_type=pl.FP16, mode="none"), target_type=pl.FP32, mode="none"
        )
        qr_dq = pl.row_expand_mul(qr_f, pl.load(qr_scale, [0, 0], [T, 1], target_memory=pl.Mem.Vec))
        qr_q, qr_s = pl.mx_quant(qr_dq, mode="mxfp8_e4m3")
        pl.store(pl.tile.reinterpret_view(qr_s, pl.UINT8), [0, 0], out_scale_u8)
        pl.store(pl.tile.reinterpret_view(qr_q, pl.FP8E4M3FN), [0, 0], out_act_fp8)
    return out_scale_u8, out_act_fp8


if __name__ == "__main__":
    if hasattr(aiv_u8, "_cache"):
        aiv_u8._cache.clear()
    data = _shared_inputs()
    cfg = dict(platform="a5", device_id=int(os.environ.get("TASK_DEVICE", "0")))
    st: dict = {}

    def golden(t):
        t["out_scale_u8"][:] = data["xs"].view(torch.uint8)
        t["out_act_fp8"][:] = data["xq"]

    def cmp(name):
        def f(a, e, **_k):
            st[name] = (a.cpu().clone(), e.cpu().clone())
            return True, ""

        f.__name__ = name
        return f

    run_jit(
        fn=aiv_u8,
        specs=[
            TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: data["qr"]),
            TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: data["qr_scale"]),
            TensorSpec("out_scale_u8", [T, _QR_KS], torch.uint8, is_output=True),
            TensorSpec("out_act_fp8", [T, Q_TILE], torch.float8_e4m3fn, is_output=True),
        ],
        golden_fn=golden,
        runtime_cfg=cfg,
        rtol=0.0,
        atol=0.0,
        compare_fn={"out_scale_u8": cmp("s"), "out_act_fp8": cmp("a")},
    )
    su = st["s"][0].numpy().reshape(-1)
    eu = st["s"][1].numpy().reshape(-1)
    print(f"u8_scale: eq={int((su == eu).sum())}/512 sample={su[:16].tolist()} exp={eu[:16].tolist()}")
    i32 = np.frombuffer(bytes(su), dtype="<u4")
    print(f"u4 sample={i32[:4].tolist()} nunique={len(np.unique(i32))}")
