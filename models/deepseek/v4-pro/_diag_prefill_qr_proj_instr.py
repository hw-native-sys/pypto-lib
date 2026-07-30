# Copyright (c) PyPTO Contributors.
"""Instruction-level blame for prefill_idx_qr_proj MX A-scale path.

Kernels (same peel: SPMD idx=0, K-chunk=0):

  path_nd      — production MIXED: mx_quant → ND TSTORE → mx_a_zz TLOAD → matmul_mx
  path_split   — AIV-only quant/store, then AIC-only load/matmul (two launches, no ExpandMixed)
  path_host_nd — host ND scale on GM + AND2ZZ TLOAD
  path_zz      — host ZZ scale + AZZ2ZZ (rewrite OFF)

Usage::

    python models/deepseek/v4-pro/_diag_prefill_qr_proj_instr.py -p a5 -d 0
    python models/deepseek/v4-pro/_diag_prefill_qr_proj_instr.py -p a5 -d 0 --mode split
"""

from __future__ import annotations

import argparse

import numpy as np
import pypto.language as pl
import torch

from config import FLASH as M, MX_BLOCK_K, INT8_AMAX_EPS, INT8_SCALE_MAX
from mx_quant_common import (
    ATOL_RTOL,
    convert_x1_scale_format,
    dynamic_mx_quant_e4m3,
    float8_e8m0_to_uint8,
    gen_mxfp8_weight_kn,
    mx_matmul_fp8,
    unpack_scale_b_nn_tiled,
)
from golden import TensorSpec, ratio_allclose, run_jit

T = 128
Q_LORA = M.q_lora_rank
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
Q_TILE = 128
Q_OUT_TILE = 256
_QR_KS = Q_TILE // MX_BLOCK_K
_QR_SPMD = IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE
_QR_K_CHUNKS = Q_LORA // Q_TILE
_WQ_B_SCALE_ROWS = _QR_SPMD * _QR_K_CHUNKS * _QR_KS


def _int8_quant_per_row(x: torch.Tensor):
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    out_i8 = torch.round(rows * scale_quant).to(torch.int32).to(torch.float16).to(torch.int8)
    return out_i8.reshape_as(x), (1.0 / scale_quant).reshape(*x.shape[:-1], 1)


@pl.jit
def aiv_quant_store(
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    out_scale_nd: pl.Out[pl.Tensor[[T, _QR_KS], pl.FP8E8M0]],
    out_act_fp8: pl.Out[pl.Tensor[[T, Q_TILE], pl.FP8E4M3FN]],
):
    """AIV-only: mx_quant + ND store. No Mat/Left → no ExpandMixed."""
    for _ in pl.spmd(1, name_hint="aiv_quant_store"):
        qr_tile = pl.load(qr, [0, 0], [T, Q_TILE], target_memory=pl.Mem.Vec)
        qr_f = pl.cast(
            pl.cast(qr_tile, target_type=pl.FP16, mode="none"), target_type=pl.FP32, mode="none"
        )
        qr_dq = pl.row_expand_mul(qr_f, pl.load(qr_scale, [0, 0], [T, 1], target_memory=pl.Mem.Vec))
        qr_q, qr_s = pl.mx_quant(qr_dq, mode="mxfp8_e4m3")
        pl.store(pl.tile.reinterpret_view(qr_s, pl.FP8E8M0), [0, 0], out_scale_nd)
        pl.store(pl.tile.reinterpret_view(qr_q, pl.FP8E4M3FN), [0, 0], out_act_fp8)
    return out_scale_nd, out_act_fp8


@pl.jit
def aic_matmul_from_gm(
    act_fp8: pl.Tensor[[T, Q_TILE], pl.FP8E4M3FN],
    scale_nd: pl.Tensor[[T, _QR_KS], pl.FP8E8M0],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[_WQ_B_SCALE_ROWS, Q_OUT_TILE], pl.FP8E8M0],
    out_acc_k0: pl.Out[pl.Tensor[[T, Q_OUT_TILE], pl.FP32]],
):
    """AIC-only: GM act/scale → Left/LeftScale → matmul_mx. No mx_quant → no ExpandMixed."""
    for _ in pl.spmd(1, name_hint="aic_matmul_from_gm"):
        wq_tile = pl.load(wq_b, [0, 0], [Q_TILE, Q_OUT_TILE], target_memory=pl.Mem.Mat)
        ws_tile = pl.load(
            wq_b_scale, [0, 0], [_QR_KS, Q_OUT_TILE], target_memory=pl.Mem.Mat, mx_layout="mx_b_nn"
        )
        act_mat = pl.load(act_fp8, [0, 0], [T, Q_TILE], target_memory=pl.Mem.Mat)
        qr_la = pl.move(act_mat, target_memory=pl.Mem.Left)
        qr_la = pl.set_validshape(qr_la, T, Q_TILE)
        qr_las = pl.move(
            pl.load(scale_nd, [0, 0], [T, _QR_KS], target_memory=pl.Mem.Mat, mx_layout="mx_a_zz"),
            target_memory=pl.Mem.LeftScale,
        )
        qr_las = pl.tget_scale_addr(qr_las, qr_la)
        qr_las = pl.set_validshape(qr_las, T, _QR_KS)
        wq_rb = pl.move(wq_tile, target_memory=pl.Mem.Right)
        wq_rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
        wq_rbs = pl.tget_scale_addr(wq_rbs, wq_rb)
        pl.store(pl.matmul_mx(qr_la, qr_las, wq_rb, wq_rbs), [0, 0], out_acc_k0)
    return out_acc_k0


@pl.jit
def path_nd_diag(
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[_WQ_B_SCALE_ROWS, Q_OUT_TILE], pl.FP8E8M0],
    out_scale_nd: pl.Out[pl.Tensor[[T, _QR_KS], pl.FP8E8M0]],
    out_act_fp8: pl.Out[pl.Tensor[[T, Q_TILE], pl.FP8E4M3FN]],
    out_acc_k0: pl.Out[pl.Tensor[[T, Q_OUT_TILE], pl.FP32]],
):
    mx_scale_ws = pl.create_tensor([T, _QR_KS], dtype=pl.FP8E8M0)
    for _ in pl.spmd(1, name_hint="path_nd_diag"):
        qr_tile = pl.load(qr, [0, 0], [T, Q_TILE], target_memory=pl.Mem.Vec)
        qr_f = pl.cast(
            pl.cast(qr_tile, target_type=pl.FP16, mode="none"), target_type=pl.FP32, mode="none"
        )
        qr_dq = pl.row_expand_mul(qr_f, pl.load(qr_scale, [0, 0], [T, 1], target_memory=pl.Mem.Vec))
        qr_q, qr_s = pl.mx_quant(qr_dq, mode="mxfp8_e4m3")
        pl.store(pl.tile.reinterpret_view(qr_s, pl.FP8E8M0), [0, 0], out_scale_nd)
        pl.store(pl.tile.reinterpret_view(qr_s, pl.FP8E8M0), [0, 0], mx_scale_ws)
        pl.store(pl.tile.reinterpret_view(qr_q, pl.FP8E4M3FN), [0, 0], out_act_fp8)

        wq_tile = pl.load(wq_b, [0, 0], [Q_TILE, Q_OUT_TILE], target_memory=pl.Mem.Mat)
        ws_tile = pl.load(
            wq_b_scale, [0, 0], [_QR_KS, Q_OUT_TILE], target_memory=pl.Mem.Mat, mx_layout="mx_b_nn"
        )
        qr_la = pl.move(
            pl.move(pl.tile.reinterpret_view(qr_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Left,
        )
        qr_la = pl.set_validshape(qr_la, T, Q_TILE)
        qr_las = pl.move(
            pl.load(
                mx_scale_ws, [0, 0], [T, _QR_KS], target_memory=pl.Mem.Mat, mx_layout="mx_a_zz"
            ),
            target_memory=pl.Mem.LeftScale,
        )
        qr_las = pl.tget_scale_addr(qr_las, qr_la)
        qr_las = pl.set_validshape(qr_las, T, _QR_KS)
        wq_rb = pl.move(wq_tile, target_memory=pl.Mem.Right)
        wq_rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
        wq_rbs = pl.tget_scale_addr(wq_rbs, wq_rb)
        pl.store(pl.matmul_mx(qr_la, qr_las, wq_rb, wq_rbs), [0, 0], out_acc_k0)
    return out_scale_nd, out_act_fp8, out_acc_k0


@pl.jit
def path_host_nd_diag(
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[_WQ_B_SCALE_ROWS, Q_OUT_TILE], pl.FP8E8M0],
    scale_nd: pl.Tensor[[T, _QR_KS], pl.FP8E8M0],
    out_acc_k0: pl.Out[pl.Tensor[[T, Q_OUT_TILE], pl.FP32]],
    out_scale_discard: pl.Out[pl.Tensor[[T, _QR_KS], pl.FP8E8M0]],
):
    """Act from mx_quant; A-scale is host ND on GM (AND2ZZ via rewrite). Isolates layout vs AIV store race."""
    for _ in pl.spmd(1, name_hint="path_host_nd_diag"):
        qr_tile = pl.load(qr, [0, 0], [T, Q_TILE], target_memory=pl.Mem.Vec)
        qr_f = pl.cast(
            pl.cast(qr_tile, target_type=pl.FP16, mode="none"), target_type=pl.FP32, mode="none"
        )
        qr_dq = pl.row_expand_mul(qr_f, pl.load(qr_scale, [0, 0], [T, 1], target_memory=pl.Mem.Vec))
        qr_q, qr_s = pl.mx_quant(qr_dq, mode="mxfp8_e4m3")
        pl.store(pl.tile.reinterpret_view(qr_s, pl.FP8E8M0), [0, 0], out_scale_discard)
        wq_tile = pl.load(wq_b, [0, 0], [Q_TILE, Q_OUT_TILE], target_memory=pl.Mem.Mat)
        ws_tile = pl.load(
            wq_b_scale, [0, 0], [_QR_KS, Q_OUT_TILE], target_memory=pl.Mem.Mat, mx_layout="mx_b_nn"
        )
        qr_la = pl.move(
            pl.move(pl.tile.reinterpret_view(qr_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Left,
        )
        qr_la = pl.set_validshape(qr_la, T, Q_TILE)
        qr_las = pl.move(
            pl.load(scale_nd, [0, 0], [T, _QR_KS], target_memory=pl.Mem.Mat, mx_layout="mx_a_zz"),
            target_memory=pl.Mem.LeftScale,
        )
        qr_las = pl.tget_scale_addr(qr_las, qr_la)
        qr_las = pl.set_validshape(qr_las, T, _QR_KS)
        wq_rb = pl.move(wq_tile, target_memory=pl.Mem.Right)
        wq_rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
        wq_rbs = pl.tget_scale_addr(wq_rbs, wq_rb)
        pl.store(pl.matmul_mx(qr_la, qr_las, wq_rb, wq_rbs), [0, 0], out_acc_k0)
    return out_acc_k0, out_scale_discard


@pl.jit
def path_zz_diag(
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[_WQ_B_SCALE_ROWS, Q_OUT_TILE], pl.FP8E8M0],
    scale_zz: pl.Tensor[[T, _QR_KS], pl.FP8E8M0],
    out_acc_k0: pl.Out[pl.Tensor[[T, Q_OUT_TILE], pl.FP32]],
    out_scale_discard: pl.Out[pl.Tensor[[T, _QR_KS], pl.FP8E8M0]],
):
    """Same matmul, but A-scale already ZZ-packed on GM (no ND staging)."""
    for _ in pl.spmd(1, name_hint="path_zz_diag"):
        qr_tile = pl.load(qr, [0, 0], [T, Q_TILE], target_memory=pl.Mem.Vec)
        qr_f = pl.cast(
            pl.cast(qr_tile, target_type=pl.FP16, mode="none"), target_type=pl.FP32, mode="none"
        )
        qr_dq = pl.row_expand_mul(qr_f, pl.load(qr_scale, [0, 0], [T, 1], target_memory=pl.Mem.Vec))
        qr_q, qr_s = pl.mx_quant(qr_dq, mode="mxfp8_e4m3")
        pl.store(pl.tile.reinterpret_view(qr_s, pl.FP8E8M0), [0, 0], out_scale_discard)
        wq_tile = pl.load(wq_b, [0, 0], [Q_TILE, Q_OUT_TILE], target_memory=pl.Mem.Mat)
        ws_tile = pl.load(
            wq_b_scale, [0, 0], [_QR_KS, Q_OUT_TILE], target_memory=pl.Mem.Mat, mx_layout="mx_b_nn"
        )
        qr_la = pl.move(
            pl.move(pl.tile.reinterpret_view(qr_q, pl.FP8E4M3FN), target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Left,
        )
        qr_la = pl.set_validshape(qr_la, T, Q_TILE)
        qr_las = pl.move(
            pl.load(scale_zz, [0, 0], [T, _QR_KS], target_memory=pl.Mem.Mat, mx_layout="mx_a_zz"),
            target_memory=pl.Mem.LeftScale,
        )
        qr_las = pl.tget_scale_addr(qr_las, qr_la)
        qr_las = pl.set_validshape(qr_las, T, _QR_KS)
        wq_rb = pl.move(wq_tile, target_memory=pl.Mem.Right)
        wq_rbs = pl.move(ws_tile, target_memory=pl.Mem.RightScale)
        wq_rbs = pl.tget_scale_addr(wq_rbs, wq_rb)
        pl.store(pl.matmul_mx(qr_la, qr_las, wq_rb, wq_rbs), [0, 0], out_acc_k0)
    return out_acc_k0, out_scale_discard


def _shared_inputs(seed: int = 0):
    torch.manual_seed(seed)
    wq_b, wq_b_scale = gen_mxfp8_weight_kn(
        (Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM),
        dequant_std=0.108,
        chan_cv=0.56,
        n_tile=Q_OUT_TILE,
        k_tile=Q_TILE,
    )
    qr_i8, qr_scale = _int8_quant_per_row(torch.rand(T, Q_LORA))
    qr_f = qr_i8.float() * qr_scale.float()
    xq, xs = dynamic_mx_quant_e4m3(qr_f[:, :Q_TILE])
    nd_u8 = float8_e8m0_to_uint8(xs).cpu().numpy()
    zz_u8 = convert_x1_scale_format(nd_u8, block_size=16, c0_size_mx=2)
    scale_nd = xs.contiguous()
    scale_zz = torch.from_numpy(np.ascontiguousarray(zz_u8)).view(torch.float8_e8m0fnu)
    w = wq_b[:Q_TILE, :Q_OUT_TILE]
    w_s = unpack_scale_b_nn_tiled(
        wq_b_scale,
        k_tile_rows=_QR_KS,
        n_tile=Q_OUT_TILE,
        logical_k=Q_LORA // MX_BLOCK_K,
        logical_n=IDX_N_HEADS * IDX_HEAD_DIM,
    )[:_QR_KS, :Q_OUT_TILE]
    acc = mx_matmul_fp8(xq, xs, w, w_s).float()
    return {
        "qr": qr_i8,
        "qr_scale": qr_scale,
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "xq": xq,
        "xs": xs,
        "scale_nd": scale_nd,
        "scale_zz": scale_zz,
        "acc": acc,
    }


def _acc_stats(actual: torch.Tensor, expected: torch.Tensor, tol: dict) -> tuple[int, int, float]:
    a = actual.float().cpu()
    e = expected.float().cpu()
    band = tol["atol"] + tol["rtol"] * e.abs()
    bad = int(((a - e).abs() > band).sum())
    zero = int(((a.abs() < 1e-12) & (e.abs() > 1e-6)).sum())
    return bad, zero, bad / max(a.numel(), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("all", "split", "mixed"),
        default="split",
        help="split=AIV-only then AIC-only (default); mixed=path_nd only; all=full blame suite",
    )
    args = parser.parse_args()
    tol = ATOL_RTOL["indexer_fp8"]
    data = _shared_inputs()
    cfg = dict(platform=args.platform, device_id=args.device)

    def _run_split():
        print("=== RUN aiv_quant_store (AIV-only, no ExpandMixed) ===")
        aiv_state = {}

        def golden_aiv(tensors):
            tensors["out_scale_nd"][:] = data["xs"]
            tensors["out_act_fp8"][:] = data["xq"]

        def cmp_aiv(name):
            def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
                aiv_state[name] = (actual.cpu().clone(), expected.cpu().clone())
                return ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)(
                    actual, expected,
                    actual_outputs=actual_outputs, expected_outputs=expected_outputs,
                    inputs=inputs, rtol=rtol, atol=atol,
                )

            cmp.__name__ = name
            return cmp

        r_aiv = run_jit(
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
            compare_fn={
                "out_scale_nd": cmp_aiv("scale"),
                "out_act_fp8": cmp_aiv("act"),
            },
        )
        aiv_ok = (
            int((aiv_state["scale"][0].view(torch.uint8) != aiv_state["scale"][1].view(torch.uint8)).sum())
            == 0
            and int((aiv_state["act"][0].view(torch.uint8) != aiv_state["act"][1].view(torch.uint8)).sum())
            == 0
        )
        print(f"AIV quant/store: {'PASS' if aiv_ok else 'FAIL'} passed={r_aiv.passed}")

        # Feed device-produced GM buffers into AIC-only matmul (second launch).
        act_dev = aiv_state["act"][0]
        scale_dev = aiv_state["scale"][0]
        print("=== RUN aic_matmul_from_gm (AIC-only, device ND scale from AIV launch) ===")
        aic_state = {}

        def golden_aic(tensors):
            tensors["out_acc_k0"][:] = data["acc"]

        def cmp_aic(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            aic_state["acc"] = (actual.cpu().clone(), expected.cpu().clone())
            return ratio_allclose(atol=tol["atol"], rtol=tol["rtol"], max_error_ratio=tol["pct"])(
                actual, expected,
                actual_outputs=actual_outputs, expected_outputs=expected_outputs,
                inputs=inputs, rtol=rtol, atol=atol,
            )

        cmp_aic.__name__ = "acc_split"
        r_aic = run_jit(
            fn=aic_matmul_from_gm,
            specs=[
                TensorSpec("act_fp8", [T, Q_TILE], torch.float8_e4m3fn, init_value=lambda: act_dev),
                TensorSpec("scale_nd", [T, _QR_KS], torch.float8_e8m0fnu, init_value=lambda: scale_dev),
                TensorSpec(
                    "wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn,
                    init_value=lambda: data["wq_b"],
                ),
                TensorSpec(
                    "wq_b_scale", [_WQ_B_SCALE_ROWS, Q_OUT_TILE], torch.float8_e8m0fnu,
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
        bad, zero, ratio = _acc_stats(aic_state["acc"][0], aic_state["acc"][1], tol)
        print(
            f"AIC matmul (device-fed): bad={bad}/{aic_state['acc'][0].numel()} "
            f"ratio={ratio:.4f} zero_act={zero} passed={r_aic.passed}"
        )

        # Control: AIC-only with host ND act/scale (no AIV involvement).
        print("=== RUN aic_matmul_from_gm (AIC-only, HOST ND act/scale) ===")
        host_state = {}

        def cmp_host(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            host_state["acc"] = (actual.cpu().clone(), expected.cpu().clone())
            return ratio_allclose(atol=tol["atol"], rtol=tol["rtol"], max_error_ratio=tol["pct"])(
                actual, expected,
                actual_outputs=actual_outputs, expected_outputs=expected_outputs,
                inputs=inputs, rtol=rtol, atol=atol,
            )

        cmp_host.__name__ = "acc_host"
        r_host = run_jit(
            fn=aic_matmul_from_gm,
            specs=[
                TensorSpec("act_fp8", [T, Q_TILE], torch.float8_e4m3fn, init_value=lambda: data["xq"]),
                TensorSpec(
                    "scale_nd", [T, _QR_KS], torch.float8_e8m0fnu, init_value=lambda: data["scale_nd"]
                ),
                TensorSpec(
                    "wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn,
                    init_value=lambda: data["wq_b"],
                ),
                TensorSpec(
                    "wq_b_scale", [_WQ_B_SCALE_ROWS, Q_OUT_TILE], torch.float8_e8m0fnu,
                    init_value=lambda: data["wq_b_scale"],
                ),
                TensorSpec("out_acc_k0", [T, Q_OUT_TILE], torch.float32, is_output=True),
            ],
            golden_fn=golden_aic,
            runtime_cfg=cfg,
            rtol=tol["rtol"],
            atol=tol["atol"],
            compare_fn={"out_acc_k0": cmp_host},
        )
        bad_h, zero_h, ratio_h = _acc_stats(host_state["acc"][0], host_state["acc"][1], tol)
        print(
            f"AIC matmul (host-fed): bad={bad_h}/{host_state['acc'][0].numel()} "
            f"ratio={ratio_h:.4f} zero_act={zero_h} passed={r_host.passed}"
        )
        print(
            f"=== SPLIT BLAME: aiv_ok={aiv_ok} aic_device_ok={bad == 0} aic_host_ok={bad_h == 0} ==="
        )
        if bad_h == 0 and bad > 0:
            print(
                "  ⇒ AND2ZZ/matmul OK without ExpandMixed; "
                "AIV-only scale/act dump or AIV→AIC handoff is the blocker"
            )
        elif bad_h == 0 and bad == 0 and aiv_ok:
            print("  ⇒ full AIV→AIC split PASS ⇒ ExpandMixed/FIFO is the mixed-path bug")
        elif bad_h > 0:
            print("  ⇒ AIC-only host-fed FAIL ⇒ AND2ZZ/layout/matmul (not ExpandMixed)")
        return aiv_ok and bad == 0 and bad_h == 0

    if args.mode == "split":
        ok = _run_split()
        raise SystemExit(0 if ok else 1)

    # ----- path_nd (mixed) -----
    nd_state = {}

    def golden_nd(tensors):
        tensors["out_scale_nd"][:] = data["xs"]
        tensors["out_act_fp8"][:] = data["xq"]
        tensors["out_acc_k0"][:] = data["acc"]

    def cmp_capture(name, strict):
        base = (
            ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
            if strict
            else ratio_allclose(atol=tol["atol"], rtol=tol["rtol"], max_error_ratio=tol["pct"])
        )

        def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            nd_state[name] = (actual.cpu().clone(), expected.cpu().clone())
            return base(
                actual, expected,
                actual_outputs=actual_outputs, expected_outputs=expected_outputs,
                inputs=inputs, rtol=rtol, atol=atol,
            )

        cmp.__name__ = name
        return cmp

    print("=== RUN path_nd (MIXED mx_quant → ND store → MX TLOAD → matmul) ===")
    r_nd = run_jit(
        fn=path_nd_diag,
        specs=[
            TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: data["qr"]),
            TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: data["qr_scale"]),
            TensorSpec(
                "wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn,
                init_value=lambda: data["wq_b"],
            ),
            TensorSpec(
                "wq_b_scale", [_WQ_B_SCALE_ROWS, Q_OUT_TILE], torch.float8_e8m0fnu,
                init_value=lambda: data["wq_b_scale"],
            ),
            TensorSpec("out_scale_nd", [T, _QR_KS], torch.float8_e8m0fnu, is_output=True),
            TensorSpec("out_act_fp8", [T, Q_TILE], torch.float8_e4m3fn, is_output=True),
            TensorSpec("out_acc_k0", [T, Q_OUT_TILE], torch.float32, is_output=True),
        ],
        golden_fn=golden_nd,
        runtime_cfg=cfg,
        rtol=tol["rtol"],
        atol=tol["atol"],
        compare_fn={
            "out_scale_nd": cmp_capture("scale_nd", True),
            "out_act_fp8": cmp_capture("act", True),
            "out_acc_k0": cmp_capture("acc_nd", False),
        },
    )
    a_ok = (
        int((nd_state["scale_nd"][0].view(torch.uint8) != nd_state["scale_nd"][1].view(torch.uint8)).sum())
        == 0
        and int((nd_state["act"][0].view(torch.uint8) != nd_state["act"][1].view(torch.uint8)).sum()) == 0
    )
    bad_nd, zero_nd, ratio_nd = _acc_stats(nd_state["acc_nd"][0], nd_state["acc_nd"][1], tol)
    print(f"A mx_quant/ND store: {'PASS' if a_ok else 'FAIL'}")
    print(f"C path_nd matmul:    bad={bad_nd}/{nd_state['acc_nd'][0].numel()} "
          f"ratio={ratio_nd:.4f} zero_act={zero_nd} passed={r_nd.passed}")

    if args.mode == "mixed":
        raise SystemExit(0 if a_ok and bad_nd == 0 else 1)

    _run_split()

    # ----- path_host_nd (rewrite ON → AND2ZZ; host ND GM, no AIV write of A-scale) -----
    host_nd_state = {}

    def golden_host_nd(tensors):
        tensors["out_acc_k0"][:] = data["acc"]
        tensors["out_scale_discard"][:] = data["xs"]

    def cmp_host_nd(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        host_nd_state["acc"] = (actual.cpu().clone(), expected.cpu().clone())
        return ratio_allclose(atol=tol["atol"], rtol=tol["rtol"], max_error_ratio=tol["pct"])(
            actual, expected,
            actual_outputs=actual_outputs, expected_outputs=expected_outputs,
            inputs=inputs, rtol=rtol, atol=atol,
        )

    cmp_host_nd.__name__ = "acc_host_nd"
    print("=== RUN path_host_nd (host ND scale → AND2ZZ TLOAD → matmul; rewrite ON) ===")
    try:
        r_host = run_jit(
            fn=path_host_nd_diag,
            specs=[
                TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: data["qr"]),
                TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: data["qr_scale"]),
                TensorSpec(
                    "wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn,
                    init_value=lambda: data["wq_b"],
                ),
                TensorSpec(
                    "wq_b_scale", [_WQ_B_SCALE_ROWS, Q_OUT_TILE], torch.float8_e8m0fnu,
                    init_value=lambda: data["wq_b_scale"],
                ),
                TensorSpec(
                    "scale_nd", [T, _QR_KS], torch.float8_e8m0fnu,
                    init_value=lambda: data["scale_nd"],
                ),
                TensorSpec("out_acc_k0", [T, Q_OUT_TILE], torch.float32, is_output=True),
                TensorSpec("out_scale_discard", [T, _QR_KS], torch.float8_e8m0fnu, is_output=True),
            ],
            golden_fn=golden_host_nd,
            runtime_cfg=dict(platform=args.platform, device_id=args.device),
            rtol=tol["rtol"],
            atol=tol["atol"],
            compare_fn={
                "out_acc_k0": cmp_host_nd,
                "out_scale_discard": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=1.0),
            },
        )
        bad_host, zero_host, ratio_host = _acc_stats(
            host_nd_state["acc"][0], host_nd_state["acc"][1], tol
        )
        print(
            f"C path_host_nd matmul: bad={bad_host}/{host_nd_state['acc'][0].numel()} "
            f"ratio={ratio_host:.4f} zero_act={zero_host} passed={r_host.passed}"
        )
        host_ok = bad_host <= round(tol["pct"] * host_nd_state["acc"][0].numel())
    except Exception as ex:
        print(f"path_host_nd FAILED: {type(ex).__name__}: {ex}")
        host_ok = None
        bad_host = -1

    # ----- path_zz (needs rewrite OFF so MX_A_ZZ stays AZZ2ZZ) -----
    from pypto.backend import _ptoas_preprocess as pp

    prev_flag = pp._ENABLE_MX_A_ZZ_TO_ND_REWRITE
    prev_req = pp._require_mx_a_zz_to_nd_rewrite
    zz_state = {}
    try:
        pp._ENABLE_MX_A_ZZ_TO_ND_REWRITE = False
        pp._require_mx_a_zz_to_nd_rewrite = lambda: None  # allow incomplete ZZ path for this probe

        def golden_zz(tensors):
            tensors["out_acc_k0"][:] = data["acc"]
            tensors["out_scale_discard"][:] = data["xs"]

        def cmp_zz(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
            zz_state["acc"] = (actual.cpu().clone(), expected.cpu().clone())
            return ratio_allclose(atol=tol["atol"], rtol=tol["rtol"], max_error_ratio=tol["pct"])(
                actual, expected,
                actual_outputs=actual_outputs, expected_outputs=expected_outputs,
                inputs=inputs, rtol=rtol, atol=atol,
            )

        cmp_zz.__name__ = "acc_zz"
        print("=== RUN path_zz (host ZZ scale → MX_A_ZZ TLOAD → matmul; rewrite OFF) ===")
        try:
            r_zz = run_jit(
                fn=path_zz_diag,
                specs=[
                    TensorSpec("qr", [T, Q_LORA], torch.int8, init_value=lambda: data["qr"]),
                    TensorSpec("qr_scale", [T, 1], torch.float32, init_value=lambda: data["qr_scale"]),
                    TensorSpec(
                        "wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.float8_e4m3fn,
                        init_value=lambda: data["wq_b"],
                    ),
                    TensorSpec(
                        "wq_b_scale", [_WQ_B_SCALE_ROWS, Q_OUT_TILE], torch.float8_e8m0fnu,
                        init_value=lambda: data["wq_b_scale"],
                    ),
                    TensorSpec(
                        "scale_zz", [T, _QR_KS], torch.float8_e8m0fnu,
                        init_value=lambda: data["scale_zz"],
                    ),
                    TensorSpec("out_acc_k0", [T, Q_OUT_TILE], torch.float32, is_output=True),
                    TensorSpec("out_scale_discard", [T, _QR_KS], torch.float8_e8m0fnu, is_output=True),
                ],
                golden_fn=golden_zz,
                runtime_cfg=dict(platform=args.platform, device_id=args.device),
                rtol=tol["rtol"],
                atol=tol["atol"],
                compare_fn={
                    "out_acc_k0": cmp_zz,
                    "out_scale_discard": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=1.0),
                },
            )
            bad_zz, zero_zz, ratio_zz = _acc_stats(zz_state["acc"][0], zz_state["acc"][1], tol)
            print(f"C path_zz matmul:    bad={bad_zz}/{zz_state['acc'][0].numel()} "
                  f"ratio={ratio_zz:.4f} zero_act={zero_zz} passed={r_zz.passed}")
            zz_ok = bad_zz <= round(tol["pct"] * zz_state["acc"][0].numel())
        except Exception as ex:
            print(f"path_zz FAILED to run (rewrite-OFF may raise): {type(ex).__name__}: {ex}")
            zz_ok = None
            r_zz = None
    finally:
        pp._ENABLE_MX_A_ZZ_TO_ND_REWRITE = prev_flag
        pp._require_mx_a_zz_to_nd_rewrite = prev_req

    print("=== BLAME ===")
    if not a_ok:
        first = "A: pl.mx_quant / ND TSTORE"
    elif bad_nd > 0 and host_ok is True and zz_ok is True:
        first = "SYNC: AIV ND store→AIC AND2ZZ race (host-ND AND2ZZ OK; device-ND fails)"
    elif bad_nd > 0 and host_ok is False and zz_ok is True:
        first = "LAYOUT: AND2ZZ/MX_A_ND vs ND GM bytes (host-ND fails; host-ZZ AZZ2ZZ OK)"
    elif bad_nd > 0 and zz_ok is False:
        first = "C: pl.matmul_mx / tget_scale_addr / LeftScale (ZZ path also fails)"
    elif bad_nd > 0 and zz_ok is None:
        first = "B-or-C: path_nd Acc fails; path_zz unavailable (check rewrite guard)"
    else:
        first = "none on K0 peel — look at later K-chunks / other SPMD idx"
    print(f"FIRST_BAD_INSTR: {first}")
    print(
        f"summary: a_ok={a_ok} path_nd_bad={bad_nd} path_host_nd_ok={host_ok} path_zz_ok={zz_ok}"
    )
