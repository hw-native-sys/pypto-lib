# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE packed dispatch (decode, single-card EP).

Pure scatter from token-major router outputs to the per-local-expert layout
consumed by ``expert_routed``. The INT8 quant of ``x_norm`` already happened in
``gate``; this kernel only moves the pre-quantized rows and dequant
scales into the recv buffers.
"""


import pypto.language as pl

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ,
                    EP_WORLD_SIZE, EP_RANK, RECV_MAX)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE
EXPERTS_START_IDX = EP_RANK * N_LOCAL_EXPERTS


@pl.jit.inline
def dispatch(
    x_norm_i8:       pl.Tensor[[T, D],    pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1],    pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq:     pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_weights:      pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32],
    recv_token:        pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1],           pl.INT32],
):
    # recv_x still uses a flat row view because its destination row is
    # data-dependent. Small route metadata uses the natural 2-D layout.
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch"):
        # Zero-init tail slots so downstream consumers (expert_routed / combine)
        # can rely on slot >= recv_expert_count[e] being neutral:
        #   - recv_scale_dq = 0 -> dequant of any recv_x tail row yields 0
        #   - recv_weights  = 0 -> combine's weighted reduction skips tail rows
        #   - recv_token    = 0 -> safe scatter target (combine ignores via count)
        # The recv_x INT8 tail is left uninitialized; pairing it with scale_dq=0
        # is sufficient to neutralize its contribution after dequant.
        for e in pl.range(N_LOCAL_EXPERTS):
            pl.write(recv_expert_count, [e, 0], pl.cast(0, pl.INT32))
            for s in pl.range(RECV_MAX):
                pl.write(recv_scale_dq, [e, s], 0.0)
                pl.write(recv_weights, [e, s], 0.0)
                pl.write(recv_token, [e, s], pl.cast(0, pl.INT32))

        for t in pl.range(T):
            for k in pl.range(TOPK):
                e_global = pl.read(indices, [t, k])
                e = pl.cast(e_global - EXPERTS_START_IDX, pl.INDEX)
                slot_i32 = pl.read(recv_expert_count, [e, 0])
                slot = pl.cast(slot_i32, pl.INDEX)
                dst = e * RECV_MAX + slot

                recv_x_flat = pl.assemble(recv_x_flat, pl.slice(x_norm_i8, [1, D], [t, 0]), [dst, 0])
                pl.write(recv_scale_dq, [e, slot], pl.read(x_norm_scale, [t, 0]))
                pl.write(recv_weights, [e, slot], pl.read(weights, [t, k]))
                pl.write(recv_token, [e, slot], pl.cast(t, pl.INT32))
                pl.write(recv_expert_count, [e, 0], pl.cast(slot_i32 + 1, pl.INT32))


@pl.jit
def dispatch_test(
    x_norm_i8:       pl.Tensor[[T, D],    pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1],    pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    recv_x:            pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8]],
    recv_scale_dq:     pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_weights:      pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.FP32]],
    recv_token:        pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX],    pl.INT32]],
    recv_expert_count: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32]],
):
    dispatch(
        x_norm_i8, x_norm_scale, indices, weights,
        recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count,
    )
    return recv_x, recv_scale_dq, recv_weights, recv_token, recv_expert_count


def golden_dispatch(tensors):
    """Torch reference for the packed dispatch contract (pure scatter)."""
    import torch

    x_norm_i8       = tensors["x_norm_i8"]        # [T, D]    int8 (pre-quantized in router)
    x_norm_scale = tensors["x_norm_scale"]  # [T, 1]    fp32 per-token dequant scale
    indices = tensors["indices"]   # [T, TOPK] int32
    weights = tensors["weights"]   # [T, TOPK] fp32

    recv_x        = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.int8)
    recv_scale_dq = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights  = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_token    = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
    cursor = [0] * N_LOCAL_EXPERTS
    for t in range(T):
        for k in range(TOPK):
            e = int(indices[t, k].item()) - EXPERTS_START_IDX
            s = cursor[e]
            assert 0 <= e < N_LOCAL_EXPERTS
            assert s < RECV_MAX, f"expert {e} received > RECV_MAX={RECV_MAX} rows"
            recv_x[e, s, :]      = x_norm_i8[t, :]
            recv_scale_dq[e, s]  = float(x_norm_scale[t, 0].item())
            recv_weights[e, s]   = float(weights[t, k].item())
            recv_token[e, s]     = t
            cursor[e] = s + 1

    recv_count = torch.zeros(N_LOCAL_EXPERTS, 1, dtype=torch.int32)
    for e in range(N_LOCAL_EXPERTS):
        recv_count[e, 0] = cursor[e]

    tensors["recv_x"][:]            = recv_x
    tensors["recv_scale_dq"][:]     = recv_scale_dq
    tensors["recv_weights"][:]      = recv_weights
    tensors["recv_token"][:]        = recv_token
    tensors["recv_expert_count"][:] = recv_count


def _valid_rows_compare(is_3d: bool = False):
    """Comparator that checks only the valid packed rows (slot < count).

    Dispatch leaves tail rows (slot >= recv_expert_count[e]) uninitialized --
    they carry no contract because every downstream consumer is count-bounded.
    Build the valid mask from the golden recv_expert_count and compare only the
    masked entries.
    """
    import torch

    def cmp(actual, expected, *, expected_outputs, rtol, atol, **_):
        count = expected_outputs["recv_expert_count"].cpu().reshape(-1, 1)        # [E, 1]
        valid = torch.arange(RECV_MAX).reshape(1, RECV_MAX) < count               # [E, RECV_MAX]
        a = actual.cpu()
        e = expected.cpu()
        if is_3d:
            valid = valid.unsqueeze(-1).expand_as(a)
        a_v = a[valid].to(torch.float32)
        e_v = e[valid].to(torch.float32)
        if torch.allclose(a_v, e_v, rtol=rtol, atol=atol):
            return True, ""
        diff = (a_v - e_v).abs()
        n_bad = int((diff > atol + rtol * e_v.abs()).sum().item())
        worst = float(diff.max().item()) if diff.numel() else 0.0
        return False, (
            f"    valid-row mismatch (rtol={rtol} atol={atol}): "
            f"{n_bad}/{a_v.numel()} bad, worst_diff={worst:.6g}"
        )

    cmp.__name__ = "valid_rows_compare"
    return cmp


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_norm_i8():
        return torch.randint(-128, 128, (T, D), dtype=torch.int8)

    def init_x_norm_scale():
        # Per-token dequant scale (strictly positive, as produced by the router).
        return (torch.rand(T, 1) + 0.01).float()

    def init_indices():
        # Each token picks TOPK distinct experts.
        rows = [torch.randperm(N_LOCAL_EXPERTS)[:TOPK] for _ in range(T)]
        return torch.stack(rows).to(torch.int32)

    def init_weights():
        # Per-row weights normalized to sum=routed_scaling_factor.
        w = torch.rand(T, TOPK) + 0.1
        w = w / w.sum(dim=-1, keepdim=True) * M.routed_scaling_factor
        return w.float()

    return [
        TensorSpec("x_norm_i8",       [T, D], torch.int8,    init_value=init_x_norm_i8),
        TensorSpec("x_norm_scale", [T, 1], torch.float32, init_value=init_x_norm_scale),
        TensorSpec("indices", [T, TOPK], torch.int32,    init_value=init_indices),
        TensorSpec("weights", [T, TOPK], torch.float32,  init_value=init_weights),
        TensorSpec("recv_x",            [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8,     is_output=True),
        TensorSpec("recv_scale_dq",     [N_LOCAL_EXPERTS, RECV_MAX],    torch.float32,  is_output=True),
        TensorSpec("recv_weights",      [N_LOCAL_EXPERTS, RECV_MAX],    torch.float32,  is_output=True),
        TensorSpec("recv_token",        [N_LOCAL_EXPERTS, RECV_MAX],    torch.int32,    is_output=True),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1],           torch.int32,    is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=dispatch_test,
        specs=build_tensor_specs(),
        golden_fn=golden_dispatch,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Tail rows (slot >= count) are uninitialized by design; only the
            # valid packed rows carry a contract.
            "recv_x":        _valid_rows_compare(is_3d=True),
            "recv_scale_dq": _valid_rows_compare(),
            "recv_weights":  _valid_rows_compare(),
            "recv_token":    _valid_rows_compare(),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
