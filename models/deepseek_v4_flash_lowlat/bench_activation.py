# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sweep activation count: does the load-balancing pre-task pay when n_active <= 24?

Active slots are scattered over the 48-slot table (torch.randperm), which is the
case the pre-task exists for: the raw grid-stride can land two live experts on
one core while others idle.
"""
import argparse, importlib, torch
from golden import TensorSpec, ratio_reldiff, run_jit
from utils import int8_quant_per_row
from expert_routed import (D, IDX_PAD, MOE_INTER, N_EXPERTS, N_SLOTS, RECV_MAX,
                           golden_expert_routed, gen_routed_weight)

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True)
p.add_argument("--n-active", type=int, required=True)
p.add_argument("-d", "--device", type=int, default=0)
p.add_argument("--dense", action="store_true", help="active slots are 0..n-1, as moe.route_group emits them")
a = p.parse_args()

torch.manual_seed(1234)
counts = torch.zeros(N_SLOTS, dtype=torch.int32)
live = (torch.arange(a.n_active) if a.dense else torch.randperm(N_SLOTS)[:a.n_active])
counts[live] = torch.randint(1, 9, (a.n_active,), dtype=torch.int32)
counts_2d = torch.zeros(N_SLOTS, IDX_PAD, dtype=torch.int32); counts_2d[:, 0] = counts
se = torch.randperm(N_EXPERTS)[:N_SLOTS].to(torch.int32)
se_2d = torch.zeros(N_SLOTS, IDX_PAD, dtype=torch.int32); se_2d[:, 0] = se

x = torch.randn(N_SLOTS, RECV_MAX, D, dtype=torch.bfloat16)
v3 = torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_SLOTS, 1, 1)
xi, xs = int8_quant_per_row(x)
xi = torch.where(v3, xi, torch.zeros_like(xi)); v2 = v3.squeeze(-1)
xs = torch.where(v2, xs.squeeze(-1), torch.zeros_like(xs.squeeze(-1)))
wt = torch.where(v2, torch.rand(N_SLOTS, RECV_MAX), torch.zeros(N_SLOTS, RECV_MAX))

def bank(tail, std):
    wi, ws = gen_routed_weight((N_SLOTS, *tail), std)
    bi = torch.zeros(N_EXPERTS, *tail, dtype=torch.int8)
    bs = torch.zeros(N_EXPERTS, tail[0], dtype=torch.float32)
    bi[se.long()] = wi; bs[se.long()] = ws
    return bi, bs

w1, w1s = bank((MOE_INTER, D), 2.47e-2)
w3, w3s = bank((MOE_INTER, D), 2.46e-2)
w2, w2s = bank((D, MOE_INTER), 2.44e-2)

mod = importlib.import_module(a.variant)
fn = getattr(mod, a.variant + "_test")
r = run_jit(
    fn=fn,
    specs=[
        TensorSpec("recv_x", [N_SLOTS, RECV_MAX, D], torch.int8, init_value=lambda: xi),
        TensorSpec("recv_scale_dq", [N_SLOTS, RECV_MAX], torch.float32, init_value=lambda: xs.float()),
        TensorSpec("recv_weights", [N_SLOTS, RECV_MAX], torch.float32, init_value=lambda: wt),
        TensorSpec("recv_expert_count", [N_SLOTS, IDX_PAD], torch.int32, init_value=lambda: counts_2d),
        TensorSpec("slot_expert", [N_SLOTS, IDX_PAD], torch.int32, init_value=lambda: se_2d),
        TensorSpec("routed_w1", [N_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w1),
        TensorSpec("routed_w1_scale", [N_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w1s),
        TensorSpec("routed_w3", [N_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w3),
        TensorSpec("routed_w3_scale", [N_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w3s),
        TensorSpec("routed_w2", [N_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2),
        TensorSpec("routed_w2_scale", [N_EXPERTS, D], torch.float32, init_value=lambda: w2s),
        TensorSpec("recv_y", [N_SLOTS, RECV_MAX, D], torch.bfloat16, is_output=True),
    ],
    golden_fn=golden_expert_routed,
    runtime_cfg=dict(platform="a2a3", device_id=a.device),
    rtol=1e-3, atol=1e-3,
    compare_fn={"recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)},
)
raise SystemExit(0 if r.passed else 1)
