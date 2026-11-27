# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check the gated-attention kernels' reference against the text-prefill one.

The Q6d kernels are gated against `reference.attn_w8a8` / `attn_truth`, which
work on the kernels' flat layout: the score scale folded into q, the gate read
out of the interleaved `q_proj` columns, attention banded over query rows.
`reference.gated_attention` is the one `test_model_reference.py` checks
against Qwen3.5's own module. This ties the first to the second at the real
head geometry -- 24 query heads, 4 KV heads, head dim 256, 64 rotary
dimensions -- so the kernels' golden inherits that check.

Host only: no NPU and no compile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.qwen3_8_27b import reference
from models.qwen3_8_27b.config import QWEN3_8_27B


TOL = 1e-12
MAX_THREADS = 32


def rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm())


def check(seq_len: int, seed: int, tol: float) -> bool:
    cfg = QWEN3_8_27B
    w = {name: reference.attn_inputs(name, seq_len, cfg, seed=seed)
         for name in ("x", "w_q", "w_q_scale", "w_k", "w_k_scale", "w_v", "w_v_scale",
                      "w_o", "w_o_scale", "q_norm_w", "k_norm_w")}
    positions = torch.arange(seq_len)
    cos, sin = reference.rotary_cos_sin(positions, cfg)
    passed = True

    # 1. the flat-layout chain against the module-checked mixer, same weights
    prefix = "layers.3.self_attn"
    module_w = {
        prefix + ".q_proj.weight": w["w_q"], prefix + ".q_proj.weight_scale": w["w_q_scale"],
        prefix + ".k_proj.weight": w["w_k"], prefix + ".k_proj.weight_scale": w["w_k_scale"],
        prefix + ".v_proj.weight": w["w_v"], prefix + ".v_proj.weight_scale": w["w_v_scale"],
        prefix + ".o_proj.weight": w["w_o"], prefix + ".o_proj.weight_scale": w["w_o_scale"],
        prefix + ".q_norm.weight": w["q_norm_w"], prefix + ".k_norm.weight": w["k_norm_w"],
    }
    theirs = reference.gated_attention(w["x"], module_w, cfg, prefix, positions)["out"]
    ours = reference.attn_truth(w["x"], w, cos, sin, cfg)
    distance = rel(ours, theirs)
    ok = distance <= tol
    passed &= ok
    print(f"  attn_truth vs gated_attention       {distance:.3e}  {'ok' if ok else 'FAIL'}")

    # 2. the banded attention against one dense softmax
    h, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    gen = torch.Generator().manual_seed(seed + 1)
    q = torch.randn(seq_len, h * d, generator=gen, dtype=torch.float64)
    k = torch.randn(seq_len, hkv * d, generator=gen, dtype=torch.float64)
    v = torch.randn(seq_len, hkv * d, generator=gen, dtype=torch.float64)
    band = reference.ATTN_BAND
    try:
        reference.ATTN_BAND = max(1, seq_len // 3)
        banded = reference.causal_gqa_attention(q, k, v, h, hkv, d, scale=d ** -0.5)
    finally:
        reference.ATTN_BAND = band
    qh = q.reshape(seq_len, h, d).transpose(0, 1)
    kh = k.reshape(seq_len, hkv, d).transpose(0, 1).repeat_interleave(h // hkv, dim=0)
    vh = v.reshape(seq_len, hkv, d).transpose(0, 1).repeat_interleave(h // hkv, dim=0)
    scores = (qh @ kh.transpose(-1, -2)) * d ** -0.5
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)
    dense = (torch.softmax(scores.masked_fill(causal, -torch.inf), dim=-1) @ vh).transpose(0, 1)
    distance = rel(banded, dense.reshape(seq_len, h * d))
    ok = distance <= tol
    passed &= ok
    print(f"  banded vs dense causal attention    {distance:.3e}  {'ok' if ok else 'FAIL'}")

    # 3. the W8A8 chain sits a quantisation floor from the truth, not at it and not far off
    floor = rel(reference.attn_w8a8(w["x"], w, cos, sin, cfg), ours)
    ok = 1e-3 < floor < 1e-1
    passed &= ok
    print(f"  attn_w8a8 vs attn_truth (the floor) {floor:.3e}  {'ok' if ok else 'FAIL'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Accepted and ignored: this check is host-only, but CI invokes every runnable
    # file in a model directory with the common platform and device flags.
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3",
                        help="accepted for CI's uniform invocation; nothing here runs on device")
    parser.add_argument("--sim", dest="execution_mode", action="store_true",
                        help="accepted for CI's uniform invocation; unused")
    parser.add_argument("-d", "--device", type=int, default=0, help="accepted for CI's uniform invocation; unused")
    parser.add_argument("--seq-len", type=int, default=67)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=TOL)
    args = parser.parse_args()

    torch.set_num_threads(min(torch.get_num_threads(), MAX_THREADS))
    torch.backends.mkldnn.enabled = False
    passed = check(args.seq_len, args.seed, args.tol)
    print(f"\n{'PASS' if passed else 'FAIL'}: gated-attention kernel reference")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
