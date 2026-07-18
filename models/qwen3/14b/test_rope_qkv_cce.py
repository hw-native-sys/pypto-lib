# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Stage 1 on-device correctness driver for the fused-prologue rope_qkv kernel.

Runs ONLY the C++ rope compute (Q/K-norm + RoPE + KV-cache writes) — no attention —
and validates its GM outputs (query, and the written k_cache / v_cache rows) against a
torch reference that mirrors decode_fwd.py's rope_qkv byte-for-byte. Proving this in
isolation separates a compute defect from the cross-core sync added in Stage 2.
"""

import argparse

import torch

from golden import TensorSpec, run_jit
from paged_attention_cce import (
    BATCH,
    BLOCK_SIZE,
    HEAD_DIM,
    KV_HIDDEN,
    NUM_HEADS,
    NUM_KV_HEADS,
    SUPPORTED_PLATFORMS,
    qwen_decode_rope_only_cce,
)

HIDDEN = NUM_HEADS * HEAD_DIM  # 5120
HALF = HEAD_DIM // 2           # 64
EPS = 1.0e-6
HEAD_DIM_INV = 1.0 / HEAD_DIM
MAX_SEQ = 512
NUM_BLOCKS = 1  # one page (BLOCK_SIZE=128 tokens) covers the 16 decode slots


def _rope_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """NeoX half-split RoPE tables (cols [0:64] and [64:128] duplicated)."""
    posv = torch.arange(MAX_SEQ).float().unsqueeze(1)
    inv_freq = 1.0 / (1.0e4 ** (torch.arange(0, HALF).float() / HALF))
    ang = posv * inv_freq.unsqueeze(0)
    return torch.cat([ang.cos(), ang.cos()], dim=1).float(), torch.cat([ang.sin(), ang.sin()], dim=1).float()


def build_specs(seed: int = 1234) -> list[TensorSpec]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    def rn(shape, std=1.0):
        return torch.empty(shape).normal_(0.0, std, generator=g)

    rope_cos, rope_sin = _rope_tables()
    seq_lens = torch.randint(1, MAX_SEQ + 1, (BATCH,), generator=g, dtype=torch.int32)
    # Distinct physical token slots per batch (no aliasing); cache_row = slot*NUM_KV_HEADS + ki.
    slot_mapping = torch.arange(BATCH, dtype=torch.int32)
    inv_rms = torch.rand(BATCH, generator=g) * 0.9 + 0.3  # deferred input-RMSNorm factor, > 0

    return [
        TensorSpec("q_proj", [BATCH, HIDDEN], torch.float32, init_value=rn([BATCH, HIDDEN], 0.02)),
        TensorSpec("k_proj", [BATCH, KV_HIDDEN], torch.float32, init_value=rn([BATCH, KV_HIDDEN], 0.02)),
        TensorSpec("v_proj", [BATCH, KV_HIDDEN], torch.float32, init_value=rn([BATCH, KV_HIDDEN], 0.02)),
        TensorSpec("inv_rms_states", [BATCH], torch.float32, init_value=inv_rms),
        TensorSpec("q_norm_weight", [1, HEAD_DIM], torch.float32, init_value=rn([1, HEAD_DIM], 0.1) + 1.0),
        TensorSpec("k_norm_weight", [1, HEAD_DIM], torch.float32, init_value=rn([1, HEAD_DIM], 0.1) + 1.0),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=seq_lens),
        TensorSpec("slot_mapping", [BATCH], torch.int32, init_value=slot_mapping),
        TensorSpec("rope_cos", [MAX_SEQ, HEAD_DIM], torch.float32, init_value=rope_cos),
        TensorSpec("rope_sin", [MAX_SEQ, HEAD_DIM], torch.float32, init_value=rope_sin),
        TensorSpec("query", [BATCH, NUM_HEADS, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec(
            "key_cache",
            [NUM_BLOCKS, BLOCK_SIZE, KV_HIDDEN],
            torch.bfloat16,
            is_output=True,
            init_value=rn([NUM_BLOCKS, BLOCK_SIZE, KV_HIDDEN], 0.01).to(torch.bfloat16),
        ),
        TensorSpec(
            "value_cache",
            [NUM_BLOCKS, BLOCK_SIZE, KV_HIDDEN],
            torch.bfloat16,
            is_output=True,
            init_value=(rn([NUM_BLOCKS, BLOCK_SIZE, KV_HIDDEN], 0.02) + 0.3).to(torch.bfloat16),
        ),
    ]


def golden_rope(values: dict[str, torch.Tensor]) -> None:
    """Torch reference mirroring decode_fwd.py rope_qkv (and golden_decode_layer's rope part).

    Kernel mul order is preserved: normed = (proj*inv_rms * norm_w) * per_head_inv_rms.
    """
    inv_rms = values["inv_rms_states"].float().reshape(BATCH, 1)
    qn = values["q_norm_weight"].float().reshape(HEAD_DIM)
    kn = values["k_norm_weight"].float().reshape(HEAD_DIM)
    seq_lens = values["seq_lens"]
    slot_mapping = values["slot_mapping"]
    rope_cos = values["rope_cos"].float()
    rope_sin = values["rope_sin"].float()

    def norm_rope(proj: torch.Tensor, weight: torch.Tensor, n_heads: int) -> torch.Tensor:
        raw = (proj.float() * inv_rms).reshape(BATCH, n_heads, HEAD_DIM)
        inv_head = torch.rsqrt(raw.pow(2).sum(-1, keepdim=True) * HEAD_DIM_INV + EPS)
        normed = (raw * weight) * inv_head  # (proj*inv_rms)*norm_w, then *per-head rms
        lo, hi = normed[..., :HALF], normed[..., HALF:]
        cos = rope_cos[seq_lens.long() - 1]  # [BATCH, HEAD_DIM]
        sin = rope_sin[seq_lens.long() - 1]
        clo = cos[:, :HALF].unsqueeze(1)  # [BATCH, 1, HALF] broadcast over heads
        chi = cos[:, HALF:].unsqueeze(1)
        slo = sin[:, :HALF].unsqueeze(1)
        shi = sin[:, HALF:].unsqueeze(1)
        rot_lo = lo * clo - hi * slo
        rot_hi = hi * chi + lo * shi
        return torch.cat([rot_lo, rot_hi], dim=-1)

    values["query"] = norm_rope(values["q_proj"], qn, NUM_HEADS).to(torch.bfloat16)
    k_rot = norm_rope(values["k_proj"], kn, NUM_KV_HEADS)  # [BATCH, NUM_KV_HEADS, HEAD_DIM]
    v_heads = (values["v_proj"].float() * inv_rms).reshape(BATCH, NUM_KV_HEADS, HEAD_DIM)

    key_cache = values["key_cache"].clone()
    value_cache = values["value_cache"].clone()
    for b in range(BATCH):
        slot = int(slot_mapping[b].item())
        block = slot // BLOCK_SIZE
        off = slot % BLOCK_SIZE
        for ki in range(NUM_KV_HEADS):
            cols = slice(ki * HEAD_DIM, (ki + 1) * HEAD_DIM)
            key_cache[block, off, cols] = k_rot[b, ki].to(torch.bfloat16)
            value_cache[block, off, cols] = v_heads[b, ki].to(torch.bfloat16)
    values["key_cache"] = key_cache
    values["value_cache"] = value_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=SUPPORTED_PLATFORMS)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    specs = build_specs()
    result = run_jit(
        fn=qwen_decode_rope_only_cce,
        specs=specs,
        golden_fn=golden_rope,
        runtime_cfg={"platform": args.platform, "device_id": args.device},
        compile_only=args.compile_only or args.platform.endswith("sim"),
        rtol=5e-3,
        atol=2e-2,
        save_data=False,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
