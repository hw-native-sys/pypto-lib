# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B decode Scope 2 — RoPE + KV cache update + grouped-query attention.
  1. K RoPE + cache write, V cache write, Q RoPE + pad
  2. QK matmul
  3. Softmax  
  4. SV matmul
  5. Online-softmax accumulation + final normalisation

Input projections are FP32; cos/sin tables are FP32; KV caches are BF16.
Output attention is BF16.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128

# Tiling constants (aligned to qwen3_32b_decode_tilelet).
Q_HEAD_BATCH = 8        # Q heads batched per attention group
Q_HEAD_PAD = 16         # padded Q rows for cube fractal alignment
SEQ_TILE = 64           # sequence tile for attention loop
SB_BATCH = 64

def build_qwen3_scope2_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    q_per_kv = num_heads // num_kv_heads
    cache_rows = batch * num_kv_heads * max_seq
    half_dim = head_dim // 2
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)
    max_ctx_blocks = (max_seq + SEQ_TILE - 1) // SEQ_TILE

    @pl.program
    class Qwen3Scope2:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_scope2(
            self,
            q_proj: pl.Tensor[[batch, hidden], pl.FP32],
            k_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            v_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            attn_out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            for b in pl.range(batch):
                ctx_len = pl.tensor.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                # Workaround
                all_q_padded = pl.create_tensor([total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)
                with pl.incore():
                    for gi in pl.range(total_q_groups):
                        all_q_padded = pl.assemble(
                            all_q_padded,
                            pl.cast(pl.full([Q_HEAD_PAD, head_dim], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                            [gi * Q_HEAD_PAD, 0],
                        )
                
                # Stage 1: K RoPE + cache update + V cache + Q RoPE + pad.
                with pl.auto_incore():
                    for ki in pl.parallel(0, num_kv_heads, chunk=8):
                        # K RoPE + cache update.
                        kv_col = ki * head_dim
                        k_lo = pl.slice(k_proj, [1, half_dim], [b, kv_col])
                        k_hi = pl.slice(k_proj, [1, half_dim], [b, kv_col + half_dim])
                        rot_lo = pl.sub(
                            pl.col_expand_mul(k_lo, cos_lo),
                            pl.col_expand_mul(k_hi, sin_lo),
                        )
                        rot_hi = pl.add(
                            pl.col_expand_mul(k_hi, cos_hi),
                            pl.col_expand_mul(k_lo, sin_hi),
                        )
                        cache_row = b * num_kv_heads * max_seq + ki * max_seq + pos
                        k_cache = pl.assemble(
                            k_cache,
                            pl.cast(rot_lo, target_type=pl.BF16),
                            [cache_row, 0],
                        )
                        k_cache = pl.assemble(
                            k_cache,
                            pl.cast(rot_hi, target_type=pl.BF16),
                            [cache_row, half_dim],
                        )
                        # V cache update.
                        v_cache = pl.assemble(
                            v_cache,
                            pl.cast(
                                pl.slice(v_proj, [1, head_dim], [b, ki * head_dim]),
                                target_type=pl.BF16,
                            ),
                            [cache_row, 0],
                        )
                        # Q RoPE + pad (ki == kvh since q_groups == 1).
                        q_base = ki * q_per_kv
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * head_dim
                            q_lo = pl.slice(q_proj, [1, half_dim], [b, q_col])
                            q_hi = pl.slice(q_proj, [1, half_dim], [b, q_col + half_dim])
                            rot_lo_bf16 = pl.cast(
                                pl.sub(
                                    pl.col_expand_mul(q_lo, cos_lo),
                                    pl.col_expand_mul(q_hi, sin_lo),
                                ),
                                target_type=pl.BF16,
                            )
                            rot_hi_bf16 = pl.cast(
                                pl.add(
                                    pl.col_expand_mul(q_hi, cos_hi),
                                    pl.col_expand_mul(q_lo, sin_hi),
                                ),
                                target_type=pl.BF16,
                            )
                            all_q_padded = pl.assemble(all_q_padded, rot_lo_bf16, [ki * Q_HEAD_PAD + qi, 0])
                            all_q_padded = pl.assemble(all_q_padded, rot_hi_bf16, [ki * Q_HEAD_PAD + qi, half_dim])

                attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                for gi in pl.range(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_padded = pl.slice(all_q_padded, [Q_HEAD_PAD, head_dim], [gi * Q_HEAD_PAD, 0])

                    # Workaround
                    all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                    all_exp_padded = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                    all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                    all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH, 1], dtype=pl.FP32)
                    all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_BATCH, 1], dtype=pl.FP32)
                    for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                        with pl.incore():
                            for si in pl.range(SB_BATCH):
                                sb = sb0 + si
                                if sb < ctx_blocks:
                                    all_raw_scores = pl.assemble(
                                        all_raw_scores,
                                        pl.full([Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32, value=0.0),
                                        [sb * Q_HEAD_PAD, 0],
                                    )
                                    all_exp_padded = pl.assemble(
                                        all_exp_padded,
                                        pl.cast(pl.full([Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                                        [sb * Q_HEAD_PAD, 0],
                                    )
                                    all_oi_tmp = pl.assemble(
                                        all_oi_tmp,
                                        pl.full([Q_HEAD_PAD, head_dim], dtype=pl.FP32, value=0.0),
                                        [sb * Q_HEAD_PAD, 0],
                                    )
                                    mi_init_flat = pl.full([1, Q_HEAD_BATCH], dtype=pl.FP32, value=0.0)
                                    all_cur_mi = pl.assemble(
                                        all_cur_mi,
                                        pl.reshape(mi_init_flat, [Q_HEAD_BATCH, 1]),
                                        [sb * Q_HEAD_BATCH, 0],
                                    )
                                    li_init_flat = pl.full([1, Q_HEAD_BATCH], dtype=pl.FP32, value=0.0)
                                    all_cur_li = pl.assemble(
                                        all_cur_li,
                                        pl.reshape(li_init_flat, [Q_HEAD_BATCH, 1]),
                                        [sb * Q_HEAD_BATCH, 0],
                                    )

                    # Stage 2: QK matmul for all active sb blocks.
                    for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                        with pl.incore():
                            for si in pl.range(SB_BATCH):
                                sb = sb0 + si
                                if sb < ctx_blocks:
                                    s0 = sb * SEQ_TILE
                                    cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                                    k_tile = pl.slice(
                                        k_cache,
                                        [SEQ_TILE, head_dim],
                                        [cache_row0, 0],
                                    )
                                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                                    all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                    # Stage 3: softmax for all active sb blocks.
                    for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                        with pl.incore():
                            for si in pl.range(SB_BATCH):
                                sb = sb0 + si
                                if sb < ctx_blocks:
                                    s0 = sb * SEQ_TILE
                                    valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                    scores_valid = pl.slice(
                                        all_raw_scores,
                                        [Q_HEAD_BATCH, SEQ_TILE],
                                        [sb * Q_HEAD_PAD, 0],
                                        valid_shape=[Q_HEAD_BATCH, valid_len],
                                    )
                                    scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                                    scores = pl.mul(scores_padded, attn_scale)
                                    cur_mi = pl.row_max(scores)
                                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                    exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                                    cur_li = pl.row_sum(exp_scores_fp32)
                                    all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                                    all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_BATCH, 0])
                                    all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_BATCH, 0])

                    # Stage 4: SV matmul for all active sb blocks.
                    for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                        with pl.incore():
                            for si in pl.range(SB_BATCH):
                                sb = sb0 + si
                                if sb < ctx_blocks:
                                    s0 = sb * SEQ_TILE
                                    cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                                    exp_tile = pl.slice(
                                        all_exp_padded,
                                        [Q_HEAD_PAD, SEQ_TILE],
                                        [sb * Q_HEAD_PAD, 0],
                                    )
                                    v_tile = pl.slice(
                                        v_cache,
                                        [SEQ_TILE, head_dim],
                                        [cache_row0, 0],
                                    )
                                    oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                    # Workaround
                    with pl.incore():
                        oi = pl.full([Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0)
                        li_flat = pl.full([1, Q_HEAD_BATCH], dtype=pl.FP32, value=0.0)
                        li = pl.reshape(li_flat, [Q_HEAD_BATCH, 1])
                        mi_flat = pl.full([1, Q_HEAD_BATCH], dtype=pl.FP32, value=0.0)
                        mi = pl.reshape(mi_flat, [Q_HEAD_BATCH, 1])

                    # Stage 5: online softmax accumulation for active sb blocks.
                    for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                        with pl.incore():
                            for si in pl.range(SB_BATCH):
                                sb = sb0 + si
                                if sb < ctx_blocks:
                                    oi_tmp_valid = pl.slice(all_oi_tmp, [Q_HEAD_BATCH, head_dim], [sb * Q_HEAD_PAD, 0])
                                    cur_mi = pl.slice(all_cur_mi, [Q_HEAD_BATCH, 1], [sb * Q_HEAD_BATCH, 0])
                                    cur_li = pl.slice(all_cur_li, [Q_HEAD_BATCH, 1], [sb * Q_HEAD_BATCH, 0])
                                    if sb == 0:
                                        oi = oi_tmp_valid
                                        li = cur_li
                                        mi = cur_mi
                                    else:
                                        mi_new = pl.maximum(mi, cur_mi)
                                        alpha = pl.exp(pl.sub(mi, mi_new))
                                        beta = pl.exp(pl.sub(cur_mi, mi_new))
                                        li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                                        oi = pl.add(pl.row_expand_mul(oi, alpha),
                                                    pl.row_expand_mul(oi_tmp_valid, beta))
                                        mi = mi_new

                    # Stage 6: normalise and write back one Q-head batch.
                    with pl.incore():
                        ctx = pl.row_expand_div(oi, li)
                        ctx_flat = pl.reshape(ctx, [1, Q_HEAD_BATCH * head_dim])
                        attn_row = pl.assemble(
                            attn_row, pl.cast(ctx_flat, target_type=pl.BF16), [0, q_base * head_dim],
                        )

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            return attn_out

    return Qwen3Scope2


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    use_max_seq: bool = False,
):
    import torch
    from pypto.runtime import TensorSpec

    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    cache_rows = batch * num_kv_heads * max_seq

    def init_seq_lens():
        if use_max_seq:
            return torch.full((batch,), max_seq, dtype=torch.int32)
        return torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    def init_q_proj():
        return torch.rand(batch, hidden) - 0.5

    def init_k_proj():
        return torch.rand(batch, kv_hidden) - 0.5

    def init_v_proj():
        return torch.rand(batch, kv_hidden) - 0.5

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    return [
        TensorSpec("q_proj", [batch, hidden], torch.float32,
                   init_value=init_q_proj),
        TensorSpec("k_proj", [batch, kv_hidden], torch.float32,
                   init_value=init_k_proj),
        TensorSpec("v_proj", [batch, kv_hidden], torch.float32,
                   init_value=init_v_proj),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32,
                   init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32,
                   init_value=init_rope_sin),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_v_cache),
        TensorSpec("attn_out", [batch, hidden], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_scope2(tensors, params):
    """PyTorch reference matching kernel BF16 precision path.

    Simulates the kernel's tiled online-softmax with BF16 matmuls:
      - Q cast to BF16 after RoPE (matching kernel QK matmul input).
      - QK/SV matmuls use BF16 inputs with FP32 accumulation.
      - BF16 round-trip on exp_scores before row_sum (matching kernel).
      - Full SEQ_TILE K/V loads with fillpad masking on scores.
    """
    import math

    import torch

    q_proj = tensors["q_proj"]              # already FP32
    k_proj = tensors["k_proj"]              # already FP32
    v_proj = tensors["v_proj"]              # FP32, cast to BF16 for cache writes
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()  # BF16, cloned to avoid side effects
    v_cache = tensors["v_cache"].clone()

    batch = q_proj.shape[0]
    hidden = q_proj.shape[1]
    kv_hidden = k_proj.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    attn_out = torch.zeros(batch, hidden, dtype=torch.float32)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        # K RoPE: all KV heads together.
        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_lo, k_hi = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [
                k_lo * cos_lo - k_hi * sin_lo,
                k_hi * cos_hi + k_lo * sin_hi,
            ],
            dim=-1,
        )

        # Update caches.
        for ki in range(num_kv_heads):
            cr = b * num_kv_heads * max_seq + ki * max_seq + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        # Q RoPE: all Q heads together.
        q_heads = q_proj[b].view(num_heads, head_dim)
        q_lo, q_hi = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [
                q_lo * cos_lo - q_hi * sin_lo,
                q_hi * cos_hi + q_lo * sin_hi,
            ],
            dim=-1,
        )

        # Grouped-query attention (tiled online softmax, matching kernel BF16 path).
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp = q_rot[q_base : q_base + Q_HEAD_BATCH, :]
                # Match kernel: Q cast to BF16 for QK matmul.
                q_grp_bf16 = q_grp.to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cb = b * num_kv_heads * max_seq + kvh * max_seq + s0

                    # Load full SEQ_TILE K/V tiles as BF16 (matching kernel).
                    k_tile = k_cache[cb : cb + SEQ_TILE, :]
                    v_tile = v_cache[cb : cb + SEQ_TILE, :]

                    # QK matmul: BF16 * BF16 → FP32.
                    raw_scores = q_grp_bf16.float() @ k_tile.float().T

                    # Fillpad invalid positions before scale.
                    if valid_len < SEQ_TILE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    # Online softmax: row_max → exp → BF16 round-trip → row_sum.
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                    # SV matmul: BF16 * BF16 → FP32.
                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                for qi in range(Q_HEAD_BATCH):
                    qh = q_base + qi
                    attn_out[b, qh * head_dim : (qh + 1) * head_dim] = ctx[qi]

    tensors["attn_out"][:] = attn_out.to(torch.bfloat16)


def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    use_max_seq: bool = False,
    platform: str = "a5",
    device_id: int = 0,
    dump_passes: bool = True,
    enable_profiling: bool = False,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    backend = BackendType.Ascend950 if platform.startswith("a5") else BackendType.Ascend910B

    program = build_qwen3_scope2_program(
        batch=batch,
        max_seq=max_seq,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq=max_seq,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        use_max_seq=use_max_seq,
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=golden_qwen3_scope2,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=1e-3,
            atol=1e-3,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=backend,
            enable_profiling=enable_profiling,
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False,
                        help="set all seq_lens to MAX_SEQ (default: random)")
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        use_max_seq=args.max_seq,
        enable_profiling=args.enable_profiling,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
