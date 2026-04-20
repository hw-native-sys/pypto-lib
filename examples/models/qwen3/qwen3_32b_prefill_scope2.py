# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B prefill Scope 2.

Runs RoPE, KV cache update, and causal attention on precomputed Q/K/V
projections. Inputs use FP32 projections and BF16 caches; attention output is
written in BF16.
"""
from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128

# Tiling constants aligned with `qwen3_32b_prefill_tilelet`.
TOK_TILE = 32
Q_HEAD_BATCH = 8        # Q heads per attention group
Q_HEAD_PAD = 16         # padded Q rows for cube alignment
SEQ_TILE = 64           # sequence tile for attention
SB_BATCH = 64

def build_prefill_scope2_program(
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
    class PrefillScope2:
        @pl.function(type=pl.FunctionType.Opaque)
        def prefill_scope2(
            self,
            q_proj: pl.Tensor[[batch, max_seq, hidden], pl.FP32],
            k_proj: pl.Tensor[[batch, max_seq, kv_hidden], pl.FP32],
            v_proj: pl.Tensor[[batch, max_seq, kv_hidden], pl.FP32],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            attn_out: pl.Out[pl.Tensor[[batch, max_seq, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, max_seq, hidden], pl.BF16]:
            for b in pl.parallel(0, batch, 1):
                seq_len_b = pl.tensor.read(seq_lens, [b])
                tok_blocks = (seq_len_b + TOK_TILE - 1) // TOK_TILE
                for p0_idx in pl.range(tok_blocks):
                    p0 = p0_idx * TOK_TILE
                    valid_tok = pl.min(TOK_TILE, seq_len_b - p0)

                    for ti in pl.range(valid_tok):
                        pos = p0 + ti
                        ctx_len = pos + 1
                        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                        cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                        sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                        cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                        cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                        sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                        sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                        # Stage 1: K RoPE + cache update + V cache + Q RoPE + pad.
                        all_q_padded = pl.create_tensor([total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)
                        with pl.at(level=pl.Level.CORE_GROUP):
                            for gi in pl.range(total_q_groups):
                                all_q_padded = pl.assemble(
                                    all_q_padded,
                                    pl.cast(pl.full([Q_HEAD_PAD, head_dim], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                                    [gi * Q_HEAD_PAD, 0],
                                )
                        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                            for ki in pl.parallel(0, num_kv_heads, chunk=8):
                                # K RoPE + cache update.
                                kv_col = ki * head_dim
                                k_lo = pl.reshape(pl.slice(k_proj, [1, 1, half_dim], [b, pos, kv_col]), [1, half_dim])
                                k_hi = pl.reshape(pl.slice(k_proj, [1, 1, half_dim], [b, pos, kv_col + half_dim]), [1, half_dim])
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
                                        pl.reshape(pl.slice(v_proj, [1, 1, head_dim], [b, pos, ki * head_dim]), [1, head_dim]),
                                        target_type=pl.BF16,
                                    ),
                                    [cache_row, 0],
                                )
                                # Q RoPE + pad (ki == kvh since q_groups == 1).
                                q_base = ki * q_per_kv
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * head_dim
                                    q_lo = pl.reshape(pl.slice(q_proj, [1, 1, half_dim], [b, pos, q_col]), [1, half_dim])
                                    q_hi = pl.reshape(pl.slice(q_proj, [1, 1, half_dim], [b, pos, q_col + half_dim]), [1, half_dim])
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

                            all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
                            all_exp_padded = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.BF16)
                            all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                            all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32)
                            all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32)

                            # Stage 2: QK matmul for all active sb blocks.
                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    s0 = sb * SEQ_TILE
                                    cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                                    k_tile = pl.slice(k_cache, [SEQ_TILE, head_dim], [cache_row0, 0])
                                    raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                                    all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                            # Stage 3: softmax for all active sb blocks.
                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    s0 = sb * SEQ_TILE
                                    valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                    scores_valid = pl.slice(
                                        all_raw_scores, [Q_HEAD_PAD, SEQ_TILE],
                                        [sb * Q_HEAD_PAD, 0],
                                        valid_shape=[Q_HEAD_BATCH, valid_len],
                                    )
                                    scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                                    scores = pl.mul(scores_padded, attn_scale)
                                    cur_mi = pl.row_max(scores)
                                    exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                    exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                    cur_li = pl.row_sum(pl.cast(exp_scores_bf16, target_type=pl.FP32))
                                    all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                                    all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_PAD, 0])
                                    all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_PAD, 0])

                            # Stage 4: SV matmul for all active sb blocks.
                            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                                for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                                    s0 = sb * SEQ_TILE
                                    cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0
                                    exp_tile = pl.slice(all_exp_padded, [Q_HEAD_PAD, SEQ_TILE], [sb * Q_HEAD_PAD, 0])
                                    v_tile = pl.slice(v_cache, [SEQ_TILE, head_dim], [cache_row0, 0])
                                    oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                                    all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                            # Stage 5a: initialize online-softmax accumulators.
                            with pl.at(level=pl.Level.CORE_GROUP):
                                oi = pl.full([Q_HEAD_PAD, head_dim], dtype=pl.FP32, value=0.0)
                                li_flat = pl.full([1, Q_HEAD_PAD], dtype=pl.FP32, value=0.0)
                                li = pl.reshape(li_flat, [Q_HEAD_PAD, 1])
                                mi_flat = pl.full([1, Q_HEAD_PAD], dtype=pl.FP32, value=0.0)
                                mi = pl.reshape(mi_flat, [Q_HEAD_PAD, 1])

                            # Stage 5b: accumulate online softmax across active sb blocks.
                            for sb0 in pl.range(0, ctx_blocks, SB_BATCH):
                                with pl.at(level=pl.Level.CORE_GROUP):
                                    for si in pl.range(SB_BATCH):
                                        sb = sb0 + si
                                        if sb < ctx_blocks:
                                            oi_tmp_valid = pl.slice(all_oi_tmp, [Q_HEAD_PAD, head_dim], [sb * Q_HEAD_PAD, 0])
                                            cur_mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                                            cur_li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
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

                            # Finalize ctx = oi / li and extract valid Q_HEAD_BATCH rows.
                            ctx_full_gm = pl.create_tensor([Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                            with pl.at(level=pl.Level.CORE_GROUP):
                                ctx_full_gm = pl.row_expand_div(oi, li)

                            with pl.at(level=pl.Level.CORE_GROUP):
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_col = (q_base + qi) * head_dim
                                    row_bf16 = pl.cast(pl.slice(ctx_full_gm, [1, head_dim], [qi, 0]), target_type=pl.BF16)
                                    attn_row = pl.assemble(attn_row, row_bf16, [0, q_col])

                        # Write the attention row back to GM.
                        attn_out = pl.assemble(attn_out, attn_row, [b, pos, 0])

            return attn_out

    return PrefillScope2


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

    def init_q_proj():
        return torch.rand(batch, max_seq, hidden) - 0.5

    def init_k_proj():
        return torch.rand(batch, max_seq, kv_hidden) - 0.5

    def init_v_proj():
        return torch.rand(batch, max_seq, kv_hidden) - 0.5

    def init_seq_lens():
        if use_max_seq:
            return torch.full((batch,), max_seq, dtype=torch.int32)
        n_blocks = max_seq // TOK_TILE
        blocks = torch.randint(1, n_blocks + 1, (batch,), dtype=torch.int32)
        return blocks * TOK_TILE

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    return [
        TensorSpec("q_proj", [batch, max_seq, hidden], torch.float32,
                   init_value=init_q_proj),
        TensorSpec("k_proj", [batch, max_seq, kv_hidden], torch.float32,
                   init_value=init_k_proj),
        TensorSpec("v_proj", [batch, max_seq, kv_hidden], torch.float32,
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
        TensorSpec("attn_out", [batch, max_seq, hidden], torch.bfloat16, is_output=True),
    ]


def golden_prefill_scope2(tensors):
    """Vectorized PyTorch reference for Scope 2.

    Matches the kernel precision path: Q is rounded to BF16 after RoPE, QK and
    SV use BF16 inputs with FP32 accumulation, `exp_scores` takes a BF16
    round-trip before reduction, and score masking follows the tiled cache load.
    """
    import math

    import torch

    q_proj = tensors["q_proj"]     # [B, S, H], FP32
    k_proj = tensors["k_proj"]     # [B, S, KV_H], FP32
    v_proj = tensors["v_proj"]     # [B, S, KV_H], FP32
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()

    batch = q_proj.shape[0]
    hidden = q_proj.shape[2]
    kv_hidden = k_proj.shape[2]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)

    attn_out = torch.zeros(batch, max_seq, hidden, dtype=torch.float32)

    for b in range(batch):
        seq_len_b = seq_lens[b].item()
        if seq_len_b <= 0 or q_groups <= 0:
            continue

        S = seq_len_b

        # Precompute RoPE tables for all active positions.
        cos_row = rope_cos[:S, :]
        sin_row = rope_sin[:S, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        # K RoPE for all tokens and KV heads, then write to cache.
        k_row = k_proj[b, :S, :].view(S, num_kv_heads, head_dim)
        k_lo, k_hi = k_row[:, :, :half], k_row[:, :, half:]
        k_rot = torch.cat([
            k_lo * cos_lo.unsqueeze(1) - k_hi * sin_lo.unsqueeze(1),
            k_hi * cos_hi.unsqueeze(1) + k_lo * sin_hi.unsqueeze(1),
        ], dim=-1).to(torch.bfloat16)
        for ki in range(num_kv_heads):
            base = b * num_kv_heads * max_seq + ki * max_seq
            k_cache[base : base + S, :] = k_rot[:, ki, :]

        # V cache update.
        v_row = v_proj[b, :S, :].view(S, num_kv_heads, head_dim)
        for ki in range(num_kv_heads):
            base = b * num_kv_heads * max_seq + ki * max_seq
            v_cache[base : base + S, :] = v_row[:, ki, :].to(torch.bfloat16)

        # Q RoPE for all tokens and Q heads, then cast to BF16.
        q_row = q_proj[b, :S, :].view(S, num_heads, head_dim)
        q_lo, q_hi = q_row[:, :, :half], q_row[:, :, half:]
        q_rot_bf16 = torch.cat([
            q_lo * cos_lo.unsqueeze(1) - q_hi * sin_lo.unsqueeze(1),
            q_hi * cos_hi.unsqueeze(1) + q_lo * sin_hi.unsqueeze(1),
        ], dim=-1).to(torch.bfloat16)

        # Vectorized causal attention: process all S positions at once per KV head.
        max_blocks = (S + SEQ_TILE - 1) // SEQ_TILE
        padded_len = max_blocks * SEQ_TILE
        ctx_lens = torch.arange(1, S + 1)
        col_idx = torch.arange(SEQ_TILE)

        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp = q_rot_bf16[:S, q_base:q_base + Q_HEAD_BATCH, :]

                # Pad cached K/V rows to the SEQ_TILE boundary.
                cache_base = b * num_kv_heads * max_seq + kvh * max_seq
                k_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                v_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                k_padded[:S] = k_cache[cache_base:cache_base + S, :]
                v_padded[:S] = v_cache[cache_base:cache_base + S, :]

                oi = li = mi = None

                for sb in range(max_blocks):
                    s0 = sb * SEQ_TILE
                    k_tile = k_padded[s0:s0 + SEQ_TILE]
                    v_tile = v_padded[s0:s0 + SEQ_TILE]

                    # QK: [S, QB, D] @ [D, T] -> [S, QB, T]
                    raw_scores = q_grp.float() @ k_tile.float().T

                    # Causal + fillpad mask.
                    valid_lens = torch.clamp(ctx_lens - s0, min=0, max=SEQ_TILE)
                    mask = col_idx.unsqueeze(0) < valid_lens.unsqueeze(1)
                    raw_scores[~mask.unsqueeze(1).expand_as(raw_scores)] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    # Online softmax: row_max -> exp -> BF16 round-trip -> row_sum.
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_bf16.float().sum(dim=-1, keepdim=True)

                    # SV: [S, QB, T] @ [T, D] -> [S, QB, D]
                    oi_tmp = exp_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi, li, mi = oi_tmp, cur_li, cur_mi
                    else:
                        active = valid_lens > 0
                        if active.any():
                            a = active
                            mi_new = torch.maximum(mi[a], cur_mi[a])
                            alpha = torch.exp(mi[a] - mi_new)
                            beta = torch.exp(cur_mi[a] - mi_new)
                            oi[a] = oi[a] * alpha + oi_tmp[a] * beta
                            li[a] = alpha * li[a] + beta * cur_li[a]
                            mi[a] = mi_new

                # Normalize and write all Q_HEAD_BATCH heads at once.
                ctx = oi / li
                attn_out[b, :S, q_base * head_dim:(q_base + Q_HEAD_BATCH) * head_dim] = \
                    ctx.reshape(S, Q_HEAD_BATCH * head_dim)

    tensors["attn_out"][:] = attn_out.to(torch.bfloat16)


def compile_and_run(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    use_max_seq: bool = False,
    platform: str = "a2a3",
    device_id: int = 0,
    dump_passes: bool = True,
    runtime_profiling: bool = False,
    runtime_dir: str | None = None,
):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from golden import RunConfig, run

    program = build_prefill_scope2_program(
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

    golden_data = str(Path(runtime_dir) / "data") if runtime_dir else None

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden_fn=golden_prefill_scope2,
        runtime_dir=runtime_dir,
        golden_data=golden_data,
        config=RunConfig(
            rtol=2e-3,
            atol=2e-3,
            compile=dict(dump_passes=dump_passes),
            runtime=dict(
                platform=platform,
                device_id=device_id,
                runtime_profiling=runtime_profiling,
            ),
        ),
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    parser.add_argument(
        "--runtime-dir", type=str, default=None, help="reuse a previous build_output dir and golden data"
    )
    parser.add_argument("--max-seq", action="store_true", default=False, help="set all seq_lens to MAX_SEQ")
    args = parser.parse_args()

    result = compile_and_run(
        platform=args.platform,
        device_id=args.device,
        use_max_seq=args.max_seq,
        runtime_profiling=args.runtime_profiling,
        runtime_dir=args.runtime_dir,
    )
    if not result.passed:
        if result.error:
            print(f"Result: {result.error}")
        raise SystemExit(1)
