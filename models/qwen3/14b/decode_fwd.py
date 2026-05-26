# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B full-layer decode forward.

Each layer runs the same fused decode body:
Scope 1:
  1. RMSNorm of input hidden states
  2. Q/K/V projection via matmul

Per-head q_norm / k_norm

Scope 2:
  1. K RoPE + paged cache write, V paged cache write, Q RoPE + pad
  2. QK matmul
  3. Softmax
  4. SV matmul
  5. Online-softmax accumulation + final normalisation

Scope 3:
  1. Output projection: attn_out × wo
  2. Residual addition with hidden_states
  3. Post-attention RMSNorm
  4. MLP: gate/up projections, SiLU activation, down projection
  5. Final residual addition

Final head is applied once after all decode layers.
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl

from config import (
    BATCH,
    BATCH_TILE,
    BLOCK_SIZE,
    BLOCK_TABLE_FLAT_DYN,
    HEAD_DIM,
    HIDDEN,
    INPUT_PROJ_K_CHUNK,
    INTERMEDIATE,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LM_HEAD_K_CHUNK,
    MAX_BLOCKS_PER_SEQ,
    MAX_SEQ,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_HEAD_BATCH,
    Q_HEAD_PAD,
    ROPE_SEQ_DYN,
    USER_BATCH_DYN,
    VOCAB,
    VOCAB_CHUNK,
)
from decode_layer import decode_layer
from rms_lm_head import rms_lm_head


@pl.jit
def decode_fwd(
    hidden_states: pl.Tensor[[USER_BATCH_DYN, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    rope_cos: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[ROPE_SEQ_DYN, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    hidden_states.bind_dynamic(0, USER_BATCH_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    slot_mapping.bind_dynamic(0, USER_BATCH_DYN)
    out.bind_dynamic(0, USER_BATCH_DYN)
    block_table.bind_dynamic(0, BLOCK_TABLE_FLAT_DYN)
    rope_cos.bind_dynamic(0, ROPE_SEQ_DYN)
    rope_sin.bind_dynamic(0, ROPE_SEQ_DYN)
    k_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)

    user_batch = pl.tensor.dim(hidden_states, 0)
    batch_padded = BATCH
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    hidden_blocks = HIDDEN // K_CHUNK

    current_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
        cur_valid = pl.min(BATCH_TILE, user_batch - b0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for kb in pl.range(hidden_blocks):
                copy_k0 = kb * K_CHUNK
                hidden_chunk = pl.slice(
                    hidden_states,
                    [BATCH_TILE, K_CHUNK],
                    [b0, copy_k0],
                    valid_shape=[cur_valid, K_CHUNK],
                )
                current_hidden = pl.assemble(current_hidden, hidden_chunk, [b0, copy_k0])

    for layer_idx in pl.range(num_layers_actual):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        current_hidden = decode_layer(
            current_hidden,
            input_rms_weight,
            wq,
            wk,
            wv,
            q_norm_weight,
            k_norm_weight,
            seq_lens,
            block_table,
            slot_mapping,
            rope_cos,
            rope_sin,
            k_cache,
            v_cache,
            wo,
            post_rms_weight,
            w_gate,
            w_up,
            w_down,
            next_hidden,
            layer_idx,
        )

    out = rms_lm_head(current_hidden, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    num_layers: int = NUM_LAYERS,
    vocab_size: int = VOCAB,
):
    import sys
    from pathlib import Path

    import torch

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from golden import TensorSpec

    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    vocab = vocab_size
    num_blocks = batch * MAX_BLOCKS_PER_SEQ
    layer_cache_rows = num_blocks * num_kv_heads * BLOCK_SIZE
    cache_rows = num_layers * layer_cache_rows
    synthetic_proj_scale = 0.5

    seq_lens_seed = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_input_rms_weight():
        return torch.rand(num_layers, hidden_size) - 0.5

    def init_wq():
        return torch.rand(num_layers * hidden_size, hidden_size) / hidden_size ** 0.5

    def init_wk():
        return torch.rand(num_layers * hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_wv():
        return synthetic_proj_scale * torch.rand(num_layers * hidden_size, kv_hidden) / hidden_size ** 0.5

    def init_q_norm_weight():
        return torch.ones(num_layers, head_dim)

    def init_k_norm_weight():
        return torch.ones(num_layers, head_dim)

    def init_seq_lens():
        return seq_lens_seed.clone()

    def init_block_table():
        return torch.arange(num_blocks, dtype=torch.int32)

    def init_slot_mapping():
        slots = torch.empty(batch, dtype=torch.int32)
        for b in range(batch):
            pos = int(seq_lens_seed[b].item()) - 1
            logical_block = pos // BLOCK_SIZE
            page_offset = pos % BLOCK_SIZE
            phys_block = b * MAX_BLOCKS_PER_SEQ + logical_block
            slots[b] = phys_block * BLOCK_SIZE + page_offset
        return slots

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return synthetic_proj_scale * (torch.rand(cache_rows, head_dim) - 0.5)

    def init_wo():
        return synthetic_proj_scale * (torch.rand(num_layers * hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(num_layers, hidden_size)

    def init_w_gate():
        return synthetic_proj_scale * (torch.rand(num_layers * hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return synthetic_proj_scale * (torch.rand(num_layers * hidden_size, inter) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return synthetic_proj_scale * (torch.rand(num_layers * inter, hidden_size) - 0.5) / inter ** 0.5

    def init_final_norm_weight():
        return torch.ones(1, hidden_size)

    def init_lm_head_weight():
        return synthetic_proj_scale * (torch.rand(vocab, hidden_size) - 0.5) / hidden_size ** 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [num_layers, hidden_size], torch.float32, init_value=init_input_rms_weight),
        TensorSpec("wq", [num_layers * hidden_size, hidden_size], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [num_layers * hidden_size, kv_hidden], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [num_layers * hidden_size, kv_hidden], torch.bfloat16, init_value=init_wv),
        TensorSpec("q_norm_weight", [num_layers, head_dim], torch.float32, init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [num_layers, head_dim], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("block_table", [batch * MAX_BLOCKS_PER_SEQ], torch.int32, init_value=init_block_table),
        TensorSpec("slot_mapping", [batch], torch.int32, init_value=init_slot_mapping),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32, init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32, init_value=init_rope_sin),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16, init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16, init_value=init_v_cache),
        TensorSpec("wo", [num_layers * hidden_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [num_layers, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [num_layers * hidden_size, inter], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [num_layers * hidden_size, inter], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [num_layers * inter, hidden_size], torch.bfloat16, init_value=init_w_down),
        TensorSpec("final_norm_weight", [1, hidden_size], torch.float32, init_value=init_final_norm_weight),
        TensorSpec("lm_head_weight", [vocab, hidden_size], torch.bfloat16, init_value=init_lm_head_weight),
        TensorSpec("out", [batch, vocab], torch.float32, is_output=True),
    ]


def golden_decode_fwd(tensors):
    """PyTorch reference for the full-layer Qwen3-14B decode program."""
    import math

    import torch

    hidden = tensors["hidden_states"].clone()
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    final_norm_weight = tensors["final_norm_weight"]
    lm_head_weight = tensors["lm_head_weight"]

    batch = hidden.shape[0]
    hidden_size = hidden.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_layers = input_rms_weight.shape[0]
    kv_hidden = wk.shape[1]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    intermediate_size = w_gate.shape[1]
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-6
    max_ctx_blocks = MAX_BLOCKS_PER_SEQ
    layer_cache_rows = batch * MAX_BLOCKS_PER_SEQ * num_kv_heads * BLOCK_SIZE

    def tiled_lm_head(lhs, rhs_t, k_chunk, vocab_chunk):
        out = torch.zeros(lhs.shape[0], rhs_t.shape[0], dtype=torch.float32)
        for k0 in range(0, lhs.shape[1], k_chunk):
            out = out + lhs[:, k0 : k0 + k_chunk].float() @ rhs_t[:, k0 : k0 + k_chunk].float().T
        return out

    for layer_idx in range(num_layers):
        layer_hidden_base = layer_idx * hidden_size
        layer_inter_base = layer_idx * intermediate_size
        layer_cache_base = layer_idx * layer_cache_rows
        layer_wq = wq[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wk = wk[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wv = wv[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_wo = wo[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_gate = w_gate[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_up = w_up[layer_hidden_base : layer_hidden_base + hidden_size, :]
        layer_w_down = w_down[layer_inter_base : layer_inter_base + intermediate_size, :]

        q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
        k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
        v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)

        for b0 in range(0, batch, BATCH_TILE):
            b_end = min(b0 + BATCH_TILE, batch)
            x_tile = hidden[b0:b_end, :].float()
            sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
            for k0 in range(0, hidden_size, INPUT_PROJ_K_CHUNK):
                x_chunk = x_tile[:, k0 : k0 + INPUT_PROJ_K_CHUNK]
                sq_sum = sq_sum + (x_chunk ** 2).sum(dim=-1, keepdim=True)
            normed = (
                x_tile
                * torch.rsqrt(sq_sum / hidden_size + eps)
                * input_rms_weight[layer_idx : layer_idx + 1, :].float()
            ).bfloat16()
            q_proj[b0:b_end, :] = (normed.float() @ layer_wq.float()).float()
            k_proj[b0:b_end, :] = (normed.float() @ layer_wk.float()).float()
            v_proj[b0:b_end, :] = (normed.float() @ layer_wv.float()).float()

        attn_out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
        for b in range(batch):
            ctx_len = int(seq_lens[b].item())
            pos = ctx_len - 1
            ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            cos_row = rope_cos[pos : pos + 1, :]
            sin_row = rope_sin[pos : pos + 1, :]
            cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
            sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

            k_heads = k_proj[b].view(num_kv_heads, head_dim)
            k_heads = (
                k_heads
                * torch.rsqrt(k_heads.pow(2).mean(dim=-1, keepdim=True) + eps)
                * k_norm_weight[layer_idx : layer_idx + 1, :].float()
            )
            k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
            k_rot = torch.cat(
                [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
                dim=-1,
            )

            slot = int(slot_mapping[b].item())
            slot_block = slot // BLOCK_SIZE
            slot_offset = slot % BLOCK_SIZE
            for ki in range(num_kv_heads):
                cache_row = layer_cache_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
                v_cache[cache_row, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

            q_heads = q_proj[b].view(num_heads, head_dim)
            q_heads = (
                q_heads
                * torch.rsqrt(q_heads.pow(2).mean(dim=-1, keepdim=True) + eps)
                * q_norm_weight[layer_idx : layer_idx + 1, :].float()
            )
            q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
            q_rot = torch.cat(
                [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
                dim=-1,
            )

            attn_row_padded = torch.zeros(1, total_q_groups * Q_HEAD_PAD * head_dim, dtype=torch.bfloat16)
            for kvh in range(num_kv_heads):
                for qg in range(q_groups):
                    gi = kvh * q_groups + qg
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)
                    oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                    li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                    mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                    for sb in range(ctx_blocks):
                        s0 = sb * BLOCK_SIZE
                        valid_len = min(BLOCK_SIZE, ctx_len - s0)
                        pbid = int(block_table[b * max_ctx_blocks + sb].item())
                        cache_row0 = layer_cache_base + (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                        k_tile = k_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]
                        v_tile = v_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]
                        raw_scores = q_grp_bf16.float() @ k_tile.float().T
                        if valid_len < BLOCK_SIZE:
                            raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                        scores = raw_scores * scale
                        cur_mi = scores.max(dim=-1, keepdim=True).values
                        exp_scores = torch.exp(scores - cur_mi)
                        exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                        cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
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
                    ctx_flat_padded_bf16 = torch.zeros(1, Q_HEAD_PAD * head_dim, dtype=torch.bfloat16)
                    ctx_flat_padded_bf16[:, : Q_HEAD_BATCH * head_dim] = ctx.reshape(1, -1).to(torch.bfloat16)
                    attn_row_padded[
                        :,
                        gi * Q_HEAD_PAD * head_dim : (gi + 1) * Q_HEAD_PAD * head_dim,
                    ] = ctx_flat_padded_bf16

            attn_row = torch.zeros(1, hidden_size, dtype=torch.bfloat16)
            for kvh in range(num_kv_heads):
                for qg in range(q_groups):
                    gi = kvh * q_groups + qg
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    attn_row[
                        :,
                        q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim,
                    ] = attn_row_padded[
                        :,
                        gi * Q_HEAD_PAD * head_dim : gi * Q_HEAD_PAD * head_dim + Q_HEAD_BATCH * head_dim,
                    ]
            attn_out[b : b + 1, :] = attn_row

        o_proj = attn_out.float() @ layer_wo.float()
        resid1 = o_proj + hidden.float()
        normed_bf16 = (
            resid1
            * torch.rsqrt(resid1.pow(2).mean(dim=-1, keepdim=True) + eps)
            * post_rms_weight[layer_idx : layer_idx + 1, :].float()
        ).bfloat16()
        gate = normed_bf16.float() @ layer_w_gate.float()
        up = normed_bf16.float() @ layer_w_up.float()
        mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
        down = mlp_bf16.float() @ layer_w_down.float()
        hidden = (down + resid1).bfloat16()

    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    final_normed = (hidden.float() * torch.rsqrt(variance + eps) * final_norm_weight.float()).bfloat16()
    tensors["out"][:] = tiled_lm_head(final_normed, lm_head_weight, LM_HEAD_K_CHUNK, VOCAB_CHUNK)


def make_pass_rate_compare(threshold: float):
    """Build a compare_fn that passes when >= `threshold` of elements are
    close (under the run's atol/rtol). Used for the BF16 long-tail on
    multi-layer decode: tolerates a small fraction of 1-2 ULP outliers
    while still catching systematic bias (which would tank the pass rate).
    """
    def cmp(actual, expected, *, rtol, atol, **_):
        import torch

        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        rate = close.float().mean().item()
        n_fail = int((~close).sum().item())
        ok = rate >= threshold
        msg = (
            f"    pass_rate={rate:.6f} (threshold {threshold:.6f}), "
            f"{n_fail}/{actual.numel()} mismatched  rtol={rtol} atol={atol}"
        )
        if not ok:
            flat_a = actual.flatten()
            flat_e = expected.flatten()
            idx = torch.where(~close.flatten())[0][:5]
            lines = [
                f"    [{i.item()}] actual={flat_a[i].item()}, expected={flat_e[i].item()}"
                for i in idx
            ]
            msg += "\n    first {} mismatches:\n".format(idx.numel()) + "\n".join(lines)
        return ok, msg

    cmp.__name__ = f"pass_rate>={threshold:.4f}"
    return cmp


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a5"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=BATCH)
    parser.add_argument("--max-seq", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--pass-rate", type=float, default=0.98,
                        help="Fraction of `out` elements that must satisfy atol/rtol. "
                             "Default 0.98 is sized for the 40-layer BF16 ULP long-tail at "
                             "the fixed default seed (measured pass_rate=0.9898), leaving "
                             "~0.9pp margin. Combined with --seed (fixed by default), CI "
                             "is deterministic; flake from seed-to-seed variance is avoided.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for input tensor generation. Fixed by default "
                             "so pass_rate measurements are reproducible across runs. "
                             "Pass an explicit value to stress-test other input distributions.")
    args = parser.parse_args()

    import torch
    torch.manual_seed(args.seed)

    if args.max_seq > MAX_SEQ:
        raise ValueError(
            f"decode_fwd JIT inline path currently supports max_seq <= {MAX_SEQ}"
        )

    result = run_jit(
        fn=decode_fwd,
        specs=build_tensor_specs(
            batch=args.batch,
            max_seq=args.max_seq,
            num_layers=args.num_layers,
        ),
        golden_fn=golden_decode_fwd,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=5e-3,
        atol=5e-3,
        compare_fn={"out": make_pass_rate_compare(args.pass_rate)},
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = ["decode_fwd", "build_tensor_specs", "golden_decode_fwd",
           "make_pass_rate_compare"]
