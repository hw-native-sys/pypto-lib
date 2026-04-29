# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP decode front scope2 (indexer prepare).

This file prepares the indexer outputs consumed by scope3:
- q = wq_b(qr), split as q_pe/q_nope with q_pe first
- k = LayerNorm(wk(hidden_states)), split as k_pe/k_nope with k_pe first
- apply non-interleaved RoPE to q_pe and k_pe
- compute per-index-head weights
- quantize q_idx_full and current-token k_idx to row-wise INT8
- update k_cache_idx_i8 and k_cache_idx_scale at the current decode position
"""

import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
Q_LORA_RANK = 1536
QK_ROPE_HEAD_DIM = 64

INDEX_HEADS = 64
INDEX_HEAD_DIM = 128

EPS = 1e-6
INT8_GROUP_SIZE = 128
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
INT8_SCALE_PACK = 8

CACHE_ROWS = BATCH * MAX_SEQ

K_CHUNK = 128
LORA_CHUNK = 128
IDX_OUT_CHUNK = 128
WEIGHTS_OUT_CHUNK = 64

if INDEX_HEAD_DIM != INT8_GROUP_SIZE:
    raise ValueError(
        f"INT8 quant path expects INDEX_HEAD_DIM == {INT8_GROUP_SIZE}, "
        f"got {INDEX_HEAD_DIM}"
    )

INDEX_Q_OUT = INDEX_HEADS * INDEX_HEAD_DIM
INDEX_Q_ROWS = BATCH * INDEX_HEADS

HIDDEN_BLOCKS = (HIDDEN + K_CHUNK - 1) // K_CHUNK
QR_BLOCKS = (Q_LORA_RANK + LORA_CHUNK - 1) // LORA_CHUNK
IDX_OUT_BLOCKS = (INDEX_Q_OUT + IDX_OUT_CHUNK - 1) // IDX_OUT_CHUNK
WK_OUT_BLOCKS = (INDEX_HEAD_DIM + IDX_OUT_CHUNK - 1) // IDX_OUT_CHUNK
WEIGHTS_BLOCKS = (INDEX_HEADS + WEIGHTS_OUT_CHUNK - 1) // WEIGHTS_OUT_CHUNK
INDEX_HEAD_DIM_INV = 1.0 / INDEX_HEAD_DIM
WEIGHT_SCALE = (INDEX_HEADS ** -0.5) * (INDEX_HEAD_DIM ** -0.5)


def build_deepseek_v3_2_decode_front_indexer_program():
    @pl.program
    class DeepSeekV32DecodeFrontIndexer:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_indexer(
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            qr: pl.Tensor[[BATCH, Q_LORA_RANK], pl.BF16],
            wq_b_idx: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
            wk_idx: pl.Tensor[[HIDDEN, INDEX_HEAD_DIM], pl.BF16],
            k_norm_weight: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            k_norm_bias: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            weights_proj: pl.Tensor[[HIDDEN, INDEX_HEADS], pl.FP32],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            k_cache_idx_i8: pl.Tensor[[CACHE_ROWS, INDEX_HEAD_DIM], pl.INT8],
            k_cache_idx_scale: pl.Tensor[[BATCH, MAX_SEQ], pl.FP32],
            q_idx_full_i8_out: pl.Tensor[[INDEX_Q_ROWS, INDEX_HEAD_DIM], pl.INT8],
            q_idx_scale_heads_out: pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32],
            weights_out: pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32],
        ) -> tuple[
            pl.Tensor[[CACHE_ROWS, INDEX_HEAD_DIM], pl.INT8],
            pl.Tensor[[BATCH, MAX_SEQ], pl.FP32],
            pl.Tensor[[INDEX_Q_ROWS, INDEX_HEAD_DIM], pl.INT8],
            pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32],
            pl.Tensor[[BATCH, INDEX_HEADS], pl.FP32],
        ]:
            q_idx_full = pl.create_tensor([BATCH, INDEX_Q_OUT], dtype=pl.BF16)
            k_idx = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)
            q_idx_full_i8 = pl.create_tensor([INDEX_Q_ROWS, INDEX_HEAD_DIM], dtype=pl.INT8)
            k_idx_i8 = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.INT8)
            k_idx_scale = pl.create_tensor([BATCH, INT8_SCALE_PACK], dtype=pl.FP32)
            weights = pl.create_tensor([BATCH, INDEX_HEADS], dtype=pl.FP32)
            q_idx_scale_heads = pl.create_tensor([BATCH, INDEX_HEADS], dtype=pl.FP32)

            # Stage 0: q_idx_full = wq_b_idx(qr), shaped as [B, INDEX_HEADS, INDEX_HEAD_DIM].
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for ob in pl.parallel(0, IDX_OUT_BLOCKS, 1, chunk=8):
                    q0 = ob * IDX_OUT_CHUNK
                    q_acc = pl.full([BATCH, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(QR_BLOCKS):
                        k0 = kb * LORA_CHUNK
                        qr_chunk = pl.slice(qr, [BATCH, LORA_CHUNK], [0, k0])
                        wq_chunk = pl.slice(wq_b_idx, [LORA_CHUNK, IDX_OUT_CHUNK], [k0, q0])
                        q_acc = pl.add(q_acc, pl.matmul(qr_chunk, wq_chunk, out_dtype=pl.FP32))
                    q_idx_full = pl.assemble(q_idx_full, pl.cast(q_acc, target_type=pl.BF16), [0, q0])

            # Stage 1: k_idx pre-projection = wk_idx(hidden_states).
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for ob in pl.parallel(0, WK_OUT_BLOCKS, 1, chunk=1):
                    k1 = ob * IDX_OUT_CHUNK
                    k_acc = pl.full([BATCH, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(hidden_states, [BATCH, K_CHUNK], [0, k0])
                        wk_chunk = pl.slice(wk_idx, [K_CHUNK, IDX_OUT_CHUNK], [k0, k1])
                        k_acc = pl.add(k_acc, pl.matmul(x_chunk, wk_chunk, out_dtype=pl.FP32))
                    k_idx = pl.assemble(k_idx, pl.cast(k_acc, target_type=pl.BF16), [0, k1])

            # Stage 2: LayerNorm on k_idx.
            with pl.at(level=pl.Level.CORE_GROUP):
                k_tile = pl.cast(pl.slice(k_idx, [BATCH, INDEX_HEAD_DIM], [0, 0]), target_type=pl.FP32)
                mean = pl.row_sum(pl.mul(k_tile, INDEX_HEAD_DIM_INV))
                centered = pl.row_expand_sub(k_tile, mean)
                var_eps = pl.row_sum(pl.mul(pl.add(pl.mul(centered, centered), EPS), INDEX_HEAD_DIM_INV))
                std = pl.reshape(
                    pl.sqrt(pl.reshape(var_eps, [1, BATCH])),
                    [BATCH, 1],
                )
                inv_std = pl.recip(std)
                normed = pl.row_expand_mul(centered, inv_std)
                gamma = pl.slice(k_norm_weight, [1, INDEX_HEAD_DIM], [0, 0])
                beta = pl.slice(k_norm_bias, [1, INDEX_HEAD_DIM], [0, 0])
                scaled = pl.col_expand_mul(normed, gamma)
                ones = pl.add(pl.sub(k_tile, k_tile), 1.0)
                k_normed = pl.add(scaled, pl.col_expand_mul(ones, beta))
                k_idx = pl.assemble(k_idx, pl.cast(k_normed, target_type=pl.BF16), [0, 0])

            # Stage 3: non-interleaved RoPE on indexer q_pe and k_pe.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b in pl.parallel(0, BATCH, 1, chunk=4):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cos_lo = pl.slice(rope_cos, [1, QK_ROPE_HEAD_DIM // 2], [pos, 0])
                    cos_hi = pl.slice(
                        rope_cos,
                        [1, QK_ROPE_HEAD_DIM // 2],
                        [pos, QK_ROPE_HEAD_DIM // 2],
                    )
                    sin_lo = pl.slice(rope_sin, [1, QK_ROPE_HEAD_DIM // 2], [pos, 0])
                    sin_hi = pl.slice(
                        rope_sin,
                        [1, QK_ROPE_HEAD_DIM // 2],
                        [pos, QK_ROPE_HEAD_DIM // 2],
                    )

                    for h in pl.range(INDEX_HEADS):
                        q_col = h * INDEX_HEAD_DIM
                        q_lo = pl.cast(
                            pl.slice(q_idx_full, [1, QK_ROPE_HEAD_DIM // 2], [b, q_col]),
                            target_type=pl.FP32,
                        )
                        q_hi = pl.cast(
                            pl.slice(
                                q_idx_full,
                                [1, QK_ROPE_HEAD_DIM // 2],
                                [b, q_col + QK_ROPE_HEAD_DIM // 2],
                            ),
                            target_type=pl.FP32,
                        )
                        q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                        q_idx_full = pl.assemble(
                            q_idx_full,
                            pl.cast(q_rot_lo, target_type=pl.BF16),
                            [b, q_col],
                        )
                        q_idx_full = pl.assemble(
                            q_idx_full,
                            pl.cast(q_rot_hi, target_type=pl.BF16),
                            [b, q_col + QK_ROPE_HEAD_DIM // 2],
                        )

                    k_lo = pl.cast(pl.slice(k_idx, [1, QK_ROPE_HEAD_DIM // 2], [b, 0]), target_type=pl.FP32)
                    k_hi = pl.cast(
                        pl.slice(k_idx, [1, QK_ROPE_HEAD_DIM // 2], [b, QK_ROPE_HEAD_DIM // 2]),
                        target_type=pl.FP32,
                    )
                    k_rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                    k_rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                    k_idx = pl.assemble(k_idx, pl.cast(k_rot_lo, target_type=pl.BF16), [b, 0])
                    k_idx = pl.assemble(
                        k_idx,
                        pl.cast(k_rot_hi, target_type=pl.BF16),
                        [b, QK_ROPE_HEAD_DIM // 2],
                    )

            # Stage 4: weights = weights_proj(hidden_states) * n_heads^-0.5 * head_dim^-0.5.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for ob in pl.parallel(0, WEIGHTS_BLOCKS, 1, chunk=1):
                    w0 = ob * WEIGHTS_OUT_CHUNK
                    w_acc = pl.full([BATCH, WEIGHTS_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(hidden_states, [BATCH, K_CHUNK], [0, k0])
                        wp_chunk = pl.slice(weights_proj, [K_CHUNK, WEIGHTS_OUT_CHUNK], [k0, w0])
                        w_acc = pl.add(
                            w_acc,
                            pl.matmul(
                                x_chunk,
                                pl.cast(wp_chunk, target_type=pl.BF16),
                                out_dtype=pl.FP32,
                            ),
                        )
                    w_scaled = pl.mul(w_acc, WEIGHT_SCALE)
                    weights = pl.assemble(weights, w_scaled, [0, w0])

            # Stage 5: Quantize indexer q/k tensors and keep INT8 outputs for scope3.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                q_idx_grouped = pl.reshape(q_idx_full, [INDEX_Q_ROWS, INDEX_HEAD_DIM])
                for b in pl.parallel(0, BATCH, 1, chunk=4):
                    for h0 in pl.range(0, INDEX_HEADS, BATCH):
                        r0 = b * INDEX_HEADS + h0
                        q_block = pl.cast(
                            pl.slice(q_idx_grouped, [BATCH, INDEX_HEAD_DIM], [r0, 0]),
                            target_type=pl.FP32,
                            mode="none",
                        )
                        q_abs = pl.maximum(q_block, pl.neg(q_block))
                        q_amax_row = pl.reshape(pl.row_max(q_abs), [1, BATCH])
                        q_amax_row = pl.maximum(
                            q_amax_row,
                            pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS),
                        )
                        q_scale_quant_row = pl.div(
                            pl.full([1, BATCH], dtype=pl.FP32, value=INT8_SCALE_MAX),
                            q_amax_row,
                        )
                        q_scale_dequant_row = pl.div(
                            pl.full([1, BATCH], dtype=pl.FP32, value=1.0),
                            q_scale_quant_row,
                        )
                        q_scale_quant = pl.reshape(q_scale_quant_row, [BATCH, 1])
                        q_scaled = pl.row_expand_mul(q_block, q_scale_quant)
                        q_i32 = pl.cast(q_scaled, target_type=pl.INT32, mode="round")
                        q_half = pl.cast(q_i32, target_type=pl.FP16, mode="round")
                        q_i8 = pl.cast(q_half, target_type=pl.INT8, mode="trunc")
                        q_idx_full_i8 = pl.assemble(q_idx_full_i8, q_i8, [r0, 0])
                        q_idx_scale_heads = pl.assemble(q_idx_scale_heads, q_scale_dequant_row, [BATCH, h0])

                for r0 in pl.parallel(0, BATCH, BATCH, chunk=1):
                    k_block = pl.cast(
                        pl.slice(k_idx, [BATCH, INDEX_HEAD_DIM], [r0, 0]),
                        target_type=pl.FP32,
                        mode="none",
                    )
                    k_abs = pl.maximum(k_block, pl.neg(k_block))
                    k_amax_row = pl.reshape(pl.row_max(k_abs), [1, BATCH])
                    k_amax_row = pl.maximum(
                        k_amax_row,
                        pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS),
                    )
                    k_scale_quant_row = pl.div(
                        pl.full([1, BATCH], dtype=pl.FP32, value=INT8_SCALE_MAX),
                        k_amax_row,
                    )
                    k_scale_dequant_row = pl.div(
                        pl.full([1, BATCH], dtype=pl.FP32, value=1.0),
                        k_scale_quant_row,
                    )
                    k_scale_quant = pl.reshape(k_scale_quant_row, [BATCH, 1])
                    k_scale_dequant = pl.reshape(k_scale_dequant_row, [BATCH, 1])
                    k_scale_pack_target = pl.full([BATCH, INT8_SCALE_PACK], dtype=pl.FP32, value=0.0)
                    k_scale_pack = pl.row_expand(k_scale_pack_target, k_scale_dequant)
                    k_scaled = pl.row_expand_mul(k_block, k_scale_quant)
                    k_i32 = pl.cast(k_scaled, target_type=pl.INT32, mode="round")
                    k_half = pl.cast(k_i32, target_type=pl.FP16, mode="round")
                    k_i8 = pl.cast(k_half, target_type=pl.INT8, mode="trunc")
                    k_idx_i8 = pl.assemble(k_idx_i8, k_i8, [r0, 0])
                    k_idx_scale = pl.assemble(k_idx_scale, k_scale_pack, [r0, 0])

            # Stage 6: update k_cache_idx_i8 and k_cache_idx_scale for the current decode token.
            k_idx_scale_flat = pl.reshape(k_idx_scale, [BATCH * INT8_SCALE_PACK])
            k_cache_idx_scale_flat = pl.reshape(k_cache_idx_scale, [BATCH * MAX_SEQ])
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b in pl.parallel(0, BATCH, 1, chunk=4):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cache_row = b * MAX_SEQ + pos
                    k_row_i8 = pl.slice(k_idx_i8, [1, INDEX_HEAD_DIM], [b, 0])
                    k_row_scale = pl.tensor.read(k_idx_scale_flat, [b * INT8_SCALE_PACK])
                    k_cache_idx_i8 = pl.assemble(k_cache_idx_i8, k_row_i8, [cache_row, 0])
                    pl.tensor.write(k_cache_idx_scale_flat, [b * MAX_SEQ + pos], k_row_scale)

            # Stage 7: export q scales, q INT8 rows, and weights for scope3.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for r0 in pl.parallel(0, INDEX_Q_ROWS, BATCH, chunk=1):
                    q_i8_tile = pl.slice(q_idx_full_i8, [BATCH, INDEX_HEAD_DIM], [r0, 0])
                    q_idx_full_i8_out = pl.assemble(q_idx_full_i8_out, q_i8_tile, [r0, 0])

                for b0 in pl.parallel(0, BATCH, BATCH, chunk=1):
                    q_scale_tile = pl.slice(q_idx_scale_heads, [BATCH, INDEX_HEADS], [b0, 0])
                    weights_tile = pl.slice(weights, [BATCH, INDEX_HEADS], [b0, 0])
                    q_idx_scale_heads_out = pl.assemble(q_idx_scale_heads_out, q_scale_tile, [b0, 0])
                    weights_out = pl.assemble(weights_out, weights_tile, [b0, 0])

            return k_cache_idx_i8, k_cache_idx_scale, q_idx_full_i8_out, q_idx_scale_heads_out, weights_out

    return DeepSeekV32DecodeFrontIndexer


def build_deepseek_v3_2_decode_front_scope2_program():
    return build_deepseek_v3_2_decode_front_indexer_program()


def golden_decode_front_indexer(tensors):
    import torch  # type: ignore[import]

    def int8_quant_groups(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.reshape(-1, INDEX_HEAD_DIM).float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = rows * scale_quant
        out_i32 = scaled.round().to(torch.int32)
        out_half = out_i32.to(torch.float16)
        out_i8 = out_half.to(torch.int8)
        scale_dequant = 1.0 / scale_quant
        return out_i8.reshape_as(x), scale_dequant

    hidden_states = tensors["hidden_states"].float()
    qr = tensors["qr"].float()
    wq_b_idx = tensors["wq_b_idx"].float()
    wk_idx = tensors["wk_idx"].float()
    k_norm_weight = tensors["k_norm_weight"].float()
    k_norm_bias = tensors["k_norm_bias"].float()
    weights_proj = tensors["weights_proj"].float()
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()
    k_cache_idx_i8 = tensors["k_cache_idx_i8"]
    k_cache_idx_scale = tensors["k_cache_idx_scale"]

    q_idx_full = (qr @ wq_b_idx).to(torch.bfloat16).float()
    k_idx = (hidden_states @ wk_idx).to(torch.bfloat16).float()

    mean = k_idx.mean(dim=-1, keepdim=True)
    centered = k_idx - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    k_idx = (centered * torch.rsqrt(var + EPS) * k_norm_weight + k_norm_bias).to(torch.bfloat16).float()

    half = QK_ROPE_HEAD_DIM // 2
    q_view = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        cos_lo = rope_cos[pos : pos + 1, :half]
        cos_hi = rope_cos[pos : pos + 1, half:QK_ROPE_HEAD_DIM]
        sin_lo = rope_sin[pos : pos + 1, :half]
        sin_hi = rope_sin[pos : pos + 1, half:QK_ROPE_HEAD_DIM]

        q_pe = q_view[b, :, :QK_ROPE_HEAD_DIM]
        q_lo = q_pe[:, :half].clone()
        q_hi = q_pe[:, half:].clone()
        q_view[b, :, :half] = q_lo * cos_lo - q_hi * sin_lo
        q_view[b, :, half:QK_ROPE_HEAD_DIM] = q_hi * cos_hi + q_lo * sin_hi

        k_lo = k_idx[b : b + 1, :half].clone()
        k_hi = k_idx[b : b + 1, half:QK_ROPE_HEAD_DIM].clone()
        k_idx[b : b + 1, :half] = k_lo * cos_lo - k_hi * sin_lo
        k_idx[b : b + 1, half:QK_ROPE_HEAD_DIM] = k_hi * cos_hi + k_lo * sin_hi
    q_idx_full = q_view.reshape(BATCH, INDEX_HEADS * INDEX_HEAD_DIM)

    weights = (hidden_states @ weights_proj.to(torch.bfloat16).float()) * (
        INDEX_HEADS ** -0.5 * INDEX_HEAD_DIM ** -0.5
    )
    q_idx_full = q_idx_full.to(torch.bfloat16).float()
    k_idx = k_idx.to(torch.bfloat16).float()
    q_idx_i8_golden, q_idx_scale_golden = int8_quant_groups(q_idx_full)
    k_idx_i8_golden, k_idx_scale_golden = int8_quant_groups(k_idx)

    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        k_cache_idx_i8[b * MAX_SEQ + pos, :].copy_(k_idx_i8_golden[b])
        k_cache_idx_scale[b, pos].copy_(k_idx_scale_golden[b, 0])

    tensors["q_idx_full_i8_out"].copy_(q_idx_i8_golden.view(BATCH * INDEX_HEADS, INDEX_HEAD_DIM))
    tensors["q_idx_scale_heads_out"].copy_(q_idx_scale_golden.view(BATCH, INDEX_HEADS))
    tensors["weights_out"].copy_(weights)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    index_q_out = INDEX_HEADS * INDEX_HEAD_DIM
    cache_rows = BATCH * MAX_SEQ
    seq_lens_data = torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_qr():
        return torch.rand(BATCH, Q_LORA_RANK) - 0.5

    def init_wq_b_idx():
        return (torch.rand(Q_LORA_RANK, index_q_out) - 0.5) / Q_LORA_RANK ** 0.5

    def init_wk_idx():
        return (torch.rand(HIDDEN, INDEX_HEAD_DIM) - 0.5) / HIDDEN ** 0.5

    def init_k_norm_weight():
        return torch.rand(1, INDEX_HEAD_DIM) - 0.5

    def init_k_norm_bias():
        return torch.rand(1, INDEX_HEAD_DIM) - 0.5

    def init_weights_proj():
        return (torch.rand(HIDDEN, INDEX_HEADS) - 0.5) / HIDDEN ** 0.5

    def init_rope():
        return torch.rand(MAX_SEQ, QK_ROPE_HEAD_DIM) - 0.5

    def init_k_cache_idx_i8():
        return torch.randint(-8, 9, (cache_rows, INDEX_HEAD_DIM), dtype=torch.int8)

    def init_k_cache_idx_scale():
        return torch.rand(BATCH, MAX_SEQ) * 0.1 + 0.001

    return [
        TensorSpec("hidden_states", [BATCH, HIDDEN], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("qr", [BATCH, Q_LORA_RANK], torch.bfloat16, init_value=init_qr),
        TensorSpec("wq_b_idx", [Q_LORA_RANK, index_q_out], torch.bfloat16, init_value=init_wq_b_idx),
        TensorSpec("wk_idx", [HIDDEN, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_wk_idx),
        TensorSpec("k_norm_weight", [1, INDEX_HEAD_DIM], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("k_norm_bias", [1, INDEX_HEAD_DIM], torch.float32, init_value=init_k_norm_bias),
        TensorSpec("weights_proj", [HIDDEN, INDEX_HEADS], torch.float32, init_value=init_weights_proj),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=seq_lens_data),
        TensorSpec("rope_cos", [MAX_SEQ, QK_ROPE_HEAD_DIM], torch.float32, init_value=init_rope),
        TensorSpec("rope_sin", [MAX_SEQ, QK_ROPE_HEAD_DIM], torch.float32, init_value=init_rope),
        TensorSpec(
            "k_cache_idx_i8",
            [cache_rows, INDEX_HEAD_DIM],
            torch.int8,
            init_value=init_k_cache_idx_i8,
            is_output=True,
        ),
        TensorSpec(
            "k_cache_idx_scale",
            [BATCH, MAX_SEQ],
            torch.float32,
            init_value=init_k_cache_idx_scale,
            is_output=True,
        ),
        TensorSpec("q_idx_full_i8_out", [BATCH * INDEX_HEADS, INDEX_HEAD_DIM], torch.int8, is_output=True),
        TensorSpec("q_idx_scale_heads_out", [BATCH, INDEX_HEADS], torch.float32, is_output=True),
        TensorSpec("weights_out", [BATCH, INDEX_HEADS], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v3_2_decode_front_scope2_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_decode_front_indexer,
        config=RunConfig(
            rtol=2e-3,
            atol=1.0,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
