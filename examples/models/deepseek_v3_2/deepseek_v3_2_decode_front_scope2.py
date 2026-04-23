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
- approximate rotate_activation with the scalar Hadamard scale
- compute per-index-head weights
- update k_cache_idx at the current decode position
- reduce q_idx_full with weights to q_idx:
  q_idx[b] = sum_h weights[b, h] * q_idx_full[b, h, :]
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
HADAMARD_SCALE = INDEX_HEAD_DIM ** -0.5

CACHE_ROWS = BATCH * MAX_SEQ

K_CHUNK = 128
LORA_CHUNK = 128
IDX_OUT_CHUNK = 128
WEIGHTS_OUT_CHUNK = 64
BATCH_TILE = 16


def build_deepseek_v3_2_decode_front_indexer_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    q_lora_rank: int = Q_LORA_RANK,
    index_heads: int = INDEX_HEADS,
    index_head_dim: int = INDEX_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
):
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    Q_LORA_RANK_CFG = q_lora_rank
    INDEX_HEADS_CFG = index_heads
    INDEX_HEAD_DIM_CFG = index_head_dim
    QK_ROPE_HEAD_DIM_CFG = qk_rope_head_dim
    INDEX_Q_OUT_CFG = index_heads * index_head_dim
    CACHE_ROWS_CFG = batch * max_seq_len

    HIDDEN_BLOCKS = (hidden_size + K_CHUNK - 1) // K_CHUNK
    QR_BLOCKS = (q_lora_rank + LORA_CHUNK - 1) // LORA_CHUNK
    IDX_OUT_BLOCKS = (INDEX_Q_OUT_CFG + IDX_OUT_CHUNK - 1) // IDX_OUT_CHUNK
    WK_OUT_BLOCKS = (index_head_dim + IDX_OUT_CHUNK - 1) // IDX_OUT_CHUNK
    WEIGHTS_BLOCKS = (index_heads + WEIGHTS_OUT_CHUNK - 1) // WEIGHTS_OUT_CHUNK
    INDEX_HEAD_DIM_INV = 1.0 / index_head_dim
    WEIGHT_SCALE = (index_heads ** -0.5) * (index_head_dim ** -0.5)
    HADAMARD_SCALE_CFG = index_head_dim ** -0.5

    @pl.program
    class DeepSeekV32DecodeFrontIndexer:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_indexer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            qr: pl.Tensor[[BATCH_CFG, Q_LORA_RANK_CFG], pl.BF16],
            wq_b_idx: pl.Tensor[[Q_LORA_RANK_CFG, INDEX_Q_OUT_CFG], pl.BF16],
            wk_idx: pl.Tensor[[HIDDEN_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
            k_norm_weight: pl.Tensor[[1, INDEX_HEAD_DIM_CFG], pl.FP32],
            k_norm_bias: pl.Tensor[[1, INDEX_HEAD_DIM_CFG], pl.FP32],
            weights_proj: pl.Tensor[[HIDDEN_CFG, INDEX_HEADS_CFG], pl.FP32],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ_CFG, QK_ROPE_HEAD_DIM_CFG], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ_CFG, QK_ROPE_HEAD_DIM_CFG], pl.FP32],
            k_cache_idx: pl.Tensor[[CACHE_ROWS_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
            q_idx: pl.Tensor[[BATCH_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, INDEX_HEAD_DIM_CFG], pl.BF16]:
            q_idx_full = pl.create_tensor([BATCH_CFG, INDEX_Q_OUT_CFG], dtype=pl.BF16)
            k_idx = pl.create_tensor([BATCH_CFG, INDEX_HEAD_DIM_CFG], dtype=pl.BF16)
            weights = pl.create_tensor([BATCH_CFG, INDEX_HEADS_CFG], dtype=pl.FP32)

            # Stage 0: q_idx_full = wq_b_idx(qr), shaped as [B, INDEX_HEADS, INDEX_HEAD_DIM].
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for ob in pl.parallel(0, IDX_OUT_BLOCKS, 1, chunk=8):
                    q0 = ob * IDX_OUT_CHUNK
                    q_acc = pl.full([BATCH_TILE, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(QR_BLOCKS):
                        k0 = kb * LORA_CHUNK
                        qr_chunk = pl.slice(qr, [BATCH_TILE, LORA_CHUNK], [0, k0])
                        wq_chunk = pl.slice(wq_b_idx, [LORA_CHUNK, IDX_OUT_CHUNK], [k0, q0])
                        q_acc = pl.add(q_acc, pl.matmul(qr_chunk, wq_chunk, out_dtype=pl.FP32))
                    q_idx_full = pl.assemble(q_idx_full, pl.cast(q_acc, target_type=pl.BF16), [0, q0])

            # Stage 1: k_idx pre-projection = wk_idx(hidden_states).
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for ob in pl.parallel(0, WK_OUT_BLOCKS, 1, chunk=1):
                    k1 = ob * IDX_OUT_CHUNK
                    k_acc = pl.full([BATCH_TILE, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [0, k0])
                        wk_chunk = pl.slice(wk_idx, [K_CHUNK, IDX_OUT_CHUNK], [k0, k1])
                        k_acc = pl.add(k_acc, pl.matmul(x_chunk, wk_chunk, out_dtype=pl.FP32))
                    k_idx = pl.assemble(k_idx, pl.cast(k_acc, target_type=pl.BF16), [0, k1])

            # Stage 2: LayerNorm on k_idx.
            with pl.at(level=pl.Level.CORE_GROUP):
                k_tile = pl.cast(pl.slice(k_idx, [BATCH_TILE, INDEX_HEAD_DIM_CFG], [0, 0]), target_type=pl.FP32)
                mean = pl.row_sum(pl.mul(k_tile, INDEX_HEAD_DIM_INV))
                centered = pl.row_expand_sub(k_tile, mean)
                var_eps = pl.row_sum(pl.mul(pl.add(pl.mul(centered, centered), EPS), INDEX_HEAD_DIM_INV))
                std = pl.reshape(
                    pl.sqrt(pl.reshape(var_eps, [1, BATCH_TILE])),
                    [BATCH_TILE, 1],
                )
                inv_std = pl.recip(std)
                normed = pl.row_expand_mul(centered, inv_std)
                gamma = pl.slice(k_norm_weight, [1, INDEX_HEAD_DIM_CFG], [0, 0])
                beta = pl.slice(k_norm_bias, [1, INDEX_HEAD_DIM_CFG], [0, 0])
                scaled = pl.col_expand_mul(normed, gamma)
                ones = pl.add(pl.sub(k_tile, k_tile), 1.0)
                k_normed = pl.add(scaled, pl.col_expand_mul(ones, beta))
                k_idx = pl.assemble(k_idx, pl.cast(k_normed, target_type=pl.BF16), [0, 0])

            # Stage 3: non-interleaved RoPE on indexer q_pe and k_pe.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cos_lo = pl.slice(rope_cos, [1, QK_ROPE_HEAD_DIM_CFG // 2], [pos, 0])
                    cos_hi = pl.slice(
                        rope_cos,
                        [1, QK_ROPE_HEAD_DIM_CFG // 2],
                        [pos, QK_ROPE_HEAD_DIM_CFG // 2],
                    )
                    sin_lo = pl.slice(rope_sin, [1, QK_ROPE_HEAD_DIM_CFG // 2], [pos, 0])
                    sin_hi = pl.slice(
                        rope_sin,
                        [1, QK_ROPE_HEAD_DIM_CFG // 2],
                        [pos, QK_ROPE_HEAD_DIM_CFG // 2],
                    )

                    for h in pl.range(INDEX_HEADS_CFG):
                        q_col = h * INDEX_HEAD_DIM_CFG
                        q_lo = pl.cast(
                            pl.slice(q_idx_full, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b, q_col]),
                            target_type=pl.FP32,
                        )
                        q_hi = pl.cast(
                            pl.slice(
                                q_idx_full,
                                [1, QK_ROPE_HEAD_DIM_CFG // 2],
                                [b, q_col + QK_ROPE_HEAD_DIM_CFG // 2],
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
                            [b, q_col + QK_ROPE_HEAD_DIM_CFG // 2],
                        )

                    k_lo = pl.cast(pl.slice(k_idx, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b, 0]), target_type=pl.FP32)
                    k_hi = pl.cast(
                        pl.slice(k_idx, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b, QK_ROPE_HEAD_DIM_CFG // 2]),
                        target_type=pl.FP32,
                    )
                    k_rot_lo = pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo))
                    k_rot_hi = pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi))
                    k_idx = pl.assemble(k_idx, pl.cast(k_rot_lo, target_type=pl.BF16), [b, 0])
                    k_idx = pl.assemble(
                        k_idx,
                        pl.cast(k_rot_hi, target_type=pl.BF16),
                        [b, QK_ROPE_HEAD_DIM_CFG // 2],
                    )

            # Stage 4: rotate_activation placeholder, apply only the Hadamard scale.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for ob in pl.parallel(0, IDX_OUT_BLOCKS, 1, chunk=8):
                    q0 = ob * IDX_OUT_CHUNK
                    q_tile = pl.cast(pl.slice(q_idx_full, [BATCH_TILE, IDX_OUT_CHUNK], [0, q0]), target_type=pl.FP32)
                    q_scaled = pl.mul(q_tile, HADAMARD_SCALE_CFG)
                    q_idx_full = pl.assemble(q_idx_full, pl.cast(q_scaled, target_type=pl.BF16), [0, q0])

                k_tile = pl.cast(pl.slice(k_idx, [BATCH_TILE, INDEX_HEAD_DIM_CFG], [0, 0]), target_type=pl.FP32)
                k_scaled = pl.mul(k_tile, HADAMARD_SCALE_CFG)
                k_idx = pl.assemble(k_idx, pl.cast(k_scaled, target_type=pl.BF16), [0, 0])

            # Stage 5: weights = weights_proj(hidden_states) * n_heads^-0.5 * head_dim^-0.5.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for ob in pl.parallel(0, WEIGHTS_BLOCKS, 1, chunk=1):
                    w0 = ob * WEIGHTS_OUT_CHUNK
                    w_acc = pl.full([BATCH_TILE, WEIGHTS_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [0, k0])
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

            # Stage 6: update k_cache_idx for the current decode token.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cache_row = b * MAX_SEQ_CFG + pos
                    k_row = pl.slice(k_idx, [1, INDEX_HEAD_DIM_CFG], [b, 0])
                    k_cache_idx = pl.assemble(k_cache_idx, k_row, [cache_row, 0])

            # Stage 7: reduce [B, H, D] q_idx_full into [B, D] q_idx for scope3.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                weights_t = pl.transpose(weights, axis1=0, axis2=1)
                for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE, chunk=1):
                    q_idx_acc = pl.full([BATCH_TILE, INDEX_HEAD_DIM_CFG], dtype=pl.FP32, value=0.0)
                    for h in pl.range(INDEX_HEADS_CFG):
                        q_h = pl.cast(
                            pl.slice(
                                q_idx_full,
                                [BATCH_TILE, INDEX_HEAD_DIM_CFG],
                                [b0, h * INDEX_HEAD_DIM_CFG],
                            ),
                            target_type=pl.FP32,
                        )
                        w_h_t = pl.slice(weights_t, [1, BATCH_TILE], [h, b0])
                        w_h = pl.reshape(w_h_t, [BATCH_TILE, 1])
                        q_idx_acc = pl.add(q_idx_acc, pl.row_expand_mul(q_h, w_h))
                    q_idx = pl.assemble(q_idx, pl.cast(q_idx_acc, target_type=pl.BF16), [b0, 0])

            return q_idx

    return DeepSeekV32DecodeFrontIndexer


def build_deepseek_v3_2_decode_front_scope2_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    q_lora_rank: int = Q_LORA_RANK,
    index_heads: int = INDEX_HEADS,
    index_head_dim: int = INDEX_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
):
    return build_deepseek_v3_2_decode_front_indexer_program(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        index_heads=index_heads,
        index_head_dim=index_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
    )


def golden_decode_front_indexer(tensors):
    import torch  # type: ignore[import]

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
    k_cache_idx = tensors["k_cache_idx"]

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

    q_idx_full = (q_idx_full * HADAMARD_SCALE).to(torch.bfloat16).float()
    k_idx = (k_idx * HADAMARD_SCALE).to(torch.bfloat16).float()

    weights = (hidden_states @ weights_proj.to(torch.bfloat16).float()) * (
        INDEX_HEADS ** -0.5 * INDEX_HEAD_DIM ** -0.5
    )
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        k_cache_idx[b * MAX_SEQ + pos, :].copy_(k_idx[b].to(torch.bfloat16))

    q_heads = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    q_idx = torch.einsum("bhd,bh->bd", q_heads, weights)
    tensors["q_idx"].copy_(q_idx.to(torch.bfloat16))


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    q_lora_rank: int = Q_LORA_RANK,
    index_heads: int = INDEX_HEADS,
    index_head_dim: int = INDEX_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    index_q_out = index_heads * index_head_dim
    cache_rows = batch * max_seq_len
    seq_lens_data = torch.randint(1, max_seq_len + 1, (batch,), dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_qr():
        return torch.rand(batch, q_lora_rank) - 0.5

    def init_wq_b_idx():
        return (torch.rand(q_lora_rank, index_q_out) - 0.5) / q_lora_rank ** 0.5

    def init_wk_idx():
        return (torch.rand(hidden_size, index_head_dim) - 0.5) / hidden_size ** 0.5

    def init_k_norm_weight():
        return torch.rand(1, index_head_dim) - 0.5

    def init_k_norm_bias():
        return torch.rand(1, index_head_dim) - 0.5

    def init_weights_proj():
        return (torch.rand(hidden_size, index_heads) - 0.5) / hidden_size ** 0.5

    def init_rope():
        return torch.rand(max_seq_len, qk_rope_head_dim) - 0.5

    def init_k_cache_idx():
        return torch.rand(cache_rows, index_head_dim) - 0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("qr", [batch, q_lora_rank], torch.bfloat16, init_value=init_qr),
        TensorSpec("wq_b_idx", [q_lora_rank, index_q_out], torch.bfloat16, init_value=init_wq_b_idx),
        TensorSpec("wk_idx", [hidden_size, index_head_dim], torch.bfloat16, init_value=init_wk_idx),
        TensorSpec("k_norm_weight", [1, index_head_dim], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("k_norm_bias", [1, index_head_dim], torch.float32, init_value=init_k_norm_bias),
        TensorSpec("weights_proj", [hidden_size, index_heads], torch.float32, init_value=init_weights_proj),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens_data),
        TensorSpec("rope_cos", [max_seq_len, qk_rope_head_dim], torch.float32, init_value=init_rope),
        TensorSpec("rope_sin", [max_seq_len, qk_rope_head_dim], torch.float32, init_value=init_rope),
        TensorSpec(
            "k_cache_idx",
            [cache_rows, index_head_dim],
            torch.bfloat16,
            init_value=init_k_cache_idx,
            is_output=True,
        ),
        TensorSpec("q_idx", [batch, index_head_dim], torch.bfloat16, is_output=True),
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
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_decode_front_indexer,
        config=RunConfig(
            rtol=2e-3,
            atol=2e-3,
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
