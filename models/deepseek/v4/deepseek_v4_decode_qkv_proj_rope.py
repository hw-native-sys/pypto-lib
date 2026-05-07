# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 single-token decode attn_norm fused + Q/KV LoRA + RoPE: produces (q, kv, qr) for the
attention body, with attn_norm fused at the front to save one GM round-trip."""


import pypto.language as pl


# Decode batch / seq
B           = 16               # demo 4
S           = 1
T           = B * S
# Hidden / Attention
D           = 4096             # flash:4096 pro:7168
H           = 64               # flash:64 pro:128
HEAD_DIM    = 512
ROPE_DIM    = 64
ROPE_HALF   = ROPE_DIM // 2
NOPE_DIM    = HEAD_DIM - ROPE_DIM
Q_LORA      = 1024             # flash:1024 pro:1536
HEAD_CHUNK  = 64
Q_LORA_TILE = 32
EPS         = 1e-6
# Derived constants for multi-function type annotations
Q_BLOCKS      = Q_LORA // Q_LORA_TILE
HEAD_GROUP    = 8
assert (H * HEAD_DIM) % (HEAD_CHUNK * HEAD_GROUP) == 0, \
    "HEAD_BLOCKS must be divisible by HEAD_GROUP"
HEAD_GROUP_BLOCKS = (H * HEAD_DIM) // (HEAD_CHUNK * HEAD_GROUP)


def build_deepseek_v4_decode_qkv_proj_rope_program():
    @pl.program
    class DeepSeekV4DecodeQkvProjRope:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_qkv_proj_rope(
            self,
            x:         pl.Tensor[[B, S, D],              pl.BF16],
            norm_w:    pl.Tensor[[D],                    pl.FP32],
            wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
            wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.BF16],
            wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
            rope_cos:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
            rope_sin:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
            gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
            gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
            q:         pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
            kv:        pl.Out[pl.Tensor[[T, HEAD_DIM],    pl.BF16]],
            qr:        pl.Out[pl.Tensor[[T, Q_LORA],      pl.BF16]],
        ):
            D_CHUNK = 512
            Q_LORA_CHUNK = Q_LORA_TILE
            KV_CHUNK = 32
            D_BLOCKS = D // D_CHUNK
            KV_BLOCKS = HEAD_DIM // KV_CHUNK

            x_flat = pl.reshape(x, [T, D])

            # Stage 0.1: fused attn_norm -> token_x_fp32
            token_x_fp32 = pl.create_tensor([T, D], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP):
                x_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
                for db in pl.range(D_BLOCKS):
                    d0 = db * D_CHUNK
                    x_chunk = pl.cast(pl.slice(x_flat, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
                    x_sq_sum = pl.add(x_sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T]))
                x_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS)))

            x_inv_rms_t = pl.reshape(x_inv_rms, [T, 1])
            for db in pl.parallel(0, D_BLOCKS, 1):
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    d0 = db * D_CHUNK
                    x_chunk = pl.cast(pl.slice(x_flat, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
                    norm_w_chunk = pl.reshape(pl.slice(norm_w, [D_CHUNK], [d0]), [1, D_CHUNK])
                    x_normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, x_inv_rms_t), norm_w_chunk)
                    token_x_fp32 = pl.assemble(token_x_fp32, x_normed, [0, d0])

            # Stage 0.2: pre-cast token_x for split AIV->AIC flow.
            token_x_bf16 = pl.create_tensor([T, D], dtype=pl.BF16)
            for db in pl.parallel(0, D_BLOCKS, 1):
                with pl.at(level=pl.Level.CORE_GROUP):
                    d0 = db * D_CHUNK
                    x_chunk_fp32 = pl.slice(token_x_fp32, [T, D_CHUNK], [0, d0])
                    token_x_bf16 = pl.assemble(token_x_bf16, pl.cast(x_chunk_fp32, target_type=pl.BF16), [0, d0])

            # Stage 1/2.1: qr = rms_norm(token_x @ wq_a, gamma_cq)
            qr_fp32 = pl.create_tensor([T, Q_LORA], dtype=pl.FP32)
            for qb in pl.parallel(0, Q_BLOCKS, 1):
                with pl.at(level=pl.Level.CORE_GROUP):
                    q0 = qb * Q_LORA_CHUNK
                    d0_0 = 0
                    x_chunk_bf16_0 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0_0])
                    w_chunk_0 = pl.slice(wq_a, [D_CHUNK, Q_LORA_CHUNK], [d0_0, q0])
                    q_acc = pl.matmul(x_chunk_bf16_0, w_chunk_0, out_dtype=pl.FP32)
                    for db in pl.range(1, D_BLOCKS):
                        d0 = db * D_CHUNK
                        x_chunk_bf16 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0])
                        w_chunk = pl.slice(wq_a, [D_CHUNK, Q_LORA_CHUNK], [d0, q0])
                        q_acc = pl.matmul_acc(q_acc, x_chunk_bf16, w_chunk)
                    qr_fp32 = pl.assemble(qr_fp32, q_acc, [0, q0])

            with pl.at(level=pl.Level.CORE_GROUP):
                qr_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
                for qb in pl.range(Q_BLOCKS):
                    q0 = qb * Q_LORA_CHUNK
                    qr_chunk = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, q0])
                    qr_sq_sum = pl.add(qr_sq_sum, pl.reshape(pl.row_sum(pl.mul(qr_chunk, qr_chunk)), [1, T]))
                qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))

            qr_inv_rms_t = pl.reshape(qr_inv_rms, [T, 1])
            for qb in pl.parallel(0, Q_BLOCKS, 1):
                with pl.at(level=pl.Level.CORE_GROUP):
                    q0 = qb * Q_LORA_CHUNK
                    qr_chunk = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, q0])
                    gamma_chunk = pl.reshape(
                        pl.cast(pl.slice(gamma_cq, [Q_LORA_CHUNK], [q0]), target_type=pl.FP32),
                        [1, Q_LORA_CHUNK],
                    )
                    qr_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_chunk)
                    qr_fp32 = pl.assemble(qr_fp32, qr_normed, [0, q0])

            # NOTE: Stage 2.2 must follow normalization, because qr_fp32 now stores
            # normalized values (after the pl.assemble above). Reordering these
            # blocks would cast un-normalized accumulators and corrupt q_proj.
            # Stage 2.2: pre-cast qr for split AIV->AIC flow.
            for qb in pl.parallel(0, Q_BLOCKS, 1):
                with pl.at(level=pl.Level.CORE_GROUP):
                    q0 = qb * Q_LORA_CHUNK
                    qr_chunk_fp32 = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, q0])
                    qr = pl.assemble(qr, pl.cast(qr_chunk_fp32, target_type=pl.BF16), [0, q0])

            # Stage 3: q_proj = qr @ wq_b (matmul + matmul_acc)
            q_proj_fp32 = pl.create_tensor([T, H * HEAD_DIM], dtype=pl.FP32)
            for hgb in pl.parallel(0, HEAD_GROUP_BLOCKS, 1):
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for hi in pl.range(HEAD_GROUP):
                        hb = hgb * HEAD_GROUP + hi
                        h0 = hb * HEAD_CHUNK
                        q0_0 = 0
                        qr_bf_0 = pl.slice(qr, [T, Q_LORA_CHUNK], [0, q0_0])
                        wq_0 = pl.slice(wq_b, [Q_LORA_CHUNK, HEAD_CHUNK], [q0_0, h0])
                        col_acc = pl.matmul(qr_bf_0, wq_0, out_dtype=pl.FP32)
                        for qb in pl.range(1, Q_BLOCKS):
                            q0 = qb * Q_LORA_CHUNK
                            qr_bf16_chunk = pl.slice(qr, [T, Q_LORA_CHUNK], [0, q0])
                            wq_chunk = pl.slice(wq_b, [Q_LORA_CHUNK, HEAD_CHUNK], [q0, h0])
                            col_acc = pl.matmul_acc(col_acc, qr_bf16_chunk, wq_chunk)
                        q_proj_fp32 = pl.assemble(q_proj_fp32, col_acc, [0, h0])

            # Stage 4: per-head RMSNorm + RoPE on q
            q_flat = pl.reshape(q, [T, H * HEAD_DIM])
            for h in pl.parallel(0, H, 1):
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    h0 = h * HEAD_DIM
                    q_head_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
                    for db in pl.range(HEAD_DIM // HEAD_CHUNK):
                        d0 = h0 + db * HEAD_CHUNK
                        q_chunk = pl.slice(q_proj_fp32, [T, HEAD_CHUNK], [0, d0])
                        q_head_sq_sum = pl.add(q_head_sq_sum, pl.reshape(pl.row_sum(pl.mul(q_chunk, q_chunk)), [1, T]))
                    q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
                    q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [T, 1])

                    for nb in pl.range(NOPE_DIM // HEAD_CHUNK):
                        n0 = nb * HEAD_CHUNK
                        q_chunk = pl.slice(q_proj_fp32, [T, HEAD_CHUNK], [0, h0 + n0])
                        q_normed = pl.row_expand_mul(q_chunk, q_head_inv_rms_t)
                        q_flat = pl.assemble(q_flat, pl.cast(q_normed, target_type=pl.BF16), [0, h0 + n0])

                    q_lo = pl.slice(q_proj_fp32, [T, ROPE_HALF], [0, h0 + NOPE_DIM])
                    q_hi = pl.slice(q_proj_fp32, [T, ROPE_HALF], [0, h0 + NOPE_DIM + ROPE_HALF])
                    q_lo_norm = pl.row_expand_mul(q_lo, q_head_inv_rms_t)
                    q_hi_norm = pl.row_expand_mul(q_hi, q_head_inv_rms_t)
                    cos_lo = pl.cast(pl.slice(rope_cos, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
                    cos_hi = pl.cast(pl.slice(rope_cos, [T, ROPE_HALF], [0, ROPE_HALF]), target_type=pl.FP32)
                    sin_lo = pl.cast(pl.slice(rope_sin, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
                    sin_hi = pl.cast(pl.slice(rope_sin, [T, ROPE_HALF], [0, ROPE_HALF]), target_type=pl.FP32)
                    q_rot_lo = pl.sub(pl.mul(q_lo_norm, cos_lo), pl.mul(q_hi_norm, sin_lo))
                    q_rot_hi = pl.add(pl.mul(q_hi_norm, cos_hi), pl.mul(q_lo_norm, sin_hi))
                    q_flat = pl.assemble(q_flat, pl.cast(q_rot_lo, target_type=pl.BF16), [0, h0 + NOPE_DIM])
                    q_flat = pl.assemble(q_flat, pl.cast(q_rot_hi, target_type=pl.BF16), [0, h0 + NOPE_DIM + ROPE_HALF])

            q = pl.reshape(q_flat, [T, H, HEAD_DIM])

            # Stage 5/6: kv = rms_norm(token_x @ wkv, gamma_ckv) + RoPE
            kv_fp32 = pl.create_tensor([T, HEAD_DIM], dtype=pl.FP32)
            for kb in pl.parallel(0, KV_BLOCKS, 1):
                with pl.at(level=pl.Level.CORE_GROUP):
                    k0 = kb * KV_CHUNK
                    d0_0 = 0
                    x_chunk_bf16_0 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0_0])
                    wkv_chunk_0 = pl.slice(wkv, [D_CHUNK, KV_CHUNK], [d0_0, k0])
                    kv_acc = pl.matmul(x_chunk_bf16_0, wkv_chunk_0, out_dtype=pl.FP32)
                    for db in pl.range(1, D_BLOCKS):
                        d0 = db * D_CHUNK
                        x_chunk_bf16 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0])
                        wkv_chunk = pl.slice(wkv, [D_CHUNK, KV_CHUNK], [d0, k0])
                        kv_acc = pl.matmul_acc(kv_acc, x_chunk_bf16, wkv_chunk)
                    kv_fp32 = pl.assemble(kv_fp32, kv_acc, [0, k0])

            with pl.at(level=pl.Level.CORE_GROUP):
                kv_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
                for kb in pl.range(KV_BLOCKS):
                    k0 = kb * KV_CHUNK
                    kv_chunk = pl.slice(kv_fp32, [T, KV_CHUNK], [0, k0])
                    kv_sq_sum = pl.add(kv_sq_sum, pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, T]))
                kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))

            kv_inv_rms_t = pl.reshape(kv_inv_rms, [T, 1])
            for nb in pl.parallel(0, NOPE_DIM // KV_CHUNK, 1):
                with pl.at(level=pl.Level.CORE_GROUP):
                    n0 = nb * KV_CHUNK
                    kv_chunk = pl.slice(kv_fp32, [T, KV_CHUNK], [0, n0])
                    gamma_kv_chunk = pl.reshape(
                        pl.cast(pl.slice(gamma_ckv, [KV_CHUNK], [n0]), target_type=pl.FP32),
                        [1, KV_CHUNK],
                    )
                    kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                    kv = pl.assemble(kv, pl.cast(kv_normed, target_type=pl.BF16), [0, n0])
            with pl.at(level=pl.Level.CORE_GROUP):
                kv_lo = pl.slice(kv_fp32, [T, ROPE_HALF], [0, NOPE_DIM])
                kv_hi = pl.slice(kv_fp32, [T, ROPE_HALF], [0, NOPE_DIM + ROPE_HALF])
                gamma_lo = pl.reshape(
                    pl.cast(pl.slice(gamma_ckv, [ROPE_HALF], [NOPE_DIM]), target_type=pl.FP32),
                    [1, ROPE_HALF],
                )
                gamma_hi = pl.reshape(
                    pl.cast(pl.slice(gamma_ckv, [ROPE_HALF], [NOPE_DIM + ROPE_HALF]), target_type=pl.FP32),
                    [1, ROPE_HALF],
                )
                kv_lo_norm = pl.col_expand_mul(pl.row_expand_mul(kv_lo, kv_inv_rms_t), gamma_lo)
                kv_hi_norm = pl.col_expand_mul(pl.row_expand_mul(kv_hi, kv_inv_rms_t), gamma_hi)
                cos_lo = pl.cast(pl.slice(rope_cos, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
                cos_hi = pl.cast(pl.slice(rope_cos, [T, ROPE_HALF], [0, ROPE_HALF]), target_type=pl.FP32)
                sin_lo = pl.cast(pl.slice(rope_sin, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
                sin_hi = pl.cast(pl.slice(rope_sin, [T, ROPE_HALF], [0, ROPE_HALF]), target_type=pl.FP32)
                kv_rot_lo = pl.sub(pl.mul(kv_lo_norm, cos_lo), pl.mul(kv_hi_norm, sin_lo))
                kv_rot_hi = pl.add(pl.mul(kv_hi_norm, cos_hi), pl.mul(kv_lo_norm, sin_hi))
                kv = pl.assemble(kv, pl.cast(kv_rot_lo, target_type=pl.BF16), [0, NOPE_DIM])
                kv = pl.assemble(kv, pl.cast(kv_rot_hi, target_type=pl.BF16), [0, NOPE_DIM + ROPE_HALF])

            return q, kv, qr

    return DeepSeekV4DecodeQkvProjRope


def golden_deepseek_v4_decode_qkv_proj_rope(tensors):
    """Torch reference: attn_norm fused, then Q/KV LoRA + RoPE (model.py 692, 495-504)."""
    import torch

    x         = tensors["x"].float()              # [B, S, D]
    norm_w    = tensors["norm_w"].float()          # [D]
    wq_a      = tensors["wq_a"].float()
    wq_b      = tensors["wq_b"].float()
    wkv       = tensors["wkv"].float()
    rope_cos  = tensors["rope_cos"].float()
    rope_sin  = tensors["rope_sin"].float()
    gamma_cq  = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

    def matmul_bf16_input_fp32(a, b):
        a_fp32 = a.to(torch.bfloat16).float()
        b_fp32 = b.to(torch.bfloat16).float()
        return torch.matmul(a_fp32, b_fp32).float()

    def apply_rope(x_rope, cos, sin):
        # x_rope: [T, ..., ROPE_DIM] using lo/hi half split.
        half = ROPE_DIM // 2
        x_lo = x_rope[..., :half]
        x_hi = x_rope[..., half:]
        cos_lo = cos[..., :half]
        cos_hi = cos[..., half:]
        sin_lo = sin[..., :half]
        sin_hi = sin[..., half:]
        while cos_lo.ndim < x_lo.ndim:
            cos_lo = cos_lo.unsqueeze(-2)
            cos_hi = cos_hi.unsqueeze(-2)
            sin_lo = sin_lo.unsqueeze(-2)
            sin_hi = sin_hi.unsqueeze(-2)
        rot_lo = x_lo * cos_lo - x_hi * sin_lo
        rot_hi = x_hi * cos_hi + x_lo * sin_hi
        return torch.cat([rot_lo, rot_hi], dim=-1)

    # attn_norm fused (model.py:692)
    token_x = rms_norm(x.view(T, D), norm_w)                        # [T, D]

    # Q path
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)   # [T, Q_LORA]
    q_full = matmul_bf16_input_fp32(qr_out, wq_b).view(T, H, HEAD_DIM)   # [T, H, HEAD_DIM]
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv                                            # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    # KV path
    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)  # [T, HEAD_DIM]
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:]  = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_out.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.02
    def init_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return torch.randn(D, Q_LORA) / (D ** 0.5)
    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / (Q_LORA ** 0.5)
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / (D ** 0.5)
    def init_cos():
        return torch.cos(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)

    return [
        TensorSpec("x",         [B, S, D],              torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w",    [D],                    torch.float32,  init_value=init_norm_w),
        TensorSpec("wq_a",      [D, Q_LORA],            torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b",      [Q_LORA, H * HEAD_DIM], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv",       [D, HEAD_DIM],          torch.bfloat16, init_value=init_wkv),
        TensorSpec("rope_cos",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_cos),
        TensorSpec("rope_sin",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_sin),
        TensorSpec("gamma_cq",  [Q_LORA],               torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM],             torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("q",         [T, H, HEAD_DIM],       torch.bfloat16, is_output=True),
        TensorSpec("kv",        [T, HEAD_DIM],          torch.bfloat16, is_output=True),
        TensorSpec("qr",        [T, Q_LORA],            torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_qkv_proj_rope_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_qkv_proj_rope,
        config=RunConfig(
            # Tightened after fixing KV RoPE gamma_ckv application.
            # On-board a2a3 validation still shows small q-path BF16 drift at 5e-3.
            rtol=6e-3,
            atol=6e-3,
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
