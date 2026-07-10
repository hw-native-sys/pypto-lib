# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 SWA (Sliding Window Attention) decode orchestration — `compress_ratio == 0` path.
Active in layers 0/1/7 of the model (3 of the 8 layers in demo). No KV compression, so neither
compressor nor indexer is invoked; topk for sparse_attn is window_topk_idxs only and the KV cache
holds only the sliding window (no compressed portion). YaRN frequency scaling is also disabled
in this path (model.py:478-479 selects base rope_theta when compress_ratio==0).
Companion files: attention_csa_draft.py (ratio=4)
                 attention_hca_draft.py (ratio=128)."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    KV_ORI_TABLE_MAX_BLOCKS,
)
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from decode_sparse_attn_swa import sparse_attn_swa


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local (SWA: ratio-0, no compressor/indexer)
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_TABLE_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
TOPK = WIN                          # SWA: sparse_attn topk = window only
SPARSE_IDX_TOPK = M.index_topk      # sparse_attn module's IDX_TOPK (static shape contract)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK
SPARSE_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS

# tiling
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE

@pl.jit.inline
def attention_swa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T, HC_MULT, D], pl.FP32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_step"):
        for b in pl.range(B):
            for s_idx in pl.range(S):
                t = b * S + s_idx
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                cos_row = pl.cast(freqs_cos[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                sin_row = pl.cast(freqs_sin[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(cos_row, target_type=pl.BF16, mode="rint")
                rope_sin_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(sin_row, target_type=pl.BF16, mode="rint")

    x_normed_t = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_norm(x_mixed, attn_norm_w, x_normed_t)
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale,
    )

    # Commit the current decode KV before attention. The SWA attention kernel
    # reads every visible row through metadata-expanded physical cache indices.
    kv_cache_flat = pl.reshape(kv_cache, [B * ORI_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM])
    with pl.spmd(T, name_hint="swa_cache_insert"):
        write_t = pl.tile.get_block_idx()
        write_row_i64 = pl.read(swa_slot_mapping, [write_t])
        if write_row_i64 >= 0:
            write_row = pl.cast(write_row_i64, pl.INDEX)
            kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn_swa(
        q, kv_cache_flat, swa_indices, swa_lens,
        attn_sink, rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def attention_swa_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.InOut[pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
):
    attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return x_out


def golden_attention_swa(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm
    from decode_sparse_attn_swa import golden_sparse_attn
    from hc_post import golden_hc_post

    # ---- Block.hc_pre (model.py:691) ----
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # ===== Attention.forward (model.py:484-543), ratio==0 branch =====
    position_ids = tensors["position_ids"].to(torch.int64)
    bsz, seqlen = B, S
    win = WIN
    rd = ROPE_HEAD_DIM

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_T = torch.empty(T, rd, dtype=freqs_cos.dtype)
    rope_sin_T = torch.empty(T, rd, dtype=freqs_sin.dtype)
    for t in range(T):
        pos = int(position_ids[t].item())
        rope_cos_T[t] = freqs_cos[pos]
        rope_sin_T[t] = freqs_sin[pos]

    # q + win kv (model.py:495-504)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_T,
        "rope_sin": rope_sin_T,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,                                                              # qr unused on SWA path
        "qr_scale": qr_scale,
    })

    kv_cache = tensors["kv_cache"]
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)

    # Current decode KV is visible to SWA through the same physical cache slots
    # that metadata points at.
    swa_slot_mapping = tensors["swa_slot_mapping"].to(torch.int64)
    for t in range(T):
        write_row = int(swa_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]

    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "swa_indices": tensors["swa_indices"],
        "swa_lens": tensors["swa_lens"],
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # ===== Block.hc_post (model.py:694) =====
    y = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y


def build_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from decode_metadata import (
        block_table,
        paged_slot_mapping,
        position_ids_from_starts,
        resolve_start_positions,
        swa_indices_and_lens,
        swa_decode_start_set,
    )
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, 0, dtype=torch.bfloat16)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.unsqueeze(-1)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x_hc():
        return torch.empty(T, HC_MULT, D).uniform_(-1, 1)
    # Real layer-0 (SWA) hc_attn scale/base (fn synthetic at real magnitude). A synthetic
    # scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out and the
    # hc residual to near-zero in x_out where quant noise blows up the relative tail.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.039
    def init_hc_attn_scale():
        return torch.tensor([2.076026, 0.018729, 0.245936])
    def init_hc_attn_base():
        return torch.tensor([
            3.9083, -2.0399, -2.2033, -2.017,
            -2.4443, -10.3158, -8.9943, -6.3581,
            9.8577, -9.5177, -24.8724, -22.8929,
            -21.545, 0.7791, -3.386, 1.1948,
            -20.9605, -0.7702, 1.4218, -4.8994,
            1.5177, -29.7663, -30.1413, -1.2413,
        ])
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return torch.randn(D, Q_LORA) / D ** 0.5
    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    def init_kv_cache():
        return init_normalized_cache((B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM))

    def init_block_table():
        return block_table(batch=B, table_blocks=ORI_TABLE_MAX_BLOCKS, physical_blocks=ORI_MAX_BLOCKS)

    def init_attn_sink():
        return torch.zeros(H)
    def init_default_start_pos():
        # Canonical SWA start-position set (sliding-window regimes + 8k long-context).
        return swa_decode_start_set(batch=B, window=WIN)
    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=B,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )
    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S).reshape(-1).contiguous()
    def init_swa_slot_mapping():
        return paged_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_block_table(),
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_swa_metadata():
        return swa_indices_and_lens(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )
    def init_swa_indices():
        return init_swa_metadata()[0].contiguous()
    def init_swa_lens():
        return init_swa_metadata()[1].contiguous()
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("swa_slot_mapping", [T], torch.int64, init_value=init_swa_slot_mapping),
        TensorSpec("swa_indices", [T, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [T], torch.int32, init_value=init_swa_lens),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch SWA set that includes the 8k point.")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=attention_swa_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_attention_swa,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            # Tightened from CANN's 1e-2 bar: realistic hc_attn gates keep x_out
            # well-conditioned (0% over 3e-3 across seeds; worst rdiff ~0.16).
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
