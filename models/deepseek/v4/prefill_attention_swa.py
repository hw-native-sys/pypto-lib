# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill SWA attention.

This kernel uses the current [PREFILL_BATCH, PREFILL_SEQ] kernel shape from
config through prefill_qkv_proj_rope. Q/KV projection is shared with
prefill_qkv_proj_rope; SWA attention reads the previous sliding-window cache
when `start_pos > 0`, uses current KV for keys inside this invocation, and
writes the current KV after attention.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ
from prefill_hc_post import golden_prefill_hc_post, prefill_hc_post
from prefill_hc_pre import golden_prefill_hc_pre, prefill_hc_pre
from prefill_qkv_proj_rope import golden_prefill_qkv_proj_rope, prefill_qkv_proj_rope_core
from prefill_swa_sparse_attn import _quant_w_per_row, golden_prefill_swa_sparse_attn, prefill_swa_sparse_attn


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HEAD_DIM = ROPE_DIM
NOPE_DIM = M.nope_head_dim
NOPE_HEAD_DIM = NOPE_DIM
Q_LORA = M.q_lora_rank
ROPE_HALF = ROPE_DIM // 2
HALF_ROPE = ROPE_HALF
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# SWA cache/topk contract. The ratio-0 path has only the sliding-window cache.
ORI_MAX_BLOCKS = 1
MAX_BLOCKS = ORI_MAX_BLOCKS
BLOCK_NUM = B * MAX_BLOCKS
TOPK = WIN
SPARSE_IDX_TOPK = M.index_topk
SPARSE_TOPK = WIN
START_POS = 0

# HC tiling, mirrored from hc_pre/hc_post but using prefill B/S/T.
MIX_PAD = 32
NEG_INF = -1e20
T_TILE = 16
RMS_T_TILE = 16
LINEAR_T_TILE = 16
COMB_T_TILE = 16
RMS_K_CHUNK = 128
LINEAR_K_CHUNK = 512
D_CHUNK = 512
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK
RMS_PIPE_STAGE = 1 if T >= 64 else 4

# SWA + o_proj tiling.
ATTN_HEAD_TILE = 16
ATTN_TASK_TILE = 2
ATTN_ONLINE_VALUE_CHUNK = 64
SPARSE_ATTN_TILE = 64
SPARSE_ATTN_BLOCKS = (WIN + SPARSE_ATTN_TILE - 1) // SPARSE_ATTN_TILE
KV_CACHE_WRITE_TILE = 16
KV_WINDOW_ROWS = T * SPARSE_ATTN_BLOCKS * SPARSE_ATTN_TILE
SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK
ROPE_TOKEN_TILE = 4
ROPE_PACK_TOKEN_TILE = 16
ROPE_PACK_SPMD_BLOCKS = (T // ROPE_PACK_TOKEN_TILE) * O_GROUPS
O_PROJ_T_TILE = 16
A_K_CHUNK = 128
A_N_CHUNK = 128
B_K_CHUNK = 128
B_N_CHUNK = 128
QUANT_CHUNK = 32
QUANT_TOKEN_TILE = 8

assert WIN == BLOCK_SIZE, "SWA prefill currently assumes one window page per batch"
assert S <= WIN, "SWA prefill tile must not exceed the sliding-window ring size"
assert H % ATTN_HEAD_TILE == 0, "attention head tile must divide H"
assert T % ATTN_TASK_TILE == 0, "attention token task tile must divide prefill T"
assert HEAD_DIM % ATTN_ONLINE_VALUE_CHUNK == 0, "online attention value chunk must divide head dim"
assert NOPE_DIM % ATTN_ONLINE_VALUE_CHUNK == 0, "online attention chunk must split noPE/rope boundary"
assert S % KV_CACHE_WRITE_TILE == 0, "KV cache write tile must divide prefill S tile"
assert T % O_PROJ_T_TILE == 0, "o_proj token tile must divide prefill T"


@pl.jit
def prefill_attention_swa(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    # Full prefill path mirrors the official block: hc_pre -> qkv/rope -> SWA
    # attention/o_proj -> KV writeback -> hc_post.
    x_mixed, post, comb = prefill_hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )

    # Reuse the shared prefill QKV/RoPE projection to stay aligned with decode.
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q, kv, qr, qr_scale = prefill_qkv_proj_rope_core(
        x_mixed,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        even_select_t,
        odd_select_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        start_pos,
    )

    # Gather the RoPE rows used later to undo RoPE on the attention output.
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    for b in pl.range(B):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_rope_rows"):
            pos = pl.cast(start_pos, pl.INDEX)
            cos_rows = pl.slice(freqs_cos, [S, ROPE_HEAD_DIM], [pos, 0])
            sin_rows = pl.slice(freqs_sin, [S, ROPE_HEAD_DIM], [pos, 0])
            rope_cos_t = pl.assemble(rope_cos_t, cos_rows, [b * S, 0])
            rope_sin_t = pl.assemble(rope_sin_t, sin_rows, [b * S, 0])

    # SWA attention computes grouped head values, projects them, then updates KV cache.
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    kv_cache, attn_out = prefill_swa_sparse_attn(
        q,
        kv,
        kv_cache,
        block_table,
        attn_sink,
        rope_cos_t,
        rope_sin_t,
        even_select_local,
        odd_select_local,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        start_pos,
    )

    # create_tensor seeds static metadata required by the JIT for hc_post input.
    attn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out = prefill_hc_post(
        attn_out_3d,
        x_hc,
        post,
        comb,
        x_out,
    )
    return kv_cache, x_out


def _quant_w_per_output_channel(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_prefill_attention_swa(tensors):
    """Torch reference for the official SWA prefill branch."""
    import torch

    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
    golden_prefill_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })
    if "x_mixed" in tensors:
        tensors["x_mixed"][:] = x_mixed
    if "post_t" in tensors:
        tensors["post_t"][:] = post
    if "comb_t" in tensors:
        tensors["comb_t"][:] = comb

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    golden_prefill_qkv_proj_rope({
        "x": x_mixed,
        "norm_w": tensors["attn_norm_w"],
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
        "start_pos": tensors["start_pos"],
    })
    if "q_out" in tensors:
        tensors["q_out"][:] = q
    if "kv_out" in tensors:
        tensors["kv_out"][:] = kv
    if "qr_out" in tensors:
        tensors["qr_out"][:] = qr
    if "qr_scale_out" in tensors:
        tensors["qr_scale_out"][:] = qr_scale

    start_pos = int(tensors["start_pos"])
    positions = torch.arange(start_pos, start_pos + S, device=tensors["freqs_cos"].device)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).unsqueeze(0).expand(B, S, ROPE_HEAD_DIM)
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).unsqueeze(0).expand(B, S, ROPE_HEAD_DIM)

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_swa_sparse_attn({
        "q": q,
        "kv": kv,
        "kv_cache": tensors["kv_cache"],
        "block_table": tensors["block_table"],
        "attn_sink": tensors["attn_sink"],
        "freqs_cos_t": rope_cos_t.reshape(T, ROPE_HEAD_DIM).contiguous(),
        "freqs_sin_t": rope_sin_t.reshape(T, ROPE_HEAD_DIM).contiguous(),
        "even_select_local": tensors["even_select_local"],
        "odd_select_local": tensors["odd_select_local"],
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
        "start_pos": tensors["start_pos"],
    })
    if "attn_out" in tensors:
        tensors["attn_out"][:] = attn_out

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_prefill_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post,
        "comb": comb,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_hc():
        return torch.randn(B, S, HC_MULT, D) * 0.05
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    def init_hc_attn_scale():
        return torch.ones(3) * 0.5
    def init_hc_attn_base():
        return torch.zeros(MIX_HC)
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
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_even_select_t():
        m = torch.zeros((ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM))
        for i in range(ROPE_HEAD_DIM // 2):
            m[i, 2 * i] = 1
        return m
    def init_odd_select_t():
        m = torch.zeros((ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM))
        for i in range(ROPE_HEAD_DIM // 2):
            m[i, 2 * i + 1] = 1
        return m
    def init_even_select_local():
        m = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            m[2 * i, i] = 1
        return m
    def init_odd_select_local():
        m = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            m[2 * i + 1, i] = 1
        return m
    def init_block_table():
        tbl = torch.full((B, MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            tbl[b, 0] = b
        return tbl
    def init_kv_cache():
        if start_pos == 0:
            return torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        return torch.randn(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) * 0.05
    def init_attn_sink():
        return torch.zeros(H)
    def init_seqused_kv():
        return torch.full((B,), min(WIN, start_pos + S), dtype=torch.int32)
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.float32, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("even_select_t", [ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_even_select_t),
        TensorSpec("odd_select_t", [ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_odd_select_t),
        TensorSpec("even_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("block_table", [B, MAX_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("start_pos", torch.int32, start_pos),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--block-dim", type=int, default=None)
    parser.add_argument("--aicpu-thread-num", type=int, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_attention_swa,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_attention_swa,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            block_dim=args.block_dim,
            aicpu_thread_num=args.aicpu_thread_num,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_allclose(atol=6e-3, rtol=2.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
