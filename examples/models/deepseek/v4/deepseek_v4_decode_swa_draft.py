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
Companion files: deepseek_v4_decode_csa.py (ratio=4)
                 deepseek_v4_decode_hca.py (ratio=128)."""


import pypto.language as pl


B = 16  # demo 4
S = 1
T = B * S
EPS = 1e-6

D = 4096  # v4-pro 7168
H = 64  # v4-pro 128
HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
Q_LORA = 1024  # v4-pro 1536
WIN = 128
SOFTMAX_SCALE = HEAD_DIM ** -0.5

HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT
HC_DIM = HC_MULT * D
HC_SINKHORN_ITER = 20
HC_EPS = 1e-6

MAX_SEQ_LEN = 4096  # v4-pro 1048576 (1M tokens)

O_LORA = 1024
O_GROUPS = 8  # v4-pro 16
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

BLOCK_SIZE = 128
ORI_MAX_BLOCKS = 1                                         # WIN==BLOCK_SIZE → 1 block per batch for ori
MAX_BLOCKS = ORI_MAX_BLOCKS                                # SWA: only ori, no cmp portion
BLOCK_NUM = B * MAX_BLOCKS

TOPK = WIN                                                 # SWA: sparse_attn topk = window only

START_POS = 3  # default for ScalarSpec; >0 (decode); SWA path has no compression-related constraint


def build_deepseek_v4_decode_swa_program():
    @pl.program
    class DeepSeekV4DecodeSwa:
        @pl.function(type=pl.FunctionType.Orchestration)
        def deepseek_v4_decode_swa(
            self,
            x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
            # hc_pre weights
            hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
            hc_attn_scale: pl.Tensor[[3], pl.FP32],
            hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
            # qkv_proj_rope weights
            attn_norm_w: pl.Tensor[[D], pl.FP32],            # Block.attn_norm.weight (model.py:680)
            wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
            wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.BF16],
            wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
            gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
            gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
            freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],  # base rope_theta (no YaRN) for SWA
            freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
            # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
            kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
            block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
            # sparse_attn
            attn_sink: pl.Tensor[[H], pl.FP32],
            # o_proj
            wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
            wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
            start_pos: pl.Scalar[pl.INT32],  # decode step; varies per call
            x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
        ):
            # TODO: orchestration body (dispatches the per-step kernels)
            return x_out

    return DeepSeekV4DecodeSwa


def golden_deepseek_v4_decode_swa(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    import torch

    from deepseek_v4_decode_hc_pre import golden_deepseek_v4_decode_hc_pre
    from deepseek_v4_decode_qkv_proj_rope_draft import golden_deepseek_v4_decode_qkv_proj_rope
    from deepseek_v4_decode_sparse_attn_draft import golden_deepseek_v4_decode_sparse_attn
    from deepseek_v4_decode_o_proj import golden_deepseek_v4_decode_o_proj
    from deepseek_v4_decode_hc_post_draft import golden_deepseek_v4_decode_hc_post

    # ---- Block.hc_pre (model.py:691) ----
    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT)
    golden_deepseek_v4_decode_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # ===== Attention.forward (model.py:484-543), ratio==0 branch =====
    start_pos = int(tensors["start_pos"])
    bsz, seqlen, _ = x_mixed.shape
    win = WIN
    rd = ROPE_HEAD_DIM

    if start_pos == 0:
        return  # prefill — decode-only orchestration skips

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    step_cos = freqs_cos[start_pos:start_pos + 1]                            # [1, rd]
    step_sin = freqs_sin[start_pos:start_pos + 1]
    rope_cos_T = step_cos.expand(T, rd).contiguous()
    rope_sin_T = step_sin.expand(T, rd).contiguous()

    # q + win kv (model.py:495-504)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.bfloat16)
    golden_deepseek_v4_decode_qkv_proj_rope({
        "x": x_mixed,
        "norm_w": tensors["attn_norm_w"],
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_T,
        "rope_sin": rope_sin_T,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,                                                              # qr unused on SWA path
    })

    # window topk only (model.py:507; ratio==0 skips lines 508-514)
    topk_idxs = torch.full((T, TOPK), -1, dtype=torch.int32)
    topk_idxs[:, :win] = torch.arange(win, dtype=torch.int32)

    # ori_kv scatter (model.py:530)
    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]
    ori_slot = start_pos % win
    for b in range(B):
        blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
        intra = ori_slot % BLOCK_SIZE
        kv_cache[blk_id, intra, 0] = kv[b]

    # sparse_attn (model.py:533); cmp_kv slot is empty on SWA → reuse same pool with empty cmp_block_table
    o = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    empty_cmp_block_table = torch.full((B, 0), -1, dtype=torch.int32)
    golden_deepseek_v4_decode_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": block_table[:, :ORI_MAX_BLOCKS],
        "cmp_kv": kv_cache,
        "cmp_block_table": empty_cmp_block_table,
        "cmp_sparse_indices": topk_idxs,
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "o": o,
    })

    # o_proj (model.py:537-542)
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_deepseek_v4_decode_o_proj({
        "o": o,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "attn_out": attn_out,
    })

    # ===== Block.hc_post (model.py:694) =====
    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_deepseek_v4_decode_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y


def build_tensor_specs():
    import torch  # type: ignore[import]
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
    def init_kv_cache():
        return torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)

    def init_block_table():
        tbl = torch.full((B, MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(MAX_BLOCKS):
                tbl[b, j] = b * MAX_BLOCKS + j
        return tbl

    def init_attn_sink():
        return torch.zeros(H)
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.float32, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("block_table", [B, MAX_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.bfloat16, init_value=init_wo_b),
        ScalarSpec("start_pos", torch.int32, START_POS),
        TensorSpec("x_out", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_swa_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_swa,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
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
