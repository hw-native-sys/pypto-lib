# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 CSA (Compressed Sparse Attention) decode orchestration — `compress_ratio == 4` path.
Active in layers 2/4/6 of the model (3 of the 8 layers in demo, 4 of 60 in v4-pro).
Composes hc_pre + qkv_proj_rope + main compressor (ratio=4, overlap=True) + indexer (ratio=4)
+ sparse_attn + o_proj + hc_post. Topk for sparse_attn is window_topk ⧺ indexer_topk.
Companion files: deepseek_v4_decode_swa.py (ratio=0, no compressor/indexer)
                 deepseek_v4_decode_hca.py (ratio=128, compressor only, no indexer)."""


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

IDX_N_HEADS = 64
IDX_HEAD_DIM = 128
IDX_TOPK = 512  # v4-pro 1024
IDX_SOFTMAX_SCALE = IDX_HEAD_DIM ** -0.5
MAX_SEQ_LEN = 4096  # v4-pro 1048576 (1M tokens)

COMPRESS_RATIO = 4  # CSA
ROTATE_MAIN = False
ROTATE_INNER = True
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)

MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_LEN = COFF * COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO

O_LORA = 1024
O_GROUPS = 8  # v4-pro 16
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

BLOCK_SIZE = 128
ORI_MAX_BLOCKS = 1                                         # WIN==BLOCK_SIZE → 1 block per batch for ori
CMP_MAX_BLOCKS = 64  # v4-pro 2048 (=MAX_SEQ_LEN/ratio/BLOCK_SIZE = 1048576/4/128)
MAX_BLOCKS = ORI_MAX_BLOCKS + CMP_MAX_BLOCKS               # logical block layout: [0..ORI) ori, [ORI..MAX) cmp
BLOCK_NUM = B * MAX_BLOCKS

TOPK = WIN + IDX_TOPK

START_POS = 3  # default for ScalarSpec; >0 (decode) and (START_POS+1)%COMPRESS_RATIO==0 to cover the full compression path
SHOULD_COMPRESS = COMPRESS_RATIO != 0 and ((START_POS + 1) % COMPRESS_RATIO) == 0
OFFSET = WIN  # added to indexer topk_idxs (model.py:432, attention.py:509)


def build_deepseek_v4_decode_csa_program():
    @pl.program
    class DeepSeekV4DecodeCsa:
        @pl.function(type=pl.FunctionType.Orchestration)
        def deepseek_v4_decode_csa(
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
            freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],  # Attention.freqs_cis (model.py:480-482)
            freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
            # main compressor (rotate=False, head_dim=HEAD_DIM)
            cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
            cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
            cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
            cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
            cmp_kv_state: pl.InOut[pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32]],
            cmp_score_state: pl.InOut[pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32]],
            # indexer + inner compressor (rotate=True, head_dim=IDX_HEAD_DIM)
            idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.BF16],
            weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
            hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by indexer's q rotation and inner Compressor
            inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
            inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
            inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
            inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
            inner_kv_state: pl.InOut[pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32]],
            inner_score_state: pl.InOut[pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32]],
            # KV cache (single PA pool: [0, WIN) is ori sliding window, [WIN, WIN+max_seq//ratio) is cmp)
            kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
            block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
            idx_kv_cache: pl.InOut[pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16]],
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

    return DeepSeekV4DecodeCsa


def golden_deepseek_v4_decode_csa(tensors):
    """End-to-end orchestration for the ratio=4 (CSA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==4 path with indexer)
    + Block.hc_post; preserves all control flow from Attention.forward (model.py:484-543)."""
    import torch

    from deepseek_v4_decode_hc_pre import golden_deepseek_v4_decode_hc_pre
    from deepseek_v4_decode_qkv_proj_rope_draft import golden_deepseek_v4_decode_qkv_proj_rope
    from deepseek_v4_decode_compressor_draft import golden_deepseek_v4_decode_compressor
    from deepseek_v4_decode_indexer_draft import golden_deepseek_v4_decode_indexer
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

    # ===== Attention.forward (model.py:484-543) =====
    start_pos = int(tensors["start_pos"])
    bsz, seqlen, _ = x_mixed.shape
    win = WIN
    ratio = COMPRESS_RATIO
    rd = ROPE_HEAD_DIM
    should_compress = ratio != 0 and ((start_pos + 1) % ratio) == 0

    if start_pos == 0:
        return  # prefill — decode-only orchestration skips

    # Slice freqs_cis at the positions used by each call site (model.py:486, 366, 404).
    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    step_cos = freqs_cos[start_pos:start_pos + 1]                            # [1, rd]
    step_sin = freqs_sin[start_pos:start_pos + 1]
    rope_cos_T = step_cos.expand(T, rd).contiguous()                          # qkv_proj_rope / sparse_attn want [T, rd]
    rope_sin_T = step_sin.expand(T, rd).contiguous()
    cmp_cos = freqs_cos[start_pos + 1 - ratio:start_pos + 2 - ratio] if ratio else step_cos  # Compressor.forward line 366
    cmp_sin = freqs_sin[start_pos + 1 - ratio:start_pos + 2 - ratio] if ratio else step_sin

    # q + win kv (model.py:495-504); attn_norm fused at input.
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
        "qr": qr,
    })
    # line 506 act_quant on kv non-rope dims — A3-skipped

    # window topk + (optional) compress topk (model.py:507-515)
    topk_idxs = torch.full((T, TOPK), -1, dtype=torch.int32)
    topk_idxs[:, :win] = torch.arange(win, dtype=torch.int32)             # line 507: get_window_topk_idxs
    if ratio:                                                              # line 508
        offset = win                                                       # line 509 (decode: kv.size(1) is N/A)
        if ratio == 4:                                                     # line 510 (indexer is not None)
            indexer_topk = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
            golden_deepseek_v4_decode_indexer({                            # line 511
                "x": x_mixed,
                "qr": qr,
                "wq_b": tensors["idx_wq_b"],
                "weights_proj": tensors["weights_proj"],
                "cos": step_cos,
                "sin": step_sin,
                "hadamard": tensors["hadamard_idx"],
                "inner_wkv": tensors["inner_wkv"],
                "inner_wgate": tensors["inner_wgate"],
                "inner_ape": tensors["inner_ape"],
                "inner_norm_w": tensors["inner_norm_w"],
                "inner_cos": cmp_cos,
                "inner_sin": cmp_sin,
                "inner_kv_state": tensors["inner_kv_state"],
                "inner_score_state": tensors["inner_score_state"],
                "idx_kv_cache": tensors["idx_kv_cache"],
                "start_pos": tensors["start_pos"],
                "offset": torch.tensor(offset, dtype=torch.int32),
                "topk_idxs": indexer_topk,
            })
            compress_topk_idxs = indexer_topk
        else:                                                              # line 512: ratio == 128, HCA path
            # line 513: get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset) — decode branch (line 270-271)
            cache_len = (start_pos + 1) // ratio
            compress_topk_idxs = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
            if cache_len > 0:
                k = min(cache_len, IDX_TOPK)
                compress_topk_idxs[:, :k] = torch.arange(k, dtype=torch.int32) + offset
        topk_idxs[:, win:win + IDX_TOPK] = compress_topk_idxs              # line 514
    topk_idxs = topk_idxs.int()                                            # line 515

    # compress kv & attn — decode branch (model.py:529-534)
    # Single merged PA pool: logical block layout per batch is [0..ORI_MAX_BLOCKS) ori, [ORI_MAX_BLOCKS..MAX) cmp.
    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]

    # line 530: self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1) — ori sliding-window scatter
    ori_slot = start_pos % win
    for b in range(B):
        blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
        intra = ori_slot % BLOCK_SIZE
        kv_cache[blk_id, intra, 0] = kv[b]

    if ratio:                                                              # line 531
        # line 532: self.compressor(x, start_pos) — Compressor.forward writes cmp_kv internally (line 376),
        # we externalize that slot write to the orch below.
        cmp_out = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)
        golden_deepseek_v4_decode_compressor({
            "x": x_mixed,
            "kv_state": tensors["cmp_kv_state"],
            "score_state": tensors["cmp_score_state"],
            "wkv": tensors["cmp_wkv"],
            "wgate": tensors["cmp_wgate"],
            "ape": tensors["cmp_ape"],
            "norm_w": tensors["cmp_norm_w"],
            "cos": cmp_cos,
            "sin": cmp_sin,
            "hadamard": torch.eye(HEAD_DIM, dtype=torch.bfloat16),         # main compressor is rotate=False; unused, identity placeholder
            "start_pos": tensors["start_pos"],
            "out": cmp_out,
        })
        if should_compress:                                                # Compressor.forward line 360 short-circuit
            # cmp portion starts at ORI_MAX_BLOCKS in the logical block table
            cmp_slot_rel = start_pos // ratio                              # relative to cmp portion
            for b in range(B):
                blk_id = int(block_table[b, ORI_MAX_BLOCKS + cmp_slot_rel // BLOCK_SIZE].item())
                intra = cmp_slot_rel % BLOCK_SIZE
                kv_cache[blk_id, intra, 0] = cmp_out[b]

    # line 533: o = sparse_attn(q, self.kv_cache[:bsz], attn_sink, topk_idxs, softmax_scale)
    # line 534: apply_rotary_emb(o[..., -rd:], freqs_cis, True) — fused inside sparse_attn
    # sparse_attn still expects two views; share the physical pool, split block_table at ORI_MAX_BLOCKS.
    o = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    golden_deepseek_v4_decode_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": block_table[:, :ORI_MAX_BLOCKS],
        "cmp_kv": kv_cache,
        "cmp_block_table": block_table[:, ORI_MAX_BLOCKS:],
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
    def init_cmp_wkv():
        return torch.randn(MAIN_OUT_DIM, D) / D ** 0.5
    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) / D ** 0.5
    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.01
    def init_cmp_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cmp_kv_state():
        return torch.zeros(B, MAIN_STATE_LEN, MAIN_OUT_DIM)
    def init_cmp_score_state():
        return torch.full((B, MAIN_STATE_LEN, MAIN_OUT_DIM), float("-inf"))
    def init_idx_wq_b():
        return torch.randn(Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM) / Q_LORA ** 0.5
    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) / D ** 0.5
    def init_hadamard_idx():
        return torch.eye(IDX_HEAD_DIM)
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) / D ** 0.5
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) / D ** 0.5
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.01
    def init_inner_norm_w():
        return torch.ones(IDX_HEAD_DIM)
    def init_inner_kv_state():
        return torch.zeros(B, INNER_STATE_LEN, INNER_OUT_DIM)
    def init_inner_score_state():
        return torch.full((B, INNER_STATE_LEN, INNER_OUT_DIM), float("-inf"))
    def init_kv_cache():
        return torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_idx_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, IDX_HEAD_DIM)

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
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec("cmp_kv_state", [B, MAIN_STATE_LEN, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_kv_state),
        TensorSpec("cmp_score_state", [B, MAIN_STATE_LEN, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_score_state),
        TensorSpec("idx_wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.bfloat16, init_value=init_idx_wq_b),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("hadamard_idx", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard_idx),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("inner_kv_state", [B, INNER_STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_kv_state),
        TensorSpec("inner_score_state", [B, INNER_STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_score_state),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("block_table", [B, MAX_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("idx_kv_cache", [B, IDX_KV_LEN, IDX_HEAD_DIM], torch.bfloat16, init_value=init_idx_kv_cache),
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
        program=build_deepseek_v4_decode_csa_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_csa,
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
