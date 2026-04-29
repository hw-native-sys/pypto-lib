# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Indexer (decode). Mirrors model.py Indexer (line 380-433);
golden is a port of forward's decode branch (prefill `start_pos == 0` path is omitted).
The inner Compressor is invoked via golden_deepseek_v4_decode_compressor (placeholder)."""


import pypto.language as pl

from deepseek_v4_decode_compressor_draft import golden_deepseek_v4_decode_compressor


B = 16  # demo 4
S = 1
T = B * S
EPS = 1e-6

D = 4096  # v4-pro 7168
Q_LORA = 1024  # v4-pro 1536
ROPE_HEAD_DIM = 64

IDX_N_HEADS = 64
IDX_HEAD_DIM = 128
IDX_NOPE_HEAD_DIM = IDX_HEAD_DIM - ROPE_HEAD_DIM
IDX_TOPK = 512  # v4-pro 1024
IDX_SOFTMAX_SCALE = IDX_HEAD_DIM ** -0.5

COMPRESS_RATIO = 4
ROTATE = True  # inner compressor always uses rotate=True (model.py:398)
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO

MAX_SEQ_LEN = 4096
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO

START_POS = 3  # >0 (decode), and (START_POS+1)%COMPRESS_RATIO==0 to cover the full inner-compressor path
SHOULD_COMPRESS = COMPRESS_RATIO != 0 and ((START_POS + 1) % COMPRESS_RATIO) == 0
OFFSET = 128  # = win in attention orch; added to topk_idxs (model.py:432)


def build_deepseek_v4_decode_indexer_program():
    @pl.program
    class DeepSeekV4DecodeIndexer:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_indexer(
            self,
            x: pl.Tensor[[B, S, D], pl.BF16],
            qr: pl.Tensor[[T, Q_LORA], pl.BF16],
            wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.BF16],
            weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
            cos: pl.Tensor[[1, ROPE_HEAD_DIM], pl.BF16],  # caller passes freqs_cis[start_pos]
            sin: pl.Tensor[[1, ROPE_HEAD_DIM], pl.BF16],
            hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],  # shared by q rotation and inner Compressor
            inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
            inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
            inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
            inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
            inner_cos: pl.Tensor[[1, ROPE_HEAD_DIM], pl.BF16],  # caller passes freqs_cis[start_pos+1-ratio]
            inner_sin: pl.Tensor[[1, ROPE_HEAD_DIM], pl.BF16],
            inner_kv_state: pl.InOut[pl.Tensor[[B, STATE_LEN, INNER_OUT_DIM], pl.FP32]],
            inner_score_state: pl.InOut[pl.Tensor[[B, STATE_LEN, INNER_OUT_DIM], pl.FP32]],
            idx_kv_cache: pl.InOut[pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16]],
            topk_idxs: pl.Out[pl.Tensor[[T, IDX_TOPK], pl.INT32]],
        ):
            # TODO: kernel implementation
            return topk_idxs

    return DeepSeekV4DecodeIndexer


def golden_deepseek_v4_decode_indexer(tensors):
    """Torch reference for Indexer.forward (decode branch; prefill omitted, fp4_act_quant identity on A3)."""
    import torch

    x = tensors["x"]
    qr = tensors["qr"].float()
    wq_b = tensors["wq_b"].float()
    weights_proj = tensors["weights_proj"].float()
    cos = tensors["cos"].float()
    sin = tensors["sin"].float()
    hadamard = tensors["hadamard"].float()
    idx_kv_cache = tensors["idx_kv_cache"]

    start_pos = START_POS
    compress_ratio = COMPRESS_RATIO
    offset = OFFSET

    bsz, seqlen, _ = x.shape
    ratio, rd = compress_ratio, ROPE_HEAD_DIM
    end_pos = start_pos + seqlen

    if start_pos == 0:
        return

    q = (qr @ wq_b).view(T, IDX_N_HEADS, IDX_HEAD_DIM)

    x_pair = q[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y0 = x0 * cos_v - x1 * sin_v
    y1 = x0 * sin_v + x1 * cos_v
    q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    q = (q.view(-1, IDX_HEAD_DIM) @ hadamard).view(T, IDX_N_HEADS, IDX_HEAD_DIM)
    # fp4_act_quant — A3-skipped

    inner_out = torch.zeros(bsz, IDX_HEAD_DIM, dtype=torch.bfloat16)
    inner_tensors = {
        "x": x,
        "kv_state": tensors["inner_kv_state"],
        "score_state": tensors["inner_score_state"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cos": tensors["inner_cos"],
        "sin": tensors["inner_sin"],
        "hadamard": tensors["hadamard"],
        "out": inner_out,
    }
    # Placeholder call — compressor's golden currently uses module-level constants
    # (HEAD_DIM=512, ROTATE=False), so this won't run end-to-end without refactor.
    golden_deepseek_v4_decode_compressor(inner_tensors)
    if SHOULD_COMPRESS:
        idx_kv_cache[:bsz, start_pos // ratio] = inner_out

    weights = (x.float().view(bsz, -1) @ weights_proj) * (IDX_SOFTMAX_SCALE * IDX_N_HEADS ** -0.5)
    weights = weights.view(T, IDX_N_HEADS)

    cache_len = end_pos // ratio
    kv_view = idx_kv_cache[:bsz, :cache_len].float()
    score = torch.einsum("thd,btd->bht", q, kv_view)
    score = (torch.relu(score) * weights.view(bsz, IDX_N_HEADS, 1)).sum(dim=1)

    k = min(IDX_TOPK, cache_len)
    _, idx = score.topk(k, dim=-1)
    topk_idxs = torch.full((bsz, IDX_TOPK), -1, dtype=torch.int32)
    topk_idxs[:, :k] = idx.to(torch.int32)
    topk_idxs[:, :k] += offset

    tensors["topk_idxs"][:] = topk_idxs


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.1
    def init_qr():
        return torch.randn(T, Q_LORA) * 0.1
    def init_wq_b():
        return torch.randn(Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM) / Q_LORA ** 0.5
    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) / D ** 0.5
    def init_cos():
        return torch.cos(torch.arange(ROPE_HEAD_DIM).reshape(1, ROPE_HEAD_DIM) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(ROPE_HEAD_DIM).reshape(1, ROPE_HEAD_DIM) * 1e-3)
    def init_hadamard():
        return torch.eye(IDX_HEAD_DIM)
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) / D ** 0.5
    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) / D ** 0.5
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.01
    def init_inner_norm_w():
        return torch.ones(IDX_HEAD_DIM)
    def init_inner_cos():
        return torch.cos(torch.arange(ROPE_HEAD_DIM).reshape(1, ROPE_HEAD_DIM) * 1e-3)
    def init_inner_sin():
        return torch.sin(torch.arange(ROPE_HEAD_DIM).reshape(1, ROPE_HEAD_DIM) * 1e-3)
    def init_inner_kv_state():
        return torch.zeros(B, STATE_LEN, INNER_OUT_DIM)
    def init_inner_score_state():
        return torch.full((B, STATE_LEN, INNER_OUT_DIM), float("-inf"))
    def init_idx_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, IDX_HEAD_DIM)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [T, Q_LORA], torch.bfloat16, init_value=init_qr),
        TensorSpec("wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("cos", [1, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("sin", [1, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("inner_cos", [1, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_inner_cos),
        TensorSpec("inner_sin", [1, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_inner_sin),
        TensorSpec("inner_kv_state", [B, STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_kv_state),
        TensorSpec("inner_score_state", [B, STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_score_state),
        TensorSpec("idx_kv_cache", [B, IDX_KV_LEN, IDX_HEAD_DIM], torch.bfloat16, init_value=init_idx_kv_cache),
        TensorSpec("topk_idxs", [T, IDX_TOPK], torch.int32, is_output=True),
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
        program=build_deepseek_v4_decode_indexer_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_indexer,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
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
