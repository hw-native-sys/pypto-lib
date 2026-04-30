# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 KV Compressor (decode incremental). Mirrors model.py Compressor (line 279-377);
golden is a direct port of forward's decode branch (prefill `start_pos == 0` path is omitted).
Configurable for compress_ratio ∈ {0, 4, 128} and rotate ∈ {False, True}."""


import pypto.language as pl


B = 16  # demo 4
S = 1
EPS = 1e-6

COMPRESS_RATIO = 4  # 0 / 4 / 128
HEAD_DIM = 512
ROTATE = False

D = 4096  # v4-pro 7168
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)

OUT_DIM = COFF * HEAD_DIM
STATE_LEN = COFF * COMPRESS_RATIO

START_POS = 3  # default for ScalarSpec; >0 (decode) and (START_POS+1)%COMPRESS_RATIO==0 to cover the full compression path
SHOULD_COMPRESS = COMPRESS_RATIO != 0 and ((START_POS + 1) % COMPRESS_RATIO) == 0


def build_deepseek_v4_decode_compressor_program():
    @pl.program
    class DeepSeekV4DecodeCompressor:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_compressor(
            self,
            x: pl.Tensor[[B, S, D], pl.BF16],
            kv_state: pl.InOut[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
            score_state: pl.InOut[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
            wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
            wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
            ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
            norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
            cos: pl.Tensor[[1, ROPE_HEAD_DIM], pl.BF16],  # caller passes freqs_cis[start_pos+1-ratio]
            sin: pl.Tensor[[1, ROPE_HEAD_DIM], pl.BF16],  # same as cos
            hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
            start_pos: pl.Scalar[pl.INT32],  # decode step; varies per call
            out: pl.Out[pl.Tensor[[B, HEAD_DIM], pl.BF16]],
        ):
            # TODO: kernel implementation
            return out

    return DeepSeekV4DecodeCompressor


def golden_deepseek_v4_decode_compressor(tensors):
    """Torch reference for Compressor.forward (decode branch; prefill omitted, quant identity on A3)."""
    import torch

    x = tensors["x"]
    kv_state = tensors["kv_state"]
    score_state = tensors["score_state"]
    wkv = tensors["wkv"].float()
    wgate = tensors["wgate"].float()
    ape = tensors["ape"].float()
    norm_w = tensors["norm_w"].float()
    cos = tensors["cos"].float()
    sin = tensors["sin"].float()
    hadamard = tensors["hadamard"].float()

    start_pos = int(tensors["start_pos"])
    compress_ratio = COMPRESS_RATIO

    bsz, _, _ = x.shape
    ratio, overlap, rotate, d, rd = compress_ratio, OVERLAP, ROTATE, HEAD_DIM, ROPE_HEAD_DIM
    dtype = x.dtype
    x = x.float()
    kv = x.view(bsz, -1) @ wkv.T
    score = x.view(bsz, -1) @ wgate.T

    if start_pos == 0:
        return

    should_compress = (start_pos + 1) % ratio == 0
    score = score + ape[start_pos % ratio]
    if overlap:
        kv_state[:bsz, ratio + start_pos % ratio] = kv
        score_state[:bsz, ratio + start_pos % ratio] = score
        if should_compress:
            kvs = torch.cat([kv_state[:bsz, :ratio, :d], kv_state[:bsz, ratio:, d:]], dim=1)
            scs = torch.cat([score_state[:bsz, :ratio, :d], score_state[:bsz, ratio:, d:]], dim=1)
            kv = (kvs * scs.softmax(dim=1)).sum(dim=1, keepdim=True)
            kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]
            score_state[:bsz, :ratio] = score_state[:bsz, ratio:]
    else:
        kv_state[:bsz, start_pos % ratio] = kv
        score_state[:bsz, start_pos % ratio] = score
        if should_compress:
            kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

    if not should_compress:
        tensors["out"][:] = torch.zeros(B, HEAD_DIM, dtype=torch.bfloat16)
        return

    kv_c = kv.squeeze(1)
    kv_c = kv_c * torch.rsqrt(kv_c.square().mean(-1, keepdim=True) + EPS) * norm_w
    kv_c = kv_c.to(dtype).float()

    x_pair = kv_c[..., -rd:].unflatten(-1, (-1, 2))
    x0, x1 = x_pair[..., 0], x_pair[..., 1]
    cos_v, sin_v = cos.view(-1), sin.view(-1)
    y0 = x0 * cos_v - x1 * sin_v
    y1 = x0 * sin_v + x1 * cos_v
    kv_c = torch.cat([kv_c[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)

    if rotate:
        kv_c = (kv_c @ hadamard).to(torch.bfloat16).float()  # rotate_activation: full Hadamard matmul (v3_2 style)
        # fp4_act_quant — A3-skipped
    else:
        pass  # act_quant — A3-skipped

    tensors["out"][:] = kv_c.to(torch.bfloat16)


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.randn(B, S, D) * 0.1
    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
    def init_score_state():
        return torch.full((B, STATE_LEN, OUT_DIM), float("-inf"))
    def init_wkv():
        return torch.randn(OUT_DIM, D) / D ** 0.5
    def init_wgate():
        return torch.randn(OUT_DIM, D) / D ** 0.5
    def init_ape():
        return torch.randn(COMPRESS_RATIO, OUT_DIM) * 0.01
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cos():
        return torch.cos(torch.arange(ROPE_HEAD_DIM).reshape(1, ROPE_HEAD_DIM) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(ROPE_HEAD_DIM).reshape(1, ROPE_HEAD_DIM) * 1e-3)
    def init_hadamard():
        return torch.eye(HEAD_DIM)
    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_score_state),
        TensorSpec("wkv", [OUT_DIM, D], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [OUT_DIM, D], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [1, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("sin", [1, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        ScalarSpec("start_pos", torch.int32, START_POS),
        TensorSpec("out", [B, HEAD_DIM], torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_compressor_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_compressor,
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
