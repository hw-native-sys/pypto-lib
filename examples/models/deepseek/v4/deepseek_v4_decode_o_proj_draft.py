# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 grouped output projection (decode): grouped wo_a low-rank reduction followed by
wo_b expansion back to model dim."""


import pypto.language as pl


B          = 16                 # demo 4
S          = 1
T          = B * S
D          = 4096               # v4-pro 7168
H          = 64                 # v4-pro 128
HEAD_DIM   = 512
O_LORA     = 1024
O_GROUPS   = 8                  # v4-pro 16
O_GROUP_IN = H * HEAD_DIM // O_GROUPS


def build_deepseek_v4_decode_o_proj_program():
    @pl.program
    class DeepSeekV4DecodeOProj:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_o_proj(
            self,
            o:        pl.Tensor[[T, H, HEAD_DIM],                 pl.BF16],
            wo_a:     pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],   pl.BF16],
            wo_b:     pl.Tensor[[D, O_GROUPS * O_LORA],           pl.BF16],
            attn_out: pl.Out[pl.Tensor[[T, D],                    pl.BF16]],
        ):
            # TODO: kernel implementation
            return attn_out

    return DeepSeekV4DecodeOProj


def golden_deepseek_v4_decode_o_proj(tensors):
    """Torch reference, direct port of model.py Attention.forward 537-542."""
    import torch

    o    = tensors["o"].float()                            # [T, H, HEAD_DIM]
    wo_a = tensors["wo_a"].float()                         # [O_GROUPS, O_LORA, O_GROUP_IN]
    wo_b = tensors["wo_b"].float()                         # [D, O_GROUPS*O_LORA]

    o_g = o.view(T, O_GROUPS, O_GROUP_IN)                  # [T, G, d]
    o_r = torch.einsum("tgd,grd->tgr", o_g, wo_a)          # [T, G, O_LORA]
    out = o_r.flatten(1) @ wo_b.T                          # [T, D]

    tensors["attn_out"][:] = out.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_o():
        return torch.randn(T, H, HEAD_DIM) * 0.05
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / (O_GROUP_IN ** 0.5)
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / ((O_GROUPS * O_LORA) ** 0.5)

    return [
        TensorSpec("o",        [T, H, HEAD_DIM],               torch.bfloat16, init_value=init_o),
        TensorSpec("wo_a",     [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b",     [D, O_GROUPS * O_LORA],         torch.bfloat16, init_value=init_wo_b),
        TensorSpec("attn_out", [T, D],                         torch.bfloat16, is_output=True),
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
        program=build_deepseek_v4_decode_o_proj_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_o_proj,
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
