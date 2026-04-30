# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 grouped output projection fused example.

The kernel packs attention heads by output-projection group, applies the
grouped ``wo_a`` LoRA projection, and then applies ``wo_b`` to produce the
final attention output.
"""


import pypto.language as pl


B          = 16                 # demo 4
S          = 1
T          = B * S
D          = 4096               # v4-pro 7168
H          = 64                 # v4-pro 128
HEAD_DIM   = 512
O_LORA     = 1024
O_GROUPS   = 8                  # v4-pro 16
HEADS_PER_GROUP = H // O_GROUPS          # 8 heads per group
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM  # 4096 (matches v4-pro by coincidence)

# Stage A tile shape (M=T; K over O_GROUP_IN; N over O_LORA).
A_K_CHUNK  = 128
A_N_CHUNK  = 128

# Stage B tile shape (M=T; K over O_GROUPS * O_LORA; N over D).
B_K_CHUNK  = 128
B_N_CHUNK  = 256


def build_deepseek_v4_decode_o_proj_program():
    assert H % O_GROUPS == 0, (
        f"H ({H}) must be divisible by O_GROUPS ({O_GROUPS})"
    )
    assert O_GROUP_IN % A_K_CHUNK == 0, (
        f"O_GROUP_IN ({O_GROUP_IN}) must be divisible by A_K_CHUNK ({A_K_CHUNK})"
    )
    assert O_LORA % A_N_CHUNK == 0, (
        f"O_LORA ({O_LORA}) must be divisible by A_N_CHUNK ({A_N_CHUNK})"
    )
    assert (O_GROUPS * O_LORA) % B_K_CHUNK == 0, (
        f"O_GROUPS * O_LORA ({O_GROUPS * O_LORA}) must be divisible by B_K_CHUNK ({B_K_CHUNK})"
    )
    assert D % B_N_CHUNK == 0, (
        f"D ({D}) must be divisible by B_N_CHUNK ({B_N_CHUNK})"
    )

    A_K_BLOCKS = O_GROUP_IN // A_K_CHUNK
    A_N_BLOCKS = O_LORA // A_N_CHUNK
    B_K_BLOCKS = (O_GROUPS * O_LORA) // B_K_CHUNK
    B_N_BLOCKS = D // B_N_CHUNK

    @pl.program
    class DeepSeekV4DecodeOProj:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_o_proj(
            self,
            o:        pl.Tensor[[T, H, HEAD_DIM],                     pl.BF16],
            wo_a:     pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],       pl.BF16],
            wo_b:     pl.Tensor[[D, O_GROUPS * O_LORA],               pl.BF16],
            attn_out: pl.Out[pl.Tensor[[T, D],                        pl.BF16]],
        ):
            o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
            o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)

            # ---- Prologue: pack o[t, g*hpg:(g+1)*hpg, :] -> o_packed[g*T+t, :] ----
            for g in pl.parallel(0, O_GROUPS, 1):
                head_off = g * HEADS_PER_GROUP
                row_base = g * T
                for t in pl.parallel(0, T, 1):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        src_3d = o[t:t + 1, head_off:head_off + HEADS_PER_GROUP, :]
                        src_2d = pl.reshape(src_3d, [1, O_GROUP_IN])
                        o_packed[row_base + t:row_base + t + 1, :] = src_2d

            # ---- Stage A: o_r[t, g, r] = sum_d o_packed[g*T+t, d] * wo_a[g, d, r] ----
            for g in pl.parallel(0, O_GROUPS, 1):
                row_base_o = g * T
                out_col_g  = g * O_LORA

                for nb in pl.parallel(0, A_N_BLOCKS, 1):
                    n0 = nb * A_N_CHUNK

                    with pl.at(level=pl.Level.CORE_GROUP):
                        xa0_chunk = o_packed[row_base_o:row_base_o + T,
                                             0:A_K_CHUNK]
                        wa0_chunk = wo_a[g:g + 1,
                                         n0:n0 + A_N_CHUNK,
                                         0:A_K_CHUNK]
                        acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                        for kb in pl.range(1, A_K_BLOCKS):
                            k0 = kb * A_K_CHUNK
                            xa_k_chunk = o_packed[row_base_o:row_base_o + T,
                                                  k0:k0 + A_K_CHUNK]
                            wa_k_chunk = wo_a[g:g + 1,
                                              n0:n0 + A_N_CHUNK,
                                              k0:k0 + A_K_CHUNK]
                            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

                    # Separate CORE_GROUP scope so the BF16 cast lands on the
                    # vector path and the GM store sees a row-major tile (the
                    # matmul accumulator is column-major in the cube buffer).
                    with pl.at(level=pl.Level.CORE_GROUP):
                        acc_a_2d = pl.reshape(acc_a, [T, A_N_CHUNK])
                        o_r[:, out_col_g + n0:out_col_g + n0 + A_N_CHUNK] = pl.cast(
                            acc_a_2d, target_type=pl.BF16
                        )

            # ---- Stage B: attn_out = o_r @ wo_b^T  ----
            for nb in pl.parallel(0, B_N_BLOCKS, 1):
                n0 = nb * B_N_CHUNK

                with pl.at(level=pl.Level.CORE_GROUP):
                    xb0_chunk = o_r[:, 0:B_K_CHUNK]
                    wb0_chunk = wo_b[n0:n0 + B_N_CHUNK, 0:B_K_CHUNK]
                    acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.range(1, B_K_BLOCKS):
                        k0 = kb * B_K_CHUNK
                        xb_k_chunk = o_r[:, k0:k0 + B_K_CHUNK]
                        wb_k_chunk = wo_b[n0:n0 + B_N_CHUNK, k0:k0 + B_K_CHUNK]
                        acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

                # Separate CORE_GROUP scope (see Stage A note above).
                with pl.at(level=pl.Level.CORE_GROUP):
                    attn_out[:, n0:n0 + B_N_CHUNK] = pl.cast(acc_b, target_type=pl.BF16)

            return attn_out

    return DeepSeekV4DecodeOProj


def golden_deepseek_v4_decode_o_proj(tensors):
    """Torch reference for model.py Attention.forward L537-541.

    Inputs:
      - ``o``    : ``[T, H, HEAD_DIM]``                (original attention output)
      - ``wo_a`` : ``[O_GROUPS, O_LORA, O_GROUP_IN]`` (upstream layout)
      - ``wo_b`` : ``[D, O_GROUPS * O_LORA]``

    The intermediate ``o_r`` is cast through BF16 to mirror the kernel's
    internal workspace dtype before being multiplied by ``wo_b``.
    """
    import torch

    o    = tensors["o"].float()                            # [T, H, HEAD_DIM]
    wo_a = tensors["wo_a"].float()                         # [G, O_LORA, O_GROUP_IN]
    wo_b = tensors["wo_b"].float()                         # [D, G*O_LORA]

    # Reconstruct the upstream einsum layouts:
    #   o    -> [B, S, G, O_GROUP_IN]    (original view of [T, H, HEAD_DIM])
    #   wo_a -> [G, R, O_GROUP_IN]
    o_model    = o.view(B, S, O_GROUPS, O_GROUP_IN)
    wo_a_model = wo_a                                      # [G, R, O_GROUP_IN]

    # Stage A: upstream einsum verbatim.
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a_model)  # [B, S, G, R]
    # Match kernel's internal BF16 workspace precision before Stage B.
    o_r = o_r.to(torch.bfloat16).float()
    # Stage B: wo_b expansion back to model dim.
    out = o_r.flatten(2).view(T, O_GROUPS * O_LORA) @ wo_b.T   # [T, D]

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
        TensorSpec("o",        [T, H, HEAD_DIM],                    torch.bfloat16, init_value=init_o),
        TensorSpec("wo_a",     [O_GROUPS, O_LORA, O_GROUP_IN],      torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b",     [D, O_GROUPS * O_LORA],              torch.bfloat16, init_value=init_wo_b),
        TensorSpec("attn_out", [T, D],                              torch.bfloat16, is_output=True),
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
