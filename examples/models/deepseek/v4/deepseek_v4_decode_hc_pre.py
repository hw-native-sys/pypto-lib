# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections pre-mix (decode): mixes the hc-stack into a single sublayer input
and produces the post/comb weights used by hc_post."""


import pypto.language as pl


B                = 16               # demo 4
S                = 1
D                = 4096             # v4-pro 7168
HC_MULT          = 4
MIX_HC           = (2 + HC_MULT) * HC_MULT
MIX_PAD          = 32
HC_PAD           = 8
HC_DIM           = HC_MULT * D
HC_SINKHORN_ITER = 20
HC_EPS           = 1e-6
NORM_EPS         = 1e-6
NEG_INF          = -1e20
T                = B * S
T_TILE           = 16
K_CHUNK          = 512
D_CHUNK          = 512
HC_DIM_INV       = 1.0 / HC_DIM
HC_DIM_BLOCKS    = HC_DIM // K_CHUNK
D_BLOCKS         = D // D_CHUNK


def build_deepseek_v4_decode_hc_pre_program():
    @pl.program
    class DeepSeekV4DecodeHcPre:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v4_decode_hc_pre(
            self,
            x:        pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
            hc_fn:    pl.Tensor[[MIX_HC, HC_DIM],   pl.FP32],
            hc_scale: pl.Tensor[[3],                pl.FP32],
            hc_base:  pl.Tensor[[MIX_HC],           pl.FP32],
            x_mixed:  pl.Out[pl.Tensor[[B, S, D],            pl.BF16]],
            post:     pl.Out[pl.Tensor[[B, S, HC_MULT],      pl.FP32]],
            comb:     pl.Out[pl.Tensor[[B, S, HC_MULT, HC_MULT], pl.FP32]],
        ):
            x_flat = pl.reshape(x, [T, HC_DIM])
            post_flat = pl.reshape(post, [T * HC_MULT])
            comb_flat = pl.reshape(comb, [T * HC_MULT * HC_MULT])
            x_flat_fp32 = pl.create_tensor([T, HC_DIM], dtype=pl.FP32)
            inv_rms = pl.create_tensor([1, T], dtype=pl.FP32)
            mixes = pl.create_tensor([T, MIX_PAD], dtype=pl.FP32)

            for kb in pl.parallel(HC_DIM_BLOCKS):
                k0 = kb * K_CHUNK
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="cast_x"):
                    x_chunk_fp32 = pl.cast(
                        pl.slice(x_flat, [T, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    x_flat_fp32 = pl.assemble(x_flat_fp32, x_chunk_fp32, [0, k0])

            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="rms"):
                sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(HC_DIM_BLOCKS, stage=4):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.slice(x_flat_fp32, [T, K_CHUNK], [0, k0])
                    sq_sum = pl.add(
                        sq_sum,
                        pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T]),
                    )
                inv_rms_val = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)))
                inv_rms = pl.assemble(inv_rms, inv_rms_val, [0, 0])

            mixes_flat = pl.reshape(mixes, [T * MIX_PAD])
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="linear"):
                for m in pl.range(MIX_HC):
                    mix_col = pl.full([1, T], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(HC_DIM_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_lin_chunk = pl.slice(x_flat_fp32, [T, K_CHUNK], [0, k0])
                        w_row = pl.slice(hc_fn, [1, K_CHUNK], [m, k0])
                        prod = pl.col_expand_mul(x_lin_chunk, w_row)
                        mix_col = pl.add(
                            mix_col,
                            pl.reshape(pl.row_sum(prod), [1, T]),
                        )
                    mix_col = pl.mul(mix_col, inv_rms)
                    mix_col_flat = pl.reshape(mix_col, [T])
                    for t in pl.unroll(T):
                        pl.write(
                            mixes_flat,
                            [t * MIX_PAD + m],
                            pl.read(mix_col_flat, [t]),
                        )
            mixes = pl.reshape(mixes_flat, [T, MIX_PAD])

            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="split_pre_post"):
                scale0 = pl.tensor.read(hc_scale, [0])
                scale1 = pl.tensor.read(hc_scale, [1])
                scale2 = pl.tensor.read(hc_scale, [2])

                ones_hc = pl.full([T, HC_PAD], dtype=pl.FP32, value=1.0)
                pre_base = pl.reshape(pl.slice(hc_base, [HC_PAD], [0]), [1, HC_PAD])
                pre_logits = pl.add(
                    pl.mul(pl.slice(mixes, [T, HC_PAD], [0, 0], valid_shape=[T, HC_MULT]), scale0),
                    pl.col_expand_mul(ones_hc, pre_base),
                )
                pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)

                post_base = pl.reshape(pl.slice(hc_base, [HC_PAD], [HC_MULT]), [1, HC_PAD])
                post_logits = pl.add(
                    pl.mul(pl.slice(mixes, [T, HC_PAD], [0, HC_MULT]), scale1),
                    pl.col_expand_mul(ones_hc, post_base),
                )
                post_pad = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)

                ones_comb = pl.full([T, HC_MULT * HC_MULT], dtype=pl.FP32, value=1.0)
                comb_base = pl.reshape(
                    pl.slice(hc_base, [HC_MULT * HC_MULT], [HC_MULT * 2]),
                    [1, HC_MULT * HC_MULT],
                )
                comb_mix = pl.slice(mixes, [T, HC_MULT * HC_MULT], [0, HC_MULT * 2])
                comb_logits = pl.add(
                    pl.mul(comb_mix, scale2),
                    pl.col_expand_mul(ones_comb, comb_base),
                )

            post_pad_flat = pl.reshape(post_pad, [T * HC_PAD])
            comb_pad = pl.create_tensor([HC_MULT, T, HC_PAD], dtype=pl.FP32)

            comb_pad = self.deepseek_v4_decode_hc_comb_tile(comb_logits, comb_pad)
            comb_pad_flat = pl.reshape(comb_pad, [HC_MULT * T * HC_PAD])

            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="write_post"):
                for t in pl.parallel(0, T, 1, chunk=16):
                    src_base = t * HC_PAD
                    dst_base = t * HC_MULT
                    for h in pl.unroll(HC_MULT):
                        src_idx = src_base + h
                        dst_idx = dst_base + h
                        pl.write(
                            post_flat,
                            [dst_idx],
                            pl.read(post_pad_flat, [src_idx]),
                        )

            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="write_comb"):
                for t in pl.parallel(0, T, 1, chunk=16):
                    dst_base = t * HC_MULT * HC_MULT
                    for c in pl.unroll(HC_MULT):
                        pl.write(
                            comb_flat,
                            [dst_base + 0 * HC_MULT + c],
                            pl.read(comb_pad_flat, [0 * T * HC_PAD + t * HC_PAD + c]),
                        )
                        pl.write(
                            comb_flat,
                            [dst_base + 1 * HC_MULT + c],
                            pl.read(comb_pad_flat, [1 * T * HC_PAD + t * HC_PAD + c]),
                        )
                        pl.write(
                            comb_flat,
                            [dst_base + 2 * HC_MULT + c],
                            pl.read(comb_pad_flat, [2 * T * HC_PAD + t * HC_PAD + c]),
                        )
                        pl.write(
                            comb_flat,
                            [dst_base + 3 * HC_MULT + c],
                            pl.read(comb_pad_flat, [3 * T * HC_PAD + t * HC_PAD + c]),
                        )

            pre_val_flat = pl.reshape(pre_val, [T * HC_PAD])
            x_mixed_view = pl.reshape(x_mixed, [T, D])
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="mix_x"):
                for t in pl.parallel(0, T, 1, chunk=16):
                    for db in pl.range(D_BLOCKS):
                        d0 = db * D_CHUNK
                        y_row = pl.full([1, D_CHUNK], dtype=pl.FP32, value=0.0)
                        for h in pl.range(HC_MULT):
                            pre_th = pl.read(pre_val_flat, [t * HC_PAD + h])
                            x_row = pl.slice(x_flat_fp32, [1, D_CHUNK], [t, h * D + d0])
                            y_row = pl.add(y_row, pl.mul(x_row, pre_th))
                        x_mixed_view = pl.assemble(
                            x_mixed_view,
                            pl.cast(y_row, target_type=pl.BF16),
                            [t, d0],
                        )
            x_mixed = pl.reshape(x_mixed_view, [B, S, D])
            return x_mixed, post, comb

        @pl.function(type=pl.FunctionType.InCore)
        def deepseek_v4_decode_hc_comb_tile(
            self,
            comb_logits: pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32],
            comb:        pl.Out[pl.Tensor[[HC_MULT, T, HC_PAD], pl.FP32]],
        ):
            row0 = pl.fillpad(pl.load(
                comb_logits,
                [0, 0 * HC_MULT],
                [T, HC_PAD],
                valid_shapes=[T, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row1 = pl.fillpad(pl.load(
                comb_logits,
                [0, 1 * HC_MULT],
                [T, HC_PAD],
                valid_shapes=[T, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row2 = pl.fillpad(pl.load(
                comb_logits,
                [0, 2 * HC_MULT],
                [T, HC_PAD],
                valid_shapes=[T, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)
            row3 = pl.fillpad(pl.load(
                comb_logits,
                [0, 3 * HC_MULT],
                [T, HC_PAD],
                valid_shapes=[T, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            ), pad_value=pl.PadValue.min)

            row_max_tmp = pl.create_tile([T, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_sum_tmp = pl.create_tile([T, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row0_exp = pl.exp(pl.row_expand_sub(row0, pl.row_max(row0, row_max_tmp)))
            row1_exp = pl.exp(pl.row_expand_sub(row1, pl.row_max(row1, row_max_tmp)))
            row2_exp = pl.exp(pl.row_expand_sub(row2, pl.row_max(row2, row_max_tmp)))
            row3_exp = pl.exp(pl.row_expand_sub(row3, pl.row_max(row3, row_max_tmp)))
            row0_soft = pl.add(pl.row_expand_div(row0_exp, pl.row_sum(row0_exp, row_sum_tmp)), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, pl.row_sum(row1_exp, row_sum_tmp)), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, pl.row_sum(row2_exp, row_sum_tmp)), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, pl.row_sum(row3_exp, row_sum_tmp)), HC_EPS)

            row0_eff = pl.tile.fillpad(pl.tile.set_validshape(row0_soft, T, HC_MULT), pad_value=pl.PadValue.zero)
            row1_eff = pl.tile.fillpad(pl.tile.set_validshape(row1_soft, T, HC_MULT), pad_value=pl.PadValue.zero)
            row2_eff = pl.tile.fillpad(pl.tile.set_validshape(row2_soft, T, HC_MULT), pad_value=pl.PadValue.zero)
            row3_eff = pl.tile.fillpad(pl.tile.set_validshape(row3_soft, T, HC_MULT), pad_value=pl.PadValue.zero)

            row_sum_tmp_iter = pl.create_tile([T, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_eff, col_sum)
            row1_cur = pl.div(row1_eff, col_sum)
            row2_cur = pl.div(row2_eff, col_sum)
            row3_cur = pl.div(row3_eff, col_sum)

            for _ in pl.unroll(HC_SINKHORN_ITER - 1):
                row0_norm = pl.row_expand_div(row0_cur, pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS))
                row1_norm = pl.row_expand_div(row1_cur, pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS))
                row2_norm = pl.row_expand_div(row2_cur, pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS))
                row3_norm = pl.row_expand_div(row3_cur, pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS))
                col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
                col_sum = pl.add(col_sum, HC_EPS)
                row0_cur = pl.div(row0_norm, col_sum)
                row1_cur = pl.div(row1_norm, col_sum)
                row2_cur = pl.div(row2_norm, col_sum)
                row3_cur = pl.div(row3_norm, col_sum)

            comb = pl.store(row0_cur, [0, 0, 0], comb)
            comb = pl.store(row1_cur, [1, 0, 0], comb)
            comb = pl.store(row2_cur, [2, 0, 0], comb)
            comb = pl.store(row3_cur, [3, 0, 0], comb)
            return comb

    return DeepSeekV4DecodeHcPre


def golden_deepseek_v4_decode_hc_pre(tensors):
    """Torch reference, direct port of model.py Block.hc_pre 674-682 + hc_split_sinkhorn."""
    import torch

    x        = tensors["x"].float()                        # [B, S, hc, D]
    hc_fn    = tensors["hc_fn"].float()                    # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()                 # [3]
    hc_base  = tensors["hc_base"].float()                  # [mix_hc]

    shape = x.size()
    x_flat = x.flatten(2)                                  # [B, S, hc*D]
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = (x_flat @ hc_fn.T) * rsqrt                     # [B, S, mix_hc]

    # hc_split_sinkhorn (port of kernel.py 372-427)
    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(*mixes.shape[:-1], HC_MULT, HC_MULT)

    # First step: row-softmax then col-normalize, with eps after softmax
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    # Sinkhorn iterations
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y = (pre.unsqueeze(-1) * x.view(shape)).sum(dim=2)     # [B, S, D]

    tensors["x_mixed"][:] = y.to(torch.bfloat16)
    tensors["post"][:]    = post_t
    tensors["comb"][:]    = comb_t


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x():
        return torch.rand(B, S, HC_MULT, D) - 0.5
    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM) / (HC_DIM ** 0.5)
    def init_hc_scale():
        return torch.ones(3) * 0.5
    def init_hc_base():
        return torch.zeros(MIX_HC)

    return [
        TensorSpec("x",        [B, S, HC_MULT, D],       torch.bfloat16, init_value=init_x),
        TensorSpec("hc_fn",    [MIX_HC, HC_DIM],         torch.float32,  init_value=init_hc_fn),
        TensorSpec("hc_scale", [3],                      torch.float32,  init_value=init_hc_scale),
        TensorSpec("hc_base",  [MIX_HC],                 torch.float32,  init_value=init_hc_base),
        TensorSpec("x_mixed",  [B, S, D],                torch.bfloat16, is_output=True),
        TensorSpec("post",     [B, S, HC_MULT],          torch.float32,  is_output=True),
        TensorSpec("comb",     [B, S, HC_MULT, HC_MULT], torch.float32,  is_output=True),
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
        program=build_deepseek_v4_decode_hc_pre_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_deepseek_v4_decode_hc_pre,
        config=RunConfig(
            rtol=4e-3,
            atol=4e-3,
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
