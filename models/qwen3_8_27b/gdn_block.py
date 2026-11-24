# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Qwen3.8-27B prefill Gated DeltaNet with reusable convolution and recurrent state."""
from functools import cache, partial
from math import gcd, isfinite
from pathlib import Path

import pypto.language as pl

import out_proj
from gated_rmsnorm import gdn_gated_rmsnorm_bf16
from gdn_layer import gdn_layer as gdn_layer_fn
import gdn_layer
from in_proj_ab import gdn_in_proj_ab
from qk_norm_gate import gdn_qk_norm_gate
from quant_x import gdn_quant_x
import in_proj_qkv
import in_proj_z
import short_conv
import solve_tril
from chunk_h import FSTATE_DYN
from chunk_cumsum import CHUNK_TILE
from config import GDN_QUANT, QWEN3_8_27B

# model config
C = QWEN3_8_27B.hidden_size                 # the residual stream
CQKV = QWEN3_8_27B.qkv_width                # what in_proj_qkv writes
CV = QWEN3_8_27B.value_width                # what in_proj_z writes, out_proj reads
H = QWEN3_8_27B.linear_num_value_heads
HG = QWEN3_8_27B.linear_num_key_heads
D = QWEN3_8_27B.linear_value_head_dim
KCONV = QWEN3_8_27B.linear_conv_kernel_dim  # convolution taps
# validation
DEFAULT_TEST_SEQ_LEN = 512
DEFAULT_CASES = (2048, 4096)
# The one-layer checkpoint fixture `loader_block_weights` writes: everything the block
# reads keeps the model's dimensions, and the two the text contract carries but the
# block never reads shrink to their smallest legal widths.
FIXTURE_VOCAB = 32
FIXTURE_INTERMEDIATE = 128

# At T = 8192 the block holds about 1.3 GB of intermediates at once: 0.6 GB
# inside the delta rule, and the projections, the convolution's three outputs
# and the quantised hand-offs on top. `gdn_layer` needs 1 GiB for its 0.6 GB,
# so this follows the same ratio, rounded up: the runtime takes powers of two
# only. Under the default 256 MiB per ring the layer alone fails with
# orch_error_code=2 HEAP_RING_DEADLOCK.
BLOCK_RING_HEAP = 4 * 1024 * 1024 * 1024

# tiling
HALO_TILE = 1024        # channels per block of the zero fill above the projection



def run_config(backend_type: str = "a2a3", sim: bool = False, device: int = 0, swimlane: int = 0,
               *, t: int = DEFAULT_TEST_SEQ_LEN) -> dict:
    config = dict(backend_type=backend_type, execution_mode=sim, device_id=device, enable_chip_swimlane=swimlane)
    if t > DEFAULT_TEST_SEQ_LEN:
        config["ring_heap"] = BLOCK_RING_HEAP
    return config


def build_kernel(t: int = DEFAULT_TEST_SEQ_LEN, h: int = H, hg: int = HG, d: int = D, chunk: int = CHUNK_TILE,
                 inline: bool = False):
    """The block at one shape, composing fourteen operators as inlined callees."""
    c, cqkv, cv, k = C, CQKV, CV, KCONV
    key_width = hg * d

    op_qkv = in_proj_qkv.build_kernel(t=t, inline=True, row_off=k - 1)
    z_m_group = gcd(t // in_proj_z.M_TILE, in_proj_z.M_GROUP)
    op_z = in_proj_z.build_kernel(t=t, m_group=z_m_group, inline=True)
    conv = dict(t=t, c=cqkv, k=k, inline=True)
    op_conv_q = short_conv.build_kernel(**conv, c_off=0, c_out=key_width,
                                        out_heads=hg, name="conv_q")
    op_conv_k = short_conv.build_kernel(**conv, c_off=key_width, c_out=key_width,
                                        out_heads=hg, name="conv_k")
    op_conv_v = short_conv.build_kernel(**conv, c_off=2 * key_width, c_out=h * d,
                                        out_heads=h, out_dtype=pl.FP16, name="conv_v")
    op_out = out_proj.build_kernel(t, cv, c, inline=True)
    op_state = short_conv.build_state_kernel(t=t, c=cqkv, k=k, inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def gdn_block(
        x: pl.Tensor[[t, c], pl.BF16],
        w_qkv: pl.Tensor[[cqkv, c], pl.INT8],
        w_qkv_scale: pl.Tensor[[cqkv], pl.FP32],
        w_z: pl.Tensor[[cv, c], pl.INT8],
        w_z_scale: pl.Tensor[[cv], pl.FP32],
        w_ab: pl.Tensor[[2 * h, c], pl.BF16],
        conv_w: pl.Tensor[[k, cqkv], pl.FP32],
        a_log: pl.Tensor[[h], pl.BF16],
        dt_bias: pl.Tensor[[h], pl.BF16],
        norm_w: pl.Tensor[[d], pl.BF16],
        w_out: pl.Tensor[[c, cv], pl.BF16],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        row_base: pl.Tensor[[t], pl.INT32],
        state_row: pl.Scalar[pl.INDEX],
        conv_state: pl.InOut[pl.Tensor[[short_conv.CSTATE_DYN, k - 1, cqkv], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, h, d, d], pl.FP32]],
        out: pl.Out[pl.Tensor[[t, c], pl.BF16]],
    ):
        conv_state.bind_dynamic(0, short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        x_q = pl.create_tensor([t, c], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        gdn_quant_x(x, x_q, x_scale)

        # The convolution's first window reads the previous chunk's K-1 projected
        # tokens from the same pool decode consumes. New sequences own a zero row.
        qkv_pad = pl.create_tensor([t + k - 1, cqkv], dtype=pl.BF16)
        cs = pl.reshape(conv_state, [pl.tensor.dim(conv_state, 0) * (k - 1), cqkv])
        cs_row = state_row * (k - 1)
        for blk in pl.spmd(cqkv // HALO_TILE, name_hint="conv_halo"):
            c0 = blk * HALO_TILE
            qkv_pad[0 : k - 1, c0 : c0 + HALO_TILE] = cs[
                cs_row : cs_row + k - 1, c0 : c0 + HALO_TILE]
        op_qkv(x_q, x_scale, w_qkv, w_qkv_scale, qkv_pad)
        op_state(qkv_pad, state_row, conv_state)

        z = pl.create_tensor([t, cv], dtype=pl.BF16)
        op_z(x_q, x_scale, w_z, w_z_scale, z)

        a_proj = pl.create_tensor([t, h], dtype=pl.BF16)
        b_proj = pl.create_tensor([t, h], dtype=pl.BF16)
        gdn_in_proj_ab(x, w_ab, a_proj, b_proj)

        q_conv = pl.create_tensor([t, hg, d], dtype=pl.BF16)
        k_conv = pl.create_tensor([t, hg, d], dtype=pl.BF16)
        v_conv = pl.create_tensor([t, h, d], dtype=pl.FP16)
        op_conv_q(qkv_pad, conv_w, row_base, q_conv)
        op_conv_k(qkv_pad, conv_w, row_base, k_conv)
        op_conv_v(qkv_pad, conv_w, row_base, v_conv)

        q_n = pl.create_tensor([t, hg, d], dtype=pl.FP16)
        k_n = pl.create_tensor([t, hg, d], dtype=pl.FP16)
        beta = pl.create_tensor([h, t], dtype=pl.FP32)
        g = pl.create_tensor([t, h], dtype=pl.FP32)
        gdn_qk_norm_gate(q_conv, k_conv, a_proj, b_proj, a_log, dt_bias, q_n, k_n, beta, g)

        o = pl.create_tensor([t, h, d], dtype=pl.FP16)
        gdn_layer_fn(q_n, k_n, v_conv, g, beta, tril, mask_strict, eye,
                 m_diag, m_low, row_base, state_row, final_state, o)

        y = pl.create_tensor([t, cv], dtype=pl.BF16)
        gdn_gated_rmsnorm_bf16(o, z, norm_w, y)

        op_out(y, w_out, out)
        return out, conv_state, final_state

    return gdn_block


def build_packed_kernel(t: int, rows: int, chunk: int = CHUNK_TILE, inline: bool = False):
    """Full GDN block with separate convolution and recurrent cache owners per row."""
    if t % 2048 or rows < 1 or rows > t // 256:
        raise ValueError("packed GDN block needs 2048-token total and complete 256-token rows")
    c, cqkv, cv, k = C, CQKV, CV, KCONV
    op_qkv = in_proj_qkv.build_kernel(t=t, inline=True, row_off=k - 1)
    op_z = in_proj_z.build_kernel(t=t, inline=True)
    conv = dict(t=t, c=cqkv, k=k, inline=True)
    op_conv_q = short_conv.build_packed_kernel(**conv, rows=rows, c_off=0, c_out=HG * D,
                                               out_heads=HG, name="conv_q_packed")
    op_conv_k = short_conv.build_packed_kernel(**conv, rows=rows, c_off=HG * D,
                                               c_out=HG * D, out_heads=HG,
                                               name="conv_k_packed")
    op_conv_v = short_conv.build_packed_kernel(**conv, rows=rows, c_off=2 * HG * D,
                                               c_out=H * D, out_heads=H,
                                               out_dtype=pl.FP16, name="conv_v_packed")
    op_state = short_conv.build_packed_state_swap_kernel(t=t, rows=rows, c=cqkv, k=k,
                                                          inline=True)
    op_layer = gdn_layer.build_packed_kernel(rows=rows, inline=True)
    op_out = out_proj.build_kernel(t, cv, c, inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def gdn_block_packed(
        x: pl.Tensor[[t, c], pl.BF16],
        w_qkv: pl.Tensor[[cqkv, c], pl.INT8],
        w_qkv_scale: pl.Tensor[[cqkv], pl.FP32],
        w_z: pl.Tensor[[cv, c], pl.INT8],
        w_z_scale: pl.Tensor[[cv], pl.FP32],
        w_ab: pl.Tensor[[2 * H, c], pl.BF16],
        conv_w: pl.Tensor[[k, cqkv], pl.FP32],
        a_log: pl.Tensor[[H], pl.BF16],
        dt_bias: pl.Tensor[[H], pl.BF16],
        norm_w: pl.Tensor[[D], pl.BF16],
        w_out: pl.Tensor[[c, cv], pl.BF16],
        tril: pl.Tensor[[chunk, chunk], pl.FP32],
        mask_strict: pl.Tensor[[chunk, chunk], pl.FP32],
        eye: pl.Tensor[[chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        row_ends: pl.Tensor[[rows], pl.INT32],
        state_rows: pl.Tensor[[rows], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[short_conv.CSTATE_DYN, k - 1, cqkv], pl.BF16]],
        final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, D, D], pl.FP32]],
        out: pl.Out[pl.Tensor[[t, c], pl.BF16]],
    ):
        conv_state.bind_dynamic(0, short_conv.CSTATE_DYN)
        final_state.bind_dynamic(0, FSTATE_DYN)
        x_q = pl.create_tensor([t, c], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        gdn_quant_x(x, x_q, x_scale)
        qkv_pad = pl.create_tensor([t + k - 1, cqkv], dtype=pl.BF16)
        for blk in pl.spmd(cqkv // HALO_TILE, name_hint="conv_halo_packed"):
            c0 = blk * HALO_TILE
            qkv_pad[0 : k - 1, c0 : c0 + HALO_TILE] = pl.full(
                [k - 1, HALO_TILE], dtype=pl.BF16, value=0.0)
        op_qkv(x_q, x_scale, w_qkv, w_qkv_scale, qkv_pad)
        row_halos = pl.create_tensor([rows * (k - 1), cqkv], dtype=pl.BF16)
        op_state(qkv_pad, row_ends, state_rows, conv_state, row_halos)
        z = pl.create_tensor([t, cv], dtype=pl.BF16)
        op_z(x_q, x_scale, w_z, w_z_scale, z)
        a_proj = pl.create_tensor([t, H], dtype=pl.BF16)
        b_proj = pl.create_tensor([t, H], dtype=pl.BF16)
        gdn_in_proj_ab(x, w_ab, a_proj, b_proj)
        q_conv = pl.create_tensor([t, HG, D], dtype=pl.BF16)
        k_conv = pl.create_tensor([t, HG, D], dtype=pl.BF16)
        v_conv = pl.create_tensor([t, H, D], dtype=pl.FP16)
        op_conv_q(qkv_pad, conv_w, row_base, row_owner, row_halos, q_conv)
        op_conv_k(qkv_pad, conv_w, row_base, row_owner, row_halos, k_conv)
        op_conv_v(qkv_pad, conv_w, row_base, row_owner, row_halos, v_conv)
        q_n = pl.create_tensor([t, HG, D], dtype=pl.FP16)
        k_n = pl.create_tensor([t, HG, D], dtype=pl.FP16)
        beta = pl.create_tensor([H, t], dtype=pl.FP32)
        g = pl.create_tensor([t, H], dtype=pl.FP32)
        gdn_qk_norm_gate(q_conv, k_conv, a_proj, b_proj, a_log, dt_bias, q_n, k_n, beta, g)
        o = pl.create_tensor([t, H, D], dtype=pl.FP16)
        op_layer(q_n, k_n, v_conv, g, beta, tril, mask_strict, eye,
                 m_diag, m_low, state_rows, row_ends, final_state, o)
        y = pl.create_tensor([t, cv], dtype=pl.BF16)
        gdn_gated_rmsnorm_bf16(o, z, norm_w, y)
        op_out(y, w_out, out)
        return out, conv_state, final_state

    return gdn_block_packed


def _cache_pools(resume: bool):
    """The decode pools with one owned row; new sequences start at zero."""
    import torch

    conv = torch.full((4, KCONV - 1, CQKV), -99.0, dtype=torch.bfloat16)
    state = torch.full((4, H, D, D), -1.0, dtype=torch.float32)
    if resume:
        gen = torch.Generator().manual_seed(91)
        conv[2] = (torch.randn(KCONV - 1, CQKV, generator=gen) * 0.1).to(torch.bfloat16)
        state[2] = torch.randn(H, D, D, generator=gen) * 0.02
    else:
        conv[2] = 0.0
        state[2] = 0.0
    return conv, state


def loader_block_weights() -> dict:
    """The block's weights as the published checkpoint loader hands them over.

    Writes the layer `load_weights` would have drawn as a checkpoint -- the two
    projections carrying ModelSlim's own INT8 recipe, everything else
    checkpoint-native -- then reads it back through `prepare_qwen38_weights`. What
    the kernel then receives has been through the index, the shard headers, dtype
    and shape validation, the symmetric zero-offset drop and the `[out, 1] -> [out]`
    scale flattening, none of which a host-side check exercises against a kernel.

    The fixture is one layer wide. `prepare_qwen38_weights` loads a whole text
    contract, so the stack is cut to a single Gated DeltaNet layer and the vocabulary
    and MLP to their smallest legal widths; every dimension the block reads stays the
    model's own.
    """
    import dataclasses
    import tempfile

    import torch

    import reference
    from contract import make_text_contract
    from weights import (default_checkpoint_path, modelslim_quantize_rows,
                         prepare_qwen38_weights, write_fixture_checkpoint)

    config = dataclasses.replace(QWEN3_8_27B, name="Qwen3.8-27B-one-gdn-layer",
                                 num_hidden_layers=1, vocab_size=FIXTURE_VOCAB,
                                 intermediate_size=FIXTURE_INTERMEDIATE)
    contract = make_text_contract(config)
    prefix = "model.language_model.layers.0.linear_attn."
    supplied = {}
    for name, value in reference.make_block_weights(QWEN3_8_27B).items():
        if name not in GDN_QUANT.weights:
            supplied[prefix + name] = value
            continue
        module = prefix + name.removesuffix(".weight")
        quantized, scale, offset = modelslim_quantize_rows(value)
        supplied[module + ".weight"] = quantized
        supplied[module + ".weight_scale"] = scale
        supplied[module + ".weight_offset"] = offset

    parent = default_checkpoint_path().parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gdn_block_checkpoint_", dir=parent) as directory:
        write_fixture_checkpoint(directory, contract, tensors=supplied)
        # The exporter clones: safetensors hands back views of a file this scope deletes.
        prepared = prepare_qwen38_weights(directory, lambda tensor: tensor.clone(), contract=contract)
    layer = {name.removeprefix(prefix): tensor for name, tensor in prepared.tensors.items()
             if name.startswith(prefix)}

    # The form the kernel takes, checked on what the loader returned rather than on the
    # specs: `reference.quantize_weights` flattens a scale on its way to the spec, so a
    # loader that stopped flattening its own would reach the kernel unnoticed.
    for name in GDN_QUANT.weights:
        module = name.removesuffix(".weight")
        scale = layer[f"{module}.weight_scale"]
        if scale.dtype != torch.float32 or tuple(scale.shape) != (layer[name].shape[0],):
            raise RuntimeError(f"{module}.weight_scale arrived as {tuple(scale.shape)} "
                               f"{scale.dtype}; the kernel takes one FP32 scale per output row")
        if f"{module}.weight_offset" in layer:
            raise RuntimeError(f"{module}.weight_offset reached the kernel; the symmetric "
                               "offset is validated and dropped by the loader")
    return layer


def build_tensor_specs(t: int = DEFAULT_TEST_SEQ_LEN, h: int = H, hg: int = HG, d: int = D,
                       chunk: int = CHUNK_TILE, weights: str | None = None,
                       resume: bool = False, checkpoint: bool = False):
    """Every weight the block reads, plus the delta rule's three constants."""
    import torch
    from golden import ScalarSpec, TensorSpec

    import reference
    from weights import load_weights, stacked_ab_weight

    @cache
    def block_weights():
        return loader_block_weights() if checkpoint else load_weights(QWEN3_8_27B, weights)

    def projection(module: str):
        """One INT8 projection and its flat scales, whichever source drew the layer.

        `reference.quantize_weights` quantizes a floating-point weight and passes an
        already-quantized one through, so both weight sources meet here.
        """
        name = f"{module}.weight"

        @cache
        def load():
            layer = reference.quantize_weights(block_weights(), names=(name,))
            return layer[name], layer[f"{module}.weight_scale"]

        return (lambda: load()[0]), (lambda: load()[1])

    w_qkv, w_qkv_scale = projection("in_proj_qkv")
    w_z, w_z_scale = projection("in_proj_z")

    def stacked_ab():
        return stacked_ab_weight(block_weights())

    def taps():
        return block_weights()["conv1d.weight"].reshape(CQKV, KCONV).t().contiguous().float()

    def param(name):
        return lambda: block_weights()[name].to(torch.bfloat16)

    def init_tril():
        return torch.tril(torch.ones(chunk, chunk, dtype=torch.float32))

    def init_mask_strict():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows > cols).float()

    return [
        TensorSpec("x", [t, C], torch.bfloat16,
                   init_value=lambda: reference.make_block_inputs(t, QWEN3_8_27B)),
        TensorSpec("w_qkv", [CQKV, C], torch.int8, init_value=w_qkv),
        TensorSpec("w_qkv_scale", [CQKV], torch.float32, init_value=w_qkv_scale),
        TensorSpec("w_z", [CV, C], torch.int8, init_value=w_z),
        TensorSpec("w_z_scale", [CV], torch.float32, init_value=w_z_scale),
        TensorSpec("w_ab", [2 * h, C], torch.bfloat16, init_value=stacked_ab),
        TensorSpec("conv_w", [KCONV, CQKV], torch.float32, init_value=taps),
        TensorSpec("a_log", [h], torch.bfloat16, init_value=param("A_log")),
        TensorSpec("dt_bias", [h], torch.bfloat16, init_value=param("dt_bias")),
        TensorSpec("norm_w", [d], torch.bfloat16, init_value=param("norm.weight")),
        TensorSpec("w_out", [C, CV], torch.bfloat16, init_value=param("out_proj.weight")),
        TensorSpec("tril", [chunk, chunk], torch.float32, init_value=init_tril),
        TensorSpec("mask_strict", [chunk, chunk], torch.float32, init_value=init_mask_strict),
        TensorSpec("eye", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.eye_block(chunk)),
        TensorSpec("m_diag", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[0]),
        TensorSpec("m_low", [chunk, chunk], torch.float16,
                   init_value=lambda: solve_tril.blk_masks(chunk)[1]),
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: torch.zeros(t, dtype=torch.int32)),
        ScalarSpec("state_row", torch.int64, 2, compile_runtime=True),
        TensorSpec("conv_state", [4, KCONV - 1, CQKV], torch.bfloat16,
                   init_value=lambda: _cache_pools(resume)[0]),
        TensorSpec("final_state", [4, H, D, D], torch.float32,
                   init_value=lambda: _cache_pools(resume)[1]),
        TensorSpec("out", [t, C], torch.bfloat16),
    ]


def block_weights(tensors):
    """The checkpoint-native weight dict `reference.block` reads, from the kernel's own.

    The int8 projections stay int8 with their scales beside them, which is the form
    `reference.weight` dequantises; `out_proj` is bf16 and carries no scale; `w_ab`'s
    halves split back into the two projections the spec concatenated; and `conv_w` turns
    back from the kernel's `[K, C]` to the checkpoint's `[C, 1, K]`.

    A composed caller needs this: its oracle has to read the same weights the device was
    given, not a second draw from the same source.
    """
    return {
        "in_proj_qkv.weight": tensors["w_qkv"],
        "in_proj_qkv.weight_scale": tensors["w_qkv_scale"],
        "in_proj_z.weight": tensors["w_z"],
        "in_proj_z.weight_scale": tensors["w_z_scale"],
        "out_proj.weight": tensors["w_out"],
        "in_proj_a.weight": tensors["w_ab"][:H],
        "in_proj_b.weight": tensors["w_ab"][H:],
        "conv1d.weight": tensors["conv_w"].t().contiguous().reshape(CQKV, 1, KCONV),
        "A_log": tensors["a_log"],
        "dt_bias": tensors["dt_bias"],
        "norm.weight": tensors["norm_w"],
    }


def _truth_weights(quantized, weights_path):
    """The floating-point layer behind the run's integers, and proof that it is that layer.

    Two recipes reach the kernel: `reference.quantize_rows` when the entry quantizes a
    floating-point layer itself, and ModelSlim's when the integers came out of a
    checkpoint. The truth weight is the same either way, so either match identifies the
    layer -- and the match is what rules out comparing against a weight the device was
    never given.
    """
    import torch

    import reference
    from weights import load_weights, modelslim_quantize_rows

    originals = load_weights(QWEN3_8_27B, weights_path)
    truth = quantized.copy()
    for name in GDN_QUANT.weights:
        weight = originals.get(name)
        if weight is None or not weight.is_floating_point():
            raise ValueError(f"quantization baseline requires the original floating-point weight {name}")
        scale_name = name.removesuffix(".weight") + ".weight_scale"
        kernel_q, kernel_scale = reference.quantize_rows(weight)
        slim_q, slim_scale, _ = modelslim_quantize_rows(weight)
        matched = (
            (torch.equal(kernel_q, quantized[name]) and torch.equal(kernel_scale, quantized[scale_name]))
            or (torch.equal(slim_q, quantized[name])
                and torch.equal(slim_scale.reshape(-1), quantized[scale_name]))
        )
        if not matched:
            raise ValueError(f"{name} differs from the run's weights and scales; use the original --weights")
        truth[name] = weight
    return truth


def _block_reference(inputs, weights, conv, state, *, quant: bool):
    import reference

    reference.host_torch_setup()
    chunk = inputs["tril"].shape[0]
    return reference.block(inputs["x"], weights, QWEN3_8_27B, chunk, quant_x=quant, quant_y=False,
                           conv_state=conv, state=state.transpose(-2, -1))


def golden_gdn_block(tensors):
    """Fill the golden output and cache rows from the supplied inputs."""
    row = int(tensors["state_row"])
    state = _block_reference(tensors, block_weights(tensors), tensors["conv_state"][row],
                             tensors["final_state"][row], quant=True)
    tensors["out"].copy_(state["out"])
    tensors["conv_state"][row].copy_(state["conv_state_out"])
    tensors["final_state"][row].copy_(state["final_state"].transpose(-2, -1))


def build_packed_tensor_specs(t: int = 2048, starts: tuple[int, ...] = (0, 256),
                              state_rows: tuple[int, ...] = (1, 3),
                              pool_rows: int = 5, weights: str | None = None,
                              resume: bool = False):
    import torch
    from golden import TensorSpec

    if (len(starts) != len(state_rows) or starts[0] != 0 or t % 2048
            or any(start % 256 for start in starts)
            or any(b - a < 256 for a, b in zip(starts, starts[1:] + (t,)))
            or len(set(state_rows)) != len(state_rows)
            or any(row < 0 or row >= pool_rows for row in state_rows)):
        raise ValueError("packed GDN block needs complete rows and distinct cache owners")
    ends = starts[1:] + (t,)

    def pools():
        conv = torch.full((pool_rows, KCONV - 1, CQKV), -99.0, dtype=torch.bfloat16)
        final = torch.full((pool_rows, H, D, D), -1.0, dtype=torch.float32)
        for owner, row in enumerate(state_rows):
            gen = torch.Generator().manual_seed(91 + owner)
            if resume:
                conv[row] = (torch.randn(KCONV - 1, CQKV, generator=gen) * 0.1).to(torch.bfloat16)
                final[row] = torch.randn(H, D, D, generator=gen) * 0.02
            else:
                conv[row] = 0.0
                final[row] = 0.0
        return conv, final

    conv, final = pools()
    base = build_tensor_specs(t=t, weights=weights, resume=resume)
    row_base = torch.empty(t, dtype=torch.int32)
    row_owner = torch.empty(t, dtype=torch.int32)
    for owner, (start, end) in enumerate(zip(starts, ends)):
        row_base[start:end] = start
        row_owner[start:end] = owner
    return base[:16] + [
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: row_base),
        TensorSpec("row_owner", [t], torch.int32, init_value=lambda: row_owner),
        TensorSpec("row_ends", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(ends, dtype=torch.int32)),
        TensorSpec("state_rows", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(state_rows, dtype=torch.int32)),
        TensorSpec("conv_state", [pool_rows, KCONV - 1, CQKV], torch.bfloat16,
                   init_value=lambda: conv),
        TensorSpec("final_state", [pool_rows, H, D, D], torch.float32,
                   init_value=lambda: final),
    ] + base[20:]


_PACKED_TRUTH: dict[str, object] = {}


def golden_gdn_block_packed(tensors, *, weights_path=None):
    import torch

    import reference

    weights = block_weights(tensors)
    truth_weights = _truth_weights(weights, weights_path)
    ends = tensors["row_ends"].tolist()
    starts = [0] + ends[:-1]
    truth_out = torch.empty(tensors["out"].shape, dtype=torch.float64)
    truth_conv = {}
    truth_final = {}
    for owner, (start, end) in enumerate(zip(starts, ends)):
        row = int(tensors["state_rows"][owner])
        conv_in = tensors["conv_state"][row].clone()
        final_in = tensors["final_state"][row].clone()
        truth = reference.block(
            tensors["x"][start:end], truth_weights, QWEN3_8_27B, CHUNK_TILE,
            quant_x=False, quant_y=False, conv_state=conv_in,
            state=final_in.transpose(-2, -1))
        truth_out[start:end].copy_(truth["out"])
        truth_conv[row] = truth["conv_state_out"]
        truth_final[row] = truth["final_state"].transpose(-2, -1)
        st = reference.block(
            tensors["x"][start:end], weights, QWEN3_8_27B, CHUNK_TILE,
            quant_x=True, quant_y=False,
            conv_state=tensors["conv_state"][row],
            state=tensors["final_state"][row].transpose(-2, -1))
        tensors["out"][start:end].copy_(st["out"].to(torch.bfloat16))
        tensors["conv_state"][row].copy_(st["conv_state_out"])
        tensors["final_state"][row].copy_(st["final_state"].transpose(-2, -1))
    _PACKED_TRUTH.clear()
    _PACKED_TRUTH.update(out=truth_out, conv_state=truth_conv, final_state=truth_final)


def compare_packed_out(actual, expected, inputs=None, **_kwargs):
    if inputs is None or "out" not in _PACKED_TRUTH:
        return False, "packed GDN block needs its golden truth for both owners"
    ends = inputs["row_ends"].tolist()
    starts = [0] + ends[:-1]
    for owner, (start, end) in enumerate(zip(starts, ends)):
        truth = _PACKED_TRUTH["out"][start:end]
        fq = expected[start:end].double()
        dev = actual[start:end].double()
        floor = float((fq - truth).norm() / truth.norm())
        over = float((dev - truth).norm() / truth.norm()) / floor
        near = float((dev - fq).norm() / truth.norm()) / floor
        row_floor = float(((fq - truth).norm(dim=-1) / truth.norm(dim=-1)).max())
        row_over = float(((dev - truth).norm(dim=-1) / truth.norm(dim=-1)).max()) / row_floor
        row_near = float(((dev - fq).norm(dim=-1) / truth.norm(dim=-1)).max()) / row_floor
        if not (over <= MAX_OVER_FLOOR and row_over <= MAX_ROW_OVER_FLOOR
                and near <= MAX_VS_GOLDEN and row_near <= MAX_ROW_VS_GOLDEN):
            return False, (f"owner {owner}: truth {over:.6f}x floor, worst row "
                           f"{row_over:.3f}x; fake-quant {near:.3f}x, "
                           f"worst row {row_near:.3f}x")
    return True, "both packed GDN owners pass the quantization-floor gate"


def compare_packed_cache(name: str, state_rows: tuple[int, ...]):
    fill = -99.0 if name == "conv_state" else -1.0

    def compare(actual, expected, **_kwargs):
        import torch

        if name not in _PACKED_TRUTH:
            return False, f"{name}: missing packed golden truth"
        for row in state_rows:
            dev = actual[row].double()
            fq = expected[row].double()
            truth = _PACKED_TRUTH[name][row].double()
            den = torch.linalg.vector_norm(truth)
            floor = float(torch.linalg.vector_norm(fq - truth) / den)
            over = float(torch.linalg.vector_norm(dev - truth) / den) / floor
            near = float(torch.linalg.vector_norm(dev - fq) / den) / floor
            peak = torch.linalg.vector_norm(truth.reshape(truth.shape[0], -1), dim=1).max()
            row_floor = float(torch.linalg.vector_norm(
                (fq - truth).reshape(truth.shape[0], -1), dim=1).max() / peak)
            row_over = float(torch.linalg.vector_norm(
                (dev - truth).reshape(truth.shape[0], -1), dim=1).max() / peak) / row_floor
            row_near = float(torch.linalg.vector_norm(
                (dev - fq).reshape(truth.shape[0], -1), dim=1).max() / peak) / row_floor
            if not (over <= 1.15 and row_over <= 1.25 and near <= 0.50 and row_near <= 0.75):
                return False, (f"{name} row {row}: truth {over:.3f}x floor, worst row "
                               f"{row_over:.3f}x; fake-quant {near:.3f}x, "
                               f"worst row {row_near:.3f}x")
        others = [row for row in range(actual.shape[0]) if row not in state_rows]
        if others and not bool((actual[others] == fill).all()):
            return False, f"{name}: unowned rows changed"
        return True, f"{name}: both owners pass; unowned rows untouched"

    return compare


MAX_OVER_FLOOR = 1.05
MAX_ROW_OVER_FLOOR = 1.25
MAX_VS_GOLDEN = 1.00
MAX_ROW_VS_GOLDEN = 1.50


def block_comparators(weights_path: str | None, resume: bool, golden_data: str | None):
    """Compare the saved golden and FP64 truth for the run's inputs and caches."""
    import torch

    @cache
    def initial_pools():
        if golden_data is None:
            return _cache_pools(resume)
        snapshot = Path(golden_data) / "in"
        conv = torch.load(snapshot / "conv_state.pt", weights_only=True)
        state = torch.load(snapshot / "final_state.pt", weights_only=True)
        return conv, state

    chains = None

    def references(inputs):
        nonlocal chains
        if chains is None:
            row = int(inputs["state_row"])
            conv, state = initial_pools()
            quantized = block_weights(inputs)
            truth_weights = _truth_weights(quantized, weights_path)
            fq = _block_reference(inputs, quantized, conv[row], state[row], quant=True)
            truth = _block_reference(inputs, truth_weights, conv[row], state[row], quant=False)
            chains = fq, truth
        return chains

    def compare_out(actual, expected, *, inputs, **_kwargs):
        if actual.shape != expected.shape:
            return False, f"output shape {tuple(actual.shape)} != golden shape {tuple(expected.shape)}"
        try:
            fq, truth = references(inputs)
        except ValueError as err:
            return False, str(err)
        truth = truth["out"]
        dev = actual.double()

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())

        floor, row_floor = rel(fq["out"], truth), worst_row(fq["out"], truth)
        if not (isfinite(floor) and floor > 0 and isfinite(row_floor) and row_floor > 0):
            return False, "quantization baseline requires finite, positive tensor and worst-row errors"
        over = rel(dev, truth) / floor
        row_over = worst_row(dev, truth) / row_floor
        near = rel(dev, expected.double()) / floor
        row_near = worst_row(dev, expected.double()) / row_floor
        ok = (over <= MAX_OVER_FLOOR and row_over <= MAX_ROW_OVER_FLOOR
              and near <= MAX_VS_GOLDEN and row_near <= MAX_ROW_VS_GOLDEN)
        detail = (f"out: truth {over:.3f}x floor, worst row {row_over:.3f}x; "
                  f"saved golden {near:.3f}x, worst row {row_near:.3f}x")
        return ok, "" if ok else detail

    def compare_cache(name, pool):
        def compare(actual, expected, *, inputs, **_kwargs):
            row = int(inputs["state_row"])
            others = [r for r in range(actual.shape[0]) if r != row]
            if others and not torch.equal(actual[others], initial_pools()[pool][others]):
                return False, f"{name}: unowned rows changed"
            try:
                fq, truth = references(inputs)
            except ValueError as err:
                return False, str(err)
            key = "conv_state_out" if name == "conv_state" else "final_state"
            fq, truth = fq[key].double(), truth[key].double()
            if name == "final_state":
                fq, truth = fq.transpose(-2, -1), truth.transpose(-2, -1)
            dev = actual[row].double()
            err = dev - expected[row].double()
            floor = fq - truth

            def worst_row(x):
                return float(x.flatten(1).norm(dim=1).max())

            floor_norm, row_floor = float(floor.norm()), worst_row(floor)
            if not (isfinite(floor_norm) and floor_norm > 0 and isfinite(row_floor) and row_floor > 0):
                return False, f"{name}: quantization floor must be finite and positive"
            over = float((dev - truth).norm()) / floor_norm
            row_over = worst_row(dev - truth) / row_floor
            near = float(err.norm()) / floor_norm
            row_near = worst_row(err) / row_floor
            ok = over <= 1.15 and row_over <= 1.25 and near <= 0.50 and row_near <= 0.75
            detail = (f"{name}: truth {over:.3f}x floor, worst row {row_over:.3f}x; "
                      f"saved golden {near:.3f}x, worst row {row_near:.3f}x")
            return ok, "" if ok else detail

        return compare

    return {"out": compare_out, "conv_state": compare_cache("conv_state", 0),
            "final_state": compare_cache("final_state", 1)}


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="override the standalone or packed token count")
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
    parser.add_argument("--checkpoint-weights", action="store_true", default=False,
                        help="only the case whose INT8 projections come back through the "
                             "checkpoint loader, which a default run also includes")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--packed", action="store_true", default=False)
    parser.add_argument("--row-starts", type=str, default="0,256")
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the block over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=range(5))
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    seq_tile = 2048 if args.packed else DEFAULT_TEST_SEQ_LEN
    cases = [args.seq_len] if args.seq_len is not None else (DEFAULT_CASES if args.packed else (DEFAULT_TEST_SEQ_LEN,))
    if any(t < seq_tile or t % seq_tile for t in cases):
        parser.error(f"--seq-len must be a positive multiple of {seq_tile}")
    if args.packed and (args.golden_data or args.runtime_dir or args.save_data):
        parser.error("packed block validation does not support save or replay options")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    if args.checkpoint_weights and (args.packed or args.weights):
        parser.error("--checkpoint-weights draws its own layer and runs the standalone block; "
                     "it takes neither --packed nor --weights")
    plan = [(t, args.packed, args.resume, args.checkpoint_weights) for t in cases]
    if args.seq_len is None and not (args.packed or args.resume or args.rounds
                                     or args.checkpoint_weights or args.save_data
                                     or args.golden_data or args.runtime_dir):
        plan.append((DEFAULT_CASES[0], True, True, False))
        if args.weights is None:
            # The same block over weights the checkpoint loader produced rather than
            # ones this entry quantized. A named real layer runs its own case instead.
            plan.append((DEFAULT_TEST_SEQ_LEN, False, False, True))

    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for t, packed, resume, checkpoint in plan:
        if packed:
            starts = tuple(int(value) for value in args.row_starts.split(","))
            state_rows = tuple(2 * owner + 1 for owner in range(len(starts)))
            print(f"[case] gdn_block packed T={t} starts={starts} resume={resume}", flush=True)
            golden_fn = None if args.compile_only else partial(golden_gdn_block_packed, weights_path=args.weights)
            result = run(
                fn=build_packed_kernel(t=t, rows=len(starts)),
                specs=build_packed_tensor_specs(t=t, starts=starts, state_rows=state_rows,
                                                pool_rows=2 * len(starts) + 1,
                                                weights=args.weights, resume=resume),
                golden_fn=golden_fn,
                config=run_config(args.backend_type, args.execution_mode, args.device,
                                  args.enable_chip_swimlane, t=t),
                compare_fn={"out": compare_packed_out,
                            "conv_state": compare_packed_cache("conv_state", state_rows),
                            "final_state": compare_packed_cache("final_state", state_rows)},
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(f"FAILED at packed GDN block T={t}")
            continue
        source = " checkpoint-weights" if checkpoint else ""
        print(f"[case] gdn_block T={t}{source} resume={resume}", flush=True)
        result = run(
            fn=build_kernel(t=t),
            specs=build_tensor_specs(t=t, weights=args.weights, resume=resume, checkpoint=checkpoint),
            golden_fn=golden_gdn_block,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=run_config(args.backend_type, args.execution_mode, args.device, args.enable_chip_swimlane, t=t),
            rtol=1e-2,
            atol=1e-5,
            compare_fn=block_comparators(args.weights, resume, args.golden_data),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}{source}")
