# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Qwen3.8-27B's decoder layer at a decode batch, in both of the model's two forms.

    residual, norm_in  = add_norm(residual, delta, input_layernorm)
    mixer_out          = mixer(norm_in)                 Gated DeltaNet, or gated attention
    residual, norm_mlp = add_norm(residual, mixer_out, post_attention_layernorm)
    mlp_out            = mlp(norm_mlp)                  gate and up, SwiGLU, down

**The layer boundary sits one addition behind the module boundary.** `delta` is the
previous layer's MLP output and `residual` the stream it belongs to, so a layer starts by
completing that addition and ends with its own MLP output still unadded.
`rmsnorm.build_residual_kernel` computes the addition and the norm that reads it as one
stage, and a decoder layer's two additions each pair with a norm that way.

The batch must be a positive multiple of 16 and at most `decode_projection.max_batch()`;
`check_batch` enforces both ends.

`norm_col_groups` and `mlp_fanned` select each stage's fanned or unfanned form. Both
default to what the layer ships; see the optimization page for what they cost.
"""
import pypto.language as pl

import a8w8_linear
import act_quant
import decode_attention
import decode_delta_rule
import decode_gated_attention
import decode_gdn_block
import decode_kv_append
import decode_short_conv
import gate_up_swiglu
import quant_x
import rmsnorm
from config import QWEN3_8_27B
from decode_projection import CUBE_WORKERS, M_FRACTAL, check_batch, padded_batch

# model config
C = QWEN3_8_27B.hidden_size                 # the residual stream
INTER = QWEN3_8_27B.intermediate_size       # the MLP's inner width
LAYERS = QWEN3_8_27B.num_hidden_layers
# the GDN mixer's widths
CQKV = decode_gdn_block.CQKV
CV = decode_gdn_block.CV
H = decode_gdn_block.H
D = decode_gdn_block.D
KCONV = decode_gdn_block.KCONV
STATE_LEN = decode_gdn_block.STATE_LEN
# the attention mixer's widths
QKV_WIDTH = decode_gated_attention.QKV_WIDTH
ATT_WIDTH = decode_gated_attention.ATT_WIDTH
KV_WIDTH = decode_gated_attention.KV_WIDTH
HEAD_DIM = decode_gated_attention.D
ROT = decode_gated_attention.ROT
PAGE = decode_kv_append.PAGE

# validation
# one padded decode batch, then two, for whichever mixer --kind selects.
DEFAULT_CASES = (16, 32)

# tiling: each stage's own decode value
NORM_COL_GROUPS = rmsnorm.COL_GROUPS    # column groups of each add-and-norm
NORM_TOK_TILE = rmsnorm.DECODE_TOK_TILE     # the unfanned form's tiles, for the A/B above
NORM_K_TILE = rmsnorm.K_TILE

# The decode tiles of the MLP's two matmuls, as (n_tile, k_tile, n_group), measured at
# decode M over 109 configurations; `m_tile` is M and `m_group` is 1. Both land on the same
# triple and both hold from M = 16 to M = 32, but for different reasons, which is why the
# sweep had to cover both: `gate_up_swiglu` wants the largest operand tile the 512 KiB Mat
# buffer allows and takes 68 patches with it, while `down_proj` wants **20 patches against
# 24 cube cores** and falls off a cliff below that -- 93.3 us at 20 blocks, 136.4 at 10,
# 243.3 at 5. `CUBE_WORKERS` blocks walk those patches; the two counts are not the same.
MLP_TILES = {
    "gate_up_swiglu": (128, 512, 2),        # 68 patches, 164.0 us, 1087 GB/s at M = 16
    "down_proj": (128, 512, 2),             # 20 patches,  93.3 us,  955 GB/s at M = 16
}

# Each quantiser's own decode column fan-out.
X_COL_GROUPS = quant_x.COL_GROUPS
ACT_COL_GROUPS = act_quant.COL_GROUPS

# A random norm weight is drawn at the checkpoint's own scale, the module being `1 + w`:
# layer 0's two have standard deviations of 0.042 and 0.037.
NORM_W_SCALE = 0.05


def run_config(backend_type: str = "a2a3", sim: bool = False, device: int = 0,
               swimlane: int = 0) -> dict:
    """The config `golden.run` needs for either layer."""
    return dict(backend_type=backend_type, execution_mode=sim, device_id=device,
                enable_chip_swimlane=swimlane)


def _build_mlp(batch: int, fanned: bool = True):
    """The four decode MLP stages as one program: quantise, gate and up with SwiGLU,
    quantise again, then down.

    | stage | shape |
    |---|---|
    | `quant_x` | `[M, 5120]` |
    | `gate_up_swiglu` | K=5120 N=17408, two INT8 weights |
    | `act_quant` | `[M, 17408]` |
    | `down_proj` | K=17408 N=5120 |

    `m_tile` is M and `m_group` is 1: below 16 rows the dequant scale column is shorter than
    the matmul's 16-row fractal, and with one m-tile there is no patch to walk. The batch
    must be a positive multiple of 16, which `decode_projection.check_batch` states and
    `decode_projection.padded_batch` produces.

    *fanned* picks each per-token quantiser's decode form, whose reduction is split across
    the column axis. The two forms are bit-identical -- a maximum is associative and the
    rescale is unchanged -- so this is a speed knob and not a numerics one.

    **Two constraints bind the tiles.** The 512 KiB Mat buffer caps `n_tile * k_tile`, and it
    binds sooner on `gate_up_swiglu` because that kernel holds two weight tiles rather than
    one: `n_tile=128 k_tile=1024` and `n_tile=256 k_tile=512` are both refused. And
    `a8w8_linear` requires an even `n_group`, its inner loop pairing n-tiles, so `down_proj`
    cannot run one n-tile per block.
    """
    check_batch(batch)
    up_n_tile, up_k_tile, up_n_group = MLP_TILES["gate_up_swiglu"]
    down_n_tile, down_k_tile, down_n_group = MLP_TILES["down_proj"]

    if fanned:
        quant = quant_x.build_fanned_kernel(batch, col_groups=X_COL_GROUPS, inline=True)
        act_q_op = act_quant.build_fanned_kernel(batch, col_groups=ACT_COL_GROUPS, inline=True)
    else:
        quant = quant_x.build_kernel(tok_tile=M_FRACTAL, group=1, inline=True)
        act_q_op = act_quant.qwen3_act_quant
    op_gate_up = gate_up_swiglu.build_kernel(m=batch, k=C, n=INTER, inline=True,
                                             m_tile=batch, n_tile=up_n_tile, k_tile=up_k_tile,
                                             m_group=1, n_group=up_n_group,
                                             workers=CUBE_WORKERS)
    # Two GM transport slots for the 16-row down projection.
    op_down = a8w8_linear.build_kernel(
        m=batch, k=INTER, n=C, name="down_proj", inline=True,
        m_tile=batch, n_tile=down_n_tile, k_tile=down_k_tile,
        m_group=1, n_group=down_n_group, ring_slots=2 if batch == 16 else 1,
        workers=CUBE_WORKERS,
    )

    @pl.jit.inline
    def qwen3_decode_mlp(
        x: pl.Tensor[[batch, C], pl.BF16],
        w_gate: pl.Tensor[[INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[INTER], pl.FP32],
        w_up: pl.Tensor[[INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[INTER], pl.FP32],
        w_down: pl.Tensor[[C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[C], pl.FP32],
        out: pl.Out[pl.Tensor[[batch, C], pl.BF16]],
    ):
        x_q = pl.create_tensor([batch, C], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, batch], dtype=pl.FP32)
        quant(x, x_q, x_scale)

        act = pl.create_tensor([batch, INTER], dtype=pl.BF16)
        op_gate_up(x_q, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, act)

        act_q = pl.create_tensor([batch, INTER], dtype=pl.INT8)
        act_scale = pl.create_tensor([1, batch], dtype=pl.FP32)
        act_q_op(act, act_q, act_scale)

        op_down(act_q, act_scale, w_down, w_down_scale, out)
        return out

    return qwen3_decode_mlp


def _parts(batch: int, norm_col_groups: int | None = NORM_COL_GROUPS,
           mlp_fanned: bool = True):
    """The padded batch, the two add-and-norm stages and the MLP, which both layers share.

    The two norms are separate instances of one builder rather than one callee called
    twice: they are the same program, but an inlined callee is substituted at its call
    site, and two instances keep the two substitutions independent.

    `norm_col_groups=None` builds each add-and-norm's unfanned form and `mlp_fanned=False`
    the MLP quantisers' unfanned form. Both default to what the layer ships.
    """
    m = padded_batch(batch)
    check_batch(m)
    if norm_col_groups is None:
        def norm():
            return rmsnorm.build_residual_kernel(tok_tile=NORM_TOK_TILE, k_tile=NORM_K_TILE,
                                                 inline=True)
    else:
        def norm():
            return rmsnorm.build_fanned_residual_kernel(m, col_groups=norm_col_groups,
                                                        inline=True)
    mlp = _build_mlp(m, fanned=mlp_fanned)
    return m, norm(), norm(), mlp


def build_gdn_kernel(batch: int, head_tile: int = decode_gdn_block.HEAD_TILE,
                     norm_col_groups: int | None = NORM_COL_GROUPS, mlp_fanned: bool = True,
                     inline: bool = False, active_rows: int = 0):
    """A linear-attention layer: the Gated DeltaNet mixer between the two norms."""
    m, norm_in, norm_post, mlp = _parts(batch, norm_col_groups, mlp_fanned)
    mixer = decode_gdn_block.build_kernel(m, head_tile=head_tile, inline=True, active_rows=active_rows)

    @(pl.jit.inline if inline else pl.jit)
    def decode_gdn_layer(
        residual: pl.Tensor[[m, C], pl.FP32],
        delta: pl.Tensor[[m, C], pl.BF16],
        norm_in_w: pl.Tensor[[C], pl.BF16],
        norm_post_w: pl.Tensor[[C], pl.BF16],
        w_qkv: pl.Tensor[[CQKV, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[CQKV], pl.FP32],
        w_z: pl.Tensor[[CV, C], pl.INT8],
        w_z_scale: pl.Tensor[[CV], pl.FP32],
        w_ab: pl.Tensor[[2 * H, C], pl.BF16],
        conv_w: pl.Tensor[[KCONV, CQKV], pl.FP32],
        a_log: pl.Tensor[[H], pl.BF16],
        dt_bias: pl.Tensor[[H], pl.BF16],
        norm_w: pl.Tensor[[D], pl.BF16],
        w_out: pl.Tensor[[C, CV], pl.BF16],
        state_rows: pl.Tensor[[m], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[decode_short_conv.CSTATE_DYN, STATE_LEN, CQKV], pl.BF16]],
        state: pl.InOut[pl.Tensor[[decode_delta_rule.DSTATE_DYN, H, D, D], pl.FP32]],
        w_gate: pl.Tensor[[INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[INTER], pl.FP32],
        w_up: pl.Tensor[[INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[INTER], pl.FP32],
        w_down: pl.Tensor[[C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[C], pl.FP32],
        residual_out: pl.Out[pl.Tensor[[m, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[m, C], pl.BF16]],
    ):
        """`delta` is the previous layer's MLP output; `residual_out` is this layer's stream."""
        resid_mixer = pl.create_tensor([m, C], dtype=pl.FP32)
        mixer_in = pl.create_tensor([m, C], dtype=pl.BF16)
        norm_in(residual, delta, norm_in_w, resid_mixer, mixer_in)

        mixer_out = pl.create_tensor([m, C], dtype=pl.BF16)
        mixer(mixer_in, w_qkv, w_qkv_scale, w_z, w_z_scale, w_ab, conv_w, a_log, dt_bias,
              norm_w, w_out, state_rows, conv_state, state, mixer_out)

        mlp_in = pl.create_tensor([m, C], dtype=pl.BF16)
        norm_post(resid_mixer, mixer_out, norm_post_w, residual_out, mlp_in)

        mlp(mlp_in, w_gate, w_gate_scale, w_up, w_up_scale, w_down, w_down_scale, mlp_out)
        return residual_out, mlp_out

    return decode_gdn_layer


def build_attn_kernel(batch: int, max_pages: int, splits: int | None = None,
                      norm_col_groups: int | None = NORM_COL_GROUPS, mlp_fanned: bool = True,
                      inline: bool = False):
    """A full-attention layer: the gated-attention mixer between the two norms."""
    m, norm_in, norm_post, mlp = _parts(batch, norm_col_groups, mlp_fanned)
    mixer = decode_gated_attention.build_kernel(m, max_pages=max_pages, splits=splits,
                                                inline=True)

    @(pl.jit.inline if inline else pl.jit)
    def decode_attn_layer(
        residual: pl.Tensor[[m, C], pl.FP32],
        delta: pl.Tensor[[m, C], pl.BF16],
        norm_in_w: pl.Tensor[[C], pl.BF16],
        norm_post_w: pl.Tensor[[C], pl.BF16],
        w_qkv: pl.Tensor[[QKV_WIDTH, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[QKV_WIDTH], pl.FP32],
        q_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        k_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[m, ROT], pl.FP32],
        rope_sin: pl.Tensor[[m, ROT], pl.FP32],
        block_table: pl.Tensor[[m, max_pages], pl.INT32],
        seq_lens: pl.Tensor[[m], pl.INT32],
        tail_mask: pl.Tensor[[PAGE + 1, PAGE], pl.FP32],
        w_out: pl.Tensor[[C, ATT_WIDTH], pl.INT8],
        w_out_scale: pl.Tensor[[C], pl.FP32],
        k_pages: pl.InOut[pl.Tensor[[decode_kv_append.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        v_pages: pl.InOut[pl.Tensor[[decode_kv_append.KVROWS_DYN, KV_WIDTH], pl.BF16]],
        w_gate: pl.Tensor[[INTER, C], pl.INT8],
        w_gate_scale: pl.Tensor[[INTER], pl.FP32],
        w_up: pl.Tensor[[INTER, C], pl.INT8],
        w_up_scale: pl.Tensor[[INTER], pl.FP32],
        w_down: pl.Tensor[[C, INTER], pl.INT8],
        w_down_scale: pl.Tensor[[C], pl.FP32],
        residual_out: pl.Out[pl.Tensor[[m, C], pl.FP32]],
        mlp_out: pl.Out[pl.Tensor[[m, C], pl.BF16]],
    ):
        """`seq_lens[b]` counts the token being decoded; the cache append writes before the read."""
        resid_mixer = pl.create_tensor([m, C], dtype=pl.FP32)
        mixer_in = pl.create_tensor([m, C], dtype=pl.BF16)
        norm_in(residual, delta, norm_in_w, resid_mixer, mixer_in)

        mixer_out = pl.create_tensor([m, C], dtype=pl.BF16)
        mixer(mixer_in, w_qkv, w_qkv_scale, q_norm_w, k_norm_w, rope_cos, rope_sin,
              block_table, seq_lens, tail_mask, w_out, w_out_scale, k_pages, v_pages,
              mixer_out)

        mlp_in = pl.create_tensor([m, C], dtype=pl.BF16)
        norm_post(resid_mixer, mixer_out, norm_post_w, residual_out, mlp_in)

        mlp(mlp_in, w_gate, w_gate_scale, w_up, w_up_scale, w_down, w_down_scale, mlp_out)
        return residual_out, mlp_out

    return decode_attn_layer


KINDS = ("gdn", "attn")


def build_kernel(kind: str, batch: int, **kwargs):
    """Either layer by name, `gdn` or `attn`."""
    if kind == "gdn":
        return build_gdn_kernel(batch, **kwargs)
    if kind == "attn":
        return build_attn_kernel(batch, **kwargs)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


__all__ = ["KINDS", "build_attn_kernel", "build_gdn_kernel",
           "build_kernel", "build_tensor_specs", "compare_fns", "gdn_weights",
           "golden_decode_layer", "mixer_oracle", "run_config", "truth"]


def build_tensor_specs(kind: str, batch: int = 16, weights: str | None = None,
                       mlp_weights: str | None = None, norm_weights: str | None = None,
                       seed: int = 7, **mixer_kwargs):
    """The mixer's own fixtures and the MLP's, plus the four tensors the layer adds.

    Both mixer spec sets and the MLP's are taken unchanged apart from dropping their `x`
    and `out`, so a layer run reads exactly the tensors its stages were gated on, and the
    oracle reads them back out of the harness rather than redrawing them.

    `residual` and `delta` are drawn at the same scale. A real residual stream grows with
    depth and is several times its per-layer delta, which would put less of the mixer's
    error into `residual_out`; equal scales are the harder case for the gate.

    The two norm weights are drawn at the checkpoint's own scale rather than unit normal.
    `1 + w` multiplies the mixer's and the MLP's input per channel, and a unit-normal draw
    would spread that factor over orders of magnitude -- including through zero -- which
    moves the per-token quantisation floor those stages are measured against.
    """
    import torch
    from golden import TensorSpec

    import mlp

    m = padded_batch(batch)
    if kind == "gdn":
        mixer_specs = decode_gdn_block.build_tensor_specs(batch=batch, weights=weights,
                                                          **mixer_kwargs)
    elif kind == "attn":
        mixer_specs = decode_gated_attention.build_tensor_specs(batch=batch, weights=weights,
                                                                **mixer_kwargs)
    else:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    mlp_specs = mlp.build_tensor_specs(t=m, weights=mlp_weights)

    def inner(specs):
        return [spec for spec in specs if spec.name not in ("x", "out")]

    def draw(offset, dtype):
        def make():
            gen = torch.Generator().manual_seed(seed + offset)
            return torch.randn(m, C, generator=gen).to(dtype)
        return make

    def norm_param(name, offset):
        def make():
            if norm_weights is not None:
                return torch.load(norm_weights, weights_only=True)[name].to(torch.bfloat16)
            gen = torch.Generator().manual_seed(seed + offset)
            return (torch.randn(C, generator=gen) * NORM_W_SCALE).to(torch.bfloat16)
        return make

    return ([
        TensorSpec("residual", [m, C], torch.float32, init_value=draw(0, torch.float32)),
        TensorSpec("delta", [m, C], torch.bfloat16, init_value=draw(1, torch.bfloat16)),
        TensorSpec("norm_in_w", [C], torch.bfloat16,
                   init_value=norm_param("input_layernorm.weight", 2)),
        TensorSpec("norm_post_w", [C], torch.bfloat16,
                   init_value=norm_param("post_attention_layernorm.weight", 3)),
    ] + inner(mixer_specs) + inner(mlp_specs) + [
        TensorSpec("residual_out", [m, C], torch.float32),
        TensorSpec("mlp_out", [m, C], torch.bfloat16),
    ])



def gdn_weights(tensors):
    """The checkpoint-native weight dict `reference.recurrent_block` reads, from the kernel's.

    The int8 projections stay int8 with their scales beside them, which is the form
    `reference.weight` dequantises; `w_ab`'s halves split back into the two projections the
    spec concatenated; and `conv_w` turns back from the kernel's `[K, C]` to the
    checkpoint's `[C, 1, K]`.
    """
    return {
        "in_proj_qkv.weight": tensors["w_qkv"],
        "in_proj_qkv.weight_scale": tensors["w_qkv_scale"],
        "in_proj_z.weight": tensors["w_z"],
        "in_proj_z.weight_scale": tensors["w_z_scale"],
        # bf16 and no scale: `reference.weight` dequantises only int8
        "out_proj.weight": tensors["w_out"],
        "in_proj_a.weight": tensors["w_ab"][:H],
        "in_proj_b.weight": tensors["w_ab"][H:],
        "conv1d.weight": tensors["conv_w"].t().contiguous().reshape(CQKV, 1, KCONV),
        "A_log": tensors["a_log"],
        "dt_bias": tensors["dt_bias"],
        "norm.weight": tensors["norm_w"],
    }


def _gdn_mixer(tensors, quant: bool):
    """`reference.recurrent_block` once per sequence, each on its own row of both pools.

    Returns the mixer and the two updated pools. Every row read is converted out of the
    harness tensor, so the two chains cannot disturb each other, and the rows `state_rows`
    does not name are carried through untouched.
    """
    import torch

    import reference

    w = gdn_weights(tensors)
    rows = tensors["state_rows"].tolist()
    conv_in = tensors["conv_state"].double()
    state_in = tensors["state"].double()
    pools = {"conv_state": conv_in.clone(), "state": state_in.clone()}

    def mixer(x):
        out = torch.empty(x.shape[0], C, dtype=torch.float64)
        for i, row in enumerate(rows):
            step = reference.recurrent_block(
                x[i : i + 1], w, QWEN3_8_27B,
                conv_state=conv_in[row],
                state=decode_delta_rule.golden_state_to_kv(state_in[row]),
                # `quant_y` stays off at both ends: the block's last hand-off is bf16 both
                # sides now, so there is no per-token quantisation of `y` to model
                quant_x=quant, quant_y=False)
            out[i] = step["out"][0]
            pools["conv_state"][row] = step["conv_state_out"]
            pools["state"][row] = decode_delta_rule.golden_state_from_kv(step["state_out"])
        return out

    return mixer, pools


def _attn_mixer(tensors, quant: bool):
    """Either decode attention oracle over a cache this call owns.

    Both append this token's K and V before reading, so the fake-quant chain writes the
    harness's own pages -- which is what the page gates then compare -- and the truth is
    given clones taken before that happens.
    """
    import reference

    w = decode_gated_attention.attn_weights(tensors)
    if quant:
        pages = {"k_pages": tensors["k_pages"], "v_pages": tensors["v_pages"]}
    else:
        pages = {"k_pages": tensors["k_pages"].clone(), "v_pages": tensors["v_pages"].clone()}
    step = reference.attn_w8a8_decode if quant else reference.attn_truth_decode

    def mixer(x):
        return step(x, w, tensors["rope_cos"], tensors["rope_sin"], QWEN3_8_27B,
                    pages["k_pages"], pages["v_pages"], tensors["block_table"],
                    tensors["seq_lens"], PAGE)

    return mixer, None


def mixer_oracle(kind: str, tensors, quant: bool):
    """Either mixer oracle closed over the harness's own tensors, and its pools."""
    return _gdn_mixer(tensors, quant) if kind == "gdn" else _attn_mixer(tensors, quant)


_TRUTH: dict[str, object] = {}


def truth(name: str):
    """The float64 value of one output, as the last golden left it."""
    return _TRUTH[name]


def golden_decode_layer(kind: str):
    """Both chains: the fake-quant one written out, the float64 truth stashed for the gates.

    The truth runs first, because the attention mixer's two oracles both append to the
    paged cache in place and only the fake-quant chain's writes are the ones the device is
    compared against.

    **The same weights serve both chains.** They are the int8 tensors the kernel reads,
    dequantised in float64 for the truth, so the floor the gates divide by is the cost of
    the activation path alone -- the per-token quantisers and the bf16 hand-offs -- with
    the weight quantisation held common. That is `mlp_truth`'s convention and
    `attn_truth_decode`'s.
    """
    import torch

    import reference

    def golden(tensors):
        reference.host_torch_setup()
        mlp_w = {name: tensors[name] for name in
                 ("w_gate", "w_gate_scale", "w_up", "w_up_scale", "w_down", "w_down_scale")}
        head = (tensors["residual"], tensors["delta"], tensors["norm_in_w"],
                tensors["norm_post_w"])

        mixer, _ = mixer_oracle(kind, tensors, quant=False)
        truth = reference.residual_layer(*head, mixer, mlp_w, QWEN3_8_27B, quant=False)
        _TRUTH["residual_out"] = truth["residual_out"]
        _TRUTH["mlp_out"] = truth["mlp_out"]

        mixer, pools = mixer_oracle(kind, tensors, quant=True)
        chain = reference.residual_layer(*head, mixer, mlp_w, QWEN3_8_27B, quant=True)
        tensors["residual_out"].copy_(chain["residual_out"].to(torch.float32))
        tensors["mlp_out"].copy_(chain["mlp_out"].to(torch.bfloat16))
        if pools is not None:
            for name, value in pools.items():
                tensors[name].copy_(value.to(tensors[name].dtype))

    return golden


# Acceptance, as ratios against the quantisation floor rather than fixed numbers, the shape
# both mixers and the MLP use: every output here is downstream of an int8 hand-off, so an
# element-wise bound would measure which side of a rounding boundary a value landed on.
# **`MAX_OVER_FLOOR` and `MAX_VS_GOLDEN` cannot both bind.** If the
# kernel may sit a whole floor away from the fake-quant chain, its distance to the truth can
# reach sqrt(2) floors when the two errors are orthogonal. The 1.05 this started at held only
# while the floor was dominated by a term the kernel reproduces exactly; taking the GDN tail to
# BF16 shrank that floor 2.8x, the kernel's own rounding is 0.36 to 0.42 of what is left, and
# the expected ratio is therefore sqrt(1 + 0.42^2) = 1.08. Three of four configurations read
# 1.042x to 1.050x against a 1.05 bound, which is a mis-calibrated gate rather than a kernel
# that moved. The binding gate is the distance to the chain the kernel implements; the truth
# ratio is a sanity bound, and a real regression shows up there as a multiple, not a percent.
MAX_OVER_FLOOR = 1.15
MAX_ROW_OVER_FLOOR = 1.25
MAX_VS_GOLDEN = 1.00
MAX_ROW_VS_GOLDEN = 1.50
MAX_POOL_REL = 2e-2


def compare_vs_floor(name: str):
    """One layer output against the float64 truth, beside what W8A8 alone costs."""
    def compare(actual, expected, **_kwargs):
        truth = _TRUTH[name].double()
        fq = expected.double()
        dev = actual.double()

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).max())

        floor, floor_row = rel(fq, truth), worst_row(fq, truth)
        got, got_row = rel(dev, truth), worst_row(dev, truth)
        vs_fq, vs_fq_row = rel(dev, fq), worst_row(dev, fq)
        ratio = got / floor if floor else float("inf")
        ratio_row = got_row / floor_row if floor_row else float("inf")
        fq_ratio = vs_fq / floor if floor else float("inf")
        fq_ratio_row = vs_fq_row / floor_row if floor_row else float("inf")
        line = (f"[stats] {name}: rel frob {got:.3e} vs the float64 truth against a "
                f"quantisation floor of {floor:.3e} ({ratio:.3f}x); worst token row "
                f"{got_row:.3e} against {floor_row:.3e} ({ratio_row:.3f}x); {vs_fq:.3e} vs "
                f"the fake-quant chain ({fq_ratio:.3f} of the floor, worst row "
                f"{fq_ratio_row:.3f})")
        ok = (ratio <= MAX_OVER_FLOOR and ratio_row <= MAX_ROW_OVER_FLOOR
              and fq_ratio <= MAX_VS_GOLDEN and fq_ratio_row <= MAX_ROW_VS_GOLDEN)
        return ok, "" if ok else line

    return compare


def compare_fns(kind: str) -> dict:
    """The gates: both layer outputs against the floor, and the mixer's pools in place."""
    gates = {name: compare_vs_floor(name) for name in ("residual_out", "mlp_out")}
    if kind == "gdn":
        gates.update({name: decode_gdn_block.compare_pool(name)
                      for name in ("conv_state", "state")})
    else:
        gates.update({name: decode_attention.compare_pages(name)
                      for name in ("k_pages", "v_pages")})
    return gates


if __name__ == "__main__":
    import argparse

    import weights
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--kind", type=str, default="gdn", choices=KINDS,
                        help="which mixer the layer is built around")
    parser.add_argument("--batch", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--head-tile", type=int, default=decode_gdn_block.HEAD_TILE,
                        help="gdn only: delta-rule head states per task")
    parser.add_argument("--context", type=int, default=320,
                        help="attn only: tokens in the cache, this one included")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="attn only: 0 derives it from --context")
    parser.add_argument("--splits", type=int, default=None, help="attn only")
    parser.add_argument("--real-layer", type=int, default=None,
                        help="use one real layer's weights: its mixer, MLP and both norms")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=range(5))
    args = parser.parse_args()
    cases = [args.batch] if args.batch else list(DEFAULT_CASES)

    paths = ({} if args.real_layer is None
             else weights.layer_paths(args.real_layer, args.kind))
    build_kwargs, spec_kwargs = {}, {}
    if args.kind == "gdn":
        build_kwargs["head_tile"] = args.head_tile
    else:
        max_pages = args.max_pages or (args.context + PAGE - 1) // PAGE
        build_kwargs.update(max_pages=max_pages, splits=args.splits)
        spec_kwargs.update(context=args.context, max_pages=max_pages)

    for batch in cases:
        print(f"[case] decode_layer batch={batch}", flush=True)
        result = run(
            fn=build_kernel(args.kind, batch, **build_kwargs),
            specs=build_tensor_specs(args.kind, batch=batch, **paths, **spec_kwargs),
            golden_fn=golden_decode_layer(args.kind),
            compare_fn=compare_fns(args.kind),
            config=run_config(args.backend_type, args.execution_mode, args.device,
                              args.enable_chip_swimlane),
            rtol=1e-2,
            atol=1e-2,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at batch={batch}")
