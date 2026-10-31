"""W8A8 routed and shared expert references and PyPTO leaf entry points."""

from __future__ import annotations

import torch
import pypto.language as pl

from .gemm import make_fp32_dequantize, make_int8_gemm, make_pack_weight
from .quant import make_dynamic_int8_quantize
from .quant import w8a8_linear


VEC_M_TILE = 8
ACT_TILE = 256


def w8a8_expert_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run one W8A8 expert projection using the shared linear golden."""

    return w8a8_linear(value, weight, scale, out_dtype=out_dtype)


def shared_expert_reference(
    value: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_weight: torch.Tensor,
    down_scale: torch.Tensor,
    clamp_limit: float = 10.0,
) -> torch.Tensor:
    """Reference for a shared W8A8 SwiGLU expert."""

    gate_up = w8a8_expert_linear(value, gate_up_weight, gate_up_scale, torch.bfloat16).float()
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=clamp_limit)
    up = up.clamp(-clamp_limit, clamp_limit)
    # vLLM-Ascend's A2/A3 clipped SwiGLU performs both products in BF16.
    sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
    silu_bf16 = gate.to(torch.bfloat16) * sigmoid_bf16
    hidden = silu_bf16 * up.to(torch.bfloat16)
    return w8a8_expert_linear(hidden, down_weight, down_scale)


def routed_expert_reference(
    value: torch.Tensor,
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w3: torch.Tensor,
    w3_scale: torch.Tensor,
    clamp_limit: float = 10.0,
) -> torch.Tensor:
    """Reference for top-k routed W8A8 SwiGLU experts.

    ``w1`` and ``w3`` are gate/up weights with shape ``[E, I, D]`` and ``w2``
    is the down weight with shape ``[E, D, I]``.  Routing is kept explicit so
    each expert boundary can be compared with the vLLM A2/A3 dump.
    """

    result = torch.zeros_like(value, dtype=torch.float32)
    for token in range(value.shape[0]):
        for route in range(expert_ids.shape[1]):
            expert = int(expert_ids[token, route])
            gate = w8a8_expert_linear(value[token:token + 1], w1[expert], w1_scale[expert]).float()
            up = w8a8_expert_linear(value[token:token + 1], w3[expert], w3_scale[expert]).float()
            gate = gate.clamp(max=clamp_limit)
            up = up.clamp(-clamp_limit, clamp_limit)
            sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
            silu_bf16 = gate.to(torch.bfloat16) * sigmoid_bf16
            hidden = silu_bf16 * up.to(torch.bfloat16)
            down = w8a8_expert_linear(hidden, w2[expert], w2_scale[expert]).float()
            result[token] += down[0] * router_weights[token, route].float()
    return result.to(torch.bfloat16)


def make_w8a8_expert_projection(capacity: int, input_width: int, output_width: int):
    """Build the shared leaf used by routed and shared W8A8 projections.

    Dispatch/combine metadata stays outside this kernel.  Each routed expert
    receives a contiguous token slice and uses the same A2/A3 W8A8 boundary as
    the shared expert: dynamic INT8 activation, INT32 dot, FP32 scales, BF16
    output.
    """

    quantize = make_dynamic_int8_quantize(capacity, input_width)
    gemm = make_int8_gemm(capacity, input_width, output_width)
    dequantize = make_fp32_dequantize(capacity, output_width)

    @pl.jit.inline
    def projection(
        value: pl.Tensor[[capacity, input_width], pl.BF16],
        packed_weight: pl.Tensor[[input_width, output_width], pl.INT8],
        channel_scale: pl.Tensor[[output_width], pl.FP32],
        payload: pl.Tensor[[capacity, input_width], pl.INT8],
        token_scale: pl.Tensor[[capacity, 1], pl.FP32],
        accumulator: pl.Tensor[[capacity, output_width], pl.INT32],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        quantize(value, payload, token_scale, active)
        gemm(payload, packed_weight, accumulator, active)
        dequantize(accumulator, token_scale, channel_scale, output, active)
        return output

    return projection


def make_shared_expert_projection(capacity: int, input_width: int, output_width: int):
    """Alias the W8A8 projection leaf for the shared expert path."""

    return make_w8a8_expert_projection(capacity, input_width, output_width)


def make_routed_expert_projection(capacity: int, input_width: int, output_width: int):
    """Alias the W8A8 projection leaf for one grouped routed expert."""

    return make_w8a8_expert_projection(capacity, input_width, output_width)


def make_shared_expert_w8a8(
    capacity: int,
    hidden: int,
    intermediate: int,
    clamp_limit: float = 10.0,
):
    """Build the complete W8A8 shared expert for one dispatched token slab.

    The activation is evaluated in FP32, narrowed to BF16 at the same point
    as vLLM-Ascend, then dynamically re-quantized before the down projection.
    A routed expert uses this exact entry point after host dispatch has made a
    contiguous ``[tokens, hidden]`` slice for one expert.
    """

    if capacity % 16 or hidden % 256 or intermediate % 256:
        raise ValueError("A2/A3 expert GEMM requires capacity multiple of 16 and K widths multiple of 256")
    gate_up_width = intermediate * 2
    quant_x = make_dynamic_int8_quantize(capacity, hidden)
    quant_hidden = make_dynamic_int8_quantize(capacity, intermediate)
    pack_gate_up = make_pack_weight(hidden, gate_up_width)
    pack_down = make_pack_weight(intermediate, hidden)
    dot_gate_up = make_int8_gemm(capacity, hidden, gate_up_width)
    dot_down = make_int8_gemm(capacity, intermediate, hidden)
    dequant_gate_up = make_fp32_dequantize(capacity, gate_up_width)
    dequant_down = make_fp32_dequantize(capacity, hidden)

    @pl.jit
    def shared_expert(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_up_weight: pl.Tensor[[hidden, gate_up_width], pl.INT8],
        gate_up_scale: pl.Tensor[[gate_up_width], pl.FP32],
        down_weight: pl.Tensor[[intermediate, hidden], pl.INT8],
        down_scale: pl.Tensor[[hidden], pl.FP32],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        x_payload = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_x(value, x_payload, x_scale, active)

        gate_up_packed = pl.create_tensor([hidden * (gate_up_width // 64), 64], dtype=pl.INT8)
        pack_gate_up(gate_up_weight, gate_up_packed)
        gate_up_acc = pl.create_tensor([capacity, gate_up_width], dtype=pl.INT32)
        dot_gate_up(x_payload, gate_up_packed, gate_up_acc, active)
        gate_up = pl.create_tensor([capacity, gate_up_width], dtype=pl.BF16)
        dequant_gate_up(gate_up_acc, x_scale, gate_up_scale, gate_up, active)

        activation = pl.create_tensor([capacity, intermediate], dtype=pl.BF16)
        for block in pl.spmd((active + VEC_M_TILE - 1) // VEC_M_TILE * (intermediate // ACT_TILE), name_hint="dsv41_a2a3_swiglu_v2"):
            token = block // (intermediate // ACT_TILE) * VEC_M_TILE
            column = block % (intermediate // ACT_TILE) * ACT_TILE
            rows = pl.min(VEC_M_TILE, active - token)
            # vLLM's A2/A3 ``SiluAndMulWithClamp`` keeps the two products in
            # BF16.  Use FP32 only for the exponential, then multiply the
            # BF16 sigmoid, gate, and up values without an FP32 promotion.
            gate = pl.cast(pl.slice(gate_up, [VEC_M_TILE, ACT_TILE], [token, column]), pl.FP32)
            up = pl.cast(pl.slice(gate_up, [VEC_M_TILE, ACT_TILE], [token, intermediate + column]), pl.FP32)
            gate = pl.minimum(gate, clamp_limit)
            up = pl.maximum(pl.minimum(up, clamp_limit), -clamp_limit)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
            sigmoid_bf16 = pl.cast(sigmoid, pl.BF16, mode="rint")
            silu = pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32))
            silu_bf16 = pl.cast(silu, pl.BF16, mode="rint")
            activated = pl.mul(pl.cast(silu_bf16, pl.FP32), up)
            activation[token:token + VEC_M_TILE, column:column + ACT_TILE] = pl.set_validshape(
                pl.cast(activated, pl.BF16, mode="rint"), rows, ACT_TILE
            )

        hidden_payload = pl.create_tensor([capacity, intermediate], dtype=pl.INT8)
        hidden_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_hidden(activation, hidden_payload, hidden_scale, active)
        down_packed = pl.create_tensor([intermediate * (hidden // 64), 64], dtype=pl.INT8)
        pack_down(down_weight, down_packed)
        down_acc = pl.create_tensor([capacity, hidden], dtype=pl.INT32)
        dot_down(hidden_payload, down_packed, down_acc, active)
        dequant_down(down_acc, hidden_scale, down_scale, output, active)
        return output

    return shared_expert


def make_routed_expert_w8a8(capacity: int, hidden: int, intermediate: int, clamp_limit: float = 10.0):
    """Alias the full expert kernel for one post-dispatch routed expert."""

    return make_shared_expert_w8a8(capacity, hidden, intermediate, clamp_limit)


__all__ = [
    "make_routed_expert_projection",
    "make_routed_expert_w8a8",
    "make_shared_expert_projection",
    "make_shared_expert_w8a8",
    "make_w8a8_expert_projection",
    "routed_expert_reference",
    "shared_expert_reference",
    "w8a8_expert_linear",
]
