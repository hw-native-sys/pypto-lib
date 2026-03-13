"""GLM-4.5 FFN common interface — quantization and activation helpers."""

import pypto.language as pl
from tensor_functions import (
    tensor_abs,
    tensor_amax,
    tensor_cast,
    tensor_div,
    tensor_full,
    tensor_mul,
    tensor_sigmoid,
    tensor_symmetric_quant,
    tensor_view,
)


@pl.function
def symmetric_quantization_per_token(
    input_tensor: pl.Tensor,
    quant_out: pl.Out[pl.Tensor],
    scale_out: pl.Out[pl.Tensor],
) -> None:
    """Per-token symmetric quantization to INT8.

    scale = max(abs(x), axis=-1) / 127
    quant = round(x / scale)
    """
    tensor_symmetric_quant(input_tensor, quant_out, scale_out)


@pl.function
def dequant_dynamic(
    in_tensor: pl.Tensor,
    scale_1: pl.Tensor,
    scale_2: pl.Tensor,
    output: pl.Out[pl.Tensor],
) -> pl.Tensor:
    """Dynamic dequantization with two scales: out = in * scale_1 * scale_2."""
    fp_val = tensor_cast(in_tensor, pl.FP32)
    tmp = tensor_mul(fp_val, scale_1)
    result = tensor_mul(tmp, scale_2)
    pl.assemble(output, result, [0, 0])
    return output


@pl.function
def swiglu(up_proj: pl.Tensor,
           output: pl.Out[pl.Tensor]) -> pl.Tensor:
    """SwiGLU activation: left * sigmoid(-left) * right.

    up_proj: [M, 2*D] — split into left [M, D] and right [M, D].
    """
    M = up_proj.shape[0]
    D = up_proj.shape[1] // 2

    left = pl.slice(up_proj, [M, D], [0, 0])
    right = pl.slice(up_proj, [M, D], [0, D])

    sig = tensor_sigmoid(left)
    gated = tensor_mul(sig, right)
    pl.assemble(output, gated, [0, 0])
    return output
