"""GLM-4.5 Shared Expert FFN with Quantization.

Fused: per-token quantize → quantized matmul (up+gate proj) → dequant →
SwiGLU → quantize → quantized matmul (down proj) → dequant.
"""

import pypto.language as pl
from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_dequant,
    tensor_matmul,
    tensor_mul,
    tensor_sigmoid,
    tensor_symmetric_quant,
    tensor_view,
)


@pl.function
def ffn_shared_expert_quant(
    hidden_states: pl.Tensor,
    w13: pl.Tensor,
    w13_scale: pl.Tensor,
    w2: pl.Tensor,
    w2_scale: pl.Tensor,
    ffn_res: pl.Out[pl.Tensor],
) -> pl.Tensor:
    """Quantized shared-expert FFN.

    hidden_states: [bs, hidden_size]
    w13:           [2*intermediate_size, hidden_size] (INT8)
    w13_scale:     [2*intermediate_size, 1]
    w2:            [hidden_size, intermediate_size] (INT8)
    w2_scale:      [hidden_size, 1]
    """
    bs = hidden_states.shape[0]
    intermediate_2 = w13.shape[0]
    intermediate = intermediate_2 // 2

    # Step 1: Quantize input
    x_quant = pl.create_tensor([bs, hidden_states.shape[1]], dtype=pl.INT8)
    x_scale = pl.create_tensor([bs, 1], dtype=pl.FP32)
    tensor_symmetric_quant(hidden_states, x_quant, x_scale)

    # Step 2: Quantized up+gate projection
    w13_t = pl.transpose(w13)
    up_gate_raw = tensor_matmul(
        tensor_cast(x_quant, pl.FP32),
        tensor_cast(w13_t, pl.FP32),
    )
    up_gate = tensor_dequant(up_gate_raw, x_scale)
    up_gate = tensor_mul(up_gate, w13_scale)

    # Step 3: SwiGLU
    gate_proj = pl.slice(up_gate, [bs, intermediate], [0, 0])
    up_proj = pl.slice(up_gate, [bs, intermediate], [0, intermediate])
    sig = tensor_sigmoid(gate_proj)
    activated = tensor_mul(sig, up_proj)

    # Step 4: Quantize activated
    act_quant = pl.create_tensor([bs, intermediate], dtype=pl.INT8)
    act_scale = pl.create_tensor([bs, 1], dtype=pl.FP32)
    tensor_symmetric_quant(activated, act_quant, act_scale)

    # Step 5: Down projection
    w2_t = pl.transpose(w2)
    down_raw = tensor_matmul(
        tensor_cast(act_quant, pl.FP32),
        tensor_cast(w2_t, pl.FP32),
    )
    down = tensor_dequant(down_raw, act_scale)
    down = tensor_mul(down, w2_scale)

    pl.assemble(ffn_res, down, [0, 0])
    return ffn_res
