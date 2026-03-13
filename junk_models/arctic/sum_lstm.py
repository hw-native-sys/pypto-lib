"""Arctic LSTM model — ported to new pypto coding style.

Implements the Snowflake Arctic LSTM speculator with:
- rms_norm_pure: RMSNorm without learnable parameters
- gelu_activation_core: GELU via sigmoid approximation
- sum_lstm_compute: full LSTM gate computation

All functions are opaque (no incore boundary specified) and call
tensor_functions primitives.
"""

import pypto.language as pl

from tensor_functions import (
    tensor_add,
    tensor_cast,
    tensor_div,
    tensor_mul,
    tensor_sigmoid,
    tensor_sqrt,
    tensor_sum,
    tensor_view,
)

BATCH_SIZE = 32
D_GATE = 4096
D_GATE_4 = 16384


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

@pl.function
def rms_norm_pure(x: pl.Tensor, epsilon: float) -> pl.Tensor:
    """Pure RMSNorm: x / sqrt(mean(x^2) + eps).

    No learnable parameters.
    """
    N = x.shape[-1]
    x_fp32 = tensor_cast(x, pl.FP32)

    sq = tensor_mul(x_fp32, x_fp32)
    sq_scaled = tensor_mul(sq, 1.0 / N)
    row_sum = tensor_sum(sq_scaled, axis=-1, keepdim=True)

    denom = tensor_add(row_sum, epsilon)
    denom = tensor_sqrt(denom)

    normed = tensor_div(x_fp32, denom)
    return tensor_cast(normed, x.dtype)


@pl.function
def gelu_activation_core(x: pl.Tensor) -> pl.Tensor:
    """GELU activation (sigmoid approximation): x * sigmoid(1.702 * x)."""
    x_scaled = tensor_mul(x, 1.702)
    sig = tensor_sigmoid(x_scaled)
    return tensor_mul(x, sig)


# -----------------------------------------------------------------------
# Main LSTM computation
# -----------------------------------------------------------------------

@pl.function
def sum_lstm_compute(
    states_4d: pl.Tensor,
    z4_4d: pl.Tensor,
    prev_cell: pl.Tensor,
    w_cell: pl.Tensor,
    b_cell: pl.Tensor,
    w_state: pl.Tensor,
    b_state: pl.Tensor,
    alpha: float,
    eps_cell: float,
    eps_state: float,
    h_out: pl.Out[pl.Tensor],
    c_out: pl.Out[pl.Tensor],
) -> None:
    """Core computation logic for Snowflake Arctic LSTM.

    Parameters
    ----------
    states_4d : [batch, 4*hidden]  pre-gate states
    z4_4d     : [batch, 4*hidden]  residual input
    prev_cell : [batch, hidden]    previous cell state
    w_cell, b_cell : [1, hidden]   cell normalisation weights/biases
    w_state, b_state : [1, hidden] state normalisation weights/biases
    alpha     : residual scaling factor
    eps_cell, eps_state : RMSNorm epsilon values
    h_out     : [batch, hidden]    output hidden state
    c_out     : [batch, hidden]    output cell state
    """
    batch_size = states_4d.shape[0]
    hidden_dim_4 = states_4d.shape[1]
    hidden_dim = prev_cell.shape[1]

    w_cell_2d = pl.reshape(w_cell, [1, hidden_dim])
    b_cell_2d = pl.reshape(b_cell, [1, hidden_dim])
    w_state_2d = pl.reshape(w_state, [1, hidden_dim])
    b_state_2d = pl.reshape(b_state, [1, hidden_dim])

    for bs in pl.range(0, batch_size, 1):
        # Step 1: input fusion (states + alpha * z4)
        states_tile = tensor_cast(
            pl.slice(states_4d, [1, hidden_dim_4], [bs, 0]), pl.FP32)
        z4_tile = tensor_cast(
            pl.slice(z4_4d, [1, hidden_dim_4], [bs, 0]), pl.FP32)
        z4_scaled = tensor_mul(z4_tile, alpha)
        fused = tensor_add(states_tile, z4_scaled)

        # Step 2: split into 4 gates
        pre_f = pl.slice(fused, [1, hidden_dim], [0, 0])
        pre_i = pl.slice(fused, [1, hidden_dim], [0, hidden_dim])
        pre_o = pl.slice(fused, [1, hidden_dim], [0, hidden_dim * 2])
        pre_c = pl.slice(fused, [1, hidden_dim], [0, hidden_dim * 3])

        # Step 3: gate activations
        f_gate = tensor_sigmoid(pre_f)
        i_gate = tensor_sigmoid(pre_i)
        o_gate = tensor_sigmoid(pre_o)

        # Step 4: cell candidate
        c_cand = rms_norm_pure(pre_c, eps_cell)
        c_cand = tensor_mul(c_cand, tensor_cast(w_cell_2d, pl.FP32))
        c_cand = tensor_add(c_cand, tensor_cast(b_cell_2d, pl.FP32))
        c_act = gelu_activation_core(c_cand)

        # Step 5: cell update  c_new = prev*f + c_act*i
        prev_tile = tensor_cast(
            pl.slice(prev_cell, [1, hidden_dim], [bs, 0]), pl.FP32)
        term1 = tensor_mul(prev_tile, f_gate)
        term2 = tensor_mul(c_act, i_gate)
        c_new = tensor_add(term1, term2)
        c_new_out = tensor_cast(c_new, states_4d.dtype)
        pl.assemble(c_out, c_new_out, [bs, 0])

        # Step 6: hidden state
        h_temp = rms_norm_pure(c_new, eps_state)
        h_temp = tensor_mul(h_temp, tensor_cast(w_state_2d, pl.FP32))
        h_temp = tensor_add(h_temp, tensor_cast(b_state_2d, pl.FP32))
        h_act = gelu_activation_core(h_temp)

        # Step 7: output  h_new = h_act * o_gate
        h_new = tensor_mul(h_act, o_gate)
        h_new_out = tensor_cast(h_new, states_4d.dtype)
        pl.assemble(h_out, h_new_out, [bs, 0])
