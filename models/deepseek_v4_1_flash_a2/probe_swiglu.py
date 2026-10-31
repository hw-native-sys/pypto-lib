"""Probe the A2/A3 SwiGLU arithmetic boundary with frozen gate/up values."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import pypto.language as pl

@pl.jit
def _probe_plain(
    value: pl.Tensor[[8, 4608], pl.BF16],
    sigmoid: pl.Out[pl.Tensor[[8, 256], pl.BF16]],
    activation: pl.Out[pl.Tensor[[8, 256], pl.BF16]],
):
    with pl.spmd(1):
        _ = pl.tile.get_block_idx()
        gate = pl.cast(pl.slice(value, [8, 256], [0, 0]), pl.FP32)
        up = pl.cast(pl.slice(value, [8, 256], [0, 2304]), pl.FP32)
        value_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0))
        sigmoid_bf16 = pl.cast(value_sigmoid, pl.BF16, mode="rint")
        silu_bf16 = pl.cast(pl.mul(gate, pl.cast(sigmoid_bf16, pl.FP32)), pl.BF16, mode="rint")
        sigmoid[:, :] = sigmoid_bf16
        activation[:, :] = pl.cast(pl.mul(pl.cast(silu_bf16, pl.FP32), up), pl.BF16, mode="rint")
    return sigmoid, activation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-up", type=Path, required=True)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    payload = torch.load(args.gate_up, map_location="cpu", weights_only=False)
    if isinstance(payload.get("nodes"), dict):
        value = payload["nodes"]["shared_gate_up"][:8].contiguous()
    else:
        value = payload["shared_gate_up"][:8, :512].contiguous()
    if value.shape[1:] != (4608,) or value.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 [8,4608] gate/up tensor, got {tuple(value.shape)} {value.dtype}")
    if value.shape[0] < 8:
        value = torch.nn.functional.pad(value, (0, 0, 0, 8 - value.shape[0]))

    from golden import TensorSpec, run

    def golden_fn(tensors):
        gate = tensors["value"][:, :256].float().clamp(max=10.0)
        up = tensors["value"][:, 2304:2560].float().clamp(-10.0, 10.0)
        sigmoid_bf16 = torch.sigmoid(gate).to(torch.bfloat16)
        silu_bf16 = (gate * sigmoid_bf16.float()).to(torch.bfloat16)
        tensors["sigmoid"][:] = sigmoid_bf16
        tensors["activation"][:] = (silu_bf16.float() * up).to(torch.bfloat16)

    for name, fn in (("plain", _probe_plain),):
        def compare(actual, expected, *, actual_outputs, **kwargs):
            del kwargs
            delta = (actual.float() - expected.float()).abs()
            print(
                f"{name}: sigmoid_row6={actual_outputs['sigmoid'][6, :8].detach().cpu().tolist()} "
                f"activation_row6={actual_outputs['activation'][6, :8].detach().cpu().tolist()} "
                f"max_abs={float(delta.max()):.6g} n_diff={int((delta > 0).sum())}",
                flush=True,
            )
            return bool(torch.isclose(actual.float(), expected.float(), rtol=0.0, atol=0.0).all()), "CPU reference mismatch"

        result = run(
            fn=fn,
            specs=[
                TensorSpec("value", [8, 4608], torch.bfloat16, init_value=value),
                TensorSpec("sigmoid", [8, 256], torch.bfloat16),
                TensorSpec("activation", [8, 256], torch.bfloat16),
            ],
            golden_fn=golden_fn,
            config=dict(backend_type="a2a3", execution_mode=args.sim, device_id=args.device),
            compare_fn={"sigmoid": compare, "activation": compare},
        )
        print(f"{name}: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
