"""Golden-harness validation for the non-GEMM A2/A3 leaf operators."""

from __future__ import annotations

import argparse

import pypto.language as pl
import torch

from .index_cache import make_index_cache_quantize, quantize_index_cache
from .norm_rope import make_rope, make_rmsnorm, rms_norm_reference, rope_reference
from .swa_cache import (
    gather_swa_cache_reference,
    make_swa_cache_gather,
    make_swa_cache_publish,
    write_swa_cache_reference,
)


CAPACITY = 16
WIDTH = 256
HEADS = 2
HEAD_DIM = 128
ROPE_DIM = 64
INDEX_HEADS = 16
INDEX_DIM = 128
ACTIVE = 5
WINDOW = 8
CACHE_BLOCKS = 2


_norm = make_rmsnorm(CAPACITY, WIDTH)
_rope = make_rope(CAPACITY, HEADS, HEAD_DIM, ROPE_DIM)
_index_quant = make_index_cache_quantize(CAPACITY, INDEX_HEADS, INDEX_DIM)
_swa_publish = make_swa_cache_publish(CAPACITY, CACHE_BLOCKS, HEAD_DIM)
_swa_gather = make_swa_cache_gather(CAPACITY, WINDOW, CACHE_BLOCKS, HEAD_DIM)


@pl.jit
def rmsnorm_single_op(
    value: pl.Tensor[[CAPACITY, WIDTH], pl.BF16],
    weight: pl.Tensor[[WIDTH], pl.BF16],
    output: pl.InOut[pl.Tensor[[CAPACITY, WIDTH], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    _norm(value, weight, output, active)
    return output


@pl.jit
def rope_single_op(
    value: pl.Tensor[[CAPACITY, HEADS * HEAD_DIM], pl.BF16],
    cosine: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[CAPACITY, ROPE_DIM // 2], pl.FP32],
    output: pl.InOut[pl.Tensor[[CAPACITY, HEADS * HEAD_DIM], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    _rope(value, cosine, sine, output, active)
    return output


@pl.jit
def index_cache_single_op(
    value: pl.Tensor[[CAPACITY, INDEX_HEADS, INDEX_DIM], pl.BF16],
    payload: pl.InOut[pl.Tensor[[CAPACITY, INDEX_HEADS, INDEX_DIM], pl.INT8]],
    scales: pl.InOut[pl.Tensor[[CAPACITY, INDEX_HEADS], pl.FP16]],
    active: pl.Scalar[pl.INT32],
):
    _index_quant(value, payload, scales, active)
    return payload, scales


@pl.jit
def swa_cache_single_op(
    values: pl.Tensor[[CAPACITY, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[CAPACITY, 2], pl.INT64],
    indices: pl.Tensor[[CAPACITY, WINDOW], pl.INT32],
    cache: pl.InOut[pl.Tensor[[CACHE_BLOCKS, 128, 1, HEAD_DIM], pl.BF16]],
    selected: pl.InOut[pl.Tensor[[CAPACITY * WINDOW, HEAD_DIM], pl.BF16]],
    active: pl.Scalar[pl.INT32],
):
    _swa_publish(values, slots, cache, active)
    _swa_gather(cache, indices, selected, active)
    return cache, selected


def _base_inputs():
    torch.manual_seed(42)
    return torch.randn(CAPACITY, WIDTH, dtype=torch.bfloat16), torch.randn(WIDTH, dtype=torch.bfloat16)


def _specs_norm():
    from golden import ScalarSpec, TensorSpec

    value, weight = _base_inputs()
    return [
        TensorSpec("value", [CAPACITY, WIDTH], torch.bfloat16, init_value=value),
        TensorSpec("weight", [WIDTH], torch.bfloat16, init_value=weight),
        TensorSpec("output", [CAPACITY, WIDTH], torch.bfloat16, init_value=torch.zeros),
        ScalarSpec("active", torch.int32, ACTIVE),
    ]


def _golden_norm(values):
    values["output"][:] = rms_norm_reference(values["value"], values["weight"])
    values["output"][ACTIVE:] = 0


def _specs_rope():
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(43)
    value = torch.randn(CAPACITY, HEADS * HEAD_DIM, dtype=torch.bfloat16)
    cosine = torch.randn(CAPACITY, ROPE_DIM // 2)
    sine = torch.randn(CAPACITY, ROPE_DIM // 2)
    return [
        TensorSpec("value", [CAPACITY, HEADS * HEAD_DIM], torch.bfloat16, init_value=value),
        TensorSpec("cosine", [CAPACITY, ROPE_DIM // 2], torch.float32, init_value=cosine),
        TensorSpec("sine", [CAPACITY, ROPE_DIM // 2], torch.float32, init_value=sine),
        TensorSpec("output", [CAPACITY, HEADS * HEAD_DIM], torch.bfloat16, init_value=torch.zeros),
        ScalarSpec("active", torch.int32, ACTIVE),
    ]


def _golden_rope(values):
    values["output"][:] = rope_reference(values["value"], values["cosine"], values["sine"], HEADS, HEAD_DIM, ROPE_DIM)
    values["output"][ACTIVE:] = 0


def _specs_index():
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(44)
    value = torch.randn(CAPACITY, INDEX_HEADS, INDEX_DIM, dtype=torch.bfloat16)
    return [
        TensorSpec("value", [CAPACITY, INDEX_HEADS, INDEX_DIM], torch.bfloat16, init_value=value),
        TensorSpec("payload", [CAPACITY, INDEX_HEADS, INDEX_DIM], torch.int8, init_value=torch.zeros),
        TensorSpec("scales", [CAPACITY, INDEX_HEADS], torch.float16, init_value=torch.zeros),
        ScalarSpec("active", torch.int32, ACTIVE),
    ]


def _golden_index(values):
    payload, scales = quantize_index_cache(values["value"])
    payload[ACTIVE:] = 0
    scales[ACTIVE:] = 0
    values["payload"][:] = payload
    values["scales"][:] = scales


def _specs_swa():
    from golden import ScalarSpec, TensorSpec

    torch.manual_seed(45)
    values = torch.randn(CAPACITY, HEAD_DIM, dtype=torch.bfloat16)
    slots = torch.full((CAPACITY, 2), -1, dtype=torch.int64)
    slots[:ACTIVE] = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]], dtype=torch.int64)
    indices = torch.full((CAPACITY, WINDOW), -1, dtype=torch.int32)
    indices[:ACTIVE] = torch.arange(ACTIVE * WINDOW, dtype=torch.int32).reshape(ACTIVE, WINDOW)
    return [
        TensorSpec("values", [CAPACITY, HEAD_DIM], torch.bfloat16, init_value=values),
        TensorSpec("slots", [CAPACITY, 2], torch.int64, init_value=slots),
        TensorSpec("indices", [CAPACITY, WINDOW], torch.int32, init_value=indices),
        TensorSpec("cache", [CACHE_BLOCKS, 128, 1, HEAD_DIM], torch.bfloat16, init_value=torch.zeros),
        TensorSpec("selected", [CAPACITY * WINDOW, HEAD_DIM], torch.bfloat16, init_value=torch.zeros),
        ScalarSpec("active", torch.int32, ACTIVE),
    ]


def _golden_swa(values):
    cache = write_swa_cache_reference(values["cache"], values["values"], values["slots"])
    selected = gather_swa_cache_reference(cache, values["indices"])
    values["cache"][:] = cache
    values["selected"][: ACTIVE * WINDOW] = selected[:ACTIVE].reshape(ACTIVE * WINDOW, HEAD_DIM)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["norm", "rope", "index", "swa", "all"], default="all")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    from golden import run

    cases = {
        "norm": (rmsnorm_single_op, _specs_norm, _golden_norm),
        "rope": (rope_single_op, _specs_rope, _golden_rope),
        "index": (index_cache_single_op, _specs_index, _golden_index),
        "swa": (swa_cache_single_op, _specs_swa, _golden_swa),
    }
    selected = cases if args.case == "all" else {args.case: cases[args.case]}
    failed = False
    for name, (fn, specs, golden) in selected.items():
        print(f"[CASE] {name}")
        result = run(
            fn=fn,
            specs=specs(),
            golden_fn=golden,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compile_only=args.compile_only,
            rtol=2.0e-2,
            atol=2.0e-2,
        )
        print(result)
        failed |= not result.passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
