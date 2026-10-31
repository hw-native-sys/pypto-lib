# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill HC-pre custom boundaries with PyPTO HC-post, Norm and shared MLP."""

from pathlib import Path

import pypto.language as pl
import torch

from .ecotech_shared import make_ecotech_shared_inline
from .hc import make_hc_post_inline
from .hc_pre_runtime import hc_pre_host
from .runtime_utils import tensor_sha as _tensor_sha
from .prefill_norm import make_prefill_norm


def _make_post(capacity, config):
    compute = make_hc_post_inline(capacity, config)

    @pl.jit
    def post(
        sublayer: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        residual: pl.InOut[pl.Tensor[[capacity, 4, 5120], pl.BF16]],
        post_mix: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, 4, 4], pl.FP32]],
        changed_hidden: pl.InOut[pl.Tensor[[capacity, 4, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(sublayer, residual, post_mix, comb, changed_hidden, active)
        return sublayer, residual, post_mix, comb, changed_hidden

    return post


def _make_copy(capacity, width, dtype, tile):
    @pl.jit.inline
    def copy_active(
        source: pl.Tensor[[capacity, width], dtype],
        target: pl.Tensor[[capacity, width], dtype],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 7) // 8) * (width // tile), name_hint="staged_shared_export"):
            token = block // (width // tile) * 8
            column = block % (width // tile) * tile
            rows = pl.min(8, active - token)
            value = pl.slice(source, [8, tile], [token, column], valid_shape=[rows, tile])
            target[token:token + 8, column:column + tile] = value
        return target

    return copy_active


def _make_norm_shared(capacity, intermediate):
    standalone = make_prefill_norm(capacity, 5120)
    norm = pl.jit.inline(standalone._func)
    if norm._func is not standalone._func or norm._func_type != "inline":
        raise RuntimeError("Require the existing formal Prefill RMSNorm function")
    shared = make_ecotech_shared_inline(capacity, 5120, intermediate)
    width = intermediate * 2
    padded = ((intermediate + 255) // 256) * 256
    input_copy = _make_copy(capacity, 5120, pl.INT8, 256)
    scale_copy = _make_copy(capacity, 8, pl.FP32, 8)
    accumulator_copy = _make_copy(capacity, width, pl.INT32, 256)
    activation_copy = _make_copy(capacity, padded, pl.FP32, 256)
    payload_copy = _make_copy(capacity, padded, pl.INT8, 256)

    @pl.jit
    def norm_shared(
        collapsed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        norm_weight: pl.InOut[pl.Tensor[[5120], pl.BF16]],
        gate_weight: pl.InOut[pl.Tensor[[5120, width], pl.INT8]],
        gate_scale: pl.InOut[pl.Tensor[[width], pl.FP32]],
        down_weight: pl.InOut[pl.Tensor[[padded, 5120], pl.INT8]],
        down_scale: pl.InOut[pl.Tensor[[5120], pl.FP32]],
        normed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        shared_output: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        input_payload: pl.InOut[pl.Tensor[[capacity, 5120], pl.INT8]],
        input_scale: pl.InOut[pl.Tensor[[capacity, 8], pl.FP32]],
        accumulator: pl.InOut[pl.Tensor[[capacity, width], pl.INT32]],
        activation: pl.InOut[pl.Tensor[[capacity, padded], pl.FP32]],
        down_payload: pl.InOut[pl.Tensor[[capacity, padded], pl.INT8]],
        down_token_scale: pl.InOut[pl.Tensor[[capacity, 8], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        norm(collapsed, norm_weight, normed, active)
        result, xq, xs, acc, act, dq, ds = shared(
            normed, gate_weight, gate_scale, down_weight, down_scale, shared_output, active,
        )
        input_copy(xq, input_payload, active)
        scale_copy(xs, input_scale, active)
        accumulator_copy(acc, accumulator, active)
        activation_copy(act, activation, active)
        payload_copy(dq, down_payload, active)
        scale_copy(ds, down_token_scale, active)
        return (
            collapsed, norm_weight, gate_weight, gate_scale, down_weight, down_scale,
            normed, result, input_payload, input_scale, accumulator, activation, down_payload, down_token_scale,
        )

    return norm_shared


class EcotechMhcFfnStaged:
    """Stage actual Prefill buffers through HC-pre and two PyPTO entries."""

    def __init__(self, capacity, config, intermediate=2304):
        expected = dict(hidden_size=5120, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-20)
        if capacity <= 0 or capacity % 16 or any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("Require capacity/16 and the original four-stream HC attributes")
        if intermediate <= 0 or intermediate % 128:
            raise ValueError("Require a positive shared intermediate divisible by128")
        self.capacity, self.config = capacity, dict(config)
        self.post = _make_post(capacity, config)
        self.norm_shared = _make_norm_shared(capacity, intermediate)

    def execute(self, inputs, outputs, *, active, runtime_config, output_dir, execute):
        """Use an executor that checks actual DSL readonly and inactive buffers."""
        if not 0 < active <= self.capacity or any(value.device.type != "cpu" for value in inputs.values()):
            raise ValueError("Require valid active rows and CPU-staged actual operands")
        directory = Path(output_dir)
        directory.mkdir(exist_ok=False)
        before = {key: _tensor_sha(value) for key, value in inputs.items()}
        tails = {key: _tensor_sha(value[active:]) for key, value in outputs.items()}
        result = {key: value.clone() for key, value in outputs.items()}
        receipt = dict(boundary="_C_ascend.npu_hc_pre_v3", actual_operands=True, native_reference_inputs=False)
        receipt["HC_boundaries"] = {}

        def hc(hidden, incoming, kind):
            keys = (kind + "_function", kind + "_scale", kind + "_base")
            function, scale, base = (inputs[key] for key in keys)
            values, native = hc_pre_host(
                hidden[:active].contiguous(), function, scale, base, incoming[:active].contiguous(),
                config=self.config, runtime_config=runtime_config, output_dir=directory / (kind + "-hc"),
            )
            receipt["HC_boundaries"][kind] = native
            for source, target in (("next_pre", "pre"), ("post", "post"), ("comb", "comb"), ("collapsed", "collapsed")):
                result[kind + "_" + target][:active].copy_(values[source])

        hc(inputs["x"], inputs["incoming"], "attn")
        scalars = [("active", torch.int32, active)]
        changed = execute(
            "prefix_HC_post", self.post,
            dict(sublayer=inputs["attention_sublayer"], residual=inputs["x"],
                 post_mix=result["attn_post"], comb=result["attn_comb"]),
            dict(changed_hidden=result["changed_hidden"]),
            scalars=scalars, active_rows=dict(changed_hidden=active),
        )
        result["changed_hidden"] = changed["changed_hidden"]
        hc(result["changed_hidden"], result["attn_pre"], "ffn")
        names = ("normed", "shared_output", "input_payload", "input_scale", "accumulator", "activation",
                 "down_payload", "down_token_scale")
        shared_inputs = {key: inputs[key] for key in ("norm_weight", "gate_weight", "gate_scale", "down_weight", "down_scale")}
        shared_inputs = dict(collapsed=result["ffn_collapsed"], **shared_inputs)
        shared = execute(
            "prefix_Norm_complete_shared", self.norm_shared, shared_inputs,
            {key: result[key] for key in names},
            scalars=scalars, active_rows={key: active for key in names},
        )
        result.update(shared)
        receipt["readonly"] = {key: _tensor_sha(value) == before[key] for key, value in inputs.items()}
        receipt["inactive"] = {key: _tensor_sha(value[active:]) == tails[key] for key, value in result.items()}
        receipt["finite"] = {key: bool(torch.isfinite(value).all()) for key, value in result.items()}
        receipt["actual_outputs"] = {key: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=_tensor_sha(value))
                                     for key, value in result.items()}
        receipt["passed"] = all(all(receipt[key].values()) for key in ("readonly", "inactive", "finite"))
        if not receipt["passed"]:
            raise RuntimeError("Staged HC/FFN readonly, inactive or finite output check failed")
        return result, receipt
