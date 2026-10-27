# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host input preparation for GLM-5.3-Flash checkpoint validation."""

import torch

from models.glm5_3_flash.config import D, EP_SIZE, FLASH, MOE_INTER, N_EXPERTS, N_LOCAL_EXPERTS


def component_options(description, *, tokens=17, layers=None):
    """Parse the common standalone component and checkpoint validation options."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hidden-input", type=Path)
    parser.add_argument("--tokens", type=int, default=tokens)
    parser.add_argument("--compile-only", action="store_true")
    if layers is not None:
        parser.add_argument("--layer", type=int, default=0)
    options = parser.parse_args()
    if options.tokens < 1:
        parser.error("--tokens must be positive")
    if layers is not None and not 0 <= options.layer < layers:
        parser.error(f"--layer must be in [0, {layers})")
    return options


def component_hidden(options, width, generator):
    """Load captured component input or construct deterministic finite BF16 rows."""
    if options.hidden_input is None:
        return torch.randn(options.tokens, width, generator=generator).bfloat16()
    value = torch.load(options.hidden_input, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (options.tokens, width):
        raise ValueError("--hidden-input must have shape [tokens, input_width]")
    if not torch.isfinite(value).all():
        raise ValueError("--hidden-input must contain finite values")
    return value.bfloat16().contiguous()


def build_checkpoint_moe_specs(checkpoint, layer_id, *, capacity, num_tokens, num_ranks, hidden=None):
    """Stage real FFN weights and synthetic or supplied pre-norm MoE activations.

    ``hidden`` is BF16 ``[num_ranks, num_tokens, D]`` after mHC pre mixing.
    Without it, random embeddings are folded with the checkpoint's FFN mHC
    weights. This fixture does not execute the preceding Transformer layers.
    """
    from golden import ScalarSpec, TensorSpec
    from models.glm5_3_flash.weights import Checkpoint, pack_int8_nz

    if num_ranks != EP_SIZE:
        raise ValueError(f"checkpoint fixture requires the compiled EP size {EP_SIZE}, got {num_ranks}")
    if not 0 <= num_tokens <= capacity:
        raise ValueError(f"num_tokens must be between 0 and capacity {capacity}, got {num_tokens}")
    if hidden is not None:
        expected = (num_ranks, num_tokens, D)
        if not isinstance(hidden, torch.Tensor) or hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != expected:
            raise ValueError(f"hidden must be a BF16 tensor with shape {expected}")
        if hidden.device.type != "cpu":
            raise ValueError("hidden must be a CPU tensor")

    checkpoint = Checkpoint(checkpoint) if not isinstance(checkpoint, Checkpoint) else checkpoint
    if not FLASH.first_k_dense_replace <= layer_id < FLASH.num_hidden_layers:
        raise ValueError("the layer entries require a sparse FFN layer in the text backbone")

    x_mixed = torch.zeros(num_ranks, capacity, D, dtype=torch.bfloat16)
    if hidden is None:
        from models.glm5_3_flash.golden import hc_mixes, hc_pre, hc_seed

        generator = torch.Generator().manual_seed(7)
        embeddings = torch.randn(num_ranks, capacity, D, generator=generator).bfloat16()
        mhc = checkpoint.mhc(layer_id, "ffn")
        with torch.no_grad():
            for rank in range(num_ranks):
                stream = hc_seed(embeddings[rank])
                mix = hc_mixes(stream, mhc["function"], mhc["scale"], mhc["base"])[0]
                x_mixed[rank, :num_tokens] = hc_pre(stream, mix)[:num_tokens]
    else:
        x_mixed[:, :num_tokens] = hidden

    # Ordered local shapes match the L3 decode/prefill MoE ABI.
    layout = {
        "norm_weight": ((D,), torch.bfloat16),
        "gate_weight": ((N_EXPERTS, D), torch.bfloat16),
        "correction_bias": ((N_EXPERTS,), torch.float32),
        "w_gate_up": ((N_LOCAL_EXPERTS, 2 * MOE_INTER, D), torch.int8),
        "w_gate_up_scale": ((N_LOCAL_EXPERTS, 2 * MOE_INTER), torch.float32),
        "w_down": ((N_LOCAL_EXPERTS, D, MOE_INTER), torch.int8),
        "w_down_scale": ((N_LOCAL_EXPERTS, D), torch.float32),
        "shared_w_gate_up": ((2 * MOE_INTER, D), torch.int8),
        "shared_w_gate_up_scale": ((2 * MOE_INTER,), torch.float32),
        "shared_w_down": ((D, MOE_INTER), torch.int8),
        "shared_w_down_scale": ((D,), torch.float32),
    }
    weights = {}
    for rank in range(num_ranks):
        local = checkpoint.ffn(layer_id, ep_rank=rank, ep=num_ranks)
        if local.keys() != layout.keys():
            raise ValueError("checkpoint FFN tensors differ from the composed MoE input names")
        for name, (shape, dtype) in layout.items():
            value = local[name]
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"{name}: expected {shape} {dtype}, got {tuple(value.shape)} {value.dtype}")
            if rank == 0:
                weights[name] = torch.empty((num_ranks, *shape), dtype=dtype)
            weights[name][rank].copy_(value)
        del local, value

    for name in ("shared_w_gate_up", "shared_w_down"):
        weights[name] = pack_int8_nz(weights[name])

    specs = [TensorSpec("x_mixed", list(x_mixed.shape), x_mixed.dtype, init_value=x_mixed)]
    specs += [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in weights.items()]
    specs += [
        TensorSpec("y", [num_ranks, capacity, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("num_ranks", torch.int32, num_ranks),
    ]
    return specs


__all__ = ["build_checkpoint_moe_specs", "component_hidden", "component_options"]
