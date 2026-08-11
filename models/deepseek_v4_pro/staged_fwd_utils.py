# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host input chaining helpers for staged DeepSeek-V4 Pro forwards."""

from dataclasses import replace

import torch
from golden import TensorSpec


def override_inputs(specs, values, *, tracked_names=(), rng_after_init=None):
    """Replace selected inputs while preserving downstream fixture RNG state."""
    tensor_specs = {spec.name: spec for spec in specs if isinstance(spec, TensorSpec)}
    tracked_names = frozenset(tracked_names)
    rng_after_init = {} if rng_after_init is None else rng_after_init

    missing_tracked = tracked_names.difference(tensor_specs)
    if missing_tracked:
        raise ValueError(f"cannot track missing tensor specs: {sorted(missing_tracked)}")

    for name, value in values.items():
        spec = tensor_specs.get(name)
        if spec is None:
            raise ValueError(f"cannot override missing tensor spec {name!r}")
        if list(value.shape) != list(spec.shape) or value.dtype != spec.dtype:
            raise ValueError(
                f"{name} override has shape/dtype {list(value.shape)}/{value.dtype}; "
                f"expected {spec.shape}/{spec.dtype}"
            )
        if name in tracked_names:
            if name not in rng_after_init:
                raise ValueError(f"missing recorded RNG state for tracked tensor {name!r}")
            rng_state = rng_after_init[name]

            def init_override(value=value, rng_state=rng_state):
                torch.random.set_rng_state(rng_state)
                return value

            spec.init_value = init_override
        else:
            spec.init_value = value

    for name in tracked_names.difference(values):
        spec = tensor_specs[name]
        source_spec = replace(spec)
        has_checkpoint = name in rng_after_init
        checkpoint = rng_after_init.get(name)

        def init_and_checkpoint(
            name=name,
            source_spec=source_spec,
            has_checkpoint=has_checkpoint,
            checkpoint=checkpoint,
        ):
            value = source_spec.create_tensor()
            if has_checkpoint:
                torch.random.set_rng_state(checkpoint)
            else:
                rng_after_init[name] = torch.random.get_rng_state().clone()
            return value

        spec.init_value = init_and_checkpoint
