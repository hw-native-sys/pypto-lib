# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import pytest
import torch

from golden import TensorSpec


MODULE_PATH = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_pro" / "staged_fwd_utils.py"
MODULE_SPEC = importlib.util.spec_from_file_location("staged_fwd_utils", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
override_inputs = MODULE.override_inputs


def _make_specs(history_draws=0):
    def init_history():
        torch.randn(history_draws)
        return torch.zeros(4)

    return [
        TensorSpec("state", [4], torch.float32, init_value=torch.randn),
        TensorSpec("history", [4], torch.float32, init_value=init_history),
        TensorSpec("weight", [4], torch.float32, init_value=torch.randn),
    ]


def test_override_replays_rng_after_tracked_state_initializer():
    with torch.random.fork_rng():
        rng_after_init = {}
        torch.manual_seed(17)
        first_specs = _make_specs()
        override_inputs(first_specs, {}, tracked_names={"state"}, rng_after_init=rng_after_init)
        first = {spec.name: spec.create_tensor() for spec in first_specs}

        torch.manual_seed(17)
        second_specs = _make_specs()
        override_inputs(
            second_specs,
            {"state": torch.ones(4)},
            tracked_names={"state"},
            rng_after_init=rng_after_init,
        )
        second = {spec.name: spec.create_tensor() for spec in second_specs}

    assert torch.equal(second["state"], torch.ones(4))
    assert torch.equal(second["weight"], first["weight"])


def test_override_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape/dtype"):
        override_inputs(_make_specs(), {"state": torch.ones(3)})


def test_override_restores_checkpoint_after_start_dependent_initializer():
    with torch.random.fork_rng():
        rng_after_init = {}
        tracked_names = {"state", "history"}
        torch.manual_seed(29)
        first_specs = _make_specs(history_draws=1)
        override_inputs(first_specs, {}, tracked_names=tracked_names, rng_after_init=rng_after_init)
        first = {spec.name: spec.create_tensor() for spec in first_specs}

        torch.manual_seed(29)
        second_specs = _make_specs(history_draws=11)
        override_inputs(
            second_specs,
            {"state": torch.ones(4)},
            tracked_names=tracked_names,
            rng_after_init=rng_after_init,
        )
        second = {spec.name: spec.create_tensor() for spec in second_specs}

    assert torch.equal(second["weight"], first["weight"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
