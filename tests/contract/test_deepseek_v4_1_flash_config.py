# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 checkpoint configuration consumed by serving and model kernels."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def config(monkeypatch):
    # Configuration metadata only needs named placeholders for dynamic shapes.
    language = ModuleType("pypto.language")
    language.dynamic = lambda name: name
    pypto = ModuleType("pypto")
    pypto.language = language
    monkeypatch.setitem(sys.modules, "pypto", pypto)
    monkeypatch.setitem(sys.modules, "pypto.language", language)
    monkeypatch.setattr(sys, "argv", ["config-contract", "--tp", "1", "--ep", "4"])
    path = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_1_flash" / "config.py"
    spec = importlib.util.spec_from_file_location("_v41_checkpoint_config", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_keeps_rms_and_hc_epsilon_distinct(config):
    assert config.FLASH.rms_norm_eps == 1e-6
    assert config.FLASH.hc_eps == 1e-6
    assert config.FLASH.hc_sinkhorn_iters == 20


def test_checkpoint_keeps_backbone_and_drafter_parameters_distinct(config):
    assert (config.FLASH.n_routed_experts, config.FLASH.num_experts_per_tok) == (384, 6)
    assert (config.DSPARK_N_ROUTED_EXPERTS, config.DSPARK_NUM_EXPERTS_PER_TOK) == (128, 3)
    assert config.FLASH.num_hidden_layers == 40
    assert config.FLASH.compress_ratios == (0, 0) + (2,) * 18 + (1,) * 20
    assert config.DSPARK_DRAFT_LAYERS == 3
    assert config.DSPARK_SPEC_TOKENS == 5
    assert config.DSPARK_TARGET_LAYER_IDS == (37, 38, 39)
    assert config.DSPARK_NOISE_TOKEN_ID == 128799
    assert config.DSPARK_MARKOV_RANK == 256
