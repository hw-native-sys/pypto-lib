# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for the DeepSeek-V4 Pro model."""

import ast
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch

from models.deepseek_v4_pro.synthetic_token_loop import (
    _check_sample,
    _create_persistent_worker,
    _prefill_prompt_row,
    _require_runtime_scalar,
    _require_scalar,
)

MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_pro"


def _load_pro_utils():
    previous_config = sys.modules.pop("config", None)
    sys.path.insert(0, str(MODEL_DIR))
    try:
        module_path = MODEL_DIR / "utils.py"
        spec = importlib.util.spec_from_file_location(
            "deepseek_v4_pro_utils_contract",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(MODEL_DIR))
        sys.modules.pop("config", None)
        if previous_config is not None:
            sys.modules["config"] = previous_config
    return module


mx = _load_pro_utils()


def _tree(name):
    return ast.parse((MODEL_DIR / name).read_text())


def _function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_all_moe_callers_match_the_typed_window_abi():
    moe_args = _function(_tree("moe.py"), "moe").args.args
    moe_arity = len(moe_args)
    moe_arg_names = [arg.arg for arg in moe_args]
    window_names = (
        "recv_meta", "recv_x", "recv_scale", "recv_aux", "recv_route",
        "arrived", "data_arrived", "routed_y_buf", "combine_arrived", "consumed",
    )
    calls = []
    for path in MODEL_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "moe":
                calls.append((path.name, node.lineno, node))

    assert calls
    assert all(len(node.args) == moe_arity and not node.keywords for _, _, node in calls), [
        (name, line, len(node.args), len(node.keywords)) for name, line, node in calls
    ]
    for name, line, node in calls:
        for window_name in window_names:
            arg = node.args[moe_arg_names.index(window_name)]
            assert isinstance(arg, ast.Name) and arg.id == window_name, (
                name, line, window_name, ast.unparse(arg)
            )


def test_mx_dispatch_uses_byte_windows_and_restores_fp8_tiles():
    source = (MODEL_DIR / "moe.py").read_text()

    assert "recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8]" in source
    assert "recv_scale: pld.DistributedTensor[[N_LOCAL * RECV_MAX, K_SCALE], pl.UINT8]" in source
    assert "raw_row_i8 = pl.reinterpret_view(raw_row, pl.INT8)" in source
    assert "recv_x_mx = pl.reinterpret_view(recv_x_raw, pl.FP8E4M3FN)" in source
    assert "recv_scale_mx = pl.reinterpret_view(recv_scale_raw, pl.FP8E8M0)" in source


def test_moe_readiness_uses_unique_padded_set_epoch_slots():
    source = (MODEL_DIR / "moe.py").read_text()
    tree = ast.parse(source)
    signal_pad = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SIGNAL_PAD"
            for target in node.targets
        )
    )

    assert isinstance(signal_pad.value, ast.Constant) and signal_pad.value.value == 128
    assert "NotifyOp.AtomicAdd" not in source
    assert source.count("NotifyOp.Set") == 4
    assert source.count("WaitCmp.Ge") == 4
    assert "offsets=[my_rank, loc_e, 0]" in source
    assert "offsets=[src, loc_e, 0]" in source
    assert "offsets=[my_rank, e, 0]" in source
    assert "offsets=[src, e, 0]" in source


def test_fixed_epoch_entrypoints_quiesce_before_local_reset():
    for name in ("prefill_layer.py", "decode_mtp.py", "prefill_mtp.py"):
        source = (MODEL_DIR / name).read_text()
        retire_start = source.index('name_hint="moe_signal_retire"')
        retire = source[retire_start:]

        assert "NotifyOp.AtomicAdd" not in retire
        assert "signal=consumed" in retire
        assert retire.index("signal=consumed") < retire.index("target=arrived")
        assert "target=consumed" in retire
        assert "target=data_arrived" in retire
        assert "target=combine_arrived" in retire
        assert "NotifyOp.Set" in retire


def test_decode_layer_cache_specs_match_static_inout_abi():
    tree = _tree("decode_layer.py")
    host = _function(tree, "l3_decode_layer")
    inout_names = {
        arg.arg for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.InOut[")
    }
    out_names = {
        arg.arg for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.Out[")
    }
    build_specs = _function(tree, "build_tensor_specs")
    cache_assignment = next(
        node for node in ast.walk(build_specs)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "mutable_cache_names" for target in node.targets)
    )

    assert inout_names == ast.literal_eval(cache_assignment.value)
    assert out_names == {"x_next"}
    # Direction is stamped from the compiled artifact, so the spec builder must
    # not re-declare it.
    assert "is_output" not in ast.unparse(build_specs)


def test_prefill_mtp_ranked_specs_preserve_inout_direction():
    tree = _tree("prefill_mtp.py")
    host = _function(tree, "l3_mtp_prefill_fwd")
    inout_names = {
        arg.arg for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.InOut[")
    }
    out_names = {
        arg.arg for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.Out[")
    }

    assert inout_names == {"kv_cache"}
    assert out_names == {"hidden_out", "pre_hc_hidden_out"}
    assert "is_output" not in ast.unparse(_function(tree, "_ranked"))

    main = _function(tree, "main")
    cache_comparators = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "mapped_pool_ratio_reldiff"
    ]
    assert len(cache_comparators) == 1
    comparator = cache_comparators[0]
    assert ast.literal_eval(comparator.args[0]) == "ori_slot_mapping"
    keywords = {keyword.arg: keyword.value for keyword in comparator.keywords}
    assert ast.unparse(keywords["mapping_shape"]) == "(N_RANKS, T)"
    assert ast.unparse(keywords["block_size"]) == "BLOCK_SIZE"
    assert ast.literal_eval(keywords["leading_rank_axis"]) is True
    assert ast.literal_eval(keywords["pool_name"]) == "kv_cache"
    assert ast.literal_eval(keywords["diff_thd"]) == 0.01
    assert ast.literal_eval(keywords["pct_thd"]) == 0.05


DECODE_ROPE_MODULES = (
    "decode_compressor_ratio4.py",
    "decode_compressor_ratio128.py",
    "decode_indexer.py",
    "decode_indexer_compressor.py",
    "decode_sparse_attn.py",
    "decode_sparse_attn_hca.py",
    "decode_sparse_attn_swa.py",
    "qkv_proj_rope.py",
)


def _is_pl_gather(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "gather":
        return False
    owner = call.func.value
    return (isinstance(owner, ast.Name) and owner.id == "pl") or (
        isinstance(owner, ast.Attribute)
        and owner.attr == "tensor"
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "pl"
    )


def test_decode_rope_contract_recognizes_gather_spellings():
    for expression in ("pl.gather(x, index=i)", "pl.tensor.gather(x, index=i)"):
        call = ast.parse(expression, mode="eval").body
        assert isinstance(call, ast.Call)
        assert _is_pl_gather(call)


def test_decode_rope_permutations_use_mask_gather_scatter_only():
    for name in DECODE_ROPE_MODULES:
        source = (MODEL_DIR / name).read_text()
        tree = ast.parse(source)
        indexed_gathers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and _is_pl_gather(node)
            and any(keyword.arg == "index" for keyword in node.keywords)
        ]

        assert not indexed_gathers, (
            name,
            [node.lineno for node in indexed_gathers],
        )
        assert "MaskPattern.P0101" in source
        assert "MaskPattern.P1010" in source
        assert "pl.tensor.scatter" in source


def _pack_checkpoint_nibbles(indices_nk):
    low = indices_nk[..., 0::2] & 0x0F
    high = indices_nk[..., 1::2] & 0x0F
    return (low | (high << 4)).to(torch.uint8)


def test_mxfp4_bridge_preserves_all_nibble_codes_and_signed_zero():
    indices = (
        torch.arange(16, dtype=torch.uint8)
        .repeat(4)
        .reshape(1, 64)
        .repeat(16, 1)
    )
    packed = _pack_checkpoint_nibbles(indices)
    scale = torch.full((16, 2), 127, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )

    data, packed_scale = mx.mxfp4_to_mxfp8_weight_kn(packed, scale)

    assert data.shape == (64, 16)
    assert data.contiguous().view(torch.uint8)[:, 0].tolist() == mx.NIBBLE_LUT * 4
    assert data.contiguous().view(torch.uint8)[8, 0].item() == 0x80
    assert torch.all(mx.unpack_b_scale(packed_scale.view(torch.uint8)) == 127)


def test_mxfp4_bridge_matches_checkpoint_dequantization_for_w13_and_w2():
    for n, k in ((32, 64), (64, 128)):
        indices = torch.arange(n * k, dtype=torch.int64).reshape(n, k) % 16
        packed = _pack_checkpoint_nibbles(indices)
        scale_codes = 100 + (
            torch.arange(n * (k // 32), dtype=torch.uint8).reshape(n, k // 32)
            % 40
        )

        data_kn, packed_scale = mx.mxfp4_to_mxfp8_weight_kn(
            packed,
            scale_codes.view(torch.float8_e8m0fnu),
        )
        logical_scale = mx.unpack_b_scale(packed_scale.view(torch.uint8))

        values_nk = mx.nibble_indices_to_fp8(indices).float()
        expected = values_nk * mx.e8m0_codes_to_fp32(
            scale_codes
        ).repeat_interleave(32, dim=-1)
        actual = data_kn.float() * mx.e8m0_codes_to_fp32(
            logical_scale
        ).repeat_interleave(32, dim=0)
        torch.testing.assert_close(actual, expected.transpose(0, 1), rtol=0, atol=0)
        assert torch.equal(logical_scale, scale_codes.transpose(0, 1))


def test_mxfp4_bridge_keeps_leading_expert_dimensions_independent():
    indices = (
        torch.arange(2 * 16 * 64, dtype=torch.int64).reshape(2, 16, 64) % 16
    )
    packed = _pack_checkpoint_nibbles(indices)
    scale_codes = torch.tensor(
        [[[121, 122]] * 16, [[137, 138]] * 16],
        dtype=torch.uint8,
    )

    data, packed_scale = mx.mxfp4_to_mxfp8_weight_kn(
        packed,
        scale_codes.view(torch.float8_e8m0fnu),
    )

    assert data.shape == (2, 64, 16)
    assert packed_scale.shape == (2, 2, 16)
    for expert in range(2):
        assert torch.equal(
            mx.unpack_b_scale(packed_scale[expert].view(torch.uint8)),
            scale_codes[expert].T,
        )


def test_mx_scale_pack_round_trips():
    a = torch.arange(32 * 4, dtype=torch.uint8).reshape(32, 4)
    b = torch.arange(4 * 32, dtype=torch.uint8).reshape(4, 32)
    batched_b = (
        torch.arange(3 * 4 * 32, dtype=torch.int32)
        .to(torch.uint8)
        .reshape(3, 4, 32)
    )
    assert torch.equal(mx.unpack_a_scale(mx.pack_a_scale(a)), a)
    assert torch.equal(mx.unpack_b_scale(mx.pack_b_scale(b)), b)
    assert torch.equal(
        mx.unpack_b_scale_batched(mx.pack_b_scale_batched(batched_b)),
        batched_b,
    )


def test_host_weight_quant_returns_cube_layout_and_packed_e8m0_scale():
    weight_nk = torch.linspace(-2.0, 2.0, 32 * 64).reshape(32, 64)

    data_kn, packed_scale = mx.host_quant_mxfp8_weight_kn(weight_nk)

    logical_scale = mx.unpack_b_scale(packed_scale.view(torch.uint8))
    restored = data_kn.float() * mx.e8m0_codes_to_fp32(
        logical_scale
    ).repeat_interleave(mx.MX_GROUP, dim=0)
    expected_data, expected_scale = mx.host_quant_mxfp8(
        weight_nk,
        return_e8m0=True,
    )
    expected = expected_data.float() * mx.e8m0_codes_to_fp32(
        expected_scale.view(torch.uint8)
    ).repeat_interleave(mx.MX_GROUP, dim=-1)
    assert data_kn.shape == (64, 32)
    assert packed_scale.shape == (2, 32)
    torch.testing.assert_close(restored, expected.T, rtol=0, atol=0)


def test_host_quant_uses_ascend_ocp_shared_exponents():
    x = torch.zeros(1, 160)
    for group, value in enumerate((0.0, 0.99, 1.0, 1.99, 2.0)):
        x[:, group * mx.MX_GROUP : (group + 1) * mx.MX_GROUP] = value

    _, scale = mx.host_quant_mxfp8(x, return_e8m0=True)

    assert scale.view(torch.uint8).tolist() == [[0, 118, 119, 119, 120]]


class _Compiled:
    def __init__(self, *infos, output_dir=None):
        self._infos = infos
        self.output_dir = output_dir

    def _get_metadata(self):
        return self._infos, None, None


def _info(name, *, shape=None, dtype="i32"):
    return types.SimpleNamespace(name=name, shape=shape, dtype=dtype)


def _dtype_module():
    module = types.ModuleType("pypto.ir.compiled_program")
    module._to_torch_dtype = lambda dtype: {
        "i32": torch.int32,
        "i64": torch.int64,
    }[dtype]
    return module


def test_require_scalar_accepts_terminal_ssa_name():
    compiled = _Compiled(_info("moe_epoch_base__ssa_v0"))

    with patch.dict(sys.modules, {"pypto.ir.compiled_program": _dtype_module()}):
        _require_scalar(compiled, "moe_epoch_base", "decode", torch.int32)


def test_require_scalar_rejects_old_artifact():
    compiled = _Compiled(_info("num_tokens"))

    with pytest.raises(ValueError, match="recompile"):
        _require_scalar(compiled, "moe_epoch_base", "decode", torch.int32)


@pytest.mark.parametrize(
    ("info", "error"),
    [
        (_info("moe_epoch_base", shape=[1]), ValueError),
        (_info("moe_epoch_base", dtype="i64"), TypeError),
    ],
)
def test_require_scalar_rejects_wrong_abi(info, error):
    with (
        patch.dict(sys.modules, {"pypto.ir.compiled_program": _dtype_module()}),
        pytest.raises(error),
    ):
        _require_scalar(_Compiled(info), "moe_epoch_base", "decode", torch.int32)


def _write_host_orch(tmp_path, scalar_argument):
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    host_orch = orchestration / "host_orch.py"
    host_orch.write_text(
        "def run(tensors, task_args):\n"
        '    epoch = tensors["moe_epoch_base__ssa_v0"]\n'
        f"    task_args.add_scalar({scalar_argument})\n"
    )


def test_require_runtime_scalar_accepts_forwarded_host_argument(tmp_path):
    _write_host_orch(tmp_path, "epoch")
    compiled = _Compiled(
        _info("moe_epoch_base__ssa_v0"),
        output_dir=tmp_path,
    )

    with patch.dict(sys.modules, {"pypto.ir.compiled_program": _dtype_module()}):
        _require_runtime_scalar(
            compiled,
            "moe_epoch_base",
            "decode",
            torch.int32,
        )


def test_require_runtime_scalar_rejects_constant_specialization(tmp_path):
    _write_host_orch(tmp_path, "0")
    compiled = _Compiled(
        _info("moe_epoch_base__ssa_v0"),
        output_dir=tmp_path,
    )

    with (
        patch.dict(sys.modules, {"pypto.ir.compiled_program": _dtype_module()}),
        pytest.raises(ValueError, match="constant-specialized.*compile_runtime=True"),
    ):
        _require_runtime_scalar(
            compiled,
            "moe_epoch_base",
            "decode",
            torch.int32,
        )


def test_create_worker_retains_persistent_windows():
    captured = {}

    def worker_type(compiled, **kwargs):
        captured["compiled"] = compiled
        captured["kwargs"] = kwargs
        return "worker"

    programs = [object(), object()]
    config = object()
    inherited = [torch.zeros(1)]
    worker = _create_persistent_worker(worker_type, programs, config, inherited)

    assert worker == "worker"
    assert captured == {
        "compiled": programs,
        "kwargs": {
            "config": config,
            "persistent": True,
            "reset_persistent_windows": False,
            "inherited_host_tensors": inherited,
        },
    }


def test_synthetic_prompt_uses_requested_active_length():
    ids, prompt_len = _prefill_prompt_row(7, None, 5)

    assert prompt_len == 5
    assert ids[:5].tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(ids[5:], torch.zeros_like(ids[5:]))


def test_check_sample_dumps_rank_divergence(tmp_path, monkeypatch, capsys):
    tensors = {
        "input_ids": torch.tensor([[1], [1]], dtype=torch.int64),
        "logit_row_indices": torch.zeros((2, 1), dtype=torch.int32),
        "pre_hc_hidden_out": torch.tensor(
            [[[[1.0, 2.0]]], [[[1.0, 3.0]]]]
        ),
        "hidden_out": torch.tensor([[[1.0, 2.0]], [[1.0, 3.0]]]),
        "logits": torch.tensor(
            [[[0.0, 3.0, 1.0, 2.0]], [[0.0, 1.0, 4.0, 2.0]]]
        ),
        "sampled_ids": torch.tensor([[[1]], [[2]]], dtype=torch.int32),
    }
    monkeypatch.setenv("DSV4_FAIL_DUMP_DIR", str(tmp_path))

    with pytest.raises(AssertionError, match="ranks sampled different tokens"):
        _check_sample(tensors, vocab_size=4, stage="decode[0]@6")

    output = capsys.readouterr().out
    assert "active_rows=[0, 0]" in output
    assert "pre_hc_hidden_out" in output
    assert "max_abs_diff_rank0=[0.0, 1.0]" in output
    assert "logits_top2: ids=[[1, 3], [2, 3]]" in output
    dump = torch.load(tmp_path / "failure_decode_0_6.pt", weights_only=True)
    assert set(dump) == set(tensors)
    assert torch.equal(dump["sampled_ids"], tensors["sampled_ids"])
