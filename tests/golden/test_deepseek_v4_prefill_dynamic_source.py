# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source contracts for the Attention-first dynamic-token migration."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "deepseek" / "v4-flash"


def _tree(filename: str) -> ast.Module:
    return ast.parse((MODEL / filename).read_text(encoding="utf-8"))


def _function(filename: str, name: str) -> ast.FunctionDef:
    return next(
        node for node in _tree(filename).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_attention_contract_uses_physical_token_shapes_below_compatibility_boundary() -> None:
    contracts = {
        "prefill_attention_swa.py": {"x_hc", "ori_slot_mapping", "position_ids", "x_out"},
        "prefill_attention_hca.py": {
            "x_hc",
            "ori_slot_mapping",
            "position_ids",
            "cmp_slot_mapping",
            "state_slot_mapping",
            "x_out",
        },
        "prefill_attention_csa.py": {
            "x_hc",
            "ori_slot_mapping",
            "position_ids",
            "cmp_slot_mapping",
            "idx_slot_mapping",
            "state_slot_mapping",
            "inner_state_slot_mapping",
            "x_out",
        },
    }
    for filename, token_params in contracts.items():
        name = filename.removesuffix(".py")
        function = _function(filename, name)
        params = tuple(arg.arg for arg in function.args.args)
        source = ast.unparse(function)
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.arg in token_params
        }

        assert "num_tokens" not in params
        assert set(annotations) == token_params
        assert annotations["x_out"] == "pl.Out[pl.Tensor]"
        assert all(annotation == "pl.Tensor" for name, annotation in annotations.items() if name != "x_out")
        assert "_T_DYN" not in source
        assert "token_count = pl.cast(num_tokens, pl.INDEX)" not in source
        assert "num_tokens = pl.tensor.dim(x_hc, 0)" in source


def test_attention_composition_boundaries_use_fixed_capacity_backing() -> None:
    common_storage = {
        "q": "[T, H, HEAD_DIM]",
        "kv": "[T, HEAD_DIM]",
        "qr": "[T, Q_LORA]",
        "qr_scale": "[T, 1]",
    }
    for filename in (
        "prefill_attention_swa.py",
        "prefill_attention_hca.py",
        "prefill_attention_csa.py",
    ):
        source = ast.unparse(_function(filename, filename.removesuffix(".py")))
        for name, storage_shape in common_storage.items():
            assert f"{name}_storage = pl.create_tensor({storage_shape}" in source
            assert f"{name} = pl.slice({name}_storage, [num_tokens" in source

    csa_source = ast.unparse(_function("prefill_attention_csa.py", "prefill_attention_csa"))
    for name, storage_shape in {
        "idx_cos": "[T, HALF_ROPE]",
        "idx_sin": "[T, HALF_ROPE]",
        "cmp_topk_indices": "[T, IDX_TOPK]",
        "idx_score_unused": "[T, INDEXER_SCORE_CAP]",
    }.items():
        assert f"{name}_storage = pl.create_tensor({storage_shape}" in csa_source
        assert f"{name} = pl.slice({name}_storage, [num_tokens" in csa_source


def test_dynamic_token_symbols_are_module_local() -> None:
    symbols = {
        "hc_pre.py": "T_DYN",
        "hc_post.py": "T_DYN",
        "rmsnorm.py": "T_DYN",
        "qkv_proj_rope.py": "T_DYN",
        "prefill_compressor_ratio4.py": "T_DYN",
        "prefill_compressor_ratio128.py": "T_DYN",
        "prefill_indexer.py": "T_DYN",
        "prefill_indexer_compressor.py": "T_DYN",
        "prefill_sparse_attn.py": "T_DYN",
        "prefill_attention_swa.py": "SWA_T_DYN",
        "prefill_attention_hca.py": "HCA_T_DYN",
        "prefill_attention_csa.py": "CSA_T_DYN",
    }
    dynamic_names = set()
    for filename, symbol in symbols.items():
        source = (MODEL / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)

        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == symbol
        )
        assert isinstance(assignment.value, ast.Call)
        assert ast.unparse(assignment.value.func) == "pl.dynamic"
        module_dynamics = {
            node.value.args[0].value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "pl.dynamic"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Constant)
        }
        assert len(module_dynamics) == 1, filename
        assert dynamic_names.isdisjoint(module_dynamics), filename
        dynamic_names.update(module_dynamics)

    for path in MODEL.glob("*.py"):
        assert "dynamic_shapes" not in path.read_text(encoding="utf-8"), path.name


def test_packed_prefill_parent_owns_its_dynamic_symbol() -> None:
    contract = _tree("prefill_dynamic.py")
    assignment = next(
        node
        for node in contract.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "PREFILL_TOKENS_DYN"
    )
    assert ast.unparse(assignment.value) == "pl.dynamic('PREFILL_TOKENS_DYN')"

    layer = _tree("prefill_layer.py")
    imports = [
        node
        for node in layer.body
        if isinstance(node, ast.ImportFrom) and node.module == "prefill_dynamic"
    ]
    assert len(imports) == 1
    assert [(alias.name, alias.asname) for alias in imports[0].names] == [("PREFILL_TOKENS_DYN", None)]

    for filename, symbol in {
        "prefill_attention_swa.py": "SWA_T_DYN",
        "prefill_attention_hca.py": "HCA_T_DYN",
        "prefill_attention_csa.py": "CSA_T_DYN",
    }.items():
        tree = _tree(filename)
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "prefill_dynamic"
            for node in tree.body
        )
        core = _function(filename, filename.removesuffix(".py"))
        test_entry = _function(filename, f"{filename.removesuffix('.py')}_test")
        assert symbol not in ast.unparse(core)
        assert symbol in ast.unparse(test_entry)


def test_all_attention_call_arities_match_leaf_contracts() -> None:
    contracts = {
        name: len(_function(filename, name).args.args)
        for filename, name in (
            ("prefill_attention_swa.py", "prefill_attention_swa"),
            ("prefill_attention_hca.py", "prefill_attention_hca"),
            ("prefill_attention_csa.py", "prefill_attention_csa"),
        )
    }
    calls = []
    for path in MODEL.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in contracts:
                calls.append((path.name, node))
                assert len(node.args) == contracts[node.func.id], f"{path.name}:{node.lineno}"
    assert len(calls) == 12


def test_fixed_capacity_parents_isolate_dynamic_attention_outputs_from_moe() -> None:
    for filename in ("prefill_fwd.py", "prefill_mtp.py"):
        tree = _tree(filename)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in {"prefill_attention_swa", "prefill_attention_hca", "prefill_attention_csa"}:
                continue
            names = {arg.id for arg in node.args if isinstance(arg, ast.Name)}
            assert not names.intersection({"x_hc", "projected", "hidden", "hidden_mid", "x_attn", "nt", "num_tokens"})
            assert any(name.endswith("_valid") for name in names)

    layer_calls = [
        node
        for node in ast.walk(_tree("prefill_layer.py"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"prefill_attention_swa", "prefill_attention_hca", "prefill_attention_csa"}
    ]
    assert len(layer_calls) == 3
    for call in layer_calls:
        assert ast.unparse(call.args[0]) == "x_hc_tile"
        assert ast.unparse(call.args[-1]) == "x_attn_valid"

    output_pairs = {
        "prefill_fwd.py": (
            ("x_attn0_valid", "x_attn0_storage", "token_count"),
            ("x_attn1_valid", "x_attn1_storage", "token_count"),
            ("x_attn_csa_valid", "x_attn_csa_storage", "token_count"),
            ("x_attn_hca_valid", "x_attn_hca_storage", "token_count"),
            ("x_attn_last_valid", "x_attn_last_storage", "token_count"),
        ),
        "prefill_mtp.py": (("x_attn_valid", "x_attn_storage", "token_count"),),
        "prefill_layer.py": (("x_attn_valid", "x_attn_storage", "valid_tok"),),
    }
    for filename, pairs in output_pairs.items():
        source = (MODEL / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for dynamic_name, storage_name, token_extent in pairs:
            storage_extent = "TOK_TILE" if filename == "prefill_layer.py" else "T"
            assert f"{storage_name} = pl.create_tensor([{storage_extent}, HC_MULT, D], dtype=pl.FP32)" in source
            assert (
                f"{dynamic_name} = pl.slice({storage_name}, [{token_extent}, HC_MULT, D], [0, 0, 0])"
                in source
            )
            assert f"pl.assemble({storage_name}, {dynamic_name}" not in source

            assert any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"prefill_attention_swa", "prefill_attention_hca", "prefill_attention_csa"}
                and isinstance(node.args[-1], ast.Name)
                and node.args[-1].id == dynamic_name
                for node in ast.walk(tree)
            )
            assert any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "moe"
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == storage_name
                for node in ast.walk(tree)
            )


def test_full_prefill_sizes_both_active_runtime_rings() -> None:
    source = (MODEL / "prefill_fwd.py").read_text(encoding="utf-8")

    assert "PREFILL_RING_HEAP = (0, 512 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 0)" in source


def test_full_prefill_reuses_fixed_capacity_buffers_across_layer_pairs() -> None:
    function = _function("prefill_fwd.py", "prefill_fwd")
    layer_loop = next(
        node
        for node in function.body
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "loop_i"
    )
    storage_names = {"x_attn_csa_storage", "hidden_mid", "x_attn_hca_storage"}

    def assigned_name(node: ast.AST):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return node.target.id
        return None

    assignments = [
        node
        for node in function.body
        if assigned_name(node) in storage_names
    ]
    assert {assigned_name(node) for node in assignments} == storage_names
    assert all(node.lineno < layer_loop.lineno for node in assignments)
    assert not storage_names.intersection(
        assigned_name(node) for node in ast.walk(layer_loop)
    )


def test_mtp_golden_limits_dynamic_attention_to_active_prefix() -> None:
    source = ast.unparse(_function("prefill_mtp.py", "golden_mtp_prefill_fwd"))

    assert "'x_hc': projected[rank, :num_tokens]" in source
    assert "'ori_slot_mapping': tensors['ori_slot_mapping'][rank, :num_tokens]" in source
    assert "'position_ids': tensors['position_ids'][rank, :num_tokens]" in source
    assert "'x_out': x_attn[rank, :num_tokens]" in source


def test_mtp_projection_and_attention_have_independent_runtime_scopes() -> None:
    function = _function("prefill_mtp.py", "mtp_prefill_fwd")
    decorator = next(node for node in function.decorator_list if isinstance(node, ast.Call))
    auto_scope = next(keyword for keyword in decorator.keywords if keyword.arg == "auto_scope")
    assert isinstance(auto_scope.value, ast.Constant) and auto_scope.value.value is False

    scopes = [node for node in function.body if isinstance(node, ast.With)]
    assert len(scopes) == 2
    scoped_calls = [
        {
            node.func.id
            for node in ast.walk(scope)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for scope in scopes
    ]
    assert "mtp_projection" in scoped_calls[0]
    assert "prefill_attention_swa" in scoped_calls[1]
    assert all("moe" not in calls for calls in scoped_calls)


def test_shared_hc_pre_has_generic_token_contract() -> None:
    token_params = {"x", "x_mixed", "post", "comb"}
    for name in ("_hc_pre_syncall", "_hc_pre_separate"):
        function = _function("hc_pre.py", name)
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.arg in token_params
        }
        assert set(annotations) == token_params
        assert annotations == {name: "pl.Tensor" for name in annotations}

    bind_function = _function("hc_pre.py", "_bind_hc_pre")
    wrappers = [
        node for node in ast.walk(bind_function) if isinstance(node, ast.FunctionDef) and node.name == "hc_pre"
    ]
    assert len(wrappers) == 2
    for wrapper in wrappers:
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in wrapper.args.args
            if arg.arg in token_params
        }
        assert set(annotations) == token_params
        assert annotations == {name: "pl.Tensor" for name in annotations}

    test_entry = _function("hc_pre.py", "hc_pre_test")
    assert "T_DYN" in ast.unparse(test_entry)


def test_hc_pre_rms_preserves_aligned_tile_precision_path() -> None:
    source = ast.unparse(_function("hc_pre.py", "_hc_pre_separate"))

    assert "if valid_rows == T_TILE:" in source
    assert "x_chunk_full = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]" in source
    assert "x_chunk_tail = pl.slice(x_flat, [T_TILE, RMS_K_CHUNK], [t0, k0], valid_shape=[valid_rows, RMS_K_CHUNK])" in source


def test_shared_hc_post_has_generic_token_contract() -> None:
    token_params = {"x", "residual", "post", "comb", "y"}
    for name in ("hc_post", "hc_post_prefill"):
        function = _function("hc_post.py", name)
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.arg in token_params
        }
        assert set(annotations) == token_params
        assert annotations == {
            "x": "pl.Tensor",
            "residual": "pl.Tensor",
            "post": "pl.Tensor",
            "comb": "pl.Tensor",
            "y": "pl.Out[pl.Tensor]",
        }

    test_entry = _function("hc_post.py", "hc_post_test")
    assert "T_DYN" in ast.unparse(test_entry)


def test_shared_attention_leaves_have_generic_token_contracts() -> None:
    contracts = {
        ("rmsnorm.py", "rms_norm"): {
            "x": "pl.Tensor",
            "x_normed": "pl.Tensor",
        },
        ("qkv_proj_rope.py", "materialize_rope_rows"): {
            "position_ids": "pl.Tensor",
            "rope_cos_t": "pl.Tensor",
            "rope_sin_t": "pl.Tensor",
        },
        ("qkv_proj_rope.py", "qkv_proj_rope"): {
            "x": "pl.Tensor",
            "rope_cos": "pl.Tensor",
            "rope_sin": "pl.Tensor",
            "q": "pl.Tensor",
            "kv": "pl.Tensor",
            "qr": "pl.Tensor",
            "qr_scale": "pl.Tensor",
        },
        ("prefill_sparse_attn.py", "prefill_sparse_attn"): {
            "q_in": "pl.Tensor",
            "swa_indices_in": "pl.Tensor",
            "cmp_indices_in": "pl.Tensor",
            "freqs_cos_in": "pl.Tensor",
            "freqs_sin_in": "pl.Tensor",
            "attn_out_dyn": "pl.Out[pl.Tensor]",
        },
    }
    for (filename, name), expected in contracts.items():
        function = _function(filename, name)
        annotations = {
            arg.arg: ast.unparse(arg.annotation)
            for arg in function.args.args
            if arg.arg in expected
        }
        assert set(annotations) == set(expected)
        assert annotations == expected


def test_prefill_attention_call_boundaries_forward_runtime_shaped_tensors() -> None:
    common_calls = {
        "hc_pre": {0: "x_hc", 4: "x_mixed", 5: "post", 6: "comb"},
        "rms_norm": {0: "x_mixed", 2: "x_normed"},
        "materialize_rope_rows": {2: "position_ids", 3: "rope_cos_t", 4: "rope_sin_t"},
        "qkv_proj_rope": {
            0: "x_normed",
            5: "rope_cos_t",
            6: "rope_sin_t",
            9: "q",
            10: "kv",
            11: "qr",
            12: "qr_scale",
        },
        "prefill_sparse_attn": {
            0: "q",
            2: "swa_indices",
            5: "cmp_indices",
            7: "rope_cos_t",
            8: "rope_sin_t",
            12: "attn_out",
        },
        "hc_post_prefill": {
            0: "attn_out",
            1: "x_hc",
            2: "post",
            3: "comb",
            4: "x_out",
        },
    }
    per_file_calls = {
        "prefill_attention_swa.py": {
            "prefill_sparse_attn": {
                0: "q",
                2: "swa_indices",
                5: "cmp_indices_dummy",
                7: "rope_cos_t",
                8: "rope_sin_t",
                12: "attn_out",
            },
        },
        "prefill_attention_hca.py": {
            "prefill_compressor_ratio128": {
                0: "x_normed",
                10: "position_ids",
                11: "cmp_slot_mapping",
                12: "state_slot_mapping",
            },
        },
        "prefill_attention_csa.py": {
            "prefill_compressor_ratio4": {
                0: "x_normed",
                10: "position_ids",
                11: "cmp_slot_mapping",
                12: "state_slot_mapping",
            },
            "prefill_indexer": {
                0: "x_normed",
                1: "qr",
                2: "qr_scale",
                6: "idx_cos",
                7: "idx_sin",
                20: "idx_score_unused",
                21: "cmp_topk_indices",
                22: "position_ids",
                23: "idx_slot_mapping",
                24: "inner_state_slot_mapping",
            },
        },
    }
    for filename, extra_calls in per_file_calls.items():
        function = _function(filename, filename.removesuffix(".py"))
        target_calls = common_calls | extra_calls
        for name, expected_args in target_calls.items():
            calls = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
            ]
            assert len(calls) == 1
            call = calls[0]
            for index, expected in expected_args.items():
                assert ast.unparse(call.args[index]) == expected


def test_attention_children_do_not_accept_num_tokens() -> None:
    functions = {
        "prefill_compressor_ratio4.py": "prefill_compressor_ratio4",
        "prefill_compressor_ratio128.py": "prefill_compressor_ratio128",
        "prefill_indexer.py": "prefill_indexer",
        "prefill_indexer_compressor.py": "prefill_indexer_compressor",
        "prefill_sparse_attn.py": "prefill_sparse_attn",
    }
    for filename, name in functions.items():
        params = tuple(arg.arg for arg in _function(filename, name).args.args)
        assert "num_tokens" not in params, f"{filename}:{name}"


def test_indexer_golden_uses_physical_token_count() -> None:
    function = _function("prefill_indexer.py", "golden_prefill_indexer_core")

    assert not any(isinstance(node, ast.Name) and node.id == "T" for node in ast.walk(function))


def test_dynamic_compressor_projection_uses_static_incore_tiles() -> None:
    for filename, name in (
        ("prefill_compressor_ratio4.py", "prefill_compressor_ratio4"),
        ("prefill_compressor_ratio128.py", "prefill_compressor_ratio128"),
    ):
        source = ast.unparse(_function(filename, name))

        assert "pl.range(matmul_tokens // TOKEN_M_TILE)" in source
        assert "pl.create_tensor([TOKEN_M_TILE, OUT_TILE]" in source
        assert "pl.create_tensor([matmul_tokens, OUT_TILE]" not in source
        assert "valid_shape=[valid_rows, K_TILE]" in source


def test_dynamic_indexer_padding_does_not_materialize_full_hidden_tiles() -> None:
    for filename, name, pad_name in (
        ("prefill_indexer.py", "prefill_indexer", "prefill_indexer_dynamic_pad_x"),
        (
            "prefill_indexer_compressor.py",
            "prefill_indexer_compressor",
            "prefill_idx_compressor_dynamic_pad_x",
        ),
    ):
        function = _function(filename, name)
        source = ast.unparse(function)
        spmd_names = {
            keyword.value.value
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "pl.spmd"
            for keyword in node.keywords
            if keyword.arg == "name_hint" and isinstance(keyword.value, ast.Constant)
        }

        assert pad_name in spmd_names
        assert "x_row = pl.tile.full([1, D]" in source
        assert "pl.load(x_in, [pad_t, 0], [1, D]" in source
        assert "x[:, :] = pl.slice(x_in, [T, D]" not in source


def test_dynamic_leaf_fixtures_accept_physical_token_count() -> None:
    fixture_calls = {
        "prefill_indexer.py": "specs=build_tensor_specs(args.start_pos, args.num_tokens)",
        "prefill_indexer_compressor.py": "specs=build_tensor_specs(args.start_pos, args.num_tokens)",
        "prefill_sparse_attn.py": "specs=build_tensor_specs(args.compress_ratio, args.num_tokens)",
    }
    for filename, fixture_call in fixture_calls.items():
        source = (MODEL / filename).read_text(encoding="utf-8")

        assert 'parser.add_argument("--num-tokens", type=int, default=T' in source
        assert fixture_call in source


def test_qkv_fixture_accepts_physical_token_count_for_one_mode() -> None:
    source = (MODEL / "qkv_proj_rope.py").read_text(encoding="utf-8")

    assert '"--num-tokens"' in source
    assert 'if args.mode == "all"' in source
    assert 'B, S = 1, args.num_tokens' in source


def test_dynamic_output_stores_do_not_receive_tensor_slices() -> None:
    for filename in ("hc_pre.py", "prefill_indexer.py", "prefill_sparse_attn.py"):
        tree = ast.parse((MODEL / filename).read_text(encoding="utf-8"))
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            tensor_slice_names = {
                node.targets[0].id
                for node in ast.walk(function)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "pl.slice"
            }
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or ast.unparse(node.func) != "pl.store":
                    continue
                source = node.args[0]
                assert not (isinstance(source, ast.Call) and ast.unparse(source.func) == "pl.slice")
                assert not (isinstance(source, ast.Name) and source.id in tensor_slice_names)


def test_packed_layer_metadata_copies_only_valid_tail_rows() -> None:
    source = ast.unparse(_function("prefill_layer.py", "_packed_token_metadata"))

    assert "base + T" not in source
    assert source.count("base + valid") == 8


def test_hc_pre_materializes_post_as_a_vec_tile_before_store() -> None:
    source = ast.unparse(_function("hc_pre.py", "_hc_pre_separate"))
    assert "post_pad_store = pl.create_tensor" in source
    assert "post_pad_store = pl.assemble(post_pad_store, post_pad" in source
    assert "post_tile = pl.load(post_pad_store" in source
    assert "pl.store(post_tile" in source


def test_hc_pre_materializes_mixed_output_as_a_vec_tile_before_store() -> None:
    for name in ("_hc_pre_syncall", "_hc_pre_separate"):
        source = ast.unparse(_function("hc_pre.py", name))
        assert "x_mixed_pad_store = pl.create_tensor" in source
        assert "x_mixed_pad_store = pl.assemble(x_mixed_pad_store, y_bf16" in source
        assert "y_out = pl.load(x_mixed_pad_store" in source
        assert "pl.store(y_out" in source


def test_hc_pre_materializes_comb_rows_as_vec_tiles_before_store() -> None:
    for name in ("_hc_pre_syncall", "_hc_pre_separate"):
        source = ast.unparse(_function("hc_pre.py", name))
        assert "comb_pad_store = pl.create_tensor" in source
        for row in range(4):
            assert f"row{row}_out = pl.set_validshape(row{row}_cur, COMB_T_TILE, HC_MULT)" in source
            assert f"pl.store(row{row}_cur" in source
            assert f"row{row}_tail = pl.load(comb_pad_store" in source
            assert f"pl.store(row{row}_out" in source
            assert f"pl.store(row{row}_tail" in source


def test_dynamic_attention_tile_kernels_load_partial_rows_as_tiles() -> None:
    for filename, name in (
        ("rmsnorm.py", "rms_norm"),
        ("qkv_proj_rope.py", "qkv_proj_rope"),
    ):
        function = _function(filename, name)
        calls = {ast.unparse(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
        assert "pl.load" in calls
        assert "pl.slice" not in calls


def test_dynamic_attention_tile_reductions_provide_scratch_tiles() -> None:
    for filename, name in (
        ("rmsnorm.py", "rms_norm"),
        ("qkv_proj_rope.py", "qkv_proj_rope"),
    ):
        function = _function(filename, name)
        reductions = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and ast.unparse(node.func) in {"pl.row_sum", "pl.row_max"}
        ]
        assert reductions
        if filename == "qkv_proj_rope.py":
            tensor_reductions = [node for node in reductions if len(node.args) == 1]
            assert {ast.unparse(node) for node in tensor_reductions} == {
                "pl.row_sum(q_head_sq_full)",
            }
            assert all(len(node.args) == 2 for node in reductions if node not in tensor_reductions)
        else:
            assert all(len(node.args) == 2 for node in reductions)

    rms_source = ast.unparse(_function("rmsnorm.py", "rms_norm"))
    assert "x_sq_sum = pl.tile.full" in rms_source
    assert "row_reduce_tmp = pl.create_tile([T_TILE, D_TILE]" in rms_source
    assert "norm_w_input = pl.load(norm_w" in rms_source

    qkv_source = ast.unparse(_function("qkv_proj_rope.py", "qkv_proj_rope"))
    assert "pl.gather" in qkv_source
    assert qkv_source.count("pl.tile.gather") == 6
    for accumulator in ("q_acc", "col_acc", "kv_acc"):
        assert f"{accumulator} = pl.create_tile" in qkv_source
    assert "pl.assemble(qr_fp32, q_acc" not in qkv_source
    assert "pl.assemble(kv_fp32, kv_acc" not in qkv_source
    assert "pl.store(q_acc" in qkv_source
    assert "pl.store(col_acc" in qkv_source
    assert "pl.store(kv_acc" in qkv_source
    assert "pl.pipeline(0, Q_LORA // Q_PROJ_TILE, stage=1)" in qkv_source
    assert "pl.spmd(H * HEAD_DIM // QPROJ_MM_N_TILE, name_hint='qproj_matmul'" in qkv_source
    qproj_n_tile = next(
        node.value
        for node in _tree("qkv_proj_rope.py").body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "QPROJ_MM_N_TILE"
    )
    assert ast.literal_eval(qproj_n_tile) == 512
    qkv_function = _function("qkv_proj_rope.py", "qkv_proj_rope")
    loads = {
        node.targets[0].id: node.value
        for node in ast.walk(qkv_function)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "pl.load"
    }
    for name in ("q_x_chunk_bf16", "qr_i8_chunk", "kv_x_chunk_bf16"):
        assert all(keyword.arg != "valid_shapes" for keyword in loads[name].keywords)
    assert ast.unparse(loads["q_x_chunk_bf16"].args[0]) == "x_matmul"
    assert ast.unparse(loads["kv_x_chunk_bf16"].args[0]) == "x_matmul"
    assert "qkv_dynamic_pad_x" in qkv_source
    assert "with pl.at(level=pl.Level.CORE_GROUP, name_hint='qkv_dynamic_pad_x')" in qkv_source
    assert "pl.spmd(T_MAX, name_hint='qkv_dynamic_pad_x')" not in qkv_source
    assert "qr_i8_matmul[ts0:ts0 + QR_M_TILE" in qkv_source
    assert "dtype=pl.INT8, value=0" not in qkv_source
    assert "pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP16, value=0.0)" in qkv_source
    assert "qr_sum_tmp = pl.create_tile([T_TILE, Q_LORA_TILE]" in qkv_source
    assert "qr_max_tmp = pl.create_tile([T_TILE, Q_LORA_TILE]" in qkv_source
    assert "pl.range(1, Q_LORA // Q_LORA_TILE)" in qkv_source
    assert "pl.pipeline(Q_LORA // Q_LORA_TILE" not in qkv_source
    assert "qr_inv_rms_row = pl.tile.rsqrt" in qkv_source
    assert "tmp=qr_rsqrt_tmp" in qkv_source
    assert "qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])" in qkv_source
    assert "pl.mul(qr_tile_amax, 1.0 / INT8_SCALE_MAX)" not in qkv_source
    assert "q_head_reduce_tmp = pl.create_tile([Q_ROPE_T_TILE, HEAD_DIM]" in qkv_source
    assert "kv_reduce_tmp = pl.create_tile([KV_RMS_T_TILE, KV_TILE]" in qkv_source
    assert "pl.range(Q_ROPE_H_TILE)" in qkv_source
    assert "pl.pipeline(Q_ROPE_H_TILE, stage=2)" in qkv_source
    assert "if q_valid_rows == Q_ROPE_T_TILE:" in qkv_source
    assert "q_head_sq_row_full = pl.row_sum(q_head_sq_full)" in qkv_source


def test_qkv_rope_gathers_include_flattened_row_offsets() -> None:
    source = ast.unparse(_function("qkv_proj_rope.py", "qkv_proj_rope"))

    assert "qrp_row_offset = pl.transpose(qrp_row_grid" in source
    assert "qrp_dup_idx = pl.add(pl.cast(qrp_dup_f" in source
    assert "qrp_swap_idx = pl.add(qrp_swap_idx_local, qrp_row_offset)" in source
    assert "kv_row_offset = pl.transpose(kv_row_grid" in source
    assert "kv_dup_idx = pl.add(pl.cast(kv_dup_f" in source
    assert "kv_swap_idx = pl.add(pl.cast(kv_swap_f" in source


def test_qkv_rope_tail_initializes_full_gather_index_tiles() -> None:
    source = ast.unparse(_function("qkv_proj_rope.py", "qkv_proj_rope"))

    assert "q_rope_swap_idx = pl.create_tensor([T_MAX, ROPE_DIM], dtype=pl.INT32)" in source
    assert "q_rope_swap_idx_local = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)" in source
    assert "pl.store(qrp_swap_idx, [qrp_t0, 0], q_rope_swap_idx)" in source
    assert "pl.set_validshape(qrp_swap_idx," not in source
    assert (
        "q_swap_idx = pl.load(q_rope_swap_idx, [q_tg, 0], "
        "[Q_ROPE_T_TILE, ROPE_DIM], target_memory="
    ) in source
    assert "q_swap_idx_full = q_rope_swap_idx_local[q_tg:q_tg + Q_ROPE_T_TILE, :]" in source


def test_sparse_dynamic_padding_is_row_tiled() -> None:
    source = ast.unparse(_function("prefill_sparse_attn.py", "prefill_sparse_attn"))
    assert "prefill_sparse_dynamic_pad_q" in source
    assert "prefill_sparse_dynamic_pad_meta" in source
    assert "pl.slice(q_in, [T, H, HEAD_DIM]" not in source
    assert "q_row = pl.tile.full([1, Q_PAD_COLS]" in source
    assert "q_row = pl.load(q_in_flat" in source
