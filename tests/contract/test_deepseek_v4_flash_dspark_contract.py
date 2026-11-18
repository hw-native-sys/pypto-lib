# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host contracts for DSpark cache slots and independent LM-head groups."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_dspark"
sys.path.insert(0, str(MODEL_DIR))

from utils import block_table, pack_nz, state_slot_mapping


def _fused_target_arguments(tp):
    source = ast.parse((MODEL_DIR / "decode_fwd_dspark.py").read_text())
    forward = next(node for node in source.body
                   if isinstance(node, ast.FunctionDef) and node.name == "l2_decode_fwd_dspark")
    scope = next(node for node in forward.body if isinstance(node, ast.With)
                 and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                         and child.func.id == "decode_fwd" for child in ast.walk(node)))
    call = next(node.value for node in scope.body
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "decode_fwd")
    statements = scope.body[:scope.body.index(next(node for node in scope.body
                                                  if isinstance(node, ast.Expr) and node.value is call))]
    assignments = [node for node in statements if isinstance(node, ast.Assign)]
    namespace = {node.id: 8 for node in ast.walk(ast.Module(body=assignments, type_ignores=[]))
                 if isinstance(node, ast.Name) and node.id.isupper()}
    namespace.update({
        "TP_SIZE": tp, "DECODE_BATCH": 64, "GROUP_DECODE_TOKENS": 512,
        "LOCAL_DECODE_BATCH": 16, "LOCAL_DECODE_TOKENS": 128,
        "DSPARK_STATE_LOCAL_BATCH": 16, "MAX_LOGIT_ROWS": 128,
        "draft_batch": 16,
        "pl": SimpleNamespace(create_tensor=lambda shape, dtype: torch.empty(shape, dtype=dtype),
                              INT32=torch.int32, INT64=torch.int64, BF16=torch.bfloat16,
                              FP32=torch.float32),
    })
    for param in forward.args.args:
        namespace.setdefault(param.arg, object())
    for name in ("csa_cmp_block_table", "csa_idx_block_table", "hca_cmp_block_table"):
        namespace[name] = torch.arange(16 * 8, dtype=torch.int32).reshape(16, 8)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "fused_target_metadata", "exec"), namespace)
    target = ast.parse((MODEL_DIR / "decode_fwd.py").read_text())
    signature = next(node for node in target.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_decode_fwd").args.args
    assert len(signature) == len(call.args)
    arguments = {param.arg: eval(compile(ast.Expression(arg), "target_argument", "eval"), namespace)
                 for param, arg in zip(signature, call.args, strict=True)}
    return arguments, namespace


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_fused_target_metadata_preserves_eight_tokens_per_request(tp) -> None:
    arguments, namespace = _fused_target_arguments(tp)
    group_batch = tp * 16
    group_tokens = group_batch * 8
    group_token_names = (
        "position_ids", "swa_slot_mapping", "csa_cmp_freqs_cos", "csa_cmp_freqs_sin",
        "csa_ori_slot_mapping", "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
        "csa_state_slot_mapping", "csa_inner_state_slot_mapping", "hca_ori_slot_mapping",
        "hca_cmp_slot_mapping", "hca_state_slot_mapping",
    )
    group_request_names = (
        "csa_compress_state_block_table", "csa_inner_compress_state_block_table",
        "hca_compress_state_block_table", "hca_cmp_freqs_cos", "hca_cmp_freqs_sin",
    )
    for name in group_token_names + group_request_names:
        actual = arguments[name]
        assert actual.shape[0] == (group_tokens if name in group_token_names else group_batch), name
        original = namespace[f"prepared_{name}"]
        assert actual.untyped_storage().data_ptr() == original.untyped_storage().data_ptr(), name
        assert original.shape[0] == (512 if name in group_token_names else 64), name
    for name in ("csa_cmp_block_table", "csa_idx_block_table", "hca_cmp_block_table",
                 "csa_kv_seq_lens", "hca_kv_seq_lens"):
        assert arguments[name].shape[0] == 16, name
    for name in ("input_ids", "position_ids_local", "freqs_cos", "freqs_sin",
                 "compressed_freqs_cos", "compressed_freqs_sin", "swa_indices", "swa_lens"):
        assert arguments[name].shape[0] == 128, name
    assert arguments["position_ids"].shape[0] // arguments["hca_compress_state_block_table"].shape[0] == 8
    assert arguments["position_ids"].shape[0] // arguments["csa_compress_state_block_table"].shape[0] == 8


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("start", [122, 250])
def test_fused_hca_boundary_keeps_request_state_isolated(tp, start) -> None:
    arguments, _ = _fused_target_arguments(tp)
    tokens = arguments["position_ids"].shape[0]
    batch = arguments["hca_compress_state_block_table"].shape[0]
    sequence = tokens // batch
    tree = ast.parse((MODEL_DIR / "decode_compressor_ratio128.py").read_text())
    golden = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "golden_compressor")
    namespace = {
        "S": sequence, "D": 2, "OUT_DIM": 2, "HEAD_DIM": 2, "ROPE_HEAD_DIM": 2,
        "EPS": 1e-6, "COMPRESS_RATIO": 128, "COMPRESS_STATE_BLOCK_SIZE": 8,
        "CMP_STORAGE_BLOCK_SIZE": 1,
    }
    exec(compile(ast.Module(body=[golden], type_ignores=[]), "hca_golden", "exec"), namespace)
    pages_per_request = 17
    logical_pages = (start + sequence + 7) // 8
    table = torch.arange(logical_pages, dtype=torch.int32).remainder(pages_per_request).expand(batch, -1).clone()
    table += torch.arange(batch, dtype=torch.int32).unsqueeze(1) * pages_per_request
    state = torch.zeros(batch * pages_per_request, 8, 4)
    state.view(batch, pages_per_request, 8, 4)[..., :2] = torch.tensor([1.0, 1.0])
    positions = torch.arange(sequence, dtype=torch.int32).repeat(batch)
    state_slots = torch.full((tokens,), -1, dtype=torch.int64)
    cache_slots = torch.full((tokens,), -1, dtype=torch.int64)
    x = torch.ones(tokens, 2)
    active = (0, batch - 1)
    for request in active:
        rows = slice(request * sequence, (request + 1) * sequence)
        positions[rows] += start
        state_slots[rows] = request * pages_per_request * 8 + positions[rows].to(torch.int64) % (pages_per_request * 8)
        cache_slots[request * sequence + (127 - start % 128)] = request
        value = torch.tensor([float(request + 2), 1.0])
        state.view(batch, pages_per_request, 8, 4)[request, ..., :2] = value
        x[rows] = value
    original_state = state.clone()
    tensors = {
        "x": x, "kv": torch.full((tokens, 2), float("nan")),
        "compress_state_block_table": table, "position_ids": positions,
        "cmp_slot_mapping": cache_slots, "state_slot_mapping": state_slots,
        "compress_state": state, "wkv": torch.eye(2), "wgate": torch.zeros(2, 2),
        "ape": torch.zeros(128, 2), "norm_w": torch.ones(2),
        "cos": torch.ones(batch, 1), "sin": torch.zeros(batch, 1),
        "cmp_kv_cache": torch.full((batch, 1, 1, 2), float("nan")),
    }
    namespace["golden_compressor"](tensors)
    for request in active:
        value = x[request * sequence]
        expected = value * torch.rsqrt(value.square().mean() + 1e-6)
        torch.testing.assert_close(tensors["cmp_kv_cache"][request, 0, 0], expected)
        torch.testing.assert_close(tensors["kv"][request * sequence], expected)
    assert tensors["cmp_kv_cache"][1:-1].isnan().all()
    assert torch.equal(state[pages_per_request:-pages_per_request],
                       original_state[pages_per_request:-pages_per_request])


@pytest.mark.parametrize("batch", [4, 16, 64])
@pytest.mark.parametrize("sequence", [6, 8])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_output_projection_tile_fits_local_capacity(batch, sequence, tp) -> None:
    source = ast.parse((MODEL_DIR / "decode_o_proj.py").read_text())
    nodes = [
        node for node in source.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_token_tile")
        or (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "O_B_T_TILE" for target in node.targets))
    ]
    assert len(nodes) == 2
    local_tokens = batch * sequence // tp
    local_capacity = (local_tokens + 15) // 16 * 16
    namespace = {"TOKEN_TILE": 16, "LOCAL_T_PAD": local_capacity, "GROUP_T_PAD": tp * local_capacity}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "decode_o_proj_tiles", "exec"), namespace)
    tile = namespace["O_B_T_TILE"]
    assert local_capacity % tile == 0
    assert namespace["GROUP_T_PAD"] % tile == 0
    for active_rows in range(1, local_tokens + 1):
        padded_rows = (active_rows + tile - 1) // tile * tile
        assert padded_rows <= local_capacity


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_markov_fixture_uses_composition_world_size(tp) -> None:
    source = ast.parse((MODEL_DIR / "dspark_markov.py").read_text())
    builder = next(node for node in source.body
                   if isinstance(node, ast.FunctionDef) and node.name == "build_tensor_specs")
    namespace = {
        "WORLD_SIZE": tp, "TP_SIZE": tp, "DSPARK_SUPPORTED_BATCHES": (4, 8, 12, 16),
        "DSPARK_QUERY_WIDTH": 7, "DSPARK_QUERY_PAD": 8, "D": 4096,
        "VOCAB": 129280, "VOCAB_PER_TP": 129280 // tp, "DSPARK_MARKOV_RANK": 256,
        "MAX_LOGIT_ROWS": 128, "LM_K_TILE": 256,
    }
    exec(compile(ast.Module(body=[builder], type_ignores=[]), "markov_fixture", "exec"), namespace)
    build = namespace["build_tensor_specs"]
    standalone = {spec.name: spec for spec in build(4, distributed=True)}
    composed = {spec.name: spec for spec in build(4, distributed=True, world_size=16)}
    assert all(spec.shape[0] == tp for spec in standalone.values())
    assert all(spec.shape[0] == 16 for spec in composed.values())
    assert composed["lm_head_weight"].shape == [16, 129280 // tp, 4096]
    hidden = composed["head_hidden"].create_tensor()
    assert hidden.shape == (16, 4, 7, 4096)
    assert not torch.equal(hidden[0], hidden[-1])
    assert composed["logit_row_indices"].create_tensor().shape == (16, 128)
    with pytest.raises(ValueError, match="positive multiple"):
        build(4, distributed=True, world_size=0)


@pytest.mark.parametrize("attention_tp,head_tp,world", [(1, 4, 16), (4, 4, 16), (2, 2, 8), (1, 8, 16)])
def test_lm_head_parallelism_is_independent(attention_tp, head_tp, world) -> None:
    config = _load_lm_head_config([
        "--tp", str(attention_tp), "--lm-head-tp", str(head_tp),
        "--ep", str(world), "--dp", str(world // attention_tp),
    ])
    assert config["TP_SIZE"] == head_tp
    assert config["WORLD_SIZE"] == world
    assert config["VOCAB_PER_TP"] == 129280 // head_tp


def _load_lm_head_config(argv):
    source = ast.parse((MODEL_DIR / "lm_head.py").read_text())
    start = next(i for i, node in enumerate(source.body)
                 if isinstance(node, ast.FunctionDef) and node.name == "_parse_int_argv")
    end = next(i for i, node in enumerate(source.body)
               if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "VOCAB_PER_TP" for t in node.targets))
    namespace = {
        "sys": SimpleNamespace(argv=["test", *argv]),
        "_TP_DEFAULT": 2, "_TP_CHOICES": (1, 2, 4, 8, 16), "VOCAB": 129280,
    }
    exec(compile(ast.Module(body=source.body[start:end + 1], type_ignores=[]), "head_config", "exec"), namespace)
    return namespace


def test_lm_head_standalone_and_composed_defaults() -> None:
    assert _load_lm_head_config(["--tp", "4", "--dp", "4"])["WORLD_SIZE"] == 16
    assert _load_lm_head_config(["--tp=4", "--ep=16"])["TP_SIZE"] == 4
    assert _load_lm_head_config([])["TP_SIZE"] == 2


@pytest.mark.parametrize("argv", [
    ["--lm-head-tp", "3"], ["--lm-head-tp", "0"],
    ["--lm-head-tp", "4", "--ep", "2"], ["--ep", "0"],
])
def test_lm_head_rejects_invalid_grouping(argv) -> None:
    with pytest.raises(ValueError):
        _load_lm_head_config(argv)


@pytest.mark.parametrize("filename,function", [("decode_fwd.py", "_decode_fwd"),
                                              ("prefill_fwd.py", "prefill_fwd")])
def test_target_lm_head_participation_is_not_gated_by_attention_activity(filename, function) -> None:
    source = ast.parse((MODEL_DIR / filename).read_text())
    forward = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == function)
    call = next(node.value for node in forward.body
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "lm_head")
    for rank in range(16):
        namespace = {"my_rank": rank, "LM_HEAD_TP_SIZE": 4}
        group = eval(compile(ast.Expression(call.args[-4]), "head_group", "eval"), namespace)
        local = eval(compile(ast.Expression(call.args[-3]), "head_rank", "eval"), namespace)
        assert (group, local) == (rank // 4 * 4, rank % 4)


@pytest.mark.parametrize("function", ["o_group_a2a", "o_proj_reduce_scatter"])
def test_singleton_output_projection_has_no_peer_credit_scope(function) -> None:
    source = ast.parse((MODEL_DIR / "decode_o_proj.py").read_text())
    kernel = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == function)
    guarded_scope = next(node for node in kernel.body if isinstance(node, ast.If)
                         and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                         and node.test.left.id == "TP_SIZE")
    for tp in (1, 2, 4):
        enabled = eval(compile(ast.Expression(guarded_scope.test), "peer_credit_guard", "eval"), {"TP_SIZE": tp})
        assert enabled == (tp > 1)
    assert any(isinstance(node, ast.keyword) and node.arg == "no_dep_args"
               for node in ast.walk(guarded_scope))


def test_prefill_logits_oracle_uses_head_group_not_attention_group() -> None:
    source = ast.parse((MODEL_DIR / "prefill_fwd.py").read_text())
    compare = next(node for node in source.body
                   if isinstance(node, ast.FunctionDef) and node.name == "logits_compare")
    namespace = {
        "TP_SIZE": 1, "LM_HEAD_TP_SIZE": 4, "MAX_LOGIT_ROWS": 1,
    }
    exec(compile(ast.Module(body=[compare], type_ignores=[]), "head_oracle", "exec"), namespace)
    weights = torch.arange(1, 17, dtype=torch.float32).view(16, 1, 1).expand(16, 16, 16).contiguous()
    hidden = torch.arange(1, 17, dtype=torch.float32).view(16, 1, 1).expand(16, 1, 16).contiguous()
    logits = torch.stack([torch.cat([
        weights[rank // 4 * 4 + shard] @ hidden[rank, 0] for shard in range(4)
    ]) for rank in range(16)]).unsqueeze(1)
    inputs = {
        "lm_head_weight": pack_nz(weights), "logit_row_indices": torch.zeros((16, 1), dtype=torch.int32),
        "query_start_loc": torch.tensor([[0, 1]] * 16),
    }
    assert namespace["logits_compare"](
        logits, None, inputs=inputs, actual_outputs={"x_out": hidden},
    )[0]
    logits[4, 0, 0] += 1
    assert not namespace["logits_compare"](
        logits, None, inputs=inputs, actual_outputs={"x_out": hidden},
    )[0]


def test_prefill_signal_retirement_exports_its_completion() -> None:
    tree = ast.parse((MODEL_DIR / "moe.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "clear_prefill_moe_signals")
    scope = next(node for node in function.body if isinstance(node, ast.With))
    assert isinstance(scope.items[0].optional_vars, ast.Name)
    returned = next(node for node in function.body if isinstance(node, ast.Return))
    assert returned.value.id == scope.items[0].optional_vars.id
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "read" and isinstance(node.args[0], ast.Name)
               and node.args[0].id == "stage_token" for node in ast.walk(scope))


def test_prefill_head_waits_for_final_moe_retirement() -> None:
    tree = ast.parse((MODEL_DIR / "prefill_fwd.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "prefill_fwd")
    calls = {node.targets[0].id: node.value for node in function.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and isinstance(node.value, ast.Call)}
    retired = next(name for name, call in calls.items()
                   if isinstance(call.func, ast.Name) and call.func.id == "clear_prefill_moe_signals")
    head = next(node.value for node in function.body
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "lm_head")
    ready = calls[head.args[-1].id]
    assert ast.unparse(ready.func) == "pl.system.task_dummy"
    deps = next(keyword.value for keyword in ready.keywords if keyword.arg == "deps")
    assert {node.id for node in deps.elts} == {retired, "final_norm_tid"}


def test_block_table_can_model_hca_deployment_request_slots() -> None:
    physical_blocks = 1088
    blocks_per_request = 17
    table = block_table(
        batch=2,
        table_blocks=blocks_per_request + 1,
        physical_blocks=physical_blocks,
        request_slots=64,
    )

    assert table.shape == (2, blocks_per_request + 1)
    assert table[0, :blocks_per_request].tolist() == [64 * block for block in range(blocks_per_request)]
    assert table[1, :blocks_per_request].tolist() == [64 * block + 1 for block in range(blocks_per_request)]
    assert table[0, blocks_per_request].item() == table[0, 0].item()
    assert set(table[0].tolist()).isdisjoint(table[1].tolist())


def test_block_table_request_slots_default_is_backward_compatible() -> None:
    default = block_table(batch=2, table_blocks=7, physical_blocks=32)
    explicit = block_table(batch=2, table_blocks=7, physical_blocks=32, request_slots=2)
    assert default.equal(explicit)


def test_block_table_rejects_fewer_request_slots_than_rows() -> None:
    with pytest.raises(ValueError, match="request_slots must be >= batch"):
        block_table(batch=2, table_blocks=7, physical_blocks=32, request_slots=1)


def test_prefill_csa_state_handoff_matches_decode_ring_across_wraps() -> None:
    block_size = 2
    storage_len = 16
    ring_pages = storage_len // block_size
    prefill_table = block_table(
        batch=1,
        table_blocks=268,
        physical_blocks=512,
        request_slots=64,
    )
    prefill_tables = prefill_table.unsqueeze(0).expand(4, -1, -1).clone()
    decode_tables = torch.full((4, 64, ring_pages), -1, dtype=torch.int32)
    decode_tables[:, 0].copy_(prefill_tables[:, 0, :ring_pages])

    for decode_start in range(512, 528):
        live_positions = torch.arange(
            decode_start - 8,
            decode_start + 8,
            dtype=torch.int64,
        ).unsqueeze(0).expand(4, -1)
        prefill_rows = state_slot_mapping(
            live_positions,
            prefill_tables[:, 0],
            state_block_size=block_size,
        )
        decode_rows = state_slot_mapping(
            live_positions % storage_len,
            decode_tables[:, 0],
            state_block_size=block_size,
        )

        assert torch.equal(decode_rows, prefill_rows)
        assert all(torch.unique(rank_rows).numel() == storage_len for rank_rows in decode_rows)
