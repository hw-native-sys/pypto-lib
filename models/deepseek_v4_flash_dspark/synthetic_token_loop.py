# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
# ci: no-sim
"""Validate the DSpark Prefill-to-Decode loop with existing model programs.

The session compiles or loads four existing public programs and runs them in
one persistent distributed worker:

``target prefill -> drafter -> Markov -> (target decode -> verify -> drafter -> Markov) x N``.

It is a validation driver, not a Serving implementation.  One logical request's
Prefill rows are rank-major sharded across each TP group; its speculative query
and verification rows are owned by the group leader.  Decode has no
inactive-query mask, so peer ranks retain valid fixture-owned dummy requests.
Target-model resident caches and the TP-sharded output head are shared across
stages, while drafter KV is retained independently.  Host verification uses the
same longest-prefix-plus-one rule required by Serving.  The bounded fixture
validates ABI, ownership, and resident-state flow; its Prefill and one-bank
Decode layer weights are independently initialized, so it is not an end-to-end
numerical golden for one checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import torch


TP_SIZE = 4
DSPARK_BATCH = 4
QUERY_WIDTH = 7
DECODE_WIDTH = QUERY_WIDTH + 1
PREFILL_TOKENS = 512
_SSA_SUFFIX = re.compile(r"__ssa_v\d+$")
_COMPILE_MARKER = "[DSPARK SESSION] runtime_dir="
_TARGET_SHARED_IO_DIRECTIONS = {
    "logit_row_indices": "In",
    "logits": "Out",
    "sampled_ids": "Out",
}


def _normalized_name(name: str) -> str:
    return _SSA_SUFFIX.sub("", name)


def _metadata(compiled):
    infos, _, _ = compiled._get_metadata()
    pairs = [(_normalized_name(info.name), info) for info in infos]
    names = [name for name, _ in pairs]
    if len(names) != len(set(names)):
        raise ValueError("compiled parameters collide after stripping SSA suffixes")
    return pairs


def _info_map(compiled):
    return dict(_metadata(compiled))


def _ordered_args(compiled, io_tensors, resident_handles):
    ordered = []
    for name, info in _metadata(compiled):
        if info.shape is None:
            raise ValueError(f"unexpected scalar parameter {name!r}")
        if name in resident_handles:
            ordered.append(resident_handles[name])
        else:
            ordered.append(io_tensors[name])
    return ordered


def _purge_model_modules(model_dir: Path) -> None:
    for module_name, module in list(sys.modules.items()):
        if module_name == "__main__":
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        try:
            Path(filename).resolve().relative_to(model_dir)
        except ValueError:
            continue
        sys.modules.pop(module_name, None)
    gc.collect()


def _materialize(spec):
    value = spec.create_tensor()
    if tuple(value.shape) != tuple(spec.shape):
        raise ValueError(
            f"{spec.name}: initializer returned {tuple(value.shape)}, expected {tuple(spec.shape)}"
        )
    return value.contiguous()


def _spec_map(specs):
    return {spec.name: spec for spec in specs if hasattr(spec, "shape")}


def _direction_name(info) -> str:
    return getattr(info.direction, "name", str(info.direction).rsplit(".", 1)[-1])


def _resident_names(specs, param_infos, *, output_state_names=frozenset()):
    return {
        spec.name
        for spec in specs
        if (
            (
                getattr(spec, "is_resident", False)
                and _direction_name(param_infos[spec.name]) not in {"Out", "InOut"}
            )
            or spec.name in output_state_names
        )
    }


def _materialize_stage(specs, resident_names, *, excluded_names=frozenset()):
    return {
        spec.name: _materialize(spec)
        for spec in specs
        if (
            hasattr(spec, "shape")
            and spec.name not in resident_names
            and spec.name not in excluded_names
        )
    }


def _share_tensors(tensors) -> None:
    for tensor in tensors.values():
        if isinstance(tensor, torch.Tensor) and not tensor.is_shared():
            tensor.share_memory_()


def _assert_shared_tensors(tensors, stage: str) -> None:
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{stage} host IO {name!r} is not a tensor")
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or not tensor.is_shared():
            raise ValueError(
                f"{stage} host IO {name!r} must be contiguous shared CPU memory created before worker fork"
            )


def _unique_storage_tensors(tensors):
    unique = []
    seen = set()
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            unique.append(tensor)
    return unique


def _assert_program_args(compiled, io_tensors, resident_handles, stage: str) -> None:
    expected = {name for name, _ in _metadata(compiled)}
    actual = set(io_tensors) | set(resident_handles)
    if expected != actual:
        raise ValueError(
            f"{stage} runtime ABI mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    from pypto.ir.compiled_program import _to_torch_dtype

    for name, info in _metadata(compiled):
        if name in resident_handles:
            continue
        if info.shape is None:
            raise ValueError(f"{stage} unexpectedly exposes scalar parameter {name!r}")
        tensor = io_tensors[name]
        expected_dtype = _to_torch_dtype(info.dtype)
        if expected_dtype is None or tensor.dtype != expected_dtype:
            raise TypeError(
                f"{stage} host IO {name!r} has dtype {tensor.dtype}, expected {expected_dtype}"
            )
        expected_shape = tuple(info.shape)
        if len(tensor.shape) != len(expected_shape) or any(
            expected_dim >= 0 and actual_dim != expected_dim
            for actual_dim, expected_dim in zip(tensor.shape, expected_shape, strict=True)
        ):
            raise ValueError(
                f"{stage} host IO {name!r} has shape {tuple(tensor.shape)}, "
                f"expected compiled shape {tuple(expected_shape)}"
            )


def _verify_target_step(
    draft_token_ids: torch.Tensor,
    target_sampled_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    start_positions: torch.Tensor,
    tp_size: int = TP_SIZE,
):
    """Accept the longest draft prefix and pack the target rows for the next drafter call."""
    if draft_token_ids.shape != (target_hidden.shape[0], QUERY_WIDTH):
        raise ValueError("draft ids must be [R,7]")
    if target_sampled_ids.shape != (target_hidden.shape[0], DECODE_WIDTH):
        raise ValueError("target sampled ids must be [R,8]")
    if target_hidden.shape[1] != DECODE_WIDTH:
        raise ValueError("target hidden must contain eight verification rows")
    if start_positions.shape != (target_hidden.shape[0] // tp_size,):
        raise ValueError("start_positions must contain one value per TP group")

    ranks = target_hidden.shape[0]
    packed_hidden = torch.zeros_like(target_hidden)
    context_positions = torch.zeros(ranks, DECODE_WIDTH, dtype=torch.int32)
    valid_rows = torch.zeros(ranks, DECODE_WIDTH, dtype=torch.bool)
    accepted_counts = torch.zeros(ranks, dtype=torch.int32)
    next_tokens = torch.zeros(ranks, dtype=torch.int64)
    anchors = torch.full((ranks,), -1, dtype=torch.int32)

    for group_index, group_base in enumerate(range(0, ranks, tp_size)):
        leader = group_base
        matched = 0
        while (
            matched < QUERY_WIDTH
            and int(draft_token_ids[leader, matched])
            == int(target_sampled_ids[leader, matched])
        ):
            matched += 1
        accepted = matched + 1
        start = int(start_positions[group_index])
        next_token = int(target_sampled_ids[leader, matched])
        packed_hidden[leader, :accepted].copy_(target_hidden[leader, :accepted])
        context_positions[leader, :accepted] = torch.arange(
            start,
            start + accepted,
            dtype=torch.int32,
        )
        valid_rows[leader, :accepted] = True
        accepted_counts[leader] = accepted
        next_tokens[leader] = next_token
        anchors[leader] = start + accepted - 1
    return SimpleNamespace(
        target_hidden=packed_hidden,
        context_positions=context_positions,
        valid_rows=valid_rows,
        accepted_counts=accepted_counts,
        next_tokens=next_tokens,
        anchors=anchors,
    )


def _context_slots(
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    valid_rows: torch.Tensor,
    block_size: int,
    request_ids: torch.Tensor | None = None,
):
    ranks, layers, _, _ = block_tables.shape
    if request_ids is None:
        request_ids = torch.zeros(positions.shape[1], dtype=torch.int64)
    if request_ids.shape != (positions.shape[1],):
        raise ValueError("request_ids must identify every local context row")
    slots = torch.full((ranks, layers, positions.shape[1]), -1, dtype=torch.int64)
    for rank in range(ranks):
        for token in range(positions.shape[1]):
            if not bool(valid_rows[rank, token]):
                continue
            position = int(positions[rank, token])
            request = int(request_ids[token])
            for layer in range(layers):
                physical = int(block_tables[rank, layer, request, position // block_size])
                slots[rank, layer, token] = physical * block_size + position % block_size
    return slots


def _replicate_group_metadata(
    local_positions: torch.Tensor,
    local_slots: torch.Tensor,
    group_positions: torch.Tensor,
    group_slots: torch.Tensor,
    tp_size: int = TP_SIZE,
) -> None:
    """Pack local rows rank-major and replicate them across each TP group."""
    ranks, local_rows = local_positions.shape
    layers = local_slots.shape[1]
    if ranks % tp_size or local_slots.shape != (ranks, layers, local_rows):
        raise ValueError("local context metadata has inconsistent rank or row dimensions")
    if group_positions.shape != (ranks, tp_size * local_rows) or group_slots.shape != (
        ranks,
        layers,
        tp_size * local_rows,
    ):
        raise ValueError("group context metadata must contain TP rank-major local rows")
    for group_base in range(0, ranks, tp_size):
        group_slice = slice(group_base, group_base + tp_size)
        packed_positions = local_positions[group_slice].reshape(-1)
        packed_slots = local_slots[group_slice].permute(1, 0, 2).reshape(layers, -1)
        group_positions[group_slice].copy_(packed_positions.unsqueeze(0).expand(tp_size, -1))
        group_slots[group_slice].copy_(packed_slots.unsqueeze(0).expand(tp_size, -1, -1))


def _refresh_drafter_query_metadata(io, block_size: int) -> None:
    ranks, batch = io["anchor_positions"].shape
    local_query_rows = io["query_group_position_ids"].shape[1] // TP_SIZE
    active_query_rows = batch * QUERY_WIDTH
    if active_query_rows > local_query_rows:
        raise ValueError("Drafter batch exceeds the fixed query metadata capacity")
    offsets = torch.arange(1, QUERY_WIDTH + 1, dtype=torch.int32)
    local_positions = torch.zeros(ranks, local_query_rows, dtype=torch.int32)
    local_positions[:, :active_query_rows].copy_(
        (io["anchor_positions"].unsqueeze(-1) + offsets).reshape(
            ranks,
            active_query_rows,
        )
    )
    request_ids = torch.zeros(local_query_rows, dtype=torch.int64)
    request_ids[:active_query_rows].copy_(
        torch.arange(batch, dtype=torch.int64).repeat_interleave(QUERY_WIDTH)
    )
    valid_rows = torch.zeros_like(local_positions, dtype=torch.bool)
    valid_rows[:, :active_query_rows] = True
    local_slots = _context_slots(
        io["block_tables"],
        local_positions,
        valid_rows,
        block_size,
        request_ids,
    )
    _replicate_group_metadata(
        local_positions,
        local_slots,
        io["query_group_position_ids"],
        io["query_group_slot_mapping"],
    )


def _import_stage(model_dir: Path, module_name: str, argv: list[str]):
    saved_argv = sys.argv
    sys.argv = [str(model_dir / f"{module_name}.py"), *argv]
    sys.path.insert(0, str(model_dir))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.argv = saved_argv
        if sys.path and sys.path[0] == str(model_dir):
            sys.path.pop(0)


def _stage_import_argv(stage: str, ep: int) -> list[str]:
    if stage == "markov":
        return ["--tp", str(TP_SIZE), "--dp", str(ep // TP_SIZE)]
    argv = ["--tp", str(TP_SIZE), "--ep", str(ep)]
    if stage == "decode":
        argv.extend(["--weight-bank-size", "1"])
    return argv


def _compile_stage_child(args) -> None:
    from golden import run
    from pypto.ir.distributed_compiled_program import DistributedConfig

    model_dir = Path(__file__).resolve().parent
    module_names = {
        "prefill": "prefill_fwd",
        "decode": "decode_fwd",
        "drafter": "dspark_drafter",
        "markov": "dspark_markov",
    }
    module = _import_stage(
        model_dir,
        module_names[args.stage],
        _stage_import_argv(args.stage, args.ep),
    )
    if args.stage == "prefill":
        fn = module.l3_prefill_fwd
        specs = module.build_tensor_specs(num_tokens=args.prefill_tokens)
    elif args.stage == "decode":
        fn = module.l3_decode_fwd
        specs = module.build_tensor_specs(
            start_pos=args.prefill_tokens,
            weight_bank_size=1,
            runtime_case="full_active",
        )
    elif args.stage == "drafter":
        fn = module.l3_dspark_drafter
        specs = module.build_tensor_specs(DSPARK_BATCH, mode="prefill")
    else:
        fn = module.l3_distributed_markov_sample
        specs = module.build_tensor_specs(DSPARK_BATCH, distributed=True)

    result = run(
        fn=fn,
        specs=specs,
        compile_only=True,
        compile_cfg={
            "distributed_config": DistributedConfig(
                device_ids=args.device_ids[: args.ep],
                num_sub_workers=0,
            ),
        },
        runtime_cfg={"platform": args.platform},
    )
    if not result.passed or result.work_dir is None:
        raise RuntimeError(result.error or f"{args.stage} compile did not produce a runtime directory")
    print(f"{_COMPILE_MARKER}{result.work_dir.resolve()}", flush=True)


def _compile_stage(stage: str, args) -> Path:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-compile",
        "--stage",
        stage,
        "--platform",
        args.platform,
        "--ep",
        str(args.ep),
        "--prefill-tokens",
        str(args.prefill_tokens),
        "--device",
        args.device,
    ]
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_root}:{old_pythonpath}" if old_pythonpath else str(repo_root)
    print(f"[DSPARK SESSION] compiling {stage}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    runtime_dir = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.startswith(_COMPILE_MARKER):
            runtime_dir = Path(line[len(_COMPILE_MARKER) :].strip()).resolve()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{stage} compile failed with exit code {return_code}")
    if runtime_dir is None:
        raise RuntimeError(f"{stage} compile did not report its runtime directory")
    return runtime_dir


def _load_model_specs(model_dir: Path, ep: int, prefill_tokens: int):
    stages = {}
    for stage, module_name in (
        ("prefill", "prefill_fwd"),
        ("decode", "decode_fwd"),
        ("drafter", "dspark_drafter"),
        ("markov", "dspark_markov"),
    ):
        module = _import_stage(model_dir, module_name, _stage_import_argv(stage, ep))
        if stage == "prefill":
            specs = module.build_tensor_specs(num_tokens=prefill_tokens)
        elif stage == "decode":
            specs = module.build_tensor_specs(
                start_pos=prefill_tokens,
                weight_bank_size=1,
                runtime_case="full_active",
            )
        elif stage == "drafter":
            specs = module.build_tensor_specs(DSPARK_BATCH, mode="prefill")
        else:
            specs = module.build_tensor_specs(DSPARK_BATCH, distributed=True)
        stages[stage] = SimpleNamespace(module=module, specs=specs)
        _purge_model_modules(model_dir)
    stages["utils"] = SimpleNamespace(module=_import_stage(model_dir, "utils", []))
    return stages


def _same_spec(left, right) -> bool:
    return tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype


_DRAFTER_DYNAMIC_CONTEXT_NAMES = frozenset(
    {"target_hidden", "context_group_position_ids", "context_group_slot_mapping"}
)


def _materialize_drafter_ios(stage, resident_names):
    prefill_io = _materialize_stage(stage.specs, resident_names)
    prefill_specs = _spec_map(stage.specs)
    decode_specs = stage.module.build_tensor_specs(DSPARK_BATCH, mode="decode")
    decode_spec_map = _spec_map(decode_specs)
    decode_names = {
        spec.name
        for spec in decode_specs
        if hasattr(spec, "shape") and spec.name not in resident_names
    }
    if decode_names != set(prefill_io):
        raise ValueError("Drafter Prefill and Decode expose different non-resident parameters")
    for name in decode_names - _DRAFTER_DYNAMIC_CONTEXT_NAMES:
        if not _same_spec(prefill_specs[name], decode_spec_map[name]):
            raise ValueError(f"Drafter Prefill and Decode disagree on {name!r}")

    decode_io = dict(prefill_io)
    target_hidden = _materialize(decode_spec_map["target_hidden"])
    context_positions = _materialize(decode_spec_map["context_group_position_ids"])
    context_slots = _materialize(decode_spec_map["context_group_slot_mapping"])
    decode_io["target_hidden"] = target_hidden[:, :DECODE_WIDTH].contiguous()
    group_decode_rows = TP_SIZE * DECODE_WIDTH
    decode_io["context_group_position_ids"] = context_positions[
        :, :group_decode_rows
    ].contiguous()
    decode_io["context_group_slot_mapping"] = context_slots[
        :, :, :group_decode_rows
    ].contiguous()
    return prefill_io, decode_io


def _compiled_alias_compatible(
    left,
    right,
    actual_shape,
    *,
    direction_name: str = "InOut",
) -> bool:
    if left.shape is None or right.shape is None:
        return False
    left_shape = tuple(left.shape)
    right_shape = tuple(right.shape)
    if len(left_shape) != len(right_shape) or len(left_shape) != len(actual_shape):
        return False
    if left.dtype != right.dtype or left.direction != right.direction:
        return False
    direction = _direction_name(left)
    if direction != direction_name:
        return False
    if any(
        left_dim >= 0 and right_dim >= 0 and left_dim != right_dim
        for left_dim, right_dim in zip(left_shape, right_shape, strict=True)
    ):
        return False
    return all(
        expected < 0 or expected == actual
        for shape in (left_shape, right_shape)
        for expected, actual in zip(shape, actual_shape, strict=True)
    )


def _reuse_target_terminal_io(stages, compiled, prefill_io, decode_io) -> None:
    prefill_specs = _spec_map(stages["prefill"].specs)
    decode_specs = _spec_map(stages["decode"].specs)
    prefill_infos = _info_map(compiled["prefill"])
    decode_infos = _info_map(compiled["decode"])
    for name, direction in _TARGET_SHARED_IO_DIRECTIONS.items():
        if not _same_spec(prefill_specs[name], decode_specs[name]) or not (
            _compiled_alias_compatible(
                prefill_infos[name],
                decode_infos[name],
                prefill_io[name].shape,
                direction_name=direction,
            )
        ):
            raise ValueError(f"target Prefill and Decode cannot share {name!r}")
        decode_io[name] = prefill_io[name]


_TARGET_CACHE_ALIASES = {
    "raw_kv_pool": "kv_cache",
    "hca_compress_state": "hca_compress_state",
    "hca_cmp_kv": "hca_cmp_kv",
    "csa_compress_state": "csa_compress_state",
    "csa_cmp_kv": "csa_cmp_kv",
    "csa_inner_compress_state": "csa_inner_compress_state",
    "csa_idx_kv_cache": "idx_kv_cache",
    "csa_idx_kv_scale": "idx_kv_scale",
}

_PREFILL_GROUP_STATE_TABLES = {
    "csa_compress_state_block_table": "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table": "csa_inner_compress_state_block_table",
    "hca_compress_state_block_table": "hca_compress_state_block_table",
}
_PREFILL_LOCAL_REQUEST_TABLES = {
    "csa_cmp_block_table": "csa_cmp_block_table",
    "csa_idx_block_table": "idx_block_table",
    "hca_cmp_block_table": "hca_cmp_block_table",
}
_PREFILL_HANDOFF_TABLE_NAMES = frozenset(
    {"ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table"}
    | set(_PREFILL_GROUP_STATE_TABLES.values())
    | set(_PREFILL_LOCAL_REQUEST_TABLES.values())
)
_DECODE_RUNTIME_METADATA_NAMES = frozenset(
    {
        *_PREFILL_GROUP_STATE_TABLES,
        *_PREFILL_LOCAL_REQUEST_TABLES,
        "position_ids_local",
        "position_ids",
        "freqs_cos_local",
        "freqs_sin_local",
        "freqs_cos",
        "freqs_sin",
        "csa_cmp_freqs_cos",
        "csa_cmp_freqs_sin",
        "hca_cmp_freqs_cos",
        "hca_cmp_freqs_sin",
        "swa_slot_mapping",
        "swa_indices",
        "swa_lens",
        "csa_ori_slot_mapping",
        "csa_window_swa_indices",
        "csa_window_swa_lens",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
        "csa_kv_seq_lens",
        "hca_ori_slot_mapping",
        "hca_window_swa_indices",
        "hca_window_swa_lens",
        "hca_cmp_slot_mapping",
        "hca_state_slot_mapping",
        "hca_kv_seq_lens",
        "logit_row_indices",
    }
)


def _prepare_resident_hosts(stages, compiled):
    prefill = stages["prefill"]
    decode = stages["decode"]
    drafter = stages["drafter"]
    markov = stages["markov"]
    prefill_specs = _spec_map(prefill.specs)
    decode_specs = _spec_map(decode.specs)

    prefill_infos = _info_map(compiled["prefill"])
    decode_infos = _info_map(compiled["decode"])
    drafter_infos = _info_map(compiled["drafter"])
    markov_infos = _info_map(compiled["markov"])

    prefill_names = _resident_names(
        prefill.specs,
        prefill_infos,
        output_state_names=frozenset(
            {*prefill.module.RESIDENT_CACHE_NAMES, "final_norm_w", "lm_head_weight"}
        ),
    )
    decode_names = _resident_names(
        decode.specs,
        decode_infos,
        output_state_names=frozenset(
            {*_TARGET_CACHE_ALIASES, "final_norm_w", "lm_head_weight"}
        ),
    )
    # The two programs intentionally use different active-row maps.
    prefill_names.discard("logit_row_indices")
    decode_names.difference_update(_DECODE_RUNTIME_METADATA_NAMES)
    drafter_names = _resident_names(
        drafter.specs,
        drafter_infos,
        output_state_names=frozenset({"kv_caches"}),
    )
    markov_names = _resident_names(
        markov.specs,
        markov_infos,
        output_state_names=frozenset({"final_norm_weight", "lm_head_weight"}),
    )

    hosts = {}
    for name in sorted(prefill_names):
        hosts[("prefill", name)] = _materialize(prefill_specs[name])

    decode_aliases = {}
    for decode_name, prefill_name in _TARGET_CACHE_ALIASES.items():
        if decode_name not in decode_names or prefill_name not in prefill_names:
            raise ValueError(f"target cache alias is missing: {decode_name} -> {prefill_name}")
        if not _compiled_alias_compatible(
            decode_infos[decode_name],
            prefill_infos[prefill_name],
            hosts[("prefill", prefill_name)].shape,
        ):
            raise ValueError(f"target cache ABI mismatch: {decode_name} != {prefill_name}")
        decode_aliases[decode_name] = ("prefill", prefill_name)

    for name in sorted(decode_names):
        if name in decode_aliases:
            continue
        if name in {"final_norm_w", "lm_head_weight"}:
            if (
                name not in prefill_names
                or not _same_spec(decode_specs[name], prefill_specs[name])
                or not _compiled_alias_compatible(
                    decode_infos[name],
                    prefill_infos[name],
                    hosts[("prefill", name)].shape,
                    direction_name="In",
                )
            ):
                raise ValueError(f"shared target head ABI mismatch: Decode {name}")
            decode_aliases[name] = ("prefill", name)
        elif name in prefill_names and _same_spec(decode_specs[name], prefill_specs[name]):
            decode_aliases[name] = ("prefill", name)
        else:
            hosts[("decode", name)] = _materialize(decode_specs[name])

    drafter_specs = _spec_map(drafter.specs)
    for name in sorted(drafter_names):
        hosts[("drafter", name)] = _materialize(drafter_specs[name])

    markov_specs = _spec_map(markov.specs)
    markov_aliases = {
        "final_norm_weight": ("prefill", "final_norm_w"),
        "lm_head_weight": ("prefill", "lm_head_weight"),
    }
    for markov_name, (_, prefill_name) in markov_aliases.items():
        if markov_name not in markov_names or prefill_name not in prefill_names:
            raise ValueError(f"shared target head input is missing: {markov_name} -> {prefill_name}")
        actual_shape = hosts[("prefill", prefill_name)].shape
        if not _same_spec(markov_specs[markov_name], prefill_specs[prefill_name]) or not (
            _compiled_alias_compatible(
                markov_infos[markov_name],
                prefill_infos[prefill_name],
                actual_shape,
                direction_name="In",
            )
        ):
            raise ValueError(f"shared target head ABI mismatch: {markov_name} != {prefill_name}")
    for name in sorted(markov_names):
        if name not in markov_aliases:
            hosts[("markov", name)] = _materialize(markov_specs[name])

    return SimpleNamespace(
        hosts=hosts,
        names={
            "prefill": prefill_names,
            "decode": decode_names,
            "drafter": drafter_names,
            "markov": markov_names,
        },
        decode_aliases=decode_aliases,
        markov_aliases=markov_aliases,
    )


def _constrain_target_lm_head(resident) -> None:
    key = ("prefill", "lm_head_weight")
    if key not in resident.hosts:
        raise ValueError("target Prefill LM head must be a shared resident tensor")
    weight = resident.hosts[key]
    weight.zero_()
    scale = 1.0 / max(weight.shape[-1], 1) ** 0.5
    for group_base in range(0, weight.shape[0], TP_SIZE):
        weight[group_base, 0].fill_(scale)
        weight[group_base, 1].fill_(-scale)



def _allocate_residents(runtime, resident):
    handles = {}
    allocated = []
    for key, host in resident.hosts.items():
        handles[key] = runtime.alloc_stacked_tensor(host)
        allocated.append(key)

    stage_handles = {
        "prefill": {
            name: handles[("prefill", name)]
            for name in resident.names["prefill"]
        },
        "decode": {},
        "drafter": {
            name: handles[("drafter", name)]
            for name in resident.names["drafter"]
        },
        "markov": {
            name: handles[resident.markov_aliases.get(name, ("markov", name))]
            for name in resident.names["markov"]
        },
    }
    for name in resident.names["decode"]:
        key = resident.decode_aliases.get(name, ("decode", name))
        stage_handles["decode"][name] = handles[key]
    return handles, allocated, stage_handles


def _require_finite_nonzero(tensor: torch.Tensor, stage: str) -> None:
    if tensor.numel() == 0:
        raise AssertionError(f"{stage} produced an empty tensor")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise AssertionError(f"{stage} produced NaN or Inf")
    if not bool(torch.count_nonzero(tensor)):
        raise AssertionError(f"{stage} produced only zeros")


def _assert_prefill_cache_tables_fit(prefill_io, resident, prefill_module) -> None:
    table_contracts = (
        ("kv_cache", "ori_block_table", prefill_module.FWD_NUM_LAYERS),
        ("hca_cmp_kv", "hca_cmp_block_table", prefill_module.HCA_NUM_LAYERS),
        ("csa_cmp_kv", "csa_cmp_block_table", prefill_module.CSA_NUM_LAYERS),
        ("idx_kv_cache", "idx_block_table", prefill_module.CSA_NUM_LAYERS),
        ("idx_kv_scale", "idx_block_table", prefill_module.CSA_NUM_LAYERS),
        (
            "hca_compress_state",
            "hca_compress_state_block_table",
            prefill_module.HCA_NUM_LAYERS,
        ),
        (
            "csa_compress_state",
            "csa_compress_state_block_table",
            prefill_module.CSA_NUM_LAYERS,
        ),
        (
            "csa_inner_compress_state",
            "csa_inner_compress_state_block_table",
            prefill_module.CSA_NUM_LAYERS,
        ),
    )
    for cache_name, table_name, layer_count in table_contracts:
        cache = resident.hosts[("prefill", cache_name)]
        if cache.shape[1] % layer_count:
            raise ValueError(f"{cache_name} does not split evenly across {layer_count} layers")
        physical_blocks = cache.shape[1] // layer_count
        table = prefill_io[table_name]
        referenced = table[table >= 0]
        if referenced.numel() and int(referenced.max()) >= physical_blocks:
            raise ValueError(
                f"{table_name} references block {int(referenced.max())}, "
                f"but {cache_name} has {physical_blocks} blocks per layer"
            )


def _copy_leader_request_table(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.ndim != 3 or source.ndim != 3 or target.shape[0] != source.shape[0]:
        raise ValueError("cache tables must be [R,B,blocks] tensors")
    if source.shape[1] < 1 or source.shape[2] < target.shape[2]:
        raise ValueError("Prefill cache table is narrower than the Decode table")
    for leader in range(0, target.shape[0], TP_SIZE):
        target[leader, 0].copy_(source[leader, 0, : target.shape[2]])


def _copy_group_request_table(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.ndim != 3 or source.ndim != 3 or target.shape[0] != source.shape[0]:
        raise ValueError("cache tables must be [R,B,blocks] tensors")
    if source.shape[1] < 1 or source.shape[2] < target.shape[2]:
        raise ValueError("Prefill cache table is narrower than the Decode table")
    target[:, 0].copy_(source[:, 0, : target.shape[2]])


def _refresh_target_decode_metadata(stages, decode_io, prefill_tables, start_positions) -> None:
    decode = stages["decode"].module
    utils = stages["utils"].module
    expected_groups = decode_io["input_ids"].shape[0] // TP_SIZE
    if start_positions.shape != (expected_groups,):
        raise ValueError(f"start_positions must contain {expected_groups} TP-group values")
    if bool((start_positions < 0).any()) or bool(
        (start_positions.to(torch.int64) + DECODE_WIDTH > decode.MODEL_CONFIG.max_position_embeddings).any()
    ):
        raise ValueError("Decode positions must fit the configured context window")

    for decode_name, prefill_name in _PREFILL_GROUP_STATE_TABLES.items():
        _copy_group_request_table(decode_io[decode_name], prefill_tables[prefill_name])
    for decode_name, prefill_name in _PREFILL_LOCAL_REQUEST_TABLES.items():
        _copy_leader_request_table(decode_io[decode_name], prefill_tables[prefill_name])

    for group_index, group_base in enumerate(range(0, decode_io["input_ids"].shape[0], TP_SIZE)):
        leader = group_base
        start_position = int(start_positions[group_index])
        group_start = start_positions[group_index : group_index + 1]
        positions = utils.position_ids_from_starts(group_start, seq=DECODE_WIDTH)
        kv_seq_len = utils.kv_seq_lens_from_starts(group_start, seq=DECODE_WIDTH)[0]
        ori_table = prefill_tables["ori_block_table"][leader : leader + 1, 0]
        hca_cmp_table = prefill_tables["hca_cmp_block_table"][leader : leader + 1, 0]
        csa_cmp_table = prefill_tables["csa_cmp_block_table"][leader : leader + 1, 0]
        idx_table = prefill_tables["idx_block_table"][leader : leader + 1, 0]
        hca_state_table = decode_io["hca_compress_state_block_table"][leader, :1]
        csa_state_table = decode_io["csa_compress_state_block_table"][leader, :1]
        inner_state_table = decode_io["csa_inner_compress_state_block_table"][leader, :1]

        ori_slots = utils.ori_slot_mapping(positions, ori_table).reshape(-1)
        swa_indices, swa_lens = utils.swa_indices_and_lens(positions, ori_table)
        hca_cmp_slots = utils.compressed_slot_mapping(
            positions,
            hca_cmp_table,
            compress_ratio=decode.HCA_COMPRESS_RATIO,
        ).reshape(-1)
        csa_cmp_slots = utils.compressed_slot_mapping(
            positions,
            csa_cmp_table,
            compress_ratio=decode.CSA_COMPRESS_RATIO,
        ).reshape(-1)
        csa_idx_slots = utils.compressed_slot_mapping(
            positions,
            idx_table,
            compress_ratio=decode.CSA_COMPRESS_RATIO,
        ).reshape(-1)
        hca_state_slots = utils.state_slot_mapping(
            positions,
            hca_state_table,
            state_block_size=decode.HCA_COMPRESS_STATE_BLOCK_SIZE,
        ).reshape(-1)
        csa_state_slots = utils.state_slot_mapping(
            positions.to(torch.int64) % decode.csa.MAIN_STATE_LEN,
            csa_state_table,
            state_block_size=decode.CSA_MAIN_STATE_BLOCK_SIZE,
        ).reshape(-1)
        inner_state_slots = utils.state_slot_mapping(
            positions.to(torch.int64) % decode.csa.INNER_STATE_LEN,
            inner_state_table,
            state_block_size=decode.CSA_INNER_STATE_BLOCK_SIZE,
        ).reshape(-1)
        rope_cos, rope_sin = utils.token_local_rope(
            decode.MODEL_CONFIG,
            0,
            positions,
            max_seq_len=decode.MODEL_CONFIG.max_position_embeddings,
            rope_dim=decode.ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
        )
        csa_cmp_positions = torch.where(
            (positions.to(torch.int64) + 1) % decode.CSA_COMPRESS_RATIO == 0,
            positions.to(torch.int64) - (decode.CSA_COMPRESS_RATIO - 1),
            torch.zeros_like(positions, dtype=torch.int64),
        )
        csa_cmp_cos, csa_cmp_sin = utils.token_local_rope(
            decode.MODEL_CONFIG,
            decode.CSA_COMPRESS_RATIO,
            csa_cmp_positions,
            max_seq_len=decode.MODEL_CONFIG.max_position_embeddings,
            rope_dim=decode.ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
        )
        hca_boundary = torch.tensor(
            [start_position - start_position % decode.HCA_COMPRESS_RATIO],
            dtype=torch.int64,
        )
        hca_cmp_cos, hca_cmp_sin = utils.token_local_rope(
            decode.MODEL_CONFIG,
            decode.HCA_COMPRESS_RATIO,
            hca_boundary,
            max_seq_len=decode.MODEL_CONFIG.max_position_embeddings,
            rope_dim=decode.ROPE_HEAD_DIM,
            dtype=torch.float32,
        )

        decode_io["position_ids_local"][leader].copy_(positions.reshape(-1))
        decode_io["freqs_cos_local"][leader].copy_(rope_cos)
        decode_io["freqs_sin_local"][leader].copy_(rope_sin)
        decode_io["swa_indices"][leader].copy_(swa_indices)
        decode_io["swa_lens"][leader].copy_(swa_lens)
        # Replace only the real request's rank-major group segment.  The other
        # ranks retain the fixture's valid dummy requests because Decode has no
        # inactive-query mask and every TP rank must execute its local rows.
        for rank in range(group_base, group_base + TP_SIZE):
            decode_io["position_ids"][rank, :DECODE_WIDTH].copy_(positions.reshape(-1))
            decode_io["freqs_cos"][rank, :DECODE_WIDTH].copy_(rope_cos)
            decode_io["freqs_sin"][rank, :DECODE_WIDTH].copy_(rope_sin)
            decode_io["csa_cmp_freqs_cos"][rank, :DECODE_WIDTH].copy_(csa_cmp_cos)
            decode_io["csa_cmp_freqs_sin"][rank, :DECODE_WIDTH].copy_(csa_cmp_sin)
            decode_io["hca_cmp_freqs_cos"][rank, 0].copy_(
                hca_cmp_cos[0, : decode.ROPE_HEAD_DIM // 2]
            )
            decode_io["hca_cmp_freqs_sin"][rank, 0].copy_(
                hca_cmp_sin[0, : decode.ROPE_HEAD_DIM // 2]
            )
            decode_io["swa_slot_mapping"][rank, :DECODE_WIDTH].copy_(ori_slots)
            decode_io["csa_ori_slot_mapping"][rank, :DECODE_WIDTH].copy_(ori_slots)
            decode_io["hca_ori_slot_mapping"][rank, :DECODE_WIDTH].copy_(ori_slots)
            decode_io["csa_cmp_slot_mapping"][rank, :DECODE_WIDTH].copy_(csa_cmp_slots)
            decode_io["csa_idx_slot_mapping"][rank, :DECODE_WIDTH].copy_(csa_idx_slots)
            decode_io["csa_state_slot_mapping"][rank, :DECODE_WIDTH].copy_(csa_state_slots)
            decode_io["csa_inner_state_slot_mapping"][rank, :DECODE_WIDTH].copy_(
                inner_state_slots
            )
            decode_io["hca_cmp_slot_mapping"][rank, :DECODE_WIDTH].copy_(hca_cmp_slots)
            decode_io["hca_state_slot_mapping"][rank, :DECODE_WIDTH].copy_(hca_state_slots)
        decode_io["csa_window_swa_indices"][leader].copy_(swa_indices)
        decode_io["hca_window_swa_indices"][leader].copy_(swa_indices)
        decode_io["csa_window_swa_lens"][leader].copy_(swa_lens)
        decode_io["hca_window_swa_lens"][leader].copy_(swa_lens)
        decode_io["csa_kv_seq_lens"][leader, 0] = kv_seq_len
        decode_io["hca_kv_seq_lens"][leader, 0] = kv_seq_len


def _prefill_next_tokens(prefill_io, tp_size: int = TP_SIZE) -> torch.Tensor:
    sampled = prefill_io["sampled_ids"][:, 0, 0].to(torch.int64)
    tokens = torch.zeros_like(sampled)
    tokens[::tp_size].copy_(sampled[::tp_size])
    return tokens


def _prepare_drafter_prefill(drafter, io, prefill_io, next_tokens) -> None:
    target_hidden = prefill_io["dspark_target_hidden"]
    if target_hidden.shape != io["target_hidden"].shape:
        raise ValueError(
            f"Prefill handoff shape {tuple(target_hidden.shape)} does not match drafter "
            f"local shape {tuple(io['target_hidden'].shape)}"
        )
    expected_group_rows = TP_SIZE * target_hidden.shape[1]
    if io["context_group_position_ids"].shape[1] != expected_group_rows or (
        io["context_group_slot_mapping"].shape[2] != expected_group_rows
    ):
        raise ValueError("Drafter Prefill context metadata does not match local target rows")
    io["target_hidden"].copy_(target_hidden)
    io["num_sampled"].zero_()
    io["last_sampled"].zero_()
    io["next_prefill_tokens"].zero_()
    io["next_prefill_tokens"][::TP_SIZE, 0].copy_(next_tokens[::TP_SIZE])
    io["anchor_positions"][::TP_SIZE, 0] = expected_group_rows - 1
    _refresh_drafter_query_metadata(io, drafter.BLOCK_SIZE)


def _prepare_markov_io(
    io,
    drafter_io,
    num_sampled,
    last_sampled,
    next_tokens,
) -> None:
    io["head_hidden"].copy_(drafter_io["head_hidden"])
    io["num_sampled"].copy_(num_sampled)
    io["last_sampled"].copy_(last_sampled)
    io["next_prefill_tokens"].zero_()
    io["next_prefill_tokens"][:, 0].copy_(next_tokens)
    io["draft_token_ids"].fill_(-1)
    io["confidence_probs"].fill_(float("nan"))


def _active_drafts(markov_io):
    drafts = markov_io["draft_token_ids"][:, 0].to(torch.int64)
    if int(drafts[::TP_SIZE].min()) < 0:
        raise AssertionError("Markov left an active draft token unwritten")
    active = torch.zeros_like(drafts)
    active[::TP_SIZE].copy_(drafts[::TP_SIZE])
    return active


def _prepare_target_decode(
    stages,
    io,
    prefill_tables,
    start_positions,
    current_tokens,
    drafts,
) -> None:
    if io["input_ids"].shape[1] != DECODE_WIDTH:
        raise ValueError(f"target Decode must expose {DECODE_WIDTH} rows per owner")
    io["input_ids"].zero_()
    io["input_ids"][::TP_SIZE, 0].copy_(current_tokens[::TP_SIZE])
    io["input_ids"][::TP_SIZE, 1:DECODE_WIDTH].copy_(drafts[::TP_SIZE])
    _refresh_target_decode_metadata(stages, io, prefill_tables, start_positions)
    io["logit_row_indices"].fill_(-1)
    io["logit_row_indices"][::TP_SIZE, :DECODE_WIDTH].copy_(
        torch.arange(DECODE_WIDTH, dtype=torch.int32)
    )
    io["dspark_target_hidden"].fill_(float("nan"))
    io["sampled_ids"].fill_(-1)


def _prepare_drafter_decode(stages, io, verified) -> None:
    drafter = stages["drafter"]
    io["target_hidden"].copy_(verified.target_hidden)
    local_slots = _context_slots(
        io["block_tables"],
        verified.context_positions,
        verified.valid_rows,
        drafter.module.BLOCK_SIZE,
    )
    _replicate_group_metadata(
        verified.context_positions,
        local_slots,
        io["context_group_position_ids"],
        io["context_group_slot_mapping"],
    )
    io["num_sampled"].zero_()
    io["num_sampled"][:, 0].copy_(verified.accepted_counts)
    io["last_sampled"].zero_()
    io["last_sampled"][:, 0].copy_(verified.next_tokens)
    io["anchor_positions"][::TP_SIZE, 0].copy_(verified.anchors[::TP_SIZE])
    _refresh_drafter_query_metadata(io, drafter.module.BLOCK_SIZE)


def _run_stage(runtime, compiled, io, resident_handles, config, stage: str) -> None:
    _assert_shared_tensors(io, stage)
    _assert_program_args(compiled, io, resident_handles, stage)
    runtime.run(
        compiled,
        *_ordered_args(compiled, io, resident_handles),
        config=config,
    )


def _run_session(args, runtime_dirs, model_dir: Path) -> None:
    from pypto.backend import BackendType
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram, DistributedConfig
    from pypto.runtime import DistributedWorker, RunConfig

    distributed_config = DistributedConfig(
        device_ids=args.device_ids[: args.ep],
        num_sub_workers=0,
    )
    compiled = {
        name: DistributedCompiledProgram.from_dir(
            path,
            platform=args.platform,
            backend_type=BackendType.Ascend950,
            distributed_config=distributed_config,
        )
        for name, path in runtime_dirs.items()
    }
    stages = _load_model_specs(model_dir, args.ep, args.prefill_tokens)
    resident = _prepare_resident_hosts(stages, compiled)
    _constrain_target_lm_head(resident)

    prefill_io = _materialize_stage(stages["prefill"].specs, resident.names["prefill"])
    drafter_prefill_io, drafter_decode_io = _materialize_drafter_ios(
        stages["drafter"],
        resident.names["drafter"],
    )
    markov_io = _materialize_stage(stages["markov"].specs, resident.names["markov"])
    max_decode_start = args.prefill_tokens + (args.decode_steps - 1) * DECODE_WIDTH
    if max_decode_start + DECODE_WIDTH > stages["decode"].module.MODEL_CONFIG.max_position_embeddings:
        raise ValueError("requested Decode rounds exceed the configured context window")
    decode_specs = stages["decode"].module.build_tensor_specs(
        start_pos=max_decode_start,
        weight_bank_size=1,
        runtime_case="full_active",
    )
    decode_io = _materialize_stage(
        decode_specs,
        resident.names["decode"],
        excluded_names=frozenset(_TARGET_SHARED_IO_DIRECTIONS),
    )
    _reuse_target_terminal_io(stages, compiled, prefill_io, decode_io)
    _assert_prefill_cache_tables_fit(prefill_io, resident, stages["prefill"].module)
    for io in (
        prefill_io,
        drafter_prefill_io,
        drafter_decode_io,
        markov_io,
        decode_io,
    ):
        _share_tensors(io)
    inherited = _unique_storage_tensors(resident.hosts.values())

    run_config_parameters = inspect.signature(RunConfig).parameters
    if "enable_chip_swimlane" in run_config_parameters:
        swimlane = {"enable_chip_swimlane": False}
    elif "enable_l2_swimlane" in run_config_parameters:
        swimlane = {"enable_l2_swimlane": False}
    else:
        raise TypeError("RunConfig exposes no supported swimlane switch")

    configs = {
        "prefill": RunConfig(
            platform=args.platform,
            device_id=0,
            backend_type=BackendType.Ascend950,
            ring_heap=stages["prefill"].module.PREFILL_RING_HEAP,
            **swimlane,
        ),
        "decode": RunConfig(
            platform=args.platform,
            device_id=0,
            backend_type=BackendType.Ascend950,
            ring_heap=stages["decode"].module.DECODE_RING_HEAP,
            **swimlane,
        ),
        "drafter": RunConfig(
            platform=args.platform,
            device_id=0,
            backend_type=BackendType.Ascend950,
            ring_heap=stages["drafter"].module._DSPARK_RING_HEAP,
            **swimlane,
        ),
        "markov": RunConfig(
            platform=args.platform,
            device_id=0,
            backend_type=BackendType.Ascend950,
            ring_heap=stages["markov"].module.LM_HEAD_RING_HEAP,
            **swimlane,
        ),
    }

    runtime = None
    handles = {}
    allocated = []
    try:
        runtime = DistributedWorker(
            list(compiled.values()),
            config=configs["prefill"],
            persistent=True,
            reset_persistent_windows=False,
            inherited_host_tensors=inherited,
        )
        handles, allocated, stage_handles = _allocate_residents(runtime, resident)
        runtime.release_inherited_host_tensor_refs()
        inherited.clear()
        resident.hosts.clear()
        gc.collect()

        _run_stage(
            runtime,
            compiled["prefill"],
            prefill_io,
            stage_handles["prefill"],
            configs["prefill"],
            "target prefill",
        )
        _require_finite_nonzero(prefill_io["dspark_target_hidden"], "target prefill hidden")
        current_tokens = _prefill_next_tokens(prefill_io)
        decode_vocab = _spec_map(stages["decode"].specs)["embed_weight"].shape[1]
        if int(current_tokens.min()) < 0 or int(current_tokens.max()) >= decode_vocab:
            raise AssertionError(f"Prefill sample is outside Decode fixture vocab 0..{decode_vocab - 1}")
        prefill_tables = {
            name: prefill_io[name]
            for name in _PREFILL_HANDOFF_TABLE_NAMES
        }

        _prepare_drafter_prefill(
            stages["drafter"].module,
            drafter_prefill_io,
            prefill_io,
            current_tokens,
        )
        _run_stage(
            runtime,
            compiled["drafter"],
            drafter_prefill_io,
            stage_handles["drafter"],
            configs["drafter"],
            "drafter prefill",
        )
        _require_finite_nonzero(
            drafter_prefill_io["head_hidden"][::TP_SIZE, 0],
            "drafter prefill head hidden",
        )

        _prepare_markov_io(
            markov_io,
            drafter_prefill_io,
            drafter_prefill_io["num_sampled"],
            drafter_prefill_io["last_sampled"],
            current_tokens,
        )
        _run_stage(
            runtime,
            compiled["markov"],
            markov_io,
            stage_handles["markov"],
            configs["markov"],
            "Markov after prefill",
        )
        _require_finite_nonzero(
            markov_io["confidence_probs"][::TP_SIZE, 0],
            "Markov confidence",
        )

        start_positions = torch.full(
            (args.ep // TP_SIZE,),
            args.prefill_tokens,
            dtype=torch.int32,
        )
        acceptance_history = []
        for step in range(args.decode_steps):
            drafts = _active_drafts(markov_io)
            if int(drafts.max()) >= decode_vocab:
                raise AssertionError(f"Markov draft exceeds Decode fixture vocab {decode_vocab}")
            _prepare_target_decode(
                stages,
                decode_io,
                prefill_tables,
                start_positions,
                current_tokens,
                drafts,
            )
            _run_stage(
                runtime,
                compiled["decode"],
                decode_io,
                stage_handles["decode"],
                configs["decode"],
                f"target decode[{step}]",
            )
            _require_finite_nonzero(
                decode_io["dspark_target_hidden"][::TP_SIZE],
                f"target decode[{step}] owner hidden",
            )
            target_samples = decode_io["sampled_ids"][:, :DECODE_WIDTH, 0].to(torch.int64)
            active_samples = target_samples[::TP_SIZE]
            if int(active_samples.min()) < 0 or int(active_samples.max()) >= decode_vocab:
                raise AssertionError(f"target decode[{step}] produced an invalid sample")
            verified = _verify_target_step(
                drafts,
                target_samples,
                decode_io["dspark_target_hidden"],
                start_positions,
            )
            group_counts = verified.accepted_counts[::TP_SIZE]
            acceptance_history.append(group_counts.tolist())
            start_positions = verified.anchors[::TP_SIZE] + 1
            current_tokens = verified.next_tokens

            _prepare_drafter_decode(
                stages,
                drafter_decode_io,
                verified,
            )
            _run_stage(
                runtime,
                compiled["drafter"],
                drafter_decode_io,
                stage_handles["drafter"],
                configs["drafter"],
                f"drafter decode[{step}]",
            )
            _require_finite_nonzero(
                drafter_decode_io["head_hidden"][::TP_SIZE, 0],
                f"drafter decode[{step}] head hidden",
            )
            _prepare_markov_io(
                markov_io,
                drafter_decode_io,
                drafter_decode_io["num_sampled"],
                drafter_decode_io["last_sampled"],
                current_tokens,
            )
            del verified
            _run_stage(
                runtime,
                compiled["markov"],
                markov_io,
                stage_handles["markov"],
                configs["markov"],
                f"Markov after decode[{step}]",
            )
            _require_finite_nonzero(
                markov_io["confidence_probs"][::TP_SIZE, 0],
                f"Markov confidence[{step}]",
            )

        print(f"[DSPARK SESSION] accepted_counts={acceptance_history}", flush=True)
        print(
            f"[DSPARK SESSION] PASS: DP{args.ep // TP_SIZE} x TP{TP_SIZE}, "
            f"MoE EP{args.ep}, "
            f"prefill + {args.decode_steps} speculative Decode rounds",
            flush=True,
        )
    finally:
        if runtime is not None:
            for key in reversed(allocated):
                try:
                    runtime.free_stacked_tensor(handles[key])
                except Exception as error:
                    print(f"[DSPARK SESSION] warning: failed to free {key}: {error}", flush=True)
            runtime.close()


def _device_ids(raw: str) -> list[int]:
    try:
        devices = [int(value) for value in raw.split(",") if value]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--device must be a comma-separated integer list") from error
    if len(set(devices)) != len(devices) or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("device IDs must be distinct and non-negative")
    return devices


def _internal_compile_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--internal-compile", action="store_true")
    parser.add_argument("--stage", choices=("prefill", "decode", "drafter", "markov"), required=True)
    parser.add_argument("--platform", choices=("a2a3",), required=True)
    parser.add_argument("--ep", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--prefill-tokens", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    args.device_ids = _device_ids(args.device)
    if len(args.device_ids) != args.ep:
        parser.error(f"EP{args.ep} requires exactly {args.ep} devices")
    _compile_stage_child(args)


def main() -> None:
    if "--internal-compile" in sys.argv:
        _internal_compile_main(sys.argv[1:])
        return

    parser = argparse.ArgumentParser(
        description="Validate target Prefill -> DSpark -> repeated target verification and DSpark Decode.",
    )
    parser.add_argument("-p", "--platform", choices=("a2a3",), default="a2a3")
    parser.add_argument("--ep", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--tp", type=int, choices=(TP_SIZE,), default=TP_SIZE)
    parser.add_argument("-d", "--device", default="0,1,2,3")
    parser.add_argument("--prefill-tokens", type=int, choices=(PREFILL_TOKENS,), default=PREFILL_TOKENS)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--prefill-runtime-dir", type=Path)
    parser.add_argument("--decode-runtime-dir", type=Path)
    parser.add_argument("--drafter-runtime-dir", type=Path)
    parser.add_argument("--markov-runtime-dir", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.device_ids = _device_ids(args.device)
    if len(args.device_ids) != args.ep:
        parser.error(f"EP{args.ep} requires exactly {args.ep} devices, got {args.device!r}")
    if args.decode_steps < 2:
        parser.error("--decode-steps must be at least 2 for state-continuity validation")

    model_dir = Path(__file__).resolve().parent
    runtime_dirs = {}
    for stage in ("prefill", "decode", "drafter", "markov"):
        runtime_dir = getattr(args, f"{stage}_runtime_dir")
        runtime_dirs[stage] = runtime_dir.resolve() if runtime_dir else _compile_stage(stage, args)
        print(f"[DSPARK SESSION] {stage}_runtime_dir={runtime_dirs[stage]}", flush=True)
    if args.compile_only:
        return
    _run_session(args, runtime_dirs, model_dir)


if __name__ == "__main__":
    main()
