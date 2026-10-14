# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""Run an EP2/EP4/EP8 prefill-to-decode token session on one worker.

By default the session uses synthetic zero-valued model weights. It validates
the serving control/data path rather than model numerics:

``token ids -> embedding -> full prefill -> LM head/sample -> repeated decode``.

With ``--weights`` (an HF checkpoint dir or a ``utils.py`` .pt cache
matching ``--ep``/``--tp``) the resident bank is instead materialized from the
drivers' own TensorSpecs: the 57 real-weight names come from the checkpoint via
``utils.apply_real_weights`` and the remaining architectural constants
(RoPE tables, the indexer Hadamard) keep their fixture initializers, so the
loop generates with real DeepSeek-V4-Flash numerics. ``--prompt``/
``--prompt-file`` then feed a natural-language prompt through the checkpoint's
``tokenizer.json`` (up to ``PROMPT_TOKENS`` ids; ``num_tokens`` and the logit
row track the real prompt length), and the sampled ids are detokenized at the
end. NOTE: only EP8 deploys the full 256-expert model; an EP2/EP4 real-weight
run uses the first ``32*EP`` experts with reduced router tables — a smoke
configuration whose generations do not represent true model output.

Prefill and decode are compiled in separate processes because their static MoE
token extents differ. The resulting programs are then loaded into one
``DistributedWorker`` and share every resident weight and cache handle. Decode
metadata is refreshed in place for each sampled token while the cache handles
remain resident on device.
"""

from __future__ import annotations

import argparse
import ast
import gc
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import torch


PROMPT_TOKENS = 128
EP_SIZE = 2
TP_SIZE = 2
# Ring 0 retains the full prefill hidden states (521 MiB for Flash's 43 layers).
# Ring 1 holds attention and dispatch scratch; 256 MiB cannot cover ring wrap
# while the first MoE allocation is still live. Rings 2/3 hold routed expert
# outputs and MXFP4 unpack intermediates until their nested scopes close.
PREFILL_RING_HEAP = (1024 * 1024 * 1024, 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)
_SSA_SUFFIX = re.compile(r"__ssa_v\d+$")


def _latency_summary(samples):
    """Summarize milliseconds using linearly interpolated sample percentiles."""
    if not samples:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p90_ms": None}
    ordered = sorted(samples)

    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(samples),
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": percentile(0.5),
        "p90_ms": percentile(0.9),
    }


def _new_result(args):
    """Use an explicit field list: never serialize the process environment."""
    return {
        "schema_version": 1,
        "status": "running",
        "mode": "real-weight" if args.weights is not None else "synthetic",
        "platform": args.platform,
        "variant": args.variant,
        "ep": args.ep,
        "tp": args.tp,
        "tp_scope": "lm_head_vocabulary",
        "devices": [int(device) for device in args.device.split(",")],
        "weights": str(args.weights) if args.weights is not None else None,
        "tokenizer": str(args.tokenizer) if args.tokenizer is not None else None,
        "active_sequences": 1,
        "rank_semantics": "EP/TP ranks cooperate on one sequence; ranks are not requests",
        "prompt_tokens": None,
        "prompt_token_ids": None,
        "decode_steps_requested": args.decode_steps,
        "generated_ids": [],
        "generated_text": None,
        "eos_id": args.eos_id,
        "stop_reason": None,
        "validation": "finite logits, greedy argmax and sampled-token agreement across ranks",
        # Every rank holds the full hidden state and full vocabulary, so the
        # ranks should be bit-identical. Token agreement alone cannot show that,
        # so the per-sampling-point spread is recorded next to it.
        "rank_spread": [],
        "ranks_bit_identical": None,
        "warmup": {"prefill_runs": 0, "decode_steps_discarded": 0},
        "measurement_scope": {
            "clock": "time.perf_counter host wall clock",
            "dispatch": "DistributedWorker.run returns after all ranks complete",
            "ttft": "resident prefill dispatch start through first validated sampled token",
            "decode_step": "previous validated token through input refresh, all-rank completion and next validation",
            "generation": "resident prefill dispatch start through last validated sampled token",
            "excluded_from_token_rates": "prompt preparation, compilation, model loading and teardown",
            "warmup": "none; first prefill and every successful decode step are included",
            "rank_spread": "scanned after the step is timed; excluded from every latency figure",
            "invocation": "after argument validation through cleanup; excludes interpreter startup and imports",
        },
        "timings_ms": {
            "prompt_prepare": None,
            "compile_prefill": 0.0,
            "compile_decode": 0.0,
            "load_artifacts": None,
            "prepare_host": None,
            "prepare_worker": None,
            "upload_resident": None,
            "load_total": None,
            "prefill_dispatch": None,
            "ttft": None,
            "decode_step_wall": [],
            "decode_dispatch": [],
            "generation_wall": None,
            "invocation_wall": None,
        },
        "reused_artifacts": {
            "prefill": args.prefill_runtime_dir is not None,
            "decode": args.decode_runtime_dir is not None,
        },
    }


def _finish_result(result):
    """Count the prefill token once and never count replicated rank outputs."""
    spread = result.get("rank_spread") or []
    result["ranks_bit_identical"] = (
        all(entry.get("bit_identical") for entry in spread) if spread else None
    )
    timings = result["timings_ms"]
    decode_samples = timings["decode_step_wall"]
    # Recover a partial continuation after a failed dispatch when possible.
    if result["generated_text"] is None and result["generated_ids"] and result["tokenizer"]:
        try:
            from tokenizers import Tokenizer

            result["generated_text"] = Tokenizer.from_file(result["tokenizer"]).decode(result["generated_ids"])
        except Exception:  # noqa: BLE001 - token ids remain available if decoding fails
            pass
    output_tokens = len(result["generated_ids"])
    decode_tokens = len(decode_samples)
    result["output_tokens_per_sequence"] = output_tokens
    result["output_tokens_total"] = output_tokens * result["active_sequences"]
    result["output_token_count_includes_eos"] = True
    result["decode_steps_done"] = decode_tokens
    result["decode_latency"] = _latency_summary(decode_samples)
    result["decode_dispatch_latency"] = _latency_summary(timings["decode_dispatch"])

    def rate(tokens, elapsed_ms):
        return tokens * 1000.0 / elapsed_ms if tokens and elapsed_ms and elapsed_ms > 0 else None

    decode_rate = rate(decode_tokens, sum(decode_samples))
    generation_rate = rate(output_tokens, timings["generation_wall"])
    sequences = result["active_sequences"]
    result["throughput"] = {
        "decode_tokens_per_second_per_sequence": decode_rate,
        "decode_tokens_per_second_aggregate": decode_rate * sequences if decode_rate is not None else None,
        "generation_tokens_per_second_per_sequence": generation_rate,
        "generation_tokens_per_second_aggregate": (
            generation_rate * sequences if generation_rate is not None else None
        ),
    }


def _write_result(path, result):
    """Atomically replace a requested result file, including failed-run results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalized_name(name):
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


def _runtime_shape(name, info, vocab_size):
    from pypto.ir.compiled_program import _to_runtime_shape

    if info.shape is None:
        raise ValueError(f"{name} is a scalar, not a tensor")
    logical_shape = list(info.shape)
    if any(dim < 0 for dim in logical_shape):
        if name != "embed_weight" or logical_shape.count(-1) != 1:
            raise ValueError(f"unsupported dynamic shape for {name}: {logical_shape}")
        logical_shape[logical_shape.index(-1)] = vocab_size
    return tuple(_to_runtime_shape(logical_shape, info.dtype))


def _empty_host_tensor(name, info, vocab_size):
    from pypto.ir.compiled_program import _to_torch_dtype

    dtype = _to_torch_dtype(info.dtype)
    if dtype is None:
        raise TypeError(f"unsupported runtime dtype for {name}: {info.dtype}")
    return torch.zeros(_runtime_shape(name, info, vocab_size), dtype=dtype)


def _ordered_args(compiled, io_tensors, resident_handles, scalars):
    ordered = []
    for name, info in _metadata(compiled):
        if info.shape is None:
            ordered.append(scalars[name])
        elif name in resident_handles:
            ordered.append(resident_handles[name])
        else:
            ordered.append(io_tensors[name])
    return ordered


def _require_scalar(compiled, name, program, dtype):
    info = _info_map(compiled).get(name)
    if info is None:
        raise ValueError(f"{program} artifact is missing scalar {name!r}; recompile it with this source")
    if info.shape is not None:
        raise ValueError(f"{program} artifact parameter {name!r} is not a scalar")

    from pypto.ir.compiled_program import _to_torch_dtype

    artifact_dtype = _to_torch_dtype(info.dtype)
    if artifact_dtype != dtype:
        raise TypeError(f"{program} artifact scalar {name!r} has dtype {artifact_dtype}, expected {dtype}")


def _is_tensor_lookup(node, artifact_name):
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "tensors"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == artifact_name
    )


def _host_orch_forwards_scalar(host_orch, artifact_name):
    """Return whether ``host_orch.py`` passes one metadata scalar to TaskArgs."""
    tree = ast.parse(host_orch.read_text(), filename=str(host_orch))
    runtime_vars = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not _is_tensor_lookup(node.value, artifact_name):
            continue
        for target in node.targets:
            runtime_vars.update(child.id for child in ast.walk(target) if isinstance(child, ast.Name))

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_scalar"
        ):
            continue
        for argument in node.args:
            if any(_is_tensor_lookup(child, artifact_name) for child in ast.walk(argument)):
                return True
            if any(isinstance(child, ast.Name) and child.id in runtime_vars for child in ast.walk(argument)):
                return True
    return False


def _require_runtime_scalar(compiled, name, program, dtype):
    """Validate both the scalar ABI and its generated host-side forwarding."""
    _require_scalar(compiled, name, program, dtype)
    artifact_name = _info_map(compiled)[name].name
    host_orch = Path(compiled.output_dir) / "orchestration" / "host_orch.py"
    if not host_orch.is_file():
        raise ValueError(f"{program} artifact is missing generated host orchestration: {host_orch}")
    try:
        forwarded = _host_orch_forwards_scalar(host_orch, artifact_name)
    except (OSError, SyntaxError) as error:
        raise ValueError(f"cannot inspect {program} artifact host orchestration: {host_orch}") from error
    if not forwarded:
        raise ValueError(
            f"{program} artifact scalar {name!r} is not forwarded through "
            "TaskArgs.add_scalar; it was likely constant-specialized. Recompile "
            "with ScalarSpec(..., compile_runtime=True) and this source"
        )


def _create_persistent_worker(worker_type, compiled, config, inherited):
    return worker_type(
        compiled, config=config,
        persistent=True, reset_persistent_windows=False,
        inherited_host_tensors=inherited,
    )


def _compile_program(model_dir, entrypoint, args, num_tokens, start_pos):
    command = [
        sys.executable,
        str(model_dir / entrypoint),
        "--variant",
        args.variant,
        "-p",
        args.platform,
        "--ep",
        str(args.ep),
        "--tp",
        str(args.tp),
        "-d",
        args.device,
        "--num-tokens",
        str(num_tokens),
        "--start-pos",
        str(start_pos),
        "--compile-only",
    ]
    env = os.environ.copy()
    env["DEEPSEEK_V4_VARIANT"] = args.variant
    repo_root = model_dir.parents[1]
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}:{old_pythonpath}" if old_pythonpath else str(repo_root)
    )

    print(f"[SESSION] compiling {entrypoint}", flush=True)
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
        match = re.match(r"\[RUN\] runtime_dir=(.+)\s*$", line)
        if match:
            runtime_dir = Path(match.group(1)).resolve()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{entrypoint} compile failed with exit code {return_code}")
    if runtime_dir is None:
        raise RuntimeError(f"{entrypoint} did not report its runtime directory")
    return runtime_dir


def _purge_model_modules(model_dir):
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


def _initialize_tid2eid(tensor, *, vocab_size, topk, num_experts):
    world_size, packed_rows, packed_topk = tensor.shape
    if packed_topk != topk or packed_rows % vocab_size != 0:
        raise ValueError(f"unexpected tid2eid shape: {tuple(tensor.shape)}")
    num_layers = packed_rows // vocab_size
    token_ids = torch.arange(vocab_size, dtype=torch.int32).reshape(vocab_size, 1)
    topk_ids = torch.arange(topk, dtype=torch.int32).reshape(1, topk)
    base = token_ids * topk + topk_ids
    for layer in range(num_layers):
        start = layer * vocab_size
        tensor[0, start : start + vocab_size].copy_(
            (base + layer * topk).remainder(num_experts)
        )
    for rank in range(1, world_size):
        tensor[rank].copy_(tensor[0])


def _materialize_spec_value(spec):
    value = spec.init_value
    if callable(value):
        value = value()
    return value


def _build_resident_hosts(prefill, prefill_compiled, decode_compiled, weight_specs=None):
    resident_names = set(prefill.RESIDENT_WEIGHT_NAMES) | set(
        prefill.RESIDENT_CACHE_NAMES
    )
    prefill_infos = _info_map(prefill_compiled)
    decode_infos = _info_map(decode_compiled)
    missing = resident_names - prefill_infos.keys()
    if missing:
        raise ValueError(f"prefill artifact is missing resident parameters: {sorted(missing)}")
    if resident_names - decode_infos.keys():
        raise ValueError("decode artifact is missing parameters shared with prefill")

    for name in sorted(resident_names):
        prefill_info = prefill_infos[name]
        decode_info = decode_infos[name]
        prefill_abi = (
            prefill_info.shape,
            str(prefill_info.dtype),
            str(prefill_info.direction),
        )
        decode_abi = (
            decode_info.shape,
            str(decode_info.dtype),
            str(decode_info.direction),
        )
        if prefill_abi != decode_abi:
            raise ValueError(
                f"resident ABI mismatch for {name}: {prefill_abi} != {decode_abi}"
            )

    from pypto.ir.compiled_program import _to_torch_dtype

    resident_hosts = {}
    total_bytes = 0
    for index, name in enumerate(sorted(resident_names), start=1):
        info = prefill_infos[name]
        spec = weight_specs.get(name) if weight_specs is not None else None
        if (
            spec is not None
            and spec.init_value is not None
            and name not in prefill.RESIDENT_CACHE_NAMES
        ):
            # Real/fixture weight: materialize the driver's own spec init (the
            # checkpoint value after apply_real_weights, or the architectural
            # fixture for names like the RoPE tables and the Hadamard index).
            tensor = _materialize_spec_value(spec)
            expected_shape = _runtime_shape(name, info, prefill.MODEL_CONFIG.vocab_size)
            expected_dtype = _to_torch_dtype(info.dtype)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name}: spec init shape {tuple(tensor.shape)} does not "
                    f"match runtime ABI {expected_shape}"
                )
            if tensor.dtype != expected_dtype:
                # Fixture inits may build in a wider dtype (e.g. fp32) than the
                # kernel ABI declares; run casts on materialization too.
                tensor = tensor.to(expected_dtype)
        else:
            tensor = _empty_host_tensor(name, info, prefill.MODEL_CONFIG.vocab_size)
            if name == "mxfp4_pair_lut":
                from utils import build_mxfp4_pair_lut

                tensor.copy_(build_mxfp4_pair_lut().unsqueeze(0).expand_as(tensor))
            if name == "tid2eid" and weight_specs is None:
                _initialize_tid2eid(
                    tensor,
                    vocab_size=prefill.VOCAB,
                    topk=prefill.TOPK,
                    num_experts=prefill.N_EXPERTS_GLOBAL,
                )
        resident_hosts[name] = tensor.contiguous()
        total_bytes += tensor.numel() * tensor.element_size()
        print(
            f"[SESSION] host resident {index}/{len(resident_names)}: {name}",
            flush=True,
        )
    print(f"[SESSION] resident host bank: {total_bytes / 2**30:.2f} GiB", flush=True)
    return resident_hosts


def _build_io(compiled, resident_names, vocab_size):
    io_tensors = {}
    scalars = {}
    for name, info in _metadata(compiled):
        if info.shape is None:
            from pypto.ir.compiled_program import _to_torch_dtype

            dtype = _to_torch_dtype(info.dtype)
            if dtype is None:
                raise TypeError(f"unsupported scalar dtype for {name}: {info.dtype}")
            scalars[name] = torch.tensor(0, dtype=dtype)
        elif name not in resident_names:
            io_tensors[name] = _empty_host_tensor(name, info, vocab_size)
    return io_tensors, scalars


def _session_table(batch, table_blocks, physical_blocks):
    # This synthetic control-path fixture intentionally aliases logical blocks
    # when its compact physical cache has fewer rows than the logical table.
    row = torch.arange(table_blocks, dtype=torch.int32).remainder(physical_blocks)
    return row.unsqueeze(0).expand(batch, -1).contiguous()


def _build_session_tables(decode):
    return {
        "block_table": _session_table(
            decode.B, decode.ORI_TABLE_MAX_BLOCKS, decode.ORI_MAX_BLOCKS
        ),
        "cmp_block_table": _session_table(
            decode.B, decode.CSA_CMP_MAX_BLOCKS, decode.CSA_CMP_BLOCK_NUM
        ),
        "idx_block_table": _session_table(
            decode.B,
            decode.CSA_IDX_CACHE_MAX_BLOCKS,
            decode.CSA_IDX_CACHE_BLOCK_NUM,
        ),
        "hca_compress_state_block_table": _session_table(
            decode.B,
            decode.HCA_COMPRESS_STATE_MAX_BLOCKS,
            decode.HCA_COMPRESS_STATE_BLOCK_NUM,
        ),
        "csa_compress_state_block_table": _session_table(
            decode.B,
            decode.CSA_MAIN_STATE_MAX_BLOCKS,
            decode.CSA_MAIN_STATE_BLOCK_NUM,
        ),
        "csa_inner_compress_state_block_table": _session_table(
            decode.B,
            decode.CSA_INNER_STATE_MAX_BLOCKS,
            decode.CSA_INNER_STATE_BLOCK_NUM,
        ),
    }


def _ranked_prefill(value, num_ranks):
    flat = value.reshape(-1)
    return flat.unsqueeze(0).expand(num_ranks, -1).contiguous()


def _prefill_prompt_row(vocab, prompt_ids, synthetic_prompt_len):
    ids_row = torch.zeros(PROMPT_TOKENS, dtype=torch.int64)
    if prompt_ids is None:
        prompt_len = synthetic_prompt_len
        ids_row[:prompt_len] = torch.arange(prompt_len, dtype=torch.int64).remainder(vocab)
    else:
        prompt_len = int(prompt_ids.numel())
        ids_row[:prompt_len] = prompt_ids.to(torch.int64)
    return ids_row, prompt_len


def _initialize_prefill_io(
    prefill, decode_metadata,
    io_tensors, scalars, tables,
    prompt_ids=None, synthetic_prompt_len=PROMPT_TOKENS,
):
    positions = torch.arange(PROMPT_TOKENS, dtype=torch.int32).reshape(1, -1)
    table_aliases = {
        "ori_block_table": "block_table",
        "cmp_block_table": "cmp_block_table",
        "idx_block_table": "idx_block_table",
        "hca_compress_state_block_table": "hca_compress_state_block_table",
        "csa_compress_state_block_table": "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table": "csa_inner_compress_state_block_table",
    }
    for destination, source in table_aliases.items():
        io_tensors[destination].copy_(
            _ranked_prefill(tables[source][0:1], prefill.N_RANKS)
        )

    io_tensors["position_ids"].copy_(
        _ranked_prefill(positions, prefill.N_RANKS)
    )
    ids_row, prompt_len = _prefill_prompt_row(prefill.VOCAB, prompt_ids, synthetic_prompt_len)
    io_tensors["input_ids"].copy_(_ranked_prefill(ids_row, prefill.N_RANKS))

    row = {name: table[0:1] for name, table in tables.items()}
    io_tensors["ori_slot_mapping"].copy_(
        _ranked_prefill(
            decode_metadata.ori_slot_mapping(
                positions, row["block_table"], block_size=prefill.BLOCK_SIZE
            ),
            prefill.N_RANKS,
        )
    )
    compressed = (
        ("hca_cmp_slot_mapping", "cmp_block_table", prefill.HCA_COMPRESS_RATIO),
        ("csa_cmp_slot_mapping", "cmp_block_table", prefill.CSA_COMPRESS_RATIO),
        ("csa_idx_slot_mapping", "idx_block_table", prefill.CSA_COMPRESS_RATIO),
    )
    for name, table_name, ratio in compressed:
        io_tensors[name].copy_(
            _ranked_prefill(
                decode_metadata.compressed_slot_mapping(
                    positions,
                    row[table_name],
                    compress_ratio=ratio,
                    block_size=prefill.BLOCK_SIZE,
                ),
                prefill.N_RANKS,
            )
        )

    state_mappings = (
        (
            "hca_state_slot_mapping",
            "hca_compress_state_block_table",
            prefill.HCA_STATE_BLOCK_SIZE,
        ),
        (
            "csa_state_slot_mapping",
            "csa_compress_state_block_table",
            prefill.CSA_STATE_BLOCK_SIZE,
        ),
        (
            "csa_inner_state_slot_mapping",
            "csa_inner_compress_state_block_table",
            prefill.INNER_STATE_BLOCK_SIZE,
        ),
    )
    for name, table_name, block_size in state_mappings:
        io_tensors[name].copy_(
            _ranked_prefill(
                decode_metadata.state_slot_mapping(
                    positions, row[table_name], state_block_size=block_size
                ),
                prefill.N_RANKS,
            )
        )

    io_tensors["logit_row_indices"].fill_(-1)
    io_tensors["logit_row_indices"][:, 0] = prompt_len - 1
    scalars["num_tokens"] = torch.tensor(prompt_len, dtype=torch.int32)


def _decode_base_shapes(decode):
    return {
        "hca_compress_state": SimpleNamespace(
            shape=[
                decode.N_RANKS,
                decode.HCA_COMPRESS_STATE_BLOCK_NUM,
                decode.HCA_COMPRESS_STATE_BLOCK_SIZE,
                decode.HCA_COMPRESS_STATE_DIM,
            ]
        ),
        "csa_compress_state": SimpleNamespace(
            shape=[
                decode.N_RANKS,
                decode.CSA_MAIN_STATE_BLOCK_NUM,
                decode.CSA_MAIN_STATE_BLOCK_SIZE,
                decode.CSA_MAIN_STATE_DIM,
            ]
        ),
        "csa_inner_compress_state": SimpleNamespace(
            shape=[
                decode.N_RANKS,
                decode.CSA_INNER_STATE_BLOCK_NUM,
                decode.CSA_INNER_STATE_BLOCK_SIZE,
                decode.CSA_INNER_STATE_DIM,
            ]
        ),
    }


def _refresh_decode_io(decode, io_tensors, scalars, tables, start_pos, tokens):
    metadata = decode.make_forward_metadata_tensors(
        _decode_base_shapes(decode),
        start_pos=start_pos,
        commit_tokens=1,
        block_tables=tables,
    )
    sequence = decode.T // decode.B
    write_mappings = (
        "ori_slot_mapping",
        "swa_slot_mapping",
        "hca_cmp_slot_mapping",
        "csa_cmp_slot_mapping",
        "csa_idx_slot_mapping",
        "hca_state_slot_mapping",
        "csa_state_slot_mapping",
        "csa_inner_state_slot_mapping",
    )
    for name in write_mappings:
        slots = metadata[name].reshape(decode.N_RANKS, decode.B, sequence)
        slots[:, 0, 1:] = -1
        slots[:, 1:, :] = -1
    for name, value in metadata.items():
        io_tensors[name].copy_(value)

    io_tensors["input_ids"].zero_()
    io_tensors["input_ids"][:, 0].copy_(tokens.to(torch.int64))
    io_tensors["logit_row_indices"].fill_(-1)
    io_tensors["logit_row_indices"][:, 0] = 0
    io_tensors["logits"].fill_(float("nan"))
    io_tensors["sampled_ids"].fill_(-1)
    scalars["num_tokens"] = torch.tensor(1, dtype=torch.int32)


def _rank_spread(io_tensors, stage):
    """Largest absolute cross-rank difference at one sampling point.

    Every rank holds the full hidden state and the full vocabulary after the
    MoE combine and the lm_head gather, so a non-zero spread is a defect even
    when all ranks still pick the same token. Recording it keeps a run that
    agreed only by luck distinguishable from one that was actually identical.
    """
    logits = io_tensors["logits"][:, 0].float()
    rows = _active_rows(io_tensors, logits.shape[0])
    # Compare every active token row, not just the sampled one: a rank can be
    # identical where it is sampled and still differ earlier in the prompt.
    active = int(rows.max()) + 1 if rows.numel() else 1
    spread = {"stage": stage, "active_rows_scanned": active}
    for name, tensor in (("logits", None),
                         ("pre_hc_hidden_out", io_tensors.get("pre_hc_hidden_out")),
                         ("hidden_out", io_tensors.get("hidden_out"))):
        if name != "logits" and tensor is None:
            continue
        selected = (logits if name == "logits"
                    else tensor[:, :active].reshape(tensor.shape[0], -1).float())
        finite = torch.isfinite(selected)
        safe = torch.where(finite, selected, torch.zeros_like(selected))
        difference = (safe - safe[0:1]).abs()
        spread[name] = {
            "max_abs_difference": round(float(difference.max()), 6),
            "ranks_differing": [rank for rank in range(safe.shape[0])
                                if bool(difference[rank].any())],
        }
    spread["bit_identical"] = not any(
        entry["ranks_differing"] for entry in spread.values() if isinstance(entry, dict)
    )
    return spread


def _check_sample(io_tensors, vocab_size, stage, spread_log=None):
    """Validate one sampling point. `spread_log` only receives failures here.

    A passing point is recorded by the caller once the step has been timed, so
    the spread scan never inflates the measured per-token latency.
    """
    logits = io_tensors["logits"][:, 0]
    sampled = io_tensors["sampled_ids"][:, 0, 0]

    def record_failure():
        if spread_log is not None:
            spread_log.append(_rank_spread(io_tensors, stage))

    finite = torch.isfinite(logits)
    if not bool(finite.all()):
        bad_by_rank = (~finite).sum(dim=-1).tolist()
        record_failure()
        _report_sample_failure(io_tensors, stage, "non-finite logits")
        raise AssertionError(
            f"{stage}: active logits contain non-finite values by rank: "
            f"{bad_by_rank}"
        )
    if not bool(((sampled >= 0) & (sampled < vocab_size)).all()):
        record_failure()
        _report_sample_failure(io_tensors, stage, "sample outside vocabulary")
        raise AssertionError(f"{stage}: sampled token is outside the vocabulary: {sampled}")
    argmax = torch.argmax(logits, dim=-1).to(torch.int32)
    greedy_match = bool(torch.equal(sampled, argmax))
    if not greedy_match:
        record_failure()
        _report_sample_failure(io_tensors, stage, "sample and argmax differ")
        raise AssertionError(
            f"{stage}: sampled token does not match the logits argmax: "
            f"sampled={sampled}, argmax={argmax}"
        )
    if sampled.numel() > 1 and not bool(torch.equal(sampled, sampled[0].expand_as(sampled))):
        record_failure()
        _report_sample_failure(io_tensors, stage, "ranks sampled different tokens")
        raise AssertionError(f"{stage}: ranks sampled different tokens: {sampled.tolist()}")
    print(
        f"[SESSION] {stage}: sampled={sampled.tolist()} "
        f"greedy_match={greedy_match}",
        flush=True,
    )
    return sampled.to(torch.int64).clone()


def _active_rows(io_tensors, world_size):
    row_indices = io_tensors.get("logit_row_indices")
    if row_indices is None:
        return torch.zeros(world_size, dtype=torch.int64)
    rows = row_indices[:, 0].to(torch.int64).cpu()
    if rows.numel() != world_size:
        return torch.zeros(world_size, dtype=torch.int64)
    return rows


def _selected_rank_rows(tensor, rows):
    if tensor.ndim < 2 or tensor.shape[0] != rows.numel():
        return tensor.reshape(tensor.shape[0], -1)
    selected = [tensor[rank, int(rows[rank])] for rank in range(rows.numel())]
    return torch.stack(selected).reshape(rows.numel(), -1)


def _report_sample_failure(io_tensors, stage, reason):
    logits = io_tensors["logits"][:, 0].float()
    rows = _active_rows(io_tensors, logits.shape[0])
    print(f"[FAILURE] {stage}: {reason}; active_rows={rows.tolist()}", flush=True)
    for name in ("pre_hc_hidden_out", "hidden_out", "logits"):
        tensor = io_tensors.get(name)
        if tensor is None:
            continue
        selected = logits if name == "logits" else _selected_rank_rows(tensor, rows).float()
        finite = torch.isfinite(selected)
        safe = torch.where(finite, selected, torch.zeros_like(selected))
        reference = safe[0:1]
        stats = f"nonfinite={((~finite).sum(dim=-1)).tolist()} sum={safe.sum(dim=-1).tolist()}"
        stats += f" sq_sum={(safe * safe).sum(dim=-1).tolist()} max_abs={safe.abs().amax(dim=-1).tolist()}"
        stats += f" max_abs_diff_rank0={(safe - reference).abs().amax(dim=-1).tolist()}"
        print(f"[FAILURE] {stage} {name}: {stats}", flush=True)
    top_values, top_indices = torch.topk(logits, k=2, dim=-1)
    print(f"[FAILURE] {stage} logits_top2: ids={top_indices.tolist()} values={top_values.tolist()}", flush=True)

    dump_dir = os.environ.get("DSV4_FAIL_DUMP_DIR")
    if not dump_dir:
        return
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage)
    path = Path(dump_dir) / f"failure_{safe_stage}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    names = (
        "input_ids",
        "logit_row_indices",
        "pre_hc_hidden_out",
        "hidden_out",
        "logits",
        "sampled_ids",
    )
    torch.save({name: io_tensors[name].clone() for name in names if name in io_tensors}, path)
    print(f"[FAILURE] {stage}: saved tensors to {path}", flush=True)


def _share_io(*io_groups):
    for io_tensors in io_groups:
        for name, tensor in list(io_tensors.items()):
            io_tensors[name] = tensor.cpu().contiguous().share_memory_()


def _shared_storage_tensors(tensors):
    """Promote host backing before naming it across forked worker processes."""
    # torch.load(mmap=True) defaults to MAP_PRIVATE. Merely listing that
    # storage as inherited does not make post-fork writes visible to a worker.
    if sys.platform == "linux":
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in tensors.values() if not tensor.is_shared()
        }
        required = sum(storages.values())
        memory = os.statvfs("/dev/shm")
        available = memory.f_bavail * memory.f_frsize
        if required > available:
            raise MemoryError(
                f"resident host bank needs {required / 2**30:.2f} GiB of shared memory; "
                f"/dev/shm has {available / 2**30:.2f} GiB available"
            )
    _share_io(tensors)
    inherited = []
    seen = set()
    for tensor in tensors.values():
        pointer = tensor.untyped_storage().data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            inherited.append(tensor)
    return inherited



def _decode_step_indices(limit, result, *, eos_id, stop_at_eos):
    """Stop before dispatch when the latest token, including prefill, is EOS."""
    for step in range(limit + 1):
        if stop_at_eos and eos_id is not None and result["generated_ids"][-1] == eos_id:
            result["stop_reason"] = "eos"
            return
        if step == limit:
            result["stop_reason"] = "decode_limit"
            return
        yield step


def _run_session(args, prefill_dir, decode_dir, model_dir, result=None):
    if result is None:
        result = _new_result(args)
    timings = result["timings_ms"]
    load_start = time.perf_counter()
    from pypto.backend import BackendType
    from pypto.ir import (
        DistributedCompiledProgram,
        DistributedConfig,
    )
    from pypto.runtime import DistributedWorker, RunConfig

    device_ids = [int(device) for device in args.device.split(",")]
    distributed_config = DistributedConfig(
        device_ids=device_ids,
        num_sub_workers=0,
    )
    prefill_compiled = DistributedCompiledProgram.from_dir(
        prefill_dir,
        platform=args.platform,
        backend_type=BackendType.Ascend950,
        distributed_config=distributed_config,
    )
    decode_compiled = DistributedCompiledProgram.from_dir(
        decode_dir,
        platform=args.platform,
        backend_type=BackendType.Ascend950,
        distributed_config=distributed_config,
    )
    for compiled, program in (
        (prefill_compiled, "prefill"),
        (decode_compiled, "decode"),
    ):
        _require_runtime_scalar(compiled, "num_tokens", program, torch.int32)
        _require_runtime_scalar(compiled, "moe_epoch_base", program, torch.int32)

    timings["load_artifacts"] = (time.perf_counter() - load_start) * 1000.0
    host_start = time.perf_counter()
    prompt_ids = getattr(args, "prompt_ids", None)
    prompt_len = int(prompt_ids.numel()) if prompt_ids is not None else args.prefill_tokens

    result["prompt_tokens"] = prompt_len
    result["prompt_token_ids"] = prompt_ids.tolist() if prompt_ids is not None else None

    import_argv = [
        str(model_dir / "synthetic_token_loop.py"),
        "--variant",
        args.variant,
        "--ep",
        str(args.ep),
        "--tp",
        str(args.tp),
    ]
    saved_argv = sys.argv
    sys.argv = import_argv
    sys.path.insert(0, str(model_dir))
    try:
        prefill = importlib.import_module("prefill_fwd")
        resident_names = set(prefill.RESIDENT_WEIGHT_NAMES) | set(
            prefill.RESIDENT_CACHE_NAMES
        )
        weight_specs = None
        if args.weights is not None:
            from utils import apply_real_weights

            specs = prefill.build_tensor_specs(start_pos=0, num_tokens=prompt_len)
            count = apply_real_weights(
                specs, args.weights, ep=prefill.N_RANKS, tp=prefill.LM_HEAD_TP_SIZE
            )
            print(f"[SESSION] real weights: {count} specs from {args.weights}", flush=True)
            weight_specs = {
                spec.name: spec
                for spec in specs
                if getattr(spec, "name", None) in prefill.RESIDENT_WEIGHT_NAMES
            }
        resident_hosts = _build_resident_hosts(
            prefill, prefill_compiled, decode_compiled, weight_specs
        )
        weight_specs = None
        prefill_io, prefill_scalars = _build_io(
            prefill_compiled, resident_names, prefill.MODEL_CONFIG.vocab_size
        )
        prefill_constants = SimpleNamespace(
            N_RANKS=prefill.N_RANKS,
            VOCAB=prefill.VOCAB,
            BLOCK_SIZE=prefill.BLOCK_SIZE,
            HCA_COMPRESS_RATIO=prefill.HCA_COMPRESS_RATIO,
            CSA_COMPRESS_RATIO=prefill.CSA_COMPRESS_RATIO,
            HCA_STATE_BLOCK_SIZE=prefill.HCA_STATE_BLOCK_SIZE,
            CSA_STATE_BLOCK_SIZE=prefill.CSA_STATE_BLOCK_SIZE,
            INNER_STATE_BLOCK_SIZE=prefill.INNER_STATE_BLOCK_SIZE,
        )
        del prefill
        _purge_model_modules(model_dir)

        sys.argv = import_argv
        decode = importlib.import_module("decode_fwd")
        decode_metadata = importlib.import_module("decode_metadata")
        if decode.ACTIVE_VARIANT != args.variant:
            raise ValueError(
                f"loaded variant {decode.ACTIVE_VARIANT}, expected {args.variant}"
            )
        tables = _build_session_tables(decode)
        _initialize_prefill_io(
            prefill_constants, decode_metadata,
            prefill_io, prefill_scalars, tables,
            prompt_ids, prompt_len,
        )
        decode_io, decode_scalars = _build_io(
            decode_compiled, resident_names, decode.MODEL_CONFIG.vocab_size
        )
        initial_tokens = torch.zeros(decode.N_RANKS, dtype=torch.int64)
        _refresh_decode_io(
            decode,
            decode_io,
            decode_scalars,
            tables,
            prompt_len,
            initial_tokens,
        )
        prefill_io["logits"].fill_(float("nan"))
        prefill_io["sampled_ids"].fill_(-1)
        _share_io(prefill_io, decode_io)

        run_config_parameters = inspect.signature(RunConfig).parameters
        if "enable_chip_swimlane" in run_config_parameters:
            swimlane_config = {"enable_chip_swimlane": 0}
        elif "enable_l2_swimlane" in run_config_parameters:
            swimlane_config = {"enable_l2_swimlane": 0}
        else:
            raise TypeError(
                "RunConfig supports neither enable_chip_swimlane nor "
                "enable_l2_swimlane"
            )

        prefill_config = RunConfig(
            platform=args.platform,
            device_id=0,
            ring_heap=PREFILL_RING_HEAP,
            **swimlane_config,
        )
        decode_config = RunConfig(
            platform=args.platform,
            device_id=0,
            ring_heap=(0, 0, 0, 1024 * 1024 * 1024),
            **swimlane_config,
        )

        inherited = _shared_storage_tensors(resident_hosts)
        timings["prepare_host"] = (time.perf_counter() - host_start) * 1000.0
        runtime = None
        handles = {}
        uploaded = []
        try:
            worker_start = time.perf_counter()
            runtime = _create_persistent_worker(
                DistributedWorker, [prefill_compiled, decode_compiled], prefill_config, inherited
            )
            timings["prepare_worker"] = (time.perf_counter() - worker_start) * 1000.0
            upload_start = time.perf_counter()
            for index, name in enumerate(sorted(resident_names), start=1):
                print(
                    f"[SESSION] upload resident {index}/{len(resident_names)}: {name}",
                    flush=True,
                )
                handles[name] = runtime.alloc_stacked_tensor(resident_hosts[name])
                uploaded.append(name)
            timings["upload_resident"] = (time.perf_counter() - upload_start) * 1000.0
            runtime.release_inherited_host_tensor_refs()
            inherited.clear()
            resident_hosts.clear()
            gc.collect()
            timings["load_total"] = (time.perf_counter() - load_start) * 1000.0

            # run() waits for every rank's result; no additional NPU dispatch or
            # synchronization is needed. Measure the resident request only.
            generation_start = time.perf_counter()
            runtime.run(
                prefill_compiled,
                *_ordered_args(
                    prefill_compiled, prefill_io, handles, prefill_scalars
                ),
                config=prefill_config,
            )
            timings["prefill_dispatch"] = (time.perf_counter() - generation_start) * 1000.0
            tokens = _check_sample(
                prefill_io, decode.MODEL_CONFIG.vocab_size, "prefill",
                spread_log=result["rank_spread"],
            )
            first_token_time = time.perf_counter()
            timings["ttft"] = (first_token_time - generation_start) * 1000.0
            result["rank_spread"].append(_rank_spread(prefill_io, "prefill"))
            timings["generation_wall"] = timings["ttft"]
            token_history = [tokens.tolist()]
            result["generated_ids"].append(int(tokens[0]))

            eos_id = getattr(args, "eos_id", None)
            previous_token_time = first_token_time
            for step in _decode_step_indices(
                args.decode_steps, result, eos_id=eos_id, stop_at_eos=args.weights is not None,
            ):
                step_start = previous_token_time
                start_pos = prompt_len + step
                _refresh_decode_io(
                    decode,
                    decode_io,
                    decode_scalars,
                    tables,
                    start_pos,
                    tokens,
                )
                decode_scalars["moe_epoch_base"] = torch.tensor(step * decode.LAST_MOE_EPOCH, dtype=torch.int32)
                dispatch_start = time.perf_counter()
                runtime.run(
                    decode_compiled,
                    *_ordered_args(
                        decode_compiled, decode_io, handles, decode_scalars
                    ),
                    config=decode_config,
                )
                dispatch_ms = (time.perf_counter() - dispatch_start) * 1000.0
                tokens = _check_sample(
                    decode_io,
                    decode.MODEL_CONFIG.vocab_size,
                    f"decode[{step}]@{start_pos}",
                    spread_log=result["rank_spread"],
                )
                token_time = time.perf_counter()
                previous_token_time = token_time
                timings["decode_step_wall"].append((token_time - step_start) * 1000.0)
                timings["decode_dispatch"].append(dispatch_ms)
                timings["generation_wall"] = (token_time - generation_start) * 1000.0
                result["rank_spread"].append(_rank_spread(decode_io, f"decode[{step}]@{start_pos}"))
                token_history.append(tokens.tolist())
                result["generated_ids"].append(int(tokens[0]))
            if result["stop_reason"] == "eos":
                print(f"[SESSION] eos after {len(token_history)} generated token(s)", flush=True)
            print(f"[SESSION] token_history={token_history}", flush=True)
            if getattr(args, "tokenizer", None):
                try:
                    from tokenizers import Tokenizer
                except ImportError as error:
                    raise SystemExit(
                        "--tokenizer needs the `tokenizers` package (pip install tokenizers)"
                    ) from error

                generated = [step_tokens[0] for step_tokens in token_history]
                text = Tokenizer.from_file(args.tokenizer).decode(generated)
                result["generated_text"] = text
                print(f"[SESSION] generated ids: {generated}", flush=True)
                print(f"[SESSION] generated text: {text!r}", flush=True)
        finally:
            if runtime is not None:
                for name in reversed(uploaded):
                    try:
                        runtime.free_stacked_tensor(handles[name])
                    except Exception as error:
                        print(
                            f"[SESSION] warning: failed to free {name}: {error}",
                            flush=True,
                        )
                runtime.close()
    finally:
        sys.argv = saved_argv
        if sys.path and sys.path[0] == str(model_dir):
            sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 prefill/decode token loop (synthetic by default; "
        "--weights/--prompt for real-weight generation)."
    )
    parser.add_argument("--variant", choices=("pro", "flash"), default="flash")
    parser.add_argument("-p", "--platform", choices=("a5",), default="a5")
    parser.add_argument("--ep", type=int, choices=(2, 4, 8), default=EP_SIZE,
                        help="EP world size / rank count; only EP8 deploys the full 256-expert model")
    parser.add_argument("--tp", type=int, choices=(TP_SIZE,), default=TP_SIZE,
                        help="LM-head vocabulary shards within the EP ranks")
    parser.add_argument("-d", "--device", default="0,1")
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--prefill-runtime-dir", type=Path)
    parser.add_argument("--decode-runtime-dir", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--result-json", type=Path,
        help="write status, generated tokens and resident host-wall timings as JSON (also on failure)",
    )
    parser.add_argument("--weights", type=str, default=None,
                        help="HF checkpoint dir or utils.py .pt cache dir "
                             "(must match --ep/--tp); real weights for the resident bank.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="natural-language prompt (requires --weights and a tokenizer)")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="read the prompt text from this file")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="tokenizer.json path; defaults to <--weights>/tokenizer.json if present")
    parser.add_argument("--no-bos", action="store_true",
                        help="do not prepend the BOS token to the prompt")
    parser.add_argument("--eos-id", type=int, default=1,
                        help="stop decoding when this token is sampled (real-weight runs only)")
    parser.add_argument(
        "--prefill-tokens", type=int, default=PROMPT_TOKENS,
        help=f"maximum active prompt length (including BOS); the fixed prefill capacity is {PROMPT_TOKENS} tokens",
    )
    args = parser.parse_args()

    device_ids = [device for device in args.device.split(",") if device]
    if len(device_ids) != args.ep:
        raise ValueError(f"EP{args.ep} requires exactly {args.ep} devices, got {args.device!r}")
    if args.decode_steps < 1:
        raise ValueError("--decode-steps must be positive")
    if not 1 <= args.prefill_tokens <= PROMPT_TOKENS:
        raise ValueError(f"--prefill-tokens must be within 1..{PROMPT_TOKENS}")

    result = _new_result(args)
    invocation_start = time.perf_counter()
    if args.result_json is not None:
        _write_result(args.result_json, result)
    try:
        prompt_start = time.perf_counter()
        args.prompt_ids = None
        if args.prompt is not None or args.prompt_file is not None:
            if args.weights is None:
                raise ValueError("--prompt/--prompt-file require --weights (real-weight run)")
            if args.prompt is not None and args.prompt_file is not None:
                raise ValueError("pass either --prompt or --prompt-file, not both")
            text = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text()
            if args.tokenizer is None:
                candidate = Path(args.weights) / "tokenizer.json"
                if candidate.is_file():
                    args.tokenizer = str(candidate)
            if args.tokenizer is None:
                raise ValueError("--tokenizer is required (no tokenizer.json under --weights)")
            try:
                from tokenizers import Tokenizer
            except ImportError as error:
                raise SystemExit(
                    "--prompt/--prompt-file need the `tokenizers` package (pip install tokenizers)"
                ) from error

            tokenizer = Tokenizer.from_file(args.tokenizer)
            ids = tokenizer.encode(text).ids
            if not args.no_bos:
                bos_id = tokenizer.token_to_id("<｜begin▁of▁sentence｜>")
                if bos_id is not None:
                    ids = [bos_id] + ids
            if not 1 <= len(ids) <= args.prefill_tokens:
                raise ValueError(
                    f"prompt is {len(ids)} tokens; must fit the configured active-token limit "
                    f"1..{args.prefill_tokens} (--prefill-tokens)"
                )
            args.prompt_ids = torch.tensor(ids, dtype=torch.int64)
            print(f"[SESSION] prompt: {len(ids)} tokens {ids}", flush=True)

        result["tokenizer"] = str(args.tokenizer) if args.tokenizer is not None else None
        result["prompt_tokens"] = int(args.prompt_ids.numel()) if args.prompt_ids is not None else args.prefill_tokens
        result["prompt_token_ids"] = args.prompt_ids.tolist() if args.prompt_ids is not None else None
        result["timings_ms"]["prompt_prepare"] = (time.perf_counter() - prompt_start) * 1000.0
        os.environ["DEEPSEEK_V4_VARIANT"] = args.variant
        model_dir = Path(__file__).resolve().parent
        prefill_dir = args.prefill_runtime_dir
        decode_dir = args.decode_runtime_dir
        if prefill_dir is None:
            compile_start = time.perf_counter()
            try:
                prefill_dir = _compile_program(model_dir, "prefill_fwd.py", args, args.prefill_tokens, 0)
            finally:
                result["timings_ms"]["compile_prefill"] = (time.perf_counter() - compile_start) * 1000.0
        if decode_dir is None:
            compile_start = time.perf_counter()
            try:
                decode_dir = _compile_program(model_dir, "decode_fwd.py", args, 1, PROMPT_TOKENS)
            finally:
                result["timings_ms"]["compile_decode"] = (time.perf_counter() - compile_start) * 1000.0
        print(f"[SESSION] prefill_runtime_dir={prefill_dir.resolve()}", flush=True)
        print(f"[SESSION] decode_runtime_dir={decode_dir.resolve()}", flush=True)
        if args.compile_only:
            result["status"] = "compile_only"
            return

        _run_session(args, prefill_dir.resolve(), decode_dir.resolve(), model_dir, result)
        result["status"] = "pass"
        mode = "real-weight" if args.weights is not None else "synthetic"
        print(f"[SESSION] PASS: EP{args.ep} {mode} token loop completed", flush=True)
    except BaseException as error:
        result["status"] = "fail"
        result["error_type"] = type(error).__name__
        raise
    finally:
        result["timings_ms"]["invocation_wall"] = (time.perf_counter() - invocation_start) * 1000.0
        _finish_result(result)
        if args.result_json is not None:
            _write_result(args.result_json, result)
            print(f"[SESSION] result JSON: {args.result_json}", flush=True)


if __name__ == "__main__":
    main()
