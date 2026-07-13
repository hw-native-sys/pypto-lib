# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3-14B external contract, colocated with the kernel entry points."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pypto.language as pl

from constants import QWEN3_14B, QWEN3_14B_TILING
from contract.base import (
    ContractRegistration,
    KernelSpec,
    LoadedKernelModules,
    ModelId,
    ModelContract,
    TensorArgSpec,
)
from weights import prepare_qwen3_weights

if TYPE_CHECKING:
    import torch


_KERNEL_DIR = Path(__file__).resolve().parent

prefill_fwd: Any | None = None
decode_fwd: Any | None = None
greedy_sample_fwd: Any | None = None


def _arg(
    name: str,
    dtype: str,
    shape: tuple[str | int, ...],
    direction: str = "in",
) -> TensorArgSpec:
    return TensorArgSpec(name=name, dtype=dtype, shape=shape, direction=direction)


_PREFILL_ARGS = (
    _arg("hidden_states", "bf16", ("PREFILL_TOKENS", "H")),
    _arg("seq_lens", "int32", ("USER_BATCH",)),
    _arg("chunk_lens", "int32", ("USER_BATCH",)),
    _arg("chunk_offsets", "int32", ("USER_BATCH",)),
    _arg("input_rms_weight", "fp32", ("L", "H")),
    _arg("wq", "bf16", ("L*H", "H")),
    _arg("wk", "bf16", ("L*H", "KVH")),
    _arg("wv", "bf16", ("L*H", "KVH")),
    _arg("q_norm_weight", "fp32", ("L", "D")),
    _arg("k_norm_weight", "fp32", ("L", "D")),
    _arg("rope_cos", "fp32", ("MAX_SEQ", "D")),
    _arg("rope_sin", "fp32", ("MAX_SEQ", "D")),
    _arg("block_table", "int32", ("BLOCK_TABLE_FLAT",)),
    _arg("slot_mapping", "int32", ("PREFILL_TOKENS",)),
    _arg("k_cache", "bf16", ("KV_CACHE_ROWS", "D"), "inout"),
    _arg("v_cache", "bf16", ("KV_CACHE_ROWS", "D"), "inout"),
    _arg("wo", "bf16", ("L*H", "H")),
    _arg("w_gate", "bf16", ("L*H", "I")),
    _arg("w_up", "bf16", ("L*H", "I")),
    _arg("w_down", "bf16", ("L*I", "H")),
    _arg("post_rms_weight", "fp32", ("L", "H")),
    _arg("final_norm_weight", "fp32", (1, "H")),
    _arg("lm_head_weight", "bf16", ("VOCAB", "H")),
    _arg("out", "fp32", ("USER_BATCH", "VOCAB"), "out"),
)

_DECODE_ARGS = (
    _arg("input_rms_weight", "fp32", ("L", "H")),
    _arg("wq", "bf16", ("L*H", "H")),
    _arg("wk", "bf16", ("L*H", "KVH")),
    _arg("wv", "bf16", ("L*H", "KVH")),
    _arg("q_norm_weight", "fp32", ("L", "D")),
    _arg("k_norm_weight", "fp32", ("L", "D")),
    _arg("seq_lens", "int32", ("USER_BATCH",)),
    _arg("block_table", "int32", ("BLOCK_TABLE_FLAT",)),
    _arg("slot_mapping", "int32", ("USER_BATCH",)),
    _arg("rope_cos", "fp32", ("MAX_SEQ", "D")),
    _arg("rope_sin", "fp32", ("MAX_SEQ", "D")),
    _arg("k_cache", "bf16", ("KV_CACHE_ROWS", "D"), "inout"),
    _arg("v_cache", "bf16", ("KV_CACHE_ROWS", "D"), "inout"),
    _arg("wo", "bf16", ("L*H", "H")),
    _arg("w_gate", "bf16", ("L*H", "I")),
    _arg("w_up", "bf16", ("L*H", "I")),
    _arg("w_down", "bf16", ("L*I", "H")),
    _arg("post_rms_weight", "fp32", ("L", "H")),
    _arg("final_norm_weight", "fp32", (1, "H")),
    _arg("lm_head_weight", "bf16", ("VOCAB", "H")),
    _arg("out", "fp32", ("USER_BATCH", "VOCAB"), "out"),
    _arg("embed_weight", "bf16", ("VOCAB", "H")),
    _arg("sampled_ids_in", "int32", ("USER_BATCH", "SAMPLED_IDS_PAD")),
    _arg("sampled_ids", "int32", ("USER_BATCH", "SAMPLED_IDS_PAD"), "out"),
    _arg("next_hidden", "bf16", ("USER_BATCH", "H"), "out"),
)


def bind_qwen3_kernel_functions(
    *,
    prefill_fwd: Any,
    decode_fwd: Any,
    greedy_sample_fwd: Any,
) -> None:
    """Bind loaded Qwen3 kernel functions to the HOST wrappers."""
    globals()["prefill_fwd"] = prefill_fwd
    globals()["decode_fwd"] = decode_fwd
    globals()["greedy_sample_fwd"] = greedy_sample_fwd


@pl.jit.host
def qwen3_prefill_host(
    hidden_states: pl.Tensor,
    seq_lens: pl.Tensor,
    chunk_lens: pl.Tensor,
    chunk_offsets: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    final_norm_weight: pl.Tensor,
    lm_head_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
) -> pl.Tensor:
    return prefill_fwd(
        hidden_states,
        seq_lens,
        chunk_lens,
        chunk_offsets,
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        block_table,
        slot_mapping,
        k_cache,
        v_cache,
        wo,
        post_rms_weight,
        w_gate,
        w_up,
        w_down,
        final_norm_weight,
        lm_head_weight,
        out,
    )


@pl.jit.host
def qwen3_decode_host(
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    final_norm_weight: pl.Tensor,
    lm_head_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
    embed_weight: pl.Tensor,
    sampled_ids_in: pl.Tensor,
    sampled_ids: pl.Out[pl.Tensor],
    next_hidden: pl.Out[pl.Tensor],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    out, sampled_ids, next_hidden = decode_fwd(
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        wo,
        w_gate,
        w_up,
        w_down,
        post_rms_weight,
        final_norm_weight,
        lm_head_weight,
        out,
        embed_weight,
        sampled_ids_in,
        sampled_ids,
        next_hidden,
    )
    return out, sampled_ids, next_hidden


@pl.jit.host
def qwen3_greedy_sample_host(
    logits: pl.Tensor,
    sampled_ids: pl.Out[pl.Tensor],
) -> pl.Tensor:
    return greedy_sample_fwd(logits, sampled_ids)


def build_prefill_compile_args(model_config: Any, runtime_config: Any) -> tuple[torch.Tensor, ...]:
    """Build dummy compile arguments for the Qwen3 prefill HOST wrapper."""
    import torch

    dims = _dims(model_config, runtime_config)
    total_tokens = dims["batch"] * dims["max_seq"]
    cache_rows = dims["batch"] * dims["runtime_cache_blocks"] * dims["layers"] * dims["kv_heads"] * dims["page"]
    return (
        torch.empty((total_tokens, dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["batch"],), dtype=torch.int32),
        torch.empty((dims["batch"],), dtype=torch.int32),
        torch.empty((dims["batch"],), dtype=torch.int32),
        torch.empty((dims["layers"], dims["hidden"]), dtype=torch.float32),
        torch.empty((dims["layers"] * dims["hidden"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["kv_hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["kv_hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"], dims["head_dim"]), dtype=torch.float32),
        torch.empty((dims["layers"], dims["head_dim"]), dtype=torch.float32),
        torch.empty((dims["max_seq"], dims["head_dim"]), dtype=torch.float32),
        torch.empty((dims["max_seq"], dims["head_dim"]), dtype=torch.float32),
        torch.empty((dims["batch"] * dims["block_table_stride"],), dtype=torch.int32),
        torch.empty((total_tokens,), dtype=torch.int32),
        torch.empty((cache_rows, dims["head_dim"]), dtype=torch.bfloat16),
        torch.empty((cache_rows, dims["head_dim"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["intermediate"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["intermediate"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["intermediate"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"], dims["hidden"]), dtype=torch.float32),
        torch.empty((1, dims["hidden"]), dtype=torch.float32),
        torch.empty((dims["vocab"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["batch"], dims["vocab"]), dtype=torch.float32),
    )


def build_decode_compile_args(model_config: Any, runtime_config: Any) -> tuple[torch.Tensor, ...]:
    """Build dummy compile arguments for the Qwen3 decode HOST wrapper."""
    import torch

    dims = _dims(model_config, runtime_config)
    cache_rows = dims["layers"] * dims["batch"] * dims["runtime_cache_blocks"] * dims["kv_heads"] * dims["page"]
    return (
        torch.empty((dims["layers"], dims["hidden"]), dtype=torch.float32),
        torch.empty((dims["layers"] * dims["hidden"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["kv_hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["kv_hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"], dims["head_dim"]), dtype=torch.float32),
        torch.empty((dims["layers"], dims["head_dim"]), dtype=torch.float32),
        torch.empty((dims["batch"],), dtype=torch.int32),
        torch.empty((dims["batch"] * dims["block_table_stride"],), dtype=torch.int32),
        torch.empty((dims["batch"],), dtype=torch.int32),
        torch.empty((dims["max_seq"], dims["head_dim"]), dtype=torch.float32),
        torch.empty((dims["max_seq"], dims["head_dim"]), dtype=torch.float32),
        torch.empty((cache_rows, dims["head_dim"]), dtype=torch.bfloat16),
        torch.empty((cache_rows, dims["head_dim"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["intermediate"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["hidden"], dims["intermediate"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"] * dims["intermediate"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["layers"], dims["hidden"]), dtype=torch.float32),
        torch.empty((1, dims["hidden"]), dtype=torch.float32),
        torch.empty((dims["vocab"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["batch"], dims["vocab"]), dtype=torch.float32),
        torch.empty((dims["vocab"], dims["hidden"]), dtype=torch.bfloat16),
        torch.empty((dims["batch"], dims["sampled_ids"]), dtype=torch.int32),
        torch.empty((dims["batch"], dims["sampled_ids"]), dtype=torch.int32),
        torch.empty((dims["batch"], dims["hidden"]), dtype=torch.bfloat16),
    )


def build_greedy_sample_compile_args(model_config: Any, runtime_config: Any) -> tuple[torch.Tensor, ...]:
    """Build dummy compile arguments for the Qwen3 greedy-sampling HOST wrapper."""
    import torch

    dims = _dims(model_config, runtime_config)
    return (
        torch.empty((dims["batch"], dims["vocab"]), dtype=torch.float32),
        torch.empty((dims["batch"], dims["sampled_ids"]), dtype=torch.int32),
    )


def build_prefill_runtime_args(
    inputs: Any,
    static: Any,
    *,
    k_cache: Any,
    v_cache: Any,
    logits: Any,
) -> tuple[Any, ...]:
    """Build arguments in qwen3_prefill_host signature order."""
    weights = static.decode_weights
    return (
        inputs.hidden,
        inputs.seq_lens,
        inputs.chunk_lens,
        inputs.chunk_offsets,
        weights["decode_input_rms_weight"],
        weights["decode_wq"],
        weights["decode_wk"],
        weights["decode_wv"],
        weights["decode_q_norm_weight"],
        weights["decode_k_norm_weight"],
        static.rope_cos,
        static.rope_sin,
        inputs.block_table,
        inputs.slot_mapping,
        k_cache,
        v_cache,
        weights["decode_wo"],
        weights["decode_w_gate"],
        weights["decode_w_up"],
        weights["decode_w_down"],
        weights["decode_post_rms_weight"],
        static.final_norm_weight,
        static.padded_lm_head_weight,
        logits,
    )


def build_decode_runtime_args(
    inputs: Any,
    static: Any,
    *,
    k_cache: Any,
    v_cache: Any,
    sampled_ids_buffer: Any,
    next_hidden_buffer: Any,
) -> tuple[Any, ...]:
    """Build arguments in qwen3_decode_host signature order."""
    weights = static.decode_weights
    return (
        weights["decode_input_rms_weight"],
        weights["decode_wq"],
        weights["decode_wk"],
        weights["decode_wv"],
        weights["decode_q_norm_weight"],
        weights["decode_k_norm_weight"],
        inputs.seq_lens,
        inputs.block_table,
        inputs.slot_mapping,
        static.rope_cos,
        static.rope_sin,
        k_cache,
        v_cache,
        weights["decode_wo"],
        weights["decode_w_gate"],
        weights["decode_w_up"],
        weights["decode_w_down"],
        weights["decode_post_rms_weight"],
        static.final_norm_weight,
        static.padded_lm_head_weight,
        inputs.logits,
        static.padded_embed_weight,
        inputs.token_ids,
        sampled_ids_buffer,
        next_hidden_buffer,
    )


def build_greedy_sample_runtime_args(
    inputs: Any,
    static: Any | None = None,
    *,
    sampled_ids_buffer: Any,
) -> tuple[Any, ...]:
    """Build arguments in qwen3_greedy_sample_host signature order."""
    return (inputs.logits, sampled_ids_buffer)


def load_qwen3_kernel_modules() -> LoadedKernelModules:
    """Load Qwen3-14B kernel functions and constants from this variant directory."""
    modules = {
        "prefill": _load_kernel_module("prefill_fwd"),
        "decode": _load_kernel_module("decode_fwd"),
        "greedy_sample": _load_kernel_module("greedy_sample"),
    }
    decode = modules["decode"]
    greedy_sample = modules["greedy_sample"]
    return LoadedKernelModules(
        functions={
            "prefill_fwd": modules["prefill"].prefill_fwd,
            "decode_fwd": decode.decode_fwd,
            "greedy_sample_fwd": greedy_sample.greedy_sample_fwd,
        },
        constants={
            "prefill_seq_tile": int(modules["prefill"].SEQ_TILE),
            "prefill_block_size": int(modules["prefill"].BLOCK_SIZE),
            "decode_seq_tile": int(decode.SEQ_TILE),
            "decode_block_size": int(decode.BLOCK_SIZE),
            "decode_batch": int(decode.BATCH),
            "decode_max_seq": int(getattr(decode, "MAX_SEQ", QWEN3_14B.max_seq)),
            "decode_vocab": int(decode.VOCAB),
            "decode_real_vocab": int(decode.REAL_VOCAB),
            "decode_num_layers": int(decode.NUM_LAYERS),
            "decode_sampled_ids_pad": int(
                getattr(decode, "SAMPLED_IDS_PAD", getattr(greedy_sample, "SAMPLED_IDS_PAD", 1))
            ),
            "greedy_sample_batch": int(greedy_sample.BATCH),
            "greedy_sample_vocab": int(greedy_sample.VOCAB),
            "greedy_sample_sampled_ids_pad": int(greedy_sample.SAMPLED_IDS_PAD),
        },
    )


def validate_qwen3_kernel_modules(
    contract: ModelContract,
    loaded_kernels: LoadedKernelModules,
    model: Any,
) -> None:
    """Validate loaded Qwen3-14B kernels against model and runtime metadata."""
    constants = loaded_kernels.constants
    config = getattr(model, "config", None)
    runtime = getattr(model, "runtime", None)
    if config is None or runtime is None:
        raise TypeError("Qwen3-14B validation expects a model object with config and runtime attributes.")
    padded_vocab = _round_up(int(config.vocab_size), QWEN3_14B_TILING.vocab_chunk)
    kernel_batch = int(contract.limits["batch"])

    _expect(constants, "prefill_seq_tile", QWEN3_14B_TILING.seq_tile, "prefill_fwd SEQ_TILE mismatch")
    _expect(constants, "prefill_block_size", QWEN3_14B_TILING.block_size, "prefill_fwd BLOCK_SIZE mismatch")
    _expect(constants, "decode_seq_tile", QWEN3_14B_TILING.seq_tile, "decode_fwd SEQ_TILE mismatch")
    _expect(constants, "decode_block_size", QWEN3_14B_TILING.block_size, "decode_fwd BLOCK_SIZE mismatch")
    _expect(constants, "decode_batch", kernel_batch, "decode_fwd fixed BATCH mismatch")
    _expect(constants, "decode_num_layers", int(config.num_hidden_layers), "decode_fwd NUM_LAYERS mismatch")
    _expect(constants, "decode_vocab", padded_vocab, "decode_fwd VOCAB mismatch")
    _expect(constants, "decode_real_vocab", int(config.vocab_size), "decode_fwd REAL_VOCAB mismatch")
    _expect(constants, "decode_sampled_ids_pad", int(contract.limits["sampled_ids_pad"]), "decode sampled width")
    _expect(constants, "greedy_sample_batch", kernel_batch, "greedy_sample_fwd fixed BATCH mismatch")
    _expect(constants, "greedy_sample_vocab", padded_vocab, "greedy_sample_fwd VOCAB mismatch")

    if int(runtime.max_seq_len) > int(constants["decode_max_seq"]):
        raise ValueError(
            f"max_model_len {runtime.max_seq_len} exceeds Qwen3-14B kernel MAX_SEQ "
            f"{constants['decode_max_seq']}. Rebuild the kernels with a larger MAX_SEQ."
        )
    if int(runtime.page_size) != QWEN3_14B_TILING.block_size:
        raise ValueError(
            f"Qwen3-14B external runtime page_size must match kernel block_size "
            f"{QWEN3_14B_TILING.block_size}, got {runtime.page_size}."
        )
    runtime_vocab_pad_multiple = getattr(runtime, "vocab_pad_multiple", None)
    if runtime_vocab_pad_multiple is None:
        raise TypeError("Qwen3-14B validation expects runtime.vocab_pad_multiple.")
    if int(runtime_vocab_pad_multiple) != QWEN3_14B_TILING.vocab_chunk:
        raise ValueError(
            f"Qwen3-14B external runtime vocab_pad_multiple must match kernel vocab_chunk "
            f"{QWEN3_14B_TILING.vocab_chunk}, got {runtime_vocab_pad_multiple}."
        )
    total_kv_pages = getattr(runtime, "total_kv_pages", None)
    if total_kv_pages is not None and int(total_kv_pages) < kernel_batch:
        raise ValueError(f"total_kv_pages must be at least kernel_batch ({kernel_batch}), got {total_kv_pages}")
    _validate_supported_shape(config)


def get_qwen3_14b_contract() -> ModelContract:
    """Return the Qwen3-14B external ABI contract."""
    return ModelContract(
        schema_version="1",
        model=ModelId(family="qwen3", variant="14b", size="14b", quant="bf16"),
        capabilities=("paged_kv", "chunked_prefill", "device_greedy_sampling", "device_embedding"),
        limits={
            "batch": QWEN3_14B.batch,
            "max_seq": QWEN3_14B.max_seq,
            "page_size": QWEN3_14B_TILING.block_size,
            "vocab": QWEN3_14B.vocab,
            "real_vocab": QWEN3_14B.real_vocab,
            "num_layers": QWEN3_14B.num_layers,
            "sampled_ids_pad": QWEN3_14B.sampled_ids_pad,
            "vocab_pad_multiple": QWEN3_14B_TILING.vocab_chunk,
            "kv_cache_layout": "BSND",
            "supported_batch_sizes": "1,16",
            "supported_platforms": "a2a3",
            "compile_platforms": "a2a3,a2a3sim",
        },
        execution={"prefill": ("prefill",), "decode": ("decode",)},
        kernels={
            "prefill": _PREFILL_STAGE,
            "decode": _DECODE_STAGE,
            "greedy_sample": _GREEDY_SAMPLE_STAGE,
        },
        kernel_binder=bind_qwen3_kernel_functions,
        prepare_weights=prepare_qwen3_weights,
        load_kernels=load_qwen3_kernel_modules,
        validate_kernels=validate_qwen3_kernel_modules,
    )


def matches_qwen3_14b_model_config(model_config: object) -> bool:
    """Return whether parsed model metadata matches this Qwen3-14B contract."""
    architectures_raw = getattr(model_config, "architectures", None) or ()
    architectures = {
        _normalize_model_config_value(str(value))
        for value in architectures_raw
    }
    architecture = getattr(model_config, "architecture", None)
    if architecture is not None:
        architectures.add(_normalize_model_config_value(str(architecture)))
    model_type = _normalize_model_config_value(str(getattr(model_config, "model_type", "") or ""))
    if not ({"qwen3", "qwen3model"} & architectures or model_type == "qwen3"):
        return False
    expected = _model_shape()
    return all(getattr(model_config, field, None) == value for field, value in expected.items())


def _load_kernel_module(module_name: str) -> Any:
    module_path = _KERNEL_DIR / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing pypto-lib Qwen3-14B kernel module: {module_path}")
    spec = importlib.util.spec_from_file_location(f"_pypto_lib_qwen3_14b_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Qwen3-14B kernel module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    shadowed_modules = {
        name: sys.modules.pop(name)
        for name in ("config", "constants")
        if _is_other_pypto_lib_short_module(name)
    }
    sys.path.insert(0, str(_KERNEL_DIR))
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(_KERNEL_DIR))
        except ValueError:
            pass
        for name, old_module in shadowed_modules.items():
            sys.modules[name] = old_module
    return module


def _is_other_pypto_lib_short_module(name: str) -> bool:
    module = sys.modules.get(name)
    module_file = str(getattr(module, "__file__", "") or "")
    pypto_lib_models_dir = str(_KERNEL_DIR.parent.parent.parent)
    return module_file.startswith(pypto_lib_models_dir) and not module_file.startswith(str(_KERNEL_DIR))


def _dims(model_config: Any, runtime_config: Any) -> dict[str, int]:
    batch = int(runtime_config.max_batch_size)
    max_seq = int(runtime_config.max_seq_len)
    page = int(runtime_config.page_size)
    if batch != QWEN3_14B.batch:
        raise ValueError(f"Qwen3-14B kernels require max_batch_size {QWEN3_14B.batch}, got {batch}")
    if max_seq > QWEN3_14B.max_seq:
        raise ValueError(f"Qwen3-14B kernels require max_seq <= {QWEN3_14B.max_seq}, got {max_seq}")
    if page != QWEN3_14B_TILING.block_size:
        raise ValueError(
            f"Qwen3-14B external runtime page_size must match kernel block_size "
            f"{QWEN3_14B_TILING.block_size}, got {page}"
        )
    kv_heads = int(model_config.num_key_value_heads)
    head_dim = int(model_config.head_dim)
    return {
        "batch": batch,
        "max_seq": max_seq,
        "page": page,
        "hidden": int(model_config.hidden_size),
        "intermediate": int(model_config.intermediate_size),
        "layers": int(model_config.num_hidden_layers),
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "kv_hidden": kv_heads * head_dim,
        "runtime_cache_blocks": (max_seq + page - 1) // page,
        "block_table_stride": (max_seq + page - 1) // page,
        "vocab": _round_up(int(model_config.vocab_size), QWEN3_14B_TILING.vocab_chunk),
        "sampled_ids": QWEN3_14B.sampled_ids_pad,
    }


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _expect(constants: Any, name: str, expected: int, message: str) -> None:
    actual = int(constants[name])
    if actual != expected:
        raise ValueError(f"{message}: {actual} != {expected}.")


def _validate_supported_shape(config: Any) -> None:
    expected = _model_shape()
    actual = {key: getattr(config, key) for key in expected}
    if actual != expected:
        mismatch = ", ".join(
            f"{key}={actual[key]} (expected {value})"
            for key, value in expected.items()
            if actual[key] != value
        )
        raise ValueError(f"Qwen3-14B kernels support one model shape only: {mismatch}")


def _model_shape() -> dict[str, int]:
    return {
        "hidden_size": QWEN3_14B.hidden,
        "intermediate_size": QWEN3_14B.intermediate,
        "num_hidden_layers": QWEN3_14B.num_layers,
        "num_attention_heads": QWEN3_14B.num_heads,
        "num_key_value_heads": QWEN3_14B.num_kv_heads,
        "head_dim": QWEN3_14B.head_dim,
    }


def _normalize_model_config_value(value: str) -> str:
    return value.lower().replace("_", "-").replace("forcausallm", "")


_PREFILL_STAGE = KernelSpec(
    name="prefill",
    public_name="qwen3_14b.prefill_fwd",
    args=_PREFILL_ARGS,
    host_jit_fn=qwen3_prefill_host,
    compile_args_builder=build_prefill_compile_args,
    runtime_args_builder=build_prefill_runtime_args,
)

_DECODE_STAGE = KernelSpec(
    name="decode",
    public_name="qwen3_14b.decode_fwd",
    args=_DECODE_ARGS,
    host_jit_fn=qwen3_decode_host,
    compile_args_builder=build_decode_compile_args,
    runtime_args_builder=build_decode_runtime_args,
)

_GREEDY_SAMPLE_STAGE = KernelSpec(
    name="greedy_sample",
    public_name="qwen3_14b.greedy_sample_fwd",
    args=(
        _arg("logits", "fp32", ("B", "VOCAB")),
        _arg("sampled_ids", "int32", ("B", "SAMPLED_IDS_PAD"), "out"),
    ),
    host_jit_fn=qwen3_greedy_sample_host,
    compile_args_builder=build_greedy_sample_compile_args,
    runtime_args_builder=build_greedy_sample_runtime_args,
)

QWEN3_14B_REGISTRATION = ContractRegistration(
    family="qwen3",
    variant="14b",
    factory=get_qwen3_14b_contract,
    matcher=matches_qwen3_14b_model_config,
)
