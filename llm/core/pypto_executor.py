# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from .executor import ModelExecutor
from .kv_cache import KvCacheManager
from .types import (
    DecodeBatch,
    DecodeResult,
    KvAllocation,
    ModelRecord,
    PrefillBatch,
    PrefillResult,
    RuntimeModel,
)


def _ensure_pypto_import(pypto_root: str | None) -> None:
    try:
        import pypto  # noqa: F401
        return
    except ImportError:
        pass

    candidates: list[Path] = []
    if pypto_root:
        candidates.append(Path(pypto_root) / "python")
    candidates.append(Path(__file__).resolve().parents[2].parent / "pypto" / "python")

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            import pypto  # noqa: F401
            return
        except ImportError:
            continue
    raise ImportError(
        "Unable to import pypto. Pass pypto_root pointing at the local PyPTO repository or install pypto."
    )


def _backend_type_for_platform(platform: str):
    from pypto.backend import BackendType

    if platform.startswith("a5"):
        return BackendType.Ascend950
    return BackendType.Ascend910B


def _rope_tables(max_seq: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv_freq)
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    return torch.cat([cos_half, cos_half], dim=-1), torch.cat([sin_half, sin_half], dim=-1)


_VOCAB_PAD_MULTIPLE = 512  # must be a multiple of qwen3_14b_lm_head.VOCAB_CHUNK (64)
_LOGITS_BATCH_TILE = 16
_QWEN14B_PAGE_SIZE = 256


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


@dataclass
class _CompiledKernels:
    prefill: object
    decode: object
    final_rms: object
    lm_head: object
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor


@dataclass
class _PrefillInputs:
    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class _DecodeInputs:
    actual_batch: int
    hidden: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class PyptoQwen14BExecutor(ModelExecutor):
    def __init__(
        self,
        kv_cache_manager: KvCacheManager,
        *,
        pypto_root: str | None = None,
        platform: str = "a2a3sim",
        device_id: int = 0,
        save_kernels_dir: str | None = None,
    ) -> None:
        super().__init__(kv_cache_manager)
        self._pypto_root = pypto_root
        self._platform = platform
        self._device_id = device_id
        self._save_kernels_dir = save_kernels_dir
        self._compiled: dict[str, _CompiledKernels] = {}

    def register_model(self, model_id: str, record: ModelRecord) -> None:
        self._compiled[model_id] = self._compile_model(record.runtime_model)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        compiled = self._compiled[model.config.model_id]
        prefill_inputs = self._prepare_prefill_inputs(model, batch)
        hidden = prefill_inputs.hidden

        for layer_idx, layer in enumerate(model.layers):
            k_cache, v_cache = self._kv_cache_manager.materialize_decode_cache(
                model.config.model_id,
                layer_idx,
            )
            out = torch.zeros_like(hidden)
            compiled.prefill(
                hidden,
                prefill_inputs.seq_lens,
                layer.input_rms_weight.view(1, -1).float().cpu(),
                self._kernel_weight(layer.wq),
                self._kernel_weight(layer.wk),
                self._kernel_weight(layer.wv),
                layer.q_norm_weight.view(1, -1).float().cpu(),
                layer.k_norm_weight.view(1, -1).float().cpu(),
                compiled.rope_cos,
                compiled.rope_sin,
                prefill_inputs.block_table,
                prefill_inputs.slot_mapping,
                k_cache,
                v_cache,
                self._kernel_weight(layer.wo),
                layer.post_rms_weight.view(1, -1).float().cpu(),
                self._kernel_weight(layer.w_gate),
                self._kernel_weight(layer.w_up),
                self._kernel_weight(layer.w_down),
                out,
                config=self._run_config(codegen_only=False),
            )
            hidden = out

        last_hidden_rows: list[torch.Tensor] = []
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
            last_hidden_rows.append(hidden[batch_idx, seq_len - 1].float())
        last_hidden = torch.stack(last_hidden_rows)
        logits = self._project_logits(model, last_hidden)
        return PrefillResult(last_hidden=last_hidden, logits=logits)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        compiled = self._compiled[model.config.model_id]
        decode_inputs = self._prepare_decode_inputs(model, batch)
        hidden = decode_inputs.hidden

        for layer_idx, layer in enumerate(model.layers):
            k_cache, v_cache = self._kv_cache_manager.materialize_decode_cache(
                model.config.model_id,
                layer_idx,
            )
            out = torch.zeros_like(hidden)
            compiled.decode(
                hidden,
                layer.input_rms_weight.view(1, -1).float().cpu(),
                self._kernel_weight(layer.wq),
                self._kernel_weight(layer.wk),
                self._kernel_weight(layer.wv),
                layer.q_norm_weight.view(1, -1).float().cpu(),
                layer.k_norm_weight.view(1, -1).float().cpu(),
                decode_inputs.seq_lens,
                decode_inputs.block_table,
                decode_inputs.slot_mapping,
                compiled.rope_cos,
                compiled.rope_sin,
                k_cache,
                v_cache,
                self._kernel_weight(layer.wo),
                layer.post_rms_weight.view(1, -1).float().cpu(),
                self._kernel_weight(layer.w_gate),
                self._kernel_weight(layer.w_up),
                self._kernel_weight(layer.w_down),
                out,
                config=self._run_config(codegen_only=False),
            )
            hidden = out

        final_hidden = hidden.float()
        logits = self._project_logits(model, final_hidden)
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        return DecodeResult(hidden_states=final_hidden, logits=logits)

    def _compile_model(self, model: RuntimeModel) -> _CompiledKernels:
        _ensure_pypto_import(self._pypto_root)
        from pypto.runtime import run
        try:
            from ..model.qwen3_14b_decode import build_qwen3_decode_program
            from ..model.qwen3_14b_final_rms import build_qwen3_final_rms_program
            from ..model.qwen3_14b_lm_head import build_qwen3_lm_head_program
            from ..model.qwen3_14b_prefill import build_qwen3_14b_prefill_program
        except ImportError:
            from model.qwen3_14b_decode import build_qwen3_decode_program
            from model.qwen3_14b_final_rms import build_qwen3_final_rms_program
            from model.qwen3_14b_lm_head import build_qwen3_lm_head_program
            from model.qwen3_14b_prefill import build_qwen3_14b_prefill_program

        self._validate_supported_shape(model)
        kernel_batch = model.runtime.max_batch_size
        self._validate_total_kv_pages(model, kernel_batch)

        prefill_program = build_qwen3_14b_prefill_program(
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            intermediate_size=model.config.intermediate_size,
        )
        decode_program = build_qwen3_decode_program(
            batch=kernel_batch,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
        )
        padded_vocab = _round_up(model.config.vocab_size, _VOCAB_PAD_MULTIPLE)
        final_rms_program = build_qwen3_final_rms_program(
            batch=_LOGITS_BATCH_TILE,
            hidden_size=model.config.hidden_size,
            eps=model.config.rms_norm_eps,
        )
        lm_head_program = build_qwen3_lm_head_program(
            batch=_LOGITS_BATCH_TILE,
            hidden_size=model.config.hidden_size,
            vocab_size=padded_vocab,
        )
        prefill = run(prefill_program, config=self._run_config(codegen_only=True))
        decode = run(decode_program, config=self._run_config(codegen_only=True))
        final_rms = run(final_rms_program, config=self._run_config(codegen_only=True))
        lm_head = run(lm_head_program, config=self._run_config(codegen_only=True))
        rope_cos, rope_sin = _rope_tables(
            model.runtime.max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
        )

        lm_head_weight = model.lm_head
        if padded_vocab != lm_head_weight.shape[0]:
            pad_rows = padded_vocab - lm_head_weight.shape[0]
            padding = torch.zeros(
                (pad_rows, lm_head_weight.shape[1]),
                dtype=lm_head_weight.dtype,
                device=lm_head_weight.device,
            )
            lm_head_weight = torch.cat([lm_head_weight, padding], dim=0)
        padded_lm_head_weight = lm_head_weight.to(torch.bfloat16).contiguous().cpu()

        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            final_rms=final_rms,
            lm_head=lm_head,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
        )

    def _project_logits(self, model: RuntimeModel, hidden: torch.Tensor) -> torch.Tensor:
        compiled = self._compiled[model.config.model_id]
        hidden_size = model.config.hidden_size
        vocab_size = model.config.vocab_size
        padded_vocab = compiled.padded_vocab

        actual_batch = hidden.shape[0]
        if actual_batch > _LOGITS_BATCH_TILE:
            raise ValueError(
                f"logit batch {actual_batch} exceeds _LOGITS_BATCH_TILE {_LOGITS_BATCH_TILE}"
            )

        x = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16)
        x[:actual_batch] = hidden.to(torch.bfloat16).cpu()
        gamma = model.final_norm_weight.view(1, hidden_size).float().cpu()
        normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16)
        compiled.final_rms(
            x,
            gamma,
            normed,
            config=self._run_config(codegen_only=False),
        )

        logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32)
        compiled.lm_head(
            normed,
            compiled.padded_lm_head_weight,
            logits_padded,
            config=self._run_config(codegen_only=False),
        )
        return logits_padded[:actual_batch, :vocab_size].to(hidden.device)

    def _prepare_prefill_inputs(
        self,
        model: RuntimeModel,
        batch: PrefillBatch,
    ) -> _PrefillInputs:
        actual_batch = self._validate_batch_size(model, len(batch.kv_allocations))
        max_seq = model.runtime.max_seq_len
        hidden_size = model.config.hidden_size
        max_blocks = self._max_blocks_per_seq(model)

        hidden = torch.zeros((actual_batch, max_seq, hidden_size), dtype=torch.bfloat16)
        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        slot_mapping = torch.full((actual_batch * max_seq,), -1, dtype=torch.int32)

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            if seq_len <= 0:
                raise ValueError("prefill seq_lens must be positive")
            if seq_len > max_seq:
                raise ValueError(f"prefill seq_len {seq_len} exceeds max_seq_len {max_seq}")
            seq_lens[batch_idx] = seq_len
            hidden[batch_idx, :seq_len, :] = (
                batch.input_embeddings[batch_idx, :seq_len, :].to(torch.bfloat16).cpu()
            )
            self._write_block_table_row(block_table, batch_idx, max_blocks, alloc)
            slot_row = self._kv_cache_manager.slot_mapping_for_positions(
                alloc,
                seq_len,
                max_tokens=max_seq,
            )
            slot_mapping[batch_idx * max_seq : (batch_idx + 1) * max_seq] = slot_row

        return _PrefillInputs(
            actual_batch=actual_batch,
            hidden=hidden,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    def _prepare_decode_inputs(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
    ) -> _DecodeInputs:
        actual_batch = self._validate_batch_size(model, len(batch.kv_allocations))
        hidden_size = model.config.hidden_size
        max_blocks = self._max_blocks_per_seq(model)

        hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        seq_lens = torch.empty((actual_batch,), dtype=torch.int32)
        block_table = torch.full((actual_batch * max_blocks,), -1, dtype=torch.int32)
        slot_mapping = torch.empty((actual_batch,), dtype=torch.int32)

        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            if seq_len <= 0:
                raise ValueError("decode seq_lens must be positive")
            if seq_len > model.runtime.max_seq_len:
                raise ValueError(
                    f"decode seq_len {seq_len} exceeds max_seq_len {model.runtime.max_seq_len}"
                )
            hidden[batch_idx, :] = batch.hidden_states[batch_idx].to(torch.bfloat16).cpu()
            seq_lens[batch_idx] = seq_len
            self._write_block_table_row(block_table, batch_idx, max_blocks, alloc)
            slot_mapping[batch_idx] = self._kv_cache_manager.slot_mapping_for_request(alloc)

        return _DecodeInputs(
            actual_batch=actual_batch,
            hidden=hidden,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    @staticmethod
    def _write_block_table_row(
        block_table: torch.Tensor,
        batch_idx: int,
        max_blocks: int,
        alloc: KvAllocation,
    ) -> None:
        row_start = batch_idx * max_blocks
        if alloc.page_ids:
            block_table[row_start : row_start + len(alloc.page_ids)] = torch.tensor(
                alloc.page_ids,
                dtype=torch.int32,
            )

    @staticmethod
    def _validate_batch_size(
        model: RuntimeModel,
        actual_batch: int,
    ) -> int:
        if actual_batch <= 0:
            raise ValueError("batch must contain at least one request")
        if actual_batch > model.runtime.max_batch_size:
            max_batch_size = model.runtime.max_batch_size
            raise ValueError(
                f"batch has {actual_batch} requests, but runtime max_batch_size is {max_batch_size}"
            )
        return actual_batch

    @staticmethod
    def _max_blocks_per_seq(model: RuntimeModel) -> int:
        return (model.runtime.max_seq_len + model.runtime.page_size - 1) // model.runtime.page_size

    @classmethod
    def _validate_total_kv_pages(cls, model: RuntimeModel, kernel_batch: int) -> None:
        if model.runtime.total_kv_pages is None:
            return
        expected_pages = kernel_batch * cls._max_blocks_per_seq(model)
        if model.runtime.total_kv_pages != expected_pages:
            raise ValueError(
                "PyPTO Qwen3-14B kernels require total_kv_pages to match the runtime batch capacity: "
                f"{model.runtime.total_kv_pages} provided, {expected_pages} required."
            )

    def _run_config(self, *, codegen_only: bool):
        from pypto.runtime import RunConfig

        return RunConfig(
            platform=self._platform,
            device_id=self._device_id,
            backend_type=_backend_type_for_platform(self._platform),
            codegen_only=codegen_only,
            save_kernels=self._save_kernels_dir is not None,
            save_kernels_dir=self._save_kernels_dir,
        )

    @staticmethod
    def _kernel_weight(weight: torch.Tensor) -> torch.Tensor:
        return weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu()

    @staticmethod
    def _validate_supported_shape(model: RuntimeModel) -> None:
        config = model.config
        expected = {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        actual = {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
        }
        if actual != expected:
            mismatch = ", ".join(f"{k}={actual[k]} (expected {v})" for k, v in expected.items() if actual[k] != v)
            raise ValueError(
                "Bundled kernels under model/ currently support Qwen3-14B layer shapes only: " + mismatch
            )
        if model.runtime.page_size != _QWEN14B_PAGE_SIZE:
            raise ValueError(
                "PyPTO Qwen3-14B kernels require runtime page_size "
                f"{_QWEN14B_PAGE_SIZE}, got {model.runtime.page_size}."
            )
