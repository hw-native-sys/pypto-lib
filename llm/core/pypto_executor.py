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
from .types import DecodeBatch, DecodeResult, ModelRecord, PrefillBatch, PrefillResult, RuntimeModel


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


@dataclass
class _CompiledKernels:
    prefill: object
    decode: object
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor


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
        if len(batch.kv_allocations) != 1:
            raise NotImplementedError("Current PyPTO kernel executor supports one request per batch.")
        compiled = self._compiled[model.config.model_id]
        seq_len = int(batch.seq_lens[0].item())
        alloc = batch.kv_allocations[0]
        max_seq = model.runtime.max_seq_len
        hidden_size = model.config.hidden_size
        max_blocks = max_seq // model.runtime.page_size

        seq_lens = torch.tensor([seq_len], dtype=torch.int32)
        block_table = torch.full((max_blocks,), -1, dtype=torch.int32)
        slot_mapping = self._kv_cache_manager.slot_mapping_for_positions(alloc, seq_len, max_tokens=max_seq)
        if alloc.page_ids:
            block_table[: len(alloc.page_ids)] = torch.tensor(alloc.page_ids, dtype=torch.int32)
        hidden = torch.zeros((1, max_seq, hidden_size), dtype=torch.bfloat16)
        hidden[0, :seq_len, :] = batch.input_embeddings[0, :seq_len, :].to(torch.bfloat16).cpu()

        for layer_idx, layer in enumerate(model.layers):
            k_cache, v_cache = self._kv_cache_manager.materialize_decode_cache(model.config.model_id, layer_idx)
            out = torch.zeros_like(hidden)
            compiled.prefill(
                hidden,
                layer.input_rms_weight.view(1, -1).float().cpu(),
                self._kernel_weight(layer.wq),
                self._kernel_weight(layer.wk),
                self._kernel_weight(layer.wv),
                layer.q_norm_weight.view(1, -1).float().cpu(),
                layer.k_norm_weight.view(1, -1).float().cpu(),
                seq_lens,
                block_table,
                slot_mapping,
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
        alloc.tokens_used = max(alloc.tokens_used, seq_len)

        last_hidden = hidden[0, seq_len - 1].float()
        logits = self._project_logits(model, last_hidden)
        return PrefillResult(last_hidden=last_hidden, logits=logits)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        if len(batch.kv_allocations) != 1:
            raise NotImplementedError("Current PyPTO kernel executor supports one request per batch.")
        compiled = self._compiled[model.config.model_id]
        hidden = batch.hidden_states.to(torch.bfloat16).cpu()
        seq_lens = batch.seq_lens.to(torch.int32).cpu()
        slot_mapping = batch.slot_mapping.to(torch.int32).cpu()
        max_blocks = model.runtime.max_seq_len // model.runtime.page_size
        block_table = torch.full((max_blocks,), -1, dtype=torch.int32)
        alloc = batch.kv_allocations[0]
        if alloc.page_ids:
            block_table[: len(alloc.page_ids)] = torch.tensor(alloc.page_ids, dtype=torch.int32)

        for layer_idx, layer in enumerate(model.layers):
            k_cache, v_cache = self._kv_cache_manager.materialize_decode_cache(model.config.model_id, layer_idx)
            out = torch.zeros_like(hidden)
            compiled.decode(
                hidden,
                layer.input_rms_weight.view(1, -1).float().cpu(),
                self._kernel_weight(layer.wq),
                self._kernel_weight(layer.wk),
                self._kernel_weight(layer.wv),
                layer.q_norm_weight.view(1, -1).float().cpu(),
                layer.k_norm_weight.view(1, -1).float().cpu(),
                seq_lens,
                block_table,
                slot_mapping,
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

        final_hidden = hidden[0].float()
        logits = self._project_logits(model, final_hidden)
        alloc.tokens_used = max(alloc.tokens_used, int(seq_lens[0].item()))
        return DecodeResult(hidden_states=final_hidden, logits=logits)

    def _compile_model(self, model: RuntimeModel) -> _CompiledKernels:
        _ensure_pypto_import(self._pypto_root)
        from pypto.runtime import run
        try:
            from ..model.qwen3_14b_decode import build_qwen3_decode_program
            from ..model.qwen3_14b_prefill import build_qwen3_14b_prefill_program
        except ImportError:
            from model.qwen3_14b_decode import build_qwen3_decode_program
            from model.qwen3_14b_prefill import build_qwen3_14b_prefill_program

        self._validate_supported_shape(model)

        prefill_program = build_qwen3_14b_prefill_program(
            batch=1,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            intermediate_size=model.config.intermediate_size,
        )
        decode_program = build_qwen3_decode_program(
            batch=1,
            max_seq=model.runtime.max_seq_len,
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            num_heads=model.config.num_attention_heads,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
        )
        prefill = run(prefill_program, config=self._run_config(codegen_only=True))
        decode = run(decode_program, config=self._run_config(codegen_only=True))
        rope_cos, rope_sin = _rope_tables(model.runtime.max_seq_len, model.config.head_dim, model.config.rope_theta)
        return _CompiledKernels(prefill=prefill, decode=decode, rope_cos=rope_cos, rope_sin=rope_sin)

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
