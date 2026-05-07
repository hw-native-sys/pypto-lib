# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import contextlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

_TIMING_ENABLED = True

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
class _KernelLayerWeights:
    input_rms_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    wo: torch.Tensor
    post_rms_weight: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class _CompiledKernels:
    prefill: object
    decode: object
    final_rms: object
    lm_head: object
    final_norm_weight: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    padded_vocab: int
    padded_lm_head_weight: torch.Tensor
    layers: list[_KernelLayerWeights]
    decode_weights: dict[str, torch.Tensor]
    # L3 gen_chunked artefacts. Populated only when l3_mode=True.
    # ONE host_orch (L3) interleaves qwen3_prefill_layer + qwen3_decode_layer (L2)
    # per layer. Python loops over num_layers // chunk_size chunks.
    gen_chunked: object | None = None
    stacked_weight_chunks: list[dict[str, torch.Tensor]] | None = None
    chunk_size: int | None = None


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


class _StackedLayerView:
    """Adapter exposing HF-format LayerWeights in kernel orientation.

    stack_layer_weights() expects each per-layer weight already in the
    orientation the kernel ingests (transposed BF16 contiguous CPU). This
    view computes that view lazily per attribute access so the stacker can
    iterate ``getattr(layer, attr)`` against the standard LayerWeights.
    """

    _KERNEL_2D_ATTRS = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")

    def __init__(self, layer) -> None:
        self._layer = layer

    def __getattr__(self, name: str) -> torch.Tensor:
        weight = getattr(self._layer, name)
        if name in self._KERNEL_2D_ATTRS:
            return weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu()
        # Norm gammas (input_rms_weight, q/k/post norm). The stacker
        # flattens to [dim] then re-stacks; cast to FP32 to match the
        # kernel signature ([num_layers, dim], FP32).
        return weight.view(-1).float().contiguous().cpu()


class PyptoQwen14BExecutor(ModelExecutor):
    def __init__(
        self,
        kv_cache_manager: KvCacheManager,
        *,
        pypto_root: str | None = None,
        platform: str = "a2a3sim",
        device_id: int = 0,
        save_kernels_dir: str | None = None,
        l3_mode: bool = False,
        decode_chunk_size: int = 4,
        # Deprecated alias kept for backward compatibility.
        l3_decode: bool = False,
    ) -> None:
        super().__init__(kv_cache_manager)
        self._pypto_root = pypto_root
        self._platform = platform
        self._device_id = device_id
        self._save_kernels_dir = save_kernels_dir
        # When True, BOTH prefill and decode are dispatched in
        # `decode_chunk_size`-layer chunks via the chunked kernels, and all
        # dispatches share a single Worker(level=3) opened by session().
        # Python loops over num_layers // chunk_size chunks per prefill/token,
        # swapping per-chunk weights and bumping kv_layer_offset_base each
        # call. KV cache is a single flat tensor (zero-copy view from pool).
        self._l3_mode = l3_mode or l3_decode
        self._decode_chunk_size = decode_chunk_size
        self._compiled: dict[str, _CompiledKernels] = {}

    def register_model(self, model_id: str, record: ModelRecord) -> None:
        self._compiled[model_id] = self._compile_model(record.runtime_model)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        compiled = self._compiled[model.config.model_id]
        prefill_inputs = self._prepare_prefill_inputs(model, batch)

        if self._l3_mode:
            decode_hidden = self._run_gen_chunked_prefill(model, compiled, prefill_inputs)
            final_hidden = decode_hidden.float()
            logits = self._project_logits(model, final_hidden)
            for batch_idx, alloc in enumerate(batch.kv_allocations):
                alloc.tokens_used = max(
                    alloc.tokens_used, int(prefill_inputs.seq_lens[batch_idx].item())
                )
            return PrefillResult(last_hidden=final_hidden, logits=logits)

        hidden = prefill_inputs.hidden
        t_prefill_start = time.perf_counter()

        for layer_idx, layer in enumerate(compiled.layers):
            k_cache, v_cache = self._kv_cache_manager.materialize_decode_cache(
                model.config.model_id,
                layer_idx,
            )
            out = torch.zeros_like(hidden)
            compiled.prefill(
                hidden,
                prefill_inputs.seq_lens,
                layer.input_rms_weight,
                layer.wq,
                layer.wk,
                layer.wv,
                layer.q_norm_weight,
                layer.k_norm_weight,
                compiled.rope_cos,
                compiled.rope_sin,
                prefill_inputs.block_table,
                prefill_inputs.slot_mapping,
                k_cache,
                v_cache,
                layer.wo,
                layer.post_rms_weight,
                layer.w_gate,
                layer.w_up,
                layer.w_down,
                out,
                config=self._run_config(codegen_only=False),
            )
            hidden = out

        if _TIMING_ENABLED:
            print(
                f"[timing] prefill: {len(model.layers)} layers, "
                f"{(time.perf_counter() - t_prefill_start) * 1000:.2f} ms",
                flush=True,
            )

        last_hidden_rows: list[torch.Tensor] = []
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            seq_len = int(batch.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, seq_len)
            last_hidden_rows.append(hidden[batch_idx, seq_len - 1].float())
        last_hidden = torch.stack(last_hidden_rows)
        logits = self._project_logits(model, last_hidden)
        return PrefillResult(last_hidden=last_hidden, logits=logits)

    def run_decode(self, model: RuntimeModel, batch: DecodeBatch) -> DecodeResult:
        # The fused decode kernel (qwen3_14b_decode_full.py) processes all
        # layers in one call: weights are pre-stacked into [num_layers * ...]
        # tensors at compile time and the KV cache is the full multi-layer
        # buffer. Argument order mirrors the kernel signature in
        # build_qwen3_decode_program.qwen3_decode.
        compiled = self._compiled[model.config.model_id]
        decode_inputs = self._prepare_decode_inputs(model, batch)
        hidden = decode_inputs.hidden
        dw = compiled.decode_weights

        if self._l3_mode:
            hidden = self._run_gen_chunked_decode(model, compiled, decode_inputs)
            final_hidden = hidden.float()
        else:
            k_cache, v_cache = self._kv_cache_manager.materialize_decode_cache_all_layers(
                model.config.model_id,
            )
            out = torch.zeros_like(hidden)

            compiled.decode(
                hidden,
                dw["decode_input_rms_weight"],
                dw["decode_wq"],
                dw["decode_wk"],
                dw["decode_wv"],
                dw["decode_q_norm_weight"],
                dw["decode_k_norm_weight"],
                decode_inputs.seq_lens,
                decode_inputs.block_table,
                decode_inputs.slot_mapping,
                compiled.rope_cos,
                compiled.rope_sin,
                k_cache,
                v_cache,
                dw["decode_wo"],
                dw["decode_post_rms_weight"],
                dw["decode_w_gate"],
                dw["decode_w_up"],
                dw["decode_w_down"],
                out,
                config=self._run_config(codegen_only=False),
            )

            final_hidden = out.float()

        logits = self._project_logits(model, final_hidden)
        for batch_idx, alloc in enumerate(batch.kv_allocations):
            alloc.tokens_used = max(alloc.tokens_used, int(batch.seq_lens[batch_idx].item()))
        return DecodeResult(hidden_states=final_hidden, logits=logits)

    def _run_gen_chunked_prefill(
        self,
        model: RuntimeModel,
        compiled: _CompiledKernels,
        prefill_inputs: _PrefillInputs,
    ) -> torch.Tensor:
        """Combined prefill + first-decode dispatch (has_prefill=1).

        For each layer chunk the ONE host_orch (L3) runs:
          1. qwen3_prefill_layer (L2) — writes KV cache for all prompt positions.
          2. qwen3_decode_layer  (L2) — attends to full KV cache just written;
             output is the decode hidden state that predicts the first new token.

        The decode input is the last prompt token's embedding extracted from the
        prefill hidden tensor. decode_seq_lens = prefill_seq_lens (= N) so the
        decode kernel uses RoPE position N-1 and attends to all N KV entries.
        """
        if compiled.gen_chunked is None or compiled.stacked_weight_chunks is None:
            raise RuntimeError(
                "L3 gen_chunked prefill requested but artefacts not compiled. "
                "Construct the executor with l3_mode=True."
            )
        chunks = compiled.stacked_weight_chunks
        chunk_size = compiled.chunk_size
        num_chunks = len(chunks)
        actual_batch = prefill_inputs.actual_batch
        hidden_size = model.config.hidden_size
        max_seq = model.runtime.max_seq_len

        k_cache_all, v_cache_all = self._kv_cache_manager.materialize_decode_cache_all_layers(
            model.config.model_id,
        )

        # Build initial decode hidden: last prompt token embedding per batch item.
        decode_hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        decode_slot_mapping = torch.zeros((actual_batch,), dtype=torch.int32)
        for b in range(actual_batch):
            seq_len_b = int(prefill_inputs.seq_lens[b].item())
            decode_hidden[b] = prefill_inputs.hidden[b, seq_len_b - 1, :]
            decode_slot_mapping[b] = int(
                prefill_inputs.slot_mapping[b * max_seq + seq_len_b - 1].item()
            )

        has_prefill = torch.tensor(True, dtype=torch.bool)
        prefill_out = torch.zeros_like(prefill_inputs.hidden)
        decode_out = torch.zeros_like(decode_hidden)

        for c in range(num_chunks):
            sw = chunks[c]
            kv_offset_scalar = torch.tensor(c * chunk_size, dtype=torch.int32)
            prefill_in = prefill_inputs.hidden if c == 0 else prefill_out
            decode_in = decode_hidden if c == 0 else decode_out
            compiled.gen_chunked(
                prefill_in,
                prefill_inputs.seq_lens,
                prefill_inputs.slot_mapping,
                decode_in,
                prefill_inputs.seq_lens,   # decode_seq_lens = prefill_seq_lens (= N)
                decode_slot_mapping,
                sw["input_rms_chunk"],
                sw["wq_chunk_flat"],
                sw["wk_chunk_flat"],
                sw["wv_chunk_flat"],
                sw["q_norm_chunk"],
                sw["k_norm_chunk"],
                compiled.rope_cos,
                compiled.rope_sin,
                prefill_inputs.block_table,
                k_cache_all,
                v_cache_all,
                sw["wo_chunk_flat"],
                sw["post_rms_chunk"],
                sw["w_gate_chunk_flat"],
                sw["w_up_chunk_flat"],
                sw["w_down_chunk_flat"],
                kv_offset_scalar,
                has_prefill,
                prefill_out,
                decode_out,
                config=self._run_config(codegen_only=False),
            )
        return decode_out

    def _run_gen_chunked_decode(
        self,
        model: RuntimeModel,
        compiled: _CompiledKernels,
        decode_inputs: _DecodeInputs,
    ) -> torch.Tensor:
        """Pure-decode dispatch (has_prefill=0).

        The ONE host_orch (L3) skips qwen3_prefill_layer; only qwen3_decode_layer
        (L2) runs for each layer in the chunk. Prefill tensors are dummies.
        """
        if compiled.gen_chunked is None or compiled.stacked_weight_chunks is None:
            raise RuntimeError(
                "L3 gen_chunked decode requested but artefacts not compiled. "
                "Construct the executor with l3_mode=True."
            )
        chunks = compiled.stacked_weight_chunks
        chunk_size = compiled.chunk_size
        num_chunks = len(chunks)
        actual_batch = decode_inputs.actual_batch
        hidden_size = model.config.hidden_size
        max_seq = model.runtime.max_seq_len

        k_cache_all, v_cache_all = self._kv_cache_manager.materialize_decode_cache_all_layers(
            model.config.model_id,
        )

        # Dummy prefill tensors — passed but not read by the kernel (has_prefill=0).
        dummy_prefill = torch.zeros((actual_batch, max_seq, hidden_size), dtype=torch.bfloat16)
        dummy_seq_lens = decode_inputs.seq_lens
        dummy_slot_mapping = torch.full((actual_batch * max_seq,), -1, dtype=torch.int32)
        dummy_prefill_out = torch.zeros_like(dummy_prefill)

        has_prefill = torch.tensor(False, dtype=torch.bool)
        decode_out = torch.zeros_like(decode_inputs.hidden)

        for c in range(num_chunks):
            sw = chunks[c]
            kv_offset_scalar = torch.tensor(c * chunk_size, dtype=torch.int32)
            decode_in = decode_inputs.hidden if c == 0 else decode_out
            compiled.gen_chunked(
                dummy_prefill,
                dummy_seq_lens,
                dummy_slot_mapping,
                decode_in,
                decode_inputs.seq_lens,
                decode_inputs.slot_mapping,
                sw["input_rms_chunk"],
                sw["wq_chunk_flat"],
                sw["wk_chunk_flat"],
                sw["wv_chunk_flat"],
                sw["q_norm_chunk"],
                sw["k_norm_chunk"],
                compiled.rope_cos,
                compiled.rope_sin,
                decode_inputs.block_table,
                k_cache_all,
                v_cache_all,
                sw["wo_chunk_flat"],
                sw["post_rms_chunk"],
                sw["w_gate_chunk_flat"],
                sw["w_up_chunk_flat"],
                sw["w_down_chunk_flat"],
                kv_offset_scalar,
                has_prefill,
                dummy_prefill_out,
                decode_out,
                config=self._run_config(codegen_only=False),
            )
        return decode_out

    def _compile_model(self, model: RuntimeModel) -> _CompiledKernels:
        _ensure_pypto_import(self._pypto_root)
        from pypto.runtime import run
        try:
            from ..model.qwen3_14b_decode import build_qwen3_decode_program
            from ..model.qwen3_14b_final_rms import build_qwen3_final_rms_program
            from ..model.qwen3_14b_gen_chunked import (
                build_qwen3_14b_gen_chunked_program,
                stack_layer_weights_chunked,
            )
            from ..model.qwen3_14b_lm_head import build_qwen3_lm_head_program
            from ..model.qwen3_14b_prefill import build_qwen3_14b_prefill_program
        except ImportError:
            from model.qwen3_14b_decode import build_qwen3_decode_program
            from model.qwen3_14b_final_rms import build_qwen3_final_rms_program
            from model.qwen3_14b_gen_chunked import (
                build_qwen3_14b_gen_chunked_program,
                stack_layer_weights_chunked,
            )
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
            num_layers=model.config.num_hidden_layers,
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

        # L3 gen_chunked artefacts (built only when l3_mode=True).
        # ONE host_orch (L3) interleaves prefill+decode or decode-only per chunk.
        gen_chunked: object | None = None
        stacked_weight_chunks: list[dict[str, torch.Tensor]] | None = None
        if self._l3_mode:
            gen_chunked_program = build_qwen3_14b_gen_chunked_program(
                num_layers=model.config.num_hidden_layers,
                chunk_size=self._decode_chunk_size,
                batch=kernel_batch,
                max_seq=model.runtime.max_seq_len,
                hidden_size=model.config.hidden_size,
                intermediate_size=model.config.intermediate_size,
                num_heads=model.config.num_attention_heads,
                num_kv_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
            )
            # gen_chunked uses in-place pl.unroll with consecutive dispatches sharing
            # the same output tensor handles (WAW).  The simpler scheduler resolves WAW
            # ordering only via tensor addresses (scalars like local_layer_idx are
            # ignored), so block_dim=3 + aicpu_thread_num=4 are required — exactly the
            # DistributedConfig that makes l3_two_l2._build_with_flag pass.
            # pypto.runtime.run hard-codes DistributedConfig(block_dim=1), so we call
            # ir.compile directly to override it.
            from pypto import ir  # noqa: PLC0415
            from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

            _rc = self._run_config(codegen_only=True)
            gen_chunked = ir.compile(
                gen_chunked_program,
                output_dir=_rc.save_kernels_dir,
                strategy=_rc.strategy,
                backend_type=_rc.backend_type,
                dump_passes=_rc.dump_passes,
                diagnostic_phase=_rc.diagnostic_phase,
                disabled_diagnostics=_rc.disabled_diagnostics,
                platform=_rc.platform,
                profiling=_rc.compile_profiling,
                distributed_config=DistributedConfig(
                    device_ids=[self._device_id],
                    block_dim=3,
                    num_sub_workers=0,
                    aicpu_thread_num=4,
                ),
            )
            # stack_layer_weights_chunked expects each weight already in
            # kernel orientation ([in_dim, out_dim] for 2D matmul weights).
            # Adapt via a per-layer view that mirrors _kernel_weight().
            stacked_layers = [_StackedLayerView(layer) for layer in model.layers]
            stacked_weight_chunks = stack_layer_weights_chunked(
                stacked_layers,
                chunk_size=self._decode_chunk_size,
                hidden=model.config.hidden_size,
                kv_hidden=model.config.num_key_value_heads * model.config.head_dim,
                inter=model.config.intermediate_size,
                head_dim=model.config.head_dim,
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
        layers = []
        for layer in model.layers:
            layers.append(self._kernel_layer_weights(layer))
            self._release_layer_weights(layer)
        final_norm_weight = model.final_norm_weight.view(1, -1).float().cpu()

        decode_weights = self._stack_decode_weights(layers)

        return _CompiledKernels(
            prefill=prefill,
            decode=decode,
            final_rms=final_rms,
            lm_head=lm_head,
            final_norm_weight=final_norm_weight,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            padded_vocab=padded_vocab,
            padded_lm_head_weight=padded_lm_head_weight,
            layers=layers,
            decode_weights=decode_weights,
            gen_chunked=gen_chunked,
            stacked_weight_chunks=stacked_weight_chunks,
            chunk_size=self._decode_chunk_size if self._l3_mode else None,
        )

    @staticmethod
    def _stack_decode_weights(layers: list[_KernelLayerWeights]) -> dict[str, torch.Tensor]:
        # Stack from already-prepared per-layer kernel weights. Each
        # _KernelLayerWeights field is already in the kernel-ready shape/dtype
        # (transposed bf16 cpu for projections, [1, N] float cpu for norms),
        # so a plain cat along dim 0 is all that's left. Reading from the
        # original model.layers here would crash because _release_layer_weights
        # has already replaced those tensors with torch.empty(0).
        def cat(attr: str) -> torch.Tensor:
            return torch.cat([getattr(l, attr) for l in layers], dim=0)

        return {
            "decode_input_rms_weight": cat("input_rms_weight").contiguous(),
            "decode_wq":               cat("wq"),
            "decode_wk":               cat("wk"),
            "decode_wv":               cat("wv"),
            "decode_q_norm_weight":    cat("q_norm_weight").contiguous(),
            "decode_k_norm_weight":    cat("k_norm_weight").contiguous(),
            "decode_wo":               cat("wo"),
            "decode_post_rms_weight":  cat("post_rms_weight").contiguous(),
            "decode_w_gate":           cat("w_gate"),
            "decode_w_up":             cat("w_up"),
            "decode_w_down":           cat("w_down"),
        }

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
        normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16)
        compiled.final_rms(
            x,
            compiled.final_norm_weight,
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

    @contextlib.contextmanager
    def session(self):
        """Lifecycle context spanning one full generate sequence.

        For L3 mode: the chunked kernels (prefill_chunked, decode_chunked)
        dispatch via chip_process internally — the parent process must NOT
        hold an active device context when chip_process is spawned, so this
        context is intentionally a no-op for L3 mode.  When pypto adds user-
        API support for a persistent Worker(level=3), this is where it would
        be opened.

        For baseline mode: also a no-op; each run_prefill / run_decode
        manages its own _l2_device_context.

        The engine calls executor.session() unconditionally so that the
        generate lifecycle is ready for the future persistent-Worker upgrade.
        """
        yield

    @contextlib.contextmanager
    def _l2_device_context(self):
        """Hold a single level-2 device context open across all inner run() calls.

        Used by the baseline (non-L3) paths for prefill and decode. Without
        this wrapper every compiled(...) call creates and destroys its own
        Worker(level=2), resulting in repeated aclrtSetDevice / aclrtResetDevice
        cycles that can deplete the AICPU stream pool (error 507899).

        Inside this block all execute_on_device calls detect the active Worker
        via _PyptoWorker.current() and reuse its already-initialised device
        context, so only one SetDevice / ResetDevice pair occurs regardless of
        how many kernel dispatches happen within the block.

        Do NOT use this wrapper when l3_mode=True: the L3 chunked paths
        (_run_prefill_l3_chunked, _run_decode_l3_chunked) dispatch via
        execute_distributed (level=3 chip_process subprocess) inside the
        shared Worker opened by session(), and must not have a level-2
        context active on entry.
        """
        from pypto.runtime import Worker  # noqa: PLC0415

        with Worker(config=self._run_config(codegen_only=False)):
            yield

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

    @classmethod
    def _kernel_layer_weights(cls, layer) -> _KernelLayerWeights:
        return _KernelLayerWeights(
            input_rms_weight=layer.input_rms_weight.view(1, -1).float().cpu(),
            wq=cls._kernel_weight(layer.wq),
            wk=cls._kernel_weight(layer.wk),
            wv=cls._kernel_weight(layer.wv),
            q_norm_weight=layer.q_norm_weight.view(1, -1).float().cpu(),
            k_norm_weight=layer.k_norm_weight.view(1, -1).float().cpu(),
            wo=cls._kernel_weight(layer.wo),
            post_rms_weight=layer.post_rms_weight.view(1, -1).float().cpu(),
            w_gate=cls._kernel_weight(layer.w_gate),
            w_up=cls._kernel_weight(layer.w_up),
            w_down=cls._kernel_weight(layer.w_down),
        )

    @staticmethod
    def _release_layer_weights(layer) -> None:
        empty = torch.empty(0)
        layer.input_rms_weight = empty
        layer.wq = empty
        layer.wk = empty
        layer.wv = empty
        layer.q_norm_weight = empty
        layer.k_norm_weight = empty
        layer.wo = empty
        layer.post_rms_weight = empty
        layer.w_gate = empty
        layer.w_up = empty
        layer.w_down = empty

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
