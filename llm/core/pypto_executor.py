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

from ._profiling import StageTimer
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


def _patch_orch_make_tensor_arg(module: object) -> None:
    """Allow ``make_tensor_arg`` in a generated orchestration module to pass
    ``ContinuousTensor`` objects through unchanged.

    The generated ``host_orch.py`` always calls ``make_tensor_arg(t)`` for
    every entry in the ``tensors`` dict.  When we pre-upload static weights
    and replace them with ``ContinuousTensor(child_memory=True)`` objects,
    the default ``make_tensor_arg`` (which expects a ``torch.Tensor``) would
    crash.  Patching it here lets child_memory tensors pass through as-is
    so the runtime skips H2D/D2H for those buffers.
    """
    try:
        from simpler.task_interface import ContinuousTensor  # noqa: PLC0415
    except ImportError:
        return
    _orig = getattr(module, "make_tensor_arg", None)
    if _orig is None or getattr(_orig, "_child_memory_patched", False):
        return

    def _patched(tensor: object) -> object:
        if isinstance(tensor, ContinuousTensor):
            return tensor
        return _orig(tensor)  # type: ignore[misc]

    _patched._child_memory_patched = True  # type: ignore[attr-defined]
    module.make_tensor_arg = _patched  # type: ignore[attr-defined]


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
    # L3-wrapped generate artifacts. Populated only when l3_mode=True.
    stacked_weights: dict[str, torch.Tensor] | None = None
    l3_generate_chip_callables: dict[str, object] | None = None
    l3_generate_entry_fn: object | None = None
    l3_generate_sub_worker_fns: dict[str, object] | None = None
    l3_generate_dc: object | None = None  # DistributedConfig
    l3_generate_platform: str | None = None
    l3_generate_runtime_name: str | None = None
    l3_generate_param_infos: object | None = None


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
        l3_trace: bool = False,
    ) -> None:
        super().__init__(kv_cache_manager)
        self._pypto_root = pypto_root
        self._platform = platform
        self._device_id = device_id
        self._save_kernels_dir = save_kernels_dir
        # When True, the engine uses run_generate_l3 which dispatches the
        # full prefill + decode loop inside a single Worker(level=3).
        self._l3_mode = l3_mode
        # When True, run_generate_l3 emits per-stage timestamp prints.
        self._l3_trace = l3_trace
        self._compiled: dict[str, _CompiledKernels] = {}

    def register_model(self, model_id: str, record: ModelRecord) -> None:
        self._compiled[model_id] = self._compile_model(record.runtime_model)

    def run_prefill(self, model: RuntimeModel, batch: PrefillBatch) -> PrefillResult:
        compiled = self._compiled[model.config.model_id]
        prefill_inputs = self._prepare_prefill_inputs(model, batch)
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

    def _compile_model(self, model: RuntimeModel) -> _CompiledKernels:
        timer = StageTimer(
            enabled=self._l3_trace,
            prefix="compile-breakdown",
            title="_compile_model stage timings",
        )

        def _mark(label: str) -> None:
            timer.mark(label)

        _ensure_pypto_import(self._pypto_root)
        from pypto.runtime import run
        try:
            from ..model.qwen3_14b_decode import build_qwen3_decode_program
            from ..model.qwen3_14b_final_rms import build_qwen3_final_rms_program
            from ..model.qwen3_14b_l3_generate import (
                build_qwen3_14b_l3_generate_program,
                stack_layer_weights_full,
            )
            from ..model.qwen3_14b_lm_head import build_qwen3_lm_head_program
            from ..model.qwen3_14b_prefill import build_qwen3_14b_prefill_program
        except ImportError:
            from model.qwen3_14b_decode import build_qwen3_decode_program
            from model.qwen3_14b_final_rms import build_qwen3_final_rms_program
            from model.qwen3_14b_l3_generate import (
                build_qwen3_14b_l3_generate_program,
                stack_layer_weights_full,
            )
            from model.qwen3_14b_lm_head import build_qwen3_lm_head_program
            from model.qwen3_14b_prefill import build_qwen3_14b_prefill_program
        _mark("imports")

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
        _mark("build_programs")
        prefill = run(prefill_program, config=self._run_config(codegen_only=True))
        _mark("compile_prefill")
        decode = run(decode_program, config=self._run_config(codegen_only=True))
        _mark("compile_decode")
        final_rms = run(final_rms_program, config=self._run_config(codegen_only=True))
        lm_head = run(lm_head_program, config=self._run_config(codegen_only=True))
        _mark("compile_final_rms_lm_head")
        rope_cos, rope_sin = _rope_tables(
            model.runtime.max_seq_len,
            model.config.head_dim,
            model.config.rope_theta,
        )

        _mark("rope_tables")

        # L3 artefacts (built only when l3_mode=True).
        l3_generate: object | None = None
        stacked_weights: dict[str, torch.Tensor] | None = None
        if self._l3_mode:
            l3_generate_program = build_qwen3_14b_l3_generate_program(
                num_layers=model.config.num_hidden_layers,
                batch=kernel_batch,
                max_seq=model.runtime.max_seq_len,
                hidden_size=model.config.hidden_size,
                intermediate_size=model.config.intermediate_size,
                num_heads=model.config.num_attention_heads,
                num_kv_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
                max_new_tokens=model.runtime.max_new_tokens,
                padded_vocab=padded_vocab,
                page_size=model.runtime.page_size,
            )
            # pypto.runtime.run hard-codes DistributedConfig(block_dim=1), so we call
            # ir.compile directly to override it.
            from pypto import ir  # noqa: PLC0415
            from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

            _rc = self._run_config(codegen_only=True)
            l3_generate = ir.compile(
                l3_generate_program,
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
            _mark("compile_l3_generate")
            # stack_layer_weights_full expects each weight already in
            # kernel orientation ([in_dim, out_dim] for 2D matmul weights).
            # Adapt via a per-layer view that mirrors _kernel_weight().
            stacked_layers = [_StackedLayerView(layer) for layer in model.layers]
            stacked_weights = stack_layer_weights_full(
                stacked_layers,
                hidden=model.config.hidden_size,
                kv_hidden=model.config.num_key_value_heads * model.config.head_dim,
                inter=model.config.intermediate_size,
                head_dim=model.config.head_dim,
            )
            _mark("stack_layer_weights")

        # L3-wrapped generate: pre-extract l3_generate setup artifacts
        # (expensive compile_and_assemble + module loading done once,
        # reused per generate call).
        l3_generate_chip_callables: dict[str, object] | None = None
        l3_generate_entry_fn: object | None = None
        l3_generate_sub_worker_fns: dict[str, object] | None = None
        l3_generate_dc: object | None = None
        l3_generate_platform: str | None = None
        l3_generate_runtime_name: str | None = None
        l3_generate_param_infos: object | None = None
        if self._l3_mode:
            from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415
            from pypto.runtime.distributed_runner import _load_generated_module  # noqa: PLC0415
            from pypto.pypto_core.ir import FunctionType  # noqa: PLC0415

            # Pre-extract l3_generate setup artifacts.
            lg_dc = l3_generate._distributed_config
            lg_output_dir = l3_generate.output_dir
            lg_chip_callables: dict[str, object] = {}
            lg_runtime_name = "tensormap_and_ringbuffer"
            lg_next_levels_dir = lg_output_dir / "next_levels"
            for func in l3_generate._program.functions.values():
                if func.func_type in (FunctionType.Orchestration, FunctionType.Opaque):
                    chip_dir = lg_next_levels_dir / func.name
                    if chip_dir.exists():
                        cc, lg_runtime_name = compile_and_assemble(
                            chip_dir, l3_generate.platform,
                        )
                        lg_chip_callables[func.name] = cc
            lg_orch_path = lg_output_dir / "orchestration" / "host_orch.py"
            lg_orch_module = _load_generated_module(lg_orch_path)
            # Patch make_tensor_arg so pre-uploaded ContinuousTensor objects
            # (child_memory=True) are passed through unchanged instead of
            # triggering a crash when the generated code calls make_tensor_arg
            # on a non-torch.Tensor value.
            _patch_orch_make_tensor_arg(lg_orch_module)
            lg_entry_fn = None
            for attr_name in ("entry", "host_orch"):
                lg_entry_fn = getattr(lg_orch_module, attr_name, None)
                if lg_entry_fn is not None:
                    break
            if lg_entry_fn is None:
                for name in dir(lg_orch_module):
                    obj = getattr(lg_orch_module, name)
                    if callable(obj) and not name.startswith("_"):
                        lg_entry_fn = obj
                        break
            lg_sub_worker_fns: dict[str, object] = {}
            lg_sub_workers_dir = lg_output_dir / "sub_workers"
            if lg_sub_workers_dir.exists():
                for py_file in sorted(lg_sub_workers_dir.glob("*.py")):
                    mod = _load_generated_module(py_file)
                    fn_name = py_file.stem
                    fn = getattr(mod, fn_name, None)
                    if fn is not None:
                        lg_sub_worker_fns[fn_name] = fn
            lg_param_infos, _, _ = l3_generate._get_metadata()

            l3_generate_chip_callables = lg_chip_callables
            l3_generate_entry_fn = lg_entry_fn
            l3_generate_sub_worker_fns = lg_sub_worker_fns
            l3_generate_dc = lg_dc
            l3_generate_platform = l3_generate.platform
            l3_generate_runtime_name = lg_runtime_name
            l3_generate_param_infos = lg_param_infos
            _mark("l3_extract_artifacts")

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
        _mark("pad_lm_head")
        layers = []
        for layer in model.layers:
            layers.append(self._kernel_layer_weights(layer))
            self._release_layer_weights(layer)
        final_norm_weight = model.final_norm_weight.view(1, -1).float().cpu()
        _mark("kernel_layer_weights")

        decode_weights = self._stack_decode_weights(layers)
        _mark("stack_decode_weights")

        timer.report()

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
            stacked_weights=stacked_weights,
            l3_generate_chip_callables=l3_generate_chip_callables,
            l3_generate_entry_fn=l3_generate_entry_fn,
            l3_generate_sub_worker_fns=l3_generate_sub_worker_fns,
            l3_generate_dc=l3_generate_dc,
            l3_generate_platform=l3_generate_platform,
            l3_generate_runtime_name=l3_generate_runtime_name,
            l3_generate_param_infos=l3_generate_param_infos,
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

    # ── L3-wrapped generate: entire prefill + decode loop in one worker.run() ──

    def run_generate_l3(
        self,
        model: RuntimeModel,
        prefill_batch: PrefillBatch,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> tuple[list[int], torch.Tensor]:
        """Run the full generate loop inside a single Worker(level=3).

        Dispatches prefill chunks + final_rms + lm_head + decode loop entirely
        within one worker.run() call, using sub_worker for CPU-side sampling
        and embedding lookup between device dispatches.

        Returns (generated_token_ids, final_hidden).
        """
        from simpler.task_interface import CallConfig  # noqa: PLC0415
        from simpler.worker import Worker  # noqa: PLC0415

        _verbose = self._l3_trace
        timer = StageTimer(
            enabled=_verbose,
            prefix="L3-breakdown",
            title="run_generate_l3 stage timings",
        )

        def _mark(label: str) -> None:
            timer.mark(label)

        compiled = self._compiled[model.config.model_id]
        if compiled.l3_generate_chip_callables is None:
            raise RuntimeError("L3 generate artifacts not compiled.")
        if max_new_tokens > model.runtime.max_new_tokens:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} exceeds compiled L3 limit "
                f"{model.runtime.max_new_tokens}"
            )

        _mark("entry_validate")
        prefill_inputs = self._prepare_prefill_inputs(model, prefill_batch)
        _mark("prepare_prefill_inputs")
        actual_batch = prefill_inputs.actual_batch
        if actual_batch != 1:
            raise ValueError(
                "run_generate_l3 currently supports batch_size=1 only; "
                f"got {actual_batch} requests."
            )
        hidden_size = model.config.hidden_size
        max_seq = model.runtime.max_seq_len
        vocab_size = model.config.vocab_size
        padded_vocab = compiled.padded_vocab

        k_cache_all, v_cache_all = self._kv_cache_manager.materialize_decode_cache_all_layers(
            model.config.model_id,
        )
        _mark("kv_cache_materialize")

        # Build initial decode hidden: last prompt token embedding per batch.
        decode_hidden = torch.zeros((actual_batch, hidden_size), dtype=torch.bfloat16)
        decode_slot_mapping = torch.zeros((actual_batch,), dtype=torch.int32)
        for b in range(actual_batch):
            seq_len_b = int(prefill_inputs.seq_lens[b].item())
            decode_hidden[b] = prefill_inputs.hidden[b, seq_len_b - 1, :]
            decode_slot_mapping[b] = int(
                prefill_inputs.slot_mapping[b * max_seq + seq_len_b - 1].item()
            )

        # Pre-allocate all shared-memory tensors for the full generate loop.
        prefill_out = torch.zeros_like(prefill_inputs.hidden).share_memory_()
        # decode_out and rms_x share one padded buffer so no CPU copy is needed
        # between them.  l3_generate writes to decode_out (the first actual_batch
        # rows); final_rms reads rms_x (all _LOGITS_BATCH_TILE rows).  The padding
        # rows stay zero throughout, satisfying the zero-pad contract of final_rms.
        _decode_out_storage = torch.zeros(
            (_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16
        ).share_memory_()
        decode_out = _decode_out_storage[:actual_batch, :]   # [actual_batch, hidden_size]
        rms_x = _decode_out_storage                          # [_LOGITS_BATCH_TILE, hidden_size]
        # final_rms / lm_head intermediates.
        rms_gamma = model.final_norm_weight.view(1, hidden_size).float().cpu().share_memory_()
        rms_normed = torch.zeros((_LOGITS_BATCH_TILE, hidden_size), dtype=torch.bfloat16).share_memory_()
        lm_head_weight = compiled.padded_lm_head_weight.share_memory_()
        logits_padded = torch.zeros((_LOGITS_BATCH_TILE, padded_vocab), dtype=torch.float32).share_memory_()
        _mark("alloc_io_buffers")
        # Sub-worker communication tensors.
        embed_tokens = model.embed_tokens.to(torch.bfloat16).cpu().share_memory_()
        _mark("embed_tokens_to_shm")
        done_flag = torch.zeros((1,), dtype=torch.int32).share_memory_()
        generated_ids = torch.full((max_new_tokens,), -1, dtype=torch.int64).share_memory_()
        token_count = torch.zeros((1,), dtype=torch.int32).share_memory_()
        # Mutable decode inputs (updated by sub_worker between steps).
        decode_hidden_buf = decode_hidden.clone().share_memory_()
        decode_seq_lens = prefill_inputs.seq_lens.clone().share_memory_()
        decode_slot_mapping_buf = decode_slot_mapping.clone().share_memory_()

        # Ensure all prefill inputs are in shared memory.
        prefill_hidden = prefill_inputs.hidden.share_memory_()
        prefill_seq_lens = prefill_inputs.seq_lens.share_memory_()
        prefill_slot_mapping = prefill_inputs.slot_mapping.share_memory_()
        block_table = prefill_inputs.block_table.share_memory_()
        rope_cos = compiled.rope_cos.share_memory_()
        rope_sin = compiled.rope_sin.share_memory_()
        k_cache_all = k_cache_all.share_memory_()
        v_cache_all = v_cache_all.share_memory_()
        _mark("kv_and_prefill_to_shm")

        # Ensure stacked weights are in shared memory.
        sm_sw = {}
        for k, v in compiled.stacked_weights.items():
            sm_sw[k] = v.share_memory_() if not v.is_shared() else v
        _mark("stacked_weights_to_shm")

        # SSA-name substrings that identify static weight parameters in the
        # tensors_dict built by _build_full_tensors().  Used in
        # generate_orch_fn to pre-upload these tensors once per generate call
        # (child_memory=True) so all decode dispatches skip H2D re-upload.
        _sw_substrings: frozenset[str] = frozenset(sm_sw.keys()) | {
            "rope_cos", "rope_sin", "final_norm_weight", "lm_head_weight",
        }

        # ── Sub-worker callable ──
        # Runs in a forked child process. Reads logits → argmax → embedding lookup
        # → writes decode_hidden_buf / decode_seq_lens / decode_slot_mapping_buf.
        _eos_id = eos_token_id
        _vocab = vocab_size
        _actual_batch = actual_batch
        _page_size = model.runtime.page_size

        # Pre-capture references for the sub_worker closure.
        # These are shared-memory tensors visible in the forked child.
        _embed_tokens = embed_tokens
        _done_flag = done_flag
        _generated_ids = generated_ids
        _token_count = token_count
        _decode_hidden_buf = decode_hidden_buf
        _decode_seq_lens = decode_seq_lens
        _decode_slot_mapping_buf = decode_slot_mapping_buf
        _logits_padded = logits_padded
        _prefill_batch = prefill_batch

        # L3 tracing: enabled by the executor's l3_trace flag (typically wired
        # to --profile-verbose). When disabled, all per-stage timestamp prints
        # are suppressed.
        _l3_trace_enabled = self._l3_trace

        # Shared-memory anchors so forked sub-worker callbacks can print times
        # relative to a common start (set at prefill submit_start in the parent).
        # Slot 0: start anchor (perf_counter seconds).
        # Slot 1: timestamp of the previous printed event (for Δprev).
        _t_anchors = torch.zeros(2, dtype=torch.float64).share_memory_()

        def _fmt_rel(t_now: float) -> str:
            t0 = float(_t_anchors[0].item())
            t_prev = float(_t_anchors[1].item())
            if t0 <= 0.0:
                return "t=+0.000ms (Δ+0.000ms)"
            rel_ms = (t_now - t0) * 1000.0
            d_ms = (t_now - t_prev) * 1000.0 if t_prev > 0.0 else 0.0
            _t_anchors[1] = t_now
            return f"t=+{rel_ms:.2f}ms (Δ+{d_ms:.2f}ms)"

        # Track when each sample_and_prepare finishes so we can split
        # chip-task time vs sub-worker IPC + work time.
        _sample_done_ts = [0.0]  # timestamp of last sample_and_prepare return

        def sample_and_prepare_fn(task_args):
            """Sub-worker: sample token from logits, prepare next decode inputs."""
            _t_enter = time.perf_counter()
            if _done_flag[0].item():
                return  # EOS already hit, no-op.

            # Read logits (written by lm_head chip task into logits_padded).
            logits = _logits_padded[0, :_vocab]
            token_id = int(logits.argmax().item())

            step = int(_token_count[0].item())
            if _l3_trace_enabled:
                _prev_done = _sample_done_ts[0]
                chip_ms = (_t_enter - _prev_done) * 1000.0 if _prev_done > 0.0 else 0.0
                print(
                    f"[L3-step] step={step:02d} sample_enter {_fmt_rel(_t_enter)}"
                    f"  chip_tasks={chip_ms:.1f}ms",
                    flush=True,
                )
            _generated_ids[step] = token_id
            _token_count[0] = step + 1

            if _eos_id is not None and token_id == _eos_id:
                _done_flag[0] = 1
                _t_exit = time.perf_counter()
                _sample_done_ts[0] = _t_exit
                if _l3_trace_enabled:
                    work_ms = (_t_exit - _t_enter) * 1000.0
                    print(
                        f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                        f"  sample_work={work_ms:.1f}ms",
                        flush=True,
                    )
                return

            if step + 1 >= max_new_tokens:
                _done_flag[0] = 1
                _t_exit = time.perf_counter()
                _sample_done_ts[0] = _t_exit
                if _l3_trace_enabled:
                    work_ms = (_t_exit - _t_enter) * 1000.0
                    print(
                        f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                        f"  sample_work={work_ms:.1f}ms",
                        flush=True,
                    )
                return

            # Embedding lookup.
            _decode_hidden_buf[0, :] = _embed_tokens[token_id]

            # Update seq_lens.
            for b in range(_actual_batch):
                new_seq_len = int(_decode_seq_lens[b].item()) + 1
                _decode_seq_lens[b] = new_seq_len
                # Update slot_mapping for next position.
                alloc = _prefill_batch.kv_allocations[b]
                page_idx = (new_seq_len - 1) // _page_size
                slot_in_page = (new_seq_len - 1) % _page_size
                if page_idx < len(alloc.page_ids):
                    _decode_slot_mapping_buf[b] = alloc.page_ids[page_idx] * _page_size + slot_in_page

            _t_exit = time.perf_counter()
            _sample_done_ts[0] = _t_exit
            if _l3_trace_enabled:
                work_ms = (_t_exit - _t_enter) * 1000.0
                print(
                    f"[L3-step] step={step:02d} sample_exit  {_fmt_rel(_t_exit)}"
                    f"  sample_work={work_ms:.1f}ms",
                    flush=True,
                )

        # ── Build the orchestrator function ──

        lg_entry_fn = compiled.l3_generate_entry_fn
        lg_chip_callables = compiled.l3_generate_chip_callables
        lg_dc = compiled.l3_generate_dc
        lg_param_infos = compiled.l3_generate_param_infos

        def _submit_l3_generate(orch, config, tensors_dict, _keep):
            """Submit one l3_generate entry_fn call (dispatches all-layers L2 tasks)."""
            lg_entry_fn(
                orch, None, config,
                tensors=tensors_dict,
                callables=lg_chip_callables,
                sub_ids=sub_ids,
                _keep=_keep,
                contexts=worker.chip_contexts,
            )

        _has_prefill_tensor = torch.tensor(True, dtype=torch.bool).share_memory_()

        def _build_full_tensors():
            """Build the tensor dict for the single l3_generate dispatch.

            host_orch now owns the full generation loop (prefill step 0 +
            pl.unroll(max_new_tokens) decode steps), so this is called once.
            has_prefill is always True: step 0 inside host_orch runs prefill_all
            then the first decode; subsequent iterations run decode-only.
            """
            td = {}
            for info, val in zip(lg_param_infos, [
                prefill_hidden,           # prefill_hidden
                prefill_seq_lens,         # prefill_seq_lens
                prefill_slot_mapping,     # prefill_slot_mapping
                decode_hidden_buf,        # decode_hidden
                decode_seq_lens,          # decode_seq_lens (mutated by sample_and_prepare sub-worker)
                decode_slot_mapping_buf,  # decode_slot_mapping
                sm_sw["input_rms_weight"],
                sm_sw["wq"],
                sm_sw["wk"],
                sm_sw["wv"],
                sm_sw["q_norm_weight"],
                sm_sw["k_norm_weight"],
                rope_cos,
                rope_sin,
                block_table,
                k_cache_all,
                v_cache_all,
                sm_sw["wo"],
                sm_sw["post_rms_weight"],
                sm_sw["w_gate"],
                sm_sw["w_up"],
                sm_sw["w_down"],
                _has_prefill_tensor,      # has_prefill = True
                prefill_out,              # prefill_out
                decode_out,               # decode_out
                rms_x,                    # rms_x  (shares storage with decode_out)
                rms_gamma,                # final_norm_weight
                rms_normed,               # rms_normed
                lm_head_weight,           # lm_head_weight_t
                logits_padded,            # logits_padded
            ], strict=True):
                if not val.is_shared():
                    val = val.share_memory_()
                td[info.name] = val
            return td

        # KV cache device pointers collected inside generate_orch_fn so the
        # post-run sync-back step can copy updated K/V values back to host.
        # Format: list of (host_ptr: int, dev_ptr: int, nbytes: int).
        _kv_dev_ptrs: list[tuple[int, int, int]] = []

        def generate_orch_fn(orch, _args, _cfg):
            _keep: list = []
            call_config = CallConfig()
            call_config.block_dim = lg_dc.block_dim
            call_config.aicpu_thread_num = lg_dc.aicpu_thread_num

            _t_pf = time.perf_counter()
            _t_anchors[0] = _t_pf
            _t_anchors[1] = _t_pf
            _sample_done_ts[0] = _t_pf  # baseline for step 0 chip_tasks measurement
            if _l3_trace_enabled:
                print(f"[L3-step] host_orch submit_start {_fmt_rel(_t_pf)}", flush=True)

            # Single dispatch: host_orch drives prefill + all decode steps
            # (pl.unroll(max_new_tokens) inside host_orch).
            td = _build_full_tensors()

            from simpler.task_interface import ContinuousTensor as _CT  # noqa: PLC0415
            from simpler_setup.torch_interop import (  # noqa: PLC0415
                torch_dtype_to_datatype as _td2dt,
            )

            # Pre-upload static weight tensors once per generate call.
            # child_memory=True → runtime skips H2D + D2H on every dispatch.
            # ~3 400 ms of init_runtime per step reduced to ~11 ms.
            for _pname, _t in list(td.items()):
                if not isinstance(_t, torch.Tensor):
                    continue
                if not any(_sub in _pname for _sub in _sw_substrings):
                    continue
                _nbytes = int(_t.nbytes)
                _dev_ptr = orch.malloc(worker_id=0, size=_nbytes)
                orch.copy_to(worker_id=0, dst=_dev_ptr, src=_t.data_ptr(), size=_nbytes)
                _shapes = tuple(int(s) for s in _t.shape)
                _dt = _td2dt(_t.dtype)
                td[_pname] = _CT.make(_dev_ptr, _shapes, _dt, child_memory=True)

            # Pre-upload KV cache (k_cache_all / v_cache_all) once per
            # generate call.  The kernel writes updated K/V values in-place on
            # device; child_memory=True skips H2D and D2H on every decode step,
            # saving ~280 ms H2D + ~360 ms D2H per step (~640 ms × 16 steps).
            # After worker.run() drains (all tasks done), a second worker.run()
            # call copies the final device state back to the host KV cache via
            # orch.copy_from so subsequent generate calls see updated values.
            for _pname, _t in list(td.items()):
                if not isinstance(_t, torch.Tensor):
                    continue
                if "k_cache_all" not in _pname and "v_cache_all" not in _pname:
                    continue
                _nbytes = int(_t.nbytes)
                _dev_ptr = orch.malloc(worker_id=0, size=_nbytes)
                orch.copy_to(worker_id=0, dst=_dev_ptr, src=_t.data_ptr(), size=_nbytes)
                _shapes = tuple(int(s) for s in _t.shape)
                _dt = _td2dt(_t.dtype)
                _kv_dev_ptrs.append((_t.data_ptr(), _dev_ptr, _nbytes))
                td[_pname] = _CT.make(_dev_ptr, _shapes, _dt, child_memory=True)

            _submit_l3_generate(orch, call_config, td, _keep)

            if _l3_trace_enabled:
                print(f"[L3-step] all submits done {_fmt_rel(time.perf_counter())}", flush=True)

        # ── Create Worker and execute ──

        lg_sub_fns = dict(compiled.l3_generate_sub_worker_fns or {})
        # Override the placeholder sample_and_prepare sub-worker emitted by the
        # l3_generate compiler with the real closure that reads shared-memory
        # tensors and performs argmax → embedding lookup → slot-map update.
        lg_sub_fns["sample_and_prepare"] = sample_and_prepare_fn

        num_sub = max(lg_dc.num_sub_workers, len(lg_sub_fns))

        _mark("setup_closures_and_buffers")

        worker = Worker(
            level=3,
            device_ids=[self._device_id],
            num_sub_workers=num_sub,
            platform=compiled.l3_generate_platform,
            runtime=compiled.l3_generate_runtime_name,
        )

        _mark("Worker_ctor")

        # Register l3_generate sub-worker callables (including sample_and_prepare).
        sub_ids: dict[str, int] = {}
        for name, fn in lg_sub_fns.items():
            sub_ids[name] = worker.register(fn)

        _mark("worker_register")

        worker.init()
        _mark("worker_init")
        try:
            _t_run_start = time.perf_counter()
            worker.run(generate_orch_fn)
            _mark("worker_run_generate")

            # Sync KV cache back to host.  worker.run() above calls _drain()
            # internally, so all chip tasks (including the last decode step)
            # have completed by the time we reach here.  The child_memory
            # buffers are still live (worker.close() not yet called), so a
            # second worker.run() can copy them back to the host tensors.
            if _kv_dev_ptrs:
                def _kv_sync_orch_fn(orch, _args, _cfg):  # noqa: E306
                    for _host_ptr, _dev_ptr, _nbytes in _kv_dev_ptrs:
                        orch.copy_from(
                            worker_id=0, dst=_host_ptr, src=_dev_ptr, size=_nbytes,
                        )
                worker.run(_kv_sync_orch_fn)
            _mark("worker_run_kv_sync")

            _t_run_end = time.perf_counter()
            print(
                f"[L3-timer] worker.run total wall-clock: "
                f"{(_t_run_end-_t_run_start)*1000:.1f}ms",
                flush=True,
            )
        finally:
            worker.close()
            _mark("worker_close")

        # Update KV allocations.
        final_token_count = int(token_count[0].item())
        for batch_idx, alloc in enumerate(prefill_batch.kv_allocations):
            base_seq = int(prefill_inputs.seq_lens[batch_idx].item())
            alloc.tokens_used = max(alloc.tokens_used, base_seq + final_token_count)

        ids = generated_ids[:final_token_count].tolist()
        ret_val = ids, decode_out[:actual_batch].float()
        _mark("post_process")

        timer.report()
        return ret_val

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

        Currently a no-op for both L3 and baseline modes — the engine calls
        executor.session() unconditionally so that the generate lifecycle is
        ready for a future persistent-Worker upgrade.
        """
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
