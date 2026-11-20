# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep native vLLM prefill and run Lib decode through Simpler in both modes."""

import hashlib
import importlib.util
import os
import re
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from prepare_runtime import prepare_artifact
from qwen_model import (
    BATCH, HEAD_DIM, HEADS, LAYERS, MAX_SEQ_LEN, PAGE,
    allocate_layer_caches, pack_model_weights, rope_tables, validate_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

from simpler.buffer import Buffer
from simpler.task_interface import CallConfig, TaskArgs, Tensor, TensorArgType
from simpler.worker import Worker
from simpler_setup.torch_interop import torch_dtype_to_datatype

TABLE_STRIDE = MAX_SEQ_LEN // PAGE
DTYPES = {"fp32": torch.float32, "bfloat16": torch.bfloat16, "int32": torch.int32}


class KernelDecode:
    def __init__(self, artifact, bridge, weights, caches, device):
        self.device = device
        self.page_count = caches[0].shape[1]
        self.pending = None
        self.failed = False
        self.closed = False
        self.audited_steps = 0
        self.audited_address_ranges = 0
        self.caches = caches
        runtime, callable_, params = prepare_artifact(artifact)
        extension = importlib.util.spec_from_file_location("_simpler_framework_queue", bridge)
        self.bridge = importlib.util.module_from_spec(extension)
        extension.loader.exec_module(self.bridge)

        self.values = {**weights, "k_cache": caches[0].view(-1, HEAD_DIM), "v_cache": caches[1].view(-1, HEAD_DIM)}
        if any(
            not cache.is_contiguous() or cache.data_ptr() != self.values[name].data_ptr()
            for cache, name in zip(caches, ("k_cache", "v_cache"), strict=True)
        ):
            raise ValueError("Lib decode must bind the native prefill KV backing")
        self.args = TaskArgs()
        self.buffers = []
        self.config = CallConfig()
        self.worker = Worker(level=2, device_id=device.index, platform="a2a3", runtime=runtime)
        access = {"In": TensorArgType.INPUT, "InOut": TensorArgType.INOUT, "Out": TensorArgType.OUTPUT_EXISTING}
        try:
            self.callable = self.worker.register(callable_)
            self.worker.init(borrowed_context_generation=1)
            for param in params:
                name = param["name"]
                rows = BATCH * TABLE_STRIDE if name == "block_table" else BATCH
                shape = tuple(n if n >= 0 else rows for n in param["shape"])
                dtype = DTYPES[param["dtype"]]
                if name not in self.values:
                    self.values[name] = torch.zeros(shape, dtype=dtype, device=device)
                value = self.values[name]
                if value.dtype != dtype or not value.is_contiguous() or len(value.shape) != len(shape) or any(
                    actual != expected for actual, expected in zip(value.shape, param["shape"]) if expected >= 0
                ):
                    raise ValueError(f"tensor differs from generated decode signature: {name}")
                buffer = Buffer.wrap(
                    address=value.data_ptr(), nbytes=value.numel() * value.element_size(),
                    location=self.worker.device_location,
                )
                self.buffers.append(buffer)
                tensor = Tensor(buffer, shapes=value.shape, dtype=torch_dtype_to_datatype(value.dtype))
                self.args.add_tensor(tensor, access[param["direction"]])
        except BaseException:
            self.worker.close()
            raise

    def retire(self):
        if self.pending is not None:
            call, ticket, captured = self.pending
            call.check()
            if captured:
                ticket._native_submission._release_model_resources()
            ticket.wait(timeout=120)
            self.pending = None

    def __call__(self, input_ids, metadata):
        self.retire()
        table = self.stage(input_ids, metadata)
        self.check_kv(table)
        stream = torch.npu.current_stream(self.device)
        return self.submit(stream, stream.npu_stream)

    def stage(self, input_ids, metadata):
        if input_ids.shape != (BATCH,) or metadata.block_tables.shape[0] != BATCH:
            raise ValueError("expected a complete 16-request decode batch")
        self.values["sampled_ids_in"][:, 0].copy_(input_ids)
        self.values["seq_lens"].copy_(metadata.seq_lens[:BATCH])
        self.values["slot_mapping"].copy_(metadata.slot_mapping[:BATCH])
        table = self.values["block_table"].view(BATCH, -1)
        table.fill_(-1)
        table[:, :metadata.block_tables.shape[1]].copy_(metadata.block_tables)
        return table

    def submit(self, stream, raw_stream, *, captured=False):
        ticket = self.worker.prepare_submission(
            self.callable, self.args, self.config, caller_stream=raw_stream,
            keepalive=(stream, *self.values.values(), *self.buffers),
        )
        try:
            if captured:
                ticket._native_submission._retain_model_resources()
            call = self.bridge.enqueue_native(ticket.native_handle, list(self.values.values()))
        except BaseException:
            ticket.cancel()
            raise
        self.pending = call, ticket, captured
        return self.values["out"]

    def check_kv(self, table):
        pages_in_use = set()
        lengths = self.values["seq_lens"].tolist()
        slots = self.values["slot_mapping"].tolist()
        for row, length, slot in zip(table.tolist(), lengths, slots, strict=True):
            page, offset = divmod(length - 1, PAGE)
            pages = row[:page + 1]
            if (length < 1 or len(pages) != page + 1 or any(not 0 <= p < self.page_count for p in pages)
                    or len(set(pages)) != len(pages) or pages_in_use.intersection(pages)
                    or slot != pages[-1] * PAGE + offset):
                raise RuntimeError("invalid or overlapping KV pages in the decode batch")
            pages_in_use.update(pages)

        ranges = []
        for cache in self.caches:
            base = cache.data_ptr()
            layer_bytes = cache.stride(0) * cache.element_size()
            page_bytes = cache.stride(1) * cache.element_size()
            for layer in range(cache.shape[0]):
                for page in pages_in_use:
                    start = base + layer * layer_bytes + page * page_bytes
                    ranges.append((start, start + page_bytes))
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise RuntimeError("overlapping K/V device address ranges in the decode batch")
        self.audited_steps += 1
        self.audited_address_ranges += len(ranges)

    def close(self):
        if self.closed:
            return
        try:
            self.retire()
        finally:
            self.worker.close()
            for buffer in self.buffers:
                buffer.close()
            self.closed = True


class KernelModel(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.simpler = None
        self.simpler_caches = None
        self.simpler_logits = False
        self.simpler_steps = 0

    def initialize_packed_weights(self, device):
        self.simpler_weights = pack_model_weights(self)
        cos, sin = rope_tables(Path(self.vllm_config.model_config.model))
        self.simpler_weights.update(rope_cos=cos.to(device), rope_sin=sin.to(device))
        torch.npu.synchronize(device)
        torch.npu.empty_cache()

    def prepare_kernel(self, device):
        config = self.vllm_config.additional_config["simpler"]
        self.simpler = KernelDecode(
            Path(config["artifact"]), Path(config["bridge"]), self.simpler_weights, self.simpler_caches, device,
        )

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        metadata = get_forward_context().attn_metadata
        entry = next(iter(metadata.values())) if metadata else None
        self.simpler_logits = False
        if entry is None or entry.num_prefills:
            if entry is not None and entry.num_decodes:
                raise ValueError("expected separate prefill and decode batches")
            return Qwen3ForCausalLM.forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        if self.simpler is None:
            self.prepare_kernel(input_ids.device)
        logits = self.simpler(input_ids, entry)
        self.simpler_steps += 1
        self.simpler_logits = True
        return logits

    def compute_logits(self, hidden_states):
        if self.simpler_logits:
            self.simpler_logits = False
            return hidden_states[:, :self.config.vocab_size].contiguous()
        return super().compute_logits(hidden_states)


class KernelRunner(NPUModelRunner):
    model_type = KernelModel

    def load_model(self):
        ModelRegistry.register_model("Qwen3ForCausalLM", self.model_type)
        NPUModelRunner.load_model(self)
        self.model.initialize_packed_weights(self.device)

    def _model_forward(self, *args, **kwargs):
        if self.input_batch.num_reqs and not self.input_batch.sampling_metadata.all_greedy:
            raise ValueError("Kernel smoke requires greedy sampling")
        return super()._model_forward(*args, **kwargs)

    def _allocate_kv_cache_tensors(self, kv_cache_config):
        def layer_index(name):
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match is None:
                raise ValueError("KV layer name does not identify a Qwen layer")
            return int(match.group(1))

        specs = self._get_layer_kv_cache_specs(kv_cache_config)
        names = sorted(specs, key=layer_index)
        if [layer_index(name) for name in names] != list(range(LAYERS)):
            raise ValueError("Kernel smoke requires one attention cache per Qwen layer")
        for spec in specs.values():
            if (spec.block_size, spec.num_kv_heads, spec.head_size, spec.dtype) != (
                PAGE, HEADS, HEAD_DIM, torch.bfloat16,
            ):
                raise ValueError("KV specification differs from the BSND decode ABI")
        tensors = kv_cache_config.kv_cache_tensors
        if len(tensors) != LAYERS or any(len(item.shared_by) != 1 for item in tensors):
            raise ValueError("Kernel smoke requires independent per-layer KV allocations")
        if len({item.size for item in tensors}) != 1:
            raise ValueError("KV layers must have equal capacity")
        page_bytes = 2 * PAGE * HEADS * HEAD_DIM * 2
        if tensors[0].size % page_bytes:
            raise ValueError("KV allocation is not page aligned")
        caches = allocate_layer_caches(pages=tensors[0].size // page_bytes, device=self.device)
        self.model.simpler_caches = caches
        return {
            name: tuple(cache[index].view(torch.int8).flatten() for cache in caches) for index, name in enumerate(names)
        }


class KernelWorker(NPUWorker):
    runner_type = KernelRunner
    capture = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_config(self.vllm_config, capture=self.capture)
        self.simpler_failed = False

    def init_device(self):
        self.local_rank = int(os.environ["TASK_DEVICE"])
        self.device = self._init_device()
        init_workspace_manager(self.device, 1)
        self.model_runner = self.runner_type(self.vllm_config, self.device)

    def _kernel_model(self):
        model = self.model_runner.model
        return model.unwrap() if isinstance(model, ACLGraphWrapper) else model

    @contextmanager
    def _execution(self):
        if self.simpler_failed:
            raise RuntimeError("failed Kernel worker cannot admit another invocation")
        try:
            yield
        except BaseException:
            self.simpler_failed = True
            model = self._kernel_model()
            if model is not None and model.simpler is not None:
                model.simpler.failed = True
            raise

    def execute_model(self, *args, **kwargs):
        with self._execution():
            return super().execute_model(*args, **kwargs)

    def sample_tokens(self, *args, **kwargs):
        with self._execution():
            return super().sample_tokens(*args, **kwargs)

    def close_simpler(self):
        model = self._kernel_model()
        if model is not None and model.simpler is not None:
            model.simpler.close()

    def shutdown(self):
        self.close_simpler()
        super().shutdown()

    def simpler_stats(self):
        model = self._kernel_model()
        backend = model.simpler
        helpers = Path(__file__).with_name("qwen_model.py")
        return {
            "steps": model.simpler_steps,
            "failed": self.simpler_failed or (backend is not None and backend.failed),
            "audited_steps": backend.audited_steps,
            "audited_address_ranges": backend.audited_address_ranges,
            "device_id": self.device.index,
            "model_sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), helpers)},
        }


class CaptureModel(KernelModel):
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if get_forward_context().capturing:
            if self.simpler.pending is not None:
                raise RuntimeError("expected one Kernel submission for the B16 graph")
            return self.simpler.submit(self.capture_stream, self.raw_capture_stream, captured=True)
        return Qwen3ForCausalLM.forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        if self.simpler is not None and hidden_states.shape[-1] == self.simpler.values["out"].shape[-1]:
            return hidden_states[:, :self.config.vocab_size].contiguous()
        return Qwen3ForCausalLM.compute_logits(self, hidden_states)


class CaptureRunner(KernelRunner):
    model_type = CaptureModel
    replays = 0
    record_only_verified = False

    def capture_model(self):
        wrapper = self.model
        if not isinstance(wrapper, ACLGraphWrapper):
            raise RuntimeError("expected vLLM-Ascend's ACLGraphWrapper")
        model = wrapper.unwrap()
        model.simpler_caches = wrapper.simpler_caches
        model.prepare_kernel(self.device)
        if torch.npu.graph.default_capture_stream is None:
            torch.npu.graph.default_capture_stream = torch.npu.Stream(device=self.device)
        model.capture_stream = torch.npu.graph.default_capture_stream
        model.raw_capture_stream = model.capture_stream.npu_stream
        model.simpler.values["out"].fill_(-999)
        consumed = super().capture_model()
        torch.npu.synchronize(self.device)
        if model.simpler.pending is None or len(wrapper.concrete_aclgraph_entries) != 1:
            raise RuntimeError("expected exactly one recorded B16 decode graph")
        call, ticket, _ = model.simpler.pending
        call.check()
        if ticket.safe_to_release or not torch.all(model.simpler.values["out"] == -999).item():
            raise RuntimeError("recording ran the decode or released graph-owned resources")
        self.record_only_verified = True
        return consumed

    def _model_forward(self, num_tokens_padded, input_ids=None, *args, **kwargs):
        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return super()._model_forward(num_tokens_padded, input_ids, *args, **kwargs)
        model = self.model.unwrap()
        metadata = next(iter(context.attn_metadata.values()))
        table = model.simpler.stage(input_ids, metadata)
        if not self.input_batch.num_reqs:
            return super()._model_forward(num_tokens_padded, input_ids, *args, **kwargs)
        if not self.record_only_verified or metadata.num_prefills or num_tokens_padded != BATCH:
            raise RuntimeError("expected a recorded graph for a full B16 decode step")
        model.simpler.check_kv(table)
        entry = self.model.concrete_aclgraph_entries[context.batch_descriptor]
        with patch.object(entry.aclgraph, "replay", wraps=entry.aclgraph.replay) as replay:
            result = super()._model_forward(num_tokens_padded, input_ids, *args, **kwargs)
        if replay.call_count != 1:
            raise RuntimeError("vLLM did not replay the recorded decode graph")
        self.replays += 1
        model.simpler_steps += 1
        return result

    def _update_full_graph_params_if_needed(self, forward_context, num_tokens_padded, positions):
        # The Lib graph reads its own staged metadata and has no native attention graph parameters.
        if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            super()._update_full_graph_params_if_needed(forward_context, num_tokens_padded, positions)


class CaptureWorker(KernelWorker):
    runner_type = CaptureRunner
    capture = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            raise ValueError("expected vLLM full decode graphs")

    def simpler_stats(self):
        return {
            **super().simpler_stats(),
            "captures": len(self.model_runner.model.concrete_aclgraph_entries),
            "replays": self.model_runner.replays,
            "record_only_verified": self.model_runner.record_only_verified,
        }

    def close_simpler(self):
        wrapper = self.model_runner.model
        if not isinstance(wrapper, ACLGraphWrapper):
            return super().close_simpler()
        backend = wrapper.unwrap().simpler
        if backend is None or backend.closed:
            return None
        torch.npu.synchronize(self.device)
        for entry in wrapper.concrete_aclgraph_entries.values():
            if entry.aclgraph is not None:
                entry.aclgraph.reset()
        wrapper.concrete_aclgraph_entries.clear()
        super().close_simpler()
        if torch.ones(1, device=self.device).add_(1).item() != 2:
            raise RuntimeError("Torch device is unusable after graph teardown")
        return {"graph_reset": True, "torch_usable": True}
