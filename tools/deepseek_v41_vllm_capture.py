# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture one physical DeepSeek V4.1 Decode step in an isolated vLLM process.

The capture is reference-only.  It uses the pinned vLLM-Ascend runner to load
the real Aurora W8A8 checkpoint, executes one short prompt followed by one
decode token, and records the runner-owned cache before and after the decode
step.  The product PyPTO package is never imported by this module.

The full checkpoint has forty layers and cannot be loaded on the two-card EP2
gate.  The temporary model root therefore changes only ``num_hidden_layers``
to ``layer_id + 1`` and leaves every weight shard and quantization description
as symlinks to the supplied checkpoint.  Each invocation captures one
physical layer independently; it never uses a product or a previous PyPTO
layer result as its input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import numpy as np
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any


SESSION = "deepseek-v4.1-flash-a3-decode"
PINNED_VLLM_ASCEND = "23d1d9055f2161b3c15613c7fc309f7d5d2ef629"
EXPECTED_SOC = "ascend910_9382"
TOPOLOGY = {
    "dp": 2,
    "tp": 1,
    "ep": 2,
    "pp": 1,
    "pcp": 1,
    "dcp": 1,
    "enable_expert_parallel": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_identity(root: Path) -> dict[str, Any]:
    import torch_npu
    import vllm
    import vllm_ascend
    import vllm_ascend.vllm_ascend_C

    native = {}
    for relative in (
        "source/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so",
        "source/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so",
        "env/lib/python3.12/site-packages/torch_npu/_C.cpython-312-aarch64-linux-gnu.so",
        "env/lib/python3.12/site-packages/torch_npu/lib/libtorch_npu.so",
    ):
        path = root / relative
        native[relative] = _sha256(path) if path.is_file() else "missing"
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "module_origins": {
            "vllm": str(vllm.__file__),
            "vllm_ascend": str(vllm_ascend.__file__),
            "vllm_ascend_C": str(vllm_ascend.vllm_ascend_C.__file__),
            "torch_npu": str(torch_npu.__file__),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "vllm-ascend", "torch-npu")
        },
        "native_sha256": native,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "soc_version": os.environ.get("SOC_VERSION", "not_exported"),
        "task_device": os.environ.get("TASK_DEVICE", "not_allocated"),
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "not_exported"),
    }


def _prepare_reference_ascend_home(output_dir: Path) -> dict[str, str]:
    """Give Triton a private CANN metadata view without changing shared CANN."""

    original = Path(os.environ.get("ASCEND_HOME_PATH", ""))
    arch = "aarch64-linux"
    source_arch = original / arch
    if not source_arch.is_dir():
        raise RuntimeError(f"ASCEND_HOME_PATH does not contain {arch}: {original}")

    private_root = output_dir / "reference_ascend_home"
    private_arch = private_root / arch
    private_arch.mkdir(parents=True, exist_ok=True)
    for child in original.iterdir():
        if child.name == arch:
            continue
        target = private_root / child.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(child)
        except FileExistsError:
            continue
    for child in source_arch.iterdir():
        target = private_arch / child.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(child)
        except FileExistsError:
            # Both external-launcher ranks prepare the shared view.
            continue

    # Triton uses this file only as a compile-cache identity. The actual CANN
    # binaries remain the symlinked files from the activated reference env.
    metadata = private_arch / "ascend_all_cann_install.info"
    if not metadata.exists():
        metadata.write_text("CANN_VERSION=9.1.0\n", encoding="utf-8")
    os.environ["ASCEND_HOME_PATH"] = str(private_root)
    return {
        "original": str(original),
        "private": str(private_root),
        "metadata": str(metadata),
        "metadata_sha256": _sha256(metadata),
    }


def _make_layer_root(checkpoint: Path, root: Path, layer_id: int) -> Path:
    """Create a small config view while retaining all checkpoint payloads."""

    root.mkdir(parents=True, exist_ok=True)
    layer_only = os.environ.get("VLLM_CAPTURE_LAYER_ONLY") == "1"
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    text["num_hidden_layers"] = layer_id + 1
    # A truncated layer view must advertise only the Engram tables that can
    # occur in its retained physical layers.  vLLM preflights every ID in
    # this list before loading any weights.
    engram_ids = [int(value) for value in text.get("engram_layer_ids", ())]
    engram_sizes = list(text.get("engram_num_embeddings", ()))
    retained_engram = [
        (layer, engram_sizes[index] if index < len(engram_sizes) else None)
        for index, layer in enumerate(engram_ids)
        if layer <= layer_id
    ]
    if layer_only:
        # The layer-only reference constructs just the requested physical
        # layer.  Keep its Engram binding, but do not make placeholder layers
        # own the earlier tables that are irrelevant to this isolated call.
        retained_engram = [item for item in retained_engram if item[0] == layer_id]
    text["engram_layer_ids"] = [layer for layer, _ in retained_engram]
    if engram_sizes:
        text["engram_num_embeddings"] = [size for _, size in retained_engram]
    # Physical L0 has no Engram table.  vLLM constructs all configured
    # Engram bindings even for a truncated model, so remove only the runtime
    # binding for this L0 reference view; the source checkpoint remains intact.
    if layer_id == 0 and 1 in text.get("engram_layer_ids", []):
        text["engram_layer_ids"] = []
        text["engram_num_embeddings"] = []
    # Do not make a layer-only config look like a different model revision.
    config["text_config"] = text
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in (
        "quant_model_description.json",
        "aurora_export_manifest.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        source = checkpoint / name
        if source.is_file():
            target = root / name
            if not target.exists():
                target.symlink_to(source)
    source_index = checkpoint / "quant_model_weights.safetensors.index.json"
    target_index = root / source_index.name
    if target_index.is_symlink() or target_index.exists():
        target_index.unlink()
    index = json.loads(source_index.read_text(encoding="utf-8"))
    filtered_map = {}
    for name, shard in index["weight_map"].items():
        match = re.match(r"layers\.(\d+)\.", name)
        if match is not None and int(match.group(1)) > layer_id:
            continue
        if layer_only and match is not None and int(match.group(1)) != layer_id:
            continue
        if name.startswith(("mtp.", "vision.", "aligner.", "image_")):
            continue
        if ".engram." in name and not layer_only:
            continue
        filtered_map[name] = shard
    index["weight_map"] = filtered_map
    target_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    needed_shards = set(filtered_map.values())
    # Engram embeddings are separate multi-gigabyte safetensors files rather
    # than ordinary quantized shards.  The reference loader binds them by
    # filename from the temporary model root, so expose only the tables owned
    # by the truncated layer view.
    for engram_layer in text.get("engram_layer_ids", ()):
        if int(engram_layer) > layer_id:
            continue
        for kind in ("weight", "scale"):
            source = checkpoint / f"engram_embed_{kind}_l{int(engram_layer)}.safetensors"
            if not source.is_file():
                raise FileNotFoundError(f"Engram table is missing for layer {engram_layer}: {source}")
            target = root / source.name
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(source)
    # Tokenizer files may be stored under an auxiliary directory in this export.
    for source in checkpoint.iterdir():
        if source.suffix == ".safetensors":
            continue
        if source.is_file() and source.name != "config.json" and not (root / source.name).exists():
            # The index also references per-layer Engram weight and scale
            # shards, whose names are not part of the quantized shard glob.
            (root / source.name).symlink_to(source)
    # The vLLM loader enumerates every tensor in each safetensors file.  Make
    # filtered local files so tensors from later physical layers cannot reach
    # the one-layer model even when they share a source shard.
    from safetensors import safe_open
    from safetensors.torch import save_file

    names_by_shard: dict[str, list[str]] = {}
    for name, shard in filtered_map.items():
        names_by_shard.setdefault(shard, []).append(name)
    for shard in sorted(needed_shards):
        source = checkpoint / shard
        target = root / shard
        if not source.is_file():
            raise FileNotFoundError(f"filtered index references missing shard {source}")
        # Engram tables are large standalone safetensors files.  Keep their
        # indexed names visible in the layer view without materializing a
        # second copy during every independent reference capture.
        if shard.startswith("engram_embed_"):
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(source)
            continue
        tensors = {}
        with safe_open(str(source), framework="pt", device="cpu") as reader:
            for name in names_by_shard[shard]:
                tensors[name] = reader.get_tensor(name)
        save_file(tensors, str(target), metadata={"format": "pt", "source": str(source)})
    (root / ".ready").write_text("complete\n", encoding="utf-8")
    return root


def _patch_layer_only_model(layer_id: int) -> dict[str, Any]:
    """Construct one physical layer while preserving its checkpoint prefix.

    The normal vLLM constructor allocates every preceding MoE layer before the
    target is reached.  On a two-card EP2 reference that exhausts device memory
    for L14 and later.  This reference-only patch leaves the target layer and
    its physical prefix intact, while placeholder layers carry the hidden
    state through the model loop without allocating weights or cache users.
    """

    import torch
    import vllm_ascend.models.deepseek_v41.model as deepseek_v41_model
    import vllm_ascend.models.deepseek_v41.vl_model as deepseek_v41_vl_model
    from vllm.model_executor.models.utils import PPMissingLayer

    class LayerOnlyIdentity(PPMissingLayer):
        def __init__(self, index: int):
            super().__init__()
            self.layer_idx = index
            self.engram = None

        def forward(self, positions, hidden_states, pre_mix, *args, **kwargs):
            return hidden_states, pre_mix

        @staticmethod
        def hc_collapse(hidden_states, pre_mix):
            return (pre_mix.unsqueeze(-1) * hidden_states.float()).sum(-2).to(hidden_states.dtype)

    class LayerOnlyCacheOwner(deepseek_v41_model.DeepseekV41DecoderLayer):
        """Register cache planes for a source layer without its MoE weights."""

        def __init__(
            self,
            vllm_config,
            prefix,
            config=None,
            topk_indices_buffer=None,
            is_draft_layer=False,
        ):
            torch.nn.Module.__init__(self)
            if config is None:
                config = deepseek_v41_model.normalize_deepseek_v41_config(
                    vllm_config.model_config.hf_config
                )
            layer_index = int(prefix.rsplit(".", 1)[-1])
            topology = deepseek_v41_model.build_layer_plan(config)
            role = topology.layer(layer_index)
            max_position_embeddings = config.rope_parameters["original_max_position_embeddings"]
            self.layer_idx = layer_index
            self.hidden_size = config.hidden_size
            self.norm_eps = config.rms_norm_eps
            self.hc_mult = config.hc_mult
            self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
            self.hc_eps = config.hc_eps
            self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe
            self.use_sequence_parallel = self.use_sequence_parallel_moe
            self.self_attn = deepseek_v41_model.DeepseekV41Attention(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                reduce_results=not self.use_sequence_parallel_moe,
                need_gather_q_kv=self.use_sequence_parallel_moe
                and deepseek_v41_model.enable_dsa_cp(),
            )
            # Model construction and metadata setup inspect these attributes,
            # while the identity forward never reads them.
            self.input_layernorm = PPMissingLayer()
            self.post_attention_layernorm = PPMissingLayer()
            self.mlp = PPMissingLayer()
            self.engram = None

        def forward(self, positions, hidden_states, pre_mix, *args, **kwargs):
            return hidden_states, pre_mix

    topology = None

    def make_layers(num_hidden_layers, layer_fn, prefix):
        if layer_id < 0 or layer_id >= num_hidden_layers:
            raise ValueError(f"target layer {layer_id} is outside {num_hidden_layers} layers")
        # Consumer layers need cache owners from their source layer.  The
        # source is kept as a lightweight attention/cache module; all other
        # prefix layers remain parameter-free identities.
        source_layers = set()
        config = None
        for cell in getattr(layer_fn, "__closure__", ()) or ():
            value = cell.cell_contents
            if hasattr(value, "cache_config") and hasattr(value, "model_config"):
                config = value.model_config.hf_config
                break
        if config is not None:
            normalized = deepseek_v41_model.normalize_deepseek_v41_config(config)
            role = deepseek_v41_model.build_layer_plan(normalized).layer(layer_id)
            if role.kv_source_layer is not None:
                source_layers.add(int(role.kv_source_layer))
            if role.index_source_layer is not None:
                source_layers.add(int(role.index_source_layer))
        modules = [
            layer_fn(prefix=f"{prefix}.{index}")
            if index == layer_id
            else (
                LayerOnlyCacheOwner(
                    next(
                        cell.cell_contents
                        for cell in (getattr(layer_fn, "__closure__", ()) or ())
                        if hasattr(cell.cell_contents, "cache_config")
                        and hasattr(cell.cell_contents, "model_config")
                    ),
                    f"{prefix}.{index}",
                )
                if index in source_layers
                else LayerOnlyIdentity(index)
            )
            for index in range(num_hidden_layers)
        ]
        return 0, num_hidden_layers, torch.nn.ModuleList(modules)

    deepseek_v41_model.make_layers = make_layers

    # The VL wrapper used by this checkpoint does not expose the language
    # model's Engram cache prefix, while the V1 Ascend runner queries it from
    # the outer model during memory profiling.  Resolve the target layer
    # directly so identity placeholders never become cache owners.
    deepseek_v41_vl_model.AscendDeepseekV41ForCausalLM.engram_cache_layer_name = property(
        lambda self: self.language_model.model.layers[layer_id]
        .self_attn.dsa_attn.swa_cache_layer.prefix
    )
    return {
        "enabled": True,
        "target_layer": layer_id,
        "prefix_preserved": True,
        "placeholder_layers": "identity_without_parameters_with_source_cache_owners",
        "product_process": False,
    }


def _patch_empty_state_cache_group() -> dict[str, Any]:
    """Keep reduced reference views valid when no Engram state is present.

    The pinned Ascend cache planner unconditionally constructs a second
    ``UniformTypeKVCacheSpecs`` group for ``CircularBufferSpec`` entries.  A
    layer-only view whose target has no Engram state has no such entries, so
    the planner raises ``StopIteration`` while profiling the cache budget.
    The product never imports this module; this patch is scoped to the
    independent reference process and preserves all non-empty groups.
    """

    import vllm_ascend.models.deepseek_v41.cache_config as cache_config
    import vllm_ascend.patch.platform.patch_kv_cache_utils as kv_cache_patch
    from dataclasses import replace
    from vllm.v1.kv_cache_interface import CircularBufferSpec, UniformTypeKVCacheSpecs

    original = cache_config.group_cache_specs

    def group_cache_specs_without_empty_state(specs):
        if specs and all(
            isinstance(spec, cache_config.AscendSlidingWindowMLASpec)
            for spec in specs.values()
        ):
            return [
                UniformTypeKVCacheSpecs.from_specs({name: spec})
                for name, spec in specs.items()
            ]
        page_sizes, layer_tuples = cache_config.get_layer_tuples(specs)
        padded = {}
        for page_size, layer_tuple in zip(page_sizes, layer_tuples):
            kv_name, index_name, *rest = layer_tuple
            folded_name = index_name + "_folded"
            has_folded = folded_name in rest
            aliases = rest[1:] if has_folded else rest
            kv_bytes = specs[kv_name].unpadded_page_size_bytes
            index_bytes = specs[index_name].unpadded_page_size_bytes
            padded[kv_name] = replace(specs[kv_name], page_size_padded=kv_bytes)
            padded[index_name] = replace(
                specs[index_name],
                page_size_padded=index_bytes if has_folded else page_size - kv_bytes,
            )
            if has_folded:
                padded[folded_name] = replace(
                    specs[folded_name],
                    page_size_padded=page_size - kv_bytes - index_bytes,
                )
            padded.update(
                (name, replace(specs[name], page_size_padded=page_size))
                for name in aliases
            )

        mla_names = [
            name for name, spec in padded.items()
            if isinstance(spec, cache_config.AscendMLAAttentionSpec)
        ]
        state_names = [
            name for name, spec in padded.items()
            if isinstance(spec, CircularBufferSpec)
        ]
        groups = []
        if mla_names:
            groups.append(
                UniformTypeKVCacheSpecs.from_specs(
                    {name: padded[name] for name in mla_names}
                )
            )
        if state_names:
            groups.append(
                UniformTypeKVCacheSpecs.from_specs(
                    {name: padded[name] for name in state_names}
                )
            )

        swa_columns = [
            [
                name
                for name in layer_tuple
                if isinstance(padded[name], cache_config.AscendSlidingWindowMLASpec)
                and ".mtp." not in f".{name}"
            ]
            for layer_tuple in layer_tuples
        ]
        for names in zip(*swa_columns):
            groups.append(
                UniformTypeKVCacheSpecs.from_specs(
                    {name: padded[name] for name in names}
                )
            )

        draft_names = [
            name
            for layer_tuple in layer_tuples
            for name in layer_tuple
            if isinstance(padded[name], cache_config.AscendSlidingWindowMLASpec)
            and ".mtp." in f".{name}"
        ]
        if draft_names:
            groups.append(
                UniformTypeKVCacheSpecs.from_specs(
                    {name: padded[name] for name in draft_names}
                )
            )
        return groups

    cache_config.group_cache_specs = group_cache_specs_without_empty_state
    kv_cache_patch.group_cache_specs = group_cache_specs_without_empty_state
    return {
        "enabled": True,
        "reference_only": True,
        "reason": "skip_empty_circular_buffer_group",
        "original": f"{original.__module__}.{original.__name__}",
    }


def _layer_state_contract(checkpoint: Path, layer_id: int) -> dict[str, Any]:
    """Resolve the immutable cache ownership fields for one physical layer."""

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    ratios = [int(value) for value in text.get("compress_ratios", ())]
    kv_sources = [int(value) for value in text.get("kv_source_layer_ids", ())]
    index_sources = [int(value) for value in text.get("index_source_layer_ids", ())]
    if not 0 <= layer_id < len(ratios):
        raise ValueError(f"layer_id {layer_id} is outside the checkpoint layer plan")
    ratio = ratios[layer_id]

    def latest(sources: list[int]) -> int | None:
        matching = [source for source in sources if source <= layer_id and ratios[source] == ratio]
        return matching[-1] if matching else None

    return {
        "layer_ratio": ratio,
        "source_layer": latest(kv_sources),
        "index_source_layer": latest(index_sources),
        "is_kv_source": layer_id in kv_sources,
        "is_index_source": layer_id in index_sources,
        "is_candidate_source": layer_id == int(text.get("candidate_source_layer_id", -1)),
        "engram_required": layer_id in {int(value) for value in text.get("engram_layer_ids", ())},
    }


def _wait_for_layer_root(root: Path, timeout_s: float = 1200.0) -> None:
    """Wait until rank zero has finished publishing the shared model view."""

    deadline = time.monotonic() + timeout_s
    ready = root / ".ready"
    while not ready.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {ready}")
        time.sleep(0.2)


def _tensor_to_list(tensor: Any) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _cache_description(cache: Any) -> dict[str, Any]:
    if isinstance(cache, (tuple, list)):
        return {"kind": "tuple", "items": [_cache_description(item) for item in cache]}
    if not hasattr(cache, "shape"):
        return {"kind": type(cache).__name__}
    return {
        "kind": "tensor",
        "shape": list(cache.shape),
        "dtype": str(cache.dtype),
        "device": str(cache.device),
        "strides": list(cache.stride()),
    }


def _clone_cache(cache: Any, block_ids: list[int] | None = None) -> Any:
    """Copy only request-owned cache pages to host memory.

    A vLLM cache tensor reserves the complete device cache budget.  Cloning
    that tensor before a decode step allocates another device-sized temporary
    and can OOM an otherwise healthy reference process.  The input contract
    only needs the pages addressed by this request, together with the full
    cache layout recorded separately, so select those rows on device first.
    """

    if isinstance(cache, tuple):
        return tuple(_clone_cache(item, block_ids) for item in cache)
    if isinstance(cache, list):
        return [_clone_cache(item, block_ids) for item in cache]
    if block_ids is None:
        return cache.detach().cpu()
    if not hasattr(cache, "shape") or not cache.shape:
        return cache.detach().cpu()
    # Slicing avoids the large temporary that some Ascend index_select
    # implementations allocate even for a one-element index.
    selected = cache[block_ids[0] : block_ids[-1] + 1]
    return selected.detach().cpu()


def _request_cache_blocks(batch: Any, valid_lengths: list[int], block_size: int = 128) -> list[int]:
    """Return only physical pages occupied by the current request(s).

    vLLM exposes a preallocated block table whose unused entries are also
    nonnegative.  The valid prefix is determined by each request's computed
    length and the configured page size.
    """

    blocks: set[int] = set()
    for request_index, group in enumerate(getattr(batch, "block_table", ())):
        # Decode slot mappings identify the physical page directly and avoid
        # interpreting the unused tail of vLLM's preallocated block table.
        slot_mapping = getattr(group, "slot_mapping", None)
        if slot_mapping is not None:
            slots = getattr(slot_mapping, "gpu", slot_mapping)
            slot = int(slots[:1].detach().cpu().reshape(-1)[0].item())
            if slot >= 0:
                blocks.add(slot // block_size)
                continue
        table = (
            group.get_numpy_array()
            if hasattr(group, "get_numpy_array")
            else group.block_table.np
        )
        table = table.reshape(table.shape[0], -1) if table.ndim > 1 else table.reshape(1, -1)
        row_index = request_index if request_index < table.shape[0] else 0
        valid_length = valid_lengths[request_index] if request_index < len(valid_lengths) else 0
        page_count = max(1, (int(valid_length) + block_size - 1) // block_size)
        for value in table[row_index, :page_count].tolist():
            value = int(value)
            if value >= 0:
                blocks.add(value)
    if not blocks:
        raise RuntimeError("decode request did not expose a valid cache block")
    return sorted(blocks)


def _save_cache_npz(path: Path, cache_before: Any, cache_after: Any) -> dict[str, Any]:
    import struct
    import zipfile
    import torch

    arrays: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    def to_numpy(value: Any) -> Any:
        if value.dtype == torch.bfloat16:
            # NumPy has no BF16 dtype. Preserve the exact 16-bit payload and
            # identify it in metadata instead of converting reference state.
            return value.contiguous().view(torch.int16).numpy()
        return value.numpy()

    def record(name: str, value: Any) -> None:
        arrays[name] = to_numpy(value)
        metadata[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "storage_dtype": str(arrays[name].dtype),
        }

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(f"{prefix}_{index}", item)
            return
        record(f"before_{prefix}", value)

    visit("cache", cache_before)

    def visit_after(prefix: str, value: Any) -> None:
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit_after(f"{prefix}_{index}", item)
            return
        record(f"after_{prefix}", value)

    visit_after("cache", cache_after)
    np.savez_compressed(path, **arrays)
    return metadata


def _save_layer_state_npz(
    path: Path,
    cache_before: Any,
    cache_after: Any,
    candidate_before: Any,
    candidate_after: Any,
    layer_state: dict[str, Any],
) -> dict[str, Any]:
    """Save the physical DSA state used by an isolated layer replay.

    The v1 cache list is ordered as the index tuple, long-context compressed
    KV plane, and the SWA alias plane.  Candidate rows are model-level buffers
    and are captured separately from the paged cache.  Only request-owned
    rows are present in ``cache_before``/``cache_after`` because the caller
    already selected the request's block IDs.
    """

    import numpy as np
    import torch

    if not isinstance(cache_before, (tuple, list)) or len(cache_before) < 3:
        raise ValueError("DeepSeek V4.1 layer state requires index, long-KV and SWA cache groups")
    index_before = cache_before[0]
    index_after = cache_after[0]
    if not isinstance(index_before, (tuple, list)) or len(index_before) < 2:
        raise ValueError("index cache group must contain payload and scale planes")

    def storage(value: Any) -> np.ndarray:
        if value.dtype == torch.bfloat16:
            return value.contiguous().view(torch.int16).numpy()
        return value.contiguous().numpy()

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}

    def record(name: str, value: Any) -> None:
        arrays[name] = storage(value)
        metadata[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "storage_dtype": str(arrays[name].dtype),
        }

    record("before_index_cache", index_before[0])
    record("after_index_cache", index_after[0])
    record("before_index_scale", index_before[1])
    record("after_index_scale", index_after[1])
    record("before_compressed_kv_cache", cache_before[1])
    record("after_compressed_kv_cache", cache_after[1])
    record("before_swa_cache", cache_before[2])
    record("after_swa_cache", cache_after[2])
    if candidate_before is None or candidate_after is None:
        raise ValueError("candidate pool state was not exposed by the reference model")
    for prefix, value in (("before", candidate_before), ("after", candidate_after)):
        if not isinstance(value, dict):
            raise ValueError("candidate state must be a mapping")
        for name in ("indices", "lengths", "topk_indices", "topk_lengths"):
            if name not in value or value[name] is None:
                raise ValueError(f"candidate state is missing {name}")
            record(f"{prefix}_candidate_{name}", value[name])
    state_meta = {
        "complete": True,
        "layer_id": int(layer_state.get("source_layer") or -1),
        "compressed_kv_cache": {
            "captured": True,
            "before": "before_compressed_kv_cache",
            "after": "after_compressed_kv_cache",
        },
        "index_cache": {
            "captured": True,
            "before": "before_index_cache",
            "after": "after_index_cache",
        },
        "index_scale": {
            "captured": True,
            "before": "before_index_scale",
            "after": "after_index_scale",
        },
        "candidate_pool": {
            "captured": True,
            "before": [
                "before_candidate_indices",
                "before_candidate_lengths",
            ],
            "after": [
                "after_candidate_indices",
                "after_candidate_lengths",
            ],
        },
        "swa_cache": {
            "captured": True,
            "before": "before_swa_cache",
            "after": "after_swa_cache",
        },
        "file": path.name,
        "arrays": metadata,
    }
    np.savez_compressed(path, **arrays)
    return state_meta


def _patch_reference_mhc_fallback() -> dict[str, str]:
    """Use vLLM's own NPU-eager mHC reference when CANN lacks HcPre symbols.

    The pinned vLLM-Ascend extension calls ``aclnnHcPre``/``aclnnHcPost``.
    The shared CANN 9.1 runtime exports only the newer ``aclnnMhcPre`` pair,
    while vLLM already carries the same mHC equations in its reference torch
    implementation.  Those tensor operations stay on the NPU and are used
    only by this independent reference process.
    """

    import torch
    import torch.nn.functional as functional
    import torch_npu
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_post_torch,
        mhc_pre_delayed_torch,
    )

    def hc_pre(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        pre_mix=None,
        *,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
    ):
        if x.dim() != 3 or x.shape[1] != hc_mult:
            raise ValueError(f"reference HcPre expects [tokens,{hc_mult},hidden], got {tuple(x.shape)}")
        post, comb, layer_input, pre = mhc_pre_delayed_torch(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps,
            hc_eps,
            hc_eps,
            2.0,
            hc_sinkhorn_iters,
            pre_mix=pre_mix,
        )
        return layer_input, post, comb, pre

    def hc_post(x, residual, post, comb):
        return mhc_post_torch(x, residual, post, comb)

    def rms_norm_cast(x, weight, epsilon):
        normalized, _ = torch_npu.npu_rms_norm(x, weight, epsilon)
        return normalized, normalized.float()

    def topk_hash_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
        input_tokens=None,
        hash_indices_table=None,
        is_padding=None,
        bias_vl=None,
        image_sentinel_lo=0,
    ):
        """Reference-only NPU implementation for the missing vLLM native op.

        DeepSeek V4.1 text rows use the checkpoint tid2eid table when present;
        image sentinel rows use dynamic score+bias selection.  The operation
        mutates the preallocated vLLM buffers just like the native op.
        """

        scores = functional.softplus(gating_output.float()).sqrt()
        top_k = topk_weights.shape[-1]
        if hash_indices_table is not None:
            if input_tokens is None:
                raise ValueError("hash routing requires input_tokens")
            token_ids = input_tokens.to(torch.int64)
            safe_ids = token_ids.clamp_min(0)
            selected_ids = hash_indices_table[safe_ids].to(torch.int64)
            if bias_vl is not None and image_sentinel_lo > 0:
                image_mask = (token_ids >= image_sentinel_lo) & (
                    token_ids < image_sentinel_lo + 5
                )
                choice = scores + bias_vl.to(scores.dtype).unsqueeze(0)
                dynamic_ids = torch.topk(
                    choice.float(), k=top_k, dim=-1, sorted=True
                ).indices
                selected_ids = torch.where(
                    image_mask.unsqueeze(-1), dynamic_ids, selected_ids
                )
        else:
            choice = scores
            if e_score_correction_bias is not None:
                choice = choice + e_score_correction_bias.to(choice.dtype)
            selected_ids = torch.topk(
                choice.float(), k=top_k, dim=-1, sorted=True
            ).indices

        selected_weights = scores.gather(1, selected_ids)
        if renormalize:
            selected_weights = selected_weights / selected_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(selected_weights.dtype).tiny)
        if routed_scaling_factor != 1.0:
            selected_weights = selected_weights * routed_scaling_factor

        topk_weights.copy_(selected_weights.to(topk_weights.dtype))
        topk_indices.copy_(selected_ids.to(topk_indices.dtype))
        # The permute/unpermute path expects the stable slot ordinal here.
        slot_ids = torch.arange(
            top_k, device=token_expert_indices.device, dtype=token_expert_indices.dtype
        ).expand_as(token_expert_indices)
        token_expert_indices.copy_(slot_ids)

    # vLLM's upstream router reaches this helper directly when its patched
    # Ascend factory is not selected early enough during model construction.
    import vllm._custom_ops as vllm_custom_ops

    vllm_custom_ops.topk_hash_softplus_sqrt = topk_hash_softplus_sqrt

    torch.ops._C_ascend.npu_hc_pre_v3 = hc_pre
    torch.ops._C_ascend.npu_hc_post = hc_post

    # CANN 9.1's A3 SparseFlashMLA ABI requires cmp_ratio=1 whenever the
    # compressed KV plane is disabled. The pinned vLLM-Ascend path uses 0 for
    # that metadata field on the physical L0 SWA layer; normalize only this
    # reference call boundary while keeping has_cmp_kv=False unchanged.
    for op_name in ("npu_sparse_flash_mla_metadata", "npu_sparse_flash_mla"):
        native_op = getattr(torch.ops._C_ascend, op_name)

        def normalize_cmp_ratio(*args, _native_op=native_op, **kwargs):
            # CANN 9.1's A3 metadata ABI accepts only the C4A compressed
            # ratio when a compressed top-k plane is present.  vLLM's
            # DeepSeek V4.1 config names the physical C2A layer as ratio 2;
            # keep that model contract in the capture while adapting only
            # this reference-native ABI boundary.
            if kwargs.get("has_cmp_kv", False) and kwargs.get("cmp_topk", 0) and kwargs.get("cmp_ratio") == 2:
                kwargs["cmp_ratio"] = 4
            return _native_op(*args, **kwargs)

        setattr(torch.ops._C_ascend, op_name, normalize_cmp_ratio)

    def scatter_nd_update_sk(cache, indices, updates):
        """Reference fallback for the missing stride-aware cache store ABI."""

        indices = indices.to(dtype=torch.int64)
        if indices.ndim == 1:
            indices = indices.unsqueeze(-1)
        valid = torch.all(indices >= 0, dim=-1)
        if not bool(valid.all().item()):
            indices = indices[valid]
            updates = updates[valid]
        if indices.numel() != 0:
            torch_npu.npu_scatter_nd_update_(cache, indices, updates.to(cache.dtype).contiguous())

    torch.ops._C_ascend.npu_scatter_nd_update_sk = scatter_nd_update_sk
    # device_op imports the fused-MoE package; establish that package first
    # so the reference-only monkey patch does not enter its import cycle.
    import vllm_ascend.ops  # noqa: F401

    # The pinned DeepSeek V4.1 path still calls the legacy CANN V2 metadata
    # symbol when the packed-cache branch is disabled.  The A3 CANN image in
    # this environment does not export that symbol, while the same pinned
    # vLLM-Ascend checkout exposes the equivalent packaged dispatcher.  Keep
    # this compatibility mapping inside the reference process only.
    qli_metadata_fallback = "native_custom_transformer"
    if os.environ.get("VLLM_CAPTURE_FORCE_QLI_PACKAGED") == "1":
        from vllm_ascend.ops import packaged_attention

        def qli_v2_metadata(*args, **kwargs):
            positional = list(args)
            if len(positional) >= 5:
                kwargs.setdefault("num_heads_q", positional[0])
                kwargs.setdefault("num_heads_k", positional[1])
                kwargs.setdefault("head_dim", positional[2])
                kwargs.setdefault("topk", positional[3])
                kwargs.setdefault("quant_mode", positional[4])
                positional = positional[5:]
            if positional:
                raise TypeError(f"unexpected QLI metadata positional arguments: {positional}")
            kwargs.pop("quant_mode", None)
            kwargs.pop("device", None)
            return packaged_attention.quant_lightning_indexer_metadata(**kwargs)

        # torch.ops namespaces materialize a lazy OpOverload even when its
        # implementation is absent from libopapi.so. Force this mapping only
        # when a caller explicitly requests the packaged diagnostic path.
        torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata = qli_v2_metadata
        qli_metadata_fallback = "packaged_attention.quant_lightning_indexer_metadata"
    from vllm_ascend.device.device_op import DeviceOperator

    DeviceOperator.rms_norm_cast = staticmethod(rms_norm_cast)

    def apply_partial_rotary_inplace(x, cos, sin, *, start, end, inverse=False):
        """Apply interleaved RoPE to the selected tail without the custom ABI.

        CANN 9.1 in this environment does not export the vLLM-Ascend
        ``InplacePartialRotaryMul`` symbol.  DeepSeek V4.1 passes the rotary
        coefficients as ``[tokens, 1, 1, rotary_dim]`` and the custom op
        rotates only ``x[..., start:end]``.  ``npu_rotary_mul`` has the same
        interleaved equation, so operating on that view preserves the
        destination shape and the caller's in-place contract.
        """

        work = x.unsqueeze(-2) if x.ndim == 2 else x
        work = work.unsqueeze(1) if work.ndim == 3 else work
        selected = work[..., start:end]
        rotated = torch_npu.npu_rotary_mul(
            selected,
            cos,
            -sin if inverse else sin,
            rotary_mode="interleave",
        )
        selected.copy_(rotated)
        return x

    DeviceOperator.apply_partial_rotary_inplace = staticmethod(apply_partial_rotary_inplace)
    return {
        "mode": "vllm_mhc_pre_post_torch_npu_fallback",
        "implementation": "vllm.model_executor.kernels.mhc.torch.mhc_pre_delayed_torch+mhc_post_torch",
        "device": "NPU tensor operations",
        "rms_norm_cast": "torch_npu.npu_rms_norm",
        "topk_softplus_sqrt": "reference NPU softplus/sqrt + topk/hash",
        "sparse_flash_mla_cmp_ratio": "reference ABI normalization 2->4 for compressed top-k on A3 CANN; preserve 0 when compressed KV is disabled",
        "partial_rotary_mul": "torch_npu.npu_rotary_mul on selected partial slice with in-place copy",
        "scatter_nd_update_sk": "torch_npu.npu_scatter_nd_update_ with padding-row filtering",
        "quant_lightning_indexer_v2_metadata": qli_metadata_fallback,
    }


def _patch_runner_capture(
    output_dir: Path,
    rank: int,
    layer_id: int,
    layer_state: dict[str, Any],
    *,
    skip_cache: bool = False,
) -> None:
    """Patch only the reference runner; the patch never enters the product."""

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    import vllm_ascend.models.deepseek_v41.model as deepseek_v41_model
    from vllm_ascend.models.deepseek_v41.model import DeepseekV41DecoderLayer, DeepseekV41MoE

    original_execute = NPUModelRunner.execute_model
    original_forward = NPUModelRunner._model_forward
    # AscendRoutedExperts calls the concrete router's _select_experts method
    # directly.  Hooking the abstract public select_experts method misses the
    # actual route tensors on the pinned vLLM-Ascend path.
    original_router_select = BaseRouter._select_experts
    original_moe_forward = DeepseekV41MoE.forward
    original_layer_forward = DeepseekV41DecoderLayer.forward
    original_muls_add = deepseek_v41_model.muls_add_triton
    state: dict[str, Any] = {"decode": False, "captured": False, "skip_cache": skip_cache}

    def wrapped_layer_forward(self, positions, hidden_states, pre_mix, *args, **kwargs):
        capture_this = (
            state["decode"]
            and int(getattr(self, "layer_idx", -1)) == layer_id
            and not state.get("layer_input_captured")
        )
        if capture_this:
            state["layer_input"] = hidden_states.detach().cpu().clone()
            state["layer_pre_mix"] = pre_mix.detach().cpu().clone()
            shared = getattr(getattr(self, "self_attn", None), "shared_state", None)
            if shared is not None:
                state["candidate_before"] = {
                    "indices": shared.candidates.detach().cpu().clone(),
                    "lengths": shared.candidate_lengths.detach().cpu().clone(),
                    "topk_indices": shared.topk_indices.detach().cpu().clone(),
                    "topk_lengths": shared.topk_lengths.detach().cpu().clone(),
                }
            state["layer_input_captured"] = True
        result = original_layer_forward(self, positions, hidden_states, pre_mix, *args, **kwargs)
        if capture_this and isinstance(result, tuple):
            state["layer_output"] = result[0].detach().cpu().clone()
            state["layer_next_pre"] = result[1].detach().cpu().clone()
            shared = getattr(getattr(self, "self_attn", None), "shared_state", None)
            if shared is not None:
                state["candidate_after"] = {
                    "indices": shared.candidates.detach().cpu().clone(),
                    "lengths": shared.candidate_lengths.detach().cpu().clone(),
                    "topk_indices": shared.topk_indices.detach().cpu().clone(),
                    "topk_lengths": shared.topk_lengths.detach().cpu().clone(),
                }
        return result

    def wrapped_muls_add(x, y, scale):
        # DeepSeekV41MoE combines the routed result and the replicated shared
        # result as ``routed * scale + shared``. Capture both operands at the
        # same NPU boundary as the final MoE output for product-side diagnosis.
        result = original_muls_add(x, y, scale)
        if state["decode"] and state.get("moe_input_captured") and not state.get("components_captured"):
            state["routed_output"] = x.detach().cpu().clone()
            state["shared_output"] = y.detach().cpu().clone()
            state["routed_scale"] = float(scale)
            state["components_captured"] = True
        return result

    def wrapped_router_select(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        result = original_router_select(
            self,
            hidden_states,
            router_logits,
            topk_indices_dtype=topk_indices_dtype,
            input_ids=input_ids,
        )
        if state["decode"] and state.get("moe_input_captured") and not state.get("route_captured"):
            weights, ids = result
            state["route_weights"] = weights.detach().cpu().clone()
            state["route_ids"] = ids.detach().cpu().clone()
            state["router_logits"] = router_logits.detach().cpu().clone()
            state["route_captured"] = True
        return result

    def wrapped_moe_forward(
        self,
        hidden_states,
        input_ids=None,
        hidden_states_fp32=None,
        already_sequence_parallel=False,
    ):
        capture_this = (
            state["decode"]
            and not state.get("moe_input_captured")
            and int(getattr(self, "layer_idx", -1)) == layer_id
        )
        if capture_this:
            state["moe_input"] = hidden_states.detach().cpu().clone()
            if hidden_states_fp32 is None:
                state["moe_router_input"] = hidden_states.float().detach().cpu().clone()
            else:
                state["moe_router_input"] = hidden_states_fp32.detach().cpu().clone()
            state["moe_input_captured"] = True
            # The fused MoE returns ``(shared, routed)`` before the model
            # applies the final routed scaling and addition.  Capture at this
            # boundary instead of relying on the optional Triton ``muls_add``
            # rewrite, which may be bypassed by the eager backend.
            original_experts_forward = self.experts.forward

            def wrapped_experts_forward(*args, **kwargs):
                experts_result = original_experts_forward(*args, **kwargs)
                if (
                    state["decode"]
                    and not state.get("components_captured")
                ):
                    if isinstance(experts_result, tuple) and len(experts_result) == 2:
                        shared_output, routed_output = experts_result
                        if shared_output is not None:
                            state["shared_output"] = shared_output.detach().cpu().clone()
                        else:
                            state["shared_output"] = routed_output.detach().cpu().clone().zero_()
                        state["routed_output"] = routed_output.detach().cpu().clone()
                    else:
                        # With mix placement enabled the fused backend folds
                        # shared and routed experts into one tensor. Preserve
                        # that boundary as routed_output and use a zero shared
                        # operand so the diagnostic remains explicit.
                        state["routed_output"] = experts_result.detach().cpu().clone()
                        state["shared_output"] = state["routed_output"].clone().zero_()
                    state["routed_scale"] = float(getattr(self, "routed_scaling_factor", 1.0))
                    state["components_captured"] = True
                return experts_result

            self.experts.forward = wrapped_experts_forward
        result = original_moe_forward(
            self,
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
            already_sequence_parallel=already_sequence_parallel,
        )
        if capture_this:
            self.experts.forward = original_experts_forward
            state["moe_output"] = result.detach().cpu().clone()
        return result

    def wrapped_forward(self, num_tokens, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs):
        if state["decode"] and not state["captured"]:
            batch = self.input_batch
            state["input_ids"] = input_ids.detach().cpu().tolist() if input_ids is not None else None
            state["positions"] = positions.detach().cpu().tolist()
            state["block_table"] = [
                (
                    group.get_numpy_array()
                    if hasattr(group, "get_numpy_array")
                    else group.block_table.np
                ).tolist()
                for group in batch.block_table
            ]
            state["slot_mapping"] = [
                group.slot_mapping.gpu[:num_tokens].detach().cpu().tolist()
                for group in batch.block_table
            ]
        result = original_forward(self, num_tokens, input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if state["decode"] and not state["captured"]:
            hidden = result[0] if isinstance(result, tuple) else result
            state["hidden"] = hidden.detach().cpu().clone()
        return result

    def wrapped_execute(self, scheduler_output, intermediate_tensors=None):
        counts = getattr(scheduler_output, "num_scheduled_tokens", None)
        total = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
        is_decode = total == 1 and counts is not None
        if is_decode and not state["captured"]:
            state["decode"] = True
            batch = self.input_batch
            state["valid_lengths"] = [
                int(value) for value in batch.num_computed_tokens_cpu[: batch.num_reqs].tolist()
            ]
            if not state["skip_cache"]:
                state["cache_block_ids"] = _request_cache_blocks(batch, state["valid_lengths"])
                state["cache_before"] = _clone_cache(
                    self.kv_caches, state["cache_block_ids"]
                )
                state["cache_layout"] = [_cache_description(cache) for cache in self.kv_caches]
        result = original_execute(self, scheduler_output, intermediate_tensors)
        if is_decode and not state["captured"]:
            if not state["skip_cache"]:
                state["cache_after"] = _clone_cache(
                    self.kv_caches, state["cache_block_ids"]
                )
            state["result_type"] = type(result).__name__
            state["captured"] = True
            state["decode"] = False
            rank_dir = output_dir / f"rank{rank}"
            rank_dir.mkdir(parents=True, exist_ok=True)
            npz_meta = {}
            layer_state_meta = None
            if not state["skip_cache"]:
                npz_meta = _save_cache_npz(
                    rank_dir / "history_cache.npz", state["cache_before"], state["cache_after"]
                )
                layer_state_meta = _save_layer_state_npz(
                    rank_dir / "layer_state_capture.npz",
                    state["cache_before"],
                    state["cache_after"],
                    state.get("candidate_before"),
                    state.get("candidate_after"),
                    layer_state,
                )
            moe_meta = None
            if (
                state.get("moe_input_captured")
                and state.get("route_captured")
                and "moe_output" in state
                and state.get("components_captured")
            ):
                import numpy as np
                import torch

                def bf16_storage(value):
                    return value.contiguous().view(torch.int16).numpy()

                moe_path = rank_dir / "moe_capture.npz"
                np.savez_compressed(
                    moe_path,
                    input_bf16=bf16_storage(state["moe_input"]),
                    router_input_fp32=state["moe_router_input"].float().numpy(),
                    output_bf16=bf16_storage(state["moe_output"]),
                    route_ids=state["route_ids"].numpy(),
                    route_weights=state["route_weights"].float().numpy(),
                    router_logits=state["router_logits"].float().numpy(),
                    routed_output_bf16=bf16_storage(state["routed_output"]),
                    shared_output_bf16=bf16_storage(state["shared_output"]),
                )
                moe_meta = {
                    "file": f"rank{rank}/moe_capture.npz",
                    "input_shape": list(state["moe_input"].shape),
                    "router_input_shape": list(state["moe_router_input"].shape),
                    "output_shape": list(state["moe_output"].shape),
                    "route_ids_shape": list(state["route_ids"].shape),
                    "route_weights_shape": list(state["route_weights"].shape),
                    "router_logits_shape": list(state["router_logits"].shape),
                    "input_dtype": "torch.bfloat16",
                    "router_input_dtype": "torch.float32",
                    "output_dtype": "torch.bfloat16",
                    "route_ids_dtype": "torch.int32",
                    "route_weights_dtype": "torch.float32",
                    "routed_output_shape": list(state["routed_output"].shape),
                    "shared_output_shape": list(state["shared_output"].shape),
                    "routed_scale": state["routed_scale"],
                    "component_dtype": "torch.bfloat16",
                }
            layer_meta = None
            if "layer_input" in state and "layer_output" in state:
                import torch

                def bf16_storage(value):
                    return value.contiguous().view(torch.int16).numpy()

                layer_path = rank_dir / "layer_capture.npz"
                np.savez_compressed(
                    layer_path,
                    input_bf16=bf16_storage(state["layer_input"]),
                    pre_mix_fp32=state["layer_pre_mix"].float().numpy(),
                    output_bf16=bf16_storage(state["layer_output"]),
                    next_pre_fp32=state["layer_next_pre"].float().numpy(),
                )
                layer_meta = {
                    "file": f"rank{rank}/layer_capture.npz",
                    "input_shape": list(state["layer_input"].shape),
                    "pre_mix_shape": list(state["layer_pre_mix"].shape),
                    "output_shape": list(state["layer_output"].shape),
                    "next_pre_shape": list(state["layer_next_pre"].shape),
                    "input_dtype": "torch.bfloat16",
                    "pre_mix_dtype": "torch.float32",
                    "output_dtype": "torch.bfloat16",
                    "next_pre_dtype": "torch.float32",
                }
            payload = {
                "rank": rank,
                "layer_id": layer_id,
                "input_ids": state.get("input_ids"),
                "positions": state.get("positions"),
                "block_table": state.get("block_table"),
                "slot_mapping": state.get("slot_mapping"),
                "valid_lengths": state.get("valid_lengths"),
                "cache_layout": state.get("cache_layout"),
                "cache_block_ids": state.get("cache_block_ids"),
                "cache_arrays": npz_meta,
                "layer_state_capture": layer_state_meta,
                "hidden": _tensor_to_list(state["hidden"]) if "hidden" in state else None,
                "moe_capture": moe_meta,
                "layer_capture": layer_meta,
                "result_type": state.get("result_type"),
            }
            (rank_dir / "capture.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return result

    NPUModelRunner._model_forward = wrapped_forward
    NPUModelRunner.execute_model = wrapped_execute
    BaseRouter._select_experts = wrapped_router_select
    DeepseekV41MoE.forward = wrapped_moe_forward
    DeepseekV41DecoderLayer.forward = wrapped_layer_forward
    deepseek_v41_model.muls_add_triton = wrapped_muls_add


def _assemble_existing(args: argparse.Namespace) -> int:
    """Assemble rank captures after a launcher stopped during vLLM teardown.

    This path is CPU-only and never imports the product package.  It is useful
    when the reference workers have already written their immutable rank cache
    snapshots but the vLLM interpreter is being torn down by torchrun.
    """

    import struct
    import zipfile

    output_dir = args.output.resolve()
    records = []
    for rank in range(2):
        record_path = output_dir / f"rank{rank}" / "capture.json"
        cache_path = output_dir / f"rank{rank}" / "history_cache.npz"
        if not record_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(f"rank {rank} capture is incomplete under {output_dir}")
        records.append(json.loads(record_path.read_text(encoding="utf-8")))

    # The cache arrays are logically tens of GiB each and are stored as
    # highly-compressible ZIP64 members.  Copy the compressed members and
    # rewrite only their ZIP headers, avoiding a multi-minute decompression and
    # recompression pass on CPU.
    merged_path = output_dir / "history_cache.npz"
    temporary_path = output_dir / "history_cache.npz.tmp"
    central_records = []
    with temporary_path.open("wb") as destination:
        for rank in range(2):
            source_path = output_dir / f"rank{rank}" / "history_cache.npz"
            with zipfile.ZipFile(source_path) as source_zip, source_path.open("rb") as source:
                for info in source_zip.infolist():
                    source.seek(info.header_offset)
                    local_header = source.read(30)
                    fields = struct.unpack("<4s5H3L2H", local_header)
                    name_length, extra_length = fields[-2:]
                    data_offset = info.header_offset + 30 + name_length + extra_length
                    source.seek(data_offset)
                    name = f"rank{rank}_{info.filename}".encode("utf-8")
                    local_offset = destination.tell()
                    local_extra = struct.pack(
                        "<HHQQ", 0x0001, 16, info.file_size, info.compress_size
                    )
                    destination.write(
                        struct.pack(
                            "<4s5H3L2H",
                            b"PK\x03\x04",
                            45,
                            info.flag_bits,
                            info.compress_type,
                            info.date_time[3] << 11
                            | info.date_time[4] << 5
                            | info.date_time[5] // 2,
                            (info.date_time[0] - 1980) << 9
                            | info.date_time[1] << 5
                            | info.date_time[2],
                            info.CRC,
                            0xFFFFFFFF,
                            0xFFFFFFFF,
                            len(name),
                            len(local_extra),
                        )
                    )
                    destination.write(name)
                    destination.write(local_extra)
                    remaining = info.compress_size
                    while remaining:
                        chunk = source.read(min(16 * 1024 * 1024, remaining))
                        if not chunk:
                            raise EOFError(f"truncated NPZ member {info.filename}")
                        destination.write(chunk)
                        remaining -= len(chunk)
                    central_records.append(
                        (
                            name,
                            info,
                            local_offset,
                        )
                    )
        central_offset = destination.tell()
        for name, info, local_offset in central_records:
            central_extra = struct.pack(
                "<HHQQQ", 0x0001, 24, info.file_size, info.compress_size, local_offset
            )
            destination.write(
                struct.pack(
                    "<4s6H3L5H2L",
                    b"PK\x01\x02",
                    (3 << 8) | 45,
                    45,
                    info.flag_bits,
                    info.compress_type,
                    info.date_time[3] << 11
                    | info.date_time[4] << 5
                    | info.date_time[5] // 2,
                    (info.date_time[0] - 1980) << 9
                    | info.date_time[1] << 5
                    | info.date_time[2],
                    info.CRC,
                    0xFFFFFFFF,
                    0xFFFFFFFF,
                    len(name),
                    len(central_extra),
                    0,
                    0,
                    0,
                    info.external_attr,
                    0xFFFFFFFF,
                )
            )
            destination.write(name)
            destination.write(central_extra)
        central_size = destination.tell() - central_offset
        destination.write(
            struct.pack(
                "<4s4H2LH",
                b"PK\x05\x06",
                0,
                0,
                len(central_records),
                len(central_records),
                central_size,
                central_offset,
                0,
            )
        )
    temporary_path.replace(merged_path)

    checkpoint = args.checkpoint.resolve()
    checkpoint_manifest = checkpoint.parent / ".pypto-a3-state" / "deepseek-v41-checkpoint-manifest.json"
    checkpoint_record = {
        "checkpoint_root": str(checkpoint),
        "formal_accuracy": "metadata_verified",
        "source": "models.deepseek_v4_1_flash_a2.decode_checkpoint_manifest",
        "manifest": str(checkpoint_manifest) if checkpoint_manifest.is_file() else None,
    }
    (output_dir / "checkpoint.json").write_text(
        json.dumps(checkpoint_record, indent=2) + "\n", encoding="utf-8"
    )

    first = records[0]
    decode_token = first.get("input_ids", [None])[0]
    prompt_ids = []
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            str(checkpoint), tokenizer_mode="deepseek_v41", trust_remote_code=False
        )
        prompt_ids = tokenizer.encode(args.prompt)[: args.history_length]
    except Exception:
        # The rank capture already contains the exact model input token and
        # position.  Keep the contract usable if tokenizer-only imports differ
        # between vLLM environment revisions.
        prompt_ids = []

    identity = _module_identity(args.release_root.resolve())
    identity["task_device"] = args.task_device
    identity["ascend_rt_visible_devices"] = args.task_device
    identity["capture_task"] = args.capture_task
    private_home = output_dir / "reference_ascend_home"
    metadata = private_home / "aarch64-linux" / "ascend_all_cann_install.info"
    identity["ascend_home"] = {
        "private": str(private_home),
        "metadata": str(metadata),
        "metadata_sha256": _sha256(metadata) if metadata.is_file() else "missing",
    }
    mhc_identity = {
        "mode": "vllm_mhc_pre_post_torch_npu_fallback",
        "implementation": "vllm.model_executor.kernels.mhc.torch.mhc_pre_delayed_torch+mhc_post_torch",
        "device": "NPU tensor operations in reference process",
    }
    topology = {**TOPOLOGY, "devices": 2}
    layer_state = _layer_state_contract(checkpoint, args.layer_id)
    decode_input = {
        "session": SESSION,
        "checkpoint": str(checkpoint),
        "layer_id": args.layer_id,
        "decode_token": decode_token,
        "history_token_length": args.history_length,
        "prompt_token_ids": prompt_ids,
        "position_ids": first.get("positions", [args.history_length]),
        "initial_kv_cache": "history_cache.npz",
        "block_table": [record.get("block_table") for record in records],
        "slot_mapping": [record.get("slot_mapping") for record in records],
        "valid_lengths": [record.get("valid_lengths") for record in records],
        "mask": {"type": "causal_swa", "window": 128, "readonly_history": True},
        "active_finished_request_state": {
            "active": [True, True],
            "finished": [False, False],
            "request_count": 1,
            "batch_padding": 1,
        },
        "engram_rows": {
            "required": layer_state["engram_required"],
            "captured": False,
            "reason": "reference runner does not expose Engram table rows in this capture",
        },
        "cache_source_ids": ["vllm_ascend.rank0", "vllm_ascend.rank1"],
        "ratio1_ratio2_state": layer_state,
        "topology": topology,
        "runtime_identity": identity,
        "reference_mhc": mhc_identity,
    }
    (output_dir / "decode_input.json").write_text(
        json.dumps(decode_input, indent=2) + "\n", encoding="utf-8"
    )
    reference = {
        "session": SESSION,
        "status": "captured",
        "role": "independent_reference_only",
        "commit": PINNED_VLLM_ASCEND,
        "vllm_reference_commit": PINNED_VLLM_ASCEND,
        "layer_id": args.layer_id,
        "topology": topology,
        "same_inputs": True,
        "same_layer": True,
        "same_topology": True,
        "same_state_boundary": True,
        "product_imported": False,
        "child_processes": True,
        "generated_token_ids": [decode_token] if decode_token is not None else [],
        "rank_capture": [
            {
                "rank": record["rank"],
                "cache_file": f"rank{record['rank']}/history_cache.npz",
                "capture_file": f"rank{record['rank']}/capture.json",
                "cache_layout": record["cache_layout"],
                "cache_arrays": record["cache_arrays"],
                "hidden_shape": [len(record["hidden"]), len(record["hidden"][0])]
                if record.get("hidden")
                else None,
                "moe_capture": record.get("moe_capture"),
                "layer_capture": record.get("layer_capture"),
                "result_type": record.get("result_type"),
            }
            for record in records
        ],
        "runtime_identity": identity,
        "reference_mhc": mhc_identity,
        "input_contract": "decode-token-cache-request-state-v1",
        "capture_command": "tools/deepseek_v41_vllm_capture.py --assemble-existing",
        "capture_task": args.capture_task,
        "generated_at_unix": time.time(),
    }
    (output_dir / "reference_capture.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "assembled", "output": str(output_dir)}, indent=2))
    return 0


def _run_capture(args: argparse.Namespace) -> int:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.layer_only:
        os.environ["VLLM_CAPTURE_LAYER_ONLY"] = "1"
    ascend_home_identity = _prepare_reference_ascend_home(output_dir)
    root = args.release_root.resolve()
    vendor_root = root / "source/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    vendor_lib = vendor_root / "op_api/lib"
    if vendor_lib.is_dir():
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
            item for item in (str(vendor_root), os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")) if item
        )
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            item for item in (str(vendor_lib), os.environ.get("LD_LIBRARY_PATH", "")) if item
        )
    # Triton A2/A3 compilation creates nested temporary files. The activated
    # reference env points TMPDIR at an NFS cache whose cleanup is racy across
    # external-launcher ranks, so keep only this process's scratch files local.
    local_tmp = Path("/tmp") / (
        f"deepseek-v41-capture-{os.getpid()}-{os.environ.get('LOCAL_RANK', '0')}"
    )
    local_tmp.mkdir(parents=True, exist_ok=True)
    for variable in ("TMPDIR", "TEMP", "TMP"):
        os.environ[variable] = str(local_tmp)
    import torch
    import torch.distributed as dist
    from vllm import LLM, SamplingParams

    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise RuntimeError("VLLM_USE_V2_MODEL_RUNNER=0 is required for the pinned reference capture")
    if os.environ.get("SOC_VERSION") != EXPECTED_SOC:
        raise RuntimeError(f"SOC_VERSION must be {EXPECTED_SOC}")
    if not os.environ.get("TASK_DEVICE"):
        raise RuntimeError("capture must run through task-submit with TASK_DEVICE")
    task_devices = {
        item.strip()
        for item in os.environ["TASK_DEVICE"].split(",")
        if item.strip()
    }
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    visible_set = {item.strip() for item in visible_devices.split(",") if item.strip()}
    if not visible_set:
        raise RuntimeError(
            "capture must export ASCEND_RT_VISIBLE_DEVICES to the queued TASK_DEVICE set"
        )
    if visible_set != task_devices:
        raise RuntimeError(
            "ASCEND_RT_VISIBLE_DEVICES must match TASK_DEVICE: "
            f"visible={visible_devices!r}, task={os.environ['TASK_DEVICE']!r}"
        )
    identity = _module_identity(root)
    identity["ascend_home"] = ascend_home_identity
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2:
        raise RuntimeError(f"reference capture requires two external-launcher ranks, got {world}")
    temporary_root = output_dir / f"reference_model_l{args.layer_id}"
    if rank == 0 and temporary_root.exists() and not (temporary_root / ".ready").is_file():
        shutil.rmtree(temporary_root)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    # Both external-launcher ranks share the output filesystem.  Publish the
    # symlink view once, otherwise concurrent tokenizer links race at startup.
    if rank == 0 and not (temporary_root / ".ready").is_file():
        _make_layer_root(args.checkpoint.resolve(), temporary_root, args.layer_id)
    else:
        _wait_for_layer_root(temporary_root)
    if args.mhc_mode != "fallback":
        raise RuntimeError(
            "the shared CANN runtime has no aclnnHcPre/aclnnHcPost; rerun with --mhc-mode fallback"
        )
    mhc_identity = _patch_reference_mhc_fallback()
    layer_only_identity = (
        _patch_layer_only_model(args.layer_id) if args.layer_only else {"enabled": False}
    )
    cache_group_identity = (
        _patch_empty_state_cache_group() if args.layer_only else {"enabled": False}
    )
    layer_state = _layer_state_contract(args.checkpoint.resolve(), args.layer_id)
    _patch_runner_capture(
        output_dir,
        rank,
        args.layer_id,
        layer_state,
        skip_cache=args.skip_cache,
    )

    llm = LLM(
        model=str(temporary_root),
        tokenizer=str(args.checkpoint.resolve()),
        tokenizer_mode="deepseek_v41",
        dtype="bfloat16",
        quantization="ascend",
        tensor_parallel_size=1,
        data_parallel_size=2,
        data_parallel_size_local=2,
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        max_num_batched_tokens=16,
        block_size=128,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        disable_custom_all_reduce=True,
        trust_remote_code=False,
        seed=17,
    )
    tokenizer = llm.get_tokenizer()
    prompt_ids = tokenizer.encode(args.prompt)
    if len(prompt_ids) < args.history_length:
        raise RuntimeError("prompt tokenization produced fewer tokens than history_length")
    prompt_ids = prompt_ids[: args.history_length]
    # Two generated tokens guarantee that the first generated token is a
    # separate Decode execution after the prompt prefill.
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=2, ignore_eos=True)
    outputs = llm.generate({"prompt_token_ids": prompt_ids}, sampling_params=sampling, use_tqdm=False)
    generated = outputs[0].outputs[0].token_ids if outputs else []
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank != 0:
        # vLLM-Ascend's external-launcher cleanup tries to destroy an HCCL
        # process group that has already been released by its worker.  The
        # rank-local capture is complete at this point.  Wait for rank zero
        # to publish the merged contract before skipping that duplicate
        # interpreter cleanup so torchrun sees a successful reference exit.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        os._exit(0)

    rank_records = []
    for peer in range(world):
        record_path = output_dir / f"rank{peer}" / "capture.json"
        cache_path = output_dir / f"rank{peer}" / "history_cache.npz"
        if not record_path.is_file() or (not args.skip_cache and not cache_path.is_file()):
            raise RuntimeError(f"rank {peer} did not produce a capture")
        if not args.skip_cache and not (output_dir / f"rank{peer}" / "layer_state_capture.npz").is_file():
            raise RuntimeError(f"rank {peer} did not produce the physical layer state capture")
        rank_records.append(json.loads(record_path.read_text(encoding="utf-8")))

    rank_state = rank_records[0].get("layer_state_capture") or {}

    import numpy as np

    if not args.skip_cache:
        merged_cache: dict[str, Any] = {}
        for peer in range(world):
            with np.load(output_dir / f"rank{peer}" / "history_cache.npz") as archive:
                merged_cache.update({f"rank{peer}_{name}": archive[name] for name in archive.files})
        np.savez(output_dir / "history_cache.npz", **merged_cache)

    asset_root = output_dir
    checkpoint_manifest = args.checkpoint.parent / ".pypto-a3-state" / "deepseek-v41-checkpoint-manifest.json"
    checkpoint_record = {
        "checkpoint_root": str(args.checkpoint.resolve()),
        "formal_accuracy": "metadata_verified",
        "source": "models.deepseek_v4_1_flash_a2.decode_checkpoint_manifest",
        "manifest": str(checkpoint_manifest) if checkpoint_manifest.is_file() else None,
    }
    (asset_root / "checkpoint.json").write_text(
        json.dumps(checkpoint_record, indent=2) + "\n", encoding="utf-8"
    )

    # The product contract uses one shared input description and a rank-indexed
    # cache archive.  Keep both rank records so communication and untouched
    # regions can be checked without reconstructing the reference process.
    layer_state = _layer_state_contract(args.checkpoint.resolve(), args.layer_id)
    decode_input = {
        "session": SESSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "layer_id": args.layer_id,
        "decode_token": int(generated[0]) if generated else None,
        "history_token_length": args.history_length,
        "prompt_token_ids": prompt_ids,
        "position_ids": [args.history_length],
        "initial_kv_cache": None if args.skip_cache else "history_cache.npz",
        "block_table": [record["block_table"] for record in rank_records],
        "slot_mapping": [record["slot_mapping"] for record in rank_records],
        "valid_lengths": [record["valid_lengths"] for record in rank_records],
        "mask": {"type": "causal_swa", "window": 128, "readonly_history": True},
        "active_finished_request_state": {
            "active": [True, True],
            "finished": [False, False],
            "request_count": 1,
            "batch_padding": 1,
        },
        "engram_rows": {
            "required": layer_state["engram_required"],
            "captured": False,
            "reason": "reference runner does not expose Engram table rows in this capture",
        },
        "cache_source_ids": ["vllm_ascend.rank0", "vllm_ascend.rank1"],
        "ratio1_ratio2_state": layer_state,
        "topology": {**TOPOLOGY, "devices": 2},
        "runtime_identity": identity,
        "reference_mhc": mhc_identity,
        "layer_only_reference": layer_only_identity,
        "cache_group_reference": cache_group_identity,
        "input_boundary_mode": (
            "target_layer_with_identity_prefix" if args.layer_only else "truncated_physical_prefix"
        ),
    }
    (asset_root / "decode_input.json").write_text(json.dumps(decode_input, indent=2) + "\n", encoding="utf-8")

    reference = {
        "session": SESSION,
        "status": "captured",
        "role": "independent_reference_only",
        "commit": PINNED_VLLM_ASCEND,
        "vllm_reference_commit": PINNED_VLLM_ASCEND,
        "layer_id": args.layer_id,
        "topology": {**TOPOLOGY, "devices": 2},
        "same_inputs": True,
        "same_layer": True,
        "same_topology": True,
        "same_state_boundary": True,
        "product_imported": False,
        "child_processes": True,
        "generated_token_ids": list(generated),
        "rank_capture": [
            {
                "rank": record["rank"],
                "cache_file": f"rank{record['rank']}/history_cache.npz",
                "capture_file": f"rank{record['rank']}/capture.json",
                "cache_layout": record["cache_layout"],
                "cache_arrays": record["cache_arrays"],
                "layer_state_capture": record.get("layer_state_capture"),
                "hidden_shape": [len(record["hidden"]), len(record["hidden"][0])] if record.get("hidden") else None,
                "moe_capture": record.get("moe_capture"),
                "layer_capture": record.get("layer_capture"),
                "result_type": record.get("result_type"),
            }
            for record in rank_records
        ],
        "runtime_identity": identity,
        "reference_mhc": mhc_identity,
        "layer_state_capture": {
            "complete": all(record.get("layer_state_capture", {}).get("complete") is True for record in rank_records),
            "compressed_kv_cache": rank_state.get("compressed_kv_cache", {"captured": False}),
            "index_cache": rank_state.get("index_cache", {"captured": False}),
            "index_scale": rank_state.get("index_scale", {"captured": False}),
            "candidate_pool": rank_state.get("candidate_pool", {"captured": False}),
            "swa_cache": rank_state.get("swa_cache", {"captured": False}),
            "per_rank": [record.get("layer_state_capture") for record in rank_records],
        },
        "layer_only_reference": layer_only_identity,
        "cache_group_reference": cache_group_identity,
        "input_boundary_mode": (
            "target_layer_with_identity_prefix" if args.layer_only else "truncated_physical_prefix"
        ),
        "input_contract": (
            "decode-token-cache-request-state-v1"
            if not args.skip_cache
            else "reference-moe-diagnostic-no-cache-v1"
        ),
        "capture_command": "tools/deepseek_v41_vllm_capture.py",
        "generated_at_unix": time.time(),
    }
    (asset_root / "reference_capture.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    print(json.dumps({"status": "captured", "output": str(asset_root), "generated": generated}, indent=2), flush=True)
    # See the rank-1 note above.  This process has already flushed all
    # contract files and must not enter the duplicate HCCL teardown path.
    os._exit(0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument(
        "--layer-only",
        action="store_true",
        help="reference-only: instantiate only the requested physical layer and identity placeholders",
    )
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0.0,
        help="reference-only vLLM CPU weight offload budget per rank",
    )
    parser.add_argument("--assemble-existing", action="store_true")
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="diagnostic-only capture: omit the large KV cache snapshots and merged cache archive",
    )
    parser.add_argument("--task-device", default="not_recorded")
    parser.add_argument("--capture-task", default="not_recorded")
    parser.add_argument(
        "--mhc-mode",
        choices=("fallback",),
        default="fallback",
        help="reference-only mHC implementation; CANN 9.1 lacks the native HcPre ABI",
    )
    parser.add_argument("--prompt", default="DeepSeek V4.1 decode capture for PyPTO.")
    args = parser.parse_args(argv)
    if args.assemble_existing:
        return _assemble_existing(args)
    return _run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
