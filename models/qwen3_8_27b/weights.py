# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fetch, validate, and stream Qwen3.8-27B text weights.

There are two deliberately separate paths:

* The Q0-Q4 compatibility path range-fetches one BF16 GDN layer from the
  official checkpoint. Existing per-block validation scripts use its small
  ``.pt`` file.
* Q5 loads all 64 text layers, embeddings, final norm, and LM head from the
  pinned Eco-Tech W8A8 checkpoint. It excludes the vision tower and MTP layer,
  validates all 1,555 checkpoint tensors against :mod:`contract`, discards the
  symmetric all-zero offsets, flattens scales to the kernel convention, and
  exports one tensor at a time so host memory need not hold a second 29 GiB
  copy while weights move to device.

The whole checkpoint is a large, explicit download:

    python models/qwen3_8_27b/weights.py --fetch-checkpoint
    python models/qwen3_8_27b/weights.py --validate-checkpoint

``--verify-index`` is the cheap third option. It checks :mod:`contract`
against the *published* checkpoint -- names, every shape and dtype, and the
byte totals -- from the index and the ten safetensors headers alone, a few
hundred KiB rather than the checkpoint's 29.92 GiB:

    python models/qwen3_8_27b/weights.py --verify-index

With no checkpoint flag the command remains CI-safe and retains the original
single-layer behavior. Missing data is reported rather than downloaded.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shlex
import shutil
import struct
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any, BinaryIO

import torch

if __package__:
    from . import reference
    from .config import Qwen38Config
    from .contract import (
        BF16,
        FP32,
        INT8,
        QWEN3_8_27B_TEXT_CONTRACT,
        Qwen38TextContract,
        WeightSpec,
    )
else:
    _REPO_ROOT = str(Path(__file__).resolve().parents[2])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from models.qwen3_8_27b import reference
    from models.qwen3_8_27b.config import Qwen38Config
    from models.qwen3_8_27b.contract import (
        BF16,
        FP32,
        INT8,
        QWEN3_8_27B_TEXT_CONTRACT,
        Qwen38TextContract,
        WeightSpec,
    )


BF16_REPO = "Qwen/Qwen3.8-27B"
BF16_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
BF16_PREFIX = "model.language_model.layers.{layer}.linear_attn."
# Q0-Q4 compatibility names.
REPO = BF16_REPO
REVISION = BF16_REVISION
PREFIX = BF16_PREFIX

ECO_REPO = "Eco-Tech/Qwen3.8-27B-w8a8"
ECO_REVISION = "ca2937f18b7e399c0ee91c728a4cde9b837aebff"
ECO_INDEX = "quant_model_weights.safetensors.index.json"
ECO_METADATA_FILES = ("config.json", "quant_model_description.json", ECO_INDEX)
# Which revision a checkpoint directory holds, written by `fetch_checkpoint` once its
# shards and index agree. While a fetch is replacing shards the stamp reads
# `<_STAMP_FETCHING> <revision>` instead, so an interrupted one stays marked.
_REVISION_STAMP = ".checkpoint_revision"
_STAMP_FETCHING = "fetching"

_SAFETENSOR_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_CONTRACT_DTYPES = {BF16: torch.bfloat16, FP32: torch.float32, INT8: torch.int8}

TensorExporter = Callable[[torch.Tensor], Any]


@dataclass(frozen=True)
class PreparedQwen38Weights:
    """Kernel/reference-ready text tensors exported from the Eco-Tech checkpoint."""

    tensors: dict[str, Any]
    layer_types: tuple[str, ...]
    checkpoint_weight_bytes: int
    resident_weight_bytes: int

    def layer(self, index: int) -> dict[str, Any]:
        """Return one layer's tensors with the layer prefix removed."""
        if not 0 <= index < len(self.layer_types):
            raise IndexError(f"layer index {index} outside [0, {len(self.layer_types)})")
        prefix = f"model.language_model.layers.{index}."
        return {name[len(prefix) :]: value for name, value in self.tensors.items() if name.startswith(prefix)}


# The tensor groups one layer can be fetched in. `mlp` and `norms` are on every
# layer; the two mixers are on the layers their type says.
MIXERS = ("linear_attn", "self_attn", "mlp", "norms")
TEXT_GROUP = "text"     # the three tensors that belong to no decoder layer
FETCH_GROUPS = MIXERS + (TEXT_GROUP,)

# The embedding table, the model's final norm and the LM head, keyed as
# `prefill_text_fwd` takes them. The two vocabulary-sized ones are 2.37 GiB each and sit in
# different shards, so this group is fetched by name rather than by layer prefix.
TEXT_TENSORS = {"embed_weight": "model.language_model.embed_tokens.weight",
                "final_norm_w": "model.language_model.norm.weight",
                "lm_w": "lm_head.weight"}
_STEMS = {"linear_attn": "linear_attn", "self_attn": "full_attn", "mlp": "mlp",
          "norms": "norms", "text": "text"}

# `norms` is the only group that is not a submodule: the two layer norms sit
# directly under the layer, so they are named rather than taken by prefix.
LAYER_NORMS = ("input_layernorm.weight", "post_attention_layernorm.weight")

# Tensors this far apart in a shard are fetched as separate range requests.
_FETCH_GAP = 1 << 20


def default_path(layer: int, quantized: bool = False, mixer: str = "linear_attn") -> Path:
    """Path for one extracted BF16 layer group: GDN, gated attention, or the MLP."""
    root = Path(__file__).resolve().parents[2]
    suffix = "_int8" if quantized else ""
    return root / "build_output" / "qwen3_8_27b" / f"{_STEMS[mixer]}_layer{layer}{suffix}.pt"


def text_path(root: str | None = None) -> Path:
    """Path for the extracted non-layer tensors: the embedding, the final norm, the head."""
    if root is not None:
        return Path(root) / "text.pt"
    return default_path(0, mixer=TEXT_GROUP).with_name("text.pt")


def layer_paths(layer: int, kind: str, root: str | None = None) -> dict[str, str]:
    """The three dumps one real decoder layer needs, keyed as a layer entry takes them.

    *kind* is "gdn" for a linear-attention layer and "attn" for full attention; *root*
    overrides the directory, for a dump set that does not sit where `--fetch` writes.
    Both layer entries read their real weights through here: this file owns the names
    `--fetch` writes, and the two entries must not import each other.
    """
    groups = {"weights": "linear_attn" if kind == "gdn" else "self_attn",
              "mlp_weights": "mlp", "norm_weights": "norms"}
    out = {}
    for key, group in groups.items():
        path = default_path(layer, mixer=group)
        if root is not None:
            path = Path(root) / path.name
        if not path.is_file():
            output = f" --out {shlex.quote(str(path))}" if root is not None else ""
            raise SystemExit(f"[weights] {path} is not on disk; fetch it with\n"
                             f"  python weights.py --fetch --layer {layer} --mixer {group}{output}")
        out[key] = str(path)
    return out


def default_checkpoint_path() -> Path:
    """Default local directory for the pinned Eco-Tech checkpoint."""
    root = Path(__file__).resolve().parents[2]
    return root / "build_output" / "qwen3_8_27b" / "Eco-Tech-Qwen3.8-27B-w8a8"


def _get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600) as response:
        if byte_range is not None and response.status != 206:
            raise RuntimeError(f"range request not honoured ({response.status}) for {url}")
        return response.read()


def fetch_layer(
    layer: int,
    repo: str = BF16_REPO,
    revision: str = BF16_REVISION,
    log: Callable[[str], None] = print,
    mixer: str = "linear_attn",
) -> dict[str, torch.Tensor]:
    """Fetch one official BF16 layer group -- a mixer, the ``mlp`` or the two ``norms`` -- by HTTP range."""
    if mixer not in MIXERS:
        raise ValueError(f"mixer must be one of {MIXERS}, got {mixer!r}")
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    prefix = f"model.language_model.layers.{layer}."
    if mixer != "norms":
        prefix += f"{mixer}."

    def wanted(name: str) -> bool:
        """Whether a prefix-stripped tensor name belongs to the requested group."""
        return name in LAYER_NORMS if mixer == "norms" else True

    index = json.loads(_get(base + "model.safetensors.index.json"))["weight_map"]
    names = [name for name in index if name.startswith(prefix) and wanted(name[len(prefix) :])]
    if not names:
        raise RuntimeError(f"layer {layer}: no {mixer} tensors in the index")
    # A shard boundary falls inside a layer every eight layers or so, which puts that
    # layer's mixer and its two norms in two files. Offsets are shard-local, so the range
    # plan below is per shard.
    shards = sorted({index[name] for name in names})

    out = {}
    for shard in shards:
        header_len = struct.unpack("<Q", _get(base + shard, (0, 7)))[0]
        header = json.loads(_get(base + shard, (8, 8 + header_len - 1)))
        header.pop("__metadata__", None)
        entries = {name[len(prefix) :]: info for name, info in header.items()
                   if name.startswith(prefix) and wanted(name[len(prefix) :])}
        # One range request per contiguous run of tensors, not one spanning all of them. A
        # module's tensors are adjacent in the shard and coalesce into a single read, but the
        # two layer norms sit at opposite ends of the layer, and the span between them is the
        # whole layer -- 731 MiB to fetch 20 KiB.
        ordered = sorted(entries.items(), key=lambda kv: kv[1]["data_offsets"][0])
        runs: list[list[int]] = []
        for _, info in ordered:
            start, end = info["data_offsets"]
            if runs and start - runs[-1][1] <= _FETCH_GAP:
                runs[-1][1] = max(runs[-1][1], end)
            else:
                runs.append([start, end])
        fetched = sum(hi - lo for lo, hi in runs)
        log(
            f"[weights] layer {layer}: {len(entries)} tensors, {fetched / 2**20:.1f} MiB in "
            f"{len(runs)} range request(s) from {shard} at revision {revision[:8]}"
        )

        data_start = 8 + header_len
        blobs = [_get(base + shard, (data_start + lo, data_start + hi - 1)) for lo, hi in runs]
        for name, info in entries.items():
            start, end = info["data_offsets"]
            run = next(i for i, (lo, hi) in enumerate(runs) if lo <= start and end <= hi)
            raw = bytearray(blobs[run][start - runs[run][0] : end - runs[run][0]])
            out[name] = (
                torch.frombuffer(raw, dtype=_SAFETENSOR_DTYPES[info["dtype"]]).reshape(info["shape"]).clone()
            )
    missing = [name[len(prefix) :] for name in names if name[len(prefix) :] not in out]
    if missing:
        raise RuntimeError(f"layer {layer}: the index names tensors no shard header carries: {missing[:3]}")
    return out


def fetch_text(
    repo: str = BF16_REPO,
    revision: str = BF16_REVISION,
    log: Callable[[str], None] = print,
) -> dict[str, torch.Tensor]:
    """Fetch the embedding table, the model's final norm and the LM head by HTTP range.

    Unlike `fetch_layer` these three are not one module in one shard, so they are looked up
    by name and fetched a shard at a time. One range request is issued per contiguous run,
    and each run's blob is released before the next is read: the two vocabulary-sized
    tensors are 2.37 GiB each and holding both at once is 4.8 GiB of host memory for no
    reason.
    """
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    index = json.loads(_get(base + "model.safetensors.index.json"))["weight_map"]
    missing = [name for name in TEXT_TENSORS.values() if name not in index]
    if missing:
        raise RuntimeError(f"not in the index at revision {revision[:8]}: {missing}")
    by_shard: dict[str, list[str]] = {}
    for name in TEXT_TENSORS.values():
        by_shard.setdefault(index[name], []).append(name)

    short = {full: key for key, full in TEXT_TENSORS.items()}
    out = {}
    for shard, names in sorted(by_shard.items()):
        header_len = struct.unpack("<Q", _get(base + shard, (0, 7)))[0]
        header = json.loads(_get(base + shard, (8, 8 + header_len - 1)))
        entries = {name: header[name] for name in names}
        ordered = sorted(entries.items(), key=lambda kv: kv[1]["data_offsets"][0])
        runs: list[list[int]] = []
        for _, info in ordered:
            start, end = info["data_offsets"]
            if runs and start - runs[-1][1] <= _FETCH_GAP:
                runs[-1][1] = max(runs[-1][1], end)
            else:
                runs.append([start, end])
        fetched = sum(hi - lo for lo, hi in runs)
        log(f"[weights] {shard}: {len(entries)} tensor(s), {fetched / 2**20:.1f} MiB in "
            f"{len(runs)} range request(s) at revision {revision[:8]}")
        data_start = 8 + header_len
        for lo, hi in runs:
            blob = _get(base + shard, (data_start + lo, data_start + hi - 1))
            for name, info in entries.items():
                start, end = info["data_offsets"]
                if not (lo <= start and end <= hi):
                    continue
                raw = bytearray(blob[start - lo : end - lo])
                dtype = _SAFETENSOR_DTYPES[info["dtype"]]
                tensor = torch.frombuffer(raw, dtype=dtype).reshape(info["shape"]).clone()
                del raw
                out[short[name]] = tensor
                log(f"[weights]   {short[name]:<14} {tuple(tensor.shape)} {tensor.dtype}")
            del blob
    return out


def load_layer(path: str | Path) -> dict[str, torch.Tensor]:
    """Load one legacy extracted GDN layer."""
    return torch.load(path, weights_only=True)


def load_weights(cfg: Qwen38Config, path: str | Path | None) -> dict[str, torch.Tensor]:
    """Load extracted layer weights or generate deterministic fixtures."""
    if path is None:
        return reference.make_block_weights(cfg)
    weights = load_layer(path)
    out_proj = weights.get("out_proj.weight")
    if out_proj is not None and not out_proj.is_floating_point():
        raise ValueError(f"{path}: out_proj.weight must be floating point. Regenerate the dump from the BF16 checkpoint.")
    return weights


def quantized_weight(
    cfg: Qwen38Config, module: str, path: str | Path | None,
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    """Return lazy initializers for a projection and its per-row scales."""
    @cache
    def load():
        name = f"{module}.weight"
        layer = reference.quantize_weights(load_weights(cfg, path), names=(name,))
        return layer[name], layer[f"{module}.weight_scale"]

    return (lambda: load()[0]), (lambda: load()[1])


def stacked_ab_weight(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack the A and B projection weights in the fused BF16 layout."""
    return torch.cat([weights["in_proj_a.weight"], weights["in_proj_b.weight"]]).to(torch.bfloat16)


def modelslim_quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce ModelSlim's symmetric per-channel INT8 min/max layout.

    Returns ``weight``, ``weight_scale`` and ``weight_offset`` with the exact
    Eco-Tech shapes. ModelSlim performs the division in the BF16 source dtype,
    then saves the BF16-rounded scale in an FP32 tensor; preserving that order
    is required for bit identity. The offset is zero because the recipe is
    symmetric.

    One detail is reproduced but unconfirmed: the ``amax`` floor is Q4's own
    ``GDN_QUANT.amax_eps``, not a value read back from ModelSlim. The bit
    identity check that established this recipe used real ``in_proj_qkv`` rows,
    whose ``amax`` is orders of magnitude above the floor, so no comparison so
    far has exercised it. A checkpoint row that is entirely zero would be the
    first, and its scale would then rest on this assumption.
    """
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError(f"expected a BF16 [out, in] weight, got {tuple(weight.shape)} {weight.dtype}")
    amax = weight.abs().amax(dim=1).clamp_min(1e-4)
    native_scale = amax / 127.0
    quantized = torch.round(weight / native_scale[:, None]).clamp(-127, 127).to(torch.int8)
    scale = native_scale.float()[:, None].contiguous()
    offset = torch.zeros_like(scale)
    return quantized, scale, offset


def _weight_map(index: Any) -> dict[str, str]:
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError("invalid safetensors index: expected a nonempty weight_map")
    for name, shard in weights.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("safetensors weight_map must map tensor names to shard filenames")
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"checkpoint index contains unsafe shard path: {shard}")
    return weights


def _load_checkpoint_index(checkpoint_dir: str | Path) -> dict[str, str]:
    path = Path(checkpoint_dir) / ECO_INDEX
    if not path.is_file():
        raise FileNotFoundError(f"missing Eco-Tech checkpoint index: {path}")
    return _weight_map(json.loads(path.read_text()))


def _text_shards(weight_map: dict[str, str], contract: Qwen38TextContract) -> dict[str, list[WeightSpec]]:
    expected = {spec.name for spec in contract.weight_specs}
    actual_text = {
        name for name in weight_map if name == "lm_head.weight" or name.startswith("model.language_model.")
    }
    missing = sorted(expected - actual_text)
    unexpected = sorted(actual_text - expected)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} tensors (first: {missing[:3]})")
        if unexpected:
            detail.append(f"unexpected {len(unexpected)} text tensors (first: {unexpected[:3]})")
        raise ValueError("checkpoint does not match Qwen3.8-27B text contract: " + "; ".join(detail))

    shards: dict[str, list[WeightSpec]] = {}
    for spec in contract.weight_specs:
        shards.setdefault(weight_map[spec.name], []).append(spec)
    return shards


def _write_revision_stamp(checkpoint_dir: Path, value: str) -> None:
    tmp = checkpoint_dir / f"{_REVISION_STAMP}.tmp"
    tmp.write_text(f"{value}\n")
    os.replace(tmp, checkpoint_dir / _REVISION_STAMP)


def _check_checkpoint_revision(checkpoint_dir: Path, revision: str) -> None:
    """Refuse a directory an interrupted or differently-pinned fetch left behind.

    ``_fetch_checkpoint`` replaces shards in place and publishes the index last, so a
    fetch that dies in between leaves bytes from two revisions under an index that
    still describes the old one. Names, shapes and dtypes survive a requant, so the
    rest of the loader's checks pass and the mix would load silently. A directory with
    no stamp was not written by ``fetch_checkpoint`` and is taken as the caller's own.
    """
    try:
        stamp = (checkpoint_dir / _REVISION_STAMP).read_text().strip()
    except FileNotFoundError:
        return
    if stamp == revision:
        return
    if stamp.startswith(_STAMP_FETCHING):
        fetching = stamp.split()[-1]
        detail = (f"a fetch of {fetching[:8]} that did not finish" if fetching == revision
                  else f"an unfinished fetch of {fetching[:8]}, over the pinned {revision[:8]}")
    else:
        detail = f"revision {stamp[:8]}, not the pinned {revision[:8]}"
    raise RuntimeError(
        f"{checkpoint_dir} holds {detail}; its shards may mix revisions. Rerun "
        f"`python models/qwen3_8_27b/weights.py --fetch-checkpoint` to complete it"
    )


def _check_local_shards(checkpoint_dir: Path, shards: dict[str, list[WeightSpec]]) -> None:
    absent_shards = sorted(name for name in shards if not (checkpoint_dir / name).is_file())
    if absent_shards:
        raise FileNotFoundError(
            f"checkpoint is missing {len(absent_shards)} required shards: {absent_shards}"
        )


def checkpoint_summary(
    checkpoint_dir: str | Path,
    contract: Qwen38TextContract = QWEN3_8_27B_TEXT_CONTRACT,
    *,
    revision: str = ECO_REVISION,
) -> dict[str, int]:
    """Validate the index, the required shards and their headers, reading no tensor data.

    The byte totals come from the shards' own ``data_offsets``, the way
    :func:`verify_published_contract` derives them from the remote headers. A
    present-and-named check cannot tell a truncated shard from a complete one,
    and reporting the contract's own constants back would say nothing about
    what is on disk.
    """
    checkpoint_dir = Path(checkpoint_dir)
    _check_checkpoint_revision(checkpoint_dir, revision)
    shards = _text_shards(_load_checkpoint_index(checkpoint_dir), contract)
    _check_local_shards(checkpoint_dir, shards)
    summary = _verify_text_shard_headers(
        shards, contract, lambda shard: _local_header(checkpoint_dir / shard), lambda _message: None,
    )
    return {**summary, "prepared_tensors": len(contract.prepared_weight_specs)}


def _remote_header(repo: str, revision: str, shard: str) -> dict[str, Any]:
    """Read one remote shard's safetensors header by HTTP range, no payload."""
    url = _modelscope_url(repo, revision, shard)
    header_len = struct.unpack("<Q", _get(url, (0, 7)))[0]
    header = json.loads(_get(url, (8, 8 + header_len - 1)))
    header.pop("__metadata__", None)
    return header


def verify_published_contract(
    contract: Qwen38TextContract = QWEN3_8_27B_TEXT_CONTRACT,
    *,
    repo: str = ECO_REPO,
    revision: str = ECO_REVISION,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Check :mod:`contract` against the published checkpoint, downloading no weights.

    The assertions in :mod:`contract` pin the table against accidental edits,
    but they compare it only with itself: they would pass unchanged if the
    published checkpoint moved underneath us. This fetches the index and the
    ten safetensors headers -- a few hundred KiB in total, no tensor payload --
    and checks the names, every shape and dtype, and the byte totals against
    what the contract claims. It needs the network and never runs unasked.
    """
    index = _weight_map(json.loads(_get(_modelscope_url(repo, revision, ECO_INDEX))))
    names_by_shard = _text_shards(index, contract)
    log(f"[weights] index at {revision[:8]}: {len(contract.weight_specs)} text tensors, names match the contract")

    return _verify_text_shard_headers(
        names_by_shard, contract, lambda shard: _remote_header(repo, revision, shard), log,
    )


def _verify_text_shard_headers(
    names_by_shard: dict[str, list[WeightSpec]],
    contract: Qwen38TextContract,
    read_header: Callable[[str], dict[str, Any]],
    log: Callable[[str], None],
) -> dict[str, int]:
    shards = sorted(names_by_shard)
    header: dict[str, Any] = {}
    for position, shard in enumerate(shards, 1):
        entries = read_header(shard)
        specs = names_by_shard[shard]
        _verify_assigned_tensors(shard, specs, entries)
        header.update((spec.name, entries[spec.name]) for spec in specs)
        log(f"[weights] [{position}/{len(shards)}] header of {shard}: {len(entries)} tensors")

    def span(spec: WeightSpec) -> int:
        start, end = header[spec.name]["data_offsets"]
        return end - start

    checkpoint_bytes = sum(span(spec) for spec in contract.weight_specs)
    resident_bytes = sum(span(spec) for spec in contract.prepared_weight_specs)
    for label, actual_bytes, claimed in (
        ("checkpoint_weight_bytes", checkpoint_bytes, contract.memory.checkpoint_weight_bytes),
        ("resident_weight_bytes", resident_bytes, contract.memory.resident_weight_bytes),
    ):
        if actual_bytes != claimed:
            raise ValueError(f"{label}: checkpoint has {actual_bytes}, contract claims {claimed}")

    log(
        f"[weights] {len(contract.weight_specs)} shapes and dtypes agree; payload {checkpoint_bytes} bytes "
        f"({checkpoint_bytes / 2**30:.2f} GiB), resident {resident_bytes} bytes "
        f"({resident_bytes / 2**30:.2f} GiB) -- both as pinned"
    )
    return {
        "text_tensors": len(contract.weight_specs),
        "shards": len(shards),
        "checkpoint_weight_bytes": checkpoint_bytes,
        "resident_weight_bytes": resident_bytes,
    }


def _verify_assigned_tensors(shard: str, specs: list[WeightSpec], entries: dict[str, Any]) -> None:
    missing = [spec.name for spec in specs if spec.name not in entries]
    if missing:
        raise ValueError(f"{shard}: index names absent from safetensors header: {missing[:3]}")
    names = {BF16: "BF16", FP32: "F32", INT8: "I8"}
    for spec in specs:
        entry = entries[spec.name]
        start, end = entry["data_offsets"]
        if tuple(entry["shape"]) != spec.shape or entry["dtype"] != names[spec.dtype] or end - start != spec.nbytes:
            raise ValueError(f"{shard}: {spec.name} shape, dtype or size disagrees with the text contract")


def _validate_tensor(tensor: torch.Tensor, spec: WeightSpec) -> None:
    expected_dtype = _CONTRACT_DTYPES[spec.dtype]
    if tensor.dtype != expected_dtype or tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: got shape={tuple(tensor.shape)} dtype={tensor.dtype}; "
            f"expected shape={spec.shape} dtype={expected_dtype}"
        )


def prepare_qwen38_weights(
    checkpoint_dir: str | Path,
    tensor_exporter: TensorExporter,
    *,
    contract: Qwen38TextContract = QWEN3_8_27B_TEXT_CONTRACT,
    verify_zero_offsets: bool = True,
    revision: str = ECO_REVISION,
) -> PreparedQwen38Weights:
    """Stream the complete text checkpoint into the runtime's tensor exporter.

    A shard is opened once and tensors are exported individually. Symmetric
    ``weight_offset`` tensors are validated and omitted; ``weight_scale`` is
    flattened from checkpoint ``[out, 1]`` to kernel ``[out]``. All remaining
    names stay checkpoint-native, so the returned dictionary also feeds the Q5a
    reference directly.
    """
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("prepare_qwen38_weights requires the safetensors package") from error

    checkpoint_dir = Path(checkpoint_dir)
    _check_checkpoint_revision(checkpoint_dir, revision)
    shards = _text_shards(_load_checkpoint_index(checkpoint_dir), contract)
    _check_local_shards(checkpoint_dir, shards)

    prepared: dict[str, Any] = {}
    for shard, specs in shards.items():
        shard_path = checkpoint_dir / shard
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            shard_keys = set(handle.keys())
            missing = [spec.name for spec in specs if spec.name not in shard_keys]
            if missing:
                raise ValueError(f"{shard}: index names absent from safetensors header: {missing[:3]}")
            for spec in specs:
                name = spec.name
                tensor = handle.get_tensor(name)
                _validate_tensor(tensor, spec)
                if spec.is_zero_offset:
                    if verify_zero_offsets and bool(torch.count_nonzero(tensor)):
                        raise ValueError(f"{name}: symmetric W8A8 checkpoint offset is not zero")
                    continue
                if spec.role == "weight_scale":
                    tensor = tensor.reshape(-1)
                prepared[name] = tensor_exporter(tensor.contiguous())

    expected_prepared = [spec.name for spec in contract.prepared_weight_specs]
    prepared = {name: prepared[name] for name in expected_prepared}
    return PreparedQwen38Weights(
        tensors=prepared,
        layer_types=contract.layer_types,
        checkpoint_weight_bytes=contract.memory.checkpoint_weight_bytes,
        resident_weight_bytes=contract.memory.resident_weight_bytes,
    )


def load_text_weights(
    checkpoint_dir: str | Path,
    *,
    contract: Qwen38TextContract = QWEN3_8_27B_TEXT_CONTRACT,
) -> dict[str, torch.Tensor]:
    """Load prepared text weights on CPU; intended for reference validation."""
    return prepare_qwen38_weights(checkpoint_dir, lambda tensor: tensor, contract=contract).tensors


def _modelscope_url(repo: str, revision: str, file_path: str) -> str:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))
    query = urllib.parse.urlencode({"Revision": revision, "FilePath": file_path})
    return f"https://modelscope.cn/api/v1/models/{quoted_repo}/repo?{query}"


def _modelscope_manifest(repo: str, revision: str) -> dict[str, dict[str, Any]]:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo.split("/"))
    query = urllib.parse.urlencode({"Revision": revision, "Root": ""})
    url = f"https://modelscope.cn/api/v1/models/{quoted_repo}/repo/files?{query}"
    response = json.loads(_get(url))
    if response.get("Code") != 200:
        raise RuntimeError(f"ModelScope manifest request failed: {response}")
    return {item["Name"]: item for item in response["Data"]["Files"] if item["Type"] == "blob"}


def _digest_marker(destination: Path) -> Path:
    """Sidecar recording the SHA-256 this code verified for ``destination``."""
    return destination.with_name(destination.name + ".sha256")


def _file_sha256(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 2**20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _file_stamp(sha256: str, status: os.stat_result) -> str:
    return f"{sha256} {status.st_dev} {status.st_ino} {status.st_size} {status.st_mtime_ns} {status.st_ctime_ns}"


def _verified_locally(destination: Path, expected_size: int, expected_sha256: str) -> bool:
    """Reuse a digest for the same file version or hash the open file."""
    try:
        with destination.open("rb") as handle:
            status = os.fstat(handle.fileno())
            if status.st_size != expected_size:
                return False
            stamp = _file_stamp(expected_sha256, status)
            try:
                cached = _digest_marker(destination).read_text().strip() == stamp
            except FileNotFoundError:
                cached = False
            if not cached and _file_sha256(handle) != expected_sha256:
                return False
            if not cached:
                _write_digest_marker(destination, stamp)
            return _file_stamp(expected_sha256, destination.stat()) == stamp
    except FileNotFoundError:
        return False


def _write_digest_marker(destination: Path, stamp: str) -> None:
    marker = _digest_marker(destination)
    with tempfile.NamedTemporaryFile("w", dir=marker.parent, prefix=f".{marker.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(stamp + "\n")
            handle.close()
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)


def _download_file(
    url: str, destination: Path, expected_size: int, expected_sha256: str,
    validate: Callable[[Path], None] | None = None,
) -> None:
    with urllib.request.urlopen(url, timeout=600) as response:
        _write_verified_file(response, destination, expected_size, expected_sha256, validate)


def _write_verified_file(
    source: BinaryIO, destination: Path, expected_size: int, expected_sha256: str,
    validate: Callable[[Path], None] | None = None,
) -> None:
    digest = hashlib.sha256()
    written = 0
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as output:
        part = Path(output.name)
        try:
            for chunk in iter(lambda: source.read(8 * 2**20), b""):
                output.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            if written != expected_size:
                raise RuntimeError(f"{destination.name}: downloaded {written} bytes, expected {expected_size}")
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(f"{destination.name}: sha256 {actual_sha256} != expected {expected_sha256}")
            output.flush()
            if validate is not None:
                validate(part)
            part.replace(destination)
            stamp = _file_stamp(expected_sha256, os.fstat(output.fileno()))
            _write_digest_marker(destination, stamp)
        finally:
            part.unlink(missing_ok=True)


def _local_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        file_size = os.fstat(handle.fileno()).st_size
        # The safetensors header is bounded to 100 MB.
        if not 0 < size <= min(file_size - 8, 100_000_000):
            raise ValueError(f"{path.name}: invalid safetensors header size")
        entries = json.loads(handle.read(size))
    if not isinstance(entries, dict):
        raise ValueError(f"{path.name}: invalid safetensors header")
    entries.pop("__metadata__", None)
    payload_size = file_size - 8 - size
    for name, entry in entries.items():
        start, end = entry["data_offsets"]
        if not 0 <= start <= end <= payload_size:
            raise ValueError(f"{path.name}: {name} payload falls outside the shard")
    return entries


def _validate_local_shard(shard: str, specs: list[WeightSpec], path: Path) -> None:
    _verify_assigned_tensors(shard, specs, _local_header(path))


def fetch_checkpoint(
    checkpoint_dir: str | Path,
    *,
    repo: str = ECO_REPO,
    revision: str = ECO_REVISION,
    log: Callable[[str], None] = print,
) -> Path:
    """Verify the pinned Eco-Tech checkpoint and publish the index after its shards."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / ".fetch.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _fetch_checkpoint(checkpoint_dir, repo, revision, log)
    return checkpoint_dir


def _fetch_checkpoint(checkpoint_dir: Path, repo: str, revision: str, log: Callable[[str], None]) -> None:
    manifest = _modelscope_manifest(repo, revision)
    missing_manifest = [name for name in ECO_METADATA_FILES if name not in manifest]
    if missing_manifest:
        raise RuntimeError(f"ModelScope manifest is missing required files: {missing_manifest}")

    index_path = checkpoint_dir / ECO_INDEX
    index_size = int(manifest[ECO_INDEX]["Size"])
    index_digest = str(manifest[ECO_INDEX]["Sha256"])
    cached_index = _verified_locally(index_path, index_size, index_digest)
    index_data = index_path.read_bytes() if cached_index else _get(_modelscope_url(repo, revision, ECO_INDEX))
    if len(index_data) != index_size or hashlib.sha256(index_data).hexdigest() != index_digest:
        raise ValueError("checkpoint index does not match the manifest")
    index = _weight_map(json.loads(index_data))
    contract = QWEN3_8_27B_TEXT_CONTRACT
    text_shards = _text_shards(index, contract)
    metadata = tuple(name for name in ECO_METADATA_FILES if name != ECO_INDEX)
    required = metadata + tuple(sorted(set(index.values())))
    missing_manifest = [name for name in required if name not in manifest]
    if missing_manifest:
        raise RuntimeError(f"ModelScope manifest is missing required files: {missing_manifest}")

    present = {
        name: _verified_locally(checkpoint_dir / name, int(manifest[name]["Size"]), str(manifest[name]["Sha256"]))
        for name in required
    }
    remaining = sum(int(manifest[name]["Size"]) for name in required if not present[name])
    if not cached_index:
        remaining += index_size
    free = shutil.disk_usage(checkpoint_dir).free
    if remaining > free:
        raise OSError(f"checkpoint needs {remaining / 2**30:.2f} GiB but only {free / 2**30:.2f} GiB is free")

    # Shards are replaced in place, so from here until the index is published the
    # directory can hold bytes from two revisions. The stamp says so; a fetch that
    # dies in between leaves it saying so, and the loader refuses the directory.
    _write_revision_stamp(checkpoint_dir, f"{_STAMP_FETCHING} {revision}")

    for position, name in enumerate(required, 1):
        info = manifest[name]
        destination = checkpoint_dir / name
        expected_size = int(info["Size"])
        specs = text_shards.get(name)
        validate = partial(_validate_local_shard, name, specs) if specs is not None else None
        if present[name]:
            if validate is not None:
                validate(destination)
            log(f"[weights] [{position}/{len(required)}] present and verified: {name}")
            continue
        log(f"[weights] [{position}/{len(required)}] downloading {name} ({expected_size / 2**30:.2f} GiB)")
        _download_file(
            _modelscope_url(repo, revision, name),
            destination,
            expected_size,
            str(info["Sha256"]),
            validate,
        )
    if not cached_index:
        _write_verified_file(io.BytesIO(index_data), index_path, index_size, index_digest)
    _write_revision_stamp(checkpoint_dir, revision)


def _print_checkpoint_summary(checkpoint_dir: Path) -> None:
    summary = checkpoint_summary(checkpoint_dir)
    memory = QWEN3_8_27B_TEXT_CONTRACT.memory
    print(
        f"[weights] {summary['text_tensors']} raw text tensors in {summary['shards']} shards; "
        f"{summary['prepared_tensors']} tensors after zero offsets are dropped"
    )
    print(
        f"[weights] checkpoint text payload {summary['checkpoint_weight_bytes'] / 2**30:.2f} GiB; "
        f"resident weight payload {summary['resident_weight_bytes'] / 2**30:.2f} GiB"
    )
    print(
        f"[weights] native-context KV {memory.max_kv_cache_bytes / 2**30:.2f} GiB; "
        f"GDN state {memory.gdn_state_bytes_per_sequence / 2**20:.0f} MiB; "
        f"convolution state {memory.conv_state_bytes_per_sequence / 2**20:.2f} MiB; "
        f"64-GiB headroom {memory.max_context_headroom_bytes / 2**30:.2f} GiB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Accepted and ignored: this is host-side, but CI invokes runnable model
    # files with one uniform platform/device command line.
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3",
                        help="accepted for CI's uniform invocation; nothing here runs on device")
    parser.add_argument("--sim", dest="execution_mode", action="store_true",
                        help="accepted for CI's uniform invocation; unused")
    parser.add_argument("-d", "--device", type=int, default=0, help="accepted for CI's uniform invocation; unused")
    checkpoint_mode = parser.add_mutually_exclusive_group()
    checkpoint_mode.add_argument("--fetch-checkpoint", action="store_true",
                                 help="download the pinned ~30 GiB Eco-Tech W8A8 checkpoint")
    checkpoint_mode.add_argument("--validate-checkpoint", action="store_true",
                                 help="validate an already downloaded checkpoint index and shard set")
    checkpoint_mode.add_argument("--verify-index", action="store_true",
                                 help="check the contract against the published checkpoint's index and "
                                 "safetensors headers; needs the network, downloads no weights")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Eco-Tech checkpoint directory (default under build_output/qwen3_8_27b)")
    parser.add_argument("--checkpoint-revision", type=str, default=ECO_REVISION)

    # Legacy one-layer options retained for Q0-Q4 validation.
    parser.add_argument("--fetch", action="store_true", help="download one official BF16 GDN layer if absent")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--mixer", type=str, default="linear_attn", choices=FETCH_GROUPS,
                        help="which group to fetch: a mixer, the MLP, the two layer norms, or `text` -- "
                        "the embedding, the final norm and the LM head, which belong to no layer")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--revision", type=str, default=BF16_REVISION)
    parser.add_argument("--quantize", action="store_true", help="write the legacy layer's Q4 W8A8 form")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else default_checkpoint_path()
    if args.fetch_checkpoint:
        fetch_checkpoint(checkpoint_dir, revision=args.checkpoint_revision)
        _print_checkpoint_summary(checkpoint_dir)
        return 0
    if args.validate_checkpoint:
        _print_checkpoint_summary(checkpoint_dir)
        return 0
    if args.verify_index:
        verify_published_contract(revision=args.checkpoint_revision)
        return 0

    if args.quantize and args.mixer != "linear_attn":
        parser.error("--quantize writes the legacy GDN layer only")
    if args.mixer == TEXT_GROUP:
        out = Path(args.out) if args.out else text_path()
        if out.is_file():
            print(f"[weights] the non-layer tensors are already on disk ({out})")
            return 0
        if not args.fetch:
            print(f"[weights] {out} is not on disk; pass --fetch to download it from {BF16_REPO}")
            return 0
        weights = fetch_text(revision=args.revision)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weights, out)
        print(f"[weights] saved to {out}")
        return 0

    # Every layer has an MLP and two norms; the two mixers are on the layers their type says.
    if args.mixer not in ("mlp", "norms"):
        expected = "linear_attention" if args.mixer == "linear_attn" else "full_attention"
        actual = QWEN3_8_27B_TEXT_CONTRACT.layer_type(args.layer)
        if actual != expected:
            raise SystemExit(f"[weights] layer {args.layer} is {actual}, which has no {args.mixer}")
    source = default_path(args.layer, mixer=args.mixer)
    if not source.is_file() and not args.fetch:
        print(
            f"[weights] layer {args.layer} is not on disk ({source}); "
            f"pass --fetch to download it from {BF16_REPO}"
        )
        return 0

    if args.quantize:
        if not source.is_file():
            weights = fetch_layer(args.layer, revision=args.revision)
            source.parent.mkdir(parents=True, exist_ok=True)
            torch.save(weights, source)
        weights = reference.quantize_weights(load_layer(source))
        out = Path(args.out) if args.out else default_path(args.layer, quantized=True)
    elif source.is_file():
        print(f"[weights] layer {args.layer} already on disk ({source})")
        weights = load_layer(source)
        out = Path(args.out) if args.out else source
    else:
        weights = fetch_layer(args.layer, revision=args.revision, mixer=args.mixer)
        out = Path(args.out) if args.out else source
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out)
    for name, tensor in weights.items():
        print(f"  {name:<24} {str(tuple(tensor.shape)):<16} {tensor.dtype}")
    print(f"[weights] saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
