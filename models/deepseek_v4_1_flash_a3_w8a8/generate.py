# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real-checkpoint greedy generation driver for the compiled A3 prefill/decode entries.

Host responsibilities only: tokenize the fixed prompt, stream checkpoint leaves
through :class:`WeightLoader` into rank-owned device buffers, build input/state
tensors the complete entries take, compile each entry once, dispatch the
compiled host entry for every forward, and sample greedily on the host.  No
model arithmetic runs in Torch here; the forward is the compiled ``prefill_fwd``
/ ``decode_fwd`` entry.

Rank mapping uses the import-time --tp/--ep configuration: WORLD=EP and
DP=EP//TP, with contiguous tensor-parallel groups. Rank r owns tp_rank=r%TP,
dp_rank=r//TP and the whole-expert EP shard beginning at r*(384//EP).
One request is replicated across DP groups. Dense TP shards repeat across DP;
routed experts remain whole-expert EP shards, independent of TP. Checkpoint
leaves upload serially without retaining aggregate host weight stacks.

Weight contract: the checkpoint is loaded as-is (pre-rotated global/query basis
and pre-norm-folded); the loader only folds ``w = (q - offset) * scale``,
transposes ``[N, K]`` to ``[K, N]``, packs NZ, slices configured TP vocab/head/heads and
pads the HC functions to 32 rows.  The host Engram tables stay out of device
memory; only the gathered per-token rows enter ``engram_lookup``.

State and input contract:

* Prefill consumes the prompt tile plus ``initial_pre`` (identity mix),
  positions/request ids, paged window and compressed metadata, RoPE tables,
  the host Engram rows, the paged KV/index/compressor caches and the selection
  slots; it returns logits plus the updated caches and selection state.
* Decode consumes and mutates the same paged ``window_cache`` /
  ``compressed_cache_*`` / ``index_cache_*`` / ``state_cache_c2a`` buffers and
  the combined ``topk_*`` / ``pool_*`` selection slots plus ``position_ids``.
  A generation step can only continue when those buffers were seeded from the
  completed prefill.

The first generation path explicitly disables Engram. Both query RoPE profiles
and all per-step paging metadata are bound; prefill hands persistent caches to
a capacity1 decode artifact. Original Engram hashing is deferred. Real-weight
allocation remains gated by the resource budget and complete-entry validation.

Usage::

    python models/deepseek_v4_1_flash_a3_w8a8/generate.py --dry-run
    python models/deepseek_v4_1_flash_a3_w8a8/generate.py --weight-check
    python models/deepseek_v4_1_flash_a3_w8a8/generate.py --contract-report
    python models/deepseek_v4_1_flash_a3_w8a8/generate.py --submit --compile-only
    python models/deepseek_v4_1_flash_a3_w8a8/generate.py --submit --device-run

Exit codes: 0 executed checks passed; 1 failure; 3 stopped on a reported
entry-contract gap (blocked, not a generation result).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os

# The generation driver binds the real per-layer expert shards, so it runs the
# fwd modules with the layer-axis expert ABI (set before those modules are
# imported).  Bounded validation processes leave this unset.
os.environ.setdefault("DEEPSEEK_PER_LAYER_EXPERTS", "1")
import re
import shlex
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

from utils import compressed_source_positions

from config import BLOCK_SIZE, DP_SIZE, EP_SIZE, FLASH as M, N_LOCAL_EXPERTS, TP_SIZE, WORLD_SIZE, parallel_topology
from tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, encode_messages, load_tokenizer
from weight_loader import WeightLoader


# Paths and defaults.
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "dsv41-flash-w8a8-a3"
LOG_DIR = EXPERIMENT_DIR / "logs"
CHECKPOINT = Path("/mnt/data/models/DeepSeek-V4.1-Flash-w8a8")
DEFAULT_PROMPT = "Who are you?"
EOS_ID = 1
PAD_ID = M.engram_pad_token_id

# Model shapes used by the driver-side input construction.
N_LAYERS = M.num_hidden_layers
D = M.hidden_size
HC_MULT = M.hc_mult
HEAD_DIM = M.head_dim
RAW_ROWS = M.sliding_window
ENGRAM_NGRAM = M.engram_max_ngram_size - 1
ENGRAM_COLS = ENGRAM_NGRAM * M.engram_n_heads
ENGRAM_K = ENGRAM_COLS * M.engram_head_dim
LOCAL_INDEX_DIM = M.index_n_heads * M.index_head_dim
LOCAL_VOCAB = M.vocab_size // TP_SIZE

# Cache-owner layers mirroring ``prefill_layer`` / ``decode_layer``.
C2A_SOURCES = (2, 8, 14)
C1A_SOURCE = 20
REINDEX_LAYERS = (24, 28, 32, 36)
KV_SLOTS = C2A_SOURCES + (C1A_SOURCE,)
IDX_SLOTS = C2A_SOURCES + (C1A_SOURCE,) + REINDEX_LAYERS

# Entry weight-argument families (the complete prefill entry's own contract).
DENSE_STACK_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_w", "sink", "wo_a", "wo_b", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    "ffn_norm_w", "gate_weight", "gate_bias",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
)
# WeightLoader returns the checkpoint's ``gate_w`` / ``gate_b`` names; the
# complete entries declare ``gate_weight`` / ``gate_bias``.
DENSE_SOURCE_ALIASES = {"gate_weight": "gate_w", "gate_bias": "gate_b"}
EXPERT_STACK_NAMES = ("w1", "w1_scale", "w3", "w3_scale", "w2", "w2_scale")
C2A_SOURCE_NAMES = (
    "comp_wkv", "comp_wgate", "comp_norm", "idx_wq_b", "idx_wq_b_scale",
    "idx_weights_proj", "idx_wk", "idx_k_norm",
)
C1A_SOURCE_NAMES = (
    "comp_wkv", "comp_norm", "idx_wq_b", "idx_wq_b_scale",
    "idx_weights_proj", "idx_wk", "idx_k_norm",
)
REINDEX_SOURCE_NAMES = ("idx_wq_b", "idx_wq_b_scale", "idx_weights_proj")
TP_SLICED_DENSE = ("wq_b", "wq_b_scale", "sink", "wo_a", "wo_b")

_TORCH_DTYPES = {
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bool": torch.bool,
}

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 3


# ---------------------------------------------------------------------------
# Logging and small host utilities
# ---------------------------------------------------------------------------


class _Tee:
    """Mirror writes to the console and one log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(path):
    """Mirror stdout/stderr into ``path``; always close the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        prior_out, prior_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(prior_out, handle)
        sys.stderr = _Tee(prior_err, handle)
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = prior_out, prior_err


def _log(topic, message):
    print(f"[GENERATE][{topic}] {message}", flush=True)


class ContractGap(RuntimeError):
    """A complete-entry interface gap that blocks a real generation step."""

    def __init__(self, topic, detail):
        super().__init__(f"{topic}: {detail}")
        self.topic = topic
        self.detail = detail


class SamplingDisagreement(RuntimeError):
    """A requested DP replica agreement check failed with structured votes."""

    def __init__(self, evidence):
        super().__init__("DP replicas disagree on the greedy token")
        self.evidence = evidence


def _sha1(tensor):
    array = tensor.detach().to(torch.float32).contiguous().numpy()
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


def _parse_int_list(text):
    if isinstance(text, (list, tuple)):
        return [int(item) for item in text]
    return [int(item) for item in str(text).split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_ENCODING_MODULES = {}


def load_shipped_encoding(model_root):
    """Import the checkpoint's authoritative ``encoding/encoding.py`` (stdlib only)."""
    path = Path(model_root) / "encoding" / "encoding.py"
    if not path.is_file():
        return None
    key = str(path.resolve())
    if key not in _ENCODING_MODULES:
        spec = importlib.util.spec_from_file_location("dsv41_shipped_encoding", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ENCODING_MODULES[key] = module
    return _ENCODING_MODULES[key]


def build_prompt(tokenizer, prompt, template, messages_json=None, prompt_file=None,
                 model_root=None, thinking_mode=None, reasoning_effort=None, case_file=None,
                 case_index=0):
    """Tokenize the fixed prompt and record the exact ids and rendered template.

    ``chat``/``encoding`` render through the checkpoint's shipped
    ``encoding/encoding.py`` when present (DSML tags with a leading space,
    multi-turn, tools, thinking mode, images).  ``chat`` keeps the simple
    single-user-turn fixture; ``encoding`` accepts the full message schema.
    ``case_file`` renders one case from the shipped ``encoding/tests`` format.
    """
    if isinstance(reasoning_effort, str) and reasoning_effort.isdigit():
        reasoning_effort = int(reasoning_effort)
    source = "tokenizer_subset"
    media_images = []
    if case_file:
        encoding = load_shipped_encoding(model_root)
        if encoding is None:
            raise ValueError(f"shipped encoding.py not found under {model_root}")
        cases = encoding.load_cases(case_file)
        if not cases:
            raise ValueError(f"{case_file}: no cases")
        if not 0 <= case_index < len(cases):
            raise ValueError(f"{case_file}: case index {case_index} outside [0, {len(cases)})")
        case = cases[case_index]
        rendered, media_images = encoding.encode_case(case, thinking_mode or "chat")
        source = "shipped_encoding_case"
        messages = case["messages"]
        thinking_mode = case.get("thinking_mode") or thinking_mode or "chat"
        reasoning_effort = case.get("reasoning_effort", reasoning_effort)
    else:
        if prompt_file:
            prompt = Path(prompt_file).read_text(encoding="utf-8")
        if messages_json:
            messages = json.loads(Path(messages_json).read_text(encoding="utf-8"))
        else:
            messages = [{"role": "user", "content": prompt}]
        if template == "plain":
            rendered = prompt
            source = "plain"
        elif template in ("chat", "encoding"):
            encoding = load_shipped_encoding(model_root)
            mode = thinking_mode or "chat"
            if encoding is not None:
                rendered = encoding.encode_messages(
                    messages, thinking_mode=mode, reasoning_effort=reasoning_effort
                )
                source = "shipped_encoding"
            elif template == "chat":
                rendered = encode_messages(messages)
                source = "tokenizer_subset"
            else:
                raise ValueError(f"shipped encoding.py not found under {model_root}")
        else:
            raise ValueError(f"unsupported template {template!r}")
    if media_images:
        raise ContractGap(
            "image-input",
            f"the prompt carries {len(media_images)} image(s); the text-only first phase "
            "cannot execute the vision tower/aligner",
        )
    ids = tokenizer.encode(rendered)
    if not ids:
        raise ValueError("prompt encoded to no token ids")
    decoded = tokenizer.decode(ids)
    if decoded != rendered:
        raise ValueError("prompt ids do not round-trip through the tokenizer")
    record = {
        "template": template,
        "source": source,
        "thinking_mode": thinking_mode,
        "reasoning_effort": reasoning_effort,
        "messages": messages,
        "rendered": rendered,
        "ids": ids,
        "length": len(ids),
        "bos_id": tokenizer.encode_added_token(BOS_TOKEN),
        "eos_id": tokenizer.encode_added_token(EOS_TOKEN),
        "pad_id": tokenizer.encode_added_token(PAD_TOKEN),
        "decode_roundtrip": decoded == rendered,
        "images": len(media_images),
    }
    _log("prompt", f"source={source} template={template} length={len(ids)} ids={ids}")
    return record


def check_prompt_fixtures(tokenizer, model_root):
    """Reproduce the checkpoint's five shipped prompt-format pairs exactly."""
    encoding = load_shipped_encoding(model_root)
    root = Path(model_root) / "encoding"
    if encoding is None:
        raise ValueError(f"shipped encoding.py not found under {model_root}")
    records = []
    errors = []
    for index in range(1, 6):
        input_file = root / "tests" / f"test_input_{index}.json"
        output_file = root / "tests" / f"test_output_{index}.txt"
        if not input_file.is_file() or not output_file.is_file():
            errors.append(f"case {index}: shipped fixture missing")
            continue
        cases = encoding.load_cases(str(input_file))
        prompts = [encoding.encode_case(case, "chat")[0] for case in cases]
        actual = "".join(prompts)
        expected = output_file.read_text(encoding="utf-8")
        ids = tokenizer.encode(actual)
        roundtrip = tokenizer.decode(ids) == actual
        record = {
            "case": index,
            "cases": len(prompts),
            "chars": len(actual),
            "tokens": len(ids),
            "prompt_match": actual == expected,
            "tokenizer_roundtrip": roundtrip,
        }
        records.append(record)
        if not record["prompt_match"]:
            errors.append(f"case {index}: rendered prompt differs from test_output_{index}.txt")
        if not roundtrip:
            errors.append(f"case {index}: shipped prompt does not round-trip through the tokenizer")
        _log(
            "prompt-fixture",
            f"case={index} cases={record['cases']} chars={record['chars']} tokens={record['tokens']} "
            f"match={record['prompt_match']} roundtrip={roundtrip}",
        )
    if errors:
        for error in errors:
            _log("prompt-fixture-FAIL", error)
        raise ValueError(f"{len(errors)} shipped prompt fixture failure(s)")
    _log("prompt-fixture", f"PASS: {len(records)} shipped encoding pairs reproduced")
    return records


# ---------------------------------------------------------------------------
# Rank mapping
# ---------------------------------------------------------------------------


def rank_mapping():
    """Return the configured DP/TP/EP rank table used by both complete entries."""
    table = []
    for rank in range(WORLD_SIZE):
        table.append({
            "rank": rank,
            "dp_rank": rank // TP_SIZE,
            "tp_rank": rank % TP_SIZE,
            "group_base": (rank // TP_SIZE) * TP_SIZE,
            "ep_rank": rank,
            "expert_begin": rank * N_LOCAL_EXPERTS,
            "expert_end": (rank + 1) * N_LOCAL_EXPERTS,
        })
    return table


def print_rank_mapping():
    _log("mapping", "world=%d = DP%d x TP%d, EP%d" % (WORLD_SIZE, WORLD_SIZE // TP_SIZE, TP_SIZE, EP_SIZE))
    for row in rank_mapping():
        _log(
            "mapping",
            "rank=%2d dp=%d tp=%d group_base=%2d ep=%2d experts=[%d,%d)"
            % (row["rank"], row["dp_rank"], row["tp_rank"], row["group_base"],
               row["ep_rank"], row["expert_begin"], row["expert_end"]),
        )


# ---------------------------------------------------------------------------
# Entry reflection: parameter names, shapes, dtypes
# ---------------------------------------------------------------------------


@dataclass
class EntryParam:
    kind: str          # "tensor" or "scalar"
    name: str
    dtype: torch.dtype
    shape: list = field(default_factory=list)


def _resolve_dim(dim, dimensions=None):
    if isinstance(dim, int):
        return dim
    name = getattr(dim, "name", str(dim))
    if dimensions is not None and name in dimensions:
        return dimensions[name]
    if "LAYERS" in str(name):
        return N_LAYERS
    raise ValueError(f"cannot resolve dynamic entry dimension {dim!r}")


def _torch_dtype(dtype):
    name = str(dtype)
    if name not in _TORCH_DTYPES:
        raise ValueError(f"unsupported entry dtype {name!r}")
    return _TORCH_DTYPES[name]


def entry_parameters(entry, dimensions=None):
    """Read the complete entry's parameter contract from its annotations."""
    fn = getattr(entry, "_func", None)
    if fn is None:
        raise ValueError("entry does not expose its annotated function")
    params = []
    for name, parameter in inspect.signature(fn).parameters.items():
        annotation = parameter.annotation
        kind = type(annotation).__name__
        if kind == "Tensor":
            shape = [_resolve_dim(dim, dimensions) for dim in annotation.shape]
            params.append(EntryParam("tensor", name, _torch_dtype(annotation.dtype), shape))
        elif kind == "Scalar":
            params.append(EntryParam("scalar", name, _torch_dtype(annotation.dtype)))
        else:
            raise ValueError(f"unsupported annotation for {name!r}: {annotation!r}")
    return params


def parameter_map(entry):
    return {param.name: param for param in entry_parameters(entry)}


def build_specs(entry, scalar_values=None):
    """Build harness specs in entry parameter order (no tensor allocation).

    Static weight specs describe leading-rank residency. PreparedDispatch owns
    the actual upload, retained handles and free operations for generation.
    """
    from golden import ScalarSpec, TensorSpec

    scalar_values = scalar_values or {}
    specs = []
    for param in entry_parameters(entry):
        if param.kind == "scalar":
            if param.name not in scalar_values:
                raise ValueError(f"missing scalar value for {param.name!r}")
            specs.append(ScalarSpec(param.name, param.dtype, scalar_values[param.name]))
        else:
            resident = "stacked" if param.name in static_weight_names() else None
            specs.append(TensorSpec(param.name, list(param.shape), param.dtype,
                                    init_value=None, resident=resident))
    return specs


def scalar_dispatch_value(value, dtype):
    """Marshal one declared scalar as a public TaskArgs Python value."""
    scalar = torch.as_tensor(value, dtype=dtype)
    if scalar.numel() != 1 or scalar.ndim != 0:
        raise ValueError("scalar arguments must be zero-dimensional values")
    return scalar.item()


def ordered_dispatch_args(entry, values, contract=None, *, overrides=None, skip_names=()):
    """Return dispatch arguments in signature order with Python scalar values."""
    args = []
    params = entry_parameters(entry) if contract is None else [contract[name] for name in entry.param_names]
    overrides = overrides or {}
    for param in params:
        if param.name in skip_names:
            continue
        if param.name not in values and param.name not in overrides:
            raise ValueError(f"missing dispatch value for {param.name!r}")
        value = overrides[param.name] if param.name in overrides else values[param.name]
        if param.kind == "scalar":
            args.append(scalar_dispatch_value(value, param.dtype))
            continue
        if list(value.shape) != list(param.shape):
            raise ValueError(f"{param.name}: dispatch shape {list(value.shape)} != entry {list(param.shape)}")
        if value.dtype != param.dtype:
            raise ValueError(f"{param.name}: dispatch dtype {value.dtype} != entry {param.dtype}")
        args.append(value)
    return args


def load_entries(tokens):
    """Import both complete entries; build prefill for the requested token tile."""
    import decode_fwd
    import prefill_fwd

    prefill_entry = prefill_fwd._prefill_entry(tokens)
    return prefill_fwd, prefill_entry, decode_fwd


def spec_contract(specs):
    """Name -> concrete :class:`EntryParam` map from a module's ``build_fwd_specs``."""
    contract = {}
    for spec in specs:
        scalar = not hasattr(spec, "shape")
        contract[spec.name] = EntryParam("scalar" if scalar else "tensor", spec.name, spec.dtype,
                                        [] if scalar else list(spec.shape))
    return contract


def _validate_fixture_abi(label, entry, specs):
    declared = list(entry.param_names)
    supplied = [spec.name for spec in specs]
    missing = sorted(set(declared) - set(supplied))
    extra = sorted(set(supplied) - set(declared))
    if missing or extra:
        raise ContractGap(
            "entry-spec-abi",
            f"{label} fixture specs and entry parameters differ: missing={missing} extra={extra}",
        )


def build_entry_contracts(tokens, decode_tokens=1, *, pages=2, batch=1):
    """Resolve both complete ABIs from annotations without fixture allocation."""
    from config import CANDIDATE_POOL_BLOCKS
    from golden import ScalarSpec, TensorSpec
    import decode_fwd
    import prefill_fwd

    if any(type(value) is not int or value < 1 for value in (tokens, decode_tokens, pages, batch)):
        raise ValueError("entry capacities must be positive integers")
    dimensions = {
        "V41_A3_B_DYN": batch, "V41_A3_TABLE_DYN": pages,
        "V41_A3_ORI_BLOCKS_DYN": batch * pages,
        "V41_A3_CMP_BLOCKS_DYN": batch * pages,
        "V41_A3_INDEX_BLOCKS_DYN": batch * pages,
        "V41_A3_STATE_BLOCKS_DYN": batch,
        "V41_A3_POOL_BLOCKS_DYN": CANDIDATE_POOL_BLOCKS,
    }
    contracts, all_specs = [], []
    for entry, rows in ((prefill_fwd._prefill_entry(tokens), tokens), (decode_fwd.decode_fwd, decode_tokens)):
        params = entry_parameters(entry, {**dimensions, "V41_A3_T_DYN": rows})
        contracts.append({param.name: param for param in params})
        specs = []
        for param in params:
            if param.kind == "scalar":
                specs.append(ScalarSpec(param.name, param.dtype, 0, compile_runtime=True))
            else:
                specs.append(TensorSpec(param.name, list(param.shape), param.dtype, init_value=None,
                                        resident="stacked" if param.name in static_weight_names() else None))
        all_specs.append(specs)
    return contracts[0], contracts[1], all_specs[0], all_specs[1]


def entry_ordered_specs(entry, specs):
    """Reorder fixture specs into the entry's declared parameter order.

    Static weights carry leading-rank residency metadata. Actual generation
    uploads and retains them through the PreparedDispatch owner.
    """
    by_name = {spec.name: spec for spec in specs}
    ordered = [by_name[name] for name in entry.param_names]
    for spec in ordered:
        if spec.name in static_weight_names():
            spec.resident = "stacked"
    return ordered


# ---------------------------------------------------------------------------
# Weight materialization
# ---------------------------------------------------------------------------


@dataclass
class RankWeights:
    rank: int
    tp_rank: int
    layers: list
    dense: dict
    experts: dict
    top: dict
    sources: dict
    engram_tables: dict


def materialize_rank(loader, layers, with_experts=True, with_engram=True):
    """Materialize one rank's ordered weight arguments for the requested layers."""
    layer_dicts = [loader.layer_dense(layer, with_engram=with_engram) for layer in layers]
    dense = {}
    for name in DENSE_STACK_NAMES:
        source = DENSE_SOURCE_ALIASES.get(name, name)
        if source in layer_dicts[0]:
            dense[name] = torch.stack([values[source] for values in layer_dicts])
    experts = {}
    if with_experts:
        shards = [loader.expert_shard(layer) for layer in layers]
        experts = {name: torch.stack([shard[name] for shard in shards]) for name in EXPERT_STACK_NAMES}
    sources = {}
    dense_by_layer = dict(zip(layers, layer_dicts))
    for prefix, source_layers, names in (
        ("c2a_", C2A_SOURCES, C2A_SOURCE_NAMES),
        ("c1a_", (C1A_SOURCE,), C1A_SOURCE_NAMES),
        ("re_", REINDEX_LAYERS, REINDEX_SOURCE_NAMES),
    ):
        if not set(source_layers) <= set(layers):
            continue
        for name in names:
            sources[prefix + name] = torch.stack([dense_by_layer[layer][name] for layer in source_layers])
    return RankWeights(
        rank=loader.rank,
        tp_rank=loader.tp_rank,
        layers=list(layers),
        dense=dense,
        experts=experts,
        top=loader.top_level_arguments(),
        sources=sources,
        engram_tables={layer: loader.engram_table(layer) for layer in M.engram_layer_ids} if with_engram else {},
    )


def materialize_all_ranks(checkpoint, allow_full=False, with_engram=True):
    """Materialize the complete configured-rank argument set (host memory heavy)."""
    estimate = WORLD_SIZE * (544179486720 // EP_SIZE + 10370426880 // TP_SIZE + 3020991936) / (1 << 30)
    if not allow_full:
        raise ContractGap(
            "full-materialize",
            f"materializing all {WORLD_SIZE} ranks needs about {estimate:.0f} GiB host memory; "
            "pass --allow-full-materialize to opt in",
        )
    rank_args = {}
    expert_by_rank = {}
    for rank in range(WORLD_SIZE):
        loader = WeightLoader(checkpoint, rank=rank)
        weights = materialize_rank(loader, list(range(N_LAYERS)), with_experts=True, with_engram=with_engram)
        rank_args[rank] = {**weights.dense, **weights.sources, **weights.top}
        expert_by_rank[rank] = weights.experts
        _log("weight", f"rank={rank} tp={loader.tp_rank} complete weights materialized")
        del loader, weights
    return rank_args, expert_by_rank


def materialize_dense_variants(checkpoint, with_engram=True):
    """Materialize the configured TP dense/source/top-level variants (no expert shards)."""
    variants = {}
    for tp_rank in range(TP_SIZE):
        loader = WeightLoader(checkpoint, rank=tp_rank)
        weights = materialize_rank(loader, list(range(N_LAYERS)), with_experts=False, with_engram=with_engram)
        variants[tp_rank] = {**weights.dense, **weights.sources, **weights.top}
        _log("weight", f"tp_rank={tp_rank} dense/source/top-level weights materialized")
        del loader, weights
    return variants


# ---------------------------------------------------------------------------
# RoPE tables (base profile and compressed/YaRN profile)
# ---------------------------------------------------------------------------


ROPE_YARN = {
    "rope_type": "yarn",
    "factor": M.rope_factor,
    "beta_fast": M.beta_fast,
    "beta_slow": M.beta_slow,
    "original_max_position_embeddings": M.original_max_position_embeddings,
}


def rope_profile_plan():
    """Return the per-layer RoPE profile from the checkpoint config (ground truth).

    ``config.json`` ``text_config.rope_scaling`` applies YaRN (factor 16,
    beta_fast 32, beta_slow 1, original 65536) to the model rotary embedding.
    Layers 0-1 use unscaled SWA ``rope_theta``; compressed layers use YaRN
    with ``compress_rope_theta``, matching the pinned original forward.
    """
    plan = []
    for layer in range(N_LAYERS):
        compressed = layer >= 2
        plan.append({
            "layer": layer,
            "profile": "compressed" if compressed else "base",
            "theta": M.compress_rope_theta if compressed else M.rope_theta,
            "yarn": ROPE_YARN if compressed else None,
        })
    return plan


def rope_profile_name(layer):
    return "compressed" if layer >= 2 else "base"


def resolve_rope_profile(contract, requested):
    """Resolve the host table layout against the entry's rope argument shape.

    Returns ``(mode, mismatched_layers)`` where mode is ``base``,
    ``compressed`` or ``per-layer``.  A single-table entry can only carry one
    profile; ``auto`` then selects the profile covering 38/40 layers and reports
    the two SWA layers as a contract gap.
    """
    if "rope_cos_swa" in contract or "rope_cos_comp" in contract:
        if requested != "auto":
            raise ValueError("the complete model requires both RoPE profiles; use --rope-profile auto")
        return "per-profile", []
    if requested in ("base", "compressed"):
        return requested, ([0, 1] if requested == "compressed" else list(range(2, N_LAYERS)))
    param = contract.get("rope_cos")
    if param is not None and len(param.shape) >= 4 and param.shape[1] == N_LAYERS:
        return "per-layer", []
    return "compressed", [0, 1]


def layer_rope_tables(profile, positions):
    """Return the (cos, sin) tables for one profile name."""
    return gather_rope_tables(positions, compressed=(profile == "compressed"))


def precompute_rope_tables(rows, compressed):
    """Return unscaled SWA or YaRN compressed FP32 rotary tables."""
    dim = M.qk_rope_head_dim
    base = M.compress_rope_theta if compressed else M.rope_theta
    frequencies = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if not compressed:
        angles = torch.outer(torch.arange(rows, dtype=torch.float32), frequencies)
        return angles.cos(), angles.sin()
    numerator = M.original_max_position_embeddings

    def corrected(rotations):
        return dim * math.log(numerator / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(corrected(M.beta_fast)), 0)
    high = min(math.ceil(corrected(M.beta_slow)), dim - 1)
    ramp = torch.arange(dim // 2, dtype=torch.float32)
    ramp = ((ramp - low) / max(high - low, 1e-3)).clamp(0, 1)
    smooth = 1 - ramp
    frequencies = frequencies / M.rope_factor * (1 - smooth) + frequencies * smooth
    angles = torch.outer(torch.arange(rows, dtype=torch.float32), frequencies)
    return angles.cos(), angles.sin()


def gather_rope_tables(positions, compressed):
    """Gather per-position rows; negative positions get identity rotation."""
    rows = max(int(positions.max()) + 1, 1) if positions.numel() else 1
    cos, sin = precompute_rope_tables(rows, compressed)
    safe = positions.clamp_min(0)
    gathered_cos = cos[safe].clone()
    gathered_sin = sin[safe].clone()
    identity = positions < 0
    gathered_cos[identity] = 1.0
    gathered_sin[identity] = 0.0
    return gathered_cos, gathered_sin


def compressed_key_tables(index_rows, ratio, base_row):
    """RoPE rows for compressed index keys: buffer row j starts at ``ratio*(base_row+j)``."""
    positions = (torch.arange(index_rows, dtype=torch.int64) + base_row) * ratio
    cos, sin = precompute_rope_tables(int(positions.max()) + 1, True)
    return cos[positions], sin[positions]


# ---------------------------------------------------------------------------
# Input and state tensor construction
# ---------------------------------------------------------------------------


def _zeros(shape, dtype):
    return torch.zeros(shape, dtype=dtype)


def _rank_repeat(tensor):
    return tensor.unsqueeze(0).repeat((WORLD_SIZE,) + (1,) * tensor.dim()).contiguous()


def _repeat_first(rows, tensor):
    if rows != WORLD_SIZE:
        raise ValueError(f"expected {WORLD_SIZE} ranks, entry has {rows}")
    return _rank_repeat(tensor)


def _rope_args(contract, positions, rope_profile, name="rope_cos"):
    """Build one rope argument, honoring a per-layer entry layout when present."""
    param = contract[name]
    mode, _ = resolve_rope_profile(contract, rope_profile)
    if mode == "per-profile":
        mode = "base" if name.endswith("_swa") else "compressed"
    if len(param.shape) >= 4 and param.shape[1] == N_LAYERS:
        cos_rows = []
        for layer in range(N_LAYERS):
            cos, _sin = layer_rope_tables(rope_profile_name(layer), positions)
            cos_rows.append(cos)
        cos_stack = torch.stack(cos_rows)
        return _rank_repeat(cos_stack)
    cos, _sin = layer_rope_tables(mode, positions)
    return _rank_repeat(cos)


def _rope_args_pair(contract, positions, rope_profile, cos_name, sin_name):
    param = contract[cos_name]
    mode, _ = resolve_rope_profile(contract, rope_profile)
    if mode == "per-profile":
        mode = "base" if cos_name.endswith("_swa") else "compressed"
    if len(param.shape) >= 4 and param.shape[1] == N_LAYERS:
        cos_rows, sin_rows = [], []
        for layer in range(N_LAYERS):
            cos, sin = layer_rope_tables(rope_profile_name(layer), positions)
            cos_rows.append(cos)
            sin_rows.append(sin)
        return _rank_repeat(torch.stack(cos_rows)), _rank_repeat(torch.stack(sin_rows))
    cos, sin = layer_rope_tables(mode, positions)
    return _rank_repeat(cos), _rank_repeat(sin)


def query_rope_inputs(contract, positions, rope_profile="auto"):
    """Bind both query profiles required by the complete forward ABI."""
    values = {}
    for cos_name, sin_name in (("rope_cos", "rope_sin"), ("rope_cos_swa", "rope_sin_swa")):
        if cos_name in contract:
            values[cos_name], values[sin_name] = _rope_args_pair(
                contract, positions, rope_profile, cos_name, sin_name,
            )
    return values


def rope_metadata(contract, requested):
    mode, mismatched = resolve_rope_profile(contract, requested)
    return {
        "requested": requested,
        "entry_mode": mode,
        "per_layer": rope_profile_plan(),
        "mismatched_layers": mismatched,
        "checkpoint_rope_scaling": ROPE_YARN,
        "note": {
            "per-profile": "entry has explicit SWA/base and compressed rope argument pairs",
            "per-layer": "entry has a per-layer rope table axis",
        }.get(mode, "single-table entry: the host cannot mix base (layers 0-1) and compressed (layers 2-39) profiles"),
    }


# Paged layout of the current complete entries.
PAGED_CACHE_NAMES = (
    "window_cache",
    "compressed_cache_c2a",
    "index_cache_c2a",
    "state_cache_c2a",
    "compressed_cache_c1a",
    "index_cache_c1a",
)
# Axis that a request plan's page/batch count fills for each paged buffer.
PAGED_CACHE_AXES = {
    "window_cache": 2,
    "compressed_cache_c2a": 2,
    "index_cache_c2a": 2,
    "state_cache_c2a": 2,
    "compressed_cache_c1a": 1,
    "index_cache_c1a": 1,
}
# The physical page ids of a prefill plan are a prefix of any longer plan that
# starts at the same position, so handoff may copy the used prefix only.
PAGED_PREFIX_DIMS = {
    "window_cache": (2,),
    "compressed_cache_c2a": (2,),
    "index_cache_c2a": (2,),
    "state_cache_c2a": (2,),
    "compressed_cache_c1a": (1,),
    "index_cache_c1a": (1,),
}


def _require_rank_axis(contract, label):
    rows = int(contract["input_ids"].shape[0])
    if rows != WORLD_SIZE:
        raise ContractGap(
            "entry-rank-axis",
            f"{label} entry expects {rows} rank rows but the A3 form is EP{WORLD_SIZE}",
        )


def _pad_2d(table, target_shape, label):
    """Pad a plan table with ``-1``/``0`` rows+columns up to the entry spec."""
    target = [int(dim) for dim in target_shape]
    if table.shape[0] > target[0] or table.shape[1] > target[1]:
        raise ContractGap(
            "paged-plan",
            f"{label}: request plan needs {list(table.shape)} but the entry spec is {target}",
        )
    fill = -1 if label == "index_block_table" else 0
    padded = torch.full(target, fill, dtype=table.dtype)
    padded[:table.shape[0], :table.shape[1]] = table
    return padded


def paged_plan(contract, start_position, length):
    """Build one request's paged metadata and check it fits the entry specs."""
    from utils import build_request_plan

    _require_rank_axis(contract, "paged-plan")
    plan1 = build_request_plan([(start_position, length)], ratio=1)
    plan2 = build_request_plan([(start_position, length)], ratio=2)
    index_table = _pad_2d(
        plan1["index_block_table"],
        contract["index_block_table"].shape[1:],
        "index_block_table",
    )
    state_table = _pad_2d(
        plan1["state_block_table"],
        contract["state_block_table"].shape[1:],
        "state_block_table",
    )
    for name, axis in PAGED_CACHE_AXES.items():
        if name == "state_cache_c2a":
            continue
        capacity = int(contract[name].shape[axis])
        if plan1["pages"] > capacity:
            raise ContractGap(
                "paged-plan",
                f"positions [{start_position}, {start_position + length}) need "
                f"{plan1['pages']} physical pages but the {name} entry spec provides {capacity}",
            )
    batch_capacity = int(contract["state_cache_c2a"].shape[2])
    if plan1["index_block_table"].shape[0] > int(contract["index_block_table"].shape[1]):
        raise ContractGap(
            "paged-plan",
            f"request batch {plan1['index_block_table'].shape[0]} exceeds the entry "
            f"index_block_table batch axis {contract['index_block_table'].shape[1]}",
        )
    if plan1["index_block_table"].shape[0] > batch_capacity:
        raise ContractGap(
            "paged-plan",
            f"request batch {plan1['index_block_table'].shape[0]} exceeds the entry "
            f"state_cache_c2a batch axis {batch_capacity}",
        )
    return {
        "plan1": plan1,
        "plan2": plan2,
        "index_block_table": index_table,
        "state_block_table": state_table,
    }


def _compressed_rope_rows(plan, ratio):
    """RoPE rows at the absolute source position of each emitted compressed key."""
    rows = compressed_source_positions(plan["position_ids"], ratio)
    count = int(rows.max()) + 2 if rows.numel() else 2
    cos, sin = precompute_rope_tables(count, True)
    return cos[rows], sin[rows]


def _plan_inputs(contract, plan):
    """Rank-replicated position/paging/RoPE arguments shared by both entries."""
    plan1, plan2 = plan["plan1"], plan["plan2"]
    positions = plan1["position_ids"]
    c2a_cos, c2a_sin = _compressed_rope_rows(plan2, ratio=2)
    c1a_cos, c1a_sin = _compressed_rope_rows(plan1, ratio=1)
    return {
        "position_ids": _rank_repeat(positions),
        "request_ids": _rank_repeat(plan1["request_ids"]),
        "window_slots": _rank_repeat(plan1["window_slots"]),
        "window_indices": _rank_repeat(plan1["window_indices"]),
        "index_block_table": _rank_repeat(plan["index_block_table"]),
        "state_block_table": _rank_repeat(plan["state_block_table"]),
        "c2a_compressed_slots": _rank_repeat(plan2["compressed_slots"]),
        "c2a_compressed_lens": _rank_repeat(plan2["compressed_lens"]),
        "c1a_compressed_slots": _rank_repeat(plan1["compressed_slots"]),
        "c1a_compressed_lens": _rank_repeat(plan1["compressed_lens"]),
        "c2a_compressed_cos": _rank_repeat(c2a_cos),
        "c2a_compressed_sin": _rank_repeat(c2a_sin),
        "c1a_compressed_cos": _rank_repeat(c1a_cos),
        "c1a_compressed_sin": _rank_repeat(c1a_sin),
    }


def _cache_and_output_buffers(contract, tokens, topk_names):
    """Zero KV/index/state caches, selection slots and output buffers."""
    values = {}
    for name in PAGED_CACHE_NAMES:
        values[name] = _zeros(contract[name].shape, contract[name].dtype)
    for name in topk_names:
        if name.startswith("topk_values") or name.startswith("pool_block_counts"):
            values[name] = _zeros(contract[name].shape, contract[name].dtype)
        else:
            values[name] = torch.full(contract[name].shape, -1, dtype=contract[name].dtype)
    values["token_counts"] = torch.full(
        (int(contract["token_counts"].shape[0]),), tokens, dtype=contract["token_counts"].dtype
    )
    values["x_hc"] = _zeros(contract["x_hc"].shape, contract["x_hc"].dtype)
    values["pre_mix"] = _zeros(contract["pre_mix"].shape, contract["pre_mix"].dtype)
    for name in ("layer_x_hc", "layer_pre_mix", "final_hidden", "logits"):
        if name in contract:
            values[name] = _zeros(contract[name].shape, contract[name].dtype)
    if "engram_lookup" in contract:
        values["engram_lookup"] = _zeros(
            contract["engram_lookup"].shape, contract["engram_lookup"].dtype
        )
    return values


def build_prefill_inputs(contract, prompt_record, start_position=0, rope_profile="auto"):
    """Build every non-weight prefill argument in the entry's concrete shapes."""
    tokens = int(contract["input_ids"].shape[1])
    prompt_ids = prompt_record["ids"]
    if len(prompt_ids) > tokens:
        raise ValueError(f"prompt length {len(prompt_ids)} exceeds the prefill tile {tokens}")
    padded = list(prompt_ids) + [PAD_ID] * (tokens - len(prompt_ids))
    ids = torch.tensor(padded, dtype=torch.int64)
    plan = paged_plan(contract, start_position, tokens)
    positions = plan["plan1"]["position_ids"]
    rope_cos, rope_sin = _rope_args_pair(contract, positions, rope_profile, "rope_cos", "rope_sin")

    initial_pre = torch.zeros(tokens, HC_MULT, dtype=torch.float32)
    initial_pre[:, 0] = 1.0

    values = {
        "input_ids": _rank_repeat(ids),
        "initial_pre": _rank_repeat(initial_pre),
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
    }
    values.update(_plan_inputs(contract, plan))
    values.update(query_rope_inputs(contract, positions, rope_profile))
    values.update(
        _cache_and_output_buffers(
            contract,
            tokens,
            (
                "topk_indices_c2a", "topk_values_c2a",
                "pool_block_ids_c2a", "pool_block_counts_c2a",
                "topk_indices_c1a", "topk_values_c1a",
                "pool_block_ids_c1a", "pool_block_counts_c1a",
            ),
        )
    )
    return values


def build_decode_state(contract, position, rope_profile="auto"):
    """Build a cold decode state with the entry's paged metadata and buffers."""
    tokens = int(contract["input_ids"].shape[1])
    if tokens != 1:
        raise ContractGap("decode-active", "single-request generation requires decode capacity1")
    plan = paged_plan(contract, position, tokens)
    positions = plan["plan1"]["position_ids"]
    rope_cos, rope_sin = _rope_args_pair(contract, positions, rope_profile, "rope_cos", "rope_sin")

    initial_pre = torch.zeros(tokens, HC_MULT, dtype=torch.float32)
    initial_pre[:, 0] = 1.0

    values = {
        "input_ids": torch.zeros(
            contract["input_ids"].shape, dtype=contract["input_ids"].dtype
        ),
        "initial_pre": _rank_repeat(initial_pre),
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
    }
    values.update(_plan_inputs(contract, plan))
    values.update(query_rope_inputs(contract, positions, rope_profile))
    values.update(
        _cache_and_output_buffers(
            contract,
            tokens,
            (
                "topk_indices", "topk_values",
                "pool_block_ids", "pool_block_counts",
            ),
        )
    )
    return values


def advance_decode_inputs(values, contract, token_id, position, rope_profile="auto"):
    """Advance one active request's metadata without resetting persistent caches."""
    if int(contract["input_ids"].shape[1]) != 1:
        raise ContractGap("decode-active", "one generated token requires capacity1")
    plan = paged_plan(contract, position, 1)
    values.update(_plan_inputs(contract, plan))
    values.update(query_rope_inputs(contract, plan["plan1"]["position_ids"], rope_profile))
    values["input_ids"].fill_(token_id)
    values["token_counts"].fill_(1)


def exact_prompt_tokens(prompt_record):
    """Return the unmodified prompt length for the bounded all-active path."""
    tokens = len(prompt_record["ids"])
    if not 2 <= tokens <= 128:
        raise ContractGap(
            "prefill-length",
            f"the first noEngram generation path supports 2..128 original prompt tokens; got {tokens}; "
            "no tokens were added or removed",
        )
    return tokens


def prefill_tile_gap(prompt_length, tile):
    """Return the padding gap for a prompt that is shorter than the prefill tile."""
    if prompt_length == tile:
        return None
    if prompt_length > tile:
        return f"prompt length {prompt_length} exceeds the prefill tile {tile}"
    return (
        f"prompt length {prompt_length} != prefill tile {tile}: padding tokens advance the "
        "carried compressor state, append pad KV rows and can emit pad compressed rows; "
        "use a prompt whose length equals the compiled tile for a real handoff"
    )


HANDOFF_SOURCES = {
    "window_cache": ("window_cache",),
    "compressed_cache_c2a": ("compressed_cache_c2a",),
    "index_cache_c2a": ("index_cache_c2a",),
    "state_cache_c2a": ("state_cache_c2a",),
    "compressed_cache_c1a": ("compressed_cache_c1a",),
    "index_cache_c1a": ("index_cache_c1a",),
}
HANDOFF_NAMES = set(HANDOFF_SOURCES)


def _copy_paged_prefix(name, source, target_param, used_pages=None):
    """Copy a prefill cache into the decode buffer, only the allocated prefix."""
    target_shape = [int(dim) for dim in target_param.shape]
    if list(source.shape) == target_shape:
        return source.contiguous()
    prefix_dims = set(PAGED_PREFIX_DIMS.get(name, ()))
    if source.dim() != len(target_shape):
        raise ContractGap(
            "decode-handoff",
            f"{name}: prefill shape {list(source.shape)} cannot map to decode {target_shape}",
        )
    for axis, (have, need) in enumerate(zip(source.shape, target_shape)):
        if have == need:
            continue
        truncation = used_pages is not None and axis == PAGED_CACHE_AXES[name] and used_pages <= need
        if axis not in prefix_dims or (need < have and not truncation):
            raise ContractGap(
                "decode-handoff",
                f"{name}: prefill shape {list(source.shape)} does not fit decode {target_shape} "
                f"(axis {axis} differs)",
            )
    region = tuple(slice(0, min(have, need)) for have, need in zip(source.shape, target_shape))
    buffer = torch.zeros(target_shape, dtype=target_param.dtype)
    buffer[region] = source[region]
    return buffer


def _copy_exact(name, source, target_param):
    if list(source.shape) != list(target_param.shape):
        raise ContractGap(
            "decode-handoff",
            f"{name}: prefill shape {list(source.shape)} != decode {list(target_param.shape)}",
        )
    return source.contiguous()


def handoff_to_decode(prefill_values, decode_contract, prompt_length, start_position=0,
                      tile=None, row_offset_c2a=0, row_offset_c1a=0, allow_padded=False):
    """Map the prefill entry's exported paged state into the decode entry.

    The prefill and decode plans number physical pages from zero, so the caches
    copy one-to-one; a shallower prefill plan is copied into the allocated
    prefix when the decode spec has more pages/batch rows. Query selection,
    hidden and mix buffers are rebuilt for the current decode token.
    """
    tile = prompt_length if tile is None else tile
    gap = prefill_tile_gap(prompt_length, tile)
    if gap and not allow_padded:
        raise ContractGap("prefill-tile", gap)
    if row_offset_c2a or row_offset_c1a:
        raise ContractGap(
            "handoff-offset",
            "both entries index paged rows by absolute position//ratio and take no base "
            "offset; a nonzero source row offset cannot be represented",
        )
    used_c2a = (start_position + prompt_length) // 2
    used_c1a = start_position + prompt_length
    used_rows = {"window_cache": start_position + prompt_length,
                 "compressed_cache_c2a": used_c2a, "index_cache_c2a": used_c2a,
                 "compressed_cache_c1a": used_c1a, "index_cache_c1a": used_c1a}
    for name, used in used_rows.items():
        axis = PAGED_CACHE_AXES[name]
        capacity = int(decode_contract[name].shape[axis]) * BLOCK_SIZE
        if used > capacity:
            raise ContractGap(
                "decode-context-cap",
                f"{name} rows used {used} exceed the {capacity} rows of the compiled paged "
                "decode cache; a longer context needs a larger page budget",
            )

        source_capacity = int(prefill_values[name].shape[axis]) * BLOCK_SIZE
        if used > source_capacity:
            raise ContractGap("prefill-context-cap", f"{name}: {used} used rows exceed source capacity {source_capacity}")

    mapping = {}
    for name in PAGED_CACHE_NAMES:
        if name in decode_contract and name in prefill_values:
            used_pages = (used_rows[name] + BLOCK_SIZE - 1) // BLOCK_SIZE if name in used_rows else None
            mapping[name] = _copy_paged_prefix(name, prefill_values[name], decode_contract[name], used_pages)
    for decode_name, sources in HANDOFF_SOURCES.items():
        if decode_name not in decode_contract or decode_name in mapping:
            continue
        if len(sources) > 1:
            if not all(source in prefill_values for source in sources):
                continue
            combined = torch.cat([prefill_values[source] for source in sources], dim=1)
            if list(combined.shape) != list(decode_contract[decode_name].shape):
                raise ContractGap(
                    "decode-handoff",
                    f"{decode_name}: combined prefill shape {list(combined.shape)} != decode "
                    f"{list(decode_contract[decode_name].shape)}",
                )
            mapping[decode_name] = combined.contiguous()
        elif sources[0] in prefill_values:
            mapping[decode_name] = _copy_exact(
                sources[0], prefill_values[sources[0]], decode_contract[decode_name]
            )
    missing = sorted(
        name for name in decode_contract
        if name in HANDOFF_NAMES and name not in mapping
    )
    if missing:
        raise ContractGap("decode-handoff", "prefill does not export " + ", ".join(missing))
    return mapping


# ---------------------------------------------------------------------------
# Engram host rows
# ---------------------------------------------------------------------------


def load_engram_hash_ids(path, layers, rows):
    """Load host-computed Engram n-gram hash ids ``{layer: [rows, 24]}``."""
    if path is None:
        raise ContractGap(
            "engram-hash",
            "the checkpoint ships the Engram tables but not the n-gram hash function. "
            "Options: (a) --engram-mode zero-diagnostic for a DIAGNOSTIC-ONLY run that zeros the "
            "host lookup rows (kv=0 => gate*value=0, so layers 1/14 contribute no Engram residual), "
            "or (b) supply a user hash spec with --engram-hash-ids (dict layer -> [rows, 24] int64)",
        )
    payload = torch.load(path, map_location="cpu")
    result = {}
    for layer in layers:
        key = str(layer)
        if key not in payload:
            raise ValueError(f"{path}: missing Engram hash ids for layer {layer}")
        ids = payload[key]
        if ids.dim() != 2 or ids.shape[1] != ENGRAM_COLS:
            raise ValueError(f"layer {layer}: hash ids must be [rows, {ENGRAM_COLS}], got {tuple(ids.shape)}")
        if ids.shape[0] < rows:
            raise ValueError(f"layer {layer}: hash ids cover {ids.shape[0]} rows, need {rows}")
        result[layer] = ids[:rows].long()
    return result


def build_engram_lookup(tables, hash_ids, start_row, tokens):
    """Gather host rows for every Engram layer and rank replica."""
    per_layer = []
    for layer in M.engram_layer_ids:
        table = tables[layer]
        ids = hash_ids[layer][start_row:start_row + tokens]
        if int(ids.max()) >= table.rows or int(ids.min()) < 0:
            raise ValueError(f"layer {layer}: Engram hash id outside [0, {table.rows})")
        per_layer.append(table.lookup(ids))
    lookup = torch.stack(per_layer).to(torch.bfloat16)
    return _rank_repeat(lookup)


def diagnostic_engram_lookup(contract):
    """Zero host lookup rows: kv=0 => gate*value=0, so Engram adds no residual.

    This is a DIAGNOSTIC-ONLY substitute for the missing n-gram hash function;
    it changes model semantics and must never be presented as a real result.
    """
    shape = contract["engram_lookup"].shape
    _log("engram", f"DIAGNOSTIC-ONLY zero Engram lookup rows: {shape}")
    return _zeros(shape, torch.bfloat16)


def materialize_engram_weights(checkpoint):
    """Materialize the two Engram projection weights and the folded gate product."""
    loader = WeightLoader(checkpoint, rank=0)
    wkv = []
    qk = []
    for layer in M.engram_layer_ids:
        dense = loader.layer_dense(layer)
        wkv.append(dense["engram_wkv"])
        qk.append(dense["engram_qk"])
    return torch.stack(wkv), torch.stack(qk)


# ---------------------------------------------------------------------------
# Weight argument assembly and contract comparison
# ---------------------------------------------------------------------------


def expert_layer_axis_gap(contract, label):
    """Return the routed-expert per-layer binding gap, if present.

    The complete entries take one contiguous per-rank expert shard with no
    layer axis, while the checkpoint ships a different EP16 shard per layer.
    """
    for name in EXPERT_STACK_NAMES:
        if name not in contract:
            continue
        shape = [int(dim) for dim in contract[name].shape]
        if len(shape) < 3 or shape[0] != WORLD_SIZE or shape[1] != N_LAYERS:
            return (
                f"{label} routed-expert argument {name} has shape {shape} with no "
                f"[EP{WORLD_SIZE}, {N_LAYERS}] rank/layer axis; the checkpoint ships a distinct "
                "EP16 expert shard per layer, so a device run cannot bind the 40 per-layer shards"
            )
    return None


def prefill_ep16_gap(contract):
    """Return the prefill routed-expert per-layer binding gap, if present."""
    return expert_layer_axis_gap(contract, "prefill")


def _assemble_rank_weights(contract, rank_args, expert_by_rank, label):
    """Stack per-rank dense/top-level/source/expert materializations into the entry ABI."""
    values = {}
    for name, param in contract.items():
        if param.kind != "tensor":
            continue
        target = [int(dim) for dim in param.shape]
        if name in EXPERT_STACK_NAMES:
            gap = expert_layer_axis_gap(contract, label)
            if gap:
                raise ContractGap(f"{label}-expert-axis", gap)
            matched = torch.stack([expert_by_rank[r][name] for r in range(WORLD_SIZE)])
        elif name in rank_args[0]:
            if not target or target[0] != WORLD_SIZE:
                raise ContractGap(
                    f"{label}-weight-map",
                    f"entry weight argument {name} has shape {target} without the EP{WORLD_SIZE} rank axis",
                )
            per_rank = [rank_args[r][name] for r in range(WORLD_SIZE)]
            sample = per_rank[0]
            if list(sample.shape) == target[1:]:
                matched = torch.stack(per_rank)
            elif sample.shape[0] == 1 and list(sample.shape[1:]) == target[1:]:
                matched = torch.stack([value.squeeze(0) for value in per_rank])
            else:
                raise ContractGap(
                    f"{label}-weight-map",
                    f"materialized per-rank {name} shape {list(sample.shape)} cannot fill the "
                    f"entry shape {target}",
                )
        else:
            continue
        if list(matched.shape) != target:
            raise ContractGap(
                f"{label}-weight-map",
                f"assembled {name} shape {list(matched.shape)} != entry {target}",
            )
        values[name] = matched
    return values


def prefill_weight_args(contract, rank_args, expert_by_rank, shared_expert_shard=None):
    """Assemble prefill weight arguments from per-rank materializations."""
    if shared_expert_shard is not None:
        _log("weight", f"shared-expert-shard={shared_expert_shard} requested")
    return _assemble_rank_weights(contract, rank_args, expert_by_rank, "prefill")


def slice_layer_stacks(rank_args, expert_by_rank, first_layer, num_layers):
    """Slice the leading layer axis for a bounded decode slice (0, 40 keeps the full set)."""
    if (first_layer, num_layers) == (0, N_LAYERS):
        return rank_args, expert_by_rank
    start, stop = first_layer, first_layer + num_layers
    sliced_ranks = {
        rank: {
            name: value[start:stop] if value.shape[0] == N_LAYERS else value
            for name, value in values.items()
        }
        for rank, values in rank_args.items()
    }
    sliced_experts = {
        rank: {name: value[start:stop] for name, value in values.items()}
        for rank, values in expert_by_rank.items()
    }
    return sliced_ranks, sliced_experts


def decode_weight_args(contract, rank_args, expert_by_rank):
    """Assemble the decode entry's dense/top-level/source/expert arguments."""
    return _assemble_rank_weights(contract, rank_args, expert_by_rank, "decode")


def decode_source_args(checkpoint, layers, rank=0):
    """Materialize one rank's cache-owner source arguments under the entry names."""
    needed = list(KV_SLOTS) + list(REINDEX_LAYERS)
    missing = [layer for layer in needed if layer not in layers]
    if missing:
        raise ValueError(f"decode source packing needs layers {needed}; missing {missing}")
    loader = WeightLoader(checkpoint, rank=rank)
    weights = materialize_rank(loader, layers, with_experts=False)
    return dict(weights.sources)


def compare_shapes(label, actual, expected_shape, expected_dtype):
    problems = []
    if list(actual.shape) != list(expected_shape):
        problems.append(f"{label}: shape {list(actual.shape)} != entry {list(expected_shape)}")
    if actual.dtype != expected_dtype:
        problems.append(f"{label}: dtype {actual.dtype} != entry {expected_dtype}")
    return problems


def contract_report(prefill_contract, decode_contract, prefill_outputs=None, decode_outputs=None):
    """Report where the complete entries cannot consume the real rank materialization."""
    gaps = []
    for label, contract in (("prefill", prefill_contract), ("decode", decode_contract)):
        gap = expert_layer_axis_gap(contract, label)
        if gap:
            gaps.append(gap)
    provided = set(prefill_outputs or ())
    required = set(decode_outputs or HANDOFF_NAMES) & HANDOFF_NAMES
    missing = sorted(
        name for name in required
        if not all(source in provided for source in HANDOFF_SOURCES[name])
    )
    if missing:
        gaps.append("prefill does not export the decode handoff state: " + ", ".join(missing))
    return gaps


# ---------------------------------------------------------------------------
# Compile-once and per-forward dispatch
# ---------------------------------------------------------------------------


def make_run_config(device_ids, backend_type="a2a3", execution_mode=False):
    from pypto.ir import DistributedConfig
    from golden.runner import normalize_config
    from pypto.runtime import ExecutionMode

    if execution_mode:
        raise ValueError("generation requires explicit ONBOARD execution; simulator is not accepted")
    config = dict(
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        backend_type=backend_type,
        execution_mode=execution_mode,
        ring_heap=[0, 0, 4 << 30, 0],
    )
    if normalize_config(config).execution_mode != ExecutionMode.ONBOARD:
        raise RuntimeError("normalized generation mode is not ONBOARD")
    return config


def compile_entry(label, entry, specs, config):
    """Compile one complete entry from its signature or bounded harness specs."""
    from golden.runner import normalize_config

    started = time.time()
    if specs is None:
        compiled = entry.compile(config=normalize_config(config))
        work_dir = compiled.output_dir
    else:
        from golden import run

        result = run(fn=entry, specs=specs, compile_only=True, config=config)
        if not result.passed:
            raise RuntimeError(f"{label} compile failed: {result.error}")
        work_dir = result.work_dir
    elapsed = time.time() - started
    metadata = list(Path(work_dir).rglob("distributed_meta.json"))
    if not metadata or any(json.loads(path.read_text()).get("platform") != "a2a3" for path in metadata):
        raise RuntimeError(f"{label} generated platform is missing or not a2a3 ONBOARD")
    _log("compile", f"{label} compiled in {elapsed:.2f}s artifact={work_dir}")
    return str(work_dir), elapsed


def reload_compiled(work_dir, config):
    """Reload a compiled artifact for repeated dispatch without recompiling."""
    from golden.runner import _reload_from_dir, normalize_config

    return _reload_from_dir(Path(work_dir), normalize_config(config))


def dispatch(compiled, entry, values, config, contract):
    """Dispatch through an already prepared resident owner."""
    if not isinstance(compiled, PreparedDispatch):
        raise TypeError("generation dispatch requires a live PreparedDispatch owner")
    return compiled.dispatch(values)


def static_weight_names():
    """Return static rank-sharded resource names for the complete entries."""
    names = set(DENSE_STACK_NAMES + EXPERT_STACK_NAMES)
    names.update(("embed_weight", "lm_head_weight", "final_norm"))
    names.update(("engram_wkv", "engram_qk", "engram_rotation"))
    for prefix, family in (("c2a_", C2A_SOURCE_NAMES), ("c1a_", C1A_SOURCE_NAMES), ("re_", REINDEX_SOURCE_NAMES)):
        names.update(prefix + name for name in family)
    return names


class PreparedDispatch:
    """Own rank-sharded static weights and stable shared host IO for one phase.

    Mutable state is copied through shared host tensors on each call. Device
    weight handles live until close; protocol domains are recreated per call.
    Handles never cross prepared-worker ownership or phase boundaries.
    """

    def __init__(self, compiled, entry, values, config, contract, weight_names=None,
                 weight_source=None, copy_chunk_bytes=64 << 20, operation_admission=None, phase=None):
        from golden.runner import normalize_config
        from pypto.runtime import ExecutionMode

        if normalize_config(config).execution_mode != ExecutionMode.ONBOARD:
            raise ValueError("prepared generation requires ONBOARD")
        self.compiled, self.entry, self.contract = compiled, entry, contract
        self.config = normalize_config(config)
        self.weight_names = (static_weight_names() if weight_names is None else set(weight_names)) & contract.keys()
        self.weight_source = weight_source
        self.extra_compiled = ()
        self.select_program = False
        if operation_admission is not None and not isinstance(operation_admission, FirstRealGenerationAdmission):
            raise TypeError("only the explicit first-real-generation admission is supported")
        self.operation_admission, self.phase = operation_admission, phase
        if operation_admission is not None:
            operation_admission.validate_contract(phase, contract)
        if type(copy_chunk_bytes) is not int or copy_chunk_bytes < 1:
            raise ValueError("copy chunk size must be a positive byte count")
        self.copy_chunk_bytes = copy_chunk_bytes
        skip = self.weight_names if weight_source is not None else ()
        ordered_dispatch_args(entry, values, contract, skip_names=skip)
        self.io = {}
        for name, param in contract.items():
            if param.kind == "tensor" and name not in self.weight_names:
                value = values[name]
                self.io[name] = value if value.is_shared() and value.is_contiguous() else value.contiguous().share_memory_()
                values[name] = self.io[name]
        self.context = None
        self.worker = None
        self.handles = {}
        self.stats = {"resident_upload_bytes": 0, "resident_allocations": 0, "dispatches": 0,
                      "rank_allocations": 0, "copy_calls": 0, "max_leaf_bytes": 0, "max_copy_bytes": 0,
                      "state_transport": "shared-host H2D/D2H each dispatch", "protocol_domains": "fresh each dispatch"}

    def open(self, values):
        if self.worker is not None:
            raise RuntimeError("prepared owner already open")
        if self.extra_compiled:
            self.context = self.compiled.prepare(self.config, extra_compiled=self.extra_compiled, persistent=False)
        else:
            self.context = self.compiled.prepare(self.config, persistent=False)
        self.worker = self.context.__enter__()
        try:
            if self.operation_admission is not None:
                self.operation_admission.guard(self.worker, self.phase, "prepared-prewarm")
                self.weight_source = self.operation_admission.weight_source(
                    self.operation_admission.checkpoint, self.worker, self.phase)
            if self.weight_source is None:
                for name in sorted(self.weight_names):
                    value = values[name]
                    self.handles[name] = self.worker.alloc_stacked_tensor(value.contiguous())
                    self.stats["resident_allocations"] += 1
                    self.stats["rank_allocations"] += int(value.shape[0])
                    self.stats["resident_upload_bytes"] += value.numel() * value.element_size()
            else:
                self._stream_weights()
        except BaseException:
            try:
                self.close()
            except BaseException as cleanup_error:
                _log("cleanup-error", repr(cleanup_error))
            raise
        return self

    def _stream_weights(self):
        """Allocate empty rank buffers and fill each exact byte range serially."""
        from pypto.runtime import StackedDeviceTensor

        world = len(self.config.distributed_config.device_ids)
        pending = {name: [] for name in sorted(self.weight_names)}
        try:
            for rank in range(world):
                for name, shards in pending.items():
                    param = self.contract[name]
                    if param.kind != "tensor" or len(param.shape) < 2 or param.shape[0] != world:
                        raise ValueError(f"static resource {name} lacks the exact worker rank axis")
                    tail = tuple(param.shape[1:])
                    itemsize = torch.empty((), dtype=param.dtype).element_size()
                    rank_bytes = math.prod(tail) * itemsize
                    if self.operation_admission is not None:
                        self.operation_admission.guard(self.worker, self.phase, "before-allocation", rank)
                    shard = self.worker.alloc_tensor(tail, param.dtype, init=None, worker_id=rank)
                    shards.append(shard)
                    if self.operation_admission is not None:
                        self.operation_admission.remaining[self.phase][rank] -= rank_bytes
                        self.operation_admission.guard(self.worker, self.phase, "after-allocation", rank)
                    self.stats["rank_allocations"] += 1
                    cursor = 0
                    iterator = iter(self.weight_source(name, rank, tail, param.dtype))
                    while True:
                        try:
                            leaf = next(iterator)
                        except StopIteration:
                            break
                        cursor = self._upload_leaf(leaf, name, rank, tail, param.dtype, shard, cursor, rank_bytes)
                        if self.operation_admission is not None:
                            self.operation_admission.copied(self.phase, leaf, cursor)
                        del leaf
                    if cursor != rank_bytes:
                        raise ValueError(f"incomplete static resource {name} rank{rank}: {cursor}/{rank_bytes} bytes")
            for name, shards in pending.items():
                self.handles[name] = StackedDeviceTensor(shards, tuple(self.contract[name].shape), tuple(range(world)))
                self.stats["resident_allocations"] += 1
        except BaseException:
            for shards in pending.values():
                for rank, shard in enumerate(shards):
                    try:
                        self.worker.free_tensor(shard, worker_id=rank)
                    except Exception as cleanup_error:
                        _log("cleanup-error", repr(cleanup_error))
            self.handles.clear()
            raise

    def _upload_leaf(self, leaf, name, rank, tail, dtype, shard, cursor, rank_bytes):
        """Validate one contiguous leaf and bound its public copy staging calls."""
        value = leaf.tensor
        if leaf.argument != name or leaf.rank != rank or tuple(leaf.rank_shape) != tail:
            raise ValueError("streamed leaf resource/rank/shape does not match its allocation")
        if value.device.type != "cpu" or not value.is_contiguous() or value.dtype != dtype:
            raise ValueError("streamed leaf must have the declared dtype in contiguous CPU memory")
        itemsize = value.element_size()
        nbytes = value.numel() * itemsize
        if type(leaf.byte_offset) is not int or leaf.byte_offset != cursor or nbytes < 1:
            raise ValueError("streamed resource has a gap, overlap, duplicate or empty leaf")
        if cursor % itemsize or cursor + nbytes > rank_bytes:
            raise ValueError("streamed leaf byte range exceeds the rank allocation")
        chunk = self.copy_chunk_bytes // itemsize * itemsize
        if chunk < 1:
            raise ValueError("copy chunk cannot hold one declared element")
        for begin in range(0, nbytes, chunk):
            count = min(chunk, nbytes - begin)
            if self.operation_admission is not None:
                self.operation_admission.guard(self.worker, self.phase, "before-copy", rank, name, count)
            self.worker.copy_to(shard.data_ptr, value.data_ptr(), count, dst_offset=cursor + begin,
                                src_offset=begin, worker_id=rank)
            if self.operation_admission is not None:
                self.operation_admission.guard(self.worker, self.phase, "after-copy", rank, name, count)
            self.stats["copy_calls"] += 1
            self.stats["max_copy_bytes"] = max(self.stats["max_copy_bytes"], count)
        self.stats["resident_upload_bytes"] += nbytes
        self.stats["max_leaf_bytes"] = max(self.stats["max_leaf_bytes"], nbytes)
        return cursor + nbytes

    def dispatch(self, values):
        if self.worker is None:
            raise RuntimeError("prepared owner is closed")
        ordered_dispatch_args(self.entry, values, self.contract, overrides=self.handles)
        args = []
        for name in self.entry.param_names:
            param = self.contract[name]
            if name in self.handles:
                args.append(self.handles[name])
            elif param.kind == "scalar":
                args.append(scalar_dispatch_value(values[name], param.dtype))
            else:
                shared = self.io[name]
                if values[name] is not shared:
                    shared.copy_(values[name])
                    values[name] = shared
                args.append(shared)
        started = time.time()
        if self.operation_admission is not None:
            self.operation_admission.before_forward(self.worker, self.phase)
        if self.select_program:
            self.worker.run(self.compiled, *args, config=self.config)
        else:
            self.worker(*args, config=self.config)
        if self.operation_admission is not None:
            self.operation_admission.dispatched[self.phase] = True
            self.operation_admission.forward_calls[self.phase] += 1
            self.operation_admission.guard(self.worker, self.phase, "after-forward")
        self.stats["dispatches"] += 1
        return time.time() - started

    def close(self):
        failures = []
        try:
            if self.worker is not None:
                if self.operation_admission is not None:
                    try:
                        self.operation_admission.guard(self.worker, self.phase, "phase-final-before-free")
                    except BaseException as cleanup_error:
                        failures.append(cleanup_error)
                for handle in self.handles.values():
                    try:
                        self.worker.free_stacked_tensor(handle)
                    except Exception as error:
                        failures.append(error)
        finally:
            self.handles.clear()
            try:
                if self.worker is not None and self.operation_admission is not None:
                    try:
                        self.operation_admission.guard(self.worker, self.phase, "phase-final-after-free")
                    except BaseException as cleanup_error:
                        failures.append(cleanup_error)
                if self.context is not None:
                    self.context.__exit__(None, None, None)
            except BaseException as cleanup_error:
                failures.append(cleanup_error)
            finally:
                self.context = self.worker = None
        if failures:
            raise RuntimeError(f"resident cleanup failed: {failures}")


class SharedWeightOwner(PreparedDispatch):
    """Two registered programs, one canonical static allocation set and shared host IO."""

    _PHASE_FIELDS = ("compiled", "entry", "contract", "io", "handles", "stats", "phase")

    def __init__(self, prefill_compiled, prefill_entry, prefill_values, prefill_contract,
                 decode_compiled, decode_entry, decode_values, decode_contract, config,
                 *, weight_source, operation_admission):
        if operation_admission is None or not operation_admission.shared_weights:
            raise ContractGap("shared-weights", "a distinct shared-owner Root operation is required")
        super().__init__(prefill_compiled, prefill_entry, prefill_values, config, prefill_contract,
                         weight_source=weight_source, operation_admission=operation_admission, phase="prefill")
        operation_admission.validate_contract("decode", decode_contract)
        self.decode_binding = PreparedDispatch(decode_compiled, decode_entry, decode_values, self.config,
                                              decode_contract, weight_source=weight_source)
        if self.weight_names != self.decode_binding.weight_names:
            raise ValueError("shared phases must bind the same static resource names")
        world = len(self.config.distributed_config.device_ids)
        for name in self.weight_names:
            source, target = prefill_contract[name], decode_contract[name]
            same_shape = list(source.shape) == list(target.shape)
            c1_view = name.startswith("c1a_") and list(target.shape) == [world, 1, *source.shape[1:]]
            if not (same_shape or c1_view) or source.dtype != target.dtype:
                raise ValueError(f"{name}: unsupported shared static phase shape/dtype")
        self.extra_compiled = (decode_compiled,)
        self.select_program = True
        self.prefill_binding = {name: getattr(self, name) for name in self._PHASE_FIELDS}
        self.decode_binding.phase = "decode"
        self.decode_binding.select_program = True
        self.decode_binding.stats.update(static_binding="borrowed from prefill bases", borrowed_views=0)
        self.alias_records = []
        self.closed = False

    def open(self, values):
        from pypto.runtime import StackedDeviceTensor

        if self.closed:
            raise RuntimeError("a closed shared owner cannot be reopened")
        try:
            super().open(values)
            world = len(self.config.distributed_config.device_ids)
            for name in sorted(self.weight_names):
                base = self.prefill_binding["handles"][name]
                target = self.decode_binding.contract[name]
                views = []
                for rank, source in enumerate(base.shards):
                    target_shape = tuple(target.shape[1:])
                    view = source if source.shape == target_shape else self.worker.view_tensor(
                        source, target_shape, byte_offset=0, worker_id=rank)
                    if (view.nbytes != source.nbytes or view.dtype != source.dtype
                            or view.buffer is not source.buffer or view.data_ptr != source.data_ptr):
                        raise ValueError("decode view must alias the complete initialized canonical base")
                    views.append(view)
                    self.alias_records.append({"rank": rank, "argument": name,
                        "base_shape": list(source.shape), "consumer_shape": list(view.shape),
                        "dtype": str(view.dtype), "base_nbytes": source.nbytes, "consumer_nbytes": view.nbytes,
                        "byte_offset": 0, "base_data_ptr": source.data_ptr, "view_data_ptr": view.data_ptr})
                self.decode_binding.handles[name] = StackedDeviceTensor(views, tuple(target.shape), tuple(range(world)))
            self.decode_binding.stats["borrowed_views"] = len(self.alias_records)
            self.operation_admission.bind_shared_decode(self.alias_records)
        except BaseException:
            try:
                self.close()
            except BaseException as cleanup_error:
                _log("cleanup-error", repr(cleanup_error))
            raise
        return self

    def activate_decode(self, values):
        if self.worker is None or self.phase != "prefill" or self.stats["dispatches"] < 1:
            raise RuntimeError("decode requires a live shared owner and completed prefill")
        # Copy returned prefill state into the decode IO that existed before prepare/fork.
        for name, shared in self.decode_binding.io.items():
            value = values[name]
            if list(value.shape) != list(shared.shape) or value.dtype != shared.dtype:
                raise ValueError(f"{name}: decode handoff shape/dtype differs from preallocated IO")
            if value is not shared:
                shared.copy_(value)
            values[name] = shared
        for name in self._PHASE_FIELDS:
            setattr(self, name, getattr(self.decode_binding, name))
        self.operation_admission.guard(self.worker, "decode", "shared-phase-boundary")
        return self

    def reuse_record(self):
        return {"mode": "one-worker/two-registered-programs", "canonical_phase": "prefill",
                "static_upload_bytes_total": self.prefill_binding["stats"]["resident_upload_bytes"],
                "base_rank_allocations_total": self.prefill_binding["stats"]["rank_allocations"],
                "static_source_reads_total": self.operation_admission.read_calls["prefill"],
                "static_source_bytes_total": self.operation_admission.source_bytes["prefill"],
                "decode_new_static_source_reads": self.operation_admission.read_calls["decode"],
                "decode_new_static_source_bytes": self.operation_admission.source_bytes["decode"],
                "decode_new_static_upload_bytes": 0, "decode_new_static_allocations": 0,
                "decode_static_binding": "same initialized bases via public same-worker views",
                "borrowed_consumer_rank_views": len(self.alias_records)}

    def close(self):
        active_phase = self.phase
        if hasattr(self, "prefill_binding"):
            for name in self._PHASE_FIELDS:
                if name != "phase":
                    setattr(self, name, self.prefill_binding[name])
        self.phase = active_phase
        try:
            super().close()
        finally:
            self.closed = True
            if hasattr(self, "decode_binding"):
                # Borrowed aliases are not allocation identities and must never be freed separately.
                self.decode_binding.handles.clear()
                self.decode_binding.worker = self.decode_binding.context = None


def disabled_engram_inputs(contract):
    """Bind inert resources for an actual runtime-disabled Engram branch."""
    values = {"engram_enabled": 0}
    for name in ("engram_lookup", "engram_wkv", "engram_qk", "engram_rotation"):
        param = contract[name]
        values[name] = torch.zeros((), dtype=param.dtype).expand(tuple(param.shape))
    return values


def preflight_dispatch(entry, contract, inputs, *, phase="decode"):
    """Validate mutable host IO and metadata-only static resource coverage."""
    names = static_weight_names() & contract.keys()
    ordered_dispatch_args(entry, inputs, contract, skip_names=names)
    for name in names:
        param = contract[name]
        tail, dtype = WeightLoader.argument_contract(name, phase=phase)
        if list(param.shape) != [WORLD_SIZE, *tail] or param.dtype != dtype:
            raise ValueError(f"{name}: loader metadata does not match complete entry")
    return {"arguments": len(contract), "streamed_static_arguments": len(names),
            "engram_enabled": scalar_dispatch_value(inputs.get("engram_enabled", 1), torch.int32)}


def streaming_weight_source(checkpoint, *, loader_factory=WeightLoader):
    """Keep one rank's header reader and yield one transformed leaf at a time."""
    current_rank, current_loader = None, None

    def source(name, rank, shape, dtype):
        nonlocal current_rank, current_loader
        if current_rank != rank:
            current_loader = None
            current_loader = loader_factory(checkpoint, rank=rank)
            current_rank = rank
        return current_loader.iter_argument_leaves(name, shape, dtype, with_engram=False)

    return source


def argument_memory_plan(contract):
    """Report logical bytes and a conservative per-device input bound."""
    logical = {}
    for name, param in contract.items():
        if param.kind == "scalar":
            continue
        logical[name] = math.prod(param.shape) * torch.empty((), dtype=param.dtype).element_size()
    upper = sum(size // WORLD_SIZE if name in static_weight_names() else size for name, size in logical.items())
    return {"logical_host_bytes": sum(logical.values()), "per_device_input_upper_bytes": upper,
            "argument_bytes": logical, "transient_and_compiler_bytes": "unmeasured",
            "allocation_allowed": False}


# ---------------------------------------------------------------------------
# Host-side greedy sampling
# ---------------------------------------------------------------------------


def _shard_argmax(logits, rank, row):
    value, local_index = logits[rank, row].max(dim=-1)
    return float(value), rank % TP_SIZE * LOCAL_VOCAB + int(local_index)


def greedy_sample(logits, row, dp_group=0, check_groups=True, *, context=None):
    """Select over TP shards and require requested DP replica agreement."""
    candidates = [_shard_argmax(logits, dp_group * TP_SIZE + tp_rank, row) for tp_rank in range(TP_SIZE)]
    best_value, best_id = max(candidates, key=lambda item: item[0])
    votes = []
    disagreements = []
    groups = range(WORLD_SIZE // TP_SIZE) if check_groups else (dp_group,)
    for group in groups:
        group_candidates = candidates if group == dp_group else [
            _shard_argmax(logits, group * TP_SIZE + tp, row) for tp in range(TP_SIZE)
        ]
        group_value, group_id = max(group_candidates, key=lambda item: item[0])
        votes.append({"dp_group": group, "token_id": group_id, "logit": group_value,
                      "shard_argmax": group_candidates})
        if group_id != best_id:
            disagreements.append({"dp_group": group, "token_id": group_id, "value": group_value})
    sample = {
        "token_id": best_id,
        "logit": best_value,
        "shard_argmax": candidates,
        "dp_disagreements": disagreements,
        "dp_votes": votes,
        "agreement_checked": check_groups,
        "context": dict(context or {}),
    }
    if check_groups and disagreements:
        evidence = {"kind": "dp_greedy_disagreement", "selected_dp_group": dp_group,
                    "row": int(row), "selection": sample}
        _log("sampling-failure", json.dumps(evidence, sort_keys=True))
        raise SamplingDisagreement(evidence)
    return sample


def state_progress(values):
    """Record persistent-state checksums and positions for diagnostics."""
    progress = {}
    for name in PAGED_CACHE_NAMES + ("x_hc", "pre_mix"):
        if name in values:
            progress[name + "_sha1"] = _sha1(values[name])
    for name in ("topk_indices", "pool_block_ids"):
        if name in values:
            progress[name + "_nonzero"] = int(torch.count_nonzero(values[name] >= 0))
    if "position_ids" in values:
        progress["position_ids"] = values["position_ids"][0].tolist()
    return progress


# ---------------------------------------------------------------------------
# HBM sampling
# ---------------------------------------------------------------------------


class HbmSampler(threading.Thread):
    """Sample ``npu-smi info`` HBM usage for the allocated devices."""

    LINE = re.compile(r"^\|\s*(\d+)\s+(\d+)\s*\|")

    def __init__(self, device_ids, interval=1.0):
        super().__init__(daemon=True)
        self.device_ids = [int(device) for device in device_ids]
        self.interval = interval
        self.samples = {device: [] for device in self.device_ids}
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval)

    def _sample_once(self):
        try:
            output = subprocess.run(
                ["npu-smi", "info"], capture_output=True, text=True, timeout=30, check=False
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return
        for line in output.splitlines():
            match = self.LINE.match(line)
            if not match:
                continue
            device = int(match.group(2))
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
            if pairs and device in self.samples:
                self.samples[device].append(int(pairs[-1][0]))

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)

    def summary(self):
        return {
            str(device): {"peak_mb": max(samples) if samples else None, "samples": len(samples)}
            for device, samples in self.samples.items()
        }


# ---------------------------------------------------------------------------
# Host checks: weight check, dry run, contract report
# ---------------------------------------------------------------------------


def check_weight_loader(args):
    """Validate the frozen loader and the per-rank EP16 placement."""
    from utils import unpack_nz

    layers = _parse_int_list(args.check_layers)
    ranks = _parse_int_list(args.check_ranks)
    errors = []
    started = time.time()
    for rank in ranks:
        loader = WeightLoader(args.checkpoint, rank=rank)
        _log("weight", f"rank={rank} tp_rank={loader.tp_rank} validating layers={layers}")
        errors.extend(f"rank {rank}: {error}" for error in loader.validate(layers, args.sample))
        expert = loader.expert_shard(layers[0])
        for name, value in expert.items():
            if value.shape[0] != N_LOCAL_EXPERTS:
                errors.append(f"rank {rank}: {name} leading axis {value.shape[0]} != {N_LOCAL_EXPERTS}")
        global_expert = loader.ep_rank * N_LOCAL_EXPERTS
        direct = loader.reader.tensor(f"layers.{layers[0]}.ffn.experts.{global_expert}.w1.weight")
        if not torch.equal(unpack_nz(expert["w1"][0]), direct.transpose(0, 1).contiguous()):
            errors.append(f"rank {rank}: local expert 0 is not global expert {global_expert}")
        _log("weight", f"rank={rank} expert shard {tuple(expert['w1'].shape)} global expert {global_expert}")
        del loader, expert
    _log("weight", f"loader validation finished in {time.time() - started:.1f}s")
    return errors


def _entry_rank_tail(contract, name):
    """Per-rank shape expected under the entry's leading EP rank axis."""
    shape = [int(dim) for dim in contract[name].shape]
    if not shape or shape[0] != WORLD_SIZE:
        return None
    return shape[1:]


def _check_rank_weights_against_prefill(contract, layers, ranks, errors, with_engram=True):
    expert_gap = expert_layer_axis_gap(contract, "prefill")
    for rank in ranks:
        loader = WeightLoader(CHECKPOINT, rank=rank)
        weights = materialize_rank(loader, layers, with_experts=True, with_engram=with_engram)
        for name, value in weights.dense.items():
            expected = contract[name]
            tail = _entry_rank_tail(contract, name)
            if tail is None:
                errors.append(f"rank {rank} {name}: entry has no EP{WORLD_SIZE} rank axis")
                continue
            target = [len(layers)] + tail[1:] if tail and tail[0] == N_LAYERS else tail
            errors.extend(compare_shapes(f"rank {rank} {name}", value, target, expected.dtype))
        for name, value in weights.experts.items():
            expected = contract[name]
            tail = _entry_rank_tail(contract, name)
            if expert_gap or tail is None or len(tail) < 2 or tail[0] != N_LAYERS:
                # Reported once by contract_report; per-layer shards cannot be
                # compared against the entry's single rank shard.
                continue
            target = [len(layers)] + tail[1:]
            errors.extend(compare_shapes(f"rank {rank} {name}", value, target, expected.dtype))
        for name, value in weights.sources.items():
            if name not in contract:
                errors.append(f"materialized source argument {name} is absent from the prefill entry")
                continue
            expected = contract[name]
            tail = _entry_rank_tail(contract, name)
            if tail is None:
                errors.append(f"rank {rank} {name}: entry has no EP{WORLD_SIZE} rank axis")
                continue
            fits = list(value.shape) == tail or (
                value.shape[0] == 1 and list(value.shape[1:]) == tail
            )
            if not fits:
                errors.append(
                    f"rank {rank} {name}: per-rank shape {list(value.shape)} != entry tail {tail}"
                )
        for name, value in weights.top.items():
            tail = _entry_rank_tail(contract, name)
            if tail is None:
                errors.append(f"top-level argument {name} has no entry rank axis")
            elif list(value.shape) != tail:
                errors.append(
                    f"top-level argument {name}: shape {list(value.shape)} != entry tail {tail}"
                )
        for layer, table in weights.engram_tables.items():
            row = table.row(table.rows // 2)
            _log(
                "engram",
                f"rank={rank} layer={layer} table rows={table.rows} width={table.width} "
                f"sampled row sha1={_sha1(row)}",
            )
        _log(
            "weight",
            f"rank={rank} tp={loader.tp_rank} layers={layers} dense={len(weights.dense)} "
            f"sources={len(weights.sources)} expert_shard={tuple(weights.experts['w1'].shape)}",
        )
        del loader, weights


def _check_decode_source_packing(decode_contract, errors):
    packed = decode_source_args(CHECKPOINT, list(KV_SLOTS) + list(REINDEX_LAYERS))
    for name, value in packed.items():
        expected = decode_contract[name]
        tail = _entry_rank_tail(decode_contract, name)
        fits = tail is not None and (
            list(value.shape) == tail
            or (value.shape[0] == 1 and list(value.shape[1:]) == tail)
        )
        if not fits:
            errors.append(
                f"decode source {name}: per-rank shape {list(value.shape)} dtype {value.dtype} "
                f"cannot fill entry {expected.shape} {expected.dtype}"
            )
    _log("weight", f"decode source slots packed and shape-checked: {sorted(packed)}")
    del packed


def append_environment_gaps(gaps, rope, prompt_length, tile):
    """Append the resolved RoPE-profile and prefill-tile gaps to a report."""
    if rope["mismatched_layers"]:
        gaps.append(
            f"rope-profile: entry_mode={rope['entry_mode']} carries one table for all 40 layers; "
            f"the checkpoint plan needs base (0-1) and compressed (2-39); mismatched layers "
            f"{rope['mismatched_layers']}"
        )
    gap = prefill_tile_gap(prompt_length, tile)
    if gap:
        gaps.append("prefill-tile: " + gap)
    return gaps


def check_engram_composition(checkpoint, prefill_contract, decode_contract):
    """Confirm the Engram FLOAT projections match the released A5 semantics.

    The checkpoint stores ``engram.wkv.weight`` as ``[KV_OUT, ENGRAM_K]``
    (25600 x 6144).  The A3 projection helper takes a logical ``[out, in]``
    weight and multiplies with ``b_trans``; the A5 loader transposes the same
    tensor to ``[ENGRAM_K, KV_OUT]`` and multiplies directly.  Both compute
    ``lookup @ W^T``.  The gate weight is the per-channel ``q_weight * k_weight``
    fold in both stacks.
    """
    loader = WeightLoader(checkpoint, rank=0)
    records = []
    errors = []
    for layer in M.engram_layer_ids:
        slot = M.engram_layer_ids.index(layer)
        dense = loader.layer_dense(layer)
        wkv = dense["engram_wkv"]
        qk = dense["engram_qk"]
        q_weight = loader.reader.tensor(f"layers.{layer}.engram.q_weight").float()
        k_weight = loader.reader.tensor(f"layers.{layer}.engram.k_weight").float()
        record = {
            "layer": layer,
            "wkv_shape": list(wkv.shape),
            "wkv_dtype": str(wkv.dtype),
            "qk_shape": list(qk.shape),
            "qk_dtype": str(qk.dtype),
            "qk_matches_product": bool(torch.equal(qk, q_weight * k_weight)),
            "a5_layout": "checkpoint [KV_OUT, ENGRAM_K] transposed to [ENGRAM_K, KV_OUT]",
            "a3_layout": "[KV_OUT, ENGRAM_K] with b_trans=True; same lookup @ W^T",
            "entry_prefill_slice": list(prefill_contract["engram_wkv"].shape[2:]),
            "entry_decode_slice": list(decode_contract["engram_wkv"].shape[2:]),
            "sha1": _sha1(wkv),
        }
        records.append(record)
        if tuple(wkv.shape) != (25600, ENGRAM_K):
            errors.append(f"layer {layer}: wkv shape {tuple(wkv.shape)} != (25600, {ENGRAM_K})")
        if not record["qk_matches_product"]:
            errors.append(f"layer {layer}: engram_qk is not q_weight * k_weight")
        for label, contract in (("prefill", prefill_contract), ("decode", decode_contract)):
            if list(contract["engram_wkv"].shape[2:]) != [25600, ENGRAM_K]:
                errors.append(f"{label} engram_wkv slice {list(contract['engram_wkv'].shape[2:])} mismatch")
            if list(contract["engram_qk"].shape[2:]) != [HC_MULT, D]:
                errors.append(f"{label} engram_qk slice {list(contract['engram_qk'].shape[2:])} mismatch")
        _log(
            "engram-float",
            f"layer={layer} wkv={record['wkv_shape']} {record['wkv_dtype']} "
            f"qk={record['qk_shape']} q*k={record['qk_matches_product']} sha1={record['sha1']}",
        )
    return records, errors


def dry_run(args):
    """Validate tokenizer, weight materialization and input construction without a device."""
    tokenizer = load_tokenizer(args.model_root)
    prompt = build_prompt(
        tokenizer, args.prompt, args.template, args.messages, args.prompt_file,
        model_root=args.model_root, thinking_mode=args.thinking_mode,
        reasoning_effort=args.reasoning_effort, case_file=args.case_file,
        case_index=args.case_index,
    )
    prefill_tokens = exact_prompt_tokens(prompt)
    print_rank_mapping()
    prompt_fixtures = check_prompt_fixtures(tokenizer, args.model_root)
    _prefill_module, prefill_entry, decode_module = load_entries(prefill_tokens)
    prefill_contract, decode_contract, _prefill_specs, _decode_specs = build_entry_contracts(
        prefill_tokens
    )
    _log("entry", f"prefill parameters={len(prefill_contract)} decode parameters={len(decode_contract)}")
    rope = rope_metadata(prefill_contract, args.rope_profile)
    _log(
        "rope",
        f"requested={rope['requested']} entry_mode={rope['entry_mode']} "
        f"mismatched_layers={rope['mismatched_layers']}",
    )
    if args.engram_mode == "disabled":
        engram_records, engram_errors = {"scope": "noEngram"}, []
    else:
        engram_records, engram_errors = check_engram_composition(CHECKPOINT, prefill_contract, decode_contract)

    prefill_values = build_prefill_inputs(
        prefill_contract, prompt,
        start_position=args.start_position, rope_profile=args.rope_profile,
    )
    for name, value in prefill_values.items():
        problems = compare_shapes(name, value, prefill_contract[name].shape, prefill_contract[name].dtype)
        if problems:
            raise ValueError("; ".join(problems))
    tile = int(prefill_contract["input_ids"].shape[1])
    _log(
        "inputs",
        f"prefill tensors built: tile={tile} prompt={len(prompt['ids'])} "
        f"positions=[{args.start_position},{args.start_position + tile})",
    )
    decode_values = build_decode_state(
        decode_contract, position=args.start_position, rope_profile=args.rope_profile
    )
    for name, value in decode_values.items():
        problems = compare_shapes(name, value, decode_contract[name].shape, decode_contract[name].dtype)
        if problems:
            raise ValueError("; ".join(problems))
    _log("inputs", f"decode state tensors built (cold start at position {args.start_position})")

    handoff = handoff_to_decode(
        prefill_values, decode_contract, len(prompt["ids"]),
        start_position=args.start_position, tile=prefill_tokens, allow_padded=True,
        row_offset_c2a=args.handoff_offset_c2a, row_offset_c1a=args.handoff_offset_c1a,
    )
    for name, value in handoff.items():
        problems = compare_shapes(name, value, decode_contract[name].shape, decode_contract[name].dtype)
        if problems:
            raise ValueError("; ".join(problems))
        decode_values[name] = value
    _log("handoff", f"prefill->decode mapping built for {len(handoff)} state tensors")

    layers = _parse_int_list(args.check_layers)
    ranks = _parse_int_list(args.check_ranks)
    errors = list(engram_errors)
    _check_rank_weights_against_prefill(prefill_contract, layers, ranks, errors,
                                       with_engram=args.engram_mode != "disabled")
    if set(KV_SLOTS + REINDEX_LAYERS) <= set(layers):
        _check_decode_source_packing(decode_contract, errors)
    else:
        _log("weight", "decode source packing skipped: --check-layers omits source layers")
    for error in errors:
        _log("weight-FAIL", error)
    if errors:
        raise ValueError(f"{len(errors)} weight contract mismatch(es)")

    gaps = contract_report(
        prefill_contract, decode_contract,
        prefill_entry.output_param_names, decode_module.decode_fwd.output_param_names,
    )
    gaps = append_environment_gaps(gaps, rope, len(prompt["ids"]), tile)
    for gap in gaps:
        _log("gap", gap)
    diagnostic_only = args.allow_padded_prefill
    if args.engram_mode == "host-ids" and args.engram_hash_ids:
        hash_ids = load_engram_hash_ids(args.engram_hash_ids, M.engram_layer_ids, args.tokens)
        tables = {layer: WeightLoader(CHECKPOINT, rank=0).engram_table(layer) for layer in M.engram_layer_ids}
        lookup = build_engram_lookup(tables, hash_ids, 0, args.tokens)
        _log("engram", f"Engram lookup built from host ids: {tuple(lookup.shape)}")
    elif args.engram_mode == "host-ids":
        gaps = gaps + [
            "engram-hash: no --engram-hash-ids; host lookup rows cannot be built. Options: "
            "(a) --engram-mode zero-diagnostic (DIAGNOSTIC-ONLY, Engram residual omitted) or "
            "(b) a user-supplied n-gram hash spec"
        ]
        _log("gap", gaps[-1])
    elif args.engram_mode == "zero-diagnostic":
        lookup = diagnostic_engram_lookup(prefill_contract)
        diagnostic_only = True
        _log(
            "policy",
            "engram-hash: zero-diagnostic mode omits the Engram residual; this run is "
            "DIAGNOSTIC-ONLY (accepting policy, not a contract gap; promote with "
            "--engram-mode host-ids + --engram-hash-ids)",
        )
    _log("dry-run", f"PASS with {len(gaps)} reported contract gap(s)")
    extras = {
        "prompt_fixtures": prompt_fixtures, "rope": rope, "engram": engram_records,
        "diagnostic_only": diagnostic_only, "engram_mode": args.engram_mode,
    }
    return prompt, gaps, extras


def contract_only(args):
    """Print the complete-entry contract versus the real rank materialization."""
    _prefill_module, prefill_entry, decode_module = load_entries(args.tokens)
    prefill_contract, decode_contract, _prefill_specs, _decode_specs = build_entry_contracts(
        args.tokens
    )
    print_rank_mapping()
    for label, contract in (("prefill", prefill_contract), ("decode", decode_contract)):
        _log(label, f"parameters={len(contract)}")
        for name, param in contract.items():
            if param.kind == "tensor":
                _log(label + "-param", f"{name}: {param.shape} {param.dtype}")
    gaps = contract_report(
        prefill_contract, decode_contract,
        prefill_entry.output_param_names, decode_module.decode_fwd.output_param_names,
    )
    rope = rope_metadata(prefill_contract, args.rope_profile)
    _log(
        "rope",
        f"requested={rope['requested']} entry_mode={rope['entry_mode']} "
        f"mismatched_layers={rope['mismatched_layers']} scaling={rope['checkpoint_rope_scaling']}",
    )
    gaps = append_environment_gaps(gaps, rope, args.tokens, args.tokens)
    for gap in gaps:
        _log("gap", gap)
    _log("contract", f"gaps={len(gaps)}")
    variants = [rank for rank in range(WORLD_SIZE) if rank % TP_SIZE == 0]
    for name in TP_SLICED_DENSE:
        shape = decode_contract[name].shape
        _log(
            "decode-check",
            f"{name}: rank_axis={shape[0]} layer_axis={shape[1]} (entry ABI: EP{WORLD_SIZE} x {N_LAYERS})",
        )
    _log("decode-check", f"dense TP variants={variants} expert ranks={list(range(WORLD_SIZE))}")
    if int(decode_contract["input_ids"].shape[1]) != 1:
        raise ValueError("generation decode contract must contain exactly one token")
    state = {
        "input_ids": torch.zeros((WORLD_SIZE, 1), dtype=torch.int64),
        "token_counts": torch.ones(WORLD_SIZE, dtype=torch.int32),
    }
    for position in (126, 127, 128, 129):
        advance_decode_inputs(state, decode_contract, 1, position, args.rope_profile)
        if int(state["window_slots"][0, 0]) != position:
            raise ValueError("single-request window slot does not advance with position")
        source_slot = position // 2 if position % 2 else -1
        if int(state["c2a_compressed_slots"][0, 0]) != source_slot:
            raise ValueError("C2 emission slot does not follow odd-position parity")
        required = {"rope_cos", "rope_sin", "rope_cos_swa", "rope_sin_swa"}
        if not required <= state.keys():
            raise ValueError("generation is missing a required query RoPE profile")
    _log("metadata-check", "PASS decode1, dual profiles, positions126-129, C2 parity and page boundary")
    _log("memory-plan", json.dumps({"prefill": argument_memory_plan(prefill_contract),
                                   "decode": argument_memory_plan(decode_contract)}))
    return gaps


# ---------------------------------------------------------------------------
# Device path
# ---------------------------------------------------------------------------


def resolve_device_ids(text):
    if not text:
        raise ValueError("device ids are required for device work; submit with task-submit and pass -d $TASK_DEVICE")
    ids = [int(item) for item in text.split(",") if item.strip() != ""]
    if len(ids) != WORLD_SIZE or len(set(ids)) != WORLD_SIZE:
        raise ValueError(f"need exactly {WORLD_SIZE} distinct device ids for EP{EP_SIZE}, got {ids}")
    return ids


def compile_only(args):
    """Compile both complete entries once on the allocated devices."""
    _require_complete_decode_slice(args)
    device_ids = resolve_device_ids(args.device)
    config = make_run_config(device_ids, args.backend_type, args.execution_mode)
    _prefill_module, prefill_entry, decode_module = load_entries(args.tokens)
    prefill_contract, decode_contract, prefill_specs, decode_specs = build_entry_contracts(
        args.tokens, pages=max(2, math.ceil((args.start_position + args.tokens + args.max_new_tokens) / BLOCK_SIZE))
    )
    # ``decode_fwd`` sizes its EP exchange windows from the trace-time token
    # bound, so set it before tracing/compiling the decode entry.
    decode_module._MOE_ROWS = int(decode_contract["position_ids"].shape[1])
    results = {}
    selected = [item.strip() for item in args.entries.split(",") if item.strip()]
    unknown = set(selected) - {"prefill", "decode"}
    if unknown:
        raise ValueError(f"unknown compile entries {sorted(unknown)}; use prefill,decode")
    if "prefill" in selected:
        results["prefill"] = compile_entry("prefill", prefill_entry, None, config)
    if "decode" in selected:
        results["decode"] = compile_entry("decode", decode_module.decode_fwd, None, config)
    return {"entries": results}


def _require_complete_decode_slice(args):
    if (args.first_layer, args.num_layers) != (0, N_LAYERS):
        raise ValueError(
            "the current decode entry is the complete "
            f"{N_LAYERS}-layer program and has no first_layer/num_layers scalar ABI; "
            "a bounded decode slice cannot be expressed"
        )


def require_full_materialization_budget(allow_full=False):
    """Hold full weights until rank materialization and upload overlap is bounded."""
    raise ContractGap(
        "host-materialization-overlap",
        "full weights remain held: streaming source/owner is implemented but bounded native reuse, "
        f"functional continuation and measured real host/HBM fit are unproved (allow_full_materialize={allow_full}); "
        "no real payload allocation is authorized by this switch",
    )


def parallel_operation_budget(prefill_contract, decode_contract, max_new_tokens):
    """Derive conservative future IO/window allowances from the configured ABI."""
    from config import DECODE_MAX_TOKENS, PREFILL_MAX_TOKENS
    from moe_exchange import COUNT_PAD

    if type(max_new_tokens) is not int or not 2 <= max_new_tokens <= 16:
        raise ValueError("bounded configurable operation requires 2..16 generated tokens")
    io = {}
    windows = {}
    static = {}
    for phase, contract, rows, capacity in (
        ('prefill', prefill_contract, 3, PREFILL_MAX_TOKENS),
        ('decode', decode_contract, 1, DECODE_MAX_TOKENS),
    ):
        memory = argument_memory_plan(contract)['argument_bytes']
        io[phase] = sum(size for name, size in memory.items() if name not in static_weight_names())
        static[phase] = sum(size // WORLD_SIZE for name, size in memory.items() if name in static_weight_names())
        # Full-global logical windows are charged as a per-rank planning bound.
        receive_rows = N_LOCAL_EXPERTS * EP_SIZE * rows
        local_windows = 2 * capacity * D * 2 + 2 * TP_SIZE * 4
        local_windows += receive_rows * (D * 2 + 8 * 4)
        local_windows += EP_SIZE * COUNT_PAD * 4 + 5 * EP_SIZE * 4
        local_windows += rows * M.num_experts_per_tok * D * 2
        windows[phase] = WORLD_SIZE * local_windows
    if static['prefill'] != static['decode']:
        raise ValueError("shared phases require equal static byte ownership")
    unspent = (1 << 30) + (8 << 30) + (128 << 20)
    communication = max(680528320, sum(windows.values()))
    initial = max(18088299200, unspent + sum(io.values()) + (max_new_tokens - 1) * io['decode'] + communication)
    return {'schema': 'configurable-real-resource-budget/1', 'topology': parallel_topology(),
            'host_floor_bytes': 64 << 30, 'disk_floor_bytes': 2 << 30, 'shm_floor_bytes': 16 << 30,
            'static_per_rank_bytes': static['prefill'], 'nonstatic_global_IO_bytes': io,
            'full_global_window_planning_bytes': windows, 'communication_reserve_bytes': communication,
            'unspent_reserve_bytes': unspent, 'initial_reserve_bytes': initial,
            'max_new_tokens': max_new_tokens, 'measurement_kind': 'CONSERVATIVE_SOURCE_PLANNING_NOT_PEAK',
            'whole_source_backing_host_allowance_bytes': 5362417664}


class FirstRealGenerationAdmission:
    """One explicitly Root-admitted real40/noEngram operation; all limits are planning."""

    def __init__(self, args):
        import weakref
        self.weakref = weakref
        path = getattr(args, 'first_real_generation_authorization', None)
        if path is None:
            require_full_materialization_budget(args.allow_full_materialize)
        self.receipt = json.loads(Path(path).read_text())
        r = self.receipt
        self.configurable = r.get('schema') == 'configurable-real-generation-authorization/1'
        operations = ('configurable-shared-real40-noEngram-greedy-N3',) if self.configurable else (
            'bounded-full-real40-noEngram-greedy-N3', 'shared-full-real40-noEngram-greedy-N3')
        if (r.get('schema') not in ('first-real-generation-authorization/1', 'configurable-real-generation-authorization/1')
                or r.get('approved') is not True or r.get('kind') != 'ACTUAL_DEVICE_FIRST_REAL_GENERATION'
                or r.get('operation') not in operations):
            raise ContractGap('first-real-admission', 'separate exact Root generation scope required')
        if not self.configurable and (TP_SIZE, EP_SIZE) != (4, 16):
            raise ContractGap('first-real-admission', 'nondefault topology requires new configurable Root schema')
        self.shared_weights = self.configurable or r['operation'] == 'shared-full-real40-noEngram-greedy-N3'
        self.shared_decode_aliases = None
        self.prompt_text = r.get('prompt_text')
        self.prompt_ids = r.get('prompt_ids')
        if (type(self.prompt_text) is not str or not self.prompt_text
                or type(self.prompt_ids) is not list or len(self.prompt_ids) != 3
                or any(type(token) is not int or not 0 <= token < M.vocab_size for token in self.prompt_ids)):
            raise ContractGap('first-real-prompt', 'receipt requires exact plain prompt text and three vocabulary IDs')
        if r.get('ring_heap') != [0, 0, 4294967296, 0] or r.get('execution_mode') != 'ONBOARD' or r.get('expert_layers') != 40:
            raise ContractGap('first-real-admission', 'explicit public config/profile binding mismatch')
        now = time.time()
        if not 0 < r.get('expires_at_epoch', 0) - now <= 3600:
            raise ContractGap('first-real-admission', 'expired or overlong grant')
        def sha(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.sha = sha
        self.plan = json.loads(Path(r['metadata_plan']).read_text())
        if sha(r['metadata_plan']) != r['metadata_plan_sha256']:
            raise ContractGap('first-real-admission', 'header/target plan changed')
        if self.configurable:
            topology = parallel_topology()
            if (r.get('topology') != topology or self.plan.get('topology') != topology
                    or any(type(r['topology'].get(key)) is not int for key in topology)
                    or self.plan.get('schema') != 'configurable-full-real-generation-metadata/1'
                    or self.plan.get('expert_layers') != 40
                    or self.plan.get('source_manifest_sha256') != r.get('source_manifest_sha256')
                    or r.get('reviewed_source_manifest_sha256') != r.get('source_manifest_sha256')):
                raise ContractGap('first-real-admission', 'new topology/full-source plan binding required')
            contracts = build_entry_contracts(3, pages=2)[:2]
            self.budget = parallel_operation_budget(*contracts, args.max_new_tokens)
            if (self.plan.get('resource_budget') != self.budget or r.get('resource_budget') != self.budget
                    or self.plan.get('static_per_rank_bytes') != self.budget['static_per_rank_bytes']
                    or type(self.plan.get('static_per_rank_bytes')) is not int):
                raise ContractGap('first-real-admission', 'fresh topology resource plan mismatch')
            if self.budget['static_per_rank_bytes'] >= 64 << 30:
                raise ContractGap('first-real-memory', 'full E40 statics alone exceed the supported A3 device budget')
            for phase in ('prefill', 'decode'):
                targets = self.plan['phases'][phase]['targets']
                keys = {(row['rank'], row['argument']) for row in targets}
                expected_keys = {(rank, name) for rank in range(WORLD_SIZE) for name in static_weight_names()}
                if (keys != expected_keys or len(targets) != len(expected_keys)
                        or self.plan['phases'][phase]['initialized_device_bytes'] != WORLD_SIZE * self.budget['static_per_rank_bytes']):
                    raise ContractGap('first-real-admission', 'complete topology static target coverage required')
        elif (self.plan['source_manifest_sha256'] != '8153c7f040bef51900a6e6c9ff27b018279b7503f31d50ec5488457e49770b28'
                or self.plan['expert_layers'] != 40 or self.plan['static_per_rank_bytes'] != 39624816576):
            raise ContractGap('first-real-admission', 'reviewed producer base/real40 schema mismatch')
        if self.shared_weights:
            expected = {"canonical_phase": "prefill", "static_upload_bytes": self.plan["phases"]["prefill"]["initialized_device_bytes"],
                        "static_source_reads": self.plan["phases"]["prefill"]["source_read_calls"],
                        "static_source_bytes": self.plan["phases"]["prefill"]["cumulative_source_bytes"],
                        "base_rank_allocations": len(self.plan["phases"]["prefill"]["targets"]),
                        "decode_static_upload_bytes": 0, "decode_static_source_reads": 0}
            actual = r.get("weight_accounting")
            integer_keys = set(expected) - {"canonical_phase"}
            if (not isinstance(actual, dict) or actual != expected
                    or any(type(actual.get(key)) is not int for key in integer_keys)):
                raise ContractGap("shared-weights", "new Root receipt must bind one-upload/borrowed-consumer accounting")
        repo = Path(__file__).resolve().parents[2]
        source_files = json.loads(Path(r['source_manifest']).read_text())
        expected_paths = self.plan['source_manifest_files']
        if set(source_files) != set(expected_paths) or len(source_files) != 45 or any(sha(repo / p) != h for p, h in source_files.items()):
            raise ContractGap('first-real-admission', 'new applied product source45 changed')
        if sha(r['source_manifest']) != r['source_manifest_sha256'] or any(
                h != expected_paths[p] for p, h in source_files.items()
                if self.configurable or p != 'models/deepseek_v4_1_flash_a3_w8a8/generate.py'):
            raise ContractGap('first-real-admission', 'new source manifest binding mismatch')
        ids = resolve_device_ids(args.device)
        if len(ids) != WORLD_SIZE or len(set(ids)) != WORLD_SIZE or ids != r.get('device_ids'):
            raise ContractGap('first-real-admission', 'actual configured ordered device IDs mismatch')
        if (args.execution_mode is not False or args.backend_type != 'a2a3'
                or args.engram_mode != 'disabled' or args.template != 'plain'
                or args.prompt != self.prompt_text or args.start_position != 0
                or type(args.max_new_tokens) is not int or type(r.get('max_new_tokens')) is not int
                or args.max_new_tokens != r.get('max_new_tokens') or not 2 <= args.max_new_tokens <= 16
                or args.repeat != 1 or args.max_time != 21600 or args.allow_padded_prefill or args.decode_state is not None
                or args.prompt_file is not None or args.messages is not None or args.case_file is not None):
            raise ContractGap('first-real-admission', 'bounded ONBOARD/noEngram/exactN3 request mismatch')
        from simpler_setup.runtime_builder import RuntimeBuilder
        library = RuntimeBuilder(platform='a2a3').get_binaries('tensormap_and_ringbuffer', build=False).host_path
        if sha(library) != '24434f8879030ef1b1f31e5fc2ecce683e392d4f1baeb5bcee852fb183976e5c':
            raise ContractGap('first-real-admission', 'installed selected TMR library changed')
        prerequisites = {
            'native_empty_calibration_1794040381/actual/calibration.json': '3bc7291e174cccc8a8f0272d9786716a78bccb2ae8336c33fdfd4e10a16927f0',
            'native_empty_reservation_1794041910/actual/probe.json': 'c4b62d80662c62058f4ba4a237e606e745b07d8506eb45c0e9ccb098aeec953e',
            'native_real_transfer_1794047567/actual/result.json': '2d0a4bd282bdb24a7620e36bd2c9a45ee46005893d5ff9363fca30047cf8b4ee',
        }
        evidence = repo / 'experiments/dsv41-flash-w8a8-a3/logs'
        if any(sha(evidence / path) != value for path, value in prerequisites.items()):
            raise ContractGap('first-real-admission', 'historical accepted prerequisites changed')
        calibration = json.loads((evidence / next(iter(prerequisites))).read_text())
        if not self.configurable and ids != calibration['binding']['device_ids']:
            raise ContractGap('first-real-admission', 'historical ordered16 mapping mismatch')
        self.checkpoint = Path(args.checkpoint).resolve()
        if str(self.checkpoint) != self.plan['checkpoint'] or sha(self.checkpoint / 'quant_model_weights.safetensors.index.json') != self.plan['index_sha256']:
            raise ContractGap('first-real-admission', 'checkpoint/index changed')
        self.output = Path(r['progress_jsonl'])
        if self.output.exists():
            raise ContractGap('first-real-admission', 'fresh exclusive progress file required')
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + 20400
        self.remaining = {p: [self.plan['static_per_rank_bytes']] * WORLD_SIZE for p in ('prefill', 'decode')}
        if self.shared_weights:
            self.remaining['decode'] = [0] * WORLD_SIZE
        self.cursors = {p: {(v['rank'], v['argument']): 0 for v in self.plan['phases'][p]['targets']} for p in ('prefill', 'decode')}
        self.sizes = {p: {(v['rank'], v['argument']): v['nbytes'] for v in self.plan['phases'][p]['targets']} for p in ('prefill', 'decode')}
        self.reads = {p: {} for p in ('prefill', 'decode')}
        self.dispatched = {'prefill': False, 'decode': False}
        self.forward_calls = {'prefill': 0, 'decode': 0}
        self.max_new_tokens = args.max_new_tokens
        self.leases = {'prefill': None, 'decode': None}
        self.last_event = 0
        self.guard_count = self.public_query_count = 0
        self.read_calls = {'prefill': 0, 'decode': 0}
        self.source_bytes = {'prefill': 0, 'decode': 0}
        self.initialized_bytes = {'prefill': 0, 'decode': 0}
        self.current_progress = {}
        self.last_public = {}
        self.query_state = {}
        self.queried_resources = set()
        self.active_rank_phase = None
        self.coverage_emitted = set()
        self.event('admitted', source=r['source_manifest_sha256'], operation=r['operation'], device_ids=ids, topology=parallel_topology())

    def event(self, stage, **fields):
        row = {'epoch': time.time(), 'stage': stage, **fields}
        with self.output.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()
            if 'final' in stage or stage in ('complete-coverage', 'borrowed-static-coverage'):
                os.fsync(stream.fileno())

    def host_guard(self):
        if time.monotonic() > self.deadline:
            raise ContractGap('first-real-timebox', 'deadline reached between synchronous calls; no retry')
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        host = int(memory['MemAvailable'].split()[0]) * 1024
        disk = os.statvfs(self.output.parent)
        shm = os.statvfs('/dev/shm')
        result = {'host': host, 'disk': disk.f_bavail * disk.f_frsize, 'shm': shm.f_bavail * shm.f_frsize}
        if result['host'] < 64 * (1 << 30) or result['disk'] < 2 * (1 << 30) or result['shm'] < 16 * (1 << 30):
            raise ContractGap('first-real-memory', 'fresh HOST/DISK/SHM planning floor failed')
        return result

    def guard(self, worker, phase, stage, rank=None, resource=None, copy_bytes=0):
        host = self.host_guard()
        ranks = range(WORLD_SIZE) if rank is None else (rank,)
        if self.configurable:
            reserve = self.budget['unspent_reserve_bytes'] if self.dispatched[phase] else self.budget['initial_reserve_bytes']
        else:
            reserve = 9797894144 if self.dispatched[phase] else 18088299200
        if phase == 'decode' and stage == 'before-forward' and self.dispatched[phase]:
            calls_left = max(0, self.max_new_tokens - 1 - self.forward_calls[phase])
            if self.configurable:
                reserve += calls_left * self.budget['nonstatic_global_IO_bytes']['decode'] + self.budget['communication_reserve_bytes']
            else:
                reserve += calls_left * 247111680 + 680528320
        rows = []
        for current in ranks:
            key = (phase, current)
            now = time.monotonic()
            state = self.query_state.setdefault(key, {'time': None, 'bytes': 0})
            if stage == 'after-copy':
                state['bytes'] += copy_bytes
            first_resource = resource is not None and (phase, current, resource) not in self.queried_resources
            source_key = self.current_progress.get('source_key')
            source_size = self.plan.get('source_descriptors', {}).get(source_key, {}).get('nbytes', 0)
            force = (rank is None or 'allocation' in stage or 'forward' in stage or 'final' in stage
                     or stage == 'prepared-prewarm' or self.active_rank_phase != key
                     or state['time'] is None or now - state['time'] >= 1
                     or (stage == 'before-source-read' and source_size >= 64 << 20)
                     or (stage == 'before-copy' and (first_resource or state['bytes'] + copy_bytes > 256 << 20))
                     or (stage == 'after-copy' and state['bytes'] >= 256 << 20))
            self.active_rank_phase = key
            if not force:
                continue
            free, total = worker.device_memory_info(worker_id=current)
            committed = worker.committed_device_memory(worker_id=current)
            if (any(type(v) is not int for v in (free, total, committed)) or not 0 <= free <= total
                    or committed < 0 or free < self.remaining[phase][current] + reserve):
                raise ContractGap('first-real-memory', f'{stage} rank{current}: free={free} total={total} committed={committed} remaining={self.remaining[phase][current]} reserve={reserve}')
            rows.append({'rank': current, 'free': free, 'total': total, 'committed': committed})
            state.update(time=now, bytes=0)
            if stage == 'before-copy' and resource is not None:
                self.queried_resources.add((phase, current, resource))
        self.guard_count += 1
        self.public_query_count += 2 * len(rows)
        for row in rows:
            self.last_public[row['rank']] = row
        now = time.monotonic()
        if now - self.last_event >= 5 or 'allocation' in stage or 'forward' in stage or 'final' in stage or stage in ('prepared-prewarm', 'shared-phase-boundary'):
            self.event(stage, phase=phase, host=host, memory=rows, reserve=reserve,
                       progress=self.current_progress, guarded_boundaries=self.guard_count,
                       public_queries=self.public_query_count)
            self.last_event = now

    def weight_source(self, checkpoint, worker, phase):
        admission = self
        class Reader:
            def __init__(self, reader, rank):
                self.reader, self.rank = reader, rank
            def __getattr__(self, name):
                return getattr(self.reader, name)
            def tensor(self, name):
                admission.current_progress = {'phase': phase, 'rank': self.rank, 'source_key': name,
                                              'source_reads_completed': admission.read_calls[phase],
                                              'source_bytes_completed': admission.source_bytes[phase]}
                admission.guard(worker, phase, 'before-source-read', self.rank)
                lease = admission.leases[phase]
                if lease is not None and lease() is not None:
                    raise ContractGap('first-real-source', 'previous transformed leaf retained before next source read')
                expected = admission.plan['source_descriptors'].get(name)
                cap = admission.plan['phases'][phase]['source_reads'].get(name, 0)
                count = admission.reads[phase].get(name, 0)
                if expected is None or count >= cap:
                    raise ContractGap('first-real-source', 'unexpected key/read cap before payload')
                shard, header_size, header = self.reader.header(name)
                info = header[name]
                actual = {'shard': shard.name, 'shape': info['shape'], 'dtype': info['dtype'],
                          'absolute_byte_range': [8 + header_size + n for n in info['data_offsets']],
                          'nbytes': self.reader.nbytes(name), 'file_size': shard.stat().st_size,
                          'file_mtime_ns': shard.stat().st_mtime_ns}
                if actual != expected:
                    raise ContractGap('first-real-source', 'source descriptor/range changed before payload')
                value = self.reader.tensor(name)
                admission.reads[phase][name] = count + 1
                admission.read_calls[phase] += 1
                admission.source_bytes[phase] += expected['nbytes']
                admission.guard(worker, phase, 'after-source-read', self.rank)
                return value
        def factory(checkpoint, rank):
            loader = WeightLoader(checkpoint, rank=rank)
            if admission.sha(loader.map_path) != admission.receipt['weight_map_sha256']:
                raise ContractGap('first-real-source', 'weight map changed')
            loader.reader = Reader(loader.reader, rank)
            return loader
        return streaming_weight_source(checkpoint, loader_factory=factory)

    def copied(self, phase, leaf, end):
        key = (leaf.rank, leaf.argument)
        if key not in self.cursors[phase] or leaf.byte_offset != self.cursors[phase][key] or end > self.sizes[phase][key]:
            raise ContractGap('first-real-coverage', 'target interval gap/overlap/overflow')
        self.initialized_bytes[phase] += end - self.cursors[phase][key]
        self.cursors[phase][key] = end
        self.leases[phase] = self.weakref.ref(leaf.tensor)
        self.current_progress = {'phase': phase, 'rank': leaf.rank, 'argument': leaf.argument,
                                 'target_byte_end': end, 'target_byte_total': self.sizes[phase][key],
                                 'initialized_global_bytes': self.initialized_bytes[phase]}

    def before_forward(self, worker, phase):
        if self.sha(self.checkpoint / 'quant_model_weights.safetensors.index.json') != self.plan['index_sha256']:
            raise ContractGap('first-real-source', 'source index changed before forward')
        if self.shared_weights and phase == 'decode':
            return self.before_shared_decode(worker)
        if self.cursors[phase] != self.sizes[phase] or self.reads[phase] != self.plan['phases'][phase]['source_reads']:
            raise ContractGap('first-real-coverage', 'NoForward until all912 targets and exact source reads complete')
        self.guard(worker, phase, 'before-forward')
        if phase not in self.coverage_emitted:
            self.event('complete-coverage', phase=phase,
                       source_manifest_sha256=self.receipt['source_manifest_sha256'],
                       metadata_plan_sha256=self.receipt['metadata_plan_sha256'],
                       initialized_device_bytes=sum(self.cursors[phase].values()),
                       completed_H2D_bytes_counter=self.initialized_bytes[phase],
                       source_read_calls=sum(self.reads[phase].values()),
                       source_read_calls_counter=self.read_calls[phase],
                       cumulative_source_bytes=sum(self.plan['source_descriptors'][key]['nbytes'] * count
                                                   for key, count in self.reads[phase].items()),
                       source_bytes_counter=self.source_bytes[phase],
                       source_reads=self.reads[phase],
                       endpoints=[{'rank': rank, 'argument': name, 'byte_end': end,
                                   'expected_nbytes': self.sizes[phase][(rank, name)]}
                                  for (rank, name), end in sorted(self.cursors[phase].items())])
            self.coverage_emitted.add(phase)

    def bind_shared_decode(self, aliases):
        if not self.shared_weights:
            raise ContractGap("shared-weights", "legacy operation cannot borrow static coverage")
        if (self.cursors["prefill"] != self.sizes["prefill"]
                or self.reads["prefill"] != self.plan["phases"]["prefill"]["source_reads"]):
            raise ContractGap("shared-weights", "canonical upload must complete before binding borrowed views")
        keyed = {(row["rank"], row["argument"]): row for row in aliases}
        if len(keyed) != len(aliases) or set(keyed) != set(self.sizes["decode"]):
            raise ContractGap("shared-weights", "all decode consumers must map once to canonical bases")
        for key, row in keyed.items():
            if (row["byte_offset"] != 0 or row["base_nbytes"] != self.sizes["prefill"][key]
                    or row["consumer_nbytes"] != self.sizes["decode"][key]
                    or row["consumer_nbytes"] != row["base_nbytes"]
                    or row["base_data_ptr"] != row["view_data_ptr"]):
                raise ContractGap("shared-weights", "borrowed decode byte coverage differs from initialized base")
        self.shared_decode_aliases = aliases

    def before_shared_decode(self, worker):
        if (self.shared_decode_aliases is None or "prefill" not in self.coverage_emitted
                or self.cursors["prefill"] != self.sizes["prefill"]
                or self.reads["prefill"] != self.plan["phases"]["prefill"]["source_reads"]
                or any(self.cursors["decode"].values()) or self.reads["decode"]
                or self.initialized_bytes["decode"] or self.read_calls["decode"] or self.source_bytes["decode"]):
            raise ContractGap("shared-weights", "decode requires borrowed coverage and zero second static IO")
        self.guard(worker, "decode", "before-forward")
        if "decode" not in self.coverage_emitted:
            self.event("borrowed-static-coverage", phase="decode", canonical_phase="prefill",
                       source_manifest_sha256=self.receipt["source_manifest_sha256"],
                       metadata_plan_sha256=self.receipt["metadata_plan_sha256"],
                       covered_initialized_bytes=sum(row["consumer_nbytes"] for row in self.shared_decode_aliases),
                       new_static_upload_bytes=0, new_static_source_reads=0, new_static_source_bytes=0,
                       canonical_static_upload_bytes=self.initialized_bytes["prefill"],
                       canonical_source_reads=self.read_calls["prefill"],
                       canonical_source_bytes=self.source_bytes["prefill"], aliases=self.shared_decode_aliases)
            self.coverage_emitted.add("decode")

    def validate_contract(self, phase, contract):
        if phase not in ('prefill', 'decode'):
            raise ContractGap('first-real-admission', 'unknown phase')
        expected = {v['argument']: v for v in self.plan['phases'][phase]['targets'] if v['rank'] == 0}
        if len(expected) != 57 or (static_weight_names() & contract.keys()) != expected.keys():
            raise ContractGap('first-real-admission', 'full57 ABI mismatch before allocation')
        for name, row in expected.items():
            param = contract[name]
            if list(param.shape) != [WORLD_SIZE, *row['shape']] or str(param.dtype) != row['dtype']:
                raise ContractGap('first-real-admission', 'E40 phase/dtype/shape mismatch before allocation')


def device_run(args):
    """Run prefill, then greedy decode steps through the compiled entries."""
    _require_complete_decode_slice(args)
    if args.engram_mode != "disabled":
        raise ContractGap("first-path-engram", "first generation path requires --engram-mode disabled (noEngram)")
    if args.shared_expert_shard is not None:
        raise ContractGap("expert-shards", "noEngram generation requires every real EP rank shard")
    admission = None
    if args.first_real_generation_authorization is None:
        require_full_materialization_budget(args.allow_full_materialize)
    else:
        admission = FirstRealGenerationAdmission(args)
        admission.host_guard()
    started = time.time()
    checkpoint = Path(args.checkpoint)
    device_ids = resolve_device_ids(args.device)
    config = make_run_config(device_ids, args.backend_type, args.execution_mode)
    tokenizer = load_tokenizer(args.model_root)
    prompt = build_prompt(
        tokenizer, args.prompt, args.template, args.messages, args.prompt_file,
        model_root=args.model_root, thinking_mode=args.thinking_mode,
        reasoning_effort=args.reasoning_effort, case_file=args.case_file,
        case_index=args.case_index,
    )
    prefill_tokens = exact_prompt_tokens(prompt)
    if admission is not None and prompt["ids"] != admission.prompt_ids:
        raise ContractGap("first-real-prompt", "actual tokenizer IDs differ from receipt-bound plain prompt")
    print_rank_mapping()

    _prefill_module, prefill_entry, decode_module = load_entries(prefill_tokens)
    prefill_contract, decode_contract, prefill_specs, decode_specs = build_entry_contracts(
        prefill_tokens, pages=max(2, math.ceil((args.start_position + prefill_tokens + args.max_new_tokens) / BLOCK_SIZE))
    )
    gaps = contract_report(
        prefill_contract, decode_contract,
        prefill_entry.output_param_names, decode_module.decode_fwd.output_param_names,
    )
    tile = int(prefill_contract["input_ids"].shape[1])
    rope = rope_metadata(prefill_contract, args.rope_profile)
    gaps = append_environment_gaps(gaps, rope, len(prompt["ids"]), tile)
    for gap in gaps:
        _log("gap", gap)
    tile_gap = prefill_tile_gap(len(prompt["ids"]), tile)
    if tile_gap and not args.allow_padded_prefill:
        raise ContractGap("prefill-tile", tile_gap)
    gap = prefill_ep16_gap(prefill_contract)
    if gap:
        raise ContractGap("prefill-ep16-axis", gap)
    engram_records = {"scope": "noEngram", "enabled": False, "original_model_parity": False}

    prefill_values = build_prefill_inputs(
        prefill_contract, prompt,
        start_position=args.start_position, rope_profile=args.rope_profile,
    )
    diagnostic_only = args.allow_padded_prefill
    prefill_values.update(disabled_engram_inputs(prefill_contract))
    preflight = preflight_dispatch(prefill_entry, prefill_contract, prefill_values, phase="prefill")
    memory = {"prefill": argument_memory_plan(prefill_contract), "decode": argument_memory_plan(decode_contract)}
    _log("preflight", json.dumps({"binding": preflight, "memory": memory}))


    sampler = HbmSampler(device_ids, args.hbm_interval)
    steps = []
    prefill_owner = decode_owner = None
    shared_weights = admission is not None and admission.shared_weights
    shared_owner = None
    try:
        sampler.start()
        prefill_work, prefill_compile_s = compile_entry(
            "prefill", prefill_entry, None, config
        )
        prefill_compiled = reload_compiled(prefill_work, config)
        if shared_weights:
            position = args.start_position + len(prompt["ids"])
            decode_values = build_decode_state(decode_contract, position=position, rope_profile=args.rope_profile)
            decode_values.update(disabled_engram_inputs(decode_contract))
            decode_module._MOE_ROWS = int(decode_values["position_ids"].shape[1])
            decode_work, decode_compile_s = compile_entry("decode", decode_module.decode_fwd, None, config)
            decode_compiled = reload_compiled(decode_work, config)
            shared_owner = SharedWeightOwner(
                prefill_compiled, prefill_entry, prefill_values, prefill_contract,
                decode_compiled, decode_module.decode_fwd, decode_values, decode_contract, config,
                weight_source=streaming_weight_source(checkpoint), operation_admission=admission)
            prefill_owner = shared_owner.open(prefill_values)
        else:
            prefill_owner = PreparedDispatch(prefill_compiled, prefill_entry, prefill_values, config, prefill_contract,
                                             weight_source=streaming_weight_source(checkpoint),
                                             operation_admission=admission, phase="prefill").open(prefill_values)
        prefill_s = 0.0
        determinism_runs = []
        for repetition in range(max(1, args.repeat)):
            if repetition:
                fresh = build_prefill_inputs(
                    prefill_contract, prompt,
                    start_position=args.start_position, rope_profile=args.rope_profile,
                )
                for name, value in fresh.items():
                    if name in prefill_values:
                        prefill_values[name].copy_(value)
            elapsed = dispatch(prefill_owner, prefill_entry, prefill_values, config, prefill_contract)
            if repetition == 0:
                prefill_s = elapsed
            repeat_sample = greedy_sample(
                prefill_values["logits"], prefill_tokens - 1, check_groups=True,
                context={"phase": "prefill", "repeat": repetition,
                         "position": args.start_position + prefill_tokens - 1,
                         "prompt_ids": list(prompt["ids"]), "generated_ids": []},
            )
            determinism_runs.append({
                "repeat": repetition,
                "seconds": elapsed,
                "logits_sha1": _sha1(prefill_values["logits"]),
                "token_id": repeat_sample["token_id"],
                "logit": repeat_sample["logit"],
                "sampling": repeat_sample,
            })
            sample = repeat_sample
        determinism = {
            "repeats": len(determinism_runs),
            "runs": determinism_runs,
            "stable": len({run["logits_sha1"] for run in determinism_runs}) == 1,
        }
        _log("run", f"prefill forward {prefill_s:.2f}s")
        _log("determinism", f"repeat={determinism['repeats']} stable={determinism['stable']}")
        generated = [sample["token_id"]]
        _log("sample", f"first token id={sample['token_id']} logit={sample['logit']:.4f} "
                       f"text={tokenizer.decode(generated)!r}")
        generation = {
            "prompt": prompt,
            "rope": rope,
            "engram": engram_records,
            "engram_mode": args.engram_mode,
            "model_scope": "noEngram",
            "original_model_parity": False,
            "diagnostic_only": diagnostic_only,
            "prefill_tokens": prefill_tokens,
            "prefill_seconds": prefill_s,
            "prefill_compile_seconds": prefill_compile_s,
            "first_token": sample,
            "determinism": determinism,
            "generated_ids": generated,
            "generated_text": tokenizer.decode(generated),
            "termination": "prefill-only",
            "positions": {"prefill_start": args.start_position, "prefill_length": len(prompt["ids"])},
            "prefill_residency": dict(prefill_owner.stats),
        }
        if not shared_weights:
            prefill_owner.close()
            prefill_owner = None
        else:
            generation["weight_residency"] = shared_owner.reuse_record()
        if sample["token_id"] == EOS_ID:
            generation["termination"] = "eos"
            return generation, sampler, gaps

        position = args.start_position + len(prompt["ids"])
        if not shared_weights:
            decode_values = build_decode_state(
                decode_contract, position=position, rope_profile=args.rope_profile
            )
        decode_values.update(handoff_to_decode(
            prefill_values, decode_contract, len(prompt["ids"]),
            start_position=args.start_position, tile=prefill_tokens,
            allow_padded=args.allow_padded_prefill,
            row_offset_c2a=args.handoff_offset_c2a, row_offset_c1a=args.handoff_offset_c1a,
        ))
        if args.decode_state is not None:
            saved = torch.load(args.decode_state, map_location="cpu")
            for name, value in saved.items():
                if name in decode_values:
                    decode_values[name] = value
            _log("state", f"decode state overridden from {args.decode_state}")
        decode_values.update(disabled_engram_inputs(decode_contract))
        # ``decode_fwd`` sizes its EP exchange windows from this trace-time
        # token bound; the JIT specialization key includes the concrete shape.
        if shared_weights:
            decode_owner = shared_owner.activate_decode(decode_values)
            prefill_owner = None
        else:
            decode_module._MOE_ROWS = int(decode_values["position_ids"].shape[1])
            decode_work, decode_compile_s = compile_entry(
                "decode", decode_module.decode_fwd, None, config,
            )
            decode_compiled = reload_compiled(decode_work, config)
            decode_owner = PreparedDispatch(decode_compiled, decode_module.decode_fwd, decode_values, config, decode_contract,
                                            weight_source=streaming_weight_source(checkpoint),
                                            operation_admission=admission, phase="decode").open(decode_values)
        decode_rows = int(decode_contract["logits"].shape[1])
        current = sample["token_id"]
        for step in range(args.max_new_tokens - 1):
            advance_decode_inputs(decode_values, decode_contract, current, position, args.rope_profile)
            elapsed = dispatch(decode_owner, decode_module.decode_fwd, decode_values, config, decode_contract)
            sample = greedy_sample(
                decode_values["logits"], decode_rows - 1, check_groups=True,
                context={"phase": "decode", "step": step + 1, "position": position,
                         "input_token": current, "prompt_ids": list(prompt["ids"]),
                         "generated_ids": list(generated)},
            )
            steps.append({
                "step": step + 1,
                "input_token": current,
                "position": position,
                "seconds": elapsed,
                "decode_compile_seconds": decode_compile_s if step == 0 else None,
                "token_id": sample["token_id"],
                "logit": sample["logit"],
                "sampling": sample,
                "state": state_progress(decode_values),
            })
            current = sample["token_id"]
            generated.append(current)
            position += 1
            if current == EOS_ID:
                break
        generation.update({
            "generated_ids": generated,
            "generated_text": tokenizer.decode(generated),
            "termination": "eos" if generated[-1] == EOS_ID else "max_new_tokens",
            "steps": steps,
            "decode_steps": len(steps),
            "positions_end": position,
            "decode_residency": dict(decode_owner.stats),
        })
        if shared_weights:
            generation["weight_residency"] = shared_owner.reuse_record()
        return generation, sampler, gaps
    finally:
        primary = sys.exc_info()[1]
        cleanup_failures = []
        for label, action in (("decode-owner", None if decode_owner is None else decode_owner.close),
                              ("prefill-owner", None if prefill_owner is None else prefill_owner.close),
                              ("sampler", sampler.stop)):
            if action is not None:
                try:
                    action()
                except BaseException as cleanup_error:
                    cleanup_failures.append((label, repr(cleanup_error)))
                    _log("cleanup-error", f"{label}: {cleanup_error!r}")
        _log("run", f"device wall {time.time() - started:.2f}s")
        _log("sampler-state", json.dumps({"alive_after_stop": sampler.is_alive() if hasattr(sampler, "is_alive") else None}))
        if cleanup_failures and primary is None:
            raise RuntimeError(f"generation cleanup failed: {cleanup_failures}")


# ---------------------------------------------------------------------------
# Queue submission
# ---------------------------------------------------------------------------


def submit(args):
    """Submit this driver's device path through the task-submit queue."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else LOG_DIR / f"worker4_submit_{timestamp}.log"
    result_path = Path(args.result_json) if args.result_json else (
        LOG_DIR / f"worker4_generate_device_{timestamp}.json"
    )
    submit_epoch = int(time.time())
    inner = [
        sys.executable, str(Path(__file__).resolve()),
        "--device", "$TASK_DEVICE",
        "--tp", str(TP_SIZE), "--ep", str(EP_SIZE),
        "--model-root", str(args.model_root), "--checkpoint", str(args.checkpoint),
        "--tokens", str(args.tokens),
        "--prompt", args.prompt,
        "--template", args.template,
        "--max-new-tokens", str(args.max_new_tokens),
        "--repeat", str(args.repeat),
        "--start-position", str(args.start_position),
        "--rope-profile", args.rope_profile,
        "--backend-type", args.backend_type,
        "--hbm-interval", str(args.hbm_interval),
        "--handoff-offset-c2a", str(args.handoff_offset_c2a),
        "--handoff-offset-c1a", str(args.handoff_offset_c1a),
        "--queue-submit-epoch", str(submit_epoch),
        "--log-file", str(log_path),
        "--result-json", str(result_path),
    ]
    if args.compile_only:
        inner.extend(["--compile-only", "--entries", args.entries])
    else:
        inner.append("--device-run")
    if args.engram_hash_ids:
        inner.extend(["--engram-hash-ids", str(args.engram_hash_ids)])
    inner.extend(["--engram-mode", args.engram_mode])
    if args.decode_state:
        inner.extend(["--decode-state", str(args.decode_state)])
    if args.messages:
        inner.extend(["--messages", str(args.messages)])
    if args.prompt_file:
        inner.extend(["--prompt-file", str(args.prompt_file)])
    if args.thinking_mode:
        inner.extend(["--thinking-mode", args.thinking_mode])
    if args.reasoning_effort is not None:
        inner.extend(["--reasoning-effort", str(args.reasoning_effort)])
    if args.case_file:
        inner.extend(["--case-file", str(args.case_file), "--case-index", str(args.case_index)])
    if args.allow_padded_prefill:
        inner.append("--allow-padded-prefill")
    if args.execution_mode:
        inner.append("--execution-mode")
    inner.extend(["--first-layer", str(args.first_layer), "--num-layers", str(args.num_layers)])
    if args.shared_expert_shard is not None:
        inner.extend(["--shared-expert-shard", str(args.shared_expert_shard)])
    if args.allow_full_materialize:
        inner.append("--allow-full-materialize")
    if args.first_real_generation_authorization is not None:
        inner += ["--first-real-generation-authorization", args.first_real_generation_authorization]

    command = "cd %s && source activate-pypto.sh && " % shlex.quote(str(ROOT))
    command += " ".join(item if item == "$TASK_DEVICE" else shlex.quote(item) for item in inner)
    queue = [
        "task-submit", "--device", "auto", "--device-num", str(WORLD_SIZE),
        "--max-time", str(args.max_time), "--run", command,
    ]
    _log("submit", "submitting: " + " ".join(queue))
    started = time.time()
    completed = subprocess.run(queue, text=True, check=False)
    _log("submit", f"task-submit exit={completed.returncode} wall={time.time() - started:.1f}s")
    if Path(log_path).exists():
        _log("submit", f"job log {log_path}")
    return completed.returncode


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------


def write_result(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    _log("result", f"wrote {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("--model-root", default=str(CHECKPOINT))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--messages", default=None, help="JSON chat messages for --template chat")
    parser.add_argument("--template", choices=["plain", "chat", "encoding"], default="chat")
    parser.add_argument("--thinking-mode", choices=["chat", "thinking"], default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--case-file", default=None, help="shipped encoding/tests case JSON")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16, help="compile/report tile (2, 16 or 128); generation uses the exact original prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1, help="repeat the prefill forward for determinism")
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--rope-profile", choices=["auto", "base", "compressed"], default="auto")
    parser.add_argument("--handoff-offset-c2a", type=int, default=0)
    parser.add_argument("--handoff-offset-c1a", type=int, default=0)
    parser.add_argument("--allow-padded-prefill", action="store_true")
    parser.add_argument("--engram-hash-ids", default=None)
    parser.add_argument("--engram-mode", choices=["disabled", "host-ids", "zero-diagnostic"],
                        default="disabled", help="first generation path explicitly disables Engram")
    parser.add_argument("--decode-state", default=None)
    parser.add_argument("--check-layers", default="0,1,20,39")
    parser.add_argument("--check-ranks", default=f"0,{WORLD_SIZE - 1}")
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-fixtures", action="store_true")
    parser.add_argument("--weight-check", action="store_true")
    parser.add_argument("--contract-report", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--device-run", action="store_true")
    parser.add_argument("--entries", default="prefill,decode")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend-type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--execution-mode", action="store_true")
    parser.add_argument("--first-layer", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=N_LAYERS)
    parser.add_argument("--shared-expert-shard", type=int, default=None)
    parser.add_argument("--allow-full-materialize", action="store_true")
    parser.add_argument("--first-real-generation-authorization", default=None,
                        help="separate Root-admitted real40/noEngram operation; default remains refused")
    parser.add_argument("--hbm-interval", type=float, default=1.0)
    parser.add_argument("--max-time", type=int, default=3600)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--queue-submit-epoch", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--result-json", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.tp, args.ep) != (TP_SIZE, EP_SIZE):
        raise ValueError("--tp/--ep must be selected before importing model modules")
    if args.compile_only:
        if args.tokens not in (2, 16, 128):
            raise SystemExit(f"unsupported prefill tile {args.tokens}; use 2, 16 or 128")
    if args.submit:
        return submit(args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "compile" if args.compile_only else ("device" if args.device_run else "host")
    log_path = Path(args.log_file) if args.log_file else LOG_DIR / f"worker4_generate_{mode}_{timestamp}.log"
    result_path = Path(args.result_json) if args.result_json else (
        LOG_DIR / f"worker4_generate_{mode}_{timestamp}.json"
    )
    started = time.time()
    device_started = time.time() if (args.compile_only or args.device_run) else None
    payload = {
        "schema": "worker4-generate/1",
        "mode": mode,
        "commands": [" ".join(sys.argv)],
        "rank_mapping": rank_mapping(),
        "topology": parallel_topology(),
        "queue_submit_epoch": args.queue_submit_epoch,
        "start_epoch": int(started),
        "status": "running",
        "blockers": [],
    }
    exit_code = EXIT_OK
    with tee_output(log_path):
        _log("start", f"mode={mode} argv={' '.join(sys.argv[1:])}")
        _log("env", f"python={sys.executable} ptoas={os.environ.get('PTOAS_ROOT', '<unset>')}")
        try:
            if args.weight_check:
                errors = check_weight_loader(args)
                payload["weight_check"] = {"errors": errors}
                if errors:
                    for error in errors:
                        _log("weight-FAIL", error)
                    exit_code = EXIT_FAIL
                else:
                    _log("weight", "PASS")
            elif args.contract_report:
                payload["contract_gaps"] = contract_only(args)
            elif args.prompt_fixtures:
                tokenizer = load_tokenizer(args.model_root)
                payload["prompt_fixtures"] = check_prompt_fixtures(tokenizer, args.model_root)
            elif args.compile_only:
                payload["compile"] = compile_only(args)
            elif args.device_run:
                generation, sampler, gaps = device_run(args)
                payload["generation"] = generation
                payload["hbm"] = sampler.summary()
                payload["contract_gaps"] = gaps
                if gaps:
                    exit_code = EXIT_BLOCKED
            else:
                prompt, gaps, extras = dry_run(args)
                payload["prompt"] = prompt
                payload["contract_gaps"] = gaps
                payload.update(extras)
        except SamplingDisagreement as error:
            payload["sampling_failure"] = error.evidence
            payload["error"] = str(error)
            _log("FAIL", json.dumps(error.evidence, sort_keys=True))
            exit_code = EXIT_FAIL
        except ContractGap as gap:
            _log("blocked", f"{gap.topic}: {gap.detail}")
            payload["blockers"].append({"topic": gap.topic, "detail": gap.detail})
            exit_code = EXIT_BLOCKED
        except SystemExit as error:
            _log("FAIL", f"entry raised SystemExit({error.code})")
            payload["error"] = f"SystemExit({error.code})"
            exit_code = EXIT_FAIL
        except Exception as error:  # noqa: BLE001 - record any entry/compiler failure
            import traceback

            traceback.print_exc()
            _log("FAIL", f"{type(error).__name__}: {error}")
            payload["error"] = f"{type(error).__name__}: {error}"
            exit_code = EXIT_FAIL
        finally:
            device_wall = (time.time() - device_started) if device_started else None
            payload["timing"] = {
                "total_seconds": time.time() - started,
                "device_wall_seconds": device_wall,
                "queue_wait_seconds": (
                    int(device_started) - args.queue_submit_epoch
                    if device_started is not None and args.queue_submit_epoch else None
                ),
            }
            payload["status"] = {
                EXIT_OK: "ok", EXIT_FAIL: "failed", EXIT_BLOCKED: "blocked",
            }.get(exit_code, "unknown")
            payload["exit_code"] = exit_code
            payload["log_file"] = str(log_path)
            write_result(result_path, payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
