# DeepSeek V4 Flash Decode-to-MTP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one PyPTO L3 benchmark that executes the complete DeepSeek V4
Flash main decode, device-side handoff, and logits-only MTP decode at EP=8,
LM-head TP=4, and 128-global / 16-local routed-expert topology.

**Architecture:** A new `@pl.jit.host` entry dispatches `decode_fwd` on every
rank, then a compact `pack_mtp_inputs` child on every rank, then a logits-only
`mtp_decode_layer_logits` child on every rank. Main and MTP stages use distinct
MoE and LM-head windows, while the embedding, RoPE tables, LM-head weight,
main sampled IDs, and main pre-HC hidden state form explicit shared or
producer-consumer edges.

**Tech Stack:** Python 3.10, PyPTO JIT/DSL, PyTorch golden fixtures, pytest,
ruff, Ascend A3 distributed runtime, CANN system profiling.

## Global Constraints

- The direct comparison boundary is the first main `pack_x_hc` embedding task
  through the final task that writes full-vocabulary MTP logits.
- The composite MTP child must execute embedding/input packing, MTP projection,
  SWA, MoE, HC head, RMSNorm, and LM head, with no post-MTP sampler.
- The standalone MTP driver must retain its sampled-output behavior.
- Main decode and MTP must use 128 global experts and 16 local experts at EP=8.
- Trace-hash mode changes only the identity presented to the gate; actual layer
  IDs continue to select weights, caches, attention kind, and MoE epochs.
- Model mode remains the default for existing standalone decode and MTP
  drivers.
- Main and MTP MoE windows and LM-head windows must remain separate.
- Main caches, MTP caches, handoff tail pools, and inter-stage buffers must
  remain device-resident across benchmark rounds.
- The initial acceptance fixture is `[1, 2, 1, 2]`; tail slots are
  `[0, 1, 2, 3]`.
- Generated `build_output/`, profiler traces, logs, and generated tensors must
  not be committed.
- Code comments, docstrings, CLI text, tests, and commit messages are English.
- Run Python checks with a pin-compatible PyPTO source/build. In this workspace
  the verified non-installing activation is:

```bash
export PYPTO_ROOT=../pypto
export PYTHONPATH="$PYPTO_ROOT/python:$PYPTO_ROOT/build/python/bindings${PYTHONPATH:+:$PYTHONPATH}"
```

---

### Task 1: Device-Side Token and Position Handoff

**Files:**
- Modify: `models/deepseek/v4-flash/decode_input_pack.py`
- Create: `tests/contract/test_deepseek_v4_flash_decode_mtp.py`

**Interfaces:**
- Consumes: main `sampled_ids [8, 8] INT32`, main
  `position_ids [8] INT32`, accepted counts `[4] INT32`, tail slots
  `[4] INT32`, token-tail pool `[4] INT64`, and position-tail pool
  `[4] INT32`.
- Produces: the `pack_mtp_inputs` device entry, MTP `input_ids [8] INT64`, MTP
  `position_ids [8] INT32`, updated token-tail pool, and updated
  position-tail pool.

- [ ] **Step 1: Write the failing handoff contract test**

Add the common path loader and the following exact behavior test to
`tests/contract/test_deepseek_v4_flash_decode_mtp.py`:

```python
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-flash"
sys.path.insert(0, str(_MODEL_DIR))


def test_handoff_packs_both_acceptance_paths_and_updates_tails() -> None:
    module = importlib.import_module("decode_input_pack")
    golden = getattr(module, "golden_pack_mtp_inputs")
    sampled = torch.full((8, 8), -777, dtype=torch.int32)
    sampled[:, 0] = torch.tensor(
        [100, 101, 200, 201, 300, 301, 400, 401],
        dtype=torch.int32,
    )
    tensors = {
        "main_sampled_ids": sampled,
        "main_position_ids": torch.tensor(
            [10, 11, 20, 21, 30, 31, 40, 41],
            dtype=torch.int32,
        ),
        "accepted_counts": torch.tensor([1, 2, 1, 2], dtype=torch.int32),
        "tail_slot_ids": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "tail_token_pool": torch.tensor([900, 901, 902, 903], dtype=torch.int64),
        "tail_position_pool": torch.tensor([9, 19, 29, 39], dtype=torch.int32),
        "mtp_input_ids": torch.zeros(8, dtype=torch.int64),
        "mtp_position_ids": torch.zeros(8, dtype=torch.int32),
    }

    golden(tensors)

    assert tensors["mtp_input_ids"].tolist() == [
        900, 100, 200, 201, 902, 300, 400, 401,
    ]
    assert tensors["mtp_position_ids"].tolist() == [
        9, 10, 20, 21, 29, 30, 40, 41,
    ]
    assert tensors["tail_token_pool"].tolist() == [100, 201, 300, 401]
    assert tensors["tail_position_pool"].tolist() == [10, 21, 30, 41]


def test_handoff_device_entry_has_the_stateful_contract() -> None:
    module = importlib.import_module("decode_input_pack")
    function = getattr(module, "pack_mtp_inputs")
    assert list(inspect.signature(function._func).parameters) == [
        "main_sampled_ids",
        "main_position_ids",
        "accepted_counts",
        "tail_slot_ids",
        "tail_token_pool",
        "tail_position_pool",
        "mtp_input_ids",
        "mtp_position_ids",
    ]


@pytest.mark.parametrize(
    ("accepted_counts", "tail_slot_ids", "sampled_token", "tail_token"),
    [
        ([0, 2, 1, 2], [0, 1, 2, 3], 100, 900),
        ([1, 2, 1, 2], [0, 1, 1, 3], 100, 900),
        ([1, 2, 1, 2], [0, 1, 2, 4], 100, 900),
        ([1, 2, 1, 2], [0, 1, 2, 3], -1, 900),
        ([1, 2, 1, 2], [0, 1, 2, 3], 100, 129280),
    ],
)
def test_handoff_fixture_rejects_invalid_metadata(
    accepted_counts,
    tail_slot_ids,
    sampled_token,
    tail_token,
) -> None:
    module = importlib.import_module("decode_input_pack")
    sampled = torch.full((8, 8), sampled_token, dtype=torch.int32)
    tails = torch.full((4,), tail_token, dtype=torch.int64)
    with pytest.raises(ValueError):
        module.validate_handoff_fixture(
            torch.tensor(accepted_counts, dtype=torch.int32),
            torch.tensor(tail_slot_ids, dtype=torch.int32),
            sampled,
            tails,
        )
```

- [ ] **Step 2: Run the test and verify the missing feature is the failure**

Run:

```bash
python -m pytest \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py::test_handoff_packs_both_acceptance_paths_and_updates_tails \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py::test_handoff_device_entry_has_the_stateful_contract \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py::test_handoff_fixture_rejects_invalid_metadata \
  -v
```

Expected: the tests fail because `golden_pack_mtp_inputs`, `pack_mtp_inputs`,
and `validate_handoff_fixture` do not exist.

- [ ] **Step 3: Implement the handoff kernel**

Add `SAMPLED_IDS_PAD` from `lm_head` and the following entry to
`decode_input_pack.py`:

```python
@pl.jit
def pack_mtp_inputs(
    main_sampled_ids: pl.Tensor[[DECODE_TOKENS, SAMPLED_IDS_PAD], pl.INT32],
    main_position_ids: pl.Tensor[[DECODE_TOKENS], pl.INT32],
    accepted_counts: pl.Tensor[[DECODE_BATCH], pl.INT32],
    tail_slot_ids: pl.Tensor[[DECODE_BATCH], pl.INT32],
    tail_token_pool: pl.InOut[pl.Tensor[[DECODE_BATCH], pl.INT64]],
    tail_position_pool: pl.InOut[pl.Tensor[[DECODE_BATCH], pl.INT32]],
    mtp_input_ids: pl.Out[pl.Tensor[[DECODE_TOKENS], pl.INT64]],
    mtp_position_ids: pl.Out[pl.Tensor[[DECODE_TOKENS], pl.INT32]],
):
    for batch_idx in pl.spmd(
        DECODE_BATCH,
        name_hint="pack_mtp_inputs",
    ):
        row0 = batch_idx * DECODE_SEQ
        row1 = row0 + 1
        slot = pl.cast(
            pl.read(tail_slot_ids, [batch_idx]),
            target_type=pl.INDEX,
        )
        accepted_count = pl.read(accepted_counts, [batch_idx])
        sampled0 = pl.cast(
            pl.read(main_sampled_ids, [row0, 0]),
            target_type=pl.INT64,
        )
        sampled1 = pl.cast(
            pl.read(main_sampled_ids, [row1, 0]),
            target_type=pl.INT64,
        )
        position0 = pl.read(main_position_ids, [row0])
        position1 = pl.read(main_position_ids, [row1])
        if accepted_count == 1:
            pl.write(
                mtp_input_ids,
                [row0],
                pl.read(tail_token_pool, [slot]),
            )
            pl.write(
                mtp_position_ids,
                [row0],
                pl.read(tail_position_pool, [slot]),
            )
            pl.write(mtp_input_ids, [row1], sampled0)
            pl.write(mtp_position_ids, [row1], position0)
            pl.write(tail_token_pool, [slot], sampled0)
            pl.write(tail_position_pool, [slot], position0)
        else:
            pl.write(mtp_input_ids, [row0], sampled0)
            pl.write(mtp_position_ids, [row0], position0)
            pl.write(mtp_input_ids, [row1], sampled1)
            pl.write(mtp_position_ids, [row1], position1)
            pl.write(tail_token_pool, [slot], sampled1)
            pl.write(tail_position_pool, [slot], position1)
    return mtp_input_ids, mtp_position_ids
```

Add `golden_pack_mtp_inputs(tensors)`, using the literal transition asserted
by the test. Add `validate_handoff_fixture(accepted_counts, tail_slot_ids,
main_sampled_ids, tail_token_pool)` that raises `ValueError` unless counts are
all 1 or 2, slots are unique and in `[0, DECODE_BATCH)`, and both sampled
column zero and tail tokens are in `[0, M.vocab_size)`.

Add `build_handoff_tensor_specs()` with the same literal fixture, mark both
tail pools as initialized outputs, and mark both MTP tensors as pure outputs.
Add a single-card CLI that calls:

```python
result = run_jit(
    fn=pack_mtp_inputs,
    specs=build_handoff_tensor_specs(),
    golden_fn=golden_pack_mtp_inputs,
    compile_only=args.compile_only,
    runtime_dir=args.runtime_dir,
    compile_cfg=dict(dump_passes=args.dump_passes),
    runtime_cfg=dict(
        platform=args.platform,
        device_id=args.device,
    ),
    rtol=0,
    atol=0,
)
```

Call `validate_handoff_fixture` before `run_jit`.

- [ ] **Step 4: Run the handoff tests and device/simulator validation**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
python models/deepseek/v4-flash/decode_input_pack.py \
  -p a2a3sim \
  -d 0
```

Expected: the contract tests pass and the harness reports exact golden
agreement. If this PyPTO revision cannot lower the distributed-free entry for
`a2a3sim`, run the same command with `-p a2a3` on device 0 and record the
simulator limitation in the task report.

- [ ] **Step 5: Commit the handoff**

```bash
git add \
  models/deepseek/v4-flash/decode_input_pack.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
git commit -m "Add: device-side DeepSeek decode-to-MTP handoff"
```

### Task 2: Trace-Hash Routing Without Model-Layer Corruption

**Files:**
- Create: `models/deepseek/v4-flash/decode_routing.py`
- Modify: `models/deepseek/v4-flash/decode_fwd.py`
- Modify: `models/deepseek/v4-flash/decode_mtp.py`
- Modify: `tests/contract/test_deepseek_v4_flash_decode_mtp.py`

**Interfaces:**
- Consumes: CLI routing name `"model"` or `"trace-hash"`, actual model layer
  IDs `0..43`, EP rank, embedding token ID, dedicated MTP routing token ID,
  route slot, and the fixed expert topology.
- Produces: `ROUTING_MODEL=0`, `ROUTING_TRACE_HASH=1`,
  `routing_mode_value(name)`, `resolve_routing_layer_id(layer_id, mode)`,
  the `build_trace_hash_tid2eid` fixture helper, and a trailing
  `routing_mode INT32` scalar
  on main/MTP child and standalone-host entries.

- [ ] **Step 1: Add failing routing tests**

Append:

```python
import ast


def _load_routing_module():
    return importlib.import_module("decode_routing")


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_node(function: ast.FunctionDef, name: str) -> ast.Call:
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def test_routing_modes_preserve_or_override_layer_identity() -> None:
    routing = _load_routing_module()
    actual_layers = list(range(44))
    assert [
        routing.resolve_routing_layer_id(layer_id, "model")
        for layer_id in actual_layers
    ] == actual_layers
    assert [
        routing.resolve_routing_layer_id(layer_id, "trace-hash")
        for layer_id in actual_layers
    ] == [0] * 44


def test_trace_hash_ep8_routes_balance_exactly() -> None:
    routing = _load_routing_module()
    table = routing.build_trace_hash_tid2eid(
        num_layers=1,
        first_layer_id=0,
        n_ranks=8,
        tokens_per_rank=8,
        vocab_size=32,
        topk=6,
        n_experts=128,
    )
    input_ids = torch.arange(8, dtype=torch.int64).expand(8, -1)
    active = torch.stack(
        [table[rank, input_ids[rank]] for rank in range(8)],
        dim=0,
    )
    counts = torch.bincount(active.reshape(-1).long(), minlength=128)
    assert counts.tolist() == [3] * 128
    assert counts.sum().item() == 8 * 8 * 6


def test_trace_hash_keeps_ep8_expert_dimensions() -> None:
    routing = _load_routing_module()
    assert routing.target_expert_topology(ep=8) == (128, 16)


def test_main_decode_keeps_actual_slice_ids_separate_from_routing_ids() -> None:
    path = _MODEL_DIR / "decode_fwd.py"
    function = _function_node(path, "decode_fwd")
    parameter_names = [arg.arg for arg in function.args.args]
    assert parameter_names[-1] == "routing_mode"
    source = ast.get_source_segment(path.read_text(), function)
    assert "csa_layer * N_EXPERTS_GLOBAL" in source
    assert "hca_layer * N_EXPERTS_GLOBAL" in source
    assert "csa_layer_last * N_EXPERTS_GLOBAL" in source
    assert "csa_routing_layer" in source
    assert "hca_routing_layer" in source
    assert "last_routing_layer" in source
    moe_calls = sorted(
        [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "moe"
        ],
        key=lambda node: node.lineno,
    )
    assert len(moe_calls) == 5
    assert [ast.unparse(call.args[-4]) for call in moe_calls] == [
        "pl.cast(0, pl.INT32)",
        "layer1_routing_layer",
        "csa_routing_layer",
        "hca_routing_layer",
        "last_routing_layer",
    ]


def test_mtp_moe_uses_a_separate_routing_identity() -> None:
    path = _MODEL_DIR / "decode_mtp.py"
    try:
        function = _function_node(path, "_mtp_decode_body")
    except StopIteration:
        function = _function_node(path, "mtp_decode_layer")
    moe_call = _call_node(function, "moe")
    assert ast.unparse(moe_call.args[8]) == "routing_input_ids"
    assert ast.unparse(moe_call.args[-4]) == "mtp_routing_layer"
    source = ast.get_source_segment(path.read_text(), function)
    assert "pl.cast(MTP_LAYER_ID, pl.INT32)" in source
    assert "routing_mode == ROUTING_TRACE_HASH" in source
```

- [ ] **Step 2: Run the routing tests and verify the expected failures**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py \
  -k "routing or expert_dimensions or actual_slice_ids" \
  -v
```

Expected: tests fail because `decode_routing.py`, the routing scalar, and
separate routing-layer variables do not exist.

- [ ] **Step 3: Implement pure routing helpers and fixture generation**

Create `decode_routing.py` with:

```python
ROUTING_MODEL = 0
ROUTING_TRACE_HASH = 1
ROUTING_MODES = {
    "model": ROUTING_MODEL,
    "trace-hash": ROUTING_TRACE_HASH,
}


def routing_mode_value(name):
    try:
        return ROUTING_MODES[name]
    except KeyError:
        raise ValueError(
            f"routing mode must be one of {sorted(ROUTING_MODES)}, got {name!r}"
        ) from None


def resolve_routing_layer_id(actual_layer_id, mode):
    mode_value = routing_mode_value(mode) if isinstance(mode, str) else mode
    if mode_value == ROUTING_TRACE_HASH:
        return 0
    if mode_value == ROUTING_MODEL:
        return actual_layer_id
    raise ValueError(f"unsupported routing mode value {mode_value}")


def target_expert_topology(ep):
    if ep != 8:
        raise ValueError(f"the comparison topology requires ep=8, got {ep}")
    return 128, 16


def build_trace_hash_tid2eid(
    *,
    num_layers,
    first_layer_id,
    n_ranks,
    tokens_per_rank,
    vocab_size,
    topk,
    n_experts,
):
    import torch

    rank = torch.arange(n_ranks, dtype=torch.int64).reshape(n_ranks, 1, 1)
    token = torch.arange(vocab_size, dtype=torch.int64).reshape(1, vocab_size, 1)
    route = torch.arange(topk, dtype=torch.int64).reshape(1, 1, topk)
    layers = []
    routes_per_layer = n_ranks * tokens_per_rank * topk
    for layer_id in range(first_layer_id, first_layer_id + num_layers):
        layer_routes = (
            layer_id * routes_per_layer
            + rank * (tokens_per_rank * topk)
            + token * topk
            + route
        ) % n_experts
        layers.append(layer_routes.to(torch.int32))
    return torch.cat(layers, dim=1).contiguous()
```

Use the standard repository copyright header and an English module docstring.

- [ ] **Step 4: Thread routing mode through main decode**

Add trailing `routing_mode: pl.Scalar[pl.INT32]` to `decode_fwd` and
`l3_decode_fwd`, and append `ScalarSpec("routing_mode", torch.int32,
routing_mode_value(routing_mode))` to `build_tensor_specs`.

At the five main MoE sites:

- Layer 0 keeps routing identity 0.
- Layer 1 derives `layer1_routing_layer` from actual identity 1 and changes it
  to 0 only in trace-hash mode.
- Each loop iteration derives `csa_routing_layer` and `hca_routing_layer`
  from the actual layer, changing each to 0 only when
  `routing_mode == ROUTING_TRACE_HASH`.
- The final layer keeps `csa_layer_last=42` for every slice and derives a
  separate `last_routing_layer`, changing only that scalar to 0.

Pass the derived identity only to `moe`'s existing `layer_id` position. Keep
all slice offsets and `moe_epoch` values unchanged.

Add a keyword-only `routing_mode="model"` argument to `build_tensor_specs`.
In trace-hash mode,
initialize the 43-layer `tid2eid` with `build_trace_hash_tid2eid` and initialize
main `input_ids` as rank-stacked `torch.arange(T)`. In model mode, retain the
current initializers. Add:

```python
parser.add_argument(
    "--routing-mode",
    choices=sorted(ROUTING_MODES),
    default="model",
)
```

Pass `args.routing_mode` into `build_tensor_specs`.

- [ ] **Step 5: Thread routing mode through standalone MTP**

Add `routing_input_ids: pl.Tensor[[T], pl.INT64]` immediately after the
embedding `input_ids` parameter of `mtp_decode_layer`, and add its
rank-stacked form at the same position in `l3_mtp_decode_layer`. Pass
`routing_input_ids`, not the embedding IDs, to `moe`.

Add trailing `routing_mode: pl.Scalar[pl.INT32]` to `mtp_decode_layer` and
`l3_mtp_decode_layer`. Immediately before the MTP MoE call:

```python
mtp_routing_layer = pl.cast(MTP_LAYER_ID, pl.INT32)
if routing_mode == ROUTING_TRACE_HASH:
    mtp_routing_layer = pl.cast(0, pl.INT32)
```

Pass `mtp_routing_layer` to `moe`; keep `MTP_LAYER_ID=43` for weight fixtures,
cache identity, and golden model behavior. Extend
`build_tensor_specs` with a keyword-only `routing_mode="model"` argument.
Only in trace-hash mode, initialize MTP `tid2eid` by calling
`build_trace_hash_tid2eid` with `num_layers=1`, `first_layer_id=43`, and the
existing rank, token, vocabulary, top-k, and expert constants. Initialize
`routing_input_ids` as rank-stacked `torch.arange(T)` in both modes; actual
MTP `input_ids` remain the standalone embedding fixture. Append the routing
`ScalarSpec`, and use
`resolve_routing_layer_id` in `golden_mtp_decode_layer`.

Add the same `--routing-mode` CLI contract as main decode and pass it into the
spec builder.

- [ ] **Step 6: Run routing regression checks**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
python -m pytest tests/golden -q
ruff check \
  models/deepseek/v4-flash/decode_routing.py \
  models/deepseek/v4-flash/decode_fwd.py \
  models/deepseek/v4-flash/decode_mtp.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
```

Expected: all contract tests pass, all 169 golden-harness tests pass, and ruff
reports no findings.

- [ ] **Step 7: Commit routing mode**

```bash
git add \
  models/deepseek/v4-flash/decode_routing.py \
  models/deepseek/v4-flash/decode_fwd.py \
  models/deepseek/v4-flash/decode_mtp.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
git commit -m "Add: trace-hash routing for DeepSeek decode"
```

### Task 3: Logits-Only MTP Child

**Files:**
- Modify: `models/deepseek/v4-flash/decode_mtp.py`
- Modify: `tests/contract/test_deepseek_v4_flash_decode_mtp.py`

**Interfaces:**
- Consumes: the existing MTP tensors and communication windows plus
  the separate MTP `routing_input_ids`, `num_tokens`, and `routing_mode`.
- Produces: shared `_mtp_decode_body`, unchanged sampled standalone
  `mtp_decode_layer`, and new `mtp_decode_layer_logits` with no sampled-ID
  parameter or greedy-sampling task.

- [ ] **Step 1: Add failing endpoint tests**

Append:

```python
def _called_function_names(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return names


def test_mtp_logits_child_has_no_sampled_ids_or_sampler() -> None:
    path = _MODEL_DIR / "decode_mtp.py"
    function = _function_node(path, "mtp_decode_layer_logits")
    parameters = [arg.arg for arg in function.args.args]
    calls = _called_function_names(function)
    assert "sampled_ids" not in parameters
    assert "lm_head" in calls
    assert "lm_head_with_sampling" not in calls
    assert "greedy_sample" not in calls


def test_standalone_mtp_keeps_sampling() -> None:
    path = _MODEL_DIR / "decode_mtp.py"
    function = _function_node(path, "mtp_decode_layer")
    parameters = [arg.arg for arg in function.args.args]
    calls = _called_function_names(function)
    assert "sampled_ids" in parameters
    assert "lm_head_with_sampling" in calls
```

- [ ] **Step 2: Run the endpoint tests and verify the missing child fails**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py \
  -k "mtp_logits_child or standalone_mtp" \
  -v
```

Expected: the logits-child test fails because `mtp_decode_layer_logits` does
not exist; the standalone-sampling test passes.

- [ ] **Step 3: Extract the shared MTP body**

Import `lm_head` alongside `lm_head_with_sampling`. Extract the current body
from embedding through final RMSNorm into a new
`@pl.jit.inline(auto_scope=False)` function named `_mtp_decode_body`.

Construct its complete typed parameter list from the existing
`mtp_decode_layer` declarations from `embed_weight` through `mtp_norm_w`,
preserving names, shapes, dtypes, and order while removing every `pl.InOut`
or `pl.Out` wrapper. Every tensor parameter of an inline function uses bare
`pl.Tensor`. Then append these declarations in this exact order:

```python
hidden_out: pl.Tensor[[T, D], pl.BF16],
next_pre_hc_hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
my_rank: pl.Scalar[pl.INT32],
num_tokens: pl.Scalar[pl.INT32],
routing_mode: pl.Scalar[pl.INT32],
```

Move the existing statements beginning with creation of `hidden_states` and
ending with `rms_norm(x_head, mtp_norm_w, hidden_out)` into the helper,
preserving their order and shapes. Retain the Task 2 `routing_input_ids` and
`mtp_routing_layer` separation at the MoE call. Return
`hidden_out, next_pre_hc_hidden`.

Do not include LM-head weights, logit-row indices, logits, sampled IDs, or
LM-head windows in the common body.

- [ ] **Step 4: Rebuild sampled and logits-only opaque entries**

Keep `mtp_decode_layer`'s current typed signature and standalone behavior. It
calls `_mtp_decode_body`, then calls `lm_head_with_sampling` with the existing
LM-head windows and sampled output.

Add `mtp_decode_layer_logits` with the same typed model/body arguments, omit
`sampled_ids`, and finish with:

```python
with pl.scope():
    lm_head(
        hidden_out,
        lm_head_weight,
        logit_row_indices,
        logits,
        lm_head_hidden_window,
        lm_head_hidden_done,
        lm_head_logits_window,
        lm_head_logits_done,
        my_rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,
        my_rank % LM_HEAD_TP_SIZE,
        LM_HEAD_COMM_EPOCH,
    )
return logits
```

Annotate the sampled entry return as `[T, D] BF16` and the logits-only entry
return as `[MAX_LOGIT_ROWS, LM_HEAD_VOCAB] FP32`. Keep the standalone host
calling the sampled entry.

- [ ] **Step 5: Run endpoint and standalone regressions**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
python -m pytest tests/golden -q
ruff check \
  models/deepseek/v4-flash/decode_mtp.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
```

Expected: endpoint contracts pass, all 169 golden-harness tests pass, and ruff
reports no findings.

- [ ] **Step 6: Commit the logits-only child**

```bash
git add \
  models/deepseek/v4-flash/decode_mtp.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
git commit -m "Add: logits-only DeepSeek MTP decode child"
```

### Task 4: Composite Decode-to-MTP L3 Driver

**Files:**
- Create: `models/deepseek/v4-flash/decode_fwd_mtp.py`
- Modify: `tests/contract/test_deepseek_v4_flash_decode_mtp.py`

**Interfaces:**
- Consumes: all 80 tensor parameters of `l3_decode_fwd`, 54 MTP-only tensors,
  four resident handoff tensors, `num_tokens INT32`, and
  `routing_mode INT32`, for 140 host parameters total.
- Produces: `l3_decode_fwd_mtp`, `build_tensor_specs`, topology validation,
  one CLI invocation, main outputs, handoff state, and full MTP logits.

The MTP child mappings are exact:

```text
embed_weight              <- embed_weight
main_pre_hc_hidden         <- pre_hc_hidden_out
position_ids               <- mtp_position_ids
freqs_cos                  <- freqs_cos
freqs_sin                  <- freqs_sin
input_ids                  <- mtp_input_ids
routing_input_ids          <- input_ids
lm_head_weight             <- lm_head_weight
sampled_ids                <- omitted
norm_w                     <- mtp_moe_norm_w
mtp_hc_head_fn             <- mtp_hc_head_fn
mtp_hc_head_scale          <- mtp_hc_head_scale
mtp_hc_head_base           <- mtp_hc_head_base
mtp_norm_w                 <- mtp_final_norm_w
hidden_out                 <- mtp_hidden_out
next_pre_hc_hidden         <- mtp_next_pre_hc_hidden
logits                     <- mtp_logits
logit_row_indices          <- mtp_logit_row_indices
all other MTP-only tensors <- the original name with one mtp_ prefix
```

- [ ] **Step 1: Add failing composite contracts**

Append:

```python
def test_composite_dispatches_decode_handoff_then_logits_mtp() -> None:
    path = _MODEL_DIR / "decode_fwd_mtp.py"
    function = _function_node(path, "l3_decode_fwd_mtp")
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {
            "decode_fwd",
            "pack_mtp_inputs",
            "mtp_decode_layer_logits",
        }
    ]
    ordered = [name for _, name in sorted(calls)]
    assert ordered == [
        "decode_fwd",
        "pack_mtp_inputs",
        "mtp_decode_layer_logits",
    ]


def test_composite_has_no_mtp_sampled_output() -> None:
    path = _MODEL_DIR / "decode_fwd_mtp.py"
    function = _function_node(path, "l3_decode_fwd_mtp")
    parameters = [arg.arg for arg in function.args.args]
    assert "sampled_ids" in parameters
    assert "mtp_sampled_ids" not in parameters
    assert "mtp_logits" in parameters


def test_composite_signature_deduplicates_shared_tensors() -> None:
    path = _MODEL_DIR / "decode_fwd_mtp.py"
    function = _function_node(path, "l3_decode_fwd_mtp")
    parameters = [arg.arg for arg in function.args.args]
    assert len(parameters) == 140
    assert len(set(parameters)) == 140
    assert parameters[-2:] == ["num_tokens", "routing_mode"]
    for omitted in [
        "mtp_embed_weight",
        "mtp_main_pre_hc_hidden",
        "mtp_freqs_cos",
        "mtp_freqs_sin",
        "mtp_lm_head_weight",
        "mtp_routing_input_ids",
        "mtp_sampled_ids",
    ]:
        assert omitted not in parameters
    for required in [
        "mtp_tail_token_pool",
        "mtp_tail_position_pool",
        "mtp_input_ids",
        "mtp_position_ids",
        "mtp_moe_norm_w",
        "mtp_final_norm_w",
        "mtp_hidden_out",
        "mtp_next_pre_hc_hidden",
        "mtp_logits",
        "mtp_logit_row_indices",
    ]:
        assert required in parameters


def test_composite_wires_shared_inputs_and_disjoint_windows() -> None:
    path = _MODEL_DIR / "decode_fwd_mtp.py"
    host = _function_node(path, "l3_decode_fwd_mtp")
    handoff_call = _call_node(host, "pack_mtp_inputs")
    assert [ast.unparse(arg) for arg in handoff_call.args] == [
        "sampled_ids[r]",
        "position_ids[r]",
        "mtp_accepted_counts[r]",
        "mtp_tail_slot_ids[r]",
        "mtp_tail_token_pool[r]",
        "mtp_tail_position_pool[r]",
        "mtp_input_ids[r]",
        "mtp_position_ids[r]",
    ]

    main_child = _function_node(_MODEL_DIR / "decode_fwd.py", "decode_fwd")
    main_call = _call_node(host, "decode_fwd")
    assert len(main_call.args) == len(main_child.args.args)
    main_mapping = {
        parameter.arg: ast.unparse(argument)
        for parameter, argument in zip(
            main_child.args.args,
            main_call.args,
            strict=True,
        )
    }

    mtp_child = _function_node(
        _MODEL_DIR / "decode_mtp.py",
        "mtp_decode_layer_logits",
    )
    mtp_call = _call_node(host, "mtp_decode_layer_logits")
    assert len(mtp_call.args) == len(mtp_child.args.args)
    mtp_mapping = {
        parameter.arg: ast.unparse(argument)
        for parameter, argument in zip(
            mtp_child.args.args,
            mtp_call.args,
            strict=True,
        )
    }
    assert {
        name: mtp_mapping[name]
        for name in [
            "embed_weight",
            "main_pre_hc_hidden",
            "position_ids",
            "freqs_cos",
            "freqs_sin",
            "input_ids",
            "routing_input_ids",
            "lm_head_weight",
        ]
    } == {
        "embed_weight": "embed_weight[r]",
        "main_pre_hc_hidden": "pre_hc_hidden_out[r]",
        "position_ids": "mtp_position_ids[r]",
        "freqs_cos": "freqs_cos[r]",
        "freqs_sin": "freqs_sin[r]",
        "input_ids": "mtp_input_ids[r]",
        "routing_input_ids": "input_ids[r]",
        "lm_head_weight": "lm_head_weight[r]",
    }

    window_names = [
        "recv_meta",
        "recv_x",
        "recv_aux",
        "recv_route",
        "arrived",
        "data_arrived",
        "routed_y_buf",
        "combine_arrived",
        "lm_head_hidden_window",
        "lm_head_hidden_done",
        "lm_head_logits_window",
        "lm_head_logits_done",
    ]
    for name in window_names:
        assert main_mapping[name] == f"main_{name}"
        assert mtp_mapping[name] == f"mtp_{name}"


def test_composite_topology_validation_targets_ep8_tp4() -> None:
    module = importlib.import_module("decode_fwd_mtp")
    module.validate_topology(ep=8, tp=4, device_ids=list(range(8)))
    for ep, tp, devices in [
        (4, 4, list(range(4))),
        (8, 8, list(range(8))),
        (8, 4, list(range(7))),
    ]:
        try:
            module.validate_topology(ep=ep, tp=tp, device_ids=devices)
        except ValueError:
            continue
        raise AssertionError((ep, tp, devices))


def test_composite_finite_oracle_rejects_invalid_logits() -> None:
    module = importlib.import_module("decode_fwd_mtp")
    expected = torch.zeros(2, dtype=torch.float32)
    assert module.finite_tensor_compare(
        torch.tensor([1.0, -2.0]),
        expected,
    )[0]
    ok, detail = module.finite_tensor_compare(
        torch.tensor([1.0, float("nan")]),
        expected,
    )
    assert not ok
    assert "1/2" in detail
```

- [ ] **Step 2: Run the composite tests and verify the missing file fails**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py \
  -k "composite" \
  -v
```

Expected: tests fail because `decode_fwd_mtp.py` does not exist.

- [ ] **Step 3: Define the composite host signature and separate windows**

Create `decode_fwd_mtp.py` with the repository copyright header,
`# ci: devices=8`, `# ci: no-sim`, and an English module docstring.

Define `l3_decode_fwd_mtp` with:

1. The exact 80 tensor parameters from `l3_decode_fwd`, preserving names,
   annotations, and order.
2. `mtp_tail_token_pool [N_RANKS, B] INT64` and
   `mtp_tail_position_pool [N_RANKS, B] INT32` as `pl.InOut`.
3. `mtp_input_ids [N_RANKS, T] INT64` and
   `mtp_position_ids [N_RANKS, T] INT32` as `pl.Out`.
4. The exact 54 MTP-only tensor parameters in original order, using the
   explicit override table below and otherwise adding one `mtp_` prefix.
5. Trailing `num_tokens: pl.Scalar[pl.INT32]` and
   `routing_mode: pl.Scalar[pl.INT32]`.

Allocate a complete main set of MoE and LM-head window buffers named
`main_recv_meta` through `main_lm_head_logits_done`, and a second complete MTP
set named `mtp_recv_meta` through `mtp_lm_head_logits_done`. Use the twelve
suffixes asserted by the contract test. Do not alias buffers or completion
counters between stages.

- [ ] **Step 4: Submit the three ordered phases**

First loop over `pld.world_size()` and call `decode_fwd` with the main tensors,
main windows, rank, and `routing_mode`.

Second loop and call:

```python
pack_mtp_inputs(
    sampled_ids[r],
    position_ids[r],
    mtp_accepted_counts[r],
    mtp_tail_slot_ids[r],
    mtp_tail_token_pool[r],
    mtp_tail_position_pool[r],
    mtp_input_ids[r],
    mtp_position_ids[r],
    device=r,
)
```

Third loop and call `mtp_decode_layer_logits`, mapping the eight shared or
produced inputs exactly as specified in the interface block. Pass only MTP
windows, rank, `num_tokens`, and `routing_mode`.

- [ ] **Step 5: Compose tensor specs without duplicating shared weights**

Implement:

```python
MTP_REUSED_NAMES = {
    "embed_weight",
    "main_pre_hc_hidden",
    "position_ids",
    "freqs_cos",
    "freqs_sin",
    "input_ids",
    "routing_input_ids",
    "lm_head_weight",
    "sampled_ids",
}

MTP_NAME_OVERRIDES = {
    "norm_w": "mtp_moe_norm_w",
    "mtp_hc_head_fn": "mtp_hc_head_fn",
    "mtp_hc_head_scale": "mtp_hc_head_scale",
    "mtp_hc_head_base": "mtp_hc_head_base",
    "mtp_norm_w": "mtp_final_norm_w",
    "hidden_out": "mtp_hidden_out",
    "next_pre_hc_hidden": "mtp_next_pre_hc_hidden",
    "logits": "mtp_logits",
    "logit_row_indices": "mtp_logit_row_indices",
}


def _mtp_spec_name(name):
    return MTP_NAME_OVERRIDES.get(name, f"mtp_{name}")


def _rename_mtp_spec(spec):
    import dataclasses

    return dataclasses.replace(spec, name=_mtp_spec_name(spec.name))
```

`build_tensor_specs` obtains standalone main and MTP specs using identical
start position, token count, cache sizes, and routing mode. Keep all main
tensor specs, remove their trailing routing scalar, add the four handoff
specs, then add every MTP tensor spec not in `MTP_REUSED_NAMES` through
`_rename_mtp_spec`. Append one `num_tokens` scalar and one `routing_mode`
scalar, then assert that the resulting names are unique and that the
composite list has 140 entries.

Use initial handoff fixtures:

```python
accepted_counts = torch.tensor([1, 2, 1, 2], dtype=torch.int32)
tail_slot_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
tail_token_pool = torch.arange(B, dtype=torch.int64)
tail_position_pool = torch.arange(B, dtype=torch.int32) + start_pos - 1
```

Expand each across `N_RANKS`. Mark main `pre_hc_hidden_out`, main
`sampled_ids`, both handoff pools, both handoff outputs, and
`mtp_tail_pre_hc_pool` as `resident="stacked"`. Preserve resident flags from
renamed MTP weights and caches. Do not create second embedding, RoPE, or
LM-head weight specs. For the composite smoke, clear inherited `is_output`
flags and set only `mtp_logits.is_output=True`; keep `mtp_logits` resident and
stacked. Intermediate outputs and mutable caches remain device-resident but do
not incur full readback.

- [ ] **Step 6: Add strict CLI validation and launch**

Implement:

```python
def validate_topology(*, ep, tp, device_ids):
    if ep != 8:
        raise ValueError(f"comparison requires ep=8, got {ep}")
    if tp != 4:
        raise ValueError(f"comparison requires tp=4, got {tp}")
    if ep % tp != 0:
        raise ValueError(f"ep must be divisible by tp, got ep={ep}, tp={tp}")
    if len(device_ids) < ep:
        raise ValueError(f"need at least {ep} devices, got {device_ids}")
```

Expose the standalone drivers' platform, device, cache-size, start-position,
compile-only, runtime-directory, pass-dump, and DFX controls. Add
`--routing-mode {model,trace-hash}`.

Add the finite-only smoke oracle:

```python
def golden_finite_smoke(_tensors):
    return None


def finite_tensor_compare(actual, _expected, **_context):
    finite = torch.isfinite(actual)
    if bool(finite.all()):
        return True, ""
    invalid = int((~finite).sum().item())
    return False, f"{invalid}/{actual.numel()} values are non-finite"
```

Call `run_jit` once with `fn=l3_decode_fwd_mtp`,
`golden_fn=golden_finite_smoke`,
`compare_fn={"mtp_logits": finite_tensor_compare}`, and an eight-device
`DistributedConfig`. This makes the run fail on non-finite MTP logits instead
of reporting validation as skipped.

- [ ] **Step 7: Run contract, lint, and signature checks**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
python -m pytest tests/golden -q
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
ruff check \
  models/deepseek/v4-flash/decode_input_pack.py \
  models/deepseek/v4-flash/decode_routing.py \
  models/deepseek/v4-flash/decode_fwd.py \
  models/deepseek/v4-flash/decode_mtp.py \
  models/deepseek/v4-flash/decode_fwd_mtp.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
```

Expected: contract tests and all 169 golden-harness tests pass; lint checks
report no findings.

- [ ] **Step 8: Compile and smoke the target graph**

Run a compile-only target first:

```bash
python models/deepseek/v4-flash/decode_fwd_mtp.py \
  -p a2a3 \
  --ep 8 \
  --tp 4 \
  --routing-mode trace-hash \
  --start-pos 8192 \
  --num-tokens 8 \
  -d 0,1,2,3,4,5,6,7 \
  --compile-only
```

Then run one correctness/smoke dispatch:

```bash
python models/deepseek/v4-flash/decode_fwd_mtp.py \
  -p a2a3 \
  --ep 8 \
  --tp 4 \
  --routing-mode trace-hash \
  --start-pos 8192 \
  --num-tokens 8 \
  -d 0,1,2,3,4,5,6,7
```

Expected: compilation succeeds, all eight rank children dispatch, the finite
check for `mtp_logits` reports PASS, and no generated artifact is staged.

- [ ] **Step 9: Commit the composite driver**

```bash
git add \
  models/deepseek/v4-flash/decode_fwd_mtp.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
git commit -m "Add: DeepSeek V4 decode-to-MTP benchmark"
```

### Task 5: Align the Comparison Expert Topology

**Files:**
- Modify: `models/deepseek/v4-flash/moe.py`
- Modify: `models/deepseek/v4-flash/gate.py`
- Modify: `models/deepseek/v4-flash/decode_routing.py`
- Modify: `tests/contract/test_deepseek_v4_flash_decode_mtp.py`

**Interfaces:**
- Consumes: the complete composite benchmark at commit `e820390` and the
  approved compare2 expert-scaling formula.
- Produces: one shared EP8 topology of 128 global routed experts and 16 local
  experts for both main decode and MTP, with exactly three active trace-hash
  routes per global expert.

- [ ] **Step 1: Write the failing topology contracts**

Add `os` and `subprocess` to the contract-test imports. Replace the two main
trace-hash topology tests and the MTP trace-hash count assertions with:

```python
def test_trace_hash_ep8_routes_balance_exactly() -> None:
    routing = _load_routing_module()
    table = routing.build_trace_hash_tid2eid(
        num_layers=1,
        first_layer_id=0,
        n_ranks=8,
        tokens_per_rank=8,
        vocab_size=32,
        topk=6,
        n_experts=128,
    )
    input_ids = torch.arange(8, dtype=torch.int64).expand(8, -1)
    active = torch.stack(
        [table[rank, input_ids[rank]] for rank in range(8)],
        dim=0,
    )
    counts = torch.bincount(active.reshape(-1).long(), minlength=128)
    assert counts.tolist() == [3] * 128


def test_trace_hash_keeps_ep8_expert_dimensions() -> None:
    routing = _load_routing_module()
    assert routing.target_expert_topology(ep=8) == (128, 16)


def test_ep8_runtime_topology_uses_128_global_and_16_local() -> None:
    env = os.environ.copy()
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_MODEL_DIR)
    if inherited_pythonpath:
        env["PYTHONPATH"] += os.pathsep + inherited_pythonpath
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import moe; print(moe.N_RANKS, moe.N_EXPERTS_GLOBAL, moe.N_LOCAL)",
            "--ep",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "8 128 16" in result.stdout.splitlines()
```

In `test_mtp_trace_routing_fixture_balances_independently`, pass
`n_experts=128`, use `minlength=128`, and replace the min/max assertions with:

```python
assert counts.tolist() == [3] * 128
```

- [ ] **Step 2: Run the topology contracts and verify RED**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py \
  -k "routes_balance_exactly or expert_dimensions or ep8_runtime_topology or mtp_trace_routing_fixture" \
  -v
```

Expected: the pure routing-balance tests pass, while
`expert_dimensions` and `ep8_runtime_topology` fail with the current
256-global / 32-local values.

- [ ] **Step 3: Preserve the compare4 boundary on the compare2 base**

In `moe.py`, update the module docstring to say every rank keeps 16 experts
and change only the import-time expert scaling:

```python
config.FLASH = dataclasses.replace(config.FLASH, n_routed_experts=config.FLASH.n_routed_experts // 16 * EP)
```

In `gate.py`, update the routing-space comment from `32*EP` to `16*EP`.

The branch is based on compare2, but the compare4 change restores device-side
input packing and the complete main tail. Signal-clear and daily-runner changes
remain inherited baseline behavior outside the compare4 feature delta.

In `decode_routing.py`, change the fixed comparison topology:

```python
def target_expert_topology(ep):
    if ep != 8:
        raise ValueError(f"the comparison topology requires ep=8, got {ep}")
    return 128, 16
```

- [ ] **Step 4: Run the focused contracts and verify GREEN**

Run the Step 2 command again.

Expected: all four selected tests pass, and the subprocess reports
`8 128 16`.

- [ ] **Step 5: Run topology and repository regressions**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
python -m pytest tests/golden -q
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
ruff check \
  models/deepseek/v4-flash/moe.py \
  models/deepseek/v4-flash/decode_routing.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
```

Expected: 24 contract tests and all 169 golden-harness tests pass; lint checks
report no findings.

- [ ] **Step 6: Compile the corrected full composite**

Run:

```bash
python models/deepseek/v4-flash/decode_fwd_mtp.py \
  -p a2a3 \
  --ep 8 \
  --tp 4 \
  --routing-mode trace-hash \
  --start-pos 8192 \
  --num-tokens 8 \
  -d 0,1,2,3,4,5,6,7 \
  --compile-only
```

Expected: compilation succeeds with `N_EXPERTS_GLOBAL=128`,
`N_LOCAL=16`, and no generated artifact is staged. The old 256/32 runtime
directory must not be reused for device timing.

- [ ] **Step 7: Commit the topology correction**

```bash
git add \
  models/deepseek/v4-flash/moe.py \
  models/deepseek/v4-flash/gate.py \
  models/deepseek/v4-flash/decode_routing.py \
  tests/contract/test_deepseek_v4_flash_decode_mtp.py
git commit -m "Fix: align DeepSeek comparison expert topology"
```

### Task 6: Final Verification and Performance Capture

**Files:**
- Verify: all branch changes
- Do not add: `build_output/`, trace JSON, logs, generated tensor data

**Interfaces:**
- Consumes: the corrected 128-global / 16-local composite executable and the
  AscendC reference interval.
- Produces: verified repository state plus PyPTO device timing evidence.

- [ ] **Step 1: Run repository verification from a clean shell**

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
python -m pytest tests/golden -v
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
ruff check .
pre-commit run --all-files
```

- [ ] **Step 2: Run a short EP8/TP4 benchmark**

```bash
PYPTO_BENCH=1 \
PYPTO_BENCH_ROUNDS=3 \
PYPTO_BENCH_WARMUP=1 \
PYPTO_BENCH_RAW=1 \
python models/deepseek/v4-flash/decode_fwd_mtp.py \
  -p a2a3 \
  --ep 8 \
  --tp 4 \
  --routing-mode trace-hash \
  --start-pos 8192 \
  --num-tokens 8 \
  -d 0,1,2,3,4,5,6,7
```

Record the headline `effective_us`, rank/slot breakdown, and
`host_union_mean_us` in the final task report. Do not present `effective_us` as
the exact first-embedding-to-final-logits system envelope. Reject any runtime
directory compiled before Task 5 because it has the old 256/32 shapes.

- [ ] **Step 3: Capture the exact comparison trace**

Capture a CANN system trace without L2 swimlane. For every steady round:

1. Start at the first main task reading `input_ids` and `embed_weight` for
   `pack_x_hc`.
2. End at the last task writing assembled full-vocabulary `mtp_logits`.
3. Report NPU0 separately.
4. Report the maximum interval over all eight ranks.
5. Report median and mean over at least three rounds.

Label the PyPTO greedy-sampling versus AscendC stochastic-sampling semantic
difference in the report.

- [ ] **Step 4: Verify branch scope**

```bash
git status --short --branch
git diff upstream/dsv4-ascendc-decode-compare2...HEAD --check
git log --oneline --decorate upstream/dsv4-ascendc-decode-compare2..HEAD
git diff --stat upstream/dsv4-ascendc-decode-compare2...HEAD
```

Expected: source, tests, design, and plan are the only tracked branch changes;
generated artifacts remain ignored and unstaged.
