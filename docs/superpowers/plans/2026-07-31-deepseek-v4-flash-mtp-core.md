# DeepSeek V4 Flash MTP Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone PyPTO MTP-core benchmark whose measured child runs
only projection, SWA, MoE, HC head, RMSNorm, and LM-head logits at the target
EP8/TP4 and 128-global / 16-local expert topology.

**Architecture:** Split the LM-head signal clear from its compute body while
retaining the existing compatibility wrapper. Add a dedicated distributed
driver that consumes host-prepared hidden and SWA metadata tensors, submits one
compute child per rank, then submits a separate cleanup child per rank.

**Tech Stack:** Python 3.10, PyPTO JIT/DSL, PyTorch golden fixtures, pytest,
ruff, Ascend A3 distributed runtime, and CANN system trace.

## Global Constraints

- The measured child starts with `mtp_projection` and ends with the LM-head
  full-vocabulary logits gather.
- The measured child contains `MTP projection -> SWA -> MoE -> HC head ->
  RMSNorm -> LM head` in that order.
- Embedding, hidden packing, SWA metadata construction, sampling, and signal
  cleanup are outside the measured child.
- The cleanup child must run after all rank compute submissions and before the
  next graph execution can reuse the fixed communication epochs.
- The existing complete Decode-to-MTP and standalone sampled-MTP behavior must
  remain unchanged.
- The target comparison uses EP=8, TP=4, 128 global experts, 16 local experts
  per rank, T=8, top-k=6, start position 8192, and trace-hash routing.
- The strict AscendC reference is Model47 task 6 through task 66: mean
  2.235998 ms and median 2.260706 ms.
- System-trace and program-level timing are allowed; in-core profiling is out
  of scope.
- Generated `build_output/`, traces, logs, and generated tensors must not be
  committed.
- Code comments, docstrings, CLI text, tests, and commit messages are English.
- Run Python checks with the pinned source/build environment:

```bash
export PYPTO_ROOT=../pypto
export PYTHONPATH="$PYPTO_ROOT/python:$PYPTO_ROOT/build/python/bindings${PYTHONPATH:+:$PYTHONPATH}"
```

---

### Task 1: Split LM-Head Compute and Signal Cleanup

**Files:**
- Modify: `models/deepseek/v4-flash/lm_head.py`
- Create: `tests/contract/test_deepseek_v4_flash_decode_mtp_core.py`

**Interfaces:**
- Consumes: the existing `lm_head` signature and its four communication
  windows.
- Produces: `lm_head_core`, `clear_lm_head_signals`, and an unchanged
  `lm_head` compatibility contract.

- [ ] **Step 1: Write the failing LM-head boundary test**

Create the contract file with the repository header and these helpers and test:

```python
from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "models" / "deepseek" / "v4-flash"
sys.path.insert(0, str(_MODEL_DIR))


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _called_function_names(function: ast.FunctionDef) -> list[str]:
    return [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def test_lm_head_keeps_cleanup_out_of_the_reusable_core() -> None:
    path = _MODEL_DIR / "lm_head.py"
    core = _function_node(path, "lm_head_core")
    cleanup = _function_node(path, "clear_lm_head_signals")
    wrapper = _function_node(path, "lm_head")

    assert "clear_lm_head_signals" not in _called_function_names(core)
    core_write_destinations = {
        ast.unparse(node.args[0])
        for node in ast.walk(core)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
    }
    cleanup_write_destinations = {
        ast.unparse(node.args[0])
        for node in ast.walk(cleanup)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write"
    }
    assert core_write_destinations.isdisjoint({"hidden_done", "logits_done"})
    assert cleanup_write_destinations == {"hidden_done", "logits_done"}
    wrapper_calls = [
        name
        for name in _called_function_names(wrapper)
        if name in {"lm_head_core", "clear_lm_head_signals"}
    ]
    assert wrapper_calls == ["lm_head_core", "clear_lm_head_signals"]

    module = importlib.import_module("lm_head")
    assert inspect.signature(module.lm_head._func) == inspect.signature(module.lm_head_core._func)
    cleanup_parameters = tuple(inspect.signature(module.clear_lm_head_signals._func).parameters)
    assert cleanup_parameters == ("completion_anchor", "hidden_done", "logits_done")
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp_core.py::test_lm_head_keeps_cleanup_out_of_the_reusable_core -v
```

Expected: FAIL because `lm_head_core` and `clear_lm_head_signals` do not exist.

- [ ] **Step 3: Extract the LM-head body without changing existing behavior**

In `lm_head.py`, rename the current compute body to `lm_head_core` and end it
immediately after `lm_head_combine_gather`. Move the existing anchored clear
scope verbatim into:

```python
@pl.jit.inline
def clear_lm_head_signals(
    completion_anchor: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_signal_clear"):
        _completion_anchor = pl.read(completion_anchor, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return completion_anchor
```

Define `lm_head` with the original signature as a wrapper that calls
`lm_head_core`, then
`clear_lm_head_signals(logits, hidden_done, logits_done)`, and returns `logits`.
Do not change any existing caller.

- [ ] **Step 4: Run the focused and existing LM-head/decode-MTP contracts**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp_core.py -v
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
ruff check models/deepseek/v4-flash/lm_head.py tests/contract/test_deepseek_v4_flash_decode_mtp_core.py
```

Expected: all tests and lint pass.

- [ ] **Step 5: Commit the compatible LM-head split**

```bash
git add models/deepseek/v4-flash/lm_head.py tests/contract/test_deepseek_v4_flash_decode_mtp_core.py
git commit -m "Refactor: split LM-head compute and signal cleanup"
```

---

### Task 2: Add the Strict MTP-Core Benchmark

**Files:**
- Create: `models/deepseek/v4-flash/decode_mtp_core.py`
- Modify: `tests/contract/test_deepseek_v4_flash_decode_mtp_core.py`

**Interfaces:**
- Consumes: prepared `hidden_states`, `prev_pre_hc_hidden`,
  `swa_slot_mapping`, `swa_indices`, and `swa_lens`; existing model weights,
  caches, route tables, communication windows, position IDs, route IDs, and
  scalar values.
- Produces: `mtp_decode_core_logits`, `mtp_decode_core_cleanup`,
  `l3_mtp_decode_core`, `build_tensor_specs`, `golden_mtp_decode_core`, and the
  CLI driver.

- [ ] **Step 1: Write the failing strict-boundary tests**

Append these tests:

```python
def test_mtp_core_contains_only_the_requested_model_stages() -> None:
    path = _MODEL_DIR / "decode_mtp_core.py"
    function = _function_node(path, "mtp_decode_core_logits")
    module = importlib.import_module("decode_mtp_core")
    core_signature = inspect.signature(module.mtp_decode_core_logits._func)
    host_signature = inspect.signature(module.l3_mtp_decode_core._func)
    prepared = (
        "hidden_states", "prev_pre_hc_hidden", "swa_slot_mapping", "swa_indices", "swa_lens",
    )
    assert tuple(core_signature.parameters)[:5] == prepared
    assert tuple(host_signature.parameters)[:5] == prepared
    required = {"position_ids", "routing_input_ids", "hidden_out", "next_pre_hc_hidden", "logits"}
    assert required <= set(core_signature.parameters)
    assert required <= set(host_signature.parameters)
    forbidden = {
        "embed_weight", "main_pre_hc_hidden", "tail_pre_hc_pool",
        "accepted_counts", "tail_slot_ids", "input_ids", "ori_block_table", "sampled_ids",
    }
    assert forbidden.isdisjoint(core_signature.parameters)
    assert forbidden.isdisjoint(host_signature.parameters)
    expected_dtypes = ("bfloat16", "fp32", "int64", "int32", "int32")
    actual_dtypes = tuple(str(core_signature.parameters[name].annotation).split(", ")[-1][:-1] for name in prepared)
    assert actual_dtypes == expected_dtypes
    assert str(core_signature.parameters["logits"].annotation).endswith(", fp32]")

    tracked = {
        "mtp_projection", "attention_swa", "moe", "hc_head", "rms_norm",
        "lm_head_core", "lm_head", "lookup_embedding", "pack_mtp_hidden",
        "build_swa_metadata", "clear_moe_signals", "clear_lm_head_signals",
        "lm_head_with_sampling", "greedy_sample",
    }
    calls = sorted(
        (
            (node.lineno, node.col_offset, node.func.id)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in tracked
        ),
    )
    assert [name for _, _, name in calls] == [
        "mtp_projection", "attention_swa", "moe", "hc_head", "rms_norm", "lm_head_core",
    ]


def test_mtp_core_cleanup_is_a_separate_child() -> None:
    path = _MODEL_DIR / "decode_mtp_core.py"
    cleanup = _function_node(path, "mtp_decode_core_cleanup")
    module = importlib.import_module("decode_mtp_core")
    cleanup_parameters = tuple(inspect.signature(module.mtp_decode_core_cleanup._func).parameters)
    assert cleanup_parameters == (
        "next_pre_hc_hidden", "logits", "arrived", "data_arrived", "combine_arrived",
        "lm_head_hidden_done", "lm_head_logits_done",
    )
    cleanup_calls = sorted(
        (
            (node.lineno, node.col_offset, node.func.id)
            for node in ast.walk(cleanup)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"clear_moe_signals", "clear_lm_head_signals"}
        ),
    )
    assert [name for _, _, name in cleanup_calls] == ["clear_moe_signals", "clear_lm_head_signals"]

    host = _function_node(path, "l3_mtp_decode_core")
    rank_loops = [
        node
        for node in host.body
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "pl.range(pld.world_size())"
    ]
    assert len(rank_loops) == 2
    loop_dispatches = [
        [
            name
            for name in _called_function_names(loop)
            if name in {"mtp_decode_core_logits", "mtp_decode_core_cleanup"}
        ]
        for loop in rank_loops
    ]
    assert loop_dispatches == [["mtp_decode_core_logits"], ["mtp_decode_core_cleanup"]]

    main = _function_node(path, "main")
    run_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_jit"
    )
    run_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in run_call.keywords}
    assert run_keywords["fn"] == "l3_mtp_decode_core"
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp_core.py -v
```

Expected: the LM-head test passes and both new tests fail because
`decode_mtp_core.py` does not exist.

- [ ] **Step 3: Implement the measured compute and cleanup children**

Create `decode_mtp_core.py` with the repository header, `# ci: devices=2`, and
this module contract:

```python
"""DeepSeek-V4 MTP core: projection, SWA, MoE, HC head, RMSNorm, and LM-head logits."""
```

`mtp_decode_core_logits` must accept the prepared tensors listed in the task
interface, then the existing projection/attention/MoE/head/LM-head weights and
windows in model order, outputs `hidden_out`, `next_pre_hc_hidden`, and
`logits`, and the scalars `my_rank`, `num_tokens`, and `routing_mode` last. Its
body must execute exactly:

```python
projected_hidden = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
with pl.scope():
    mtp_projection(
        hidden_states, prev_pre_hc_hidden,
        enorm_w, hnorm_w,
        e_proj_w, e_proj_w_scale, e_proj_smooth,
        h_proj_w, h_proj_w_scale, h_proj_smooth,
        projected_hidden,
    )

x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
with pl.scope():
    attention_swa(
        projected_hidden,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w,
        wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache,
        swa_slot_mapping, swa_indices, swa_lens, position_ids,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_attn,
    )

mtp_routing_layer = pl.cast(MTP_LAYER_ID, pl.INT32)
if routing_mode == ROUTING_TRACE_HASH:
    mtp_routing_layer = pl.cast(0, pl.INT32)
with pl.scope():
    moe(
        x_attn,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, routing_input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        next_pre_hc_hidden,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        mtp_routing_layer, num_tokens, my_rank,
        pl.cast(MTP_MOE_EPOCH, pl.INT32),
    )

x_head = pl.create_tensor([T, D], dtype=pl.BF16)
with pl.scope():
    hc_head(next_pre_hc_hidden, mtp_hc_head_fn, mtp_hc_head_scale, mtp_hc_head_base, x_head)
    rms_norm(x_head, mtp_norm_w, hidden_out)

with pl.scope():
    lm_head_core(
        hidden_out, lm_head_weight, logit_row_indices, logits,
        lm_head_hidden_window, lm_head_hidden_done,
        lm_head_logits_window, lm_head_logits_done,
        my_rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,
        my_rank % LM_HEAD_TP_SIZE,
        LM_HEAD_COMM_EPOCH,
    )
return logits
```

Do not call a preparation, sampling, or clear function in this child.

`mtp_decode_core_cleanup` accepts `next_pre_hc_hidden`, `logits`, the three MoE
signal windows, and the two LM-head done windows. It calls
`clear_moe_signals`, then `clear_lm_head_signals`, and returns
`logits`.

- [ ] **Step 4: Implement the distributed host, specs, golden, and CLI**

`l3_mtp_decode_core` allocates the same eight MoE buffers and four LM-head
buffers used by `l3_mtp_decode_layer`. Use two separate rank loops:

```python
for r in pl.range(pld.world_size()):
    mtp_decode_core_logits(
        hidden_states[r], prev_pre_hc_hidden[r],
        swa_slot_mapping[r], swa_indices[r], swa_lens[r],
        position_ids[r],
        enorm_w[r], hnorm_w[r],
        e_proj_w[r], e_proj_w_scale[r], e_proj_smooth[r],
        h_proj_w[r], h_proj_w_scale[r], h_proj_smooth[r],
        hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r],
        attn_norm_w[r],
        wq_a[r], wq_b[r], wq_b_scale[r],
        wkv[r], gamma_cq[r], gamma_ckv[r],
        freqs_cos[r], freqs_sin[r],
        kv_cache[r],
        attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r],
        hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
        norm_w[r], gate_w[r], gate_bias[r], tid2eid[r], routing_input_ids[r],
        routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
        routed_w2[r], routed_w2_scale[r],
        shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
        shared_w2[r], shared_w2_scale[r],
        mtp_hc_head_fn[r], mtp_hc_head_scale[r], mtp_hc_head_base[r], mtp_norm_w[r],
        lm_head_weight[r], logit_row_indices[r],
        hidden_out[r], next_pre_hc_hidden[r], logits[r],
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        lm_head_hidden_window, lm_head_hidden_done,
        lm_head_logits_window, lm_head_logits_done,
        r, num_tokens, routing_mode,
        device=r,
    )

for r in pl.range(pld.world_size()):
    mtp_decode_core_cleanup(
        next_pre_hc_hidden[r], logits[r],
        arrived, data_arrived, combine_arrived,
        lm_head_hidden_done, lm_head_logits_done,
        device=r,
    )
```

`build_tensor_specs` calls `decode_mtp.build_tensor_specs` and reuses only
the compute weights, caches, routing inputs, outputs, row indices, and scalars.
It adds ranked host-prepared specs with these exact shapes and dtypes:

```text
hidden_states       [N_RANKS, T, D]          torch.bfloat16
prev_pre_hc_hidden  [N_RANKS, T, HC_MULT, D] torch.float32
swa_slot_mapping    [N_RANKS, T]             torch.int64
swa_indices         [N_RANKS, T, WIN]        torch.int32
swa_lens            [N_RANKS, T]             torch.int32
```

Build metadata from the requested `ori_block_num` and `position_ids` using
`decode_metadata.block_table`, `paged_slot_mapping`, and
`swa_indices_and_lens`; do not use the standalone SWA fixture's fixed cache
capacity. Preserve `resident` and `is_output` properties on reused specs and
mark all five prepared tensors `resident="stacked"`. Exclude `embed_weight`,
`main_pre_hc_hidden`, `tail_pre_hc_pool`, `accepted_counts`, `tail_slot_ids`,
`input_ids`, `ori_block_table`, and `sampled_ids`.

`golden_mtp_decode_core` calls, per rank, `golden_mtp_projection`,
`golden_attention_swa`, `golden_hc_head`, and `golden_rms_norm`; it calls the
distributed `golden_moe` once and `golden_lm_head` once. It uses the prepared
metadata directly and maps `routing_input_ids` to the MoE golden `input_ids`.

The CLI matches `decode_mtp.py` for platform, EP, TP, devices, start position,
token count, cache blocks, routing mode, L2 swimlane, compile-only, runtime-dir,
and pass dumps. Add `--finite-only`; when set, use a no-op golden and
`compare_fn={"logits": finite_tensor_compare}`. Otherwise run the exact core
golden and the existing relative-difference comparators. Validate device count,
`ep % tp == 0`, import-time EP/TP agreement, and the target EP8/TP4 topology
when reporting comparison results.

- [ ] **Step 5: Apply the kernel style pass and run contract/lint checks**

Apply the `fmt-coding-style` workflow only to the new
`models/deepseek/v4-flash/decode_mtp_core.py`, then run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp_core.py -v
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
ruff check models/deepseek/v4-flash/decode_mtp_core.py models/deepseek/v4-flash/lm_head.py tests/contract/test_deepseek_v4_flash_decode_mtp_core.py
ruff check --select F --ignore-noqa models/deepseek/v4-flash/decode_mtp_core.py
python -m py_compile models/deepseek/v4-flash/decode_mtp_core.py
```

Expected: all tests, lint, and import-time compilation pass.

- [ ] **Step 6: Commit the strict MTP-core benchmark**

```bash
git add models/deepseek/v4-flash/decode_mtp_core.py tests/contract/test_deepseek_v4_flash_decode_mtp_core.py
git commit -m "Add: DeepSeek V4 MTP-core benchmark"
```

---

### Task 3: Validate the Target Run and Update the Chinese Report

**Files:**
- Modify in the primary worktree: `docs/2026-07-31-deepseek-v4-flash-decode-mtp-compare4.md`
- Keep untracked: generated runtime directories, logs, and profiler traces.

**Interfaces:**
- Consumes: the completed `decode_mtp_core.py` runner and the provided AscendC
  trace values.
- Produces: correctness evidence, target EP8/TP4 measurements, a verified
  compute/cleanup trace boundary, and the updated Chinese comparison report.

- [ ] **Step 1: Run a two-device exact-golden correctness check**

Submit with PTOAS 0.54:

```bash
task-submit --ptoas 0.54 --device 0,2 --timeout 0 --max-time 1800 --run \
  'python models/deepseek/v4-flash/decode_mtp_core.py -p a2a3 --ep 2 --tp 2 -d "$TASK_DEVICE" --start-pos 8192 --num-tokens 8 --routing-mode trace-hash --enable-l2-swimlane 0'
```

Expected: exact golden validation passes. Record that this uses 16 local
experts per rank but is not the EP8/TP4 performance topology.

- [ ] **Step 2: Compile and smoke-test the target eight-device topology**

Run:

```bash
task-submit --ptoas 0.54 --device 0,2,4,6,8,10,12,14 --timeout 0 --max-time 1800 --run \
  'python models/deepseek/v4-flash/decode_mtp_core.py -p a2a3 --ep 8 --tp 4 -d "$TASK_DEVICE" --start-pos 8192 --num-tokens 8 --routing-mode trace-hash --enable-l2-swimlane 0 --finite-only'
```

Expected: EP8/TP4 compilation succeeds, outputs are finite, and the command
exits zero.

- [ ] **Step 3: Inspect the generated program boundary**

Read the generated `kernel_config.py` and verify that the compute child begins
with the projection RMSNorm task, ends with `lm_head_combine_gather`, and does
not contain embedding, hidden packing, SWA metadata builders, sampling, or
signal-clear tasks. Verify that the cleanup child contains only
`moe_signal_clear` and `lm_head_signal_clear`.

- [ ] **Step 4: Run the target 5+100 regression benchmark**

Run:

```bash
task-submit --ptoas 0.54 --device 0,2,4,6,8,10,12,14 --timeout 0 --max-time 1800 --run \
  'PYPTO_BENCH=1 PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_RAW=1 python models/deepseek/v4-flash/decode_mtp_core.py -p a2a3 --ep 8 --tp 4 -d "$TASK_DEVICE" --start-pos 8192 --num-tokens 8 --routing-mode trace-hash --enable-l2-swimlane 0 --finite-only > decode_mtp_core_ep8_tp4_8k_perf_100r.log 2>&1'
```

Expected: 800 rank-round compute samples and 800 cleanup samples, exit zero,
and finite logits.

- [ ] **Step 5: Capture and parse a system trace without in-core profiling**

Capture at least one warmup and three steady executions. Select only the first
top-level child, `mtp_decode_core_logits`, on each rank. Report NPU0 and the
maximum rank interval for every steady round, plus median and mean. Confirm the
second child is cleanup and exclude it from the direct comparison.

- [ ] **Step 6: Update the Chinese comparison report**

Keep the complete Scheme 4 numbers as historical context. Add a distinct
MTP-core section with:

- the Task6-to-Task66 AscendC samples and 2.235998 ms mean;
- the prepared-input and excluded-work list;
- the new PyPTO task IDs, commit, PTOAS version, commands, correctness result,
  100-round distribution, and system-trace samples;
- direct delta and ratio only between the strict MTP-core intervals;
- an explicit statement that 52.731107 ms is not the comparison denominator.

- [ ] **Step 7: Run final repository checks**

Run:

```bash
python -m pytest tests/contract/test_deepseek_v4_flash_decode_mtp_core.py tests/contract/test_deepseek_v4_flash_decode_mtp.py -v
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
ruff check .
pre-commit run --all-files
```

Expected: all checks pass and no generated artifact is staged.
