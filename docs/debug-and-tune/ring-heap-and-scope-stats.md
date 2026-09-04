# Ring Heap and Scope Stats

Intermediate GM tensors — everything `pl.create_tensor` produces in
orchestration — are not malloc'd. They live in the simpler runtime's
**four ring heaps**, and the only levers over them are *where you place a
runtime scope* and *how big you size each ring*. This page covers the ring
model, how `pl.scope` shapes intermediate-tensor lifetime, and how
`scope_stats` measures the result.

Read it when a run dies with `HEAP_RING_DEADLOCK`, when a longer sequence
length or a bigger EP degree stops fitting, or before changing
`ring_heap` on a kernel whose peaks nobody has measured.

---

## 1. The four rings

The T&R runtime (`tensormap_and_ringbuffer`, the default for every platform
this repo targets) does not keep one global pool. It keeps
`CHIP_MAX_RING_DEPTH = 4` independent **ring sets**, and the scope nesting
depth at submit time selects which one a task uses:

```text
scope depth 0  ->  ring 0 = { task window, heap, dep pool, fanin pool }
scope depth 1  ->  ring 1 = { ... }
scope depth 2  ->  ring 2 = { ... }
scope depth >=3 ->  ring 3 = { ... }        # clamped: everything deeper folds in here
```

Depth 0 is the root scope the executor wraps around
`aicpu_orchestration_entry`; the first runtime scope in generated
orchestration code is depth 1. Each ring reclaims on its own watermark
(`last_task_alive`), which is the whole point of the split: an inner
scope's tasks can be reclaimed without waiting for the outer scope's tasks
to finish.

| Per-ring resource | Compile default (a2a3, **per ring**) | Per-task override |
|---|---|---|
| Task window (task slots) | 16384 | `RunConfig.ring_task_window` |
| Heap (bytes for task outputs / intermediates) | 256 MiB | `RunConfig.ring_heap` |
| Dep-list pool (fanin spill entries) | 16384 | `RunConfig.ring_dep_pool` |

The tensormap is **not** per ring — it is one global table (65536 entries by
default) and appears as a single row in every report.

> Sizing is **per task**, not per process — there is no environment variable
> for it. An L3 run takes the compile defaults unless its entry passes
> `RunConfig` fields; a single-chip run takes the compile defaults whatever it
> passes, because that dispatch path never applies the overrides (§4). Never
> assume — read the effective capacities off the `scope_stats` metadata line
> (§5).

Upstream reference: simpler's
[`MULTI_RING.md`](https://github.com/hw-native-sys/simpler/blob/main/src/a2a3/runtime/tensormap_and_ringbuffer/docs/MULTI_RING.md)
— a runtime-internal design note, so it is only on GitHub, not on the
published [simpler docs](https://www.pypto.ai/simpler/).

---

## 2. What happens to an intermediate tensor

A `pl.create_tensor` in an orchestration function compiles to an
`alloc_tensors(...)` call — a **kernel-less task** that owns a heap block on
the ring of the scope it sits in:

```cpp
// build_output/<prog>/orchestration/<prog>.cpp
uint32_t tmp_ci_shapes[2] = {8192, 1024};
TensorCreateInfo tmp_ci(tmp_ci_shapes, 2, DataType::FLOAT32);
TaskOutputTensors alloc_0 = alloc_tensors(tmp_ci);   // 32 MiB off the current ring
const Tensor& tmp = alloc_0.get_ref(0);
```

That block is released when the owning task reaches `CONSUMED`, which needs
**both** conditions:

1. every consumer task has completed (fanout refcount drained), **and**
2. the scope that submitted it has **closed** (the scope holds its own
   reference, `FANOUT_SCOPE_BIT`, released at scope exit).

Reclaim is then strictly **FIFO per ring**: `heap_tail` follows
`last_task_alive`, so the oldest live allocation pins everything allocated
behind it on that ring.

Three consequences drive every tuning decision on this page:

- **Memory is not freed at last use — it is freed at scope exit.** A buffer
  whose consumers finished long ago still occupies the ring until its scope
  closes.
- **A loop without an inner scope accumulates.** Every iteration's
  intermediate stays live until the *enclosing* scope ends. In the extreme —
  everything at depth 0 — nothing can ever be reclaimed before the program
  ends, and the ring can only fail, never drain.
- **One long-lived allocation blocks the whole ring behind it** (head-of-line
  blocking), because the reclaim pointer is a single FIFO watermark.

---

## 3. Placing scopes with `pl.scope`

By default (`auto_scope=True`) the compiler inserts an AUTO scope around the
function body **and around each `for` / `if` body**, including inside every
`@pl.jit.inline` callee it inlines. That is convenient but blind: a model
with 6–7 nesting levels collapses everything past depth 3 into ring 3, which
then carries the entire program while rings 0–2 sit empty.

To place scopes yourself, turn the automatic ones off and write them:

```python
@pl.jit(auto_scope=False)                       # or @pl.jit.inline(auto_scope=False)
def prefill_fwd(...):
    for layer_idx in pl.range(num_layers):
        with pl.scope():                        # ring 1 — one layer
            for p0 in pl.range(tok_blocks):
                with pl.scope():                # ring 2 — one token block
                    tmp = pl.create_tensor(...)  # freed when this scope closes
                    ...
```

Rules and gotchas:

- `pl.scope()` is legal only in orchestration code, and AUTO mode
  (`pl.scope()`) requires `auto_scope=False` on the enclosing function.
  `pl.scope(mode=pl.ScopeMode.MANUAL)` / `pl.manual_scope()` is a
  *dependency-tracking* choice, not a ring choice, and an AUTO scope may not
  nest inside a MANUAL one.
- `pl.create_tensor` inside a `pl.at` block yields a **tile**, not a GM
  tensor. Allocate GM scratch just outside the `pl.at` that first writes it,
  inside the scope that should own its lifetime.
- Keep loop-carried values (`hidden_states = layer(...)`, output params) out
  of the inner scope's ownership — they must outlive it, and a value still
  read after its scope closed keeps its block live anyway.
- Nesting past depth 3 buys nothing: ring 3 is a clamp, so a 5-deep chain
  just concentrates pressure there.
- A scope is also a scheduling boundary — the runtime drains scope
  bookkeeping at `scope_end`. Very fine scopes around a handful of tasks cost
  more than they reclaim.

### Example in this repo

[`models/deepseek_v4_flash_mtp/decode_fwd.py`](../../models/deepseek_v4_flash_mtp/decode_fwd.py)
is the reference shape: `auto_scope=False` on both the entry and the inlined
body, the ping-pong carriers (`x_attn0` / `x_attn1` / `hidden`) created outside
every scope, and one `with pl.scope()` per attention / MoE stage — with the
callees (`moe`, `attention_swa`) adding the deeper levels themselves.

---

## 4. Sizing the rings

When the peaks are genuinely needed — a big cross-phase payload, a long
prefill — size the rings explicitly. Each knob takes either a scalar
(broadcast to all four rings) or four per-ring values, where `0` means "leave
this ring at the compile default". Precedence is just two tiers now:
`RunConfig` field > compile default.

golden's `run` builds one `RunConfig` from `config` and hands it to the
dispatch, which builds a `CallConfig` and transcribes the ring sizes into
`runtime_env`:

```python
PREFILL_RING_HEAP = (0, 0, 2 * 1024 * 1024 * 1024, 0)   # ring 2 only
config=dict(platform=args.platform, ring_heap=PREFILL_RING_HEAP)
```

Live examples: [`models/deepseek_v4_pro/prefill_fwd.py`](../../models/deepseek_v4_pro/prefill_fwd.py),
[`models/deepseek_v4_flash_mtp/prefill_layer.py`](../../models/deepseek_v4_flash_mtp/prefill_layer.py)
(per-ring tuple), [`models/deepseek_v4_flash_dspark/decode_layer.py`](../../models/deepseek_v4_flash_dspark/decode_layer.py)
(scalar, broadcast to all four rings).

### The single-chip path

L2 ring sizing used to be a dead knob: golden's `run` dispatched a single-chip
program through `execute_compiled`, which has no ring parameters, and
`CompiledProgram.__call__` forwarded only `platform`, `device_id`, `dfx` and
`aicpu_thread_num`. A `ring_heap` was accepted, validated, and then dropped.

Both halves have since moved. PyPTO's `_execute_on_device` now takes the whole
`RunConfig` and calls `_apply_ring_overrides` on the `CallConfig` it builds,
and golden's `run` dispatches L2 through `CompiledProgram.__call__` with that
same config — so the fields reach the runtime on both paths. **This is read
off the plumbing, not measured**: no L2 entry in this repository sizes its
rings yet, so if you are the first, confirm the value landed (`scope_stats`,
§5) before trusting it.

Kernel-side remains the cheaper fix when a ring overflows: cut the peak with a
`pl.scope` (§3), shrink the intermediates, or split the scope. A few leaves
carry a `*_RING_HEAP` constant recording what they would need.

---

## 5. Measuring with `scope_stats`

`scope_stats` records one sample at every scope boundary — task-window,
heap, dep-pool head/tail plus tensormap usage — so a peak can be attributed
to a **scope site**, not just a resource.

### Enable it

`enable_scope_stats` is one of the five DFX flags the golden harness
forwards (`enable_chip_swimlane`, `enable_dump_args`, `enable_pmu`,
`enable_dep_gen`, `enable_scope_stats`). Add the flag to the entry's argparse
and pass it through:

```python
parser.add_argument("--enable-scope-stats", action="store_true", default=False)
...
config=dict(
    platform=args.platform,
    device_id=args.device,
    enable_scope_stats=args.enable_scope_stats,
)
```

Any DFX flag forces `save_kernels=True`, so the work dir survives the run.
It works on both the L2 and L3 paths, and it needs an actual execution —
`--compile-only` never reaches the runtime, so it records nothing.

### Read the output

```text
<work_dir>/dfx_outputs/scope_stats/scope_stats.jsonl        # L2
<work_dir>/dfx_outputs/rank<N>/d<K>/scope_stats/...jsonl    # L3: per rank, per dispatch
```

Line 1 is run metadata — schema `version`, `fatal`, `dropped`, and the
**effective capacities** `task_window_max` / `heap_max` / `dep_pool_max`
(arrays indexed by ring) and `tensormap_max`. This line is the fastest way
to confirm a ring override actually took effect. Every later line is one
`begin` or `end` sample carrying `site` (`<orchestration>.cpp:<line>`),
`depth`, `ring`, and the head/tail pairs. Heap counters are monotonic
cumulative bytes in `version` 6 (they wrapped in version 5).

Pair a `begin` with its `end` to get the metrics:

| Metric | Formula | Read it for |
|---|---|---|
| `scope_high_water` | `end.head − begin.tail` | Upper bound on the pressure this scope reached (entry backlog + everything it allocated). Not capped — a streaming scope can report more than the ring. |
| `real_occupancy` | `end.head − end.tail` | What is still held when the scope exits. Always within `[0, cap]`. |
| `scope_alloc` | `end.head − begin.head` | Allocation throughput of the scope, independent of reclaim. |

### Render or summarize

```bash
python -m simpler_setup.tools.scope_stats_plot \
    build_output/<prog>_<ts>/dfx_outputs/scope_stats/scope_stats.jsonl
# writes scope_stats.html next to the input
```

The runner prints a hint naming `runtime/tools/scope_stats_plot.py`; that
path does not exist in a pypto-lib checkout — use the module form above.

For a quick per-ring number, pair the records yourself:

```python
import json, sys
from collections import defaultdict

lines = [json.loads(line) for line in open(sys.argv[1])]
meta, records, open_scopes = lines[0], lines[1:], {}
peak = defaultdict(lambda: (0, 0, ""))          # ring -> (high_water, live_at_exit, site)
for r in records:
    if r["phase"] == "begin":
        open_scopes[r["depth"]] = r
        continue
    b = open_scopes.pop(r["depth"], None)
    if b is None:
        continue
    hw = r["heap_end"] - b["heap_start"]        # scope_high_water
    live = r["heap_end"] - r["heap_start"]      # real_occupancy
    prev = peak[r["ring"]]
    peak[r["ring"]] = (max(hw, prev[0]), max(live, prev[1]), r["site"] if hw > prev[0] else prev[2])

print(f"fatal={meta['fatal']} dropped={meta['dropped']}")
for ring, (hw, live, site) in sorted(peak.items()):
    cap = meta["heap_max"][ring]
    print(f"ring {ring}: high_water {hw/2**20:7.1f} MiB ({100*hw/cap:5.1f}% of {cap/2**20:.0f} MiB)"
          f"  live_at_exit {live/2**20:7.1f} MiB  peak site {site}")
```

### Interpret it

| Observation | Meaning | Action |
|---|---|---|
| One ring near or over capacity, others near zero | auto-scope collapse, or all work at one depth | place scopes so the load spreads over rings 0–3 (§3) |
| High `scope_high_water`, low `real_occupancy` | the scope streams: it allocates a lot but reclaim keeps up | fine — the ring is doing its job |
| `real_occupancy` high at scope exit | something long-lived is pinned in that ring | shrink the cross-phase payload, or move it to an outer scope that closes later |
| Only `begin` records at the tail of the file | the run wedged inside that scope | the last open `site` is where the ring filled |
| `dropped > 0` | collector back-pressure — records are missing | peaks are lower bounds; re-run with less concurrency before trusting them |
| `fatal: true` | a fatal was latched mid-run | records after it are diagnostic only |

Measure under the capacity the run will actually have. Reclaim only happens
once the allocator has to wait, so on an oversized ring nothing is ever
reclaimed and a scoped program reports the same frontier as an unscoped one —
the scoping only shows its worth when the ring is tight enough to fill.

---

## 6. Failure modes this explains

Every one of these surfaces host-side as the generic
`507018 ACL_ERROR_RT_AICPU_EXCEPTION`. The real cause is the
`orch_error_code=` line in the host log, and the runtime's own hint points
at `scope_stats`:

| `orch_error_code` | Name | What it means | First move |
|---|---|---|---|
| 1 | `SCOPE_DEADLOCK` | one scope submitted more tasks than the ring's task window; slots only free at `scope_end` | split the scope, or raise `ring_task_window` |
| 2 | `HEAP_RING_DEADLOCK` | the ring ran out of heap bytes (and slots) — no further task can be admitted | scope the intermediates, shrink them, or raise `ring_heap` |
| 3 | `FLOW_CONTROL_DEADLOCK` | task window blocked while heap is free — usually nesting on the *same* ring | move the nested submission to another ring |
| 4 | `DEP_POOL_OVERFLOW` | a task's fanin edges overflowed the ring's dep pool | cut the fanin, or raise `ring_dep_pool` |
| 11 | `TENSORMAP_OVERFLOW` | the global tensormap wedged | extreme scale — check the tensormap row in the report |

Oversizing has its own failure: the static arena scales with every ring's
window / dep-pool / heap, and a run that pins all three large enough dies
with an `rtMalloc 207001` OOM before it ever reaches the kernel.

---

## Related

- [Debugging](debugging.md) — device logs for a hang, `runtime_dir` reuse,
  the other four DFX flags.
- [Dependencies and scheduling](dependency-and-scheduling.md) — the task
  graph and scheduler behind the rings: what a task is, how edges are formed,
  and what fills a window.
- [Performance tuning](performance-tuning.md) — chip swimlane and PMU; scope
  placement changes the schedule too, so re-measure wall time after it.
- [PyPTO coding style](../pypto-coding/pypto-coding-style.md) — `pl.at`
  scopes (InCore), which are a different thing from the runtime scopes here.
- simpler's [Scope stats](https://www.pypto.ai/simpler/dfx/scope-stats/)
  — the collector's own reference: report layout, JSONL schema history, and
  the AICPU / host internals.
