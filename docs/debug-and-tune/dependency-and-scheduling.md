# Dependencies and Scheduling

Two independent layers decide when a kernel runs on an Ascend NPU.

1. **The dependency graph** says what must happen before what. It is built while
   the orchestration program runs, from tensor arguments and from edges you
   declare. Get it wrong and results are wrong — usually intermittently.
2. **The runtime scheduler** picks, among all orders the graph permits, when
   each task is actually issued to a core. Get it wrong and results are right
   but late.

Most "why is this kernel wrong" answers live in the first layer, and most "why
is there a 4 µs hole in my swimlane" answers live in the second. This guide
builds both from the ground up, shows which artifact proves which claim, and
ends with the loop that turns those artifacts into a shorter schedule.

The authoritative references are upstream:
[Tasks and Ordering](https://www.pypto.ai/pypto/user/tasks/00-model/) for the
dependency model and the keywords, and the
[simpler](https://www.pypto.ai/simpler/) runtime docs for what the scheduler
does with them. This page is the pypto-lib view — the mechanism as it appears
in our models and our capture artifacts, plus the failures this repository has
actually hit.

## The unit of scheduling is a task

A **task** is one dispatch. The runtime never executes an orchestration
function statement by statement; it builds a graph of tasks and runs whatever
is ready.

Four spellings produce a task, and they produce only two shapes:

| Spelling | Available in | Shape |
|----------|--------------|-------|
| `with pl.at(level=..., name_hint=...) as tid:` | `@pl.jit`, `@pl.function` | One task |
| `result, tid = pl.submit(self.kernel, ...)` | `@pl.program` classes | One task |
| `with pl.spmd(n, ...) as tid:` | `@pl.jit`, `@pl.function` | One task, `n` logical blocks |
| `result, tid = pl.spmd_submit(self.kernel, ..., core_num=n)` | `@pl.program` classes | One task, `n` logical blocks |

Two properties follow, and both matter downstream:

- **A mixed cube+vector region is still one task.** `pl.at` carrying both a
  `pl.matmul` and the vector op consuming it dispatches as a single task with
  up to three active subslots — `[AIC, AIV0, AIV1]` — that complete
  independently and retire together. It is one node in the graph, one row in
  `deps.json`, and several physical rows in a swimlane.
- **An SPMD launch is still one task.** `pl.spmd(n)` is one scheduling unit
  fanned out over `n` logical blocks. It carries one TaskId, so a consumer
  waits on the whole fan-out, not on a block. Blocks are dispatched
  individually — with more blocks than cores the launch proceeds in waves —
  unless `sync_start=True` forces an atomic all-or-nothing launch.

Each task also carries a **resource shape** — `MIX` (a cluster of 1 AIC + 2
AIV), `AIC`, or `AIV` — derived from its active subslots. The shape decides
which ready queue it enters and which cores can take it. A task never waits on
a core it could not have used.

> **Statement order expresses nothing.** Two dispatches written one after the
> other are ordered only if something says so: a buffer overlap the runtime can
> see, or an edge you declared. Source adjacency is not that something.

## How an edge comes to exist

There are exactly two mechanisms, and they compose:

```text
final wait set  =  TensorMap-inferred edges  ∪  declared deps=
```

### TensorMap inference

The orchestrator keeps a map from tensor memory regions to the task that last
wrote them. On every submit it does three things with the task's arguments:

| Step | Applies to | Effect |
|------|-----------|--------|
| Creator retention | Every tensor argument, any direction | Edge on the task that created that tensor |
| Producer lookup | `In` / `InOut` | Edge on the registered producer of any **overlapping** region |
| Producer registration | `Out` / `InOut` | This task becomes the registered producer |

So parameter directions are not documentation — they are the input to
dependency inference. Declaring an `InOut` buffer as `Out` tells the runtime
that nothing needs to finish before this task writes it.

Which hazards that covers:

| Hazard | Tracked | Why |
|--------|---------|-----|
| RAW — read after write | Yes | The reader looks up the current writer and takes an edge |
| WAW — write after write | Yes | The new writer takes an edge on the prior writer, then replaces it |
| WAR — write after read | **No** | A writer would have to find every in-flight reader; that is a per-write walk over a reader set on the orchestration hot path |

WAR being untracked is a deliberate runtime trade-off
([WAR anti-dependencies](https://www.pypto.ai/simpler/war-anti-dependency/)),
not a defect. When a pure reader must finish before a later overwrite, that
edge is yours to declare — and declare it with `deps=` on the writer rather
than by promoting the reader to `InOut`, which makes the reader a writer and
serializes it against every other reader of that buffer.

The overlap test works on buffer regions and is conservative: a region it
cannot prove disjoint is treated as overlapping. That is where false
serialization comes from — most often a loop whose iterations write disjoint
slices of one output but look to the map like one buffer.

### Declared edges

Bind a task's TaskId and name it later:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="producer") as first:
    ...
with pl.at(level=pl.Level.CORE_GROUP, name_hint="consumer", deps=[first]) as second:
    ...
```

Three shapes worth knowing:

- **Fan-in through an array.** One TaskId names one task. To wait on a loop's
  worth of producers, collect them into a `pl.array` of `pl.TASK_ID` and pass
  the array — or the per-index list, which is what `models/qwen3_14b/decode_fwd.py`
  relies on when the array is hoisted across a scope boundary.
- **A join with no work.** `pl.system.task_dummy(deps=[...])` returns a TaskId
  that collapses several producers into one handle.
- **`deps=` needs the `as` form.** `with pl.spmd(n):` and `for i in pl.spmd(n):`
  run the same work without naming it and reject `deps=` outright.

Declared edges are not tied to manual scope. They compose with inference, so
one `deps=` inside an ordinary auto scope is the precision tool for the one
edge the inference could not reach.

### Turning inference off

Four opt-outs, from narrowest to widest. Each is an assertion the compiler
**cannot check**:

| Construct | Scope of the claim | Used in this repo |
|-----------|--------------------|-------------------|
| `pl.at(..., no_dep_args=[t])` | One tensor, one task | `models/qwen3_14b/decode_layer_a8w8.py:669` |
| `pl.no_dep(t)` at a call argument | One tensor, one task (`@pl.program` form) | — |
| `pl.create_tensor(..., manual_dep=True)` | One tensor, its whole lifetime | `models/deepseek_v4_pro/moe.py:493` |
| `with pl.manual_scope():` | Every task in the region | `models/qwen3_14b/decode_fwd.py:314` |

Prefer the narrowest one that expresses the claim. `manual_scope` says "I own
the entire graph here" — inside it the runtime skips creator retention,
producer lookup and producer registration alike, so every edge in the region is
one you wrote, including the ones that used to be correct for free.

### What this looks like in `deps.json`

Every edge records where it came from, which makes the mechanism directly
observable:

| `source` | Meaning |
|----------|---------|
| `creator` | Creator retention — the task that allocated the tensor |
| `tensormap` | An overlap lookup hit; carries `overlap: covered \| other` and both slice geometries |
| `explicit` | A declared `deps=` edge; `arg` is `-1` because it belongs to no argument slot |

An edge you expected but cannot find in `deps.json` does not exist at run time.

## What happens at run time

Two AICPU roles run concurrently. On a2a3 the AICPU has four threads: one
orchestrator and three schedulers.

**The orchestrator thread** executes the orchestration program and submits
tasks. Per submit it allocates a task-ring slot and heap space for outputs,
copies parameters, performs the TensorMap lookup and insert, records the
producer set, wires fanout edges, and publishes the task to a ready queue if
nothing is outstanding.

**The scheduler threads** each own a share of the cores and run a two-phase
loop: drain completions, then dispatch. Completion polls each core's `COND`
register for FIN, aggregates the task's subslots, and walks the finished task's
fanout list incrementing each consumer's released-producer count — the consumer
whose producers are all released moves to a ready queue. Dispatch pops a ready
task whose resource shape matches an available core, writes a dispatch payload,
and signals the core.

That gives one task the following timeline, and every timestamp in a level-4
capture is a point on it:

```text
 submit ──► ready ──► dispatch ──► start ──► end ──► finish ──► consumed
    │         │          │           │        │        │           │
 orch      every      payload      core     kernel   AICPU     fanout
 thread   producer    written    picks it   returns  sees FIN  released,
          FIN'd      to a core      up                         slot freed
```

An early-dispatched task takes the same path with `dispatch` moved ahead of
`ready`: the payload is staged on a gated core and released by a doorbell when
the last producer finishes.

The four timestamps a swimlane record carries:

| Field | Stamped by | Means |
|-------|-----------|-------|
| `dispatch_time_us` | AICPU | The scheduler handed this block to a core |
| `start_time_us` | AICore | The core began executing the kernel |
| `end_time_us` | AICore | The kernel returned |
| `finish_time_us` | AICPU | The scheduler observed FIN and can release consumers |

`end → finish` is FIN detection latency, not compute. A consumer becomes ready
at its last producer's **finish**, not its `end`.

### Queues and their order

Ready work is partitioned by resource shape, and each shape has a normal lane
and a speculative lane:

| Source | Regular lanes | `sync_start` lane |
|--------|---------------|-------------------|
| Normal (all producers done) | `ready_queues[MIX\|AIC\|AIV]` | `ready_sync_queues[MIX\|AIC\|AIV]` |
| Early (speculative pre-stage) | `early_dispatch_queues[MIX\|AIC\|AIV]` | `early_sync_start_queue` |

Within a source the order is `sync_start` ▸ MIX ▸ AIC/AIV, and per shape idle
cores before busy ones (a busy core can take a gated pending slot, promoted on
completion). Across sources, **normal strictly precedes early** — speculative
work runs only once both normal lanes are empty. A ready task therefore never
loses a core to a speculative one.

### Back-pressure

The orchestrator can outrun the scheduler, so submission blocks when the ring
holding task slots, output heap bytes, or dependency-list entries is full. It
resumes when the scheduler retires tasks and advances that ring's watermark.

This is ordinary flow control until it becomes a deadlock: a task's fanout
count includes a reference from its owning runtime scope, released only when
`scope_end()` runs — and `scope_end()` is called by the orchestrator, the very
thread blocked waiting for space. A scope that submits more tasks than its
ring's window can hold therefore cannot drain itself. The runtime detects the
condition and exits with an `orch_error_code` naming which resource wedged.

Sizing the rings, placing scopes to shorten intermediate-tensor lifetime, and
reading the per-scope peaks are their own topic — see
[Ring Heap and Scope Stats](ring-heap-and-scope-stats.md).

## Observing it

None of the above is inferable from a kernel's source. Every claim in the rest
of this page — which edge exists, when a task was dispatched, what a gap was
waiting for — is settled by two artifacts and their join.

### The artifacts

| File | Produced by | Carries |
|------|------------|---------|
| `deps.json` | `--enable-dep-gen` | The task graph: tasks, `kernel_ids`, `block_num`, `early_dispatch`, annotated edges |
| `chip_swimlane_records.json` | `--enable-chip-swimlane` | Per-task timing and scheduler/orchestrator phases; **no** edges |
| `name_map*.json` | Compile | Callable id → source name |
| `dispatch_program.json` | Compile | Which program a dispatch directory belongs to |
| `merged_swimlane*.json` | `swimlane_converter` | The two joined, for Perfetto |

Timing and topology are deliberately separate: per-task fanout on the device
hot path would cost a ~1 KB GM store and a linked-list walk per task. The
converter joins them offline, which also means one `deps.json` can serve many
timing runs of the same topology.

### Capture levels

| Level | Adds |
|-------|------|
| 1 | AICore `start` / `end` per task |
| 2 | + AICPU `dispatch` / `finish` |
| 3 | + scheduler phase records |
| 4 | + orchestrator phase records |

Anything that reasons about dispatch — gap attribution, early-dispatch proof —
needs level 4. Note this is the chip swimlane's *perf level*; it is unrelated to
the runtime hierarchy's L4 worker level.

**Check the entry's `argparse` before assuming a flag shape.** Across
`models/` and `examples/` the flag is declared six different ways in 128
places: 68 use `action="store_true"` (a bare flag means level 4), 28 use
`nargs="?", const=1, choices=(0, 1, 2)` (a bare flag means level **1**, and
level 4 is rejected outright), 17 allow 4 explicitly, 5 make a bare flag mean
4, and 7 require an explicit value. There is no safe default form; read the
target's `--help`.

### Identity chain

Runtime task IDs are capture-local — they are execution instances, not source
identifiers, and they do not survive a rebuild. To get from a swimlane row back
to a line of Python:

```text
dispatch_program.json
  → next_levels/<program>/kernel_config.py
  → name_map*.json :: callable_id_to_name
  → deps.json :: tasks[task_id].kernel_ids
  → source name_hint or submitted callee
```

A MIX task maps to more than one callable name — check every active
`kernel_ids` slot. Compiler suffixes such as `_0` can be inlined copies of one
shared `name_hint`, so one source site may own several runtime occurrences.

### Validating a capture before trusting it

Join the streams with `swimlane_converter.read_perf_data()`; the raw file holds
separate cycle-domain streams that must not be joined by hand. Then require:

- `chip_swimlane_level == 4` in the records file;
- raw AICore rows, raw AICPU rows and joined rows have **equal counts**;
- for every timed task, `joined physical rows == deps.block_num × active kernel_ids slots`.

The last check is the one that matters most: a single dropped SPMD or MIX row
turns a `partial` early-dispatch result into a false `full N/N`.

Distributed captures live under `<work_dir>/dfx_outputs/rank{r}/d{k}/`, where
`d{k}` is that card's k-th L2 dispatch. Match the program by
`dispatch_program.json` — two programs can reuse a `func_id`.

### Reconstructing the critical path

```bash
python -m simpler_setup.tools.critical_path <work_dir> --stdout
```

It computes two paths, and the difference is the point:

- **Static CPM** — the duration-weighted longest dependency path assuming
  unlimited cores. This is what the *graph* costs.
- **Observed** — the as-executed backward blame path through data and same-core
  predecessors. This is what the *run* cost, and its task compute plus stalls
  tile the measured AICore makespan.

Use Observed to decide where time went; use Static CPM as a cross-check. A
large gap between them is a scheduling problem, not a graph problem.

The makespan covers first AICore start through last AICore end — it excludes
host and orchestrator front time and the AICPU tail. Level-4 collection has
observer cost, so use it for causal evidence and take production numbers from
an unprofiled benchmark; see [Performance Tuning](performance-tuning.md).

## Tuning the schedule

Everything above describes the machine. This section is the method: a loop that
turns a level-4 capture into a shorter schedule, one question at a time.

```text
1. find the Observed critical path        ── only its tasks can shorten the makespan
2. split every gap on that path           ── FIN detect | undispatched | pickup
3. flag the producers of a gapped task    ── allow_early_resolve=True on each one
4. prove the task was actually staged     ── dispatch < producer FIN, per block
5. if a sibling took the window, evict it ── unflagged dummy on the SIBLING
6. re-measure, and re-derive the path     ── it moves once the head shortens
```

**1. Find the path.** Reconstruct the Observed critical path from the capture
(see [Reconstructing the critical path](#reconstructing-the-critical-path)).
Only tasks on that path can shorten the makespan; time spent on anything else
is slack until the path itself moves.

**2. Split every gap on the path.** A gap is not one thing — it is producer-FIN
detection, ready-but-undispatched scheduler delay, or post-dispatch pickup, and
each has a different remedy. The taxonomy below does the splitting.

**3. Flag the producers.** The undispatched and pickup parts of a gap in front
of a *short* task are what speculative early dispatch removes. Flag every
direct producer of that task — all of them, because one unflagged producer
disables the whole opportunity.

**4. Prove the window was actually won.** A flag in the source is not a result.
Check the dispatch timestamp against the producers' FIN, per physical block,
and accept `full`, `partial` or `none` as the answer.

**5. If a sibling took the window, evict it.** When the path task is
structurally eligible yet still not staged, look at what else shares its
producer. Normal work outranks early work and the early lane is serviced only
once both normal lanes are empty, so a non-critical sibling resolving off the
same producer can consume the opportunity first:

```text
c ──► a     on the critical path — wants the early window
└──► b      not on the path — currently taking it
```

Adding an **unflagged** dummy predecessor to `b` makes `b` structurally
ineligible and leaves the window to `a`. Add it to `b` only — never to `a`, and
never to the shared producer `c`, which would delay both.

**6. Re-measure, then re-derive the path.** Shortening the head of a path
usually moves the path. Judge the change on wall time from an unprofiled
benchmark, not on the level-4 capture that has observer cost built in, and not
on core busy time.

The rest of this section is the reference for steps 2 through 5.

### Reading the gaps

Split the interval before a task into three causes, because each has a
different fix:

| Interval | Name | Meaning |
|----------|------|---------|
| producer `end` → producer `finish` | FIN detection | The scheduler had not yet noticed the producer finished |
| last producer `finish` → `dispatch` | Ready-but-undispatched | The task was legal to run and no core took it |
| `dispatch` → `start` | Pickup / gate | The core had not yet picked it up, or the task is early-staged and still gated |

With these four values folded across every physical block of a task:

```text
data_ready(C)     = max(pred.end_time_us)
observed_ready(C) = max(pred.finish_time_us)
dispatch(C)       = min(C block dispatch_time_us)
start(C)          = min(C block start_time_us)
```

Attribution rules that keep a gap analysis honest:

- A **ready-but-undispatched** gap is a scheduler or occupancy claim, and
  naming a blocking task requires proving that every compatible core lacked a
  free slot throughout the interval — not merely that some other task
  overlapped it. Level-4 scheduler phase records carry counts, not causal task
  IDs; temporal proximity is not causality.
- A **pickup** gap can be same-core contention, but only if the occupying task
  ran until `start(C)`. A task that freed the core before `data_ready(C)` did
  not gate anything.
- `dispatch(C) < observed_ready(C)` means the task was dispatched before its
  producers finished. That is not a paradox — it is speculative early dispatch,
  and the remaining wait is a gate, not a queue.

### Speculative early dispatch

The one lever that removes the ready-but-undispatched and pickup gaps together.
The scheduler may pre-stage a not-yet-released task onto an idle core, gated,
and release it with a doorbell the instant its producers finish. The core is
already holding the payload, so pickup latency is off the critical path.

It is enabled by a **producer-side** hint:

```python
with pl.spmd(token_tiles, name_hint="rms_norm", allow_early_resolve=True) as rms_tid:
    ...
```

`allow_early_resolve=True` is accepted by `pl.at`, `pl.spmd`, `pl.submit` and
`pl.spmd_submit`. It is pure scheduling — results are unaffected.

Four conditions decide whether it does anything:

1. **Every** direct producer of the consumer must be flagged or already
   complete. Flagging one producer of a three-producer consumer buys nothing.
   This is why the hint tends to be applied along a whole chain, as in
   `models/qwen3_14b/decode_fwd.py`.
2. A producer propagates eligibility only once **fully published** —
   `published_block_count == logical_block_num`. A 50-block SPMD producer on 24
   cores dispatches in waves, and a consumer that gated every slot after the
   first wave would strand the remaining producer blocks. Full publication is
   the deadlock guard.
3. Normal work outranks early work. Under a saturated engine there is no early
   window to win.
4. A predicated task never early-dispatches, and a `sync_start` cohort needs
   all-or-nothing capacity.

#### Proving a task was actually early-dispatched

`deps.json::tasks[].early_dispatch` is **not** proof. It records that the task
carries the producer-side flag — a statement about that task's consumers, not
about the task itself. Two things must hold together:

```text
structural:  every direct producer is an allocation source or has early_dispatch=true,
             and at least one is a non-allocation task
temporal:    block.dispatch_time_us + tol < max(producer finish_time_us)
             where tol = 2 * 1e6 / metadata.clock_freq_hz   (two clock ticks)
```

An SPMD task can satisfy this on some physical rows and not others, so the
honest answer is `full N/N`, `partial N/M`, or `none` — never a bare yes. If
the timestamps look early but the graph is not structurally eligible, the
capture and the source disagree; suspect a stale artifact before believing the
timestamps.

### Deliberately delaying a task

`pl.system.task_dummy` has uses beyond joining producers, because a task with
no work still costs one dispatch hop. Three idioms appear in this tree, and
they are easy to confuse:

```python
# 1. Join: collapse several producers into one handle.
gate = pl.system.task_dummy(deps=[tid_a, tid_b])

# 2. Delay by one hop: a real producer, plus a hop, before a non-critical
#    consumer — so it stops resolving at the same instant as the critical one.
#    models/deepseek_v4_flash_dspark/decode_swa.py:171
late_dep = pl.system.task_dummy(deps=[rms_tid])
qkv_proj_rope(..., late_dep)

# 3. Placeholder: a standalone entry has no fused-path producer to fence
#    against, so an empty dummy satisfies the shared signature and is ready on
#    submit.  models/deepseek_v4_flash_dspark/decode_indexer.py:428
late_dep = pl.system.task_dummy(deps=[])
```

Form 2 is the deliberate one. Its comment in `decode_swa.py` states the intent
plainly — *"Dispatch barrier: kv_proj_matmul resolves one hop after
rms_norm"* — and it costs a dispatch to buy the ordering.

The fourth form is step 5 of the loop: an **unflagged** dummy predecessor makes
its consumer structurally ineligible for early dispatch — it breaks condition 1
above — which is how a non-critical sibling is pushed out of a speculative
window a critical-path task needs. No model in this tree currently ships one,
because it is a scheduling experiment rather than a structural requirement: it
is defensible only with before/after level-4 evidence that the sibling stopped
being staged *and* the protected task started, plus a wall-time result. Keep
`deps=` empty — `task_dummy(deps=[c_tid])` adds a real hop after `c` and tests
something else entirely.

`task_dummy` accepts only `deps=`; it cannot itself carry
`allow_early_resolve`. All of these trade throughput for ordering, none changes
results, and none is justified without a before/after measurement.

## Traps this repository has hit

Each of these produced wrong results, not a diagnostic.

**WAR through two views of one tensor.** Tracking keys on the tensor value, so
two distinct `pl.reshape` views of the same external `inout` tensor look
independent. A task reading through one view and a later task writing in place
through the other get no edge, and the write races the read. Seen on the decode
KV-cache writeback: `kv_cache` validated clean while `x_out` failed around 7%
with non-deterministic indices — the signature of a race, not of a numerics
bug. The live fix is a no-op self-copy that marks the parameter `inout` before
the gather, so the writeback takes a WAW edge on it; `add_inout` is a
param-level property, so one tile touch is enough. See the `kv_touch` scope in
`models/deepseek_v4_pro/decode_sparse_attn.py:239` and its siblings in the
`_csa` variants.

**A GM round-trip registered as `add_output`.** A GM tensor written by one task
and read by a later one must be registered on the producing task as
`add_inout`, not `add_output`. `add_output` records only the write, so the
consumer's read-after-write edge never exists. A clean A/B on
`decode_compressor_ratio4.py` gave 20/20 passes with `add_inout` and 16/20 with
`add_output`, the failures splitting between wrong values and a 507046 AICPU
sync timeout. Worth knowing when reading generated orchestration under
`build_output/<case>/orchestration/`.

**A `manual_dep=True` tensor rides only on the explicit chain.** Its ordering
comes from `deps=` and nothing else, for the tensor's whole lifetime. Removing
one `deps=[tid]` to re-anchor a wait silently dropped the only edge ordering a
cumsum before its consumer — no compile error, stale reads. The restored form
is visible as the two-element dependency in
`models/deepseek_v4_flash_dspark/moe.py:289`. Before cutting an explicit edge,
check whether any task it transitively orders reads a `manual_dep` tensor.

**A conservative overlap that was not a dependency.** A `pl.range` loop whose
iterations write disjoint slices of one output serializes on a WAW chain. The
opt-outs fix it by assertion; slicing the output in orchestration so each task
receives a distinct region fixes it by construction, at the cost of more
orchestration tensors to register and walk. On a dispatch-bound graph that cost
can exceed the serialization it removed — measure both ends.

## See also

- [Performance Tuning](performance-tuning.md) — where the swimlane fits in the
  L2 → L1/L0 tuning flow.
- [Ring Heap and Scope Stats](ring-heap-and-scope-stats.md) — the ring model
  behind the back-pressure above, and how to size it.
- [Debugging](debugging.md) — device logs for a hang, and 507xxx triage.
- [PyPTO coding style](../pypto-coding/pypto-coding-style.md) — the kernel forms
  and loop constructs these tasks are written with.

Upstream references:

- [The dependency model](https://www.pypto.ai/pypto/user/tasks/00-model/),
  [Runtime scopes](https://www.pypto.ai/pypto/user/tasks/01-scopes/),
  [Declaring an edge](https://www.pypto.ai/pypto/user/tasks/02-submit/), and
  [Refining the graph](https://www.pypto.ai/pypto/user/tasks/03-tuning/) — the
  authoritative treatment of inference, `deps=`, the opt-outs, `predicate=`
  and `allow_early_resolve=`.
- [Managing dependencies](https://www.pypto.ai/pypto/user/performance/03-dependencies/)
  and [Runtime overhead](https://www.pypto.ai/pypto/user/performance/02-runtime-overhead/)
  — false serialization and per-task cost, upstream.
- [chip swimlane profiling](https://www.pypto.ai/simpler/dfx/chip-swimlane-profiling/)
  and [dep_gen](https://www.pypto.ai/simpler/dfx/dep-gen/) — artifact schemas.
- [Orchestrator](https://www.pypto.ai/simpler/orchestrator/) and
  [WAR anti-dependencies](https://www.pypto.ai/simpler/war-anti-dependency/) —
  submission semantics and why WAR is not tracked.
