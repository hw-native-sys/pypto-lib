# Performance Tuning

A practical guide for tuning pypto-lib kernels on Ascend NPU (A3 / 910C).
The flow is two-tiered: first balance the inter-kernel schedule on the
AICPU side (chip swimlane), then optimize each kernel's internal pipeline
(L1/L0 swimlane + PMU).

For the underlying levels see simpler's
[Hierarchical Level Runtime](https://www.pypto.ai/simpler/hierarchical-level-runtime/):
L2 = one chip (AICPU + AIC/AIV cores), L1 = die / L2 cache, L0 = single
compute core.

---

## Measuring — the benchmark loop (`PYPTO_BENCH`)

Tuning needs a number before and after. Set `PYPTO_BENCH=1` and every
`run` / `run_jit` call in the process times the kernel on device after its
correctness dispatch — no `--benchmark` flag, no edit to the model file:

```bash
PYPTO_BENCH=1 python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0
```

```
[RUN]   effective_us (100 rounds) min=520.1 median=538.4 mean=539.9 max=602.0
```

**Effective** is the framework's post-graph-build execution window on
device (`orch` ∪ `sched` — the old device-log "Total"), recovered from the
runtime's `[STRACE]` markers. Quote `mean=`: daily CI's per-case perf number
is exactly this field of exactly this line
([daily_ci.yml](../../.github/workflows/daily_ci.yml)), so a local mean is
directly comparable to the dashboard.

Requirements: a real device — a `*sim` platform prints
`effective_us unavailable: no device-domain spans` — and a runtime built
with `SIMPLER_PROFILING`. A `runtime_dir=` replay has no live
`CompiledProgram` and skips benchmarking with a `[RUN] benchmark skipped`
note.

### Multi-card (L3) output

A distributed program adds a per-rank breakdown and a context line:

```
[RUN]   effective_us (100 rounds) min=520.1 median=538.4 mean=539.9 max=602.0
[RUN]     rank 10: eff_us min=500.0 median=510.0 mean=511.0 max=520.0
[RUN]     rank 11: eff_us min=520.1 median=538.4 mean=539.9 max=602.0
[RUN] benchmark kernel=moe_ep2 l3_resident=1 rounds=100 ranks=2 host_union_mean_us=900 host_mean_us=950
```

- The headline is the **per-round max across ranks** — the round ends when
  the slowest card finishes. The `eff_us` lines expose the cross-card
  imbalance that max hides; a persistent gap between ranks is a load-balance
  problem, not a kernel problem.
- A `rank N: eff_us` line **sums** that card's dispatches within a round (a card
  runs them serially). When a card dispatches more than once per round, each rank
  line gains a nested `slot` line per dispatch so you can see which dispatch owns
  the time:

  ```text
  [RUN]     rank 10: eff_us min=500.0 median=510.0 mean=510.0 max=520.0
  [RUN]       slot 0 (prefill_orch): eff_us min=200.0 median=205.0 mean=205.0 max=210.0
  [RUN]       slot 1 (decode_orch): eff_us min=300.0 median=305.0 mean=305.0 max=310.0
  [RUN]     rank 11: eff_us min=520.1 median=561.0 mean=561.0 max=602.0
  [RUN]       slot 0 (decode_orch): eff_us min=520.1 median=561.0 mean=561.0 max=602.0
  ```

  `slot` is the dispatch's position within its rank's round (slot 1 is the same
  dispatch in every round), and the name in parentheses is the orchestration
  function it runs. Once the slot lines appear, every rank's dispatches are
  listed — including single-dispatch ranks like 11 above, whose slot line
  necessarily restates its rank line — so the breakdown stays a complete tree.
  The slot lines are omitted entirely when every card dispatches exactly once per
  round, and also when a card's dispatch *order* varies between rounds — a slot
  then names no single callable, so pypto reports no per-dispatch view rather
  than mislabelling it.
- `host_union_mean_us` is the cross-rank host-timeline window
  (`max(end) - min(start)`), so it captures start skew and overlap, but
  includes host dispatch overhead.
- `fallback_flattened=1` means a rank's dispatch count was not divisible by
  `warmup + rounds` (a non-deterministic dispatch shape), so per-round
  segmentation was abandoned and the numbers are a pooled per-dispatch
  sample — treat them as indicative only.

### Low-latency layer and full-forward comparisons

Use complementary measurements for `models/deepseek_v4_flash_lowlat/`:

| Purpose | Collection | Reported metric |
|---|---|---|
| Screen a layer change | One level-1 chip swimlane dispatch, without a benchmark loop | Select the rank with the shortest Worker View receive-to-end span; report that same rank's device `orch` / `sched` union |
| Confirm the full-forward effect | `PYPTO_BENCH=1`, with fixed warmup and measured rounds | Compute each rank's median effective time, then report the lowest of those medians |
| Explain the dependency and dispatch changes | A separate level-4 capture | Critical-path tasks, preceding gaps, and actual dispatch evidence |

BENCH is the collection mechanism; effective time is the device metric it
collects. The full-forward convention here is `min(rank medians)`. It excludes
host setup and reduces the influence of rank start skew, so label it explicitly
and keep it distinct from the headline maximum across ranks and host-inclusive
step latency. Retain the CI headline convention above when reporting CI results.

Freeze one input/golden snapshot per layer case. Freeze full-forward inputs
separately with `decode_fwd.py --save-data`, then replay them with
`--input-data <snapshot>/data`; see [Save and Replay](../run-and-validate/save-and-replay.md).
Use identical rounds, warmups, shapes, pins, and devices within each comparison.
The measured examples in the [DeepSeek case study](deepseek-v4-decode-optimization.md#5-low-latency-variant-evaluation)
use 100 measured rounds / 5 warmups for the default forward fixture, and
50 / 5 for a separate 43-bank fixture.

A single dispatch does not guarantee cold L2. Repeated layer execution can
reuse cached weights, and the default forward fixture reuses one MoE bank
across layers. When testing weight traffic or prefetching, also compare with
separate bank addresses (`--moe-banks 43` for the 43-layer fixture). Keep the
weight values and other inputs fixed while changing the address working set,
and compare each candidate against the baseline with the same bank count.
Separate banks do not guarantee cold L2 between benchmark rounds either.

Validate layer outputs against their goldens and exercise both gate routing
modes, including zero and partial active-token counts. The current full
forward has no `golden_fn`: its passing result is an execution check, not a
numerical comparison. Do not combine timing numbers from different profiling
levels. Reconcile physical records with dependency counts; level-4 merged
traces also contain allocation and dummy-task display slices.

### Knobs

| Env | Default | Effect |
|-----|---------|--------|
| `PYPTO_BENCH` | off | Enables the timed loop. Any value except `""` / `0` / `false` / `False` is on. |
| `PYPTO_BENCH_ROUNDS` | `100` | Timed rounds. 100 rounds is ~0.1 s of device time for a decode step but minutes for a long prefill or a multi-card run — drop it while iterating. |
| `PYPTO_BENCH_WARMUP` | `5` | Leading launches discarded before measurement. The resident L3 path always keeps ≥ 1 (its first warmup launch doubles as the validation dispatch). |
| `PYPTO_BENCH_RAW` | off | Prints every measured dispatch's Effective sample, one line per rank, in dispatch order. Use it when a summary looks suspicious — start-up drift, a bimodal rank, one card lagging. |

A malformed or out-of-range value warns and falls back to the default
rather than failing the run. Daily CI sets none of the three, so its numbers
always come from the 100 / 5 baseline; if you change the loop sizes locally,
compare only against other runs with the same sizes.

```bash
# Quick iteration on a long prefill, with the raw per-dispatch samples.
PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=10 PYPTO_BENCH_WARMUP=2 PYPTO_BENCH_RAW=1 \
  python models/deepseek_v4_flash_mtp/prefill_fwd.py -p a2a3 -d 0
```

When only the timing changes between iterations — not the numerics — save the
golden once and replay it via `golden_data=`, cutting the torch recompute out
of every later run. See
[Save and Replay Golden Data](../run-and-validate/save-and-replay.md).

---

## Part 1 — L2 tuning (inter-kernel schedule)

### Capture

Run the case with `--enable-chip-swimlane`. The runtime writes raw per-task
chip swimlane records under the build directory and, on a real-device platform,
converts them to a merged swimlane:

```bash
python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0 --enable-chip-swimlane
```

```
build_output/<ProgramName>_<ts>/dfx_outputs/
├── chip_swimlane_records.json
├── deps.json                    # real-device graph pass
└── merged_swimlane_<ts>.json   # real device only; open this
```

Two viewers work:

- Open `merged_swimlane_<ts>.json` in <https://ui.perfetto.dev/>.
- Or open `chip_swimlane_records.json` directly with the
  [pypto-toolkit VSCode extension](https://marketplace.visualstudio.com/items?itemName=CANN-PUB.pypto-toolkit).

Simulator platforms emit `chip_swimlane_records.json` but intentionally skip
the merged conversion because their records do not yet include the task
metadata the converter requires. Use a real-device capture when you need the
merged Perfetto view and dependency arrows.

The trace shows one lane per AICPU / AIC / AIV with task name, duration
and dependency edges — gaps and stalls are visible directly.

### What to look for

Look for these shapes on the swimlane that indicate a problem:

| Symptom | Likely cause | Fix |
|---|---|---|
| Cores idle while AICPU lane is solid | Kernels too small; AICPU scheduling is the bottleneck | Make kernels larger (item 2) |
| Long tail on a single AIC/AIV | One kernel is too big and serializes | Split it (item 3) |
| Cube / vector unit utilization low even though kernel is busy | Tile size under-fills the user-visible on-chip buffers | Re-tile against `Mat` / `Acc` for cube or `Vec` for vector work (item 4) |
| Cube lane busy while vector lane idle (or vice versa) | Vec/cube epilogue is split into separate kernels | Merge into a mixed kernel (item 2c) |
| Sequential AICPU dispatch trail per region | Region issues one kernel per iteration | Use `pl.spmd` to dispatch a block fan-out once (item 5) |

A gap on this trace is not automatically a scheduling problem: the interval
before a task splits into producer-FIN detection, ready-but-undispatched
scheduler delay, and post-dispatch pickup, and each has a different fix. See
[Dependencies and Scheduling](dependency-and-scheduling.md) for how edges are
formed, what the four per-task timestamps mean, and how to attribute a gap
without guessing.

### Decide the bound class before choosing a fix

The symptom table narrows the candidates; it does not say whether the chip or
the scheduler is the constraint. Run the analyzer on a level-4 capture and read
its verdict before editing anything:

```bash
python -m simpler_setup.tools.sched_overhead_analysis \
  --chip-swimlane-records-json <dfx_outputs>/chip_swimlane_records.json \
  --deps-json <dfx_outputs>/deps.json
```

It prints a one-line verdict — SCHEDULER-BOUND or COMPUTE-BOUND — plus the
AICPU phase split, and the dominant phase names the edit:

| Dominant AICPU phase | The constraint | The edit |
|---|---|---|
| Dispatch | submit count | Group stages, `pl.spmd` (item 5) |
| Complete | dependency-edge count | Fewer created tensors, `manual_dep`, `no_dep_args`, restated `deps=` — see [Dependencies and Scheduling](dependency-and-scheduling.md) |
| Idle | the scheduler is no longer binding | Stop here and go back to compute (items 2–4) |

Watch the trend across iterations: **rising Idle is the proof that a change
removed the binding constraint**; Idle that stays flat means it did not. Read
per-engine starvation before "freeing" a resource — an engine that was never
starved gains nothing from the capacity returned to it.

**The capture must be level 4.** `aicpu_tasks` is empty below it and the tool
has nothing to report. An `--enable-chip-swimlane` declared
`action="store_true"` yields level 1, not 4; the entry needs
`type=int, nargs="?", const=4, choices=range(5)` before capturing.

This analysis and the critical path can disagree. `critical_path` attributes a
stall to the core being busy (`core-wait`), which reads as a resource problem
when the AICPU is simply never dispatching; on a dispatch-heavy kernel, believe
the phase split. Treat any single capture's "scheduler injects X %" figure as
directional — it swings more than the effects being measured. The reproducible
signals are the summed phase totals, the edge / task / submit counts, and the
pop hit rate.

Then state the mechanism before editing. Name (a) which task the trace says is
the bottleneck, (b) what that task's own numbers say is limiting it — per-block
execution time against head overhead, occupancy, or stall class — and (c) how
the proposed change removes *that*. A sweep run without (a)–(c) can find a win
and still leave you unable to predict the next one, or to know whether it
transfers to another shape. When a result contradicts the stated mechanism,
re-profile before keeping it.

### Tuning rules

#### 1. Use `pl.range` vs. `pl.parallel` correctly

`pl.parallel` declares iterations are independent — the compiler may
distribute them across cores. `pl.range` is strict sequential and forces
a dependency chain. Use `pl.parallel` whenever there is no carried state,
and reserve `pl.range` for accumulators or stateful loops.

```python
# the batch tile is independent — pl.parallel
for b0 in pl.parallel(0, BATCH, BATCH_TILE):
    ...
```

A `pl.range` over an independent dimension forces the swimlane into a
single lane; switching to `pl.parallel` is usually the largest single
win at this stage.

#### 2. Kernels too small — make each kernel do more

When the swimlane shows cores idling while the AICPU lane is fully
saturated, the AICPU dispatcher is the bottleneck. Target ~50 µs per
kernel on A3 / 910C (smaller kernels add dispatch overhead that the AICPU
can't hide). Three ways to grow each kernel:

**a. Fold outer iterations into the core.** Move part of an outer
`pl.range` / `pl.parallel`'s iterations **into** the `pl.at` region as an
inner `pl.range`, so each dispatched kernel processes a tile of iterations
instead of one:

```python
# Before: one kernel per outer iteration — many tiny dispatches
for b in pl.parallel(0, BATCH):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="step"):
        ...

# After: fold BATCH_TILE iterations into each kernel via an inner pl.range
for b0 in pl.parallel(0, BATCH, BATCH_TILE):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="step"):
        for b in pl.range(b0, b0 + BATCH_TILE):
            ...
```

**b. Merge consecutive `pl.at` blocks.** Adjacent `pl.at` regions in the
same scope each become a separate kernel with an AICPU hand-off between
them. Fuse back-to-back regions into one `pl.at` so a single kernel covers
the whole sequence:

```python
# Before: two adjacent regions → two kernels + a hand-off
with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
    ...
with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
    ...

# After: one region → one kernel
with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_q_proj"):
    ...   # rmsnorm, then q_proj
```

The sharpest form of this test: **when two tasks run on the same engine and the
only edge between them is producer → consumer, the boundary buys nothing.** They
cannot run concurrently in any case, so it adds no parallelism while costing a
submit, a scheduler round trip, and a barrier if either side is grouped. Read
the engine from the `deps.json::kernel_ids` slot position (0 = AIC, 1 = AIV) and
check the fan-out degree; same engine plus sole consumer is the signature.

**c. Merge cube + vector into a mixed kernel.** When a matmul (cube) and
its epilogue (cast / add / norm — vector) sit in separate `pl.at` regions,
every projection generates two kernels and an AICPU hand-off between them.
Place both inside the **same** `pl.at` and the compiler co-schedules cube
and vector on the right unit internally, removing the hand-off:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
    for kb in pl.pipeline(0, HIDDEN // K_STEP, stage=2):
        ...
        q_acc = pl.matmul_acc(q_acc, tile_a, tile_b)     # cube
    q_bf16 = pl.cast(q_acc, target_type=pl.BF16)         # vector
    q_proj[b0:b0 + BATCH_TILE, q0:q0 + Q_OUT_CHUNK] = q_bf16
```

A mixed kernel is not free. On a2a3 each AIC is paired with 2 AIV and a MIX task
**reserves all three for its whole duration**, so a long cube phase with a short
epilogue holds far more vector capacity than it uses. Weigh that reservation
against the vector engine's *starvation* figure rather than its occupancy: a
large reservation on an under-subscribed engine costs nothing, and splitting one
back out pays only to the extent that engine was the constraint. When a split is
worth making, group the freed vector stage rather than emitting one submit per
item — see item 5.

#### 3. Kernels too big — split and parallelize

When one kernel dominates the swimlane and the rest of the chip waits on
it, the kernel is too coarse. Pull a `pl.range` out of the `pl.at` and
convert it to a `pl.parallel` chunk loop so each chunk becomes its own
InCore kernel scheduled across cores:

```python
# Before: one giant InCore region over all q_out blocks
with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
    for q0 in pl.range(0, hidden, Q_OUT_CHUNK):
        ...

# After: each q-chunk is its own kernel, parallel across cores
for q0 in pl.parallel(0, hidden, Q_OUT_CHUNK):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
        ...
```

#### 4. Tiling — fill the core-internal buffers

Each AIC / AIV core has fixed on-chip buffers. At the DSL level, cube tiles
directly consume `Mat` (L1 operand storage) and `Acc` (L0C accumulator
storage), while vector tiles consume `Vec` (UB). The tile sizes declared in
your `pl.slice` / `pl.matmul` (typically `BATCH_TILE`, `K_STEP`,
`Q_OUT_CHUNK`, …) control these spaces.

- Too small → buffers are under-utilized, cube/vector throughput drops
  proportionally, MTE2 issues many small loads.
- Too large → tile spills, the compiler falls back to smaller transfer
  units, or compile-time shape checks fail.

**Check actual occupancy.** Every compile writes a per-kernel buffer
report to

```
build_output/<ProgramName>_<ts>/report/memory_after_AllocateMemoryAddr.txt
```

listing, for each compute function, how full each on-chip space runs
against its hardware limit (on the illustrated 910C configuration: vector
`Vec` has a 184 KB compiler-safe limit within the 192 KB physical UB; cube
`Mat` is 512 KB, `Left` / `Right` are 64 KB each, and `Acc` is 128 KB):

```
--- gather_kv ---
  Space  |  Used       |  Limit      |  Usage   |  MemRefs
  -------+-------------+-------------+----------+---------
  Vec    |   129.0 KB  |   184.0 KB  |   70.1%  |  2

--- kv_proj_matmul ---
  Space  |  Used       |  Limit      |  Usage   |  MemRefs
  -------+-------------+-------------+----------+---------
  Mat    |    80.0 KB  |   512.0 KB  |   15.6%  |  4
  Left   |    32.0 KB  |    64.0 KB  |   50.0%  |  1
  Right  |    16.0 KB  |    64.0 KB  |   25.0%  |  1
  Acc    |     4.0 KB  |   128.0 KB  |    3.1%  |  1
```

Scan the `Usage` column for `Mat`, `Acc`, and `Vec`. These are the
user-visible constraints affected by the M/N/K or vector fragment. `Left`
and `Right` report the L0A/L0B staging chosen by the compiler for the L1
fragment; they can routinely read close to 100% and are not independent DSL
tile budgets. Do not shrink a tile merely to reduce a `Left` or `Right`
percentage.

Grow the space that limits the task: `Mat` for operand fragments, `Acc` for
the output fragment, or `Vec` for vector working data. An oversized plain
matmul output may be compiler-subtiled through L0C, so `Acc` is often a
performance boundary rather than an immediate compile failure; extra tiles
still add FIXPIPE drains. The exact build report and compile result are
authoritative.

Practical procedure:

1. Start from the natural problem dimensions (`BATCH`, `HIDDEN`, …).
2. Pick `K_STEP` and the output-chunk size so `Mat` and `Acc` stay within
   the intended bounds without forcing inefficient compiler sub-tiling.
3. Sweep one tile dim up/down by 2× and re-measure with PMU — keep the
   size that pushes the cube (or vector) unit closer to 100 %.

The K loop is then driven by `pl.pipeline(stage=2 or 4)` so the next
tile's MTE2 overlaps the current tile's compute (see Part 2 item 2).
For the complete M/N/K constraint model and empirical sweep method, see
[Cube Tile Tuning](cube-tile-tuning.md).

#### 5. `pl.spmd` for parallel sub-kernel dispatch

`pl.spmd(N)` dispatches `N` blocks of an InCore body in parallel from
**one** AICPU schedule entry, instead of N successive `pl.parallel +
pl.at` dispatches. When a region has many parallel chunks and AICPU
overhead is visible per iteration on the swimlane, replace the explicit
`for ... in pl.parallel: with pl.at: ...` pattern with `pl.spmd`:

```python
# qwen3_14b/decode_layer_a8w8.py: one AICPU dispatch fans out every q_proj block
for q_grid in pl.spmd(Q_ON * N_SUB, name_hint="q_proj_fused_dequant"):
    ...
```

Collapsing N dispatches into one schedule entry cuts AICPU scheduling
overhead sharply. The win is largest for **MPMD-shaped** regions with
heavy fan-in / fan-out — where each block depends on (or feeds) many
others, the AICPU would otherwise track a dependency edge per block, and
`pl.spmd` replaces that whole fan with a single dispatch and its barrier.

Use `pl.spmd` once the per-iteration body is self-contained and the
AICPU lane shows a dispatch trail; keep the explicit form when you need
to nest named sub-regions inside the chunk.

On a SCHEDULER-BOUND kernel the unit of cost is the **submit** — not the task
and not the block. Two consequences follow.

**Blocks are free where submits are not.** `pl.spmd(N)` submits once whatever N
is, so raising the block count costs no AICPU work while smoothing the wave
across cores. Fix ragged packing by adding blocks before touching anything else.
The floor is the cache line: blocks narrow enough to fall under it lose more in
per-block setup than they gain in spread.

**A grouped `pl.spmd` is also a barrier across its whole group.** Folding many
per-item submits into one trades scheduler serialization for dependency
serialization, so group the cheap stages and leave the long ones per-item.
Grouping every stage of a region can easily be a net loss — the scheduler's
share of the critical path collapses while critical-path *compute* rises by
more — and the submit count has a floor below which the curve turns back up. How
far to group depends on how sparse the work is: a grouped stage pays for the
inactive items, an ungrouped stage skips them.

#### 6. Sharding a stage across cards — price what does not shard, and count the barriers

A shard divides the arithmetic on the axis you split. It divides nothing else,
and the two terms it leaves behind usually decide the verdict.

**Whatever the group shares does not shard with it.** Before splitting an axis,
ask what one iteration of the loop reads that the others also read; that operand
is streamed in full on every card no matter how the axis is cut.

- Head-sharding MLA attention gave `qproj_matmul` a full ~9x (core 1783.5 → 193.9
  us) because each card then streams only its own head group's `wq_b`. The same
  shard moved `qk_pv` only **1.4x** (AIC 495.5 → 361.8), because MQA gives it one
  KV head shared by all 64 q heads: each card still pulls the whole
  `[ATTN_K_TILE, HEAD_DIM]` tile per (token, sparse block). The shard removed the
  arithmetic and left the memory traffic.
- The indexer's `score` batches its whole token group into one cube N, so its
  cache page walk is shared. Halving the tokens while leaving the walk identical
  cut core only 20 %, not 50 % — it is page-walk bound, and a token shard cannot
  reach that term.

**Count the barriers the axis costs before you count the core time it saves.**
An axis whose producer and consumer are sharded differently needs a transpose,
and a transpose needs a barrier. Barriers do not share the rank skew between
them: a second one placed before the ranks have reconverged charges the **full**
remaining skew, so adding one roughly doubles the total spin.

Measured on the DeepSeek-V4 CSA decode layer, a2a3 8-card TP, T = 8, summing
every cross-rank sync point per round across all 8 ranks:

| Indexer shard | `score` core | sync total per round | layer (fastest-rank median) |
| --- | --- | --- | --- |
| none (replicated) | 2513 us | 504.5 us | 521.3 us |
| token axis, no exchange | 1685 us (-33 %) | 15 740 us | **511.1 us** |
| cache axis, one exchange | 1460 us (-42 %) | 42 684 us | 516.6 us |

The cache shard won every per-stage number — better core, and a `score → topk →
plan` chain of 57.7 us against the token shard's 68.0 — and still lost the layer,
because its exchange bought a second barrier. **At these shapes one barrier is
worth more than 40 % of a stage's core time.**

**A shard only pays if the stage gates something.** Token-sharding the same
layer's attention cut `qk_pv` core 4.35x (11 497 → 2888 us) and its span
116.1 → 32.3 us, for no wall-time movement at all, because `qk_pv` starts when
the indexer's `score → topk → plan` chain releases it and not before. Read what
gates a stage's **start** before shrinking the stage.

**Corollary for measurement.** A fastest-rank number hides all of this: the
fastest rank is the one every barrier is free on, by construction. Quote the
sync total across all ranks beside it — see
[Benchmarking Rules](../../.claude/rules/benchmarking.md).

---

## Part 2 — L1 / L0 tuning (intra-kernel)

Once L2 is balanced, individual kernels become the bottleneck. Two
artifacts drive intra-kernel tuning:

### Capture

PMU counters per kernel:

```bash
python models/deepseek_v4_flash_mtp/decode_sparse_attn.py -p a2a3 -d 0 --enable-pmu 2
# → build_output/<...>/dfx_outputs/pmu.csv
```

Not every kernel exposes `--enable-pmu`; a kernel that does not can still be
captured by passing `runtime_cfg={"enable_pmu": 2}` to its `run` / `run_jit`
call (the harness bundles it into the runtime's DFX options).

For a per-kernel intra-core swimlane, use
[In-Core Simulator Profiling](incore-simulator-profiling.md).
It explains how the repository workflow builds a standalone single-core
testcase from the generated `.cpp` and sibling `.pto`, runs it under
`msprof op simulator`, validates that data-dependent work actually executed,
and cleans the Insight trace for Perfetto.

For phase timing inside a multi-core extern on real hardware, use
[`cce-incore-profiling.md`](cce-incore-profiling.md). It covers
per-core on-device timestamps, collective-barrier interpretation, and exact
partitions that reconcile internal phases with the L2 task total.

### Tuning rules

#### 1. Fix tile-shape MTE hints from `perf_hints.log`

Every compile writes a perf-hint log next to the memory report:

```
build_output/<ProgramName>_<ts>/report/perf_hints.log
```

The compiler flags every `tile.load` / `tile.store` whose innermost
(trailing) dimension is smaller than the 512 B L2 cache line — the case
that forces MTE into many short, cache-line-straddling transfers. Each
hint carries the exact source location:

```
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost
dim = 256B; recommended >= 512B for backend a2a3 (L2 cache line = 512B).
Consider increasing tile shape on the innermost axis.
at models/deepseek_v4_flash_mtp/qkv_proj_rope.py:68:4
```

Walk the log and widen the trailing tile dimension at each flagged site
so the innermost slice is a multiple of 512 B (item 3 gives the per-dtype
element counts). Bringing every flagged `tile.load` / `tile.store` up to
≥ 512 B is usually the single biggest MTE-efficiency win at this level.

#### 2. `pl.pipeline` for ping-pong on the K loop

Inside a `pl.at` region, the reduction loop of a matmul (the K loop)
should be `pl.pipeline(..., stage=2 or 4)`. The compiler replicates the
loop body `stage` times for ping-pong buffering, so MTE2 (load) overlaps
with cube/vec compute on alternating tiles.

```python
# stage=4 for the largest input-projection K dim
for kb in pl.pipeline(HIDDEN // K_STEP, stage=4):
    ...

# stage=2 is the common default
for kb in pl.pipeline(0, hidden // K_STEP, stage=2):
    ...
```

A `pl.range` here forces strictly serial K iterations — the cube unit
will stall on every load. Always prefer `pl.pipeline` in the K loop.

#### 3. Watch `pl.slice` / `pl.assemble` granularity

MTE transfers prefer 512-byte aligned addresses and lengths on A3 / 910C.
Pick the trailing-dim tile size so the slice is a multiple of 512 B:

- BF16 (2 B/element) → trailing dim multiple of 256 elements
- FP32 (4 B/element) → trailing dim multiple of 128 elements
- INT8 (1 B/element) → trailing dim multiple of 512 elements

Misaligned slices fall back to slower paths visible as long MTE2 bars in
the kernel-insight swimlane. In the qwen3-14b kernels, all `K_STEP` /
`Q_OUT_CHUNK` constants are picked to keep the inner load 512 B aligned.

#### 4. Read PMU utilization

Recommended PMU counters to collect per kernel:

```
pmu_total_cycles
vec_busy_cycles        cube_busy_cycles        scalar_busy_cycles
mte1_busy_cycles       mte2_busy_cycles        mte3_busy_cycles
fixpipe_cycles
```

What each pipe means in context:

| Counter | Cube kernel (AIC) | Vector kernel (AIV) |
|---|---|---|
| `mte1_busy_cycles` | L1 → L0 (operand staging into cube) | — |
| `mte2_busy_cycles` | GM → L1 (operand load from device memory) | GM → UB (input load) |
| `mte3_busy_cycles` | — | UB → GM (output store) |
| `fixpipe_cycles`   | L0C → GM (cube result write-out) | — |
| `cube_busy_cycles` | cube compute | — |
| `vec_busy_cycles`  | — | vector compute |

The bottleneck pipe should sit near 100 % of `pmu_total_cycles`; the
others run overlapped underneath it. Targets:

- **Cube kernel**: `max(mte2_busy_cycles, cube_busy_cycles) / pmu_total_cycles ≈ 100 %`.
  Either the L1 load or the cube compute is saturated — whichever the
  shape is bound by.
- **Vector kernel**: `max(mte2_busy_cycles, vec_busy_cycles) / pmu_total_cycles ≈ 100 %`.
  Either the GM→UB load or the vector compute is saturated. For very
  store-heavy kernels, `mte3_busy_cycles` can be the bottleneck instead.

If both compute and MTE2 are well below 100 %, open the kernel-insight
swimlane: gaps usually mean (a) a missing `pl.pipeline` on the K loop,
(b) suboptimal instruction scheduling, or (c) incorrectly placed
synchronization barriers.
