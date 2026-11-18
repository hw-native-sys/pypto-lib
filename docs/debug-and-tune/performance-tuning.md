# Performance Tuning

A practical guide for tuning pypto-lib kernels on Ascend A2/A3 and A5.
Start with the chip swimlane to locate the bottleneck, then choose task
organization or intra-kernel optimization. Iterate between these levels as
the bottleneck moves.

For the underlying levels see simpler's
[Hierarchical Level Runtime](https://www.pypto.ai/simpler/hierarchical-level-runtime/):
L2 = one chip (AICPU + AIC/AIV cores), L1 = die / L2 cache, L0 = single
compute core.

---

## Measuring — the benchmark loop (`PYPTO_BENCH`)

The examples below use V4-MTP's CSA decode operator,
`models/deepseek_v4_flash_mtp/decode_csa.py`.
Tuning needs a number before and after. Set `PYPTO_BENCH=1` and every
`run` call in the process times the kernel on device after its
correctness dispatch — no `--benchmark` flag, no edit to the model file:

```bash
PYPTO_BENCH=1 python models/deepseek_v4_flash_mtp/decode_csa.py -p a2a3 -d 0
```

```
[RUN]   effective_us (100 rounds) min=520.1 median=538.4 mean=539.9 max=602.0
```

**Effective** is the framework's post-graph-build execution window on
device (`orch` ∪ `sched` — the old device-log "Total"), recovered from the
runtime's `[STRACE]` markers. Quote `median=` for single-card comparisons.
Daily CI uses the same median convention. Older multi-card Daily CI reports
used the fastest rank's mean; recompute those baselines from their raw samples
before comparing them with medians. For multi-card tuning, use the fastest
rank's median as described below.

Requirements: a real device — a `*sim` platform prints
`effective_us unavailable: no device-domain spans` — and a runtime built
with `SIMPLER_PROFILING`. A `runtime_dir=` replay benchmarks the replayed
build, so a hand-edited `.cpp` can be timed without recompiling; only a spec
with a stepped scalar skips it, with a `[RUN] benchmark skipped` note.

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
- For tuning comparisons, report the **lowest per-rank median** from the
  `rank N: eff_us` lines to reduce the effect of start skew. This is not
  end-to-end step latency; identify the convention explicitly and compare
  like for like. Retain persistent rank imbalance in the report; a faster
  rank alone does not prove an improvement in the whole distributed step.
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

### Knobs

| Env | Default | Effect |
|-----|---------|--------|
| `PYPTO_BENCH` | off | Enables the timed loop, summary, and raw samples. Any value except `""` / `0` / `false` / `False` is on. |
| `PYPTO_BENCH_ROUNDS` | `100` | Timed rounds. 100 rounds is ~0.1 s of device time for a decode step but minutes for a long prefill or a multi-card run — drop it while iterating. |
| `PYPTO_BENCH_WARMUP` | `5` | Leading launches discarded before measurement. The resident L3 path always keeps ≥ 1 (its first warmup launch doubles as the validation dispatch). |

Every benchmark prints each measured dispatch's Effective sample, one line
per rank in dispatch order, after measurement. Use these samples to spot
start-up drift, a bimodal rank, or one card lagging.

A malformed or out-of-range value warns and falls back to the default
rather than failing the run. The 100 / 5 default is the baseline every
reported number should come from; if you change the loop sizes, compare only
against other runs with the same sizes.

```bash
# Quick CSA decode iteration, with the raw per-dispatch samples.
PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=10 PYPTO_BENCH_WARMUP=2 \
  python models/deepseek_v4_flash_mtp/decode_csa.py -p a2a3 -d 0
```

When only the timing changes between iterations — not the numerics — save the
golden once and replay it via `golden_data=`, cutting the torch recompute out
of every later run. See
[Save and Replay Golden Data](../run-and-validate/save-and-replay.md).

---

## Part 1 — L2 tuning (inter-kernel schedule)

### Capture

Run the case with `--enable-chip-swimlane`. The runtime writes raw per-task
chip swimlane records under the build directory. Open these records directly
for routine analysis; real-device runs also attempt merged-trace conversion:

```bash
python models/deepseek_v4_flash_mtp/decode_csa.py -p a2a3 -d 0 --enable-chip-swimlane
```

```
build_output/<ProgramName>_<ts>/dfx_outputs/
├── chip_swimlane_records.json  # preferred viewer input
├── deps.json                    # real-device graph pass
└── merged_swimlane_<ts>.json   # optional Perfetto view, real device only
```

The flag takes a capture **level**, and a bare flag means level 1 — per-task
AICore timing, which is what reading the L2 schedule needs. Raise it only when
the question requires it; each level records more and perturbs the timing it
measures. Gap attribution and early-dispatch proofs need
`--enable-chip-swimlane 4` — see
[Capture levels](dependency-and-scheduling.md#capture-levels).

Prefer the **PyPTO3 Toolkit VSCode extension**, documented in the
[`hw-native-sys/pypto-tools` repository](https://github.com/hw-native-sys/pypto-tools)
and the
[PyPTO3 Toolkit guide](https://www.pypto.ai/pypto-tools/). Right-click
`chip_swimlane_records.json` and select **PyPTO3 Toolkit: Open File**. For an
L3 capture, open the records for the selected rank and dispatch, for example
`build_output/<ProgramName>_<ts>/dfx_outputs/rank0/d0/chip_swimlane_records.json`.
Keep matching `deps.json` and `name_map*.json` alongside the records so the
extension can show dependencies and function statistics. Available worker,
scheduler, and orchestrator views depend on the capture level.

As a secondary option, open `merged_swimlane_<ts>.json` in
[Perfetto](https://ui.perfetto.dev/). Keep this format for Perfetto-specific
inspection; it is not the default file to open or deliver.

Simulator platforms emit `chip_swimlane_records.json` but intentionally skip
merged conversion because their records do not yet include the task metadata
the converter requires. Dependency inspection requires a matching graph;
use a real-device capture when the simulator does not provide it.

### Timing one stage of a full network — task-timing slots

Use task-timing slots when you already know which stage to measure and a full
swimlane is too large or perturbs the schedule. Slots provide repeated stage
measurements within the full program; use the swimlane to investigate why a
stage is slow. See [Task-Timing Slots](task-timing-slots.md) for instrumentation,
replay, and interpretation of dispatch and finish timestamps.

### What to look for

Look for these shapes on the swimlane that indicate a problem:

| Symptom | Likely cause | Fix |
|---|---|---|
| Cores idle while AICPU lane is solid | Kernels too small; AICPU scheduling is the bottleneck | [Coarsen task granularity](#adjust-task-granularity-along-the-critical-path) |
| Long tail on a single AIC/AIV | One kernel is too big and serializes | [Split critical work](#adjust-task-granularity-along-the-critical-path) |
| Cube / vector unit utilization low even though kernel is busy | Tile size under-fills the user-visible on-chip buffers | [Tune tiles against buffer constraints](#improve-intra-kernel-efficiency) |
| Cube lane busy while vector lane idle (or vice versa) | Vec/cube epilogue is split into separate kernels | [Evaluate MIX fusion](#fuse-tasks-with-the-full-dataflow-in-mind) |
| A stage re-reads the same weights every layer, MTE2-bound, with a large unrelated stage in between | The weights are evicted from L2 before the next use | [Warm reused weights](#reduce-redundant-work-and-improve-data-reuse) |
| Sequential AICPU dispatch trail per region | Region issues one kernel per iteration | [Dispatch blocks with `pl.spmd`](#adjust-task-granularity-along-the-critical-path) |

A gap on this trace is not automatically a scheduling problem: the interval
before a task splits into producer-FIN detection, ready-but-undispatched
scheduler delay, and post-dispatch pickup, and each has a different fix. See
[Dependencies and Scheduling](dependency-and-scheduling.md) for how edges are
formed, what the four per-task timestamps mean, and how to attribute a gap
without guessing.

### Tuning rules

Use these as hypotheses to test on the current workload. Recheck the critical
path after each change; a task's criticality can change with the schedule.

#### Adjust task granularity along the critical path

For critical-path tasks, try smaller work units and more parallelism until
scheduling, synchronization, or lost data reuse offsets the gain. Further
splitting has diminishing returns.

For noncritical tasks, try larger work units and fewer workers. Use the time
available before their consumers need the results to reduce dispatch and
resource contention. Larger tasks must neither delay those consumers nor hold
cores long enough to block critical work.

**Expose independent iterations.**

Use outer `pl.parallel` when independent iterations should be exposed to
the scheduler. Use `pl.range` for carried state or to batch work inside one
InCore task, including independent work that benefits from coarser granularity.
Independence permits parallel execution; it does not require a separate task
for every iteration.

```python
# the batch tile is independent — pl.parallel
for b0 in pl.parallel(0, BATCH, BATCH_TILE):
    ...
```

Expose independent work with `pl.parallel`, then measure whether the added
parallelism shortens the critical path.

**Split critical work across cores.**

When one InCore body contains independent chunks, expose those chunks as
parallel blocks of a `pl.spmd` task. This example assumes `hidden` is divisible
by `Q_OUT_CHUNK`. Otherwise, round the block count up and handle the final
partial chunk in the body; floor division would omit it.

```python
# Before: one giant InCore region over all q_out blocks
with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
    for q0 in pl.range(0, hidden, Q_OUT_CHUNK):
        ...

# After: one scheduling unit with parallel blocks of the same body
for q_block in pl.spmd(hidden // Q_OUT_CHUNK, name_hint="q_proj"):
    q0 = q_block * Q_OUT_CHUNK
    ...
```

**Coarsen noncritical work inside each core.** Move part of an outer
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

**Choose SPMD or MPMD around the dependency structure.**

An InCore launch and a `pl.spmd` group each form one scheduling unit. The
SPMD unit contains multiple blocks of the same InCore body. An outer
`pl.parallel` can expose independent chains of InCore launches or SPMD groups
to the scheduler (MPMD).

For a single InCore body that needs parallel execution, prefer `pl.spmd`:

```python
for q_grid in pl.spmd(Q_ON * N_SUB, name_hint="q_proj_fused_dequant"):
    ...
```

For multiple dependent InCore stages repeated over independent chunks,
consider `pl.parallel` around the chain, with each stage expressed as an
InCore launch or a `pl.spmd` group. This exposes opportunities to overlap
stages from different chunks and hide stalls:

```text
Chunk 0: A0 -> B0 -> C0
Chunk 1: A1 -> B1 -> C1
Chunk 2: A2 -> B2 -> C2
```

With chunk-local dependencies, B0 can become ready when A0 finishes while
A1 and A2 are still running. In contrast, grouping all A blocks into one
SPMD task and all B blocks into another makes the completion dependency
coarser: B waits for the whole A group. MPMD can preserve more overlap at
the cost of additional scheduling units and dependency tracking.

Choose between these forms by measuring the scheduling overhead against the
overlap gained. Preserve true cross-chunk dependencies and scratch-buffer
lifetimes; do not assume chunks are independent merely because their work
can be split. SPMD is a useful default for one parallel body, not a universal
replacement for parallel chains of kernels.

#### Reduce redundant work and improve data reuse

Share quantized inputs, reuse weights across more rows, and eliminate
unnecessary intermediate tensors and layout conversions. Choose compute and
transfer tiles for the reuse they provide; quantization and output stages can
retain smaller tiles when needed.

**Warm weights that will be reused.**

When a stage re-reads a fixed weight set every layer and something between two
layers evicts it, an SDMA cache warm (`pl.prefetch`) can hide the reload behind
compute that is already running. It writes no tensor, so it is free to try and
free to delete — but a partial or oversized warm costs more than it saves. See
[L2 Prefetch](l2-prefetch.md).

**Use NZ layout and cache bypass where appropriate.**

NZ layout can avoid repeated weight reformatting during loads. Cache bypass
is a separate candidate for streamed weights with no useful reuse, to avoid
evicting reusable data. Declare the actual byte layout and test the cache
policy independently:

| Declaration | What it changes | What it costs |
| --- | --- | --- |
| `pl.Tensor[[E, N, K], pl.INT8, pl.NZ]` | the cube loads the weight NZ→NZ instead of reformatting ND on the way into L1, so the contiguous run is one fractal column block rather than one K-tile row | the fixture must write the bytes in fractal order (`utils.pack_nz`) and the golden must read them back (`utils.unpack_nz`) |
| `pl.set_cache_policy(w, pl.CachePolicy.BYPASS)` at the top of the reading scope | the load is issued against the device's uncached alias, so it does not evict what *does* have reuse | nothing to the kernel; the author asserts nobody writes those bytes while it runs |

NZ changes the contiguous transfer shape as well as removing ND-to-NZ
conversion. Its benefit depends on the logical tile, physical strides, dtype,
and target load implementation; do not infer a fixed bandwidth gain from the
layout name.

Keep the fixture, checkpoint upload, kernel annotation, and golden consistent
with the physical byte order. Merely adding `pl.NZ` does not repack data.
For supported slicing, reshaping, stores, and dtype-dependent fractal sizes,
consult the selected PyPTO revision's
[BlockNzTensorViews rules](https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/passes/16-block_nz_tensor_views.md).
NZ fractal constraints are distinct from GM access granularity and L2 cache
line size. Validate the exact target and shape with the compiler.

**Measure both separately.** They are independent, and which one pays depends on
whether the ND load is already at its bandwidth ceiling: at a coarse blocking
where ND already saturates, NZ can be worth nothing while the bypass still pays,
and at a finer blocking the order reverses. One `PYPTO_BENCH` run per variant on
the same frozen inputs settles it.

#### Fuse tasks with the full dataflow in mind

Before fusing stages, check other consumers, available parallelism, load
balance, and write ownership. A fused task can save transfers while delaying
another branch that previously consumed an intermediate result earlier.
Fewer tasks alone do not establish a performance improvement.

**Merge consecutive `pl.at` blocks.** Adjacent `pl.at` regions in the
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

**Combine Cube and Vector stages into MIX.** Try this when cooperating stages
frequently exchange data. Tune their pipeline, tile sizes, and buffering
together to reduce intermediate transfers and keep both sides progressing.

When a matmul (cube) and its epilogue (cast / add / norm — vector) sit in
separate `pl.at` regions,
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

#### Improve intra-kernel efficiency

Faster compute and transfers shorten critical tasks or free resources for
other work. Check the overall effect when an optimization consumes additional
cores or bandwidth.

Choose tiles against the generated memory report at
`report/memory_after_AllocateMemoryAddr.txt`: `Mat`, `Acc`, and `Vec` are the
relevant DSL constraints; `Left` and `Right` reflect compiler-managed staging.
See [Cube Tile Tuning](cube-tile-tuning.md) for buffer interpretation and the
M/N/K sweep method, and [L1/L0 tuning](#part-2-l1-l0-tuning-intra-kernel) below
for transfer granularity, pipelining, and PMU analysis.

---

## Part 2: L1-L0 tuning (intra-kernel)

When the swimlane identifies expensive kernels, inspect their internal
pipelines and resource use. Kernel improvements can change task granularity
and the critical path, so revisit the L2 schedule after significant changes.

### Capture

For per-kernel PMU counters on `decode_csa.py`, add `enable_pmu=2` to the
existing `config=dict(...)` passed to `run`, preserving its other settings
and validation options, then run:

```bash
python models/deepseek_v4_flash_mtp/decode_csa.py -p a2a3 -d 0
# → build_output/<...>/dfx_outputs/pmu.csv
```

This entry does not expose `--enable-pmu`; `RunConfig` carries the setting
to the runtime as a DFX option. Remove the setting for ordinary benchmark
comparisons.

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

### Platform-specific constraints

Use the selected PyPTO revision and target backend as the source of truth for
sizes and alignment. Distinguish performance recommendations from operation
legality and allocator alignment; they are not interchangeable.

The current backend definitions expose these values:

| Backend | GM access granularity | PH001 recommended minimum trailing dimension | L2 cache line |
| --- | ---: | ---: | ---: |
| `a2a3` / Ascend910B | 512 B | 512 B | 512 B |
| `a5` / Ascend950 | 128 B | 128 B | 512 B |

Check `GetGmAccessGranularityBytes`, `GetRecommendedInnermostDimBytes`, and
`GetL2CacheLineBytes` in the
[A2/A3 handler](https://github.com/hw-native-sys/pypto/blob/main/include/pypto/backend/910B/backend_910b_handler.h)
and [A5 handler](https://github.com/hw-native-sys/pypto/blob/main/include/pypto/backend/950/backend_950_handler.h).
These links point to main; use the files at the revision that compiled the
case when reproducing a measurement.

For buffer capacities, core counts, and allocator alignment, inspect PyPTO's
[SoC definitions](https://github.com/hw-native-sys/pypto/blob/main/src/backend/common/soc.cpp)
and the target's `Backend::GetMemSize`, `GetCoreCount`, and `GetMemAlignment`.
Use `report/memory_after_AllocateMemoryAddr.txt` to check the actual allocation
for each candidate, including buffering and compiler-reserved space. Use the
operator's verifier and generated code for dtype, layout, and instruction
constraints; do not carry another platform's tile limits into a sweep.

### Tuning rules

**Reduce redundant transfers and computation.** Check InCore loops for
repeated loads of unchanged data, quantization, layout conversions, and
type casts. Hoist loop-invariant work and reuse intermediate results where
dependencies and numerical behavior permit. Check the generated memory
report: keeping intermediates alive longer increases on-chip memory usage
and can constrain tile sizes or pipeline buffering.

#### 1. Fix tile-shape MTE hints from `perf_hints.log`

Every compile writes a perf-hint log next to the memory report:

```
build_output/<ProgramName>_<ts>/report/perf_hints.log
```

PH001 flags GM tile accesses whose innermost dimension is below the target
backend's recommended size. This is a performance advisory, not a legality
error. For example, an A2/A3 build may report:

```
[perf_hint PH001] TileInnermostDimGranularity: tile.load has innermost
dim = 256B; recommended >= 512B for backend a2a3 (L2 cache line = 512B).
Consider increasing tile shape on the innermost axis.
at models/deepseek_v4_flash_mtp/qkv_proj_rope.py:68:4
```

Inspect the flagged accesses on expensive paths and try wider contiguous
transfers where shapes and buffer budgets allow. On A5, a 128 B access meets
the current hint threshold even though the L2 cache line is 512 B. Measure
whether further widening helps; extra padding, redundant work, or reduced
parallelism can erase the transfer benefit.

#### 2. `pl.pipeline` for ping-pong on the K loop

Try `pl.pipeline` for matmul K loops to overlap loads with computation.
Compare depths such as 2 and 4 against the current implementation, checking
the generated pipeline and buffer allocation. The useful depth depends on
the target, tile sizes, K-loop length, and available memory.

```python
# Candidate depth 4: check its extra buffering cost
for kb in pl.pipeline(HIDDEN // K_STEP, stage=4):
    ...

# Candidate depth 2
for kb in pl.pipeline(0, hidden // K_STEP, stage=2):
    ...
```

A plain `pl.range` does not request this software pipeline, but its presence
alone does not prove a stall on every load. Inspect actual load/compute
overlap, and keep additional pipeline stages only when timing improves.

#### 3. Watch `pl.slice` / `pl.assemble` granularity

Choose trailing tile extents using the target's GM access granularity and
the element size. For byte-addressable dtypes, these counts correspond to one
access unit in the current backend definitions:

| Dtype | A2/A3: 512 B | A5: 128 B |
| --- | ---: | ---: |
| BF16 (2 B) | 256 elements | 64 elements |
| FP32 (4 B) | 128 elements | 32 elements |
| INT8 (1 B) | 512 elements | 128 elements |

These are transfer-sizing guides, not universal legal shape or address
alignment requirements. A larger trailing dimension does not guarantee
aligned row addresses: inspect base offsets, row strides, physical layout,
and tail handling as well. Packed dtypes and NZ layouts require their own
physical-layout calculation. Evaluate padding and buffer costs before
widening tiles to a full cache line.

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

Use `*_busy_cycles / pmu_total_cycles` to locate the pipelines that occupy
most of the kernel's execution window. Busy time measures occupancy, not
useful FLOPs or transferred bytes: high MTE2 occupancy does not prove peak
bandwidth, and high Cube occupancy does not prove efficient matrix shapes.
Pipelines overlap, so their busy ratios need not sum to 100%.

For a dominant transfer pipe, inspect bytes moved, layout conversions, and
access granularity. For a dominant compute pipe, inspect useful work, tile
shape, and instruction mix. Compare achieved throughput against an appropriate
reference for the same target, dtype, and operation when one is available.

Use the intra-core trace to inspect overlap, dependencies, synchronization,
and pipeline startup or drain. Low compute and MTE2 ratios may also mean
another pipe dominates; include MTE1, MTE3, FIXPIPE, and scalar work in the
analysis. Select changes by measured latency, using PMU to explain the result.
