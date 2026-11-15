---
name: perf-ideal-schedule
description: Capture a level-1 PyPTO swimlane and estimate a resource-constrained ideal task schedule, including oversized launches, off-path core contention, MIX placement, and actionable parallel-demand diagnostics. Use for best-schedule kernel latency estimates and dependency-only versus finite-core comparisons, rather than benchmark medians or observed dispatch-gap attribution.
---

# Ideal Schedule Performance

Run the requested case with numerical validation, collect a level-1 chip
swimlane, and compute a physically core-feasible ideal schedule. Report it as
**a conservative ordered-wave task-gang model estimate**, never as measured latency or a
proof of hardware-optimal execution.

## Capture

Use [perf](../perf/SKILL.md) for case/platform resolution, queue allocation,
environment checks, capture ownership and validation. Force `PYPTO_BENCH=0`
and `--enable-chip-swimlane 1`. Preserve the user's shapes, routing, EP and
experts per rank; obtain the actual device count from the workload. Do not
silently substitute perf-ci defaults for a requested configuration.

Prefer one fresh capture. Reuse compatible compiled builds and frozen golden
inputs for subsequent runs following the benchmarking rules. A failed run does
not become a valid measurement by skipping validation or relaxing a threshold.
Any user-authorized tolerance change remains explicit in the result.

Keep each dispatch's complete artifact set together:

```text
chip_swimlane_records.json
deps.json
name_map*.json
dispatch_program.json       # when emitted
```

Use the work directory returned by this run, not the newest directory globally.
For a provided capture, verify its program/configuration and known validation
status. Raw timing artifacts alone do not establish numerical correctness.
The helper requires matching dispatch_program.json identities across ranks.
Check that the matching identities also represent the requested workload. For a comparison with
several dispatches per rank, retain every matched dispatch, sum their included
AICore spans to select the fastest rank, and sum that rank's separate schedule
estimates. This does not jointly reschedule separate dispatch programs or
simulate cross-rank communication. Report other ranks rather than hiding skew.

## Solve

The helper requires the active PyPTO runtime (for `read_perf_data`) and
OR-Tools. If OR-Tools is absent, install it into an isolated target or use an
existing compatible isolated installation, retaining the selected runtime's
Python environment. Do not embed a host-specific solver location in the skill:

```bash
python -m pip install --target /tmp/perf-ideal-schedule-deps ortools
PYTHONPATH="/tmp/perf-ideal-schedule-deps${PYTHONPATH:+:$PYTHONPATH}" \
  python .claude/skills/perf-ideal-schedule/scripts/analyze.py \
  <work-dir> --expected-ranks <N> --output-dir <report-dir>
```

For one explicit rank/dispatch, pass its `chip_swimlane_records.json` instead.
That analyzes only the supplied rank and cannot establish the fastest rank.
Distributed directory inputs require `--expected-ranks` so an incomplete rank
cannot silently disappear. The current helper supports A5's contiguous
32-AIC/64-AIV topology; unsupported topologies fail explicitly. Launches larger
than the engine capacity become ordered waves, including a partial final wave.
Every wave retains the original task's maximum worker duration. Incoming
dependencies enter the first wave; consumers wait for the last. The report
retains original task IDs, wave sizes and the original dependency-only value;
the primary dependency path and bounds use the expanded graph. Raw captures
are never rewritten.

Weights default to **execution only**: `max_worker(end - start)`. This reproduces
the agreed kernel-only metric. Local setup is excluded from both duration and
resource occupancy in this mode. To include mandatory AICore local setup, add:

```text
--include-setup
```

This uses `max_worker(end - receive)` on the **same worker**. Never add a
separately maximized setup value to a separately maximized execution value.
State the chosen mode and compare only identical modes. `receive -> start`
may include local gating/pickup effects; it is not guaranteed to be removable
initialization code. AICPU dispatch/queue gaps are excluded in both modes.

`moe_signal_clear` is excluded by default, preserving its graph edges at zero
weight; use `--keep-signal-clear` when that cleanup belongs to the requested
scope. Additional exclusions require an explicit scope justification and are
listed with `--exclude-name NAME`. Do not drop producer or side-branch tasks
merely because they are absent from the dependency critical path.

## Model and checks

The helper reads and joins rows through `swimlane_converter.read_perf_data`,
checks capture level, execution epoch, physical cores, raw/decoded row counts,
and each task's `block_num * active kernel slots`, including separate AIC/AIV
worker counts. It retains every dependency edge, regardless of optional
`flags`, and zero-duration allocation/dummy nodes, and rejects graph cycles.

For each included task:

- Use its longest physical worker duration, including all AIC/AIV lanes of MIX.
- Start all its blocks together as an indivisible gang; hold every occupied
  core until this task duration expires. For oversized launches, apply this
  rule separately to sequential capacity-sized waves. This is conservative for launches
  that can actually stagger blocks or release short worker lanes early.
- Allocate pure AIC/AIV blocks from their corresponding capacity. MIX requires
  an AIC and its associated one/two AIV cores in the same cluster.
- Allow new task order and new core assignments. Discard observed same-core
  predecessor edges and inter-task idle gaps.

Minimize completion of **all included compute tasks**, including off-path cache
publication/pool outputs, rather than stopping at an arbitrarily named HC post.
This accounts for branches competing for cores; branches with sufficient slack
can still be hidden.

The helper first constructs a physical list schedule, prioritizing the longest
remaining dependency chain. It groups ready tasks by core demand, and validates
precedence, exact worker counts, MIX slot pairing and core overlap independently.
This provides a feasible fallback and a CP-SAT warm start.

`--solver auto` (default) uses CP-SAT up to `--exact-task-limit 500` expanded
graph nodes, then uses the verified list schedule for larger graphs. Use
`--solver heuristic` for a quick estimate or `--solver exact` to explicitly
attempt CP-SAT on a larger graph. `--timeout` bounds each optimization and
placement solve separately; parsing, graph construction and checks are outside
that limit. Do not keep increasing timeouts without a useful bound improvement.

CP-SAT solves engine-capacity scheduling, then checks physical paired-core
placement at those times. On timeout or unavailable placement, retain the engine
candidate/status as diagnostics and report the independently verified fallback.
A candidate without placement is never promoted to a feasible schedule. Report
the chosen method, fallback reason, dependency/workload lower bound, solver bound
when available, and feasible-minus-bound gap. `HEURISTIC` is an upper bound within
this model and carries no optimality claim, even if it happens to meet the bound.

- `OPTIMAL` plus a paired-core witness proves optimality **within this model**.
- `FEASIBLE` is a model-feasible upper bound; report the solver lower bound too.
- The unlimited-resource dependency path is a cross-check, not a substitute
  for finite-core scheduling. Report the additional contention separately.

Do not describe this as a hardware lower bound. Fixed measured durations retain
kernel-internal waits and capture-time contention. The model does not recompute
memory bandwidth, cache interference, remote-peer progress, descriptor slots,
or changes in kernel duration under a different execution order. Ordered waves
also impose extra barriers and charge the slowest worker to every wave; they do not reconstruct actual wave timing. A finer
per-block model can reduce the conservative gang-model result.

## Parallelism diagnostics and optimization advice

Surface capacity warnings in the user-facing result, not only in JSON:

- **Oversized launch:** name, original task ID, block count, AIC/AIV demand,
  capacity, wave count and partial final wave. Suggest inspecting tiling,
  coarsening work per block or moving independent iterations into a block-local
  loop, subject to buffer limits and measured load balance.
- **Competing branches:** the expanded DAG's unlimited-core ASAP timeline
  identifies potential demand above 32 AIC / 64 AIV. Report the time, engine,
  demand/capacity, concurrent task count and contributing names/IDs. The report
  also samples the largest overloaded intervals by excess core-time; this ranks
  inspection candidates, not predicted latency savings. This is
  hypothetical dependency-ready demand, not observed occupancy or a proof that
  every feasible schedule must contend at that point. Use the verified optimized
  timeline to assess actual model delay before recommending changes.

Compare worker demand, not task count alone: one MIX task may occupy the whole
chip. Suggest fewer workers or coarser tasks on noncritical branches when that
could leave cores for critical work. Extra waves can improve load balance;
oversubscription alone does not establish a performance defect. Never suggest
blindly clamping block counts, removing work, or serializing all parallel loops.
Any kernel change needs numerical validation and unprofiled wall-time evidence.

Launch metadata cannot reveal every logical tile inside a worker. When observed
worker durations are imbalanced even though the launch fits, inspect generated
kernel loop/tile counts and the short final wave (for example, 40 logical chunks
on 32 workers). Explain this separately from exposed launch oversubscription;
do not infer logical tile counts from task counts or treat model reservations
as evidence of actual idle cores.

## Deliver

The helper writes `results.json` (per-task timing and explicit core placement)
and `report.md`, including per-rank sums across dispatches. Keep them beside the selected raw captures or link to those
captures without losing dependencies and names. Include configuration,
revision/toolchain, validation status, selected rank/devices and weight mode.
A useful table is:

```text
case | dependency path us | feasible schedule us | model bound us | status
```

Call out major contention sources using the actual optimized task timeline;
solver-selected ordering is one optimum, not proof of a unique causal order.
Compare CANN or another reference only when requested. State differences in
routing/EPLB, EP topology, weights and measurement boundaries; do not equate
CANN whole-operator duration with a measured per-core statistic. Production
wall-time claims still require an unprofiled benchmark through `perf`.

## Helper validation

After changing scheduling logic, run `scripts/test_scheduling.py` with the same
isolated OR-Tools environment. It covers multi-wave work conservation and
precedence, partial/MIX waves, pressure diagnostics, timeout fallback and a large
parallel DAG. Reanalyze compatible existing captures for real-data regressions;
a new device capture is unnecessary for changes confined to this analyzer.
