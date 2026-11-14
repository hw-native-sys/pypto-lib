---
name: perf-ideal-schedule
description: Capture a level-1 PyPTO swimlane and estimate a resource-constrained ideal task schedule, including off-path core contention and MIX placement. Use for best-schedule kernel latency estimates and dependency-only versus finite-core comparisons, rather than benchmark medians or observed dispatch-gap attribution.
---

# Ideal Schedule Performance

Run the requested case with numerical validation, collect a level-1 chip
swimlane, and compute a physically core-feasible ideal schedule. Report it as
**a conservative task-gang model estimate**, never as measured latency or a
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
32-AIC/64-AIV topology; unsupported topologies or multi-wave launch sizes fail
explicitly instead of producing guessed results.

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
  core until this task duration expires. This is conservative for launches
  that can actually stagger blocks or release short worker lanes early.
- Allocate pure AIC/AIV blocks from their corresponding capacity. MIX requires
  an AIC and its associated one/two AIV cores in the same cluster.
- Allow new task order and new core assignments. Discard observed same-core
  predecessor edges and inter-task idle gaps.

Minimize completion of **all included compute tasks**, including off-path cache
publication/pool outputs, rather than stopping at an arbitrarily named HC post.
This accounts for branches competing for cores; branches with sufficient slack
can still be hidden.

CP-SAT first solves engine-capacity scheduling, then verifies a physical
paired-core assignment at those exact times. Independent checks reject core
capacity excess or overlapping reservations. If placement is unavailable,
retain the engine result only as diagnostic information and do not report it
as a feasible best schedule. Emit the engine candidate, bound, status, and
placement status even on placement failure; mark that dispatch and its rank
feasible total as unavailable. Timeouts are configurable with `--timeout`.

- `OPTIMAL` plus a paired-core witness proves optimality **within this model**.
- `FEASIBLE` is a model-feasible upper bound; report the solver lower bound too.
- The unlimited-resource dependency path is a cross-check, not a substitute
  for finite-core scheduling. Report the additional contention separately.

Do not describe this as a hardware lower bound. Fixed measured durations retain
kernel-internal waits and capture-time contention. The model does not recompute
memory bandwidth, cache interference, remote-peer progress, descriptor slots,
or changes in kernel duration under a different execution order. A finer
per-block model can reduce the conservative gang-model result.

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
