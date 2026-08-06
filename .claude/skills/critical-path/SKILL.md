---
name: critical-path
description: Find the execution critical path of a PyPTO operator from a level-4 L2 swimlane, report every path task and its preceding gap, mark gaps over 1 µs and tasks actually dispatched early, and identify defensible dispatch or core blockers. Use for `/critical-path operator`, operator latency investigations, multi-card fast-rank analysis, scheduler gaps, and early-dispatch verification.
---

# Critical Path

Analyze one operator run in two stages:

1. Capture a level-4 L2 swimlane and reconstruct the official Observed critical
   path with `simpler_setup.tools.critical_path`.
2. Investigate every path gap over 1 µs using task dependencies, AICPU
   dispatch/finish timestamps, and physical-core occupancy.

Call this profiling tier **"L2 swimlane perf level 4"**. Do not confuse it with
the runtime hierarchy's L4 worker level.

## 1. Resolve the operator and capture command

Treat the argument to `/critical-path` as either a Python path or an operator /
program name.

1. If it is a path, use that file.
2. Otherwise, search `models/` and `examples/` for an exact filename, class, or
   program-name match. Ask the user only if more than one plausible operator
   remains.
3. Reuse the operator's normal platform, shapes, weights, and multi-card device
   list. A smaller or synthetic case answers a different performance question.
4. Inspect the operator's `argparse` definition before choosing the swimlane
   flag:
   - `action="store_true"`: pass bare `--enable-l2-swimlane`; `True` maps to
     level 4 in the runtime binding.
   - An integer / optional-value argument accepting 4: pass
     `--enable-l2-swimlane 4` explicitly. A bare flag often means level 1.
   - An argument whose choices exclude 4: stop and explain that this CLI cannot
     produce the required capture. Do not analyze level 1/2 as level 4 or
     silently edit the operator.

Activate the worktree environment before running:

```bash
source temp/set_env.sh
source "$PYPTO_ROOT/toolchain/versions.env"
SKILL_PTOAS_VERSION=${PTOAS_VERSION#v}
```

Create a timestamp marker immediately before the run so an older build is never
selected by mtime:

```bash
CAPTURE_MARKER=$(mktemp)
touch "$CAPTURE_MARKER"
task-submit --device auto --ptoas "$SKILL_PTOAS_VERSION" \
  --timeout 3600 --max-time 0 --run \
  'cd <absolute-worktree-path> && \
   SKILL_PTOAS_INSTALL="$PTOAS_ROOT" && source temp/set_env.sh && \
   if test -f "$SKILL_PTOAS_INSTALL/ptoas" && test -x "$SKILL_PTOAS_INSTALL/ptoas"; then \
     export PTOAS_ROOT="$SKILL_PTOAS_INSTALL"; \
   elif test -f "$SKILL_PTOAS_INSTALL/bin/ptoas" && test -x "$SKILL_PTOAS_INSTALL/bin/ptoas"; then \
     export PTOAS_ROOT="$SKILL_PTOAS_INSTALL/bin"; \
   else echo "invalid PTOAS install: $SKILL_PTOAS_INSTALL" >&2; exit 2; fi && \
   export PATH="$PTOAS_ROOT:$PATH" && \
   python <operator.py> <normal operator arguments> --device "$TASK_DEVICE" \
     --enable-l2-swimlane 4'
find build_output -type f -name l2_swimlane_records.json -newer "$CAPTURE_MARKER" -print
rm -f "$CAPTURE_MARKER"
```

Use the correct bare-flag form instead when required by the operator CLI. Adapt
the device option to the executable; for a multi-card run, request the normal
card count with `--device-num N` and preserve its normal device-list convention.
Keep `$TASK_DEVICE` inside the single-quoted submitted command so it expands only
after `task-submit` assigns the NPU. Read the repository's declared PTOAS
version, let `task-submit` select that installation, preserve its injected path
across `temp/set_env.sh`, and then normalize the archive layout as shown.
Require a regular executable file: `test -x` alone is insufficient because it
also succeeds for a searchable directory. Archive layouts differ, so prefer a
working top-level wrapper and otherwise use the `bin/` wrapper; do not assume
either layout blindly.
A toolchain preflight or compile failure is not an NPU initialization failure;
do not report it as one.

On current onboard PyPTO, enabling the swimlane performs a graph-only dep-gen
pass in a child process and a separate clean timing pass automatically. Analyze
the timing pass. Never intentionally co-run dep-gen instrumentation in the
timing pass; it perturbs the numbers. If an older runtime lacks the automatic
two-pass flow, capture `deps.json` in a separate run of the same topology and
join it with the timing artifacts offline.

## 2. Validate one fresh artifact set

For every selected single-card or distributed dispatch directory, require:

```text
l2_swimlane_records.json   # legacy l2_perf_records.json is also readable
deps.json
name_map*.json
merged_swimlane*.json      # optional for analysis; useful in Perfetto
```

Distributed artifacts use:

```text
<work_dir>/dfx_outputs/rank{r}/d{k}/
```

`d{k}` is that card's kth L2 dispatch. Use `dispatch_program.json` to match the
requested operator; never compare different programs that happen to reuse the
same `func_id`.

Verify every records file is truly level 4:

```bash
jq -e '.l2_swimlane_level == 4' <records>
```

Reject the comparison if any candidate rank is missing dependencies, names, or
the requested dispatch. Silently dropping an incomplete rank can select the
wrong "fast card".

Use `swimlane_converter.read_perf_data()` for all task-level analysis. The raw
file contains separate cycle-domain streams; do not join them with ad hoc
`json.load` code. Reject the capture unless raw AICore rows, raw AICPU rows, and
joined rows have equal counts. For every task in `deps.json`, also require:

```text
joined physical rows == deps.block_num * active deps.kernel_ids slots
```

This catches a dropped SPMD/MIX row before it can produce a false `full N/N`
early-dispatch result or incomplete blocker window.

## 3. Reconstruct and report the path

Run the canonical analyzer as a module:

```bash
python -m simpler_setup.tools.critical_path <work_dir> --stdout
```

The canonical module writes reports beside the capture. If any selected
artifact directory is not writable, copy the complete selected artifact set to
a writable directory created with `mktemp -d`, run both analyzers on that exact
copy, and report both the original and working-copy paths. Do not modify
permissions or write into a shared/read-only capture.

This writes `critical_path_report.md` and, when a merged trace exists,
`CPM_static.json` / `CPM_observed.json` beside each records file.

It computes both paths:

- **Static CPM**: duration-weighted longest dependency path with unlimited
  cores. Report its duration as a cross-check.
- **Observed**: the as-executed backward blame path through data and same-core
  predecessors. Use this as the primary path because its task compute and
  preceding stalls tile the measured AICore makespan.

Then generate the operator-facing report:

```bash
python .claude/skills/critical-path/scripts/report.py \
  <work_dir> --operator <operator-or-dispatch-program> \
  -o <work_dir>/critical_path_summary.md
```

For an unambiguous direct build with no `dispatch_program.json`, still pass the
requested operator to the bundled helper: it is used only as the display label.
The helper refuses this fallback when a rank contains multiple dispatches. Use
`--rank rankN` only when a reliable external per-rank operator timer should
override automatic selection.

The helper:

- compares equivalent ranks by summed `dispatch → finish` elapsed and selects
  the smallest value, matching the requested fast-card convention;
- also reports the AICore makespan used by the critical-path algorithm;
- lists every Observed-path task in execution order;
- labels each task's full wall span separately from its non-overlapped Observed
  compute contribution, because wall spans can overlap and need not sum to the
  makespan;
- displays the first row's gap as `—`;
- marks a later row 🐌 only when its Observed-path stall is **strictly greater
  than 1.0 µs**;
- reports the gap kind and task ID so repeated operator names remain
  distinguishable.

If one rank contains several selected `d{k}` dispatches, sum their elapsed time
to select the rank and report each dispatch's path separately. State explicitly
that fast-card selection overrides the common end-to-end slowest-card headline.

## 4. Mark actual early dispatch

Do not add ⭐ merely because the current task has
`deps.json::early_dispatch=true`. That field marks an
`allow_early_resolve` **producer**, not a consumer that was dispatched early.

For consumer `C`:

1. Deduplicate its direct dependency edges.
2. Apply the runtime's structural rule: every direct predecessor must be either
   an alloc source or carry `early_dispatch=true`, and at least one predecessor
   must be a non-alloc early producer.
3. Fold every predecessor's AICPU finish time across its blocks:
   `observed_ready(C) = max(pred.finish_time_us)`.
4. Add ⭐ only if structurally eligible and at least one block of `C` satisfies:

   ```text
   C.dispatch_time_us + tolerance < observed_ready(C)
   ```

   Use the same two-clock-tick tolerance as `critical_path.py`.

Report `⭐ partial N/M` when only some SPMD/MIX rows prove early dispatch. If the
timestamps look early but the graph is not structurally eligible, report a
dependency/timing mismatch rather than adding ⭐.

## 5. Investigate every 🐌 task

Split the interval before task `C` into distinct causes:

```text
producer end → producer finish    FIN detection / dependency resolution
last producer FIN → C dispatch    ordinary-ready but undispatched scheduler delay
C dispatch → C start              post-dispatch gate / pickup / core wait
```

Use:

```text
data_ready(C) = max(pred.end_time_us)
observed_ready(C) = max(pred.finish_time_us)
dispatch(C) = min(block dispatch_time_us)
start(C) = min(block start_time_us)
```

Apply these attribution rules:

- If `dispatch(C) < observed_ready(C)` and ⭐ is proven, no predecessor blocked
  `C`'s dispatch; it was already dispatched. Analyze its later gate/start wait.
- List every non-alloc direct predecessor lacking
  `early_dispatch=true` as an **early-dispatch policy blocker**. Such a task
  prevents `C` from entering speculative early dispatch.
- If `dispatch(C) > observed_ready(C)`, report the post-FIN ready→dispatch
  scheduler delay. Report `data_ready → observed_ready` separately as producer
  FIN detection/dependency resolution. Ordinary level-4 scheduler phase rows
  carry counts, not causal task IDs; do not invent a named blocker from temporal
  proximity.
- Compute task-global `start(C) = min(block start_time_us)`. Inspect only the
  physical row(s) whose start realizes that global earliest start. For each such
  row, inspect its core only during
  `[max(data_ready(C), row.dispatch_time_us), start(C))`, and name another task
  as a **same-core start blocker** only when its occupancy continues to
  `start(C)` within tolerance. An earlier task that freed the core before data
  readiness did not gate this critical-path start. This remains start
  contention, not automatically a dispatch blocker; keep it separate from
  proven full-engine dispatch saturation.

For a strong named **dispatch resource blocker** claim, prove capacity
saturation rather than inspecting only the eventual core:

1. Reconstruct each physical row's conservative descriptor occupancy as
   `[dispatch_time_us, finish_time_us)`.
2. Account for the runtime's running and pending descriptor slots.
3. Show that all compatible cores required by `C`'s engine/launch shape lacked a
   free slot throughout the post-FIN ready→dispatch interval.
4. List the tasks occupying the slots that must finish before capacity appears.

For MIX, SPMD, or `sync_start` launches, inspect the cluster/block requirement
manually; task records alone may not prove simultaneous launch capacity. If
saturation cannot be proved, label overlapping tasks as candidates or report
the scheduler delay as unattributed.

To visualize engine-level ready-undispatched intervals:

```bash
python -m simpler_setup.tools.swimlane_converter \
  <records> --deps-json <deps.json> --overhead \
  -o <dispatch-dir>/merged_swimlane_overhead.json
```

## 6. Present the result

Return:

1. Selected rank/device, selection rule, dispatch→finish operator elapsed, and
   AICore makespan.
2. The complete Observed-path table:

   ```text
   # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers
   ```

3. A blocker table for all 🐌 rows, separating:
   - early-dispatch policy blockers;
   - proven dispatch resource blockers;
   - post-dispatch same-core blockers;
   - unnamed scheduler delays where task-level causality is unavailable.
4. The Static CPM duration as a cross-check.
5. These limitations:
   - the critical-path makespan covers first AICore start through last AICore
     end, not host/orchestrator front time or AICPU/host tail time;
   - level-4 collection has observer cost;
   - production performance should still come from an unprofiled benchmark.

Never call a coincident task a blocker without the corresponding dependency,
policy, or capacity evidence.
