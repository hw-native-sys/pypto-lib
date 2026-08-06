---
name: early-dispatch
description: Enable and verify speculative early dispatch for a named small PyPTO operator by marking every direct producer with allow_early_resolve=True, using before/after level-4 L2 swimlanes to measure its start gap and prove actual scheduler staging. Use for `/early-dispatch operator-name`, reducing gaps before short operators, adding producer-side early-resolve hints, or diagnosing why an eligible task was not early-dispatched.
---

# Early Dispatch

Make one named small operator start sooner by opting in every direct producer,
then prove the result with fresh before/after **L2 swimlane perf level 4**
captures.

Keep these invariants:

- Add `allow_early_resolve=True` to the target's direct **producers**, not to the
  target. Marking the target only helps its own consumers.
- Treat `deps.json::tasks[].early_dispatch=true` as a compiled producer flag,
  not proof that the task itself was dispatched early.
- Call a target actually early-dispatched only when it is structurally eligible
  and its per-block dispatch timestamp precedes the latest direct-producer FIN.
- Change PyPTO operator source only. Do not patch the runtime scheduler.

## 1. Resolve the target and its enclosing run

Treat the argument as the exact compiled task/operator name shown in
`name_map*.json`. Also accept an unsuffixed MIX source name only when one
logical task maps to exactly `<name>_aic` and `<name>_aiv`; show both compiled
names in the report.

1. Search `models/` and `examples/` for an exact `name_hint`, function, or
   submitted-callee match.
2. Identify the normal executable that contains the target. Preserve its real
   shapes, weights, topology, device list, and arguments.
3. If the name resolves to multiple task IDs or source sites with different
   predecessor sets, list the candidates and require the intended task ID or
   source occurrence before editing. A runtime task ID is an execution
   instance, not a source identifier.
4. Inspect the executable's `argparse` definition before selecting the swimlane
   flag:
   - For `action="store_true"`, pass bare `--enable-l2-swimlane`; runtime `True`
     maps to level 4.
   - For an integer/optional-value argument accepting `4`, pass
     `--enable-l2-swimlane 4`.
   - If the argument rejects level 4, stop. Do not analyze level 1/2 or silently
     change the CLI.

Activate the worktree environment:

```bash
source temp/set_env.sh
source "$PYPTO_ROOT/toolchain/versions.env"
SKILL_PTOAS_VERSION=${PTOAS_VERSION#v}
```

## 2. Capture and inspect the baseline before editing

Create a timestamp marker immediately before running the unchanged operator.
Select only artifacts newer than the marker:

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
   python <operator.py> <normal arguments> --device "$TASK_DEVICE" \
     --enable-l2-swimlane 4'
find build_output -type f -name l2_swimlane_records.json -newer "$CAPTURE_MARKER" -print
rm -f "$CAPTURE_MARKER"
```

Use the correct bare-flag form when required and adapt the device option to the
operator. For multi-card runs, request the normal count with `--device-num N`
and preserve the executable's device-list convention. Keep `$TASK_DEVICE`
inside the single-quoted submitted command so it expands after NPU assignment.
Read the repository's declared PTOAS version, let `task-submit` select that
installation, preserve its injected path across `temp/set_env.sh`, and then
normalize the archive layout as shown. Require a regular executable file:
`test -x` alone also succeeds for a searchable directory. Prefer a working
top-level wrapper and otherwise use the `bin/` wrapper; do not assume either
layout blindly. Current onboard PyPTO performs a dependency-generation pass and
a clean timing pass automatically; analyze the timing pass. Never combine extra
dependency instrumentation with that timing pass.
Do not describe a toolchain preflight or compile failure as an NPU
initialization failure.

Verify that the two passes route the same task graph. Golden/host tensor inputs
are replayed faithfully, but the dep-gen child rebuilds device-resident inputs
as zero-filled tensors. If a device-resident tensor value controls task
creation, predicates, expert selection, loop counts, or dispatch order, stop
unless an equivalent graph capture using the real control values is available.
Matching row counts and block shapes are necessary checks, not proof that two
value-routed graphs have identical callable identities.

Require, for every selected rank/dispatch:

```text
l2_swimlane_records.json
deps.json
name_map*.json
dispatch_program.json     # required when multiple programs/dispatches must be distinguished
```

Read the raw records and require `l2_swimlane_level == 4`. Use
`swimlane_converter.read_perf_data()` for the AICore/AICPU clock join; never
join raw cycle streams by hand. Reject a capture unless raw AICore rows, raw
AICPU rows, and joined rows have equal counts. For every timed logical task,
also require:

```text
joined physical rows == deps.block_num * active deps.kernel_ids slots
```

This prevents a dropped physical row from being mislabeled `full N/N`. A direct
capture with exactly one dispatch per rank may omit `dispatch_program.json`.
When a rank has multiple dispatches, require the marker on every dispatch and
select the enclosing program by exact name.

Run the bundled helper before any source modification:

```bash
python .claude/skills/early-dispatch/scripts/report.py \
  <baseline-build-dir> \
  --target <operator-name> \
  --operator <enclosing-dispatch-program> \
  --json-out <baseline-build-dir>/early_dispatch_baseline.json \
  -o <baseline-build-dir>/early_dispatch_baseline.md
```

Omit `--operator` only for an unambiguous single-program build. Let the first
report enumerate repeated names, then use `--occurrence N` for a stable
before/after selection. Use `--task-id <id>` only for capture-local forensics.
The helper defaults to the rank with the smallest summed dispatch-to-finish
time; for a causal before/after comparison, prefer `--rank rankN` to keep the
same physical rank.

For logical target `C`, compute:

```text
S(C) = min(C block start_time_us)
E(P) = max(P block end_time_us)
F(P) = max(P block finish_time_us)

kernel start gap = S(C) - max(E(P))       # requested start-versus-end metric
post-FIN gap     = S(C) - max(F(P))
early lead       = max(F(P)) - min(C block dispatch_time_us)
```

Deduplicate direct dependency edges by `(pred, succ)`. Fold every physical
block of every non-allocation direct producer. If any such producer lacks joined
timing, mark the metric and actual-dispatch result unverifiable; do not silently
use the timed subset.

Apply a two-clock-tick tolerance:

```text
tol_us = 2 * 1e6 / metadata.clock_freq_hz
```

A target block is actually early-staged only when:

1. every direct producer is an allocation or has
   `deps.json::early_dispatch=true`;
2. at least one producer is a non-allocation task; and
3. `target_block.dispatch_time_us + tol_us < max(F(P))`.

Report `full N/N physical rows`, `partial N/M physical rows`, or `none`.

### Baseline stop and warning rules

- If the selected target is already actually early-dispatched, stop without
  modifying source and explain the proof.
- If multiple same-name occurrences are mixed, report each occurrence and do
  not apply a blanket edit until the intended task ID/source site is clear.
- If every direct producer is already flagged but actual dispatch did not
  happen, do not add duplicate flags. Diagnose the scheduler/launch condition.
- If the baseline kernel start gap is **strictly below 1 µs**, warn that the
  change may have little benefit, then continue only when a producer policy
  edit is still needed.

## 3. Map every direct producer back to source

Use this identity chain:

```text
dispatch_program.json
  → program next_levels/<program>/kernel_config.py
  → name_map*.json::callable_id_to_name
  → deps.json::tasks[task_id].kernel_ids
  → source name_hint or submitted callee
```

For every target task ID:

1. Collect and deduplicate every `deps.json::edges[].pred`.
2. Ignore predecessors absent from `deps.json::tasks`; they are allocation
   sources and are already early-resolvable.
3. Inspect all active `kernel_ids` slots. A MIX task can map to more than one
   callable name.
4. Locate the exact producer launch site. Compiler suffixes such as `_0` can be
   inlined copies of one shared `name_hint`; understand the full edit scope.
5. Verify the capture agrees with source. If the source already contains literal
   `True` but `deps.json` says false, stop: the artifact is stale, came from
   another checkout, or the source match is wrong.
6. Inspect each producer's fan-out. The flag affects all its consumers, not only
   this target.

Add the literal keyword to every unique, non-allocation direct producer:

```python
with pl.at(
    level=pl.Level.CORE_GROUP,
    name_hint="producer",
    allow_early_resolve=True,
):
    ...

for block in pl.spmd(
    block_count,
    name_hint="producer",
    allow_early_resolve=True,
):
    ...

result, task_id = pl.submit(
    self.producer,
    value,
    allow_early_resolve=True,
)
```

`pl.spmd_submit(...)` accepts the same keyword. If a site has literal `False`,
replace it; never append a duplicate keyword. The parser requires boolean
literal `True`, not a variable or `1`.

Handle exceptional forms deliberately:

- A plain kernel call has no task metadata keyword. Convert that specific
  launch to `pl.submit`/`pl.spmd_submit` while preserving outputs and TaskId
  semantics.
- Do not add task metadata to an inner `pl.spmd` nested in `pl.cluster()`; the
  parser rejects it. Flag the outer Group submission, which may require a
  deliberate refactor.
- `pl.system.task_dummy` cannot currently express this flag in model source.
  Stop and explain instead of inventing a keyword or broadening the task into a
  compiler/runtime feature change.
- A predicated consumer never early-dispatches. Do not edit its producers and
  promise success.

Make only the producer annotations required by the selected target. Run the
relevant formatting, compile, and correctness tests.

Verify the rebuilt codegen before profiling:

```bash
AFTER_BUILD=build_output/<new-build>
rg -n 'set_allow_early_resolve\(true\)' "$AFTER_BUILD" \
  -g '**/orchestration/*.cpp'

while IFS= read -r deps_file; do
  jq -r '.tasks[] |
    [.task_id, .early_dispatch, (.kernel_ids | join(","))] | @tsv' \
    "$deps_file"
done < <(find "$AFTER_BUILD" -type f -name deps.json \
  -path '*/dfx_outputs/*' -print)
```

Direct builds place these under `<build>/orchestration/` and
`<build>/dfx_outputs/`; multi-program builds can nest them under
`next_levels/<program>/orchestration/` and `dfx_outputs/rankN/dK/`. Restrict the
inspection to the exact dispatch selected in the baseline report.

Every selected non-allocation direct producer must now have
`early_dispatch=true`.

## 4. Capture and compare after the edit

Run the exact same level-4 command, configuration, and topology. Use a new
timestamp marker and reject stale artifacts. Then compare against the saved
baseline:

```bash
python .claude/skills/early-dispatch/scripts/report.py \
  <after-build-dir> \
  --target <operator-name> \
  --operator <enclosing-dispatch-program> \
  --baseline-json <baseline-build-dir>/early_dispatch_baseline.json \
  --json-out <after-build-dir>/early_dispatch_after.json \
  -o <after-build-dir>/early_dispatch_comparison.md
```

Use the same `--occurrence` selection as the baseline. With `--baseline-json`,
the helper automatically pins the baseline physical rank and rejects an
explicit conflicting `--rank`. It matches repeated names by dispatch and
occurrence, then verifies the target and direct-predecessor identity, logical
block, kernel-slot, and physical-row signatures before comparing. Do not assume
runtime task IDs remain stable across builds.

## 5. Diagnose a target that was not early-dispatched

Always give the user the exact after-artifact directory. Diagnose in this order:

1. **Producer policy:** list every non-allocation direct producer still missing
   `early_dispatch=true`. This is a proven blocker.
2. **Unsupported target:** inspect source for a dispatch predicate or a resource
   shape other than AIC/AIV/MIX. The runtime rejects those from early staging.
3. **Late target submission:** use the target's L4
   `aicpu_orchestrator_phases` envelope. If its submit starts after the latest
   producer FIN, the early window already closed and normal dispatch is
   expected.
4. **Producer publication:** every producer's full logical block set must be
   launch-visible before it propagates eligibility. A first block alone is not
   enough.
5. **Scheduler priority/capacity:** normal ready queues strictly outrank early
   queues; PMU activity disables early staging; an AIC/AIV task needs a free
   running or pending descriptor slot. For a named resource blocker, prove both
   slots on every compatible core were occupied throughout the opportunity
   window.
6. **Launch shape:** a `sync_start` target needs global all-or-nothing capacity.
   SPMD/MIX tasks can be only partially staged; report `N/M`.

The level-4 scheduler `early_dispatch` phase contains only an aggregate
`tasks_processed` count, not task IDs. Prove a specific target using its
dispatch timestamp plus dependency FIN timestamps. Treat queue-depth, PMU, and
core-occupancy correlations as candidates unless they establish the required
policy or complete capacity condition. Never invent a named blocker.

For Perfetto scheduler inspection, generate an overhead view:

```bash
python -m simpler_setup.tools.swimlane_converter \
  <records> --deps-json <deps.json> --overhead \
  -o <dispatch-dir>/merged_swimlane_overhead.json
```

## 6. Report the decision

Return:

1. Source files and producer launch sites changed.
2. Baseline and after artifact directories.
3. Actual early-dispatch status (`full`, `partial`, or `none`) with task IDs.
4. A bordered before/after table with one row per state. Show the two raw
   timestamps used to calculate the gap, not only the derived gap. For multiple
   direct producers, `latest direct-predecessor end` is their maximum end time;
   `earliest target start` is the minimum start across the target successor's
   physical rows:

   ```text
   task | state | latest direct-predecessor end µs | earliest target start µs | signed start gap µs | gap saved vs before µs
   ```

5. If actual early dispatch failed, list the proven blocker or the narrowest
   supported scheduler/launch diagnosis and link the artifact.
6. If actual early dispatch succeeded:
   - when every selected after-gap is **strictly below 1 µs**, state that the
     modification is effective under the requested target-gap criterion and
     should be kept after required correctness tests pass;
   - otherwise, show the source diff and ask whether the user wants to retain
     the change. Do not revert it before the user answers.

Report any separate end-to-end or producer-fan-out regression rather than
hiding it behind the target-gap decision.

Level-4 collection has observer cost. Use it for causal scheduling evidence,
not as the final production-latency headline.
