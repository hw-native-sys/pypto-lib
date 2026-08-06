---
name: add-dummy
description: Protect an Observed-critical-path PyPTO operator by stopping a named early-dispatched noncritical sibling from consuming the same speculative dispatch opportunity. Add an unflagged pl.system.task_dummy(deps=[]) only to that sibling's direct deps, then use before/after level-4 L2 swimlanes to prove the protected sibling was early-dispatched and compare shared-producer-end to protected-operator-end latency. Use for `/add-dummy operator-name`, sibling early-dispatch contention, or dependency-only dummy scheduling experiments.
---

# Add Dummy

Treat the argument as **`b`**, the early-dispatched sibling to suppress. Find
the protected critical sibling **`a`** and their shared direct producer **`c`**
from the canonical **Observed path**:

```text
c ──> a     Observed critical-path data edge; preserve early dispatch
└──> b      noncritical sibling; add an unflagged dummy predecessor
```

Keep these invariants:

- Add the dummy to `b`, never to `a` or `c`.
- A sibling shares at least one direct producer; it need not have an identical
  full predecessor set.
- Preserve `b`'s own `allow_early_resolve` metadata exactly. Do not add it when
  absent or change its value; that flag controls `b`'s consumers, not whether
  `b` itself is early-dispatched.
- Use AICore end timestamps for the requested `c end → a end` metric. Use
  AICPU dispatch/FIN timestamps separately to prove actual early dispatch.
- Do not use Static CPM to choose `c → a`. Use the Observed path produced by
  `simpler_setup.tools.critical_path`.

## 1. Capture a fresh level-4 baseline

Resolve the normal executable and source site for the exact operator argument.
Preserve real shapes, weights, topology, device list, and arguments. Activate
the worktree environment:

```bash
source temp/set_env.sh
source "$PYPTO_ROOT/toolchain/versions.env"
SKILL_PTOAS_VERSION=${PTOAS_VERSION#v}
```

Inspect the executable's `argparse` definition before adding the swimlane flag:

- `action="store_true"`: pass bare `--enable-l2-swimlane`; runtime `True` maps
  to level 4.
- An integer/optional-value argument accepting `4`: pass
  `--enable-l2-swimlane 4`.
- If the CLI excludes level 4, stop. Do not substitute level 1 or 2.

Submit the real NPU run through the device queue; do not infer device health
from an unassigned shell:

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

Use the operator's bare swimlane flag when required and adapt its device option.
For multi-card runs, request the normal count with `--device-num N` and preserve
the operator's device-list convention. Keep `$TASK_DEVICE` inside the
single-quoted submitted command. Read the repository's declared PTOAS version,
let `task-submit` select that installation, preserve its injected path across
`temp/set_env.sh`, and then normalize the archive layout as shown. Require a
regular executable file: `test -x` alone also succeeds for a searchable
directory. Prefer a working top-level wrapper and otherwise use the `bin/`
wrapper; do not assume either layout blindly.
Do not describe a toolchain preflight or compile failure as an NPU
initialization failure.

Create a timestamp marker immediately before running and accept only artifacts
newer than it. Current onboard PyPTO automatically performs separate graph and
timing passes; analyze the timing pass. Require each selected dispatch to have:

```text
l2_swimlane_records.json
deps.json
name_map*.json
dispatch_program.json       # mandatory when multiple programs/dispatches exist
```

Require `l2_swimlane_level == 4`, a complete AICore/AICPU clock join, and the
same value-routed task topology in the deps and timing passes. Device-resident
control tensors are zero-filled in the graph-only child; stop if that can alter
task routing and no equivalent capture with real control values exists.

Generate the canonical path artifacts and inspect its **Observed path**:

```bash
python -m simpler_setup.tools.critical_path <baseline-build-dir> --stdout
```

Then save the deterministic baseline selection:

```bash
python .claude/skills/add-dummy/scripts/report.py \
  <baseline-build-dir> \
  --suppressed <operator-b> \
  --operator <enclosing-dispatch-program> \
  --json-out <baseline-build-dir>/add_dummy_baseline.json \
  -o <baseline-build-dir>/add_dummy_baseline.md
```

Omit `--operator` only for an unambiguous single-program capture. The helper
accepts a MIX source name when its compiled names are exactly that name plus
`_aic`/`_aiv`. It defaults to the rank with the smallest summed
dispatch-to-finish time. Pin that selected rank for the after capture.

Use `--suppressed-occurrence N` when `b` repeats. If protected candidates are
ambiguous, add exact `--protected <a>` / `--producer <c>` and their occurrence
selectors. Runtime task IDs are capture-local; do not reuse them after rebuild.

## 2. Enforce baseline preconditions

Proceed only when the helper proves all of these:

1. `b` is actually early-dispatched (`full N/N physical rows` or explicitly
   reported `partial N/M physical rows`). Structural producer flags alone are
   not proof.
2. `b` is not on the Observed path.
3. An adjacent Observed-path `data-wait` edge `c → a` is a real dependency
   edge, and `c` is also a direct predecessor of `b`.
4. `a` is structurally early-dispatch eligible. Adding a dummy to `b` cannot
   repair an unflagged producer of `a`.
5. `b` does not already have a dummy predecessor.

If any condition fails, stop without source modification. If multiple source
sites or `(c, a, b)` occurrences remain, list them and require an exact choice.

For each selected logical task, fold every physical block:

```text
E(c) = max(c block end_time_us)
E(a) = max(a block end_time_us)
baseline interval = E(a) - E(c)
```

Prove a task `x` was actually early-dispatched only when every direct producer
is an allocation or has `deps.json::early_dispatch=true`, at least one producer
is non-allocation, and a block satisfies:

```text
x.dispatch_time_us + 2 clock ticks < max(direct producer finish_time_us)
```

`deps.json::tasks[x].early_dispatch` describes `x` as a producer for its own
consumers. It does not prove `x` was dispatched early.

## 3. Map `b` to its exact source launch

Use:

```text
dispatch_program.json
  → next_levels/<program>/kernel_config.py
  → name_map*.json callable IDs
  → deps.json task kernel_ids
  → exact source name_hint/submitted callee
```

Inspect every active MIX kernel slot and the complete source expansion. A
single source site can generate several runtime occurrences. Report that
fan-out before editing, because one shared dummy with many consumers adds
scheduler work to all of them.

Also enumerate every other direct consumer of `c`. This invocation suppresses
only the named `b`; if another consumer is actually early-dispatched, do not
claim that `a` is the only early-dispatched sibling.

## 4. Add exactly one empty, unflagged dummy edge

Create the dummy in the same live orchestration scope immediately before `b`:

```python
seed_dummy = pl.system.task_dummy(deps=[])
```

Append `seed_dummy` to `b`'s existing dependencies without removing or
reordering any real dependency. Preserve `b`'s original
`allow_early_resolve` keyword and value exactly; the examples omit that
unrelated metadata:

```python
with pl.at(
    level=pl.Level.CORE_GROUP,
    name_hint="operator_b",
    deps=[existing_tid, seed_dummy],
):
    ...

out, b_tid = pl.submit(
    self.operator_b,
    value,
    deps=[existing_tid, seed_dummy],
)
```

For an existing `with pl.spmd(...) as tid` launch, append the same dependency.
A plain `with pl.spmd(...)` without `as tid` cannot carry `deps`; add a TaskId
capture even when it is otherwise unused. A `for i in pl.spmd(...)` launch also
cannot carry `deps`; convert only that launch to the captured context-manager
form while preserving its block-index semantics:

```python
seed_dummy = pl.system.task_dummy(deps=[])
with pl.spmd(
    blocks,
    name_hint="operator_b",
    deps=[seed_dummy],
) as _b_tid:
    i = pl.tile.get_block_idx()
    ...
```

Apply these restrictions:

- Keep the dummy's `deps` exactly empty. `task_dummy(deps=[c_tid])` adds a real
  post-`c` hop and tests a different optimization.
- `task_dummy` accepts only the required `deps=` keyword; it has no
  `allow_early_resolve` parameter.
- Do not reference a TaskId from an already-closed inner scope. Codegen rejects
  that edge.
- Plain kernel calls cannot carry `deps`; convert the exact launch to
  `pl.submit`/`pl.spmd_submit` only after preserving outputs and TaskId
  semantics.
- Inner launches inside `pl.cluster()` cannot carry task-level deps because
  they become one Group dispatch. Stop and explain rather than attaching the
  dummy at the wrong level.
- Modify only `b`. Do not remove early flags or alter `a`, `c`, the scheduler,
  or TensorMap dependencies.

Run formatting, compilation, and the relevant correctness tests. Inspect
generated orchestration for `rt_submit_dummy_task`, then verify in rebuilt
`deps.json` that:

- the dummy has all-negative `kernel_ids`, `early_dispatch=false`, and no args;
- exactly one explicit `dummy → b` edge was added;
- every original predecessor of `b` remains;
- `c → a` and all of `a`'s predecessor signatures are unchanged.

## 5. Capture and compare after the edit

Run the identical level-4 command into a new artifact directory. Reuse the
baseline rank and compare by program, compiled name set, occurrence, and block
shape—not task ID:

```bash
python .claude/skills/add-dummy/scripts/report.py \
  <after-build-dir> \
  --suppressed <operator-b> \
  --baseline-json <baseline-build-dir>/add_dummy_baseline.json \
  --json-out <after-build-dir>/add_dummy_after.json \
  -o <after-build-dir>/add_dummy_comparison.md
```

The helper requires the new direct dummy to also appear in the L4 scheduler's
task-ID-bearing `dummy_task` record. It verifies that `b` is no longer actually
early-dispatched and reports whether `a` is `full`, `partial`, or `none`.
Ordinary scheduler `early_dispatch` phases contain only aggregate counts; never
use them as target-specific proof.

## 6. Report the result and decision

Return:

1. The changed source site and exact dependency diff.
2. The baseline and after artifact directories.
3. `a`, `b`, and `c` compiled names, occurrences, and capture-local task IDs.
4. `a` and `b` actual early-dispatch status before and after.
5. This table:

   ```text
   c → a | before c-end→a-end µs | after c-end→a-end µs | improvement µs
   ```

Decide as follows:

- If `b` remains early-dispatched, the dummy was not wired as intended. Do not
  claim success.
- If `a` was not early-dispatched after the edit, give the after-artifact path
  and diagnose `a` before recommending retention.
- If every after interval is **strictly smaller**, state that the L4-observed
  change has benefit and should be kept after correctness tests and repeated
  DFX-off performance validation pass.
- Otherwise, state that no benefit was measured and ask whether to restore the
  source change. Do not revert before the user answers.

L4 collection has observer cost and normal run-to-run noise. Label the result
as L4-observed; do not replace the production benchmark headline with one
instrumented sample.
