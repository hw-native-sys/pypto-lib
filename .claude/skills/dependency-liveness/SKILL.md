---
name: dependency-liveness
description: Diagnose and repair missing or unsafe PyPTO task dependencies by proving generated producer-to-consumer reachability across dynamic branches, restoring the narrowest explicit TaskId edge, or splitting a core-holding distributed wait into publish/defer-wait/finalize tasks. Use for intermittent races, missing tasks, non-uniform-rank deadlocks, stuck notify blocks, and TENSOR_WAIT_TIMEOUT investigations.
---

# Dependency Liveness

Diagnose a PyPTO dependency or scheduler-liveness failure from generated graph
and runtime evidence, then apply the narrowest source repair that matches the
proven failure class.

Read these canonical guides before editing a kernel:

- [Dependencies and Scheduling](../../../docs/debug-and-tune/dependency-and-scheduling.md)
- [L3 Programming](../../../docs/pypto-coding/l3-programming.md), especially
  `defer_wait`, wait anchoring, and payload visibility
- [Debugging](../../../docs/debug-and-tune/debugging.md), especially dependency
  graph generation and runtime-hang evidence

Do not treat the two repair modes as interchangeable:

- `deps=` restores semantic ordering that the generated graph does not contain.
- `defer_wait` prevents an already-ordered distributed wait from occupying a
  core needed for progress.

## 1. Preserve the failing topology

Reproduce the real branch, loop count, rank occupancy, and communication epoch.
Do not replace a non-uniform distributed failure with a balanced or single-card
case and claim the dependency is fixed.

Capture a fresh dependency graph with the target entry's normal command and
`--enable-dep-gen`. When runtime dispatch behavior matters, also capture chip
swimlane perf level 4 in a separate clean timing pass. Current onboard PyPTO
normally performs the graph and timing passes separately; never deliberately
combine extra dependency instrumentation with the timing pass.

For every selected rank and dispatch require:

```text
deps.json
name_map*.json
dispatch_program.json       # required when more than one program is present
chip_swimlane_records.json  # required for a scheduler-liveness conclusion
```

Reject stale or incomplete artifacts. Device-resident control values can make
the graph-only pass choose a different branch from the timing pass; compare the
task names, occurrence counts, block counts, and dynamic routing before joining
their conclusions.

## 2. Prove task existence and reachability

Run the bundled graph checker on the build root or one dispatch directory:

```bash
python .claude/skills/dependency-liveness/scripts/report.py \
  <build-output> \
  --producer <producer-name> \
  --consumer <consumer-name> \
  --pair-offset <consumer-occurrence-offset> \
  --operator <dispatch-program> \
  --json-out <build-output>/dependency_liveness.json \
  -o <build-output>/dependency_liveness.md
```

`--pair-offset 1` checks producer occurrence `i` against consumer occurrence
`i + 1`, which is useful for adjacent layer or epoch handoffs. Use explicit
`--producer-occurrence` and `--consumer-occurrence` for one exact pair. Runtime
task IDs are capture-local; use them only for forensic selection in one build.

The report separates four outcomes:

1. **Producer absent.** The expected task was predicated away, never submitted,
   or removed because its result was not semantically consumed.
2. **Consumer absent.** The downstream branch did not materialize; an edge edit
   cannot repair a task that does not exist.
3. **Both tasks exist but no path exists.** The generated graph permits the
   consumer to run before the producer.
4. **A direct or transitive path exists.** Do not add another edge merely
   because the runtime later stalled; investigate scheduling and resources.

Compare every dynamic variant that can remove intermediate work: active and
empty owners, full and ragged batches, zero-token paths, first and later
epochs, and both sides of a conditional branch. A path present only through an
optional attention or compute task is not a valid cross-stage control contract.

## 3. Classify before editing

### Missing task

If a named task is absent from `deps.json`, first verify that the expected
source launch is part of the compiled program. Common causes are:

- a runtime predicate or zero-trip loop;
- an inline call whose returned tensor or completion is not consumed;
- dead-code elimination after a side-effect was expressed only through an
  unused value;
- inspecting the wrong inlined occurrence or dispatch program.

Restore the semantic output or completion use. Do not create a dummy edge to a
task that code generation removed, and do not force an optional compute branch
to run solely to recover an accidental dependency.

### Missing graph edge

If both tasks exist but the path is absent, determine which mechanism should
have expressed it:

- Fix an incorrect `In` / `Out` / `InOut` declaration or wrong tensor slice
  when TensorMap should infer a true RAW or WAW dependency.
- Add an explicit `deps=` edge for WAR ordering, communication side effects,
  control-only completion, optional-branch handoff, or a dependency carried by
  a `manual_dep=True` tensor.
- Preserve the producer TaskId with the `as tid` form. A `for ... in pl.spmd`
  form cannot carry `deps=`; convert only the required launch to the captured
  context-manager form.
- Pass a completion across an inline boundary as `pl.Scalar[pl.TASK_ID]` when
  the consumer is outside the producer's lexical scope.
- Use `pl.system.task_dummy(deps=[...])` only as a no-work fan-in when several
  producers need one completion handle.

Prefer one semantic producer-to-consumer edge over a full-stage barrier. Never
add a chain of dummy tasks to make a failing schedule happen to serialize.
After editing, rerun the report and require the repaired path on every relevant
rank and branch.

### Ordered graph with a runtime stall

When the required path exists, use level-4 scheduler records and device logs to
identify the stalled task, physical blocks, running/pending slots, and free
cores. A resource deadlock is supported when evidence shows all of the
following:

- a later distributed wait is running while holding a compatible core;
- a preceding publish or notify task has an unretired block;
- peer progress requires that missing block;
- the missing block was never executed, rather than executed and lost;
- heap, task-window, dependency-pool, and TensorMap exhaustion are absent.

Do not infer this from `TENSOR_WAIT_TIMEOUT` alone. A timeout is the terminal
symptom of many graph, communication, and capacity failures.

## 4. Split a core-holding wait

Use `pld.system.defer_wait()` only after the ordered-graph resource cycle is
proved. Split the original combined task into three responsibilities:

```text
publish
  - compute and publish local metadata or payload
  - issue the matching arrival notify

wait
  - register pld.system.defer_wait() only
  - perform no calculation that must resume after the condition

finalize
  - deps=[publish_tid, wait_tid]
  - consume remote metadata and publish the downstream output/completion
```

`defer_wait` leaves its TaskId incomplete and releases the core. It never
resumes the kernel body, so every operation logically after the condition must
move into `finalize`. Keep the wait in a dedicated registration-only task; do
not mix it with route counting, stores, cumsum, output assembly, or another
notify.

This repair changes resource liveness, not data visibility. A peer observing a
notify does not automatically prove that a preceding non-draining
`remote_store` payload is visible. Preserve the L3 guide's `put` / notify
ordering contract and validate payload contents separately.

Do not add full cross-layer serialization when a wait-only split breaks the
resource cycle. A later epoch may register its deferred wait early as long as
it cannot occupy the core required by the preceding notify.

## 5. Validate the repair

Use the same topology, inputs, and control values before and after. Require:

1. compile, runtime, and numerical/cache validation pass;
2. the expected tasks exist in every selected graph;
3. the required producer-to-consumer paths are present in active and empty or
   ragged variants;
4. no unrelated dependency edge or whole-stage barrier was introduced;
5. every deferred wait has a dependent finalize task and no post-wait work in
   its own body;
6. notify/publish blocks fully retire and no core-holding wait remains;
7. repeated runs cover the race, not just one successful schedule;
8. heap, task-window, dependency-pool, and TensorMap statistics remain within
   capacity.

For multi-epoch communication, run at least two epochs and verify that signals
are monotonic and cannot be satisfied by an earlier epoch. For a failure caused
by non-uniform occupancy, retain both balanced and non-uniform gates: balanced
proves the base graph still works, while non-uniform exercises the repaired
branch.

## 6. Report

Report:

- exact source task names and occurrences;
- build and dispatch artifact paths;
- whether each expected task existed before and after;
- direct or shortest transitive path, including edge sources;
- the dynamic rank/branch comparison;
- the proven failure class;
- the minimal source change;
- correctness, repeated-runtime, and capacity results.

State explicitly whether the missing notify was never executed or was executed
without the expected remote effect. Do not attribute a lib graph defect to the
compiler, runtime scheduler, or ISA merely because a lowering change altered
the timing that exposed it.
