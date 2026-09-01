---
name: optimize
description: Run one measured performance-tuning loop on a PyPTO kernel or model — read the builtin tuning rules and the local lessons log to rank candidates, freeze one golden and one baseline, try candidates one at a time on device, decide each with the wall-time rule, and record the lesson learned. Use for `/optimize target`, speeding up an operator or model step, choosing between tuning levers, or resuming tuning where a previous session stopped.
---

# Optimize

Orchestrate one tuning loop. The technical method lives in
`docs/debug-and-tune/`; the mechanics of each individual lever live in the
focused skills. This skill decides **what to try, in what order, and what was
learned** — it does not re-document tiling or dispatch.

Keep these invariants:

- **Wall time decides.** Follow [`benchmarking.md`](../../rules/benchmarking.md)
  exactly, including its four-row verdict table. Never report a busy-time or
  utilization improvement as a speedup.
- **One frozen golden, one baseline, one compile per variant.** Iteration cost
  is dominated by work that is not the thing being measured.
- **One independent variable at a time.** Two changes measured together teach
  nothing about either.
- **A lesson is a prior, not a measurement.** The log reorders candidates; it
  never supplies a number.
- **Revert what does not win**, and log it anyway.
- Preserve numerical behaviour. A speedup that changes results is a different
  task — validate every kept variant.

## 1. Resolve the target and the case

1. Treat the argument as a Python path, an operator, or a model name. Search
   `models/` and `examples/` for an exact filename, class, program, or
   `name_hint` match. Ask only if more than one plausible target remains.
2. Use the `run-model-cases` skill to pick the case that actually represents the
   workload — its real platform, shapes, weights, and device list. A smaller or
   synthetic case answers a different performance question.
3. Read the target's `argparse` before choosing flags. `--save-data`,
   `--golden-data`, and the swimlane flags are entry-specific, not universal.
4. Confirm a real device is available. A `*sim` platform can confirm compile and
   correctness; it cannot rank two variants, so it cannot run this loop.

## 2. Read the rules, then the lessons

Follow [`optimization-lessons.md`](../../rules/optimization-lessons.md) Step 1.

1. Read the builtin rule for the levers in play: `performance-tuning.md` Part 1
   (L2 / inter-kernel) and Part 2 (L1 / L0 intra-kernel), plus
   `dependency-and-scheduling.md`, `cube-tile-tuning.md`, or
   `ring-heap-and-scope-stats.md` as the symptom demands, and the matching case
   study for a model in the same family.
2. Read `OPTIMIZATION_LESSONS.md` at the main repo root (`git worktree list`,
   first entry). Filter to entries whose `Target`, `Scope`, or `Lever` touches
   this model, operator class, or platform. Age does not weaken an entry — only
   a contradicting measurement does. When an entry's `Scope` names a pin chain
   that has since moved, plan to re-measure it rather than skipping it.
3. Emit a **ranked candidate list before touching code**. For each candidate
   give the lever, the expected mechanism, and the rule or lesson behind it.
   List separately the candidates you are *skipping*, each with the entry that
   ruled it out. Show this list to the user before spending device time.

## 3. Establish the baseline once

1. Capture the evidence that says where the time goes before changing anything:
   a level-4 chip swimlane, and PMU counters when the suspicion is intra-kernel.
   Use `critical-path` for an operator whose latency is the question.
2. Freeze the golden with `--save-data` via the `test-with-golden` skill. Record
   the snapshot path.
3. Measure the baseline once with `PYPTO_BENCH=1` and the rounds you will use
   for every later comparison. Keep its build directory and data until the whole
   comparison is reported.
4. State the metric convention up front: the `[RUN] effective_us … mean=`
   headline for a single card, the lowest per-rank mean for multi-card tuning.
   Never mix the two inside one comparison.

## 4. Try one candidate

For each candidate, in ranked order:

1. State the mechanism first, per **Decide the bound class before choosing a
   fix** in `performance-tuning.md`: which task the trace says is the
   bottleneck, what its own numbers say limits it, and how this edit removes
   that. A candidate you cannot state that for is not ready to compile.
2. Delegate the mechanics to the focused skill that owns the lever —
   `cube-tile-tuning`, `early-dispatch`, `add-dummy`, `incore-profiling`. Do not
   reimplement their procedures here.
3. Make the smallest edit that tests the hypothesis.
4. Validate correctness against the frozen golden.
5. Measure with the baseline's exact platform, device list, rounds, and warmup.
6. Batch a parameter sweep into one process rather than one process per value;
   log the sweep as a single attempt with its conclusion.
7. Cut rounds while iterating, and restore the baseline's round count for the
   number you report.

## 5. Decide, then log

1. Apply the verdict table from `benchmarking.md`: wall down → keep; wall flat
   with busy down → `inconclusive`, keep only if it unblocks a named follow-up,
   and say so; wall up → revert.
2. Revert immediately on a loss. Do not carry a losing change into the next
   candidate.
3. **Write the lesson entry now**, while the numbers are in front of you. Give
   its `Repro` as a command that works from a clean tree — the session's own
   `build_output/` paths do not go in the log —
   format, gates, and sort order in
   [`optimization-lessons.md`](../../rules/optimization-lessons.md). Log
   `reverted` and `inconclusive` outcomes with the same care as wins; they are
   what stop the next session from repeating the attempt.
4. Skip the entry only when a gate rejects it, and say which gate.
5. Log a correctness or precision fix nowhere. Log a suspected compiler defect
   in the other file, per
   [`problem-handling.md`](../../rules/problem-handling.md).

**Report each decided attempt to the user before moving on** — the perf gain,
why it won or did not (from the trace, not the hypothesis), and the new trace
files. Format and the two cases that deserve extra words are in
[`benchmarking.md`](../../rules/benchmarking.md#every-finished-optimization-reports-three-things).
Report losses with the same care as wins.

Then return to step 4 with the next candidate, re-ranking if what you learned
changes the order.

## 6. Close out

1. Re-read `OPTIMIZATION_LESSONS.md` end to end. Re-measure entries this task
   may have disproved, remove only those a measurement now contradicts, and
   confirm the file is still sorted.
2. Remove diagnostic instrumentation and revalidate.
3. Report: the baseline, each candidate with its verdict and numbers, the
   composite result, and every convention behind the numbers — platform and
   device, rounds and warmup, metric convention, whether the golden was
   replayed.
4. Summarize the lessons written this session, and hint that
   `/promote-lessons` can fold them into the checked-in guides.

## Scope

- Tune only the target the user named.
- Do not commit, push, or open a PR unless asked.
- Do not edit the checked-in guides in `docs/` from this skill. Lessons land in
  the local log; `/promote-lessons` moves them, with the user's review.
- Keep traces, builds, and frozen goldens under `build_output/`; never stage
  them.
