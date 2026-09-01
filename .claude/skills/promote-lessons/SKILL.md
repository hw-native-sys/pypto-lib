---
name: promote-lessons
description: Fold measured entries from the local OPTIMIZATION_LESSONS.md log into the checked-in tuning guides and skills — cluster related entries, apply the promotion bar, route each lesson to the guide or skill that owns it, rewrite it in that document's voice, and delete the promoted entry from the log. Use for `/promote-lessons`, "summarize what you learned into the rules", or consolidating measured tuning knowledge into the checked-in guides.
---

# Promote Lessons

Move what has been learned locally into the expert-verified layer. This is the
only path from `OPTIMIZATION_LESSONS.md` into `docs/`, and it runs **only when
the user asks**.

Keep these invariants:

- **Lossy by design.** Most entries never graduate. A log of 20 entries that
  promotes 3 is working correctly.
- **Never invent.** Every promoted claim carries the measurement that produced
  it, restated in the destination's voice. If the evidence is one run on one
  shape, the promoted text says so or the entry does not go.
- **Never name the local log in checked-in text.** Readers cannot see it.
  Describe the measured result, not the entry.
- **The user reviews before anything lands.** Show the diff; do not commit.

## 1. Read and cluster

1. Read `OPTIMIZATION_LESSONS.md` at the main repo root (`git worktree list`,
   first entry). If it is absent or empty, say so and stop.
2. Group entries that describe the same underlying mechanism, even when they
   name different kernels. Three single-kernel entries that all say "the cache
   line beat the tile-size heuristic" are one cross-model lesson, and that
   cluster is worth more than any of its members alone.
3. For each cluster, state the mechanism in one sentence and list the measured
   evidence behind it.

## 2. Apply the promotion bar

A cluster is promotable when **all** of these hold:

| Test | Requirement |
| ---- | ----------- |
| Reach | `Generality` is `model-family` or `cross-model`, or the cluster raises a set of single-kernel entries to that reach |
| Evidence | `Confidence` is `repeated` or `cross-model` — a single measurement on one shape is a data point, not a rule |
| Currency | The code it describes still exists, and no later measurement contradicts it. Age alone never disqualifies an entry — but if the pin chain moved since it was measured, re-measure before publishing it into `docs/` |
| Novelty | Not already stated in `docs/` or a skill. Search first |
| Non-contradiction | It does not contradict a checked-in guide. If it does, stop and raise it with the user — one of the two is wrong, and resolving that is the real task |

Report the clusters that fail, with the test that failed. They stay in the log;
a failed bar usually means "needs one more measurement", not "discard".

## 3. Route each promoted lesson

| Kind of lesson | Destination |
| -------------- | ----------- |
| A general lever — how to structure kernels, tiles, or the schedule | `docs/debug-and-tune/performance-tuning.md`, as a numbered tuning rule in Part 1 (L2) or Part 2 (L1/L0) |
| Matmul row / N / K tile sizing | `docs/debug-and-tune/cube-tile-tuning.md` |
| Task edges, dispatch, or scheduler behaviour | `docs/debug-and-tune/dependency-and-scheduling.md` |
| What worked on one model, in the order it happened, with the limit hit | the model's case study — `deepseek-v4-decode-optimization.md`, `qwen3-14b-optimization.md` — including negative results, as in its "did not hold up" sections |
| How to measure or report a number | `.claude/rules/benchmarking.md` |
| A procedural invariant for one lever | that lever's `SKILL.md` invariants — `cube-tile-tuning`, `early-dispatch`, `add-dummy`, `critical-path` |
| A new model with no case study yet | propose a new `docs/debug-and-tune/<model>-optimization.md` and confirm with the user before creating it |

Add an index row in `docs/debug-and-tune/index.md` for any new page.

## 4. Write it in the destination's voice

- Match the surrounding section's structure: the case studies lead with the
  problem and the limit found; `performance-tuning.md` leads with a numbered
  rule and a before/after code sketch.
- Keep numbers with their conditions — platform, shape, rounds. A bare "9%
  faster" is not reusable.
- English only, no private information: no usernames, no absolute paths under a
  home directory, no machine names.
- Comments and prose say what, not why-it-was-hard.

## 5. Delete what was promoted

1. Remove each promoted entry from `OPTIMIZATION_LESSONS.md` — **not** a status
   change. An entry that lives in both places drifts, and the checked-in copy is
   the one readers trust.
2. Re-sort the remaining entries.
3. From a worktree, use the locked read-modify-write recipe in
   [`problem-handling.md`](../../rules/problem-handling.md#writing-from-a-worktree),
   with an entry heading as the anchor.

## 6. Report

Show the user:

- Which clusters were promoted, and to which file.
- The full diff of every checked-in file touched.
- Which clusters were held back, and the bar test each one failed.
- What remains in the local log.

Do not commit or open a PR unless asked. When the user does ask, use the
`git-commit` and `github-pr` skills, and describe the measured findings — never
the local log — in the message.
