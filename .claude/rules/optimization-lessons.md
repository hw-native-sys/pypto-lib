# Optimization Lessons

## Core Principle

**Read the accumulated lessons before tuning. Write one back after every
decided attempt.**

Two layers, and they are not interchangeable:

| Layer | Where | Verified by | Standing |
| ----- | ----- | ----------- | -------- |
| Builtin rules | `docs/debug-and-tune/`, [`benchmarking.md`](benchmarking.md), the tuning skills | expert review, checked in | authoritative |
| Local lessons | `OPTIMIZATION_LESSONS.md` at the main repo root | this developer's own measurements | priors, not facts |

**This file records how to make kernels faster — nothing else.** A lesson here
is about performance: what moved wall time, what did not, and the mechanism
that explains it. Lessons about how to write a *correct* kernel do not belong
here, however hard-won:

| What you learned | Where it goes |
| ---------------- | ------------- |
| A tuning lever moved (or failed to move) wall time | here |
| A diagnostic method that tells you where the time goes | here |
| How to express something correctly in the DSL | `docs/pypto-coding/pypto-coding-style.md` |
| A numerical / rounding / dtype fix | `docs/debug-and-tune/precision-tuning.md` |
| A toolchain defect — DSL is right, compiler is wrong | `KNOWN_PYPTO_ISSUES.md`, per [`problem-handling.md`](problem-handling.md) |
| A kernel bug you just fixed | the fix is the record; log nothing |

### Two kinds of entry

| Kind | What it is | What proves it |
| ---- | ---------- | -------------- |
| `change` | A code edit that did or did not move wall time | a device baseline → variant comparison |
| `method` | A way of finding out where the time goes, or of running the loop itself | the capture or tool output it was read from, and what it revealed that the alternative did not |

A `method` entry has no baseline and no variant — there is nothing to compare,
because nothing was edited. It is still a lesson, and often the more valuable
one: a `change` entry tells you what worked on one kernel, a `method` entry
tells you how to find that out on the next one. The gates below apply to each
kind differently; everything else in this file is the same for both.

The split matters because the two kinds age differently and are read at
different moments. A correctness rule is consulted while writing a kernel and
is either right or wrong; a performance lesson is consulted while ranking
candidates and is only ever a prior. Mixing them makes the log unreadable at
exactly the moment it is supposed to save time.

A lesson **reorders which candidate you try first**. It never replaces a
measurement, never justifies a claimed speedup on its own, and never overrides a
builtin rule. When a lesson contradicts `docs/`, the doc wins — and the
contradiction goes in the entry, because one of the two is wrong and that is
worth knowing.

## Step 1 — Read Before You Tune

This fires on **any** performance change — including an ad-hoc one nobody
called "tuning", and including a question about why something is slow. Waiting
to be handed a formal tuning task is how the read gets skipped.

1. Read the builtin rule for the lever in question — `performance-tuning.md`
   Part 1 (L2) or Part 2 (L1/L0), `cube-tile-tuning.md`,
   `dependency-and-scheduling.md`, `ring-heap-and-scope-stats.md`, and the
   matching case study (`deepseek-v4-decode-optimization.md`,
   `qwen3-14b-optimization.md`).
2. Read [`benchmarking.md`](benchmarking.md). It defines what counts as a win;
   nothing below is meaningful without it.
3. Read `OPTIMIZATION_LESSONS.md` at the main repo root
   (`git worktree list --porcelain | head -1`), filtered to entries whose
   `Target`, `Scope`, or `Lever` matches this model, operator class, or
   platform.
4. Produce a ranked candidate list **before touching code**, with the log's
   `reverted` and `inconclusive` entries already excluded — and say out loud
   which past entry ruled each candidate in or out.

Skipping step 3 is how a session spends a day re-deriving a dead end that is
already written down.

## When to Log

```text
Optimization attempt reaches a decision (kept or reverted)
├─ Was it a correctness, precision, or compile fix?
│  └─ YES → do not log. This file is about speed only.
├─ Is it a `change` (a code edit) or a `method` (how to find the bottleneck)?
├─ `change`, and it was not measured on a device under benchmarking.md
│  conventions?
│  └─ do not log. An unmeasured hunch is not a lesson.
├─ `method` with no capture or tool output behind it?
│  └─ do not log. An unevidenced method is an opinion.
└─ YES → does it pass the three gates below?
   ├─ YES → log it — including, especially, when the verdict is `reverted`
   └─ NO  → do not log; fix what the gate caught first
```

**Log the failures.** A `reverted` entry is the highest-value kind: it is the
one thing a fresh session would otherwise spend a day re-discovering. The
canonical guides already do this — see `3.5 A scheduling change that did not
hold up` in the DeepSeek V4 case study.

**Log per decided attempt, not per tool call.** One entry when a change is
kept or reverted. Not one per benchmark round, per candidate tile size inside a
single sweep (log the sweep and its conclusion), or per intermediate edit.

## Three Gates

1. **Evidence gate** — different for each kind.
   - `change`: a real device, `benchmarking.md` conventions. Baseline and
     variant share rounds, warmup, platform, device topology, and one frozen
     golden. A `*sim` run cannot produce a performance lesson; it can only
     confirm the change compiles and validates.
   - `method`: name the capture, tool output, or measurement the conclusion was
     read from, and say what it told you that the obvious alternative did not.
     "Run both tools" is not a lesson; "they disagreed on this run, and the
     second one was right, here is the number" is. A method with no evidence
     behind it is an opinion — do not log it.
2. **Generality gate** — the `Lesson` line must be actionable on a *different*
   kernel, shape, or session. "Set `BATCH_TILE = 8` in `decode_hca.py`" is a
   changelog line. "On a2a3, an N tile below the 512 B cache line costs more in
   MTE than it saves in L0C, so widen N before shrinking rows" is a lesson. If
   you cannot write that sentence, do not log the entry.
3. **Novelty gate** — not already covered by `docs/` or by an existing entry.
   If an entry already covers it, append the new evidence to that entry, raise
   its `Confidence`, append `(updated YYYY-MM-DD)` to its `Date`, and re-sort.
   Never add a second entry for the same lesson.

**Do NOT log:** anything from the lower half of the scope table above —
correctness, precision, DSL-usage, or compile fixes, however much they taught
you; a re-measurement of a variant already logged; a change you have not
measured; a simulator number; work still actively iterating — log when the
attempt is *decided*.

## Entry Format

```markdown
## [Short Title — the lesson, not the edit]

- **Date**: YYYY-MM-DD
- **Kind**: change | method
- **Target**: models/<family>/<file>.py::<operator>, or `general`
- **Lever**: [builtin rule or skill this came from, e.g.
  `performance-tuning.md#2-kernels-too-small` or `cube-tile-tuning`; `new` when
  no builtin rule suggested it]
- **Hypothesis**: [what you expected to move, and why]
- **Change**: [what was actually edited — `file:line`, one line of description]
- **Measurement**: [platform + device; rounds/warmup; metric convention;
  baseline → variant with the delta; golden replayed?]
- **Verdict**: kept | reverted | inconclusive
- **Lesson**: [one sentence a future session acts on]
- **Scope**: [shapes, platform, and pins this holds for; what would make it stop
  being true]
- **Repro**: [the exact command that re-measures this comparison from a clean
  tree, including rounds and warmup; never a `build_output/` path]
- **Generality**: cross-model | model-family | single-kernel
- **Confidence**: single | repeated | cross-model

---
```

A well-formed `Measurement` line looks like:

```text
a2a3 device 0; 100 rounds / 5 warmup; fastest-rank mean; 812.4 µs → 731.9 µs
(-9.9%); golden replayed
```

A well-formed `Repro` line looks like:

```text
PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=100 python models/deepseek_v4_flash_lowlat/decode_hca.py \
  -p a2a3 -d 0 --save-data      # then replay with --golden-data <work_dir>/data
```

**Never record a `build_output/` path.** An entry is permanent; a frozen
golden, a runtime dir, and a swimlane are not. The snapshot expires when specs,
inputs, or the golden logic change, the runtime dir expires when the kernel
source moves, and both are cleaned routinely — so within a week most recorded
paths are dead, and a dead path is worse than none because someone will try it.
The command regenerates whatever is missing and stays readable forever. Reusing
a live snapshot is a *session* concern, governed by
[`benchmarking.md`](benchmarking.md); the log does not carry it.

Pin the revision instead of the artifact: `Scope` names the pypto-lib revision
and toolchain pins the numbers were taken at, so a reader knows what the repro
command reproduces.

**Verdict** for a `method` entry is `adopted`, `rejected`, or `inconclusive` —
whether the technique earned its place in the loop. For a `change` entry it
follows the table in [`benchmarking.md`](benchmarking.md): wall time
decides. Wall down is `kept` whichever way busy time moved. Wall flat with busy
down is `inconclusive`, never a win. Wall up is `reverted`.

**Generality** is the claim's reach and is the primary sort key.
**Confidence** is the evidence behind it: `single` = one measured comparison,
`repeated` = re-measured across separate compiles or shapes, `cross-model` =
held on a second model.

## Sort Order

One list, no expiry. **Every add, update, or removal leaves the file sorted** —
never plain-append.

**Effective date** = the most recent date in `Date`: `2026-06-26 (updated
2026-08-30)` → `2026-08-30`. Record an edit by appending `(updated
YYYY-MM-DD)`, never by overwriting the original date.

| Key | Rule |
| --- | ---- |
| Sort key 1 | Generality: cross-model → model-family → single-kernel |
| Sort key 2 | Effective date: newest → oldest |

**Lessons do not expire on a calendar.** A measured mechanism holds until a
measurement says otherwise; the age of an entry is not evidence against it, and
nothing is demoted for sitting still. An entry leaves this file exactly three
ways: a new measurement disproves it, `/promote-lessons` moves it into the
checked-in guides, or the code it describes no longer exists.

`Scope` records the toolchain the lesson was measured under so a reader can
judge it. A pin bump is a reason to re-measure that entry the next time it is
relevant — not a reason to mark it down unread.

## On Task Completion

Before finishing any task that tuned for speed:

1. Read every entry.
2. Re-measure any entry this task's kernel change or toolchain bump may have
   disproved, and remove the ones that no longer hold. Removal takes a
   measurement, never a hunch or a date. Do not revert a workaround an entry
   names when it lives in a model this task did not touch — report it instead.
3. Confirm the file is still sorted.
4. Present the remaining lessons to the user as a summary.
5. Hint: "You can fold any of these into the checked-in guides with
   `/promote-lessons`."

**Do NOT promote on your own.** Promotion edits expert-verified, checked-in
documentation and is the user's call.

## Writing from a Worktree

Claude Code blocks `Edit` / `Write` against the main checkout from an isolated
worktree, and optimization work runs in worktrees constantly. Use the locked
read-modify-write recipe in
[`problem-handling.md`](problem-handling.md#writing-from-a-worktree) verbatim,
with `OPTIMIZATION_LESSONS.md` as the path and an entry heading as the anchor.
Its constraints all apply here: never `cd` into the main checkout, never
plain-append, hold the lock across the whole read-modify-write, require exactly
one anchor match and abort otherwise, and diff afterward.

Do **not** create a worktree-local `OPTIMIZATION_LESSONS.md`. One file, at the
main repo root (`git worktree list`, first entry).

## Important

- `OPTIMIZATION_LESSONS.md` and `OPTIMIZATION_LESSONS/` are gitignored —
  local-only, per-developer, never shared via git.
- **Never reference the file or its entries in shared artifacts.** Commit
  messages, PR descriptions, and GitHub issues must not name it or quote it;
  external readers cannot see it. Describe the measured result, not the entry.
- A lesson is never evidence for a performance claim to the user. Quote the
  measurement, and say when it was taken.
- Toolchain misbehaviour found while tuning belongs in
  [`problem-handling.md`](problem-handling.md)'s log, not this one. The two
  files do not overlap: that one records what the compiler got wrong, this one
  records what we learned about making kernels fast.
