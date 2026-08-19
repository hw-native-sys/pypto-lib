# Problem Handling and PyPTO Issue Tracking

## Core Principle

**pypto-lib is a model zoo, not a compiler.** When something here misbehaves the
fault is either *ours* (a kernel / harness / config mistake — fix it) or
*upstream's* (pypto, simpler, ptoas, pto-isa — log it, then work around it).

**Never silently work around a problem you believe is upstream.** The workaround
ships, the model looks healthy, and the compiler bug stays invisible to everyone
who would have fixed it.

```text
Technical problem encountered
├─ Fault is in pypto-lib's own code (kernel, harness, config)?
│  └─ YES → fix it as part of the current task. Do not log.
└─ You believe the DSL / model code is correct and the toolchain is at fault?
   └─ YES → log it to KNOWN_PYPTO_ISSUES.md — always, no exceptions
            └─ Does it block the current task?
               ├─ YES → also stop, inform the user, wait for their decision
               └─ NO  → continue with the current task
```

**Logging and blocking are independent decisions.** This differs from a normal
"log only non-blocking issues" rule: a compiler bug that blocks you is *more*
worth recording, not less. Log first (the entry outlives the conversation), then
stop and ask.

## The Trigger That Matters Most

> **If you believe the DSL is correct, but the program fails to compile or
> produces a wrong result — log an entry.**

That is the whole point of this file. Concretely, log when:

- A construct sanctioned by `docs/pypto-coding/pypto-coding-style.md` fails to
  compile (pypto traceback, IR validation error, codegen / g++ error, ptoas
  "failed to legalize", …).
- The kernel compiles and runs but the golden comparison FAILs, and the model
  code is doing what the style guide says it should.
- Execution hangs, crashes (`507xxx`, AICPU exception), or the scheduler times
  out on a program the DSL says is legal.
- The compiler accepts a construct it cannot lower, and only dies deep in the
  backend with a message naming internal SSA the user never wrote (a missing
  front-end diagnostic is itself a loggable defect).
- You had to write the kernel in an unnatural way to make it work. **The
  workaround does not cancel the entry — the workaround is the evidence.**

### Suspected owner

Name the owner in the entry; `unclear` is a valid answer.

| Owner | Local root | Owns |
|---|---|---|
| `pypto` | `$PYPTO_ROOT` | IR generation, lowering, codegen |
| `simpler` | `$PYPTO_ROOT/runtime` | on-device / sim execution, task-graph build & execute (AICPU + AICore) |
| `ptoas` | `$PTOAS_ROOT` | PTO bytecode assembly & optimization |
| `pto-isa` | `$PTO_ISA_ROOT` | virtual tile-ISA implementations |

A runtime crash / hang / AICPU error is a **simpler** issue, not a pypto one.

## Before You Blame the Toolchain

Three gates. Pass all three, then log — otherwise you are recording your own
mistake as a compiler bug.

1. **Style gate** — re-read `docs/pypto-coding/pypto-coding-style.md` for the
   construct in question. If the guide does not sanction what you wrote, fix the
   kernel.
2. **Pin gate** — a mismatched pypto / simpler / ptoas / pto-isa combination
   produces failures that belong to nobody. Confirm the checkouts match the pin
   chain in `docs/get-started/installation.md`; if they do not, align them and
   re-run before logging.
3. **Own-code gate** — for wrong numbers, rule out the lib-side causes in
   `docs/debug-and-tune/precision-tuning.md` first: cast/rounding mode vs torch,
   dtype alignment, double-casts, a wrong torch golden, an unreasonably tight
   tolerance.

**Do NOT log:** issues you are actively fixing; lib-side kernel bugs; DSL misuse;
limitations already documented in `docs/`; environment / setup breakage (use the
`setup-env` skill); failures caused by off-pin checkouts.

**When unsure whether it is ours or theirs:** log it with
`Suspected owner: unclear` and say so in the description. An entry that turns out
to be our own bug costs one deletion; an unlogged compiler bug costs a release.

## Blocking Problems

**A problem is blocking when you cannot make meaningful progress on the current
task without resolving it.** Examples: the kernel will not compile at all, the
harness cannot produce a golden, results are wrong in a way that may mean your
change is wrong, a device hang that eats every run.

**What to do:** 1. **Log it** if it passed the three gates. 2. **Stop** — no
workarounds, no assumptions. 3. **Describe the problem clearly** — what happened,
what you expected, why it blocks progress. 4. **Present options** with
trade-offs. 5. **Wait for the user's decision.**

**When unsure if blocking:** err on the side of asking. If the problem might
affect numerical correctness, treat it as blocking.

## The Log File

**Path:** `KNOWN_PYPTO_ISSUES.md` at the **main repository root** — always the
main checkout, even when working in a git worktree (`git worktree list`, first
entry). It is gitignored: local-only, per-developer, never shared via git.

It contains **unresolved** issues only; resolved ones are removed entirely. Two
top-level sections — active entries, then stale ones. Same entry format in both;
stale entries add a `Status` line.

```markdown
# Known PyPTO Issues

## [Short Title — the defect, not the symptom you hit]

- **Date**: YYYY-MM-DD
- **Found during**: [what task surfaced it]
- **Suspected owner**: pypto | simpler | ptoas | pto-isa | unclear
- **Symptom class**: compile | runtime | hang | precision | performance
- **Environment**: [platform + versions — see "Environment" below]
- **Description**: [actual vs. expected behaviour; why the DSL is believed
  correct; what it costs the model]
- **Minimal repro**: [see "Minimal Reproducible Case" — mandatory, never `N/A`]
- **Workaround**: [what the model does instead and where it lives, so it can be
  reverted when the bug is fixed; `none` if there is none]
- **Location**: [lib-side `file:line` where it surfaces; upstream `file:line` if
  identified]
- **Severity**: low | medium | high
- **Filed**: hw-native-sys/<repo>#NNN   [omit until filed]

---

# Stale (no update in over 2 months as of YYYY-MM-DD)

> Entries below are dated before YYYY-MM-DD. Kept for reference; re-verify before
> acting.

## [Short Title]

- **Date**: YYYY-MM-DD
- **Status**: stale (>2 months old as of YYYY-MM-DD)
- ... (remaining fields unchanged)

---
```

**Severity:** `high` = blocks a model build, or corrupts numerics with no
workaround. `medium` = a workaround exists but costs something real (performance,
precision, or a shape/feature we cannot use). `low` = cosmetic, diagnostic
quality, or a documentation gap.

**Environment:** one line — platform first, then the five components and CANN,
e.g. `a2a3 (device 0); pypto-lib 6741e82, pypto a1b2c3d, simpler 4e5f6a7, ptoas
1.4.0, pto-isa 89ab0cd, CANN 8.0.RC3`. The `create-issue` skill's Step 2 has the
exact collection commands — reuse them rather than re-deriving. Record `unknown`
for anything undetectable. A detached HEAD is normal for simpler and pto-isa and
still has a revision: record the short hash and mark it, `4e5f6a7 (detached)` —
never `detached` alone, which throws away what the reader needs to reproduce.

## Minimal Reproducible Case

**Entries in this repo carry more context than a one-line note, and every entry
needs a minimal reproducible case.** A pointer to a 400-line model kernel is not
a repro — nobody upstream will bisect it, and neither will you in two months.

A repro qualifies when it is:

- **Reduced** — the smallest program that still shows the behaviour. Strip other
  layers, shrink shapes to the minimum that still triggers it, drop unrelated
  ops, collapse loops.
- **Self-contained** — imports nothing from `models/` or `examples/`, needs no
  local weights or scratch files, runs standalone.
- **Cheap to run** — prefer compile-only and the simulator (`-p a2a3sim`) so a
  reader without a device can reproduce it. If it only reproduces on hardware,
  say so explicitly and give the platform and device flags.
- **Exact** — the literal command plus the observed output: the error-message
  tail for compile/runtime failures, or the first failing element with its
  relative error for precision failures.

**Where to put it:** inline the kernel in the entry when it is ≲80 lines.
Beyond that, write the runnable file to `KNOWN_PYPTO_ISSUES/<slug>.py` (also
gitignored) and inline the failing fragment plus that path.

**If reduction fails** after honest effort: still log — with the smallest case you
reached, an explicit `Minimal repro: not yet minimized —` prefix describing what
you already ruled out, and mention it to the user when you report. Do not skip the
entry, and do not pass off an unreduced case as minimal.

## Sections and Sort Order

**Every add, update, or removal must leave the file sorted** — never append to the
end of the file (the end is *inside* the Stale section).

**Effective date** = the most recent date in the `Date` field: `2026-06-26
(updated 2026-06-28)` → `2026-06-28`. Record an edit by appending `(updated
YYYY-MM-DD)` rather than overwriting the original date.

| Step | Rule |
| ---- | ---- |
| Section | Effective date on or after (today − 2 calendar months) → `# Known PyPTO Issues`; earlier → `# Stale (...)` |
| Sort key 1 | Severity: high → medium → low |
| Sort key 2 | Effective date: newest → oldest |

Each section is sorted independently: a new `high` entry goes to the very top of
`# Known PyPTO Issues`; a new `low` entry goes *before* every older `low` entry.

**On every touch:** insert or move the entry to its sorted position (a `Severity`
or date change moves it); a removal moves nothing else. **Migration runs both
ways** whenever an effective date crosses the cutoff — moving *into* Stale, add
`- **Status**: stale (>2 months old as of <today>)` right after the `Date` line;
moving *back into* active because an update refreshed the date, delete that
`Status` line. Either way, re-sort within the new section and refresh the dates in
the Stale heading and its blockquote.

## How to Log

1. Determine the main repo root (`git worktree list` — first entry)
2. Read `KNOWN_PYPTO_ISSUES.md` (create it if absent)
3. Check the issue is not already logged (avoid duplicates); if it is, append
   `(updated YYYY-MM-DD)` to its `Date`, add the new evidence, re-sort
4. Reduce the case (see "Minimal Reproducible Case") and collect the environment
5. Find the insertion anchor — the first entry this one sorts *before*; when
   none, the `# Stale` heading
6. Insert the entry there — verify it meets the bar above before saving
7. Continue with the current task when the issue is non-blocking — do not chase
   the logged bug now. When it blocks the task, stop after logging, present the
   options, and wait for the user's decision (see "Blocking Problems")

## Writing from a Worktree

While a session is isolated in a worktree, Claude Code **blocks `Edit` / `Write` /
`NotebookEdit` against the main checkout**. Reads still work. Do **not** create a
worktree-local `KNOWN_PYPTO_ISSUES.md` as a workaround — that fragments the file,
which is exactly what the "main repo root" rule prevents.

Use a Bash command instead, run from the worktree cwd — isolation checks test a
command's *working directory* and *git redirects*, not file writes by absolute
path:

```bash
python3 - <<'PY'
import fcntl, os
p = '/abs/path/to/main-repo/KNOWN_PYPTO_ISSUES.md'
entry_text = """## Short Title

- **Date**: YYYY-MM-DD
- ... (remaining fields)

---

"""
# Start of the heading line this entry sorts BEFORE; "\n# Stale" (a prefix of the
# dated heading) is the anchor when the entry sorts last among active entries.
anchor = "\n## Heading of the first entry that sorts after this one"
# One lock file guards the whole read-modify-write: another worktree session can
# otherwise read the same text and overwrite this entry with its own.
with open(p + '.lock', 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        text = open(p).read()
    except FileNotFoundError:  # first entry in this checkout: seed both sections
        text = "# Known PyPTO Issues\n\n# Stale (no update in over 2 months as of YYYY-MM-DD)\n"
    if text.count(anchor) != 1: raise SystemExit("anchor not unique — abort, no write")
    idx = text.index(anchor) + 1
    tmp = p + '.tmp'
    open(tmp, 'w').write(text[:idx] + entry_text + text[idx:])
    os.replace(tmp, p)  # atomic: readers never see a half-written log
PY
```

The heredoc is quoted (`<<'PY'`), so the entry text passes through verbatim —
define it inside the snippet rather than via a shell variable.

Constraints that make this work:

- **Never `cd` into the main checkout**, and never point `git -C` / `--git-dir` /
  `GIT_DIR` / `GIT_WORK_TREE` at it — each is independently blocked.
- **Keep the command simple.** Compound commands (`&&`, `;`-chains, redirects) are
  refused as unverifiable. When even the heredoc is refused, write the same script
  to a scratch file and run it by path rather than simplifying the script.
- **Never plain-append** (`open(p, 'a')`): the file's end is inside the Stale
  section, so the entry lands unsorted in the wrong section.
- **Hold the lock across the whole read-modify-write and replace atomically.**
  Two worktree sessions logging at once would otherwise both read the original
  text, and the later write would silently drop the earlier entry.
- **Anchor on entry heading text, never line numbers — and require exactly one
  match.** Count matches first and **abort without modifying the file** on zero or
  more than one; never rewrite the first match.
- **Back it up first** (`cp` to a scratch dir) and diff afterward to confirm only
  the intended hunk changed. The diff proves *something* changed, not that the
  *right* entry changed — it supplements the unique-match check, it does not
  replace it.

## On Task Completion

**Before finishing any task, revisit `KNOWN_PYPTO_ISSUES.md`:**

1. Read all entries
2. Re-run the minimal repro of any entry the current task's toolchain bump or
   change might have fixed — **this is what the repro is for**. Remove entries
   that no longer reproduce. Revert a workaround the entry names only when it
   lives in code this task already touches; when it sits in another model, leave
   it alone and report it, since removing it changes kernel behaviour the user
   did not ask you to touch.
3. Move any active entry now older than 2 months into Stale, then confirm both
   sections are sorted (severity high→low, date newest→oldest)
4. Present the remaining issues to the user as a summary
5. Hint: "You may want to file any of these upstream with `/create-issue`" — the
   skill takes the problem as its input and has no known-issue picker, so hand it
   the entry's description and repro, then record the URL it returns in that
   entry's `Filed` field

**Do NOT ask the user to fix these issues now** — just inform them.

## Important

- `KNOWN_PYPTO_ISSUES.md` and `KNOWN_PYPTO_ISSUES/` are in `.gitignore` —
  local-only tracking, independent per developer, never shared via git
- **Never reference the file or its entries in shared artifacts** — commit
  messages, PR descriptions, and GitHub issues must not name it or quote it.
  External readers cannot see it. Describe the actual problem, not the local entry.
- **Always write to the main repo root**, never a worktree directory — from a
  worktree this requires Bash, since `Edit` is blocked
- Use the `create-issue` skill to promote an entry to a real GitHub issue; it
  reproduces on the current environment, collects the pin-bound versions, and
  routes to the owning repo. Record the resulting URL in the entry's `Filed`
  field and keep the entry until the bug is actually fixed.
