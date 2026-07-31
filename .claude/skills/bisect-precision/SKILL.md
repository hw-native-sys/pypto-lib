---
name: bisect-precision
description: Locate the PyPTO or simpler commit that introduced a pypto-lib precision regression while keeping ptoas and pto-isa fixed. Use when a case has a known-good PyPTO revision and a reproducible bad revision; perform a second-level simpler bisect only when the PyPTO culprit is a runtime submodule bump.
---

# Bisect a Precision Regression

Read [Precision Regression Bisect](../../../docs/debug-and-tune/precision-regression-bisect.md)
before changing a checkout. That guide is the canonical method and reporting
reference. Use this skill to manage the live repositories, repeated installs,
classification, progress updates, and guaranteed restoration.

## Boundaries

- Bisect PyPTO first. Each PyPTO commit selects its `runtime` (simpler)
  submodule revision.
- Keep the installed `ptoas` and `pto-isa` revisions fixed and report them.
- Bisect simpler only when the first bad PyPTO commit is a runtime gitlink
  bump.
- Use the exact same pypto-lib case, platform, device, fixed data, and
  comparator at every revision.
- Do not treat unrelated build, environment, or device failures as bad
  precision.

## Gather required context

Resolve:

- pypto-lib case and full command;
- platform and device;
- known-good PyPTO commit;
- confirmed-bad PyPTO commit;
- PyPTO repository and its `runtime` submodule path;
- installed `ptoas` and `pto-isa` versions; and
- fixed replay data when available.

Ask for a known-good commit if none is supplied or discoverable from explicit
task context. Do not read private memory files or infer one from an unrelated
historical note.

## Protect user state

Before bisecting, record:

```bash
git -C <pypto-dir> status --short --branch
git -C <pypto-dir> rev-parse HEAD
git -C <pypto-dir> symbolic-ref --short -q HEAD
git -C <pypto-dir>/runtime status --short --branch
git -C <pypto-dir>/runtime rev-parse HEAD
```

If either checkout has uncommitted user changes, do not stash, discard, or
overwrite them. Use a safe dedicated checkout when practical or ask the user
to choose how to proceed.

Verify that both endpoint commits exist, the known-good commit is an ancestor
of the bad commit, and both endpoints reproduce under the same environment.
Record the original refs for restoration.

## Optional generated-source triage

Build both endpoints with the fixed input and compare only textual generated
sources in `kernels/`, `ptoas/`, and `orchestration/`. Report whether they
differ, but do not change the first-level strategy based on this hint.

## First-level PyPTO loop

Start `git bisect` with the confirmed endpoints. At every selected revision:

1. Run `git submodule update --init runtime`.
2. Reinstall editable PyPTO and simpler with `--no-build-isolation`.
3. Run the exact pypto-lib command.
4. Mark:
   - `good` only for the expected numerical pass;
   - `bad` only for the reproduced precision failure;
   - `skip` for incompatible APIs, installation failures, device failures,
     crashes unrelated to validation, or ambiguous results.
5. Report progress as commit, subject, verdict, and remaining range.

Never classify a generic nonzero exit as a precision failure. Preserve enough
raw output to justify every verdict.

When Git identifies the first bad commit, capture its full metadata, stat, and
relevant diff before resetting the bisect.

## Second-level simpler loop

Inspect whether the PyPTO culprit changes only the `runtime` gitlink. If so:

1. Read the old pointer from `<culprit>^:runtime`.
2. Read the new pointer from `<culprit>:runtime`.
3. Keep PyPTO at the culprit and install it once.
4. Bisect the simpler checkout between those pointers.
5. Reinstall simpler and run the same pypto-lib command for each revision.
6. Apply the same good/bad/skip rules and progress reporting.

If PyPTO source also changed, do not assume simpler is solely responsible.
Report the PyPTO culprit unless the user requests a separate experiment.

## Restore in all outcomes

Restoration is mandatory after success, failure, cancellation, or excessive
skips:

1. Reset an active simpler bisect.
2. Reset an active PyPTO bisect.
3. Restore the recorded PyPTO branch/ref and commit.
4. Synchronize `runtime` to the restored gitlink.
5. Reinstall editable PyPTO and simpler.
6. Confirm both repository statuses and HEADs.
7. Rerun the original bad endpoint when feasible to confirm environment
   restoration.

If restoration fails, stop other work and report the exact repository state;
never leave the toolchain silently pinned to a bisect midpoint.

## Report

Return:

- reproduction command and fixed-data source;
- good and bad PyPTO endpoints;
- fixed `ptoas` and `pto-isa` revisions;
- result of generated-source triage, if performed;
- every skipped commit and reason;
- first bad PyPTO commit and relevant changes;
- the PyPTO runtime bump and first bad simpler commit, when applicable;
- a causal hypothesis clearly distinguished from the bisect result; and
- final restored refs, statuses, and validation result.
