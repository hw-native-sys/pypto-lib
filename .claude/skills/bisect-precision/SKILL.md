---
name: bisect-precision
description: Locate the PyPTO or simpler commit that introduced a pypto-lib precision regression while keeping ptoas and pto-isa fixed. Use when a case has a known-good PyPTO revision and a reproducible bad revision; perform a second-level simpler bisect only when the PyPTO culprit is a runtime submodule bump.
---

# Bisect a Precision Regression

## Boundaries

- Bisect PyPTO first. Each PyPTO commit selects its `runtime` (simpler)
  submodule revision.
- Keep the installed `ptoas` and `pto-isa` revisions fixed and report them.
- If the original good and bad observations used different PTOAS or PTO ISA
  revisions, first reproduce both endpoints with one common assembler and ISA
  revision. Otherwise the search does not isolate a single variable.
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

Define task-local paths without embedding a machine-specific layout:

```bash
PYPTO_DIR=/path/to/pypto
SIMPLER_DIR="${PYPTO_DIR}/runtime"
PYPTO_LIB_DIR=/path/to/pypto-lib
```

Before bisecting, record both repositories' status, commit, and whether each
started on a named branch or detached HEAD:

```bash
git -C "${PYPTO_DIR}" status --short --branch
git -C "${PYPTO_DIR}" rev-parse HEAD
git -C "${PYPTO_DIR}" symbolic-ref --short -q HEAD
git -C "${SIMPLER_DIR}" status --short --branch
git -C "${SIMPLER_DIR}" rev-parse HEAD
git -C "${SIMPLER_DIR}" symbolic-ref --short -q HEAD
```

If either checkout has uncommitted user changes, do not stash, discard, or
overwrite them. Use a safe dedicated checkout when practical or ask the user
to choose how to proceed.

Resolve and inspect the endpoints:

```bash
PYPTO_GOOD=<known-good-commit>
PYPTO_BAD="$(git -C "${PYPTO_DIR}" rev-parse <confirmed-bad-ref>)"

git -C "${PYPTO_DIR}" rev-parse "${PYPTO_GOOD}^{commit}"
git -C "${PYPTO_DIR}" rev-parse "${PYPTO_BAD}^{commit}"
git -C "${PYPTO_DIR}" merge-base --is-ancestor \
  "${PYPTO_GOOD}" "${PYPTO_BAD}"
git -C "${PYPTO_DIR}" log --oneline \
  "${PYPTO_GOOD}..${PYPTO_BAD}"
```

If the good commit is not an ancestor of the bad commit, identify a merge-base
or a known-good commit on the relevant history. Reproduce both endpoints under
the same environment before starting. Record the original refs separately from
the endpoint values so restoration does not assume the bad endpoint was the
user's original checkout.

## Optional generated-source triage

Build both endpoints with the fixed input and compare only generated textual
`.cpp` and `.pto` sources under `kernels/`, `ptoas/`, and `orchestration/`.
Ignore timestamps, shared objects, and other compiled artifacts. Different
sources suggest a compiler change; identical sources make a runtime change
more likely. Treat either result only as triage and do not change the
PyPTO-first strategy.

## First-level PyPTO loop

Start with the confirmed endpoints:

```bash
git -C "${PYPTO_DIR}" bisect start
git -C "${PYPTO_DIR}" bisect bad "${PYPTO_BAD}"
git -C "${PYPTO_DIR}" bisect good "${PYPTO_GOOD}"
```

At every selected revision:

1. Run `git submodule update --init runtime`.
2. Reinstall editable PyPTO and simpler with `--no-build-isolation`.
3. Run the exact pypto-lib command.
4. Mark:
   - `good` only for the expected numerical pass;
   - `bad` only for the reproduced precision failure;
   - `skip` for incompatible APIs, installation failures, device failures,
     crashes unrelated to validation, or ambiguous results.
5. Report progress as commit, subject, verdict, and remaining range.

Use this command skeleton for each candidate:

```bash
git -C "${PYPTO_DIR}" submodule update --init runtime
python -m pip install --no-build-isolation -e "${PYPTO_DIR}"
python -m pip install --no-build-isolation -e "${SIMPLER_DIR}"

cd "${PYPTO_LIB_DIR}"
python <case.py> -p <platform> -d <device> <replay-arguments>
```

Then record exactly one supported verdict:

```bash
git -C "${PYPTO_DIR}" bisect good
git -C "${PYPTO_DIR}" bisect bad
git -C "${PYPTO_DIR}" bisect skip
```

Never classify a generic nonzero exit as a precision failure. Preserve enough
raw output to justify every verdict.

When Git identifies the first bad commit, capture its full metadata, stat, and
relevant diff before resetting the bisect:

```bash
git -C "${PYPTO_DIR}" show --format=fuller --stat <pypto-culprit>
git -C "${PYPTO_DIR}" diff \
  "<pypto-culprit>^" "<pypto-culprit>"
```

## Second-level simpler loop

Inspect the runtime pointer separately:

```bash
git -C "${PYPTO_DIR}" diff \
  "<pypto-culprit>^" "<pypto-culprit>" -- runtime
```

If the PyPTO culprit changes only the `runtime` gitlink:

1. Read the old pointer from `<culprit>^:runtime`.
2. Read the new pointer from `<culprit>:runtime`.
3. Keep PyPTO at the culprit and install it once.
4. Bisect the simpler checkout between those pointers.
5. Reinstall simpler and run the same pypto-lib command for each revision.
6. Apply the same good/bad/skip rules and progress reporting.

Extract the endpoints and start the nested search with PyPTO held at the
culprit:

```bash
SIMPLER_GOOD="$(
  git -C "${PYPTO_DIR}" rev-parse "<pypto-culprit>^:runtime"
)"
SIMPLER_BAD="$(
  git -C "${PYPTO_DIR}" rev-parse "<pypto-culprit>:runtime"
)"

git -C "${PYPTO_DIR}" checkout <pypto-culprit>
python -m pip install --no-build-isolation -e "${PYPTO_DIR}"

git -C "${SIMPLER_DIR}" bisect start
git -C "${SIMPLER_DIR}" bisect bad "${SIMPLER_BAD}"
git -C "${SIMPLER_DIR}" bisect good "${SIMPLER_GOOD}"
```

For each simpler candidate, reinstall only simpler, run the same command, and
apply the same good/bad/skip rules:

```bash
python -m pip install --no-build-isolation -e "${SIMPLER_DIR}"

cd "${PYPTO_LIB_DIR}"
python <case.py> -p <platform> -d <device> <replay-arguments>
```

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

Use the recorded original branch or detached commit for each checkout:

```bash
if git -C "${SIMPLER_DIR}" rev-parse --verify BISECT_HEAD >/dev/null 2>&1; then
  git -C "${SIMPLER_DIR}" bisect reset
fi
if git -C "${PYPTO_DIR}" rev-parse --verify BISECT_HEAD >/dev/null 2>&1; then
  git -C "${PYPTO_DIR}" bisect reset
fi

git -C "${PYPTO_DIR}" checkout <original-pypto-branch-or-commit>
git -C "${PYPTO_DIR}" submodule update --init runtime
git -C "${SIMPLER_DIR}" checkout <original-simpler-branch-or-commit>

python -m pip install --no-build-isolation -e "${PYPTO_DIR}"
python -m pip install --no-build-isolation -e "${SIMPLER_DIR}"

git -C "${PYPTO_DIR}" status --short --branch
git -C "${PYPTO_DIR}" rev-parse HEAD
git -C "${SIMPLER_DIR}" status --short --branch
git -C "${SIMPLER_DIR}" rev-parse HEAD
```

If restoration fails, stop other work and report the exact repository state;
never leave the toolchain silently pinned to a bisect midpoint.

## Report

Return:

- reproduction command and fixed-data source;
- good and bad PyPTO endpoints;
- fixed `ptoas` and `pto-isa` revisions;
- result of generated-source triage, if performed;
- every skipped commit and reason;
- first bad PyPTO commit, subject, author, date, and relevant changes;
- the PyPTO runtime bump that introduced the simpler revision and the first bad
  simpler commit, when applicable;
- a causal hypothesis clearly distinguished from the bisect result; and
- final restored refs, statuses, and validation result.

The search identifies the first revision correlated with the regression. Keep
that result distinct from causality, which requires reading the relevant diff
and reproducing the specific numerical change.
