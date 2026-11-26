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

Check the existing managed ISA path with `simpler_setup.pto_isa.get_pto_isa_clone_path()`
before building. Use a separate checkout and environment if an install or
resolver could replace user changes in PyPTO, simpler or its managed ISA tree.

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
the same environment using the checks below before starting. Record the original
refs and runtime build targets for restoration.

## Hold the assembler and ISA fixed

Record the common full ISA commit as `FIXED_ISA`, the selected target as
`RUNTIME_PLATFORM` (`a2a3`, `a5`, `a2a3sim` or `a5sim`), and the effective
PTOAS launcher's path and full `--version` output. Keep the same assembler
installation throughout. At each endpoint and midpoint, use the active PyPTO
revision's launcher discovery to confirm that path and version.

Simpler can resolve its managed ISA during installation. Check the candidate
pin **before installing simpler**, then verify the effective checkout in the
same environment that will compile the case:

```bash
install_bisect_runtime() {
  local pin isa_dir isa_head
  pin=$(cat "${SIMPLER_DIR}/pto_isa.pin") || return 1
  if [ "$pin" != "${FIXED_ISA:?set the common full ISA commit}" ]; then
    printf 'Candidate ISA pin %s differs from fixed ISA %s\n' "$pin" "$FIXED_ISA" >&2
    return 1
  fi
  python -m pip install --no-build-isolation -e "${SIMPLER_DIR}" \
    --config-settings="build.targets=build_package_${RUNTIME_PLATFORM:?select the case target}" || return 1
  isa_dir=$(python -c 'from pypto.runtime import ensure_pto_isa_root; print(ensure_pto_isa_root())') || return 1
  isa_head=$(git -C "$isa_dir" rev-parse HEAD) || return 1
  printf 'Effective ISA: %s\n' "$isa_head"
  [ "$isa_head" = "$FIXED_ISA" ]
}
```

For an older revision, use its equivalent install and verification procedure
if it can establish the same inputs. Skip a midpoint with an incompatible
pin or unverifiable toolchain and record the reason. If either endpoint cannot
reproduce with the fixed inputs, report the comparison as inconclusive.
`PTO_ISA_ROOT` cannot override a pin selected by the managed resolver.

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
2. Reinstall editable PyPTO and run `install_bisect_runtime`.
3. After the toolchain checks pass, run the exact pypto-lib command.
4. Mark:
   - `good` only for the expected numerical pass;
   - `bad` only for the reproduced precision failure;
   - `skip` for incompatible APIs, installation failures, device failures,
     crashes unrelated to validation, or ambiguous results.
5. Report progress as commit, subject, verdict, and remaining range.

Prepare each candidate with:

```bash
git -C "${PYPTO_DIR}" submodule update --init runtime &&
  python -m pip install --no-build-isolation -e "${PYPTO_DIR}" &&
  install_bisect_runtime
```

After a successful setup, run the recorded reproduction command from
`PYPTO_LIB_DIR`. Record exactly one supported verdict:

```bash
git -C "${PYPTO_DIR}" bisect good
git -C "${PYPTO_DIR}" bisect bad
git -C "${PYPTO_DIR}" bisect skip
```

Never classify a generic nonzero exit as a precision failure. Preserve enough
raw output to justify every verdict.

If skips leave several possible first bad revisions, report the candidate set.
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
4. Recheck both simpler endpoints with PyPTO held at the culprit. Start the
   nested bisect only if the old pointer passes and the new pointer reproduces.
5. Run `install_bisect_runtime` for each candidate before running the same case.
6. Apply the same good/bad/skip rules and progress reporting.

Extract the endpoints and hold PyPTO at the culprit:

```bash
SIMPLER_GOOD="$(
  git -C "${PYPTO_DIR}" rev-parse "<pypto-culprit>^:runtime"
)"
SIMPLER_BAD="$(
  git -C "${PYPTO_DIR}" rev-parse "<pypto-culprit>:runtime"
)"

git -C "${PYPTO_DIR}" checkout <pypto-culprit>
python -m pip install --no-build-isolation -e "${PYPTO_DIR}"
```

After confirming each endpoint with `install_bisect_runtime` and the case,
start the nested bisect:

```bash
git -C "${SIMPLER_DIR}" bisect start
git -C "${SIMPLER_DIR}" bisect bad "${SIMPLER_BAD}"
git -C "${SIMPLER_DIR}" bisect good "${SIMPLER_GOOD}"
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

Use the recorded original branch or detached commit for each checkout.
Reapply the original runtime build targets when reinstalling:

```bash
if git -C "${SIMPLER_DIR}" bisect log >/dev/null 2>&1; then
  git -C "${SIMPLER_DIR}" bisect reset
fi
if git -C "${PYPTO_DIR}" bisect log >/dev/null 2>&1; then
  git -C "${PYPTO_DIR}" bisect reset
fi

git -C "${PYPTO_DIR}" checkout <original-pypto-branch-or-commit>
git -C "${PYPTO_DIR}" submodule update --init runtime
git -C "${SIMPLER_DIR}" checkout <original-simpler-branch-or-commit>

python -m pip install --no-build-isolation -e "${PYPTO_DIR}"
# Reinstall simpler with the original flags recorded above.

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
