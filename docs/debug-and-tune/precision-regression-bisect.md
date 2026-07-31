# Precision Regression Bisect

Use this workflow when the same pypto-lib case passed with an older PyPTO
revision and now fails numerical validation. The first search tracks PyPTO
commits and the simpler revision pinned by each PyPTO commit. If the first bad
PyPTO commit only advances the `runtime` submodule, run a second search inside
simpler.

This workflow deliberately keeps `ptoas` and `pto-isa` fixed. If either
changed between the good and bad observations, first reproduce both endpoints
with one common assembler and ISA revision; otherwise the bisect does not
isolate a single variable.

For techniques that localize the numerical error inside one revision, see
[Precision Tuning](../precision-tuning.md) and
[Debugging](../debugging.md).

## Establish comparable endpoints

Before changing a checkout, record:

- the pypto-lib case and exact command;
- platform and device;
- a saved input/golden dataset when available;
- the known-good PyPTO commit;
- the currently confirmed bad PyPTO commit;
- installed `ptoas` and `pto-isa` revisions;
- PyPTO and simpler repository status; and
- the original branch or detached HEAD for both repositories.

The good and bad endpoints must run the same case, data, comparison, and
environment. A failure to build, a device outage, or an unrelated runtime
exception is not evidence of a precision regression.

Do not start in a checkout with uncommitted changes that the process could
overwrite. Move the changes to a safe branch or use a dedicated checkout.

Define paths without embedding a machine-specific layout:

```bash
PYPTO_DIR=/path/to/pypto
SIMPLER_DIR="${PYPTO_DIR}/runtime"
PYPTO_LIB_DIR=/path/to/pypto-lib
```

Confirm the endpoints and ancestry:

```bash
PYPTO_GOOD=<known-good-commit>
PYPTO_BAD="$(git -C "${PYPTO_DIR}" rev-parse HEAD)"

git -C "${PYPTO_DIR}" merge-base --is-ancestor \
  "${PYPTO_GOOD}" "${PYPTO_BAD}"
git -C "${PYPTO_DIR}" log --oneline \
  "${PYPTO_GOOD}..${PYPTO_BAD}"
```

If the known-good commit is not an ancestor of the bad commit, identify a
merge-base or a good commit on the relevant history before continuing.

## Optional triage: compare generated sources

Build the same case at both endpoints and compare generated textual sources:

```text
build_output/<good>/kernels/
build_output/<good>/ptoas/
build_output/<good>/orchestration/

build_output/<bad>/kernels/
build_output/<bad>/ptoas/
build_output/<bad>/orchestration/
```

Different `.cpp` or `.pto` files suggest compiler code generation changed.
Identical generated sources make a runtime change more likely. This is only a
triage signal: the first-level bisect still runs across PyPTO because each
commit also selects its simpler submodule revision.

Ignore timestamps and compiled objects when comparing builds.

## First-level bisect: PyPTO

Start the search:

```bash
git -C "${PYPTO_DIR}" bisect start
git -C "${PYPTO_DIR}" bisect bad "${PYPTO_BAD}"
git -C "${PYPTO_DIR}" bisect good "${PYPTO_GOOD}"
```

For every selected PyPTO commit:

1. Synchronize the pinned runtime submodule.
2. Reinstall PyPTO and simpler from those checkouts.
3. Run the exact pypto-lib validation command.
4. Classify only the intended precision verdict.

```bash
git -C "${PYPTO_DIR}" submodule update --init runtime
python -m pip install --no-build-isolation -e "${PYPTO_DIR}"
python -m pip install --no-build-isolation -e "${SIMPLER_DIR}"

cd "${PYPTO_LIB_DIR}"
python <case.py> -p <platform> -d <device> <replay-arguments>
```

Mark the revision:

```bash
git -C "${PYPTO_DIR}" bisect good
git -C "${PYPTO_DIR}" bisect bad
git -C "${PYPTO_DIR}" bisect skip
```

Use `good` only when the intended numerical check passes and `bad` only when
that check reproduces the regression. Use `skip` for an incompatible
intermediate API, installation failure, device failure, or any result that
cannot be classified. Do not label every nonzero process exit as `bad`.

After each step, record the short hash, subject, result, and relevant output.
Continue until Git identifies the first bad PyPTO commit.

## Decide whether to bisect simpler

Inspect the culprit:

```bash
git -C "${PYPTO_DIR}" show --stat <pypto-culprit>
git -C "${PYPTO_DIR}" diff \
  "<pypto-culprit>^" "<pypto-culprit>" -- runtime
```

If compiler or other PyPTO sources changed, report the PyPTO commit as the
culprit and inspect its relevant diff.

If the commit only changes the `runtime` gitlink, extract the old and new
simpler revisions:

```bash
SIMPLER_GOOD="$(
  git -C "${PYPTO_DIR}" rev-parse "<pypto-culprit>^:runtime"
)"
SIMPLER_BAD="$(
  git -C "${PYPTO_DIR}" rev-parse "<pypto-culprit>:runtime"
)"
```

Keep PyPTO checked out at the culprit so only simpler changes during the
second-level search.

## Second-level bisect: simpler

Install the culprit PyPTO compiler, then start inside the submodule:

```bash
git -C "${PYPTO_DIR}" checkout <pypto-culprit>
python -m pip install --no-build-isolation -e "${PYPTO_DIR}"

git -C "${SIMPLER_DIR}" bisect start
git -C "${SIMPLER_DIR}" bisect bad "${SIMPLER_BAD}"
git -C "${SIMPLER_DIR}" bisect good "${SIMPLER_GOOD}"
```

For each selected simpler commit:

```bash
python -m pip install --no-build-isolation -e "${SIMPLER_DIR}"

cd "${PYPTO_LIB_DIR}"
python <case.py> -p <platform> -d <device> <replay-arguments>
```

Apply the same good/bad/skip rules. The result is the first bad simpler commit;
the PyPTO gitlink bump remains the commit that introduced it into the
toolchain.

## Always restore the environment

Reset every active bisect and restore the original recorded revisions,
including the submodule pointer:

```bash
git -C "${SIMPLER_DIR}" bisect reset
git -C "${PYPTO_DIR}" bisect reset

git -C "${PYPTO_DIR}" checkout "${PYPTO_BAD}"
git -C "${PYPTO_DIR}" submodule update --init runtime
python -m pip install --no-build-isolation -e "${PYPTO_DIR}"
python -m pip install --no-build-isolation -e "${SIMPLER_DIR}"
```

Run restoration after success, failure, or interruption. If either repository
started on a named branch rather than at `PYPTO_BAD`, restore the recorded
branch and confirm its HEAD. Re-run the bad endpoint once to prove the editable
environment matches the original state.

## Report the result

Include:

- case, platform, device, and fixed-data source;
- known-good and known-bad endpoint commits;
- fixed `ptoas` and `pto-isa` revisions;
- first bad PyPTO commit, subject, author, and date;
- first bad simpler commit when a gitlink bump required a second bisect;
- skipped revisions and why they could not be classified;
- the relevant source diff and a concise precision hypothesis; and
- confirmation that PyPTO and simpler were restored.

The bisect identifies the first revision correlated with the regression. A
causal explanation still requires reading the diff and reproducing the
specific numerical change.
