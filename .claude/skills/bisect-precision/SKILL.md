---
name: bisect-precision
description: Locate which pypto commit introduced a precision regression. Only pypto and its corresponding simpler (submodule) are tracked — ptoas and pto-isa versions are not part of the bisect. If the culprit is a simpler submodule bump, performs a second-level bisect within simpler.
---

# Bisect Precision Regression

Locate the exact commit that introduced a precision regression in a pypto-lib
example by binary-searching through pypto history. Simpler (runtime) is a git
submodule of pypto, so each pypto commit also pins a specific simpler version.
**This skill only tracks pypto + simpler** — ptoas and pto-isa are assumed
stable and are NOT bisected, even if pypto's CI config references newer
versions.

## Repository Layout

| Component | Path | How it's tracked |
|-----------|------|------------------|
| **pypto** (compiler) | `<pypto_dir>` | Independent repo — the **first-level bisect target** |
| **simpler** (runtime) | `<pypto_dir>/runtime` | Git submodule of pypto — the **second-level bisect target** (only if needed) |
| **ptoas** (assembler) | external binary | **Not tracked by this skill.** Assumed stable at the currently installed version. |
| **pto-isa** | instruction set definitions | **Not tracked by this skill.** Assumed stable at the currently installed version. |

Typical absolute paths:

```
/data/<user>/newpto/pypto          ← pypto repo root
/data/<user>/newpto/pypto/runtime  ← simpler (submodule)
/data/<user>/newpto/pypto-lib      ← this library
```

## Prerequisites

- The example script must have been passing before and is now failing precision.
- The user knows (or MEMORY.md records) a **known-good pypto commit**. If not
  available, ask the user. A known-good simpler commit is **not required** — it
  is derived automatically from pypto's submodule pointer.
- The example must be runnable via `python <script> -p <platform> -d <device>`.
- pypto repo is locally available.

## Workflow

### Phase 0: Gather Context

1. Ask the user which example script is failing (if not already clear).
2. Read `memory/MEMORY.md` for a known-good pypto commit. If none recorded, ask
   the user.
3. Determine repo paths:
   ```bash
   PYPTO_DIR=$(cd ../../pypto && pwd)
   SIMPLER_DIR=$PYPTO_DIR/runtime
   ```
4. Record the **current HEAD** of pypto as the known-bad commit:
   ```bash
   PYPTO_BAD=$(git -C $PYPTO_DIR rev-parse HEAD)
   ```
5. Determine the run command. Typical pattern:
   ```bash
   source run.sh  # or manually export env vars
   python models/qwen3/<script>.py -p a2a3 -d <device>
   ```

### Phase 1: Triage — Compiler vs Runtime (Optional Quick Check)

Compare the **generated code** from a known-good build against the current (bad)
build to get an early signal on whether the regression is in codegen or runtime.

#### Step 1a: Generate a good build output (if needed)

```bash
# Save current state
PYPTO_BAD=$(git -C $PYPTO_DIR rev-parse HEAD)

# Checkout known-good pypto commit (this also sets the submodule pointer)
git -C $PYPTO_DIR checkout <pypto_good_commit>
git -C $PYPTO_DIR submodule update --init runtime
pip install --no-build-isolation -e $PYPTO_DIR
pip install --no-build-isolation -e $SIMPLER_DIR

# Run the example to generate a good build_output
python <script> -p <platform> -d <device>
GOOD_BUILD=$(ls -td build_output/<ProgramName>_* | head -1)

# Restore to the bad commit
git -C $PYPTO_DIR checkout $PYPTO_BAD
git -C $PYPTO_DIR submodule update --init runtime
pip install --no-build-isolation -e $PYPTO_DIR
pip install --no-build-isolation -e $SIMPLER_DIR

# Run the example again to generate the bad build_output
python <script> -p <platform> -d <device>
BAD_BUILD=$(ls -td build_output/<ProgramName>_* | head -1)
```

#### Step 1b: Diff generated code

```bash
diff -rq $GOOD_BUILD/kernels/      $BAD_BUILD/kernels/
diff -rq $GOOD_BUILD/ptoas/        $BAD_BUILD/ptoas/
diff -rq $GOOD_BUILD/orchestration/ $BAD_BUILD/orchestration/
```

**Interpretation** (informational only — bisect always starts with pypto):

| Diff result | Meaning |
|---|---|
| **Files differ** in kernels/, ptoas/, or orchestration/ | Compiler (pypto) generated different code — likely a pypto code change |
| **No differences** (only timestamps / .o binaries) | Same generated code — likely a simpler (runtime) change via submodule bump |

Report the triage result to the user before proceeding. This does not change the
bisect strategy (always bisect pypto first), but helps set expectations.

### Phase 2: First-Level Bisect — pypto

#### Determine bisect parameters

```bash
GOOD_COMMIT=<pypto_known_good>
BAD_COMMIT=$PYPTO_BAD

# Count commits in range
git -C $PYPTO_DIR log --oneline $GOOD_COMMIT..$BAD_COMMIT | wc -l
```

Report the commit count to the user (~log2(N) bisect steps).

#### Run the bisect

```bash
cd $PYPTO_DIR
git bisect start
git bisect bad $BAD_COMMIT
git bisect good $GOOD_COMMIT
```

For each bisect step:

1. **Sync submodule and install:**
   ```bash
   git -C $PYPTO_DIR submodule update --init runtime
   pip install --no-build-isolation -e $PYPTO_DIR
   pip install --no-build-isolation -e $SIMPLER_DIR
   ```
   If install fails (e.g. incompatible API change), mark as skip:
   ```bash
   git bisect skip
   ```

2. **Run the example:**
   ```bash
   cd <pypto_lib_dir>
   python <script> -p <platform> -d <device>
   ```

3. **Check result:**
   - If output contains `PASS` → `cd $PYPTO_DIR && git bisect good`
   - If output contains `FAIL` or non-zero exit → `cd $PYPTO_DIR && git bisect bad`
   - If crash / unrelated error → `cd $PYPTO_DIR && git bisect skip`

4. **Report progress** to the user after each step:
   ```
   Step N/~M: <commit_short> — <commit_message> → good/bad/skip
   ```

5. Repeat until git bisect identifies the first bad commit.

#### Bisect complete

```bash
CULPRIT=$(git -C $PYPTO_DIR bisect view --oneline)
git -C $PYPTO_DIR bisect reset
```

#### Evaluate the culprit

Check whether the first bad commit is a **submodule bump** (i.e. it only changes
the `runtime/` submodule pointer):

```bash
git -C $PYPTO_DIR show <culprit> --stat
```

- If the commit touches **only** `runtime` (submodule pointer change), or the
  commit message matches patterns like `chore(runtime): bump simpler`, proceed
  to **Phase 2b** (second-level bisect in simpler).
- Otherwise, the regression is in **pypto itself** — skip to Phase 3.

### Phase 2b: Second-Level Bisect — simpler (only if needed)

This phase runs only when the first-level bisect identified a submodule bump
commit as the culprit. The regression is somewhere in the simpler commits between
the old and new submodule pointers.

#### Determine simpler bisect range

```bash
# The culprit pypto commit bumped runtime/ from OLD_PTR to NEW_PTR
# Get the submodule pointer BEFORE the culprit (the good simpler version)
SIMPLER_GOOD=$(git -C $PYPTO_DIR show <culprit>^:runtime | head -1)
# If the above doesn't work, use:
SIMPLER_GOOD=$(git -C $PYPTO_DIR rev-parse <culprit>^:runtime)

# Get the submodule pointer AT the culprit (the bad simpler version)
SIMPLER_BAD=$(git -C $PYPTO_DIR rev-parse <culprit>:runtime)

# Pin pypto at the culprit commit (the compiler code didn't change)
git -C $PYPTO_DIR checkout <culprit>
pip install --no-build-isolation -e $PYPTO_DIR

# Count simpler commits in range
git -C $SIMPLER_DIR log --oneline $SIMPLER_GOOD..$SIMPLER_BAD | wc -l
```

Report the simpler commit range to the user.

#### Run the simpler bisect

```bash
cd $SIMPLER_DIR
git bisect start
git bisect bad $SIMPLER_BAD
git bisect good $SIMPLER_GOOD
```

For each bisect step:

1. **Install:**
   ```bash
   pip install --no-build-isolation -e $SIMPLER_DIR
   ```

2. **Run the example:**
   ```bash
   cd <pypto_lib_dir>
   python <script> -p <platform> -d <device>
   ```

3. **Check result and mark** (same as first-level: PASS → good, FAIL → bad,
   crash → skip).

4. **Report progress** after each step.

5. Repeat until bisect identifies the first bad simpler commit.

#### Simpler bisect complete

```bash
SIMPLER_CULPRIT=$(git -C $SIMPLER_DIR bisect view --oneline)
git -C $SIMPLER_DIR bisect reset
```

### Phase 3: Report

Present to the user:

1. **Regression source**: compiler (pypto) or runtime (simpler).
2. **First bad commit**: hash, message, author, date.
   - If second-level bisect was performed, report both the pypto submodule bump
     commit and the specific simpler commit.
3. **What changed**: `git show <commit> --stat` summary.
4. **Diff highlights**: key changes that likely affect precision (look for dtype
   changes, cast operations, memory layout changes, optimization pass changes).
5. **Restore state**: confirm pypto is back to its original HEAD.

```bash
# Restore pypto to original HEAD (submodule follows automatically)
git -C $PYPTO_DIR checkout $PYPTO_BAD
git -C $PYPTO_DIR submodule update --init runtime
pip install --no-build-isolation -e $PYPTO_DIR
pip install --no-build-isolation -e $SIMPLER_DIR
```

## Important Notes

- **Always restore pypto** to its original HEAD after bisect, regardless of
  outcome. Use `git submodule update --init runtime` to sync simpler back.
- **Each pypto checkout changes the submodule pointer** — always run
  `git submodule update --init runtime` after each checkout to keep simpler in
  sync. This is the key simplification: no manual simpler pinning needed.
- **Each `pip install`** of pypto takes ~30s-2min (C++ recompilation). Simpler is
  similar. Factor this into time expectations.
- **Environment variables**: make sure PTOAS_ROOT, PTO_ISA_ROOT, ASCEND_HOME_PATH,
  and runtime pool sizes (PTO2_RING_*) are set. Check `run.sh` in the project root.
- **Build output comparison** ignores `.o` / `.so` files (binary artifacts) — only
  compare `.cpp` and `.pto` source files.
- If the example uses `--clear-cache`, include that flag during bisect to avoid
  stale cached data affecting results.
- If `pip install` fails on a specific commit (e.g., API breaking change), use
  `git bisect skip` — git will try adjacent commits.

## Asking the User

At any point, if information is missing:

| Missing info | What to ask |
|---|---|
| Known-good pypto commit | "What is the last pypto commit where precision was passing?" |
| Which build_output is good | "Which build_output timestamp was from a passing run?" |
| Platform / device | "Which platform and device ID should I use? (e.g., `-p a2a3 -d 5`)" |
| Script to run | "Which example script is failing? (e.g., `qwen3_14b_prefill.py`)" |
