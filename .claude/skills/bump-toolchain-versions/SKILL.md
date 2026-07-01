---
name: bump-toolchain-versions
description: Update the ptoas and/or pto-isa toolchain versions that CI depends on across the three repos (simpler, pypto, pypto-lib), then open the PRs. First confirms the target ptoas version with the user (latest release vs. a user-specified tag) rather than assuming latest. Use when the user wants to bump ptoas to a new PTOAS release, move pto-isa to a newer commit, or sync toolchain versions across simpler/pypto/pypto-lib.
---

# Bump CI Toolchain Versions (ptoas / pto-isa) across simpler, pypto, pypto-lib

SOP for advancing the two external toolchain versions the CI pipelines depend on:

- **`ptoas`** — the assembler binary on the codegen → device path
  (`github.com/hw-native-sys/PTOAS` releases).
- **`pto-isa`** — the ISA header repo (`github.com/hw-native-sys/pto-isa`),
  pinned by a 40-char commit SHA.

## Ownership model — who owns what, who inherits

There is exactly ONE source of truth per artifact. Everything else derives from
it. Never edit a derived value directly.

| Artifact | Source of truth (edit HERE) | Derives automatically |
| --- | --- | --- |
| **pto-isa** | `simpler/pto_isa.pin` (repo root, 40-hex SHA) | **pypto**: via the `runtime/` submodule pointer → `runtime/pto_isa.pin`. **pypto-lib**: CI clones pypto HEAD and reads `/tmp/pypto/runtime/pto_isa.pin`. |
| **ptoas** | `pypto/toolchain/versions.env` (`PTOAS_VERSION` + `PTOAS_SHA256_AARCH64` + `PTOAS_SHA256_X86_64`) | **pypto-lib**: CI clones pypto HEAD and reads `/tmp/pypto/toolchain/versions.env`. |

Consequences:

- **simpler does NOT use ptoas at all** — only pto-isa. Never touch ptoas in simpler.
- **pypto-lib needs NO code edit for either bump.** Its CI
  (`.github/workflows/ci.yml` + `daily_ci.yml`) does
  `git clone --recurse-submodules --depth=1 …/pypto.git /tmp/pypto` then reads
  `PTOAS_*` from `versions.env` and `PTO_ISA_COMMIT` from `runtime/pto_isa.pin`.
  It inherits both once pypto's `main` is updated. (Only act on pypto-lib if it
  still hard-codes `PTOAS_VERSION:`/`PTO_ISA_COMMIT:` inline in its workflow env —
  that is the OLD layout; if you see it, the fix is to migrate it to the
  derive-from-pypto block, which is a separate task — flag it, don't silently bump.)
- pypto's guard job (`toolchain`) fails the build if a literal PTOAS sha256 or
  pto-isa commit is hard-coded inline in any workflow. Editing `versions.env` /
  the submodule pointer is the ONLY sanctioned path.
- **⚠️ pypto's pto-isa CANNOT be bumped in isolation.** pypto has no independent
  pto-isa pin: the `toolchain` job derives `PTO_ISA_COMMIT` *only* from
  `runtime/pto_isa.pin` — i.e. whatever simpler commit the `runtime/` submodule
  points at — and the guard forbids pinning it anywhere else. That submodule
  points at a *whole* simpler commit, so the only lever to move pypto's pto-isa
  is to move the runtime pointer, which also moves the **runtime** by however
  many commits it currently lags simpler `main`. Unless that lag is a commit or
  two, "propagate pto-isa into pypto" is really a **runtime version bump** (often
  needing pypto code adaptation) — the pypto team's routine `chore(runtime): bump
  runtime submodule to simpler main …` work, NOT a mechanical part of a toolchain
  version bump. The pto-isa version bump proper is the **simpler pin PR** (Step 1);
  pypto picks the new pin up on the team's next runtime bump. See Step 2b.

### Propagation order (when doing a full pto-isa bump end to end)

1. **simpler PR** (bump `pto_isa.pin`) — this IS the pto-isa version bump. It
   merges to `simpler/main` and is the source of truth. For the pto-isa side your
   job ends here (a ptoas PR below is independent).
2. **pypto** picks up the new pto-isa only when its `runtime/` submodule is
   bumped, and that drags the full runtime lag (see the ⚠️ consequence above), so
   it is the pypto team's routine runtime bump — NOT a step you perform blindly.
   Do it here (Step 2b) ONLY if the runtime lag is trivial *and* the user
   explicitly wants pto-isa in pypto now; otherwise flag it and stop at the
   simpler PR. If you are *also* bumping ptoas, that is a separate `versions.env`
   PR (Step 2a) and does not depend on any of this.
3. **pypto-lib** inherits both automatically on its next CI run — no PR.

A ptoas-only bump touches **only** pypto (`versions.env`); simpler is irrelevant.
A pto-isa-only bump is a **simpler pin PR** (Step 1); its arrival in pypto is a
separate runtime-bump decision (Step 2b), not an automatic or mechanical step.

## Prerequisites

- `gh` authenticated (`gh auth status`). Direct push to `hw-native-sys/*` is
  usually NOT available (triage only) — use the fork-and-PR flow below.
- **Resolve your fork owner** (your authenticated GitHub login). Every command
  below refers to it as `$GH_USER`:

  ```bash
  GH_USER="$(gh api user --jq .login)"
  echo "fork owner: $GH_USER"
  ```

  Forks live at `$GH_USER/simpler` and `$GH_USER/pypto`. Create either on demand
  (no-op if it already exists):

  ```bash
  gh repo fork hw-native-sys/simpler --clone=false    # and/or hw-native-sys/pypto
  ```

- **Resolve your commit identity** — do NOT hard-code a name or email. Read it
  from git config; if unset, ask the user to confirm the values to commit under:

  ```bash
  GIT_NAME="$(git config user.name)"
  GIT_EMAIL="$(git config user.email)"
  # If either is empty, confirm with the user, then set explicitly, e.g.:
  #   GIT_NAME="<confirmed name>"; GIT_EMAIL="<confirmed email>"
  [ -n "$GIT_NAME" ] && [ -n "$GIT_EMAIL" ] || echo "identity unset — confirm with user"
  ```

  Every commit below passes these via
  `git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL"`, so the SOP works even
  when no global identity is set.
- **Commit messages and PR descriptions must NOT contain any AI co-author or
  "Generated with/by" line** (`Co-Authored-By`, `🤖 Generated with …`, etc.).
  pypto's own rules also forbid this. Always grep-verify before pushing.

## Step 0 — Gather the new values (do this first, verify before editing)

### pto-isa latest commit

```bash
git ls-remote https://github.com/hw-native-sys/pto-isa.git HEAD refs/heads/main
# take the 40-hex SHA (HEAD == main)
```

### Confirm the target ptoas version WITH THE USER (do NOT assume "latest")

Before gathering hashes, decide which ptoas version to pin. **Always confirm with
the user first — do not silently pick the latest release.** Unless the user's
request already named an explicit version (e.g. "bump ptoas to v0.46"), which
counts as the answer, ask.

1. Show what is available and what is currently pinned:

   ```bash
   CUR="$(grep -E '^PTOAS_VERSION=' <pypto>/toolchain/versions.env | cut -d= -f2)"
   echo "currently pinned: $CUR"
   gh release list --repo hw-native-sys/PTOAS --limit 10   # newest first; note the top tag = latest
   ```

2. Ask the user to choose (use `AskUserQuestion`), offering:
   - **Latest** — the top tag from `gh release list` (state the concrete tag, e.g.
     "Use latest (v0.47)"). Recommended default.
   - **Specify a version** — let the user type an exact tag (e.g. `v0.46`). Accept
     it as `VER`, then validate it exists before proceeding (next step's
     `gh release view` errors on a bad tag — surface that to the user, don't guess).
   - (If the named/typed version equals `$CUR`, there is nothing to bump — tell the
     user and stop.)

3. Set `VER` to the chosen tag and carry it through the rest of Step 0 and Step 2.

### ptoas version + sha256 (BOTH arches)

The release ships `ptoas-bin-aarch64.tar.gz` and `ptoas-bin-x86_64.tar.gz`.
`versions.env` records the sha256 of each archive. With the user-confirmed `VER`,
download and compute:

```bash
cd /tmp && rm -rf ptoas_dl && mkdir ptoas_dl && cd ptoas_dl
VER=<user-confirmed tag, e.g. v0.46>   # from the confirmation step above — NOT hard-coded
gh release view "$VER" --repo hw-native-sys/PTOAS   # confirm it exists + lists both bin assets
gh release download "$VER" --repo hw-native-sys/PTOAS --pattern "ptoas-bin-aarch64.tar.gz"
gh release download "$VER" --repo hw-native-sys/PTOAS --pattern "ptoas-bin-x86_64.tar.gz"
sha256sum ptoas-bin-aarch64.tar.gz ptoas-bin-x86_64.tar.gz
```

**Self-check the method:** also download the CURRENT pinned version's aarch64
archive and confirm its recomputed sha256 equals the value already in
`versions.env`. If it matches, your download+hash method is sound and the new
hashes are trustworthy. (Done this way, a bad mirror or truncated download is
caught before it reaches a PR.) Clean up `/tmp/ptoas_dl` after.

## Step 1 — simpler PR: bump `pto_isa.pin` (pto-isa bumps only)

```bash
cd <simpler>
git remote add fork "https://github.com/$GH_USER/simpler.git" 2>/dev/null; git fetch fork
git checkout -b ci/bump-pto-isa-<short-sha> origin/main
printf '%s\n' "<new-40-hex-sha>" > pto_isa.pin      # single line + trailing newline
git add pto_isa.pin
git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
    commit -m "ci: bump pinned pto-isa to <short-sha>"   # body: old→new dates, fast-forward note
git --no-pager log -1 --format='%B' | grep -iE "co-authored|generated (with|by)|🤖" \
    && echo FORBIDDEN || echo CLEAN
git push -u fork ci/bump-pto-isa-<short-sha>
gh pr create --repo hw-native-sys/simpler --base main \
    --head "$GH_USER:ci/bump-pto-isa-<short-sha>" \
    --title "ci: bump pinned pto-isa to <short-sha>" --body "<why/what/impact>"
```

Good PR body facts to include (verify each):
- old→new commit + dates (`git log -1 --format='%h %ci' <sha>` in a bare clone).
- Confirm fast-forward: `git merge-base --is-ancestor <old> <new>` → only-forward.
- One-line summary of the ISA fixes in the range (`git log --oneline old..new`).
- Downstream: pypto derives pto-isa from this pin via its `runtime/` submodule,
  so it picks the new commit up whenever that submodule is next bumped — a runtime
  bump, NOT automatic on merge here (see the ⚠️ consequence + Step 2b). pypto-lib
  then inherits from pypto's `main`.

## Step 2 — pypto PR

### 2a. ptoas bump → edit `toolchain/versions.env`

```bash
cd <pypto>
git remote add fork "https://github.com/$GH_USER/pypto.git" 2>/dev/null; git fetch fork
git checkout -b ci/bump-ptoas-<VER> origin/main
# edit toolchain/versions.env: PTOAS_VERSION + both PTOAS_SHA256_* (values from Step 0)
git add toolchain/versions.env
git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
    commit -m "ci: bump ptoas to <VER>"     # body: note runtime doesn't use ptoas; pto-isa unaffected
git --no-pager log -1 --format='%B' | grep -iE "co-authored|generated (with|by)|🤖" \
    && echo FORBIDDEN || echo CLEAN
git push -u fork ci/bump-ptoas-<VER>
gh pr create --repo hw-native-sys/pypto --base main \
    --head "$GH_USER:ci/bump-ptoas-<VER>" \
    --title "ci: bump ptoas to <VER>" --body "<why/what/impact>"
```

### 2b. pto-isa → the `runtime/` submodule pointer (usually NOT your step)

**Read the ⚠️ consequence in the ownership model first.** pypto's pto-isa moves
only by moving the `runtime/` submodule, which points at a whole simpler commit —
so this is a **runtime bump**, not a clean pto-isa bump. Gate on the lag before
doing anything:

```bash
cd <pypto>
CUR=$(git ls-tree origin/main runtime | awk '{print $3}')          # current runtime pointer
HEAD=$(cd <simpler> && git rev-parse origin/main)                  # simpler main HEAD
git -C <simpler> rev-list --count "$CUR".."$HEAD"                  # how many runtime commits it would drag
```

- **Lag is more than a commit or two → STOP.** This is the pypto team's
  `chore(runtime): bump runtime submodule to simpler main …` work — it may need
  code adaptation a toolchain bump has no business making. The new pto-isa is
  already the source of truth on `simpler/main` (Step 1) and reaches pypto on the
  team's next runtime bump. Tell the user; do **not** open this PR.
- **Lag is trivial AND the user explicitly wants pto-isa in pypto now** → proceed
  as its OWN PR (never share the ptoas PR — different scope/risk), targeting
  simpler `main` HEAD to match the team's convention:

```bash
cd <pypto>
git checkout -b chore/bump-runtime-simpler-<short-sha> origin/main
# Move only the gitlink — no full submodule clone needed:
git update-index --cacheinfo 160000,<simpler-main-HEAD-40hex>,runtime
git --no-pager diff --stat origin/main            # must show ONLY `runtime | 2 +-`
git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
    commit -m "chore(runtime): bump runtime submodule to simpler main <short-sha>"  # body: pin old→new + runtime delta
git --no-pager log -1 --format='%B' | grep -iE "co-authored|generated (with|by)|🤖" \
    && echo FORBIDDEN || echo CLEAN
git push -u fork chore/bump-runtime-simpler-<short-sha>
```

The `toolchain` job re-reads `runtime/pto_isa.pin` from the new submodule content
and moves pypto's pto-isa in lockstep — **along with the whole runtime delta**,
which the PR body must name and CI (build + system-tests) must gate.

## Step 3 — pypto-lib: verify inheritance (usually no PR)

No edit needed with the current CI. After pypto's `main` updates, pypto-lib's
next CI run clones pypto HEAD and picks up both values. To confirm the current
layout still derives (not hard-codes):

```bash
gh api repos/hw-native-sys/pypto-lib/contents/.github/workflows/ci.yml --jq '.content' \
  | base64 -d | grep -nE "versions.env|runtime/pto_isa.pin|clone .*pypto.git"
```

If instead you see literal `PTOAS_VERSION:` / `PTO_ISA_COMMIT:` under a job's
`env:`, the workflow is on the old hard-coded layout — tell the user; migrating
it to the derive-from-pypto block is a separate change, not part of a routine bump.

## CI gotchas (observed — don't misdiagnose these as your change breaking things)

### 1. First run on a NEW ptoas version: shared-cache `.tmp` race

The onboard jobs (`system-tests`, `dist-system-tests`, `pypto-lib-model`) mount
one shared host dir as the ptoas cache and download to an identical
`…-<VER>-<sha>.tar.gz.tmp`. On the FIRST run after a version bump the cache is
cold, so parallel jobs race the same `.tmp`:

- `sha256sum: … .tmp: No such file or directory` (a sibling `mv`'d it away), or
- `sha256sum: WARNING: 1 computed checksum did NOT match … FAILED` (interleaved writes).

This is **infrastructure, not your sha256**. Fix: **re-run once the cache is warm**
— the first job to finish populates the cache, later runs hit "Cache hit". A
clean second run confirms it. (Root cause is the workflow using a non-unique
`.tmp` name with no lock — a separate CI fix, not this PR's job.)

### 2. Onboard flakes (507018-style abrupt exit)

A `system-tests` run can die with `Process completed with exit code 1`
immediately after a test name prints (no assertion, no pytest summary) — an
onboard device timeout/reap, often intermittent. Re-run; if it reproduces at the
**same** test across runs, THEN pull the device log and investigate a real
regression (see simpler's `running-onboard.md` 507018 triage table).

### Re-triggering CI without admin rights

`gh run rerun` needs admin on `hw-native-sys/*` (you have triage). Instead push a
fresh SHA to the fork branch — content-preserving amend then force-push:

```bash
cd <repo> && git checkout <pr-branch>
git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" \
    commit --amend --no-edit --date=now        # new SHA, same tree/message/author
git --no-pager log -1 --format='%B' | grep -iE "co-authored|generated (with|by)|🤖" \
    && echo FORBIDDEN || echo CLEAN
git push --force-with-lease fork <pr-branch>
```

Verify content is unchanged first (`git diff origin/main --stat` shows only the
intended file). This starts a brand-new run on the PR.

## Verify / monitor

```bash
gh pr checks <PR#> --repo hw-native-sys/<repo>
# robust per-job terminal-state read (gh pr checks can momentarily return empty):
gh run view <RUN_ID> --repo hw-native-sys/<repo> --json jobs \
  --jq '.jobs[] | select(.name|test("system-tests|dist-system-tests")) | "\(.name): \(.status)/\(.conclusion)"'
```

- `toolchain` job green ⇒ the ptoas sha256 + pto-isa commit resolved and the
  no-hard-coded-literal guard passed. This is the direct signal your `versions.env`
  edit is well-formed.
- Onboard jobs stay `queued` until a self-hosted NPU device frees up — queuing is
  normal, not a failure.

## Checklist

- [ ] ptoas target version confirmed WITH THE USER (latest vs. a user-specified tag) — not assumed.
- [ ] Step 0 values gathered; method self-checked against the current pinned version.
- [ ] simpler: only `pto_isa.pin` changed (pto-isa bumps); fast-forward confirmed. **This is the whole pto-isa version bump.**
- [ ] pypto: `versions.env` both sha256 updated (ptoas). Runtime-submodule move (pto-isa) is a *runtime* bump — the lag was checked, and it was done ONLY if trivial + user wants it now, else flagged to the user and left to the pypto team (Step 2b).
- [ ] pypto-lib: confirmed derive-from-pypto (no PR) — or flagged if hard-coded.
- [ ] No `Co-Authored-By` / "Generated with/by" in any commit or PR body (grep-verified).
- [ ] `toolchain` job green on each PR; onboard failures triaged (cache race / flake ⇒ re-run).
