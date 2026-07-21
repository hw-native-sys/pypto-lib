---
name: create-issue
description: Reproduce a reported problem on the current environment, collect the pin-bound dependency versions, write a Background section, then route the issue to the repo that owns the fault (pypto / simpler / ptoas / pto-isa) or file it in pypto-lib when the boundary is unclear. Use when the user wants to file a bug, request a feature, or create any GitHub issue.
---

# Create GitHub Issue

Every issue this skill files **opens with a Background section** that states the
pypto-lib–side problem and the exact reproduction environment. From that
Background you decide where the issue belongs:

- **Boundary clear** → the fault sits in one downstream repo (`pypto`,
  `simpler`, `ptoas`, or `pto-isa`). File it **there**, driving that repo's own
  `create-issue` skill (or `gh` directly if it has none), with our Background at
  the top of the body.
- **Boundary unclear** → the fault is in pypto-lib's own kernel/harness code, or
  it cannot be isolated to a single downstream repo. File it **in pypto-lib**
  using this repo's `.github/ISSUE_TEMPLATE/`.

**An issue must be reproducible by someone who only has the public repos.** Many
pypto-lib cases are local scratch files, `_draft.py`, or uncommitted edits — a
bare path to one of those tells the reader nothing. Whenever the repro file is
not on a branch they can fetch, the issue carries its **content** (inlined, or
attached via the web UI). See Steps 3c–3d.

```text
reproduce on CURRENT env → make repro self-contained → collect versions
                             → diagnose → check pin consistency
                                                              │
                                              match ──────────┤──────── mismatch
                                                              │            │
                                                              │      ask: align & re-run,
                                                              │      or file as-is (note it)
                                                              ▼
                                            write Background → route by diagnosis
                                                              │
                        ┌─────────────────────────────────────┼─────────────────────────┐
                     clear                                  clear                      unclear
                  (pypto/simpler)                       (ptoas/pto-isa)             (pypto-lib
                        │                                    │                    own code / mixed)
             delegate to that repo's                file directly via gh        file here via
             create-issue skill                     (no local skill)            ISSUE_TEMPLATE
```

Reproduce on the environment the user **actually hit the bug on** — do **not**
reset pypto to latest `main` first, or the Background would describe a different
environment than the one that reproduces. We only offer to change versions in
Step 4, when they turn out to be off their pins.

Feature requests and documentation issues have no reproduction/environment and no
downstream owner — they always go to **pypto-lib**. For those, skip Steps 2–6 and
jump to Step 7.

## Repo map

| Component | GitHub repo | Local root | Owns |
|---|---|---|---|
| pypto-lib | `hw-native-sys/pypto-lib` | this repo | kernels, models, golden harness |
| pypto | `hw-native-sys/pypto` | `$PYPTO_ROOT` | IR generation, lowering, codegen |
| simpler | `hw-native-sys/simpler` | `$PYPTO_ROOT/runtime` | on-device / sim execution, task-graph build/execute (AICPU + AICore) |
| ptoas | `hw-native-sys/PTOAS` | `$PTOAS_ROOT` (binary) / `../PTOAS` (source) | PTO bytecode assembly & optimization |
| pto-isa | `hw-native-sys/pto-isa` | `$PTO_ISA_ROOT` | virtual tile-ISA implementations |

`simpler` is the pypto **runtime** submodule (submodule name `simpler`, path
`$PYPTO_ROOT/runtime`). It tracks `hw-native-sys/simpler`, and it is a
**distinct** issue owner from pypto — a runtime crash / hang / AICPU error is a
**simpler** issue, not a pypto one.

## Step 1: Authenticate

```bash
gh auth status
```

If not authenticated, tell the user to run `gh auth login` and **stop**.

## Step 2: Ensure toolchain present & collect current versions

**Bug reports only.** (Feature/documentation → skip to Step 7.)

Use the `/setup_env` skill only to **install anything missing** (pypto not
importable, ptoas absent, pto-isa not checked out). **Do not** `git checkout main`
/ `git pull` / `git submodule update` on pypto — that would overwrite the very
environment we are trying to reproduce. Whatever pypto commit/branch the user is
on is the anchor; we validate the rest against it in Step 4.

Collect the five component versions of the **current** environment (short hashes
for display in the Background table):

```bash
# pypto-lib
git rev-parse --short HEAD
git branch --show-current

# pypto (the anchor)
git -C "$PYPTO_ROOT" rev-parse --short HEAD
git -C "$PYPTO_ROOT" branch --show-current

# simpler (== pypto/runtime submodule; usually detached HEAD)
git -C "$PYPTO_ROOT/runtime" rev-parse --short HEAD
git -C "$PYPTO_ROOT/runtime" branch --show-current   # often empty (detached)

# pto-isa (usually detached HEAD)
git -C "$PTO_ISA_ROOT" rev-parse --short HEAD
git -C "$PTO_ISA_ROOT" branch --show-current         # often empty (detached)

# ptoas version
"$PTOAS_ROOT/ptoas" --version 2>/dev/null || echo "not found"

# CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || echo "not detected"
```

Submodule / pinned checkouts (simpler, pto-isa) are typically in **detached
HEAD**, so `branch --show-current` prints nothing — in that case record the
branch as `detached` rather than leaving it blank.

Record every value. If a version cannot be detected, use "unknown" and continue.
If `/setup_env` cannot even make pypto importable, note the failure, **skip
Steps 3–4**, and treat the boundary as **unclear** (file in pypto-lib).

## Step 3: Reproduce (on the current environment)

Confirm the problem before filing. **Bug reports only.**

### 3a: Identify the reproduction script

- If the user gave a specific script/command, use it.
- If the issue names a file (e.g. "`decode_attention_csa.py` FAILs x_out"), use
  that file with the platform/device it fails on (`-p a2a3`, `-d <id>`).
- If unclear, **ask the user** which script reproduces the problem.

### 3b: Run it

```bash
python3 <reproduction_script> -p <platform> -d <device>
```

Capture stdout + stderr, the failing platform, and the **symptom class**:

| Symptom class | Looks like |
|---|---|
| precision | wrong output vs torch golden (rel-L2 / cosine / % error), a FAIL at some element |
| performance | latency/throughput worse than target (e.g. 250us vs 50us) |
| hang | device/AICPU never returns, watchdog/timeout, task-ring deadlock |
| compile | pypto traceback, IR validation, codegen/g++ error, ptoas assembly error |
| runtime | 507xxx / device error / crash during execution |

If the script **succeeds**, tell the user "The issue does not reproduce in the
current environment" and ask whether to still file. If no → **stop**.

### 3c: Check whether the repro script is visible to the reader

Many pypto-lib repro cases are **local-only** — a scratch kernel, a `_draft.py`,
or an edit that has not been committed and pushed. A path like
`models/qwen3/14b/bench_qk_compare.py` is **meaningless to whoever reads the
issue** if that file does not exist on any branch they can fetch. Determine the
visibility of every file the repro needs (the script plus any local helper
modules / kernels it imports):

Build the list of required files first — the entry script **plus every local
module / kernel source it imports or references** (`grep` its imports, its
`aic_source` / `aiv_source` / `include_dirs`, and any data file it opens). Every
file on that list is checked, and every file on that list must end up visible;
one invisible helper is enough to make the issue unreproducible.

```bash
# tracked at all?
git ls-files --error-unmatch <file> 2>/dev/null || echo "UNTRACKED"

# tracked but with uncommitted edits?
git diff --stat HEAD -- <file>

# committed — but does that commit actually exist on the remote?
# `git branch -r --contains` only reads LOCAL remote-tracking refs, which can be
# stale; ask the remote itself, and confirm the branch still contains the commit.
git ls-remote origin refs/heads/<branch>            # empty => branch not on remote
git merge-base --is-ancestor HEAD origin/<branch> \
  && echo "HEAD is on origin/<branch>" || echo "HEAD NOT PUSHED"
```

Also confirm the remote is one the **reader** can reach: a commit pushed only to
your personal fork, or to a private repo the issue's audience has no access to,
is no more visible than an untracked file. When in doubt, treat it as invisible.

| Result | Reader can see it? | Action |
|---|---|---|
| Tracked, clean, and the commit is fetchable from a remote the reader can access | yes | cite the path + commit; no need to inline |
| Untracked / `_draft.py` / uncommitted edits / unpushed commit / fork-only or private remote | **no** | the issue **must carry the source** — see 3d |

Never file an issue whose only pointer to the repro is a local path.

### 3d: Make a local-only repro self-contained

When 3c says the reader cannot see the file, do **one** of the following —
inlining is the default; attach only when inlining is impossible:

1. **Minimize, then inline.** Cut the case down to the smallest script that
   still reproduces (strip unrelated layers/benches/variants, prefer `-p
   <sim>` when the symptom survives there), re-run it to confirm it still
   reproduces, and paste the **full source** into the issue body inside a
   collapsed block:

   ````markdown
   <details>
   <summary>Reproduction case — <code>repro_qk_mmad.py</code> (not committed; paste locally to run)</summary>

   ```python
   <full file content>
   ```

   </details>
   ````

   Repeat one `<details>` block per local-only file the case needs, and say in
   the body where each file is expected to sit in the tree (imports depend on
   it), e.g. "save as `models/qwen3/14b/repro_qk_mmad.py`".

2. **Attach the file** — only when it is genuinely too large to inline
   (roughly >500 lines, a binary, or a build/log artifact). `gh issue create`
   **cannot upload attachments**, so this requires the GitHub web UI: create
   the issue first, then drag-and-drop the file onto the issue page, and leave
   a placeholder line in the body (`Reproduction case attached below.`) so it
   is obvious the attachment is expected. Tell the user explicitly that they
   must do this drag-and-drop step.

Large supporting artifacts (full compile logs, generated IR / `.ptobc`,
`build_output/` dumps, device logs) follow the same rule: inline the relevant
excerpt in the body, and attach the full file via the web UI when it matters.

Cover **every** file on the Step 3c list — each one gets a citation, an inline
block, or a stated attachment. A case whose entry script is inlined but whose
imported helper is only named by path is still unreproducible.

#### Sanitize before publishing

Inlining or attaching turns local content into a **public, permanently indexed**
artifact. Before it goes in the body, scan every file you are about to publish
and redact:

- credentials & secrets — tokens, keys, passwords, `PRIVATE_KEY`, `.netrc` /
  `~/.ssh` contents, anything read from a secrets env var;
- internal-only identifiers — intranet hostnames / URLs, internal IPs, ticket or
  device-farm IDs, unreleased product or customer names;
- personal information — usernames, emails, and absolute paths that embed them
  (`/data/<user>/…`, `/home/<user>/…`). This is also project rule 4 in
  `.claude/CLAUDE.md`: **no private information in code or docs**;
- proprietary third-party code — vendor sources you are not licensed to
  redistribute. Reference these by upstream path/version instead of pasting them.

```bash
# quick sweep over the files about to be published
grep -nEi 'password|passwd|secret|token|api[_-]?key|private[_-]?key|BEGIN [A-Z ]*PRIVATE KEY' <files>
grep -nE '/home/[^/ ]+|/data/[^/ ]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' <files>
```

Replace hits with placeholders (`<user>`, `<internal-host>`, `<token>`) and keep
the repro runnable. If a file cannot be published without losing something that
must stay private, **do not publish it** — write a minimal clean-room repro that
exercises the same code path instead, and say so in the issue.

**Show the user the exact content to be published and get explicit approval
before it is sent.** The Step 8 preview is that checkpoint — publishing source
is not covered by a general "go ahead and file the issue".

### 3e: Diagnose the faulty component

Map the error to the owning repo. **In pypto-lib the majority of downstream
faults are pypto (compile/codegen) or simpler (runtime/hang).**

| Stage | Component | Error signals |
|---|---|---|
| IR generation, lowering, codegen | **pypto** | Python traceback in pypto modules, IR validation errors, codegen/g++ failures, wrong output produced at compile time, orch SSA/mapping errors |
| On-device / sim execution, task-graph build/execute (AICPU + AICore) | **simpler** | 507xxx AICPU errors, stream-sync failures, task-ring / dep-pool deadlock, runtime hangs, device sync/watchdog errors |
| PTO assembly & optimization | **ptoas** | ptoas error messages, assembly syntax errors, optimizer miscompiles traced to a ptoas version |
| Tile-ISA implementation | **pto-isa** | errors originating in pto-isa sources, a missing/unimplemented tile op, wrong results traced to a specific tile-ISA kernel |
| pypto-lib's own kernel/model/harness code | **pypto-lib** (unclear boundary) | the bug is in the kernel authoring under `models/`/`examples/`, the golden harness, or cannot be pinned to one downstream repo |

**State the diagnosis** to the user: "This appears to be a **<component>** issue
because …". If you cannot isolate a single downstream repo, say so — the boundary
is **unclear** and the issue will be filed in pypto-lib.

## Step 4: Check version consistency against the pins

The four toolchain versions are **bound**, with the current pypto checkout as the
anchor:

- pypto pins **simpler** via its `runtime` submodule gitlink
- pypto pins **ptoas** via `toolchain/versions.env` (`PTOAS_VERSION`)
- simpler (the `runtime` submodule) pins **pto-isa** via `runtime/pto_isa.pin`

Compute each **expected** value from the current pypto checkout and compare it to
what is actually installed. Compare **like-for-like**: full hashes vs full
hashes, and normalize the ptoas string (`--version` prints `ptoas 0.48`, the pin
is `v0.48`).

```bash
# simpler: expected = pypto's recorded submodule commit (full), actual = runtime HEAD (full)
git -C "$PYPTO_ROOT" rev-parse HEAD:runtime
git -C "$PYPTO_ROOT/runtime" rev-parse HEAD

# pto-isa: pin is a FULL 40-char hash; compare against the full actual HEAD
tr -d '[:space:]' < "$PYPTO_ROOT/runtime/pto_isa.pin"
git -C "$PTO_ISA_ROOT" rev-parse HEAD

# ptoas: normalize both sides to a bare version, then compare
( . "$PYPTO_ROOT/toolchain/versions.env" && echo "${PTOAS_VERSION#v}" )   # e.g. 0.48
"$PTOAS_ROOT/ptoas" --version 2>/dev/null | sed 's/^ptoas //'             # e.g. 0.48
```

If every component matches its pin, proceed to Step 5.

**If any component is off its pinned version** (a stale ptoas binary, a pto-isa
checkout at a different commit, or a locally-bumped simpler), the reproduction
environment is **mismatched** — the bug may be an artifact of the mismatch rather
than a real defect. **Ask the user** (via `AskUserQuestion`) whether to:

1. **Align to the pins and re-run** (recommended) — check out the expected
   simpler / pto-isa commits, reinstall the pinned ptoas via `/setup_env`, then
   repeat Step 3 on the consistent environment. Use the **aligned** versions in
   the Background; **or**
2. **File as-is** — keep the mismatched versions, and note the mismatch
   explicitly in the Background (list expected vs actual for each off component).

Do not silently proceed on a mismatch — either align the versions or record the
divergence in the issue.

## Step 5: Write the Background section

Build the Background block now — it leads **every** bug issue body, whichever repo
it lands in:

````markdown
## Background

In pypto-lib, `<file path>` on `<platform>` has a <precision | performance | hang
| compile | runtime> problem: <one-line symptom, e.g. "x_out ~27% rel-L2 vs torch
golden", "decode latency 250us vs 50us target", "AICPU 507018 timeout / device
hang">.

Reproduce with:

    python <reproduction_script> -p <platform> -d <device>

<If the repro file is committed AND pushed:>
Repro case: `<path>` at pypto-lib `<commit>` (branch `<pushed branch>`).

<If the repro file is local-only (Step 3c said the reader cannot see it) —
inline it here — one block per file on the Step 3c list, sanitized per 3d:>
The repro case is **not committed** — save the files below at the paths shown to
run it. Files required: `<path 1>`, `<path 2>`, …

<details>
<summary>Reproduction case — <code>&lt;filename&gt;</code> (save as <code>&lt;path&gt;</code>)</summary>

```python
<full file content>
```

</details>

<details>
<summary>Helper — <code>&lt;filename&gt;</code> (save as <code>&lt;path&gt;</code>)</summary>

```cpp
<full file content>
```

</details>

Reproduction environment:

| Component | Version |
|---|---|
| pypto-lib | `<commit>` (branch: `<branch>`) |
| pypto | `<commit>` (branch: `<branch>`) |
| simpler | `<commit>` (branch: `<branch or detached>`) |
| ptoas | `<version>` |
| pto-isa | `<commit>` |
| CANN | `<version or "not detected">` |

<If Step 4 found a mismatch and the user chose to file as-is, add here:>
Version mismatch (off pinned): <component> expected `<pin>`, actual `<installed>`.

Diagnosis: **<pypto | simpler | ptoas | pto-isa | unclear>** — <brief reason>.
````

## Step 6: Route to the target repo

Pick the destination from the Step 3e diagnosis. This step only **assembles the
body and picks the target** — the actual `gh issue create` happens once, in
Step 9, after the Step 8 preview.

| Diagnosis | Target repo | How to file |
|---|---|---|
| pypto | `hw-native-sys/pypto` | **delegate**: `$PYPTO_ROOT/.claude/skills/create-issue/SKILL.md` |
| simpler | `hw-native-sys/simpler` | **delegate**: `$PYPTO_ROOT/runtime/.claude/skills/create-issue/SKILL.md` |
| ptoas | `hw-native-sys/PTOAS` | file directly via `gh` (no local skill) |
| pto-isa | `hw-native-sys/pto-isa` | file directly via `gh` (no local skill) |
| unclear / pypto-lib own code | `hw-native-sys/pypto-lib` | stay here → Step 7 |

### 6a: Dedup (run first, before assembling the final body)

**Launch a `general-purpose` agent** (via `Task`, **model: haiku**) to check for
duplicates. Keep the main context clean. Give the agent the issue
summary/keywords, the diagnosed component, and the target repo, plus these exact
instructions:

> **IMPORTANT: ONLY use `gh` CLI with an explicit `--repo OWNER/REPO` flag. Do
> NOT read source code or explore the repo. Your sole job is to check GitHub
> issues for duplicates.**
>
> Search **both** `hw-native-sys/pypto-lib` **and** the diagnosed target repo
> (`hw-native-sys/pypto`, `hw-native-sys/simpler`, `hw-native-sys/PTOAS`, or
> `hw-native-sys/pto-isa`). For each repo:
>
> Step A — scan open titles:
> ```bash
> gh issue list --repo OWNER/REPO --state open --limit 500 --json number,title,labels \
>   --jq '.[] | "\(.number)\t\(.title)\t\(.labels | map(.name) | join(","))"'
> ```
> Step B — deep-read up to 3 candidates: `gh issue view NUMBER --repo OWNER/REPO`
> (body only, skip `--comments` unless ambiguous).
>
> Return EXACTLY one of `DUPLICATE REPO#N`, `RELATED REPO#N1 REPO#N2 ...`, or
> `NO_MATCH`, plus 1–3 sentences.

Act on the verdict:

- `DUPLICATE REPO#N` → do **not** create; tell the user the existing issue and
  repo. **Stop.**
- `RELATED REPO#N1 ...` → proceed; add `Related: REPO#N1, REPO#N2` to the body.
- `NO_MATCH` → proceed.

### 6b: Delegate to a downstream repo's create-issue skill (pypto / simpler)

When the target repo ships its own `create-issue` skill:

1. **Read** the target `SKILL.md` (path in the routing table) and follow **its**
   flow — its templates, required fields, and any project-board steps are
   authoritative for that repo. (Dedup is already done in 6a — don't repeat it.)
2. **Do not re-run `/setup_env` or re-collect versions.** Reuse the Background
   from Step 5 verbatim.
3. The **Background section must be the first thing in the issue body**, above
   whatever the target template's first field is (fold the target's
   "Description"/"Environment" content in after it; don't duplicate the
   environment table).
4. Assemble the target-repo body and the `gh issue create --repo <target>`
   invocation, but **do not run it yet** — hand off to Step 8 (preview) → Step 9.

If the target skill cannot be read (repo not checked out locally), fall back to
6c.

### 6c: Assemble a direct body for a repo without a skill (ptoas / pto-isa)

These repos have no `create-issue` skill. Fetch their template if present, else
use a plain body. Always lead with the Background section:

```markdown
## Background
<Background block from Step 5>

### Details / Repro steps
<full error output, stack trace, and how it was traced to this component>
```

Target repo: `hw-native-sys/PTOAS` or `hw-native-sys/pto-isa`, label `bug`. Hand
off to Step 8 → Step 9 to create.

## Step 7: File in pypto-lib (feature/docs, or unclear-boundary bug)

Used for feature requests, documentation issues, and any bug whose boundary is
unclear or which lives in pypto-lib's own code.

### 7a: Dedup

Run the Step 6a dedup agent, but search **only** `hw-native-sys/pypto-lib`.

### 7b: Classify

Read `.github/ISSUE_TEMPLATE/` and match the request:

| Template | Use when | Labels |
|---|---|---|
| `bug_report.yml` | compile/codegen/runtime error, incorrect output, hang | `bug` |
| `feature_request.yml` | new tensor function, new example/model, API improvement | `enhancement` |
| `documentation.yml` | missing / incorrect / unclear docs | `documentation` |

If ambiguous, clarify with `AskUserQuestion`.

### 7c: Fill required fields

Fill every field marked `required: true`. Ask the user for anything you cannot
infer (use `AskUserQuestion` for dropdowns). Auto-fill where possible:

- **Title prefix**: `[Bug]` / `[Feature]` / `[Docs]`.
- **Host Platform**: `uname -s -m` → `Linux (aarch64)` / `Linux (x86_64)` /
  `macOS (arm64)` / `Other`.
- **Environment**: reuse the Step 2 versions (bug reports).

### 7d: Body format

`gh issue create` takes a markdown body. For a **bug report**, lead with the
Background section from Step 5, then the template fields:

```markdown
## Background
<Background block from Step 5 — includes the Environment table and Diagnosis>

### Description
<clear description of the bug and how to reproduce it>

### Host Platform
`<os> <arch>`

### Additional Context
<anything else; Related: pypto-lib#N if applicable>
```

For **feature/docs**, there is no Background/Environment — just fill the template
sections (`### Summary`, `### Motivation / Use Case`, etc.).

## Step 8: Preview & confirm

Before creating **any** issue, print the full title, labels, **target repo**, and
body in a code block for the user, and wait for confirmation or edits. Make the
target repo explicit ("Filing to **hw-native-sys/simpler**"), since the routing
in Step 6 may send it away from pypto-lib.

Then walk the **Step 3c required-file list** against the assembled body and
confirm, file by file, that each one is either cited at a commit the reader can
fetch, inlined, or explicitly attached. Imports that resolve through the package
path rather than an explicit filename are the usual miss — check them by name,
not by scanning the body for paths. If any file is unaccounted for, go back to
Step 3d before creating.

Check the body once more for **dangling local references**: every file path it
mentions must either be resolvable in a pushed branch of a repo the reader can
fetch, or be accompanied by its inlined content / a stated attachment.

When the body carries inlined source, this preview is also the **publication
approval point** (Step 3d): show the user the actual content being published,
confirm the sanitization sweep found nothing left to redact, and get an explicit
go-ahead before creating.

## Step 9: Create

Run the single `gh issue create` for the confirmed target:

```bash
gh issue create \
  --repo <target repo> \
  --title "[Prefix] Short description" \
  --label "label1" \
  --body "$(cat <<'EOF'
## Background
...
### Description
...
EOF
)"
```

Write the body to a temp file and pass `--body-file` when it carries inlined
sources — a heredoc with nested code fences is easy to mangle, and `gh` rejects
bodies over ~65k characters (if you hit that, minimize the repro further rather
than truncating it).

Display the resulting issue URL. If Step 3d chose **attachment** for any file,
tell the user exactly which files still need to be drag-and-dropped onto the
issue page — the issue is incomplete until they do. Otherwise, mention that any
extra build output / device logs they want to add go on via the GitHub web UI.

## pypto-lib template field reference

- **Bug (`[Bug]`)** — required: Description, Environment, Host Platform. Include:
  Diagnosis (inside Background). Optional: Additional Context.
- **Feature (`[Feature]`)** — required: Summary, Motivation / Use Case. Optional:
  Proposed API / Behavior, Alternatives Considered, Additional Context.
- **Documentation (`[Docs]`)** — required: Documentation Location, What's Wrong
  or Missing?. Optional: Suggested Improvement, Additional Context.

## Checklist

- [ ] gh CLI authenticated
- [ ] Toolchain present (missing pieces installed; pypto NOT reset to main) and
      5 current-env component versions collected
- [ ] Reproduced on the current environment; symptom class + faulty component
      diagnosed
- [ ] Required-file list built (entry script + every local helper it imports);
      visibility of each confirmed against the **remote**, not just local
      remote-tracking refs
- [ ] Every local-only / uncommitted file minimized and **inlined in the body**
      (or explicitly attached), never referenced by bare local path
- [ ] Published content sanitized (no secrets, internal hosts, usernames /
      absolute paths, or proprietary vendor source) and user-approved
- [ ] Versions checked against pins (like-for-like); on mismatch, aligned & re-run
      or divergence noted in Background
- [ ] Background section written (symptom + reproduction env table + diagnosis)
- [ ] Dedup checked (pypto-lib + target repo)
- [ ] Routed: delegated to pypto/simpler skill, assembled ptoas/pto-isa direct
      body, or kept in pypto-lib when the boundary is unclear
- [ ] Issue previewed with explicit target repo and confirmed
- [ ] Issue created once (Step 9) and URL displayed
