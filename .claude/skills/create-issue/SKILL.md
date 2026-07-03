---
name: create-issue
description: Reproduce a reported problem, collect dependency versions, and create a GitHub issue. Use when the user wants to file a bug, request a feature, or create any GitHub issue.
---

# Create GitHub Issue

Create issues that follow `.github/ISSUE_TEMPLATE/` templates exactly, after attempting to reproduce the problem first.

## Step 1: Authenticate

```bash
gh auth status
```

If not authenticated, tell the user to run `gh auth login` and **stop**.

## Step 2: Setup Environment & Collect Dependency Versions

**Only required for bug reports.** For feature requests and documentation issues, **skip Step 2 and Step 3** entirely — proceed directly to Step 4.

Use the `/setup_env` skill to ensure the development environment is ready (pypto, ptoas, pto-isa), with the following adjustments to ensure we test against the latest code:

- **pypto**: pull latest `main` and reinstall (this also updates the bundled `runtime` submodule)
- **ptoas**: install if missing (via `/setup_env` Step 5)
- **pto-isa**: clone at the pinned commit if missing (via `/setup_env`)

Before pulling, determine the correct remote for each repo. The upstream remote may be `origin` or `upstream` depending on whether the user cloned from the original repo or a fork:

```bash
# Helper: find the remote pointing to the upstream repo
# Usage: get_upstream_remote <repo_dir> <upstream_url_keyword>
# Example: get_upstream_remote "$PYPTO_ROOT" "hw-native-sys/pypto"
get_upstream_remote() {
    local repo_dir="$1" keyword="$2"
    cd "$repo_dir"
    for remote in $(git remote); do
        if git remote get-url "$remote" 2>/dev/null | grep -q "$keyword"; then
            echo "$remote"
            return
        fi
    done
    echo "origin"  # fallback
}

PYPTO_REMOTE=$(get_upstream_remote "$PYPTO_ROOT" "hw-native-sys/pypto")
```

Then pull the latest code:

```bash
# pypto: ensure latest main (includes runtime submodule)
cd "$PYPTO_ROOT"
git fetch "$PYPTO_REMOTE"
git checkout main
git pull "$PYPTO_REMOTE" main
git submodule update --init --recursive
rm -rf build/
python3 -m pip install -e .
pip install "$PYPTO_ROOT/runtime"
```

After setup, collect dependency versions:

```bash
# pypto-lib commit-id (short, 7 chars)
git rev-parse --short HEAD

# pypto commit-id (short, 7 chars) + branch
git -C "$PYPTO_ROOT" log -1 --format="%h"
git -C "$PYPTO_ROOT" branch --show-current

# pypto runtime submodule commit-id (short, 7 chars) — part of pypto, record for completeness
git -C "$PYPTO_ROOT/runtime" log -1 --format="%h"

# pto-isa commit-id (short, 7 chars) + branch
git -C "$PTO_ISA_ROOT" log -1 --format="%h"
git -C "$PTO_ISA_ROOT" branch --show-current

# CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || echo "not detected"
```

### ptoas Version Detection

Detect ptoas installation and version:

```bash
"$PTOAS_ROOT/ptoas" --version 2>/dev/null || echo "not found"
```

If ptoas is **not found** or `PTOAS_ROOT` is not set, install it following the version pinned in the **pypto** checkout's `toolchain/versions.env` (`$PYPTO_ROOT/toolchain/versions.env`, look for `PTOAS_VERSION=...`; pto-isa is pinned separately in `$PYPTO_ROOT/runtime/pto_isa.pin`). See the `/setup_env` skill for the full install. After installation, re-run the version check.

Record the detected ptoas version for use in the Environment table.

Record all other values. If any version cannot be detected, use "unknown" and continue.

If `/setup_env` fails entirely, **skip Step 3** (reproduction), note the failure, and proceed directly to Step 4.

## Step 3: Try to Reproduce

This step attempts to confirm the issue before filing. **Only applies to bug reports**, not feature requests or documentation issues.

### 3a: Identify the Reproduction Script

- If the user provided a specific script or command, use that.
- If the issue is about a specific example (e.g., "softmax_example.py fails"), use that file from `examples/`.
- If unclear, **ask the user** which script reproduces the problem.

### 3b: First Reproduction Attempt (current environment)

Run the reproduction script in the current environment (pypto on latest `main`, with its pinned `runtime` submodule):

```bash
python3 <reproduction_script>
```

Capture stdout and stderr. If the script **succeeds** (no error), report to the user: "The issue does not reproduce in the current environment." Ask if they still want to file. If no, **stop**.

### 3c: Diagnose the Faulty Component

Analyze the error output to determine which component is at fault. **In this repo the vast majority of issues are pypto issues** — default to pypto unless the error clearly points elsewhere. Note that the **runtime** (on-device / simulation execution) is the `pypto/runtime` submodule and is therefore part of **pypto** — runtime crashes are pypto issues.

| Stage | Component | Error Signals |
|---|---|---|
| IR generation & compilation, on-device / simulation execution | **pypto** | Python traceback in pypto modules, IR validation errors, codegen failures, runtime crashes, incorrect output values, hangs during execution, device errors |
| PTO assembly & optimization | **ptoas** | ptoas error messages, assembly syntax errors, optimizer crashes |
| Tile-ISA implementation | **pto-isa** | Errors originating in pto-isa sources, a missing/unimplemented tile op, or incorrect results traced to a specific tile-ISA kernel |

**State your diagnosis** to the user: "This appears to be a **pypto** / **ptoas** / **pto-isa** issue because ..."

Record the diagnosed component for inclusion in the issue body.

### 3d: Decision

| Diagnosis | Result (latest pypto main) | Action |
|---|---|---|
| pypto | Reproduces | Confirmed bug — proceed to file |
| ptoas | Reproduces | Confirmed bug — proceed to file |
| pto-isa | Reproduces | Confirmed bug — proceed to file |
| any | Does not reproduce | Tell user: "Cannot reproduce." Ask if still want to file. |

## Step 4: Check for Existing Issues

**Launch a `general-purpose` agent** (via `Task` tool, **model: haiku**) to perform the dedup check. This keeps the main context clean and fast.

**Agent prompt must include:** the issue summary/keywords, the diagnosed component (from Step 3c, if available), and these exact instructions:

> **IMPORTANT: ONLY use `gh` CLI commands with explicit `--repo OWNER/REPO` flag. Do NOT read source code, test files, or explore the repository. Your sole job is to check GitHub issues for duplicates.**
>
> Search the following repos based on the diagnosed component:
> - **Always** search `hw-native-sys/pypto-lib`
> - If diagnosed as **pypto** issue, also search `hw-native-sys/pypto`
> - If diagnosed as **pto-isa** issue, also search `hw-native-sys/pto-isa`
>
> For each repo, follow the two-step process below. Then return EXACTLY one of: `DUPLICATE REPO#N`, `RELATED REPO#N1 REPO#N2 ...`, or `NO_MATCH`. Keep your response to 1-3 sentences plus the verdict.

### Two-Step Search Process (for the agent)

**Step A — Scan all open issue titles** (repeat for each target repo):

```bash
gh issue list --repo OWNER/REPO --state open --limit 500 --json number,title,labels \
  --jq '.[] | "\(.number)\t\(.title)\t\(.labels | map(.name) | join(","))"'
```

Scan output for keywords related to the new issue.

**Step B — Deep-read candidates only (max 3):**

For each title that looks related (up to 3), fetch context:

```bash
gh issue view NUMBER --repo OWNER/REPO
```

Only read body — skip `--comments` unless the body is ambiguous. Determine if it's truly the same issue or just superficially similar.

### Decision rules (agent returns)

- **Exact match** (same root cause/request) → return `DUPLICATE REPO#N` (e.g., `DUPLICATE hw-native-sys/pypto#42`)
- **Related but different** → return `RELATED REPO#N1 REPO#N2 ...`
- **No matches in any repo** → return `NO_MATCH`

### How to act on the result

- `DUPLICATE REPO#N` → Do NOT create. Tell the user the existing issue and which repo it's in. **Stop here.**
- `RELATED REPO#N1 ...` → Proceed, reference in body: `Related: REPO#N1, REPO#N2`
- `NO_MATCH` → Proceed normally.

## Step 5: Classify the Issue

Read `.github/ISSUE_TEMPLATE/` to get the current templates, then match the user's description to the correct template:

| Template | Use When | Labels |
| -------- | -------- | ------ |
| `bug_report.yml` | Compilation error, codegen error, runtime error, incorrect output | `bug` |
| `feature_request.yml` | New tensor function, new example/model, API improvement | `enhancement` |
| `documentation.yml` | Missing, incorrect, or unclear docs | `documentation` |

**Classification rules:**

- If about a crash, error, or incorrect behavior → `bug_report.yml`
- If requesting a new capability or improvement → `feature_request.yml`
- If about docs being wrong/missing → `documentation.yml`

**If ambiguous**, ask the user to clarify using `AskUserQuestion`.

## Step 6: Gather Required Fields

Each template has **required fields** (marked `required: true` in the YAML). You MUST fill every required field.

**Ask the user** for any required information you cannot infer. Use `AskUserQuestion` for dropdown selections.

**For fields you can auto-fill:**

- **Title prefix**: Use the template's title prefix (`[Bug]`, `[Feature]`, `[Docs]`)
- **Host Platform**: Run `uname -s -m` to detect OS and arch. Map to: `Linux aarch64` → `Linux (aarch64)`, `Linux x86_64` → `Linux (x86_64)`, `Darwin arm64` → `macOS (arm64)`. Fall back to `Other` if unrecognized.
- **Environment**: Use the values collected in Step 2 (all commit IDs are 7-char short hashes).

## Step 7: Format the Issue Body

Since `gh issue create` uses markdown body (not YAML form fields), format the body to match the template structure using markdown sections:

```markdown
### Field Label

Field content here

### Another Field

More content
```

**For dropdown fields**, state the selected value as plain text.

**For all bug reports**, include these sections automatically:

```markdown
### Diagnosis

**pypto** / **ptoas** / **pto-isa** — <brief reason for the diagnosis>

### Description

<clear description of the bug and how to reproduce it>

### Environment

| Component | Version |
|---|---|
| pypto-lib | `<7-char commit>` |
| pypto | `<7-char commit>` (branch: `<branch>`) |
| pypto runtime (submodule) | `<7-char commit>` |
| pto-isa | `<7-char commit>` (branch: `<branch>`) |
| ptoas | `<version>` |
| CANN | `<version or "not detected">` |

### Host Platform

`<os> <arch>` (e.g., Linux aarch64, Linux x86_64, macOS arm64)

### Attachments
```

After creating the issue, **prompt the user**: "If you have relevant source files or build output, please attach them to the issue via the GitHub web UI (drag-and-drop on the issue page)."

## Step 8: Preview and Confirm

Before creating the issue, **print the full issue content** to the user for review:

1. Display the formatted title, labels, and body in a code block so the user can verify.
2. **Determine the target repository** based on the diagnosis:
   - Default: the current repository (pypto-lib), determined by `gh repo view --json nameWithOwner -q .nameWithOwner`
   - If the diagnosis clearly points to **pypto** (including runtime) → ask the user: "This issue appears to be a pypto problem. Would you like to file it to **hw-native-sys/pypto** instead of pypto-lib?"
   - If the diagnosis clearly points to **pto-isa** → ask the user: "This issue appears to be a pto-isa problem. Would you like to file it to **hw-native-sys/pto-isa** instead of pypto-lib?"
   - If **ptoas** or unclear → file to pypto-lib (default).
3. Wait for the user to confirm or request changes before proceeding to Step 9.

## Step 9: Create the Issue

```bash
gh issue create \
  --repo TARGET_REPO \
  --title "[Prefix] Short description" \
  --label "label1" --label "label2" \
  --body "$(cat <<'EOF'
### Field 1
content

### Field 2
content
EOF
)"
```

**After creation**, display the issue URL to the user.

## Template Field Reference

### Bug Report (`[Bug]`)

Required: Description, Environment, Host Platform (dropdown)
Auto-included: Diagnosis
Optional: Additional Context

### Feature Request (`[Feature]`)

Required: Summary, Motivation / Use Case
Optional: Proposed API / Behavior, Alternatives Considered, Additional Context

### Documentation (`[Docs]`)

Required: Documentation Location, What's Wrong or Missing?
Optional: Suggested Improvement, Additional Context

## Checklist

- [ ] gh CLI authenticated
- [ ] Environment set up (pypto latest main) and versions collected
- [ ] Reproduction attempted and faulty component diagnosed
- [ ] Searched for existing issues (dedup)
- [ ] Issue classified to correct template, all required fields filled
- [ ] Issue content previewed to user
- [ ] Target repo confirmed (default pypto-lib, or pypto/pto-isa if applicable)
- [ ] Issue created and URL displayed
