---
name: create-issue
description: Reproduce a reported problem, collect dependency versions, and create a GitHub issue. Use when the user wants to file a bug, request a feature, or create any GitHub issue.
---

# Create GitHub Issue

Create issues that follow `.github/ISSUE_TEMPLATE/` templates exactly, after attempting to reproduce the problem first.

## Step 0: Determine Input Source

Check how the issue was triggered:

**A) From `KNOWN_ISSUES.md`** — If user says "create issue from known issues", "/create-issue known", or similar:

1. Read `KNOWN_ISSUES.md` from project root
2. If file doesn't exist or has no entries, tell user "No known issues found" and **stop**
3. List all entries with their title, severity, and brief description
4. Present the list and ask the user which issue they want to file
5. **Verify the selected issue is still real and unresolved:**
   - If `Location` is present and not `N/A`:
     - Read the file(s) mentioned and check if the problem still exists in the current code
     - If **resolved**: remove the entry from `KNOWN_ISSUES.md`, inform user, and **stop**
     - If **still present**: proceed to Step 1 using the issue's description as input
   - If `Location` is missing or `N/A`:
     - Ask the user to confirm whether the issue is still valid based on the description
     - If **no longer valid**: remove the entry from `KNOWN_ISSUES.md`, inform user, and **stop**
     - If **still valid**: proceed to Step 1 using the issue's description as input
6. After the GitHub issue is created, **remove the entry** from `KNOWN_ISSUES.md` (the issue is now tracked on GitHub)

**B) Direct user input** — Normal flow, proceed to Step 1 with user-provided description.

## Step 1: Authenticate

```bash
gh auth status
```

If not authenticated, tell the user to run `gh auth login` and **stop**.

## Step 2: Setup Environment & Collect Dependency Versions

**Only required for bug reports.** For feature requests and documentation issues, **skip Step 2 and Step 3** entirely — proceed directly to Step 4.

Use the `/setup_env` skill to ensure the development environment is ready (pypto, ptoas, simpler), with the following adjustments to ensure we test against the latest code:

- **pypto**: pull latest `main` and reinstall
- **simpler**: pull latest `stable`
- **ptoas**: install if missing (via `/setup_env` Step 5)

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
SIMPLER_REMOTE=$(get_upstream_remote "$SIMPLER_ROOT" "ChaoWao/simpler")
```

Then pull the latest code:

```bash
# pypto: ensure latest main
cd "$PYPTO_ROOT"
git fetch "$PYPTO_REMOTE"
git checkout main
git pull "$PYPTO_REMOTE" main
rm -rf build/
python3 -m pip install -e .

# simpler: ensure latest stable
cd "$SIMPLER_ROOT"
git fetch "$SIMPLER_REMOTE"
git checkout stable
git pull "$SIMPLER_REMOTE" stable
```

After setup, collect dependency versions:

```bash
# pypto-lib commit-id (short, 7 chars)
git rev-parse --short HEAD

# pypto commit-id (short, 7 chars) + branch
git -C "$PYPTO_ROOT" log -1 --format="%h"
git -C "$PYPTO_ROOT" branch --show-current

# simpler commit-id (short, 7 chars) + branch
git -C "$SIMPLER_ROOT" log -1 --format="%h"
git -C "$SIMPLER_ROOT" branch --show-current

# CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || echo "not detected"
```

### ptoas Installation Check

ptoas version cannot be detected programmatically. Only verify that it is installed:

```bash
# Linux: check binary exists
test -x "$PTOAS_ROOT/ptoas" || test -x "$PTOAS_ROOT/bin/ptoas" && echo "installed" || echo "not found"

# macOS: check pip package
python3 -c "import ptoas" 2>/dev/null && echo "installed" || echo "not found"
```

If ptoas is **not installed**, use the `/setup_env` skill's Step 5 to install it before proceeding.

The ptoas version will be confirmed with the user later in Step 7 when formatting the issue body. If installed via `/setup_env`, use the version pinned in the setup_env skill (check `SKILL.md` for the current pinned version, e.g., `v0.6`).

Record all other values. If any version cannot be detected, use "unknown" and continue.

If `/setup_env` fails entirely, **skip Step 3** (reproduction), note the failure, and proceed directly to Step 4.

## Step 3: Try to Reproduce

This step attempts to confirm the issue before filing. **Only applies to bug reports**, not feature requests or documentation issues.

### 3a: Identify the Reproduction Script

- If the user provided a specific script or command, use that.
- If the issue is about a specific example (e.g., "softmax_example.py fails"), use that file from `examples/`.
- If unclear, **ask the user** which script reproduces the problem.

### 3b: First Reproduction Attempt (current environment)

Run the reproduction script in the current environment (simpler on its current branch, typically `stable`):

```bash
python3 <reproduction_script>
```

Capture stdout and stderr. If the script **succeeds** (no error), report to the user: "The issue does not reproduce in the current environment." Ask if they still want to file. If no, **stop**.

### 3c: Diagnose the Faulty Component

The compilation pipeline has three stages. Analyze the error output to determine which component is at fault:

| Stage | Component | Error Signals |
|---|---|---|
| 1. IR generation & compilation | **pypto** | Python traceback in pypto modules, IR validation errors, codegen failures |
| 2. PTO assembly & optimization | **ptoas** | ptoas error messages, assembly syntax errors, optimizer crashes |
| 3. On-device / simulation execution | **simpler** | Runtime crashes, incorrect output values, hangs during execution, device errors |

**State your diagnosis** to the user: "This appears to be a **pypto** / **ptoas** / **simpler** issue because ..."

Record the diagnosed component for inclusion in the issue body.

### 3d: Retry with simpler main (only for simpler issues)

Since Step 2 already updated pypto to latest main and simpler to latest stable, the only remaining retry is switching simpler to `main`.

**Only if the error is diagnosed as a simpler (runtime) issue**, try switching simpler to the latest `main` branch:

```bash
cd "$SIMPLER_ROOT"
git checkout main
git pull "$SIMPLER_REMOTE" main
```

Re-run the reproduction script.

- If the **same error** persists → confirmed bug on both simpler stable and main, record both commits in the Environment table, proceed to file.
- If the **error changes** → **go back to Step 3c** to re-diagnose.
- If **fixed** → tell user: "Issue is fixed on simpler main. Consider updating." Ask if they still want to file. If no, **stop**.

If the error is a **pypto** or **ptoas** issue, **skip this step** — pypto is already on latest main, and ptoas cannot be easily switched.

### 3e: Decision

| Diagnosis | Result (latest pypto main + simpler stable) | After simpler main | Action |
|---|---|---|---|
| pypto | Reproduces | (skipped) | Confirmed bug — proceed to file |
| ptoas | Reproduces | (skipped) | Confirmed bug — proceed to file |
| simpler | Reproduces | Reproduces | Confirmed bug — list both commits |
| simpler | Reproduces | Error changes | **Go back to Step 3c** — re-diagnose |
| simpler | Reproduces | Fixed | Tell user, ask if still want to file |
| any | Does not reproduce | — | Tell user: "Cannot reproduce." Ask if still want to file. |

## Step 4: Check for Existing Issues

**Launch a `general-purpose` agent** (via `Task` tool, **model: haiku**) to perform the dedup check. This keeps the main context clean and fast.

**Agent prompt must include:** the issue summary/keywords, the diagnosed component (from Step 3c, if available), and these exact instructions:

> **IMPORTANT: ONLY use `gh` CLI commands with explicit `--repo OWNER/REPO` flag. Do NOT read source code, test files, or explore the repository. Your sole job is to check GitHub issues for duplicates.**
>
> Search the following repos based on the diagnosed component:
> - **Always** search `hw-native-sys/pypto-lib`
> - If diagnosed as **pypto** issue, also search `hw-native-sys/pypto`
> - If diagnosed as **simpler** issue, also search `ChaoWao/simpler`
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

### ptoas Version Confirmation

Before formatting the Environment table, **ask the user** to confirm the ptoas version:

- "Was ptoas installed via `/setup_env`?" — If **yes**, read the pinned version from `.claude/skills/setup_env/SKILL.md` (look for `PTOAS_VERSION=...`) and use that.
- If **no**, ask the user to provide the version they installed.

**For all bug reports**, include these sections automatically:

```markdown
### Diagnosis

**pypto** / **ptoas** / **simpler** — <brief reason for the diagnosis>

### Description

<clear description of the bug and how to reproduce it>

### Environment

| Component | Version |
|---|---|
| pypto-lib | `<7-char commit>` |
| pypto | `<7-char commit>` (branch: `<branch>`) |
| simpler | `<7-char commit>` (branch: `<branch>`) |
| ptoas | `<version>` |
| CANN | `<version or "not detected">` |

### Host Platform

`<os> <arch>` (e.g., Linux aarch64, Linux x86_64, macOS arm64)

If the issue is a simpler problem and was tested on both `stable` and `main` branches, list both in the Environment table:

| Component | Version |
|---|---|
| simpler (stable) | `<7-char commit>` |
| simpler (main) | `<7-char commit>` |

### Attachments
```

After creating the issue, **prompt the user**: "If you have relevant source files or build output, please attach them to the issue via the GitHub web UI (drag-and-drop on the issue page)."

## Step 8: Preview and Confirm

Before creating the issue, **print the full issue content** to the user for review:

1. Display the formatted title, labels, and body in a code block so the user can verify.
2. **Determine the target repository** based on the diagnosis:
   - Default: the current repository (pypto-lib), determined by `gh repo view --json nameWithOwner -q .nameWithOwner`
   - If the diagnosis clearly points to **pypto** → ask the user: "This issue appears to be a pypto problem. Would you like to file it to **hw-native-sys/pypto** instead of pypto-lib?"
   - If the diagnosis clearly points to **simpler** → ask the user: "This issue appears to be a simpler problem. Would you like to file it to **ChaoWao/simpler** instead of pypto-lib?"
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

If the issue was sourced from `KNOWN_ISSUES.md`, **remove the entry** from the file.

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

- [ ] Input source determined (KNOWN_ISSUES.md or direct)
- [ ] gh CLI authenticated
- [ ] Environment set up (pypto latest main, simpler latest stable) and versions collected
- [ ] Reproduction attempted and faulty component diagnosed
- [ ] If simpler issue: retried with simpler main
- [ ] Searched for existing issues (dedup)
- [ ] Issue classified to correct template, all required fields filled
- [ ] Issue content previewed to user
- [ ] Target repo confirmed (default pypto-lib, or pypto/simpler if applicable)
- [ ] Issue created and URL displayed
- [ ] If from KNOWN_ISSUES.md: entry removed from file
