---
name: fix-pr
description: Fix GitHub PR issues — address review comments and resolve CI failures in a loop until the PR is fully clean. Fetches CI errors online and triages review feedback. Use when fixing PR problems, addressing review comments, or resolving CI failures.
---

# Fix PR Workflow

Fix PR issues (review comments, CI failures) in a loop until the PR is clean. Steps 1-7 repeat;
max 5 iterations.

Input: PR number (`123`, `#123`), branch name, or no argument (uses the current branch).

## 1. Match input to PR

```bash
gh pr view <number> --json number,title,headRefName,state   # by number
gh pr list --head "$(git branch --show-current)" --json number,title,state
```

## 2. Detect issues

Run both in parallel:

```bash
OWNER=$(gh repo view --json owner -q '.owner.login')
NAME=$(gh repo view --json name -q '.name')

gh api graphql \
  -F owner="$OWNER" -F name="$NAME" -F number=<NUMBER> \
  -f query='
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        nodes {
          id isResolved
          comments(last: 1) {
            nodes { id databaseId body author { login } path line }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}' > /tmp/threads.json

gh pr checks <NUMBER>
```

If `pageInfo.hasNextPage` is true, re-run with `-F cursor="<endCursor>"` and accumulate nodes
across pages before counting.

Report: "**Iteration N** — X unresolved comments, Y failed/pending checks."

**Exit condition:** all checks green AND no unresolved comments. Pending checks are NOT clean.

## 3. Classify

**Review comments** — filter `isResolved: false`. Bot reviewers (CodeRabbit, Copilot) are
classified by content, exactly like humans.

| Category | Meaning | Examples |
| -------- | ------- | -------- |
| **A: Actionable** | Code changes required | Bugs, missing validation, races |
| **B: Discussable** | May skip if the code already follows `docs/` or `.claude/rules/` | Style preferences, premature optimization |
| **C: Informational** | Resolve without changes | Acknowledgments, "optional" suggestions |

When unsure, default to B.

**CI failures** — fetch logs online; reproduce locally only as a last resort.

```bash
gh pr checks <NUMBER> --json name,state,link | jq '.[] | select(.state == "FAILURE")'

# link format: https://github.com/<owner>/<repo>/actions/runs/<RUN_ID>/job/<JOB_ID>
RUN_ID=$(echo "$LINK" | sed -En 's|.*/runs/([0-9]+)/.*|\1|p')
JOB_ID=$(echo "$LINK" | sed -En 's|.*/job/([0-9]+).*|\1|p')

gh run view --job "$JOB_ID" --log-failed   # single job — works while the run is still going
gh run view "$RUN_ID" --log-failed         # whole run — needs status == "completed"
```

`gh run view <RUN_ID> --log-failed` requires **every** job in the run to be finished, so on this
repo's long device jobs prefer the per-job form. Large logs:
`... --log-failed 2>&1 | grep -E "error:|FAILED|fatal" | head -20`. External (non-Actions) checks
have no run ID — open the `link` URL directly.

## 4. Confirm with user

Present everything in one numbered list:

```text
Review Comments:
  1. [A] models/deepseek/v4-flash/moe.py:42 — Missing bounds check (alice)
  2. [B] golden/runner.py:15 — Style suggestion (coderabbitai)
CI Failures:
  3. [CI] lint — golden/runner.py:10: F401 unused import
```

Recommend addressing A + CI. Do NOT resolve any comment without user consent. On later
iterations, reuse a prior "address all" policy for the same categories; ask only about genuinely
new or ambiguous issues.

## 5. Fix and fold into the PR's commit

Read the affected files, make the edits, then stage **only the paths you touched** — never
`git add -A` / `git add .`, which sweeps unrelated worktree edits into the PR's commit.

**Never append a standalone `fix(pr): ...` commit.** The PR stays as the commits it already had,
with the fixes folded in — reviewers see a clean diff, not a log of review churn. This applies on
every iteration.

```bash
git add path/to/fixed1.py path/to/fixed2.md   # explicit paths, including new files
git fetch upstream 2>/dev/null || git fetch origin
BASE_REF=$(git rev-parse --verify upstream/main >/dev/null 2>&1 && echo upstream/main || echo origin/main)
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count)
```

| `COMMITS_AHEAD` | How to fold the fix in |
| --------------- | ---------------------- |
| `0` | **Error — stop.** Nothing ahead of base to fold into. |
| `1` | `git commit --amend -F <msgfile>` — write the updated message to a file first. Never bare `--amend` (opens an editor and hangs in a non-interactive shell), never `--no-edit` (leaves the message stale). |
| `> 1` | Capture the original message (`git log -1 --format='%B' $(git rev-list --reverse "$BASE_REF"..HEAD \| head -1)`), `git reset --soft "$BASE_REF"`, then recommit once via `/git-commit` starting from that message. |

**Evolve the message, don't freeze or replace it.** It must describe the commit's *final combined*
diff, keeping the original `Type: subject` and intent:

- A fix that changes no behavior (typo, lint, comment) → message may stay as-is.
- A fix that changes behavior or scope → work it into the existing body; change the subject only
  if the headline scope actually changed.
- Never `Fix: resolve review comments for #123` — that replaces a description of the code with a
  changelog of review churn.

```bash
git push --force-with-lease
```

If the subject or body changed, keep the PR in sync: `gh pr edit <NUMBER> --title ... --body ...`
(the PR body becomes the squash-merge entry on `main`).

## 6. Resolve comment threads

Reply with
`gh api repos/{owner}/{repo}/pulls/<number>/comments/<comment_id>/replies -f body='...'`, then
resolve via the GraphQL `resolveReviewThread` mutation. Use single quotes for `-f body=` and
`--jq`; `gh`'s jq processor reads `$` as a variable sign.

Templates — Fixed: "Fixed in `<commit>` — description" / Skip: "Current code follows `<rule>`" /
Acknowledged: "Acknowledged, thank you!"

## 7. Wait and re-check

```bash
gh pr checks <NUMBER> --watch --interval 60
```

Do not block on `sleep`. Run the watch in the background if CI may outlast the command timeout,
and poll with a plain `gh pr checks <NUMBER>` in between. Then loop back to step 2.

**Safeguards:** stop after 5 iterations and report what remains; if the same issue reappears after
a fix, flag it instead of retrying blindly; respect user interruptions.

## Error handling

| Error | Action |
| ----- | ------ |
| PR not found | `gh pr list`; ask the user to confirm |
| Not authenticated | Tell the user to run `gh auth login` |
| Unclear comment | Category B — discuss |
| CI logs unavailable | Fall back to a local reproduce |
| Same failure persists | Flag to the user; do not retry the same fix |
