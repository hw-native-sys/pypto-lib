---
name: merge-pr
description: Bring an approved GitHub PR to the /github-pr title and body conventions, then squash-merge it with that exact text as the commit message. Use when the user asks to merge a PR, land a PR, or clean up a PR's title and description before merging.
---

# Merge PR Workflow

The repo squash-merges, so the PR's title and the body file written in step 4
become the commit subject and body on `main` verbatim. Rewrite them to the
`/github-pr` conventions **before** merging, never after.

Input: PR number (`942`, `#942`), branch name, or no argument (current branch).

Several steps below rest on the repo's current merge settings — squash-only,
`PR_TITLE` / `PR_BODY` commit defaults, and a `main` protected by a repository
**ruleset**. Re-derive them rather than trusting this paragraph if a merge
behaves unexpectedly:

```bash
gh api repos/hw-native-sys/pypto-lib \
  -q '"squash=\(.allow_squash_merge) title=\(.squash_merge_commit_title) msg=\(.squash_merge_commit_message)"'
gh api repos/hw-native-sys/pypto-lib/rulesets -q '.[] | "\(.id) \(.name) \(.enforcement)"'
gh api repos/hw-native-sys/pypto-lib/rulesets/<id> \
  -q '"bypass=\(.bypass_actors|length)", (.rules[].type)'
```

**`gh api .../branches/main/protection` returns 404 here, and that is not a
verdict.** The legacy endpoint cannot see rulesets; `main` is protected.
Ruleset `default` (13870312) is active with an **empty bypass list** — repo
admins included — and carries `deletion`, `non_fast_forward`,
`required_linear_history`, `pull_request` (0 required approvals) and
`required_status_checks` naming only `pre-commit` and `unit-tests`, non-strict.

Two consequences run through the steps below: the merge is gated server-side on
those two checks, and **`main` can never be force-pushed**, so a wrong commit
message is permanent. Get the message right before step 5, not after.

Merge only when the user asked for it. Steps 1-3 only read. Step 4 rewrites
someone else's PR title and body on GitHub — visible to the author and
reviewers, though reversible; do it as part of a merge the user asked for, not
as unprompted tidying. Step 5 is the irreversible one.

Several PR numbers may be given. Process them **one at a time**, start to
finish, before moving to the next. Never merge them in parallel: each merge
moves `main`, so a PR that was `MERGEABLE` a moment ago can turn `DIRTY`
against the new `main`, and every remaining PR's green checks were run against
a base that no longer exists. Re-read step 2 for each PR after the previous one
lands — do not carry over a readiness verdict.

## 1. Resolve the repo and the PR

The clone is usually a fork — `origin` is the personal fork, `upstream` is the
real repo, and a bare `gh pr view` cannot tell which one owns the PR. Resolve
the repo first and pass `-R "$REPO"` on **every** `gh` call below.

```bash
git remote -v
REPO=$(gh repo view "$(git remote get-url upstream 2>/dev/null || git remote get-url origin)" \
  --json nameWithOwner -q .nameWithOwner)
```

```bash
gh pr view <number> -R "$REPO" --json number,title,body,state,isDraft,headRefName,baseRefName,url
gh pr list -R "$REPO" --head "$(git branch --show-current)" --json number,title,state   # no argument
```

`baseRefName` must be `main`. A PR stacked on another branch merges into that
branch, not into `main` — its body is not what lands on `main` and the checks
mean something different. Stop and confirm with the user instead of merging it.

## 2. Check merge readiness

Two separate gates. GitHub enforces the first; this skill enforces the second,
because the ruleset requires only `pre-commit` and `unit-tests` — nothing on
the server stops a merge over a red `a2a3`, a red `serving-*`, or an
unaddressed review.

### Server-side merge state

```bash
gh pr view <number> -R "$REPO" --json state,isDraft,mergeable,mergeStateStatus \
  -q '"\(.state) draft=\(.isDraft) \(.mergeable) \(.mergeStateStatus)"'
```

| Signal | Merge when |
| ------ | ---------- |
| `state` / `isDraft` | `OPEN` and not a draft |
| `mergeable` | `MERGEABLE` — `CONFLICTING` means the author must rebase; report and stop |
| `mergeStateStatus` | `CLEAN` or `UNSTABLE` — `DIRTY` and `BLOCKED` stop the merge |

`UNKNOWN` on either field is not a verdict: GitHub computes mergeability lazily
and returns `UNKNOWN` until it finishes (a `gh pr list` query returns it far
more often than `gh pr view`). Wait a second and re-query; never read it as
"not mergeable".

**`UNSTABLE` carries no information about *which* check is red.** Only
`pre-commit` and `unit-tests` are required, so every other check that is not
`SUCCESS` — one red `sim` job, a genuinely broken `a2a3`, or a run still in
flight — yields `UNSTABLE` just the same. Never infer from `UNSTABLE` that only
the exempt jobs failed; the check list below is the only thing that decides.

`BLOCKED` **does** occur: it is what a red or still-pending `pre-commit` /
`unit-tests` looks like from the merge state. Wait for them, or route to
`/fix-pr`; there is nothing to merge past.

`BEHIND` does not occur — `required_status_checks` is non-strict
(`strict_required_status_checks_policy: false`), so the ruleset never asks for
an up-to-date branch. A PR whose head is behind `main` still reads `CLEAN` /
`UNSTABLE`. That is why the stale-base risk in the multi-PR note above has to
be handled by re-checking, not by watching for a status.

### Checks

Read the per-check verdict with `gh pr checks`, not `statusCheckRollup`: the
rollup's check-run entries have no `state` field (it is `status` +
`conclusion`, and only legacy status contexts carry `state`), while
`gh pr checks` normalizes every entry into a documented `bucket` —
`pass` / `fail` / `pending` / `skipping` / `cancel`.

```bash
gh pr checks <number> -R "$REPO" --json name,bucket,state \
  -q '[.[] | select(.bucket != "pass")] | .[] | "\(.bucket)\t\(.state)\t\(.name)"'
```

Enumerate every line it prints and classify it — this is a positive
determination, never an inference from the merge state:

| Bucket | Meaning |
| ------ | ------- |
| `skipping` | Normal. `sim`, `a2a3`, and `serving-*` gate on `detect-changes` and skip when the PR touches nothing they cover |
| `pending` | Unfinished — watch it, below |
| `fail` / `cancel` | Blocks the merge, **unless the name matches the exempt list below** |

Nothing printed means every check passed. A `fail` or `cancel` outside the
exempt list is not ready: report it and stop, or route to `/fix-pr`. Do not
merge past it, and do not reach for `--admin` — the ruleset's bypass list is
empty, so no account can override it and the flag buys only a confusing error.

**Exempt: the `sim` jobs — every entry named `sim` or starting with `sim (`.**
Match the prefix, not a fixed pair of names: the matrix reports
`sim (a2a3sim)` and `sim (a5sim)` once it runs, and collapses to a single bare
`sim` when `detect-changes` skips it.

Nobody maintains simulator CI at present, so a red `sim` entry carries no
signal and its log is not worth reading. The simulator cannot execute whole
classes of kernel — most often multi-rank comm (CP / EP / TP entries), and it
also enforces A5 buffer limits and pto-isa CPU templates that no A2/A3 device
run ever sees — so a device-validated kernel routinely fails there while `a2a3`
and `serving-*` pass.

Do **not** open a failing `sim` job, fetch its log, or diagnose the error: name
the red `sim` entries in the report and merge. The exemption is by job name
only — every other red check still stops the merge, including `serving-*`,
which is easy to lose in a list that is mostly `sim` noise.

### Pending checks — watch, do not poll

`pending` is not a failure, only an unfinished run. Block on it instead of
re-running the listing in a loop; each poll costs a round trip and the `a2a3`
and `serving-*` jobs run for tens of minutes.

```bash
gh pr checks <number> -R "$REPO" --watch -i 30
```

`--watch` blocks until every check finishes, whatever produced it. Do not
build the wait out of `gh run watch` instead: that needs a run id scraped from
each check's `link`, and a check from any integration other than GitHub Actions
has no `/runs/<id>` — CodeRabbit reports an empty `link` — so the scrape
silently drops it and the wait returns while it is still running.

`--watch` exits non-zero when a check ends red. That is the exemption table's
business, not a reason to stop. Re-read the merge state and the check list
after it returns; the tables above still decide.

### Unresolved review feedback

The merge state carries no review information at all here, and `reviews` only
carries submitted verdicts. A reviewer who left inline comments without
submitting `CHANGES_REQUESTED` shows up in neither, and merging drops those
threads on the floor.

Inline threads:

```bash
gh api graphql -F owner="${REPO%/*}" -F name="${REPO#*/}" -F number=<number> -f query='
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
        nodes {
          isResolved isOutdated
          comments(first: 5) { nodes { author { login } path line body } }
        }
      }
    }
  }
}'
```

Top-level review verdicts and PR comments are **not** review threads and the
query above cannot see them — CodeRabbit posts its summary that way, and so
does a human who reviews from the conversation tab:

```bash
gh pr view <number> -R "$REPO" --json reviews,comments -q '
  (.reviews[]  | "review  \(.author.login) \(.state): \([.body|split("\n")[]|select(test("^\\s*(<!--|$)")|not)][0] // "(empty)")"),
  (.comments[] | "comment \(.author.login): \([.body|split("\n")[]|select(test("^\\s*(<!--|$)")|not)][0] // "(empty)")")'
```

Report every inline thread with `isResolved: false` and `isOutdated: false` —
author, `path:line`, and the comment's first line — plus any review or comment
raising something the diff does not already address. **Do not classify them and
merge anyway.** Judging someone else's review comment as a nit is the user's
call, not this skill's: list them, say the merge is otherwise ready, and let the
user decide.

## 3. Read the PR's real scope

Derive the rewrite from the diff, never from the existing PR body — that body
is usually the author's web-page-style description, and its headings, test
plans, and rationale sections are exactly what must not land in `git log`.

```bash
gh pr diff <number> -R "$REPO"
gh pr view <number> -R "$REPO" --json commits -q '.commits[].messageHeadline'
gh pr view <number> -R "$REPO" --json commits -q '.commits[].messageBody' \
  | grep -inE 'co-authored-by|generated with|claude'
```

The last one says whether GitHub has an AI trailer to harvest at merge time —
step 5's reason for `--body-file`, and what step 6 checks against.

## 4. Rewrite title and body

Apply the `/github-pr` title and body rules in full. The ones that PRs
authored elsewhere most often violate:

- **Title** — `Type: description`, under 72 characters, one of the seven
  `/git-commit` types. A conventional-commit title (`fix(qwen3): ...`) always
  gets rewritten. Write it **without** a `(#N)` suffix; step 5 appends that.
- **No markdown headings at all** — `## Summary`, `## Validation`,
  `## Test plan`, `## Related Issues` all go. Fold anything worth keeping into
  bullets or a trailing paragraph.
- Plain `-` bullets wrapped at 72 characters, one body for the squashed whole,
  every bullet verifiable from the diff.
- Drop process prose: pre-commit runs, CI job names, compilation paths tried,
  reviewer round-trips.
- Rationale that survives is a trailing paragraph, not a section. A cross-repo
  reference (`hw-native-sys/pypto#2273`) belongs inline in that paragraph; only
  a real issue this PR closes gets a bare trailing `Fixes #123`.
- **Strip AI-authorship trailers and branding.** `Co-Authored-By: Claude`,
  any `Co-authored-by:` naming an assistant rather than a person,
  `Generated with Claude Code`, and 🤖 lines. Step 5 lands this body in the
  commit as written, so anything left here shows up in `git log` on `main`. A
  `Co-authored-by:` naming a real human stays.

Write the body to a file. Step 5 has to hand `gh pr merge` the same text, and
a file is the only way both commands get it byte-identical:

```bash
BODY=$(mktemp)   # keep this file — step 5 merges with it
cat > "$BODY" <<'EOF'
- Key change 1
- Key change 2
EOF
gh pr edit <number> -R "$REPO" --title "Type: concise description" --body-file "$BODY"
```

Skip the edit when the title and body already conform; say so rather than
rewording for its own sake. Still produce the file — step 5 needs it either
way:

```bash
BODY=$(mktemp)
gh pr view <number> -R "$REPO" --json body -q .body > "$BODY"
```

## 5. Squash-merge

**Pass `--body-file`. Never pass `--subject`.**

The `PR_BODY` default is not the PR body. GitHub harvests every
`Co-authored-by:` trailer out of the head branch's commits and appends it to
the squash message below the body (after a `---------` separator when the PR
has several commits) — and step 4 cannot prevent it, because the PR body and
the squash commit message are different texts. #1126 was merged with a bare
`gh pr merge --squash` after a clean `gh pr edit`, and
`Co-authored-by: Claude Opus 5 ...` is on `main` permanently.

`--body-file` sets the commit body explicitly, and the harvested trailer then
does not appear. Confirmed with a matched pair of squash merges — one branch
commit carrying a `Co-authored-by:` trailer, `PR_TITLE` / `PR_BODY` defaults,
merged both ways: the bare merge landed the trailer, the `--body-file` merge
landed exactly the file's text plus one `(#N)`. A landed `--body-file` merge
alone proves nothing unless the branch commits actually carried a trailer;
check them before believing a clean result.

Leave the subject on the `PR_TITLE` default. Overriding it suppresses the
automatic `(#N)` and makes it your job to re-append it, which is the only way
`... (#942) (#942)` ever reaches `main`.

A `Co-authored-by:` naming a real human is worth keeping: write it into the
body file yourself, as a last line after a blank line.

`--match-head-commit` pins the merge to the commit step 2 was checked against,
so a push landing in between aborts the merge instead of silently merging
unreviewed, unchecked work.

```bash
HEAD_SHA=$(gh pr view <number> -R "$REPO" --json headRefOid -q .headRefOid)
gh pr merge <number> -R "$REPO" --squash --body-file "$BODY" --match-head-commit "$HEAD_SHA"
```

A failed `gh pr merge` is not proof the merge did not happen — the GraphQL call
can time out after the merge has landed. Re-read the state before retrying; a
blind retry against an already-merged PR fails with an unrelated-looking error.

```bash
gh pr view <number> -R "$REPO" --json state,mergedAt -q '"\(.state) \(.mergedAt)"'
```

Never `--merge` or `--rebase`. **Never `--delete-branch`, and never delete the
branch by hand afterwards — neither the remote one nor the local one.** The
upstream repo's `delete_branch_on_merge` does not reach a fork, so a PR raised
from `origin` keeps its head branch after the merge; that is the intended end
state, not something to tidy up.

## 6. Verify

```bash
gh pr view <number> -R "$REPO" --json state,mergedAt,title
git fetch upstream 2>/dev/null || git fetch origin
git log upstream/main -1 --format='%s%n%b'   # origin/main on a single-remote clone
git log upstream/main -1 --format='%b' | grep -nE 'Co-authored-by|Generated with|Claude|^-{5,}$'
```

`upstream` is the real repo when the clone has both remotes; `origin` is then a
personal fork and its `main` lags. Confirm the landed subject and body match
the file from step 4 — one `(#N)`, and the `grep` printing nothing but a real
human `Co-authored-by:`. This is the last chance to see a trailer that slipped
through, and `main` cannot be force-pushed, so it stays: report it rather than
attempting a fix.

## Reporting

State the old title and the new one, what was cut from the body and why, the
red `sim` jobs the merge went in over, any other non-passing check, any
unresolved review feedback found in step 2, and the merged commit — including
that its message carries no AI trailer. Do not
restate the body rules back to the user.

With several PRs, report one row per PR — number, landed subject, and the
status (merged / blocked and why) — after the last one finishes.
