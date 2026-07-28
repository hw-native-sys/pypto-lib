---
name: github-pr
description: Create or update a GitHub pull request after committing and pushing changes. Use when the user asks to create a PR, submit changes for review, or open a pull request.
---

# GitHub Pull Request Workflow

## 1. Check state

```bash
BRANCH_NAME=$(git branch --show-current)
git status --porcelain
git fetch upstream 2>/dev/null || git fetch origin
if git rev-parse --verify upstream/main >/dev/null 2>&1; then
  BASE_REF=upstream/main
else
  BASE_REF=origin/main
fi
git rev-list HEAD --not "$BASE_REF" --count
```

A branch is **effectively on main** when its name is `main`/`master` **or** it has zero commits
ahead of `$BASE_REF`.

| Effectively on main? | Uncommitted? | Action |
| -------------------- | ------------ | ------ |
| Yes | Yes | New branch, commit via `/git-commit`, then PR |
| Yes | No | Error — nothing to PR |
| No | Yes | Commit via `/git-commit` on this branch, then PR |
| No | No | Already committed — push and PR |

## 2. Branch (if needed)

The name comes from the commit subject, which does not exist yet when the work is uncommitted:
decide the `Type: subject` from the staged diff first (same rules as `/git-commit`), branch on it,
then commit. Do NOT ask the user.

| Commit type | `Add:` | `Fix:` | `Perf:` | `Refactor:` | `Docs:` | `CI:` | `Chore:` |
| ----------- | ------ | ------ | ------- | ----------- | ------- | ----- | -------- |
| Prefix | `feat/` | `fix/` | `perf/` | `refactor/` | `docs/` | `ci/` | `chore/` |

Slug = subject after `Type: `, lowercased, non-alphanumerics to hyphens, trailing hyphens
stripped, truncated to 50 characters.

```bash
git checkout -b <prefix><slug>
```

## 3. Existing PR?

```bash
gh pr list --head "$BRANCH_NAME" --state open
```

If one exists, show it with `gh pr view` and exit.

## 4. Rebase and push

```bash
git rebase "$BASE_REF"
git push --set-upstream origin "$BRANCH_NAME"      # first push
git push --force-with-lease origin "$BRANCH_NAME"  # after a rebase — never --force
```

On conflicts: resolve, `git add <file>`, `git rebase --continue`; `git rebase --abort` if stuck.

## 5. Create PR

```bash
gh auth status
```

If `gh` is missing or unauthenticated, do not install it — give the user the manual URL
`https://github.com/hw-native-sys/pypto-lib/compare/main...$BRANCH_NAME` and stop.

Read the PR's exact scope first; the body describes this and nothing else:

```bash
git log "$BASE_REF"..HEAD --format='%B'   # commits this PR adds
git diff "$BASE_REF"...HEAD --stat        # files it touches
```

```bash
gh pr create \
  --base main \
  --head "$BRANCH_NAME" \
  --title "Type: concise description" \
  --body "$(cat <<'EOF'
- Key change 1
- Key change 2
EOF
)"
```

**Title** — the commit subject verbatim: `Type: description`, under 72 characters, one of the
seven types in `/git-commit`. It becomes the squash-merge subject on `main`, so never the
conventional-commit `feat(scope):` form.

**Body** — the repo squash-merges, so this text lands verbatim in `git log` on `main`. Write it as
a commit body, not as a web page:

- One commit body for the **squashed whole**, not a concatenation of the per-commit bodies. A
  single-commit PR reuses that commit's body verbatim.
- With several commits, describe the net `git diff "$BASE_REF"...HEAD` — the end state a reader of
  `main` sees. Merge bullets that touch the same thing, and drop any step the later commits undid:
  "add X" + "fix X" + "rename X to Y" is one bullet about Y, never three. A bullet that only makes
  sense as history ("revert the earlier attempt", "address review feedback") does not belong.
- Derived **only** from `$BASE_REF..HEAD` — never from the conversation that produced it, from
  commits already on `main`, or from tool output.
- Plain `-` bullets wrapped at 72 characters. **No markdown headings at all** — no `## Summary`,
  no `## Test plan`, `## Testing`, `## Validation`, `## Notes`, or `## Related Issues`.
- A `Perf:` PR ends with the measured before -> after and the configuration measured on, as a
  trailing sentence rather than a section.
- Link an issue with a bare trailing `Fixes #123`, and only when a real issue number exists. Omit
  the line otherwise — never a heading followed by `None` / `N/A` / `(if applicable)`.
- Every bullet verifiable from the diff. No lint / `pre-commit` runs, syntax checks, commands
  invoked, files read, debugging detours, irreproducible measurements, or follow-up ideas.
- No AI co-author footers or branding.
