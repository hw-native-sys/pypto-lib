---
name: github-pr
description: Create or update a GitHub pull request after committing and pushing changes. Use when the user asks to create a PR, submit changes for review, or open a pull request.
---

# GitHub Pull Request Workflow

## Task Tracking

Create tasks to track progress through this workflow:

1. Prepare branch & commit
2. Check for existing PR
3. Fetch upstream & rebase
4. Push to remote
5. Create PR

## Step 1: Check Current State

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

A branch "needs a new branch" when it is effectively on main — either the branch
name is `main`/`master`, **or** it has zero commits ahead of the base ref.

## Step 2: Route

| Needs new branch? | Uncommitted changes? | Action |
| ----------------- | -------------------- | ------ |
| Yes | Yes | Create new branch, commit via `/git-commit`, then create PR |
| Yes | No | Error — nothing to PR |
| No | Yes | Commit on current branch via `/git-commit`, then create PR |
| No | No | Already committed — proceed to push and create PR |

### Create Branch (if needed)

Auto-generate a branch name with a meaningful prefix. Do NOT ask the user.

| Prefix | Usage |
| ------ | ----- |
| `feat/` | new example or tensor function |
| `fix/` | bug fix |
| `docs/` | documentation changes |
| `ci/` | CI/CD changes |
| `refactor/` | restructuring |

```bash
git checkout -b <branch-name>
```

### Commit (if uncommitted changes)

Delegate to `/git-commit` skill.

## Step 3: Check for Existing PR

```bash
gh pr list --head "$BRANCH_NAME" --state open
```

If PR already exists, display with `gh pr view` and exit.

## Step 4: Rebase and Push

```bash
git fetch upstream 2>/dev/null || git fetch origin
git rebase "$BASE_REF"
```

**On conflicts**:

```bash
git status
# Edit files, remove markers
git add path/to/resolved/file
git rebase --continue
# If stuck: git rebase --abort
```

**Push**:

```bash
# First push
git push --set-upstream origin "$BRANCH_NAME"

# After rebase (use --force-with-lease, NOT --force)
git push --force-with-lease origin "$BRANCH_NAME"
```

## Step 5: Create PR

**Check gh CLI**:

```bash
gh auth status
```

**If gh NOT available or not authenticated**: Report to user and provide manual URL:
`https://github.com/hw-native-sys/pypto-lib/compare/main...BRANCH_NAME`

**If gh available**:

```bash
gh pr create \
  --title "Brief description of changes" \
  --body "$(cat <<'EOF'
## Summary
- Key change 1
- Key change 2

## Related Issues
Fixes #ISSUE_NUMBER (if applicable)
EOF
)"
```

**Rules**:
- Auto-generate title and body from commit messages
- Keep title under 72 characters
- Do NOT add AI co-author footers or branding

## Common Issues

| Issue | Solution |
| ----- | -------- |
| PR already exists | `gh pr view` then exit |
| Merge conflicts | Resolve, `git add`, `git rebase --continue` |
| Push rejected | `git push --force-with-lease` |
| gh not authenticated | Tell user to run `gh auth login` |
| Wrong upstream branch | Use `git rebase upstream/BRANCH` |

## Checklist

- [ ] Branch prepared (created from main if needed)
- [ ] Changes committed via `/git-commit`
- [ ] No existing PR for branch (exit if found)
- [ ] Fetched upstream and rebased successfully
- [ ] Pushed with `--force-with-lease`
- [ ] PR created with clear title and summary
- [ ] No AI co-author footers

## Remember

- Always rebase before creating PR
- Use `--force-with-lease`, not `--force`
- Don't auto-install gh CLI - let user do it
