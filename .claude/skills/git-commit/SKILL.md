---
name: git-commit
description: Complete git commit workflow including pre-commit checks, staging, message generation, and verification. Use when creating commits or preparing changes for commit.
---

# Git Commit Workflow

## Prerequisites

Check what changed to determine review needs:

```bash
git diff --name-only
git diff --cached --name-only
```

| File Types Changed | Run Example Validation |
| ------------------ | ---------------------- |
| Python (`.py`) in `examples/` or `projects/` | Yes — run a related example |
| Docs only (`.md`) | Skip |
| Config only (`.yml`, `.json`, `.gitignore`) | Skip |

## Pre-Commit Review

Before committing, review changes against the coding style:

```bash
git diff --staged
```

Check for:
- `import pypto.language as pl` (not other aliases)
- Correct `pl.FunctionType` usage (InCore / Orchestration / Opaque)
- Proper parameter directions (`pl.Out`, `pl.InOut`)
- No hardcoded absolute paths or private information
- Comments and docstrings in English

## Stage Changes

Stage related changes together. Never stage build artifacts (`build_output/`, `__pycache__/`, `*.so`).

```bash
git add path/to/file1.py path/to/file2.py
git diff --staged  # Review before committing
```

## Commit Message Format

### Subject Line

`Type: concise description` (under 72 characters, imperative mood, no period)

**Types**:

| Type | Usage |
| ---- | ----- |
| **Add** | new example, model, or tensor function |
| **Fix** | bug fix |
| **Update** | enhancement to existing example or function |
| **Refactor** | restructuring without behavior change |
| **Docs** | documentation changes |
| **CI** | CI/CD pipeline changes |
| **Chore** | config, gitignore, tooling |

### Body (required for multi-file changes)

Separate from subject by a blank line. Explain **what** changed and **why**. Use bullet points for multiple items. Wrap at 72 characters.

**Good examples**:

```text
Add: Qwen3-32B single-layer decode example

- Batch=16 with per-session variable context length
- Fused outer loops for attention and MLP
- All GM slices >= 512B alignment
```

```text
Fix: softmax numerics in paged attention example

Row-max subtraction was applied after exponentiation,
causing overflow for large logits.
```

**Simple changes (body optional)**:

```text
Docs: clarify incore scope memory constraints
```

## Co-Author Policy

**Never** add AI co-author lines. Commits reflect human authorship only.

## Post-Commit Verification

```bash
git log -1              # Check message
git show HEAD --stat    # Verify staged files
```

## Checklist

- [ ] Only relevant files staged (no build artifacts)
- [ ] Code follows `pypto-frontend-coding-style.md` conventions
- [ ] No hardcoded paths or private information
- [ ] Message format: `Type: description` (under 72 chars, imperative, no period)
- [ ] Body included for multi-file changes (what + why)
- [ ] No AI co-authors
