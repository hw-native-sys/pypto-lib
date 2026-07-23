---
name: git-commit
description: Complete git commit workflow including pre-commit checks, staging, message generation, and verification. Use when creating commits or preparing changes for commit.
---

# Git Commit Workflow

## 1. Review

```bash
git diff --name-only   # what changed
git diff --staged      # review before committing
```

Check what the hooks cannot: `import pypto.language as pl` (not other aliases), correct
`pl.FunctionType` (InCore / Orchestration / Opaque), correct parameter directions (`pl.Out`,
`pl.InOut`), no hardcoded absolute paths or private information. See `docs/pypto-coding-style.md`.

If the change touches a kernel under `examples/` or `models/`, run it once before committing.
Docs- or config-only changes need no run.

## 2. Hooks

```bash
pre-commit run --all-files
```

| Hook | What it checks |
|------|----------------|
| `ruff-check` | Python linting — runs with `--fix`, so re-stage what it rewrites |
| `check-headers` | Copyright header format |
| `check-english-only` | Code comments and docstrings are in English |

**Never bypass.** No `--no-verify`, no `SKIP=...`. Any file the commit touches must leave it with
zero violations, including violations that predate the change.

## 3. Stage

Stage related changes together. Never stage build artifacts (`build_output/`, `__pycache__/`, `*.so`).

```bash
git add path/to/file1.py path/to/file2.py
```

## 4. Message

### Subject

`Type: concise description` — under 72 characters, imperative mood, no period.

**These seven types and no others**:

| Type | Usage |
| ---- | ----- |
| **Add** | new example, model, tensor function, or capability on an existing one |
| **Fix** | bug fix — correctness, precision, hang, or build failure |
| **Perf** | speed / memory optimization with unchanged numerics |
| **Refactor** | restructuring without behavior change |
| **Docs** | documentation changes |
| **CI** | CI/CD pipeline changes |
| **Chore** | config, gitignore, tooling |

### Body

Required for multi-file changes, optional for one-liners. Blank line after the subject, wrapped at
72 characters, bullets for multiple items. Explain **what** changed and **why**.

**The body describes the staged diff, not the session that produced it** — it becomes the PR body
and, after squash-merge, the permanent `git log` entry. Every line must be verifiable from
`git diff --staged`. Never include: lint / `ruff` / `pre-commit` / syntax-compile runs, commands
invoked, files read, the debugging path taken, measurements a reviewer cannot reproduce, or
follow-up ideas absent from the diff.

A `Perf:` body states the measured before -> after **and** the configuration measured on.

**Good**:

```text
Add: Qwen3-32B single-layer decode example

- Batch=16 with per-session variable context length
- Fused outer loops for attention and MLP
- All GM slices >= 512B alignment
```

```text
Perf: fold RoPE into DeepSeek V4 gate FFN norm

- Drop the fp32 widen buffer; quantize straight from the norm output
- Fan the gate matmul over experts so GATE_N_TILE shrinks 16 -> 1

Isolated gate latency 55 -> 46 us (decode, ep2, a2a3).
```

**Bad**:

```text
x  Added new example.                # Past tense, has period
x  fix bug / WIP                     # Lowercase type, not descriptive
x  Update: tune MoE tile sizes       # 'Update' is not a type -- use Perf
x  feat(dsv4): split v4 into two     # Conventional style -- use 'Add: ...'
x  Perf: speed up SWA decode         # No before -> after number
x  Fix: hc_pre scratch shapes        # ...body listing "ran ruff, ran pre-commit"
```

### Co-authors

Never credit AI assistants. Human contributors only: `Co-authored-by: Name <email>`.

## 5. Verify

```bash
git log -1 && git show HEAD --stat
```

Amend only if not yet pushed: `git commit --amend`.
