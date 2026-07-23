# PyPTO-Lib Codex Instructions

This repository keeps AI project policy in `.claude/`. Treat `.claude/` as the
authoritative source of truth; this file is only the Codex entrypoint.

## Read First

Before making code changes, reviewing code, committing changes, or working on
kernel behavior:

- Read `.claude/CLAUDE.md`
- Read task-relevant files in `.claude/rules/` when present
- Follow `.claude/skills/*/SKILL.md` when the task matches a documented workflow
- Read `docs/pypto-coding-style.md` before writing or modifying kernels

Task mapping:

- Environment setup: `.claude/skills/setup_env/SKILL.md`
- Precision debugging: `.claude/skills/bisect-precision/SKILL.md`
- Performance profiling: `.claude/skills/incore-profiling/SKILL.md`
- Cube tile tuning: `.claude/skills/cube-tile-tuning/SKILL.md`
- Commit workflow: `.claude/skills/git-commit/SKILL.md`
- PR workflow: `.claude/skills/github-pr/SKILL.md`
- PR review fixes: `.claude/skills/fix-pr/SKILL.md`
- Issue creation: `.claude/skills/create-issue/SKILL.md`
- AscendC workflows: `.claude/skills/cannbot-skills/*/SKILL.md`

When a Claude skill or agent refers to `Task`, a subagent, or Claude-only
plugins:

- Execute the workflow directly in Codex
- Use parallel tool calls when safe
- Treat any agent-specific instructions as checklists, not as a separate runtime

## Working Agreements

- Keep changes scoped to the requested kernel, model, test, or documentation area
- Prefer existing project patterns and examples over new abstractions
- Keep public documentation and examples aligned when behavior changes
- Do not commit generated build artifacts from `build_output/`
- Treat credentials, local paths with usernames, and machine-specific state as
  off-limits unless the user explicitly asks for them
- Keep code comments and documentation in English unless the user explicitly
  requests otherwise

## Preferred Commands

```bash
# Run an example on the simulator
python examples/beginner/hello_world.py -p a2a3sim

# Run a model on real NPU device 0
python models/qwen3/14b/decode_fwd.py -p a2a3 -d 0

# Run golden harness unit tests
python -m pytest tests/golden -v

# Run repository lint checks
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
ruff check .
```

Every executable kernel or model script generally accepts
`-p {a2a3,a2a3sim,a5,a5sim}` and `-d <device_id>`.

## Repository Map

- `examples/`: self-contained kernels for learning and reference patterns
- `models/`: end-to-end LLM kernels organized by model family
- `golden/`: compile/run/validate harness against torch references
- `tests/`: lint checks and golden harness unit tests
- `docs/`: coding style, compile/runtime workflow, performance tuning,
  precision tuning, and debugging references
- `.claude/skills/`: task-specific workflows shared with Claude Code
