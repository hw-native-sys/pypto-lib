# PyPTO-Lib Agent Instructions

PyPTO-Lib provides PyPTO kernels and end-to-end LLM models for Ascend NPUs,
plus the `golden/` harness for compilation, execution, and Torch validation.
This file is the shared project entry point; `AGENTS.md` adapts it for Codex.

## Working agreements

- Keep changes scoped to the requested kernel, model, test, or documentation
  area. Prefer existing project patterns when they agree with current rules.
- `tests/` primarily tests the `golden` harness. `tests/contract/` tests model
  integration with serving, not model computation or implementation structure.
  Model development should leave these tests unchanged unless the task
  specifically requires harness or serving-contract test changes.
- Keep public documentation and examples aligned when behavior changes.
- Keep agent constraints in `.claude/rules/`, task execution workflows in
  `.claude/skills/`, and public technical guidance in `docs/`. Link to the
  owning source instead of maintaining duplicate instructions or technical
  explanations.
- Do not commit generated artifacts from `build_output/`.
- Treat credentials, local paths with usernames, and machine-specific state
  as off-limits unless the user explicitly asks for them. Do not include
  private information in code or documentation.
- Write code comments and documentation in English unless the user explicitly
  requests otherwise. Follow the [language policy](rules/language-policy.md).
- For review-only tasks, inspect and report findings without modifying files
  unless the user also requests fixes. Apply referenced workflows as review
  criteria rather than executing their editing steps.

## Read before the relevant task

Read the applicable guidance before editing or executing the task. Follow a
matching skill in `.claude/skills/` when the task has a documented workflow;
the table below lists common entry points, not an exhaustive skill catalog.

| Task | Required guidance and workflow |
| --- | --- |
| Model implementation, optimization, or review | [Model rules](rules/models.md) |
| Writing or modifying kernels | [L2 programming](../docs/pypto-coding/l2-programming.md), relevant sibling pages in [PyPTO coding](../docs/pypto-coding/index.md), and [formatting workflow](skills/fmt-coding-style/SKILL.md) |
| Environment setup | [Setup skill](skills/setup-env/SKILL.md) and [installation guide](../docs/get-started/installation.md) |
| Running a model | [Model case selection](skills/run-model-cases/SKILL.md) |
| Harness or validation changes | [Golden harness](../docs/run-and-validate/golden-harness.md) and [compile/runtime workflow](../docs/run-and-validate/compile-runtime-workflow.md) |
| Performance measurement or tuning | [Benchmarking rules](rules/benchmarking.md) and [performance guide](../docs/debug-and-tune/performance-tuning.md); use the matching profiling or tuning skill |
| Cube tile tuning | [Cube tile tuning skill](skills/cube-tile-tuning/SKILL.md) |
| In-core profiling | [In-core profiling skill](skills/incore-profiling/SKILL.md) |
| Precision debugging | [Precision guide](../docs/debug-and-tune/precision-tuning.md); use [precision bisect](skills/bisect-precision/SKILL.md) for a reproducible revision regression |
| Compile/runtime failures or suspected upstream bugs | [Problem handling rules](rules/problem-handling.md) and [debugging guide](../docs/debug-and-tune/debugging.md) |
| Commit, PR, review fixes, or issue creation | [Commit](skills/git-commit/SKILL.md), [PR](skills/github-pr/SKILL.md), [PR fixes](skills/fix-pr/SKILL.md), or [issue](skills/create-issue/SKILL.md) workflow |

For routine kernel changes, apply the formatting rules to new or modified
code. Run a whole-file style pass only when requested or explicitly required
by the active workflow; avoid unrelated formatting changes.

Never silently work around a suspected upstream bug: follow the problem
handling rules to record it and handle blocking failures. For performance
work, optimize device wall time first; core busy time explains the result.
Follow the benchmarking rules for replay, sampling, and distributed metrics.

## Repository map

- `examples/{beginner,intermediate,advanced}/`: self-contained DSL examples.
- `models/<family>_<version>[_<quant>]/`: one directory per model build;
  see [model documentation](../docs/models/index.md) for each implementation.
- `golden/`: compile/run/validate harness against Torch references.
- `tests/`: unit tests, model interface contracts, and existing lint checks.
- `docs/`: canonical public technical guidance.
- `.claude/rules/`: agent constraints; `.claude/skills/`: task workflows.
- `build_output/`: generated compilation artifacts and execution output.

Files ending in `_draft.py` are works in progress and excluded from CI.

## Toolchain and execution

The selected PyPTO checkout is the version source of truth: its `runtime/`
gitlink pins simpler, `toolchain/versions.env` pins PTOAS, and
`runtime/pto_isa.pin` pins PTO ISA. Follow the installation guide and setup
skill when preparing the environment.

Platform and device arguments are entry-specific. Inspect the selected
script's `--help` and the [platform guide](../docs/get-started/first-kernel.md)
before running it. Common commands from the repository root:

```bash
python examples/beginner/hello_world.py -p a2a3 --sim
python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0
python -m pytest tests/golden -v
python tests/lint/check_headers.py
python tests/lint/check_english_only.py
ruff check .
```

Choose validation appropriate to the changed area; this command list is not
an instruction to run every check for every task. Report what was validated
and any checks that were not run.
