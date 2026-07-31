# Model Support Matrix

The `models/` tree contains full-forward experiments, layer and operator
harnesses, imported implementation modules, and work-in-progress files. A
Python filename by itself is not a support claim.

This matrix is derived from tracked source files and from the repository's
[regular CI workflow](../../.github/workflows/ci.yml) and
[daily model workflow](../../.github/workflows/daily_ci.yml).

## Status vocabulary

- **Full forward** implements a multi-layer prefill or decode path. Its CLI may
  still default to a smaller golden fixture; read the family page before
  running it.
- **Component harness** has a `__main__` entry point and validates an operator,
  attention path, layer, or split of a model.
- **Regeneration-only** has a CLI but produces source artifacts rather than
  validating a runtime model path.
- **Library module** has no `__main__` entry point and is exercised only through
  an importing harness.
- **Draft** is named `*_draft.py`, is work in progress, and is excluded from
  model CI sweeps.

Platform reporting also has two separate meanings:

- **Declared platform** is accepted by that entry point's CLI.
- **CI coverage** is a platform on which a workflow schedules the entry point.
  It records the configured coverage, not the result of the latest run.

## Family summary

Counts below include tracked Python files only.

| Family | Runnable harnesses | Drafts | Other modules | Implemented scope | Configured CI coverage |
| --- | ---: | ---: | ---: | --- | --- |
| [Qwen3-14B](qwen3.md#qwen3-14b) | 6 runtime + 1 regeneration | 3 | 8 | BF16 prefill/decode, A8W8 decode component, sampling and attention components | A2/A3 for all non-draft CLIs; A2/A3 sim and A5 sim for three generic components; one-card A2/A3 serving job on relevant PRs |
| [Qwen3-32B](qwen3.md#qwen3-32b) | 2 | 1 | 0 | Two single-layer decode layouts | A2/A3, A2/A3 sim, A5 sim |
| [DeepSeek V3.2-EXP](deepseek.md#deepseek-v32-exp) | 3 | 0 | 0 | Decode front/back and reduced prefill-back components | A2/A3, A2/A3 sim, A5 sim |
| [DeepSeek V4 Flash](deepseek.md#deepseek-v4) | 37 | 0 | 5 | Prefill/decode forward experiments plus layer, attention, MoE, MTP, and cache components | A2/A3 for 37 CLIs; A2/A3 sim and A5 sim for 34 non-device-only CLIs |
| [DeepSeek V4 Pro](deepseek.md#deepseek-v4) | 35 | 0 | 3 | Prefill/decode forward experiments plus layer, attention, MoE, MTP, and cache components | A5 daily job is configured for 35 CLIs; it is not part of the aggregate required summary |

The DeepSeek V4 PR path also triggers a shared eight-card serving accuracy job.
That integration test is separate from the per-file harness sweeps and should
not be read as evidence that every V4 component is an eight-card entry point.

## Important limits

- Qwen3-14B is the only Qwen variant with an external model contract and
  serving CI integration in this repository.
- Qwen3-32B provides single-layer decode harnesses, not an end-to-end model
  runner.
- DeepSeek V3.2-EXP contains split component harnesses, not a full model
  forward.
- DeepSeek V4 full-forward and distributed layer cases default to two ranks.
  Some CLIs accept EP sizes 2, 4, and 8, but the per-file CI marker covers the
  default two-rank case only.
- A platform accepted by a parser is not necessarily device-tested in CI. In
  particular, generic A5 declarations are often covered only by `a5sim`; the
  dedicated A5 device sweep targets DeepSeek V4 Pro.
- The repository does not include model checkpoints. Most direct script entry
  points use synthetic Golden Harness fixtures; serving CI obtains weights from
  the runner environment.
