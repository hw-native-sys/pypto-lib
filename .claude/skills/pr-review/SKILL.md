---
name: pr-review
description: Review or re-review pypto-lib model development and optimization PRs against models.md. Use for /pr-review with a model PR number or URL, model-rule compliance checks, or a fresh look after a model PR update.
---

# Review a pypto-lib model PR

Use the [project working agreements](../../CLAUDE.md) throughout. Keep
findings in the conversation unless the user asks to post them or take
further action.

## Establish the revision

Resolve the PR in its target repository. For a number, use the upstream
pypto-lib repository. For a URL, use its repository. For example:

```sh
gh pr view 1937 -R hw-native-sys/pypto-lib --json title,body,baseRefOid,headRefOid,files
gh pr diff 1937 -R hw-native-sys/pypto-lib
```

Inspect the merge-base-to-head diff and read callers and surrounding code at
that head. On a re-review, recheck earlier findings against the new revision
and inspect subsequent changes. Identify which findings still apply.

## Model development and optimization

Read the [model rules](../../rules/models.md) and each affected model's
`docs/models/<model>/index.md`. Review the change against its stated goal,
performance targets and supported configurations. Apply linked coding-style
guidance as review criteria.

- Follow a changed interface from its real callers into the kernel. For a
  new mode or parameter, identify which callers need a different computation
  or schedule and which merely forward it. For a new `TaskId`, trace the
  producer and consumer through tensor reads, writes and intermediate tasks.
- For shape-dependent changes, distinguish configured capacity, live work and
  internal padding. Work through the tiling and SPMD launch for relevant
  shapes on supported targets, including where the shape crosses a branch or
  the work exceeds the available cores. For a specialized fast path, compare
  its claimed range with the measurements. Use the
  [benchmarking rules](../../rules/benchmarking.md) to interpret timing data.
- Follow changed operators, imports and wrappers far enough to establish who
  owns the computation and whether affected callers migrated. Apply the
  file ownership, composition, validation and CI rules to the relevant files
  in the diff.
- Trace the executable validation from its inputs to the affected output or
  state. Identify a concrete wrong result the comparison would catch or miss.
  Check which paths the reported CI jobs or author runs actually exercised.
  Use a focused local check when it would resolve a material uncertainty.
- If the model PR also edits CI or golden validation, ask what would fail
  or go untested with the existing setup. For CI, identify the affected case
  or platform. For a model's Torch golden, check which expected behavior
  changed and why. For shared `golden/` edits, identify the capability the
  model's own validation entry and existing harness cannot provide. Recommend
  dropping changes that the model work does not require.

## Report findings

Lead with the conclusion for the reviewed head. Order findings by impact and
link each to the relevant file and line at that revision. State the triggering
configuration, affected caller or rule, consequence, and smallest actionable
change. Distinguish demonstrated defects, gaps in verification and optional
cleanup when explaining what should block approval. If there are no actionable
findings, say so. Summarize evidence from CI, the author and your own checks,
including any material path that remains unverified.
