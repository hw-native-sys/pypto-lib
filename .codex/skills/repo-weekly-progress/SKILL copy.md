---
name: repo-weekly-progress
description: Generate AI-written weekly progress reports for GitHub repository hw-native-sys/pypto-lib by inspecting commits, PRs, issues, and targeted diffs on demand; reports are organized by model, real operator/module impact, and change type.
---

# Repo Weekly Progress

Use this skill to track weekly progress for the GitHub repository
`hw-native-sys/pypto-lib` and generate a Markdown report.

Codex should write the report by directly inspecting the repository, PRs,
issues, and targeted diffs. Do **not** rely on a script-generated report as the
source of truth. The bundled script is optional and may be used only as a rough
evidence cache; it is not authoritative and its classifications must be checked
against actual model-specific diffs.

## Defaults

- Repository: `hw-native-sys/pypto-lib`
- Statistics window: the most recent 7 days
- Output format: Markdown
- Default output path: `notes/weekly-YYYY-MM-DD-to-YYYY-MM-DD.md`
- Evidence sources: one-week before/after git diff, commits, PRs, and issues

## Report Format

Write the report in Chinese with this explicit structure:

1. Title: `# PyPTO-Lib 周报：YYYY-MM-DD ~ YYYY-MM-DD（模型与算子进展）`.
2. Opening blockquote with scope, filtering criteria, commit counts, changed-file
   counts, PR/issue counts, and whether GitHub evidence was collected.
3. `## 概览`: a table with `Commit`, `PR`, `作者`, `主题`, `模型 / 范围`, and `类型`.
4. `## Owner 索引`: a table with owner, commit count, and covered model/range.
5. `## 算子总体改动`: the main reporting section. Group it by
   `model/range -> operator/module`, not by operator alone. Only include an
   operator/module under a model when that model's own changed files, functions,
   or PR diff directly support it. For each model and operator/module pair,
   classify changes into the four main types: `性能优化`, `新实现`, `bugfix`, and
   `重构`, and include a `汇报要点` that explains the meaning of the changes.
6. `## 模型主线`: a short bullet list of the main model/range buckets.
7. Numbered model/range sections such as `## 1. DeepSeek V4 / DSV4`,
   `## 2. Qwen3`, and `## 3. Cross-Model / Infrastructure`.
8. Each model/range section should use the hierarchy
   `model/range -> phase -> operator/module -> change type`. For example:
   `Qwen3 -> Decode -> Attention -> 性能优化`, or
   `DeepSeek V4 / DSV4 -> Prefill -> Compressor -> 新实现`.
9. Each model/range section should include summary, evidence, touched files,
   changed functions or entry points, related issues, and numbered PR/commit
   subsections when there are concrete commits.
10. `## 关注项`: open issues or unresolved PRs worth tracking.
11. Appendices for diff summary, all commits, PRs, and issues.

## Report Principle

The main reporting narrative should start from **operator/module changes grouped
by model**, because this is the most useful view for weekly progress reporting.
Detailed evidence sections may still use model -> phase (`Decode`, `Prefill`,
`MTP`, `MoE / Router`, `Shared / Config`, etc.) -> operator/module -> change
type. Do not lead with a flat list of PRs, issues, or commits. PRs, issues, and
commits are evidence, not the top-level structure.

Avoid inferred or broadcast classifications. A model/operator claim is valid
only if the relevant model-specific diff supports it. For example, do not report
`Qwen3 -> Indexer` just because a multi-model commit or title mentions indexer;
Qwen3 must have changed indexer-related files, symbols, or PR diff content.

For each meaningful model/range and operator summary, explain:

1. What changed during the window.
2. Which files, paths, functions, kernels, or entry points changed.
3. Why the changes matter for model execution, validation, performance,
   maintainability, or correctness.
4. Which PRs, issues, or commits support the claim.

## Primary Model / Scope Taxonomy

Use these top-level buckets for the numbered body sections:

- DeepSeek V4 / DSV4
- DeepSeek V3.2
- Qwen3
- Kimi / K2
- MILM
- Cross-Model / Infrastructure
- Other Models

## Operator Taxonomy

Use these labels for `## 算子总体改动` and per-PR analysis when applicable:

- Sparse/SWA/CSA Attention
- Compressor
- Decode
- Prefill
- Indexer
- Cache / Paged Metadata
- RMSNorm / RoPE
- MoE / Router
- Dynamic Shape / Auto Chunk

## Required Workflow

1. Before collecting evidence, update the local repository so the report is
   based on the latest remote state. Prefer `git status --short` first, then
   `git pull --ff-only` when the working tree state allows it. If local changes
   are present, do not overwrite them; either ask the user or continue with a
   clear note that the report used the current local checkout.
2. Establish the time window. Default to the last 7 days unless the user gives
   explicit dates.
3. Collect a broad evidence index:
   - `git log --since=<since> --until=<until> --oneline --decorate`
   - `git diff <base>..<head> --stat`
   - `git diff <base>..<head> --name-only`
   - `gh search prs repo:hw-native-sys/pypto-lib 'updated:>=YYYY-MM-DD'`
   - `gh search issues repo:hw-native-sys/pypto-lib 'updated:>=YYYY-MM-DD'`
4. Identify likely reporting themes manually from paths and titles. Start with
   model directories, especially `models/deepseek/v4/` and `models/qwen3/`.
5. For each candidate theme, inspect targeted evidence:
   - Use `git show --stat <sha>` and `git show -- <path>` for relevant commits.
   - Use `gh pr view <number>` and `gh pr diff <number> --name-only` for PRs
     whose commit evidence is insufficient or not merged locally.
   - Open only the file diffs needed to understand meaning; avoid dumping huge
     diffs into the report.
6. Classify evidence hierarchically:
   - First by model/range.
   - Then by phase: `Decode`, `Prefill`, `MTP`, `MoE / Router`,
     `Shared / Config`, `Validation / Workflow`.
   - Then by actual operator/module found in that model's changed files.
   - Then by change type: `性能优化`, `新实现`, `bugfix`, `重构`.
7. Write `## 算子总体改动` as the main report section. It should be a concise
   narrative or table by `model -> operator/module`; every row must be supported
   by model-specific file paths, functions, or PR diff evidence.
8. Write model detail sections only for evidence that matters. Omit weak
   low-signal entries rather than forcing every changed file into the report.
9. Keep raw PR/issue/commit lists in appendices only.

## Optional Script

`scripts/generate_report.py` exists only as an optional evidence-cache helper.
Use it when a quick JSON/Markdown scaffold is useful, but never accept its
classification without checking targeted diffs. Prefer direct AI-led inspection
for final reports.

## Quality Bar

- Lead with a concise scope note, overview table, and owner index.
- Avoid claiming progress from a PR title alone; anchor claims to changed paths,
  functions, entry points, or diff size where possible.
- The operator summary must be directly usable for a weekly status report:
  explain the meaning, not just changed files.
- Avoid a global operator-only table because it mixes models. Use model ->
  operator/module grouping.
- In model sections, preserve hierarchy: model -> phase -> operator/module ->
  change type.
- Separate merged or completed work from exploratory, draft, or unresolved work.
- Include a final appendix with PRs, issues, commits, and the diff summary.
- If a model/range has no concrete diff evidence, either omit it or mark it as
  "watching" with the reason.
