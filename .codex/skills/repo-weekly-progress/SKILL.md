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
- Default output path: `notes/weekly-YYYY-MM-DD-to-YYYY-MM-DD.md` using UTC dates for the start and end of the window (inclusive)
- Evidence sources: commits, PRs, and issues whose timestamps fall within the window `[since, until]` (default most recent 7 days); diffs from `git diff <commit_at_since>..<commit_at_until>` where `<commit_at_since>` is the repository HEAD at the start of the window and `<commit_at_until>` is the repository HEAD at the end of the window

## Report Format

Write the report in Chinese with this explicit structure:

1. Title: `# PyPTO-Lib 周报：YYYY-MM-DD ~ YYYY-MM-DD（模型与算子进展）`.
2. Opening blockquote with scope, filtering criteria, commit counts, changed-file
   counts, PR/issue counts, and whether GitHub evidence was collected.
   Example:
   > Window: 2026-05-22 to 2026-05-29 UTC
   > Evidence: 12 commits, 3 PRs, 1 issue, GitHub evidence collected
   > Scope: models/qwen3, models/deepseek/v4
3. `## 概览`: a table with `Commit`, `PR`, `作者`, `主题`, `模型 / 范围`, and `类型`.
   Example:
   | Commit | PR | 作者 | 主题 | 模型 / 范围 | 类型 |
   | --- | --- | --- | --- | --- | --- |
   | abc123 | #456 | Alice | DSV4 Compressor 新实现 | DeepSeek V4 / Prefill | 新实现 |
4. `## Owner 索引`: a table with owner, commit count, and covered model/range.
5. `## 算子总体改动`: the main reporting section. Group it by
   `model/range -> operator/module`, not by operator alone. Include an
   operator/module under a model only if one or more of the following hold:
   (a) at least one changed file path is inside the model's directory
   (e.g. `models/<model>/...`), (b) a changed file modifies functions or
   symbols whose names match the operator/module, or (c) a PR diff includes
   changes to files or tests that reference the operator/module. Otherwise mark
   the model or operator as `watching`. For each model and operator/module pair,
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
12. Render every PR and issue reference as a Markdown link. Use
    `https://github.com/hw-native-sys/pypto-lib/pull/<number>` for PRs and
    `https://github.com/hw-native-sys/pypto-lib/issues/<number>` for issues,
    including overview tables, evidence bullets, focus items, and appendices.

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
Ignore changes to vendored/generated files, build artifacts, generated code, or
binaries for operator-level classification unless they include source-code
changes affecting operators. Treat paths such as `third_party/`, `build/`,
`dist/`, `generated/`, or `vendor/` as non-authoritative evidence unless
manually whitelisted.

For each meaningful model/range and operator summary, explain:

1. What changed during the window.
2. Which files, paths, functions, kernels, or entry points changed.
3. Why the changes matter for model execution, validation, performance,
   maintainability, or correctness.
4. Which PRs, issues, or commits support the claim.

For every item classified as `性能优化`, inspect the PR body and/or commit
message for optimization effect, before/after latency, throughput, task count,
memory footprint, or other performance data. Report the concrete numbers when
available. If no public before/after data is present, explicitly mark the item
as "未提供公开性能数据" and describe only the structural optimization and
validation evidence.

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

1. Before collecting evidence, confirm the local repository state. Use
   `git status --short`; then attempt `git pull --ff-only` to sync with remote,
   even if current worktree is not clean. If `git pull --ff-only` fails,
   record the git error text in the opening blockquote, do not attempt a
   non-fast-forward merge, and proceed using the current local HEAD as the
   report head commit. If local uncommitted changes are present, do not
   overwrite them. If the user is unavailable, proceed with the current checkout
   and prefix the report with:
   `NOTE: local uncommitted changes present; report based on working tree at
   <timestamp>`.
2. Establish the time window. Default to the most recent 7 days unless the user
   gives explicit dates. Interpret explicit dates as `YYYY-MM-DD` in UTC. If the
   user provides a timezone, convert it to UTC and document the timezone used.
3. Collect a broad evidence index:
   - Define `<commit_at_since>` as the commit at the start of the window on the
     repository's default branch (or the merge-base with the active feature
     branch), and `<commit_at_until>` as the commit at the end of the window.
     If multiple candidate commits exist, prefer commits reachable from
     `origin/main` and document the chosen SHAs in the report.
   - `git log --since=<since> --until=<until> --oneline --decorate`
   - `git diff <commit_at_since>..<commit_at_until> --stat`
   - `git diff <commit_at_since>..<commit_at_until> --name-only`
   - `gh search prs repo:hw-native-sys/pypto-lib 'updated:>=<since> updated:<=<until>'`
   - `gh search issues repo:hw-native-sys/pypto-lib 'updated:>=<since> updated:<=<until>'`
   - If the `gh` CLI is unavailable or authentication fails, fall back to the
     GitHub REST API where possible, or annotate missing PR/issue metadata and
     proceed using local git history only. If network access is unavailable,
     generate a local-only report and explicitly mark missing remote evidence.
4. Identify likely reporting themes manually from paths and titles. Start with
   model directories, especially `models/deepseek/v4/` and `models/qwen3/`.
5. For each candidate theme, inspect targeted evidence:
   - Use `git show --stat <sha>` and `git show -- <path>` for relevant commits.
   - Use `gh pr view <number>` and `gh pr diff <number> --name-only` for PRs
     whose commit evidence is insufficient or not merged locally.
   - If `gh pr view` fails, record the PR number and URL and continue using
     available commit SHAs. Mark PR evidence as `incomplete` in the appendices.
   - Include file diffs only when they touch relevant operators/functions. For
     diffs larger than 500 lines, include a 20-line context around changed
     functions and provide an overall line-count summary instead of the full
     diff.
6. Classify evidence hierarchically:
   - First by model/range.
   - Then by phase: `Decode`, `Prefill`, `MTP`, `MoE / Router`,
     `Shared / Config`, `Validation / Workflow`.
   - Then by actual operator/module found in that model's changed files.
   - Then by change type: `性能优化`, `新实现`, `bugfix`, `重构`.
   - If a change fits multiple categories, use precedence:
     `bugfix` > `新实现` > `性能优化` > `重构`. If ambiguous, include a brief
     justification sentence.
7. Write `## 算子总体改动` as the main report section. It should be a concise
   narrative or table by `model -> operator/module`; every row must be supported
   by model-specific file paths, functions, or PR diff evidence.
8. Write model detail sections only for evidence that matters. Omit weak
   low-signal entries rather than forcing every changed file into the report.
9. Keep raw PR/issue/commit lists in appendices only.
10. If a model/range has no concrete diff evidence, mark it as `watching` with a
    short explanation rather than inferring progress from titles alone.

## Optional Script

`scripts/generate_report.py` exists only as an optional evidence-cache helper.
Use it when a quick JSON/Markdown scaffold is useful, but never accept its
classification without checking targeted diffs. Prefer direct AI-led inspection
for final reports.

## Quality Bar

- Lead with a concise scope note, overview table, and owner index.
- Avoid claiming progress from a PR title alone; anchor claims to changed paths,
  functions, entry points, or diff size where possible.
- For `性能优化` entries, include before/after numbers or clearly mark that no
  public performance data was found in the inspected PR/commit evidence.
- The operator summary must be directly usable for a weekly status report:
  explain the meaning, not just changed files.
- Avoid a global operator-only table because it mixes models. Use model ->
  operator/module grouping.
- In model sections, preserve hierarchy: model -> phase -> operator/module ->
  change type.
- Separate merged or completed work from exploratory, draft, or unresolved work.
- Include a final appendix with PRs, issues, commits, and the diff summary.
- Make all PR and issue numbers clickable Markdown links, not plain `#123`
  text.
- If a model/range has no concrete diff evidence, either omit it or mark it as
  "watching" with the reason.
