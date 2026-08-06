#!/usr/bin/env python3
"""Collect rough weekly-report evidence for pypto-lib.

This helper is intentionally not the final report writer. Its classifications are
heuristic and must be checked against targeted diffs before writing a report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_REPO = "hw-native-sys/pypto-lib"
DEFAULT_OUTPUT_DIR = "notes"


MODEL_RULES = [
    (
        "DeepSeek V4 / DSV4",
        [
            "models/deepseek/v4",
            "deepseek/v4",
            "deepseek_v4",
            "deepseek v4",
            "dsv4",
            "ds_v4",
        ],
    ),
    ("DeepSeek V3.2", ["models/deepseek/v3_2", "deepseek_v3_2", "deepseek v3.2"]),
    ("Qwen3", ["models/qwen3", "qwen3"]),
    ("Kimi / K2", ["models/kimi", "kimi", "k2"]),
    ("MILM", ["models/milm", "milm"]),
]


OPERATOR_RULES = [
    (
        "Sparse/SWA/CSA Attention",
        ["attention", "attn", "sparse", "swa", "csa"],
    ),
    ("Compressor", ["compressor", "compress"]),
    ("Decode", ["decode", "decoding"]),
    ("Prefill", ["prefill", "prefilling"]),
    ("Indexer", ["indexer", "indexing"]),
    (
        "Cache / Paged Metadata",
        ["cache", "paged", "page_metadata", "paged_metadata", "metadata"],
    ),
    ("RMSNorm / RoPE", ["rmsnorm", "rms_norm", "rope", "rotary"]),
    ("MoE / Router", ["moe", "router", "routing", "expert"]),
    (
        "Dynamic Shape / Auto Chunk",
        ["dynamic", "dynamic_shape", "auto_chunk", "autochunk", "chunk", "shape"],
    ),
]


PHASE_RULES = [
    ("Decode", ["decode", "decoding"]),
    ("Prefill", ["prefill", "prefilling"]),
    ("MTP", ["mtp"]),
    ("MoE / Router", ["moe", "router", "expert"]),
    ("Shared / Config", ["config", "combine", "common"]),
    ("Validation / Workflow", ["golden", "test", ".github", "ci.yml", "daily_ci"]),
]


@dataclass
class CommandResult:
    ok: bool
    stdout: str
    stderr: str = ""


@dataclass
class Commit:
    sha: str
    short_sha: str
    date: str
    author: str
    subject: str
    files: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    symbols: list[str] = field(default_factory=list)


@dataclass
class PullRequest:
    number: int
    title: str
    state: str
    merged_at: str | None
    updated_at: str | None
    url: str


@dataclass
class Issue:
    number: int
    title: str
    state: str
    updated_at: str | None
    url: str


@dataclass
class ModuleBucket:
    files: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)
    commits: list[Commit] = field(default_factory=list)
    prs: list[PullRequest] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


def run(cmd: list[str], cwd: Path, timeout: int = 60) -> CommandResult:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(False, "", str(exc))
    return CommandResult(proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip())


def require_git_repo(root: Path) -> None:
    result = run(["git", "rev-parse", "--show-toplevel"], root)
    if not result.ok:
        raise SystemExit("error: run this script from inside a git repository")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect rough evidence for an AI-written weekly progress report."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository name")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument("--since", help="Start date/time understood by git")
    parser.add_argument("--until", help="End date/time understood by git")
    parser.add_argument("--base-ref", help="Explicit base ref for diff")
    parser.add_argument("--head-ref", default="HEAD", help="Explicit head ref for diff")
    parser.add_argument(
        "--output",
        help="Write Markdown to this file (default: notes/weekly-YYYY-MM-DD-to-YYYY-MM-DD.md)",
    )
    parser.add_argument(
        "--evidence-output",
        help="Write structured evidence JSON to this file (default: notes/weekly-YYYY-MM-DD-to-YYYY-MM-DD-evidence.json)",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Skip gh CLI collection and use local git evidence only",
    )
    return parser.parse_args()


def default_window(days: int) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    return since.isoformat(timespec="seconds"), now.isoformat(timespec="seconds")


def date_part(value: str) -> str:
    return value[:10] if len(value) >= 10 else datetime.now().date().isoformat()


def default_output_path(since: str, until: str) -> Path:
    return Path(DEFAULT_OUTPUT_DIR) / f"weekly-{date_part(since)}-to-{date_part(until)}.md"


def default_evidence_path(since: str, until: str) -> Path:
    return Path(DEFAULT_OUTPUT_DIR) / f"weekly-{date_part(since)}-to-{date_part(until)}-evidence.json"


def find_base_ref(root: Path, since: str) -> str:
    before = run(["git", "rev-list", "-n", "1", f"--before={since}", "HEAD"], root)
    if before.ok and before.stdout:
        return before.stdout

    first = run(["git", "rev-list", "--max-parents=0", "HEAD"], root)
    if first.ok and first.stdout:
        return first.stdout.splitlines()[0]
    return "HEAD"


def git_diff_stat(root: Path, base_ref: str, head_ref: str) -> str:
    result = run(["git", "diff", "--stat", f"{base_ref}..{head_ref}"], root)
    return result.stdout if result.ok else ""


def git_changed_files(root: Path, base_ref: str, head_ref: str) -> list[str]:
    result = run(["git", "diff", "--name-only", f"{base_ref}..{head_ref}"], root)
    if not result.ok or not result.stdout:
        return []
    return sorted({line for line in result.stdout.splitlines() if line.strip()})


def git_hunk_symbols(root: Path, base_ref: str, head_ref: str) -> dict[str, set[str]]:
    changed = git_changed_files(root, base_ref, head_ref)
    symbols: dict[str, set[str]] = defaultdict(set)
    for path in changed:
        diff = run(
            ["git", "diff", "--unified=0", f"{base_ref}..{head_ref}", "--", path],
            root,
            timeout=30,
        )
        if not diff.ok:
            continue
        for line in diff.stdout.splitlines():
            if not line.startswith("@@"):
                continue
            suffix = line.rsplit("@@", 1)[-1].strip()
            if suffix:
                symbols[path].add(suffix[:120])
    return symbols


def parse_numstat(stdout: str) -> tuple[list[str], int, int]:
    files: list[str] = []
    additions = 0
    deletions = 0
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        add, delete, path = parts[0], parts[1], parts[2]
        files.append(path)
        if add.isdigit():
            additions += int(add)
        if delete.isdigit():
            deletions += int(delete)
    return files, additions, deletions


def git_commit_symbols(root: Path, sha: str) -> list[str]:
    result = run(["git", "show", "--unified=0", "--format=", sha], root, timeout=30)
    if not result.ok or not result.stdout:
        return []
    symbols: list[str] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():
        if not line.startswith("@@"):
            continue
        suffix = line.rsplit("@@", 1)[-1].strip()
        if not suffix or suffix in seen:
            continue
        if not ("def " in suffix or suffix.startswith("class ")):
            continue
        seen.add(suffix)
        symbols.append(suffix[:120])
        if len(symbols) >= 12:
            break
    return symbols


def git_commits(root: Path, since: str, until: str) -> list[Commit]:
    fmt = "%H%x1f%h%x1f%ad%x1f%an%x1f%s"
    result = run(
        [
            "git",
            "log",
            f"--since={since}",
            f"--until={until}",
            "--date=short",
            f"--pretty=format:{fmt}",
        ],
        root,
    )
    if not result.ok or not result.stdout:
        return []

    commits: list[Commit] = []
    for line in result.stdout.splitlines():
        parts = line.split("\x1f", 4)
        if len(parts) != 5:
            continue
        sha, short_sha, date, author, subject = parts
        stat_result = run(
            ["git", "show", "--numstat", "--pretty=format:", sha],
            root,
            timeout=30,
        )
        files: list[str] = []
        additions = 0
        deletions = 0
        if stat_result.ok and stat_result.stdout:
            files, additions, deletions = parse_numstat(stat_result.stdout)
        symbols = git_commit_symbols(root, sha)
        commits.append(
            Commit(
                sha,
                short_sha,
                date,
                author,
                subject,
                files,
                additions,
                deletions,
                symbols,
            )
        )
    return commits


def gh_available() -> bool:
    return shutil.which("gh") is not None


def collect_prs(root: Path, repo: str, since: str) -> list[PullRequest]:
    result = run(
        [
            "gh",
            "search",
            "prs",
            f"repo:{repo}",
            f"updated:>={since[:10]}",
            "--json",
            "number,title,state,closedAt,updatedAt,url",
            "--limit",
            "80",
        ],
        root,
        timeout=45,
    )
    if not result.ok or not result.stdout:
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    prs: list[PullRequest] = []
    for item in payload:
        prs.append(
            PullRequest(
                number=int(item.get("number", 0)),
                title=item.get("title", ""),
                state=item.get("state", ""),
                merged_at=item.get("closedAt"),
                updated_at=item.get("updatedAt"),
                url=item.get("url", ""),
            )
        )
    return prs


def collect_issues(root: Path, repo: str, since: str) -> list[Issue]:
    result = run(
        [
            "gh",
            "search",
            "issues",
            f"repo:{repo}",
            f"updated:>={since[:10]}",
            "--json",
            "number,title,state,updatedAt,url",
            "--limit",
            "80",
        ],
        root,
        timeout=45,
    )
    if not result.ok or not result.stdout:
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    issues: list[Issue] = []
    for item in payload:
        issues.append(
            Issue(
                number=int(item.get("number", 0)),
                title=item.get("title", ""),
                state=item.get("state", ""),
                updated_at=item.get("updatedAt"),
                url=item.get("url", ""),
            )
        )
    return issues


def classify_by_rules(text: str, rules: list[tuple[str, list[str]]], default: str) -> set[str]:
    lowered = text.lower()
    matches = {
        module
        for module, keywords in rules
        if any(keyword in lowered for keyword in keywords)
    }
    return matches or {default}


def classify_operator_text(text: str) -> set[str]:
    return classify_by_rules(text, OPERATOR_RULES, "Other / Cross-Cutting")


def classify_model_text(text: str) -> set[str]:
    return classify_by_rules(text, MODEL_RULES, "Cross-Model / Infrastructure")


def classify_phase_text(text: str) -> set[str]:
    return classify_by_rules(text, PHASE_RULES, "Other / Shared")


def classify_operator_file(path: str) -> set[str]:
    modules = classify_operator_text(path)
    if modules != {"Other / Cross-Cutting"}:
        return modules

    if path.startswith("golden/") or path.startswith("tests/"):
        return {"Validation / Tests"}
    if path.startswith("docs/") or path.startswith(".github/"):
        return {"Docs / Workflow"}
    return modules


def classify_model_file(path: str) -> set[str]:
    modules = classify_model_text(path)
    if modules != {"Cross-Model / Infrastructure"}:
        return modules
    if path.startswith("models/deepseek/"):
        return {"DeepSeek V4 / DSV4"}
    if path.startswith("models/qwen3/"):
        return {"Qwen3"}
    if path.startswith("models/"):
        return {"Other Models"}
    return modules


def prefer_specific_models(modules: set[str]) -> set[str]:
    specific = modules.difference({"Cross-Model / Infrastructure"})
    return specific or modules


def phase_for_path(path: str) -> str:
    phases = classify_phase_text(path)
    ordered = [phase for phase, _ in PHASE_RULES if phase in phases]
    return ordered[0] if ordered else "Other / Shared"


def phase_names_for_commit(commit: Commit, files: list[str] | None = None) -> list[str]:
    paths = files if files is not None else commit.files
    phases: set[str] = set()
    for path in paths:
        phases.add(phase_for_path(path))
    if not phases:
        phases.update(classify_phase_text(commit.subject))
    ordered = [phase for phase, _ in PHASE_RULES if phase in phases]
    extras = sorted(phases.difference(ordered))
    return ordered + extras


def build_buckets(
    changed_files: list[str],
    hunk_symbols: dict[str, set[str]],
    commits: list[Commit],
    prs: list[PullRequest],
    issues: list[Issue],
) -> dict[str, ModuleBucket]:
    buckets: dict[str, ModuleBucket] = defaultdict(ModuleBucket)

    for path in changed_files:
        for module in classify_model_file(path):
            buckets[module].files.add(path)
            buckets[module].functions.update(hunk_symbols.get(path, set()))

    for commit in commits:
        modules: set[str] = set()
        for path in commit.files:
            modules.update(classify_model_file(path))
        subject_modules = classify_model_text(commit.subject)
        if subject_modules != {"Cross-Model / Infrastructure"}:
            modules.update(subject_modules)
        modules = prefer_specific_models(modules)
        for module in modules:
            buckets[module].commits.append(commit)

    for pr in prs:
        for module in classify_model_text(pr.title):
            buckets[module].prs.append(pr)

    for issue in issues:
        for module in classify_model_text(issue.title):
            buckets[module].issues.append(issue)

    return dict(buckets)


def build_operator_buckets(
    changed_files: list[str],
    hunk_symbols: dict[str, set[str]],
    commits: list[Commit],
    prs: list[PullRequest],
    issues: list[Issue],
) -> dict[str, ModuleBucket]:
    buckets: dict[str, ModuleBucket] = defaultdict(ModuleBucket)

    for path in changed_files:
        for operator in classify_operator_file(path):
            buckets[operator].files.add(path)
            buckets[operator].functions.update(hunk_symbols.get(path, set()))

    for commit in commits:
        operators: set[str] = set()
        for path in commit.files:
            operators.update(classify_operator_file(path))
        subject_operators = classify_operator_text(commit.subject)
        if subject_operators != {"Other / Cross-Cutting"}:
            operators.update(subject_operators)
        for operator in operators:
            buckets[operator].commits.append(commit)

    for pr in prs:
        for operator in classify_operator_text(pr.title):
            buckets[operator].prs.append(pr)

    for issue in issues:
        for operator in classify_operator_text(issue.title):
            buckets[operator].issues.append(issue)

    return dict(buckets)


def render_list(items: list[str], empty: str = "_None found._") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def table_cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def inline_code(text: str) -> str:
    return f"`{text.replace('`', chr(92) + '`')}`"


def commit_type(subject: str) -> str:
    lowered = subject.lower()
    if lowered.startswith("merge "):
        return "internal"
    if lowered.startswith(("feat", "add ", "implement ")):
        return "new"
    if lowered.startswith("fix"):
        return "fix"
    if lowered.startswith(("perf", "optimize ", "tune ", "fuse ", "mix-fuse ")):
        return "perf"
    if lowered.startswith(("refactor", "rewrite ", "migrate ", "adapt ", "align ", "update ")):
        return "refactor"
    if lowered.startswith("docs"):
        return "docs"
    if lowered.startswith("test"):
        return "test"
    if lowered.startswith("chore"):
        return "chore"
    return "change"


def change_type_zh(kind: str) -> str:
    return {
        "perf": "性能优化",
        "new": "新实现",
        "fix": "bugfix",
        "refactor": "重构",
        "docs": "文档",
        "test": "测试",
        "chore": "内部",
        "internal": "内部",
    }.get(kind, "其他")


def is_visible_commit(commit: Commit) -> bool:
    kind = commit_type(commit.subject)
    if kind in {"chore", "internal"}:
        return False
    lowered = commit.subject.lower()
    return not (lowered.startswith("ci") or lowered.startswith("style"))


def pr_number_from_subject(subject: str) -> str:
    marker = subject.rfind("(#")
    if marker == -1:
        return "-"
    end = subject.find(")", marker)
    if end == -1:
        return "-"
    number = subject[marker + 2 : end]
    return f"#{number}" if number.isdigit() else "-"


def clean_subject(subject: str) -> str:
    text = subject.strip()
    if text.startswith("Merge pull request"):
        return text
    marker = text.rfind("(#")
    if marker != -1 and text.endswith(")"):
        return text[:marker].strip()
    return text


def narrative_subject(subject: str) -> str:
    text = clean_subject(subject)
    for prefix in ("Refactor ", "Add ", "Fix "):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    if ":" not in text:
        return text
    prefix, rest = text.split(":", 1)
    lowered = prefix.lower()
    conventional = (
        lowered in {"feat", "fix", "perf", "refactor", "docs", "test", "chore"}
        or lowered.startswith(("feat(", "fix(", "perf(", "refactor(", "docs(", "test("))
    )
    return rest.strip() if conventional and rest.strip() else text


def module_names_for_commit(commit: Commit) -> list[str]:
    modules: set[str] = set()
    for path in commit.files:
        modules.update(classify_model_file(path))
    modules.update(classify_model_text(commit.subject))
    modules = prefer_specific_models(modules)
    ordered = [module for module, _ in MODEL_RULES if module in modules]
    extras = sorted(modules.difference(ordered))
    return ordered + extras


def short_modules(modules: list[str], limit: int = 2) -> str:
    if not modules:
        return "Other / Cross-Cutting"
    if len(modules) <= limit:
        return ", ".join(modules)
    return ", ".join(modules[:limit]) + " 等"


def operator_names_for_commit(commit: Commit) -> list[str]:
    operators: set[str] = set()
    for path in commit.files:
        operators.update(classify_operator_file(path))
    subject_operators = classify_operator_text(commit.subject)
    if subject_operators != {"Other / Cross-Cutting"}:
        operators.update(subject_operators)
    ordered = [operator for operator, _ in OPERATOR_RULES if operator in operators]
    extras = sorted(operators.difference(ordered))
    return ordered + extras


def unique_commits(commits: list[Commit]) -> list[Commit]:
    seen: set[str] = set()
    result: list[Commit] = []
    for commit in commits:
        if commit.sha in seen:
            continue
        seen.add(commit.sha)
        result.append(commit)
    return result


def module_prs_for_commits(commits: list[Commit], prs: list[PullRequest]) -> list[PullRequest]:
    wanted = {
        int(pr[1:])
        for pr in (pr_number_from_subject(commit.subject) for commit in commits)
        if pr.startswith("#")
    }
    by_number = {pr.number: pr for pr in prs}
    return [by_number[number] for number in sorted(wanted, reverse=True) if number in by_number]


def overview_rows(commits: list[Commit]) -> list[str]:
    rows = ["| Commit | PR | 作者 | 主题 | 模型 / 范围 | 类型 |"]
    rows.append("| ------ | -- | ---- | ---- | ----------- | ---- |")
    for commit in commits:
        modules = short_modules(module_names_for_commit(commit))
        rows.append(
            "| "
            + " | ".join(
                [
                    table_cell(commit.short_sha),
                    table_cell(pr_number_from_subject(commit.subject)),
                    table_cell(commit.author),
                    table_cell(inline_code(clean_subject(commit.subject))),
                    table_cell(modules),
                    table_cell(commit_type(commit.subject)),
                ]
            )
            + " |"
        )
    return rows


def owner_index_rows(commits: list[Commit]) -> list[str]:
    by_owner: dict[str, list[Commit]] = defaultdict(list)
    for commit in commits:
        by_owner[commit.author].append(commit)

    rows = ["| Owner | 提交数 | 覆盖模型 / 范围 |"]
    rows.append("| ----- | ------ | ------------- |")
    for owner, owner_commits in sorted(
        by_owner.items(), key=lambda item: (-len(item[1]), item[0].lower())
    ):
        topics: list[str] = []
        seen: set[str] = set()
        for commit in owner_commits:
            for module in module_names_for_commit(commit):
                if module in seen:
                    continue
                seen.add(module)
                topics.append(module)
        rows.append(
            f"| {table_cell(owner)} | {len(owner_commits)} | {table_cell('、'.join(topics[:8]))} |"
        )
    return rows


def module_sort_key(item: tuple[str, ModuleBucket]) -> tuple[int, str]:
    module, bucket = item
    score = len(bucket.files) * 3 + len(bucket.commits) * 2 + len(bucket.prs) + len(bucket.issues)
    return (-score, module)


def model_sort_key(item: tuple[str, ModuleBucket]) -> tuple[int, int, str]:
    model, bucket = item
    priority = {
        "DeepSeek V4 / DSV4": 0,
        "Qwen3": 1,
        "DeepSeek V3.2": 2,
        "Kimi / K2": 3,
        "MILM": 4,
        "Other Models": 5,
        "Cross-Model / Infrastructure": 9,
    }.get(model, 6)
    score = len(bucket.files) * 3 + len(bucket.commits) * 2 + len(bucket.prs) + len(bucket.issues)
    return (priority, -score, model)


def evidence_links(
    commits: list[Commit], prs: list[PullRequest], issues: list[Issue]
) -> str:
    parts: list[str] = []
    commit_refs = [inline_code(commit.short_sha) for commit in unique_commits(commits)[:4]]
    if commit_refs:
        parts.append("commit " + "、".join(commit_refs))

    pr_refs = [f"[#{pr.number}]({pr.url})" for pr in prs[:4]]
    if pr_refs:
        parts.append("PR " + "、".join(pr_refs))

    issue_refs = [f"[#{issue.number}]({issue.url})" for issue in issues[:4]]
    if issue_refs:
        parts.append("issue " + "、".join(issue_refs))

    return "；".join(parts) if parts else "本地 diff"


def summarize_module(module: str, bucket: ModuleBucket) -> str:
    files = sorted(bucket.files)
    lowered = " ".join(files + [commit.subject for commit in bucket.commits]).lower()
    if module == "DeepSeek V4 / DSV4":
        return "DeepSeek V4 是本周期主线，改动覆盖 decode、prefill、compressor、indexer、RoPE/RMSNorm、MoE 与 sparse attention 等路径。"
    if module == "DeepSeek V3.2":
        return "DeepSeek V3.2 主要是 prefill/front 路径的跟进改动。"
    if module == "Qwen3":
        return "Qwen3 侧主要集中在 14B decode/prefill 路径、dynamic dims、page size 和 fused attention。"
    if module == "Kimi / K2":
        return "Kimi/K2 侧主要关注 decode 入口和模型示例路径的跟进。"
    if module == "MILM":
        return "MILM 侧主要关注 decode 入口和模型示例路径的跟进。"
    if module == "Cross-Model / Infrastructure":
        return "跨模型 / 基础设施改动主要涉及 CI、golden、测试、文档或通用配置。"
    if module == "Decode":
        return "Decode 侧主要围绕单 token 推理、KV/page metadata、compressor 与 attention 子路径推进。"
    if module == "Prefill":
        return "Prefill 侧主要围绕长序列批处理、SWA/CSA/HCA attention、compressor 和 indexer 链路推进。"
    if module == "Compressor":
        return "Compressor 侧主要围绕 ratio4/ratio128、state paging、indexer 协同和缓存契约调整。"
    if module == "Sparse/SWA/CSA Attention":
        return "Attention 侧主要围绕 sparse/SWA/CSA/HCA 的统一实现、调度策略和 golden 复用推进。"
    if module == "Qwen3":
        return "Qwen3 侧主要集中在 14B decode/prefill 路径、dynamic dims、page size 和 fused attention。"
    if "decode" in lowered and "prefill" in lowered:
        return f"{module} 同时覆盖 decode 与 prefill 路径，重点是统一入口、缓存契约和算子拆分。"
    if "decode" in lowered:
        return f"{module} 主要集中在 decode 链路，涉及单 token 推理路径、paged metadata 或 cache 访问。"
    if "prefill" in lowered:
        return f"{module} 主要集中在 prefill 链路，涉及长序列批处理、attention/compressor 调度或 metadata 组织。"
    if "compress" in lowered:
        return f"{module} 主要围绕 compressor 数据路径、ratio 配置和状态分页契约推进。"
    if "rope" in lowered or "rms" in lowered:
        return f"{module} 主要围绕 RMSNorm/RoPE 子路径的实现形态和数据搬运方式调整。"
    if "test" in lowered or "golden" in lowered:
        return f"{module} 主要补充验证覆盖，便于定位模型算子正确性和回归风险。"
    return f"{module} 有一组跨文件改动，需要结合下方文件和入口继续提炼实际影响。"


def format_pr(pr: PullRequest) -> str:
    return f"[#{pr.number}]({pr.url}) {pr.title}（{pr.state}）"


def format_issue(issue: Issue) -> str:
    return f"[#{issue.number}]({issue.url}) {issue.title}（{issue.state}）"


def change_size(commit: Commit) -> str:
    if commit.additions == 0 and commit.deletions == 0:
        return ""
    return f"，diff 规模约 +{commit.additions}/-{commit.deletions}"


def commit_summary(commit: Commit, module: str, touched: list[str]) -> str:
    subject = narrative_subject(commit.subject)
    kind = commit_type(commit.subject)
    operators = [
        operator
        for operator in operator_names_for_commit(commit)
        if operator not in {"Other / Cross-Cutting", "Docs / Workflow"}
    ]
    operator_note = f"，覆盖 {'、'.join(operators[:3])}" if operators else ""
    symbol_note = ""
    if commit.symbols:
        symbol_note = f"，触及入口 {', '.join(inline_code(symbol) for symbol in commit.symbols[:3])}"
    if touched:
        paths = "、".join(inline_code(path) for path in touched[:4])
        suffix = f"，主要涉及 {paths}"
    else:
        suffix = ""
    suffix = f"{suffix}{operator_note}{symbol_note}{change_size(commit)}"

    if kind == "new":
        return f"新增 {subject}{suffix}。"
    if kind == "fix":
        return f"修复 {subject}{suffix}。"
    if kind == "perf":
        return f"优化 {subject}{suffix}。"
    if kind == "refactor":
        return f"重构 {subject}{suffix}。"
    return f"{module} 方向推进 {subject}{suffix}。"


def phase_sort_key(phase: str) -> tuple[int, str]:
    priority = {
        "Decode": 0,
        "Prefill": 1,
        "MTP": 2,
        "MoE / Router": 3,
        "Shared / Config": 4,
        "Validation / Workflow": 8,
        "Other / Shared": 9,
    }.get(phase, 6)
    return (priority, phase)


def operator_breakdown_rows(commits: list[Commit]) -> list[str]:
    by_operator: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for commit in commits:
        kind = commit_type(commit.subject)
        for operator in operator_names_for_commit(commit):
            if operator in {"Other / Cross-Cutting", "Docs / Workflow", "Validation / Tests"}:
                continue
            by_operator[operator][kind] += 1

    if not by_operator:
        return []

    rows = ["| 算子 / 模块 | 性能优化 | 新实现 | bugfix | 重构 | 其他 |"]
    rows.append("| ----------- | -------- | ------ | ------ | ---- | ---- |")
    for operator in sorted(by_operator):
        counts = by_operator[operator]
        other = sum(
            value
            for key, value in counts.items()
            if key not in {"perf", "new", "fix", "refactor"}
        )
        rows.append(
            f"| {table_cell(operator)} | {counts['perf']} | {counts['new']} | {counts['fix']} | {counts['refactor']} | {other} |"
        )
    return rows


def group_model_commits_by_phase(module: str, commits: list[Commit]) -> dict[str, list[tuple[Commit, list[str]]]]:
    grouped: dict[str, list[tuple[Commit, list[str]]]] = defaultdict(list)
    for commit in commits:
        model_files = [path for path in commit.files if module in classify_model_file(path)]
        phases = phase_names_for_commit(commit, model_files)
        for phase in phases:
            phase_files = [path for path in model_files if phase_for_path(path) == phase]
            grouped[phase].append((commit, phase_files or model_files))
    return dict(grouped)


def model_priority(model: str) -> int:
    return {
        "DeepSeek V4 / DSV4": 0,
        "Qwen3": 1,
        "DeepSeek V3.2": 2,
        "Kimi / K2": 3,
        "MILM": 4,
        "Other Models": 5,
        "Cross-Model / Infrastructure": 9,
    }.get(model, 6)


def operator_priority(operator: str) -> int:
    return {
        "Decode": 0,
        "Prefill": 1,
        "Sparse/SWA/CSA Attention": 2,
        "Compressor": 3,
        "Indexer": 4,
        "Cache / Paged Metadata": 5,
        "RMSNorm / RoPE": 6,
        "MoE / Router": 7,
        "Dynamic Shape / Auto Chunk": 8,
    }.get(operator, 20)


def summarize_model_operator(model: str, operator: str, commits: list[Commit]) -> str:
    text = " ".join(commit.subject.lower() for commit in commits)
    phases: list[str] = []
    seen_phases: set[str] = set()
    for commit in commits:
        for phase in phase_names_for_commit(commit):
            if phase in seen_phases:
                continue
            seen_phases.add(phase)
            phases.append(phase)
    phase_text = "、".join(phases[:3]) if phases else "相关路径"

    if operator == "Compressor":
        if "ratio4" in text or "ratio-4" in text:
            return f"{model} 的 compressor 重点推进 ratio4/ratio128 路径、state paging 与 indexer 协同，覆盖 {phase_text}。"
        return f"{model} 的 compressor 主要围绕压缩状态、缓存契约和与 attention/indexer 的数据衔接，覆盖 {phase_text}。"
    if operator == "Sparse/SWA/CSA Attention":
        if "swa" in text or "sparse" in text:
            return f"{model} 的 attention 重点在 SWA/sparse/CSA 路径统一、调度调整和 golden 复用，覆盖 {phase_text}。"
        return f"{model} 的 attention 主要调整注意力算子实现与验证路径，覆盖 {phase_text}。"
    if operator == "RMSNorm / RoPE":
        return f"{model} 的 RMSNorm/RoPE 主要围绕 RoPE 数据重排、scatter/gather 替代和前后处理融合，覆盖 {phase_text}。"
    if operator == "Indexer":
        return f"{model} 的 indexer 改动集中在 top-k/index 生成、KV compressor 对接和分页 cache contract，覆盖 {phase_text}。"
    if operator == "Cache / Paged Metadata":
        return f"{model} 的 cache / paged metadata 改动主要服务于 per-row start_pos、block table 和 vLLM-style serving contract。"
    if operator == "MoE / Router":
        return f"{model} 的 MoE/router 改动集中在 router、dispatch、expert、combine 等专家路径组织。"
    if operator == "Decode":
        return f"{model} 的 decode 改动集中在单 token 推理路径、attention/compressor 融合和动态 metadata 输入。"
    if operator == "Prefill":
        return f"{model} 的 prefill 改动集中在长序列批处理、chunk 化、attention/compressor 调度和 metadata 组织。"
    if operator == "Dynamic Shape / Auto Chunk":
        return f"{model} 的 dynamic shape / auto chunk 改动主要用于减少手写 chunk 调度、对齐运行时动态维输入。"
    return f"{model} 的 {operator} 改动覆盖 {phase_text}，需要结合 evidence JSON 继续提炼最终汇报话术。"


def render_operator_summary(commits: list[Commit]) -> list[str]:
    lines: list[str] = []
    lines.append("## 算子总体改动")
    lines.append("")
    grouped: dict[str, dict[str, list[Commit]]] = defaultdict(lambda: defaultdict(list))
    for commit in unique_commits([commit for commit in commits if is_visible_commit(commit)]):
        models = [
            model
            for model in module_names_for_commit(commit)
            if model != "Cross-Model / Infrastructure"
        ]
        operators = [
            operator
            for operator in operator_names_for_commit(commit)
            if operator not in {"Other / Cross-Cutting", "Docs / Workflow", "Validation / Tests"}
        ]
        for model in models:
            for operator in operators:
                grouped[model][operator].append(commit)

    if not grouped:
        lines.append("_未识别到明确的算子维度改动。_")
        lines.append("")
        return lines

    lines.append("> 本节是汇报主线：先按模型归类，再看每个算子/模块本周的性能优化、新实现、bugfix 和重构。")
    lines.append("")
    for model in sorted(grouped, key=lambda item: (model_priority(item), item)):
        lines.append(f"### {model}")
        lines.append("")
        rows = ["| 算子 / 模块 | 阶段 | 性能优化 | 新实现 | bugfix | 重构 | 汇报要点 |"]
        rows.append("| ----------- | ---- | -------- | ------ | ------ | ---- | -------- |")
        for operator in sorted(
            grouped[model],
            key=lambda item: (operator_priority(item), item),
        ):
            op_commits = unique_commits(grouped[model][operator])
            type_counts = defaultdict(int)
            phases: list[str] = []
            seen_phases: set[str] = set()
            for commit in op_commits:
                type_counts[commit_type(commit.subject)] += 1
                for phase in phase_names_for_commit(commit):
                    if phase in seen_phases:
                        continue
                    seen_phases.add(phase)
                    phases.append(phase)
            rows.append(
                "| "
                + " | ".join(
                    [
                        table_cell(operator),
                        table_cell("、".join(phases[:4])),
                        table_cell(type_counts["perf"]),
                        table_cell(type_counts["new"]),
                        table_cell(type_counts["fix"]),
                        table_cell(type_counts["refactor"]),
                        table_cell(summarize_model_operator(model, operator, op_commits)),
                    ]
                )
                + " |"
            )
        lines.extend(rows)
        lines.append("")
    lines.append("")
    return lines


def commit_to_dict(commit: Commit) -> dict[str, object]:
    return {
        "sha": commit.sha,
        "short_sha": commit.short_sha,
        "date": commit.date,
        "author": commit.author,
        "subject": commit.subject,
        "type": commit_type(commit.subject),
        "change_type": change_type_zh(commit_type(commit.subject)),
        "pr": pr_number_from_subject(commit.subject),
        "models": module_names_for_commit(commit),
        "phases": phase_names_for_commit(commit),
        "operators": operator_names_for_commit(commit),
        "files": commit.files,
        "additions": commit.additions,
        "deletions": commit.deletions,
        "symbols": commit.symbols,
    }


def pr_to_dict(pr: PullRequest) -> dict[str, object]:
    return {
        "number": pr.number,
        "title": pr.title,
        "state": pr.state,
        "closed_at": pr.merged_at,
        "updated_at": pr.updated_at,
        "url": pr.url,
        "models": sorted(classify_model_text(pr.title)),
        "operators": sorted(classify_operator_text(pr.title)),
        "phases": sorted(classify_phase_text(pr.title)),
    }


def issue_to_dict(issue: Issue) -> dict[str, object]:
    return {
        "number": issue.number,
        "title": issue.title,
        "state": issue.state,
        "updated_at": issue.updated_at,
        "url": issue.url,
        "models": sorted(classify_model_text(issue.title)),
        "operators": sorted(classify_operator_text(issue.title)),
        "phases": sorted(classify_phase_text(issue.title)),
    }


def bucket_to_dict(bucket: ModuleBucket) -> dict[str, object]:
    return {
        "files": sorted(bucket.files),
        "functions": sorted(bucket.functions),
        "commits": [commit.short_sha for commit in unique_commits(bucket.commits)],
        "prs": [pr.number for pr in bucket.prs],
        "issues": [issue.number for issue in bucket.issues],
    }


def build_evidence_payload(
    repo: str,
    since: str,
    until: str,
    base_ref: str,
    head_ref: str,
    diff_stat: str,
    changed_files: list[str],
    commits: list[Commit],
    prs: list[PullRequest],
    issues: list[Issue],
    buckets: dict[str, ModuleBucket],
    operator_buckets: dict[str, ModuleBucket],
    github_attempted: bool,
) -> dict[str, object]:
    return {
        "repo": repo,
        "window": {"since": since, "until": until},
        "diff": {
            "base_ref": base_ref,
            "head_ref": head_ref,
            "stat": diff_stat,
            "changed_files": changed_files,
        },
        "github_attempted": github_attempted,
        "classification": {
            "models": {name: bucket_to_dict(bucket) for name, bucket in buckets.items()},
            "operators": {
                name: bucket_to_dict(bucket) for name, bucket in operator_buckets.items()
            },
        },
        "commits": [commit_to_dict(commit) for commit in commits],
        "pull_requests": [pr_to_dict(pr) for pr in prs],
        "issues": [issue_to_dict(issue) for issue in issues],
        "writer_notes": [
            "Use this JSON as evidence, not as final prose.",
            "Write final analysis with model -> phase -> operator hierarchy.",
            "For operator changes, group by performance optimization, new implementation, bugfix, and refactor.",
        ],
    }


def render_report(
    repo: str,
    since: str,
    until: str,
    base_ref: str,
    head_ref: str,
    diff_stat: str,
    changed_files: list[str],
    commits: list[Commit],
    prs: list[PullRequest],
    issues: list[Issue],
    buckets: dict[str, ModuleBucket],
    operator_buckets: dict[str, ModuleBucket],
    github_attempted: bool,
) -> str:
    visible_commits = [commit for commit in commits if is_visible_commit(commit)]
    skipped_count = len(commits) - len(visible_commits)
    sorted_buckets = sorted(buckets.items(), key=model_sort_key)
    top_modules = [
        module
        for module, bucket in sorted_buckets
        if module != "Cross-Model / Infrastructure"
        and (bucket.commits or bucket.files or bucket.prs or bucket.issues)
    ][:5]

    lines: list[str] = []
    lines.append(f"# PyPTO-Lib 周报：{date_part(since)} ~ {date_part(until)}（模型与算子进展）")
    lines.append("")
    lines.append("> 本周报面向 `hw-native-sys/pypto-lib` 的模型、算子、验证与运行入口进展。")
    lines.append("> ")
    lines.append(
        f"> 统计窗口 `{date_part(since)}` ~ `{date_part(until)}`，基于 `{base_ref[:12]}..{head_ref}` diff；"
        f"共 {len(commits)} 个提交，其中纳入正文 **{len(visible_commits)}** 个，"
        f"跳过内部 / chore / merge 提交 **{skipped_count}** 个。"
        f"本区间涉及 {len(changed_files)} 个变更文件、{len(prs)} 个 PR、{len(issues)} 个 issue。"
    )
    if github_attempted and (prs or issues):
        lines.append("> GitHub PR / issue 证据已通过 `gh` 采集；正文按模型 / 范围归并，PR、issue、commit 仅作为证据。")
    else:
        lines.append("> GitHub PR / issue 证据不可用或为空；正文主要基于本地 git diff 与 commit。")
    lines.append("> 本文件是脚本生成的证据骨架；最终周报应结合同名 `*-evidence.json` 由 AI / 人工补充判断与取舍。")
    lines.append("")

    lines.append("## 概览")
    lines.append("")
    if visible_commits:
        lines.extend(overview_rows(visible_commits))
    else:
        lines.append("_本区间没有识别到可放入正文的用户可见提交。_")
    lines.append("")

    lines.append("## Owner 索引")
    lines.append("")
    if visible_commits:
        lines.extend(owner_index_rows(visible_commits))
    else:
        lines.append("_无。_")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.extend(render_operator_summary(visible_commits))
    lines.append("---")
    lines.append("")

    if top_modules:
        lines.append("## 模型主线")
        lines.append("")
        for module in top_modules:
            lines.append(f"- {module}")
        lines.append("")
        lines.append("---")
        lines.append("")

    section_no = 1
    for module, bucket in sorted_buckets:
        module_commits = [commit for commit in unique_commits(bucket.commits) if is_visible_commit(commit)]
        if not (module_commits or bucket.files or bucket.prs or bucket.issues):
            continue

        lines.append(f"## {section_no}. {module}")
        lines.append("")
        lines.append(f"- **摘要**：{summarize_module(module, bucket)}")
        lines.append(f"- **证据**：{evidence_links(module_commits, bucket.prs, bucket.issues)}")

        file_items = sorted(bucket.files)[:10]
        if file_items:
            extra = f"（另有 {len(bucket.files) - len(file_items)} 个）" if len(bucket.files) > len(file_items) else ""
            lines.append(f"- **涉及文件**{extra}：{', '.join(inline_code(path) for path in file_items)}")

        function_candidates = [
            symbol
            for symbol in sorted(bucket.functions)
            if "def " in symbol or symbol.startswith("class ")
        ]
        function_items = (function_candidates or sorted(bucket.functions))[:8]
        if function_items:
            lines.append(f"- **函数 / 入口**：{', '.join(inline_code(symbol) for symbol in function_items)}")

        if bucket.issues:
            lines.append(f"- **关联 issue**：{'; '.join(format_issue(issue) for issue in bucket.issues[:6])}")

        if module_commits:
            lines.append("")
            grouped = group_model_commits_by_phase(module, module_commits)
            phase_no = 1
            for phase in sorted(grouped, key=phase_sort_key):
                phase_items = grouped[phase]
                phase_commits = [commit for commit, _ in phase_items]
                lines.append(f"### {section_no}.{phase_no} {phase}")
                lines.append("")
                phase_rows = operator_breakdown_rows(phase_commits)
                if phase_rows:
                    lines.extend(phase_rows)
                    lines.append("")
                for item_no, (commit, phase_files) in enumerate(phase_items[:8], start=1):
                    pr_ref = pr_number_from_subject(commit.subject)
                    pr_suffix = "" if pr_ref == "-" else f" ({pr_ref})"
                    lines.append(f"#### {section_no}.{phase_no}.{item_no} {clean_subject(commit.subject)}{pr_suffix}")
                    lines.append("")
                    lines.append(f"- **作者**：{commit.author}")
                    lines.append(f"- **类型**：{change_type_zh(commit_type(commit.subject))}")
                    touched = phase_files[:8]
                    if touched:
                        lines.append(f"- **涉及文件**：{', '.join(inline_code(path) for path in touched)}")
                    lines.append(f"- **摘要**：{commit_summary(commit, module, touched)}")
                    lines.append("")
                phase_no += 1

        module_prs = module_prs_for_commits(module_commits, prs)
        loose_prs = [pr for pr in bucket.prs if pr not in module_prs]
        if module_prs or loose_prs:
            lines.append("**相关 PR**：")
            lines.append("")
            for pr in (module_prs + loose_prs)[:8]:
                lines.append(f"- {format_pr(pr)}")
            lines.append("")

        section_no += 1

    lines.append("## 关注项")
    lines.append("")
    open_issues = [issue for issue in issues if issue.state.lower() == "open"]
    if open_issues:
        for issue in open_issues[:12]:
            lines.append(f"- {format_issue(issue)}")
    else:
        lines.append("- 本区间未采集到打开状态的 issue。")
    lines.append("")

    lines.append("## 附录：Diff 摘要")
    lines.append("")
    if diff_stat:
        lines.append("```text")
        lines.append(diff_stat)
        lines.append("```")
    else:
        lines.append("_No diff stat available._")
    lines.append("")

    lines.append("## 附录：全部 Commit")
    lines.append("")
    commit_lines = [
        f"`{commit.short_sha}` {commit.date} {commit.subject} ({commit.author})"
        for commit in commits
    ]
    lines.append(render_list(commit_lines))
    lines.append("")

    lines.append("## 附录：Pull Requests")
    lines.append("")
    pr_lines = [
        f"[#{pr.number}]({pr.url}) {pr.title} [{pr.state}] updated `{pr.updated_at}`"
        for pr in prs
    ]
    lines.append(render_list(pr_lines))
    lines.append("")

    lines.append("## 附录：Issues")
    lines.append("")
    issue_lines = [
        f"[#{issue.number}]({issue.url}) {issue.title} [{issue.state}] updated `{issue.updated_at}`"
        for issue in issues
    ]
    lines.append(render_list(issue_lines))
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    require_git_repo(root)

    since, until = (args.since, args.until) if args.since and args.until else default_window(args.days)
    base_ref = args.base_ref or find_base_ref(root, since)
    head_ref = args.head_ref

    changed_files = git_changed_files(root, base_ref, head_ref)
    hunk_symbols = git_hunk_symbols(root, base_ref, head_ref)
    diff_stat = git_diff_stat(root, base_ref, head_ref)
    commits = git_commits(root, since, until)

    github_attempted = False
    prs: list[PullRequest] = []
    issues: list[Issue] = []
    if not args.no_github and gh_available():
        github_attempted = True
        prs = collect_prs(root, args.repo, since)
        issues = collect_issues(root, args.repo, since)

    buckets = build_buckets(changed_files, hunk_symbols, commits, prs, issues)
    operator_buckets = build_operator_buckets(changed_files, hunk_symbols, commits, prs, issues)
    report = render_report(
        args.repo,
        since,
        until,
        base_ref,
        head_ref,
        diff_stat,
        changed_files,
        commits,
        prs,
        issues,
        buckets,
        operator_buckets,
        github_attempted,
    )

    evidence = build_evidence_payload(
        args.repo,
        since,
        until,
        base_ref,
        head_ref,
        diff_stat,
        changed_files,
        commits,
        prs,
        issues,
        buckets,
        operator_buckets,
        github_attempted,
    )

    output_path = Path(args.output) if args.output else default_output_path(since, until)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    evidence_path = (
        Path(args.evidence_output)
        if args.evidence_output
        else default_evidence_path(since, until)
    )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    print(f"wrote {evidence_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
