# PyPTO-Lib Codex Entry Point

Before making or reviewing changes, read and follow
[`.claude/CLAUDE.md`](.claude/CLAUDE.md), the shared project instructions.
Read the task-relevant rules and follow the matching skills linked there;
do not rely on automatic loading of `.claude/rules/` in Codex.

## Codex workflow adaptation

When a Claude skill or agent refers to `Task`, a subagent, or Claude-only
plugins:

- Execute the workflow directly in Codex.
- Use parallel tool calls when safe.
- Treat agent-specific instructions as checklists, not as a separate runtime.

Keep shared project policy in `.claude/CLAUDE.md` and `.claude/rules/`, rather
than duplicating it here.
