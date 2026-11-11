# Language Policy

## Core Principle

**Always use English as the intermediate/working language. Match the user's language only in final outputs.**

## What Must Be in English (Intermediate Work)

- Code comments and docstrings
- Commit messages
- Tool invocations and tool-related descriptions
- All text between tool calls (narration, status lines, transitional text)
- Agent prompts and agent task descriptions
- Search queries and grep patterns

## What Must Match the User's Language (Final Outputs)

- **Only the final consolidated response** at the end of a task (summary, conclusion)
- Plans and proposals presented to the user
- Questions asked to the user

## Why

- English is the lingua franca of the codebase, tools, and technical context
- Consistent intermediate language avoids mixed-language confusion in logs and tool calls
- User-facing output in their language ensures clear communication
