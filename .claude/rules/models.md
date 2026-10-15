---
paths:
  - "models/**/*.py"
---

# Model implementation rules

Apply these rules when adding, modifying, or reviewing code under `models/`.
Existing code is not a template when it conflicts with these rules. Keep
cleanup scoped to the changed operator and its affected callers.

## File ownership

- **Operator files** normally contain the kernel implementation first, its
  corresponding Torch golden second, and validation setup and an executable
  CI entry point last. Small operators such as RMSNorm may contain only
  kernels, without a golden or executable entry point; validate them through
  their consumers where appropriate.
- **Decode and prefill** should normally have separate files for the same
  operator, especially for large operators. Their shapes and performance
  requirements can require different tiling, scheduling, and buffers. Small
  operators may share a file when the implementation fits both paths. Within
  each path, standalone validation and model composition should use the same
  compute implementation.
- **`config.py`** contains model configuration only. Shared operator parameters
  do not belong there. Keep tiling, scheduling knobs, operator execution
  shapes, and validation or benchmark settings with their owning operator or
  entry point, even when multiple callers use them.
- **`utils.py`** contains shared host utilities, never kernel implementations
  (including undecorated bodies later wrapped in JIT decorators). Shared
  kernels belong in files named for their operator responsibility.

## Dependency direction

File dependencies must form an acyclic graph. Parent operators import child
operators, never the reverse. Kernel and golden implementations must follow
this same dependency direction; a child must not import its parent's golden
or validation helpers.

Decode and prefill files must not depend on each other. Put small shared
operators in independent child operator files and shared host helpers in
utilities. Do not hide dependency cycles with deferred imports.

## Reuse and replacement

Before adding compute code, inspect existing implementations and their callers
within the current model. Layer and forward entries should compose existing
operators instead of copying their compute logic. When fusion or performance
specialization requires a separate implementation, explain the differences
and validation evidence in the change report. This does not require merging
decode and prefill implementations with different shape or performance needs.

When replacing or migrating an implementation, update affected callers and
remove obsolete implementations, parameters, exports, validation entries, and
documentation in the same change. Retain old interfaces or forwarding wrappers
only for an explicit compatibility requirement. A staged migration must identify
the remaining consumers and the conditions for completing the migration.

## Coding style

Follow the rules in the [`fmt-coding-style` workflow](../skills/fmt-coding-style/SKILL.md)
for new or modified model code, including golden and validation code.
Pay particular attention to parameter grouping and line breaks. Apply the
workflow's documented exceptions; do not substitute a formatter's defaults.
Run a whole-file style pass only when requested or explicitly required by the
active workflow. During review-only tasks, check compliance and report findings
without applying edits unless fixes were requested.

## Default validation

Use the existing [golden harness](../../docs/run-and-validate/golden-harness.md)
for executable operator files. A default invocation should, where applicable,
validate a bounded set of meaningful scenarios covering relevant shapes,
branches, and boundaries on the selected platform and devices. Validate each
case and make the invocation fail if any case fails.

Support performance configurations through explicit options, rather than as
validation defaults. Large benchmark shapes and repeated timing sweeps must
not replace the default correctness suite. Keep case selection and defaults
in the owning executable file, not in model configuration.

When implementing or optimizing models, avoid changing `tests/` unless the
model change specifically requires it.

## Agent checklist

Before editing, inspect the affected operator's imports and callers and choose
its owning file according to these rules. Before finishing an implementation
change, check both kernel and golden dependency directions, ensure changed
code follows the formatting rules, check for duplicated compute and obsolete code left by replacements,
and verify the relevant standalone or composition entry. Report which
default scenarios were exercised and any validation not run.
For review-only tasks, use this checklist to assess the change and report
findings; do not perform edits or cleanup.
