---
paths:
  - "models/**/*.py"
---

# Model implementation rules

Apply these rules when adding, modifying, or reviewing code under `models/`.
Existing code is not a template when it conflicts with these rules. Keep
cleanup scoped to the changed operator and its affected callers.
Review-only tasks must not edit or clean up files unless fixes were requested.

## File ownership

- **Operator files** normally contain the kernel implementation first, its
  corresponding Torch golden second, and validation setup and an executable
  CI entry point last. Small operators such as RMSNorm may contain only
  kernels; a separate golden or executable entry point is not required.
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
or validation helpers. These constraints also apply to function-local imports.

Decode and prefill files must not depend on each other. Put small shared
operators in independent child operator files and shared host helpers in
utilities. Do not hide dependency cycles with deferred imports.

## Reuse and replacement

Reuse suitable kernels, host utilities, and validation tools already present
in the repository. Their absence from the current conversation context does
not justify reimplementing them. Separate implementations require a concrete
semantic, dependency, or measured performance need; explain why existing
capabilities cannot meet it. Do not couple independent model directories just
to reuse a helper.

Reuse `golden` comparators for numerical checks. Custom comparators should
add only validation semantics that existing comparators cannot express,
delegating supported checks to them. Do not retain comparator wrappers
solely for diagnostic output or temporary debugging checks.

Layer and forward entries must compose existing operators instead of copying
their compute logic. Fusion or different decode and prefill requirements may
justify separate implementations under the same requirement above.

New files, public entry points, and wrappers require a concrete consumer or
contract that existing owners and entries cannot reasonably support. Do not
create a parallel composition or validation interface solely to demonstrate
a capability.

When replacing or migrating an implementation, update affected callers and
remove obsolete implementations, parameters, exports, validation entries, and
documentation in the same change. Retain old interfaces or forwarding wrappers
only for an explicit compatibility requirement. A staged migration must identify
the remaining consumers and the conditions for completing the migration.

## Scope and compatibility

Keep additions of documentation, tests, and benchmarks to what the requested
change clearly needs. Prefer existing documentation and validation entry
points; keep one-off experiment material in PRs or artifacts.

Preserve agreed operator contracts and implementation constraints, including
dtypes, scale encoding, layout, quantization boundaries, sharding, and state
ownership. Intentional contract changes must update affected callers,
including drafts. Do not silently change interfaces or relax acceptance
criteria. Do not claim compatibility or capabilities based on a removed
implementation or evidence from a different source or scope.

## Coding style

New or modified code must follow the
[`fmt-coding-style` workflow](../skills/fmt-coding-style/SKILL.md), including
golden and validation code. Do not broaden a routine change into whole-file
formatting unless requested or required by the active workflow.
