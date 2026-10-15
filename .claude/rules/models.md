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

## Discovery, reuse, and replacement

Before adding compute or support code, search the repository for existing
capabilities; absence from the current conversation context is not evidence
that an implementation is missing. Search by behavior as well as names, then
inspect definitions and callers in the current model, shared utilities and
harness, and relevant sibling models.
Include kernels, goldens, quantization, comparisons, metadata, fixtures, and
CLI or validation helpers in this search.

Prefer direct reuse, then a minimal extension of the owning implementation;
extract shared code for real consumers when needed. Check candidate contracts
before deciding they cannot be reused. Respect dependency direction and avoid
coupling independent model directories to reuse a helper.
Layer and forward entries should compose existing operators. Separate
implementations need concrete semantic, dependency, or measured performance
reasons and validation evidence in the change report. Different decode and
prefill requirements can justify separate implementations.

New files, public entry points, and wrappers require a concrete consumer or
contract that existing owners and entries cannot reasonably support. Do not
create a parallel composition or validation interface solely to demonstrate
a capability.

When replacing or migrating an implementation, update affected callers and
remove obsolete implementations, parameters, exports, validation entries, and
documentation in the same change. Retain old interfaces or forwarding wrappers
only for an explicit compatibility requirement. A staged migration must identify
the remaining consumers and the conditions for completing the migration.

## Computation and integration

Preserve the requested computation and implementation constraints, including
dtype and scale encoding, layout, quantization and rounding boundaries,
sharding, and persistent-state ownership. Inspect affected operator, layer,
and forward callers, including drafts; check changed interfaces at their call
sites as well as through standalone validation.

Keep references independent of the device results they validate and aligned
with the intended numerical contract. Stage checks using actual intermediate
inputs do not establish full-chain correctness. Do not relax acceptance gates
solely to make a changed implementation pass.

For stateful or tiled changes, check publication and consumer ordering,
scratch reuse, repeated invocations, empty groups, partial tiles, and workspace
bounds at the promised shapes. Account for inferred and explicit dependencies
and actual communication-window lifetimes before changing ordering or epochs.

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

Exercise the actual supported CLI or callable entry and a relevant affected
consumer; importing a module or testing another invocation mode is not enough.
Tie validation evidence to the delivered source and final scope. After rebases
or scope changes, rerun affected checks and distinguish retained evidence from
checks not rerun or capabilities removed.

Keep documentation, tests, and benchmarks scoped to necessary contract changes
or meaningful regressions. Prefer extending existing validation entry points;
keep one-off experiment evidence in PRs or artifacts. Avoid changing `tests/`
unless the model change specifically requires it.

## Agent checklist

Before editing, inspect repository capabilities, imports and callers, and
choose the owning file and implementation constraints. Before finishing,
check kernel and golden dependency directions (including function-local
imports), formatting, duplicated compute or helpers, obsolete code, and
affected entries. Briefly report reuse decisions, reasons for necessary new
implementations or infrastructure, the source revision and default scenarios
validated, and checks not run. For review-only tasks, assess and report
without edits or cleanup.
