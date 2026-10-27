---
paths:
  - "models/**/*.py"
---

# Model implementation rules

Apply these constraints when writing, modifying, or reviewing model code.
Existing code is not a template when it conflicts with these rules.

All new or modified model code, including kernels, goldens, and validation
entries, must follow the project [coding style](../skills/fmt-coding-style/SKILL.md).

## Development priorities

- Before developing or reviewing a model, read its
  `docs/models/<model>/index.md`. Use the `Development priorities` fields
  (`Performance target`, `Deferred performance work`, and
  `Compatibility requirements`) to guide implementation tradeoffs. Do not
  infer deferred work or reduced compatibility when these fields are absent.
- Apply optimizations to the existing shared implementation and adapt all
  affected callers. Do not add configuration branches, specialized entry
  points, or parallel implementations, or retain an old path, solely to
  preserve performance for configurations whose optimization is deferred.
  Reconsider performance-driven branches or file splits when those
  configurations become active targets and measurements justify the split.
- Deferred performance work does not waive correctness or interface contracts
  for supported paths. Check new branches and implementation variants against
  the model's priorities during review.

## File ownership

- Operator files own their kernels, corresponding Torch goldens, and validation
  entry points, in that order. Small operators may contain only kernels,
  without separate goldens or validation entries.
- Decode and prefill normally have separate files for large operators. Share
  small operators when both paths can use the same implementation.
- `config.py` contains model configuration only. Tiling, scheduling, execution
  shapes, and validation defaults belong to their owning operator or entry.
- `utils.py` contains shared host utilities, never kernel implementations,
  including undecorated bodies later wrapped in JIT decorators.

## Dependency direction

- Imports must be acyclic. Parent operators import child operators, never the
  reverse. The same direction applies to kernels, goldens, validation helpers,
  and function-local imports.
- Decode and prefill files must not import each other. Shared kernels belong
  in independent child operator files; shared host helpers belong in utilities.
- Do not hide dependency cycles with deferred imports or couple independent
  model directories solely to reuse a helper.

## Reuse and migration

- Reuse existing operators and host utilities. Standalone validation and model
  composition must use the same compute implementation.
- For small changes to an existing operator, modify that operator and update
  all affected callers. Do not add a wrapper, factory, or near-duplicate for
  the new path while leaving other paths on the old implementation. An extra
  dependency, output cast, fixed argument, or compile-time variant alone does
  not justify a new entry point.
- Separate implementations require a concrete semantic, dependency, or measured
  performance need within the model's development priorities that the existing
  implementation cannot meet. New files and entry points require an actual
  consumer and a distinct responsibility.
- When replacing an implementation or interface, update affected callers,
  including drafts, and remove obsolete code, parameters, exports, validation
  entries, and documentation in the same change. Keep compatibility wrappers
  only for an explicit compatibility requirement. A staged migration must
  identify remaining consumers and the condition for removing the old path.
- Reuse `golden` comparators. Custom comparators may add unsupported validation
  semantics, but must delegate supported checks to existing comparators.
  Do not keep wrappers solely for diagnostic output or temporary debugging.

## Composition depth

- Keep the forward-to-computation call chain within three or four levels by
  default, counting both endpoints. For example,
  `forward -> attention -> QKV / attention kernel / output projection`
  is three levels.
- Remove single-path layers that merely forward arguments or delegate the
  same operation to another wrapper or generated function. Each retained
  layer must own meaningful operator computation or necessary scheduling.
- Low-level GEMM, quantization, and reduction primitives may add depth when
  they perform real computation or necessary scheduling. Do not reduce depth
  by copying child operators' main computation into layer or forward entries;
  these entries must compose existing operators unless a separate
  implementation meets the justification above.
- Parent operators may directly implement simple operations that connect child
  operators, such as reshapes, slices, casts, simple elementwise computation,
  and dependency or synchronization handling. These operations need not have
  separate wrappers. Do not use them to duplicate a child operator's main
  computation. Consider moving connecting logic into its owning operator file
  when it has multiple actual callers and a distinct computation responsibility.
- Reuse and depth constraints apply to kernels, goldens, and validation entries.

## Contracts and validation

- Model development should normally leave `tests/` unchanged.
- Preserve operator contracts: dtypes, scale encoding, layouts, quantization
  boundaries, sharding, and state ownership. Do not silently change interfaces,
  relax acceptance criteria, or claim support beyond the implemented and
  validated configurations.
- Executable operator files use the existing
  [golden harness](../../docs/run-and-validate/golden-harness.md). Their default
  invocation provides bounded correctness coverage of relevant shapes,
  branches, and boundaries. Any failing subcase must make the entry fail.
- Performance configurations require explicit options; large benchmark shapes
  and timing sweeps must not replace default correctness coverage. Case
  selection and validation defaults belong to the executable entry, not model
  configuration.
