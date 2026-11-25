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
- **Keep the model index stable.** Approve changes to
  `docs/models/<model>/index.md` only when readers need a new or corrected
  description of the model's architecture, development priorities, public
  interface, or supported configurations. Request removal of additions about
  internal refactors, tile choices, fixtures, or individual benchmark results
  before approval. Record per-change tuning details and measurements in the PR.
- Apply optimizations to the existing shared implementation and adapt all
  affected callers. Do not add configuration branches, specialized entry
  points, or parallel implementations, or retain an old path, solely to
  preserve performance for configurations whose optimization is deferred.
  Reconsider performance-driven branches or file splits when those
  configurations become active targets and measurements justify the split.
- Deferred performance work does not waive correctness or interface contracts
  for supported paths. Check new branches and implementation variants against
  the model's priorities during review.
- Size each SPMD launch to the work and the AICs or AIVs available in the
  target device allocation, including any virtual partition. For example,
  an A5 allocation may expose 32 AICs and 64 AIVs, and an A3 allocation
  may expose 24 AICs and 48 AIVs. Cap workers at the actual allocation's
  limit for their core type. For mixed kernels, respect both limits.
  When there are more tiles than workers, distribute all tiles among the
  workers, including the tail. In performance PRs,
  check that the launch stays within these limits as supported shapes and
  tiling parameters vary.

## File ownership

- Operator files own their kernels, corresponding Torch goldens, and validation
  entry points, in that order. Small operators may contain only kernels,
  without separate goldens or validation entries.
- Decode and prefill normally have separate files for large operators. Share
  small operators when both paths can use the same implementation.
- `config.py` owns model configuration, including capacities that define
  caller-visible tensor shapes, such as the maximum decode batch per rank.
  The caller supplies the active workload for each invocation. The owning
  operator or entry determines tiling, internal padding, and launch shapes
  and owns its scheduling choices and validation defaults.
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
- Before changing a shared signature for one use case, weigh the benefit to
  that caller against the changes required across other callers. If most
  callers would only forward the argument or pass a placeholder, first check
  whether the caller that needs it can handle the requirement locally.
- Prefer automatic dependency inference from tensor reads and writes. Before
  adding a `TaskId` argument, identify the caller that needs the ordering and
  the tasks it must order. Check whether existing dependencies already order
  them, including through intermediate tasks. Add an explicit dependency
  only where the required ordering is missing.
- For small changes to an existing operator, modify that operator and update
  all affected callers. Do not add a wrapper, factory, or near-duplicate for
  the new path while leaving other paths on the old implementation. An extra
  dependency, output cast, fixed argument, or compile-time variant alone does
  not justify a new entry point.
- Separate implementations require a concrete semantic, dependency, or measured
  performance need within the model's development priorities that the existing
  implementation cannot meet. New files and entry points require an actual
  consumer and a distinct responsibility.
- For a kernel mode argument, such as an integer selector or a boolean flag
  (including compile-time flags), check what varies and why the caller needs
  to select it. If it switches between distinct algorithms, such as separate
  decode and prefill implementations, give each algorithm its own kernel.
  If one algorithm can express both, derive its tiling from the input shape
  and hardware constraints where possible. For example, choose between tile
  sizes by comparing padded work or tile counts. Expose a tiling parameter
  when the caller has information or a scheduling choice the kernel lacks.
- For parameters that vary widely across workloads, introduce a performance
  branch only when the optimization benefits a meaningful range of values.
  For example, a speedup at `token == 128` alone does not justify a fast
  path. If a short-sequence strategy benefits token counts up to 128, select
  it with `if token <= 128` and check the benefit at representative lengths
  across that range.
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

- **At most one contract test file per model.** The only file a model PR may
  add or update under `tests/` is that model's serving contract test in
  `tests/contract/`. Use the existing file if present. Limit its assertions to
  the interface observable by serving callers. Request removal of any other
  `tests/` additions or edits before approval. Operator correctness
  and boundary cases belong in the model's executable validation entry.
- Extend the existing model entry when it can exercise the changed behavior.
- Update CI configuration when the change affects platform selection, case
  discovery, or execution.
- Preserve operator contracts: dtypes, scale encoding, layouts, quantization
  boundaries, sharding, and state ownership. Do not silently change interfaces,
  relax acceptance criteria, or claim support beyond the implemented and
  validated configurations.
- Executable operator files use the existing
  [golden harness](../../docs/run-and-validate/golden-harness.md). Their default
  invocation provides bounded correctness coverage of relevant shapes,
  branches, and boundaries. Choose inputs that drive the model caller's path
  through the changed computation or state transition. Check the affected
  output or state so that skipping the computation or writing the wrong value
  would fail the case. Any failing subcase must make the entry fail.
- Performance configurations require explicit options; large benchmark shapes
  and timing sweeps must not replace default correctness coverage. Case
  selection and validation defaults belong to the executable entry, not model
  configuration.

## Simulator CI eligibility

Support Tile instructions and CCE operations already handled by the simulator
(e.g. `pipe_barrier`, `set_flag`, and `wait_flag`). For other CCE features that
require substantial simulator changes, skip simulator CI by adding only a
standalone `# ci: no-sim` near the top of each affected executable Python entry.
