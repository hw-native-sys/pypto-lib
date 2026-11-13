---
name: manual-scope-lifetime
description: Diagnose PyPTO scratch-buffer lifetime failures in manual scopes by separating execution ordering from allocation retention and validating a narrow AUTO-scope repair. Use for ordered attention or Prefill graphs that fail after repeated dispatches, producer timeouts near reusable scratch, or suspected premature heap reuse.
---

# Manual Scope Buffer Lifetime

Prove how a scratch allocation stays alive, rather than adding another task
ordering edge to an already ordered computation.

Read these canonical guides before changing a scope:

- [Dependencies and Scheduling](../../../docs/debug-and-tune/dependency-and-scheduling.md),
  especially creator retention and the HCA scratch-lifetime example
- [Ring Heap and Scope Stats](../../../docs/debug-and-tune/ring-heap-and-scope-stats.md)
- [Debugging](../../../docs/debug-and-tune/debugging.md)

If a required producer-to-consumer path is missing, use
[Dependency Liveness](../dependency-liveness/SKILL.md) for that separate repair.
A running distributed wait that prevents another task from acquiring a core
also belongs there; restoring allocation retention does not repair that cycle.

## 1. Preserve the failure and identify the runtime

Record the Lib and toolchain revisions, selected runtime implementation, heap
configuration, shapes, ranks, chunking, and request sequence. Verify whether
that runtime can reclaim allocations while later tasks are still in flight.
Do not apply a heap-reuse explanation to an executor that keeps the whole
graph's arena resident until run completion without further evidence.

Capture the earliest device error and the compiled program that actually ran.
A producer timeout followed by empty-output or reference assertions is one
failure chain; the final assertion does not identify the original failing op.
Use the current build's task names and kernel map, not kernel numbers copied
from another compilation.

Keep a cold, multi-chunk or repeated-dispatch reproducer. A short first request
or a prefix hit may bypass the allocation turnover that exposes the problem.
Change one factor at a time; do not combine a scope experiment with a heap,
SDK, transport, checkpoint, or scheduling change.

## 2. Audit ordering and retention separately

Capture fresh `deps.json`, `name_map*.json`, and generated orchestration code
with the target entry's supported dependency-generation options. Include
`dispatch_program.json` when several programs share the output directory.
Keep profiling and dependency captures separate when the runtime requires it;
match task occurrences and dynamic branches before comparing them.

For each suspect scratch buffer, identify:

| Identity | Evidence to record |
| --- | --- |
| Allocation creator | Task that allocated the backing storage, not just the latest writer |
| Aliases and views | Shared backing allocation, offsets, shapes, and versions |
| Writers and consumers | Exact task occurrences that access the allocation |
| Scope ownership | Allocation's scope and where its ownership reference is released |
| Retention | `creator` edges or an equivalent explicit lifetime guarantee |

First check that the required execution paths exist, using the existing
dependency-liveness graph reporter where appropriate. Then inspect allocation
retention independently. An `explicit` path from gather to attention to merge
does not by itself show that those tasks retain the original allocator.
A path through a completed writer or a dummy join is not a substitute for
checking the runtime's retention rules.

Compare `scope` modes and edge `source` values (`creator`, `tensormap`,
`explicit`). Inspect `manual_dep=True`, `no_dep_args`, and `pl.no_dep` separately:
changing the enclosing scope does not remove a tensor-specific opt-out.
An absent creator edge is a lead, not proof of an overwrite; an equivalent
manual ownership contract may already protect the allocation.

## 3. Test the narrow scope repair

When the manual region lacks a complete lifetime contract, try ordinary
`pl.scope()` around the affected orchestration region. Preserve the existing
explicit `deps=`, tensor-specific `manual_dep=True`, and early-resolve hints
for this controlled comparison. Explicit dependencies remain valid in AUTO.
Do not convert unrelated manual regions or add a whole-forward barrier.

Inspect inline callers and callees before editing. The runtime forbids AUTO
inside MANUAL, so converting only an inner helper can create an invalid nest.
Convert the smallest enclosing manual region needed to make the repair legal.
Also preserve the entry's explicit-scope configuration, such as
`auto_scope=False`; do not change compiler-owned scope placement accidentally.

Recompile in a fresh output directory, then verify generated scope modes and
the allocator's retention relationships. Compare kernel sources as well as
the orchestration graph: identical kernels help isolate a metadata/lifetime
change, but AUTO may also add TensorMap ordering. Do not claim a retention-only
causal experiment unless those additional paths are checked too.

Do not replace this experiment with longer timeouts, global serialization,
disabling prefix caching, or blanket removal of early resolution. A larger
heap or different timing can hide a missing lifetime contract.

## 4. Validate the lifetime contract

Use the existing harness and authorized runtime reproducer. Check:

- The original cold input sequence and repeated chunk/dispatch turnover,
  including the first dispatch and later reuse of the same scratch geometry.
- Full and partial tiles, nonaligned positions, long history, and concurrent
  or ragged requests when they participate in the failure.
- Numerical outputs and any affected KV/state/checkpoint contents against
  the existing reference, including cold and prefix-reused paths.
- Normal completion, resource retirement, and heap/task/dependency-pool peaks.
  AUTO tracking may increase metadata or keep necessary allocations alive
  longer; distinguish a genuine capacity failure from the original symptom.

Separate component, end-to-end, and performance conclusions. A one-token
Prefill probe does not validate long Decode or checkpoint restoration. Run
uninstrumented timing only after correctness passes, using the same load and
heap configuration; dependency-capture timing is not a performance result.

## 5. Report the supported conclusion

Report the original error, buffer/creator/consumer identities, relevant scopes,
before/after execution paths and retention relationships, exact source edit,
and what was validated. Preserve the distinction between:

- a confirmed missing automatic retention relationship;
- a scope repair that removes the observed failure under the tested load;
- a directly observed premature reclaim or overwrite, if one was captured.

The first two do not automatically prove the third or an SDK defect. If the
current DSL contract is satisfied and evidence still implicates the toolchain,
follow the repository's problem-handling rules instead of silently shipping a
workaround as a model fix.
