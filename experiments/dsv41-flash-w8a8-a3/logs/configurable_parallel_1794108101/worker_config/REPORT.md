# Configurable topology and collectives worker — FINAL/PARK

Exclusive product changes: config.py, moe_exchange.py, tp_reduce.py. No other
product files, shared ledgers, A5/runtime/dependencies/checkpoint files changed by
this worker. Main owns every external consumer migration and full entries.

Import-time --tp N/--tp=N and --ep N/--ep=N resolve defaults4/16. TP1/2/4/8,
EP2/4/8/16 require EP>=TP and exact divisibility; WORLD=EP, DP=EP//TP and
local_experts=384/EP. Attention/index heads, output groups and vocabulary must
partition. Invalid/malformed/conflicting-repeat flags fail on config import.
add_parallel_arguments(parser) registers flags for executable CLIs;
parallel_topology() returns lowercase tp/ep/dp/world/local_experts. This is a
configuration domain, not a claim every topology has native/full-real fit proof.

Actual exchange functions are now moe_ep / golden_moe_ep. No compatibility fork
or duplicate compute was added. Main migrates external imports/callers. Existing
all-rank zero-active protocol and typed Tensor/TASK_ID completion are retained.

Two concrete smallEP extent defects repaired. Per-source distributed lanes keep
capacity and N_RANKS*capacity max routes/expert. Local compute slabs independently
use ceil(N_RANKS*capacity/16) hardware16-row chunks, with padded slab_stride;
compact and return indexing both use that padded stride. Clear owns all padded
rows; expert tasks retain16 rows and zero weights on padding. No EP16 geometry
change for positive capacity. COUNT_PAD now ceil(N_LOCAL/32)*32 INT32 slots,
providing192/96/64/32 for EP2/4/8/16 and128-byte-disjoint metadata rows; old32
would truncate local experts at smallEP. Both are storage extents, not math or
communication protocol changes. Compact scalar-store warnings remain conservative:
slab_w/slab_route rows each16*4=64bytes and expert padded stride gives disjoint
row ownership. Generated/native liveness still needs independent proof.

TP reduction already uses configuredTP peer loops/groups. Numerical algorithm,
signals, dynamic bounds and dependencies unchanged. Its existing standalone test
HOST/rank input partial is now explicitly InOut because the existing reuse anchor
writes the same value. Initial compile warned InParamWritten; only the two test
formal annotations were corrected and TP-only signatures regenerated. No warning
suppression or compiler workaround. Initial artifacts/logs are retained.

## Actual validation (self-evidence only)

- task_20261108_112536_120403119287 --no-device COMPLETE0:220 controls,13valid
  pairs/both flag syntaxes/default and invalid flags; actual DSL slab assignments
  source-evaluated for capacities1/2/3/17 with independent extent/uniqueness checks;
  actual golden route/return/idle function exercised with a bounded independent
  algebraic expert oracle. Distinct source values and forced remote destinations
  check actual counts/sentinel preservation; physical-capacity negative differs.
  Exact original numerical golden body AST preserved. TP group sums are analytical
  controls, not execution of a native collective. No full expert math fixture.
- task_20261108_112737_121474511280 and
  task_20261108_112737_121474626844 --no-device COMPLETE0: actual moeffn count
  CAPACITY1 plus public tp_reduce_test signatures compile for TP1EP2 and TP8EP16.
  Explicit a2a3/ONBOARD/ring[0,0,4GiB,0], no TensorSpec.create_tensor or prepare;
  COUNT_PAD exact live values/extents asserted. Initial TP input-direction warning
  preserved in *_initial.log; count artifacts stay valid (TP test-only correction
  is outside their dependency set).
- task_20261108_112905_12247186866 and
  task_20261108_112905_122472127228 --no-device COMPLETE0: corrected TP test ONLY
  signatures compile both profiles, warning absent. No duplicate count compiles.

All actual tasks use task-submit; raw logs and per-program file manifests/metadata
are in this directory. compile.py selects signature-only entry.compile(config)
with sample creation and publicprepare fenced. No fixture/header/tensor payload,
NPU/native device allocation, runtime edits, fullentry compile or full-real claim.
The count companion is the existing accepted rank-leading HOST/ND→CHIP/NZ probe
adapted solely to actual import-time EP_SIZE and capacity1; it calls actualmoe_ffn.

Native EP2/TP1 and EP16/TP8 routing/TP reductions/zero-active/counts and full-entry
checks are pending separate Root GO and independent validator. EP8 fullE40 whole-
expert residency remains infeasible under current device budget; configuration
acceptance is not a full-materialization authorization. Main integration owns
new shape/resource/plan/admission bindings and actual real generation milestone.

Worker measured record is worker-agent only, supplied once to Main for import.
No shared TIME_LOG/TASKS writes. All worker jobs ended, sources frozen/PARK.
