# Configurable topology integration — early source checkpoint, FINAL / PARK

No concrete new blocking source bug found in the exact inspected snapshot. This
is NOT coherent-freeze, CPU, compiler, runtime, resource-fit or mathematical
acceptance. Main was still completing its implementation/metadata/tests; no freeze
was requested. All45 files were copied once before inspection with per-file SHA,
byte length and read timestamp in snapshot_manifest.json. They are exact inspected
bytes, not an atomic/live current45 declaration. No imports/tests/headers/fixtures/
compile/prepare/NPU/payload/grants or protected/shared source writes. Root will
assign a single frozen-candidate independent batch later.

## Implemented prior findings, checked in snapshot

- config.py167ff defaults TP4/EP16, finite TP1/2/4/8 and EP2/4/8/16, accepts both
  flag spellings, rejects conflicting repetitions/missing values, requires EP>=TP
  and EP divisibleTP. WORLD=EP, DP=EP/TP; local experts=384/EP. Dependent shapes
  freeze at import; executable parsers consume the same flags.
- moe_exchange55 COUNT_PAD=ceil(N_LOCAL/32)*32 preserves128B row alignment and
  holds local48/96/etc instead of fixed32. Buffers/consumers use that symbol.
- moe_exchange107..123 now computes ceil((EP*capacity)/16) chunks, padded
  slab_stride=chunks*16 and recv_rows=N_LOCAL*slab_stride. Crucially compact_flat_row
  at231 AND return_base at291 use the same padded expert stride; expert chunk
  views use expert_id*chunks. Full clear covers all padded rows. Raw network
  lanes/aux_stage/return_stage retain actual EP*capacity layout; actual route
  counts are not replaced by padded token counts. These are coherent source
  repairs, not just a ceiling expression with stale reshape/offsets.
- Exact snapshot search found no executable moe_ep16/golden_moe_ep16 references
  after rename to moe_ep/golden_moe_ep across the45 files. Callers carry the typed
  Tensor/TaskID tuple and active-prefix propagation. Hardware GEMM/aux tile16,
  HC_MULT4, state parity2 and layer-family period4 remain architectural constants.
- Indexer37/layer_reference52 still compute ALL32 replicated index heads. No new
  model tp_score_reduce call appeared. No A5 MXFP4/MXFP8 path was transplanted;
  routed weights remain INT8 NZ, scales FP32 and activation/output BF16 boundaries.

## New admission/source/CLI branch

Snapshot generate2427 parallel_operation_budget derives static and nonstatic IO
from actual annotation contracts; window rows/metadata depend on EP/TP/COUNT_PAD.
All returned quantities are labelled CONSERVATIVE_SOURCE_PLANNING_NOT_PEAK.
max(old_floor,derived_bound), old minimum communication allowance and64GiB HOST/
2GiB DISK/16GiB SHM floors are conservative policy, NOT new measured peak or fit.
Post-prewarm/free checks still charge remaining static+positive reserve, without
subtracting ring/committed twice. SmallerEP realE40 static denial remains relevant:
EP8 routed weights alone exceed recorded physical card capacity. No actual new
resource plan or readback/forward measurements were reviewed here.

New configurable schema/operation is explicitly shared-only. Receipt and plan must
match effective topology and derived budget; expert_layers40 and complete57*WORLD
static target coverage are required. Each initial budget/IO check uses configured
contracts. IDs/remaining arrays/HBM query ranks/validate_contract/submit device count
now use WORLD_SIZE. CLI forwards --tp/--ep to the normal child. Greedy TP candidate
and DP-vote loops derive from TP_SIZE and WORLD_SIZE/TP_SIZE, rather than four DP
votes. Changed topology still needs concrete data/negative/ABI proof at freeze.

Configurable source binding compares plan.source_manifest_sha256, receipt source
and reviewed-source digest, then exact45 pathset/live hashes/whole manifest digest
and ALL45 expected hashes INCLUDING generate.py. The manifest is supplied externally
by a reviewed plan/Root receipt; no new self-hash literal/cycle is introduced.
Source hash ALONE is insufficient because import-time topology changes shapes;
topology/plan/receipt/config/ABI must remain jointly bound in later actual evidence.

Legacy schema is refused outside default TP4/EP16. At default it retains old hard
baseline8153 and existing generate-only math comparison exception, while actual45
files remain checked. HARD default still raises without explicit authorization.
Historical calibration/empty/transfer receipts remain historical prerequisites;
new configurable branch does not masquerade them as new fit measurements or use
an old physical mapping as a current assignment. Actual configured ordered IDs
must match the new receipt. No valid Root runtime receipt was created/tested here.

SharedWeightOwner/PreparedDispatch/source loader/complete-byte upload/early sharedIO/
C1 singleton views/pointer+byte provenance/free canonical bases once/normal close
remain the intended existing path. New branch uses it, not a parallel driver.
Canonical completion and zero decode static IO checks remain enforced. Checkpoint
index/header-before-source-read/leaf lifetime and durable coverage paths remain.
Source-only review does not establish implementation-level close/firsterror or
physical memory behavior; existing native proof remains its original topology.

## Finite remaining acceptance points — one later batch, not new prerequisite rounds

Use coherent frozen hashes/config context, default inheritance plus TP8/EP16/DP2
normal metadata/ABI and requested smallerEP/T1 padding boundaries. Meaningful
negatives should exercise actual admission/topology/source/plan/accounting predicates,
not just a metadata self-comparison. Reuse already generated fresh artifacts when
appropriately bound; do not recompile merely for confidence.

The existing MoE golden still accepts one replicated local expert-weight shard;
layer_reference._moe_weight_shard takes rank0 for that fixture. That is an explicit
coverage limit, not evidence of different per-rank expert owners. An independent
small nonidentical expert-owner oracle/control belongs in the single acceptance
batch if claiming that placement. Similarly TP vocab-boundary/unique-shard winners
and every requested DP vote should catch symmetry-masked grouping mistakes.

New real header plan must bind topology-specific57 target shapes/bytes/read
schedules, sameowner alias counts, source45, actual compile profile and ordered
allocation. Resource/cleanup review is separately owned; do not infer these from
copied old floors. Quantization/original-parity/noEngram scope remains unchanged.
Current case/source acceptance claims are not extended; observer race68 remains
unapproved/unapplied. No actual text/NPU authority is supplied by this checkpoint.
