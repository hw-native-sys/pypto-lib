# Recoverable split validation — artifact prototype and exact future integration

Status: CPU toy persistence protocol implemented/tested; full model/native stage
integration NOT implemented or authorized. split_checkpoint.py deliberately
refuses state payloads over16MiB and every production/native resume. No current
prefix tensors were saved; no running process was inspected, injected or changed.
reviewer_helper_reference.py is an immutable read-only copy of the active helper,
not a replacement or runnable new NPU entry. Root model45 sources are unchanged.

## Why the current run cannot be recovered from this preparation

The current helper dispatches prefix, then validate('prefill128', ...) computes
expensive CPU expected/staged goldens BEFORE close_owner. Its actual tensors live
only in that process. This worker has no authorization to inspect its memory,
change it or save its tensors. If that job stops without a checkpoint, this
prototype cannot fabricate/reconstruct actual native outputs from goldens.

## Tested library

save_checkpoint atomically publishes actual.pt, initial.pt and manifest.json by
fsync+directory rename. Destination must be fresh. Only nonstatic argument values
are serialized; values are independently cloned to copied CPU storage. Initial
oracle inputs must not alias actual storage. A static manifest records header /
logical-byte hashes, but no static tensor payload. Expanded/noncontiguous static
views are hashed recursively with8MiB byte chunks, avoiding a global contiguous
stack materialization. Source file hashes, compiled ABI metadata hash, ONBOARD /
a2a3 mode, phase and owner epoch are required and rechecked on load.

load_checkpoint verifies file and per-tensor dtype/shape/byte hashes, restores
independent actual and initial tensors, then requires deterministic static recipe
reconstruction with EXACT static-name/header/byte-hash match. A changed seed,
missing key, source, ABI or payload fails rather than using a reference cache.
Toy cpu_validate only publishes a hash-bound receipt AFTER its validator returns
all requested outputs as passing; native bytes alone do not create a gate.
All receipts are explicitly toy-only and native_resume_authorized=false.
require_resume refuses every production use. No CLI triggers native work.

## Concrete three-stage future helper

An assigned/reviewed production adapter must implement these stages in NEW
outputs, never edit the active helper or treat this toy prototype as acceptance.

A. CPU preparation (--no-device). Pin all45 source hashes, actual entry parameter
order/dtypes/shapes/directions, platform/mode, token/seed/fixture version,
DEEPSEEK_PER_LAYER_EXPERTS=0 and page compaction plan. Reconstruct the owning
synthetic fixture using p.build_fwd_specs(tokens=128,seed=0,compile_only=False)
(which calls torch.manual_seed(seed)), compact pages to2, overwrite actual
input IDs/plan exactly as the current helper and disable Engram. Hash ALL57
static arguments once while NO cards are held; save only recipe and hashes,
never the global tensor stacks. The recipe's reproducibility remains unproven
at full40 scale until a separate reconstruction yields the identical manifest.
Unknown/nondeterministic initialization or changed hash must abort.

B. Future separately authorized native prefix stage. Rebuild the frozen recipe
and verify every static hash before preparing. Retain independent nonstatic
initial clones BEFORE dispatch. Use the existing current helper's public source /
shared-bank/free guard and explicit ONBOARD/a2a3 configuration; do not weaken it.
After successful native terminal/copyback, but BEFORE any CPU golden:

    dispatch actual prefix
    persist ALL actual nonstatic entry values and initial oracle clones
    persist actual output/cache/state hashes, ABI and source identity
    atomically publish native-bytes checkpoint (NOT numerical PASS)
    close prefix owner, record close/memory status, exit the NPU task

The production adapter must pass a precomputed immutable static manifest into
its persistence path so it does not rehash111GB logical stacks while holding
cards; the tiny prototype currently hashes inline and is not suitable for that
large path. Saved actual entry values include metadata/token_counts/positions,
all cache/state, layer snapshots, logits, pre_mix and output buffers, not only
the rows a decoder needs. Scalars need explicit dtype/value records or0d tensors.
Mode/generated a2a3/native-log terminal proof and compiled metadata bytes/hash
must accompany the manifest. Failed native runs cannot publish a successful
copyback claim. A complete checkpoint may precede owner-close failure, but that
failure blocks a later GPU stage until disposition; it is not silently ignored.

C. CPU golden stage (--no-device). Verify source/ABI/checkpoint hashes before
loading, rebuild/verify static values, and combine separately loaded actual and
initial nonstatic dictionaries with immutable static references. Set the owning
module's _FIXTURE_INITIAL to the independent initial inputs. Use the SAME
source-defined _device_outputs, _run_composed_golden and make_fwd_comparators:

    expected = p._run_composed_golden(initial,128)
    validate_golden(p._device_outputs(actual,128), expected,
        rtol=.02,atol=.02,compare_fn=p.make_fwd_comparators(128),inputs=initial)

Record the EXACT phase-specific source output names/results: PREFIX17 (3 base+14 grouped), DECODE19 (5 base+14 grouped); the source comparators may
condition on actual previous-layer snapshots. Preserve that staged scope and
unconditioned logit differences without claiming strict cumulative equivalence.
ONLY after every gate passes publish an atomic receipt bound to the checkpoint
manifest, source/ABI/static hashes and exact validator identity. A boolean flag
or successful native runtime alone is insufficient. Rejection/CPU interruption
must leave the immutable actual checkpoint usable for later CPU retry.

D. Future authorized decode NPU stage. Require the accepted prefix CPU receipt,
owner-close disposition and every original hash before prepare. Restore ACTUAL
returned prefix cache/state bytes, not computed expected tensors. Build capacity1
metadata and adapt only the established C1 prefill/decode singleton view. Verify
handoff bytes and clone isolation. Compile/load ONE decode artifact and create
ONE prepared owner for128/129/130 and the stale129 branch; protocol domains reset
per current public helper. Save every actual returned step's complete nonstatic
values and its independent initial oracle clones atomically BEFORE costly
validation. Store parent checkpoint hash, position, artifact identity, owner epoch
and original prefix row hashes at each step; derive control state from ACTUAL
saved128 bytes, not a later positive state or reference.

CPU validation of persisted decode steps can run after GPU release. If CPU gates
are deferred while all three native steps run on one owner, mark subsequent
steps explicitly conditional/unaccepted until EACH earlier checkpoint validates.
Never issue a complete-chain PASS from native-only steps. Preserve one-owner /
same-artifact proof and per-step protocol/memory/terminal metadata. Restore
ACTUAL128 state, build the correct129 oracle inputs, inject only stale128 query /
slot/lens/RoPE metadata, run, then persist the control output and correct initial
oracle. Only numerical rejection of the newly published WINDOW and C2 rows,
with original prefix rows unchanged, satisfies the negative; runtime exceptions
never satisfy it. All19 DECODE positive gates and that row control must pass before a
final chain receipt. Host checkpoint failures and CPU reference failures remain
explicit failures, not recoverable green status.

## Unclosed engineering/fit gates

Full prefix actual+initial nonstatic copies are about2x7.61GB for two-page128
contracts, plus transient copying/serialization/staged goldens. Static fixture
reconstruction and allocator-retained host memory are separate. Production must
preflight measured host/PSS/SHM/disk headroom, atomic-filesystem behavior and large
IO cost. The toy16MiB cap cannot just be removed under this assignment. Need a
separate large-IO adapter authorization/budget and native acceptance. Static
precomputation handoff, all57 hashes/reproducibility, native-close checkpoints,
actual ABI receipts, phase-specific owning-gate CPU stage and same-owner decode/control adapters
are specified above but not run or fully wired here. This is partial preparation,
not ready-to-run full-prefix recovery or an alternative production path.

## CPU toy evidence

Queued task task_20261107_113628_340042520249 exit0. Tested exact roundtrip,
mutation-safe copied bytes, clone isolation, expanded/static bounded hash parity,
no static payloads and atomic directory visibility. Eleven controls reject:
native-only checkpoint as pass, toy receipt for native resume, wrong phase,
wrong static reconstruction, changed source hash, SIM mode, initial/actual alias,
incomplete owning gate set, missing required reference scope, payload-byte corruption and tensor-header corruption.
The per-step example is toy decode128 only. No large tensors, real checkpoint
payload, actual prefix, NPU or full golden was used. See toy_results.json/raw log.

## Required dual-reference scope preserved

The current helper first runs a full UNCONDITIONED40-layer reference. During
validate_golden the comparators lazily run a SECOND40-layer STAGED reference
conditioned on actual previous-layer snapshots. Some direct pool/count gates
consume unconditioned expected values; the unconditioned logit delta is also
reported. Neither pass can be dropped or replaced by the other for speed.
Actual and initial checkpoints must be durable BEFORE EITHER pass. CPU receipts
must bind success of both scopes and the exact phase-specific source gates (prefix17/decode19); a scope-only
PASS cannot authorize decode. Prototype receipts now require explicit
validated_scopes={unconditioned:true,staged:true}; a missing-scope control fails.
Toy scope flags test this protocol requirement only, not real model math.

Phase schema correction: a fixed19 assumption for prefix is incorrect. Future
adapters must derive actual keys from each owning _device_outputs and comparator
map, checking their agreement; prefix exports17 and decode exports19. The toy
library takes explicit expected_names and never hardcodes either count.
