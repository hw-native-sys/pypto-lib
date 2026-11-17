# Checkpoint pipeline v2 — actual adapters wired, native/large IO UNTESTED

Only this NEW checkpoint_pipeline_v2 directory was written. Old sealed
checkpoint_pipeline92a810 evidence, root45 sources, runtime/harness, active
reviewer helper/process and reviewer outputs were untouched. No NPU, fullprefix,
large IO, actual static materialization or real checkpoint payload was used.

pipeline.py is a concrete five-stage CLI, not a TODO adapter:

1. static-plan reconstructs the actual frozen seed0/reused1expert-layer prefix
   fixture, compacts2 pages, disables Engram and hashes all57 statics logically
   in8MiB chunks without saving global stacks. Saves Torch/Python/thread/env /
   seed recipe, source hashes and separate prefix/decode static header manifests
   (established C1 singleton adaptation). This REAL branch was not run.
2. prefix reconstructs/verifies every static byte hash BEFORE upload, clones
   initial ALLnonstatic inputs, uses the existing actual helper prepare/shared
   bank/free guard and public owner, dispatches, then atomically saves copied
   actual+initial nonstatic bytes BEFORE EITHER CPU reference. Saves copied
   compiled ABI JSON, original artifact identity, mode, actual owner config,
   source hashes, Torch RNG bytes/hash and recipe. It closes the owner and only
   then publishes hash-bound owner_closed. No numerical receipt is created.
3. validate independently rebuilds/byte-verifies static fixture values, loads
   actual and initial copied tensors and validates the owning module unchanged:
   PREFIX17 exports, UNCONDITIONED full40 reference first, then original
   validate_golden comparators with lazy SECOND STAGED40 reference. Direct gates
   and unchanged thresholds are preserved; unconditioned logit max delta is
   saved. Both scopes and exact output keys must pass before atomic CPU receipt.
4. decode requires accepted prefix receipt/close/source/ABI/static identity,
   restores ACTUAL prefix state, prepares ONE decode artifact/owner, dispatches
   128/129/130, preserving old prefix rows, and atomically saves each complete
   actual/input checkpoint immediately. Owner epoch is unique PID+UUID and is
   common to all four calls. The stale129 control restores ACTUAL saved128
   caches and wrong128 metadata against a separately cloned correct129 oracle.
   Runtime exceptions never count as rejection. All four calls close before
   per-checkpoint owner_closed markers. Native-only chain remains UNVALIDATED.
5. validate-chain verifies phase/positions/source, parent checkpoint chain and
   identical owner epoch/compiled ABI/artifact across three positives+control;
   runs DECODE19 unchanged gates with BOTH references for each positive, then
   original numerical published WINDOW/C2-row rejection for stale129. Final
   receipt binds prefix/step CPU receipts and control. A toy receipt can never
   unlock production/native stages. Failed revalidation removes stale own PASS
   receipts first; immutable copied bytes are retained for retry.

The actual RealProvider imports the active helper READ-ONLY and uses h.prepare,
h.g/h.p/h.d, clone/static helpers and original validation/comparator functions.
It verifies actual imported module __file__ and SHA against the frozen_source
copy plus all45 root hashes, preventing cached draft modules from being used.
Real default CLI refuses before provider/fixture construction. --execute needs
later rootGO with exact pipeline/store/helper/root-manifest hashes, assigned
stage and measured positive host/disk largeIO budget. Native stages additionally
require16 queue devices. No such authorization file is provided. The16MiB toy
cap remains unless this guarded REAL route is authorized; no environment/private
attribute bypass is used. Work/checkpoint destinations refuse reuse before any
native prepare. Failed partial decode must restart as a newly assigned whole
phase on a fresh work copy; it cannot claim continuity of a lost owner.

## Actual CPU checks and remaining uncertainty

The SAME pipeline.py CLI was executed in separate CPU processes with tiny
ToyProvider tensors through all five stages. This exercises actual persistence,
receipt/close gates, actual-byte restoration, static reconstruction contract,
unique owner/artifact/position/parent chain and final control flow. The fake
provider is explicitly CPU_MOCKED/TOY_ONLY_PASS, not numerical model evidence.
Tests reject native-only prefix as a decode gate, real CLI without rootGO,
corrupt payload, altered owner identity and missing/invalid receipt. Serialized
payloads exclude static w. Final revalidation invalidates stale PASS markers.

check_real_bindings.py imports the ACTUAL frozen model modules and executes
owning _device_outputs/comparator key construction with META tensors only,
while fixture builders and PreparedDispatch are patched to DENY invocation.
It confirms prefix17/decode19 schemas and complete nonstatic argument names.
Sealed readiness metadata confirms42 nonstatic tensor arguments/7,609,876,736B
prefix and38/247,111,680B decode; scalar arguments are separately persisted as
0d tensors. No actual7.6GB buffer was allocated.

Initial task114721_409482818759 reached the toy stage sequence but failed the
TEST's readiness-file parent path. The raw failure is preserved. Corrected
114825_41003157293 passed the actual CLI sequence. Extended115013_41334002265
passed CLI plus actual-frozen-module META bindings. Final run/hash is recorded
in FREEZE.json after small source/config/RNG metadata hardening.

REAL static seed/byte reproducibility at full logical103GB, actual atomic15GB+
IO throughput/disk/host/SHM peak, NPU terminal/copyback/close behavior, numerical
17/19 gates and real same-owner3step/control remain UNTESTED. The code branches
are wired, but no hardware or large IO execution is authorized by this artifact.
Future receipts must use measured budgets; positive allocation/checkpoint hashes
are not numerical acceptance. Root must independently review/test native-only
branches before a future assigned run. Current long reviewer run is unchanged.

Final CPU task115250_41698095476 COMPLETEexit0, including the full actual CLI /
tiny-provider sequence and real frozen-model import/META binding checks. The
last REAL-only guard hardening additionally requires helper-dependency hashes
and a hash-bound measured largeIO receipt. Those REAL positive authorization /
large-memory branches remain deliberately unexecuted; no valid rootGO file or
fabricated measurement is supplied. Current source syntax loads in the passing
CLI tests. No further source edits/jobs occurred before sealing.

The copied checkpoint includes its compiled ABI JSON, so its metadata survives
loss of the original prefix build directory; source and artifact identity remain
bound. Parent/owner/artifact chain checks run before offline chain validation.
A failed/repeated native phase cannot overwrite old checkpoints: it must restart
on a fresh assigned work copy; CPU revalidation invalidates old own PASS receipts
before checking bytes. Actual cached/buffer lifecycle, source-reconstruction
memory and full IO durability still require separately authorized testing.
