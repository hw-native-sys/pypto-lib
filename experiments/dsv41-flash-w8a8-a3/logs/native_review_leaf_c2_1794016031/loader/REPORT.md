# True-leaf streaming loader — CPU-validated worker handoff

Exclusive model scope: weight_loader.py only. Main owns generate.py/upload/
coverage/cleanup; C2 worker owns compressor.py. No other model/runtime/harness/
checkpoint writes, shared ledgers, NPU jobs or real checkpoint payload reads.
This is implementation self-evidence, not independent acceptance or native
streaming/real-fit/generation proof.

## API

WeightLoader.argument_contract(argument,layer_count=None) returns a rank-local
shape tuple and torch dtype using model/schema metadata only. It handles dense
layer axes, routed experts, source-slot families, aliases, top-level and Engram
resource contracts without a weight read. It matches every static argument in
the preserved real40-layer compiled metadata, including32-head index replication.

WeightLoader.iter_argument_leaves(argument,rank_shape,dtype,with_engram=False,
zero_chunk_bytes=64<<20) yields WeightLeaf with fields:

- argument,rank,byte_offset,tensor,rank_shape;
- logical_prefix=(layer,local_expert) for experts or(layer,) for dense/source;
- source_names for routed weights/offsets/scales (optional empty for other
  resource types,whose names/transforms remain in the owning provider/schema).

Descriptors reject invalid EP rank,nonpositive shape,empty/non-CPU/noncontiguous
payload,negative/misaligned/out-of-bounds byte offsets. Iterator validates rank
shape and dtype before reading a leaf,then checks transformed payload shape.
Main must still validate complete monotone coverage,worker/rank mapping and
allocation cleanup; those are upload-owner responsibilities.

Routed leaves use existing expert_tensor/expert_scale, preserving the retained
F14 folded return,representability rejection,transpose,per-matrix NZ order and
FP32 scale vectors. Offset=(layer*N_LOCAL_EXPERTS+local)*leaf_bytes. Dense/source
leaves use new layer_argument,which shares the existing transform helpers.
layer_dense now composes that named provider and preserves old optional
dependency groups and aggregate output values. No new stream invokes
layer_dense,expert_shard,materialize,stacked or torch.stack.

Source slots remain C2 layers2/8/14,C1 layer20,reindex24/28/32/36. TP vocabulary/
attention/group slicing,indexer replication and gate aliases stay unchanged.
Disabled Engram resources validate their existing ABI and yield bounded zero
chunks with exact offsets; no real table/matrix/rotation read occurs.

The iterator releases its own payload reference when resumed. The consumer must
delete the previous leaf AND byte/tensor views before next(iterator);ordinary
Python for-loop variables otherwise retain a prior leaf until reassignment.
Main explicitly acknowledged an upload helper followed by del leaf. No loader
payload cache or descriptor list is introduced. Reader metadata/consumed names
are retained,not tensors. Existing reader full-source backing is still part of
the readiness memory allowance;copy chunks alone do not bound it to64MiB.

## Verification

Final queue job task_20261107_093733_94784622146 recorded in final_queue.log
exits0; scoped ruff and py_compile pass on final source. The task activated
activate-pypto.sh, then executed python check.py, python check_headers.py,
ruff check models/deepseek_v4_1_flash_a3_w8a8/weight_loader.py and
python -m py_compile models/deepseek_v4_1_flash_a3_w8a8/weight_loader.py
(artifact scripts use their full repo-relative paths). check.py uses a
synthetic reader only and validates:

- 2ranks×3layers×2experts for all six routed weight/scale arguments; independent
  explicit16-row/32-column NZ tile visitation plus asymmetric q-offset,transpose,
  scales and byte reconstruction;
- every dense/optional/Engram aggregate field against the preserved original
  loader at four synthetic layers,plus representative streamed dense/source
  aliases and fixed source slots;
- forbidden aggregate-method spies and weakrefs showing old payloads die when
  the consumer releases each leaf before advancing;
- disabled bounded zero resources with a payload-denial reader;
- nine effective controls:negative/unaligned/overflow descriptors,fractional/
  out-of-range offset folds,wrong dtype,wrong rank/layer/expert data.

check_headers.py patches the payload method to reject and compares metadata-only
leaf contracts to every static real40-layer compiled argument. It creates no
checkpoint loader payload and no large fixture/device allocation. Metadata
comparison is a schema proof,not current native acceptance or real fit.

Raw helper failures are preserved: synthetic HC scale naming,missing expert key
roster,relative metadata path,and dtype alias. They are test-fixture mistakes;
no compiler/runtime fault or numerical-control success is inferred from them.
initial_pass.log records the first full synthetic positive before final metadata
coverage additions;final_queue.log is the final combined positive.

## Remaining gates and recovery

Main must integrate actual empty-rank allocation,serial copies,complete interval
coverage/failure cleanup and supported scalar marshaling,then run bounded native
offset-copy/readback/reuse/source-mutation/state/RSS/HBM proof. This worker ran
NO native/device test and claims NO peak,full-weight fit,text or C2 fix. Real
weights stay held behind the current full-materialization guard.

Fresh CPU reruns: copy check.py,check_headers.py and weight_loader_before.py into
a fresh artifact directory at the same logs/unit/worker depth,then run both
scripts through task-submit --no-device from repo root with activate-pypto.sh.
check_headers.py resolves preserved compile metadata relative to logs;adapt that
path in the fresh copy if using a different directory depth. Do not overwrite
the sealed worker results. Source hashes,results,manifest and unique measured
record accompany this report. Worker PARKED after final collector exits.
