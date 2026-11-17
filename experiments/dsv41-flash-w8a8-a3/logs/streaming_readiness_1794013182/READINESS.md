# Streaming weight ownership readiness — read-only

Current reviewed45-source manifest remains
1e9496bfdeb177c13f00a2c813a3ecde3242b5bbb622ae44c5a1b5a14377ae43.
This new artifact prepares a future bounded unit; it implements no model/runtime/
harness change and starts no worker,NPU job,checkpoint payload read or full
allocation. Independent frozen acceptance continues separately. All source
hashes matched before/after the queued CPU preparations. Main PARKS after this
handoff; implementation needs the next observer assignment.

## Concrete next milestone

Replace full-host rank materialization/global torch.stack with one-leaf streaming
into empty rank-owned device allocations. Prove the exact driver owner on a
small asymmetric synthetic source, including scale/NZ/offset correctness,
coverage,one-pass uploads,repeated native dispatch,phase cleanup and bounded
host/staging lifetime. Keep real-weight allocation HELD. A successful tiny proof
does not establish complete-model HBM fit,decode numerics or text generation.

This can be implemented with public installed APIs; no runtime/harness edits
are needed. Future exact ownership: Main owns generate.py and integration;
one worker exclusively owns weight_loader.py and its synthetic tests. utils.py,
weight_map.py,both forwards and all operators stay read-only initially. If a
bounded row-reader later becomes necessary,record transfer of utils.py first.
No original Engram hashing is needed for decision10's noEngram first path.

## Supported APIs and compile path

| Operation | Existing source and contract |
| --- | --- |
| Signature-only compile | JITFunction.compile(config=...) with NO tensor arguments reads full Tensor annotations; dynamic dimensions and runtime Scalar parameters need no values. pypto/jit/decorator.py:1258 onward. |
| Empty rank allocation | DistributedWorker.alloc_tensor(tail_shape,dtype,init=None,worker_id=r); tracked by that worker,rollback on failure. runtime_base.py:254. |
| Incremental upload | DistributedWorker.copy_to(shard.data_ptr,leaf.data_ptr(),nbytes,dst_offset=offset,worker_id=r). Base allocation pointer plus explicit byte offset; bounds checked. distributed_runner.py:2768 onward. |
| Handle aggregation | StackedDeviceTensor(shards,full_shape,worker_ids=range(16)); each shard has shape full_shape[1:],same dtype,and must reside on the worker used by device=r. device_tensor.py:137. |
| Cleanup | free_tensor(shard,worker_id=r) or free_stacked_tensor; complete all cleanup attempts,close owner on any error. Empty-stack construction already appears in golden._alloc_empty_stacked_tensor; reuse public APIs,not that private helper as an application dependency. |
| Transport/lifetime | Explicit copy_to stages post-prepare contiguous CPU sources through a runtime-owned shared Buffer and releases it after the copy. Dispatch IO still requires stable shared mappings before prepare. Do not pass short-lived ordinary host tensors as per-call shared arguments. |

Queued CPU check_signature_compile.py actually compiled the current40-layer
decode,EXPERT_LAYERS40,from its signature alone: ONBOARD target,metadataa2a3,
expert prefix[16,40,24],zero supplied tensor arguments,zero fixture builders,
zero device dispatches. signature_compile.json records the artifact/hash.
This removes any need to build635GB torch.empty fixture arguments just to compile.

Future Main constructs a metadata-only concrete EntryParam/spec contract from
the annotations and explicit dimensions: prefill tile,decode1,batch1,physical
cache pages,index-table width,state blocks and pool2048. Preserve scalar/layout/
direction/argument order. Compile without samples; allocate only genuine IO.
Do not call build_fwd_specs/materialize_all_ranks/_assemble_rank_weights to
obtain this contract. Current fixed-capacity first path remains explicitly
bounded; this does not close general F09 dynamic/multi-request requirements.

## Streaming sequence and exact byte layout

1. Define/validate all shape/dtype/layout/source descriptors without payload.
   Establish contiguous shared host metadata/cache/output buffers before prepare.
   Open one phase owner with persistent=False so protocol domains are fresh per
   dispatch. Do not manufacture a full host tensor for any static argument.
2. Allocate each static argument's full per-rank device tail with init=None.
   Track (argument,rank,allocation); aggregate a public stacked handle only when
   all rank shards exist. Do not dispatch until every required byte is filled.
3. Iterate one transformed leaf at a time,upload serially in <=64MiB copy calls,
   then drop its source/transformed references. No concurrent16-rank upload
   pool,no layer/rank lists of payloads and no global torch.stack. Keep the source
   alive until copy_to returns. Header/schema metadata may remain cached.
4. Maintain a coverage ledger of destination byte intervals. Require no gaps,
   overlap,wrong rank or out-of-bounds writes before marking an argument ready.
   On any provider/upload failure,prevent dispatch and free every owned shard.
5. Dispatch through the SAME prepared owner with retained handles and shared IO;
   prove later calls perform no static reupload and ignore host-source mutation.

| Family | Leaf and destination prefix |
| --- | --- |
| Routed W1/W3/W2 | Use existing expert_tensor(layer,local_expert,which);globalexpert=r*24+local. Tail[40,24,K,N]. Offset=((layer*24+local)*matrix_elements)*itemsize. Pack each matrix independently before byte slicing. |
| Routed scales | Existing expert_scale returns FP32 per-output-channel vector;W1/W3 length2304,W2 length5120. Offset=(layer*24+local)*vector_bytes. No dequantization or scale dtype change. |
| Dense40-layer weights | One named transformed layer argument;offset=layer*rank_layer_bytes. Preserve every existing TP slice,head/group reshape,HC padding,FP32/BF16 boundary and alias gate_weight->gate_w/gate_bias. |
| C2/C1/reindex resources | Existing source layers2/8/14,20,and24/28/32/36 map to their current source-slot axes. Preserve index-specific TP/replication rules exactly;do not reinterpret them while streaming. |
| Embed/head/final norm | Preserve TP vocabulary slicing and replicated norm. One TP leaf can upload sequentially to ranks tp,tp+4,tp+8,tp+12,then release it;no four-copy host stack. |
| Disabled Engram ABI | Keep engram_enabled=0 and bind the unchanged resource shapes. Initialize inert resources with bounded zero chunks;do not read tables or real Engram matrices/rotation. No uninitialized-resource assumption or Engram restoration claim. |

Existing WeightLoader._read has no payload cache;SafeTensorReader retains headers
only. However materialize(layer,name) currently expands an entire layer_dense or
expert_shard,and expert_shard/stacked retain lists before stacking. The new
stream must bypass those aggregate paths. Reuse expert_tensor/expert_scale for
routed leaves;factor dense named transforms into a shared one-argument provider
so layer_dense can compose the same implementation without a duplicated math
path. Proposed descriptor/yield fields:argument,rank,logical source prefix,
destination byte offset,contiguous CPU tensor,source names,dtype/layout and
expected rank shape. Existing public aggregate APIs may retain semantics for
their remaining callers;the new driver must not use them for real loading.

W8A8 correctness is unchanged:fold q-offset only when representable in INT8,
retain the F14 returned folded payload,reject fractional/out-of-range folds,
transpose [N,K]->[K,N],then pack_nz. Preserve per-channel FP32 scales separately;
do not multiply them into INT8 values. Already-rotated/norm-folded checkpoint
weights stay as exported. Host metadata may label a stacked argument ND while
individual leaf bytes remain NZ;that is not permission to upload row-major
matrices. NZ rows/C0 alignment and exact byte order must match utils.pack_nz.

## Header-grounded lifetime budget

check_metadata.py denies SafeTensorReader.tensor;zero payload calls occurred.
It read applicable non-Engram headers from263 shards,not real weight arrays.
Current metadata shows:

| Quantity | Bytes |
| --- | ---: |
| Static rank weights in the real40-layer ABI |39,624,816,576 per rank |
| Largest resident single rank argument W1 |11,324,620,800 |
| One routed INT8 expert matrix[2304,5120] |11,796,480 |
| Largest full INT8 source Wq_b[32768,1280] |41,943,040 |
| Largest non-Engram source embed[129280,5120] BF16 |1,323,827,200 |
| Maximum copy staging chunk proposed |67,108,864 |

One expert leaf is about11.8MB;the device W1 allocation is11.3GB,which does NOT
require an11.3GB host source. Scale/leaf ordering is explicit. Serial chunked
copy_to bounds the upload staging live at64MiB. Existing reader.tensor still
reads a whole SOURCE tensor into bytes plus bytearray;_vocab_slice can keep that
entire1.32GB backing alive through a smaller TP view. Therefore claiming that
all host payloads are below64MiB would be false unless a bounded source reader
is later implemented.

Conservative LIVE payload planning allowance with existing reads is
max(24*largest_INT8_source,4*largest_dense_top_source)+64MiB =5,362,417,664 bytes
(about5GiB),not16/40 copies. The24 factor covers INT8->FP64 offset-fold and
round/check temporaries;the4 factor covers full source bytes/bytearray,
converted/sliced/packed backing;the staging term is one serialized copy chunk.
Add actual shared IO,initial/reference clones,headers/indices,compiler buffers,
process baseline and runtime bookkeeping. Python/CPU allocator RSS reservation
is NOT bounded by this live-reference arithmetic;measure its high-water plateau
over many leaves and verify payload weakrefs/references are dropped. No claimed
absolute RSS or full-model fit follows from this allowance.

The current bad path retains rank_args/expert_by_rank,then another global
torch.stack,and possibly upload staging. Those terms disappear from the
streaming design. Static device weight total remains about39.6GB/rank,plus
cache/selection/output buffers,EP windows and actual graph transient high-water.
Do not sum only static weights or extrapolate a tiny512B peak. Real HBM and host
fit remain UNKNOWN;all real loading guards stay active until measured gates pass.

## Phase and cache ownership decision

Keep sequential phase owners initially. Run prefill,read back actual mutable
state into its stable shared buffers,validate/copy the required persistent
prefix,close/free prefill owner,then stream decode's static weights into its own
owner. Handles cannot cross independent prepared-worker lifetimes. This costs a
second streamed checkpoint pass;it avoids overlapping two full device weight
sets and is a correctness-first supported route. Cross-phase device-handle
sharing would require a separate supported lifecycle design;do not assume it.

Within decode,retain static handles over all steps. Cache/state remain stable
shared-host H2D/D2H each dispatch,exactly as the reviewed owner proposes. No
resident-KV or zero-D2H claim. Keep actual-returned-state versus oracle-clone
isolation,per-step paging/RoPE refresh,and fresh protocol domains. Any future
resident cache extension needs explicit readback/handoff and independent gates;
it is unnecessary for this first streaming unit.

## Exact bounded implementation/validation gate

Future Main: generate.py only,metadata-only contract,stream-aware existing owner,
partial-allocation cleanup,coverage and sequential phase handoff. Future worker:
weight_loader.py only,named-leaf iterator plus existing-transform reuse and
synthetic source tests. No model arithmetic/operators/fwd/golden changes;no
runtime/harness edits. Both write exclusive artifact roots. Independent sibling
reviewer acceptance remains observer-owned.

First CPU milestone: asymmetric tiny BF16/FP32/INT8 sources with at least2ranks,
3layers and2experts;independent analytic q-offset/transpose/NZ/scales plus
existing-loader comparison. Reconstruct streamed bytes and compare EXACTLY;
reject swapped rank/layer/expert offsets,wrong scale vectors,incomplete/duplicate
coverage,nonrepresentable offset,wrong dtype and wrong-worker handles. Prove no
aggregate materialize_rank/expert_shard/torch.stack path runs. Fault-inject each
allocation/upload to verify all partial shards free and no dispatch occurs.
Prove signature-only compile uses no global fixture tensors.

Then one bounded explicitONBOARD native job using the exact streaming owner and
synthetic payloads (fewMiB,never real checkpoint weights):empty public allocations,
multiple nonzero destination offsets,exact readback/canary gaps,three repeated
dispatches with changing runtime scalar/metadata and cumulative actual state,
host-source mutation after upload,stable static handles and cleanup. Record
bytes,live CPU payload/staging count,RSS/shared memory,per-device baseline/peak,
normalisedONBOARD,a2a3 metadata and native logs. Compare a one-leaf source with a
many-leaf source at fixed chunk size to establish host high-water plateau,not
growth with total model bytes. Phase close/reopen must invalidate old handles
and carry correct shared host cache bytes. Wrong-byte/rank/scale numerical
mutants must reject by data comparison. Invalid dtype/offset/worker contracts
must reject through explicit validation before dispatch;a device fault is never
a successful numerical control.

Only after this bounded unit AND the separately reviewed full-entry/continuation
functional gates pass:measure the actual synthetic full-forward IO/window/
transient peak,add the real static-weight delta and documented allocator margin,
validate largest-contiguous-allocation headroom on every allocated device,and
stage further bounded memory-fit trials. Do not allocate a124GB replicated input
or read full real weights while those budgets are unresolved. Full model loading
and real greedy tokens/text remain later gates. No ETA/percentage is justified.

Readiness CPU failures are preserved:attempt1 wrong artifact-relative manifest
path;attempt2 missing fp32 dtype alias. Corrected header/API check passes.
They changed no model or checkpoint bytes and are not runtime/negative controls.
Final source hash check and measured Main ledger accompany this report.
