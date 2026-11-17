# Full F09 HOST static slicing — independent source-only preflight

COMPATIBLE with the pinned public StackedDeviceTensor whole-shard indexing contract,
by SOURCE inspection. All57 static arguments have exactly one generated HOST use,
integer leading rank and complete trailing extents matching the actual full-fixture
logical shapes/typed ABI. No named counterexample found. No CPU test/import/getter
execution, tensor/fixture, compile, prepare, NPU, input copies, grant, model/runtime
or frozen-artifact write. This does not prove full native dispatch or numerical
correctness, and authorizes no fullcapture.

Exact full artifact is unchanged independent405 decode_capacity2:
logs/independent_f09_typed_completion_1794079859/build/decode_capacity2;
file-manifest5262e68afa0b53291ed285e31170b3c017674c3c25dfa9aec78eb7fab5777532.
Generated HOST SHA9c7293be934b8b65018535af56c4619da9bfccb1d3b3c32fde6af6668b236050.
Adapter398 mixed table points count-active to45c3b4c7..., count-old to1c09c6ce...,
and FULL to this OLD5262 entry explicitly NOT recompiled. It does not select either
count artifact for full-capture. Source405 remains UNAPPLIED/liveM02 untouched.
Full source/ABI paths and digests are in static_HOST_inventory.json.

The earlier count failure is not reproduced in this full generated source. All six
routed/shared HOST formals are ND (decode_fwd.py1208..1218 and distributed_meta),
while CHIP local formals retain NZ as appropriate. Generated HOST lines75..80 use
w1/w3[rank,0:1,0:24,0:5120,0:2304] and w2[rank,0:1,0:24,0:2304,0:5120].
The0:1 is a COMPLETE trailing singleton expert-layer axis, not a leading Slice.
Shared weights use[rank,0:40,0:D,0:INTER] (w2 transposed dimensions), also complete.
C1 singleton source-weight axes likewise retain full0:1 and pass this source rule.

Actual full producer does not perform count capture's f[name][:,0] selection.
fixture_factory changes request geometry only; capture uses the FULL fixture/static
set and PreparedDispatch.open alloc_stacked_tensor(value.contiguous()).
_layer_expert_weight1567ff builds[16,24,D,INTER] then unsqueeze(1)/expand with E1,
so full w1/w3 shape[16,1,24,5120,2304], w2[16,1,24,2304,5120]; it retains the layer
axis. _layer_weight1561 builds[16,40,*tail] for shared matrices. pack_nz in utils38
reorders bytes then reshapes BACK to original logical.shape; it does not replace
producer logical shape with blocked rank dimensions. Thus contiguous upload and
StackedDeviceTensor.full_shape retain the declared logical dimensions. Entry
spec/contract uses these values; full owner has57 statics, not18 count statics.

Pinned public device_tensor.py222..260 accepts built-in int leading index and only
full trailing slice(0,tail[axis]) or[:] (with supportedellipsis expansion). Generated
for rank__idx_v0 in range(...) produces Python int; every static uses that name and
_submit_chip uses that same rank. All57 metadata layouts are ND. Other dynamically
sliced nonstatic tensors are pre-fork shared Torch tensors, not static Stackeds.

Finite named-counterexample search covered leading Slice/non-rank index, partial
or stepped trailing slice, extra/missing dimensions, producer-schema versus metadata
shape disagreement, multiple/missing HOST static uses. Inventory retains each
name/producer shape/metadata shape/layout/dtype/HOST expression/line/full-extent
assessment. It is AST/JSON/source analysis, not runtime method execution or an
additional CPU acceptance suite. Existing schema derivation/source405 provenance
is reused; no full fixture/header/model-loop repetition.

Disposition: no known whole-shard HOST indexing blocker in full5262 under adapter398.
Do not recompile solely for this concern. Root separately owns count-old control,
fresh resources, exact fullprofile/source/prepared-image binding and actual full
capture GO. Runtime image/liveness/cache/oracle171/full F09/general batching or
physical-T support remain outside this finite source preflight. No future result
is inferred from current count-native/counterpart runs.
