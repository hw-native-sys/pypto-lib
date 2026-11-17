# F09 active-prefix operator draft — FINAL/PARK

Exclusive artifactcopy only: moe_exchange.py SHA256 `52be1ffae5c3a6956ce17745b5e7d15f3facc35fe9a48f12319ddd6914494bdc`;
patch SHA256 `ea2ef47fa26a9a7df6e6cb525248af147ca815977fb8c9171927bef47475ea47`. Live baseoperator SHA256
`951378f40c2ab7cb4f1573aeda72548bab017ed76c545641620e6e3502b4c780`. LiveCCsource45 was rehashed unchanged in queued test. No model,
sharedledger, prior evidence, dependency or runtime edit; no compile/NPU/payload.
Policy read output was initially redirected to a temporary /tmp file then moved
into this owned directory; no live source or other agent evidence was altered.

## Exact interface and behavior

moe_ffn adds num_tokens INT32 scalar afterepoch, before rowsconstexpr. rows is
receive/return capacity. Seven scratch buffers retain physical-T backing; only
views use active length. HCpre/norm/gate/EP/shared/combine/HCpost process the active
prefix and final outputs use active subviews. All16 ranks still call EP even when
localactive0, serving incoming peers. moe_ep16 itself is AST-identical (F10stride,
layout, signals, reductions unchanged); call now capacity=rows,rows=0. Host must
validate0<=num_tokens<=physicalcapacity; invalid scalar is not silently clamped.
Emptyphysical tensor is not a supported scenario; targetphysicalcapacity2 remains
allocated even atactive0. Compiler support for zero-length views and composed
runtimeactive HC/Cube paths is PENDING; no workaround or success is inferred.

Golden optional tensors['active_counts'] requiresint32/int64[ranks] within0..T;
absent meansallT. Dispatch loops independently use per-sourcecounts, destination
experts stillserveother sources, shared experts process active source prefixes,
combine writes only active rows and preserves supplied inactive routed values.
The golden owns no persistentcache/state; full idle carry/cache checks belong to
Main's caller/reference integration, not this operator output sentinel test.

## Exact tiny CPU evidence

task_20261107_234951_118999710010 COMPLETE0/max180/--no-device. Executed ACTUAL original and changed golden
functions extracted by AST in owned namespaces, D4/T2/16ranks, with an owned
explicit affine expert leafstub (not quantization/native math). Sourcecountvector
[1]*8+[0]*4+[2]*4 yields96 routed rows vs original192; all routes targetrank8,
which itself hasactive0, proving remote destination work is retained. Input IDs
independently count per-source repeats; affine expected active data matches, all
inactive sentinel rows unchanged. Allzero counts produce no routes/shared work.
Explicitall2 and defaultallT match originalbitwise under the same leafstub.
Five invalid count shape/dtype/range controls reject. AST verifies unchanged
moe_ep16, scalar ABI and seven physicalbacking allocations. No actualgolden data
is derived from device output; no fullfixtures/newquantref/globalTorchpatch.

Allactive kernel/native equivalence and quantizedexpert results were NOT run;
CPUallactive leaf-control equivalence and unchanged primitive source are only
bounded evidence. Signaturecompile/fullentrycountprobe/native19gates/idle carry
and sameartifact threecall acceptance remain pending Root's future assignment.
No other operator or standalone caller source was changed in this artifact.

Main can copy this PARKED file into its own proposed integration directory;
operator/ raw files remain frozen afterward. No background jobs/writers remain.
Measured worker interval1794066473->1794066732=259s; record imported nowhere yet.
