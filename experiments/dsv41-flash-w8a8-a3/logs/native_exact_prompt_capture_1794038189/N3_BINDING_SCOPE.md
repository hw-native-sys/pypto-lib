# N3 binding and shared ABI scope

The copied distributed ABI SHA9734f8235dd25d522bb24300a6b801d8d412eaf7202a10973b322fa258262620
is identical to the accepted older N128 parameter-layout ABI. This is expected:
its token dimensions are `-1`. It binds parameter inventory, directions,
dtypes, layouts, and fixed extents; its hash alone does not prove token count3.

The concrete N3 chain uses independently recorded existing evidence:

1. Frozen reviewer case_n3.py invokes the NEW producer's annotation factory with
   `build_entry_contracts(3,pages=2)` before any fixture. It then calls the
   unchanged original fixture builder `build_fwd_specs(3,seed=0)` and cached
   `_prefill_entry(3)`, whose cache key is the requested token bound.
2. The original `_entry_specs` checks complete declared/supplied parameter-set
   equality, emits actual tensor shapes, and feeds NEW generate.spec_contract,
   compile_entry, PreparedDispatch, and dispatch. `prefill3_memory.json` records
   input_ids384 B =16x3x8 and total nonstatic363,060,736 B. Scalar engram_enabled
   remains a public typed zero-dimensional int32 argument.
3. The actual generated `orchestration/host_orch.py` derives `T_DYN` from
   `tensors["input_ids__ssa_v0"].shape[1]` and supplies per-rank `0:T_DYN` views
   to the chip program. Its source hash is recorded in native_execution_proof.
   The artifact was freshly compiled for this case, not relabeled old output.
4. The immutable capsule contains43 actual and43 independently cloned pristine
   initial nonstatic tensors and57 static descriptions, completing100 argument
   headers. Both input_ids have shape[16,3], original IDs[100,101,102], all16
   token_counts3 and positions[0,1,2]. Offline reconstruction checks every
   header against the concrete N3 source contract, plus all payload/static
   byte hashes. The generic ABI must match every direction/layout and be able
   to represent each concrete shape; no `-1` is treated as literal3.
5. Native logs contain one dispatch with sixteen unique worker node.complete
   terminals, outcome0, plus copied CPU output bytes and normal owner close.
   The manifest binds this artifact/owner/source/runtime and copied ABI. The
   effective normalized owner configuration is ONBOARD/ring[0,0,4294967296,0],
   the generated target is a2a3, and installed TMR shared-bank0 library SHA is
   pinned24434f8879030ef1b1f31e5fc2ecce683e392d4f1baeb5bcee852fb183976e5c.

The native-only chain proves execution and saved case identity. Owning original
prefix17 CPU gates are a separate result. It does not prove original checkpoint
parity, forty distinct expert layers, real-weight fit, generation or text.
