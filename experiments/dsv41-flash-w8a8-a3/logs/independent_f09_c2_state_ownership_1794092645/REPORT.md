# Independent C2 state ownership review — SOURCE BUG, FINAL / PARK

Frozen405 source contains a concrete active-prefix state-write violation matching
the reported failure class. This is a lib-side bound error, not evidence for a
compiler/runtime fault. No CPU test/task, tensor/payload read, fixture/golden,
compile, prepare, NPU, model/helper/runtime change or input copy was performed.
Exact actual post-call1 values remain UNKNOWN because the failed capture did not
save them. The native065800_182518612591 failure is preserved, never a PASS.

## Exact source and lowering

model/compressor.py c2a_compressor receives num_tokens(active). At95..103 it
allocates projection/latent buffers with physical xdim and defines tokens=xdim.
Pair pooling108 correctly loops to num_tokens. The persistent state writer146
instead loops to physical tokens; its last-of-request boundary149 also compares
token+1 to physical tokens. It writes state block*STATE_CAPACITY+position%capacity
at159..160. These are the only persistent state writes in this function.

The active scalar reaches this function correctly: decode_fwd.py1104 reads
num_tokens=token_counts[rank], encoder C2 source calls pass it at489, and
model/decode_c2a_full.py135..138 passes it into c2a_compressor. The bug is inside
compressor state flush, not rank metadata or the call boundary.

Actual unchanged5262 generated decode_fwd_rank.cpp2727ff Task70 supplies T_DYN to
c2a_state_write, whereas preceding Task69 pool supplies num_tokens. Generated
kernels/aiv/c2a_state_write.cpp133ff loops against that physical parameter. Source
and lowering agree. No unexpected compiler rewrite is needed to explain the
source violation. Task70 retains explicit pool_tid dependency; projection read
anchors at144..145 order projection writes even when an even-position pool body
is empty. A correction must preserve those anchors and dependency.

## Intended ownership and finite scenario

Pinned config: DP4/TP4/EP16, STATE_CAPACITY2; no W8A8 math/weight/sharding change is
justified. Requests A0/B1 map to state blocks A1/B0 independently of page maps
A[3,1]/B[0,2]. Odd127 consumes its OWN request's previous even126 projection at
parity0, then stores current projection at parity1. Even126/128 stores parity0.
State is request/block indexed; there is no state alias in the two supplied maps.

At middle call all supplied positions are127:

| DP/ranks | active | Intended state writes | Frozen physicalT2 writer |
| --- | --- | --- | --- |
| 0/rank0..3 | 1 | A1 parity1 only | A1 parity1 AND inactive B0 parity1 |
| 1/rank4..7 | 1 | A1 parity1 only | A1 parity1 AND inactive B0 parity1 |
| 2/rank8..11 | 0 | None; all A/B state preserved | Both A1/B0 parity1 |
| 3/rank12..15 | 2 | A1/B0 parity1 | Same intended two writes |

Thus rank0/request1 must retain BOTH B0 state rows; the source still writes its
row1. First call has active2 and position126, so both row0 writes are legal and
passing its retention gate is consistent. Next position128/127 resumed-call
behavior was not executed after the failure and is not claimed here.

Both write loop AND last-of-request boundary must be active-bound. Merely stopping
loop at active can fail to flush the final active row when the next physical
inactive row has the same request ID. For active0 the bounded loop must issue no
state write. For active1 odd127 its own A1 parity0 input is read for pooling and
only A1 parity1 updated. For active2 both own request rows update with current
parity; the opposite row remains protected. No tolerance change or idle-write
exception is appropriate.

Other physical-tensor usages in c2a_compressor95..164 allocate/project/normalize
scratch at physical capacity; they do not directly write persistent state. Pool
workers are max(1,min(active,POOL_WORKERS)), but its loop is active-bounded, so
active0 has an empty pool. Final RMS covers physical latent rows (undefined unused
scratch may exist); this source-only review does not turn that into a separate
state ownership claim or full inactive-output acceptance.

## Golden and retention predicate

layer_reference.py365..379 slices row inputs to each rank's active prefix;
make_layer_golden413..417 skips active0 and uses x_hc[:active]. It passes active
attn_input and matching positions/requests to _compressor_rows441..445, which
clones state and calls original golden_c2a_compressor. That golden's x.shape[0]
is therefore ACTIVE here. Its last-row loop261..267 writes only the final active
row of each request. It does not share the kernel's physical-bound bug on this
composed active-prefix path. No new reference arithmetic is needed for the fix.

retained_equal in adapter398 uses block=1-request and written={position%2} only
when request<active; otherwise keeps BOTH rows. Under this exact one-row/request
B2 scenario that ownership predicate is correct. It preserves active unwritten
parity plus entire idle block; it is not overly strict. No need to exempt request1
or rewrite request maps to make the failed case pass.

Comparison caveat: torch.equal is exact element equality, not byte equality.
NaN is non-reflexive, and +0/-0 compare equal. The missing post-call1 snapshot and
this review's no-payload scope mean it cannot determine the observed mismatch's
finite values, NaN locations or exact byte deltas. A separately owned saved-data
finite/carry audit may narrow that; it cannot recover unsaved post-call1 bytes.
The source active-prefix violation is proven independently of this uncertainty.

## Capture provenance and next evidence

Original drive saves initial, dispatches, then calls retained_equal BEFORE saving
actual. This exception prevented post-call1 capture. Saving actual+source/metadata/
owner/artifact identity before evaluating retention is necessary to preserve future
failures; it must not convert a failing retention result into acceptance. Main owns
that new artifact fix; no old evidence is changed here.

Minimal new source remedy is active-bound flush loop and active-bound terminal-row
check, preserving projection dtype/rounding/pooling/block/parity/dependencies. Fresh
source manifest/signature/prepared-image lineage is required for any later hardware
trial, not relabeling old405/5262. CPU active0/1/2/end-of-active-prefix and exact
retain controls, then separately authorized native evidence, remain future work.
Original full parity/general F09/realE40/fullhistory claims are not established by
this source diagnosis. Actual full119 registrations/116unique+3legal aliases are
an unrelated accepted ownership inventory, not C2 retention proof.
