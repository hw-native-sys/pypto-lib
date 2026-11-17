# C2 active ownership and failure checkpoint repair — FINAL / PARK

NEW artifact-only candidate, NOT live application. Main exclusive/no worker (adapter
was already implemented before optional delegation). Relative to PARENT405 copied
model45, ONLY compressor.py changes two statewriter bounds: range(tokens)→range(num_tokens)
and token+1==tokens→token+1==num_tokens. Other44 exact405; live M02 separately unchanged.
Physical projection/latent/state capacities, pool dependency, projection read anchors,
all original goldens and full171 numerical comparator expressions/tolerances unchanged.

Proven SOURCE bug: pair pooling already uses active num_tokens, but the old state
flush used physical xdim2. Middle rank0 active1 with requests0/1/positions127 writes
inactive requestB/stateblock0/parity1. Active0 likewise writes physical rows. Source
last-of-request also failed to flush last ACTIVE row when inactive tail shared its
request. Existing layer golden slices :active before actual compressor golden,
so it already owns active-only behavior. Both statewrite bounds must change.
No new scheduler/dependency edge is invented; dependency-liveness skill/source guides
used to preserve valid physical anchors and explicit pool→writer ordering. Runtime
graph/liveness is NOT proved by source/compile. No deferred wait/dummy scheduling fix.

Actual failed065800 remains FAILED after TWO dispatches: call0 initial+actual+record,
call1initial, postcall1 LOST because old retention ran before save. Ownernormaltrue/
primaryretention/cleanup[], outerrecords0 because drive never returned. Root sealed
native8a37de20/runtimefdec47b2; observed31births/raw16release/16imagefree facts remain.
Independent saved-snapshot audit957ad51f/070731 found all floating saved tensors
FINITE, sixcachecarry actual0→initial1 byteequal, preidleB finite. Preexisting NaN
falsefailure ruled out; exact actualpost1changedbytes remain UNKNOWN. This source
repair is grounded independently, not invented reconstruction of missing bytes.

Capture fix: drive saves actual poststate then writes pending descriptor+metadata
hashes/history-before/owner/artifact identity BEFORE retention. A rejected check keeps
those bytes and FAIL/primary/trace record. capture close receipt keeps outerreturned
records plus separate durable_post_dispatch_records and retention_validated_call_records,
so partial0/1 evidence is no longer erased by outer records0. Successful3record path
remains compatible; CPU admission additionally requires three PASS post descriptors.
No fail-positive numeric gate is added. All actual bytes are stored without static
payloads using original self-contained clone serializer.

Ownership comparator now requires same shape/dtype and exact contiguous BYTEviews.
This correctly treats identical NaN payloads as unchanged, distinguishes +0/-0 and
handles noncontiguous inputs. It strengthens byteownership; original full171 numeric
comparators/finite policy are untouched. It does not treat NaN numeric results as PASS.

ONE --no-device batch071536_1890249939 COMPLETE0:
- Actual statewriter AST body vs actual UNCHANGED original compressor golden active0,
  1,2, nonidentity A1/B0 state map/opposite parity and same-request inactive-tail case.
  Old physical bounds fail active0/1 cases; new states match expected exactly. Tiny
  data/projection dimensions test ownership/control, not full native arithmetic.
- NaN-payload/signedzero/noncontiguous/shape/dtype byte controls pass.
- Actual drive with CPU fake owner injected middle idleB corruption: call1.actual +
  FAILmetadata present, hashverified poison saved, two durable/onevalidated records
  despite outer0; call2 never begins. This is a CPU capture test, not native evidence.
- FRESH full zero-sample signature PASS45.014s, new source45
  8153c7f040bef51900a6e6c9ff27b018279b7503f31d50ec5488457e49770b28.
  Public full ABI metadata remains EXACT8ba379ca…e78770; new filemanifest510dd3c7…f8642.
  No TensorSpec.create_tensor/fullfixture/fullreference/prepare/NPU/weights/grant.

ALL registered statewriter variants inspected in this full source profile: ONE
c2a_state_write func59/callTask70; no C2B/suffixed variant registered. Generated call
now supplies num_tokens rather than T_DYN and keeps pool dependency plus kv/score
inputs. Physical positive-capacity projection anchors remain. Other profiles are
not claimed. Compiler unused-token/anchor/ScalarWriteLineShared warnings preserved;
no out-of-scope cleanup or numerical acceptance is inferred.

Fresh FULL required because shared compressor compute changed. Count probes import
moe_ffn/HC/gate/experts/GEMM, not compressor; their sources/dependencies remain byte
identical. Old count proofs/profiles retain OLD405 ancestry in count_ancestry.json;
no duplicate count compile and NO relabel as8153. New route permits only full-capture/
full-offline, binds only fresh full profile. New adapter SOURCE_PIN/route/new45 bind
this candidate; helperEE and oraclee45 unchanged. Source-file golden hashes changed
only via kernel-containing file; golden AST itself unchanged, as checked.

Futurecase/launcher/pins are new and default REFUSE with PENDING EXTERNAL NEW Root
lifecycle/source wrapper path+SHA. Olde76/source405 cannot authorize8153. Root may
compose prior actual helperEE public119registry/116unique evidence as ancestry only,
then require separate new full ParentGO/freshPREQUEUE/work. Budget includes scalar4:
reserve10157899716/prestatic16621778820, unchanged95tensorABI/57static+38IO, sixsnapshot
2965339800B, HOST300GiB/DISK16GiB/SHM32GiB; no measuredfit claim. Work absent now.

Next: independent SOURCE review result first, Root NEW wrapper, then reviewer ONE
fresh PREQUEUE. No repeated controls/compile/micro/NPU by Main. Actual repaired
full3call/retention/171/math/liveness gates remain OPEN, as do generalF09/physical-T/
originalparity. Model8153 is UNAPPLIED/liveM02 frozen. Main FINAL/PARK/no writers/jobs.
