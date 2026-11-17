# ONE next F09 step: capacity2 active-S / two-request isolation

Planning only. CC/generate416 and all sealed artifacts remain immutable while
231342 runs. That current realcase failure takes priority. No product patch,
kernel implementation, compilation, NPU, realpayload, fixture materialization or
actual grant is included here. Scope is one complete DECODE artifact, fixed
physicalcapacity2/B2/pages2, not the whole serving feature set.

## Proven current gaps, with supported-case limits

| Source / exact owner | Observation | Consequence / future scope |
|---|---|---|
|decode_fwd.py:1102/1133/1160|rank reads token_counts and passes num_tokens through encoder/decoder attention|There is a runtime active prefix in the ABI; symbol presence alone does not prove its consumers|
|decode_fwd.py:537/824|both FFN calls pass constexpr rows=_MOE_ROWS, no active scalar|fullentry FFN does not receive active-S|
|moe_exchange.py:362/432;moe_ep16:100/145|moe_ffn calls moe_ep16(capacity=rows,rows=rows); tokens=rows and counts iterate every physicalrow|route population remains the compiledbound, including inactive tail; this is source proof, NOT a demonstrated current allactive numerical failure|
|decode_fwd.py:1598/1839|tokens2 fixture already uses TWO requests but token_counts always equals2|allactiveB2 numerical passes alone cannot accept varying/idle lifecycle|
|decode_fwd.py:1859/1925;layer_reference.py:336|reference dict omits token_counts; golden uses physicaltokens|current19 comparisons can mirror the static-row behavior; active-aware independent oracle binding must be fixed before calling it an F09 positive|
|generate.py:925/1068/1103|paged_plan builds exactlyone request; decode helpers requirecapacity1|intentional current single-stream scope, not a newly proven supported-input hostbug|
|generate.py:504/1680|ordered_dispatch_args checks exact resolvedshape; prepared owner keeps fixedsharedIO|physicalT2->T1 is rejected before dispatch; this plan does NOT certify physical dynamic-axis variation|
|utils.py:202/238|window/compressed helpers support explicit request tables; build_request_plan rebuilds/renumbers pages for positive querylengths|use explicit persistent table/state identities across idle calls; do not filter-and-renumber requests or treat length0 rejection as a bug|

Current realN3 prefill anddecode1 have active==physicalcapacity, so this finding
does not retract the accepted firstshort[3,1] or presume a failure in currentOnce.
F10 primitive dynamic_harness uses activeviews and rows=0 with capacity2, and its
unequal/idle proof is a useful basis. It is NOT fullentry acceptance. A5 source
prefill_fwd.py:828-876 is semantic guidance only: stable perrequest pagedstate,
zero-count ranks participate without publishing, completion before state advance;
its FP8/MX kernels/runtime handling are not imported into A3.

## Exact future ownership proposal (not dispatched)

- Main owns decode_fwd.py and prefill_fwd.py callers/fixture/integration and
  layer_reference.py active-count reference plumbing. Prefill caller needs the
  shared-signature propagation even though acceptance target is decode only.
- If a worker is useful AFTER Root implementation GO, one operator worker owns
  ONLY moe_exchange.py plus its own scoped FFN/count artifacts. It threads actual
  runtime count and bounds active work while preserving staticcapacity layout,
  source-gated peer participation and acceptedF10 stride. Main does not concurrently
  write that file; a concrete signature contract is recorded before dispatch.
- Main-owned NEW artifact case driver/metadata/durable result files, using existing
  decode build/spec/owner framework. No modelutils change is presently justified;
  no generate.py realprompt expansion or checkpointloader edits in this F09 step.
  Shared phase files need a specific source-backed transfer only if actually
  required; no broad advance ownership of every attention module.

Candidate implementation basis, not an assumed compiler fix: pass num_tokens to
moe_ffn, bound HC/gate/exchange/HCpost active views, use dynamic active length with
constexpr capacity rather than pinning both to physicalrows. All16 EP ranks still
serve remote experts even if their own sourcecount=0. ExistingF10 exact activeview
pattern may compose; prove it by a future cheap source-bound signature compile
before any device trial. Do not bypass or rewrite a compiler failure. Allactive
oldpaths require scoped regression. No implementation is performed now.

The current realgeneration admission intentionally pins44 math files against
its accepted sourceplan. A future F09 math change will therefore require a NEW
reviewed source/base/plan/admission binding before any realweights reuse. Do not
silently relax that equality check or relabel old611/3dd67 as the new program.
The bounded syntheticE1 path below needs no real-weight admission/grant.

## One artifact / one owner / three calls

Use explicit DEEPSEEK_PER_LAYER_EXPERTS=0 BEFORE imports, full40dense layers with
one reused bounded syntheticexpertlayer/noEngram, same TP4DP4EP16/W8A8 and fixed
capacity2. Staticweights generated once, no realcheckpoint reader and no1.49TB
reload. Future driver must assert E1 and refuse E40/checkpointweight loading.

Fixed index table (logicalpage->physical): A[3,1], B[0,2]; stateblocks A->1,B->0.
Four physicalpages,2stateblocks,2logicalpages/request. Stable across calls.
Seed reproducible distinct cache/history throughposition125 and distinct A/B
compressor states; record independent initialbytes. Queryrows are prefixactive.

|Call|active counts perDPgroup (repeated over4TP peers)|querypositions pergroup|Purpose|
|---|---|---|---|
|0|2,2,2,2|A126/B126 allgroups|two live requests, even C2 partial carry|
|1|1,1,0,2|A127/B127 tailmetadata; first2groups onlyA active, group2 bothidle, group3 bothactive|active-S1/0 and idleB carry/history isolation, unequalEP sourcecounts|
|2|2,2,2,2|groups0/1 A128/B127; group2 A127/B127; group3 A128/B128|resume from actualreturned state, page127->128 and delayedB C2 completion|

All three physical tensor shapes stay T2/B2/pagecapacity4. ActiveScalar2->1/0->2
varies in ONE compiled artifact. This is bounded ACTIVE-S and request-isolation
proof, NOT physical tensor dim2<->1, variablephysicalB, allbatchlimits or decision9
closure. Same artifact/hash/owner/statichandles/config must be recorded percall;
no per-count recompilation, owner reset or reference-clone substitution as nextstate.
The actual poststate fromcall0 feeds1 and actualpost1 feeds2.

Deliberately keep inactive-tail metadata as valid-looking stale B positions/slots
and vary its input data. Runtime active prefix must gate publication/routing;
zeroing tail slots alone is not enough evidence. Addresses remain valid to avoid
an out-of-range crash masquerading as a numerical control. Egcall1 group0 tailB
haswindow127 and completedC2row63 but count1: B pages/state must remain unchanged.
Call2 group0 A128 mapswindow128, B127 mapswindow127, C2 B63 maps63; atcall1 A127
C2row63 maps447. Integer formulas and actualutilities agree in scenario.json.

## Reference and meaningful controls

Keep existing threshold values. Reuse current golden owners and exact-S32
reference context as appropriate; no new quantization/wholemodelGolden framework.
However thread independent perrank activecounts through the references; do NOT
reuse the present physicalrowloop unchanged or derive expected masks from output.
Preserve pristine reference initialclones, source contracts and both original
unconditioned/staged scopes where used; compare only semantically valid outputrows,
plus explicit physical persistent-state ownership. Idle ranks still host experts.

1. Positive actual40layer outputs on active rows at existing tolerances; actual
   published row gates and all previously valid rows retained. For idle requestB
   compare ALL owned cache pages/validrows and its3 C2 stateblocks byte-exact before
   vs actualafter, excluding only legitimate activeA writes. Retention must grow;
   a wholecache1% allowance cannot replace these checks.
2. Independent source-route counts: expected per source=active_count*TOPK.
   With TOPK6, call1 expected source vector[6]*8+[0]*4+[12]*4, total96; old
   fixedphysical2 FFN yields[12]*16,total192. A companion numerical count probe
   of the EXACT shared FFN/exchange owner plus generated fullentry scalar-to-
   activeview reachability is required, since current19 exports do not expose
   recv_meta. That probe alone does not accept fullentry; combine it with actual
   three-call fullentry outputs/state and binding review. Do not call mere source
   spelling or a standaloneF10 pass fullentry evidence.
3. Old physical-row-routing control in a separate frozen artifact: pin FFN to2
   instead of active scalar. It must numerically reject routepopulation96vs192;
   any fullentry state/output consequences are separately recorded. Runtime
   failure/timeout is NOT this negative's PASS and active rows might coincidentally
   match because experts are tokenwise: route-count evidence is essential.
4. Valid-address mapping controls: alias B's page row to A, or map B's stateblock
   toA. Independent correct-table reference/idle-history checks must reject the
   resulting numerical ownership violation, not just detect an exception. Poison
   recently retained B row/carry in copiedactualbytes and confirm focused checker
   rejects below the global1% allowance. This validates the gate, not the model.

## Minimal future execution route (requires separate RootGO)

Current check_small_plan.py is an executable tinyCPU scenario/shape/sourceproof
only. No native acceptance helper is falsely claimed to exist yet.
After future boundedimplementation/sourcefreeze, add ONE artifact case adapter
f09_decode_case.py over decode_fwd.build_fwd_specs and existing PreparedDispatch/
publicrunconfig, not a serving framework. Parameterize the existing fixture's
request geometry before allocation (avoid default1250/1253's20pages); reuse E1
seed weights and metadata/functions. Capture actualnonstatic initial/output after
EACH of3synchronous calls before references, then free/normalclose; separate CPU
validation of those exact bytes. Staticseed/bytehash provenance, no saved103GB
static tensor stack. No inline longCPUreference while holding16cards.

Future command contract (path refers to that UNIMPLEMENTED nextscope adapter):

```bash
task-submit --no-device --max-time 300 --run '... python NEW_F09/f09_decode_case.py --stage metadata-and-signature --scenario THIS_UNIT/scenario.json --expert-layers 1'
task-submit --timeout 2400 --device auto --device-num 16 --max-time 1800 --run '... DEEPSEEK_PER_LAYER_EXPERTS=0 python NEW_F09/f09_decode_case.py --stage capture --device "$TASK_DEVICE" --execution-mode ONBOARD --scenario THIS_UNIT/scenario.json --artifact ONE_FROZEN_CAPACITY2_ARTIFACT'
task-submit --no-device --max-time 7200 --run '... python NEW_F09/f09_decode_case.py --stage validate --actual-captures NEW_F09/actual --scenario THIS_UNIT/scenario.json'
```

Those are precise future stage/CLI requirements, NOT runnable existingfiles or
current grant. Existing default decode_fwd.py --tokens2 --engram-disabled is an
allactive2/tworequest baseline only; its inlineCPUreference, default1250geometry,
uncountedactive FFN and publicheap config are not silently accepted as this
three-call route. Future adapter uses explicit normalizedONBOARD/a2a3/public
ring[0,0,4GiB,0], sameconfig everydispatch, generateda2a3/native logs, actual16IDs,
source/artifact/schema/mode pins, bytebudget/fresh HOST/DISK/SHM/publicHBM before
fixture/owner, firsterror/cleanup/sampler/PID-lock evidence and no blindretry.
E1 still has a large bounded hostfixture; derive stage bytes first and preserve
host/IO/clone/serialization margin rather than assuming 'synthetic' means tiny.
This unit allocated only tiny plan tensors and did not run any of these stages.

PhysicalT2<->1/B2<->1 on ONE artifact, unequalEOS/request-slot reuse, arbitrary
page maps/longerhistory/retainedslots, fullE40 numericalparity, Engram/quantprimitive
and engineeringcloseout remain subsequent gates. No generalF09 closure follows.
