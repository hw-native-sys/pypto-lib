## Current configurable A3 TP/EP milestone — CPU/signatures; native pending (2026-11-08)

New user scope makes TP/EP configurable while retaining defaultTP4/EP16. Current
source45 `f0e91a30e4117e110be183ef44acf2874eea813cc6f72ac7c7fa44bc4ede3a7f`
uses import-time `--tp N`/`--tp=N`, `--ep N`/`--ep=N`; TP1/2/4/8, EP2/4/8/16,
EP>=TP and divisible give13 configurations. WORLD=EP, DP=EP//TP; invalid flags
reject early. [Implementation/evidence](logs/configurable_parallel_1794108101/REPORT.md).
Configuration/CPU coverage is distinct from actual full-model device coverage.
A5 syntax is a reference, not a claim of A5 fullentry TP>1 support.

SmallEP compute slabs now ceil-pad to16, COUNT_PAD ceil-pads localexpert count
to32, with actual network lane stride preserved. All MoE callers use topology-
neutral names and the shared typed completion/zero-active peer-service contract.
Goldens preserve destination-rank weights; loader retains W8A8/NZ/offset/FP32
scale math and whole-expert EP sharding. Indexer32heads and sharedFFN remain
replicated; only actual TP-owned attention/head/vocab resources shrink. Local
HEADS8 attention uses explicit16-row physical padding and stores only8 live heads.
No A5 quantization, model threshold, runtime or Engram/MTP restoration change.

Worker config/loader CPU checks cover13 configurations and small nondefault
signatures. Main directed120237 CPU admission/ownership controls pass; preserved
v1 fixture collision and two own attention compile failures are detailed in the
report. Fresh normal E40 TP8EP16 pair121454 COMPLETE0: prefillN3 100formals,
decode1 96formals,57statics each,45.976/43.460s, zero tensor samples/prepare/NPU.
No real checkpoint header/payload was read by this implementation unit.

New metadata `a7b187a084e304b5203aeb2b0897dcb291550fbcf6c5bbaea6d1fe42f49a2a46`
for TP8EP16DP2 derives38,328,513,216B/rank/global613,256,211,456B; largest static
11,324,620,800B unchanged.164192source reads745846365184B/110368leaves remain
unchanged because sources are read before slicing. Old e13/sourceB resource
bindings are not relabeled. EP8 full routedweights alone68,022,435,840B/rank are
unfit; smallerEP component/layer validation remains a separate supported scope.

The configurable actual Root schema binds exact ALL45 current sources/topology/
newplan/resourcebudget and one-upload accounting; defaultHARD and legacy guards
are preserved. Planningfloors remain conservative, not actual peak measurements.
Root-owned independent CPU/resource/PREQUEUE and smallEP/TP/HEADS8 native proofs
are pending, followed by a separate realTP8EP16 allocation/payload/generation GO.
No actual configurable text or native acceptance is claimed here.

Historical SourceA171 synthetic fixedT2B2 acceptance and SourceB real one-upload/
15greedy-call bounded facts remain valid only at their original sources. Whole
SourceB FAIL218/PSSnull and oldM02FAIL183 stay unchanged; text quality/original
numerical+quant/history parity/fullpeak/general serving/Engram stay OPEN. Main
freezes this coherent source milestone and continues only the separately owned
closed operation-wrapper preparation; older current-status overlays below are
historical and superseded for live source identity.

