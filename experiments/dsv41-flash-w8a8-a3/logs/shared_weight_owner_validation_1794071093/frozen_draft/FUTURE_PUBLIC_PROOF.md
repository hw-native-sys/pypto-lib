# Separate future public two-program proof — no execution grant

After the active case's terminal/audit/seals and separate Root GO, use TWO devices
for an API-only toy, NOT a TP4/DP4/EP16 model accuracy claim. No634GB realweights
or fullfixture is needed. Current draft unit does not compile or execute this.

1. Define two small normal @pl.jit.host programs registered together. Canonical
   BF16 staticW and C1W are [2,128,128], allocated once via one prepared worker;
   decode's C1 parameter is [2,1,128,128], with public perrank view_tensor inserts
   atbyte_offset0. Two statics ×two ranks ×128×128×2B =131072 staticH2D bytes once.
2. Compile BOTH beforeprepare. Allocate prompt/token/state/logits sharedhostIO
   for both programs BEFORE prepare/fork. Public
   prefill.prepare(config,extra_compiled=[decode],persistent=False), then
   worker.run(prefill,...,config=same_normalized_ONBOARD_config) and
   worker.run(decode,...,config=sameconfig). No register hotload or ambiguous
   worker(*args) in multi-program mode. No cross-owner handle/privateptr.
3. Constant exact arithmetic: prompt1, W2, C1W3 ->prefillstate6. Copy the actual
   returnedstate into existing earlydecodeIO. Decode input1 ->12; nextinput2 ->19;
   nextinput3 ->27 if each call stores state+input+W+C1. All BF16-exact integer
   values. Use public readback/output buffers to verify bothweights and actual
   returned-state continuity, while recording allocation/source/copy/run counters.
4. Assert oneprepare, fourbasehandles, one131072B staticupload, zero decode
   staticH2D/rereads, C1sameBuffer/pointer/nbytes/dtype, sameworker/program selectors,
   no perstep upload, basefrees exactlyonce afterbothphases and borrowedaliases
   NEVERfree. Error-path CPU injection proves primaryerror/attemptall behavior;
   actual native terminal/close/PID+birth/locks need their own evidence.
5. Meaningful numerical control: pass a legal sameowner C1 view of W2 instead of
   C1W3; decodeactual11 differs from independent expected12. Addresses/shapes/dtypes
   remain valid, so runtime exception is not the negative PASS. Also require
   unregisteredprogram/foreignowner/freedhandle/aliasfree reject before unsafe
   dispatch as separate API contract controls, not numerical accuracy evidence.

Use public config explicitONBOARD/a2a3/ring[0,0,4294967296,0]/persistentFalse,
source+artifact+actualdevice IDs and freshHOST/DISK/SHM/HBM/positive safety floors;
no automaticheapresize/retry. An explicit small Root-owned helper/command will
be needed for this proof; it is NOT claimed implemented by this planningfile.
Source mode/library/view/runtime provenance is pinned in api_sources.json and
independent prefill_decode_weight_reuse_review_1794067346/REPORT.md.

After that boundednativeproof, source application and any fullmodel shared-owner
trial require their own Root decisions. Real16 canonical initialization remains
633997065216B/912basehandles/164192logicalreads/745846365184sourceB; decode is a
borrowed consumer withzero secondstaticH2D/rereads,112 newC1 view objects and912
consumer bindings. Other800same-shape bindings can reuse existing shard objects.
No need to pretend two complete-phase upload counters. Bothphase mutablehostIO
still copies percall; this proposal reuses STATICWEIGHTS, not device-residentKV.

Two registered programs can add code/inputslot/runtime memory beyond the old
singleprogram prewarm. Sameworker does not mean zero extra footprint or oneHOST
slot. Currentmodelplanningfloors remain unchanged and may conservatively STOP;
collect actual prewarm/dispatch/free snapshots rather than doublecharge ring or
assume oldfit proves newpeak. Pinned TMRsharedbank is backend-specific, not a
private168capacity multiplier. Fullmodel peak/performance/coherence remainunproven.
The separate HC BF16-boundary finding and any repetitive-text cause are outside
this transport/API proof and must not be hidden by a speedup.
