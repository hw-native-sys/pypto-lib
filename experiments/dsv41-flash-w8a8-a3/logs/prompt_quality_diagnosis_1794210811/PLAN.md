# TP4 multi-prompt quality diagnosis — bounded plan

Primary ONLY TP4/DP4/EP16, currentf0 A3 W8A8/noEngram/noMTP. Default source and
all old sealed results remain immutable baselines. TP8 is a later controlled
comparison ONLY if a concrete TP4 anomaly needs it, not a second firstload.

## Fixed representative cases before actual execution

1. Plain: What is the capital of France? Answer in one word. Expected Paris;
   assess plain continuation separately from instruction following.
2. Correct checkpoint nonthinking chat: SAME text. Expected Paris.
3. Correct checkpoint chat: What is 2 + 3? Reply with only the number. Expected5.
4. Correct checkpoint chat: 中国的首都是哪里？只回答城市名。 Expected北京.
5. Plain: The opposite of hot is. Expected a relevant continuation such as cold.
6. Plain: 一年有十二个. Expected 月/月份 or a semantically correct continuation.

Maximum32 newly generated tokens percase, naturalEOS, at least one case allowed
full32token continuation for coherence/repetition observation; no forcedEOS,
no teacherforcing. Original exact IDs/rendered bytes/N and semantic expectations
will be recorded by queued tokenizer CPU BEFORE any real generation. No hidden
padding/truncation or meaningless N3 selection. Plain intentionally has noBOS;
chat uses checkpoint shippedencoding BOS+User+content+Assistant+</think>, no
trailingEOS. Template pairing separates formatting from numerical hypotheses.

## Minimal one-upload driver design and ownership

Main exclusive product writable generate.py ONLY initially. All math/44other
source, tokenizer.py, loader, config and kernels stay unchanged. Main owns finite
suite CLI/admission/resource/metadata/signature wiring and existing owner adapter.
Worker prompt_cases owns ONLY new artifact worker_cases/**, explicit sixcases,
canonical tokenizer/renderer provenance and CPUtokenization/entryshape budget
ledger; no product/sharedledger writes. No worker source overlap.

Use existing normal entry/kernel/WeightLoader/PreparedDispatch/SharedWeightOwner,
not duplicated model compute or serving framework. One canonical57resources/
912bases expectedTP4 static39,624,816,576B/rank/global633,997,065,216, independently
recomputed CURRENT from contracts/source descriptors; one source schedule only.
Decode0staticIO/borrow112C1+800same aliases/freebasesexactlyonce.

Preferred finite route: ONE prefill artifact with protocol capacity rounded-up
to16(max of actual encodedNs)<=128, actualHOST input/output token axes remain
exact N percase and token_counts=N; capacity is only protocol/scratch bound,
not extra prompt tokens. moe_ffn already passes capacity bound with rows=0,
so actual active view drives expert routes. Prove public dynamic-shape binding/
source before using this route. If it is not supported, register finite exact-N
prefill artifacts BEFORE one prepare instead; never newweightupload perprompt.
One decode1 artifact is common. No runtime/privatepointer/hotregistration.

All per-case prefillHOST IO maps and the common decodeHOST IO are shared BEFORE
worker prepare/fork. Each case starts from independently rebuilt fresh prompt/
window/compressed/index/compressor/selection/HC state copied INTO thoseexisting
buffers; no previous case cache enters next reference/dispatch. Do not reset
protocol globals byhand: persistentFalse gives fresh public commdomains percall.
Initial full-byte mutable-state hashes/storage IDs and after-case diagnostics
prove reset and alias lifetime; poison prior-case cache tests must reject carry.
Sameowner program selection/canonical handles persist; allcases finish/raise
through firsterror-preserving attempt-all cleanup.

## Budget, binding and validation sequence

New finite prompt-suite Root schema binds exactcurrent45/source/TP4metadata/
allcases renderedbytes+IDs/N/templates/caps/order, compiledcapacity+actualNs,
static/readcoverage, runtime/config/tokenizer and positive resource budget.
Old N3/cap16/plain singleprompt legacy admission remains intact/defaultHARD.
Newbudget includes all pre-fork sharedHOST IO and actually supported deviceIO
slot/lifetime bounds across cumulativecalls, not oldN3/TP8budget or assumed
GMbank0=>HOSTslot0 reuse. Maintain source5362417664 planningallowance and actual
publicmemory/hostdiskSHM queries/oneupload coverage guards; no doublecharge
committed/ring. If public retainedIO lifetime is unbounded, report exactblocker
before realGO, do not invent a fit/relax safeguard.

Next: tokenizer/metadata-only CPU, minimalhost source adaptation, meaningful
small publicowner/reset/cap/guard tests and fresh zero-sample TP4 signatures,
all through task-submit --no-device. No hugefixtures/realweightheaders/payload/
prepare/NPU here. Independent source/resource/template review before Root
authorizes one resident multi-prompt auto16 ONBOARD/a2a3 operation. Need actual
percase text/IDs/EOS/calls/DP4votes and predeclared semantic verdicts; anomalies
get the cheapest concrete counterexample/format vs state vs TP diagnosis.
No fulloriginalparity/quality claim from prior one completion or Engram guess.
