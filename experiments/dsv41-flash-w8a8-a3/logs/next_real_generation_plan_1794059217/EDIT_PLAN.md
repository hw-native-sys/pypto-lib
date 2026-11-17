# Minimal finite longer-generation proposal — DO NOT APPLY DURING204310

Precondition: the CURRENT D77/204310 max4 case must first terminate normally and
pass its independent post-audit, including external tokenizer POST and actual
state/history/strictDP/counter/cleanup evidence. Its realprefill1.85s/token3`!`
is a reported milestone only. This artifact does not assume decode or final
acceptance. Any currentcase failure takes priority over this proposal.

## Precise proposed generate.py edits (one normal path, no new kernel)

Keep default HARD refusal, same existing CLI flag/loader/PreparedDispatch/compiled
entries/cleanup/liveguard/cadence. Future edits are limited to
`FirstRealGenerationAdmission.__init__` and the post-tokenizer binding in
`device_run`; none are applied here.

| Current D77 admission | Proposed finite contract |
|---|---|
|operation literal `first-full-real40-noEngram-greedy-N3`|NEW distinct operation `bounded-full-real40-noEngram-greedy`; old receipts do not admit it|
|args.prompt exactly `Hello, world`|args.prompt exactly NEW receipt `prompt_text`, a nonempty string; still direct `--template plain`, no message/file/case/template bypass|
|max_new_tokens in(2,4)|integer cap2..16 inclusive, exactly equals NEW receipt field; keep max1 outside this unit to avoid the existing unnecessary decode-owner preparation at max1|
|post-tokenizer IDs exactly `[19923,14,2058]`|receipt `prompt_ids` must be list of2..128 true integers in[0,129280); actual post-tokenizer `prompt['ids']` must equal the list, without append/padding/remapping|
|exactN3 literal in error/documentation|explicit finite plainN2..128, max16, startposition0, single request clonedDP4; errors report unsupported0/1/>128 IDs or cap outside2..16|

Concrete edit order:

1. Parse/validate `self.prompt_ids=r.get('prompt_ids')` and
   `self.prompt_text=r.get('prompt_text')` inside admission, before any prepare or
   source payload. Require trueinteger IDs (bool is not integer here),2..128,
   0<=ID<129280, strtextnonempty; require args.prompt exactly self.prompt_text.
2. Replace only the operation literal and fixed text/maxcap predicates above;
   keep new source45, exact45 pathset, unchanged44, actualordered16, root/index/
   header/map/TMR/config/entry TTL<=3600/start0/Engramdisabled/strictDP/repeat1/
   sixhour bounded policy and other existing restrictions.
3. Replace `device_run` literal ID comparison with
   `prompt['ids'] != admission.prompt_ids`; existing exact_prompt_tokens enforces
   N2..128 and actualN drives existing factory/lastrowN-1/position/handoff.
4. Keep all generation math/history/EOS/strictDP unchanged. No second path,
   callback injection, new kernel, larger heap, altered golden/threshold,
   hiddenprompt tokens, source-key expansion, new global weights or owner reuse
   across prompts. Main-owned draft/application would require a separate RootGO.

A NEW applied source45 must be built after any future authorized edit; D77 is the
current producer, not the producer of that edit. Only generate changes; all44
math/loader files must match the reviewed predecessor. New actual receipt binds
that source45, prompt_text/IDs, maxcount, same planned checkpoint/root and exact
external tokenizer hashes/code/results (prequeue/allocatedprerun/postrun). The
current metadata-plan3dd67 can be reused ONLY as the static weight/source plan:
its stored Hello-world prompt remains honest historical provenance, not the new
prompt binding. Never rewrite it or label it as a fresh case.

## Smallest first longer case

Plain `Once upon a` is already actual-tokenizer verified as `[16600,4465,260]`,
exactN3, no BOS/chat wrapper/appending/padding. Max16 total generated greedy IDs,
not16 decode calls: first comes from prefillrow2, then at most15 decode calls
input each actual preceding greedy ID atpositions3..17. ActualEOS may stop
sooner. This preserves the current N3/prefill anddecode1 tensor shapes and avoids
newprefill-size compilation as the first longertext experiment. Existing compiled
math artifact may be reused only after honest source/ABI binding review; a new
host producer fingerprint cannot be silently called the old producer.

Expected delivery is exactprompt/generatedIDs, UTFtext, all4DP votes/scores,
positions/state summaries, truthfulEOS or maxbudget termination and any truncation.
There is no canned story/keyword/token-answer requirement. Sixteen tokens may
still end midsentence; report it rather than append or hide truncation. Original
reference runtime is unavailable per ACCEPTANCE.md, so semantic coherence/relevance
needs explicit user/reviewer review and is not a hard token-parity gate. A later
repeat on an accepted fixed case is still needed for reproducibility; do not
claim it from one run. Do not add multi-prompt state-reset/resident-reuse machinery
merely to avoid another upload in this unit.

## Reused shape/source evidence and honest limits

Tiny queuedCPU checks use actual build_entry_contracts without compilation,
fixtures or tensor payload forN2/3/17/128,max16,start0. All have pages2 since
ceil((N+16)/128)<=2; both complete57 static shape/dtype maps equal currentN3.
Thus the same source key/range/offsetfold/scale/NZ/TPDP mapping and fullweight
plan can be reused, conditional on unchanged44 code and independent contract
review. Decodeinput shape[16,1] and global logical nonstatic247,111,680B remain.
No263-header/139614-key derivation is repeated.

This is META only. A newN uses the existing prefill entry specialization/factory;
it is not proof of dynamicS in ONE unchanged compiled artifact (decision9), nor
actual execution for allN2..128. The first longerN3 case stays capacity1/allactive.
N128+16 would cross a page atposition128 and extend beyond the previously bounded
128/129/130 chain; those extra states/publications need separate actual evidence.
Neither metadata nor a coherent16-token string replaces that numerical gate.

## Conservative future-call budget

Keep57 static39,624,816,576B/rank, GLOBAL633,997,065,216B eachphase and sequential
prefill/decode owners. Source caps remain unique139614keys/557336782272B;
164192 logicalreads/745846365184B eachphase; two-phase1491692730368B source and
1267994130432B H2D maximum (early EOS may avoid decode). No new full source reads
now or unique-source bytecount inflation across repeatedDP/TP/phase reads.

Current formula already accepts the new maxcount algebra; do not change it:

- Upload and firstcall perowner: publicfree>=remainingstatic+18,088,299,200B.
- Postforward: positive planningfloor9,797,894,144B.
- Laterdecode preforward: floor +remainingcalls_including_this*247,111,680
  +680,528,320 nominalcomm. With15decode calls, second requires13,937,985,984B,
  final requires10,725,534,144B. Fifteen fullglobal IO allowances pluscomm andfloor
  total14,185,097,664B, below the existing18.088GB firstcall reserve. These are
  planning arithmetic, not observed allocation attribution or a fit guarantee.

Each future call can retain a HOST pipeline-slot allocation; sharedGMbank0 is
not onehostslot. Do not zero futureIO based on guessed committed deltas. All
HOST64GiB/DISK2GiB/SHM16GiB, explicit Rootapproved1s/256MiB/source64MiB/firstcopy/
alloc/all16final/query-failure policies and8GiB safety remain. Actual peak/slot
residency/15-call lifecycle remains unmeasured until that operation. The active
case's observations may only be incorporated after its terminal audit. No
throughput extrapolation or claim that6hours suffices; cooperative20400s atsync
boundaries/queue21600/client>=22200 remain explicit STOP, no blindretry/heapresize.
