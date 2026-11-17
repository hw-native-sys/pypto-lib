# Empty-fit post-allocation reserve check — artifact-only narrow repair

The candidate is a NEW copy of sealed generation_entry_1794035858/fit_probe.py.
Only two lines were added to run_phase after existing post-query telemetry is
written: require after.free >= remaining[rank] + limits['reserve'][rank], else
raise Refused. patch.diff is the entire change. Product45source762dc, old31
artifacts and all old bundles remain unchanged. This unit has no NPU, native
owner, tensor payload, real/large allocation, calibration or guard-unblock grant.

## Defect and fix

The old function checked free before each allocation, but only recorded free
afterwards. Extra allocator overhead on the final allocation could consume the
required reserve without any later precheck. Independent reviewer153727 and
Main's own baseline154012 both reproduce false EMPTY_ALLOCATIONS_PASS. On rank15,
one largest tiny buffer leaves free50<reserve100; the final57th tiny allocation
leaves free80<100 while all912 handles still count as a successful aggregate.

The new check uses the SAME remaining-static and reserve definitions as before.
Public free already reflects committed/prewarm memory, so committed is NOT
subtracted again. No new budget term, calibration assumption, runtime setting,
4GiB/TMR requirement, source pin,57-resource target or default refusal changed.
The post-query snapshot and JSONL are persisted before rejection. The handle is
already registered before schema/query/check can fail; original finally tries
all frees and closes the owner, preserving the first error over cleanup errors.
All allocator/runtime/measurement/cleanup algorithms remain original.

## Actual CPU evidence

All commands are task-submit --no-device --max-time120 --run 'bash <UNIT>/<script>'.
run_baseline.sh, run_candidate.sh and run_remaining.sh activate the selected
environment in the repository. Fakes use Python integers and metadata only:
57 resources of8bytes/rank, no Torch tensor/device memory or real owner.

- 154012_97688112377 COMPLETE0: OLD largest rank15 free50 and OLD final57th
  aggregate rank15 free80 both falsely PASS with reserve100. Cleanup succeeds;
  this demonstrates the guard gap, not an allocation/runtime failure.
- 154106_97930419199 COMPLETE0: new largest/final57th cases reject at the
  post check; all16 or912 registered handles receive free attempts and normal
  owner close. Normal57 exact-boundary run passes912 handles/all16 ranks,
  including finalfree==reserve100. Large committed values already included in
  free demonstrate no double subtraction. An asymmetric rank7 overrun stops
  there after8*57 handles, not only at rank15. A post-allocation query failure
  still frees all912 registered handles and normal-closes. Guard+free+close
  failures preserve the original guard error; query+free failures preserve the
  original query error; cleanup-only failure cannot become PASS. Fresh largest
  and aggregate owners close independently with unchanged baseline semantics.
- 154246_98320312878 COMPLETE0: after FIRST allocation free547 is above reserve100
  but below remaining448+reserve100=548. The new guard rejects before a second
  allocation and frees its sole handle/normal-closes. This proves the remaining
  target term was not omitted in favor of a reserve-only check.

Initial153948_97467023896 failed parsing the NEW test helper's conditional
expression (missing whitespace before literal57). It never ran the probe.
Raw syntax failure is preserved; the unused helper was removed and the corrected
baseline/candidate tests above pass. No product/candidate algorithm was modified
to accommodate a test failure. Candidate source has no unexpected failure.

## Frozen ownership and limits

Main exclusively owns this NEW artifact directory; no worker or reviewer files
were written. Existing generation_entry31 manifest and product45 hashes verify
unchanged before/after. Main records measured work/TASKS/TIME_LOG under lock.
All own CPU jobs ended; Main PARKED for root-assigned independent reviewer_fit
validation on this frozen candidate. This patch is not applied to product or
old evidence. It supplies no actual memory-fit/calibration result and does not
remove fullweight guard. Reviewer_fit's separate post-aggregate baseline and
forward-peak attribution questions are intentionally not implemented here.
