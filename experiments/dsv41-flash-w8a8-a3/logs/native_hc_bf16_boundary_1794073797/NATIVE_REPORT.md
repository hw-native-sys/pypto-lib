# Native HC boundary microproof — FAILED / not accepted

One authorized native case, task **task_20261108_015136_337085225477**, allocated device **[0]**, completed **exit1**. No retry, resize or implementation/runtime change. The proposed candidate passed its native comparator; the overall case failed the old-control evidence assertion. This does **not** close native HC acceptance or establish repetition causality.

## Exact scope and lineage

Root GO `observer-20261108-hc-bf16-native-micro` was recorded after allocation in `authorization.json`. The unit byteforked the immutable HC validation source/compiled inputs at the same relative directory depth, verified its 159-file manifest `7546c8fa98d4a5c09313b35e8b612de106d276b5aae1ec8f1e16454136e54093`, and executes only this NEW clone. Inherited seals are preserved as `input_*`; lineage records every original input hash. Parent artifacts, Main's HC draft, source45CC, F09 and shared-owner units remain unchanged.

Actual source candidate is the genuine proposed `hc_post_test` → `hc_post`, SHA256 `c6111634f78d53ad0cb4196a533a2d89097d4cfa0859688382c0072b60840885`; the genuine original local compiled operator supplies the negative control. Both retain their signature-compiled a2a3/ONBOARD artifacts. `probe.py` is byte-identical to the frozen validation helper. Five synthetic rows, H4/D5120, FP32 residual storage with BF16-exact input values and FP32 coefficients; signed, midpoint, nonidentity and exact cases. Actual original pinned reference supplies expected output. Ordinary HC gate and additional bitwise/BF16-representability checks were unchanged. Fixed operator-specific ring heap **64MiB**, not the full-model 4GiB budget. No real weights or full model.

Prequeue CPU task `task_20261108_015023_336824625378` exit0 verified available HOST=1723063881728, DISK=8324263182336, SHM=1079343157248 bytes, exceeding required 8/2/1GiB floors; source45, candidate, compiled and runtime/library pins verified before native submission. Exact native command is in `submitted_command.txt`; client timeout1200/max-task600/auto1.

## Observed result and first cause

`probe_native.log:38` records candidate `'output' PASS`, shape `(5,4,5120)`, FP32. Its comparator requires the existing ordinary gate AND bitwise original equality AND every FP32 value BF16-representable. Thus the single candidate dispatch passed all three checks on 102400 values. Generated RINT roundtrip source remains pinned in `generated_cast_evidence.json` and both native generated artifacts are retained.

`probe_native.log:78` records old-control `'output' FAIL` after an ordinary successful runtime dispatch. The helper then fails at **probe.py:187**, specifically:

`assert seen[0]['different_values'] > 0 and seen[0]['nonrepresentable_values'] > 0`

The check establishes that the required pair of positive old-control counts was **not** satisfied. Exact counts cannot be recovered from retained evidence: comparison receipts were only in process-local memory/RunResult.error, and `native_result.json` is written after all assertions. It was never written. The source/log do not establish which count was zero or what the unexpected old output values were. A comparator FAIL alone is insufficient to claim the required numerical old-control rejection. No numeric counts from the earlier CPU case are substituted for these missing native values.

This is the first observed failure and an evidence-retention flaw in the prepared helper. The old-control result additionally requires investigation; no compiler/runtime defect is assigned from this incomplete evidence. `supervisor_result.json` records child exit1, normal_child_exit=false and native_result_exists=false. Its first_error=null only means the supervisor itself raised no monitoring exception; it does not erase the child assertion. There was no blind retry, extra device run, private API or patch.

## Cleanup and one metadata audit

Both runtime calls report `Callable image freed` in the raw native log; normal process success is **not** claimed. Raw lock log reports exactly `[npu-lock] 已释放设备 0 的锁`. Outside-queue terminal status is `completed (exit=1)` and fresh queue status is 0/16 with no pending/running jobs.

One post-terminal metadata CPU task **task_20261108_015341_33775953750**, exit0, reports **18 checks passed / 0 unmet**. It verifies honest failed-case status, exact authorization, candidate PASS + old comparator FAIL lines, both callable images freed, raw lock0 release, all **10 recorded PID/birth identities absent**, lock PID absent, unchanged source45CC/parent159/candidate/old/config/original files, unchanged original compiled inputs and runtime libraries, and resource floors. Audit process PID/birth is separately confirmed absent after terminal; final queue remains empty. This metadata audit PASS does not convert native FAILED into success.

## Remaining evidence and handoff

The failed case is sealed as-is. A future separately authorized corrected probe should durably write each comparator receipt (counts and representative first mismatches) before assertions, including failure paths, and retain original/actual values for the old control. That correction and any new native run are **not performed here**. The candidate's observed native pass is bounded to this synthetic operator input set; overall microproof acceptance remains open until the required old-control evidence is resolved. No full-model parity, F09 or repeated-text repair claim.

All writes are confined to this NEW unit. Root imports once. Shared writes=0. PARKED after terminal failure and metadata audit; no remaining native task.
