# HC captured-native / independent-original proof — FINAL / PARK

**Native case FAILED (exit1); offline original comparison PASS.** These remain distinct statuses. Captured candidate native output matches the pinned original exactly on this synthetic HC boundary case. The strict native helper failed because its golden output buffer aliased its dispatch output. No retry, helper fix, compiler/runtime change, or third NPU case was performed.

## Actual captured values

Native task **task_20261108_021439_342901924784**, actual device **[0]**, executed two genuine L2 calls using the fixed 64MiB ring and a2a3/ONBOARD compiled artifacts. Both runtime calls returned; candidate callback passed, old callback failed; the unchanged old assertion failed at `probe.py:199`. `native_result.json` and both `*_run_result.json` retain the failure and full comparison receipts.

One offline audit, **task_20261108_021816_343981631498**, `--no-device`, **exit0**, independently recomputed the unchanged pinned original `hc_post_reference` and checked exact saved native tensors:

| Saved native output vs independent original | Candidate | Old control |
| --- | ---: | ---: |
| Values compared | 102400 | 102400 |
| Numerical mismatches | **0** | **61440** |
| Non-BF16-representable FP32 values | **0** | **61440** |
| Maximum absolute error | **0** | **0.00390625** |
| Mismatches per fixture row | 0,0,0,0,0 | 15360,20480,20480,5120,0 |

First old mismatch is flat index **0**: actual **1.001953125** (FP32 bits **1065369600**) versus original **1.0** (bits **1065353216**). Old native output is also bit-exact with the unchanged actual old CPU golden; candidate native output is bit-exact with actual proposed CPU golden and independent original. Thus no compiler arithmetic fault is indicated by these data.

The native callback's old receipt instead reported zero actual-vs-expected differences because its captured expected tensor was overwritten: old saved actual and saved expected are identical, and **both differ from independent original in 61440 values**. The old callback correctly detected 61440 nonrepresentable values, so the combined comparator failed; the later required `different_values > 0` condition failed. The candidate callback result alone was not treated as independent original parity; the offline recomputation supplies that missing independent check.

## Provenance and exact commands

Unit `native_hc_durable_receipts_1794075226` byteforked all 237 frozen correction files from manifest **`bab7d916f2f27c58c413189aff3915ac2b09dda243db1b2aee2ce5ef2864ca18`** at the same relative depth. Inherited seals remain `input_*`; lineage preserves all source hashes. Probe/receipt store/math/reference/140 original compiled inputs stayed byte-identical. Candidate HC is **`c6111634f78d53ad0cb4196a533a2d89097d4cfa0859688382c0072b60840885`**; original helper file is **`577c667fbf8f8b6f20de8a0ac54c6c0435605aaee1f6c2ccc463b446055bd286`**, from pinned rc2 archive **`48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c`**, commit-prefix **8fde6d4e335fe300e2b949aab0e31659cce0811d**.

Current producer stays **b3e19fd001c74f7a5b497c791c01924ca09c0678df66114fce599e890d8fee06**, generate **fba8a4ef9a06f694fa53278bb80ae2ad70d111336dbead938ab9f8511d27e16e**. Historical CC is not labeled current. No live model/math file changed; HC draft remains unapplied.

Exact native command, also in `submitted_command.txt`:

```bash
task-submit --timeout 1200 --device auto --device-num 1 --max-time 600 --run 'bash experiments/dsv41-flash-w8a8-a3/logs/native_hc_durable_receipts_1794075226/run_native_clone.sh'
```

Exact offline command, also in `audit_command.txt`:

```bash
task-submit --no-device --max-time 180 --run 'source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/native_hc_durable_receipts_1794075226/audit_native_receipts.py'
```

Native output pairs are immutable under `receipts/{candidate,old_control}/.../{actual,expected}.pt`, with individual SHA256 in their receipts. `audit_reconstructed_fixtures.pt` contains exact CPU reconstruction of the immutable source-defined five fixtures; **native input tensors were not separately snapshotted**. `audit_original_expected.pt` contains independently recomputed output from those exact fixture definitions. The audit does not present reconstructed inputs as measured device-input snapshots. `numerical_proof_pins.json` pins every fixture/reference/output/audit/probe byte used for the conclusion. No real checkpoint payload or full model.

## Exact helper alias cause

The callback contract is correct: `golden/validation.py:104-120` calls `compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol)` using the dispatch output and stored golden tensor. There was no callback argument reversal or cached-golden replay.

The own helper creates **every** TensorSpec, including pure output, with tensor-valued `init_value=v.clone()` (`probe.py:178`). `golden/spec.py:163-164` returns `self.init_value.to(dtype=self.dtype)`; same dtype returns the same storage. `golden/runner.py:378` calls `create_tensor()` for dispatch buffers; `:1452` calls the **same output spec** again for golden scratch. The original golden writes correct expected values there, then device output copyback overwrites that shared storage before validation. `.cpu()` in validation does not clone CPU tensors.

The audit reproduces this using the exact public output TensorSpec API: two `create_tensor()` calls have identical data pointers; writing independent original through golden scratch is visible in dispatch storage, and writing saved old native values through dispatch storage overwrites golden scratch. Native pointers were not directly recorded; the source/API proof plus immutable value evidence establish this own-helper storage mistake. No compiler blame or project harness edit is proposed from it. A future helper correction would require a fresh output allocation strategy, but none was implemented in this case.

## Cleanup, honest status and release

Prequeue task **task_20261108_021416_34283994164**, exit0, verified HOST8GiB/DISK2GiB/SHM1GiB floors and all current source/candidate/compiled/runtime pins. The floors were checked again after actual allocation; scoped GO **observer-20261108-hc-durable-receipts-native1** was created only after TASK_DEVICE=0.

Offline audit reports **31 passed / 0 unmet**: exact tensor hashes/counts/per-row differences/original semantics, source/parent/previous-failed-unit/compiled/runtime identity, both callable images freed, all **9 recorded own PID/birth identities absent**, raw one-device release, resource floors, empty queue. Raw native log confirms `[npu-lock] 已释放设备 0 的锁`. Native process success is not claimed: supervisor child exit1 and native entry/result FAILED remain preserved. Audit PID/birth is separately confirmed absent after completion; final queue is 0/16 with no pending/running jobs.

Known separate fit task **task_20261108_021440_342919127289** terminated exit1 at syntax parsing, as reported by Root; it was a CPU-only job, outside this task's PID tree and not another NPU case. It was terminal before the global queue0 receipt. No other process was acted upon.

Root may evaluate captured-native candidate plus independent-original CPU comparison for this **synthetic HC BF16 boundary only**. The whole native helper remains FAILED. No repeated-text repair, original full-model numerical parity, F09, dynamic-serving acceptance, or generation performance claim. All writes confined to this NEW unit, shared writes=0; Root imports once. FINAL/PARK after failure plus successful independent offline proof; no remaining task and no retry authorization used.
