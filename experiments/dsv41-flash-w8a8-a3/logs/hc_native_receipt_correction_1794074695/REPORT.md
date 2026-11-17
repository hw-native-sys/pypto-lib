# HC native helper durable-evidence correction — FINAL / PARK

**Preparation PASS; no new NPU run or signature compilation.** The failed native old-control discrepancy remains **UNKNOWN**. This unit corrects evidence retention only; it does not close native HC acceptance or text-quality causality.

## Frozen inputs and current producer

Independent preparation comes from `hc_bf16_boundary_validation_1794072636`, manifest `7546c8fa98d4a5c09313b35e8b612de106d276b5aae1ec8f1e16454136e54093`; failed native case remains immutable at `native_hc_bf16_boundary_1794073797`, manifest `31731a5902348a664cb4c07dcfa9374770334f73ee99b166c383c698d5ebf794`. Their genuine proposed/old HC modules, original helper, config and all 140 signature-compiled inputs are reused byte-identically in this new unit. Proposed HC SHA remains `c6111634f78d53ad0cb4196a533a2d89097d4cfa0859688382c0072b60840885`.

The **current** producer is Main's finalized generate-only application: `current_source45.json` SHA256 **`b3e19fd001c74f7a5b497c791c01924ca09c0678df66114fce599e890d8fee06`**, live generate **`fba8a4ef9a06f694fa53278bb80ae2ad70d111336dbead938ab9f8511d27e16e`**. Main's final 17-file manifest is `9aeb50ef3e9f5b8e90993063d7ccb03a668010a310cd16aa83063ee69d94ede5`. All 45 live files were verified against this binding. Only generate differs from historical CC; HC and layer-reference remain unchanged. Earlier environment/compile provenance mentioning CC is historical, not a current-source label. HC draft remains UNAPPLIED.

## Evidence correction

`probe.py` now delegates comparisons to `receipt_store.py` and retains each run result before the existing acceptance assertions:

- Save exact synthetic actual/expected tensors (FP32 `[5,4,5120]`, 409600 data bytes each) before summary or ordinary comparator code can raise. Record tensor paths, SHA256, shape and dtype.
- Persist bitwise/numerical mismatch counts, BF16 nonrepresentability counts, finite checks, max absolute error, twelve top differences with actual/expected bits, and per-row counts **before** ordinary comparison.
- Persist comparator success/error in `finally`. Atomic temp-file replacement plus file/directory fsync makes completed receipts durable.
- Persist every harness RunResult, including its error text, or raised exception immediately before assertions; always write final `native_result.json` in `finally`. Entry refusal/preflight errors additionally produce `native_entry_result.json` and a final failure stub if the pair never started.

The original fixtures, reference computations, compile path, output contracts and all math/helper functions other than native orchestration are AST-identical. All original pair acceptance assertions are AST-identical, including **positive old-control mismatch AND positive nonrepresentability counts**. Candidate still requires the existing ordinary ratio comparator, bitwise original equality and every FP32 value exactly BF16-representable. The ordinary `atol=1e-6,rtol=1e-5,max_error_ratio=5e-3` and harness `rtol=atol=1e-3` are unchanged. No weakening, new numerical reference or recompilation.

## Queued CPU evidence

Task **task_20261108_020903_341558728088**, `--no-device`, activated environment, **exit0**, verifies six fake-run cases against the actual corrected evidence/acceptance path:

| Injection | Required observed outcome |
| --- | --- |
| Valid candidate + ordinary old synthetic control | Pair PASS; both exact tensor pairs retained |
| Candidate wrong values | Existing acceptance fails; candidate tensors/result/final failure retained |
| Old wrong but BF16-representable values | Existing strict old assertion fails; saved counts **102400 different, 0 nonrepresentable** |
| Runtime exception before old comparison | Candidate tensors survive; old RUN_EXCEPTION and final failure retained; no invented old tensor |
| Ordinary comparator exception | Exact tensors and numerical summary survive; comparator/run/final exception retained |
| Missing comparator | Existing missing-receipt assertion fails; returned result/final failure retained; zero invented tensor pairs |

Every saved tensor was reloaded on CPU; file hashes, dtype, shape and numerical counts were rechecked. The BF16-representable wrong-output injection is an intentional test scenario, **not a reconstruction or diagnosis of the failed native old output**. CPU-only context markers distinguish these files from device evidence.

Task **task_20261108_021046_341972114274**, `--no-device`, **exit0**, separately proves a native entry refused for missing Root GO writes durable failure receipts without calling the pair or NPU. No native execution was authorized in this unit.

Two earlier CPU test attempts failed due test-only mistakes (literal AST formatting comparison; an injected value assumed representable before explicit rounding). Their logs and partial receipts remain under `cpu_attempt*`; both were corrected in the CPU test without changing production math or assertions. No compiler/runtime fault is alleged.

## Exact prepared command and next boundary

`run_native_AFTER_NEW_ROOT_GO.sh` is concrete and frozen. `NEXT_COMMAND.txt` records:

```bash
task-submit --timeout 1200 --device auto --device-num 1 --max-time 600 --run 'bash experiments/dsv41-flash-w8a8-a3/logs/hc_native_receipt_correction_1794074695/run_native_AFTER_NEW_ROOT_GO.sh'
```

**Not submitted.** A future Root-authorized native unit must bytefork these immutable inputs and rebind that outer shell path to the new unit before execution, as with the prior native case. The helper uses its own HERE for output and compiled artifacts, requires one actual TASK_DEVICE and a new `--root-go`, verifies the new current45/source/compiled pins, normalizes a2a3/ONBOARD, and reuses the same two finite genuine L2 programs with fixed 64MiB ring. It must stop and preserve exact receipts on any failure; no automatic retry or assertion change.

`helper.patch`, `final_source_review.json`, CPU results, exact helper/command/source/compiled pins and complete manifest make this reviewable. All writes are confined to this NEW directory; shared writes=0. Root imports once. No model/golden/dependency/runtime/old-unit changes, NPU/grant, real weights, full model or extra compile. PARKED; no active task.
