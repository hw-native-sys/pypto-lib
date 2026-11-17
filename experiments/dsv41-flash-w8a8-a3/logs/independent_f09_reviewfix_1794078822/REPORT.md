# Independent repaired F09 review and zero-sample compile — FINAL/PARK

Disposition: **FAILED compile; no generated dependency or native acceptance**. The repaired source addresses the four old76 review findings at source level, but its optional tensor/TaskId return interface fails actual public compilation. Stop was honored at the first failed stage; no retry, old-physical compile, fullentry compile, model fixture, payload allocation or device call followed.

## Frozen identity

Reviewed Main `../f09_zero_active_reviewfix_1794077743` FINAL/PARK. Bundle93 manifest SHA256 `188d06bcf2695519e392a91161fab6bef270d136f9cf5d5768e7a9b468b8590f`; copied45 UNAPPLIED source manifest `f08bb59b008df8ee1f2d4588414de9ff52976848f85e089d57d4dac47257401f`. All 93 bundle and 45 copiedsource files checked independently before imports, all OK. Model operator `4fc355b970b069df9c5d2862eec539efdb18c45f6236586d2c777e208035533b`, adapter `b48405c45e46c6185b730bd1711154a2cb74c31e4fbd19228130b2a472587589`, probe `c3aad8e9ad2596545557d092587584be471bc8da0fad5364bade7f3608a1391e`. Before/after pins and all45 live M02 checks remain OK. Live manifest remains e07b28713eadd00a306fd60264223dbf6ceec53761c7ac48c347032011c9ca25. No frozen artifact or live source was written.

Original semantic reference ancestry is unchanged from old76 review: upstream rc2 revision `8fde6d4e335fe300e2b949aab0e31659cce0811d`; archived tests/deepseek_v41_utils.py file SHA256 `577c667fbf8f8b6f20de8a0ac54c6c0435605aaee1f6c2ccc463b446055bd286`; original MoE/model file `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`. This unit adds no original-framework execution. Current HC post BF16 RINT→FP32 boundary, quantization arithmetic, thresholds and fullentry19 ABI are unchanged from old76/current M02.

## New blocking finding C1: conditional variable return arity is inferred as TensorType

Actual task `task_20261108_031644_2608314306` completed **exit1** at first `count_active` signature, after 4.86 seconds in the driver. Public JIT import succeeded; public `entry.compile(config=...)`, called through the frozen `generate.compile_entry(..., specs=None, ...)`, failed parsing `moe_ffn`:

```text
TupleGetItemExpr requires tuple to have TupleType, got TensorType
```

Trigger: `model/moe_exchange.py:436` unconditionally unpacks `routed_active, ep_done = moe_ep16(..., return_completion=True)`; the primitive contains a conditional `return routed, return_consumed_tid` followed by top-level `return routed`. The corresponding `moe_ffn` uses the same conditional tuple versus final tensor pattern, and the probe also unpacks it. This internal unconditional unpack affects the shared FFN path even when its external caller uses the default tensor return.

Read-only diagnosis against the actual pinned specializer (installed/check-out files hash-identical, SHA256 `e7ac2ebe42d8980e1b9ad777e81faa95511e0b0de2ffc8b96698caa9609fa154`): `jit/specializer.py:_infer_return_type` line 1386 walks the original AST breadth-first and selects the first value-return. The final top-level tensor return is encountered before the nested tuple return. Return inference happens at line 2022 before body transformation; a true constexpr argument therefore does not make this inference a tuple. Its tuple logic additionally states nonTensor elements such as TASK_ID need a matching explicit tuple annotation; these helpers have no such annotation.

Public L3 documentation supports passing `pl.Scalar[pl.TASK_ID]` across inline boundaries; public types document `pl.TaskId` and tuple return annotations. They do not establish this variable-arity/unannotated conditional contract. Classify C1 as an **own DSL/interface error as written**, not a proven compiler defect. No upstream issue was created. The pin gate passes: pypto `38a8aa6da24c62bf0c6f6f9ef94deb513fbb4d2e`, runtime/simpler `31f9e72ee6832cd198b77df18c1c305e7fd83676` matches gitlink, installed PTOAS metadata 0.72 matches pinned v0.72, ISA `f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7` matches runtime pin. No environment mutation was made.

Minimum future source remedy: choose a stable, explicitly typed public return/completion contract and update every actual shared-operator consumer coherently. A fixed Tensor+TASK_ID tuple is a candidate; do not assume it compiles before a fresh authorized public signature check. Do not add a near-duplicate FFN or private compiler patch merely to retain the old private tensor return. This reviewer implemented no remedy.

## Prior findings review

| Finding | Repaired frozen source | Evidence limit |
| --- | --- | --- |
| R1 legal zero-active anchors | reuse wait reads documented zero-initialized consumed[self,0]; return signal reads alwayswritten expert_counts[0]; release reads initialized return_consumed[self,0] | Source-valid accesses, actual codegen blocked by C1 |
| R1 always-existing EP completion | capacity2 combine has inner token<active guard, no outer active predicate; release depends on combine; probe explicitly depends on returned ep_done | Intended count→push→return→ready→combine→release→export path visible in source; no emitted orchestration/task graph exists |
| R2 verdict scope | Adapter builds fixed-independent-output selector from same gate factory; executes and records unconditioned_direct19 and original_staged19 separately; preserves failures and requires both for BOTH_ALL_PASS | Same gate expressions/tolerances; no full math run; within-call reference starts from actual prior state, not whole independent sequence rollout |
| R3 state preservation | Exact complement of position%STATE_CAPACITY checked on active requests; all state rows checked on idle requests; bounded STATE_CAPACITY2 enforced | Correct for one query/request per call; no native state capture |
| R4 negative specificity | Exact INT32/nonnegative [dest16,src16,expert24] forced-rank8 experts0..5 matrix checked before specific192-vs96 verdict | No malformed generic ValueError becomes PASS; no actual capture/native result |

The patch retains every rank's EP participation at local-active0. Runtime active scalar controls source route population; capacity remains2. Middle rank8 local-active0 must still serve96 remote routes. Source matrix oracle independently expects active source counts, and old-physical control differs solely by active=capacity2. New pureOut export is a separate diagnostic, not an invented fullentry output.

The B2/T2/four-page, A[3,1]/B[0,2], state A1/B0 and positions126/127/128 scenario remains fixed. Actual-post→next-initial, persisted pristine clones, independent static reconstruction and exact growing history checks remain intact in the offline design. No candidate/golden expected-storage alias or tolerance workaround was added. General goal9 runtime physical B/T variation, EOS/slot reuse and full original parity remain open; decision10 temporarily disables Engram only and does not waive those requirements.

## Actual execution and artifacts

One queued CPU task only, exact command in COMMANDS.md. Compile IDs0..15 were logical topology metadata, not allocated physical devices. Explicit configuration normalizes to ONBOARD/a2a3; device initialization/preparation was never invoked. Profile: E1 synthetic expert-axis ABI, shared actual FFN, physical EP capacity2, dynamic T/B signature annotations retained; future desired dispatch geometry T2/B2/logicalpages2/physicalpages4/stateblocks2. No tensor sample or TensorSpec.create_tensor/full fixture/header/reader/payload call was made. The signature stage failed before generated metadata, kernels or orchestration could be emitted.

Driver records: environment.json, compile_started.json, compile_failure.json, compile_ended.json, raw queue.log. Successful compile stages=0. `count_old_physical` and `decode_capacity2` were NOT ATTEMPTED by stop-on-first-failure. Consequently there are no source-bound emitted HOST/CHIP/metadata/Out/InOut dependency results to accept. Runtime deps.json/name_map and local0 scheduler liveness would in any case need a separately authorized native capture.

Applied dependency-liveness, task-submit-npu and environment-setup workflows as relevant; read canonical scheduling/L3/L2/debugging/installation instructions. No tiny suite was repeated. Main's fake DSL TaskIds/import self-evidence remain source preparation evidence only. Root and Main received C1 immediately and were told task exit/no active reviewer job before any terminal process audit.

## Recovery / PARK

Stop here on frozen93. Future source repairs require NEW owned artifact unit and freeze, then independent source review and fresh specifically authorized compilation. Preserve this failed task and do not relabel it successful. No blind retry, successful artifact recompile, live application or native GO is granted. Current fullreal shared HC work uses unchanged live M02 and is unaffected by these copied-source failures. Root imports the adjacent measured work record; shared writes=0.
