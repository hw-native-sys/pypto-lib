# Complete-entry noEngram worker handoff

Implemented actual Engram bypass and corrected complete-forward expert ABI.
Actual ONBOARD40-layer prefill2 numerical validation passes. Decode1 native
execution is BLOCKED by halMemCtl EACCES/RunPreparationRefused code13 before
comparison. No continuous-state/page-boundary,real-checkpoint or generalF09
acceptance is claimed. Main integrates and independent reviewer accepts.

## Exclusive ownership and implementation

Worker fullentry_noengram owned prefill_fwd.py,decode_fwd.py,moe_exchange.py,
layer_reference.py and this artifact directory. Only the two forward files
changed;moe_exchange/layer_reference remain unchanged. No shared ledger/model
outside scope,checkpoint/PyPTO/runtime writes,commits,external filing or resets.

New top-level engram_enabled:pl.Scalar[pl.INT32] follows engram_rotation in
both signatures and every actual intermediate caller. Owning ScalarSpec uses
compile_runtime=True so the scalar survives as an actual runtime parameter.
Owning default value1 preserves enabled behavior;value0 gates actual Engram
compute and copies incoming HC stream. Owning goldens honor this switch.
Both CLIs expose --engram-disabled. Main driver binds0 explicitly and reports
noEngram scope. Inert arguments remain ABI-present; no original Engram hash
implementation or new lookup contract was introduced.

Existing expert ABI defects were independently exposed during decode1/2
specialization: _RANK_W*/_HOST_W* omitted N_LOCAL_EXPERTS=24 while local kernels
and fixture/loader shards require it. Prefill encoder/decoder/l2/top signatures
additionally retained rank-only declarations despite indexing an expert-layer
axis. Corrected shape hierarchy is local[24,...],rank[layer,24,...],host[16,
layer,24,...] for W1/W3/W2 and scales. Bounded layer axis1 clamps existing layer
selection;real generation layer axis40 remains supported. No kernel math,
threshold,topology,quantization,accepted rotary/order/router or F10 layout change.
GeneralF09 runtime active-count/unequal/idle/multi-request work remains OPEN.

## Execution evidence and honest boundaries

All new configs explicitly execution_mode=False. normalize_config reports
ExecutionMode.ONBOARD. Generated distributed_meta.json records platforma2a3;
native Callable image/runtime logs prove actual execution where stated below.
No SIM result contributes. Scripts explicitly reject SIM mode/metadata.

- task_20261106_211309_199942521675 --no-device: bounded expert-layer1
  complete decode1 and prefill2 compile PASS,24.08s/25.62s.
- task_20261106_211615_204403832262 --no-device: real expert-layer40 ABI
  complete decode1/prefill2 compile PASS,24.47s/26.12s. Compile only;no real
  weights loaded. compile_realabi_queue.log and generatedmeta preserve shapes.
- task_20261106_211421_201926815403 auto16devices0–15 exit0: actual noEngram
  prefill2 complete40-layer runtime90.83s. Initial runner PASS explicitly says
  validation skipped(no golden_fn);that line alone is not numerical evidence.
  Subsequently native.py executes owning composed40-layer golden and all
  owning comparisons,then prints COMPLETE40_NOENGRAM_NUMERICAL_PASS prefill.
  prefill_native.json records final passed:true,overall243.03s after setup.
  Native Callable logs and generateda2a3 are in prefill_native_queue.log and
  prefill_native_final/.../distributed_meta.json. This fixture is isolated
  synthetic plan at1248,not a valid accumulated real-prefix lifecycle proof.
- task_20261106_212340_215181910924 auto16devices0–15 exit1: actual decode1
  compiled and native images loaded,but runtime preparation failed. FIRST
  error at decode_native_queue.log2199: halMemCtl rc13 after13attempts/2000ms;
  following get_aicore_regs(AIC_CTRL)/init_aicore_register_addresses(Ctrl)
  failed13. Final error: LocalMailboxEndpoint worker3 failed with
  RunPreparationRefused:prepare_native_run code13. Subsequent507901hdcdisconnect,
  507018faults and teardown timeout are retained,not numerical controls. Job
  eventually exited1 and released all16device locks. No new launches/resets.
- task_20261106_212002_209169223268 --no-device syntax/ruff bothsources PASS.

This preparation failure matches the separately recorded reviewer's EACCES
surface;root cause remains unproved. It is an environment/preparation blocker,
not evidence that the model's math is wrong or that DSL/runtime has a defect.
No speculative KNOWN_PYPTO_ISSUES claim or permission workaround is justified.

## Memory and still-unrun continuation

Bounded1expertlayer fixture decode1:104424152580logical bytes,
99128843780unique host storage bytes,6526509536logical bytes/rank.
Prefill2:104482127108logical,99186818308storage,6530132944/rank.
Real40expertlayer shape decode1:634999152132logical,39687447008/rank;
prefill2:635057126660logical,39691070416/rank. Expanded synthetic layer views
keep hoststorage~99GB but do not reduce the true40layer logicaldevice layout.
These omit EP windows/allocator/transient/workspaces and are NOT measuredpeakHBM
or proof of actual checkpoint fit. Main owns header-only actualcheckpointbudget.
No realweights/Engram tables loaded by worker. Native prefill did not sample
HBM;claim no peak. continuation.py includes sampler for subsequent work.

Prepared continuation.py uses real Main driver metadata/builders,handoff of
actual returned persistent caches,prefill128 at0..127 then decode128/129/130,
with runtime_dir reuse of ONE decode artifact and per-step own references.
It is NOT LAUNCHED or validated because actual decode1 preparation is blocked.
It must receive host/API review before use. No paper claim that this script
proves persistent-state/page/parity correctness. Expected future nativeproof
must also verify untouched rows/native logs/mode and distinguish per-step
reference conditioning on actual prior state from strict cumulative reference.

## Preserved own failures and recovery

compile_attempt1/2/3.log are helper import mistakes before compilation
(golden.config,DistributedConfig import locations),not upstream bugs.
compile_missing_expert_axis.log and t2_queue.log reproduce actual preexisting
shape metadata failures. compile_attempt_order.log is helper unordered specs
ABI rejection after codegen;fixed by owning _entry_specs. compile_prefill_axis
captures remaining oldprefill shape before fix. First nativeprefill task
211201_19857995168 imported intermediate source before finalprefill fix and
failed compilation;prefill_native_attempt1.log preserved. Fresh final job above
runs corrected source. None of these failures counts as a numerical control.

Pins read from selected tree:PyPTO38a8aa6da24c62bf0c6f6f9ef94deb513fbb4d2e;
simpler31f9e72ee6832cd198b77df18c1c305e7fd83676;
PTOASv0.72 aarch64wheelSHA69bd51ccd3d9cc4ad0a6a7b296b46dc533d60b214ded9450fb0f90a83a765ca6;
pto-isaf83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7.

Fresh repro (after environment disposition;do not retry devices automatically):

task-submit --device auto --device-num 16 --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && export PATH="$PTOAS_ROOT/bin:$PATH" && export OMP_NUM_THREADS=16 && export PYPTO_PROG_BUILD_DIR=/absolute/fresh/build && python experiments/dsv41-flash-w8a8-a3/logs/generation_noengram_1793970065/worker_fwd/native.py decode'

The helper writes resultJSON beside itself;copy helper to freshreviewdirectory
or adapt its artifact root before independent reruns,never overwrite evidence.
No existing SIM helpers are reused. Source hashes,mode proofs,task IDs and
unique measured work record are in summary.json/source_hashes.json/work_records.
Worker will PARK with all own jobs completed. Main must retain separate
OPEN F05/F07 original controls/generalF09/F13 restoration/full-generation gates.

Scope clarification: the actual prefill numerical run uses EXPERT_LAYERS=1,
a single synthetic24-expert shard per rank reused across all40layer calls;
dense weights carry all40layers. This is40-layer COMPUTE composition,not forty
distinct expert checkpoint shards. The EXPERT_LAYERS=40 true checkpoint layout
was compiled only and its635GBglobal logicalfixture is not realweight execution.
