# Production generation host repair — applied, CPU checked, frozen

Only models/deepseek_v4_1_flash_a3_w8a8/generate.py changed. The reviewed isolated
exact-prompt/DP draft is now applied, plus explicit PUBLIC ring_heap=[0,0,4GiB,0]
in make_run_config. Old45ab1 becomes manifest762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7;
generate.py SHA685cf3cfadf36c92b5f9c6280f9f9d90ebf0e3244133a2b550adea83ceb29cfb.
All other44 model/operator/golden/loader sources are byte-identical. Full-weight
guard and PreparedDispatch body are AST-identical; topology/math/thresholds and
stream/scalar/owner algorithms are unchanged. No NPU, checkpoint tensor payload,
large fixture/allocation, numerical forward or text generation ran in this unit.

## Resulting behavior

Original tokenized prompt IDs are preserved, with exact allactive prefillN2..128
and existing factoryN, sampling rowN-1, start+N position/cache handoff. Empty,
length1 and>128 prompts fail length preflight without adding/removing IDs. No
Please-continue text or hidden padding tokens are appended. Compile/report
--tokens keeps explicit2/16/128 buckets; device generation derives actualN.
General inactive-padding/F09 support does not follow from this bounded path.

Requested DP replica checking now enforces agreement across all4 DP groups on
TP4 reconstructed greedy token IDs. Both prefill and every decode sample retain
all4 votes, shard/global IDs/scores and phase/position/input/history context.
Disagreement fails with structured JSON/log evidence; failed tokens do not
silently enter history. EOS/history arithmetic remains unchanged. First path
is explicitly labeled noEngram, not original-model parity. Hard full-weight
guard still rejects even the opt-in flag; its function is unchanged.

make_run_config sets supported targeted4GiB ring2 through the public builder.
PreparedDispatch normalizes once for its retained owner and passes that SAME
config object to prepare and every dispatch. No runtime/private field or class
algorithm changed.4GiB matches independently proven native prefill/decode
configuration on oldab1; it is not a new memory-fit claim or arbitrary heap retry.

## Fresh applied-source evidence

All checks use task-submit --no-device; old draft PASS is not reused as current
execution. Exact commands are in run_cpu.sh/run_fit_cpu.sh and raw queue logs.

- task_20261107_152005_92670518025 COMPLETE0: applied-source host tests,
  actual public-config normalization/PreparedDispatch with tiny mocked worker,
  real40 signature META and ruff. Actual prompt lengths2/3/6/17/127/128 preserve
  IDs/factory rows/lastvalid/handoff/positions; unsupported0/1/129 reject.
  TP global reconstruction/ties and6 independent DP group/phase rejection cases
  pass. Actual main CLI structured failures at prefill and laterdecode preserve
  votes/history; EOS initial/later cases pass. Old saved source reproduces
  appended3->16 prompt, poison-tail winner15 and decodeposition141 instead128,
  plus non-gating DP disagreement. CPU fake backend is explicit; no native claim.
- Same task: actual make_run_config normalized ONBOARD[0,0,4294967296,0]; actual
  unchanged PreparedDispatch.prepare and4 calls receive the SAME owner config.
  Simulator request rejects. Actual real40 metadataN3/17/128 and decode1,57statics
  each, no tensor fixture/samples/global stacks/compile/device. PreparedDispatch
  and require_full_materialization_budget ASTs equal saved pre-edit source.
- task_20261107_152251_9335458061 COMPLETE0: first current-source empty-fit
  metadata/plan/default-refusal check. Preserved as fit_binding_initial_pass.log.
- task_20261107_152352_9373759600 COMPLETE0: extends only the fit-binding check
  with missing/2GiB/3GiB ring rejection, both phase57-resource metadata/bytes and
  default execution refusal. No memory allocation is performed. The copied
  artifact fit_probe.py differs from sealed old helper only in source-manifest
  path rebinding and exact public4GiB requirement; diff is fit_probe_rebind.patch.

No unexpected implementation/self-test failure occurred. Injected DP/length/
configuration/guard rejections are deliberate controls. Logs whose requested
CLI mode says device have explicit CPUfake annotations and are not ONBOARD
execution evidence. Existing N3/6/17 zero-sample signature compiles on unchanged
forward files are supporting historical compile evidence only; no redundant
full compile or model execution was run for this host-only diff.

## Independent acceptance and fit route ready for review

INDEPENDENT_ACCEPTANCE_PLAN.md proposes minimal native exactN3 prefill original
17gates, actual row2 strict sampling and partial-C2 actual-state handoff at
position3, optionally4. Actual+initial bytes persist before BOTH CPU reference
passes, owner closes before offline evaluation, source16/12 mutable metadata
and unchanged17/19 comparator semantics remain. Existing accepted N128 chain
already covers page128/parity on identical math files; it does not accept this
new producer automatically. No native test is authorized by this plan.

Canonical V3 is hardN128/oldab1. Old evaluator0929/reference_context also checks
old ROOT45 and cannot run unchanged after generate.py updates. The next small
independent case needs an explicitly reviewed artifact binding of new producer
762dc/generate685cf and44 unchanged mathematical source hashes, reusing the
accepted exact-S32 proof/comparator implementation with NEW context/policy
receipt hashes. No old source/policy/receipt rewrite or blind guard bypass is
proposed. First normal text remains blocked on subsequent independent source/
functional/fit/real-checkpoint gates.

FIT_RAMP_PLAN.md binds the existing largest->CLOSE/FRESHowner->57-resource empty
probe to current source and4GiB. Rankstatic39,624,816,576B, largest11,324,620,800B;
perrank public free guard includes remainingstatic, conservative phaseIO,
nominalcomm, measured extra peak and explicit margin without doublecharging
committed/prewarm heap or168 identitycapacity. Defaultrefuse, actual backend
contract verification and cleanup/STOP rules remain. It has CPU metadata proof
only, no physical allocation/first-touch/upload fit proof. Full-weight guard
remains HARD; no real calibration/allocation authorization is supplied.

Prior independent oldab1 source prefix17 plus actual128/129/130 original19each
and correct129 WINDOW/C2 numerical rejection are sealed in
native_decode_capture_v3_1794031922. Scope is bounded syntheticE1/noEngram/
originalstaged comparators, not strict all-output unconditioned history, actual
40expert checkpoint or text. No old evidence file changed. Mutable-count
append-only erratum verified prefix16 raw/unique and decode12; earlier Main18
was shorthand error, not a bridge implementation fault or exported-gate change.

Main updates current MODULES/INTERFACE/HANDOFF overlays and TASKS/TIME_LOG under
lock. No worker required. At freeze all own jobs/writers PARKED. Independent
review/next native or memory-probe execution belongs to observer assignments.
