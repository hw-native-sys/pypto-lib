# noEngram generation driver — implemented, native decode blocked

User decision10 permits temporarily disabling Engram for first generation.
This unit implements that explicit path; it does not establish original
full-Engram parity. No checkpoint weights or Engram tables were materialized.
The bounded functional milestone remains INCOMPLETE: native40-layer prefill
numerics pass, but native decode1 is blocked before numerical comparison by
halMemCtl/RunPreparationRefused code13. No continuous decode or text is claimed.

## Evidence correction carried forward

The prior cohort custom Main order/layer and worker basis/cap helpers omitted
execution_mode and generated a2a3sim. Their raw logs are preserved and their
ONBOARD claims RETRACTED. Only corrected reviewer explicitONBOARD16-device order
and two staged layers support bounded F06 acceptance. F05/F07 actualL2 defaults/
controls remain OPEN after native permission failures; the prior cohort is not
accepted wholesale. See native_accept_engram_1793967192/REVIEW.md. Historical
mode audit is independently owned; this unit does not infer that all previous
model math is invalid or that every old runner used the simulator.

Every new native helper explicitly sets execution_mode=False and checks
normalized ONBOARD plus generated platforma2a3; successful native evidence also
contains Callable images/device runtime logs. Allocation/exit0 alone is not
proof. CPU host/compile results are labeled separately.

## Implementation and ownership

Main exclusively edited generate.py and weight_loader.py. Reserved utils.py,
weight_map.py and structural_check.py remained unchanged. Worker
/root/main_native/fullentry_noengram exclusively edited prefill_fwd.py and
decode_fwd.py; its reserved moe_exchange.py/layer_reference.py remained unchanged.
No layer/config transfer,additional implementing worker,protected-source edit,
commit or external filing occurred. Before dispatch,TASKS recorded exact scope
and scalar ABI; all new artifact scripts live under this unit.

- Both forward entry/helper ABIs place runtime INT32 engram_enabled immediately
  after engram_rotation. Owning validation defaults1 preserve enabled behavior;
  --engram-disabled/first driver path supply0,branching around actual Engram
  compute. Inert lookup/projection/gate/rotation arguments remain ABI-bound.
  Main skips real Engram table and dense Engram matrix loading when disabled.
- Full-forward routed expert annotations omitted the24-local-expert axis despite
  fixtures/loader supplying it. Worker restored rank/host per-layer/local-expert
  dimensions for all six weight/scale arguments through actual forward bodies.
  Real driver contract is [16,40,24,...]; synthetic owning default reuses one
  expert layer over40 actual dense layers. Both layouts compile ONBOARD.
- Driver derives a separate capacity1 decode contract from prefill2/16/128.
  It binds both query profiles: unscaled base for SWA0/1,original YaRN compressed
  for2–39. Complete-entry profile overrides are rejected. Compressed source
  coordinates continue using the protected absolute-source helper.
- Each decode step refreshes request/position IDs,WINDOW slots/indices,C1/C2
  emission slots/lens,query and source RoPE,index/state tables. It consumes one
  current token,with one active token per rank. Metadata refresh preserves
  persistent cache object identities. State-cache batch capacity is checked as
  batch capacity,not incorrectly compared with logical page count.
- Handoff copies only six persistent WINDOW/compressed/index/compressor buffers.
  Query topk/pool,x/pre are initialized for the new query; prefillT query axes
  are not copied into decode1. Actual continued device state remains unvalidated.
- Concrete TensorSpec/ScalarSpec contracts validate full ordered dispatch without
  resolving dynamic dimensions from annotations. Pre-allocation preflight fills
  only weight placeholders with meta tensors; missing nonweight inputs reject.
  Actual binding is checked again at dispatch. Explicit disabled scope is emitted
  as model_scope=noEngram,original_model_parity=False. Checkpoint/model-root flags
  propagate through submit; actual device materialization uses requested path.

## Host and compile evidence

Main task_20261106_211853_207656517162 passed initial independent dual-profile
equations at nonidentity positions,missing-SWA numerical/ABI control,decode1,
real40x24 expert shapes and actual metadata progression126–129 with retained
cache sentinel identity. Subsequent preflight refinement rejects missing
nonweight inputs before allocation. Final host task213339_22998301198 passed
refined controls,header-only accounting,syntax and lint. Final
task_20261106_213600_231275812939 passed permanent --contract-report metadata
checks plus independent check_driver.py and scoped lint on final source.
Header-only budget touched274 safetensors headers and ZERO weight payload bytes.

main/check_driver.py checks original unscaled base and separate YaRN equations,
correct dual names,wrong/missing profile controls,capacity1,scalar handling,
local-expert/layer axes,per-step WINDOW/C2 parity/page progression and persistent
object identity. Main readiness generation_readiness_1793967196/probe_results
preserves negative-before old wrong-profile/hash behavior. No host result is a
device-state numerical proof. Permanent --contract-report now checks decode1,
required profiles and positions126–129 before printing memory estimates.

Worker task_20261106_211309_199942521675 compiles bounded decode1/prefill2;
task_20261106_211615_204403832262 compiles both actual40-expert-layer ABI entries.
Both explicitONBOARD/generateda2a3. See worker report for paths/hashes/commands.
Initial helper import/order errors and missing expert-axis compile failures are
preserved,not counted as negative numerical successes or upstream defects.

## Actual native evidence and blocker

task_20261106_211421_201926815403: auto16 devices0–15,native40-layer prefill2,
noEngram. Native runtime90.83s then owning composed/staged numerical validation
passes all compared outputs. Initial runtime PASS explicitly skipped validation;
only subsequent COMPLETE40_NOENGRAM_NUMERICAL_PASS establishes the numerical
result. Synthetic one-expert-layer shard is reused across40 distinct dense
layers; this is not the real checkpoint or valid-prefix continuous decode.

task_20261106_212340_215181910924: actual40-layer decode1 fails before numerical
comparison. FIRST error in worker_fwd/decode_native_queue.log: halMemCtl rc13
get_aicore_reg_info/get_aicore_regs(AIC_CTRL),then
init_aicore_register_addresses(Ctrl) failed13. Later hdcdisconnect507901 and
device fault507018 are aftermath. Final error:
RunPreparationRefused:prepare_native_run failed with code13(worker3).
All16 locks were released; no own NPU job remains. Similar preparation errors
were independently observed by reviewer. Owner/root cause remain unproved;
this is an environment/preparation blocker,not a proven model-math/compiler
defect. No reset,permission bypass,runtime edit,other-user-job action or false
numerical-control success was used. Original logs are immutable.

Worker prepared continuation.py for a genuine128-token prefix at0..127 followed
by three decode steps128..130 using ONE decode artifact and actual returned
state. It was NOT LAUNCHED after decode preparation failed. Thus state handoff,
multipledecode,cache overwrite/retention,parity/page crossing and stale-metadata
negative controls remain required. Same source coordinates passing a prior
phase test cannot substitute for this full-entry proof.

## Memory and next recovery

Real40-expert-layer input layout totals~635GB logical global bytes (~39.69GB
perrank under ideal rank sharding); conservative per-device input upper bound
is~124.9GB if ordinary nonresident global arguments replicate. That upper bound
is an uncertainty bound,not a measured allocation. Snapshot/EP/cache/compiler
transients and overlapping prefill/decode buffers must be included. Header-only
checkpoint storage:558396604416 expert bytes,17196936008 dense/top bytes,
221819890240 Engram bytes (omitted first path). Header storage is not runtime
peakHBM. Native peak/rank-residency mechanism and real fit remain UNKNOWN.
main/header_budget.json and driver_results.json retain full counts/assumptions.
No full-weight/generation job is authorized by these estimates alone.

After environment recovery,first rerun minimal explicitONBOARD preparation with
fresh outputs through task-submit,then worker decode1 numerical case. Only then
run prepared valid128prefix->3decode with all mode/runtime guards. Verify actual
outputs feed next step,retained cache regions,reference logits/state and stale
metadata negative control. Freeze for observer-owned independent acceptance.
GeneralF09 unequal/idle/multi-request same-artifact lifecycle,F05/F07 actual
controls,original Engram restoration/hash,F13,full checkpoint/TP/CI/offset
incidence,primitive/tie parity and generation remain OPEN/HELD. First path may
remain capacity1/noEngram without closing those broader contracts.

Main/workers stop at this blocked coherent checkpoint. TASKS/TIME_LOG and
after.json/artifacts.json provide exact hashes/recovery. No accepted coherent
text or real-checkpoint logits/tokens/HBM exist in this unit.
