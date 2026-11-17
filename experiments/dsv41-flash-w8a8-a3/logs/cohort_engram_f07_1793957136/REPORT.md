# Engram F05/F06 and F07 candidate-cap cohort

Engram now gates the incoming layer stream in the original checkpoint basis
before HC pre and attention. A global rotation block is loaded and bound
through actual layer/full-forward/real-weight interfaces. The candidate pool
now uses the original2048-block cap throughout its owning compute/default
paths and integrated state contracts. Main and two disjoint workers implemented
and self-validated this cohort; independent sibling acceptance is still required.

## Engram equation, casts and resource

Authority is pinned official rc2
`8fde6d4e335fe300e2b949aab0e31659cce0811d`, the checkpoint config/rotation export
contract and its original Engram layer/common source. Source bytes are checked
against the preserved archive by worker evidence. The owning implementation now:

1. Projects BF16 lookup rows with BF16 wkv and rounds the projection to BF16,
   matching the original BF16 linear activation before gate FP32 arithmetic.
2. Restores only gate hidden in FP32 using repeated block32 `h_rot @ R.T`.
   Original-channel q*k and key norms/gate use that restored hidden.
3. Keeps V and residual in their already-rotated basis, adding `gate * v_rot`
   without a second value rotation. The selected external HC ABI is FP32, so
   the result stays FP32, matching original `.to(hidden.dtype)` for that ABI.
4. Uses the original signed clamped square-root/sigmoid rule at zero/tiny dot
   products; the prior sign-ratio approximation was not equivalent there.

`engram_rotation` is FP32[32,32] rank-local and[EP_SIZE,32,32] on distributed
host entries, shared across both Engram layers without a layer axis.
`WeightLoader.engram_rotation()` reads only the bounded optional rotation map,
validates the declared basis and identical diagonal32blocks/zero off-block
entries, caches the stored block, and fails explicitly on absent/unsupported
metadata. It loads no Engram tables or full model weights. The actual checkpoint
block SHA256 is68fef81121ad4d31c34d7659a684bff34d2fc1de23bc46201cf6cf35606fca1a;
max orthogonality error5.96e-8. Synthetic valid/cache/invalid shape/structure/
metadata/missing-map checks and a real map-only read pass.

Actual top-level loader arguments, generator rank materialization and rank
assembler retain the resource. CPU checks inspect the actual prefill/decode
entry contracts and assemble the real block exactly into[16,32,32], with model
weight reads stubbed to avoid full loads. Generator device-run bindings also
provide it. Other generator F11–F13 profile/dispatch/history/lifecycle defects
remain open; this does not claim successful generation or full argument parity.

Worker `/root/main_native/engram_basis` exclusively owned engram.py. Five
permanent defaults now cover randomT8, original/analytic sparseT1/T7, zero-dotT1
and strict original BF16-projection/FP32-residual randomT3. The compiled dynamic
operator is reused across shapes. Original layer/common AST execution and a
separate analytic reference agree. Additional original-source positives and
six numerical controls pass under task_20261106_174208_14992512691,device0,exit0.
Control failing lanes are: old114298/143360,missing restore114102/143360,wrong
transpose136184/143360,double value rotation139453/143360,FP32 projection
45810/61440,BF16 output112239/143360. These are numerical rejections,not compile
exceptions. Seven worker positives include the five permanent defaults plus
two original-reference executions; repeated designs are not claimed as unique.

## Before-attention ordering and actual integration

Main updates all six prefill layer assemblers,the decode assembler,both prefill
full-forward layer bodies,the decode encoder,owning references and real/fixture
bindings. Engram writes a separate explicit attention-input tensor before HC
pre and attention; non-Engram branches preserve their input through an exact
copy. This materialization gives the compiler concrete metadata across the
conditional branch and preserves decode's separate carry-buffer ownership.
Incoming pre_mix semantics and all accepted attention transforms are preserved.
The residual side of attention HC post also uses the updated stream.

The test oracle executes original rc2 Engram equations independently,then derives
the first hidden/norm/MLA/cache boundary. Published rows are compared directly,
without conditioning expected cache contents on an already-wrong device input.
All16 devices pass prefill/decode SWA and C2 prologs; C2 uses the actual rotation
map with synthetic model weights. Both frozen post-attention placement mutants
fail published window-cache validation numerically. These controls isolate order
after the basis fix,so a matching candidate golden cannot conceal the old order.

Two full staged layer checks also pass using the real rotation block and original
Engram golden: prefill SWA/Engram and decode C2/Engram. All staged outputs,
FFN/routing/residual hooks,caches and2048-wide pool storage pass. Permanent layer
defaults add `c2a_full_layer14_engram`; the existing layer1 Engram case remains.
The published window-row ordering gate is retained for Engram layer defaults.
These are two bounded layer cases,not the full ordinary layer suites or a
40-layer full-model device acceptance.

## F07 real2048 cap

Worker `/root/main_native/candidate_cap` exclusively owns config,indexer,four
full-source phases and two C1 reindex files. Config derives cap2048 from the
model; six phase constants/defaults and the single owning indexer entry/golden
use it. The obsolete64-only duplicate entry/argument/golden is removed.
Source-pool eligibility,restricted reindex consumption,width/tail and padding
contracts are preserved.

Actual shared publication defaults cover65/2048/2049/4097 blocks with exact
counts/IDs/-1 padding and forced latest low-score block behavior. An old64-cap
control numerically rejects count64 against expected65. Final worker device
task_20261106_173654_1116058823,device1,exit0 covers18 positives (indexer9 and
six affected phase entries/nine cases) plus that control. CPU provenance/shape/
AST/lint and compile-only checks pass. Shared layer/forward/generator contracts
now expose actual2048 storage; decode keeps its existing unified pool argument,
while prefill has family-specific arguments. This is not proof of the still-open
complete decode pool/state publication issue.

## Main checks, failures and exact scope

Main downstream task_20261106_193338_93787422054,device0,exit0 runs18 executables
and45 default numerical cases: all12 phase entries,MLA,o_proj,Engram,router,
compressor and indexer. Four order prologs plus two full layers give51 Main
positive executions; two post-order numerical controls are separate. Worker7
and18 positives/controls remain separately attributed and include repeat cases.

| Main queue task | Result |
| --- | --- |
| task_20261106_192720_8778103856 | First successful complete decode compile after metadata fixes,exit0 |
| task_20261106_193356_94261820924 | Fresh both-forward compiles,exits0,epochs1793964844–1793964903 |
| task_20261106_193157_91349222430 | Independent original prefill SWA first-cache prolog on16devices,exit0 |
| task_20261106_194242_106676619610 | Remaining3 positive prologs+2 post-order controls,16devices,epochs1793965367–1793965794,exit0 |
| task_20261106_194242_106687724879 | Full staged prefill SWA/decode C2 with real block,16devices,epochs1793965799–1793966071,exit0 |
| main/loader_queue.log | Synthetic rotation controls and real optional-map-only loader check,exit0 |
| main/binding_queue.log | Actual loader/rank carrier/contracts/assembler resource and2048 storage checks,exit0 |
| task_20261106_200221_125074919456 | Final16-file/45-source,counts,worker manifests,lint/syntax collector,exit0 |

All jobs use task-submit allocation; CPU/compile uses --no-device. Scripts
activate the selected environment and pinned PTOAS. Both workers and all Main
NPU jobs have released their allocations. Initial Main16-device work waited in
the queue while one-card checks occupied a device; queued time was not reported
as device execution. Measured worker wall epochs include intervening latency,
are not CPU time,and are imported once rather than summed as elapsed wall time.

Retained own-code failures include: conditional attention alias metadata loss,
a typed-alias retry,duplicate dimension names corrected to one existing tokens
dimension,a wrong old compile-runner path,an omitted staged-golden import,and
two binding-test assumptions about pool argument names/counts. The final binding
test inspects the actual declared pool arguments instead of inventing a new ABI.
Worker projection/restoration/control compile mistakes are preserved in worker
reports. No such error counts as a successful negative control or upstream bug.
Only documentation wording changed after successful compute validation; no
subsequent compute transformation was made. Final lint/syntax passes.

Final45-source after.json SHA256
7c5b8df2d57ca06a49106b528c2a018d36cd1524f4df8418e59ffa64b7d24d7f.
Exactly16 sources change:seven Main integration/loader files,engram.py and the
eight cap files. Other29 sources remain byte-identical,including accepted MLA,
o_proj,rope,paged cache,router,distributed exchange and absolute-coordinate
utility. F14 expert-fold behavior is preserved within the edited loader.
Worker manifests and per-source hashes,all output verdicts/negative errors,
generated rotation/attention-input bindings and commands are in unit_summary,
after.json,generated_evidence/excerpts and worker reports.

## Fresh independent reruns

Run the16-device groups before one-card sweeps to avoid holding one card while
a full-group allocation is waiting. Choose fresh absolute output directories:

```bash
task-submit --device auto --device-num 16 --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_engram_f07_1793957136/main/run_order.sh /absolute/fresh/order'
task-submit --device auto --device-num 16 --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_engram_f07_1793957136/main/run_layers.sh /absolute/fresh/layers'
task-submit --device auto --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_engram_f07_1793957136/run_downstream.sh /absolute/fresh/defaults'
task-submit --device auto --max-time 900 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_engram_f07_1793957136/worker_engram/run.sh /absolute/fresh/basis'
task-submit --device auto --max-time 1800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_engram_f07_1793957136/worker_cap/run.sh /absolute/fresh/cap'
```

Main ran the first SWA smoke separately,then ORDER_SKIP_SMOKE=1 for the remaining
suite. Independent reruns should leave it unset to run all six cases together.
Both workers expose fresh-output CPU helpers in their reports. Main loader and
binding helpers currently write beside their scripts; copy Main's helper files
to a fresh folder for CPU reruns (they resolve model paths from repository cwd),
or inspect the preserved CPU evidence. Do not overwrite frozen artifacts.
The accepted F02 compile runner accepts a fresh directory for both forwards.

## Limits and recovery

No original torch_npu primitive/reduction/quantization-tie parity,complete-model
checkpoint parity or real generation is claimed. Actual map loading is distinct
from real weight/table execution; no full checkpoint weights were loaded here.
F09 active-row/full-forward lifecycle,F11–F13 generation,offset incidence,
full ordinary layer suites,standalone TP,CI and full-model pool/state/device
gates remain OPEN/HELD. Generation stays HELD. No commits,external filing,
checkpoint/PyPTO writes,simulator acceptance or relaxed numerical gate.

Main updated INTERFACE/WEIGHT_MANIFEST actual resource/ordering contracts and
MODULES/HANDOFF implementation/recovery status. Both workers' unique records
were imported once under the shared lock. Final model source is frozen for
observer-owned sibling reviewer acceptance. Main/worker tests do not self-accept.
All unit-owned jobs/tool sessions/writers must stop at this checkpoint; resume
from TASKS/TIME_LOG,this report,after.json and worker reports only after the next
bounded assignment.
