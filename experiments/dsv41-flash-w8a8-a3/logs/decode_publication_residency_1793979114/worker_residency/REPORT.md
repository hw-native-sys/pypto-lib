# Prepared residency worker — final disposition

BLOCKED queue timeout before device allocation; native reuse/peak UNVALIDATED. Earlier pending checkpoints below are historical.

Exclusive ownership: generate.py,weight_loader.py and worker_residency/.
Only generate.py changed; weight_loader.py remains unchanged. No Main
decode/prefill/operator,shared ledger/runtime/harness/checkpoint edits,full weights,
extra agents,commits,filing,admin/reset or permission bypass actions.

The prior direct compiled(*plainTorchArgs) path did not invoke the resident
mechanism. TensorSpec resident metadata used only for compile could not upload
or retain buffers. This repair uses the supported prepared DistributedWorker
API; PLAN.md cites actual runtime/harness ownership contracts. It does not
claim private runtime changes or infer every ordinary input was replicated.

PreparedDispatch normalizes/requires ONBOARD,establishes shared CPU IO before
prepare,then opens one nonpersistent-protocol prepared worker per phase. Static
leading-rank resources upload once via alloc_stacked_tensor; handles replace
host weights on each actual rt call. All dense/expert/source/top/inert Engram
weight resources use the same supported leading-rank mechanism. The actual
driver opens a prefill owner,dispatches repeats,closes it before opening decode,
then retains decode handles across generated steps. All resident handles are
freed before context close,including failure cleanup. No owner/handle crosses
phase boundaries. Old misleading spec-only upload prose was corrected.

Per-call IO (metadata,query scratch,mutable caches and outputs) uses fixed
shared-memory tensors inherited by the prepared worker. Replacement metadata
gets copied into those mappings. Returned outputs are those actual shared
tensors. Cache/state is still H2D/D2H per dispatch; this is NOT device-resident
KV state. persistent=False recreates collective protocol domains each call
while static allocation handles remain live. No retained signal-window reset
assumption is silently introduced.

CPU task_20261106_233921_327837616760 passes a tiny2-rank ONBOARD compile with
generated platforma2a3 and scoped lint. Initial task233834_32708847949 failed
because my proof imported pypto.distributed rather than the owning
pypto.language.distributed; failure preserved in compile_attempt1.log and fixed
without model/runtime changes. CPU task_20261106_234536_34107581219 passes exact
class controls using a fake public owner:shared IO before prepare,one static
allocation despite host-weight mutation,new metadata copied into stable shared
IO,cumulative state and final free. It executes no device work and is not
hardware acceptance. It also passes generate.py syntax/lint.

Main authorized one tiny2-device proof after its full16 baseline had waited
for other allocations. Native task_20261106_234723_349266526109,max-time180,
timed out at600s and was auto-cancelled BEFORE allocation. The failure log is
native_attempt1_queue_timeout.log; it is not a runtime/numerical failure.
Main authorized one resubmission with queue timeout3600/max-time180:
task_20261106_235855_377411318121 is PENDING. No allocation or native numerical
verdict/peak measurement yet. proof.py uses exact g.PreparedDispatch and
g.dispatch with512bytes of static weights,three calls,new IO allocations,
host-weight-mutation control,exact cumulative output checks,normalized ONBOARD,
generateda2a3 and HbmSampler. Successful native results will be recorded in
proof_results.json; absence of that file means no native success is claimed.
Upload-byte counters describe requested supported-API payload sizes,not a bus
traffic/bandwidth measurement. Native host-weight mutation and retained-handle
checks corroborate reuse if execution passes. Peak samples are absolute device
HBM usage;the512-byte delta may be below npu-smi resolution.

Main additionally requested a narrow handoff repair within generate.py: a
prefill fixture can have more allocated pages than decode even when its used
prefix fits. Handoff now validates used raw WINDOW and C1/C2 compressed/index
rows against BOTH source and target capacity,then permits trimming only their
page axes under the validated used-pages bound. Nonpage dimensions and state
batch shrink remain strict. check_handoff.py passes exact retained-prefix copy,
used WINDOW overflow rejection and unknown state-batch truncation rejection;
scoped lint also passes. This is a CPU contract proof,not device continuation.

The HBM sampler also shadowed threading.Thread._stop with an Event,which could
make join fail after execution. It now uses _stop_event. CPU owner/handoff/
sampler lifecycle and lint pass task_20261107_001006_407935031969. A queued
CPU-only npu-smi format read verifies the physical-ID/HBM parser,not this
worker's peak usage. The final tiny proof includes a changing runtime scalar
after prepare; its explicitONBOARD/a2a3 compile passes as well.

The memory plan now derives static rank bytes from the implemented leading-rank
allocations,with conservative global nonresident IO bytes and explicit unknown
compiler/EP/cache/snapshot/transient overlap. Tiny proof is not a full-model
memory-fit proof. No real weights or risky124GB allocation was attempted.
Concrete host overlap also keeps real weights HELD: materialize_all_ranks
retains rank_args/expert_by_rank (~634GB global),then
prefill_weight_args/_assemble_rank_weights torch.stack creates another global
set (~634GB),and prepared uploads can add temporary staging. This can exceed
1.5TiB host memory despite an ideal per-rank HBM fit. No unbounded materializer
rewrite or large allocation was attempted in this unit.
The driver now raises host-materialization-overlap even when
--allow-full-materialize is supplied. CPU spy task_20261107_003914_64390413673
proves both default and opt-in paths reject before tokenizer/full materializer;
owner/handoff/sampler controls and lint also pass on that source. The owner is
wired into future driver dispatch but real device_run remains held at this gate.
Streaming materialization is a separate remaining first-generation requirement,
not established by the tiny owner test. A source-grounded future route uses
rt.alloc_tensor for one worker shard at a time and public StackedDeviceTensor
aggregation,as golden._alloc_empty_stacked_tensor already does,avoiding global
torch.stack;this is a proposal,not implemented/proven streaming loading.
First path stays noEngram; original hash/Engram restoration and general F09
are unchanged/open. Main independently owns decode publication/state repair
and coherent milestone freeze; independent sibling acceptance remains required.

## Final frozen worker disposition

BLOCKED queue timeout before device allocation; native reuse/peak UNVALIDATED.
All worker jobs ended; source writes stopped; observer-owned independent acceptance still required.
