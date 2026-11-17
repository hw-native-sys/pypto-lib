# Public admission configuration route — read-only result

There is NO supported admission-depth knob on the current PyPTO
RunConfig -> compile_entry -> DistributedCompiledProgram.prepare ->
DistributedWorker route. No private attribute mutation, environment override,
source patch or replacement runtime is proposed. The frozen45 model sources
remain equal to manifest ab1ced6d6ecbf9887297458703ba9b74e2c9b8bc57608370b4c2e0310e027e37.

## Exact public API distinction

| Layer | Result |
| --- | --- |
| simpler.Worker public constructor config | pipeline_depth=1, pending_run_depth=1, launch_depth=1 are recognized integer policies |
| PyPTO RunConfig / golden.normalize_config | No such fields; rejected as unexpected arguments |
| DistributedConfig | No such constructor fields |
| DistributedCompiledProgram.prepare / DistributedWorker | No such keyword parameters |
| simpler CallConfig / CallConfig.runtime_env | No such settable fields; AttributeError |
| PyPTO runtime _construct_worker | Forwards only level, device_ids, num_sub_workers, platform, runtime, enable_sdma and optional startup_timeout_s |

Queued CPU task105953_250971121513 PASS exercised all five candidate keys:
pipeline_depth, pending_run_depth, launch_depth, max_in_flight, max_inflight.
results.json preserves35 exact rejection messages. It also evaluated the
underlying source's pure depth validator for accepted1 /1-1-1 policies and
12 rejected bool/float/string/negative controls. That is a CPU source-validator
check, not a native Worker startup or an operational use of private control.
All45 source hashes match before/after; no compilation was needed because the
current public route rejects the arguments before compile/prepare.

The accepted underlying integer policies mean different things: pipeline_depth
requests native run-resource identity capacity; pending_run_depth bounds logical
admission FIFO occupancy; launch_depth bounds simultaneously launched logical
runs. They do not set ring byte sizes. An explicit policy disables automatic
endpoint-derived admission in the underlying source. Admitted capacity and
constructed/resident banks still require their own proof.

To expose these controls through the current PyPTO wrapper would need a new
public API and forwarding implementation. That is outside this read-only unit
and would require separately assigned upstream work; it is not a model fix and
was not implemented. Creating a standalone simpler.Worker is a different serving
integration, not an already supported shortcut through compiled.prepare.

## Actual frozen prefill call count

The exact frozen generated prefill HOST body has ONE _submit_chip site inside
its rank loop. _submit_chip forwards to one mutually exclusive
orch.submit_next_level branch (DFX directory or no directory). The actual
full40-layer rank computation is the single l2_prefill_fwd chip callable, with
its internal core tasks/layer loops, rather than40 independent native chip
invocations from the HOST level.

Queued CPU task110158_251543913684 PASS executed that SAME generated HOST body
with metadata-only torch tensors and fake ABI/domain submission sinks. It
recorded16 chip calls total and exactlyONE for each rank0..15. host_calls.json
records the source path/hash and trace. No model arithmetic, real argument
payloads, native submits, prewarm or device allocations occurred. This trace
checks HOST dispatch multiplicity; it does not establish actual admitted depth,
bank construction, memory residency, native completion or numeric correctness.

The frozen configuration has enable_chip_swimlane=0. This qualification matters:
PyPTO's onboard swimlane mode may execute a graph/timing pair, and that would
change invocation counts. The ordinary synchronous PreparedDispatch.dispatch
calls worker(*args, config=self.config); DistributedWorker.__call__ reaches a
blocking submit(...).result(). Source control alone does not prove that all
resource cleanup/cache banks have retired, so the reviewer still owns actual
bank/handle telemetry.

If the underlying depth1 policy were exposed, one complete prefill graph would
still submit its16 rank endpoints and preserve TP4/DP4/EP16 communication and
intra-graph AICPU/AICore parallelism. It would restrict overlapping outer logical
runs/program invocations, not serialize the16 ranks or alter40 layers, W8A8,
RoPE, or cache semantics. No native claim is made for an unexposed setting.
Importantly, depth1 cannot remove a single graph's large scratch allocation or
its internal ring-placement requirements.

## Disposition

The missing wrapper knob does NOT mean168 identity-capacity entries allocate
168 arena heaps. The reviewer independently audits actual admitted and
constructed state. A named identity directory, a grant, prewarm bank and live
execution bank are distinct quantities. This unit does not duplicate that audit
or approve a revised memory guard.

Observer/reviewer can use the exact one-chip-call-per-rank HOST evidence along
with pinned per-NativeRunHandle bank lifetime and actual startup/admission
telemetry to correct their own guard. Any resumed native trial remains under
observer assignment. Main has no NPU jobs or implementing workers and parks
with the model/runtime/harness frozen. This is API/source readiness, not fit,
continuation, generation or whole-model acceptance.
