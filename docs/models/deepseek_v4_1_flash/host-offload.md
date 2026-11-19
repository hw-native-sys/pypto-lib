# Engram Host offload measurements and interface limits

The integrated implementation uses one shared lookup and the existing
Prefill/Decode computation bodies. Current validation targets PyPTO
`864c8b86b2bae5c53b951d880b6794c5887cb0d2` on
`integration/urma-ccu-test`, pinned simpler
`390beebce6a8a3079e075a2e99315670eae4cff1`, PTOAS 0.72 and pinned
PTO ISA `8fb67fa7e5e02ae942b708c59f65d7b8380ae526`.
Earlier measurement snapshots below used the preceding pinned toolchain and
remain historical results. Eight-device hardware execution is deferred.

The forward ABI now takes hash IDs and Device scales. Each L2 forward allocates
its BF16 lookup result and two Device staging tensors internally, calls the
shared lookup inline, and composes the existing model computation. There is no
separate blocking Host lookup dispatch. The new plain-Tensor URMA endpoint
support removes Host staging from the remote path. This graph permits overlap;
its realized whole-model overlap still requires hardware profiling.

## Current integration validation

The updated branch compiles and assembles the actual integrated Prefill and
Decode entries for Sliced ENP and Full-Vocab Local on both four and eight ranks
(eight combinations). Prefill uses capacity 8192; Decode retains its dynamic
physical-token and flattened-lookup dimensions. These are offline checks.
CPU checks cover checkpoint shard reconstruction, padding, exact scale uploads,
and uniform useful-payload accounting for 72 and 8192 tokens/rank.

A supplied full checkpoint also passes bounded CPU preflight: Engram tensor
layouts, the complete compressed-token map and hash parameters, and 8192-token
Prefill plus 72-token continuation/rollback hashes agree exactly with its official
inference implementation. Selected real FP8/scale rows, shard boundaries and
padding match direct checkpoint reads. Prefill, chunk continuation and Decode
input preparation now supplies the computed flattened IDs directly. These checks
do not allocate complete Host windows or establish device numerical acceptance.

The unmodified pinned runtime rejects four-card execution before the first
lookup at its new Host-URMA capability precheck. On this Ascend950DT environment, direct
ACL queries return success with AIV Host-map capability 1 and UDMA capability 0.
The active CANN root is 9.3.0, with compiler/OPP package metadata reporting 9.2.0.
Both lookup strategies fail domain creation because the current PyPTO Host-domain
lowering requests an URMA workspace even for local-only Full-Vocab reads. This
is a separate boundary from the deferred intermittent scheduler timeout in
PyPTO #3465. It does not establish whether the cause is hardware, the local
CANN/driver combination, or the capability contract.

A user-authorized local runtime experiment adds an opt-in bypass named
`SIMPLER_EXPERIMENTAL_IGNORE_HOST_UDMA_CAPABILITY=1`. The default rejection
remains enabled; registration, mapping, transfers and numerical checks still run.
This switch exists only in the locally patched runtime, not in the upstream pin.
With that patch, ENP4 and Full-Vocab Local each pass four changing-input exact
BF16 checks at both 72 and 8192 tokens/rank: 16 successful dispatches in total.
The tables contain 3700 synthetic rows per layer. Useful payload is 864 KiB
and 96 MiB per rank respectively. These checks validate the direct-Device
staging lookup, not complete-table registration, model accuracy, performance,
or resolution of the intermittent timeout.

With the same diagnostic patch, four-card ENP4 also completes a real-checkpoint
generation run using all 40 layers, capacity-16 chunked Prefill and one-token
continuation through the same Prefill program. The registered payload totals
196,616,542,208 bytes, including per-layer/rank padding; each Device retains
6,144,266,944 scale bytes. Four prompts generate two tokens each, including
`5`, `Paris` and the Chinese greeting for "hello" for the first three prompts. Logits are finite and
Device greedy outputs match copied-logit argmax. This is a generation smoke
check, not a whole-model reference comparison or the requested 8192/72-token
end-to-end benchmark.

A second four-card run completes the switch from chunked Prefill to the
independent integrated Decode program, sharing resident weights, caches and
Device scales. Decode uses one active token per rank in a 192-row physical
allocation. Both generated token IDs match the Prefill-only continuation for
all four prompts. This checks greedy-output agreement, not complete-logit
numerical equivalence. Both programs retain their own Host windows, totaling
393,233,084,416 payload bytes. Normal shutdown completes successfully. The first
dispatch of each program includes lazy Host-window initialization; neither cold
wall time is a steady-state fetch or Decode measurement.

Checkpoint generation exposes `--ring-heap-mib` to size each of the four
runtime output heaps. The default is 4096 MiB per ring. EP4 checkpoint weights
occupy approximately 75.37 GiB per Device and full Engram scales another
5.72 GiB. A four-card attempt with the default heaps runs out of HBM while
allocating weights, before the first forward. The successful capacity-16 run
uses 1024 MiB per ring. Heap sizing
must also accommodate the forward's actual scratch peak; allocation success
alone does not validate execution at that size.

## Prepare and run

Activate the matching PyPTO/runtime, CANN and PTOAS environment. CPU preparation
creates the table, hashes, frozen Torch reference outputs and assembled device
artifacts before an NPU allocation. Run from the repository root:

```bash
python models/deepseek_v4_1_flash/engram_host_offload.py --compile-only --ranks 4 --tokens 72 --strategy sliced_enp --artifact build_output/engram-enp4-72
task-submit --device auto --device-num 4 --max-time 180 \
  'python models/deepseek_v4_1_flash/engram_host_offload.py --artifact build_output/engram-enp4-72 --devices "$TASK_DEVICE"'
task-submit --device auto --device-num 4 --max-time 300 \
  'python models/deepseek_v4_1_flash/engram_host_offload.py --artifact build_output/engram-enp4-72 --devices "$TASK_DEVICE" --benchmark --rounds 100 --warmup 5'
```

The queued process must inherit the selected environment. Use separate artifact
directories for `full_vocab_local` and for 8192 tokens. `--ranks 8` prepares
ENP8 or eight-rank Full-Vocab without allocating NPUs. Each deployment uses one
ENP group; two ENP4 groups on eight devices are outside the interface.

`--table-rows` controls the synthetic rows per layer, rounded up to the rank
count, and must cover the small correctness hash buckets. Larger tables also
increase preparation, Host initialization and Device scale memory. A small,
warm synthetic table cannot determine the released checkpoint's performance.
The benchmark cases use fixed seeded random IDs spanning the complete prepared
table. Default correctness cases additionally cover changing request histories,
rollback, inactive ranks, duplicates, shard boundaries and staging tails.

Missing or stale artifacts fail execution. Preparation is never silently run
inside a device allocation. A failed dispatch or comparison makes the entry
fail and prevents a successful benchmark result from being written. Retain the
queue log and `benchmark.strace`, including failure diagnostics.

## Metrics and contention

`benchmark-results.json` preserves the source identities, actual device order,
table extent, sample count, warmups and raw samples. Every measured mode uses
the same retained allocation and one compiled dynamic lookup:

| Field | Meaning |
| --- | --- |
| Rank P50/P99 | Device effective orchestration/scheduler lookup window, including GET completion, local/staged MTE, Device scale reads, dequantization and output stores |
| Effective GB/s | Useful FP8 bytes for that rank divided by its median lookup time; decimal GB/s |
| Fastest-rank median | Smallest rank median for kernel tuning, retaining every other rank's distribution |
| Group maximum | Per-round maximum of the rank effective windows; includes rank start skew within their own windows and is not a synchronized multi-device interval |
| Host submit-to-complete | Whole blocking lookup call, including Host dispatch; explicit output copy-back is outside this interval |
| Process-tree PSS | Proportional Host memory summed across the parent and children; shared pages are not counted in full for each process |

P99 uses nearest-rank selection and requires at least 100 samples. Timing uses
the existing PyPTO STRACE parser; missing device samples or an unknown
PID-to-device mapping fail the report. Registration/setup and explicit result
copy-back are outside the measured lookup windows. The benchmark checks outputs
after the first and final dispatch of every mode with the common golden
comparator at zero tolerance. Correctness validation also exercises changing
inputs separately.

For sliced ENP, the local-only and remote-only cases mask complementary rows of
the same mixed input with negative IDs. Each path's actual row requests and
payload bytes therefore match between isolation and mixed execution. Inactive
rows still execute output zero stores; their payload contribution is zero.
The mixed graph permits overlap between a batch's consumption and another
batch's remote GET. Capture a separate correctness run with
`--enable-chip-swimlane 4` to establish actual overlap and staging dependencies.
Benchmark mode rejects this flag because graph instrumentation perturbs timing.

`mixed_over_sum_isolated` compares mixed completion with the sum of the two
isolated completion medians. It includes scheduler, zero-store and computation
costs and does not isolate link contention or identify an individual path's
slowdown. Separate concurrent local/remote completion spans, physical Host
bandwidth, Unified Bus traffic and Host–Device utilization require additional
platform profiling; their fields remain unavailable instead of being inferred
from useful payload bytes. No cold-cache or cache-flush behavior is assumed.

## What the selected interfaces can express

| Requirement | Evidence and remaining boundary |
| --- | --- |
| Local registered Host reads | `alloc_window_buffer(memory_source="host")` and MTE access are expressible; Full-Vocab lookup uses no peer GET |
| Sliced ENP4/ENP8 | One row-sharded domain with explicit rank ownership; both rank counts compile and assemble |
| Asynchronous remote completion | `tensor.get_async(..., engine=URMA)` produces an InOut Device staging tensor; a dependent In consumer waits for deferred completion |
| Bounded staging reuse | Two distinct Device staging tensors and explicit consumer TaskIds express WAR protection; 32 GETs per task bound the request count |
| Shared physical Full-Vocab | Missing public L3 external-storage binding: `window_sources` copies initialization bytes into independent rank windows |
| Direct remote Host-to-HBM GET | Supported by the updated test branch; the window-bound Host source transfers into an ordinary local Device tensor |
| Overlap with the whole model | Lookup and the existing Prefill/Decode body share one L2 graph; realized overlap requires a hardware trace |
| Concurrent path timing | The scheduler can express independent work, but the combined consume kernel does not expose separate local/remote completion boundaries. Current isolated/mixed samples do not satisfy per-path concurrent slowdown attribution |

Passing the same tensor once per rank to `window_sources`, or listing it in
`inherited_host_tensors`, does not share the registered communication windows.
The latter supports inherited Host transfer buffers, not external domain
storage binding. The earlier native `_map_host_ddr` experiment with one shared
backing passed four devices at 1 MiB, 64 MiB and 1 GiB. It uses a private API and
does not provide a production L3 storage-ownership contract.

The synchronous `tile.get` permits a plain local Tensor destination, but it
uses the MTE peer-address path. Host domains expose remote memory-region
addresses through the URMA workspace, not ordinary peer device mappings. This
API is therefore not an established replacement for the Host URMA fetch.

A public shared-binding contract needs caller-owned Host storage, its byte
range and alignment, per-device mapped addresses, registration handles, and
retention through deferred requests. Unregistration must wait for every
consumer and imported peer mapping to retire. Pointer equality across devices
is not evidence of shared physical pages. No private mapper or runtime patch is
used by the production controller.

PyPTO #3482 now confirms that the row-sharded approach is a supported way to
meet the current requirement: each rank initializes one Host shard and remote
rows use the asynchronous Host-to-local-HBM GET path. The async GET item has
been merged into the integration test branch. The remaining request in that
issue is shared external backing for Full-Vocab Local.

The integration baseline uses Sliced ENP: `EngramHostTables` streams one shard
per rank per compiled program and uploads scales once to Device. The integrated Prefill/Decode
entries perform lookup inside their L2 graph and reuse the existing computation
bodies. The historical separate-lookup benchmark remains available for isolated
lookup validation; checkpoint generation no longer invokes that entry.

### Prefill/Decode Host-window ownership

Persistent domains are namespaced by compiled program in the selected PyPTO
runtime. A worker dispatching both integrated Prefill and Decode retains two
Host payload allocations, even when both use the same `engram_payload` init key
and source objects. `window_sources` does not alias those destination allocations.
No public cross-program window binding or per-program domain release is exposed.
The generation controller budgets both program allocations when a separate
Decode artifact is supplied and reports this count explicitly. Single-program
continuation through the Prefill composition retains one allocation. Device
scales, model weights and KV caches remain shared across both programs.

### Shared inline composition

A single L2 entry can call the existing Host lookup inline, independent
RMSNorm, and a consumer of both results. Four-rank ENP4 and Full-Vocab probes
each pass four changing-input cases at 72 tokens/rank with exact BF16
comparison. Separate dependency and level-4 timing passes verify that both
producers reach the consumer, neither producer depends on the other, and
lookup tasks actually overlap RMSNorm. In the final captured case, overlap of
their core execution intervals is 102.6–103.5 microseconds for Full-Vocab and
93.8–109.9 microseconds for ENP4 across ranks. These are small-table composition
checks, not end-to-end speedup or URMA wire-transfer overlap measurements.

The existing complete Prefill and Decode bodies also compile and assemble
inside a shared inline composition for ENP4, ENP8 and Full-Vocab at both rank
counts. Prefill uses capacity 8192; Decode retains its dynamic token dimension.
The experiment reuses each forward body and adds no alternate computation
implementation. Full-model runtime correctness and performance remain pending;
eight-rank checks are offline only.

The remaining integration boundaries are explicit:

- Bind Decode's flattened index length as its own dynamic dimension and check
  that it equals `2 * physical_tokens * 24` before dispatch. A product of a
  dynamic dimension is not itself a signature dimension accepted by the
  signature-only compile entry.
- Pass the returned lookup tensor, reshaped to the forward layout, into its
  consumers. Keep reusable staging protected by consumer-to-next-writer TaskIds;
  putting calls in an inline function does not supply WAR dependencies.
- The current fetch tasks declare the entire BF16 lookup output as InOut.
  TensorMap therefore chains those writers, and an Engram consumer waits for
  the final writer. Layer or row streaming needs narrower producer/consumer
  regions and staging lifetime protection in the shared graph.
- A lookup loop emitted before all model work can delay submission of that
  work when runtime task capacity fills. Shared inline permits concurrency;
  fine scheduling and its benefit still require full-model measurement.

The generation controller now uses this composition through the integrated forward ABI.

## Registration, address space and memory

In the selected simpler source, `Orchestrator.allocate_domain` requires each
buffer and the covering window to be a multiple of 64 bytes. The A5 Host
allocator rounds the whole window to the system page size, creates a unique
POSIX shared mapping for each rank, calls
`aclrtHostRegisterV2(..., ACL_HOST_REG_MAPPED)`, obtains the local device address
with `aclrtHostGetDevicePointer`, and registers the URMA workspace when needed.
Peer memory-region addresses are resolved through URMA and must not be used as
ordinary local device pointers. Each owning context retains its mapping until
communication cleanup is confirmed.

No full-table registration size/count limit, per-device virtual-address ceiling
or maximum simultaneous shared mapping count has been established. The 1 GiB
experiment is a tested size, not a hardware limit. There is no full-capacity
950DT compatibility claim. Record NUMA page placement, NPU topology and the
actual loaded runtime/HCOMM libraries before attributing a bottleneck to the
interconnect. Compiler/OPP metadata on the experiment host says 9.2.0; its SDK
directory name alone does not establish the loaded component versions.

The released tables contain 768,022,850 rows. Before small padding/alignment,
they need approximately 183.1 GiB of FP8 payload and 5.72 GiB of Device scales
per consuming rank. Sliced aggregate Host payload is approximately 183.1 GiB.
The implemented Full-Vocab replication needs approximately 732.4 GiB on four
ranks or 1464.9 GiB on eight. A shared physical backing would need approximately
183.1 GiB. These are storage estimates, not measured registration limits.
The controller checks available RAM and rejects an infeasible replicated
Full-Vocab allocation without changing strategy. It streams checkpoint slices
and padding into registration-owned storage and uses bounded shared scratch for
the one-time scale upload.

For real-checkpoint end-to-end measurements, the existing generation report
contains raw phase/tokens-per-rank samples, metadata/upload time, exposed lookup
wait, forward wall time, output copy-back/greedy checks and their sum. Setup is
excluded from those per-forward samples. The first token sampled from Prefill
is not counted as a Decode forward. These Host wall intervals differ from the
lookup's device effective window. The single-request generation entry does not
itself implement the 12-request × 6-position DSpark workload; its packed input
preparation supports the corresponding hash layout. Full checkpoint numerical
acceptance and representative end-to-end comparisons remain required.

The intermittent S3 Host-window timeout is tracked in
[PyPTO #3465](https://github.com/hw-native-sys/pypto/issues/3465). Continuing
implementation while it is deferred does not make failed dispatches or partial
samples successful. The [feature issue](https://github.com/hw-native-sys/pypto-lib/issues/1830)
tracks hardware and model acceptance separately.

## Validation snapshot, 2026-11-17

Four-rank synthetic ENP4 and Full-Vocab pass four changing-input exact BF16
checks at 72 tokens/rank. Full-Vocab also passes all four at 8192 tokens/rank.
The revised ENP4 pipeline and 8192-token lookup still encounter S3 timeouts;
its matched benchmark finishes local-only samples but fails in remote-only.
There is no accepted comparison of the integrated lookup implementations.

Full-Vocab's unprofiled resident benchmarks use 100 samples after 5 warmups:

| Tokens/rank | Devices | Fastest-rank median | Group maximum P50 / P99 |
| --- | --- | --- | --- |
| 72 | 0,1,2,3 | 8.197 ms | 8.285 / 8.334 ms |
| 8192 | 1,2,3,5 | 911.954 ms | 984.137 / 1022.878 ms |

Both use only 3700 padded rows/layer, retained warm tables and frozen independent
goldens; first/final output checks pass at zero tolerance. Per-rank effective
useful-byte rates are 0.1071–0.1079 GB/s at 72 tokens and 0.1023–0.1104 GB/s at
8192. These include the lookup computation and scheduling boundary above and
are not measurements of physical Host bandwidth or a released-table bottleneck.

A separate four-device Full-Vocab profiling run completes exact checks and
exports UB, HBM, LLC, NPU memory and Host memory CSVs. UB entries are all zero
and only identify port 0; no separate PCIe/CCU/SIO CSV is exported. Counter
coverage and attribution are unverified, so zeros do not establish zero
physical traffic. Captured system counters include setup and teardown. The
profiled timing is kept separate from the unprofiled table above.

The complete four-rank 8192-capacity Prefill forward compiles and assembles;
real-checkpoint forward numerical and end-to-end performance acceptance remain
pending. Eight-device hardware was not run.

## Direct simpler transport check, 2026-11-18

A separate four-rank benchmark bypasses PyPTO imports and code generation with
handwritten C++ AICPU orchestration and AIV kernels. Both paths pass exact
byte comparisons before and after 100 samples following 5 warmups. The
synthetic table is 1 GiB; each scenario runs both strategies sequentially on
the same allocated devices with the same frozen indices and goldens.

| Tokens/rank | Full-Vocab median | ENP4 local-only median | ENP4 remote-only median | ENP4 mixed median |
| ---: | ---: | ---: | ---: | ---: |
| 72 | 0.230 ms | 0.221 ms | 6.861 ms | 6.913 ms |
| 8192 | 20.145 ms | 20.562 ms | 759.897 ms | 765.229 ms |

Numbers are fastest-rank medians of device wall time. Useful payload remains
864 KiB or 96 MiB/rank, with exact 25% local and 75% remote ownership for ENP4.
Full-Vocab reads the entire payload locally. The native ENP pipeline issues
up to 32 individual 256-byte GETs per task through two 8-KiB Host staging
slots. Local and remote consumers have independent outputs; narrow views and
explicit staging WAR dependencies express their concurrency. AUTO scopes
cover 2048 requests each to bound task retention at Prefill sizes.

This baseline favors Full-Vocab fetch latency, while ENP mixed latency is
close to remote-only latency. These totals do not establish each concurrent
path's slowdown, physical Host bandwidth or URMA peak throughput. The task
geometry, small requests and bounded pipeline remain implementation choices.
The benchmark replays indices, so even the larger table does not guarantee
cache misses. It omits model hashing, scale reads, dequantization and model
computation; its timings cannot be compared directly with the integrated
BF16-output lookup numbers above or interpreted as end-to-end speedups.

Full-Vocab uses four independent Host windows, totaling 4 GiB of payload
allocation; ENP4 allocates 1 GiB across its shards. Simpler's public
`create_buffer` / `TensorTransfer.MAP` contract supports owned direct L2
workers, and does not supply a shared-backing L3 multi-device contract.
No eight-device hardware was run, and the earlier intermittent S3 issue
remains unresolved.

## Private shared-backing Full-Vocab check, 2026-11-18

The public L3 allocation was followed by a separate pure-simpler transport
check. Four independent L2 processes opened one POSIX shared backing, mapped it
with the native `_map_host_ddr` helper, and passed the resulting per-device
address to the same frozen callable. The table is 1 GiB and the physical
payload allocation is one copy. Decode (72 tokens/rank) and Prefill (8192
tokens/rank) both passed exact byte comparisons on every rank for 100 samples
after 5 warmups. A same-path baseline used four independent `create_buffer`
allocations and `TensorTransfer.MAP`; it also passed exact checks.

| Tokens/rank | Shared backing: fastest / slowest rank P50 | Replicated L2: fastest / slowest rank P50 | Shared physical copies |
| ---: | ---: | ---: | ---: |
| 72 | 4.925 / 9.436 ms | 4.445 / 4.984 ms | 1 |
| 8192 | 26.215 / 26.891 ms | 26.120 / 26.730 ms | 1 |

These are independent L2 submit timings and are not comparable with the L3
orchestration timings above. The 72-token runs used different allocated card
sets, so their rank spread is descriptive rather than an isolated hardware
penalty. The result establishes that simpler's lower-level mapper can support
shared physical Host storage; it does not establish a public L3 ownership
contract or the registration limit for the approximately 183 GiB production
table. The private mapper also requires explicit per-device mapping lifetime
management and must not be used as a production API dependency.
