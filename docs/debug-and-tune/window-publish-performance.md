# Window publication: current C++ alignment

The initial measurements below describe revision `963bd6aa` after alignment to upstream/main, fetched on 2026-11-13. The remote remains at `8bbc8f5ca6ac3fd72f9004f509099083d5be10c3`; rebasing the PR onto it was a no-op. The earlier PR revision `504bd07d` is retained as the before-alignment comparator. No compiler or runtime source changes are included.

## Algorithm and data movement

Both C++ and Python use the following contract:

- BF16 source `[72,512]`, BF16 gamma `[512]`, FP32 cosine/sine `[72,32]`, int64 slots `[72]`, packed uint8 cache `[780,128,544]`. Each slot contains 512 FP8 payload bytes and 32 bytes of BF16 scales.
- RMSNorm over 512 values, eight segments accumulated per 64 lanes, pairwise reduction, epsilon `1e-6`, and BF16 normalization rounding. RoPE rotates the final 64 values as 32 interleaved pairs, with BF16 rounding after rotation.
- Group-32 absmax clamped to `1e-4`, multiplied by `1/448`, then rounded upward to a power of two through `(bits + 8388607) >> 23` and `exponent << 23`. The exact FP32 power-of-two scale is used for division and packed as BF16. FP8 conversion uses round R with saturation disabled.
- A 128-slot scan, negative-slot rejection, page ownership `(slot // 128) % workers`, and batches of eight accepted rows. Pending rows survive a scan boundary.
- Input runs consist of consecutive tokens accepted by the same worker, bounded by the scan and remaining batch capacity. Each run emits three 2D loads into pending rows. No token-index staging buffer or second gather loop remains.
- Output runs contain consecutive increasing destination slots only. Duplicate or descending slots split runs and retain source-order publication. Each run emits separate 2D payload and scale stores.
- RoPE storage is compact `[8,32]`, without the former upper-half zero initialization. The vector load matches the C++ row stride of 32 FP32 values; only its lower 32 lanes contribute to rotation. At revision `963bd6aa`, full batches passed constant eight and tails passed the pending row count.

The source encodes dynamic transfer extents using `valid_shape`; generated C++ confirms dynamic row counts for the three TLOADs and both TSTOREs. The output loop uses explicit `pl.while_` state because a reduced natural-while inline example exposes a ConvertToSSA defect; this is recorded in the local upstream-issue log.

## Real-device protocol and results

Frozen CI-derived 72-token inputs and initial cache contents are replayed. Each worker-count comparison uses one allocated device, three groups of 10 warmups and 100 timed rounds, with reversed version order in group two. Compilation is offline through `task-submit --no-device`; device replay prohibits compilation and uses `task-submit --device auto`. `effective_us` is the standalone-wrapper device-domain orchestration/scheduler metric, not pure kernel time or full-model decode latency.

| Workers / device | Version | Group medians (us) | Median of medians (us) | Group P95 (us) |
|---|---|---|---:|---|
| 1 / 5 | baseline | 54.60, 52.25, 53.95 | 53.95 | 57.60, 56.40, 57.20 |
| 1 / 5 | previous | 67.95, 67.40, 67.35 | 67.40 | 71.50, 70.40, 70.20 |
| 1 / 5 | current | 47.10, 48.35, 46.25 | 47.10 | 50.10, 51.60, 49.20 |
| 12 / 7 | baseline | 36.20, 38.30, 36.75 | 36.75 | 39.60, 41.40, 39.70 |
| 12 / 7 | previous | 40.55, 41.20, 42.60 | 41.20 | 43.30, 44.70, 45.80 |
| 12 / 7 | current | 38.80, 39.30, 39.60 | 39.30 | 41.80, 42.80, 44.10 |

P95 is the nearest-rank 95th percentile of each 100-sample group. Raw samples and min/mean/median/P95/max are retained in the local metrics evidence. Cross-device absolute timings are not used to assign an optimization benefit.

| Workers | Before -> aligned source (us) | Change versus before | Aligned versus C++ |
|---|---:|---:|---:|
| 1 | 67.40 -> 47.10 | -30.1% | -12.7% |
| 12 | 41.20 -> 39.30 | -4.6% | +6.9% |

These are source-only results, with no generated-code barrier removal. No future latency target is claimed.

## Correctness and separate kernel captures

Whole-cache byte comparisons against the current C++ baseline pass all eight cases at both one and twelve workers: normal72, empty0, tail9, masked/duplicate slots, fragmented accepted runs with ownership changes, all-invalid slots, nonconsecutive destinations, and a separately compiled 136-token case crossing the 128-slot scan boundary. Unwritten cache bytes are included. New slot patterns obtain their expected outputs from the frozen C++ binary; existing cases reuse archived C++ outputs.

Workflow captures are separate from timed benchmarks. Both worker-count captures run in separate allocations after the original 300-second job limit interrupted capture; all three benchmark groups had already completed and are retained. The capture card is recorded separately. Each row below is a single capture of one or 12 physical kernel events, using explicit `kernel-duration-us`, excluding setup-inclusive Worker spans.

| Workers / version | Kernel-event median (us) | Kernel-event max (us) |
|---|---:|---:|
| 1/baseline | 32.658 | 32.658 |
| 1/current | 26.318 | 26.318 |
| 12/baseline | 6.868 | 7.088 |
| 12/current | 8.503 | 8.772 |

The single-capture event distribution is not the repeated benchmark distribution. Concurrent kernel sums do not give wall time. Full-model integration latency and new-version simulator pipeline timing are not measured in this update. Historical separate-cache simulation results do not describe this packed-cache image.

## Synchronization and remaining lowering differences

- [PTOAS #1754](https://github.com/hw-native-sys/PTOAS/issues/1754) tracks redundant VF VV_ALL. Full-row specialization and bounded dynamic tails remain in source; this update does not patch generated synchronization.
- [PTOAS #1756](https://github.com/hw-native-sys/PTOAS/issues/1756) tracks false metadata MTE3/Scalar dependencies. The current packed uint8 path avoids the earlier metadata handshake. Required DMA readiness, buffer reuse, ordered publication and final drain remain.
- Earlier PR revision 504bd07d measured 12-worker 41.55 us source versus 39.40 us after diagnostic tail-VV_ALL removal; its metadata-removal count was zero. Those values belong to the earlier image and must not be presented as results for this aligned source. The new simulator-only VV_ALL diagnostic is reported below.
- Python still packs scale words in registers and writes an aligned 256-byte vector; C++ uses ONEPT_B32 stores per pair. Python splits seven ordinary segments and the final RoPE segment; C++ branches inside an eight-segment loop. Mathematical stages and rounding are aligned on tested inputs, while instruction selection differs.
- Python uses compiler-generated waits, address/clamp operations and MTE3 barriers. Generated C++ currently places an MTE3 barrier before each payload and scale store; C++ places one after each output run. The compiler policy is not changed by this operator migration.

## Toolchain and evidence

PyPTO `9b00701173d80f4b1e0b356fbd326c574e6fe1a3`, PTOAS 0.72, CANN 9.3, A5/dav_3510, runtime `355ce3015e3e344d1fc3a61c206ee01823ce292c`, PTO ISA `5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`.

- Initial natural-while compile exposed an inline SSA defect; the explicit-state form compiles. The reduced reproducer fails before PTOAS.
- Successful compile/assembly tasks: `task_20261113_102042_15071279875`, `task_20261113_102126_155900311295`, `task_20261113_102215_161541311255`. The first assembly sweep encountered the preserved failed artifact; the corrected sweep selected only completed artifacts.
- Twelve-worker correctness and benchmark: `task_20261113_102259_16574158548`; completed benchmark samples precede its 300-second capture timeout.
- One-worker correctness/benchmark: `task_20261113_102459_17594103410`; its capture also hit the 300-second limit after all benchmark groups completed.
- Separate captures and cross-scan checks: twelve workers `task_20261113_102939_202151119007`, one worker `task_20261113_103204_214552026572`.
- Repository hooks, focused lint and QKV import check pass (`task_20261113_102508_176702610748`); the commit transaction repeats required checks.

Source SHA256:

- `models/deepseek_v4_1_flash/window_publish.py`: `d30728e9f6d784cc92d14119d23513cdb32b3667331f787c63df098df09249b8`.
- `models/deepseek_v4_1_flash/qkv_proj_rope.py`: `592678676fcd2f299c723ffb65aa9338283eaa3f6fcc84e70667acbf96456c49`.

## Worker scaling and tail specialization

Baseline: pypto-lib upstream/main 8bbc8f5ca6ac3fd72f9004f509099083d5be10c3, models/deepseek_v4_1_flash/kernels/norm_rope_publish.cpp (archived as replay/baseline.cpp). Before optimization: 963bd6aa, models/deepseek_v4_1_flash/window_publish.py. Same packed uint8 cache bytes, baseline ABI [780,128,544], Python ABI [99840,544].

The frozen 72-slot input has twelve runs of six consecutive slots on distinct page owners. A single worker runs nine full batches; twelve workers each scan 72 slots and run one six-row tail. The run order and numerical algorithm are unchanged.

| Simulator replay | AIV0 span (µs) | Main VF start (µs) | Main VF duration (µs) | Dynamic VF barriers |
|---|---:|---:|---:|---:|
| 1 C++ | 28.349 | 3.278 | 13.878 total / 9 | 0 |
| 1 Python before | 24.562 | 3.487 | 11.939 total / 9 | 0 |
| 12 C++ | 6.569 | 4.465 | 1.354 | 0 |
| 12 Python before | 8.911 | 4.746 | 3.462 | 61 |
| 12 Python diagnostic, VV_ALL removed | 6.709 | — | 1.444 | 12 |
| 12 Python source specialization | 6.918 | 4.792 | 1.198 | 0 |

The one-core replay sets logical worker 0 and logical block count to 1 or 12; it does not model 12-core contention. The raw trace anchor is the first positive-duration AIV0 event. All reported replay cache bytes match C++. Source specialization has four short post-compute VF events associated with branch-result tile moves; their intervals overlap, so they cannot be added to makespan.

12-worker C++ → Python before: first useful MTE2 1.083 → 2.092 µs; last useful MTE2 completion 3.597 → 3.925 µs; same five transfers, busy interval union 1.414 → 1.094 µs. First MTE3 5.823 → 8.211 µs, after VF completes at 5.819 → 8.208 µs. Two writebacks, busy union 0.624 → 0.567 µs. The later MTE3 launch follows the longer VF, rather than longer DMA service. Startup has a smaller independent cost: first compute VF is 0.281 µs later. After specialization the first MTE3 is 6.104 µs and last DMA completion is 6.782 µs. For one worker, the eight VF-to-VF intervals are 10.092 µs total in C++ and 8.080 µs in Python.

Original C++ postprocess (baseline.cpp:102,168):
```cpp
for (uint16_t row = 0; row < rows; ++row) {
    // RMSNorm, RoPE and FP8 conversion omitted.
    vsts(codes, (__ubuf__ uint32_t*)factors, row * 8 + segment, ONEPT_B32, full);
}
```
Original Python dynamic VF (vf/vf_1.mlir:54,118):
```mlir
// Outer loop: row < %arg1; inner loop: seven ordinary segments.
pto.vsts %115, %9[%80], %10 {dist = "PK4_B32"} : !pto.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mask<b32>
...
pto.mem_bar "VV_ALL"
```
At rows=6 there are 42 segment barriers, six row-end barriers, and one before the final scale-vector store; twelve directed barriers complete the count of 61. Payload UB is [1024,4096) over six rows; scale words are stored at [0,256); source BF16 is read from [6400,12544), weight from [5376,6400), and RoPE physical reads from cosine [14592,15488) and sine [15616,16512). Only the lower 32 RoPE lanes are consumed. The VF reads and writes are disjoint in this path; caller-level DMA dependencies remain. The independent VV_ALL-only diagnostic isolates about 94% of the simulator gap.

The source experiment dispatches constant row counts 2–7; one row keeps the dynamic fallback because a constant one-row VF exceeds PTOAS 0.72's layout frontier limit. Full batches still use constant eight. All 2–8 VFs lower without VV_ALL; the fallback retains the original barrier sites. Bounded dynamic loops and bounded row indices retained VV_ALL; the guarded fixed loop was rejected by VF codegen; segment unrolling hit the layout search limit. The raw simulator data includes the corresponding instruction PCs and transfer sizes.


The one-worker source specialization replay takes 24.713 µs versus 24.562 µs before; its nine compute VFs total 11.937 µs versus 11.939 µs. The full-batch path still executes zero VF memory barriers.

| Workers / device | C++ baseline (µs) | Python before (µs) | Python specialized (µs) |
|---|---:|---:|---:|
| 1 / 5 | 53.10 | 47.60 | 47.00 |
| 12 / 0 | 38.00 | 39.30 | 36.50 |

The same-device comparisons use three groups of 10 warmups and 100 timed rounds, with the order reversed in group two. The table reports the median of the three group medians for standalone `effective_us`. The 12-worker improvement is 2.80 µs (7.1%) versus the prior Python source. C++ and the specialized Python variant alternate in relative order across groups; the 1.50 µs difference between their median values is specific to this replay. All eight cases at each worker count match the entire C++ cache byte-for-byte, including untouched bytes. The 12-worker benchmark completed before a separate workflow-capture attempt failed during runtime initialization.

Task evidence: corrected baseline replays `task_20261113_105442_340851431053` and `task_20261113_105443_34097149529`; specialization replay `task_20261113_111037_687371964` and `task_20261113_111453_96008429010`; device validation `task_20261113_111304_84246422730` and `task_20261113_111301_83963430859`.

The separate 12-worker workflow capture on device 2 completed (task `task_20261113_111815_11604203433`). Across 12 kernel events, median/max `kernel-duration-us` were 7.059/7.183 µs for C++, 8.548/8.693 µs for Python before, and 6.716/6.952 µs after specialization. The one-worker optimized workflow capture was not collected after a runtime initialization failure; its repeated benchmark and simulator replay completed.
