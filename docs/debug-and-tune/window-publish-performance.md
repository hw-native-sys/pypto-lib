# Window publication: current C++ alignment

Measurements below describe the source after alignment to upstream/main, fetched on 2026-11-13. The remote remains at `8bbc8f5ca6ac3fd72f9004f509099083d5be10c3`; rebasing the PR onto it was a no-op. The earlier PR revision `504bd07d` is retained as the before-alignment comparator. No compiler or runtime source changes are included.

## Algorithm and data movement

Both C++ and Python use the following contract:

- BF16 source `[72,512]`, BF16 gamma `[512]`, FP32 cosine/sine `[72,32]`, int64 slots `[72]`, packed uint8 cache `[780,128,544]`. Each slot contains 512 FP8 payload bytes and 32 bytes of BF16 scales.
- RMSNorm over 512 values, eight segments accumulated per 64 lanes, pairwise reduction, epsilon `1e-6`, and BF16 normalization rounding. RoPE rotates the final 64 values as 32 interleaved pairs, with BF16 rounding after rotation.
- Group-32 absmax clamped to `1e-4`, multiplied by `1/448`, then rounded upward to a power of two through `(bits + 8388607) >> 23` and `exponent << 23`. The exact FP32 power-of-two scale is used for division and packed as BF16. FP8 conversion uses round R with saturation disabled.
- A 128-slot scan, negative-slot rejection, page ownership `(slot // 128) % workers`, and batches of eight accepted rows. Pending rows survive a scan boundary.
- Input runs consist of consecutive tokens accepted by the same worker, bounded by the scan and remaining batch capacity. Each run emits three 2D loads into pending rows. No token-index staging buffer or second gather loop remains.
- Output runs contain consecutive increasing destination slots only. Duplicate or descending slots split runs and retain source-order publication. Each run emits separate 2D payload and scale stores.
- RoPE storage is compact `[8,32]`, without the former upper-half zero initialization. The vector load matches the C++ row stride of 32 FP32 values; only its lower 32 lanes contribute to rotation. Full batches pass constant eight and tails pass the pending row count.

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
- Earlier PR revision 504bd07d measured 12-worker 41.55 us source versus 39.40 us after diagnostic tail-VV_ALL removal; its metadata-removal count was zero. Those values belong to the earlier image and must not be presented as results for this aligned source. No new VV_ALL/MTE3_S-removal diagnostic is measured here.
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
