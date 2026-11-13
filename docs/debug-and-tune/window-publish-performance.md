# Window publication: PyPTO migration performance

This report describes a migration candidate, not a performance improvement over the current C++ kernel. Measurements are from real A5 hardware on 2026-11-13. The PR preserves the current packed cache and power-of-two scale encoding.

## Baseline and scope

The baseline is `models/deepseek_v4_1_flash/kernels/norm_rope_publish.cpp` from pypto-lib `8bbc8f5ca6ac3fd72f9004f509099083d5be10c3`. It already batches contiguous source rows and contiguous destination runs. Earlier separate-payload/separate-scale measurements use an older implementation and must not be treated as the baseline for this PR.

The current operator uses the Python implementation in `models/deepseek_v4_1_flash/window_publish.py`. Its caller retains page ownership and the existing task dependency. Cache storage is uint8 `[780,128,544]`: 512 FP8 payload bytes followed by 32 bytes holding 16 BF16 power-of-two scales per slot. Inputs are BF16 `[72,512]`, BF16 weight `[512]`, FP32 cosine/sine `[72,32]`, and int64 slots `[72]`. The normal workload executes 72 valid tokens; one-worker and 12-worker wrappers are measured separately.

The frozen input tensors were replayed, including the initial cache contents. Correctness compares every output cache byte against the exact current C++ result, including unmodified slots. Tests cover normal72, empty0, tail9, masked/duplicate slots, and a separately compiled136-token case crossing the128-slot scan boundary. Each case passes at1 and12 workers. The diagnostic passes normal/empty/tail/duplicate cases; cross-scan coverage belongs to the unmodified source candidate.

## Measurement protocol

Compilation and assembly run offline through `task-submit --no-device`. Device tasks use `task-submit --device auto` and the allocated `TASK_DEVICE`; compilation is explicitly forbidden in the device replay. Each comparison uses one device allocation, three groups with10 warmups and100 timed rounds per group, alternates version order, and retains all100 samples. Benchmark tracing is disabled. Tables report the median of the three group medians; P95 is the nearest-rank95th percentile.

`effective_us` is the golden harness device-domain orchestration/scheduler metric for this standalone wrapper. It excludes host setup, compilation, first loading and warmup; it is not pure kernel time or a full-model decode latency. Workflow captures run separately with chip-swimlane level4 and use the explicit `kernel-duration-us` field. Worker bars include setup and are not substituted for kernel duration. Cross-core sums are not wall time.

## Real-device benchmark

| Workers / device | Version | Group medians (us) | Median of medians (us) |
|---|---|---|---:|
| 1 / 3 | baseline | 54.85, 54.85, 55.10 | 54.85 |
| 1 / 3 | current | 68.95, 67.90, 69.00 | 68.95 |
| 1 / 3 | diagnostic | 68.65, 67.80, 66.90 | 67.80 |
| 12 / 4 | baseline | 35.40, 36.80, 39.00 | 36.80 |
| 12 / 4 | current | 41.05, 41.55, 42.00 | 41.55 |
| 12 / 4 | diagnostic | 37.90, 40.30, 39.40 | 39.40 |

The diagnostic is a separately compiled image, not shipped source: it removes only generated VF `VV_ALL` calls, preserving directed `VLD_VST`/`VST_VLD`, DMA readiness/reuse protection, store ordering and final drain. The new packed-cache caller already has zero metadata MTE3-to-S handshake flags in its kernel body. Full-batch `rows=8` specialization already eliminates all full-batch VF memory barriers; the diagnostic removes three static `VV_ALL` sites only from the dynamic-tail VF. It does not apply a compiler fix.

| Workers | C++ -> current effective_us | Current regression | Measured diagnostic effective_us | Target |
|---|---:|---:|---:|---|
| 1 | 54.85 -> 68.95 | 25.7% | 67.80 | <= 54.85 us, matching baseline |
| 12 | 36.80 -> 41.55 | 12.9% | 39.40 | <= 36.80 us, matching baseline |

The target is an acceptance goal, not an achieved result or a forecast. The diagnostic is the measured evidence for synchronization-related potential on this exact workload. It does not establish that a future compiler release or source rewrite will reproduce that result. In the one-worker normal72 case the dynamic-tail VF is not executed, so its diagnostic timing difference must not be attributed to eliminating executed barriers. Code placement and run-to-run variance remain possible explanations.

## Separate kernel captures

Each row below describes one clean workflow capture, not100 timed kernel samples. Twelve workers generate12 physical AIV kernel events. These are standalone Window workflows; full decode/CI workload integration timing is unmeasured.

| Workers / version | Kernel events | Per-event median (us) | Per-event max (us) | Sum across events (us) |
|---|---:|---:|---:|---:|
| 1 / baseline | 1 | 31.622 | 31.622 | 31.622 |
| 1 / current | 1 | 47.651 | 47.651 | 47.651 |
| 1 / diagnostic | 1 | 47.627 | 47.627 | 47.627 |
| 12 / baseline | 12 | 7.304 | 7.721 | 87.415 |
| 12 / current | 12 | 10.148 | 10.487 | 121.897 |
| 12 / diagnostic | 12 | 8.454 | 9.129 | 101.666 |

The sums include parallel execution and cannot be interpreted as elapsed time. Kernel makespan/overlap is not inferred from the setup-inclusive Worker span.

## Remaining implementation differences and issue tracking

- C++ scans128 slots and batches consecutive accepted source rows into two-dimensional TLOADs. Python also scans128 slots, but stages token indices and issues per-row input/cosine/sine loads in a separate flush loop. A full Python batch has24 source/RoPE transfers.

- C++ groups consecutive destination slots into two-dimensional TSTORE runs. Python publishes each accepted row separately, with separate payload and scale transfers. It preserves source order, including repeated slots, but does not yet match the baseline transfer grouping.

- Python keeps zero-initialized64-column RoPE backing buffers while consuming32 logical cosine/sine values per row. The mathematical masks and rounding match the baseline on the listed cases; compacting initialization/storage is a separate follow-up requiring measurement.

- Full-batch calls pass constant8; the tail passes the actual pending count. This avoids the full-batch dynamic VF synchronization cost. No generated-code synchronization deletions are part of the PR source.

- [PTOAS #1754](https://github.com/hw-native-sys/PTOAS/issues/1754) tracks redundant `VV_ALL` in bounded nested Window VF loops. It remains relevant to dynamic tails and is not closed by this migration.

- [PTOAS #1756](https://github.com/hw-native-sys/PTOAS/issues/1756) tracks false MTE3/Scalar metadata dependencies for bounded dynamic UB `alloc_tile` addresses. The older FP8 reinterpretation path reproduced it; the current uint8 packed-cache publication path avoids that handshake. This PR does not fix the compiler analysis and must not close the issue.

Matching the latest baseline requires addressing transfer grouping and remaining startup/address-preparation costs, then remeasuring. The earlier approximately60us expected result for the separate-cache layout cannot be carried across the changed algorithm, storage and C++ implementation.

## Environment and evidence

Selected PyPTO: `9b00701173d80f4b1e0b356fbd326c574e6fe1a3`; PTOAS0.72, CANN9.3, A5/dav_3510; runtime `355ce3015e3e344d1fc3a61c206ee01823ce292c`; PTO ISA `5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`. No shared runtime/compiler changes.

Final formatted source was recompiled and its location-stripped caller PTO, generated caller C++, and both VF PTO inputs matched the measured build. Repository pre-commit hooks, focused lint and QKV/Window import integration passed. This is operator validation, not a full-model regression run.

Task evidence:

- Offline compilation: `task_20261113_093634_336968817686`;12 workers `task_20261113_093910_35166808899`;136-token shape `task_20261113_094225_369769922645`.
- Source-to-measured-code comparison: `task_20261113_094409_37969282243`.
- Single-worker benchmark/workflow: `task_20261113_094317_374603620271`.
- Twelve-worker benchmark/workflow: `task_20261113_094828_402120524879`.
- Initial cross-scan replays rejected the incompatible72-token artifact; separate136-token builds resolved the ABI mismatch. Failed attempts are excluded from all timing tables.

Source SHA256:

- `models/deepseek_v4_1_flash/window_publish.py`: `5420f24c1b1389a5e06a98c4aef425f9cf5eac3ccbf68a6412c9b8ccd80c75c1`.
- `models/deepseek_v4_1_flash/qkv_proj_rope.py`: `592678676fcd2f299c723ffb65aa9338283eaa3f6fcc84e70667acbf96456c49`.
