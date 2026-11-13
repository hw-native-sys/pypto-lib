# Window publication performance report

## Result and scope

The Python Tile/RegScope implementation matches the native C++ output byte-for-byte on the eight tested patterns at both one and twelve workers. For the frozen 72-token replay, current source achieves **47.05 µs at one worker** and **36.45 µs at twelve workers**, versus C++ **53.10/37.95 µs**. These are medians of three group medians for standalone `effective_us`, including device-domain orchestration/scheduling. They are not pure kernel times or full-model decode latency.

The twelve-worker source improvement over the algorithm-aligned Python predecessor is 2.85 µs (7.3%). Normal compiler lowering removes redundant synchronization in constant-row VFs. **No generated PTO, MLIR, LLVM or C++ synchronization removal ships in this PR.** A separately identified VV_ALL-only simulator experiment is diagnostic evidence.

This report consolidates the latest matched measurements. Older timing campaigns are excluded from the comparison tables. Results establish the measured operator-replay target; they do not establish every CI shape or full-model performance.

## Versions and reproducibility

- Native baseline: `models/deepseek_v4_1_flash/kernels/norm_rope_publish.cpp` at `8bbc8f5ca6ac3fd72f9004f509099083d5be10c3`. Its SHA256 is `b335430b873f0345e4e9d5e6ebc9af203cd37ccd0933063518489fe01ecb62e6`, unchanged at the initial rebased base `f33b566ba880b6592db563d2d2215f5b5e6e3cfb` and final publication base `914ec5fad8efc12703ce12ca3d3f481371e42648`.
- Python before tail specialization: revision `963bd6aa`; current source: PR #1810, measured specialization image. The report-only publication does not change the measured operator.
- PyPTO `9b00701173d80f4b1e0b356fbd326c574e6fe1a3`; runtime `355ce3015e3e344d1fc3a61c206ee01823ce292c`; PTO ISA `5487a6b0e8d9bd3653dbee8d7c874c667ee8db2f`; PTOAS 0.72; CANN 9.3; A5/dav_3510; Linux aarch64; Python 3.10.
- Single-rank execution. No compiler/runtime source modifications. Baseline ABI `[780,128,544]` and flattened Python ABI `[99840,544]` describe identical packed bytes.
- Current operator SHA256: `ec103e2b53c114e2cf976439f34e5506ac134ab0ee8255c574c29bf6acc84993`.
- Serialized frozen-input SHA256: `cd5665b2765374ec9cda8d56222574f874d484c08e2bc087f950f6eeac3119e8`. The historical random seed is unavailable; all variants replay identical serialized tensors and initial cache bytes.

## Algorithm and data movement

Both C++ and Python use the following contract:

- BF16 source `[72,512]`, BF16 gamma `[512]`, FP32 cosine/sine `[72,32]`, int64 slots `[72]`, packed uint8 cache `[780,128,544]`. Each slot contains 512 FP8 payload bytes and 32 bytes of BF16 scales.
- RMSNorm over 512 values, eight segments accumulated per 64 lanes, pairwise reduction, epsilon `1e-6`, and BF16 normalization rounding. RoPE rotates the final 64 values as 32 interleaved pairs, with BF16 rounding after rotation.
- Group-32 absmax clamped to `1e-4`, multiplied by `1/448`, then rounded upward to a power of two through `(bits + 8388607) >> 23` and `exponent << 23`. The exact FP32 power-of-two scale is used for division and packed as BF16. FP8 conversion uses round R with saturation disabled.
- A 128-slot scan, negative-slot rejection, page ownership `(slot // 128) % workers`, and batches of eight accepted rows. Pending rows survive a scan boundary.
- Input runs consist of consecutive tokens accepted by the same worker, bounded by the scan and remaining batch capacity. Each run emits three 2D loads into pending rows. No token-index staging buffer or second gather loop remains.
- Output runs contain consecutive increasing destination slots only. Duplicate or descending slots split runs and retain source-order publication. Each run emits separate 2D payload and scale stores.
- RoPE storage is compact `[8,32]`, without the former upper-half zero initialization. The vector load matches the C++ row stride of 32 FP32 values; only its lower 32 lanes contribute to rotation. Full batches pass constant eight; the current source dispatches constant tails 2–7 and retains a dynamic one-row fallback.

The source encodes dynamic transfer extents using `valid_shape`; generated C++ confirms dynamic row counts for the three TLOADs and both TSTOREs. The output loop uses explicit `pl.while_` state because a reduced natural-while inline example exposes a ConvertToSSA defect; see [PyPTO #3415](https://github.com/hw-native-sys/pypto/issues/3415).

## CI call mapping and coverage

`.github/scripts/perf_cases.json` configures these A5, single-device decode cases with twelve `--start-pos` values of 8192. Each reaches `qkv_proj_rope.py` → `window_publish` → the `norm_rope_pub_win` SPMD task, preserving `deps=[cache_ready]`. Worker count is bounded by `max(1, min(num_tokens, 12))`.

| CI case | Direct caller under `models/deepseek_v4_1_flash/` | Status in this report |
|---|---|---|
| dsv4.1-flash-decode-swa-b12-8k | `decode_swa.py` | Static call mapping only |
| dsv4.1-flash-decode-c1a-full-b12-8k | `decode_c1a_full.py` | Static call mapping only |
| dsv4.1-flash-decode-c1a-reindex-b12-8k | `decode_c1a_reindex.py` | Static call mapping only |
| dsv4.1-flash-decode-c1a-reuse-b12-8k | `decode_c1a_reuse.py` | Static call mapping only |
| dsv4.1-flash-decode-c2a-full-b12-8k | `decode_c2a_full.py` | Static call mapping only |
| dsv4.1-flash-decode-c2a-reuse-b12-8k | `decode_c2a_reuse.py` | Static call mapping only |

The decoder tensor shapes and effective token counts were not captured from these full CI workflows in this campaign. Prefill callers also reach the shared helper; their larger dynamic shapes are not benchmarked here. The 72-token frozen standalone input is CI-derived evidence, but is not asserted to cover every manifest case. CI-workload target-kernel durations, full-model latency, and workload-versus-standalone scheduling differences remain **unverified**.

## Measurement protocol

Compilation, assembly, simulation, statistical analysis and checks use `task-submit --no-device`. Already-compiled hardware replay uses `task-submit --device auto`, reads `$TASK_DEVICE`, and disables compilation. Each worker-count comparison keeps all three versions on the same allocated card: one-worker device 5, twelve-worker device 0. Separate twelve-worker workflow captures use device 2. No available card-clock/thermal telemetry was retained; cross-card absolute timings are not treated as an optimization effect.

Each benchmark has three groups of ten warmups and 100 timed rounds, with version order baseline/before/current reversed in group two. Compile, initialization, first loading and warmups are excluded. `effective_us` comes from `BenchmarkStats.per_round("effective")`: the framework post-graph-build device execution window covering orchestration/scheduling (`orch∪sched`). It includes wrapper overhead. The samples are rounded to 0.1 µs when recorded. P95 uses nearest rank, `ceil(0.95*N)`. The headline is the median of group medians; the pooled 300-sample median is separately labeled.

### Repeated device benchmark

All values are µs; every row contains 300 timed samples.

| Workers / device | Version | Group medians | Group P95 | Headline |
|---|---|---|---|---:|
| 1 / 5 | C++ | 52.70, 53.30, 53.10 | 56.00, 57.00, 57.80 | 53.10 |
| 1 / 5 | Python before | 46.80, 48.30, 47.60 | 51.00, 51.10, 52.50 | 47.60 |
| 1 / 5 | Python current | 47.05, 46.60, 47.25 | 50.70, 50.00, 50.40 | 47.05 |
| 12 / 0 | C++ | 36.60, 37.95, 39.40 | 40.80, 40.60, 41.70 | 37.95 |
| 12 / 0 | Python before | 39.45, 38.80, 39.30 | 43.60, 41.90, 42.40 | 39.30 |
| 12 / 0 | Python current | 37.65, 34.50, 36.45 | 40.40, 38.90, 39.80 | 36.45 |

| Workers | Version | Min | Mean | Pooled median | P95 | Max |
|---|---|---:|---:|---:|---:|---:|
| 1 | C++ | 49.700 | 53.364 | 53.100 | 57.000 | 65.400 |
| 1 | Python before | 43.100 | 47.735 | 47.600 | 51.300 | 57.200 |
| 1 | Python current | 42.700 | 47.213 | 47.000 | 50.400 | 55.400 |
| 12 | C++ | 32.000 | 38.115 | 38.200 | 41.600 | 45.700 |
| 12 | Python before | 34.200 | 39.190 | 39.150 | 42.700 | 53.500 |
| 12 | Python current | 30.600 | 36.269 | 36.200 | 39.900 | 51.400 |

Current versus C++ headline differences are −6.05 µs (−11.4%) at one worker and −1.50 µs (−4.0%) at twelve. Relative to Python before, they are −0.55 µs (−1.2%) and −2.85 µs (−7.3%). Group variation matters: twelve-worker current is slower than C++ in group one and faster in groups two and three. No confidence interval or universal four-percent gain is claimed.

### Correctness

Eight whole-cache byte comparisons pass at each worker count: normal72, empty0, tail9, masked/duplicate slots, fragmented accepted runs with ownership changes, all-invalid slots, nonconsecutive destinations, and separately compiled cross-scan136. Comparison includes untouched cache bytes. New slot patterns obtain their reference from the frozen native binary. All simulator comparisons below are also byte-exact. This is tested-input evidence rather than a proof over all floating-point inputs or dynamic shapes.

## Separate hardware workflow evidence

Each capture contains one rank, one SPMD task (`r1t0`, FuncId 0, taskId 4294967296, epoch 1), and twelve physical Worker kernel records. Native `baseline_kernel_spmd` and Python `norm_rope_publish_spmd` match by standalone call boundary, packed-cache contract and shape. The mapping artifacts contain `deps.json` and `name_map*.json`; the per-event raw trace-array indices below make the match auditable.

Durations use explicit `args["kernel-duration-us"]`. Worker `dur` contains `local_setup_us` and cannot be substituted. Exact pure-kernel start/end timestamps are unavailable, so pure-kernel interval union, span and overlap are **N/A**. The sum below is cumulative concurrent-core work, not wall time. Capture overhead is excluded from the regular benchmark above.

| Version | Events / participating Worker tracks | Min | Mean | Median | P95 | Max | Cumulative kernel µs |
|---|---:|---:|---:|---:|---:|---:|---:|
| C++ | 12 / 12 | 6.804 | 7.032 | 7.058 | 7.183 | 7.183 | 84.385 |
| Python before | 12 / 12 | 8.146 | 8.466 | 8.547 | 8.693 | 8.693 | 101.591 |
| Python current | 12 / 12 | 6.223 | 6.605 | 6.716 | 6.952 | 6.952 | 79.259 |

The current one-worker hardware capture is **unavailable**. Retry `task_20261113_162610_231827619372` failed during runtime initialization with code 507033 before kernel execution. Its completed benchmark and simulator measurements remain valid independent evidence. Historical one-worker captures are not substituted for the current image.

### Raw capture and event mapping

Artifact roots below are retained local task artifacts, not public downloads. Paths are relative to their respective retained capture root; raw generated files are intentionally not checked into the repository.

- C++: `_jit_latest_baseline_check_20261113_093912_db86sqlo/dfx_outputs/merged_swimlane_20261113_111908.json`; companion `deps.json` and `name_map__jit_latest_baseline_check_20261113_093912_db86sqlo.json` in the same directory.
- Python before: `_jit_latest_window_check_20261113_102045_y7dffo6z/dfx_outputs/merged_swimlane_20261113_111937.json`; companion `deps.json` and `name_map__jit_latest_window_check_20261113_102045_y7dffo6z.json` in the same directory.
- Python current: `_jit_latest_window_check_20261113_110957_8n0l3rwm/dfx_outputs/merged_swimlane_20261113_112005.json`; companion `deps.json` and `name_map__jit_latest_window_check_20261113_110957_8n0l3rwm.json` in the same directory.

Every row has the task/epoch mapping stated above. `id` is the raw event identifier, not an inferred logical block ID. Units are µs; `setup` and `Worker dur` are separate from kernel time.

| Version | Array index | Event id | CoreId | pid/tid | Worker ts | Setup | Worker dur | Kernel |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| C++ | 21 | 0 | 34 | 4/10340 | 363.338 | 5.392 | 12.460 | 7.068 |
| C++ | 22 | 1 | 35 | 4/10350 | 363.519 | 5.248 | 12.052 | 6.804 |
| C++ | 23 | 2 | 42 | 4/10420 | 363.674 | 5.259 | 12.185 | 6.926 |
| C++ | 24 | 3 | 43 | 4/10430 | 363.834 | 4.932 | 11.981 | 7.049 |
| C++ | 25 | 4 | 50 | 4/10500 | 363.968 | 4.854 | 11.810 | 6.956 |
| C++ | 26 | 5 | 51 | 4/10510 | 364.115 | 4.730 | 11.645 | 6.915 |
| C++ | 27 | 6 | 58 | 4/10580 | 364.257 | 4.538 | 11.650 | 7.112 |
| C++ | 28 | 7 | 59 | 4/10590 | 364.400 | 4.372 | 11.341 | 6.969 |
| C++ | 29 | 8 | 66 | 4/10660 | 364.434 | 4.573 | 11.724 | 7.151 |
| C++ | 30 | 9 | 67 | 4/10670 | 364.498 | 4.346 | 11.513 | 7.167 |
| C++ | 31 | 10 | 74 | 4/10740 | 364.601 | 4.284 | 11.467 | 7.183 |
| C++ | 32 | 11 | 75 | 4/10750 | 364.681 | 4.375 | 11.460 | 7.085 |
| Python before | 21 | 0 | 32 | 4/10320 | 359.409 | 4.989 | 13.682 | 8.693 |
| Python before | 22 | 1 | 33 | 4/10330 | 359.536 | 4.975 | 13.570 | 8.595 |
| Python before | 23 | 2 | 40 | 4/10400 | 359.689 | 5.043 | 13.359 | 8.316 |
| Python before | 24 | 3 | 41 | 4/10410 | 359.833 | 4.707 | 13.344 | 8.637 |
| Python before | 25 | 4 | 48 | 4/10480 | 359.948 | 4.557 | 13.067 | 8.510 |
| Python before | 26 | 5 | 49 | 4/10490 | 360.077 | 4.444 | 13.029 | 8.585 |
| Python before | 27 | 6 | 56 | 4/10560 | 360.182 | 4.645 | 13.267 | 8.622 |
| Python before | 28 | 7 | 57 | 4/10570 | 360.304 | 4.357 | 12.985 | 8.628 |
| Python before | 29 | 8 | 64 | 4/10640 | 360.387 | 4.713 | 12.869 | 8.156 |
| Python before | 30 | 9 | 65 | 4/10650 | 360.481 | 4.544 | 12.690 | 8.146 |
| Python before | 31 | 10 | 72 | 4/10720 | 360.559 | 4.400 | 12.712 | 8.312 |
| Python before | 32 | 11 | 73 | 4/10730 | 360.637 | 4.468 | 12.859 | 8.391 |
| Python current | 21 | 0 | 32 | 4/10320 | 399.067 | 5.260 | 12.212 | 6.952 |
| Python current | 22 | 1 | 33 | 4/10330 | 399.199 | 5.223 | 11.981 | 6.758 |
| Python current | 23 | 2 | 40 | 4/10400 | 399.348 | 5.246 | 11.952 | 6.706 |
| Python current | 24 | 3 | 41 | 4/10410 | 399.493 | 4.912 | 11.782 | 6.870 |
| Python current | 25 | 4 | 48 | 4/10480 | 399.615 | 4.872 | 11.492 | 6.620 |
| Python current | 26 | 5 | 49 | 4/10490 | 399.736 | 4.644 | 11.447 | 6.803 |
| Python current | 27 | 6 | 56 | 4/10560 | 399.847 | 4.745 | 11.499 | 6.754 |
| Python current | 28 | 7 | 57 | 4/10570 | 399.962 | 4.459 | 11.185 | 6.726 |
| Python current | 29 | 8 | 64 | 4/10640 | 400.052 | 4.722 | 10.956 | 6.234 |
| Python current | 30 | 9 | 65 | 4/10650 | 400.143 | 4.441 | 10.675 | 6.234 |
| Python current | 31 | 10 | 72 | 4/10720 | 400.217 | 4.422 | 10.645 | 6.223 |
| Python current | 32 | 11 | 73 | 4/10730 | 400.298 | 4.374 | 10.753 | 6.379 |

Workflow stage attribution beyond the combined benchmark window, and full CI-workload DAG timing, remain unverified. The captures prove per-Worker kernel durations; they do not identify the benchmark difference as pure compute savings.

## Simulator pipeline analysis

The replay models worker 0 with logical worker count one or twelve on one physical core; AIV0 is analyzed, anchored at its first positive-duration event. It does not model twelve-core contention or real-device orchestration. The frozen slots contain twelve runs of six slots on distinct page owners: one worker processes nine full-eight batches; twelve workers each scan 72 slots and compute one six-row tail. Scaling changes the executed VF path as well as work per core.

| Replay | AIV0 span µs | Main VF start µs | Compute VF duration µs | Dynamic VF memory barriers |
|---|---:|---:|---:|---:|
| 1 C++ | 28.349 | 3.278 | 13.878 total / 9 VFs | 0 |
| 1 Python before | 24.562 | 3.487 | 11.939 total / 9 | 0 |
| 1 Python current | 24.713 | — | 11.937 total / 9 | 0 |
| 12 C++ | 6.569 | 4.465 | 1.354 | 0 |
| 12 Python before | 8.911 | 4.746 | 3.462 | 61 |
| 12 VV_ALL-only diagnostic | 6.709 | — | 1.444 | 12 |
| 12 Python current | 6.918 | 4.792 | 1.198 | 0 |

The one-worker source change slightly increases simulator span by 0.151 µs while its compute VF total is effectively unchanged. Twelve-worker current remains 0.349 µs slower than native in this one-core simulator, although the captured hardware kernel median is lower. Simulator span, hardware per-kernel events, and repeated wrapper latency measure different environments/boundaries; none predicts the other exactly.

### Pipeline launches and dependencies

Five useful MTE2 transfers load shared gamma (1024 bytes), slots (576), source (6144), cosine (768) and sine (768). The final three are run transfers; their launches overlap. Two MTE3 transfers publish payload (3072 bytes) and scales (192). All following intervals are simulator AIV0 µs.

| Version | Pipe | First launch | Last completion | Busy interval union | Events |
|---|---|---:|---:|---:|---:|
| C++ | SCALAR | 0.000 | 6.569 | 3.918 | 1209 |
| C++ | MTE2 | 1.083 | 3.597 | 1.414 | 5 |
| C++ | SIMD | 4.456 | 5.816 | 1.187 | 2203 |
| C++ | MTE3 | 5.823 | 6.447 | 0.624 | 2 |
| Python before | SCALAR | 0.000 | 8.911 | 3.585 | 1130 |
| Python before | MTE2 | 2.092 | 3.925 | 1.094 | 5 |
| Python before | SIMD | 1.688 | 8.204 | 3.181 | 2102 |
| Python before | MTE3 | 8.211 | 8.782 | 0.567 | 2 |
| Python current | SCALAR | 0.000 | 6.918 | 3.600 | 1237 |
| Python current | MTE2 | 2.057 | 3.906 | 1.097 | 5 |
| Python current | SIMD | 1.744 | 6.097 | 1.083 | 2205 |
| Python current | MTE3 | 6.104 | 6.782 | 0.673 | 2 |

SIMD includes non-compute queue events and overlapping VF activity; its interval union is not the main-VF duration. SCALAR count alone is not a critical-path estimate.

The causal chain is slot scan/address preparation → run MTE2 readiness → main VF → publication readiness → MTE3 → final drain. Python before launches MTE2 1.008 µs later than native but finishes only 0.328 µs later; its busy MTE2 union is shorter. Main VF starts 0.281 µs later, then takes 2.107 µs longer. The first MTE3 moves from 5.823 to 8.211 µs, following main-VF completion at 5.819 versus 8.208. This identifies long dynamic VF synchronization as the main regression, with a smaller independent startup cost. It does not support a missing-MTE2-concurrency explanation.

Current main VF starts at 4.792 µs and lasts 1.198 µs; first MTE3 is 6.104 µs and last completion 6.782 µs. Four generated branch-result tile-copy VFs start at 5.920, 6.001, 6.028 and 6.056 µs, with durations 0.098, 0.045, 0.045 and 0.045. They overlap; their durations must not be added to makespan. Generated C++ contains twelve static TMOV sites, of which the six-row path executes four such short events. For one worker the eight inter-batch VF gaps total 10.092 µs in C++ versus 8.080 µs in Python before; they are repeated scan/transfer/publication preparation, not eight extra compute batches.

### VF instruction mix

Counts below are dynamic simulator instruction counts, not cycle attribution. Conversion, loads, multiplies, predicates and stores dominate counts; barriers additionally impose ordering stalls that counts alone do not quantify.

| Instruction | C++ | Python before | Python current |
|---|---:|---:|---:|
| RV_VCVT_F2F | 300 | 300 | 300 |
| RV_VMUL | 168 | 168 | 168 |
| RV_VLD | 48 | 90 | 158 |
| RV_PLT | 2 | 96 | 164 |
| RV_VST | 0 | 42 | 110 |
| RV_VDUP | 150 | 102 | 102 |
| RV_VCMAX | 96 | 96 | 96 |
| RV_VSHRI | 96 | 96 | 96 |

### Synchronization and UB evidence

Native postprocess (`norm_rope_publish.cpp`, row loop and scale store):

```cpp
for (uint16_t row = 0; row < rows; ++row) {
    // RMSNorm, RoPE and FP8 conversion omitted.
    vsts(codes, (__ubuf__ uint32_t*)factors,
         row * 8 + segment, ONEPT_B32, full);
}
```

Original dynamic Python VF (`vf/vf_1.mlir`, excerpt; original barrier retained):

```mlir
// Outer loop: row < %arg1; inner loop: seven ordinary segments.
pto.vsts %115, %9[%80], %10 {dist = "PK4_B32"}
  : !pto.vreg<256xui8>, !pto.ptr<ui8, ub>, !pto.mask<b32>
// Other original instructions omitted; not a standalone reproducer.
pto.mem_bar "VV_ALL"
```

At six rows, 42 inner-segment VV_ALL, six row-end VV_ALL and one final VV_ALL total 49; six VLD_VST and six VST_VLD give 61 memory barriers. The diagnostic removes three static VV_ALL sites in lowered LLVM, corresponding to those 49 dynamic events, while retaining all twelve directed events. Its simulator span is 6.709 µs. This is not a blanket claim that every directed barrier is redundant.

In that original dynamic VF allocation, payload writes span UB `[1024,4096)`, scale writes `[0,256)`, source reads `[6400,12544)`, gamma reads `[5376,6400)`, cosine physical reads `[14592,15488)`, and sine physical reads `[15616,16512)`. RoPE row stride is 32 FP32 values while vector loads physically read 64; only the lower 32 lanes participate. The listed reads and output writes are disjoint, and valid consumer lanes match native. These addresses belong to the original diagnostic allocation, not the current specialized allocation. Caller DMA readiness/reuse dependencies remain necessary.

Current `_flush` passes literal row counts 2–7 with populated-prefix views; full batches pass eight. All these VFs lower without memory barriers under the pinned compiler. One row keeps the validated dynamic fallback because constant-one specialization exhausts PTOAS 0.72's layout frontier budget (288). No manual barrier editing is needed for the measured current image.

## Remaining differences and dependencies

- [PTOAS #1754](https://github.com/hw-native-sys/PTOAS/issues/1754): generic redundant VV_ALL analysis remains open; source specialization handles lengths 2–8. The dynamic one-row case retains generated synchronization.
- [PTOAS #1756](https://github.com/hw-native-sys/PTOAS/issues/1756): generic false MTE3/Scalar metadata dependencies remain open. Current packed uint8 storage avoids the historical handshake. The current diagnostic removes zero MTE3_S instructions, so **no separate MTE3_S speedup is claimed**.
- [PyPTO #3415](https://github.com/hw-native-sys/pypto/issues/3415): inline natural-while SSA can lose scalar carry and introduce undefined tensor carry. The shipped output-run loop uses explicit `pl.while_` row state and does not rebind cache aliases; correctness is validated with that workaround.
- Lowering still differs: branch-result tile copies, register packing into an aligned 256-byte scale vector versus native ONEPT_B32 stores, split ordinary/RoPE segment handling, scalar/address startup, and compiler-generated barriers before individual payload/scale stores versus native after each output run. These are follow-up candidates, not measured additional gains.

## Commands, artifacts and validation ledger

All execution uses the queue. Representative command forms below refer to retained local harnesses; prepare the pinned toolchain and frozen input before replay. Harnesses are experiment artifacts, not installed public commands.

```bash
# Offline compile, assemble and simulator replay
 task-submit --no-device --run 'bash "$EVIDENCE_ROOT/sim.sh" 12 baseline'
 task-submit --no-device --run 'bash "$EVIDENCE_ROOT/sim.sh" 12 dispatch'
# Precompiled correctness and three-group benchmark; harness reads TASK_DEVICE
 task-submit --device auto --run 'bash "$EVIDENCE_ROOT/board.sh" 12'
# Separate precompiled capture
 task-submit --device auto --run 'bash "$EVIDENCE_ROOT/capture.sh" 12'
# Recompute statistics from retained per-round samples and raw events
 task-submit --no-device --run 'python "$EVIDENCE_ROOT/report_stats.py"'
```

The queue submission environment must export `EVIDENCE_ROOT`. Local evidence contains `benchmark-{1,12}/{baseline,current,optimized}/group-{1,2,3}/run.log`, `correctness-{1,12}/<case>/run.log`, `metrics.json`, and `report-statistics.json`. Raw A5 traces remain under `build_output/window-workers/sim-<workers>-<variant>/collect/out/OPPROF_*/simulator/core0.veccore0/trace.json`, with generated replay arguments and VF images under `build_output/window-workers/replay/` and the associated VF directories. Incorrect two-dimensional native replay descriptors were preserved under `invalid-descriptor-*` and excluded; valid native replay uses the original three-dimensional descriptor.

| Evidence | Queue task |
|---|---|
| Baseline compile/assembly | `task_20261113_102042_15071279875`, `task_20261113_102126_155900311295`, `task_20261113_102215_161541311255` |
| Corrected native simulator replay | `task_20261113_105442_340851431053`, `task_20261113_105443_34097149529` |
| Source-specialization simulation | `task_20261113_111037_687371964`, `task_20261113_111453_96008429010` |
| Correctness and repeated hardware benchmarks | `task_20261113_111304_84246422730`, `task_20261113_111301_83963430859` |
| Successful twelve-worker hardware capture | `task_20261113_111815_11604203433` |
| Statistical aggregation | `task_20261113_162811_245300929243` |
| Report checks | `task_20261113_163418_28651057506`, `task_20261113_163518_29323111437` |
| Unavailable one-worker capture retry | `task_20261113_162610_231827619372` (initialization failure, excluded) |

Repository checks include all-file pre-commit, focused Ruff, QKV/Window integration import, docs navigation/public-doc validation and nine documentation unit tests, run again in the report publication transaction. At PR source head `4e2fc2d5`, remote pre-commit, unit tests, Docs build and A5 device checks passed. The [A5 simulation check](https://github.com/hw-native-sys/pypto-lib/actions/runs/43361667421/job/131676943899) failed because VPTO helpers reject `platform='a5sim'` (CPU simulation) for device VPTO objects. That failure is an unresolved CI limitation and is distinct from the successful queued A5 instruction-simulator replay above. New report-commit checks must be inspected independently; this report does not claim all CI is green.
