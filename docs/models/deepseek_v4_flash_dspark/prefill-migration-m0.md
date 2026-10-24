# Prefill migration M0 baseline

Tracking: [pypto-lib #1562](https://github.com/hw-native-sys/pypto-lib/issues/1562).
This baseline excludes zigzag CP and every optimization dependent on it.
M0 changes benchmark fixtures only; migration optimizations have not been applied.

## Version and measurement contract

| Component | Revision |
| --- | --- |
| Model source | `f2c80074c0b06ab5bf87fc2bfb06db417fcd35f1` plus the M0 fixture patch |
| Latest PyPTO main resolved for this baseline | `c42e22e9210f5d92a6fcad36d1c8c6e149745073` |
| Pinned simpler runtime | `d8f997fabd1b6988b69a94feef61ab669dda0eea` |
| Pinned PTOAS | `0.67` |
| Pinned PTO ISA | `b1d460ee8c001585ede36b68ba3678dc5a226bbf` |
| CANN / Torch | `9.0.0` / `2.10.0+cpu` |

Measurements use a2a3 hardware, 100 measured rounds after 5 warmup rounds,
with raw per-rank samples retained. The tuning metric is the fastest rank's
median device-domain effective time; the headline is the per-round maximum
across ranks. Retain both when comparing changes. The headline includes dispatch
skew and rank imbalance. Neither value includes compilation, golden computation,
or serving latency.

Inputs and weights are synthetic harness fixtures preserving the implementation's
quantization boundaries. They are not checkpoint weights. Capture a frozen
snapshot once with `--save-data`, then use `--golden-data <work_dir>/data`
for compatible optimization comparisons. `--runtime-dir <work_dir>` also reuses
the compiled program when only runtime configuration changes. The generic-wave
entry did not set an explicit Torch seed; its frozen snapshot is the source of
truth for exact replay. Attention, production MoE, and full prefill use the
explicit seeds shown in the commands below.

## Separate module baselines

| Module | Workload | Fastest-rank median ms | Headline max-rank median ms |
| --- | --- | ---: | ---: |
| SWA | TP4, 8192 group tokens | 11.6787 | 37.9522 |
| HCA | TP4, 8192 group tokens | 22.9582 | 57.7540 |
| CSA | TP4, 8192 group tokens | 23.7464 | 87.8042 |
| Production MoE, hash routing, layer 0 | EP8/TP4, 8192 tokens per active group | 23.7678 | 73.7118 |
| Production MoE, gated routing, layer 3 | EP8/TP4, 8192 tokens per active group | 24.6340 | 59.6966 |
| Generic MoE wave diagnostic | EP8, 128 tokens/rank, balanced routing | 0.7216 | 2.6713 |

Attention uses four cards with 2048 local queries per rank and replicated KV/state.
MoE uses eight cards and **16 experts per card**, preserving the local density
of EP16 with 256 global experts. The EP8 surrogate has 128 total experts and two
active TP4 groups; it does not establish EP16 routing capacity or full-request
latency. The production MoE fixture calls the existing `prefill_layer_moe` /
`prefill_moe` path and includes HC processing, wave staging, exchanges, expert
computation, CP gather, and HC post. The generic 128-token wave is a diagnostic.

The SWA and HCA 8K fixtures initially exhausted their default 256 MiB and 1 GiB
ring heaps. Their successful runs replayed the original builds and goldens with
4 GiB per ring. CSA and production MoE passed with their existing 1 GiB per-ring
settings. Keep these settings fixed for each module's future comparisons.

From the repository root, after activating the pinned environment, submit through
the site's NPU queue. For example, a four-card attention baseline is:

```bash
task-submit --device auto --device-num 4 --ptoas 0.67 --max-time 0 \
  --run 'PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_RAW=1 python models/deepseek_v4_flash_dspark/prefill_swa.py -p a2a3 -d $TASK_DEVICE --tp 4 --token-count 8192 --seed 20261023 --ring-heap-mib 4096 --save-data'
```

Use `prefill_hca.py` with the same flags. For `prefill_csa.py`, omit the ring-heap
override to match its captured baseline. Production MoE is:

```bash
task-submit --device auto --device-num 8 --ptoas 0.67 --max-time 0 \
  --run 'PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_RAW=1 python models/deepseek_v4_flash_dspark/prefill_layer.py -p a2a3 -d $TASK_DEVICE --moe-only --ep 8 --tp 4 --experts-per-rank 16 --token-count 8192 --layer-id 0 --seed 20260825 --save-data'
```

Repeat the distinct gated-routing case with `--layer-id 3`, keeping a separate
build/data directory. `--moe-only` omits attention execution and seeds its stage
input; it preserves the normal full-layer entry when the flag is absent.

## Correctness coverage

All six TP4 attention boundary cases passed their existing numerical comparators:

- `--case ragged2`: two requests with lengths 3 and 4, one padded physical row,
  prefixes 126 and 30, request/rank crossings, and ratio-4/ratio-128 boundaries.
- `--start-pos 126 --token-count 516`: a nonzero-prefix chunk against seeded
  resident cache and compressor state.

These cases passed for SWA, HCA, and CSA. Attention outputs, caches, and applicable
compressor/indexer states use the existing references and tolerances.
The existing DSpark host contract suite also passed 50 tests; those are contract
checks, separate from NPU computation validation.

A production MoE partial-wave case uses `--token-count 513` on EP8/TP4: 129 physical
rows per rank, including a partial second 128-row wave. It passed the existing numerical
comparators and is recorded separately from the 8K performance runs.

Single calls against seeded state do not establish multi-call prefix reuse,
window reuse across independent requests, or prefill-to-decode continuity.
Those remain explicit migration acceptance items in #1562.

## Full prefill

The full baseline uses EP16/TP4/DP4 on 16 cards with **one global 8192-token
request** and DP groups 1, 2, and 3 inactive. It is a separate run from saturated
module surrogates. The initial complete baseline is measured once; the next
complete measurement is reserved for completion of the migration optimizations.

```bash
task-submit --device auto --device-num 16 --ptoas 0.67 --max-time 0 \
  --run 'PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_WARMUP=5 PYPTO_BENCH_RAW=1 python models/deepseek_v4_flash_dspark/prefill_fwd.py -p a2a3 -d $TASK_DEVICE --ep 16 --tp 4 --experts-per-rank 16 --num-tokens 8192 --inactive-dp-groups 1 2 3 --ring-heap-mib 4096 --seed 20260824 --save-data'
```

The complete forward requires exactly zero `x_hc` on inactive groups, matching
the existing kernel's pre-head zeroing; inactive caches and attention stages must
remain unchanged. The first attempt exposed the old comparator's contradictory
input-preservation expectation and stopped before timing. Its evidence is retained
separately; the initial-baseline retry uses the same compiled kernel and saved data.

The existing resident harness uploads `x_hc` and persistent caches once and reuses
them across all warmup/timed dispatches. The first dispatch is validated; later
rounds evolve resident state. Label this measurement **resident-state replay**; it
does not measure independent fresh requests with state restored before each round.
Future full comparisons must retain this same replay contract and initial snapshot.

The complete forward has finite/output-consistency and selected head checks; it
has no independent mathematical reference for all 43 transformer layers. A full
forward pass must not be described as complete numerical correctness.

## Local evidence layout

The captured run lives under `artifacts/dspark-prefill-m0-20261023/`. Each module
has its own `RECORD.md`, `invocation.json`, `result.json`, `run.log`, and snapshot
inventory. Build and data directories remain under `build_output/`. Failed
attempts are retained separately and contribute no accepted timing. The
`environment/` directory records exact pins, runtime binary pin metadata,
activation isolation, fixture patch, commands, and task IDs. `applicability.md`
records existing DSpark equivalents and each migration candidate's conditions.
Generated build/data files are local artifacts and are not committed.

## Full prefill result

Full baseline completed: headline max-rank median **2105.1323 ms**; fastest-rank median **2082.8021 ms** (may include idle ranks). EP16/TP4/DP4, 16 devices, one global 8192-token request, inactive groups 1/2/3, 100 measured / 5 warmup rounds. All 16 raw sample streams contain 100 measurements. Existing finite/consistency/head checks passed; this is not an independent 43-layer mathematical golden. Timing uses the existing resident-state replay: x_hc and persistent caches are uploaded once, the first dispatch is validated, and subsequent timed rounds reuse/evolve state. This is not independent fresh-request latency. The first attempt stopped before timing because the comparator required inactive `x_hc` preservation while the existing kernel explicitly zeros it before the head. The corrected comparator requires exact zeros; cache/state preservation and tolerances are unchanged. The retry uses the same compiled kernel and frozen data. Original evidence is retained in `full_prefill_attempt1/`.
