# Compensated Cube scoring for the DSpark CSA decode Indexer

The opt-in `cube_compensated` implementation contracts the 64-head score on
Cube and sends one FP32 score per candidate to Vector. It preserves the
original score gate with compensated FP16 operands and FP32 accumulation.

## Measured result

The current Cube path is slower than the new Vector baseline after
[PR #1255](https://github.com/hw-native-sys/pypto-lib/pull/1255). The GM
payload reduction does not produce a latency improvement against this main
revision. The compensated path remains opt-in; Vector remains the default.

TP=1, runtime B=16, eight queries per request, all 16 start positions 131072.
The 2026-09-19 rebase comparison uses exact main commit
`d4b05f91c097e0318c50bae522eb51e1aa231ff4` as the Vector baseline and the
rebased Cube implementation at `92c3e39` on the same device 8. All four
Indexer/CSA implementations were freshly compiled. Both paths replay the
same frozen inputs and use the original comparators. Main's 384-candidate
Vector tile and column reduction are preserved; the Cube pipe is separately
sized for its fixed 256-candidate tile.

Each operation uses A-B-B-A order, five warmups and 100 timed rounds per
session, for 200 samples per implementation. Profiling is disabled for all
complete-operation timings. Positive changes below mean longer latency.

| Operation and metric | Main Vector (us) | Compensated (us) | Change |
|---|---:|---:|---:|
| Indexer median | 1268.21 | 1357.32 | +7.03% |
| Indexer mean | 1271.24 | 1372.26 | +7.95% |
| CSA mean | 2301.77 | 2383.87 | +3.57% |
| CSA median | 1918.75 | 2011.07 | +4.81% |
| CSA p95, nearest rank | 2805.46 | 3094.28 | +10.29% |

CSA remains bimodal: 95/200 Vector samples and 73/200 Cube samples exceed
2500 us. Individual Vector session medians are 2685.18 and 1911.16 us;
Cube session medians are 2009.77 and 2011.07 us. All samples are retained,
and the median alone is sensitive to the mode mix. Vector session means
are 2332.79 and 2270.75 us, both below Cube's 2387.75 and 2379.98 us.
PR #1255's reported 1957.61 us median used device 0 and a different frozen
fixture; it is not the measured baseline for this paired device-8 run.

A separate same-device, standalone Indexer L4 capture measured
`indexer_score_topk_leaf` at **1101.20 -> 1183.46 us (+7.47%)**.
This is one task-span observation per implementation, including on-core
waits and start skew; it is not a benchmark median or pure Cube compute time.
The additional coefficient preparation task spans 11.36 us.

The measured toolchain was PyPTO `2f892f96564d`, runtime `097735888e6d`,
PTOAS 0.61, PTO ISA `03e45c4bda48`, and CANN 9.0.0 on A2/A3.
Each fresh process begins with the frozen fixture. The standard benchmark
repeats the same fixed positions without restoring tensors each round;
metadata checks confirm disjoint historical-state reads and current writes.
This is a repeated fixed-step measurement, not consecutive decode positions.

All eight runs pass the stock precision gates. All 128 Top-512 sets match between
Vector and Cube; 80 returned index positions differ only in ordering within
those sets. Ranked scores and the 65,536 scores matched by candidate ID have
zero outliers, with maximum absolute difference 3.0517578125e-5. Complete CSA
output is bitwise identical across all 2,097,152 FP32 values. Both
implementations repeat consistently. The committed validation record keeps
the older comparisons against `eacafcf` and `f3167ec` as historical evidence;
their speedup claims do not apply to the new main baseline.

The C2V representation uses one FP32 value instead of 64 INT32 values per
candidate: 1/64 of the original bytes per candidate. The current Vector
baseline uses 384-candidate tiles (96 KiB each); Cube uses 256-candidate tiles
(1 KiB each). With each path's own tail padding, the target issues 11,344
Vector tiles versus 16,464 Cube tiles, or 1063.5 MiB versus 16.078125 MiB in
each direction. The total padded-byte ratio is therefore approximately
1.5118%, rather than exactly 1/64. This is logical GM payload calculated
from the tile shapes, not measured physical HBM traffic or a proportional
latency claim. Precision compensation, Key loading, local transfers,
synchronization, and TopK still take time.

## Execution

1. Query preprocessing, Key compression/cache writes, and head-weight
   projection retain their original dependencies.
2. An AIV preparation task computes `c = FP32(query_scale * weight)`, scales
   it by 16384, and writes three FP16 coefficient residuals. It also prepares
   21 KiB of reusable Cube constants. There is no full-cache Key-scale
   packing/gather pass.
3. The mixed leaf task uses 24 Cube / 48 Vector workers. Cube computes exact
   INT32 QK, forms two exact nonnegative FP16 R limbs locally, and contracts
   them with the three coefficient limbs using six FP32-accumulating GEMVs.
   The large R tensor never visits GM. This is the mathematical Matmul
   replacement, implemented with multiple Cube operations for precision.
4. Key tiles use two L1 buffers. The next Key DMA overlaps current math;
   low-limb Fixpipe movement overlaps high-limb GEMVs. Constants remain
   resident. Immutable page-table cache lines are invalidated once per leaf.
5. An eight-slot compact-score GM pipe feeds both paired AIVs. Vector applies
   the original FP32 Key scale to the compact score and accumulates each
   half-leaf in UB. TopK starts as soon as that leaf arrives, while Cube
   advances to the next leaf. Sort width is 512/1024/2048/4096 according to
   valid length; partial padding retains the original finite `-FLT_MAX` sentinel.
6. The unchanged query merge combines half-leaf Top-512 pairs.

Vector therefore performs compact-score scaling and TopK; the original
head-sized Broadcast/Mul/ReduceSum is eliminated. See [NUMERICS.md](NUMERICS.md)
for the bit-alias construction, coefficient bound, signed-cancellation
regression, and FP32 accumulation-order limitations.

## Selection and validation

Set `DSPARK_INDEXER_SCORE_IMPL=cube_compensated` before importing the model.
The default remains `vector`. The Cube path requires the DSpark 64-head,
128-wide, 32-row-page, S=8 configuration and the tested A2/A3 toolchain.
There is no automatic runtime coefficient-range guard or fallback. Arbitrary
signed FP32 inputs are not promised to be bitwise equivalent; inputs outside
the documented numerical contract should use `vector`.

From an initialized worktree, portable arithmetic tests need only PyTorch:

```bash
python models/deepseek_v4_flash_dspark/kernels/indexer_cube_score/test_compensated_numerics.py
```

The mixed-kernel device smoke checks shuffled physical pages, empty queries,
all important tail lengths, untouched storage, signed cancellation, and true
INT8 extremes. Run it on a device allocated by the normal task queue:

```bash
python models/deepseek_v4_flash_dspark/kernels/indexer_cube_score/test_fused.py -d "$TASK_DEVICE"
```

For the complete TP1 CSA target, pass all 16 positions explicitly; a single
scalar start position selects batch one in the CSA runner:

```bash
DSPARK_INDEXER_SCORE_IMPL=cube_compensated python models/deepseek_v4_flash_dspark/decode_csa.py \
  --tp 1 -d "$TASK_DEVICE" \
  --start-pos 131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072,131072
```

[validation_results.json](validation_results.json) records the benchmark
configuration, pinned toolchain, source hashes, and measured results.
