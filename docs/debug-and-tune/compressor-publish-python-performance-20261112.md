# C1A compressor publication: opt-in Python baseline

The existing `compressor.c1a_compressor_publish` and all model callers continue
to use the C++ extern. The new
`compressor_publish.c1a_compressor_publish_python` is for explicit evaluation,
with the same argument order and completion-task return contract. No workflow
selects it automatically. The entry includes the independently validated
removal of cosine/sine zero initialization.

```python
from compressor_publish import c1a_compressor_publish_python

published = c1a_compressor_publish_python(
    source, norm_weight, latent, cosine, sine, slots, cache,
    num_tokens, cache_ready, source_ready,
)
```

The fused Tile/RegScope implementation retains unrotated BF16 latent and
packed MXFP4 cache rows: 256 payload bytes plus 32 BF16 scales (64 bytes).
Invalid slots update latent only. Physical-block ownership preserves
last-writer token order for repeated slots. BF16 rounding before latent,
BF16 rounding after tail RoPE and nearest-even E2M1 packing are preserved.

## Measurement boundary

The standalone stage is extracted from `dsv4.1-flash-decode-c1a-full-b12-8k`:
source BF16 [72,512], cosine/sine FP32 [72,32], weight BF16 [512], slots INT64
[72], cache UINT8 [780,128,1,320], latent BF16 [72,512], runtime INT32 valid
count. Eight AIV workers process batches of at most eight rows. This is not
a full decode or CI-workflow measurement.

Timings are single-device dispatch `effective_us`, excluding compilation,
host initialization, validation and warmup, not individual AIV kernel time.
Each implementation has three interleaved groups of 100 samples after ten
warmup iterations on the same allocated device. P95 uses linear interpolation.

The shared installation uses CANN 9.3.0, A5, PTOAS 0.72 and CCEC/LLVM 15.0.5.
Installed PyPTO/simpler distributions report 0.1.0; actual managed PTO ISA is
f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7. Exact wheel source commits were not
embedded. Source-pin-aligned compatibility remains unverified. Each comparison
uses the same installation and frozen inputs.

## Final public-entry baseline

The final public entry passed all eight numerical cases on device 7. These
measurements use the actual proposed Python entry, not a renamed result from
an earlier diagnostic. The existing C++ path is the comparison baseline.

| Entry | Median (us) | P95 (us) | Samples | vs C++ |
| --- | ---: | ---: | ---: | ---: |
| Existing C++ extern | 39.1510 | 42.7518 | 300 | — |
| Opt-in Python, no rotary zero initialization | 49.6825 | 53.3907 | 300 | +26.90% |

The Python entry remains slower, which is why activation is deferred.
Compilation task: task_20261112_154301_37573205929. Numerical and benchmark
task: task_20261112_154426_38205822805, exit 0. The frozen baseline and proposed
entry share input tensors and toolchain; both are standalone publication.


## Independent alignment experiments

Only the zero-initialization change is included. The other changes below are
independent diagnostics and their reductions must not be added together.

| Implementation / change | Median (us) | P95 (us) | vs Python control |
| --- | ---: | ---: | ---: |
| C++ extern | 38.2430 | 41.5731 | — |
| Fused Python control | 50.2975 | 54.4141 | — |
| Remove rotary zero initialization | 48.4520 | 52.0230 | -3.67% |
| Payload row stride: 512 to 256 bytes | 48.8870 | 52.2245 | -2.80% |
| BF16 scale output stride: 128 to 64 bytes | 49.1180 | 53.3413 | -2.35% |
| Simplify owner arithmetic | 49.9110 | 53.5909 | -0.77% |

Each row has 300 samples from device 1. Removing initialization reduced all
three group medians. Other candidates had mixed group-level changes; pooled
reductions are not guarantees of stable speedups. The fastest candidate still
remained 26.70% slower than C++ in this capture.

A separate latent stage-split comparison had a Python control of 49.2575 us
(P95 52.7626 us), versus 48.9350 us (P95 52.1313 us), a small -0.65% change.
It is not established as a stable win and is not included.

## Paired code evidence

The original initialization was:

```python
cosv = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
sinv = pl.tile.full([8, 64], dtype=pl.FP32, value=0.0)
```

The proposed entry allocates without initialization:

```python
cosv = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
sinv = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
```

Active rows are loaded before use, and RoPE consumes only their lower 32
lanes. The physical 64-element load stride is retained. Generated C++ has
zero `TEXPANDS` calls instead of two. The simulator did not show a smaller
whole window, so its vector pipeline does not explain the device gain alone.

The C++ grouped-maximum stage uses native intrinsics:

```cpp
vand(magnitude, (RegTensor<uint16_t>&)values, abs_mask, full16,
     MODE_ZEROING);
vcgmax(maximum, magnitude, full16, MODE_ZEROING);
```

Python uses four XOR partner exchanges; each stage combines lanes with:

```python
maximum = pl.reg.maximum(maximum, pl.reg.gather_lanes(maximum, partner), full)
```

This lowers to selects and maxima rather than `VCGMAX`. Python also writes
512 repeated FP32 scales per row instead of 32 compact values. This is a
16-fold difference in this logical scratch write, not total UB traffic.
Masked group reduction and compact-scale experiments failed PTOAS layout
selection with `VMI-LAYOUT-CONTRACT` frontier budget exhausted (`limit=288`).
Their performance is unmeasured; no compiler or runtime fix is included.

The C++ baseline publishes normalized latent before its in-place RoPE:

```cpp
normalize(input.data(), input.data(), weight.data(), rows);
set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
// Store latent before overwriting the same UB buffer.
set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
rotate(input.data(), input.data(), cosine.data(), sine.data(), rows);
```

Python exposes latent after the full fused RegScope:

```python
norm, payload, bf_factors = _process_batch(
    flat_inputs, weight, pl.reshape(cosv, [512]), pl.reshape(sinv, [512]), rows
)
```

The larger producer region explains later latent readiness. Its V-to-MTE3
wait is a real dependency. The stage-split diagnostic advances readiness but
adds UB reads and RegScope boundaries. Necessary DMA/result and cross-batch
buffer-reuse ordering are retained.

## In-core simulator evidence

CANN `msprof op simulator` replays all 72 tokens on one logical worker,
selecting core0.veccore0, not the eight-worker device schedule. Times are
relative to the first Scalar event.

| Implementation | Window (us) | First latent DMA (us) | Pre-MTE Scalar records | SMEM_BAR records |
| --- | ---: | ---: | ---: | ---: |
| C++ extern | 47.2283 | 4.2633 | 62 | 288 |
| Fused Python control | 111.4989 | 10.7083 | 105 | 2808 |
| Without rotary zero initialization | 111.8028 | 11.1822 | 105 | 2808 |
| Payload stride diagnostic | 111.0944 | 10.5200 | 105 | 2808 |
| BF16 output-scale stride diagnostic | 110.9267 | 10.6744 | 105 | 2808 |
| Owner arithmetic diagnostic | 111.1800 | 10.5894 | 105 | 2808 |
| Latent stage-split diagnostic | 113.6456 | 5.3850 | 105 | 3456 |

All replays perform 220 MTE2 transfers (93,760 bytes) and 216 MTE3 transfers
(96,768 bytes). Counts are simulator records, not necessarily retired machine
instructions. Overlapping intervals must not be summed as exclusive cost.
The split adds 4.9267 us of MTE3/vector overlap but increases the whole window
by 2.1467 us. RVECLD records rise from 3888 to 5040. Matching earlier latent
readiness alone does not close the gap. Raw JSON/CSV is preserved because the
installed cleaner cannot decode these CANN 9.3.0 binary SOURCE records.

## Validation and activation follow-up

The checked token counts are 0, 1, 7, 8, 9 and 72. Repeated slots, all-invalid
slots, inactive latent and untouched cache bytes are also checked.
Before switching model callers:

- Resolve grouped-max/compact-scale lowering and repeat device measurements.
- Audit PTOAS `PTOInsertVecScopeMemBar` and `PTOInsertSync` across full/tail
  paths and buffer reuse. Extra Scalar/MTE3 and per-store barriers have not
  been certified redundant.
- Repeat with source-pin-aligned PyPTO/runtime packages.
- Validate the operator in the full CI workflow; no such timing is claimed.

Local evidence is retained under `build_output/compressor_publish_20261112/`:
`pr-validation/` for final-entry samples, `alignment-20261112/` for independent
experiments, and individual build directories for source, IR, assembly and
deployed images. Generated artifacts are excluded from the PR.
