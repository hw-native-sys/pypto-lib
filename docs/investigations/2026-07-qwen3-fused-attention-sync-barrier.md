# Qwen3 Fused RoPE-Attention Sync Barrier Decision

- Date: 2026-07-20
- Status: accepted for the PR796 C220/A3 implementation
- Pull request: [pypto-lib#796](https://github.com/hw-native-sys/pypto-lib/pull/796)
- Tested parent: `f6478a99e3046ace2a72cb70822ff21fb9a32145`

## Decision

Remove the two `dsb(DSB_DDR)` calls that surrounded `SyncAll<false>` at the
boundary between the fused RoPE producer and attention consumer. Keep the
explicit `PipeBarrier<PIPE_ALL>` and the mixed AIC/AIV `SyncAll<false>`.

This decision applies to the current data path:

```text
AIV TSTORE/MTE3 -> GM -> mixed-core SyncAll<false> -> AIC MTE2 -> attention
```

It does not apply to the separate metadata acquisition path. That path uses
cache-backed scalar reads after `dcci`, so its `dsb(DSB_DDR)` remains.

## Why the barriers were reconsidered

The removed barriers were added under the assumption that `SyncAll<false>`
synchronized core arrival but did not publish completed MTE3 writes to the AIC
consumer. That assumption was stronger than the implementation and device
evidence supported for this pure DMA path:

- The C220 CANN 9.0.0 `SyncAllImpl<false>` implementation begins with
  `PipeBarrier<PIPE_ALL>()` before its mixed AIC/AIV FFTS handshake.
- The CANN 9.0.0 matmul-all-reduce implementation has an analogous sequence:
  AIV `DataCopyPad` writes converted bias to GM, `SyncAll<false>` synchronizes
  the mixed cores, and the AIC matmul consumes that GM buffer without an
  intervening DDR barrier.
- The generated Qwen RoPE producer writes Q and paged K/V through `TSTORE`, and
  the attention consumer reads through the MTE path rather than cache-backed
  scalar GM stores.

The relevant CANN sources are, relative to the CANN installation:

- `aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_sync_impl.h`
- `opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/`
  `matmul_all_reduce/common.h`
- `opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/`
  `matmul_all_reduce/arch32/matmul_all_reduce_base.h`

These observations motivated a production-candidate test without the two DDR
barriers; device validation, rather than source analogy alone, determines the
decision.

## Validation design

The candidate was tested on C220/A3 hardware with CANN 9.0.0. Each case ran the
fused 40-layer forward path plus LM head for batch 16 and one decode step. The
sequence cap was 3338. Max-length and varied-length fixtures covered seeds 1234,
2027, and 4093; seed 1234 was repeated three times in each length mode on the
same device.

This is the repository's synthetic full-depth validation harness, not a trained
Qwen3 checkpoint. It generates random inputs and single-layer weights, then
reuses those weights for all 40 layers before applying a generated LM head. The
stack exercises the fused producer/consumer path repeatedly and compounds a
stale or partial read, but it does not cover 40 distinct trained layers or
replace real-checkpoint model validation.

The pinned compiler dependencies were:

- PyPTO: `869d5b6518ea816f43769a8e3ab22831b0b214f4`; and
- PTO ISA: `83d01313d9bfc247c4b7c8bcf969d1019f0d106f`.

Every case used a separate program build directory so same-second JIT directory
reuse could not mix binaries. The matrix therefore contains ten independent
build-and-run validations:

| Case | Classification | Argmax | Sample | Max abs | Mean abs | P99 abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| max-s1234-r1 | CLEAN | 16/16 | 16/16 | 0.02006054 | 0.00297024 | 0.00980854 |
| varied-s1234-r1 | TIE_ONLY | 15/16 | 16/16 | 0.02182579 | 0.00304710 | 0.01005483 |
| max-s2027 | CLEAN | 16/16 | 16/16 | 0.02037764 | 0.00290220 | 0.00948429 |
| varied-s2027 | CLEAN | 16/16 | 16/16 | 0.02007198 | 0.00293272 | 0.00954491 |
| max-s1234-r2 | CLEAN | 16/16 | 16/16 | 0.02162695 | 0.00297850 | 0.00986169 |
| varied-s1234-r2 | CLEAN | 16/16 | 16/16 | 0.02250862 | 0.00302958 | 0.01001343 |
| max-s4093 | CLEAN | 16/16 | 16/16 | 0.01956129 | 0.00287545 | 0.00934699 |
| varied-s4093 | CLEAN | 16/16 | 16/16 | 0.02059984 | 0.00303412 | 0.00990486 |
| max-s1234-r3 | CLEAN | 16/16 | 16/16 | 0.02165055 | 0.00296911 | 0.00983972 |
| varied-s1234-r3 | CLEAN | 16/16 | 16/16 | 0.02240777 | 0.00304791 | 0.01007125 |

`Sample` compares the sampled token with the kernel argmax. Continuous-value
validation uses the existing `torch.isclose` thresholds of `rtol=5e-2` and
`atol=5e-2`.

Aggregate results:

- 9 CLEAN, 1 TIE_ONLY, 0 HARD_FAIL;
- strict kernel/reference argmax: 159/160;
- sampled token versus kernel argmax: 160/160;
- logits within tolerance: 24,330,240/24,330,240;
- NaN or Inf values: 0;
- max-absolute-error average/median: 0.02106910/0.02111340; and
- mean-absolute-error average/median: 0.00297869/0.00297437.

The campaign process returned nonzero because the strict argmax check treats
any mismatch as a failure. The mismatch was diagnosed before accepting the
candidate rather than being discarded from the result.

## Near-tie diagnosis

The only mismatch was row 14 of `varied-s1234-r1`:

- kernel top-2 IDs: `[15950, 19261]`, margin 0.00527430;
- reference top-2 IDs: `[19261, 15950]`, margin 0.00008821; and
- row max/mean absolute error: 0.02124333/0.00303630.

The candidates are each other's top-2. For the identical seed-1234 varied
fixture, all three kernel runs returned ID 15950 on that row. Only the first
reference run selected ID 19261; the next two reference runs selected 15950.
Kernel argmax and sampled-token arrays were otherwise identical across all
three repeats.

Neither path was bitwise deterministic. Across same-input repeats, the maximum
kernel/reference logit drift was 0.01194/0.01222 in max mode and
0.01139/0.01204 in varied mode. Comparable drift on both paths, full continuous
agreement, mutual top-2 candidates, and a reference margin below `1e-4` support
classifying this result as a near-tie rather than stale-data corruption.

## Decision boundary

The ten full-depth runs found no hard precision regression after removing the
two DDR barriers. That is sufficient to accept the no-DSB candidate for the
current PR796 synchronization and data-publication path. It is not a general
statement that a mixed-core barrier makes every GM access coherent, nor is the
synthetic harness a real-checkpoint accuracy qualification.

Reopen this decision if any of the following changes:

- the producer begins using direct scalar or cache-backed GM stores;
- the consumer stops using the coherent MTE load path;
- the AIC/AIV participant topology or phase ownership changes;
- the target architecture or CANN `SyncAllImpl` implementation changes; or
- repeated full-stack validation shows nonfinite values, tolerance failures,
  sampled-token instability, or non-tie argmax mismatches.

When cache-backed scalar publication is involved, review the required `dcci`
and `dsb` sequence independently. Do not use this record to remove the metadata
barrier or barriers from unrelated kernels.

## Representative validation command and retained evidence

Use a unique build directory and a data path without an `in/` subdirectory so
the seed controls the generated fixture:

```bash
case_name=max-s1234-r1
device_id=0
case_dir="build/no-dsb-validation/${case_name}"
PTO2_MANUAL_MAX_SEQ=3338 \
PYPTO_PROG_BUILD_DIR="${case_dir}/programs" \
python models/qwen3/14b/decode_fwd.py \
  -p a2a3 -d "${device_id}" --validate-fwd --fwd-layers 40 \
  --decode-steps 1 --seed 1234 \
  --data-dir "${case_dir}/fresh-generated-fixture" --max-seq
```

Omit `--max-seq` for a varied-length case and use a different `case_name` for
every build-and-run entry.

This production command reports the standard argmax, sampled-token, close-ratio,
and maximum-error checks. The per-row, distribution, and repeat-drift metrics in
this record came from temporary host-side dump instrumentation that ran after
device execution and was removed from the production candidate.

Raw logits, reference outputs, compiler reports, and the aggregation script are
diagnostic artifacts and remain under the ignored
`build/pr796-no-dsb-precision/` directory on the validation workspace. They are
not production inputs and are not committed.

For the reusable timestamp method used to separate producer, collective, and
consumer intervals, see
[`incore-timestamp-profiling.md`](../incore-timestamp-profiling.md).
