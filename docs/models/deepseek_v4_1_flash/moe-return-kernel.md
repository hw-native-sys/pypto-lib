# Decode MoE return kernel

The Flash TopK-6 decode path uses a stock ISA C++ kernel for expert return,
arrival polling, shared-plus-routed reduction, and packet-marker cleanup.
The measured 48- and 72-token kernel durations are close to the AscendC A5 MTE
`MoEDistributeCombine` implementation under the conditions below. This is a
kernel comparison, not a full-MoE or runtime speedup claim.

## Integration and numerical contract

[`decode_moe_combine.py`](../../../models/deepseek_v4_1_flash/decode_moe_combine.py)
provides both the production `combine_routed` call and its standalone golden
validation. It consumes the existing routed FFN's compact BF16 W2 output;
route weighting stays before activation quantization. Each active token has
six unique destination slots. The receiver accumulates the BF16 shared result
first, then slots 0 through 5 in FP32, and rounds once to BF16. Inactive physical
rows are zeroed.

The native body uses 64 AIV workers, initially split into 32 senders and 32
receivers. Senders participate in remaining token reductions after finishing
their sends. A row occupies 22 packets of 480 payload bytes and 32 marker bytes:
11,264 bytes, represented as 5,632 BF16 elements in the communication window.
The packet window **must be 512-byte aligned**. The host entries allocate it
first at the communication domain base; allocating small metadata before it
would not preserve that requirement. The receiver clears consumed markers
inside the kernel. The next input-dispatch handshake fences peer consumption
before the window is reused. The standalone entry provides an explicit round
release for the same purpose.

The decode MoE, layer, forward, and recurrent DSpark target entries use this
packed window. DSpark's
TopK-3 drafter keeps the existing scatter/reduce path and raw window. Prefill
continues using its existing scatter path. The FFN arithmetic is shared.
The window-context implementation accommodates the existing EP2/4/8
configuration range; the device evidence here is EP2, not an EP32 execution.

## Matched 48-token measurements

Ascend950DT_9581, physical EP2, TP1, 48 tokens per rank, hidden size 5,120,
TopK 6, 12 experts per rank, 24 valid rows per expert (288 per rank).
Both paths use BF16 inputs/outputs, FP32 accumulation, and at most 64 AIV cores.
Shared input is zero. CANN receives FP32 unit route weights; the C++ input is
already weighted. CANN still performs its unit-weight multiplication.

Each cell is the median of 100 samples after 5 warmups, in microseconds.
Ranks are reported separately, without selecting the fastest rank.

| Remote / local expert rows per rank | C++ rank 0 | CANN rank 0 | C++ rank 1 | CANN rank 1 |
| --- | ---: | ---: | ---: | ---: |
| 9 / 279 | 17.214 | 15.997 | 14.055 | 15.844 |
| 144 / 144 | 58.501 | 58.302 | 49.660 | 57.438 |
| 279 / 9 | 100.009 | 102.088 | 95.197 | 100.913 |

The 144-row case is uniform EP2 routing. The 9-row case models the average
traffic to one of 32 expert ranks: `48 * 6 / 32 = 9` rows. The 279-row case
concentrates the total EP32 remote volume into one peer. These last two cases
are traffic models; neither reproduces EP32 fan-out, topology, or latency.

### Timing boundary

- **C++:** earliest AIV kernel-body start to latest AIV kernel-body end in a
  dispatch, from the raw Level-1 device trace. This includes internal polling,
  reduction, and protocol cleanup. Runtime setup, scheduling, and retirement
  outside the body are excluded.
- **CANN:** raw non-mix AIV context start/end counters, converted with each
  capture's saved HWTS frequency. The intervals were checked against profiler
  `aiv_time` and database records, joining repeated task IDs by timestamp too.
  This is not the profiler's enclosing `TaskDuration` or `WaitTime`.

Both measurements cover the communication kernel's execution, including its
internal wait for arrivals. Their instrumentation boundaries are not identical
instructions, so the table supports comparable kernel latency rather than a
precise percentage speedup. The supplementary maximum per-core C++ execution
metric is retained in the data file and is not substituted for either column.
An initial inter-rank barrier is outside both measured kernels.

## Matched 72-token measurement

The additional uniform EP2 case has 72 active tokens, 432 valid expert rows per
rank, 36 rows per expert, and 216 remote rows. Dtypes, accumulation, maximum
64-AIV budget, warmup/sample counts, source implementation, and timing boundary
match the 48-token comparison. Both implementations ran on devices 4 and 5.

| Case | C++ rank 0 | CANN rank 0 | C++ rank 1 | CANN rank 1 |
| --- | ---: | ---: | ---: | ---: |
| 72 active tokens, 216 remote rows | 82.321 | 80.739 | 73.420 | 80.803 |

The production C++ path has 80 physical output rows and zeros the final eight;
the CANN fixture has 72 physical rows. Both return the same 72 valid BF16 rows.
The C++ interval includes this padding work. The same native body supports both
token counts; there is no separate tuned 48-token implementation. This was the
only new matched performance condition collected for the PR; the existing
48-token captures were retained.

The composed MoE passed its original golden tolerances for active-token counts
`[48, 48]`, `[72, 72]`, and `[0, 37]`. Standalone 48- and 72-token routing cases
passed exact two-round checks, including different shared/payload data in the
second round and reuse of the communication window.

## Sources and retained evidence

[`moe-return-kernel-results.json`](moe-return-kernel-results.json) contains the
per-sample durations, both ranks, raw-evidence hashes, configuration, and source
revisions. The 48-token measurements were reused, not recollected for this PR.
Recompiling this PR's native source produced the same 10,784-byte `.text` as the
measured kernel, with SHA256
`180f7d7fe6cf4de34f84d3f2122297273e2cd86c86c6932ee266dcc9f686d918`.

Communication-context offsets are derived with `offsetof` from the compiling
runtime's `CommContext` header. This supports the newer context layout while
retaining byte-identical executable sections under the measured runtime. The
JSON records both measured and current source hashes; the timing values refer
to the explicitly recorded toolchain.

The reference is the A5 MTE class in
[ops-transformer at 02eb562f](https://github.com/Ascend/ops-transformer/blob/02eb562f0ea3657498a9e7801b1e03e9f7258bea/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h),
also used by V3. It was built through that repository's official fast-launch
example, using its explicit host tiling rather than the ACLNN tiler. The local
build adapts the host field name to `enablePreReduce=false` and links the
available runtime interfaces; it does not change the kernel algorithm. The
installed compiler/runtime components report CANN 9.2.0. The JSON records the
PyPTO, runtime, and PTO ISA revisions as well.

## Validation and reproduction

A5 device execution is required: the pinned CPU simulator does not implement
the VF scope, direct DMA, prefix scan, and cache-control intrinsics used here.
The standalone entry rejects `--sim` explicitly.

The standalone entry checks two consecutive calls using the same communication
window and different payload/shared inputs. It covers distributed routing,
concentrated experts, empty experts, unequal active-token counts, all-zero
counts, poisoned unused capacity, and zeroed padding. Outputs are compared
exactly against the ordered FP32/BF16 reference.

Run from the repository root with an A5 environment and an allocated device
pair (replace `0,1` with the assigned devices):

```bash
python models/deepseek_v4_1_flash/decode_moe_combine.py -p a5 -d 0,1 --tokens 48
python models/deepseek_v4_1_flash/decode_moe_combine.py -p a5 -d 0,1 --tokens 72
python models/deepseek_v4_1_flash/decode_moe.py -p a5 -d 0,1 --experts-per-rank 12 --num-tokens 48
python models/deepseek_v4_1_flash/decode_moe.py -p a5 -d 0,1 --experts-per-rank 12 --num-tokens 72
```

The return entry uses physical extents 48 and 80 respectively. The composed
MoE entry retains its existing 80-row physical extent. Its reduced-expert
validation gate has 24 total experts at EP2/E12; this does not change the
384-expert production gate contract or measure gate performance.

For the uniform communication fixture and a new kernel capture:

```bash
PYPTO_BENCH=1 PYPTO_BENCH_ROUNDS=100 PYPTO_BENCH_WARMUP=5 \
  python models/deepseek_v4_1_flash/decode_moe_combine.py -p a5 -d 0,1 \
  --tokens 48 --case uniform --benchmark-fixture --enable-chip-swimlane 1
```

Use `--case pair` or `--case remote` for the other traffic models. The first
native return task in each dispatch uses the recorded benchmark payload; the
second changes the payload to exercise reuse and must be kept separate when
extracting timings. Inspect raw kernel-body timestamps rather than the
harness's runtime timing headline. The 48-token benchmark fixture is
byte-identical to the retained measurements. At 72 tokens, use `--case uniform`
for 216 remote rows and 36 received rows per expert; the closest integer
single-peer model is 14 rows, whereas the EP32 expectation is 13.5.

For a CANN reproduction, use the pinned fast-launch example and the same
`build_case("uniform", benchmark=True)` payload/route data. Run dispatch once
outside timing to obtain the implementation's paired assist metadata, map each
expanded row back to its source token and TopK slot, then replace its payload
with that slot's BF16 value. Set unit expert scales, zero shared input, 24 total
experts, and communication quantization mode 0. Capture combine-only graph
replays after an inter-rank start barrier, with 5 warmups and 100 measured calls.
Check the first and final output exactly. Keep dispatch and metadata preparation
outside the reported combine interval.
