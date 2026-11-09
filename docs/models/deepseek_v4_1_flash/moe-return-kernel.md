# Decode MoE return kernel

The Flash TopK-6 decode path uses a stock ISA C++ kernel for expert return,
arrival polling, shared-plus-routed reduction, and packet-marker cleanup.
The tables compare the original split DSL, initial C++ implementation, and
revised C++ implementation with AscendC A5 MTE `MoEDistributeCombine`.
The revised source removes redundant work, but per-rank latency changes are
mixed and do not establish a universal speedup. This is a
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

The **split DSL** column reports the original split DSL's stages separately:
S = scatter, N = notify, W = defer-wait registration, R = reduce. Each value
is the median of the stage's maximum per-core body execution time. W is not
the communication-wait duration. The deferred wait takes place outside these
bodies and overlaps runtime scheduling, so the retained traces do not provide
a comparable complete-return kernel latency for the original path. Do not
sum these stages or calculate a before/after speedup from this table.

Both C++ columns and CANN cover their respective complete fused kernel
execution intervals, as defined below. The original M48 DSL uses 64 scatter
workers, 24 reduce workers, and one each for notify/wait; both fused kernels
use 64 AIV. The common maximum budget is 64 AIV, not identical work assignment.

| Remote / local rows per rank | Rank | Split DSL S / N / W / R | Initial C++ | Revised C++ | CANN |
| --- | --- | --- | ---: | ---: | ---: |
| 9 / 279 | 0 | 3.023 / 1.522 / 1.178 / 4.491 | 17.214 | 20.476 | 15.997 |
| 9 / 279 | 1 | 2.967 / 1.569 / 1.316 / 4.532 | 14.055 | 13.632 | 15.844 |
| 144 / 144 | 0 | 35.096 / 1.551 / 1.255 / 5.271 | 58.501 | 58.500 | 58.302 |
| 144 / 144 | 1 | 35.511 / 1.518 / 1.249 / 5.217 | 49.660 | 49.068 | 57.438 |
| 279 / 9 | 0 | 74.558 / 1.519 / 1.274 / 5.419 | 100.009 | 102.745 | 102.088 |
| 279 / 9 | 1 | 75.628 / 1.517 / 1.208 / 5.397 | 95.197 | 91.394 | 100.913 |

For small M48 traffic, the larger of the two rank medians increased from
17.214 to 20.476 us; this is an observed regression, not a speedup. The initial
capture used devices 0/1 on November 6, and the revised capture used devices
1/3 on November 9. Both ranks and the original samples remain visible.

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

## Matched 72-token measurements

Both M72 cases have 72 active tokens, 432 valid expert rows per rank, and 36
rows per expert. The small-traffic case has 14 remote / 418 local rows; the
uniform EP2 case has 216 remote / 216 local rows. Dtypes, accumulation, maximum
64-AIV budget, warmup/sample counts, and kernel timing definitions match M48.
The original DSL uses 64 scatter, 40 reduce, and one notify/wait worker each.

| Remote / local rows per rank | Rank | Split DSL S / N / W / R | Initial C++ | Revised C++ | CANN |
| --- | --- | --- | ---: | ---: | ---: |
| 14 / 418 | 0 | 3.808 / 1.395 / 1.202 / 4.615 | 16.997 | 20.977 | 17.798 |
| 14 / 418 | 1 | 3.798 / 1.731 / 1.361 / 4.611 | 22.805 | 17.148 | 17.898 |
| 216 / 216 | 0 | 55.042 / 1.468 / 1.256 / 5.341 | 82.321 | 82.204 | 80.739 |
| 216 / 216 | 1 | 56.716 / 1.763 / 1.313 / 5.330 | 73.420 | 72.848 | 80.803 |

For small M72 traffic, rank 1 changed from 22.805 to 17.148 us, while rank 0
changed from 16.997 to 20.977 us. The larger rank median therefore changed
from 22.805 to 20.977 us, not to 17.148 us. The slower rank changed; the
measurements do not establish that a fixed 5-us cost was eliminated or that
both ranks are consistently below 20 us. The initial M72 small-traffic capture used devices 1/2; the revised capture
used devices 1/3, in a separate run.

The C++ and original DSL paths have 80 physical output rows and zero the final
eight; the CANN fixtures have 72 physical rows. All return the same 72 valid
BF16 rows. The C++ interval includes this padding work. The same native body
supports both token counts; there is no separate tuned M48 implementation.

All initial C++ and CANN samples were retained. The current revised C++ was
captured once per listed condition on November 9, on devices 1/3, with 5
warmups and 100 samples per rank. No original C++ or CANN performance
condition was recollected. The older M72 uniform C++/CANN captures used
devices 4/5. Device and source identities are retained per capture.

The raw level-1 trace contains body start/end per physical core, not logical
block roles, packet-arrival times, or poll/copy/reduction phase boundaries.
For the initial M72 small-traffic rank 1, the maximum-core execution median
was 21.871 us, versus its 22.805-us kernel interval. Thus the gap was not only
an envelope around shorter cores. This evidence still cannot assign the
roughly 5-us CANN gap entirely to communication waits or any individual
source operation. Kernel-internal waiting remains included in every column.

The composed MoE passed its original golden tolerances for active-token counts
`[48, 48]`, `[72, 72]`, and `[0, 37]`. Standalone 48- and 72-token routing cases
passed exact two-round checks, including different shared/payload data in the
second round and reuse of the communication window.

## Sources and retained evidence

[`moe-return-kernel-results.json`](moe-return-kernel-results.json) contains the
per-sample durations, original DSL stage timings, both ranks, raw-evidence
hashes, configuration, and source revisions. The original DSL was extracted
from pypto-lib revision `2562a862f917ed76b67cdb064d3a9b74e156e8b4`, with the
preceding FFN outside measurement and an explicit start barrier. M72 changes
only its shapes and active-token count. These are historical baseline
measurements, not a new benchmark of current main. The 48-token measurements
were reused, not recollected for this PR.
The initial C++ reference had a 10,784-byte `.text` with SHA256
`180f7d7fe6cf4de34f84d3f2122297273e2cd86c86c6932ee266dcc9f686d918`.
That identity proof applies only to the initial implementation. The revised
kernel has different executable bytes, recorded under `revised_cpp`; its
source and runtime-loaded binaries were checked against the new build.
The original `cases` and `implementation` records remain unchanged.

Communication-context offsets still come from the compiling runtime's
`CommContext` header through `offsetof`. The dtype, 64-AIV budget, token
assignment, packet layout, input weights, and ordered accumulation contract
are unchanged by the revisions.

The reference is the A5 MTE class in
[ops-transformer at 02eb562f](https://github.com/Ascend/ops-transformer/blob/02eb562f0ea3657498a9e7801b1e03e9f7258bea/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h),
also used by V3. It was built through that repository's official fast-launch
example, using its explicit host tiling rather than the ACLNN tiler. The local
build adapts the host field name to `enablePreReduce=false` and links the
available runtime interfaces; it does not change the kernel algorithm. The
installed compiler/runtime components report CANN 9.2.0. The JSON records the
PyPTO, runtime, and PTO ISA revisions as well.

## AscendC detail audit

The comparison uses the pinned A5 MTE source in the measured unquantized mode.
General CANN features that are disabled in this fixture are not copied into
the fixed model kernel. The JSON retains the full detail audit.

| Detail | Source comparison | Current change |
| --- | --- | --- |
| Poll-flag initialization | Initial C++ cleared 1,152 int32 elements each poll; CANN clears only the comparison tail | Clear only 0-48 tail elements, retaining the V-to-MTE2 ordering fence |
| Sender packet markers | Initial C++ issued up to eight separate fills; CANN fills used buffers together | One fill of the used packet slots; event lifetime is unchanged |
| Output padding | C++ has padded physical rows; the CANN M72 fixture does not | Only blocks owning padding rows initialize/store zeros; output shape is unchanged |
| Route descriptors | CANN receives prepared triples; C++ reconstructs them inside the kernel | Merge contiguous route reads within each expert across source ranks; 54 to 43 route DMA calls per rank for M72 small traffic |
| Core split and token order | Both use send/receive roles, two-token progression, ordered slot prefixes and similar sender permutations | Preserve these policies; a different send ordinal alone does not prove a timing cause |
| Shared and weighted inputs | C++ starts from shared and consumes already-weighted rows; CANN adds shared later and multiplies route scales | Preserve the model's numerical/input contract; matched fixtures keep shared zero and CANN scales one |
| Buffer reuse and completion | The persistent C++ caller and CANN launch have different state/fence ownership | Preserve descriptor invalidation, draining and peer-release requirements |

The initialization-only candidate and a candidate without marker batching
were also checked, using separate source revisions and no baseline recapture.
Neither established a universal latency improvement. The final source adds
descriptor coalescing and retains the explicit mixed per-rank measurements
above; no source-level work count is converted into an unmeasured us saving.

The descriptor change was checked on 1,881 CPU ranges across EP2/4/8 metadata,
including empty experts and partial source segments. This is an index-mapping
check, not an EP4/8 device-performance claim. The revised native kernel passed
exact two-round device checks for all six routing cases at both token counts.
Six additional exact cases cover the production extent of 80 physical rows
with 48 active tokens, including mixed active counts `[0, 37]`.

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
