# Decode MoE return kernel

The Flash TopK-6 decode path uses a stock ISA C++ kernel for expert return,
arrival polling, shared-plus-routed reduction, and packet-marker cleanup.
The tables compare the final C++ implementation with AscendC A5 MTE
`MoEDistributeCombine`. Both columns report complete kernel execution
intervals, with the instrumentation differences described below. This is a
kernel comparison, not a full-MoE or runtime speedup claim.

## Integration and numerical contract

[`decode_moe_combine.py`](../../../models/deepseek_v4_1_flash/decode_moe_combine.py)
provides both the production `combine_routed` call and its standalone golden
validation. It consumes the existing routed FFN's compact BF16 W2 output;
route weighting stays before activation quantization. Each active token has
six unique destination slots. The receiver accumulates the BF16 shared result
first, then slots 0 through 5 in FP32, and rounds once to BF16. Only rows
`[0, active)` are written; inactive physical output rows are left unchanged.
The composed MoE passes `active` to HC post, which does not load inactive
sublayer rows and uses zero sublayer contribution for their residual mixing.
The allocated capacity remains aligned, without a separate zero-fill kernel.

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

Final C++ and CANN both use 64 AIV cores. Each column covers its complete
fused kernel execution interval, including internal arrival polling and
protocol cleanup.

| Remote / local rows per rank | Rank | Final C++ | CANN |
| --- | --- | ---: | ---: |
| 9 / 279 | 0 | 16.752 | 15.997 |
| 9 / 279 | 1 | 12.734 | 15.844 |
| 144 / 144 | 0 | 57.686 | 58.302 |
| 144 / 144 | 1 | 49.174 | 57.438 |
| 279 / 9 | 0 | 100.891 | 102.088 |
| 279 / 9 | 1 | 92.975 | 100.913 |

For small M48 traffic, final C++ measures 16.752 / 12.734 us
on ranks 0 / 1. Report both ranks; selecting only the faster rank would hide
the rank imbalance. These are separate captures from the retained CANN data.

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

| Remote / local rows per rank | Rank | Final C++ | CANN |
| --- | --- | ---: | ---: |
| 14 / 418 | 0 | 16.886 | 17.798 |
| 14 / 418 | 1 | 19.902 | 17.898 |
| 216 / 216 | 0 | 71.150 | 80.739 |
| 216 / 216 | 1 | 82.757 | 80.803 |

For small M72 traffic, final C++ measures 16.886 / 19.902 us
on ranks 0 / 1, versus CANN's 17.798 / 17.898 us. The measurements do not
isolate the effect of padding-store removal from capture/device variability.

C++ retains 80 physical output rows; the CANN fixture has 72. Final C++
leaves the eight inactive rows untouched and returns the same 72 valid BF16
rows. The same native body
supports both token counts; there is no separate tuned M48 implementation.

Existing raw measurements remain archived. Final C++ was captured once per listed condition on November 9, on devices 2/3,
with 5 warmups and 100 samples per rank. M48 CANN captures used devices 0/1
on November 6; M72 uniform used 4/5, and M72 small traffic used 1/2.
Device and source identities are retained per capture. No existing
baseline performance condition was recollected; only the new implementation
was sampled. The table reports the active-prefix source, not an intermediate
C++ version.

The raw level-1 trace contains body start/end per physical core, not logical
block roles, packet-arrival times, or poll/copy/reduction phase boundaries.
It cannot attribute rank differences precisely to communication waits or
individual source operations. Kernel-internal waiting remains included in
the final C++ and CANN columns.

Before the active-prefix update, the composed MoE passed its original golden
tolerances for active-token counts `[48, 48]`, `[72, 72]`, and `[0, 37]`.
This update compiled the composed MoE and validated the changed combine and
HC kernels separately, as detailed below; full composed device validation
was not rerun locally. Standalone cases include changed second-round data
and reuse of the communication window.

## Sources and retained evidence

[`moe-return-kernel-results.json`](moe-return-kernel-results.json) contains
per-sample durations for both ranks, raw-evidence hashes, configuration, and
source revisions. The tables use `active_prefix_cpp` for the final C++ source,
runtime-loaded native text and samples, and retained CANN records from `cases`.
Earlier measurements remain archived for provenance and are not part of the
performance comparison. No existing reference measurements were recollected.

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
| Output capacity | C++ retains an aligned physical extent; the CANN M72 fixture uses 72 rows | Write only active rows. HC post masks inactive sublayer loads while preserving residual outputs; no padding stores or separate zero-fill task |
| Route descriptors | CANN receives prepared triples; C++ reconstructs them inside the kernel | Merge contiguous route reads within each expert across source ranks; 54 to 43 route DMA calls per rank for M72 small traffic |
| Core split and token order | Both use send/receive roles, two-token progression, ordered slot prefixes and similar sender permutations | Preserve these policies; a different send ordinal alone does not prove a timing cause |
| Shared and weighted inputs | C++ starts from shared and consumes already-weighted rows; CANN adds shared later and multiplies route scales | Preserve the model's numerical/input contract; matched fixtures keep shared zero and CANN scales one |
| Buffer reuse and completion | The persistent C++ caller and CANN launch have different state/fence ownership | Preserve descriptor invalidation, draining and peer-release requirements |

The final source includes descriptor coalescing and the initialization
changes above. Intermediate measurements remain available in the retained
evidence; no source-level work count is converted into an unmeasured saving.

The descriptor change was checked on 1,881 CPU ranges across EP2/4/8 metadata,
including empty experts and partial source segments. This is an index-mapping
check, not an EP4/8 device-performance claim. The active-prefix native kernel
passed exact two-round device checks for
all six routing cases at 48/48, 48/80 and 72/80 active/physical extents.
Inactive output rows retain distinct first/second-round sentinels. Six exact
HC post checks cover active counts 0, 37, 48, 72, 80 and all rows; inactive
sublayer inputs contain NaNs, while expected outputs use zero sublayer
contributions and unchanged residual mixing. The ordinary HC post entry
also passed with the explicit all-row contract.

## Validation and reproduction

A5 device execution is required: the pinned CPU simulator does not implement
the VF scope, direct DMA, prefix scan, and cache-control intrinsics used here.
The standalone entry rejects `--sim` explicitly.

The standalone entry checks two consecutive calls using the same communication
window and different payload/shared inputs. It covers distributed routing,
concentrated experts, empty experts, unequal active-token counts, all-zero
counts, poisoned unused capacity, and preserved output-tail sentinels. Outputs
are compared exactly against the ordered FP32/BF16 reference.

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
