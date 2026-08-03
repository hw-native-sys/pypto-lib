# DeepSeek V4 Flash Decode-to-MTP Performance Design

## Status

The conversational design was approved on 2026-07-30. The expert-topology
amendment was approved later that day: reuse only the comparison branch's
128-global / 16-local expert topology while preserving the complete
decode-to-MTP graph described here.

The implementation branch is
`perf/dsv4-ascendc-decode-mtp-compare4`, based on
`upstream/dsv4-ascendc-decode-compare2` commit `11cba4e`. The compare4
change restores the complete decode-to-MTP graph boundaries that the
narrower compare2 benchmark removes.

## Objective

Add a reproducible PyPTO 3.0 benchmark for the complete DeepSeek V4 Flash
decode-to-MTP path with EP=8, LM-head TP=4, 128 global routed experts, and 16
local routed experts per EP rank. The comparison interval starts at the first
main-decode embedding kernel and ends at the final cast after the MTP LM head.

The benchmark must execute the main 43-layer decode, the decode-to-MTP
handoff, and the depth-one MTP decode in one distributed invocation. It must
not estimate the result by adding measurements from two standalone programs.

## AscendC Reference

The local `trace_view_a3_decode.json` file contains three repeated NPU0
executions. Its direct device-kernel intervals are:

| Round | Main decode | Handoff gap | MTP | Full interval |
|---|---:|---:|---:|---:|
| 1 | 46.325086 ms | 3.994020 ms | 2.322726 ms | 52.641832 ms |
| 2 | 46.401346 ms | 4.139383 ms | 2.295967 ms | 52.836696 ms |
| 3 | 46.317065 ms | 4.202484 ms | 2.195244 ms | 52.714793 ms |

Round 3 is both the last steady-state sample and the median full interval. It
is the initial reference value.

The exact boundaries are:

- Start: Model 48
  `aclnnEmbedding_GatherV2AiCore_GatherV2`.
- End: Model 47 task 66
  `aclnnInplaceCopy_CastAiCore_Cast`, including its duration.

The trace contains four casts with the final cast name inside each MTP model.
Only task 66 is the endpoint.

The handoff is real work rather than idle time. It contains sampling,
acceptance checks, synchronization, an HCCL all-reduce, buffer copies, and the
next model launch. The AscendC trace records NPU0 only and does not independently
prove the reported EP=8/TP=4 topology.

Every repetition contains 44 `MoeGatingTopKHash` kernels: one for each of the
43 main layers and one for the MTP layer. The comparison therefore needs an
explicit trace-hash routing mode in addition to the normal model routing mode.

The reference trace is an input artifact and must not be committed.

## Existing PyPTO Components

`models/deepseek/v4-flash/decode_fwd.py` already implements:

- the 43-layer main decode;
- `pre_hc_hidden_out` from the final main-model MoE;
- HC head and final RMSNorm;
- TP-grouped LM head;
- fused greedy sampling into `sampled_ids`.

`models/deepseek/v4-flash/decode_mtp.py` already implements:

- embedding and accepted-window packing;
- MTP projection;
- pure SWA attention;
- one MoE layer;
- HC head and final RMSNorm;
- TP-grouped LM head and greedy sampling.

The main `pre_hc_hidden_out` and MTP `main_pre_hc_hidden` tensors both have
per-rank shape `[8, 4, 4096]` and dtype FP32. They can share one resident
buffer without a host round trip.

The standalone MTP driver currently initializes `main_pre_hc_hidden`,
accepted metadata, and token IDs independently. Running it after the
standalone main driver would not create a real dataflow edge.

## Chosen Architecture

Add a new composite driver:

`models/deepseek/v4-flash/decode_fwd_mtp.py`

It will expose one `@pl.jit.host` entry with three ordered phases:

```text
submit decode_fwd on every rank
              |
              v
submit the compact handoff on every rank
              |
              v
submit mtp_decode_layer on every rank
```

All main-decode submissions occur before any MTP submission. Per-rank tensor
dependencies order the handoff after the main sampled IDs and pre-HC hidden
state. MTP communication naturally waits for all participating ranks. Phase
one does not add an artificial sleep or a synthetic delay to reproduce the
AscendC gap.

The main and MTP stages use separate MoE and LM-head communication windows.
This avoids reusing completion counters before their clear operations are
proven complete and keeps persistent benchmark rounds safe.

The composite entry reuses the main driver's greedy-sampling output for its
handoff. Its MTP child uses a logits-only LM-head endpoint: it stops after the
full-vocabulary logits are assembled and does not run a second greedy sampler.
The standalone MTP driver retains its existing sampled-output behavior.

The existing standalone drivers remain runnable with their current default
model behavior.

## Expert Topology

The comparison topology is EP=8, LM-head TP=4, 128 global routed experts, and
16 local routed experts per EP rank. Main decode and MTP use the same topology.
The import-time expert scaling matches the
`dsv4-ascendc-decode-compare2` branch:

```python
config.FLASH.n_routed_experts // 16 * EP
```

At EP=8, the model's base 256 experts become 128 global experts and therefore
16 local experts per rank. The current branch uses compare2 as its Git base
and reuses this topology. Compare4 intentionally overrides compare2's
host-prepared embedding/metadata path, deleted main-model tail, and
standalone-only timing structure.

With eight ranks, eight active tokens per rank, and top-k six, trace-hash mode
creates 384 active routes. The rank-aware round-robin fixture distributes
those routes exactly three times over each of the 128 global experts.

## Handoff Contract

Add a focused device-side handoff helper in
`models/deepseek/v4-flash/decode_input_pack.py`. It consumes:

- main `sampled_ids[:, 0]`;
- main `position_ids`;
- deterministic `accepted_counts`, with each value restricted to 1 or 2;
- deterministic `tail_slot_ids`;
- resident tail-token and tail-position pools.

For each two-row batch window:

- acceptance count 2 commits both main sampled-token rows;
- acceptance count 1 commits the previous resident tail followed by the first
  main sampled-token row;
- the last committed token and position replace that slot's resident tail.

The existing `pack_mtp_hidden` helper applies the matching rule to the
main-model pre-HC hidden state and the resident tail-hidden pool.

The handoff writes resident per-rank MTP token IDs and positions. Token IDs are
INT64 and positions are INT32, matching `mtp_decode_layer`.

The initial benchmark fixture uses the existing deterministic acceptance
pattern `[1, 2, 1, 2]` and tail slots `[0, 1, 2, 3]` on every rank. This is a
performance fixture, not a production speculative scheduler.

## Routing Modes

The composite driver provides:

- `model`: use each layer's real routing identity. The first three layers use
  hash routing and later layers, including MTP layer 43, use the learned gate.
- `trace-hash`: preserve the actual layer's weights, caches, attention type,
  MoE epoch, and output head while presenting hash-routing identity to the gate
  for all 44 MoE calls.

The target AscendC comparison runs in `trace-hash` mode. A rank-aware
round-robin `tid2eid` fixture distributes active routes across the eight EP
ranks as evenly as the fixed route count permits.

The MTP child keeps its handoff token IDs for embedding, but uses a separate
`routing_input_ids` tensor for hash-table lookup. In the standalone fixture
this tensor is rank-stacked `0..7`; in the composite it reuses the original
main-decode `input_ids`, which have the same sequence. This separation is
required because sampled handoff IDs are not consecutive and would otherwise
destroy the intended EP8 route balance. It adds no composite host parameter.

Trace-hash mode must not:

- change the comparison topology of 128 global routed experts;
- change the per-rank local expert count of 16;
- delete either HC head, RMSNorm, or LM head, or remove the main handoff sampler;
- change the standalone MTP model's sampled-output behavior;
- replace actual layer IDs used for weight or cache slicing;
- change the attention layer schedule.

Although the branch is based on `dsv4-ascendc-decode-compare2`, compare4
restores the first embedding boundary and the main decode tail. The complete
decode-to-MTP graph therefore remains the comparison boundary while the
compare2 history supplies the shared baseline and expert topology.

## Tensor Ownership and Residency

The composite fixture deliberately shares:

- embedding weight;
- RoPE cosine and sine tables;
- TP-sharded LM-head weight.

It keeps distinct, prefixed tensors for:

- MTP projection weights and scales;
- MTP SWA weights and KV cache;
- MTP MoE weights;
- MTP HC-head and final-norm weights;
- main logits and sampled IDs, plus composite MTP logits;
- main and MTP communication windows and signal counters.

Main caches, MTP caches, static weights, the pre-HC handoff buffer, and the
tail pools remain device-resident across benchmark rounds. Generated build
artifacts stay under `build_output/` and are not committed.

## Command-Line Contract

The composite driver supports the existing platform, device, cache-size,
start-position, compile-only, runtime-directory, and DFX controls used by the
standalone drivers. Its target command is equivalent to:

```bash
PYPTO_BENCH=1 \
python models/deepseek/v4-flash/decode_fwd_mtp.py \
  -p a2a3 \
  --ep 8 \
  --tp 4 \
  --routing-mode trace-hash \
  --start-pos 8192 \
  --num-tokens 8 \
  -d 0,1,2,3,4,5,6,7
```

This comparison driver intentionally accepts only EP=8 and TP=4. It also
rejects a device list shorter than eight entries.

## Measurement Contract

The primary direct comparison is a CANN system trace:

1. Warm up the composite invocation.
2. Capture repeated steady-state invocations without L2 swimlane enabled.
3. On each rank, start at the first main task that reads `input_ids` and
   `embed_weight` for `pack_x_hc`. Main metadata work that completes before this
   task is outside the reference boundary.
4. End at the last task that writes the assembled full-vocabulary MTP logits.
   This is the semantic equivalent of the AscendC final MTP cast; the composite
   path has no post-MTP greedy-sampling task.
5. Report NPU0 separately for direct comparison with the provided NPU0 trace.
6. Report the maximum interval across all PyPTO ranks as the distributed
   critical-path bound.
7. Report median and mean over at least three steady samples.

`PYPTO_BENCH effective_us` remains a regression metric. For a composite L3
program it sums child effective windows on each rank and can exclude idle
space between child dispatches, so it is not the primary cross-framework
number.

`host_union_mean_us` and per-slot decode/handoff/MTP samples are diagnostics.
L2 swimlane capture runs separately because it executes mutable workloads more
than once and does not provide one cross-dispatch system envelope.

## Sampling Limitation

The AscendC gap performs stochastic sampling and collective acceptance work.
The PyPTO main driver currently performs fused greedy sampling, and PyPTO's
device RNG support is not available for A3.

Phase one measures the natural PyPTO greedy decode-to-MTP chain and labels the
sampling semantic difference in every comparison report. It does not add fake
delay, host sleep, or unrelated kernels to imitate the AscendC gap.

Exact stochastic parity is a separate follow-up. It requires a confirmed
sampling algorithm, externally supplied random noise for A3, and an explicit
acceptance/all-reduce implementation.

## Validation

Validation proceeds from small to full scope:

1. Unit-test the handoff fixture and both acceptance cases against a Torch
   reference, including resident tail updates.
2. Verify that default `model` routing preserves existing standalone behavior.
3. Verify that `trace-hash` selects hash routing without changing expert
   dimensions or deleting tail components, and assert 128 global / 16 local
   experts.
4. Compile the composite graph at the target EP=8/TP=4, 128/16 topology.
5. Run an EP=8/TP=4 device smoke test and check finite outputs.
6. Run short warmup/timing loops, then the full benchmark loop.
7. Capture a system trace and verify the start/end kernels, 43+1 MoE stages,
   both LM heads, and the final MTP logits write with no later sampler.
8. Run repository lint and relevant golden-harness tests.

The current shell environment is not a valid baseline for the golden-harness
suite: the installed `pypto` package lacks `pypto.ir` and
`pypto.runtime.debug`. Implementation starts by activating the repository's
pin-compatible development environment and rerunning the baseline tests.

## Error Handling

Fixture construction validates shapes, dtypes, accepted counts, tail slots,
the exact EP=8/TP=4 and 128-global / 16-local topology, and device count before
compilation.

The process exits nonzero when compilation, dispatch, finite-output checks, or
trace-boundary validation fails. Performance summaries are emitted only after
a successful smoke run.

## Non-Goals

Phase one does not:

- implement a production speculative scheduler;
- reproduce AscendC stochastic sampling exactly;
- modify `pypto-serving`;
- change model dimensions or quantization;
- make expert topology configurable beyond the approved 128/16 comparison;
- optimize individual kernels before the full-chain baseline exists;
- commit profiler traces, logs, generated weights, or build output.
