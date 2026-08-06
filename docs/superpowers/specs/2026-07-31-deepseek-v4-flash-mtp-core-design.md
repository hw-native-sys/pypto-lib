# DeepSeek V4 Flash MTP Core Performance Design

## Status

The conversational design was approved on 2026-07-31. This design adds an
isolated MTP-core benchmark next to the existing complete Decode-to-MTP Scheme
4 benchmark. It does not redefine or remove the complete benchmark.

## Objective

Measure only the following DeepSeek V4 Flash MTP compute chain:

```text
MTP projection -> SWA -> MoE -> HC head -> RMSNorm -> LM head logits
```

The target comparison topology is EP=8, LM-head TP=4, 128 global routed
experts, 16 local routed experts per EP rank, eight active tokens per rank,
and trace-hash routing.

The measured child must consume prepared tensors. Embedding lookup, accepted
hidden packing, and SWA metadata construction are fixture preparation and are
not part of the measured dispatch. Greedy sampling and persistent-signal
cleanup are also outside the measured dispatch.

This runner and the host-prepared decode-to-logits runner described in
[DeepSeek V4 Flash Decode-to-Logits Performance Design](2026-07-30-deepseek-v4-flash-decode-logits-design.md)
are independent benchmark intervals with different prepared inputs. Run them
as separate processes; do not add their timings or interpret them as an
end-to-end Decode-to-MTP measurement.

## AscendC Reference

The strict reference interval in `trace_view_a3_decode.json` starts at Model47
task 6, the first projection `RmsNorm`, and ends after Model47 task 66,
`aclnnInplaceCopy_CastAiCore_Cast`.

| Round | Projection-to-final-cast interval |
|---|---:|
| 1 | 2.287126 ms |
| 2 | 2.260706 ms |
| 3 | 2.160163 ms |
| Median | 2.260706 ms |
| Mean | 2.235998 ms |

The Model47 embedding-to-final-cast interval has a 2.271312 ms mean and is not
the selected boundary. The 52.731107 ms mean is the complete 43-layer decode,
handoff, and MTP interval and must not be compared with this core-only runner.

## Prepared Inputs

Each rank receives these values before the measured child is submitted:

- `hidden_states [T, D] BF16`, equivalent to the completed MTP embedding
  output;
- `prev_pre_hc_hidden [T, HC_MULT, D] FP32`, equivalent to completed accepted
  hidden packing;
- `swa_slot_mapping [T] INT64`;
- `swa_indices [T, WIN] INT32`;
- `swa_lens [T] INT32`;
- `position_ids [T] INT32` for RoPE;
- `routing_input_ids [T] INT64` for trace-hash routing;
- the existing projection, attention, MoE, HC-head, final-norm, and LM-head
  weights, KV cache, routing table, communication windows, and scalar values.

The standalone fixture constructs these tensors on the host through
`TensorSpec` initializers. No preparation child is submitted in the benchmark
program.

## Chosen Architecture

Add `models/deepseek/v4-flash/decode_mtp_core.py` with three entries:

1. `mtp_decode_core_logits` is the measured rank-local child. Its first model
   operation is `mtp_projection`; it then calls `attention_swa`, `moe`,
   `hc_head`, final `rms_norm`, and an LM-head body that ends at the logits
   gather.
2. `mtp_decode_core_cleanup` clears the MoE and LM-head signal windows. It is a
   separate rank-local child and reads the core outputs as completion anchors.
3. `l3_mtp_decode_core` allocates the existing distributed windows, submits
   the compute child for every rank, then submits the cleanup child for every
   rank.

`models/deepseek/v4-flash/lm_head.py` will expose:

- `lm_head_core`, containing the existing projection, communication, and final
  logits gather without signal clearing;
- `clear_lm_head_signals`, containing the current anchored clear scope;
- the existing `lm_head`, retained as a compatibility wrapper that calls both
  functions in the original order.

The existing `decode_mtp.py`, `decode_fwd_mtp.py`, standalone sampled MTP path,
and complete Scheme 4 program retain their behavior.

## Dispatch and Timing Contract

The L3 order is:

```text
submit mtp_decode_core_logits on every rank
submit mtp_decode_core_cleanup on every rank
```

The compute child is one top-level `MIX_AIC` program per rank. The cleanup
child is a second top-level program. Cross-framework system-trace measurements
select only the compute child, from its first task through its last logits
gather task. The cleanup child remains in every graph execution so fixed
communication epochs can be reused safely across warmup and measured rounds.

Measurements report NPU0 and the maximum interval across all eight ranks.
At least three steady system-trace samples are required. The regression run
uses five warmup rounds and 100 measured rounds with L2 swimlane disabled.

## Correctness Strategy

Contract tests enforce the callable boundary and dispatch order. A two-rank
exact-golden run may be used as the lower-cost numerical check; it preserves 16
local experts per rank but is not a performance comparison. The official
eight-rank EP8/TP4 run checks finite outputs before performance collection.

The target run must verify:

- EP=8, TP=4, 128 global experts, and 16 local experts per rank;
- 384 active routes with exactly three trace-hash routes per global expert;
- one compute child followed by one cleanup child per rank;
- no embedding, accepted-hidden packing, SWA metadata builder, sampler, MoE
  signal clear, or LM-head signal clear inside the compute child;
- the compute child's final task is the LM-head logits gather;
- repeated rounds complete without stale signal counters.

## Reporting

The Chinese comparison report keeps the complete 52.731107 ms Scheme 4 data as
context and adds a distinct "Scheme 4 MTP-core isolated interval" section. It
uses the 2.235998 ms AscendC mean and 2.260706 ms median for direct comparison.
It must not compare the new PyPTO core measurement with the complete AscendC
Decode-to-MTP interval.

## Non-Goals

This change does not:

- remove or alter the complete Decode-to-MTP benchmark;
- include Model47 embedding or input preparation in the timed child;
- include sampling or acceptance logic;
- optimize individual kernels or inspect in-core timing;
- change model dimensions, quantization, expert topology, or routing semantics;
- commit generated build output, device logs, or profiler traces.
