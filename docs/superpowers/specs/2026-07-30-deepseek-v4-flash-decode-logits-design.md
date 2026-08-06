# DeepSeek V4 Flash Decode-to-Logits Performance Design

## Objective

Retain a dedicated host-prepared decode-to-logits benchmark on
`perf/dsv4-eplb-decode-logits-and-mtp-core`. Its workload matches the original
`dsv4-ascendc-decode-compare2` reference, with one intentional difference in
the decode tail:

- Compare2 ends after the final layer's MoE output.
- The decode-to-logits benchmark continues through `hc_head`, final RMSNorm,
  and the logits-only LM-head, ending at FP32 logits before greedy sampling.

The historical compare3 result measured with 32 local experts and two active
tokens is invalid for a direct compare2/compare3 comparison and must be
replaced.

This runner and the strict MTP-core runner described in
[DeepSeek V4 Flash MTP Core Performance Design](2026-07-31-deepseek-v4-flash-mtp-core-design.md)
are independent benchmark intervals with different prepared inputs. Run them
as separate processes; do not add their timings or interpret them as an
end-to-end Decode-to-MTP measurement.

## Reference Workload

The decode-to-logits benchmark must match these compare2 settings:

| Setting | Required value |
|---|---:|
| EP world size | 8 |
| LM-head TP world size | 4 |
| Global routed experts | 128 |
| Routed experts per EP rank | 16 |
| Active tokens per rank | 8 |
| Static decode capacity | B=4, S=2, T=8 |
| Decode position | 8192 |
| Device IDs | 0,2,4,6,8,10,12,14 |
| Routing fixture | Balanced round-robin |
| Input preparation | Host-prepared |

The expert topology is derived from the Flash preset's 256 experts:

```text
N_EXPERTS_GLOBAL = 256 // 16 * EP = 128
N_LOCAL = N_EXPERTS_GLOBAL // EP = 16
```

## Implementation Design

### Expert topology

Keep `models/deepseek/v4-flash/moe.py` on compare2's
`n_routed_experts // 16 * EP` scaling. Update the module documentation to state
that every supported EP configuration retains 16 experts per rank.

Do not copy compare2's unrelated removal of the standalone `moe_test` signal
cleanup.

### Host-prepared decode inputs

Port compare2's host-prepared input boundary into
`models/deepseek/v4-flash/decode_fwd_logits.py`:

- Accept the prepared `x_hc` tensor instead of performing embedding and
  `pack_x_hc` inside the device graph.
- Accept slot mappings, sparse-attention indices and lengths, and compression
  mappings as host inputs instead of building decode metadata in the device
  graph.
- Keep all 43 attention and MoE layers unchanged after that boundary.

The module exposes `decode_fwd_logits` as the rank-local child and
`l3_decode_fwd_logits` as the distributed benchmark entry.

The first timed device task must therefore be the first layer's `hc_pre_rms`,
matching compare2 and the AscendC crop that begins at the first `HcPre`.

### Balanced routing fixture

Port compare2's deterministic routing workload:

- Use the same round-robin `tid2eid` mapping for every forward layer.
- Offset active `input_ids` by rank so the eight ranks consume one contiguous
  route sequence.
- Use the hash-layer-zero routing table consistently in every forward MoE
  invocation.

With EP8, T8, and top-k 6, this produces 384 active routes, or exactly three
routes per one of the 128 global experts in every layer.

### Decode-to-logits tail

Preserve the logits-only tail:

```text
final MoE output
  -> hc_head
  -> final RMSNorm
  -> logits-only LM-head
  -> FP32 logits
```

Do not expose `sampled_ids` and do not call greedy sampling or ArgMax. Preserve
the LM-head TP4 communication buffers and completion protocol.

### Benchmark invocation

Use this workload for formal measurements:

```bash
PYPTO_BENCH=1 \
PYPTO_BENCH_WARMUP=5 \
PYPTO_BENCH_ROUNDS=100 \
PYPTO_BENCH_RAW=1 \
python models/deepseek/v4-flash/decode_fwd_logits.py \
  -p a2a3 \
  --ep 8 \
  --tp 4 \
  -d 0,2,4,6,8,10,12,14 \
  --start-pos 8192 \
  --num-tokens 8 \
  --enable-l2-swimlane 0
```

Use a separate single-dispatch L2 swimlane run only to verify graph boundaries
and task structure. Do not use an instrumented L2 timing as the formal latency.

## Verification

### Contract tests

Use `tests/contract/test_deepseek_v4_flash_decode_fwd_logits.py` to import the
real model with `--ep 8 --tp 4` and verify:

- `N_RANKS == 8`
- `N_EXPERTS_GLOBAL == 128`
- `N_LOCAL == 16`
- `LM_HEAD_TP_SIZE == 4`
- the device and host decode signatures expose FP32 `logits`
- neither signature exposes `sampled_ids`
- the device graph accepts `x_hc` and the host-prepared mapping tensors
- the device graph does not accept `embed_weight` or `block_counts`

### Repository checks

Run the focused contract test, the golden test suite, repository pre-commit
hooks, Python compilation, and `git diff --check`.

### Device checks

Compile and run on the eight even device IDs. Inspect the generated graph and
L2 trace to verify:

- the first semantic task is `hc_pre_rms`
- all 43 layers execute
- `hc_head` and final RMSNorm follow the final MoE
- the final semantic output task is the LM-head logits gather
- no greedy-sampling task is present

Collect five warmups and 100 uninstrumented rounds. Report the per-round
slowest-rank effective latency, distribution statistics, and comparison with
the supplied AscendC trace.

## Acceptance Criteria

- The compare2 reference and decode-to-logits benchmark have identical EP,
  TP, expert topology, token count, routing fixture, input preparation,
  sequence position, and device mapping.
- Their only intended graph difference is the decode-to-logits
  `hc_head + RMSNorm + logits-only LM-head` tail.
- All contract and golden tests pass.
- The aligned device benchmark produces 100 parseable measured rounds without
  flattened fallback.
- The obsolete 32-expert/two-token compare3 result is not reported as the
  aligned result.

## Non-Goals

- Changing compare2.
- Adding greedy sampling to the decode-to-logits benchmark.
- Modifying attention, MoE, hc-head, RMSNorm, or LM-head kernel algorithms.
- Porting compare2's daily-CI automation or unrelated standalone MoE cleanup.
- Rewriting the historical compare3 commits onto compare2.
